"""Post-recognition region repair for dense newspaper pages.

:mod:`newspaper_ocr.layout_processor` runs *before* OCR, so it only ever sees
geometry.  That is enough for near-identical boxes, but not for the defect that
dominates dense 6-9 column broadsheets: the detector emits both a tall
full-column region *and* the individual paragraph regions inside it.  Those two
detections have different shapes — tall-narrow against wide-short — so IoU and
containment dedup leave both standing, and OCR then reads the same passage
twice, at two different qualities.  The duplicate shows up in the page JSON, in
full-text search, and in anything that cites those region ids.

The repair stage runs *after* recognition, where the text is available to prove
what is redundant:

1. :meth:`RegionRepair.dedupe_text` — drop a region only when it is *provably*
   redundant: same text in the same place, or text that is a strict substring of
   an overlapping region's text.  Never on fuzzy or token-overlap similarity: a
   column read and its paragraph reads each carry OCR-variant tokens the other
   lacks, so fuzzy dropping loses unique text.
2. :meth:`RegionRepair.split_containers` — when a region's height is mostly
   covered by smaller regions sharing its column, it is a duplicate container
   read.  Rather than drop it (which would lose whatever the paragraph reads
   missed *between* them), keep the vertical strips nothing else covers, re-OCR
   those, and drop the container.  Covered text survives through the cleaner
   inner reads; uncovered text survives as new strip regions.
3. :meth:`RegionRepair.merge_fragments` — union clusters of overlapping
   fragments of one display ad and re-OCR the union once, so the ad reads as
   prose instead of shards.  Guarded against swallowing a neighbour that is not
   part of the cluster, which would duplicate *its* text into the merge.
4. :func:`find_duplicate_pages` — vendor PDFs often carry the same physical page
   twice as two different scans (different crop, contrast and md5, so hashing
   misses them).  Compared on recognized text, they are obvious.

Every pass is non-destructive: :meth:`RegionRepair.repair` snapshots the
recognized regions into :attr:`~newspaper_ocr.models.PageLayout.raw_regions` and
always recomputes from that snapshot, so repair is idempotent, re-runnable with
different thresholds, and cannot destroy the raw OCR layer.  Repair that edits
in place has exactly one failure mode, and it is the expensive one: re-OCR is
the only way back.

Thresholds below are the ones validated end-to-end on *The Negro World*
(1921-1933); see the class docstring for what each one trades off.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, replace
from typing import Callable, Sequence

from PIL import Image

from newspaper_ocr.models import TIMEOUT_TEXT, BBox, PageLayout, Region

logger = logging.getLogger(__name__)

#: Re-OCR callback: takes a page crop, returns ``(text, status)`` — the shape
#: :func:`newspaper_ocr.recognizers.base.recognize_crop` guarantees.  Repair
#: needs nothing else from a recognizer, so any callable of this shape (a remote
#: service, a cached lookup, a test double) can drive the re-OCR passes.
RecognizeCrop = Callable[[Image.Image], "tuple[str, str]"]

#: Labels a container candidate may carry.  Restricting to text keeps the split
#: pass away from figures and tables, where the inner regions are captions and
#: cells rather than a second reading of the same prose.
CONTAINER_LABELS = frozenset(
    {"text", "paragraph_title", "doc_title", "figure_title", "abstract"}
)

#: Statuses whose text is real enough to trust in place of a container's read.
_USABLE_STATUSES = frozenset({"ok", "repetition"})

#: Text that is a marker, not content.  Two regions that both timed out carry
#: identical text without being duplicates of each other.
_PLACEHOLDER_TEXTS = frozenset({TIMEOUT_TEXT.casefold()})

#: Above this ``SequenceMatcher.ratio()`` two pages are the same physical page.
#: Measured across 60 clean issues: true duplicate scans land at 0.47-0.96,
#: distinct newspaper pages at or below 0.06, and nothing legitimate falls in
#: between — so the gap, not the exact value, is doing the work.
DUPLICATE_PAGE_THRESHOLD = 0.35

_WS = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Geometry / text helpers
# ---------------------------------------------------------------------------

def _tup(r: Region) -> tuple[int, int, int, int]:
    return (r.bbox.x0, r.bbox.y0, r.bbox.x1, r.bbox.y1)


def _area(b: tuple[int, int, int, int]) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _inter(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    union = _area(a) + _area(b) - _inter(a, b)
    return _inter(a, b) / union if union > 0 else 0.0


def _inside_frac(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> float:
    """Fraction of *inner*'s area that falls within *outer*."""
    a = _area(inner)
    return _inter(inner, outer) / a if a > 0 else 0.0


def _x_inside_frac(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> float:
    """Fraction of *inner*'s width that falls within *outer*'s column."""
    w = inner[2] - inner[0]
    if w <= 0:
        return 0.0
    return max(0, min(inner[2], outer[2]) - max(inner[0], outer[0])) / w


def _norm(text: str) -> str:
    """Whitespace- and case-normalized text, or ``""`` when it isn't content.

    Two reads of one passage differ in line breaks and capitalization far more
    often than in letters, so comparing raw text would miss duplicates that a
    human would call identical.  Placeholders are normalized *out*: they mark
    the absence of a read, and treating them as text makes every timed-out
    region a duplicate of every other one.
    """
    if not text:
        return ""
    normalized = _WS.sub(" ", text).strip().casefold()
    if normalized in _PLACEHOLDER_TEXTS:
        return ""
    return normalized


_STATUS_RANK = {"ok": 3, "repetition": 2, "timeout": 1, "error": 0}


def _quality(r: Region) -> tuple[int, int, float]:
    """Sort key for "which read of the same text do we keep": status, then length."""
    return (_STATUS_RANK.get(r.status, 0), len(r.text or ""), r.confidence)


def _copy(r: Region) -> Region:
    """Shallow copy that shares images but not the mutable containers.

    Repair hands out its own region objects so that mutating a repaired layout
    can never reach back into ``raw_regions``.
    """
    return replace(r, lines=list(r.lines))


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union of closed-open ``(start, end)`` intervals, sorted and non-overlapping."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _gaps(span: tuple[int, int], covered: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The parts of *span* that *covered* (already merged) does not cover."""
    out: list[tuple[int, int]] = []
    cursor = span[0]
    for start, end in covered:
        if start > cursor:
            out.append((cursor, min(start, span[1])))
        cursor = max(cursor, end)
        if cursor >= span[1]:
            break
    if cursor < span[1]:
        out.append((cursor, span[1]))
    return [(s, e) for s, e in out if e > s]


# ---------------------------------------------------------------------------
# RegionRepair
# ---------------------------------------------------------------------------

@dataclass
class RepairReport:
    """What one :meth:`RegionRepair.repair` call actually changed."""

    deduped: int = 0
    containers_split: int = 0
    containers_dropped: int = 0
    strips_added: int = 0
    clusters_merged: int = 0
    regions_before: int = 0
    regions_after: int = 0

    @property
    def changed(self) -> bool:
        return bool(
            self.deduped
            or self.containers_split
            or self.containers_dropped
            or self.clusters_merged
        )


class RegionRepair:
    """Text-aware repair of recognized regions.

    Parameters
    ----------
    enabled:
        When False, :meth:`repair` is a pass-through.
    duplicate_iou:
        Two regions with identical text count as the same detection only when
        their boxes overlap at least this much.  Keeps a standing head
        ("PAGE TWO") that legitimately appears twice on a page from
        deduplicating itself away.
    substring_overlap:
        A region whose text is a strict substring of another's is dropped only
        when at least this fraction of its area sits inside that other region —
        the text has to be redundant *in the same place*, not merely repeated
        elsewhere on the page.
    min_substring_chars:
        Substring dedup ignores texts shorter than this.  Short strings ("THE
        END", a date line) turn up inside longer ones by coincidence.
    container_coverage / min_inner_regions:
        A region is a duplicate container read when at least *min_inner_regions*
        smaller regions sharing its column cover at least *container_coverage*
        of its height.  Only inner regions that carry usable text count: empty
        detections cover geometry, not words, and letting them vote would drop a
        container whose text nothing else holds.
    column_overlap:
        How much of a candidate inner region's width must fall inside the
        container's column for it to count as "inside" it.
    min_strip_height:
        Uncovered strips shorter than this are dropped rather than re-OCR'd —
        below roughly a line's height there is nothing to read, and re-OCR of a
        sliver costs a model call to produce noise.
    merge_iou:
        Pairwise IoU at which two fragments are taken to belong to one display
        ad.
    swallow_frac / max_union_page_frac:
        The anti-swallow guards.  A merge is abandoned when its union bbox would
        engulf *swallow_frac* of a region that is not in the cluster (re-OCR
        would duplicate that neighbour's text into the merge) or when the union
        covers more than *max_union_page_frac* of the page (a "cluster" that
        size is a detector failure, not an ad).
    nesting_frac:
        A region this fully inside another is nested rather than a shard of the
        same ad, so the two never join a merge cluster.  It has to sit well
        above ``swallow_frac``: two shards of one ad offset diagonally already
        put two thirds of each inside the other without either containing
        anything.
    merge_labels:
        Restrict fragment merging to these labels.  ``None`` (the default)
        considers every label, which is what dense mixed pages need; set it to
        e.g. ``{"image", "figure"}`` on a corpus where ads are labeled reliably.
    """

    def __init__(
        self,
        enabled: bool = True,
        duplicate_iou: float = 0.6,
        substring_overlap: float = 0.5,
        min_substring_chars: int = 16,
        container_coverage: float = 0.6,
        min_inner_regions: int = 2,
        column_overlap: float = 0.5,
        min_strip_height: int = 40,
        merge_iou: float = 0.3,
        swallow_frac: float = 0.6,
        max_union_page_frac: float = 0.6,
        nesting_frac: float = 0.9,
        merge_labels: frozenset[str] | set[str] | None = None,
        container_labels: frozenset[str] | set[str] = CONTAINER_LABELS,
    ) -> None:
        self.enabled = enabled
        self.duplicate_iou = duplicate_iou
        self.substring_overlap = substring_overlap
        self.min_substring_chars = min_substring_chars
        self.container_coverage = container_coverage
        self.min_inner_regions = min_inner_regions
        self.column_overlap = column_overlap
        self.min_strip_height = min_strip_height
        self.merge_iou = merge_iou
        self.swallow_frac = swallow_frac
        self.max_union_page_frac = max_union_page_frac
        self.nesting_frac = nesting_frac
        self.merge_labels = frozenset(merge_labels) if merge_labels else None
        self.container_labels = frozenset(container_labels)
        #: Stats from the most recent :meth:`repair` call.
        self.last_report = RepairReport()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def repair(
        self,
        layout: PageLayout,
        recognize: RecognizeCrop | None = None,
    ) -> PageLayout:
        """Repair *layout*'s regions in reading order, non-destructively.

        The first call snapshots the recognized regions into
        ``layout.raw_regions``; every call — the first and each later one —
        recomputes ``layout.regions`` from that snapshot.  Running repair twice,
        or re-running it with different thresholds, therefore gives the same
        answer as running it once with the final thresholds, and nothing a
        repair pass does can reach the raw layer.

        *recognize* re-reads a crop of the page, returning ``(text, status)``;
        :func:`newspaper_ocr.recognizers.base.recognize_crop` adapts any
        region-capable recognizer to it.  Without it the passes that need fresh
        text are skipped rather than approximated: a container with uncovered
        strips is kept whole, and fragments are left unmerged.  Both are the
        lossless choice — the wrong one is stitching together text nobody read.

        The report is returned via :attr:`last_report` for callers that want to
        log what changed; the layout itself is the return value so the call
        chains like the other stages.
        """
        report = RepairReport(regions_before=len(layout.regions))
        self.last_report = report

        if not self.enabled:
            report.regions_after = len(layout.regions)
            return layout

        if layout.raw_regions is None:
            layout.raw_regions = list(layout.regions)

        regions = [_copy(r) for r in layout.raw_regions]
        regions = self.dedupe_text(regions, report)
        regions = self.split_containers(regions, layout, recognize, report)
        regions = self.merge_fragments(regions, layout, recognize, report)

        layout.regions = regions
        report.regions_after = len(regions)
        if report.changed:
            logger.info(
                "region repair: %d -> %d regions (deduped %d, containers split %d, "
                "dropped %d, strips %d, merges %d)",
                report.regions_before,
                report.regions_after,
                report.deduped,
                report.containers_split,
                report.containers_dropped,
                report.strips_added,
                report.clusters_merged,
            )
        return layout

    # ------------------------------------------------------------------
    # Pass 1 – lossless text dedup
    # ------------------------------------------------------------------

    def dedupe_text(
        self, regions: list[Region], report: RepairReport | None = None
    ) -> list[Region]:
        """Drop regions whose text another region provably already carries.

        Two cases, both lossless:

        * **Same text, same place** — identical normalized text and boxes
          overlapping by ``duplicate_iou``.  The better read stays: higher
          status first, then longer text, then higher detection confidence.
        * **Strict substring** — one region's text appears whole inside an
          overlapping region's text.  The contained read adds no words, so it
          goes.

        What is deliberately *not* here is fuzzy matching.  A column read and
        its paragraph reads overlap heavily in tokens while each holds
        OCR-variants the other lacks; dropping on similarity would throw away
        text no other region has.  Everything short of provable redundancy is
        left for :meth:`split_containers`, which resolves the same duplication
        without deleting anything.
        """
        report = report if report is not None else RepairReport()
        if len(regions) < 2:
            return regions

        texts = [_norm(r.text) for r in regions]
        boxes = [_tup(r) for r in regions]
        drop: set[int] = set()

        for i in range(len(regions)):
            if i in drop or not texts[i]:
                continue
            for j in range(i + 1, len(regions)):
                if j in drop or not texts[j]:
                    continue

                if texts[i] == texts[j]:
                    if _iou(boxes[i], boxes[j]) < self.duplicate_iou:
                        continue
                    loser = j if _quality(regions[i]) >= _quality(regions[j]) else i
                elif texts[i] in texts[j]:
                    loser = self._substring_loser(i, j, texts, boxes)
                elif texts[j] in texts[i]:
                    loser = self._substring_loser(j, i, texts, boxes)
                else:
                    continue

                if loser is None:
                    continue
                drop.add(loser)
                if loser == i:
                    break

        report.deduped += len(drop)
        return [r for k, r in enumerate(regions) if k not in drop]

    def _substring_loser(
        self,
        inner: int,
        outer: int,
        texts: list[str],
        boxes: list[tuple[int, int, int, int]],
    ) -> int | None:
        """Index to drop for ``texts[inner] in texts[outer]``, or None to keep both."""
        if len(texts[inner]) < self.min_substring_chars:
            return None
        if _inside_frac(boxes[inner], boxes[outer]) < self.substring_overlap:
            return None
        return inner

    # ------------------------------------------------------------------
    # Pass 2 – container split
    # ------------------------------------------------------------------

    def split_containers(
        self,
        regions: list[Region],
        layout: PageLayout,
        recognize: RecognizeCrop | None = None,
        report: RepairReport | None = None,
    ) -> list[Region]:
        """Replace duplicate container reads with their uncovered strips.

        A container is a region whose height is mostly accounted for by smaller
        regions in the same column that carry usable text.  Dropping it outright
        would lose whatever those inner reads missed in the gaps between them,
        so instead the gaps become their own regions, re-OCR'd from the page
        image, and the container goes.

        Fully covered containers need no re-OCR and are simply dropped.  A
        container with gaps is kept intact when there is no *recognize*
        callback, or when re-OCR of a strip fails outright — a dropped container
        whose strips came back empty is exactly the data loss this stage exists
        to prevent.
        """
        report = report if report is not None else RepairReport()
        if len(regions) < self.min_inner_regions + 1:
            return regions

        used_ids = {r.id for r in regions if r.id}
        out: list[Region] = []

        for idx, region in enumerate(regions):
            inners = self._inner_regions(idx, regions)
            if len(inners) < self.min_inner_regions:
                out.append(region)
                continue

            box = _tup(region)
            span = (box[1], box[3])
            height = span[1] - span[0]
            if height <= 0:
                out.append(region)
                continue

            covered = _merge_intervals(
                [
                    (max(span[0], _tup(r)[1]), min(span[1], _tup(r)[3]))
                    for r in inners
                ]
            )
            coverage = sum(e - s for s, e in covered) / height
            if coverage < self.container_coverage:
                out.append(region)
                continue

            gaps = [
                (s, e)
                for s, e in _gaps(span, covered)
                if e - s >= self.min_strip_height
            ]

            if not gaps:
                # Fully covered: the inner reads hold every line the container
                # does, so dropping it costs nothing and needs no model call.
                report.containers_dropped += 1
                continue

            if recognize is None:
                out.append(region)
                continue

            strips = self._recognize_strips(region, gaps, layout, used_ids, recognize)
            if strips is None:
                # Re-OCR failed on at least one strip — keep the container.
                out.append(region)
                continue

            out.extend(strips)
            report.containers_split += 1
            report.strips_added += len(strips)

        return out

    def _inner_regions(self, idx: int, regions: list[Region]) -> list[Region]:
        """Smaller, text-carrying regions sharing *regions[idx]*'s column."""
        container = regions[idx]
        if container.label not in self.container_labels:
            return []
        box = _tup(container)
        container_area = _area(box)
        if container_area <= 0:
            return []

        inners: list[Region] = []
        for k, other in enumerate(regions):
            if k == idx:
                continue
            if other.status not in _USABLE_STATUSES or not _norm(other.text):
                continue
            ob = _tup(other)
            if _area(ob) >= container_area:
                continue
            if _x_inside_frac(ob, box) < self.column_overlap:
                continue
            if min(ob[3], box[3]) - max(ob[1], box[1]) <= 0:
                continue
            inners.append(other)
        return inners

    def _recognize_strips(
        self,
        container: Region,
        gaps: list[tuple[int, int]],
        layout: PageLayout,
        used_ids: set[str],
        recognize: RecognizeCrop,
    ) -> list[Region] | None:
        """Re-OCR each uncovered strip, or None if any strip could not be read."""
        box = _tup(container)
        strips: list[Region] = []
        for n, (y0, y1) in enumerate(gaps):
            crop = layout.image.crop((box[0], y0, box[2], y1))
            text, status = recognize(crop)
            if status not in _USABLE_STATUSES:
                logger.debug(
                    "container %s: strip %d re-OCR failed (%s); keeping container",
                    container.id or "?",
                    n,
                    status,
                )
                return None
            if not text.strip():
                # A blank gap — rules, whitespace, the tail of a column. Nothing
                # was lost, so there is nothing to carry forward.
                continue
            strips.append(
                Region(
                    bbox=BBox(box[0], y0, box[2], y1),
                    image=crop,
                    label=container.label,
                    text=text,
                    confidence=container.confidence,
                    status=status,
                    id=_derive_id(container.id, f"s{n}", used_ids),
                )
            )
        return strips

    # ------------------------------------------------------------------
    # Pass 3 – fragmented-ad merge
    # ------------------------------------------------------------------

    def merge_fragments(
        self,
        regions: list[Region],
        layout: PageLayout,
        recognize: RecognizeCrop | None = None,
        report: RepairReport | None = None,
    ) -> list[Region]:
        """Union overlapping fragments of one display ad and re-OCR them once.

        Display ads come out of the detector as shards — a border piece, a
        slogan, a price — each read on its own into something that is not
        prose.  Reading the union once gives coherent text.

        Two guards keep a merge from *creating* duplicate text, and both are
        load-bearing.  A union bbox that engulfs a region outside the cluster
        re-reads that neighbour's words into the merge while the neighbour keeps
        them too, so such a merge is abandoned; so is one whose union spans most
        of the page, which is a detector failure rather than an ad.  Without a
        *recognize* callback there is no coherent text to put in the merged
        region, so the fragments are left alone.
        """
        report = report if report is not None else RepairReport()
        if recognize is None or len(regions) < 2:
            return regions

        clusters = self._cluster_fragments(regions)
        if not clusters:
            return regions

        page_area = _area((0, 0, layout.width, layout.height)) or _area(
            (0, 0, layout.image.width, layout.image.height)
        )
        used_ids = {r.id for r in regions if r.id}
        merged_at: dict[int, Region] = {}
        absorbed: set[int] = set()

        for cluster in clusters:
            union = _union_box([_tup(regions[k]) for k in cluster])
            if page_area and _area(union) > page_area * self.max_union_page_frac:
                continue
            if self._swallows_outsider(union, cluster, regions):
                continue

            crop = layout.image.crop(union)
            text, status = recognize(crop)
            if status not in _USABLE_STATUSES or not text.strip():
                # Nothing usable came back; the fragments' own text is all
                # there is, so keep it.
                continue

            head = min(cluster)
            biggest = max(cluster, key=lambda k: _area(_tup(regions[k])))
            merged_at[head] = Region(
                bbox=BBox(*union),
                image=crop,
                label=regions[biggest].label,
                text=text,
                confidence=max(regions[k].confidence for k in cluster),
                status=status,
                id=_derive_id(regions[head].id, "m", used_ids),
            )
            absorbed.update(cluster)
            report.clusters_merged += 1

        if not merged_at:
            return regions

        out: list[Region] = []
        for k, region in enumerate(regions):
            if k in merged_at:
                out.append(merged_at[k])
            elif k not in absorbed:
                out.append(region)
        return out

    def _cluster_fragments(self, regions: list[Region]) -> list[list[int]]:
        """Group region indices connected by pairwise IoU >= ``merge_iou``.

        Nested pairs are not fragments and never join a cluster: when one region
        sits almost entirely inside the other they are a container and its
        contents — a column and its paragraphs — which :meth:`split_containers`
        resolves by keeping the finer reads.  Merging them instead would re-OCR
        the whole column and throw those reads away, which is the duplicate this
        module exists to remove, arrived at from the other direction.
        """
        eligible = [
            k
            for k, r in enumerate(regions)
            if self.merge_labels is None or r.label in self.merge_labels
        ]
        if len(eligible) < 2:
            return []

        parent = {k: k for k in eligible}

        def find(k: int) -> int:
            while parent[k] != k:
                parent[k] = parent[parent[k]]
                k = parent[k]
            return k

        joined = False
        for a_i, a in enumerate(eligible):
            for b in eligible[a_i + 1 :]:
                if self._is_fragment_pair(regions[a], regions[b]):
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra
                        joined = True
        if not joined:
            return []

        groups: dict[int, list[int]] = {}
        for k in eligible:
            groups.setdefault(find(k), []).append(k)
        return [sorted(g) for g in groups.values() if len(g) > 1]

    def _is_fragment_pair(self, a: Region, b: Region) -> bool:
        """True when two regions look like shards of one ad rather than nesting."""
        ba, bb = _tup(a), _tup(b)
        if _iou(ba, bb) < self.merge_iou:
            return False
        inner, outer = (ba, bb) if _area(ba) <= _area(bb) else (bb, ba)
        return _inside_frac(inner, outer) < self.nesting_frac

    def _swallows_outsider(
        self,
        union: tuple[int, int, int, int],
        cluster: Sequence[int],
        regions: list[Region],
    ) -> bool:
        """True when *union* engulfs a region that is not part of *cluster*."""
        members = set(cluster)
        for k, region in enumerate(regions):
            if k in members:
                continue
            if _inside_frac(_tup(region), union) >= self.swallow_frac:
                return True
        return False


def _union_box(boxes: Sequence[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _derive_id(base: str, suffix: str, used: set[str]) -> str:
    """A new id derived from *base*, recorded in *used* so it stays unique.

    Repair invents regions the pipeline never numbered, and downstream consumers
    key on region ids.  Deriving ``r7s0`` from the container ``r7`` keeps the
    provenance readable instead of renumbering the page and invalidating every
    reference that already points into it.
    """
    if not base:
        return ""
    candidate = f"{base}{suffix}"
    n = 1
    while candidate in used:
        candidate = f"{base}{suffix}_{n}"
        n += 1
    used.add(candidate)
    return candidate


# ---------------------------------------------------------------------------
# Pass 4 – page-level duplicate-scan detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DuplicatePage:
    """One page that repeats an earlier one, and how strongly."""

    index: int
    duplicate_of: int
    similarity: float


def page_text(page: PageLayout | str) -> str:
    """The comparable text of a page, from either a layout or a plain string."""
    return page if isinstance(page, str) else page.text


def page_similarity(a: str, b: str) -> float:
    """Similarity of two page texts, 0.0-1.0, for duplicate-scan detection.

    Uses :meth:`difflib.SequenceMatcher.ratio` — the real one.  ``quick_ratio``
    is tempting and wrong here: as a frequency-based upper bound it scores *all*
    newspaper pages 0.7-0.95, because any two pages of English prose use the
    same letters in roughly the same proportions.  It is useful only as a
    pre-filter (a value below the threshold rules out a match), which
    :func:`find_duplicate_pages` uses it for; the decision has to come from
    ``ratio()``, which compares order as well as inventory.
    """
    return _matcher(_norm(a), _norm(b)).ratio()


def _matcher(a: str, b: str) -> difflib.SequenceMatcher:
    matcher = difflib.SequenceMatcher()
    matcher.set_seqs(a, b)
    return matcher


def find_duplicate_pages(
    pages: Sequence[PageLayout | str],
    threshold: float = DUPLICATE_PAGE_THRESHOLD,
    min_chars: int = 200,
) -> list[DuplicatePage]:
    """Find pages that are re-scans of an earlier page in the same sequence.

    Microfilm and vendor PDFs (NewsBank and friends) routinely carry the same
    physical page twice: a "+2 offset" rescan, a front page shot three times, a
    whole second section reprinted — in one 159-issue corpus, 19 issues (12%)
    carried 34 such pages.  The scans differ in crop, contrast and therefore
    md5, so byte or hash dedup never sees them; their *text* gives them away.

    Each page is reported against the earliest page it matches, so a page
    scanned three times yields two findings pointing at the first copy rather
    than a chain.  Pages with fewer than *min_chars* characters are skipped:
    two nearly-blank pages match each other perfectly and mean nothing.

    Returns findings in page order; nothing is modified, since which copy of a
    duplicated page to keep is a decision for the caller (the later scan is
    often the better one).
    """
    texts = [_norm(page_text(p)) for p in pages]
    found: list[DuplicatePage] = []

    for j in range(1, len(texts)):
        if len(texts[j]) < min_chars:
            continue
        matcher = difflib.SequenceMatcher()
        matcher.set_seq2(texts[j])
        for i in range(j):
            if len(texts[i]) < min_chars:
                continue
            matcher.set_seq1(texts[i])
            # Both are upper bounds on ratio(), cheapest first: a page that
            # can't reach the threshold never pays for the real comparison.
            if matcher.real_quick_ratio() < threshold:
                continue
            if matcher.quick_ratio() < threshold:
                continue
            score = matcher.ratio()
            if score >= threshold:
                found.append(DuplicatePage(index=j, duplicate_of=i, similarity=score))
                break

    return found
