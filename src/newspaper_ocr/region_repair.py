"""Post-recognition region repair for dense multi-column newspaper pages.

`LayoutProcessor` cleans the layout *before* OCR (geometric dedup, gap fill,
reading order). Some defects only become visible once you have the recognized
*text*, and this stage fixes those. It runs AFTER recognition and is
**non-destructive**: it returns a new :class:`PageLayout`; the input is untouched.

Passes
------
1. **Lossless dedup** — drop a region only when provably redundant: near-identical
   box (IoU >= ``iou_dup``, keep the higher-``status``/longer read) or strict
   substring containment (inner text is a substring of the outer). No fuzzy/token
   dropping: a full-column read and its paragraph reads each carry OCR-variant
   tokens the other lacks, so fuzzy dropping would lose unique text.
2. **Container split** — PP-DocLayout often emits BOTH a tall full-column read AND
   the paragraph reads inside it (different shapes, so geometric dedup misses
   them; after OCR the text is duplicated). When a tall region shares a column
   with >= 2 smaller regions covering >= ``split_cover`` of its height, split it:
   keep only the vertical strips NOT covered by the inner regions (re-OCR each via
   ``recognizer``) and drop the container. Covered text survives via the cleaner
   inner regions; uncovered text (paragraphs the inner reads missed in their gaps)
   survives as new strip regions -> lossless AND de-duplicated. Requires a
   recognizer; without one the container is kept as-is.
3. **Fragmented-ad merge** — cluster overlapping fragments (pairwise IoU >=
   ``iou_merge``) that are partial detections of one display ad, union them, and
   re-OCR the union once for coherent text. Guarded: skip a merge whose union
   would engulf a non-member region (>= ``swallow_cover`` of it inside the union)
   — otherwise the re-OCR duplicates that sibling's text — or cover more than
   ``page_frac_guard`` of the page.
"""
from __future__ import annotations

import re
from newspaper_ocr.models import BBox, PageLayout, Region
from newspaper_ocr.recognizers.base import RegionRecognizer


def _area(b: BBox) -> int:
    return max(0, b.width) * max(0, b.height)


def _inter(a: BBox, b: BBox) -> int:
    ix = min(a.x1, b.x1) - max(a.x0, b.x0)
    iy = min(a.y1, b.y1) - max(a.y0, b.y0)
    return max(0, ix) * max(0, iy)


def _iou(a: BBox, b: BBox) -> float:
    i = _inter(a, b)
    u = _area(a) + _area(b) - i
    return i / u if u else 0.0


def _cover(a: BBox, b: BBox) -> float:
    """Fraction of box ``a`` that lies inside box ``b``."""
    return _inter(a, b) / _area(a) if _area(a) else 0.0


def _union(boxes: list[BBox]) -> BBox:
    return BBox(min(b.x0 for b in boxes), min(b.y0 for b in boxes),
               max(b.x1 for b in boxes), max(b.y1 for b in boxes))


def _norm(s: str) -> str:
    return re.sub(r"\W+", "", (s or "").lower())


def _toks(s: str) -> set[str]:
    return set(re.findall(r"[a-z]{4,}", (s or "").lower()))


class RegionRepair:
    def __init__(
        self,
        recognizer: RegionRecognizer | None = None,
        *,
        iou_dup: float = 0.6,
        iou_merge: float = 0.3,
        cover_in: float = 0.85,
        page_frac_guard: float = 0.6,
        swallow_cover: float = 0.6,
        split_cover: float = 0.6,
        min_strip: int = 60,
        lossless: bool = True,
    ) -> None:
        self.recognizer = recognizer
        self.lossless = lossless
        self.iou_dup = iou_dup
        self.iou_merge = iou_merge
        self.cover_in = cover_in
        self.page_frac_guard = page_frac_guard
        self.swallow_cover = swallow_cover
        self.split_cover = split_cover
        self.min_strip = min_strip
        self.actions: list[tuple] = []      # (kind, ...) log of what was done

    # -- re-OCR one crop through the recognizer, returning (text, status) --------
    def _ocr(self, page_img, box: BBox, label: str) -> tuple[str, str]:
        if self.recognizer is None:
            return "", "error"
        crop = page_img.crop(box.to_tuple())
        r = self.recognizer.recognize(Region(bbox=box, image=crop, label=label))
        return (r.text or "").strip(), r.status

    # -- public entry -----------------------------------------------------------
    def repair(self, layout: PageLayout) -> PageLayout:
        self.actions = []
        regions = list(layout.regions)
        regions = self._dedup(regions)
        regions = self._split_containers(regions, layout.image)
        regions = self._merge_fragments(regions, layout.image,
                                        layout.width, layout.height)
        for i, r in enumerate(regions):
            r.id = f"r{i}"
        return PageLayout(image=layout.image, regions=regions,
                          width=layout.width, height=layout.height,
                          lines_detected=layout.lines_detected)

    # -- pass 1: lossless dedup -------------------------------------------------
    def _dedup(self, regions: list[Region]) -> list[Region]:
        n = len(regions)
        drop = [False] * n
        for i in range(n):
            if drop[i]:
                continue
            for j in range(i + 1, n):
                if drop[j]:
                    continue
                bi, bj = regions[i].bbox, regions[j].bbox
                if _iou(bi, bj) >= self.iou_dup:
                    a, b = regions[i], regions[j]
                    keep_i = (a.status == "ok", len(a.text or ""), _area(bi)) >= \
                             (b.status == "ok", len(b.text or ""), _area(bj))
                    v = j if keep_i else i
                    u = i if keep_i else j
                    if self.lossless and _toks(regions[v].text) - _toks(regions[u].text):
                        regions[u].text = ((regions[u].text or "") + "\n"
                                           + (regions[v].text or "")).strip()
                    drop[v] = True
                    self.actions.append(("dup", v))
                    if v == i:
                        break
                else:
                    ci, cj = _cover(bi, bj), _cover(bj, bi)
                    if ci >= self.cover_in or cj >= self.cover_in:
                        inner, outer = (i, j) if ci >= cj else (j, i)
                        ti, to = _norm(regions[inner].text), _norm(regions[outer].text)
                        if ti and ti in to:               # strict substring -> lossless
                            drop[inner] = True
                            self.actions.append(("contained", inner))
                            if inner == i:
                                break
        return [r for k, r in enumerate(regions) if not drop[k]]

    # -- pass 2: split duplicated container reads -------------------------------
    def _split_containers(self, regions: list[Region], page_img) -> list[Region]:
        if self.recognizer is None:
            return regions
        keep = [True] * len(regions)
        added: list[Region] = []
        order = sorted(range(len(regions)), key=lambda i: -_area(regions[i].bbox))
        for a in order:
            if not keep[a]:
                continue
            A = regions[a].bbox
            if A.height < 400:
                continue
            intervals, inner_idx = [], []
            for b in range(len(regions)):
                if b == a or not keep[b]:
                    continue
                B = regions[b].bbox
                xov = min(A.x1, B.x1) - max(A.x0, B.x0)
                if xov > 0.5 * min(A.width, B.width) and _area(B) < _area(A):
                    y1, y2 = max(A.y0, B.y0), min(A.y1, B.y1)
                    if y2 > y1:
                        intervals.append((y1, y2))
                        inner_idx.append(b)
            if len(intervals) < 2:
                continue
            merged: list[list[int]] = []
            for y1, y2 in sorted(intervals):
                if merged and y1 <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], y2)
                else:
                    merged.append([y1, y2])
            covered = sum(y2 - y1 for y1, y2 in merged)
            if covered < self.split_cover * A.height:
                continue
            strips, cur = [], A.y0
            for y1, y2 in merged:
                if y1 - cur > self.min_strip:
                    strips.append((cur, y1))
                cur = max(cur, y2)
            if A.y1 - cur > self.min_strip:
                strips.append((cur, A.y1))
            recovered = []
            for y1, y2 in strips:
                bb = BBox(A.x0, y1, A.x1, y2)
                text, status = self._ocr(page_img, bb, regions[a].label)
                if text:
                    recovered.append(Region(bbox=bb, image=page_img.crop(bb.to_tuple()),
                                            label=regions[a].label, text=text, status=status))
            # Lossless guard: only drop the container if every token it holds is
            # reproduced by the inner regions + the recovered strips. Otherwise its
            # OCR caught words the paragraph reads missed/varied on -> keep it.
            if self.lossless:
                kept_tok = set().union(*[_toks(regions[b].text) for b in inner_idx]) if inner_idx else set()
                kept_tok |= set().union(*[_toks(s.text) for s in recovered]) if recovered else set()
                if _toks(regions[a].text) - kept_tok:
                    self.actions.append(("split_kept", a))   # unique tokens -> don't drop
                    continue
            keep[a] = False                    # covered text lives in the inner regions
            added.extend(recovered)            # uncovered text preserved as strips
            self.actions.append(("split", a, len(recovered)))
        return [r for k, r in enumerate(regions) if keep[k]] + added

    # -- pass 3: merge fragmented ads (anti-swallow) ----------------------------
    def _clusters(self, regions: list[Region]) -> list[list[int]]:
        n = len(regions)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(n):
            for j in range(i + 1, n):
                if _iou(regions[i].bbox, regions[j].bbox) >= self.iou_merge:
                    parent[find(i)] = find(j)
        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        return [g for g in groups.values() if len(g) >= 2]

    def _merge_fragments(self, regions, page_img, page_w, page_h) -> list[Region]:
        clusters = self._clusters(regions)
        out: list[Region] = []
        done: set[int] = set()
        page_area = page_w * page_h
        for idx in range(len(regions)):
            c = next((cl for cl in clusters if idx in cl), None)
            if c is None:
                out.append(regions[idx])
                continue
            if c[0] in done:
                continue
            done.add(c[0])
            ub = _union([regions[i].bbox for i in c])
            members = set(c)
            if _area(ub) > self.page_frac_guard * page_area or any(
                k not in members and _cover(regions[k].bbox, ub) >= self.swallow_cover
                for k in range(len(regions))
            ):
                for i in c:                    # skip the merge; keep members as-is
                    out.append(regions[i])
                self.actions.append(("skip_merge", len(c)))
                continue
            labels = [regions[i].label for i in c]
            label = "advertisement" if any(l in ("doc_title", "figure_title") for l in labels) \
                else max(set(labels), key=labels.count)
            text, status = self._ocr(page_img, ub, label)
            if not text:                        # fallback: concatenate member reads
                text = "\n".join(regions[i].text for i in c if regions[i].text)
                status = "ok" if text else "error"
            elif self.lossless:                 # append any member text the re-OCR missed
                have = _toks(text)
                extra = [regions[i].text for i in c
                         if regions[i].text and (_toks(regions[i].text) - have)]
                if extra:
                    text = text + "\n" + "\n".join(extra)
                    self.actions.append(("merge_kept_member", len(extra)))
            out.append(Region(bbox=ub, image=page_img.crop(ub.to_tuple()),
                              label=label, text=text, status=status))
            self.actions.append(("merge", len(c)))
        return out
