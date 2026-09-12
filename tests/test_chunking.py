"""Tests for vertical chunking helpers."""

from newspaper_ocr import chunking


class TestChunkSpans:
    def test_short_region_is_a_single_span(self):
        assert chunking.chunk_spans(400, chunk_height=500) == [(0, 400)]
        assert chunking.chunk_spans(500, chunk_height=500) == [(0, 500)]

    def test_tall_region_splits_with_overlap(self):
        spans = chunking.chunk_spans(1200, chunk_height=500, overlap=50)
        # bands: 0-500, 450-950, 900-1200
        assert spans == [(0, 500), (450, 950), (900, 1200)]

    def test_spans_cover_full_height_and_end_exactly(self):
        spans = chunking.chunk_spans(1337, chunk_height=500, overlap=50)
        assert spans[0][0] == 0
        assert spans[-1][1] == 1337
        # consecutive bands overlap by exactly `overlap`
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            assert a1 - b0 == 50


class TestMergeChunkTexts:
    def test_empty(self):
        assert chunking.merge_chunk_texts([]) == ""

    def test_single(self):
        assert chunking.merge_chunk_texts(["hello world"]) == "hello world"

    def test_overlap_is_deduplicated(self):
        # end of chunk A repeats at the start of chunk B
        a = "the quick brown fox jumps over the lazy dog"
        b = "over the lazy dog and then went to sleep"
        merged = chunking.merge_chunk_texts([a, b])
        assert merged == "the quick brown fox jumps over the lazy dog and then went to sleep"
        assert merged.count("over the lazy dog") == 1

    def test_no_overlap_joins_with_newline(self):
        merged = chunking.merge_chunk_texts(["first paragraph here", "totally different second"])
        assert merged == "first paragraph here\ntotally different second"
