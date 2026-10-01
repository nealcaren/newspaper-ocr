import json
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner
from newspaper_ocr.cli import _parse_pages, main


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "backend" in result.output
    assert "output" in result.output


def test_cli_missing_file():
    runner = CliRunner()
    result = runner.invoke(main, ["nonexistent.jpg"])
    assert result.exit_code != 0


def _make_pdf(path, n_pages):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for i in range(n_pages):
        doc.new_page(width=200, height=100).insert_text((20, 50), f"Page {i + 1}")
    doc.save(str(path))


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Swap in a Pipeline that reports each page image it receives."""
    seen = []

    class FakeFormatter:
        def format(self, layout):
            return layout.text

    class FakePipeline:
        formatter = FakeFormatter()

        def __init__(self, **kwargs):
            pass

        def analyze(self, image):
            from newspaper_ocr.models import PageLayout
            if getattr(image, "size", None) is None:  # a path, not a PDF page
                return SimpleNamespace(text="image", regions=[])
            seen.append(image.size)
            if image.size == (1, 1):
                raise RuntimeError("model fell over")
            return SimpleNamespace(text=f"page {len(seen)}", regions=[])

        def run(self, image):
            return self.formatter.format(self.analyze(image))

        def ocr(self, path):
            return "image"

    monkeypatch.setattr("newspaper_ocr.Pipeline", FakePipeline)
    return seen


def test_cli_pdf_writes_one_file_per_page(tmp_path, fake_pipeline):
    pdf = tmp_path / "issue.pdf"
    _make_pdf(pdf, 3)
    out = tmp_path / "out"
    result = CliRunner().invoke(main, [str(pdf), "--outdir", str(out)])
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in out.iterdir()) == [
        "issue_p001.txt", "issue_p002.txt", "issue_p003.txt"]
    assert (out / "issue_p002.txt").read_text() == "page 2"


def test_cli_pdf_page_selection_and_rotation(tmp_path, fake_pipeline):
    pdf = tmp_path / "issue.pdf"
    _make_pdf(pdf, 4)
    out = tmp_path / "out"
    result = CliRunner().invoke(
        main, [str(pdf), "--outdir", str(out), "--pages", "2,4", "--rotate", "90"])
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in out.iterdir()) == ["issue_p002.txt", "issue_p004.txt"]
    width, height = fake_pipeline[0]
    assert height > width  # landscape page turned upright


def test_cli_pdf_rejects_out_of_range_pages(tmp_path, fake_pipeline):
    pdf = tmp_path / "issue.pdf"
    _make_pdf(pdf, 2)
    result = CliRunner().invoke(main, [str(pdf), "--pages", "3"])
    assert result.exit_code != 0
    assert "only 2 pages" in result.output


@pytest.mark.parametrize("spec", ["0", "3-1", "x", "1-"])
def test_parse_pages_rejects_bad_specs(spec):
    with pytest.raises(click.BadParameter):
        _parse_pages(spec)


def test_parse_pages():
    assert _parse_pages("1-3,7") == [0, 1, 2, 6]


def test_cli_batch_skip_existing_log_and_input_root(tmp_path, fake_pipeline):
    root = tmp_path / "in"
    (root / "1960").mkdir(parents=True)
    _make_pdf(root / "1960" / "a.pdf", 2)
    _make_pdf(root / "1960" / "b.pdf", 1)
    listing = tmp_path / "list.txt"
    listing.write_text(f"{root / '1960' / 'a.pdf'}\n{root / '1960' / 'b.pdf'}\n")
    out, log = tmp_path / "out", tmp_path / "log.jsonl"
    args = ["--files-from", str(listing), "--outdir", str(out), "--skip-existing",
            "--log", str(log), "--input-root", str(root)]

    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in (out / "1960").iterdir()) == [
        "a_p001.txt", "a_p002.txt", "b_p001.txt"]

    (out / "1960" / "a_p002.txt").unlink()
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    records = [json.loads(ln) for ln in log.read_text().splitlines()]
    assert [r["status"] for r in records[3:]] == ["skipped", "ok", "skipped"]
    assert records[4]["page"] == 2 and records[4]["seconds"] >= 0


def test_cli_batch_keeps_going_after_a_failing_page(tmp_path, fake_pipeline, monkeypatch):
    from PIL import Image
    import newspaper_ocr.pdf as pdf
    _make_pdf(tmp_path / "issue.pdf", 3)
    real = pdf.page_image
    # Page 2 decodes to a 1x1 image, which the fake pipeline fails on.
    monkeypatch.setattr(pdf, "page_image", lambda doc, page, **k: (
        Image.new("RGB", (1, 1)) if page.number == 1 else real(doc, page, **k)))
    out, log = tmp_path / "out", tmp_path / "log.jsonl"
    result = CliRunner().invoke(
        main, [str(tmp_path / "issue.pdf"), "--outdir", str(out), "--log", str(log)])
    assert result.exit_code == 1
    assert sorted(p.name for p in out.iterdir()) == ["issue_p001.txt", "issue_p003.txt"]
    failed = [json.loads(ln) for ln in log.read_text().splitlines()][1]
    assert failed["status"] == "error" and "model fell over" in failed["error"]


def test_cli_shard_splits_inputs(tmp_path, fake_pipeline):
    for name in "abcde":
        _make_pdf(tmp_path / f"{name}.pdf", 1)
    out = tmp_path / "out"
    paths = [str(tmp_path / f"{n}.pdf") for n in "abcde"]
    result = CliRunner().invoke(main, paths + ["--shard", "1/2", "--outdir", str(out)])
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in out.iterdir()) == ["b_p001.txt", "d_p001.txt"]


@pytest.mark.parametrize("spec", ["2/2", "-1/3", "x", "1"])
def test_cli_rejects_bad_shard(tmp_path, spec):
    result = CliRunner().invoke(main, [str(tmp_path), "--shard", spec])
    assert result.exit_code != 0


def test_cli_batch_flags_need_outdir(tmp_path):
    result = CliRunner().invoke(main, [str(tmp_path), "--skip-existing"])
    assert result.exit_code != 0 and "--outdir" in result.output


def test_cli_log_inside_new_outdir(tmp_path, fake_pipeline):
    _make_pdf(tmp_path / "issue.pdf", 1)
    out = tmp_path / "out"
    result = CliRunner().invoke(
        main, [str(tmp_path / "issue.pdf"), "--outdir", str(out), "--log", str(out / "log.jsonl")])
    assert result.exit_code == 0, result.output
    assert (out / "log.jsonl").exists()
