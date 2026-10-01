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

    class FakePipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, image):
            seen.append(image.size)
            return f"page {len(seen)}"

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
