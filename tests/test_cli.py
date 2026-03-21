from click.testing import CliRunner
from newspaper_ocr.cli import main


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
