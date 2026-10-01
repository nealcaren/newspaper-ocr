import pytest

from newspaper_ocr import repetition
from newspaper_ocr.markup import to_plain

# Box score as MinerU2.5 returns it (Daily Tar Heel, March 4, 1960).
BOX_SCORE = (
    "<table><tr><td>CLEMSON</td><td>G</td><td>F</td><td>P</td><td>T</td></tr>"
    + "".join(
        f"<tr><td>{name}</td><td>3</td><td>8-10</td><td>1</td><td>14</td></tr>"
        for name in ["Patterson", "Krajack", "Mahaffey", "Gibbons", "Carver", "Wallace"]
    )
    + "</table>"
)


def test_table_becomes_tab_separated_rows():
    lines = to_plain(BOX_SCORE).split("\n")
    assert lines[0] == "CLEMSON\tG\tF\tP\tT"
    assert lines[1] == "Patterson\t3\t8-10\t1\t14"
    assert len(lines) == 7


def test_table_truncated_mid_tag():
    assert to_plain("<table><tr><td>Wallace</td><td>0</td><td") == "Wallace\t0"


def test_html_entities_and_breaks():
    assert to_plain("Smith &amp; Sons<br>Durham") == "Smith & Sons\nDurham"


def test_angle_brackets_in_prose_are_kept():
    assert to_plain("x < y and a > b") == "x < y and a > b"


@pytest.mark.parametrize("raw, plain", [
    (r"Made \(\$ 25,000\)in Three Months", "Made $25,000 in Three Months"),
    (r"a subscription of \(500 presented", "a subscription of $500 presented"),
    (r"held the floor \(4^{12}\) hours", "held the floor 4½ hours"),
    (r"\(\frac{1}{2}\) mile", "½ mile"),
    (r"low \(32^{\circ}\) today", "low 32° today"),
    (r"\( \mathrm{b} \) and \(\left( {x - {2x}}\right)\)", "b and ( x - 2x)"),
])
def test_latex(raw, plain):
    assert to_plain(raw) == plain


def test_markdown():
    text = "Dennis Rash  \nReceive UP\n## Endorsement\n**Bold** 5 * 3 *star*"
    assert to_plain(text) == "Dennis Rash\nReceive UP\nEndorsement\nBold 5 * 3 *star*"


def test_plain_text_unchanged():
    text = "Entries accompanied by a $2 fee. Goal $3,000.\n\nNext paragraph."
    assert to_plain(text) == text


def test_table_is_not_a_repetition_loop():
    assert not repetition.has_repetition(BOX_SCORE)
    assert repetition.has_repetition("the same phrase again " * 10)
