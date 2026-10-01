"""Turn VLM markup (HTML tables, LaTeX, Markdown) into plain newspaper text.

Document VLMs — MinerU2.5 above all — answer in the markup they were trained to
emit: tables as ``<table><tr><td>…``, anything math-like as ``\\( … \\)``, and
Markdown hard breaks, headings and bold.  On a newspaper that markup is noise, so
:func:`to_plain` strips it by default (``Pipeline(markup="raw")`` keeps it).

What it handles, all seen in real MinerU output:

*HTML tables* (box scores, stock tables, mastheads) become one line per row
with tab-separated cells; other known tags are dropped and entities unescaped.
Text that merely contains ``<`` is left alone.

*LaTeX* spans are flattened: ``\\$`` → ``$``, ``\\frac{1}{2}`` → ``½``,
``\\left``/``\\right``/``\\mathrm`` wrappers and braces removed.  MinerU also
reads a printed dollar sign as an opening ``\\(`` with no closing ``\\)``; a
lone opener becomes ``$`` again.

*Markdown* hard breaks (two trailing spaces), ``#`` headings and ``**bold**``
markers are removed.  Single-``*`` italics are left alone — newspapers use
asterisks for footnotes and bylines.
"""
from __future__ import annotations

import html
import re
from html.parser import HTMLParser

MODES = ("plain", "raw")

# Tags VLMs emit. Only text containing one of these is treated as HTML.
_HTML_TAG = re.compile(
    r"</?(?:table|thead|tbody|tfoot|tr|td|th|br|p|div|span|sup|sub|b|i|u|em"
    r"|strong|caption|h[1-6])\b[^>]*>",
    re.IGNORECASE,
)
_CELL_END = {"td", "th"}
_LINE_END = {"tr", "br", "p", "div", "caption", "h1", "h2", "h3", "h4", "h5", "h6"}


class _TextExtractor(HTMLParser):
    """Collect text, emitting a tab after each cell and a newline after each row."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _CELL_END:
            self.parts.append("\t")
        elif tag in _LINE_END:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def strip_html(text: str) -> str:
    """Strip HTML tags (tables to tab-separated rows); other text is untouched."""
    return _strip_html(text) if _HTML_TAG.search(text) else text


def _strip_html(text: str) -> str:
    # A region cut off mid-tag (e.g. by repetition truncation) ends in "<td".
    text = re.sub(r"<[^<>]*$", "", text)
    parser = _TextExtractor()
    parser.feed(text)
    parser.close()
    out = "".join(parser.parts)
    # Trailing cell tabs and blank rows left by the row/cell markers.
    lines = [ln.rstrip("\t ").replace("\t\t", "\t") for ln in out.split("\n")]
    return "\n".join(ln for ln in lines if ln.strip())


# Superscript digit pairs MinerU writes for printed fractions, e.g. "4\(^{12}\)".
_FRACTIONS = {"12": "½", "14": "¼", "34": "¾", "13": "⅓", "23": "⅔", "18": "⅛"}
_SYMBOLS = {
    r"\therefore": "∴", r"\circ": "°", r"\times": "×", r"\cdot": "·",
    r"\prime": "′", r"\pm": "±", r"\div": "÷", r"\lbrack": "[", r"\rbrack": "]",
    r"\%": "%", r"\$": "$", r"\&": "&", r"\#": "#", r"\_": "_",
}
_MATH_SPAN = re.compile(r"\\\((.*?)\\\)|\\\[(.*?)\\\]", re.DOTALL)
_FRAC = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_WRAPPER = re.compile(r"\\(?:mathrm|text|textbf|mathbf|operatorname|mathit)\s*\{([^{}]*)\}")
_SUPER_PAIR = re.compile(r"\^\s*\{?\s*(\d)\s*(\d)\s*\}?")


def _flatten_math(body: str) -> str:
    for cmd, sym in _SYMBOLS.items():
        body = body.replace(cmd, sym)
    body = re.sub(r"\\(?:left|right)\b\s*", "", body)
    body = _WRAPPER.sub(r"\1", body)
    # Unwrap brace groups that aren't fraction arguments, innermost first, so
    # nested fractions reduce to the simple form _FRAC matches.
    prev = None
    while prev != body:
        prev = body
        body = re.sub(r"(?<![}a-z])\{([^{}]*)\}", r"\1", body)
    body = _FRAC.sub(
        lambda m: _FRACTIONS.get(m[1].strip() + m[2].strip(),
                                 f"{m[1].strip()}/{m[2].strip()}"),
        body,
    )
    body = _SUPER_PAIR.sub(lambda m: _FRACTIONS.get(m[1] + m[2], m[0]), body)
    body = re.sub(r"\^\s*(?=[°′])", "", body)  # "32^{\circ}" -> "32°"
    body = re.sub(r"\\[a-zA-Z]+\s*", "", body)  # anything else unknown
    body = body.replace("{", "").replace("}", "")
    body = re.sub(r"\s+", " ", body).strip()
    return re.sub(r"\$ (?=\d)", "$", body)  # "\$ 25,000" -> "$25,000"


def _strip_latex(text: str) -> str:
    text = _MATH_SPAN.sub(lambda m: _flatten_math(m[1] if m[1] is not None else m[2]),
                          text)
    # MinerU drops the space after a closing "\)": "$25,000in Three Months".
    text = re.sub(r"(\$[\d,.]+)(?=[A-Za-z])", r"\1 ", text)
    # Leftover openers had no closer: MinerU's reading of a printed "$".
    return re.sub(r"\\\(\s*", "$", text)


def _strip_markdown(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)            # hard breaks / trailing space
    text = re.sub(r"(?m)^#{1,6}[ \t]+", "", text)      # headings
    return re.sub(r"\*\*(\S(?:.*?\S)?)\*\*", r"\1", text)  # bold


def to_plain(text: str) -> str:
    """Return *text* with HTML, LaTeX and Markdown markup removed."""
    if not text:
        return text
    text = strip_html(text)
    if "\\(" in text or "\\[" in text:
        text = _strip_latex(text)
    return _strip_markdown(text).strip()
