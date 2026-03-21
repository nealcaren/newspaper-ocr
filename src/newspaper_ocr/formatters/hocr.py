from newspaper_ocr.formatters.base import Formatter
from newspaper_ocr.models import PageLayout


class HocrFormatter(Formatter):
    def format(self, layout: PageLayout) -> str:
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">',
            '<html xmlns="http://www.w3.org/1999/xhtml">',
            "<head><title>OCR Output</title></head>",
            "<body>",
            f'  <div class="ocr_page" title="bbox 0 0 {layout.width} {layout.height}">',
        ]
        for i, region in enumerate(layout.regions):
            rb = region.bbox
            lines.append(f'    <div class="ocr_carea" id="block_{i}" title="bbox {rb.x0} {rb.y0} {rb.x1} {rb.y1}">')
            for j, line in enumerate(region.lines):
                lb = line.bbox
                lines.append(f'      <span class="ocr_line" id="line_{i}_{j}" title="bbox {lb.x0} {lb.y0} {lb.x1} {lb.y1}">{_escape(line.text)}</span>')
            lines.append("    </div>")
        lines.append("  </div>")
        lines.append("</body>")
        lines.append("</html>")
        return "\n".join(lines)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
