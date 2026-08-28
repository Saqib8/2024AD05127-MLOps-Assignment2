"""Render REPORT.md to a PDF.

Converts the markdown to a styled HTML page, then drives a headless Chrome or
Edge to print it. Both browsers ship with Windows or are already installed, so
this needs no LaTeX and no pandoc.

    python scripts/make_report_pdf.py
"""
from __future__ import annotations

import argparse
import base64
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]

BROWSERS = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]

CSS = """
@page { size: A4; margin: 17mm 15mm 18mm 15mm; }

body {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 10.5pt;
  line-height: 1.5;
  color: #1a1a1a;
  margin: 0;
}

h1 {
  font-size: 20pt;
  margin: 0 0 4pt 0;
  padding-bottom: 6pt;
  border-bottom: 2px solid #333;
  page-break-after: avoid;
}
h2 {
  font-size: 14pt;
  margin: 20pt 0 7pt 0;
  padding-bottom: 3pt;
  border-bottom: 1px solid #bbb;
  page-break-after: avoid;
}
h3 {
  font-size: 11.5pt;
  margin: 14pt 0 5pt 0;
  page-break-after: avoid;
}
h2 + p, h3 + p { margin-top: 0; }

p { margin: 0 0 8pt 0; text-align: justify; }

/* the title block at the top is a run of bold lines, keep it tight */
h1 + p { text-align: left; }

table {
  border-collapse: collapse;
  width: 100%;
  margin: 9pt 0 12pt 0;
  font-size: 9.5pt;
  page-break-inside: avoid;
}
th, td {
  border: 1px solid #c4c4c4;
  padding: 4pt 7pt;
  text-align: left;
  vertical-align: top;
}
th { background: #eceff2; font-weight: bold; }
tr:nth-child(even) td { background: #f8f9fa; }

code {
  font-family: Consolas, "Courier New", monospace;
  font-size: 9pt;
  background: #f2f3f5;
  padding: 1pt 3pt;
  border-radius: 2px;
}
pre {
  background: #f6f7f9;
  border: 1px solid #ddd;
  border-left: 3px solid #888;
  padding: 7pt 9pt;
  font-size: 8.5pt;
  line-height: 1.4;
  overflow-x: auto;
  white-space: pre-wrap;
  word-wrap: break-word;
  page-break-inside: avoid;
  margin: 8pt 0;
}
pre code { background: none; padding: 0; font-size: 8.5pt; }

ul, ol { margin: 0 0 9pt 0; padding-left: 18pt; }
li { margin-bottom: 3pt; }

img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 10pt auto;
  page-break-inside: avoid;
}

hr {
  border: none;
  border-top: 1px solid #d5d5d5;
  margin: 16pt 0;
}

strong { color: #000; }
a { color: #14507d; text-decoration: none; }
"""


def find_browser() -> Path:
    for path in BROWSERS:
        if path.exists():
            return path
    raise SystemExit("no Chrome or Edge found, cannot render the PDF")


def embed_images(html: str, base: Path) -> str:
    """Inline every local image as a data URI.

    Headless Chrome reading a file:// page is fussy about relative image paths.
    Embedding them sidesteps the problem entirely and makes the HTML portable.
    """

    def replace(match):
        src = match.group(1)
        if src.startswith(("http://", "https://", "data:")):
            return match.group(0)
        path = (base / src).resolve()
        if not path.exists():
            print("  warning: image not found, skipping:", src)
            return match.group(0)
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        suffix = path.suffix.lstrip(".").lower()
        mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
        return 'src="data:image/{};base64,{}"'.format(mime, payload)

    return re.sub(r'src="([^"]+)"', replace, html)


def build_html(source: Path) -> str:
    body = markdown.markdown(
        source.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    body = embed_images(body, source.parent)
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>MLOps Assignment 2</title>"
        "<style>{}</style></head><body>{}</body></html>".format(CSS, body)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render REPORT.md to PDF")
    parser.add_argument("--source", default=str(ROOT / "REPORT.md"))
    parser.add_argument("--output", default=str(ROOT / "REPORT.pdf"))
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)

    if not source.exists():
        raise SystemExit("{} does not exist".format(source))

    html = build_html(source)

    # a real file on disk rather than a temp handle, because the browser has to
    # open it by URL after this process has written and closed it
    with tempfile.NamedTemporaryFile(
        "w", suffix=".html", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(html)
        html_path = Path(handle.name)

    browser = find_browser()
    print("rendering with", browser.name)

    command = [
        str(browser),
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        "--print-to-pdf={}".format(output),
        html_path.as_uri(),
    ]

    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    html_path.unlink(missing_ok=True)

    if not output.exists():
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit("the browser did not produce a PDF")

    print("wrote {} ({:.2f} MB)".format(output, output.stat().st_size / 1048576))


if __name__ == "__main__":
    main()
