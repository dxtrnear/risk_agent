"""Split the policy PDF into clauses, keyed by the document's own numbering."""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pypdf import PdfReader

import config

# Chapters 3-13 (coverage, exclusions, franchise, claims, indemnification, contract
# life, formules, personal data) span printed pages 14-73, i.e. PDF indices 13-72.
# Chapters 1, 2 (intro/contacts), 14-16 (glossary, RC info sheet, AXA's corporate
# bylaws) carry nothing a claims decision would cite.
FIRST_PAGE_INDEX = 13
LAST_PAGE_INDEX = 72

HEADING_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,4})\.\s*(\S.*)$")
NOISE_LINE_RES = [
    re.compile(r"^\s*Sommaire\s+p\.\s*1\s+Définitions\s+p\.\s*74\s*$"),
    re.compile(r"^Conditions générales Mon Auto"),
    re.compile(r"^\s*\d{1,3}\s*$"),  # bare page number
]
LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}
MIN_WORDS = 15  # drops near-empty "umbrella" chapter headings with no body of their own


def _clean(text: str) -> str:
    for bad, good in LIGATURES.items():
        text = text.replace(bad, good)
    return re.sub(r"\s+", " ", text).strip()


def _is_noise(line: str) -> bool:
    return any(pattern.match(line) for pattern in NOISE_LINE_RES)


def chunk(pdf_path) -> list[dict]:
    """Parse the policy PDF into {clause_id, title, text} dicts."""
    reader = PdfReader(pdf_path)
    pages = [p.extract_text(extraction_mode="layout") for p in reader.pages[FIRST_PAGE_INDEX : LAST_PAGE_INDEX + 1]]
    lines = "\n".join(pages).split("\n")

    clauses: list[dict] = []
    current_id: str | None = None
    current_title = ""
    current_chapter: int | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current_id is None:
            return
        text = _clean(" ".join(buffer))
        if len(text.split()) >= MIN_WORDS:
            clauses.append({"clause_id": current_id, "title": current_title, "text": text})

    for line in lines:
        if _is_noise(line):
            continue

        match = HEADING_RE.match(line.strip())
        if match:
            number, title = match.groups()
            top_level = int(number.split(".")[0])
            is_new_chapter = "." not in number and top_level == (current_chapter or 2) + 1
            is_same_chapter = top_level == current_chapter

            if is_new_chapter or is_same_chapter:
                flush()
                current_id, current_title, buffer = number, _clean(title), []
                current_chapter = top_level
                continue
            # else: a locally-scoped "Article N" sub-list reusing x.y numbering
            # (e.g. inside 11.11) — not a real clause boundary, keep as body text.

        buffer.append(line)

    flush()
    return clauses


if __name__ == "__main__":
    clauses = chunk(config.POLICY_PDF)
    config.CLAUSES_JSON.write_text(json.dumps(clauses, ensure_ascii=False, indent=2))
    print(f"Saved {len(clauses)} clauses to {config.CLAUSES_JSON}")
