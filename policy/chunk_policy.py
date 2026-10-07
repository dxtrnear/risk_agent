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

# Excluded: low-relevance to claims decisions (4.3 legal-protection procedure, all of
# 11 contract-lifecycle administration, all of 13 personal-data/GDPR) and/or
# disproportionately large for what they add (5.2.1-5.2.3 assistance/towing logistics;
# 4.3.1 alone is 17.5K chars, the single biggest chunk in the index). 5.2.4 (Contenu du
# véhicule) is kept: 6.6's own exclusion text defers to it by name for unfixed objects.
EXCLUDED_IDS = {"5.2", "5.2.1", "5.2.2", "5.2.3"}
EXCLUDED_CHAPTERS = {"11", "13"}


def _is_excluded(clause_id: str) -> bool:
    if clause_id in EXCLUDED_IDS:
        return True
    if clause_id.split(".")[0] in EXCLUDED_CHAPTERS:
        return True
    return clause_id == "4.3" or clause_id.startswith("4.3.")


def _clean(text: str) -> str:
    for bad, good in LIGATURES.items():
        text = text.replace(bad, good)
    return re.sub(r"\s+", " ", text).strip()


def _is_noise(line: str) -> bool:
    return any(pattern.match(line) for pattern in NOISE_LINE_RES)


def chunk(pdf_path) -> list[dict]:
    """Parse the policy PDF into {clause_id, title, text} dicts."""
    reader = PdfReader(pdf_path)
    pages = [
        p.extract_text(extraction_mode="layout")
        for p in reader.pages[FIRST_PAGE_INDEX : LAST_PAGE_INDEX + 1]
    ]
    lines = "\n".join(pages).split("\n")

    clauses: list[dict] = []
    current_id: str | None = None
    current_title = ""
    current_chapter: int | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current_id is None or _is_excluded(current_id):
            return
        text = _clean(" ".join(buffer))
        if len(text.split()) >= MIN_WORDS:
            clauses.append(
                {"clause_id": current_id, "title": current_title, "text": text}
            )

    for line in lines:
        if _is_noise(line):
            continue

        match = HEADING_RE.match(line.strip())
        if match:
            number, title = match.groups()
            top_level = int(number.split(".")[0])
            is_new_chapter = (
                "." not in number and top_level == (current_chapter or 2) + 1
            )
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


# 10.1 ("Modalités de gestion") bundles claims-handling procedure for four unrelated
# garanties under one heading -- 11,937 chars, the largest chunk in the index. Split
# on its internal sub-headings so each garantie's procedure sits with its own topic.
# Unlike every other clause_id in this file, "10.1.1".."10.1.4" are NOT printed in the
# PDF -- 10.1 has no native sub-numbering here -- they're assigned below.
SPLIT_10_1_MARKERS = [
    ("10.1.1", "Responsabilité civile", "Que faisons-nous en cas de sinistre « Responsabilité Civile » ?"),
    ("10.1.2", "Bris de glace", "Que faisons-nous en cas de sinistre Bris de glace ?"),
    (
        "10.1.3",
        "Dommages subis par le véhicule et vol",
        "Que faisons-nous en cas de sinistre « Dommages subis par le véhicule » ?",
    ),
    ("10.1.4", "Garantie du conducteur", "Règles propres à la Garantie du conducteur"),
]


def _split_clause_10_1(clauses: list[dict]) -> list[dict]:
    idx = next((i for i, c in enumerate(clauses) if c["clause_id"] == "10.1"), None)
    if idx is None:
        return clauses
    text = clauses[idx]["text"]

    positions = [(cid, title, text.find(marker)) for cid, title, marker in SPLIT_10_1_MARKERS]
    if any(pos == -1 for _, _, pos in positions):
        raise ValueError("10.1 split marker not found in source text -- update SPLIT_10_1_MARKERS")

    parts = []
    for i, (cid, title, start) in enumerate(positions):
        end = positions[i + 1][2] if i + 1 < len(positions) else len(text)
        parts.append({"clause_id": cid, "title": title, "text": text[start:end].strip()})

    return clauses[:idx] + parts + clauses[idx + 1 :]


if __name__ == "__main__":
    clauses = chunk(config.POLICY_PDF)
    clauses = _split_clause_10_1(clauses)
    config.CLAUSES_JSON.write_text(json.dumps(clauses, ensure_ascii=False, indent=2))
    print(f"Saved {len(clauses)} clauses to {config.CLAUSES_JSON}")
