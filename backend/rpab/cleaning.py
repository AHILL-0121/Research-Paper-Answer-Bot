"""Step 1b: clean extracted page text without losing citation metadata.

Rules, applied in this order to every page:
1. Replace typographic ligatures (U+FB00-FB06) with plain letters, e.g. "ﬁ" -> "fi".
2. Decode lines written in a shifted font encoding. Some figures (ReAct) embed fonts whose
   character codes sit 29 below the real letters, so "Think" is extracted as "7KLQN" and the
   space as the control character U+0003. A line where U+0003 stands in for the space is shifted
   back; any other stray control character (a symbol glyph such as a check mark) is deleted.
3. Remove the arXiv side stamp, e.g. "arXiv:1706.03762v7 [cs.CL] 2 Aug 2023".
4. Drop repeated header/footer lines: a line at the top or bottom edge of a page whose
   shape (digits replaced by "#") appears at the edges of many pages of the same paper.
   This catches page numbers and running heads like "Published as a conference paper at ICLR 2023".
5. Re-join words hyphenated across a line break, e.g. "ef-\\nficient" -> "efficient".
6. Normalise whitespace.

The raw text is kept beside the cleaned text in ``pages.parquet`` for debugging.
"""

import re
from collections import Counter

import pandas as pd
from langchain_core.documents import Document

LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}
LIGATURE = re.compile("[ﬀ-ﬆ]")
ARXIV_STAMP = re.compile(r"arXiv:\d{4}\.\d{4,5}v\d+\s*\[[^\]]+\]\s*\d{1,2}\s*[A-Z][a-z]{2}\s*\d{4}")
HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
CONTROL_CHARS = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f]")  # never part of real text
SHIFTED_SPACE = "\x03"  # how a space comes out of the shifted font (32 - 29)
FONT_SHIFT = 29
EDGE = 2  # how many lines at the top and bottom of a page count as its edges


def _unshift(line: str) -> str:
    """Move every character of a shifted-encoding line back by FONT_SHIFT code points."""
    return "".join(chr(ord(c) + FONT_SHIFT) if ord(c) + FONT_SHIFT <= 126 else c for c in line)


def _is_shifted(line: str) -> bool:
    """A shifted line uses U+0003 as its space, so it has more of them than real spaces."""
    return line.count(SHIFTED_SPACE) > line.count(" ")


def decode_shifted_lines(text: str) -> tuple[str, int, int]:
    """Decode shifted-font lines, then delete stray control characters.

    Returns the text, the number of lines decoded and the number of stray characters removed.
    """
    lines = text.split("\n")
    decoded = [_unshift(line) if _is_shifted(line) else line for line in lines]
    n_decoded = sum(a != b for a, b in zip(lines, decoded))
    text, n_stray = CONTROL_CHARS.subn("", "\n".join(decoded))
    return text, n_decoded, n_stray


def _shape(line: str) -> str:
    """Normalise a line so "Page 3" and "Page 14" look the same."""
    return re.sub(r"\d+", "#", line.strip().lower())


def _lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def find_repeated_lines(page_texts: list[str], min_share: float) -> set[str]:
    """Shapes that sit at a page edge on at least ``min_share`` of a paper's pages."""
    if len(page_texts) < 4:  # too few pages to tell a header from content
        return set()
    counts = Counter()
    for text in page_texts:
        lines = _lines(text)
        counts.update({_shape(line) for line in lines[:EDGE] + lines[-EDGE:]})
    return {shape for shape, n in counts.items() if n >= min_share * len(page_texts)}


def clean_text(text: str, repeated: set[str]) -> tuple[str, dict]:
    """Apply the cleaning rules to one page and count what each rule changed."""
    log = {"ligatures_fixed": len(LIGATURE.findall(text))}
    text = LIGATURE.sub(lambda m: LIGATURES[m.group()], text)

    text, log["shifted_lines_decoded"], log["control_chars_removed"] = decode_shifted_lines(text)

    text, log["arxiv_stamps_removed"] = ARXIV_STAMP.subn("", text)

    lines = _lines(text)
    edges = set(range(min(EDGE, len(lines)))) | set(range(max(len(lines) - EDGE, 0), len(lines)))
    kept = [line for i, line in enumerate(lines) if not (i in edges and _shape(line) in repeated)]
    log["edge_lines_removed"] = len(lines) - len(kept)
    text = "\n".join(kept)

    text, log["hyphens_joined"] = HYPHEN_BREAK.subn(r"\1\2", text)

    text = re.sub(r"[ \t ]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, log


def clean_documents(docs: list[Document], min_share: float) -> tuple[list[Document], pd.DataFrame]:
    """Clean every page; return cleaned Documents and a per-page log with the raw text."""
    by_paper: dict[str, list[Document]] = {}
    for doc in docs:
        by_paper.setdefault(doc.metadata["paper_id"], []).append(doc)

    cleaned, rows = [], []
    for paper_id, paper_docs in by_paper.items():
        repeated = find_repeated_lines([d.page_content for d in paper_docs], min_share)
        for doc in paper_docs:
            text, log = clean_text(doc.page_content, repeated)
            cleaned.append(Document(page_content=text, metadata=dict(doc.metadata)))
            rows.append({**doc.metadata, "raw_text": doc.page_content, "clean_text": text,
                         "raw_chars": len(doc.page_content), "clean_chars": len(text), **log})
    return cleaned, pd.DataFrame(rows)
