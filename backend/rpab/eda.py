"""Helpers for the exploratory analysis in notebook section 2."""

import re

import pandas as pd
import tiktoken

BACK_MATTER_HEADING = re.compile(r"^\s*(?:references|bibliography)\s*$", re.IGNORECASE | re.MULTILINE)


def count_tokens(texts: pd.Series, encoding: str = "cl100k_base") -> pd.Series:
    """Token count per text. cl100k is an OpenAI tokenizer, so counts approximate Llama's."""
    enc = tiktoken.get_encoding(encoding)
    return texts.map(lambda text: len(enc.encode(text, disallowed_special=())))


def back_matter_split(pages: pd.DataFrame) -> pd.DataFrame:
    """Per paper: the page where the reference list starts, and body vs back-matter page counts.

    Back matter is everything from the "References" heading onward (references and appendices).
    The heading page itself is counted as back matter even though it usually ends with body text.
    """
    rows = []
    for paper_id, group in pages.sort_values("page").groupby("paper_id"):
        hits = group[(group["page"] > 1) & group["clean_text"].str.contains(BACK_MATTER_HEADING)]
        start = int(hits["page"].iloc[0]) if len(hits) else None
        total = len(group)
        body = (start - 1) if start else total
        rows.append({"paper_id": paper_id, "references_page": start, "body_pages": body,
                     "back_matter_pages": total - body, "total_pages": total})
    return pd.DataFrame(rows).set_index("paper_id")
