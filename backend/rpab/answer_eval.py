"""Step 7: answer evaluation, with manual scores first and an LLM judge checked against them.

Every answer is scored on the four dimensions of doc 05 (section 8.3) with one rubric,
``RUBRIC``, which the manual scorer and the judge both follow:

- relevance, accuracy, groundedness: 1 to 5
- hallucination: True when any claim is not supported by the three passages

Refusals are scored by whether refusing was right: a refusal on an out-of-scope question is
a perfect answer, a refusal on an answerable question fails relevance and accuracy.

Manual scores live in ``data/eval/answer_scores_manual.csv`` (hand-made data, tracked in git).
Generated answers and judge scores are cached in ``artifacts/answers/`` so the notebook can be
re-run without regenerating them.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from rpab.schemas import AskRequest

DIMENSIONS = ["relevance", "accuracy", "groundedness"]
FAILURE_CAUSES = ["retrieval_miss", "bad_chunk", "ambiguous_question", "ignored_context", "other"]

RUBRIC = """Score the answer on four dimensions.

relevance (1-5): does the answer address what was asked?
  5 fully addresses the question; 3 addresses part of it; 1 off-topic.
accuracy (1-5): does the answer agree with the reference answer?
  5 every key fact correct and complete; 4 correct with a minor omission; 3 partly correct or
  missing a key part; 2 mostly wrong; 1 wrong.
groundedness (1-5): what share of the answer's claims is supported by the passages?
  5 all; 4 nearly all; 3 about half; 2 few; 1 none.
hallucination (true/false): true if ANY claim is not supported by the passages, even a claim that
  happens to be true from outside knowledge.

Refusals ("I don't know based on the provided papers."):
- on a question the papers cannot answer: relevance 5, accuracy 5, groundedness 5, hallucination false.
- on a question the papers can answer: relevance 1, accuracy 1, groundedness 5, hallucination false.
An answer given to a question the papers cannot answer gets accuracy 1."""

NO_REFERENCE = ("(none: the papers do not answer this question, so the correct response is the refusal "
                "\"I don't know based on the provided papers.\")")


def model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


# ---- generation --------------------------------------------------------------------------

def generate_answers(pipeline, questions: list[dict]) -> pd.DataFrame:
    """Answer every question with the full pipeline; one row per question with timings,
    the passages shown to the model and whether a gold page was among them."""
    rows = []
    for q in questions:
        response = pipeline.answer(AskRequest(question=q["question"]))
        gold_in_context = any(s.page in q["gold_pages"].get(s.paper_id, []) for s in response.sources)
        rows.append({
            "question_id": q["id"], "split": q["split"], "type": q["type"], "answerable": q["answerable"],
            "question": q["question"], "reference_answer": q.get("reference_answer") or "",
            "model": pipeline.model_name, "answer": response.answer, "refused": response.refused,
            "weak_context": response.weak_context, "gold_in_context": gold_in_context if q["answerable"] else None,
            "sources": json.dumps([{"rank": s.rank, "paper_id": s.paper_id, "page": s.page, "chunk_id": s.chunk_id,
                                    "score": round(s.score, 4)} for s in response.sources]),
            "passages": json.dumps([s.full_text for s in response.sources], ensure_ascii=False),
            "removed_citations": json.dumps(response.removed_citations),
            "retrieval_ms": response.retrieval_ms, "first_token_ms": response.first_token_ms,
            "latency_ms": response.latency_ms,
        })
    return pd.DataFrame(rows)


def load_or_generate(pipeline, questions: list[dict], path: Path, rerun: bool = False) -> pd.DataFrame:
    path = Path(path)
    if path.exists() and not rerun:
        return pd.read_parquet(path)
    answers = generate_answers(pipeline, questions)
    path.parent.mkdir(parents=True, exist_ok=True)
    answers.to_parquet(path, index=False)
    return answers


def format_passages(row) -> str:
    sources, passages = json.loads(row["sources"]), json.loads(row["passages"])
    return "\n\n".join(f"[{s['rank']}] ({s['paper_id']}, page {s['page']})\n{text.strip()}"
                       for s, text in zip(sources, passages))


# ---- LLM judge ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are grading answers produced by a question-answering system over AI research papers.
The system saw only the numbered passages below and had to answer from them, citing [1], [2], [3].

{rubric}

Question: {question}

Reference answer: {reference}

Passages the system saw:
{passages}

System answer:
{answer}

Reply with a JSON object only, in this form:
{{"relevance": <1-5>, "accuracy": <1-5>, "groundedness": <1-5>, "hallucination": <true|false>,
"reason": "<one sentence in English>"}}"""


def _clip_score(value) -> int:
    return int(min(5, max(1, round(float(value)))))


def judge_answers(answers: pd.DataFrame, config, model: str | None = None) -> pd.DataFrame:
    """Score every answer with the judge model at temperature 0, in JSON mode."""
    from langchain_ollama import ChatOllama

    judge = ChatOllama(model=model or config.judge_model, base_url=config.ollama_base_url, temperature=0,
                       seed=config.seed, num_ctx=config.llm_num_ctx, format="json")
    rows = []
    for _, row in answers.iterrows():
        prompt = JUDGE_PROMPT.format(rubric=RUBRIC, question=row["question"],
                                     reference=row["reference_answer"] if row["answerable"] else NO_REFERENCE,
                                     passages=format_passages(row), answer=row["answer"])
        reply = json.loads(judge.invoke(prompt).content)
        rows.append({"question_id": row["question_id"], "answer_model": row["model"],
                     **{d: _clip_score(reply[d]) for d in DIMENSIONS},
                     "hallucination": bool(reply["hallucination"]) if not isinstance(reply["hallucination"], str)
                     else reply["hallucination"].strip().lower() == "true",
                     "reason": str(reply.get("reason", "")), "judge": "llm_judge"})
    return pd.DataFrame(rows)


def load_or_judge(answers: pd.DataFrame, config, path: Path, rerun: bool = False) -> pd.DataFrame:
    path = Path(path)
    if path.exists() and not rerun:
        return pd.read_parquet(path)
    scores = judge_answers(answers, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(path, index=False)
    return scores


# ---- manual scores and agreement ---------------------------------------------------------

def load_manual_scores(path: Path) -> pd.DataFrame:
    """The hand-scored sheet, checked against the rubric's ranges and failure causes."""
    scores = pd.read_csv(path, dtype={"failure_cause": "string", "notes": "string"})
    for d in DIMENSIONS:
        bad = scores[~scores[d].between(1, 5)]
        if not bad.empty:
            raise ValueError(f"{d} outside 1-5 for {', '.join(bad['question_id'])}")
    scores["hallucination"] = scores["hallucination"].astype(str).str.lower().map({"true": True, "false": False})
    unknown = set(scores["failure_cause"].dropna()) - set(FAILURE_CAUSES)
    if unknown:
        raise ValueError(f"Unknown failure causes: {', '.join(sorted(unknown))}")
    return scores


def agreement(manual: pd.DataFrame, judge: pd.DataFrame) -> pd.DataFrame:
    """Judge vs manual per dimension: exact and within-one agreement, quadratic-weighted kappa,
    and the judge's mean offset (positive = the judge is more lenient)."""
    from sklearn.metrics import cohen_kappa_score

    both = manual.merge(judge, on="question_id", suffixes=("_manual", "_judge"))
    rows = []
    for d in DIMENSIONS:
        m, j = both[f"{d}_manual"], both[f"{d}_judge"]
        kappa = cohen_kappa_score(m, j, weights="quadratic", labels=[1, 2, 3, 4, 5])
        rows.append({"dimension": d, "questions": len(both), "exact": (m == j).mean(),
                     "within_1": ((m - j).abs() <= 1).mean(), "kappa_quadratic": kappa,
                     "judge_minus_manual": (j - m).mean()})
    m, j = both["hallucination_manual"], both["hallucination_judge"]
    rows.append({"dimension": "hallucination", "questions": len(both), "exact": (m == j).mean(),
                 "within_1": np.nan, "kappa_quadratic": cohen_kappa_score(m, j, labels=[False, True]),
                 "judge_minus_manual": j.mean() - m.mean()})
    return pd.DataFrame(rows).set_index("dimension")


def score_summary(scores: pd.DataFrame, answers: pd.DataFrame) -> pd.Series:
    """Mean scores, hallucination rate and refusal correctness for one set of scored answers."""
    both = answers.merge(scores, on="question_id")
    out = both[DIMENSIONS].mean()
    out["hallucination_rate"] = both["hallucination"].mean()
    out["correct_refusals"] = f"{(both['refused'] & ~both['answerable']).sum()}/{(~both['answerable']).sum()}"
    out["wrong_refusals"] = f"{(both['refused'] & both['answerable']).sum()}/{both['answerable'].sum()}"
    out["median_first_token_ms"] = both["first_token_ms"].median()
    out["median_latency_ms"] = both["latency_ms"].median()
    return out
