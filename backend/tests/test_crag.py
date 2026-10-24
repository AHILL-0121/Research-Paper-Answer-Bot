"""Corrective RAG web fallback (stretch S3) with a stub search and stub reranker: no network, no models."""

from rpab.crag import corrective_retrieve
from rpab.prompts import format_context
from rpab.rag import to_sources
from rpab.schemas import RetrievedChunk


def paper(i: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=f"lora_p00{i}_c000", paper_id="lora", paper_title="LoRA", page=i,
                          text=f"paper passage {i}", score=score, rerank_score=score)


def web(i: int) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=f"web_{i}", paper_id="web", paper_title=f"Web result {i}", page=0,
                          text=f"web snippet {i}", score=0.0, origin="web", url=f"https://example.org/{i}")


class Reranker:
    def __init__(self, scores: dict[str, float]):
        self.scores = scores

    def score(self, question, texts):
        return [self.scores.get(t.split(". ", 1)[-1], 0.0) for t in texts]


class Retriever:
    def __init__(self, reranker=None):
        self.reranker = reranker

    def confidence(self, results):
        return (results[0].rerank_score, "rerank") if results else (None, "none")


class Search:
    def __init__(self, results=None, fail=False):
        self.results, self.fail = results or [], fail

    def search(self, question):
        if self.fail:
            raise TimeoutError("no answer")
        return self.results


DOCS = [paper(1, 0.30), paper(2, 0.20), paper(3, 0.10)]


def test_reranker_decides_which_web_results_get_in():
    retriever = Retriever(Reranker({"web snippet 1": 0.90, "web snippet 2": 0.15}))
    merged, note = corrective_retrieve("q", DOCS, retriever, Search([web(1), web(2)]), k=3)
    assert [c.chunk_id for c in merged] == ["web_1", "lora_p001_c000", "lora_p002_c000"]
    assert "0.30, below the 0.72 threshold" in note and "1 of the 3 passages come from the web" in note


def test_papers_win_when_no_web_result_scores_higher():
    retriever = Retriever(Reranker({"web snippet 1": 0.05}))
    merged, note = corrective_retrieve("q", DOCS, retriever, Search([web(1)]), k=3)
    assert all(c.origin == "paper" for c in merged) and "uses the papers only" in note


def test_without_a_reranker_the_top_paper_passage_is_kept():
    merged, _ = corrective_retrieve("q", DOCS, Retriever(None), Search([web(1), web(2), web(3)]), k=3)
    assert [c.chunk_id for c in merged] == ["lora_p001_c000", "web_1", "web_2"]


def test_a_failed_search_falls_back_to_the_papers():
    merged, note = corrective_retrieve("q", DOCS, Retriever(None), Search(fail=True), k=3)
    assert merged == DOCS and "search failed (TimeoutError)" in note


def test_web_passages_are_labelled_in_the_prompt_and_the_sources():
    context = format_context([DOCS[0], web(1)])
    assert "[1] Paper: LoRA | Page: 1" in context and "[2] Web: Web result 1 | https://example.org/1" in context
    sources = to_sources([DOCS[0], web(1)])
    assert sources[1].origin == "web" and sources[1].url == "https://example.org/1" and sources[1].paper_id is None
