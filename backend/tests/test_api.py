"""API contract tests over a fake pipeline: no GPU, models or Ollama needed.

Run from the backend folder: ``.venv\\Scripts\\python -m pytest``
"""

import json
import threading
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda

from api.main import create_app
from api.runtime import Runtime
from rpab.config import CONFIG
from rpab.schemas import AskRequest, AskResponse, Source

DEAD_OLLAMA = "http://127.0.0.1:9"  # nothing listens on the discard port
QUESTION = {"question": "What does LoRA freeze?"}

SOURCES = [Source(rank=i, paper_title="LoRA", page=4, snippet="Low-rank adaptation", full_text="Low-rank adaptation",
                  score=0.9, chunk_id=f"lora_p004_c00{i}", paper_id="lora") for i in (1, 2, 3)]


class FakePipeline:
    model_name = "fake"

    def __init__(self, fail: Exception | None = None, rewrite: str = "What rank r did the LoRA paper use?"):
        self.fail = fail
        self.requests: list[AskRequest] = []
        self.llm = RunnableLambda(lambda prompt: rewrite)  # the question rewriter's chat model

    def warm_up(self, retrieval: bool = True, llm: bool = True) -> None:
        pass

    def stream(self, request: AskRequest):
        self.requests.append(request)
        yield {"event": "sources", "sources": SOURCES, "weak_context": False, "retrieval_ms": 5}
        if self.fail:
            raise self.fail
        for piece in ["LoRA freezes", " the weights [1]."]:
            yield {"event": "token", "text": piece}
        yield {"event": "done", "response": AskResponse(answer="LoRA freezes the weights [1].", sources=SOURCES,
                                                        strategy_used=request.strategy, latency_ms=10,
                                                        first_token_ms=4)}

    def answer(self, request: AskRequest) -> AskResponse:
        for event in self.stream(request):
            if event["event"] == "done":
                return event["response"]


@pytest.fixture
def config(tmp_path):
    """The real config with chat history in a throwaway database."""
    return replace(CONFIG, paths=replace(CONFIG.paths, chat_db=tmp_path / "chat.db"))


def client_for(config, pipeline=None, loader=None) -> TestClient:
    runtime = Runtime(config, loader=loader or (lambda cfg: pipeline or FakePipeline()))
    return TestClient(create_app(runtime, background_load=False))


def sse_events(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_ask_returns_the_pipeline_answer(config):
    pipeline = FakePipeline()
    with client_for(config, pipeline) as client:
        reply = client.post("/ask", json={**QUESTION, "paper_ids": ["lora"]})
    assert reply.status_code == 200
    body = reply.json()
    assert body["answer"] == "LoRA freezes the weights [1]."
    assert [s["rank"] for s in body["sources"]] == [1, 2, 3]
    assert pipeline.requests[0].paper_ids == ["lora"]
    assert body["rewritten_question"] is None  # no session, no memory


def test_stream_sends_sources_tokens_then_done(config):
    with client_for(config) as client:
        reply = client.post("/ask/stream", json=QUESTION)
    assert reply.headers["content-type"].startswith("text/event-stream")
    events = sse_events(reply.text)
    assert [kind for kind, _ in events] == ["sources", "token", "token", "done"]
    assert len(events[0][1]["sources"]) == 3
    assert "".join(data["text"] for kind, data in events if kind == "token") == events[-1][1]["answer"]


def test_validation_errors_are_friendly(config):
    with client_for(config) as client:
        short = client.post("/ask", json={"question": "hi"})
        unknown = client.post("/ask", json={**QUESTION, "paper_ids": ["nope"]})
        bad_session = client.post("/ask", json={**QUESTION, "session_id": "../etc"})
    assert short.status_code == 422 and short.json()["error"]["code"] == "invalid_request"
    assert "question" in short.json()["error"]["message"]
    assert unknown.status_code == 422 and unknown.json()["error"]["code"] == "unknown_paper"
    assert bad_session.status_code == 422 and "session_id" in bad_session.json()["error"]["message"]


def test_missing_index_is_reported(config, tmp_path):
    config = replace(config, paths=replace(config.paths, chunks=tmp_path / "chunks.parquet"))
    with client_for(config) as client:
        health = client.get("/health").json()
        reply = client.post("/ask", json=QUESTION)
        history = client.get("/sessions")
    assert health["status"] == "error" and health["error"]["code"] == "index_not_built"
    assert reply.status_code == 503 and reply.json()["error"]["code"] == "index_not_built"
    assert history.status_code == 200  # chat history does not need the models


def test_ollama_down_is_reported(config):
    config = replace(config, ollama_base_url=DEAD_OLLAMA)
    pipeline = FakePipeline(fail=ConnectionError("Failed to connect to Ollama"))
    with client_for(config, pipeline) as client:
        reply = client.post("/ask", json=QUESTION)
        stream = client.post("/ask/stream", json=QUESTION)
        health = client.get("/health").json()
    assert reply.status_code == 503 and reply.json()["error"]["code"] == "ollama_unavailable"
    assert sse_events(stream.text)[-1] == ("error", reply.json()["error"])
    assert health["ollama_running"] is False and health["status"] == "ready"


def test_unexpected_errors_hide_the_trace(config, monkeypatch):
    pipeline = FakePipeline(fail=KeyError("secret internals"))
    config = replace(config, llm_model="present:latest")
    monkeypatch.setattr("rpab.errors.ollama_models", lambda cfg: ["present:latest"])
    with client_for(config, pipeline) as client:
        reply = client.post("/ask", json=QUESTION)
    assert reply.status_code == 500
    assert reply.json()["error"]["code"] == "internal_error" and "secret" not in reply.text


def test_requests_wait_while_models_load(config):
    release = threading.Event()

    def slow_loader(cfg):
        release.wait(5)
        return FakePipeline()

    runtime = Runtime(config, loader=slow_loader)
    with TestClient(create_app(runtime, background_load=True)) as client:
        loading = client.post("/ask", json=QUESTION)
        assert client.get("/health").json()["status"] == "loading"
        release.set()
    assert loading.status_code == 503 and loading.json()["error"]["code"] == "warming_up"


def test_papers_and_cors(config):
    with client_for(config) as client:
        papers = client.get("/papers").json()
        preflight = client.options("/ask", headers={"Origin": "http://localhost:3000",
                                                    "Access-Control-Request-Method": "POST"})
    assert {"paper_id", "paper_title", "num_chunks"} <= papers[0].keys()
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_sessions_remember_rewrite_and_clear(config):
    pipeline = FakePipeline(rewrite="What rank r did the LoRA paper use?")
    session = {"session_id": "tab-1234abcd"}
    with client_for(config, pipeline) as client:
        first = client.post("/ask", json={"question": "What is LoRA?", **session}).json()
        stream = sse_events(client.post("/ask/stream", json={"question": "What rank did they use?",
                                                             **session}).text)
        listed = client.get("/sessions").json()
        detail = client.get(f"/sessions/{session['session_id']}").json()
        cleared = client.delete(f"/sessions/{session['session_id']}")
        gone = client.get(f"/sessions/{session['session_id']}")
        clear_again = client.delete(f"/sessions/{session['session_id']}")

    # The first question has no history, so it is used as is; the follow-up is rewritten before retrieval
    assert first["rewritten_question"] == "What is LoRA?" and first["rewrite_ms"] is None
    assert stream[0] == ("rewrite", {"rewritten_question": "What rank r did the LoRA paper use?",
                                     "rewrite_ms": stream[0][1]["rewrite_ms"]})
    assert [r.question for r in pipeline.requests] == ["What is LoRA?", "What rank r did the LoRA paper use?"]
    assert stream[-1][1]["rewritten_question"] == "What rank r did the LoRA paper use?"

    assert listed == [{**listed[0], "session_id": "tab-1234abcd", "title": "What is LoRA?", "num_messages": 4}]
    roles = [(m["role"], m["content"]) for m in detail["messages"]]
    assert roles == [("user", "What is LoRA?"), ("assistant", "LoRA freezes the weights [1]."),
                     ("user", "What rank did they use?"), ("assistant", "LoRA freezes the weights [1].")]
    assert detail["messages"][2]["rewritten_question"] == "What rank r did the LoRA paper use?"
    assert [s["rank"] for s in detail["messages"][1]["sources"]] == [1, 2, 3]

    assert cleared.status_code == 204 and clear_again.status_code == 204
    assert gone.status_code == 404 and gone.json()["error"]["code"] == "session_not_found"
