"""API contract tests over a fake pipeline: no GPU, models or Ollama needed.

Run from the backend folder: ``.venv\\Scripts\\python -m pytest``
"""

import json
import threading
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.runtime import Runtime
from rpab.config import CONFIG
from rpab.schemas import AskRequest, AskResponse, Source

DEAD_OLLAMA = "http://127.0.0.1:9"  # nothing listens on the discard port

SOURCES = [Source(rank=i, paper_title="LoRA", page=4, snippet="Low-rank adaptation", full_text="Low-rank adaptation",
                  score=0.9, chunk_id=f"lora_p004_c00{i}", paper_id="lora") for i in (1, 2, 3)]


class FakePipeline:
    model_name = "fake"

    def __init__(self, fail: Exception | None = None):
        self.fail = fail
        self.requests: list[AskRequest] = []

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
                                                        strategy_used=request.strategy, latency_ms=10)}

    def answer(self, request: AskRequest) -> AskResponse:
        for event in self.stream(request):
            if event["event"] == "done":
                return event["response"]


def client_for(pipeline=None, config=CONFIG, loader=None) -> TestClient:
    runtime = Runtime(config, loader=loader or (lambda cfg: pipeline or FakePipeline()))
    return TestClient(create_app(runtime, background_load=False))


def sse_events(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_ask_returns_the_pipeline_answer():
    pipeline = FakePipeline()
    with client_for(pipeline) as client:
        reply = client.post("/ask", json={"question": "What does LoRA freeze?", "paper_ids": ["lora"]})
    assert reply.status_code == 200
    body = reply.json()
    assert body["answer"] == "LoRA freezes the weights [1]."
    assert [s["rank"] for s in body["sources"]] == [1, 2, 3]
    assert pipeline.requests[0].paper_ids == ["lora"]


def test_stream_sends_sources_tokens_then_done():
    with client_for() as client:
        reply = client.post("/ask/stream", json={"question": "What does LoRA freeze?"})
    assert reply.headers["content-type"].startswith("text/event-stream")
    events = sse_events(reply.text)
    assert [kind for kind, _ in events] == ["sources", "token", "token", "done"]
    assert len(events[0][1]["sources"]) == 3
    assert "".join(data["text"] for kind, data in events if kind == "token") == events[-1][1]["answer"]


def test_validation_errors_are_friendly():
    with client_for() as client:
        short = client.post("/ask", json={"question": "hi"})
        unknown = client.post("/ask", json={"question": "What does LoRA freeze?", "paper_ids": ["nope"]})
    assert short.status_code == 422 and short.json()["error"]["code"] == "invalid_request"
    assert "question" in short.json()["error"]["message"]
    assert unknown.status_code == 422 and unknown.json()["error"]["code"] == "unknown_paper"


def test_missing_index_is_reported(tmp_path):
    config = replace(CONFIG, paths=replace(CONFIG.paths, chunks=tmp_path / "chunks.parquet"))
    with client_for(config=config) as client:
        health = client.get("/health").json()
        reply = client.post("/ask", json={"question": "What does LoRA freeze?"})
    assert health["status"] == "error" and health["error"]["code"] == "index_not_built"
    assert reply.status_code == 503 and reply.json()["error"]["code"] == "index_not_built"


def test_ollama_down_is_reported():
    config = replace(CONFIG, ollama_base_url=DEAD_OLLAMA)
    pipeline = FakePipeline(fail=ConnectionError("Failed to connect to Ollama"))
    with client_for(pipeline, config=config) as client:
        reply = client.post("/ask", json={"question": "What does LoRA freeze?"})
        stream = client.post("/ask/stream", json={"question": "What does LoRA freeze?"})
        health = client.get("/health").json()
    assert reply.status_code == 503 and reply.json()["error"]["code"] == "ollama_unavailable"
    assert sse_events(stream.text)[-1] == ("error", reply.json()["error"])
    assert health["ollama_running"] is False and health["status"] == "ready"


def test_unexpected_errors_hide_the_trace():
    pipeline = FakePipeline(fail=KeyError("secret internals"))
    config = replace(CONFIG, llm_model="present:latest")
    with client_for(pipeline, config=config) as client, \
            pytest.MonkeyPatch.context() as patch:
        patch.setattr("api.runtime.ollama_models", lambda cfg: ["present:latest"])
        reply = client.post("/ask", json={"question": "What does LoRA freeze?"})
    assert reply.status_code == 500
    assert reply.json()["error"]["code"] == "internal_error" and "secret" not in reply.text


def test_requests_wait_while_models_load():
    release = threading.Event()

    def slow_loader(cfg):
        release.wait(5)
        return FakePipeline()

    runtime = Runtime(CONFIG, loader=slow_loader)
    with TestClient(create_app(runtime, background_load=True)) as client:
        loading = client.post("/ask", json={"question": "What does LoRA freeze?"})
        assert client.get("/health").json()["status"] == "loading"
        release.set()
    assert loading.status_code == 503 and loading.json()["error"]["code"] == "warming_up"


def test_papers_and_cors():
    with client_for() as client:
        papers = client.get("/papers").json()
        preflight = client.options("/ask", headers={"Origin": "http://localhost:3000",
                                                    "Access-Control-Request-Method": "POST"})
    assert {"paper_id", "paper_title", "num_chunks"} <= papers[0].keys()
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:3000"

