"""FastAPI service over the RAG pipeline, used by the Next.js app.

Run from the backend folder:

    .venv\\Scripts\\uvicorn api.main:app --port 8000

Endpoints: ``GET /health``, ``GET /papers``, ``POST /ask`` (one JSON answer),
``POST /ask/stream`` (Server-Sent Events: ``rewrite`` when the request has a session, ``sources``,
then ``token`` pieces, then ``done`` with the full ``AskResponse``, or ``error``) and the chat
history: ``GET /sessions``, ``GET /sessions/{id}``, ``DELETE /sessions/{id}`` (stretch S1). Requests and responses use the same Pydantic
models as ``rpab.rag.answer``. Errors have the shape ``{"error": {"code": ..., "message": ...}}``.
"""

import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import Depends, FastAPI, Path, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel

from api.runtime import ApiError, Runtime, diagnose, full_tag, ollama_models
from rpab.config import CONFIG
from rpab.papers import list_papers, unknown_paper_ids
from rpab.schemas import AskRequest, AskResponse, PaperInfo, SessionDetail, SessionSummary

CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]  # the Next.js dev server
LOCK_TIMEOUT_S = 120  # a question waits this long for the one before it, then gets "busy"
STRATEGIES = ["dense", "mmr", "hybrid", "hybrid_rerank", "multi_query"]
SESSION_ID = Path(pattern=r"^[A-Za-z0-9_-]{8,64}$")


class ErrorInfo(BaseModel):
    code: str
    message: str


class Health(BaseModel):
    status: Literal["loading", "ready", "error"]
    error: Optional[ErrorInfo] = None  # why the pipeline is not ready
    ollama_running: bool
    llm_model: str
    llm_available: bool  # pulled into Ollama
    embedding_model: str
    reranker: bool
    default_strategy: str
    strategies: list[str]
    profile: str
    num_papers: int


def create_app(runtime: Runtime | None = None, background_load: bool = True) -> FastAPI:
    runtime = runtime or Runtime(CONFIG)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime.start(background=background_load)
        yield

    app = FastAPI(title="Research Paper Answer Bot API", version="0.5", lifespan=lifespan)
    app.state.runtime = runtime
    app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST", "DELETE"],
                       allow_headers=["Content-Type"])

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError):
        return JSONResponse(exc.body(), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        first = exc.errors()[0]
        field = ".".join(str(part) for part in first["loc"] if part != "body")
        return JSONResponse(ApiError(422, "invalid_request", f"{field}: {first['msg']}").body(), status_code=422)

    def pipeline():
        return runtime.ready_chat()

    def checked(request: AskRequest) -> AskRequest:
        unknown = unknown_paper_ids(request.paper_ids, runtime.config)
        if unknown:
            raise ApiError(422, "unknown_paper", f"No ingested paper has the id {', '.join(unknown)}. "
                           "GET /papers lists the ids.")
        return request

    def acquire() -> None:
        if not runtime.lock.acquire(timeout=LOCK_TIMEOUT_S):
            raise ApiError(503, "busy", "Another question is still being answered. Try again shortly.")

    @app.get("/health", response_model=Health)
    def health() -> Health:
        config = runtime.config
        models = ollama_models(config)
        error = runtime.error if runtime.state == "error" else None
        return Health(
            status=runtime.state,
            error=ErrorInfo(code=error.code, message=error.message) if error else None,
            ollama_running=models is not None,
            llm_model=config.llm_model,
            llm_available=models is not None and full_tag(config.llm_model) in models,
            embedding_model=config.embedding_models[config.embedding_choice],
            reranker=config.use_reranker,
            default_strategy=config.retrieval_strategy if config.use_reranker else "hybrid",
            strategies=STRATEGIES,
            profile=config.profile,
            num_papers=len(list_papers(config)),
        )

    @app.get("/papers", response_model=list[PaperInfo])
    def papers() -> list[PaperInfo]:
        return list_papers(runtime.config)

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest = Depends(checked), rag=Depends(pipeline)) -> AskResponse:
        acquire()
        try:
            return rag.answer(request)
        except Exception as exc:
            raise diagnose(exc, runtime.config) from exc
        finally:
            runtime.lock.release()

    @app.post("/ask/stream", response_class=EventSourceResponse)
    def ask_stream(request: AskRequest = Depends(checked), rag=Depends(pipeline)) -> Iterator[ServerSentEvent]:
        # The generator runs after the response has started, so even "busy" arrives as an error event
        try:
            acquire()
        except ApiError as exc:
            yield ServerSentEvent(event="error", data=exc.body()["error"])
            return
        try:
            for event in rag.stream(request):
                kind = event.pop("event")
                data = event["response"] if kind == "done" else event
                yield ServerSentEvent(event=kind, data=data)
        except Exception as exc:
            yield ServerSentEvent(event="error", data=diagnose(exc, runtime.config).body()["error"])
        finally:
            runtime.lock.release()

    @app.get("/sessions", response_model=list[SessionSummary])
    def sessions() -> list[SessionSummary]:
        return runtime.store.list_sessions()

    @app.get("/sessions/{session_id}", response_model=SessionDetail)
    def session(session_id: str = SESSION_ID) -> SessionDetail:
        # Full passages come from the chunk table once the pipeline has loaded; snippets until then
        detail = runtime.chat.session(session_id) if runtime.chat else runtime.store.get_session(session_id)
        if detail is None:
            raise ApiError(404, "session_not_found", "This chat has no saved messages.")
        return detail

    @app.delete("/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str = SESSION_ID) -> Response:
        runtime.store.delete_session(session_id)  # clearing an empty chat is not an error
        return Response(status_code=204)

    return app


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per Hugging Face or Ollama request otherwise
app = create_app()
