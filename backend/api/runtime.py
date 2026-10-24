"""The API's hold on the RAG pipeline: background loading, one request at a time, friendly errors.

Loading bge-m3, the reranker and the chat model takes 10–30 s, so the server starts at once and
loads in a thread; ``/health`` reports ``loading`` until it is done. The GPU serves one question
at a time, so requests take ``lock`` in turn. Failures are mapped to an ``ApiError`` with a code
the apps can react to and a message a user can act on; stack traces stay in the server log.
"""

import logging
import threading
from typing import Callable

import httpx

from rpab.config import CONFIG

log = logging.getLogger("rpab.api")


class ApiError(Exception):
    """An error the apps show as is: ``{"error": {"code": ..., "message": ...}}``."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message

    def body(self) -> dict:
        return {"error": {"code": self.code, "message": self.message}}


def index_not_built() -> ApiError:
    return ApiError(503, "index_not_built", "The search index has not been built. Run setup.bat, or "
                    "`python scripts/ingest.py` from the backend folder, then restart the API.")


def full_tag(model: str) -> str:
    """Ollama lists ``llama3.1`` as ``llama3.1:latest``."""
    return model if ":" in model else f"{model}:latest"


def ollama_models(config=CONFIG) -> list[str] | None:
    """Models pulled into Ollama, or None when Ollama does not answer."""
    try:
        reply = httpx.get(f"{config.ollama_base_url}/api/tags", timeout=2)
        reply.raise_for_status()
    except httpx.HTTPError:
        return None
    return [m["name"] for m in reply.json().get("models", [])]


def diagnose(exc: Exception, config=CONFIG) -> ApiError:
    """Turn a failure while answering into the error the user should see."""
    if isinstance(exc, ApiError):
        return exc
    from rpab.embeddings import GeminiQuotaExhausted

    if isinstance(exc, GeminiQuotaExhausted):
        return ApiError(503, "embedding_quota", "The Gemini embedding quota for today is used up. Switch to the "
                        "local embeddings (RPAB_EMBEDDING=open) or try again after the daily reset.")
    models = ollama_models(config)
    if models is None:
        return ApiError(503, "ollama_unavailable", "Ollama is not running. Start the Ollama app (or run "
                        "`ollama serve`) and ask again.")
    if full_tag(config.llm_model) not in models:
        return ApiError(503, "model_missing", f"The model {config.llm_model} is not available in Ollama. "
                        f"Run `ollama pull {config.llm_model}` and ask again.")
    log.exception("Unexpected error while answering", exc_info=exc)
    return ApiError(500, "internal_error", "Something went wrong while answering. The API log has the details.")


class Runtime:
    """Owns the pipeline. ``state`` is ``loading``, ``ready`` or ``error``."""

    def __init__(self, config=CONFIG, loader: Callable | None = None):
        self.config = config
        self.loader = loader
        self.state, self.error, self.pipeline = "loading", None, None
        self.lock = threading.Lock()

    def start(self, background: bool = True) -> None:
        if background:
            threading.Thread(target=self.load, name="pipeline-loader", daemon=True).start()
        else:
            self.load()

    def load(self) -> None:
        try:
            if not (self.config.paths.chunks.exists() and self.config.paths.chroma_dir.exists()):
                raise index_not_built()
            loader = self.loader
            if loader is None:
                from rpab.rag import load_pipeline as loader
            try:
                pipeline = loader(self.config)
            except Exception as exc:
                from chromadb.errors import NotFoundError

                if isinstance(exc, (NotFoundError, FileNotFoundError)):
                    raise index_not_built() from exc
                raise
            pipeline.warm_up(llm=False)
            self.pipeline, self.state = pipeline, "ready"
            log.info("Pipeline ready (%s)", pipeline.model_name)
        except Exception as exc:
            self.error = exc if isinstance(exc, ApiError) else ApiError(
                503, "load_failed", f"The answer pipeline failed to load: {exc.__class__.__name__}. "
                "The API log has the details.")
            self.state = "error"
            log.error("Pipeline failed to load: %s", self.error.message, exc_info=exc)
            return
        # Loading the chat model into memory is worth doing early, but Ollama may start after the API
        try:
            with self.lock:
                pipeline.warm_up(retrieval=False)
        except Exception as exc:
            log.warning("Chat model warm-up skipped: %s", diagnose(exc, self.config).message)

    def ready_pipeline(self):
        """The loaded pipeline, or the ``ApiError`` that explains why there is none yet."""
        if self.state == "ready":
            return self.pipeline
        if self.state == "loading":
            raise ApiError(503, "warming_up", "The models are still loading (about 30 seconds after start). "
                           "Ask again in a moment.")
        raise self.error
