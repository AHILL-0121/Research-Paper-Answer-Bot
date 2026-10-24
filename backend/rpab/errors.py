"""Errors the apps show to users: a stable code to react to and a message a user can act on.

Shared by the API and the Streamlit app. ``diagnose`` turns any failure while answering into
one of these; the stack trace goes to the log, never to the screen.
"""

import logging

import httpx

from rpab.config import CONFIG

log = logging.getLogger("rpab")


class ApiError(Exception):
    """An error the apps show as is: ``{"error": {"code": ..., "message": ...}}``."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message

    def body(self) -> dict:
        return {"error": {"code": self.code, "message": self.message}}


def index_not_built() -> ApiError:
    return ApiError(503, "index_not_built", "The search index has not been built. Run setup.bat, or "
                    "`python scripts/ingest.py` from the backend folder, then restart the app.")


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
    return ApiError(500, "internal_error", "Something went wrong while answering. The server log has the details.")
