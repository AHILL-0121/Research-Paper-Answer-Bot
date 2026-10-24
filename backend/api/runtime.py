"""The API's hold on the RAG pipeline: background loading, one request at a time, friendly errors.

Loading bge-m3, the reranker and the chat model takes 10–30 s, so the server starts at once and
loads in a thread; ``/health`` reports ``loading`` until it is done. The GPU serves one question
at a time, so requests take ``lock`` in turn. Failures are mapped to an ``ApiError`` with a code
the apps can react to and a message a user can act on; stack traces stay in the server log.
"""

import logging
import threading
from typing import Callable

from rpab.config import CONFIG
from rpab.errors import ApiError, diagnose, index_not_built
from rpab.memory import ChatStore, Conversation

log = logging.getLogger("rpab.api")


class Runtime:
    """Owns the pipeline and the chat history. ``state`` is ``loading``, ``ready`` or ``error``;
    the history is usable at once, without waiting for the models."""

    def __init__(self, config=CONFIG, loader: Callable | None = None):
        self.config = config
        self.loader = loader
        self.state, self.error, self.pipeline, self.chat = "loading", None, None, None
        self.store = ChatStore(config.paths.chat_db)
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
            self.pipeline, self.chat = pipeline, Conversation(pipeline, self.store, self.config)
            self.state = "ready"
            log.info("Pipeline ready (%s)", pipeline.model_name)
        except Exception as exc:
            self.error = exc if isinstance(exc, ApiError) else ApiError(
                503, "load_failed", f"The answer pipeline failed to load: {exc.__class__.__name__}. "
                "The server log has the details.")
            self.state = "error"
            log.error("Pipeline failed to load: %s", self.error.message, exc_info=exc)
            return
        # Loading the chat model into memory is worth doing early, but Ollama may start after the API
        try:
            with self.lock:
                pipeline.warm_up(retrieval=False)
        except Exception as exc:
            log.warning("Chat model warm-up skipped: %s", diagnose(exc, self.config).message)

    def ready_chat(self) -> Conversation:
        """The pipeline with memory, or the ``ApiError`` that explains why it is not loaded yet."""
        if self.state == "ready":
            return self.chat
        if self.state == "loading":
            raise ApiError(503, "warming_up", "The models are still loading (about 30 seconds after start). "
                           "Ask again in a moment.")
        raise self.error
