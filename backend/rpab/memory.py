"""Stretch goal S1: conversational memory over the RAG pipeline.

- ``ChatStore`` keeps sessions, messages and their sources in SQLite (doc 05, section 7);
  deleting a session cascades to its messages and sources.
- ``QuestionRewriter`` condenses a follow-up ("What rank did they use?") into a standalone
  question with the chat model, from the last ``memory_turns`` messages.
- ``Conversation`` wraps ``RagPipeline``: it rewrites, retrieves and answers with the rewritten
  question only (never the raw follow-up, doc 03 section 5), and saves the turn.

Sessions are keyed by an id the client chooses, one per browser tab, so users never share memory.
"""

import re
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser

from rpab.config import CONFIG
from rpab.prompts import CONDENSE_PROMPT
from rpab.rag import CITATION_WITH_SPACE, RagPipeline
from rpab.schemas import AskRequest, AskResponse, ChatMessage, SessionDetail, SessionSummary, Source

TITLE_CHARS = 60
HISTORY_CHARS = 600  # per message shown to the rewriter; long answers add little context

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    last_active  TEXT NOT NULL,
    title        TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    message_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    role         TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content      TEXT NOT NULL,
    rewritten_question TEXT,
    refused      INTEGER NOT NULL DEFAULT 0,
    strategy     TEXT,
    latency_ms   INTEGER,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS message_sources (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id   INTEGER NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
    rank         INTEGER NOT NULL CHECK (rank BETWEEN 1 AND 3),
    chunk_id     TEXT,
    paper_title  TEXT NOT NULL,
    page         INTEGER,
    snippet      TEXT NOT NULL,
    score        REAL,
    origin       TEXT NOT NULL DEFAULT 'paper' CHECK (origin IN ('paper', 'web')),
    url          TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_sources_message ON message_sources(message_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ChatStore:
    """SQLite chat history. Each call opens its own connection, so the store is safe across threads."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")  # off by default in SQLite; the cascades need it
        return db

    def list_sessions(self) -> list[SessionSummary]:
        with closing(self._connect()) as db:
            rows = db.execute("""SELECT s.*, COUNT(m.message_id) AS num_messages FROM sessions s
                                 LEFT JOIN messages m ON m.session_id = s.session_id
                                 GROUP BY s.session_id ORDER BY s.last_active DESC""").fetchall()
        return [SessionSummary(**dict(r)) for r in rows]

    def get_session(self, session_id: str) -> SessionDetail | None:
        with closing(self._connect()) as db:
            session = db.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if session is None:
                return None
            messages = db.execute("SELECT * FROM messages WHERE session_id = ? ORDER BY message_id",
                                  (session_id,)).fetchall()
            sources: dict[int, list[Source]] = {}
            for s in db.execute("""SELECT ms.* FROM message_sources ms JOIN messages m USING (message_id)
                                   WHERE m.session_id = ? ORDER BY ms.message_id, ms.rank""", (session_id,)):
                sources.setdefault(s["message_id"], []).append(Source(
                    rank=s["rank"], paper_title=s["paper_title"], page=s["page"], snippet=s["snippet"],
                    full_text=s["snippet"], score=s["score"], origin=s["origin"], url=s["url"],
                    chunk_id=s["chunk_id"]))
        return SessionDetail(**dict(session), num_messages=len(messages), messages=[
            ChatMessage(**{**dict(m), "refused": bool(m["refused"])}, sources=sources.get(m["message_id"], []))
            for m in messages])

    def recent_messages(self, session_id: str, n: int) -> list[tuple[str, str]]:
        """The last ``n`` messages as (role, content), oldest first. User turns show the standalone
        question they were rewritten to, so earlier follow-ups do not add unresolved references."""
        with closing(self._connect()) as db:
            rows = db.execute("SELECT role, COALESCE(rewritten_question, content) AS content FROM messages "
                              "WHERE session_id = ? ORDER BY message_id DESC LIMIT ?", (session_id, n)).fetchall()
        return [(r["role"], r["content"]) for r in reversed(rows)]

    def add_turn(self, session_id: str, question: str, rewritten: str | None, response: AskResponse) -> None:
        """Save a question and its answer, creating the session on its first question."""
        now = _now()
        with closing(self._connect()) as db, db:
            db.execute("INSERT INTO sessions (session_id, created_at, last_active, title) VALUES (?, ?, ?, ?) "
                       "ON CONFLICT (session_id) DO UPDATE SET last_active = excluded.last_active",
                       (session_id, now, now, question[:TITLE_CHARS]))
            db.execute("INSERT INTO messages (session_id, role, content, rewritten_question, created_at) "
                       "VALUES (?, 'user', ?, ?, ?)", (session_id, question, rewritten, now))
            cursor = db.execute("INSERT INTO messages (session_id, role, content, refused, strategy, latency_ms, "
                                "created_at) VALUES (?, 'assistant', ?, ?, ?, ?, ?)",
                                (session_id, response.answer, int(response.refused), response.strategy_used,
                                 response.latency_ms, now))
            db.executemany("INSERT INTO message_sources (message_id, rank, chunk_id, paper_title, page, snippet, "
                           "score, origin, url) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           [(cursor.lastrowid, s.rank, s.chunk_id, s.paper_title, s.page, s.snippet, s.score,
                             s.origin, s.url) for s in response.sources[:3]])

    def delete_session(self, session_id: str) -> bool:
        """Clear history: the cascade removes the session's messages and sources."""
        with closing(self._connect()) as db, db:
            return db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,)).rowcount > 0


class QuestionRewriter:
    """Condenses a follow-up into a standalone question. With no history, the question is used as is."""

    def __init__(self, llm):
        self.chain = CONDENSE_PROMPT | llm | StrOutputParser()

    @staticmethod
    def format_history(history: list[tuple[str, str]]) -> str:
        """The history in two labelled blocks, so a reference such as "it" is read against the latest
        topic, not the first one in the window: the earlier questions alone (their answers' wording
        pulled references towards old topics), then the most recent question and answer in full."""
        lines = []
        for role, content in history:
            text = " ".join(CITATION_WITH_SPACE.sub("", content).split())
            if len(text) > HISTORY_CHARS:
                text = text[:HISTORY_CHARS].rsplit(" ", 1)[0] + " …"
            lines.append((role, f"{'User' if role == 'user' else 'Assistant'}: {text}"))
        users = [i for i, (role, _) in enumerate(history) if role == "user"]
        split = users[-1] if users else 0
        earlier = [line for role, line in lines[:split] if role == "user"]
        recent = [line for _, line in lines[split:]]
        blocks = (["Earlier in the conversation:\n" + "\n".join(earlier)] if earlier else []) + \
                 ["Most recent exchange:\n" + "\n".join(recent)]
        return "\n\n".join(blocks)

    @staticmethod
    def clean(reply: str, question: str) -> str:
        """First line of the reply without labels or quotes; the original question if that looks wrong."""
        line = next((x.strip() for x in reply.splitlines() if x.strip()), "")
        line = re.sub(r"^(?:standalone question|question)\s*:\s*", "", line, flags=re.I).strip().strip('"“”')
        return line if 3 <= len(line) <= 1000 else question

    def rewrite(self, history: list[tuple[str, str]], question: str) -> str:
        if not history:
            return question
        reply = self.chain.invoke({"history": self.format_history(history), "question": question})
        return self.clean(reply, question)


class Conversation:
    """The pipeline with memory. A request without ``session_id`` is answered on its own."""

    def __init__(self, pipeline: RagPipeline, store: ChatStore, config=CONFIG,
                 rewriter: QuestionRewriter | None = None, chunk_text: Callable[[str], str | None] | None = None):
        self.pipeline, self.store, self.config = pipeline, store, config
        self.rewriter = rewriter or QuestionRewriter(pipeline.llm)
        self.chunk_text = chunk_text or self._chunk_text_from_retriever

    def _chunk_text_from_retriever(self, chunk_id: str) -> str | None:
        chunks = getattr(getattr(self.pipeline, "retriever", None), "chunks", None)
        return chunks.at[chunk_id, "text"] if chunks is not None and chunk_id in chunks.index else None

    def stream(self, request: AskRequest) -> Iterator[dict]:
        """``RagPipeline.stream``'s events, preceded by ``{"event": "rewrite", ...}`` for a session."""
        if request.session_id is None:
            yield from self.pipeline.stream(request)
            return
        start = time.perf_counter()
        history = self.store.recent_messages(request.session_id, self.config.memory_turns)
        standalone = self.rewriter.rewrite(history, request.question)
        rewrite_ms = round((time.perf_counter() - start) * 1000) if history else None
        yield {"event": "rewrite", "rewritten_question": standalone, "rewrite_ms": rewrite_ms}
        for event in self.pipeline.stream(request.model_copy(update={"question": standalone})):
            if event["event"] == "done":
                done, shift = event["response"], rewrite_ms or 0  # latencies count from the request
                response = done.model_copy(update={
                    "rewritten_question": standalone, "rewrite_ms": rewrite_ms, "latency_ms": done.latency_ms + shift,
                    "first_token_ms": None if done.first_token_ms is None else done.first_token_ms + shift})
                self.store.add_turn(request.session_id, request.question, standalone, response)
                event = {"event": "done", "response": response}
            yield event

    def answer(self, request: AskRequest) -> AskResponse:
        for event in self.stream(request):
            if event["event"] == "done":
                return event["response"]
        raise RuntimeError("The answer stream ended without a response")

    def session(self, session_id: str) -> SessionDetail | None:
        """A stored session, with each source's full passage restored from the chunk table."""
        detail = self.store.get_session(session_id)
        if detail is None:
            return None
        for message in detail.messages:
            for source in message.sources:
                text = self.chunk_text(source.chunk_id) if source.chunk_id else None
                if text:
                    source.full_text = text
        return detail
