"""Chat history store and follow-up rewriter (stretch S1), without models."""

import sqlite3

from langchain_core.runnables import RunnableLambda

from rpab.memory import ChatStore, QuestionRewriter
from rpab.schemas import AskResponse, Source


def response(answer: str = "LoRA freezes the weights [1].") -> AskResponse:
    sources = [Source(rank=i, paper_title="LoRA", page=4, snippet="s", full_text="s", score=0.9,
                      chunk_id=f"lora_p004_c00{i}") for i in (1, 2, 3)]
    return AskResponse(answer=answer, sources=sources, strategy_used="hybrid_rerank", latency_ms=10)


def test_recent_messages_are_the_last_n_oldest_first(tmp_path):
    store = ChatStore(tmp_path / "chat.db")
    for i in range(4):
        store.add_turn("session-1", f"question {i}", None, response(f"answer {i}"))
    assert store.recent_messages("session-1", 6) == [
        ("user", "question 1"), ("assistant", "answer 1"), ("user", "question 2"),
        ("assistant", "answer 2"), ("user", "question 3"), ("assistant", "answer 3")]
    assert store.recent_messages("other-session", 6) == []


def test_delete_cascades_to_messages_and_sources(tmp_path):
    store = ChatStore(tmp_path / "chat.db")
    store.add_turn("session-1", "What is LoRA?", None, response())
    store.add_turn("session-2", "What is QLoRA?", None, response())
    assert store.delete_session("session-1") and not store.delete_session("session-1")
    with sqlite3.connect(tmp_path / "chat.db") as db:
        counts = [db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("sessions", "messages", "message_sources")]
    assert counts == [1, 2, 3]  # only session-2 is left: 2 messages, 3 sources


def test_rewriter_skips_the_llm_without_history_and_cleans_its_reply():
    calls = []
    rewriter = QuestionRewriter(RunnableLambda(lambda prompt: calls.append(prompt) or
                                               'Standalone question: "What rank r did the LoRA paper use?"\nExtra'))
    assert rewriter.rewrite([], "What is LoRA?") == "What is LoRA?" and calls == []
    history = [("user", "What is LoRA?"), ("assistant", "LoRA freezes the weights [1].")]
    assert rewriter.rewrite(history, "What rank did they use?") == "What rank r did the LoRA paper use?"
    # Citation markers are not context for the rewriter
    assert "[1]" not in calls[0].to_string() and "LoRA freezes the weights" in calls[0].to_string()


def test_rewriter_keeps_the_question_when_the_reply_is_unusable():
    assert QuestionRewriter.clean("", "What rank did they use?") == "What rank did they use?"
    assert QuestionRewriter.clean("x" * 2000, "What rank did they use?") == "What rank did they use?"


def test_history_shows_follow_ups_in_their_standalone_form(tmp_path):
    store = ChatStore(tmp_path / "chat.db")
    store.add_turn("session-1", "What is LoRA?", "What is LoRA?", response())
    store.add_turn("session-1", "How does QLoRA build on it?", "How does QLoRA build on LoRA?", response())
    assert [content for role, content in store.recent_messages("session-1", 6) if role == "user"] == [
        "What is LoRA?", "How does QLoRA build on LoRA?"]
    # The raw question is what the user typed, and what the chat shows
    assert store.get_session("session-1").messages[2].content == "How does QLoRA build on it?"
