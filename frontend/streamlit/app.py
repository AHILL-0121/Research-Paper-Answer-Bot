"""Research Paper Answer Bot: the Streamlit app (stretch goal S2), following plan/SampleUI.html.

Three panes as in the sample: the sidebar is the rail (new chat, recent chats, settings), the main
area holds the conversation and a reader with the cited passage or the paper library. Answers
stream from the same ``Conversation`` the API uses, so follow-ups are rewritten (S1) and chats
are saved to ``backend/artifacts/chat.db``.

Run from this folder (stop the API first: both load the models onto the same GPU):

    ..\\..\\backend\\.venv\\Scripts\\streamlit run app.py
"""

import sys
import threading
import time
import uuid
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import streamlit as st  # noqa: E402

import ui  # noqa: E402
from rpab.config import CONFIG  # noqa: E402
from rpab.errors import diagnose, index_not_built  # noqa: E402
from rpab.memory import ChatStore, Conversation  # noqa: E402
from rpab.papers import list_papers  # noqa: E402
from rpab.schemas import AskRequest, AskResponse  # noqa: E402

EXAMPLES = [  # empty state: one question per paper family, as in the sample, with a short paper label
    ("What is the main idea behind the Transformer architecture?", "Attention Is All You Need"),
    ("How does LoRA reduce the number of trainable parameters?", "LoRA"),
    ("What is retrieval-augmented generation and why was it proposed?", "RAG"),
    ("How does chain-of-thought prompting improve reasoning?", "Chain-of-Thought"),
]
DEMO = [  # doc 03 section 10: lookup, method detail, comparison, out of scope, follow-up
    "Which retriever and which generator does the RAG model combine?",
    "How are the LoRA matrices A and B initialised, and why?",
    "How do LoRA and QLoRA differ in how they keep the frozen pretrained weights?",
    "How does Mamba's selective state space model replace attention?",
    "What data type does it use for the frozen weights?",
]
DEFAULT_STRATEGY = CONFIG.retrieval_strategy if CONFIG.use_reranker else "hybrid"
EMBEDDING_SLUGS = {"open": "bge_m3", "commercial": "gemini_embedding_001"}

st.set_page_config(page_title="Research Paper Answer Bot", page_icon=":material/menu_book:", layout="wide",
                   initial_sidebar_state="expanded")
ss = st.session_state


# ---- resources ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def store() -> ChatStore:
    return ChatStore(CONFIG.paths.chat_db)


@st.cache_resource(show_spinner="Loading the models, about 20 seconds the first time…")
def engine() -> tuple[Conversation, threading.Lock]:
    """The pipeline with memory, loaded once per server process, and the lock that keeps the GPU
    to one question at a time across browser tabs."""
    from chromadb.errors import NotFoundError

    from rpab.rag import load_pipeline

    try:
        pipeline = load_pipeline(CONFIG)
    except (NotFoundError, FileNotFoundError) as exc:
        raise index_not_built() from exc
    pipeline.warm_up(llm=False)
    try:
        pipeline.warm_up(retrieval=False)  # loads the chat model; Ollama may not be running yet
    except Exception:
        pass
    return Conversation(pipeline, store(), CONFIG), threading.Lock()


@st.cache_data(ttl=300, show_spinner=False)
def papers():
    return list_papers(CONFIG)


def index_ready() -> bool:
    return CONFIG.paths.chunks.exists() and CONFIG.paths.chroma_dir.exists() and bool(papers())


# ---- state and actions -------------------------------------------------------------------

def new_session_id() -> str:
    return f"tab-{uuid.uuid4().hex[:12]}"


def init_state() -> None:
    ss.setdefault("session_id", new_session_id())
    ss.setdefault("turns", [])  # {"question", "response": AskResponse | None, "error": dict | None, "scope"}
    ss.setdefault("open", None)  # (turn index, source rank) shown in the reader
    ss.setdefault("reader_tab", "Library")
    ss.setdefault("strategy", DEFAULT_STRATEGY)
    ss.setdefault("debug", False)
    ss.setdefault("pending", None)
    for p in papers():
        ss.setdefault(f"paper-{p.paper_id}", True)


def selected_papers() -> list[str]:
    return [p.paper_id for p in papers() if ss.get(f"paper-{p.paper_id}", True)]


def scope_label(selected: list[str]) -> str:
    total = len(papers())
    return f"all {total} papers" if len(selected) == total else f"{len(selected)} of {total} papers"


def ask(question: str) -> None:
    ss.pending = question


def reset_chat() -> None:
    ss.session_id, ss.turns, ss.open, ss.reader_tab, ss.pending = new_session_id(), [], None, "Library", None


def new_chat() -> None:
    reset_chat()
    st.toast("New chat started")


def clear_history() -> None:
    store().delete_session(ss.session_id)
    reset_chat()
    st.toast("History cleared for this chat")


def load_session(session_id: str) -> None:
    conversation = engine_if_loaded()
    detail = conversation.session(session_id) if conversation else store().get_session(session_id)
    if detail is None:
        return
    turns, messages = [], detail.messages
    for user, assistant in zip(messages[::2], messages[1::2]):
        response = AskResponse(answer=assistant.content, sources=assistant.sources, refused=assistant.refused,
                               rewritten_question=user.rewritten_question,
                               strategy_used=assistant.strategy or DEFAULT_STRATEGY,
                               latency_ms=assistant.latency_ms or 0)
        turns.append({"question": user.content, "response": response, "error": None, "scope": "the selected papers"})
    ss.session_id, ss.turns, ss.open, ss.reader_tab, ss.pending = session_id, turns, None, "Library", None


def engine_if_loaded() -> Conversation | None:
    """The loaded conversation, without triggering a model load."""
    return ss.get("_engine")


def open_source(turn: int, rank: int) -> None:
    ss.open, ss.reader_tab = (turn, rank), "Passage"


def step_source(delta: int) -> None:
    turn, rank = ss.open
    ss.open = (turn, rank + delta)


def retry(turn: int) -> None:
    question = ss.turns[turn]["question"]
    del ss.turns[turn]
    ss.open = None
    ask(question)


def select_all(value: bool) -> None:
    for p in papers():
        ss[f"paper-{p.paper_id}"] = value


def strategy_changed() -> None:
    st.toast(f"Next question uses {ui.STRATEGY_LABELS[ss.strategy]} retrieval")


# ---- rail --------------------------------------------------------------------------------

def render_rail():
    """The sidebar; returns the slot where the model-loading note goes."""
    with st.sidebar:
        st.html(f'<div class="wordmark"><b>Research Paper Answer Bot</b>'
                f'<span>RAG over {len(papers())} GenAI papers</span></div>')
        st.button("New chat", key="new_chat", icon=":material/add:", use_container_width=True, on_click=new_chat)

        sessions = store().list_sessions()[:8]
        if sessions:
            st.html('<div class="rail-h">Recent</div>')
            for k, s in enumerate(sessions):
                current = s.session_id == ss.session_id
                st.button(s.title or "Untitled chat", key="sess-current" if current else f"sess-{k}",
                          use_container_width=True, on_click=load_session, args=(s.session_id,),
                          help=s.title if s.title and len(s.title) > 30 else None)

        st.html('<div class="rail-h">Settings</div>')
        st.selectbox("Retrieval strategy", list(ui.STRATEGY_LABELS), key="strategy",
                     format_func=ui.STRATEGY_LABELS.get, on_change=strategy_changed,
                     help="Applies to the next question. Hybrid + rerank is the strategy chosen in the notebook.")
        st.toggle("Show retrieval details", key="debug")
        with st.expander("Demo questions"):
            for k, question in enumerate(DEMO):
                st.button(question, key=f"textbtn-demo-{k}", on_click=ask, args=(question,))

        embed = CONFIG.embedding_models[CONFIG.embedding_choice].split("/")[-1]
        st.html('<dl class="spec">'
                f'<dt>LLM</dt><dd>{ui.esc(CONFIG.llm_model)} · Ollama</dd>'
                f'<dt>Embed</dt><dd>{ui.esc(embed)}</dd>'
                f'<dt>Rerank</dt><dd>{"bge-reranker-v2-m3" if CONFIG.use_reranker else "off"}</dd>'
                f'<dt>Index</dt><dd>{ui.esc(CONFIG.chunk_spec.slug)}</dd></dl>')
        models_slot = st.empty()
        st.button("Clear history", key="textbtn-clear", on_click=clear_history,
                  help="Deletes this chat and its saved messages")
    return models_slot


# ---- conversation ------------------------------------------------------------------------

def render_empty() -> None:
    if not index_ready():
        st.html('<section class="intro"><h1>No papers indexed yet</h1><div class="setup">'
                '<strong>Run the ingestion step first.</strong><p>From the <code>backend</code> folder run '
                '<code>python scripts/ingest.py</code>. It loads the PDFs in <code>data/pdfs</code> and builds the '
                'vector and BM25 indexes. Then reload this page.</p></div></section>')
        return
    library = papers()
    pages = sum(p.num_pages or 0 for p in library)
    chunks = sum(p.num_chunks or 0 for p in library)
    st.html('<section class="intro"><h1>What do you want to find in the papers?</h1>'
            '<p class="lede">Answers are written only from the indexed papers. Every answer lists the three '
            'passages it used, with the paper title and page, so you can check the claim yourself.</p>'
            f'<div class="stats"><span><b>{len(library)}</b> papers</span><span><b>{pages}</b> pages</span>'
            f'<span><b>{chunks:,}</b> chunks</span></div>'
            '<span class="label examples-label">Try a question</span></section>')
    for k, (question, paper) in enumerate(EXAMPLES):
        with st.container(key=f"hit-ex-{k}"):
            st.html(f'<div class="ex"><span class="q">{ui.esc(question)}<span class="arrow" aria-hidden="true">→'
                    f'</span></span><span class="p">{ui.esc(paper)}</span></div>')
            st.button(f"Ask: {question}", key=f"btn-ex-{k}", on_click=ask, args=(question,))


def suggestions(turn: int) -> list[str]:
    asked = {t["question"] for t in ss.turns}
    return [q for q, _ in EXAMPLES if q not in asked][:2]


def render_turn(t: int, turn: dict) -> None:
    response: AskResponse | None = turn["response"]
    st.html(f'<h2 class="ask">{ui.esc(turn["question"])}</h2>'
            + ui.searched(response.rewritten_question if response else None, turn["question"]))
    if turn["error"]:
        err = turn["error"]
        st.html(ui.error_box(err["message"], err["details"]))
        st.button("Try again", key=f"act-retry-{t}", icon=":material/refresh:", on_click=retry, args=(t,))
        return

    if response.refused:
        with st.container(key=f"refusal-{t}"):
            st.html(ui.refusal_note(response.answer))
            chips = st.columns(2, gap="small")
            for k, question in enumerate(suggestions(t)):
                chips[k].button(question, key=f"chip-{t}-{k}", on_click=ask, args=(question,))

    if response.sources:
        st.html(ui.sources_header(response.refused))
        with st.container(key=f"srcs-{'dim-' if response.refused else ''}{t}"):
            cards = st.columns(3, gap="small")
            for i, source in enumerate(response.sources[:3]):
                with cards[i], st.container(key=f"hit-src-{t}-{source.rank}"):
                    st.html(ui.source_card(source, t, response.strategy_used, ss.open == (t, source.rank)))
                    st.button(f"Open source {source.rank} in the reader", key=f"btn-src-{t}-{source.rank}",
                              on_click=open_source, args=(t, source.rank))

    if not response.refused:
        st.html(f'<p class="prose">{ui.with_citations(response.answer, response.sources, t)}</p>')
    collection = f"papers__{EMBEDDING_SLUGS[CONFIG.embedding_choice]}__{CONFIG.chunk_spec.slug}"
    st.html(ui.trace(response, turn["scope"], CONFIG.llm_model, ss.debug, ss.session_id,
                     CONFIG.confidence_thresholds.get("rerank"), collection))
    st.button("Ask again", key=f"act-retry-{t}", icon=":material/refresh:", on_click=retry, args=(t,))


def live_html(question: str, live: dict) -> str:
    parts = [f'<h2 class="ask">{ui.esc(question)}</h2>', ui.searched(live["rewritten"], question)]
    if live["sources"] is None or not live["text"]:
        parts.append(ui.steps(live["stage"], live["details"]))
    if live["sources"]:
        cards = "".join(ui.source_card(s, -1, live["strategy"], False) for s in live["sources"][:3])
        parts.append(ui.sources_header(False)
                     + f'<div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px">{cards}</div>')
    if live["text"]:
        parts.append(f'<p class="prose">{ui.esc(live["text"])}<span class="caret"></span></p>')
    return "".join(parts)


def run_pending(question: str) -> None:
    """Answer the pending question, streaming into the conversation, then store the turn and rerun."""
    ss.pending = None
    question = " ".join(question.split())
    if len(question) < 3:
        st.toast("Type a longer question")
        return
    if len(question) > 1000:
        question = question[:1000]
        st.toast("Question shortened to 1,000 characters")
    selected = selected_papers()
    if not selected:
        st.toast("Select at least one paper in the Library tab")
        return
    scope = scope_label(selected)
    request = AskRequest(question=question, session_id=ss.session_id, strategy=ss.strategy,
                         paper_ids=None if len(selected) == len(papers()) else selected)
    live = {"stage": 0, "details": {}, "rewritten": None, "sources": None, "text": "", "strategy": ss.strategy}
    slot = st.empty()
    slot.html(live_html(question, live))
    turn = {"question": question, "response": None, "error": None, "scope": scope}
    try:
        conversation, lock = engine()
        ss._engine = conversation
        with lock:
            last_draw = 0.0
            for event in conversation.stream(request):
                kind = event["event"]
                if kind == "rewrite":
                    live["rewritten"] = event["rewritten_question"]
                elif kind == "sources":
                    live.update(sources=event["sources"], stage=2,
                                details={0: f"{scope} · 20 candidates", 1: f"{len(event['sources'])} kept"})
                elif kind == "token":
                    live["text"] += event["text"]
                    if time.perf_counter() - last_draw < 0.05:  # redraw at most 20 times a second
                        continue
                elif kind == "done":
                    turn["response"] = event["response"]
                    continue
                slot.html(live_html(question, live))
                last_draw = time.perf_counter()
    except Exception as exc:
        error = diagnose(exc, CONFIG)
        turn["error"] = {"code": error.code, "message": error.message,
                         "details": f"{exc.__class__.__name__}: {exc}"[:800]}
    ss.turns.append(turn)
    st.rerun()


# ---- reader ------------------------------------------------------------------------------

def current_source():
    if not ss.open:
        return None
    turn_index, rank = ss.open
    if turn_index >= len(ss.turns) or not ss.turns[turn_index]["response"]:
        return None
    response = ss.turns[turn_index]["response"]
    if not 1 <= rank <= len(response.sources):
        return None
    return response, response.sources[rank - 1]


def render_reader() -> None:
    with st.container(key="reader"):
        st.segmented_control("Reader", ["Passage", "Library"], key="reader_tab", label_visibility="collapsed")
        if ss.reader_tab == "Passage":
            render_passage()
        else:
            render_library()


def render_passage() -> None:
    current = current_source()
    if current is None:
        st.html('<div class="placeholder">Select a source card or a citation such as [1] '
                'to read the passage on its page.</div>')
        return
    response, source = current
    n = len(response.sources)
    pager = st.columns([4, 1, 1], gap="small", vertical_alignment="center")
    pager[0].html(f'<div class="pager">Source {source.rank} of {n}</div>')
    pager[1].button("", key="act-prev", icon=":material/chevron_left:", help="Previous source",
                    disabled=source.rank == 1, on_click=step_source, args=(-1,))
    pager[2].button("", key="act-next", icon=":material/chevron_right:", help="Next source",
                    disabled=source.rank == n, on_click=step_source, args=(1,))
    paper = next((p for p in papers() if p.paper_id == source.paper_id or p.paper_title == source.paper_title), None)
    st.html(ui.reader_page(source, paper, response.answer, response.refused))
    st.html(ui.reader_facts(source, paper))
    st.caption("Citation (copy with the icon)")
    st.code(f"{source.paper_title}, p. {source.page}", language=None, wrap_lines=True)


def render_library() -> None:
    library = papers()
    if not library:
        st.html('<div class="placeholder">No papers indexed yet. Run the ingestion step first.</div>')
        return
    top = st.columns([3, 1, 1], gap="small", vertical_alignment="center")
    top[0].html(f'<div class="lib-top">{len(selected_papers())} of {len(library)} papers searched</div>')
    top[1].button("All", key="textbtn-all", on_click=select_all, args=(True,))
    top[2].button("None", key="textbtn-none", on_click=select_all, args=(False,))
    with st.container(key="library"):
        for p in library:
            meta = " · ".join(x for x in [str(p.year or ""), f"{p.num_pages} pp" if p.num_pages else "",
                                          f"{p.num_chunks} chunks" if p.num_chunks else ""] if x)
            st.checkbox(f"{p.paper_title}  \n:small[:gray[{meta}]]", key=f"paper-{p.paper_id}")


# ---- page --------------------------------------------------------------------------------

def main() -> None:
    init_state()
    st.html(f"<style>{ui.theme_css(st.context.theme.type)}{ui.CSS}</style>")
    ready = index_ready()
    typed = st.chat_input("Ask a question about the papers...", disabled=not ready)
    if typed:
        ask(typed)

    models_slot = render_rail()

    convo, reader = st.columns([1.6, 1], gap="large")
    with convo, st.container(key="convo"):
        if ss.pending:
            st.html(ui.HIDE_STALE)
        if not ss.turns and not ss.pending:
            render_empty()
        for t, turn in enumerate(ss.turns):
            render_turn(t, turn)
        if ss.pending and ready:
            run_pending(ss.pending)
    with reader:
        render_reader()

    # Load the models after the first paint, so the page appears at once
    if ready and "_engine" not in ss:
        with models_slot:
            try:
                ss._engine = engine()[0]
            except Exception as exc:
                st.caption(f"Models not loaded: {diagnose(exc, CONFIG).message}")


main()
