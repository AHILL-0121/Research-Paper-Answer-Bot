"use client";

// The whole app: one client component that owns the state and talks to the FastAPI service.
// Layout and states follow plan/SampleUI.html: rail · conversation · reader.

import { useCallback, useEffect, useRef, useState } from "react";
import { Composer } from "./Composer";
import { Icon, IconSprite } from "./Icon";
import { Rail, type Theme } from "./Rail";
import { Reader, type ReaderSource } from "./Reader";
import { cardId, Turn, type TurnState, WithCode } from "./Turn";
import {
  API_URL, ApiError, askStream, deleteSession, getHealth, getPapers, getSession, listSessions, unreachable,
} from "@/lib/api";
import { newSessionId, STRATEGY_LABELS } from "@/lib/text";
import type { ChatMessage, Health, PaperInfo, SessionSummary, Source, Strategy } from "@/lib/types";

const EXAMPLES: [string, string][] = [
  ["What is the main idea behind the Transformer architecture?", "Attention Is All You Need"],
  ["How does LoRA reduce the number of trainable parameters?", "LoRA"],
  ["What is retrieval-augmented generation and why was it proposed?", "RAG"],
  ["How does chain-of-thought prompting improve reasoning?", "Chain-of-Thought"],
];
const ALL_STRATEGIES = Object.keys(STRATEGY_LABELS) as Strategy[];
const SESSION_KEY = "rpab-session"; // per browser tab (sessionStorage), so tabs never share memory

type Open = { turn: number; rank: number } | null;
type Preview = { x: number; y: number; above: boolean; source: Source } | null;

export default function AnswerBot() {
  const [health, setHealth] = useState<Health | null>(null);
  const [apiDown, setApiDown] = useState(false);
  const [papers, setPapers] = useState<PaperInfo[] | null>(null); // null until loaded
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [sessionId, setSessionId] = useState(storedSessionId);
  const [turns, setTurns] = useState<TurnState[]>([]);
  const [open, setOpen] = useState<Open>(null);
  const [tab, setTab] = useState<"passage" | "library">("library");
  const [chosenStrategy, setStrategy] = useState<Strategy | null>(null); // null: the API's default
  const [debug, setDebug] = useState(false);
  const [web, setWeb] = useState(false); // S3 web fallback, off until the user turns it on
  const [busy, setBusy] = useState(false);
  const [draft, setDraft] = useState("");
  const [theme, setTheme] = useState<Theme>(storedTheme);
  const [railOpen, setRailOpen] = useState(false);
  const [readerOpen, setReaderOpen] = useState(false);
  const [flash, setFlash] = useState<Open>(null);
  const [preview, setPreview] = useState<Preview>(null);
  const [toast, setToast] = useState<string | null>(null);

  const abort = useRef<AbortController | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const nearBottom = useRef(true);
  const toastTimer = useRef<ReturnType<typeof setTimeout>>(undefined);

  const say = useCallback((message: string) => {
    setToast(message);
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 2400);
  }, []);

  const refreshSessions = useCallback(() => {
    listSessions().then(setSessions).catch(() => {});
  }, []);

  // ---- startup: health (polled until ready), papers, saved chats, this tab's session, theme ----
  useEffect(() => {
    let stop = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const h = await getHealth();
        if (stop) return;
        setHealth(h);
        setApiDown(false);
        timer = setTimeout(poll, h.status === "loading" ? 2000 : 15000);
      } catch {
        if (stop) return;
        setApiDown(true);
        timer = setTimeout(poll, 3000);
      }
    };
    poll();
    return () => { stop = true; clearTimeout(timer); };
  }, []);

  useEffect(() => {
    if (apiDown || !health) return;
    getPapers().then((list) => {
      setPapers(list);
      setSelected((prev) => (prev.size ? prev : new Set(list.map((p) => p.paper_id))));
    }).catch(() => {});
    refreshSessions();
  }, [apiDown, health?.status, refreshSessions]); // eslint-disable-line react-hooks/exhaustive-deps

  // A reload continues this tab's chat; a new tab has nothing saved yet
  useEffect(() => {
    if (resumedSession) getSession(sessionId).then((d) => setTurns(toTurns(d.messages))).catch(() => {});
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { setRailOpen(false); setReaderOpen(false); setPreview(null); }
    };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  }, []);

  // Follow a streaming answer unless the reader has scrolled up
  useEffect(() => {
    const el = scroller.current;
    if (el && nearBottom.current) el.scrollTop = el.scrollHeight;
  }, [turns]);

  // ---- derived ----
  const strategy: Strategy = chosenStrategy ?? health?.default_strategy ?? "hybrid_rerank";
  const ready = health?.status === "ready";
  const indexMissing = (health?.status === "error" && health.error?.code === "index_not_built") ||
    (health?.status === "ready" && health.num_papers === 0);
  const library = papers ?? [];
  const allSelected = library.length > 0 && selected.size === library.length;
  const scopeLabel = allSelected || !library.length ? "All papers" : `${selected.size} of ${library.length} papers`;
  const scope = allSelected ? `all ${library.length} papers` : `${selected.size} of ${library.length} papers`;
  const chatTitle = turns[0]?.question ?? "New chat";

  const current: ReaderSource | null = (() => {
    if (!open) return null;
    const t = turns[open.turn];
    const sources = t?.response?.sources ?? t?.sources;
    const source = sources?.[open.rank - 1];
    if (!t || !source) return null;
    return { source, total: sources.length, answer: t.response?.answer ?? t.text, refused: !!t.response?.refused };
  })();

  // ---- actions ----
  const updateTurn = (id: string, patch: (t: TurnState) => Partial<TurnState>) =>
    setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, ...patch(t) } : t)));

  async function ask(raw: string) {
    let question = raw.replace(/\s+/g, " ").trim();
    if (busy) return;
    if (apiDown) return say("The API is not reachable");
    if (!ready) return say(health?.status === "loading" ? "The models are still loading" : "The API is not ready");
    if (question.length < 3) return say("Type a longer question");
    if (question.length > 1000) { question = question.slice(0, 1000); say("Question shortened to 1,000 characters"); }
    if (!selected.size) return say("Select at least one paper in the Library tab");

    const turn: TurnState = {
      id: crypto.randomUUID(), question, status: "loading", rewritten: null, sources: [], text: "", strategy, scope,
    };
    setTurns((ts) => [...ts, turn]);
    setDraft("");
    setBusy(true);
    nearBottom.current = true;
    const controller = new AbortController();
    abort.current = controller;
    let finished = false;
    try {
      await askStream(
        { question, session_id: sessionId, strategy, web_fallback: web, paper_ids: allSelected ? null : [...selected] },
        (e) => {
          if (e.event === "rewrite") updateTurn(turn.id, () => ({ rewritten: e.rewritten_question }));
          else if (e.event === "sources") updateTurn(turn.id, () => ({ sources: e.sources, webNote: e.web_note }));
          else if (e.event === "token") updateTurn(turn.id, (t) => ({ status: "streaming", text: t.text + e.text }));
          else if (e.event === "done") {
            finished = true;
            updateTurn(turn.id, () => ({ status: "done", response: e.response, sources: e.response.sources }));
          } else if (e.event === "error") {
            finished = true;
            updateTurn(turn.id, () => ({ status: "error", error: e.error }));
            setDraft(question);
          }
        },
        controller.signal,
      );
      if (!finished) throw new ApiError("stream_ended", "The answer stopped before it was complete. Try again.");
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      const e = err instanceof ApiError ? err : unreachable();
      updateTurn(turn.id, () => ({ status: "error", error: { code: e.code, message: e.message } }));
      setDraft(question);
      if (e.code === "api_unreachable") setApiDown(true);
    } finally {
      if (abort.current === controller) { abort.current = null; setBusy(false); }
      refreshSessions();
    }
  }

  function retry(index: number) {
    const question = turns[index].question;
    setTurns((ts) => ts.filter((_, i) => i !== index));
    setOpen(null);
    void ask(question);
  }

  function openSource(turn: number, rank: number) {
    setOpen({ turn, rank });
    setTab("passage");
    setPreview(null);
    if (matchMedia("(max-width: 1180px)").matches) setReaderOpen(true);
  }

  function cite(turn: number, rank: number) {
    openSource(turn, rank);
    document.getElementById(cardId(turn, rank))?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    setFlash({ turn, rank });
    setTimeout(() => setFlash(null), 1300);
  }

  function showPreview(el: HTMLElement | null, source?: Source) {
    if (!el || !source) return setPreview(null);
    const r = el.getBoundingClientRect();
    const above = r.top > 180;
    setPreview({ x: Math.max(16, Math.min(r.left + r.width / 2 - 160, innerWidth - 336)), y: above ? r.top - 8 : r.bottom + 8, above, source });
  }

  function startChat(message: string) {
    abort.current?.abort();
    abort.current = null;
    setBusy(false);
    const id = newSessionId();
    try { sessionStorage.setItem(SESSION_KEY, id); } catch {}
    setSessionId(id);
    setTurns([]);
    setOpen(null);
    setTab("library");
    setDraft("");
    setRailOpen(false);
    say(message);
  }

  async function clearHistory() {
    try { await deleteSession(sessionId); } catch { /* nothing saved yet, or the API is down */ }
    startChat("History cleared for this chat");
    refreshSessions();
  }

  async function loadSession(id: string) {
    try {
      const detail = await getSession(id);
      abort.current?.abort();
      setBusy(false);
      setTurns(toTurns(detail.messages));
      setSessionId(id);
      try { sessionStorage.setItem(SESSION_KEY, id); } catch {}
      setOpen(null);
      setTab("library");
      setRailOpen(false);
    } catch (e) {
      say(e instanceof ApiError ? e.message : "Could not open that chat");
    }
  }

  function changeTheme(t: Theme) {
    setTheme(t);
    try {
      if (t === "system") { localStorage.removeItem("theme"); document.documentElement.removeAttribute("data-theme"); }
      else { localStorage.setItem("theme", t); document.documentElement.setAttribute("data-theme", t); }
    } catch {}
  }

  async function copy(text: string, done: string) {
    try { await navigator.clipboard.writeText(text); say(done); } catch { say("Copying isn't available here"); }
  }

  const suggestions = (except: string) => EXAMPLES.map(([q]) => q).filter((q) => q !== except && !turns.some((t) => t.question === q)).slice(0, 2);

  // ---- render ----
  return (
    <>
      <IconSprite />
      <div className="shell">
        <Rail
          open={railOpen} health={health} numPapers={library.length} sessions={sessions} sessionId={sessionId}
          debug={debug} theme={theme} onNewChat={() => startChat("New chat started")} onLoad={loadSession}
          onDebug={setDebug} onTheme={changeTheme} onClear={clearHistory} web={web}
          onWeb={(on) => { setWeb(on); say(on ? "Web fallback on: weak matches are also sent to DuckDuckGo" : "Web fallback off"); }}
        />

        <main className="convo">
          <header className="convo-head">
            <button className="icon-btn menu-btn" aria-label="Open chats and settings" onClick={() => setRailOpen(true)}>
              <Icon name="menu" />
            </button>
            <span className="title">{chatTitle}</span>
            <span className="sid" title="Session id">{sessionId}</span>
            <button className="icon-btn reader-btn" aria-label="Open reader" onClick={() => setReaderOpen(true)}>
              <Icon name="book" />
            </button>
          </header>

          <div className="scroll" ref={scroller}
               onScroll={(e) => { const el = e.currentTarget; nearBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80; setPreview(null); }}>
            <div className="col">
              <Banners apiDown={apiDown} health={health} indexMissing={indexMissing} />
              {turns.length === 0 ? (
                <Intro apiDown={apiDown} indexMissing={indexMissing} papers={library} canAsk={ready} onAsk={ask} />
              ) : (
                turns.map((t, i) => (
                  <Turn
                    key={t.id} turn={t} index={i} debug={debug} health={health} sessionId={sessionId}
                    openRank={open?.turn === i ? open.rank : null} flashRank={flash?.turn === i ? flash.rank : null}
                    suggestions={suggestions(t.question)}
                    onOpen={(rank) => openSource(i, rank)} onCite={(rank) => cite(i, rank)} onPreview={showPreview}
                    onRetry={() => retry(i)} onAsk={ask} onCopy={copy}
                  />
                ))
              )}
            </div>
          </div>

          <div className="dock">
            <Composer
              draft={draft} disabled={busy || !ready} strategy={strategy}
              strategies={health?.strategies?.length ? health.strategies : ALL_STRATEGIES}
              scopeLabel={scopeLabel} model={health?.llm_model ?? ""}
              onDraft={setDraft} onSubmit={() => ask(draft)}
              onStrategy={(s) => { setStrategy(s); say(`Next question uses ${STRATEGY_LABELS[s]} retrieval`); }}
              onScope={() => { setTab("library"); if (matchMedia("(max-width: 1180px)").matches) setReaderOpen(true); }}
            />
            <p className="fine">
              {web
                ? "Web fallback is on: when the papers match weakly, the question is also sent to DuckDuckGo, and web sources are marked WEB."
                : "Answers use only the indexed papers. Each one lists the 3 passages it was written from."}
            </p>
          </div>
        </main>

        <Reader
          tab={tab} open={readerOpen} current={current} papers={library} loading={papers === null && !apiDown} selected={selected} indexMissing={indexMissing}
          onTab={setTab} onStep={(d) => open && setOpen({ ...open, rank: open.rank + d })}
          onToggle={(id) => setSelected((s) => { const n = new Set(s); if (n.has(id)) n.delete(id); else n.add(id); return n; })}
          onSelectAll={(all) => setSelected(all ? new Set(library.map((p) => p.paper_id)) : new Set())}
          onCopy={copy} onClose={() => setReaderOpen(false)}
        />
        <div className={`scrim ${railOpen || readerOpen ? "open" : ""}`} onClick={() => { setRailOpen(false); setReaderOpen(false); }} />
      </div>

      {preview && (
        <div className="pop" role="tooltip"
             style={{ left: preview.x, top: preview.y, transform: preview.above ? "translateY(-100%)" : undefined }}>
          <div className="h">
            <span>{preview.source.paper_title.split(":")[0]}</span>
            <span>{preview.source.origin === "web" ? "web" : `p. ${preview.source.page}`}</span>
          </div>
          <p>{preview.source.snippet}</p>
        </div>
      )}
      <div className={`toast ${toast ? "show" : ""}`} role="status" aria-live="polite">{toast}</div>
    </>
  );
}

let resumedSession = false; // this tab already had a session id before this page load

function storedSessionId(): string {
  let id = "";
  try { id = sessionStorage.getItem(SESSION_KEY) ?? ""; } catch {}
  resumedSession = !!id;
  if (!id) {
    id = newSessionId();
    try { sessionStorage.setItem(SESSION_KEY, id); } catch {}
  }
  return id;
}

function storedTheme(): Theme {
  try {
    const saved = localStorage.getItem("theme");
    if (saved === "light" || saved === "dark") return saved;
  } catch {}
  return "system";
}

function toTurns(messages: ChatMessage[]): TurnState[] {
  const turns: TurnState[] = [];
  for (let i = 0; i + 1 < messages.length; i += 2) {
    const user = messages[i], bot = messages[i + 1];
    turns.push({
      id: `saved-${bot.message_id}`, question: user.content, status: "done", rewritten: user.rewritten_question ?? null,
      sources: bot.sources, text: "", strategy: bot.strategy ?? "hybrid_rerank", scope: "the selected papers",
      response: {
        answer: bot.content, sources: bot.sources, refused: bot.refused, rewritten_question: user.rewritten_question,
        strategy_used: bot.strategy ?? "hybrid_rerank", latency_ms: bot.latency_ms ?? 0,
        used_web_fallback: bot.sources.some((s) => s.origin === "web"),
        weak_context: false, removed_citations: [],
      },
    });
  }
  return turns;
}

function Banners({ apiDown, health, indexMissing }: { apiDown: boolean; health: Health | null; indexMissing: boolean }) {
  if (apiDown || !health || indexMissing) return null;
  const notes: string[] = [];
  if (health.status === "loading") notes.push("The models are loading (about 20 seconds after the API starts). You can ask once they are ready.");
  if (health.status === "error" && health.error) notes.push(health.error.message);
  if (!health.ollama_running) notes.push("Ollama is not running. Start the Ollama app (or run `ollama serve`) before asking.");
  else if (!health.llm_available) notes.push(`The model ${health.llm_model} is not in Ollama. Run \`ollama pull ${health.llm_model}\`.`);
  return (
    <>
      {notes.map((n) => (
        <div key={n} className="banner" role="status"><Icon name="alert" /><span><WithCode text={n} /></span></div>
      ))}
    </>
  );
}

function Intro({ apiDown, indexMissing, papers, canAsk, onAsk }: {
  apiDown: boolean; indexMissing: boolean; papers: PaperInfo[]; canAsk: boolean; onAsk: (q: string) => void;
}) {
  if (apiDown) {
    return (
      <section className="intro">
        <h1>Can&apos;t reach the API</h1>
        <div className="setup">
          <strong>Start the answer service, then this page connects by itself.</strong>
          <p>From the <code>backend</code> folder run <code>.venv\Scripts\uvicorn api.main:app --port 8000</code>.
            This page looks for it at <code>{API_URL}</code>.</p>
        </div>
      </section>
    );
  }
  if (indexMissing) {
    return (
      <section className="intro">
        <h1>No papers indexed yet</h1>
        <div className="setup">
          <strong>Run the ingestion step first.</strong>
          <p>From the <code>backend</code> folder run <code>python scripts/ingest.py</code>. It loads the PDFs in{" "}
            <code>data/pdfs</code> and builds the vector and BM25 indexes. Then restart the API.</p>
        </div>
      </section>
    );
  }
  const pages = papers.reduce((a, p) => a + (p.num_pages ?? 0), 0);
  const chunks = papers.reduce((a, p) => a + (p.num_chunks ?? 0), 0);
  return (
    <section className="intro">
      <h1>What do you want to find in the papers?</h1>
      <p className="lede">Answers are written only from the indexed papers. Every answer lists the three passages it used, with the paper title and page, so you can check the claim yourself.</p>
      {papers.length > 0 && (
        <div className="stats">
          <span><b>{papers.length}</b> papers</span><span><b>{pages}</b> pages</span>
          <span><b>{chunks.toLocaleString("en-US")}</b> chunks</span>
        </div>
      )}
      <p className="examples-label">Try a question</p>
      <ul className="examples">
        {EXAMPLES.map(([q, paper]) => (
          <li key={q}>
            <button disabled={!canAsk} onClick={() => onAsk(q)}>
              <span className="q">{q}<span className="arrow" aria-hidden="true">→</span></span>
              <span className="p">{paper}</span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
