"use client";

import { Icon } from "./Icon";
import type { Health, SessionSummary } from "@/lib/types";

export type Theme = "system" | "light" | "dark";

type Props = {
  open: boolean;
  health: Health | null;
  numPapers: number;
  sessions: SessionSummary[];
  sessionId: string;
  debug: boolean;
  theme: Theme;
  onNewChat: () => void;
  onLoad: (sessionId: string) => void;
  onDebug: (on: boolean) => void;
  onTheme: (theme: Theme) => void;
  onClear: () => void;
};

export function Rail({ open, health, numPapers, sessions, sessionId, debug, theme, ...on }: Props) {
  const embed = health?.embedding_model.split("/").pop();
  return (
    <nav className={`rail ${open ? "open" : ""}`} aria-label="Chats and settings">
      <div className="wordmark">
        <b>Research Paper Answer Bot</b>
        <span>RAG over {numPapers || "the"} GenAI papers</span>
      </div>
      <button className="new-chat" onClick={on.onNewChat}><Icon name="plus" />New chat</button>
      <section>
        <h2>Recent</h2>
        {sessions.length ? (
          <ul className="sessions">
            {sessions.slice(0, 12).map((s) => (
              <li key={s.session_id}>
                <button aria-current={s.session_id === sessionId} title={s.title ?? undefined} onClick={() => on.onLoad(s.session_id)}>
                  {s.title || "Untitled chat"}
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p className="empty-note">Your chats appear here.</p>
        )}
      </section>
      <div className="rail-foot">
        {health && (
          <dl className="spec">
            <dt>LLM</dt><dd>{health.llm_model} · Ollama</dd>
            <dt>Embed</dt><dd>{embed}</dd>
            <dt>Rerank</dt><dd>{health.reranker ? "bge-reranker-v2-m3" : "off"}</dd>
            <dt>Index</dt><dd>{health.chunking}</dd>
          </dl>
        )}
        <label className="switch">
          <span>Show retrieval details</span>
          <input type="checkbox" checked={debug} onChange={(e) => on.onDebug(e.target.checked)} />
        </label>
        <label className="theme">
          <span>Theme</span>
          <select value={theme} onChange={(e) => on.onTheme(e.target.value as Theme)}>
            <option value="system">System</option>
            <option value="light">Light</option>
            <option value="dark">Dark</option>
          </select>
        </label>
        <button className="text-btn" onClick={on.onClear}>Clear history</button>
      </div>
    </nav>
  );
}
