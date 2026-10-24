"use client";

import { Icon } from "./Icon";
import { citationText, isReranked, splitCitations, strategyLabel } from "@/lib/text";
import type { AskResponse, Health, Source } from "@/lib/types";

export type TurnState = {
  id: string;
  question: string;
  status: "loading" | "streaming" | "done" | "error";
  rewritten: string | null;
  sources: Source[];
  text: string; // the answer so far, while streaming
  strategy: string; // requested; the response says which one ran
  scope: string; // "all 11 papers" or "3 of 11 papers"
  response?: AskResponse;
  error?: { code: string; message: string };
};

const STEPS = ["Searching papers", "Ranking passages", "Writing answer"];

/** Inline `code` in server messages, rendered without HTML injection. */
export function WithCode({ text }: { text: string }) {
  return (
    <>
      {text.split(/`([^`]+)`/).map((part, i) => (i % 2 ? <code key={i}>{part}</code> : part))}
    </>
  );
}

function Ticks({ score }: { score: number }) {
  return (
    <span className="ticks" aria-hidden="true">
      {[0.2, 0.4, 0.6, 0.8, 0.9].map((t) => <i key={t} className={score >= t ? "on" : ""} />)}
    </span>
  );
}

function Score({ source, strategy }: { source: Source; strategy: string }) {
  if (isReranked(strategy) && source.score >= 0 && source.score <= 1) {
    return <><Ticks score={source.score} /><span>{source.score.toFixed(2)}</span></>;
  }
  return <span>{strategy === "hybrid" ? "rrf" : "cos"} {source.score.toFixed(3)}</span>;
}

export const cardId = (turn: number, rank: number) => `src-${turn}-${rank}`;

type Props = {
  turn: TurnState;
  index: number;
  openRank: number | null; // the source of this turn shown in the reader
  flashRank: number | null;
  debug: boolean;
  health: Health | null;
  sessionId: string;
  suggestions: string[];
  onOpen: (rank: number) => void;
  onCite: (rank: number) => void;
  onPreview: (el: HTMLElement | null, source?: Source) => void;
  onRetry: () => void;
  onAsk: (question: string) => void;
  onCopy: (text: string, done: string) => void;
};

export function Turn({ turn, index, openRank, flashRank, debug, health, sessionId, suggestions, ...on }: Props) {
  const r = turn.response;
  const refused = !!r?.refused;
  const strategy = r?.strategy_used ?? turn.strategy;
  const showSteps = turn.status === "loading" || (turn.status === "streaming" && !turn.text);
  const stage = turn.sources.length ? 2 : 0;

  return (
    <article className="turn">
      <h2 className="ask">{turn.question}</h2>
      {turn.rewritten && turn.rewritten.trim() !== turn.question.trim() && (
        <div className="searched">Searched as <q>{turn.rewritten}</q></div>
      )}

      {turn.status === "error" && turn.error ? (
        <div className="err" role="alert">
          <span className="k">Answer failed</span>
          <p><WithCode text={turn.error.message} /></p>
          <p className="help" style={{ fontSize: 13.5, color: "var(--ink-2)" }}>Your question is back in the box below.</p>
          <details>
            <summary>Technical details</summary>
            <pre>{`code: ${turn.error.code}\nstrategy: ${turn.strategy}\nscope: ${turn.scope}`}</pre>
          </details>
          <button className="btn" onClick={on.onRetry}><Icon name="redo" />Try again</button>
        </div>
      ) : (
        <>
          {showSteps && (
            <div className="live" role="status">
              <ol className="steps">
                {STEPS.map((label, k) => {
                  const state = k < stage ? "done" : k === stage ? "now" : "";
                  const detail = k === 0 && stage > 0 ? `${turn.scope} · 20 candidates`
                    : k === 1 && stage > 1 ? `${turn.sources.length} kept` : "";
                  return (
                    <li key={label} className={state}>
                      <span className="pip" /><span>{label}{state === "now" ? "…" : ""}</span><span className="t">{detail}</span>
                    </li>
                  );
                })}
              </ol>
            </div>
          )}

          {refused && (
            <div className="note">
              <span className="k">Not in the indexed papers</span>
              <p className="said">{r!.answer}</p>
              <p className="help">The closest passages are below so you can see what was searched. Try asking about something the papers cover:</p>
              <div className="chips">
                {suggestions.map((q) => <button key={q} onClick={() => on.onAsk(q)}>{q}</button>)}
              </div>
            </div>
          )}

          {turn.sources.length > 0 && (
            <>
              <div className="label-row">
                <span className="label">{refused ? "Closest passages" : "Sources"}</span>
                <span className="hint">{refused ? "low relevance" : "passages the answer was written from"}</span>
              </div>
              <div className={`srcs ${refused ? "dim" : ""}`}>
                {turn.sources.slice(0, 3).map((s) => (
                  <button
                    key={s.rank}
                    id={cardId(index, s.rank)}
                    className={`src ${s.origin === "web" ? "web" : ""} ${flashRank === s.rank ? "flash" : ""}`}
                    aria-pressed={openRank === s.rank}
                    aria-label={`Source ${s.rank}: ${s.paper_title}${s.origin === "web" ? ", web result" : `, page ${s.page}`}. Open in reader.`}
                    onClick={() => on.onOpen(s.rank)}
                  >
                    <span className="n">{s.rank}</span>
                    <span className="t" title={s.paper_title}>{s.paper_title}</span>
                    <span className="m">
                      {s.origin === "web" ? <span className="tag">WEB</span> : `p. ${s.page}`}
                      <Score source={s} strategy={strategy} />
                    </span>
                  </button>
                ))}
              </div>
            </>
          )}

          {turn.status === "streaming" && turn.text && (
            <p className="prose">{turn.text}<span className="caret" /></p>
          )}
          {turn.status === "done" && r && !refused && (
            <p className="prose">
              {splitCitations(r.answer, r.sources.length).map((piece, i) => {
                if (!piece.cite) return piece.text;
                const s = r.sources[piece.cite - 1];
                return (
                  <button
                    key={i}
                    className={`cite ${s.origin === "web" ? "web" : ""}`}
                    aria-label={`Source ${piece.cite}: ${citationText(s)}`}
                    onClick={() => on.onCite(piece.cite!)}
                    onMouseEnter={(e) => on.onPreview(e.currentTarget, s)}
                    onMouseLeave={() => on.onPreview(null)}
                    onFocus={(e) => on.onPreview(e.currentTarget, s)}
                    onBlur={() => on.onPreview(null)}
                  >
                    {piece.text}
                  </button>
                );
              })}
            </p>
          )}

          {turn.status === "done" && r && (
            <Foot response={r} turn={turn} debug={debug} health={health} sessionId={sessionId}
                  onRetry={on.onRetry} onCopy={on.onCopy} />
          )}
        </>
      )}
    </article>
  );
}

function Foot({ response: r, turn, debug, health, sessionId, onRetry, onCopy }: {
  response: AskResponse; turn: TurnState; debug: boolean; health: Health | null; sessionId: string;
  onRetry: () => void; onCopy: (text: string, done: string) => void;
}) {
  const label = strategyLabel(r.strategy_used);
  const reranked = isReranked(r.strategy_used);
  const sec = (r.latency_ms / 1000).toFixed(1);
  const best = r.sources[0]?.score;
  const threshold = health?.weak_threshold;
  const summary = r.refused && reranked && best !== undefined && threshold
    ? `${label} · best score ${best.toFixed(2)} ${best < threshold ? "<" : "≥"} ${threshold.toFixed(2)} · ${sec} s`
    : `${label} · ${reranked ? "20 candidates → " : ""}${r.sources.length} · ${sec} s`;

  const rows: [string, string][] = [];
  if (r.rewrite_ms) rows.push(["Rewrote the follow-up as a standalone question", `${r.rewrite_ms} ms`]);
  if (r.retrieval_ms != null) {
    rows.push([`Searched ${turn.scope}${reranked ? ", reranked 20 candidates, kept 3" : ""}`, `${r.retrieval_ms} ms`]);
  }
  const genMs = r.latency_ms - (r.retrieval_ms ?? 0) - (r.rewrite_ms ?? 0);
  rows.push([
    r.refused ? "Model declined: the passages do not answer it" : `Wrote the answer with ${health?.llm_model ?? "the local model"}`,
    `${genMs} ms${r.first_token_ms && !r.refused ? ` · first word ${r.first_token_ms} ms` : ""}`,
  ]);
  const slug = (health?.embedding_model ?? "").split("/").pop()!.replace(/[^a-z0-9]+/gi, "_").toLowerCase();
  const copyAnswer = () => onCopy(
    `${r.answer}\n\n${r.sources.map((s) => `[${s.rank}] ${citationText(s)}`).join("\n")}`, "Answer and sources copied");

  return (
    <div className="foot">
      <div className="foot-row">
        {!r.refused && <button className="act" onClick={copyAnswer}><Icon name="copy" />Copy answer</button>}
        <button className="act" onClick={onRetry}><Icon name="redo" />Ask again</button>
      </div>
      <details className="trace" open={debug || undefined}>
        <summary><Icon name="chev" className="chev" />{summary}</summary>
        <div className="trace-body">
          <ol className="steps">
            {rows.map(([what, time]) => (
              <li key={what} className="done"><span className="pip" /><span>{what}</span><span className="t">{time}</span></li>
            ))}
          </ol>
          {debug && (
            <>
              <dl className="kv">
                <dt>Query used</dt><dd>{r.rewritten_question ?? "the question as typed"}</dd>
                <dt>Strategy</dt><dd>{r.strategy_used}</dd>
                <dt>Weak context</dt><dd>{r.weak_context ? "yes, the prompt asked for a refusal unless a passage answers directly" : "no"}</dd>
                <dt>Removed citations</dt><dd>{r.removed_citations.length ? r.removed_citations.join(", ") : "none"}</dd>
                {health && <><dt>Collection</dt><dd>{`papers__${slug}__${health.chunking}`}</dd></>}
                <dt>Session</dt><dd>{sessionId}</dd>
              </dl>
              <div className="tbl-wrap">
                <table className="tbl">
                  <thead><tr><th>#</th><th>chunk_id</th><th className="r">score</th></tr></thead>
                  <tbody>
                    {r.sources.map((s) => (
                      <tr key={s.rank}><td>{s.rank}</td><td>{s.chunk_id ?? ""}</td><td className="r">{s.score.toFixed(3)}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      </details>
    </div>
  );
}
