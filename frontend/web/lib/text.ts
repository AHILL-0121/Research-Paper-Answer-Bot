import type { Source, Strategy } from "./types";

export const STRATEGY_LABELS: Record<Strategy, string> = {
  dense: "Dense",
  mmr: "MMR",
  hybrid: "Hybrid",
  hybrid_rerank: "Hybrid + rerank",
  multi_query: "Multi-query",
};

export const strategyLabel = (s: string) => STRATEGY_LABELS[s as Strategy] ?? s;
export const isReranked = (s: string) => s === "hybrid_rerank" || s === "multi_query";
export const shortTitle = (title: string) => title.split(":")[0];
export const citationText = (s: Source) =>
  s.origin === "web" ? `${s.paper_title}, ${s.url ?? ""}` : `${s.paper_title}, p. ${s.page}`;

export type Piece = { text: string; cite?: number };

/** Split an answer into text and citation markers; markers outside 1..n are dropped. */
export function splitCitations(answer: string, n: number): Piece[] {
  const pieces: Piece[] = [];
  let last = 0;
  for (const m of answer.matchAll(/\[(\d+)\]/g)) {
    pieces.push({ text: answer.slice(last, m.index) });
    const k = Number(m[1]);
    if (k >= 1 && k <= n) pieces.push({ text: m[0], cite: k });
    last = (m.index ?? 0) + m[0].length;
  }
  pieces.push({ text: answer.slice(last) });
  return pieces.filter((p) => p.text);
}

const words = (s: string) => new Set((s.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter((w) => w.length > 3));

/** The passage split into sentences, with the one that best matches the answer's [rank]
 * sentences marked (at least 3 shared words). The same rule as the Streamlit app. */
export function highlight(passage: string, answer: string, rank: number): { text: string; mark: boolean }[] {
  const flat = passage.split(/\s+/).join(" ").trim();
  const cited = answer.split(/(?<=[.!?])\s+/).filter((s) => s.includes(`[${rank}]`));
  const sentences = flat.split(/(?<=[.!?])\s+(?=[A-Z(])/);
  if (!cited.length || sentences.length < 2) return [{ text: flat, mark: false }];
  const wanted = new Set(cited.flatMap((s) => [...words(s)]));
  const overlap = sentences.map((s) => [...words(s)].filter((w) => wanted.has(w)).length);
  const best = overlap.indexOf(Math.max(...overlap));
  return sentences.map((text, i) => ({ text: i < sentences.length - 1 ? text + " " : text, mark: i === best && overlap[best] >= 3 }));
}

export const newSessionId = () => `web-${crypto.randomUUID().replace(/-/g, "").slice(0, 12)}`;
