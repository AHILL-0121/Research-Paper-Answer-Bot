"use client";

import { Icon } from "./Icon";
import { citationText, highlight, shortTitle } from "@/lib/text";
import type { PaperInfo, Source } from "@/lib/types";

export type ReaderSource = { source: Source; total: number; answer: string; refused: boolean };

type Props = {
  tab: "passage" | "library";
  open: boolean; // drawer state on narrow screens
  current: ReaderSource | null;
  papers: PaperInfo[];
  loading: boolean; // the library has not arrived yet
  selected: Set<string>;
  indexMissing: boolean;
  onTab: (tab: "passage" | "library") => void;
  onStep: (delta: number) => void;
  onToggle: (paperId: string) => void;
  onSelectAll: (all: boolean) => void;
  onCopy: (text: string, done: string) => void;
  onClose: () => void;
};

export function Reader({ tab, open, current, papers, loading, selected, indexMissing, ...on }: Props) {
  return (
    <aside className={`reader ${open ? "open" : ""}`} aria-label="Reader">
      <div className="reader-head">
        <div className="tabs" role="tablist">
          {(["passage", "library"] as const).map((t) => (
            <button key={t} role="tab" id={`tab-${t}`} aria-selected={tab === t} aria-controls="reader-body" onClick={() => on.onTab(t)}>
              {t === "passage" ? "Passage" : "Library"}
            </button>
          ))}
        </div>
        <button className="icon-btn reader-close" aria-label="Close reader" onClick={on.onClose}><Icon name="x" /></button>
      </div>
      <div className="reader-body" id="reader-body" role="tabpanel" aria-labelledby={`tab-${tab}`}>
        {tab === "library"
          ? <Library papers={papers} loading={loading} selected={selected} indexMissing={indexMissing} onToggle={on.onToggle} onSelectAll={on.onSelectAll} />
          : <Passage current={current} papers={papers} onStep={on.onStep} onCopy={on.onCopy} />}
      </div>
    </aside>
  );
}

function Passage({ current, papers, onStep, onCopy }: {
  current: ReaderSource | null; papers: PaperInfo[]; onStep: (d: number) => void; onCopy: (t: string, d: string) => void;
}) {
  if (!current) {
    return <div className="placeholder">Select a source or a citation such as [1] to read the passage on its page.</div>;
  }
  const { source: s, total, answer, refused } = current;
  const isWeb = s.origin === "web";
  const paper = papers.find((p) => p.paper_id === s.paper_id || p.paper_title === s.paper_title);
  const parts = refused || isWeb ? [{ text: s.full_text.split(/\s+/).join(" "), mark: false }] : highlight(s.full_text, answer, s.rank);
  const pdf = paper?.arxiv_id ? `https://arxiv.org/pdf/${paper.arxiv_id}#page=${s.page}` : null;

  return (
    <>
      <div className="pager">
        <span>Source {s.rank} of {total}</span>
        <div>
          <button aria-label="Previous source" disabled={s.rank === 1} onClick={() => onStep(-1)}><Icon name="left" /></button>
          <button aria-label="Next source" disabled={s.rank === total} onClick={() => onStep(1)}><Icon name="chev" /></button>
        </div>
      </div>
      <div className={`page ${isWeb ? "web" : ""}`}>
        <div className="running">
          <span>{isWeb ? "Web result" : shortTitle(s.paper_title)}</span>
          <span>{isWeb ? "not from the library" : paper?.year ?? ""}</span>
        </div>
        <p className="body">{parts.map((p, i) => (p.mark ? <mark key={i}>{p.text}</mark> : <span key={i}>{p.text}</span>))}</p>
        {isWeb && s.url
          ? <a href={s.url} target="_blank" rel="noopener noreferrer">{s.url}</a>
          : <div className="folio">{s.page}</div>}
      </div>
      <dl className="kv">
        <dt>Paper</dt><dd>{s.paper_title}</dd>
        <dt>{isWeb ? "Source" : "Page"}</dt><dd>{isWeb ? "web search" : paper?.num_pages ? `${s.page} of ${paper.num_pages}` : s.page}</dd>
        {s.chunk_id && <><dt>Chunk</dt><dd>{s.chunk_id}</dd></>}
        <dt>Score</dt><dd>{s.score.toFixed(3)}</dd>
      </dl>
      <div className="reader-actions">
        <button className="btn" onClick={() => onCopy(citationText(s), "Citation copied")}><Icon name="copy" />Copy citation</button>
        {pdf && !isWeb && (
          <a className="btn" href={pdf} target="_blank" rel="noopener noreferrer"><Icon name="external" />Page {s.page} on arXiv</a>
        )}
      </div>
    </>
  );
}

function Library({ papers, loading, selected, indexMissing, onToggle, onSelectAll }: {
  papers: PaperInfo[]; loading: boolean; selected: Set<string>; indexMissing: boolean;
  onToggle: (id: string) => void; onSelectAll: (all: boolean) => void;
}) {
  if (loading) return <div className="placeholder">Loading the library…</div>;
  if (indexMissing || !papers.length) {
    return <div className="placeholder">No papers indexed yet. Run the ingestion step first.</div>;
  }
  return (
    <>
      <div className="lib-top">
        <p>{selected.size} of {papers.length} papers searched</p>
        <div>
          <button className="text-btn" onClick={() => onSelectAll(true)}>All</button>
          <button className="text-btn" onClick={() => onSelectAll(false)}>None</button>
        </div>
      </div>
      <ul className="lib">
        {papers.map((p) => (
          <li key={p.paper_id}>
            <label>
              <input type="checkbox" checked={selected.has(p.paper_id)} onChange={() => onToggle(p.paper_id)} />
              <span className="t">{p.paper_title}</span>
              <span className="m">
                {[p.year, p.num_pages && `${p.num_pages} pp`, p.num_chunks && `${p.num_chunks} chunks`].filter(Boolean).join(" · ")}
              </span>
            </label>
          </li>
        ))}
      </ul>
    </>
  );
}
