"""HTML and CSS for the Streamlit app, following plan/SampleUI.html.

Streamlit widgets cannot be styled one by one, so the parts that carry the design (the answer,
source cards, the reader page, the steps list) are rendered as escaped HTML here, and app.py
lays clickable widgets over them. Every string from the papers or the model goes through
``esc`` before it reaches the page.
"""

import html
import re

from rpab.schemas import AskResponse, PaperInfo, Source

STRATEGY_LABELS = {"dense": "Dense", "mmr": "MMR", "hybrid": "Hybrid", "hybrid_rerank": "Hybrid + rerank",
                   "multi_query": "Multi-query"}
STEPS = ["Searching papers", "Ranking passages", "Writing answer"]
CITATION = re.compile(r"\[(\d+)\]")
WORD = re.compile(r"[a-z0-9]+")


def esc(text) -> str:
    return html.escape(str(text), quote=True)


def short_title(title: str) -> str:
    return title.split(":")[0]


# ---- theme -------------------------------------------------------------------------------

LIGHT = {"desk": "#ECEEF1", "sheet": "#FFFFFF", "ink": "#1A1D24", "ink-2": "#4A505C", "ink-3": "#737A89",
         "rule": "#D8DBE1", "rule-2": "#E8EAEE", "pen": "#2F3C9E", "pen-soft": "#E7E9F7", "marker": "#FBE38A",
         "marker-ink": "#362C00", "note-bg": "#F2F3F6", "danger": "#A12A2A", "danger-bg": "#FBEDEC",
         "shadow": "0 1px 0 rgba(26,29,36,.04), 0 10px 30px -14px rgba(26,29,36,.28)"}
DARK = {"desk": "#111317", "sheet": "#181B21", "ink": "#E6E8EC", "ink-2": "#B4B9C3", "ink-3": "#868D9A",
        "rule": "#2C3038", "rule-2": "#22262D", "pen": "#A9B2FF", "pen-soft": "#262B4C", "marker": "#5E4F10",
        "marker-ink": "#FDEDB0", "note-bg": "#1F2229", "danger": "#F2A6A2", "danger-bg": "#351C1C",
        "shadow": "0 1px 0 rgba(0,0,0,.3), 0 12px 32px -14px rgba(0,0,0,.7)"}


def _tokens(values: dict) -> str:
    return "".join(f"--{k}:{v};" for k, v in values.items())


def theme_css(theme_type: str | None) -> str:
    """Colour tokens: the OS preference until Streamlit reports its theme, then Streamlit's choice."""
    if theme_type in ("light", "dark"):
        return f":root{{{_tokens(LIGHT if theme_type == 'light' else DARK)}}}"
    return f":root{{{_tokens(LIGHT)}}} @media (prefers-color-scheme: dark){{:root{{{_tokens(DARK)}}}}}"


CSS = """
:root { --f-ui: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
        --f-read: "Source Serif 4", Georgia, serif; --f-mono: "IBM Plex Mono", Consolas, monospace; }
.block-container { padding-top: 2.2rem; padding-bottom: 6rem; max-width: 1400px; }
[data-testid="stSidebar"] { background: var(--desk); }
[data-testid="stSidebarUserContent"] { padding-top: 1.2rem; }
.mono { font-family: var(--f-mono); }

/* Rail */
.wordmark b { display: block; font: 600 17px/1.15 var(--f-read); letter-spacing: -.01em; color: var(--ink); }
.wordmark span { display: block; font: 11px/1.4 var(--f-mono); color: var(--ink-3); margin-top: 3px; }
.rail-h { margin: 6px 0 2px 2px; font: 500 11px/1 var(--f-mono); letter-spacing: .08em; text-transform: uppercase;
          color: var(--ink-3); }
.spec { display: grid; grid-template-columns: auto 1fr; gap: 3px 10px; font: 12px/1.45 var(--f-mono); margin: 0; }
.spec dt { color: var(--ink-3); } .spec dd { margin: 0; color: var(--ink-2); overflow-wrap: anywhere; }
[class*="st-key-sess-"] button { justify-content: flex-start; border: 0; background: transparent; padding: 4px 8px;
          min-height: 0; color: var(--ink-2); }
[class*="st-key-sess-"] button p { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 14px; }
[class*="st-key-sess-"] button:hover { background: var(--rule-2); color: var(--ink); }
.st-key-sess-current button { background: var(--sheet); box-shadow: inset 0 0 0 1px var(--rule); color: var(--ink); }
.st-key-new_chat button { justify-content: flex-start; background: var(--sheet); font-weight: 500; }
[class*="st-key-textbtn-"] button { border: 0; background: transparent; color: var(--pen); padding: 0 2px;
          min-height: 0; justify-content: flex-start; }
[class*="st-key-textbtn-"] button:hover { text-decoration: underline; text-underline-offset: 3px; background: none; }

/* Conversation */
.intro { display: flex; flex-direction: column; gap: 14px; padding-top: 3vh; max-width: 760px; }
.intro h1 { margin: 0; padding: 0; font: 600 30px/1.2 var(--f-read); letter-spacing: -.015em; max-width: 22ch;
            color: var(--ink); }
.intro .lede { margin: 0; color: var(--ink-2); max-width: 58ch; font-size: 15.5px; }
.stats { display: flex; flex-wrap: wrap; gap: 4px 16px; font: 12.5px var(--f-mono); color: var(--ink-3);
         font-variant-numeric: tabular-nums; }
.stats b { color: var(--ink); font-weight: 500; }
.label { font: 500 11px/1 var(--f-mono); letter-spacing: .08em; text-transform: uppercase; color: var(--ink-3); }
.examples-label { margin: 18px 0 4px; display: block; }
.ex { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 4px 18px; align-items: baseline;
      padding: 12px 4px; border-bottom: 1px solid var(--rule-2); }
.st-key-hit-ex-0 .ex { border-top: 1px solid var(--rule-2); }
.ex .q { font: 17px/1.4 var(--f-read); color: var(--ink); }
.ex .p { font: 12px var(--f-mono); color: var(--ink-3); white-space: nowrap; }
.arrow { display: inline-block; transition: transform .15s; color: var(--ink-3); margin-left: 6px; }
[class*="st-key-hit-ex-"]:hover .q { color: var(--pen); }
[class*="st-key-hit-ex-"]:hover .arrow { transform: translateX(3px); color: var(--pen); }
.setup { border: 1px solid var(--rule); border-radius: 8px; padding: 14px 16px; background: var(--note-bg);
         max-width: 58ch; color: var(--ink); }
.setup p { margin: 6px 0 0; color: var(--ink-2); }
.setup code, .err code { font: 13px var(--f-mono); background: var(--sheet); padding: 1px 5px; border-radius: 3px;
         border: 1px solid var(--rule); color: var(--ink); }

.ask { font: 600 22px/1.3 var(--f-read); letter-spacing: -.01em; margin: 26px 0 2px; padding: 0; color: var(--ink);
       overflow-wrap: anywhere; }
.searched { font-size: 13px; color: var(--ink-3); margin-bottom: 4px; }
.searched q { color: var(--ink-2); font-style: italic; font-family: var(--f-read); font-size: 14.5px; }
.label-row { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; margin: 8px 0 2px; }
.label-row .hint { font-size: 12.5px; color: var(--ink-3); }

.src { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 2px 10px; padding: 10px 11px; height: 100%;
       border: 1px solid var(--rule); border-radius: 6px; background: var(--sheet); scroll-margin-top: 90px;
       transition: border-color .15s, background .15s; }
.src .n { grid-row: span 2; font: 500 13px/1.5 var(--f-mono); color: var(--pen); }
.src .t { font-size: 13.5px; font-weight: 500; line-height: 1.35; color: var(--ink); overflow: hidden;
          display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
.src .m { display: flex; align-items: center; gap: 8px; font: 12px var(--f-mono); color: var(--ink-3);
          font-variant-numeric: tabular-nums; }
.src.on, .src:target { border-color: var(--pen); background: var(--pen-soft); }
.src:target { animation: flash 1.2s ease-out; }
@keyframes flash { 0% { box-shadow: 0 0 0 4px var(--pen-soft); } 100% { box-shadow: 0 0 0 0 transparent; } }
[class*="st-key-srcs-dim-"] .src { opacity: .78; }
[class*="st-key-hit-src-"]:hover .src { border-color: var(--ink-3); }
.ticks { display: inline-flex; gap: 2px; }
.ticks i { width: 4px; height: 9px; border-radius: 1px; background: var(--rule); }
.ticks i.on { background: var(--ink-2); }

.prose { font: 17.5px/1.65 var(--f-read); margin: 10px 0 4px; max-width: 66ch; overflow-wrap: break-word;
         color: var(--ink); }
.cite { font: 500 12.5px/1 var(--f-mono); color: var(--pen) !important; padding: 1px 2px; border-radius: 3px;
        vertical-align: 2px; text-decoration: none !important; }
.cite:hover, .cite:focus-visible { background: var(--pen-soft); }
.caret { display: inline-block; width: .5em; height: 1.05em; vertical-align: -3px; background: var(--ink);
         margin-left: 2px; animation: blink 1s steps(2) infinite; }
@keyframes blink { 50% { opacity: 0; } }

.note, [class*="st-key-refusal-"] { display: flex; flex-direction: column; gap: 6px; padding: 14px 16px;
        border-radius: 8px; background: var(--note-bg); margin: 6px 0; }
[class*="st-key-refusal-"] .note { padding: 0; margin: 0; background: none; }
.note .k, .err .k { font: 500 11px/1 var(--f-mono); letter-spacing: .08em; text-transform: uppercase; color: var(--ink-3); }
.note p, .err p { margin: 0; color: var(--ink); }
.note .said { font: 17.5px/1.5 var(--f-read); }
.note .help { color: var(--ink-2); font-size: 14px; }
.err { padding: 14px 16px; border-radius: 8px; background: var(--danger-bg); display: flex; flex-direction: column;
       gap: 8px; margin: 6px 0; }
.err .k { color: var(--danger); }
.err details summary { cursor: pointer; font-size: 13px; color: var(--ink-2); }
.err pre { margin: 8px 0 0; white-space: pre-wrap; font: 12.5px/1.5 var(--f-mono); color: var(--ink-2);
           background: none; padding: 0; }
[class*="st-key-chip-"] button { border-radius: 999px; padding: 4px 12px; min-height: 0; background: var(--sheet);
          text-align: left; }
[class*="st-key-chip-"] button p { font-size: 13.5px; white-space: normal; }
[class*="st-key-chip-"] button:hover { border-color: var(--pen); color: var(--pen); }
[class*="st-key-act-"] button { border: 0; background: transparent; color: var(--ink-3); padding: 4px 8px;
          min-height: 0; }
[class*="st-key-act-"] button p { font-size: 13px; }
[class*="st-key-act-"] button:hover { background: var(--rule-2); color: var(--ink); }

.trace { border-top: 1px solid var(--rule-2); padding-top: 6px; margin-top: 4px; }
.trace summary { list-style: none; cursor: pointer; display: inline-flex; gap: 6px; align-items: center;
                 font: 12px var(--f-mono); color: var(--ink-3); padding: 5px 8px; border-radius: 6px;
                 font-variant-numeric: tabular-nums; }
.trace summary::-webkit-details-marker { display: none; }
.trace summary::before { content: "›"; display: inline-block; transition: transform .15s; font-size: 14px; }
.trace[open] summary::before { transform: rotate(90deg); }
.trace summary:hover { background: var(--rule-2); color: var(--ink); }
.trace-body { margin: 6px 0 0; padding: 12px 14px; border: 1px solid var(--rule-2); border-radius: 8px;
              display: flex; flex-direction: column; gap: 12px; }
.steps { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 7px; }
.steps li { display: grid; grid-template-columns: 16px minmax(0, 1fr) auto; gap: 10px; align-items: center;
            font-size: 14px; color: var(--ink-3); margin: 0; }
.steps li.done { color: var(--ink-2); } .steps li.now { color: var(--ink); font-weight: 500; }
.steps .t { font: 12px var(--f-mono); color: var(--ink-3); font-variant-numeric: tabular-nums; }
.pip { width: 10px; height: 10px; border-radius: 50%; border: 1.5px solid var(--rule); justify-self: center; }
.done .pip { background: var(--ink-2); border-color: var(--ink-2); }
.now .pip { border-color: var(--pen); border-top-color: transparent; width: 12px; height: 12px;
            animation: spin .8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.live { padding: 14px 16px; border: 1px solid var(--rule-2); border-radius: 8px; margin: 6px 0; }
.kv { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 4px 14px; margin: 0; font-size: 13px; }
.kv dt { color: var(--ink-3); } .kv dd { margin: 0; font-family: var(--f-mono); font-size: 12.5px;
         overflow-wrap: anywhere; color: var(--ink); }
.tbl { width: 100%; border-collapse: collapse; font: 12px/1.4 var(--f-mono); font-variant-numeric: tabular-nums; }
.tbl th, .tbl td { text-align: left; padding: 5px 8px 5px 0; border: 0; border-bottom: 1px solid var(--rule-2);
                   white-space: nowrap; background: none; }
.tbl th { color: var(--ink-3); font-weight: 500; } .tbl td { color: var(--ink); }
.tbl .r { text-align: right; }

/* Reader */
[data-testid="stColumn"]:has(.st-key-reader) { position: sticky; top: 3.75rem; align-self: flex-start; }
.st-key-reader { background: var(--desk); border-radius: 10px; padding: 12px 14px 16px;
                 max-height: calc(100vh - 16rem); overflow-y: auto; }
.pager { display: flex; align-items: center; justify-content: space-between; font: 12px var(--f-mono);
         color: var(--ink-3); }
.page { background: var(--sheet); border-radius: 2px; box-shadow: var(--shadow); padding: 18px 20px 14px;
        display: flex; flex-direction: column; gap: 14px; min-height: 300px; }
.running { display: flex; justify-content: space-between; gap: 12px; font: 10.5px/1.3 var(--f-mono);
           letter-spacing: .06em; text-transform: uppercase; color: var(--ink-3); padding-bottom: 8px;
           border-bottom: 1px solid var(--rule-2); }
.running span:first-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.page .body { font: 15px/1.6 var(--f-read); margin: 0; hyphens: auto; color: var(--ink); overflow-wrap: anywhere; }
.page mark { background: var(--marker); color: var(--marker-ink); padding: .05em .12em; border-radius: 2px;
             box-decoration-break: clone; -webkit-box-decoration-break: clone; }
.folio { text-align: center; font: 12px var(--f-read); color: var(--ink-3); }
.placeholder { color: var(--ink-3); font-size: 14px; text-align: center; padding: 40px 12px;
               border: 1px dashed var(--rule); border-radius: 8px; }
.lib-top { font: 12px var(--f-mono); color: var(--ink-3); }
.st-key-library [data-testid="stCheckbox"] { background: var(--sheet); padding: 8px 10px; margin: 0;
          border: 1px solid var(--rule); border-bottom-width: 0; }
.st-key-library [data-testid="stElementContainer"]:first-child [data-testid="stCheckbox"] { border-radius: 8px 8px 0 0; }
.st-key-library [data-testid="stElementContainer"]:last-child [data-testid="stCheckbox"] {
          border-bottom-width: 1px; border-radius: 0 0 8px 8px; }
.st-key-library { gap: 0; }
.st-key-library [data-testid="stElementContainer"], .st-key-library [data-testid="stCheckbox"],
.st-key-library label { width: 100%; }
.st-key-library label p { font: 15px/1.35 var(--f-read); }
.st-key-library [data-testid="stWidgetLabel"] + div, .st-key-library small { font: 11.5px var(--f-mono); }

/* Clickable HTML: an invisible button is laid over the card it belongs to */
[class*="st-key-hit-"] { position: relative; gap: 0; }
[class*="st-key-hit-"] [data-testid="stElementContainer"]:has(button) { position: absolute; inset: 0; margin: 0;
          z-index: 2; height: 100%; }
[class*="st-key-hit-"] [data-testid="stElementContainer"]:has(button) > div,
[class*="st-key-hit-"] .stButton, [class*="st-key-hit-"] button { width: 100%; height: 100%; }
[class*="st-key-hit-"] button { opacity: 0; cursor: pointer; }
[class*="st-key-hit-"]:has(button:focus-visible) { outline: 2px solid var(--pen); outline-offset: 2px; border-radius: 6px; }

/* Composer */
[data-testid="stChatInput"] { max-width: 760px; }
[data-testid="stChatInput"] textarea { font-size: 16px; }
.fine { font-size: 12px; color: var(--ink-3); }

/* Below the sample's 1180 px breakpoint the reader moves under the conversation */
@media (max-width: 1180px) {
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-reader) { flex-wrap: wrap; }
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-reader) > [data-testid="stColumn"] {
    width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important; }
  [data-testid="stColumn"]:has(.st-key-reader) { position: static; }
  .st-key-reader { max-height: none; } }
@media (max-width: 900px) { .ask { font-size: 20px; } .prose { font-size: 16.5px; }
  [class*="st-key-srcs-"] [data-testid="stHorizontalBlock"] { flex-direction: column; } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important;
          transition: none !important; } }
"""


# ---- pieces ------------------------------------------------------------------------------

def ticks(score: float) -> str:
    bars = "".join(f'<i class="{"on" if score >= t else ""}"></i>' for t in (0.2, 0.4, 0.6, 0.8, 0.9))
    return f'<span class="ticks" aria-hidden="true">{bars}</span>'


def score_label(source: Source, strategy: str) -> str:
    """Rerank scores are probabilities, shown with ticks; fused and cosine scores are not comparable to them."""
    if strategy in ("hybrid_rerank", "multi_query") and 0 <= source.score <= 1:
        return f"{ticks(source.score)}<span>{source.score:.2f}</span>"
    kind = "rrf" if strategy == "hybrid" else "cos"
    return f"<span>{kind} {source.score:.3f}</span>"


def source_card(source: Source, turn: int, strategy: str, active: bool) -> str:
    return (f'<div class="src{" on" if active else ""}" id="c-{turn}-{source.rank}">'
            f'<span class="n">{source.rank}</span><span class="t" title="{esc(source.paper_title)}">'
            f'{esc(source.paper_title)}</span><span class="m">p. {source.page}{score_label(source, strategy)}</span></div>')


def with_citations(answer: str, sources: list[Source], turn: int) -> str:
    """Escaped answer text with each [n] as a link that scrolls to and highlights source card n."""
    def chip(match: re.Match) -> str:
        n = int(match.group(1))
        if not 1 <= n <= len(sources):
            return ""
        s = sources[n - 1]
        tip = esc(f"{short_title(s.paper_title)}, p. {s.page}: {s.snippet}")
        return f'<a class="cite" href="#c-{turn}-{n}" title="{tip}" aria-label="Source {n}">[{n}]</a>'
    return CITATION.sub(chip, esc(answer))


def steps(stage: int, details: dict[int, str]) -> str:
    items = []
    for k, label in enumerate(STEPS):
        state = "done" if k < stage else "now" if k == stage else ""
        items.append(f'<li class="{state}"><span class="pip"></span><span>{label}{"…" if state == "now" else ""}</span>'
                     f'<span class="t">{esc(details.get(k, ""))}</span></li>')
    return f'<div class="live" role="status"><ol class="steps">{"".join(items)}</ol></div>'


def searched(rewritten: str | None, question: str) -> str:
    if not rewritten or rewritten.strip() == question.strip():
        return ""
    return f'<div class="searched">Searched as <q>{esc(rewritten)}</q></div>'


def sources_header(refused: bool) -> str:
    label, hint = ("Closest passages", "low relevance") if refused else ("Sources", "passages the answer was written from")
    return f'<div class="label-row"><span class="label">{label}</span><span class="hint">{hint}</span></div>'


def refusal_note(answer: str) -> str:
    return ('<div class="note"><span class="k">Not in the indexed papers</span>'
            f'<p class="said">{esc(answer)}</p><p class="help">The closest passages are below so you can see what '
            'was searched. Try asking about something the papers cover:</p></div>')


def error_box(message: str, details: str) -> str:
    return ('<div class="err" role="alert"><span class="k">Answer failed</span>'
            f'<p>{inline_code(message)}</p>'
            f'<details><summary>Technical details</summary><pre>{esc(details)}</pre></details></div>')


def inline_code(text: str) -> str:
    """Escape, then turn `backticks` into code spans."""
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", esc(text))


def trace(response: AskResponse, scope: str, llm: str, debug: bool, session_id: str | None,
          threshold: float | None, collection: str) -> str:
    sec = response.latency_ms / 1000
    label = STRATEGY_LABELS.get(response.strategy_used, response.strategy_used)
    reranked = response.strategy_used in ("hybrid_rerank", "multi_query")
    best = response.sources[0].score if response.sources else None
    if response.refused and reranked and best is not None and threshold is not None:
        summary = f"{label} · best score {best:.2f} {'<' if best < threshold else '≥'} {threshold:.2f} · {sec:.1f} s"
    else:
        summary = f"{label} · {'20 candidates → ' if reranked else ''}{len(response.sources)} · {sec:.1f} s"
    rows = []
    if response.rewrite_ms:
        rows.append(("Rewrote the follow-up as a standalone question", f"{response.rewrite_ms} ms"))
    if response.retrieval_ms is not None:
        rows.append((f"Searched {scope}" + (", reranked 20 candidates, kept 3" if reranked else ""),
                     f"{response.retrieval_ms} ms"))
    writing = "Model declined: the passages do not answer it" if response.refused else f"Wrote the answer with {llm}"
    gen_ms = response.latency_ms - (response.retrieval_ms or 0) - (response.rewrite_ms or 0)
    rows.append((writing, f"{gen_ms} ms" + (f" · first word {response.first_token_ms} ms"
                                           if response.first_token_ms and not response.refused else "")))
    body = '<ol class="steps">' + "".join(
        f'<li class="done"><span class="pip"></span><span>{esc(a)}</span><span class="t">{esc(b)}</span></li>'
        for a, b in rows) + "</ol>"
    if debug:
        weak = "yes, the prompt asked for a refusal unless a passage answers directly" if response.weak_context else "no"
        kv = [("Query used", response.rewritten_question or "the question as typed"), ("Strategy", response.strategy_used),
              ("Weak context", weak), ("Removed citations", ", ".join(response.removed_citations) or "none"),
              ("Collection", collection), ("Session", session_id or "none")]
        body += '<dl class="kv">' + "".join(f"<dt>{esc(a)}</dt><dd>{esc(b)}</dd>" for a, b in kv) + "</dl>"
        body += ('<table class="tbl"><thead><tr><th>#</th><th>chunk_id</th><th class="r">score</th></tr></thead><tbody>'
                 + "".join(f'<tr><td>{s.rank}</td><td>{esc(s.chunk_id or "")}</td><td class="r">{s.score:.3f}</td></tr>'
                           for s in response.sources) + "</tbody></table>")
    return (f'<details class="trace"{" open" if debug else ""}><summary>{esc(summary)}</summary>'
            f'<div class="trace-body">{body}</div></details>')


def highlight(passage: str, answer: str, rank: int) -> str:
    """The passage with the sentence that best matches the answer's [rank] sentences marked."""
    cited = [s for s in re.split(r"(?<=[.!?])\s+", answer) if f"[{rank}]" in s]
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", " ".join(passage.split()))
    if not cited or len(sentences) < 2:
        return esc(" ".join(passage.split()))
    wanted = {w for s in cited for w in WORD.findall(s.lower()) if len(w) > 3}
    overlap = [len(wanted & set(WORD.findall(s.lower()))) for s in sentences]
    best = max(range(len(sentences)), key=overlap.__getitem__)
    return " ".join(f"<mark>{esc(s)}</mark>" if i == best and overlap[best] >= 3 else esc(s)
                    for i, s in enumerate(sentences))


def reader_page(source: Source, paper: PaperInfo | None, answer: str, refused: bool) -> str:
    body = esc(" ".join(source.full_text.split())) if refused else highlight(source.full_text, answer, source.rank)
    year = paper.year if paper and paper.year else ""
    return (f'<div class="page"><div class="running"><span>{esc(short_title(source.paper_title))}</span>'
            f'<span>{esc(year)}</span></div><p class="body">{body}</p><div class="folio">{source.page}</div></div>')


def reader_facts(source: Source, paper: PaperInfo | None) -> str:
    pages = f"{source.page} of {paper.num_pages}" if paper and paper.num_pages else str(source.page)
    kv = [("Paper", source.paper_title), ("Page", pages), ("Chunk", source.chunk_id or ""), ("Score", f"{source.score:.3f}")]
    return '<dl class="kv">' + "".join(f"<dt>{esc(a)}</dt><dd>{esc(b)}</dd>" for a, b in kv) + "</dl>"

# While a question runs, the previous screen's elements stay on the page, faded, until the run
# replaces them; under a streaming answer that is confusing, so they are hidden in the conversation.
HIDE_STALE = '<style>.st-key-convo [data-stale="true"] { display: none; }</style>'
