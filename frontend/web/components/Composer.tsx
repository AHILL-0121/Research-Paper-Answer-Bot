"use client";

import { useEffect, useRef } from "react";
import { Icon } from "./Icon";
import { STRATEGY_LABELS } from "@/lib/text";
import type { Strategy } from "@/lib/types";

type Props = {
  draft: string;
  disabled: boolean;
  strategy: Strategy;
  strategies: Strategy[];
  scopeLabel: string;
  model: string;
  onDraft: (text: string) => void;
  onSubmit: () => void;
  onStrategy: (s: Strategy) => void;
  onScope: () => void;
};

export function Composer({ draft, disabled, strategy, strategies, scopeLabel, model, ...on }: Props) {
  const box = useRef<HTMLTextAreaElement>(null);

  // Grow with the text up to the CSS max-height; refocus when an answer finishes
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [draft]);
  useEffect(() => {
    if (!disabled) box.current?.focus({ preventScroll: true });
  }, [disabled]);

  return (
    <form className="composer" onSubmit={(e) => { e.preventDefault(); on.onSubmit(); }}>
      <label htmlFor="q" className="sr-only">Question</label>
      <textarea
        id="q"
        ref={box}
        rows={1}
        value={draft}
        disabled={disabled}
        placeholder="Ask a question about the papers..."
        onChange={(e) => on.onDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); on.onSubmit(); }
        }}
      />
      <div className="tools">
        <label className="tool" title="Retrieval strategy">
          <Icon name="layers" />
          <span className="sr-only">Retrieval strategy</span>
          <select value={strategy} onChange={(e) => on.onStrategy(e.target.value as Strategy)}>
            {strategies.map((s) => <option key={s} value={s}>{STRATEGY_LABELS[s]}</option>)}
          </select>
        </label>
        <button type="button" className="tool" title="Choose which papers to search" onClick={on.onScope}>
          <Icon name="filter" /><span>{scopeLabel}</span>
        </button>
        <span className="grow" style={{ flex: 1 }} />
        <span className="model">{model}</span>
        <button type="submit" className="send" aria-label="Send question" disabled={disabled || draft.trim().length < 3}>
          <Icon name="up" />
        </button>
      </div>
    </form>
  );
}
