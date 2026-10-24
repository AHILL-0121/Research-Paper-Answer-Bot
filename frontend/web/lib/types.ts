// Mirrors of the Pydantic models in backend/rpab/schemas.py and backend/api/main.py.
// Keep the field names and optionality in step with those files.

export type Strategy = "dense" | "mmr" | "hybrid" | "hybrid_rerank" | "multi_query";

export interface Source {
  rank: number; // 1 to final_k; the answer cites it as [rank]
  paper_title: string;
  page: number;
  snippet: string;
  full_text: string;
  score: number;
  origin: "paper" | "web";
  url?: string | null; // web sources only (S3)
  chunk_id?: string | null;
  paper_id?: string | null;
}

export interface AskRequest {
  question: string; // 3 to 1,000 characters
  session_id?: string | null; // 8-64 of [A-Za-z0-9_-]; omit for a question without memory
  strategy?: Strategy;
  paper_ids?: string[] | null;
  final_k?: number;
  web_fallback?: boolean; // S3: search the web when the papers' best passage is weak
}

export interface AskResponse {
  answer: string;
  sources: Source[];
  refused: boolean;
  rewritten_question?: string | null;
  strategy_used: string;
  latency_ms: number;
  used_web_fallback: boolean; // at least one source is a web result
  web_note?: string | null; // why the web was searched and what came of it
  weak_context: boolean;
  retrieval_ms?: number | null;
  rewrite_ms?: number | null;
  first_token_ms?: number | null;
  removed_citations: string[];
}

export interface PaperInfo {
  paper_id: string;
  paper_title: string;
  year?: number | null;
  arxiv_id?: string | null;
  url?: string | null;
  num_pages?: number | null;
  num_chunks?: number | null;
}

export interface ErrorInfo {
  code: string;
  message: string;
}

export interface Health {
  status: "loading" | "ready" | "error";
  error?: ErrorInfo | null;
  ollama_running: boolean;
  llm_model: string;
  llm_available: boolean;
  embedding_model: string;
  reranker: boolean;
  default_strategy: Strategy;
  strategies: Strategy[];
  profile: string;
  num_papers: number;
  chunking: string;
  weak_threshold?: number | null;
  web_search: boolean; // the S3 web fallback can run
}

export interface SessionSummary {
  session_id: string;
  title?: string | null;
  created_at: string;
  last_active: string;
  num_messages: number;
}

export interface ChatMessage {
  message_id: number;
  role: "user" | "assistant";
  content: string;
  rewritten_question?: string | null;
  refused: boolean;
  strategy?: string | null;
  latency_ms?: number | null;
  created_at: string;
  sources: Source[];
}

export interface SessionDetail extends SessionSummary {
  messages: ChatMessage[];
}

// Server-Sent Events from POST /ask/stream
export type StreamEvent =
  | { event: "rewrite"; rewritten_question: string; rewrite_ms: number | null }
  | { event: "sources"; sources: Source[]; weak_context: boolean; retrieval_ms: number; web_note?: string | null }
  | { event: "token"; text: string }
  | { event: "done"; response: AskResponse }
  | { event: "error"; error: ErrorInfo };
