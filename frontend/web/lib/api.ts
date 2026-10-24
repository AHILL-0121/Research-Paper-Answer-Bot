// Client for the FastAPI service (backend/api/main.py). The browser calls it directly; the API's
// CORS settings allow the Next.js dev server on port 3000.

import type {
  AskRequest, ErrorInfo, Health, PaperInfo, SessionDetail, SessionSummary, StreamEvent,
} from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

/** An error with the API's stable code, or `api_unreachable` when the server did not answer. */
export class ApiError extends Error {
  constructor(public code: string, message: string, public status?: number) {
    super(message);
  }
}

export const unreachable = () =>
  new ApiError("api_unreachable", `Can't reach the API at ${API_URL}. Start it from the backend folder with ` +
    "`.venv\\Scripts\\uvicorn api.main:app --port 8000`, then try again.");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, init);
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw unreachable();
  }
  if (!res.ok) throw await toApiError(res);
  return (res.status === 204 ? undefined : await res.json()) as T;
}

async function toApiError(res: Response): Promise<ApiError> {
  try {
    const body = (await res.json()) as { error?: ErrorInfo };
    if (body.error) return new ApiError(body.error.code, body.error.message, res.status);
  } catch {
    // not JSON: fall through
  }
  return new ApiError("http_error", `The API answered with status ${res.status}.`, res.status);
}

export const getHealth = () => request<Health>("/health");
export const getPapers = () => request<PaperInfo[]>("/papers");
export const listSessions = () => request<SessionSummary[]>("/sessions");
export const getSession = (id: string) => request<SessionDetail>(`/sessions/${encodeURIComponent(id)}`);
export const deleteSession = (id: string) =>
  request<void>(`/sessions/${encodeURIComponent(id)}`, { method: "DELETE" });

/** POST /ask/stream and hand each Server-Sent Event to `onEvent` as it arrives. Validation and
 * "not ready" errors come back as HTTP errors before the stream starts and are thrown as ApiError;
 * failures during the answer arrive as an `error` event. */
export async function askStream(body: AskRequest, onEvent: (e: StreamEvent) => void, signal: AbortSignal) {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw unreachable();
  }
  if (!res.ok || !res.body) throw await toApiError(res);

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value.replace(/\r\n?/g, "\n");
    let end: number;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const parsed = parseEvent(block);
      if (parsed) onEvent(parsed);
    }
  }
}

function parseEvent(block: string): StreamEvent | null {
  let kind = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith(":")) continue; // keep-alive comment
    const colon = line.indexOf(":");
    const field = colon < 0 ? line : line.slice(0, colon);
    const value = colon < 0 ? "" : line.slice(colon + 1).replace(/^ /, "");
    if (field === "event") kind = value;
    else if (field === "data") data.push(value);
  }
  if (!data.length) return null;
  const payload = JSON.parse(data.join("\n"));
  switch (kind) {
    case "done":
      return { event: "done", response: payload };
    case "error":
      return { event: "error", error: payload };
    case "rewrite":
    case "sources":
    case "token":
      return { event: kind, ...payload };
    default:
      return null;
  }
}
