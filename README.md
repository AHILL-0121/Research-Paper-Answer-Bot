# Research Paper Answer Bot

A retrieval-augmented generation (RAG) system over seminal Generative AI research papers.
Ask a question in plain language and get an answer written only from the papers, with the
top 3 supporting passages cited by paper title and page number.

Capstone project for the GenAI Pinnacle Plus program.

## Stack

| Part | Choice |
|---|---|
| Orchestration | LangChain (LCEL) |
| LLM | `llama3.1:8b` via Ollama (local) |
| Embeddings compared | `BAAI/bge-m3` (open-source) vs `gemini-embedding-001` (commercial, free tier) |
| Vector store | ChromaDB (persistent, cosine) |
| Keyword search | `rank_bm25` |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Apps | Streamlit, and Next.js over FastAPI |

## Layout

```text
backend/
  rpab/          Python package with all pipeline logic (config, loading, chunking, retrieval, RAG)
  api/           FastAPI service used by the Next.js app
  scripts/       Command-line entry points (environment check, ingestion)
  notebooks/     capstone.ipynb, the main deliverable; imports from rpab
  data/pdfs/     Source PDFs (not committed) and manifest.json
  data/eval/     Labelled evaluation questions
  artifacts/     Generated indexes and logs (not committed, except experiments.csv)
frontend/
  streamlit/     Streamlit app
  web/           Next.js app
slides/          Final presentation
```

## Setup

Requires Python 3.12 and [Ollama](https://ollama.com). An NVIDIA GPU with at least 6 GB is
recommended but not required. From the repository root:

```powershell
setup.bat            # Windows
./setup.sh           # macOS / Linux
```

The script detects the hardware, creates `backend/.venv`, installs the requirements with the
matching PyTorch build (CUDA or CPU), checks Ollama and pulls the model, asks the runtime
choices, reuses or builds the index, and finishes with `scripts/check_env.py`. Useful options:

| Command | Effect |
|---|---|
| `setup.bat --yes` | Accept every default without questions |
| `setup.bat --yes --embedding open` | Fully local: bge-m3 embeddings, no API key, nothing sent to Google |
| `setup.bat --dry-run` | Show what would happen without changing anything |
| `setup.bat --help` | All options |

Choices are saved as `RPAB_*` variables in `backend/.env.local` (git-ignored); rerun setup to
change them. Only machine-dependent choices are offered. Experiment results such as the
chunking stay fixed in `backend/rpab/config.py`.

### Hardware profiles

Timings measured on an i7-12700H laptop with an RTX 4060 (8 GB), with the GPU hidden for the
CPU column. A 200-token answer from a prompt of 3 passages (about 1,250 tokens):

| | GPU profile | CPU profile |
|---|---|---|
| LLM | `llama3.1:8b` | `llama3.2:3b` (reduced quality) |
| Reranker | on (0.4 s per question) | off (it would add 22 s per question) |
| First word / full answer | 3.2 s / 7.6 s | 29 s / 42 s |
| Building the index | 11 s (bge-m3) | about 15 min (bge-m3 or Gemini free tier) |

Without a GPU, Gemini embeddings are the faster option because no local embedding model has to
run. The submission zip ships a prebuilt index in `backend/artifacts/`, which setup reuses.

### Manual setup

```powershell
cd backend
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
copy .env.example .env.local    # then add GOOGLE_API_KEY (free from Google AI Studio)
ollama pull llama3.1
.venv\Scripts\python scripts\check_env.py
```

`check_env.py` confirms the imports, the GPU, the Ollama model (including the 8k context
window) and the Gemini key.

## Building the knowledge base

Put the paper PDFs in `backend/data/pdfs/`. Each file name must contain its arXiv id
(e.g. `LoRA - arXiv 2106.09685.pdf`) and the id must be listed in `backend/data/papers.json`.

```powershell
cd backend
.venv\Scripts\python scripts\ingest.py
```

This runs three stages: `load` (read and clean every page, write `data/pdfs/manifest.json`
and `artifacts/pages.parquet`, print a quality report), `chunk` (write the canonical
`artifacts/chunks.parquet`) and `index` (embed the chunks and build one Chroma collection
per embedding model under `artifacts/chroma/`). Stages can be run on their own, and
`--models open` indexes with the local model only:

```powershell
.venv\Scripts\python scripts\ingest.py chunk index --models open
```

Then open `notebooks/capstone.ipynb` and run all cells.

## Streamlit app

The Streamlit app answers in the same process, with no API needed. Start Ollama, then:

```powershell
cd frontend\streamlit
..\..\backend\.venv\Scripts\streamlit run app.py
```

It opens at `http://localhost:8501`. The page appears at once and the models load in the
background (about 20 s on the GPU, once per server). The layout follows the design mockup:
chats and settings on the left, the conversation in the middle, and a reader on the right that
shows a cited passage on its page (with the sentence the answer drew on highlighted) or the
paper library, whose checkboxes limit the search to chosen papers. Citation markers such as [1]
jump to their source card; clicking a card opens it in the reader. "Show retrieval details"
adds the query used, the weak-context flag and chunk scores to every answer. Chats are saved
in `backend/artifacts/chat.db`, so follow-up questions use the conversation (see Memory below).

Run either the Streamlit app or the API, not both: each loads the embedding model and the
reranker onto the GPU.

## Web app

The Next.js app (`frontend/web`) runs over the API. Start Ollama and the API (see below), then:

```powershell
cd frontend\web
npm install
npm run dev
```

It opens at `http://localhost:3000` with the same three panes as the Streamlit app, and adds
answers streamed word by word, citation chips that open their passage and preview it on hover,
copy buttons, a link to the cited page on arXiv, saved chats that survive a page reload, and a
light, dark or system theme. Below 1180 px the reader becomes a drawer, below 760 px the chats
and settings do too. If the API is not running, the page says how to start it and connects by
itself once it is up.

## API

The FastAPI service serves the Next.js app. Start Ollama first, then from `backend/`:

```powershell
.venv\Scripts\uvicorn api.main:app --port 8000
```

The server answers at once and loads the models in the background (about 20 s on the GPU);
`/health` reports `loading` until they are ready. Interactive docs are at
`http://localhost:8000/docs`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Pipeline state, Ollama and model availability, active models and strategies |
| `GET /papers` | The ingested papers with ids for the paper filter |
| `POST /ask` | One answer as JSON (`AskRequest` in, `AskResponse` out) |
| `POST /ask/stream` | The same answer as Server-Sent Events: `rewrite` (with a session), `sources`, `token` pieces, then `done` (or `error`) |
| `GET /sessions` | Saved chats, most recent first |
| `GET /sessions/{id}` | One chat with its messages and their sources |
| `DELETE /sessions/{id}` | Clear history: deletes the chat, its messages and sources |

**Memory.** A request with a `session_id` (8–64 letters, digits, `-` or `_`, chosen by the client,
one per browser tab) is part of a conversation: the first question creates the session, and each
follow-up is rewritten by the chat model into a standalone question from the last 6 messages
before retrieval, so "What data type does it introduce?" searches for "What data type does QLoRA
introduce?". The rewritten question is returned as `rewritten_question`. History is stored in
`backend/artifacts/chat.db` (SQLite). Without a `session_id`, each question stands alone.

Errors come back as `{"error": {"code": ..., "message": ...}}` with a message a user can act on,
for example `ollama_unavailable`, `model_missing`, `index_not_built` or `warming_up`. The API
handles one question at a time because the models share one GPU.

Tests run without models or Ollama: `.venv\Scripts\python -m pytest`.
