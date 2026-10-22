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

## Setup (Windows)

Requires Python 3.12, an NVIDIA GPU (optional but recommended) and [Ollama](https://ollama.com).

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
