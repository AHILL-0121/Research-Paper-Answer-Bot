"""Smoke-test the environment before any pipeline work.

Checks, in order: library imports, GPU visibility, the local Ollama model
(including that ``num_ctx`` really enlarges the context window), and the
Gemini embedding key. Run from the backend folder:

    .venv\\Scripts\\python scripts\\check_env.py
"""

import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rpab.config import CONFIG  # noqa: E402

IMPORTS = [
    "langchain", "langchain_core", "langchain_community", "langchain_classic",
    "langchain_text_splitters", "langchain_experimental", "langchain_ollama",
    "langchain_huggingface", "langchain_chroma", "langchain_google_genai",
    "chromadb", "sentence_transformers", "rank_bm25", "pypdf", "pdfplumber",
    "tiktoken", "pandas", "pyarrow", "numpy", "matplotlib", "joblib", "pydantic",
    "streamlit", "fastapi", "uvicorn", "torch",
]


def report(name: str, ok: bool | None, detail: str = "") -> bool:
    """Print one aligned result line. ``ok=None`` means the check was skipped."""
    status = {True: "PASS", False: "FAIL", None: "SKIP"}[ok]
    print(f"[{status}] {name:<28} {detail}")
    return ok is not False


def check_imports() -> bool:
    failed = []
    for module in IMPORTS:
        try:
            importlib.import_module(module)
        except Exception as exc:  # report every broken import, not just the first
            failed.append(f"{module} ({exc.__class__.__name__})")
    return report("imports", not failed, ", ".join(failed) or f"{len(IMPORTS)} modules")


def check_gpu() -> bool:
    import torch

    if not torch.cuda.is_available():
        if CONFIG.profile == "cpu":
            return report("cuda", None, f"CPU profile, torch {torch.__version__}")
        return report("cuda", False, f"torch {torch.__version__} sees no GPU")
    props = torch.cuda.get_device_properties(0)
    return report("cuda", True, f"{props.name}, {props.total_memory / 2**30:.1f} GiB, torch {torch.__version__}")


def check_ollama() -> bool:
    """Send a ~6k-token prompt; if num_ctx were ignored, Ollama would truncate it."""
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model=CONFIG.llm_model,
        base_url=CONFIG.ollama_base_url,
        temperature=CONFIG.llm_temperature,
        num_ctx=CONFIG.llm_num_ctx,
        num_predict=5,
    )
    filler = "The transformer uses attention. " * 1000
    start = time.perf_counter()
    try:
        reply = llm.invoke(f"{filler}\n\nReply with the single word OK.")
    except Exception as exc:
        return report("ollama", False, f"{exc.__class__.__name__}: {exc}")
    seconds = time.perf_counter() - start
    prompt_tokens = reply.response_metadata.get("prompt_eval_count", 0)
    ok_reply = report("ollama model", True, f"{CONFIG.llm_model} replied {reply.content.strip()!r} in {seconds:.1f} s")
    ok_ctx = report("ollama num_ctx", prompt_tokens > 4096,
                    f"{prompt_tokens} prompt tokens evaluated (num_ctx={CONFIG.llm_num_ctx})")
    return ok_reply and ok_ctx


def check_gemini() -> bool:
    if not CONFIG.google_api_key:
        return report("gemini embeddings", None, "GOOGLE_API_KEY not set in backend/.env.local or backend/.env")
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    try:
        emb = GoogleGenerativeAIEmbeddings(model=CONFIG.embedding_models["commercial"],
                                           google_api_key=CONFIG.google_api_key)
        vector = emb.embed_query("multi-head attention")
    except Exception as exc:
        return report("gemini embeddings", False, f"{exc.__class__.__name__}: {exc}")
    return report("gemini embeddings", True, f"{CONFIG.embedding_models['commercial']}, dim {len(vector)}")


def main() -> int:
    print(f"Python {sys.version.split()[0]}  |  backend: {CONFIG.paths.data_dir.parent}\n")
    results = [check_imports()]
    if results[0]:
        results += [check_gpu(), check_ollama(), check_gemini()]
    print("\nAll checks passed." if all(results) else "\nSome checks failed.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
