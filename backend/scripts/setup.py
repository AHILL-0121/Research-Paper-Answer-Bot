"""Set up the Research Paper Answer Bot on this machine.

Run it through ``setup.bat`` (Windows) or ``setup.sh`` (macOS/Linux) from the repository
root. It uses only the standard library, because it runs before the virtual env exists.

Steps:
  1. Detect the hardware (NVIDIA GPU and its memory) and pick a profile:
       gpu  bge-m3 and the reranker on the GPU, llama3.1 8B
       cpu  no reranker (22 s per question on CPU), llama3.2 3B (reduced quality)
  2. Create backend/.venv and install the requirements, with the CUDA or the CPU build
     of PyTorch to match (the CUDA build adds about 3.7 GB a CPU-only machine never uses).
  3. Check Ollama and offer to pull the chosen model.
  4. Ask the runtime choices (embedding provider, LLM, reranker) and save them as RPAB_*
     variables in backend/.env.local; rpab/config.py reads them as overrides.
     Experiment results such as the chunking are not offered: they are fixed by the notebook.
  5. Check the index. An unzipped submission already contains backend/artifacts/, so the
     index is reused; otherwise the script offers to build it with scripts/ingest.py.
  6. Run scripts/check_env.py.

Examples:
    setup.bat                                  interactive, defaults from the hardware
    setup.bat --yes                            accept every default (for evaluators)
    setup.bat --yes --embedding open           fully local, no API key
    setup.bat --dry-run                        show what would happen, change nothing
"""

import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent
VENV = BACKEND / ".venv"
REQUIREMENTS = BACKEND / "requirements.txt"
PYTHON_VERSION = (3, 12)  # requirements.txt is pinned on 3.12
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
# llama3.1 8B (4-bit, about 4.9 GB) plus an 8k-token KV cache (about 1 GB) needs roughly 6 GB of VRAM
MIN_VRAM_GB = 6
PROFILES = {
    "gpu": {"llm": "llama3.1:latest", "reranker": "on"},
    "cpu": {"llm": "llama3.2:3b", "reranker": "off"},
}
CPU_INDEX_MINUTES = 15  # bge-m3 on a fast laptop CPU (i7-12700H), measured 23 Oct 2026


class Setup:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.env_file = Path(args.env_file)
        self.notes: list[str] = []

    # ---- output and prompts -------------------------------------------------------------

    @staticmethod
    def step(title: str) -> None:
        print(f"\n== {title}")

    @staticmethod
    def info(message: str) -> None:
        print(f"   {message}")

    def ask(self, question: str, options: list[str], default: str) -> str:
        if self.args.yes:
            self.info(f"{question} -> {default}")
            return default
        shown = "/".join(o.upper() if o == default else o for o in options)
        while True:
            answer = input(f"   {question} [{shown}]: ").strip().lower() or default
            if answer in options:
                return answer
            print(f"   Please answer one of: {', '.join(options)}")

    def confirm(self, question: str, default: bool = True) -> bool:
        return self.ask(question, ["y", "n"], "y" if default else "n") == "y"

    def run(self, command: list[str], **kwargs) -> int:
        """Run a command, or only print it in a dry run."""
        print(f"   $ {' '.join(str(c) for c in command)}")
        if self.args.dry_run:
            return 0
        return subprocess.run([str(c) for c in command], **kwargs).returncode

    # ---- detection ----------------------------------------------------------------------

    @staticmethod
    def detect_gpu() -> tuple[str, float] | None:
        """Name and memory (GB) of the first NVIDIA GPU, or None."""
        if not shutil.which("nvidia-smi"):
            return None
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=20, check=True).stdout
        except (subprocess.SubprocessError, OSError):
            return None
        name, memory_mb = out.strip().splitlines()[0].rsplit(",", 1)
        return name.strip(), round(float(memory_mb) / 1024, 1)

    @staticmethod
    def venv_python() -> Path:
        return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def venv_json(self, code: str, extra_env: dict[str, str] | None = None) -> dict | None:
        """Run a snippet with the venv's Python and parse the JSON line it prints."""
        if not self.venv_python().exists():
            return None
        result = subprocess.run([str(self.venv_python()), "-c", code], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", cwd=BACKEND,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1",
                                     **(extra_env or {})})
        lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
        return json.loads(lines[-1]) if result.returncode == 0 and lines else None

    @staticmethod
    def ollama_exe() -> str | None:
        found = shutil.which("ollama")
        default = Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        return found or (str(default) if os.name == "nt" and default.exists() else None)

    @staticmethod
    def ollama_models(base_url: str) -> list[str] | None:
        """Pulled model tags, or None when the Ollama server does not answer."""
        try:
            with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as response:
                return [m["name"] for m in json.load(response)["models"]]
        except (OSError, ValueError, KeyError):
            return None

    # ---- .env.local ---------------------------------------------------------------------

    def read_env(self) -> dict[str, str]:
        values = {}
        if self.env_file.exists():
            for line in self.env_file.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    key, value = line.split("=", 1)
                    values[key.strip()] = value.strip()
        return values

    def write_env(self, updates: dict[str, str]) -> None:
        """Set keys in .env.local, keeping every other line (comments, other keys) as it was."""
        lines = self.env_file.read_text(encoding="utf-8").splitlines() if self.env_file.exists() else []
        remaining = dict(updates)
        for i, line in enumerate(lines):
            key = line.split("=", 1)[0].strip()
            if "=" in line and not line.lstrip().startswith("#") and key in remaining:
                lines[i] = f"{key}={remaining.pop(key)}"
        if remaining:
            lines += ["", "# Written by scripts/setup.py; rerun setup to change"]
            lines += [f"{key}={value}" for key, value in remaining.items()]
        for key, value in updates.items():
            self.info(f"{key}={'(hidden)' if 'KEY' in key else value}")
        if self.args.dry_run:
            self.info(f"(dry run: {self.env_file.name} not written)")
            return
        self.env_file.write_text("\n".join(lines).strip("\n") + "\n", encoding="utf-8")
        self.info(f"Saved to {self.env_file}")

    # ---- steps --------------------------------------------------------------------------

    def choose_profile(self) -> tuple[str, tuple[str, float] | None]:
        self.step("1. Hardware")
        gpu = self.detect_gpu()
        if gpu:
            self.info(f"GPU: {gpu[0]}, {gpu[1]} GB")
        else:
            self.info("No NVIDIA GPU found (nvidia-smi not available or no device)")
        detected = "gpu" if gpu and gpu[1] >= MIN_VRAM_GB else "cpu"
        if gpu and detected == "cpu":
            self.info(f"Under {MIN_VRAM_GB} GB of GPU memory: llama3.1 8B will not fit, so the CPU profile is suggested")
        profile = self.args.profile or self.ask("Hardware profile", ["gpu", "cpu"], detected)
        if profile == "cpu":
            self.notes.append("CPU profile: reduced answer quality (3B model, no reranking) and about 40 s per answer")
        return profile, gpu

    def install(self, gpu: tuple[str, float] | None) -> None:
        self.step("2. Python environment")
        if sys.version_info[:2] != PYTHON_VERSION:
            raise SystemExit(f"   Python {PYTHON_VERSION[0]}.{PYTHON_VERSION[1]} is required, this is "
                             f"{sys.version.split()[0]}. Install it from https://www.python.org/downloads/")
        if self.args.skip_install:
            self.info("Skipped (--skip-install)")
            return
        uv = shutil.which("uv")
        if not self.venv_python().exists():
            self.run([uv, "venv", VENV, "--python", sys.executable] if uv else [sys.executable, "-m", "venv", VENV])
        pip = ([uv, "pip", "install", "--python", self.venv_python()] if uv
               else [self.venv_python(), "-m", "pip", "install"])

        # A CUDA build of PyTorch runs on a CPU too, so it is only replaced when there is no GPU at all
        want_cuda = gpu is not None and sys.platform != "darwin"
        torch = self.venv_json("import json, torch; print(json.dumps({'cuda': torch.version.cuda}))")
        if want_cuda or (torch and torch.get("cuda") and not self.confirm(
                "PyTorch with CUDA is installed but there is no GPU. Replace it with the CPU build (frees ~3.7 GB)?")):
            self.info("Installing requirements (CUDA build of PyTorch)" if want_cuda else "Keeping the installed PyTorch")
            if self.run(pip + ["-r", REQUIREMENTS]) != 0:
                raise SystemExit("   Installing the requirements failed; see the pip output above")
            return

        # CPU build: install torch from the CPU index (PyPI on macOS), then everything else without the torch pin
        lines = REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        pin = next(line for line in lines if line.startswith("torch=="))
        version = pin.split("==")[1].split("+")[0]
        index = [] if sys.platform == "darwin" else ["--index-url", TORCH_CPU_INDEX]
        self.info("Installing the CPU build of PyTorch")
        # pip and uv both count an installed 2.x.y+cu128 as satisfying ==2.x.y, so a CUDA build is
        # replaced explicitly (torch only; its dependencies are already installed)
        replace = (["--reinstall-package", "torch"] if uv else ["--force-reinstall", "--no-deps"]) \
            if torch and torch.get("cuda") else []
        if self.run(pip + [f"torch=={version}"] + index + replace) != 0:
            raise SystemExit("   Installing PyTorch failed; see the output above")
        rest = [line for line in lines if not line.startswith(("torch==", "--extra-index-url"))]
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
            handle.write("\n".join(rest) + "\n")
        try:
            if self.run(pip + ["-r", handle.name]) != 0:
                raise SystemExit("   Installing the requirements failed; see the pip output above")
        finally:
            os.unlink(handle.name)

    def choose_runtime(self, profile: str) -> dict[str, str]:
        self.step("3. Runtime choices")
        env = self.read_env()
        key = os.getenv("GOOGLE_API_KEY") or env.get("GOOGLE_API_KEY")
        self.info("Embedding: open = bge-m3 (the notebook's choice; fully local); commercial = Gemini (ties bge-m3 "
                  "once results are reranked; needs internet and a free key, and sends questions to Google)")
        embedding = self.args.embedding or self.ask("Embedding model", ["open", "commercial"], "open")
        updates = {}
        if embedding == "commercial" and not key:
            if self.args.yes:
                raise SystemExit("   --embedding commercial needs GOOGLE_API_KEY in the environment or backend/.env.local")
            key = getpass.getpass("   Gemini API key (free at https://aistudio.google.com/apikey; input hidden): ").strip()
            if not key:
                raise SystemExit("   No key given. Rerun with --embedding open for a fully local setup.")
            updates["GOOGLE_API_KEY"] = key
        if embedding == "open":
            self.notes.append("Embeddings are local (bge-m3); the Gemini collection is not used")

        llm = self.args.llm or (PROFILES[profile]["llm"] if self.args.yes else
                                input(f"   Ollama model [{PROFILES[profile]['llm']}]: ").strip() or PROFILES[profile]["llm"])
        reranker = self.args.reranker or self.ask("Cross-encoder reranker", ["on", "off"], PROFILES[profile]["reranker"])
        if reranker == "on" and profile == "cpu":
            self.notes.append("Reranker is on without a GPU: about 22 s extra per question")
        updates.update({"RPAB_PROFILE": profile, "RPAB_EMBEDDING": embedding, "RPAB_LLM": llm,
                        "RPAB_RERANKER": reranker, "RPAB_DEVICE": "auto"})
        self.write_env(updates)
        return updates

    def ollama(self, llm: str) -> None:
        self.step("4. Ollama")
        if self.args.skip_ollama:
            self.info("Skipped (--skip-ollama)")
            return
        exe = self.ollama_exe()
        if not exe:
            self.info("Ollama is not installed. Get it from https://ollama.com/download, then rerun setup.")
            self.notes.append("Ollama missing: answers cannot be generated until it is installed")
            return
        base_url = self.read_env().get("OLLAMA_BASE_URL") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        models = self.ollama_models(base_url)
        if models is None:
            self.info(f"Ollama is installed but not answering at {base_url}. Start the Ollama app, then rerun setup.")
            self.notes.append("Ollama not running: start it before using the bot")
            return
        if llm in models or f"{llm}:latest" in models:
            self.info(f"{llm} is already pulled")
        elif self.confirm(f"Pull {llm} now (several GB)?"):
            if self.run([exe, "pull", llm]) != 0:
                self.notes.append(f"Pulling {llm} failed; run: ollama pull {llm}")

    def index(self, embedding: str, profile: str) -> None:
        self.step("5. Knowledge base index")
        if self.args.skip_index:
            self.info("Skipped (--skip-index)")
            return
        status = self.venv_json(
            "import json, sys; sys.path.insert(0, '.')\n"
            "from rpab.config import CONFIG\n"
            "from rpab.vectorstore import collection_name, get_client\n"
            "import pandas as pd\n"
            "p = CONFIG.paths\n"
            "slug = {'open': 'bge_m3', 'commercial': 'gemini_embedding_001'}[CONFIG.embedding_choice]\n"
            "name = collection_name(slug, CONFIG.chunk_spec.slug)\n"
            "chunks = len(pd.read_parquet(p.chunks)) if p.chunks.exists() else 0\n"
            "found = {c.name: c for c in get_client(p.chroma_dir).list_collections()} if p.chroma_dir.exists() else {}\n"
            "print(json.dumps({'collection': name, 'chunks': chunks, 'vectors': found[name].count() if name in found else 0,\n"
            "                  'pages': p.pages.exists(), 'pdfs': len(list(p.pdf_dir.glob('*.pdf')))}))",
            extra_env={"RPAB_EMBEDDING": embedding},  # the new choice, even in a dry run
        )
        if status is None:
            self.info("Could not inspect the index (is the environment installed?)")
            return
        if status["chunks"] and status["vectors"] == status["chunks"]:
            self.info(f"Found {status['collection']} with {status['vectors']} vectors; nothing to build")
            return
        if not status["pages"] and not status["pdfs"]:
            self.info("No index and no PDFs. Unzip the submission (it ships backend/artifacts/), or put the papers "
                      "in backend/data/pdfs/ and rerun setup.")
            self.notes.append("No index yet: the bot cannot answer until it is built")
            return
        stages = ([] if status["pages"] else ["load"]) + ["chunk", "index", "--models", embedding]
        estimate = f" (about {CPU_INDEX_MINUTES} min{' on CPU' if embedding == 'open' else ' on the Gemini free tier'})" \
            if profile == "cpu" or embedding == "commercial" else ""
        if self.confirm(f"Build the index now{estimate}?"):
            if self.run([self.venv_python(), BACKEND / "scripts" / "ingest.py", *stages], cwd=BACKEND) != 0:
                self.notes.append("Building the index failed; rerun: python scripts/ingest.py " + " ".join(stages))
        else:
            self.notes.append("Index not built; later run: python scripts/ingest.py " + " ".join(stages))

    def check(self) -> None:
        self.step("6. Environment check")
        if self.args.skip_check:
            self.info("Skipped (--skip-check)")
            return
        self.run([self.venv_python(), BACKEND / "scripts" / "check_env.py"], cwd=BACKEND)

    def main(self) -> None:
        print("Research Paper Answer Bot setup" + (" (dry run: nothing is changed)" if self.args.dry_run else ""))
        profile, gpu = self.choose_profile()
        self.install(gpu)
        choices = self.choose_runtime(profile)
        self.ollama(choices["RPAB_LLM"])
        self.index(choices["RPAB_EMBEDDING"], profile)
        self.check()
        self.step("Done")
        for note in self.notes:
            self.info(f"- {note}")
        self.info("Rerun setup at any time to change these choices.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up the Research Paper Answer Bot.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--yes", action="store_true", help="accept every default without asking")
    parser.add_argument("--dry-run", action="store_true", help="show what would happen without changing anything")
    parser.add_argument("--profile", choices=["gpu", "cpu"], help="override the detected hardware profile")
    parser.add_argument("--embedding", choices=["commercial", "open"], help="commercial = Gemini, open = bge-m3")
    parser.add_argument("--llm", help="Ollama model tag, e.g. llama3.1:latest or llama3.2:3b")
    parser.add_argument("--reranker", choices=["on", "off"])
    parser.add_argument("--skip-install", action="store_true", help="do not create the venv or install packages")
    parser.add_argument("--skip-ollama", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-check", action="store_true", help="do not run scripts/check_env.py at the end")
    parser.add_argument("--env-file", default=str(BACKEND / ".env.local"), help=argparse.SUPPRESS)
    return parser.parse_args(argv)


if __name__ == "__main__":
    Setup(parse_args()).main()
