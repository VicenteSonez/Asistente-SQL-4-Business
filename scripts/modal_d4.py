"""Deliverable 4 on Modal (modal.com): training, evaluation and the video demo on one H100.

The measured D4 run used this script. Every function clones the repository at a pushed ref (branch,
tag or commit), runs the same scripts as the Colab notebook, keeps a copy of its outputs in the
Volume ``sql4business-d4-results`` and returns them, so they land in this checkout at the same paths
(``results/deliverable4/``, ``docs/deliverable4_results.tex``). Adapter weights (*.safetensors) stay
in the Volume and on Hugging Face, not in git.

    modal run scripts/modal_d4.py::train --smoke                 # short training check, publishes nothing
    modal run scripts/modal_d4.py::train                         # QLoRA + push to Hugging Face
    modal run scripts/modal_d4.py::evaluate --limit 2 --configs agentes_ft   # smoke test (not saved here)
    modal run scripts/modal_d4.py::evaluate                      # four systems + control, 27 questions
    modal run scripts/modal_d4.py::demo --ids q12,q11-p          # base vs fine-tuned Planner, live messages

Needs ``modal setup`` and, for publishing, a Modal secret ``huggingface`` with HF_TOKEN (write access).
Costs GPU time: no container is kept warm, model weights are cached in the Volume ``sql4business-hf-cache``.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/VicenteSonez/Asistente-SQL-4-Business.git"
DEFAULT_REF = "deliverable4"
HF_REPO = "VicenteSonez/sql4business-planificador-qlora"
GPU = "H100"
WORKDIR = "/root/repo"


def _pins():
    """requirements.txt (the pinned versions of the measured runs), read where the image is built."""

    if not modal.is_local():
        return []
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(*_pins())
    .env({"HF_HOME": "/hf-cache", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
)
hf_cache = modal.Volume.from_name("sql4business-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("sql4business-d4-results", create_if_missing=True)
app = modal.App("sql4business-d4", image=image)

GPU_OPTIONS = dict(gpu=GPU, cpu=8.0, memory=32768, scaledown_window=2,
                   volumes={"/hf-cache": hf_cache, "/vol": results_volume})


# --------------------------------------------------------------------------- remote side


def _checkout(ref):
    subprocess.run(["git", "clone", "--quiet", REPO_URL, WORKDIR], check=True)
    subprocess.run(["git", "-C", WORKDIR, "checkout", "--quiet", ref], check=True)
    commit = subprocess.run(["git", "-C", WORKDIR, "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.strip()
    print(f"ref {ref} -> commit {commit} | GPU {gpu}", flush=True)
    return commit


def _run(*args):
    print("$ python", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=WORKDIR, check=True)


def _collect(*relative):
    """Files under the given repo paths, without adapter weights; also copied to the results Volume."""

    import shutil

    stamp = time.strftime("%Y%m%d-%H%M%S")
    files = {}
    for rel in relative:
        base = Path(WORKDIR) / rel
        paths = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file()) if base.exists() else []
        for path in paths:
            key = path.relative_to(WORKDIR).as_posix()
            backup = Path("/vol") / stamp / key
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            if not key.endswith(".safetensors"):
                files[key] = path.read_bytes()
    results_volume.commit()
    hf_cache.commit()
    print(f"saved a copy in the Volume sql4business-d4-results:/{stamp} ({len(files)} files returned)", flush=True)
    return files


@app.function(timeout=2 * 3600, secrets=[modal.Secret.from_name("huggingface")], **GPU_OPTIONS)
def train_remote(ref, smoke):
    _checkout(ref)
    if smoke:
        _run("scripts/train_planner.py", "--limit", "16", "--epochs", "1", "--eval-every", "1",
             "--output-dir", "results/deliverable4/prueba_adapter")
        return _collect("results/deliverable4/prueba_adapter")
    _run("scripts/train_planner.py", "--push-to-hub", HF_REPO)
    return _collect("results/deliverable4/adapter", "results/deliverable4/adapter_hf.json")


@app.function(timeout=4 * 3600, **GPU_OPTIONS)
def evaluate_remote(ref, limit, configs):
    _checkout(ref)
    args = ["scripts/evaluate_d4.py"]
    if limit:
        args += ["--limit", str(limit)]
    if configs:
        args += ["--configs", configs]
    _run(*args)
    if limit:
        return _collect("results/deliverable4/prueba")
    return _collect("results/deliverable4", "docs/deliverable4_results.tex")


@app.function(timeout=3600, **GPU_OPTIONS)
def demo_remote(ref, ids):
    _checkout(ref)
    for question_id in ids:
        _run("scripts/run_agents.py", "--id", question_id, "--compare", "--no-baseline")
    return _collect("results/deliverable4/demo")


# --------------------------------------------------------------------------- local side


def _save(files, target):
    for key, data in files.items():
        path = Path(target) / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print(f"wrote {len(files)} files under {target}")


def _scratch(name):
    folder = Path(os.environ.get("TEMP", "/tmp")) / "sql4business-d4" / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


@app.local_entrypoint()
def train(ref: str = DEFAULT_REF, smoke: bool = False):
    files = train_remote.remote(ref, smoke)
    _save(files, _scratch("smoke-train") if smoke else ROOT)


@app.local_entrypoint()
def evaluate(ref: str = DEFAULT_REF, limit: int = 0, configs: str = ""):
    files = evaluate_remote.remote(ref, limit, configs)
    _save(files, _scratch("smoke-eval") if limit else ROOT)


@app.local_entrypoint()
def demo(ref: str = DEFAULT_REF, ids: str = "q12,q11-p"):
    files = demo_remote.remote(ref, [i.strip() for i in ids.split(",") if i.strip()])
    _save(files, ROOT)
