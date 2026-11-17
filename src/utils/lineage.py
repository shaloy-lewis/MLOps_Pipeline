"""
lineage.py  –  Lineage Tracking
=================================
MLOps Lesson: "If you can't reproduce a model, you can't debug it."

WHAT IS LINEAGE
---------------
Lineage is the chain of provenance for a trained model:

    Dataset version  →  Code version  →  Environment  →  Model artifact

When a model starts making wrong predictions in production, the first
question is always: "What changed?"  Without lineage, answering that
question means archaeology through logs, git blame, and guesswork.

With lineage, you open logs/lineage.json, find the run that produced the
production model, and immediately see:

  * Exactly which CSV files were used (via SHA-256 hash)
  * Which git commit the training code was at
  * Which Python version and library versions were active
  * Which model file was produced

WHAT LINEAGE IS NOT
-------------------
Lineage is not a backup system.  It is a pointer map.  The data and model
files still need to be stored somewhere safe.  Lineage just tells you which
versions to look for.

PRODUCTION TOOLS THAT DO THIS
------------------------------
  * MLflow: automatic artifact and parameter logging
  * DVC (Data Version Control): git-like versioning for datasets
  * Vertex AI Metadata: Google Cloud's lineage service
  * SageMaker Experiments: AWS equivalent

We implement it manually here so you understand what those tools are
abstracting.
"""

import os
import sys
import json
import hashlib
import datetime
import subprocess
import importlib.metadata

from src.config.config import LOG_DIR, SEED

LINEAGE_PATH     = os.path.join(LOG_DIR, "lineage.json")
TRACKED_PACKAGES = ["scikit-learn", "numpy", "pandas", "fastapi", "uvicorn"]


# ── helper utilities ──────────────────────────────────────────────────────────

def _file_hash(path: str) -> str:
    """
    Compute a short SHA-256 fingerprint of a file.

    A 12-character prefix is unique enough for change detection but short
    enough to be readable in a log.  If the hash changes, the file changed.
    """
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:12]
    except FileNotFoundError:
        return "file-not-found"


def _git_hash() -> str:
    """
    Return the current HEAD commit hash (short form).
    Returns 'no-git' if not running inside a git repository.

    WHY GIT HASH NOT JUST A TIMESTAMP
    ------------------------------------
    Two runs at different timestamps might use identical code.
    Two runs at the same timestamp (unlikely but possible) might use
    different code.  The commit hash is the unambiguous code identifier.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        return result.stdout.strip() or "no-git"
    except Exception:
        return "no-git"


def _package_versions() -> dict:
    """Return installed version strings for key packages."""
    versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


# ── main entry point ──────────────────────────────────────────────────────────

def record_lineage(train_path: str, current_path: str, model_path: str) -> dict:
    """
    Build and save a lineage record linking this training run to its inputs
    and environment.

    Parameters
    ----------
    train_path   : path to the training CSV (e.g. data/train.csv)
    current_path : path to the current/production CSV
    model_path   : path to the saved .pkl model file

    Returns
    -------
    dict  – the lineage record (also appended to logs/lineage.json)
    """
    record = {
        "run_id":     datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "timestamp":  str(datetime.datetime.now()),
        "seed":       SEED,
        "git_commit": _git_hash(),
        "python":     sys.version.split()[0],
        "packages":   _package_versions(),
        "data": {
            "train": {
                "path":   train_path,
                "sha256": _file_hash(train_path),
            },
            "current": {
                "path":   current_path,
                "sha256": _file_hash(current_path),
            },
        },
        "model_output": model_path,
    }

    # Append to the lineage log (one record per training run)
    existing = []
    if os.path.exists(LINEAGE_PATH):
        with open(LINEAGE_PATH) as f:
            existing = json.load(f)

    existing.append(record)
    with open(LINEAGE_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    # Summary print
    print(f"  Run ID   : {record['run_id']}")
    print(f"  Git      : {record['git_commit']}")
    print(f"  Python   : {record['python']}")
    print(f"  sklearn  : {record['packages'].get('scikit-learn', 'n/a')}")
    print(f"  Train    : {train_path}  [{record['data']['train']['sha256']}]")
    print(f"  Current  : {current_path}  [{record['data']['current']['sha256']}]")
    print(f"  Model    : {model_path}")
    print(f"  Written  : {LINEAGE_PATH}")

    return record
