"""
versioning.py  –  Model Registry
==================================
MLOps Lesson: "A model file is not a deployed model until it's registered."

WHY A REGISTRY
--------------
Without a registry, you have a folder of .pkl files with timestamps in the
name.  Questions that become impossible to answer:

  * "Which model is currently serving production traffic?"
  * "When was model v3 promoted?  Who approved it?"
  * "If the current model is wrong, what was the previous one?"

A registry is the source of truth.  The serving layer reads from it;
the training pipeline writes to it.  This is the contract between training
and inference.

HOW IT WORKS
------------
Every model goes through a lifecycle:

    candidate  →  staging  →  production
                                  ↓
                              (retired when next model is promoted)

`candidate` : freshly trained, evaluation gates passed
`staging`   : passes integration tests (in a real system)
`production`: serving live traffic
`retired`   : previously production, demoted by a newer model

The registry is a JSON file (registry.json) so you can open it with any
text editor and understand it.  In real systems (MLflow, SageMaker Model
Registry, Vertex AI) this is a database with a REST API.

PRODUCTION NOTE
---------------
In production, promoting to 'production' would require a human approval
step or a successful canary rollout.  Here we automate it to keep the
pipeline runnable, but the structure supports a manual gate.
"""

import os
import json
import pickle
import datetime

from src.config.config import MODEL_DIR, SEED

REGISTRY_PATH = os.path.join(MODEL_DIR, "registry.json")


# ── registry I/O ──────────────────────────────────────────────────────────────

def _load_registry() -> dict:
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {"models": []}


def _save_registry(registry: dict):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


# ── save ──────────────────────────────────────────────────────────────────────

def save_model(model, metrics: dict, feature_cols: list) -> tuple:
    """
    Persist a trained model to disk and register it as a 'candidate'.

    The model file is named with a timestamp and F1 score so the filename
    itself is informative (no need to open the registry to spot the best model).

    Parameters
    ----------
    model        : fitted sklearn model
    metrics      : dict with accuracy, precision, recall, f1
    feature_cols : list of feature column names (needed by the serving layer
                   to reconstruct the correct input shape)

    Returns
    -------
    (path, version) tuple
      path    : absolute path to the saved .pkl
      version : version string (timestamp) used as the registry key
    """
    version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname   = f"model_{version}_f1_{round(metrics.get('f1', 0), 3):.3f}.pkl"
    path    = os.path.join(MODEL_DIR, fname)

    with open(path, "wb") as f:
        pickle.dump(model, f)

    entry = {
        "version":      version,
        "model_file":   fname,
        "metrics":      metrics,
        "feature_cols": feature_cols,
        "seed":         SEED,
        "status":       "candidate",
        "created_at":   str(datetime.datetime.now()),
        "promoted_at":  None,
        "retired_at":   None,
    }

    registry = _load_registry()
    registry["models"].append(entry)
    _save_registry(registry)

    print(f"  Saved  : {fname}")
    print(f"  Status : candidate")
    print(f"  Metrics: acc={metrics.get('accuracy', 0):.3f}  "
          f"prec={metrics.get('precision', 0):.3f}  "
          f"rec={metrics.get('recall', 0):.3f}  "
          f"f1={metrics.get('f1', 0):.3f}")

    return path, version


# ── promote ───────────────────────────────────────────────────────────────────

def promote(version: str, to_status: str = "staging") -> dict:
    """
    Advance a model version through the registry lifecycle.

    Allowed transitions
    -------------------
    candidate  → staging
    staging    → production   (retires the current production model)
    production → retired      (only via a newer promotion)

    Parameters
    ----------
    version   : the version string (YYYYMMDD_HHMMSS) to promote
    to_status : 'staging' or 'production'

    Returns the updated registry entry.
    """
    if to_status not in ("staging", "production"):
        raise ValueError(f"Invalid status '{to_status}'. Use 'staging' or 'production'.")

    registry = _load_registry()

    # Retire the current production model before crowning a new one
    if to_status == "production":
        for m in registry["models"]:
            if m["status"] == "production":
                m["status"]     = "retired"
                m["retired_at"] = str(datetime.datetime.now())
                print(f"  Retired : {m['model_file']}  (previously production)")

    # Promote the target version
    for m in registry["models"]:
        if m["version"] == version:
            m["status"]      = to_status
            m["promoted_at"] = str(datetime.datetime.now())
            _save_registry(registry)
            print(f"  Promoted: {m['model_file']}  ->  status='{to_status}'")
            return m

    raise KeyError(
        f"Version '{version}' not found in registry.  "
        "Available versions: " +
        str([m["version"] for m in registry["models"]])
    )


# ── load production model ─────────────────────────────────────────────────────

def get_production_model():
    """
    Load and return the model currently marked as 'production' in the registry.

    WHY NOT JUST LOAD THE NEWEST .pkl FILE
    ----------------------------------------
    The newest file on disk might be a candidate that failed evaluation,
    a model being tested, or a corrupted write.  The registry is the
    authoritative record of what is approved for production.

    Returns
    -------
    (model, registry_entry) tuple

    Raises RuntimeError if no production model exists.
    """
    registry = _load_registry()

    # Sort newest-first so that if somehow two entries are 'production',
    # we get the most recently promoted one
    for m in sorted(registry["models"], key=lambda x: x.get("promoted_at") or "", reverse=True):
        if m["status"] == "production":
            pkl_path = os.path.join(MODEL_DIR, m["model_file"])
            if not os.path.exists(pkl_path):
                raise FileNotFoundError(
                    f"Registry points to '{pkl_path}' but the file does not exist.  "
                    "The model file may have been deleted manually."
                )
            with open(pkl_path, "rb") as f:
                model = pickle.load(f)
            print(f"  Loaded production model: {m['model_file']}")
            return model, m

    raise RuntimeError(
        "No production model found in registry.  "
        "Run `python main.py` to train and register a model."
    )


# ── list all versions ─────────────────────────────────────────────────────────

def list_models() -> list:
    """Return all registry entries sorted newest-first."""
    registry = _load_registry()
    return sorted(registry["models"], key=lambda x: x["created_at"], reverse=True)
