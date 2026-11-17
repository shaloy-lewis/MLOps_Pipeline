"""
tracking/experiment.py  –  Experiment Tracking
=================================================
MLOps Lesson: "If you don't log it, you can't compare it."

WHY EXPERIMENT TRACKING
------------------------
Every training run is an experiment:
  * Different hyperparameters (n_estimators, learning_rate, num_leaves)
  * Different feature sets (e.g., with or without SpecialDay)
  * Different training data versions (Feb-Oct only vs pooled with Nov-Dec)

Without tracking, answering "which configuration gave the best recall?"
means re-running everything from memory.  With tracking, it is a
five-second query of a CSV or database.

HOW THIS SCALES
---------------
This CSV-based tracker works fine for one person running a few dozen
experiments.  It breaks when:
  * Multiple engineers run experiments in parallel (CSV write conflicts)
  * You have thousands of runs (CSV becomes slow to query)
  * You want to visualise training curves, not just final metrics

The production solution is MLflow (open-source, self-hosted) or a cloud
service (Weights & Biases, Neptune, Comet ML).  They all implement the
same concept: "append one row per run with params + metrics + metadata."
"""

import os
import datetime

import pandas as pd

from src.config.config import LOG_DIR


def log_experiment(params: dict, metrics: dict, extra: dict = None):
    """
    Append one experiment run to logs/experiments.csv.

    Parameters
    ----------
    params  : hyperparameters used in this run  (e.g. {"n_estimators": 100})
    metrics : evaluation metrics                (e.g. {"accuracy": 0.85, "f1": 0.79})
    extra   : any other metadata               (e.g. {"dataset": "train.csv"})
    """
    row = {
        **params,
        **metrics,
        "timestamp": str(datetime.datetime.now()),
    }
    if extra:
        row.update(extra)

    path = os.path.join(LOG_DIR, "experiments.csv")
    df   = pd.DataFrame([row])

    if not os.path.exists(path):
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode="a", header=False, index=False)

    print(f"  Logged to {path}")
    print(f"  Params : {params}")
    print(f"  Metrics: { {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()} }")
