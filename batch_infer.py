"""
batch_infer.py  —  Batch Inference
=====================================
Scores a CSV of e-commerce sessions using the production model.
Typical use: score all sessions from the post-deployment Nov-Dec batch
and write a predictions CSV for downstream analysis or reporting.

ONLINE vs BATCH INFERENCE
--------------------------
Online (serve.py):
  Trigger   : real-time event (session active, visitor browsing)
  Latency   : must be < 100ms  (request is waiting)
  Throughput: 1 session at a time

Batch (this file):
  Trigger   : scheduled job or data file arrival (e.g., end-of-day export)
  Latency   : doesn't matter  (no one is waiting interactively)
  Throughput: score thousands of sessions in a single model.predict() call

COMMON MISTAKE
--------------
Looping over rows and calling the /predict endpoint one at a time is NOT
batch inference — it is online inference with scheduling overhead.
True batch inference calls model.predict(X) once on the full matrix.
LightGBM releases the GIL and parallelises across trees automatically.

HOW TO RUN
----------
    # Score the default post-deployment batch (Nov-Dec sessions):
    python batch_infer.py

    # Score a custom file:
    python batch_infer.py --input data/current.csv --output data/scored.csv
"""

import argparse
import time
import datetime

import numpy as np
import pandas as pd

from src.utils.paths import create_directories
from src.features.build_features import build_features
from src.registry.versioning import get_production_model
from src.monitoring.logging import log_predictions, log_latency


def run_batch_inference(input_path: str, output_path: str):
    print("\n" + "*" * 60)
    print("  BATCH INFERENCE — E-Commerce Sessions")
    print("*" * 60)

    # ── load data ─────────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading sessions from '{input_path}'...")
    df = pd.read_csv(input_path)
    print(f"  {len(df):,} sessions loaded.  Columns: {len(df.columns)}")

    # ── load model ────────────────────────────────────────────────────────────
    print("\n[2/4] Loading production model from registry...")
    model, meta = get_production_model()
    print(f"  Model version    : {meta['version']}")
    print(f"  F1 at deployment : {meta['metrics'].get('f1', 'n/a')}")

    # ── build features ────────────────────────────────────────────────────────
    # training=False → use saved medians from training, not from this batch.
    # This is the critical distinction: inference must use the same transforms
    # that were applied at training time, not re-fit on the current batch.
    print("\n[3/4] Building features using training-time pipeline...")
    X, y_true, feature_cols = build_features(df, training=False)

    # Align to the exact column set the model was trained on
    if "feature_cols" in meta:
        trained_cols = meta["feature_cols"]
        X_df = pd.DataFrame(X, columns=feature_cols)
        for c in trained_cols:
            if c not in X_df.columns:
                X_df[c] = 0.0
        X = X_df[trained_cols].values.astype(np.float32)
        feature_cols = trained_cols

    # ── score ─────────────────────────────────────────────────────────────────
    print(f"\n[4/4] Scoring {len(df):,} sessions...")
    start   = time.time()
    preds   = model.predict(X)
    probs   = model.predict_proba(X)[:, 1]
    elapsed = time.time() - start

    log_latency(start)
    log_predictions(preds)

    # ── attach results to DataFrame ───────────────────────────────────────────
    df["predicted_purchase"]      = preds
    df["purchase_probability"]    = probs.round(4)
    df["intent_level"] = pd.cut(
        probs,
        bins   = [0, 0.25, 0.55, 1.01],
        labels = ["low", "medium", "high"],
        right  = False,
    )
    df["model_version"] = meta["version"]
    df["scored_at"]     = str(datetime.datetime.now())

    df.to_csv(output_path, index=False)

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n  Sessions scored      : {len(df):,}")
    print(f"  Elapsed time         : {elapsed*1000:.1f} ms  "
          f"({elapsed/len(df)*1000:.3f} ms per session)")
    print(f"  Predicted purchase % : {preds.mean():.1%}")
    print(f"  Avg purchase prob    : {probs.mean():.3f}")
    print(f"\n  Intent breakdown:")
    intent_counts = df["intent_level"].value_counts()
    for level in ["low", "medium", "high"]:
        n   = int(intent_counts.get(level, 0))
        pct = n / len(df) * 100
        bar = "|" * int(pct / 2)
        print(f"    {level:<8} {n:>5} ({pct:5.1f}%)  {bar}")

    print(f"\n  Results saved to : {output_path}")
    print(f"  Logs updated     : logs/predictions.csv  |  logs/latency.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch inference: score a CSV of e-commerce session records."
    )
    parser.add_argument(
        "--input",
        default="data/current.csv",
        help="Path to input CSV (default: data/current.csv — the post-deployment Nov-Dec batch)",
    )
    parser.add_argument(
        "--output",
        default="data/batch_predictions.csv",
        help="Path to output CSV (default: data/batch_predictions.csv)",
    )
    args = parser.parse_args()

    create_directories()
    run_batch_inference(args.input, args.output)
