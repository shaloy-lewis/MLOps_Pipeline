"""
prepare.py  -  Data Preparation
==================================
MLOps Lesson: "Data preparation is version control for your inputs."

WHY NOT JUST LOAD DATA INLINE
------------------------------
By saving the loaded data to CSV files and checking for their existence
before reloading, we ensure:

  1. Training and monitoring use the EXACT SAME rows (not a new random draw
     each run) — reproducibility. If you delete data/train.csv, the next
     run re-downloads from UCI, giving you a fresh snapshot.

  2. You can swap in a real export by dropping a new data/train.csv — the
     pipeline picks it up automatically on next run.

  3. The data files can be hashed and tracked by the lineage module,
     so you know exactly which data any given model was trained on.

TEMPORAL SPLIT DESIGN
---------------------
Training data  : sessions from Feb through Oct  (~82% of all sessions)
Current batch  : sessions from Nov and Dec      (~18% — the holiday season)

The split simulates a production deployment:
  Phase 1 (Feb-Oct): All historical session data — used for training.
  Phase 2 (Nov-Dec): Post-deployment live traffic — arrives after the model
                     is deployed. Used for drift monitoring and batch scoring.
  Phase 3 (Dec+):    Retrain decision based on drift + performance evidence.
"""

import os

import pandas as pd

from src.data.loader import load_shopper_data
from src.config.config import DATA_DIR, SEED


def prepare_data() -> tuple:
    """
    Return (train_df, current_df) DataFrames.

    * train_df   : sessions from Feb-Oct (non-holiday browsing behaviour)
                   Saved to data/train.csv on first run.

    * current_df : sessions from Nov-Dec (holiday season — the drift batch)
                   Saved to data/current.csv on first run.

    Both DataFrames have identical schemas. The pipeline trains on train_df
    and uses current_df for drift detection and batch inference.
    """
    train_path   = os.path.join(DATA_DIR, "train.csv")
    current_path = os.path.join(DATA_DIR, "current.csv")

    if not os.path.exists(train_path) or not os.path.exists(current_path):
        print("  Preparing data (first run — downloading or generating)...")
        train_df, current_df = load_shopper_data(seed=SEED)
        os.makedirs(DATA_DIR, exist_ok=True)
        train_df.to_csv(train_path,   index=False)
        current_df.to_csv(current_path, index=False)
        print(f"  Saved: {train_path}  ({len(train_df):,} rows)")
        print(f"  Saved: {current_path}  ({len(current_df):,} rows)")
    else:
        train_df   = pd.read_csv(train_path)
        current_df = pd.read_csv(current_path)

    # Ensure Revenue is int (CSV round-trip may change dtype)
    for df in [train_df, current_df]:
        if "Revenue" in df.columns:
            df["Revenue"] = df["Revenue"].astype(int)
        if "Weekend" in df.columns:
            df["Weekend"] = df["Weekend"].astype(str)

    return train_df, current_df
