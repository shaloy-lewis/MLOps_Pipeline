"""
run_pipeline.py  -  ML Pipeline Orchestrator
=============================================
Executes all pipeline stages in order. A single call to run_pipeline()
reproduces the full training, evaluation, registration, and monitoring run.

STAGES
------
  0   Initialisation     set seed, create directories
  1   Data loading       load train (Feb-Oct) + current (Nov-Dec) CSVs
  2   Validation         abort if data quality checks fail
  3   Features           impute NaNs, one-hot encode, align columns
  4   Train/test split   80/20 stratified split within training months
  5   Training           fit LightGBM on 80% split with balanced class weights
  6   Evaluation         F1 minority class + AUC-ROC + AUC-PR on TEST set
  6b  Slice evaluation   per-VisitorType and per-Month performance
  7   Gates              block promotion if any threshold fails
  8   Experiment log     append run to experiments.csv
  9   Registry           save .pkl, register candidate -> staging -> production
  10  Drift detection    compare train vs current (holiday) distribution
  11  Predictions        score current batch + log latency + log predictions
  12  Lineage            link model -> data -> code -> environment

IN A PRODUCTION SYSTEM
-----------------------
This orchestrator would be replaced by Airflow, Prefect, Kubeflow Pipelines,
or a managed pipeline service (Vertex AI Pipelines, SageMaker Pipelines).
The stage order and concepts are identical; only the execution infrastructure
changes.
"""

import os
import time

import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.reproducibility import set_seed
from src.utils.paths import create_directories
from src.utils.lineage import record_lineage

from src.data.prepare import prepare_data
from src.data.validate import ShoppersValidator

from src.features.build_features import build_features

from src.models.train import train_model
from src.models.evaluate import evaluate, gate, slice_evaluation
from src.models.predict import predict

from src.tracking.experiment import log_experiment
from src.registry.versioning import save_model, promote

from src.monitoring.drift import detect_drift
from src.monitoring.logging import log_predictions, log_latency

from src.config.config import DATA_DIR, SEED


def _banner(stage_num, title):
    """Print a consistent stage header so the output is easy to scan."""
    print("\n" + "=" * 65)
    print(f"  Stage {stage_num}: {title}")
    print("=" * 65)


def run_pipeline(extra_train_df=None, params: dict = None):
    """
    Run the complete MLOps pipeline end-to-end.

    Parameters
    ----------
    extra_train_df : optional DataFrame to POOL with the default training data.
                     Used by retrain.py's 'mixed' strategy — combines the original
                     training data with the current (holiday) batch before retraining.

    params         : optional hyperparameter overrides for LightGBM.
                     Used by retrain.py's 'reexperiment' strategy.
    """
    print("\n" + "*" * 65)
    print("  E-COMMERCE PURCHASE PREDICTION — MLOps PIPELINE")
    print("  UCI Online Shoppers Purchasing Intention  |  Sakar et al. (2019)")
    print("*" * 65)
    print("""
  It is October. The team has all historical session data from
  February through October. The model will be trained on this data
  and deployed before the holiday season begins.

  Nov-Dec data does not exist at training time. It will arrive
  post-deployment as live traffic. That is what we monitor for drift.
""")

    # Stage 0 — Initialisation
    _banner("0", "Initialisation")
    set_seed()
    create_directories()

    # Stage 1 — Data Loading
    _banner("1", "Data Loading")
    train_df, current_df = prepare_data()

    # If retrain strategy pools extra data, merge it in here
    if extra_train_df is not None:
        import pandas as pd
        original_n = len(train_df)
        train_df = pd.concat([train_df, extra_train_df], ignore_index=True)
        print(f"  [Retrain] Pooled {original_n:,} original + "
              f"{len(extra_train_df):,} new rows = {len(train_df):,} total training rows")

    print(f"  Training data  (Feb-Oct) : {train_df.shape[0]:,} rows — everything we have at training time")
    print(f"  Post-deployment (Nov-Dec): {current_df.shape[0]:,} rows — arrives after the model goes live")
    print(f"")
    print(f"  Purchase rate in training data   : {train_df['Revenue'].mean():.1%}")
    print(f"  Purchase rate post-deployment    : {current_df['Revenue'].mean():.1%}  "
          f"(+{(current_df['Revenue'].mean()/train_df['Revenue'].mean()-1):.0%} — holiday effect)")
    print(f"")
    print(f"  NOTE: The model is trained on Feb-Oct only. Nov-Dec data is used exclusively")
    print(f"  for post-deployment monitoring and drift detection, not for training.")
    print("\n  Feature preview (first 3 rows of training data):")
    print(train_df[["PageValues", "BounceRates", "ExitRates",
                     "ProductRelated", "VisitorType", "Month", "Revenue"]
                   ].head(3).to_string(index=False))

    # Stage 2 — Data Validation
    _banner("2", "Data Validation")
    ShoppersValidator(train_df).validate()
    print("\n  Validating current (holiday) batch...")
    ShoppersValidator(current_df).validate()

    # Stage 3 — Feature Engineering
    _banner("3", "Feature Engineering")
    print("  Training set:")
    X_all, y_all, feature_cols = build_features(train_df, training=True)
    print("\n  Current batch:")
    X_curr, y_curr, _ = build_features(current_df, training=False)
    print(f"\n  Feature matrix shape : {X_all.shape}")
    print(f"  Features ({len(feature_cols)}): {', '.join(feature_cols[:6])} ...")

    # Stage 4 — Train/Test Split
    # ──────────────────────────────────────────────────────────────────────────
    # WHY WE SPLIT HERE (not in evaluate.py)
    # The split is part of the pipeline, not part of evaluation. The pipeline
    # is responsible for deciding what data the model trains on vs what it is
    # evaluated against. Keeping the split here makes the data flow explicit.
    #
    # WHY STRATIFY
    # With 15.5% positive rate, a random split could by chance put most buyers
    # in the training set and almost none in the test set. Stratification
    # preserves the positive rate in both splits.
    _banner("4", "Train / Test Split (Pre-Deployment Validation)")
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_all, y_all, np.arange(len(y_all)),
        test_size=0.20,
        random_state=SEED,
        stratify=y_all,
    )
    print(f"  Training set : {X_train.shape[0]:,} rows  (purchase rate: {y_train.mean():.1%})")
    print(f"  Hold-out set : {X_test.shape[0]:,} rows  (purchase rate: {y_test.mean():.1%})")
    print(f"")
    print(f"  Both splits are from Feb-Oct data only.")
    print(f"  The hold-out is our pre-deployment quality gate — it simulates")
    print(f"  unseen sessions within the same time period we trained on.")
    print(f"  The Nov-Dec data is NOT used here. It does not exist at training time.")

    # Stage 5 — Model Training
    _banner("5", "Model Training (LightGBM)")
    print("  Training on the 80% Feb-Oct split.")
    print("  The hold-out 20% remains untouched until Stage 6.")
    model = train_model(X_train, y_train, params=params)

    # Stage 6 — Pre-deployment evaluation on the hold-out set
    # The hold-out is 20% of Feb-Oct data — same time period, unseen sessions.
    # This is our confidence check before deployment. If gates pass, the model
    # goes live and starts scoring November traffic.
    _banner("6", "Pre-Deployment Evaluation (Hold-out Test Set)")
    print("  Evaluating on the 20% hold-out — sessions the model never saw during training.")
    print("  These are Feb-Oct sessions. If this passes, the model deploys to production.")
    metrics = evaluate(model, X_test, y_test, feature_cols, label="Hold-out (20% of Feb-Oct)")

    # Stage 6b — Slice Evaluation
    # Use the raw train_df rows that ended up in the test split (by index)
    train_df_test_slice = train_df.iloc[idx_test].reset_index(drop=True)

    print("\n  --- Slice evaluation by VisitorType ---")
    slice_evaluation(model, X_test, y_test, train_df_test_slice, "VisitorType", feature_cols)

    print("\n  --- Slice evaluation by Month ---")
    slice_evaluation(model, X_test, y_test, train_df_test_slice, "Month", feature_cols)

    # Stage 7 — Deployment gates
    _banner("7", "Deployment Gates")
    print("  Checking minimum performance thresholds before allowing deployment.")
    print("  If any gate fails, the pipeline stops here. The model is not deployed.")
    gate(metrics)

    # Stage 8 — Experiment Tracking
    _banner("8", "Experiment Tracking")
    log_experiment(
        params={
            "model":         "LGBMClassifier",
            "n_estimators":  model.n_estimators,
            "learning_rate": model.learning_rate,
            "num_leaves":    model.num_leaves,
        },
        metrics=metrics,
        extra={
            "dataset":        "train.csv (Feb-Oct)",
            "n_train":        len(X_train),
            "n_test":         len(X_test),
            "n_features":     len(feature_cols),
            "pos_rate_train": round(float(y_train.mean()), 4),
            "pos_rate_test":  round(float(y_test.mean()), 4),
        },
    )

    # Stage 9 — Model Registry and Deployment
    _banner("9", "Model Registry — Deploying to Production")
    model_path, version = save_model(model, metrics, feature_cols)

    print()
    promote(version, to_status="staging")
    promote(version, to_status="production")
    print("\n  Registry written to models/registry.json")
    print()
    print("  The model is now in production. It will start scoring live")
    print("  November sessions as they arrive. The feature pipeline saved")
    print("  in Stage 3 ensures inference uses the same transforms as training.")
    print()
    print("  NOTE: In a real system, promotion to 'production' requires manual")
    print("  sign-off or a canary rollout gate before full traffic is served.")

    # Stage 10 — Post-Deployment Drift Monitoring
    # ──────────────────────────────────────────────────────────────────────────
    # The model is now live. November sessions are arriving.
    # We compare the distribution of incoming features against what the model
    # was trained on. If key features have shifted, the model may be operating
    # outside the distribution it learned from.
    _banner("10", "Post-Deployment Drift Monitoring")
    print("  The model is live. November-December sessions are arriving.")
    print("  Comparing incoming feature distributions against the Feb-Oct training baseline.")
    drift_results = detect_drift(train_df, current_df)

    n_drifted = sum(1 for v in drift_results.values() if v.get("flagged"))
    if n_drifted > 0:
        print(
            f"\n  {n_drifted} feature(s) have shifted since the model was trained."
        )
        print(
            "  Holiday traffic behaves differently from the Feb-Oct sessions"
            " the model learned from."
        )
        print(
            "  Run retrain.py to evaluate whether retraining is warranted."
        )

    # Stage 11 — Batch Scoring of Post-Deployment Traffic
    _banner("11", "Batch Scoring — Post-Deployment November Traffic")
    print("  Scoring the Nov-Dec sessions that have arrived post-deployment.")
    print("  In production this runs continuously as sessions complete.")
    print("  Ground truth labels (did they buy?) accumulate and feed back into")
    print("  monitoring so we can measure real-world model performance over time.")
    start_time = time.time()
    preds = predict(model, X_curr)
    log_latency(start_time)
    log_predictions(preds)
    print(f"  Scored {len(preds):,} post-deployment sessions")
    print(f"  Predicted purchase rate : {preds.mean():.1%}  "
          f"(training purchase rate was {float(y_train.mean()):.1%})")
    print("  Logs: logs/predictions.csv  |  logs/latency.csv")

    # Stage 12 — Lineage Recording
    _banner("12", "Lineage Recording")
    train_path   = os.path.join(DATA_DIR, "train.csv")
    current_path = os.path.join(DATA_DIR, "current.csv")
    record_lineage(train_path, current_path, model_path)

    print("\n" + "*" * 65)
    print("  Pipeline complete. Model is deployed.")
    print()
    print("  The model is now scoring live November traffic.")
    print("  Stage 10 has already flagged distribution drift in incoming sessions.")
    print()
    print("  Next steps:")
    print("    python retrain.py --dry-run     # evaluate the drift — should we retrain?")
    print("    uvicorn serve:app --reload       # serve the model via REST API")
    print("    python batch_infer.py            # score a batch of incoming sessions")
    print("*" * 65)

    return metrics
