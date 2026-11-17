# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding="utf-8")

"""
retrain.py  -  Drift-Triggered Retraining with Three Strategies
================================================================
Evaluates four rules to determine whether the production model should be
retrained, and executes one of three retraining strategies based on the
type and severity of drift detected.

CONTEXT
-------
It is now late November / early December.

The model was trained in October on Feb-Oct data and deployed just before
the holiday season. Since deployment, 4,725 new sessions have accumulated
from November and December. Ground truth labels (did they purchase?) have
been confirmed for completed sessions.

The drift monitor (Stage 10 of the pipeline) has already flagged:
  - PageValues has shifted up +34%  (holiday buyers view higher-value pages)
  - ProductRelated page depth up +48%  (more browsing before purchase)
  - BounceRates have changed  (engaged holiday shoppers behave differently)
  - ExitRates have changed  (session length and exit patterns shifted)
  - SpecialDay values are active  (promotional calendar effects)

These are the four most important features in the model. The question is
no longer "is there drift?" — there clearly is. The question is:
"Should we retrain, and if so, how?"

THE FOUR TRIGGER RULES
-----------------------
Rule 1  IMMEDIATE RETRAIN — any high-importance feature drifts
  PageValues, BounceRates, ExitRates, or ProductRelated crosses its threshold.
  WHY: These features explain >60% of LightGBM's feature importance.
  A drift in the dominant signal is sufficient evidence the model is stale.

Rule 2  FEATURE COUNT GATE — >= 2 features drift
  WHY: One noisy column might be transient (one bad day of tracking).
  Two features drifting simultaneously is almost certainly a real shift.

Rule 3  F1 DEGRADATION GATE — F1 drops > 5pp from baseline
  WHY: Even without feature drift, concept drift (the relationship between
  features and label changing) can degrade performance without moving input
  distributions. If F1 drops, retrain regardless of feature signals.

Rule 4  MINIMUM INTERVAL GUARD — never retrain more often than every 14 days
  WHY: E-commerce is fast-moving, but retraining has costs (compute, validation,
  deployment risk). 14 days prevents over-reaction to day-to-day noise while
  allowing rapid response to genuine seasonal shifts.

THREE RETRAINING STRATEGIES
----------------------------
Once a retrain is triggered, you have three choices. This script implements all three.

Strategy 1: FINETUNE  (fastest, lowest compute)
  Continue training the existing model on the new (holiday) data only.
  LightGBM supports warm start via init_model — new trees are added to the
  existing ensemble.
  WHEN BEST: Drift is mild. The core signal is the same; the model just needs
  to adjust its decision boundary slightly.
  RISK: Can over-specialise to holiday data and forget non-holiday patterns
  (catastrophic forgetting). Fine-tuning on too few new samples can make
  the model worse.

Strategy 2: MIXED RETRAIN  (recommended for seasonal drift)
  Pool a portion of the original training data with ALL of the new data,
  then train a fresh model from scratch with the same hyperparameters.
  Default mix: 50% of original + 100% of new data.
  WHEN BEST: Seasonal concept drift — the holiday period is genuinely different
  but the model still needs to work on non-holiday traffic too (next year,
  the model may be deployed year-round).
  RISK: Slightly more compute than finetune. No forgetting risk.

Strategy 3: REEXPERIMENT  (most thorough, highest compute)
  Search a hyperparameter grid on the combined data. Choose the best config
  by F1 on a fresh validation split, then promote the winner.
  WHEN BEST: The drift is large and you suspect the original hyperparameters
  are no longer optimal (e.g., learning rate or num_leaves may need adjustment
  for the new distribution).
  RISK: Takes longer. Risk of overfitting to the current evaluation split
  if the grid is too large.

USAGE
-----
    python retrain.py                           # evaluate all 4 rules, auto-decide
    python retrain.py --strategy finetune       # force fine-tune strategy
    python retrain.py --strategy mixed          # force mixed retrain (recommended)
    python retrain.py --strategy reexperiment   # force full hyperparam search
    python retrain.py --force                   # retrain unconditionally (mixed strategy)
    python retrain.py --dry-run                 # evaluate rules + show decision; no action
"""

import os
import ast
import sys
import argparse
import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.config import (
    LOG_DIR, DATA_DIR, SEED,
    RETRAIN_MIN_FEATURES,
    RETRAIN_HIGH_IMPORTANCE_FEATURES,
    F1_DEGRADATION_THRESHOLD,
    MIN_RETRAIN_INTERVAL_DAYS,
)
from src.pipeline.run_pipeline import run_pipeline


DRIFT_LOG      = os.path.join(LOG_DIR, "drift.csv")
RETRAIN_LOG    = os.path.join(LOG_DIR, "retrain.csv")
EXPERIMENT_LOG = os.path.join(LOG_DIR, "experiments.csv")


# ── logging ──────────────────────────────────────────────────────────────────

def _log_retrain_event(triggered_by: str, n_flagged: int, action: str, detail: str = ""):
    row = pd.DataFrame([{
        "timestamp":    str(datetime.datetime.now()),
        "triggered_by": triggered_by,
        "n_flagged":    n_flagged,
        "action":       action,
        "detail":       detail,
    }])
    if not os.path.exists(RETRAIN_LOG):
        row.to_csv(RETRAIN_LOG, index=False)
    else:
        row.to_csv(RETRAIN_LOG, mode="a", header=False, index=False)


# ── rule evaluations ──────────────────────────────────────────────────────────

def _check_drift() -> tuple:
    if not os.path.exists(DRIFT_LOG):
        print("  [WARN] No drift log found. Run main.py first to generate it.")
        return 0, [], [], "never"
    df = pd.read_csv(DRIFT_LOG)
    if df.empty:
        return 0, [], [], "never"
    latest    = df.iloc[-1]
    n_flagged = int(latest.get("n_flagged", 0))
    ts        = latest.get("timestamp", "unknown")
    try:
        flagged   = ast.literal_eval(str(latest.get("flagged", "[]")))
    except Exception:
        flagged = []
    try:
        hi_imp = ast.literal_eval(str(latest.get("hi_importance_drift", "[]")))
    except Exception:
        hi_imp = []
    return n_flagged, flagged, hi_imp, ts


def _check_f1_degradation() -> tuple:
    """
    Compare most recent F1 against the previous run's F1.
    Returns (degraded: bool, delta: float, current_f1, baseline_f1)
    """
    if not os.path.exists(EXPERIMENT_LOG):
        return False, 0.0, None, None
    df = pd.read_csv(EXPERIMENT_LOG)
    if "f1" not in df.columns or len(df) < 2:
        return False, 0.0, None, None
    df_sorted    = df.sort_values("timestamp").dropna(subset=["f1"])
    if len(df_sorted) < 2:
        return False, 0.0, None, None
    current_f1   = float(df_sorted["f1"].iloc[-1])
    baseline_f1  = float(df_sorted["f1"].iloc[-2])
    delta        = baseline_f1 - current_f1
    return delta > F1_DEGRADATION_THRESHOLD, round(delta, 4), round(current_f1, 4), round(baseline_f1, 4)


def _check_min_interval() -> tuple:
    if not os.path.exists(RETRAIN_LOG):
        return True, None
    df = pd.read_csv(RETRAIN_LOG)
    completed = df[df["action"].str.contains("retrain_complete", na=False)]
    if completed.empty:
        return True, None
    last_ts      = pd.to_datetime(completed["timestamp"].iloc[-1])
    days_elapsed = (datetime.datetime.now() - last_ts).days
    return days_elapsed >= MIN_RETRAIN_INTERVAL_DAYS, days_elapsed


# ── retraining strategies ────────────────────────────────────────────────────

def _strategy_finetune():
    """
    Strategy 1: FINETUNE
    Warm-start LightGBM — add new trees on top of the existing production model
    using the current (holiday) batch only.

    How LightGBM warm start works:
      model.fit(X_new, y_new, init_model=existing_model)
      This initialises the boosting from the existing model's predictions and
      adds n_estimators more trees. The original trees are NOT modified.

    Result: the ensemble now has original_n + new_n trees, where the new trees
    correct residuals specific to the holiday distribution.
    """
    import pickle
    import lightgbm as lgb
    from src.features.build_features import build_features
    from src.models.evaluate import evaluate, gate
    from src.registry.versioning import save_model, promote

    print("\n  [Strategy: FINETUNE] Loading current (holiday) batch...")
    current_path = os.path.join(DATA_DIR, "current.csv")
    if not os.path.exists(current_path):
        print("  [ERROR] data/current.csv not found. Run main.py first.")
        return None

    current_df = pd.read_csv(current_path)
    current_df["Revenue"] = current_df["Revenue"].astype(int)
    current_df["Weekend"] = current_df["Weekend"].astype(str)

    # Load production model from registry
    from src.registry.versioning import get_production_model
    existing_model, existing_meta = get_production_model()
    if existing_model is None:
        print("  [ERROR] No production model found. Run main.py first.")
        return None

    print(f"  Loaded production model: {existing_meta.get('version', 'unknown')}")
    print(f"  Current model trees: {existing_model.n_estimators}")

    # Build features using TRAINING-TIME medians (already saved in feature_pipeline.json)
    X_curr, y_curr, feature_cols = build_features(current_df, training=False)

    # Split holiday data into train/val for fine-tuning evaluation
    X_ft_train, X_ft_val, y_ft_train, y_ft_val = train_test_split(
        X_curr, y_curr, test_size=0.25, random_state=SEED, stratify=y_curr
    )

    print(f"  Fine-tuning on {len(X_ft_train):,} holiday sessions "
          f"(+{existing_model.n_estimators} existing trees)...")

    # LightGBM warm start: init_model resumes from where the existing model left off
    finetuned = lgb.LGBMClassifier(
        n_estimators     = 150,     # add 150 more trees on top of existing
        learning_rate    = 0.03,    # lower LR for fine-tuning
        num_leaves       = existing_model.num_leaves,
        scale_pos_weight = (y_ft_train == 0).sum() / max((y_ft_train == 1).sum(), 1),
        random_state     = SEED,
        n_jobs           = -1,
        verbose          = -1,
    )
    finetuned.fit(X_ft_train, y_ft_train, init_model=existing_model.booster_)

    print("\n  Evaluation of fine-tuned model on holiday validation set:")
    metrics_ft = evaluate(finetuned, X_ft_val, y_ft_val, feature_cols,
                          label="Fine-tune val (25% holiday hold-out)")

    print(f"\n  Before finetune F1: {existing_meta.get('f1', 'N/A')}")
    print(f"  After  finetune F1: {metrics_ft['f1']:.4f}")

    return finetuned, metrics_ft, feature_cols


def _strategy_mixed(mix_ratio: float = 0.50):
    """
    Strategy 2: MIXED RETRAIN  (recommended for seasonal drift)
    Pool mix_ratio of original training data + 100% of post-deployment data.
    Train a fresh LightGBM from scratch with the same production hyperparameters.

    WHY THIS WORKS BETTER THAN FINETUNE FOR SEASONAL DRIFT
    -------------------------------------------------------
    The model needs to serve traffic year-round (not just in November/December).
    Pure fine-tuning on holiday data risks catastrophic forgetting of Feb-Oct
    patterns. Mixed retrain keeps both distributions in the training data.

    Default mix_ratio=0.50: 50% of original Feb-Oct + 100% of Nov-Dec.
    With 7,317 original and 4,725 current rows, the mixed set is:
      3,658 (50% of Feb-Oct) + 4,725 (all Nov-Dec) = 8,383 rows.
    Holiday sessions represent ~56% of the retrain set — proportional
    to how much their behaviour has shifted.
    """
    train_path   = os.path.join(DATA_DIR, "train.csv")
    current_path = os.path.join(DATA_DIR, "current.csv")

    if not os.path.exists(train_path) or not os.path.exists(current_path):
        print("  [ERROR] data files missing. Run main.py first.")
        return None

    original_df = pd.read_csv(train_path)
    current_df  = pd.read_csv(current_path)

    sample_n   = max(int(len(original_df) * mix_ratio), 500)
    sampled_df = original_df.sample(n=sample_n, random_state=SEED)

    print(f"\n  [Strategy: MIXED RETRAIN]")
    print(f"  Original training data (Feb-Oct)  : {len(original_df):,} rows")
    print(f"  Sampled ({mix_ratio:.0%} of original)     : {len(sampled_df):,} rows")
    print(f"  Post-deployment data  (Nov-Dec)   : {len(current_df):,} rows")
    print(f"  Mixed training set                : {len(sampled_df) + len(current_df):,} rows")
    print(f"  Holiday share in retrain set      : {len(current_df)/(len(sampled_df)+len(current_df)):.0%}")

    # Combine sampled original + all current data, then retrain from scratch
    combined_df = pd.concat([sampled_df, current_df], ignore_index=True)
    metrics = run_pipeline(extra_train_df=combined_df)
    return metrics


def _strategy_reexperiment():
    """
    Strategy 3: REEXPERIMENT
    Try multiple hyperparameter configurations on the combined (mixed) dataset.
    Pick the best by F1 on a validation split.

    WHY THIS EXISTS
    ---------------
    The original hyperparameters (num_leaves=63, lr=0.05) were tuned on the
    non-holiday distribution. After a major drift, the optimal configuration
    may be different:
      - Fewer leaves (simpler model) if the holiday data is smaller
      - Lower learning rate if the new data is noisier
      - Different scale_pos_weight if the positive rate changed

    This strategy makes that search explicit and auditable.
    """
    from src.models.train import train_model
    from src.models.evaluate import evaluate
    from src.features.build_features import build_features

    print("\n  [Strategy: REEXPERIMENT] Loading mixed dataset...")

    train_path   = os.path.join(DATA_DIR, "train.csv")
    current_path = os.path.join(DATA_DIR, "current.csv")

    original_df = pd.read_csv(train_path)
    current_df  = pd.read_csv(current_path)

    # Pool 60% original + 100% new for re-experiment
    sampled_df = original_df.sample(n=int(len(original_df) * 0.60), random_state=SEED)
    combined   = pd.concat([sampled_df, current_df], ignore_index=True)
    combined["Revenue"] = combined["Revenue"].astype(int)
    combined["Weekend"] = combined["Weekend"].astype(str)

    print(f"  Combined set: {len(combined):,} rows "
          f"(purchase rate: {combined['Revenue'].mean():.1%})")

    # Build features — re-use existing training-time medians
    X_all, y_all, feature_cols = build_features(combined, training=False)

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all
    )

    # Hyperparameter grid — keep small to avoid too-long runs in a course context
    grid = [
        {"n_estimators": 400,  "learning_rate": 0.05, "num_leaves": 31},
        {"n_estimators": 500,  "learning_rate": 0.05, "num_leaves": 63},
        {"n_estimators": 500,  "learning_rate": 0.03, "num_leaves": 63},
        {"n_estimators": 600,  "learning_rate": 0.03, "num_leaves": 127},
    ]

    print(f"\n  Searching {len(grid)} hyperparameter configurations...\n")
    print(f"  {'Config':<50} {'F1':>6}  {'AUC-ROC':>8}  {'AUC-PR':>7}")
    print(f"  {'-'*50}  {'-'*6}  {'-'*8}  {'-'*7}")

    best_f1     = -1
    best_params = None
    best_model  = None
    best_metrics = None

    for i, p in enumerate(grid, 1):
        m = train_model(X_train, y_train, params=p)
        from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
        preds = m.predict(X_val)
        proba = m.predict_proba(X_val)[:, 1]
        f1    = f1_score(y_val, preds, zero_division=0)
        auc   = roc_auc_score(y_val, proba)
        pr    = average_precision_score(y_val, proba)

        mark = " <<< BEST" if f1 > best_f1 else ""
        desc = (f"n={p['n_estimators']}, lr={p['learning_rate']}, "
                f"leaves={p['num_leaves']}")
        print(f"  [{i}] {desc:<46} {f1:>6.4f}  {auc:>8.4f}  {pr:>7.4f}{mark}")

        if f1 > best_f1:
            best_f1      = f1
            best_params  = p
            best_model   = m
            best_metrics = {"f1": f1, "auc_roc": auc, "auc_pr": pr,
                            "precision": 0.0, "recall": 0.0, "accuracy": 0.0}

    print(f"\n  Best config: {best_params}  →  F1={best_f1:.4f}")

    # Re-run full pipeline with the winning hyperparameters
    print("\n  Re-running full pipeline with best hyperparameters...")
    run_pipeline(extra_train_df=current_df, params=best_params)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Drift-triggered retraining with three strategies."
    )
    parser.add_argument(
        "--strategy",
        choices=["finetune", "mixed", "reexperiment"],
        default=None,
        help=(
            "finetune     = warm-start LightGBM on holiday data only\n"
            "mixed        = pool original + holiday data, retrain from scratch\n"
            "reexperiment = hyperparam search on combined data (most thorough)"
        ),
    )
    parser.add_argument("--force",   action="store_true",
                        help="Retrain unconditionally (uses 'mixed' strategy by default).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Evaluate rules and print decision; do NOT retrain.")
    parser.add_argument("--mix-ratio", type=float, default=0.50,
                        help="Fraction of original data to pool in 'mixed' strategy (default: 0.50).")
    args = parser.parse_args()

    print("\n" + "*" * 70)
    print("  RETRAIN DECISION — E-Commerce Purchase Prediction Model")
    print("*" * 70)
    print("""
  It is late November. The model has been live for ~4 weeks.
  4,725 post-deployment sessions have accumulated (Nov-Dec data).
  Ground truth labels are available for completed sessions.

  Drift was flagged in Stage 10. This script evaluates whether
  the evidence is strong enough to trigger a retrain, and if so,
  which strategy is appropriate for this type of drift.
""")

    # ── Force override ────────────────────────────────────────────────────────
    if args.force:
        strategy = args.strategy or "mixed"
        print(f"\n  [OVERRIDE] --force set. Strategy: {strategy}")
        _log_retrain_event("forced", 0, "retrain_started", f"--force, strategy={strategy}")
        if not args.dry_run:
            _execute_strategy(strategy, args.mix_ratio)
            _log_retrain_event("forced", 0, "retrain_complete", f"strategy={strategy}")
        else:
            print("  [DRY RUN] Would retrain. Exiting without action.")
        print("*" * 70)
        return

    # ── Rule 4: minimum interval ──────────────────────────────────────────────
    print("\n[Rule 4] Minimum retrain interval check...")
    interval_ok, days_since = _check_min_interval()
    if days_since is not None:
        print(f"  Days since last retrain : {days_since}")
        print(f"  Minimum interval        : {MIN_RETRAIN_INTERVAL_DAYS} days")
    else:
        print("  No prior retrain found — interval check passes by default.")

    if not interval_ok:
        print(
            f"\n  [BLOCKED] Last retrain was {days_since} day(s) ago "
            f"(minimum: {MIN_RETRAIN_INTERVAL_DAYS} days).\n"
            "  E-commerce retraining cadence: bi-weekly minimum to avoid "
            "over-fitting to daily noise (one-day traffic spikes, A/B test "
            "cross-contamination). Use --force if a major event requires "
            "immediate action (e.g., a flash sale that changes the distribution overnight)."
        )
        _log_retrain_event("drift_check", 0, "skipped-within-min-interval",
                           f"{days_since}d < {MIN_RETRAIN_INTERVAL_DAYS}d")
        print("*" * 70)
        return

    # ── Rules 1+2: drift signals ──────────────────────────────────────────────
    print("\n[Rule 1 + 2] Checking drift log...")
    n_flagged, flagged, hi_imp_flagged, last_ts = _check_drift()

    print(f"  Last drift check             : {last_ts}")
    print(f"  Features flagged             : {n_flagged}")
    print(f"  Flagged features             : {flagged or 'none'}")
    print(f"  High-importance flagged      : {hi_imp_flagged or 'none'}")
    print(f"  Rule 1 threshold             : ANY high-importance feature drifts")
    print(f"  Rule 2 threshold             : >= {RETRAIN_MIN_FEATURES} features drift")

    # ── Rule 3: F1 degradation ────────────────────────────────────────────────
    print("\n[Rule 3] F1 degradation check...")
    f1_degraded, f1_delta, curr_f1, base_f1 = _check_f1_degradation()
    if curr_f1 is not None:
        print(f"  Baseline F1 (last train)     : {base_f1:.4f}")
        print(f"  Current F1 (latest run)      : {curr_f1:.4f}")
        print(f"  F1 drop                      : {f1_delta:.4f}  "
              f"(threshold: {F1_DEGRADATION_THRESHOLD})")
        if f1_degraded:
            print(f"  [TRIGGER] F1 degradation exceeds threshold.")
    else:
        print("  Not enough F1 history in experiments.csv — Rule 3 skipped.")

    # ── Decision ─────────────────────────────────────────────────────────────
    print("\n[Decision] Evaluating all active rules...")

    rule1_fires = bool(hi_imp_flagged)
    rule2_fires = n_flagged >= RETRAIN_MIN_FEATURES
    rule3_fires = f1_degraded

    if rule1_fires:
        decision = "retrain"
        recommended_strategy = "mixed"
        reason = (
            f"Rule 1: high-importance feature(s) drifted: {hi_imp_flagged}. "
            "These drive >60% of LightGBM's feature importance. "
            "Recommended strategy: 'mixed' (pool original + holiday data)."
        )
    elif rule2_fires:
        decision = "retrain"
        recommended_strategy = "mixed"
        reason = (
            f"Rule 2: {n_flagged} features drifted "
            f"(threshold: {RETRAIN_MIN_FEATURES}). Features: {flagged}. "
            "Recommended strategy: 'mixed' retrain."
        )
    elif rule3_fires:
        decision = "retrain"
        recommended_strategy = "reexperiment"
        reason = (
            f"Rule 3: F1 degraded by {f1_delta:.4f} "
            f"(threshold: {F1_DEGRADATION_THRESHOLD}). "
            "F1 drop without feature drift suggests CONCEPT DRIFT — the "
            "relationship between features and purchase intent has changed. "
            "Recommended strategy: 'reexperiment' (search for new hyperparameters)."
        )
    elif n_flagged > 0:
        decision = "skip"
        recommended_strategy = None
        reason = (
            f"{n_flagged} feature(s) flagged but below Rule 2 threshold, "
            "no high-importance drift, and F1 is stable. Monitoring but not retraining."
        )
    else:
        decision = "skip"
        recommended_strategy = None
        reason = "No drift detected. F1 stable. Model is current."

    print(f"\n  Decision             : {decision.upper()}")
    print(f"  Reason               : {reason}")
    if recommended_strategy:
        print(f"  Recommended strategy : {recommended_strategy}")

    print()
    print("  Strategy comparison guide:")
    print("    finetune     — warm-start on Nov-Dec data only; fast but risks forgetting Feb-Oct patterns")
    print("    mixed        — pool 50% Feb-Oct + 100% Nov-Dec; model keeps both distributions  [RECOMMENDED]")
    print("    reexperiment — full hyperparam search on combined data; use when F1 has degraded too")

    # ── Act ───────────────────────────────────────────────────────────────────
    print("\n[Action]")

    if decision == "skip":
        print("  No retraining needed. Exiting.")
        _log_retrain_event("drift_check", n_flagged, "skipped-no-trigger", reason)
        print(f"\n  Audit log: {RETRAIN_LOG}")
        print("*" * 70)
        return

    if args.dry_run:
        print(f"  [DRY RUN] Decision = RETRAIN but --dry-run flag set. No action taken.")
        _log_retrain_event("dry_run", n_flagged, "dry_run-would_retrain", reason)
        print("*" * 70)
        return

    strategy = args.strategy or recommended_strategy or "mixed"
    triggered_by = (
        "rule1_hi_importance" if rule1_fires else
        "rule2_feature_count" if rule2_fires else
        "rule3_f1_degradation"
    )

    print(f"  Executing strategy   : {strategy}")
    print(f"  Triggered by         : {triggered_by}")
    print(f"  Reason               : {reason}\n")

    _log_retrain_event(triggered_by, n_flagged, f"retrain_started-strategy={strategy}", reason)
    _execute_strategy(strategy, args.mix_ratio)
    _log_retrain_event(triggered_by, n_flagged, f"retrain_complete-strategy={strategy}", reason)

    print(f"\n  Audit log: {RETRAIN_LOG}")
    print("*" * 70)


def _execute_strategy(strategy: str, mix_ratio: float):
    """Dispatch to the appropriate retraining strategy."""
    if strategy == "finetune":
        _strategy_finetune()

    elif strategy == "mixed":
        _strategy_mixed(mix_ratio=mix_ratio)

    elif strategy == "reexperiment":
        _strategy_reexperiment()

    else:
        print(f"  [ERROR] Unknown strategy: '{strategy}'. "
              "Use: finetune | mixed | reexperiment")
        sys.exit(1)


if __name__ == "__main__":
    main()
