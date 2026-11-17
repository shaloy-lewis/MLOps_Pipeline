"""
validate.py  -  Data Validation
=================================
MLOps Lesson: "Data validation is the first line of defence in every ML pipeline."

WHY THIS EXISTS
---------------
Most ML bugs are not model bugs — they are data bugs. Common real-world
failures this validator catches before training:

  1. The JavaScript tracking pixel fires incorrectly — PageValues arrives
     as a string "12.5" instead of float 12.5.
     → LightGBM raises a confusing type error deep inside fit().
     Catching it here gives a clear, located error message.

  2. A new analytics platform uses -1 as a sentinel for "untracked session"
     in BounceRates instead of NULL.
     → The model trains on -1 (physically impossible — rates are 0-1),
     silently learning a spurious pattern.

  3. A data export omits November sessions entirely (timezone bug).
     → current_df has only December, skewing the drift detection.
     Catching low row counts here prevents a silent bad result.

  4. A web framework upgrade changes VisitorType to "returning_visitor"
     (lowercase) instead of "Returning_Visitor".
     → pd.get_dummies() creates a new OHE column that didn't exist at
     training time. Shape mismatch at inference time.

  5. The Revenue column arrives as bool strings "True"/"False" instead of
     int 0/1 after a CSV round-trip.
     → Metrics functions silently compute on wrong types.

DESIGN PRINCIPLE
----------------
  * Errors   → abort the pipeline (DataValidationError is raised)
  * Warnings → log and continue (imputation handles minor missingness)
"""

import pandas as pd
from src.config.config import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET,
    NULL_THRESHOLD, COLUMN_RANGES,
    VALID_VISITOR_TYPES, VALID_MONTHS, VALID_WEEKEND,
)


class DataValidationError(Exception):
    """
    Raised when the dataset fails one or more hard validation checks.
    The pipeline must not proceed past this point.
    """


class ShoppersValidator:
    """
    Runs a full suite of quality checks on an Online Shoppers DataFrame.

    Usage
    -----
        validator = ShoppersValidator(df)
        validator.validate()   # raises DataValidationError on failure

    Checks performed
    ----------------
    1. Required columns present
    2. Column types (numeric vs object)
    3. Null rates (warns below threshold, errors above)
    4. Value ranges (hard physical bounds)
    5. Categorical label whitelist (VisitorType, Month, Weekend)
    6. Target column integrity (binary, not pathologically imbalanced)
    7. Minimum row count (too few rows = sampling/pipeline failure)
    """

    _EXPECTED_TYPES = (
        {col: "numeric" for col in NUMERIC_FEATURES}
        | {col: "object" for col in CATEGORICAL_FEATURES}
    )

    def __init__(self, df: pd.DataFrame):
        self.df       = df
        self.errors   = []
        self.warnings = []

    # ── individual checks ────────────────────────────────────────────────────

    def check_required_columns(self):
        """Abort immediately if expected columns are missing."""
        required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
        missing  = required - set(self.df.columns)
        if missing:
            self.errors.append(
                f"Missing columns: {sorted(missing)}. "
                "Schema may have changed upstream — verify the data export."
            )

    def check_types(self):
        """
        Verify numeric columns are numeric and categorical columns are strings.

        WHY THIS MATTERS
        ----------------
        If PageValues arrives as '12.5' (string), LightGBM will raise a
        confusing 'could not convert string to float' error during fit().
        Catching it here gives a clear, actionable error message.
        """
        for col, expected in self._EXPECTED_TYPES.items():
            if col not in self.df.columns:
                continue
            is_numeric = pd.api.types.is_numeric_dtype(self.df[col])
            if expected == "numeric" and not is_numeric:
                self.errors.append(
                    f"Type error: '{col}' should be numeric but is "
                    f"dtype={self.df[col].dtype}. "
                    "Possible cause: tracking system changed export format."
                )
            elif expected == "object" and is_numeric:
                self.errors.append(
                    f"Type error: '{col}' should be string/categorical but is "
                    f"dtype={self.df[col].dtype}. "
                    "Expected text labels (e.g. 'Returning_Visitor', 'Nov')."
                )

    def check_nulls(self):
        """
        Check null rates for every numeric feature.

        E-commerce clickstream data has legitimate zero values (e.g., zero
        Informational page visits), but true NULLs above 8% signal a
        tracking/instrumentation failure rather than organic user behaviour.
        """
        for col in NUMERIC_FEATURES:
            if col not in self.df.columns:
                continue
            null_pct = self.df[col].isnull().mean()
            if null_pct > NULL_THRESHOLD:
                self.errors.append(
                    f"Null rate too high: '{col}' is {null_pct:.1%} null "
                    f"(limit = {NULL_THRESHOLD:.0%}). "
                    "Likely cause: JavaScript tracker is broken or feed failed."
                )
            elif null_pct > 0:
                self.warnings.append(
                    f"'{col}': {null_pct:.1%} null (within tolerance — "
                    "will be imputed with training median)."
                )

    def check_ranges(self):
        """
        Verify all numeric features stay within hard physical bounds.

        WHY HARD BOUNDS (not outlier detection)
        ----------------------------------------
        A BounceRate of -1 is not a statistical outlier — it is a sentinel
        value from a broken tracking implementation. Hard bounds catch
        this unambiguously without needing to model the data distribution.
        """
        for col, (lo, hi) in COLUMN_RANGES.items():
            if col not in self.df.columns:
                continue
            col_data = self.df[col].dropna()
            n_below  = int((col_data < lo).sum())
            n_above  = int((col_data > hi).sum())
            if n_below > 0:
                self.errors.append(
                    f"Range error: '{col}' has {n_below} value(s) below "
                    f"minimum {lo}. Min observed: {col_data.min():.4f}. "
                    "Check for sentinel values (e.g. -1 for 'untracked')."
                )
            if n_above > 0:
                self.warnings.append(
                    f"'{col}' has {n_above} value(s) above max {hi} — "
                    f"will be clipped at feature engineering. Max: {col_data.max():.2f}."
                )

    def check_categoricals(self):
        """
        Verify categorical columns only contain known labels.

        WHY THIS IS CRITICAL AT INFERENCE TIME
        ----------------------------------------
        If a new traffic source introduces VisitorType='Bot', pd.get_dummies()
        creates a column that didn't exist at training time. The model receives
        a feature vector of the wrong shape and either crashes or silently
        misclassifies every session from that traffic source.

        Catching unknown labels here forces explicit resolution before training.
        """
        checks = {
            "VisitorType": VALID_VISITOR_TYPES,
            "Month":       VALID_MONTHS,
        }
        for col, valid_set in checks.items():
            if col not in self.df.columns:
                continue
            observed = set(self.df[col].dropna().unique())
            unknown  = observed - valid_set
            if unknown:
                self.errors.append(
                    f"Unknown categories in '{col}': {unknown}. "
                    f"Allowed values: {valid_set}. "
                    "Update VALID_* sets in config.py if this is intentional."
                )

        # Weekend is stored as string "True"/"False" after CSV round-trip
        if "Weekend" in self.df.columns:
            observed_weekend = set(self.df["Weekend"].dropna().unique())
            if not observed_weekend.issubset({"True", "False", True, False, "true", "false", 0, 1}):
                self.errors.append(
                    f"Unexpected values in 'Weekend': {observed_weekend}. "
                    "Expected 'True' / 'False' (string or bool)."
                )

    def check_target(self):
        """
        Ensure the target column is binary and not pathologically imbalanced.

        Purchase rate ~15.5% IS EXPECTED for this dataset (Sakar et al. 2019).
        The model and thresholds in config.py are calibrated for this.
        What we flag here is SEVERE DEVIATION from ~15.5% — which signals
        a label encoding bug or data sampling error, not a real change.
        """
        if TARGET not in self.df.columns:
            return
        unique_vals = set(self.df[TARGET].dropna().unique())
        if not unique_vals.issubset({0, 1}):
            self.errors.append(
                f"Target '{TARGET}' contains non-binary values: {unique_vals}. "
                "Expected only 0 (no purchase) and 1 (purchase). "
                "If Revenue is still bool, run: df['Revenue'] = df['Revenue'].astype(int)"
            )
        if TARGET in self.df.columns and pd.api.types.is_numeric_dtype(self.df[TARGET]):
            pos_rate = self.df[TARGET].mean()
            if pos_rate < 0.03:
                self.warnings.append(
                    f"Unusually low purchase rate: {pos_rate:.1%}. "
                    "Expected ~15.5% (Sakar et al. 2019). "
                    "Possible label encoding error or heavy sampling bias."
                )
            elif pos_rate > 0.55:
                self.warnings.append(
                    f"Unusually high purchase rate: {pos_rate:.1%}. "
                    "Expected ~15.5%. Check for data leakage or label error."
                )

    def check_min_rows(self, min_rows: int = 200):
        """
        Abort if the DataFrame has too few rows.

        WHY 200: With <200 sessions, the 80/20 train/test split leaves
        fewer than 40 test samples — too few for reliable F1 estimation.
        This most likely signals a pipeline failure (empty file, bad query),
        not a real data scarcity issue.
        """
        if len(self.df) < min_rows:
            self.errors.append(
                f"Too few rows: {len(self.df)} (minimum: {min_rows}). "
                "Pipeline likely received an empty or near-empty dataset. "
                "Check the data export / download step."
            )

    # ── orchestration ─────────────────────────────────────────────────────────

    def validate(self):
        """
        Run all checks, print a summary, and raise DataValidationError if
        any hard check failed.
        """
        print("  Running data quality checks...")
        self.check_min_rows()
        self.check_required_columns()
        self.check_types()
        self.check_nulls()
        self.check_ranges()
        self.check_categoricals()
        self.check_target()

        for w in self.warnings:
            print(f"  [WARN]  {w}")

        if self.errors:
            print(f"\n  [FAIL]  {len(self.errors)} validation error(s):")
            for e in self.errors:
                print(f"    [x]  {e}")
            raise DataValidationError(
                f"Data validation failed with {len(self.errors)} error(s). "
                "Fix the data before the pipeline can proceed."
            )

        pos_str = ""
        if TARGET in self.df.columns and pd.api.types.is_numeric_dtype(self.df[TARGET]):
            pos_str = f"  Purchase rate: {self.df[TARGET].mean():.1%}"
        n_warn = len(self.warnings)
        print(
            f"  [PASS]  All hard checks passed."
            f"  Shape: {self.df.shape}"
            + pos_str
            + (f"  ({n_warn} warning(s))" if n_warn else "")
        )
