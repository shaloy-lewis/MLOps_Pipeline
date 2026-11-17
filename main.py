"""
main.py  -  Pipeline Entry Point
=================================
Runs the complete MLOps pipeline end-to-end.

    python main.py

THE SITUATION
-------------
It is October. The team has 7,317 e-commerce sessions from February
through October. A model has been selected (model_selection.py). Now the
full production pipeline is built and the model is deployed.

At deployment time:
  - We only have Feb-Oct data. November has not happened yet.
  - We train on 80% of that (Feb-Oct), hold out 20% for pre-deployment testing.
  - The Nov-Dec data in data/current.csv represents what arrives AFTER deployment.
    It did not exist at training time. We cannot and do not train on it.

WHAT THIS PIPELINE DOES
------------------------
  Train set    : 7,317 sessions, Feb-Oct (all historical data at training time)
  Hold-out test: 20% of Feb-Oct — for pre-deployment validation only
  Current batch: 4,725 sessions, Nov-Dec — post-deployment incoming traffic

Expected results on the pre-deployment hold-out test set:
  F1 (minority class, Revenue=True) : 0.67 – 0.76
  AUC-ROC                           : 0.92 – 0.96
  AUC-PR                            : 0.72 – 0.80

AFTER DEPLOYMENT (what Stage 10 simulates)
-------------------------------------------
Once live, the model starts scoring November sessions as they arrive.
Stage 10 compares the incoming November feature distributions against
what the model was trained on. Drift is detected. This triggers the
retrain decision.

    python retrain.py --dry-run              # evaluate rules, show recommendation
    python retrain.py --strategy mixed       # pool original + new data (recommended)
    python retrain.py --strategy finetune    # warm-start on new data only
    python retrain.py --strategy reexperiment  # full hyperparameter search

SERVING AND BATCH INFERENCE
----------------------------
    uvicorn serve:app --reload --port 8000      # REST API (open /docs for schema)
    python batch_infer.py --input data/current.csv --output data/scored.csv

SEE ALSO
--------
    baseline.py        # first-pass pitfalls — what went wrong before this pipeline
    model_selection.py # model comparison that justified the LightGBM selection
    GUIDE.html         # full walkthrough, open in browser
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # fix Windows console Unicode for ucimlrepo

from src.pipeline.run_pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()
