# MLOps Pipeline — E-Commerce Purchase Prediction

An end-to-end MLOps pipeline in pure Python — no MLflow, no DVC, no cloud.
Every concept is implemented from scratch so you understand what production
tools are actually doing under the hood.

---

## The Situation

An e-commerce retailer has been collecting session data since February.
It is now **October**. They want to predict which browsing sessions will end
in a purchase, so the platform can serve targeted offers before visitors leave.

The team has **7,317 sessions (Feb–Oct)**. That is all the data that exists.
November and December have not happened yet.

The model needs to be deployed before the holiday season begins.

---

## Why Not Use All the Data?

The Nov–Dec data does not exist at training time. The model is built in
October on Feb–Oct sessions and deployed. November traffic arrives after
the model is already live — it cannot be used for training without leaking
future information into the model.

This temporal boundary is deliberate and important:
- Training on Nov–Dec before deployment would mean training on data
  the model should be *predicting against*, not learning from.
- The hold-out test set is 20% of Feb–Oct data — same time period,
  unseen sessions — used for pre-deployment validation only.
- Nov–Dec data is used exclusively for post-deployment monitoring.

---

## Run Order

```bash
# Install dependencies
pip install -r requirements.txt

# Run in order:
python baseline.py          # October: first-pass pitfalls — what goes wrong
python model_selection.py   # October: rigorous comparison — select the model
python main.py              # October: build pipeline, deploy

# After deployment — drift response (pick one strategy):
python retrain.py --dry-run                  # evaluate drift — should we retrain?
python retrain.py --strategy mixed           # recommended for seasonal drift
python retrain.py --strategy finetune        # fast warm-start on new data only
python retrain.py --strategy reexperiment    # full hyperparameter search

# Serve the model:
uvicorn serve:app --reload --port 8000       # REST API — open /docs for schema
python batch_infer.py                        # score the Nov–Dec batch CSV
```

---

## baseline.py — First Pass (October)

A junior data scientist takes a first look at the Feb–Oct dataset.
The code runs without errors. The results look reasonable. The issues are invisible.

| # | Pitfall | Consequence |
|---|---|---|
| 1 | Hardcoded paths | Breaks on any machine except the author's |
| 2 | LabelEncoder on nominal categories | Month treated as Feb < Mar < ... — false ordinal signal |
| 3 | Rapid sweep sorted by Accuracy | 92% accuracy model catches only 58% of buyers |
| 4 | Model selected by accuracy, not F1 | 92.4% accuracy reported, 80 buyers missed (~$6,800 cost) |
| 5 | No feature pipeline saved | LabelEncoder fails on unseen Nov-Dec months at inference |
| 6 | No experiment tracking | No record of config, no reproducibility |
| 7 | No drift monitoring | Model silently fails when holiday traffic arrives |

---

## model_selection.py — Model Comparison (October)

Before building the production system, the team runs a rigorous comparison.
Same data, same split, same metric for all four candidates.

The key insight: a model that achieves 88% accuracy by predicting "no purchase"
for every session is useless. The business metric is F1 on the minority class.

| Model | F1 (buyers) | Why it performs this way |
|---|---|---|
| Logistic Regression | ~0.45–0.55 | Linear boundary cannot capture PageValues threshold |
| Random Forest | ~0.55–0.62 | Handles non-linearity; no sequential error correction |
| XGBoost | ~0.62–0.70 | Sequential boosting improves boundary cases |
| **LightGBM** | **~0.67–0.76** | **Leaf-wise growth extracts more from skewed PageValues** |

The winning model and its hyperparameters are locked into the production pipeline.
This script is the audit trail for that decision.

---

## main.py — Production Pipeline (October → Deployed)

12 stages that train, validate, register, and monitor the model.

| Stage | What it does |
|---|---|
| 0 | Set seed, create directories |
| 1 | Load Feb–Oct training data + Nov–Dec post-deployment batch |
| 2 | Validate schema, null rates, ranges — abort on failure |
| 3 | Feature engineering — medians saved for inference-time consistency |
| **4** | **80/20 stratified split of Feb–Oct only — pre-deployment hold-out** |
| 5 | Train LightGBM on 80% split with `scale_pos_weight` for imbalance |
| **6** | **Evaluate on 20% hold-out — F1 ~0.67–0.76, honest numbers** |
| 7 | Deployment gates — F1 ≥ 0.65, Recall ≥ 0.60, AUC-PR ≥ 0.60 |
| 8 | Log run to experiments.csv |
| 9 | Save model, promote: candidate → staging → production |
| **10** | **Post-deployment drift monitoring — compare incoming Nov–Dec traffic against training distribution** |
| 11 | Score Nov–Dec batch — 4,725 sessions arriving post-deployment |
| 12 | Record lineage: model ↔ data ↔ code ↔ environment |

---

## retrain.py — Drift Response (November / December)

The model is live. 4,725 November–December sessions have accumulated.
Stage 10 flagged distribution drift in the incoming traffic.

| Feature | Training (Feb–Oct) | Post-deployment (Nov–Dec) | Shift |
|---|---|---|---|
| PageValues | 5.3 | 7.0 | +34% |
| ProductRelated | 26.6 | 39.4 | +48% |
| BounceRates | 0.022 | 0.025 | +16% |
| ExitRates | 0.043 | 0.049 | +12% |

These are the four most important features. Rule 1 fires: immediate retrain.
Three strategies are available depending on the nature of the drift:

```bash
python retrain.py --strategy finetune       # warm-start on Nov-Dec only; fast
python retrain.py --strategy mixed          # pool Feb-Oct + Nov-Dec; recommended
python retrain.py --strategy reexperiment   # full hyperparameter search on combined data
```

| Strategy | When to use |
|---|---|
| `finetune` | Mild drift, same underlying concept, tight compute budget |
| `mixed` | Seasonal shift — model must keep working on both normal and holiday traffic |
| `reexperiment` | F1 has degraded significantly — current hyperparameters may be stale |

---

## Serving and Batch Inference

```bash
uvicorn serve:app --reload --port 8000   # REST API — open /docs for schema
python batch_infer.py                    # score a batch of incoming sessions
```

---

## What This Implements From Scratch

| This file | The production tool it maps to |
|---|---|
| `experiments.csv` | MLflow, Weights & Biases, Neptune |
| `models/registry.json` | MLflow Model Registry, SageMaker Model Registry |
| `retrain.py` rules | Airflow DAG, Prefect Flow, Kubeflow Pipeline |
| `drift.csv` | Evidently AI, WhyLabs, Arize |
| `serve.py` | BentoML, Seldon, SageMaker endpoint |

---

**[GUIDE.html](GUIDE.html)** — open in Chrome/Edge for the complete walkthrough.
