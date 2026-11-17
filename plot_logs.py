#!/usr/bin/env python3
"""
plot_logs.py -- Visualize E-Commerce MLOps Pipeline logs.

Usage:
    python plot_logs.py             # show all charts
    python plot_logs.py --log exp   # experiments only
    python plot_logs.py --log drift # drift only

Reads: logs/experiments.csv, logs/drift.csv, logs/latency.csv
Opens charts in your browser (no matplotlib needed -- pure HTML).
"""

import csv
import json
import argparse
import webbrowser
from pathlib import Path
from datetime import datetime

LOGS = Path(__file__).parent / "logs"
OUTPUT = Path(__file__).parent / "logs_report.html"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_experiment_chart(rows: list[dict]) -> str:
    if not rows:
        return "<p>No experiment data yet. Run <code>python main.py</code> first.</p>"

    labels = [r.get("timestamp", f"run {i+1}")[:19] for i, r in enumerate(rows)]
    metrics = ["accuracy", "precision", "recall", "f1"]
    colors = {"accuracy": "#4e79a7", "precision": "#f28e2b", "recall": "#e15759", "f1": "#76b7b2"}

    series = []
    for m in metrics:
        vals = [safe_float(r.get(m)) for r in rows]
        series.append({"name": m, "color": colors[m], "values": vals})

    # Build SVG bar chart
    w, h, pad_l, pad_b, pad_t, pad_r = 820, 300, 60, 50, 20, 20
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_b - pad_t
    n = len(rows)
    group_w = plot_w / max(n, 1)
    bar_w = group_w / (len(metrics) + 1)

    svg_bars = []
    for i, row in enumerate(rows):
        for j, s in enumerate(series):
            v = s["values"][i]
            if v is None:
                continue
            bar_h = v * plot_h
            x = pad_l + i * group_w + j * bar_w + bar_w * 0.5
            y = pad_t + (plot_h - bar_h)
            svg_bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
                f'fill="{s["color"]}" opacity="0.85">'
                f'<title>{s["name"]}: {v:.4f} @ {labels[i]}</title></rect>'
            )

    # Y-axis ticks
    ticks = [0, 0.25, 0.5, 0.75, 1.0]
    tick_svg = "".join(
        f'<line x1="{pad_l}" y1="{pad_t + plot_h - t*plot_h:.1f}" '
        f'x2="{pad_l + plot_w}" y2="{pad_t + plot_h - t*plot_h:.1f}" '
        f'stroke="#ddd" stroke-width="1"/>'
        f'<text x="{pad_l - 6}" y="{pad_t + plot_h - t*plot_h + 4:.1f}" '
        f'text-anchor="end" font-size="11" fill="#666">{t:.2f}</text>'
        for t in ticks
    )

    # X-axis labels
    x_labels = "".join(
        f'<text x="{pad_l + (i + 0.5) * group_w:.1f}" y="{h - 8}" '
        f'text-anchor="middle" font-size="10" fill="#888">{labels[i][5:]}</text>'
        for i in range(n)
    )

    # Legend
    legend = "".join(
        f'<rect x="{pad_l + j*110}" y="{h - 42}" width="12" height="12" fill="{s["color"]}"/>'
        f'<text x="{pad_l + j*110 + 16}" y="{h - 32}" font-size="12" fill="#333">{s["name"]}</text>'
        for j, s in enumerate(series)
    )

    return f"""
<h3>Experiment Metrics Over Runs</h3>
<svg width="{w}" height="{h}" style="font-family: sans-serif; display:block; overflow:visible;">
  {tick_svg}
  {"".join(svg_bars)}
  {x_labels}
  {legend}
  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+plot_h}" stroke="#999" stroke-width="1"/>
  <line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{pad_l+plot_w}" y2="{pad_t+plot_h}" stroke="#999" stroke-width="1"/>
</svg>
<table>
<tr><th>Run</th><th>Timestamp</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>n_features</th></tr>
{"".join(
    f'<tr><td>{i+1}</td><td>{r.get("timestamp","")[:19]}</td>'
    f'<td>{r.get("accuracy","")}</td><td>{r.get("precision","")}</td>'
    f'<td>{r.get("recall","")}</td><td>{r.get("f1","")}</td>'
    f'<td>{r.get("n_features","")}</td></tr>'
    for i, r in enumerate(rows)
)}
</table>"""


def build_drift_chart(rows: list[dict]) -> str:
    if not rows:
        return "<p>No drift data yet. Run <code>python main.py</code> first.</p>"

    features = list({r["feature"] for r in rows if "feature" in r})
    timestamps = sorted({r.get("timestamp", "")[:19] for r in rows})

    rows_html = "".join(
        f'<tr><td>{r.get("timestamp","")[:19]}</td>'
        f'<td>{r.get("feature","")}</td>'
        f'<td>{r.get("train_mean","")}</td>'
        f'<td>{r.get("current_mean","")}</td>'
        f'<td>{r.get("mean_shift","")}</td>'
        f'<td style="background:{"#fee2e2" if r.get("drifted","").lower()=="true" else "#dcfce7"}">'
        f'{"DRIFTED" if r.get("drifted","").lower()=="true" else "OK"}</td></tr>'
        for r in rows
    )

    return f"""
<h3>Drift Detection Log</h3>
<table>
<tr><th>Timestamp</th><th>Feature</th><th>Train Mean</th><th>Current Mean</th><th>Mean Shift</th><th>Status</th></tr>
{rows_html}
</table>"""


def build_latency_chart(rows: list[dict]) -> str:
    if not rows:
        return "<p>No latency data yet. Run batch inference first.</p>"

    rows_html = "".join(
        f'<tr><td>{r.get("timestamp","")[:19]}</td>'
        f'<td>{r.get("n_samples","")}</td>'
        f'<td>{r.get("latency_ms","")}</td>'
        f'<td>{r.get("model_version","")}</td></tr>'
        for r in rows
    )

    return f"""
<h3>Inference Latency Log</h3>
<table>
<tr><th>Timestamp</th><th>Samples</th><th>Latency (ms)</th><th>Model Version</th></tr>
{rows_html}
</table>"""


def build_registry_section() -> str:
    path = Path(__file__).parent / "models" / "registry.json"
    if not path.exists():
        return "<p>No registry.json yet.</p>"
    reg = json.loads(path.read_text(encoding="utf-8"))
    model_list = reg.get("models", []) if isinstance(reg, dict) else reg
    rows = "".join(
        f'<tr>'
        f'<td>{m.get("version","")}</td>'
        f'<td style="background:{"#fef9c3" if m.get("status")=="production" else ""}">'
        f'{m.get("status","")}</td>'
        f'<td>{m.get("registered_at","")[:19] if m.get("registered_at") else ""}</td>'
        f'<td>{m.get("metrics",{}).get("accuracy","")}</td>'
        f'<td>{m.get("metrics",{}).get("recall","")}</td></tr>'
        for m in model_list
    )
    return f"""
<h3>Model Registry</h3>
<table>
<tr><th>Version</th><th>Status</th><th>Registered</th><th>Accuracy</th><th>Recall</th></tr>
{rows}
</table>"""


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 900px; margin: 40px auto; padding: 0 24px; color: #1a1a1a; }
h1 { font-size: 22px; border-bottom: 2px solid #333; padding-bottom: 8px; }
h3 { font-size: 16px; margin-top: 36px; color: #333; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }
th { background: #333; color: white; padding: 7px 10px; text-align: left; }
td { padding: 6px 10px; border-bottom: 1px solid #eee; }
tr:nth-child(even) td { background: #f9f9f9; }
code { background: #f4f4f4; padding: 1px 5px; border-radius: 3px; font-size: 12px; }
p { color: #666; }
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", choices=["exp", "drift", "latency", "all"], default="all")
    args = parser.parse_args()

    exp_rows = read_csv(LOGS / "experiments.csv")
    drift_rows = read_csv(LOGS / "drift.csv")
    latency_rows = read_csv(LOGS / "latency.csv")

    sections = []
    if args.log in ("exp", "all"):
        sections.append(build_experiment_chart(exp_rows))
        sections.append(build_registry_section())
    if args.log in ("drift", "all"):
        sections.append(build_drift_chart(drift_rows))
    if args.log in ("latency", "all"):
        sections.append(build_latency_chart(latency_rows))

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>MLOps Pipeline Logs</title>
<style>{CSS}</style></head>
<body>
<h1>MLOps Pipeline -- Run Report ({now})</h1>
{"".join(sections)}
</body></html>"""

    OUTPUT.write_text(html, encoding="utf-8")
    print(f"Written: {OUTPUT}")
    webbrowser.open(OUTPUT.as_uri())


if __name__ == "__main__":
    main()
