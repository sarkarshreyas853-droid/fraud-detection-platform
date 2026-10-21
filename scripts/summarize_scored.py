"""Summarise data/scored.jsonl: how good were the live decisions?

Run:  python scripts/summarize_scored.py
Needs a finished replay (python streaming/producer.py ...), because it compares the
service's alerts with the true labels that the replay sends along.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "data" / "scored.jsonl"
if not path.exists():
    sys.exit("No data/scored.jsonl yet. Start the service and run the producer first.")

rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
df = pd.DataFrame(rows)
df = df[df["label"].notna()].copy()
if df.empty:
    sys.exit("scored.jsonl has no labelled rows (the replay sends labels; live traffic would not).")

y = df["label"].astype(int)
alert = df["is_alert"].astype(bool)
tp = int((alert & (y == 1)).sum())
fp = int((alert & (y == 0)).sum())
fn = int((~alert & (y == 1)).sum())
tn = int((~alert & (y == 0)).sum())
frauds = int(y.sum())

print(f"Transactions scored      : {len(df):,}")
print(f"  of which truly fraud   : {frauds:,}  ({frauds / len(df):.1%})")
print(f"Alerts raised            : {int(alert.sum()):,}  ({alert.mean():.1%} of traffic)")
print()
print(f"Caught fraud (TP)        : {tp:,}")
print(f"False alarms (FP)        : {fp:,}")
print(f"Missed fraud (FN)        : {fn:,}")
print(f"Correctly left alone (TN): {tn:,}")
print()
print(f"Precision (alerts that were real fraud): {tp / max(tp + fp, 1):.2f}")
print(f"Recall    (fraud that was caught)      : {tp / max(tp + fn, 1):.2f}")
if 0 < frauds < len(df):
    ap = average_precision_score(y, df["fraud_score"])
    print(f"PR-AUC on this window                  : {ap:.3f}   (doing nothing smart would score {frauds / len(df):.3f})")
print()
print("Note: this is only the last-24-hours replay window, so the numbers are not")
print("comparable to the full test-set results in the model card.")