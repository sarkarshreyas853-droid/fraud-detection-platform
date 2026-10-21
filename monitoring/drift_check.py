"""Day 11: drift monitoring and the retraining trigger.

Compares recent traffic ("current") with the most recent data the model was checked on ("reference":
its validation period from the model card). Using the whole training period instead would make the
receiver-history features (counts that only grow over time) look drifted forever, even after retraining.

  python monitoring/drift_check.py                       # current = what the live service scored (data/scored.jsonl)
  python monitoring/drift_check.py --start-step 672      # current = 24 hours of history from that step

Writes
  reports/drift_report.html   visual report (Evidently), open it in a browser
  reports/drift_summary.json  numbers + the retrain decision
  reports/RETRAIN_NEEDED      only created when a rule fires (a scheduler or CI job can watch for it)

Decision rules (edit the constants below):
  1. Data drift:   at least DRIFT_SHARE_LIMIT of the model's features have PSI >= PSI_LIMIT
  2. Performance:  PR-AUC on labelled current data < PERF_RATIO_LIMIT x the model card's test PR-AUC
                   (only checked when the current data holds at least MIN_FRAUD_FOR_PERF fraud cases)
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

PSI_LIMIT = 0.2            # common rule of thumb: <0.1 stable, 0.1-0.2 moderate, >0.2 significant shift
DRIFT_SHARE_LIMIT = 0.5    # retrain if half or more of the features shifted significantly
PERF_RATIO_LIMIT = 0.8     # retrain if PR-AUC fell below 80% of the test PR-AUC at training time
MIN_FRAUD_FOR_PERF = 30    # fewer fraud cases than this gives a PR-AUC too noisy to act on
REF_SAMPLE = 50_000        # rows of reference data sent to the Evidently report (keeps it fast)
REPORTS = ROOT / "reports"


def psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Population Stability Index with bins taken from the reference quantiles."""
    ref, cur = ref.dropna().to_numpy(float), cur.dropna().to_numpy(float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                                    # (almost) constant or binary feature: use its values
        vals = np.union1d(np.unique(ref), np.unique(cur))
        r = np.array([(ref == v).mean() for v in vals])
        c = np.array([(cur == v).mean() for v in vals])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref, edges)[0] / len(ref)
        c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def load_current(a, features):
    if a.start_step is None:
        if not config.SCORED_LOG.exists():
            sys.exit("No data/scored.jsonl. Replay some traffic first, or pass --start-step.")
        rows = [json.loads(x) for x in config.SCORED_LOG.open(encoding="utf-8") if x.strip()]
        cur = pd.DataFrame([r["features"] for r in rows])[features]
        cur["isFraud"] = [r.get("label") for r in rows]
        cur["fraud_score"] = [r["fraud_score"] for r in rows]
        steps = [r["step"] for r in rows]
        return cur, f"live service log ({len(cur):,} scored transactions, steps {min(steps)}-{max(steps)})"
    feats = pd.read_parquet(config.FEATURES_PARQUET, columns=features + ["step", "isFraud"])
    cur = feats[(feats.step >= a.start_step) & (feats.step < a.start_step + a.hours)].drop(columns="step")
    if cur.empty:
        sys.exit("No rows in that window.")
    return cur.reset_index(drop=True), f"history steps {a.start_step}-{a.start_step + a.hours - 1} ({len(cur):,} rows)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-step", type=int, default=None, help="use history from this step instead of the live log")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--no-html", action="store_true", help="skip the Evidently report")
    a = ap.parse_args()

    card = json.loads(config.MODEL_CARD_PATH.read_text())
    features = card["features"]
    ref_lo, ref_hi = card["split"]["train_max_step"] + 1, card["split"]["valid_max_step"]

    feats = pd.read_parquet(config.FEATURES_PARQUET, columns=features + ["step"])
    ref = feats[(feats.step >= ref_lo) & (feats.step <= ref_hi)][features].reset_index(drop=True)
    cur, cur_desc = load_current(a, features)
    print(f"Reference: the model's validation period, steps {ref_lo}-{ref_hi} ({len(ref):,} rows)")
    print(f"Current  : {cur_desc}\n")

    # 1. data drift per feature
    scores = {f: psi(ref[f], cur[f]) for f in features}
    drifted = [f for f, v in scores.items() if v >= PSI_LIMIT]
    print(f"{'feature':22s} {'PSI':>7s}")
    for f, v in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"{f:22s} {v:7.3f}  {'DRIFT' if v >= PSI_LIMIT else ('moderate' if v >= 0.1 else '')}")
    drift_share = len(drifted) / len(features)

    # 2. performance, if labels are available
    perf = None
    lab = cur[cur["isFraud"].notna()] if "isFraud" in cur else cur.iloc[0:0]
    n_fraud = int(lab["isFraud"].sum()) if len(lab) else 0
    if n_fraud >= MIN_FRAUD_FOR_PERF and n_fraud < len(lab):
        if "fraud_score" in lab:
            p = lab["fraud_score"].to_numpy()
        else:                                             # history mode: score with the current model
            import xgboost as xgb
            m = xgb.XGBClassifier(); m.load_model(config.MODEL_PATH)
            p = m.predict_proba(lab[features])[:, 1]
        y = lab["isFraud"].astype(int).to_numpy()
        perf = dict(pr_auc=float(average_precision_score(y, p)), baseline=float(y.mean()),
                    reference_test_pr_auc=float(card["test_pr_auc"]), n_fraud=n_fraud, n=int(len(y)))
        perf["ratio"] = perf["pr_auc"] / perf["reference_test_pr_auc"]

    # 3. decision
    reasons = []
    if drift_share >= DRIFT_SHARE_LIMIT:
        reasons.append(f"data drift: {len(drifted)}/{len(features)} features have PSI >= {PSI_LIMIT}")
    if perf and perf["ratio"] < PERF_RATIO_LIMIT:
        reasons.append(f"performance: PR-AUC {perf['pr_auc']:.3f} is {perf['ratio']:.0%} of the training-time "
                       f"{perf['reference_test_pr_auc']:.3f} (limit {PERF_RATIO_LIMIT:.0%})")
    retrain = bool(reasons)

    print(f"\nDrifted features: {len(drifted)}/{len(features)} ({drift_share:.0%}); rule fires at {DRIFT_SHARE_LIMIT:.0%}")
    if perf:
        if a.start_step is None:
            print("(Scores come from the live log, i.e. from the model that was serving when they were made.)")
        print(f"PR-AUC now {perf['pr_auc']:.3f} vs {perf['reference_test_pr_auc']:.3f} at training time "
              f"({perf['ratio']:.0%}); rule fires below {PERF_RATIO_LIMIT:.0%}. Baseline {perf['baseline']:.3f}, {n_fraud} fraud.")
    else:
        print(f"Performance not checked (needs labels and at least {MIN_FRAUD_FOR_PERF} fraud cases; found {n_fraud}).")

    REPORTS.mkdir(exist_ok=True)
    flag = REPORTS / "RETRAIN_NEEDED"
    summary = dict(checked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), model=card["run"],
                   reference=f"steps {ref_lo}-{ref_hi}", current=cur_desc, psi=scores, psi_limit=PSI_LIMIT,
                   drifted_features=drifted, drift_share=drift_share, drift_share_limit=DRIFT_SHARE_LIMIT,
                   performance=perf, perf_ratio_limit=PERF_RATIO_LIMIT, retrain=retrain, reasons=reasons)
    (REPORTS / "drift_summary.json").write_text(json.dumps(summary, indent=2))
    if retrain:
        flag.write_text("\n".join(reasons) + "\n")
        print("\nDECISION: RETRAIN")
        for r in reasons:
            print("  -", r)
        print("Next: python scripts/retrain.py")
    else:
        flag.unlink(missing_ok=True)
        print("\nDECISION: no retraining needed")

    # 4. visual report
    if not a.no_html:
        os.environ.setdefault("DO_NOT_TRACK", "1")       # switch off Evidently's usage telemetry
        try:
            from evidently import Report
            from evidently.presets import DataDriftPreset
            r = ref.sample(min(REF_SAMPLE, len(ref)), random_state=0)
            snap = Report([DataDriftPreset()]).run(current_data=cur[features], reference_data=r)
            snap.save_html(str(REPORTS / "drift_report.html"))
            print("\nVisual report: reports/drift_report.html (open it in your browser)")
            print("Note: Evidently's per-column tests are very sensitive on large samples; the decision above uses PSI.")
        except Exception as e:
            print(f"\n(Evidently report skipped: {e!r})")
    print("Summary: reports/drift_summary.json")


if __name__ == "__main__":
    main()
