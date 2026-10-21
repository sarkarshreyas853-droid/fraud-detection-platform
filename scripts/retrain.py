"""Day 11: retrain a challenger model and promote it only if it beats the current one (the champion).

  python scripts/retrain.py              # evaluate on the hours the live service scored, promote if better
  python scripts/retrain.py --dry-run    # compare only, change nothing
  python scripts/retrain.py --eval-start 672 --hours 24

How it stays honest
  * Evaluation window: the hours being monitored (default: the steps in data/scored.jsonl).
  * The challenger only learns from steps BEFORE that window (labels you would really have by then):
    the last 15% of those steps validate it (early stopping + threshold), the rest train it.
  * Champion and challenger are scored on the same unseen window.
  * Promotion needs a PR-AUC gain of at least MIN_GAIN. The old model is archived in model/archive/.
  * Restart the API afterwards so it loads the new model.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

MIN_GAIN = 0.01
PARAMS = dict(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
              tree_method="hist", eval_metric="aucpr", early_stopping_rounds=30, random_state=42, n_jobs=-1)


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.nanargmax(f1))])


def scores(y, p, thr):
    flag = p >= thr
    tp = int((flag & (y == 1)).sum())
    k = max(1, int(len(y) * 0.01))
    top = np.argsort(-p)[:k]
    return dict(pr_auc=float(average_precision_score(y, p)), precision=tp / max(1, int(flag.sum())),
                recall=tp / max(1, int(y.sum())), alerts=int(flag.sum()),
                recall_at_1pct_alerts=float(y[top].sum() / max(1, y.sum())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-start", type=int, default=None, help="first step of the evaluation window")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--dry-run", action="store_true", help="compare only, do not replace the model")
    a = ap.parse_args()

    card = json.loads(config.MODEL_CARD_PATH.read_text())
    features = card["features"]
    feats = pd.read_parquet(config.FEATURES_PARQUET, columns=features + ["step", "isFraud"])

    start = a.eval_start
    if start is None:
        if config.SCORED_LOG.exists():
            start = min(json.loads(x)["step"] for x in config.SCORED_LOG.open(encoding="utf-8") if x.strip())
        else:
            start = int(feats.step.max()) - a.hours + 1
    end = start + a.hours
    history = feats[feats.step < start]
    window = feats[(feats.step >= start) & (feats.step < end)]
    if window.isFraud.nunique() < 2:
        sys.exit("The evaluation window needs both fraud and normal transactions. Pick another --eval-start.")

    t_valid = int(start - 1 - 0.15 * (start - 1))          # last 15% of the history validates
    tr, va = history[history.step <= t_valid], history[history.step > t_valid]
    print(f"Challenger: train steps 1-{t_valid} ({len(tr):,} rows), validate {t_valid + 1}-{start - 1} ({len(va):,} rows)")
    print(f"Evaluation window (unseen by both): steps {start}-{end - 1} ({len(window):,} rows, "
          f"{int(window.isFraud.sum())} fraud, baseline PR-AUC {window.isFraud.mean():.3f})")
    print(f"Champion: '{card['run']}', trained on steps 1-{card['split']['train_max_step']}\n")

    spw = float((tr.isFraud == 0).sum() / max(1, (tr.isFraud == 1).sum())) if card.get("class_weighting", True) else 1.0
    challenger = xgb.XGBClassifier(**PARAMS, scale_pos_weight=spw)
    challenger.fit(tr[features], tr.isFraud, eval_set=[(va[features], va.isFraud)], verbose=False)
    ch_thr = best_f1_threshold(va.isFraud.to_numpy(), challenger.predict_proba(va[features])[:, 1])

    champion = xgb.XGBClassifier()
    champion.load_model(config.MODEL_PATH)

    y = window.isFraud.to_numpy().astype(int)
    res = {"champion": scores(y, champion.predict_proba(window[features])[:, 1], card["threshold"]),
           "challenger": scores(y, challenger.predict_proba(window[features])[:, 1], ch_thr)}
    print(f"{'':11s} {'PR-AUC':>7s} {'precision':>9s} {'recall':>7s} {'alerts':>7s} {'recall@1%':>9s}")
    for k, r in res.items():
        print(f"{k:11s} {r['pr_auc']:7.3f} {r['precision']:9.2f} {r['recall']:7.2f} {r['alerts']:7,} {r['recall_at_1pct_alerts']:9.2f}")

    gain = res["challenger"]["pr_auc"] - res["champion"]["pr_auc"]
    promote = gain >= MIN_GAIN
    print(f"\nPR-AUC gain: {gain:+.3f} (needs {MIN_GAIN:+.3f})")

    log = dict(at=datetime.now(timezone.utc).isoformat(timespec="seconds"), eval_steps=[start, end - 1],
               challenger_train_max_step=t_valid, results=res, gain=gain, promoted=bool(promote and not a.dry_run))
    reports = ROOT / "reports"; reports.mkdir(exist_ok=True)
    with (reports / "retrain_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(log) + "\n")

    if not promote:
        print("DECISION: keep the champion. The challenger is not clearly better.")
        return
    if a.dry_run:
        print("DECISION: the challenger would be promoted (dry run, nothing changed).")
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive = ROOT / "model" / "archive"; archive.mkdir(exist_ok=True)
    shutil.copy(config.MODEL_PATH, archive / f"{stamp}_model.json")
    shutil.copy(config.MODEL_CARD_PATH, archive / f"{stamp}_model_card.json")
    challenger.save_model(config.MODEL_PATH)
    new_card = dict(card, run=f"retrained {stamp}", threshold=ch_thr,
                    test_pr_auc=res["challenger"]["pr_auc"], test_precision=res["challenger"]["precision"],
                    test_recall=res["challenger"]["recall"],
                    test_recall_at_1pct_alerts=res["challenger"]["recall_at_1pct_alerts"],
                    split={"train_max_step": t_valid, "valid_max_step": start - 1},
                    evaluated_on_steps=[start, end - 1], previous=card["run"])
    config.MODEL_CARD_PATH.write_text(json.dumps(new_card, indent=2))
    (reports / "RETRAIN_NEEDED").unlink(missing_ok=True)
    print(f"DECISION: challenger PROMOTED. Old model saved in model/archive/{stamp}_*.")
    print("Restart the API so it loads the new model.")


if __name__ == "__main__":
    main()
