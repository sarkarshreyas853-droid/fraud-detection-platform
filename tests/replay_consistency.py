"""Check that the LIVE service computes the same features and scores as the TRAINING pipeline.

Replays the first N transactions of the replay window through the API (in-process, no server or
Kafka needed) and compares against the Day 3-4 feature table and the Day 5-6 model.

Run:  python tests/replay_consistency.py            (after scripts/prepare_feast.py)
Then run prepare_feast.py again to reset the live state before starting the real service.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=5000)
    a = ap.parse_args()

    replay = pd.read_parquet(config.REPLAY_PARQUET).head(a.n)
    offline = pd.read_parquet(config.FEATURES_PARQUET)
    print(f"Replaying {len(replay):,} transactions through the API...")

    results = []
    with TestClient(app) as client:
        card = client.get("/health").json()
        feats = card["features"]
        for r in replay.itertuples(index=False):
            resp = client.post("/score", json={
                "tx_id": int(r.tx_id), "step": int(r.step), "type": r.type,
                "amount": float(r.amount), "nameDest": r.nameDest, "label": int(r.label)})
            resp.raise_for_status()
            results.append(resp.json())

    online = pd.DataFrame([{**x["features"], "score": x["fraud_score"], "tx_id": x["tx_id"]} for x in results]).set_index("tx_id")
    expected = offline.loc[online.index, feats]

    all_ok = True
    for c in feats:
        ok = np.allclose(online[c].to_numpy(), expected[c].to_numpy(), rtol=1e-6, atol=1e-6)
        all_ok &= ok
        print(f"  {c:22s} live == training: {ok}")

    import xgboost as xgb
    m = xgb.XGBClassifier(); m.load_model(str(config.MODEL_PATH))
    offline_scores = m.predict_proba(expected)[:, 1]
    score_ok = np.allclose(online["score"].to_numpy(), offline_scores, atol=1e-5)
    all_ok &= score_ok
    print(f"  {'fraud score':22s} live == offline model: {score_ok}")

    alerts = sum(x["is_alert"] for x in results)
    print(f"\nAlerts raised: {alerts:,} of {len(results):,}")
    assert all_ok, "Live and training results differ. Send the output above to Claude."
    print("Live-vs-training consistency check PASSED")


if __name__ == "__main__":
    main()
