"""Prepare the Feast feature store and the replay file.

What it does:
  1. Picks a replay window: the last --hours hours of the data (default 24).
  2. Rebuilds each receiver's history from BEFORE that window (same logic as Day 3-4).
  3. Registers the feature view in Feast and loads that history into the online store.
  4. Writes data/replay.parquet: the transactions to replay, in time order.

Run it again any time you want to reset the live state before a fresh replay.
"""
import argparse
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "feature_repo"))

import config  # noqa: E402
from feast import FeatureStore  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(config.FEATURES_PARQUET))
    ap.add_argument("--hours", type=int, default=24, help="length of the replay window in hours (steps); also applies with --start-step")
    ap.add_argument("--start-step", type=int, default=None, help="override the first step of the replay window")
    a = ap.parse_args()

    cols = ["step", "nameDest", "amount", "isFraud", "type_is_transfer"]
    config.REPLAY_PARQUET.parent.mkdir(exist_ok=True)
    feats = pd.read_parquet(a.features, columns=cols)
    feats.index.name = "tx_id"
    max_step = int(feats["step"].max())
    start = a.start_step if a.start_step is not None else max_step - a.hours + 1
    print(f"Data has steps up to {max_step}. Replay window: steps {start} to {start + a.hours - 1}.")

    # 1. replay file
    end = start + a.hours  # exclusive: the window is [start, start + hours)
    window = feats[(feats["step"] >= start) & (feats["step"] < end)].reset_index()
    window["type"] = np.where(window["type_is_transfer"] == 1, "TRANSFER", "CASH_OUT")
    window = window.sort_values(["step", "tx_id"], kind="mergesort")
    replay = window[["tx_id", "step", "type", "amount", "nameDest", "isFraud"]].rename(columns={"isFraud": "label"})
    replay.to_parquet(config.REPLAY_PARQUET, index=False)
    receivers = set(replay["nameDest"])
    print(f"Replay file: {len(replay):,} transactions, {int(replay.label.sum()):,} fraud, {len(receivers):,} receivers")

    # 2. receiver history strictly before the window
    hist = feats[(feats["nameDest"].isin(receivers)) & (feats["step"] < start)][["nameDest", "step", "amount"]].reset_index()
    if hist.empty:
        raise SystemExit("No receiver history before the replay window. Use a smaller --hours or --start-step.")
    hist = hist.sort_values(["nameDest", "step", "tx_id"], kind="mergesort")
    g = hist.groupby("nameDest", sort=False)
    hist["rx_txn_count_total"] = g.cumcount() + 1
    hist["rx_amount_sum_total"] = g["amount"].cumsum()
    snap = hist.drop_duplicates(["nameDest", "step"], keep="last").copy()
    snap["rx_last_step"] = snap["step"].astype("int64")
    snap["event_timestamp"] = config.BASE_TS + pd.to_timedelta(snap["step"], unit="h")
    snap["rx_txn_count_total"] = snap["rx_txn_count_total"].astype("int64")
    snap["rx_amount_sum_total"] = snap["rx_amount_sum_total"].astype("float64")
    snap = snap[["nameDest", "event_timestamp", "rx_txn_count_total", "rx_amount_sum_total", "rx_last_step"]]
    (config.REPO_PATH / "data").mkdir(exist_ok=True)
    snap.to_parquet(config.REPO_PATH / "data" / "receiver_snapshots.parquet", index=False)
    print(f"History snapshots: {len(snap):,} rows for {snap.nameDest.nunique():,} receivers")

    # 3. Feast: fresh online store, register, load latest state
    for f in ("online_store.db", "registry.db"):
        p = config.REPO_PATH / "data" / f
        if p.exists():
            p.unlink()
    import feature_defs
    store = FeatureStore(repo_path=str(config.REPO_PATH))
    store.apply([feature_defs.receiver, feature_defs.receiver_stats])
    store.materialize(
        start_date=(config.BASE_TS - pd.Timedelta(days=1)).to_pydatetime(),
        end_date=(config.BASE_TS + pd.Timedelta(hours=int(start))).to_pydatetime(),
    )
    print("Feast online store loaded.")

    if config.SCORED_LOG.exists():
        config.SCORED_LOG.unlink()
    print("\nDone. Next: run the consistency test, then start the API.")


if __name__ == "__main__":
    main()
