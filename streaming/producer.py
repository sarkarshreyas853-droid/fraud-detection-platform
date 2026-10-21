"""Replay transactions in time order.

  python streaming/producer.py --mode http  --rate 100      # straight to the API (no Kafka needed)
  python streaming/producer.py --mode kafka --rate 100      # into the Kafka topic (needs docker compose up)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def payload(r):
    return {"tx_id": int(r.tx_id), "step": int(r.step), "type": r.type, "amount": float(r.amount),
            "nameDest": r.nameDest, "label": int(r.label)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["http", "kafka"], default="http")
    ap.add_argument("--rate", type=float, default=100, help="transactions per second (0 = as fast as possible)")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many (0 = all)")
    a = ap.parse_args()

    df = pd.read_parquet(config.REPLAY_PARQUET)
    if a.limit:
        df = df.head(a.limit)
    delay = 1.0 / a.rate if a.rate > 0 else 0
    print(f"Replaying {len(df):,} transactions via {a.mode} at {a.rate}/s")

    if a.mode == "kafka":
        from confluent_kafka import Producer
        p = Producer({"bootstrap.servers": config.KAFKA_BOOTSTRAP})
        errors = []

        def on_delivery(err, msg):
            if err is not None:
                errors.append(str(err))

        for i, r in enumerate(df.itertuples(index=False), 1):
            p.produce(config.KAFKA_TOPIC, key=r.nameDest, value=json.dumps(payload(r)), callback=on_delivery)
            p.poll(0)
            if delay:
                time.sleep(delay)
            if i % 500 == 0:
                print(f"  sent {i:,}")
        left = p.flush(30)
        if errors or left:
            raise SystemExit(f"Kafka delivery problem ({len(errors)} errors, {left} undelivered). First: {errors[:1]}")
    else:
        import httpx
        with httpx.Client(base_url=config.API_URL, timeout=30) as c:
            alerts = 0
            for i, r in enumerate(df.itertuples(index=False), 1):
                resp = c.post("/score", json=payload(r))
                resp.raise_for_status()
                alerts += resp.json()["is_alert"]
                if delay:
                    time.sleep(delay)
                if i % 500 == 0:
                    print(f"  scored {i:,} | alerts so far {alerts:,}")
    print("Replay finished.")


if __name__ == "__main__":
    main()
