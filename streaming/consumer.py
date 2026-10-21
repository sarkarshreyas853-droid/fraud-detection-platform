"""Read transactions from Kafka and send each one to the scoring API.

  python streaming/consumer.py            (stops by itself after --idle-exit seconds with no new messages)
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idle-exit", type=int, default=20, help="stop after this many idle seconds (0 = run forever)")
    a = ap.parse_args()

    import httpx
    from confluent_kafka import Consumer

    c = Consumer({
        "bootstrap.servers": config.KAFKA_BOOTSTRAP,
        "group.id": "fraud-scorer",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
    })
    c.subscribe([config.KAFKA_TOPIC])
    print(f"Listening on topic '{config.KAFKA_TOPIC}' at {config.KAFKA_BOOTSTRAP} ...")

    n = alerts = 0
    last_msg = time.time()
    with httpx.Client(base_url=config.API_URL, timeout=30) as http:
        try:
            while True:
                msg = c.poll(1.0)
                if msg is None:
                    if a.idle_exit and time.time() - last_msg > a.idle_exit:
                        print("No new messages. Stopping.")
                        break
                    continue
                if msg.error():
                    print("Kafka error:", msg.error())
                    continue
                last_msg = time.time()
                resp = http.post("/score", json=json.loads(msg.value()))
                resp.raise_for_status()
                n += 1
                alerts += resp.json()["is_alert"]
                if n % 500 == 0:
                    print(f"  scored {n:,} | alerts {alerts:,}")
        finally:
            c.close()
    print(f"Done. Scored {n:,} transactions, {alerts:,} alerts.")


if __name__ == "__main__":
    main()
