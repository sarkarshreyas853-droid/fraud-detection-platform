"""Real-time scoring logic.

A transaction is scored using receiver history from the Feast online store. That history only
contains hours BEFORE the transaction's own hour, exactly like the training features. To keep
that true while transactions stream in, new information is held back and committed to Feast
only when the clock moves to the next hour ("commit on hour end").
"""
import json
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from feast import FeatureStore

import config


class Scorer:
    def __init__(self):
        self.card = json.loads(config.MODEL_CARD_PATH.read_text())
        self.features = self.card["features"]
        self.threshold = float(self.card["threshold"])
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(config.MODEL_PATH))
        self.store = FeatureStore(repo_path=str(config.REPO_PATH))
        self.lock = threading.Lock()
        self.current_step = None
        self.pending = {}          # receiver -> state before this hour + what arrived this hour
        self.stats = {"scored": 0, "alerts": 0, "late": 0}

    # ---- Feast helpers -------------------------------------------------
    def _read_state(self, dest):
        out = self.store.get_online_features(
            features=[f"{config.FEATURE_VIEW}:{n}" for n in config.STATE_FEATURES],
            entity_rows=[{"nameDest": dest}],
        ).to_dict()
        cnt = out["rx_txn_count_total"][0]
        total = out["rx_amount_sum_total"][0]
        last = out["rx_last_step"][0]
        if cnt is None:
            return 0, 0.0, None
        return int(cnt), float(total), int(last)

    def _commit(self, hour_step):
        """Write the finished hour's updates to the online store."""
        if not self.pending:
            return
        rows = []
        for dest, s in self.pending.items():
            rows.append({
                "nameDest": dest,
                "event_timestamp": config.BASE_TS + pd.Timedelta(hours=int(hour_step)),
                "rx_txn_count_total": int(s["cnt0"] + s["n"]),
                "rx_amount_sum_total": float(s["sum0"] + s["amt"]),
                "rx_last_step": int(hour_step),
            })
        self.store.write_to_online_store(feature_view_name=config.FEATURE_VIEW, df=pd.DataFrame(rows))
        self.pending = {}

    # ---- the one public method ----------------------------------------
    def score(self, tx):
        with self.lock:
            step, dest, amount = int(tx["step"]), tx["nameDest"], float(tx["amount"])
            late = self.current_step is not None and step < self.current_step
            if late:
                self.stats["late"] += 1          # out-of-order event: scored, but does not change state
            else:
                if self.current_step is not None and step > self.current_step:
                    self._commit(self.current_step)
                self.current_step = step

            if dest in self.pending:
                cnt, total, last = self.pending[dest]["cnt0"], self.pending[dest]["sum0"], self.pending[dest]["last0"]
            else:
                cnt, total, last = self._read_state(dest)

            avg = total / cnt if cnt > 0 else 0.0
            row = {
                "rx_prev_count": float(cnt),
                "rx_avg_amount": avg,
                "rx_hours_since_last": float(step - last) if last is not None else -1.0,
                "amount": amount,
                "hour_of_day": float(step % 24),
                "type_is_transfer": 1.0 if tx["type"] == "TRANSFER" else 0.0,
                "amount_to_rx_avg": amount / avg if avg > 0 else 0.0,
            }
            x = pd.DataFrame([[row[f] for f in self.features]], columns=self.features)
            prob = float(self.model.predict_proba(x)[:, 1][0])

            if not late:
                p = self.pending.setdefault(dest, {"cnt0": cnt, "sum0": total, "last0": last, "n": 0, "amt": 0.0})
                p["n"] += 1
                p["amt"] += amount

            alert = prob >= self.threshold
            self.stats["scored"] += 1
            self.stats["alerts"] += int(alert)
            return {
                "tx_id": tx.get("tx_id"),
                "step": step,
                "fraud_score": prob,
                "is_alert": bool(alert),
                "threshold": self.threshold,
                "late_event": late,
                "features": {k: row[k] for k in self.features},
                "model": self.card["run"],
                "scored_at": datetime.now(timezone.utc).isoformat(),
            }
