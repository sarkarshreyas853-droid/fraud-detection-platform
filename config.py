"""Shared settings. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parent
REPO_PATH = ROOT / "feature_repo"                       # Feast repository
FEATURES_PARQUET = ROOT / "data" / "features_full.parquet"   # from the Day 3-4 outputs
REPLAY_PARQUET = ROOT / "data" / "replay.parquet"            # written by scripts/prepare_feast.py
SCORED_LOG = ROOT / "data" / "scored.jsonl"                  # one line per scored transaction
MODEL_PATH = ROOT / "model" / "model.json"                   # from the Day 5-6 outputs
MODEL_CARD_PATH = ROOT / "model" / "model_card.json"

# Step 0 of PaySim = this moment. One step = one hour. Must match the Day 3-4 notebook.
BASE_TS = pd.Timestamp("2026-01-01", tz="UTC")

FEATURE_VIEW = "receiver_stats"
STATE_FEATURES = ["rx_txn_count_total", "rx_amount_sum_total", "rx_last_step"]

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "transactions")
