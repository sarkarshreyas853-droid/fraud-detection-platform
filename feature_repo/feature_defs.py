"""Feast definitions: receiver history, one row per (receiver, hour) snapshot."""
from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64

receiver = Entity(name="receiver", join_keys=["nameDest"], value_type=ValueType.STRING, description="Account that receives the money")

receiver_source = FileSource(
    path=str(Path(__file__).parent / "data" / "receiver_snapshots.parquet"),
    timestamp_field="event_timestamp",
)

receiver_stats = FeatureView(
    name="receiver_stats",
    entities=[receiver],
    ttl=timedelta(days=365),
    schema=[
        Field(name="rx_txn_count_total", dtype=Int64),
        Field(name="rx_amount_sum_total", dtype=Float64),
        Field(name="rx_last_step", dtype=Int64),
    ],
    source=receiver_source,
)
