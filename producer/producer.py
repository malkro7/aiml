# producer/producer.py
import json
import math
import os
import socket
import time
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer

# ⬇️ Helpers from your refactored common/preprocess.py
from common.preprocess import df_from_records, expected_columns

# ---------- Config (override via env if needed) ----------
KAFKA_HOST = os.getenv("KAFKA_HOST", "kafka")
KAFKA_PORT = int(os.getenv("KAFKA_PORT", "9092"))
BOOTSTRAP = f"{KAFKA_HOST}:{KAFKA_PORT}"

TOPIC_LOW = os.getenv("TOPIC_LOW", "amount_low")
TOPIC_MED = os.getenv("TOPIC_MED", "amount_medium")
TOPIC_HIGH = os.getenv("TOPIC_HIGH", "amount_high")

CSV_PATH = Path(os.getenv("CSV_PATH", "/data/PS_20174392719_1491204439457_log.csv"))

SEND_SLEEP_S = float(os.getenv("SEND_SLEEP_S", "0.005"))  # small delay to simulate realtime
LINGER_MS = int(os.getenv("KAFKA_LINGER_MS", "5"))
ACKS = os.getenv("KAFKA_ACKS", "all")

REQUIRED_COLUMNS = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",           # keep label if present in your CSV
]


def wait_for_kafka(host=KAFKA_HOST, port=KAFKA_PORT, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=5):
                print(f"✅ Kafka is up at {host}:{port}")
                return
        except Exception as e:
            print(f"⏳ Waiting for Kafka... {e}")
            time.sleep(2)
    raise TimeoutError("Kafka did not become available in time.")


def _clean_value(v):
    """Make values JSON-serializable and avoid NaN (which JSON doesn't like)."""
    if v is None:
        return None
    # Convert pandas/NumPy types to native Python
    if hasattr(v, "item"):
        v = v.item()
    # Replace NaN/inf with None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _clean_record(d: dict) -> dict:
    """Clean a record dict so json.dumps(..., allow_nan=False) won't fail."""
    return {k: _clean_value(v) for k, v in d.items()}


def main():
    wait_for_kafka()

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v, allow_nan=False).encode("utf-8"),
        linger_ms=LINGER_MS,
        acks=ACKS,
    )
    print("✅ Kafka producer started")

    # 1) Load RAW CSV first (guarantees 'amount' exists here if dataset is correct)
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")
    raw_df = pd.read_csv(CSV_PATH)

    if "amount" not in raw_df.columns:
        raise ValueError("'amount' not in CSV; got: " + ", ".join(raw_df.columns))

    # 2) Compute thresholds from RAW (always safe)
    q1 = raw_df["amount"].quantile(0.33)
    q2 = raw_df["amount"].quantile(0.66)
    print(f"ℹ️ amount buckets: <= {q1:.2f} (low), <= {q2:.2f} (med), > {q2:.2f} (high)")

    # 3) Prune for streaming using your helpers, but FORCE‑KEEP 'amount' (+ target if present)
    # df = df_from_records(raw_df)  # drops step/nameOrig/nameDest/isFlaggedFraud
    # cols = expected_columns()     # what your preprocessor knows from meta.json

    # keep = ["amount"] + cols
    # if "isFraud" in df.columns:
    #     keep.append("isFraud")
    # # dedupe while preserving order
    # seen = set()
    # keep = [c for c in keep if (c not in seen and not seen.add(c))]
    # # select existing columns only
    # existing = [c for c in keep if c in df.columns]
    # df = df[existing].copy()

    # if "amount" not in df.columns:
    #     # should not happen because we force-kept 'amount', but guard anyway
    #     raise ValueError("Expected 'amount' column is missing after pruning.")

    df = df_from_records(raw_df)  # already drops step/nameOrig/nameDest/isFlaggedFraud

    # Ensure required columns exist; fail fast if they don't
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input missing required columns: {missing}. "
                     f"Got: {list(df.columns)}")

    # Keep ONLY the raw columns expected on the wire
    df = df[REQUIRED_COLUMNS].copy()

    # 4) Stream records
    sent = 0
    for _, row in df.iterrows():
        amt = row["amount"]
        # Topic selection
        if amt <= q1:
            topic = TOPIC_LOW
        elif amt <= q2:
            topic = TOPIC_MED
        else:
            topic = TOPIC_HIGH

        record = _clean_record(row.to_dict())
        producer.send(topic, value=record)
        sent += 1

        # Small sleep to simulate realtime; reduce for speed if needed
        if SEND_SLEEP_S > 0:
            time.sleep(SEND_SLEEP_S if sent < 2000 else 0)

    producer.flush()
    print(f"✅ Finished streaming {sent} records to Kafka across "
          f"{TOPIC_LOW}/{TOPIC_MED}/{TOPIC_HIGH}")


if __name__ == "__main__":
    main()
