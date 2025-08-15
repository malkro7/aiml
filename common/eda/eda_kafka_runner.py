# common/eda/eda_kafka_runner.py
"""
Kafka → DataFrame → single-PDF EDA (no CSV files required).
Does NOT expect 'nameOrig', 'nameDest', 'step', or 'isFlaggedFraud'.

Env:
  BOOTSTRAP_SERVERS: kafka:9092
  KAFKA_TOPICS: fraud.high,fraud.medium,fraud.low
  GROUP_ID: eda-client1
  MAX_MESSAGES: 300000 (default 200000)
  TIMEOUT_SEC: 90
  CLIENT_ID: for output naming
  EDA_SAMPLE: optional downsample before plotting (int)
  KEY_PATH: if payload is nested (e.g., "payload")
  STRICT_KEYS: true|false (default false) – if true, drop rows missing REQUIRED keys
"""

import os, time, json, datetime as dt
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from eda_fraud_report import run  # uses the PDF generator you already have

# ── Expected post-preprocess schema ──────────────────────────────────────────
# Keep this minimal & robust. We'll accept extras but only coerce known numerics.
REQUIRED_KEYS = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
]

NUMERIC_KEYS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
]

# ── Kafka consumer factory (lazy import so unit tests don't need kafka-python)
def _mk_consumer(topics: List[str], servers: List[str], group_id: str):
    from kafka import KafkaConsumer
    return KafkaConsumer(
        *topics,
        bootstrap_servers=servers,
        group_id=group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda v: v.decode("utf-8") if v else None,
        consumer_timeout_ms=1000,
        max_partition_fetch_bytes=5 * 1024 * 1024,
        fetch_max_bytes=64 * 1024 * 1024,
    )

def _pluck(record: Dict[str, Any], key_path: Optional[str]) -> Dict[str, Any]:
    if key_path:
        return record.get(key_path, {})
    return record

def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    for k in NUMERIC_KEYS:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")
    if "type" in df.columns:
        df["type"] = df["type"].astype("category")
    return df

def _collect_from_kafka(
    topics: List[str],
    servers: List[str],
    group_id: str,
    max_messages: int,
    timeout_sec: int,
    key_path: Optional[str],
    strict_keys: bool,
) -> pd.DataFrame:
    consumer = _mk_consumer(topics, servers, group_id)
    start = time.monotonic()
    rows = []

    try:
        while len(rows) < max_messages and (time.monotonic() - start) < timeout_sec:
            for msg in consumer:
                val = msg.value
                if not isinstance(val, dict):
                    continue
                rec = _pluck(val, key_path)

                # Build row for REQUIRED_KEYS only; ignore nameOrig/nameDest/step/isFlaggedFraud
                row = {k: rec.get(k, None) for k in REQUIRED_KEYS}

                if strict_keys and any(row[k] is None for k in REQUIRED_KEYS):
                    continue

                # If producer includes any extra fields (e.g., engineered deltas),
                # merge them in without forcing a schema.
                for k, v in rec.items():
                    if k not in row:
                        row[k] = v

                rows.append(row)
                if len(rows) >= max_messages:
                    break
            time.sleep(0.15)
    finally:
        consumer.close()

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = _coerce_types(df)
    return df

def main():
    servers = os.getenv("BOOTSTRAP_SERVERS", "kafka:9092").split(",")
    topics = [t.strip() for t in os.getenv("KAFKA_TOPICS", "fraud.transactions").split(",") if t.strip()]
    client_id = os.getenv("CLIENT_ID", "X")
    group_id = os.getenv("GROUP_ID", f"eda-{client_id}")
    max_messages = int(os.getenv("MAX_MESSAGES", "200000"))
    timeout_sec = int(os.getenv("TIMEOUT_SEC", "90"))
    key_path = os.getenv("KEY_PATH")
    strict_keys = os.getenv("STRICT_KEYS", "false").lower() == "true"
    sample = int(os.getenv("EDA_SAMPLE")) if os.getenv("EDA_SAMPLE") else None

    out_dir = "/data/eda"
    os.makedirs(out_dir, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
    out_prefix = os.path.join(out_dir, f"client{client_id}_{ts}")

    print(f"[EDA-KAFKA] servers={servers} topics={topics} group={group_id} "
          f"max={max_messages} timeout={timeout_sec}s strict={strict_keys} key_path={key_path}")

    df = _collect_from_kafka(
        topics=topics,
        servers=servers,
        group_id=group_id,
        max_messages=max_messages,
        timeout_sec=timeout_sec,
        key_path=key_path,
        strict_keys=strict_keys,
    )

    if df.empty:
        print("[EDA-KAFKA] No messages collected — skipping EDA.")
        return

    # The PDF generator already tolerates missing 'step'/'isFlaggedFraud' and drops them if present.
    if "isFraud" not in df.columns:
        raise SystemExit("[EDA-KAFKA] 'isFraud' column missing in consumed records.")

    # Write a temp CSV into the bind mount and invoke the existing report pipeline
    tmp_csv = os.path.join(out_dir, f"client{client_id}_{ts}_sample.csv")
    df.to_csv(tmp_csv, index=False)
    print(f"[EDA-KAFKA] Collected rows: {len(df):,}. Temp CSV -> {tmp_csv}")

    run(tmp_csv, out_prefix, sample=sample)

if __name__ == "__main__":
    main()
