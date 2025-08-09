# producer/producer.py
import pandas as pd
import json
import time
import socket

from kafka import KafkaProducer
from common.preprocessor import load_and_preprocess

def wait_for_kafka(host='kafka', port=9092, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=5):
                print(f"✅ Kafka is up at {host}:{port}")
                return True
        except Exception as e:
            print(f"⏳ Waiting for Kafka... {e}")
            time.sleep(2)
    raise TimeoutError("Kafka did not become available in time.")

wait_for_kafka()

producer = KafkaProducer(
    bootstrap_servers='kafka:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

print("✅ Kafka producer started")

# Load data
csv_path = "/data/PS_20174392719_1491204439457_log.csv"
df = load_and_preprocess(csv_path)

# Define thresholds
q1 = df['amount'].quantile(0.33)
q2 = df['amount'].quantile(0.66)

for _, row in df.iterrows():
    amount = row['amount']
    record = row.to_dict()

    if amount <= q1:
        topic = 'amount_low'
    elif amount <= q2:
        topic = 'amount_medium'
    else:
        topic = 'amount_high'

    producer.send(topic, value=record)
    time.sleep(0.01)  # simulate real-time

print("✅ Finished streaming data to Kafka")