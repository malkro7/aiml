# client1/client.py

import os
import json
import socket
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from kafka import KafkaConsumer
import flwr as fl
from flwr.client import start_client

# Your model factory (keep as in your project)
from common.client_logic import get_model

# Centralized preprocessing helpers (preprocessor.joblib + meta.json)
from common.preprocess import df_from_records, transform_for_training, expected_columns

CLIENT_ID = os.getenv("CLIENT_ID", "1")

# Map client to topic (adjust if your topics differ)
TOPIC_MAP = {
    "1": "amount_low",
    "2": "amount_medium",
    "3": "amount_high",
}

# ----- Utilities -----
def wait_for_kafka(host="kafka", port=9092, timeout=60):
    """Block until Kafka is reachable to avoid connection races."""
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

def kafka_batch_to_xy(topic: str, batch_size: int = 5000):
    """Consume a fixed batch from Kafka and return (X, y, df_raw) using the centralized preprocessor."""
    print(f"🔄 Consuming messages from Kafka topic '{topic}'...")
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers="kafka:9092",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id=f"client-{CLIENT_ID}",
    )

    records = []
    for msg in consumer:
        records.append(msg.value)
        if len(records) >= batch_size:
            break
    consumer.close()

    print(f"✅ Received {len(records)} records from '{topic}'")

    # Build DataFrame with the same drops used during centralized training
    df_raw = df_from_records(records)

    # Use the exact same ColumnTransformer (OHE 'type' + scale numerics)
    X, y = transform_for_training(df_raw)
    return X, y, df_raw

def perform_eda(df: pd.DataFrame, client_id: str):
    """Basic EDA snapshots; safe with mixed dtypes."""
    print(f"[Client {client_id}] 🧪 Performing EDA on in-memory data")

    # Summary (prints to logs)
    print("📋 Data summary (head):")
    print(df.head(5))
    print("📊 Class distribution:")
    if "isFraud" in df.columns:
        print(df["isFraud"].value_counts())

    # Ensure output dir exists; save under /data so you can see it on host if /data is bind-mounted
    out_dir = "/data"
    class_plot = os.path.join(out_dir, f"eda_client_{client_id}_class_dist.png")
    hist_plot = os.path.join(out_dir, f"eda_client_{client_id}_features_hist.png")

    # Class distribution
    if "isFraud" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x="isFraud", data=df)
        plt.title(f"Client {client_id} - Class Distribution")
        plt.tight_layout()
        plt.savefig(class_plot)
        plt.close()
        print(f"[Client {client_id}] 📂 Saved class distribution → {class_plot}")

    # Numeric histograms (avoid non-numeric columns like 'type')
    num_df = df.select_dtypes(include=["number"]).drop(columns=[c for c in ["isFraud"] if c in df.columns], errors="ignore")
    if not num_df.empty:
        num_df.hist(figsize=(12, 8), bins=50)
        plt.tight_layout()
        plt.savefig(hist_plot)
        plt.close()
        print(f"[Client {client_id}] 📂 Saved numeric histograms → {hist_plot}")

# ----- Flower client -----
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.topic = TOPIC_MAP.get(client_id, "amount_low")

        # MLflow
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment(f"client_{self.client_id}_experiment")

        # Ensure Kafka is up (best-effort)
        try:
            wait_for_kafka(host="kafka", port=9092, timeout=60)
        except Exception as e:
            print(f"⚠ Kafka wait error (continuing): {e}")

        # ---- Load one batch from Kafka and preprocess with centralized preprocessor ----
        self.x_train, self.y_train, self.df_raw = kafka_batch_to_xy(self.topic, batch_size=5000)

        # EDA on the raw (pre-transform) DataFrame so plots are interpretable
        perform_eda(self.df_raw, self.client_id)

        # For demo, reuse same data for test (you can change this)
        self.x_test, self.y_test = self.x_train, self.y_train

        # Build local model with the correct input shape (matches centralized meta["input_dim"])
        self.model = get_model(input_shape=(self.x_train.shape[1],))

    # Flower hooks
    def get_parameters(self, config):
        print(f"[Client {self.client_id}] Sending initial parameters to server")
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32, verbose=0)
        loss, acc = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        print(f"[Client {self.client_id}] Training done - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        with mlflow.start_run(run_name=f"client_{self.client_id}_run", nested=True):
            mlflow.log_metric("train_loss", float(loss))
            mlflow.log_metric("train_accuracy", float(acc))
        return self.model.get_weights(), len(self.x_train), {"accuracy": float(acc), "loss": float(loss)}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"[Client {self.client_id}] Evaluation - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        return float(loss), len(self.x_test), {"accuracy": float(acc), "loss": float(loss)}

if __name__ == "__main__":
    client = FlowerClient(CLIENT_ID).to_client()
    # NOTE: match the server address/port to your server container (compose)
    start_client(server_address="server:8085", client=client)
