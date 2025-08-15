# client1/client.py

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import mlflow
from kafka import KafkaConsumer
import flwr as fl
from flwr.client import start_client
from common.client_logic import get_model

CLIENT_ID = os.getenv("CLIENT_ID", "1")

# Map client to topic
TOPIC_MAP = {
    "1": "amount_low",
    "2": "amount_medium",
    "3": "amount_high"
}

# ✅ Wait for Kafka before consuming
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

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.topic = TOPIC_MAP.get(client_id, "amount_low")
# ✅ Connect MLflow
        mlflow.set_tracking_uri("http://mlflow:5000")  # internal service name
        mlflow.set_experiment(f"client_{self.client_id}_experiment")
 # ✅ Consume data
        self.df = self._consume_from_kafka()
        self.perform_eda()

        # Features (X) and target (y)
        self.x_train = self.df.drop("isFraud", axis=1).values
        self.y_train = self.df["isFraud"].values
        # For demo, use same as test
        self.x_test = self.x_train
        self.y_test = self.y_train

        # Build model with correct input shape
        self.model = get_model(input_shape=(self.x_train.shape[1],))

    def _consume_from_kafka(self):
        print(f"[Client {self.client_id}] 🔄 Consuming messages from Kafka topic '{self.topic}'...")
        consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers="kafka:9092",
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id=f'client-{self.client_id}'
        )

        records = []
        # Consume a certain number of messages (or you can use a timer)
        for msg in consumer:
            records.append(msg.value)
            # 👇 Stop condition (e.g., after 5000 records)
            if len(records) >= 5000:
                break

        consumer.close()
        print(f"[Client {self.client_id}] ✅ Received {len(records)} records.")
        return pd.DataFrame(records)

    def perform_eda(self):
        print(f"[Client {self.client_id}] 🧪 Performing EDA on in-memory data")

        print("📋 Data summary:")
        print(self.df.describe())

        print("📊 Class distribution:")
        print(self.df["isFraud"].value_counts())
    #save graphs
        out_path1 = f"/data/eda_client_{self.client_id}_class_dist.png"
        print(f"[Client {self.client_id}] >>> Saving class dist plot to {out_path1}")
        # Plot class distribution
        plt.figure(figsize=(6, 4))
        sns.countplot(x="isFraud", data=self.df)
        plt.title(f"Client {self.client_id} - Class Distribution")
        plt.savefig(out_path1)
    #save graphs
        out_path2 = f"/data/eda_client_{self.client_id}_features_hist.png"
        # Plot histograms of features
        self.df.drop("isFraud", axis=1).hist(figsize=(12, 8))
        plt.tight_layout()
        plt.savefig("out_path2")
        print(f"[Client {self.client_id}] 📂 EDA plots saved.")

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
        return self.model.get_weights(), len(self.x_train), {"accuracy": acc, "loss": loss}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"[Client {self.client_id}] Evaluation - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        return loss, len(self.x_test), {"accuracy": acc, "loss": loss}


if __name__ == "__main__":
    client = FlowerClient(CLIENT_ID).to_client()
    start_client(server_address="server:8085", client=client)