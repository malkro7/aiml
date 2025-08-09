# common/client_logic.py

import json
import os
import pandas as pd
from kafka import KafkaConsumer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

TOPIC_MAP = {
    "1": "amount_low",
    "2": "amount_medium",
    "3": "amount_high"
}

def load_data(client_id: str):
    topic = TOPIC_MAP.get(client_id, "amount_low")

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers="kafka:9092",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset='earliest',
        group_id=f"client_{client_id}"
    )

    X, y = [], []

    for message in consumer:
        data = message.value
        y.append(data["Class"])
        X.append([data[f"V{i}"] for i in range(1, 29)] + [data["Amount"]])
        if len(X) >= 512:
            break

    df_X = pd.DataFrame(X)
    df_y = pd.Series(y)

    return train_test_split(df_X, df_y, test_size=0.2, random_state=42)

def get_model(input_shape=(29,)):
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
