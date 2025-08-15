# /app/centralized_model/centralized_train.py
# (mounted from ./centralized_model/centralized_train.py)

import sys, os
sys.path.insert(0, os.path.abspath("/app"))  # ensure /app is importable

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    roc_curve, auc as sk_auc,
    precision_recall_curve, average_precision_score
)

# Import your model factory
from common.model import get_model  # or from common.model import build_model if you added wrapper

# -----------------------
# Config
# -----------------------
DATA_CSV = os.getenv("CENTRAL_DATA", "/data/PS_20174392719_1491204439457_log.csv")
OUT_DIR  = Path(os.getenv("MODEL_OUT", "/models/baseline"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET_COL   = "isFraud"
DROP_COLS    = ["isFlaggedFraud", "step", "nameOrig", "nameDest"]
USE_FRACTION = float(os.getenv("USE_FRACTION", "0.2"))  # tweak as needed
EPOCHS       = int(os.getenv("EPOCHS", "5"))
BATCH_SIZE   = int(os.getenv("BATCH_SIZE", "2048"))

# -----------------------
# Load & prepare data
# -----------------------
df = pd.read_csv(DATA_CSV)
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
if USE_FRACTION < 1.0:
    df = df.sample(frac=USE_FRACTION, random_state=RANDOM_STATE)

y = df[TARGET_COL].astype(int).values
X_df = df.drop(columns=[TARGET_COL])

categorical_cols = [c for c in X_df.columns if c == "type"]  # PaySim categorical
numeric_cols = [c for c in X_df.columns if c not in categorical_cols]

# OHE compatible across sklearn versions
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", ohe, categorical_cols),
        ("num", Pipeline([("scaler", StandardScaler())]), numeric_cols),
    ],
    remainder="drop",
)

X_train_df, X_tmp_df, y_train, y_tmp = train_test_split(
    X_df, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
)
X_val_df, X_test_df, y_val, y_test = train_test_split(
    X_tmp_df, y_tmp, test_size=0.5, stratify=y_tmp, random_state=RANDOM_STATE
)

X_train = preprocessor.fit_transform(X_train_df).astype("float32")
X_val   = preprocessor.transform(X_val_df).astype("float32")
X_test  = preprocessor.transform(X_test_df).astype("float32")

# Persist preprocessor for FL clients
import joblib
joblib.dump(preprocessor, OUT_DIR / "preprocessor.joblib")

# Feature names (optional)
feature_names = []
try:
    fn = []
    if categorical_cols:
        cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols).tolist()
        fn.extend(cat_names)
    fn.extend(numeric_cols)
    feature_names = fn
except Exception:
    pass

# Class weights (highly imbalanced)
classes = np.unique(y_train)
cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weights = {int(c): float(w) for c, w in zip(classes, cw)}

# -----------------------
# Build / train model
# -----------------------
input_dim = X_train.shape[1]
model = get_model(input_shape=(input_dim,))
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=2, restore_best_weights=True, monitor="val_auc", mode="max"
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=2,
)

# -----------------------
# Save history + plots
# -----------------------
with open(OUT_DIR / "history.json", "w") as f:
    json.dump(history.history, f, indent=2)

def plot_series(hist: dict, keys, title, filename):
    plt.figure()
    for k in keys:
        if k in hist:
            plt.plot(hist[k], label=k)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename)
    plt.close()

hist = history.history
plot_series(hist, ["loss", "val_loss"], "Loss", "loss.png")
plot_series(hist, ["accuracy", "val_accuracy"], "Accuracy", "accuracy.png")
plot_series(hist, ["auc", "val_auc"], "AUC", "auc.png")
plot_series(hist, ["precision", "val_precision"], "Precision", "precision.png")
plot_series(hist, ["recall", "val_recall"], "Recall", "recall.png")

# ROC & PR curves on test set
y_proba = model.predict(X_test, verbose=0).ravel()
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = sk_auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "roc_curve.png")
plt.close()

prec, rec, _ = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)
plt.figure()
plt.plot(rec, prec, label=f"AP={ap:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "pr_curve.png")
plt.close()

# -----------------------
# Evaluate & save weights/meta
# -----------------------
eval_dict = dict(zip(model.metrics_names, model.evaluate(X_test, y_test, verbose=0)))
with open(OUT_DIR / "eval.json", "w") as f:
    json.dump(eval_dict, f, indent=2)

# Keras >=3 requires .weights.h5 suffix
model.save_weights(OUT_DIR / "model.weights.h5")

with open(OUT_DIR / "meta.json", "w") as f:
    json.dump(
        {
            "input_dim": int(input_dim),
            "categorical_cols": categorical_cols,
            "numeric_cols": numeric_cols,
            "feature_names": feature_names,
        },
        f,
        indent=2,
    )

print("Baseline + plots saved to", OUT_DIR)
