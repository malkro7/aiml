import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_CSV = os.getenv("CENTRAL_DATA", "/data/PS_20174392719_1491204439457_log.csv")
OUT_DIR = Path(os.getenv("EDA_OUT", "/models/baseline/eda"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

DROP_COLS = ["isFlaggedFraud", "step", "nameOrig", "nameDest"]  # same as training
TARGET = "isFraud"

df = pd.read_csv(DATA_CSV)
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

# 1) Class balance
ax = df[TARGET].value_counts().sort_index().plot(kind="bar")
ax.set_title("Class Balance (isFraud)"); ax.set_xlabel("isFraud"); ax.set_ylabel("count")
plt.tight_layout(); plt.savefig(OUT_DIR / "01_class_balance.png"); plt.close()

# 2) Transaction amount distribution (log scale)
ax = np.log1p(df["amount"]).plot(kind="hist", bins=100)
ax.set_title("Amount Distribution (log1p)"); ax.set_xlabel("log1p(amount)")
plt.tight_layout(); plt.savefig(OUT_DIR / "02_amount_hist_log1p.png"); plt.close()

# 3) Categorical 'type' counts
if "type" in df.columns:
    ax = df["type"].value_counts().plot(kind="bar")
    ax.set_title("Type Counts"); ax.set_xlabel("type"); ax.set_ylabel("count")
    plt.tight_layout(); plt.savefig(OUT_DIR / "03_type_counts.png"); plt.close()

# 4) Numeric correlation heatmap (quick)
num_cols = df.select_dtypes(include=[np.number]).columns
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(8,6))
cax = ax.imshow(corr, interpolation="nearest")
ax.set_title("Numeric Feature Correlation")
ax.set_xticks(range(len(num_cols))); ax.set_yticks(range(len(num_cols)))
ax.set_xticklabels(num_cols, rotation=90, fontsize=7); ax.set_yticklabels(num_cols, fontsize=7)
fig.colorbar(cax)
plt.tight_layout(); plt.savefig(OUT_DIR / "04_corr_heatmap.png"); plt.close()

print(f"EDA images saved to {OUT_DIR}")
