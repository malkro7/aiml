# /app/common/preprocess.py
import os, json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

_PREPROC = None
_META = None

PREPROCESSOR_PATH = Path(os.getenv("PREPROCESSOR_PATH", "/models/baseline/preprocessor.joblib"))
META_PATH         = Path(os.getenv("META_PATH", "/models/baseline/meta.json"))

DROP_COLS = ["isFlaggedFraud", "step", "nameOrig", "nameDest"]
TARGET    = "isFraud"

def _load_meta():
    global _META
    if _META is None and META_PATH.exists():
        _META = json.load(open(META_PATH))
    return _META or {}

def load_preprocessor():
    global _PREPROC
    if _PREPROC is None:
        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(f"Preprocessor not found at {PREPROCESSOR_PATH}")
        _PREPROC = joblib.load(PREPROCESSOR_PATH)
    return _PREPROC

def expected_columns():
    """Return the raw column names the preprocessor expects, from meta.json."""
    meta = _load_meta()
    cats = meta.get("categorical_cols", [])
    nums = meta.get("numeric_cols", [])
    # Order doesn’t have to be exact as ColumnTransformer selects by name,
    # but we’ll assemble DataFrame with these present.
    return cats + nums

def df_from_records(records):
    """
    records: list[dict] or dict -> pandas.DataFrame
    Drops training-time DROP_COLS, preserves TARGET if present.
    """
    if isinstance(records, dict):
        df = pd.DataFrame([records])
    else:
        df = pd.DataFrame(records)

    # Drop columns we never used for training
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    return df

# --- PATCH START: replace transform_for_training() with this ---
def transform_for_training(df: pd.DataFrame):
    """
    Returns (X, y) where:
      - X is np.ndarray[float32] after ColumnTransformer (OHE + scale)
      - y is None if TARGET not present
    """
    # target
    y = df[TARGET].astype(int).values if TARGET in df.columns else None

    meta = _load_meta()
    cats = meta.get("categorical_cols", [])
    nums = meta.get("numeric_cols", [])

    # 1) Ensure all expected columns exist with safe defaults
    for c in cats:
        if c not in df.columns:
            df[c] = ""                  # categorical default as empty string
        else:
            # make sure dtype is string-like/object
            df[c] = df[c].astype("string").fillna("")

    for c in nums:
        if c not in df.columns:
            df[c] = np.nan              # numeric default as NaN
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")  # coerce bad values to NaN

    # 2) Order columns exactly as the preprocessor expects
    cols = cats + nums
    df_ordered = df[[c for c in cols if c in df.columns]].copy()

    # 3) Transform
    preproc = load_preprocessor()
    X = preproc.transform(df_ordered).astype("float32")
    return X, y
# --- PATCH END ---



#######Older Version##############
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# import joblib

# def load_and_preprocess(csv_path: str):
# # === Load dataset ===
#     df = pd.read_csv(csv_path)

# # === Drop unwanted columns ===
# # Drop IDs and step (no modeling significance)
#     df = df.drop(['nameOrig', 'nameDest', 'step'], axis=1)

# # === Label encode 'type' ===
#     le = LabelEncoder()
#     df['type'] = le.fit_transform(df['type'])

# # === Separate features and target ===
#     y = df['isFraud']                          # target label
#     X = df.drop(['isFraud', 'isFlaggedFraud'], axis=1)  # features only

# # === Normalize features ===
#     scaler = MinMaxScaler()
#     X[X.columns] = scaler.fit_transform(X)

# # === Combine features and target back for streaming ===
#     processed_df = pd.concat([X, y], axis=1)

# # (Optional) Save encoders for clients to use same transform later
#     joblib.dump(le, 'label_encoder.pkl')
#     joblib.dump(scaler, 'minmax_scaler.pkl')

#     print("✅ Preprocessing done! Ready for streaming.")
#     print(processed_df.head())

#     return processed_df
