import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

def load_and_preprocess(csv_path: str):
# === Load dataset ===
    df = pd.read_csv(csv_path)

# === Drop unwanted columns ===
# Drop IDs and step (no modeling significance)
    df = df.drop(['nameOrig', 'nameDest', 'step'], axis=1)

# === Label encode 'type' ===
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])

# === Separate features and target ===
    y = df['isFraud']                          # target label
    X = df.drop(['isFraud', 'isFlaggedFraud'], axis=1)  # features only

# === Normalize features ===
    scaler = MinMaxScaler()
    X[X.columns] = scaler.fit_transform(X)

# === Combine features and target back for streaming ===
    processed_df = pd.concat([X, y], axis=1)

# (Optional) Save encoders for clients to use same transform later
    joblib.dump(le, 'label_encoder.pkl')
    joblib.dump(scaler, 'minmax_scaler.pkl')

    print("✅ Preprocessing done! Ready for streaming.")
    print(processed_df.head())

    return processed_df
