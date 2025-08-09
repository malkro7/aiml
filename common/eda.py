import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("/data/PS_20174392719_1491204439457_log.csv")

# Quick info
print(df.info())
print(df.describe())
print(df['type'].value_counts())  # or whatever target/feature you care about

# Check class imbalance (assuming 'isFraud' is target)
if 'isFraud' in df.columns:
    print(df['isFraud'].value_counts(normalize=True))

# Plot distributions
df.hist(figsize=(15,10))
plt.tight_layout()
plt.savefig("/data/eda_histograms.png")  # Save plots instead of showing in container
plt.close('all')

# Correlation heatmap
num_cols = df.select_dtypes(include=['number'])
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("/data/eda_correlation.png")
plt.close('all')

# Boxplot for skew
for col in df.select_dtypes(include='number').columns:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot for {col}")
    plt.savefig(f"/data/eda_boxplot_{col}.png")
    plt.close('all')
