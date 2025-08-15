# common/eda/eda_fraud_report.py
import argparse, os, textwrap, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def safe_top_counts(series, topn=20):
    vc = series.value_counts(dropna=False).head(topn)
    total = len(series)
    df = vc.rename_axis(series.name).reset_index(name="count")
    df["pct"] = (df["count"] / total * 100).round(2)
    return df

def add_table_page(pdf, title, df, note=None, max_rows=25):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 landscape
    ax.axis("off")
    ax.set_title(title, fontsize=16, weight="bold", loc="left")
    show_df = df.copy()
    if len(show_df) > max_rows:
        show_df = show_df.head(max_rows)
        note = (note or "") + f"\n(Showing first {max_rows} rows)"
    table = ax.table(cellText=show_df.values,
                     colLabels=show_df.columns,
                     cellLoc="left", loc="upper left")
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.2)
    if note: ax.text(0.01, 0.02, textwrap.fill(note, 150), fontsize=9, va="bottom")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

def add_hist(pdf, df, col, bins=50):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[col].dropna(), bins=bins)
    ax.set_title(f"Distribution of {col}"); ax.set_xlabel(col); ax.set_ylabel("Count")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

def add_hist_by_class(pdf, df, col, target="isFraud", bins=60, sample_cap=200_000):
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = df[[col, target]].dropna()
    if len(plot_df) > sample_cap:
        plot_df = plot_df.sample(sample_cap, random_state=42)
    for c in sorted(plot_df[target].unique()):
        ax.hist(plot_df.loc[plot_df[target] == c, col], bins=bins,
                histtype="step", linewidth=1.5, label=f"{target}={c}", alpha=0.9)
    ax.set_title(f"{col} — overlay by {target}"); ax.set_xlabel(col); ax.set_ylabel("Count")
    ax.legend(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

def add_box_by_class(pdf, df, col, target="isFraud", sample_cap=100_000):
    plot_df = df[[col, target]].dropna()
    if len(plot_df) > sample_cap:
        plot_df = plot_df.sample(sample_cap, random_state=42)
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_df[target] = plot_df[target].astype(str)
    plot_df.boxplot(column=col, by=target, ax=ax)
    ax.set_title(f"{col} by {target}"); ax.set_xlabel(target); ax.set_ylabel(col)
    plt.suptitle("")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

def add_corr_heatmap(pdf, corr):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns))); ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("Correlation Matrix (Pearson)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

def try_html_profile(df, output_html):
    try:
        from ydata_profiling import ProfileReport
        profile = ProfileReport(df, title="Fraud Dataset Profile", explorative=True,
                                correlations={"pearson": {"calculate": True}},
                                samples={"head": 0, "tail": 0, "random": 0})
        profile.to_file(output_html)
        return True, None
    except Exception as e:
        return False, str(e)

def build_pdf(df, pdf_path, source_name, dropped_note, sample_used=None):
    with PdfPages(pdf_path) as pdf:
        # Cover
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        ax.text(0.02, 0.85, "Exploratory Data Analysis Report", fontsize=20, weight="bold")
        ax.text(0.02, 0.78, f"File: {source_name}", fontsize=12)
        ax.text(0.02, 0.74, f"Rows: {len(df):,} | Columns: {df.shape[1]}", fontsize=12)
        ax.text(0.02, 0.70, dropped_note, fontsize=11)
        ax.text(0.02, 0.66, "Target: isFraud", fontsize=11)
        if sample_used:
            ax.text(0.02, 0.62, f"Sampled: {sample_used:,} rows (for speed)", fontsize=11)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Dtypes & nulls
        dtypes_df = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str)})
        add_table_page(pdf, "Data Types", dtypes_df)

        nulls_df = df.isna().sum().reset_index()
        nulls_df.columns = ["column", "null_count"]
        nulls_df["null_pct"] = (nulls_df["null_count"] / len(df) * 100).round(3)
        add_table_page(pdf, "Null Value Summary", nulls_df.sort_values("null_count", ascending=False))

        # Class imbalance
        class_counts = df["isFraud"].value_counts().rename_axis("isFraud").reset_index(name="count")
        class_counts["pct"] = (class_counts["count"]/class_counts["count"].sum()*100).round(3)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(class_counts["isFraud"].astype(str), class_counts["count"])
        ax.set_title("Class Distribution — isFraud"); ax.set_xlabel("isFraud"); ax.set_ylabel("Count")
        for i, row in class_counts.iterrows():
            ax.text(i, row["count"], f'{row["pct"]:.2f}%', ha="center", va="bottom", fontsize=9)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Numeric & categorical splits
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "isFraud" in numeric_cols: numeric_cols.remove("isFraud")
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Distributions for numeric columns
        for col in numeric_cols:
            add_hist(pdf, df, col)
            add_hist_by_class(pdf, df, col, target="isFraud")
            add_box_by_class(pdf, df, col, target="isFraud")

        # Categorical summaries (top-k only)
        for col in categorical_cols:
            top_df = safe_top_counts(df[col], topn=20)
            add_table_page(pdf, f"Top values — {col}", top_df,
                           note="High-cardinality columns summarized by top 20 values.")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(top_df[col].astype(str)[::-1], top_df["count"][::-1])
            ax.set_title(f"Top 20 values — {col}"); ax.set_xlabel("Count")
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Correlation matrix (numeric only)
        if numeric_cols:
            corr = df[numeric_cols + ["isFraud"]].corr()
            add_corr_heatmap(pdf, corr)
            tgt_corr = corr["isFraud"].drop("isFraud").sort_values(ascending=False).reset_index()
            tgt_corr.columns = ["feature", "corr_with_isFraud"]
            add_table_page(pdf, "Correlation with isFraud (numeric features)", tgt_corr)

        # Notes
        fig, ax = plt.subplots(figsize=(11.69, 8.27)); ax.axis("off")
        ax.text(0.02, 0.9, "Notes & Next Steps", fontsize=16, weight="bold")
        notes = """
        • High-cardinality IDs (nameOrig, nameDest) summarized by top counts only.
        • Consider log-scaling for skewed monetary features (amount/balances).
        • For mixed types, consider point-biserial correlation or mutual information.
        • Next: feature engineering (balance deltas/ratios), leakage checks, stratified splits.
        """
        ax.text(0.02, 0.84, textwrap.dedent(notes), fontsize=11)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

def run(csv_path, out_prefix, sample=None):
    df = pd.read_csv(csv_path)
    for col in ["step", "isFlaggedFraud"]:
        if col in df.columns: df.drop(columns=[col], inplace=True)
    if "isFraud" not in df.columns:
        raise ValueError("Column 'isFraud' not found in dataset.")
    sample_used = None
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=42).reset_index(drop=True)
        sample_used = sample
    pdf_path = f"{out_prefix}_report.pdf"
    build_pdf(df, pdf_path, os.path.basename(csv_path),
              dropped_note="Dropped: step, isFlaggedFraud", sample_used=sample_used)
    # optional HTML profile
    ok, err = try_html_profile(df.copy(), f"{out_prefix}_profile.html")
    if ok:
        print(f"HTML profile saved to: {out_prefix}_profile.html")
    else:
        print("HTML profile not generated (optional):", err)
    print(f"PDF report saved to: {pdf_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--out", default="/data/eda/report")
    ap.add_argument("--sample", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    run(args.csv_path, args.out, sample=args.sample)

if __name__ == "__main__":
    main()
