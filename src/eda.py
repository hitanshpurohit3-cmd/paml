# src/eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import os

sns.set_context("talk")
sns.set_style("white")


# -----------------------------
# CONFIG
# -----------------------------
RANDOM_STATE = 42
OUTPUT_DIR = "output/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(path):
    df = pd.read_csv(path)

    df = df.rename(columns={"status": "label"})
    df["label"] = df["label"].astype(int)

    return df


# -----------------------------
# BASIC OVERVIEW
# -----------------------------
def dataset_summary(df):
    print("\nShape:", df.shape)
    print("\nClass Distribution:\n", df["label"].value_counts(normalize=True))

    print("\nMissing Values:\n", df.isnull().sum())


# -----------------------------
# 1. ADVANCED CORRELATION HEATMAP
# -----------------------------
def correlation_heatmap(df):
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(14, 10))

    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8}
    )

    plt.title("Correlation Structure of Parkinson Features", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/full_correlation.png")
    #plt.show()
    plt.close()


# -----------------------------
# 2. TARGET-CORRELATION (SORTED)
# -----------------------------
def target_correlation(df):
    corr = df.corr(numeric_only=True)["label"].drop("label")

    corr = corr.sort_values(ascending=False)

    plt.figure(figsize=(6, 10))

    sns.heatmap(
        corr.to_frame(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title("Feature Influence on Parkinson's Target", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/target_correlation.png")
    # plt.show()
    plt.close()

    return corr


# -----------------------------
# 3. STATISTICAL SIGNIFICANCE
# -----------------------------
def statistical_tests(df):
    features = df.select_dtypes(include=np.number).columns.drop("label")

    results = []

    for col in features:
        group1 = df[df["label"] == 0][col]
        group2 = df[df["label"] == 1][col]

        stat, p = ttest_ind(group1, group2) # type: ignore

        results.append((col, p))

    results = pd.DataFrame(results, columns=["feature", "p_value"])
    results = results.sort_values("p_value")

    print("\nTop Statistically Significant Features:")
    print(results.head(10))

    return results


# -----------------------------
# 4. HIGH CORRELATION FILTER
# -----------------------------
def high_corr_heatmap(df, threshold=0.75):
    corr = df.corr(numeric_only=True).abs()

    mask = corr > threshold

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr * mask, cmap="Reds")

    plt.title(f"High Correlation Features (>{threshold})")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/high_correlation.png")
    # plt.show()
    plt.close()


# -----------------------------
# 5. FEATURE INTENSITY STRUCTURE
# -----------------------------
def feature_intensity(df):
    numeric_df = df.select_dtypes(include=np.number)

    norm = (numeric_df - numeric_df.mean()) / numeric_df.std()

    plt.figure(figsize=(14, 8))
    sns.heatmap(norm.iloc[:60], cmap="viridis")

    plt.title("Normalized Feature Intensity (Sampled Rows)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_intensity.png")
    # plt.show()
    plt.close()


# -----------------------------
# 6. CLASS-WISE DISTRIBUTION (RESEARCH GRADE)
# -----------------------------
def class_distribution(df, top_features):
    for feature in top_features[:5]:
        plt.figure()

        sns.kdeplot(
            data=df,
            x=feature,
            hue="label",
            fill=True
        )

        plt.title(f"{feature} Distribution by Class")
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{OUTPUT_DIR}/classWise_distribution.png")
        plt.close()

#------------------------------
#7. MISSING VALUES
#------------------------------
def plo_missing(df):
    plt.figure(figsize=(10,6))
    sns.heatmap(df.isnull(),cbar=False)
    plt.title("Missing Values Heatmap")
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/missing_value.png")
    plt.close()

# -----------------------------
# MASTER PIPELINE
# -----------------------------
def run_eda(path):
    df = load_data(path)

    dataset_summary(df)

    correlation_heatmap(df)

    corr = target_correlation(df)

    stats = statistical_tests(df)

    high_corr_heatmap(df)

    feature_intensity(df)

    # pick top features based on correlation
    top_features = corr.abs().sort_values(ascending=False).index[:10]

    class_distribution(df, top_features)

    return df, corr, stats
