"""
Real Dataset Loader
Loads, merges, and preprocesses all 7 phishing email datasets from the archive.
"""

import os
import re
import pandas as pd
import numpy as np
from typing import Optional


ARCHIVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "archive")


# ─── Individual Loaders ─────────────────────────────────────────────────

def _load_phishing_email(archive_dir: str) -> pd.DataFrame:
    """Load phishing_email.csv — has 'text_combined' and 'label'."""
    path = os.path.join(archive_dir, "phishing_email.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["subject", "body", "label", "source"])

    df = pd.read_csv(path, usecols=["text_combined", "label"])
    # Split text_combined: first line → subject, rest → body
    df["subject"] = df["text_combined"].apply(
        lambda x: str(x).split("\n")[0][:200] if pd.notna(x) else ""
    )
    df["body"] = df["text_combined"].apply(
        lambda x: "\n".join(str(x).split("\n")[1:]) if pd.notna(x) else ""
    )
    df["source"] = "phishing_email"
    return df[["subject", "body", "label", "source"]]


def _load_subject_body_csv(archive_dir: str, filename: str) -> pd.DataFrame:
    """Load CSVs that have 'subject', 'body', 'label' columns (Enron, Ling)."""
    path = os.path.join(archive_dir, filename)
    if not os.path.exists(path):
        return pd.DataFrame(columns=["subject", "body", "label", "source"])

    df = pd.read_csv(path, usecols=["subject", "body", "label"])
    df["source"] = filename.replace(".csv", "")
    return df[["subject", "body", "label", "source"]]


def _load_full_csv(archive_dir: str, filename: str) -> pd.DataFrame:
    """Load CSVs with sender/receiver/subject/body/label/urls (CEAS_08, Nazario, etc.)."""
    path = os.path.join(archive_dir, filename)
    if not os.path.exists(path):
        return pd.DataFrame(columns=["subject", "body", "label", "source"])

    df = pd.read_csv(path, usecols=["subject", "body", "label"])
    df["source"] = filename.replace(".csv", "")
    return df[["subject", "body", "label", "source"]]


# ─── Main Loader ────────────────────────────────────────────────────────

def load_all_datasets(archive_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load and merge all 7 phishing email datasets into a unified DataFrame.

    Returns:
        DataFrame with columns: subject, body, label, source
        label: 0 = legitimate, 1 = phishing
    """
    archive_dir = archive_dir or ARCHIVE_DIR

    frames = []

    # 1. phishing_email.csv (82K rows)
    frames.append(_load_phishing_email(archive_dir))

    # 2. Enron.csv (30K rows) — subject, body, label
    frames.append(_load_subject_body_csv(archive_dir, "Enron.csv"))

    # 3. Ling.csv (3K rows) — subject, body, label
    frames.append(_load_subject_body_csv(archive_dir, "Ling.csv"))

    # 4. CEAS_08.csv (39K rows) — full format
    frames.append(_load_full_csv(archive_dir, "CEAS_08.csv"))

    # 5. Nazario.csv (1.5K rows) — full format
    frames.append(_load_full_csv(archive_dir, "Nazario.csv"))

    # 6. Nigerian_Fraud.csv (3.3K rows) — full format
    frames.append(_load_full_csv(archive_dir, "Nigerian_Fraud.csv"))

    # 7. SpamAssasin.csv (5.8K rows) — full format
    frames.append(_load_full_csv(archive_dir, "SpamAssasin.csv"))

    df = pd.concat(frames, ignore_index=True)
    return df


def preprocess(df: pd.DataFrame, max_body_len: int = 2000, max_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Clean and preprocess the merged dataset.

    Steps:
        1. Drop rows with missing body/label
        2. Fill missing subjects with empty string
        3. Truncate very long bodies
        4. Remove exact duplicates
        5. Ensure label is int (0/1)
        6. Optionally downsample to max_samples (balanced)
    """
    df = df.copy()

    # Drop rows where body or label is missing
    df = df.dropna(subset=["body", "label"])
    df["subject"] = df["subject"].fillna("")

    # Ensure string types
    df["subject"] = df["subject"].astype(str)
    df["body"] = df["body"].astype(str)

    # Truncate very long bodies
    df["body"] = df["body"].str[:max_body_len]

    # Remove exact text duplicates
    df = df.drop_duplicates(subset=["subject", "body"], keep="first")

    # Ensure label is int
    df["label"] = df["label"].astype(int)

    # Filter out any labels that aren't 0 or 1
    df = df[df["label"].isin([0, 1])]

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Optionally balance and downsample
    if max_samples is not None:
        n_per_class = max_samples // 2
        phishing = df[df["label"] == 1].head(n_per_class)
        legit = df[df["label"] == 0].head(n_per_class)
        df = pd.concat([phishing, legit], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def get_dataset_info(df: pd.DataFrame) -> dict:
    """Return summary statistics about the dataset."""
    source_counts = df["source"].value_counts().to_dict() if "source" in df.columns else {}
    return {
        "total_samples": len(df),
        "phishing_count": int((df["label"] == 1).sum()),
        "legitimate_count": int((df["label"] == 0).sum()),
        "phishing_ratio": round((df["label"] == 1).mean(), 4),
        "sources": source_counts,
        "avg_body_length": int(df["body"].str.len().mean()),
    }


if __name__ == "__main__":
    print("Loading all datasets...")
    df = load_all_datasets()
    print(f"Raw dataset: {len(df)} rows")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    print(f"Sources:\n{df['source'].value_counts()}")

    print("\nPreprocessing...")
    df = preprocess(df)
    print(f"Cleaned dataset: {len(df)} rows")
    info = get_dataset_info(df)
    for k, v in info.items():
        print(f"  {k}: {v}")
