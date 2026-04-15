"""
Model Training Script
Run this to train and save the phishing detection model using real datasets.
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from ml.dataset import load_all_datasets, preprocess, get_dataset_info
from ml.model import PhishingDetector


def main():
    print("=" * 60)
    print("  Phishing Email Detection — Model Training")
    print("  Using Real Datasets from archive/")
    print("=" * 60)

    # 1. Load all datasets
    print("\n[1/5] Loading datasets from archive/...")
    t0 = time.time()
    df = load_all_datasets()
    print(f"  ✓ Raw dataset: {len(df):,} samples loaded in {time.time()-t0:.1f}s")
    print(f"  ✓ Sources:")
    for src, count in df["source"].value_counts().items():
        print(f"      {src}: {count:,}")

    # 2. Preprocess
    print("\n[2/5] Preprocessing...")
    df = preprocess(df)
    info = get_dataset_info(df)
    print(f"  ✓ Cleaned dataset: {info['total_samples']:,} samples")
    print(f"  ✓ Phishing: {info['phishing_count']:,} ({info['phishing_ratio']:.1%})")
    print(f"  ✓ Legitimate: {info['legitimate_count']:,} ({1-info['phishing_ratio']:.1%})")
    print(f"  ✓ Avg body length: {info['avg_body_length']} chars")

    # 3. Initialize model
    print("\n[3/5] Initializing model...")
    detector = PhishingDetector()

    # 4. Train
    print("\n[4/5] Training Random Forest classifier...")
    print("  This may take a few minutes on the full dataset...")
    t0 = time.time()
    metrics = detector.train(
        subjects=df["subject"].tolist(),
        bodies=df["body"].tolist(),
        senders=["unknown@unknown.com"] * len(df),  # not all datasets have sender
        labels=df["label"].tolist(),
        test_size=0.2,
    )
    elapsed = time.time() - t0

    print(f"\n  Training completed in {elapsed:.1f}s")
    print(f"\n  Results:")
    print(f"  ╔══════════════════════════════╗")
    print(f"  ║  Accuracy:   {metrics['accuracy']:.2%}       ║")
    print(f"  ║  Precision:  {metrics['precision']:.2%}       ║")
    print(f"  ║  Recall:     {metrics['recall']:.2%}       ║")
    print(f"  ║  F1 Score:   {metrics['f1_score']:.2%}       ║")
    print(f"  ╚══════════════════════════════╝")

    cm = metrics["confusion_matrix"]
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>15} Predicted Legit  Predicted Phish")
    print(f"  {'Actual Legit':>15}     {cm[0][0]:>5}           {cm[0][1]:>5}")
    print(f"  {'Actual Phish':>15}     {cm[1][0]:>5}           {cm[1][1]:>5}")

    # 5. Save model
    print("\n[5/5] Saving model...")
    detector.save()

    # Print top feature importances
    print("\n  Top 10 Feature Importances:")
    for i, (name, imp) in enumerate(list(detector.feature_importances.items())[:10]):
        bar = "█" * int(imp * 500)
        print(f"  {i+1:>2}. {name:<35} {imp:.4f} {bar}")

    print("\n" + "=" * 60)
    print("  Model training complete! Ready to serve predictions.")
    print("=" * 60)


if __name__ == "__main__":
    main()
