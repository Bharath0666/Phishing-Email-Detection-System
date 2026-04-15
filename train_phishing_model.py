# PhishGuard AI — Model Training Notebook
# ========================================
# Train a phishing email detection model using 7 real-world email datasets.
# Upload the archive/ folder to Colab or mount Google Drive.
#
# To use in Google Colab:
#   1. Upload the 7 CSV files from the archive/ folder
#   2. Run all cells
#   3. Download the exported model artifacts (zip file)

"""
## Setup and Imports
"""

# !pip install scikit-learn pandas numpy joblib scipy  # Uncomment if running in Colab

import os
import re
import time
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from scipy.sparse import hstack, csr_matrix
from urllib.parse import urlparse


"""
## 1. Load All Datasets
"""

# Set this to the path where your CSV files are located
# In Colab: upload files or mount drive and set path
ARCHIVE_DIR = "./archive"  # Change this to your path

def load_phishing_email(archive_dir):
    path = os.path.join(archive_dir, "phishing_email.csv")
    if not os.path.exists(path):
        print(f"  ⚠ {path} not found, skipping")
        return pd.DataFrame(columns=["subject", "body", "label", "source"])
    df = pd.read_csv(path, usecols=["text_combined", "label"])
    df["subject"] = df["text_combined"].apply(lambda x: str(x).split("\n")[0][:200] if pd.notna(x) else "")
    df["body"] = df["text_combined"].apply(lambda x: "\n".join(str(x).split("\n")[1:]) if pd.notna(x) else "")
    df["source"] = "phishing_email"
    return df[["subject", "body", "label", "source"]]

def load_simple_csv(archive_dir, filename):
    path = os.path.join(archive_dir, filename)
    if not os.path.exists(path):
        print(f"  ⚠ {path} not found, skipping")
        return pd.DataFrame(columns=["subject", "body", "label", "source"])
    df = pd.read_csv(path, usecols=["subject", "body", "label"])
    df["source"] = filename.replace(".csv", "")
    return df[["subject", "body", "label", "source"]]

print("Loading all datasets...")
frames = [
    load_phishing_email(ARCHIVE_DIR),
    load_simple_csv(ARCHIVE_DIR, "Enron.csv"),
    load_simple_csv(ARCHIVE_DIR, "Ling.csv"),
    load_simple_csv(ARCHIVE_DIR, "CEAS_08.csv"),
    load_simple_csv(ARCHIVE_DIR, "Nazario.csv"),
    load_simple_csv(ARCHIVE_DIR, "Nigerian_Fraud.csv"),
    load_simple_csv(ARCHIVE_DIR, "SpamAssasin.csv"),
]

df = pd.concat(frames, ignore_index=True)
print(f"\n✓ Total raw samples: {len(df):,}")
print(f"\nSource distribution:")
print(df["source"].value_counts().to_string())
print(f"\nLabel distribution:")
print(df["label"].value_counts().to_string())


"""
## 2. Preprocess Data
"""

print("\nPreprocessing...")
df = df.dropna(subset=["body", "label"])
df["subject"] = df["subject"].fillna("")
df["subject"] = df["subject"].astype(str)
df["body"] = df["body"].astype(str)
df["body"] = df["body"].str[:2000]
df = df.drop_duplicates(subset=["subject", "body"], keep="first")
df = df[df["label"].isin([0, 1])]
df["label"] = df["label"].astype(int)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✓ Cleaned dataset: {len(df):,} samples")
print(f"  Phishing: {(df['label']==1).sum():,} ({(df['label']==1).mean():.1%})")
print(f"  Legitimate: {(df['label']==0).sum():,} ({(df['label']==0).mean():.1%})")
print(f"  Avg body length: {df['body'].str.len().mean():.0f} chars")


"""
## 3. Feature Engineering
"""

# ── Heuristic Feature Extraction ──

URGENCY_KEYWORDS = [
    "urgent", "immediately", "suspend", "verify", "expire", "limited",
    "action required", "act now", "within 24 hours", "within 48 hours",
    "right away", "as soon as possible", "time-sensitive", "critical",
    "deadline", "warning", "alert", "unauthorized", "compromised",
    "restricted", "locked", "disabled", "terminated", "penalty",
]

CREDENTIAL_PHRASES = [
    "verify your identity", "confirm your password", "update your credentials",
    "enter your password", "social security number", "credit card number",
    "bank account", "routing number", "login credentials", "security question",
    "date of birth", "personal information", "billing information",
    "payment details", "account number", "verify your account",
    "confirm your account", "reset your password", "update your information",
]

THREAT_PHRASES = [
    "account will be suspended", "account will be closed",
    "permanently banned", "legal action", "unauthorized access",
    "fraud detected", "security breach", "data breach",
    "failure to comply", "failure to act", "reported to authorities",
]

REWARD_PHRASES = [
    "congratulations", "you have won", "prize", "reward", "gift card",
    "lottery", "selected as winner", "claim your", "free",
    "exclusive offer", "limited time offer",
]

SUSPICIOUS_TLDS = [".xyz", ".top", ".club", ".work", ".click", ".link",
                   ".info", ".online", ".site", ".tech", ".space", ".fun",
                   ".icu", ".buzz", ".gq", ".ml", ".cf", ".ga", ".tk"]

URL_SHORTENERS = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
                  "is.gd", "v.gd", "buff.ly", "rebrand.ly", "cutt.ly"]

HEURISTIC_FEATURE_NAMES = [
    "urgency_keyword_count", "urgency_density",
    "credential_phrase_count", "threat_phrase_count", "reward_phrase_count",
    "url_count", "ip_url_count", "shortener_url_count", "suspicious_tld_count",
    "sender_domain_digit_count", "sender_subdomain_count", "sender_suspicious_tld",
    "total_word_count", "exclamation_count", "caps_word_ratio",
    "has_html_tags", "subject_has_urgency", "subject_has_re_fw", "subject_all_caps",
    "special_char_ratio",
]


def count_keyword_matches(text, keywords):
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)

def keyword_density(text, keywords):
    words = text.split()
    if not words:
        return 0.0
    return count_keyword_matches(text, keywords) / len(words)

def extract_urls(text):
    return re.findall(r'https?://[^\s<>"\')\]]+|www\.[^\s<>"\')\]]+', text, re.IGNORECASE)

def extract_heuristic_features(subject, body, sender=""):
    full_text = f"{subject} {body}"
    features = {}
    features["urgency_keyword_count"] = count_keyword_matches(full_text, URGENCY_KEYWORDS)
    features["urgency_density"] = keyword_density(full_text, URGENCY_KEYWORDS)
    features["credential_phrase_count"] = count_keyword_matches(full_text, CREDENTIAL_PHRASES)
    features["threat_phrase_count"] = count_keyword_matches(full_text, THREAT_PHRASES)
    features["reward_phrase_count"] = count_keyword_matches(full_text, REWARD_PHRASES)

    urls = extract_urls(body)
    features["url_count"] = len(urls)
    ip_url_count = shortener_count = suspicious_tld_count = 0
    for url in urls:
        try:
            parsed = urlparse(url)
            host = parsed.hostname or ""
            if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', host):
                ip_url_count += 1
            if any(s in host for s in URL_SHORTENERS):
                shortener_count += 1
            if any(host.endswith(tld) for tld in SUSPICIOUS_TLDS):
                suspicious_tld_count += 1
        except:
            pass
    features["ip_url_count"] = ip_url_count
    features["shortener_url_count"] = shortener_count
    features["suspicious_tld_count"] = suspicious_tld_count

    sender_lower = sender.lower()
    sender_domain = sender_lower.split("@")[-1] if "@" in sender_lower else ""
    features["sender_domain_digit_count"] = len(re.findall(r'\d', sender_domain))
    features["sender_subdomain_count"] = sender_domain.count(".") if sender_domain else 0
    features["sender_suspicious_tld"] = 1.0 if any(sender_domain.endswith(tld) for tld in SUSPICIOUS_TLDS) else 0.0

    words = full_text.split()
    features["total_word_count"] = len(words)
    features["exclamation_count"] = full_text.count("!")
    features["caps_word_ratio"] = sum(1 for w in words if w.isupper() and len(w) > 1) / max(len(words), 1)
    features["has_html_tags"] = 1.0 if re.search(r'<[a-zA-Z][^>]*>', body) else 0.0
    features["subject_has_urgency"] = 1.0 if count_keyword_matches(subject, URGENCY_KEYWORDS) > 0 else 0.0
    features["subject_has_re_fw"] = 1.0 if re.match(r'^(re|fw|fwd):', subject, re.IGNORECASE) else 0.0
    features["subject_all_caps"] = 1.0 if subject == subject.upper() and len(subject) > 3 else 0.0
    special_chars = sum(1 for c in full_text if not c.isalnum() and not c.isspace())
    features["special_char_ratio"] = special_chars / max(len(full_text), 1)
    return features

def features_to_vector(features):
    return [features.get(name, 0.0) for name in HEURISTIC_FEATURE_NAMES]


print("Extracting heuristic features...")
t0 = time.time()
heuristic_vectors = []
for i, row in df.iterrows():
    feats = extract_heuristic_features(row["subject"], row["body"])
    heuristic_vectors.append(features_to_vector(feats))
    if (i + 1) % 20000 == 0:
        print(f"  Processed {i+1:,}/{len(df):,} samples...")

heuristic_matrix = np.array(heuristic_vectors)
print(f"✓ Heuristic features extracted in {time.time()-t0:.1f}s")
print(f"  Shape: {heuristic_matrix.shape}")


"""
## 4. TF-IDF Vectorization
"""

print("\nBuilding TF-IDF features...")
texts = [f"{s} {b}" for s, b in zip(df["subject"], df["body"])]
labels = df["label"].values

# Train/test split
texts_train, texts_test, y_train, y_test, h_train, h_test = train_test_split(
    texts, labels, heuristic_matrix,
    test_size=0.2, random_state=42, stratify=labels
)

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True,
    min_df=3,
    max_df=0.95,
)

t0 = time.time()
tfidf_train = tfidf.fit_transform(texts_train)
tfidf_test = tfidf.transform(texts_test)
print(f"✓ TF-IDF done in {time.time()-t0:.1f}s")
print(f"  Vocabulary size: {len(tfidf.vocabulary_):,}")
print(f"  Train shape: {tfidf_train.shape}")

# Combine features
X_train = hstack([tfidf_train, csr_matrix(h_train)])
X_test = hstack([tfidf_test, csr_matrix(h_test)])
print(f"  Combined features: {X_train.shape[1]:,}")


"""
## 5. Train Random Forest
"""

print("\nTraining Random Forest classifier...")
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

t0 = time.time()
clf.fit(X_train, y_train)
train_time = time.time() - t0
print(f"✓ Training complete in {train_time:.1f}s")


"""
## 6. Evaluate
"""

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 50)
print("  MODEL EVALUATION RESULTS")
print("=" * 50)
print(f"  Accuracy:   {accuracy:.4f} ({accuracy:.2%})")
print(f"  Precision:  {precision:.4f} ({precision:.2%})")
print(f"  Recall:     {recall:.4f} ({recall:.2%})")
print(f"  F1 Score:   {f1:.4f} ({f1:.2%})")
print(f"  ROC-AUC:    {roc_auc:.4f}")
print(f"\n  Confusion Matrix:")
print(f"  {'':>15} Predicted Legit  Predicted Phish")
print(f"  {'Actual Legit':>15}     {cm[0][0]:>5}           {cm[0][1]:>5}")
print(f"  {'Actual Phish':>15}     {cm[1][0]:>5}           {cm[1][1]:>5}")
print("=" * 50)

print("\n  Full Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))


"""
## 7. Feature Importances
"""

n_tfidf = tfidf_train.shape[1]
importances = clf.feature_importances_
tfidf_names = tfidf.get_feature_names_out()

# Top TF-IDF features
tfidf_importance = importances[:n_tfidf]
top_tfidf_indices = np.argsort(tfidf_importance)[-15:][::-1]
feature_importances = {}
for idx in top_tfidf_indices:
    feature_importances[f"word: {tfidf_names[idx]}"] = round(float(tfidf_importance[idx]), 6)

# Heuristic feature importances
heuristic_importance = importances[n_tfidf:]
for name, imp in zip(HEURISTIC_FEATURE_NAMES, heuristic_importance):
    feature_importances[name] = round(float(imp), 6)

# Sort
feature_importances = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))

print("\nTop 15 Feature Importances:")
for i, (name, imp) in enumerate(list(feature_importances.items())[:15]):
    bar = "█" * int(imp * 500)
    print(f"  {i+1:>2}. {name:<35} {imp:.4f} {bar}")


"""
## 8. Export Model Artifacts
"""

OUTPUT_DIR = "./models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

joblib.dump(clf, os.path.join(OUTPUT_DIR, "classifier.joblib"))
joblib.dump(tfidf, os.path.join(OUTPUT_DIR, "tfidf_vectorizer.joblib"))

metrics = {
    "accuracy": round(accuracy, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1_score": round(f1, 4),
    "roc_auc": round(roc_auc, 4),
    "confusion_matrix": cm.tolist(),
    "train_size": len(y_train),
    "test_size": len(y_test),
}

with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "feature_importances.json"), "w") as f:
    json.dump(feature_importances, f, indent=2)

print(f"\n✓ Model artifacts saved to {OUTPUT_DIR}/")
print(f"  - classifier.joblib")
print(f"  - tfidf_vectorizer.joblib")
print(f"  - metrics.json")
print(f"  - feature_importances.json")

# Optional: Create zip for download in Colab
# import shutil
# shutil.make_archive("phishguard_model", "zip", OUTPUT_DIR)
# print("✓ Created phishguard_model.zip for download")

print("\n🎉 Training complete! Copy the models/ folder to your project.")
