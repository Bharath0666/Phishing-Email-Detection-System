"""
ML Model Module
Trains and serves a Random Forest classifier for phishing email detection.
Combines TF-IDF text features with cybersecurity heuristic features.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from scipy.sparse import hstack, csr_matrix

from .feature_extract import (
    extract_heuristic_features, features_to_vector, HEURISTIC_FEATURE_NAMES
)


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class PhishingDetector:
    """
    Phishing email detection model combining TF-IDF text vectorization
    with cybersecurity heuristic features in a Random Forest classifier.
    """

    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            min_df=3,
            max_df=0.95,
        )
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.is_trained = False
        self.metrics: Dict = {}
        self.feature_importances: Dict[str, float] = {}

    def _combine_text(self, subjects: List[str], bodies: List[str]) -> List[str]:
        """Combine subject and body into single text for TF-IDF."""
        return [f"{s} {b}" for s, b in zip(subjects, bodies)]

    def _get_heuristic_matrix(
        self, subjects: List[str], bodies: List[str], senders: List[str]
    ) -> np.ndarray:
        """Extract heuristic features for all samples."""
        feature_vectors = []
        for subj, body, sender in zip(subjects, bodies, senders):
            feats = extract_heuristic_features(subj, body, sender)
            feature_vectors.append(features_to_vector(feats))
        return np.array(feature_vectors)

    def train(
        self,
        subjects: List[str],
        bodies: List[str],
        senders: List[str],
        labels: List[int],
        test_size: float = 0.2,
    ) -> Dict:
        """
        Train the phishing detection model.

        Returns metrics dict with accuracy, precision, recall, f1, and confusion matrix.
        """
        # Combine text
        texts = self._combine_text(subjects, bodies)

        # Split data
        (
            texts_train, texts_test,
            subj_train, subj_test,
            body_train, body_test,
            send_train, send_test,
            y_train, y_test,
        ) = train_test_split(
            texts, subjects, bodies, senders, labels,
            test_size=test_size, random_state=42, stratify=labels
        )

        # TF-IDF features
        tfidf_train = self.tfidf_vectorizer.fit_transform(texts_train)
        tfidf_test = self.tfidf_vectorizer.transform(texts_test)

        # Heuristic features
        heuristic_train = self._get_heuristic_matrix(subj_train, body_train, send_train)
        heuristic_test = self._get_heuristic_matrix(subj_test, body_test, send_test)

        # Combine features
        X_train = hstack([tfidf_train, csr_matrix(heuristic_train)])
        X_test = hstack([tfidf_test, csr_matrix(heuristic_test)])

        # Train
        self.classifier.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.classifier.predict(X_test)
        y_prob = self.classifier.predict_proba(X_test)

        self.metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "test_size": len(y_test),
            "train_size": len(y_train),
        }

        # Feature importances
        n_tfidf = tfidf_train.shape[1]
        importances = self.classifier.feature_importances_
        tfidf_names = self.tfidf_vectorizer.get_feature_names_out()

        # Top TF-IDF features
        tfidf_importance = importances[:n_tfidf]
        top_tfidf_indices = np.argsort(tfidf_importance)[-15:][::-1]
        self.feature_importances = {}
        for idx in top_tfidf_indices:
            self.feature_importances[f"word: {tfidf_names[idx]}"] = round(
                float(tfidf_importance[idx]), 6
            )

        # Heuristic feature importances
        heuristic_importance = importances[n_tfidf:]
        for name, imp in zip(HEURISTIC_FEATURE_NAMES, heuristic_importance):
            self.feature_importances[name] = round(float(imp), 6)

        # Sort by importance
        self.feature_importances = dict(
            sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)
        )

        return self.metrics

    def predict(
        self, subject: str, body: str, sender: str
    ) -> Dict:
        """
        Predict whether an email is phishing.

        Returns:
            dict with keys:
                - prediction: "phishing" or "legitimate"
                - confidence: float 0-1
                - risk_score: float 0-100
                - risk_factors: list of identified risk factors
                - heuristic_features: dict of extracted features
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() first or load a model.")

        text = f"{subject} {body}"
        tfidf_vec = self.tfidf_vectorizer.transform([text])
        heuristic_feats = extract_heuristic_features(subject, body, sender)
        heuristic_vec = np.array([features_to_vector(heuristic_feats)])
        X = hstack([tfidf_vec, csr_matrix(heuristic_vec)])

        proba = self.classifier.predict_proba(X)[0]
        phishing_prob = float(proba[1])
        prediction = "phishing" if phishing_prob >= 0.5 else "legitimate"

        # Identify risk factors
        risk_factors = self._identify_risk_factors(heuristic_feats)

        return {
            "prediction": prediction,
            "label": 1 if prediction == "phishing" else 0,
            "confidence": round(max(proba), 4),
            "phishing_probability": round(phishing_prob, 4),
            "risk_score": round(phishing_prob * 100, 1),
            "risk_factors": risk_factors,
            "heuristic_features": {k: round(v, 4) if isinstance(v, float) else v
                                   for k, v in heuristic_feats.items()},
        }

    def _identify_risk_factors(self, features: Dict[str, float]) -> List[Dict[str, str]]:
        """Identify human-readable risk factors from heuristic features."""
        factors = []

        if features.get("urgency_keyword_count", 0) >= 2:
            factors.append({
                "category": "Social Engineering",
                "severity": "high",
                "description": "Multiple urgency keywords detected — a common phishing tactic to pressure quick action",
            })
        elif features.get("urgency_keyword_count", 0) >= 1:
            factors.append({
                "category": "Social Engineering",
                "severity": "medium",
                "description": "Urgency language detected in the email",
            })

        if features.get("credential_phrase_count", 0) >= 1:
            factors.append({
                "category": "Credential Harvesting",
                "severity": "critical",
                "description": "Email requests personal credentials or sensitive information",
            })

        if features.get("threat_phrase_count", 0) >= 1:
            factors.append({
                "category": "Intimidation",
                "severity": "high",
                "description": "Threatening language used to coerce action (account suspension, legal action, etc.)",
            })

        if features.get("reward_phrase_count", 0) >= 1:
            factors.append({
                "category": "Lure",
                "severity": "medium",
                "description": "Reward or prize language detected — potential social engineering lure",
            })

        if features.get("ip_url_count", 0) >= 1:
            factors.append({
                "category": "Suspicious URL",
                "severity": "critical",
                "description": "IP-address based URL detected — often used in phishing attacks to avoid domain blocking",
            })

        if features.get("shortener_url_count", 0) >= 1:
            factors.append({
                "category": "Obfuscated URL",
                "severity": "high",
                "description": "URL shortener detected — used to hide malicious destination",
            })

        if features.get("suspicious_tld_count", 0) >= 1:
            factors.append({
                "category": "Suspicious Domain",
                "severity": "medium",
                "description": "Link uses a suspicious top-level domain often associated with phishing",
            })

        if features.get("sender_domain_digit_count", 0) >= 1:
            factors.append({
                "category": "Sender Spoofing",
                "severity": "high",
                "description": "Sender domain contains digit substitutions (homoglyph attack, e.g., paypa1 → paypal)",
            })

        if features.get("caps_word_ratio", 0) > 0.15:
            factors.append({
                "category": "Stylistic Anomaly",
                "severity": "low",
                "description": "High proportion of ALL-CAPS words — typical of urgency-driven phishing emails",
            })

        if features.get("subject_all_caps", 0) == 1.0:
            factors.append({
                "category": "Stylistic Anomaly",
                "severity": "low",
                "description": "Subject line is entirely in uppercase — unusual for legitimate emails",
            })

        if not factors:
            factors.append({
                "category": "Clean",
                "severity": "info",
                "description": "No significant phishing indicators detected",
            })

        return factors

    def save(self, path: Optional[str] = None):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained.")

        save_dir = path or MODEL_DIR
        os.makedirs(save_dir, exist_ok=True)

        joblib.dump(self.classifier, os.path.join(save_dir, "classifier.joblib"))
        joblib.dump(self.tfidf_vectorizer, os.path.join(save_dir, "tfidf_vectorizer.joblib"))
        with open(os.path.join(save_dir, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=2)
        with open(os.path.join(save_dir, "feature_importances.json"), "w") as f:
            json.dump(self.feature_importances, f, indent=2)

        print(f"Model saved to {save_dir}")

    def load(self, path: Optional[str] = None):
        """Load a trained model from disk."""
        load_dir = path or MODEL_DIR

        self.classifier = joblib.load(os.path.join(load_dir, "classifier.joblib"))
        self.tfidf_vectorizer = joblib.load(os.path.join(load_dir, "tfidf_vectorizer.joblib"))

        metrics_path = os.path.join(load_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                self.metrics = json.load(f)

        fi_path = os.path.join(load_dir, "feature_importances.json")
        if os.path.exists(fi_path):
            with open(fi_path) as f:
                self.feature_importances = json.load(f)

        self.is_trained = True
        print(f"Model loaded from {load_dir}")
