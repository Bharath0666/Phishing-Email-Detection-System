"""
Flask Backend — Phishing Email Detection API
Serves the ML model, URL analyzer, and header analyzer as REST endpoints.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from ml.model import PhishingDetector
from ml.dataset import load_all_datasets, preprocess, get_dataset_info
from analyzers.url_analyzer import analyze_urls_in_text
from analyzers.header_analyzer import analyze_headers


# ─── App Setup ───────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# Load model
detector = PhishingDetector()
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
dataset_info = {}

if os.path.exists(os.path.join(MODEL_DIR, "classifier.joblib")):
    detector.load(MODEL_DIR)
    print("✓ Model loaded successfully")
else:
    print("⚠ No trained model found. Training on real datasets...")
    df = load_all_datasets()
    df = preprocess(df)
    dataset_info = get_dataset_info(df)
    print(f"  Dataset: {dataset_info['total_samples']:,} samples")
    metrics = detector.train(
        subjects=df["subject"].tolist(),
        bodies=df["body"].tolist(),
        senders=["unknown@unknown.com"] * len(df),
        labels=df["label"].tolist(),
    )
    detector.save(MODEL_DIR)
    print(f"✓ Model trained — Accuracy: {metrics['accuracy']:.2%}")


# ─── Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": detector.is_trained,
    })


@app.route("/api/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "metrics": detector.metrics,
        "feature_importances": dict(list(detector.feature_importances.items())[:20]),
        "model_type": "Random Forest (200 estimators) + TF-IDF (5000 features)",
        "features": "TF-IDF bigram features + 20 cybersecurity heuristic features",
        "dataset_info": dataset_info,
        "training_data": "7 real-world email datasets (~165K emails)",
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Analyze an email for phishing indicators.

    Expects JSON:
    {
        "subject": "...",
        "body": "...",
        "sender": "...",
        "headers": {                    // optional
            "from": "...",
            "reply_to": "...",
            "authentication_results": "...",
            ...
        }
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    subject = data.get("subject", "")
    body = data.get("body", "")
    sender = data.get("sender", "")
    headers = data.get("headers", {})

    if not body and not subject:
        return jsonify({"error": "Please provide at least the email body or subject"}), 400

    # 1. ML Prediction
    ml_result = detector.predict(subject, body, sender)

    # 2. URL Analysis
    url_result = analyze_urls_in_text(body)

    # 3. Header Analysis — auto-simulate from sender if no raw headers provided
    if headers:
        header_result = analyze_headers(headers)
    else:
        # Simulate basic header analysis from the sender email address
        simulated_headers = {}
        if sender:
            simulated_headers["from"] = sender
            # Simulate reply-to / return-path mismatch if sender looks suspicious
            sender_domain = sender.split("@")[-1] if "@" in sender else ""
            simulated_headers["return_path"] = sender
            # Check for digit substitutions in domain (paypa1 → paypal)
            import re as _re
            if sender_domain and _re.search(r'\d', sender_domain.split(".")[0]):
                # Likely spoofed — simulate a mismatch
                simulated_headers["reply_to"] = f"real-reply@{sender_domain.replace('1','l').replace('0','o')}"
            # No auth results = suspicious
            simulated_headers["authentication_results"] = ""
        header_result = analyze_headers(simulated_headers) if simulated_headers else {
            "anomaly_score": 0,
            "total_checks": 0,
            "findings": [{"type": "Not Analyzed", "severity": "info",
                           "detail": "No sender email provided for header simulation"}],
        }

    # 4. Compute overall risk score with intelligent weighting
    ml_risk = ml_result["risk_score"]
    url_risk = url_result.get("max_threat_score", 0)
    header_risk = header_result.get("anomaly_score", 0)

    # Base weighted score
    overall_risk = (ml_risk * 0.55) + (url_risk * 0.25) + (header_risk * 0.20)

    # ── Critical Factor Boosting ──
    # If ML says phishing with high confidence, enforce minimum risk
    phishing_prob = ml_result.get("phishing_probability", 0)
    if phishing_prob >= 0.80:
        overall_risk = max(overall_risk, 80)
    elif phishing_prob >= 0.65:
        overall_risk = max(overall_risk, 65)
    elif phishing_prob >= 0.50:
        overall_risk = max(overall_risk, 50)

    # If critical risk factors exist (credential harvesting, brand impersonation, etc.)
    has_critical = any(
        f.get("severity") == "critical"
        for f in ml_result.get("risk_factors", [])
    )
    has_critical_url = any(
        f.get("severity") == "critical"
        for ua in url_result.get("url_analyses", [])
        for f in ua.get("findings", [])
    )
    if has_critical or has_critical_url:
        overall_risk = max(overall_risk, 70)

    # If multiple high-severity signals converge, boost further
    high_signal_count = sum([
        1 if ml_risk >= 60 else 0,
        1 if url_risk >= 40 else 0,
        1 if header_risk >= 25 else 0,
    ])
    if high_signal_count >= 2:
        overall_risk = max(overall_risk, 75)

    overall_risk = min(round(overall_risk, 1), 100)

    # ── Verdict Thresholds (tight, no contradictions) ──
    if overall_risk >= 80:
        verdict = "phishing"
        verdict_text = "Critical Risk — Phishing Detected"
    elif overall_risk >= 60:
        verdict = "phishing"
        verdict_text = "High Risk — Likely Phishing"
    elif overall_risk >= 30:
        verdict = "suspicious"
        verdict_text = "Moderate Risk — Suspicious Email"
    else:
        verdict = "legitimate"
        verdict_text = "Low Risk — Likely Legitimate"

    response = {
        "verdict": verdict,
        "verdict_text": verdict_text,
        "overall_risk_score": overall_risk,
        "ml_analysis": ml_result,
        "url_analysis": url_result,
        "header_analysis": header_result,
    }

    return jsonify(response)


# ─── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
