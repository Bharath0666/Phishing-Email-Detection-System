# 🛡️ PhishGuard AI — Phishing Email Detection System

A full-stack, AI-powered phishing email detection system that combines **Machine Learning**, **URL threat intelligence**, and **Email header forensics** to identify phishing attacks in real time.

Built with a **Random Forest classifier** trained on **165,000+ real-world emails** across 7 industry-standard datasets, achieving **94.67% accuracy** and a **97.52% phishing recall rate**.

---

## 🎯 Key Features

| Feature | Description |
|---|---|
| **ML Classification** | Random Forest with TF-IDF (5,000 features) + 20 cybersecurity heuristic features |
| **URL Threat Analysis** | 13-point deep URL inspection — IP-based URLs, brand impersonation, homoglyph attacks, Shannon entropy, URL shorteners, suspicious TLDs |
| **Header Forensics** | SPF/DKIM/DMARC validation, From/Reply-To mismatch detection, display name spoofing, relay hop analysis |
| **Multi-Signal Fusion** | Intelligent weighted scoring (ML 55% + URL 25% + Header 20%) with critical-factor boosting |
| **Real-Time Web UI** | Dark-themed dashboard with animated risk gauge, severity-coded findings, and interactive analysis panels |

---

## 🏗️ Architecture

```
┌───────────────────────────────────┐
│          Frontend (HTML/CSS/JS)   │
│  ┌─────────────────────────────┐  │
│  │  PhishGuard AI Dashboard    │  │
│  │  - Email input form         │  │
│  │  - Risk gauge visualization │  │
│  │  - Analysis result panels   │  │
│  └──────────┬──────────────────┘  │
│             │ REST API             │
├─────────────┼─────────────────────┤
│          Flask Backend            │
│  ┌──────────┴──────────────────┐  │
│  │     /api/analyze (POST)     │  │
│  │     /api/health  (GET)      │  │
│  │     /api/model-info (GET)   │  │
│  └──┬──────────┬──────────┬────┘  │
│     │          │          │       │
│  ┌──┴───┐ ┌───┴────┐ ┌───┴────┐  │
│  │  ML  │ │  URL   │ │ Header │  │
│  │Engine│ │Analyzer│ │Analyzer│  │
│  └──────┘ └────────┘ └────────┘  │
└───────────────────────────────────┘
```

---

## 📊 Model Performance

Trained on 7 real-world email datasets and evaluated on a held-out test set of 31,842 samples:

| Metric | Score |
|---|---|
| **Accuracy** | 94.67% |
| **Precision** | 92.50% |
| **Recall** | 97.52% |
| **F1 Score** | 94.94% |

**Confusion Matrix:**

|  | Predicted Legitimate | Predicted Phishing |
|---|---|---|
| **Actual Legitimate** | 14,205 | 1,293 |
| **Actual Phishing** | 405 | 15,939 |

> The model prioritizes **high recall** (97.52%) to minimize missed phishing emails — the costliest error in email security.

---

## 🔬 Detection Techniques

### 1. ML-Based Content Analysis
- **TF-IDF Vectorization** — Extracts 5,000 bigram text features with sublinear TF scaling
- **Cybersecurity Heuristics** — 20 handcrafted features including:
  - Urgency keyword density & credential harvesting phrases
  - Threat/reward language detection
  - URL pattern analysis (IP-based, shorteners, suspicious TLDs)
  - Sender domain anomalies (digit substitutions, subdomain depth)
  - Stylistic signals (caps ratio, exclamation density, HTML presence)

### 2. URL Threat Analysis (13 checks)
- Protocol security (HTTP vs HTTPS)
- IP-based URL detection
- URL shortener identification
- Brand impersonation & homoglyph attack detection
- Shannon entropy analysis (context-aware)
- Suspicious TLD & excessive subdomain detection
- `@` symbol injection & credential harvesting path keywords
- URL length scoring & path depth analysis

### 3. Email Header Forensics (7 checks)
- From / Reply-To domain mismatch
- From / Return-Path mismatch
- SPF, DKIM, DMARC authentication validation
- Display name spoofing detection
- Suspicious X-Mailer identification
- Message-ID domain consistency
- Relay hop count analysis

### 4. Multi-Signal Risk Scoring
- Weighted fusion: **ML (55%) + URL (25%) + Header (20%)**
- Critical-factor boosting when ML confidence ≥ 80%
- Signal convergence amplification (multiple high-severity indicators)
- Four-tier verdict system: Critical Risk → High Risk → Moderate Risk → Low Risk

---

## 📂 Project Structure

```
Phishing-Email-Detection-System/
├── app.py                      # Flask backend — API routes & risk scoring engine
├── train_phishing_model.py     # Standalone training script (Colab-compatible)
├── train_model.py              # Quick training entry point
├── requirements.txt            # Python dependencies
│
├── ml/                         # Machine Learning module
│   ├── model.py                # PhishingDetector class (train, predict, save/load)
│   ├── feature_extract.py      # Heuristic feature engineering (20 features)
│   └── dataset.py              # Dataset loader & preprocessor
│
├── analyzers/                  # Security analysis modules
│   ├── url_analyzer.py         # 13-point URL threat analysis
│   └── header_analyzer.py      # 7-point email header forensics
│
├── models/                     # Trained model artifacts
│   ├── classifier.joblib       # Serialized Random Forest model
│   ├── tfidf_vectorizer.joblib # Fitted TF-IDF vectorizer
│   ├── metrics.json            # Evaluation metrics
│   └── feature_importances.json
│
└── static/                     # Frontend
    ├── index.html              # Dashboard UI
    ├── style.css               # Dark theme styles & animations
    └── app.js                  # Frontend logic & API integration
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Bharath0666/Phishing-Email-Detection-System.git
cd Phishing-Email-Detection-System

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

The app will start at **http://localhost:5000**. The pre-trained model loads automatically from the `models/` directory.

> **Note:** If no trained model is found, the app will attempt to train one from scratch using datasets in the `archive/` directory.

---

## 🗃️ Training Datasets

The model was trained on 7 real-world email datasets (~165K emails total):

| Dataset | Description |
|---|---|
| **CEAS_08** | Conference on Email and Anti-Spam 2008 corpus |
| **Enron** | Enron email corpus (legitimate emails) |
| **Ling** | Ling-Spam dataset |
| **Nazario** | Jose Nazario's phishing corpus |
| **Nigerian_Fraud** | Nigerian fraud / 419 scam emails |
| **SpamAssasin** | SpamAssassin public corpus |
| **phishing_email** | Aggregated phishing email collection |

> Datasets are excluded from this repository due to size (~260 MB). Place CSV files in an `archive/` directory to retrain.

---

## 🔌 API Reference

### `POST /api/analyze`

Analyze an email for phishing indicators.

**Request Body:**
```json
{
    "subject": "URGENT: Verify your account",
    "body": "Click here to verify: http://paypa1-secure.xyz/login",
    "sender": "security@paypa1.com",
    "headers": {
        "from": "\"PayPal Security\" <security@paypa1.com>",
        "reply_to": "real-reply@paypal.com",
        "authentication_results": "spf=fail; dkim=fail"
    }
}
```

**Response:**
```json
{
    "verdict": "phishing",
    "verdict_text": "Critical Risk — Phishing Detected",
    "overall_risk_score": 92.5,
    "ml_analysis": { "prediction": "phishing", "confidence": 0.97, "risk_factors": [...] },
    "url_analysis": { "total_urls": 1, "max_threat_score": 85, "url_analyses": [...] },
    "header_analysis": { "anomaly_score": 55, "findings": [...] }
}
```

### `GET /api/health`
Returns system health and model status.

### `GET /api/model-info`
Returns model metrics, feature importances, and training info.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python, Flask, Flask-CORS |
| **ML** | scikit-learn (Random Forest, TF-IDF), NumPy, Pandas, SciPy |
| **Frontend** | HTML5, CSS3 (dark theme, glassmorphism), Vanilla JavaScript |
| **Serialization** | Joblib |

---

## 📄 License

This project is open source and available for educational and research purposes.
