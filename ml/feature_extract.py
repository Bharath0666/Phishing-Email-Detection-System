"""
Feature Extraction Module
Extracts NLP and cybersecurity-specific heuristic features from emails.
"""

import re
import math
from typing import Dict, List, Tuple
from urllib.parse import urlparse


# ─── Urgency & Phishing Keyword Sets ────────────────────────────────────

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
    "date of birth", "mother's maiden name", "personal information",
    "billing information", "payment details", "account number",
    "verify your account", "confirm your account", "reset your password",
    "update your information",
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

SUSPICIOUS_TLDS = [
    ".xyz", ".top", ".club", ".work", ".click", ".link",
    ".info", ".online", ".site", ".tech", ".space", ".fun",
    ".icu", ".buzz", ".gq", ".ml", ".cf", ".ga", ".tk",
]

URL_SHORTENERS = [
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
    "is.gd", "v.gd", "buff.ly", "rebrand.ly", "cutt.ly",
]


def extract_urls(text: str) -> List[str]:
    """Extract all URLs from text."""
    url_pattern = r'https?://[^\s<>"\')\]]+|www\.[^\s<>"\')\]]+'
    return re.findall(url_pattern, text, re.IGNORECASE)


def count_keyword_matches(text: str, keywords: List[str]) -> int:
    """Count how many keyword patterns appear in text."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def keyword_density(text: str, keywords: List[str]) -> float:
    """Calculate the density of keyword matches relative to total words."""
    words = text.split()
    if not words:
        return 0.0
    matches = count_keyword_matches(text, keywords)
    return matches / len(words)


def extract_heuristic_features(subject: str, body: str, sender: str) -> Dict[str, float]:
    """
    Extract cybersecurity-focused heuristic features from an email.

    Returns a dictionary of feature names to float values.
    """
    full_text = f"{subject} {body}"
    features = {}

    # 1. Urgency indicators
    features["urgency_keyword_count"] = count_keyword_matches(full_text, URGENCY_KEYWORDS)
    features["urgency_density"] = keyword_density(full_text, URGENCY_KEYWORDS)

    # 2. Credential request indicators
    features["credential_phrase_count"] = count_keyword_matches(full_text, CREDENTIAL_PHRASES)

    # 3. Threat/fear indicators
    features["threat_phrase_count"] = count_keyword_matches(full_text, THREAT_PHRASES)

    # 4. Reward/too-good-to-be-true indicators
    features["reward_phrase_count"] = count_keyword_matches(full_text, REWARD_PHRASES)

    # 5. URL analysis
    urls = extract_urls(body)
    features["url_count"] = len(urls)

    ip_url_count = 0
    shortener_count = 0
    suspicious_tld_count = 0
    for url in urls:
        try:
            parsed = urlparse(url)
            host = parsed.hostname or ""
            # IP-based URL
            if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', host):
                ip_url_count += 1
            # URL shortener
            if any(s in host for s in URL_SHORTENERS):
                shortener_count += 1
            # Suspicious TLD
            if any(host.endswith(tld) for tld in SUSPICIOUS_TLDS):
                suspicious_tld_count += 1
        except Exception:
            pass

    features["ip_url_count"] = ip_url_count
    features["shortener_url_count"] = shortener_count
    features["suspicious_tld_count"] = suspicious_tld_count

    # 6. Sender analysis
    sender_lower = sender.lower()
    sender_domain = sender_lower.split("@")[-1] if "@" in sender_lower else ""

    # Check for number substitutions in domain (e.g., paypa1.com, g00gle)
    digit_in_domain = len(re.findall(r'\d', sender_domain))
    features["sender_domain_digit_count"] = digit_in_domain

    # Check for excessive subdomains
    features["sender_subdomain_count"] = sender_domain.count(".") if sender_domain else 0

    # Check for suspicious sender TLD
    features["sender_suspicious_tld"] = 1.0 if any(
        sender_domain.endswith(tld) for tld in SUSPICIOUS_TLDS
    ) else 0.0

    # 7. Content style features
    words = full_text.split()
    features["total_word_count"] = len(words)
    features["exclamation_count"] = full_text.count("!")
    features["caps_word_ratio"] = (
        sum(1 for w in words if w.isupper() and len(w) > 1) / max(len(words), 1)
    )

    # 8. HTML content hints
    features["has_html_tags"] = 1.0 if re.search(r'<[a-zA-Z][^>]*>', body) else 0.0

    # 9. Subject line features
    features["subject_has_urgency"] = 1.0 if count_keyword_matches(subject, URGENCY_KEYWORDS) > 0 else 0.0
    features["subject_has_re_fw"] = 1.0 if re.match(r'^(re|fw|fwd):', subject, re.IGNORECASE) else 0.0
    features["subject_all_caps"] = 1.0 if subject == subject.upper() and len(subject) > 3 else 0.0

    # 10. Special character ratio
    special_chars = sum(1 for c in full_text if not c.isalnum() and not c.isspace())
    features["special_char_ratio"] = special_chars / max(len(full_text), 1)

    return features


HEURISTIC_FEATURE_NAMES = [
    "urgency_keyword_count", "urgency_density",
    "credential_phrase_count", "threat_phrase_count", "reward_phrase_count",
    "url_count", "ip_url_count", "shortener_url_count", "suspicious_tld_count",
    "sender_domain_digit_count", "sender_subdomain_count", "sender_suspicious_tld",
    "total_word_count", "exclamation_count", "caps_word_ratio",
    "has_html_tags",
    "subject_has_urgency", "subject_has_re_fw", "subject_all_caps",
    "special_char_ratio",
]


def features_to_vector(features: Dict[str, float]) -> List[float]:
    """Convert feature dict to ordered vector."""
    return [features.get(name, 0.0) for name in HEURISTIC_FEATURE_NAMES]
