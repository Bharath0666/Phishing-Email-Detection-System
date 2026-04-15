"""
URL Threat Analyzer
Analyzes URLs found in emails for phishing indicators.
"""

import re
from typing import Dict, List
from urllib.parse import urlparse


# Known suspicious patterns
SUSPICIOUS_TLDS = [
    ".xyz", ".top", ".club", ".work", ".click", ".link",
    ".info", ".online", ".site", ".tech", ".space", ".fun",
    ".icu", ".buzz", ".gq", ".ml", ".cf", ".ga", ".tk",
]

URL_SHORTENERS = [
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
    "is.gd", "v.gd", "buff.ly", "rebrand.ly", "cutt.ly",
    "short.to", "lnkd.in", "surl.li",
]

# Common brands often impersonated
BRAND_DOMAINS = {
    "paypal": "paypal.com",
    "google": "google.com",
    "microsoft": "microsoft.com",
    "apple": "apple.com",
    "amazon": "amazon.com",
    "netflix": "netflix.com",
    "facebook": "facebook.com",
    "instagram": "instagram.com",
    "twitter": "twitter.com",
    "linkedin": "linkedin.com",
    "dropbox": "dropbox.com",
    "chase": "chase.com",
    "wellsfargo": "wellsfargo.com",
    "bankofamerica": "bankofamerica.com",
}

# Common homoglyph substitutions
HOMOGLYPHS = {
    "a": ["@", "4", "à", "á", "â", "ã", "ä"],
    "e": ["3", "è", "é", "ê", "ë"],
    "i": ["1", "!", "l", "|", "ì", "í", "î", "ï"],
    "o": ["0", "ò", "ó", "ô", "õ", "ö"],
    "l": ["1", "|", "I"],
    "s": ["5", "$"],
    "t": ["7", "+"],
    "g": ["9", "q"],
}


def analyze_url(url: str) -> Dict:
    """
    Analyze a single URL for phishing indicators.
    
    Performs deep analysis including:
    - Protocol security, IP-based URLs, URL shorteners
    - Brand impersonation, homoglyph detection
    - Shannon entropy (randomness detection)
    - URL length scoring, subdomain depth
    - Path keyword analysis, @ symbol injection
    - Suspicious TLD detection

    Returns a dict with:
        - url: the original URL
        - threat_score: 0-100
        - findings: list of finding dicts
    """
    findings = []
    threat_score = 0

    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        scheme = parsed.scheme or ""
        path = parsed.path or ""
        full_url = url
    except Exception:
        return {
            "url": url,
            "threat_score": 50,
            "findings": [{"type": "Parse Error", "severity": "medium",
                          "detail": "Could not parse URL structure"}],
        }

    # 1. HTTP vs HTTPS
    if scheme == "http":
        findings.append({
            "type": "Insecure Protocol",
            "severity": "medium",
            "detail": "Uses HTTP (unencrypted) instead of HTTPS",
        })
        threat_score += 15

    # 2. IP-based URL
    if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', hostname):
        findings.append({
            "type": "IP-Based URL",
            "severity": "critical",
            "detail": f"URL points to raw IP address ({hostname}) instead of a domain name",
        })
        threat_score += 35

    # 3. URL shortener
    if any(shortener in hostname for shortener in URL_SHORTENERS):
        findings.append({
            "type": "URL Shortener",
            "severity": "high",
            "detail": "Uses a URL shortening service to obscure the destination",
        })
        threat_score += 25

    # 4. Suspicious TLD
    matched_tld = None
    for tld in SUSPICIOUS_TLDS:
        if hostname.endswith(tld):
            matched_tld = tld
            break
    if matched_tld:
        findings.append({
            "type": "Suspicious TLD",
            "severity": "high",
            "detail": f"Domain uses suspicious top-level domain '{matched_tld}' — commonly abused in phishing",
        })
        threat_score += 20

    # 5. Subdomain depth analysis
    subdomain_count = hostname.count(".")
    if subdomain_count >= 4:
        findings.append({
            "type": "Excessive Subdomains",
            "severity": "critical",
            "detail": f"URL has {subdomain_count} subdomain levels — likely hiding the real domain in a deep subdomain chain",
        })
        threat_score += 25
    elif subdomain_count >= 3:
        findings.append({
            "type": "Deep Subdomains",
            "severity": "high",
            "detail": f"URL has {subdomain_count} subdomain levels — may be trying to hide the real domain",
        })
        threat_score += 20

    # 6. Brand impersonation (domain contains brand name but isn't the real domain)
    for brand, real_domain in BRAND_DOMAINS.items():
        if brand in hostname.replace(".", "").lower() and real_domain not in hostname:
            findings.append({
                "type": "Brand Impersonation",
                "severity": "critical",
                "detail": f"Domain contains '{brand}' but is not the legitimate {real_domain}",
            })
            threat_score += 30
            break

    # 7. Homoglyph / digit substitution
    domain_base = hostname.split(".")[0] if hostname else ""
    digit_subs = len(re.findall(r'\d', domain_base))
    if digit_subs >= 1 and not re.match(r'^\d+$', domain_base):
        findings.append({
            "type": "Homoglyph Attack",
            "severity": "high",
            "detail": "Domain contains digit substitutions that may mimic a legitimate domain",
        })
        threat_score += 20

    # 8. Shannon Entropy — context-aware (entropy ALONE is not sufficient)
    # Many legit domains have high entropy (CDNs, UUID subdomains, etc.)
    # Only flag when combined with other suspicious signals.
    if hostname and not re.match(r'^\d{1,3}(\.\d{1,3}){3}$', hostname):
        entropy = _shannon_entropy(domain_base)
        has_sus_tld = matched_tld is not None
        has_brand_kw = any(bk in hostname.replace(".", "").lower() for bk in BRAND_DOMAINS)

        if entropy > 4.0 and (has_sus_tld or has_brand_kw):
            findings.append({
                "type": "High Entropy Domain",
                "severity": "high",
                "detail": f"Randomly generated domain (entropy: {entropy:.2f}) combined with {'suspicious TLD' if has_sus_tld else 'brand keyword'} — strong phishing indicator",
            })
            threat_score += 20
        elif entropy > 3.5 and (has_sus_tld or has_brand_kw):
            findings.append({
                "type": "Moderate Entropy Domain",
                "severity": "medium",
                "detail": f"Above-average randomness (entropy: {entropy:.2f}) combined with {'suspicious TLD' if has_sus_tld else 'brand keyword'}",
            })
            threat_score += 10
        elif entropy > 4.0:
            # High entropy alone — informational only, no score impact
            findings.append({
                "type": "Elevated Entropy",
                "severity": "info",
                "detail": f"Domain has high randomness (entropy: {entropy:.2f}) — not necessarily malicious alone",
            })

    # 9. URL Length scoring
    url_len = len(full_url)
    if url_len > 200:
        findings.append({
            "type": "Extremely Long URL",
            "severity": "high",
            "detail": f"URL is {url_len} characters long — excessive length used to hide malicious content",
        })
        threat_score += 20
    elif url_len > 100:
        findings.append({
            "type": "Long URL",
            "severity": "medium",
            "detail": f"URL is {url_len} characters long — above average, may contain obfuscation",
        })
        threat_score += 10

    # 10. '@' symbol in URL (used to trick browsers)
    if "@" in url:
        findings.append({
            "type": "@ Symbol in URL",
            "severity": "critical",
            "detail": "URL contains @ symbol — browsers may ignore everything before @, redirecting to attacker's domain",
        })
        threat_score += 30

    # 11. Suspicious path keywords (credential harvesting)
    suspicious_paths = ["login", "verify", "secure", "update", "confirm",
                        "account", "signin", "auth", "credential", "password",
                        "banking", "wallet", "recover", "validate"]
    path_lower = path.lower()
    path_matches = [p for p in suspicious_paths if p in path_lower]
    if len(path_matches) >= 2:
        findings.append({
            "type": "Credential Harvesting Path",
            "severity": "high",
            "detail": f"URL path contains multiple credential-related keywords: {', '.join(path_matches)}",
        })
        threat_score += 20
    elif path_matches:
        findings.append({
            "type": "Suspicious Path",
            "severity": "medium",
            "detail": f"URL path contains credential-related keywords: {', '.join(path_matches)}",
        })
        threat_score += 10

    # 12. Path depth — deeply nested paths are suspicious
    path_depth = len([p for p in path.split("/") if p])
    if path_depth >= 5:
        findings.append({
            "type": "Deep Path Nesting",
            "severity": "medium",
            "detail": f"URL has {path_depth} directory levels — deep nesting can indicate obfuscation",
        })
        threat_score += 10

    # 13. Port number in URL
    if parsed.port and parsed.port not in [80, 443]:
        findings.append({
            "type": "Unusual Port",
            "severity": "medium",
            "detail": f"URL uses non-standard port {parsed.port}",
        })
        threat_score += 10

    # Cap score at 100
    threat_score = min(threat_score, 100)

    if not findings:
        findings.append({
            "type": "Clean",
            "severity": "info",
            "detail": "No significant threats detected in this URL",
        })

    return {
        "url": url,
        "hostname": hostname,
        "threat_score": threat_score,
        "findings": findings,
    }


def _shannon_entropy(text: str) -> float:
    """Calculate Shannon entropy of a string (higher = more random)."""
    if not text:
        return 0.0
    import math
    freq = {}
    for c in text.lower():
        freq[c] = freq.get(c, 0) + 1
    length = len(text)
    entropy = 0.0
    for count in freq.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 4)


def analyze_urls_in_text(text: str) -> Dict:
    """
    Extract and analyze all URLs found in email body text.

    Returns:
        dict with:
            - total_urls: count of URLs found
            - max_threat_score: highest threat score among URLs
            - avg_threat_score: average threat score
            - url_analyses: list of per-URL analysis results
    """
    url_pattern = r'https?://[^\s<>"\')\]]+|www\.[^\s<>"\')\]]+'
    urls = re.findall(url_pattern, text, re.IGNORECASE)

    if not urls:
        return {
            "total_urls": 0,
            "max_threat_score": 0,
            "avg_threat_score": 0,
            "url_analyses": [],
        }

    analyses = [analyze_url(url) for url in set(urls)]  # deduplicate
    scores = [a["threat_score"] for a in analyses]

    return {
        "total_urls": len(urls),
        "unique_urls": len(analyses),
        "max_threat_score": max(scores),
        "avg_threat_score": round(sum(scores) / len(scores), 1),
        "url_analyses": analyses,
    }
