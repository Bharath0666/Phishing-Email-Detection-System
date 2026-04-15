"""
Email Header Analyzer
Analyzes email headers for spoofing and anomaly indicators.
"""

import re
from typing import Dict, List, Optional


def analyze_headers(headers: Dict[str, str]) -> Dict:
    """
    Analyze email headers for phishing / spoofing indicators.

    Expected headers dict keys (all optional):
        - from: sender email/display
        - reply_to: reply-to address
        - return_path: return path
        - received: received chain (can be multiline)
        - x_mailer: X-Mailer header
        - authentication_results: SPF/DKIM/DMARC results
        - message_id: Message-ID
        - content_type: Content-Type

    Returns:
        dict with:
            - anomaly_score: 0-100
            - findings: list of finding dicts
    """
    findings = []
    anomaly_score = 0

    from_addr = headers.get("from", "")
    reply_to = headers.get("reply_to", "")
    return_path = headers.get("return_path", "")
    received = headers.get("received", "")
    x_mailer = headers.get("x_mailer", "")
    auth_results = headers.get("authentication_results", "")
    message_id = headers.get("message_id", "")
    content_type = headers.get("content_type", "")

    # 1. From / Reply-To mismatch
    if from_addr and reply_to:
        from_domain = _extract_domain(from_addr)
        reply_domain = _extract_domain(reply_to)
        if from_domain and reply_domain and from_domain != reply_domain:
            findings.append({
                "type": "From/Reply-To Mismatch",
                "severity": "high",
                "detail": f"From domain ({from_domain}) differs from Reply-To domain ({reply_domain}) — possible spoofing",
            })
            anomaly_score += 25

    # 2. From / Return-Path mismatch
    if from_addr and return_path:
        from_domain = _extract_domain(from_addr)
        return_domain = _extract_domain(return_path)
        if from_domain and return_domain and from_domain != return_domain:
            findings.append({
                "type": "From/Return-Path Mismatch",
                "severity": "high",
                "detail": f"From domain ({from_domain}) differs from Return-Path domain ({return_domain})",
            })
            anomaly_score += 20

    # 3. SPF/DKIM/DMARC analysis
    if auth_results:
        auth_lower = auth_results.lower()
        if "spf=fail" in auth_lower or "spf=softfail" in auth_lower:
            findings.append({
                "type": "SPF Failure",
                "severity": "critical",
                "detail": "SPF check failed — sender's IP is not authorized to send on behalf of this domain",
            })
            anomaly_score += 30

        if "dkim=fail" in auth_lower:
            findings.append({
                "type": "DKIM Failure",
                "severity": "critical",
                "detail": "DKIM signature verification failed — email may have been tampered with",
            })
            anomaly_score += 30

        if "dmarc=fail" in auth_lower:
            findings.append({
                "type": "DMARC Failure",
                "severity": "critical",
                "detail": "DMARC check failed — high likelihood of domain spoofing",
            })
            anomaly_score += 30

        if "spf=pass" in auth_lower and "dkim=pass" in auth_lower:
            findings.append({
                "type": "Authentication Passed",
                "severity": "info",
                "detail": "SPF and DKIM both passed — sender authentication looks legitimate",
            })
    elif from_addr:
        # Context-aware: Missing auth is only high-severity if the sender
        # claims to be a well-known brand (which WOULD have SPF/DKIM).
        # Small legit domains often lack proper auth — don't over-penalize.
        from_domain = _extract_domain(from_addr) or ""
        BRAND_KEYWORDS = [
            "paypal", "google", "microsoft", "apple", "amazon", "netflix",
            "facebook", "instagram", "linkedin", "chase", "wellsfargo",
            "bankofamerica", "dropbox", "twitter", "adobe", "zoom", "slack",
        ]
        is_brand_domain = any(bk in from_domain for bk in BRAND_KEYWORDS)
        if is_brand_domain:
            findings.append({
                "type": "Missing Authentication",
                "severity": "high",
                "detail": f"No SPF/DKIM/DMARC results for brand domain '{from_domain}' — legitimate brands always authenticate",
            })
            anomaly_score += 20
        else:
            findings.append({
                "type": "Missing Authentication",
                "severity": "low",
                "detail": "No SPF/DKIM/DMARC authentication results — many small domains lack proper auth configuration",
            })
            anomaly_score += 5

    # 4. X-Mailer analysis
    if x_mailer:
        suspicious_mailers = ["phpmailer", "swiftmailer", "mass mailer",
                              "bulk mail", "email blaster"]
        if any(m in x_mailer.lower() for m in suspicious_mailers):
            findings.append({
                "type": "Suspicious Mailer",
                "severity": "medium",
                "detail": f"Email sent using bulk/mass mailing tool: {x_mailer}",
            })
            anomaly_score += 15

    # 5. Message-ID analysis
    if message_id:
        # Check if Message-ID domain matches From domain
        mid_domain = _extract_domain(message_id)
        from_domain = _extract_domain(from_addr)
        if mid_domain and from_domain and mid_domain != from_domain:
            findings.append({
                "type": "Message-ID Mismatch",
                "severity": "medium",
                "detail": f"Message-ID domain ({mid_domain}) doesn't match sender domain ({from_domain})",
            })
            anomaly_score += 10

    # 6. Received chain analysis
    if received:
        # Count relay hops
        hop_count = received.lower().count("received:")
        if not hop_count:
            hop_count = received.count("\n") + 1

        if hop_count > 6:
            findings.append({
                "type": "Excessive Relay Hops",
                "severity": "medium",
                "detail": f"Email passed through {hop_count} relay servers — unusual routing",
            })
            anomaly_score += 10

    # 7. Display name vs email mismatch (spoofed display name)
    if from_addr:
        display_match = re.match(r'^"?([^"<]+)"?\s*<([^>]+)>', from_addr)
        if display_match:
            display_name = display_match.group(1).strip().lower()
            actual_email = display_match.group(2).strip().lower()
            # Check if display name looks like an email but differs
            if "@" in display_name and display_name != actual_email:
                findings.append({
                    "type": "Display Name Spoofing",
                    "severity": "critical",
                    "detail": f"Display name shows '{display_match.group(1).strip()}' but actual email is '{actual_email}'",
                })
                anomaly_score += 25

    anomaly_score = min(anomaly_score, 100)

    if not findings:
        findings.append({
            "type": "Clean",
            "severity": "info",
            "detail": "No significant header anomalies detected",
        })

    return {
        "anomaly_score": anomaly_score,
        "total_checks": 7,
        "findings": findings,
    }


def _extract_domain(text: str) -> Optional[str]:
    """Extract domain from an email address or header value."""
    # Try to find email pattern first
    email_match = re.search(r'[\w.+-]+@([\w.-]+)', text)
    if email_match:
        return email_match.group(1).lower()
    # try bare domain
    domain_match = re.search(r'@([\w.-]+)', text)
    if domain_match:
        return domain_match.group(1).lower()
    return None
