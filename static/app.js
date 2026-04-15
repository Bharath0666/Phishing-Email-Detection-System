/**
 * PhishGuard AI — Frontend Application
 * Handles email analysis requests, renders results with animations.
 */

(() => {
    "use strict";

    // ─── DOM References ────────────────────────────────────────
    const $ = (sel) => document.querySelector(sel);
    const btnAnalyze = $("#btnAnalyze");
    const btnClear = $("#btnClear");
    const inputSection = $("#inputSection");
    const loadingSection = $("#loadingSection");
    const resultsSection = $("#resultsSection");
    const headerStatus = $("#headerStatus");

    // ─── API ───────────────────────────────────────────────────
    const API_BASE = "";

    async function apiPost(endpoint, data) {
        const res = await fetch(`${API_BASE}${endpoint}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ error: "Request failed" }));
            throw new Error(err.error || `HTTP ${res.status}`);
        }
        return res.json();
    }

    async function apiGet(endpoint) {
        const res = await fetch(`${API_BASE}${endpoint}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
    }

    // ─── Health Check ──────────────────────────────────────────
    async function checkHealth() {
        try {
            const data = await apiGet("/api/health");
            const dot = headerStatus.querySelector(".status-dot");
            const txt = headerStatus.querySelector("span:last-child");
            if (data.model_loaded) {
                dot.classList.add("online");
                txt.textContent = "Model Ready";
            } else {
                txt.textContent = "Model Not Loaded";
            }
        } catch {
            const txt = headerStatus.querySelector("span:last-child");
            txt.textContent = "Server Offline";
        }
    }

    // ─── Model Info ────────────────────────────────────────────
    async function loadModelInfo() {
        try {
            const data = await apiGet("/api/model-info");
            const grid = $("#modelInfoGrid");

            const items = [
                { label: "Model Type", value: data.model_type || "Random Forest + TF-IDF", small: false },
                { label: "Training Data", value: data.training_data || "Real-world datasets", small: false },
                { label: "Features", value: data.features || "—", small: true },
            ];

            if (data.metrics) {
                items.push(
                    { label: "Accuracy", value: `${(data.metrics.accuracy * 100).toFixed(1)}%`, small: false },
                    { label: "Precision", value: `${(data.metrics.precision * 100).toFixed(1)}%`, small: false },
                    { label: "Recall", value: `${(data.metrics.recall * 100).toFixed(1)}%`, small: false },
                    { label: "F1 Score", value: `${(data.metrics.f1_score * 100).toFixed(1)}%`, small: false },
                    { label: "Train / Test", value: `${data.metrics.train_size?.toLocaleString() || "—"} / ${data.metrics.test_size?.toLocaleString() || "—"}`, small: false },
                );
            }

            grid.innerHTML = items.map(item => `
                <div class="info-item">
                    <div class="info-item__label">${item.label}</div>
                    <div class="info-item__value ${item.small ? "info-item__value--sm" : ""}">${item.value}</div>
                </div>
            `).join("");
        } catch {
            $("#modelInfoGrid").innerHTML = '<p class="text-muted">Could not load model info.</p>';
        }
    }

    // ─── Analyze Email ─────────────────────────────────────────
    async function analyzeEmail() {
        const subject = $("#subject").value.trim();
        const body = $("#body").value.trim();
        const sender = $("#sender").value.trim();

        if (!subject && !body) {
            shakeElement(btnAnalyze);
            return;
        }

        // Build headers object
        const headers = {};
        const hdrFrom = $("#hdrFrom").value.trim();
        const hdrReplyTo = $("#hdrReplyTo").value.trim();
        const hdrAuth = $("#hdrAuth").value.trim();
        const hdrXMailer = $("#hdrXMailer").value.trim();

        if (hdrFrom) headers.from = hdrFrom;
        if (hdrReplyTo) headers.reply_to = hdrReplyTo;
        if (hdrAuth) headers.authentication_results = hdrAuth;
        if (hdrXMailer) headers.x_mailer = hdrXMailer;

        // Show loading
        btnAnalyze.disabled = true;
        resultsSection.style.display = "none";
        loadingSection.style.display = "";

        try {
            const result = await apiPost("/api/analyze", {
                subject,
                body,
                sender,
                headers: Object.keys(headers).length > 0 ? headers : undefined,
            });
            renderResults(result);
        } catch (err) {
            alert(`Analysis failed: ${err.message}`);
        } finally {
            btnAnalyze.disabled = false;
            loadingSection.style.display = "none";
        }
    }

    // ─── Render Results ────────────────────────────────────────
    function renderResults(data) {
        resultsSection.style.display = "";

        // Verdict Banner
        const banner = $("#verdictBanner");
        banner.className = `verdict-banner verdict-banner--${data.verdict}`;

        const icons = { phishing: "🚨", suspicious: "⚠️", legitimate: "✅" };
        $("#verdictIcon").textContent = icons[data.verdict] || "❓";
        $("#verdictTitle").textContent = data.verdict_text;
        $("#verdictSubtitle").textContent = `Overall Risk: ${data.overall_risk_score}/100`;

        // Risk Gauge
        animateGauge(data.overall_risk_score);

        // ML Panel
        const ml = data.ml_analysis;
        $("#mlBadge").textContent = ml.prediction.toUpperCase();
        $("#mlBadge").className = `badge badge--${ml.prediction}`;

        $("#mlStats").innerHTML = `
            <div class="stat">
                <span class="stat__value" style="color: ${ml.risk_score >= 70 ? "var(--sev-critical)" : ml.risk_score >= 40 ? "var(--accent-orange)" : "var(--accent-green)"}">${ml.risk_score}</span>
                <span class="stat__label">Risk Score</span>
            </div>
            <div class="stat">
                <span class="stat__value">${(ml.confidence * 100).toFixed(0)}%</span>
                <span class="stat__label">Confidence</span>
            </div>
            <div class="stat">
                <span class="stat__value">${(ml.phishing_probability * 100).toFixed(0)}%</span>
                <span class="stat__label">Phish Prob</span>
            </div>
        `;

        // Risk Factors
        const rfContainer = $("#riskFactors");
        rfContainer.innerHTML = ml.risk_factors.map(f => renderFinding(f.category, f.severity, f.description)).join("");

        // URL Panel
        const url = data.url_analysis;
        const urlThreat = url.max_threat_score || 0;
        $("#urlBadge").textContent = urlThreat >= 50 ? "THREATS FOUND" : url.total_urls > 0 ? "SCANNED" : "NO URLS";
        $("#urlBadge").className = `badge ${urlThreat >= 50 ? "badge--phishing" : "badge--safe"}`;

        $("#urlStats").innerHTML = `
            <div class="stat">
                <span class="stat__value">${url.total_urls || 0}</span>
                <span class="stat__label">URLs Found</span>
            </div>
            <div class="stat">
                <span class="stat__value" style="color: ${urlThreat >= 50 ? "var(--sev-critical)" : "var(--accent-green)"}">${urlThreat}</span>
                <span class="stat__label">Max Threat</span>
            </div>
        `;

        const urlFindings = $("#urlFindings");
        if (url.url_analyses && url.url_analyses.length > 0) {
            urlFindings.innerHTML = url.url_analyses.flatMap(ua =>
                ua.findings.map(f => renderFinding(f.type, f.severity, f.detail))
            ).join("");
        } else {
            urlFindings.innerHTML = renderFinding("No URLs", "info", "No URLs were found in the email body");
        }

        // Header Panel
        const hdr = data.header_analysis;
        const hdrScore = hdr.anomaly_score || 0;
        $("#headerBadge").textContent = hdrScore >= 30 ? "ANOMALIES" : "CLEAN";
        $("#headerBadge").className = `badge ${hdrScore >= 30 ? "badge--suspicious" : "badge--safe"}`;

        $("#headerStats").innerHTML = `
            <div class="stat">
                <span class="stat__value" style="color: ${hdrScore >= 30 ? "var(--accent-orange)" : "var(--accent-green)"}">${hdrScore}</span>
                <span class="stat__label">Anomaly Score</span>
            </div>
            <div class="stat">
                <span class="stat__value">${hdr.total_checks || 0}</span>
                <span class="stat__label">Checks Run</span>
            </div>
        `;

        const hdrFindings = $("#headerFindings");
        hdrFindings.innerHTML = hdr.findings.map(f => renderFinding(f.type, f.severity, f.detail)).join("");

        // Smooth scroll to results
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function renderFinding(type, severity, detail) {
        return `
            <div class="finding finding--${severity}">
                <div class="finding__header">
                    <span class="finding__type">${escapeHtml(type)}</span>
                    <span class="finding__severity finding__severity--${severity}">${severity}</span>
                </div>
                <p class="finding__detail">${escapeHtml(detail)}</p>
            </div>
        `;
    }

    // ─── Risk Gauge Animation ──────────────────────────────────
    function animateGauge(score) {
        const arc = $("#riskArc");
        const numEl = $("#riskNumber");
        const circumference = 327; // 2 * π * 52
        const target = circumference - (score / 100) * circumference;

        // Color based on score
        let color;
        if (score >= 70) color = "var(--sev-critical)";
        else if (score >= 40) color = "var(--accent-orange)";
        else color = "var(--accent-green)";

        arc.style.stroke = color;
        numEl.style.color = color;

        // Animate arc
        arc.style.strokeDashoffset = circumference;
        requestAnimationFrame(() => {
            arc.style.strokeDashoffset = target;
        });

        // Animate number
        animateNumber(numEl, 0, score, 1200);
    }

    function animateNumber(el, from, to, duration) {
        const start = performance.now();
        const diff = to - from;

        function step(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            el.textContent = Math.round(from + diff * eased);
            if (progress < 1) requestAnimationFrame(step);
        }

        requestAnimationFrame(step);
    }

    // ─── Utilities ─────────────────────────────────────────────
    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    function shakeElement(el) {
        el.style.animation = "none";
        el.offsetHeight; // trigger reflow
        el.style.animation = "shake 0.5s ease";
    }

    function clearForm() {
        $("#subject").value = "";
        $("#body").value = "";
        $("#sender").value = "";
        $("#hdrFrom").value = "";
        $("#hdrReplyTo").value = "";
        $("#hdrAuth").value = "";
        $("#hdrXMailer").value = "";
        resultsSection.style.display = "none";
    }

    // ─── Event Listeners ───────────────────────────────────────
    btnAnalyze.addEventListener("click", analyzeEmail);
    btnClear.addEventListener("click", clearForm);

    // Ctrl+Enter to analyze
    document.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            analyzeEmail();
        }
    });

    // ─── Init ──────────────────────────────────────────────────
    checkHealth();
    loadModelInfo();
})();
