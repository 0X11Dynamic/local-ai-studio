//! Best-effort secret redaction for logs, audit records and UI previews.
//! This is a safety net, not the primary defence: secrets should never reach
//! log calls in the first place.

use std::sync::LazyLock;

use regex::Regex;

static PATTERNS: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    vec![
        // Authorization headers / bearer tokens
        (Regex::new(r"(?i)(authorization\s*[:=]\s*)(bearer\s+)?[A-Za-z0-9._~+/=-]{8,}").unwrap(), "${1}${2}[REDACTED]"),
        (Regex::new(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}").unwrap(), "Bearer [REDACTED]"),
        // Common provider token shapes
        (Regex::new(r"\bhf_[A-Za-z0-9]{20,}").unwrap(), "hf_[REDACTED]"),
        (Regex::new(r"\bsk-[A-Za-z0-9_-]{16,}").unwrap(), "sk-[REDACTED]"),
        (Regex::new(r"\bgh[pousr]_[A-Za-z0-9]{20,}").unwrap(), "gh_[REDACTED]"),
        (Regex::new(r"\bAKIA[0-9A-Z]{16}\b").unwrap(), "AKIA[REDACTED]"),
        // key=value / "key": "value" for secret-looking keys
        (
            Regex::new(r#"(?i)("?(?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key|client[_-]?secret|passphrase)"?\s*[:=]\s*"?)[^"\s,;}&]{3,}"#).unwrap(),
            "${1}[REDACTED]",
        ),
        // PEM private keys
        (
            Regex::new(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----").unwrap(),
            "[REDACTED PRIVATE KEY]",
        ),
    ]
});

/// Replace secret-looking substrings with `[REDACTED]`.
pub fn redact(input: &str) -> String {
    let mut out = input.to_string();
    for (re, rep) in PATTERNS.iter() {
        if re.is_match(&out) {
            out = re.replace_all(&out, *rep).into_owned();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::redact;

    #[test]
    fn redacts_common_secret_shapes() {
        let s = redact("Authorization: Bearer abcdefghijklmnop123 and hf_ABCDEFGHIJKLMNOPQRSTUV");
        assert!(!s.contains("abcdefghijklmnop123"), "{s}");
        assert!(!s.contains("ABCDEFGHIJKLMNOPQRSTUV"), "{s}");
        let j = redact(r#"{"api_key": "12345678", "name": "ok"}"#);
        assert!(!j.contains("12345678") && j.contains("\"name\": \"ok\""), "{j}");
        let p = redact("password=hunter22 user=bob");
        assert!(!p.contains("hunter22") && p.contains("user=bob"), "{p}");
    }

    #[test]
    fn leaves_plain_text_alone() {
        let t = "Get-ChildItem -LiteralPath 'C:\\Users' | Select-Object Name";
        assert_eq!(redact(t), t);
    }
}
