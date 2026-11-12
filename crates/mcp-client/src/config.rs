//! MCP server configuration and `mcpServers` JSON import/export (the format
//! used by Claude Desktop, Cursor and LM Studio).
//!
//! Secret values (env vars and headers that look like credentials, and bearer
//! tokens) are split out into [`Secrets`], which the app stores in Windows
//! Credential Manager. The non-secret [`ServerConfig`] goes in SQLite.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub const SECRET_PLACEHOLDER: &str = "<secret>";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ServerConfig {
    /// "stdio" or "http"
    pub transport: String,
    pub command: String,
    pub args: Vec<String>,
    /// Non-secret environment variables.
    pub env: BTreeMap<String, String>,
    pub cwd: Option<String>,
    pub url: String,
    /// Non-secret HTTP headers.
    pub headers: BTreeMap<String, String>,
    /// Names of env vars whose values live in the credential vault.
    pub secret_env: Vec<String>,
    /// Names of headers whose values live in the credential vault.
    pub secret_headers: Vec<String>,
    /// A bearer token is stored in the vault.
    pub bearer: bool,
    /// Per-call timeout.
    pub timeout_secs: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            transport: "stdio".into(),
            command: String::new(),
            args: vec![],
            env: BTreeMap::new(),
            cwd: None,
            url: String::new(),
            headers: BTreeMap::new(),
            secret_env: vec![],
            secret_headers: vec![],
            bearer: false,
            timeout_secs: 60,
        }
    }
}

/// Secret parts of a server config (stored in the credential vault as JSON).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Secrets {
    pub env: BTreeMap<String, String>,
    pub headers: BTreeMap<String, String>,
    pub bearer: Option<String>,
}

impl Secrets {
    pub fn is_empty(&self) -> bool {
        self.env.is_empty() && self.headers.is_empty() && self.bearer.as_deref().is_none_or(str::is_empty)
    }
}

impl ServerConfig {
    pub fn validate(&self) -> Result<(), String> {
        match self.transport.as_str() {
            "stdio" => {
                if self.command.trim().is_empty() {
                    return Err("a stdio server needs a command".into());
                }
            }
            "http" => {
                let u = self.url.trim();
                if !(u.starts_with("http://") || u.starts_with("https://")) {
                    return Err("an HTTP server needs a URL starting with http:// or https://".into());
                }
                // Remote services must use TLS; plain http only for this PC.
                if u.starts_with("http://")
                    && !(u.starts_with("http://127.0.0.1") || u.starts_with("http://localhost") || u.starts_with("http://[::1]"))
                {
                    return Err("remote MCP servers must use https:// (plain http is only allowed for localhost)".into());
                }
            }
            other => return Err(format!("unknown transport '{other}' (use stdio or http)")),
        }
        if self.timeout_secs == 0 || self.timeout_secs > 3600 {
            return Err("timeout must be 1-3600 seconds".into());
        }
        Ok(())
    }
}

/// Heuristic for values that should not live in plain config.
pub fn looks_secret(name: &str) -> bool {
    let n = name.to_ascii_uppercase();
    ["KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "AUTH", "CREDENTIAL", "COOKIE"].iter().any(|k| n.contains(k))
}

/// One parsed entry of an `mcpServers` document.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedServer {
    pub name: String,
    pub config: ServerConfig,
    pub secrets: Secrets,
    pub disabled: bool,
}

/// Parse `{"mcpServers": {...}}` (or a bare `{name: {...}}` map).
pub fn parse_mcp_servers(json: &str) -> Result<Vec<ImportedServer>, String> {
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| format!("not valid JSON: {e}"))?;
    let map = v
        .get("mcpServers")
        .or_else(|| v.get("servers"))
        .unwrap_or(&v)
        .as_object()
        .ok_or("expected an object with an \"mcpServers\" map")?;
    let mut out = Vec::new();
    for (name, s) in map {
        let o = s.as_object().ok_or(format!("server '{name}' is not an object"))?;
        let mut config = ServerConfig::default();
        let mut secrets = Secrets::default();
        if let Some(url) = o.get("url").or_else(|| o.get("serverUrl")).and_then(|u| u.as_str()) {
            config.transport = "http".into();
            config.url = url.to_string();
        } else {
            config.transport = "stdio".into();
            config.command = o
                .get("command")
                .and_then(|c| c.as_str())
                .ok_or(format!("server '{name}' has neither \"command\" nor \"url\""))?
                .to_string();
            config.args = o
                .get("args")
                .and_then(|a| a.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
                .unwrap_or_default();
            config.cwd = o.get("cwd").and_then(|c| c.as_str()).map(str::to_string);
        }
        if let Some(env) = o.get("env").and_then(|e| e.as_object()) {
            for (k, val) in env {
                let val = val.as_str().map(str::to_string).unwrap_or_else(|| val.to_string());
                if looks_secret(k) {
                    config.secret_env.push(k.clone());
                    if val != SECRET_PLACEHOLDER {
                        secrets.env.insert(k.clone(), val);
                    }
                } else {
                    config.env.insert(k.clone(), val);
                }
            }
        }
        if let Some(h) = o.get("headers").and_then(|e| e.as_object()) {
            for (k, val) in h {
                let val = val.as_str().unwrap_or_default().to_string();
                if k.eq_ignore_ascii_case("authorization") {
                    if let Some(tok) = val.strip_prefix("Bearer ").or_else(|| val.strip_prefix("bearer ")) {
                        config.bearer = true;
                        if tok != SECRET_PLACEHOLDER {
                            secrets.bearer = Some(tok.to_string());
                        }
                        continue;
                    }
                }
                if looks_secret(k) {
                    config.secret_headers.push(k.clone());
                    if val != SECRET_PLACEHOLDER {
                        secrets.headers.insert(k.clone(), val);
                    }
                } else {
                    config.headers.insert(k.clone(), val);
                }
            }
        }
        let disabled = o.get("disabled").and_then(|d| d.as_bool()).unwrap_or(false);
        config.validate().map_err(|e| format!("server '{name}': {e}"))?;
        out.push(ImportedServer { name: name.clone(), config, secrets, disabled });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// Export as an `mcpServers` document. Secret values become placeholders.
pub fn export_mcp_servers(servers: &[(String, ServerConfig, bool)]) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (name, c, enabled) in servers {
        let mut o = serde_json::Map::new();
        if c.transport == "http" {
            o.insert("url".into(), c.url.clone().into());
            let mut h: serde_json::Map<String, serde_json::Value> = c.headers.iter().map(|(k, v)| (k.clone(), v.clone().into())).collect();
            for k in &c.secret_headers {
                h.insert(k.clone(), SECRET_PLACEHOLDER.into());
            }
            if c.bearer {
                h.insert("Authorization".into(), format!("Bearer {SECRET_PLACEHOLDER}").into());
            }
            if !h.is_empty() {
                o.insert("headers".into(), h.into());
            }
        } else {
            o.insert("command".into(), c.command.clone().into());
            o.insert("args".into(), c.args.clone().into());
            if let Some(cwd) = &c.cwd {
                o.insert("cwd".into(), cwd.clone().into());
            }
        }
        let mut env: serde_json::Map<String, serde_json::Value> = c.env.iter().map(|(k, v)| (k.clone(), v.clone().into())).collect();
        for k in &c.secret_env {
            env.insert(k.clone(), SECRET_PLACEHOLDER.into());
        }
        if !env.is_empty() {
            o.insert("env".into(), env.into());
        }
        if !enabled {
            o.insert("disabled".into(), true.into());
        }
        map.insert(name.clone(), o.into());
    }
    serde_json::json!({ "mcpServers": map })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imports_lmstudio_style_config_and_splits_secrets() {
        let doc = r#"{
          "mcpServers": {
            "local-fs": { "command": "python", "args": ["C:\\x\\server.py"],
                          "env": { "MCP_FS_HOME": "C:\\Users\\test", "OPENAI_API_KEY": "sk-abc123" } },
            "remote": { "url": "https://mcp.example.com/mcp",
                        "headers": { "Authorization": "Bearer tok-999", "X-Team": "a" } },
            "off": { "command": "node", "disabled": true }
          }
        }"#;
        let s = parse_mcp_servers(doc).unwrap();
        assert_eq!(s.len(), 3);
        let fs = s.iter().find(|x| x.name == "local-fs").unwrap();
        assert_eq!(fs.config.env.get("MCP_FS_HOME").unwrap(), "C:\\Users\\test");
        assert!(!fs.config.env.contains_key("OPENAI_API_KEY"));
        assert_eq!(fs.secrets.env.get("OPENAI_API_KEY").unwrap(), "sk-abc123");
        let r = s.iter().find(|x| x.name == "remote").unwrap();
        assert!(r.config.bearer && r.secrets.bearer.as_deref() == Some("tok-999"));
        assert_eq!(r.config.headers.get("X-Team").unwrap(), "a");
        assert!(s.iter().find(|x| x.name == "off").unwrap().disabled);

        // Export never contains the secret values; re-import keeps the shape.
        let all: Vec<_> = s.iter().map(|x| (x.name.clone(), x.config.clone(), !x.disabled)).collect();
        let out = export_mcp_servers(&all).to_string();
        assert!(!out.contains("sk-abc123") && !out.contains("tok-999"), "{out}");
        let again = parse_mcp_servers(&out).unwrap();
        assert_eq!(again.iter().find(|x| x.name == "local-fs").unwrap().config, fs.config);
        assert!(again.iter().all(|x| x.secrets.is_empty()), "placeholders are not imported as secrets");
    }

    #[test]
    fn validation() {
        assert!(parse_mcp_servers(r#"{"mcpServers":{"x":{"url":"http://evil.example/mcp"}}}"#).unwrap_err().contains("https"));
        assert!(parse_mcp_servers(r#"{"mcpServers":{"x":{"url":"http://127.0.0.1:3000/mcp"}}}"#).is_ok());
        assert!(parse_mcp_servers(r#"{"mcpServers":{"x":{"args":[]}}}"#).is_err());
        assert!(parse_mcp_servers("not json").is_err());
        assert!(looks_secret("GITHUB_TOKEN") && looks_secret("api_key") && !looks_secret("HOME"));
    }
}
