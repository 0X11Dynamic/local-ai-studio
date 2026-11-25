//! Which command-line flags an installed llama-server build supports, read from
//! its own `--help` output. llama.cpp renames flags over time (for example
//! `--mlock`/`--no-mmap` became `--load-mode`), so the app never assumes.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

static FLAG_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)(?:^|[\s,])(--?[a-zA-Z][a-zA-Z0-9-]*)").unwrap());
/// e.g. "--spec-type none,draft-simple,draft-eagle3,draft-mtp,..."
static SPEC_TYPE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)^--spec-type\s+([a-z0-9][a-z0-9,-]*)").unwrap());

/// Bump when `parse` learns something new, so installed runtimes are probed
/// again (see `RuntimeManager::reprobe_outdated`). 1: `spec_types`.
pub const RUNTIME_CAPS_VERSION: u32 = 1;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeCapabilities {
    /// Every flag spelling that appears in the help text, e.g. "--ctx-size", "-c".
    pub flags: BTreeSet<String>,
    /// Allowed KV cache types, from the `--cache-type-k` help line.
    pub cache_types: Vec<String>,
    /// Allowed `--load-mode` values, when the build has that flag.
    pub load_modes: Vec<String>,
    /// Speculative decoding types `--spec-type` accepts (e.g. "draft-mtp").
    #[serde(default)]
    pub spec_types: Vec<String>,
    /// [`RUNTIME_CAPS_VERSION`] of the parser that read the help text; 0 in
    /// manifests written before it existed.
    #[serde(default)]
    pub version: u32,
}

impl RuntimeCapabilities {
    pub fn has(&self, flag: &str) -> bool {
        self.flags.contains(flag)
    }

    pub fn parse(help: &str) -> Self {
        let mut flags = BTreeSet::new();
        for line in help.lines() {
            // Flag spellings appear in the left column (40 chars) before the
            // description. Long flag lists overflow the column and then have no
            // description on that line, so take the whole line in that case.
            let left = match line.get(..40) {
                Some(head) if head.ends_with(' ') => head,
                _ => line,
            };
            if !left.starts_with('-') {
                continue;
            }
            for c in FLAG_RE.captures_iter(left) {
                flags.insert(c[1].trim_end_matches(',').to_string());
            }
        }
        let cache_types = help
            .lines()
            .skip_while(|l| !l.contains("--cache-type-k"))
            .take(4)
            .find_map(|l| l.split("allowed values:").nth(1))
            .map(|s| s.split(',').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect())
            .unwrap_or_else(|| vec!["f16".into(), "q8_0".into(), "q4_0".into()]);
        let mut load_modes = Vec::new();
        if flags.contains("--load-mode") {
            let block: Vec<&str> = help.lines().skip_while(|l| !l.contains("--load-mode")).take(14).collect();
            for l in block.iter().skip(1) {
                let t = l.trim();
                if let Some(rest) = t.strip_prefix("- ")
                    && let Some((mode, _)) = rest.split_once(':')
                {
                    load_modes.push(mode.trim().to_string());
                }
                if t.starts_with('(') {
                    break;
                }
            }
        }
        let spec_types = SPEC_TYPE_RE
            .captures(help)
            .map(|c| c[1].split(',').map(str::to_string).filter(|t| !t.is_empty()).collect())
            .unwrap_or_default();
        Self { flags, cache_types, load_modes, spec_types, version: RUNTIME_CAPS_VERSION }
    }

    /// The build can run speculative decoding of this type (`--spec-type`).
    pub fn has_spec_type(&self, t: &str) -> bool {
        self.has("--spec-type") && self.spec_types.iter().any(|s| s == t)
    }

    /// Run `<exe> --help` and parse it.
    pub async fn probe(server_exe: &Path) -> std::io::Result<Self> {
        crate::vcrt::ensure_next_to(server_exe);
        let mut cmd = tokio::process::Command::new(server_exe);
        cmd.arg("--help").stdin(std::process::Stdio::null()).kill_on_drop(true);
        if let Some(dir) = server_exe.parent() {
            cmd.current_dir(dir);
        }
        #[cfg(windows)]
        cmd.creation_flags(crate::job::CREATE_NO_WINDOW);
        let out = tokio::time::timeout(std::time::Duration::from_secs(30), cmd.output())
            .await
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::TimedOut, "llama-server --help timed out"))??;
        let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
        text.push_str(&String::from_utf8_lossy(&out.stderr));
        let caps = Self::parse(&text);
        if !caps.has("--ctx-size") {
            return Err(std::io::Error::other(format!(
                "{} does not look like llama-server (no --ctx-size in --help output)",
                server_exe.display()
            )));
        }
        Ok(caps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
-c,    --ctx-size N                     size of the prompt context (default: 0, 0 = loaded from model)
-fa,   --flash-attn [on|off|auto]       set Flash Attention use ('on', 'off', or 'auto', default: 'auto')
-ctk,  --cache-type-k TYPE              KV cache data type for K
                                        allowed values: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1
                                        (default: f16)
-kvo,  --kv-offload, -nkvo, --no-kv-offload
                                        whether to enable KV cache offloading (default: enabled)
-lm,   --load-mode MODE                 model loading mode (default: auto)
                                        - auto: mmap, unless a device does not support it
                                        - none: no special loading mode
                                        - mmap: memory-map model
                                        - mlock: force system to keep model in RAM
                                        - mmap+mlock: mmap + force system to keep model in RAM
                                        - dio: use DirectIO if available

                                        (env: LLAMA_ARG_LOAD_MODE)
-ngl,  --gpu-layers, --n-gpu-layers N   max. number of layers to store in VRAM, either an exact number,
--spec-type none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,ngram-simple,ngram-map-k,ngram-map-k4v,ngram-mod,ngram-cache
                                        comma-separated list of types of speculative decoding to use (default:
                                        none)
-fitt, --fit-target MiB0,MiB1,MiB2,...
                                        target margin per device for --fit, comma-separated list of values,
-fitc, --fit-ctx N                      minimum ctx size that can be set by --fit option, default: 4096
";

    #[test]
    fn parses_flags_cache_types_and_load_modes() {
        let c = RuntimeCapabilities::parse(SAMPLE);
        for f in ["-c", "--ctx-size", "-fa", "--flash-attn", "-nkvo", "--no-kv-offload", "--load-mode", "-ngl", "--n-gpu-layers"] {
            assert!(c.has(f), "missing {f}: {:?}", c.flags);
        }
        assert!(!c.has("--mlock"));
        assert!(c.cache_types.contains(&"q8_0".to_string()) && c.cache_types.len() == 9);
        assert_eq!(c.load_modes, vec!["auto", "none", "mmap", "mlock", "mmap+mlock", "dio"]);
    }

    #[test]
    fn parses_spec_types_and_fit_flags() {
        let c = RuntimeCapabilities::parse(SAMPLE);
        assert!(c.has("--spec-type") && c.has("--fit-target") && c.has("-fitt") && c.has("--fit-ctx"), "{:?}", c.flags);
        assert_eq!(c.spec_types.len(), 11, "{:?}", c.spec_types);
        assert!(c.has_spec_type("draft-mtp") && !c.has_spec_type("draft-nope"));
        assert_eq!(c.version, RUNTIME_CAPS_VERSION);
        // A build without the flag, and a manifest saved before these fields.
        let old = RuntimeCapabilities::parse("-c, --ctx-size N x\n");
        assert!(old.spec_types.is_empty() && !old.has_spec_type("draft-mtp"));
        let saved: RuntimeCapabilities = serde_json::from_str(r#"{"flags": ["--spec-type"], "cache_types": [], "load_modes": []}"#).unwrap();
        assert_eq!(saved.version, 0);
        assert!(!saved.has_spec_type("draft-mtp"));
    }
}
