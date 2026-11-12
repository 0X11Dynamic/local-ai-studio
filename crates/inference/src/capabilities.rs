//! Which command-line flags an installed llama-server build supports, read from
//! its own `--help` output. llama.cpp renames flags over time (for example
//! `--mlock`/`--no-mmap` became `--load-mode`), so the app never assumes.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

static FLAG_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)(?:^|[\s,])(--?[a-zA-Z][a-zA-Z0-9-]*)").unwrap());

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeCapabilities {
    /// Every flag spelling that appears in the help text, e.g. "--ctx-size", "-c".
    pub flags: BTreeSet<String>,
    /// Allowed KV cache types, from the `--cache-type-k` help line.
    pub cache_types: Vec<String>,
    /// Allowed `--load-mode` values, when the build has that flag.
    pub load_modes: Vec<String>,
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
        Self { flags, cache_types, load_modes }
    }

    /// Run `<exe> --help` and parse it.
    pub async fn probe(server_exe: &Path) -> std::io::Result<Self> {
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
}
