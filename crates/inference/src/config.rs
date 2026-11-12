//! Load-time settings for llama-server and their mapping to command-line flags.
//! Every field here requires a model reload to change; generation-time settings
//! (temperature, top-p, max tokens, seed, stop) are per request and live in the chat layer.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::capabilities::RuntimeCapabilities;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", content = "count", rename_all = "snake_case")]
pub enum GpuLayers {
    /// Let llama.cpp fit as many layers as memory allows (`-ngl auto`).
    Auto,
    /// Offload every layer (`-ngl all`).
    All,
    /// Exact number of layers (`-ngl N`).
    Count(u32),
    /// No GPU at all (`-ngl 0` and `--device none`).
    CpuOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlashAttention {
    Auto,
    On,
    Off,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadMode {
    Auto,
    Mmap,
    Mlock,
    MmapMlock,
    None,
}

impl LoadMode {
    fn as_flag_value(self) -> &'static str {
        match self {
            LoadMode::Auto => "auto",
            LoadMode::Mmap => "mmap",
            LoadMode::Mlock => "mlock",
            LoadMode::MmapMlock => "mmap+mlock",
            LoadMode::None => "none",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LoadConfig {
    /// Allocated context (`-c`). 0 = Auto: the app picks the largest size that
    /// fits free memory, up to the trained context, before launching (see
    /// `model_registry::estimate::auto_n_ctx`); 0 is never passed to llama-server.
    pub n_ctx: u32,
    pub gpu_layers: GpuLayers,
    /// Generation threads (`-t`). None = llama.cpp default.
    pub threads: Option<u32>,
    /// Prompt-processing threads (`-tb`).
    pub threads_batch: Option<u32>,
    pub batch_size: Option<u32>,
    pub ubatch_size: Option<u32>,
    pub flash_attention: FlashAttention,
    pub cache_type_k: String,
    pub cache_type_v: String,
    /// false = keep the KV cache in system RAM (`--no-kv-offload`).
    pub kv_offload: bool,
    pub rope_scaling: Option<String>,
    pub rope_freq_base: Option<f32>,
    pub rope_freq_scale: Option<f32>,
    pub load_mode: LoadMode,
    /// Server slots (`-np`). V1 default is 1 so the whole context serves one chat.
    pub parallel: u32,
    /// Load the vision projector (mmproj) found next to the model, if any.
    pub load_projector: bool,
    /// Let llama.cpp shrink unset values to fit memory (`--fit`).
    pub fit: bool,
    /// "Share with CPU": the app picks a GPU/CPU split that fits free VRAM at
    /// load time (the user's choice; saved with per-model defaults).
    pub share_with_cpu: bool,
    /// MoE models: keep the expert weights of the first N layers on the CPU
    /// (`--n-cpu-moe N`). Set by the app when it resolves Share with CPU.
    pub n_cpu_moe: Option<u32>,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            n_ctx: 8192,
            gpu_layers: GpuLayers::Auto,
            threads: None,
            threads_batch: None,
            batch_size: None,
            ubatch_size: None,
            flash_attention: FlashAttention::Auto,
            cache_type_k: "f16".into(),
            cache_type_v: "f16".into(),
            kv_offload: true,
            rope_scaling: None,
            rope_freq_base: None,
            rope_freq_scale: None,
            load_mode: LoadMode::Auto,
            parallel: 1,
            load_projector: false,
            fit: true,
            share_with_cpu: false,
            n_cpu_moe: None,
        }
    }
}

/// Result of mapping a config to flags: the argv, plus settings the installed
/// build cannot honour (shown to the user, never silently dropped).
#[derive(Debug, Clone, Serialize)]
pub struct BuiltArgs {
    pub args: Vec<String>,
    pub unsupported: Vec<String>,
}

impl LoadConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.n_ctx != 0 && self.n_ctx < 256 {
            return Err("context length must be 0 (auto) or at least 256".into());
        }
        if self.n_ctx > 4_194_304 {
            return Err("context length above 4M tokens is not supported".into());
        }
        if self.parallel == 0 || self.parallel > 8 {
            return Err("parallel slots must be between 1 and 8".into());
        }
        if let (Some(b), Some(ub)) = (self.batch_size, self.ubatch_size)
            && ub > b
        {
            return Err("micro-batch size cannot exceed batch size".into());
        }
        for t in [self.threads, self.threads_batch].into_iter().flatten() {
            if t == 0 || t > 512 {
                return Err("thread count must be between 1 and 512".into());
            }
        }
        Ok(())
    }

    pub fn to_args(
        &self,
        model: &Path,
        projector: Option<&Path>,
        port: u16,
        caps: &RuntimeCapabilities,
    ) -> BuiltArgs {
        self.to_args_named(model, projector, port, caps, None)
    }

    /// `alias` is the name the server reports as the model id (instead of the
    /// full file path) in its OpenAI-compatible responses.
    pub fn to_args_named(
        &self,
        model: &Path,
        projector: Option<&Path>,
        port: u16,
        caps: &RuntimeCapabilities,
        alias: Option<&str>,
    ) -> BuiltArgs {
        let mut a: Vec<String> = Vec::new();
        let mut unsupported = Vec::new();
        let mut push = |flag: &str, val: Option<String>, label: &str, a: &mut Vec<String>| {
            if caps.has(flag) {
                a.push(flag.into());
                if let Some(v) = val {
                    a.push(v);
                }
            } else {
                unsupported.push(format!("{label} ({flag} not in this llama.cpp build)"));
            }
        };
        a.extend(["--model".into(), model.to_string_lossy().into_owned()]);
        a.extend(["--host".into(), "127.0.0.1".into(), "--port".into(), port.to_string()]);
        if let Some(name) = alias
            && caps.has("--alias")
        {
            a.extend(["--alias".into(), name.to_string()]);
        }
        push("--ctx-size", Some(self.n_ctx.to_string()), "context length", &mut a);
        match self.gpu_layers {
            GpuLayers::Auto => push("--n-gpu-layers", Some("auto".into()), "GPU layers", &mut a),
            GpuLayers::All => push("--n-gpu-layers", Some("all".into()), "GPU layers", &mut a),
            GpuLayers::Count(n) => push("--n-gpu-layers", Some(n.to_string()), "GPU layers", &mut a),
            GpuLayers::CpuOnly => {
                push("--n-gpu-layers", Some("0".into()), "GPU layers", &mut a);
                push("--device", Some("none".into()), "CPU-only device", &mut a);
            }
        }
        if let Some(t) = self.threads {
            push("--threads", Some(t.to_string()), "threads", &mut a);
        }
        if let Some(t) = self.threads_batch {
            push("--threads-batch", Some(t.to_string()), "batch threads", &mut a);
        }
        if let Some(b) = self.batch_size {
            push("--batch-size", Some(b.to_string()), "batch size", &mut a);
        }
        if let Some(b) = self.ubatch_size {
            push("--ubatch-size", Some(b.to_string()), "micro-batch size", &mut a);
        }
        let fa = match self.flash_attention {
            FlashAttention::Auto => "auto",
            FlashAttention::On => "on",
            FlashAttention::Off => "off",
        };
        push("--flash-attn", Some(fa.into()), "Flash Attention", &mut a);
        if self.cache_type_k != "f16" || caps.has("--cache-type-k") {
            push("--cache-type-k", Some(self.cache_type_k.clone()), "K cache type", &mut a);
        }
        if self.cache_type_v != "f16" || caps.has("--cache-type-v") {
            push("--cache-type-v", Some(self.cache_type_v.clone()), "V cache type", &mut a);
        }
        if !self.kv_offload {
            push("--no-kv-offload", None, "KV cache in system RAM", &mut a);
        }
        if let Some(s) = &self.rope_scaling {
            push("--rope-scaling", Some(s.clone()), "RoPE scaling", &mut a);
        }
        if let Some(v) = self.rope_freq_base {
            push("--rope-freq-base", Some(v.to_string()), "RoPE base frequency", &mut a);
        }
        if let Some(v) = self.rope_freq_scale {
            push("--rope-freq-scale", Some(v.to_string()), "RoPE frequency scale", &mut a);
        }
        if self.load_mode != LoadMode::Auto {
            if caps.has("--load-mode") {
                push("--load-mode", Some(self.load_mode.as_flag_value().into()), "load mode", &mut a);
            } else {
                // Older builds: separate mlock / no-mmap switches.
                match self.load_mode {
                    LoadMode::Mlock | LoadMode::MmapMlock => push("--mlock", None, "memory lock", &mut a),
                    LoadMode::None => push("--no-mmap", None, "disable mmap", &mut a),
                    _ => {}
                }
            }
        }
        if let Some(n) = self.n_cpu_moe {
            push("--n-cpu-moe", Some(n.to_string()), "MoE experts on CPU", &mut a);
        }
        push("--parallel", Some(self.parallel.to_string()), "parallel slots", &mut a);
        if caps.has("--fit") {
            a.extend(["--fit".into(), if self.fit { "on" } else { "off" }.into()]);
        }
        match (self.load_projector, projector) {
            (true, Some(p)) => push("--mmproj", Some(p.to_string_lossy().into_owned()), "vision projector", &mut a),
            _ if caps.has("--no-mmproj") => a.push("--no-mmproj".into()),
            _ => {}
        }
        // Native tool calling needs the Jinja chat template engine.
        if caps.has("--jinja") {
            a.push("--jinja".into());
        }
        // The built-in web UI is not part of this app; keep the surface minimal.
        if caps.has("--no-webui") {
            a.push("--no-webui".into());
        }
        if caps.has("--offline") {
            a.push("--offline".into());
        }
        // Verbosity 4 ("trace") is where llama.cpp reports layer offload and
        // per-device buffer sizes, which the app shows as measured values.
        if caps.has("--verbosity") {
            a.extend(["--verbosity".into(), "4".into()]);
        }
        if caps.has("--log-colors") {
            a.extend(["--log-colors".into(), "off".into()]);
        }
        BuiltArgs { args: a, unsupported }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(help: &str) -> RuntimeCapabilities {
        RuntimeCapabilities::parse(help)
    }

    const NEW_HELP: &str = "\
-c,    --ctx-size N                     x
-ngl,  --gpu-layers, --n-gpu-layers N   x
-dev,  --device <dev1,dev2,..>          x
-fa,   --flash-attn [on|off|auto]       x
-ctk,  --cache-type-k TYPE              x
-ctv,  --cache-type-v TYPE              x
-kvo,  --kv-offload, -nkvo, --no-kv-offload
-lm,   --load-mode MODE                 x
-np,   --parallel N                     x
-fit,  --fit [on|off]                   x
-ncmoe, --n-cpu-moe N                   x
--jinja, --no-jinja                     x
--ui,  --webui, --no-ui, --no-webui     x
--offline                               x
--mmproj-auto, --no-mmproj, --no-mmproj-auto
";

    #[test]
    fn maps_settings_to_real_flags() {
        let cfg = LoadConfig {
            n_ctx: 16384,
            gpu_layers: GpuLayers::Count(20),
            kv_offload: false,
            load_mode: LoadMode::Mlock,
            cache_type_k: "q8_0".into(),
            ..Default::default()
        };
        let b = cfg.to_args(Path::new("C:/m/x.gguf"), None, 5555, &caps(NEW_HELP));
        let s = b.args.join(" ");
        assert!(s.contains("--ctx-size 16384"), "{s}");
        assert!(s.contains("--n-gpu-layers 20"), "{s}");
        assert!(s.contains("--no-kv-offload"), "{s}");
        assert!(s.contains("--load-mode mlock"), "{s}");
        assert!(s.contains("--cache-type-k q8_0"), "{s}");
        assert!(s.contains("--port 5555") && s.contains("--host 127.0.0.1"), "{s}");
        assert!(s.contains("--jinja") && s.contains("--no-webui") && s.contains("--no-mmproj"), "{s}");
        assert!(b.unsupported.is_empty(), "{:?}", b.unsupported);
    }

    #[test]
    fn moe_experts_on_cpu_flag_and_old_configs_still_load() {
        let cfg = LoadConfig { n_cpu_moe: Some(12), share_with_cpu: true, ..Default::default() };
        let s = cfg.to_args(Path::new("m.gguf"), None, 1, &caps(NEW_HELP)).args.join(" ");
        assert!(s.contains("--n-cpu-moe 12"), "{s}");
        let old = cfg.to_args(Path::new("m.gguf"), None, 1, &caps("-c, --ctx-size N x\n"));
        assert!(old.unsupported.iter().any(|u| u.contains("--n-cpu-moe")), "{:?}", old.unsupported);
        // A config saved before these fields existed still parses.
        let saved: LoadConfig = serde_json::from_str(r#"{"n_ctx": 4096, "gpu_layers": {"mode": "all"}}"#).unwrap();
        assert!(!saved.share_with_cpu && saved.n_cpu_moe.is_none() && saved.n_ctx == 4096);
    }

    #[test]
    fn reports_settings_the_build_lacks() {
        let cfg = LoadConfig { threads: Some(8), ..Default::default() };
        let b = cfg.to_args(Path::new("m.gguf"), None, 1, &caps(NEW_HELP));
        assert!(b.unsupported.iter().any(|u| u.contains("--threads")), "{:?}", b.unsupported);
    }

    #[test]
    fn cpu_only_disables_devices_and_validation_rejects_nonsense() {
        let cfg = LoadConfig { gpu_layers: GpuLayers::CpuOnly, ..Default::default() };
        let s = cfg.to_args(Path::new("m.gguf"), None, 1, &caps(NEW_HELP)).args.join(" ");
        assert!(s.contains("--n-gpu-layers 0") && s.contains("--device none"), "{s}");
        assert!(LoadConfig { n_ctx: 10, ..Default::default() }.validate().is_err());
        assert!(LoadConfig { batch_size: Some(256), ubatch_size: Some(512), ..Default::default() }.validate().is_err());
        assert!(LoadConfig::default().validate().is_ok());
    }

    #[test]
    fn config_json_round_trips_with_defaults_for_missing_fields() {
        let c: LoadConfig = serde_json::from_str(r#"{"n_ctx": 4096, "gpu_layers": {"mode": "count", "count": 12}}"#).unwrap();
        assert_eq!(c.n_ctx, 4096);
        assert_eq!(c.gpu_layers, GpuLayers::Count(12));
        assert_eq!(c.parallel, 1);
    }
}
