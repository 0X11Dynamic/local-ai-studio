//! What a model supports, read from its own files rather than guessed:
//! thinking (and how it is controlled), tool calling, and image / audio input.
//!
//! Signals:
//! - the chat template (`tokenizer.chat_template` in the GGUF, or the Hub's
//!   `gguf.chat_template` before download): whether it reads `enable_thinking`
//!   or `reasoning_effort`, what it emits for the generation prompt, and
//!   whether it renders a `tools` list;
//! - the vision projector (mmproj) next to the model: its header says whether it
//!   has a vision and/or audio encoder.
//!
//! A `<think>` marker alone is not a signal: non-thinking models such as
//! Qwen3-4B-Instruct-2507 keep it in the template to strip old reasoning from
//! history. `None` always means "unknown", never a guess.

use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::gguf::{self, GgufHeader, Value};

/// Bump when detection changes so stored rows are re-detected at startup.
/// 2: also back-fills the attention layout (sliding window, shared KV, MTP
/// layers) the memory estimate reads.
pub const CAPS_VERSION: u32 = 2;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingControl {
    /// No template, or one we can't read.
    #[default]
    Unknown,
    /// The template neither starts nor controls reasoning.
    None,
    /// On/off through a chat-template kwarg (`thinking_kwarg`).
    Toggle,
    /// Only an effort level (gpt-oss style `reasoning_effort`).
    Effort,
    /// The generation prompt always opens a `<think>` block; it can't be turned off.
    Always,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Capabilities {
    /// 0 = never detected (rows stored by an older app version).
    pub version: u32,
    /// "gguf" (local header) or "hf" (Hub metadata before download).
    pub source: String,
    pub thinking: Option<bool>,
    pub thinking_control: ThinkingControl,
    /// Kwarg name for the on/off toggle: "enable_thinking" or "thinking".
    pub thinking_kwarg: Option<String>,
    /// Whether thinking is on when the kwarg isn't sent.
    pub thinking_default: Option<bool>,
    /// `reasoning_effort` values the template accepts, lowest first.
    pub effort_levels: Vec<String>,
    pub effort_default: Option<String>,
    pub tools: Option<bool>,
    /// True only when a vision projector is present (attached locally, or in the repo).
    pub vision: Option<bool>,
    pub audio: Option<bool>,
    /// Media kinds the chat template knows how to render: "image", "audio", "video".
    pub template_media: Vec<String>,
    /// The template handles images but no projector file was found.
    pub needs_projector: bool,
}

const EFFORT_ORDER: [&str; 7] = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];

static GEN_PROMPT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\{%-?\s*if\s+add_generation_prompt\b[^%]*-?%\}").unwrap());
static ENDIF_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\{%-?\s*endif").unwrap());
static TOOLS_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\btools\b").unwrap());
static THINKING_VAR_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\bthinking\s+is\s+(?:not\s+)?(?:defined|undefined)").unwrap()
});
static KWARG_DEFAULT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:enable_thinking|\bthinking)\s*\|\s*default\(\s*(true|false|True|False)\s*\)").unwrap()
});
static KWARG_SET_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"set\s+(?:enable_thinking|thinking)\s*=\s*(true|false|True|False)").unwrap());
static DEFAULT_ON_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:enable_thinking|\bthinking)\s+is\s+(?:defined\s+and\s+(?:not\s+(?:enable_thinking|thinking)\b|(?:enable_thinking|thinking)\s+is\s+false)|undefined)")
        .unwrap()
});
static DEFAULT_OFF_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:enable_thinking|\bthinking)\s+is\s+defined\s+and\s+(?:enable_thinking|thinking)\b(?:\s+is\s+true)?\s*(?:%|\)|and|or)").unwrap()
});
static EFFORT_WINDOW_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)reasoning_effort.{0,200}").unwrap());
static QUOTED_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"['"]([a-z]+)['"]"#).unwrap());
static EFFORT_DEFAULT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"reasoning_effort\s*(?:\|\s*default\(\s*|=\s*)['"]([a-z]+)['"]"#).unwrap()
});

/// Detect capabilities from a chat template. `None` (no template) gives all-unknown.
pub fn from_template(template: Option<&str>, source: &str) -> Capabilities {
    let mut c = Capabilities { version: CAPS_VERSION, source: source.to_string(), ..Default::default() };
    let Some(t) = template.filter(|t| !t.trim().is_empty()) else {
        return c;
    };

    // Tool calling: the template renders the `tools` list it is given.
    c.tools = Some(TOOLS_RE.is_match(t));

    let lower = t.to_ascii_lowercase();
    for kind in ["image", "audio", "video"] {
        if lower.contains(kind) {
            c.template_media.push(kind.to_string());
        }
    }

    // Thinking controls.
    let kwarg = if t.contains("enable_thinking") {
        Some("enable_thinking")
    } else if THINKING_VAR_RE.is_match(t) {
        Some("thinking")
    } else {
        None
    };
    let has_effort = t.contains("reasoning_effort");
    if has_effort {
        let mut levels: Vec<&str> = Vec::new();
        for w in EFFORT_WINDOW_RE.find_iter(t) {
            for q in QUOTED_RE.captures_iter(w.as_str()) {
                let v = q.get(1).map(|m| m.as_str()).unwrap_or("");
                if EFFORT_ORDER.contains(&v) && !levels.contains(&v) {
                    levels.push(v);
                }
            }
        }
        if levels.len() < 2 {
            // gpt-oss reads the value without listing the allowed ones (only its
            // default appears); use its documented levels.
            for l in ["low", "medium", "high"] {
                if !levels.contains(&l) {
                    levels.push(l);
                }
            }
        }
        levels.sort_by_key(|l| EFFORT_ORDER.iter().position(|o| o == l));
        c.effort_levels = levels.into_iter().map(str::to_string).collect();
        c.effort_default = EFFORT_DEFAULT_RE.captures(t).map(|m| m[1].to_string());
    }

    if let Some(k) = kwarg {
        c.thinking = Some(true);
        c.thinking_control = ThinkingControl::Toggle;
        c.thinking_kwarg = Some(k.to_string());
        c.thinking_default = if let Some(m) = KWARG_DEFAULT_RE.captures(t).or_else(|| KWARG_SET_RE.captures(t)) {
            Some(m[1].eq_ignore_ascii_case("true"))
        } else if DEFAULT_ON_RE.is_match(t) {
            Some(true)
        } else if DEFAULT_OFF_RE.is_match(t) {
            Some(false)
        } else {
            None
        };
    } else if has_effort {
        c.thinking = Some(true);
        c.thinking_control = ThinkingControl::Effort;
        c.thinking_default = Some(true);
    } else if generation_prompt_opens_think(t) {
        c.thinking = Some(true);
        c.thinking_control = ThinkingControl::Always;
        c.thinking_default = Some(true);
    } else {
        c.thinking = Some(false);
        c.thinking_control = ThinkingControl::None;
    }
    c
}

/// True when the `{% if add_generation_prompt %}` block emits an opening
/// `<think>` (the assistant turn always starts reasoning).
fn generation_prompt_opens_think(t: &str) -> bool {
    GEN_PROMPT_RE.find_iter(t).any(|m| {
        let rest = &t[m.end()..];
        let end = ENDIF_RE.find(rest).map(|e| e.start()).unwrap_or(rest.len());
        let block = &rest[..end];
        block.contains("<think>") && !block.contains("</think>")
    })
}

/// What a projector header says it encodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjectorKinds {
    pub vision: bool,
    pub audio: bool,
}

pub fn projector_kinds(h: &GgufHeader) -> ProjectorKinds {
    let flag = |k: &str| match h.kv.get(k) {
        Some(Value::Bool(b)) => Some(*b),
        Some(v) => v.as_u64().map(|n| n != 0),
        None => None,
    };
    let vision = flag("clip.has_vision_encoder");
    let audio = flag("clip.has_audio_encoder").unwrap_or(false);
    // Older projectors carry no flags: they are vision-only.
    ProjectorKinds { vision: vision.unwrap_or(!audio), audio }
}

pub fn read_projector(path: &Path) -> Option<ProjectorKinds> {
    gguf::read_header(path).ok().map(|h| projector_kinds(&h))
}

/// Apply what is known about the local projector file. `None` = no projector
/// file; `Some(None)` = a projector file that couldn't be read.
pub fn apply_projector(c: &mut Capabilities, projector: Option<Option<ProjectorKinds>>) {
    match projector {
        Some(Some(k)) => {
            c.vision = Some(k.vision);
            c.audio = Some(k.audio);
        }
        Some(None) => {
            c.vision = Some(true);
            c.audio = None;
        }
        None => {
            c.vision = Some(false);
            c.audio = Some(false);
        }
    }
    c.needs_projector = c.vision != Some(true) && c.template_media.iter().any(|m| m == "image");
}

/// Hub repos: projector files tell us vision is possible; their headers aren't
/// readable before download, so audio stays unknown when one exists.
pub fn apply_repo_projector(c: &mut Capabilities, repo_has_projector: bool, tags_say_vision: bool) {
    if repo_has_projector {
        c.vision = Some(true);
        c.audio = if c.template_media.iter().any(|m| m == "audio") { None } else { Some(false) };
    } else {
        c.vision = Some(false);
        c.audio = Some(false);
    }
    c.needs_projector = !repo_has_projector && (tags_say_vision || c.template_media.iter().any(|m| m == "image"));
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN3_INSTRUCT_2507: &str = "{%- if tools %}{{- '<|im_start|>system\\n' }}{%- for tool in tools %}{{- tool | tojson }}{%- endfor %}{%- endif %}\
{%- for message in messages %}{%- if '</think>' in content %}{%- set content = content.split('</think>')[-1] %}{%- endif %}\
{%- if message.tool_calls %}<tool_call>{%- endif %}{%- endfor %}\
{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}";

    const DEEPSEEK_R1: &str = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}\
{%- for message in messages %}{%- for tool in message['tool_calls']%}<｜tool▁calls▁begin｜>{%- endfor %}{%- endfor -%}\
{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n'}}{% endif %}";

    const QWEN3_TOGGLE_EFFORT: &str = "{%- if tools %}{%- for tool in tools %}{%- endfor %}{%- endif %}\
{%- if enable_thinking is undefined or enable_thinking is true %}\n\
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}\n\
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}\n\
        {{- raise_exception('Unexpected reasoning effort') }}\n    {%- endif %}\n{%- endif %}\n\
{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n\
    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n\
    {%- else %}\n        {{- '<think>\\n' }}\n    {%- endif %}\n{%- endif %}";

    const GLM_V: &str = "{%- if tools -%}{% for tool in tools %}{% endfor %}{%- endif -%}\
{%- if item.type == 'image' %}<|begin_of_image|><|image|><|end_of_image|>{%- endif %}\
{%- if add_generation_prompt -%}\n<|assistant|>\n{{'<think></think>\\n' if (enable_thinking is defined and not enable_thinking) else ''}}\n{%- endif -%}";

    const GEMMA4: &str = "{%- set enable_thinking = enable_thinking | default(false) -%}\
{%- if enable_thinking or tools -%}{%- for tool in tools -%}{%- endfor -%}{%- endif -%}\
{%- if item['type'] == 'image' -%}<|image|>{%- elif item['type'] == 'audio' -%}<|audio|>{%- endif -%}\
{%- if add_generation_prompt -%}{{- '<|turn>model\\n' -}}{%- endif -%}";

    const GPT_OSS: &str = "{%- if not reasoning_effort is defined %}{%- set reasoning_effort = \"medium\" %}{%- endif %}\
{{- \"Reasoning: \" + reasoning_effort + \"\\n\\n\" }}{%- if tools -%}{%- endif -%}\
{%- if add_generation_prompt -%}<|start|>assistant{%- endif -%}";

    #[test]
    fn qwen3_instruct_2507_is_tools_without_thinking() {
        let c = from_template(Some(QWEN3_INSTRUCT_2507), "gguf");
        assert_eq!(c.tools, Some(true));
        assert_eq!(c.thinking, Some(false));
        assert_eq!(c.thinking_control, ThinkingControl::None);
        assert!(c.template_media.is_empty());
    }

    #[test]
    fn deepseek_r1_always_thinks_and_has_no_tools() {
        let c = from_template(Some(DEEPSEEK_R1), "gguf");
        assert_eq!(c.thinking_control, ThinkingControl::Always);
        assert_eq!(c.thinking, Some(true));
        assert_eq!(c.tools, Some(false), "message.tool_calls alone is not a tools list");
    }

    #[test]
    fn qwen3_toggle_with_effort_levels() {
        let c = from_template(Some(QWEN3_TOGGLE_EFFORT), "gguf");
        assert_eq!(c.thinking_control, ThinkingControl::Toggle);
        assert_eq!(c.thinking_kwarg.as_deref(), Some("enable_thinking"));
        assert_eq!(c.thinking_default, Some(true));
        assert_eq!(c.effort_levels, vec!["low", "medium", "xhigh"]);
        assert_eq!(c.effort_default.as_deref(), Some("xhigh"));
        assert_eq!(c.tools, Some(true));
    }

    #[test]
    fn glm_toggle_default_on_and_images() {
        let c = from_template(Some(GLM_V), "gguf");
        assert_eq!(c.thinking_control, ThinkingControl::Toggle);
        assert_eq!(c.thinking_default, Some(true));
        assert_eq!(c.template_media, vec!["image"]);
    }

    #[test]
    fn gemma4_toggle_default_off_with_image_and_audio() {
        let mut c = from_template(Some(GEMMA4), "gguf");
        assert_eq!(c.thinking_control, ThinkingControl::Toggle);
        assert_eq!(c.thinking_default, Some(false));
        assert_eq!(c.template_media, vec!["image", "audio"]);
        apply_projector(&mut c, None);
        assert_eq!(c.vision, Some(false));
        assert!(c.needs_projector);
        apply_projector(&mut c, Some(Some(ProjectorKinds { vision: true, audio: true })));
        assert_eq!((c.vision, c.audio, c.needs_projector), (Some(true), Some(true), false));
    }

    #[test]
    fn gpt_oss_effort_only() {
        let c = from_template(Some(GPT_OSS), "gguf");
        assert_eq!(c.thinking_control, ThinkingControl::Effort);
        assert_eq!(c.effort_levels, vec!["low", "medium", "high"]);
        assert_eq!(c.effort_default.as_deref(), Some("medium"));
    }

    #[test]
    fn thinking_kwarg_variant() {
        let t = "{%- if not thinking is defined %}{%- set thinking = false %}{%- endif %}{%- if add_generation_prompt %}{% if thinking %}<think>{% else %}</think>{% endif %}{%- endif %}";
        let c = from_template(Some(t), "gguf");
        assert_eq!(c.thinking_kwarg.as_deref(), Some("thinking"));
        assert_eq!(c.thinking_default, Some(false));
    }

    #[test]
    fn no_template_is_unknown_not_guessed() {
        let c = from_template(None, "gguf");
        assert_eq!(c.version, CAPS_VERSION);
        assert_eq!((c.thinking, c.tools, c.vision), (None, None, None));
        assert_eq!(c.thinking_control, ThinkingControl::Unknown);
    }

    #[test]
    fn repo_projector_rules() {
        let mut c = from_template(Some(GEMMA4), "hf");
        apply_repo_projector(&mut c, true, true);
        assert_eq!((c.vision, c.audio, c.needs_projector), (Some(true), None, false));
        let mut c = from_template(Some(QWEN3_INSTRUCT_2507), "hf");
        apply_repo_projector(&mut c, false, false);
        assert_eq!((c.vision, c.needs_projector), (Some(false), false));
    }

    #[test]
    fn old_rows_without_caps_deserialize_as_version_zero() {
        let mut v = serde_json::to_value(crate::gguf::ModelMetadata::default()).unwrap();
        v.as_object_mut().unwrap().remove("caps");
        let m: crate::gguf::ModelMetadata = serde_json::from_value(v).unwrap();
        assert_eq!(m.caps.version, 0);
    }

    /// Read-only check against real model files: `LAS_REAL_MODELS=<dir> cargo test -p model-registry real_models -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn real_models() {
        let Ok(dir) = std::env::var("LAS_REAL_MODELS") else { return };
        let r = crate::scan::scan(Path::new(&dir));
        for m in r.models {
            let projector = m.projector_path.as_ref().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned());
            println!("{} [projector {projector:?}] => {}", m.display_name, serde_json::to_string(&m.metadata.caps).unwrap());
        }
    }
}
