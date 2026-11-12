//! Streaming chat against the loaded llama-server (OpenAI-compatible SSE).
//!
//! The server applies the model's own chat template (`--jinja`). This module
//! sends the messages, parses the SSE stream into typed deltas (content and
//! reasoning are separate channels), and returns real statistics: server
//! timings (prompt and generation speed), token usage, time to first token
//! measured here, and the stop reason. Cancelling drops the HTTP stream, which
//! makes llama-server stop generating.

use std::time::{Duration, Instant};

use app_core::{AppError, AppResult};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::client::LlamaClient;

/// Generation-time settings. None = server/model default.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SamplingParams {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub max_tokens: Option<u32>,
    pub seed: Option<i64>,
    pub stop: Vec<String>,
    /// Sent as `chat_template_kwargs` (thinking on/off, reasoning effort).
    /// Decided per request from the loaded model's capabilities, so never saved
    /// with a chat's or task's settings.
    #[serde(skip)]
    pub template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
}

impl SamplingParams {
    pub fn validate(&self) -> Result<(), String> {
        if let Some(t) = self.temperature
            && !(0.0..=5.0).contains(&t)
        {
            return Err("temperature must be between 0 and 5".into());
        }
        if let Some(p) = self.top_p
            && !(0.0..=1.0).contains(&p)
        {
            return Err("top-p must be between 0 and 1".into());
        }
        if let Some(p) = self.min_p
            && !(0.0..=1.0).contains(&p)
        {
            return Err("min-p must be between 0 and 1".into());
        }
        if let Some(r) = self.repeat_penalty
            && !(0.0..=3.0).contains(&r)
        {
            return Err("repetition penalty must be between 0 and 3".into());
        }
        if self.max_tokens == Some(0) {
            return Err("max tokens must be at least 1".into());
        }
        if self.stop.len() > 8 || self.stop.iter().any(|s| s.is_empty() || s.len() > 64) {
            return Err("up to 8 stop sequences of 1-64 characters".into());
        }
        Ok(())
    }

    fn apply(&self, body: &mut serde_json::Map<String, serde_json::Value>) {
        use serde_json::json;
        if let Some(v) = self.temperature {
            body.insert("temperature".into(), json!(v));
        }
        if let Some(v) = self.top_p {
            body.insert("top_p".into(), json!(v));
        }
        if let Some(v) = self.top_k {
            body.insert("top_k".into(), json!(v));
        }
        if let Some(v) = self.min_p {
            body.insert("min_p".into(), json!(v));
        }
        if let Some(v) = self.repeat_penalty {
            body.insert("repeat_penalty".into(), json!(v));
        }
        if let Some(v) = self.max_tokens {
            body.insert("max_tokens".into(), json!(v));
        }
        if let Some(v) = self.seed {
            body.insert("seed".into(), json!(v));
        }
        if !self.stop.is_empty() {
            body.insert("stop".into(), json!(self.stop));
        }
        if let Some(k) = self.template_kwargs.as_ref().filter(|k| !k.is_empty()) {
            body.insert("chat_template_kwargs".into(), serde_json::Value::Object(k.clone()));
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatDelta {
    Content { text: String },
    Reasoning { text: String },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ChatStats {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    /// Prompt tokens served from the server's prompt cache.
    pub cached_tokens: Option<u64>,
    pub prompt_ms: Option<f64>,
    pub prompt_tokens_per_sec: Option<f64>,
    pub generation_ms: Option<f64>,
    pub tokens_per_sec: Option<f64>,
    /// Measured by the app: request sent -> first content or reasoning token.
    pub time_to_first_token_ms: Option<f64>,
    pub total_ms: f64,
    /// stop | length | cancelled | error
    pub stop_reason: String,
}

/// A tool call requested by the model (OpenAI format, arguments as JSON text).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Default)]
pub struct ChatOutput {
    pub content: String,
    pub reasoning: String,
    pub stats: ChatStats,
    /// Tool calls, in the order the model produced them (finish reason "tool_calls").
    pub tool_calls: Vec<ToolCall>,
}

/// Incremental SSE parser: feed bytes, get complete `data:` payloads.
#[derive(Default)]
pub struct SseParser {
    buf: String,
}

impl SseParser {
    pub fn feed(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buf.push_str(&String::from_utf8_lossy(bytes));
        let mut out = Vec::new();
        loop {
            let sep = self.buf.find("\n\n").map(|i| (i, 2)).or_else(|| self.buf.find("\r\n\r\n").map(|i| (i, 4)));
            let Some((i, n)) = sep else { break };
            let event: String = self.buf.drain(..i + n).collect();
            let data: Vec<&str> = event
                .lines()
                .filter_map(|l| l.strip_prefix("data:"))
                .map(|d| d.strip_prefix(' ').unwrap_or(d))
                .collect();
            if !data.is_empty() {
                out.push(data.join("\n"));
            }
        }
        out
    }
}

/// Stream a chat completion. `on_delta` gets each content/reasoning piece in
/// order. On cancel, returns what was generated so far with stop_reason "cancelled".
pub async fn stream_chat(
    client: &LlamaClient,
    messages: &serde_json::Value,
    params: &SamplingParams,
    cancel: &CancellationToken,
    on_delta: impl FnMut(ChatDelta),
) -> AppResult<ChatOutput> {
    stream_chat_tools(client, messages, None, params, cancel, on_delta).await
}

/// Like [`stream_chat`], offering `tools` (OpenAI function definitions). Tool
/// calls are assembled from the streamed deltas and returned complete; the
/// caller only acts on them after the stream has finished.
pub async fn stream_chat_tools(
    client: &LlamaClient,
    messages: &serde_json::Value,
    tools: Option<&serde_json::Value>,
    params: &SamplingParams,
    cancel: &CancellationToken,
    mut on_delta: impl FnMut(ChatDelta),
) -> AppResult<ChatOutput> {
    params.validate().map_err(AppError::InvalidInput)?;
    let mut body = serde_json::Map::new();
    body.insert("messages".into(), messages.clone());
    if let Some(t) = tools.filter(|t| t.as_array().is_some_and(|a| !a.is_empty())) {
        body.insert("tools".into(), t.clone());
    }
    body.insert("stream".into(), serde_json::json!(true));
    body.insert("stream_options".into(), serde_json::json!({ "include_usage": true }));
    params.apply(&mut body);

    let started = Instant::now();
    let req = client
        .http()
        .post(format!("{}/v1/chat/completions", client.base_url()))
        .json(&serde_json::Value::Object(body))
        .timeout(Duration::from_secs(3600))
        .send();
    let resp = tokio::select! {
        r = req => r.map_err(|e| AppError::Runtime(format!("could not reach the model backend: {e}")))?,
        _ = cancel.cancelled() => return Ok(cancelled(ChatOutput::default(), started)),
    };
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        let msg = serde_json::from_str::<serde_json::Value>(&text)
            .ok()
            .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
            .unwrap_or(text);
        if msg.contains("exceeds the available context") || msg.contains("context size") {
            return Err(AppError::InvalidInput(format!("the conversation no longer fits in the model's context: {msg}")));
        }
        return Err(AppError::Runtime(format!("backend returned HTTP {status}: {msg}")));
    }

    let mut out = ChatOutput::default();
    let mut parser = SseParser::default();
    let mut stream = resp.bytes_stream();
    let mut finish: Option<String> = None;
    // index -> partial call
    let mut calls: std::collections::BTreeMap<u64, ToolCall> = std::collections::BTreeMap::new();
    loop {
        let chunk = tokio::select! {
            c = stream.next() => c,
            _ = cancel.cancelled() => {
                // Dropping the stream closes the connection; llama-server stops.
                drop(stream);
                return Ok(cancelled(out, started));
            }
        };
        let Some(chunk) = chunk else { break };
        let bytes = chunk.map_err(|e| AppError::Runtime(format!("stream interrupted: {e}")))?;
        for data in parser.feed(&bytes) {
            if data.trim() == "[DONE]" {
                continue;
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(&data) else { continue };
            if let Some(err) = v["error"]["message"].as_str() {
                return Err(AppError::Runtime(format!("backend error: {err}")));
            }
            if let Some(choice) = v["choices"].get(0) {
                let d = &choice["delta"];
                if let Some(t) = d["reasoning_content"].as_str().filter(|t| !t.is_empty()) {
                    first_token(&mut out.stats, started);
                    out.reasoning.push_str(t);
                    on_delta(ChatDelta::Reasoning { text: t.to_string() });
                }
                if let Some(t) = d["content"].as_str().filter(|t| !t.is_empty()) {
                    first_token(&mut out.stats, started);
                    out.content.push_str(t);
                    on_delta(ChatDelta::Content { text: t.to_string() });
                }
                if let Some(tcs) = d["tool_calls"].as_array() {
                    first_token(&mut out.stats, started);
                    for (i, tc) in tcs.iter().enumerate() {
                        let idx = tc["index"].as_u64().unwrap_or(i as u64);
                        let e = calls.entry(idx).or_default();
                        if let Some(id) = tc["id"].as_str().filter(|s| !s.is_empty()) {
                            e.id = id.to_string();
                        }
                        if let Some(n) = tc["function"]["name"].as_str() {
                            e.name.push_str(n);
                        }
                        if let Some(a) = tc["function"]["arguments"].as_str() {
                            e.arguments.push_str(a);
                        }
                    }
                }
                if let Some(f) = choice["finish_reason"].as_str() {
                    finish = Some(f.to_string());
                }
            }
            if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
                out.stats.prompt_tokens = u["prompt_tokens"].as_u64();
                out.stats.completion_tokens = u["completion_tokens"].as_u64();
                out.stats.cached_tokens = u["prompt_tokens_details"]["cached_tokens"].as_u64();
            }
            if let Some(t) = v.get("timings").filter(|t| t.is_object()) {
                out.stats.prompt_ms = t["prompt_ms"].as_f64();
                out.stats.prompt_tokens_per_sec = t["prompt_per_second"].as_f64();
                out.stats.generation_ms = t["predicted_ms"].as_f64();
                out.stats.tokens_per_sec = t["predicted_per_second"].as_f64();
            }
        }
    }
    out.stats.total_ms = started.elapsed().as_secs_f64() * 1000.0;
    out.stats.stop_reason = finish.unwrap_or_else(|| "stop".into());
    out.tool_calls = calls
        .into_values()
        .enumerate()
        .map(|(i, mut c)| {
            if c.id.is_empty() {
                c.id = format!("call_{i}");
            }
            c
        })
        .collect();
    Ok(out)
}

fn first_token(stats: &mut ChatStats, started: Instant) {
    if stats.time_to_first_token_ms.is_none() {
        stats.time_to_first_token_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
    }
}

fn cancelled(mut out: ChatOutput, started: Instant) -> ChatOutput {
    out.stats.total_ms = started.elapsed().as_secs_f64() * 1000.0;
    out.stats.stop_reason = "cancelled".into();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_parser_handles_split_chunks_and_crlf() {
        let mut p = SseParser::default();
        assert!(p.feed(b"data: {\"a\":").is_empty());
        let got = p.feed(b"1}\n\ndata: [DONE]\n\n");
        assert_eq!(got, vec!["{\"a\":1}".to_string(), "[DONE]".to_string()]);
        let got = p.feed(b"event: x\r\ndata: hi\r\n\r\n: comment\n\n");
        assert_eq!(got, vec!["hi".to_string()]);
    }

    #[test]
    fn sampling_validation() {
        assert!(SamplingParams::default().validate().is_ok());
        assert!(SamplingParams { temperature: Some(7.0), ..Default::default() }.validate().is_err());
        assert!(SamplingParams { top_p: Some(1.5), ..Default::default() }.validate().is_err());
        assert!(SamplingParams { stop: vec!["".into()], ..Default::default() }.validate().is_err());
        assert!(SamplingParams { max_tokens: Some(0), ..Default::default() }.validate().is_err());
    }

    #[test]
    fn template_kwargs_are_sent_but_never_saved() {
        let mut k = serde_json::Map::new();
        k.insert("enable_thinking".into(), serde_json::json!(false));
        let p = SamplingParams { temperature: Some(0.5), template_kwargs: Some(k), ..Default::default() };
        let mut body = serde_json::Map::new();
        p.apply(&mut body);
        assert_eq!(body["chat_template_kwargs"], serde_json::json!({ "enable_thinking": false }));
        let saved = serde_json::to_value(&p).unwrap();
        assert!(saved.get("template_kwargs").is_none(), "{saved}");
        let back: SamplingParams = serde_json::from_value(saved).unwrap();
        assert!(back.template_kwargs.is_none());
        // Empty or absent: nothing is sent.
        let mut body = serde_json::Map::new();
        SamplingParams { template_kwargs: Some(serde_json::Map::new()), ..Default::default() }.apply(&mut body);
        assert!(!body.contains_key("chat_template_kwargs"));
    }
}
