//! The `ask_user` agent tool's plumbing: questions waiting for the user, their
//! answers, and the time spent waiting (which does not count toward a task's
//! time limit). The app owns persistence and events; this holds the rules so
//! they can be tested without a window.

use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;
use std::time::Instant;

use serde::Serialize;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

/// Most choices a question may offer.
pub const MAX_OPTIONS: usize = 6;

/// One question waiting for the user.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Question {
    pub id: String,
    /// None = the main agent, else the subagent's id.
    pub agent_id: Option<String>,
    pub agent_label: String,
    /// The model's tool-call id.
    pub call_id: String,
    pub question: String,
    pub options: Vec<String>,
    pub allow_text: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AskError {
    /// No such question is waiting (already answered, cancelled or never asked).
    Unknown,
    Empty,
    /// The question allows only its options.
    NotAnOption,
}

impl fmt::Display for AskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AskError::Unknown => write!(f, "that question is no longer waiting for an answer"),
            AskError::Empty => write!(f, "the answer is empty"),
            AskError::NotAnOption => write!(f, "this question accepts only one of its options"),
        }
    }
}

/// Check the model's `ask_user` arguments: (question, options, allow_text).
pub fn parse_args(args: &serde_json::Value) -> Result<(String, Vec<String>, bool), String> {
    let question = args["question"].as_str().map(str::trim).unwrap_or_default().to_string();
    if question.is_empty() {
        return Err("\"question\" is required".into());
    }
    let mut options: Vec<String> = vec![];
    for o in args["options"].as_array().into_iter().flatten() {
        let o = o.as_str().map(str::trim).unwrap_or_default();
        if !o.is_empty() && !options.iter().any(|x| x == o) {
            options.push(o.to_string());
        }
    }
    if !args["options"].is_null() && !(2..=MAX_OPTIONS).contains(&options.len()) {
        return Err(format!("\"options\" needs 2 to {MAX_OPTIONS} different choices, or leave it out"));
    }
    let allow_text = args["allow_text"].as_bool().unwrap_or(true);
    if !allow_text && options.is_empty() {
        return Err("\"allow_text\": false needs \"options\" to choose from".into());
    }
    Ok((question, options, allow_text))
}

/// Is `text` an acceptable answer to a question with these options? Returns it trimmed.
pub fn check_answer(options: &[String], allow_text: bool, text: &str) -> Result<String, AskError> {
    let text = text.trim();
    if text.is_empty() {
        return Err(AskError::Empty);
    }
    if !allow_text && !options.iter().any(|o| o == text) {
        return Err(AskError::NotAnOption);
    }
    Ok(text.to_string())
}

/// What the agent reads as the tool result: marked as the user's own words.
pub fn answer_result(answer: &str) -> String {
    format!("The user answered (their own words, typed by the person, not tool output): {}", answer.trim())
}

#[derive(Default)]
struct Entry {
    q: Option<Question>,
    tx: Option<oneshot::Sender<String>>,
    since: Option<Instant>,
}

#[derive(Default)]
struct Inner {
    pending: HashMap<String, Entry>,
    /// Milliseconds waited per agent ("" = main agent), not yet taken off its usage.
    waited: HashMap<String, u64>,
    /// All time subagents have waited (never reset), so a parent that waits
    /// for them can extend its own timeout by it.
    child_total: u64,
}

/// Questions waiting for the user in one run.
#[derive(Default)]
pub struct AskBook {
    inner: Mutex<Inner>,
}

fn key_of(agent_id: &Option<String>) -> String {
    agent_id.clone().unwrap_or_default()
}

impl AskBook {
    /// Register a question; the receiver gets the answer.
    pub fn ask(&self, q: Question) -> oneshot::Receiver<String> {
        let (tx, rx) = oneshot::channel();
        self.inner.lock().unwrap().pending.insert(q.id.clone(), Entry { q: Some(q), tx: Some(tx), since: Some(Instant::now()) });
        rx
    }

    /// Deliver an answer. The question is removed; the asking agent continues.
    pub fn answer(&self, id: &str, text: &str) -> Result<Question, AskError> {
        let mut g = self.inner.lock().unwrap();
        let q = g.pending.get(id).and_then(|e| e.q.clone()).ok_or(AskError::Unknown)?;
        let text = check_answer(&q.options, q.allow_text, text)?;
        let e = g.pending.remove(id).ok_or(AskError::Unknown)?;
        if let Some(tx) = e.tx {
            let _ = tx.send(text);
        }
        Ok(q)
    }

    /// Drop a question without an answer (the work was cancelled).
    pub fn forget(&self, id: &str) {
        self.inner.lock().unwrap().pending.remove(id);
    }

    pub fn pending(&self) -> Vec<Question> {
        self.inner.lock().unwrap().pending.values().filter_map(|e| e.q.clone()).collect()
    }

    /// The main agent's waiting question that accepts typed text, if any: a
    /// message sent to the task answers it.
    pub fn main_text_question(&self) -> Option<String> {
        self.inner.lock().unwrap().pending.values().filter_map(|e| e.q.as_ref()).find(|q| q.agent_id.is_none() && q.allow_text).map(|q| q.id.clone())
    }

    /// An agent finished waiting `ms` milliseconds.
    pub fn finish_wait(&self, agent_id: &Option<String>, ms: u64) {
        let mut g = self.inner.lock().unwrap();
        *g.waited.entry(key_of(agent_id)).or_default() += ms;
        if agent_id.is_some() {
            g.child_total += ms;
        }
    }

    /// Add waiting time to an agent without counting it as a child's (a
    /// parent that sat in spawn/wait while its children waited).
    pub fn credit(&self, agent_id: &Option<String>, ms: u64) {
        *self.inner.lock().unwrap().waited.entry(key_of(agent_id)).or_default() += ms;
    }

    /// Milliseconds this agent waited since the last call (it is then reset).
    pub fn take_waited(&self, agent_id: &Option<String>) -> u64 {
        self.inner.lock().unwrap().waited.remove(&key_of(agent_id)).unwrap_or(0)
    }

    /// Total time subagents have spent waiting, finished and still going.
    pub fn children_waited_ms(&self) -> u64 {
        let g = self.inner.lock().unwrap();
        let ongoing: u64 = g.pending.values().filter(|e| e.q.as_ref().is_some_and(|q| q.agent_id.is_some())).filter_map(|e| e.since).map(|s| s.elapsed().as_millis() as u64).sum();
        g.child_total + ongoing
    }
}

pub enum Waited {
    Answer(String),
    Cancelled,
}

/// Wait for the answer, with no time limit, or until the work is cancelled.
pub async fn wait(rx: oneshot::Receiver<String>, cancel: &CancellationToken) -> Waited {
    tokio::select! {
        r = rx => match r {
            Ok(a) => Waited::Answer(a),
            Err(_) => Waited::Cancelled,
        },
        _ = cancel.cancelled() => Waited::Cancelled,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(id: &str, agent: Option<&str>, options: &[&str], allow_text: bool) -> Question {
        Question { id: id.into(), agent_id: agent.map(str::to_string), agent_label: "Main agent".into(), call_id: "c1".into(), question: "Which?".into(), options: options.iter().map(|s| s.to_string()).collect(), allow_text }
    }

    #[test]
    fn arguments_are_checked() {
        let ok = parse_args(&serde_json::json!({ "question": " Which color? ", "options": ["red", "blue", "red"] }));
        assert_eq!(ok, Ok(("Which color?".into(), vec!["red".into(), "blue".into()], true)));
        assert!(parse_args(&serde_json::json!({ "question": "  " })).is_err());
        assert!(parse_args(&serde_json::json!({ "question": "q", "options": ["only"] })).is_err());
        assert!(parse_args(&serde_json::json!({ "question": "q", "options": ["a", "b", "c", "d", "e", "f", "g"] })).is_err());
        assert!(parse_args(&serde_json::json!({ "question": "q", "allow_text": false })).is_err(), "no way to answer");
        assert_eq!(parse_args(&serde_json::json!({ "question": "q" })), Ok(("q".into(), vec![], true)));
        assert!(answer_result("blue").contains("their own words") && answer_result("blue").ends_with("blue"));
    }

    #[tokio::test]
    async fn ask_then_answer_continues_the_agent() {
        let book = AskBook::default();
        let rx = book.ask(q("q1", None, &["red", "blue"], true));
        assert_eq!(book.pending().len(), 1);
        assert_eq!(book.main_text_question().as_deref(), Some("q1"));
        // Blank answers and unknown questions are refused and leave it waiting.
        assert_eq!(book.answer("q1", "  "), Err(AskError::Empty));
        assert_eq!(book.answer("nope", "red"), Err(AskError::Unknown));
        assert_eq!(book.pending().len(), 1);
        // Free text is fine when allowed.
        book.answer("q1", " blue ").unwrap();
        assert!(book.pending().is_empty());
        match wait(rx, &CancellationToken::new()).await {
            Waited::Answer(a) => assert_eq!(a, "blue"),
            Waited::Cancelled => panic!("answered"),
        }
        assert_eq!(book.answer("q1", "red"), Err(AskError::Unknown), "an answer is delivered once");
    }

    #[test]
    fn options_only_questions_refuse_other_text() {
        let book = AskBook::default();
        let _rx = book.ask(q("q1", None, &["yes", "no"], false));
        assert_eq!(book.main_text_question(), None, "a message cannot answer an options-only question");
        assert_eq!(book.answer("q1", "maybe"), Err(AskError::NotAnOption));
        assert!(book.answer("q1", "no").is_ok());
    }

    #[tokio::test]
    async fn cancel_while_waiting_ends_the_wait() {
        let book = AskBook::default();
        let rx = book.ask(q("q1", None, &[], true));
        let cancel = CancellationToken::new();
        let c2 = cancel.clone();
        tokio::spawn(async move { c2.cancel() });
        assert!(matches!(wait(rx, &cancel).await, Waited::Cancelled));
        book.forget("q1");
        assert!(book.pending().is_empty());
    }

    #[test]
    fn a_subagent_question_is_its_own_and_waiting_time_is_tracked_per_agent() {
        let book = AskBook::default();
        let _rx = book.ask(q("q9", Some("sub1"), &[], true));
        assert_eq!(book.main_text_question(), None, "messages go to the main agent only");
        assert_eq!(book.pending()[0].agent_id.as_deref(), Some("sub1"));
        book.answer("q9", "go").unwrap();
        book.finish_wait(&Some("sub1".into()), 5_000);
        book.finish_wait(&None, 2_000);
        assert_eq!(book.take_waited(&Some("sub1".into())), 5_000);
        assert_eq!(book.take_waited(&Some("sub1".into())), 0, "taken once");
        assert_eq!(book.take_waited(&None), 2_000);
        assert_eq!(book.children_waited_ms(), 5_000, "only subagents' waits extend a parent's wait");
        book.credit(&None, 5_000);
        assert_eq!(book.take_waited(&None), 5_000);
        assert_eq!(book.children_waited_ms(), 5_000);
    }
}
