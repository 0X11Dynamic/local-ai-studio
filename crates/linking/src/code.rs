//! The one-time linking code shown on the helper.
//!
//! - 6 random digits, valid for [`CODE_TTL`] (10 minutes).
//! - Single use: a successful link consumes it.
//! - One attempt at a time; every attempt that does not succeed (wrong code,
//!   or a connection that gives up half way) counts as a failure, and after
//!   [`MAX_FAILURES`] the code is locked. With SPAKE2 each attempt is exactly
//!   one guess, so an attacker gets at most 5 guesses in 1,000,000.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::LinkError;

pub const CODE_TTL: Duration = Duration::from_secs(10 * 60);
pub const MAX_FAILURES: u32 = 5;
pub const CODE_DIGITS: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CodeState {
    Open,
    Used,
    Locked,
    Expired,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodeStatus {
    /// Digits only, e.g. "482913" (shown as "482 913").
    pub code: String,
    pub state: CodeState,
    pub seconds_left: u64,
    pub failures: u32,
}

struct Active {
    code: String,
    expires: Instant,
    failures: u32,
    in_flight: bool,
    used: bool,
}

impl Active {
    fn state(&self) -> CodeState {
        if self.used {
            CodeState::Used
        } else if self.failures >= MAX_FAILURES {
            CodeState::Locked
        } else if Instant::now() >= self.expires {
            CodeState::Expired
        } else {
            CodeState::Open
        }
    }
}

#[derive(Default)]
pub struct CodeSlot {
    inner: Mutex<Option<Active>>,
}

impl CodeSlot {
    pub fn new() -> Self {
        Self::default()
    }

    /// Make a new code (replacing any old one) valid for [`CODE_TTL`].
    pub fn new_code(&self) -> CodeStatus {
        self.new_code_with_ttl(CODE_TTL)
    }

    pub fn new_code_with_ttl(&self, ttl: Duration) -> CodeStatus {
        let code = random_code();
        *self.inner.lock().unwrap() = Some(Active { code, expires: Instant::now() + ttl, failures: 0, in_flight: false, used: false });
        self.status().expect("just set")
    }

    pub fn cancel(&self) {
        *self.inner.lock().unwrap() = None;
    }

    pub fn status(&self) -> Option<CodeStatus> {
        let g = self.inner.lock().unwrap();
        g.as_ref().map(|a| CodeStatus {
            code: a.code.clone(),
            state: a.state(),
            seconds_left: a.expires.saturating_duration_since(Instant::now()).as_secs(),
            failures: a.failures,
        })
    }

    /// True while a code can still be used.
    pub fn is_open(&self) -> bool {
        self.status().is_some_and(|s| s.state == CodeState::Open)
    }

    /// Start a pairing attempt against the current code.
    pub fn begin(&self) -> Result<Attempt<'_>, LinkError> {
        let mut g = self.inner.lock().unwrap();
        let a = g.as_mut().ok_or(LinkError::NoCode)?;
        match a.state() {
            CodeState::Open => {}
            CodeState::Locked => return Err(LinkError::CodeLocked),
            CodeState::Used | CodeState::Expired => return Err(LinkError::NoCode),
        }
        if a.in_flight {
            return Err(LinkError::Busy);
        }
        a.in_flight = true;
        Ok(Attempt { slot: self, code: a.code.clone(), done: false })
    }

    fn finish(&self, code: &str, success: bool) -> Option<LinkError> {
        let mut g = self.inner.lock().unwrap();
        // The code may have been replaced or cancelled meanwhile.
        let a = g.as_mut().filter(|a| a.code == code)?;
        a.in_flight = false;
        if success {
            a.used = true;
            return None;
        }
        a.failures += 1;
        Some(if a.failures >= MAX_FAILURES { LinkError::CodeLocked } else { LinkError::WrongCode { remaining: MAX_FAILURES - a.failures } })
    }
}

/// One pairing attempt. Dropping it without [`Attempt::succeed`] counts as a failure.
pub struct Attempt<'a> {
    slot: &'a CodeSlot,
    code: String,
    done: bool,
}

impl Attempt<'_> {
    pub fn code(&self) -> &str {
        &self.code
    }

    pub fn succeed(mut self) {
        self.done = true;
        self.slot.finish(&self.code, true);
    }

    /// Record a failure; returns the error to report (wrong code or locked).
    pub fn fail(mut self) -> LinkError {
        self.done = true;
        self.slot.finish(&self.code, false).unwrap_or(LinkError::NoCode)
    }
}

impl Drop for Attempt<'_> {
    fn drop(&mut self) {
        if !self.done {
            self.slot.finish(&self.code, false);
        }
    }
}

/// Keep the digits of what the user typed ("482 913", "482-913").
pub fn normalize_code(input: &str) -> String {
    input.chars().filter(|c| c.is_ascii_digit()).collect()
}

fn random_code() -> String {
    // Rejection sampling for a uniform 0..1_000_000.
    const LIMIT: u32 = u32::MAX - (u32::MAX % 1_000_000);
    loop {
        let mut b = [0u8; 4];
        getrandom::fill(&mut b).expect("OS random number generator");
        let n = u32::from_le_bytes(b);
        if n < LIMIT {
            return format!("{:06}", n % 1_000_000);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_six_digits_and_vary() {
        let s = CodeSlot::new();
        let a = s.new_code().code;
        let b = s.new_code().code;
        assert_eq!(a.len(), CODE_DIGITS);
        assert!(a.chars().all(|c| c.is_ascii_digit()));
        // 1 in a million to collide; two more draws make a flake impossible in practice.
        assert!(a != b || a != s.new_code().code);
        assert_eq!(normalize_code(" 482-913 "), "482913");
    }

    #[test]
    fn five_failures_lock_the_code_and_drop_counts_as_failure() {
        let s = CodeSlot::new();
        s.new_code();
        for i in 1..MAX_FAILURES {
            let e = s.begin().unwrap().fail();
            assert!(matches!(e, LinkError::WrongCode { remaining } if remaining == MAX_FAILURES - i), "{e:?}");
        }
        // An abandoned attempt is a failure too: the 5th.
        drop(s.begin().unwrap());
        assert!(matches!(s.begin(), Err(LinkError::CodeLocked)));
        assert_eq!(s.status().unwrap().state, CodeState::Locked);
    }

    #[test]
    fn one_attempt_at_a_time_single_use_and_expiry() {
        let s = CodeSlot::new();
        assert!(matches!(s.begin(), Err(LinkError::NoCode)));
        s.new_code();
        let a = s.begin().unwrap();
        assert!(matches!(s.begin(), Err(LinkError::Busy)));
        a.succeed();
        assert!(matches!(s.begin(), Err(LinkError::NoCode)), "a used code can't be used again");
        s.new_code_with_ttl(Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(5));
        assert!(matches!(s.begin(), Err(LinkError::NoCode)));
        assert_eq!(s.status().unwrap().state, CodeState::Expired);
        s.cancel();
        assert!(s.status().is_none());
    }
}
