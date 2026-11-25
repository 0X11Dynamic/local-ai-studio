//! The generation gate: how many replies the model backend generates at once
//! (its parallel slots), with a sticky hold for agents.
//!
//! With one slot, agents that alternate per model step make the backend throw
//! away and re-read the other agent's prompt on every switch (the slot's cache
//! holds one prompt). So an agent keeps its [`Seat`]'s permit between
//! consecutive model steps while it only runs fast local tools, and gives it
//! up when it waits on something slow (a tool call longer than
//! [`SLOW_TOOL_AFTER`], or one that is slow by nature), ends its turn with a
//! final answer or stop, or after [`MAX_CONSECUTIVE_STEPS`] steps /
//! [`MAX_HOLD`] when another agent is waiting. Waiters queue in FIFO order, so
//! nobody starves.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;

/// A tool call running longer than this lets go of the gate.
pub const SLOW_TOOL_AFTER: Duration = Duration::from_secs(2);
/// Most consecutive model steps one agent keeps the gate for while another waits.
pub const MAX_CONSECUTIVE_STEPS: u32 = 8;
/// Longest one agent keeps the gate across steps while another waits.
pub const MAX_HOLD: Duration = Duration::from_secs(45);

/// Tools that wait on something external: the gate is released before they run.
/// `shell_execute` is not listed: most commands are fast, and releasing for
/// them made two agents alternate on one slot, so the backend (SWA models
/// cannot reuse a cache past the other agent's prompt) re-read both prompts
/// every step. A command still running after `SLOW_TOOL_AFTER` lets go.
pub fn is_slow_tool(name: &str) -> bool {
    matches!(name, "web_fetch" | "web_search" | "ask_user" | "wait_subagents" | "spawn_subagents")
}

struct Inner {
    sem: Arc<Semaphore>,
    waiting: AtomicUsize,
    /// Permits handed out so far (a switch between agents costs one).
    acquires: AtomicUsize,
}

/// Cheap to clone; all clones share the same permits.
#[derive(Clone)]
pub struct Gate(Arc<Inner>);

impl Gate {
    pub fn new(slots: usize) -> Self {
        Gate(Arc::new(Inner { sem: Arc::new(Semaphore::new(slots)), waiting: AtomicUsize::new(0), acquires: AtomicUsize::new(0) }))
    }

    /// One agent's place at the gate.
    pub fn seat(&self) -> Seat {
        Seat { gate: self.clone(), held: Mutex::new(None), steps: AtomicU32::new(0) }
    }

    /// Agents currently waiting for a permit.
    pub fn waiting(&self) -> usize {
        self.0.waiting.load(Ordering::SeqCst)
    }

    /// How many times a permit was handed out (tests count hand-offs with it).
    pub fn acquires(&self) -> usize {
        self.0.acquires.load(Ordering::SeqCst)
    }

    pub fn available(&self) -> usize {
        self.0.sem.available_permits()
    }
}

struct Held {
    _permit: OwnedSemaphorePermit,
    since: Instant,
}

/// Why waiting for the gate ended without a permit.
#[derive(Debug, PartialEq, Eq)]
pub enum GateStop {
    Cancelled,
    TimeLimit,
}

pub struct Seat {
    gate: Gate,
    held: Mutex<Option<Held>>,
    /// Consecutive steps run on the current hold.
    steps: AtomicU32,
}

impl Seat {
    pub fn gate(&self) -> &Gate {
        &self.gate
    }

    pub fn is_holding(&self) -> bool {
        self.held.lock().unwrap().is_some()
    }

    /// Let go of the permit (a no-op when not holding).
    pub fn release(&self) {
        self.held.lock().unwrap().take();
        self.steps.store(0, Ordering::SeqCst);
    }

    /// Hold the permit for the next model step: keep the one we have, unless
    /// others wait and the fairness cap is reached, else queue for one.
    pub async fn acquire(&self, cancel: &CancellationToken, time_left: Duration) -> Result<(), GateStop> {
        let over_cap = {
            let h = self.held.lock().unwrap();
            h.as_ref().is_some_and(|h| self.gate.waiting() > 0 && (self.steps.load(Ordering::SeqCst) >= MAX_CONSECUTIVE_STEPS || h.since.elapsed() >= MAX_HOLD))
        };
        if over_cap {
            self.release();
        }
        if self.is_holding() {
            self.steps.fetch_add(1, Ordering::SeqCst);
            return Ok(());
        }
        struct Waiting<'a>(&'a AtomicUsize);
        impl Drop for Waiting<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::SeqCst);
            }
        }
        self.gate.0.waiting.fetch_add(1, Ordering::SeqCst);
        let _w = Waiting(&self.gate.0.waiting);
        let permit = tokio::select! {
            p = self.gate.0.sem.clone().acquire_owned() => p.map_err(|_| GateStop::Cancelled)?,
            _ = cancel.cancelled() => return Err(GateStop::Cancelled),
            _ = tokio::time::sleep(time_left) => return Err(GateStop::TimeLimit),
        };
        self.gate.0.acquires.fetch_add(1, Ordering::SeqCst);
        *self.held.lock().unwrap() = Some(Held { _permit: permit, since: Instant::now() });
        self.steps.store(1, Ordering::SeqCst);
        Ok(())
    }

    /// Run a tool call without keeping the gate through a slow one.
    pub async fn during_tool<T>(&self, tool: &str, fut: impl std::future::Future<Output = T>) -> T {
        if is_slow_tool(tool) {
            self.release();
            return fut.await;
        }
        tokio::pin!(fut);
        tokio::select! {
            r = &mut fut => r,
            _ = tokio::time::sleep(SLOW_TOOL_AFTER) => {
                self.release();
                fut.await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NEVER: Duration = Duration::from_secs(3600);

    /// Two agents run `steps` steps each, with `tool` between steps.
    /// Returns the gate hand-offs.
    async fn two_agents(steps: u32, tool: &'static str, tool_ms: u64) -> usize {
        let gate = Gate::new(1);
        let mut hs = vec![];
        for _ in 0..2 {
            let seat = gate.seat();
            hs.push(tokio::spawn(async move {
                let cancel = CancellationToken::new();
                for _ in 0..steps {
                    seat.acquire(&cancel, NEVER).await.unwrap();
                    tokio::time::sleep(Duration::from_millis(5)).await; // generating
                    seat.during_tool(tool, tokio::time::sleep(Duration::from_millis(tool_ms))).await;
                }
                seat.release();
            }));
        }
        for h in hs {
            h.await.unwrap();
        }
        gate.acquires()
    }

    #[tokio::test]
    async fn fast_tools_keep_the_gate_so_agents_alternate_less() {
        // 2 agents x 6 steps: handing off every step would be 12 acquires.
        let n = two_agents(6, "fs_read", 1).await;
        assert!(n <= 4, "{n} hand-offs");
        // A slow tool gives the gate up every step: it alternates freely.
        let slow = two_agents(6, "web_fetch", 1).await;
        assert!(slow > n, "slow {slow} vs fast {n}");
    }

    #[tokio::test]
    async fn nobody_starves_past_the_cap() {
        let gate = Gate::new(1);
        let (a, b) = (gate.seat(), gate.seat());
        let cancel = CancellationToken::new();
        a.acquire(&cancel, NEVER).await.unwrap();
        let cb = cancel.clone();
        let waiter = tokio::spawn(async move {
            b.acquire(&cb, NEVER).await.unwrap();
            b
        });
        while gate.waiting() == 0 {
            tokio::task::yield_now().await;
        }
        // A keeps going on fast steps, up to the cap, then B must get in.
        // Once A yields it queues behind B (B holds), so A's acquire stalls.
        let mut steps = 1;
        loop {
            match tokio::time::timeout(Duration::from_millis(50), a.acquire(&cancel, NEVER)).await {
                Ok(r) => r.unwrap(),
                Err(_) => break,
            }
            steps += 1;
            assert!(steps <= MAX_CONSECUTIVE_STEPS + 2, "B starved for {steps} steps");
        }
        assert!(steps <= MAX_CONSECUTIVE_STEPS, "A ran {steps} steps");
        assert!(waiter.await.unwrap().is_holding());
    }

    #[tokio::test]
    async fn a_long_tool_call_releases() {
        let gate = Gate::new(1);
        let seat = gate.seat();
        seat.acquire(&CancellationToken::new(), NEVER).await.unwrap();
        assert_eq!(gate.available(), 0);
        let s = &seat;
        let g = gate.clone();
        let (_, during) = tokio::join!(
            s.during_tool("fs_read", tokio::time::sleep(Duration::from_secs(4))),
            async {
                tokio::time::sleep(SLOW_TOOL_AFTER + Duration::from_millis(500)).await;
                g.available()
            }
        );
        assert_eq!(during, 1, "gate free while the tool still runs");
        assert!(!seat.is_holding());
    }

    #[tokio::test]
    async fn stop_while_holding_or_waiting_releases() {
        let gate = Gate::new(1);
        let holder = gate.seat();
        let cancel = CancellationToken::new();
        holder.acquire(&cancel, NEVER).await.unwrap();
        // A waiter that is cancelled gives up and is no longer counted.
        let waiter = gate.seat();
        let c2 = CancellationToken::new();
        let c3 = c2.clone();
        let h = tokio::spawn(async move { waiter.acquire(&c3, NEVER).await });
        while gate.waiting() == 0 {
            tokio::task::yield_now().await;
        }
        c2.cancel();
        assert_eq!(h.await.unwrap(), Err(GateStop::Cancelled));
        assert_eq!(gate.waiting(), 0);
        // The holder stops: releasing (or dropping the seat) frees the gate.
        holder.release();
        assert_eq!(gate.available(), 1);
        let again = gate.seat();
        again.acquire(&cancel, NEVER).await.unwrap();
        drop(again);
        assert_eq!(gate.available(), 1);
    }
}
