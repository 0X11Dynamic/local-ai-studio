//! Load progress: how far a model load has got on this PC and on each linked
//! PC, as a throttled event stream (`EngineEvent::LoadProgress`).
//!
//! Local signal (Windows): the backend process's own counters, read while it
//! loads. `ReadTransferCount` counts bytes read with ReadFile (`--no-mmap`),
//! the working-set growth counts pages touched through the memory map (the
//! default); the larger of the two, over the model's size, is the share read
//! so far. It is capped at 99% until llama.cpp logs "offloaded N/M layers",
//! which is the reliable "weights are in place" signal.
//!
//! Linked PC signal: tensor data the tunnel carried to that PC since the
//! load started, plus what its weight cache supplied, both read from the
//! ggml-rpc stream itself (`linking::rpc_tap`), over the PC's real share:
//! the backend's "RPCn model buffer size" line, logged after the layers are
//! assigned and before any weight is sent. Until that line the share is
//! unknown (0%, bytes only). "Cached" is shown only when the helper answered
//! cache lookups with "have it".
//!
//! "Weights in place" is llama.cpp starting to build the context
//! ("llama_context: constructing ..."), which follows the last weight; the
//! "offloaded N/M layers" line comes before any weight is read or sent.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde::Serialize;

/// At most this many progress events per second.
pub const MAX_EVENTS_PER_SEC: u32 = 5;
/// Speeds are measured over this window.
const RATE_WINDOW_SECS: f64 = 5.0;
/// Below this speed there is no meaningful "time left".
const MIN_RATE_FOR_ETA: f64 = 50_000.0;
/// No new bytes for this long = stalled (no speed, no time left).
const STALL_SECS: f64 = 1.5;

/// A linked PC taking part in a load: where its bytes are counted.
#[derive(Debug, Clone)]
pub struct PeerProbe {
    pub id: String,
    pub name: String,
    /// The local tunnel endpoint llama-server uses (`127.0.0.1:port`).
    pub endpoint: String,
    /// Weight bytes this PC is expected to receive (0 = unknown until the
    /// backend logs its buffer size).
    pub expected_bytes: u64,
    /// Live count of tensor data sent to it (cumulative; the load's own
    /// share is measured from the value when the load starts).
    pub sent: Arc<AtomicU64>,
    /// Live count of bytes its weight cache supplied instead (cumulative).
    pub cached: Arc<AtomicU64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadPhase {
    /// Backend started, nothing read yet.
    Starting,
    /// Weights are being read and sent.
    Loading,
    /// Weights are in place; the backend is warming up.
    Finishing,
    Done,
    Failed,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct DeviceProgress {
    /// "local" or the linked PC's id.
    pub id: String,
    pub name: String,
    pub linked: bool,
    pub done_bytes: u64,
    pub total_bytes: u64,
    /// 0..=100.
    pub percent: f64,
    /// Speed of the data really moving (read here, or sent to a linked PC).
    pub bytes_per_sec: Option<f64>,
    pub eta_secs: Option<u64>,
    /// The helper reported cache hits: some weights came from its cache.
    pub cached: bool,
    /// Bytes its cache supplied (part of `done_bytes`).
    pub cached_bytes: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LoadProgress {
    pub phase: LoadPhase,
    /// 0..=100 over all devices.
    pub percent: f64,
    pub eta_secs: Option<u64>,
    pub elapsed_secs: f64,
    pub devices: Vec<DeviceProgress>,
    pub error: Option<String>,
}

impl LoadProgress {
    pub fn failed(error: &str) -> Self {
        Self { phase: LoadPhase::Failed, percent: 0.0, eta_secs: None, elapsed_secs: 0.0, devices: Vec::new(), error: Some(error.to_string()) }
    }
}

pub fn percent_of(done: u64, total: u64) -> f64 {
    if total == 0 { 0.0 } else { (done as f64 / total as f64 * 100.0).clamp(0.0, 100.0) }
}

/// Seconds left at `rate` bytes/s; None when the speed is too low to say.
pub fn eta_secs(remaining: u64, rate: Option<f64>) -> Option<u64> {
    let r = rate?;
    if r < MIN_RATE_FOR_ETA || !r.is_finite() {
        return None;
    }
    Some((remaining as f64 / r).ceil() as u64)
}

/// Bytes per second over the last few seconds.
#[derive(Debug, Default)]
pub struct RateWindow {
    samples: VecDeque<(f64, u64)>,
}

impl RateWindow {
    pub fn push(&mut self, t: f64, bytes: u64) {
        self.samples.push_back((t, bytes));
        while self.samples.len() > 2 && t - self.samples[1].0 >= RATE_WINDOW_SECS {
            self.samples.pop_front();
        }
        // Idle time before the data started moving is not part of its speed
        // (VM test: the linked PC's row waited 10 s for its share, then
        // showed 1-2 MB/s and a 400-800 s "time left" while 100 MB/s flowed).
        while self.samples.len() > 2 && self.samples[0].1 == self.samples[1].1 {
            self.samples.pop_front();
        }
    }

    /// Bytes per second; None until 0.5 s of data is in, and while nothing
    /// moved for [`STALL_SECS`] (a stall has no meaningful speed or time left).
    pub fn rate(&self) -> Option<f64> {
        let (t0, b0) = *self.samples.front()?;
        let (t1, b1) = *self.samples.back()?;
        // When the count last changed: the first sample of the trailing run
        // of equal counts.
        let last_change = self.samples.iter().rev().take_while(|s| s.1 == b1).last().map_or(t1, |s| s.0);
        if t1 - last_change >= STALL_SECS {
            return None;
        }
        (t1 - t0 >= 0.5).then(|| b1.saturating_sub(b0) as f64 / (t1 - t0))
    }
}

/// Lets at most [`MAX_EVENTS_PER_SEC`] events through; `force` always passes.
#[derive(Debug)]
pub struct Throttle {
    min: Duration,
    last: Option<Duration>,
}

impl Default for Throttle {
    fn default() -> Self {
        Self::new(MAX_EVENTS_PER_SEC)
    }
}

impl Throttle {
    pub fn new(per_sec: u32) -> Self {
        Self { min: Duration::from_secs_f64(1.0 / per_sec.max(1) as f64), last: None }
    }

    /// `now` is time since the load started.
    pub fn ready(&mut self, now: Duration, force: bool) -> bool {
        let ok = force || self.last.is_none_or(|l| now.saturating_sub(l) >= self.min);
        if ok {
            self.last = Some(now);
        }
        ok
    }
}

/// One reading of the counters.
#[derive(Debug, Clone, Default)]
pub struct Sample {
    pub elapsed_secs: f64,
    /// Bytes of the model read by this PC's backend so far.
    pub local_bytes: u64,
    /// Each probe's cumulative sent counter, in probe order.
    pub peer_sent: Vec<u64>,
    /// Each probe's cumulative cache-hit bytes, in probe order.
    pub peer_cached: Vec<u64>,
    /// llama.cpp logged that the weights are in place.
    pub weights_done: bool,
}

struct PeerTrack {
    id: String,
    name: String,
    expected: u64,
    baseline: u64,
    baseline_cached: u64,
    /// Bytes really sent.
    rate: RateWindow,
    /// Bytes in place (sent + from the cache), for the time left.
    done_rate: RateWindow,
}

pub struct ProgressTracker {
    model_bytes: u64,
    /// The overall percent never goes back (a linked PC's share becoming
    /// known adds to the total mid-load).
    last_percent: f64,
    local_rate: RateWindow,
    peers: Vec<PeerTrack>,
}

impl ProgressTracker {
    pub fn new(model_bytes: u64, probes: &[PeerProbe]) -> Self {
        Self {
            model_bytes,
            last_percent: 0.0,
            local_rate: RateWindow::default(),
            peers: probes
                .iter()
                .map(|p| PeerTrack {
                    id: p.id.clone(),
                    name: p.name.clone(),
                    expected: p.expected_bytes,
                    baseline: probe_sent(p),
                    baseline_cached: probe_cached(p),
                    rate: RateWindow::default(),
                    done_rate: RateWindow::default(),
                })
                .collect(),
        }
    }

    /// Replace a peer's expected bytes (from the backend's buffer-size line).
    pub fn set_expected(&mut self, idx: usize, bytes: u64) {
        if let Some(p) = self.peers.get_mut(idx)
            && bytes > 0
        {
            p.expected = bytes;
        }
    }

    pub fn update(&mut self, s: &Sample) -> LoadProgress {
        let t = s.elapsed_secs;
        let total = self.model_bytes;
        let local_done = if s.weights_done { total } else { s.local_bytes.min(total) };
        self.local_rate.push(t, local_done);
        let local_rate = self.local_rate.rate();
        let mut devices = vec![DeviceProgress {
            id: "local".into(),
            name: "This PC".into(),
            linked: false,
            done_bytes: local_done,
            total_bytes: total,
            percent: if s.weights_done { 100.0 } else { percent_of(local_done, total).min(99.0) },
            bytes_per_sec: if s.weights_done { None } else { local_rate },
            eta_secs: if s.weights_done { Some(0) } else { eta_secs(total - local_done, local_rate) },
            cached: false,
            cached_bytes: 0,
        }];
        for (i, p) in self.peers.iter_mut().enumerate() {
            let sent = s.peer_sent.get(i).copied().unwrap_or(0).saturating_sub(p.baseline);
            let from_cache = s.peer_cached.get(i).copied().unwrap_or(0).saturating_sub(p.baseline_cached);
            let got = sent + from_cache;
            p.rate.push(t, sent);
            p.done_rate.push(t, got);
            let expected = p.expected;
            let (done, total, percent, rate, eta) = if s.weights_done {
                let total = expected.max(got);
                (total, total, 100.0, None, Some(0))
            } else if expected == 0 {
                // Share not known yet: bytes only.
                (got, got, 0.0, p.rate.rate(), None)
            } else {
                let done = got.min(expected);
                (done, expected, percent_of(done, expected).min(99.0), p.rate.rate(), eta_secs(expected - done, p.done_rate.rate()))
            };
            devices.push(DeviceProgress {
                id: p.id.clone(),
                name: p.name.clone(),
                linked: true,
                done_bytes: done,
                total_bytes: total,
                percent,
                bytes_per_sec: rate,
                eta_secs: eta,
                cached: from_cache > 0,
                cached_bytes: from_cache,
            });
        }
        let sum_total: u64 = devices.iter().map(|d| d.total_bytes).sum();
        let sum_done: u64 = devices.iter().map(|d| d.done_bytes).sum();
        let (phase, percent) = if s.weights_done {
            (LoadPhase::Finishing, 99.0)
        } else if local_done == 0 && devices.iter().all(|d| d.done_bytes == 0) {
            (LoadPhase::Starting, 0.0)
        } else {
            (LoadPhase::Loading, percent_of(sum_done, sum_total).min(99.0).max(self.last_percent))
        };
        self.last_percent = percent;
        let eta = if s.weights_done { None } else { devices.iter().filter_map(|d| d.eta_secs).max() };
        LoadProgress { phase, percent, eta_secs: eta, elapsed_secs: t, devices, error: None }
    }

    /// The last event of a successful load.
    pub fn finished(&self, last: &LoadProgress, elapsed_secs: f64) -> LoadProgress {
        let mut p = last.clone();
        p.phase = LoadPhase::Done;
        p.percent = 100.0;
        p.eta_secs = Some(0);
        p.elapsed_secs = elapsed_secs;
        for d in &mut p.devices {
            d.percent = 100.0;
            d.done_bytes = d.total_bytes;
            d.bytes_per_sec = None;
            d.eta_secs = Some(0);
        }
        p
    }
}

/// (bytes read with ReadFile, working set) of a process, if it can be read.
#[cfg(windows)]
pub fn process_counters(pid: u32) -> Option<(u64, u64)> {
    use windows::Win32::Foundation::CloseHandle;
    use windows::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows::Win32::System::Threading::{GetProcessIoCounters, IO_COUNTERS, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};
    unsafe {
        let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid).ok()?;
        let mut io = IO_COUNTERS::default();
        let io_ok = GetProcessIoCounters(h, &mut io).is_ok();
        let mut mem = PROCESS_MEMORY_COUNTERS::default();
        let mem_ok = GetProcessMemoryInfo(h, &mut mem, std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32).is_ok();
        let _ = CloseHandle(h);
        if !io_ok && !mem_ok {
            return None;
        }
        Some((io.ReadTransferCount, mem.WorkingSetSize as u64))
    }
}

#[cfg(not(windows))]
pub fn process_counters(_pid: u32) -> Option<(u64, u64)> {
    None
}

/// Model bytes read so far from two readings of the backend's counters:
/// the read-counter growth or the working-set growth, whichever is larger.
pub fn local_read_bytes(first: (u64, u64), now: (u64, u64)) -> u64 {
    now.0.saturating_sub(first.0).max(now.1.saturating_sub(first.1))
}

/// Read the live sent counter of a probe.
pub fn probe_sent(p: &PeerProbe) -> u64 {
    p.sent.load(Ordering::Relaxed)
}

/// Read the live cache-hit counter of a probe.
pub fn probe_cached(p: &PeerProbe) -> u64 {
    p.cached.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1 << 30;

    fn probe(id: &str, expected: u64, c: &Arc<AtomicU64>) -> PeerProbe {
        PeerProbe { id: id.into(), name: format!("PC {id}"), endpoint: "127.0.0.1:1".into(), expected_bytes: expected, sent: c.clone(), cached: Arc::new(AtomicU64::new(0)) }
    }

    #[test]
    fn percent_and_eta_math() {
        assert_eq!(percent_of(0, 0), 0.0);
        assert_eq!(percent_of(50, 200), 25.0);
        assert_eq!(percent_of(300, 200), 100.0);
        // 600 MB left at 2 MB/s = 300 s; too slow or unknown = no ETA.
        assert_eq!(eta_secs(600_000_000, Some(2_000_000.0)), Some(300));
        assert_eq!(eta_secs(1, Some(10.0)), None);
        assert_eq!(eta_secs(1, None), None);
        assert_eq!(eta_secs(1, Some(f64::NAN)), None);
    }

    #[test]
    fn rate_window_measures_recent_speed() {
        let mut r = RateWindow::default();
        assert_eq!(r.rate(), None);
        r.push(0.0, 0);
        r.push(0.2, 100); // too short a span
        assert_eq!(r.rate(), None);
        r.push(1.0, 10_000_000);
        assert!((r.rate().unwrap() - 10_000_000.0).abs() < 1.0);
        // Old samples fall out: the speed drops to the recent one.
        for i in 2..=20 {
            r.push(i as f64, 10_000_000 + (i as u64 - 1) * 1_000_000);
        }
        assert!((r.rate().unwrap() - 1_000_000.0).abs() < 1.0, "{:?}", r.rate());
    }

    #[test]
    fn idle_time_and_stalls_give_no_speed() {
        // VM cold load (2026-11-23): nothing for 10 s, then ~100 MB/s.
        let mut r = RateWindow::default();
        for i in 0..50 {
            r.push(i as f64 * 0.2, 0);
        }
        assert_eq!(r.rate(), None, "idle: no speed");
        for k in 1..=5u64 {
            r.push(9.8 + k as f64 * 0.2, k * 20_000_000);
        }
        let v = r.rate().unwrap();
        assert!((v - 100_000_000.0).abs() < 1.0, "{v}");
        // The data stops: after the stall time there is no speed (and so no ETA).
        for k in 1..=10 {
            r.push(11.0 + k as f64 * 0.2, 100_000_000);
        }
        assert_eq!(r.rate(), None);
        assert_eq!(eta_secs(1_000_000_000, r.rate()), None);
    }

    #[test]
    fn throttle_allows_five_per_second_and_forced_events() {
        let mut t = Throttle::default();
        let mut passed = 0;
        // A poll every 10 ms for 3 s.
        for i in 0..300 {
            if t.ready(Duration::from_millis(i * 10), false) {
                passed += 1;
            }
        }
        assert!((14..=16).contains(&passed), "{passed} events in 3 s");
        // A forced event always passes and restarts the interval.
        assert!(t.ready(Duration::from_millis(2999), true));
        assert!(!t.ready(Duration::from_millis(3000), false));
        assert!(t.ready(Duration::from_millis(3200), false));
    }

    #[test]
    fn local_only_load_reports_reading_then_finishing() {
        let mut tr = ProgressTracker::new(10 * GB, &[]);
        let p = tr.update(&Sample { elapsed_secs: 0.0, ..Default::default() });
        assert_eq!(p.phase, LoadPhase::Starting);
        let p = tr.update(&Sample { elapsed_secs: 2.0, local_bytes: 2 * GB, ..Default::default() });
        assert_eq!(p.phase, LoadPhase::Loading);
        assert_eq!(p.devices.len(), 1);
        assert!((p.percent - 20.0).abs() < 0.01);
        assert!((p.devices[0].bytes_per_sec.unwrap() - GB as f64).abs() < 1.0);
        assert_eq!(p.devices[0].eta_secs, Some(8));
        // Over-reading (memory-map noise) never shows 100% before the weights are in.
        let p = tr.update(&Sample { elapsed_secs: 5.0, local_bytes: 12 * GB, ..Default::default() });
        assert!(p.percent <= 99.0 && p.devices[0].percent <= 99.0);
        let p = tr.update(&Sample { elapsed_secs: 6.0, local_bytes: 10 * GB, weights_done: true, ..Default::default() });
        assert_eq!(p.phase, LoadPhase::Finishing);
        assert_eq!(p.devices[0].percent, 100.0);
        let done = tr.finished(&p, 7.0);
        assert_eq!((done.phase, done.percent), (LoadPhase::Done, 100.0));
    }

    fn sample(t: f64, local: u64, sent: u64, cached: u64, done: bool) -> Sample {
        Sample { elapsed_secs: t, local_bytes: local, peer_sent: vec![sent], peer_cached: vec![cached], weights_done: done }
    }

    #[test]
    fn linked_pc_progress_uses_bytes_sent_since_the_load_started() {
        let c = Arc::new(AtomicU64::new(5 * GB)); // an earlier load's total
        let probes = [probe("a", 4 * GB, &c)];
        let mut tr = ProgressTracker::new(10 * GB, &probes);
        tr.update(&sample(0.0, GB, 5 * GB, 0, false));
        let p = tr.update(&sample(4.0, 3 * GB, 6 * GB, 0, false));
        let peer = &p.devices[1];
        assert!(peer.linked && !peer.cached);
        assert!((peer.percent - 25.0).abs() < 0.01, "{}", peer.percent);
        assert!((peer.bytes_per_sec.unwrap() - (GB / 4) as f64).abs() < 1.0);
        assert_eq!(peer.eta_secs, Some(12)); // 3 GB left at 0.25 GB/s
        assert_eq!(p.eta_secs, Some(14)); // the slowest device decides
        // Overall counts both devices' bytes: (3 + 1) / (10 + 4).
        assert!((p.percent - 4.0 / 14.0 * 100.0).abs() < 0.01, "{}", p.percent);
    }

    #[test]
    fn the_transfer_after_the_layer_line_is_tracked_until_the_weights_are_in() {
        // VM test (2026-11-23): llama.cpp logs "offloaded 37/37 layers" and
        // the RPC0 buffer size at 0.85 s, then sends 1.43 GB to the linked PC.
        // The row follows the real bytes against the real share and is never
        // 100% before the weights are in place.
        let c = Arc::new(AtomicU64::new(0));
        let mut tr = ProgressTracker::new(2_500_000_000, &[probe("vm", 0, &c)]);
        let p = tr.update(&sample(1.0, 0, 0, 0, false));
        assert_eq!((p.devices[1].percent, p.devices[1].total_bytes), (0.0, 0));
        tr.set_expected(0, 1_434_000_000);
        let mut last = 0.0;
        for k in 1..=10u64 {
            let p = tr.update(&sample(1.0 + k as f64, 2_000_000_000, k * 140_000_000, 0, false));
            let d = &p.devices[1];
            assert!(d.percent >= last && d.percent <= 99.0, "{k}: {}", d.percent);
            assert_eq!(d.total_bytes, 1_434_000_000, "the total never moves");
            assert!((d.bytes_per_sec.unwrap() - 140_000_000.0).abs() < 1.0);
            assert!(!d.cached, "nothing came from a cache");
            assert!(p.percent < 99.0 && p.phase == LoadPhase::Loading);
            last = d.percent;
        }
        assert!(last > 97.0);
        // The overall bar never went back when the share became known.
        let mut tr = ProgressTracker::new(2_500_000_000, &[probe("vm", 0, &c)]);
        let before = tr.update(&sample(1.0, 400_000_000, 0, 0, false)).percent;
        tr.set_expected(0, 1_434_000_000);
        assert!(tr.update(&sample(1.2, 420_000_000, 0, 0, false)).percent >= before);
        let p = tr.update(&sample(12.0, 2_500_000_000, 1_420_000_000, 0, true));
        assert_eq!((p.phase, p.devices[1].percent), (LoadPhase::Finishing, 100.0));
    }

    #[test]
    fn cached_only_when_the_helper_reported_hits() {
        let c = Arc::new(AtomicU64::new(0));
        let probes = [probe("a", 0, &c)];
        let cache = probes[0].cached.clone();
        let mut tr = ProgressTracker::new(10 * GB, &probes);
        tr.set_expected(0, 4 * GB);
        // The local read is far ahead and almost nothing was sent yet: still not "cached".
        let p = tr.update(&sample(1.0, 5 * GB, 4096, 0, false));
        assert!(!p.devices[1].cached);
        // The helper answered "have it" for 3 GB; 0.5 GB was really sent.
        cache.store(3 * GB, Ordering::Relaxed);
        let p = tr.update(&sample(3.0, 8 * GB, GB / 2, 3 * GB, false));
        let d = &p.devices[1];
        assert!(d.cached);
        assert_eq!(d.cached_bytes, 3 * GB);
        assert_eq!(d.done_bytes, 3 * GB + GB / 2);
        assert!((d.percent - 87.5).abs() < 0.01, "{}", d.percent);
        // Stays cached to the end (never flips back during the last step).
        let p = tr.update(&sample(9.0, 10 * GB, GB, 3 * GB, true));
        assert!(p.devices[1].cached && p.devices[1].percent == 100.0);
    }

    #[test]
    fn expected_bytes_can_be_refined_and_failures_carry_the_error() {
        let c = Arc::new(AtomicU64::new(0));
        let mut tr = ProgressTracker::new(10 * GB, &[probe("a", 0, &c)]);
        // Unknown share: bytes are shown but no percent from them.
        let p = tr.update(&sample(1.0, GB, GB / 2, 0, false));
        assert_eq!(p.devices[1].done_bytes, GB / 2);
        assert_eq!(p.devices[1].percent, 0.0);
        tr.set_expected(0, 2 * GB);
        let p = tr.update(&sample(2.0, 2 * GB, GB, 0, false));
        assert!((p.devices[1].percent - 50.0).abs() < 0.01);
        let f = LoadProgress::failed("out of memory");
        assert_eq!(f.phase, LoadPhase::Failed);
        assert_eq!(f.error.as_deref(), Some("out of memory"));
    }

    #[test]
    fn local_read_takes_the_larger_counter_growth() {
        assert_eq!(local_read_bytes((100, 1000), (600, 1200)), 500); // ReadFile path
        assert_eq!(local_read_bytes((100, 1000), (150, 9000)), 8000); // memory-map path
        assert_eq!(local_read_bytes((100, 1000), (50, 500)), 0);
    }
}
