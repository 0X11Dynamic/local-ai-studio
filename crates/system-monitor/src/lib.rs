//! Hardware detection and live resource metrics.
//!
//! CPU and RAM come from `sysinfo`. NVIDIA GPU name, VRAM, utilization and
//! compute capability come from NVML (`nvml.dll`, installed with the driver).
//! When NVML is missing, GPU fields are reported as unavailable rather than guessed.

use std::sync::Mutex;

use nvml_wrapper::Nvml;
use nvml_wrapper::enums::device::UsedGpuMemory;
use serde::Serialize;
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

#[derive(Debug, Clone, Serialize)]
pub struct HardwareInfo {
    pub os: String,
    pub cpu_name: String,
    pub physical_cores: Option<usize>,
    pub logical_cores: usize,
    pub total_ram_bytes: u64,
    pub available_ram_bytes: u64,
    pub gpus: Vec<GpuInfo>,
    /// Why GPU metrics are missing, when they are (e.g. NVML not found).
    pub gpu_note: Option<String>,
    /// Highest CUDA version the installed driver supports, e.g. "13.1".
    pub cuda_driver_version: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    pub vendor: String,
    pub vram_total_bytes: u64,
    pub vram_used_bytes: u64,
    pub driver_version: Option<String>,
    /// e.g. "12.0" for Blackwell consumer cards.
    pub compute_capability: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResourceSnapshot {
    pub cpu_usage_percent: f32,
    pub ram_used_bytes: u64,
    pub ram_total_bytes: u64,
    pub gpus: Vec<GpuUsage>,
    /// Resident memory of this app process.
    pub app_memory_bytes: u64,
    /// Resident memory of the inference backend process, if one is running.
    pub backend_memory_bytes: Option<u64>,
    /// VRAM used by the backend process, if the driver reports it per process.
    /// Windows WDDM drivers often do not; then this is None (not zero).
    pub backend_vram_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GpuUsage {
    pub index: u32,
    pub utilization_percent: Option<u32>,
    pub vram_used_bytes: u64,
    pub vram_total_bytes: u64,
}

pub struct SystemMonitor {
    sys: Mutex<System>,
    nvml: Option<Nvml>,
    nvml_error: Option<String>,
}

impl SystemMonitor {
    /// Fails only if sysinfo does not support this OS.
    pub fn new() -> Result<Self, String> {
        let mut sys = System::new().map_err(|e| format!("system metrics unavailable: {e}"))?;
        sys.refresh_memory();
        sys.refresh_cpu_all();
        let (nvml, nvml_error) = match Nvml::init() {
            Ok(n) => (Some(n), None),
            Err(e) => {
                tracing::info!("NVML unavailable: {e}");
                (None, Some(format!("NVIDIA NVML not available ({e}); GPU metrics disabled")))
            }
        };
        Ok(Self { sys: Mutex::new(sys), nvml, nvml_error })
    }

    pub fn hardware_info(&self) -> HardwareInfo {
        let mut sys = self.sys.lock().unwrap();
        sys.refresh_memory();
        let cpu_name = sys
            .cpus()
            .first()
            .map(|c| c.brand().trim().to_string())
            .unwrap_or_else(|| "Unknown CPU".into());
        let logical = sys.cpus().len();
        let physical = System::physical_core_count().ok();
        let os = format!(
            "{} {}",
            System::name().unwrap_or_else(|_| "Windows".into()),
            System::os_version().unwrap_or_default()
        );
        let total = sys.total_memory();
        let avail = sys.available_memory();
        drop(sys);

        let mut gpus = Vec::new();
        let mut cuda_driver_version = None;
        let mut gpu_note = self.nvml_error.clone();
        if let Some(nvml) = &self.nvml {
            let driver = nvml.sys_driver_version().ok();
            cuda_driver_version = nvml
                .sys_cuda_driver_version()
                .ok()
                .map(|v| format!("{}.{}", v / 1000, (v % 1000) / 10));
            match nvml.device_count() {
                Ok(n) => {
                    for i in 0..n {
                        let Ok(dev) = nvml.device_by_index(i) else { continue };
                        let mem = dev.memory_info().ok();
                        gpus.push(GpuInfo {
                            index: i,
                            name: dev.name().unwrap_or_else(|_| "NVIDIA GPU".into()),
                            vendor: "NVIDIA".into(),
                            vram_total_bytes: mem.as_ref().map(|m| m.total).unwrap_or(0),
                            vram_used_bytes: mem.as_ref().map(|m| m.used).unwrap_or(0),
                            driver_version: driver.clone(),
                            compute_capability: dev
                                .cuda_compute_capability()
                                .ok()
                                .map(|c| format!("{}.{}", c.major, c.minor)),
                        });
                    }
                }
                Err(e) => gpu_note = Some(format!("NVML device query failed: {e}")),
            }
        }
        HardwareInfo {
            os,
            cpu_name,
            physical_cores: physical,
            logical_cores: logical,
            total_ram_bytes: total,
            available_ram_bytes: avail,
            gpus,
            gpu_note,
            cuda_driver_version,
        }
    }

    /// Live metrics. `backend_pid` is the llama-server process, if running.
    pub fn snapshot(&self, backend_pid: Option<u32>) -> ResourceSnapshot {
        let mut sys = self.sys.lock().unwrap();
        sys.refresh_memory();
        sys.refresh_cpu_usage();
        let me = Pid::from_u32(std::process::id());
        let mut pids = vec![me];
        if let Some(p) = backend_pid {
            pids.push(Pid::from_u32(p));
        }
        sys.refresh_processes_specifics(
            ProcessesToUpdate::Some(&pids),
            true,
            ProcessRefreshKind::nothing().with_memory(),
        );
        let app_memory_bytes = sys.process(me).map(|p| p.memory()).unwrap_or(0);
        let backend_memory_bytes =
            backend_pid.and_then(|p| sys.process(Pid::from_u32(p)).map(|p| p.memory()));
        let snap_cpu = sys.global_cpu_usage();
        let ram_used = sys.used_memory();
        let ram_total = sys.total_memory();
        drop(sys);

        let mut gpus = Vec::new();
        let mut backend_vram_bytes = None;
        if let Some(nvml) = &self.nvml
            && let Ok(n) = nvml.device_count()
        {
            for i in 0..n {
                let Ok(dev) = nvml.device_by_index(i) else { continue };
                let mem = dev.memory_info().ok();
                if let (Some(bp), Ok(procs)) = (backend_pid, dev.running_compute_processes()) {
                    for p in procs {
                        if p.pid == bp
                            && let UsedGpuMemory::Used(b) = p.used_gpu_memory
                        {
                            *backend_vram_bytes.get_or_insert(0) += b;
                        }
                    }
                }
                gpus.push(GpuUsage {
                    index: i,
                    utilization_percent: dev.utilization_rates().ok().map(|u| u.gpu),
                    vram_used_bytes: mem.as_ref().map(|m| m.used).unwrap_or(0),
                    vram_total_bytes: mem.as_ref().map(|m| m.total).unwrap_or(0),
                });
            }
        }
        ResourceSnapshot {
            cpu_usage_percent: snap_cpu,
            ram_used_bytes: ram_used,
            ram_total_bytes: ram_total,
            gpus,
            app_memory_bytes,
            backend_memory_bytes,
            backend_vram_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_real_cpu_and_ram() {
        let m = SystemMonitor::new().unwrap();
        let hw = m.hardware_info();
        assert!(hw.logical_cores > 0);
        assert!(hw.total_ram_bytes > 1 << 30, "expected more than 1 GiB RAM");
        println!("{hw:#?}");
        let s = m.snapshot(None);
        assert!(s.app_memory_bytes > 0);
    }
}
