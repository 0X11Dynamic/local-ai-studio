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
    /// Every display adapter Windows reports (DXGI), including AMD and Intel
    /// GPUs that NVML can't see. Software adapters are left out.
    pub adapters: Vec<DisplayAdapter>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DisplayAdapter {
    pub name: String,
    /// NVIDIA | AMD | Intel | other
    pub vendor: String,
    pub dedicated_vram_bytes: u64,
}

impl DisplayAdapter {
    /// A GPU worth running models on with the Vulkan backend: a non-NVIDIA
    /// adapter with at least 2 GiB of its own VRAM (Radeon RX, Intel Arc).
    /// Integrated graphics report little or no dedicated memory.
    pub fn is_vulkan_candidate(&self) -> bool {
        self.vendor != "NVIDIA" && self.vendor != "other" && self.dedicated_vram_bytes >= 2 << 30
    }
}

fn vendor_name(id: u32) -> &'static str {
    match id {
        0x10de => "NVIDIA",
        0x1002 | 0x1022 => "AMD",
        0x8086 => "Intel",
        _ => "other",
    }
}

/// Hardware display adapters from DXGI. Empty if DXGI is unavailable.
#[cfg(windows)]
pub fn display_adapters() -> Vec<DisplayAdapter> {
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, IDXGIFactory1};
    let mut out = Vec::new();
    // SAFETY: plain COM calls; every returned interface is owned and released on drop.
    let Ok(factory) = (unsafe { CreateDXGIFactory1::<IDXGIFactory1>() }) else { return out };
    for i in 0.. {
        let Ok(adapter) = (unsafe { factory.EnumAdapters1(i) }) else { break };
        let Ok(d) = (unsafe { adapter.GetDesc1() }) else { continue };
        if d.Flags & (DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0 {
            continue;
        }
        let len = d.Description.iter().position(|&c| c == 0).unwrap_or(d.Description.len());
        let a = DisplayAdapter {
            name: String::from_utf16_lossy(&d.Description[..len]).trim().to_string(),
            vendor: vendor_name(d.VendorId).into(),
            dedicated_vram_bytes: d.DedicatedVideoMemory as u64,
        };
        // DXGI can list one GPU more than once (one entry per output path).
        if !out.iter().any(|x: &DisplayAdapter| x.name == a.name && x.dedicated_vram_bytes == a.dedicated_vram_bytes) {
            out.push(a);
        }
    }
    out
}

#[cfg(not(windows))]
pub fn display_adapters() -> Vec<DisplayAdapter> {
    Vec::new()
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
    /// Display adapters don't change while the app runs; read once.
    adapters: Vec<DisplayAdapter>,
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
        Ok(Self { sys: Mutex::new(sys), nvml, nvml_error, adapters: display_adapters() })
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
            adapters: self.adapters.clone(),
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
        // Every Windows PC has at least one hardware display adapter.
        #[cfg(windows)]
        assert!(!hw.adapters.is_empty(), "DXGI reported no display adapters");
        let s = m.snapshot(None);
        assert!(s.app_memory_bytes > 0);
    }
}
