// Typed wrappers for the Rust IPC commands (src-tauri/src/commands).
// Types are written by hand to match the serde output of the Rust structs.
import { invoke } from "@tauri-apps/api/core";

export interface ApiError { code: string; message: string }

export function errorMessage(e: unknown): string {
  if (e && typeof e === "object" && "message" in e) return String((e as ApiError).message);
  return String(e);
}

// ---------- system ----------
export interface AppPaths {
  data_dir: string; db_path: string; runtimes_dir: string; logs_dir: string;
  skills_dir: string; default_models_dir: string; lmstudio_models_dir: string | null;
}
export interface GpuInfo {
  index: number; name: string; vendor: string; vram_total_bytes: number; vram_used_bytes: number;
  driver_version: string | null; compute_capability: string | null;
}
export interface HardwareInfo {
  os: string; cpu_name: string; physical_cores: number | null; logical_cores: number;
  total_ram_bytes: number; available_ram_bytes: number; gpus: GpuInfo[];
  gpu_note: string | null; cuda_driver_version: string | null;
}
export interface GpuUsage { index: number; utilization_percent: number | null; vram_used_bytes: number; vram_total_bytes: number }
export interface ResourceSnapshot {
  cpu_usage_percent: number; ram_used_bytes: number; ram_total_bytes: number; gpus: GpuUsage[];
  app_memory_bytes: number; backend_memory_bytes: number | null; backend_vram_bytes: number | null;
}

// ---------- runtimes ----------
export interface RuntimeCapabilities { flags: string[]; cache_types: string[]; load_modes: string[] }
export interface InstalledRuntime {
  id: string; tag: string; variant: string; server_path: string; installed_at: string;
  source: string; capabilities: RuntimeCapabilities;
}
export interface AssetRef { name: string; url: string; size: number; sha256: string | null }
export interface AvailableRuntime {
  id: string; tag: string; variant: string; published_at: string; assets: AssetRef[];
  total_size: number; recommended: boolean; note: string;
}
export interface InstallProgress {
  runtime_id: string; phase: "download" | "extract" | "probe" | "done"; asset: string | null;
  downloaded: number; total: number | null; bytes_per_sec: number;
}

// ---------- models ----------
export interface ModelMetadata {
  architecture: string | null; name: string | null; size_label: string | null; file_type: number | null;
  quantization: string | null; context_length: number | null; block_count: number | null;
  embedding_length: number | null; head_count: number | null; head_count_kv: number | null;
  expert_count: number | null; vocab_size: number | null; has_chat_template: boolean;
  split_count: number | null; is_projector: boolean; gguf_version: number;
}
export interface ModelView {
  id: string; repo_id: string | null; path: string; format: string; quantization: string | null;
  size: number; display_name: string; source: string; projector_path: string | null;
  status: "ok" | "missing" | "invalid"; status_detail: string | null;
  created_at: string; updated_at: string; meta: ModelMetadata; shards: string[]; active: boolean;
}
export interface ImportReport { registered: string[]; invalid: string[]; skipped: { path: string; reason: string }[] }

export type GpuLayers =
  | { mode: "auto" } | { mode: "all" } | { mode: "cpu_only" } | { mode: "count"; count: number };
export interface LoadConfig {
  n_ctx: number; gpu_layers: GpuLayers; threads: number | null; threads_batch: number | null;
  batch_size: number | null; ubatch_size: number | null; flash_attention: "auto" | "on" | "off";
  cache_type_k: string; cache_type_v: string; kv_offload: boolean; rope_scaling: string | null;
  rope_freq_base: number | null; rope_freq_scale: number | null;
  load_mode: "auto" | "mmap" | "mlock" | "mmap_mlock" | "none"; parallel: number;
  load_projector: boolean; fit: boolean;
  /** Split the model between GPU and CPU from free VRAM at load time. */
  share_with_cpu?: boolean;
  /** MoE: expert weights of the first N layers stay on the CPU (set by the app). */
  n_cpu_moe?: number | null;
}
export interface LoadDefaults {
  config: LoadConfig; source: "model_default" | "last_used" | "app_default";
  has_projector: boolean; trained_context: number | null; total_layers: number | null;
}
export interface MemoryEstimate {
  weights_bytes: number; kv_cache_bytes: number; compute_buffer_bytes: number; vram_bytes: number;
  ram_bytes: number; total_layers: number; offloaded_layers: number; notes: string[];
}
export interface EstimateView {
  estimate: MemoryEstimate; free_vram_bytes: number | null; total_vram_bytes: number | null;
  available_ram_bytes: number; suggested_gpu_layers: number | null; fits: boolean; warnings: string[];
  /** The split "Share with CPU" would pick right now (null without GPU info). */
  share_split: ShareSplit | null;
  /** Off, and the model doesn't fit in free VRAM: suggest turning it on. */
  suggest_share: boolean;
}
export interface ShareSplit {
  gpu_layers: number; total_layers: number; n_cpu_moe: number | null; all_fit: boolean; vram_bytes: number; summary: string;
}

// ---------- engine ----------
export type EngineStateName = "unloaded" | "loading" | "loaded" | "unloading" | "failed";
export interface BufferReport { device: string; kind: string; mib: number }
export interface DeviceReport { device: string; name: string; total_mib: number; free_mib: number }
export interface ServerProps {
  n_ctx: number | null; total_slots: number | null; model_path: string | null;
  chat_template: string | null; build_info: string | null; vision: boolean;
}
export interface EngineStatus {
  state: EngineStateName; model_id: string | null; instance_id: string | null; runtime_id: string | null;
  pid: number | null; port: number | null; config: LoadConfig | null; args: string[];
  unsupported_settings: string[]; props: ServerProps | null; gpu_layers_offloaded: [number, number] | null;
  buffers: BufferReport[]; devices: DeviceReport[]; warnings: string[]; load_seconds: number | null; error: string | null; error_hint: string | null;
}
export type EngineEvent =
  | { type: "status"; status: EngineStatus }
  | { type: "log"; line: string; stream: "stdout" | "stderr" }
  | { type: "instance_ended"; instance_id: string };

export const api = {
  getPaths: () => invoke<AppPaths>("get_paths"),
  getHardware: () => invoke<HardwareInfo>("get_hardware"),
  getResources: () => invoke<ResourceSnapshot>("get_resources"),
  getSettings: () => invoke<Record<string, unknown>>("get_settings"),
  setSetting: (key: string, value: unknown) => invoke<void>("set_setting", { key, value }),

  listRuntimes: () => invoke<InstalledRuntime[]>("list_runtimes"),
  fetchAvailableRuntimes: () => invoke<AvailableRuntime[]>("fetch_available_runtimes"),
  installRuntime: (runtime: AvailableRuntime) => invoke<InstalledRuntime>("install_runtime", { runtime }),
  cancelRuntimeInstall: (id: string) => invoke<void>("cancel_runtime_install", { id }),
  registerCustomRuntime: (path: string) => invoke<InstalledRuntime>("register_custom_runtime", { path }),
  removeRuntime: (id: string) => invoke<void>("remove_runtime", { id }),

  listModels: () => invoke<ModelView[]>("list_models"),
  importModels: (path: string) => invoke<ImportReport>("import_models", { path }),
  scanLmStudio: () => invoke<ImportReport>("scan_lmstudio"),
  refreshModels: () => invoke<ModelView[]>("refresh_models"),
  removeModel: (id: string) => invoke<void>("remove_model", { id }),
  deleteModelFiles: (id: string, confirmName: string) => invoke<string[]>("delete_model_files", { id, confirmName }),
  estimateMemory: (id: string, config: LoadConfig) => invoke<EstimateView>("estimate_memory", { id, config }),
  getLoadDefaults: (id: string) => invoke<LoadDefaults>("get_load_defaults", { id }),
  saveModelDefaults: (id: string, config: LoadConfig) => invoke<void>("save_model_defaults", { id, config }),
  resetModelDefaults: (id: string) => invoke<void>("reset_model_defaults", { id }),

  loadModel: (modelId: string, config: LoadConfig, runtimeId: string | null) =>
    invoke<EngineStatus>("load_model", { modelId, config, runtimeId }),
  unloadModel: () => invoke<EngineStatus>("unload_model"),
  cancelLoad: () => invoke<void>("cancel_load"),
  engineStatus: () => invoke<EngineStatus>("engine_status"),
  engineLogs: (last?: number) => invoke<string[]>("engine_logs", { last }),
};
