// Linked PCs: one PC lends its GPU/CPU to another over the app's own
// encrypted link (pairing with a one-time code, pinned keys).
import { invoke } from "@tauri-apps/api/core";

export interface DeviceInfo { name: string; description: string; total_mib: number; free_mib: number }

export interface HelperSettings {
  port: number;
  /** rpc-server devices, e.g. ["CUDA0"]; empty = all. */
  devices: string[];
  /** Default CPU threads a linked PC gets; null = all cores but 2. */
  threads: number | null;
  /** Older builds' single "memory to lend" (MiB); moved to the two defaults below. */
  ram_cap_mib: number | null;
  /** Default most GPU memory (MiB) per linked PC; null = 75% of free GPU memory. */
  share_gpu_mib: number | null;
  /** Default most RAM (MiB) per linked PC when the CPU is shared; null = 50% of available RAM. */
  share_ram_mib: number | null;
  /** Keep received weights on disk (rpc-server -c). */
  cache: boolean;
  /** Size cap of that cache in GB (oldest files go first). */
  cache_cap_gb: number;
  /** lz4 inside the tunnel (off by default). */
  compress: boolean;
  runtime_id: string | null;
}

export interface HelperLive {
  helping: boolean;
  rpc_endpoint: string | null;
  runtime_id: string | null;
  runtime_tag: string | null;
  runtime_variant: string | null;
  devices: DeviceInfo[];
  available_devices: DeviceInfo[];
  ram_cap_mib: number | null;
  error: string | null;
}

export interface CodeView { code: string; state: "open" | "used" | "locked" | "expired"; seconds_left: number; failures: number }

export interface ReachReport {
  local_ip: string | null;
  port: number;
  upnp: { kind: "not_tried" | "no_gateway" | "mapped" | "map_failed"; external_ip?: string | null; error?: string; external_port?: number; gateway?: string };
  public_ip: string | null;
  verdict: "reachable" | "needs_port_forward" | "cgnat" | "unknown";
  message: string;
  steps: string[];
}

/** What one PC lends another (effective values). */
export interface ShareLimits { gpu_mib: number; ram_mib: number; threads: number; paused: boolean }

/** Per-PC limits as stored; null = the default. */
export interface PeerShare { gpu_mib: number | null; ram_mib: number | null; threads: number | null; paused: boolean }

/** What this PC lends to a linked PC (enforced here) and what it uses now. */
export interface LendView {
  limits: ShareLimits;
  stored: PeerShare;
  used_gpu_mib: number;
  used_ram_mib: number;
  shares_gpu: boolean;
  shares_ram: boolean;
  refused: number;
  refused_bytes: number;
  last_refused_secs_ago: number | null;
}

/** A device a linked PC lends a load. */
export interface LinkedShare { peer: string; endpoint: string; device: string; bytes: number; cpu: boolean; cap_bytes: number | null }

export interface PeerView {
  id: string;
  name: string;
  fingerprint: string;
  /** Address this PC dials (only PCs it linked to with their code). */
  address: string | null;
  /** PCs that linked to this one: the IP they connected from (shown only; behind NAT it can't be dialled). */
  seen_from: string | null;
  /** This PC can connect to it. */
  dialable: boolean;
  same_network: boolean;
  split_layers: boolean;
  /** "Send whole jobs" (a later phase): stored only. */
  send_jobs: boolean;
  initiated: boolean;
  online: boolean;
  rtt_ms: number | null;
  bandwidth_mbps: number | null;
  added_ms_per_token: number | null;
  runtime_tag: string | null;
  helping: boolean;
  devices: DeviceInfo[];
  ram_cap_mib: number | null;
  last_seen_at: string | null;
  connected_here: boolean;
  /** Bytes of this PC's weights in the helper cache here. */
  cache_bytes: number;
  lends: LendView;
  /** What it lets this PC use there; null = unknown or an older version. */
  offers: ShareLimits | null;
  offers_enforced: boolean;
  /** Its last refusal of this PC's memory request. */
  refusal: string | null;
}

/** The helper's weight cache on disk. */
export interface CacheView { enabled: boolean; bytes: number; cap_bytes: number; dir: string | null }

export interface SessionEvent { id: number; peer_key: string; peer_name: string; remote: string; mode: "control" | "rpc"; open: boolean }

export interface LinkStatus {
  name: string;
  fingerprint: string;
  local_ip: string | null;
  helper: { enabled: boolean; settings: HelperSettings; live: HelperLive; listening: string | null; listen_port: number | null; bind_notice: string | null; port_notice: string | null; log_tail: string[]; cache: CacheView };
  code: CodeView | null;
  reach: ReachReport | null;
  peers: PeerView[];
  sessions: SessionEvent[];
}

export interface SplitEstimate { peer_id: string; name: string; online: boolean; rtt_ms: number | null; added_ms_per_token: number | null }
export interface SplitPreview {
  peers: SplitEstimate[];
  alone_tok_s: number | null;
  /** Estimate range (tokens/s): weights per device / its memory bandwidth, plus one round trip per linked PC. */
  est_low_tok_s: number | null;
  est_high_tok_s: number | null;
  /** What the estimate assumed, one line per device and the network. */
  est_basis: string[];
  /** Why there is no estimate. */
  est_missing: string | null;
  split_tok_s: number | null;
  split_replies: number;
  suggest_off: boolean;
  message: string | null;
}

export const api9 = {
  status: () => invoke<LinkStatus>("linking_status"),
  setName: (name: string) => invoke<LinkStatus>("linking_set_name", { name }),
  setHelper: (enabled: boolean, settings: HelperSettings) => invoke<LinkStatus>("linking_set_helper", { enabled, settings }),
  startCode: () => invoke<CodeView>("linking_start_code"),
  cancelCode: () => invoke<void>("linking_cancel_code"),
  pair: (address: string, port: number | null, code: string) => invoke<PeerView>("linking_pair", { address, port, code }),
  connect: (peerId: string) => invoke<PeerView>("linking_connect", { peerId }),
  setSplit: (peerId: string, on: boolean) => invoke<PeerView>("linking_set_split", { peerId, on }),
  setSendJobs: (peerId: string, on: boolean) => invoke<PeerView>("linking_set_send_jobs", { peerId, on }),
  unlink: (peerId: string) => invoke<boolean>("linking_unlink", { peerId }),
  /** Delete every cached weight file; returns the bytes freed. */
  clearCache: () => invoke<number>("linking_clear_cache"),
  checkReach: (tryUpnp: boolean) => invoke<ReachReport>("linking_check_reach", { tryUpnp }),
  splitPreview: (modelId: string | null) => invoke<SplitPreview>("linking_split_preview", { modelId }),
  /** What this PC lends to a linked PC (null fields = the defaults). */
  setPeerShare: (peerId: string, share: PeerShare) => invoke<PeerView>("linking_set_peer_share", { peerId, share }),
};

/** "482913" -> "482 913" */
export const showCode = (c: string) => (c.length === 6 ? `${c.slice(0, 3)} ${c.slice(3)}` : c);
