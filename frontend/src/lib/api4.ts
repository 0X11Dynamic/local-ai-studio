// Typed wrappers for phase 4 (execution environments and built-in tools).
import { invoke } from "@tauri-apps/api/core";

export type EnvKind = "local" | "ssh" | "docker" | "sandbox";
export type Policy = "allow" | "ask" | "deny";
export interface BuiltinToolView { name: string; description: string; enabled: boolean; policy: Policy; default_policy: Policy }
export interface Health { ok: boolean; detail: string; checked_at: string }
export interface EnvView {
  id: string; kind: EnvKind; name: string; label: string; config: Record<string, unknown>; has_secret: boolean;
  last_health: Health | null; last_health_at: string | null; tools: BuiltinToolView[];
}
export interface SshProfile { host: string; port: number; user: string; auth: "key" | "password"; key_path: string; roots: string[]; windows: boolean }
export interface SshSecrets { password?: string | null; passphrase?: string | null }
export interface Mount { host: string; container: string; read_only: boolean }
export interface DockerConfig { host: string; image: string; name?: string; network: boolean; mounts: Mount[]; memory_mb: number; cpus: number; pids: number; keep: boolean }
export interface HostKey { fingerprint: string; algorithm: string; trusted: boolean }
export interface DispatchOutcome { status: string; content: string; audit_id: string | null }

export const BLANK_SSH: SshProfile = { host: "", port: 22, user: "", auth: "key", key_path: "", roots: [], windows: false };
export const BLANK_DOCKER: DockerConfig = {
  host: "npipe:////./pipe/docker_engine", image: "alpine:3.24", network: false, mounts: [], memory_mb: 2048, cpus: 2, pids: 512, keep: false,
};

export const api4 = {
  list: () => invoke<EnvView[]>("list_environments"),
  setLocalRoots: (roots: string[]) => invoke<string[]>("set_local_roots", { roots }),
  /** This PC: "ask" = paths outside the allowed folders ask each time; "off" = allowed folders only. */
  setLocalOutside: (mode: "ask" | "off") => invoke<string>("set_local_outside", { mode }),
  health: (id: string) => invoke<Health>("environment_health", { id }),
  addSsh: (name: string, profile: SshProfile, secrets: SshSecrets) => invoke<string>("add_ssh_environment", { name, profile, secrets }),
  updateSsh: (id: string, name: string, profile: SshProfile, secrets: SshSecrets | null) => invoke<void>("update_ssh_environment", { id, name, profile, secrets }),
  hostKey: (host: string, port: number) => invoke<HostKey>("ssh_host_key", { host, port }),
  trustHost: (host: string, port: number, fingerprint: string) => invoke<void>("ssh_trust_host", { host, port, fingerprint }),
  addDocker: (name: string, config: DockerConfig) => invoke<string>("add_docker_environment", { name, config }),
  updateDocker: (id: string, name: string, config: DockerConfig) => invoke<void>("update_docker_environment", { id, name, config }),
  dockerStatus: (host: string) => invoke<string>("docker_status", { host }),
  wslStatus: () => invoke<string>("wsl_status"),
  createSandbox: (name: string, network: boolean) => invoke<string>("create_sandbox", { name, network }),
  remove: (id: string) => invoke<void>("delete_environment", { id }),
  setToolPref: (envId: string, tool: string, enabled: boolean, policy: Policy) => invoke<void>("set_builtin_tool_pref", { envId, tool, enabled, policy }),
  runTool: (envId: string, tool: string, args: unknown, conversationId: string | null = null) =>
    invoke<DispatchOutcome>("run_builtin_tool", { envId, tool, arguments: args, conversationId }),
};

export const KIND_LABEL: Record<EnvKind, string> = { local: "This PC", ssh: "SSH", docker: "Docker", sandbox: "Sandbox" };
