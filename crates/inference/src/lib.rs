//! Inference backend management.
//!
//! The app never loads model code in-process. Instead it installs official
//! llama.cpp builds (`runtime`), launches `llama-server` as a supervised child
//! process inside a Windows Job Object (`supervisor`, `job`), and talks to it
//! over loopback HTTP (`client`). `config` maps the user's load settings to the
//! flags the installed build actually supports (`capabilities`).

pub mod capabilities;
pub mod chat;
pub mod client;
pub mod config;
pub mod job;
pub mod progress;
pub mod runtime;
pub mod supervisor;
pub mod vcrt;

pub use capabilities::RuntimeCapabilities;
pub use chat::{ChatDelta, ChatOutput, ChatStats, SamplingParams, ToolCall};
pub use client::{Health, LlamaClient, ServerProps};
pub use config::{DEFAULT_FIT_TARGET_MIB, FlashAttention, GpuLayers, LoadConfig, LoadMode};
pub use progress::{DeviceProgress, LoadPhase, LoadProgress, PeerProbe};
pub use runtime::{
    AvailableRuntime, DriverRequirement, GpuHint, InstallProgress, InstalledRuntime, RuntimeManager, cuda_driver_requirement,
    driver_text, parse_driver_version,
};
pub use supervisor::{Engine, EngineEvent, EngineState, EngineStatus, LoadRequest};
