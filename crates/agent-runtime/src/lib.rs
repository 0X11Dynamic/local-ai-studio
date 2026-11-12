//! Agent runtime: the tools exposed to a model, the permission-aware
//! dispatcher, and the tool-call loop. Phase 5 adds the task runner,
//! budgets across replies and subagents on top of this.

pub mod agent_loop;
pub mod dispatch;
pub mod task;
pub mod tools;
pub mod xml;

pub use agent_loop::{LoopHooks, LoopOutcome, ToolMode, TurnCtx, TurnEnd, TurnResult, run_reply, turn};
pub use dispatch::{ApprovalRequest, CallResult, DispatchOutcome, ToolEvent, ToolHost, dispatch, validate_args};
pub use tools::{BUILTIN_PREFIX, ExposedTool, Limits, model_tool_name, openai_tools, tools_section};
