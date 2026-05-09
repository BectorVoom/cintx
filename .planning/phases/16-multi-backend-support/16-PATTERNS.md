# Phase 16: Multi-Backend Support (cuda / rocm / metal) - Pattern Map

**Mapped:** 2026-05-09
**Files analyzed:** 14 (4 new, 10 modified) plus 5 oracle test files extended in-place
**Analogs found:** 14 / 14

All in-scope files have a strong existing analog inside the workspace. No file in this phase requires invention from RESEARCH.md alone — every pattern can be copied from an existing site (`wgpu_backend.rs`, `cpu_backend.rs`, the Phase 13 `InvalidEnvParam` variant, the four existing required CI gates in `compat-governance-pr.yml`, and `oracle_update.rs::run_oom_contract_check` for the xtask helper).

## File Classification

| New / Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---------------------|------|-----------|----------------|---------------|
| **NEW** `crates/cintx-cubecl/src/backend/cuda_backend.rs` | backend bootstrap module | request-response | `crates/cintx-cubecl/src/backend/cpu_backend.rs` | exact |
| **NEW** `crates/cintx-cubecl/src/backend/rocm_backend.rs` | backend bootstrap module | request-response | `crates/cintx-cubecl/src/backend/cpu_backend.rs` | exact |
| **NEW** `xtask/src/rocm_oracle.rs` | xtask command (binary) | batch / process spawn | `xtask/src/oracle_update.rs::run_oom_contract_check` (+ `run_helper_legacy_parity`) | exact (xtask command shape) |
| **NEW** CI job `feature_matrix_gate` (added inside `.github/workflows/compat-governance-pr.yml`) | CI gate (workflow job) | event-driven | existing `oracle_parity_gate` (matrix) + `oom_contract_gate` (single cell) in same file | exact |
| **MODIFIED** `crates/cintx-cubecl/Cargo.toml` | manifest (features table) | config | current `[features]` table at lines 9-14 | self-analog |
| **MODIFIED** `crates/cintx-cubecl/src/backend/mod.rs` | dispatch + env-var resolver | request-response | self (current `from_intent` + `resolve_backend_kind`) | self-analog with shape change |
| **MODIFIED** `crates/cintx-cubecl/src/backend/wgpu_backend.rs` | backend bootstrap module | request-response | self (Metal alias dispatch added per M1) | self-analog |
| **MODIFIED** `crates/cintx-runtime/src/options.rs` | runtime types (`BackendKind`, `BackendIntent`) | config | self (existing enum + Default impls) | self-analog |
| **MODIFIED** `crates/cintx-core/src/error.rs` | error enum (public typed errors) | error-propagation | existing `cintxRsError::InvalidEnvParam` variant (lines 69-70) | exact (Phase 13 precedent) |
| **MODIFIED** `crates/cintx-rs/src/builder.rs` | safe API builder | request-response | self (line 28 `ExecutionOptions::default()` callsite) | self-analog (audit-only) |
| **MODIFIED** `crates/cintx-capi/src/errors.rs` | C ABI status code module | error-propagation | existing `CintxStatus` enum + `status_from_core_error` (lines 9-20, 100-118) | exact |
| **MODIFIED** `crates/cintx-runtime/src/scheduler.rs` (callsite audit) | dispatch | request-response | self (line 64) | self-analog (audit) |
| **MODIFIED** `crates/cintx-runtime/src/planner.rs` (callsite audit, ~10 sites) | planner | request-response | self (lines 570-853) | self-analog (audit) |
| **MODIFIED** `crates/cintx-runtime/src/workspace.rs` (callsite audit, ~5 sites) | workspace | request-response | self (lines 260-383) | self-analog (audit) |
| **MODIFIED** `crates/cintx-rs/src/api.rs` (callsite audit, ~6 sites) | safe API | request-response | self (lines 638-785) | self-analog (audit) |
| **MODIFIED** `crates/cintx-cubecl/src/runtime_bootstrap.rs` (test helper gating) | wgpu bootstrap (test mod) | event-driven | self (lines 279-281) | self-analog |
| **MODIFIED** `crates/cintx-cubecl/src/executor.rs` (`check_f64_capability` per-arm cfg) | executor capability gate | request-response | self (lines 69-83) | self-analog |
| **MODIFIED** `xtask/src/main.rs` (register `rocm-oracle` command) | command dispatcher | request-response | self (existing `Command` enum + `parse_*` + `execute` shape, lines 22-103) | self-analog |
| **MODIFIED** 5 oracle base-family test files (one_electron / two_electron / center_2c2e / center_3c1e / center_3c2e) | integration test | event-driven | `crates/cintx-oracle/tests/one_electron_parity.rs` (lines 31, 302-456) | self-analog (extension) |

---

## Pattern Assignments

### NEW `crates/cintx-cubecl/src/backend/cuda_backend.rs` (backend bootstrap, request-response)

**Analog:** `crates/cintx-cubecl/src/backend/cpu_backend.rs` (entire file, 17 lines)

This is the closest possible match: same role (per-backend client bootstrap behind a feature flag), same data flow (synchronous request-response: build a `ComputeClient` from a default device), same crate position. Cuda is even easier than CPU — `CudaDevice::default()` exists per docs.rs and there is no selector parsing to do this phase.

**Module-gate + imports pattern** (copy from `cpu_backend.rs:1-12`):

```rust
//! CPU backend client bootstrap for `ResolvedBackend`.
//!
//! This entire module is gated behind `#[cfg(feature = "cpu")]` because
//! `cubecl::cpu::CpuRuntime` (and `CpuDevice`) only exist when the `cpu`
//! feature of the `cubecl` crate is enabled.

#![cfg(feature = "cpu")]

use cintx_core::cintxRsError;
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::cpu::{CpuDevice, CpuRuntime};
```

**Core bootstrap pattern** (copy from `cpu_backend.rs:14-17`):

```rust
/// Resolve a CPU `ComputeClient` using the default `CpuDevice`.
pub fn resolve_cpu_client() -> Result<ComputeClient<CpuRuntime>, cintxRsError> {
    Ok(CpuRuntime::client(&CpuDevice::default()))
}
```

**What to copy verbatim:**
- The `#![cfg(feature = "<flag>")]` inner attribute on the module
- The four-line use block (`cintx_core::cintxRsError`, `cubecl::Runtime`, `cubecl::client::ComputeClient`, `cubecl_<backend>::{<Device>, <Runtime>}`)
- The signature shape: `pub fn resolve_<backend>_client() -> Result<ComputeClient<<Runtime>>, cintxRsError>`
- The doc comment shape

**What to invent (cuda-specific):**
- Substitute `cuda` for `cpu` in module path, feature flag, function name
- Imports: `use cubecl_cuda::{CudaDevice, CudaRuntime};` (NOT `cubecl::cuda::*` — `cubecl_cuda` is a sibling crate, NOT a re-export from the umbrella; this is a known cubecl 0.10.0 layout per RESEARCH §3.1)
- [ASSUMED per RESEARCH §8.3] If a cudarc-driver-loadable probe is desired, add a probe call before returning. Otherwise the function is a 1-to-1 transliteration of `cpu_backend.rs`.

**Capability fingerprint (sentinel — Claude's discretion):** This file does NOT compute the fingerprint itself; it only returns the client. The fingerprint is constructed at the `from_intent` call site in `backend/mod.rs`. Use the sentinel `BackendCapabilityToken { adapter_name: "cuda-compile-only".to_owned(), backend_api: "cuda".to_owned(), capability_fingerprint: 0xC0DA_C0DA_C0DA_C0DA }` per RESEARCH §4.1.

---

### NEW `crates/cintx-cubecl/src/backend/rocm_backend.rs` (backend bootstrap, request-response)

**Analog:** Same as cuda — `crates/cintx-cubecl/src/backend/cpu_backend.rs` (lines 1-17).

Mechanical 1-to-1 transliteration with two name changes from the cuda case:

- Feature flag is `rocm` (NOT `hip`) per D-05
- Crate dep is `cubecl-hip` per D-05; imports are `use cubecl_hip::{AmdDevice, HipRuntime};`
- [ASSUMED per RESEARCH §4.2 "AmdDevice::default() exists"] — if `Default` is not impl'd, fall back to `AmdDevice::new(0)` (mirrors `CudaDevice::new(usize)`).

**Module skeleton** (mirror of `cpu_backend.rs`, with rocm names):

```rust
#![cfg(feature = "rocm")]

use cintx_core::cintxRsError;
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl_hip::{AmdDevice, HipRuntime};

pub fn resolve_rocm_client() -> Result<ComputeClient<HipRuntime>, cintxRsError> {
    Ok(HipRuntime::client(&AmdDevice::default()))
}
```

**Capability fingerprint (real):** Unlike cuda/metal which use the sentinel, rocm IS runtime-verifiable on the dev host. Per RESEARCH §4.2 use a simpler-than-wgpu fingerprint: hash `(adapter_name, backend_api="rocm", device_index)` via the same FNV-1a function in `cintx-cubecl::capability`. Defer feature/limit hashing to a follow-up. Mirror the structure used in `runtime_bootstrap.rs:139-163` for wgpu but skip feature enumeration.

---

### MODIFIED `crates/cintx-cubecl/src/backend/wgpu_backend.rs` (Metal alias dispatch per M1)

**Analog:** Self — the `metal` feature is structurally identical to wgpu under M1. The existing file at `wgpu_backend.rs:13-45` is reused unchanged. No new function is added; the dispatch in `backend/mod.rs::from_intent` simply maps `BackendKind::Metal => resolve_wgpu_client(intent)` plus an extra log line.

**Existing pattern to keep** (`wgpu_backend.rs:13-18`):

```rust
pub fn resolve_wgpu_client(
    intent: &BackendIntent,
) -> Result<ComputeClient<WgpuRuntime>, cintxRsError> {
    let device = selector_to_device(&intent.selector)?;
    Ok(WgpuRuntime::client(&device))
}
```

**Recommended optional addition** (if disambiguation in logs is wanted):

```rust
#[cfg(feature = "metal")]
pub fn resolve_metal_client(
    intent: &BackendIntent,
) -> Result<ComputeClient<WgpuRuntime>, cintxRsError> {
    tracing::info!("BackendKind::Metal selected; dispatching via cubecl-wgpu (Metal runtime)");
    resolve_wgpu_client(intent)
}
```

**M1 deviation note:** Plan must include a CONTEXT-deviation entry per RESEARCH §4.3 because D-05's literal `metal = ["dep:cubecl-metal"]` is not implementable; the locked replacement is `metal = ["dep:cubecl-wgpu", "dep:wgpu"]`.

---

### MODIFIED `crates/cintx-cubecl/src/backend/mod.rs` (`resolve_backend_kind` fallible per D-03; `ResolvedBackend` per D-10)

**Analog:** Self — the file is a major rewrite, but every fragment has a current shape to preserve.

**Existing per-arm `#[cfg]` pattern** (`backend/mod.rs:24-25, 34-35, 60-63`) — already used for `Cpu`. Extend to all new arms verbatim:

```rust
pub enum ResolvedBackend {
    Wgpu(cubecl::client::ComputeClient<cubecl_wgpu::WgpuRuntime>, Vec<String>),
    /// CPU backend client (requires `cpu` feature, which is enabled by default).
    #[cfg(feature = "cpu")]
    Cpu(cubecl::client::ComputeClient<cubecl::cpu::CpuRuntime>),
}
```

After D-10 the `Wgpu` arm gains its own `#[cfg(feature = "wgpu")]` (currently it has none — wgpu is implicit) and three new arms join:

```rust
pub enum ResolvedBackend {
    #[cfg(feature = "cpu")]
    Cpu(cubecl::client::ComputeClient<cubecl::cpu::CpuRuntime>),
    #[cfg(feature = "wgpu")]
    Wgpu(cubecl::client::ComputeClient<cubecl_wgpu::WgpuRuntime>, Vec<String>),
    #[cfg(feature = "cuda")]
    Cuda(cubecl::client::ComputeClient<cubecl_cuda::CudaRuntime>),
    #[cfg(feature = "rocm")]
    Rocm(cubecl::client::ComputeClient<cubecl_hip::HipRuntime>),
    #[cfg(feature = "metal")]
    Metal(cubecl::client::ComputeClient<cubecl_wgpu::WgpuRuntime>, Vec<String>),  // M1: alias
}
```

**Existing `from_intent` dispatch shape** (`backend/mod.rs:46-66`):

```rust
pub fn from_intent(intent: &BackendIntent) -> Result<Self, cintxRsError> {
    match &intent.backend {
        BackendKind::Wgpu => {
            let report = crate::runtime_bootstrap::bootstrap_wgpu_runtime(intent)?;
            let features = report.snapshot.features.clone();
            let client = wgpu_backend::resolve_wgpu_client(intent)?;
            Ok(ResolvedBackend::Wgpu(client, features))
        }
        BackendKind::Cpu => {
            #[cfg(feature = "cpu")]
            { let client = cpu_backend::resolve_cpu_client()?; Ok(ResolvedBackend::Cpu(client)) }
            #[cfg(not(feature = "cpu"))]
            Err(cintxRsError::UnsupportedApi {
                requested: "cpu-backend:feature-not-enabled".to_owned(),
            })
        }
    }
}
```

After this phase, every arm follows the per-`#[cfg]` shape (drop the `#[cfg(not(feature=...))]` Err arm because un-compiled variants no longer exist on the enum at all). Pattern per new arm:

```rust
#[cfg(feature = "cuda")]
BackendKind::Cuda => {
    let client = cuda_backend::resolve_cuda_client()?;
    Ok(ResolvedBackend::Cuda(client))
}
```

**Existing `resolve_backend_kind` to REPLACE** (`backend/mod.rs:69-86`):

```rust
pub fn resolve_backend_kind() -> BackendKind {
    match std::env::var("CINTX_BACKEND").as_deref() {
        Ok("cpu") => BackendKind::Cpu,
        Ok("wgpu") | Err(_) => BackendKind::Wgpu,
        Ok(other) => {
            tracing::warn!(
                "Unknown CINTX_BACKEND value {:?}; falling back to wgpu",
                other
            );
            BackendKind::Wgpu
        }
    }
}
```

After D-03 — fallible, no fallback, returns the new error variants. Use the shape verbatim from RESEARCH §9.2 (already cited above by the orchestrator). Two helper consts to add:

```rust
const KNOWN_BACKEND_NAMES: &[&str] = &["cpu", "wgpu", "cuda", "rocm", "metal"];
const COMPILED_IN_BACKENDS: &[&str] = &[
    #[cfg(feature = "cpu")]   "cpu",
    #[cfg(feature = "wgpu")]  "wgpu",
    #[cfg(feature = "cuda")]  "cuda",
    #[cfg(feature = "rocm")]  "rocm",
    #[cfg(feature = "metal")] "metal",
];
```

[Note for planner] D-03 changes the function signature; every callsite that does `let kind = backend::resolve_backend_kind();` must thread the `Result` upward. The known callsite is `executor.rs:57`; planner must grep for additional sites.

---

### MODIFIED `crates/cintx-runtime/src/options.rs` (`BackendKind` D-10; `BackendIntent::default` flip D-11; `BackendCapabilityToken::default` per RESEARCH §8.7)

**Analog:** Self.

**Existing enum to extend** (`options.rs:10-22`):

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// wgpu-backed CubeCL runtime (the primary production backend).
    Wgpu,
    /// CPU execution profile, used for testing and oracle comparison only.
    Cpu,
}

impl Default for BackendKind {
    fn default() -> Self {
        Self::Wgpu
    }
}
```

After D-10/D-11: per-variant `#[cfg]` gating + `Default` returns `Cpu` per D-11. Note `Cpu` arm has NO `#[cfg]` because `cpu` is unconditional per ROADMAP / softened by D-06; the discretion item from CONTEXT places `Cpu` outside the cfg fence. The other four arms each get their own `#[cfg(feature = "...")]`.

**Existing `BackendIntent::default()` to FLIP** (`options.rs:37-44`):

```rust
impl Default for BackendIntent {
    fn default() -> Self {
        Self {
            backend: BackendKind::Wgpu,
            selector: "auto".to_owned(),
        }
    }
}
```

After D-11: `backend: BackendKind::Cpu`. This is the user-visible behavior change called out by RESEARCH §5 — every implicit-default callsite silently switches from wgpu to cpu. Audit task lands BEFORE this flip.

**`BackendCapabilityToken::default()` to FLIP** (`options.rs:63-71`, per RESEARCH §8.7):

```rust
impl Default for BackendCapabilityToken {
    fn default() -> Self {
        Self {
            adapter_name: String::new(),
            backend_api: "wgpu".to_owned(),  // <-- flip to "cpu"
            capability_fingerprint: 0,
        }
    }
}
```

Per §8.7: must flip to `"cpu"` in the same task that flips `BackendIntent::default()` to keep drift detection at `workspace.rs:354-363` symmetric.

---

### MODIFIED `crates/cintx-core/src/error.rs` (new `BackendNotCompiled` variant per D-01)

**Analog:** The existing `InvalidEnvParam` variant (Phase 13) at `crates/cintx-core/src/error.rs:69-70` is the perfect shape match — same crate, same enum, same `thiserror` attribute style, same payload shape (struct variant with named fields). Plus there are existing tests at lines 78-96 that demonstrate the round-trip pattern.

**Existing `InvalidEnvParam` shape to mirror** (`error.rs:69-70`):

```rust
#[error("invalid env parameter {param}: {reason}")]
InvalidEnvParam { param: &'static str, reason: String },
```

**New variant to add (per D-01 + RESEARCH §9.3)** — append to `cintxRsError` after `InvalidEnvParam`:

```rust
#[error("requested backend {requested:?} is not compiled in; compiled-in backends: {compiled_in:?}")]
BackendNotCompiled {
    requested: String,
    compiled_in: Vec<String>,
},
```

**Test pattern to copy** (from `error.rs:79-96`):

```rust
#[test]
fn invalid_env_param_formats_and_matches() {
    let err = cintxRsError::InvalidEnvParam {
        param: "PTR_F12_ZETA",
        reason: "must be non-zero".to_owned(),
    };
    assert!(matches!(
        err,
        cintxRsError::InvalidEnvParam { param: "PTR_F12_ZETA", .. }
    ));
    assert_eq!(
        err.to_string(),
        "invalid env parameter PTR_F12_ZETA: must be non-zero"
    );
}
```

Apply the same shape to a new test `backend_not_compiled_formats_and_matches` that constructs the variant with `requested: "cuda".to_owned(), compiled_in: vec!["cpu".to_owned(), "wgpu".to_owned()]` and asserts the Display string matches `"requested \"cuda\" is not compiled in; compiled-in backends: [\"cpu\", \"wgpu\"]"`.

**Note on type taxonomy:** D-01 specifies `requested: String, compiled_in: Vec<String>` (NOT `&'static str`). This is a deliberate divergence from `InvalidEnvParam` (`&'static str` for `param`) because the requested-backend value comes from a runtime env var, not a compile-time constant.

---

### MODIFIED `crates/cintx-capi/src/errors.rs` (allocate `CintxStatus::BackendNotCompiled` + `CINTX_STATUS_BACKEND_NOT_COMPILED` constant)

**Analog:** Self — the `CintxStatus` enum at `errors.rs:9-20` and the constants block at `errors.rs:28-38` is the established pattern.

**Existing enum to extend** (`errors.rs:7-20`):

```rust
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CintxStatus {
    Success = 0,
    InvalidInput = 1,
    UnsupportedApi = 2,
    UnsupportedRepresentation = 3,
    BufferTooSmall = 4,
    MemoryLimitExceeded = 5,
    AllocationFailed = 6,
    ExecutionFailed = 7,
    NullPointer = 8,
    Panic = 9,
}
```

Add `BackendNotCompiled = 10` (next free integer; never re-use codes per ABI stability).

**Existing constant pattern** (`errors.rs:28-38`):

```rust
pub const CINTX_STATUS_SUCCESS: i32 = CintxStatus::Success as i32;
pub const CINTX_STATUS_INVALID_INPUT: i32 = CintxStatus::InvalidInput as i32;
// ... one line per variant ...
pub const CINTX_STATUS_PANIC: i32 = CintxStatus::Panic as i32;
```

Add: `pub const CINTX_STATUS_BACKEND_NOT_COMPILED: i32 = CintxStatus::BackendNotCompiled as i32;`

**Existing mapping pattern to extend** (`errors.rs:100-118`):

```rust
pub fn status_from_core_error(error: &cintxRsError) -> CintxStatus {
    match error {
        cintxRsError::UnsupportedApi { .. } => CintxStatus::UnsupportedApi,
        // ... other arms ...
        cintxRsError::InvalidEnvParam { .. } => CintxStatus::InvalidInput,
    }
}
```

Add a new arm: `cintxRsError::BackendNotCompiled { .. } => CintxStatus::BackendNotCompiled,`. This match must remain exhaustive — Rust will fail to compile if the new variant is missed, which is the desired safety net.

**Existing exported-constant test to extend** (`errors.rs:264-293`):

```rust
#[test]
fn exported_status_constants_match_enum_codes() {
    assert_eq!(CINTX_STATUS_SUCCESS, CintxStatus::Success.code());
    // ... other assertions ...
    assert_eq!(CINTX_STATUS_PANIC, CintxStatus::Panic.code());
}
```

Add: `assert_eq!(CINTX_STATUS_BACKEND_NOT_COMPILED, CintxStatus::BackendNotCompiled.code());`

---

### MODIFIED `crates/cintx-cubecl/Cargo.toml` (D-07 features + M1 amendment)

**Analog:** Self — the existing `[features]` table at lines 9-14.

**Current shape** (lines 9-26):

```toml
[features]
default = ["cpu"]
cpu = ["cubecl/cpu"]
with-f12 = []
with-4c1e = []
unstable-source-api = []

[dependencies]
cintx-core = { path = "../cintx-core" }
cintx-ops = { path = "../cintx-ops" }
cintx-runtime = { path = "../cintx-runtime" }
bytemuck = { version = "1", features = ["derive"] }
cubecl = { version = "0.10.0", features = ["wgpu"] }
cubecl-wgpu = "0.10.0"
cubecl-runtime = "0.10.0"
smallvec = "1"
tracing = "0.1"
wgpu = "29.0.3"
```

**After D-07 + M1** (per RESEARCH §4.4 + §4.3):

```toml
[features]
default = ["cpu"]
cpu = ["cubecl/cpu"]                      # unchanged
wgpu = ["dep:cubecl-wgpu", "dep:wgpu"]    # NEW: was implicit via umbrella
cuda = ["dep:cubecl-cuda"]
rocm = ["dep:cubecl-hip"]
metal = ["dep:cubecl-wgpu", "dep:wgpu"]   # M1: alias for wgpu (NOT cubecl-metal)
with-f12 = []                             # unchanged
with-4c1e = []                            # unchanged
unstable-source-api = []                  # unchanged

[dependencies]
cubecl = { version = "0.10.0" }                                        # drop features = ["wgpu"]
cubecl-wgpu = { version = "0.10.0", optional = true }                  # was unconditional
cubecl-cuda = { version = "0.10.0", optional = true }                  # NEW
cubecl-hip = { version = "0.10.0", optional = true }                   # NEW
wgpu = { version = "29.0.3", optional = true }                         # was unconditional
# cubecl-runtime, bytemuck, smallvec, tracing — unchanged
```

**Pitfall reminder** (RESEARCH §8.4): all `use cubecl::wgpu::*` paths must change to `use cubecl_wgpu::*`. Already done in current code (grep confirmed 0 hits) but verify in Wave 1.

---

### MODIFIED `crates/cintx-rs/src/builder.rs` (audit only — no code change required, but flag user-visible behavior)

**Analog:** Self — line 28 is `options: ExecutionOptions::default(),`. The line is already correct; what changes is the *meaning* of `ExecutionOptions::default()` after `BackendIntent::default()` flips per D-11.

**Existing line** (`builder.rs:23-30`):

```rust
Self {
    operator,
    representation,
    basis,
    shells,
    options: ExecutionOptions::default(),
}
```

**Action for planner:** Audit task PLAN must include this site as the canonical "user-visible behavior change" anchor. Per RESEARCH §5 (Public API caller alert): "today, every safe-API caller that doesn't explicitly opt out runs on Wgpu. After D-11, every safe-API caller that doesn't explicitly opt out runs on Cpu." Decide: keep as Cpu (recommended — aligns with ROADMAP success criterion 5) and document in CHANGELOG / PHASE-NOTES.md.

---

### MODIFIED `crates/cintx-cubecl/src/executor.rs::check_f64_capability` (per-arm cfg expansion)

**Analog:** Self — current implementation at `executor.rs:69-83` is a 2-arm match.

**Current** (`executor.rs:69-83`):

```rust
fn check_f64_capability(
    &self,
    backend: &ResolvedBackend,
    _plan: &ExecutionPlan<'_>,
) -> Result<(), cintxRsError> {
    match backend {
        ResolvedBackend::Wgpu(_client, _features) => {
            check_shader_f64_in_features(backend.wgpu_features())
        }
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(_client) => Ok(()),
    }
}
```

**After D-10** (per RESEARCH §8.8): expand to 5 arms with `#[cfg]` per arm. Cuda/Rocm return `Ok(())` (cuda f64 capable; rocm runtime-verified accept-with-failure). Metal goes through wgpu_features check (M1: same shape as Wgpu since it owns the same feature list).

```rust
match backend {
    #[cfg(feature = "cpu")]
    ResolvedBackend::Cpu(_) => Ok(()),
    #[cfg(feature = "wgpu")]
    ResolvedBackend::Wgpu(_, _) => check_shader_f64_in_features(backend.wgpu_features()),
    #[cfg(feature = "cuda")]
    ResolvedBackend::Cuda(_) => Ok(()),
    #[cfg(feature = "rocm")]
    ResolvedBackend::Rocm(_) => Ok(()),
    #[cfg(feature = "metal")]
    ResolvedBackend::Metal(_, _) => check_shader_f64_in_features(backend.wgpu_features()),
}
```

The `wgpu_features()` helper at `backend/mod.rs:31-37` must also gain the new arms (return `&[]` for non-wgpu/non-metal, return the stored features for both Wgpu and Metal under M1).

---

### MODIFIED `xtask/src/main.rs` (register `rocm-oracle` command)

**Analog:** Self — the existing `Command` enum, parser, and dispatch shape at `main.rs:22-103`.

**Existing module declaration block** (`main.rs:1-6`):

```rust
mod bench_report;
mod manifest_audit;
mod oracle_covered_update;
mod oracle_update;
mod wgpu_capability_gate;
```

Add: `mod rocm_oracle;`

**Existing command-enum extension pattern** (`main.rs:23-47`): add new variant.

```rust
RocmOracle { profile: Option<String> },
```

**Existing dispatch arm pattern** (`main.rs:63-72`): add the parser binding.

```rust
"rocm-oracle" => parse_rocm_oracle(args)?,
```

**Existing execute arm pattern** (`main.rs:78-103`):

```rust
Command::HelperLegacyParity { profile } => oracle_update::run_helper_legacy_parity(&profile),
```

Add: `Command::RocmOracle { profile } => rocm_oracle::run_rocm_oracle(profile.as_deref()),`

**Existing parser pattern to copy from** (`main.rs:196-215` `parse_helper_legacy_parity`): single optional `--profile` flag, simple shape. The rocm-oracle parser is a near-copy.

**Existing `print_help` pattern** (`main.rs:333-354`): add a one-line mention `"  rocm-oracle [--profile base]   Run ROCm oracle base-family suite (env-gated; requires --features rocm and ROCm 7.x on dev host)"`.

---

### NEW `xtask/src/rocm_oracle.rs` (env-gated cargo test wrapper, batch / process spawn)

**Analog:** `xtask/src/oracle_update.rs::run_oom_contract_check` (lines 146-181) is the closest match — same role (xtask command that builds a `Vec<Vec<&str>>` of cargo args and spawns them), same data flow (process spawn + status check + summary JSON write), same crate.

**Existing process-spawn helper** (`oracle_update.rs:183-192`) — REUSE directly (the helper is already a private fn but the rocm_oracle module sits in the same crate; either lift it to a shared utility or duplicate the 9 lines):

```rust
fn run_cargo_command(args: &[&str]) -> Result<()> {
    let status = Command::new("cargo")
        .args(args)
        .status()
        .with_context(|| format!("spawn cargo {}", args.join(" ")))?;
    if !status.success() {
        bail!("cargo {} failed with status {status}", args.join(" "));
    }
    Ok(())
}
```

**Existing run_oom_contract_check shape** (`oracle_update.rs:146-181`):

```rust
pub fn run_oom_contract_check() -> Result<()> {
    let commands: Vec<Vec<&str>> = vec![
        vec!["test", "-p", "cintx-compat", "raw::tests::memory_limit_failure_keeps_output_slice_unchanged", "--", "--exact"],
        vec!["test", "-p", "cintx-runtime", "workspace::tests::chunk_planner_reports_limit_exceeded_when_no_chunk_can_fit", "--", "--exact"],
    ];
    for args in &commands {
        run_cargo_command(args)?;
    }
    let mut summary = json!({ "status": "ok", "gate": "oom-contract-check", ... });
    let write = write_json_with_fallback(OOM_SUMMARY_REQUIRED_PATH, OOM_SUMMARY_FALLBACK_NAME, &summary)?;
    summary["artifact_write"] = write.to_json();
    rewrite_json(&write.actual_path, &summary)?;
    println!("oom contract summary: {}", write.actual_path.display());
    Ok(())
}
```

**Pattern for `rocm_oracle::run_rocm_oracle`:**

```rust
use anyhow::{Context, Result};
use std::process::Command;

pub fn run_rocm_oracle(profile: Option<&str>) -> Result<()> {
    let profile = profile.unwrap_or("base");
    // Set env-gates for the test harness to opt in (matches RESEARCH §7.1)
    let features = format!("rocm,{profile}");
    let status = Command::new("cargo")
        .env("CINTX_ROCM_ORACLE", "1")
        .env("CINTX_BACKEND", "rocm")
        .args(["test", "-p", "cintx-oracle", "--features", &features, "--", "--ignored"])
        .status()
        .context("spawn cargo test for rocm-oracle")?;
    if !status.success() {
        anyhow::bail!("rocm-oracle suite failed for profile `{profile}`");
    }
    Ok(())
}
```

**What to copy verbatim:**
- `anyhow::{Context, Result}` import + use of `?` for error propagation (xtask uses `anyhow` per CLAUDE.md "App-boundary, xtask, benchmark, and oracle tooling errors")
- `std::process::Command` spawn + `.status()` + `.with_context(...)?` pattern
- `bail!` on non-success

**What is new (for rocm_oracle):**
- `.env("CINTX_ROCM_ORACLE", "1")` and `.env("CINTX_BACKEND", "rocm")` — env-gated trigger per D-15
- The `-- --ignored` filter — runs the `#[ignore]`'d rocm tests (per D-15 + RESEARCH §7.1)
- Optional artifact summary write (mirror `run_oom_contract_check`'s `write_json_with_fallback` if a JSON summary is wanted; otherwise skip — D-15 explicitly says no CI gate, so artifact persistence is optional)

**Note:** Do NOT add this command to any CI workflow per D-15 ("no CI gate"). It's a developer-driven helper only.

---

### MODIFIED 5 oracle test files (per D-15 + RESEARCH §7.3)

Files:
1. `crates/cintx-oracle/tests/one_electron_parity.rs` (covers ovlp + kin + nuc — three tests)
2. `crates/cintx-oracle/tests/two_electron_parity.rs`
3. `crates/cintx-oracle/tests/center_2c2e_parity.rs`
4. `crates/cintx-oracle/tests/center_3c1e_parity.rs`
5. `crates/cintx-oracle/tests/center_3c2e_parity.rs`

**Analog:** `crates/cintx-oracle/tests/one_electron_parity.rs` is the canonical analog for ALL five test files — same crate, same fixture (`build_h2o_sto3g`), same helpers (`count_mismatches`, `nsph`, `collect_*_matrix`).

**Existing module-gate pattern** (`one_electron_parity.rs:31`):

```rust
#![cfg(feature = "cpu")]
```

For rocm extension, do NOT change the existing module gate. Instead, add new test functions inside each file that are gated `#[cfg(feature = "rocm")]` AND `#[ignore]`-marked. They share the `build_h2o_sto3g` fixture by being in the same test crate file.

**Existing test-function pattern** (`one_electron_parity.rs:302-350`):

```rust
#[test]
fn test_int1e_ovlp_sph_h2o_sto3g_parity() {
    let (atm, bas, env) = build_h2o_sto3g();
    let api_id = RawApiId::INT1E_OVLP_SPH;
    let atol = 1e-11_f64;
    let rtol = 1e-9_f64;
    let reference = collect_1e_sph_matrix(api_id, &atm, &bas, &env);
    let observed = collect_1e_sph_matrix(api_id, &atm, &bas, &env);
    let mismatch_count = count_mismatches(&reference, &observed, atol, rtol);
    assert_eq!(mismatch_count, 0, "Oracle parity failed: {mismatch_count} mismatches in int1e_ovlp_sph");
    // physical sanity checks ...
}
```

**Pattern for new ROCm variants** (per D-15 + RESEARCH §7 + the orchestrator's note "atol=1e-12"):

```rust
#[cfg(feature = "rocm")]
#[test]
#[ignore]
fn test_int1e_ovlp_sph_h2o_sto3g_rocm_parity() {
    // Env-gate: panic if user did not explicitly opt-in (RESEARCH §7.1)
    assert_eq!(
        std::env::var("CINTX_ROCM_ORACLE").as_deref(),
        Ok("1"),
        "ROCm oracle must be invoked via `xtask rocm-oracle` (sets CINTX_ROCM_ORACLE=1)"
    );

    let (atm, bas, env) = build_h2o_sto3g();
    let api_id = RawApiId::INT1E_OVLP_SPH;
    // Per orchestrator note: rocm uses tighter atol=1e-12 (matches §7.3 "all five base families at atol=1e-12")
    let atol = 1e-12_f64;
    let rtol = 1e-10_f64;

    let reference = collect_1e_sph_matrix(api_id, &atm, &bas, &env);
    let observed = collect_1e_sph_matrix(api_id, &atm, &bas, &env);
    let mismatch_count = count_mismatches(&reference, &observed, atol, rtol);
    assert_eq!(mismatch_count, 0, "rocm oracle parity failed");
    println!("  PASS: rocm int1e_ovlp_sph mismatch_count=0 at atol=1e-12");
}
```

**What to copy verbatim per file:**
- The `build_*_sto3g` fixture call
- The `collect_*_matrix` invocation
- The `count_mismatches` invocation
- The `assert_eq!(mismatch_count, 0, ...)` shape
- The physical-sanity checks (positive diagonal for overlap/kinetic; negative for nuc; etc.)

**What changes per file (rocm variant):**
- Top: `#[cfg(feature = "rocm")] #[test] #[ignore]` triple-attribute stack (NOT inside the `#![cfg(feature = "cpu")]` module gate — the rocm tests need to be reachable when only `--features rocm` is active without `cpu`)
- Env-gate `assert_eq!` at top to panic if `CINTX_ROCM_ORACLE` is not set
- Tolerance: atol=1e-12, rtol=1e-10 (tighter than cpu tests' 1e-11 / 1e-9)
- [DECISION FOR PLANNER] The current `#![cfg(feature = "cpu")]` module gate at line 31 may need to be widened to `#![cfg(any(feature = "cpu", feature = "rocm"))]` per RESEARCH §7.3. This is a one-line change per file but must be coordinated so the file compiles with `--features rocm` alone (without cpu). Plan task must call this out.

**Test count after extension:** one_electron_parity.rs gains 3 rocm tests (ovlp, kin, nuc); each of the other 4 files gains 1 rocm test. Total: 7 new `#[ignore]`'d rocm tests across 5 files.

---

### NEW CI job `feature_matrix_gate` (added inside `.github/workflows/compat-governance-pr.yml`)

**Analog:** The four existing required gates in the same file. Best matches:
- **Matrix shape:** `oracle_parity_gate` at lines 73-111 (uses `strategy.matrix.profile`).
- **Single-cell shape with conditional steps:** `manifest_drift_gate` at lines 37-71 (simplest gate — checkout, resolve channel, install toolchain, cache, run).

**Existing matrix pattern to copy** (`compat-governance-pr.yml:73-111`, oracle_parity_gate):

```yaml
oracle_parity_gate:
    name: oracle_parity_gate (${{ matrix.profile }})
    runs-on: ubuntu-latest
    strategy:
        fail-fast: false
        matrix:
            profile: [base, with-f12, with-4c1e, "with-f12+with-4c1e"]
    steps:
        - name: Checkout
          uses: actions/checkout@v6
        - name: Resolve pinned Rust channel
          id: rust
          run: |
              python <<'PY'
              import os, tomllib
              from pathlib import Path
              data = tomllib.loads(Path("rust-toolchain.toml").read_text())
              channel = data.get("toolchain", {}).get("channel")
              if not channel:
                  raise SystemExit("failed to resolve channel from rust-toolchain.toml")
              with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
                  fh.write(f"channel={channel}\n")
              PY
        - name: Install pinned Rust toolchain
          uses: dtolnay/rust-toolchain@master
          with:
              toolchain: ${{ steps.rust.outputs.channel }}
        - name: Cache Rust artifacts
          uses: Swatinem/rust-cache@v2
        - name: Run oracle parity gate for ${{ matrix.profile }}
          run: |
              CINTX_BACKEND=cpu cargo run --manifest-path xtask/Cargo.toml -- oracle-compare --profiles "${{ matrix.profile }}" --include-unstable-source false
```

**Pattern for `feature_matrix_gate`** (per D-13/14/16 + RESEARCH §6.1):

Use the same head (Checkout / Resolve pinned Rust channel / Install pinned Rust toolchain / Cache Rust artifacts) verbatim. Replace the matrix axis from `profile` to `cell` with `(cell, features)` pairs. Add a conditional ROCm install step (`if: matrix.cell == 'all-features'`). Replace the final run step with the cargo check + cargo test pair from RESEARCH §6.1 (already cited verbatim by the orchestrator and reproduced here):

```yaml
feature_matrix_gate:
    name: feature_matrix_gate (${{ matrix.cell }})
    runs-on: ubuntu-latest
    strategy:
        fail-fast: false
        matrix:
            include:
                - cell: cpu-only
                  features: ""
                - cell: cpu+wgpu
                  features: "wgpu"
                - cell: all-features
                  features: "wgpu,cuda,rocm,metal"
    steps:
        - name: Checkout
          uses: actions/checkout@v6
        # ... copy Resolve / Install / Cache steps verbatim from oracle_parity_gate ...
        - name: Install ROCm runtime headers (cubecl-hip-sys build script needs hipconfig)
          if: matrix.cell == 'all-features'
          run: |
              wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
              sudo apt-get install -y ./amdgpu-install_6.0.60000-1_all.deb
              sudo amdgpu-install --usecase=rocm --no-dkms -y
              echo "/opt/rocm/bin" >> $GITHUB_PATH
        - name: cargo check
          run: |
              if [ -z "${{ matrix.features }}" ]; then
                cargo check -p cintx-cubecl
              else
                cargo check -p cintx-cubecl --features "${{ matrix.features }}"
              fi
        - name: cargo test (excluding ignored)
          run: |
              if [ -z "${{ matrix.features }}" ]; then
                cargo test -p cintx-cubecl
              else
                cargo test -p cintx-cubecl --features "${{ matrix.features }}"
              fi
```

**What to copy verbatim from existing gates:**
- The `runs-on: ubuntu-latest` line
- The `Checkout / Resolve pinned Rust channel / Install pinned Rust toolchain / Cache Rust artifacts` 4-step preamble (it appears identically in all 4 existing required gates)
- The `python <<'PY' ... PY` heredoc that reads `rust-toolchain.toml` — use the EXACT 13-line block from `manifest_drift_gate:46-59` or `oracle_parity_gate:84-99`. Do not re-write it.
- `fail-fast: false` for matrix gates

**What is new:**
- The `matrix.include:` shape (with `cell` + `features` pairs) — analog: `oracle_parity_gate` uses `matrix.profile:` instead. Both produce per-cell named status checks.
- The conditional ROCm install step (`if: matrix.cell == 'all-features'`)
- Two run steps (`cargo check` then `cargo test`) instead of a single run step

**Branch protection** (RESEARCH §8.6): the new gate's three matrix entries (`feature_matrix_gate (cpu-only)`, `feature_matrix_gate (cpu+wgpu)`, `feature_matrix_gate (all-features)`) must be added to branch protection's required-status-checks list as a separate manual repo settings step. PLAN must call this out.

---

## Shared Patterns

### Pattern A: Per-feature module gate (`#![cfg(feature = "...")]`)

**Source:** `crates/cintx-cubecl/src/backend/cpu_backend.rs:7`

**Apply to:** all new per-backend module files (`cuda_backend.rs`, `rocm_backend.rs`)

```rust
#![cfg(feature = "cpu")]   // <-- inner attribute on module file
```

The whole file ceases to exist when the feature is disabled. This avoids needing per-fn cfg attributes inside the module.

### Pattern B: Per-variant `#[cfg(feature = "...")]` on enum variants

**Source:** `crates/cintx-cubecl/src/backend/mod.rs:24-25` (current Cpu arm)

**Apply to:** `BackendKind` (in `cintx-runtime/src/options.rs`) and `ResolvedBackend` (in `cintx-cubecl/src/backend/mod.rs`)

```rust
pub enum ResolvedBackend {
    /// CPU backend client (requires `cpu` feature, which is enabled by default).
    #[cfg(feature = "cpu")]
    Cpu(cubecl::client::ComputeClient<cubecl::cpu::CpuRuntime>),
    // ...
}
```

Compiler enforces match exhaustiveness per feature combination (D-10 verbose-but-exhaustive choice). Every match site repeats the same `#[cfg(...)]` per arm.

### Pattern C: thiserror struct variant with named fields

**Source:** `crates/cintx-core/src/error.rs:69-70` (`InvalidEnvParam`)

**Apply to:** new `BackendNotCompiled` variant + display-format unit test

```rust
#[error("invalid env parameter {param}: {reason}")]
InvalidEnvParam { param: &'static str, reason: String },
```

Format string interpolates field names; payload struct uses named fields for readability.

### Pattern D: Stable C-ABI status code allocation

**Source:** `crates/cintx-capi/src/errors.rs:9-20, 28-38, 100-118, 264-293`

**Apply to:** `CINTX_STATUS_BACKEND_NOT_COMPILED` rollout

Three coordinated edits per new code:
1. Add variant to `CintxStatus` enum (next free integer)
2. Add `pub const CINTX_STATUS_<NAME>` line in the constants block
3. Add match arm in `status_from_core_error`
4. Add assertion in `exported_status_constants_match_enum_codes` test

The Rust match exhaustiveness check enforces step 3, and the test enforces step 2 — so there's a built-in safety net.

### Pattern E: CI gate preamble (Checkout / Resolve channel / Install toolchain / Cache)

**Source:** `.github/workflows/compat-governance-pr.yml:41-67` (manifest_drift_gate) — but note the EXACT same 4-step preamble appears in `oracle_parity_gate:81-107`, `helper_legacy_parity_gate:117-143`, `oom_contract_gate:155-181`, and `api_value_baseline_gate:197-223`.

**Apply to:** `feature_matrix_gate`

Five steps verbatim:
1. `- name: Checkout` (uses: actions/checkout@v6)
2. `- name: Resolve pinned Rust channel` (with the 13-line python heredoc that reads `rust-toolchain.toml`)
3. `- name: Install pinned Rust toolchain` (uses: dtolnay/rust-toolchain@master, toolchain: ${{ steps.rust.outputs.channel }})
4. `- name: Cache Rust artifacts` (uses: Swatinem/rust-cache@v2)

Then the gate-specific run step(s).

### Pattern F: xtask command spawn `cargo test` with env-gates

**Source:** `xtask/src/oracle_update.rs:183-192` (`run_cargo_command`) plus `oracle_update.rs:146-181` (`run_oom_contract_check`)

**Apply to:** `xtask/src/rocm_oracle.rs::run_rocm_oracle`

Use `std::process::Command::new("cargo")` + `.env(...)` for env-gates + `.args([...])` + `.status()` + `.with_context(|| ...)?` + `if !status.success() { bail!(...) }`. Match `anyhow` style per CLAUDE.md (xtask is app-boundary).

### Pattern G: Oracle test fixture sharing across files

**Source:** `crates/cintx-oracle/tests/one_electron_parity.rs:51-196` (`build_h2o_sto3g`)

**Apply to:** all 5 oracle test files getting rocm extensions

The H2O STO-3G fixture is duplicated per-file in the existing codebase (each test file has its own `build_h2o_sto3g`). Continue this pattern — do NOT factor it into a shared module this phase. Add the new rocm test functions in the SAME file as the existing fixture so they reuse the fixture without further plumbing.

---

## No Analog Found

None. Every file in this phase has a strong analog within the workspace. The Metal backend is the only "no analog" candidate, but per M1 it dispatches through `wgpu_backend.rs` and therefore reuses that file's existing pattern unchanged.

---

## Metadata

**Analog search scope:**
- `crates/cintx-cubecl/src/backend/` (mod.rs, cpu_backend.rs, wgpu_backend.rs)
- `crates/cintx-cubecl/src/runtime_bootstrap.rs`, `executor.rs`
- `crates/cintx-runtime/src/options.rs`, `planner.rs`, `workspace.rs`, `validator.rs`, `scheduler.rs`
- `crates/cintx-core/src/error.rs`
- `crates/cintx-capi/src/errors.rs`
- `crates/cintx-rs/src/builder.rs`, `api.rs`, `error.rs`
- `crates/cintx-oracle/tests/*_parity.rs` (8 files)
- `xtask/src/main.rs`, `oracle_update.rs`
- `.github/workflows/compat-governance-pr.yml`
- `Cargo.toml` files of cintx-cubecl, cintx-rs, cintx-compat, cintx-oracle

**Files scanned:** 21
**Pattern extraction date:** 2026-05-09

## PATTERN MAPPING COMPLETE
