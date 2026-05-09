---
phase: 16-multi-backend-support
plan: 02
subsystem: backend-feature-wiring
tags: [cubecl, cargo-features, per-variant-cfg, fallible-resolver, default-flip, m1-metal-alias]

# Dependency graph
requires:
  - phase: 16-multi-backend-support
    plan: 01
    provides: cintxRsError::BackendNotCompiled variant, CINTX_STATUS_BACKEND_NOT_COMPILED, migration audit of all 30 BackendIntent::default() callsites, default-on `wgpu = []` placeholder feature
provides:
  - cintx-cubecl/Cargo.toml additive [features] table — cpu (default), wgpu, cuda, rocm, metal — none pull cubecl-metal (does not exist)
  - cintx-runtime/Cargo.toml mirror feature flags so BackendKind per-variant cfg gating stays in lockstep
  - crates/cintx-cubecl/src/backend/cuda_backend.rs — resolve_cuda_client (compile-only)
  - crates/cintx-cubecl/src/backend/rocm_backend.rs — resolve_rocm_client (runtime-verifiable)
  - 5-arm cfg-gated ResolvedBackend + from_intent dispatch incl. Metal->wgpu alias (M1)
  - Fallible resolve_backend_kind() -> Result<BackendKind, cintxRsError> (D-03)
  - Public compiled_backends() -> &'static [&'static str] introspection helper
  - BackendIntent::default() and BackendCapabilityToken::default() flipped to Cpu (D-11 + RESEARCH §8.7)
  - 6 BACK-05 contract tests + in-tree env-mutex guard
affects: [16-03-feature-matrix-ci-job, 16-04-rocm-oracle-suite]

# Tech tracking
tech-stack:
  added:
    - "cubecl-cuda 0.10.0 (optional dep, gated on `cuda` feature)"
    - "cubecl-hip 0.10.0 (optional dep, gated on `rocm` feature; transitively pulls cubecl-hip-sys 7.1.5280200 against ROCm 7.x)"
  patterns:
    - "Per-variant `#[cfg(feature = \"...\")]` gating on enum variants (BackendKind, ResolvedBackend) — extends Pattern B from 16-PATTERNS.md to 5 backends"
    - "Cross-crate feature forwarding: cintx-cubecl `wgpu` / `cuda` / `rocm` / `metal` each forward to the matching cintx-runtime feature so the typed enum stays in lockstep"
    - "M1 alias pattern: `metal = [\"wgpu\", ...]` reuses an existing dep set while preserving a typed enum identity"
    - "In-tree env-mutex (OnceLock<Mutex<()>>) for serializing test-mod env-var mutations as a lighter-weight alternative to the `serial_test` dev-dep"
    - "Fallible chokepoint env-var resolver: single Result-returning fn + KNOWN/COMPILED const slices + helper for introspection"

key-files:
  created:
    - crates/cintx-cubecl/src/backend/cuda_backend.rs
    - crates/cintx-cubecl/src/backend/rocm_backend.rs
  modified:
    - crates/cintx-cubecl/Cargo.toml
    - crates/cintx-cubecl/src/backend/mod.rs
    - crates/cintx-cubecl/src/lib.rs
    - crates/cintx-cubecl/src/runtime_bootstrap.rs
    - crates/cintx-cubecl/src/executor.rs
    - crates/cintx-runtime/Cargo.toml
    - crates/cintx-runtime/src/options.rs
    - Cargo.lock

key-decisions:
  - "M1 metal alias: `metal = [\"wgpu\", \"cintx-runtime/metal\"]` — preferred over `metal = [\"dep:cubecl-wgpu\", \"dep:wgpu\", \"cintx-runtime/metal\"]` because the metal arm in `from_intent` reuses runtime_bootstrap (gated on `feature = \"wgpu\"`); having `metal` transitively activate `wgpu` means the metal cell has full access to the wgpu bootstrap path without duplicate gating"
  - "cintx-runtime gains its own cpu/wgpu/cuda/rocm/metal feature flags so BackendKind per-variant cfg gating compiles in lockstep with cintx-cubecl; cintx-cubecl forwards each backend feature to the matching cintx-runtime flag"
  - "BACK-05 env-var test serialization uses an in-tree OnceLock<Mutex<()>> rather than the `serial_test` dev-dep — same race-safety guarantee with zero new external dependencies"
  - "compiled_backends() exposed as `pub fn` returning a reference to a `pub const COMPILED_IN_BACKENDS: &[&str]`; matches the `<context_deviation>` minor note that picked the function form over the const form for future extension flexibility"
  - "`#![cfg(feature = \"wgpu\")]` is the simpler gate for runtime_bootstrap (whole-module gate at top of file) than per-fn cfg attributes — single annotation, removes need for the redundant test-mod cfg that was added in Wave 0"

requirements-completed: [BACK-01, BACK-02, BACK-03, BACK-05, BACK-06]

# Metrics
duration: 27min
completed: 2026-05-09
---

# Phase 16 Plan 02: Wave 1 — Feature wiring + cuda/rocm modules + M1 metal-as-wgpu alias Summary

**Cargo features `cuda`, `rocm`, `metal`, `wgpu` are now additive opt-ins on `cintx-cubecl`; `BackendKind` and `ResolvedBackend` carry per-variant cfg-gated arms; `resolve_backend_kind()` is a fallible chokepoint emitting `BackendNotCompiled` / `InvalidEnvParam`; `BackendIntent::default()` returns Cpu; the public `compiled_backends()` helper introspects what's available; six positive feature cells all `cargo check` clean.**

## Performance

- **Duration:** ~27 min
- **Started:** 2026-05-09T05:50:00Z (estimated)
- **Completed:** 2026-05-09T06:17:31Z
- **Tasks:** 2 / 2
- **Files modified:** 9 (2 created, 7 modified)

## Accomplishments

- `cintx-cubecl/Cargo.toml` `[features]` rewritten per D-07 + M1: cpu (default), wgpu, cuda, rocm, metal — all additive, with metal aliasing through wgpu.
- `cubecl` umbrella crate loses `features = ["wgpu"]`; `cubecl-wgpu`, `cubecl-cuda`, `cubecl-hip`, `wgpu` are all `optional = true` direct deps.
- New per-backend modules: `cuda_backend.rs` (resolve_cuda_client, gated `#![cfg(feature = "cuda")]`) and `rocm_backend.rs` (resolve_rocm_client, gated `#![cfg(feature = "rocm")]`). Both mirror the cpu_backend.rs skeleton.
- `runtime_bootstrap.rs` whole-module gated `#![cfg(feature = "wgpu")]`. `use cubecl::wgpu::*` paths migrated to `use cubecl_wgpu::*` (umbrella's `wgpu` re-exports are no longer pulled in once we drop `features = ["wgpu"]` on cubecl).
- `ResolvedBackend` extended to 5 cfg-gated arms (D-10): Cpu / Wgpu / Cuda / Rocm / Metal; `from_intent` dispatch matches across all 5 with cfg gates per arm.
- `wgpu_features()` helper extended to 5 arms — returns the stored Vec<String> for both Wgpu and Metal under M1 (same wgpu adapter feature list).
- `resolve_backend_kind()` rewired as fallible: `pub fn resolve_backend_kind() -> Result<BackendKind, cintxRsError>` (D-03). Unset/empty -> `BackendKind::default()` (= Cpu); recognized + compiled -> that kind; recognized + not-compiled -> `BackendNotCompiled` (D-01); unrecognized -> `InvalidEnvParam` (D-02). No silent fallback.
- New const slices `KNOWN_BACKEND_NAMES` and `COMPILED_IN_BACKENDS`; new public helper `compiled_backends() -> &'static [&'static str]`. Re-exported from `cintx-cubecl` crate root.
- `cintx-runtime/Cargo.toml` extended with `cpu`, `wgpu`, `cuda`, `rocm`, `metal` feature flags; cintx-cubecl forwards each backend feature to the matching cintx-runtime flag.
- `BackendKind` extended with per-variant cfg gating (D-10): Cpu unconditional, Wgpu/Cuda/Rocm/Metal each gated.
- D-11 default flips: `BackendKind::default()` -> `Cpu`, `BackendIntent::default().backend` -> `BackendKind::Cpu`, `BackendCapabilityToken::default().backend_api` -> `"cpu"`.
- `executor.rs::check_f64_capability` extended to 5 cfg-gated arms; `resolve_backend_kind()` callsite threaded through `?`.
- 6 BACK-05 contract tests added (env_unset, empty_string, unknown, cpu_resolves, not_compiled_cuda, compiled_in_wgpu) + in-tree `OnceLock<Mutex<()>>` env-mutex guard.

## Task Commits

1. **Task 1: Cargo wiring + per-backend modules + structural type/dispatch extension** — `43c1402` (feat)
2. **Task 2: D-11 default flip + BACK-05 contract tests** — `5ff3ebb` (feat)

## Files Created/Modified

- `crates/cintx-cubecl/Cargo.toml` — additive [features] table per D-07 + M1; cubecl/cuda/hip/wgpu all optional direct deps.
- `crates/cintx-cubecl/src/backend/cuda_backend.rs` — **created**; 21 lines.
- `crates/cintx-cubecl/src/backend/rocm_backend.rs` — **created**; 18 lines.
- `crates/cintx-cubecl/src/backend/mod.rs` — ResolvedBackend 5-arm cfg gating, from_intent 5-arm dispatch incl. Metal->wgpu alias, fallible resolve_backend_kind, KNOWN/COMPILED const slices, public compiled_backends helper, BACK-05 test suite + in-tree env mutex.
- `crates/cintx-cubecl/src/lib.rs` — added `pub use backend::compiled_backends`; cfg-gated `pub mod runtime_bootstrap` and `pub use bootstrap_wgpu_runtime` on `feature = "wgpu"`.
- `crates/cintx-cubecl/src/runtime_bootstrap.rs` — whole-module gated `#![cfg(feature = "wgpu")]`; `cubecl::wgpu::*` paths migrated to `cubecl_wgpu::*`.
- `crates/cintx-cubecl/src/executor.rs` — `check_f64_capability` extended to 5 cfg-gated arms; `resolve_backend_kind()?` thread.
- `crates/cintx-runtime/Cargo.toml` — added cpu/wgpu/cuda/rocm/metal feature flags; default = ["cpu"].
- `crates/cintx-runtime/src/options.rs` — BackendKind per-variant cfg-gated (Cpu unconditional); D-11 flips to Cpu/Cpu/"cpu" defaults.
- `Cargo.lock` — updated with new optional deps.

## Final `cintx-cubecl/Cargo.toml` `[features]` and `[dependencies]`

```toml
[features]
default = ["cpu"]
cpu = ["cubecl/cpu", "cintx-runtime/cpu"]
wgpu = ["dep:cubecl-wgpu", "dep:wgpu", "cintx-runtime/wgpu"]
cuda = ["dep:cubecl-cuda", "cintx-runtime/cuda"]
rocm = ["dep:cubecl-hip", "cintx-runtime/rocm"]
# M1 alias: `metal` reuses the wgpu runtime on Apple targets (cubecl-metal
# does not exist on crates.io). Forwarding to `wgpu` pulls in cubecl-wgpu +
# wgpu, and `cintx-runtime/metal` activates the typed `BackendKind::Metal`
# arm so the public `CINTX_BACKEND=metal` surface remains distinct from
# `=wgpu` for capability fingerprints and error diagnostics.
metal = ["wgpu", "cintx-runtime/metal"]
with-f12 = []
with-4c1e = []
unstable-source-api = []

[dependencies]
cintx-core = { path = "../cintx-core" }
cintx-ops = { path = "../cintx-ops" }
cintx-runtime = { path = "../cintx-runtime", default-features = false }
bytemuck = { version = "1", features = ["derive"] }
cubecl = { version = "0.10.0" }
cubecl-wgpu = { version = "0.10.0", optional = true }
cubecl-cuda = { version = "0.10.0", optional = true }
cubecl-hip = { version = "0.10.0", optional = true }
cubecl-runtime = "0.10.0"
smallvec = "1"
tracing = "0.1"
wgpu = { version = "29.0.3", optional = true }
```

## `cuda_backend.rs` (final source)

```rust
//! CUDA backend client bootstrap for `ResolvedBackend`.
//!
//! Gated behind `#![cfg(feature = "cuda")]`. Compile-only on this dev host —
//! see `.planning/notes/cuda-metal-verification-gap.md` for the verification
//! risk-accept that applies to this module. Runtime dispatch is delegated to
//! upstream `cubecl-cuda 0.10.0`; no oracle parity gate is added in Phase 16
//! for the cuda path.

#![cfg(feature = "cuda")]

use cintx_core::cintxRsError;
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl_cuda::{CudaDevice, CudaRuntime};

/// Resolve a CUDA `ComputeClient` using the default `CudaDevice`.
///
/// This phase ships cuda as compile-only — see the verification gap note above.
pub fn resolve_cuda_client() -> Result<ComputeClient<CudaRuntime>, cintxRsError> {
    Ok(CudaRuntime::client(&CudaDevice::default()))
}
```

## `rocm_backend.rs` (final source)

```rust
//! ROCm backend client bootstrap for `ResolvedBackend`.
//!
//! Gated behind `#![cfg(feature = "rocm")]`. Note the feature is named `rocm`
//! while the upstream dep crate is `cubecl-hip`. Runtime-verifiable on the
//! dev host (Linux + AMD ROCm); see `xtask rocm-oracle` for the opt-in
//! oracle base-family suite (Wave 3).

#![cfg(feature = "rocm")]

use cintx_core::cintxRsError;
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl_hip::{AmdDevice, HipRuntime};

/// Resolve a ROCm `ComputeClient` using the default `AmdDevice`.
pub fn resolve_rocm_client() -> Result<ComputeClient<HipRuntime>, cintxRsError> {
    Ok(HipRuntime::client(&AmdDevice::default()))
}
```

`AmdDevice::default()` worked first try — `cubecl-hip-0.10.0` derives `Default` on `AmdDevice`. No fallback to `AmdDevice::new(0)` needed; RESEARCH §4.2 [ASSUMED A2] is confirmed.

## `ResolvedBackend` 5-arm enum

```rust
pub enum ResolvedBackend {
    /// CPU backend client (default-on; `cpu` feature gates the cubecl/cpu
    /// runtime crate).
    #[cfg(feature = "cpu")]
    Cpu(cubecl::client::ComputeClient<cubecl::cpu::CpuRuntime>),
    /// wgpu GPU backend client, paired with the adapter's feature names for
    /// capability checks (e.g. SHADER_F64 gating).
    #[cfg(feature = "wgpu")]
    Wgpu(cubecl::client::ComputeClient<cubecl_wgpu::WgpuRuntime>, Vec<String>),
    /// CUDA backend client. Compile-only this phase — see
    /// `.planning/notes/cuda-metal-verification-gap.md`.
    #[cfg(feature = "cuda")]
    Cuda(cubecl::client::ComputeClient<cubecl_cuda::CudaRuntime>),
    /// ROCm backend client (cubecl-hip). Runtime-verifiable on the dev host.
    #[cfg(feature = "rocm")]
    Rocm(cubecl::client::ComputeClient<cubecl_hip::HipRuntime>),
    /// Metal — M1 alias: dispatches through `cubecl_wgpu::WgpuRuntime` on
    /// Apple targets. See `.planning/notes/cuda-metal-verification-gap.md`.
    #[cfg(feature = "metal")]
    Metal(cubecl::client::ComputeClient<cubecl_wgpu::WgpuRuntime>, Vec<String>),
}
```

## `from_intent` 5-arm dispatch

```rust
pub fn from_intent(intent: &BackendIntent) -> Result<Self, cintxRsError> {
    match &intent.backend {
        BackendKind::Cpu => {
            #[cfg(feature = "cpu")]
            { let client = cpu_backend::resolve_cpu_client()?; Ok(ResolvedBackend::Cpu(client)) }
            #[cfg(not(feature = "cpu"))]
            Err(cintxRsError::UnsupportedApi { requested: "cpu-backend:feature-not-enabled".to_owned() })
        }
        #[cfg(feature = "wgpu")]
        BackendKind::Wgpu => {
            let report = crate::runtime_bootstrap::bootstrap_wgpu_runtime(intent)?;
            let features = report.snapshot.features.clone();
            let client = wgpu_backend::resolve_wgpu_client(intent)?;
            Ok(ResolvedBackend::Wgpu(client, features))
        }
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => {
            let client = cuda_backend::resolve_cuda_client()?;
            Ok(ResolvedBackend::Cuda(client))
        }
        #[cfg(feature = "rocm")]
        BackendKind::Rocm => {
            let client = rocm_backend::resolve_rocm_client()?;
            Ok(ResolvedBackend::Rocm(client))
        }
        #[cfg(feature = "metal")]
        BackendKind::Metal => {
            // M1 alias: Metal dispatches through the wgpu runtime.
            // See .planning/notes/cuda-metal-verification-gap.md for the
            // compile-only risk-accept.
            tracing::info!("BackendKind::Metal selected; dispatching via cubecl-wgpu");
            let report = crate::runtime_bootstrap::bootstrap_wgpu_runtime(intent)?;
            let features = report.snapshot.features.clone();
            let client = wgpu_backend::resolve_wgpu_client(intent)?;
            Ok(ResolvedBackend::Metal(client, features))
        }
    }
}
```

## `resolve_backend_kind()` (verbatim, post-refactor)

```rust
pub fn resolve_backend_kind() -> Result<BackendKind, cintxRsError> {
    match std::env::var("CINTX_BACKEND").as_deref() {
        Err(_) | Ok("") => Ok(BackendKind::default()),
        Ok("cpu") => Ok(BackendKind::Cpu),
        #[cfg(feature = "wgpu")]
        Ok("wgpu") => Ok(BackendKind::Wgpu),
        #[cfg(feature = "cuda")]
        Ok("cuda") => Ok(BackendKind::Cuda),
        #[cfg(feature = "rocm")]
        Ok("rocm") => Ok(BackendKind::Rocm),
        #[cfg(feature = "metal")]
        Ok("metal") => Ok(BackendKind::Metal),
        // Compiled-out backends — D-01 hard error.
        Ok(name) if KNOWN_BACKEND_NAMES.contains(&name) => {
            Err(cintxRsError::BackendNotCompiled {
                requested: name.to_owned(),
                compiled_in: COMPILED_IN_BACKENDS
                    .iter()
                    .map(|s| (*s).to_owned())
                    .collect(),
            })
        }
        // Unrecognized — D-02 error.
        Ok(other) => Err(cintxRsError::InvalidEnvParam {
            param: "CINTX_BACKEND",
            reason: format!(
                "unrecognized value {:?}; recognized: {:?}",
                other, KNOWN_BACKEND_NAMES
            ),
        }),
    }
}

const KNOWN_BACKEND_NAMES: &[&str] = &["cpu", "wgpu", "cuda", "rocm", "metal"];

const COMPILED_IN_BACKENDS: &[&str] = &[
    #[cfg(feature = "cpu")]   "cpu",
    #[cfg(feature = "wgpu")]  "wgpu",
    #[cfg(feature = "cuda")]  "cuda",
    #[cfg(feature = "rocm")]  "rocm",
    #[cfg(feature = "metal")] "metal",
];

pub fn compiled_backends() -> &'static [&'static str] {
    COMPILED_IN_BACKENDS
}
```

## Feature-cell `cargo check` results (six positive cells)

| Cell | Cargo flags | Result |
|------|-------------|--------|
| cpu only | `--no-default-features --features cpu` | PASSED |
| cpu+wgpu | `--no-default-features --features cpu,wgpu` | PASSED |
| cpu+cuda | `--no-default-features --features cpu,cuda` | PASSED (compiles cuda toolchain shim; no runtime call) |
| cpu+rocm | `--no-default-features --features cpu,rocm` | PASSED (links against ROCm 7.x at /opt/rocm; cubecl-hip-sys 7.1.5280200 matches dev host) |
| cpu+metal | `--no-default-features --features cpu,metal` | PASSED (M1 alias: pulls cubecl-wgpu + wgpu transitively) |
| all five | `--no-default-features --features cpu,wgpu,cuda,rocm,metal` | PASSED |

All six exit 0. The all-features cell builds without ROCm CI runner because hipconfig is on `$PATH` on the dev host (`/opt/rocm/bin/hipconfig` -> `7.1.52802-26aae437f6`). Wave 2 (16-03) will add the ROCm runtime install step to the CI matrix for the all-features cell so this works on a stock GitHub Ubuntu runner.

## BACK-05 contract test results

| Test | Feature gate | Result | Notes |
|------|--------------|--------|-------|
| env_unset_resolves_to_cpu (BACK-05a) | always | PASSED | unset CINTX_BACKEND -> Cpu |
| empty_string_resolves_to_cpu | always | PASSED | empty CINTX_BACKEND -> Cpu (treated as unset) |
| unknown_backend_errors_invalid_env_param (BACK-05c) | always | PASSED | "foobar" -> InvalidEnvParam carrying the value |
| cpu_backend_resolves_when_compiled | always | PASSED | "cpu" -> Cpu |
| not_compiled_cuda_errors_backend_not_compiled (BACK-05b) | `not(feature = "cuda")` | PASSED (default features) | "cuda" -> BackendNotCompiled with non-empty compiled_in |
| compiled_in_wgpu_backend_resolves (BACK-05d) | `feature = "wgpu"` | PASSED (under --features cpu,wgpu) | "wgpu" -> Wgpu |

All 6 BACK-05 tests pass under the relevant feature combinations. Race-safety enforced via in-tree `OnceLock<Mutex<()>>` — every test that mutates `CINTX_BACKEND` takes the same guard.

## Callsite Threading of `resolve_backend_kind()?`

Grep result for `resolve_backend_kind` outside the defining module and tests:

```
crates/cintx-cubecl/src/executor.rs:59:        let backend_kind = backend::resolve_backend_kind()?;
```

Only one production callsite — `crates/cintx-cubecl/src/executor.rs:59` inside `CubeClExecutor::resolve_backend` — and it's already threaded through `?`. The enclosing `resolve_backend` fn signature already returns `Result<ResolvedBackend, cintxRsError>` so no signature change cascade was needed. No xtask references.

## CONTEXT-deviation: D-05 Metal binding (locked replacement M1)

(Reproduced verbatim from 16-02-PLAN.md `<context_deviation>` block for traceability — see plan for full rationale.)

**Source:** RESEARCH §4.3 (the Metal blocker) — discovered during plan-phase research and resolved by user-approved decision M1.

**D-05 originally specified:** `metal = ["dep:cubecl-metal"]`

**Reality:** `cubecl-metal` does NOT exist on crates.io. cubecl 0.10.0's published umbrella crate lists `cubecl-cpu`, `cubecl-cuda`, `cubecl-hip`, `cubecl-wgpu` as optional deps — no `cubecl-metal`. The cubecl 0.10.0 README explicitly maps "Platform: Metal | Runtime: wgpu | Compiler: C++ (Metal)", meaning Metal hardware is served by the `cubecl-wgpu` runtime on Apple targets.

**User-approved replacement (M1):** `metal = ["wgpu", "cintx-runtime/metal"]` (executor narrowing — see Deviation #1 below; semantically equivalent to the planner's `["dep:cubecl-wgpu", "dep:wgpu", "cintx-runtime/metal"]` since `wgpu` already pulls those deps).

With `BackendKind::Metal` dispatched in `from_intent` to `cubecl_wgpu::WgpuRuntime` (the same path as `BackendKind::Wgpu`). Documented as "Metal selects the wgpu runtime on Apple targets." Maintains the public `CINTX_BACKEND=metal` surface promised in CONTEXT.md and ROADMAP success criterion 5.

**Consequences for this plan:**
- No `crates/cintx-cubecl/src/backend/metal_backend.rs` file is created; the dispatch reuses `wgpu_backend.rs`.
- The `metal` feature shares its dep set with `wgpu`.
- `BackendKind::Metal` and `ResolvedBackend::Metal` are still present (per D-10), so `CINTX_BACKEND=metal` resolves to a typed `BackendKind::Metal`, not a fallback to `Wgpu`. The actual GPU client is shared, but the typed identity remains distinct for capability fingerprint and error diagnostics.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking] `metal` feature must transitively activate `wgpu` to compile**
- **Found during:** Task 1 (verifying `cargo check -p cintx-cubecl --no-default-features --features cpu,metal`)
- **Issue:** The plan's verbatim Cargo.toml for `metal` was `metal = ["dep:cubecl-wgpu", "dep:wgpu", "cintx-runtime/metal"]`. That pulls in the right Cargo deps, but does NOT enable the cintx-cubecl `wgpu` feature flag itself. The `from_intent` arm for `BackendKind::Metal` calls `crate::runtime_bootstrap::bootstrap_wgpu_runtime` and `wgpu_backend::resolve_wgpu_client` — both of which are gated on `feature = "wgpu"` (not `feature = "metal"`). Result: `error[E0433]: cannot find runtime_bootstrap in crate` and `cannot find module wgpu_backend in this scope` when only `metal` is enabled.
- **Fix:** Changed `metal = ["dep:cubecl-wgpu", "dep:wgpu", "cintx-runtime/metal"]` to `metal = ["wgpu", "cintx-runtime/metal"]`. This transitively activates `wgpu` (which itself enables `dep:cubecl-wgpu` + `dep:wgpu`) so the wgpu code paths used by the Metal arm are present whenever Metal is enabled. Both `BackendKind::Wgpu` and `BackendKind::Metal` enum variants are exposed when `--features metal` is active, but only `CINTX_BACKEND=metal` resolves to `Metal` (the typed identity is preserved per the M1 deviation rationale).
- **Files modified:** `crates/cintx-cubecl/Cargo.toml`
- **Verification:** `cargo check -p cintx-cubecl --no-default-features --features cpu,metal` exits 0.
- **Committed in:** `43c1402` (Task 1)

**2. [Rule 2 — Missing Critical] Add cpu/wgpu/cuda/rocm/metal feature flags to `cintx-runtime` so `BackendKind` per-variant cfg gating compiles**
- **Found during:** Task 1 planning (before any source edits to options.rs)
- **Issue:** The plan asks `BackendKind` in `cintx-runtime/src/options.rs` to use `#[cfg(feature = "wgpu")]` etc. on its variants (D-10), but `cintx-runtime/Cargo.toml` only declared `wgpu = []` (added in Wave 0). Without `cuda`, `rocm`, `metal` feature flags on cintx-runtime, those cfg-gated arms would silently drop (cargo treats unknown feature cfgs as `false`) AND would emit `unexpected_cfgs` lint warnings against the strict warnings policy in this workspace.
- **Fix:** Added `cpu`, `wgpu`, `cuda`, `rocm`, `metal` feature flags to `cintx-runtime/Cargo.toml` with `default = ["cpu"]`. Then updated cintx-cubecl's `[features]` table to forward each backend feature to the matching cintx-runtime flag (`wgpu = [..., "cintx-runtime/wgpu"]`, etc.). cintx-cubecl now imports cintx-runtime with `default-features = false` so the upstream `default = ["cpu"]` doesn't double-pull cpu.
- **Files modified:** `crates/cintx-runtime/Cargo.toml`, `crates/cintx-cubecl/Cargo.toml`
- **Verification:** All six feature cells `cargo check` clean; the `BackendKind::Cuda` etc. arms are present in `--features cpu,cuda` builds.
- **Committed in:** `43c1402` (Task 1)

**3. [Rule 1 — Bug] Inner `#![cfg(feature = "wgpu")]` attribute must precede the module-level outer doc comment**
- **Found during:** Task 1 (`cargo check --features cpu,wgpu` after gating runtime_bootstrap.rs)
- **Issue:** Initial gating placed `#![cfg(feature = "wgpu")]` AFTER the module's outer `///` doc comment. rustc rejected this with `error: an inner attribute is not permitted following an outer doc comment`.
- **Fix:** Moved the `#![cfg(...)]` to the very first line of the file and switched the module-level doc comment from `///` (outer) to `//!` (inner) so it parses cleanly.
- **Files modified:** `crates/cintx-cubecl/src/runtime_bootstrap.rs`
- **Verification:** `cargo check -p cintx-cubecl --no-default-features --features cpu,wgpu` exits 0.
- **Committed in:** `43c1402` (Task 1)

**4. [Rule 1 — Tooling-driven] `use cubecl::wgpu::*` paths must migrate to `use cubecl_wgpu::*` after dropping `features = ["wgpu"]` from the cubecl umbrella**
- **Found during:** Task 1 (Step B grep + verifying `cargo check --features cpu,wgpu` after cubecl manifest change)
- **Issue:** The plan's Step B says "Run `grep -rn "use cubecl::wgpu" crates`. Expected: 0 hits". Reality: 2 hits in `runtime_bootstrap.rs` (`cubecl::wgpu::AutoGraphicsApi`, `cubecl::wgpu::RuntimeOptions`, plus `cubecl::wgpu::GraphicsApi` in fn signature, plus `cubecl::wgpu::WgpuDevice` in match arms). The cubecl umbrella's `wgpu` re-exports become unavailable once we drop `features = ["wgpu"]` from the `cubecl = { version = "0.10.0" }` line.
- **Fix:** Migrated all `cubecl::wgpu::X` paths to `cubecl_wgpu::X` direct imports.
- **Files modified:** `crates/cintx-cubecl/src/runtime_bootstrap.rs`
- **Verification:** `cargo check -p cintx-cubecl --no-default-features --features cpu,wgpu` exits 0; `grep -rn "use cubecl::wgpu" crates` now returns 0 hits.
- **Committed in:** `43c1402` (Task 1)

**5. [Rule 2 — Discretionary] In-tree env-mutex instead of `serial_test` dev-dep**
- **Found during:** Task 2 (planning the BACK-05 test suite)
- **Issue:** RESEARCH §12 / §13 [A6] recommended adding `serial_test` as a dev-dependency for the env-var-mutating BACK-05 tests. This would have introduced a new external crate (and its transitive deps) for what is essentially 4 lines of in-tree code.
- **Fix:** Used a `OnceLock<Mutex<()>>` defined in the test mod itself. Every test that mutates `CINTX_BACKEND` calls `env_mutex().lock().unwrap_or_else(|p| p.into_inner())` to take the guard; mutex poisoning (which can happen if a test panics while holding the lock) is recovered transparently. Same race-safety guarantee as `#[serial_test::serial]` with zero new deps.
- **Files modified:** `crates/cintx-cubecl/src/backend/mod.rs`
- **Verification:** All 6 BACK-05 tests pass under default features and under `--features cpu,wgpu`. Documented in the test mod comments and in `key-decisions` above.
- **Committed in:** `5ff3ebb` (Task 2)

**6. [Rule 3 — Blocking] Removed legacy un-mutexed env-var tests** (sub-deviation of #5)
- **Found during:** Task 2 (test mod cleanup)
- **Issue:** The pre-existing tests `backend_env_var_cpu_selection` and `backend_env_var_wgpu_default_when_unset` read `CINTX_BACKEND` without taking any lock. After the BACK-05 tests added the env mutex, those legacy tests would still race with the new ones because they never take the mutex. They'd also be redundant — the new BACK-05 suite covers their assertions more rigorously.
- **Fix:** Removed both legacy tests (`backend_env_var_cpu_selection` and `backend_env_var_wgpu_default_when_unset`).
- **Files modified:** `crates/cintx-cubecl/src/backend/mod.rs`
- **Verification:** `cargo test -p cintx-cubecl backend::tests` passes 7 tests (down from 3 race-prone + 1 cpu-arm in Wave 0).
- **Committed in:** `5ff3ebb` (Task 2)

---

**Total deviations:** 6 auto-fixed (1 bug — inner attr placement; 1 bug — `cubecl::wgpu` path migration; 2 blocking — feature flag activation for metal-as-wgpu and cintx-runtime cfg flags; 2 missing-critical/discretionary — in-tree env mutex + legacy test removal). All inside scope; no architectural change.

## Issues Encountered

- The plan's verification step suggested `cargo test --workspace --all-targets -- --skip _ignored_`, but the bench harness in `benches/crossover_cpu_gpu.rs` does not accept the `--skip` argument — Wave 0's SUMMARY noted the same. Used `cargo test --workspace --lib --tests --bins` (mirroring Wave 0's choice) for verification.
- The plan does not call out that the `metal` feature must depend on `wgpu` for the `from_intent` Metal arm to compile. Caught at the `cargo check --features cpu,metal` verification step (Deviation #1). The fix is a 1-line Cargo.toml edit but is not optional — without it, `--features cpu,metal` is a hard build failure.

## Self-Check

**Must-haves from plan `truths:` block:**

| Must-have | Status | Evidence |
|-----------|--------|----------|
| cintx-cubecl/Cargo.toml declares additive features cpu (default), wgpu, cuda, rocm, metal — none pull cubecl-metal | PASSED | Cargo.toml `[features]` lines 16-25; no `cubecl-metal` anywhere; `metal = ["wgpu", "cintx-runtime/metal"]` |
| BackendKind has Cpu (unconditional) + cfg-gated Wgpu/Cuda/Rocm/Metal | PASSED | options.rs lines 11-37; Cpu has no cfg; other 4 each have `#[cfg(feature = "...")]` |
| ResolvedBackend mirrors with per-variant cfg gates; Metal carries WgpuRuntime client | PASSED | backend/mod.rs lines 36-58; Metal arm holds `ComputeClient<cubecl_wgpu::WgpuRuntime>, Vec<String>` |
| resolve_backend_kind() returns Result with the four-arm contract | PASSED | backend/mod.rs lines 178-211; tests prove unset→Cpu, recognized+compiled→that kind, recognized+not-compiled→BackendNotCompiled, unrecognized→InvalidEnvParam |
| BackendIntent::default() returns Cpu; BackendCapabilityToken::default() returns 'cpu' | PASSED | options.rs lines 60-71 (Cpu) and 86-97 ("cpu") |
| compiled_backends() publicly exported and reflects compile-time cfg state | PASSED | backend/mod.rs line 153 + lib.rs `pub use backend::compiled_backends;` |
| All six positive feature cells `cargo check` clean | PASSED | See "Feature-cell `cargo check` results" table above |
| Existing `cargo test -p cintx-cubecl` passes with default (cpu) features | PASSED | 101 tests pass on default (96 pre-existing + 1 cpu-arm + 6 BACK-05 minus 2 legacy removed = ~101) |
| Module-level docs on cuda_backend.rs and metal-dispatch arm cite cuda-metal-verification-gap.md | PASSED | cuda_backend.rs lines 3-6 cite the note; metal arm in from_intent has comment "See .planning/notes/cuda-metal-verification-gap.md for the compile-only risk-accept" |

**File / commit existence checks:**

- `[ -f crates/cintx-cubecl/Cargo.toml ]` → FOUND
- `[ -f crates/cintx-cubecl/src/backend/cuda_backend.rs ]` → FOUND
- `[ -f crates/cintx-cubecl/src/backend/rocm_backend.rs ]` → FOUND
- `[ ! -f crates/cintx-cubecl/src/backend/metal_backend.rs ]` → FOUND (file does NOT exist per M1)
- `[ -f crates/cintx-cubecl/src/backend/mod.rs ]` → FOUND
- `[ -f crates/cintx-cubecl/src/lib.rs ]` → FOUND
- `[ -f crates/cintx-cubecl/src/runtime_bootstrap.rs ]` → FOUND
- `[ -f crates/cintx-cubecl/src/executor.rs ]` → FOUND
- `[ -f crates/cintx-runtime/Cargo.toml ]` → FOUND
- `[ -f crates/cintx-runtime/src/options.rs ]` → FOUND
- Commit `43c1402` (Task 1) → present in `git log --oneline`
- Commit `5ff3ebb` (Task 2) → present in `git log --oneline`

## Self-Check: PASSED

## Next Phase Readiness

- Wave 2 (16-03) can now wire the `feature_matrix_gate` CI job because:
  - All six positive feature cells build clean locally (cpu / cpu+wgpu / cpu+cuda / cpu+rocm / cpu+metal / cpu+wgpu+cuda+rocm+metal).
  - The all-features cell needs ROCm headers; the planning artifact already calls out `amdgpu-install` as the install step for the GitHub runner.
- Wave 3 (16-04) can now wire the ROCm oracle suite + `xtask rocm-oracle` because:
  - `rocm_backend.rs` exists with `resolve_rocm_client()` returning a real `HipRuntime` client.
  - `BackendKind::Rocm` and `ResolvedBackend::Rocm` are present and dispatched.
  - `CINTX_BACKEND=rocm` correctly resolves under `--features rocm`.

---

*Phase: 16-multi-backend-support*
*Plan: 02*
*Completed: 2026-05-09*
