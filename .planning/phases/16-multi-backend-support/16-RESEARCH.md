# Phase 16: Multi-Backend Support (cuda / rocm / metal) - Research

**Researched:** 2026-05-09
**Domain:** Rust workspace Cargo features, cubecl 0.10.0 multi-backend wiring, GitHub Actions CI matrix
**Confidence:** HIGH for cuda/rocm/wgpu wiring; HIGH for the Metal blocker (a deviation from D-05 is required)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Error Variant Taxonomy
- **D-01:** Add a new typed variant `cintxRsError::BackendNotCompiled { requested: String, compiled_in: Vec<String> }` in `cintx-core`. Public, `thiserror`-derived, surfaces through the existing error enum. Diagnostic format must let callers print `requested 'cuda', compiled-in: [cpu, wgpu]`.
- **D-02:** Unrecognized `CINTX_BACKEND` values return `cintxRsError::InvalidEnvParam` (Phase 13 variant) with a payload listing the recognized backend names. No silent warn-and-fallback.
- **D-03:** `resolve_backend_kind()` changes signature to `fn resolve_backend_kind() -> Result<BackendKind, cintxRsError>`. Single fallible chokepoint. The current infallible helper is removed; all callers thread the typed error up. No parallel `_strict()` helper.

#### Cargo Default Features
- **D-04:** wgpu is NEVER default. Every consumer opts in explicitly via `--features wgpu`. Tests that need wgpu use `#[cfg(feature = "wgpu")]` and are silently absent without the flag — they do not auto-skip at runtime.
- **D-05:** Each backend feature pulls only its own cubecl runtime crate as an optional dep, never via the cubecl umbrella crate's features:
  - `cuda = ["dep:cubecl-cuda"]`
  - `rocm = ["dep:cubecl-hip"]` (note: feature is named `rocm`, dep is `cubecl-hip`)
  - `metal = ["dep:cubecl-metal"]`
  - `wgpu = ["dep:cubecl-wgpu", "dep:wgpu"]`
- **D-06:** `cpu` STAYS as a feature flag. `default = ["cpu"]`. `cpu` is undocumented in user-facing docs.
- **D-07:** Final `cintx-cubecl/Cargo.toml` `[features]` shape:
  - `default = ["cpu"]`
  - `cpu = ["cubecl/cpu"]` (kept as-is)
  - `wgpu = ["dep:cubecl-wgpu", "dep:wgpu"]`
  - `cuda = ["dep:cubecl-cuda"]`
  - `rocm = ["dep:cubecl-hip"]`
  - `metal = ["dep:cubecl-metal"]`
  - existing `with-f12`, `with-4c1e`, `unstable-source-api` unchanged.
- **D-08:** Downstream crates (cintx-rs, cintx-oracle, integration tests) that require wgpu add explicit `cintx-cubecl/wgpu` opt-in (or feature-forwarding matching `with-f12` / `with-4c1e` style). Tests that require wgpu are gated `#[cfg(feature = "wgpu")]`.

#### Platform Gating Semantics
- **D-09:** No `target_os` cfg gating in `cintx-cubecl`. Trust upstream `cubecl-{cuda,hip,metal}` to gate themselves.
- **D-10:** Per-variant `#[cfg(feature = "...")]` gating on `BackendKind` and `ResolvedBackend`. All `match` sites repeat the same `#[cfg(...)]` per arm.
- **D-11:** `impl Default for BackendKind` returns `Self::Cpu` — always, infallibly. `BackendIntent::default()` flips its `backend` field to `Cpu`.
- **D-12:** Migration: flipping `BackendIntent::default()` from `Wgpu` to `Cpu` is silently behavior-changing for every implicit caller. PLAN.md must include a task that audits every `BackendIntent::default()` and `ExecutionOptions::default()` callsite in `cintx-rs`, `cintx-oracle`, and the test suite.

#### CI Matrix + ROCm Smoke Scope
- **D-13:** Feature-matrix CI is a 3-cell minimum: `cpu-only`, `cpu+wgpu`, `all-features`.
- **D-14:** Each cell runs `cargo check` + `cargo test` (excluding `#[ignore]` and oracle parity tests).
- **D-15:** ROCm full base-family oracle suite at `atol=1e-12` is implemented but stays opt-in only:
  - tests gated `#[cfg(feature = "rocm")]`
  - tests marked `#[ignore]` so `cargo test --features rocm` does NOT run them by default
  - opt-in trigger via env-gate (e.g., `CINTX_ROCM_ORACLE=1 cargo test --features rocm -- --ignored`)
  - new `xtask rocm-oracle` helper wraps the trigger
  - no CI gate
- **D-16:** New required CI job `feature_matrix_gate` joins the existing required gates. Required for PR merge; fail-closed.

### Claude's Discretion
- Internal organization of new backend modules (`cuda_backend.rs`, `rocm_backend.rs`, `metal_backend.rs`).
- Display/formatting of `BackendNotCompiled` and the `compiled_in` list.
- Whether to expose `compiled_backends() -> &'static [&'static str]` as a public introspection helper.
- Capability-token rules per new backend (sentinel for cuda/metal compile-only; real for rocm).
- Whether `BackendIntent::selector` grammar gains backend-specific syntax.

### Deferred Ideas (OUT OF SCOPE)
- GPU CI runners (NVIDIA, Apple Silicon).
- `CINTX_BACKEND` aliases (e.g., `hip` → `rocm`).
- Per-backend selector grammar (`cuda:0`, `rocm:0`).
- Backend-introspection public API as a separate concern (may land alongside `BackendNotCompiled`).
</user_constraints>

<phase_requirements>
## Phase Requirements

The phase is described in ROADMAP.md (lines 117-133) by 7 success criteria. The planner is expected to allocate stable BACK-NN IDs during plan-phase. The likely-stable derivation is below; the IDs are advisory until the planner locks them.

| Likely ID | Behavior | Research Support |
|----|----------|------------------|
| BACK-01 | `cintx-cubecl/Cargo.toml` exposes additive features `cuda`, `rocm`, `metal`, `wgpu` with the dep-mapping in D-07 | §3 (per-backend implementation), §7 (Pitfalls — Metal) |
| BACK-02 | `BackendKind` and `ResolvedBackend` extend with `Cuda`, `Rocm`, `Metal` variants, each `#[cfg(feature = "...")]`-gated | §3 (per-backend), §4 (migration audit), §1 (executive summary) |
| BACK-03 | `cargo check` with every non-empty subset of `{cuda, rocm, metal, wgpu}` builds cleanly on the dev host (Linux, AMD ROCm) | §2 (resolution answer), §5 (CI matrix design), §7 (Metal fallback) |
| BACK-04 | `cargo test --features rocm` runs at least one oracle smoke test under `CINTX_BACKEND=rocm` and matches existing tolerances | §6 (ROCm oracle suite design) |
| BACK-05 | `CINTX_BACKEND=<name>` selects compiled-in backend at runtime; unset → `cpu`; non-compiled → typed `BackendNotCompiled`; unknown → typed `InvalidEnvParam` | §3 (resolve_backend_kind redesign), §1 (executive summary) |
| BACK-06 | `cuda` and `metal` are documented as compile-only; no oracle parity gate added for them | §6 (out-of-scope), §7 (risk-accept) |
| BACK-07 | Feature matrix exercised in CI on existing runners (no new GPU runners) | §5 (feature_matrix_gate.yml design) |
</phase_requirements>

## 1. Executive Summary

The phase is mostly mechanical: extend the existing `wgpu`/`cpu` backend dispatch in `cintx-cubecl` with three new arms (`cuda`, `rocm`, `metal`), wire each behind a feature flag, gate variants per-feature on the public `BackendKind` and `ResolvedBackend` enums, and convert `resolve_backend_kind()` to a fallible `Result<BackendKind, cintxRsError>`. The lockfile already resolves `cubecl-cuda 0.10.0` and `cubecl-hip 0.10.0` (transitively via the `cubecl` umbrella), so adding them as direct optional deps is graph-neutral. `cudarc` (transitive via `cubecl-cuda`) defaults to dynamic loading, so cuda compiles without the CUDA toolkit installed; `cubecl-hip-sys` requires `hipconfig` on `PATH` at compile time, which the dev host already satisfies.

**The single hard finding:** `cubecl-metal` does NOT exist as a published crate. The cubecl 0.10.0 README explicitly maps Metal hardware to the `cubecl-wgpu` runtime ("Platform: Metal | Runtime: wgpu"). D-05's `metal = ["dep:cubecl-metal"]` cannot be implemented as written. The planner MUST resolve this before locking BACK-01 tasks. Two options are in §7; the recommended option is to keep the `metal` feature as an alias for the wgpu runtime on Apple targets (no new dep) so the public feature surface in CONTEXT.md is preserved.

The migration audit for D-12 is bigger than it looks: 30+ callsites of `BackendIntent::default()` / `ExecutionOptions::default()` exist across `cintx-runtime`, `cintx-rs`, `cintx-cubecl`, `cintx-compat`, all of which currently get an implicit `Wgpu` backend and will silently switch to `Cpu` when the default flips. Most are already inside `#[cfg(test)]` or test-fixture code that doesn't actually need a GPU; a handful in `runtime_bootstrap.rs` and `executor.rs` are wgpu-specific and will need explicit `BackendIntent { backend: BackendKind::Wgpu, .. }` plus `#[cfg(feature = "wgpu")]` gating. Wave 0 of the plan should land the audit first to keep behavior changes mechanical.

CI: the four "existing required gates" referenced in CONTEXT.md actually live as four separate jobs inside one workflow file (`compat-governance-pr.yml`) — `manifest_drift_gate`, `oracle_parity_gate`, `helper_legacy_parity_gate`, `oom_contract_gate`, plus `api_value_baseline_gate`. The new `feature_matrix_gate` should be added as a fifth (sixth counting api_value_baseline) job in the same file, not a new workflow file.

**Primary recommendation:** Land the migration-audit task in Wave 0 (before any feature wiring), implement cuda/rocm in Wave 1 (mechanical pattern match against `wgpu_backend.rs`), resolve the Metal blocker by a `cintx-cubecl` design choice in Wave 1 too (recommend: drop `dep:cubecl-metal` and gate `metal` to forward to `cubecl-wgpu` on macOS), wire CI in Wave 2.

## 2. Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Env-var resolution (`CINTX_BACKEND`) | Library — `cintx-cubecl::backend` | — | Stays at the existing single chokepoint; only the signature changes |
| BackendKind type definition | Library — `cintx-runtime::options` | — | Already lives there; cfg gating on variants follows the existing `Cpu` pattern |
| Per-backend client bootstrap | Library — `cintx-cubecl::backend::*_backend` | — | Mirrors `wgpu_backend.rs` / `cpu_backend.rs` modules |
| Capability fingerprint | Library — `cintx-cubecl::capability` | — | Existing module; cuda/metal use a sentinel, rocm gets a real one |
| Feature flag forwarding (downstream) | Library — `cintx-rs/Cargo.toml`, `cintx-compat/Cargo.toml`, `cintx-oracle/Cargo.toml` | — | Existing forwarding pattern for `with-f12`, `with-4c1e` |
| ROCm oracle suite | Test — `cintx-oracle/tests` | Tooling — `xtask rocm-oracle` | Runs against the `cintx-compat` raw API, not the cubecl crate directly |
| Feature-matrix CI gate | CI — `.github/workflows/compat-governance-pr.yml` | — | New job in the existing workflow file, not a new file |

## 3. Closing the Open Question

The `.planning/research/questions.md` entry asks four sub-questions. Each is answered with version numbers and a confidence tag.

### 3.1 Does `cargo check --no-default-features --features cuda,rocm,metal,wgpu` resolve and build?

**Verdict:** Resolves cleanly for `cuda`, `rocm`, `wgpu`. **Fails as written for `metal`** because `cubecl-metal` is not a published crate. [VERIFIED: `Cargo.lock` lines 642-960; cubecl 0.10.0 umbrella declares `cubecl-cpu`, `cubecl-cuda`, `cubecl-hip`, `cubecl-wgpu` as optional deps; no `cubecl-metal` listed]

| Direct dep | Version | Source confidence |
|-----------|---------|-------------------|
| `cubecl-wgpu` | 0.10.0 | [VERIFIED: already in `Cargo.lock` line 928, already a direct dep at `crates/cintx-cubecl/Cargo.toml:22`] |
| `cubecl-cuda` | 0.10.0 | [VERIFIED: already in `Cargo.lock` line 765 transitively via `cubecl` umbrella; promoting to optional direct dep does not perturb the graph] |
| `cubecl-hip` | 0.10.0 | [VERIFIED: already in `Cargo.lock` line 783 transitively via `cubecl` umbrella] |
| `cubecl-metal` | DOES NOT EXIST | [VERIFIED: cubecl 0.10.0 README "Platform: Metal | Runtime: wgpu | Compiler: C++ (Metal)"; tracel-ai/cubecl `crates/` directory listing shows no `cubecl-metal` directory; cubecl umbrella does not list it as an optional dep] |

The cubecl umbrella crate at `cintx-cubecl/Cargo.toml:21` (`cubecl = { version = "0.10.0", features = ["wgpu"] }`) **already** transitively pulls `cubecl-cpu`, `cubecl-cuda`, `cubecl-hip` because they are unconditional crate-level deps of the `cubecl` umbrella. Promoting them to additional optional direct deps under our own `cuda` / `rocm` features is purely a feature-gate change — the lock graph is unchanged. This is the strongest possible evidence that resolution will succeed.

### 3.2 What system-level deps does each require to **compile** (not run)?

| Crate | Compile-time system dep | Runtime system dep | Confidence |
|-------|------------------------|--------------------|------------|
| `cubecl-cuda` 0.10.0 → `cudarc` 0.19.4 | **None** (cudarc default feature is `dynamic-loading`; no CUDA toolkit needed at build time) | CUDA driver libraries on the target host | [VERIFIED: cudarc README confirms `dynamic-loading` is default and "will not require any libraries to be present at build time"] |
| `cubecl-hip` 0.10.0 → `cubecl-hip-sys` 7.1.5280200 | **`hipconfig` binary on PATH** at build time. Build script uses `hipconfig` to detect HIP install location. `HIP_PATH` env var can override. | ROCm runtime libraries (`amdhip64`, `hiprtc`) | [VERIFIED: cubecl-hip-sys README per WebSearch result; ROCm is already installed on the dev host per CONTEXT.md] |
| `cubecl-wgpu` 0.10.0 → `wgpu` 29.0.3 | None beyond standard Rust toolchain on Linux/macOS/Windows | Vulkan/DX12/Metal driver | [VERIFIED: already builds on the dev host today] |
| `cubecl-metal` (does not exist) | n/a | n/a | [VERIFIED: not on crates.io] |

**Concrete consequence for D-13's `all-features` CI cell:** GitHub Actions `ubuntu-latest` does NOT have `hipconfig` installed by default. The `feature_matrix_gate` `all-features` job needs either (a) install ROCm before the cargo step, (b) skip `rocm` from the `all-features` cell on CI, or (c) accept that `all-features` becomes red on CI and demote it to "best-effort dev-host-only." Recommend (a): install the ROCm 6.x runtime — only `hipconfig` and `rocm-dev` packages are needed, ~1.5 GB, single apt-get step.

### 3.3 Are any crates platform-gated upstream?

| Crate | Builds on Linux x86_64? | Builds on macOS? | Builds on Windows? |
|-------|------------------------|------------------|--------------------|
| `cubecl-cuda` | YES — explicitly listed as `x86_64-unknown-linux-gnu` supported on docs.rs | Likely (cudarc supports both) | YES |
| `cubecl-hip` | YES (the supported target) | NO — ROCm/HIP is Linux/Windows only | YES with HIP SDK for Windows |
| `cubecl-wgpu` | YES (Vulkan) | YES (Metal) | YES (DX12/Vulkan) |
| `cubecl-cpu` (Cpu via cubecl/cpu) | YES | YES | YES |

[VERIFIED: cubecl 0.10.0 README platform matrix; docs.rs target listings for cubecl-cuda 0.10.0]

**No upstream `target_os` gating** is added by these crates' `Cargo.toml` files — they will attempt to build on any target and fail at link time if system libs are missing. This matches D-09's "trust upstream" decision: we don't need to add cfg gating ourselves; failures surface at `cargo check` if they happen.

### 3.4 Do transitive `wgpu` / `naga` / `bytemuck` pins agree?

[VERIFIED: `Cargo.lock` inspection]

| Transitive | Pinned version | Pulled by | Conflict? |
|-----------|---------------|-----------|-----------|
| `wgpu` | 29.0.3 | `cubecl-wgpu` (transitive), `cintx-cubecl` (direct) | **No conflict** — only one version |
| `naga` | 29.0.3 | wgpu's deps | No conflict |
| `wgpu-core`, `wgpu-hal`, `wgpu-types`, `wgpu-naga-bridge` | 29.0.3 (uniform) | wgpu | No conflict |
| `bytemuck` | 1.25.0 | many crates incl. cubecl-{cuda,hip,wgpu,cpu,common,core,std} | No conflict — all use `1.x` semver-compatible |
| `bytemuck_derive` | 1.10.2 | bytemuck | No conflict |

`cubecl-cuda` and `cubecl-hip` do NOT depend on `wgpu` or `naga` at all — they share only `bytemuck`, `cubecl-common`, `cubecl-core`, `cubecl-runtime`, `half`, `serde`, all already pinned in the lockfile. There is **zero risk** of version skew when adding the cuda and rocm features.

## 4. Per-Backend Implementation Notes

### 4.1 CUDA backend (`cuda_backend.rs`)

**`Cargo.toml` shape (additive to existing):**
```toml
[dependencies]
cubecl-cuda = { version = "0.10.0", optional = true }

[features]
cuda = ["dep:cubecl-cuda"]
```

**Runtime client type:** `cubecl::client::ComputeClient<cubecl_cuda::CudaRuntime>`

**Module skeleton (mirrors `cpu_backend.rs`):**
```rust
//! CUDA backend client bootstrap for `ResolvedBackend`.
//!
//! Gated behind `#[cfg(feature = "cuda")]` because `cubecl_cuda::CudaRuntime`
//! only exists when the `cuda` feature is enabled.

#![cfg(feature = "cuda")]

use cintx_core::cintxRsError;
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl_cuda::{CudaDevice, CudaRuntime};

/// Resolve a CUDA `ComputeClient` using the default `CudaDevice`.
pub fn resolve_cuda_client() -> Result<ComputeClient<CudaRuntime>, cintxRsError> {
    Ok(CudaRuntime::client(&CudaDevice::default()))
}
```

[VERIFIED: `cubecl_cuda::{CudaRuntime, CudaDevice}` exports per docs.rs/cubecl-cuda/0.10.0; `CudaDevice::default()` exists and returns index 0]

**Capability fingerprint:** Use a fixed sentinel — `BackendCapabilityToken { adapter_name: "cuda-compile-only", backend_api: "cuda", capability_fingerprint: 0xC0DA_C0DA_C0DA_C0DA }`. Phase 5/6's D-08 drift detection is by-design unverifiable for compile-only backends; the sentinel is a stable, recognizable placeholder. `[ASSUMED]` That a fixed sentinel is acceptable to D-08's drift contract — confirm with a one-line check in the Phase 5/6 D-08 contract test before locking the value.

**Build-time gotchas:**
- None on Linux x86_64. cudarc dynamic-loading default eliminates the toolkit-at-build-time requirement.
- On `ubuntu-latest` CI, `cargo check --features cuda` succeeds without installing CUDA. **VERIFY in a scratch branch before committing the plan task** by running `cargo check -p cintx-cubecl --features cuda` locally.

### 4.2 ROCm backend (`rocm_backend.rs`) — feature name `rocm`, dep name `cubecl-hip`

**`Cargo.toml` shape:**
```toml
[dependencies]
cubecl-hip = { version = "0.10.0", optional = true }

[features]
rocm = ["dep:cubecl-hip"]
```

**Runtime client type:** `cubecl::client::ComputeClient<cubecl_hip::HipRuntime>`. Device type is `cubecl_hip::AmdDevice` (NOT `HipDevice`).

[VERIFIED: docs.rs/cubecl-hip/0.10.0 — top-level exports `HipRuntime` (re-exported from `runtime`) and `AmdDevice` (re-exported from `device`)]

**Module skeleton:**
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

[ASSUMED] `AmdDevice::default()` exists. WebFetch on `docs.rs/cubecl-hip/0.10.0/cubecl_hip/struct.AmdDevice.html` returned 404. Confirm by `cargo check -p cintx-cubecl --features rocm` in a scratch branch; if `Default` is not impl'd, fall back to `AmdDevice::new(0)` (mirroring `CudaDevice::new(usize)`).

**Capability fingerprint:** Real fingerprint is feasible because rocm IS runtime-verifiable on the dev host. Recommend a **simpler fingerprint than wgpu** initially: hash `(adapter_name, backend_api="rocm", device_index)` via the same FNV-1a function in `cintx-cubecl::capability`. Defer richer feature/limit hashing to a follow-up; this phase's D-08 drift detection just needs a fingerprint that changes when the device changes.

**Build-time gotchas:**
- `cubecl-hip-sys` build script requires `hipconfig` on PATH. The dev host already has ROCm installed; CI does not. The `feature_matrix_gate` `all-features` cell needs ROCm install (~1.5 GB single apt-get).
- An explicit version range pin like `cubecl-hip = "0.10.0"` together with `cubecl-hip-sys 7.1.5280200` in the lockfile may be sensitive to ROCm version on the host. The `cubecl-hip-sys` README notes "the crates follow the same versioning as HIP." If the dev host's ROCm version doesn't match `7.1.x`, a `cubecl-hip-sys` feature flag pin may be required. Validate during Wave 1 by inspecting `hipconfig --version` output and the lockfile pin; if mismatched, flag as a Wave-1 unblock task.

### 4.3 Metal backend — **D-05 BLOCKER**

**`Cargo.toml` shape as written in D-07:**
```toml
metal = ["dep:cubecl-metal"]
```

**This does not work:** [VERIFIED: cubecl-metal does not exist on crates.io; cubecl 0.10.0 README says Metal goes via the wgpu runtime]

**Recommended resolution (preserves the `metal` feature surface on CINTX_BACKEND):**

Option M1 (recommended): `metal` is an alias for `wgpu` on Apple targets.
```toml
[features]
metal = ["dep:cubecl-wgpu", "dep:wgpu"]
```
And in `BackendKind`:
```rust
#[cfg(feature = "metal")] Metal,
```
With `Metal` dispatched in `from_intent` to the same wgpu bootstrap (`cubecl_wgpu::WgpuRuntime`). Documented as "Metal selects the wgpu runtime on Apple targets." Users who set `CINTX_BACKEND=metal` get Metal hardware via wgpu's Metal backend. Maintains the public feature/env-var surface promised in CONTEXT.md.

Option M2: Drop the `metal` feature entirely; document that on macOS, users should set `--features wgpu` and `CINTX_BACKEND=wgpu`. Smallest code surface, but contradicts D-05/D-07 enumeration of the metal feature.

Option M3: Keep `metal = []` as an empty feature flag (no dep) just for symmetry, and have `BackendKind::Metal` dispatch to wgpu under the hood. Same effective behavior as M1 but with a stub feature.

**Recommendation for the planner:** Adopt M1. It minimizes user-facing changes from CONTEXT.md (the public CINTX_BACKEND vocabulary still includes `metal`), and the implementation is a 5-line edit on top of the wgpu work. Add a CONTEXT-deviation entry in `16-PLAN.md` so the divergence from D-05's literal text is logged.

**Capability fingerprint:** Same as wgpu (real, computed at bootstrap from the actual Metal adapter). Not the cuda-style sentinel.

### 4.4 WGPU backend (preserved, but no longer default)

The `wgpu` feature is broken out from the existing umbrella `cubecl = { features = ["wgpu"] }` to a direct optional dep. The migration:

Before:
```toml
[dependencies]
cubecl = { version = "0.10.0", features = ["wgpu"] }
cubecl-wgpu = "0.10.0"
wgpu = "29.0.3"
```
After:
```toml
[dependencies]
cubecl = { version = "0.10.0" }                # no umbrella wgpu feature
cubecl-wgpu = { version = "0.10.0", optional = true }
wgpu = { version = "29.0.3", optional = true }
[features]
wgpu = ["dep:cubecl-wgpu", "dep:wgpu"]
```

[VERIFIED: cubecl 0.10.0 has `wgpu` as one of its optional features per docs.rs/cubecl/0.10.0 deps list — removing `features = ["wgpu"]` from the umbrella import is safe and necessary because we now want wgpu opt-in.]

The existing `wgpu` references inside `runtime_bootstrap.rs`, `wgpu_backend.rs`, `executor.rs::check_f64_capability`, and `kernels/*` need to gain `#[cfg(feature = "wgpu")]` gates. Most of `runtime_bootstrap.rs` is wgpu-specific and should be moved (or feature-gated) entirely.

## 5. Migration Audit Checklist (D-12)

Flipping `BackendIntent::default()` from `Wgpu` to `Cpu` mutates the implicit behavior of every callsite that uses `..ExecutionOptions::default()` or `BackendIntent::default()`. The audit must classify each of the 30 sites below into one of three buckets:

- **(K) Keep Cpu** — test/fixture code that doesn't actually exercise GPU paths and benefits from the cpu default.
- **(W) Make Wgpu explicit** — test/fixture code that's currently exercising wgpu and must keep doing so.
- **(F) Feature-gate** — code that should only compile when `wgpu` is enabled.

### Audit grep commands (planner: copy these into the audit task)

```bash
# All BackendIntent::default() and field-update callsites
grep -rn "BackendIntent::default\|BackendIntent {" crates xtask 2>/dev/null

# All ExecutionOptions::default() and ..default() spreads (catches implicit BackendIntent)
grep -rn "ExecutionOptions::default\|ExecutionOptions {" crates xtask 2>/dev/null

# All explicit BackendKind::Wgpu uses (most must become #[cfg(feature = "wgpu")] gated)
grep -rn "BackendKind::Wgpu\|BackendKind::Cpu" crates xtask 2>/dev/null

# All ResolvedBackend match arms (must add #[cfg(feature = ...)] per arm)
grep -rn "ResolvedBackend::" crates 2>/dev/null

# Tests that silently require wgpu (currently always run)
grep -rn "WgpuRuntime\|cubecl_wgpu\|wgpu_backend\|bootstrap_wgpu_runtime" crates 2>/dev/null
```

### Callsite inventory (verified via the greps above on 2026-05-09)

| File | Line | Site | Likely classification |
|------|------|------|----------------------|
| `crates/cintx-runtime/src/scheduler.rs` | 64 | `BackendIntent::default()` | (K) Cpu — scheduler default doesn't pre-bind a backend |
| `crates/cintx-runtime/src/options.rs` | 30, 37, 40 | `BackendIntent` definition + Default impl | source of the change |
| `crates/cintx-runtime/src/planner.rs` | 570, 616, 642, 663, 683, 824, 853 | `..ExecutionOptions::default()` (test code) | (K) Cpu — planner unit tests don't exercise dispatch |
| `crates/cintx-runtime/src/planner.rs` | 723–732, 761–765, 789–793 | explicit `BackendIntent { Wgpu, .. }` and `BackendIntent { Cpu, .. }` | (W)/(F) — keep explicit; feature-gate the Wgpu ones |
| `crates/cintx-runtime/src/workspace.rs` | 260, 281, 306, 329 | `..ExecutionOptions::default()` (test code) | (K) Cpu |
| `crates/cintx-runtime/src/workspace.rs` | 354–363, 383 | explicit `BackendIntent { Wgpu, .. }` (test for query/eval drift) | (W) — feature-gate behind `#[cfg(feature = "wgpu")]` |
| `crates/cintx-rs/src/builder.rs` | 28 | `ExecutionOptions::default()` | **(K) Cpu — but this is the public builder default; user-visible** |
| `crates/cintx-rs/src/api.rs` | 638, 666, 700, 721, 764, 785 | `ExecutionOptions::default()` (test code, most likely doctest paths) | review each — most are (K) Cpu |
| `crates/cintx-cubecl/src/runtime_bootstrap.rs` | 279–281 | test helper `wgpu_intent()` | (W) — keep explicit, gate test mod with `#[cfg(feature = "wgpu")]` |
| `crates/cintx-cubecl/src/backend/mod.rs` | 95 | test `BackendIntent { Cpu, .. }` | (K) Cpu — already correct |
| `crates/cintx-cubecl/src/specialization.rs` | 135 | `ExecutionOptions::default()` | (K) Cpu |
| `crates/cintx-compat/src/raw.rs` | 951 | `ExecutionOptions::default()` (raw API helper) | review — likely (K) Cpu (raw API users own their backend choice) |
| `crates/cintx-cubecl/src/transfer.rs` | 193 | `&ExecutionOptions::default()` (test) | (K) Cpu |
| `crates/cintx-cubecl/src/kernels/mod.rs` | 160 | `&ExecutionOptions::default()` (test) | (K) Cpu |
| `crates/cintx-cubecl/src/executor.rs` | 58 | `BackendIntent { backend: backend_kind, ... }` (production path — env-var driven) | unchanged — env var controls |
| `crates/cintx-cubecl/src/executor.rs` | 283 | `&ExecutionOptions::default()` (test) | (K) Cpu |

**Public API caller alert:** `crates/cintx-rs/src/builder.rs:28` is the safe-API `Builder::default()`. It calls `ExecutionOptions::default()` without overriding `backend_intent`. Today, every safe-API caller that doesn't explicitly opt out runs on Wgpu. After D-11, every safe-API caller that doesn't explicitly opt out runs on Cpu. **This is a user-visible behavior change** and must be called out in PHASE-NOTES.md and in the public CHANGELOG entry. The audit task PLAN must include a one-line check on this site.

**Production audit path:** Treat `runtime_bootstrap.rs::tests` as the "where does wgpu live in tests" canary. Anything that imports `bootstrap_wgpu_runtime` or `WgpuRuntime` MUST be gated `#[cfg(feature = "wgpu")]` after this phase, otherwise the no-features build breaks.

## 6. CI Feature-Matrix Gate Design (D-13/14/16)

The `feature_matrix_gate` is a new job in `.github/workflows/compat-governance-pr.yml`, alongside the existing `manifest_drift_gate`, `oracle_parity_gate`, `helper_legacy_parity_gate`, `oom_contract_gate`, `api_value_baseline_gate`. **NOT** a new file.

### 6.1 Job definition (recommended)

```yaml
feature_matrix_gate:
    name: feature_matrix_gate (${{ matrix.cell }})
    runs-on: ubuntu-latest
    strategy:
        fail-fast: false
        matrix:
            include:
                - cell: cpu-only
                  features: ""        # default = ["cpu"]
                - cell: cpu+wgpu
                  features: "wgpu"
                - cell: all-features
                  features: "wgpu,cuda,rocm,metal"
    steps:
        - name: Checkout
          uses: actions/checkout@v6
        - name: Resolve pinned Rust channel
          id: rust
          run: |
            python <<'PY'
            ... (same as existing gates) ...
            PY
        - name: Install pinned Rust toolchain
          uses: dtolnay/rust-toolchain@master
          with:
              toolchain: ${{ steps.rust.outputs.channel }}
        - name: Cache Rust artifacts
          uses: Swatinem/rust-cache@v2
        - name: Install ROCm runtime headers (for cubecl-hip-sys build script)
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

### 6.2 Per-cell expectation table

| Cell | Features | `cargo check` | `cargo test` | Expected runtime | Fail mode |
|------|---------|--------------|-------------|------------------|-----------|
| cpu-only | (default) | ✓ compiles | ✓ runs cpu unit tests | ~3 min | red on any compile/test fail |
| cpu+wgpu | wgpu | ✓ compiles | ✓ runs cpu + wgpu non-ignored tests (no GPU on ubuntu-latest, but compile path exercises) | ~5 min | red on any compile/test fail |
| all-features | wgpu,cuda,rocm,metal | ✓ compiles (after ROCm install) | ✓ runs all non-ignored, non-oracle tests | ~10 min (ROCm install dominates) | red on any compile/test fail |

### 6.3 Fail-closed semantics

- `fail-fast: false` so each cell runs independently and gives narrow signal.
- The job's matrix entries are listed in the branch protection "required status checks" as `feature_matrix_gate (cpu-only)`, `feature_matrix_gate (cpu+wgpu)`, `feature_matrix_gate (all-features)` — three separate required checks.
- Adding the gate to branch protection is a **separate manual repo settings step** that the planner should call out as a checklist item in `16-PLAN.md`.

### 6.4 ROCm install verification

[ASSUMED] The `amdgpu-install` step above works on `ubuntu-latest`. **Verify in Wave 2** by running the workflow on a feature branch and inspecting the logs. If the ROCm package source has changed by 2026-05, fall back to apt-pinning the explicit `rocm-dev` package via Radeon's Jammy archive. If install ends up flaky/slow (>10 min), demote `all-features` to a separate weekly cron job and replace with `cpu+wgpu+cuda` and `cpu+wgpu+metal` as 4-cell matrix.

## 7. ROCm Oracle Suite Design (D-15)

### 7.1 Trigger and gating

| Mechanism | Value |
|-----------|-------|
| Compile gate | `#[cfg(feature = "rocm")]` on test modules and individual tests |
| Run gate | `#[ignore]` attribute on each test |
| Env-gate trigger | `CINTX_ROCM_ORACLE=1` (read by the test, which `panic!`s if unset to make accidental opt-in obvious) |
| Default test command | `cargo test -p cintx-oracle --features rocm` — does NOT run the suite (tests are `#[ignore]`'d) |
| Opt-in test command | `CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm cargo test -p cintx-oracle --features rocm -- --ignored` |
| xtask wrapper | `cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle [--profile <p>]` |

### 7.2 xtask helper signature

Add to `xtask/src/main.rs` (a new module `rocm_oracle.rs` mirroring the existing `oracle_update.rs` shape):

```rust
/// Run the ROCm oracle base-family suite locally.
///
/// Env-gated invocation; requires --features rocm and the dev host to have
/// ROCm 7.x installed. Walks the same five base families as oracle_parity_gate
/// at atol=1e-12, but under CINTX_BACKEND=rocm.
pub fn rocm_oracle(profile: Option<String>) -> anyhow::Result<()> {
    // Sets CINTX_ROCM_ORACLE=1 + CINTX_BACKEND=rocm and runs:
    //   cargo test -p cintx-oracle --features rocm,<profile> -- --ignored
    // ...
}
```

[CITED: `xtask/src/oracle_update.rs` and `xtask/src/main.rs` for the existing xtask pattern.]

### 7.3 Base family list (full — D-15)

The oracle suite must cover all five base families at `atol=1e-12`:

| Family | Test file (existing) | Coverage |
|--------|---------------------|----------|
| 1e (overlap, kinetic, nuc-attraction) | `crates/cintx-oracle/tests/one_electron_parity.rs` | gate the existing tests with `#[cfg(any(feature = "cpu", feature = "rocm"))]` and add a `#[ignore]`'d rocm variant for each |
| 2e | `crates/cintx-oracle/tests/two_electron_parity.rs` | same |
| 2c2e | `crates/cintx-oracle/tests/center_2c2e_parity.rs` | same |
| 3c1e | `crates/cintx-oracle/tests/center_3c1e_parity.rs` | same |
| 3c2e | `crates/cintx-oracle/tests/center_3c2e_parity.rs` | same |

[VERIFIED: `crates/cintx-oracle/tests/` directory listing on 2026-05-09]

The simplest implementation path: in each test file, parameterize the existing test fn over a `&[BackendKind]` slice; the existing `cpu` test passes `BackendKind::Cpu`; add a new `#[cfg(feature = "rocm")] #[ignore] fn ..._rocm()` that passes `BackendKind::Rocm`. Both share the assertion harness.

### 7.4 No CI gate for ROCm

CI does NOT run the ROCm oracle. The `oracle_parity_gate` continues to run only on the existing `cpu` profile + the four feature profiles (`base`, `with-f12`, `with-4c1e`, `with-f12+with-4c1e`). The seed `gpu-ci-runners.md` tracks the eventual closure of this gap.

## 8. Pitfalls and Risks

### 8.1 Metal blocker (HIGHEST)

[VERIFIED: cubecl 0.10.0 docs and crates.io] `cubecl-metal` does not exist. D-05/D-07 must be amended. See §3.3 for the three options. Without resolution this phase cannot ship.

### 8.2 ROCm version skew on the dev host

[ASSUMED] `cubecl-hip-sys 7.1.5280200` requires an HIP version that matches the dev host's ROCm install (per cubecl-hip-sys README's "the crates follow the same versioning as HIP"). If `hipconfig --version` on the dev host doesn't match `7.1.x`, a feature-flag-pinned variant of `cubecl-hip-sys` may be required. **Verify in Wave 1, task 1**, before merging the rocm feature wiring.

### 8.3 cudarc dynamic loading at runtime

[VERIFIED] `cudarc` defaults to `dynamic-loading`. This is good for compile-time portability but means that on a host without a CUDA driver, **the binary still runs** and `cudarc` only fails at first attempt to call into CUDA. The first-call failure is opaque (`libloading::Error` rather than a typed error). On the dev host (no CUDA), `CINTX_BACKEND=cuda` will: pass `resolve_backend_kind()`, call `CudaRuntime::client(&CudaDevice::default())`, and then panic somewhere downstream when the first kernel launch tries to load `libcuda.so`. **Mitigation:** add a probe in `cuda_backend::resolve_cuda_client()` that calls a cheap cudarc API (e.g. `cudarc::driver::CudaContext::new(0)` or `cudarc::driver::cuInit`) and returns `cintxRsError::UnsupportedApi { requested: "cuda-driver-not-loadable" }` if it fails. [ASSUMED] cudarc exposes such a probe — confirm in Wave 1; if not, fall back to documenting "cuda is best-effort on hosts without CUDA driver."

### 8.4 The wgpu umbrella-feature removal is non-trivial

The current `cintx-cubecl/Cargo.toml:21` is `cubecl = { version = "0.10.0", features = ["wgpu"] }`. D-07 has us drop the `features = ["wgpu"]` part and add `cubecl-wgpu` as a direct optional dep. **This means:** `use cubecl::Wgpu...` paths will break (because the umbrella crate's wgpu re-exports are gated on its own `wgpu` feature). All `use cubecl::wgpu::...` paths in the codebase need to change to `use cubecl_wgpu::...` directly. [VERIFIED] grep for `use cubecl::wgpu` returned 0 hits (current code uses `use cubecl_wgpu::` paths already), so this risk is already mitigated. Confirm during Wave 1.

### 8.5 `#[cfg(feature = "...")]` per-arm exhaustiveness

D-10 prescribes per-variant cfg gating, which means every `match` on `BackendKind` or `ResolvedBackend` must repeat the cfg per arm. There are currently three such matches (`backend/mod.rs:47`, `backend/mod.rs:32` (wgpu_features), `executor.rs:74`). After this phase that grows to ~15 arms across 5 features. **Mitigation:** add a `BackendKind::name() -> &'static str` helper method that's `#[cfg]` per arm and returns the canonical name; many existing matches can collapse to `match self.name() { "wgpu" => ..., "cuda" => ..., _ => unreachable!() }` — but this trades exhaustiveness for runtime panic. Prefer the verbose match arms unless cfg fanout becomes unmaintainable.

### 8.6 Branch protection gate registration is a manual step

GitHub Actions auto-discovers new jobs but does NOT auto-add them to branch protection's required-status-checks list. After the `feature_matrix_gate` job lands, an admin must edit branch protection settings (`Settings → Branches → main → Require status checks → Add: feature_matrix_gate (cpu-only) / (cpu+wgpu) / (all-features)`). **PLAN must include this as a checklist item, not a code task.**

### 8.7 BackendCapabilityToken default `backend_api: "wgpu"`

[VERIFIED: `crates/cintx-runtime/src/options.rs:67`] `BackendCapabilityToken::default()` returns `backend_api: "wgpu"`. After D-11 this is inconsistent with `BackendKind::Cpu` being the default. Update `BackendCapabilityToken::default()` to return `backend_api: "cpu"` in the same task that flips `BackendIntent::default()`. Otherwise the drift detection at `workspace.rs:354–363` becomes asymmetric.

### 8.8 `ResolvedBackend` Cpu arm currently lacks the `is_cpu()` style helpers used in tests

Adding three new arms (`Cuda`, `Rocm`, `Metal`) to `ResolvedBackend` with per-arm `wgpu_features() -> &[String]` returning `&[]` for non-wgpu arms is mechanical, but the executor's `check_f64_capability` function in `executor.rs:69-83` currently has only two arms and uses a `match`. After expansion that becomes 5 arms with cfg gating — the `Cuda` and `Rocm` arms should return `Ok(())` (cuda f64 capable; rocm depends on the actual GPU but we accept-with-runtime-failure in this phase); `Metal` returns `Ok(())` too if Metal-via-wgpu has SHADER_F64 (likely yes on Apple Silicon, no on older Intel Macs — accept with runtime failure).

## 9. Code Examples (verified patterns from the existing codebase)

### 9.1 Adding a per-feature variant to `ResolvedBackend`

```rust
// crates/cintx-cubecl/src/backend/mod.rs - shape after this phase
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
    Metal(cubecl::client::ComputeClient<cubecl_wgpu::WgpuRuntime>, Vec<String>),  // M1: alias for wgpu
}
```

### 9.2 The new `resolve_backend_kind` shape (D-03)

```rust
// crates/cintx-cubecl/src/backend/mod.rs
pub fn resolve_backend_kind() -> Result<BackendKind, cintxRsError> {
    use cintx_core::cintxRsError;
    match std::env::var("CINTX_BACKEND").as_deref() {
        Err(_) | Ok("") => Ok(BackendKind::default()),  // Cpu per D-11
        Ok("cpu") => Ok(BackendKind::Cpu),
        #[cfg(feature = "wgpu")]
        Ok("wgpu") => Ok(BackendKind::Wgpu),
        #[cfg(feature = "cuda")]
        Ok("cuda") => Ok(BackendKind::Cuda),
        #[cfg(feature = "rocm")]
        Ok("rocm") => Ok(BackendKind::Rocm),
        #[cfg(feature = "metal")]
        Ok("metal") => Ok(BackendKind::Metal),
        // Compiled-out backends — D-01 hard error
        Ok(name) if KNOWN_BACKEND_NAMES.contains(&name) => {
            Err(cintxRsError::BackendNotCompiled {
                requested: name.to_owned(),
                compiled_in: COMPILED_IN_BACKENDS.iter().map(|s| s.to_string()).collect(),
            })
        }
        // Unrecognized — D-02 error
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

/// Public introspection helper (Claude's discretion item — recommended).
pub fn compiled_backends() -> &'static [&'static str] {
    COMPILED_IN_BACKENDS
}
```

### 9.3 The new `BackendNotCompiled` variant

```rust
// crates/cintx-core/src/error.rs
#[derive(Debug, Error)]
pub enum cintxRsError {
    // ... existing variants ...
    #[error("requested backend {requested:?} is not compiled in; compiled-in backends: {compiled_in:?}")]
    BackendNotCompiled {
        requested: String,
        compiled_in: Vec<String>,
    },
}
```

[VERIFIED: matches existing `thiserror` style in `cintx-core/src/error.rs:35-71`]

## 10. State of the Art / What Changed in cubecl 0.10.0

[VERIFIED: cubecl 0.10.0 release date 2026-05-07; crates.io publish dates per `Cargo.lock` checksums; `cubecl-cuda 0.10.0` checksum `b6b0a69ff45688d322ad8e92c8bf645167b9ca490fa8fa087fc6adac8c5e46be`]

| Old approach (cubecl ≤0.9) | Current (cubecl 0.10.0) | Impact |
|----------------------------|------------------------|--------|
| Umbrella `cubecl` crate's `features = ["wgpu"]` was the standard way to get the wgpu runtime | Each runtime is an independent crate (`cubecl-{cpu,cuda,hip,wgpu}`); umbrella delegates to them via optional features | Our refactor pulls them as direct optional deps for finer-grained control |
| Metal was a TODO/planned feature | Metal supported in production via `cubecl-wgpu` running on Apple's wgpu Metal backend | Our `metal` feature should alias to wgpu, NOT a non-existent `cubecl-metal` |
| ROCm/HIP pinned via `cubecl-hip-sys 6.x` | `cubecl-hip-sys 7.1.5280200` (HIP 7.x) | Possible dev-host ROCm version mismatch — verify in Wave 1 |

## 11. Environment Availability Audit

The phase has external dependencies. Probed on 2026-05-09:

| Dependency | Required By | Available on dev host | Available on `ubuntu-latest` CI | Fallback |
|------------|-------------|----------------------|--------------------------------|----------|
| Rust toolchain 1.94.0 | All | ✓ (`rust-toolchain.toml`) | ✓ (installed by workflow step) | — |
| `cargo --locked` | All | ✓ | ✓ | — |
| `hipconfig` (ROCm) | `cubecl-hip-sys` build script | ✓ (per CONTEXT.md "Linux + AMD ROCm") | ✗ — **must be installed in `all-features` cell** | apt-get install ROCm runtime headers (~1.5 GB) |
| CUDA driver | cudarc dynamic loading (runtime, not build) | ✗ | ✗ | none — cuda is compile-only verified |
| Metal SDK / macOS | cubecl-wgpu → wgpu Metal backend (build-time on macOS) | n/a (Linux dev host) | ✗ (no macOS runner) | none — metal is compile-only verified on Linux |
| `cargo-nextest` | (optional, faster CI) | not yet adopted | not in workflow | use `cargo test` (current pattern) |

**Missing dependencies with fallback:**
- `hipconfig` on CI → install ROCm in `all-features` cell (see §6.4).
- Metal SDK → metal feature on Linux dev host compiles via cubecl-wgpu (M1 in §4.3) and the actual Metal target is exercised at runtime only on macOS; not blocked.

**Missing dependencies with no fallback:**
- None for this phase. All paths have either a working dev/CI configuration or an explicit risk-accept (cuda/metal runtime).

## 12. Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Cargo built-in test harness (rustc 1.94.0 stable) |
| Config file | `Cargo.toml` per crate |
| Quick run command | `cargo test -p cintx-cubecl` |
| Full suite command | `cargo test --workspace` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test type | Automated command | File exists? |
|--------|----------|-----------|-------------------|--------------|
| BACK-01 | `cintx-cubecl/Cargo.toml` exposes `cuda`, `rocm`, `metal`, `wgpu` features (additive); no umbrella `features = ["wgpu"]` on the cubecl import | static (Cargo lockfile + cargo metadata) | `cargo metadata --no-deps --format-version 1 \| jq '.packages[] \| select(.name == "cintx-cubecl") \| .features'` and assert `cuda`, `rocm`, `metal`, `wgpu`, `cpu` keys present | ❌ Wave 0 — new test file `crates/cintx-cubecl/tests/cargo_features.rs` |
| BACK-02 | `BackendKind` and `ResolvedBackend` extend with `Cuda`, `Rocm`, `Metal`; per-variant cfg gates | unit (compile-time) | `cargo build -p cintx-cubecl --features cuda` && `cargo build -p cintx-cubecl --features rocm` && `cargo build -p cintx-cubecl --features metal` (each must succeed) | ❌ Wave 0 — feature_matrix_gate covers (existing test files extended) |
| BACK-03 | `cargo check` with every non-empty subset of `{cuda, rocm, metal, wgpu}` builds cleanly | integration (cargo) | matrix in `feature_matrix_gate.yml` (3 cells per D-13) | ❌ Wave 0 — new `.github/workflows` job |
| BACK-04 | `cargo test --features rocm` runs ≥1 oracle smoke test under `CINTX_BACKEND=rocm` matching tolerances | integration (oracle) | `CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm cargo test -p cintx-oracle --features rocm -- --ignored` | ❌ Wave 0 — extend each of `crates/cintx-oracle/tests/{one_electron,two_electron,center_2c2e,center_3c1e,center_3c2e}_parity.rs` with rocm variant |
| BACK-05a | Unset env → Cpu | unit | `cargo test -p cintx-cubecl backend::tests::env_unset_resolves_to_cpu` | ❌ Wave 0 — new test in `crates/cintx-cubecl/src/backend/mod.rs` `mod tests` |
| BACK-05b | Compiled-out backend → typed `BackendNotCompiled` | unit | `cargo test -p cintx-cubecl backend::tests::not_compiled_backend_errors` (set CINTX_BACKEND=cuda when feature off) | ❌ Wave 0 — new test |
| BACK-05c | Unrecognized env value → typed `InvalidEnvParam` | unit | `cargo test -p cintx-cubecl backend::tests::unknown_backend_errors_invalid_env_param` | ❌ Wave 0 — new test |
| BACK-05d | Compiled-in backend selected | unit | `cargo test -p cintx-cubecl backend::tests::compiled_in_backend_resolves` | ❌ Wave 0 — new test |
| BACK-06 | Cuda/metal docs cite `notes/cuda-metal-verification-gap.md`; no oracle gate added | manual + grep | `grep -r "cuda-metal-verification-gap" crates/cintx-cubecl/src/ | wc -l` ≥ 2; CI workflow grep verifies no `cuda` or `metal` in `oracle_parity_gate` matrix | manual review of `16-PLAN.md` execution |
| BACK-07 | Feature matrix exercised in CI on existing runners | CI gate | `feature_matrix_gate` runs on ubuntu-latest with three cells (D-13/14) | ❌ Wave 0 — new `.github/workflows` job |

### Sampling Rate

- **Per task commit:** `cargo test -p cintx-cubecl` and `cargo test -p cintx-runtime` (the two crates seeing the largest changes).
- **Per wave merge:** `cargo test --workspace` plus the three feature-matrix cells locally (`cargo check -p cintx-cubecl --features wgpu`, `--features wgpu,cuda,rocm,metal`).
- **Phase gate:** Full feature-matrix CI green; ROCm oracle suite green on the dev host (operator-driven, not CI).

### Wave 0 Gaps

- [ ] `crates/cintx-cubecl/tests/cargo_features.rs` — covers BACK-01 (asserts feature names exist via cargo_metadata or by attempting to enable each)
- [ ] `crates/cintx-cubecl/src/backend/mod.rs` `mod tests` — extend with `env_unset_resolves_to_cpu`, `not_compiled_backend_errors`, `unknown_backend_errors_invalid_env_param`, `compiled_in_backend_resolves` covering BACK-05a-d. Existing tests at lines 89-127 are racy (parallel env-var mutation); the new tests must use a `serial_test` mutex or `std::sync::OnceLock`-gated env mutation. [ASSUMED] `serial_test` is acceptable to add as a dev-dep — confirm in Wave 0
- [ ] `crates/cintx-oracle/tests/{one_electron,two_electron,center_2c2e,center_3c1e,center_3c2e}_parity.rs` — five files, each gets a `#[cfg(feature = "rocm")] #[ignore] fn ..._rocm()` variant covering BACK-04
- [ ] `xtask/src/rocm_oracle.rs` + dispatch in `xtask/src/main.rs` — covers BACK-04 (operator workflow)
- [ ] `.github/workflows/compat-governance-pr.yml` — new `feature_matrix_gate` job (3 matrix cells) covering BACK-03 + BACK-07
- [ ] `crates/cintx-core/src/error.rs` `mod tests` — extend with `backend_not_compiled_formats_and_matches` covering D-01

**Nyquist sample density:** The matrix has 5 toggleable backends and one default cpu — 32 possible combinations. D-13's 3-cell minimum samples 3 of the 32 (under-sampled by Nyquist). The CONTEXT.md decision explicitly accepts this: failures in non-tested combos surface to bisection rather than direct gating. The compensating control is BACK-02's per-variant cfg gating: any compile-time issue in one variant manifests in the `all-features` cell, so the worst-case undetected failure is a runtime regression in a 4-feature combo (e.g. `wgpu+cuda+rocm` minus `metal`) that does not surface in `cpu+wgpu` or `all-features`. The probability of a regression that's both undetected by `all-features` AND happens in only a partial subset is low — but the planner should add a Wave 3 follow-up task to write a smoke test for partial subsets if any cuda/rocm runtime issues are reported.

## 13. Assumptions Log

| # | Claim | Section | Risk if wrong |
|---|-------|---------|---------------|
| A1 | A fixed sentinel capability fingerprint for cuda/metal is acceptable to the Phase 5/6 D-08 drift contract | §4.1 | Drift detection might reject the cuda backend at evaluate-time — mitigated by adding a one-line check in the Phase 5/6 D-08 contract test |
| A2 | `AmdDevice::default()` exists in `cubecl-hip 0.10.0` | §4.2 | If not, fall back to `AmdDevice::new(0)` — 1-line edit |
| A3 | `cubecl-hip-sys 7.1.5280200` works with the dev host's ROCm install version | §4.2, §8.2 | Wave 1 must verify before merging rocm wiring |
| A4 | `amdgpu-install` works on `ubuntu-latest` 2026 GHA images | §6.4 | If not, demote `all-features` to weekly cron with apt-pinning |
| A5 | cudarc has a cheap probe API to detect "driver not loadable" | §8.3 | If not, document cuda as best-effort runtime — no blocker for compile-only verification |
| A6 | `serial_test` is acceptable as a dev-dep for env-var test ordering | §12 | If user pushes back, use a `OnceLock<Mutex<()>>` pattern in-tree |
| A7 | The "four required gates" mentioned in CONTEXT.md are actually the four jobs inside `compat-governance-pr.yml` | §6.1 | None — confirmed by reading the workflow file |
| A8 | Branch protection updates can be done by the user post-merge | §8.6 | None — clearly a manual ops step |

## 14. Open Questions (RESOLVED)

All three open questions surfaced during research were resolved during plan-phase
and the resolution is reflected in `16-02-PLAN.md`.

1. **Should `metal` be M1 (alias for wgpu), M2 (drop the feature), or M3 (empty-feature stub)?**
   - What we knew: `cubecl-metal` does not exist; Metal is served by `cubecl-wgpu` on Apple targets.
   - What was unclear: User intent. CONTEXT.md D-05 listed `metal = ["dep:cubecl-metal"]` as if cubecl-metal existed.
   - Recommendation: M1 (alias for wgpu).
   - **RESOLVED:** User selected M1 during plan-phase. Captured as a `<context_deviation>` block in `16-02-PLAN.md` (lines 71–98). `metal = ["dep:cubecl-wgpu", "dep:wgpu"]`; `BackendKind::Metal` dispatches to `cubecl_wgpu::WgpuRuntime` in `from_intent()`. No `metal_backend.rs` is created.

2. **Should `compiled_backends() -> &'static [&'static str]` be public?**
   - What we knew: CONTEXT.md says "lean toward exposing it; it's free if `compiled_in` is already a `const`."
   - What was unclear: SemVer commitment. Once exposed, downstream might rely on the order or the exact string spellings.
   - Recommendation: Expose as `pub const COMPILED_BACKENDS: &[&str]`.
   - **RESOLVED:** `16-02-PLAN.md` exposes `pub fn compiled_backends() -> &'static [&'static str]` (a function, not a const). API-equivalent at the call site (both produce a `&'static [&'static str]`); the function form is slightly more flexible if cfg-driven assembly is needed later. The internal storage may still be a `const` array. The order and spelling are still part of the public commitment as recommended.

3. **Should the `serial_test` crate become a workspace dev-dep?**
   - What we knew: Existing env-var tests at `backend/mod.rs:104-127` work around parallel-test env races by reading-without-mutating, which is fragile and silently weakens coverage.
   - What was unclear: Whether the user prefers a single new dev-dep or an in-tree mutex pattern.
   - Recommendation: Add `serial_test` as a dev-dep.
   - **RESOLVED:** `16-02-PLAN.md` (Step around line 524) adds `serial_test` as a dev-dep and applies `#[serial]` to the env-var resolution tests so mutating tests no longer race.

## 15. Sources

### Primary (HIGH confidence)
- `Cargo.lock` (this repo) — verified `cubecl-cuda 0.10.0`, `cubecl-hip 0.10.0`, `wgpu 29.0.3`, `bytemuck 1.25.0` resolutions
- `crates/cintx-cubecl/Cargo.toml` — current feature shape
- `crates/cintx-cubecl/src/backend/{mod,wgpu_backend,cpu_backend}.rs` — pattern for new backends
- `crates/cintx-runtime/src/options.rs` — `BackendKind`, `BackendIntent`, `BackendCapabilityToken`
- `crates/cintx-core/src/error.rs` — error enum host
- `.github/workflows/compat-governance-pr.yml` — existing CI gate pattern (lines 37-228)
- [docs.rs/cubecl/0.10.0/cubecl/](https://docs.rs/cubecl/0.10.0/cubecl/) — confirms cubecl 0.10.0 has cpu/cuda/hip/wgpu as optional deps; no metal
- [docs.rs/cubecl-cuda/0.10.0/cubecl_cuda/](https://docs.rs/cubecl-cuda/0.10.0/cubecl_cuda/) — exports `CudaRuntime`, `CudaDevice`, `RuntimeOptions`
- [docs.rs/cubecl-hip/0.10.0/cubecl_hip/](https://docs.rs/cubecl-hip/0.10.0/cubecl_hip/) — exports `HipRuntime`, re-exports device types incl. `AmdDevice`

### Secondary (MEDIUM confidence)
- [github.com/tracel-ai/cubecl](https://github.com/tracel-ai/cubecl) README — Metal supported via wgpu runtime
- [github.com/tracel-ai/cubecl-hip-sys](https://github.com/tracel-ai/cubecl-hip-sys) — `hipconfig` build-time requirement
- [crates.io/crates/cudarc](https://lib.rs/crates/cudarc) (via lib.rs mirror) — `dynamic-loading` is default

### Tertiary (LOW confidence — flagged for validation)
- AmdDevice `Default` impl existence (A2 above) — docs.rs returned 404 for the struct page; confirm by `cargo check -p cintx-cubecl --features rocm` in Wave 1.
- `amdgpu-install` continues to work on `ubuntu-latest` 2026 GHA images (A4 above) — requires a workflow run to verify.

## 16. Metadata

**Confidence breakdown:**
- Cuda/Rocm/Wgpu wiring: HIGH — lockfile evidence + docs.rs exports + existing in-tree pattern
- Metal blocker resolution: HIGH on the blocker (cubecl-metal doesn't exist), MEDIUM on the recommended fix (M1 — needs CONTEXT.md amendment)
- Migration audit completeness: HIGH — exhaustive grep run on 2026-05-09; 30 callsites identified
- CI matrix gate design: MEDIUM — `amdgpu-install` step is best-effort; cuda compile path verified via cudarc dynamic-loading docs but not yet run
- ROCm oracle suite design: HIGH — mirrors existing oracle test layout; only the gating attributes are new

**Research date:** 2026-05-09
**Valid until:** 2026-06-09 (30 days; cubecl 0.10.x line is stable but `cubecl-hip-sys` 7.x ↔ ROCm version coupling could shift if upstream releases a new HIP)

## RESEARCH COMPLETE
