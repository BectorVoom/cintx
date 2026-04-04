# Phase 7: Executor Infrastructure Rewrite - Research

**Researched:** 2026-04-03
**Domain:** CubeCL client API migration, ResolvedBackend enum dispatch, CPU backend integration, RecordingExecutor removal, f64 precision strategy
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Backend enum design**
- D-01: Use a `ResolvedBackend` enum with `Wgpu(ComputeClient<WgpuRuntime>)` and `Cpu(ComputeClient<CpuRuntime>)` arms. Per-arm kernel dispatch via match. Future cuda/rocm/metal backends add new enum arms.
- D-02: This approach is required because `BackendExecutor` is used as `&dyn BackendExecutor` in `planner::evaluate`, which rules out generics on `CubeClExecutor<R: Runtime>`.

**CPU backend integration**
- D-03: Both wgpu and cpu backends must pass oracle parity independently. Tests run on both.
- D-04: Backend selection is a runtime choice made by the user via `CINTX_BACKEND=wgpu|cpu` environment variable. The executor reads this at init time to resolve which `ResolvedBackend` arm to construct.
- D-05: `cubecl-cpu` is already in Cargo.lock as a transitive dependency. Both backends are always compiled — no feature gate. The env var controls which one is used at runtime.

**CubeCL client API pattern**
- D-06: Executor internals use `WgpuRuntime::client(&device)` / `CpuRuntime::client(&device)` directly. Buffer management uses `client.create()`, `client.empty()`, `client.read()`. Kernels use `#[cube(launch)]` with `ArrayArg::from_raw_parts`. Reference pattern: `docs/manual/Cubecl/Cubecl_vector.md`.
- D-07: Buffer lifecycle (create/read/empty) lives inside each kernel family module, not centralized in the executor. Each family has different input shapes and buffer counts.

**RecordingExecutor removal**
- D-08: RecordingExecutor is deleted in this phase. Once the executor uses direct `client.read()` for buffer retrieval, the recording wrapper is unnecessary. Staging output flows directly from the client read result into `io.staging_output()`.

**f64 precision strategy**
- D-09: Both backends must produce f64-precision results. wgpu path gates on `SHADER_F64` capability and returns `UnsupportedApi` when absent. CPU backend always supports f64 natively.

**Carried forward from Phase 5**
- D-10: Backend auto-selects wgpu when `CINTX_BACKEND` is unset; fails closed with typed error when no valid adapter/capability is available.
- D-11: Backend intent is control-plane metadata carried via runtime options.
- D-12: Staging ownership: backend staging-only, compat retains final caller-visible flat writes.

### Claude's Discretion
- Exact `ResolvedBackend` enum field names and module placement within cintx-cubecl
- How `bytemuck` promotion to direct dep is handled (already in Cargo.lock)
- Exact error messages for `SHADER_F64` absence
- Internal helper functions for client initialization and buffer marshaling

### Deferred Ideas (OUT OF SCOPE)
- CUDA/ROCm/Metal backend arms — architecture supports them via ResolvedBackend enum, implementation deferred to v1.2+
- Screening/batching optimizations — performance work after correctness proven
- Workgroup sizing strategy — post-v1.1 specialization
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EXEC-06 | Executor internals use CubeCL client API directly (`WgpuRuntime::client()`, `client.create()`/`client.read()`/`client.empty()`, `ArrayArg::from_raw_parts`) | Client API pattern confirmed from `docs/manual/Cubecl/Cubecl_vector.md`; `stage_device_buffers` probe path identified for removal in `transfer.rs` |
| EXEC-07 | RecordingExecutor removed from cintx-compat and cintx-rs — real kernel values flow through `io.staging_output()` directly | RecordingExecutor located in `cintx-compat/src/raw.rs` lines 21-71 and `cintx-rs/src/api.rs` line 158 and 394-430; deletion path clear once executor writes staging |
| EXEC-08 | ResolvedBackend enum dispatches between Wgpu and Cpu runtime arms with per-arm kernel launch | Full enum design pattern provided in ARCHITECTURE.md; `BackendKind::Cpu` already exists in `cintx-runtime/src/options.rs` |
| EXEC-09 | CPU backend enabled via `cpu = ["cubecl/cpu"]` feature in cintx-cubecl for CI oracle testing without GPU | `cubecl` 0.9.0 exposes `"cpu"` feature flag confirmed; `cubecl-cpu 0.9.0` already in Cargo.lock |
| VERI-06 | f64 precision strategy resolved — CPU backend as primary oracle path; wgpu SHADER_F64 tested opportunistically | `SHADER_F64` collection already in `runtime_bootstrap.rs`; strategy: gate wgpu on capability, CPU always f64 |
</phase_requirements>

---

## Summary

Phase 7 rewrites `CubeClExecutor` internals to use the CubeCL client API directly, replacing the stub host-side probe path (`TransferPlan::stage_device_buffers`) with real `client.create()`/`client.empty()`/`client.read()` calls. The rewrite introduces a `ResolvedBackend` enum that stores a pre-initialized `ComputeClient<WgpuRuntime>` or `ComputeClient<CpuRuntime>` handle, resolving the object-safety conflict between generic `Runtime` bounds and the `&dyn BackendExecutor` usage in `planner::evaluate`. `RecordingExecutor` in both `cintx-compat` and `cintx-rs` is deleted once real kernel values flow directly into `io.staging_output()` — the recording indirection exists only because the stub executor never wrote values.

This is a structural wiring phase — no real physics kernels are written yet. All five family stubs (`one_electron`, `two_electron`, `center_2c2e`, `center_3c1e`, `center_3c2e`) update their signatures to accept `&ResolvedBackend` and `&mut [f64]` staging, but continue returning zeros as their body until Phase 9. The new f64 strategy is codified: CPU backend is the oracle comparison path (always f64); wgpu path gates on `SHADER_F64` and returns `UnsupportedApi` when absent.

The primary risk is double-initialization of the CubeCL wgpu runtime. The existing `OnceLock` in `runtime_bootstrap.rs` guards the `Auto` selector only — the new `ResolvedBackend` cache must extend this guard to cover all selectors, or the test suite will panic on the second test that touches the GPU path.

**Primary recommendation:** Follow the 7-step build order from ARCHITECTURE.md — add `ResolvedBackend` skeleton first (no behavior change), update kernel signatures second, wire client API into the executor path third, then delete `RecordingExecutor` only after staging is confirmed populated.

---

## Standard Stack

### Core (already in workspace — no new crates required at workspace level)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cubecl` | `0.9.0` | CubeCL compute backend; `"cpu"` feature enables CpuRuntime | Already pinned in `cintx-cubecl/Cargo.toml` |
| `cubecl-wgpu` | `0.9.0` | `WgpuRuntime`, `WgpuDevice` — wgpu client arm | Already direct dep in `cintx-cubecl` |
| `cubecl-runtime` | `0.9.0` | `ComputeClient<R>` trait and infrastructure | Already direct dep in `cintx-cubecl` |
| `bytemuck` | `1.25.0` (already in Cargo.lock) | `cast_slice` for `client.create(&[u8])` and `client.read()` output | No new crate — promote to direct dep in `cintx-cubecl/Cargo.toml` |
| `wgpu` | `26.0.1` | Feature flag access for `SHADER_F64` capability check | Already direct dep in `cintx-cubecl` |

### New Feature Flags Required

```toml
# crates/cintx-cubecl/Cargo.toml
[features]
default = []
with-f12 = []
with-4c1e = []
cpu = ["cubecl/cpu"]      # NEW: enables CpuRuntime for oracle parity tests without GPU

[dependencies]
bytemuck = "1"             # NEW: promote from transitive to direct dep
```

D-05 locks the decision: both backends are always compiled when the `cpu` feature is enabled — the `CINTX_BACKEND` env var selects at runtime. No backend-specific `#[cfg(feature)]` guards on the production code path.

### New Files Required in `cintx-cubecl`

| File | Purpose |
|------|---------|
| `src/backend/mod.rs` | `ResolvedBackend` enum, `BackendIntentKey`, `from_intent()` constructor, client cache |
| `src/backend/wgpu.rs` | `WgpuRuntime` client bootstrap (refactored out of `runtime_bootstrap.rs`) |
| `src/backend/cpu.rs` | `CpuRuntime::client(&CpuDevice::default())` bootstrap (new) |

### Modified Files

| File | Change |
|------|--------|
| `crates/cintx-cubecl/src/executor.rs` | Remove `preflight_wgpu` inline; add `resolve_client()`; remove `transfer_plan.stage_device_buffers()`; pass `&backend` + `io.staging_output()` to `launch_family` |
| `crates/cintx-cubecl/src/kernels/mod.rs` | Change `FamilyLaunchFn` signature; update `launch_family` |
| `crates/cintx-cubecl/src/kernels/one_electron.rs` | New signature — stubs still return zeros |
| `crates/cintx-cubecl/src/kernels/two_electron.rs` | New signature |
| `crates/cintx-cubecl/src/kernels/center_2c2e.rs` | New signature |
| `crates/cintx-cubecl/src/kernels/center_3c1e.rs` | New signature |
| `crates/cintx-cubecl/src/kernels/center_3c2e.rs` | New signature |
| `crates/cintx-cubecl/Cargo.toml` | Add `bytemuck = "1"` direct dep; add `cpu` feature |
| `crates/cintx-compat/src/raw.rs` | Delete `RecordingExecutor` struct (lines 21-71); update `eval_raw` to call executor directly |
| `crates/cintx-rs/src/api.rs` | Delete local `RecordingExecutor` (lines 394-430); update `evaluate` call at line 158 |

### Unchanged Files

`src/specialization.rs`, `src/resident_cache.rs`, `src/transform/`, `src/capability.rs`, `src/runtime_bootstrap.rs` (only narrowed to wgpu; bootstrap coordination moves to `backend/`), all `cintx-core`, `cintx-ops`, `cintx-runtime`, `cintx-capi`, `cintx-oracle`, `xtask` — zero public API changes.

---

## Architecture Patterns

### Recommended Project Structure (new files only)

```
crates/cintx-cubecl/src/
├── backend/
│   ├── mod.rs        # ResolvedBackend enum + cache + from_intent()
│   ├── wgpu.rs       # WgpuRuntime bootstrap helper
│   └── cpu.rs        # CpuRuntime bootstrap helper
├── executor.rs       # Modified: uses backend::ResolvedBackend
├── kernels/
│   ├── mod.rs        # Modified: new FamilyLaunchFn signature
│   ├── one_electron.rs   # Modified: new signature, stub body unchanged
│   └── ...
```

### Pattern 1: ResolvedBackend Enum (D-01)

**What:** Store pre-initialized `ComputeClient<R>` handles inside an enum, keyed on `BackendKind`. Dispatch per-arm by matching on the enum.

**When to use:** Whenever the executor needs a client handle. Resolve once at init/first-use; cache in `CubeClExecutor` behind a `Mutex<HashMap<BackendIntentKey, ResolvedBackend>>`.

```rust
// Source: .planning/research/ARCHITECTURE.md — Part 4

pub enum ResolvedBackend {
    Wgpu(ComputeClient<WgpuRuntime>),
    Cpu(ComputeClient<CpuRuntime>),
}

impl ResolvedBackend {
    pub fn from_intent(intent: &BackendIntent) -> Result<Self, cintxRsError> {
        match intent.backend {
            BackendKind::Wgpu => {
                let device = selector_to_wgpu_device(&intent.selector)?;
                let client = WgpuRuntime::client(&device);
                Ok(Self::Wgpu(client))
            }
            BackendKind::Cpu => {
                let client = CpuRuntime::client(&CpuDevice::default());
                Ok(Self::Cpu(client))
            }
        }
    }
}
```

**CubeClExecutor new shape:**

```rust
pub struct CubeClExecutor {
    resident_cache: DeviceResidentCache,
    backend_cache: Mutex<HashMap<BackendIntentKey, ResolvedBackend>>,
}
```

### Pattern 2: Updated FamilyLaunchFn Signature (D-07)

**What:** Pass `&ResolvedBackend` and `&mut [f64]` staging slice into family launch functions instead of `&TransferPlan`. Buffer lifecycle lives inside family modules.

```rust
// Source: .planning/research/ARCHITECTURE.md — Part 6

pub type FamilyLaunchFn = fn(
    &ResolvedBackend,
    &ExecutionPlan<'_>,
    &SpecializationKey,
    &mut [f64],   // staging slice — direct write target
) -> Result<ExecutionStats, cintxRsError>;
```

`launch_family` in `kernels/mod.rs` changes to match. `CubeClExecutor::execute` passes `&resolved_backend` and `io.staging_output()` instead of `&transfer_plan`. `TransferPlan` is retained as a planning/metrics struct but `stage_device_buffers` is removed.

### Pattern 3: Client API Buffer Lifecycle (D-06)

**What:** Inside each family module, use the matched client arm to create buffers, launch kernel, and read back.

```rust
// Source: docs/manual/Cubecl/Cubecl_vector.md (project reference)

// 1. Upload inputs
let input_buffer = client.create(bytemuck::cast_slice(&input_data));
// 2. Reserve output
let output_buffer = client.empty(output_byte_count);
// 3. Launch — all handles bound to named lets before ArrayArg construction
my_kernel::launch::<WgpuRuntime>(
    &client,
    CubeCount::Static(1, 1, 1),
    CubeDim::new(n as u32, 1, 1),
    unsafe { ArrayArg::from_raw_parts::<f64>(&input_buffer, n, 1) },
    unsafe { ArrayArg::from_raw_parts::<f64>(&output_buffer, n, 1) },
);
// 4. Read back
let output_bytes = client.read(vec![output_buffer.binding()]);
let values: &[f64] = bytemuck::cast_slice(&output_bytes[0]);
staging.copy_from_slice(values);
```

For per-arm dispatch inside a family function, match on `ResolvedBackend` twice (once for create/empty, once for read) — or use a macro dispatch helper as shown in ARCHITECTURE.md Part 4.

### Pattern 4: CINTX_BACKEND Environment Variable (D-04)

**What:** Read `CINTX_BACKEND` at executor init to determine `BackendKind`. Wire into `BackendIntent` before constructing the `ResolvedBackend`.

```rust
fn resolve_backend_kind() -> BackendKind {
    match std::env::var("CINTX_BACKEND").as_deref() {
        Ok("cpu") => BackendKind::Cpu,
        Ok("wgpu") | Err(_) => BackendKind::Wgpu,  // default to wgpu when unset
        Ok(other) => {
            tracing::warn!("Unknown CINTX_BACKEND value '{other}'; defaulting to wgpu");
            BackendKind::Wgpu
        }
    }
}
```

D-10 mandates wgpu when `CINTX_BACKEND` is unset.

### Pattern 5: f64 Capability Gate (D-09)

**What:** Before launching any f64 kernel via wgpu, check the preflight report for `SHADER_F64`. If absent, return `UnsupportedApi`.

```rust
fn check_f64_capability(report: &WgpuPreflightReport) -> Result<(), cintxRsError> {
    if !report.snapshot.features.iter().any(|f| f == "SHADER_F64") {
        return Err(cintxRsError::UnsupportedApi {
            requested: "wgpu-capability:missing_shader_f64".to_owned(),
        });
    }
    Ok(())
}
```

CPU backend never needs this check — `CpuRuntime` always supports f64 natively.

### Anti-Patterns to Avoid

- **Generic `CubeClExecutor<R: Runtime>`:** Breaks `BackendExecutor` object safety since it is used as `&dyn BackendExecutor`. Use the `ResolvedBackend` enum instead (D-02).
- **Buffer lifecycle in executor.rs:** Putting `client.create`/`client.empty`/`client.read` in a generic loop requires the executor to know family-specific input shapes. Keep buffer ops inside family modules (D-07).
- **Removing `RecordingExecutor` before staging is populated:** The recording wrapper exists because the stub executor writes no values. Delete it only after the real client.read path commits values to staging before `execute()` returns (D-08).
- **Temporary handle expressions in `ArrayArg`:** `ArrayArg::from_raw_parts::<T>(&client.empty(n), ...)` creates a temporary handle that drops at statement end. Always bind every handle to a named `let` before any `ArrayArg` construction (Pitfall 1 from PITFALLS.md).
- **Direct `init_setup` calls outside the bootstrap cache:** `cubecl::wgpu::init_setup` panics if called twice for the same device. All device init must go through the `OnceLock`/`Mutex<HashMap>` cache in `backend/mod.rs` (Pitfall 2 from PITFALLS.md).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| GPU buffer I/O byte conversion | Custom `unsafe` pointer casts | `bytemuck::cast_slice` | `bytemuck::Pod` trait bounds guarantee safety; already in Cargo.lock |
| Backend client initialization | Direct `init_setup` calls in each kernel module | `ResolvedBackend::from_intent()` behind the `Mutex<HashMap>` cache | Prevents double-init panic (Pitfall 2); centralizes selector logic |
| Per-arm kernel dispatch boilerplate | Duplicate match arms in every family fn | Dispatch macro `launch_on_backend!` as shown in ARCHITECTURE.md Part 4 | Eliminates copy-paste across 5+ family modules |
| f64 capability detection | Ad-hoc feature flag string inspection | Reuse `WgpuCapabilitySnapshot.features` from existing preflight report | Already collected by `collect_feature_names()` in `runtime_bootstrap.rs` |

**Key insight:** The `client.create`/`client.empty`/`client.read` API is the complete buffer management surface — there is no lower-level wiring needed. Any code that reimplements parts of this is adding unnecessary complexity.

---

## Common Pitfalls

### Pitfall 1: ArrayArg handle lifetime — use-after-free on the GPU
**What goes wrong:** `ArrayArg::from_raw_parts::<T>(&client.empty(n), ...)` creates a temporary `Handle` that drops at statement end while the kernel dispatch is still enqueued. Result is silent zero/garbage output.
**Why it happens:** Rust's borrow checker enforces the reference lifetime syntactically inside the `unsafe` block, but a temporary in a function call argument position drops immediately.
**How to avoid:** Bind every handle to a named `let` before constructing any `ArrayArg`. Pattern: `let out = client.empty(n); unsafe { ArrayArg::from_raw_parts::<f64>(&out, n, 1) }`.
**Warning signs:** Staging output is all-zeros even though no wgpu-capability error was returned; inserting a sentinel value in an input buffer shows the readback does not reflect it.

### Pitfall 2: CubeCL double-initialization panic
**What goes wrong:** Calling `cubecl::wgpu::init_setup` twice for the same device panics: "A server is still registered for device ...". The existing `OnceLock` in `runtime_bootstrap.rs` only guards `AdapterSelector::Auto`. The new `from_intent()` constructor must preserve this invariant.
**Why it happens:** CubeCL registers a compute server per device. The wgpu bootstrap path calls `init_setup` during first client acquisition.
**How to avoid:** Cache `ResolvedBackend` instances in `CubeClExecutor` behind a `Mutex<HashMap<BackendIntentKey, ResolvedBackend>>`. Call init only on first access for each `(BackendKind, selector_string)` key. For wgpu, the `from_intent()` call must route through `bootstrap_wgpu_runtime()` (which owns the `OnceLock`) — do not call `init_setup` directly.
**Warning signs:** Test binary panics on the second test that accesses GPU; panic message contains "A server is still registered".

### Pitfall 3: RecordingExecutor capture race with staging population
**What goes wrong:** If `RecordingExecutor::execute` calls `io.staging_output()` and the inner executor has not yet committed values (i.e., `client.read()` result has not been copied into the staging slice before `execute()` returns), the recording captures zeros.
**Why it happens:** The recording pattern assumes staging is populated before `execute()` returns. Any async or deferred write breaks this assumption.
**How to avoid:** The new executor must copy `client.read()` bytes into `io.staging_output()` synchronously before returning `Ok(stats)`. Do not remove `RecordingExecutor` before adding a test that asserts `io.staging_output()` is non-zero after a real (or cpu-backend) kernel run.
**Warning signs:** `eval_raw` returns non-zero `RawEvalSummary.not0 == 0` even when the direct safe API path shows non-zero output.

### Pitfall 4: SHADER_F64 absent from WGSL — silent f32 precision loss
**What goes wrong:** Launching a `f64` kernel on wgpu without `SHADER_F64` either silently runs in f32 (wrong results) or causes a GPU validation error. Oracle tolerances require ~12 significant digits; f32 gives ~7.
**Why it happens:** WebGPU/WGSL does not support f64 arithmetic natively. `SHADER_F64` requires Vulkan with `VK_KHR_shader_float64`; Metal and WebGPU targets do not have it.
**How to avoid:** Gate every f64 kernel dispatch on the `SHADER_F64` capability check (D-09). Return `UnsupportedApi` with `wgpu-capability:missing_shader_f64` when absent. CPU backend is always f64 — use it as the oracle comparison path (EXEC-09, VERI-06).
**Warning signs:** Oracle comparisons fail with `max_abs_error` ~1e-7 (f32 noise floor) instead of within 1e-11 tolerance.

### Pitfall 5: `TransferPlan::stage_device_buffers` removal breaks `transfer_bytes` accounting
**What goes wrong:** When `stage_device_buffers` is removed, the `transfer_bytes` metric in `ExecutionStats` goes to zero unless explicitly recomputed from the actual `client.create` buffer sizes.
**Why it happens:** The existing `transfer_bytes` in `TransferPlan` is computed from the host-side probe; once the probe call is gone, nothing repopulates it.
**How to avoid:** Compute `transfer_bytes` inside each family launch function as the sum of all `client.create` buffer sizes plus the output buffer size, and return it in `ExecutionStats`.
**Warning signs:** `stats.transfer_bytes == 0` after `execute()` completes successfully; OOM accounting becomes inaccurate.

---

## Code Examples

### Complete client API cycle (wgpu path)

```rust
// Source: docs/manual/Cubecl/Cubecl_vector.md (project reference — HIGH confidence)

use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

#[cube(launch)]
fn array_multiply_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let tid = ABSOLUTE_POS;
    if tid < input.len() && tid < output.len() {
        let multiplier = (tid + 2) as f32;
        output[tid] = input[tid] * multiplier;
    }
}

let device = WgpuDevice::default();
let client = WgpuRuntime::client(&device);

let input_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
let input_bytes = bytemuck::cast_slice(&input_data);
let output_size = input_data.len() * std::mem::size_of::<f32>();

// Bind handles to named lets before ArrayArg construction (Pitfall 1 avoidance)
let input_buffer = client.create(input_bytes);
let output_buffer = client.empty(output_size);

array_multiply_kernel::launch::<WgpuRuntime>(
    &client,
    CubeCount::Static(1, 1, 1),
    CubeDim::new(4, 1, 1),
    unsafe { ArrayArg::from_raw_parts::<f32>(&input_buffer, input_data.len(), 1) },
    unsafe { ArrayArg::from_raw_parts::<f32>(&output_buffer, input_data.len(), 1) },
);

let output_bytes = client.read(vec![output_buffer.binding()]);
let output_data: &[f32] = bytemuck::cast_slice(&output_bytes[0]);
```

### ResolvedBackend dispatch in a family function

```rust
// Source: .planning/research/ARCHITECTURE.md — Part 5 (HIGH confidence)

pub fn launch_one_electron(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    let output_count = staging.len();
    let output_bytes = output_count * std::mem::size_of::<f64>();

    // Create buffers per-arm
    let (output_buf, exp_buf) = match backend {
        ResolvedBackend::Wgpu(c) => {
            (c.empty(output_bytes), c.create(bytemuck::cast_slice(exponents(plan))))
        }
        ResolvedBackend::Cpu(c) => {
            (c.empty(output_bytes), c.create(bytemuck::cast_slice(exponents(plan))))
        }
    };

    // Launch kernel per-arm
    match backend {
        ResolvedBackend::Wgpu(c) => {
            one_electron_stub::launch::<WgpuRuntime>(
                c, CubeCount::Static(1,1,1), CubeDim::new(output_count as u32,1,1),
                unsafe { ArrayArg::from_raw_parts::<f64>(&exp_buf, exponent_count, 1) },
                unsafe { ArrayArg::from_raw_parts::<f64>(&output_buf, output_count, 1) },
            );
        }
        ResolvedBackend::Cpu(c) => {
            one_electron_stub::launch::<CpuRuntime>(
                c, CubeCount::Static(1,1,1), CubeDim::new(output_count as u32,1,1),
                unsafe { ArrayArg::from_raw_parts::<f64>(&exp_buf, exponent_count, 1) },
                unsafe { ArrayArg::from_raw_parts::<f64>(&output_buf, output_count, 1) },
            );
        }
    }

    // Read back synchronously
    let raw = match backend {
        ResolvedBackend::Wgpu(c) => c.read(vec![output_buf.binding()]),
        ResolvedBackend::Cpu(c) => c.read(vec![output_buf.binding()]),
    };
    let values: &[f64] = bytemuck::cast_slice(&raw[0]);
    staging.copy_from_slice(values);

    Ok(ExecutionStats {
        transfer_bytes: output_bytes + exp_bytes.len(),
        chunk_count: 1,
        ..Default::default()
    })
}
```

### CubeClExecutor::execute revised skeleton

```rust
// Source: .planning/research/ARCHITECTURE.md — Part 3 (HIGH confidence)

fn execute(
    &self,
    plan: &ExecutionPlan<'_>,
    io: &mut ExecutionIo<'_>,
) -> Result<ExecutionStats, cintxRsError> {
    self.ensure_supported_family(plan)?;
    io.ensure_output_contract()?;

    // Ownership contract enforcement (unchanged)
    if io.backend_output_ownership() != OutputOwnership::BackendStagingOnly { ... }
    if io.final_write_ownership() != OutputOwnership::CompatFinalWrite { ... }

    let specialization = SpecializationKey::from_plan(plan);
    let _resident = self.resident_cache.resident_metadata(...);

    // Resolve or retrieve cached backend client
    let backend = self.resolve_backend(&plan.workspace.backend_intent)?;

    // launch_family now takes (&ResolvedBackend, plan, spec, staging)
    let staging = io.staging_output();
    let mut stats = kernels::launch_family(&backend, plan, &specialization, staging)?;

    transform::apply_representation_transform(plan.representation, staging)?;

    stats.peak_workspace_bytes = stats.peak_workspace_bytes.max(io.workspace().len());
    stats.planned_batches = io.chunk().work_unit_count.max(1);
    Ok(stats)
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `TransferPlan::stage_device_buffers` (host Vec probe) | `client.create()`/`client.empty()`/`client.read()` per family | Phase 7 (v1.1) | Eliminates stub path; real GPU memory allocation |
| `RecordingExecutor<E>` captures staging after execute | `io.staging_output()` populated by `client.read()` before execute returns | Phase 7 (v1.1) | Removes indirection; compat reads staging directly |
| `CUBECL_RUNTIME_PROFILE=cpu` env var (Phase 2) | `CINTX_BACKEND=wgpu|cpu` env var (Phase 7) | Phase 7 (v1.1) | Project-owned variable; decoupled from CubeCL internals |
| Single wgpu-only backend | `ResolvedBackend` enum with Wgpu + Cpu arms | Phase 7 (v1.1) | Enables CI without GPU via `--features cpu` |

**Deprecated/outdated:**
- `TransferPlan::stage_device_buffers` and `stage_output_buffer`: removed in Phase 7; `TransferPlan` retained as planning/metrics struct.
- `RecordingExecutor` in both `cintx-compat/src/raw.rs` and `cintx-rs/src/api.rs`: deleted in Phase 7.
- `CubeClExecutor::preflight_wgpu` as a standalone method: absorbed into `resolve_backend()` in `backend/mod.rs`.

---

## Open Questions

1. **Does `CpuRuntime::client(&CpuDevice::default())` also require an `OnceLock` guard?**
   - What we know: `WgpuRuntime` panics on double-init. CPU backend is a different runtime type with different internal registration semantics.
   - What's unclear: Whether `CpuRuntime` shares the same panic-on-double-init behavior.
   - Recommendation: Apply the same `Mutex<HashMap>` keying pattern defensively until confirmed otherwise. Cost is minimal.

2. **Phase 7 writes stub zeros — does the `eval_raw` non-zero test need to be relaxed?**
   - What we know: `cintx-compat/src/raw.rs` includes a test asserting `bytes_written > 0`. Phase 6 decision (assert bytes_written > 0 for staging path tests) locked this.
   - What's unclear: Whether this test passes when stub kernels still return zeros via the new direct client path.
   - Recommendation: The test should be updated to assert that `execute()` succeeds (not errors) and staging is populated (length matches expected), with value correctness deferred to Phase 9. Or gate the non-zero value assertion behind `#[cfg(feature = "cpu")]` with a real kernel body.

3. **Where does `preflight_wgpu` / wgpu bootstrap live after `backend/mod.rs` is added?**
   - What we know: `runtime_bootstrap.rs` currently owns the OnceLock and all wgpu bootstrap logic. `backend/wgpu.rs` is the new home per ARCHITECTURE.md.
   - What's unclear: Whether `runtime_bootstrap.rs` can be deprecated in-phase or kept as a thin wrapper.
   - Recommendation: Keep `runtime_bootstrap.rs` as a thin re-export shim initially. Full removal can happen in a follow-up clean-up once all callers are updated.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain | All compilation | ✓ | 1.92.0 (pinned 1.94.0 in rust-toolchain.toml) | — |
| `cargo` | Build and test | ✓ | 1.92.0 | — |
| `cargo-nextest` | Faster parallel test execution | ✗ | — | `cargo test` (standard) |
| wgpu-capable GPU | wgpu backend tests | Unknown (WSL2 environment) | — | CPU backend via `--features cpu` |
| `cubecl-cpu 0.9.0` | CPU backend (`cpu` feature) | ✓ (in Cargo.lock as transitive dep) | 0.9.0 | — |
| `bytemuck 1.25.0` | Buffer byte conversion | ✓ (in Cargo.lock as transitive dep) | 1.25.0 | — |

**Missing dependencies with no fallback:**
- None that block the core rewrite.

**Missing dependencies with fallback:**
- `cargo-nextest`: Use `cargo test` instead. Install separately if needed: `cargo install cargo-nextest`.
- wgpu GPU adapter (WSL2 may lack GPU passthrough): Use `--features cpu` for all test/verification. All oracle parity tests are designed to run without GPU via the CPU backend (EXEC-09, VERI-06).

**WSL2 GPU note:** The environment is WSL2 (`Linux 6.6.87.2-microsoft-standard-WSL2`). wgpu adapter availability depends on whether GPU passthrough (via Mesa/DirectX translation layer) is configured. The phase is designed to succeed without GPU: VERI-06 explicitly routes oracle parity through the CPU backend. Do not assume GPU is available in CI.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in test (`#[test]`) via `cargo test` |
| Config file | `rust-toolchain.toml` pins `1.94.0` |
| Quick run command | `cargo test -p cintx-cubecl --features cpu 2>&1 \| tail -20` |
| Full suite command | `cargo test --workspace --features cpu 2>&1 \| tail -40` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EXEC-06 | Executor calls `client.create()`/`client.empty()`/`client.read()` — not `stage_device_buffers` | unit | `cargo test -p cintx-cubecl --features cpu execute_uses_direct_client_api` | ❌ Wave 0 |
| EXEC-07 | `RecordingExecutor` deleted; `eval_raw` reads staging directly | unit | `cargo test -p cintx-compat --features cpu eval_raw_reads_staging_directly` | ❌ Wave 0 |
| EXEC-08 | `ResolvedBackend::from_intent` returns Wgpu arm for Wgpu intent, Cpu arm for Cpu intent | unit | `cargo test -p cintx-cubecl --features cpu resolved_backend_from_intent_selects_correct_arm` | ❌ Wave 0 |
| EXEC-09 | Tests run under `--features cpu` without GPU hardware | integration | `cargo test -p cintx-cubecl --features cpu` (all tests pass, no GPU required) | ❌ Wave 0 |
| VERI-06 | `SHADER_F64` absent → `UnsupportedApi` with `wgpu-capability:missing_shader_f64` | unit | `cargo test -p cintx-cubecl shader_f64_absent_returns_unsupported_api` | ❌ Wave 0 |
| EXEC-06 (regression) | Existing executor tests pass with new signature | regression | `cargo test -p cintx-cubecl` (existing tests in `executor.rs` all pass) | ✅ existing |
| EXEC-08 (env var) | `CINTX_BACKEND=cpu` selects Cpu arm; unset selects Wgpu arm | unit | `cargo test -p cintx-cubecl --features cpu backend_env_var_selection` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `cargo test -p cintx-cubecl --features cpu 2>&1 | tail -20`
- **Per wave merge:** `cargo test --workspace --features cpu 2>&1 | tail -40`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `crates/cintx-cubecl/src/backend/mod.rs` — `ResolvedBackend` unit tests: `from_intent` selects correct arm, env var routing, Mutex cache prevents double-init
- [ ] `crates/cintx-cubecl/src/backend/cpu.rs` — CPU client bootstrap test (requires `--features cpu`)
- [ ] `crates/cintx-cubecl/src/executor.rs` — test: `execute_uses_direct_client_api` asserts `stage_device_buffers` is gone; direct client path runs
- [ ] `crates/cintx-compat/src/raw.rs` — test: `eval_raw_reads_staging_directly` asserts no `RecordingExecutor` in call chain; staging populated before `execute` returns
- [ ] `crates/cintx-cubecl/src/kernels/mod.rs` — update existing `family_registry_resolves_base_slice` test to compile with new `FamilyLaunchFn` signature
- [ ] Feature install: `cpu = ["cubecl/cpu"]` in `cintx-cubecl/Cargo.toml` — required before any `--features cpu` test can compile

---

## Project Constraints (from CLAUDE.md)

| Directive | Impact on Phase 7 |
|-----------|------------------|
| CubeCL is the primary compute backend; host CPU work limited to planning/validation/marshaling | All kernel dispatch stays inside `cintx-cubecl`; host side only marshals bytes and calls `client.create`/`client.read` |
| Public library errors use `thiserror` v2 (`cintxRsError`) | `ResolvedBackend::from_intent` returns `Result<_, cintxRsError>`; no `anyhow` in public-facing code |
| CLI/xtask/oracle use `anyhow` | `cintx-oracle` and `xtask` oracle parity test helpers can use `anyhow` for error context |
| Deliverables written to `/mnt/data` are mandatory | Oracle parity artifacts (if any are generated in Phase 7) must target `/mnt/data` |
| Safe API first, raw compat second, C ABI third | `RecordingExecutor` removal must not break the raw compat API; `eval_raw` must remain functional |
| Error handling: `BackendCapabilityToken` carries fingerprint | `ResolvedBackend::from_intent` should update/verify the capability token fingerprint after resolving the wgpu client |
| GSD workflow enforcement: all edits through GSD commands | Confirmed — this research is part of the GSD workflow |

---

## Sources

### Primary (HIGH confidence)

- `docs/manual/Cubecl/Cubecl_vector.md` (project reference) — canonical client API pattern: `client.create`, `client.empty`, `client.read`, `ArrayArg::from_raw_parts`
- `.planning/research/ARCHITECTURE.md` — ResolvedBackend enum design, 7-step build order, full data flow diagram, integration point analysis
- `.planning/research/STACK.md` — CubeCL client API pattern variants, CPU backend initialization (`CpuRuntime::client`), `bytemuck` usage, `cubecl/cpu` feature flag
- `.planning/research/PITFALLS.md` — ArrayArg handle lifetime (Pitfall 1), double-init panic (Pitfall 2), RecordingExecutor capture race (Pitfall 7), f64/SHADER_F64 (Pitfall 3)
- `crates/cintx-cubecl/src/executor.rs` — current `CubeClExecutor` implementation; `preflight_wgpu`, `stage_device_buffers` call, ownership contract
- `crates/cintx-cubecl/src/runtime_bootstrap.rs` — `OnceLock` double-init guard, `bootstrap_wgpu_runtime`, `collect_feature_names` (includes `SHADER_F64`)
- `crates/cintx-cubecl/src/kernels/mod.rs` — current `FamilyLaunchFn` signature, `launch_family`, family dispatch map
- `crates/cintx-compat/src/raw.rs` — `RecordingExecutor` definition (lines 21-71)
- `crates/cintx-rs/src/api.rs` — `RecordingExecutor` usage (lines 158, 394-430)
- `crates/cintx-cubecl/Cargo.toml` — current feature flags and dependencies
- `.planning/phases/07-executor-infrastructure-rewrite/07-CONTEXT.md` — locked decisions D-01 through D-12

### Secondary (MEDIUM confidence)

- `crates/cintx-runtime/src/options.rs` — `BackendKind::Cpu` already exists; `BackendIntent` struct confirmed
- `crates/cintx-runtime/src/dispatch.rs` — `OutputOwnership::BackendStagingOnly` / `CompatFinalWrite` contract confirmed unchanged

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all library choices verified in Cargo.lock and existing source
- Architecture: HIGH — `ResolvedBackend` enum design, signature changes, and removal targets directly verified against current source
- Pitfalls: HIGH for CubeCL API specifics (verified from project reference files and codebase); MEDIUM for `CpuRuntime` double-init behavior (pattern applied defensively)

**Research date:** 2026-04-03
**Valid until:** 2026-05-03 (CubeCL 0.9.x is stable; check if 0.10+ releases before next phase)
