# Architecture Patterns

**Project:** cintx v1.1 — CubeCL Direct Client API & Multi-Backend Design
**Researched:** 2026-04-02
**Confidence:** HIGH (primary evidence from direct codebase inspection + CubeCL runtime source)

---

## Part 1: Existing Component Map (v1.0 baseline)

| Component | Responsibility | Communicates With |
|-----------|----------------|-------------------|
| `cintx-core` | Domain types (`Atom`, `Shell`, `BasisSet`, `OperatorId`, `cintxRsError`) | All crates |
| `cintx-ops` | Manifest lock resolver, `OperatorDescriptor`, symbol metadata | Runtime, xtask, oracle |
| `cintx-runtime` | Planner, validator, scheduler, workspace, `BackendExecutor` trait, `ExecutionIo`, `DispatchDecision`, `BackendKind`/`BackendIntent` | Core, cubecl, compat, rs |
| `cintx-cubecl` | `CubeClExecutor` impl, kernel family stubs, `TransferPlan`, `SpecializationKey`, `runtime_bootstrap`, resident cache | Runtime (BackendExecutor contract) |
| `cintx-compat` | Raw `atm`/`bas`/`env` compat API, `RecordingExecutor` wrapper, layout writers | Runtime, cubecl (via RecordingExecutor), capi |
| `cintx-rs` | Safe Rust facade, builders, prelude | Runtime, compat |
| `cintx-capi` | Optional C ABI shim | Compat, rs |
| `cintx-oracle` | Vendored libcint build, comparison harness | Compat, xtask |
| `xtask` | Manifest audit, oracle update, CI gates | ops, runtime, compat |

---

## Part 2: v1.1 Target Architecture

### What Changes vs. What Stays

**Stays identical (no change to public contracts):**
- `BackendExecutor` trait in `cintx-runtime` (`supports`, `query_workspace`, `execute`)
- `ExecutionIo` staging/workspace contract (backend writes staging, compat writes final flat)
- `DispatchDecision` ownership model (`BackendStagingOnly` / `CompatFinalWrite`)
- `ExecutionPlan`, `WorkspaceQuery`, `ExecutionOptions`, `BackendIntent`, `BackendCapabilityToken`
- `SpecializationKey` and `TransferPlan` structures
- All `cintx-compat` public surface
- All `cintx-rs` public surface

**Changes inside `cintx-cubecl` only:**
- `CubeClExecutor::execute` internals: replace `TransferPlan::stage_device_buffers` (host-side probe) with real `client.create`/`client.empty`/kernel launch/`client.read` cycle
- `runtime_bootstrap.rs`: extend to also bootstrap `CpuRuntime` (for `BackendKind::Cpu`)
- Each kernel family module: replace stub return with `#[cube(launch)]` kernel invocation
- `RecordingExecutor` in `cintx-compat`: remove once executor writes real values directly to staging

**New components inside `cintx-cubecl`:**
- `backend/mod.rs` — backend trait + enum dispatch (see section 4)
- `backend/wgpu.rs` — `WgpuRuntime` client bootstrap and kernel launch helpers
- `backend/cpu.rs` — `CpuRuntime` client bootstrap (same launch path, different runtime type)

---

## Part 3: Replacing CubeClExecutor Internals

### Current flow (stub)

```
executor.execute(plan, io)
  -> preflight_wgpu(plan)                 // capability check only
  -> TransferPlan::stage_device_buffers() // host Vec probe, no real GPU allocation
  -> kernels::launch_family(...)          // stub returns zeros
  -> transform::apply_representation_transform(staging)
```

### Target flow (direct client API)

```
executor.execute(plan, io)
  -> preflight (capability token check)
  -> resolve_client(plan.workspace.backend_intent)   // -> ComputeClient<R>
  -> marshal_inputs(plan, client)                    // client.create(input_bytes)
  -> output_buf = client.empty(output_byte_count)    // output buffer reservation
  -> kernels::launch_family(client, plan, spec, input_bufs, output_buf)
       -> family_kernel::launch::<R>(&client, cube_count, cube_dim, ArrayArg::from_raw_parts(...))
  -> raw_bytes = client.read(vec![output_buf.binding()])
  -> copy bytemuck::cast_slice(&raw_bytes[0]) into io.staging_output()
  -> transform::apply_representation_transform(staging)
```

### Key invariant preserved

The staging slice in `io.staging_output()` is the only host-visible output from the executor. The `client.read()` result feeds directly into that slice. The compat layer still owns the final flat write from staging to the caller's buffer. No change to `OutputOwnership` semantics.

### What happens to RecordingExecutor

`RecordingExecutor` in `cintx-compat/src/raw.rs` exists because the stub executor never wrote real values into staging — the recording wrapper captured them after the fact. Once the real executor writes correct values into `io.staging_output()` during `execute()`, `RecordingExecutor` can be removed and `eval_raw` can read `io.staging_output()` directly through the normal execute path. This is a deletion, not a rewrite.

---

## Part 4: Backend Trait / Enum Design

### Recommended design: enum dispatch at bootstrap, generic-free executor

The problem with generics over `Runtime` (e.g. `CubeClExecutor<R: Runtime>`) is that `BackendExecutor` is already an object-safe trait used as `&dyn BackendExecutor`. Introducing a generic parameter breaks object safety.

Instead: resolve the backend at bootstrap time and store the result as an enum of pre-resolved client handles.

```rust
// crates/cintx-cubecl/src/backend/mod.rs

pub enum ResolvedBackend {
    Wgpu(ComputeClient<WgpuRuntime>),
    Cpu(ComputeClient<CpuRuntime>),
    // Future: Cuda(ComputeClient<CudaRuntime>), Metal(ComputeClient<MetalRuntime>)
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

Kernel launch is done via a macro or a dispatch function that matches on the enum and calls `launch::<WgpuRuntime>` or `launch::<CpuRuntime>` in each arm. This avoids dynamic dispatch at the compute level while keeping `CubeClExecutor` object-safe through `BackendExecutor`.

```rust
// Dispatch helper for launching a kernel regardless of backend
macro_rules! launch_on_backend {
    ($backend:expr, $kernel:path, $client:ident, $($args:expr),*) => {
        match $backend {
            ResolvedBackend::Wgpu(ref $client) => $kernel::<WgpuRuntime>($($args),*),
            ResolvedBackend::Cpu(ref $client) => $kernel::<CpuRuntime>($($args),*),
        }
    }
}
```

### Where to resolve the backend

The `CubeClExecutor` holds a `DeviceResidentCache` today. In v1.1 it should also hold a `OnceLock<ResolvedBackend>` (or a thread-safe lazy cache keyed on `BackendIntent`). The first call to `execute` with a given intent resolves and caches the backend. This mirrors the existing `DEFAULT_REPORT` OnceLock in `runtime_bootstrap.rs`.

```rust
pub struct CubeClExecutor {
    resident_cache: DeviceResidentCache,
    backend_cache: Mutex<HashMap<BackendIntentKey, ResolvedBackend>>,
}
```

`BackendIntentKey` is a cheap hashable representation of `(BackendKind, selector_string)`.

### Adding CubeCL CPU dependency

Add to `crates/cintx-cubecl/Cargo.toml`:

```toml
cubecl-cpu = { version = "0.9.0", optional = true }

[features]
cpu-backend = ["cubecl-cpu"]
```

The `BackendKind::Cpu` arm is conditionally compiled behind `#[cfg(feature = "cpu-backend")]` and returns a clear `UnsupportedApi` error when the feature is absent (matching the existing `with-4c1e` pattern).

---

## Part 5: Buffer Lifecycle Placement

### Where create/read/empty live

Buffer lifecycle belongs inside the kernel family launch functions, not in `CubeClExecutor::execute` or `TransferPlan`. The reasoning:

- Each kernel family has different input shape, number of buffers, and output layout. A generic "create all inputs" loop in the executor would need to reconstruct family-specific knowledge.
- `SpecializationKey` already carries family identity and angular momentum. The family launch function already has full context.
- Keeping buffer ops inside each family module makes the staging→host copy a local, reviewable operation per family.

### Revised kernel family signature

```rust
// Family launch functions receive the resolved backend client
pub fn launch_one_electron(
    client: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    // 1. Marshal inputs
    let exp_bytes  = bytemuck::cast_slice(shell_exponents(plan));
    let coeff_bytes = bytemuck::cast_slice(shell_coefficients(plan));
    let coord_bytes = bytemuck::cast_slice(shell_coordinates(plan));
    let output_count = staging.len();
    let output_bytes = output_count * size_of::<f64>();

    // 2. Create device buffers
    let (exp_buf, coeff_buf, coord_buf, output_buf) = match client {
        ResolvedBackend::Wgpu(c) => (
            c.create(exp_bytes), c.create(coeff_bytes),
            c.create(coord_bytes), c.empty(output_bytes),
        ),
        ResolvedBackend::Cpu(c) => (
            c.create(exp_bytes), c.create(coeff_bytes),
            c.create(coord_bytes), c.empty(output_bytes),
        ),
    };

    // 3. Launch kernel
    launch_on_backend!(client, one_electron_kernel::launch, c,
        &c, cube_count, cube_dim,
        unsafe { ArrayArg::from_raw_parts::<f64>(&exp_buf, exp_count, 1) },
        unsafe { ArrayArg::from_raw_parts::<f64>(&coeff_buf, coeff_count, 1) },
        unsafe { ArrayArg::from_raw_parts::<f64>(&coord_buf, coord_count, 1) },
        unsafe { ArrayArg::from_raw_parts::<f64>(&output_buf, output_count, 1) },
    );

    // 4. Read back — single synchronous call
    let raw = match client {
        ResolvedBackend::Wgpu(c) => c.read(vec![output_buf.binding()]),
        ResolvedBackend::Cpu(c)  => c.read(vec![output_buf.binding()]),
    };
    let values: &[f64] = bytemuck::cast_slice(&raw[0]);
    staging.copy_from_slice(values);

    Ok(ExecutionStats { transfer_bytes: output_bytes + exp_bytes.len() + ..., .. })
}
```

`TransferPlan` is retained as a planning/metrics struct (it correctly computes `staging_elements`, `transfer_bytes`, and `workspace_bytes` from the plan). It stops being the thing that allocates or stages device buffers. The call to `transfer_plan.stage_device_buffers()` in `executor.rs` is removed.

### f64 on GPU

CubeCL kernels default to `f32`. Integral values require `f64` (double precision). The wgpu `SHADER_F64` feature (already collected in `capability.rs`) must be present and the kernel must use `f64` element type. Check the `WgpuCapabilitySnapshot` feature list in the preflight report before launching a `f64` kernel; if absent, fail with a typed `UnsupportedApi` with reason `wgpu-capability:missing_shader_f64`. CPU runtime supports `f64` natively.

---

## Part 6: Kernel Composition with #[cube(launch)]

### Pattern from the reference example

```rust
use cubecl::prelude::*;
use cubecl_wgpu::WgpuRuntime;

#[cube(launch)]
fn my_kernel(input: &Array<f64>, output: &mut Array<f64>) {
    let tid = ABSOLUTE_POS;
    if tid < input.len() {
        output[tid] = /* ... compute ... */;
    }
}
```

The `#[cube(launch)]` macro generates a `my_kernel::launch::<R>` free function parameterized on the runtime `R`. Calling it requires:
- `&ComputeClient<R>` — the client
- `CubeCount` — workgroup count (x, y, z)
- `CubeDim` — workgroup dimensions
- `ArrayArg` per array parameter — wraps a `Handle` with element count and vectorization

### How this composes with kernels/mod.rs

`kernels/mod.rs` currently defines `FamilyLaunchFn` as a function pointer:

```rust
pub type FamilyLaunchFn = fn(
    &ExecutionPlan<'_>,
    &SpecializationKey,
    &TransferPlan,
) -> Result<ExecutionStats, cintxRsError>;
```

This signature must be extended to carry the resolved backend client:

```rust
pub type FamilyLaunchFn = fn(
    &ResolvedBackend,
    &ExecutionPlan<'_>,
    &SpecializationKey,
    &mut [f64],  // staging slice — direct write target
) -> Result<ExecutionStats, cintxRsError>;
```

`TransferPlan` is no longer passed through the launch function pointer since buffer lifecycle is the family's responsibility. The `stage_output_buffer` and `stage_device_buffers` methods on `TransferPlan` become dead code and can be removed in a follow-up.

`launch_family` in `kernels/mod.rs` changes signature to match, and `CubeClExecutor::execute` passes `&resolved_backend` and `io.staging_output()` instead of `&transfer_plan`.

### Workgroup sizing

Each kernel family must derive `CubeCount` and `CubeDim` from the plan. The initial conservative choice: `CubeDim::new(64, 1, 1)` and `CubeCount::Static(ceil(output_elements / 64), 1, 1)`. Specialization-aware sizing (based on `SpecializationKey::shell_angular_momentum`) can be added in a later phase.

---

## Part 7: Updated Data Flow (v1.1)

```
Caller (safe API or compat)
  |
  v
cintx-runtime: query_workspace(op, rep, basis, shells, opts)
  -> resolves BackendIntent from opts
  -> plans chunks, emits WorkspaceQuery (backend_intent stored)

cintx-runtime: evaluate(plan, opts, allocator, executor)
  -> drift detection: opts.backend_intent == plan.workspace.backend_intent
  -> per chunk: staging alloc + workspace alloc
  -> executor.execute(plan, &mut io)
       |
       v
cintx-cubecl: CubeClExecutor::execute(plan, io)
  -> resolve_client(plan.workspace.backend_intent)
       -> ResolvedBackend::Wgpu(client) or ResolvedBackend::Cpu(client)
  -> kernels::launch_family(&backend, plan, &spec, io.staging_output())
       |
       v
  family module (e.g. one_electron.rs):
    -> client.create(input data bytes)    // H2D transfers
    -> client.empty(output byte count)    // output buffer
    -> kernel::launch::<R>(&client, count, dim, ArrayArg...)
    -> client.read([output_buf.binding()]) // D2H readback
    -> copy values into staging slice
    -> return ExecutionStats (transfer_bytes, not0, ...)
       |
       v
  CubeClExecutor: transform::apply_representation_transform(staging)
  -> staging is now correct kernel output values, possibly c2s transformed
  -> io ownership: staging stays BackendStagingOnly

  cintx-runtime: observe_transfer_bytes, observe_not0
  
cintx-compat: RecordingExecutor deleted
  -> eval_raw reads staging through normal io path
  -> layout writer copies from staging into caller's flat buffer (CompatFinalWrite)
```

---

## Part 8: Integration Points with Existing Crates

### cintx-runtime (no API change)

- `BackendExecutor` trait: unchanged. `CubeClExecutor` still implements it.
- `ExecutionIo::staging_output()`: unchanged. Family launch functions write into it.
- `BackendKind::Cpu` already exists in `options.rs`. No new types needed.
- `BackendIntent::selector` already carries adapter selection strings. Extend `runtime_bootstrap.rs` to handle `BackendKind::Cpu` path.

### cintx-cubecl (primary change site)

**New files:**
- `src/backend/mod.rs` — `ResolvedBackend` enum, `BackendIntentKey`, cache logic
- `src/backend/wgpu.rs` — `WgpuRuntime` bootstrap helper (refactored from `runtime_bootstrap.rs`)
- `src/backend/cpu.rs` — `CpuRuntime` bootstrap helper (new, behind `cpu-backend` feature)

**Modified files:**
- `src/executor.rs` — remove `preflight_wgpu` inline call, add `resolve_client`, remove `transfer_plan.stage_device_buffers()`, pass `&backend` to `launch_family`
- `src/kernels/mod.rs` — change `FamilyLaunchFn` signature, update `launch_family`
- `src/kernels/one_electron.rs` — replace stub with real `#[cube(launch)]` kernel + buffer ops
- `src/kernels/two_electron.rs` — same
- `src/kernels/center_2c2e.rs` — same
- `src/kernels/center_3c1e.rs` — same
- `src/kernels/center_3c2e.rs` — same
- `src/transfer.rs` — remove `stage_device_buffers` and `stage_output_buffer` (planning fields retained)
- `src/runtime_bootstrap.rs` — narrow to wgpu-only; bootstrap coordination moved to `backend/`
- `Cargo.toml` — add `cubecl-cpu` optional dep, `cpu-backend` feature

**Unchanged files:**
- `src/specialization.rs` — no change
- `src/resident_cache.rs` — no change
- `src/transform/` — no change
- `src/capability.rs` — no change (still used for preflight report)

### cintx-compat (deletion)

- Remove `RecordingExecutor` struct and `impl BackendExecutor for RecordingExecutor`
- `eval_raw` reads `io.staging_output()` through standard path after `executor.execute`
- No public API change

### Other crates (no change)

`cintx-core`, `cintx-ops`, `cintx-rs`, `cintx-capi`, `cintx-oracle`, `xtask` — zero modifications required for the executor rewrite.

---

## Part 9: Suggested Build Order for the Refactor

Ordered by dependency: each step must compile and pass existing tests before the next begins.

### Step 1: Add `ResolvedBackend` enum (no behavior change)

Create `src/backend/mod.rs` with the `ResolvedBackend` enum skeleton. Wire `from_intent` to call the existing `bootstrap_wgpu_runtime` for wgpu. CPU arm returns `UnsupportedApi` unconditionally (feature stub). No changes to `executor.rs` yet. All existing tests pass.

**Deliverable:** `backend/mod.rs` compiles, `ResolvedBackend::from_intent` works for wgpu.

### Step 2: Extend `FamilyLaunchFn` signature, update `kernels/mod.rs`

Change `FamilyLaunchFn` to accept `&ResolvedBackend` and `&mut [f64]` staging. Update `launch_family`. Update each stub family module to accept the new signature but still return zeros. Update `executor.rs` to call `resolve_client` and pass `&backend` + `io.staging_output()`. Remove `transfer_plan.stage_device_buffers()` call. Remove the `let transfer = ...` line.

**Deliverable:** codebase compiles with new signature, all existing contract tests pass (zeros from stubs still expected).

### Step 3: Implement real GPU kernel for `one_electron`

Write `#[cube(launch)]` kernels for the 1e family (overlap, kinetic, nuclear attraction). Implement real `client.create`/`client.empty`/`launch`/`client.read` cycle in `one_electron.rs`. Add `SHADER_F64` preflight check. Run `cargo test -p cintx-cubecl` on GPU hardware; verify staging values are non-zero for non-trivial input.

**Deliverable:** one_electron family produces real integral values on GPU.

### Step 4: Remove `RecordingExecutor` from `cintx-compat`

With real values flowing into staging through the standard execute path, `eval_raw` no longer needs the recording wrapper. Delete `RecordingExecutor`. Update `eval_raw` to call `executor.execute` directly. Run compat tests.

**Deliverable:** `cintx-compat` compiles without `RecordingExecutor`; compat tests pass.

### Step 5: Implement real GPU kernels for remaining families (2e, 2c2e, 3c1e, 3c2e)

Same pattern as step 3, per family. Each family is independently testable. Oracle comparison gates run after each family.

**Deliverable:** All five base families produce real GPU values.

### Step 6: Add `CpuRuntime` backend (cpu-backend feature)

Add `cubecl-cpu` dependency. Implement `backend/cpu.rs`. Wire `BackendKind::Cpu` in `ResolvedBackend::from_intent`. Test that the same `#[cube(launch)]` kernels run on CPU runtime with `BackendKind::Cpu` intent. Use as oracle comparison path in CI (no GPU required).

**Deliverable:** `cargo test -p cintx-cubecl --features cpu-backend` passes without GPU.

### Step 7: Oracle parity validation

Run the oracle harness against upstream libcint 6.1.3 for all five families across the compatibility matrix. Fix numerical discrepancies. Lock the pass/fail thresholds into CI gates.

**Deliverable:** Oracle gates pass; v1.1 release criteria met.

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Generics on CubeClExecutor for Runtime type

```rust
// BAD — breaks BackendExecutor object safety
pub struct CubeClExecutor<R: Runtime> { ... }
impl<R: Runtime> BackendExecutor for CubeClExecutor<R> { ... }
```

`BackendExecutor` is used as `&dyn BackendExecutor` in `cintx-runtime::evaluate`. Adding a generic parameter makes the struct non-object-safe. Use the `ResolvedBackend` enum approach instead (see section 4).

### Anti-Pattern 2: Moving buffer lifecycle into executor.rs

Putting `client.create`/`client.empty`/`client.read` in a generic loop inside `CubeClExecutor::execute` would require the executor to know each family's input structure. This recreates the problem `TransferPlan` tried to solve. Keep buffer ops inside family modules where family-specific knowledge lives.

### Anti-Pattern 3: Keeping RecordingExecutor after real kernel values arrive

Once kernels write real values into `io.staging_output()`, keeping `RecordingExecutor` adds a confusing indirection. Staging output values are already accessible to the compat layer through the `ExecutionIo` contract. Remove it immediately in step 4.

### Anti-Pattern 4: Skipping the SHADER_F64 preflight check

Libcint integrals require 64-bit floating point. Launching a `f64` kernel on a wgpu device without `SHADER_F64` support produces silent wrong results or a GPU validation error. The capability preflight already collects feature flags. Add the check before any `f64` kernel launch.

### Anti-Pattern 5: Panics from double-init of CubeCL runtime

The existing `OnceLock` cache in `runtime_bootstrap.rs` exists because `cubecl::wgpu::init_setup` panics if called twice for the same device. The `ResolvedBackend` cache must preserve this invariant. Store clients keyed on `(BackendKind, selector)` in a `Mutex<HashMap>` and call init only on first access.

---

## Scalability Considerations

| Concern | Current (v1.1) | Future |
|---------|----------------|--------|
| Backend variants | Wgpu + Cpu via enum arms | Add Cuda/Metal arms; no API surface change |
| Kernel specialization | Single kernel per family | Per-l-value specialized kernels via SpecializationKey |
| Multi-device | Single client per intent | Client pool keyed on intent; compat layer unchanged |
| Async execution | Synchronous `client.read()` | `read_async()` available in CubeCL; expose behind feature flag |
| Memory pressure | OnceLock + chunk planner unchanged | Per-family buffer pool reuse using `client.allocation_mode()` |

---

## Sources

- CubeCL `ComputeClient` API: https://docs.rs/cubecl-runtime/0.9.0/cubecl_runtime/client/struct.ComputeClient.html (HIGH confidence — official docs)
- CubeCL `CpuRuntime` implementation: https://github.com/tracel-ai/cubecl/blob/v0.9.0/crates/cubecl-cpu/src/runtime.rs (HIGH confidence — source)
- Reference `#[cube(launch)]` pattern: `docs/manual/Cubecl/Cubecl_vector.md` in this repository (HIGH confidence — project-provided)
- `BackendExecutor` trait and `ExecutionIo` contract: `crates/cintx-runtime/src/dispatch.rs` (HIGH confidence — codebase)
- `CubeClExecutor` current implementation: `crates/cintx-cubecl/src/executor.rs` (HIGH confidence — codebase)
- `runtime_bootstrap.rs` OnceLock double-init guard: `crates/cintx-cubecl/src/runtime_bootstrap.rs` (HIGH confidence — codebase)
- `RecordingExecutor` in compat: `crates/cintx-compat/src/raw.rs` lines 21–71 (HIGH confidence — codebase)
- `BackendKind::Cpu` exists: `crates/cintx-runtime/src/options.rs` (HIGH confidence — codebase)
- `cubecl-cpu` 0.9.0 published: https://crates.io/crates/cubecl-cpu (MEDIUM confidence — crates.io search; docs.rs build failed but source confirmed on GitHub)
