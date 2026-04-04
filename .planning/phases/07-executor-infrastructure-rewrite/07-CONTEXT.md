# Phase 7: Executor Infrastructure Rewrite - Context

**Gathered:** 2026-04-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 7 rewrites executor internals to use the CubeCL client API directly, introduces a ResolvedBackend enum for multi-backend dispatch (wgpu + cpu), and resolves the f64 precision strategy. This is the prerequisite gate for all real kernel work in Phases 8-10. RecordingExecutor removal and buffer lifecycle changes are in scope.

</domain>

<decisions>
## Implementation Decisions

### Backend enum design
- **D-01:** Use a `ResolvedBackend` enum with `Wgpu(ComputeClient<WgpuRuntime>)` and `Cpu(ComputeClient<CpuRuntime>)` arms. Per-arm kernel dispatch via match. Future cuda/rocm/metal backends add new enum arms.
- **D-02:** This approach is required because `BackendExecutor` is used as `&dyn BackendExecutor` in `planner::evaluate`, which rules out generics on `CubeClExecutor<R: Runtime>`.

### CPU backend integration
- **D-03:** Both wgpu and cpu backends must pass oracle parity independently. Tests run on both. This is the strongest correctness guarantee.
- **D-04:** Backend selection is a runtime choice made by the user via `CINTX_BACKEND=wgpu|cpu` environment variable. The executor reads this at init time to resolve which `ResolvedBackend` arm to construct.
- **D-05:** ~~Both backends are always compiled -- no feature gate.~~ **REVISED (checker review):** EXEC-09 requires `cpu = ["cubecl/cpu"]` as a Cargo feature because `cubecl::cpu::CpuRuntime` only exists when cubecl's `cpu` feature is enabled -- the type literally does not compile without it. Resolution: the `cpu` feature exists per EXEC-09 and is declared as a **default feature** (`default = ["cpu"]`) so both backends compile by default, honoring D-05's original intent that both backends are always available. The `#[cfg(feature = "cpu")]` gate on the `Cpu` enum arm is a technical necessity (the type does not exist without the feature), not a user-facing opt-in. The env var controls which backend is *used* at runtime.

### CubeCL client API pattern
- **D-06:** Executor internals use `WgpuRuntime::client(&device)` / `CpuRuntime::client(&device)` directly. Buffer management uses `client.create()`, `client.empty()`, `client.read()`. Kernels use `#[cube(launch)]` with `ArrayArg::from_raw_parts`. Reference pattern: `docs/manual/Cubecl/Cubecl_vector.md`.
- **D-07:** Buffer lifecycle (create/read/empty) lives inside each kernel family module, not centralized in the executor. Each family has different input shapes and buffer counts.

### RecordingExecutor removal
- **D-08:** RecordingExecutor is deleted in this phase. Once the executor uses direct `client.read()` for buffer retrieval, the recording wrapper is unnecessary. Staging output flows directly from the client read result into `io.staging_output()`.

### f64 precision strategy
- **D-09:** Both backends must produce f64-precision results. wgpu path gates on `SHADER_F64` capability and returns `UnsupportedApi` when absent. CPU backend always supports f64 natively.

### Carried forward from Phase 5
- **D-10:** Backend auto-selects wgpu when `CINTX_BACKEND` is unset; fails closed with typed error when no valid adapter/capability is available (Phase 5 D-01).
- **D-11:** Backend intent is control-plane metadata carried via runtime options (Phase 5 D-03).
- **D-12:** Staging ownership: backend staging-only, compat retains final caller-visible flat writes (Phase 5 D-06).

### Claude's Discretion
- Exact `ResolvedBackend` enum field names and module placement within cintx-cubecl
- How `bytemuck` promotion to direct dep is handled (already in Cargo.lock)
- Exact error messages for `SHADER_F64` absence
- Internal helper functions for client initialization and buffer marshaling

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### CubeCL client API pattern
- `docs/manual/Cubecl/Cubecl_vector.md` — Reference implementation pattern for direct client API usage
- `docs/cubecl_error_guideline.md` — CubeCL error patterns and `#[cube]` constraints (if exists)

### Executor and runtime (files to modify)
- `crates/cintx-cubecl/src/executor.rs` — Current `CubeClExecutor` to be rewritten
- `crates/cintx-cubecl/src/runtime_bootstrap.rs` — Current wgpu bootstrap; will need cpu path
- `crates/cintx-cubecl/src/kernels/mod.rs` — Kernel dispatch; signature change for client API
- `crates/cintx-runtime/src/dispatch.rs` — `BackendExecutor` trait and `ExecutionIo`
- `crates/cintx-runtime/src/planner.rs` — `query_workspace()` and `evaluate()` dispatcher

### RecordingExecutor removal targets
- `crates/cintx-compat/src/raw.rs` — RecordingExecutor defined at line 21; eval_raw uses it
- `crates/cintx-rs/src/api.rs` — Safe facade RecordingExecutor usage

### Phase 5 decisions (carry forward)
- `.planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/05-CONTEXT.md` — D-01 through D-16 locked decisions

### Research
- `.planning/research/ARCHITECTURE.md` — ResolvedBackend design, 7-step build order
- `.planning/research/STACK.md` — CubeCL client API patterns, cubecl-cpu integration
- `.planning/research/PITFALLS.md` — ArrayArg handle lifetime, f64 constraints

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `CubeClExecutor` struct in `executor.rs` — public interface preserved, internals rewritten
- `bootstrap_wgpu_runtime()` in `runtime_bootstrap.rs` — existing wgpu init; extend for cpu
- `WgpuPreflightReport` with `fingerprint` field — reuse for capability checking
- `BackendCapabilityToken` — already carries fingerprint, can carry f64 capability flag

### Established Patterns
- `BackendExecutor` trait used as `&dyn BackendExecutor` — object safety constraint drives enum approach
- `ExecutionIo::staging_output()` — standard staging handoff path
- `BackendIntent` enum with `Wgpu`/`Cpu` variants — already exists from Phase 5
- Feature gating pattern: `with-f12`, `with-4c1e` — but backend selection uses env var instead

### Integration Points
- `planner::evaluate()` calls `executor.execute()` — executor interface must remain compatible
- `raw::eval_raw()` wraps executor in RecordingExecutor — must switch to direct `client.read()`
- `api::SessionRequest::evaluate()` — safe facade uses same RecordingExecutor pattern
- `cintx-cubecl/Cargo.toml` — needs `cubecl-cpu` as direct dependency

</code_context>

<specifics>
## Specific Ideas

- User explicitly referenced `docs/manual/Cubecl/Cubecl_vector.md` as the pattern to follow: `WgpuRuntime::client(&device)`, `client.create(input_bytes)`, `client.empty(size)`, `client.read(bindings)`, `ArrayArg::from_raw_parts`
- `#[cube(launch)]` macro for all kernel functions — no plain Rust inside `#[cube]` bodies
- `CINTX_BACKEND` env var is the user selection mechanism (not API-level BackendIntent)

</specifics>

<deferred>
## Deferred Ideas

- CUDA/ROCm/Metal backend arms — architecture supports them via ResolvedBackend enum, implementation deferred to v1.2+
- Screening/batching optimizations — performance work after correctness proven
- Workgroup sizing strategy — post-v1.1 specialization

</deferred>

---

*Phase: 07-executor-infrastructure-rewrite*
*Context gathered: 2026-04-03*
