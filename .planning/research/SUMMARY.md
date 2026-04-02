# Project Research Summary

**Project:** cintx v1.1
**Domain:** GPU quantum chemistry integral evaluation — CubeCL direct client API, real kernel compute, oracle parity
**Researched:** 2026-04-02
**Confidence:** HIGH (codebase + CubeCL API evidence); MEDIUM (numerical precision, multi-backend scheduling)

## Executive Summary

cintx v1.1 transitions from stub infrastructure to real GPU compute. The v1.0 baseline delivered a complete execution pipeline — safe Rust API, raw compat layer, planner, chunking, staging/fingerprint propagation, and oracle harness — but all kernel family launch functions return zeros. The v1.1 goal is to replace those stubs with real `#[cube(launch)]` kernels that produce libcint 6.1.3-compatible results, validated against the oracle harness at per-family numerical tolerances. The architecture is already correct; the work is purely inside `cintx-cubecl`.

The recommended implementation approach is: (1) rewrite the executor to use CubeCL's direct client API (`client.create`, `client.empty`, `client.read`) with the `ResolvedBackend` enum pattern, (2) implement a CPU backend path so oracle parity tests run in CI without GPU hardware, (3) build Gaussian primitive infrastructure and Boys function in-house as `#[cube]` functions (no viable Rust crate exists), then (4) implement kernel families in dependency order — 1e first, 2e next, then 2c2e/3c1e/3c2e. The only algorithm decision with lasting consequences is choosing Rys quadrature for 2e/2c2e/3c2e and McMurchie-Davidson for 1e/3c1e, which matches the memory footprint and GPU thread structure constraints established by GPU4PySCF and similar production GPU integral codes.

The dominant risk is the f64 / wgpu `SHADER_F64` constraint: WGSL does not support 64-bit shader arithmetic on Metal or WebGPU targets, and the 1e oracle tolerance is 1e-11. This must be resolved before writing a single kernel — the decision is to run oracle parity tests against the CubeCL CPU backend (which supports f64 natively) rather than depending on hardware `SHADER_F64`. A second critical risk is that all Boys function helpers and Obara-Saika/Rys recurrence helpers must be annotated `#[cube]` — calling plain Rust functions from inside `#[cube]` produces a hard compile error (E0433). Both risks are fully characterized and have clear mitigations.

## Key Findings

### Recommended Stack

The stack is locked for v1.1. No new workspace-level crate additions are needed. Two crate-level changes are required in `crates/cintx-cubecl/Cargo.toml`: promote `bytemuck` from transitive to direct dependency (for `cast_slice` in client buffer I/O), and add a `cpu = ["cubecl/cpu"]` feature flag to enable `CpuRuntime` for CI test paths. All Boys function, Rys quadrature, and Obara-Saika recurrence math is implemented in-house as `#[cube]` functions — there are no viable Rust crates for this domain on crates.io as of 2026-04-02.

**Core technologies:**
- `cubecl 0.9.0` (locked): GPU+CPU compute backend — `#[cube(launch)]` macro generates backend-generic kernel launchers; the same kernel compiles for wgpu and cpu runtime without code changes.
- `cubecl-wgpu 0.9.0`: default GPU backend — requires `SHADER_F64` check before any double-precision kernel dispatch.
- `cubecl/cpu` feature: CPU runtime backend — runs real `#[cube]` kernels on host; primary oracle parity path when GPU `SHADER_F64` is unavailable.
- `bytemuck 1.25.0` (promote to direct dep): zero-copy `cast_slice` for `client.create(&[u8])` and `client.read` output deserialization.
- `thiserror 2.0.18`: typed public error surface — `UnsupportedApi` with structured reasons such as `missing_shader_f64`.
- `Rust 1.94.0` (pinned in `rust-toolchain.toml`): reproducible compiler for oracle baseline and manifest audits.

**Critical kernel constraint:** Every function called from inside a `#[cube]` function must itself be annotated `#[cube]`. Standard library math (`f64::sqrt`, `f64::erf`) is host-side only; use CubeCL's built-in Float primitives inside kernels. Violating this produces compile error E0433 with no useful source location.

### Expected Features

See `.planning/research/FEATURES.md` for full detail.

**Must have (table stakes for v1.1):**
- CubeCL direct client API rewrite — prerequisite for all real kernel work; removes the `RecordingExecutor` wrapper from `cintx-compat`.
- Configurable multi-backend support (wgpu + cpu) — CPU backend required for GPU-free CI oracle parity.
- 1e real kernel (overlap, kinetic, nuclear attraction) — simplest family; validates the end-to-end pipeline.
- 2e real kernel (ERI core, Rys quadrature) — most complex; tightest oracle tolerance (1e-12 atol, 1e-10 rtol).
- 2c2e, 3c1e, and 3c2e real kernels — build on Boys/Rys infrastructure from 2e.
- Gaussian primitive evaluation and contraction infrastructure — shared scaffolding all kernels depend on.
- Cart-to-spherical transform (real Condon-Shortley convention, not the current placeholder) — required for any caller requesting `Representation::Spheric`.
- Oracle parity gate passing for all five base families across the family-specific tolerances.

**Should have (competitive differentiators, after oracle parity):**
- Exponential screening (Schwarz / `pdata.cceij` pair pre-filtering) — performance optimization, does not affect correctness.
- Optimizer cache non-zero contraction index lists — throughput improvement for contracted basis sets.
- Batched shell-quartet kernel dispatch — reduces launch overhead for large basis sets.

**Defer to post-v1.1:**
- Spinor representation kernels (complex interleaved output, c2spinor transform).
- F12/STG/YP range-separated operator kernels.
- GTG family — excluded entirely (upstream has known bugs; `resolve_family_name` must never match `"gtg"`).
- Asynchronous public API — all public APIs remain synchronous.
- Host-CPU integral computation outside the CubeCL CPU backend — explicitly excluded by the architecture constraint.

### Architecture Approach

The architecture does not change for external callers. All v1.1 changes are strictly inside `cintx-cubecl`. The `BackendExecutor` trait, `ExecutionIo` staging contract, `DispatchDecision` ownership model, and all `cintx-rs` / `cintx-compat` public surfaces are frozen. The internal change is: replace the stub `TransferPlan::stage_device_buffers` probe with a real `client.create` / kernel launch / `client.read` cycle inside each kernel family module. Backend dispatch uses a `ResolvedBackend` enum (`Wgpu(ComputeClient<WgpuRuntime>)` / `Cpu(ComputeClient<CpuRuntime>)`) that preserves object safety for `&dyn BackendExecutor` without introducing generics into the public crate boundary.

**Major components:**
1. `backend/mod.rs` (new in `cintx-cubecl`) — `ResolvedBackend` enum, `from_intent` factory, `Mutex<HashMap<BackendIntentKey, ResolvedBackend>>` cache to prevent double-init panics.
2. `kernels/{family}.rs` (rewrite stubs) — each family module owns its buffer lifecycle: `client.create` inputs, `client.empty` output, `launch::<R>`, `client.read`, copy into staging slice.
3. `kernels/mod.rs` (signature update) — `FamilyLaunchFn` gains `&ResolvedBackend` and `&mut [f64]` staging; `TransferPlan` is retained for metrics but stops allocating device buffers.
4. `cintx-compat/src/raw.rs` (deletion) — `RecordingExecutor` is removed once executor writes real values into staging before returning.
5. `math/` module (new in `cintx-cubecl`) — Boys function + Rys roots/weights as `#[cube]` functions using CubeCL built-in Float primitives; no `libm` inside kernel code.

**Data flow (v1.1 target):**
```
Caller -> cintx-rs -> cintx-runtime (plan + chunk)
  -> CubeClExecutor::execute
    -> ResolvedBackend::from_intent (singleton per selector)
    -> kernels::launch_family(&backend, plan, &spec, staging)
      -> client.create(input_bytes)       // H2D
      -> client.empty(output_bytes)       // GPU buffer
      -> kernel::launch::<R>(...)         // execute
      -> client.read([output_buf])        // D2H
      -> staging.copy_from_slice(values)  // commit
    -> transform::apply_representation_transform(staging)
  -> cintx-compat layout writer (Cartesian or Spheric final write)
```

### Critical Pitfalls

See `.planning/research/PITFALLS.md` for full detail on all 16 pitfalls.

1. **`ArrayArg` handle lifetime / use-after-free on GPU** (Pitfall 1) — bind every handle to a named `let` before any `ArrayArg` construction; never pass a temporary expression. Silent wrong results (zeros or garbage) with no Rust safety error surfaced.

2. **CubeCL device double-initialization panic** (Pitfall 2) — all device initialization must go through the existing `bootstrap_wgpu_runtime` singleton guard; never call `init_setup` directly from kernel modules. Cache `ResolvedBackend` in a `Mutex<HashMap>` keyed on `(BackendKind, selector)`.

3. **f64 unavailable in WGSL shaders** (Pitfall 3) — `SHADER_F64` is absent on Metal and WebGPU targets; all oracle parity tests must run against the CPU backend (`--features cpu`), not wgpu. Add explicit `UnsupportedApi` gate when f64 is required and `SHADER_F64` is absent; never silently fall back to f32.

4. **Boys function numerical breakdown on GPU** (Pitfall 4) — use upward recurrence only (downward recurrence is unstable for small `x`); validate Boys function standalone against CPU oracle before embedding in any kernel. Nuclear attraction oracle failures while overlap passes is the diagnostic signal.

5. **Recurrence cancellation for l >= 2 shells** (Pitfall 5) — OS recurrence accumulates errors for f-function and higher angular momentum. Use McMurchie-Davidson for `l >= 2`; test near-nuclear geometries explicitly. Failures are geometry-dependent, not random.

6. **Cart-to-sph staging size mismatch** (Pitfall 6) — spherical shells have fewer components than Cartesian (d-shell: 5 sph vs 6 cart). `TransferPlan::chunk_staging_elements` must be updated in lockstep with the real c2s transform; validate staging sizing before replacing the c2s placeholder or oracle reports `INFINITY` error silently.

7. **`RecordingExecutor` removal timing** (Pitfall 7) — do not remove before the inner executor reliably populates staging on return; do not leave it in place after real values flow into staging, as the double-capture path produces silently wrong results in the raw API.

## Implications for Roadmap

Based on combined research, the natural phase structure follows the feature dependency graph in FEATURES.md: infrastructure first, simplest kernel family second, most complex family third, remaining families in parallel, oracle gate closure last.

### Phase 1: Executor Infrastructure Rewrite

**Rationale:** All real kernel work is blocked until the executor uses the direct client API. This is the prerequisite gate — no kernel can write real values into staging until `client.create`/`client.read` replace the stub host-side probe. `RecordingExecutor` removal is synchronized here because its removal depends on the executor reliably populating staging.
**Delivers:** `ResolvedBackend` enum, wgpu and CPU backend bootstrap, `FamilyLaunchFn` signature update (`&ResolvedBackend` + `&mut [f64]` staging), stub kernels still return zeros but now through the real buffer lifecycle path. `RecordingExecutor` deleted from `cintx-compat`.
**Addresses:** TS1 (direct client API), TS2 (configurable multi-backend).
**Avoids:** Pitfalls 1, 2, 7 (handle lifetime, double-init panic, RecordingExecutor capture ordering).
**Research flag:** Standard patterns — CubeCL client API is fully documented in project reference files; `ResolvedBackend` enum pattern is specified in ARCHITECTURE.md. No research-phase needed.

### Phase 2: Gaussian Primitive Infrastructure and Boys Function

**Rationale:** Every kernel family depends on correct primitive evaluation, pair data computation, contraction accumulation, and Boys function. Building this as a validated standalone layer before any family kernel prevents repeated debugging of shared math. The Boys function domain boundaries and Rys polynomial fit table coverage must be confirmed against `libcint-master/src/fmt.c` and `polyfits.c` before any kernel uses them.
**Delivers:** `#[cube]` Boys function (upward recurrence + asymptotic expansion), Rys root/weight polynomial fit tables, Gaussian product center and pair data computation, contraction accumulation loop, `DeviceResidentCache` entries for Boys table and transform matrices.
**Addresses:** TS8 (Gaussian primitive infrastructure), Boys function component of TS4.
**Avoids:** Pitfall 4 (Boys function domain branching instability), Pitfall 14 (CubeCL type inference failures in `#[cube]` helpers).
**Research flag:** Needs `/gsd:research-phase` — Boys function GPU domain branching strategy, Rys polynomial coefficient table coverage limits, and `#[cube]` math primitive constraints need a locked design before implementation. Source: `libcint-master/src/fmt.c`, `polyfits.c`, and CubeCL Float built-in API.

### Phase 3: 1e Real Kernel and Cart-to-Sph Transform

**Rationale:** The 1e family (overlap, kinetic, nuclear attraction) is the simplest kernel family and the fastest path to a real end-to-end pipeline validation. Overlap does not require Boys function — it catches contraction and transform bugs in isolation. Nuclear attraction adds Boys function — it catches Boys precision bugs as a separate signal. Cart-to-sph transform must be implemented alongside 1e because spherical output is the primary representation used by callers, and the staging size contract must be locked before any other family can use spheric output.
**Delivers:** Real `#[cube(launch)]` kernels for `int1e_ovlp`, `int1e_kin`, `int1e_nuc`. Real Condon-Shortley c2s transform with correct staging buffer sizing for `Representation::Spheric`. Oracle parity gate passing for 1e family (atol 1e-11, rtol 1e-9).
**Addresses:** TS3 (1e real kernel), TS9 (cart-to-sph transform).
**Avoids:** Pitfall 3 (f64/WGSL — oracle must run under CPU backend), Pitfall 5 (OS recurrence breakdown for l >= 2), Pitfall 6 (staging size mismatch after real c2s).
**Research flag:** Needs `/gsd:research-phase` — Obara-Saika vs. McMurchie-Davidson algorithm choice for l >= 2, f64 precision strategy for the wgpu path, and c2s coefficient sourcing from `libcint-master/src/cart2sph.c` all need confirmation before coding starts.

### Phase 4: 2e ERI Real Kernel (Rys Quadrature)

**Rationale:** The 2e kernel is the most complex and most computationally important. Implementing it after the 1e pipeline is validated ensures the Boys function, Rys root infrastructure, and staging path are proven in a simpler context first. The 2e kernel has the tightest oracle tolerance (1e-12 atol) and is the most likely to require iteration.
**Delivers:** Real `#[cube(launch)]` kernel for four-center ERI (`int2e_sph`) using Rys quadrature. Boys function from Phase 2 reused directly. Oracle parity gate passing for 2e family.
**Addresses:** TS4 (2e real kernel).
**Avoids:** Pitfall 5 (recurrence cancellation for high-l shells), Pitfall 13 (nondeterministic reduction order breaking oracle flakiness), Pitfall 10 (transfer_bytes undercount in ExecutionStats).
**Research flag:** Strongly recommend `/gsd:research-phase` — Rys quadrature GPU recurrence for four-center ERIs, shell quartet thread mapping, workgroup sizing strategy, and angular momentum specialization for the 2e kernel are complex enough to warrant a dedicated research pass before coding starts.

### Phase 5: 2c2e, 3c1e, and 3c2e Real Kernels and Oracle Gate Closure

**Rationale:** These three families share Boys function and Rys infrastructure from Phases 2 and 4. They have lower individual complexity than the 2e kernel (fewer indices, smaller recurrence, looser oracle tolerances) and can be implemented in parallel after Phase 4 confirms the shared infrastructure is correct. Oracle parity gate closure for all five base families completes v1.1.
**Delivers:** Real kernels for `int2c2e_sph`, `int3c1e_sph`, `int3c2e_sph`. Oracle parity gates passing for all five base families (2c2e/3c2e atol 1e-9, 3c1e atol 1e-7). `Phase2ParityReport` with `mismatch_count == 0` across the compatibility matrix.
**Addresses:** TS5, TS6, TS7 (2c2e, 3c1e, 3c2e real kernels), TS10 (oracle parity gate).
**Avoids:** Pitfall 11 (near-zero oracle tolerance path for symmetry-forbidden integrals), Pitfall 16 (oracle fixture sync — fixtures must be added per family before the family PR is merged).
**Research flag:** Standard patterns — algorithms are straightforward simplifications of the 4c ERI case, building directly on Phase 4 infrastructure. No separate research-phase anticipated unless 3c1e three-center geometry adds unexpected recurrence complexity.

### Phase Ordering Rationale

- Phases 1-2 establish the execution path and shared math before any family kernel is written; this prevents each family from independently re-implementing or debugging the same infrastructure and avoids the same numerical pitfalls being rediscovered five times.
- Phase 3 (1e) before Phase 4 (2e) exploits the fact that 1e overlap has no Boys function dependency, isolating contraction and c2s bugs from precision bugs. The debugging signal from the oracle is cleanest when one variable changes at a time.
- Phases 4 and 5 follow the feature dependency graph exactly: 2e must precede 2c2e/3c2e because the Boys function table and Rys infrastructure are developed and empirically validated for 2e first, then reused.
- The f64 / CPU backend oracle parity strategy threads through all phases: every oracle comparison test must be gated on `--features cpu` until wgpu `SHADER_F64` support is empirically confirmed on CI hardware. This is not optional — silently running f32 with oracle tolerances written for f64 produces an always-failing CI gate.

### Research Flags

Phases requiring `/gsd:research-phase` during planning:
- **Phase 2 (Boys function / primitive infrastructure):** Boys function GPU domain branching strategy, Rys polynomial coefficient table coverage, and `#[cube]` math primitive constraints need a locked design document before implementation begins.
- **Phase 4 (2e ERI kernel):** Rys quadrature GPU recurrence for four-center ERIs, shell quartet thread mapping, and angular momentum specialization are complex enough to warrant a research pass to avoid costly rework.

Phases with standard patterns (skip research-phase):
- **Phase 1 (executor rewrite):** CubeCL client API is fully documented in project reference files; `ResolvedBackend` enum pattern is specified in ARCHITECTURE.md with code examples.
- **Phase 3 (1e kernel):** Overlap and kinetic integrals are well-documented in the quantum chemistry literature; c2s coefficients are in `libcint-master/src/cart2sph.c`. A research-phase is warranted only if the f64/wgpu decision needs hardware data from Phase 1 before proceeding.
- **Phase 5 (2c2e/3c1e/3c2e):** Builds on validated Phase 4 infrastructure; algorithm extensions are straightforward simplifications of the 4c ERI case.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | CubeCL 0.9.0 API confirmed from project reference files and crates.io; `bytemuck` and `cubecl-cpu` already in lockfile as transitive deps; no new workspace crates needed; `bytemuck::Pod` trait bounds confirmed for f32/f64. |
| Features | HIGH | Oracle tolerances sourced directly from `compare.rs` lines 21-31; feature dependency graph confirmed from codebase inspection; algorithm selection (Rys vs. OS vs. McMurchie-Davidson) supported by multiple peer-reviewed GPU quantum chemistry papers. |
| Architecture | HIGH | `ResolvedBackend` enum pattern, `FamilyLaunchFn` signature change, and `RecordingExecutor` deletion all derived from direct codebase inspection of `executor.rs`, `raw.rs`, `transfer.rs`, and `dispatch.rs`. |
| Pitfalls | HIGH (CubeCL-specific), MEDIUM (numerical) | Handle lifetime, double-init panic, and RecordingExecutor timing pitfalls confirmed from codebase. Boys function and recurrence breakdown pitfalls are literature-sourced and require empirical validation during kernel development. |

**Overall confidence:** HIGH for the execution path and infrastructure design; MEDIUM for the numerical precision outcomes on real hardware until oracle parity is empirically confirmed.

### Gaps to Address

- **wgpu f64 on CI hardware:** Whether the CI GPU adapter supports `SHADER_F64` is unknown. The CPU backend oracle path is the primary mitigation, but empirical confirmation on CI hardware is needed in Phase 1. Until confirmed, treat the wgpu backend as f32-only for oracle purposes.
- **Boys function boundary conditions:** The exact domain boundaries for upward recurrence vs. asymptotic expansion per order n are not locked down in research. These must be sourced from `libcint-master/src/fmt.c` during Phase 2 and validated standalone before any kernel uses Boys function output.
- **Rys polynomial fit table coverage:** The `libcint-master/src/polyfits.c` polynomial fits cover roots/weights up to a maximum quadrature degree. If any kernel family requires roots beyond the table, a fallback (numerical root-finding on CPU, uploaded once to `DeviceResidentCache`) is needed. Must be verified during Phase 2 research.
- **Workgroup sizing strategy:** The conservative `CubeDim::new(64, 1, 1)` baseline is used for all Phases 1-5. Per-family and per-angular-momentum specialization is deferred to post-v1.1, but `SpecializationKey` must be designed in Phase 1 to accommodate it without API changes later.

## Sources

### Primary (HIGH confidence)
- CubeCL 0.9.0 client API — `docs/manual/Cubecl/Cubecl_vector.md`, `Cubecl_shared_memory.md`, `Cubecl_multi_compute.md`, `cubecl_reduce_sum.md` (project reference files).
- `#[cube]` constraint (no plain Rust calls from kernel, error E0433) — `docs/manual/Cubecl/cubecl_error_solution_guide/` (project reference file).
- `BackendExecutor` / `ExecutionIo` contract — `crates/cintx-runtime/src/dispatch.rs` (codebase).
- `CubeClExecutor` current implementation — `crates/cintx-cubecl/src/executor.rs` (codebase).
- `RecordingExecutor` — `crates/cintx-compat/src/raw.rs` lines 21-71 (codebase).
- Oracle tolerance constants — `crates/cintx-oracle/src/compare.rs` lines 21-31 (codebase).
- libcint Boys function reference — `libcint-master/src/fmt.c`, `rys_roots.c`, `rys_wheeler.c` (vendored source).
- `cubecl-cpu 0.9.0` and `bytemuck 1.25.0` in lockfile — `Cargo.lock` (local evidence).
- wgpu `SHADER_F64` feature status — https://github.com/gfx-rs/wgpu/issues/1143.
- CubeCL 0.9.0 feature list (`cpu`, `wgpu`, `cuda`, `hip`) — crates.io API.

### Secondary (MEDIUM confidence)
- GPU4PySCF Rys quadrature GPU implementation — https://arxiv.org/html/2407.09700v1.
- GPU Boys function gridded Taylor expansion — https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.8328.
- McMurchie-Davidson GPU ERI — https://www.mdpi.com/2076-3417/15/5/2572.
- 3-center ERI GPU implementation — https://www.researchgate.net/publication/396374573.
- TeraChem f-function McMurchie-Davidson GPU — https://arxiv.org/html/2406.14920v1.
- Multi-backend generic dispatch — derived from CubeCL matmul example (`docs/manual/Cubecl/cubecl_matmul_gemm_example.md`).
- libcint paper (DRK algorithm, cart2sph) — https://ar5iv.labs.arxiv.org/html/1412.0649.

### Tertiary (LOW confidence)
- CUDA/Metal/ROCm backend gotchas — insufficient hardware coverage to verify; no CI hardware validation done. Address during backend expansion post-v1.1.
- Boys function crate (`boys` on crates.io) — confirmed non-viable (depends on GSL via `rgsl`, experimental at 0.1.0); in-house implementation is the correct path.

---
*Research completed: 2026-04-02*
*Ready for roadmap: yes*
