# CubeCL Speed Optimization Plan

Status: proposed  
Scope: large-scale refactor permitted  
Primary compatibility target: libcint 6.1.3  
CubeCL target: pinned workspace version `0.10.0`

## 1. Purpose

This plan changes cintx from a correctness-first, one-shell-tuple-per-launch implementation into a batched CubeCL execution engine. The primary objective is warm throughput on real integral workloads without weakening result compatibility, typed failure behavior, memory limits, or backend portability.

The optimization order is deliberate:

1. Measure real work and make costs attributable.
2. Remove repeated setup, allocation, upload, launch, and readback costs.
3. Give chunks real, disjoint work ranges.
4. Expose enough parallel work to CubeCL to fill a device.
5. Move final-layout transforms to the device.
6. Tune launch geometry, vectorization, shared memory, and specialization only after the architecture is batch-capable.

Instruction-level changes made before steps 2–4 are unlikely to materially improve end-to-end performance.

## 2. Sources and constraints

The plan is based on:

- The local CubeCL manual at `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl`, especially the sections on profiling, launch overhead, lazy execution, memory preallocation, buffer slicing, grid-stride occupancy, memory coalescing, vectorization, comptime specialization, loop unrolling, kernel fusion, planes, shared memory, double buffering, compilation caching, and autotuning.
- Current CubeCL documentation for runtime-generic kernels, global profiling/autotune/compilation configuration, launch settings, and bounded memory-pool configuration.
- The current cintx execution path in `cintx-rs`, `cintx-compat`, `cintx-runtime`, and `cintx-cubecl`.
- The project design requirement that host CPU work remain planning, validation, marshaling, and oracle/test glue; integral computation stays on CubeCL.
- The project verification requirement of libcint 6.1.3 oracle compatibility, the compiled manifest lock, feature-matrix validation, and artifacts under `/tmp/cintx_artifacts`.

The pinned CubeCL `0.10.0` API is authoritative for implementation. Examples from CubeCL `main` or the manual must be compiled in a small spike before adoption when their exact API surface differs from `0.10.0`.

## 3. Current-state diagnosis

### 3.1 The dominant bottleneck is orchestration

The current core kernels are serial device programs:

- `one_electron_scalar_kernel`, `two_electron_scalar_kernel`, and the center-family kernels guard all work with `UNIT_POS == 0`.
- Their launchers use `CubeCount::Static(1, 1, 1)` and `CubeDim::new_1d(1)`.
- A typical 1e call uploads exponents, coefficients, coordinates, and charges; allocates G/Rys/output buffers; launches one unit; immediately calls `read_one_unchecked`; and converts the readback to a new `Vec`.
- A typical 2e call performs eight input uploads, four device allocations, one single-unit launch, one synchronous readback, and a host-side representation transform/scatter.

The implementation contains roughly 50 production occurrences of `UNIT_POS == 0`, roughly 60 `CubeDim::new_1d(1)` launches across kernels/math, hundreds of `client.create*` calls, and many immediate readbacks. These are architectural costs, not isolated slow loops.

### 3.2 Caches do not yet make data device-resident

- `BackendCache` resolves a live backend on every `query_workspace` and `execute` call.
- `CubeClExecutor::execute` reads backend choice again instead of using the `BackendIntent` already captured by the plan.
- A new `CubeClExecutor` is constructed for every safe-API evaluation and every raw evaluation.
- `DeviceResidentCache` stores only host `ResidentMetadata`; it stores no CubeCL handles.
- Its current basis hash covers structure but not atom coordinates, exponents, coefficients, ECP data, or other numeric content. That hash is not safe for a future device-data cache without being strengthened.
- `RawOptimizerHandle` currently stores only symbol/workspace hints. It performs no screening, preprocessing, plan caching, or device residency.

### 3.3 Chunking is not computational chunking

The safe and raw paths allocate a full output buffer for every chunk. Family launchers ignore `chunk.work_unit_start` and `chunk.work_unit_count`, recompute the complete shell-tuple result, and overwrite the accumulator with the last identical full result. More chunks therefore mean repeated compute, repeated allocation, and repeated transfer rather than bounded disjoint work.

This must be fixed before buffer pooling or asynchronous submission, otherwise those features only make redundant work cheaper.

### 3.4 Final transforms and staging remain on the host

Cartesian results are read to the host before spherical/spinor transformation and contraction-major scatter. Transform helpers allocate multiple temporary vectors. The public execution paths also allocate a full accumulator plus a full per-chunk staging buffer and copy between them.

### 3.5 Existing benchmark coverage cannot gate this refactor

`benches/micro_families.rs`, `macro_molecules.rs`, and `crossover_cpu_gpu.rs` execute synthetic trigonometric loops and modeled CPU/GPU times rather than cintx kernels. They cannot measure CubeCL launch count, JIT time, allocation, transfer, occupancy, or integral throughput.

The existing real `benchmark_speed` test records the current CubeCL CPU path at about 26–174 times slower than direct SIMD across the major families, with 2e around 1.05 ms per tested call. This is useful historical evidence, but the benchmark shape is dominated by synchronous single-tuple calls and is not a substitute for batched CPU/GPU measurements.

## 4. Performance contract

### 4.1 Correctness and safety gates

Every optimization must preserve:

- f64 oracle gates at `atol = 1e-12`, `rtol = 1e-12` for all release-blocking families.
- The existing explicit f32 policy and its separate oracle tolerance.
- Optimizer-on versus optimizer-off equivalence within the same oracle policy.
- Deterministic output layout and component order.
- No caller-visible partial output on allocation, validation, launch, or readback failure.
- Typed OOM and memory-limit failures.
- Safe API first, raw compatibility API second, C ABI third.
- Feature-matrix and compiled-manifest lock conformance.

Optimization must not rely on relaxed floating-point math, reassociation, unordered floating atomics, or an approximate screening threshold in the default strict mode.

### 4.2 Quantitative performance gates

Phase 0 will establish exact baselines. The following become release goals after calibration:

- Batched warm throughput: at least 10x the current CubeCL CPU geometric mean for batches of 256 or more shell tuples, without a regression in any individual base family larger than 5% from the preceding accepted phase.
- GPU throughput: faster than libcint for at least one contiguous range above the measured crossover for 1e, 2c2e, 3c1e, 3c2e, and 2e; record the crossover rather than assuming one universal batch size.
- Submission: one kernel launch per `(kernel class, chunk, stage)`, never one launch per shell tuple.
- Transfer: basis-invariant numeric data uploaded once per `(device, basis, precision)` residency; one batched final readback per evaluation unless a measured pipeline requires two.
- Allocation: zero device allocation and zero large host allocation in the warm inner submission loop after arena growth has stabilized.
- Chunking: every work item evaluated exactly once regardless of chunk count.
- Memory: peak device and host memory stay within the declared plan plus documented pool/cache overhead.
- Cold/warm split: JIT/autotune time is reported separately and never hidden in steady-state throughput.

Single-tuple raw-call latency remains measured and regression-gated, but it is not the main GPU success metric. The synchronous libcint-compatible call shape provides little work over which to amortize device submission.

## 5. Target architecture

```text
Safe batch API / raw optimizer-backed calls
                  |
        validation + request normalization
                  |
        BatchExecutionPlan + KernelClass buckets
                  |
       persistent CubeClContext per device
       /              |                 \
DeviceBasisCache  DeviceArenaPool   Kernel/Tune cache
       \              |                 /
       packed SoA batch descriptors + output offsets
                  |
      batched grid-stride CubeCL kernels
                  |
    fused or staged device-side final transform
                  |
          one final-layout readback
                  |
       transactional caller-visible commit
```

### 5.1 Persistent execution context

Add a long-lived `CubeClContext` owned by a safe API session and shared through the raw optimizer path. It contains:

- One resolved backend/client and immutable capability snapshot.
- A device-resident basis cache.
- A bounded device arena/workspace pool.
- A kernel specialization and compilation-cache namespace.
- An autotune cache keyed by device and kernel class.
- Queue/submission state and profiling counters.

Backend selection comes from `ExecutionOptions.backend_intent`, not a second environment-variable read during execution. Environment parsing may populate default options at an outer boundary, but query and execute use the same resolved context.

### 5.2 Real device-resident basis representation

Replace metadata-only residency with backend-specific handles for a packed representation:

- Atom coordinates as SoA `x[]`, `y[]`, `z[]`; charges/model data in separate narrow arrays where possible.
- Shell metadata arrays for atom index, angular momentum, `nprim`, `nctr`, kappa, exponent offset, coefficient offset, and AO offset.
- Contiguous exponent and row-major coefficient buffers.
- ECP and operator tables in separate optional buffers.
- Cartesian-to-spherical/spinor coefficient tables.
- Optional device-computed primitive-pair tables owned by the optimizer cache.

The cache key includes backend/device identity, precision, representation-dependent tables, feature/codegen version, and a content hash over all numeric basis data using stable byte/to-bits hashing. Cache growth is bounded with explicit byte accounting and LRU/lease-based eviction. In-use entries cannot be evicted.

### 5.3 Batch and specialization model

Introduce a `BatchExecutionPlan` containing disjoint work items. Bucket items by a finite `KernelClass`, initially:

```text
family/operator + representation + precision
+ angular-momentum tuple + nroots
+ ibase/kbase + contraction mode
+ derivative/component rank + transform kind
```

Each bucket packs compact SoA descriptors:

- Shell indices, output offsets, output lengths, scratch offsets.
- Per-item centers/origins or references into resident atom arrays.
- Flags that remain truly dynamic.

Comptime specialization is used for structural decisions that remove meaningful branches: operator kind, angular-momentum tuple, nroots, HRR branch, derivative rank, representation transform, and single/general contraction. Dynamic values remain dynamic to avoid unbounded JIT variants. Common sizes are bucketed; rare sizes use a generic fallback kernel.

### 5.4 Genuine chunk semantics

A chunk is a disjoint range of batch-item indices, not a fraction of a workspace estimate. The kernel receives the range and output-offset table and evaluates only those items. The plan computes:

- Resident bytes.
- Per-item and per-bucket scratch bytes.
- Chunk scratch/output bytes.
- Transfer bytes.
- Arena alignment and maximum allocation.

All chunks write into disjoint regions of an internal final-layout output buffer. The raw caller buffer is updated only after all chunks and the final readback succeed. This retains the no-partial-write guarantee while eliminating repeated full-block staging.

### 5.5 Parallel kernel hierarchy

Use a staged parallelization strategy so numerical ordering changes are controlled:

1. **Lane-per-tuple baseline.** `ABSOLUTE_POS` selects one shell tuple from a homogeneous bucket. Each unit preserves the current primitive/contraction order. A grid-stride loop lets a hardware-sized grid process an arbitrary batch. Scratch is transposed as `[local_scratch_index][batch_item]` so adjacent lanes access adjacent addresses.
2. **Vectorized lane-per-tuple.** For homogeneous buckets, test `Vector<F, N>` factors 1, 2, and 4 across independent tuples. Use an aligned main kernel and scalar tail. Do not vectorize across a floating reduction whose lane order would change.
3. **Plane-per-tuple.** For high-work 2e/3c2e/F12 buckets, assign one plane to a tuple. Use plane shuffles for broadcast and integer mask reductions. Floating accumulation remains in a fixed order until a dedicated oracle experiment proves a deterministic cooperative reduction.
4. **Cube-per-tuple/tile.** Use shared memory only where recurrence or transform data is reused enough to offset barriers and occupancy loss. Shared-memory size is comptime and part of tuning viability.

The lane-per-tuple kernel is the correctness reference for all cooperative variants.

### 5.6 Device-side final layout

Move Cartesian scatter, cart-to-spherical, and cart-to-spinor work to CubeCL. Cache transform coefficients as resident buffers. Compare two variants per class:

- Fused recurrence/contraction/transform/write when register and instruction pressure remain acceptable.
- A two-stage device pipeline with a reused arena slice when fusion reduces occupancy.

Both variants write the final component-leading, contraction-major, possibly complex-interleaved layout before D2H. Host transforms remain as oracle/reference implementations until every migrated family passes device-versus-host transform parity.

## 6. Implementation phases

### Phase 0 — Establish trustworthy measurements

Files: `benches/*`, `crates/cintx-oracle/tests/benchmark_speed.rs`, `xtask/src/bench_report.rs`, new profiling helpers in `cintx-cubecl`.

Work:

1. Replace synthetic benchmark bodies with real `query_workspace`/evaluate calls over fixed molecular and basis fixtures.
2. Add batch sizes `1, 8, 32, 128, 512, 2048` and family/representation/angular-momentum buckets.
3. Measure cold JIT, warm launch, packing, H2D, kernel, D2H, transform, final write, allocation count, launch count, and peak bytes separately.
4. Add CPU, WGPU, and ROCm profiles where hardware is available. CUDA/Metal remain explicitly unverified until runners exist.
5. Enable CubeCL profiling/compilation/autotune logging through a checked-in example `cubecl.toml`; add an optional `profile-tracy` development feature if supported by pinned `0.10.0`.
6. Add backend timestamp-query support when the capability snapshot permits it; otherwise report wall time and mark device time unavailable.
7. Record device name, capability fingerprint, driver/backend, CubeCL version, Rust version, feature set, git revision, warmup count, sample count, p50/p95, and oracle status.
8. Write JSON/JSONL artifacts under `/tmp/cintx_artifacts`, including a machine-readable baseline and human-readable summary.

Exit gate:

- Benchmarks execute real cintx code and can distinguish JIT, allocation, transfer, kernel, and transform time.
- Repeated warm runs have an agreed noise envelope.
- No performance claim is accepted from the old synthetic benchmark rows.

### Phase 1 — Persistent context, backend resolution, and arenas

Files: `crates/cintx-cubecl/src/executor.rs`, `backend/*`, `resident_cache.rs` (replace/rename), new `context.rs`, `arena.rs`, `device_basis.rs`; safe API session wiring; compat optimizer wiring.

Work:

1. Construct `CubeClContext` from the query-time `BackendIntent` and capability token.
2. Resolve/bootstrap the client once and store it for query plus execution.
3. Add a safe API `Session`/`Engine` that owns an `Arc<CubeClContext>` and supports repeated and batched evaluation without exposing backend-specific types.
4. Make raw optimizer handles own or reference the prepared typed basis, packed metadata, context lease, and plan cache. Preserve the no-optimizer path as a correct but less amortized path.
5. Implement a fallible, size-classed host staging pool and CubeCL device arena with high-water reuse.
6. Use a small number of physical allocations with aligned logical slices for scratch/output regions.
7. Bound CubeCL pool growth using supported `MemoryConfiguration`/pool options; map pool exhaustion to typed cintx errors.
8. Add cache/arena byte counters, hit/miss counters, allocation counters, and tracing spans.

Exit gate:

- Repeated identical calls do not re-bootstrap the backend.
- Warm repeated calls do not allocate new device buffers after pool stabilization.
- Cache identity tests prove that different numeric basis contents cannot alias.
- Concurrent context use has a documented queue/lease policy and passes concurrency tests.

### Phase 2 — Batch API, packed SoA input, and correct chunking

Files: `cintx-runtime/src/planner.rs`, `workspace.rs`, `dispatch.rs`, new batch-plan module; `cintx-rs` batch/into APIs; `cintx-compat` optimizer path.

Work:

1. Add safe Rust APIs for `evaluate_batch` and `evaluate_batch_into` over many shell tuples.
2. Define `KernelClass`, bucket construction, output-offset tables, and per-item status/not0 fields.
3. Pack descriptors as SoA using fallible allocations; parallelize only marshaling with Rayon when measurement justifies it.
4. Redefine `ChunkInfo` in terms of `item_start/item_count` plus byte regions. Remove the current proportional-output approximation.
5. Make workspace estimation account for resident, arena, packed descriptor, scratch, internal output, and transfer bytes separately.
6. Execute each item exactly once and merge disjoint output ranges.
7. Keep a transactional internal output for raw APIs, committing to the caller only on full success.
8. Make metrics report actual launches, actual H2D/D2H, cache bytes, allocation count, and items completed; stop double-counting transfer bytes.

Exit gate:

- Batch output equals concatenated scalar output for every supported arity/representation.
- Results and `not0` are invariant across forced chunk sizes.
- A counted test proves every item executes exactly once.
- OOM/failure injection proves the raw output remains unchanged.

### Phase 3 — Batched pilot kernels

Files: new shared batch-kernel utilities plus `one_electron.rs` and `two_electron.rs` pilot paths.

Pilot set:

- 1e overlap/kinetic Cartesian: simple recurrence and launch-overhead proof.
- 2e Cartesian scalar with `nroots <= 5`: heavy recurrence and scratch-layout proof.

Work:

1. Implement lane-per-tuple grid-stride kernels using resident basis buffers and packed item descriptors.
2. Select a hardware-sized grid; begin with plane-aligned `CubeDim` candidates 64/128/256 rather than a fixed value.
3. Store scratch as local-index-major/batch-item-minor for coalesced cross-lane accesses.
4. Preserve each tuple's current primitive and contraction accumulation order.
5. Specialize the same structural arguments already proven useful (`op_kind`, l tuple, nroots, ibase/kbase, contraction mode).
6. Use `launch_unchecked` only after a host validator proves every offset and maximum index. Keep grid-stride range guards.
7. Submit all bucket kernels before a single readback.
8. Retain the scalar launcher behind a test-only/reference route until parity and performance gates pass.

Exit gate:

- Exact or oracle-tolerance agreement with the old scalar device path, direct SIMD, and libcint for the pilot envelope.
- One launch covers the complete bucket/chunk.
- Warm batch-256 throughput improves by at least 5x in the pilot before migrating more families.

### Phase 4 — Migrate base scalar families

Migration order:

1. 2c2e and 3c1e.
2. Remaining 1e scalar/nuclear/rinv/moment variants.
3. 3c2e scalar.
4. 2e derivative and Hessian variants within their validated Rys envelope.

Work:

- Extract duplicated recurrence/shape/packing logic into `#[cube]` helpers and host-side plan descriptors.
- Replace per-backend five-arm call duplication with runtime-generic launch adapters owned by the context.
- Eliminate per-launch `to_vec` copies of shell exponents/coefficients.
- Eliminate immediate output readbacks and host `Vec` returns from `run_*_device`.
- Convert exact-zero coefficient skipping and uniform branches to comptime/predicated forms where safe.
- Unroll only small comptime loops. Compare code size, register pressure, JIT time, and kernel time before accepting full unrolling.

Exit gate:

- All base Cartesian families use batched resident-data paths.
- No base-family hot launcher calls `client.create*`, `client.empty`, or `client.read*` per shell tuple.
- Base-family oracle and feature gates pass.

### Phase 5 — Device transforms, fusion, and readback reduction

Files: `transform/c2s.rs`, `transform/c2spinor.rs`, family launchers, device coefficient cache.

Work:

1. Port spherical transforms as tiled CubeCL kernels using resident coefficient tables.
2. Port spinor/complex transforms with paired/interleaved vector loads where both real and imaginary values are consumed together.
3. Write final contraction-major/component-leading offsets directly from device kernels.
4. Compare fused versus staged device transforms for each kernel class.
5. Reuse one arena across Cartesian intermediates and final outputs; slice it rather than allocate per logical tensor.
6. Batch all required readback handles in one `client.read` call if more than one result/status buffer remains.
7. Move `not0` calculation to a device status output. Use integer mask reductions; do not horizontally reduce f64 values or use nondeterministic floating atomics.

Exit gate:

- No production spherical/spinor path transforms a full result on the host.
- D2H contains final-layout values only.
- Device transform output matches the retained host reference across l/kappa/contraction/property cases.

### Phase 6 — Autotuned launch geometry and cooperative kernels

Files: new `tuning.rs`; batch kernels; context configuration/cache.

Tune by device fingerprint and coarse workload key, not exact dimensions. Candidate dimensions include:

- Kernel mode: lane-per-tuple, vectorized lane-per-tuple, plane-per-tuple, cube-per-tuple.
- `CubeDim`: viable plane-aligned values, usually 64/128/256 subject to backend limits.
- Grid occupancy multiplier/cube count.
- Vectorization factor 1/2/4.
- Scratch tile and shared-memory size.
- Fused versus staged transform.
- Optional double-buffered shared-memory staging for measured memory-latency-bound kernels.

Work:

1. Query target properties/capabilities and reject candidates that exceed workgroup, shared-memory, buffer, or feature limits before compilation.
2. Use persistent autotune caching with a schema/codegen version in the key.
3. Bound cold-start work with priority groups and a safe default. Provide `off`, `balanced`, and `extensive` policies.
4. Prewarm the finite common specialization set in benchmarks/release packaging where appropriate.
5. Use CubeCL compilation logs to detect specialization explosion and cache misses.
6. Add shared-memory tiling only where the profile proves reuse. Place `sync_cube` barriers outside divergent paths.
7. Add plane operations only on backends that advertise them; use `PLANE_DIM`, never a hard-coded warp size.
8. Treat async/double buffering as an experimental candidate until pinned CubeCL codegen and profiler evidence show real overlap.

Exit gate:

- Cached tuning adds negligible warm overhead.
- Tuning never changes results.
- The selected variant beats or matches the safe default beyond the configured noise threshold.
- Cache cardinality and disk/device memory have hard bounds.

### Phase 7 — Real optimizer and device precomputation

Files: `cintx-compat/src/optimizer.rs`, safe optimizer API, device-basis/plan caches, optional preprocessing kernels.

Work:

1. Cache raw-to-typed basis conversion and validated layout in `RawOptimizerHandle`.
2. Build shell-pair/quartet metadata, sorted index tables, and exact-zero coefficient masks once.
3. Compute floating primitive-pair data on CubeCL into resident buffers so host CPU remains control-plane only.
4. Reuse pair tables across 2e/3c2e/F12 calls.
5. Add mathematically conservative screening only after an explicit accumulated-error bound is defined. Default strict mode must not silently introduce approximate screening.
6. Make optimizer-on/off parity a release gate across random and adversarial cases.

Exit gate:

- Optimizer handles demonstrably reduce packing/upload/compute work.
- Optimizer state invalidates on any basis/operator/environment input that affects results.
- No approximate threshold is active in strict mode without a documented proof and oracle gate.

### Phase 8 — Optional and unstable families

Migration order after the base path is stable:

1. ECP.
2. F12/STG/YP.
3. 4c1e and high-root paths.
4. GIAO/complex and relativistic sigma families not completed earlier.
5. Unstable-source families.

Each family must use the shared context, batch plan, arenas, offsets, metrics, and final-layout pipeline. Family-specific monolithic exceptions require benchmark evidence and must remain visible in the plan/metrics; they may not silently fall back to repeated full work.

### Phase 9 — Release hardening and cleanup

Work:

- Delete superseded per-tuple launchers only after their reference tests have equivalent independent oracle coverage.
- Enforce lint checks that reject `CubeDim::new_1d(1)`, immediate `read_one*`, or per-tuple `client.create*` in production family launchers unless allowlisted with rationale.
- Document cache lifetime, memory limits, tuning policy, cold-start behavior, and backend verification status.
- Calibrate performance thresholds from stable dedicated runners and switch benchmark reports from calibration to enforce mode.
- Produce final speed, memory, transfer, crossover, and parity artifacts under `/tmp/cintx_artifacts`.

## 7. Verification strategy

### PR gates

- `cargo test --locked` for affected crates and oracle smoke sets.
- Batch-versus-scalar, chunk-invariance, cache-identity, no-partial-write, and transform-reference tests.
- `cargo-hack` feature matrix for CPU plus compile-only backend combinations.
- Doctests and compile-fail tests for new safe batch/session APIs.
- Formatting, clippy, manifest audit, and artifact-schema validation.
- A short CPU performance smoke check that detects catastrophic launch/allocation regressions but does not enforce noisy microsecond thresholds.

### Nightly gates

- Full libcint oracle matrix for base, optional, and unstable profiles.
- `proptest` over shell tuples, batch partitions, output offsets, and forced memory limits.
- `cargo-mutants` on planner/chunk/output-contract logic with reviewed, expiring waivers.
- `cargo-llvm-cov` as a gap-reporting tool, not proof of correctness.
- Miri for unsafe staging casts, raw views, and `ArrayArg::from_raw_parts` host contracts where supported.
- Loom for context/cache/arena concurrency and lease lifetimes.
- Fuzzing for raw `atm/bas/env/dims/shls` parsing and offset calculations.
- Kani or equivalent bounded verification for chunk coverage/disjointness and output-offset arithmetic where practical.
- Dedicated warm/cold CPU and available GPU benchmark suites.

### Release gates

- Full compiled-manifest and feature-matrix audit.
- Full oracle parity at the project tolerance.
- Optimizer-on/off and scalar/batch/chunk equivalence.
- Multi-device CubeCL consistency for verified backends.
- Enforced throughput, memory, transfer, allocation, and crossover thresholds on pinned runners.
- No unresolved unverified claim: CUDA/Metal or any unavailable device profile is reported as unverified, not passed.
- Complete artifacts in `/tmp/cintx_artifacts` with commands, versions, device metadata, thresholds, and residual risks.

## 8. Instrumentation and artifact schema

Extend `ExecutionStats` rather than inferring work from buffer lengths. Record at least:

- `items_planned`, `items_executed`, `bucket_count`, `chunk_count`.
- `kernel_launch_count`, `readback_count`.
- `jit_compile_ns`, `autotune_ns`, `pack_ns`, `h2d_ns`, `kernel_ns`, `d2h_ns`, `transform_ns`, `final_write_ns`.
- `h2d_bytes`, `d2h_bytes`, `resident_bytes`, `arena_peak_bytes`, `host_peak_bytes`.
- `device_allocations`, `host_allocations`, cache hits/misses/evictions.
- Kernel class, selected variant, cube count/dim, vectorization, shared-memory bytes.
- Backend/device/capability fingerprint and whether timing is wall-clock or device timestamp.

Required optimization artifacts:

- `cintx_cubecl_baseline.json`
- `cintx_cubecl_profile.jsonl`
- `cintx_cubecl_autotune.json`
- `cintx_cubecl_speed_report.json`
- `cintx_cubecl_memory_report.json`
- `cintx_cubecl_oracle_summary.json`
- `cintx_cubecl_unverified_matrix.json`

## 9. Risks and controls

| Risk | Control |
|---|---|
| Floating summation order changes parity | Lane-per-tuple reference keeps current order; cooperative reductions land only behind oracle comparison. |
| Specialization/JIT explosion | Finite `KernelClass` buckets, generic rare fallback, cache-cardinality metrics, schema versioning. |
| Shared-memory/register pressure lowers occupancy | Autotune with viability filtering; retain non-shared baseline; profile after every change. |
| Device cache returns stale basis data | Content-complete stable hash plus device/precision/codegen key and invalidation tests. |
| Arena or CubeCL pool grows without bound | Explicit byte cap, LRU/leases, typed pool-exhaustion mapping, peak-byte gates. |
| Lazy execution hides a failure until readback | Keep transactional internal output and map deferred errors before caller commit. |
| Raw API cannot amortize single synchronous calls | Make optimizer/context persistence effective; report single-call latency separately from batch throughput. |
| Backend feature/API differences | Runtime-generic core, capability-gated variants, compile matrix, device-specific verification labels. |
| Host transforms become an accidental second compute backend | Retain them only as reference/oracle code after device paths land; production path stays CubeCL. |
| Optimization work targets synthetic results | Phase 0 replaces synthetic benchmarks and is a hard dependency for all speed acceptance. |

## 10. Recommended first implementation slice

The first code change should not touch every family. Implement one vertical slice:

1. Real benchmark and profiling artifacts for 1e overlap/kinetic and 2e Cartesian.
2. Persistent CPU/WGPU-or-ROCm context and device arena.
3. Device-resident atom/shell/exponent/coefficient buffers.
4. Batch planner with correct disjoint chunks.
5. Lane-per-tuple batched kernels for the two pilot families.
6. One final-layout readback and transactional commit.
7. Scalar/batch/chunk/libcint parity plus batch-size throughput curves.

Only after that slice meets the Phase 3 exit gate should the refactor fan out across the remaining 40,000+ lines of family kernels. This produces an early proof of both the simple and recurrence-heavy execution shapes while keeping rollback and numerical diagnosis tractable.
