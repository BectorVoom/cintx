# CubeCL Math Speed Optimization Plan

Status: proposed implementation plan
Date: 2026-08-21
Scope: `crates/cintx-cubecl/src/math` and the kernel/runtime seams required to make
that math fast
Compatibility target: libcint 6.1.3
Companion document: `docs/design/cubecl_speed_optimization_plan.md`

## 1. Outcome

The objective is to make the CubeCL implementation faster than libcint on a
representative production workload while preserving libcint-compatible results.
This is a throughput objective, not a claim that every isolated one-shell GPU call
must beat a mature scalar CPU library. The release claim is earned only when the
same integral corpus, inputs, outputs, warm-up policy, and accounting boundary are
used for both implementations.

The initial performance gate is:

- On the pinned release GPU, warm end-to-end cintx throughput, including packing,
  host-to-device transfer, kernel execution, device-to-host transfer, and output
  transformation, must beat single-threaded libcint by more than measurement noise
  over the full release corpus.
- The target margin is at least 1.20x geometric-mean throughput at batch sizes 256
  and 1024, with the lower bound of a 95% confidence interval greater than 1.00x.
- No required integral family may regress below 0.95x libcint at batch size 1024.
  A documented exception requires a follow-up milestone and may not be hidden by
  the aggregate score.
- High-order Rys evaluation must perform no nested host `CpuRuntime` launches, no
  per-primitive device allocation, and no intermediate host readback.
- After context and shape warm-up, the timed path must perform no heap or device
  allocation except an explicitly tested arena growth event.
- Oracle tolerances, operation order requirements, OOM behavior, and public API
  semantics may not be weakened to obtain a speedup.

Phase 0 records the variance of the actual benchmark machines. It may tighten the
numeric margin, but it may not redefine a loss as a win.

## 2. Why a math-specific plan is necessary

The repository-wide speed plan correctly prioritizes persistent contexts, batching,
resident input data, arenas, and tuned launch geometry. The math layer has additional
cliffs that those changes alone cannot remove:

1. Low-order Rys roots are large, branch-heavy generated formulas duplicated between
   host and device implementations.
2. Roots 6 through 12 cross a host/device boundary. Some paths allocate many buffers,
   launch one-thread CubeCL CPU kernels, read intermediate arrays back, launch an
   eigensolver, and then upload data again.
3. Production recurrence kernels duplicate scalar `pdata`, VRR, and HRR operations
   instead of consuming a shared, specialized math schedule.
4. STG/F12 roots remain host-side and allocate temporary vectors per primitive.
5. ECP quadrature, Bessel/K-Taylor evaluation, radial integration, and contractions
   are predominantly host-side. A 2,047-point quadrature grid is regenerated during
   evaluation even though it is invariant.
6. Current one-item kernels commonly launch `CubeCount(1), CubeDim(1)`. Reducing the
   arithmetic instruction count cannot compensate for allocation, launch, and
   synchronization overhead at that granularity.

The optimization unit must therefore be a batch of mathematical work items, not a
single invocation of an isolated formula.

## 3. Evidence reviewed

This plan is based on:

- The local CubeCL manual under
  `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl`, especially its
  chapters on batching, launch overhead, grid-stride kernels, vectorization, memory
  coalescing, shared memory, execution hierarchy, planes, compile-time
  specialization, fusion, compilation caching, preallocation, asynchronous staging,
  profiling, and autotuning.
- Current CubeCL documentation for `/tracel-ai/cubecl`, retrieved through Context7.
  It confirms compile-time specialization, `CubeCount`/`CubeDim` launch geometry,
  vectorized data, `AutotuneKey`/`TunableSet`, and profiling/autotune configuration.
- The current math, kernel, batch-pilot, runtime, oracle benchmark, Criterion, and
  `/tmp/cintx_artifacts` code and outputs.
- `docs/rust_crate_test_guideline.md` for the verification and test gates.

Examples in the local manual span more than one CubeCL API generation. Every runtime
or memory API must first be compiled as a small CubeCL 0.10.0 spike; this plan adopts
the optimization principles, not unverified example syntax.

## 4. Current-state diagnosis

### 4.1 Module map

| Area | Current role | Dominant cost or risk | Intended state |
|---|---|---|---|
| `boys.rs` | Host/device incomplete-gamma and accurate `F0` path | Dynamic order loops; two different error-function accuracy contracts | Fixed-order, accurate, register-resident device helpers selected at compile time |
| `rys.rs` | Generated roots 1-5, host duplicates, dispatcher | Large instruction bodies, region branches, duplicated source, JIT/code-size pressure | Generated common specification with specialized device and host-reference emitters |
| `rys_wheeler.rs` | Roots 6-12 via Jacobi, Schmidt, Laguerre, double-double variants | Per-call buffers, one-thread launches, host readbacks, nested eigensolve | One fused batch lane or bounded two-stage batched device pipeline |
| `eigh.rs` | Small symmetric tridiagonal eigensolver | `CpuRuntime`, `CubeDim(1)`, multiple buffers/readbacks | In-kernel fixed-order helper or one-item-per-lane batched kernel |
| `pdata.rs` | Primitive-pair scalar data | Host allocation/call repetition and duplicated device formulas | Inline scalar/register helper plus batched structure-of-arrays preparation |
| `obara_saika.rs` | Generic recurrence steps | Helpers do not express offset layouts used by production kernels | Generated/specialized offset-aware recurrence schedules |
| `stg.rs` | F12/STG roots and weights | Host-only table interpolation/DCT/Clenshaw work and temporary vectors | Batched device implementation with resident tables |
| `ecp_k_taylor.rs` | K-function tables and approximation | Large immutable tables and host-heavy use | One resident table set per device and specialized device evaluator |
| `radial_quadrature.rs` | Gauss-Chebyshev/Hermite utilities | Repeated invariant grid generation; large host loops | Precomputed resident grids and device adaptive reduction |
| `bessel.rs` | Modified spherical Bessel functions | Potentially variable convergence loops; weak production reachability | Retain as reference or specialize only after call-graph proof |
| `roots_jacobi_data.rs` | Large coefficient tables | Risk of repeated upload or embedding in every compiled kernel | Versioned, device-resident immutable table buffers |

### 4.2 High-order Rys cliff

The expensive boundary is structurally equivalent to:

```text
integral primitive
  -> host high-order root dispatcher
  -> allocate/upload Wheeler/Jacobi inputs
  -> launch one work unit
  -> read recurrence coefficients
  -> allocate/upload eigensolver inputs
  -> launch one work unit
  -> read eigenvalues/eigenvectors/status
  -> allocate/upload root transform inputs
  -> launch one work unit
  -> read roots and weights
  -> resume host recurrence or upload results to another kernel
```

The exact route varies by root count, but the architectural defect is common. Roots
6 and 7 also retain a host route because a prior direct device route changed the
floating-point environment enough to create roughly `1e-11` downstream Hessian
differences. That is evidence that operation order and FMA behavior are part of the
compatibility contract; it is not a reason to retain nested one-item launches.

### 4.3 Low-order roots and Boys

Roots 1-5 are already callable from device kernels, which is the correct boundary,
but the generated piecewise expressions are very large. Different root counts,
argument regions, and derivative families can inflate compilation time, instruction
cache use, register pressure, and divergence. The general Boys implementation has a
dynamic order and a lower-accuracy error-function approximation, while the newer
strict `boys_f0_f64` path uses the Cody coefficients needed by the batched ssss pilot.
These contracts must be unified without reducing numerical accuracy.

### 4.4 Recurrence and temporary data

The public recurrence helpers are small, but production kernels often reproduce the
formulas with custom base offsets. Root data, Cartesian intermediates, and contraction
scratch are consequently materialized in global buffers more often than necessary.
The correct layout changes with the parallel mapping:

- Lane-per-integral kernels need adjacent lanes to access adjacent item values.
- A cooperative plane or cube needs contiguous root/Cartesian tiles for its members.
- Fixed, small angular momenta benefit from compile-time schedules and unrolling.
- Large schedules must avoid full unrolling and excessive register lifetime.

### 4.5 STG/F12 and ECP

F12 evaluation calls host STG root generation per primitive. STG currently performs
table lookup, Clenshaw/DCT-style transforms, and normalization using temporary host
vectors.

ECP evaluation performs substantially more than planning on the host. It generates
the 2,047-point maximum Gauss-Chebyshev grid, evaluates radial functions and
K-Taylor/Bessel terms, performs adaptive loops, and contracts results in allocated
vectors. This conflicts with the project boundary that host CPU work is limited to
planning, validation, marshaling, and verification glue. ECP is therefore a device
migration project, not an instruction-level tuning task.

### 4.6 Existing measurements

Historical scalar CubeCL CPU raw-call measurements are tens to hundreds of times
slower than libcint for several small integrals because they measure tiny launches.
The current s/s batch pilot is directionally useful: the local Criterion result for
256 overlap items is about 0.43 ms versus about 5.26 ms for repeating the scalar path,
roughly a 12x batching improvement. It is not proof of a libcint win because the
fixture, backend identity, timing boundary, and libcint batch comparison are not
paired in that Criterion report.

The aggregate artifact reports are also not suitable for a release claim: some omit
adapter identity, device timestamps, memory telemetry, an autotune report, or an
oracle report, and some aggregate heterogeneous Criterion and JSONL samples. Phase 0
replaces this with paired evidence.

## 5. Performance and correctness contract

### 5.1 Fair comparison with libcint

Each published comparison must record:

- Git revision, Cargo lock hash, Rust version, CubeCL version, libcint 6.1.3 revision,
  enabled features, build profile, and compiler flags.
- CPU model, core affinity, libcint thread count, GPU adapter/driver/runtime, power
  mode, and whether clocks are fixed or observed.
- Integral family, derivative rank, angular momenta, primitive/contraction counts,
  root count, representation, batch size, and output bytes.
- Cold compile/autotune time separately from warm execution.
- Packing, H2D, kernel, D2H, transformation, and total end-to-end time separately.
- Median, p95, dispersion, sample count, and confidence interval for the paired
  speedup. Throughput and latency must both be derivable from raw samples.
- Identical screening, normalization, ordering, and requested outputs. Skipping work
  on only one side invalidates the comparison.

The primary baseline is single-threaded libcint because one GPU submission is a
parallel throughput operation. A secondary comparison against explicitly pinned
multi-threaded libcint is required for capacity planning and must not be conflated
with the primary gate.

### 5.2 Numerical contract

- Preserve the existing family-specific oracle tolerances and special-value behavior.
- Test branch boundaries using predecessor/exact/successor floating-point values.
- Preserve deterministic root/weight ordering and reduction order where required by
  the libcint oracle.
- Do not enable fast math, contraction/FMA, reduced precision, or reassociation as a
  blanket optimization. Each candidate is off by default until a complete downstream
  oracle corpus passes on every supported backend.
- Double-double paths remain double-double until evidence proves a simpler path meets
  the same downstream contract.
- Allocation failure returns a typed error and does not partially modify caller-visible
  output.

### 5.3 Host/device boundary

The timed production path may use the host for validation, bucketing, launch planning,
marshaling, and final API view construction. Root generation, eigensolving, recurrence,
radial quadrature, adaptive convergence decisions, and contraction are compute and
must execute in CubeCL kernels.

## 6. Target architecture

### 6.1 Separate reference math from executable device math

Refactor `math` into clear roles without changing the public crate API:

```text
math/
  host_ref/          deterministic oracle/reference implementations
  device/            small #[cube] arithmetic building blocks
  generated/         checked-in generated formulas and table manifests
  layout.rs          validated item/scratch/table layouts
  classify.rs        host shape classification only
  tests/             shared fixtures and parity/property tests
```

The exact file move can be incremental. The invariant is that production kernels do
not call a helper that launches another kernel or performs a readback. A `#[cube]`
function may call other `#[cube]` functions; host reference functions never appear in
the device call graph.

### 6.2 Math work classes

The host maps requests into a bounded `MathKernelClass` containing only structural
properties that change generated code or memory requirements:

```text
family
derivative_rank
angular_momentum_tuple
root_algorithm_and_count
representation
precision_mode
contraction_bucket
```

Continuous values such as exponents and coordinates are data, not specialization
keys. Primitive and batch counts use coarse buckets. This prevents compilation and
autotune-key explosion.

### 6.3 Resident immutable data

One device context owns versioned handles for:

- Rys/Jacobi coefficient tables.
- STG root/weight/interpolation tables.
- ECP K-Taylor tables.
- All Gauss-Chebyshev grids and weights, generated once at build time or context
  initialization. Lower adaptive levels reference strided subsets of the maximum
  grid instead of regenerating trigonometric values.
- Basis, ECP, environment, transform, and recurrence metadata already covered by the
  repository-wide residency plan.

The cache key includes a schema version and content checksum. Context construction
may upload these buffers; evaluation may not.

### 6.4 Reusable mutable storage

Introduce a fallible `MathScratchArena` with validated logical slices for:

- packed structure-of-arrays descriptors;
- roots, weights, recurrence coefficients, and eigensolver state;
- recurrence and contraction tiles;
- adaptive-active masks and compacted work queues;
- final device output.

The arena grows transactionally before launch. Scratch uses a slot-major layout
`scratch[slot * padded_items + item]` for lane-per-item kernels so neighboring lanes
coalesce. Cooperative variants use a separately validated tile layout; one physical
layout is not forced onto every execution hierarchy.

### 6.5 Kernel execution variants

Every class starts with a correct lane-per-item grid-stride implementation. Expensive
classes may add:

- one plane per integral, with plane-width-aligned cube dimensions;
- one cube per integral for large recurrence/contraction tiles;
- multiple small integrals per lane when launch and indexing dominate;
- fused root-to-recurrence kernels;
- a bounded two-stage roots/recurrence pipeline when fusion causes register spilling
  or occupancy collapse.

Adjacent lanes must read adjacent descriptors and write adjacent output values. Shared
memory is used only for demonstrated cross-unit reuse; its allocation and barrier cost
must be included in the candidate benchmark.

## 7. Work plan

### Phase 0: establish a trustworthy baseline

1. Add a math benchmark driver that emits one JSONL record per sample and runs CubeCL
   and libcint on the same generated input immediately adjacent in time.
2. Define a checked-in workload manifest covering the API manifest's production
   families, not only s shells or one primitive.
3. Benchmark batches `1, 8, 32, 128, 256, 1024, 4096`; record the actual item and
   output count after screening.
4. Add device-side timestamps where the backend supports them and retain host
   end-to-end wall time as the release metric.
5. Add counters for allocations, uploads, downloads, launches, bytes, compiled kernel
   variants, autotune trials, and host math fallbacks.
6. Split cold compile, cold cache, warm cache, and steady-state results.
7. Write all raw data and environment metadata to `/tmp/cintx_artifacts`.

Exit gate: two repeated baseline runs on the pinned machine agree within the recorded
variance, and every release workload has a paired libcint sample and oracle result.

### Phase 1: create optimization seams without changing results

1. Move host-only implementations behind an explicit `host_ref` namespace.
2. Make every production device primitive a non-launching `#[cube]` helper. Rename any
   helper that owns a launch so the boundary is visible.
3. Create a generated-source manifest for Rys/STG/ECP tables and formulas containing
   the upstream version, generator version, input checksum, and output checksum.
4. Replace duplicated host/device formula maintenance with one generator specification
   that emits a device form and an independently callable host-reference form.
5. Define validated descriptor and scratch layout types using checked size/offset
   arithmetic.
6. Instrument nested launches and host math fallbacks; tests must be able to assert
   they are zero for a selected production path.

Exit gate: oracle output is unchanged, generated files reproduce byte-for-byte, and
the benchmark change is statistically neutral within 5%.

### Phase 2: make tables, descriptors, and scratch resident

1. Extend the current resident cache from metadata-only entries to owned CubeCL buffer
   handles with explicit lifetime and device identity.
2. Upload immutable math tables once per context and expose read-only typed views.
3. Allocate descriptor, scratch, and output arenas per stream/context and reuse them.
4. Implement capacity growth as allocate-then-swap so OOM leaves the old arena and
   caller output intact.
5. Pack batch descriptors as structure-of-arrays. Separate frequently read scalar
   fields from rare metadata and avoid copying full `atm`/`bas`/`env` records.
6. Compile small CubeCL 0.10.0 probes for buffer overwrite, logical slicing, asynchronous
   submission, and completion semantics before adopting any manual example API.

Exit gate: a warm repeated batch records zero table uploads and zero allocations; one
descriptor upload and one final-output download are the default accounting boundary.

### Phase 3: specialize Boys and roots 1-5

1. Define one strict f64 Boys accuracy contract. Generalize the accurate Cody-based
   path to fixed requested orders and retain the current host implementation as a
   reference.
2. Make order and small loop bounds compile-time values. Unroll only short bounded
   loops verified not to cause spills or code-size regressions.
3. Specialize Rys by root count before launch; remove the dynamic root-count branch
   from device hot loops.
4. Keep roots and weights in scalar/local storage until their final consumer whenever
   live-range analysis permits.
5. Benchmark these bounded variants:
   - current scalar Horner/piecewise formula;
   - generated balanced expression tree where dependencies permit;
   - vector-across-items evaluation;
   - region-bucketed kernels when batch divergence is measurable;
   - fused root generation plus first recurrence layer.
6. Measure compile time, binary size, instructions, registers, local-memory spills,
   occupancy, and warm runtime. A runtime win that creates unacceptable cold-start or
   cache size is rejected or limited to common classes.
7. Preserve exact boundary behavior around every piecewise interval and large/small
   argument asymptote.

Exit gate: roots 1-5 and all downstream integral families pass oracle tests on CPU and
GPU backends, and the chosen variant improves the relevant steady-state class without
violating compile-cache budgets.

### Phase 4: replace the high-order Wheeler/eigensolver pipeline

This is the highest-priority math change.

1. Extract the fixed-size tridiagonal eigensolver body from `eigh.rs` into a
   non-launching device helper parameterized by compile-time order up to 13.
2. Implement a lane-per-item Wheeler pipeline that performs moment/coefficient
   construction, the eigensolve, sorting/normalization, and root/weight transformation
   without leaving the kernel.
3. Use fixed-size local arrays first. Add transposed arena scratch only when compiler
   output or profiling proves spilling/register pressure is worse.
4. Implement separate compile-time classes for Jacobi, Schmidt, Laguerre, and
   double-double arithmetic. Do not carry inactive algorithm branches in one kernel.
5. Add a two-stage batched alternative—coefficient construction followed by batched
   eigensolve/transform—only as an autotune candidate for classes where fusion reduces
   occupancy.
6. Reproduce the roots 6/7 floating-point-environment discrepancy with a focused test
   fixture. Audit contraction, FMA, subnormal, iteration termination, and eigenvalue
   ordering. Fix the device operation order rather than routing through `CpuRuntime`
   or loosening Hessian tolerances.
7. Bound every iterative loop and return per-item status. Reduce status once per batch;
   one failing item must not expose partial output as a successful result.
8. Delete the production nested-launch/readback route only after feature-matrix and
   oracle gates pass. Keep the old code temporarily behind a reference-only feature
   for differential diagnosis, never automatic timed fallback.

Exit gate: all roots 6-12 paths have zero nested launches, allocations, and intermediate
readbacks; downstream derivative and Hessian oracle cases pass; high-order batch
throughput is at least 2x the old CubeCL route before proceeding.

### Phase 5: fuse primitive data and recurrence schedules

1. Turn `pdata` into an inlinable register-return device primitive. Batch-precompute
   only fields reused across enough contractions to repay materialization.
2. Extend Obara-Saika helpers with validated base offsets and compile-time angular
   bounds so production kernels stop duplicating recurrence formulas.
3. Generate recurrence schedules by family, derivative rank, and angular-momentum
   tuple. Constant-fold zero terms and coefficient values without specializing on
   continuous input data.
4. Compare root-major, Cartesian-major, and item-major temporary layouts with actual
   coalescing and occupancy measurements.
5. Fuse roots -> primitive data -> VRR -> HRR -> contraction -> `gout` for small and
   common classes. For large classes, split at the boundary that minimizes total bytes
   and register lifetime.
6. Accumulate contractions on device. Apply Cartesian/spherical/spinor transforms in
   the same submission graph and download only caller-requested output.
7. Generalize the s/s grid-stride pilot to real angular momenta, primitives,
   contractions, and derivative families. Fixed `CubeDim(64)` and a fixed cube-count
   cap become candidates, not policy.

Exit gate: the common one-electron, two-electron, 2c2e, and 3c2e batches use one
submission graph, no host integral math, and no global intermediate that a measured
fused variant can avoid.

### Phase 6: move STG/F12 roots to CubeCL

1. Store STG tables in the immutable resident set with checked indices and checksums.
2. Port interpolation, Clenshaw/DCT evaluation, root refinement, and weight
   normalization to fixed-root-count device helpers.
3. Use lane-per-primitive first; add plane cooperation only if table evaluation or
   normalization has demonstrable reuse.
4. Fuse STG root production with F12 recurrence so roots and weights need not be
   written globally for common classes.
5. Add asymptotic and table-boundary fixtures plus random downstream F12 oracle cases.

Exit gate: F12 evaluation performs no host STG math or temporary vector allocation,
and all manifest-listed F12 operators pass the libcint oracle.

### Phase 7: migrate ECP radial math

1. Generate the nested Gauss-Chebyshev nodes/weights once, verify their relation
   between levels, and upload them with the ECP tables.
2. Port `ecprad`, type-1 radial parts, type-2 factors, K-Taylor/Bessel evaluation, and
   spherical contraction to CubeCL device helpers.
3. Start with one item per lane and a level-by-level active mask. For divergent
   convergence, benchmark queue compaction between levels against masked execution.
4. Tile the radial grid so values are generated, consumed, and reduced without a
   full `items x 2047` global temporary. Use plane reductions only where lane grouping
   and barriers are profitable.
5. Make angular momentum, ECP channel type, and quadrature-level ceiling compile-time
   where the variant count remains bounded.
6. Treat K-Taylor tables as shared read-only data, not per-kernel embedded constants.
   Benchmark table locality and a compact coefficient layout.
7. Accumulate type-1/type-2 contributions and perform final transforms on device.
8. Retain the host routines solely as references for differential tests.

Exit gate: host ECP work is limited to validation, bucketing, packing, and launch;
adaptive convergence and every numerical contraction execute on device; memory use is
bounded by the arena planner rather than quadrature grid size times the entire batch.

### Phase 8: bounded autotuning and compilation caching

Tune only choices with a measured tradeoff:

- cube dimensions `32/64/128/256` subject to backend limits and plane alignment;
- work items per lane;
- lane, plane, or cube ownership;
- vector width;
- fused versus two-stage high-order roots;
- fused versus split recurrence;
- local, shared, or transposed global scratch;
- masked versus compacted ECP adaptive work.

An `AutotuneKey` contains device fingerprint, backend, math class, batch bucket,
precision mode, and relevant layout version. It does not contain exact exponents,
coordinates, or every primitive count. Persisted tuning and compilation-cache entries
include CubeCL version, driver/compiler identity, kernel schema hash, and table schema
hash. Stale entries fail closed and retune.

Autotune trials use preallocated scratch and are excluded from warm release timing.
The untuned default must remain correct and reasonably performant.

Exit gate: a fresh context can reuse valid cached compilation/tuning data, cache
invalidation tests pass, and the number of variants stays within an explicit disk and
cold-start budget.

### Phase 9: family rollout and removal of obsolete paths

Roll out in measured order:

1. overlap/kinetic/nuclear and common low-order one-electron derivatives;
2. 2c2e and 3c2e;
3. two-electron low-root classes;
4. two-electron high-root gradients/Hessians;
5. F12/STG;
6. ECP;
7. less common transforms and unstable/raw compatibility entry points.

For each family, enable the new route only after its oracle, allocation, launch-count,
memory-pressure, feature-matrix, and performance gates pass. Once all supported
backends pass, remove the obsolete production route and keep only a minimal reference
implementation needed for tests.

## 8. File-level change map

| File or area | Planned change |
|---|---|
| `math/boys.rs` | Establish strict fixed-order f64 device helpers; separate approximate/reference APIs; add threshold fixtures |
| `math/rys.rs` | Generate host/device forms from one specification; specialize by root count and region; control code-size budget |
| `math/rys_wheeler.rs` | Replace buffer-owning launch helpers with fused/batched device algorithms; retain host reference only |
| `math/eigh.rs` | Extract compile-time fixed-order non-launching eigensolver; add batched wrapper and per-item status |
| `math/pdata.rs` | Add inlinable register helper and SoA batch representation; remove production formula duplication |
| `math/obara_saika.rs` | Add offset-aware device helpers and generated recurrence schedules |
| `math/stg.rs` | Port table interpolation/refinement/normalization to resident-table device kernels |
| `math/ecp_k_taylor.rs` | Version/checksum tables, upload once, and add specialized device evaluator |
| `math/radial_quadrature.rs` | Precompute invariant grids; implement tiled device adaptive quadrature |
| `math/bessel.rs` | Prove production reachability; retain as host reference or specialize bounded device paths |
| `math/roots_jacobi_data.rs` | Move data behind a resident-table manifest and prevent repeated kernel embedding/upload |
| `kernels/batch_pilot.rs` | Promote useful grid-stride machinery into the production batch executor; remove fixed launch policy |
| family kernels | Consume shared math helpers/schedules, fuse stages, and eliminate scalar host fallbacks |
| runtime/cache | Own actual buffer handles, scratch arenas, counters, compilation cache, and bounded autotune state |
| `cintx-oracle` benchmark | Replace unpaired scalar timing with paired manifest-driven libcint comparison |

Before deleting apparently unused Bessel or Gauss-Hermite code, rerun CodeGraph and
feature-matrix reachability checks; tests alone are not proof that a raw or optional
API does not use it.

## 9. Benchmark corpus

### 9.1 Math microbenchmarks

- Boys orders used by all supported derivative ranks at zero, subnormal, small,
  branch-boundary, moderate, and asymptotic arguments.
- Rys roots 1-12 with every region boundary, near-degenerate roots, random logarithmic
  arguments, and downstream root/weight consumption.
- Wheeler coefficient construction and eigensolve separately and fused.
- VRR/HRR schedules across representative s through maximum supported angular
  momenta, derivative ranks, and contraction depths.
- STG table interpolation/refinement across boundaries and asymptotic regimes.
- ECP grids at each adaptive level, K-Taylor regimes, type-1/type-2 radial paths, and
  convergence distributions.

Microbenchmarks diagnose causes; they do not establish the libcint win.

### 9.2 End-to-end release corpus

The checked-in manifest must include:

- every claimed API family and derivative rank;
- low, medium, and high angular momentum;
- single and multiple primitive/contraction cases;
- screened and unscreened batches;
- Cartesian, spherical, and spinor outputs where supported;
- root counts 1-12;
- ordinary molecular, diffuse, near-coincident-center, large-exponent-ratio, and
  numerically difficult cases;
- F12/STG and ECP workloads;
- batch sizes `1, 8, 32, 128, 256, 1024, 4096`.

Report per-family results and an unweighted family geometric mean so a large easy
batch cannot hide a slow or incorrect family. Also publish a usage-weighted aggregate
for practical capacity planning, with its weights checked into the manifest.

## 10. Verification plan

This section follows `docs/rust_crate_test_guideline.md`. Test code is created only
after reviewing that guideline again at implementation time.

### 10.1 Specification-to-test-to-gate map

| Requirement | Tests/evidence | Gate |
|---|---|---|
| libcint-compatible outputs | Manifest oracle examples plus randomized/proptest inputs and downstream derivative cases | PR for touched classes; full corpus nightly/release |
| Piecewise boundary parity | predecessor/exact/successor fixtures for Boys, Rys, STG, K-Taylor | PR |
| No nested high-order launch | launch/allocation/readback counters around roots 6-12 production batches | PR |
| Chunk/batch invariance | same ordered inputs at multiple chunk sizes and launch geometries | PR |
| Deterministic output ordering | repeated and permuted-batch metamorphic tests | PR |
| Safe layouts | proptest checked offsets/sizes/strides; Kani proofs for bounded index arithmetic where practical | PR plus nightly Kani |
| OOM stop contract | fault-injected arena growth and unchanged-output assertions | PR |
| Cache correctness | version/device/schema mismatch and concurrent-context tests | PR; loom if shared synchronization is introduced |
| Backend portability | CPU CubeCL plus each supported GPU backend oracle subset | PR CPU; GPU nightly/release |
| Feature completeness | `cargo hack` feature powerset/selected matrix and compiled manifest lock | CI/release |
| Performance win | Paired raw JSONL samples and statistical report versus libcint | Pinned-hardware release gate |
| No warm allocation/upload regression | counters and memory telemetry under repeated batch | Nightly/release |

### 10.2 Required commands and tiers

For touched code, the ordinary PR tier includes:

- `cargo fmt --check`
- `cargo check --workspace --all-targets`
- `cargo clippy --workspace --all-targets` with the repository lint policy
- `cargo test --workspace --locked`
- focused CPU CubeCL math/device parity tests
- relevant property tests
- the selected `cargo hack` feature matrix

Nightly or scheduled verification includes:

- full randomized libcint oracle comparison;
- `cargo mutants` on new classifiers, layouts, threshold logic, and recurrence helpers;
- `cargo llvm-cov` with a reviewed coverage report rather than a percentage-only gate;
- Miri for unsafe buffer/view/lifetime code;
- Kani for bounded offset, capacity, and transaction invariants where finite models are
  useful;
- loom only if cache/arena synchronization becomes shared concurrent state;
- all available GPU backends and memory-pressure tests;
- stable performance and compilation-cache smoke tests.

Fuzzing is appropriate for raw descriptor validation or generated manifest parsing if
those surfaces accept untrusted bytes. It is not added merely to exercise numeric
functions already better covered by structured property generators.

### 10.3 Numerical cases

At minimum, tests cover:

- `+0`, small positive values, subnormals, threshold neighbors, large finite values,
  infinities/NaNs where the public contract defines them;
- root counts 1-12 and every algorithm switch;
- repeated/near-repeated eigenvalues and maximum iteration behavior;
- extreme exponent ratios and near-coincident centers;
- all transform/derivative consumers, because root-level agreement alone is
  insufficient;
- batch permutation, duplication, chunking, and padding invariance;
- forced allocation failure before launch and device error after launch.

## 11. Profiling and acceptance workflow

For each optimization candidate:

1. Record the pre-change paired baseline and oracle hash.
2. Capture a CubeCL profile with launch count, dispatch geometry, device time, memory
   transfers, and synchronization points.
3. Inspect generated backend code where possible for unrolling, vectorization,
   register use, spills, barriers, and accidental scalarization.
4. Change one architectural variable or keep a labeled factorial experiment.
5. Re-run identical inputs, correctness checks, memory counters, cold compilation,
   and warm performance.
6. Keep the change only if the confidence interval, family distribution, and resource
   counters explain a real improvement.

Wall-clock timing around asynchronous submission must synchronize at the measurement
boundary. Submission time alone is never reported as kernel execution time.

## 12. Required artifacts

Release and major milestone runs write machine-readable files under
`/tmp/cintx_artifacts`:

- `cubecl_math_environment.json`
- `cubecl_math_workload_manifest.lock.json`
- `cubecl_math_baseline_samples.jsonl`
- `cubecl_math_profile.jsonl`
- `cubecl_math_autotune.json`
- `cubecl_math_compilation_cache.json`
- `cubecl_math_oracle_report.json`
- `cubecl_math_speed_vs_libcint.json`
- `cubecl_math_memory_report.json`
- `cubecl_math_generated_tables_manifest.json`
- `cubecl_math_unverified_matrix.json`

The speed report links every aggregate to raw sample identifiers. The unverified
matrix explicitly lists missing hardware, families, feature combinations, or oracle
coverage; absent evidence is never represented as a pass.

## 13. Safety and failure invariants

- Validate every output length, descriptor offset, table index, scratch capacity, and
  multiplication/addition used in allocation before an unchecked launch.
- Use `launch_unchecked` only behind a safe host proof that is unit-tested and, where
  useful, model-checked.
- No floating-point atomics are introduced for reductions that require deterministic
  libcint-compatible ordering.
- Kernel status is collected without publishing partial output as success.
- Arena growth is fallible and transactional. Never overwrite the caller's destination
  before all required work and transformations have completed successfully.
- Cache handles are tied to the originating device/client and schema; cross-device
  reuse is rejected.
- Autotune failure falls back to a correct bounded default, not to host numerical
  computation.

## 14. Suggested implementation sequence

Each item should be a reviewable commit or short stack with its own evidence:

1. Paired benchmark manifest, device identity, raw samples, and counters.
2. Host-reference/device namespace and generated-table manifest.
3. Resident math tables and reusable scratch arena.
4. Accurate fixed-order Boys helpers and low-root specialization.
5. Non-launching fixed-order eigensolver.
6. Fused/batched roots 6-12 with the roots 6/7 FP discrepancy resolved.
7. `pdata` and generated recurrence schedules.
8. Common-family fused batch kernels and tuned execution hierarchy.
9. STG/F12 device migration.
10. ECP grid/table residency and device radial pipeline.
11. Bounded autotune and persistent compilation-cache validation.
12. Full manifest rollout, obsolete fallback removal, and release benchmark.

Do not begin with broad expression-level rewrites of the generated Rys formulas. The
largest known costs are granularity, transfers, allocation, and host/device crossings;
those architectural changes also create the benchmark needed to judge arithmetic
rewrites correctly.

## 15. Rollback and diagnostics

During migration, each new family route has a development-only selector for new,
reference, and differential execution. Release builds default to the verified route.
A failed oracle or performance gate reverts one family/class without reverting the
resident context or benchmark infrastructure.

Diagnostic reference paths may remain slow and host-based behind test/dev features.
They must be visibly labeled, excluded from published timing, and unavailable as a
silent production fallback.

## 16. Definition of done

The math optimization project is complete only when:

- The compiled API manifest has no claimed family using host numerical integration,
  root generation, eigensolving, recurrence, or contraction in the CubeCL route.
- Warm evaluation uses resident immutable inputs and reusable arenas with bounded,
  measured transfers and submissions.
- Roots 1-12, Boys, recurrences, STG/F12, ECP, and all downstream consumers meet their
  libcint oracle contract on the supported feature/backend matrix.
- OOM and device failures are typed, deterministic, and cause no partial caller-visible
  writes.
- The pinned-hardware paired benchmark satisfies the performance gates in Section 1,
  with raw evidence and environment identity under `/tmp/cintx_artifacts`.
- Documentation distinguishes verified backends/results from unverified ones and
  records residual risks.

## 17. Verified facts and open hypotheses at plan time

Verified from the current source and local artifacts:

- One-item launch patterns and nested high-order root/eigensolver boundaries exist.
- STG/F12 and ECP contain significant host numerical work and per-call allocations.
- The batch pilot materially reduces repeated scalar-call overhead for its limited
  s/s fixture.
- Current aggregate artifacts are insufficient for a defensible libcint-win claim.
- The current CubeCL API supports the core concepts required here: compile-time
  specialization, launch geometry, vectorization, autotuning, and profiling controls.

Not yet verified and therefore Phase 0/implementation obligations:

- Which supported GPU, family mix, and batch size first crosses libcint.
- Whether fused high-order roots outperform a two-stage batch after register pressure
  and occupancy are accounted for.
- The best execution hierarchy and cube dimension per backend/class.
- Whether region bucketing low-order Rys formulas repays its extra classification and
  launch cost.
- Which CubeCL 0.10.0 buffer update, slicing, async, and persistent cache APIs are
  portable across all selected backends.
- Whether Bessel and Gauss-Hermite code has production reachability outside the paths
  observed during this audit.

These uncertainties are reasons for explicit experiments and gates, not reasons to
retain the known per-primitive launch architecture.
