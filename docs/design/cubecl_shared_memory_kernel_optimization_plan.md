# CubeCL Shared-Memory Kernel Optimization Plan

Status: proposed

Scope: every CubeCL launch kernel in the current `cintx-cubecl` working tree

Primary compatibility target: libcint 6.1.3

CubeCL target: pinned workspace version `0.10.0`

Primary performance objective: beat optimized, single-threaded libcint on a representative warm batched workload without weakening correctness, OOM behavior, or backend portability

## 1. Outcome

This plan turns the current mostly serial CubeCL integral kernels into batched,
cooperative kernels that use shared memory when it produces measured reuse. It is a
specialized extension of:

- [`cubecl_speed_optimization_plan.md`](./cubecl_speed_optimization_plan.md), which
  establishes the persistent-context and batch architecture;
- [`cubecl_math_speed_optimization_plan.md`](./cubecl_math_speed_optimization_plan.md),
  which covers the mathematical pipeline and high-order Rys work; and
- [`cintx_detailed_design.md`](./cintx_detailed_design.md), which defines the public
  compatibility, OOM, verification, and artifact contracts.

The goal is speed, not maximum shared-memory allocation. Every launch kernel is
assessed in this plan, but a kernel keeps a no-shared implementation when it has no
cross-unit reuse or when barriers and occupancy loss make shared memory slower. The
autotuner may select `NoShared` as the winning production variant. Forcing shared
memory into streaming probes or independent lane-per-item kernels would work against
the requested libcint speed win.

The critical sequencing rule is:

> First expose cooperative work and batch enough shell tuples to occupy the device;
> then use shared memory to broadcast recurrence data, tile contractions, and reduce
> repeated global traffic.

Adding `SharedMemory` to the current unit-0 kernels without changing work ownership
cannot improve them because no second unit consumes the staged data.

## 2. Evidence reviewed

The plan is grounded in the following current evidence:

- CodeGraph exploration of all kernel definitions, launch wrappers, executor paths,
  batch planning, and benchmark/oracle paths in this repository.
- The current on-disk source under `crates/cintx-cubecl/src` at plan time.
- Pinned CubeCL `0.10.0` source in the Cargo registry, not only examples from CubeCL
  `main`.
- Current CubeCL documentation retrieved through Context7 for launch settings,
  benchmarking, compile-time specialization, and autotuning.
- The local CubeCL manuals under
  `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl`, especially shared
  memory, synchronization, grid-stride occupancy, planes, coalescing, and double
  buffering.
- The current Criterion suites, the three-way oracle benchmark, benchmark report
  tooling, benchmark thresholds, oracle tests, and Rust test governance document.

Pinned CubeCL `0.10.0` facts that implementation may rely on:

- `SharedMemory::<T>::new(size)` creates a shared array and requires `size` to be
  available at compile time (`#[comptime]`).
- `SharedMemory::<T>::new_aligned(size, alignment)` is available when an explicit
  layout alignment is required.
- `sync_cube()` provides the portable cube-wide shared-memory visibility and execution
  barrier.
- `client.properties().hardware.max_shared_memory_size` exposes the runtime's maximum
  shared-memory bytes per cube.
- A launch that exceeds the limit fails with a shared-memory resource error.
- `SharedMemory::free` exists, but it is unsafe and requires uniform control flow and
  no live references. It is an optional late optimization, not a Phase 1 dependency.

The Context7 documentation describes runtime autotuning and cached selections, while
the pinned source confirms the exact shared-memory and barrier APIs used by this plan.
Examples from a newer CubeCL branch must still pass a `0.10.0` compile/run spike before
being admitted.

## 3. Audited kernel inventory and present bottleneck

### 3.1 Inventory

The current tree contains 54 top-level `#[cube(launch...)]` definitions:

| Class | Count | Notes |
|---|---:|---|
| Integral-family kernels | 46 | Three batch pilots and 43 family kernels |
| Production math kernels | 7 | Six Rys/Wheeler kernels and one eigensolver kernel |
| Diagnostic kernel | 1 | FMA fidelity probe |
| Additional nested test kernel | 1 | Boys F0 device sweep; not part of the top-level count |

All 54 top-level definitions and the nested Boys test kernel are assigned a disposition
in Section 8.

### 3.2 Current execution shape

The central performance problem is verified, not hypothetical:

- No current `cintx-cubecl` source kernel uses `SharedMemory`, `sync_cube`, or another
  cube-wide shared-memory barrier.
- Forty-three integral kernels contain `if UNIT_POS == 0` and perform all recurrence,
  contraction, and output work on one unit.
- Those guarded kernels are typically launched as one 256-unit plane-aligned cube. The
  remaining 255 units wait while unit 0 performs the complete nested loop.
- The three batch pilot kernels are the only integral launch kernels that already map
  work across `ABSOLUTE_POS` or plane lanes.
- Production high-order Rys/eigensolver launchers use one-unit CubeCL CPU launches and
  several global scratch buffers/readbacks.
- The single-tuple benchmark history reports CubeCL CPU latencies around 21-27 us for
  the small base families and about 1,052 us for the tested 2e call, versus roughly
  0.18-4.1 us for libcint. These numbers are historical evidence only: no current
  performance baseline artifacts exist in `/tmp/cintx_artifacts`, so Phase 0 must
  remeasure the current working tree.

Shared memory is therefore part of a work-decomposition refactor, not a local storage
substitution.

### 3.3 Compute paths outside the launch inventory

Several registered families or operator arms remain partly or wholly host-routed,
including portions of 2e derivatives/Hessians/GIAO/spinor handling, F12 recurrence,
ECP radial work, and representation transforms. Shared memory cannot optimize host
code. Their migration belongs to the repository-wide speed/math plans and must precede
or accompany the relevant rows in Section 8.

No release note may say "all kernels use shared memory" when a family still performs
its dominant recurrence or transform on the host. Report the actual device stages and
their timing shares.

## 4. Performance, correctness, and safety contract

### 4.1 What "faster than libcint" means

The primary release comparison is a paired, warm, end-to-end throughput comparison:

1. Use identical molecules, basis data, operator symbols, representations, shell
   tuples, output layouts, and screening policy.
2. Reuse a prepared cintx session/device context and the corresponding libcint
   optimizer object. Exclude one-time JIT/context/optimizer construction from both warm
   tracks and report it separately.
3. Time cintx packing, submission, required transfers, device work, readback, and final
   caller-visible output commit. A kernel-only time is diagnostic and cannot establish
   the public speed claim.
4. Time libcint across the same ordered batch of shell tuples with one host thread for
   the primary baseline. Record a multi-threaded libcint secondary baseline separately.
5. Interleave or randomize paired samples to reduce thermal and frequency drift. Save
   every raw sample, not only Criterion estimates.
6. Publish results per backend. A ROCm win cannot be reported as a WGPU, CUDA, Metal,
   or CubeCL CPU win.

Release target after Phase 0 calibration:

- At batch sizes 256 and 1024, the designated production GPU achieves at least 1.20x
  geometric-mean end-to-end throughput over single-threaded libcint across the required
  stable corpus.
- At batch size 1024, no required stable family is below 0.95x libcint.
- Each base family (`1e`, `2c2e`, `3c1e`, `3c2e`, and `2e`) has at least one contiguous
  measured batch-size range above 1.00x libcint; its crossover is recorded explicitly.
- A shared-memory variant must be at least 1.05x faster than its no-shared control in
  device time and must not regress end-to-end p50 or p95 by more than 2%. Otherwise the
  no-shared variant remains selected.
- Optional and unstable profiles publish their own per-family ratios. They cannot be
  hidden inside a favorable aggregate.

These are gates for making a speed claim, not promises that measurement has already
established.

### 4.2 Numerical contract

The default strict path preserves the project-wide f64 oracle policy:

- `atol = 1e-12` and `rtol = 1e-12` for release-blocking libcint comparisons.
- Existing f32 policy and tolerances remain separate.
- Output shape, component order, contraction order, Cartesian ordering, spherical and
  spinor layout, complex interleaving, and `not0` behavior remain unchanged.
- Shared memory may broadcast values and eliminate duplicate loads, but the first
  production cooperative variant assigns one output element to one unit and accumulates
  primitive/root contributions in the same lexical order as the scalar reference.
- Do not use unordered floating atomics for integral accumulation.
- Do not use a cross-lane floating reduction in strict mode unless a dedicated oracle
  experiment proves it across the entire affected envelope. Existing plane reductions
  remain confined to already-gated pilots until that proof exists.
- No relaxed math, reassociation, fast-math flags, or approximate screening is admitted
  to the strict path.

The old scalar device implementation remains a test/reference variant until its
replacement passes scalar-versus-shared, chunk-invariance, backend-consistency, and
libcint oracle gates.

### 4.3 Failure and memory contract

- Query/planning computes shared bytes before launch.
- If no shared variant fits the runtime limit, select a smaller tile or `NoShared`;
  never launch an oversized kernel and hope for backend-dependent behavior.
- Shared-memory choice is included in the workspace/specialization plan so query and
  execute cannot disagree.
- Device resource failure maps to a typed public error without partial caller output.
- Internal output remains transactional until all chunks and readbacks succeed.
- Cache and autotune growth is bounded and byte-accounted.
- Warm shared-memory kernels must not introduce new host or device heap allocations in
  the submission loop after arenas stabilize.

## 5. Shared-memory design rules

### 5.1 Use shared memory only for measured reuse

Valid uses in cintx are:

- broadcast of shell descriptors, centers, exponents, coefficients, roots, weights, or
  recurrence values to several units in one cube;
- recurrence scratch that is produced once and consumed by many output-element units;
- tiled matrix-like ECP and Cartesian/spherical/spinor transforms;
- deterministic staging/reordering that converts scattered global access into
  coalesced loads/stores;
- a bounded per-plane or per-cube partial reduction when its numerical order is
  explicitly accepted; and
- double-buffered primitive/root tiles after the synchronous single-buffer version wins.

Invalid default uses are:

- copying data into shared memory for the same unit that would have read it once;
- staging purely streaming arrays with no reuse;
- allocating the complete worst-case tensor when a tile would preserve occupancy;
- using shared memory as a substitute for a persistent device-resident basis cache; or
- adding barriers inside divergent branches.

### 5.2 Portable cooperative skeleton

Each shared variant follows one uniform skeleton:

```rust
#[cube(launch_unchecked)]
fn cooperative_kernel<F: Float + CubeElement>(
    // resident inputs and batch descriptors
    #[comptime] tile_elems: u32,
) {
    let lane = UNIT_POS as usize;
    let active = /* range test; do not return before a barrier */;
    let mut tile = SharedMemory::<F>::new(tile_elems as usize);

    // All lanes participate; inactive lanes write neutral values.
    cooperative_load_or_zero(&mut tile, lane, active);
    sync_cube();

    // Lanes own disjoint output elements and retain reference reduction order.
    consume_tile_for_owned_outputs(&tile, lane, active);
    sync_cube(); // required before the next producer overwrites tile
}
```

The exact implementation uses compile-time constants accepted by CubeCL `0.10.0`;
the pseudocode illustrates control flow, not a drop-in signature.

### 5.3 Capability-aware layouts

Add a backend-neutral `SharedLayout` to the specialization plan:

```text
SharedLayout
  variant
  cube_dim
  descriptor_elems
  root_elems
  recurrence_tile_elems
  transform_tile_elems
  partial_elems
  buffer_count
  alignment
  total_bytes
```

For every candidate:

```text
total_bytes = aligned_sum(region_elements * element_bytes)
total_bytes <= client.properties().hardware.max_shared_memory_size
total_bytes <= configured_portable_cap_for_backend_profile
```

The planner also applies an occupancy reserve. It should initially prefer layouts that
consume no more than half the per-cube limit, leaving room for at least two resident
cubes when registers and backend limits permit. This is a heuristic, not a fabricated
occupancy measurement; backend profiling decides the final variant.

Static capacity classes avoid unbounded compilation:

- cube dimensions: 32, 64, 128, and 256, filtered by plane alignment and device limits;
- descriptor/root tile classes: small, medium, and large based on audited maximum
  indices;
- one or two recurrence buffers;
- transform tiles chosen from a bounded candidate set; and
- a generic no-shared fallback.

`SharedMemory::new_aligned` is used only where vectorized loads or a backend profiler
shows alignment matters. Padding/swizzling is measured per backend; it is not applied
blindly because vectorized contiguous layouts can already avoid bank conflicts.

### 5.4 Barrier rules

Every shared kernel must satisfy all of these rules:

- Every live unit reaches each `sync_cube()` the same number of times.
- Bounds checks set an `active` predicate; they do not `return`, `break`, or terminate
  before a barrier shared by active units.
- Tail lanes initialize their shared slots with mathematical neutral values.
- There is a barrier after cooperative population and before the first cross-unit read.
- There is a barrier after the last consumer and before any producer overwrites the
  same region.
- Buffer toggles and loop counts are uniform compile-time or cube-wide values.
- A barrier audit is part of code review and a backend timeout/watchdog test.

### 5.5 Layout and coalescing

Use different layouts for different reuse directions:

- Resident batch inputs: structure of arrays so adjacent units load adjacent items.
- Per-tuple shared descriptors: compact field-major regions loaded cooperatively.
- Recurrence tensors: axis-major and root-contiguous, matching the hottest contraction
  reads; use tiles when the full tensor would reduce occupancy.
- Per-output accumulators: registers owned by one unit, not shared memory.
- Rare shared partials: `[plane][component]` or `[component][plane]` selected by
  measured bank behavior and coalescing.
- ECP/transform matrices: contiguous tiles with the reduction dimension contiguous;
  preserve the original `kk` loop order within each output accumulator.

Shared arrays must have a host-side maximum-index proof. `launch_unchecked` is allowed
only after the validator proves every global and shared range for the selected layout.

### 5.6 Double buffering policy

Double buffering is Phase 8, never the first implementation:

1. Establish a winning synchronous single-buffer shared variant.
2. Profile global-load stalls and barrier cost.
3. Add two compile-time shared tiles and a uniform ping-pong schedule.
4. Prove parity and shared-byte/occupancy viability.
5. Keep the single-buffer variant as the portable control.

Ordinary loads followed by `sync_cube()` do not guarantee true asynchronous copy/math
overlap on every backend. Experimental TMA/barrier APIs are backend-gated and may only
be used in a separate CUDA specialization after CubeCL `0.10.0` capability, compiler,
and runtime tests pass. They cannot become the only implementation of a public family.

## 6. Target execution hierarchy

### 6.1 Work ownership

The baseline cooperative mapping is:

```text
one homogeneous bucket
  -> one or more chunks
    -> one cube per shell tuple or tuple tile
      -> one leader/plane produces roots and sequential recurrence segments
      -> all units consume shared recurrence data
      -> each unit owns disjoint Cartesian/component/contraction output elements
      -> each owned output accumulates primitive/root contributions in reference order
```

For small tuples, one cube may process several tuples, with one plane per tuple. For
large 2e/F12/ECP tuples, one or more cubes may process a tuple tile, but cross-cube
floating accumulation requires a staged deterministic merge and is not the initial
strict variant.

### 6.2 Variant set

Use a bounded variant taxonomy shared by all families:

| Variant | Purpose |
|---|---|
| `NoSharedLane` | Correct batched control; one independent item per lane |
| `SharedDescriptor` | Cooperatively stage small reused descriptor/table data |
| `SharedRecurrence` | Produce recurrence/root data once, consume across output lanes |
| `SharedTiled` | Tile large recurrence, transform, or ECP matrix work |
| `SharedPlanePartials` | Combine plane-owned partials through a small shared region |
| `SharedDoubleBuffer` | Ping-pong a measured memory-bound tile |
| `FusedNoShared` | Fuse stages when global traffic falls without useful cube reuse |

The specialization/autotune key contains:

```text
device fingerprint + backend + CubeCL/codegen version
+ family/operator + representation + precision
+ angular-momentum tuple + nroots + ibase/kbase
+ contraction mode + derivative/component rank
+ batch-size bucket + shared variant/layout + cube dimension
```

Cold autotune cost is reported separately and cached. The shipped/default cache is
advisory; unknown devices run the bounded tuner or use the conservative control.

### 6.3 Reusable dataflow templates

The per-kernel plan refers to these templates:

- **T-A: independent batch lane.** Grid-stride one item per unit. No shared memory
  unless several items deliberately share a descriptor tile.
- **T-B: 2-center recurrence broadcast.** One cube owns a shell pair. Stage shell data;
  leader or plane computes the next primitive/root recurrence tile; output units read
  it and update disjoint AO/component/contraction accumulators in reference order.
- **T-C: multi-center Rys recurrence broadcast.** One cube owns a triple/quartet. Stage
  centers/exponents/coefficients and roots; generate axis/root recurrence tiles;
  distribute Cartesian output elements across units.
- **T-D: tiled contraction/transform.** Cooperatively load matrix/tensor tiles;
  output units retain the original reduction loop order. Suitable for F12 contraction,
  ECP angular work, and device transforms.
- **T-E: plane-per-item math batch.** One plane owns an independent roots/eigensolver
  item. Shared memory holds reusable coefficient tables or matrix state only when
  multiple plane units consume it.
- **T-F: validation/streaming control.** Direct coalesced global access; no shared
  memory because there is no reuse.

## 7. Prerequisite refactors

Shared variants depend on the earlier architecture work and should not clone the
single-tuple launcher pattern.

1. Complete persistent `CubeClContext`, device-resident basis/table handles, reusable
   arenas, and query-time backend identity.
2. Make `BatchExecutionPlan` chunks cover disjoint item ranges and execute each item
   exactly once.
3. Pack homogeneous `KernelClass` descriptor arrays as SoA.
4. Remove one-launch/one-readback-per-tuple behavior from production batch paths.
5. Move or fuse host transforms before claiming a whole-family device speed win.
6. Add real device timing/profiling and per-stage counters.
7. Retain scalar/no-shared paths as references and fallbacks during rollout.

## 8. Per-kernel shared-memory plan

### 8.1 Batch pilot kernels (3)

| Kernel | Current shape | Planned disposition | Required proof |
|---|---|---|---|
| `ss_grid_stride_kernel` | T-A; one independent s-s item per lane | Keep as the no-shared control. Optionally stage a descriptor tile only when a bucket reuses shell data across several items; otherwise shared memory adds traffic and a barrier. | Coalescing profile and `SharedDescriptor` must beat control by 1.05x device time. |
| `ss_plane_cooperative_kernel` | One plane per item; primitive pairs distributed and combined by `plane_sum` | Add a cube variant for primitive-pair counts larger than one plane: plane sums write one shared partial per plane, `sync_cube`, and one plane completes a fixed-order merge. Also test shared exponent/coefficient staging. | Existing plane variant parity, deterministic merge oracle, plane-count/batch crossover. |
| `eri_ssss_grid_stride_kernel` | T-A; one independent primitive `(ss|ss)` item per lane | Keep T-A for small primitive products. Add T-C plane/cube-per-quartet for contracted ssss batches, staging four shell descriptors and root/weight data; use shared partials only under a separate strict-order experiment. | Contracted-batch benchmark must show a crossover; scalar ssss remains control. |

### 8.2 One-electron kernels (16)

All 16 kernels currently execute under `UNIT_POS == 0`. Their shared variants remove
that guard and use output ownership plus T-B.

| Kernel | Shared regions and work mapping | Special risk / exit gate |
|---|---|---|
| `one_electron_scalar_kernel` | Stage exponents, coefficients, centers, and nuclear descriptors. For overlap/kinetic, generate one primitive-pair G tile and let lanes own Cartesian/contraction outputs. For nuclear attraction, stream one center/root tile at a time. | First pilot. Preserve operator `comptime` branches and primitive/center/root order. Must beat the batched no-shared control and libcint over a measured range. |
| `one_electron_grad_bra_kernel` | Shared `g` plus one derivative tile; lanes own the 3 component-leading output elements for each AO/contraction pair. Derivative values may be generated into shared storage or computed from shared `g` at consumption time, whichever uses less shared memory. | Validate headroom indices and component ordering against gradient oracle cases. |
| `one_electron_grad_both_kernel` | Shared base recurrence; compute bra/ket derivative views from the same tile. Partition lanes by AO pair and derivative component, retaining primitive order per output. | Do not duplicate `g1` and `g2` if on-the-fly reads reduce bytes without increasing global traffic. |
| `one_electron_gradgrad_bra_ovlp_kernel` | Tile `g`, `g1`, `g2`, `g3` by axis/primitive pair rather than allocating the full worst-case tensors. Lanes own nine Hessian components. | High shared-byte pressure. Require capacity proof for every supported `l`; fall back to a smaller tile or no-shared variant. |
| `one_electron_p4_kernel` | Stage the extended recurrence tensor and let lanes evaluate independent Cartesian/component outputs. Reuse derivative ladders across all p4 component combinations. | Register and instruction pressure may favor a two-stage tiled kernel. Gate fused and staged variants separately. |
| `one_electron_irp_kernel` | Stage recurrence and origin/coordinate data once per pair. Distribute AO and irp components across lanes. | Origin-dependent index/headroom tests and complex/GIAO-adjacent layout checks. |
| `one_electron_giao_ovlp_kernel` | Shared recurrence plus London/gauge scalars; lanes own real component blocks that later feed complex materialization. | The complete family win requires device-side complex/layout materialization, not only recurrence speed. |
| `one_electron_giao_nuc_kernel` | T-B nuclear center/root tiles plus gauge values. Double buffering may overlap the next center/root tile only after the single-buffer version wins. | Many barriers can dominate small atom counts; retain no-shared and fused controls. |
| `one_electron_grad_kin_both_kernel` | Shared kinetic recurrence and derivative tiles; output lanes own component/AO elements and accumulate in reference primitive order. | Audit all `±2` ladder headroom and kinetic derivative order. |
| `one_electron_nuc_grad_kernel` | Shared Rys/root and nuclear G tiles, consumed by lanes for gradient components. Stage atom coordinate/charge tiles when several lanes reuse them. | Atom/root loop barriers are expensive; tune roots-per-tile and center grouping. |
| `one_electron_rinv_kernel` | Same T-B nuclear template for one inverse-distance origin, with a small shared root/weight region and G tile. | Likely high-value simple nuclear pilot after overlap. |
| `one_electron_drinv_kernel` | Shared rinv recurrence plus derivative tile; distribute derivative components/AO pairs. | Verify origin selection and derivative signs over randomized oracle cases. |
| `one_electron_nuc_grad_both_kernel` | Shared base recurrence with both-side derivative views; lanes own six component groups. | Capacity and barrier count must be compared with recomputation from a smaller base tile. |
| `one_electron_nuc_gradgrad_bra_kernel` | Axis/root tiling of the four recurrence/derivative views; lanes own nine Hessian components. | Do not instantiate full maximum shared tensors when occupancy falls below the tuned floor. |
| `one_electron_gradgrad_bra_kin_kernel` | Shared extended kinetic G tiles and derivative views, partitioned by axis and output component. | Highest 1e recurrence footprint; staged variant is expected to be safer than full fusion. |
| `one_electron_moment_kernel` | Stage recurrence and moment-origin data; distribute Cartesian/moment components across lanes. Bucket by moment rank so tile dimensions are compile-time bounded. | Test low/high moment rank, general contractions, and non-tensor/moment layout families. |

Common exit gate for this group: every supported 1e operator mapped to a modified kernel
passes scalar/shared/libcint parity, forced chunk-size invariance, f32 policy tests, and
backend resource-limit fallback. The release aggregate may not represent only overlap,
kinetic, and nuclear attraction.

### 8.3 Multi-center, two-electron, and F12 kernels (8)

| Kernel | Shared regions and work mapping | Special risk / exit gate |
|---|---|---|
| `two_electron_scalar_kernel` | T-C. One cube owns a shell quartet or a small tuple tile. Stage four shell descriptors, roots/weights, and an axis/root G tile. Lanes own Cartesian/contraction outputs and accumulate primitive quartets in the existing `pi,pj,pk,pl` order. | Primary heavy pilot. Full `3*g_size` may exceed a portable budget; tile roots/axes and specialize by `nroots`, `ibase`, `kbase`, and `l` tuple. |
| `center_2c2e_kernel` | T-C with two shells. Stage roots/weights and recurrence once per primitive pair; lanes own Cartesian/contraction outputs. | Small tuples may be faster with T-A. Autotune by `l`, nroots, and batch bucket. |
| `center_3c1e_kernel` | Stage three shell descriptors and polynomial recurrence tile. Parallelize Cartesian triple outputs while preserving primitive triple order. | Low arithmetic cases may be launch/barrier bound; require end-to-end evidence. |
| `center_3c2e_scalar_kernel` | Stage `urys`, `wrys`, `g`, `g_split`, and HRR work in bounded phases. A leader/plane produces an `(axis, root, k)` tile; lanes consume it for `(i,j,k)` Cartesian outputs. | Current `g + g_split + work` footprint is large. Use shared-memory liveness/free only after a uniform-control proof; otherwise use explicitly reused regions. |
| `center_3c2e_ip1_kernel` | Reuse the scalar T-C base with one derivative view and lanes split by three output components. | Verify derivative headroom and component-leading scatter. |
| `center_3c2e_ip2_kernel` | Reuse T-C with the ket-center derivative view, distributing three components and AO triples. | Separate oracle coverage from ip1; signs and center selection are independent risks. |
| `center_4c1e_kernel` | Stage four shell descriptors and HRR/Cartesian tiles. Lanes own quartet output elements; primitive quartet order remains scalar. | Restricted `Validated4C1E` envelope remains unchanged. Shared optimization cannot expand support. Require oracle plus identity tests for every tuned bucket. |
| `f12_cart_contraction_kernel` | T-D. Cooperatively stage root/axis G tiles and Cartesian component triples. Each lane computes one or more output elements with the original `irys` order. Compare full-tile, root-tile, and fused-direct-global variants. | The current kernel accelerates only contraction; the family claim requires F12/STG/YP root/recurrence migration and one final device layout/readback. |

The 2e derivative, Hessian, GIAO, and relativistic host paths are not silently covered
by optimizing `two_electron_scalar_kernel`. They require their own device kernels or a
documented shared generic kernel before the manifest families count toward the speed
goal.

### 8.4 Sigma and ECP kernels (11)

| Kernel | Shared regions and work mapping | Special risk / exit gate |
|---|---|---|
| `sigma_ov_kernel` | T-B shared overlap/derivative recurrence, with lanes owning sigma component/AO outputs. | Validate spinor transform input order and all four sigma blocks. |
| `sigma_nuc_kernel` | Shared nuclear root/G tiles and sigma assembly data; distribute AO/component outputs. | Barrier count across atoms/roots versus recomputation must be profiled. |
| `sigma_nuc_gauge_kernel` | Same as `sigma_nuc_kernel`, plus shared gauge/origin scalars and gauge-folded component tiles. | Complex/GIAO roundtrip remains release-blocking. |
| `sigma_p_kernel` | Shared base and derivative recurrence tiles; lanes own the four generic sigma-p G-tensor groups per AO/contraction pair. | Begin with single-contraction buckets, then general contractions. |
| `sigma_p_cg_sa10sp_kernel` | Share the common sigma-p recurrence, gauge data, and sa10sp intermediate groups; distribute component blocks. | Large component fan-out makes shared reuse promising but increases register pressure. |
| `sigma_p_spgsp_kernel` | Shared recurrence/origin data; lanes own spgsp component blocks. | Preserve component mix and complex transform input exactly. |
| `sa01_rys_kernel` | Shared roots/weights and one axis G tile per primitive pair/root. Lanes own 9 sigma groups x 4 GC blocks x AO pairs. | Current output fan-out offers high reuse. Tune root-at-a-time versus multi-root tiles. |
| `spgnucsp_rys_kernel` | Shared nuclear Rys/G tile and London phase; lanes own 12-component mixes. | Double buffering across roots/centers only after barrier cost is measured. |
| `spgsa01_rys_kernel` | Shared Rys recurrence plus sa01/spg component data; partition the large output group across lanes. | Shared bytes and registers both high; staged output may beat fusion. |
| `ecp_angular_kernel` | T-D. Tile angular tables and radial values, distribute output AO pairs, and preserve inner angular reduction order. | Compare with direct global reads for small `l`; do not stage tables used once. |
| `ecp_type2_angular_kernel` | Highest-priority T-D/GEMM-like candidate. Stage `prad` and `angi` tiles, materialize the reused dgemm-1 `buf` tile once, then reuse it across `angj` columns instead of recomputing `bufv` inside every output. | Must preserve `kk`/`kk2` order and fit bounded `(li,lj,lc)` tile classes. This can remove substantial duplicate work, but ECP radial host work must also be measured. |

### 8.5 Unstable-source kernels (8)

| Kernel | Shared regions and work mapping | Special risk / exit gate |
|---|---|---|
| `grids_scalar_kernel` | T-B/T-C depending on operator shape. Stage root/axis G data and distribute grid/AO outputs. If grid points are independent, keep a T-A no-shared variant. | Bucket divergence by grid/operator class; never serialize a large grid in one cube. |
| `grids_deriv_kernel` | Shared base and derivative tiles, lanes owning grid x derivative x AO outputs. | Derivative headroom and grid ordering require randomized ROCm oracle coverage. |
| `origi_scalar_kernel` | Shared origin-dependent recurrence and component data; distribute AO outputs. | Common-origin roundtrip and unstable oracle gate. |
| `origi_ip2_kernel` | Shared base plus ip2 derivative view; lanes own derivative components. | Separate random parity corpus from scalar `origi`. |
| `origk_scalar_kernel` | Shared k-origin recurrence and complex component staging; distribute AO outputs. | Complex ordering and f32 path require explicit coverage. |
| `origk_ip1_kernel` | Shared base plus ip1 derivative tile. | Randomized derivative/origin parity and shared-cap fallback. |
| `ssc_scalar_kernel` | T-C for the scalar spin-spin/current-current recurrence. Stage roots and reused G groups; distribute component/AO outputs. | High component count and unstable coverage make it a late rollout. |
| `breit_g_kernel` | T-C/T-D. Stage roots/weights and the three G axes; lanes assemble disjoint Breit component outputs. | Preserve all relativistic signs, component order, and random ROCm parity. |

Unstable-source variants do not block the base-profile libcint win, but a build with
`unstable-source-api` enabled cannot claim completion until these eight kernels pass
their profile-specific parity and performance gates.

### 8.6 Math and diagnostic kernels (8 top-level + 1 nested test)

| Kernel | Planned disposition | Rationale / gate |
|---|---|---|
| `jacobi_tridiag_kernel` | Convert to a batched T-E kernel. Stage immutable `alpha`, `beta`, `rn_part2`, and `sn` tiles once per cube when several same-`n` items reuse them. Keep per-item Wheeler state in registers or a plane-owned shared slice only if measured. | Current one-unit CPU launch/global scratch path must disappear from production batches. |
| `jacobi_transform_kernel` | Prefer fusion into the preceding roots kernel or T-F parallel root writes. Do not add shared memory by default. | Each output root is independent and the small inputs have little reuse; a barrier is likely slower. |
| `schmidt_kernel` | Batched T-E; one plane per item with shared small matrix state only when plane units cooperate on matrix columns/rows. Preserve bounded iteration/status. | Numerical ordering and convergence status are release-blocking. |
| `ljacobi_tridiag_kernel` | Batched T-E double-double path. Stage immutable tables; use carefully laid-out hi/lo shared matrix tiles only if more than one unit consumes them. | Roots 6/7 discrepancy and true-FMA policy must be resolved before tuning. |
| `llaguerre_tridiag_kernel` | Same T-E strategy, bucketed by `n`; reuse table/matrix tiles across plane units. | Double-double shared footprint is large; no-shared/register control is mandatory. |
| `lschmidt_kernel` | Batched T-E with bounded shared hi/lo matrix tiles and per-item status. | Shared footprint, barriers, and convergence branches require uniform scheduling. |
| `fma_probe_kernel` | Keep T-F with direct coalesced accesses and no shared memory. | It is a fidelity probe with no cross-unit reuse; shared memory would invalidate its purpose as a minimal lowering test. |
| `cint_diagonalize_kernel` | Replace one-unit launches with a batched small-matrix T-E kernel. Compare leader-serial state in shared memory against a plane-cooperative fixed-order eigensolver. | Cooperative rotations/reductions may change rounding; scalar order remains the strict control. |
| `boys_f0_sweep_kernel` (nested test) | Keep T-F and no shared memory. | Independent input/output sweep used to validate f64 lowering; no reuse exists. |

The math plan's preferred end state is to inline/fuse roots and small eigensolvers into
the consuming integral batch where that removes launches and readbacks. Shared memory
is a tool for that fused kernel, not a reason to retain a standalone stage.

## 9. Shared-memory sizing and planning

### 9.1 Host-side size proofs

For every `KernelClass`, add a pure host calculator that returns exact maxima for:

- descriptor and coefficient elements;
- roots and weights;
- each recurrence axis/root tile;
- derivative views;
- HRR/split work;
- transform or ECP tiles;
- per-plane partials; and
- alignment/padding.

All arithmetic uses checked operations and returns typed planning errors on overflow.
The query records both full-tensor bytes and selected tile bytes for observability.

Representative formulas are derived from the existing kernels, for example:

```text
1e g_per_axis = (nmax + 1) * (lj_ext + 1)
1e full_g     = 3 * g_per_axis

3c2e g_size   = nroots * (li + lj + 1) * (lk + 1)
3c2e split     = nroots * (lk + 1) * (lj + 1) * (li + 1)

2e/F12 sizes  = existing di/dk/dl/dj/g_size shape calculators
```

The implementation must call the same shape helpers used by the kernel launcher or a
single shared pure shape module; duplicate formulas are prohibited.

### 9.2 Capacity policy

For each backend/device:

1. Read `max_shared_memory_size` from CubeCL properties.
2. Apply a configured portable cap and safety/alignment margin.
3. Generate only capacity classes that fit.
4. Estimate how many cubes the shared-byte choice permits, then confirm with profiling.
5. If a full tensor does not fit, tile roots, axes, Cartesian outputs, or matrix rows.
6. If no profitable tile fits, use `NoSharedLane` or a fused register/global variant.

The selected capacity class is part of the compiled specialization key. Runtime data
must not determine `SharedMemory::new` size without compile-time specialization.

### 9.3 Shared-memory liveness

Prefer one explicitly partitioned shared slab or non-overlapping typed regions. Reuse a
region only after a cube barrier establishes that no consumer remains. CubeCL's unsafe
`SharedMemory::free` may be evaluated later to let the compiler reuse storage across
phases, but only when:

- control flow is uniform;
- all references are dead;
- IR/backend inspection confirms the expected shared-byte reduction; and
- tests cover every backend that enables the optimization.

## 10. Implementation phases

### Phase 0 - Freeze inventory and establish trustworthy baselines

Work:

1. Add a generated/audited kernel inventory containing file, symbol, family, feature,
   current launch geometry, `UNIT_POS` use, scratch buffers, and host/device stage.
2. Add a CubeCL `0.10.0` compile/run spike for `SharedMemory::new`,
   `SharedMemory::new_aligned`, `sync_cube`, f32/f64, capacity-limit reporting, and all
   enabled runtimes.
3. Replace or extend benchmarks so every timed row runs real cintx work.
4. Rerun the current single-tuple three-way benchmark and add real batched GPU/libcint
   pairs.
5. Collect kernel/device timing, occupancy/resource data where available, global-load
   traffic, barrier time, shared bytes, registers, launches, transfers, allocations,
   and p50/p95 end-to-end latency.
6. Write raw samples and environment identity to `/tmp/cintx_artifacts`.

Exit gate:

- Inventory count matches the source and fails CI on an unclassified new launch kernel.
- Every required family has a paired libcint baseline.
- Noise envelope, warmup, sample count, backend, and timing boundary are explicit.
- Shared-memory API spike passes or the plan is revised before production edits.

### Phase 1 - Shared planning, validation, and observability infrastructure

Files: `plane.rs`, `specialization.rs`, executor/context/capability modules, a new
`shared_memory.rs` or equivalent, benchmark telemetry, and runtime planning structures.

Work:

1. Define `SharedVariant`, `SharedLayout`, capacity classes, and autotune keys.
2. Add checked per-family size calculators and a capability-aware selector.
3. Add shared-byte, barrier-count, tile-count, active-unit, and fallback-reason metrics.
4. Add launch validators for all global/shared ranges before `launch_unchecked`.
5. Add a no-shared control to every tuning candidate set.
6. Add deterministic cache serialization/versioning and bounded growth.

Exit gate:

- Query and execute agree on one specialization/layout.
- Forced low shared-memory limits select a smaller tile/no-shared path without partial
  writes.
- Metrics distinguish `not_profitable`, `capacity`, `backend_unverified`, and compile/
  launch failures.

### Phase 2 - Complete batch/residency prerequisites

Work:

1. Finish persistent context and device basis/table residency.
2. Make chunks disjoint item ranges and submit buckets before collective readback.
3. Promote T-A grid-stride batching beyond the s-s pilots.
4. Ensure warm submission performs zero uploads for resident tables/basis data and zero
   arena growth after stabilization.

Exit gate:

- One launch covers a bucket/chunk rather than one tuple.
- Batch output equals ordered scalar concatenation.
- Every item executes exactly once across forced chunk partitions.

### Phase 3 - Three representative shared-memory pilots

- Pilot A: `one_electron_scalar_kernel` overlap/kinetic T-B.
- Pilot B: `two_electron_scalar_kernel` nroots <= 5 T-C.
- Pilot C: `ecp_type2_angular_kernel` T-D.

These cover recurrence broadcast, high-work Rys recurrence, and GEMM-like tiling.

Work:

1. Implement single-buffer synchronous shared variants.
2. Keep output accumulators lane-owned and primitive/reduction order unchanged.
3. Sweep bounded cube dimensions and tile/capacity classes.
4. Compare full tensor, tiled tensor, fused no-shared, and batched no-shared controls.
5. Inspect generated IR/backend code and record actual shared bytes/resource failures.

Exit gate:

- All three pass strict oracle and scalar/shared parity over their full supported pilot
  envelopes.
- At least two of the three show >= 1.05x device-time improvement over no-shared and no
  end-to-end regression.
- The 1e and 2e pilots each demonstrate a measured batch crossover versus libcint or
  produce evidence that a non-shared bottleneck must be fixed first.

### Phase 4 - One-electron and sigma base rollout

Order:

1. 1e overlap, kinetic, rinv, and nuclear scalar.
2. 1e gradient and both-side derivative variants.
3. 1e Hessian/p4/moment variants.
4. Sigma overlap/nuclear/generic sigma-p.
5. Gauge and high-component sigma-p variants.

Exit gate per kernel:

- complete operator-to-kernel oracle coverage;
- size proof for every supported angular-momentum/contraction bucket;
- shared/no-shared benchmark and selected variant artifact;
- no family hidden by an aggregate; and
- host transform time either migrated or reported as remaining dominant work.

### Phase 5 - Multi-center base rollout

Order:

1. 2c2e and 3c1e.
2. 3c2e scalar, ip1, and ip2.
3. 2e scalar common buckets, then high-`l`/capacity fallbacks.
4. 4c1e only within the existing validated envelope.

Exit gate:

- all base families have a contiguous libcint win range or an explicit unresolved
  bottleneck report;
- 4c1e oracle plus identity gates pass;
- no oversized shared launch is possible; and
- cross-backend consistency passes on the supported runner matrix.

### Phase 6 - F12, ECP, optional, and unstable families

Work:

1. Complete F12/STG/YP device root/recurrence path, then tune the shared contraction.
2. Complete ECP radial device work and fuse/stage it with angular tiles where profitable.
3. Roll out unstable-source kernels in the order listed in Section 8.5.
4. Maintain feature-profile-specific inventories and artifacts.

Exit gate:

- every compiled optional/unstable launch kernel has a selected disposition and parity
  gate;
- enabled profiles do not silently use host compute for a claimed optimized stage; and
- per-profile performance is reported separately.

### Phase 7 - High-order Rys and eigensolver batching

Work follows the math plan:

1. Remove per-item one-unit CPU launches and intermediate readbacks.
2. Batch by math class and `nroots`.
3. Compare fused per-integral roots, plane-per-item shared math, and two-stage batched
   pipelines.
4. Resolve f64/double-double/FMA discrepancies before accepting a faster variant.

Exit gate:

- no nested math launch/readback remains in a production integral batch;
- roots 1-12 and downstream integrals pass the oracle corpus; and
- shared math is selected only where it beats fused/no-shared controls.

### Phase 8 - Double buffering, transforms, and bounded autotuning

Work:

1. Add double buffering only to profile-proven memory-bound tiles.
2. Move Cartesian-to-spherical/spinor transforms to T-D device kernels or fuse them.
3. Evaluate vectorized cooperative loads and aligned shared layouts.
4. Add bounded runtime autotuning and shippable cache records.
5. Reject variants that exceed compile-time, register, shared-byte, or cold-start budgets.

Exit gate:

- double-buffer variants show a statistically meaningful gain over single-buffer
  variants;
- one final-layout readback is the normal batch path;
- cold/autotune cost is explicit; and
- cache invalidation is tied to device, codegen, features, and CubeCL version.

### Phase 9 - Release hardening and libcint win decision

Work:

1. Run the full paired corpus on pinned hardware.
2. Run the complete oracle/feature/helper/transform/OOM test matrix.
3. Produce speed, crossover, memory, transfer, occupancy, and residual-risk artifacts.
4. Remove obsolete variants only after two accepted baselines show they are unnecessary.
5. Document unsupported backends/families as unverified rather than extrapolating.

Exit gate:

- Section 4.1 speed gates pass on the designated production GPU;
- Section 4.2 numerical gates pass;
- no caller-visible partial writes or untyped resource failures occur;
- the kernel inventory has no unclassified rows; and
- all required artifacts exist under `/tmp/cintx_artifacts`.

## 11. Benchmark corpus and measurement protocol

### 11.1 Required dimensions

Cover at least:

- batch sizes `1, 8, 32, 128, 256, 1024, 4096`;
- Cartesian, spherical, spinor, and complex outputs where supported;
- f64 for the release comparison and f32 for its separate policy;
- s, p, d, f, and the maximum supported angular momentum, with mixed-l tuples;
- single and general contractions;
- primitive counts 1, 3, 6, and a high-production bucket;
- roots 1-5 and high-order roots 6-12 where used;
- scalar, gradient, Hessian, GIAO, relativistic, F12, ECP, and unstable profiles;
- screened and unscreened batches; and
- small molecules plus representative medium/large production fixtures.

Bucket-level benchmark rows must name the exact operator symbol and kernel variant.
Family aggregates are derived, never the only retained data.

### 11.2 Timing tracks

Record four separate tracks:

| Track | Included work | Use |
|---|---|---|
| Cold end-to-end | context/JIT/autotune/residency plus evaluation | Startup and first-use cost |
| Warm end-to-end | pack, submit, transfer, device, readback, commit | Primary libcint comparison |
| Device stage | timestamp/profiler kernel stages where supported | Diagnose shared-memory effect |
| Kernel micro | isolated tile/recurrence/transform | Tune cause; never prove public win |

Warmup and sample counts are calibrated in Phase 0. Report raw samples, median, p95,
median absolute deviation or confidence interval, and outlier policy.

### 11.3 Required counters

- kernel launch and readback counts;
- H2D/D2H bytes;
- device and host allocations, arena growth, and resident-cache hits;
- planned and actual shared bytes;
- cube dimension/count and active item count;
- tile and barrier counts;
- fallback/selected variant;
- compile and autotune time;
- register/occupancy or closest backend evidence when available;
- output elements and integral items per second; and
- oracle maximum absolute/relative error for the same case.

Do not use the current aggregate threshold baselines to claim a libcint win. Extend
`bench-report` with paired libcint ratios and fail closed when real samples are missing;
threshold fallback rows must remain labelled unmeasured.

## 12. Verification plan

The test plan follows `docs/rust_crate_test_guideline.md`: passing examples or coverage
alone do not prove conformance, and verified/unverified scope must remain explicit.

### 12.1 Specification-to-test-to-gate map

| Requirement | Verification | Gate |
|---|---|---|
| libcint-compatible f64 outputs | Existing manifest oracle fixtures plus randomized inputs for every touched operator/bucket | PR for touched kernels; full nightly/release |
| Scalar/shared equivalence | Run the same ordered input through scalar, no-shared batch, and each shared variant | PR |
| Chunk/geometry invariance | Force chunk sizes, cube dimensions, shared tile sizes, and batch permutations | PR |
| Accumulation-order preservation | Adversarial exponent/coefficient cases and bit/ULP diagnostics in addition to tolerance | PR/nightly |
| Uniform barriers/no races | Static review checklist, backend timeout tests, varied tail-lane cases, Miri for host validators, backend race-oriented stress | PR/nightly |
| Shared capacity correctness | Boundary tests at exact fit, one byte/element over, low artificial limits, f32/f64 | PR |
| No partial writes | Allocation/compile/launch/readback failure injection with sentinel caller buffers | PR |
| OOM-safe planning | Checked arithmetic properties, forced limits, allocator/device resource failures | PR/nightly |
| Backend portability | CPU compile/reference, WGPU capability run, ROCm/CUDA/Metal where runners exist | Nightly/release; absent runners unverified |
| Performance win | Paired raw cintx/libcint samples on pinned hardware | Release |
| No hidden host compute | Stage counters, launch/transfer traces, and host profile | Nightly/release |
| Inventory completeness | Source inventory generation versus checked-in disposition table | PR |

### 12.2 Mandatory Rust verification layers

- `cargo test` unit/integration/regression and doctests.
- `proptest` for shape/size arithmetic, shared-layout selection, chunking, permutation,
  output offsets, and forced memory limits.
- `cargo-mutants` focused on size checks, barrier predicates represented in host
  planning code, fallback selection, and transactional output behavior.
- `cargo-hack` feature matrix for base, `with-f12`, `with-4c1e`, backends, precision,
  and `unstable-source-api` combinations.
- `cargo-llvm-cov` reporting with unverified device-only paths called out.
- `trybuild` or `ui_test` for any new public compile-time batch/session contracts.

Conditional layers:

- Miri for unsafe host buffer/`ArrayArg` validation and raw compatibility plumbing.
- Loom for shared context/arena/autotune-cache concurrency.
- Kani for bounded shared-size/offset/no-overlap invariants where tractable.
- `cargo-fuzz` for hostile raw `atm/bas/env/shls/dims` and batch descriptor inputs.

Tool unavailability, timeouts, waived mutants, unsupported runtimes, and unexecuted
hardware are recorded as residual risk; they are not converted into pass claims.

### 12.3 CI tiers

PR:

- format/clippy;
- affected crate and oracle tests;
- scalar/shared/no-shared parity for touched kernels;
- layout properties and capacity boundary tests;
- base feature combinations;
- inventory drift check; and
- a short benchmark smoke that detects catastrophic regression but does not enforce the
  libcint release ratio on unpinned hosts.

Nightly:

- full oracle profiles and randomized corpus;
- full feature matrix, property, mutation, Miri/Loom/Kani/fuzz schedules as applicable;
- supported GPU backend consistency;
- resource pressure and fault injection;
- warm allocation/upload stability; and
- extended batch/cube/tile invariance.

Release:

- pinned hardware/software paired benchmark corpus;
- all oracle, helper, transform, optimizer, OOM, and feature gates;
- required artifact schema validation;
- explicit verified/unverified backend table; and
- Section 4.1 libcint win decision.

## 13. Required artifacts

Write at least these deliverables under `/tmp/cintx_artifacts`:

- `cintx_shared_memory_kernel_inventory.json`
- `cintx_shared_memory_baseline_raw.jsonl`
- `cintx_shared_memory_baseline_summary.md`
- `cintx_shared_memory_layout_catalog.json`
- `cintx_shared_memory_autotune_results.jsonl`
- `cintx_shared_memory_selected_variants.json`
- `cintx_shared_memory_parity_report.json`
- `cintx_shared_memory_resource_report.json`
- `cintx_shared_memory_vs_libcint_raw.jsonl`
- `cintx_shared_memory_vs_libcint_summary.json`
- `cintx_shared_memory_vs_libcint_summary.md`
- `cintx_shared_memory_residual_risks.md`

Every performance artifact includes:

- git revision and dirty-state marker;
- Cargo.lock hash, Rust version, CubeCL version, features, and optimization profile;
- libcint revision/build flags and optimizer/thread settings;
- device/backend/driver/runtime/capability fingerprint;
- shared-memory limit and selected layout;
- warmup/sample/timing method;
- operator/representation/l tuple/nroots/contraction/batch details;
- raw timings and derived ratios; and
- oracle status for the same case.

## 14. File-level change map

| Area | Planned changes |
|---|---|
| `crates/cintx-cubecl/src/plane.rs` | Shared launch geometry, plane/cube helpers, capacity-aware cube candidates |
| New shared-memory module | `SharedVariant`, layout regions, cooperative load/reduce helpers, barrier-safe primitives |
| `specialization.rs` | Add shared variant/layout/cube dimensions and bounded autotune keys |
| executor/context/capability | Read max shared bytes, select/cache variants, expose timing/resource metrics |
| `batch_pilot.rs` | Retain T-A controls; add shared plane/cube pilots |
| `kernels/one_electron.rs` | T-B shared recurrence rollout for all 16 launch kernels |
| `kernels/two_electron.rs` | T-C scalar 2e pilot and later generic derivative/device migration seams |
| center-family modules | T-C recurrence/HRR/output tiling per Section 8.3 |
| `kernels/f12.rs` | T-D contraction plus integration with device F12 math |
| sigma modules | Shared recurrence/component staging per Section 8.4 |
| `kernels/ecp.rs` | T-D angular tiling and reusable type-2 intermediate |
| unstable modules | Late profile-gated rollout per Section 8.5 |
| `math/rys_wheeler.rs`, `math/eigh.rs` | Batched T-E and fused consumer paths |
| transform modules | T-D device transforms and resident coefficient tables |
| runtime planner/batch/workspace | Exact shared bytes, capacity/fallback, disjoint chunks, OOM accounting |
| benches/oracle/xtask | Paired libcint cases, raw samples, ratios, inventory and artifact validation |
| CI | PR/nightly/release tiers and pinned hardware release gate |

File names for new modules are suggestions. Preserve the crate boundaries and avoid
exposing CubeCL types through the public safe API.

## 15. Risks and controls

| Risk | Control |
|---|---|
| Shared memory is added but units remain idle | Require active-unit metrics and cooperative output ownership before tuning |
| Barriers cost more than saved loads | Always benchmark a no-shared control; 1.05x device gain threshold |
| Shared footprint reduces occupancy | Capacity classes, half-limit initial reserve, tiled fallback, profiler evidence |
| Floating reduction order changes results | Lane-owned outputs, reference primitive/root order, no unordered atomics |
| Divergent barrier deadlock | Uniform loop/barrier rules, tail neutral values, timeout/stress tests |
| Bank conflicts or scattered loads erase gains | Contiguous/vectorized layouts first; backend profile before padding/swizzle |
| Variant/JIT explosion | Bounded structural keys and capacity classes; generic fallback; cache accounting |
| Double buffering does not overlap on a backend | Keep synchronous control; require measured gain; backend-gate async APIs |
| Host transforms/math dominate after kernel speedup | Stage timing and migration gates before family speed claim |
| ECP/F12/unstable coverage is hidden by core aggregate | Per-profile and per-family rows; no aggregate-only acceptance |
| Current benchmarks give misleading ratios | Paired end-to-end protocol and raw samples; fail closed on missing data |
| Shared resource errors leak or partially write output | Query-time sizing, typed errors, transactional internal output |
| Worktree evolution makes inventory stale | Generated inventory drift check in PR CI |

Rollback is variant-level: retain the previous accepted no-shared/scalar route until a
shared variant passes two consecutive accepted baselines. A backend-specific regression
disables only that selected variant through the versioned tuning policy; it does not
remove family support.

## 16. Recommended first implementation slice

The smallest slice that validates the design without spreading changes across every
family is:

1. Kernel inventory and real paired batch/libcint benchmark artifacts.
2. `SharedVariant`/`SharedLayout` planning and max-shared-memory capability capture.
3. One generic barrier-safe cooperative-load helper plus capacity boundary tests.
4. T-A no-shared and T-B shared variants for batched Cartesian 1e overlap/kinetic.
5. T-C shared variant for a bounded 2e nroots<=5 bucket.
6. T-D shared intermediate for bounded ECP type-2 angular buckets.
7. Scalar/no-shared/shared/libcint parity for all three pilots.
8. Cube/tile/autotune sweep and end-to-end crossover plots.
9. A go/no-go report that identifies whether recurrence, transforms, transfers, or
   submission is now dominant before wider migration.

Do not begin by editing all 54 top-level kernels. Prove the three dataflow templates,
then migrate one kernel group at a time with a measurable exit gate.

## 17. Definition of done

This plan is complete only when:

- the inventory classifies every current and newly added launch kernel;
- each production kernel has a validated no-shared/shared/fused disposition rather
  than an assumed shared-memory benefit;
- all supported outputs satisfy the libcint oracle and project numerical policy;
- batch, chunk, geometry, and selected variant do not change logical results;
- shared-memory limits, OOM, and launch failures are typed and transactional;
- warm execution is resident, allocation-stable, and uses batched launches/readbacks;
- all required base families have a measured libcint crossover;
- the designated production GPU meets the 1.20x geometric-mean release goal without a
  required-family regression below 0.95x at batch 1024;
- optional/unstable and unsupported backend scope is reported honestly;
- PR, nightly, and release verification tiers pass; and
- raw evidence, selected variants, parity, speed, memory, and residual risks are saved
  under `/tmp/cintx_artifacts`.

## 18. Verified facts and open hypotheses at plan time

Verified from the current source and tooling:

- 54 top-level launch kernels exist, plus one nested test-only Boys sweep.
- 43 integral kernels serialize all work behind `UNIT_POS == 0`.
- No current cintx kernel uses CubeCL shared memory or a cube-wide barrier.
- The pinned CubeCL API supports compile-time shared arrays, aligned shared arrays,
  `sync_cube`, max-shared-memory queries, and explicit resource failure.
- Current single-tuple CubeCL CPU performance is far behind libcint for the audited
  historical cases.
- Current `/tmp/cintx_artifacts` contents do not contain a fresh performance baseline
  sufficient for a libcint-win claim.

Open hypotheses that measurement must resolve:

- Which GPU/backend and batch size first beats libcint for every base family.
- Whether shared recurrence broadcast beats batched no-shared recomputation for each
  angular-momentum/root bucket.
- Whether full, root-tiled, or axis-tiled G storage gives the best occupancy/performance
  tradeoff.
- Whether ECP type-2 shared intermediate reuse remains significant after radial work is
  moved to the device.
- Whether plane-cooperative high-order roots preserve the strict f64 oracle envelope.
- Which kernels benefit from double buffering on each backend.
- Whether device transforms should be fused or staged after shared recurrence tuning.

Until these hypotheses have raw paired evidence, the plan describes candidate
optimizations and release gates, not an achieved libcint speed win.
