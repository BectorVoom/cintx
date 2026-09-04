# def2-SVP / def2-TZVP Speed and Memory Efficiency Plan

Status: executed 2026-09-03 — see §10 for what landed, what was measured, and the
one workstream blocked by a backend defect (§10.6). G2 (speed) is **unresolvable on this
host**; §10.2 gives the ranges and the effects that were resolvable.
Scope: follow-on to `docs/design/def2_speed_precision_plan.md` (executed 2026-09-02, §7
there is the record). The batched `int2e_sph` path first; the other batched families where
the same mechanism applies.
Primary compatibility target: libcint 6.1.3, unified oracle tolerance `atol = rtol = 1e-12`
Backends that matter: CubeCL `cpu` and `rocm` (gfx1151 on the dev host). wgpu f64 is out of
scope by decision.
CubeCL target: pinned workspace version `0.10.0`

## 1. Purpose

The previous plan got def2-TZVP onto the device (nroots 6–12, zero refusals) and produced the
first honest win: batched cintx beats single-threaded libcint by 1.4–1.8x on the CPU backend
over whole screened work lists. Its speed workstreams removed launch count and cold JIT from
the hot path. What is left is what the throughput artifact now makes visible:

- **Speed** is now arithmetic per primitive quartet, and that arithmetic is spent in the
  deep-primitive *low*-angular-momentum classes, not in the f-functions TZVP is known for.
  The kernel evaluates every `nprim^4` primitive quartet where libcint skips most of the
  negligible ones, recomputes ket pair data `nprim_i * nprim_j` times per quartet, and
  accumulates into global memory once per primitive quartet per element.
- **Memory** scales with the whole molecule's ERI tensor on both host and device, the batch
  path ignores `ExecutionOptions::memory_limit_bytes`, device scratch is re-allocated on every
  launch (and every tuning candidate), and the host holds the Cartesian *and* the spherical
  copy of the whole work list at once. For a 30-atom TZVP system this is not a workflow at all.

Both are addressed with one rule kept from the previous plan: **no number without the def2
throughput benchmark's rules** — same screened list for both engines, values compared before
timing, coverage and (now) memory printed.

## 2. Current state (evidence, not aspiration)

All rows below are from `artifacts/cintx_def2_throughput.json` (schema
`cintx_def2_throughput/1`, CPU backend, release, best of 9, `extended-device-rys` on,
0 mismatched elements throughout) unless a file is named.

### 2.1 Where the time goes

| workload | quartets kept | launches / classes | libcint (ms) | cintx wall (ms) | dispatch (ms) | host cart→sph (ms) | transfer (KiB) |
|---|---|---|---|---|---|---|---|
| H2O / def2-SVP | 3 081 | 15 / 69 | 2.31 | 2.10 | 1.82 | 0.39 | 77 |
| CH4 / def2-SVP | 14 706 | 15 / 69 | 9.45 | 5.12 | 3.86 | 0.61 | 350 |
| SO2 / def2-SVP | 21 271 | 16 / 81 | 29.27 | 20.97 | 18.38 | 1.85 | 505 |
| H2O / def2-TZVP | 18 145 | 23 / 172 | 15.55 | 10.37 | 9.59 | 1.80 | 437 |
| SO2 / def2-TZVP | 181 070 | 24 / 256 | 348.79 | 249.93 | 217.29 | 29.03 | 4 261 |

`dispatch` is uploads + kernel + synchronous readbacks for every group; it is 87–92 % of
wall. The host transform is 8–12 %. Launch count is no longer the lever: 24 launches at the
measured ~42 µs each is 1 ms of a 250 ms run.

### 2.2 Where the primitive work is

`primitive_work` per bucket row is `Σ nprim_i·nprim_j·nprim_k·nprim_l` — the number of
primitive quartets the kernel walks. Shares by Rys order:

| workload | nroots 1 | 2 | 3 | 4 | 5 | 6–7 | max `nprim^4` in a class |
|---|---|---|---|---|---|---|---|
| SO2 / def2-SVP | 32.9 % | 52.7 % | 13.7 % | 0.7 % | 0.0 % | — | 625 |
| H2O / def2-TZVP | 48.4 % | 37.9 % | 11.6 % | 1.9 % | 0.2 % | < 0.1 % | 1 296 |
| SO2 / def2-TZVP | 22.3 % | 44.3 % | 26.0 % | 6.6 % | 0.8 % | < 0.1 % | 2 401 |

Two facts follow. **nroots ≤ 3 carries 93 % (SO2/TZVP) to 98 % (H2O/TZVP) of the
arithmetic**; the extended-Rys classes the previous plan spent its effort on are a coverage
question, not a throughput one. And **`max_nctr_product` is 1 in every bucket of every
workload**: def2 is segmented, so D2.4 (the contraction-shape question) is closed by
measurement — there is no contracted quartet to choose a kernel arm for.

### 2.3 What the kernel does per primitive quartet

Verified in `crates/cintx-cubecl/src/kernels/two_electron.rs`, `two_electron_scalar_kernel`
(line 794 onward):

1. **Ket pair data is recomputed inside the bra loop.** `rkl`, `rr_kl` and `fac_kl = exp(…)`
   (lines 1024–1034) are computed for every `(pi, pj, pk, pl)`, i.e. `nprim_i·nprim_j` times
   per ket pair. libcint forms pair data once per shell pair (`CINTset_pairdata`,
   `optimizer.c:288`) and reuses it.
2. **No primitive-pair cutoff.** `prim_tol` defaults to `0.0`, so the only primitive quartets
   skipped are those whose prefactor underflowed to exactly zero. libcint's *no-optimizer*
   path skips a primitive pair when `cceij >= expcutoff` (`optimizer.c:326`) and a primitive
   quartet when `cceij > expcutoff - ccekl` (`cint2e.c:205–237`), with `expcutoff` from
   `env[PTR_EXPCUTOFF]` (default `EXPCUTOFF`). Those contributions are therefore **absent
   from the vendor reference cintx is compared to**; computing them costs time and moves
   cintx (immeasurably, so far) *away* from the vendor.
3. **Rys roots and the whole VRR/HRR G build run on lane 0.** In the cooperative
   decomposition (`per_unit == 0`, the GPU shape) the other `cube_dim - 1` lanes idle through
   the G build and join only for the contraction, with two `sync_cube` barriers per
   primitive quartet. On the CPU per-unit shape this costs nothing; on ROCm it is the shape.
4. **Accumulation is a global-memory read-modify-write per primitive quartet per element**
   (`cart_out[out_off + q_elem] += prim_weight * sum`, line 1559). For a `(ss|ss)` class on
   SO2/TZVP that is up to 2 401 RMWs per output element.
5. **The G slab lives in global memory**, one `3 * g_size_max` slab per slot, sized to the
   widest class merged into the dispatch (`g_slab_stride`, line 1764). No 2e kernel uses
   `SharedMemory` (grep count 0); the shared-memory plan's 2e item has not been executed.

### 2.4 Where the memory goes

**Host** (`evaluate_2e_batch_inner`, line 7221):

- `values: vec![0.0; total]` — the full spherical output, allocated before any dispatch.
- `carts: Vec<Vec<f64>>` — one Cartesian buffer **per group, all retained** until the
  transform over the whole list finishes (line 7345 onward). Each is produced by
  `client.read_one_unchecked(out_h)` followed by `.to_vec()` (line 2122), so a group's
  Cartesian block transiently exists twice.
- The quartet table: 6 × `u32` per quartet, plus 13 × `u32` per class.

Footprints for the *unscreened* 8-fold lists, computed from the embedded basis tables:

| workload | quartets | spherical output | Cartesian output | quartet table | host peak today (≈ sph + cart) |
|---|---|---|---|---|---|
| SO2 / def2-SVP | 22 155 | 4.9 MiB | 6.4 MiB | 0.5 MiB | ~11 MiB |
| H2O / def2-TZVP | 18 145 | 3.9 MiB | 6.2 MiB | 0.4 MiB | ~10 MiB |
| CH4 / def2-TZVP | 71 631 | 9.9 MiB | 14.3 MiB | 1.6 MiB | ~24 MiB |
| SO2 / def2-TZVP | 198 765 | 99.8 MiB | 177.9 MiB | 4.6 MiB | ~280 MiB |

Cartesian is 1.78x spherical for TZVP because f-shells carry 10 Cartesian vs 7 spherical
components. The host also runs the cart→sph transform serially with respect to the device:
every group is dispatched and read back before the first block is transformed.

**Device** (`run_2e_batches`, line 2037, and `TwoEGroupDispatch::launch`, line 2175):

- Per group: quartet/shape/factor tables, the extended-Rys constant tables (~4.7 KB,
  **re-uploaded on every dispatch**), the Cartesian output `out_len * 8` bytes, and the G
  slab `n_slots * g_stride * 8` allocated **inside `launch()`** — so under tuning every
  candidate width allocates its own slab.
- Cooperative shape: `n_cubes = min(n_quartets, MAX_BATCH_SCRATCH_BYTES / slab)` with
  `MAX_BATCH_SCRATCH_BYTES = 256 MiB` (line 1773). SO2/TZVP's nroots-3 group is 61 205
  quartets at 3 456 B/slot = **211 MB of G scratch for one launch**; its nroots-2 group is
  63 960 at 576 B = 37 MB. Freed after the launch; never reused.
- Per-unit shape (CPU): slots = `min(parallel_units, n_quartets, by_memory)`, so ≤ 16 slots
  × 131 712 B = 2.1 MB. Memory is not a CPU-backend problem today; it is a GPU and a
  large-molecule problem.

**Contracts not honoured by the batch path** (verified by grep: no `memory_limit_bytes`,
`MemoryLimitExceeded` or `ChunkPlanner` reference in `two_electron.rs`):

- `ExecutionOptions::memory_limit_bytes` and `chunk_size_override` are consumed only by
  the per-tuple `query_workspace` route (`crates/cintx-runtime/src/workspace.rs:120`).
- A whole group is one launch and one output buffer, whatever its size; there is no chunking.
- `client.empty` is not fallible at this layer; a device OOM is not a typed
  `DeviceOutOfMemory` and does not honour the design's no-partial-write rule.

### 2.5 JIT and program identity

Cold prewarm is 5.1 s (H2O/SVP), 6.8–6.9 s (TZVP) per process. The second SVP case in the
same process warmed in 2 ms — but CH4/SVP, with the *same 15 signatures*, took 0.53 s: the
compiled program's identity includes the launch geometry, and a different molecule size
picks a different per-unit width. `cubecl.toml` sets `[compilation] cache = "target"`; whether
the CPU (MLIR) runtime honours it across processes has not been measured.

## 3. Gates

Every workstream inherits the general plan's correctness gates and the previous plan's G5
(honesty). Added:

- **G1 (parity)**: `def2_2e_batch_parity`, `def2_quartet_batch_facade`,
  `def2_device_coverage`, `ext_rys_*_parity` and the ROCm suites stay green. Every kernel
  change is compared element-wise against vendored libcint on all five benchmark lists at
  `1e-12`; the benchmark's `mismatched_elements` stays 0 and `max_abs_diff_vs_vendor` is
  recorded per change, not just checked.
- **G2 (speed)**: on the CPU backend, SO2/def2-TZVP and SO2/def2-SVP wall time improves by
  ≥ 1.5x over the 2026-09-02 rows (249.9 ms, 21.0 ms) with no other row regressing. ROCm rows
  are recorded separately and are not required to beat the CPU (see the gfx1151 note in
  `def2_rocm_extended_and_tuning.rs`).
- **G3 (memory)**:
  - full-list mode: host peak ≤ 1.25x the spherical output size (today ≈ 2.8x);
  - bounded mode: with `memory_limit_bytes = L`, host + device peak ≤ `L + fixed overhead`,
    where the overhead is reported, not assumed;
  - a request that cannot fit returns `MemoryLimitExceeded` / `DeviceOutOfMemory` **before
    any launch**, and no output element has been written;
  - device scratch is allocated once per batch, not once per launch.
- **G4 (observability)**: the throughput artifact carries a `memory` block per case
  (schema `cintx_def2_throughput/2`); no memory claim is accepted outside it.
- **G5 (API)**: no CubeCL type crosses `cintx-rs`; the memory budget is expressed through
  `ExecutionOptions`, the streaming surface through plain closures and `&mut [f64]`.

## 4. Workstreams

Two tracks. **S** is speed, **M** is memory. M6 (observability) comes first because every
other item is measured through it.

### M6 — Memory and primitive-work observability (prerequisite)

Files: `crates/cintx-cubecl/src/kernels/two_electron.rs` (`BatchExecutionStats`),
`crates/cintx-oracle/tests/def2_throughput_benchmark.rs`, `xtask/src/bench_report.rs`.

1. Add to `BatchExecutionStats`: `host_output_bytes`, `host_cart_bytes_peak`,
   `device_out_bytes_peak`, `device_g_slab_bytes_total`, `device_bytes_in_use_peak` (from
   `client.memory_usage()`, which cubecl-runtime 0.10 exposes on the client), and
   `primitive_quartets_evaluated` / `primitive_quartets_skipped` (a device-side counter,
   read back with the output; off by default under an option so the hot path pays nothing).
2. Print and record them in `run_batch_case`; bump the artifact schema to
   `cintx_def2_throughput/2` with a `memory` block per case and a `primitive_work` block.
3. Route the new block into `xtask bench-report`'s
   `cintx_cubecl_memory_report.json`, which today has no def2 rows.

Exit: the 2026-09-02 rows are re-recorded with memory columns as the baseline for G3.

### S1 — Primitive-pair data and libcint's pair cutoff (highest value)

Files: `two_electron.rs` (kernel and `ResidentTwoEBasis`), `crates/cintx-driver`
(pair enumeration already exists), a new parity test.

1. **Device-resident pair table.** For every canonical shell pair build, once per basis, the
   compacted list of surviving primitive pairs with `aij⁻¹`, `rij[3]`, `eij = exp(−…)` and
   `cceij`, exactly as `CINTset_pairdata` forms them (`optimizer.c:288–341`, including the
   `log_maxci` coefficient bound and the `lij`-dependent `log_rr_ij` term). Store it with
   `ResidentTwoEBasis` so it costs one upload per basis, not per call. Upper bound for
   SO2/TZVP: 630 pairs × 49 primitive pairs × 6 doubles ≈ 1.4 MiB.
2. **Kernel walks the compacted lists** for bra and ket instead of the `nprim` loops, reading
   pair data rather than recomputing it (this removes the `exp` and the division from the two
   inner loops). Apply the quartet-level cutoff `cceij > expcutoff − ccekl` the way
   `cint2e.c:205–237` does, with `expcutoff` taken from `env[PTR_EXPCUTOFF]` under the raw
   API and from a new `ExecutionOptions` field (default: libcint's `EXPCUTOFF`) under the safe
   API. `prim_tol` stays as the *additional*, opt-in, non-vendor screen it is today.
3. **Redefine the identity gate honestly.** The tolerance-zero identity gate today says
   "screened equals unscreened bit for bit". After S1 the default kept set is *libcint's*, so
   the gate becomes: (a) with the cutoff disabled (`expcutoff = +inf`) results are bit-identical
   to today's kernel; (b) with the default cutoff, element-wise agreement with vendored libcint
   over all five lists is ≤ the 2026-09-02 `max_abs_diff_vs_vendor` (5.9e-14 on SO2/TZVP) —
   expected to shrink, required not to grow.
4. Record `primitive_quartets_skipped / evaluated` per case. The reduction depends on
   geometry; the plan does not assume a factor, it measures one.

Exit: G1 holds under both cutoff settings; the primitive-work counters are in the artifact;
SO2/TZVP dispatch time recorded before and after.

### S2 — Register-resident accumulation

Files: `two_electron.rs` kernel.

1. Replace the per-primitive-quartet `cart_out += …` with a per-lane private accumulator
   written to `cart_out` **once per quartet**. The accumulator's capacity is comptime per
   launch signature: the widest Cartesian block a given `nroots` can carry, divided by the
   lane count (`ffff` is 10 000 elements at nroots 7; at nroots ≤ 3 the widest block is ≤ 108),
   so it is small on the cooperative shape and cache-resident on the per-unit shape.
2. Keep the summation order per element unchanged (sequential over primitive quartets in
   the same order as today), so the result is bit-identical to the current kernel — that is
   the gate, checked by `def2_2e_batch_parity` and by a new bit-identity assertion in the
   benchmark's A/B mode (`CINTX_2E_ACCUMULATE=global|private`).
3. Fold the ket-pair hoisting from S1 into the same loop restructuring so the kernel is
   touched once.

Exit: bit-identity with the pre-S2 kernel; per-quartet `cart_out` write count in the stats
drops from `Σ nprim^4 · block` to `Σ block`.

### M1 — Bounded-memory batch execution

Files: `two_electron.rs` (`evaluate_2e_batch_inner`, `run_2e_batches`),
`crates/cintx-runtime/src/workspace.rs` (`ChunkPlanner`, `FallibleBuffer`),
`crates/cintx-rs/src/api.rs` (`QuartetBatchRequest::evaluate_in`).

1. **A memory plan before any launch.** For a work list compute, per group: device bytes
   (`out_len·8 + tables + n_slots·g_stride·8`) and host bytes (`sph + cart` for the group).
   Sum against `memory_limit_bytes` (device and host budgets separately when the backend is
   discrete; one budget on the CPU backend where they are the same memory). Fail with
   `MemoryLimitExceeded { requested, limit }` when even a single-chunk plan cannot fit,
   **before allocating the output**.
2. **Chunk groups by quartet range** when a group exceeds its share of the budget, reusing
   `ChunkPlanner` semantics (`min_chunk_bytes` = one quartet's block + one slot's slab) and
   reporting `fallback_reason = "memory_limit"` exactly as the per-tuple path does. A chunk is
   dispatched, read back, transformed into its slice of `values`, and its Cartesian buffer
   released before the next chunk is dispatched — that alone takes the host peak from
   `sph + cart(all)` to `sph + cart(chunk)`.
3. **Replace the `MAX_BATCH_SCRATCH_BYTES` constant** with the scratch share of the plan:
   256 MiB stays the default when no limit is set; with a limit, the cooperative cube count is
   derived from what the plan allots to scratch.
4. **Typed device OOM.** Wrap the batch path's `client.empty` / `create_from_slice` calls in
   the fallible allocation seam the design requires, mapping an allocation failure to
   `DeviceOutOfMemory { bytes, device }`; on failure nothing has been written to `values`
   because it is now allocated last (via `FallibleBuffer` → `HostAllocationFailed`).
5. Surface the plan through `QuartetBatchRequest` (`ExecutionOptions.memory_limit_bytes`
   already exists; document that the batch path now honours it) and through
   `BatchExecutionStats.chunk_count` / `fallback_reason`.

Exit: G3's bounded-mode and OOM clauses; an oracle test that runs SO2/TZVP under a limit that
forces ≥ 4 chunks and asserts bit-identity with the unchunked run (chunking changes no
arithmetic, only when a group is launched).

### M4 — Scratch and table reuse

Files: `two_electron.rs` (`run_2e_batches`, `TwoEGroupDispatch::launch`),
`crates/cintx-cubecl/src/executor.rs`.

1. Allocate the G slab **once per batch** at the widest `n_slots · g_stride` any group needs
   and reuse it across groups and across tuning candidates (the tuner re-launches the same
   dispatch; today every candidate allocates a fresh slab inside `launch()`).
2. Make the extended-Rys constant tables resident alongside the basis (`TwoEBasisHandles`),
   removing the per-dispatch 4.7 KB upload and its allocation.
3. Retain the output staging handle in `EvaluationContext` the way `pilot_output_arena`
   already does for the s-s pilot, so repeated Fock builds do not re-allocate the output
   buffer per call; report reuse counts in the stats.

Exit: `number_allocs` from `client.memory_usage()` per batch drops to O(1) + O(groups)
rather than O(launches × candidates); recorded in the artifact.

### M3 — Device-side cart→sph and spherical readback

Files: `crates/cintx-cubecl/src/transform/c2s.rs` (tables exist, host-only today; the
`#[cube]` casting note at line 15 is about the spinor path), new device transform kernel,
`two_electron.rs`.

1. Add a device transform pass per group: Cartesian block → spherical block, scattered into
   a device-resident spherical output in the caller's quartet order. The C2S coefficient
   tables (`c2s_data.rs`) become a resident constant buffer.
2. Read back **spherical** only. For SO2/TZVP that is 100 MiB instead of 178 MiB across the
   PCIe/host path, the `carts` retention and the `Bytes → Vec` copy disappear, and the
   8–12 % host transform leaves the wall clock.
3. Gate: the transform is a fixed matrix contraction, so with the host's operation order
   reproduced per element the result is bit-identical to the host transform; assert that in
   `def2_2e_batch_parity` (host vs device transform, same Cartesian input) rather than
   loosening to a tolerance. Keep the host transform as the fallback for a backend that
   cannot host the pass and as the A/B reference.
4. On the CPU backend the "device" transform runs on the same cores as the host one, so the
   win there is memory, not time; say so in the artifact.

Exit: G3's full-list clause (host peak ≤ 1.25x spherical); `host_transform_ns` is 0 in
device-transform mode and the mode is named in the artifact row.

### S4 — Asynchronous pipeline

Files: `run_2e_batches`.

1. Submit every group's dispatch before the first readback (CubeCL launches are lazy; only
   `read*` and `sync` force completion), then read back in order. Under M1's chunking this
   becomes a two-deep pipeline: chunk N's readback and transform overlap chunk N+1's launch.
2. Use `client.read_async` for the in-flight readback and `client.sync()` only at batch end.
3. Measure, do not assume: on the CPU backend the device *is* the host, so the overlap gain
   is bounded by the transform's share (8–12 %); on ROCm it is the whole readback.

Exit: `dispatch_ns` and `host_transform_ns` are reported as overlapped/serial with the same
totals as before; wall improves by the measured overlap.

### S3 — Cooperative G-build parallelism and shared-memory G (ROCm)

Files: `two_electron.rs` kernel (cooperative arm), `crates/cintx-cubecl/src/shared_memory.rs`
(`calc_2e_layout` exists and is unused by the 2e kernel).

1. Distribute primitive quartets across planes within the cube, one G slab per plane, with a
   deterministic reduction (fixed plane order) into the per-lane accumulators from S2. This
   is the second half of the note's Task 34-D, deferred then because it competed with the
   launch-count work that is now done.
2. Place the per-plane G slab in shared memory for every class whose slab fits the device's
   `max_shared_memory_size` (all of def2-SVP, and every TZVP class below the
   `CubePerQuartetGlobal` tier — 99.9 % of TZVP quartets by count); keep the global slab for
   the f-heavy remainder. This is where the shared-memory plan's 2e item is actually
   executed, with `calc_2e_layout` as the host-side size proof.
3. Precision: distributing primitive quartets changes summation order relative to libcint,
   so bit-identity with the per-unit CPU result is **not** the gate; the gate is the recorded
   per-backend divergence budget the precision plan defines (Phase 6 there), measured
   against vendored libcint at 1e-12 on the five lists, with the number written down.

Exit: ROCm rows in the artifact for all five lists with the divergence recorded; the 2e
kernel reports its shared-memory variant through `SharedMemoryMetrics`.

### M2 — Streaming consumer surface

Files: `crates/cintx-rs/src/api.rs`.

1. `QuartetBatchRequest::evaluate_into(&mut [f64])` — caller-owned output, no second copy.
2. `QuartetBatchRequest::for_each_chunk(|chunk: QuartetBatchChunk<'_>| …)` — the M1 chunks
   handed to the caller as they complete, so the peak host footprint is one chunk regardless
   of molecule size. This is the surface a direct-SCF J/K build needs; the J/K build itself is
   out of scope.
3. Both routed through the manifest lock and `xtask manifest-audit` like every public entry.

Exit: a facade test that evaluates SO2/TZVP through `for_each_chunk` under a 32 MiB limit
and reproduces the full-list values bit for bit.

### S5 — JIT program stability and cross-process cache

Files: `crates/cintx-cubecl/src/plane.rs` (`per_unit_width`), `tuning.rs`
(`LaunchGeometryKey` anchoring), `cubecl.toml`.

1. Quantize the per-unit width and cooperative cube count to the same power-of-two anchors
   the tuner's key already uses, so different molecule sizes with the same signature set hit
   the same compiled program (today CH4 after H2O recompiles for 0.53 s).
2. Measure whether `[compilation] cache = "target"` survives a process restart on the CPU
   and ROCm runtimes; record cold-start-with-cache as a third column beside cold and warm.

Exit: second-process cold start for a def2 basis recorded; the prewarm's `signatures` count
equals the number of programs actually compiled across a benchmark run.

### S6 — Batched Schwarz table

Files: `crates/cintx-driver/src/screening.rs`, `crates/cintx-rs/src/api.rs`.

Today `build_schwarz_table` evaluates each `(ij|ij)` diagonal through a per-pair
`DiagonalEvaluator`, and the benchmark builds it from the vendor. A production caller has no
vendor, so add a batched builder: one `evaluate_2e_quartet_batch` over the diagonal
quartets, then the same block-maximum rule. Expose it on the safe API so the screened list a
caller runs is the one the benchmark measured.

Exit: `zero_tolerance_screening_is_the_identity` extended to the batched builder; the
benchmark records `schwarz_build_ms` per engine.

### S7 — Carried from the previous plan

D3.2 (tuning wiring for derivative/σ launchers) and D3.3 (vectorization factor) are
unchanged in status. D3.3 is deliberately sequenced **after S2**: a vectorized contraction
loop over a global-memory accumulator would vectorize the read-modify-write, not the work.

## 5. Ordering and dependencies

```text
M6 (observability) ──┬─→ S1 (pair table + cutoff) ─→ S2 (private accumulate) ─→ S3 (ROCm coop + shared G)
                     │                                       │
                     ├─→ M1 (bounded memory) ─┬─→ M4 (reuse) ├─→ S7/D3.3 (vectorization)
                     │                        └─→ S4 (async pipeline)
                     ├─→ M3 (device c2s) ─────────→ M2 (streaming surface)
                     └─→ S5, S6 (independent, any time)
```

S1 and M1 can proceed in parallel after M6. S2 touches the same loop as S1 and lands after
it. M3 needs M1's per-chunk release to realize its host-memory win. S3 needs a ROCm runner
and S2's accumulator.

## 6. Risks

| Risk | Control |
|---|---|
| The pair cutoff changes results a downstream test pinned bit-for-bit | S1.3 keeps a disable switch under which the kernel is bit-identical to today; the default's gate is vendor agreement not growing, on all five lists. |
| A memory plan that is wrong in either direction (refuses fitting work, or admits work that OOMs) | Plan bytes come from the same `g_slab_stride`, `out_len` and table lengths the dispatch uses; a test asserts plan == `memory_usage()` delta within a stated slack. |
| Chunking reorders arithmetic | It cannot: a chunk is a quartet range, each quartet's arithmetic is self-contained. The bit-identity test in M1.exit is the proof. |
| Device transform drifts from the host transform | Bit-identity gate, host transform kept as reference and fallback. |
| Cooperative reduction order diverges from libcint at high `l` | S3 measures against the vendor and records the divergence; it is a GPU-only path and the CPU result is unchanged. |
| Async submission masks a device error until the batch end | Every readback is checked; a failed chunk fails the batch with a typed error and no partial `values` (M1.4). |
| CPU-backend "device" allocations are host RAM, so the two budgets double-count | M1.1 uses one budget when the backend reports unified memory; the artifact says which. |
| Tuning noise repeats the Phase 6 no-win | Unchanged: CPU default stays `off`; GPU claims only from device timestamps. |

## 7. Verification

Tests to add (all under `crates/cintx-oracle/tests/`, CPU-gated unless named).
**What actually landed is in §10.1**; this list is the plan as written, kept so
the difference is visible rather than quietly edited away.

- `def2_pair_cutoff_parity` — S1. **Landed**, with the gate restated: the loop
  reorder means bit-identity against the pre-S1 kernel is neither available nor
  the right thing to ask for, so the gate is that unscreened dispatches every
  primitive quartet and that screening never worsens vendor agreement.
- `def2_batch_memory_plan` — M1. **Landed**, minus the fault-injecting allocator:
  `DeviceOutOfMemory` needs a backend that can be made to fail on demand, and
  neither available backend offers one.
- `def2_accumulator_ab` — S2, not foreseen here. **Landed**: bit-identity between
  the two accumulation settings, and the interleaved in-process timing A/B.
- `def2_batched_schwarz` — S6, not foreseen here. **Landed**.
- `def2_device_c2s_parity` — M3. **Not landed**; M3 is not taken (§10.4).
- `def2_streaming_facade` — M2. **Landed**.
- `def2_rocm_cooperative_parity` — S3. **Not landed** as its own file; the
  cooperative arm is covered by the existing `def2_batch_rocm_parity` and
  `def2_rocm_extended_and_tuning`, both re-run green after S1 and S2 (§10.6).
- `def2_throughput_benchmark` — every case gains the memory block. A/B modes:
  `CINTX_2E_ACCUMULATE=global` and `CINTX_2E_PIPELINE=async` landed;
  `CINTX_2E_CHUNK_MIB` replaced the planned transform switch, since the
  memory/speed trade turned out to be chunk size rather than transform location.

Commands (unchanged form):

```text
CINTX_ORACLE_BUILD_VENDOR=1 cargo test --release -p cintx-oracle --features cpu,extended-device-rys \
  --test def2_throughput_benchmark -- --ignored --nocapture --test-threads=1
CINTX_ORACLE_BUILD_VENDOR=1 CINTX_ROCM_ORACLE=1 cargo test --release -p cintx-oracle \
  --features cpu,rocm,extended-device-rys --test def2_rocm_extended_and_tuning -- --ignored --nocapture
```

Artifacts, under `/tmp/cintx_artifacts` and `/mnt/data`: `cintx_def2_throughput.json`
(schema 2, memory + primitive-work blocks), `cintx_cubecl_memory_report.json` (def2 rows),
`cintx_cubecl_autotune.json` (unchanged), and a CHANGELOG entry per workstream in the
established evidence-first style.

## 8. Explicitly out of scope

- **D1.3**, the `extended-device-rys` default flip — still blocked on making the precision
  budget measure the vendor (`compare.rs` fixture loop); nothing here changes that.
- **wgpu f64** — not a target.
- **`int3c1e_p2`** evaluating plain `int3c1e` — recorded, not in the def2 path.
- **A whole-manifest high-angular-momentum sampler** — the right next project after the
  previous plan, but a verification project, not a def2 optimization.
- **Direct-SCF J/K on device** — M2 provides the streaming surface it would consume; the
  contraction itself is a consumer's concern.
- **Packing the quartet table** below 6 × `u32` — 4.6 MiB on the largest list; not worth an
  encoding until M1's plan shows it on the critical path.

## 9. Facts this plan rests on, and how they were established

- Time and launch rows: `artifacts/cintx_def2_throughput.json` (2026-09-02 run).
- Primitive-work shares, `nprim^4` maxima and `max_nctr_product == 1`: the `bucket_rows`
  of the same artifact, aggregated by `nroots`.
- Kernel behaviour (§2.3): `two_electron.rs` lines 794–1608 read on 2026-09-03.
- Memory behaviour (§2.4): `evaluate_2e_batch_inner` (7221–7470), `run_2e_batches`
  (2037–2125), `TwoEGroupDispatch::launch` (2175–2219), `two_e_cube_count` (1836),
  `per_unit_cube_dim` (1745), `g_slab_stride` (1764), `MAX_BATCH_SCRATCH_BYTES` (1773).
- Output footprints: computed from `crates/cintx-basis/data/def2-{svp,tzvp}.nwchem` shell
  compositions over the 8-fold canonical list; these are unscreened upper bounds.
- libcint's cutoff: `libcint-master/src/optimizer.c:288–341` (`CINTset_pairdata`) and
  `src/cint2e.c:87–237` (`CINT2e_loop_nopt`, the path the vendor oracle takes).
- CubeCL 0.10 client surface used by M1/M4/S4: `memory_usage`, `memory_cleanup`,
  `read_async`, `sync`, `flush` in `cubecl-runtime-0.10.0/src/client.rs`;
  `MemoryConfiguration::{SubSlices, ExclusivePages, Custom}` in `memory_management/mod.rs`.

## 10. Execution record (2026-09-03)

Everything below is a measurement or a landed change, with the test or artifact
that carries it. Where a workstream was not done, it says so and why.

### 10.1 What landed

| Item | Status | Evidence |
|---|---|---|
| M6 memory + primitive-work observability | done | `BatchExecutionStats` memory fields; `crate::memory_probe`; artifact schema `cintx_def2_throughput/2`; def2 rows in `cintx_cubecl_memory_report` schema 2 |
| S1 pair table + libcint `expcutoff` | done | `crates/cintx-cubecl/src/kernels/pair_table.rs`; `def2_pair_cutoff_parity` (4 tests) |
| S2 private accumulation | done, effect mostly below this host's resolution | `def2_accumulator_ab`; §10.5 |
| M1 bounded-memory chunked execution | done | `def2_batch_memory_plan` (4 tests); `TwoEBatchOptions::memory_limit_bytes` |
| M4 scratch and table reuse | done | one shared G slab and one Rys-table upload per run; allocation counts in the artifact |
| M3 device-side cart→sph | done | `crates/cintx-cubecl/src/kernels/c2s_device.rs`; `def2_device_c2s_parity` |
| S4 asynchronous pipeline | done, measured no-win on CPU | `CINTX_2E_PIPELINE=async`; §10.3 |
| S3 cooperative G / shared memory | integrated, **disabled by a backend defect** | §10.6; `shared_memory::tests::shared_memory_through_a_slice_round_trips` |
| M2 streaming consumer surface | done | `stream_2e_quartet_batch`, `QuartetBatchRequest::{for_each_chunk, evaluate_into}`; `def2_streaming_facade` (4 tests) |
| S5 JIT program stability | done | power-of-two launch widths; prewarm 0.53 s → 0.0025 s, §10.2 |
| S6 batched Schwarz table | done | `schwarz_bounds`, `SchwarzTable::from_pair_values`; `def2_batched_schwarz` |
| S7 carried items (D3.2, D3.3) | unchanged | still open, as the previous plan left them |

### 10.2 The numbers

**Vendor agreement (G1).** S1 reorders the primitive loops onto libcint's own
nesting and adopts its pair-data association. Agreement improved everywhere and
regressed nowhere:

| workload | before | after |
|---|---|---|
| H2O / def2-SVP | 3.331e-15 | 2.665e-15 |
| CH4 / def2-SVP | 2.665e-15 | 2.442e-15 |
| SO2 / def2-SVP | 5.773e-15 | 5.329e-15 |
| H2O / def2-TZVP | 4.441e-15 | 4.441e-15 |
| SO2 / def2-TZVP | 5.918e-14 | **3.245e-14** |

`mismatched_elements` stays 0 on every workload, and ROCm agrees with the vendor
at 2.554e-15 on H2O/def2-SVP — marginally better than the CPU backend.

**The cutoff (S1).** Screened against unscreened, in one process, against the
same vendor reference:

| workload | primitive quartets | dropped | vendor diff, unscreened | screened |
|---|---|---|---|---|
| H2O / def2-SVP | 38 111 | 0.0% | 2.665e-15 | 2.665e-15 |
| CH4 / def2-SVP | 154 846 | 2.2% | 2.442e-15 | 2.442e-15 |
| SO2 / def2-SVP | 458 741 | 26.3% | 5.329e-15 | 5.329e-15 |
| H2O / def2-TZVP | 156 940 | 4.7% | 4.441e-15 | 4.441e-15 |

The two settings differ from each other by at most 6.7e-29 and from the vendor
by identical amounts. libcint's threshold discards terms that are genuinely
below the noise floor, and up to a quarter of the arithmetic with them.

**Memory (G3).** SO2/def2-TZVP, the heaviest workload, spherical output 99.8 MiB:

| mode | Cartesian peak | host peak | chunks | dispatches |
|---|---|---|---|---|
| default (unbounded) | 177.9 MiB | 2.78x output | 1 | 24 |
| `memory_limit_bytes` = output + 40 MiB | 20.0 MiB | **1.20x** output | 9 | 162 |

Device-side, M4 took planned allocations from 144 to 98 on the same workload and
made the G-tensor scratch one allocation per run rather than one per dispatch.

**Speed (G2).** Ratios against vendored libcint, both engines in the same
process on the same screened list, best of 9, values compared before timing, 0
mismatched elements throughout.

| workload | before this plan | recorded artifact | range across six runs |
|---|---|---|---|
| H2O / def2-SVP | 1.05x | 1.16x | 0.98 – 1.28x |
| CH4 / def2-SVP | 1.47x | 1.80x | 1.37 – 1.86x |
| SO2 / def2-SVP | 1.50x | 1.18x | 1.18 – 1.68x |
| H2O / def2-TZVP | 1.06x | 1.61x | 0.93 – 1.61x |
| SO2 / def2-TZVP | 1.40x | 1.28x | 1.20 – 1.54x |

**G2 cannot be resolved on this host, and that is the finding.** The gate asked
for ≥1.5x on both SO2 rows. SO2/def2-SVP produced 1.68x in one run and 1.18x in
another, from the same binary on the same work list; SO2/def2-TZVP ranged 1.20x
to 1.54x. The spread is three to four times the improvement the gate was
written to detect, so no figure in the middle column should be quoted on its
own — including the ones that look like a pass.

What *is* solid, because each was measured by switching a setting inside one
process rather than by comparing processes:

- **S1's cutoff drops 2.2–26.3% of primitive quartets at identical vendor
  agreement** — the screened and unscreened runs differ from each other by at
  most 6.7e-29.
- **S1 improves vendor agreement**, including 5.918e-14 → 3.245e-14 on
  SO2/def2-TZVP.
- **S2 is worth 3–6% on CH4/def2-SVP** and nothing distinguishable elsewhere
  (§10.5), bit-identical throughout.
- **S5 turns a 0.53 s recompilation into a 0.0025 s cache hit.**
- **A 32 MiB chunk ceiling costs 21% of wall time and saves 140 MiB** (§10.2,
  memory).

Those are the claims this work can support. A whole-workload speed factor is not
one of them on this machine.

**JIT (S5).** Quantizing the per-unit launch width to a power of two makes
molecules that share a signature set share compiled programs. CH4/def2-SVP
following H2O/def2-SVP in one process: **0.53 s → 0.0025 s**, a full cache hit.

### 10.3 S4: measured, and not worth turning on here

`CINTX_2E_PIPELINE=async` keeps one dispatch in flight while its predecessor is
read back and transformed. On the CPU backend, over the full def2 scope, it
moved SO2/def2-TZVP from 0.27404 s to 0.27162 s — 0.9%, inside the run-to-run
noise — and made two smaller workloads slower. It stays off by default, and off
unconditionally when a memory budget is set, because the overlap keeps two
groups' output buffers alive and that is exactly what a budget is asking not to
happen.

This is the result the plan predicted: on the CubeCL CPU runtime the "device" is
the same cores the transform runs on, so there is nothing to overlap with. The
switch is left in place because on a backend with a real transfer it is a
different measurement.

### 10.4 M3: landed, bit-identical, and honest about where it pays

`crates/cintx-cubecl/src/kernels/c2s_device.rs` contracts each Cartesian block
against the `c2s` tables on the device and scatters it into the chunk's
spherical output, so the readback *is* the caller's answer. One quartet per work
item, grid-stride, no barriers — the transform is embarrassingly parallel and has
none of the shared-recurrence structure that makes the 2e kernel cooperative.
Selected by `CINTX_2E_TRANSFORM=device`, off by default.

**The gate is bit-identity**, not a tolerance: the transform reorders nothing, so
a device implementation walking the same axes in the same order must produce the
same bits. `def2_device_c2s_parity` records the host result and compares the
device run against it element by element — 1 208   063 elements across three
fixtures, all identical.

Two things were worth the trouble to get right:

- **`l <= 1` axes are skipped, not applied.** `C2S_L0` and `C2S_L1` are identity
  matrices, so applying them looks harmless. It is not: an identity contraction
  still evaluates `1.0*x + 0.0*y + 0.0*z`, and `-0.0 + 0.0` is `+0.0`. A block
  holding a negative zero would come back with its sign flipped — invisible to
  every tolerance-based gate, and a change in `to_bits`.
- **A value-returning `if` on a runtime predicate is not what it looks like** in
  `#[cube]`. The first version selected its ping-pong buffer that way and
  produced 1 127 wrong elements out of 53 237 — wrong values, not wrong rounding.
  Rewritten as statement-level mutation, it was exact. The 2e kernel's existing
  `coop`/`punit` arithmetic exists for the same reason; this is a second
  instance of the same trap.

**What it buys, measured on the CPU backend:**

| workload | readback, host transform | device transform | host transform time |
|---|---|---|---|
| SO2 / def2-SVP | 6.4 MiB (1.30x output) | 4.9 MiB (1.00x) | 2.4 ms → 0 |
| H2O / def2-TZVP | 6.2 MiB (1.61x output) | 3.9 MiB (1.00x) | 2.0 ms → 0 |
| SO2 / def2-TZVP | 172.8 MiB (1.79x output) | 96.6 MiB (1.00x) | 44.7 ms → 0 |

The Cartesian intermediate never reaches the host at all, and the readback falls
to exactly the caller's output.

**Where it does not pay is the CPU backend**, and the plan predicted that: the
"device" is the same cores, so the transform moves rather than disappears, and
SO2/def2-TZVP's dispatch time rose from 246.7 ms to 280.5 ms while 44.7 ms left
the host side — roughly a wash, with the smaller workloads slightly worse. It is
off by default for that reason. The backend where a 76 MiB readback saving is a
real bus transfer is a discrete GPU, and the development host has none:
gfx1151's memory is unified. `ci/colab_t4_verification.sh` is what settles it on
one.

### 10.5 S2 landed; its effect is mostly below this host's resolution

The design in §4 does not survive contact with the shapes: an accumulator sized
to the widest Cartesian block is 10 000 f64 for an `(ff|ff)` class, which is not
private storage on any backend. What landed instead is the workable form —
a per-work-item array holding `ceil(block_len / lanes)` elements, with the
original read-modify-write as the fallback above its capacity.

The capacity is comptime **per decomposition**, and that matters more than it
looks. A work item is an OS thread in the per-unit shape and a *lane* in the
cooperative one, so a single ceiling of 256 f64 would be 2 KB per unit on the
CPU — 32 KB across 16 units, fine — and 2 KB per lane on a GPU, which is 512 KB
of private storage for a 256-wide cube. That would spill, and cost far more
occupancy than the read-modify-write it replaced. So it is 256 slots per-unit
and 64 cooperative, which is generous either way: `cooperative_cube_dim` sizes
the cube from the block, so a lane's share stays near 1 for narrow classes and
reaches 40 for the widest `(ff|ff)` block at 256 lanes.

At 256 the per-unit ceiling covers every `nroots ≤ 3` class, which is where
93–98% of all primitive work sits.

Accumulation order is unchanged, so the two settings are **bit-identical** —
asserted element by element in `def2_accumulator_ab`, and that is what makes the
timing question separable from the correctness one.

The timing question is answered as well as this machine allows: both settings
alternated **inside one process**, interleaved repeat by repeat, best of seven,
three times over.

| workload | run 1 | run 2 | run 3 |
|---|---|---|---|
| H2O / def2-SVP | 1.008x | 0.997x | 1.044x |
| CH4 / def2-SVP | 1.036x | 1.036x | 1.059x |
| SO2 / def2-SVP | 1.032x | 1.070x | 0.962x |

Only CH4/def2-SVP reproduces a positive sign across all three runs, at 3–6%.
The other two straddle 1.0. So: **a small gain on one workload, nothing
distinguishable on the others, and no measured loss anywhere.** It is kept on by
default because it is bit-identical and because the pattern it removes — a
read-modify-write through a kernel-argument pointer, once per primitive quartet
per element — is a much larger effect on a backend where `cart_out` is device
memory rather than the same cache the kernel is already running out of.

An earlier cross-process comparison of the same two settings produced a 14% win
on one workload and a 41% loss on another, from the same pair of binaries. That
is what §10.7 is about, and why the A/B is a test rather than two benchmark runs.

### 10.6 S3: integrated, then disabled by a backend defect

The integration is small and, on paper, right. `Array` and `SharedMemory` both
implement `ListMut`, so both yield a `Slice<F, ReadWrite>`; the kernel selects
between them at JIT time and the ~700 lines of recurrences below bind whichever
they were given, written once. The host enables it only for the cooperative
decomposition, only when the dispatch's widest class fits `SHARED_G_SLOTS`
(6 144 f64 = 48 KiB, which covers every def2-SVP class and 99.7% of
SO2/def2-TZVP's quartets), and only when the backend reports room.

**On ROCm it produces wrong values, silently.** Bisected on gfx1151, with every
step re-run under a renamed kernel — see §10.6.1, without which the bisect
reaches the wrong answer:

- **Allocation is innocent.** An unconditional, never-read 48 KiB `SharedMemory`
  declaration, everything else global, leaves the kernel exactly correct.
- **Use is the trigger, and one element is enough.** Unit 0 writes a single
  shared element; one `sync_cube()`; every unit reads it back and multiplies an
  exactly-1.0 factor by it, with the G tensor still entirely in global memory.
  That corrupts 15 598 of 53 237 elements by up to 0.9 absolute. Some work items
  read something other than what unit 0 wrote.
- Not the size: 48 KiB, 4 KiB and 512 bytes all fail, against 64 KiB reported.
- Not the tuner: fails with `CINTX_AUTOTUNE=off`.
- Not cross-lane sharing: fails at `CINTX_2E_CUBE_DIM=1`, where a cube is one
  lane and nothing is shared between lanes.
- Not a barrier in divergent control flow: no `sync_cube()` in this kernel sits
  inside a `lane == 0` region.

And *not* the primitive.
`shared_memory::tests::shared_memory_through_a_slice_round_trips` exercises the
same shape — direct and through `to_slice_mut()`, at the same 48 KiB and 256
lanes, beside a private `Array`, under a read-modify-write recurrence of the
shape the VRR/HRR has — and round-trips exactly on the same device.

So shared memory works in a small kernel and does not work in this one, which is
the largest in the project by a wide margin. That points at the compiler rather
than the code, and is what a report upstream should lead with.

Both findings are written up as a standalone defect report at
`.planning/notes/rocm-shared-memory-miscompile.md`, with the environment, the
minimal reproduction, the eliminated hypotheses and the verified commands. That
is the document to hand to anyone picking this up, or to base an upstream report
on.

### 10.6.1 The finding that came out of debugging it

**Compiled kernels are cached by signature, not by body.** Editing a `#[cube]`
kernel's body without changing its parameters or comptime settings can leave the
*previous* compiled kernel running on ROCm. `~/.cache/comgr` serves it silently,
and the test reads whatever the old code produced.

This is worth more than the S3 result itself, because it silently invalidates
GPU experiments. Three iterations of the S3 isolation probe were lost to it: the
body was rewritten twice and the device kept returning the first version's
output. It surfaced only when a version was written whose expected values
differed from what the stale kernel produced — an earlier version had "passed"
by writing and reading back the same array the stale kernel happened to write.

Two working rules follow, and both are now in the probe's own documentation:

- When changing a kernel body for an experiment, **rename the kernel** or clear
  `~/.cache/comgr`. A signature change (adding a parameter or a comptime
  setting) also suffices, which is why the ordinary S1/S2 work never hit this.
- **Initialise output buffers with a sentinel** rather than using
  `client.empty`, which hands back recycled device pages. A launch that never
  happens reads back as some earlier buffer's contents, and `sync()` and
  `read_one` both return `Ok`.

The integration is kept rather than deleted, because reproducing it is the
expensive part and it is a few lines from working once the defect is found. It
is off unless `CINTX_2E_SHARED_G=1`, and `def2_batch_rocm_parity` fails loudly
when it is set. The probe stays in the tree as the reproduction harness, so
whoever picks it up starts from "not the primitive" instead of re-deriving it.

**A second GPU vendor is the next datum.** If CUDA shows the same corruption the
defect is CubeCL-level and worth reporting upstream; if it does not, it is
AMD-specific and S3 is viable on NVIDIA today. `ci/colab_t4_verification.sh`
runs that comparison behind `CINTX_TRY_SHARED_G=1`.

What *was* confirmed on the device: the cooperative arm still works after S1 and
S2. S1 moved `sync_cube` barriers inside the restructured loops, and both cutoff
tests are cube-uniform — every lane of a cooperative cube walks the same quartet
and the same pair rows — so the barriers stay convergent. `def2_batch_rocm_parity`
and `def2_rocm_extended_and_tuning` are green, including the extended `nroots`
6–7 path at 2.997e-14 against the vendor and `int2e_sph` at 2.554e-15,
marginally better than the CPU backend.

### 10.7 What the gates say

- **G1 (parity)** — met. Every def2 gate green on CPU and ROCm; vendor agreement
  improved on four of five workloads and unchanged on the fifth.
- **G2 (speed)** — **unresolvable on this host**. Both SO2 rows straddle the
  1.5x the gate asks for across six runs of the same binary, with a spread three
  to four times the margin. §10.2 reports the ranges and lists the effects that
  *were* resolvable, each by an in-process switch.
- **G3 (memory)** — met for the bounded mode (1.20x on the heaviest workload,
  typed refusal before any allocation, scratch allocated once per run). The
  full-list clause is **not** met by default, deliberately: chunking to 32 MiB
  costs 21% of wall time on SO2/def2-TZVP to save 140 MiB, and that trade is the
  caller's to make. Both points are measured and recorded in
  `DEFAULT_CHUNK_CART_BYTES`.
- **G4 (observability)** — met. Schema `cintx_def2_throughput/2` carries a
  `memory` and a `primitive_work` block per case.
- **G5 (API)** — met. No CubeCL type crosses `cintx-rs`; the budget is an
  `ExecutionOptions` field and the streaming surface is a closure over a
  borrowed `&[f64]`.

### 10.8 A note on the benchmark's own noise

Absolute times on this host vary by up to 2x between processes for identical
work — vendored libcint's own H2O/def2-SVP figure ranged from 2.3 ms to 4.9 ms
across the runs above. Every claim here is therefore either a ratio measured
inside one process, or an A/B switched inside one process
(`CINTX_2E_CHUNK_MIB`, `CINTX_2E_PIPELINE`, screened vs unscreened residencies).
Cross-run absolute comparisons appear nowhere, and a future session should hold
to that: the machine will not support them.
