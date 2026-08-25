# Changelog

All notable changes to **cintx** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed — the host cart-to-sph transform is no longer the bottleneck (2026-08-25, Task 36-T0/T1/T2)
- **Measured first, then acted on.** New opt-in instrumentation
  (`CINTX_HOST_TRANSFORM_PROFILE=1`) splits `host_transform_ns` into allocate / c2s / scatter.
  The plan expected allocation to be a candidate; it is 0-15 %. **The c2s arithmetic is
  68-81 %** on every workload measured. It is opt-in because three clock reads per 27-element
  block would otherwise make `host_transform_ns` a probe artifact.
- **Identity axes are no longer transformed.** `C2S_L0` and `C2S_L1` are identity matrices, so
  an `l <= 1` axis is a copy dressed as a matrix product. `cart_to_sph_2e_into` already skipped
  them; the 1-, 2- and 3-index transforms did not, and on a def2-SVP work list that is most
  axes. All four now route through one `c2s_apply` axis-plan driver. **The serial transform is
  6x cheaper on the 3-index families.**
- **Allocations hoisted**: `cart_to_sph_{1e,2c2e,3c1e,3c2e}` gained `_into` forms taking
  caller-owned output and scratch, and all six batch transform loops allocate once per run
  rather than once per contraction block.
- **The transform runs across threads** (`rayon`), `unsafe`-free: `offsets` is a running total
  in the caller's order, so repeated `split_at_mut` hands each tuple a disjoint `&mut [f64]`
  and the borrow checker proves they do not alias. Each output element is produced by exactly
  one tuple — the transform writes, it never accumulates — so **the split reorders no summation
  and bit-identity holds by construction**, not by tolerance. Every element-by-element parity
  gate is unchanged and green. `CINTX_HOST_TRANSFORM_THREADS` pins the worker count.
- Below a measured threshold (4 096 tuples) the transform stays serial: after the change those
  lists take a fraction of a millisecond and the fan-out costs more than it saves.
- **Against vendored libcint 6.1.3, CPU backend**: `int2e` CH4/def2-SVP 1.42x -> **1.88x
  faster**; `int2e` H2O 1.03-1.15x faster; `int3c2e_ip1` 1.59x slower -> **1.28x faster**;
  RI-J def2/J ~2.3x -> 1.35x slower; RI-J def2/JK ~1.9x -> 1.23x slower.

### Added — pair and triple batch surfaces on the safe API (2026-08-25, Task 35-F2)
- `PairBatchRequest` and `TripleBatchRequest` join `QuartetBatchRequest`, returning a shared
  `ShellListBatchOutput`. Before this a safe-API consumer could batch `int2e` and nothing else
  without depending on the backend crate, which the project's API ordering says it should not
  have to.
- Scope is **symbol-exact**, resolved through the compiled manifest and refused before any
  device work: `int1e_{ovlp,kin,nuc}_sph`, `int1e_ip{ovlp,kin,nuc}_sph`, `int2c2e_sph` for
  pairs; `int3c2e_sph` and `int3c2e_ip{1,2}_sph` for triples. Symbol-exact rather than
  family-wide because `int1e_ipovlp_sph` and `int1e_ovlp_sph` share a family and are different
  integrals.
- Five new `CubeClExecutor` methods carry backend resolution and the f64 capability check, so
  no CubeCL type reaches the facade.

### Added — device-resident basis for every batched family (2026-08-25, Task 34-C2)
- `evaluate_{2c2e_pair,1e_pair,1e_deriv_pair}_batch_resident`. `int2e` and `int3c2e` already
  had theirs.
- `OneEFlatBasis` and `TwoC2eFlatBasis` are **removed**: they were a third and fourth spelling
  of the four buffers `ResidentBasis` already holds, and every family now uploads through one
  path.
- Gated two-sidedly per family: `basis_upload_bytes` is the full upload on the first call and
  **0** on every later one, transfer strictly decreases, **and** every value is bit-identical
  to the throwaway-residency path. Either half alone is worthless.

### Added — primitive screening for `int3c2e` and the 1e nuclear arm (2026-08-25, Task 34-D2)
- `evaluate_3c2e_triple_batch_with` / `evaluate_1e_pair_batch_with` (and `_resident_with`) take
  `BatchOptions` — the family-neutral alias for `TwoEBatchOptions`.
- The 1e nuclear `fac1` carries `-Z_C` and is **negative**, so the test is on its magnitude;
  the branch is on values uniform across the cube, so the `sync_cube` barriers inside it are
  still reached by every lane or by none.
- Landed in the same commit as its gate: at `primitive_tolerance == 0` the only primitives
  dropped are those whose `fac1` underflowed to exactly zero, so the result is **bit-identical**.
  A screening bug reads as a speed-up; that identity is the only thing between "faster" and
  "wrong".

### Changed — the last per-tuple families are batched (2026-08-25, Task 35-D wave 5)
- **`int3c1e` and `int4c1e`**: the two genuine scalar families that were never batched. Both
  already launched once per *contraction* tuple with the coefficient columns sliced host-side,
  so a work row here is a *(shell tuple, contraction tuple)* pair — reproducing that arithmetic
  exactly while collapsing `nctr_i * nctr_j * nctr_k` (`* nctr_l`) launches into one.
  `int3c1e` also gained `evaluate_3c1e_triple_batch{,_resident}`.
- **`center_4c1e` needed two independent slab strides** — its `[gx|gy|gz]` G-tensor and its 1D/2D
  polynomial scratch are read at unrelated offsets, unlike every other family — and its host
  loop *sums* the contraction blocks, so rows are emitted in the order that sum was performed
  in. Any other order would reassociate the additions.
- **The ten σ·p relativistic kernels**: `sigma_p`, `sigma_p_cg_sa10sp`, `sigma_p_spgsp`,
  `sa01_rys`, `spgnucsp_rys`, `spgsa01_rys`, `sigma_nuc`, `sigma_nuc_gauge`, `sigma_ov`. All
  reuse wave 3's `OneEDerivLaunchGroup` and `one_e_deriv_single_pair_group`, which is what made
  ten conversions tractable.
- **The two ECP angular kernels** now take one dispatch per shell pair instead of `nci * ncj`:
  their host precompute is already laid out contraction-tuple-major, so the kernel indexes it by
  row. The intra-cube split and accumulation order are untouched, so the byte-identity gate
  holds. Batching across *shell pairs* still needs the radial precompute batched — host work.
- **`f12_cart_contraction_kernel` is deliberately not converted**, and the reasoning is recorded
  at the call site: it is launched once per *primitive quartet* with a host-computed `g`, so
  collapsing the launches would mean materializing every `nprim^4` G tensor first while leaving
  the dominant arithmetic on the host. The conversion worth doing is porting `fill_g_tensor_f12`
  to the device.
- **One real bug, caught by the existing gate.** Rewriting a launcher means rewriting its
  comptime `match`, and `sigma_ov`'s four-family dispatch lost a case — `spsp` launched as
  `srsr`. It surfaced as 24 mismatched elements in `int1e_spsp_spinor` at `nctr > 1` against
  vendored libcint, on the full-suite run rather than the targeted one. Every converted
  launcher's dispatch was then re-checked arm-by-arm against the original.

### Changed — 13 more per-tuple families are batched (2026-08-25, Task 35-D waves 3 and 4)
- **Wave 3, the 1e gradient/Hessian set**: `int1e_ipovlpip`, `int1e_ipkinip`, `int1e_ipnucip`,
  `int1e_ipipovlp`, `int1e_ipipkin`, `int1e_ipipnuc`/`int1e_ipiprinv`.
- **Wave 4, the 1e special families**: `int1e_rinv`, `int1e_drinv`, `int1e_p4`,
  `int1e_irp`/`int1e_ipipr`, the eight moment operators, the five GIAO overlap-engine families
  and the six GIAO nuclear-engine ones.
- Same acceptance bar as waves 1-2: the per-tuple entry point is a **one-tuple batch through
  the same kernel**, so every existing parity test covers the batched code. Each conversion
  also collapsed a five-arm backend `match` into one dispatcher.
- `int1e_irp`, the moments and both GIAO engines take `drj = rj - origin`, which is **per pair**
  — the base families measure from a common origin and the `_origj` variants from `rj` itself.
  The host resolves that choice and the batch carries the resolved vector.
- The `#[cube]` recurrence helpers (`d_i_1e_into`, `d_j_1e_into`, `rcj_1e_into` and the five
  `*_flat` tensor helpers) gained a `gbase` parameter, so a slot's slab base threads through
  them rather than being patched at every call site.

### Added — the f64:f32 arithmetic ratio on gfx1151 is measured (2026-08-25, Part 6)
- `cintx_cubecl::measure_precision_ratio` plus an opt-in oracle test: a dependent FMA chain,
  no memory traffic, both precisions through the same kernel source and launch geometry.
- **gfx1151 is ~1:10 at saturation** (f64 saturates at ~43 GFMA/s, f32 reaches ~400) — not the
  1:16 or 1:32 a consumer part is often assumed to be, nor the 1:2 of a discrete HPC card. In
  the latency-bound regime a Rys/VRR recurrence chain actually occupies, it is ~1:2.5-3.5.
- Reported as a sweep rather than a point, and with **no asserted bound**: what the ratio
  "should" be is exactly what was unknown, so an assertion would be a guess dressed as a gate.

### Added — an inline device diagonalizer (2026-08-25, groundwork for Task 33-01)
- `eigh::cint_diagonalize_dev`, factored out of `cint_diagonalize_kernel` with the kernel
  reduced to a one-line wrapper so both paths run the same code. The extended Rys path is
  confined to the host because its solvers are reachable only by *launching* them, and the
  Jacobi arm's eigensolve was the piece with no callable form at all.


### Changed — the derivative families are batched, not just batch-capable (2026-08-25, Task 35-D)
- **`int3c2e_ip1` / `int3c2e_ip2`** (RI-J gradients): 1 728 launches on the def2-SVP water
  triple list become **4** — one per Rys order rather than one per triple. **25x** faster than
  the per-triple path, and 1.5-1.6x of vendored libcint where it was ~40x slower.
- **`int1e_ipovlp` / `int1e_ipkin` / `int1e_ipnuc`** (nuclear gradients): 144 launches become
  **1**, **1** and **3**. The first two collapse to a single dispatch because, once the shape
  scalars are per-pair, nothing is left to specialize on — `op_kind` is fixed by the caller's
  operator. 25-33x faster than the per-pair path; `int1e_ipovlp` reaches **parity** with libcint.
- The per-tuple compatibility API now evaluates its one tuple as a **one-tuple launch group
  through the same kernel**, so every existing parity test covers the batched code rather than
  a second path that merely ought to agree. Results are bit-identical to what the per-tuple
  path produced, and match vendored libcint to 6.7e-16 … 1.4e-13.
- New batched surface: `evaluate_3c2e_deriv_triple_batch{,_resident}` with
  `ThreeC2eDerivFamily`, and `evaluate_1e_deriv_pair_batch` with `OneEDerivOperator`. Without
  these the kernels would be batch-capable but no caller could batch.
- Each conversion also collapsed a five-arm backend `match` — one arm per runtime, identical
  apart from the type — into one dispatcher, removing ~110 lines per family.
- All five run correctly on the ROCm cooperative path (0 mismatches vs vendored libcint). They
  are gated on vendor agreement rather than the scalar families' eps-of-block-scale bound: a
  gradient kernel builds second differences, and the cancellation there turns a 2-ULP
  difference in an intermediate into 5.8e-14 in the result while the block scale stays O(10).

### Added — def2/J and def2/JK auxiliary bases, and the RI-J work-list layout (2026-08-25)
- `StandardBasis::{Def2JFit, Def2JkFit}`, vendored from the Basis Set Exchange at the same
  software (0.12) and data (1, Turbomole 7.3) version as the three files already present, so
  all five are one consistent snapshot. Resolvable by either the literature name (`def2/J`)
  or the BSE export name (`def2-universal-jfit`).
- `StandardBasis::is_auxiliary()`. A fitting basis is not an orbital basis, and mixing them
  yields a plausible-looking calculation of the wrong thing, so the distinction is on the type
  rather than left to the name.
- `to_raw_arrays_with_auxiliary()` emits the orbital and auxiliary shells into one `bas` array,
  with `RawArrays::{orbital_shells, auxiliary_shells}` naming the ranges — the layout a
  `(mu nu | P)` list needs. It refuses an orbital basis in the auxiliary slot.
  `RawArrays` gains `n_orbital_shells` as the split point.
- `evaluate_3c2e_triple_batch_resident()` and the `ResidentBasis` alias: the flattened basis
  form is family-independent, so `int3c2e` reuses the device residency `int2e` already had
  instead of re-uploading exponents, coefficients and centres on every call (Task 34-C2).
  `ThreeC2eFlatBasis` removed — it was a second spelling of the same four buffers.
- `device_rys_ceiling`: a per-backend device Rys ceiling with a backend-generic FMA-fusion
  probe (Phase 33, task 33-05 scaffolding). The ceiling stays at 5 unless *both* the new
  `extended-device-rys` feature is on *and* the probe passed on that backend. Measured:
  `fma` fuses bit-for-bit on **both** the CPU and ROCm backends.
- `BatchExecutionStats::{launch_classes, max_g_slab_bytes}`, so the launch merge below and the
  scratch it costs are observable rather than asserted.

### Changed — one kernel dispatch per comptime signature, not per angular-momentum class (2026-08-25)
- **`int2e` (Task 35-M1).** `two_electron_scalar_kernel` has exactly three comptime
  parameters — `ibase`, `kbase`, `nroots`. Every other shape scalar was already a runtime
  value, so they moved from the launch arguments into per-class device arrays and the quartet
  row gained a class index. **69 launch classes collapse to 15 dispatches** on H2O/def2-SVP.
- **`int3c2e` / `int2c2e` / `int1e_*` (Task 35-M2).** `nroots` alone is comptime for these,
  so the merge is larger: 27→4, 9→3, and 9→**1** for overlap and kinetic, which are not Rys
  quadratures at all.
- Results are **bit-identical** to the per-tuple path and to vendored libcint throughout: each
  class indexes only the leading `3 * g_size` of a slab sized to the widest class in its
  dispatch, so a narrow class touches exactly what it did when it launched alone.
- Measured against vendored libcint (single-threaded, best of 25, three repeats):
  `int2e` on CH4/def2-SVP goes from 1.28x faster to **1.4–1.5x faster**; on H2O/def2-SVP from
  **1.43x slower to parity**. `int3c2e` 1.40 → 1.17 ms; `int1e_ovlp` 0.28–0.30 → 0.11–0.14 ms.
  Both remaining gaps are now ~40 % *serial host* cart-to-sph, which no backend change touches.
- `cintx-rs`'s `BatchExecutionStats` now reports `bucket_count` as the angular-momentum class
  count and `chunk_count`/`kernel_launch_count` as the dispatch count. They coincided before
  this change.

### Fixed — 1e batch normalization scaled a whole dispatch buffer (2026-08-25)
- The `int1e_*` host transform applied `common_fac_sp(li) * common_fac_sp(lj)` to the entire
  class buffer. Once Task 35-M2 let several angular-momentum classes share one dispatch buffer
  that would have scaled neighbouring classes by the wrong factor. Each class now records its
  half-open span and scales only that. Caught before it shipped by the bit-identity gate.

### Fixed — `F::new(0.0)` was silently falling back to `f32` (2026-08-25)
- 608 sites across the `#[cube]` kernels relied on an untyped float literal falling back to
  `f32` because `f32: From<f64>` is not satisfied. rustc reports this as
  *"previously accepted ... will become a hard error in a future release"*. The literals were
  already `f32`, so appending the explicit suffix changes no value — only the reader's
  certainty and the code's survival of a future compiler.

### Fixed — a doubled comptime barrier guard and a vestigial oracle counter (2026-08-25)
- `one_electron_scalar_kernel` had `if comptime!(per_unit == 0) { if comptime!(per_unit == 0)
  { sync_cube(); } }`; the inner guard was redundant.
- The oracle helper/transform comparison incremented a mismatch counter at 20 sites, each
  immediately followed by `bail!`, making the final aggregate check unreachable. The function
  is fail-fast by construction and each `bail!` already names the specific disagreement.

### Added — ROCm executes the cooperative kernel decomposition for the first time (2026-08-25)
- The `per_unit == 0` topology (one tuple per cube, the cube splitting the contraction, real
  `sync_cube` barriers) was compiled for every backend and never *executed* in CI. All five
  batched families now run on gfx1151 and match vendored libcint with **0 mismatches** at the
  1e-10 oracle tolerance (max abs diff 4.4e-16 … 2.7e-14).
- CPU and ROCm results are **not** bit-identical, and the reason is not the launch topology:
  the AMD compiler contracts multiply-add pairs the CPU backend leaves separate. The measured
  divergence is 0.26–2.72 eps of the block's largest element, and on `int2e` the ROCm result is
  *closer* to vendored libcint than the CPU one. The gate is an eps-of-scale bound accordingly.

### Changed — clippy is clean under `-D warnings` (2026-08-25)
- `cargo clippy --workspace --all-targets -- -D warnings` passes, from ~2 078 unique warnings,
  landed by lint rather than by sweep and never with `--fix` on a transcribed-table module.
  Every `#[allow]` carries a reason. The `rys_wheeler` host long-double/`Dd` chain is recorded
  as a superseded but independent cross-check of the device dd kernels that replaced it, rather
  than deleted or silently allowed.

### Fixed — general contraction collapsed to one block on the device `int2c2e` and `int3c2e` paths (2026-08-24)
- **`int2c2e` and `int3c2e` returned wrong values for any shell with `nctr > 1`.**
  Both device kernels summed every contraction-coefficient product into a single scalar
  (`prim_coeff += c_i*c_k` / `c_i*c_j*c_k`) and accumulated it into one Cartesian block, rather
  than producing one block per contraction tuple. That is correct only when every `nctr == 1`.
  Measured against vendored libcint on a `nprim = 3, nctr = 2` shell: `int2c2e` off by
  **2.775e1**, `int3c2e` off by **5.407e0**.

  def2-SVP and def2-TZVP are fully segmented, so no def2 fixture could distinguish the two
  layouts — the same blind spot that hid the two host-fallback bugs below. Found by extending
  `general_contraction_device_indexing` (which already asked this of the 2e path) to `int2c2e`,
  `int3c2e` and the `int1e_*` trio; the 1e family was already correct.

  The transform side carried the matching assumption and is fixed with it: `int2c2e`'s
  representation dispatch transformed only the first block and copied it linearly to staging
  (so even the already-correct host `nroots > 5` arm lost its result), and `int3c2e`'s `swap_ij`
  transpose ran over the whole read-back buffer as one block. Both now scatter per contraction
  block into the `c*n + m` AO grid, and `int3c2e` swaps the contraction index alongside the shell
  indices. Spinor general contraction for these two families is not wired through the spinor
  transforms and now returns a typed `UnsupportedApi` instead of silently transforming block zero.

### Added — batched shell-pair and shell-triple evaluation for `int1e_*`, `int2c2e`, `int3c2e` (Phase 35, 2026-08-24)
- `evaluate_1e_pair_batch` (+ `OneEOperator`, `BatchAtom`), `evaluate_2c2e_pair_batch` and
  `evaluate_3c2e_triple_batch` evaluate a whole work list in one dispatch **per launch class**
  instead of one per tuple, reading a flattened basis through an index table exactly as the 2e
  batch path does. The single-tuple entry points were rewritten as one-tuple batches, so every
  pre-existing parity test now covers the batched kernel.

  Measured on H2O (best of 9, CubeCL CPU backend), batched vs the per-tuple CubeCL route:
  `int1e_ovlp` 12.9x (def2-SVP) / 24.2x (def2-TZVP), `int1e_kin` 12.8x, `int1e_nuc` 16.6x,
  `int2c2e` 18.3x, `int3c2e` **26.6x** — the last bringing `int3c2e` to within 2.4x of libcint.
  Every value is bit-identical to the per-tuple path and matches vendored libcint.

  `int3c2e`'s `swap_ij` canonicalization is resolved once per class rather than per triple, since
  it depends only on the class's `(li, lj)`.

### Added — public shell-quartet batch surface (Task 34-F, 2026-08-24)
- `QuartetBatchRequest` / `QuartetBatchOutput` and `evaluate_shell_quartets{,_in}` submit an entire
  quartet work list through the safe facade, returning AO blocks plus `BatchExecutionStats` so a
  claimed speed-up stays auditable. No CubeCL type appears in the surface. Scope-gated to
  `int2e_sph` + `Spheric` + `F64` before any device work: accepting `int2e_ip1_sph` here would
  return undifferentiated integrals under a derivative operator's name.

### Added — device-resident 2e basis (Task 34-C) and primitive-quartet screening (Task 34-D) (2026-08-24)
- `ResidentTwoEBasis` uploads a flattened basis once and keeps it on the device across calls, so a
  repeated Fock build transfers only its quartet tables. It is bound to the backend arm it was
  uploaded on and refuses a mismatched one rather than indexing another device's memory.
  `BatchExecutionStats::basis_upload_bytes` makes the amortization observable rather than asserted.
- `TwoEBatchOptions::primitive_tolerance` skips primitive quartets whose G-tensor scale factor
  `sqrt(a0/a1^3) * common_factor * exp(-mu_ij R_ij^2) * exp(-mu_kl R_kl^2)` does not exceed it.
  The default `0.0` is **exact**: it drops only quartets whose factor underflowed to exactly zero,
  and `primitive_screening_at_zero_is_the_identity` pins that bit-for-bit. Screening on the full
  scale factor rather than the exponential prefactor alone matters because `sqrt(a0/a1^3)` is not
  O(1) — for diffuse primitives it is large.

### Changed — one quartet per unit on the CubeCL CPU backend (2026-08-24)
- The batched kernels carry a comptime `per_unit` flag selecting between one tuple per **cube**
  with the cube cooperating on it (the GPU shape) and one tuple per **unit** with the barriers
  *comptime-removed* (the CPU shape, where a unit is an OS thread and `cube_count` lowers to a
  sequential loop). The barriers must be removed rather than skipped: units walk different tuples,
  so their trip counts differ and any barrier inside the loop is divergent.
- Per-slot G slabs are padded to a 64-byte cache line, and the per-unit walk is blocked rather than
  interleaved — without either, concurrent units share cache lines on the G tensor and on
  neighbouring `cart_out` blocks.
- `plane::per_unit_width` sizes the unit count by the work available, not by
  `available_parallelism` alone. Waking a unit costs ~2 us; H2O/def2-SVP `int2c2e` classes are
  ~16 pairs each and 16 units was **3x slower** than 4.

### Changed — allocation-free, identity-skipping `cart_to_sph_2e` (2026-08-24)
- The 2e cart-to-sph transform allocated four `Vec`s per call and ran all four axis transforms
  unconditionally. `C2S_L0` and `C2S_L1` are identity matrices, so an s/p axis is a copy — and a
  def2-SVP work list is mostly s and p. `cart_to_sph_2e_into` skips identity axes and ping-pongs
  through caller-owned buffers; `cart_to_sph_2e` is now a thin wrapper. Host transform time fell
  2.9x on H2O/def2-SVP and 2.5x on CH4, which was a third of the batched run's wall-clock.
- `BatchExecutionStats` gained `dispatch_ns` and `host_transform_ns` so the split between backend
  dispatch and serial host work stays attributable instead of being measured once and discarded.

### Fixed — `cintx-cubecl` did not compile without the `cpu` feature (2026-08-24)
- `math/rys_wheeler.rs` and `math/eigh.rs` named `cubecl::cpu::CpuRuntime` with no
  `cfg(feature = "cpu")` gate, so `--no-default-features --features wgpu` (or `cuda`, `rocm`,
  `metal`) failed with 13 errors. Those are **host** helper routines feeding the `nroots > 5`
  fallback, not device kernels, so they must not force `cubecl/cpu` into a GPU-only build. The
  device-launching solvers are now cfg-gated and the identical pure-host solvers stand in when the
  `cpu` feature is off.
- `cintx-compat` gained `wgpu`/`cuda`/`metal` forwards (it had only `cpu` and `rocm`) and
  `cintx-driver` gained all four, so a backend profile is selectable end to end.
- `ci/feature-matrix.yml` gained a `backend_profile_matrix` job that actually `cargo check`s
  `cintx-cubecl`, `cintx-compat` and `cintx-driver` under each backend-only profile. The previous
  matrix legs differed only in which artifacts they uploaded, so this class of defect was invisible
  to CI.

### Fixed — clippy deny-level findings across the workspace (2026-08-24)
- `cargo clippy --workspace --all-targets` is error-free (was 75-82 findings). `approx_constant` on
  transcribed libcint constants and `erasing_op` on the deliberate `0 * stride` column alignment in
  kernel bodies are now allowed at the narrowest scope with the reason stated; the two
  `nonminimal_bool` findings were real redundancy in `rys_wheeler.rs`, where the vendor's nested
  `if (b[i] < 1e-14) { if (b[i] < 0.) ... }` had been flattened to a condition whose first half
  could not fail — now flattened to the branch that actually decides. Warnings (~2 674) are
  unchanged and remain a separate task.

### Fixed — missing Gaussian-product prefactor on the `int2e` host Rys fallback (2026-08-24)
- **`int2e` was wrong by a constant factor for multi-centre quartets above the device Rys ceiling.**
  `fill_g_tensor_2e` does not compute the bra/ket overlap exponentials; its caller folds
  `exp(-a_i a_j |R_i-R_j|^2 / (a_i+a_j)) * exp(-a_k a_l |R_k-R_l|^2 / (a_k+a_l))` into `fac_env` —
  which the device kernel does (as `fac1`), which `launch_two_electron_hess2e` does, and which
  every `int3c2e` caller does. The `nroots > MAX_DEVICE_NROOTS` arm of
  `launch_two_electron_typed` passed the bare `common_factor` instead, dropping both.

  Every *single-centre* quartet has both factors equal to 1, so the omission was invisible to the
  fixtures that reached this arm. Found by the def2-TZVP class sweep: `(p,f|f,f)` and
  `(f,f|p,f)` — the only Rys-6 classes in H2O/def2-TZVP that put a shell on a **different atom**
  from the rest — came out a uniform **5.37x** too large (max abs diff 5.84e-1), while the
  O-centred representatives of the same classes were correct to 4e-16.
  `def2_tzvp_host_rys_diagnostic` records the structure that identified it (a constant
  `actual/expected` ratio across the whole block, not a permutation).

### Fixed — transposed contraction coefficients on the host Rys fallback (2026-08-24)
- **`int2e`, `int1e_nuc` and `int2c2e` read the wrong contraction coefficient for
  general-contraction shells whenever the quartet's Rys order exceeded the device ceiling.**
  `Shell::coefficients` is primitive-major (`coeff[p * nctr + c]`, WR-03 in `cintx_compat::raw`)
  and the device kernels read it that way; the three host fallback loops taken when
  `nroots > MAX_DEVICE_NROOTS` indexed it contraction-major (`coeff[c * nprim + p]`) instead.

  The two layouts coincide whenever `nctr == 1` or `nprim == 1`, which is every shell in the
  fixtures the fallback had been exercised with — so the bug was invisible until a real basis
  put a *contracted* shell in a class above the ceiling. Found by the def2-TZVP class sweep:
  `(p,f|f,f)` (Rys order 6) was the only H2O/def2-TZVP class combining a contracted shell with
  `nroots > 5`, and it was wrong by up to **5.84e-1**.

### Changed — backend-aware launch topology: the CubeCL CPU runtime gets a single-unit cube (2026-08-24)
- **Every cooperative kernel was launching a 256-unit cube on the CPU backend, and that was the
  dominant cost of the entire compute path.** Read from `cubecl-cpu-0.10.0`: `execute_data`
  spawns **one OS thread per cube unit** (growing the worker pool past `available_parallelism`),
  `sync_cube` is a **global spin-wait barrier** across every unit, and `cube_count` lowers to a
  sequential `scf.for` *inside* each unit — so on the CPU runtime the cube dimension is an
  OS-thread count, not a vector width, and the grid is not a parallelism axis at all.

  The 2e kernel calls `sync_cube()` twice per **primitive** quartet, so a def2-SVP `(s,s|s,s)`
  quartet (7^4 = 2401 primitive quartets) paid 4802 barriers, each rendezvousing 256 threads
  16x-oversubscribed on 16 cores.

  `plane::cooperative_cube_dim::<R>()` and `plane::backend_plane_cube_dim::<R>()` now return a
  single unit on the CPU runtime and keep the plane-aligned cube on GPU runtimes; all 59 launch
  sites across the kernel crate use them. Kernels are unchanged: every one of them partitions
  with `UNIT_POS == 0` guards, `idx % CUBE_DIM == UNIT_POS` selection, or
  `i = UNIT_POS; i += CUBE_DIM` stride loops, all of which cover the full index space at any cube
  dimension. **Cost changed; results did not.**

  Measured on H2O/def2-SVP (`artifacts/34-A0_cube_dim_ab.md`): the 236-quartet whole-workload
  benchmark went from **125.4 s to 0.0086 s** (~530 ms -> ~36 us per quartet, **14 600x**), and
  the gap to libcint 6.1.3 from **390 000x to 58x**. The class-complete parity sweep stayed at
  **69/69 classes, 0 mismatches**, max |diff| 2.665e-15.

### Added — batched shell-quartet evaluation for `int2e` (2026-08-24)
- **`two_electron_scalar_kernel` is now batched**: one dispatch per *launch class* instead of one
  per shell quartet. The basis is flattened once (`exps`/`coeffs`/`centers` plus a per-shell
  `[exp_off, coeff_off, nprim, nctr]` table) and quartets are an index table
  (`[si, sj, sk, sl, out_off]`), so `nroots`, the HRR branch and the G-tensor extents stay
  comptime within a dispatch — which is exactly what `cintx-driver`'s bucketing guarantees.
  Each cube owns a `3 * g_size` slab and walks the list grid-stride; the Rys roots became
  kernel-local arrays (every read of them already sat inside the `UNIT_POS == 0` region that
  writes them), removing two buffers from every launch.

  The single-quartet compat path (`eval_raw`) marshals a one-element batch through the *same*
  kernel, so every existing 2e parity test covers the batched code at `n_quartets == 1`.
- **`evaluate_2e_quartet_batch`**: public batched entry point returning spherical AO blocks plus
  `BatchExecutionStats` (launch count, readback count, transfer bytes), so a claimed speed-up
  stays auditable. Gated by `def2_2e_batch_parity`, which requires the batched output to be
  **bit-identical** to the per-quartet path and to match vendored libcint, and asserts that a
  batch launches once per launch class.
- **One basis upload per run, not per class.** The backend is resolved once for a whole batched
  run rather than once per launch class, so the flattened basis is uploaded a single time and its
  handles are reused across every dispatch. `def2_2e_batch_parity` pins the reported
  `transfer_bytes` to exactly "one basis + the per-class quartet tables". Retaining the basis
  across *separate calls* in `DeviceResidentCache` is still open.

  Measured (H2O / def2-SVP, 3081 quartets, 69 launch classes): **3 081 launches -> 69**,
  **~36 us -> ~2.4 us per quartet**, and the gap to single-threaded libcint 6.1.3 from 58x to
  **2.5-3.0x**. Full numbers in `artifacts/def2_throughput_report_2026-08-24.md`.

### Fixed — device `kj2d` HRR loop bound (2026-08-24)
- **`int2e` produced wrong values on the device path for `ibase == false && kbase == true`
  quartets with `li >= 1` and `ll >= 1`.** The device `kj2d` HRR branch bounded its second
  transfer loop by `di` where libcint (`g2e.c:552`) and cintx's own host `hrr_kj2d_4d` both use
  `dk`. With `ibase == 0`, `di == nroots` while `dk == nroots * (li + 1)`, so the loop silently
  under-wrote every `i >= 1` plane of the G-tensor. On H2O/def2-SVP this broke the `(p,p,d,p)`,
  `(p,d,d,p)` and `(d,d,d,p)` classes, with errors up to **1.17e+1**.

  It escaped detection because the branch's only device test was `(s,s,p,s)` — `li == 0`, where
  `dk == di` makes the bug invisible, and `ll == 0`, where the loop never executes. Found by
  driving a full def2-SVP basis through a class-complete sweep against vendored libcint
  (66/69 classes correct before, 69/69 after). Three in-crate regression tests added at
  `(p,p,d,p)`, `(p,d,d,p)`, `(d,d,d,p)`.

### Added — def2 basis-set catalog and batched-driver foundation (2026-08-24)
- **`cintx-basis`**: def2-SVP, def2-TZVP and def2-ECP vendored from the Basis Set Exchange
  (v0.12, Turbomole 7.3 data), with an NWChem parser (orbital + ECP sections, general
  contractions, Fortran `D` exponents), libcint-exact two-stage normalization
  (`CINTgto_norm` then PySCF's contracted self-overlap renorm), `BasisSet` construction with
  ECP shells for Z >= 37, and raw `atm`/`bas`/`env` emission. Normalization is gated against
  **vendored libcint itself** — a correctly normalized contracted AO has unit self-overlap, so
  libcint must report `S_ii == 1` — plus a direct comparison against the vendor's
  `CINTgto_norm` FFI.
- **`cintx-driver`**: the host half of a batched shell-quartet pipeline — canonical 8-fold
  quartet enumeration, Cauchy-Schwarz screening (with a `tolerance == 0` identity gate),
  angular-momentum launch-class bucketing with the `g_size` formula mirrored from
  `build_2e_shape`, three-tier launch classification from the G-tensor footprint, and
  execution with auditable statistics behind a pluggable `QuartetEvaluator`.
- **`def2_throughput_benchmark`**: whole-workload cintx-vs-libcint comparison over the identical
  screened quartet list, reporting warm-up separately from steady state and refusing to print a
  speed verdict for an incorrect or incomplete run. Results in
  `artifacts/def2_throughput_report.md`.


### Added — gradient-gap Wave 5 (2026-08-22)
- **`int1e_pnucp` and `int1e_prinvp`** (cart + sph + spinor, `component_rank` 1),
  byte-identical to vendored libcint 6.1.3 at `atol=1e-12` across s/p/d shells.
  These are the X2C **base** families; Wave 3 had shipped only their derivatives, so
  `pyscf/x2c/x2c.py` — which calls `int1e_pnucp` directly to build the X2C
  Hamiltonian — was not satisfiable even though `sfx2c1e_grad.py` was.
- **`unsupported_policy` on manifest rows** (`policy`, `reason`, `owner`), carried by
  `compiled_manifest.lock.json` through the generator into `ManifestEntry`. It
  distinguishes three states the manifest previously could not tell apart:
  *unproven*, *declared-but-fail-closed-by-design*, and *no upstream oracle exists*.
  `xtask manifest-audit` reports each separately; only genuinely-unproven rows fail
  the gate, and a row claiming both `oracle_covered = true` and a policy now fails
  unconditionally.
- **`PARITY_BASELINE` in `xtask manifest-audit`** — every libcint symbol cintx does
  not yet implement, each tagged with the phase that owns closing it. The audit fails
  if a symbol is unsupported but absent from the baseline (a new omission — the defect
  class PARITY-01 exists to catch), or present but no longer unsupported (a stale
  entry). This replaces the plan's proposed default-on `CINTX_PARITY_STRICT`, which
  would have been permanently red because 43 of the 50 entries legitimately belong to
  Phases 30 and 31. `CINTX_PARITY_STRICT=1` keeps its meaning (list must be empty) and
  becomes Phase 31's exit gate.

### Fixed — gradient-gap Wave 5 (2026-08-22)
- **`int3c2e_ip1_spinor` / `int3c2e_ip2_spinor` failed closed for `nctr_k > 1`** while
  carrying `oracle_covered = true`. The shared arity-3 spinor derivative transform
  pinned the auxiliary-k axis to a single spherical axis, so their coverage had been
  proven on a fixture that could not have exercised a general-contracted aux-k. Aux-k
  now carries its own contraction axis and both rows are **re-proven at
  `nctr_i = nctr_j = nctr_k = 2`**. The same fix removes the `nctr_k > 1` rejection
  from `int3c1e_ip1_spinor` and `int3c1e_iprinv_spinor`.
- **`int1e_prinvp` reached its kernel with no rinv origin.** Its symbol contains
  neither `iprinv` nor the `int1e_rinv_`/`int1e_drinv_` prefix, so neither
  `is_iprinv_family_symbol` nor `validate_rinv_orig_env_params` matched it. Both now
  name it, so a missing `PTR_RINV_ORIG` fails at the typed boundary.

### Deferred — gradient-gap Wave 5 (2026-08-22)
- **`int1e_ecp_iprinv_spinor` is permanently deferred for v1.4**, not merely
  unimplemented. No oracle exists in any vendored source: libcint has no ECP, and
  PySCF's `nr_ecp_deriv.c` contains no spinor code (only `nr_ecp.c`'s `ECPso_spinor`,
  a different operator). Under the byte-identity rule the row could never reach
  `oracle_covered = true` even if implemented. It stays fail-closed at
  `kernels/ecp.rs:2047` with the reason recorded in the manifest.
- **Six spinor rows have no upstream oracle** and keep `oracle_covered = false`
  permanently: `int3c1e_spinor`, `int3c1e_ip1_spinor`, `int3c1e_iprinv_spinor`
  (`CINT3c1e_spinor_drv` is `fprintf` + `exit(1)`), and `int2c2e_ip1_spinor`,
  `int2c2e_ip2_spinor`, `int2c2e_ip1ip2_spinor` (upstream stubs that write nothing and
  return 0 — they fail *silently*). cintx evaluates the first five; byte-identity is
  unobtainable. `int2c2e_spinor` is unaffected — its driver stubs only for
  `ncomp > 1`, and the base family has `ncomp == 1`.
- **32 Phase-24/26 spinor rows handed back** to
  `.planning/notes/phase-24-26-spinor-completion-PLAN.md` rather than absorbed into
  Wave 5. They are documented D-09/D-11 deferrals from two closed phases, not this
  wave's debt.

### Changed
- **BREAKING (next release):** `Builder::default()` and any safe-API caller that
  uses `..ExecutionOptions::default()` now resolves to `BackendKind::Cpu`
  (previously `BackendKind::Wgpu`). Callers that need the wgpu backend must
  opt in explicitly via `BackendIntent { backend: BackendKind::Wgpu, .. }` and
  enable the `wgpu` feature on `cintx-cubecl`. This aligns the implicit
  default with Phase 16's `CINTX_BACKEND` unset-env-var contract (defaults to
  cpu) per ROADMAP success criterion 5. Migration: pass an explicit
  `BackendIntent` (any production wgpu code already does this), or set
  `CINTX_BACKEND=wgpu` and call `--features wgpu` at the consumer.

### Added
- `cintxRsError::BackendNotCompiled { requested: String, compiled_in: Vec<String> }`
  typed error variant in `cintx-core`. Surfaces through the existing public error
  enum; rendered Display matches `requested "<name>" is not compiled in;
  compiled-in backends: ["<a>", "<b>"]`. Used in Wave 1 by the fallible
  `resolve_backend_kind() -> Result<BackendKind, cintxRsError>` rewire. (Phase 16,
  D-01.)
- `CintxStatus::BackendNotCompiled = 10` and `CINTX_STATUS_BACKEND_NOT_COMPILED`
  C-ABI status code in `cintx-capi`, with mapping arm in `status_from_core_error`
  and exported-constant test coverage. Stable code; never to be re-used.
