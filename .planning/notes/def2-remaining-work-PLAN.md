# Remaining work — def2 + throughput

**Status**: Plan for what is left after the 2026-08-24 session
**Predecessor**: `.planning/notes/def2-basis-cubecl-throughput-PLAN.md` (Phases 32/36 complete)
**Date**: 2026-08-24

---

## Execution log — 2026-08-24, continuation session

Everything the first log's "What is left" listed is now either **done** or
explicitly deferred with a reason. Full numbers and method:
`artifacts/def2_remaining_work_report_2026-08-24.md`.

**Successor plan**: `.planning/notes/post-phase-35-remaining-work-PLAN.md`
covers everything still open — launch-class merging, Phase 33, the derivative
families, def2/J, device-resident output, clippy warnings and GPU verification.

| Item | Status | Evidence |
|---|---|---|
| Backend feature gating (Phase 33 prerequisite, Part 4-02 finding) | **done** | all 5 backend-only profiles compile; new `backend_profile_matrix` CI leg |
| CPU one-quartet-per-unit kernel mode | **done**, extended to all four batched families | comptime `per_unit` flag; `plane::per_unit_width` |
| Host `cart_to_sph_2e` allocation + identity-axis cost | **done** | `cart_to_sph_2e_into`; host transform 2.5-2.9x faster |
| 34-C device-resident basis, cross-call half | **done** | `ResidentTwoEBasis`; `resident_basis_uploads_once_and_changes_nothing` |
| 34-D primitive screening | **done** | `TwoEBatchOptions::primitive_tolerance`; `primitive_screening_at_zero_is_the_identity` |
| 34-F public API | **done** | `QuartetBatchRequest` / `evaluate_shell_quartets`; `def2_quartet_batch_facade` |
| Phase 35 batching (`int1e_*`, `int2c2e`, `int3c2e`) | **done** | `def2_pair_batch_parity`; 12.8x-26.6x, bit-identical |
| Part 4-01 manifest audit | **re-run, `status: ok`** | new public rows do not disturb the lock |
| Part 4-03 clippy | **deny-level clean** (was 75-82 errors) | ~2 674 warnings remain — separate task |
| **Two new correctness bugs** (`int2c2e`, `int3c2e` general contraction) | **found and fixed** | `general_contraction_device_indexing` extended to all five families |
| Phase 33 itself | **not started** | see below |

### The measurement that redirected the throughput work

Splitting `evaluate_2e_quartet_batch` into `dispatch_ns` + `host_transform_ns`
(now carried in `BatchExecutionStats`, not printed and discarded) and solving
`T = n*k + launches*c` across the H2O and CH4 points gave **k ~= 0.18 us of
arithmetic per quartet and c ~= 42 us per launch**. The kernel was already at
libcint's speed; the *launch* was 55% of the H2O run, and the host cart-to-sph
was another third of CH4's.

So the two changes that moved the number were not kernel arithmetic at all: an
allocation-free, identity-axis-skipping `cart_to_sph_2e`, and sizing the per-unit
width by the work available instead of by `available_parallelism`. Waking a unit
costs ~2 us; H2O/def2-SVP `int2c2e` classes are ~16 pairs each and 16 units was
**3x slower** than 4.

**Result**: CH4/def2-SVP 1.70 -> **0.52 us/quartet**, and 2.6x slower than
libcint -> **1.28x faster**. H2O/def2-SVP is 1.43x slower, still launch-bound at
69 dispatches.

### The bugs the def2 fixtures could not see

`general_contraction_device_indexing` had settled the general-contraction
question for the device 2e path. Asking it of the other four families found the
`int2c2e` and `int3c2e` device kernels summing every contraction coefficient
product into one scalar and writing a single Cartesian block — correct only when
every `nctr == 1`, which is every def2 shell. Errors of 2.8e1 and 5.4e0. Both
fixed, along with the matching assumption on the transform side (`int2c2e`
transformed only block zero; `int3c2e`'s `swap_ij` transpose ran over the whole
buffer as one block). Spinor general contraction for those two families now
fails closed instead of silently transforming block zero.

### Deferred, with reasons

- **Phase 33** — not started. Its prerequisite is done (the `cubecl::cpu` gating
  fix), but its highest-risk item, 33-05's per-backend proof that the compiler
  does not contract `two_sum`/`two_prod` into FMA, cannot be discharged on this
  host for any GPU backend. Starting the ceiling raise without it would land
  fail-open code.
- **def2/J, def2/JK aux bases** (Q4) — not added. The batched 3c2e/2c2e paths
  take any shell list and are tested against def2-SVP AO shells; a realistic
  RI-J benchmark still needs the auxiliary sets.
- **34-E device-resident output mode** — not added; the collective-readback half
  was already done by 34-B.
- **Clippy warnings** — errors cleared, ~2 674 warnings remain.
- **Open questions 1-5** — still open. Q1: 2e has now reached parity on the CPU
  backend. The next lever is launch *count*, not arithmetic — merging classes
  that share `(ibase, kbase, nroots)` and moving the per-class shape parameters
  into a per-quartet table would take 69 launches to roughly 15.

---

## Execution log — 2026-08-24 session

This plan has been executed through 34-B. What changed relative to the plan as
written, and why, is recorded here; the sections below are the original plan and
are left intact so the deltas are visible.

| Task | Status | Evidence |
|---|---|---|
| 34-A0 CubeDim A/B | **done** | `artifacts/34-A0_cube_dim_ab.md` |
| 34-A cooperative G-tensor | **re-scoped and done** — see below | `plane::cooperative_cube_dim`, 69/69 class sweep |
| 34-B grid over quartets | **done** | batched `two_electron_scalar_kernel`, `def2_2e_batch_parity`; 3081 launches -> 69, ~36 us -> ~1.7 us/quartet |
| 34-C device-resident basis | **half done** | one basis upload per run (was one per class); cross-call residency open |
| 34-E collective readback | **done as part of 34-B** | one readback per launch class |
| Phase 35 (cube dim, all families) | **done** | 59 launch sites moved to `backend_plane_cube_dim` |
| Part 4-04 provenance | **done** | `crates/cintx-basis/data/README.md` |
| Part 4-05 crate docs | **done** | README source tree, design doc §4.5 |
| Part 0 full oracle regression | **done, green** | 91/91 binaries, 325 cases, 0 failures |
| Part 4-01 manifest audit (+ lock) | **done** | `status: ok` with the new crates present |
| Part 4-02 feature matrix | **done, with findings** | see "What is left" |
| Part 4-03 clippy / fmt | **fmt done; clippy pre-existing baseline** | see "What is left" |
| 34-C / 34-D / 34-F, Phase 33, Phase 35 batching | **not started** | see "What is left" |

### 34-A0's answer inverted 34-A

The A/B the plan asked for came back decisively **against** the plan's own
hypothesis. On the CubeCL CPU runtime, `CubeDim = 1` is between **28x** and
**~4.9e5x faster** than the shipped 256-unit cube, because (read from
`cubecl-cpu-0.10.0`):

- one cube unit is **one OS thread**, and the pool grows past
  `available_parallelism` on demand;
- `sync_cube` is a **global spin-wait barrier** over every unit;
- `cube_count` lowers to a **sequential `scf.for` inside each unit**, so the
  grid is not a parallelism axis on this backend at all.

The 2e kernel synchronises twice per *primitive* quartet, so a def2-SVP
`(s,s|s,s)` quartet paid 4802 barriers across 256 oversubscribed threads.
Distributing the VRR/HRR build across the cube — 34-A as written — would have
*added* `nmax+mmax+li+ll` further barriers per primitive quartet. The
barrier-count check the plan required comes back negative before any code is
written, so 34-A landed as its inverse: a backend-aware cube dimension, with the
cooperative shape preserved for GPU backends where `sync_cube` is a workgroup
barrier and the grid is real parallelism.

**Result**: H2O/def2-SVP whole-workload benchmark 125.4 s -> 0.0086 s
(~530 ms -> ~36 us per quartet), and the gap to libcint from 390 000x to 58x,
with the class-complete sweep unchanged at 69/69.

### 34-B took the rest of the launch overhead

With the barriers gone, the remaining ~36 us/quartet was per-call overhead, not
arithmetic: twelve buffer allocations, a dispatch and a blocking readback per
shell quartet. Batching by launch class removes all three.

| Case | quartets | launches | us/quartet | vs libcint |
|---|---|---|---|---|
| H2O / def2-SVP, screened | 3 081 | **69** | 2.28 | 2.9x slower |
| CH4 / def2-SVP, screened | 14 706 | **69** | **1.70** | 2.6x slower |

Byte-identical to vendored libcint (max abs diff 2.7e-15) and **bit-identical**
to cintx's own per-quartet path. Full numbers:
`artifacts/def2_throughput_report_2026-08-24.md`.

End to end for the session: **~530 ms -> ~1.7 us per quartet, about 310 000x**,
and 390 000x slower than libcint -> **2.5x** slower, single-threaded on both
sides.

### What Part 0 found

The full oracle suite now runs to completion (**91/91 binaries, 325 test cases,
0 failures**) — and it takes **under a minute**, because the same barrier
pathology that made the benchmark 390 000x slow also made the suite untestable.
Closing that gap is what turned Part 0 from "hours per test, never completed"
into a routine gate.

Running it surfaced **two real correctness bugs**, both on the host
`nroots > MAX_DEVICE_NROOTS` fallback and both invisible to every prior fixture:

1. **Missing Gaussian-product prefactor** in the `int2e` host arm — it passed
   the bare `common_factor` to `fill_g_tensor_2e` instead of
   `common_factor * fac_ij * fac_kl`. Every single-centre quartet has both
   factors equal to 1, so only *multi-centre* quartets above the Rys ceiling
   were wrong; def2-TZVP `(p,f|f,f)` came out a uniform **5.37x** too large.
2. **Transposed contraction coefficients** in the `int2e`, `int1e_nuc` and
   `int2c2e` host arms — `coeff[c*nprim + p]` where `Shell::coefficients` is
   `coeff[p*nctr + c]`. Harmless whenever `nctr == 1` or `nprim == 1`, which was
   every fixture that had reached those arms.

Both are fixed; `def2_tzvp_host_rys_diagnostic` records the structure that
identified the first (a constant `actual/expected` ratio across the block, not
a permutation).

Two harness faults were also fixed rather than papered over: the ECP oracle
tests read `data/cu_lanl2dz.json` relative to the crate root, and
`zero_tolerance_screening_is_the_identity` asserted that a 1e-10 screen must
drop work on H2O/def2-SVP — it must not, because every shell pair in that
molecule sits within ~1.8 bohr. The screen's negative control now derives its
tolerance from `max_q^2` instead.

### What is left

- **34-C device-resident basis, cross-call half** — the basis is now uploaded
  once per *run* (backend resolved once for the whole batch, handles reused
  across all 69 dispatches; `transfer_bytes` 116 -> 60 KiB on H2O/def2-SVP, and
  the test pins it to "one basis + the quartet tables"). Retaining it in
  `DeviceResidentCache` across *separate calls* is the remaining step. Note it
  is worth very little on the CPU backend — the uploads are memcpy — so it
  should be landed and measured on a GPU backend, not here.
- **34-D primitive screening / primitive-loop parallelism** — untouched.
- **34-F public API** (`QuartetBatchRequest`) — untouched. It adds public rows
  and so must land together with the compiled-manifest lock.
- **Phase 33** (device Rys nroots 6-12) — untouched. Related finding from
  Part 4-02: `cintx-cubecl` does **not** compile under
  `--no-default-features --features wgpu` (or `rocm`) because
  `math/rys_wheeler.rs` and `math/eigh.rs` name `cubecl::cpu::CpuRuntime`
  without a `cfg(feature = "cpu")` gate — 14 errors, all pre-existing. With the
  default `cpu` feature on (what CI uses) both GPU profiles compile. Phase 33
  should fix that gating while it is in that code.
- **Phase 35 batching** — only the cube-dimension half is done (all 59 launch
  sites). `int3c2e`, `int2c2e` and `int1e_*` still launch once per shell tuple;
  they need the same batching treatment 34-B gave `int2e`.
- **A CPU-specific "one quartet per unit" kernel mode.** The CPU runtime runs
  the whole batch in a single unit today (its `cube_count` is a sequential
  loop), so 15 of 16 cores are idle. Mapping quartets to *units* instead of
  cubes — per-unit G slab, comptime-skipped `sync_cube` — is the obvious next
  multiplier and is the one change that could take the remaining 2.5x to a win.
- **Part 4-03 clippy**: `cargo clippy --workspace --all-targets` is **not**
  clean at HEAD and was not before this work — 75 deny-level findings, all in
  pre-existing kernel/math files (`PIE4` vs `FRAC_PI_4`, `0u32 * stride`), plus
  ~2400 warnings. `cintx-basis` and `cintx-driver` themselves are error-free.
  Making the workspace clippy-clean is its own task.
- **Open questions 1-5 remain open.** Note for Q1: this host has an AMD Radeon
  860M (gfx1151, 8 CUs, integrated). Its f64 rate makes it an unlikely
  *throughput* target against a 16-core CPU running libcint; it is useful as a
  **correctness** target for the GPU launch topology, not as the benchmark
  platform the plan's "beat libcint" goal needs.

---

## 0. Where things actually stand

Completed and verified last session:

| Item | Evidence |
|---|---|
| `cintx-basis` (def2-SVP / TZVP / ECP, parser, normalization, raw emission) | 22 unit tests + 5 vendor-gated oracle tests |
| `cintx-driver` (enumeration, screening, bucketing, tiering, stats) | 15 tests |
| def2 1e overlap parity, every shell pair, both bases | 0 mismatches, ~1e-16 |
| def2-SVP 2e parity, all 69 launch classes | 69/69 after the fix |
| `kj2d` HRR loop-bound bug | fixed + 3 in-crate regression tests |
| Whole-workload benchmark | `artifacts/def2_throughput_report.md` |

**The measured gap**: 236 def2-SVP quartets take cintx **125.4 s** vs libcint **0.0003 s** —
about **390 000x**. Warm-up is 613 ms/class; steady state ~530 ms/quartet, so the cost does not
amortize.

### 0.1 The root cause, now pinned to specific lines

`two_electron_scalar_kernel` is launched with `single_cube_count()` (**1 cube**) and
`standard_plane_cube_dim()` (**256 units**) — `two_electron.rs:1386-1389`. Inside the kernel, per
primitive quartet:

| Lines | Work | Executed by |
|---|---|---|
| 774-787 | Rys roots | **unit 0 only** |
| 829-1160 | **entire G-tensor build (VRR + HRR, ~330 lines)** | **unit 0 only** |
| 1161 | `sync_cube()` | all |
| 1161-1308 | contraction + output accumulation (`q_elem % CUBE_DIM == UNIT_POS`) | distributed |
| 1309 | `sync_cube()` | all |

So the dominant cost — building a G-tensor of up to 1 125 elements x 3 axes — is **single-threaded
on unit 0**, while 255 units sit at a barrier. And this happens **once per primitive quartet**: a
def2-SVP oxygen `(d d|d d)` quartet has 7^4 = 2 401 primitive quartets, so ~2 401 barrier
round-trips per shell quartet with 255/256 units idle across the expensive stretch.

There are therefore **three independent unexploited axes of parallelism**, and the plan below
attacks them in decreasing value-per-effort order:

1. **Within a quartet** — the G-tensor build is serial on one unit (§1.1).
2. **Across quartets** — one launch per quartet instead of one launch per class (§1.2).
3. **Across primitive quartets** — the primitive loop is serial within a cube (§1.4).

### 0.2 Outstanding verification debt (blocking)

The full 2e oracle regression suite launched after the `kj2d` fix **never completed** — its output
file is empty, the run was stopped, and `two_electron_parity` alone had been running for hours.

What *was* verified after the fix:
- all 18 in-crate `cintx-cubecl` 2e device tests (every HRR branch, ip1/ip2) — pass
- 3 new `kj2d` regression tests at `(p,p,d,p)`, `(p,d,d,p)`, `(d,d,d,p)` — pass
- def2-SVP class-complete sweep, 69/69 — pass

What was **not** verified: `two_electron_parity`, `hess2e_parity`, `center_3c2e_parity`,
`center_2c2e_parity`, `gradient_gap_tier1_2e`, `giao_2e_parity`, `deriv34_parity`,
`int2e_ip2_parity` and the other ~80 oracle test files. The fix is one token and matches both
libcint and cintx's own host path, so regression risk is low — but "low" is not "measured", and
this must close before anything else lands.

---

## Part 1 — Phase 34: throughput (the actual "beat libcint" work)

### 1.1 Task 34-A — Parallelize the G-tensor build across the cube *(highest value)*

**Problem**: `two_electron.rs:829-1160` runs entirely under `if UNIT_POS == 0`.

**Change**: distribute the VRR and HRR loops over the cube's units, indexing by
`UNIT_POS`/`CUBE_DIM` the same way the contraction block at 1161-1308 already does.

Structure, matching libcint's own loop shape:
- **VRR** (`vrr_fill_axis`) walks `(n, m)` for each root; the `nroots x (nmax+1) x (mmax+1)`
  index space is the natural unit assignment, with a `sync_cube()` between recursion levels
  because level `n` reads level `n-1`.
- **HRR** (`hrr_*_4d`) walks `(i, j, k, l)` planes; within one `i` (or `l`) level the writes are
  independent, so the plane can be split by unit with a `sync_cube()` between levels.
- The three axes (x/y/z) are fully independent and can be a first, trivial 3-way split if a
  staged landing is wanted.

**Barrier count matters**: naive per-level syncs could add more barriers than they remove. Count
them explicitly — VRR has `nmax+mmax` levels, HRR has `li + ll` levels — and compare against the
current 2 barriers per primitive quartet.

**Acceptance**
- All 69 def2-SVP classes still byte-identical to vendor (`def2_2e_class_diagnostic`).
- All 18 in-crate device tests + 3 `kj2d` regression tests pass.
- Measured steady-state ms/quartet on the benchmark improves; record the factor.

**Risk**: medium. Touching the VRR/HRR body is the highest-risk edit in this plan. The
class-complete sweep is a strong net, but land it behind a feature flag
(`cubecl-2e-cooperative-g`) so it can be A/B'd and reverted independently.

**Cheap prerequisite experiment (34-A0)**: before writing any of it, A/B the launch with
`CubeDim::new_1d(1)` vs `new_1d(256)` on one `(d,d|d,d)` quartet. If 1 unit is not markedly
slower, the 255 units are contributing nothing and the parallel fraction is even smaller than
assumed — which changes how much 34-A can possibly buy. **This is a 1-line change and one
measurement; do it first and let the number decide the effort budget.**

### 1.2 Task 34-B — Grid over quartets (batched launch)

**Change**: `single_cube_count()` -> `plane::linear_grid_cube_count(n_quartets, ...)` (the helper
already exists, `plane.rs:58`), one cube per quartet, quartet identity from `CUBE_POS`.

Requires:
- **Quartet descriptor table** (device array): per quartet, 4 shell ids + the output offset.
  ~20 bytes/quartet vs the current per-quartet re-upload of every exponent and coefficient.
- **Flattened basis arrays** + per-shell offset table, so a shell's exponents/coefficients are
  indexed rather than passed as their own `Array`. This changes the kernel signature from 8
  per-shell arrays to 2 flat arrays + an offset table.
- **Per-cube scratch**: `g`, `urys`, `wrys` are currently single per-launch buffers. They become
  `n_cubes`-strided slabs (Tier C) or `SharedMemory` (Tier B), sized by
  `shared_memory::calc_2e_layout`.
- Comptime parameters (`ibase`, `kbase`, `nroots`) stay comptime — they are constant within a
  launch class, which is exactly what `cintx-driver`'s bucketing guarantees.

**Payoff**: 3 081 launches -> 69 for H2O/def2-SVP, and the per-class warm-up (613 ms) is paid once
per launch instead of being re-paid per quartet.

**Acceptance**: batched output byte-identical to the per-quartet path on all def2-SVP and
def2-TZVP fixtures; `BatchStats.kernel_launch_count == number of buckets`.

**Risk**: medium-high — it is the largest signature change. Do it *after* 34-A so the two are
separately attributable.

### 1.3 Task 34-C — Device-resident basis

Extend `resident_cache::DeviceResidentCache` (keyed by the existing IEEE-exact `basis_hash`) to
retain the flattened exponent/coefficient/centre arrays from 34-B across calls, so a Fock build
uploads the basis once rather than once per launch.

**Acceptance**: `BatchStats.transfer_bytes` for the second and later iterations drops to the
descriptor table only; results unchanged.

**Risk**: low. Additive, and the cache already exists with the right key.

### 1.4 Task 34-D — Primitive-quartet parallelism and screening

Two changes to the innermost loop:

- **Primitive-pair screening**: skip primitive pairs whose prefactor
  `exp(-a_i a_j |R_ij|^2 / (a_i + a_j))` is negligible. Precompute surviving pair lists per shell
  pair on the host, keep them device-resident, and iterate the compacted list. def2-TZVP carbon
  has 11 s-primitives, so 11^4 ~= 14 600 primitive quartets per shell quartet; this is typically a
  3-10x reduction.
- **Distribute the primitive loop** across planes within a cube (each plane owns a slice of the
  primitive-quartet list, with a final reduction into `cart_out`). Only worth doing after 34-A,
  because it competes for the same units.

**Acceptance**: with the screening threshold at 0, results are byte-identical to unscreened —
the same identity gate `cintx-driver` already enforces for Schwarz screening.

**Risk**: medium. Screening thresholds are a correctness surface; the zero-threshold identity test
is the guard.

### 1.5 Task 34-E — Collective readback and device-resident output

- One `client.read` per bucket rather than per quartet (the s-s pilot already proves this shape
  and instruments `readback_ns`).
- Reuse the pilot's retained output-staging arena; surface `output_staging_reuses` so allocation
  churn regressions are visible.
- Add a **device-resident output mode** where AO blocks stay on device for a downstream consumer.
  A 30-atom def2-TZVP system (~700 AO) has a ~1.9 TB dense ERI tensor; host materialization is not
  a real workflow and benchmarking with it measures PCIe, not the kernel.

**Risk**: low-medium.

### 1.6 Task 34-F — Public API

Add to `cintx-rs`, alongside the existing per-request surface:

```rust
pub struct QuartetBatchRequest<'basis> { /* operator, representation, basis, tolerance, options */ }
pub fn evaluate_shell_quartets(req: QuartetBatchRequest<'_>) -> Result<QuartetBatchOutput, FacadeError>;
```

`QuartetBatchOutput` carries the AO blocks plus `BatchExecutionStats` so a claimed speedup stays
auditable. Route it through the compiled manifest lock and `xtask manifest-audit` from the start
(see §4).

**Risk**: low, but it is public surface — it must not leak CubeCL types (project constraint).

---

## Part 2 — Phase 33: device Rys nroots 6-12 (required for def2-TZVP)

**Corrected scope** (from the predecessor plan's §1.3): the device Wheeler path is *not* unwired.
`rys_roots_host_wheeler` (`rys_wheeler.rs:3175`) already dispatches nroots 8-12 to
`rys_jacobi_device` / `lrys_schmidt_device` / `lrys_laguerre_device` — but each of those
**creates a `CpuRuntime` client and launches a kernel for one `(nroots, x)` evaluation**
(`rys_jacobi_device:1404`). `fill_g_tensor_2e` calls it once per primitive quartet.

nroots 6-7 (the def2-TZVP main-group f range) deliberately stay on pure host code per a documented
parity decision (`rys_wheeler.rs:3188-3198`): a device launch in the family hot path perturbed the
host g-tensor accumulation by ~1e-11 and tripped the `hess2e` flat-atol=1e-12 gate.

### Tasks

- **33-01** — Promote the `#[cube]` helpers (`gamma_inc_like_dev`, `r_dsmit_dev`,
  `cint_polynomial_roots_dev`, and the dd chain) to `pub(crate)` and add an **inline** device entry
  `rys_roots_dev(nroots, x, u, w, scratch...)` callable from *inside* `two_electron_scalar_kernel`
  — not a separate launch.
- **33-02** — Size the Wheeler scratch through `shared_memory::calc_math_layout` and register it in
  the layout catalog: `fmt_ints` 64, `cs` `(nroots+1)^2 <= 169`, `rt` 12, `acomp` 144, `vscr` 32,
  `flag` 1 — about **3.4 KB of f64 per work-item**. Bounds-validate at nroots = 12.
- **33-03** — Raise `MAX_DEVICE_NROOTS` 5 -> 12 **per family, behind a flag**, each gated on its own
  oracle parity test. Never in one commit.
- **33-04** — Accuracy gate: device vs `rys_roots_host_wheeler` over a log-spaced `x` sweep
  (1e-8 .. 1e6) x nroots 6..12 at the existing oracle tolerance.
- **33-05** — **Highest-risk item**: verify the compiler does not contract `two_sum`/`two_prod`
  into FMA and destroy the double-double error-free transform. The existing `fma_probe` established
  this for the CubeCL 0.10 CPU backend; it must be re-established per GPU backend. If a backend
  fails, gate that backend to nroots <= 5.
- **33-06** — Re-examine the nroots 6-7 host-only decision **once roots are inline**. The original
  rationale was launch-induced FP-environment perturbation; with no launch, the rationale may no
  longer hold. Re-run `hess2e_parity` to decide. Do not loosen the tolerance.

**Sequencing note**: the device Wheeler kernels take `#[comptime] nroots`, which specializes one
kernel per Rys order — exactly what 34-B's per-class bucketing provides. **Do 33 and 34-B together**,
not in sequence.

---

## Part 3 — Phase 35: the other families

Apply the 34 treatment to shell **pairs** and 3-index lists:

| Family | Work list (H2O/def2-SVP -> TZVP) | Priority |
|---|---|---|
| `int3c2e` | `nbas^2 x naux` | **highest** — RI-J with def2/J is the dominant cost in real def2 workflows, and its list is more uniform than 2e's, so it buckets almost perfectly |
| `int2c2e` | `naux^2` | high — small and trivially batched |
| `int1e_*` (ovlp/kin/nuc) | 78 -> 190 pairs | medium — cheap per item, but 30-160x slower than libcint today |

Requires the def2/J and def2/JK **auxiliary** basis sets in `cintx-basis` (same parser, same
normalization path — a small addition, open question Q4 below).

**Acceptance**: byte-identical to vendor; >= 10x current CubeCL throughput on 1e.

---

## Part 4 — Release-gate hygiene (must not be skipped)

The two new crates are in `default-members` but have not been through the project's release gates.

- **4-01** — `xtask manifest-audit` with `cintx-basis` / `cintx-driver` present; regenerate
  `compiled_manifest.lock.json` if 34-F adds public API rows.
- **4-02** — Feature-matrix CI: the new crates must build under every feature combination already
  in CI (`cpu`, `wgpu`, `cuda`, `rocm`, `metal`, `with-f12`, `with-4c1e`, `unstable-source-api`).
  `cintx-driver` has a `cpu` feature forwarding to `cintx-compat/cpu` — verify the others.
- **4-03** — `cargo clippy --workspace --all-targets` and `cargo fmt --check` on the new crates.
- **4-04** — Licensing/provenance for the vendored BSE data: the files carry the BSE header, but
  add an explicit provenance note (source URL, version, retrieval date) to
  `crates/cintx-basis/data/README.md` and confirm redistribution terms for a public library.
- **4-05** — Document `cintx-basis` and `cintx-driver` in the top-level README and the design doc's
  crate-layout section, which currently lists only the original nine crates.

---

## 5. Sequencing

```
Part 0  full 2e oracle regression      [BLOCKING — nothing lands until green]
   |
   +--> 34-A0  CubeDim A/B measurement [1 line; decides 34-A's budget]
   |       |
   |       +--> 34-A  cooperative G-tensor build
   |               |
   |               +--> 34-B  grid over quartets  <---- do jointly ---->  Phase 33 (inline Rys)
   |                       |
   |                       +--> 34-C  device-resident basis
   |                       +--> 34-D  primitive screening + parallelism
   |                       +--> 34-E  collective readback
   |                               |
   |                               +--> 34-F  public API --> Part 4 (gates)
   |                               +--> Phase 35 (1e / 2c2e / 3c2e)
   |
   +--> Part 4-04/4-05 (provenance, docs) — independent, can run any time
```

**Earliest meaningful throughput result**: Part 0 + 34-A0 + 34-A. That alone should move the
530 ms/quartet number, and it needs none of the signature churn in 34-B.

---

## 6. Effort and risk

| Task | Size | Risk | Blast radius |
|---|---|---|---|
| Part 0 (regression) | hours of machine time, no code | — | — |
| 34-A0 | minutes | none | none (measurement) |
| 34-A cooperative G-tensor | large | **medium-high** | 2e kernel body |
| 34-B grid over quartets | large | medium-high | kernel signature + all call sites |
| 34-C resident basis | medium | low | additive |
| 34-D primitive screening | medium | medium | correctness surface |
| 34-E readback/output | medium | low-medium | additive |
| 34-F public API | small | low | manifest lock |
| Phase 33 | large | **high** (dd/FMA) | rys + 2e kernel |
| Phase 35 | large | medium | 3 more kernels |
| Part 4 | small | low | CI |

---

## 7. Risks carried forward

| Risk | Impact | Mitigation |
|---|---|---|
| **f64 on GPU** — consumer NVIDIA runs f64 at 1/32-1/64; wgpu often lacks `SHADER_F64` | Could erase the entire win | `check_shader_f64_in_features` already gates it. **Measure the backend's f64:f32 ratio before committing to 34-B's tiering.** |
| **dd miscompilation** (FMA contraction) at nroots >= 6 | Silent accuracy loss | 33-05; per-backend gate to nroots <= 5 on failure |
| **Oracle suite too slow to iterate against** | Long feedback loops; this already bit us | Use `def2_2e_class_diagnostic` (44 s, 69 classes) as the fast inner gate; run the full suite only at phase boundaries |
| **34-A barrier overhead** exceeds the parallelism gained | 34-A is a wash | Count barriers explicitly before implementing; 34-A0 bounds the upside first |
| **34-B tier C bandwidth** — f/g quartets stream a 128-570 KB G-tensor through global memory | TZVP win much smaller than SVP | Ship simple, measure the roofline, then tile `(dlj, dll)` into shared memory |
| **Screening bug reads as a speedup** | Wrong results reported as a win | `tolerance = 0` identity gate (already enforced) |

---

## 8. Open questions (answers change the plan)

1. **Which backend is the throughput target** — CUDA / ROCm / wgpu / CPU? This decides the f64
   strategy and whether Tier C is viable at all. *On the CPU backend, parity with libcint remains
   the realistic ceiling — a win needs a GPU backend.*
2. **Is device-resident output acceptable** as the primary benchmark mode, or must results be
   host-materialized (capping any GPU win at PCIe bandwidth)?
3. **Is def2-ECP (Z >= 37) in scope now**, or main-group first? The ECP data and shells are already
   built and parsed; only the integral path is untested for those elements.
4. **def2/J and def2/JK auxiliary bases** — in scope? They are what makes Phase 35's 3c2e/2c2e work
   worth doing, and are how def2 sets are actually used in production.
5. **Does the `hess2e` 1e-12 gate still bind** for nroots 6-7 once roots are inline (33-06)?
