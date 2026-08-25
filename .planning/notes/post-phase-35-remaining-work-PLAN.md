# Remaining work — after Phase 35

**Status**: Plan for what is left after the 2026-08-24 continuation session
**Predecessor**: `.planning/notes/def2-remaining-work-PLAN.md` (34-C/D/F + Phase 35 complete)
**Evidence for every "measured" claim below**:
`artifacts/def2_remaining_work_report_2026-08-24.md`
**Date**: 2026-08-24

> **Execution status, 2026-08-25** — see
> `artifacts/post_phase_35_progress_report_2026-08-25.md` for the measurements.
>
> | part | state |
> |---|---|
> | Part 1 — 35-M1, 35-M2 | **done**. 69→15 dispatches for 2e; 27→4, 9→3, 9→1 for the others. Bit-identical. |
> | Part 6 — clippy | **done**. `-D warnings` clean, ~2 078 → 0. |
> | Part 7 — ROCm | **done, with a correction**: the "byte-identical to the CPU results" bar is not achievable and the reason is not the launch topology (the AMD compiler contracts FMA where the CPU one does not). Correctness against vendored libcint is 0 mismatches for all five families. |
> | Part 2 — 33-05 | **discharged for ROCm**, which the plan recorded as impossible on this host: `fma` fuses bit-for-bit. Per-backend ceiling scaffolding landed; the raise itself is untouched. |
> | Part 4 — def2/J, def2/JK | **done**, with the RI-J benchmark. It reports cintx ~2x slower there, and shows def2-TZVP + def2/J falling outside the nroots ceiling — the first real workload Phase 33 would unblock. |
> | Part 3 — 35-D | **first two waves done**: `int3c2e_ip1`/`ip2` (1 728 launches -> 4, 25x) and `int1e_ipovlp`/`ipkin`/`ipnuc` (144 -> 1/1/3, 25-33x). 10 more 1e kernels plus `sigma_p`, `ecp`, `f12`, `center_3c1e`/`4c1e` and the `unstable-source-api` set remain per-tuple. |
> | Part 3 — 34-C2 | **done for `int3c2e`** (the family RI-J needs). 35-F2, 34-D2 and 34-C2 for the other families remain. |
> | Part 5 | still blocked on open question 2. |
>
> **Successor**: `.planning/notes/post-phase-35-continuation-PLAN.md` carries
> everything still open, re-prioritised against what the measurements below
> turned up.
>
> **The headline correction**: Part 1's projection was 0.56–0.59 us/quartet on
> H2O/def2-SVP. The measurement is 0.72–0.85, which is *parity* with libcint, not
> the 1.3–1.4x the plan called its honest floor. CH4 does beat the projection's
> spirit (1.4–1.5x faster, up from 1.28x). Both remaining gaps — 2e on H2O and
> RI-J — now spend ~40 % of their wall-clock in the **serial host cart-to-sph**,
> which no launch-count or backend work touches. That is the next bottleneck.

---

## 0. Where things actually stand

Gates at the time of writing:

```
cargo test --workspace --exclude cintx-oracle    28 binaries, 635 passed, 0 failed
CINTX_ORACLE_BUILD_VENDOR=1 cargo test --release -p cintx-oracle --features cpu
                                                 95 binaries, 358 passed, 0 failed
cargo clippy --workspace --all-targets           0 errors, ~2 674 warnings
cargo fmt --all --check                          clean
xtask manifest-audit --check-lock                status: ok
cargo check -p cintx-cubecl --no-default-features --features {cpu,wgpu,cuda,rocm,metal}
                                                 all five compile
```

Throughput, CubeCL CPU backend vs vendored libcint 6.1.3 single-threaded, best of 9:

| Family | tuples | launches | batched | vs libcint |
|---|---|---|---|---|
| `int2e` CH4/def2-SVP screened | 14 706 | 69 | 0.52 us/quartet | **1.28x faster** |
| `int2e` H2O/def2-SVP screened | 3 081 | 69 | 1.17 us/quartet | 1.43x slower |
| `int3c2e` H2O/def2-SVP | 1 728 | 27 | 1.40 ms | 2.4x slower |
| `int2c2e` H2O/def2-SVP | 144 | 9 | 0.19 ms | 4x slower |
| `int1e_*` H2O/def2-SVP | 144 | 9 | 0.28-0.30 ms | 6-14x slower |

### 0.1 The one number that shapes this plan

Instrumenting the batched 2e run (`BatchExecutionStats::dispatch_ns` /
`host_transform_ns`) and solving `T = n*k + launches*c` across the H2O
(3 081 quartets) and CH4 (14 706 quartets) points gave:

- **k ~= 0.18 us** of arithmetic per 2e quartet
- **c ~= 42 us** of per-launch overhead

At 69 launches that overhead is **55% of the H2O run**. The kernel is already at
libcint's speed; what is left on the CPU backend is *launch count*, not
arithmetic. That is why Part 1 leads.

The 42 us decomposes into three buffer allocations, a `cubecl-cpu` dispatch that
sends every unit an mpsc message and clones the binding table per unit
(~2 us/unit), and a blocking readback. None of it is under cintx's control except
by making fewer launches.

---

## Part 1 — Launch-class merging *(highest value on any backend)*

### 1.1 The opportunity, counted

`two_electron_scalar_kernel` takes exactly **three** comptime parameters:
`ibase`, `kbase`, `nroots`. Everything else that varies with
`(li, lj, lk, ll)` — `di`, `dk`, `dl`, `dj`, `g_size`, `nmax`, `mmax`,
`g2d_ijmax`, `g2d_klmax`, `common_factor` — is already a **runtime scalar**
argument (`TwoEClassParams`).

So a launch class does not have to be one `(li, lj, lk, ll)` tuple. It only has
to be one `(ibase, kbase, nroots)` signature. Enumerating all 81 l-quartets over
`{s, p, d}` (the def2-SVP envelope):

```
l-quartets                      : 81   (69 reached by H2O/def2-SVP)
distinct (ibase, kbase, nroots) : 16
```

**69 launches -> 16.** Projecting the H2O run from the fit above — 3 650 us
total = 542 us arithmetic + 69 x 36 us launch + 600 us host cart-to-sph — dropping
to 16 launches gives **~1 720 us, i.e. 0.56 us/quartet, about 1.4x FASTER than
libcint** (2 380 us on the same list). Re-fitting `c` from the earlier 3 450 us
dispatch measurement instead gives 1 820 us / 0.59 us/quartet / 1.3x faster, so
the projection is stable to the input it is most sensitive to.

That is a projection, not a measurement, and it assumes the merged launch costs
what an unmerged one does — the merged launch does more work per dispatch, so
treat ~1.3x as the honest floor. It needs no GPU, no new data, and no new public
surface.

The g_size spread inside a signature is what it costs:

| signature | classes | g_size min..max |
|---|---:|---|
| ibase=0 kbase=0 nroots=2 | 11 | 6..24 |
| ibase=0 kbase=0 nroots=3 | 14 | 27..144 |
| ibase=0 kbase=0 nroots=4 | 7 | 180..480 |
| ibase=0 kbase=0 nroots=5 | 1 | 1125 |

Worst case ~5.3x, and the slab is per *slot* (16 on this host), so the merged
scratch stays in the tens of KB. That is affordable; the tiering ceiling in
`two_e_cube_count` already bounds it.

### 1.2 Task 35-M1 — per-tuple shape table

**Change**: move `TwoEClassParams`' runtime scalars out of the launch arguments
and into a device array indexed per quartet.

- The quartet row grows from `[si, sj, sk, sl, out_off]` to include a **class
  index**; a parallel `class_params` array carries the 10 shape scalars +
  `common_factor` per class.
- `g_stride` is sized to the signature's **maximum** `g_size`, not the class's.
- `two_e_cube_dim` / `per_unit_cube_dim` take the max `g_size` for the same
  reason.
- Grouping in `evaluate_2e_batch_inner` keys on `(ibase, kbase, nroots)`; the
  `(li,lj,lk,ll)` grouping stays as the *sub*-grouping that decides the host
  cart-to-sph and the output block lengths.

**Acceptance**
- `BatchStats.kernel_launch_count == number of distinct (ibase,kbase,nroots)`,
  and strictly less than the number of l-classes on the same list.
- Still **bit-identical** to the per-quartet path on every def2-SVP fixture —
  `def2_2e_batch_parity` already asserts this and must stay green unchanged.
- `def2_2e_class_diagnostic` still 69/69.
- Record the new us/quartet on both H2O and CH4.

**Risk**: medium. It is the same class of signature change as 34-B, but the
kernel body barely moves — the shape scalars become array reads at the top of
the quartet loop instead of launch arguments. The bit-identity gate is the net.

### 1.3 Task 35-M2 — same treatment for `int3c2e` / `int2c2e` / `int1e_*`

`int3c2e` is the family RI-J spends its time in and has 27 launches on a
19-shell H2O list; its comptime set is `nroots` alone (plus `op_kind` for 1e), so
the merge factor is larger still. Do it after 35-M1 proves the shape.

**Acceptance**: as 35-M1, per family, against `def2_pair_batch_parity`.

---

## Part 2 — Phase 33: device Rys nroots 6-12

**Unchanged in substance from the predecessor plan's Part 2**; what changed is
that its stated prerequisite is now done. `cintx-cubecl` compiles under
`--no-default-features --features {wgpu,cuda,rocm,metal}`; the 13
`cubecl::cpu::CpuRuntime`-without-a-cfg-gate errors in `math/rys_wheeler.rs` and
`math/eigh.rs` are fixed, the host solvers are cfg-selected, and
`ci/feature-matrix.yml`'s `backend_profile_matrix` job builds all five profiles
so the defect class cannot return.

### Tasks (carried forward verbatim in intent)

- **33-01** — promote the `#[cube]` helpers (`gamma_inc_like_dev`, `r_dsmit_dev`,
  `cint_polynomial_roots_dev`, the dd chain) to `pub(crate)` and add an **inline**
  device entry `rys_roots_dev(nroots, x, u, w, scratch...)` callable from inside
  `two_electron_scalar_kernel` — not a separate launch.
- **33-02** — size the Wheeler scratch through `shared_memory::calc_math_layout`:
  `fmt_ints` 64, `cs` `(nroots+1)^2 <= 169`, `rt` 12, `acomp` 144, `vscr` 32,
  `flag` 1 — about **3.4 KB of f64 per work-item**. Bounds-validate at nroots=12.
  Note this now interacts with the per-slot slab padding introduced for the
  per-unit decomposition: 3.4 KB x 16 units is 54 KB per launch, which is fine on
  CPU and is a real shared-memory decision on GPU.
- **33-03** — raise `MAX_DEVICE_NROOTS` 5 -> 12 **per family, behind a flag**,
  each gated on its own oracle parity test. Never in one commit.
- **33-04** — accuracy gate: device vs `rys_roots_host_wheeler` over a log-spaced
  `x` sweep (1e-8 .. 1e6) x nroots 6..12 at the existing oracle tolerance.
- **33-05** — **the blocking item**: verify the compiler does not contract
  `two_sum`/`two_prod` into FMA and destroy the double-double error-free
  transform. `fma_probe` establishes this for the CubeCL 0.10 **CPU** backend
  only. It cannot be discharged for any GPU backend on the current dev host
  (integrated gfx1151; no CUDA device). **Until it is, 33-03 must not raise the
  ceiling for a backend whose probe has not run** — otherwise the ceiling raise
  is fail-open. Concretely: make the ceiling a per-backend value that defaults to
  5 and is raised only where the probe passed, and land that scaffolding *first*.
- **33-06** — re-examine the nroots 6-7 host-only decision once roots are inline.
  The original rationale was launch-induced FP-environment perturbation
  (`rys_wheeler.rs:3188-3198`); with no launch it may no longer hold. Re-run
  `hess2e_parity` to decide. **Do not loosen the tolerance.**

**Sequencing note (still true)**: the device Wheeler kernels take
`#[comptime] nroots`, which specializes one kernel per Rys order — exactly what
launch-class bucketing provides. Do 33 and Part 1 in a compatible order: Part 1
merges classes *by* `nroots`, so it makes 33's comptime specialization cheaper,
not harder.

**Blocked on**: hardware for 33-05, for any backend other than CPU.

---

## Part 3 — Finish the batching surface

Phase 35 batched the **scalar** kernel of each family. The derivative and
special families are still one launch per tuple. Count of remaining
`single_cube_count()` production launch sites:

| File | sites | What |
|---|---:|---|
| `one_electron.rs` | 16 | grad / ipipovlp / p4 / irp / GIAO / kinetic-grad / nuc-grad / rinv |
| `sigma_p.rs` | 6 | sa01 / spgsa01 families |
| `ecp.rs` | 4 | ECP type-1/type-2 |
| `center_3c2e.rs` | 3 | ip1 / ip2 / hess |
| `center_3c1e.rs` | 2 | |
| `center_4c1e.rs` | 2 | `with-4c1e` |
| `f12.rs` | 2 | `with-f12` |
| `sigma_1e_nuc.rs` | 2 | |
| `unstable/{breit,grids,origi,origk,ssc}.rs` | 16 | `unstable-source-api` |
| `sigma_1e.rs` | 1 | |
| `center_2c2e.rs`, `two_electron.rs` | 1 each | in-crate f32 genericity tests only |

### 3.1 Task 35-D — derivative families

Priority order by real workload weight: `int3c2e_ip1`/`ip2` (RI-J gradients),
then `int1e_ip*` (nuclear gradients), then the rest.

The pattern is established and mechanical now — flattened basis + tuple table +
per-slot slabs + the `per_unit` comptime flag — and the acceptance bar is fixed:
**bit-identity against the per-tuple path**, enforced by rewriting the per-tuple
entry point as a one-tuple batch so every existing parity test covers the batched
kernel.

### 3.2 Task 35-F2 — facade parity for the pair/triple batches

`QuartetBatchRequest` (34-F) exposes only `int2e_sph` through `cintx-rs`.
`evaluate_1e_pair_batch`, `evaluate_2c2e_pair_batch` and
`evaluate_3c2e_triple_batch` are `cintx-cubecl`-only, so a safe-API consumer
cannot reach them without depending on the backend crate — which the project's
API-ordering constraint says it should not have to.

**Change**: a `PairBatchRequest` / `TripleBatchRequest` (or one generalized
`ShellListRequest` keyed on the operator's arity) alongside
`QuartetBatchRequest`, with the same scope gate — resolve the operator through
the manifest, accept only the plain scalar symbol for the family, refuse the
rest before any device work.

**Risk**: low, but it is public surface. Route it through the compiled-manifest
lock and `xtask manifest-audit` as 34-F was.

### 3.3 Task 34-C2 — resident basis for the other families

`ResidentTwoEBasis` exists only for 2e. `int3c2e` re-flattens and re-uploads the
basis on every call, which for an RI-J build that evaluates the same
`nbas^2 x naux` list per SCF iteration is the same waste 34-C removed for 2e.

**Acceptance**: `stats.basis_upload_bytes == 0` on the second and later
evaluations, values bit-identical — the same two-sided gate
`resident_basis_uploads_once_and_changes_nothing` uses.

### 3.4 Task 34-D2 — primitive screening for the other families

`TwoEBatchOptions::primitive_tolerance` is 2e-only. The same
`fac1`-based test applies to 3c2e and to the 1e nuclear arm.

**Acceptance**: the tolerance-zero identity gate, per family.

---

## Part 4 — def2/J and def2/JK auxiliary bases (open question Q4)

The batched `int3c2e` / `int2c2e` paths take any shell list and are tested
against def2-SVP AO shells, so they are not blocked on this. What *is* blocked is
an honest RI-J benchmark: the work list that matters is `nbas^2 x naux` with a
real auxiliary basis, not `nbas^3`.

**Change**: add def2/J and def2/JK to `cintx-basis` — same parser, same
normalization path, same BSE provenance note in `crates/cintx-basis/data/README.md`.

**Acceptance**
- Auxiliary shells round-trip through `to_raw_arrays` and match vendored libcint
  on `int2c2e` and `int3c2e` for every element in the catalogue.
- An RI-J-shaped benchmark: `(mu nu | P)` over the screened AO pair list x the
  auxiliary shells, cintx batched vs libcint, with the launch count reported.

**Risk**: low. It is a data addition plus one parity test; the parser and
normalization are already exercised by def2-SVP/TZVP.

---

## Part 5 — 34-E's last sub-item: device-resident output

Everything else in 34-E landed with 34-B (one readback per launch class,
retained output staging, `readback_ns` instrumentation). What remains:

**Change**: a mode where AO blocks stay on device for a downstream consumer
rather than being read back.

**Why it still matters**: a 30-atom def2-TZVP system (~700 AO) has a ~1.9 TB
dense ERI tensor. Host materialization is not a real workflow, and benchmarking
with it measures PCIe rather than the kernel. This is the change that makes a GPU
number meaningful — and it is coupled to **open question 2**, which is not
settled: if results must be host-materialized, any GPU win is capped at PCIe
bandwidth and Part 1's launch-count work is worth more than a GPU port.

**Blocked on**: answering Q2. Do not build it before then.

---

## Part 6 — Clippy warnings

`cargo clippy --workspace --all-targets` is **error-free**. ~2 674 warnings
remain, and they are not uniform — most are mechanical, a few are worth reading:

| Count | Lint | Disposition |
|---:|---|---|
| 1 598 | `excessive_precision` | Transcribed libcint tables. `#![allow]` at the table modules with the same provenance rationale already used for `approx_constant`. **Do not truncate the literals.** |
| 608 | `default_numeric_fallback`-style trait-bound fallback notes | Emitted from inside `#[cube]` expansion; investigate whether one `#[allow]` on the macro output kills all of them. |
| 92 | `unnecessary_cast` | Mechanical. |
| 86 | `no_effect` | **Read each one.** This is the same family as the `erasing_op` findings, and two of those turned out to be real redundancy in `rys_wheeler.rs`. |
| 34 | `needless_range_loop` | Mechanical, but kernel index arithmetic is often clearer as written; prefer `#[allow]` with a reason over a rewrite that obscures a stride. |
| 34 | `manual_slice_size_calculation` | Mechanical. |
| 24 | `dead_code` | **Read each one.** A never-used function is either a gap or a leftover; both are worth knowing. |
| 18 | `missing_safety_doc` in `cintx-capi` | Real. The C ABI shim's `unsafe` fns should document their contracts. |
| 12 | `never_read` | **Read each one.** |

**Acceptance**: `cargo clippy --workspace --all-targets -- -D warnings` clean,
with every `#[allow]` carrying a reason. Land it in slices by lint, not one
sweep, so the ~50 findings worth reading are not buried in the 1 598 that are not.

**Risk**: low individually; the risk is a mass `cargo clippy --fix` silently
changing a transcribed constant. Do not use `--fix` on the table modules.

---

## Part 7 — GPU verification

Everything above is measured on the CubeCL **CPU** backend. The cooperative
(`per_unit == 0`) code path — one tuple per cube, the cube cooperating on its
contraction, real `sync_cube` barriers — is compiled but **never executed** in
CI, for any of the four batched families.

**Change**
- Run the def2 parity suites on the ROCm backend on the dev host (gfx1151). It is
  not a throughput target — its f64 rate against a 16-core CPU running libcint
  makes that implausible — but it is the only available **correctness** target
  for the GPU launch topology.
- Gate on `check_shader_f64_in_features` for wgpu as the existing code does.
- Measure the backend's f64:f32 ratio before committing to any tiering decision
  (this is the predecessor plan's carried-forward risk and is still unanswered).

**Acceptance**: `def2_2e_batch_parity`, `def2_pair_batch_parity` and
`def2_2e_class_diagnostic` green on ROCm, byte-identical to the CPU results.

**Risk**: medium — this is where a latent divergence in the cooperative path
would first appear, which is exactly why it is worth doing.

---

## 8. Sequencing

```
Part 1  35-M1 launch-class merging (2e)      [highest value, no new hardware]
   |
   +--> 35-M2 merging for 3c2e/2c2e/1e
   |
   +--> Part 7  ROCm correctness run   <---- unblocks the cooperative path ---->
   |        |
   |        +--> Part 2  Phase 33   (33-05 scaffolding first; ceiling raise
   |                                 only per backend whose probe passed)
   |
   +--> Part 3  35-D derivative batching
   |        +--> 35-F2 facade parity  --> manifest lock
   |        +--> 34-C2 resident basis (other families)
   |        +--> 34-D2 screening (other families)
   |
   +--> Part 4  def2/J + def2/JK  --> the RI-J benchmark that makes 3c2e's
   |                                  numbers mean something
   |
   +--> Part 6  clippy warnings   [independent, can run any time]
   |
   +--> Part 5  device-resident output   [BLOCKED on open question 2]
```

**Earliest meaningful result**: Part 1 alone. It needs no new hardware, no new
data, and no new public surface, and the projection in 1.1 takes H2O/def2-SVP 2e
from ~1.5x slower than libcint to ~1.3-1.4x faster.

---

## 9. Effort and risk

| Task | Size | Risk | Blast radius |
|---|---|---|---|
| 35-M1 class merging (2e) | medium | medium | kernel signature + batch host path |
| 35-M2 class merging (others) | medium | medium | three more kernels |
| Phase 33 | large | **high** (dd/FMA) | rys + every family's ceiling |
| 35-D derivative batching | large | medium | ~40 launch sites |
| 35-F2 facade parity | small | low | public surface + manifest lock |
| 34-C2 / 34-D2 | small each | low | additive |
| def2/J + def2/JK | small | low | data + one parity test |
| Device-resident output | medium | medium | output contract |
| Clippy warnings | medium | low | wide but mechanical |
| ROCm verification | small (if it passes) | medium | reveals, does not change |

---

## 10. Risks carried forward

| Risk | Impact | Mitigation |
|---|---|---|
| **f64 on GPU** — consumer NVIDIA runs f64 at 1/32-1/64; wgpu often lacks `SHADER_F64` | Could erase any GPU win | `check_shader_f64_in_features` already gates it. Measure the ratio before tiering. |
| **dd miscompilation** (FMA contraction) at nroots >= 6 | Silent accuracy loss | 33-05 **first**; per-backend ceiling defaulting to 5. |
| **Cooperative path never executed** | A divergence in the GPU shape is invisible today | Part 7. |
| **Merged-class scratch** — slab sized to the signature's max `g_size` (up to 5.3x the class's own) | Wasted scratch, possible tier pressure | Measured spread is 27..144 worst case; the `MAX_BATCH_SCRATCH_BYTES` ceiling already bounds it. Report the slab size in stats. |
| **Screening bug reads as a speedup** | Wrong results reported as a win | The `tolerance = 0` identity gate, per family (34-D2). |
| **Mass clippy `--fix` on transcribed tables** | Silent change to a frozen constant | Never `--fix` the table modules; land by lint, not by sweep. |

---

## 11. Open questions

Carried forward unchanged except where this session narrowed them.

1. **Which backend is the throughput target** — CUDA / ROCm / wgpu / CPU?
   *Narrowed*: on the CPU backend, 2e has now **reached parity** (1.28x faster on
   CH4, 1.43x slower on the smaller H2O list). Parity is no longer the ceiling
   assumption it was; Part 1 says the CPU backend can go further still. The
   question is now whether a GPU port is worth more than launch-count work, and
   that depends on Q2.
2. **Is device-resident output acceptable** as the primary benchmark mode, or
   must results be host-materialized? **This blocks Part 5** and materially
   changes the value of a GPU port.
3. **Is def2-ECP (Z >= 37) in scope now**, or main-group first? The ECP data and
   shells are already built and parsed; only the integral path is untested for
   those elements.
4. **def2/J and def2/JK auxiliary bases** — in scope? *Narrowed*: Phase 35 no
   longer depends on them (the batch paths take any shell list), but the RI-J
   benchmark does. Part 4.
5. **Does the `hess2e` 1e-12 gate still bind** for nroots 6-7 once roots are
   inline (33-06)?
