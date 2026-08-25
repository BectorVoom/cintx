# Post-Phase-35 continuation — what landed, 2026-08-25

Work against `.planning/notes/post-phase-35-continuation-PLAN.md`.
Every number below is measured on this host, not projected.

**Host**: AMD Ryzen AI 7 350 (16 threads) + Radeon 860M (gfx1151), Linux 7.1.9,
CubeCL 0.10.0, vendored libcint 6.1.3.

---

## Gate state

```
cargo fmt --all --check                                   clean
cargo clippy --workspace --all-targets -- -D warnings     clean
  (also clean under CINTX_ORACLE_BUILD_VENDOR=1, which compiles the
   vendor-gated test files the default run skips)
cargo test --workspace --exclude cintx-oracle             648 passed, 0 failed  (was 641)
CINTX_ORACLE_BUILD_VENDOR=1 cargo test --release
  -p cintx-oracle --features cpu                          375 passed, 0 failed  (was 370)
cargo check -p cintx-cubecl --no-default-features
  --features {cpu,wgpu,cuda,rocm,metal}                   all five compile
cargo check -p cintx-cubecl --all-targets
  --features {with-4c1e, with-f12, both}                  all three compile
(cd xtask && cargo run -- manifest-audit --check-lock)    status: ok, 0 uncovered
ROCm suites (CINTX_ROCM_ORACLE=1, opt-in)                 5 tests green,
                                                          0 mismatches vs vendor
                                                          across all 9 families
```

---

## Part 1 — the host cart-to-sph transform

### 36-T0 — attribute the cost *(the measurement that redirected the rest)*

New opt-in instrumentation (`CINTX_HOST_TRANSFORM_PROFILE=1`) splits
`host_transform_ns` three ways — allocate / c2s / scatter — into new
`BatchExecutionStats` fields, printed by the three benchmarks that already show
the dispatch/transform split. It is opt-in because three `Instant::now()` calls
per 27-element block would otherwise make `host_transform_ns` a probe artifact;
the counters stay `0` on an ordinary run, which reads as "not measured" rather
than "measured as zero".

**Measured, before any of 36-T1/T2:**

| workload | allocate | c2s | scatter |
|---|---:|---:|---:|
| `int2e` CH4/def2-SVP | 0.2 % | **68.3 %** | 31.5 % |
| `int3c2e` RI-J def2/J | 10.1 % | **80.9 %** | 9.0 % |
| `int3c2e` RI-J def2/JK | 9.6 % | **80.8 %** | 9.6 % |
| `int3c2e_ip1` | 14.6 % | **71.4 %** | 14.0 % |
| `int3c2e_ip2` | 15.1 % | **70.9 %** | 14.0 % |

**The plan expected allocation to be a candidate; it is 0–15 %.** The c2s
arithmetic is 68–81 % everywhere. That measurement is what the rest of Part 1
acted on, and it is why 36-T1 grew a second half the plan did not anticipate.

### 36-T1 — remove the allocations, *and* stop transforming identity axes

Two changes, both aimed by 36-T0:

1. **Allocations hoisted.** `cart_to_sph_{1e,2c2e,3c1e,3c2e}` gained `_into`
   forms taking caller-owned output and scratch (`cart_to_sph_2e_into` already
   had them), and every one of the six batch transform loops now allocates once
   per run instead of once per contraction block. `transpose_ij_3idx` too.
2. **Identity axes skipped.** `C2S_L0` and `C2S_L1` are identity matrices, so an
   `l <= 1` axis is a copy dressed as a matrix product. `cart_to_sph_2e_into`
   already skipped them; the 1-, 2- and 3-index transforms did not, and on a
   def2-SVP work list that is most axes. All four now route through one
   `c2s_apply` axis-plan driver, which also removes four copies of the same
   ping-pong buffer logic.

**Measured effect on the serial transform** (`CINTX_HOST_TRANSFORM_THREADS=1`):

| workload | before | after | |
|---|---:|---:|---|
| `int3c2e_ip1` | 1.164 ms | **0.192 ms** | 6.1x |
| `int3c2e_ip2` | 1.193 ms | **0.196 ms** | 6.1x |
| RI-J def2/J | 1.121 ms | **0.179 ms** | 6.3x |
| RI-J def2/JK | 1.143 ms | **0.447 ms** | 2.6x |
| `int2e` CH4 | 2.356 ms | **1.854 ms** | 1.27x |

### 36-T2 — parallelise the transform

`rayon` over tuples, `unsafe`-free. The disjointness argument the plan asked to
be stated in writing:

> A tuple's destination is `values[offsets[n] .. offsets[n] + len_n]`, and
> `offsets` is a running total in the caller's order, so the blocks are
> contiguous, non-overlapping and in order. Repeated `split_at_mut` hands out one
> `&mut [f64]` per tuple with no aliasing and no raw pointers. **Each output
> element is produced by exactly one tuple — the transform writes, it never
> accumulates — so the split reorders no summation.** Bit-identity holds by
> construction, not by tolerance, and every existing element-by-element parity
> gate is unchanged and green.

Per-worker scratch comes from `rayon`'s per-worker init, so 36-T1's removed
allocation does not come back. The s/p normalization prepass mutates the
Cartesian buffer and stays serial, ahead of the transform.

`CINTX_HOST_TRANSFORM_THREADS` pins the worker count (`1` = serial, the A/B
baseline), mirroring the `CINTX_2E_PER_UNIT` precedent. Setting
`CINTX_HOST_TRANSFORM_PROFILE` forces serial, because a per-block wall-clock
split means nothing summed across racing workers.

**One thing the measurement forced that the plan did not foresee**: parallelising
*hurt* the short lists. After 36-T1 those transforms are a fraction of a
millisecond in total and the fan-out costs more than it saves:

| work list | tuples | serial | parallel |
|---|---:|---:|---:|
| `int2e` CH4/def2-SVP | 14 706 | 1.85 ms | **0.72 ms** |
| `int3c2e` RI-J def2/J | 1 950 | **0.18 ms** | 0.25 ms |
| `int3c2e_ip1` | 1 728 | **0.19 ms** | 0.32 ms |

So the transform stays serial below a measured threshold (4 096 tuples,
`CINTX_HOST_TRANSFORM_MIN_JOBS` to override). A Fock build's real lists are far
above it; the threshold only decides the regime where the answer was
"don't bother".

### Part 1, end to end, vs vendored libcint

| workload | before Part 1 | after Part 1 |
|---|---|---|
| `int2e` CH4/def2-SVP, 14 706 quartets | 0.449 us/quartet, 1.42x faster | **0.343 us/quartet, 1.88x faster** |
| `int2e` H2O/def2-SVP, 3 081 quartets | ~parity | **1.03–1.15x faster** |
| `int3c2e_ip1` | 1.59x slower | **1.28x faster** (52.6x vs per-triple) |
| `int3c2e_ip2` | 1.52x slower | **1.11x slower** (35.4x vs per-triple) |
| RI-J def2/J, 1 950 triples | 0.83 us/triple, ~2.3x slower | **0.498 us/triple, 1.35x slower** |
| RI-J def2/JK, 2 886 triples | 0.63 us/triple, ~1.9x slower | **0.408 us/triple, 1.23x slower** |

`int3c2e_ip1` crossing from 1.59x slower to 1.28x faster is the single largest
move, and it came from the identity-axis skip — not from the launch merging, and
not from the threading.

---

## Part 3 — the small carried-forward items

### 35-F2 — facade parity for the pair/triple batches

`PairBatchRequest` and `TripleBatchRequest` join `QuartetBatchRequest` on the
safe API, returning a shared `ShellListBatchOutput`. Before this, a safe-API
consumer could batch `int2e` and nothing else without depending on the backend
crate, which the project's API ordering says it should not have to.

- **Pairs**: `int1e_{ovlp,kin,nuc}_sph`, `int1e_ip{ovlp,kin,nuc}_sph`,
  `int2c2e_sph`.
- **Triples**: `int3c2e_sph`, `int3c2e_ip{1,2}_sph`.
- Symbol-exact scope, resolved through the compiled manifest, refused **before
  any device work**. Symbol-exact rather than family-wide for the reason
  `QuartetBatchRequest` already is: `int1e_ipovlp_sph` and `int1e_ovlp_sph` share
  a family and are different integrals.
- Five new `CubeClExecutor` methods carry the backend resolution and the f64
  capability check, so no CubeCL type reaches the facade.

**Gate** (`def2_shell_list_batch_facade`, 3 tests): every batched symbol against
vendored libcint over the full def2-SVP list, launch-count assertions, and one
rejection per out-of-scope axis and operator. `xtask manifest-audit --check-lock`
stays `ok`.

### 34-C2 — resident basis for the remaining families

`evaluate_{2c2e_pair,1e_pair,1e_deriv_pair}_batch_resident` added; `int2e` and
`int3c2e` already had theirs. `OneEFlatBasis` and `TwoC2eFlatBasis` are deleted —
they were third and fourth spellings of the four buffers `ResidentBasis` already
holds, and every family now uploads through one path.

**Gate**: the same two-sided assertion, per family — `basis_upload_bytes` is the
full upload on the first call and **0** on every later one, transfer strictly
decreases, **and** every value is bit-identical to the throwaway-residency path.
Either half alone is worthless.

### 34-D2 — primitive screening for the remaining families

`primitive_tolerance` reaches `int3c2e` and the 1e nuclear arm through
`evaluate_3c2e_triple_batch_with` / `evaluate_1e_pair_batch_with` and their
`_resident_with` forms. `TwoEBatchOptions` gains the family-neutral alias
`BatchOptions`, the same arrangement `ResidentBasis` has.

Two things needed care and are recorded in the code:

- The 3c2e test is the 2e one verbatim, on the same `fac1` — the scalar every
  element of the primitive triple's G-tensor is built from.
- **The 1e nuclear `fac1` is negative** (it carries `-Z_C`), so the test is on
  its magnitude. The branch is on values uniform across the cube, so the
  `sync_cube` barriers inside it are still reached by every lane or by none.

**Gate**, in the same commit as the screening: at `primitive_tolerance == 0` the
only primitives dropped are those whose `fac1` underflowed to exactly zero, so
the result is **bit-identical**. A screening bug reads as a speed-up; this
identity is the only thing standing between "faster" and "wrong".

---

## Part 2 — task 35-D, waves 3 and 4 *(13 kernels)*

Both waves converted per the plan's five fixed steps, with the same acceptance
bar: **the per-tuple entry point is rewritten as a one-tuple batch through the
same kernel**, so every existing parity test covers the batched code. Each
conversion also collapsed a five-arm backend `match` into one dispatcher.

### Wave 3 — the 1e gradient/Hessian set

| family | kernel |
|---|---|
| `int1e_ipovlpip` | `one_electron_grad_both_kernel` |
| `int1e_ipkinip` | `one_electron_grad_kin_both_kernel` (8 slabs) |
| `int1e_ipnucip` | `one_electron_nuc_grad_both_kernel` (comptime `nroots`) |
| `int1e_ipipovlp` | `one_electron_gradgrad_bra_ovlp_kernel` |
| `int1e_ipipkin` | `one_electron_gradgrad_bra_kin_kernel` (16 tensors in one slab) |
| `int1e_ipipnuc` / `int1e_ipiprinv` | `one_electron_nuc_gradgrad_bra_kernel` |

### Wave 4 — the 1e special families

| family | kernel | comptime |
|---|---|---|
| `int1e_rinv` | `one_electron_rinv_kernel` | `nroots` |
| `int1e_drinv` | `one_electron_drinv_kernel` | `nroots` |
| `int1e_p4` | `one_electron_p4_kernel` | – |
| `int1e_irp` / `int1e_ipipr` | `one_electron_irp_kernel` | `op_kind` |
| moment (`r`/`rr`/`rrr`/`rrrr`/`r2`/`r4`/`z`/`zz`) | `one_electron_moment_kernel` | `op_mode`, `moment_order`, `rank` |
| GIAO overlap-engine (5 families) | `one_electron_giao_ovlp_kernel` | `op_kind` |
| GIAO nuclear-engine (6 families) | `one_electron_giao_nuc_kernel` | `op_kind`, `rank`, `nroots` |

**The plan's merge-key question, checked per kernel and answered**: for `moment`,
`moment_order` and `rank` are both functions of `op_mode` through
`moment_params`, and for `giao_nuc`, `rank` is a function of `op_kind` — so in
both cases the caller's operator fixes them and `nroots` (where present) is the
only merge key. The plan's suspicion was right.

**Two things the conversions needed that the plan's recipe did not cover:**

- `int1e_irp`, the moment families and both GIAO engines take `drj = rj - origin`,
  which is **per pair**, not per class: the base families measure from a common
  origin and the `_origj` variants from `rj` itself. The host already resolves
  that choice, so the batch carries the resolved vector in a `pair_drj` array
  rather than re-deriving it on device.
- The `#[cube]` recurrence helpers (`d_i_1e_into`, `d_j_1e_into`, `rcj_1e_into`
  and the five `*_flat` tensor helpers) all indexed from zero. Each gained a
  `gbase` parameter, so a slot's slab base threads through them rather than being
  patched at every call site.

A new `one_e_deriv_single_pair_group` helper builds the one-pair group and the
basis residency every converted family's per-tuple entry point needs — the same
twenty lines that would otherwise be copied into each launcher.

---

## Part 2 — task 35-D, wave 5 *(the rest of the per-tuple families)*

Same acceptance bar as waves 3-4: the per-tuple entry point is rewritten as a
one-tuple (or one-row) batch through the same kernel, so every existing parity
test covers the batched code.

### The two genuine scalar families

| family | kernel | shape |
|---|---|---|
| `int3c1e` | `center_3c1e_kernel` | shell triple |
| `int4c1e` | `center_4c1e_kernel` | shell quartet |

Both needed a new launch-group type, and both revealed something the 1e
conversions did not: **their per-tuple path already launched once per
*contraction* tuple**, with the coefficient columns sliced host-side first. So a
row here is a *(shell tuple, contraction tuple)* pair rather than a shell tuple,
which reproduces exactly that arithmetic while collapsing
`nctr_i * nctr_j * nctr_k` (and `* nctr_l`) launches into one.

`center_4c1e` also needed two independent slab strides — the `[gx|gy|gz]`
G-tensor and the 1D/2D polynomial scratch are read at unrelated offsets, so
unlike every other family they do not share a stride. And its host loop *sums*
the contraction blocks rather than scattering them, so the row order is
load-bearing: rows are emitted `ci` outermost, matching the order the sum was
performed in, because any other order would reassociate the additions.

`center_3c1e` also gained a public batched surface —
`evaluate_3c1e_triple_batch{,_resident}` — with the same parallel host
cart-to-sph transform the other triple families use.

### The σ·p relativistic set — 10 kernels

`sigma_p.rs`: `sigma_p`, `sigma_p_cg_sa10sp`, `sigma_p_spgsp`, `sa01_rys`,
`spgnucsp_rys`, `spgsa01_rys`. `sigma_1e_nuc.rs`: `sigma_nuc`,
`sigma_nuc_gauge`. `sigma_1e.rs`: `sigma_ov`. All reuse `OneEDerivLaunchGroup`
and `one_e_deriv_single_pair_group` from wave 3, which is what made ten
conversions tractable.

**One real bug this pass caught, and how.** Rewriting a launcher means rewriting
its comptime dispatch, and two of those `match` arms lost a case: `sigma_ov`'s
four families collapsed to three (so `spsp` launched as `srsr`), and `sigma_p`'s
`tensor_rank` gained a `2` arm the operator never produces. The first showed up
as 24 mismatched elements in `int1e_spsp_spinor` at `nctr > 1` — caught by the
existing vendor gate, on the full-suite run rather than the targeted one. Every
converted launcher's dispatch was then re-checked arm-by-arm against `HEAD`.
**The lesson for the remaining conversions: diff the comptime `match` against
the original, not just the kernel body.**

### Re-scoped, with the reason recorded in the code

- **The two ECP angular kernels.** The original plan called these the cheapest
  conversions in the list; the correction is in the previous report. What *was*
  possible turned out to be worth doing: `rad_ang_all` / `rad_all` are already
  laid out contraction-tuple-major before the loop, so each kernel now takes a
  row table and one dispatch covers all `nci * ncj` tuples where there used to be
  one launch each. The intra-cube split and the accumulation order are untouched,
  so the byte-identity gate holds. Batching across *shell pairs* still needs the
  radial precompute batched, which stays host work.
- **`f12_cart_contraction_kernel` — not converted, deliberately.** The plan asked
  for a check before converting. The check says no: it is launched once per
  *primitive quartet* and its input `g` is produced by the host immediately
  before the launch, so collapsing the launches means materializing every
  `nprim^4` G tensor first — tens of megabytes — while leaving the arithmetic
  that dominates the loop on the host. The conversion worth doing is porting
  `fill_g_tensor_f12` to the device, which is a different and larger task. The
  reasoning is recorded at the call site, not only here.

---

## Part 6 — the f64:f32 ratio on gfx1151 *(the plan's unanswered question)*

New `cintx_cubecl::measure_precision_ratio` and an opt-in oracle test. A
dependent chain of FMAs, one per work item, no memory traffic, both precisions
through the same kernel source and launch geometry.

```
  cpu        8192 items x 4096 FMAs   f64  3.614 ms (  9.28 GFMA/s)   f32  3.647 ms (  9.20 GFMA/s)   ratio  0.99x
  rocm       2048 items x 4096 FMAs   f64  0.318 ms ( 26.41 GFMA/s)   f32  0.089 ms ( 94.10 GFMA/s)   ratio  3.56x
  rocm       8192 items x 4096 FMAs   f64  0.911 ms ( 36.83 GFMA/s)   f32  0.356 ms ( 94.35 GFMA/s)   ratio  2.56x
  rocm      65536 items x 4096 FMAs   f64  6.217 ms ( 43.18 GFMA/s)   f32  1.072 ms (250.36 GFMA/s)   ratio  5.80x
  rocm     262144 items x 4096 FMAs   f64 28.977 ms ( 37.06 GFMA/s)   f32  2.870 ms (374.10 GFMA/s)   ratio 10.10x
```

**gfx1151 is roughly 1:10 for f64:f32 at saturation** — not the 1:16 or 1:32 a
consumer part is often assumed to be, and not the 1:2 of a discrete HPC card.
f64 saturates at ~43 GFMA/s and does not move after 65 k work items; f32 keeps
climbing to ~400 GFMA/s.

The ratio is reported as a **sweep, not a point**, because the two ends mean
different things and an integral kernel lives at both: a short work list is
latency-bound — and a dependent FMA chain is exactly the shape a Rys/VRR
recurrence has — where the ratio is ~2.5–3.5x; a Fock-sized list saturates the
device, where it is ~10x.

**What this settles**: 10x is the ceiling on what precision tiering could recover
on gfx1151, and only on the *arithmetic* share of a kernel — not on its
transcendental calls or its memory traffic. That is a real number to plan
against; it is not the 30x that would have made tiering unavoidable, nor the 2x
that would have made it pointless. No bound is asserted in the test: what the
ratio "should" be is precisely what was unknown, so an assertion would be a guess
dressed as a gate.

---

## Part 4 — groundwork only

`eigh::cint_diagonalize_dev` is factored out of `cint_diagonalize_kernel` as an
inline `#[cube]` callee, with the kernel reduced to a one-line wrapper so both
paths run the same code. That is 33-01's first prerequisite: the extended Rys
path is confined to the host because its solvers are reachable only by
*launching* them, and the Jacobi arm's eigensolve was the piece with no callable
form at all.

**The rest of 33-01 is not done, and the reason is worth recording**: an inline
`rys_roots_dev` covering nroots 6..12 has to reproduce four solver arms (f64
Jacobi, f64 Schmidt, dd Schmidt, dd Jacobi + dd Laguerre), each with its own
constant tables that must arrive as `&Array<f64>` kernel arguments, plus
`segment_solve`'s error fallback — which on device cannot be "call the host".
That is a focused piece of work with its own risk budget, not a tail-end item.

---

## What is not done

- **35-D wave 6** (`unstable-source-api`, 8 sites): the plan asks for an explicit
  decision rather than a drift to the end of the list. On the evidence here the
  answer is **no**: these sit behind a feature gate with no production consumer,
  and waves 3–4 show a conversion costs a day per kernel and buys a launch count
  nobody is paying.
- **Phase 33 (33-01…33-04, 33-06)**: groundwork only, as above.
- **Part 5, device-resident output**: still blocked on open question 2.

---

## Open questions — updated

1. **Which backend is the throughput target?** *Narrowed again, in the CPU's
   favour.* After Part 1 the CPU backend beats vendored libcint on `int2e` at
   both sizes (1.03–1.88x), on `int3c2e_ip1` (1.28x) and on every batched 1e
   gradient (1.21–1.89x); RI-J is within 1.23–1.35x. The host transform is no
   longer the largest cost anywhere — it is 10–30 % of the batched wall-clock.
   Combined with the 1:10 f64 ratio measured on gfx1151, a GPU port is not the
   obvious next lever.
2. **Is device-resident output acceptable?** Unchanged. Still blocks Part 5.
3. **Is def2-ECP (Z >= 37) in scope?** Untouched.
4. ~~def2/J and def2/JK~~ — answered previously.
5. **Does the `hess2e` 1e-12 gate still bind for nroots 6-7 once roots are
   inline?** Still needs 33-01.
6. **Is `unstable-source-api` worth batching?** **Answered: no** — see above.
7. **New — is precision tiering worth pursuing on gfx1151?** The ceiling is 10x
   on arithmetic alone, and an integral kernel's recurrence chains sit nearer the
   2.5–3.5x latency-bound end. Not the first lever to reach for.
