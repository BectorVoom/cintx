# Remaining work — after the 2026-08-25 continuation *(second pass)*

> **SUPERSEDED (2026-08-25)** by
> `.planning/notes/post-wave5-remaining-tasks-PLAN.md`, which is the
> authoritative task list. This file is kept for the record of what waves 3–5
> found; act on the newer plan, not this one.

**Status**: what is left of the original continuation plan
**Predecessor**: that plan's Part 1, Part 3, Part 6 and 35-D waves 3–5 are complete
**Evidence for every "measured" claim below**:
`artifacts/post_phase_35_continuation_report_2026-08-25.md`
**Date**: 2026-08-25

---

## 0. Where things actually stand

```
cargo fmt --all --check                                   clean
cargo clippy --workspace --all-targets -- -D warnings     clean (also under
                                                          CINTX_ORACLE_BUILD_VENDOR=1)
cargo test --workspace --exclude cintx-oracle             648 passed, 0 failed
CINTX_ORACLE_BUILD_VENDOR=1 cargo test --release
  -p cintx-oracle --features cpu                          375 passed, 0 failed
cargo check -p cintx-cubecl --no-default-features
  --features {cpu,wgpu,cuda,rocm,metal}                   all five compile
cargo check -p cintx-cubecl --all-targets
  --features {with-4c1e, with-f12, both}                  all three compile
(cd xtask && cargo run -- manifest-audit --check-lock)    status: ok, 0 uncovered
ROCm suites (CINTX_ROCM_ORACLE=1, opt-in)                 5 tests green, 0 mismatches
```

**Every per-tuple launch site named by task 35-D is now batched**, except the two
described under Part B below.

Throughput on the CubeCL CPU backend vs vendored libcint 6.1.3, single-threaded,
**after Part 1**:

| family | state | vs libcint |
|---|---|---|
| `int2e` CH4/def2-SVP, 14 706 quartets | 15 dispatches (of 69 classes) | **1.88x faster** |
| `int2e` H2O/def2-SVP, 3 081 quartets | 15 dispatches | **1.03–1.15x faster** |
| `int3c2e_ip1` | 4 dispatches (of 27) | **1.28x faster** |
| `int3c2e_ip2` | 4 dispatches (of 27) | 1.11x slower |
| `int1e_ipovlp` / `ipkin` / `ipnuc` | 1 / 1 / 3 dispatches | **1.21–1.89x faster** |
| RI-J `(mu nu \| P)`, def2-SVP + def2/J | 5 dispatches (of 45) | 1.35x slower |
| RI-J, def2-SVP + def2/JK | 5 dispatches (of 45) | 1.23x slower |

### 0.1 The number that shaped the last pass, and the one that shapes this one

The previous plan was written around "the serial host cart-to-sph transform is
now the largest single cost". **It no longer is.** 36-T0 measured the transform's
internals and found the c2s arithmetic at 68–81 %, not allocation; removing the
identity-axis work (`l <= 1` axes are identity matrices) cut the serial transform
6x on the 3-index families, and parallelising cut the large 2e list a further
2.6x. The transform is now 10–30 % of a batched run.

**What is left is backend dispatch.** That is where the remaining gap to libcint
lives on RI-J and `int3c2e_ip2`, and it is the thing Phase 33's ceiling raise and
any GPU work would touch.

---

## Part A — Phase 33, the ceiling raise *(now the largest remaining item)*

Still motivated by a real blocked workload: H2O/def2-**TZVP** + def2/J needs
`nroots = 6` for `l = (3,3,4)` and the RI-J benchmark reports and skips it.

**Done**: 33-05 (backend-generic FMA probe, green on CPU and ROCm), the
per-backend ceiling scaffolding, and — new in this pass —
`eigh::cint_diagonalize_dev`, the inline `#[cube]` callee factored out of
`cint_diagonalize_kernel` with the kernel reduced to a one-line wrapper.

**33-01 — the inline device Rys entry.** The remaining work, and the shape it
actually has:

- `rys_roots_host_wheeler` dispatches four solver arms across nroots 6..12:
  f64 Jacobi (6,7,8 at `x <= 11`), f64 Schmidt (6,7 at `x > 11`), dd Schmidt
  (8 at `x > 11`), dd Jacobi + dd Laguerre (9..12, breakpoints 10/18/22).
  An inline entry needs all four as `#[cube]` callees.
- Each arm's constant tables (`JACOBI_ALPHA`, `JACOBI_BETA`, `JACOBI_RN_PART2`,
  `JACOBI_SN`, the Laguerre tables) are host arrays today and must arrive as
  `&Array<f64>` kernel arguments — which means every family kernel that opts in
  grows those parameters.
- `segment_solve`'s error fallback is "call the host `rys_schmidt`". On device
  that fallback has to be inline too.
- The Jacobi arm's eigensolve was the piece with **no** callable form; that is
  what `cint_diagonalize_dev` now provides.

**33-02** — scratch through `shared_memory::calc_math_layout`: `fmt_ints` 64,
`cs` `(nroots+1)^2 <= 169`, `rt` 12, `acomp` 144, `vscr` 32, `flag` 1 — ~3.4 KB
of f64 per work-item. Bounds-validate at nroots=12. Interacts with the merged
per-slot slab: 3.4 KB x 16 units is 54 KB per launch, fine on CPU and a real
shared-memory decision on GPU.

**33-03** — flip `extended-device-rys` **per family**, each on its own oracle
parity test. Never in one commit. The ceiling machinery already refuses to raise
without both the feature and a passing probe.

**33-04** — accuracy gate: inline device vs `rys_roots_host_wheeler` over a
log-spaced `x` sweep (1e-8 … 1e6) x nroots 6..12 at the oracle tolerance. **Land
this before 33-03**: the feature is off by default, so 33-01/33-02 can go in
additively and a failing 33-04 stops the work with the tree still green.

**33-06** — re-examine the nroots 6–7 host-only decision once roots are inline.
The original rationale was launch-induced FP-environment perturbation; with no
launch it may no longer hold. Re-run `hess2e_parity` to decide. **Do not loosen
the tolerance.**

**Note for whoever does 33-01**: the FMA hazard 33-05 guards is narrower than
feared. `two_prod_dev` asks for an `fma` explicitly and wants it; `two_sum_dev`
contains no multiply-add, so contraction cannot reach it. The host
`two_sum`/`two_prod` in `rys_wheeler.rs` are a superseded, unreachable reference.

**Blocked on**: nothing, for CPU and ROCm. Still no probe on CUDA / wgpu / Metal
— no device on this host, and the per-backend ceiling is what keeps that
fail-closed.

---

## Part B — 35-D wave 5 *(done, except one deliberate exclusion)*

**Converted**: `center_3c1e`, `center_4c1e`, and the ten σ·p kernels
(`sigma_p`, `sigma_p_cg_sa10sp`, `sigma_p_spgsp`, `sa01_rys`, `spgnucsp_rys`,
`spgsa01_rys`, `sigma_nuc`, `sigma_nuc_gauge`, `sigma_ov`). The two
`ecp_*_angular` kernels are batched over their contraction tuples.

`center_3c1e` also gained `evaluate_3c1e_triple_batch{,_resident}`.

### B.1 What is deliberately **not** converted

**`f12_cart_contraction_kernel`.** The plan asked for a check before converting;
the check says no, and the reasoning is recorded at the call site in `f12.rs`.
It is launched once per *primitive quartet* with a host-computed `g`, so
collapsing the launches means materializing every `nprim^4` G tensor first —
tens of megabytes — while leaving the dominant arithmetic (the G fill) on the
host. The conversion worth doing is porting `fill_g_tensor_f12` onto the device
so the whole primitive-quartet loop becomes one dispatch. **That is the item to
pick up, not the contraction kernel's signature.**

### B.2 What is left in the ECP path

The two angular kernels now take one dispatch per shell pair instead of
`nci * ncj`. Batching across *shell pairs* additionally needs the radial
precompute (`rad_ang_all` / `rad_all`) batched — host work in `ecp_type1_cart` /
`ecp_type2_cart`, not a kernel rewrite.

### B.3 A lesson worth carrying into Part A

Rewriting a launcher means rewriting its comptime `match`, and that is where
this wave's only real bug came from: `sigma_ov`'s four-family dispatch lost a
case, so `spsp` launched as `srsr`. It surfaced as 24 mismatched elements at
`nctr > 1` in the vendor gate — on the full-suite run, not the targeted one.
**Diff every comptime `match` against the original, and run the whole suite
before believing a family is done.**

## Part C — 35-D wave 6 *(decided: no)*

`grids` (2), `origi` (2), `origk` (2), `ssc` (1), `breit` (1) — 8 sites,
~2 500 lines behind `unstable-source-api`, with no production consumer.

**Decision, on the evidence of waves 3–4: do not convert them.** A conversion
costs roughly a day per kernel and buys a launch count nobody is paying. Revisit
only if one of these gains a consumer.

---

## Part D — device-resident output *(still blocked)*

Unchanged: a mode where AO blocks stay on device for a downstream consumer rather
than being read back. Still blocked on open question 2.

One thing has changed against it: the host transform is no longer the bottleneck
Part 1 found it to be, so the readback-avoidance argument has to stand on its own
rather than borrowing the transform's cost.

---

## Part E — GPU verification beyond correctness

- **The f64:f32 ratio on gfx1151 is measured**: ~1:10 at saturation (f64 saturates
  at ~43 GFMA/s, f32 reaches ~400), ~1:2.5–3.5 in the latency-bound regime a
  recurrence chain actually occupies. `cintx_cubecl::measure_precision_ratio` and
  `rocm_precision_ratio` carry it. **The carried-forward risk is discharged.**
- **No CUDA / wgpu / Metal execution anywhere.** Compile-only, as before.
- **No GPU throughput claim**, deliberately, and now with a number behind the
  reticence: an integrated GPU at 43 GFMA/s of f64 against a 16-core CPU running
  libcint.

---

## Sequencing

```
Part A  Phase 33     [the remaining item; land 33-04 before 33-03]
   |
Part B  residue      [f12 G-tensor on device; ECP radial precompute]
   |
Part D  device-resident output   [BLOCKED on open question 2]

Part C  wave 6       [decided: no]
```

---

## Effort and risk

| Task | Size | Risk | Blast radius |
|---|---|---|---|
| 33-01 inline device Rys | large | **high** | rys + every family that opts in |
| 33-02 scratch sizing | medium | medium | shared-memory layout |
| 33-03 per-family flip | medium | medium | one family per commit |
| 33-04 accuracy gate | small | none | reveals, does not change |
| 33-06 re-examine 6–7 | small | low | one decision |
| f12 G-tensor on device | large | medium | f12 kernel + primitive loop |
| ECP radial precompute batching | medium | medium | host driver, not the kernel |
| device-resident output | medium | medium | output contract |

---

## Risks carried forward

| Risk | Impact | Mitigation |
|---|---|---|
| **dd miscompilation (FMA)** at nroots >= 6 on an unprobed backend | Silent accuracy loss | Per-backend ceiling + probe, both required; defaults to 5 |
| **Shared-buffer scaling across merged classes** | Silent cross-class corruption; has occurred twice | Record a half-open span per class; the bit-identity gate catches it |
| **Scratch slabs of one kernel drifting apart in stride** | Out-of-bounds or cross-slot reads | One stride and one base per kernel, sized to the widest class |
| **Screening bug reads as a speedup** | Wrong results reported as a win | The `tolerance = 0` identity gate, in the same commit |
| **Bit-identity across backends is not achievable** | A gate written to demand it fails forever | Established: gate on vendor agreement; eps-of-scale for scalar families and *not* for derivative ones |
| **A parallel transform read as a loosened gate** | Reviewer distrust | The disjoint-output argument is in `transform::host_batch`'s module docs, not only in the code |
| **The parallel-transform threshold is host-specific** | A different machine wants a different number | `CINTX_HOST_TRANSFORM_MIN_JOBS` overrides it; the measurement behind the default is in the doc comment |

---

## Open questions

1. **Which backend is the throughput target?** *Narrowed again, in the CPU's
   favour.* The CPU backend now beats vendored libcint on `int2e` at both sizes,
   on `int3c2e_ip1`, and on every batched 1e gradient; RI-J is within 1.23–1.35x.
   With gfx1151 measured at ~43 GFMA/s of f64, a GPU port is not the obvious next
   lever — Phase 33 and the remaining dispatch cost are.
2. **Is device-resident output acceptable** as the primary benchmark mode, or must
   results be host-materialized? **Still blocks Part D.**
3. **Is def2-ECP (Z >= 37) in scope**, or main-group first? Untouched.
4. ~~def2/J and def2/JK~~ — answered.
5. **Does the `hess2e` 1e-12 gate still bind** for nroots 6–7 once roots are
   inline? Needs 33-01.
6. ~~Is `unstable-source-api` worth batching at all?~~ — **answered: no** (Part C).
7. **Is precision tiering worth pursuing on gfx1151?** The measured ceiling is
   ~10x on arithmetic alone, and a recurrence chain sits nearer the 2.5–3.5x
   latency-bound end. Not the first lever to reach for.
