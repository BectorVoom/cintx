# cintx — all remaining tasks

**Status**: the authoritative task list. **Supersedes**
`.planning/notes/post-phase-35-continuation-PLAN.md`, whose Parts 1, 3, 6 and
35-D waves 3–5 are complete.
**Evidence for every "measured" claim**:
`artifacts/post_phase_35_continuation_report_2026-08-25.md`
**Date**: 2026-08-25

---

## 0. Where things stand

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
cargo check --locked --workspace                          ok
(cd xtask && cargo run -- manifest-audit --check-lock)    status: ok, 0 uncovered
ROCm suites (CINTX_ROCM_ORACLE=1, opt-in)                 5 tests, 0 mismatches
```

**Every per-tuple launch site named by task 35-D is batched**, except the two
recorded as deliberate exclusions in §1.

Against vendored libcint 6.1.3, CubeCL CPU backend, single-threaded:

| workload | vs libcint |
|---|---|
| `int2e` CH4/def2-SVP, 14 706 quartets | **1.88x faster** |
| `int2e` H2O/def2-SVP, 3 081 quartets | **1.03–1.15x faster** |
| `int3c2e_ip1` | **1.28x faster** |
| `int1e_ipovlp` / `ipkin` / `ipnuc` | **1.21–1.89x faster** |
| `int3c2e_ip2` | 1.11x slower |
| RI-J `(mu nu \| P)`, def2-SVP + def2/J | 1.35x slower |
| RI-J, def2-SVP + def2/JK | 1.23x slower |

**The host cart-to-sph transform is no longer the bottleneck** — it is 15–30 % of
a batched run after tasks 36-T1/T2. What is left is backend dispatch, which is
what Task A touches.

---

## 1. Decisions already made — do not re-litigate

| Decision | Verdict | Why | Revisit when |
|---|---|---|---|
| **35-D wave 6** (`unstable-source-api`: `grids` ×2, `origi` ×2, `origk` ×2, `ssc`, `breit` — 8 sites, ~2 500 lines) | **No** | Feature-gated, no production consumer. Waves 3–5 cost ~a day per kernel and buy a launch count nobody pays. | One of them gains a consumer. |
| **Batching `f12_cart_contraction_kernel` as-is** | **No** | Launched once per *primitive quartet* with a host-computed `g`; collapsing launches means materializing every `nprim^4` G tensor first. Reasoning is at the call site in `f12.rs`. | Never — the useful work is **Task B** instead. |
| **36-T3, folding c2s into the kernel** | **Defer** | 36-T0 said c2s was 68–81 % of the transform; 36-T1/T2 then cut the transform to 15–30 % of the run. Folding it on-device now removes at most that, plus a readback shrink of `ncart/nsph` (1.2x at `l=2`, 1.5x at `l=4`) — against "large, high risk, every kernel + the output contract". | Only **together with Task D**, and only if open question 2 is answered "device-resident output is acceptable". Planned separately it is not worth its risk. |

---

## Task A — Phase 33: raise the device Rys ceiling *(the largest remaining item)*

**Motivation, and it is a real workload, not a synthetic one**: H2O/def2-**TZVP**
+ def2/J needs `nroots = 6` for `l = (3,3,4)`. The RI-J benchmark reports and
skips it today. `device_nroots_ceiling` returns `BASE_DEVICE_NROOTS = 5`
everywhere.

**Already done**: 33-05 (backend-generic FMA-fusion probe, green on CPU *and*
ROCm), the per-backend ceiling scaffolding
(`crates/cintx-cubecl/src/device_rys_ceiling.rs`, `extended-device-rys` feature
off by default), and `eigh::cint_diagonalize_dev` — the inline `#[cube]` callee
factored out of `cint_diagonalize_kernel`, with the kernel reduced to a one-line
wrapper so both paths run the same code.

### A1 — the inline device Rys entry (33-01)

Add `#[cube] pub(crate) fn rys_roots_ext_dev(...)` in
`crates/cintx-cubecl/src/math/rys_wheeler.rs`, callable from *inside* a family
kernel rather than by launching. It must reproduce `rys_roots_host_wheeler`'s
per-`nroots` dispatch (`rys_wheeler.rs`, `rys_roots_host_wheeler`):

| nroots | `x <= bp` | `x > bp` | bp |
|---|---|---|---|
| 6, 7 | f64 Jacobi | f64 Schmidt | 11 |
| 8 | f64 Jacobi | **dd** Schmidt | 11 |
| 9 | dd Jacobi | dd Laguerre | 10 |
| 10, 11 | dd Jacobi | dd Laguerre | 18 |
| 12 | dd Jacobi | dd Laguerre | 22 |

Each arm already exists as a `#[cube(launch)]` kernel; the work is factoring the
bodies into `#[cube]` callees the way `cint_diagonalize_dev` was:
`jacobi_tridiag_kernel` + `jacobi_transform_kernel`, `schmidt_kernel`,
`lschmidt_kernel`, `ljacobi_tridiag_kernel`, `llaguerre_tridiag_kernel`.

**Three findings that change the shape of this task from what the previous plan
assumed** — all verified against the source:

1. **Only the Jacobi arms need constant tables.** The Laguerre arm builds its
   moments from `t` in-kernel (`llaguerre_moments_dev`), and both Schmidt arms
   take `turnover: f64` as a **scalar** the host reads from
   `TURNOVER_POINT[nroots * 2]`. The tables are
   `JACOBI_{ALPHA,BETA,RN_PART2,SN}` and `LJACOBI_{ALPHA,BETA,RN_PART2,SN}` in
   `math/roots_jacobi_data.rs` — 48/48/88/88 f64 each, **≈ 4.4 KB for all eight**.
2. **Therefore do not grow every family kernel by eight array parameters.**
   Concatenate the eight tables into one `rys_tables: &Array<f64>` uploaded once
   per run, with comptime offsets. A family kernel that opts in then grows
   **one** array argument plus its scratch, not nine.
3. **`segment_solve`'s fallback is "call the host `rys_schmidt`" on error.**
   On device that fallback has to be inline too — the f64 Schmidt arm is already
   the callee it needs, so wire it as the error path rather than inventing one.

**Note on the FMA hazard**: it is narrower than the original plan feared.
`two_prod_dev` asks for an `fma` explicitly and *wants* it; `two_sum_dev`
contains no multiply-add, so contraction cannot reach it. The host
`two_sum`/`two_prod` in `rys_wheeler.rs` are a superseded, unreachable reference
and are annotated as such.

**Acceptance**: compiles under all five backend features; nothing wired into a
family kernel yet; no behaviour change (the feature is off).

### A2 — scratch sizing (33-02)

Route the per-work-item scratch through `shared_memory::calc_math_layout`, which
already has entries for `jacobi_tridiag`, `llaguerre_tridiag`, `schmidt`,
`lschmidt` and `cint_diagonalize`.

**The previous plan's ~3.4 KB/work-item estimate is the *f64 Schmidt* arm only.
Measured from the allocation sites, the dd arms are about twice that**, and they
set the sizing:

| arm | f64 words | ≈ bytes |
|---|---:|---:|
| f64 Schmidt (`fmt_ints` 64, `cs` ≤169, `rt` 12, `acomp` 144, `vscr` 32, `flag` 1) | 422 | 3.4 KB |
| f64 Jacobi + eigensolve | ~365 | 2.9 KB |
| dd Jacobi + eigensolve | ~780 | 6.2 KB |
| **dd Schmidt** (`fmh`+`fml` 128, `csh`+`csl`+`csf` 3×169, `rt` 12, `acomp` 144, `vh`+`vl` 64, `flag` 1) | **856** | **6.8 KB** |

**Consequence to decide before A3**: 6.8 KB × 16 units ≈ **109 KB per launch**,
over the 64 KB shared-memory budget typical on GPU. Either the extended path
drops to global scratch on GPU, or the per-launch unit count drops for the
classes that need it. Bounds-validate at `nroots = 12`.

**Acceptance**: the layout function returns a size that matches what the kernel
indexes, asserted at `nroots = 12`; no behaviour change.

### A3 — accuracy gate (33-04) — **land this before A4**

New oracle test: inline device vs `rys_roots_host_wheeler` over a log-spaced `x`
sweep (1e-8 … 1e6) × `nroots` 6..=12, at the existing oracle tolerance.

**Order matters.** The feature is off by default, so A1 and A2 land additively
and a failing A3 stops the work with the tree still green. Doing A4 first would
put a possibly-wrong path into a family before anything measured it.

**Acceptance**: green across the whole sweep, or a recorded, bounded explanation
of every point that is not.

### A4 — flip `extended-device-rys` per family (33-03)

One family per commit, each gated on its own oracle parity test. **Never several
in one commit.** The ceiling machinery already refuses to raise without both the
feature and a passing probe, and a test asserts that a passing probe *alone* does
not raise it.

Suggested order, by workload weight: `int3c2e` (unblocks def2-TZVP + def2/J,
the motivating case) → `int2e` → `int2c2e` → `int1e_*`.

**Acceptance per family**: its parity test green, and the def2-TZVP + def2/J RI-J
benchmark stops reporting a skip.

### A5 — re-examine the nroots 6–7 host-only decision (33-06)

The current escape hatch routes `nroots` 6,7 through the *host* because a device
kernel **launch** in the family hot path perturbed the host FP environment by
~1e-11 and tripped the flat `atol=1e-12` `hess2e` gate. With the roots inline
there is no launch, so the rationale may no longer hold.

Re-run `hess2e_parity` to decide. **Do not loosen the tolerance** — if it still
binds, the escape hatch stays and that is the answer.

### A6 — what stays blocked

No FMA probe on CUDA / wgpu / Metal, because there is no device on this host
(`xtask wgpu-capability-gate --profiles base` reports
`status=capability-unavailable adapter_found=false`). The per-backend ceiling is
exactly what keeps that **fail-closed** rather than fail-open, so this blocks
nothing else.

---

## Task B — port `fill_g_tensor_f12` onto the device

The residue of wave 5's F12 decision. `fill_g_tensor_f12`
(`kernels/f12.rs`) runs on the host once per primitive quartet, and
`f12_cart_contraction_kernel` is launched once per quartet immediately after.
Porting the fill makes the whole primitive-quartet loop one dispatch and removes
the reason the contraction kernel could not be batched.

**Shape**: the same conversion the wave 3–5 families had — flat basis, a row per
(shell quartet, primitive quartet) or per shell quartet with the primitive loop
in-kernel, slot/lane prologue, per-slot G slab. `F12Shape` already carries the
extents.

**Acceptance**: bit-identity against the per-quartet path, enforced by rewriting
the per-quartet entry as a one-row batch, plus the existing `with-f12` vendor
gates.

**Watch for**: this is a `with-f12` feature-gated module, so `cargo clippy
--workspace --all-targets` does **not** cover it. Run
`cargo clippy -p cintx-cubecl --all-targets --features with-f12 -- -D warnings`
explicitly. (That feature set has pre-existing `excessive_precision` noise on the
frozen literal tables; it is not a gate today.)

---

## Task C — batch the ECP radial precompute

The two angular kernels now take **one dispatch per shell pair** instead of
`nci * ncj`. What remains is batching across *shell pairs*, which needs the
radial precompute batched: `rad_ang_all` in `ecp_type1_cart`
(`kernels/ecp.rs`) and `rad_all` in `ecp_type2_cart`.

This is **host driver work, not a kernel rewrite** — the kernels already index
their input by row. Both drivers build their radial tensor inside a
primitive/level loop with a convergence test (`ecp_type2_cart` iterates to
`LEVEL_MAX` per `(ic, jc, lab)`), so the batching question is whether shell pairs
can share that loop or must each run it to their own convergence.

**Acceptance**: byte-identity on `ecp_iprinv_parity`, `safe_api_ecp_parity` and
`ecp_libecpint_crosscheck_parity`; `ecp_random_rocm_parity` still green.

---

## Task D — device-resident output *(BLOCKED)*

A mode where AO blocks stay on device for a downstream consumer instead of being
read back. **Blocked on open question 2** and unchanged in substance.

One thing has moved *against* it: the host transform is no longer the bottleneck
that motivated it, so the readback-avoidance argument now has to stand on its
own rather than borrowing the transform's cost. If OQ2 is answered
"device-resident is acceptable", plan this **together with 36-T3** (§1) rather
than separately — an on-device c2s and a device-resident spherical output are the
same design.

---

## Task E — GPU execution coverage beyond ROCm *(BLOCKED on hardware)*

- **ROCm is fully covered**: all nine batched families run the cooperative
  (`per_unit == 0`) path and match vendored libcint with 0 mismatches; the
  f64:f32 ratio is measured (~1:10 at saturation, ~1:2.5–3.5 latency-bound).
- **CUDA / wgpu / Metal are compile-only.** No adapter on this host — check with
  `cargo run --manifest-path xtask/Cargo.toml -- wgpu-capability-gate --profiles base`.
- **No GPU throughput claim is made**, deliberately: an integrated GPU at
  ~43 GFMA/s of f64 against a 16-core CPU running libcint.

Nothing to do here until hardware exists. Recorded so it is not mistaken for an
oversight.

---

## Task F — decide the def2-ECP scope *(open question 3, needs an owner)*

`cintx-basis` already builds and parses def2-ECP shells for `Z >= 37`
(`crates/cintx-basis/src/build.rs`). **The integral path is untested for those
elements.** The question is whether heavy-element ECP is in scope or main-group
comes first.

This is a scoping decision, not an engineering task — but it has been carried
unanswered across three plans, and the cost of answering it (extend the ECP
oracle fixture to one `Z >= 37` atom and see what breaks) is small compared to
the cost of discovering the answer later.

---

## Sequencing

```
Task A  Phase 33            A1 → A2 → A3 (gate) → A4 (per family) → A5
   |                        A1/A2/A3 are additive; the feature stays off
   |                        until A3 is green.
   |
Task B  f12 G-tensor on device      ]  independent of A; either order
Task C  ECP radial precompute       ]
   |
Task F  def2-ECP scope decision     [cheap; unblocks nothing but informs C]
   |
Task D  device-resident output   [BLOCKED on OQ2; plan with 36-T3]
Task E  CUDA/wgpu/Metal          [BLOCKED on hardware]
```

---

## Effort and risk

| Task | Size | Risk | Blast radius |
|---|---|---|---|
| A1 inline device Rys | large | **high** | `rys_wheeler.rs` + every family that opts in |
| A2 scratch sizing | medium | medium | shared-memory layout; a GPU budget decision |
| A3 accuracy gate | small | none | reveals, does not change |
| A4 per-family flip | medium | medium | one family per commit |
| A5 re-examine 6–7 | small | low | one decision |
| B f12 G-tensor on device | large | medium | `f12.rs` kernel + primitive loop |
| C ECP radial precompute | medium | medium | host driver in `ecp.rs` |
| D device-resident output | medium | medium | output contract |
| F def2-ECP scope | small | none | a fixture and an answer |

---

## Risks carried forward

| Risk | Impact | Mitigation |
|---|---|---|
| **A rewritten launcher loses a comptime `match` arm** | Silent wrong family; cost us `spsp` launching as `srsr` in wave 5 | **Diff every comptime `match` against the original**, and run the *whole* oracle suite — the targeted test passed while the full suite failed |
| **dd miscompilation (FMA)** at nroots ≥ 6 on an unprobed backend | Silent accuracy loss | Per-backend ceiling + probe, both required; defaults to 5 |
| **Extended-Rys scratch exceeds GPU shared memory** | A GPU launch that cannot be issued, or a silent drop to global | Decide it explicitly in A2 (~6.8 KB/work-item × 16 units ≈ 109 KB) |
| **Shared-buffer scaling across merged classes** | Silent cross-class corruption; has occurred twice | Record a half-open span per class; the bit-identity gate catches it |
| **Scratch slabs of one kernel drifting apart in stride** | Out-of-bounds or cross-slot reads | One stride and one base per kernel, sized to the widest class — except `center_4c1e`, which deliberately carries two |
| **Screening bug reads as a speedup** | Wrong results reported as a win | The `tolerance = 0` identity gate, in the same commit as the screening |
| **Bit-identity across backends is not achievable** | A gate written to demand it fails forever | Established: gate on vendor agreement; eps-of-scale for scalar families and *not* for derivative ones (cancellation) |
| **Feature-gated code escapes the clippy gate** | `with-f12` / `with-4c1e` regressions land unseen | Run clippy explicitly under those features when touching them (Task B) |
| **The parallel-transform threshold is host-specific** | Another machine wants a different number | `CINTX_HOST_TRANSFORM_MIN_JOBS` overrides it; the measurement behind the default is in the doc comment |

---

## Open questions

1. **Which backend is the throughput target?** *Narrowed, in the CPU's favour.*
   The CPU backend now beats vendored libcint on `int2e` at both sizes, on
   `int3c2e_ip1`, and on every batched 1e gradient; RI-J is within 1.23–1.35x.
   With gfx1151 measured at ~43 GFMA/s of f64, a GPU port is not the next lever —
   Task A and the remaining dispatch cost are.
2. **Is device-resident output acceptable** as the primary benchmark mode, or must
   results be host-materialized? **Blocks Task D and shapes 36-T3.**
3. **Is def2-ECP (`Z >= 37`) in scope**, or main-group first? → **Task F.**
4. **Does the `hess2e` 1e-12 gate still bind** for nroots 6–7 once the roots are
   inline? → **A5**, needs A1 first.
5. ~~def2/J and def2/JK~~ — answered: added, parity-gated, benchmarked.
6. ~~Is `unstable-source-api` worth batching?~~ — answered: **no** (§1).
7. ~~Is precision tiering worth pursuing on gfx1151?~~ — answered: the measured
   ceiling is ~10x on arithmetic alone, and a recurrence chain sits nearer the
   2.5–3.5x latency-bound end. **Not the first lever to reach for.**
