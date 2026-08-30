# Numerical Precision Optimization Plan (f64 accuracy / libcint parity)

Status: in progress
Scope: cross-cutting; touches the oracle tolerance model, the Rys/Boys math path, and the backend divergence contract
Primary compatibility target: libcint 6.1.3
Precision in scope: **f64** — the default `PrecisionKind::F64` path
Precision explicitly out of scope: the opt-in `PrecisionKind::F32` path (frozen by PREC-04 / D-08)

## 1. Purpose

cintx already passes its oracle gates. This plan is not about making red gates
green — it is about the fact that **nobody can currently say how much accuracy
margin those green gates are carrying**, and that in three places the margin is
known to be either unmeasured, borrowed from a deliberately degraded reference,
or absent.

The objective is a managed error budget: every integral family, representation,
and Rys order has a *measured* worst-case error, a *justified* tolerance derived
from that measurement, and a *ratchet* that fails when the measurement regresses
— even while the gate still nominally passes.

The optimization order is deliberate:

1. Make the error observable before changing any tolerance.
2. Unify the tolerance model so a single place decides what "parity" means.
3. Ratchet each family down to its measured floor and guard the headroom.
4. Close the one family that is known to be numerically wrong.
5. Establish an accuracy reference independent of the vendored build, so
   "bit-compatible with libcint" and "numerically correct" stop being conflated.
6. Only then move the high-order path onto the device and widen the backend set.

Steps 4-6 are not safely gateable before steps 1-3 exist. Tightening a tolerance
without a measurement is guesswork; moving code across the host/device boundary
without an error budget is how the def2-TZVP Rys-6 class defect got in.

## 2. Sources and constraints

Based on:

- The oracle comparison model in `crates/cintx-oracle/src/compare.rs`.
- The per-family oracle tests under `crates/cintx-oracle/tests/`.
- The Rys/Boys math path in `crates/cintx-cubecl/src/math/` and the per-backend
  ceiling machinery in `crates/cintx-cubecl/src/device_rys_ceiling.rs`.
- The vendored-libcint build configuration in `crates/cintx-oracle/build.rs`.
- The Phase 30 GIAO×σ outcome recorded in `.planning/STATE.md` and
  `crates/cintx-oracle/tests/giao_sigma_1e_parity.rs`.
- The measured reports in `artifacts/` (`def2_throughput_report.md`,
  `def2_remaining_work_report_2026-08-24.md`,
  `post_phase_35_progress_report_2026-08-25.md`,
  `post_phase_35_continuation_report_2026-08-25.md`,
  `simd_cubecl_libcint_3way_parity_report.md`).
- `docs/design/cubecl_math_speed_optimization_plan.md` §5.2, whose numerical
  contract this plan operationalizes rather than replaces.

Binding project constraints:

- The **f32 tolerance model is frozen** (`F32_UNIFIED_RTOL`, `F32_UNIFIED_ATOL`,
  `f32_tolerance_for_family`). This plan does not touch it. Where it shares
  measurement infrastructure, the f64 side is the caller, not the owner.
- Host CPU work stays planning, validation, marshaling, and oracle glue. Any
  arithmetic this plan moves must move *toward* CubeCL, never away from it.
- Deliverables under `/mnt/data` remain mandatory.
- Public library errors use `thiserror` v2; xtask/oracle/bench code uses `anyhow`.
- Fail-closed by construction: no tolerance may be widened, and no
  `oracle_covered` flag flipped, without a recorded measurement and a dated
  justification.

## 3. Current-state diagnosis

### 3.1 Two disjoint tolerance models coexist, and the looser one is undocumented

`tolerance_for_family()` returns the same triple for every family
(`compare.rs:21-23`, `:148-156`):

```
UNIFIED_ATOL    = 1e-12
UNIFIED_RTOL    = 1e-12
ZERO_THRESHOLD  = 1e-18
```

The match arms above it are, by their own comment, "documentation only". So the
manifest-driven fixture sweep holds every family to 1e-12.

The hand-written per-family oracle tests do not use that function. They carry
hardcoded literals — and **the same family is held to different tolerances
inside a single file**, spanning up to eight orders of magnitude:

| File | Applied `atol` values, by line | Spread |
|---|---|---|
| `center_3c1e_parity.rs` | `:298` 1e-7 · `:420` 1e-12 · `:653` 1e-12 | 1e-7 … 1e-12 |
| `center_2c2e_parity.rs` | `:251` 1e-15 · `:310` 1e-9 · `:415` 1e-12 · `:591` 1e-12 | 1e-9 … 1e-15 |
| `one_electron_parity.rs` | `:313,:370,:422` 1e-11 · `:708,:730,:752` 1e-12 | 1e-11 … 1e-12 |
| `two_electron_parity.rs` | `:253` 1e-12 · `:343` 1e-12 | — |
| `grids_random_rocm_parity.rs` | `:185` 1e-11 | — |

In each of the first three files the loose value guards the CPU vendor-parity
test and the tight value guards the ROCm or idempotency arms of the *same
family* — and both pass. That is only possible if the loose bound is not
binding.

`oracle_gate_closure.rs:10-14` records the loose ladder (1e-11 / 1e-9 / 1e-7)
as the intended contract, attributing it to "RESEARCH.md D-06".
`center_3c2e_parity.rs` shows the drift in its clearest form: its module header
still states "atol 1e-9 for 3c2e per phase research D-06" (`:7`) while every
test in the file actually applies 1e-12 (`:236`, `:329`, `:566`), and a comment
at `:227` explains the switch to "the Phase 15 unified atol=1e-12". The header
was never updated. Three problems follow:

- **The ladder and the unified sweep disagree about the same families.** If
  `int3c1e` genuinely needs 1e-7, the unified 1e-12 sweep covering it should be
  failing. If it passes at 1e-12, the 1e-7 literal is stale by five orders.
  Exactly one of those is true and the repository does not say which.
- **The documented contract is provably stale in at least one place** (3c2e),
  which means it cannot be trusted as the source of truth for the others.
- **The ladder is unsourced at the point of use.** A reader of
  `center_3c1e_parity.rs` sees a bare `let atol = 1e-7_f64;` and cannot tell
  whether it encodes a real numerical property of the 3c1e recurrence or a bound
  chosen before a bug was fixed.

Unmeasured headroom is unmanaged risk: a regression that degrades `int3c1e` from
(say) 1e-14 to 1e-8 is a four-order accuracy loss that every gate reports as
green.

### 3.2 There is no error budget

`diff_summary()` computes `max_abs_error` and `max_rel_error` for every
comparison and then discards them everywhere except the parity artifact. No
per-family, per-representation, per-Rys-order record of observed error exists;
no history of it exists; nothing fails when it grows.

Consequently:

- No tolerance change can be justified from data.
- Accuracy regressions *within* tolerance are structurally invisible.
- The f32 model's own procedure ("run the sweep, measure max rel error, set
  rtol = 10x, rounded up" — `compare.rs:189-203`) has been written down for the
  f32 side and never built, which is why its per-family floors are all still the
  `1e-4` placeholder. The f64 side needs the same harness and has less excuse.

### 3.3 One family is known to be numerically wrong

`int1e_spgsa01_spinor` (rank 9, RINV Rys, both-side `g1 = D_J + D_I`, 36-component
gout) builds, runs, and produces all nine non-zero components — with a **~0.5%
uniform residual** in the `D_I`-in-`g1` → `g3` → `g7` chain. It is the only
family reading `g0` at bra `li+3`.

- Gate `#[ignore]`d: `giao_sigma_1e_parity.rs:899-902`.
- `oracle_covered = false`, `unsupported_policy = fail_closed`
  (`crates/cintx-ops/src/generated/api_manifest.rs:7119-7134`).
- Isolation already done (recorded in `.planning/STATE.md`): `D_J`-only gives
  1.2e-2, full gives 6e-5, G-tensor headroom ruled out, transcription verified
  line-by-line against `g2e.c`.

A *uniform* residual at 0.5% is the signature of a missing or misscaled term,
not of accumulated rounding. This is the single genuine correctness defect in the
f64 path; everything else in this plan is margin management.

### 3.4 High-order Rys accuracy is inherited from a deliberately degraded reference

`crates/cintx-oracle/build.rs:173-189` rewrites the vendored config to disable
`HAVE_SQRTL` and `HAVE_QUADMATH_H`. The C `lrys_*` long-double arms therefore
degrade to `c99_sqrtl` (one Babylonian f64-sqrt refinement) and `c99_expl`
(= `exp`). `rys_wheeler.rs` **replicates that degradation** in Rust so the two
agree — the module documents this as the "long-double caveat".

For `nroots` 9..12 the vendored dispatch is entirely `lrys_jacobi` /
`lrys_laguerre` (the table in `rys_wheeler.rs`), so at those orders cintx is
bit-compatible with a reference that is itself running below the precision its
algorithm assumes. The nroots 8..12 sweep validates *agreement*; it validates
nothing about *accuracy*.

This is a defensible compatibility choice — matching upstream-as-built is the
project's stated goal — but the absolute error at high `nroots` is currently
unknown, and a downstream consumer computing `(ff|ff)` integrals has no number
to reason about.

### 3.5 The accurate path and the fast path are different code

`BASE_DEVICE_NROOTS = 5`; `EXTENDED_DEVICE_NROOTS = 12` is reachable only behind
all three of: the `extended-device-rys` feature (off by default), a per-backend
FMA-fusion probe that passed **in this process**, and a per-family opt-in
(`device_rys_ceiling.rs:26-38,:51,:57`). The ceiling "still reads 5 everywhere"
(`post_phase_35_progress_report_2026-08-25.md:165`).

So every shell tuple with `nroots >= 6` runs the host Wheeler/Jacobi path. Two
consequences:

- It violates the architecture constraint that arithmetic lives in CubeCL.
- It creates a numerical seam at `nroots = 5|6`. The def2-TZVP `(1,3,3,3)` /
  `(3,3,1,3)` mismatch was precisely a defect of that class — a host-fallback
  path diverging from the device path — and it was found by a benchmark, not by
  a gate (`def2_throughput_report.md:53-74`).

`rys_roots_host` panics above 12 (`rys.rs:7621-7623`), which is correct — the
vendor has no reference beyond its own quadmath ceiling — but it means the
`nroots > 12` envelope is a hard capability wall, not a precision question.

Note: `.planning/todos/pending/rys-nroots-ge6-wheeler-fallback.md` is partly
stale. Its claim (1) was discharged by Phase 25 (host Wheeler 6..12 landed); its
claim (2), an `ang_momentum > 4` gate in `executor.rs`, no longer matches the
tree — the surviving ceiling is family-scoped (`center_4c1e.rs:982`) plus
`SPHERIC_L_MAX` in `cintx-core/src/error.rs:38`. That todo needs re-triage
before it is cited as evidence for anything.

### 3.6 Cross-backend bit-identity is not achievable, and no divergence budget exists

Measured, not assumed: the ROCm compiler contracts FMA where the CPU compiler
does not, so CPU and ROCm results are not bit-identical. Correctness against
vendored libcint is nonetheless 0 mismatches for all five families tested
(`post-phase-35-remaining-work-PLAN.md`, Part 7).

Two distinct things live here and the repository is careful to separate them —
this plan must stay equally careful:

- The **33-05 double-double hazard** (a backend pre-rounding `a*b` inside
  `two_prod`, silently collapsing 106-bit arithmetic to f64) was *probed and
  discharged on gfx1151*: `fused=true, divergent=0/6` on both CPU and ROCm
  (`post_phase_35_progress_report_2026-08-25.md:156-163`). `two_sum_dev`
  contains no multiply-add, so contraction cannot reach it.
- The **Part-7 FMA contraction** is a different, benign-for-dd effect that
  changes last-bit results across backends.

What is missing is a recorded, gated **per-backend divergence budget**: for each
family and backend pair, the measured worst-case divergence, and a gate that
fails when it grows. Without it, "ROCm differs from CPU in the last bits" and
"ROCm has a numerical bug" are indistinguishable.

### 3.7 A third numerics implementation exists with a divergent constant policy

`cintx-simd` carries its own Boys/Rys transcription. `cintx-simd/src/boys.rs:4`
defines `PIE4` as `std::f64::consts::FRAC_PI_4`; `cintx-cubecl/src/math/rys.rs:35`
transcribes the libcint literal `0.78539816339744827900` and explicitly forbids
using `FRAC_PI_4`, on the grounds that result compatibility is decided by the
exact bits.

Those two happen to be the same f64. The *policy* divergence is the problem: one
port treats constants as mathematical values, the other as transcribed bit
patterns. That difference is invisible where the values coincide and produces
last-bit drift where they do not.

The 3-way parity report gates SIMD/CubeCL/libcint at atol 1e-9..1e-12 depending
on family (`simd_cubecl_libcint_3way_parity_report.md` §3) — the same unmeasured
ladder as §3.1, now spanning three implementations.

### 3.8 `ZERO_THRESHOLD` is inert

The comparison branches (`compare.rs`, `diff_summary`) are:

```
abs_ref < zero_threshold  ->  abs_error <= atol
otherwise                 ->  abs_error <= atol + rtol * abs_ref
```

With `atol = rtol = 1e-12` and `zero_threshold = 1e-18`, the two branches differ
by at most `rtol * abs_ref <= 1e-30` — thirty orders below `atol`, and far below
f64 resolution relative to it. The small-value branch therefore has no effect on
any verdict.

That is not itself a bug, but it means the codebase *appears* to have a
deliberate near-zero policy and does not. Any future change to `atol`/`rtol`
silently changes whether it stays inert. It needs either a real policy or
removal.

## 4. Precision contract (what this plan commits to)

1. **Compatibility first.** Where cintx and vendored libcint 6.1.3 disagree,
   cintx is wrong — including where libcint is less accurate. Absolute-accuracy
   work (Phase 4) *reports*; it never overrides a parity gate.
2. **Every tolerance is derived, dated, and sourced.** No bare numeric literal
   in a test. A tolerance is `measured_max_error x margin`, recorded with the
   commit and corpus that measured it.
3. **Ratchets only tighten.** Loosening requires an explicit exception row with
   an owner and a reason, in one central table, reviewed like an API change.
4. **Measured, not asserted.** No claim about accuracy, FMA behavior, or backend
   divergence enters this repository without the measurement that produced it
   and the host it ran on.
5. **The f64 tolerance model stays authoritative for `oracle_covered`.** The f32
   model (D-09) is additive and never relaxes an f64 flag.
6. **No blanket fast math, reassociation, or reduced precision.** Unchanged from
   `cubecl_math_speed_optimization_plan.md` §5.2.
7. **Fail closed.** An unmeasured family, an unprobed backend, or an unraised
   ceiling keeps the conservative default.

## 5. Work plan

### Phase 0 — Make the error observable (no behavior change)

Deliverable: an **error-budget artifact** emitted by every parity run.

- Extend `FixtureParityResult` / the parity report to persist, per
  `(family, symbol, representation, backend, nroots_class)`: `n_elements`,
  `max_abs_error`, `max_rel_error`, the tolerance actually applied, and
  `headroom = tolerance / observed`.
- Add an `xtask error-budget` command that runs the corpus and writes
  `/mnt/data/cintx_precision_budget.json` plus a human-readable
  `artifacts/precision_error_budget_<date>.md`.
- Instrument the hand-written per-family tests to report through the same sink
  rather than only asserting.

Exit: one artifact that answers "what is the worst observed error for every
family, and how much margin is the gate carrying" — for the first time.
**No tolerance is changed in this phase.**

### Phase 1 — Unify the tolerance model

- Make `tolerance_for_family()` the single source of truth, backed by a real
  per-family table (not documentation-only match arms).
- Replace every hardcoded literal listed in §3.1 with a lookup.
- For each family whose measured error (Phase 0) exceeds the unified 1e-12,
  record an explicit exception row: family, tolerance, measured error, reason,
  owner, date. For each that does not, delete the loose literal outright.
- Resolve the §3.1 contradiction explicitly: state, per family, whether the
  D-06 ladder is a real numerical property or a stale bound.

Exit: `grep` finds no bare tolerance literal in `crates/cintx-oracle/tests/`;
every exception is one row in one table.

### Phase 2 — Ratchet and guard

- Set each family's gate to `measured_max_error x margin` (proposed margin: 10x,
  matching the f32 procedure's shape), floored at 1e-12 for families already
  byte-identical.
- Add a **headroom regression gate**: the run fails if observed error for any
  family exceeds its recorded budget entry, even when still inside tolerance.
- Wire it into the existing `manifest-audit` / oracle CI tier so it gates
  releases the way the manifest lock does.

Exit: an accuracy regression that stays inside tolerance now fails CI.

### Phase 3 — Close `int1e_spgsa01`

The one known-wrong family (§3.3).

- Apply the cart-discriminator dual-verification method (the technique that
  resolved 30-01c, recorded in the `spike-findings-cintx` skill and
  `.planning/notes/phase-30-wave1-engine-class-split-PLAN.md`) to localize the
  0.5% uniform residual in the `D_I`-in-`g1` → `g3` → `g7` chain.
- Prior from the isolation already recorded: a uniform multiplicative residual
  with `D_J`-only at 1.2e-2 and full at 6e-5 points at a term scaling or a
  gauge/center factor in the both-side `g1` composition, not at accumulated
  rounding. Test that hypothesis first; do not start from a rounding analysis.
- Un-`#[ignore]` the gate and flip `oracle_covered = true` **only** on
  byte-identity at atol 1e-12, per the T-30-01d-06 no-over-claim rule.

Exit: 9/9 1e GIAO×σ families byte-identical; the 1e half of GIAO-03 closed.

### Phase 4 — An accuracy reference independent of the vendored build

Separates "bit-compatible with libcint" from "numerically correct" (§3.4).

- Build a host-only, test-only high-precision reference for Rys roots/weights
  and Boys at `nroots` 1..12 (MPFR/`rug`, or the existing double-double host
  path extended), *outside* the shipped crates and outside the timed path.
- Report, per `nroots` and `x` regime, the absolute error of: (a) vendored
  libcint as cintx builds it, (b) cintx's f64 path, (c) the delta between them.
- Publish as `artifacts/rys_absolute_accuracy_<date>.md`. This is a **reporting**
  deliverable — it does not gate, and it never overrides a parity gate (§4.1).
- Expected outcome to confirm or refute: that the `HAVE_SQRTL`/`HAVE_QUADMATH_H`
  disable costs measurable digits at `nroots >= 9`, and how many.

Exit: a number a downstream consumer can cite for high-`nroots` accuracy.

### Phase 5 — Unify the high-order path onto the device

Under the budget from Phases 0-2, and only after them.

- Land the Phase 33 inline device Rys work (33-01..33-04) so `nroots` 6..12
  executes in CubeCL, removing the host/device numerical seam of §3.5.
- Gate each family's ceiling raise on: the per-backend FMA-fusion probe, a green
  per-family oracle parity test, **and** an error-budget entry showing no
  headroom loss versus the host path.
- Keep `device_rys_ceiling`'s fail-closed structure exactly as built — the raise
  is per family, per backend, and never a side effect of a passing probe.
- Answer the standing open question 5 with data: does the `hess2e` 1e-12 gate
  still bind at `nroots` 6-7 once roots are inline?

Exit: the def2-TZVP + def2/J envelope runs on device with no accuracy loss
recorded in the budget.

### Phase 6 — Per-backend divergence budget

- Extend the budget artifact with a backend axis: for every family, record
  divergence for each available backend pair (cpu/wgpu/cuda/rocm/metal).
- Gate on *growth*, not on zero — bit-identity across backends is not achievable
  (§3.6) and the plan must not pretend otherwise.
- Keep the 33-05 probe result and the Part-7 contraction effect recorded as the
  two separate phenomena they are; a single "FMA" line item would lose the
  distinction that makes the dd path safe on gfx1151.

Exit: "ROCm differs in the last bits" is a budgeted, bounded, monitored fact
rather than an anecdote.

### Phase 7 — Reconcile the third implementation and freeze constant provenance

- Write down one constant-provenance policy: precision-critical constants are
  transcribed from vendored libcint verbatim, never recomputed from `std`
  consts, in **every** implementation.
- Apply it to `cintx-simd` (starting with `PIE4`, `boys.rs:4`) and add a test
  that pins each transcribed constant's bit pattern to the vendored source.
- Fold the 3-way parity report onto the unified tolerance table from Phase 1.

Exit: three implementations, one constant policy, one tolerance table.

### Phase 8 — Special-value and small-value policy

- Decide `ZERO_THRESHOLD`'s fate (§3.8): give it a real policy (e.g. an
  absolute-only regime meaningfully above `atol`) or delete it and its branch.
- Audit NaN/Inf/subnormal handling on both host and device paths; add
  predecessor/exact/successor boundary tests at each branch point in the Rys
  domain segmentation (`x = 3e-7`, `x = 15`, `x = 33`, the per-`nroots`
  breakpoints 10/11/18/22, and `35 + nroots*5`).
- Audit screening thresholds (`nonzero_threshold` and friends) for the f64 path
  specifically — they are currently written as an `F::PRECISION`-conditional and
  the f64 arm has never been justified independently of the f32 arm.

Exit: every numerical branch boundary in the f64 path has a test standing on it.

## 6. File-level change map

| Area | Files | Phase |
|---|---|---|
| Tolerance model, diff summary, budget sink | `crates/cintx-oracle/src/compare.rs` | 0,1,2,8 |
| Per-family gates (literal removal) | `crates/cintx-oracle/tests/{center_3c1e,center_2c2e,center_3c2e,one_electron,two_electron,oracle_gate_closure}_parity.rs` | 0,1 |
| Budget command + artifacts | `xtask/src/` (new `error_budget` module) | 0,2,6 |
| GIAO×σ rank-9 chain | `crates/cintx-cubecl/src/kernels/` (sigma_p / GIAO nuclear engine), `crates/cintx-oracle/tests/giao_sigma_1e_parity.rs`, `crates/cintx-ops/src/generated/api_manifest.rs` | 3 |
| High-precision reference (test-only) | new dev-dependency crate or `cintx-oracle` dev target | 4 |
| Device Rys 6..12 | `crates/cintx-cubecl/src/math/{rys,rys_wheeler}.rs`, `device_rys_ceiling.rs` | 5 |
| Backend divergence | `device_rys_ceiling.rs`, budget artifact, ROCm/wgpu oracle tests | 6 |
| Constant provenance | `crates/cintx-simd/src/boys.rs`, `crates/cintx-cubecl/src/math/rys.rs` | 7 |
| Branch boundaries | `crates/cintx-cubecl/tests/{boys,rys}_tests.rs`, `crates/cintx-oracle/tests/rys_nroots_sweep_parity.rs` | 8 |

## 7. Verification

Required commands per phase (all with the existing vendor gate):

```
cargo test --workspace --exclude cintx-oracle
CINTX_ORACLE_BUILD_VENDOR=1 cargo test --release -p cintx-oracle --features cpu
xtask manifest-audit --check-lock
xtask error-budget                          # new in Phase 0
cargo check -p cintx-cubecl --no-default-features --features {cpu,wgpu,cuda,rocm,metal}
```

Phase-specific gates:

- **Phase 0**: the budget artifact exists and is non-empty for every family in
  the manifest lock. No test verdict changes.
- **Phase 1**: no bare tolerance literal remains in the oracle test tree; every
  exception row has owner + date + measured value.
- **Phase 2**: an injected 10x accuracy regression in any family fails CI while
  still passing the raw tolerance assertion. Prove this with a deliberate
  temporary perturbation, then revert it.
- **Phase 3**: `giao_sigma_1e_parity` runs un-`#[ignore]`d, 0 mismatches at
  atol 1e-12; `manifest-audit` green with `int1e_spgsa01_spinor`
  `oracle_covered = true`.
- **Phase 4**: report artifact only; no gate change.
- **Phase 5**: per-family oracle parity green on device at raised ceiling, and
  budget headroom not worse than the host path's recorded entry.
- **Phase 6**: divergence budget populated for every backend present on the
  runner; absent backends recorded as unmeasured, never as zero.
- **Phase 7**: constant bit-pattern test passes across `cintx-simd` and
  `cintx-cubecl`.
- **Phase 8**: boundary tests present at every listed branch point.

## 8. Required artifacts

- `/mnt/data/cintx_precision_budget.json` — machine-readable budget, per
  `(family, symbol, representation, backend, nroots_class)`.
- `artifacts/precision_error_budget_<date>.md` — human-readable snapshot.
- `artifacts/rys_absolute_accuracy_<date>.md` — Phase 4 reference report.
- `artifacts/backend_divergence_<date>.md` — Phase 6.
- An updated tolerance-exception table checked into the repository (proposed:
  `docs/design/precision_tolerance_table.md`), referenced from `compare.rs`.

## 9. Safety and failure invariants

- No `oracle_covered` flag flips without byte-identity at the family's recorded
  tolerance; no over-claim (T-30-01d-06).
- No tolerance widens without an exception row.
- A missing measurement fails closed: unmeasured family → unified 1e-12;
  unprobed backend → base ceiling; unraised family → host path.
- Allocation failure keeps returning a typed error with no partial writes.
- The f32 model is not modified by any phase here.

## 10. Suggested sequence

Phases 0 → 1 → 2 are strictly ordered and should land before anything else;
they are also the cheapest. Phase 3 is independent of them and can run in
parallel — it is a correctness fix, not a margin question, and it is the only
item here that changes what cintx can claim to support. Phase 4 is independent
and reporting-only. Phases 5 and 6 both depend on 0-2. Phases 7 and 8 are
independent cleanups that benefit from 1.

Recommended first slice: **Phase 0 plus Phase 3**, since Phase 0 unblocks
everything downstream and Phase 3 closes the last open f64 defect.

## 11. Definition of done

1. Every family in the manifest lock has a measured error entry, a derived
   tolerance, and a headroom guard.
2. No bare tolerance literal exists in the oracle test tree; every exception is
   one dated, owned row in one table.
3. `int1e_spgsa01_spinor` is byte-identical and `oracle_covered = true`.
4. The absolute accuracy of the Rys path at `nroots` 1..12 is a published
   number, separate from the parity verdict.
5. `nroots` 6..12 executes on device for at least the def2-TZVP + def2/J
   envelope, with no headroom loss.
6. Per-backend divergence is budgeted and gated on growth.
7. All three implementations share one constant-provenance policy.
8. Every numerical branch boundary in the f64 path carries a boundary test.

## 11.1 Implementation record (2026-08-30)

Verified in scope:

- Phases 0--2 have a checked-in CPU error-budget baseline, the `xtask
  error-budget` producer/ratchet, and release/PR CI wiring. The baseline is
  deliberately only a CPU measurement; absent backends are not represented as
  zero.
- Phase 3 is closed: `int1e_spgsa01_spinor` is unignored, passes the vendor
  gate at `1e-12`, and is marked `oracle_covered`.
- Phase 4 has a reproducible 100-decimal-digit mpmath reference generator and
  reporting command. Its first report found and the implementation fixed the
  unported `x <= 3e-7` table branch for nroots 6--12. The report remains
  reporting-only, as required.
- Phase 7 has one exported, verbatim `PIE4` bit pattern plus a cross-crate
  bit-pattern test. Phase 8 removes the inert f64 near-zero branch and adds
  predecessor/exact/successor Rys segmentation tests.

Not yet verified:

- Phase 5's full device-envelope exit remains open: the existing inline
  extended path is feature/probe/family gated, and the nroots 6--7 CPU-runtime
  launch escape hatch means this document cannot claim all nroots 6--12 execute
  on device.
- Phase 6 requires measurements on each available non-CPU backend. This runner
  has only produced the CPU baseline, so no backend-pair divergence budget has
  been claimed.
- Phase 8's Rys boundary coverage is present; the separate NaN/Inf/subnormal
  and f64-screening audits remain open.

## 12. Verified facts and open hypotheses at plan time

**Verified in-tree (2026-08-30, branch `d-pbc-24-range-omega`):**

- `UNIFIED_ATOL = UNIFIED_RTOL = 1e-12`, `ZERO_THRESHOLD = 1e-18`; the
  `tolerance_for_family` match arms are documentation-only (`compare.rs:21-23`,
  `:130-156`).
- The per-family `atol` literals of §3.1 exist at the cited lines, including the
  same-file spread (3c1e 1e-7 vs 1e-12; 2c2e 1e-15 / 1e-9 / 1e-12) and the stale
  3c2e module header (`center_3c2e_parity.rs:7` says 1e-9; the file applies
  1e-12).
- `int1e_spgsa01_spinor` gate is `#[ignore]`d with the recorded ~0.5% residual
  (`giao_sigma_1e_parity.rs:899-902`); its manifest row is `fail_closed`
  (`api_manifest.rs:7119-7134`).
- `HAVE_SQRTL` and `HAVE_QUADMATH_H` are disabled in the vendored build
  (`crates/cintx-oracle/build.rs:173-189`); `rys_wheeler.rs` documents matching
  that degradation.
- `BASE_DEVICE_NROOTS = 5` (`device_rys_ceiling.rs:51`),
  `EXTENDED_DEVICE_NROOTS = 12` (`:57`), triple fail-closed guard documented at
  `:26-38`; `rys_roots_host` dispatches 6..=12 to
  the host Wheeler path and panics above 12 (`rys.rs:7621-7623`).
- The 33-05 probe measured `fused=true, divergent=0/6` on both CPU and ROCm
  (`post_phase_35_progress_report_2026-08-25.md:156-163`).
- `cintx-simd/src/boys.rs:4` uses `FRAC_PI_4`; `cintx-cubecl/src/math/rys.rs:35`
  transcribes the libcint literal and forbids `FRAC_PI_4`.
- The `rys-nroots-ge6-wheeler-fallback` todo's `executor.rs` `l > 4` gate claim
  does not match the current tree.

**Open hypotheses — to be settled by Phase 0, not assumed:**

- That the D-06 tolerance ladder is stale rather than a real numerical property
  for at least `int3c1e` (1e-7) and `int2c2e`/`int3c2e` (1e-9). Most families
  are believed to sit far inside 1e-12, but this is inference from the unified
  sweep passing, not measurement.
- That the `spgsa01` residual is a term-scaling or center/gauge factor rather
  than accumulated rounding. Supported by its uniformity and by the recorded
  isolation, not yet proven.
- That the `HAVE_SQRTL` disable costs measurable digits at `nroots >= 9`.
  Plausible from the algorithm; unmeasured.
- That moving `nroots` 6..12 to the device preserves current accuracy. The
  budget from Phases 0-2 is what makes this checkable rather than hopeful.
