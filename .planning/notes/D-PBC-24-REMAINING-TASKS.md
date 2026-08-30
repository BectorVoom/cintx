---
carryover: D-PBC-24
title: "Remaining tasks after D-PBC-24 stages 0–3"
parent: ".planning/notes/D-PBC-24-cintx-range-omega-PLAN.md"
recorded: 2026-08-30
updated: 2026-08-30
status: PARTIALLY DONE — see "Status as of 2026-08-30 (second pass)" below
repos:
  cintx: /home/user/Documents/workspace/cintx
  pyscf_rs: /home/user/Documents/workspace/pyscf_rs
---

# D-PBC-24 — what is left

## Status as of 2026-08-30 (second pass)

Worked in the order this file's own §"Suggested order" gives.

| item | state | where |
|---|---|---|
| **P0** — batch surfaces fail closed | **DONE** | cintx `d-pbc-24-range-omega`; `range_omega_batch_scope.rs` (5 tests) |
| **Commit the cintx work** | **DONE** | cintx `74bec6c` |
| **P3-1** — ω into `aux_e2` / `fill_2c2e` / `pbc_intor` | **DONE** | pyscf_rs `d-pbc-24-stage-5-omega`; `tests/incore.rs` gates `SR + LR == full` on the assembled tensor |
| **P3-8** — Phase 4's RSH assertion | **DONE**, and it was worse than recorded | pyscf_rs; see below |
| **P2-1** — derivative rows | **DONE** for the `ip` family | cintx; 18 rows vs vendor, worst \|diff\| 2.776e-15 |
| **P2-3** — f32 under ω | **DONE** | `range_omega_safe_api_roundtrip.rs` |
| **P2-4** — `int2e_spinor` under ω | **DONE** | `range_omega_parity.rs`, vendor-gated |
| **P4** — `orig{i,k}_genctr_parity` module gate | **DONE** | one line each |
| **P1** — device omega arms | **NOT STARTED** | unchanged; see below |
| **P2-2** — `int1e_grids` | **DECIDED: no** — leave it refused | no consumer exists; see below |
| **P3-2 … P3-7** — `_RSGDFBuilder`, `_RSMDFBuilder`, `RSDF`, `rsjk`, Gate 3 | **NOT STARTED** | the blocker is gone; the port is not written |

**Two corrections to this file, found while doing the work.**

1. **P3-1's ⚠️ is answered: `aux_e2` uses the SCALAR path**, not
   `TripleBatchRequest`/`PairBatchRequest`. `incore/int3c.rs::eval3c` builds a
   plain `SessionRequest` per triple. So P0 was never a prerequisite for stage
   5 — though it was still a live silent-substitution path for anyone who did
   batch.

2. **P3-8 understated its own severity.** This file says "that slot is now read
   by cintx's raw path, so the contract test can become a numerical one". The
   raw path does read it — but `pyscf_gto::intor` does not go through the raw
   path: it builds cintx's `BasisSet` from `mol._atom`/`_basis` and calls the
   SAFE API, which takes its parameters from `ExecutionOptions` and never sees
   the caller's `_env`. So `intor_with_omega` and `get_k_with_omega` were not
   merely untested — they were returning **full-range** integrals under a
   range-separated name, which is the exact failure this whole carryover exists
   to prevent. The fix was an `intor_with_options` seam, not a test.

**P2-1 came in far cheaper than the 2–4 days estimated here**, for a reason
worth recording: §"Priority 1" is right that the DEVICE kernels have no omega
branch, but almost every derivative launcher was already a HOST path
(`launch_two_electron_ip1`'s own comment says so). So step 4 of P2-1's recipe
("route each to the host under a set ω") was already true for all but two rows.
The exceptions were `int3c2e_ip1`/`ip2`, whose whole per-triple core runs on
`center_3c2e_{ip1,ip2}_kernel`; they got a host arm
(`host_3c2e_deriv_cart_blocks`), the same move the scalar `int3c2e` made in
stage 2.

**What P2-1 deliberately did NOT widen:** the GIAO/gauge rows (`g1`, `gg1`,
`g1g2`, `ig1`, `ipvg{1,2}_xp1`, `ip1v_r{c,}1`) and the relativistic σ·p/σ·r
spinor rows. They read `env[8]` upstream and their launchers are host-routed,
so they would very likely just work — but nothing gates them under a set ω, and
`supports_range_omega` is now defined as exactly what
`range_omega::derivative_headroom` covers, which is exactly what the vendor
sweep compares. Widening the scope past the gate is how a full-range substitute
ships.

**What is genuinely left**, in the order this file already recommends:

* **P3-2 … P3-7** — the value. `_RSGDFBuilder`'s `get_2c2e` / `outcore_auxe2` /
  `add_ft_j3c` / `solve_cderi` / `_RSNucBuilder`, then `_RSMDFBuilder`, then
  `Gdf::prefer_ccdf = false`, then `rsjk`, then Gate 3. Every integral they
  need now exists and is gated; what is missing is the several hundred lines of
  `rsdf_builder.py` / `rsjk.py` that assemble them. The refusals and their
  tests are still in place and their MESSAGES have been rewritten to say so —
  `CINTX_SR_GAP` is now `RS_BUILDER_GAP` — because leaving them naming a cintx
  gap that no longer exists would send the next reader to re-derive it.
* **P1** — the device omega arms. **Deliberately not started**, on this file's
  own gating: "P1-a — long range on-device, *if stage 5 finds the host route
  too slow*". Stage 5 has measured nothing yet, because P3-2 … P3-7 are not
  written, so doing P1-a now would be optimising against no evidence. The
  analysis in §"Priority 1" still holds unchanged, except that P2-1 added
  `int3c2e_ip1`/`ip2` to the list of families with a host arm to fall back to.
* **P2-2** — `int1e_grids`. **Decided: leave it refused.** The task was "decide
  whether any consumer needs it before spending anything here", and the answer
  is no: the only mention of `int1e_grids` anywhere in `pyscf_rs` is a
  `layout_table.rs` row (`pyscf-gto/src/layout_table.rs:172`) — no caller, no
  test, and nothing in the periodic or RSH paths reaches it. The fail-closed
  posture is correct and costs nothing until something does. Revisit if a
  consumer appears; the work is then the same shape as P2-1 (a headroom entry
  plus a launcher that reads ω), except that `g1e_grids.c:31,98` reads the slot
  through a DIFFERENT prologue than `CINTg0_2e`, so
  `math::range_separation` would not be reusable as-is.

---


Stages 0–3 landed in cintx on 2026-08-30 (see §9 of the parent plan). `int2e`,
`int3c2e` and `int2c2e` honour `env[PTR_RANGE_OMEGA]` end to end on the
**scalar, single-tuple** paths, verified against vendored libcint 6.1.3 at
3.4e-14.

This file is everything still open, in priority order. Every file:line was
verified against the trees on 2026-08-30.

**One correction to the parent plan before anything else.** §4 stage 5 says
"Once stage 2 is in a released cintx" and §6 says "Stage 5 pins a cintx
version". Neither is true today: `pyscf_rs/Cargo.toml:152-153` carries

```toml
[patch.crates-io]
cintx     = { path = "../cintx" }
```

so pyscf_rs already builds against this working tree. **Stage 5 is not gated on
a release** — it is gated only on this work being committed.

---

## Priority 0 — the batch surfaces silently ignore `range_omega`

This is a live instance of the exact failure the parent plan forbids in
writing: *"a full-range substitute must never ship: it runs, converges, and is
silently a different method."*

Four batch entry points hold an `ExecutionOptions` and never read
`range_omega` from it. A caller who sets `.with_range_omega(-0.8)` and then
batches gets the **full-range** integrals back, with no error.

`pyscf-pbc-df`'s `aux_e2` route is exactly `int3c2e` + `int2c2e`, which are
`TripleBatchRequest` and `PairBatchRequest` — so this is not hypothetical for
stage 5, it is the first thing stage 5 would hit.

| # | surface | site | reaches |
|---|---|---|---|
| **P0-1** | `TripleBatchRequest::evaluate_in` | `crates/cintx-rs/src/api.rs:3401` → `check_batch_request_scope` at `:3417` | `int3c2e` device batch |
| **P0-2** | `PairBatchRequest::evaluate_in` | `crates/cintx-rs/src/api.rs:3252` → `check_batch_request_scope` at `:3269` | `int2c2e` device batch |
| **P0-3** | `QuartetBatchRequest::evaluate_in` | `crates/cintx-rs/src/api.rs:2879`, inline guards at `:2898-2917` | `int2e_sph` device batch |
| **P0-4** | `eri_ssss_batch_inputs` pilot | `crates/cintx-rs/src/api.rs:786-810` | `int2e_cart` (s,s\|s,s) pilot |

`check_batch_request_scope` (`api.rs:3097-3118`) is the shared guard for P0-1
and P0-2 — one insertion covers both. P0-3 carries its own inline copy of the
same three checks and needs the same line; consider folding it onto the shared
helper while there.

`ss_batch_inputs` (`api.rs:737`) is **not** affected: it admits only operators
`0 | 3` (overlap, kinetic), and libcint's 1e kernels never read `env[8]`.

**Two acceptable fixes, in order of preference:**

1. **Fail closed** — reject a non-zero `range_omega` on every batch surface
   with `UnsupportedApi`, naming the scalar path as the supported route. Cheap
   (~4 lines plus tests), correct, and unblocks stage 5 to use the scalar path
   knowingly. Do this first regardless.
2. **Route the batch to the host engine**, mirroring what
   `launch_two_electron_typed` / `launch_center_3c2e_typed` /
   `launch_center_2c2e_typed` already do. Larger; only worth it once stage 5
   measures the scalar path as too slow.

**Acceptance for (1):** a test per surface asserting `UnsupportedApi` whose
message names `range_omega`; and a test that `range_omega = None` /
`Some(0.0)` still batches, byte-identical to today.

**Size:** half a day for (1).

---

## Priority 1 — stage 4, the device omega arms

Not started, and correctly so: range separation routes to the host Rys engine,
explicitly and `tracing::debug!`-logged, in all three launchers
(`two_electron.rs`, `center_3c2e.rs`, `center_2c2e.rs`, each computing
`route_host = is_range_separated(range_omega)`).

What makes it non-trivial, verified in the kernel source:

* `two_electron_scalar_kernel` (`crates/cintx-cubecl/src/kernels/two_electron.rs:793`)
  takes `#[comptime] nroots: u32` and selects `rys_root{1..5}`
  (`crates/cintx-cubecl/src/math/rys.rs:114` onward) at a **single** argument.
  Short range needs `rys_roots(rorder, x)` *and* `rys_roots(rorder, theta*x)`,
  plus a per-root rescaling — two evaluations into one root array.
* Long range is the easier half: one evaluation at a scaled `x`, a scaled
  `fac1`, and a `u → ut/(u + 1 − ut)` sweep. It could land on its own.
* Short range at `rys_order = 3` doubles to `nroots = 6`, above
  `BASE_DEVICE_NROOTS = 5` (`crates/cintx-cubecl/src/device_rys_ceiling.rs:51`).
  It therefore needs `EXTENDED_DEVICE_NROOTS = 12` (`:57`), which
  `device_nroots_ceiling` (`:325`) only grants when the `extended-device-rys`
  feature is on, the family is in `runs_extended_rys` (`:296`), **and** the
  backend's FMA-fusion probe passes. At `rys_order ≤ 2` (`nroots ≤ 4`) the base
  ceiling suffices.
* `sr_rys_roots_host` (`rys_wheeler.rs`) — the `rys_order > 3` arm — is
  double-double host code with a `Vec`-allocating Wheeler/eigensolve chain. It
  is not portable to `#[cube]` without a substantial rewrite, so the device
  arms would cover the doubled-root regime only and keep routing the rest to
  the host.

**Suggested staging:**

* **P1-a** long range on-device for all three families (one root evaluation,
  no root-count change) — the cheap win.
* **P1-b** short range on-device at `rys_order ≤ 2` (`nroots ≤ 4`, inside the
  base ceiling).
* **P1-c** short range at `rys_order = 3` behind `extended-device-rys`.
* **P1-d** leave `rys_order > 3` short range on the host, permanently, and say
  so in the module docs.

**Acceptance:** `range_omega_parity.rs`'s vendor gate must hold with the host
route disabled, and a device-vs-host cross-check per family in the style of
`center_3c2e.rs`'s `scalar_device_tests`.

**Size:** unknown, as the parent plan said. P1-a alone is plausibly 2–3 days.

---

## Priority 2 — cintx follow-ups

### P2-1 — widen `range_omega` to the derivative rows

`supports_range_omega` (`crates/cintx-runtime/src/range_omega.rs:105`) admits
`operator_name == "electron-repulsion"` on the `2e` / `3c2e` / `2c2e` families
only. Every derivative row of those families — `ip1`, `ip2`, `ipip1`,
`ipvip1`, `ip1ip2`, `ipspsp1`, `spsp1`, the `rel2e` sigma family — **does**
read `env[8]` upstream, because `g2e.c:171` shares `CINTg0_2e` with the whole
`int2e_*` symbol space. They refuse today rather than evaluate.

Needed for: range-separated *gradients* and Hessians. Not needed for Phase 14
Gate 3, RSDF, RSMDF or `rsjk`.

What it takes:

1. Widen `supports_range_omega`.
2. Make `rys_roots_for_request` (`crates/cintx-runtime/src/planner.rs`)
   account for the derivative `ng[IINC]`/`[JINC]`/`[KINC]`/`[LINC]` raises —
   its `rys_order = (Σ l)/2 + 1` is only exact for the scalar rows, which is
   precisely why the scope is narrow today.
3. Pass `range_omega` into each derivative launcher's `grad_shape`
   (`build_2e_shape` → `build_2e_shape_omega`) and swap `fill_g_tensor_2e` for
   `fill_g_tensor_2e_range` at each site. There are ~14 such call sites across
   `two_electron.rs`, `center_3c2e.rs` and `center_2c2e.rs`.
4. Route each to the host under a set ω, as the scalar paths do.

**Acceptance:** extend `range_omega_parity.rs`'s sweep to the derivative
symbols and hold the same 3.4e-14 bar.

**Size:** 2–4 days.

### P2-2 — `int1e_grids` reads `env[8]` upstream and we refuse it

`g1e_grids.c:31,98` reads `PTR_RANGE_OMEGA`; cintx's grids family does not
implement it, so `family_consumes_range_omega`
(`crates/cintx-runtime/src/range_omega.rs:132`) routes it into
`supports_range_omega`, which refuses. That is the fail-closed posture, and it
is correct — but it means a caller inside a PySCF `range_coulomb(omega)` block
who evaluates `int1e_grids` gets an error rather than a result. Decide whether
any consumer needs it before spending anything here.

**Size:** unknown; nothing currently asks for it.

### P2-3 — the `f32` precision path under `range_omega`

`rys_roots_range_separated` (`crates/cintx-cubecl/src/math/range_separation.rs`)
is `f64`-only and the three prologues call it in `f64` regardless of
`plan.precision`. That matches the full-range host arms, which also compute in
`f64` and cast at the staging write, so it is believed correct — but it is
untested. Add one `PrecisionKind::F32` case to `range_omega_parity.rs` at the
family's f32 tolerance, or state explicitly that f32 + range separation is
unsupported.

**Size:** half a day.

### P2-4 — spinor `int2e_spinor` under `range_omega` is accepted but untested

`int2e_spinor` has `operator_name == "electron-repulsion"` and
`canonical_family == "2e"`, so `supports_range_omega` admits it, and it falls
through `launch_two_electron_typed` to the scalar section, which routes host
and then applies the spinor transform downstream of `cart_blocks`. That should
be right by construction. No test covers it. Either add one to
`range_omega_parity.rs` or narrow `supports_range_omega` to Cart/Spheric.

**Size:** half a day.

---

## Priority 3 — stage 5, the pyscf_rs consumer work

Not started; it is a different repo. **No cintx release is needed** (see the
correction at the top) — only a commit.

Everything below is unblocked *including* the d/f-basis cases the parent plan's
§4.3 deferred to stage 3, because stage 3 landed alongside stage 2.

### P3-1 — thread ω into `aux_e2`

`crates/pyscf-pbc-df/src/incore/int3c.rs:319` (`aux_e2`) and `:345`
(`aux_e2_intor`), plus `crates/pyscf-pbc-df/src/outcore.rs:282`.

Put ω into the `SessionRequest`'s `ExecutionOptions` **before**
`query_workspace` — it sizes the short-range doubled Rys roots, and changing it
afterwards is rejected as backend contract drift.

**No change to `build_image_expanded_with_aux`** (`pyscf-gto/src/projection.rs:569`).
The ω travels in the options, not in the basis, which is exactly why the
parent plan's §1c ("the periodic driver never builds an `_env`") was never an
obstacle to the real fix.

⚠️ **Check first whether these call the scalar or the batch surface.** If they
use `TripleBatchRequest`/`PairBatchRequest`, P0 must land first or they will
silently get full-range integrals.

### P3-2 — `_RSGDFBuilder`

14-07 sub-tasks 7b/7c: `get_2c2e`, `outcore_auxe2`, `add_ft_j3c`,
`solve_cderi`, and `_RSNucBuilder`. 7a is already shipped and gated —
`crates/pyscf-pbc-df/src/rsdf_builder/omega.rs`, 10 tests at 1e-12 against
`measurements/omega.out`.

### P3-3 — `mdf::_RSMDFBuilder`

And re-point Gate 2 at `measurements/mdfladder.out`, which was recorded on this
route and which 14-06 had to replace with `mdfladder_cc.out`.

### P3-4 — Task 7d: flip `Gdf::prefer_ccdf` to `false`

`crates/pyscf-pbc-df/src/gdf/mod.rs:58` (field), `:90` (default `true`),
`:181` (branch), `:207` (the `NotYetImplemented` it raises today).

**This moves a committed reference energy**: diamond 2×2×2 goes from
**−10.93209469510988** (CC) to **−10.93209529106394** (RS), a documented
5.960e-07 step. One-line cited edit, as 14-07 Task 7d requires.

### P3-5 — `rsjk` in `pyscf-pbc-scf`

`crates/pyscf-pbc-scf/src/rsjk.rs`. Gate against **FFTDF, not GDF**
(14-08 Task 5.3: gating an exact builder against a fitted one hides a real
error behind the 1.2e-3 fitting gap).

### P3-6 — delete the refusals and their tests

| refusal | site |
|---|---|
| `CINTX_SR_GAP` constant | `crates/pyscf-pbc-df/src/rsdf_builder/mod.rs:93` |
| `RsGdfBuilder::build` | `crates/pyscf-pbc-df/src/rsdf_builder/mod.rs:160` |
| `Rsdf::build` | `crates/pyscf-pbc-df/src/rsdf.rs:87` (const use at `:30`, doc at `:11`) |
| `density_fit(DfKind::Rsdf)` | `crates/pyscf-pbc-df/src/density_fit.rs:105` |
| `RangeSeparatedJkBuilder::{build, get_jk}` | `crates/pyscf-pbc-scf/src/rsjk.rs:123`, `:142` (re-export at `:60`) |

And their tests:

* `crates/pyscf-pbc-df/tests/rsdf.rs:137` — `density_fit_refuses_rsdf_and_names_the_gap`
* `crates/pyscf-pbc-df/tests/rsdf_builder.rs:411` — `rs_gdf_builder_refuses_and_names_the_cintx_gap`
* `crates/pyscf-pbc-scf/tests/rsjk.rs:79` — `rsjk_refuses_and_names_the_cintx_gap`

Delete them **last**, in the same commit as the feature that replaces each. They
assert the refusal message text, so a silent full-range substitution turns them
red rather than green — which is their whole purpose.

### P3-7 — run Gate 3

From `measurements/builders.out` and `ccdf.py`, upstream 2.12.1:

| pair | diamond 2×2×2 | diamond gamma | He-fcc 2×2×2 |
|---|---|---|---|
| **GDF − RSDF** | **1.353e-08** | **4.566e-09** | **1.113e-10** |
| CC route − RS route | 5.960e-07 | 4.502e-06 | 5.222e-10 |

Gate 3 is `|E(GDF) − E(RSDF)|` landing on the first row within a factor of 2,
plus RSDF's own converged energy against upstream at 1e-11 on He-fcc
(**−2.80842508705964**) and ≤3e-8 on diamond (**−10.93209530458920** /
**−10.14369691810517**).

Assert the second row too. Two independent implementations of the same fitted
quantity reproducing upstream's own *disagreement* between its two routes says
more than either matching alone.

### P3-8 — Phase 4's CAM-B3LYP/H2O RSH assertion

The molecular half of the same gap, CI-gated since 2026.
`crates/pyscf-gto/src/range_coulomb.rs`'s `OmegaGuard` (`:64-102`) writes
`mol._env[8]` and restores it on unwind; that slot is now read by cintx's raw
path, so the contract test can become a numerical one.

---

## Priority 4 — not D-PBC-24, but red

`cargo test --workspace --features cpu` fails 6 tests that have nothing to do
with this work:

```
origi_genctr_parity:  test_int1e_origi_genctr_parity, ..._determinism
origk_genctr_parity:  test_int3c1e_origk_{scalar,ip1}_genctr_{parity,determinism}
```

All six panic with `source-only symbol ... requires feature 'unstable-source-api'`.
Both files came from commit `2494278`; their own headers say they need
`--features cpu,unstable-source-api`, but they lack the
`#![cfg(feature = "unstable-source-api")]` module gate their siblings carry
(compare `origk_ip1_random_rocm_parity.rs:24`). All six pass with the feature
enabled — verified 2026-08-30.

**Fix:** add the module gate to both files. One line each. Left alone here
because they are someone else's in-flight work and the intended gate may
differ (they may want the non-vendor half to run unconditionally).

---

## Suggested order

1. **P0** — fail the batch surfaces closed. Half a day, and it removes a live
   silent-substitution path.
2. **Commit the cintx work.** pyscf_rs picks it up through the path dep.
3. **P3-1 → P3-7** — stage 5, in the order listed. This is where the value is:
   Gate 3, `_RSGDFBuilder`, `_RSMDFBuilder`, `RSDF`, `rsjk`.
4. **P1-a** — long range on-device, if stage 5 finds the host route too slow.
5. **P2-1** — derivative rows, when a range-separated gradient is actually
   asked for.

P4 is independent of all of it and can go any time.
