---
phase: 21-coulomb-gradient-intors
plan: 06
subsystem: kernels
tags: [int3c2e_ip1, three-center-two-electron-gradient, cubecl, oracle-parity, gout_ip1, pitfall-4, GRAD-08, R1, F-order]

# Dependency graph
requires:
  - phase: 21 (plan 05)
    provides: pub(crate) gout_ip1 / nabla1i_2e / F12Shape (F12-free derivative math) + the build_2e_shape(li+1,...) + fill_g_tensor_2e + transpose-to-component-leading gradient recipe
  - phase: 21 (plan 02)
    provides: manifest id.operator "ip1" for int3c2e_ip1 + oracle base-profile parity harness (compare.rs int3c2e_ip1 mapping)
  - phase: 10-2e-2c2e-3c1e-3c2e-real-kernels
    provides: launch_center_3c2e scalar path / fill_g_tensor_2e / TwoEShape / rys_roots_host
  - phase: 18-sessionrequest-arity-ge3-dispatch
    provides: arity-3 safe-API dispatch + component_axis_leading layout (staging_elements = base × component_count)
provides:
  - int3c2e_ip1 REAL 3-component ∇_A derivative kernel (launch_center_3c2e_ip1) replacing the operator-blind scalar stub (Risk R1 CLOSED)
  - byte-identity vs vendored libcint 6.1.3 at atol=1e-12 for H2O/STO-3G cart+sph triples
  - component-leading [3, nk, nj, ni] F-order (same convention as int2e_ip1)
  - vendor_int3c2e_ip1_sph FFI wrapper (_cart pre-existed) + int3c2e_ip1_sph bindgen allowlist entry
  - oracle references flipped from plain vendor_int3c2e_* to the REAL derivative vendor_int3c2e_ip1_*
  - oracle_covered=true for int3c2e_ip1_{cart,sph} in the manifest
affects: [pyscf-rs-DF-gradient-runtime, future-gradient-intors]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "3c2e ip1 gradient = build_2e_shape(li+1, lj, 0, lk) [Pitfall-4 kl mapping] + fill_g_tensor_2e + f12::gout_ip1 + transpose to component-leading"
    - "Phantom 2e lk-slot (exponent 0 at the real-k center) maps the real third center k into the 2e ll-slot; gout_ip1 called with (li, lj, lk=0, ll=real_k)"
    - "Reuse the SHARED 2e recurrence (fill_g_tensor_2e) instead of the 3c2e-native fill so the G-tensor layout matches gout_ip1's di/dj/dk/dl strides"
    - "Element-for-element vendor byte-identity comparison IS the F-order layout gate"

key-files:
  created:
    - .planning/phases/21-coulomb-gradient-intors/deferred-items.md
  modified:
    - crates/cintx-cubecl/src/kernels/center_3c2e.rs
    - crates/cintx-cubecl/src/kernels/two_electron.rs
    - crates/cintx-oracle/src/vendor_ffi.rs
    - crates/cintx-oracle/build.rs
    - crates/cintx-oracle/tests/safe_api_arity3_parity.rs
    - crates/cintx-oracle/tests/center_3c2e_parity.rs
    - crates/cintx-oracle/tests/oracle_gate_closure.rs
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-ops/src/generated/api_manifest.csv

key-decisions:
  - "Branched on plan.descriptor.operator_name() == \"ip1\" (CONFIRMED: 21-02 set id.operator to \"ip1\" for int3c2e_ip1 entries; api_manifest.rs[19/20] carry operator_name \"ip1\")"
  - "Built the derivative G-tensor through fill_g_tensor_2e (the SHARED 2e recurrence) with the 3c2e phantom-lk mapping, NOT the 3c2e-native fill_g_tensor_3c2e — the native fill uses a [m=k][n=ij][root] layout incompatible with gout_ip1's di/dj/dk/dl strides; fill_g_tensor_2e produces exactly the layout gout_ip1 indexes"
  - "Folded the intra-pair Gaussian product factors into fac_env (common_factor × pdata_ij.fac × pdata_kl.fac) — fill_g_tensor_2e computes only the bra-ket Rys prefactor, not the pdata fac; omitting pdata_ij.fac caused the initial vendor-parity mismatch (the bug found via the vendor gate, fixed before commit)"
  - "Made TwoEShape / build_2e_shape / fill_g_tensor_2e / two_e_shape_as_f12 pub(crate) (Rule 3 enabling change) so center_3c2e.rs reuses the 2e machinery + gout_ip1 verbatim"
  - "Added int3c2e_ip1_sph to the oracle bindgen allowlist (Rule 3) — it is in cint_funcs.h but was absent from allowlist_function so the FFI binding was not generated"
  - "Reframed oracle_gate_3c2e_spinor to assert the R5 UnsupportedApi contract (no plain int3c2e_spinor manifest symbol exists; the test had borrowed INT3C2E_IP1_SPINOR while it was a scalar stub)"
  - "Switched the plain 3c2e family gate in oracle_gate_all_five_families to RawApiId::Symbol(\"int3c2e_sph\") (was borrowing INT3C2E_IP1_SPH); int3c2e_ip1_spinor oracle_covered stays false (R5, mirrors the int2e_ip1_spinor precedent)"

patterns-established:
  - "Pattern: 3c2e-family plain-Coulomb gradient = 2e recurrence (real i raised to li+1, real j, phantom 2e lk=0, real k→ll slot) + gout_ip1 at base li + per-component cart_to_sph_3c2e"
  - "Pattern: confirm the dispatcher's operator_name string against the manifest before branching (21-02 changed it from electron-repulsion to ip1)"

requirements-completed: [GRAD-08]

metrics:
  duration_min: 95
  tasks: 2
  files_changed: 10
  completed: "2026-05-26"
---

# Phase 21 Plan 06: int3c2e_ip1 Real Derivative Kernel Summary

Replaced the operator-blind scalar `int3c2e_ip1` stub with a real 3-component ∇_A derivative kernel (reusing `gout_ip1` verbatim through the 3c2e Pitfall-4 kl mapping) and flipped the oracle gate from plain `vendor_int3c2e_*` to the REAL `vendor_int3c2e_ip1_*` — byte-identical at atol=1e-12 on H2O/STO-3G, closing the latent silent-wrong runtime path consumed by pyscf-grad's DF-gradient (Risk R1).

## What Shipped

### Task 1 — Real int3c2e_ip1 derivative kernel (commits 78cecd1 RED, 5cf4442 GREEN)
- New `launch_center_3c2e_ip1<F>` branch in `launch_center_3c2e_typed`, selected by
  `plan.descriptor.operator_name() == "ip1"`. The plain `"electron-repulsion"` path is
  untouched (additive branch).
- The branch builds the plain Coulomb G-tensor via the **shared 2e recurrence**
  `fill_g_tensor_2e` with `build_2e_shape(li+1, lj, 0, lk)` (Pitfall-4: real k → 2e `ll`
  slot, phantom 2e `lk`-slot with exponent 0 at the real-k center, bra `i` raised to
  `li+1` for the `∇_i` headroom), then calls `gout_ip1` verbatim at BASE li, transposes
  the interleaved `gout[n*3+comp]` into component-leading `[3, nk, nj, ni]` F-order, and
  runs a per-component `cart_to_sph_3c2e` for the sph rep.
- Guards: spinor → `UnsupportedApi` (R5), `nroots > 5` → `UnsupportedApi` (R2), both
  fail-closed before any compute.
- Made `TwoEShape` / `build_2e_shape` / `fill_g_tensor_2e` / `two_e_shape_as_f12`
  `pub(crate)` so the 3c2e branch reuses the 2e machinery.
- Unit tests (cubecl, cpu): component-count (3× multiplier — `(p,s,s)` produces 9, the
  scalar stub wrote only 3), NOT-equal-to-plain (R1 regression proof — the first
  component block is NOT byte-equal to the plain integral), determinism (D-10),
  spinor-reject (R5). All 4 pass; 8 plain-path 3c2e tests still green.

### Task 2 — Oracle flip + FFI + manifest (commit 1527591)
- Added `vendor_int3c2e_ip1_sph` FFI wrapper (mirrors the pre-existing `_cart`) and
  added `int3c2e_ip1_sph` to the oracle bindgen `allowlist_function` regex.
- FLIPPED oracle references from plain `vendor_int3c2e_*` to the REAL derivative
  `vendor_int3c2e_ip1_*`:
  - `safe_api_arity3_parity.rs`: both `test_int3c2e_ip1_{cart,sph}_safe_api_parity`
    now use `vendor_int3c2e_ip1_{cart,sph}` with the 3-component buffer (`3 * ni*nj*nk`);
    the kernel-misnomer header comment is rewritten.
  - `center_3c2e_parity.rs`: the sph vendor-parity test references
    `vendor_int3c2e_ip1_sph` (atol lifted 1e-9 → 1e-12, buffer 3-component); the
    cpu-only idempotency test and the ROCm idempotency test buffers were enlarged to
    `3 * ni*nj*nk`.
- Flipped `oracle_covered` true for `int3c2e_ip1_{cart,sph}` in
  `compiled_manifest.lock.json`; regenerated `api_manifest.rs` + `api_manifest.csv`
  (build.rs regen). `int3c2e_ip1_spinor` stays false (R5, mirrors `int2e_ip1_spinor`).

## Verification Results

- `cargo test -p cintx-cubecl --features cpu --lib int3c2e_ip1` → 4/4 pass.
- `cargo test -p cintx-cubecl --features cpu --lib center_3c2e` → 8/8 (no plain-path regression).
- `cargo test -p cintx-cubecl --features cpu --lib` → 202/202 pass.
- Vendor gate (`CINTX_ORACLE_BUILD_VENDOR=1`, double-gated with `--features cpu`):
  - `center_3c2e_parity` → 2/2 (vendor parity byte-identical at atol=1e-12 vs the REAL `vendor_int3c2e_ip1_sph`).
  - `safe_api_arity3_parity` → 8/8 (incl. both flipped int3c2e_ip1 cart+sph tests, 125 triples each).
  - `oracle_gate_closure` → 10/10 (incl. the reframed R5 spinor gate + the plain-3c2e family gate).
- `cargo test -p cintx-oracle --features cpu` (no vendor) → all green.
- `CINTX_BACKEND=cpu cargo check --workspace --features cpu` → exits 0.

### Plan-required record items
- **operator_name string the dispatcher branches on:** `"ip1"` — CONFIRMED to match
  21-02's manifest change (`OPERATOR_DESCRIPTORS[19]`/`[20]` carry `operator_name: "ip1"`
  for `int3c2e_ip1_cart`/`_sph`).
- **NOT-equal-to-plain regression result:** for a `(p,s,s)` triple the ip1 first
  component block (lanes 0..3) is NOT byte-equal to the plain `int3c2e_sph` integral
  (`s-s-s` ∇_i = `[-0.4078, -0.0422, 0.0]` with the z-lane vanishing by in-plane
  symmetry) — proves the scalar stub is gone. The vendor gate further confirms the full
  derivative is byte-identical to libcint's own `int3c2e_ip1`.
- **oracle-flip diff:** `safe_api_arity3_parity.rs` (2 tests + header), `center_3c2e_parity.rs`
  (3 buffer/reference sites), `oracle_gate_closure.rs` (`eval_3c2e` symbol + the spinor gate).
- **manifest:** `oracle_covered` flipped true for `int3c2e_ip1_{cart,sph}` (was false
  pending this plan); spinor stays false (R5).

## Threat Model Dispositions

| Threat ID | Disposition | Outcome |
|-----------|-------------|---------|
| T-21-06-01 (silent-wrong scalar) | mitigate | CLOSED — real `gout_ip1` derivative + oracle flipped to `vendor_int3c2e_ip1_*`; NOT-equal-to-plain unit test + 3-component multiplier + vendor byte-identity prove the stub is gone. |
| T-21-06-02 (Pitfall-4 mapping broken) | mitigate | Preserved `build_2e_shape(li+1, lj, 0, lk)` + phantom-lk fill; vendor byte-identity vs libcint's int3c2e_ip1 catches a mismapped axis (it did — see Deviations). |
| T-21-06-03 (plain-path regression) | mitigate | ip1 branch is additive; 8 plain 3c2e unit tests + the plain 3c2e family oracle gate stay green. |
| T-21-06-04 (high-l/spinor unverified output) | mitigate | nroots>5 → UnsupportedApi (R2); spinor → UnsupportedApi (R5), both fail-closed; the reframed `oracle_gate_3c2e_spinor` asserts the R5 contract. |
| T-21-06-SC (supply chain) | accept | No new external packages; only existing workspace crates edited. |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Missing intra-pair Gaussian product factor in the ip1 G-tensor**
- **Found during:** Task 1 vendor-parity verification (the vendor gate failed with
  observed values ~2.9–6.6× the reference before the fix).
- **Issue:** `fill_g_tensor_2e` computes only the bra-ket Rys prefactor, not the
  per-pair `exp(-ai*aj/zeta*|ri-rj|^2)` product factors (which come from
  `compute_pdata_host`). The first cut passed `common_factor` directly as `fac_env`,
  dropping `pdata_ij.fac` (and the phantom `pdata_kl.fac`).
- **Fix:** Compute `pdata_ij` and `pdata_kl` per primitive triple and pass
  `common_factor * pdata_ij.fac * pdata_kl.fac` as `fac_env`, mirroring
  `launch_two_electron_ip1`'s `quartet_fac`.
- **Files modified:** `crates/cintx-cubecl/src/kernels/center_3c2e.rs`
- **Commit:** 5cf4442 (corrected in the same GREEN commit before the vendor gate passed).

**2. [Rule 3 - Blocking] `int3c2e_ip1_sph` absent from the oracle bindgen allowlist**
- **Found during:** Task 2.
- **Issue:** `vendor_int3c2e_ip1_sph` could not be added because the bindgen
  `allowlist_function` regex omitted `int3c2e_ip1_sph` (only `int3c2e_ip1_cart` was
  listed), so the FFI binding would not be generated.
- **Fix:** Added `int3c2e_ip1_sph` to the allowlist regex in `crates/cintx-oracle/build.rs`.
- **Commit:** 1527591.

**3. [Rule 3 - Blocking] Two oracle_gate_closure tests relied on the scalar stub**
- **Found during:** Task 2 vendor-gate verification (2 failures: `oracle_gate_3c2e_spinor`
  BufferTooSmall, `oracle_gate_all_five_families` 3c2e mismatch).
- **Issue:** Both tests borrowed `INT3C2E_IP1_{SPH,SPINOR}` to exercise the PLAIN 3c2e
  kernel (because the stub was operator-blind). With the real derivative shipped, the
  ip1 symbol now produces 3 components / rejects spinor.
- **Fix:** `oracle_gate_all_five_families`'s 3c2e family gate switched to the plain
  `RawApiId::Symbol("int3c2e_sph")`; `oracle_gate_3c2e_spinor` reframed to assert the R5
  `UnsupportedApi` contract (full planner-sized buffer so the dispatch reaches the
  kernel's R5 guard rather than the earlier BufferTooSmall planner check).
- **Files modified:** `crates/cintx-oracle/tests/oracle_gate_closure.rs`
- **Commit:** 1527591.

## Known Stubs

None. The int3c2e_ip1 scalar stub this plan was created to remove is gone (Risk R1
closed); the kernel now ships the real derivative.

## Deferred Issues

See `.planning/phases/21-coulomb-gradient-intors/deferred-items.md`:
- **D-21-06-A:** `xtask manifest-audit --check-lock` reports 37 `uncovered_stable_entries`
  — PRE-EXISTING and phase-wide (the Phase 21 gradient operators from waves 01–05 whose
  `oracle_covered` flags are not yet flipped). 21-06 REDUCES the count (flips
  int3c2e_ip1_{cart,sph} true). This is a phase-completion reconciliation task, not a
  per-plan blocker. `int3c2e_ip1_spinor` stays uncovered intentionally (R5, mirroring
  int2e_ip1_spinor). NOT caused by this plan.
- **D-21-06-B:** pre-existing unused-import/variable warnings in f12.rs and unstable.rs
  (not in 21-06's touched files).

## Worktree-Path Incident (resolved)

Early in execution, several Bash `cd /home/user/Documents/workspace/cintx` calls drifted
the cwd into the MAIN repo, so the first round of Edit/git-add operations landed in the
main repo (`fix/general-contraction-nctr-1e`) instead of the worktree. Caught before any
commit (#3097/#3099): reverted the main-repo staging, captured the edits as a patch, and
re-applied them inside the worktree. All subsequent operations used the worktree-absolute
path and `git -C "$WT"`. No commits were made to the main repo.
