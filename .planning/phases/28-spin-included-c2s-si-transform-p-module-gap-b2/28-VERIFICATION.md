---
phase: 28-spin-included-c2s-si-transform-p-module-gap-b2
verified: 2026-05-31T12:00:00Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
---

# Phase 28: Spin-Included c2s_si Transform + σ·p Module (Gap B2) Verification Report

**Phase Goal:** Deliver the spin-included `c2s_si` 4-block (`gc_x/gc_y/gc_z/gc_1`) spinor transform + a generic σ·p G-tensor assembler module, with the `int1e_sp` Spinor dispatch wired end-to-end, validated against a kappa-bearing relativistic oracle fixture at atol=1e-12 so the σ-coupling matches libcint `c2s_si_1e`. Plus the "p-module-gap-b2" gap closure: manifest/FFI infra for `int1e_sp_spinor` registered WITHOUT flipping its oracle coverage (D-01 — it must stay oracle_covered=false).

**Requirement:** FND-05 (foundation for REL-01 / GIAO-03)

**Verified:** 2026-05-31T12:00:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `cart_to_spinor_si_2d` owns the KET→BRA transpose, uses spinor_len sizing (never hardcoded 4l+2), and fail-closes (BufferTooSmall) before writes | ✓ VERIFIED | c2spinor.rs L673-757: per-block closure transpose at L723-735; `di = spinor_len(li, kappa_i as i32)` at L686-687; `BufferTooSmall` guard at L712-716; `ChunkPlanFailed` guards at L697-709; no 4l+2 literal in si_2d body |
| 2 | `apply_bra_si_block` transcribes `a_bra_cart2spinor_si` signs (`+ca_i*vz / -cb_r*vy / +cb_i*vx`), NOT `apply_si_block` signs; does NOT delegate to `apply_si_block` | ✓ VERIFIED | c2spinor.rs L1046-1049 verbatim: `sa_r += ca_r*v1 + ca_i*vz - cb_r*vy + cb_i*vx`; `apply_si_block` not called within `apply_bra_si_block` or `apply_bra_si_block_one`; unit test `apply_bra_si_block_l1_kappa_neg1_hand_derived` + sign-discrepancy guard pass (42/42 lib tests green) |
| 3 | `sigma_p.rs` generic σ·p #[cube] assembler exists, rank-parameterized, emits 4 pre-blocked component-leading gc blocks; scalar slot (gc_1) == 0 for int1e_sp | ✓ VERIFIED | sigma_p.rs: 5 `#[cube]` annotations; `#[comptime] tensor_rank: u32` at L138; `N_GC = 4` blocks; component-leading writes at L289-292 (`base + comp*block_len + elem`); scalar slot writes `+= F::new(0.0)` (zero, already zeroed); `mod sigma_p` in mod.rs L26; 3/3 device-vs-host tests pass |
| 4 | `int1e_sp` Spinor dispatch arm in one_electron.rs is wired, detects by symbol name (`op_name == "sp"`), routes assembler → `cart_to_spinor_si_2d`, handles nctr>1, and has a fail-closed staging guard (CR-01, commit 8d049cc) | ✓ VERIFIED | one_electron.rs L9113 `let is_sp = op_name == "sp"` (symbol detection, no positional int); L10495-10560 dispatch arm: sigma_p assembler call, per-(ci,cj) fold loop, `cart_to_spinor_si_2d` call; `staging_required = ni_sp*nj_sp*2; if staging.len() < staging_required { return Err(BufferTooSmall...)` at L10554-10560 BEFORE any scatter; no nctr>1 rejection copied from sf_2d path |
| 5 | D-01 invariant: `int1e_sp_spinor` is in compiled_manifest.lock.json with oracle_covered=false; SC#4 guard (`is_skipped_spinor_fixture` + `if fixture.skipped { continue }`) prevents flip; FND-05 byte-identity proved without flipping any flag; non-square nctr>1 kappa≠0 fixture; 6/6 vendor parity tests pass at atol=1e-12 | ✓ VERIFIED | manifest entry confirmed: `"oracle_covered": false` with `"symbol": "int1e_sp_spinor"`; `is_skipped_spinor_fixture` in compare.rs L303; SC#4 guard comment in oracle_covered_update.rs L51-60; `vendor_int1e_sp_spinor` in vendor_ffi.rs L4139+; `build_kappa_spinor_fixture` (p kappa=+1 LT, d kappa=-1 GT, nctr=2, non-square) + `build_heavy_atom_spinor_fixture` in fixtures.rs; **`CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test si_transform_parity` → 6/6 PASS** including `test_int1e_sp_kappa_spinor_byte_identity` at atol=1e-12 and `test_no_silent_skip` |

**Score:** 5/5 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/cintx-cubecl/src/transform/c2spinor.rs` | `apply_bra_si_block` + `cart_to_spinor_si_2d` | ✓ VERIFIED | Both functions present; bra at L952, 2D transform at L673; 42/42 lib tests pass |
| `crates/cintx-cubecl/src/kernels/sigma_p.rs` | Generic σ·p `#[cube]` assembler + `launch_int1e_sp_spinor_pair` | ✓ VERIFIED | File exists; 5 `#[cube]` annotations; `run_sigma_p_on_backend` pub(crate); `launch_int1e_sp_spinor_pair` pub; 3/3 device-vs-host tests pass |
| `crates/cintx-cubecl/src/kernels/mod.rs` | `pub mod sigma_p` declaration | ✓ VERIFIED | L26: `pub mod sigma_p;` |
| `crates/cintx-oracle/tests/si_transform_parity.rs` | End-to-end byte-identity proof vs vendor at atol=1e-12 | ✓ VERIFIED | File exists; calls `vendor_int1e_sp_spinor`; atol=1e-12; no-silent-skip assertion; D-01 oracle_covered=false assertion; 6/6 pass under double gate |
| `crates/cintx-oracle/src/fixtures.rs` | `build_kappa_spinor_fixture` + heavy-atom fixture | ✓ VERIFIED | Both functions present; kappa=+1/−1 set; p nctr=2; non-square p×d |
| `crates/cintx-ops/generated/compiled_manifest.lock.json` | `int1e_sp_spinor` row with oracle_covered=false | ✓ VERIFIED | Row present; `"oracle_covered": false` confirmed |
| `crates/cintx-oracle/src/vendor_ffi.rs` | `vendor_int1e_sp_spinor` FFI shim + extern | ✓ VERIFIED | Function at L4139; extern declaration present; compiles under vendor gate |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `cart_to_spinor_si_2d` | `apply_bra_si_block` | stage 1 bra call | ✓ WIRED | L743 call inside si_2d |
| `cart_to_spinor_si_2d` | `apply_ket_transform` | stage 2 ordinary ket reuse | ✓ WIRED | `apply_ket_transform` called 4 times in file; called within si_2d |
| `sigma_p.rs assembler` | Pauli g1×g0×g0 mix | `s[0]=g1x*g0y*g0z` etc | ✓ WIRED | sigma_p.rs L277 `s0 = g1x * g0y * g0z` + L278/279 for gc_y/gc_z; matches `CINTgout1e_int1e_sp` |
| `kernels/mod.rs` | `sigma_p` module | module declaration | ✓ WIRED | `pub mod sigma_p;` at L26 |
| `one_electron.rs int1e_sp Spinor arm` | `cart_to_spinor_si_2d` | dispatch after σ·p assembler | ✓ WIRED | `cart_to_spinor_si_2d::<F>(` at L10575; import at L25 |
| `si_transform_parity.rs` | `vendor_int1e_sp_spinor` | byte-identity compare at atol=1e-12 | ✓ WIRED | L170 call to `vendor_int1e_sp_spinor`; L287 mismatch assert at ATOL=1e-12 |
| `oracle_covered_update.rs guard` | skipped-fixture continue | `if fixture.skipped { continue }` | ✓ WIRED | L61 `if fixture.skipped`; `is_skipped_spinor_fixture` at compare.rs L303 routes int1e_sp_spinor as skipped |
| `vendor_int1e_sp_spinor` | `ffi::int1e_sp_spinor` | extern FFI call | ✓ WIRED | vendor_ffi.rs: `ffi::int1e_sp_spinor(` at L4149 |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `cart_to_spinor_si_2d` | `staging` spinor output | `apply_bra_si_block` + `apply_ket_transform` acting on real gc blocks | Yes — CG coeffs from coeff tables, gc blocks from sigma_p assembler | ✓ FLOWING |
| `sigma_p_kernel` | `gc_out` 4-block accumulator | VRR g0 + nabla g1 from real exponents/coefficients passed per primitive pair | Yes — real prim contractions from input shell data | ✓ FLOWING |
| `si_transform_parity.rs` | `cintx` spinor buffer | `launch_int1e_sp_spinor_pair` → `run_sigma_p_on_backend` → `cart_to_spinor_si_2d` | Yes — 6/6 vendor parity tests with non-zero output assertion | ✓ FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| c2spinor lib tests (apply_bra_si_block unit test + sign guard) | `cargo test -p cintx-cubecl --lib transform::c2spinor` | 42 passed, 0 failed | ✓ PASS |
| sigma_p device-vs-host + layout + scalar-slot-zero tests | `cargo test -p cintx-cubecl --features cpu --lib sigma_p` | 3 passed, 0 failed | ✓ PASS |
| si_transform_parity non-vendor smoke (sizing asserts + cintx eval) | `cargo test -p cintx-oracle --features cpu --test si_transform_parity` | 4 passed, 0 failed | ✓ PASS |
| FND-05 PRIMARY: byte-identity vs vendor at atol=1e-12, no-silent-skip, D-01 assert | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test si_transform_parity` | **6 passed, 0 failed** (incl. `test_int1e_sp_kappa_spinor_byte_identity` + `test_no_silent_skip`) | ✓ PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| FND-05 | 28-01, 28-02, 28-03, 28-04 | Spin-included c2s_si 4-block transform + σ·p G-tensor assembler, kappa-bearing oracle at atol=1e-12, c2s_si_1e compatible | ✓ SATISFIED | All four plans deliver their pieces; end-to-end byte-identity proven 6/6 under double gate |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `sigma_p.rs` | ~297 | `gc_out[b1] += F::new(0.0)` — spurious load-zero-store on pre-zeroed scalar slot | ⚠️ Warning (WR-01 from REVIEW) | Hot-loop wasted memory ops; Phase 29 must replace with real scalar when tensor_rank>1; NOT a correctness blocker for int1e_sp (gc_1=0 is correct) |
| `sigma_p.rs` | ~436-523 | `run_sigma_p_on_backend` match has no `_ =>` catch-all — non-exhaustive on zero-backend builds | ⚠️ Warning (WR-02 from REVIEW) | Compile error on zero-backend builds; accepted as project-wide pattern (matches one_electron.rs) |
| `fixtures.rs` | ~329-333 | Comment says "ROW-major" but bytes are COLUMN-major (libcint convention) | ⚠️ Warning (WR-03 from REVIEW) | Misleading only; behavior is correct (extract_shell transposes correctly) |

**Notes on anti-patterns:** All three warnings were identified in the 28-REVIEW.md and are non-blocking. CR-01 (the actual blocker — missing staging guard before scatter) was fixed in commit 8d049cc before this verification.

No STUB, MISSING, or ORPHANED artifacts found. The `run_sigma_p_on_backend` stub (`#[allow(dead_code)]` at plan-02 time) is now live — promoted to `pub(crate)` and called from both the dispatch arm and `launch_int1e_sp_spinor_pair`.

---

### Human Verification Required

None. All primary verifications are automated and passed programmatically under the double gate.

---

### Gaps Summary

No gaps. All 5 must-haves verified with codebase evidence:

1. `cart_to_spinor_si_2d` is substantive, owns the KET→BRA transpose per gc block, sizes via `spinor_len`, and fail-closes before any write.
2. The `a_bra_cart2spinor_si` sign convention (`+ca_i*vz`, `-cb_r*vy`, `+cb_i*vx`) is transcribed verbatim — distinct from `apply_si_block`'s signs; no delegation to `apply_si_block`.
3. `sigma_p.rs` #[cube] assembler is rank-parameterized, emits pre-blocked component-leading gc blocks, gc_1=0 for int1e_sp, wired into mod.rs.
4. The `int1e_sp` Spinor dispatch arm is live in one_electron.rs, symbol-detected (`op_name == "sp"`), routes `cart_to_spinor_si_2d`, handles nctr>1, and has the CR-01 fail-closed staging guard preceding all scatter writes.
5. D-01 maintained: `int1e_sp_spinor` manifest row stays `oracle_covered=false`; SC#4 guard routes it as skipped; vendor byte-identity proved in a dedicated transform test (not a flag flip); 6/6 tests pass under double gate at atol=1e-12 including the non-square nctr>1 kappa≠0 primary fixture and the `test_no_silent_skip` assertion.

---

_Verified: 2026-05-31T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
