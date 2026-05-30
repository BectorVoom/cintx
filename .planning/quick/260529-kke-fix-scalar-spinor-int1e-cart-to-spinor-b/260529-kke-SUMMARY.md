---
phase: quick-260529-kke
plan: 01
subsystem: testing
tags: [spinor, cart2spinor, int1e, oracle, libcint, parity, cubecl]

# Dependency graph
requires:
  - phase: quick-260529-jtd
    provides: gradient-arm cart→spinor bra-major transpose (the mirror this fix copies)
  - phase: quick-260529-imi
    provides: device scalar 1e kernel that emits ket-major Cartesian blocks
provides:
  - int1e_{ovlp,kin,nuc}_spinor byte-parity with libcint 6.1.3 on asymmetric cross blocks
  - asymmetric p×d scalar-spinor vendor parity regression test
affects: [spinor, one-electron, cart2spinor]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Non-square p×d cross-block fixture to expose cart↔spinor orientation bugs that square symmetric blocks hide"

key-files:
  created:
    - crates/cintx-oracle/tests/one_electron_scalar_spinor_parity.rs
  modified:
    - crates/cintx-cubecl/src/kernels/one_electron.rs

key-decisions:
  - "Used a NON-SQUARE p×d cross block (not p×p) because the scalar overlap p×p Cartesian block is intrinsically transpose-symmetric and cannot expose a ket-major↔bra-major misread"

patterns-established:
  - "Pattern: orientation bugs in square symmetric operators require a non-square (different-l) shell pair to surface"

requirements-completed: [QUICK-260529-kke]

# Metrics
duration: ~25min
completed: 2026-05-29
---

# Phase quick-260529-kke: Scalar Spinor int1e cart→spinor Orientation Fix Summary

**Transpose the device scalar kernel's ket-major Cartesian block to bra-major before `cart_to_spinor_sf_2d`, making int1e_{ovlp,kin,nuc}_spinor byte-parity-clean (232→0 mismatches) vs libcint 6.1.3 on an asymmetric p×d cross block — the scalar mirror of the 260529-jtd gradient fix.**

## Performance

- **Duration:** ~25 min
- **Completed:** 2026-05-29T06:01:21Z
- **Tasks:** 2
- **Files modified:** 2 (1 created, 1 modified)

## Accomplishments
- New double-gated vendor parity test driving a NON-SQUARE p⁺×d⁺ cross block (shell 0 = p/l=1 nci=3, shell 1 = d/l=2 ncj=6, distinct atoms + distinct exponents).
- Confirmed the bug is real: 232 mismatches per operator vs vendored libcint on the unfixed scalar spinor arm.
- One-block transpose (ket-major → bra-major) added to the `Representation::Spinor` scalar 1e arm, mirroring the gradient arm from 260529-jtd.
- After the fix: 0 mismatches for int1e_{ovlp,kin,nuc}_spinor; all pre-existing tests still pass.
- `cart_to_spinor_sf_2d` / `c2spinor.rs` left unchanged (gradient path depends on its bra-major contract).

## Fix diff location

`crates/cintx-cubecl/src/kernels/one_electron.rs`, `Representation::Spinor =>` scalar 1e arm (around line 2902–2922). The direct
`cart_to_spinor_sf_2d::<F>(staging, &cart_blocks, ...)` call was replaced with a transposed copy:

```rust
let mut cart_bra_major = vec![0.0f64; nci * ncj];
for ic in 0..nci {
    for jc in 0..ncj {
        cart_bra_major[ic * ncj + jc] = cart_blocks[jc * nci + ic];
    }
}
cart_to_spinor_sf_2d::<F>(staging, &cart_bra_major, li, kappa_i, lj, kappa_j)?;
```

`nci`, `ncj` are in scope (defined ~lines 2510–2511; `block_len = nci*ncj` ~line 2781). The `sp_scale` normalization applied to `cart_blocks` earlier is untouched — the transpose reads the already-scaled buffer.

## Before / After mismatch counts (vs vendored libcint 6.1.3, atol=1e-12)

| Operator              | Before (unfixed) | After (fixed) |
| --------------------- | ---------------- | ------------- |
| int1e_ovlp_spinor     | 232 mismatches   | 0 mismatches  |
| int1e_kin_spinor      | 232 mismatches   | 0 mismatches  |
| int1e_nuc_spinor      | 232 mismatches   | 0 mismatches  |

Before-fix run: `test result: FAILED. 3 passed; 3 failed` (3 smoke pass, 3 asym parity fail).
After-fix run: `test result: ok. 6 passed; 0 failed`.

## Exact vendor cargo/env invocations used

```bash
# RED (Task 1) and GREEN (Task 2) parity:
CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
  --test one_electron_scalar_spinor_parity -- --nocapture

# Regression — jtd gradient spinor parity (still 0 mismatches):
CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
  --test one_electron_grad_spinor_parity        # → 8 passed

# Regression — scalar-spinor idempotency / arity2 parity:
cargo test -p cintx-oracle --features cpu --test safe_api_arity2_parity   # → 4 passed

# Regression — cubecl crate unit tests:
cargo test -p cintx-cubecl --features cpu        # → all suites pass (246/8/5/13/4/11/5/9/0)
```

## c2spinor.rs confirmation

`crates/cintx-cubecl/src/transform/c2spinor.rs` was **NOT modified** (`git diff --name-only` shows only `one_electron.rs`). The bra-major reader contract (`cart[n*ncj + j]` in `apply_bra_block`) is preserved, which the gradient path (260529-jtd) depends on.

## Task Commits

1. **Task 1: Failing asymmetric p×d vendor parity test** - `18bf045` (test)
2. **Task 2: Transpose scalar spinor cart block to bra-major** - `f4230c6` (fix)

## Files Created/Modified
- `crates/cintx-oracle/tests/one_electron_scalar_spinor_parity.rs` - New double-gated vendor parity test (3 smoke + 3 asym parity) on a non-square p×d fixture.
- `crates/cintx-cubecl/src/kernels/one_electron.rs` - One-block ket-major→bra-major transpose in the scalar `Representation::Spinor` arm.

## Decisions Made
- **Non-square p×d instead of p×p fixture.** The plan suggested two distinct p shells (p⁺×p⁺). Empirically, the scalar OVERLAP Cartesian p×p block is intrinsically transpose-SYMMETRIC (S_{μν}==S_{νμ}) regardless of centers/exponents — verified the raw cart cross block came back symmetric to ~1e-16 even with a fully general displacement, and the spinor parity PASSED pre-fix. A non-square 3×6 (p×d) block cannot be its own transpose, so the ket-major↔bra-major misread becomes an unambiguous 232-element mismatch. This is documented in the test header and the `assert_fixture_asymmetric` guard. See Deviations.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug, in the test fixture] Switched the fixture from p×p to non-square p×d**
- **Found during:** Task 1 (RED verification)
- **Issue:** The plan's prescribed two-distinct-p-shell (p⁺×p⁺) fixture did NOT fail pre-fix. The scalar overlap p×p Cartesian cross block is transpose-symmetric (off-diagonals S_{μν}==S_{νμ}) by the structure of the overlap integral, independent of centers/exponents — confirmed by a temporary diagnostic showing cart-block transpose-asymmetry ≈ 1.1e-16 and vendor==cintx pre-fix. A symmetric block hides a ket-major↔bra-major misread, so the test could not satisfy the strict failing-first gate.
- **Fix:** Changed shell 1 from a p shell (l=1) to a d shell (l=2), yielding a non-square 3×6 cross block. A non-square block has no transpose-symmetry escape hatch: reading a ket-major (6 outer × 3 inner) buffer as bra-major (3 outer × 6 inner) addresses entirely different elements. Updated `assert_fixture_asymmetric` to require l0=1, l1=2, l0≠l1, distinct atoms; updated matrix-size assertions (n_sp = 6+10 = 16 → 512). Re-ran RED: 232 mismatches per operator. The fix then drives all three to 0.
- **Files modified:** crates/cintx-oracle/tests/one_electron_scalar_spinor_parity.rs (test only; production fix unchanged from plan)
- **Verification:** RED 232/232/232 mismatches before fix; GREEN 0/0/0 after fix; all regressions pass.
- **Committed in:** 18bf045 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (fixture correction to honor the strict failing-first TDD gate).
**Impact on plan:** The production fix is exactly as planned (one-block ket-major→bra-major transpose mirroring jtd). Only the test fixture's shell choice changed, to produce a genuinely orientation-sensitive (non-square) block. No scope creep; no change to `cart_to_spinor_sf_2d`.

## Issues Encountered
- Initial p×p fixture passed pre-fix (transpose-symmetric overlap block). Diagnosed with temporary cart-block dumps (asymmetry ≈ 1.1e-16), then switched to a non-square p×d block. Diagnostics removed before committing.

## User Setup Required
None - no external service configuration required. Vendor parity requires `CINTX_ORACLE_BUILD_VENDOR=1` + `--features cpu` (slow libcint build), as standard for oracle parity tests.

## Next Phase Readiness
- Scalar and gradient spinor 1e paths are now both orientation-correct vs libcint 6.1.3.
- Pattern note for future work: square symmetric operators (overlap p×p) cannot expose cart↔spinor orientation bugs; use a non-square (different-l) shell pair as the regression probe.

## Self-Check: PASSED

- FOUND: crates/cintx-oracle/tests/one_electron_scalar_spinor_parity.rs
- FOUND: crates/cintx-cubecl/src/kernels/one_electron.rs
- FOUND: .planning/quick/260529-kke-fix-scalar-spinor-int1e-cart-to-spinor-b/260529-kke-SUMMARY.md
- FOUND commit: 18bf045 (Task 1 test)
- FOUND commit: f4230c6 (Task 2 fix)

---
*Phase: quick-260529-kke*
*Completed: 2026-05-29*
