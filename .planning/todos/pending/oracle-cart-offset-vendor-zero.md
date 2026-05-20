---
id: oracle-cart-offset-vendor-zero
created: 2026-05-20T23:38:38Z
status: pending
severity: medium
source: phase-20 regression gate
resolves_phase:
---

# Pre-existing oracle lib-test failure: CINTshells_cart_offset vendor=0

## What
Under `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle`, 4 `compare::tests` **lib** unit tests fail:
- `evaluated_output_parity_and_optimizer_equivalence_hold`
- `helper_coverage_matches_manifest`
- `parity_artifacts_are_written`
- `parity_mismatch_report_is_written_before_failure`

All fail at the same point: `CINTshells_cart_offset[4] mismatch: cintx=8 vendor=0` (compare.rs ~line 645-650).

## Why it's NOT a Phase 20 regression
- The phase-20 diff (8997703..HEAD) does not touch `cart_offset` in any file (cintx-compat/helpers.rs, vendor_ffi.rs, or the comparison in compare.rs — phase 20 only *added* f32-tolerance lines).
- Every **integration** oracle parity test passes (one_electron 6/6, two_electron 2/2, all center families, f12 15/15, f32_parity 11/11, safe_api, ecp) — and those call the same `CINTshells_cart_offset` helper. The helper works in integration; only the lib unit-test fixture sees vendor=0 (vendor FFI returns zeros in the lib-test harness context).

## Hypothesis
Environmental / test-harness: the vendor FFI `ao_loc` array is not populated (returns 0) in the `compare::tests` lib-unit context, while it is correctly populated in the integration test binaries. Likely pre-existing; surfaced now because the routine `--features cpu` CI gate does not set `CINTX_ORACLE_BUILD_VENDOR=1`.

## Next step
Confirm by running the lib test at a pre-phase-20 commit (e.g. 8997703) with the vendor build; if it reproduces, file as a standalone oracle-harness bug independent of the precision refactor.
