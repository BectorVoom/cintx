---
id: oracle-cart-offset-vendor-zero
created: 2026-05-20T23:38:38Z
status: pending
severity: medium
source: phase-20 regression gate
resolves_phase:
blocks_phase_24_gate: false
classification: standalone oracle-harness bug (pre-existing)
repro_commit: 8997703
repro_result: reproduced (pre-phase-20)
triaged: 2026-05-30 (Phase 24 Plan 01 Task 3 / OQ-2)
---

# Pre-existing oracle lib-test failure: CINTshells_cart_offset vendor=0

## What
Under `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --lib`,
the `compare::tests` **lib** unit tests fail:
- `evaluated_output_parity_and_optimizer_equivalence_hold`
- `parity_artifacts_are_written`
- `parity_mismatch_report_is_written_before_failure`
- `helper_coverage_matches_manifest` (fails at the pre-phase-20 baseline; now PASSES
  on HEAD after the helper-surface hardening in 260529-lbr/mfh/mqo — so HEAD shows 3
  failures, the baseline showed 4; same root cause).

All fail at the same root point: `CINTshells_cart_offset[4] mismatch: cintx=8 vendor=0`
(compare.rs — the helper-parity assertion that gates the in-crate lib-unit parity
report). On HEAD the surviving three now surface it as a downstream
`oracle parity failed with N mismatches` panic (compare.rs:1617) because the offending
`ao_loc` feeds the integral comparison, but the underlying defect is the same
`cart_offset[4] vendor=0`.

## Triage result (Phase 24 Plan 01 Task 3 / OQ-2)

**REPRODUCED at pre-phase-20 commit `8997703`** ("docs(20): record planning completion").
Ran `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --lib` in a
throwaway detached worktree at `8997703`:

```
thread '...evaluated_output_parity_and_optimizer_equivalence_hold' panicked at
  crates/cintx-oracle/src/compare.rs:1428:74:
parity report: CINTshells_cart_offset[4] mismatch: cintx=8 vendor=0
...
test result: FAILED. 8 passed; 4 failed; 0 ignored
```

This confirms the failure **pre-dates Phase 24 AND Phase 20** — it is **pre-existing
oracle-harness noise**, NOT introduced by any moment/multipole work (Phase 24) or the
f32 precision refactor (Phase 20).

## Disposition: standalone oracle-harness bug — does NOT block the Phase 24 gate

- `blocks_phase_24_gate: false`. The Phase 24 acceptance gate is **`--test` integration
  parity** (`moment_*_parity.rs` and the existing `one_electron`/`two_electron`/center/
  f12/ecp integration suites), which all PASS — those call the same
  `CINTshells_cart_offset` helper and it works correctly in the integration-test binary
  context.
- The failure is confined to the `compare::tests` **lib-unit** harness, where the vendor
  FFI `ao_loc` (`CINTshells_cart_offset`) returns `0` instead of the populated offset
  array. Hypothesis (unchanged): an environment/harness artifact of the lib-unit context
  (vendor `ao_loc` not populated), surfaced only because the routine `--features cpu` CI
  gate does not set `CINTX_ORACLE_BUILD_VENDOR=1`.
- **Pre-existing** — tracked here as a standalone oracle-harness bug independent of the
  precision refactor and the moment families. Fixing the lib-unit `ao_loc` population is
  out of scope for Phase 24; the integration gate is the merge-blocking contract.

## Next step (deferred, non-blocking)
Investigate why the vendor `ao_loc`/`CINTshells_cart_offset` returns 0 specifically in
the `compare::tests` lib-unit binary (vs the integration binaries where it is correct).
Likely a missing setup call or a context difference in how the lib-unit harness invokes
the vendor offset helper. Not required for any Phase 24+ gate.
