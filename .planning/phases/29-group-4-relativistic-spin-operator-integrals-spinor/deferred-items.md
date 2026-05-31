# Deferred Items — Phase 29

Out-of-scope discoveries logged during execution (NOT fixed — pre-existing or owned by other plans).

## 29-03

- **`cintx-oracle` lib test `fixtures::tests::unstable_source_fixtures_require_opt_in` fails under `--features cpu`.**
  - Pre-existing on branch `fix/general-contraction-nctr-1e`; verified by reverting the 29-03 `fixtures.rs` change and re-running — fails identically (0 passed / 1 failed) with a clean tree.
  - Root cause: the test asserts that explicit `unstable_source` mode includes source-only fixtures, which requires the `unstable-source-api` feature; the plan's verification profile (`--features cpu` only) does not enable it. Not related to the 2e transform suite or the new fixture.
  - Disposition: out of scope for 29-03 (SCOPE BOUNDARY). Not a regression from this plan. Belongs to whoever owns the unstable-source-api feature-matrix test gating.

## 29-04

- **`cintx-oracle` lib tests `compare::tests::{evaluated_output_parity_and_optimizer_equivalence_hold, parity_artifacts_are_written, parity_mismatch_report_is_written_before_failure}` fail under `--features cpu`** (158 mismatches in the full fixture-matrix oracle parity).
  - Pre-existing on branch `fix/general-contraction-nctr-1e`; verified by checking out the baseline manifest at commit `ac1d313` (parent of all 29-04 commits) and re-running `compare::tests::` — fails identically with 158 mismatches, with NONE of the 29-04 changes present.
  - Root cause: a pre-existing whole-fixture-matrix oracle parity issue in `compare.rs`, independent of the spsp1 row / 2e transform suite. The new `int2e_spsp1_spinor` manifest row is `oracle_covered=false`, so it is not exercised by the `compare::tests` parity matrix at all.
  - Disposition: out of scope for 29-04 (SCOPE BOUNDARY). Not a regression. Also note the 29-03-documented `fixtures::tests::unstable_source_fixtures_require_opt_in` failure persists (same root cause as above).
  - The 29-04 deliverable — the D-03 BLOCKING gate `--test si_2e_transform_parity` — is GREEN (4/4 under the vendor double-gate, 3/3 determinism-only).
