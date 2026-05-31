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

## 29-05

- **`crates/cintx-cubecl/src/kernels/two_electron.rs` has an UNCOMMITTED working-tree hunk adding `pub fn int2e_common_factor` that is NOT present in the committed tree at HEAD.**
  - Discovered during 29-05 final verification via `git diff`. The committed 29-04 test `crates/cintx-oracle/tests/si_2e_transform_parity.rs` (commit `e40adbc`) references `int2e_common_factor`, but the function was never committed — it only exists in the dirty working tree (it is part of the parked pre-existing WIP the executor was told not to stage). A clean checkout of HEAD would therefore fail to compile the 29-04 `si_2e_transform_parity` test.
  - Root cause: a 29-04 commit-hygiene gap — the `int2e_common_factor` definition hunk landed in the working tree but was omitted from commit `fece759` (which the 29-04 SUMMARY claims added it). Pre-existing before 29-05 began; the executor's sequential-mode instructions explicitly direct it not to stage parked WIP.
  - Disposition: out of scope for 29-05 (SCOPE BOUNDARY — not caused by, and not owned by, this plan). NOT staged by 29-05. 29-05's own deliverables (`build.rs`, the 15 manifest rows, the 15 vendor shims, `rel_2e_sigma_parity.rs`) do NOT depend on `int2e_common_factor` (the new scaffold stubs the cintx side). All 29-05 verification ran green in the current working tree. Whoever finalizes the 29-04 WIP must commit this hunk so the 29-04 test builds on a clean checkout (likely folded into 29-06, which wires the 2e launchers).
  - **RESOLVED before 29-06:** `int2e_common_factor` was committed in `dd5b772` (`fix(29-04): commit int2e_common_factor def referenced by committed si_2e test`). A clean checkout of HEAD now builds the 29-04 test. No 29-06 action needed.

## 29-06

- **`cintx-oracle` lib tests `compare::tests::{evaluated_output_parity_and_optimizer_equivalence_hold, parity_artifacts_are_written, parity_mismatch_report_is_written_before_failure}` (now 180 mismatches) + `fixtures::tests::unstable_source_fixtures_require_opt_in` fail under `--features cpu`.**
  - Pre-existing, independent of 29-06. **Verified:** reverted the 29-06 `oracle_covered` flip (stashed `compiled_manifest.lock.json` + regenerated `.rs`/`.csv`) and re-ran `compare::tests::evaluated_output_parity_and_optimizer_equivalence_hold` → still **180 mismatches** (identical to post-flip). So the Task-3 `oracle_covered=true` flip did NOT change the count — the 16 Group-4 2e σ families are not exercised by the `compare::tests` generic fixture-matrix parity (they have no generic-matrix fixture; their byte-identity is proven by the dedicated `rel_2e_sigma_parity.rs` vendor double-gate test, which is 18/18 green). The mismatch count drifted 158→180 across plans from unrelated fixture-matrix changes, not from these flips.
  - Root cause: the same pre-existing whole-fixture-matrix oracle parity issue in `compare.rs` documented under 29-04/29-05, plus the unstable-source-api feature-gating test (29-03). Tracked in project memory `oracle_vendor_lib_tests_uncovered`.
  - Disposition: out of scope for 29-06 (SCOPE BOUNDARY — pre-existing, not caused by this plan). The 29-06 deliverables — all 16 Group-4 2e σ families byte-identical to vendored libcint 6.1.3 at atol=1e-12 on the non-square kappa fixture, flipped `oracle_covered=true` spinor-only, manifest-audit green, no OperatorId drift — are all GREEN.
