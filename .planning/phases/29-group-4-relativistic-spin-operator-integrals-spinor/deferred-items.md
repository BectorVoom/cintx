# Deferred Items — Phase 29

Out-of-scope discoveries logged during execution (NOT fixed — pre-existing or owned by other plans).

## 29-03

- **`cintx-oracle` lib test `fixtures::tests::unstable_source_fixtures_require_opt_in` fails under `--features cpu`.**
  - Pre-existing on branch `fix/general-contraction-nctr-1e`; verified by reverting the 29-03 `fixtures.rs` change and re-running — fails identically (0 passed / 1 failed) with a clean tree.
  - Root cause: the test asserts that explicit `unstable_source` mode includes source-only fixtures, which requires the `unstable-source-api` feature; the plan's verification profile (`--features cpu` only) does not enable it. Not related to the 2e transform suite or the new fixture.
  - Disposition: out of scope for 29-03 (SCOPE BOUNDARY). Not a regression from this plan. Belongs to whoever owns the unstable-source-api feature-matrix test gating.
