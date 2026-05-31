# Phase 28 — Deferred Items (out-of-scope discoveries)

## From Plan 28-03 execution (2026-05-31)

### Pre-existing oracle-parity baseline noise (NOT caused by 28-03)

Running `oracle-covered-update` / the full `cintx-oracle` parity tests
(`evaluated_output_parity_and_optimizer_equivalence_hold`,
`parity_artifacts_are_written`, `parity_mismatch_report_is_written_before_failure`)
fails with **158 mismatches** on the current working tree:

- **154 × `missing_raw_api_mapping`** — GIAO / origj / giao families that carry a
  manifest row but have NO `raw_api_for_symbol` arm in `compare.rs`:
  `int1e_a01gp_*`, `int1e_cg_a11part_*`, `int1e_cg_irxp_*`, `int1e_giao_a11part_*`,
  `int1e_giao_irjxp_*`, `int1e_ia01p_*`, `int1e_drinv_*`, `int1e_*_origj_*`,
  `int2e_g1_*`, `int2e_ig1_*`, `int2e_gg1_*`, `int2e_g1g2_*`, etc.
- **4 × `legacy_eval`**.

**Evidence this is pre-existing and out-of-scope for 28-03:**
- These manifest rows are present at `HEAD~1` (before the 28-03 manifest-row commit).
- `compare.rs` at `HEAD~1` already had NO raw_api mapping for these symbols
  (`git show HEAD~1:crates/cintx-oracle/src/compare.rs | grep -c int1e_a01gp_cart` → 0).
- `int1e_sp_spinor` (the 28-03 family) is **NOT** in the mismatch list — it is correctly
  recorded as a `skipped` spinor fixture by `is_skipped_spinor_fixture` (SC#4 verified).

These belong to the in-flight Phase-26 GIAO / Phase-28 spin-included family wiring
(raw-API mappings + kernels), not to the 28-03 manifest/FFI/SC#4 infrastructure task.
Per the executor SCOPE BOUNDARY rule they were NOT fixed here.

**SC#4 verification path used instead of the full vendor run:** the dedicated unit test
`compare::tests::sc4_int1e_sp_spinor_is_skipped_not_covered` proves
`int1e_sp_spinor` is skipped (kept out of `covered_symbols`) and has no `RawApiId`,
which is exactly what makes `oracle-covered-update`'s `if fixture.skipped { continue; }`
guard refuse to flip it. The lock was confirmed unmodified (stays `oracle_covered=false`)
after the bailed `oracle-covered-update` run.
