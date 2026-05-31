---
phase: 26-group-5-spin-free-giao-nmr-integrals-complex
plan: 04
subsystem: kernels
tags: [giao, a01gp, fail-closed, unsupported-api, gap-closure, cr-01, oracle]

# Dependency graph
requires:
  - phase: 26-group-5-spin-free-giao-nmr-integrals-complex
    plan: 02
    provides: "int1e_a01gp registered/dispatchable in the giao_nuc_op table (op_kind 3, rank 9) with the known-wrong rank-9 kernel arm + #[ignore]d parity test"
provides:
  - "Fail-closed UnsupportedApi guard for int1e_a01gp (cart + sph) before any compute / write_giao_complex_staging — closes VERIFICATION CR-01 / SC-2 gap"
  - "Non-vendor-gated test (test_int1e_a01gp_is_fail_closed) locking the fail-closed contract on the default cpu profile"
affects: [26-05-a01gp-kernel-parity, giao, complex-output]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Fail-closed gap closure: a registered-but-known-wrong family stays dispatchable in the op table but the COMPUTE arm returns UnsupportedApi before any buffer write — caller gets a typed error, never silently-wrong numbers"
    - "Contract-lock test in a SEPARATE non-vendor-gated module (cfg(feature=\"cpu\") only, NOT has_vendor_libcint) so the fail-closed guarantee runs on the default profile even when the vendor oracle is unavailable"

key-files:
  created: []
  modified:
    - crates/cintx-cubecl/src/kernels/one_electron.rs
    - crates/cintx-oracle/tests/giao_1e_parity.rs

key-decisions:
  - "Guard placed at the TOP of the giao_nuc nuclear-engine arm, immediately after the spinor UnsupportedApi check and BEFORE the Rys-nroots guard / buffer alloc / run_1e_giao_nuc / write_giao_complex_staging — guard line 8787 precedes the nuclear-arm staging write at 8859"
  - "a01gp stays in the giao_nuc_op dispatch table (Some((3,9))) so the descriptor still resolves and the manifest/RawApiId surface is unchanged — only COMPUTE is refused (Plan 26-05 removes the guard once vendor parity lands)"
  - "The fail-closed test lives in a NEW mod fail_closed gated only on feature=\"cpu\" (not has_vendor_libcint) so the contract is ALWAYS covered; the test buffer is sized 2x (complex_output) so an undersized-output error cannot masquerade as the fail-closed outcome"
  - "test_int1e_a01gp_parity left #[ignore]d — Plan 26-05 owns its removal after the kernel is corrected"

patterns-established:
  - "No-silent-wrong-output enforcement: when a family is registered but not yet byte-identical, add a fail-closed UnsupportedApi compute guard + a non-vendor contract test rather than leaving a dispatchable wrong-number path"

requirements-completed: []

# Metrics
duration: 12min
completed: 2026-05-31
---

# Phase 26 Plan 04: int1e_a01gp Fail-Closed Gap Closure Summary

**Added a fail-closed `UnsupportedApi` guard so `eval_raw(int1e_a01gp)` (cart + sph) returns a typed error before any compute instead of the known-wrong rank-9 buffer, plus a non-vendor-gated test that locks the fail-closed contract on the default profile — closing the VERIFICATION CR-01 / SC-2 gap.**

## Performance

- **Duration:** ~12 min
- **Tasks:** 2
- **Files modified:** 2 (0 created, 2 modified)

## Accomplishments

- **Task 1 — Guard (`one_electron.rs`):** Added an `op_name == "a01gp"` guard at the top of the `giao_nuc_op` nuclear-engine arm (`if let Some((op_kind, rank)) = giao_nuc_op {`), immediately after the existing spinor `UnsupportedApi` check and before the Rys-nroots guard, buffer allocation, `run_1e_giao_nuc_on_backend`, and `write_giao_complex_staging`. The guard returns `cintxRsError::UnsupportedApi`. A `// CR-01 / 26-04: fail-closed until kernel parity (26-05) lands` comment documents that Plan 26-05 removes it. The `giao_nuc_op` table is untouched — a01gp stays registered/dispatchable so the descriptor resolves; only the compute is refused. No partial-write `if dst < staging.len()` guard was added (project memory: monolithic writers). Verified: guard at line 8787 precedes the nuclear-arm `write_giao_complex_staging` at 8859; `cargo build -p cintx-cubecl --features cpu` exits 0.
- **Task 2 — Contract test (`giao_1e_parity.rs`):** Added a NEW `mod fail_closed` gated only on `#[cfg(feature = "cpu")]` (NOT `has_vendor_libcint`), so it runs on the default non-vendor test profile. `test_int1e_a01gp_is_fail_closed` drives `eval_raw(INT1E_A01GP_CART, ...)` and `eval_raw(INT1E_A01GP_SPH, ...)` on the existing `build_h2o_sto3g_common_orig` cross-center non-square H1×O block and asserts each returns `Err(cintxRsError::UnsupportedApi { .. })`. The output buffer is sized `2×` (complex_output) so an undersized-output error cannot masquerade as the fail-closed outcome. The old `test_int1e_a01gp_parity` stays `#[ignore]`d (26-05 owns its removal). Verified: `cargo test -p cintx-oracle --features cpu test_int1e_a01gp_is_fail_closed` runs 1 test and passes.

## Task Commits

1. **Task 1: fail-closed UnsupportedApi guard for int1e_a01gp** — `2c5dc0d` (fix)
2. **Task 2: assert a01gp raw path is fail-closed via UnsupportedApi** — `6841223` (test)

## Deviations from Plan

None - plan executed exactly as written.

## Threat Model Coverage

Both STRIDE register mitigations are honored:

- **T-26-05 (Information disclosure — wrong data):** mitigated by the Task 1 fail-closed guard — `int1e_a01gp` callers now receive a typed `UnsupportedApi` error, never a silently-wrong numerical buffer, BEFORE any compute or staging write.
- **T-26-06 (Tampering — regression):** mitigated by the Task 2 non-vendor-gated test, which locks the fail-closed contract so any future un-guarding (e.g. a premature 26-05 guard removal without parity) regresses loudly on the default profile.

## Threat Flags

None. No new trust boundaries introduced — the change only ADDS a fail-closed early return on the existing `caller → eval_raw` path; it removes a wrong-output surface rather than adding any new surface.

## Known Stubs

- **`int1e_a01gp` (rank-9, NABLA-RINV CROSS P)** — the kernel arm is still known-wrong (~2x on a subset of ket-varying components 1..8) and remains `oracle_covered=false` with its parity test `#[ignore]`d. This is intentional and now SAFE: the public raw path is fail-closed (returns `UnsupportedApi`) so no wrong numbers escape. Plan 26-05 owns the kernel correction, the guard removal, and un-ignoring `test_int1e_a01gp_parity`. This stub does NOT block the plan goal — the plan's goal is precisely to make the public surface fail-closed while correctness is pending.

## Self-Check: PASSED

- FOUND: crates/cintx-cubecl/src/kernels/one_electron.rs (guard at line 8787, precedes nuclear-arm write_giao_complex_staging at 8859)
- FOUND: crates/cintx-oracle/tests/giao_1e_parity.rs (test_int1e_a01gp_is_fail_closed at line 333, non-vendor-gated)
- FOUND: .planning/phases/26-group-5-spin-free-giao-nmr-integrals-complex/26-04-SUMMARY.md
- FOUND commit 2c5dc0d (Task 1 — fix: guard)
- FOUND commit 6841223 (Task 2 — test: fail-closed contract)
