---
phase: 24-group-3-position-multipole-moment-integrals
verified: 2026-05-30T08:30:00Z
status: passed
score: 4/4 must-haves verified
overrides_applied: 0
re_verification: null
gaps: []
human_verification: []
---

# Phase 24: Group 3 — Position / Multipole-Moment Integrals Verification Report

**Phase Goal:** Group 3 — Position / Multipole-Moment Integrals. Dipole through hexadecapole moments (int1e_r/rr/rrr/rrrr, r2/r4, z/zz, p4, rinv/drinv, irp) plus _origj variants, gated on the non-zero gauge-origin fixture.
**Verified:** 2026-05-30T08:30:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #  | Truth                                                                                                                                         | Status     | Evidence                                                                                                                      |
|----|-----------------------------------------------------------------------------------------------------------------------------------------------|------------|-------------------------------------------------------------------------------------------------------------------------------|
| 1  | MOM-01: int1e_r and int1e_r_origj match at atol=1e-12 against a non-zero gauge-origin fixture (cart+sph)                                      | ✓ VERIFIED | manifest `rank=3`, `oracle_covered=true`; `one_electron_moment_kernel` wired; `moment_r_parity` 2/2 GREEN (orchestrator confirmed)   |
| 2  | MOM-02: int1e_rr/r2/z/zz and their _origj variants match at atol=1e-12 (cart+sph)                                                            | ✓ VERIFIED | manifest ranks correct (rr=9, r2/z/zz=1 each); `moment_low_parity` 8/8 GREEN; `is_origj` branch live in dispatch            |
| 3  | MOM-03: int1e_rrr/rrrr/r4 (octupole/hexadecapole) match at atol=1e-12 (cart+sph), ket-side headroom from ng[1]; rrr_origj/rrrr_origj absent  | ✓ VERIFIED | manifest rrr=27, rrrr=81, r4=1; rrr_origj/rrrr_origj absent from manifest AND test files; `moment_high_parity` 4/4 GREEN    |
| 4  | MOM-04: int1e_p4, int1e_drinv, plain int1e_rinv, int1e_irp all match at atol=1e-12 (cart+sph); each uses the correct origin slot             | ✓ VERIFIED | 4 kernels (p4/irp/rinv/drinv) present in one_electron.rs; `moment_nontensor_parity` 4/4 GREEN; PTR_RINV_ORIG used for rinv/drinv, PTR_COMMON_ORIG for irp |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact                                                              | Expected                                                          | Status     | Details                                                                                                         |
|-----------------------------------------------------------------------|-------------------------------------------------------------------|------------|-----------------------------------------------------------------------------------------------------------------|
| `crates/cintx-ops/generated/compiled_manifest.lock.json`             | 18 families × 3 forms = 54 entries; exact component_rank         | ✓ VERIFIED | 54 entries confirmed; all 18 operators × {cart,sph,spinor}; ranks exact (r=3,rr=9,rrr=27,rrrr=81,r2/r4/z/zz/rinv/p4=1,drinv=3,irp=9); all cart/sph `oracle_covered=true`, all spinor `oracle_covered=false` |
| `crates/cintx-compat/src/raw.rs`                                      | RawApiId consts for all 18 base families + 6 _origj variants     | ✓ VERIFIED | INT1E_R_CART, INT1E_RR_CART, INT1E_RRR_CART, INT1E_RRRR_CART, INT1E_R2_CART, INT1E_R4_CART, INT1E_Z_CART, INT1E_ZZ_CART + all _ORIGJ_ + INT1E_RINV/DRINV/P4/IRP all confirmed at lines 193–278 |
| `crates/cintx-cubecl/src/kernels/one_electron.rs`                    | 5 device kernels; is_origj/is_rinv/is_drinv/is_p4/is_irp arms    | ✓ VERIFIED | `one_electron_moment_kernel` (L4561), `one_electron_irp_kernel` (L1935), `one_electron_rinv_kernel` (L3134), `one_electron_drinv_kernel` (L3483), `one_electron_p4_kernel` (L1561); all dispatch arms live at L5822-L6444 |
| `crates/cintx-oracle/tests/moment_common.rs`                         | rank-parameterized vendor_parity + env_with_rinv_origin + non_square_shell_pair | ✓ VERIFIED | All three helpers confirmed present; ATOL=1e-12; cross_center_non_square_shell_pair also present |
| `crates/cintx-oracle/tests/moment_r_parity.rs`                       | MOM-01 parity scaffold                                            | ✓ VERIFIED | File exists (2,469 bytes); r + r_origj tests |
| `crates/cintx-oracle/tests/moment_low_parity.rs`                     | MOM-02 parity scaffold                                            | ✓ VERIFIED | File exists (4,103 bytes); rr/r2/z/zz + _origj |
| `crates/cintx-oracle/tests/moment_high_parity.rs`                    | MOM-03 parity scaffold; NO rrr/rrrr origj                        | ✓ VERIFIED | File exists (3,406 bytes); rrr/rrrr/r4 + r4_origj; comment explicitly excludes rrr/rrrr origj |
| `crates/cintx-oracle/tests/moment_nontensor_parity.rs`               | MOM-04 parity scaffold; non-zero rinv origin                     | ✓ VERIFIED | File exists (5,158 bytes); p4/irp/rinv/drinv; `env_with_rinv_origin([0.5, -0.3, 0.8])` confirmed |
| `crates/cintx-oracle/build.rs`                                        | allowlist_function regex includes all 36 new symbols              | ✓ VERIFIED | All 24 base + 12 _origj symbols confirmed in the single allowlist_function call at L358; NO rrr_origj/rrrr_origj |
| `crates/cintx-oracle/src/vendor_ffi.rs`                              | 36 vendor FFI wrappers                                            | ✓ VERIFIED | grep count = 40 (includes vendored functions for both sph and cart × multiple families ≥ 36) |
| `.planning/todos/pending/oracle-cart-offset-vendor-zero.md`          | OQ-2 triaged; blocks_phase_24_gate: false                        | ✓ VERIFIED | `blocks_phase_24_gate: false`, repro_commit `8997703`, classification `standalone oracle-harness bug (pre-existing)` |

### Key Link Verification

| From                               | To                                    | Via                                        | Status     | Details                                                                                       |
|------------------------------------|---------------------------------------|--------------------------------------------|------------|-----------------------------------------------------------------------------------------------|
| `one_electron.rs` moment dispatch  | `env[PTR_COMMON_ORIG]`               | Cluster A base families gauge origin        | ✓ WIRED    | `common_orig` read live in `eval_raw` (raw.rs:774); passed through plan to kernel             |
| `one_electron.rs` `is_origj` branch | ket basis center `rj`               | `_origj` origin-source: `drj = 0`         | ✓ WIRED    | `is_origj = op_name.ends_with("_origj")` at L5859; `drj` zeroed for `_origj` at L5903       |
| `raw.rs` `is_rinv_family_symbol`    | `env[PTR_RINV_ORIG]`                 | rinv/drinv read the rinv slot, not common  | ✓ WIRED    | `is_rinv_family_symbol` matches `int1e_rinv_*`/`int1e_drinv_*`; OR'd into PTR_RINV_ORIG read block at L762 |
| `one_electron_p4_kernel`           | `d_i_1e_into` / `d_j_1e_into`        | BOTH-side +2 headroom (Pitfall 4)          | ✓ WIRED    | Kernel at L1561; both bra and ket raised by 2 (`nmax=li+lj+4`, `lj_ext=lj+2`); confirmed in 24-04 SUMMARY |
| `one_electron_irp_kernel`          | `PTR_COMMON_ORIG` via `rcj_1e_into`  | gauge-origin family, ket +2 headroom       | ✓ WIRED    | `rcj_1e_into` helper added; `drj = rj - common_orig`; is_irp at L5837 drops from rejection guard |
| `RawApiId` consts                  | manifest symbol strings               | Self::Symbol("...") exact-match            | ✓ WIRED    | Every const string (`"int1e_r_cart"` etc.) verified to match manifest `id.symbol` field       |

### Data-Flow Trace (Level 4)

Phase 24 families are compute kernels evaluated against a vendor oracle, not UI rendering components. The observable "data flowing" evidence is the vendor parity gate (18 parity tests, 18/18 GREEN at atol=1e-12 confirmed by orchestrator), not a UI data source trace. Level 4 is satisfied by the parity result.

| Family           | Kernel          | Oracle Fixture              | atol=1e-12 Result | Status     |
|------------------|-----------------|-----------------------------|--------------------|------------|
| r/r_origj        | moment_kernel   | build_h2o_sto3g_common_orig | 2/2 GREEN          | ✓ FLOWING  |
| rr/r2/z/zz + _origj | moment_kernel | same + cross-center _origj | 8/8 GREEN          | ✓ FLOWING  |
| rrr/rrrr/r4      | moment_kernel   | build_h2o_sto3g_common_orig | 4/4 GREEN          | ✓ FLOWING  |
| rinv/drinv       | rinv/drinv kernels | env_with_rinv_origin [0.5,-0.3,0.8] | 2+2 GREEN  | ✓ FLOWING  |
| p4               | p4_kernel       | cross-center (H1-1s × O-2p) | 1 GREEN            | ✓ FLOWING  |
| irp              | irp_kernel      | build_h2o_sto3g_common_orig | 1 GREEN            | ✓ FLOWING  |

**Note:** p4 and the _origj even-moment families require cross-center blocks. Same-center ⟨s|∇⁴|p⟩ = 0 by parity (for p4) and same-center even-moment _origj = 0 by construction. The test fixtures were corrected in plans 24-02 and 24-04 to use cross-center blocks; both vendor and cintx confirmed genuinely non-zero there.

### Behavioral Spot-Checks

| Behavior                                        | Command / Evidence                                                    | Status   |
|-------------------------------------------------|-----------------------------------------------------------------------|----------|
| All 18 parity tests pass under vendor gate      | Orchestrator-confirmed: moment_r 2/2, moment_low 8/8, moment_high 4/4, moment_nontensor 4/4 | ✓ PASS |
| Workspace build clean                           | Orchestrator-confirmed: workspace build clean                         | ✓ PASS   |
| cubecl --lib 280 tests pass (no regression)     | Orchestrator-confirmed: 280/280                                       | ✓ PASS   |
| compat 43/43 + ops 11/11 (no regression)        | Orchestrator-confirmed: 43/43, 11/11                                  | ✓ PASS   |
| Prior one-electron parity 40/40 (no regression) | Orchestrator-confirmed: 40/40                                         | ✓ PASS   |
| rrr_origj / rrrr_origj absent everywhere        | `grep -E 'rrr_origj|rrrr_origj' tests/` = 0 matches (only a clarifying comment in moment_high_parity.rs doc) | ✓ PASS |
| rinv/drinv read PTR_RINV_ORIG not PTR_COMMON_ORIG | `grep 'rinv\|drinv' raw.rs | grep 'COMMON_ORIG'` = 0 matches       | ✓ PASS   |
| MOM-04 genuinely complete after all 5 plans    | is_rinv/is_drinv/is_p4/is_irp all live in dispatch; 4/4 nontensor parity GREEN | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plans    | Description                                                                              | Status      | Evidence                                                                               |
|-------------|-----------------|------------------------------------------------------------------------------------------|-------------|----------------------------------------------------------------------------------------|
| MOM-01      | 24-01, 24-02    | Dipole int1e_r / int1e_r_origj, non-zero gauge origin, atol=1e-12 (cart+sph)           | ✓ SATISFIED | manifest rank=3; oracle_covered=true; moment_r_parity 2/2 GREEN                       |
| MOM-02      | 24-01, 24-02    | int1e_rr/r2/z/zz + _origj, atol=1e-12 (cart+sph)                                      | ✓ SATISFIED | manifest ranks correct; moment_low_parity 8/8 GREEN                                   |
| MOM-03      | 24-01, 24-02    | int1e_rrr/rrrr/r4, ket headroom, no rrr/rrrr _origj, atol=1e-12 (cart+sph)            | ✓ SATISFIED | manifest rrr=27/rrrr=81/r4=1; rrr/rrrr _origj absent; moment_high_parity 4/4 GREEN   |
| MOM-04      | 24-01, 24-03, 24-04, 24-05 | int1e_p4/drinv/rinv/irp, atol=1e-12 (cart+sph); correct origin slots | ✓ SATISFIED | All 4 sub-families fully wired; moment_nontensor_parity 4/4 GREEN after plan 24-05   |

**MOM-04 completeness confirmation:** Plan 24-03 partially satisfied MOM-04 (rinv/drinv only) and prematurely marked it complete in its `requirements-completed` field before p4 and irp landed. Plan 24-04 added p4 and plan 24-05 added irp — both also mark `requirements-completed: [MOM-04]`. After all five plans, MOM-04 is NOW genuinely complete: is_rinv, is_drinv, is_p4, and is_irp are all live dispatch arms in `one_electron.rs`, all 4 sub-families have manifest entries with `oracle_covered=true` for cart/sph, and `moment_nontensor_parity` runs 4/4 GREEN under the vendor double-gate. The premature marking in 24-03 was a documentation inaccuracy, not a functional gap.

**REQUIREMENTS.md traceability accuracy:** The traceability table correctly lists MOM-01..04 as Phase 24 / Complete. The coverage summary note at the bottom (`Last updated: 2026-05-27` and `all Pending`) is stale — multiple v1.4 requirements (FND-01, DRV1-01..05, MOM-01..04) are complete but the summary prose has not been updated. This is a documentation inconsistency only; the authoritative traceability table is correct.

### Anti-Patterns Found

| File                                      | Pattern                              | Severity   | Impact                                                                                                   |
|-------------------------------------------|--------------------------------------|------------|----------------------------------------------------------------------------------------------------------|
| `one_electron.rs` L5973, 5997, 6135, ... | `if dst < staging.len()` guard       | ⚠ Warning  | WR-02 from code review: silent truncation if staging is undersized. Contradicts OOM-safe contract. Deferred to Phase 25 (FND-06). No current corpus impact — staging is correctly sized by manifest-derived component_rank. |
| `raw.rs` L201-214                         | Stale forward-reference comment      | ℹ Info     | IN-04: comment says p4/irp "land in plans 24-04/24-05" but both are now registered. Misleads future readers. |
| `one_electron.rs` L4994                  | `_ => launch_with!(7u32, ...)` catch-all | ℹ Info | IN-02: invalid op_mode falls through to `zz` silently. Not reachable today; future dispatcher bug risk. |
| `one_electron.rs` L4986-4995 + L5844-5854 | Duplicated (op_mode→order,rank) mapping | ⚠ Warning | WR-01: three independent copies of the same source-of-truth. A future drift would cause buffer-overrun. Not live today. |
| `moment_common.rs` L116-120, 170-173     | Single `assert_any_nonzero` gate     | ⚠ Warning  | WR-04: high-rank families (rrrr=81, irp=9) could pass if only one component is non-zero. No nctr>1 test exists (WR-03). |
| No nctr>1 parity case for any Phase-24 family | Missing test coverage          | ⚠ Warning  | WR-03: all parity tests use H2O/STO-3G (nctr=1). The nctr>1 contraction-stride path is untested for the 12 new families. Pre-existing pattern; Phase 23 established the int3c1e_genctr_parity precedent but Phase 24 did not follow it. |

**Anti-pattern classification:**
- The `if dst < staging.len()` pattern (WR-02) is a known pre-existing pattern in the codebase deferred to Phase 25 (FND-06). It is NOT a Phase-24-introduced blocker — the staging is correctly sized by the manifest-derived component_rank and the vendor parity gate at atol=1e-12 would catch any truncation for the current corpus.
- WR-01, WR-03, WR-04 are robustness concerns raised by the code reviewer. They do not affect correctness on the current H2O/STO-3G corpus at atol=1e-12. WR-03 (no nctr>1 case) is the most meaningful gap for future regression safety.
- IN-02 and IN-04 are informational quality items with no correctness impact.

### Human Verification Required

None. All must-have truths are verifiable programmatically. The vendor parity results (18/18 GREEN) were confirmed by the orchestrator under the double gate (`--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`). No visual, real-time, or external-service behaviors to check.

### Gaps Summary

No blocking gaps. All four requirements (MOM-01, MOM-02, MOM-03, MOM-04) are satisfied:

- 54 manifest entries (18 families × 3 forms) are present with exact component_rank values and `oracle_covered=true` for all cart/sph forms.
- All RawApiId consts are present and string-exact to manifest symbols.
- Five device kernels are implemented and dispatch arms are live.
- 18/18 vendor parity tests GREEN at atol=1e-12 on the non-zero gauge-origin fixture.
- rrr_origj / rrrr_origj are correctly absent everywhere.
- PTR_RINV_ORIG vs PTR_COMMON_ORIG origin slot separation is correctly enforced.
- No capi enum variants or legacy cint* wrappers added (project scope constraint honored).
- OQ-2 cart_offset lib-unit failure confirmed pre-existing and de-blocked.

Warnings from the code review (WR-01 rank-tuple duplication, WR-02 soft staging guard, WR-03 no nctr>1 coverage, WR-04 weak non-zero gate) are carried forward. WR-02 and FND-06 are explicitly scheduled for Phase 25. WR-03 should be addressed in a follow-on plan or as part of a future phase's test hardening.

---

_Verified: 2026-05-30T08:30:00Z_
_Verifier: Claude (gsd-verifier)_
