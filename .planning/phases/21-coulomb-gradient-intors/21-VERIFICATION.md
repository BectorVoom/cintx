---
phase: 21-coulomb-gradient-intors
verified: 2026-05-26T22:30:00Z
status: human_needed
score: 9/10 must-haves verified
overrides_applied: 0
human_verification:
  - test: "Run the full vendor-gated gradient oracle suite: CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test two_electron_ip1_parity --test center_3c2e_parity --test one_electron_grad_parity --test one_electron_nuc_grad_parity --test ecp_iprinv_parity"
    expected: "All five test binaries exit 0 with 0 mismatches at atol=1e-12: two_electron_ip1_parity (2 vendor tests), center_3c2e_parity (1 vendor test), one_electron_grad_parity (4 vendor tests), one_electron_nuc_grad_parity (4 vendor tests), ecp_iprinv_parity (2 vendor tests). The parity assertions must EXECUTE — not silently skip — confirming CINTX_ORACLE_BUILD_VENDOR=1 was set."
    why_human: "Vendor-gated tests require CINTX_ORACLE_BUILD_VENDOR=1 and a vendored libcint build available in the environment; without them the parity assertions compile out (cfg gate: has_vendor_libcint / has_vendor_pyscf_nr_ecp) and silently produce 0 tests run. The orchestrator confirmed GREEN under both flags on the final merged tree but this verification agent cannot rebuild the vendor shim."
---

# Phase 21: Plain-Coulomb Gradient Integral Families Verification Report

**Phase Goal:** cintx implements the 6 plain-Coulomb first-derivative integral families (int2e_ip1, int1e_ipovlp, int1e_ipkin, int1e_ipnuc, int1e_iprinv, ECPscalar_iprinv) byte-identical to libcint 6.1.3 under the oracle gate, repairs the registered-but-stubbed int3c2e_ip1, and adds the PTR_RINV_ORIG env slot.
**Verified:** 2026-05-26T22:30:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | PTR_RINV_ORIG env slot plumbed end-to-end (GRAD-01) | VERIFIED | `planner.rs`: `rinv_orig: Option<[f64; 3]>` field confirmed at line 53. `validator.rs`: `validate_rinv_orig_env_params` at line 182 with `"PTR_RINV_ORIG"` typed error. `raw.rs`: `pub const PTR_RINV_ORIG: usize = 4` at line 49, read block at lines 599-611, `is_iprinv_family_symbol` predicate at line 764. `cintx-rs/src/builder.rs`: `with_rinv_origin` setter at line 102. 5 unit tests pass: `cargo test -p cintx-runtime rinv_orig` exits 0 with 5/5. |
| 2 | All 6 families + int3c2e_ip1 registered in manifest with component_rank 3 and matching RawApiId consts / legacy wrappers (GRAD-02) | VERIFIED | Manifest confirms all 21 gradient operator rows (7 families × 3 reps). `raw.rs` lines 147–185: RawApiId consts for all 7 families (INT1E_IPOVLP_*, INT1E_IPKIN_*, INT1E_IPNUC_*, INT1E_IPRINV_*, INT2E_IP1_*, INT3C2E_IP1_*, INT1E_ECP_IPRINV_*). `legacy.rs` lines 228–280: all cint legacy macro wrappers present. Manifest-audit exits 0 with status:ok, has_mismatch:false, 0 uncovered_stable_entries. `CINTX_BACKEND=cpu cargo check --workspace --features cpu` exits 0. |
| 3 | int1e_ipovlp (cart + sph, 3 components) byte-identical to libcint 6.1.3 at atol=1e-12 (GRAD-03) | VERIFIED* | Kernel branch confirmed in `one_electron.rs` line 883 (`is_ipovlp`), gradient path at line 919. Determinism tests (`test_int1e_ipovlp_sph_determinism`, `test_int1e_ipovlp_cart_determinism`) pass. `oracle_covered=True` for cart+sph in manifest. Vendor parity tests (`test_int1e_ipovlp_sph_h2o_sto3g_parity` / `cart`) exist in `one_electron_grad_parity.rs` gated on `has_vendor_libcint`. Orchestrator confirmed GREEN at atol=1e-12. (*Human vendor-build confirmation required — see below.) |
| 4 | int1e_ipkin (cart + sph, 3 components) byte-identical to libcint 6.1.3 at atol=1e-12 (GRAD-04) | VERIFIED* | `contract_ipkin` at `one_electron.rs` line 373. Determinism tests pass. Vendor parity tests in `one_electron_grad_parity.rs` gated on `has_vendor_libcint`. `oracle_covered=True` for cart+sph. |
| 5 | int1e_ipnuc (cart + sph, 3 components) byte-identical to libcint 6.1.3 at atol=1e-12 (GRAD-05) | VERIFIED* | Atom-loop branch in `one_electron.rs` line 884 (`is_ipnuc`), gradient path lines 936–1000. Determinism test (`test_int1e_ipnuc_sph_determinism`) passes. Vendor parity tests (`test_int1e_ipnuc_sph_h2o_sto3g_parity` / `cart`) in `one_electron_nuc_grad_parity.rs` gated on `has_vendor_libcint`. `oracle_covered=True` for cart+sph. |
| 6 | int1e_iprinv (cart + sph, 3 components, single rinv origin) byte-identical to libcint 6.1.3 at atol=1e-12 (GRAD-06) | VERIFIED* | Single-origin branch in `one_electron.rs` line 886 (`is_iprinv`), origin resolved at lines 920–933. Determinism / origin-sensitivity test (`test_int1e_iprinv_sph_origin_sensitivity`) passes. Vendor parity tests (`test_int1e_iprinv_sph_h2o_sto3g_parity` / `cart`, sweeping rinv over each nucleus) gated on `has_vendor_libcint`. `oracle_covered=True` for cart+sph. |
| 7 | int2e_ip1 (arity-4, 3 components, component-leading [3,nl,nk,nj,ni] F-order) byte-identical at atol=1e-12 for s/p/d quartets (GRAD-07) | VERIFIED* | `launch_two_electron_ip1` at `two_electron.rs` line 617. `gout_ip1` verbatim reuse from `f12.rs`. Determinism + nonzero sentinel test passes. Vendor parity tests (`oracle_parity_int2e_ip1_sph_spd` / `cart`) gated on `has_vendor_libcint`. Test file comment at lines 14-17 confirms element-for-element comparison IS the F-order layout gate. `oracle_covered=True` for cart+sph. |
| 8 | int3c2e_ip1 ships real derivative kernel (R1 stub replaced) with oracle reference flipped to vendor_int3c2e_ip1; matches at atol=1e-12 (GRAD-08) | VERIFIED* | `launch_center_3c2e_ip1` at `center_3c2e.rs` line 339. R1 unit tests at lines 1056-1142 prove: (a) 3-component output (not scalar), (b) all-component lanes are nonzero, (c) output differs from plain scalar 3c2e. Vendor parity test `test_center_3c2e_sph_h2o_sto3g_vendor_parity` in `center_3c2e_parity.rs` line 225 uses `vendor_int3c2e_ip1_sph` (the R1 flip). CR-01 fix (commit f470329) removed the broken scalar-buffer legacy-parity block. `oracle_covered=True` for cart+sph. |
| 9 | ECPscalar_iprinv (per-nucleus ECP force, single rinv origin) byte-identical to libcint at atol=1e-12 on Cu/LANL2DZ (GRAD-09) | VERIFIED* | `select_iprinv_slots` at `ecp.rs` line 612. `is_iprinv` path at line 1433. Spinor rejected with UnsupportedApi at line 1490. Vendor parity tests `test_ECPscalar_iprinv_cart_cu_lanl2dz_parity` / `_sph` in `ecp_iprinv_parity.rs` gated on `has_vendor_libcint AND has_vendor_pyscf_nr_ecp`. `oracle_covered=True` for cart+sph. |
| 10 | Phase verification: F-order layout validated, manifest oracle_covered flips green, pyscf_rs handoff note exists, ROADMAP/STATE/REQUIREMENTS updated (GRAD-10) | PARTIAL | **Layout validation**: documented in `phase-21-pyscf-rs-handoff.md` section 2 — element-for-element byte-identity IS the gate. **Manifest**: all 14 gradient cart+sph operator rows are `oracle_covered=True`; spinor stays False (R5); manifest-audit exits 0. **Handoff note**: `.planning/notes/phase-21-pyscf-rs-handoff.md` exists with 15 references to workflow_dispatch/RHF/geomopt/int3c2e_ip1. **ROADMAP/STATE/REQUIREMENTS**: partially updated — v1.3 milestone completed list shows `[x] Phase 21 (completed 2026-05-26)` and all 8 plan checkboxes `[x]`, but the Progress table row still reads `0/8 | Planned`, STATE.md still says `Executing Phase 21`, and GRAD-05..GRAD-10 checkboxes still `[ ]` in REQUIREMENTS.md. Orchestrator SUMMARY-08 explicitly states these finalization edits are orchestrator-owned (worktree auto-skip). This is documented behavior, not a defect. |

**Score:** 9/10 truths verified (Truth 10 is PARTIAL — the implementation-level work is done; the tracking-doc finalization is orchestrator-owned and explicitly deferred)

Notes on VERIFIED* items: These truths are substantively VERIFIED through code existence, wiring, determinism tests, manifest oracle_covered=True, and orchestrator-confirmed GREEN vendor parity. The asterisk indicates the vendor byte-identity assertion itself requires CINTX_ORACLE_BUILD_VENDOR=1 which cannot be replicated programmatically in this environment.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/cintx-runtime/src/planner.rs` | rinv_orig field + OperatorEnvParams | VERIFIED | `pub rinv_orig: Option<[f64; 3]>` at line 53 with doc comment |
| `crates/cintx-runtime/src/validator.rs` | validate_rinv_orig_env_params gate | VERIFIED | Function at line 182; 5 unit tests passing |
| `crates/cintx-compat/src/raw.rs` | PTR_RINV_ORIG const + eval_raw block + predicate | VERIFIED | Const at line 49; read block lines 599-611; predicate at line 764 |
| `crates/cintx-rs/src/builder.rs` | with_rinv_origin setter | VERIFIED | `pub fn with_rinv_origin` at line 102 |
| `crates/cintx-ops/generated/compiled_manifest.lock.json` | All 7 gradient families registered; cart+sph oracle_covered=True | VERIFIED | 21 operator rows confirmed; 14 cart+sph True; 7 spinor False (R5 by design) |
| `crates/cintx-cubecl/src/kernels/one_electron.rs` | ipovlp/ipkin/ipnuc/iprinv gradient kernels | VERIFIED | All 4 gradient branches wired; spinor guard present; 3-component staging |
| `crates/cintx-cubecl/src/kernels/two_electron.rs` | int2e_ip1 gradient kernel | VERIFIED | `launch_two_electron_ip1` at line 617; gout_ip1 reuse; F-order layout |
| `crates/cintx-cubecl/src/kernels/center_3c2e.rs` | int3c2e_ip1 real derivative kernel | VERIFIED | `launch_center_3c2e_ip1` at line 339; R1 unit tests confirm scalar stub is gone |
| `crates/cintx-cubecl/src/kernels/ecp.rs` | ECPscalar_iprinv per-nucleus selector | VERIFIED | `select_iprinv_slots` + `is_iprinv` dispatch path; single-atom selection |
| `crates/cintx-oracle/tests/one_electron_grad_parity.rs` | ipovlp/ipkin vendor parity tests | VERIFIED | Tests exist; determinism tests pass; vendor tests gated on has_vendor_libcint |
| `crates/cintx-oracle/tests/one_electron_nuc_grad_parity.rs` | ipnuc/iprinv vendor parity tests | VERIFIED | Tests exist; determinism tests pass; sweep over each H2O nucleus for iprinv |
| `crates/cintx-oracle/tests/two_electron_ip1_parity.rs` | int2e_ip1 F-order + vendor parity | VERIFIED | Tests exist; determinism+sentinel test passes |
| `crates/cintx-oracle/tests/center_3c2e_parity.rs` | int3c2e_ip1 real gradient parity | VERIFIED | vendor_int3c2e_ip1_sph reference (R1 flip) confirmed at line 254 |
| `crates/cintx-oracle/tests/ecp_iprinv_parity.rs` | ECPscalar_iprinv Cu/LANL2DZ parity | VERIFIED | Tests exist; gated on has_vendor_libcint AND has_vendor_pyscf_nr_ecp |
| `crates/cintx-oracle/src/vendor_ffi.rs` | All gradient vendor FFI wrappers | VERIFIED | vendor_int1e_ipovlp/ipkin/ipnuc/iprinv/int2e_ip1/int3c2e_ip1/ECPscalar_iprinv all present |
| `crates/cintx-oracle/src/compare.rs` | CR-01 fix: broken int3c2e_ip1 legacy-parity blocks removed | VERIFIED | Lines 937-942 contain comment redirecting to dedicated center_3c2e_parity.rs; no scalar-buffer block for sph or cart int3c2e_ip1 in verify_legacy_wrapper_parity |
| `.planning/notes/phase-21-pyscf-rs-handoff.md` | pyscf_rs Phase 7 un-gate note + R3 validation + int3c2e_ip1 R1 history | VERIFIED | File exists; 15 keyword matches for workflow_dispatch/RHF/geomopt/int3c2e_ip1 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `raw.rs::eval_raw` | `validator::validate_rinv_orig_env_params` | call after reading env[4..7] | VERIFIED | `raw.rs` line 611: `cintx_runtime::validator::validate_rinv_orig_env_params(plan.descriptor.operator_name(), &plan.operator_env_params)?` |
| `raw.rs::eval_raw` | `plan.operator_env_params.rinv_orig` | env[PTR_RINV_ORIG..+3] read block | VERIFIED | `raw.rs` lines 605-609: `rinv_orig = Some([x, y, z])` guarded by `env.len() >= PTR_RINV_ORIG + 3` |
| `one_electron.rs::launch_1e` | `iprinv_origin` from `plan.operator_env_params.rinv_orig` | destructuring + defensive gate | VERIFIED | `one_electron.rs` lines 920-933: resolves rinv_orig with typed error on None |
| `ecp.rs::launch_ecp` | `select_iprinv_slots` | rinv_orig resolution + Euclidean distance match | VERIFIED | `ecp.rs` lines 1533-1549: origin resolved from `plan.operator_env_params.rinv_orig` |
| `center_3c2e.rs` | `launch_center_3c2e_ip1` | `plan.descriptor.operator_name() == "ip1"` | VERIFIED | `center_3c2e.rs` line 616 |
| `two_electron.rs` | `launch_two_electron_ip1` | `plan.descriptor.operator_name() == "ip1"` | VERIFIED | `two_electron.rs` line 956-958 |
| `center_3c2e_parity.rs` | `vendor_int3c2e_ip1_sph` | R1 oracle flip | VERIFIED | `center_3c2e_parity.rs` line 254: calls `vendor_ffi::vendor_int3c2e_ip1_sph` (not plain vendor_int3c2e_sph) |
| `REQUIREMENTS.md` | GRAD-01..GRAD-10 traceability | status flipped to Complete | PARTIAL | GRAD-01..04 show `[x]` + traceability Complete; GRAD-05..10 still `[ ]` + Pending — orchestrator-owned finalization (SUMMARY-08 documented) |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `one_electron.rs` gradient path | staging (3-component) | `contract_grad_1e_bra` / `contract_ipkin` / `contract_ipnuc_iprinv` Rys contraction | Yes — G-tensor from Obara-Saika + Rys quadrature, not hardcoded | FLOWING |
| `two_electron.rs::launch_two_electron_ip1` | staging (3-component) | `fill_g_tensor_2e` + `rys_roots_host` + `gout_ip1` | Yes — plain Coulomb G-tensor with li+1 headroom | FLOWING |
| `center_3c2e.rs::launch_center_3c2e_ip1` | staging (3-component) | `fill_g_tensor_3c2e` + `gout_ip1` | Yes — 3c2e G-tensor with Pitfall-4 kl mapping | FLOWING |
| `ecp.rs` iprinv path | gctr (3-component gradient) | `compute_type1_pair_grad` + `compute_type2_pair_grad` on selected atom only | Yes — K-Taylor radial tables (Phase 19), single-atom selection via `select_iprinv_slots` | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| PTR_RINV_ORIG validator rejects None for iprinv | `cargo test -p cintx-runtime rinv_orig` | 5/5 passing in 0.00s | PASS |
| 1e gradient determinism (ipovlp/ipkin) | `cargo test -p cintx-oracle --features cpu --test one_electron_grad_parity` | 4/4 passing | PASS |
| 1e nuclear gradient determinism (ipnuc/iprinv) | `cargo test -p cintx-oracle --features cpu --test one_electron_nuc_grad_parity` | 2/2 passing | PASS |
| int2e_ip1 determinism + nonzero sentinel | `cargo test -p cintx-oracle --features cpu --test two_electron_ip1_parity` | 1/1 passing | PASS |
| int3c2e_ip1 nonzero + idempotency | `cargo test -p cintx-oracle --features cpu --test center_3c2e_parity` | 1/1 passing | PASS |
| Manifest audit | `cargo run --manifest-path xtask/Cargo.toml -- manifest-audit` | status:ok, has_mismatch:false, 0 uncovered_stable_entries | PASS |
| Workspace compile | `CINTX_BACKEND=cpu cargo check --workspace --features cpu` | exits 0 | PASS |
| Vendor byte-identity parity (all 5 gradient suites) | Requires CINTX_ORACLE_BUILD_VENDOR=1 + --features cpu | Cannot run without vendor shim in this environment | SKIP (human verification required) |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| GRAD-01 | 21-01 | PTR_RINV_ORIG env slot end-to-end | SATISFIED | rinv_orig field, validator, PTR_RINV_ORIG=4, eval_raw read block, with_rinv_origin setter; 5 unit tests green |
| GRAD-02 | 21-02 | 6 families + int3c2e_ip1 registered with component_rank:3, RawApiId, legacy wrappers, CAPI | SATISFIED | All 21 manifest rows confirmed; RawApiId consts in raw.rs; legacy wrappers in legacy.rs; manifest-audit green |
| GRAD-03 | 21-03 | int1e_ipovlp at atol=1e-12 | SATISFIED (vendor TBC) | Kernel exists; oracle_covered=True; determinism tests pass; vendor parity confirmed by orchestrator |
| GRAD-04 | 21-03 | int1e_ipkin at atol=1e-12 | SATISFIED (vendor TBC) | Kernel exists; oracle_covered=True; determinism tests pass; vendor parity confirmed by orchestrator |
| GRAD-05 | 21-04 | int1e_ipnuc at atol=1e-12 | SATISFIED (vendor TBC) | Atom-loop kernel exists; oracle_covered=True; determinism tests pass; vendor parity confirmed by orchestrator |
| GRAD-06 | 21-04 | int1e_iprinv at atol=1e-12 | SATISFIED (vendor TBC) | Single-origin kernel exists; oracle_covered=True; origin-sensitivity test passes; vendor parity confirmed by orchestrator |
| GRAD-07 | 21-05 | int2e_ip1 F-order at atol=1e-12 | SATISFIED (vendor TBC) | Kernel exists; F-order confirmed in test comments; oracle_covered=True; vendor parity confirmed by orchestrator |
| GRAD-08 | 21-06 | int3c2e_ip1 real derivative kernel + oracle flip | SATISFIED (vendor TBC) | Real kernel replaces stub; R1 unit tests prove scalar stub gone; oracle reference flipped to vendor_int3c2e_ip1; oracle_covered=True |
| GRAD-09 | 21-07 | ECPscalar_iprinv at atol=1e-12 on Cu/LANL2DZ | SATISFIED (vendor TBC) | Per-nucleus selector kernel exists; oracle_covered=True; vendor parity confirmed by orchestrator |
| GRAD-10 | 21-08 | Phase verification + F-order validation + handoff note + ROADMAP/STATE/REQUIREMENTS | PARTIALLY SATISFIED | F-order validated, manifest flips done, handoff note exists; ROADMAP progress table / STATE / GRAD-05..10 checkboxes still pending orchestrator finalization (documented intent) |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `two_electron.rs` | 783, 824, 1061, 1119 | `if dst < staging.len()` scatter guard — silent partial write instead of typed error | Warning (WR-03 from code review) | Violates CLAUDE.md "no partial writes" contract; a staging-buffer sizing regression produces truncated gradient silently. Not addressed post-review. |
| `center_3c2e.rs` | 497, 531 | Same `if dst < staging.len()` scatter guard | Warning (WR-03) | Same impact — gradient component may be silently dropped |
| `one_electron.rs` | 1026, 1050, 1192, 1232 | Same `if dst < staging.len()` scatter guard | Warning (WR-03) | Same impact |
| `compare.rs` | 147, 218, 362 | `Box::leak` on tolerance fast path for unknown family strings | Warning (WR-02 from code review) | Unbounded leak if many unique family strings; low risk in current call patterns |
| `compare.rs` | 424-436 | No explicit spinor arm in `eval_legacy_symbol` for new ip families | Warning (WR-04 from code review) | If spinor-gradient exclusion in fixtures.rs regresses, error message is misleading |
| `ecp.rs` | 612-624 | `select_iprinv_slots` uses Euclidean distance match rather than integer atom index | Warning (WR-06 from code review) | Diverges from vendor's integer-index selection; untested for degenerate multi-center ECP geometries |

All blockers from the code review (CR-01, CR-02) have been addressed:
- CR-01: Fixed in commit `f470329` — the broken int3c2e_ip1 scalar-buffer blocks were removed from `verify_legacy_wrapper_parity`; gradient byte-identity is now exclusively owned by the dedicated `*_parity.rs` test files.
- CR-02: Addressed by the same commit — the matrix-driven parity path for gradient families routes correctly; the misleading "upstream proxy" label is gone.

The remaining six warnings (WR-01 through WR-06) are robustness / maintainability concerns that do not block the phase goal. WR-03 (silent scatter truncation) is the most concerning because it violates the project's stated no-partial-writes contract, but is not a BLOCKER for the current oracle-verified implementations.

### Human Verification Required

#### 1. Vendor-Gated Byte-Identity Oracle Suite

**Test:** With CINTX_ORACLE_BUILD_VENDOR=1 set, run:

```
CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
  --test two_electron_ip1_parity \
  --test center_3c2e_parity \
  --test one_electron_grad_parity \
  --test one_electron_nuc_grad_parity \
  --test ecp_iprinv_parity
```

**Expected:** All five binaries exit 0. Confirm the tests EXECUTE and are not silently skipped — the output should show `running N tests` where N > 0 for each binary (not `running 0 tests`). Expected counts:
- `two_electron_ip1_parity`: 2 vendor tests (`oracle_parity_int2e_ip1_sph_spd`, `oracle_parity_int2e_ip1_cart_spd`)
- `center_3c2e_parity`: 1 vendor test (`test_center_3c2e_sph_h2o_sto3g_vendor_parity`)
- `one_electron_grad_parity`: 4 vendor tests (ipovlp and ipkin, sph and cart)
- `one_electron_nuc_grad_parity`: 4 vendor tests (ipnuc and iprinv, sph and cart)
- `ecp_iprinv_parity`: 2 vendor tests (`test_ECPscalar_iprinv_cart_cu_lanl2dz_parity`, `test_ECPscalar_iprinv_sph_cu_lanl2dz_parity`)

All assertions must report 0 mismatches at atol=1e-12.

**Why human:** The parity tests are double-gated on `--features cpu` AND `CINTX_ORACLE_BUILD_VENDOR=1` → the `has_vendor_libcint` / `has_vendor_pyscf_nr_ecp` cfg. Without both flags, parity assertions compile out and the test binary reports `running 0 tests` — producing a silently vacuous pass. This environment cannot rebuild the vendored libcint shim.

### Gaps Summary

No BLOCKER gaps were found. All 6 gradient kernel families have substantive implementations, real parity test suites, and `oracle_covered=True` in the manifest. The int3c2e_ip1 stub has been replaced with a real derivative kernel and the oracle reference flipped. The CR-01 and CR-02 code review blockers were addressed in commit `f470329`.

The one PARTIAL truth (GRAD-10: tracking-doc finalization) is entirely orchestrator-owned by documented intent in SUMMARY-08 and the orchestrator context. The ROADMAP progress table row and STATE still reflect the pre-completion state; GRAD-05..10 checkboxes in REQUIREMENTS.md are still `[ ]`. These are not implementation defects — the phase.complete step is what closes them.

The human_needed status reflects one non-negotiable constraint: the vendor byte-identity claim (atol=1e-12 for all 6 gradient families) cannot be verified programmatically in this environment without the vendored libcint build. The orchestrator's explicit GREEN confirmation is acknowledged but cannot substitute for a re-run here.

---

_Verified: 2026-05-26T22:30:00Z_
_Verifier: Claude (gsd-verifier)_
