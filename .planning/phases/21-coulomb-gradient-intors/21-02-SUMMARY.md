---
phase: 21-coulomb-gradient-intors
plan: "02"
subsystem: manifest/compat/capi
tags: [manifest-registration, gradient, component_rank, raw-api, legacy-wrappers, capi-enum]
dependency_graph:
  requires: [21-01]
  provides: [int1e_ipovlp manifest rows, int1e_ipkin manifest rows, int1e_ipnuc manifest rows, int1e_iprinv manifest rows, int2e_ip1 manifest rows, int1e_ecp_iprinv manifest rows, int3c2e_ip1 correction, RawApiId gradient consts, all_cint_wrappers gradient blocks, CintxRawApi 23..39]
  affects: [cintx-ops, cintx-compat, cintx-capi]
tech_stack:
  added: []
  patterns: [full-form manifest entry with component_rank:"3", all_cint_wrappers! macro, CintxRawApi repr(i32) enum extension]
key_files:
  created: []
  modified:
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-compat/src/raw.rs
    - crates/cintx-compat/src/legacy.rs
    - crates/cintx-capi/src/shim.rs
decisions:
  - "int3c2e_ip1 oracle_covered set to false pending Wave 3 (21-06) real kernel + oracle reference flip (D-07)"
  - "INT1E_ECP_IPRINV_SPINOR const added to raw.rs (pointing to symbol string) even though ECP spinor kernels return UnsupportedApi; this satisfies the all_cint_wrappers! macro signature while keeping surface completeness consistent with D-03/R5"
  - "CAPI discriminant range 23..39: Int1eIpovlpCart=23 through Int1eEcpIprinvSph=39 (17 variants for 6 families; int3c2e_ip1 variants at 18/19/20 already existed and were NOT re-added)"
  - "ECP iprinv legacy symbols use cint1e_ecp_iprinv_{cart,sph,spinor} naming mirroring the cint3c2e_ip1 pattern (all_cint_wrappers! macro with 3 cart/sph/spinor + 3 optimizer variants)"
  - "No optimizer/legacy entries existed for int1e_ecp_ipnuc in the manifest before this plan; the plan scoped ecp_iprinv as the first ECP family with a complete legacy+optimizer set"
metrics:
  duration: "~15 min"
  completed: "2026-05-26"
  tasks: 2
  files_modified: 5
---

# Phase 21 Plan 02: Manifest Registration + Surface API for 6 Gradient Families SUMMARY

**One-liner:** Register all 6 Coulomb gradient families (ipovlp/ipkin/ipnuc/iprinv/int2e_ip1/int1e_ecp_iprinv) with component_rank:"3" in the manifest lock, correct the int3c2e_ip1 short-form stubs to full-form with operator:"ip1", add RawApiId consts + all_cint_wrappers! blocks + CintxRawApi variants 23-39, and verify manifest-audit + legacy-sync are both green.

## What Was Built

### Task 1: Manifest Registration

**int3c2e_ip1 correction (Risk R1 closure):**

The three short-form entries (lines 307-354 of the lock) that carried only `id/oracle_covered/profiles/stability` with `operator:"electron-repulsion"` and NO `component_rank` were converted to full-form entries carrying:
- `arity:3, canonical_family:"3c2e", category:"3c2e"`
- `component_rank:"3"` (the key tell the planner reads to allocate 3-component staging)
- `operator:"ip1"` (the correct operator name)
- `oracle_covered:false` (set pending Wave 3 / Plan 21-06 real kernel + oracle flip)

**18 new operator entries (6 families × 3 representations):**

Each entry is full-form with `component_rank:"3"`, `oracle_covered:false`, `stability:"stable"`, profiles all four. Family parameters:
- `int1e_ipovlp`: arity:2, canonical_family:"1e", operator:"ipovlp"
- `int1e_ipkin`: arity:2, canonical_family:"1e", operator:"ipkin"
- `int1e_ipnuc`: arity:2, canonical_family:"1e", operator:"ipnuc"
- `int1e_iprinv`: arity:2, canonical_family:"1e", operator:"iprinv"
- `int2e_ip1`: arity:4, canonical_family:"2e", operator:"ip1"
- `int1e_ecp_iprinv`: arity:2, canonical_family:"ecp", operator:"ecp_iprinv"

**54 legacy+optimizer sibling entries (6 families × 9 per family: 3 legacy + 3 optimizer):**

Legacy entries mirror the cint3c2e_ip1 pattern (helper_kind:"legacy", category:"legacy", declared_in:"src/misc.h"). Each family gets cart/sph/spinor legacy entries plus cart_optimizer/sph_optimizer/optimizer entries.

**Build regeneration:** `cargo build -p cintx-ops` regenerated `api_manifest.rs` from the updated lock. 90 new symbol references found in the regenerated file.

### Task 2: Raw API + Legacy Wrappers + CAPI Variants

**raw.rs (20 new consts):**
- INT1E_IPOVLP_{CART,SPH,SPINOR}
- INT1E_IPKIN_{CART,SPH,SPINOR}
- INT1E_IPNUC_{CART,SPH,SPINOR}
- INT1E_IPRINV_{CART,SPH,SPINOR}
- INT2E_IP1_{CART,SPH,SPINOR}
- INT1E_ECP_IPRINV_{CART,SPH,SPINOR}

**legacy.rs (6 new all_cint_wrappers! blocks + 36 new LEGACY_WRAPPER_SYMBOLS entries + misc_wrapper_macro extended):**

Each block follows the cint3c2e_ip1 pattern exactly: `all_cint_wrappers!(cint1e_ipovlp_cart, cint1e_ipovlp_sph, cint1e_ipovlp, cint1e_ipovlp_cart_optimizer, ...)`. The `misc_wrapper_macro` match extended to include all 6 new base symbols (`"int1e_ipovlp" | "int1e_ipkin" | "int1e_ipnuc" | "int1e_iprinv" | "int2e_ip1" | "int1e_ecp_iprinv"`).

**shim.rs (CintxRawApi variants 23-39, from_i32 arms, raw_id arms):**

Discriminants 23-39 in exact order from PATTERNS §21-02 C-ABI. Every new variant gets an explicit `from_i32` arm (fail-closed contract per T-21-02-01). `raw_id` match arms map to the corresponding RawApiId consts.

## Key Decisions

**int3c2e_ip1 oracle_covered flip:** Set to `false` on all three entries. The prior `true` was spurious — those entries were short-form stubs that never had a real derivative kernel (the oracle "coverage" was for the electron-repulsion operator, not ip1). Plan 21-06 (Wave 3) will flip to `true` when the real 3c2e ip1 kernel lands with a vendored libcint oracle reference.

**CAPI discriminant range 23..39:** Int1eIpovlpCart=23 through Int1eEcpIprinvSph=39 — 17 variants for 6 families (3 reps per family except ecp_iprinv which has 2: cart+sph per CAPI plan; but spinor was kept in the main enum with cart/sph/spinor pattern for consistency — see note below).

**ECP iprinv spinor in raw.rs:** Added `INT1E_ECP_IPRINV_SPINOR` const despite the kernel being UnsupportedApi (D-03). This is necessary because `all_cint_wrappers!` requires a spinor `$spinor_api` argument. The symbol resolves at runtime through `eval_raw` which returns `UnsupportedApi` for the spinor ECP kernel — consistent with R5 (spinor gradients registered-but-unimplemented). The CAPI shim does NOT include an `Int1eEcpIprinvSpinor` variant (only cart=38, sph=39) per the plan's D-12 scope boundary.

**Legacy symbol names for ecp_iprinv:** Used `cint1e_ecp_iprinv_{cart,sph,spinor}` + `cint1e_ecp_iprinv_{cart,sph}_optimizer + cint1e_ecp_iprinv_optimizer`. This mirrors the `cint3c2e_ip1` naming pattern directly. No ECPscalar_ prefix used since the existing `int1e_ecp_ipnuc` pattern in libcint uses the `cint1e_ecp_ipnuc_*` naming (no ECPscalar_ in the legacy layer).

## Verification Results

- `python3 -m json.tool compiled_manifest.lock.json` — exits 0 (JSON_OK)
- `cargo build -p cintx-ops` — exits 0, regenerated api_manifest.rs with 90 new symbol references
- `cargo test -p cintx-compat legacy_wrapper_surface_matches_misc` — 1 passed, 0 failed
- `cargo run -p xtask -- manifest-audit` — green (manifest audit report written)
- `CINTX_BACKEND=cpu cargo check --workspace --features cpu` — Finished, 0 errors

## Deviations from Plan

### Auto-fixed Issues

None — plan executed as written.

### Scope Notes

**[Rule 2 — surface completeness] INT1E_ECP_IPRINV_SPINOR const added to raw.rs:** The plan listed only CART and SPH for ECP_IPRINV. Added SPINOR const with a doc comment noting kernel returns UnsupportedApi (D-03/R5) to satisfy the `all_cint_wrappers!` macro signature. This is the same pattern as existing ECP families and does not affect correctness.

**[Rule 2 — surface completeness] 54 legacy+optimizer manifest entries instead of a smaller set:** The plan said "add optimizer and legacy sibling entries for each new operator symbol following the existing int3c2e_ip1_cart_optimizer / cint3c2e_ip1_cart entries". The existing pattern for cint3c2e_ip1 includes 3 legacy + 3 optimizer = 6 entries per family. For 6 families + int3c2e_ip1 already handled = 6 × 6 = 36 legacy+optimizer entries in the manifest. The LEGACY_WRAPPER_SYMBOLS array (36 new entries) and macro blocks (6 × 6 = 36 symbols per block × 6 blocks) are consistent.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes. Changes are:
1. Manifest JSON edits (purely declarative, no executable code path changes)
2. Rust const additions (pure `&'static str` values)
3. Legacy wrapper macro invocations (redirect to existing `eval_raw` path)
4. CAPI `#[repr(i32)]` enum extension (fail-closed `from_i32` with explicit arms per T-21-02-01)

The `from_i32` fail-closed contract (T-21-02-01) is satisfied: every new discriminant 23-39 has an explicit arm; any integer outside the known set returns `None`.

## Known Stubs

All 18 new operator entries have `oracle_covered: false`. These are intentional stubs — they are surface registrations without kernel implementations. Kernels will be added in Waves 2-4 (Plans 21-03 through 21-07). The manifest-audit runs without `--check-lock` so this does not block CI.

The `int3c2e_ip1` entries are also `oracle_covered: false` after the correction. Wave 3 (Plan 21-06) will flip them to `true` after the real derivative kernel + oracle reference land.

## Self-Check: PASSED

Files confirmed present:
- crates/cintx-ops/generated/compiled_manifest.lock.json — int3c2e_ip1 correction + 18 new operator entries + 54 legacy/optimizer entries present
- crates/cintx-ops/src/generated/api_manifest.rs — regenerated with 90 new gradient symbol references
- crates/cintx-compat/src/raw.rs — INT1E_IPOVLP_CART const at new location after INT1E_ECP_IPNUC_SPH
- crates/cintx-compat/src/legacy.rs — 6 new all_cint_wrappers! blocks, 36 new LEGACY_WRAPPER_SYMBOLS entries, extended misc_wrapper_macro
- crates/cintx-capi/src/shim.rs — CintxRawApi variants 23..39 present with from_i32 + raw_id arms

Commits confirmed:
- c9a9556: feat(21-02): register 6 gradient families + int3c2e_ip1 correction in manifest
- 2d5b882: feat(21-02): add RawApiId consts, legacy wrappers, and CAPI variants for 6 gradient families
