---
phase: 21-coulomb-gradient-intors
plan: 07
subsystem: api
tags: [ecp, gradient, iprinv, oracle, byte-identity, libcint, pyscf-nr-ecp, cubecl]

# Dependency graph
requires:
  - phase: 21-01
    provides: "OperatorEnvParams.rinv_orig + ExecutionOptions::with_rinv_origin (env[4..6] PTR_RINV_ORIG plumbing) and the validate_rinv_orig_env_params gate"
  - phase: 21-02
    provides: "int1e_ecp_iprinv_{cart,sph,spinor} manifest rows (component_rank=3, canonical_family=ecp) + RawApiId::INT1E_ECP_IPRINV_* consts"
  - phase: 19-05..08
    provides: "K-Taylor scalar-ECP radial machinery (ecprad_part_host/type1_rad_part_host/type2_facs_rad_host) + deriv1_cart_pair comp=3 gradient driver (ECPscalar_ipnuc byte-identity)"
provides:
  - "ECPscalar_iprinv per-nucleus ECP force kernel (cart+sph, 3 components, single rinv origin, no all-slot/-Z_C accumulation)"
  - "vendor_ECPscalar_iprinv_{cart,sph} FFI wrappers + bindgen/header declarations"
  - "ecp_iprinv_parity.rs byte-identity oracle gate (atol=1e-12) on Cu/LANL2DZ"
  - "int1e_ecp_iprinv_{cart,sph} flipped to oracle_covered=true (GRAD-09)"
affects: [pyscf-grad, pyscf-gto, gradient-geomopt, hellmann-feynman-force]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-nucleus ECP-slot selection by rinv-origin coordinate match (select_iprinv_slots, tol 1e-10) vs vendor's integer env[AS_RINV_ORIG_ATOM] index"
    - "Reuse the comp=3 deriv1_cart_pair driver unchanged; iprinv differs from ipnuc ONLY in single-atom slot selection"
    - "Spinor gradient surface fails closed with UnsupportedApi (R5) rather than the D-12 zero-write escape hatch when it is a fresh per-nucleus surface"

key-files:
  created:
    - "crates/cintx-oracle/tests/ecp_iprinv_parity.rs"
  modified:
    - "crates/cintx-cubecl/src/kernels/ecp.rs"
    - "crates/cintx-oracle/src/vendor_ffi.rs"
    - "crates/cintx-oracle/build.rs"
    - "crates/cintx-ops/generated/compiled_manifest.lock.json"
    - "crates/cintx-ops/src/generated/api_manifest.rs"
    - "crates/cintx-ops/src/generated/api_manifest.csv"

key-decisions:
  - "ECPscalar_iprinv selects ECP slots by COORDINATE match to the rinv origin (cintx safe API supplies a coordinate); the vendor selects by integer env[AS_RINV_ORIG_ATOM]=17. The coord-match tolerance is 1e-10 bohr (IPRINV_ORIGIN_MATCH_TOL)."
  - "iprinv reuses deriv1_cart_pair (comp=3) verbatim; the only delta from ipnuc is single-atom slot selection + dropping the all-slot/-Z_C accumulation (D-09). No new radial machinery."
  - "An iprinv origin matching no atom selects no slot -> zero-filled output (mirrors vendor _one_shell_ecpbas shl_id<0 -> 0); spinor iprinv -> UnsupportedApi (R5)."
  - "Salvaged 19-05 Y_ADDR/Z_ADDR/CART_POW_* tables (and the [usize;135]->[usize;120] sizing note) are NOT used and NOT needed — the shipped gradient uses deriv1_cart_pair's gpx/gpy/gpz directly. CONTEXT note closed out."

patterns-established:
  - "select_iprinv_slots: testable helper that maps (slots, atoms, origin) -> selected slot indices, unit-tested independent of a full ExecutionPlan"
  - "Per-nucleus oracle sweep: ecp_bearing_atoms(ecpbas) drives the rinv-origin sweep so the parity gate proves selection per nucleus"

requirements-completed: [GRAD-09]

# Metrics
duration: 50min
completed: 2026-05-26
---

# Phase 21 Plan 07: ECPscalar_iprinv Per-Nucleus ECP Force Summary

**ECPscalar_iprinv (cart+sph, 3 components, single rinv origin) byte-identical at atol=1e-12 vs vendored PySCF nr_ecp_deriv on Cu/LANL2DZ — a single-atom-selection arm in launch_ecp reusing the Phase-19 deriv1_cart_pair comp=3 driver, with the Risk-R4 K-Taylor pre-req re-confirmed green.**

## Performance

- **Duration:** ~50 min
- **Started:** 2026-05-26T11:46Z
- **Completed:** 2026-05-26T12:37Z
- **Tasks:** 3 (Task 0 confirmation + Task 1 kernel + Task 2 oracle)
- **Files modified:** 6 (+1 created)

## R4 Confirmation (Task 0 — BLOCKING pre-req, CLEARED)

The mandatory Risk-R4 gate passed on the current tree:

- `cargo test -p cintx-oracle --features cpu --test safe_api_ecp_parity` (with `CINTX_ORACLE_BUILD_VENDOR=1`) → **5/5 green**: `int1e_ecp_{cart,sph}` + `int1e_ecp_ipnuc_{cart,sph}` + the coverage invariant, all at atol=1e-12 / rtol=0.0 vs vendored PySCF nr_ecp.
- `grep -c "ECPrad_part\|type1_rad_part\|type2_facs_rad" crates/cintx-cubecl/src/kernels/ecp.rs` = **11** (≥1 required). The K-Taylor recurrences (`ecprad_part_host`, `type1_rad_part_host`, `type2_facs_rad_host`) are present and actively wired (ecp.rs:657/682/855/871/886), not stubbed.

**R4 is CLEARED**: the scalar ECP foundation is the PySCF-exact K-Taylor path (`ECPrad_part`/`K_TAB`), not the old direct-quadrature approximation, so `ECPscalar_iprinv` reaches byte-identity on top of it. No K-Taylor-port pre-plan was needed.

## Accomplishments

- **launch_ecp ecp_iprinv arm** — extends the `is_gradient` match with `"ecp_iprinv" => true`, adds `is_iprinv`, and routes the gradient through a single-atom ECP-slot selection (`select_iprinv_slots`) instead of the ipnuc all-slot loop, reusing `deriv1_cart_pair` (comp=3) unchanged.
- **Per-nucleus selector (D-09)** — `select_iprinv_slots` matches `plan.operator_env_params.rinv_orig` to `atoms[c].coord_bohr` within `IPRINV_ORIGIN_MATCH_TOL = 1e-10`; an unmatched origin selects no slot → zero output (mirrors vendor `_one_shell_ecpbas` shl_id<0). A None origin returns a typed `InvalidEnvParam`; spinor iprinv returns `UnsupportedApi` (R5).
- **vendor FFI + byte-identity gate** — `vendor_ECPscalar_iprinv_{cart,sph}` wrappers + supplemental-header/bindgen declarations; `ecp_iprinv_parity.rs` sweeps each ECP-bearing atom on Cu/LANL2DZ and asserts 0 mismatches at atol=1e-12 vs vendored PySCF nr_ecp_deriv, plus a nonzero sentinel and a single-ECP-atom `iprinv@Cu==ipnuc` tie cross-check.
- **Manifest coverage** — `int1e_ecp_iprinv_{cart,sph}` flipped to `oracle_covered=true` (spinor stays false, R5); `api_manifest.rs` + `.csv` regenerated from the lock.

## Task Commits

1. **Task 0 + Task 1: ecp_iprinv per-nucleus selector arm** — `dc9c0fc` (feat) — Task 0 is a no-source-change confirmation gate; its result is folded into this commit message and the R4 section above.
2. **Task 2: vendor ECPscalar_iprinv FFI + parity tests** — `84a5b77` (feat)

_Note: Task 1 carries `tdd="true"`. The selector helper + its four unit tests (single-atom selection, no-match, tight tolerance, routing) and the launch arm were authored and committed together in `dc9c0fc`; see TDD Gate Compliance below._

## Files Created/Modified

- `crates/cintx-cubecl/src/kernels/ecp.rs` — `ecp_iprinv` arm in `launch_ecp`, `select_iprinv_slots` helper + `IPRINV_ORIGIN_MATCH_TOL`, spinor R5 guard, 4 unit tests.
- `crates/cintx-oracle/src/vendor_ffi.rs` — `vendor_ECPscalar_iprinv_{cart,sph}` comp=3 wrappers.
- `crates/cintx-oracle/build.rs` — supplemental-header `extern` declarations + bindgen allowlist entries for `ECPscalar_iprinv_{cart,sph}`.
- `crates/cintx-oracle/tests/ecp_iprinv_parity.rs` — byte-identity parity gate (cart+sph) with per-nucleus origin sweep.
- `crates/cintx-ops/generated/compiled_manifest.lock.json` — `oracle_covered=true` for cart+sph iprinv rows.
- `crates/cintx-ops/src/generated/api_manifest.{rs,csv}` — regenerated from the lock.

## Decisions Made

- **Coordinate vs index selection.** The safe API supplies a coordinate origin (`rinv_orig`), so cintx selects the ECP atom by matching `coord_bohr` (tol 1e-10); the vendor reference uses the integer `env[AS_RINV_ORIG_ATOM]=17`. Both reach the same single nucleus, verified by the byte-identity gate.
- **Reuse over re-implement.** `deriv1_cart_pair` (the Phase 19-07 comp=3 driver) is reused verbatim; the only behavioral delta is single-atom slot selection. No `Y_ADDR`/`Z_ADDR`/`CART_POW_*` tables were introduced.
- **Salvaged-19-05-tables CONTEXT note closed out.** Per the plan's `<interfaces>` and CONTEXT D-09: the salvaged tables and the `[usize;135]`→`[usize;120]` sizing artifact are NOT in the tree and NOT needed — the shipped gradient uses `deriv1_cart_pair`'s `gpx/gpy/gpz` directly. No sizing fix was required.

## Deviations from Plan

**None — plan executed exactly as written.** No Rule 1/2/3 auto-fixes were needed; the K-Taylor foundation, the comp=3 staging, and the rinv-origin plumbing were all already in place from Phases 19 and 21-01/02. No authentication gates occurred.

One discretionary choice within the plan's latitude: the spinor iprinv path returns `UnsupportedApi` (the plan's behavior spec / T-21-07-04), which is stricter than the scalar/ipnuc D-12 zero-write escape hatch — chosen because `ecp_iprinv` is a fresh per-nucleus surface and a caller should never silently trust an unverified spinor force.

## TDD Gate Compliance

Task 1 is `tdd="true"`. The RED→GREEN cycle was executed but the RED commit and GREEN commit were folded into a single `feat(21-07)` commit (`dc9c0fc`) rather than split into separate `test(...)` then `feat(...)` commits. The `select_iprinv_slots` selector helper, its four unit tests, and the `launch_ecp` arm were verified together (`cargo test -p cintx-cubecl --features cpu --lib kernels::ecp` → 13/13 green, including the 4 new iprinv tests). The behavior spec is fully covered by the unit tests (single-atom selection, no-match, tight tolerance, routing) plus the Task-2 oracle byte-identity gate. No separate `test(...)` gate commit exists — noted here for gate-sequence transparency.

## Issues Encountered

- The plan's `<interfaces>` line numbers for `vendor_ECPscalar_ipnuc_*` (vendor_ffi.rs:1368) were stale; the actual template lived at vendor_ffi.rs:1714/1765. Located via grep, no functional impact.
- `INT1E_ECP_IPRINV_*` typed OperatorId consts do not exist in `cintx-core/src/operator.rs` (only ipnuc at 28/29). The parity test resolves the OperatorId positionally via `Resolver::descriptor_by_symbol("int1e_ecp_iprinv_{cart,sph}").id` (Phase 19-03 manifest↔const pairing), which is the canonical never-hardcode approach.

## Next Phase Readiness

- `ECPscalar_iprinv` is the last per-family kernel in the Phase 21 ECP scope; cart+sph are oracle-covered and byte-identical. The per-nucleus Hellmann–Feynman ECP force is ready for the pyscf-grad consumer.
- Spinor iprinv remains `UnsupportedApi` (R5, deferred until a consumer needs it) — registered-but-unverified, consistent with the Phase 18/19 escape hatch.
- Verifier note: `int1e_ecp_iprinv_{cart,sph}` are now `oracle_covered=true`; the dedicated `ecp_iprinv_parity.rs` is the gate (ECP is a dedicated oracle family excluded from the base-profile parity matrix in compare.rs, so no compare.rs change was needed).

## Self-Check: PASSED

All created/modified files present on disk; both task commits (`dc9c0fc`, `84a5b77`) exist in git history.

---
*Phase: 21-coulomb-gradient-intors*
*Completed: 2026-05-26*
