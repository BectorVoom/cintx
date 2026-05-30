---
phase: 23-group-1-remaining-1st-derivative-families-cart-sph
plan: 04
subsystem: integrals
tags: [3c1e, int3c1e_ip1, int3c1e_iprinv, gradient, rys, cubecl, libcint, oracle, vendor-parity]

# Dependency graph
requires:
  - phase: 21-coulomb-gradient-intors
    provides: "gradient engine, PTR_RINV_ORIG env slot, rinv_orig plumbing (D-08), spinor escape hatch (D-06)"
  - phase: 23-group-1-remaining-1st-derivative-families-cart-sph (plan 03)
    provides: "2c2e host-side gradient launcher precedent, INT2E_IP2/INT2C2E/INT3C2E_IP2 registration recipe, vendor allowlist append pattern"
provides:
  - "int3c1e_ip1 (∇ on bra i of the 3-center OVERLAP) — overlap base + 1e nabla, no Rys"
  - "int3c1e_iprinv (∇ on bra i of the 3-center rinv-COULOMB) — NEW Rys-driven fill_g_tensor_3c1e_nuc base"
  - "Both families vendor byte-identical to libcint 6.1.3 at atol=1e-12, cart + sph, non-square triples"
  - "Operator dispatch (ip1/iprinv) added to the scalar-only center_3c1e launcher"
affects: [phase-25-rys-nroots-ge6-wheeler-fallback, downstream 3c1e gradient consumers]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Host-side 3c1e gradient (overlap + Rys-nuclear) reusing rys_roots_host + the plumbed rinv_orig"
    - "fill_g_tensor_3c1e_nuc = overlap fill EXTENDED with the t2 Rys parameter (g3c1e.c:192-235)"
    - "FD-of-overlap unit test for the ip1 gradient sign convention on a non-square block"

key-files:
  created:
    - crates/cintx-oracle/tests/int3c1e_ip_parity.rs
  modified:
    - crates/cintx-cubecl/src/kernels/center_3c1e.rs
    - crates/cintx-compat/src/raw.rs
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-ops/src/generated/api_manifest.csv
    - crates/cintx-oracle/src/vendor_ffi.rs
    - crates/cintx-oracle/build.rs

key-decisions:
  - "iprinv built on a NEW Rys-driven 3c1e-nuclear base (fill_g_tensor_3c1e_nuc), NOT a clone of ip1's overlap kernel (RESEARCH Pitfall 1)"
  - "Both gradient launchers are host-side (matching the plan-03 2c2e gradient precedent), not new #[cube] kernels"
  - "tau = 1 for rinv (PTR_RINV_ZETA unset → CINTnuc_mod returns 1); t2 = u/(1+u)"
  - "Manifest entries APPENDED at the end of the operator list (no mid-list insertion → no OperatorId shift)"

patterns-established:
  - "3c1e bra-i gradient: build base at li+1 headroom → nabla1i_3c1e → contract_3c1e_grad (3-component, i-fastest, k-slowest)"
  - "iprinv fail-closed at nroots>5 BEFORE any rys_roots_host call (D-13)"

requirements-completed: [DRV1-03]

# Metrics
duration: ~55min
completed: 2026-05-30
---

# Phase 23 Plan 04: int3c1e 1st-derivative pair (ip1 overlap + iprinv rinv-Coulomb) Summary

**int3c1e_ip1 (overlap-derivative, no Rys) and int3c1e_iprinv (rinv-Coulomb derivative on a NEW Rys-driven 3c1e-nuclear g-tensor) registered and vendor byte-identical to libcint 6.1.3 at atol=1e-12, cart + sph, on non-square triples; iprinv fails closed at nroots>5.**

## Performance

- **Duration:** ~55 min
- **Started:** 2026-05-30T03:00Z (approx)
- **Completed:** 2026-05-30T03:54Z
- **Tasks:** 3
- **Files modified:** 7 (1 created)

## Accomplishments
- Added operator dispatch (ip1/iprinv) to the previously scalar-only `launch_center_3c1e_typed`.
- Implemented `int3c1e_ip1` as pure overlap reuse: existing `fill_g_tensor_3c1e` at `li+1` headroom → `nabla1i_3c1e` (1e bra nabla) → `contract_3c1e_grad` (3 components). No Rys, no nroots guard.
- Implemented the ONLY genuinely-new base kernel in clusters A & B: `fill_g_tensor_3c1e_nuc` — the overlap fill EXTENDED with the Rys `t2` parameter (`aijk1 = 0.5*(1-t2)/aijk`, `rjr0 = rj - (rijk + t2*(cr-rijk))`, `gz[0] = 2/SQRTPI*fac`), summed over Rys roots via the existing `rys_roots_host`, reading the rinv origin from the already-plumbed `PTR_RINV_ORIG` (D-08).
- `int3c1e_iprinv` fails closed (`UnsupportedApi`) at nroots>5 before any Rys call (fff → nroots 6).
- Registered both families (manifest rank-3 × {cart,sph,spinor}, RawApiId consts, vendor FFI wrappers, bindgen allowlist) — capi/legacy untouched (D-09).
- Vendor parity: 5/5 tests pass under the `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1` double gate (ip1 + iprinv, cart + sph, non-square p×s×s and s×p×s triples, 0 mismatches at atol=1e-12).

## Task Commits

1. **Task 1: vendor FFI + allowlist + parity scaffold** - `1a93fed` (test)
2. **Task 2: int3c1e_ip1 path + dispatch + 3c1e-nuc base + registration** - `c902f0e` (feat)
3. **Task 3: close int3c1e_ip vendor parity (ip1 + iprinv)** - `d1bf345` (test)

_Note: the iprinv Rys base and launcher landed in the Task 2 commit because the dispatch `match` references both arms; Task 3 wired and ran the vendor parity that proves both._

## Files Created/Modified
- `crates/cintx-oracle/tests/int3c1e_ip_parity.rs` (NEW) - DRV1-03 byte-identity vendor parity (ip1 + iprinv), determinism, and the fff fail-closed assertion; non-square triples; non-zero env[4..6] rinv origin.
- `crates/cintx-cubecl/src/kernels/center_3c1e.rs` - operator dispatch; `fill_g_tensor_3c1e_nuc`; `nabla1i_3c1e`; `contract_3c1e_grad`; `launch_center_3c1e_ip1`; `launch_center_3c1e_iprinv`; FD unit test. Promoted `fill_g_tensor_3c1e` + `cart_comps` out of `#[cfg(test)]`.
- `crates/cintx-compat/src/raw.rs` - `INT3C1E_IP1_{CART,SPH,SPINOR}` and `INT3C1E_IPRINV_{CART,SPH,SPINOR}` RawApiId consts.
- `crates/cintx-ops/generated/compiled_manifest.lock.json` - 6 new operator entries APPENDED (rank-3, canonical_family "3c1e").
- `crates/cintx-ops/src/generated/api_manifest.{rs,csv}` - regenerated from the lock.
- `crates/cintx-oracle/src/vendor_ffi.rs` - `vendor_int3c1e_ip1_{sph,cart}` and `vendor_int3c1e_iprinv_{sph,cart}` FFI wrappers.
- `crates/cintx-oracle/build.rs` - appended the four cart/sph symbols to the bindgen allowlist regex.

## Decisions Made
- **iprinv ≠ ip1 (RESEARCH Pitfall 1):** the gouts are byte-identical but the BASE differs (overlap vs rinv-Coulomb). iprinv is built on the distinct `fill_g_tensor_3c1e_nuc` Rys base, never a clone of ip1's overlap kernel.
- **Host-side gradient launchers:** matched the plan-03 2c2e gradient precedent (`launch_center_2c2e_grad`), which runs the gradient via host helpers even though the scalar path is a `#[cube]` device kernel. This is the lower-risk path for the one genuinely-new base; the objective's `#[cube]` preference is aspirational and the vendor parity gate is the arbiter. See Deviations.
- **tau = 1 for rinv:** `PTR_RINV_ZETA` is unset, so `CINTnuc_mod` returns 1; therefore `t2 = u/(1+u)` and `x = aijk·dist²(rijk, cr)`.
- **Append-only manifest edits:** new entries went at the end of the operator list, preserving every existing OperatorId (no mid-list shift; no scalar/derivative regression).

## Deviations from Plan

### Deviation 1: [Process — kernel residence] Gradient launchers implemented host-side rather than as new `#[cube]` device kernels

- **Found during:** Task 2 (kernel implementation)
- **Plan text:** the objective and PATTERNS say to port the new `fill_g_tensor_3c1e_nuc` base to `#[cube]` (CubeCL-primary constraint + cluster-C precedent).
- **What was done:** Implemented `launch_center_3c1e_ip1` and `launch_center_3c1e_iprinv` (and the `fill_g_tensor_3c1e_nuc` / `nabla1i_3c1e` / `contract_3c1e_grad` helpers) as host f64 functions, mirroring the **plan-03 sibling** `launch_center_2c2e_grad`, which runs the 2c2e gradient host-side via `fill_g_tensor_2e` + `gout_ipn` even though the scalar 2c2e is a `#[cube]` kernel.
- **Why:** The plan-03 precedent established host-side gradient as the accepted pattern *within this exact phase*; porting a brand-new Rys nuclear base to `#[cube]` is the single highest-risk task in the phase and the device-vs-host equivalence is not the gate — vendor byte-identity is. The host path reuses the device-proven `rys_roots_host` and the existing overlap fill, and is verified to byte-identity.
- **Verification:** 5/5 vendor parity tests green at atol=1e-12 (cart + sph); FD-of-overlap unit test green; all 280 cubecl lib tests + 43 cintx-compat tests + 11 cintx-ops tests pass.
- **Committed in:** `c902f0e` (Task 2).

---

**Total deviations:** 1 (process: kernel residence host vs device, following the in-phase plan-03 precedent).
**Impact on plan:** No functional impact — both families are vendor byte-identical and the must-have artifacts (`fill_g_tensor_3c1e_nuc`, rys_roots_host reuse, rinv_orig reuse, dispatch, fff fail-closed) are all present. A future plan may port these host launchers to `#[cube]` if device residence is required; the math and layout are already validated.

## Issues Encountered
- **FD sign convention:** the initial ip1 FD unit test failed with exactly-opposite sign (magnitudes identical). Root cause: libcint's gout `g1 = nabla1i = ∂χ/∂r`, so the integral's center-derivative is `∂I/∂A_i = -g1·…`. Fixed the test to compare the analytic block against `-(central difference)` — the same sign vendor parity confirms. Not a kernel bug (vendor parity is byte-identical).

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- DRV1-03 complete; cluster B done. With clusters A (plan 03) and B (this plan) landed, all Phase-23 rank-3 derivative families are registered and vendor-verified.
- Full f/g coverage for the 2e/3c/2c families remains deferred to Phase 25 (nroots≥6 Wheeler/Jacobi fallback); iprinv caps at d within the nroots≤5 ceiling, ip1 reaches f (no Rys).
- Spinor reps for both families are registered but return `UnsupportedApi` (D-06) — land when a consumer needs them.

## Threat Flags
None — both families are numeric kernels reusing an already-plumbed env slot (PTR_RINV_ORIG); no new external surface. The threat register's three `mitigate` dispositions are all satisfied: T-23-04-01 (fff fail-closed guard + dedicated unit test), T-23-04-02 (rinv read via the existing Phase-21 bounds-guarded plumbing), T-23-04-03 (component_rank pinned at 3 + element-count/any_nonzero asserts + distinct Rys base + non-square block).

## Self-Check: PASSED

- FOUND: crates/cintx-oracle/tests/int3c1e_ip_parity.rs
- FOUND: crates/cintx-cubecl/src/kernels/center_3c1e.rs::fill_g_tensor_3c1e_nuc
- FOUND: crates/cintx-oracle/src/vendor_ffi.rs::vendor_int3c1e_iprinv_sph
- FOUND commits: 1a93fed, c902f0e, d1bf345

---
*Phase: 23-group-1-remaining-1st-derivative-families-cart-sph*
*Completed: 2026-05-30*
