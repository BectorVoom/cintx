---
phase: 23-group-1-remaining-1st-derivative-families-cart-sph
plan: 03
subsystem: integrals
tags: [int3c2e_ip2, gradient, rys, cubecl, nabla1l, oracle-parity, libcint, pitfall-2]

# Dependency graph
requires:
  - phase: 23-group-1-remaining-1st-derivative-families-cart-sph (plan 01)
    provides: "pub(crate) nabla1l_2e (G2E_D_L) + Nabla1Center::L + gout_ipn parameterized single-side contraction in f12.rs"
  - phase: 23-group-1-remaining-1st-derivative-families-cart-sph (plan 02)
    provides: "int2e_ip2 / int2c2e_ip pattern + the vendor_ffi/build.rs/manifest append precedent on the shared registration files"
  - phase: 21-coulomb-gradient-intors
    provides: "int3c2e_ip1 device kernel (center_3c2e_ip1_kernel) + launcher + the 3c2e Pitfall-2 ll-slot mapping (build_2e_shape(li+1,lj,0,lk))"
provides:
  - "int3c2e_ip2 (DRV1-05): ∇ on the auxiliary k center of the 3-center-2-electron Coulomb integral, cart+sph, vendor byte-identity at atol=1e-12"
  - "center_3c2e_ip2_kernel (#[cube] device kernel) + run_3c2e_ip2_device + launch_center_3c2e_ip2 + the \"ip2\" operator dispatch branch in center_3c2e.rs"
  - "2 vendor FFI wrappers (vendor_int3c2e_ip2_sph/_cart) + bindgen allowlist + 3 manifest entries (component_rank 3) + 3 RawApiId consts"
  - "cluster A (rank-3 ket/remaining-center) COMPLETE"
affects: [23-cluster-B-int3c1e, future-3c2e-DF-gradient-consumers]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "int3c2e_ip2 = the ip1 device kernel with the headroom moved from the bra i-ladder (li+1) to the auxiliary-k ll-ladder (lk+1): build_2e_shape(li, lj, 0, lk+1), G2E_D_L nabla on the ll slot at exponent ak (Pitfall 2 — NOT nabla1k_2e, which touches the phantom 2e lk slot)"
    - "NEW-family manifest entries MUST be APPENDED at the END of the lock operator list, never inserted mid-list: OperatorId is the lock array position (build.rs: id = idx), so a mid-list insert silently shifts every later id and breaks positional lookup (Resolver::descriptor uses OPERATOR_DESCRIPTORS.get(id.raw()))"
    - "spd 3-center shared-coefficient byte-identity fixture (3 distinct atoms so ∇_k does not vanish by symmetry) is the required style for the 3c2e Rys gradient oracle gate"

key-files:
  created:
    - crates/cintx-oracle/tests/int3c2e_ip2_parity.rs
  modified:
    - crates/cintx-cubecl/src/kernels/center_3c2e.rs
    - crates/cintx-compat/src/raw.rs
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-ops/src/generated/api_manifest.csv
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-oracle/src/vendor_ffi.rs
    - crates/cintx-oracle/build.rs

key-decisions:
  - "Applied nabla1l_2e on the 2e ll slot (real aux k), NOT nabla1k_2e, per RESEARCH Pitfall 2: cintx maps the real aux k into the ll slot and leaves the 2e lk slot a phantom s-function. Confirmed against libcint int3c2e.c:99 (G2E_D_K on the real aux k) and guarded by test_ip2_not_equal_ip1."
  - "Cloned the ip1 device kernel into a dedicated center_3c2e_ip2_kernel rather than parameterizing the existing kernel — the headroom slot, the nabla recurrence, and the contraction base-l differ, and a clone keeps the proven ip1 path byte-identical (no regression risk)."
  - "Appended the 3 ip2 manifest entries at the END of the operator list (not after int3c2e_ip1) to preserve the positional OperatorId invariant (see Deviations Rule 1)."

patterns-established:
  - "Pitfall-2 wrong-slot guard: a device-level test_ip2_not_equal_ip1 on a NON-SQUARE p×d block asserts int3c2e_ip2 != int3c2e_ip1 — a wrong-slot nabla would make them equal."
  - "Auxiliary-center 3c2e gradient: raise the ll slot (real aux k) by +1 instead of the bra i; the rest of the VRR/HRR/Rys machinery is reused verbatim from ip1."

requirements-completed: [DRV1-05]

# Metrics
duration: ~80min
completed: 2026-05-30
---

# Phase 23 Plan 03: int3c2e_ip2 Auxiliary-Center Gradient Summary

**int3c2e_ip2 (the ∇ on the 3-center-2-electron auxiliary k center) registered and byte-identical to vendored libcint 6.1.3 at atol=1e-12 for cart+sph, applying the derivative on the 2e `ll` slot via `nabla1l_2e` (RESEARCH Pitfall 2) — completing cluster A.**

## What Was Built

This plan was the one non-mechanical cluster-A family. The 3c2e g-tensor in cintx maps the
real auxiliary `k` into the 2e `ll` slot (the 2e `lk` slot is a phantom s-function), so the
ip2 auxiliary derivative had to be taken on the `ll` slot via `nabla1l_2e` (added in plan 01),
not the naive `nabla1k_2e` (which would touch the phantom slot).

- **`center_3c2e_ip2_kernel`** (`#[cube(launch)]`, generic over `F: Float`): a faithful clone
  of `center_3c2e_ip1_kernel` with the headroom moved from the bra `i`-ladder (`li+1`) to the
  auxiliary-`k` `ll`-ladder (`lk+1`). It builds the plain-Coulomb g-tensor through the shared 2e
  VRR/HRR recurrence at `build_2e_shape(li, lj, 0, lk+1)`, applies the `G2E_D_L` recurrence on
  the `ll` slot at exponent `ak`, and contracts the standard 3-component i-fastest mixing
  (`s[0]=g1x·g0y·g0z`, …) per `int3c2e.c:99`. `run_3c2e_ip2_device` dispatches it across all
  backends (cpu/wgpu/cuda/rocm/metal).
- **`launch_center_3c2e_ip2`**: spinor reject (D-06), `nroots>5` fail-closed (D-13), the same
  cart/sph component-leading `[3,nk,nj,ni]` staging transpose as ip1.
- **`"ip2"` operator dispatch** added after the existing `"ip1"` branch in
  `launch_center_3c2e_typed`.
- Registration (D-11 recipe): `vendor_int3c2e_ip2_sph/_cart` FFI wrappers, the bindgen
  allowlist append (`int3c2e_ip2_sph|int3c2e_ip2_cart`; `int3c2e.c` already in the source list),
  3 manifest lock entries (`component_rank 3`, operator `ip2`, cart/sph `oracle_covered=true`,
  spinor `false`), and the `INT3C2E_IP2_{CART,SPH,SPINOR}` RawApiId consts.

## Verification

- **Vendor byte-identity (DRV1-05 gate):** `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle
  --features cpu --test int3c2e_ip2_parity` → `running 2 tests`, `test result: ok.`;
  `int3c2e_ip2_sph` and `int3c2e_ip2_cart` each PASS over **27 triples, 0 mismatches at
  atol=1e-12** vs `vendor_int3c2e_ip2_{sph,cart}`. The spd 3-center fixture exercises non-square
  i×j / i×aux blocks (p×d×s probe), `any_nonzero`, and element-count = `3*ni*nj*nk` (D-14).
- **Device-vs-host (`#[cube]` correctness):** 7 `ip2_device_tests` pass, including a NON-SQUARE
  `pds` block and `test_ip2_not_equal_ip1` (Pitfall 2 wrong-slot guard: ip2 ≠ ip1).
- **No regression:** the full `kernels::center_3c2e` suite is 29/29 green (the 2 scalar tests
  that briefly failed during a mid-list manifest insert are green after the fix — see
  Deviations).
- **manifest-audit:** `status: ok`, `oracle_coverage.uncovered_count: 0`, 0 missing in
  generated/lock, no profile-scope mismatch.
- **ID preservation:** `cintx-ops` 11/11 green, incl. `ecp_operator_ids_match_constants`
  (int4c1e_cart still id 24, ECP 26–29 unchanged).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Manifest entries must be APPENDED at the end of the operator list, not inserted after int3c2e_ip1**
- **Found during:** Task 2 (after running the full `center_3c2e` device suite).
- **Issue:** The plan/PATTERNS said to clone the `int3c2e_ip1` lock entries (lines 307–402) for
  ip2 — naturally read as "insert right after ip1". Doing so put the 3 new entries at lock array
  positions ~22–24. Because `cintx-ops/build.rs` assigns `OperatorId = lock array index`
  (`id: OperatorId::new(idx)` with `entry: &MANIFEST_ENTRIES[idx]`) and `Resolver::descriptor`
  looks up positionally (`OPERATOR_DESCRIPTORS.get(id.raw())`), the insert silently shifted
  `int3c2e_cart` (id 22 → 25), `int3c2e_sph`, `int4c1e_cart` (id 24), and every later id by +3.
  The scalar 3c2e tests (`OperatorId::new(22)`) then ran the ip2 path and produced `0`, failing
  `test_center_3c2e_parity_f64` and `test_center_3c2e_f32_smoke`.
- **Fix:** Reverted the mid-list insert and APPENDED the 3 ip2 entries at the END of the operator
  list (after `cint2e_ip1_optimizer`, id 189) so they take ids 190/191/192 while every existing
  id is preserved — matching plan 02's proven append position. Re-ran: scalar tests green, ip2
  parity green, `ecp_operator_ids_match_constants` green.
- **Files modified:** `crates/cintx-ops/generated/compiled_manifest.lock.json` (+ regenerated
  `api_manifest.{rs,csv}`).
- **Commit:** e351ac8

## Known Stubs

None — the kernel is fully wired and vendor-verified; the spinor representation is intentionally
registered-but-`UnsupportedApi` per D-06 (resolves when a consumer needs spinor gradients).

## Self-Check: PASSED

- FOUND: crates/cintx-oracle/tests/int3c2e_ip2_parity.rs
- FOUND: commit 0bc89d6 (test scaffold + vendor FFI + allowlist)
- FOUND: commit e351ac8 (kernel + launcher + dispatch + manifest + RawApiId)
- int3c2e_ip2 cart/sph vendor parity: 0 mismatches at atol=1e-12 over 27 triples each (N>0)
- manifest-audit: status ok, 0 uncovered

## Commits

- `0bc89d6` test(23-03): add int3c2e_ip2 vendor FFI + allowlist + parity scaffold (RED)
- `e351ac8` feat(23-03): implement int3c2e_ip2 (nabla1l_2e on the ll slot, DRV1-05)
