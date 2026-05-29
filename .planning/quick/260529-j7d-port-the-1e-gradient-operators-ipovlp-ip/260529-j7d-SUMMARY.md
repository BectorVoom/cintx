---
phase: quick-260529-j7d
plan: 01
subsystem: cintx-cubecl 1e gradient kernels
tags: [cubecl, device-kernel, gradient, ipovlp, ipkin, ipnuc, iprinv, rocm, oracle]
requires:
  - "one_electron_scalar_kernel device infra (quick-260529-imi)"
  - "center_3c2e_ip1 dual-path gradient template"
  - "obara_saika #[cube] vrr/hrr/vrr_2e helpers"
provides:
  - "1e gradient device kernels (ipovlp/ipkin bra-nabla; ipnuc/iprinv shared root-VRR+nabla)"
  - "run_1e_grad_bra_on_backend / run_1e_nuc_grad_on_backend 5-arm dispatch"
  - "randomized ROCm gradient idempotency oracle (4 operators, 48 cases)"
affects:
  - "launch_one_electron_typed gradient block (live dispatch)"
tech-stack:
  added: []
  patterns:
    - "shared origins-parameterized device kernel for ipnuc + iprinv"
    - "component-leading 3-component output preserved on-device"
    - "nroots>5 fail-closed guard before any Rys call"
key-files:
  created:
    - crates/cintx-oracle/tests/one_electron_grad_random_rocm_parity.rs
  modified:
    - crates/cintx-cubecl/src/kernels/one_electron.rs
decisions:
  - "Tasks 1 + 2 committed as ONE feat commit: the live dispatch rewire references BOTH runner functions, so the file cannot compile cleanly with only one kernel present — splitting would produce an intermediate non-building commit."
  - "fill_g_tensor_overlap + cart_comps moved under #[cfg(test)] (only test callers remain) plus their host-helper imports (vrr/hrr_step_host, rys_roots_host, compute_pdata_host) to keep the non-test build warning-free; the plan noted fill_g_tensor_overlap 'stays non-test' but no non-test caller survives the rewire."
  - "nuclear-gradient nroots = (li+lj+1)/2 + 1 (the +1 bra headroom raises the root ceiling by one vs the scalar nuclear path)."
metrics:
  duration: ~35 min
  completed: 2026-05-29
---

# Phase quick-260529-j7d Plan 01: Port 1e Gradient Operators to CubeCL Device Kernels Summary

On-device port of all four int1e gradient operators (ipovlp/ipkin bra-nabla and ipnuc/iprinv shared origins-parameterized nuclear-gradient) to `#[cube(launch)]` kernels generic over `F: Float`, closing the gap quick-260529-imi left after porting only the scalar operators; verified byte/atol-stable against the existing host parity suites and proven on real ROCm hardware (gfx1152) with mismatch_count=0.

## What Was Built

- **`one_electron_grad_bra_kernel` `#[cube(launch)]`** — ipovlp (op_kind=0) and ipkin (op_kind=1). Builds the overlap base G-tensor in-kernel (vrr+hrr), applies `nabla1i` into `g1`, and for ipkin computes `D_j^2` of both `g0` and `g1` (`d2g0`/`d2g1`). G-sizing per comptime op_kind: ipovlp nmax=li+lj+1/lj_ext=lj; ipkin nmax=li+lj+3/lj_ext=lj+2. Component mixing s0=g1x·g0y·g0z (ipovlp) / -0.5·(D²(g1x)·g0y·g0z + …) (ipkin), verbatim from the host references.
- **`one_electron_nuc_grad_kernel` `#[cube(launch)]`** — ipnuc and iprinv share ONE kernel parameterized by `origin_coords` (norig·3) + `origin_charges` (norig) + `norig`. Per-origin (ordered low→high for bit-stable reduction, D-10): Rys roots via comptime `rys_root1..5`, per-root root-VRR (`vrr_2e_step`, b10=rt) + HRR + `nabla1i` + 3-component accumulation. nmax=li+lj+1 (+1 bra headroom), lj_ext=lj.
- **Device runners + dispatch** — `run_1e_grad_bra_device`/`run_1e_grad_bra_on_backend` and `run_1e_nuc_grad_device`/`run_1e_nuc_grad_on_backend`, each a 5-arm Cpu/Wgpu/Cuda/Rocm/Metal dispatch cloned from the scalar path. Both kernels perform the primitive × contraction accumulation in-kernel and return the per-(ci,cj) 3-component component-leading buffer.
- **Live dispatch rewire** — `launch_one_electron_typed`'s gradient block now dispatches all four operators through the device backend. Spinor rejection (before compute) and the iprinv `None`-origin → `InvalidEnvParam` gate preserved. Added the nuclear-gradient `nroots>5` fail-closed `UnsupportedApi` guard BEFORE any device/Rys call (T-j7d-01). Post-device `common_fac_sp` scale + component-leading sph/cart staging unchanged.
- **Host references → `#[cfg(test)]`** — `contract_grad_1e_bra`, `contract_ipkin`, `contract_nuclear_grad`, plus `fill_g_tensor_overlap` and `cart_comps` (only test callers survive), kept as cross-check references.
- **ROCm gradient oracle** — `one_electron_grad_random_rocm_parity.rs`: drives int1e_{ipovlp,ipkin,ipnuc,iprinv}_sph TWICE per random H2O/STO-3G system (48 cases) via the raw API on the ROCm device; iprinv writes a fixed rinv origin into env[PTR_RINV_ORIG..+3]. Asserts mismatch_count=0 + any_nonzero.

## Verification

- **Device-vs-host cross-checks** (CpuRuntime, f64, atol=1e-12/rtol=1e-10): `test_device_matches_host_ipovlp`, `test_device_matches_host_ipkin` (both over (0,0),(0,1),(1,0),(1,1),(2,2)) and `test_device_matches_host_nuclear_grad` (ipnuc 2-origin over (0,0),(0,1),(1,0),(1,1) + iprinv single-origin over (0,0),(1,1)) — all PASS.
- **Existing host parity suites through the new device path**: `one_electron_grad_parity` (4 tests) and `one_electron_nuc_grad_parity` (2 tests) — all PASS.
- **Full one_electron lib suite**: 31 passed / 0 failed.
- **ROCm oracle (RAN on gfx1152 via `cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle`)**: `test_int1e_grad_random_rocm_idempotency ... ok`; the full base-profile suite reported `rocm-oracle suite passed for profile base`. The test asserts `mismatch_count == 0` and `any_nonzero == true` across 48 cases × 4 operators × 2 evaluations — **mismatch_count=0, any_nonzero=true**.
- **Build clean**: `cargo build -p cintx-cubecl --features cpu` → 0 warnings; `cargo build -p cintx-oracle --features rocm --tests` clean (pre-existing warnings in other test files only).
- Spinor gradient still rejected (`UnsupportedApi`) and nuclear nroots>5 fail-closed — guard tests `test_ip*_spinor_returns_unsupported` / `test_iprinv_none_origin_returns_typed_error` pass.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Tasks 1 & 2 landed in one feat commit**
- **Found during:** Task 1 (`<done>`: contract_grad_1e_bra/contract_ipkin must be `#[cfg(test)]` AND `cargo build` clean).
- **Issue:** Moving the bra host fns under `#[cfg(test)]` while the live launcher still called them (Task 2 owns the rewire) would break the non-test build. The live dispatch rewire (Task 2) references both `run_1e_grad_bra_on_backend` and `run_1e_nuc_grad_on_backend`, so the file only compiles once both kernels exist.
- **Fix:** Implemented both kernels + runners + the full rewire together, then committed as a single `feat` commit covering Tasks 1 and 2. Each task's cross-check test still exists independently and passes.
- **Files modified:** crates/cintx-cubecl/src/kernels/one_electron.rs
- **Commit:** 9f3b9b2

**2. [Rule 3 - Blocking] `fill_g_tensor_overlap`/`cart_comps` + host-helper imports gated `#[cfg(test)]`**
- **Found during:** Task 1 build.
- **Issue:** After the rewire, the only remaining callers of `fill_g_tensor_overlap`, `cart_comps`, `vrr_step_host`/`hrr_step_host`, `rys_roots_host`, and `compute_pdata_host` (top-level import) are in the test module → dead-code / unused-import warnings in a non-test build. The plan noted `fill_g_tensor_overlap` "stays non-test", but no non-test caller survives.
- **Fix:** Gated these under `#[cfg(test)]` to keep the non-test build warning-free.
- **Files modified:** crates/cintx-cubecl/src/kernels/one_electron.rs
- **Commit:** 9f3b9b2

## Known Stubs

None. All four gradient operators compute on-device on the live path; no placeholder/empty-data paths remain.

## Threat Flags

None. No new network/auth/file/schema surface; the three mitigated threats (T-j7d-01 nroots>5 guard, T-j7d-02 iprinv origin gate, T-j7d-03 spinor rejection) are all implemented and the accepted T-j7d-04 (empty-origins) reuses the scalar path's len-1 dummy-buffer convention.

## Commits

- 9f3b9b2 — feat(quick-260529-j7d): port 1e gradient operators (ipovlp/ipkin/ipnuc/iprinv) to cubecl device kernels
- a2260ba — test(quick-260529-j7d): randomized ROCm oracle for int1e gradient operators

## Self-Check: PASSED

- FOUND: crates/cintx-oracle/tests/one_electron_grad_random_rocm_parity.rs
- FOUND: crates/cintx-cubecl/src/kernels/one_electron.rs
- FOUND: .planning/quick/260529-j7d-.../260529-j7d-SUMMARY.md
- FOUND commit: 9f3b9b2
- FOUND commit: a2260ba
