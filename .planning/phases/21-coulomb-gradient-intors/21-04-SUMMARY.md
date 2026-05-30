---
phase: 21-coulomb-gradient-intors
plan: "04"
subsystem: kernels/one_electron + oracle/vendor_ffi + oracle/tests
tags: [gradient, ipnuc, iprinv, nuclear-attraction, rinv-origin, hellmann-feynman, oracle-parity, GRAD-05, GRAD-06]
dependency_graph:
  requires: ["21-01", "21-02"]
  provides:
    - int1e_ipnuc kernel (cart+sph, atom-loop, -Z_C)
    - int1e_iprinv kernel (cart+sph, single rinv origin, factor 1.0)
    - contract_nuclear_grad shared helper
    - vendor_int1e_ipnuc_{sph,cart} FFI wrappers
    - vendor_int1e_iprinv_{sph,cart} FFI wrappers
    - one_electron_nuc_grad_parity tests
  affects: [cintx-cubecl, cintx-oracle]
tech_stack:
  added: []
  patterns:
    - bra-nabla (∂/∂Ai) on the root-dependent nuclear Rys G-tensor with +1 bra headroom
    - (origin, charge_factor) parametrization shared by ipnuc (atom-loop) and iprinv (single-origin)
    - PTR_ENV_START-aligned oracle fixture so the rinv slot env[4..6] never collides with atom coords
key_files:
  created:
    - crates/cintx-oracle/tests/one_electron_nuc_grad_parity.rs
  modified:
    - crates/cintx-cubecl/src/kernels/one_electron.rs
    - crates/cintx-oracle/src/vendor_ffi.rs
    - crates/cintx-oracle/build.rs
decisions:
  - "contract_nuclear_grad reuses the contract_nuclear per-root nuclear G-tensor math (root-dependent vrr_2e_step_host, Phase 09) but with nmax = li+lj+1 (+1 bra headroom for the nabla ix+1 access) and applies the CINTnabla1i_1e bra-derivative inline (same mixing rule as 21-03's contract_grad_1e_bra)"
  - "ipnuc/iprinv differ ONLY in the (origin, charge_factor) list: ipnuc = [(atoms[c].coord_bohr, -Z_C) for all nuclei, low->high order (D-10)]; iprinv = [(rinv_orig, 1.0)] (single origin, no -Z_C, D-08)"
  - "ipnuc charge sign confirmed against vendor: fac1 = 2*PI * (-Z_C) * fac / aij matches libcint g1e.c:233 (CINTg1e_nuc point-nucleus branch, tau=1.0 from CINTnuc_mod)"
  - "iprinv prefactor is +1.0 (no -Z_C) confirmed against vendor: libcint g1e.c:227 (nuc_id<0 branch) uses fac1 = 2*PI * fac * tau / aij with cr = env + PTR_RINV_ORIG"
  - "iprinv None origin is rejected at the kernel boundary with InvalidEnvParam { param: PTR_RINV_ORIG } (defensive ok_or, never .unwrap()-panic) — T-21-04-01; the 21-01 validator is the primary gate"
  - "Oracle fixture is PTR_ENV_START-aligned (env[0..20] reserved). The shared one_electron_grad_parity.rs fixture packs coords at env[0..9], which collides with the rinv slot env[4..6] and is NOT reusable for iprinv (Rule 3 blocking adjustment)"
  - "flip oracle_covered to true for int1e_ipnuc and int1e_iprinv (all reps) in the manifest lock once the 21-08 manifest-audit pass runs — currently false"
metrics:
  duration: "~30 min"
  completed: "2026-05-26"
  tasks: 2
  files_modified: 4
---

# Phase 21 Plan 04: int1e_ipnuc + int1e_iprinv Nuclear-Gradient Kernels SUMMARY

**One-liner:** int1e_ipnuc (∑_C (-Z_C)·∇ over all nuclei) and int1e_iprinv (single rinv origin, factor +1.0) 3-component bra-nabla kernels sharing one `contract_nuclear_grad` helper; vendor byte-identity at atol=1e-12 (worst |diff| 7.99e-15 / 1.06e-15) confirmed vs libcint 6.1.3 for all 4 variants.

## What Was Built

### Task 1: ipnuc + iprinv gradient branches in one_electron.rs

**`contract_nuclear_grad`** — the shared nuclear-gradient helper:
- Builds the per-Rys-root nuclear G-tensor exactly as `contract_nuclear` (root-dependent `vrr_2e_step_host` c00/b10, Phase 09 decision) but with `nmax = li + lj + 1` (one extra bra level so the `ix+1` nabla access is valid)
- Applies the `CINTnabla1i_1e` bra-derivative inline (`f[jx*dj+ix] = ix·g[ix-1] - 2·ai·g[ix+1]`), the same axis-mixing rule as 21-03's `contract_grad_1e_bra`
- Parametrized over a `&[([f64; 3], f64)]` list of `(origin, charge_factor)` pairs:
  - **ipnuc**: `[(atoms[c].coord_bohr, -(Z_C as f64)) for all nuclei]`, ordered low→high atom index (D-10 bit-stable reduction)
  - **iprinv**: `[(rinv_orig, 1.0)]` (single entry, no `-Z_C`, D-08)
- Base nuclear prefactor `fac1 = 2*PI * charge_factor * fac / aij` matches libcint `g1e.c` exactly (lines 227/233)
- Returns `Vec<f64>` of length `3 * nci * ncj` in component-leading layout

**Dispatcher extension** — added `is_ipnuc`/`is_iprinv`, widened the rejection guard, widened the spinor guard (→ `UnsupportedApi`, R5/D-03), and added the nuclear-gradient branch to the existing 3-component gradient path. For iprinv, the origin is resolved up front with a defensive `ok_or(InvalidEnvParam { param: "PTR_RINV_ORIG", .. })` (never panics, never reads a garbage origin — T-21-04-01).

**Unit tests added (6):**
- `test_ipnuc_component_count` — s-s → 3, p-s → 9
- `test_ipnuc_determinism` — bit-identical across two calls (3-nucleus origin list, ordered reduction)
- `test_iprinv_origin_sensitivity` — **the key proof**: two DIFFERENT origins produce DIFFERENT output (origin consumed, not ignored)
- `test_iprinv_none_origin_returns_typed_error` — iprinv dispatch with `rinv_orig == None` returns `InvalidEnvParam`, never panics
- `test_ipnuc_spinor_returns_unsupported`, `test_iprinv_spinor_returns_unsupported` — `UnsupportedApi` for spinor (the spinor guard fires before the origin is consumed)

### Task 2: Vendor FFI wrappers + oracle parity tests

**`vendor_ffi.rs`** — 4 new wrappers (`vendor_int1e_ipnuc_{sph,cart}`, `vendor_int1e_iprinv_{sph,cart}`) mirroring the ipovlp template. The iprinv wrapper doc comments explicitly state the caller MUST set `env[PTR_RINV_ORIG..+3]` before calling.

**`build.rs`** — added `int1e_ipnuc_sph|int1e_ipnuc_cart|int1e_iprinv_sph|int1e_iprinv_cart` to the bindgen allowlist. `grad1.c` (which defines these symbols) was already in the cc::Build from 21-03; no supplemental header edit needed (the symbols are in `cint_funcs.h`).

**`one_electron_nuc_grad_parity.rs`** — 6 tests (2 always-run + 4 vendor parity):
- ipnuc determinism + nonzero sentinel (always run)
- iprinv determinism + origin-sensitivity (always run)
- ipnuc cart/sph byte-identity vs vendor at atol=1e-12
- iprinv cart/sph byte-identity vs vendor at atol=1e-12, **sweeping the rinv origin over each nucleus** (O, H1, H2) to prove single-origin parity per atom

## Charge-Factor Sign + Prefactor Confirmation (vs libcint g1e.c)

| Operator | libcint branch | Prefactor | cintx implementation |
|----------|----------------|-----------|----------------------|
| ipnuc    | g1e.c:233 (point nucleus, nuc_id≥0) | `2*PI * -abs(Z_C) * fac * tau / aij` (tau=1) | `charge_factor = -(atom.atomic_number as f64)` per nucleus, summed |
| iprinv   | g1e.c:227 (nuc_id<0, cr = env + PTR_RINV_ORIG) | `2*PI * fac * tau / aij` (no charge, tau=1) | `charge_factor = 1.0`, single origin from `rinv_orig` |

`CINTnuc_mod` (g1e.c:190-206) returns `tau=1.0` for point nuclei (zeta=0), so the tau factor folds to 1.0 in both cases — matching the existing `contract_nuclear` prefactor.

## Oracle Parity Results (H2O STO-3G, atol=1e-12, vendored libcint 6.1.3)

| Operator | Representation | Result | Worst \|diff\| |
|----------|----------------|--------|----------------|
| int1e_ipnuc  | sph  | 0 mismatches | 7.994e-15 |
| int1e_ipnuc  | cart | 0 mismatches | 7.994e-15 |
| int1e_iprinv | sph  | 0 mismatches (per-nucleus sweep) | 1.055e-15 |
| int1e_iprinv | cart | 0 mismatches (per-nucleus sweep) | 1.055e-15 |

All worst-case diffs are ~3 orders of magnitude below atol=1e-12. The iprinv per-nucleus sweep (O, H1, H2) proves the single-origin selection matches libcint for each atom independently.

## nmax Headroom Record

| Operator | nmax Formula | Reason |
|----------|-------------|--------|
| ipnuc/iprinv | li + lj + 1 | nuclear VRR + bra nabla accesses g[ix+1] → 1 extra bra level (nrys_roots = (li+lj+1)/2 + 1) |

## Note for Manifest Audit (21-08)

The `int1e_ipnuc` and `int1e_iprinv` manifest entries (all cart/sph/spinor reps) currently carry `"oracle_covered": false` with `"component_rank": "3"` (registered by 21-02). Now that the oracle parity tests are green, the cart/sph entries should be flipped to `"oracle_covered": true` in the 21-08 manifest-audit pass. (Spinor reps return `UnsupportedApi` per D-03 and stay uncovered.)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Oracle fixture env layout incompatible with iprinv**
- **Found during:** Task 2 (writing the iprinv parity test)
- **Issue:** The shared `one_electron_grad_parity.rs` `build_h2o_sto3g` fixture packs atom coordinates at `env[0..9]`, which directly collides with `PTR_RINV_ORIG = 4..6` (and `PTR_RINV_ZETA = 7`). Setting the rinv origin at `env[4..6]` for the iprinv test would clobber the H1/H2 nucleus coordinates, producing a garbage integral for both cintx and vendor.
- **Fix:** Built a dedicated `build_h2o_sto3g_envstart()` fixture in the new test file that reserves `env[0..PTR_ENV_START=20]` for libcint global slots and places all user data (coords, exponents, coefficients) at `env[20..]`. This is the libcint-documented env layout and keeps the rinv slot disjoint from atom data. Matches the Phase 10-02 decision ("env user data MUST start at PTR_ENV_START=20").
- **Files modified:** crates/cintx-oracle/tests/one_electron_nuc_grad_parity.rs (new file — fixture is self-contained)
- **Commit:** 898477e

### Worktree Path Note (process, not a code deviation)

The first Edit/Bash pass accidentally targeted the MAIN repo (`/home/user/Documents/workspace/cintx/...`) instead of the worktree, because absolute paths were derived from the orchestrator's cwd rather than the worktree root (#3099). Detected before committing (worktree `git status` was empty). Recovery: saved the one_electron.rs delta as a patch, reverted the unintended main-repo edit (`git checkout -- crates/cintx-cubecl/src/kernels/one_electron.rs`), and re-applied the patch inside the worktree. All commits live on the per-agent branch; no main-repo state was committed. Subsequent edits used worktree-rooted paths.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes. The changes are:
1. In-process Rust kernel math (nuclear G-tensor + bra-nabla)
2. Vendor FFI wrappers (unsafe but guarded by `#[cfg(has_vendor_libcint)]`)
3. Test-only file addition

Threat register coverage:
- **T-21-04-01** (iprinv reached with None origin): mitigated — defensive `ok_or(InvalidEnvParam)` at the kernel boundary; unit test `test_iprinv_none_origin_returns_typed_error` confirms no panic.
- **T-21-04-02** (ipnuc missing an atom / wrong charge sign): mitigated — oracle byte-identity at atol=1e-12 vs `vendor_int1e_ipnuc` (which sums all nuclei with `-Z_C`) catches a missing atom or sign flip; the `assert_any_nonzero` sentinel catches zero-fill.
- **T-21-04-03** (iprinv wrong/ignored origin): mitigated — `test_iprinv_origin_sensitivity` (different origin → different result) + the per-nucleus oracle sweep proving each origin matches libcint exactly.
- **T-21-04-04** (non-deterministic atom-loop reduction): mitigated — ordered low→high atom-index reduction; `test_ipnuc_determinism` enforces bit-stability.
- **T-21-04-SC** (package installs): N/A — no new external packages.

## Verification Results

- `cargo test -p cintx-cubecl --features cpu --lib ipnuc` — 3 passed, 0 failed
- `cargo test -p cintx-cubecl --features cpu --lib iprinv` — 3 passed, 0 failed
- `cargo test -p cintx-oracle --features cpu --test one_electron_nuc_grad_parity` — 2 passed (determinism + origin-sensitivity), 0 failed
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test one_electron_nuc_grad_parity` — 6 passed (incl. 4 vendor byte-identity), 0 failed
- `cargo test -p cintx-oracle --features cpu --test one_electron_grad_parity` — 4 passed (no ipovlp/ipkin regression)
- `cargo test -p cintx-cubecl --features cpu --lib nuclear` — 1 passed (no scalar nuclear regression)
- `CINTX_BACKEND=cpu cargo check --workspace --features cpu` — 0 errors

## Self-Check: PASSED

Files confirmed present (worktree):
- crates/cintx-cubecl/src/kernels/one_electron.rs — `contract_nuclear_grad`, `is_ipnuc`/`is_iprinv` dispatcher branches, `rinv_orig` consumption present
- crates/cintx-oracle/src/vendor_ffi.rs — `vendor_int1e_ipnuc_sph` at line 530, `vendor_int1e_iprinv_cart` at line 629
- crates/cintx-oracle/build.rs — bindgen allowlist additions present
- crates/cintx-oracle/tests/one_electron_nuc_grad_parity.rs — 6 tests, 5 `#[cfg(has_vendor_libcint)]`, PTR_RINV_ORIG used

Commits confirmed:
- a7a8734: feat(21-04): implement int1e_ipnuc (atom-loop) + int1e_iprinv (single-origin) gradient branches
- 898477e: feat(21-04): add vendor FFI wrappers + oracle parity tests for ipnuc/iprinv (cart+sph)
