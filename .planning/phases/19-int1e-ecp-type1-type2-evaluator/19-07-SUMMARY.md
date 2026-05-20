---
phase: 19-int1e-ecp-type1-type2-evaluator
plan: 07
subsystem: kernel
tags: [ecp, gradient, ipnuc, deriv1, byte-identity, pyscf-nr-ecp-deriv, component-rank-3, manifest-lock, oracle-covered]

requires:
  - phase: 19-int1e-ecp-type1-type2-evaluator
    provides: "Plan 06 (scalar ecp_type1_cart / ecp_type2_cart drivers + angular splice — the gradient re-evaluates these at l±1); Plan 05 (K-Taylor radial host ports the scalar drivers stand on — nr_ecp_deriv.c introduces no new radial machinery); Plan 04 (vendor_ECPscalar_ipnuc_{cart,sph} FFI stubs + bindgen allowlist; the launch_ecp UnsupportedApi gradient stub this plan replaces); Plan 01 (Cu/LANL2DZ fixture, the two ipnuc manifest rows, lock-canonical regeneration)"
  - phase: 17-real-integral-evaluation-in-safe-api
    provides: "SessionRequest::evaluate dispatch; component_rank=3 staging layout (component_axis_leading=true); atol=1e-12, rtol=0.0 tolerance"
  - phase: 15-oracle-tolerance-unification-manifest-lock-closure
    provides: "manifest-audit oracle-coverage gate (--check-lock); lock JSON canonical + build.rs regenerates CSV/rs"
provides:
  - "Gradient ECP kernel in kernels/ecp.rs: deriv1_cart_pair (port of nr_ecp_deriv.c::_deriv1_cart) + compute_type1_pair_grad / compute_type2_pair_grad + l_down / l_up + raise_idx + ecp_scalar_prim_pair_cart, reaching atol=1e-12 byte-identity vs vendored PySCF nr_ecp_deriv over the full Cu/LANL2DZ Cartesian product × 3 components"
  - "launch_ecp operator_name == \"ecp_ipnuc\" arm: 3-component ∂/∂A_i^{x,y,z} gradient, F-order [axis, ao_j, ao_i] (axis slowest), per-axis cart_to_sph_1e for sph, with a typed buffer-size invariant for component_rank=3"
  - "Two un-ignored, by-default-passing gradient parity tests (test_int1e_ecp_ipnuc_{cart,sph}_safe_api_parity) + collect_safe_api_ecp_ipnuc_matrix / collect_ecp_ipnuc_matrix_vendor collectors"
  - "vendor_ECPscalar_ipnuc_{cart,sph} hardened with buffer-length debug_assert + nr_ecp_deriv.c line-cited rustdoc (T-19-23 mitigation)"
  - "int1e_ecp_ipnuc_{cart,sph} flipped to oracle_covered=true in compiled_manifest.lock.json (canonical) with api_manifest.{csv,rs} regenerated; manifest-audit zero diff; --check-lock uncovered_count=0 (all four ECP rows + the milestone oracle-coverage gate fully closed for ECP)"
affects:
  - "Phase 19 Plan 08 (optional libecpint secondary cross-check) — the gradient is now closed against PySCF nr_ecp_deriv; 19-08 is the only remaining (optional) Phase 19 work"
  - "pyscf_rs consumer (issue #11) — int1e_ecp_ipnuc_{cart,sph} gradients now byte-identical through the SessionRequest safe-API path"

tech-stack:
  added: []
  patterns:
    - "Pattern: gradient-as-re-evaluation-at-l±1. nr_ecp_deriv.c's _deriv1_cart adds NO new radial/angular machinery — it computes ∂/∂A χ = 2 a_i χ_{l+1} − l χ_{l-1} by re-calling the SAME scalar ECPtype1_cart/ECPtype2_cart drivers on per-primitive uncontracted 'fake' shells at l+1 (_l_down) and l-1 (_l_up), dividing out expi·expj, then re-applying the AO contraction coefficients. The cintx port reuses 19-06's ecp_type1_cart/ecp_type2_cart verbatim via single-primitive Shells with shifted angular momentum; the l±1 common-factor (CINTcommon_fac_sp) shifts come for free inside the scalar drivers."
    - "Pattern: derive the libcint _y_addr/_z_addr raise-tables from cintx's own cart_comps ordering rather than transcribing libcint's GLOBAL addr layout. The C global _y_addr/_z_addr are indexed by libcint's whole-table cart offset; cintx's scalar drivers produce buffers in cintx's cart_comps order (= PySCF's i-down/j-down enumeration), so raise_idx(l,i,axis) computed from cart_comps is the correct, ordering-consistent analog. Verified: x-raise is the identity prefix in both orderings (matching _l_down's outx[...i]=buf1[...i])."
    - "Pattern: identical-index dual collector to dodge symmetry masking. The scalar ECP matrix is symmetric so a transposed read is harmless; the per-axis gradient is NOT symmetric, so both collect_safe_api_ecp_ipnuc_matrix and collect_ecp_ipnuc_matrix_vendor read each per-pair buffer with the SAME index pair[axis*ni*nj + jj*ni + ii] — making the parity comparison apples-to-apples and a transposed axis a genuine test failure (T-19-22 mitigation)."

key-files:
  created:
    - ".planning/phases/19-int1e-ecp-type1-type2-evaluator/19-07-SUMMARY.md"
  modified:
    - "crates/cintx-cubecl/src/kernels/ecp.rs (gradient driver deriv1_cart_pair + compute_type{1,2}_pair_grad + l_down/l_up/raise_idx/ecp_scalar_prim_pair_cart; launch_ecp ecp_ipnuc arm with component_rank=3 buffer-size invariant + per-axis cart_to_sph_1e; module rustdoc gradient section; 4 new module tests; latent usize-underflow fix in the scalar radial loop start update; +~570 lines)"
    - "crates/cintx-oracle/src/vendor_ffi.rs (buffer-length debug_assert + nr_ecp_deriv.c-cited rustdoc on vendor_ECPscalar_ipnuc_{cart,sph})"
    - "crates/cintx-oracle/tests/safe_api_ecp_parity.rs (two ipnuc parity tests + two component-rank-3 collectors; file rustdoc Status → scalar+gradient closed)"
    - "crates/cintx-ops/generated/compiled_manifest.lock.json (oracle_covered=true on int1e_ecp_ipnuc_{cart,sph})"
    - "crates/cintx-ops/src/generated/api_manifest.csv / api_manifest.rs (regenerated by build.rs from the lock)"

key-decisions:
  - "Implemented the gradient as a single deriv1_cart_pair driver that BOTH Type-1 and Type-2 route through, because nr_ecp_deriv.c's _deriv1_cart sums Type-1+Type-2 into the same per-fake-shell buffer (lines 256-259/265-268). compute_type1_pair_grad (lc=-1) and compute_type2_pair_grad (lc≥0) are thin, plan-named aliases over it. The Local-vs-Projected split is carried as the lc parameter into the inner ecp_scalar_prim_pair_cart, which dispatches to ecp_type1_cart or ecp_type2_cart."
  - "Built per-primitive single-primitive 'fake' Shells (coefficient == exponent, matching _uncontract_bas's PTR_COEFF=PTR_EXP) with the angular momentum forced to li±1, and divided by expi·expj per pair — reproducing PySCF's uncontraction exactly while reusing the 19-06 scalar drivers untouched. The l±1 CINTcommon_fac_sp normalization is applied automatically inside ecp_type{1,2}_cart (they read cint_common_fac_sp of the shifted l), so only the _l_down/_l_up cart→sph compensation factors (-2/√3, -2·0.4886, √3, 1/0.4886) are transcribed verbatim."
  - "Derived raise_idx (the y/z angular-raise mapping) from cintx's cart_comps ordering instead of transcribing libcint's global _y_addr/_z_addr tables (nr_ecp_deriv.c:56-75). The C tables index libcint's whole-table cart layout, which differs from cintx's per-shell cart_comps order; since the cintx scalar drivers emit buffers in cart_comps order, the cart_comps-derived raise index is the only consistent choice. Verified x-raise is the identity prefix in cintx ordering (matches _l_down outx[...i]=buf1[...i]) via a unit test."
  - "Component axis order needs NO transpose. Read nr_ecp_deriv.c:240-280 (gctrx/gctry/gctrz = three consecutive dij blocks, comp slowest) + ECPscalar_distribute (nr_ecp.c:6133-6152, copies [comp][i+j*di]) and confirmed PySCF's [comp, ao_j, ao_i] (ao_i fastest within each dij block, di=ni for single contraction) is IDENTICAL to cintx's required F-order [axis, ao_j, ao_i] (D-11). Both the kernel staging write and both test collectors index pair[axis*ni*nj + jj*ni + ii] directly."
  - "Fixed a latent usize underflow in the scalar radial convergence loop's `start = (start-1)/2` update (both ecp_type1_cart and ecp_type2_cart). C uses signed int where (0-1)/2 == 0 (harmless, loop exits); Rust usize panicked when a non-converging far-field pair drove start to 0 at the finest level. Replaced with start.saturating_sub(1)/2 — byte-identical to the C arithmetic. Surfaced by the new zero-overlap gradient sanity test; the Cu/LANL2DZ scalar parity converges before underflow so it was previously dormant. Rule 1 auto-fix."

requirements-completed: [ECP-01, ECP-02, ECP-04, ECP-05]
# ECP-04 (byte-identity across all four scalar+gradient symbols) is now FULLY
# closed (the scalar half closed in 19-06; the gradient half closes here).
# ECP-05 (gradients delivered in scope per D-10) closes here. ECP-01/ECP-02
# (Type-1/Type-2 kernels) are re-exercised by the gradient (which re-evaluates
# them at l±1) and remain closed. All five Phase 19 requirements (ECP-01..05)
# are now closed at exact atol=1e-12 byte-identity.

# Metrics
duration: ~40min
completed: 2026-05-20
---

# Phase 19 Plan 07: ECP Gradient (ipnuc) Byte-Identity Close Summary

**One-liner:** Ported PySCF nr_ecp_deriv.c's `_deriv1_cart` — the 3-component `∂/∂A_i^{x,y,z}` ECP gradient (`2 a_i χ_{l+1} − l χ_{l-1}` via `_l_down`/`_l_up`, reusing the 19-06 scalar `ecp_type1_cart`/`ecp_type2_cart` drivers at l±1) — into `launch_ecp`'s `ecp_ipnuc` arm, writing F-order `[axis, ao_j, ao_i]`, reaching `atol=1e-12, rtol=0.0` byte-identity vs vendored PySCF nr_ecp_deriv (worst-case `|diff| ≈ 5.7e-14` per axis) and flipping the last two ECP manifest rows to `oracle_covered=true`.

## Performance

- **Duration:** ~40 min
- **Completed:** 2026-05-20
- **Tasks:** 2 (gradient kernel + vendor FFI; parity-close + manifest-flip)
- **Files modified:** 5 (1 kernel, 1 vendor FFI, 1 test, lock + 2 regenerated manifest)

## Tasks Completed

### Task 1: Gradient branch in launch_ecp + ipnuc vendor FFI hardening — commit `ecd00ca`

- **`deriv1_cart_pair`** ports `_deriv1_cart` (nr_ecp_deriv.c:201-286): per-primitive-pair, uncontract to single-primitive fake shells (coeff == exponent), re-evaluate the scalar (Type-1 + Type-2) drivers at `li+1` → `l_down`, and at `li-1` → `l_up` (only if `li>0`), divide out `expi·expj`, re-apply AO contraction coefficients into the three F-order axis blocks `[axis, ao_j, ao_i]`.
- **`l_down`** (nr_ecp_deriv.c:148-172) writes `2·a_i·χ_{li+1}` with the cart→sph compensation (`-2/√3` for li=0, `-2·0.4886` for li=1, `-2` otherwise); **`l_up`** (nr_ecp_deriv.c:174-199) adds the `(pow+1)`-weighted `χ_{li-1}` terms. Both re-index the scalar buffers via **`raise_idx`** (derived from cintx `cart_comps`, the consistent analog of libcint's global `_y_addr`/`_z_addr`).
- **`compute_type1_pair_grad`** (lc=-1) / **`compute_type2_pair_grad`** (lc≥0) are thin aliases over `deriv1_cart_pair`; the `launch_ecp` `ecp_ipnuc` arm dispatches Local→type1-grad / Projected(l)→type2-grad, sizes a `3·nci·ncj` cart buffer, applies `cart_to_sph_1e` per axis for sph, and rejects an undersized staging slice with a typed `ChunkPlanFailed` (component_rank=3 invariant, T-19-23).
- **4 new module tests** (total 9): positive `ecp_ipnuc`-resolves assertion (replacing the 19-04 reject-stub), `raise_idx` correctness, far-separated zero-overlap sanity, on-center finiteness.
- **vendor_ECPscalar_ipnuc_{cart,sph}** gained a buffer-length `debug_assert!` + nr_ecp_deriv.c line-cited rustdoc (the wrappers themselves landed in 19-04).
- **Rule 1 fix:** latent `usize` underflow in the scalar radial loop `start` update → `saturating_sub` (byte-identical to C `int` arithmetic).

### Task 2: Two ipnuc parity tests at atol=1e-12 + flip oracle_covered — commit `f7e78aa`

- **`test_int1e_ecp_ipnuc_{cart,sph}_safe_api_parity`** (no `#[ignore]`): zero mismatches over the full Cu/LANL2DZ Cartesian product × 3 components at `atol=1e-12, rtol=0.0`.
- **`collect_safe_api_ecp_ipnuc_matrix`** / **`collect_ecp_ipnuc_matrix_vendor`**: both assemble a `3·n_ao·n_ao` axis-outer matrix, reading each per-pair buffer with the IDENTICAL index `pair[axis*ni*nj + jj*ni + ii]` — apples-to-apples, no transpose (T-19-22).
- Flipped `oracle_covered=true` on `int1e_ecp_ipnuc_{cart,sph}` in the canonical lock JSON; `build.rs` regenerated `api_manifest.{csv,rs}`; `manifest-audit` zero diff; `--check-lock` `uncovered_count=0`.

## Output items requested by the plan

- **Actual PySCF nr_ecp_deriv C symbol names + line ranges used:** `ECPscalar_ipnuc_cart` (nr_ecp_deriv.c:366-375, `_cart_factory(_deriv1_cart, comp=3)`), `ECPscalar_ipnuc_sph` (nr_ecp_deriv.c:453-462, `_sph_factory(_deriv1_cart, comp=3)`), `_deriv1_cart` (201-286), `_l_down` (148-172), `_l_up` (174-199). The sph factory applies the per-component cart→sph transform via `ECPscalar_c2s_factory` (nr_ecp.c:6058-6107) + `ECPscalar_distribute` (nr_ecp.c:6133-6152).
- **Observed worst-case `|diff|` per axis:** `5.684e-14` for axis x/y/z, both cart and sph (measured via a scratch diagnostic, then removed). Three orders of magnitude inside `atol=1e-12`.
- **Did collect_ecp_ipnuc_matrix_vendor need a transpose for axis ordering?** **No.** PySCF writes `[comp, dij]` with `comp` slowest (gctrx|gctry|gctrz, lines 240-245) and `dij` F-order `n + j*di + i` (ao_i fastest, line 278-280); for single contraction (`di = ni`) this is exactly cintx's required `[axis, ao_j, ao_i]`. PySCF's comp == cintx's axis (same {x,y,z} order). No transpose in the kernel or either collector — confirmed by reading `_deriv1_cart` + `ECPscalar_distribute`, not assumed.
- **Algorithmic corrections required during execution:** **One** — the latent `usize` underflow in the scalar radial loop `start` update (Rule 1, Task 1). The gradient algorithm itself passed byte-identity on the FIRST run after Task 1: no sign-convention fix on the `(r − A_C)`/`2 a_i` factor, no normalization fix on the `±1` angular-momentum-shifted terms, no axis transpose. The careful pre-implementation read of `_deriv1_cart`/`_l_down`/`_l_up` and the cart_comps-derived `raise_idx` (vs the libcint global addr tables) landed correct. The `_l_down`/`_l_up` cart→sph compensation factors were transcribed verbatim; the l±1 `CINTcommon_fac_sp` shifts apply automatically inside the reused scalar drivers.

## Decisions Made

See frontmatter `key-decisions`. Summary: unified `deriv1_cart_pair` driver (Type-1+Type-2 both route through it, mirroring `_deriv1_cart`); per-primitive uncontracted fake Shells reusing the 19-06 scalar drivers at l±1; `raise_idx` derived from cintx `cart_comps` (not libcint's global addr tables); no axis transpose (PySCF == cintx layout); `saturating_sub` underflow fix.

## Deviations from Plan

### Rule 1 — Auto-fix: latent usize underflow in the scalar radial loop start update

- **Found during:** Task 1 (the new zero-overlap gradient sanity test panicked at `start = (start-1)/2`).
- **Issue:** The scalar `ecp_type1_cart`/`ecp_type2_cart` convergence loop updates `start = (start-1)/2` at every level including `LEVEL_MAX`, where `start == 0`. In C (`int`) this is `(0-1)/2 == 0` (harmless, the `while` exits); in Rust `usize` it panics. Dormant for the Cu/LANL2DZ scalar sweep (which converges before the finest level), but a far-field gradient pair drives the loop to non-convergence at `LEVEL_MAX`.
- **Fix:** `start = start.saturating_sub(1) / 2` in both scalar drivers — byte-identical to the C integer arithmetic.
- **Files modified:** `crates/cintx-cubecl/src/kernels/ecp.rs`.
- **Commit:** `ecd00ca`.

### Note — vendor ipnuc wrappers already existed (19-04)

Task 1 Step 4 says "extend vendor_ffi.rs with vendor_ECPscalar_ipnuc_{cart,sph}". Those wrappers already landed in 19-04 Task 2 (so the bindgen allowlist + link surface were in place). This plan instead HARDENED them with the buffer-length `debug_assert!` and the nr_ecp_deriv.c-cited rustdoc the plan's threat model (T-19-23) prescribes. No functional change to the FFI signature.

### Scope — host-only kernel (D-16, plan-sanctioned)

The gradient driver is pure host Rust with no `#[cube]` body, the same D-16 host-first decision documented for the 19-05/19-06 scalar ports (CubeCL "primary backend" deviation tracked in 19-CONTEXT.md "Deferred Ideas"). The byte-identity gate runs CPU-vs-C on `--features cpu`.

## Issues Encountered

The chief risk — the component-axis ordering — was resolved by reading `_deriv1_cart` + `ECPscalar_distribute` (no transpose needed). The secondary risk — the `_y_addr`/`_z_addr` raise tables — was resolved by deriving `raise_idx` from cintx's `cart_comps` (the ordering the cintx scalar drivers emit) rather than transcribing libcint's global addr layout. Both confirmed by the first-run byte-identity pass.

## Known Stubs

None. `deriv1_cart_pair` performs real arithmetic ported verbatim from nr_ecp_deriv.c and is gated by the two byte-identity parity tests. The `Representation::Spinor` zero-write in `launch_ecp` is the documented D-12 "compiled-but-unverified" path (spinor is out of the Phase 19 parity sweep), not a stub.

## Threat Flags

None. The only new extern "C" surface (`ECPscalar_ipnuc_*`) was already on the bindgen allowlist (19-04); no new network/auth/file/schema surface introduced.

## Next Phase Readiness

- ECP-01, ECP-02, ECP-03, ECP-04, ECP-05 are all closed at exact `atol=1e-12`. The milestone oracle-coverage gate (`manifest-audit --check-lock`) reports `uncovered_count=0` for the ECP family — all four rows covered.
- Phase 19 is feature-complete pending only the OPTIONAL 19-08 libecpint secondary cross-check.

## Self-Check: PASSED

Files verified to exist on disk:
- `crates/cintx-cubecl/src/kernels/ecp.rs` — FOUND (deriv1_cart_pair, compute_type{1,2}_pair_grad, l_down/l_up/raise_idx, ecp_ipnuc arm)
- `crates/cintx-oracle/src/vendor_ffi.rs` — FOUND (debug_assert + cited rustdoc on both ipnuc wrappers)
- `crates/cintx-oracle/tests/safe_api_ecp_parity.rs` — FOUND (two ipnuc tests, two collectors, no #[ignore])
- `crates/cintx-ops/generated/compiled_manifest.lock.json` — FOUND (ipnuc oracle_covered=true)
- `crates/cintx-ops/src/generated/api_manifest.csv` / `.rs` — FOUND (all four ECP rows true)
- `.planning/phases/19-int1e-ecp-type1-type2-evaluator/19-07-SUMMARY.md` — FOUND

Commits verified to exist:
- `ecd00ca` (Task 1 — feat) — FOUND
- `f7e78aa` (Task 2 — test) — FOUND

Substantive gates verified:
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --locked -p cintx-oracle --features cpu --test safe_api_ecp_parity` — 5 passed (2 scalar + 2 gradient + coverage invariant), 0 failed, 0 ignored
- `cargo test --locked -p cintx-cubecl --lib kernels::ecp` — 9 passed
- `cargo build --locked --workspace` — exit 0
- `cargo run --manifest-path xtask/Cargo.toml --locked -- manifest-audit` — exit 0, zero diff
- `cargo run --manifest-path xtask/Cargo.toml --locked -- manifest-audit --check-lock` — exit 0, uncovered_count=0
- CSV: int1e_ecp_{cart,sph,ipnuc_cart,ipnuc_sph} all oracle_covered=true

---
*Phase: 19-int1e-ecp-type1-type2-evaluator*
*Completed: 2026-05-20*
