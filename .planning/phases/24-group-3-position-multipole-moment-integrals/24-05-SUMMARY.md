---
phase: 24-group-3-position-multipole-moment-integrals
plan: 05
subsystem: kernels
tags: [cubecl, kernel, irp, overlap-derivative, gauge-origin, parity]

# Dependency graph
requires:
  - phase: 24-group-3-position-multipole-moment-integrals
    plan: 01
    provides: vendor_int1e_irp_{cart,sph} FFI wrappers + bindgen allowlist (int1e_irp_{sph,cart}) + RED moment_nontensor_parity irp scaffold (same-center non-square block)
  - phase: 24-group-3-position-multipole-moment-integrals
    plan: 02
    provides: Cluster A moment kernel (overlap VRR/HRR + per-axis moment ladder) + manifest/RawApiId registration recipe
  - phase: 24-group-3-position-multipole-moment-integrals
    plan: 03
    provides: INT1E_IRP_* RawApiId symbol consts (declared in raw.rs, no manifest/kernel until this plan)
  - phase: 24-group-3-position-multipole-moment-integrals
    plan: 04
    provides: p4 overlap-derivative rank-1 kernel + d_i_1e_into/d_j_1e_into nabla helpers + component-leading 9-component staging pattern (cloned from ipovlpip)
provides:
  - int1e_irp (rank 9) manifest entries x {cart,sph,spinor}
  - one_electron_irp_kernel - the 3x3 (∇-axis ⊗ r-axis) tensor on the overlap-derivative engine (no Rys), reads PTR_COMMON_ORIG via rcj_1e_into, ket +2 headroom
  - rcj_1e_into #[cube] helper (libcint G1E_RCJ: RCJ[j][i] = src[j+1][i] + drj_axis*src[j][i])
  - is_irp dispatch arm in launch_one_electron_typed (fail-closed li+lj+2>8; spinor → UnsupportedApi)
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "int1e_irp = the 3x3 ∇⊗r tensor built from the overlap base g0 + THREE derived tensors: g1=D_J(g0) (∇ on ket), g2=RCJ(g0) (r on ket, drj=rj-common_orig), g3=D_J(g2) (∇ on the r-block). The 9 components s[0..8] are copied VERBATIM from intor1.c:2788-2816 per-axis g-table — s0=g3·g0·g0, s1=g2·g1·g0, ..., s8=g0·g0·g3 — NEVER re-derived"
    - "irp is a GAUGE-ORIGIN family (like Cluster A r-moments): the r part reads env[PTR_COMMON_ORIG] via drj=rj-common_orig. Because the gauge origin is non-zero ([0.5,-0.3,0.8]), irp is genuinely NON-ZERO on a SAME-center non-square s×p block — unlike p4 (∇⁴, origin-free, even-parity) which required a CROSS-center pair in 24-04. The irp test correctly stays on the same-center moment_common_orig_test! macro"
    - "rcj_1e_into is the device analogue of libcint's CINTx1j_1e/G1E_RCJ — a per-axis ket position-multiply. Combined with d_j_1e_into it composes the D_J(RCJ(g0)) chain. Headroom: g3=D_J(g2) reads g2 at j+1 and g2=RCJ reaches lj+1 reading g0 at j+1, so g0 must span lj+2 → nmax=li+lj+2, lj_ext=lj+2 (ket +2, ng={0,2,...})"

key-files:
  created: []
  modified:
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-ops/src/generated/api_manifest.csv
    - crates/cintx-cubecl/src/kernels/one_electron.rs

key-decisions:
  - "Built g2=RCJ(g0) as a FULL per-axis tensor (over j..=lj+1) via a new rcj_1e_into helper rather than inlining the moment ladder. The irp chain needs g3=D_J(g2), and d_j_1e_into operates on a whole 3-axis G-tensor, so g2 must exist as a complete tensor (not a per-element scalar). rcj_1e_into mirrors the structure of d_j_1e_into/d_i_1e_into for consistency and is the exact device form of libcint's G1E_RCJ macro"
  - "irp test stays on the SAME-center non-square block (the default O-1s × O-2p from non_square_shell_pair()). irp reads the non-zero gauge origin, so ⟨s|i·r×∇|p⟩ is non-trivially non-zero there — the parity gate is substantive. This is the opposite of 24-04's p4, which is origin-free and even-parity (zero same-center) and needed a cross-center pair. The prior-work note's caution about parity-zero does NOT apply to irp because the r-part carries the non-zero origin offset"
  - "component_rank = 9 (exact). The 9 components are emitted with comp as the OUTER staging stride (cart_out[base + comp*block_len + elem]), matching the component-leading layout the planner/c2s pipeline expects (cloned from the ipovlpip/p4 rank-9 staging). A too-low rank would truncate trailing components (D-08) — the non-square s×p parity gate at atol=1e-12 catches any permutation/truncation"

patterns-established:
  - "Gauge-origin overlap-derivative families (irp) are non-zero on a SAME-center non-square block when the operator carries an absolute-position term (drj ≠ 0); only PURE-derivative even families (p4) need a cross-center block. Choose the parity block by whether the operator reads a non-zero origin"

requirements-completed: [MOM-04]

# Metrics
duration: 22min
completed: 2026-05-30
---

# Phase 24 Plan 05: Cluster D int1e_irp (i·r×∇) Summary

**`int1e_irp` (i·r×∇, rank 9) — the 3×3 (∇-axis ⊗ r-axis) tensor on the overlap-derivative engine (no Rys), reading the gauge origin `env[PTR_COMMON_ORIG]` via `drj=rj-common_orig`, with ket +2 headroom — now matches vendored libcint 6.1.3 at atol=1e-12 (cart+sph) on a NON-SQUARE SAME-center block (O-1s × O-2p) on the non-zero gauge-origin fixture `[0.5,-0.3,0.8]`, with the 9 components `s[0..8]` copied verbatim from `intor1.c:2788-2816`. This is the FINAL plan of Phase 24 — all four MOM-04 sub-families (rinv, drinv, p4, irp) are now genuinely complete.**

## Performance

- **Duration:** ~22 min
- **Completed:** 2026-05-30
- **Tasks:** 1

## Accomplishments
- Registered 3 manifest entries (`int1e_irp` × {cart,sph,spinor}) with EXACT `component_rank "9"`, `operator "irp"`; cart/sph `oracle_covered=true`, spinor `oracle_covered=false` → `UnsupportedApi` (D-09). `cargo build -p cintx-ops` auto-regenerated `api_manifest.{rs,csv}`; `manifest-audit` status `ok`.
- Reused the `INT1E_IRP_*` RawApiId consts already declared in `raw.rs` by 24-03 — verified present (raw.rs:212-214), NOT re-added (per prior-work note).
- Implemented `one_electron_irp_kernel` (`#[cube(launch)]`, rank 9): builds the overlap base `g0`, then `g1=D_J(g0)` (∇ on ket), `g2=RCJ(g0)` (r on ket, gauge-origin offset `drj`), `g3=D_J(g2)` (∇ on the r-block) via the proven `d_j_1e_into` nabla helper plus a new `rcj_1e_into` position-multiply helper, and emits the 9 components `s[0..8]` in the VERBATIM `intor1.c:2788-2816` order. Ket +2 headroom (`nmax=li+lj+2`, `lj_ext=lj+2`, `ng={0,2,...}`).
- Added `rcj_1e_into` — the device form of libcint's `G1E_RCJ`/`CINTx1j_1e` per-axis ket position-multiply (`RCJ[j][i] = src[j+1][i] + drj_axis·src[j][i]`), structured to mirror `d_j_1e_into` so the `D_J(RCJ(g0))` chain composes cleanly.
- Wired the `is_irp` dispatch arm: reads the gauge origin from `plan.operator_env_params.common_orig` (defaults `[0,0,0]`), computes `drj`, calls `run_1e_irp_on_backend` (5-backend dispatch), applies `common_fac_sp` normalization, and stages the 9 components component-leading (cart + sph via `cart_to_sph_1e`). Fail-closed `li+lj+2>8` guard (UnsupportedApi, never truncate). Spinor → `UnsupportedApi`. Dropped `is_irp` from the 1e rejection guard.
- **Vendor parity GREEN** under the `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1` double gate: `test_int1e_irp_parity` byte-identical to vendored libcint 6.1.3 at atol=1e-12 (cart+sph, all 9 components) on the non-square same-center block on the non-zero gauge fixture. No regression: cubecl `--lib` 280/280, ops `--lib` 11/11, `moment_nontensor_parity` 4/4 green (p4 + rinv + drinv + irp), `manifest-audit` status `ok`.

## Task Commits

1. **Task 1 — register + implement int1e_irp (rank 9, gauge-origin overlap-derivative)** — see git log (feat)

## Decisions Made
- **g2=RCJ as a full tensor + new rcj_1e_into helper:** the irp chain needs `g3=D_J(g2)`, and `d_j_1e_into` operates on a whole 3-axis G-tensor, so `g2` must exist as a complete tensor. Added `rcj_1e_into` (the exact device form of libcint's `G1E_RCJ`) mirroring the `d_j_1e_into` structure, rather than reusing the per-element `moment_axis_ladder` from the Cluster-A moment kernel (which returns scalars, not a composable tensor).
- **Same-center parity block (no test change needed):** irp reads the non-zero gauge origin, so it is non-trivially non-zero on the default same-center O-1s × O-2p non-square block. The 24-01 RED scaffold's `moment_common_orig_test!` (same-center) is correct as written — unlike p4 (24-04) which is origin-free/even-parity and needed a cross-center pair.

## Deviations from Plan

**None — plan executed exactly as written.** The kernel math matched vendor on the first parity run on the same-center block the scaffold provided; no test-block change (the p4-style cross-center fix from 24-04 was NOT needed for irp because irp carries the non-zero gauge origin). No architectural changes.

## Threat Surface
No new trust boundaries. **T-24-05-01** (non-finite gauge origin in `env[1..3]`) is mitigated by the existing `validate_common_orig_env_params` (validator.rs) which rejects NaN/inf before the kernel and `eval_raw`'s bounds-guarded slot read — irp consumes the already-validated slot, no new validation gap. **T-24-05-02** (transposed/permuted 9-component emission) is mitigated by copying `s[0..8]` verbatim from `intor1.c:2788-2816` and gating on a NON-SQUARE block at atol=1e-12 (a square block is transpose-symmetric and hides the bug); `component_rank "9"` matches the true output (no truncation). No threat flags.

## Known Stubs
None for Cluster D (irp fully wired: manifest + RawApiId + kernel + vendor parity). Spinor irp is an intentional `UnsupportedApi` (D-09), registered for surface completeness. This is the LAST plan of Phase 24 — no remaining MOM-04 RED state.

## Self-Check: PASSED

- Created file exists: `.planning/phases/24-group-3-position-multipole-moment-integrals/24-05-SUMMARY.md` (FOUND).
- Modified files confirmed: `one_electron.rs` (`is_irp` arm + `one_electron_irp_kernel` + `rcj_1e_into` FOUND), `compiled_manifest.lock.json` (`int1e_irp_cart`, `component_rank "9"`), `api_manifest.{rs,csv}` (regenerated).
- `INT1E_IRP_CART` const present in raw.rs (from 24-03, not duplicated).
- Parity gate: `test_int1e_irp_parity` GREEN under the vendor double-gate at atol=1e-12 (cart+sph, 9 components) on the non-square same-center block; full `moment_nontensor_parity` 4/4; cubecl `--lib` 280/280; ops `--lib` 11/11; `manifest-audit` ok.

---
*Phase: 24-group-3-position-multipole-moment-integrals*
*Completed: 2026-05-30*
