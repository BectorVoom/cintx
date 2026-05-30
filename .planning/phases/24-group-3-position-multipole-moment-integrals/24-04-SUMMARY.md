---
phase: 24-group-3-position-multipole-moment-integrals
plan: 04
subsystem: kernels
tags: [cubecl, kernel, p4, overlap-derivative, both-side-headroom, parity, multipole]

# Dependency graph
requires:
  - phase: 24-group-3-position-multipole-moment-integrals
    plan: 01
    provides: vendor_int1e_p4_{cart,sph} FFI wrappers + rank-parameterized vendor_parity + RED moment_nontensor_parity p4 scaffold + cross_center_non_square_shell_pair helper
  - phase: 24-group-3-position-multipole-moment-integrals
    plan: 02
    provides: Cluster A moment kernel + manifest/RawApiId registration recipe + cross-center parity pattern for even-parity families
  - phase: 24-group-3-position-multipole-moment-integrals
    plan: 03
    provides: INT1E_P4_* RawApiId symbol consts (declared in raw.rs, no manifest/kernel until this plan)
  - phase: 23-group-1-remaining-1st-derivative-families-cart-sph
    provides: both-side derivative #[cube] nabla helpers (d_i_1e_into / d_j_1e_into) + the kinetic both-side kernel template (nmax=li+lj+4, lj_ext)
provides:
  - int1e_p4 (rank 1) manifest entries × {cart,sph,spinor}
  - one_electron_p4_kernel — ∇⁴ Laplacian-of-Laplacian on the overlap G-tensor (no Rys), BOTH-side +2 headroom
  - is_p4 dispatch arm in launch_one_electron_typed (fail-closed li+lj+4>8; spinor → UnsupportedApi)
affects: [24-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "int1e_p4 = ∇⁴ built ENTIRELY from the existing #[cube] nabla helpers (d_i_1e_into / d_j_1e_into): g0 (overlap base) → dj2=D_J²(g0) → di2=D_I²(g0) → di2dj2=D_I²(D_J²(g0)). The rank-1 Laplacian² contraction s0+2·s4+2·s8+s40+2·s44+s80 (intor1.c:2534) survives to only FOUR distinct g-tensors of the 16 libcint builds — copied verbatim, never re-derived"
    - "p4 raises angular-momentum headroom on BOTH bra and ket (ng={2,2,...}): nmax=li+lj+4, lj_ext=lj+2, bra VRR built to li+2. This distinguishes it from Cluster A's ket-only headroom (Pitfall 4). The same nmax=li+lj+4 sizing as the kinetic both-side kernel — reused as the structural template"
    - "p4 reads NO origin (pure derivative operator). Even-parity ⟨s|∇⁴|p⟩ on a SAME-center block is identically zero (vendor included), so the parity gate uses a CROSS-center non-square pair (H1-1s × O-2p) — the same even-parity cross-center pattern the _origj even-moment families required in 24-02"

key-files:
  created: []
  modified:
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-ops/src/generated/api_manifest.csv
    - crates/cintx-cubecl/src/kernels/one_electron.rs
    - crates/cintx-oracle/tests/moment_nontensor_parity.rs

key-decisions:
  - "p4 only needs FOUR of libcint's 16 g-tensors. The final rank-1 contraction s0+2·s4+2·s8+s40+2·s44+s80 references only g0, g3=D_J²(g0), g12=D_I²(g0), g15=D_I²(D_J²(g0)). I built exactly those four (via two D_J + two D_I + a two-step mixed chain) rather than replicating libcint's full g0..g15 cascade — fewer scratch tensors, identical result. The two unused axes (b2x of dj2, c2z of di2) are deliberately not read"
  - "p4 parity uses a CROSS-center non-square block. The 24-01 RED scaffold defaulted p4 to the SAME-center O-1s × O-2p pair; ∇⁴ is even and reads no origin, so ⟨s|∇⁴|p⟩ same-center = 0 by parity (cintx AND vendor), tripping the non-zero guard. Added a dedicated moment_common_orig_test_cross macro that drives vendor_parity_at with cross_center_non_square_shell_pair() (H1-1s × O-2p) — substantive AND still non-square (D-07 transpose gate preserved). The irp test (24-05) stays on the same-center macro; irp reads PTR_COMMON_ORIG and is non-zero there"
  - "Fail-closed at li+lj+4>8 (UnsupportedApi), never truncate (T-24-04-02). On STO-3G (li,lj≤1) the internal nmax≤6, well within the li+lj<=8 overlap-derivative engine limit"

patterns-established:
  - "Even-parity operators with no origin (p4, and prospectively any pure-derivative even family) MUST gate on a CROSS-center non-square block — a same-center block gives identically-zero integrals on both sides and trivially trips assert_any_nonzero"

requirements-completed: [MOM-04]

# Metrics
duration: 38min
completed: 2026-05-30
---

# Phase 24 Plan 04: Cluster C int1e_p4 (∇⁴) Summary

**`int1e_p4` (∇⁴, rank 1) — the Laplacian-of-Laplacian built on the overlap-derivative engine (no Rys) with BOTH-side +2 angular-momentum headroom (ng={2,2,...}) — now matches vendored libcint 6.1.3 at atol=1e-12 (cart+sph) on a NON-SQUARE CROSS-center block (H1-1s × O-2p), with the rank-1 contraction `s0+2·s4+2·s8+s40+2·s44+s80` copied verbatim from `intor1.c:2534`.**

## Performance

- **Duration:** ~38 min
- **Completed:** 2026-05-30
- **Tasks:** 1

## Accomplishments
- Registered 3 manifest entries (p4 × {cart,sph,spinor}) with EXACT `component_rank "1"`, `operator "p4"`; cart/sph `oracle_covered=true`, spinor `oracle_covered=false` → `UnsupportedApi` (D-09). `cargo build -p cintx-ops` auto-regenerated `api_manifest.{rs,csv}`; `manifest-audit` status `ok`.
- Reused the `INT1E_P4_*` RawApiId consts already declared in `raw.rs` by 24-03 — verified present, NOT re-added (per prior-work note).
- Implemented `one_electron_p4_kernel` (`#[cube(launch)]`, rank 1): builds the overlap base G-tensor `g0` then the four surviving derivative tensors `g3=D_J²(g0)`, `g12=D_I²(g0)`, `g15=D_I²(D_J²(g0))` via the proven `d_i_1e_into` / `d_j_1e_into` nabla helpers, and emits the rank-1 Laplacian² contraction verbatim. BOTH-side +2 headroom (`nmax=li+lj+4`, `lj_ext=lj+2`, bra VRR to `li+2`) — the Pitfall-4 distinction from Cluster A's ket-only headroom.
- Wired the `is_p4` dispatch arm with the 5-backend run helper (`run_1e_p4_on_backend`) and a fail-closed `li+lj+4>8` guard (UnsupportedApi, never truncate). Spinor → `UnsupportedApi`. Dropped `is_p4` from the 1e rejection guard.
- Corrected the p4 parity test to a CROSS-center non-square block (∇⁴ is even, reads no origin → same-center ⟨s|∇⁴|p⟩ is parity-zero on both sides). Added `moment_common_orig_test_cross` driving `vendor_parity_at(cross_center_non_square_shell_pair())`; irp stays on the same-center macro for 24-05.
- **Vendor parity GREEN** under the `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1` double gate: `test_int1e_p4_parity` byte-identical to vendored libcint 6.1.3 at atol=1e-12 (cart+sph) on the non-square cross-center block. No regression: cubecl `--lib` 280/280, compat `--lib` 43/43, ops `--lib` 11/11, moment_nontensor_parity 3/4 green (p4 + rinv + drinv; irp fails-closed at the resolver as MissingSymbol — Cluster D / plan 24-05, the expected RED state), manifest-audit ok.

## Task Commits

1. **Task 1 — register + implement int1e_p4 (rank 1, both-side headroom)** — `975c5ff` (feat)

## Decisions Made
- **Four tensors, not sixteen:** the rank-1 Laplacian² contraction collapses libcint's full g0..g15 cascade to four distinct g-tensors (g0, D_J², D_I², D_I²·D_J²). I built exactly those — minimal scratch, identical result — rather than replicating the unused intermediates.
- **Cross-center parity block (test fix):** ∇⁴ is even and origin-free, so the 24-01 RED default (same-center O-1s × O-2p) gives identically-zero p4 on both cintx and vendor. The fix is a cross-center non-square pair (H1-1s × O-2p) — substantive AND transpose-safe — applied via a dedicated p4-only macro so the irp same-center test (24-05) is untouched.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking test scaffold defect] p4 parity test required a CROSS-center non-square block**
- **Found during:** Task 1 (RED→GREEN: the p4 parity test failed `assert_any_nonzero` with "block is all-zero").
- **Issue:** The 24-01 RED scaffold drove `test_int1e_p4_parity` through the shared `moment_common_orig_test` macro, which defaults to the SAME-center non-square pair (O-1s × O-2p). But `int1e_p4 = ∇⁴` is an EVEN (4-derivative) operator that reads NO origin, so ⟨s|∇⁴|p⟩ on a same-center block is identically zero by parity — for BOTH cintx and vendored libcint. The test could never pass on that block regardless of kernel correctness (I confirmed the cintx kernel produces zero there even with the math replaced by a plain overlap — pure parity, not a kernel bug).
- **Fix:** Added a `moment_common_orig_test_cross` macro that drives `vendor_parity_at` with `cross_center_non_square_shell_pair()` (H1-1s × O-2p) — non-square AND cross-center, so the integral is genuinely non-zero and the comparison is substantive (the same even-parity cross-center pattern the `_origj` even-moment families used in 24-02). The `irp` test stays on the original same-center macro (irp reads PTR_COMMON_ORIG and is non-zero there; it remains 24-05's RED target).
- **Files modified:** crates/cintx-oracle/tests/moment_nontensor_parity.rs
- **Commit:** `975c5ff`

**Total deviations:** 1 auto-fixed (blocking test-scaffold defect: wrong shell pair for the even-parity p4 operator). No architectural changes. The p4 kernel math matched vendor on the first parity run on the corrected block — no math defect.

## Threat Surface
No new trust boundaries. T-24-04-01 (ket-only headroom mistake → wrong l>0 result) is mitigated by building BOTH the bra (D_I²) AND ket (D_J²) Laplacians with `ng={2,2,...}` headroom and gating on a NON-SQUARE l-bearing block (l_j=1) at atol=1e-12 — a ket-only error would diverge there. T-24-04-02 (internal nmax exceeding the engine limit) is mitigated by the `li+lj+4>8` fail-closed guard (UnsupportedApi, never an OOB read); on STO-3G nmax≤6. No new env-validation surface (p4 reads no origin). No threat flags.

## Known Stubs
None for Cluster C (p4 fully wired: manifest + RawApiId + kernel + vendor parity). Spinor p4 is an intentional `UnsupportedApi` (D-09), registered for surface completeness. The `int1e_irp` MissingSymbol fail-closure observed in `moment_nontensor_parity` is the EXPECTED RED state for plan 24-05 (Cluster D) — irp has no manifest row/kernel yet, so its dispatch fails closed at the resolver (no partial/incorrect result).

## Self-Check: PASSED

- Created file exists: `.planning/phases/24-group-3-position-multipole-moment-integrals/24-04-SUMMARY.md` (FOUND).
- Commit present in git history: `975c5ff` (FOUND).
- Modified files confirmed: one_electron.rs (is_p4 arm FOUND), compiled_manifest.lock.json (int1e_p4_cart, component_rank "1"), api_manifest.{rs,csv}, moment_nontensor_parity.rs.
- Parity gate: `test_int1e_p4_parity` GREEN under the vendor double-gate at atol=1e-12 (cart+sph) on the non-square cross-center block; cubecl --lib 280/280, compat 43/43, ops 11/11, manifest-audit ok.

---
*Phase: 24-group-3-position-multipole-moment-integrals*
*Completed: 2026-05-30*
