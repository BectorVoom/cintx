---
phase: 25-group-2-hessian-higher-order-derivatives
plan: 03
subsystem: kernels
tags: [libcint, hessian, ipip, rank-9, cubecl, oracle, vendor-parity, 1e]

requires:
  - phase: 25-01
    provides: Rys nroots>=6 host engine (FND-02) — admits Hessian-elevated d-shells
  - phase: 25-02
    provides: fail-closed rank-9/81 staging (FND-06) — unconditional scatter, single upfront assertion
  - phase: 23
    provides: first-order nabla/gout_ipN 1e engine + rank-9 both-side launcher (ipovlpip/ipkinip/ipnucip)
provides:
  - int1e_ipipovlp / ipipnuc / ipipkin / ipiprinv registered (component_rank=9, cart+sph), byte-identical to vendor libcint 6.1.3 at atol=1e-12
  - bra-only rank-9 ∇² device kernels + shared gradgrad_bra_contract gout helper
  - 8 vendor_ffi wrappers + bindgen allowlist for the 4 ipip cart/sph symbols
affects: [25-04, 25-05, 25-06, hess]

tech-stack:
  added: []
  patterns:
    - "Bra-only ∇² 1e Hessian = first-order D_I engine applied twice (g1=D_I(g0,i+1), g2=D_I(g0,i), g3=D_I(g1,i)) atop the overlap (no-Rys) or nuclear (Rys) base"
    - "Shared #[cube] gout-permutation helper (gradgrad_bra_contract) reused by the overlap and nuclear bra-only kernels — the gout block is identical across ipipovlp/ipipnuc/ipiprinv"
    - "Kinetic ipipkin: 16-tensor flat buffer (d_i_1e_flat/d_j_1e_flat) + verbatim 27-term gout with the -½ kinetic factor folded into cintx's direct s-contraction"

key-files:
  created:
    - crates/cintx-oracle/tests/hess1e_ipip_parity.rs
  modified:
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-compat/src/raw.rs
    - crates/cintx-cubecl/src/kernels/one_electron.rs
    - crates/cintx-oracle/build.rs
    - crates/cintx-oracle/src/vendor_ffi.rs

key-decisions:
  - "ipipovlp/ipipnuc/ipiprinv share one gout permutation helper (gradgrad_bra_contract); only the G-tensor base (overlap vs nuclear Rys) differs"
  - "ipipkin's -½ kinetic factor is folded into the gout (cintx contracts s directly; libcint scales in CINT1e_drv) — without it observed = 2× vendor"
  - "Bra-only headroom is ng={2,0,...} (nmax=li+lj+2, lj_ext=lj) for ovlp/nuc/rinv; ipipkin uses ng={2,2,...} (nmax=li+lj+4, lj_ext=lj+2) per hess.c"
  - "spinor ipip reps registered with oracle_covered=false → UnsupportedApi (D-11); no capi/legacy surface"

patterns-established:
  - "Pattern: bra-only second-derivative 1e families reuse the d_i_1e_into helper twice; no new VRR/HRR math"
  - "Pattern: NON-SQUARE p×d single-block parity in addition to the full multi-shell matrix (D-09 transpose gate)"

requirements-completed: [HESS-01]

duration: 70min
completed: 2026-05-30
---

# Phase 25 Plan 03: HESS-01 rank-9 1e Hessian families Summary

**int1e_ipipovlp/ipipnuc/ipipkin/ipiprinv registered (component_rank=9, cart+sph) and byte-identical to vendor libcint 6.1.3 at atol=1e-12 — bra-only ∇² device kernels reusing the Phase-23 first-order D_I engine twice, with the verbatim hess.c gout permutation.**

## Performance

- **Duration:** ~70 min
- **Started:** 2026-05-30T14:02Z
- **Completed:** 2026-05-30
- **Tasks:** 3 (Task 0 RED scaffold, Task 1 register+implement, Task 2 vendor+parity)
- **Files modified:** 6 (1 created, 5 modified)

## Accomplishments
- All 4 rank-9 1e Hessian families registered (manifest + RawApiId + launcher dispatch + vendor FFI + oracle) and vendor-parity green at atol=1e-12, cart+sph, every one of the 9 components.
- Three new bra-only ∇² `#[cube]` device kernels: overlap (no-Rys), nuclear/rinv (Rys), and the 16-tensor kinetic — all generic over `F` and routed through the existing 5-arm backend dispatch.
- Parity gated on a NON-SQUARE p×d block AND the full H2O STO-3G multi-shell matrix (D-09); manifest-audit green (status ok, 0 uncovered stable entries).

## Task Commits

1. **Task 0: RED parity scaffold** - `257df49` (test)
2. **Task 1: register + implement 4 families** - `fdd4dda` (feat)
3. **Task 2: vendor FFI + parity green + oracle_covered** - `69a5133` (feat)

## Files Created/Modified
- `crates/cintx-oracle/tests/hess1e_ipip_parity.rs` - vendor-gated `hess1e_ipip` parity for all 4 families; NON-SQUARE p×d + full matrix; PTR_ENV_START-aligned env; ipiprinv sets a nonzero rinv origin.
- `crates/cintx-ops/generated/compiled_manifest.lock.json` - 12 ipip entries (cart/sph/spinor, component_rank=9); cart+sph flipped oracle_covered=true after parity.
- `crates/cintx-compat/src/raw.rs` - 12 RawApiId consts (INT1E_IPIP{OVLP,NUC,KIN,RINV}_{CART,SPH,SPINOR}).
- `crates/cintx-cubecl/src/kernels/one_electron.rs` - bra-only kernels (`one_electron_gradgrad_bra_ovlp_kernel`, `one_electron_nuc_gradgrad_bra_kernel`, `one_electron_gradgrad_bra_kin_kernel`) + `gradgrad_bra_contract` + `d_i_1e_flat`/`d_j_1e_flat` helpers + launcher dispatch (`is_rank9_bra`).
- `crates/cintx-oracle/build.rs` - bindgen allowlist extended with the 8 cart/sph ipip symbols (no new `.file()`; hess.c already compiles).
- `crates/cintx-oracle/src/vendor_ffi.rs` - 8 safe wrappers.

## Decisions Made
- Reused the libcint gout permutation `[s0,s3,s6,s1,s4,s7,s2,s5,s8]` verbatim across ipipovlp/ipipnuc/ipiprinv via one shared `#[cube]` helper — these three share an identical s-tensor; only the G0 base differs.
- ipipkin ported as a distinct 16-tensor kernel (its D_J²·D_I² recipe and 27-term gout differ entirely from the other three).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] ipipkin -½ kinetic factor**
- **Found during:** Task 2 (vendor parity)
- **Issue:** ipipkin observed output was exactly 2× the vendor reference. libcint's `CINTgout1e_int1e_ipipkin` emits `gout = -(s)` where `s` carries the full ∇²_j; the kinetic operator's ½ is applied later in `CINT1e_drv`. cintx contracts `s` directly into staging, so the ½ must be folded into the gout.
- **Fix:** Changed the 9 gout terms from `-s` to `-0.5·s`.
- **Files modified:** crates/cintx-cubecl/src/kernels/one_electron.rs
- **Verification:** `test_int1e_ipipkin_h2o_sto3g_parity` (and the other 7) pass at atol=1e-12 cart+sph.
- **Committed in:** `69a5133` (Task 2 commit)

**2. [Rule 3 - Blocking] PTR_ENV_START-aligned test fixture**
- **Found during:** Task 0 (test scaffold)
- **Issue:** The Phase-23 both-side fixture packs atom coords at env[0..9], which clobbers PTR_RINV_ORIG (env[4..6]) — not usable for the ipiprinv family (single rinv origin).
- **Fix:** Reserved env[0..PTR_ENV_START); ipiprinv test injects a nonzero rinv origin via `env_with_rinv_origin`.
- **Files modified:** crates/cintx-oracle/tests/hess1e_ipip_parity.rs
- **Verification:** ipiprinv parity green at the shifted origin.
- **Committed in:** `257df49` (Task 0) / used in `69a5133`

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking)
**Impact on plan:** Both essential for correctness; no scope creep. The ipipkin ½ was the one substantive numerical bug; everything else matched the libcint recipe on first parity.

## Issues Encountered
- xtask is a standalone cargo project (own Cargo.lock), not a workspace member — `manifest-audit` must be run from `xtask/` (`cd xtask && cargo run -- manifest-audit`), not via `-p xtask` at the workspace root.

## Known Stubs
None — all 4 cart+sph families are fully wired and vendor-parity green. Spinor reps are intentional `UnsupportedApi` stubs (D-11), resolved when spinor-derivative transforms land (Phases 27/28).

## Threat Flags
None — the new surface (operator-name dispatch on `ipipovlp|ipipnuc|ipipkin|ipiprinv`) is numerical/component correctness, fully covered by the threat register (T-25-09 component_rank=9, T-25-10 verbatim gout + non-square gate, T-25-11 double-gated parity), all mitigated.

## Next Phase Readiness
- Cluster A (HESS-01) complete; the bra-only ∇² + shared gout-contract pattern and the env-aligned non-square parity harness are reusable for Plans 25-04..06.
- All four edited shared files (manifest lock, raw.rs, build.rs, vendor_ffi.rs) were appended additively so the later three Wave-2 plans append cleanly.
- Worktree integration N/A (sequential executor on main working tree).

## Self-Check: PASSED

All created files exist on disk; all three task commits (257df49, fdd4dda, 69a5133) present in git history.

---
*Phase: 25-group-2-hessian-higher-order-derivatives*
*Completed: 2026-05-30*
