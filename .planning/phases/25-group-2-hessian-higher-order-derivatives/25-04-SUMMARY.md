---
phase: 25-group-2-hessian-higher-order-derivatives
plan: 04
subsystem: kernels
tags: [libcint, hessian, 2e, ipip, rank-9, rank-81, cubecl, oracle, vendor-parity, re-home]

requires:
  - phase: 25-01
    provides: Rys nroots>=6 host engine (FND-02) — Hessian-elevated d-quartets reach the host path
  - phase: 25-02
    provides: fail-closed rank-9/81 staging (FND-06) — unconditional scatter, single upfront assertion
  - phase: 23
    provides: first-order nabla1{i,j,k,l}_2e engine + gout_ip1/gout_ipn host-derivative pattern
  - phase: 13
    provides: f12.rs gout_ipip1/ipvip1/ip1ip2 rank-9 Hessian gout helpers (reused verbatim)
provides:
  - int2e_ipip1 / int2e_ipvip1 re-homed from unstable::source::2e to STABLE (one canonical entry per symbol, no alias)
  - int2e_ip1ip2 (rank 9) + int2e_ipip1ipip2 (rank 81, 4th-order 2e) registered fresh
  - all 4 families byte-identical to vendor libcint 6.1.3 at atol=1e-12 (cart+sph, NON-SQUARE)
  - host-routed launch_two_electron_hess2e (fill_g_tensor_2e / FND-02) + new rank-81 gout_ipip1ipip2
  - 8 vendor_ffi wrappers + bindgen allowlist for the 4 cart/sph symbols
affects: [25-05, 25-06, hess]

tech-stack:
  added: []
  patterns:
    - "2e Hessian = first-order nabla1{i,j,k}_2e engine composed twice atop the plain Coulomb G-tensor (fill_g_tensor_2e), routed through the HOST Rys path so nroots>=6 d-quartets hit FND-02"
    - "Rank-9 Hessian gout helpers (gout_ipip1/ipvip1/ip1ip2) authored for F12 in Phase 13 are representation-agnostic — reused verbatim for the plain Coulomb launcher by making them pub(crate)"
    - "Rank-81 gout_ipip1ipip2: 16-buffer G2E_D_K/G2E_D_I composition + 81-term s[] triple product + column-major 9x9 reorder, copied 1:1 from hess.c CINTgout2e_int2e_ipip1ipip2"
    - "D-07 re-home: delete unstable lock entries + repoint the two source-only-gate raw.rs tests to a still-source-only symbol (int2e_breit_r1p2_spinor) instead of deleting them"

key-files:
  created:
    - crates/cintx-oracle/tests/hess2e_parity.rs
  modified:
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-compat/src/raw.rs
    - crates/cintx-cubecl/src/kernels/f12.rs
    - crates/cintx-cubecl/src/kernels/two_electron.rs
    - crates/cintx-oracle/build.rs
    - crates/cintx-oracle/src/vendor_ffi.rs

key-decisions:
  - "Reused the Phase-13 f12.rs gout_ipip1/ipvip1/ip1ip2 helpers verbatim (made pub(crate)) — the Hessian gout permutation is identical between F12 and plain Coulomb; only the G-tensor base differs (fill_g_tensor_2e vs fill_g_tensor_f12)"
  - "Single generalized host launcher launch_two_electron_hess2e dispatched by a Hess2eKind enum carrying per-family (i_inc,j_inc,k_inc) headroom — avoids 4 near-duplicate launchers"
  - "D-07 re-home preserved the two source-only-gate raw.rs tests by repointing them to int2e_breit_r1p2_spinor (still source-only, arity-4) rather than deleting coverage"
  - "Registered cart+sph+spinor per symbol (12 entries) matching the int2e_ip1/ip2 stable pattern; spinor stays oracle_covered=false -> UnsupportedApi (D-11)"

patterns-established:
  - "Pattern: high-rank (81) 2e derivative families reuse the verbatim hess.c gout via the existing nabla1*_2e composition; no new VRR/HRR math"
  - "Pattern: parametric vendor parity test (HessFamily/Case structs) sweeping 4 families x 2 reps x ranks 9/81 over a single non-square shell sweep"

requirements-completed: [HESS-02]

duration: 55min
completed: 2026-05-31
---

# Phase 25 Plan 04: HESS-02 2e Hessian families Summary

**int2e_ipip1/ipvip1 re-homed from unstable::source::2e to stable + int2e_ip1ip2 (rank 9) and int2e_ipip1ipip2 (rank 81, 4th-order 2e) registered fresh — all four byte-identical to vendor libcint 6.1.3 at atol=1e-12 (cart+sph, NON-SQUARE), host-routed through fill_g_tensor_2e (FND-02) reusing the Phase-13 Hessian gout helpers verbatim plus a new 16-buffer rank-81 gout.**

## Performance

- **Duration:** ~55 min
- **Started:** 2026-05-30
- **Completed:** 2026-05-31
- **Tasks:** 3 (Task 0 RED scaffold, Task 1 re-home+register+implement, Task 2 vendor+parity)
- **Files modified:** 7 (1 created, 6 modified)

## Accomplishments
- int2e_ipip1 / int2e_ipvip1 moved out of `unstable::source::2e` (2 lock entries deleted) into the stable 2e family as one canonical entry per symbol — zero `unstable::source::2e` entries remain, no alias, no lingering unstable feature-gate.
- int2e_ip1ip2 (rank 9) and int2e_ipip1ipip2 (rank 81 — the first rank-81 2e family, exercising the FND-06 rank-81 staging path) registered fresh with their true component_rank.
- New host-routed `launch_two_electron_hess2e` launcher + `Hess2eKind` dispatch; all four families route through the HOST `fill_g_tensor_2e` (FND-02 Rys), so nroots>=6 Hessian-elevated d-quartets are served (ceiling = HOST_RYS_NROOTS_CEILING=12; >12 fail-closed).
- New `gout_ipip1ipip2` (rank 81): 16-buffer G2E_D_K/G2E_D_I composition + verbatim 81-term s[] triple product + column-major reorder, copied 1:1 from `hess.c`.
- Vendor parity green at atol=1e-12, cart+sph, NON-SQUARE bra x ket, every component (ranks 9/9/9/81); manifest-audit status ok, 0 uncovered stable entries.

## Task Commits

1. **Task 0: RED parity scaffold** - `0b2ce0c` (test)
2. **Task 1: re-home + register + implement 4 families** - `b4503a6` (feat)
3. **Task 2: vendor FFI + parity green + oracle_covered** - `02e5317` (feat)

## Files Created/Modified
- `crates/cintx-oracle/tests/hess2e_parity.rs` - vendor-gated `hess2e_ipip` parity for all 4 families; parametric HessFamily/Case sweep, NON-SQUARE bra x ket (distinct l incl. the rank-81 case); determinism+shape test pins NCOMP*ni*nj*nk*nl (catches rank-81 truncation).
- `crates/cintx-ops/generated/compiled_manifest.lock.json` - DELETE 2 unstable entries; ADD 12 stable entries (cart/sph/spinor; ipip1/ipvip1/ip1ip2 rank 9, ipip1ipip2 rank 81); 8 cart+sph flipped oracle_covered=true after parity.
- `crates/cintx-compat/src/raw.rs` - 12 RawApiId consts (INT2E_IPIP1/IPVIP1/IP1IP2/IPIP1IPIP2 cart+sph+spinor); 2 source-only-gate tests repointed to int2e_breit_r1p2_spinor.
- `crates/cintx-cubecl/src/kernels/f12.rs` - gout_ipip1/ipvip1/ip1ip2 made pub(crate); NEW gout_ipip1ipip2 (rank 81, verbatim hess.c).
- `crates/cintx-cubecl/src/kernels/two_electron.rs` - Hess2eKind enum + launch_two_electron_hess2e host launcher + operator_name() dispatch arms.
- `crates/cintx-oracle/build.rs` - bindgen allowlist extended with the 8 cart/sph symbols (no new .file(); hess.c already compiles).
- `crates/cintx-oracle/src/vendor_ffi.rs` - 8 safe wrappers (4-shell arity; rank 9 + rank 81).

## Decisions Made
- The Hessian gout permutation is identical between the F12 and plain-Coulomb paths (only the G-tensor base differs), so the three rank-9 helpers were reused verbatim; only the rank-81 gout was new.
- A single generalized launcher with a `Hess2eKind` headroom descriptor replaces four near-duplicate launchers.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Repoint two source-only-gate tests after the D-07 re-home**
- **Found during:** Task 1 (manifest re-home)
- **Issue:** `crates/cintx-compat/src/raw.rs` had two `#[cfg(not(feature = "unstable-source-api"))]` tests (`safe_facade_gate_rejects_source_only_symbol_without_unstable_feature`, `source_only_symbol_requires_unstable_feature`) asserting that `int2e_ipip1_sph` is rejected as a source-only symbol. Promoting it to stable (D-07) would have made those tests fail (the symbol is no longer gated).
- **Fix:** Repointed both tests to `int2e_breit_r1p2_spinor` — a still-source-only, arity-4 symbol — preserving the unstable-feature-gate coverage instead of deleting it.
- **Files modified:** crates/cintx-compat/src/raw.rs
- **Committed in:** `b4503a6` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (blocking). No bugs: all four families matched the libcint recipe on first parity (the rank-9 gout helpers were already proven for F12; the rank-81 gout was copied verbatim from hess.c).
**Impact on plan:** Required for correctness of the re-home; no scope creep.

## Issues Encountered
- xtask is a standalone cargo project (own Cargo.lock), not a workspace member — `manifest-audit` runs from `xtask/` (`cd xtask && cargo run -- manifest-audit`), not via `-p xtask` at the workspace root.
- The `oracle_covered` flip used a JSON re-serializer (sort_keys) which reorders keys within each lock entry; this is cosmetic — both manifest-audit sides derive from the lock, so the regenerated CSV/RS and the audit are unaffected (status ok).

## Known Stubs
None — all 4 cart+sph families are fully wired and vendor-parity green. Spinor reps are intentional `UnsupportedApi` stubs (D-11), resolved when spinor-derivative transforms land (Phases 27/28).

## Threat Flags
None — the new surface (operator-name dispatch on `ipip1|ipvip1|ip1ip2|ipip1ipip2`) is numerical/component-correctness, fully covered by the threat register: T-25-12 (duplicate symbol — zero `unstable::source::2e` remain), T-25-13 (rank-81 truncation — component_rank=81, 81-component non-square parity green), T-25-14 (transpose — verbatim gout + non-square gate), T-25-15 (silent skip — double-gated parity, N>0). All mitigated.

## Next Phase Readiness
- Cluster B (HESS-02) complete; the host-routed `launch_two_electron_hess2e` + verbatim-gout reuse pattern and the parametric non-square parity harness are reusable for Plans 25-05/06.
- All four shared Wave-2 files (manifest lock, raw.rs, build.rs, vendor_ffi.rs) were edited additively (deletions limited to the two retired unstable entries) so the later Wave-2 plans append cleanly.
- Worktree integration: N/A (sequential executor on the main working tree; D-06 merge-base check applies only to worktree-parallelized clusters).

## Self-Check: PASSED

All created files exist on disk; all three task commits (0b2ce0c, b4503a6, 02e5317) present in git history.

---
*Phase: 25-group-2-hessian-higher-order-derivatives*
*Completed: 2026-05-31*
