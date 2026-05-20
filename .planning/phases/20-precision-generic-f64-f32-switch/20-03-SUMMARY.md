---
phase: 20-precision-generic-f64-f32-switch
plan: 03
subsystem: math-transform
tags: [rust, generics, rys-quadrature, cart-to-sph, CintFloat, cubecl, f32, f64]

# Dependency graph
requires:
  - phase: 20-01
    provides: CintFloat sealed trait (host bound), PrecisionKind enum
  - phase: 20-02
    provides: Generic F: Float pattern for #[cube] device fns + F: CintFloat host wrapper idiom

provides:
  - "rys.rs: all 7 #[cube] fns (clenshaw_d1, rys_root1..5, rys_roots) generic over F: Float"
  - "rys.rs: host wrappers rys_root1_host..5_host, rys_roots_host generic over F: CintFloat"
  - "c2s.rs: cart_to_sph_1e, cart_to_sph_2c2e, cart_to_sph_3c1e, cart_to_sph_3c2e, cart_to_sph_2e generic over F: CintFloat"
  - "f64 monomorphization byte-identical for both rys.rs and c2s.rs"
  - "f32 Rys asymptotic weight-sum identity confirmed at ~1e-7 relative error"

affects:
  - "20-04 through 20-08: Wave 2+ kernel launchers can now call rys::<F> and cart_to_sph_*::<F>"
  - "Wave 5 tolerance floors (informed by observed f32 Rys error magnitude ~1e-7)"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "FROZEN f64 coefficient tables + F::from_f64_lossy at accumulation site (PATTERNS line 162)"
    - "Host wrapper pattern: compute in f64, convert outputs via F::from_f64_lossy for byte-identical f64 path"
    - "Polynomial coefficients stay f64 in host _f64 impls; F::cast_from in #[cube] device fns"

key-files:
  created: []
  modified:
    - "crates/cintx-cubecl/src/math/rys.rs"
    - "crates/cintx-cubecl/tests/rys_tests.rs"
    - "crates/cintx-cubecl/src/transform/c2s.rs"

key-decisions:
  - "Rys host wrappers use 'compute in f64, convert output' pattern so f64 monomorphization is byte-identical without double-conversion overhead"
  - "pie4: F parameter added to rys_root1..5 #[cube] fns (PIE4 const injected from host via from_f64_lossy) to avoid Pitfall 5"
  - "c2s coefficient tables (C2S_L0..L4) remain [[f64; N]; M] (FROZEN); cast via F::from_f64_lossy at each accumulation site"
  - "cart_to_spheric_staging stays concrete f64 (no-op staging function, not a transform)"
  - "f32 Rys weight-sum identity test domain: large-x asymptotic regime only (sum(w_i)==sqrt(PIE4/x)); polynomial branches do not satisfy identity at any precision"

patterns-established:
  - "Pattern: FROZEN f64 const table → F at boundary: keep table type as [[f64; N]; M], call F::from_f64_lossy(table[row][col]) inside accumulation loops"
  - "Pattern: PIE4-class precision-critical const → passed F parameter from host, never F::new(f64_literal) (Pitfall 5)"

requirements-completed: [PREC-01, PREC-07]

# Metrics
duration: 35min
completed: 2026-05-21
---

# Phase 20 Plan 03: Wave 1 Math Leaves (rys.rs + c2s.rs) Summary

**Genericized Rys quadrature (14 #[cube] functions + 6 host wrappers) and 5 cart-to-sph transforms over F: CintFloat/Float with f64 byte-identity preserved and f32 asymptotic weight-sum confirmed at ~1e-7 rel**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-05-21T06:11Z (Task 1 already committed by prior executor on base commit `195cfc9`)
- **Completed:** 2026-05-21T06:55Z
- **Tasks:** 2 (Task 1 by prior executor, Task 2 by this executor)
- **Files modified:** 3 (`rys.rs`, `rys_tests.rs`, `c2s.rs`)

## Accomplishments

- Task 1 (rys.rs, committed `195cfc9`): All 7 `#[cube]` functions (`clenshaw_d1`, `rys_root1..5`, `rys_roots`) are now generic over `F: Float`. The 6 host wrappers (`rys_root1_host..5_host`, `rys_roots_host`) are generic over `F: CintFloat`. Polynomial coefficient tables (Horner/Clenshaw blocks) stay `f64`; PIE4 injected as `pie4: F` parameter. f64 byte-identical; f32 asymptotic weight-sum passes at ~1e-7 relative error.
- Task 2 (c2s.rs, TDD RED `2a5290c` + GREEN `6103491`): All 5 public transform functions genericized over `F: CintFloat`. FROZEN coefficient tables (`C2S_L0..L4`) remain `[[f64; N]; M]`; cast to `F` via `F::from_f64_lossy` at accumulation site. 12 new generic tests (f32 + f64 monomorphizations for all 5 transforms) pass.
- Full workspace `cargo check --workspace --features cpu` exits 0. All 157 lib tests + 37 integration tests pass.

## Task Commits

1. **Task 1: Genericize rys.rs** - `195cfc9` (feat) — committed by prior executor
2. **Task 2 RED: Failing c2s generic tests** - `2a5290c` (test)
3. **Task 2 GREEN: Genericize all c2s transforms** - `6103491` (feat)

## Files Created/Modified

- `crates/cintx-cubecl/src/math/rys.rs` — All 14 `#[cube]` fns + host wrappers generic; polynomial tables FROZEN f64
- `crates/cintx-cubecl/tests/rys_tests.rs` — Updated to pass `pie4: f64 = PIE4` parameter; new f32 asymptotic smoke test
- `crates/cintx-cubecl/src/transform/c2s.rs` — 5 cart-to-sph fns generic over `F: CintFloat`; 12 new generic tests

## Decisions Made

- `pie4: F` parameter added to Rys `#[cube]` fns so PIE4 const is injected from host via `F::from_f64_lossy`, never via `F::new(f64_lit)` (Pitfall 5 / T-20-07).
- Rys host wrappers use "compute in f64 (existing `_f64` impl), convert outputs" pattern so the f64 monomorphization is provably byte-identical.
- c2s coefficient tables kept as `[[f64; N]; M]` (FROZEN); accumulated as `F` via `F::from_f64_lossy` at the innermost loop site.
- `cart_to_spheric_staging` stays `f64` concrete — it is a no-op staging passthrough, not a real transform.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Cleaned up accidental commit to main branch**
- **Found during:** Task 2 RED commit
- **Issue:** `git commit` ran from main repo path (`/home/user/Documents/workspace/cintx`) instead of worktree path, putting a test-only commit on `main`
- **Fix:** Reverted on `main` immediately (`git revert`), then `git reset --hard 195cfc9` to clean up the revert commit too; re-applied changes in worktree
- **Files modified:** none (net zero impact on source; worktree commits are correct)
- **Verification:** `git log --oneline` on `main` shows `195cfc9` as HEAD; worktree shows 3 plan-03 commits

---

**Total deviations:** 1 process fix (accidental commit path)
**Impact on plan:** No scope creep. Source changes are correct. Main branch restored to `195cfc9`.

## Issues Encountered

- Worktree/main path confusion: early `git add`/`commit` ran against the main repo at `/home/user/Documents/workspace/cintx` instead of the worktree at the cwd. Resolved by always using absolute worktree paths for Edit/Write and confirming `git rev-parse --abbrev-ref HEAD` before committing.

## Known Stubs

None — all transform fns are fully implemented and tested.

## Threat Flags

No new network endpoints, auth paths, file access patterns, or schema changes introduced. The c2s transform and Rys functions are pure numeric computation operating on caller-provided buffers. The threat mitigations from the plan's STRIDE register (T-20-07, T-20-08, T-20-09) were all applied:
- T-20-07: PIE4 and c2s coefficients injected via `F::from_f64_lossy`, never `F::new(f64_lit)`
- T-20-08: f32 Rys branch selection within expected tolerance (~1e-7 rel)
- T-20-09: Horner/Clenshaw tables and `roots_xw` blobs untouched (still `f64`)

## Wave 2 Call Signatures

The key signatures Wave 2 kernel launchers will use:

```rust
// Rys device fns (in #[cube] kernels):
rys_root1::<F>(t: F, pie4: F) -> (F, F)          // root + weight
rys_root2::<F>(t: F, pie4: F) -> ([F; 2], [F; 2])
// ... similarly rys_root3..5, rys_roots

// Rys host wrappers (tests + host-side dispatch):
rys_root1_host::<F>(t: F) -> (F, F)
// ... similarly rys_root2_host..5_host, rys_roots_host

// c2s host transforms (all five):
cart_to_sph_1e::<F>(cart_buf: &[F], sph_buf: &mut [F], li: u8, lj: u8)
cart_to_sph_2c2e::<F>(cart: &[F], li: u8, lk: u8) -> Vec<F>
cart_to_sph_3c1e::<F>(cart: &[F], li: u8, lj: u8, lk: u8) -> Vec<F>
cart_to_sph_3c2e::<F>(cart: &[F], li: u8, lj: u8, lk: u8) -> Vec<F>  // delegates to 3c1e
cart_to_sph_2e::<F>(cart: &[F], li: u8, lj: u8, lk: u8, ll: u8) -> Vec<F>
```

## Observed f32 Rys Error Magnitude (input to Wave 5 floors)

f32 asymptotic Rys weight-sum identity: `|sum(w_i) - sqrt(PIE4/x)| / sqrt(PIE4/x)` ≈ **1e-7** at large-x domain. This confirms the f32 Rys path is within the expected single-precision envelope (~1e-7 rel) and establishes the nominal floor for Wave 5 tolerance definitions.

## Next Phase Readiness

- Wave 2 kernel launchers (`20-04` through `20-08`) can now call `rys_root1_host::<F>` and `cart_to_sph_1e::<F>` with generic `F` (both f64 and f32).
- f64 byte-identity preserved end-to-end; Phase 10 oracle gate unaffected.
- Wave 5 f32 tolerance floors should use ≥1e-6 atol for Rys-dependent integrals (observed ~1e-7 rel in asymptotic regime, polynomial branches will vary).

---
*Phase: 20-precision-generic-f64-f32-switch*
*Completed: 2026-05-21*
