---
phase: 14-unstable-source-api-families
plan: 02
subsystem: kernels
tags: [origi, origk, ssc, unstable-source-api, oracle-parity, 1e, 3c1e, 3c2e]

requires:
  - phase: 14-01
    provides: "Feature gates, manifest entries, vendor FFI, kernel stubs"
provides:
  - "Real launch_origi with 4-variant dispatch through 1e G-tensor infrastructure"
  - "Real launch_origk with 6-variant dispatch through 3c1e G-tensor infrastructure"
  - "Real launch_ssc through 3c2e infrastructure with SSC c2s (k stays Cartesian)"
  - "11 oracle parity tests (10 pass at atol=1e-12, 1 deferred)"
affects: [14-03, 14-04, 14-05]

tech-stack:
  added: []
  patterns:
    - "G1E_R_I/R_K pointer-shift pattern for origin-displaced operators"
    - "D_I/D_J nabla derivative operators for ip1/ip2 variants"
    - "SSC c2s: spherical on i,j; Cartesian on k"
    - "Block-major multi-component layout for c2s transform"

key-files:
  created: []
  modified:
    - "crates/cintx-cubecl/src/kernels/unstable.rs"
    - "crates/cintx-oracle/tests/unstable_source_parity.rs"
    - "crates/cintx-compat/Cargo.toml"
    - "crates/cintx-compat/src/raw.rs"
    - "crates/cintx-runtime/src/dispatch.rs"
    - "crates/cintx-ops/src/generated/api_manifest.rs"

key-decisions:
  - "Fix unstable-source-api feature forwarding: cintx-compat must forward to cintx-cubecl for kernel dispatch"
  - "Fix source-only profile gate: check unstable-source profile for source-only entries instead of base profile"
  - "Extend DispatchFamily to map unstable family names (unstable::source::origi -> OneElectron, etc.)"
  - "Fix manifest component_rank for derivative variants: rank '1' should be ncomp=3, not multiplier=1"
  - "Block-major multi-component output layout: each component as separate Cartesian block for c2s"

patterns-established:
  - "Unstable-source dispatch: family_name uses unstable::source:: prefix, canonical_family uses short name"
  - "ip1/ip2 derivative gout: D_I/D_J applied per-axis with R_K/R_I shifts, multinomial expansion for r^n"

requirements-completed: [USRC-01, USRC-04, USRC-05]

duration: 52min
completed: 2026-04-05
---

# Phase 14 Plan 02: Origi/Origk/SSC Kernel Families Summary

**Origin-displaced r^n (origi/origk) and spin-spin contact (ssc) kernels with 10/11 oracle parity tests at atol=1e-12**

## Performance

- **Duration:** 52 min
- **Started:** 2026-04-05T21:30:55Z
- **Completed:** 2026-04-05T22:23:00Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- Implemented launch_origi: 4 variants (r2, r4, r2_ip2, r4_ip2) using 1e G-tensor with elevated i-ceiling
- Implemented launch_origk: 6 variants (r2, r4, r6, ip1_r2, ip1_r4, ip1_r6) using 3c1e G-tensor with elevated k-ceiling
- Implemented launch_ssc: standard 3c2e gout with SSC c2s transform (k stays Cartesian)
- 10 of 11 oracle parity tests pass at atol=1e-12 against vendored libcint 6.1.3

## Task Commits

1. **Task 1: Implement kernel launch functions** - `6dd7078` (feat)
2. **Task 2: Oracle parity tests + infrastructure fixes** - `b1a4f0f` (feat)

## Files Created/Modified
- `crates/cintx-cubecl/src/kernels/unstable.rs` - Real origi/origk/ssc kernel implementations
- `crates/cintx-oracle/tests/unstable_source_parity.rs` - 11 oracle parity tests
- `crates/cintx-compat/Cargo.toml` - Feature forwarding fix
- `crates/cintx-compat/src/raw.rs` - Source-only profile gate fix
- `crates/cintx-runtime/src/dispatch.rs` - DispatchFamily unstable family support
- `crates/cintx-ops/src/generated/api_manifest.rs` - Restored entries + component_rank fixes

## Decisions Made
- Fix unstable-source-api feature forwarding: cintx-compat must forward to cintx-cubecl
- Fix source-only profile gate: check unstable-source profile instead of base
- Extend DispatchFamily for unstable families (origi->1e, origk->3c1e, ssc->3c2e, grids->1e, breit->2e)
- Fix manifest component_rank for derivative variants: rank-1 tensors need ncomp=3 not 1
- Block-major multi-component output layout for c2s compatibility

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Feature forwarding gap: unstable-source-api not reaching cintx-cubecl**
- **Found during:** Task 2 (oracle parity tests)
- **Issue:** cintx-compat/Cargo.toml had `unstable-source-api = []` without forwarding to cintx-cubecl
- **Fix:** Changed to `unstable-source-api = ["cintx-cubecl/unstable-source-api"]`
- **Files modified:** crates/cintx-compat/Cargo.toml
- **Committed in:** b1a4f0f

**2. [Rule 1 - Bug] Source-only symbols rejected by active profile check**
- **Found during:** Task 2
- **Issue:** validate_profile_and_source_gate checked source-only entries against base/with-f12/with-4c1e profiles, but unstable entries use "unstable-source" profile
- **Fix:** Added early-return path for source-only symbols that checks "unstable-source" profile directly
- **Files modified:** crates/cintx-compat/src/raw.rs
- **Committed in:** b1a4f0f

**3. [Rule 1 - Bug] DispatchFamily did not recognize unstable family names**
- **Found during:** Task 2
- **Issue:** DispatchDecision::from_manifest_family only matched base families; unstable::source::origi/origk/ssc/grids/breit unrecognized
- **Fix:** Added suffix-matching arms for unstable family names mapping to base dispatch families
- **Files modified:** crates/cintx-runtime/src/dispatch.rs
- **Committed in:** b1a4f0f

**4. [Rule 1 - Bug] Manifest component_rank "1" parsed as multiplier 1 instead of ncomp=3**
- **Found during:** Task 2
- **Issue:** Derivative variant entries had component_rank "1" (tensor rank) but the parser treated it as a literal multiplier, yielding ncomp=1 instead of 3
- **Fix:** Changed component_rank from "1" to "3" for all ip1/ip2 variants; "2" to "9" for ipip variants
- **Files modified:** crates/cintx-ops/src/generated/api_manifest.rs
- **Committed in:** b1a4f0f

---

**Total deviations:** 4 auto-fixed (3 bugs, 1 blocking)
**Impact on plan:** All fixes necessary for correct dispatch of unstable-source symbols through the eval_raw pipeline. No scope creep.

## Deferred Issues
- **ip1_r6_origk parity**: 1 mismatch at shls [3,4,0] with error ~6.6e-7 (atol=1e-12). The ip1_r6 gout has complex multinomial expansion with D_I applied across elevated k-boundary. Needs investigation of G-tensor boundary interaction for high-order k-shifted D_I.

## Issues Encountered
- Manifest entries missing from worktree due to WIP commit divergence; restored from main branch
- G1E_R_I/R_K operations are pointer shifts in libcint, translating to index shifts in flat G-tensor

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Origi (4 symbols) and SSC (1 symbol) fully oracle-verified
- Origk base (3 symbols) and ip1_r2 fully oracle-verified; ip1_r4 passes, ip1_r6 deferred
- Grids and breit remain stubs for Plan 03/04

---
*Phase: 14-unstable-source-api-families*
*Completed: 2026-04-05*

## Self-Check: PASSED
- [x] crates/cintx-cubecl/src/kernels/unstable.rs exists with launch_origi, launch_origk, launch_ssc
- [x] crates/cintx-oracle/tests/unstable_source_parity.rs exists with 11 test functions
- [x] Commit 6dd7078 exists (Task 1)
- [x] Commit b1a4f0f exists (Task 2)
