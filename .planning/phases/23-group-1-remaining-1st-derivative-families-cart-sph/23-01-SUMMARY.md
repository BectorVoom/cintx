---
phase: 23-group-1-remaining-1st-derivative-families-cart-sph
plan: 01
subsystem: cubecl-kernels
tags: [f12, nabla, gradient, single-side-contraction, int2e_ip2, int2c2e, int3c2e_ip2, G2E_D_L]

# Dependency graph
requires:
  - phase: 21-f12-single-side-gradient-engine
    provides: "f12.rs single-side gradient engine (nabla1i_2e, gout_ip1, F12Shape) reused verbatim"
provides:
  - "pub(crate) nabla1j_2e and nabla1k_2e (promoted from private fn) callable from sibling kernel modules"
  - "new pub(crate) nabla1l_2e (G2E_D_L) for the 2e ll-slot auxiliary-k derivative (int3c2e_ip2, Pitfall 2)"
  - "pub(crate) Nabla1Center{I,J,K,L} enum + gout_ipn: nabla-parameterized single-side contraction"
  - "gout_ip1 refactored to a byte-identical Nabla1Center::I wrapper over gout_ipn"
affects: [23-02, 23-03, two_electron.rs, center_2c2e.rs, center_3c2e.rs]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single source of truth for the s[0..2] single-side mixing body, parameterized over which center the nabla acts on (enum dispatch instead of duplicated gout siblings)"
    - "Public gradient symbols delegate to the parameterized core so prior families (int2e_ip1) stay byte-identical"

key-files:
  created: []
  modified:
    - crates/cintx-cubecl/src/kernels/f12.rs

key-decisions:
  - "Chose plan option (a): generalize gout_ip1 via a Nabla1Center enum + exponent rather than copy-paste gout_ipk/gout_ipl siblings — keeps the s[0..2] mixing body in one place (least duplication)"
  - "gout_ip1 kept its exact public signature and now delegates to gout_ipn(.., Nabla1Center::I, ai) so all existing callers (two_electron.rs, center_3c2e.rs) and int2e_ip1 are byte-unchanged"
  - "nabla1l_2e mirrors nabla1k_2e on the ll loop bound and dl stride, using nabla1l_breit (breit.rs:1206) as the in-tree structural authority for the +dl/-dl offsets (Don't Hand-Roll the G2E_D_L recurrence)"
  - "Unit tests build the G-tensor shape with the matching CEILING angular momentum (e.g. ll_ceil = ll+1) exactly as the real launchers do, so the top base-l-level g[+dl] read stays in bounds"

patterns-established:
  - "Nabla-parameterized contraction: add a derivative family by passing Nabla1Center::{I,J,K,L} + exponent, no new mixing-body copy"

requirements-completed: []

# Metrics
duration: 12min
completed: 2026-05-30
---

# Phase 23 Plan 01: Promote & Parameterize the f12.rs Single-Side Gradient Engine Summary

**Promoted nabla1j/k_2e to pub(crate), added nabla1l_2e (G2E_D_L) for the 2e ll-slot, and parameterized the single-side contraction over Nabla1Center{I,J,K,L} so cluster A's ket/remaining/auxiliary-center derivative families can reuse the Phase-21 engine — int2e_ip1 stays byte-identical.**

## Performance

- **Duration:** ~12 min
- **Tasks:** 2
- **Files modified:** 1 (crates/cintx-cubecl/src/kernels/f12.rs)

## Accomplishments
- `nabla1j_2e` and `nabla1k_2e` promoted from private `fn` to `pub(crate) fn` — sibling launchers (two_electron.rs, center_2c2e.rs, center_3c2e.rs) can now call them (the prior E0603 is gone).
- New `pub(crate) fn nabla1l_2e` implements the G2E_D_L recurrence on the `ll` loop bound and `dl` stride, the structural mirror of `nabla1k_2e` / `nabla1l_breit`. This is required because cintx maps the real 3c2e auxiliary `k` into the 2e `ll` slot (int3c2e_ip2, RESEARCH Pitfall 2) — `nabla1k_2e` would touch the phantom slot.
- New `pub(crate) enum Nabla1Center{I,J,K,L}` + `pub(crate) fn gout_ipn`: the s[0..2] mixing body is now the single source of truth, parameterized over center + exponent.
- `gout_ip1` is now a thin `Nabla1Center::I` wrapper over `gout_ipn`; its signature and numeric output are byte-identical (verified bit-for-bit in a regression test), so Phase-21 `int2e_ip1` does not regress.
- 5 new unit tests added; full `cintx-cubecl --lib` suite green at 263 passed (up from 258).

## Task Commits

Each task was committed atomically:

1. **Task 1: Promote nabla1j_2e/nabla1k_2e and add nabla1l_2e** - `e6f2d45` (feat)
2. **Task 2: Parameterize the single-side contraction over which-nabla** - `a95f490` (feat)

## Files Created/Modified
- `crates/cintx-cubecl/src/kernels/f12.rs` - Promoted nabla1j/k_2e visibility; added nabla1l_2e (G2E_D_L); added Nabla1Center enum + gout_ipn parameterized contraction; gout_ip1 now delegates to gout_ipn(I); added 5 unit tests.

## Decisions Made
- **Parameterization shape:** Chose plan option (a) (enum-dispatched generalization) over option (b) (gout_ipk/gout_ipl siblings) to keep the mixing body in exactly one place.
- **Byte-identity strategy:** gout_ip1 delegates to gout_ipn rather than being rewritten, so the int2e_ip1 path is provably unchanged; a dedicated `to_bits()` regression test enforces it.
- **Test headroom:** Tests construct the F12Shape with the same ceiling angular momentum the real launchers use (ll_ceil = ll+1, etc.) so the top base-level +dl/-dl reads stay in bounds — this surfaced and was fixed during Task 1 (initial tests over-read the l=top slot).

## Deviations from Plan

None - plan executed exactly as written. Both tasks implemented as specified (Task 1 promotion + nabla1l_2e; Task 2 chose the documented option (a) generalization).

## Issues Encountered
- **Initial nabla1l unit tests read out of bounds** (resolved during Task 1, before commit): the first draft built the G-tensor shape at base `ll` and the `g[+dl]` read at the top base l-level overflowed the block. Fixed by building the test shape with the ceiling angular momentum (`ll_ceil = ll+1`) exactly as the production launchers do — this is not a kernel bug, the operator math is correct; the test fixtures simply needed the same headroom the real callers provide. All 3 nabla1l tests then passed.

## Known Stubs
None. nabla1l_2e and gout_ipn are full implementations with passing unit tests. They are unused by production launchers in this plan (plans 02/03 wire them into int2e_ip2 / int2c2e / int3c2e_ip2), which produces an expected `dead_code` *warning* (not error) on the default `cargo build`; the in-crate unit tests reference every new symbol so `cargo test` builds clean.

## Threat Flags
None - this plan edits internal host helpers operating on already-validated, pre-allocated g-tensor buffers (no network, auth, untrusted input, or new env-slot plumbing). The threat-model mitigations T-23-01-01 (nabla1l stride/bounds) and T-23-01-02 (gout_ip1 byte-identity) are both backed by the new unit tests.

## Next Phase Readiness
- f12.rs now exposes i/j/k/l nabla operators and a nabla-parameterized contraction (`gout_ipn`) to sibling launchers — plans 02 and 03 are unblocked.
- Plan 02 (int2e_ip2, int2c2e_ip1/ip2) and plan 03 (int3c2e_ip2 via the ll slot) can call `gout_ipn(.., Nabla1Center::{J,K,L}, exponent)` directly.

---
*Phase: 23-group-1-remaining-1st-derivative-families-cart-sph*
*Completed: 2026-05-30*
