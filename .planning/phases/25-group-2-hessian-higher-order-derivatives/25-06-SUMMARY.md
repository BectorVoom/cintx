---
phase: 25-group-2-hessian-higher-order-derivatives
plan: 06
subsystem: kernels
tags: [libcint, hessian, deriv3, deriv4, 3rd-order, 4th-order, rank-27, rank-81, dual-headroom, 1e, host-rys, oracle, vendor-parity]

requires:
  - phase: 25-01
    provides: Rys nroots>=6 host engine (FND-02) — the bra/ket +2/+3 headroom can elevate the nuclear Rys nroots beyond MAX_DEVICE_NROOTS=5; rys_roots_host serves 6..12
  - phase: 25-02
    provides: fail-closed high-rank staging (FND-06) — single upfront BufferTooSmall assertion, unconditional rank-27/81 scatter (no per-element dst<len guard)
  - phase: 23
    provides: first-order nabla 1e engine (G1E_D_I/D_J recurrence) composed 3x/4x; the host nuclear Rys G-tensor (contract_nuclear VRR+HRR) reused as the deriv34 base
provides:
  - int1e_ipipipnuc / int1e_ipipiprinv (deriv3, rank 27, bra ∇∇∇) registered + byte-identical to vendor libcint 6.1.3
  - int1e_ipipnucip / int1e_ipiprinvip (deriv3, rank 27, bra ∇∇ + ket ∇) registered + byte-identical
  - int1e_ipipipiprinv (deriv4, rank 81, bra ∇∇∇∇) registered + byte-identical
  - int1e_ipiprinvipip (deriv4, rank 81, ket ∇∇ + bra ∇∇ — the dual bra+2/ket+2 headroom anchor) + byte-identical
  - int1e_ipipiprinvip (deriv4, rank 81, bra ∇∇∇ + ket ∇) + byte-identical
  - deriv3.c + deriv4.c added to the oracle cc::Build (ROADMAP SC7); all 7 families cart+sph at atol=1e-12 on a NON-SQUARE p×d block; oracle_covered=true; manifest-audit green
affects: [hess, deriv]

tech-stack:
  added: []
  patterns:
    - "A per-family op-sequence engine (deriv34.rs): each 3rd/4th-order family = the first-order D_I (bra) / D_J (ket) nabla recurrence applied 3x/4x on the nuclear/rinv Rys G-tensor, with the family's verbatim (i_off,j_off) op targets + s-table + gout permutation read 1:1 from deriv3.c/deriv4.c. One generic Rust function covers all 7 families (data-driven by FamilySpec)."
    - "HOST-routed via rys_roots_host (FND-02) since the bra/ket +2/+3 headroom raises nmax = (li+max_i_off+1)+(lj+max_j_off+1), elevating the nuclear Rys nroots beyond the device MAX_DEVICE_NROOTS=5 cap; fail-closed at HOST_RYS_NROOTS_CEILING_1E=12."
    - "deriv3.c/deriv4.c cart+sph symbols were already declared in cint_funcs.h (the suppl-header #include), so only the .file() additions + allowlist were needed — NO new suppl-header extern decls (unlike the unstable-source families)."

key-files:
  created:
    - crates/cintx-cubecl/src/kernels/deriv34.rs
    - crates/cintx-oracle/tests/deriv34_parity.rs
  modified:
    - crates/cintx-cubecl/src/kernels/mod.rs
    - crates/cintx-cubecl/src/kernels/one_electron.rs
    - crates/cintx-compat/src/raw.rs
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-oracle/build.rs
    - crates/cintx-oracle/src/vendor_ffi.rs

key-decisions:
  - "Per-op (i_off,j_off) targets are load-bearing, NOT a single per-family headroom constant. The first determinism attempt used one global i_inc/j_inc applied to every op and over-read the source buffer (index-out-of-bounds on ipipnucip's D_J). Each G2E_D_*(g_dst, g_src, i_l+N, j_l+M) op fills exactly i in 0..=li+N, j in 0..=lj+M and reads the source one level above; g0 headroom is derived as li+max_i_off+1 / lj+max_j_off+1 from the op list."
  - "ipipipnuc == ipipiprinv and ipipnucip == ipiprinvip share IDENTICAL op sequences, s-tables, and gout permutations — the nuc/rinv distinction is ONLY the Coulomb-center list (nuclear sums over all atoms with -Z_C; rinv is a single origin +1). One FamilySpec per (op-sequence, gout) pair, dispatched by op_name; the origin list is the only branch."
  - "All four deriv3 families share one 27-entry s-table; all three deriv4 families share one 81-entry s-table. Only the op sequence (which D_I/D_J build g1..g15) and the gout permutation differ per family — copied verbatim."
  - "Spinor reps registered (manifest entries exist) but launcher returns UnsupportedApi (D-11); cart+sph oracle_covered=true, spinor oracle_covered=false. No capi enum variants, no legacy cint* wrappers."

patterns-established:
  - "Pattern: a data-driven FamilySpec { rank, nbuf, ops: &[Op{DI/DJ, src, dst, i_off, j_off}], s_table, gout_perm } makes adding any further deriv-N 1e family a table edit, not new control flow."
  - "Pattern: parameterized NCOMP (27/81) vendor parity collectors over a shared NON-SQUARE p×d fixture — one deriv34_ipipip harness covers ranks 27 AND 81."

requirements-completed: [HESS-04]

duration: 38min
completed: 2026-05-31
---

# Phase 25 Plan 06: HESS-04 3rd/4th-order derivative families Summary

**The complete HESS-04 3rd-order (`deriv3.c`, rank 27) and 4th-order (`deriv4.c`, rank 81) 1e derivative roster — int1e_ipipipnuc, int1e_ipipiprinv, int1e_ipipnucip, int1e_ipiprinvip (27) and int1e_ipipipiprinv, int1e_ipiprinvipip, int1e_ipipiprinvip (81) — registered and byte-identical to vendor libcint 6.1.3 at atol=1e-12 (cart+sph, every component) via a new host f64 per-family op-sequence engine that composes the first-order D_I/D_J nabla recurrence on the nuclear/rinv Rys G-tensor, HOST-routed (FND-02) and gated on a NON-SQUARE p×d block.**

## Performance

- **Duration:** ~38 min
- **Completed:** 2026-05-31
- **Tasks:** 4 (Task 0 roster lock + build wiring, Task 1 RED parity scaffold, Task 2 deriv3 register+parity, Task 3 deriv4 register+parity+audit)
- **Files:** 8 (2 created, 6 modified)

## Locked Roster (Task 0 — grepped from libcint source, NOT guessed)

Authoritative roster from `grep -oE "int1e_[a-z0-9]+" libcint-master/src/autocode/deriv3.c deriv4.c | grep -E "ipipip|ipipipip"` cross-referenced against `include/cint_funcs.h` (deduped across `_cart`/`_sph`/`_spinor`/`_optimizer`):

**3rd-order — `deriv3.c`, component_rank 27** (per-family headroom annotated):
| Family | Derivative | Op chain (verbatim) | Headroom |
|--------|-----------|---------------------|----------|
| `int1e_ipipipnuc` | `<∇∇∇ i \| NUC \| j>` | D_I×7 (bra) | bra+3 |
| `int1e_ipipiprinv` | `<∇∇∇ i \| RINV \| j>` | D_I×7 (bra) | bra+3 |
| `int1e_ipipnucip` | `<∇∇ i \| NUC \| ∇ j>` | D_J(bra+2), then D_I×6 | bra+2 / ket+1 |
| `int1e_ipiprinvip` | `<∇∇ i \| RINV \| ∇ j>` | D_J(bra+2), then D_I×6 | bra+2 / ket+1 |

**4th-order — `deriv4.c`, component_rank 81** (dual headroom):
| Family | Derivative | Op chain (verbatim) | Headroom |
|--------|-----------|---------------------|----------|
| `int1e_ipipipiprinv` | `<∇∇∇∇ i \| RINV \| j>` | D_I×15 (first op bra+3) | bra+4 (`i_off` max 3 +1) |
| `int1e_ipiprinvipip` | `<∇∇ i \| RINV \| ∇∇ j>` | D_J×3 (one ket+1), then D_I×12 | **bra+2 AND ket+2** (D-09 dual headroom) |
| `int1e_ipipiprinvip` | `<∇∇∇ i \| RINV \| ∇ j>` | D_J(bra+3), then D_I×14 | bra+3 / ket+1 |

Spinor variants exist in `cint_funcs.h` → registered (manifest) but `UnsupportedApi` (D-11).

## Task Commits

1. **Task 0: roster lock + deriv3.c/deriv4.c oracle build wiring** — `ffe3509` (feat)
2. **Task 1: deriv34_ipipip RED parity scaffold (NON-SQUARE p×d, ranks 27/81)** — `1849a5b` (test)
3. **Task 2: register + implement deriv3 (rank-27) roster, parity green** — `d006b23` (feat)
4. **Task 3: register deriv4 (rank-81) roster, dual headroom, parity green, audit ok** — `0c35993` (feat)

## Accomplishments
- `deriv3.c` + `deriv4.c` added to the oracle `cc::Build` `.file()` list + `rerun-if-changed` (they were NOT previously in the build; ROADMAP SC7); all cart/sph symbols already in `cint_funcs.h` so no suppl-header extern decls needed.
- New `crates/cintx-cubecl/src/kernels/deriv34.rs`: a data-driven per-family op-sequence engine (FamilySpec + Op{DI/DJ, src, dst, i_off, j_off}) covering all 7 families; host f64, `rys_roots_host` for arbitrary nroots.
- Launcher dispatch in `one_electron.rs` (`is_deriv34`): rank-27/81 cart+sph staging scatter, fail-closed at `HOST_RYS_NROOTS_CEILING_1E=12`, spinor → `UnsupportedApi`.
- 7 families registered: 21 manifest entries (cart+sph+spinor; cart+sph component_rank 27/81 + oracle_covered=true), 21 RawApiId consts, 16 build.rs allowlist symbols (cart+sph), 14 vendor_ffi wrappers.
- Vendor parity GREEN: 14/14 tests (7 families × determinism + parity), cart+sph, atol=1e-12, every one of the 27/81 components, NON-SQUARE p×d (bra l=1 != ket l=2). `manifest-audit` status ok, 0 uncovered stable entries.

## Decisions Made
- Per-op `(i_off, j_off)` targets are load-bearing (see Deviations §1): each `G2E_D_*` op fills exactly its `i_l+N`/`j_l+M` range; g0 headroom is `li+max_i_off+1` / `lj+max_j_off+1`.
- The nuc/rinv distinction within a (op-sequence, gout) pair is ONLY the Coulomb-center origin list — one FamilySpec serves both, dispatched by `op_name`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Per-family single-headroom over-read on the mixed (D_J) families**
- **Found during:** Task 2 (ipipnucip / ipiprinvip determinism — index-out-of-bounds panic at deriv34.rs, `the len is 28 but the index is 28`)
- **Issue:** The first deriv34.rs design carried one global `i_inc`/`j_inc` per family and applied that same range to every op. The `D_J` op on ipipnucip then read the source at `j+1` beyond the g0 ket headroom (`(lj+2)*dj` into a buffer sized for `lj+1`). The all-D_I families (ipipipnuc/ipipiprinv) were immune because their ops never raised ket headroom.
- **Fix:** Encode each op's exact `(i_off, j_off)` target verbatim from its `G2E_D_*(…, i_l+N, j_l+M)` call; fill `i in 0..=li+i_off`, `j in 0..=lj+j_off`; derive g0 headroom as `li+max_i_off+1` / `lj+max_j_off+1` from the op list.
- **Files modified:** crates/cintx-cubecl/src/kernels/deriv34.rs
- **Committed in:** `d006b23` (Task 2 commit)

**Total deviations:** 1 auto-fixed (bug). All 7 families matched vendor on the first parity once the per-op targets were correct (the dual-headroom deriv4 family `ipiprinvipip` passed on the NON-SQUARE p×d block immediately — a ket-headroom miss would have failed there).
**Impact on plan:** Required for correctness; no scope creep.

## Note on Acceptance Criterion AC2 grep (Tasks 2/3)
The plan's AC2 grep (`grep -A4 'int1e_ipipipnuc' … | grep -c '"27"'`) returns 0 because the manifest lock stores `component_rank` BEFORE the `id`/`symbol` block, not within 4 lines after it (same lock layout noted in 25-05). The ranks ARE correct (verified directly: `int1e_ipipipnuc_{cart,sph}` rank=27 oracle_covered=true; `int1e_ipipipiprinv_{cart,sph}` rank=81 oracle_covered=true) and are end-to-end gated by the byte-identity parity (a too-low rank would silently truncate trailing components and fail the 27/81-component non-square parity). Only the grep target line-window was mis-specified.

## Known Stubs
None — all 7 cart+sph families are fully wired and vendor-parity green. Spinor entries are registered → `UnsupportedApi` (D-11; spinor Hessian transforms land in Phases 27/28 per CONTEXT deferred scope).

## Threat Flags
None — the new surface (operator-name dispatch on the `ipipip*`/`ipipipip*` families in one_electron.rs + the deriv34.rs engine) is numerical/component-correctness, fully covered by the threat register: T-25-20 (roster completeness — locked by grep, recorded above), T-25-21 (rank truncation — component_rank=27/81, full-component non-square parity green), T-25-22 (dual-headroom miss — `ipiprinvipip` ket+2 gated by the NON-SQUARE bra(p) != ket(d) block), T-25-23 (missing build source — deriv3.c/deriv4.c added + oracle compiles), T-25-24 (silent skip — double-gated parity, 14 tests N>0), T-25-25 (anchor-tuple reuse — each mixed sibling uses its OWN op-target sequence), T-25-26 (context exhaustion — deriv3/deriv4 split across Task 2/Task 3). All mitigated.

## Next Phase Readiness
- Cluster D (HESS-04) complete — the FINAL Wave-2 family plan. The four shared Wave-2 files (manifest lock, raw.rs, build.rs, vendor_ffi.rs) were appended additively.
- The data-driven `deriv34.rs` FamilySpec engine makes any further deriv-N 1e family a table edit.
- Worktree integration: N/A (sequential executor on the main working tree; the D-06 `merge-base --is-ancestor` post-wave check applies only to worktree-parallelized clusters — this plan ran sequentially on `fix/general-contraction-nctr-1e`).

## Self-Check: PENDING
