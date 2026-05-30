# Phase 25: Group 2 — Hessian & Higher-Order Derivatives - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-30
**Phase:** 25-group-2-hessian-higher-order-derivatives
**Areas discussed:** Wheeler nroots≥6 scope (FND-02), FND-06 staging fail-close, Sequencing & clustering, 2e Hessian promotion (HESS-02)

---

## Wheeler nroots≥6 fallback (FND-02) — fidelity approach

| Option | Description | Selected |
|--------|-------------|----------|
| Port libcint's path verbatim | Replicate libcint 6.1.3's actual high-nroots numerical path (modified-moments → tridiagonal/Jacobi → root-polish) literally, host-side, mirroring the ECP K-Taylor host-first precedent. Highest byte-identity confidence. | ✓ |
| Independent Golub-Welsch | Clean-room Jacobi-matrix eigenvalue quadrature; mathematically equivalent but last-ULP / root-ordering risk vs 1e-12 byte-identity. | |
| You decide | Researcher derives the exact scheme and picks for reliability, defaulting to faithful port. | |

**User's choice:** Port libcint's path verbatim (host-side).
**Notes:** Mirrors the Phase-19 ECP "port the exact upstream machinery host-first" precedent. Captured as D-01.

---

## Wheeler nroots≥6 fallback (FND-02) — validation range + executor gate scope

| Option | Description | Selected |
|--------|-------------|----------|
| Validate 6..~13, open l-gate to validated max | General algorithm; dedicated vendor parity sweep nroots 6..~13; extend executor `l>4` gate to the max-l the roots are validated for (g/h/i as covered). Satisfies ROADMAP SC1 as forward-looking foundation. | ✓ |
| Minimal — only what the corpus exercises | Lift the panic just for Hessian-elevated d-shells (~nroots 6–7); leave the l>4 gate + higher validation to a later heavy-element phase. Deviates from SC1. | |
| General algorithm, no fixed cap | Arbitrary-n with no ceiling; validate a representative sweep and open the gate generally above the validated range. | |

**User's choice:** Validate 6..~13, open l-gate to validated max.
**Notes:** Phase-25 corpus (H2O/STO-3G ≤ d) triggers nroots 6 via Hessian elevation but never reaches g/h, so the gate extension is forward-looking foundation validated on the nroots sweep. Captured as D-02/D-03.

---

## FND-06 fail-closed staging — assertion site + guard-replacement breadth

| Option | Description | Selected |
|--------|-------------|----------|
| Planner upfront + strip all guards | One upfront `BufferTooSmall` assertion at the planner staging-alloc boundary, then remove the per-element `if dst < staging.len()` guards across ALL kernels. Single contract point. | ✓ |
| Only the rank≥9 high-rank paths | Upfront assertion but strip guards only in the rank≥9 paths this phase touches; leave rank≤3 kernels' guards. | |
| Per-launcher assertion | Assertion in each kernel launcher rather than the planner; strip guards behind it. | |

**User's choice:** Planner upfront + strip all guards.
**Notes:** Captured as D-04. Anchored to `planner.rs` `parse_component_multiplier` sizing boundary; ~20 guard sites enumerated in CONTEXT canonical refs.

---

## FND-06 fail-closed staging — rank-81 OOM re-validation

| Option | Description | Selected |
|--------|-------------|----------|
| New rank-81 test, tight mem budget | Dedicated test sets a memory limit smaller than rank-81 staging requires; asserts typed OOM/BufferTooSmall stop with NO partial write. Exercises new assertion + ChunkPlanner OOM-safe-stop together. | ✓ |
| Extend existing OOM test | Add a rank-81 case to the existing chunk-planner OOM test. | |
| You decide | Planner picks the cleanest proof of fail-closed + no partial write. | |

**User's choice:** New rank-81 test, tight mem budget.
**Notes:** Captured as D-05. Aligns with the CLAUDE.md "fallible allocation + typed failure + no partial writes" non-negotiable.

---

## Sequencing & plan clustering

| Option | Description | Selected |
|--------|-------------|----------|
| 2 foundation plans, then clustered families | Plan 1 = FND-02, Plan 2 = FND-06 (both merge before families); then low-rank-first clusters A (int1e rank-9) → B (2e Hessian set) → C (2c2e/3c2e ipip) → D (3rd/4th-order), worktree-parallelized. | ✓ |
| 1 combined foundation plan | FND-02 + FND-06 in a single gating plan, then the same A→D clusters. | |
| Foundations interleaved per-cluster | FND-06 first, then pull FND-02 in just ahead of the first Rys-based cluster that needs nroots≥6. | |

**User's choice:** 2 foundation plans, then clustered families.
**Notes:** Captured as D-06. Mirrors Phase-24 D-01 shared-construction clustering. Worktree auto-merge is inconsistent — verify post-wave with `merge-base --is-ancestor`.

---

## 2e Hessian promotion from `unstable` (HESS-02)

| Option | Description | Selected |
|--------|-------------|----------|
| Re-home to stable, drop unstable entries | Move int2e_ipip1/ipvip1 out of `unstable::source::2e` into the stable map (add cart, set rank, wire stable launcher + vendor FFI + test, flip oracle_covered); register ip1ip2/ipip1ipip2 fresh; delete the unstable stubs → one canonical entry per symbol. | ✓ |
| Extend unstable entries in-place | Keep them under `unstable::source::2e`, add cart/rank/oracle_covered there + a stable route alongside. | |
| You decide | Planner chooses move-vs-alias per manifest-audit duplicate handling, defaulting to one canonical stable entry. | |

**User's choice:** Re-home to stable, drop unstable entries.
**Notes:** Captured as D-07. Current manifest has only int2e_ipip1_sph / int2e_ipvip1_sph (sph-only, oracle_covered=false, no rank); ip1ip2/ipip1ipip2 are new.

---

## Claude's Discretion

- Exact `component_rank` (9/27/81) and libcint gout component-index order per family — derive from `deriv3.c`/`deriv4.c`, gate with a non-square block.
- The exact libcint `rys_roots.c` n>5 routine structure and how literally the host port mirrors its control flow (as long as the nroots 6..~13 sweep is byte-identical).
- The complete HESS-04 3rd/4th-order "and siblings" roster — derive from libcint 6.1.3, do not guess.
- Per-`vendor_*`-test corpus shell-tuple selection (subject to the non-square bra×ket rule).
- Whether moment/derivative kernels are one parameterized `#[cube]` entry (comptime deriv order) or order-specialized launchers.

## Deferred Ideas

- Lanthanide / f-projector ECP validation (step 3 of the rys-nroots-ge6 todo) → later heavy-element phase.
- Spinor Hessian representations → registered `UnsupportedApi`; land with Gap B1/B2 (Phases 27/28).
- g/h-basis end-to-end family parity → future heavy-element work (Phase 25 only opens the gate + validates roots).
