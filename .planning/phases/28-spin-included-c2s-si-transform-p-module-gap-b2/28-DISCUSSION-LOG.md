# Phase 28 Discussion Log

**Phase:** 28 — Spin-Included `c2s_si` Transform + σ·p Module (Gap B2)
**Date:** 2026-05-31
**Mode:** discuss (default, interactive)

This log is a human-reference record of the discussion. The canonical decisions
live in `28-CONTEXT.md`.

## Gray Areas Presented

Phase 28 is heavily compatibility-constrained (the σ-coupling must match libcint
`c2s_si_1e` byte-for-byte), so most implementation detail is researcher/planner
territory. Four genuine user-facing decisions were surfaced; the user chose to
discuss all four.

## Area 1 — Phase-28 Flip Scope

- **Question:** Does Phase 28 flip any σ family to `oracle_covered`, or is it transform+assembler infrastructure only?
- **Options:**
  - Flip only the 1 vehicle *(Recommended)* — flip the single validation-vehicle σ family; all others → Phase 29.
  - Infrastructure-only (flip 0) — transform/component-level test only; all σ families flip in Phase 29.
- **Selected:** Flip only the 1 vehicle.
- **Notes:** Gives FND-05 a real registered byte-identity oracle anchor without over-claiming coverage for families whose kernels land in Phase 29. → CONTEXT D-01.

## Area 2 — Validation Vehicle

- **Question:** Which single σ family drives the end-to-end byte-identity proof (SC#3)?
- **Options:**
  - `int1e_sp` *(Recommended)* — σ·p on the bra only; thinnest exercise of both new pieces.
  - `int1e_spsp` — canonical Dirac σ·p·…·σ·p (both sides); heavier first proof.
  - `int1e_sigma` — pure σ, no p; under-tests the σ·p deliverable.
- **Selected:** `int1e_sp`.
- **Notes:** libcint `c2s_si_1e` mixes σ on the BRA (`a_bra_cart2spinor_si` over `gc_x/y/z/1`), ket is ordinary — `int1e_sp` emits exactly those 4 gc blocks. It is the building block spsp/spnucsp/sprinvsp compose from in Phase 29. → CONTEXT D-02.

## Area 3 — σ·p Assembler Design

- **Question:** How much architecture to front-load, and host-vs-device for the Pauli gout?
- **Options:**
  - Reusable module now *(Recommended)* — standalone generic `#[cube]` σ·p emitter for the whole Phase-29 σ-group to reuse.
  - Minimal now, generalize in 29 — narrowest path, refactor later.
- **Selected:** Reusable module now.
- **Notes:** This phase IS the foundation for Groups 4/6/GIAO×σ; front-load it. User did not correct the stated host/device reading → si_2d transform is a HOST fn in `c2spinor.rs` (like `sf_2d`); the σ·p gout that emits `gc_x/y/z/1` is a DEVICE `#[cube]` step (like nabla/gout). → CONTEXT D-03, D-04.

## Area 4 — Kappa Fixture Design

- **Question:** Which fixture gives genuine kappa≠0 spinor shells, and which adversarial properties?
- **Options:**
  - D-08 geometry + kappa≠0 *(Recommended)* — reuse Phase 27's non-square + nctr>1 geometry but with genuine kappa≠0 (GT/LT-only sizing).
  - Real heavy-atom basis — relativistic 2c basis on a heavy element; larger, slower, less surgical.
  - Both: adversarial + 1 heavy — strongest coverage, more fixture work.
- **Selected:** D-08 geometry + kappa≠0.
- **Notes:** Adds the `di = 2l`/`2l+2` GT/LT-only sizing path that B1's kappa=0 fixture structurally could not test, while keeping every prior landmine. Sibling `build_kappa_spinor_fixture` in `fixtures.rs`. → CONTEXT D-05.

## Carried-Forward Conventions (not re-asked)

- New-family surface = manifest + RawApiId + kernel + vendor-FFI + oracle only; no capi/legacy.
- Interleaved `[re,im]` column-major complex staging; sizing from `spinor_len`.
- Vendor parity double-gated (`--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`) + no-silent-skip assertion.
- KET→BRA transpose / nctr>1 coeff-transpose / component_rank-truncation landmines.
- **Design-spike-as-hard-gate** (B1 D-11 precedent + roadmap research flag) → CONTEXT D-06.
- libcint 6.1.3 is byte-authoritative.

## Deferred Ideas

- Group-4 σ families beyond `int1e_sp` → Phase 29.
- `iket_si` 2D + 2e si transforms (`c2s_si_2e*`) → Phases 30/31.
- GIAO×σ slice (Phase 30); gauge/Breit–Gaunt 2e (Phase 31).

## Claude's Discretion

- Fixture element/kappa assignments (within non-square + nctr>1 + kappa≠0 constraints).
- Module naming/factoring; plan boundaries; exact `int1e_sp` gout ordering (resolve in spike).
