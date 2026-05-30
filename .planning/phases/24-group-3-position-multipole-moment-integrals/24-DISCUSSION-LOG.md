# Phase 24: Group 3 — Position / Multipole-Moment Integrals - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-30
**Phase:** 24-group-3-position-multipole-moment-integrals
**Areas discussed:** Family clustering & sequencing, _origj variant mechanism, Rank-81 staging (24↔25 boundary), rinv/drinv/p4/irp grouping

---

## Family clustering & sequencing

| Option | Description | Selected |
|--------|-------------|----------|
| By operator construction, low-rank first | Cluster A overlap-derived tensors (one parameterized kernel) / B rinv group / C p4 / D irp; A first | ✓ |
| By multipole rank | dipole → quadrupole → octupole → hexadecapole → misc; groups by component_rank not kernel reuse | |
| You decide / planner clusters | Leave clustering to the planner | |

**User's choice:** By operator construction, low-rank first.
**Notes:** Mirrors Phase 23's shared-kernel-reuse clustering. `_origj` variants land alongside their base in Cluster A (same kernel, origin-source branch).

---

## _origj variant mechanism

| Option | Description | Selected |
|--------|-------------|----------|
| Separate operator names + origin-source flag | Each _origj is its own manifest operator/RawApiId; shared kernel branches env[PTR_COMMON_ORIG] vs ket-center coord | ✓ |
| Single operator + origin-mode descriptor flag | One operator per base + origin-mode flag; fewer entries but diverges from libcint symbol surface | |
| You decide | Defer mechanism | |

**User's choice:** Separate operator names + origin-source flag.
**Notes:** Realizes Phase 22 D-04's "kernel-side coordinate choice"; keeps manifest parity-complete and preserves per-symbol vendor_* test pattern.

---

## Rank-81 staging (Phase 24 ↔ Phase 25 boundary)

| Option | Description | Selected |
|--------|-------------|----------|
| Keep the boundary (existing staging as-is) | parse_component_multiplier sizes any rank; guards never trip when sized right; FND-06 stays Phase 25 | ✓ |
| Pull FND-06 forward into Phase 24 | Land the fail-closed BufferTooSmall assertion + rank-81 OOM test here | |
| You decide | Defer | |

**User's choice:** Keep the boundary.
**Notes:** Phase 24's gate is byte-identity parity, not OOM-safety; output is complete & correct at rank 81 with existing staging. Cross-link the FND-06 dependency in the plan.

---

## rinv / drinv / p4 / irp grouping

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse existing kernels; fail-closed > nroots5 | rinv = int1e_nuc Rys at common origin, charge=1, no sum; drinv = +1 root; p4/irp = overlap-derivative engine; fail-closed nroots>5 | ✓ |
| Dedicated cluster + research spike | Treat as distinct cluster needing its own spike | |
| You decide | Defer kernel-reuse decisions | |

**User's choice:** Reuse existing kernels; fail-closed > nroots5.
**Notes:** Cover up to the nroots≤5 ceiling on the corpus (which does not reach it); Phase 23 D-13 fail-closed precedent. Folded nroots≥6 todo is the cross-link.

---

## Todo folding

Both matched todos folded into Phase 24:
- `oracle-cart-offset-vendor-zero` — vendor-gate lib-test failure that will re-surface when Phase 24 runs the gate; confirm/fix or convert to tracked harness bug so the gate isn't blocked by pre-existing noise.
- `rys-nroots-ge6-wheeler-fallback` — folded as a cross-link/boundary marker only (resolves_phase:25); Phase 24 fail-closes above the ceiling and does not implement the fallback.

## Claude's Discretion

- Exact `component_rank` value + libcint gout component order per family (derived from libcint source by researcher/planner, gated by the non-square-block discipline).
- Whether Cluster A's moment kernel is one comptime-parameterized `#[cube]` entry or a small family of order-specialized launchers.
- Per-test corpus shell-tuple selection (subject to the non-square bra×ket requirement).

## Deferred Ideas

- FND-06 fail-closed high-rank (rank-81) staging refactor + OOM re-validation → Phase 25.
- nroots≥6 Wheeler/Jacobi fallback (FND-02) → Phase 25.
- Spinor moment representations → land when a consumer needs them (registered → UnsupportedApi this phase).
