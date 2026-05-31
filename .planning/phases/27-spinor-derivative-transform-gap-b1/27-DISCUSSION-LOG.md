# Phase 27: Spinor-Derivative Transform (Gap B1) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-31
**Phase:** 27-spinor-derivative-transform-gap-b1
**Areas discussed:** Flip scope, Transform API shape, Parity fixture design, Research spike

---

## Flip scope

### Q1 — Which family set should flip oracle_covered=true this phase?

| Option | Description | Selected |
|--------|-------------|----------|
| All arity-2 1e sf | Every spin-free arity-2 1e ip family via sf_2d (ranks 3/9/27/81) | |
| Minimal quartet only | Just rank-3 ipovlp/ipkin/ipnuc/iprinv | |
| Also arity-3/4 (2e/3c2e/2c2e) | Everything spin-free incl. int2e_ip, int3c2e_ip, int2c2e_ip; needs sf_4d + sf_3c2e derivative variants | ✓ |

**User's choice:** Also arity-3/4 — broadest spin-free scope.

### Q2 — Confirm the exclusion boundary

| Option | Description | Selected |
|--------|-------------|----------|
| Exclude ECP only | All 30 spin-free families flip; only int1e_ecp_iprinv_spinor stays UnsupportedApi | |
| Also defer arity-4 2e | Flip 1e + 3c2e + 2c2e + 3c1e now; defer the arity-4 int2e_ip* set (sf_4d variant) | ✓ |
| Let me specify | Free-text include/exclude list | |

**User's choice:** Also defer arity-4 2e.
**Notes:** Final scope = all arity-2 1e sf (sf_2d) + arity-3 (sf_3c2e) + 2c2e. Deferred: arity-4 int2e_ip* (sf_4d) and the R5 ECP family. All in-scope scalar cart forms confirmed oracle_covered=true during discussion.

---

## Transform API shape

### Q1 — How should the derivative transform be structured?

| Option | Description | Selected |
|--------|-------------|----------|
| Thin generic wrapper | cart_to_spinor_sf_derivative_2d/_3c2e loops the verified per-component sf transform N times | ✓ |
| Distinct per-rank functions | Separate _3/_9/_27/_81 functions | |
| You decide | Researcher/planner chooses factoring | |

**User's choice:** Thin generic wrapper.
**Notes:** libcint has no derivative-specific c2s — its int1e driver loops over components calling CINTc2s per component. Distinct per-rank funcs would duplicate identical coupling.

### Q2 — Where should the KET→BRA transpose live?

| Option | Description | Selected |
|--------|-------------|----------|
| Inside the wrapper | Wrapper transposes each per-component device block before sf_2d/sf_3c2e | ✓ |
| Caller transposes | Each launcher transposes per component (the exact pattern that caused the scalar bug) | |
| Defer to spike | Spike decides ownership | |

**User's choice:** Inside the wrapper — centralize the landmine.

---

## Parity fixture design

### Q1 — What geometry should the vendor spinor parity fixture use?

| Option | Description | Selected |
|--------|-------------|----------|
| Non-square + nctr>1 + kappa=0 | Maximally adversarial single fixture; defeats transpose-symmetry, coeff-transpose, and both-blocks sizing at once | ✓ |
| Non-square, definite kappa, nctr=1 | Closer to a physical relativistic shell; skips both-blocks + coeff-transpose checks | |
| You decide | Spike/researcher picks concrete params | |

**User's choice:** Non-square + nctr>1 + kappa=0.

### Q2 — How many families get a dedicated vendor byte-identity test?

| Option | Description | Selected |
|--------|-------------|----------|
| Per-path × per-rank-tier | Dedicated test per transform path and rank tier; rest flip via manifest + no-silent-skip assertion | ✓ |
| Per-operator exhaustive | A test for every flipped family (~24) | |
| Single anchor only | One test on int1e_ipovlp_spinor | |

**User's choice:** Per-path × per-rank-tier.

---

## Research spike

### Q1 — Run a spike before planning?

| Option | Description | Selected |
|--------|-------------|----------|
| Targeted layout probe | Short spike: dump device derivative block layout, confirm stride, then plan | |
| Skip spike, go straight to plan | Let plan-phase research derive layout; adversarial fixture catches errors | |
| Full design spike | Complete 1-day spike across all rank tiers and both transform paths | ✓ |

**User's choice:** Full design spike.
**Notes:** Residual unknown to nail empirically: exact device-emitted derivative cart block layout ([comp][ket][bra] component-outer) and per-component stride into staging, verified against hand-checked vendor values.

---

## Claude's Discretion

- Exact molecule/basis for the fixture (within the non-square + nctr>1 + kappa=0 constraints).
- Internal stride arithmetic and loop factoring within the wrappers.
- Whether int3c1e shares the sf_3c2e derivative wrapper or needs a thin sibling.
- Plan boundaries between the sf_2d-path and sf_3c2e-path families.

## Deferred Ideas

- Arity-4 int2e_ip* spinor families (need a sf_4d derivative variant) — follow-up phase.
- int1e_ecp_iprinv_spinor — R5/ECP-spinor relativistic track (Phase 29), not Gap B1.
