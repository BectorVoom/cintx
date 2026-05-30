# Phase 23: Group 1 — Remaining 1st-Derivative Families (cart/sph) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-29
**Phase:** 23-group-1-remaining-1st-derivative-families-cart-sph
**Areas discussed:** Family grouping & waves, Angular-momentum coverage, Both-side rank-9 glue, Spinor representation policy

---

## Family grouping & waves

### Q1 — split into PLAN.md files

| Option | Description | Selected |
|--------|-------------|----------|
| By engine reuse (~3 plans) | Cluster by shared kernel: (A) ket/remaining-center rank-3, (B) 3c1e pair, (C) both-side rank-9 1e | ✓ |
| One plan per family (8 plans) | Max granularity; highest parallelism, most scaffold overhead | |
| Two plans (rank-3 vs rank-9) | Split only by the architectural seam | |
| You decide | Defer grouping to the planner | |

**User's choice:** By engine reuse (~3 plans)

### Q2 — execution sequencing

| Option | Description | Selected |
|--------|-------------|----------|
| Rank-9 first (de-risk) | Land the novel both-side glue first | |
| Rank-3 first (momentum) | Land the 5 low-risk rank-3 families first, rank-9 last | ✓ |
| All parallel | Dispatch A/B/C concurrently via worktrees | |
| You decide | Defer to planner | |

**User's choice:** Rank-3 first (momentum)

---

## Angular-momentum coverage

### Q1 (initial) — coverage target

**User's first choice:** "Attempt full f/g" — flagged by Claude as a scope expansion requiring
the deferred nroots≥6 Wheeler fallback. Claude paused to research the Wheeler-fallback todo
and the Rys ceiling before locking.

**Research surfaced:** the nroots≥6 Wheeler/Jacobi fallback is a tracked todo assigned to
**Phase 25** (`resolves_phase: 25`), is a substantial standalone effort (new Rys roots 6–13,
`executor.rs` `l>4` gate raise, lanthanide-ECP validation), and the per-family ceiling is
not uniform: 2-center 1e families reach **f** within nroots≤5, while the 2e/3c/2c group is
capped at **d** (same wall as Phase-21 `ip1`). Full f/g for all 8 is not achievable without
the Phase-25 work.

### Q2 (re-decided with the per-family ceiling)

| Option | Description | Selected |
|--------|-------------|----------|
| Max-within-ceiling per family | f for 2-center 1e (ipovlpip/ipkinip/ipnucip), d for 2e/3c/2c; no Wheeler work | ✓ |
| Pull Wheeler into Phase 23 | Add nroots≥6 + l>4 gate raise; absorbs Phase-25 scope | |
| s/p/d everywhere (Phase-21 parity) | Uniform, simplest; leaves easy 1e f-coverage on the table | |
| Defer the decision to research | Note both ceilings; planner picks | |

**User's choice:** Max-within-ceiling per family
**Notes:** Preserves the "zero new foundations" charter; full f/g + Wheeler stays a Phase-25 dependency.

---

## Both-side rank-9 glue

### Q1 — layout derivation + transpose-hazard validation

| Option | Description | Selected |
|--------|-------------|----------|
| Mirror libcint gout + non-square test | Derive order from `CINTgout1e_int1e_ipovlpip` verbatim; gate with non-square (p×d) block | ✓ |
| Reuse 2e ip1ip2 ordering | Adopt `gout_ip1ip2` convention (different index semantics — may mislead) | |
| Trust planner auto-layout | `component_rank=9` + square-block test (square blocks hide transposes) | |
| You decide | Defer to researcher | |

**User's choice:** Mirror libcint gout + non-square test

---

## Spinor representation policy

### Q1 — spinor reps for the 8 families

| Option | Description | Selected |
|--------|-------------|----------|
| Register-but-UnsupportedApi (Phase 21 D-03) | Register spinor reps; kernels return UnsupportedApi | ✓ |
| Omit spinor reps entirely | cart+sph only; breaks manifest-completeness pattern | |
| You decide | Follow manifest-audit + Phase-21 precedent | |

**User's choice:** Register-but-UnsupportedApi (Phase 21 D-03)

---

## Claude's Discretion

- Exact oracle fixtures / shell-tuple coverage beyond the s/p/d(/f) minimum per family.
- Center-index selection detail for ket-side / remaining-center derivatives (apply the same
  non-square-block validation discipline where the block is rectangular).
- Whether `int3c2e_ip2` needs anything beyond the Phase-21 `int3c2e_ip1` repair as a base.

## Deferred Ideas

- Full f/g coverage for the 2e/3c/2c families → **Phase 25** (nroots≥6 Wheeler fallback +
  `l>4` gate raise + lanthanide-ECP validation; todo `rys-nroots-ge6-wheeler-fallback`).
- `oracle-cart-offset-vendor-zero` — separate `CINTshells_cart_offset[4]` lib-test bug;
  reviewed, not folded (belongs to the oracle/helper-coverage track).
