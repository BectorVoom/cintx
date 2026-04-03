# Phase 10: 2e, 2c2e, 3c1e, 3c2e Real Kernels and Oracle Gate Closure - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-03
**Phase:** 10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure
**Areas discussed:** Kernel architecture, Oracle parity strategy, v1.0 UAT items, Implementation order

---

## Kernel Architecture

### Code Structure

| Option | Description | Selected |
|--------|-------------|----------|
| Mirror libcint sources | Each family gets its own G-fill following its libcint counterpart. Share math primitives but not orchestration. Most faithful to upstream. | ✓ |
| Shared Rys-based G-fill | Factor out common Rys-quadrature G-tensor builder parameterized by center count. Reduces duplication but adds abstraction. | |
| You decide | Claude picks based on oracle parity and code clarity. | |

**User's choice:** Mirror libcint sources
**Notes:** None

### Angular Momentum Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Full angular momentum | Support up to g-function (l=4) from the start for all families. Math primitives already handle arbitrary l. | ✓ |
| Start with s/p, expand later | Implement s/p first for faster validation, then add d/f/g. More incremental but risks rework. | |
| You decide | Claude picks based on complexity and testing needs. | |

**User's choice:** Full angular momentum
**Notes:** None

### Compute Pattern

| Option | Description | Selected |
|--------|-------------|----------|
| Host-side only | Pure Rust host computation using *_host() math wrappers. Same pattern as Phase 9. | ✓ |
| GPU #[cube] kernels | Write real CubeCL kernels for GPU execution. Higher risk due to MLIR limitations. | |
| You decide | Claude picks based on CubeCL readiness. | |

**User's choice:** Host-side only
**Notes:** None

---

## Oracle Parity Strategy

### Verification Method

| Option | Description | Selected |
|--------|-------------|----------|
| Build vendored libcint | Compile libcint-master/ with cc crate, call upstream C via bindgen FFI, compare numerically. Most rigorous. | ✓ |
| Idempotency only | Two eval_raw calls produce same result, like Phase 9. Simpler but doesn't prove upstream compatibility. | |
| Both approaches | Idempotency as smoke test, vendored libcint as real oracle gate. | |

**User's choice:** Build vendored libcint
**Notes:** None

### Test Cases

| Option | Description | Selected |
|--------|-------------|----------|
| H2O cc-pVDZ | Matches success criteria. 3 atoms, s/p/d shells. | |
| H2O STO-3G + cc-pVDZ | Two-tier: fast smoke + full coverage. | |
| Multiple molecules | H2O + H2 + CH4 across STO-3G and cc-pVDZ. More coverage. | ✓ |

**User's choice:** Multiple molecules
**Notes:** None

---

## v1.0 UAT Items

| Option | Description | Selected |
|--------|-------------|----------|
| Automated tests | Integration tests in CI for both items: eval_raw() non-zero + C ABI status == 0. | ✓ |
| Manual verification | Run manually, document results in report. | |
| CI gate + manual GPU | Automated for eval_raw, manual for C ABI on real GPU. | |

**User's choice:** Automated tests
**Notes:** None

---

## Implementation Order

### Family Sequence

| Option | Description | Selected |
|--------|-------------|----------|
| 2c2e → 3c1e → 3c2e → 2e | Ascending complexity. Each builds on prior. | ✓ |
| 2e first, rest follows | Hardest first. If 2e works, simpler families are straightforward. | |
| You decide | Claude picks based on code dependencies. | |

**User's choice:** 2c2e → 3c1e → 3c2e → 2e
**Notes:** None

### Plan Granularity

| Option | Description | Selected |
|--------|-------------|----------|
| One plan per family | 4 kernel plans + 5th for oracle gate closure + UAT items. Oracle parity per family (VERI-05). | ✓ |
| Batch similar families | Group 2c2e+3c2e (Rys-based) and 3c1e separately. Fewer plans. | |
| You decide | Claude determines plan boundaries. | |

**User's choice:** One plan per family
**Notes:** None

---

## Claude's Discretion

- Internal G-tensor array sizing and indexing per family
- Rys quadrature integration details
- Test fixture design for per-family oracle comparison
- Oracle gate closure CI structure
- Plan internal task boundaries

## Deferred Ideas

- GPU-side #[cube] kernels -- future optimization
- Screening/batching -- post-correctness performance work
- Higher angular momentum (l>=5) -- register pressure risk
- Spinor representation kernels -- v1.2
- F12/STG/YP optional families -- v1.2
