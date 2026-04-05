# Phase 13: F12/STG/YP Kernels - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-05
**Phase:** 13-f12-stg-yp-kernels
**Areas discussed:** zeta=0 validation, Derivative symbols, STG root table embedding, Oracle fixture scope

---

## zeta=0 validation

| Option | Description | Selected |
|--------|-------------|----------|
| Reject with typed error | Validator rejects with InvalidEnvParam before kernel launch. Fail-closed, consistent with UnsupportedApi pattern. | ✓ |
| Produce Coulomb-equivalent | Kernel detects zeta=0, falls back to standard Rys quadrature with tracing warning. | |
| You decide | Claude picks approach fitting existing patterns. | |

**User's choice:** Reject with typed error
**Notes:** Fail-closed approach prevents silent Coulomb fallback that burned libcint users. Validation at plan level, not kernel level.

---

## Derivative symbols

| Option | Description | Selected |
|--------|-------------|----------|
| Separate kernel entry points | Each derivative variant gets its own launch function. Matches Phase 10 pattern. | ✓ |
| Shared base + derivative layer | Base kernel computes primitives, derivative decoration step applies gradient operators. | |
| You decide | Claude picks based on libcint structure and existing dispatch pattern. | |

**User's choice:** Separate kernel entry points
**Notes:** All 10 symbols (5 STG + 5 YP) get individual launch functions in f12.rs.

---

## STG root table embedding

| Option | Description | Selected |
|--------|-------------|----------|
| Static arrays, host-side | Embed DATA_X/DATA_W as static arrays, compute CINTstg_roots on host, upload results. | ✓ |
| Static arrays, device-side | Embed tables and run Clenshaw recurrence inside #[cube] on device. | |
| You decide | Claude picks based on table size and CubeCL constraints. | |

**User's choice:** Static arrays, host-side
**Notes:** Follows established Phase 8 pattern (rys_roots_host). Avoids GPU memory pressure from large tables.

---

## Oracle fixture scope

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse H2O/STO-3G | Same fixtures as existing oracle tests. Consistent env layout. | ✓ |
| Add higher-AM basis | cc-pVDZ to stress-test at higher nroots. | |
| Both | H2O/STO-3G primary + cc-pVDZ supplementary. | |
| You decide | Claude picks best coverage vs complexity balance. | |

**User's choice:** Reuse H2O/STO-3G
**Notes:** Consistent with entire oracle suite. Tests s and p angular momenta.

## Claude's Discretion

- Internal factoring of Clenshaw recurrence helpers within math/stg.rs
- Exact OperatorEnvParams struct layout and naming
- Order of 10 symbol implementations within plans
- Whether STG/YP share common pdata setup before diverging at root computation
- Test molecule zeta value choice

## Deferred Ideas

None — discussion stayed within phase scope.
