# Phase 11: Helper/Transform Completion & 4c1e Real Kernel - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-04
**Phase:** 11-Helper/Transform Completion & 4c1e Real Kernel
**Areas discussed:** Tolerance strategy, 4c1e kernel approach, Oracle coverage gaps, CI gate wiring

---

## Tolerance strategy

| Option | Description | Selected |
|--------|-------------|----------|
| atol=1e-12 for new symbols | New Phase 11 symbols use 1e-12 from the start. Existing per-family constants stay until Phase 15 unifies everything. | |
| Tighten all families now | Phase 11 also tightens existing families (1e, 2c2e, 3c1e) to 1e-12. Front-loads Phase 15 work but may expose kernel bugs. | ✓ |
| Keep per-family approach | Add 4c1e at its own tolerance, transforms/wrappers inherit family tolerance. Phase 15 unifies later. | |

**User's choice:** Tighten all families now
**Notes:** Front-loads Phase 15 unification. If existing kernels fail at 1e-12, they're buggy and must be fixed.

| Option | Description | Selected |
|--------|-------------|----------|
| Float atol=1e-12 for CINTgto_norm | CINTgto_norm is the only float-returning helper. Use atol=1e-12 for it, exact integer equality for all other helpers. | ✓ |
| Exact equality for all helpers | Treat CINTgto_norm as integer too (bit-exact f64). Strict but may fail on FP rounding differences. | |

**User's choice:** Float atol=1e-12 for CINTgto_norm
**Notes:** CINTgto_norm is a floating-point normalization factor; exact bit equality would be too strict.

---

## 4c1e kernel approach

| Option | Description | Selected |
|--------|-------------|----------|
| Adapt existing 2e Rys infrastructure | Reuse existing Rys quadrature, pair data, and VRR infrastructure from 2e/3c kernels with 4-center 1e operator routing. | ✓ |
| Port g4c1e.c directly | Translate upstream g4c1e.c into Rust/CubeCL literally. Creates parallel code path. | |
| You decide | Claude picks during planning. | |

**User's choice:** Adapt existing 2e Rys infrastructure
**Notes:** Keeps code consistent with existing kernel patterns.

| Option | Description | Selected |
|--------|-------------|----------|
| cintx-compat::workaround module | New workaround module in cintx-compat. Clean separation from real kernel. | ✓ |
| Inside center_4c1e.rs as fallback | Workaround lives alongside real kernel as alternative code path. | |
| You decide | Claude decides during planning. | |

**User's choice:** cintx-compat::workaround module
**Notes:** Clean separation between real kernel (cintx-cubecl) and workaround path (cintx-compat).

| Option | Description | Selected |
|--------|-------------|----------|
| Keep current envelope | Cart/sph, scalar, max(l)<=4 stays as-is. Matches 4C1E-01 and 4C1E-03 requirements exactly. | ✓ |
| Tighten to sph-only | Only support int4c1e_sph initially. Reduces testing surface but conflicts with 4C1E-01. | |

**User's choice:** Keep current envelope
**Notes:** Envelope matches requirements exactly. No changes needed.

---

## Oracle coverage gaps

| Option | Description | Selected |
|--------|-------------|----------|
| Manifest-driven gap analysis | Query compiled manifest lock for all helper_kind entries. Diff against IMPLEMENTED_*_SYMBOLS. | ✓ |
| Upstream header scan | Scan libcint's cint_funcs.h and misc.h for public symbols. Cross-reference with manifest. | |
| You decide | Claude picks during planning. | |

**User's choice:** Manifest-driven gap analysis
**Notes:** The manifest is the source of truth for what needs oracle coverage.

| Option | Description | Selected |
|--------|-------------|----------|
| Eval-based comparison | Call each wrapper via eval_raw with test fixtures, compare against vendored libcint at atol=1e-12. | ✓ |
| Wrapper-to-base equivalence | Verify each wrapper produces identical output to its base symbol. | |
| Both approaches | Belt and suspenders — both eval-based and equivalence. | |

**User's choice:** Eval-based comparison
**Notes:** Same approach as base family oracle tests. Directly compares against vendored libcint.

---

## CI gate wiring

| Option | Description | Selected |
|--------|-------------|----------|
| Extend existing gate in-place | Update existing helper_legacy_parity_gate to cover newly oracle-wired symbols. No new CI jobs. | ✓ |
| Add separate comprehensive gate | Keep existing gate, add new full_helper_oracle_gate. | |
| You decide | Claude decides during planning. | |

**User's choice:** Extend existing gate in-place
**Notes:** Gate already loops over four profiles. Just expand symbol coverage.

| Option | Description | Selected |
|--------|-------------|----------|
| Part of existing oracle_parity_gate | 4c1e parity runs inside oracle_parity_gate when with-4c1e profile active. | ✓ |
| Separate 4c1e parity gate | Dedicated CI job for 4c1e oracle parity. | |

**User's choice:** Part of existing oracle_parity_gate
**Notes:** Already profile-gated, no new CI jobs needed.

---

## Claude's Discretion

- Exact Rys quadrature adaptation details for 4c1e (center routing, pair data construction)
- Oracle fixture molecule/shell choices for new symbols
- Internal module organization within the workaround module
- Order of implementation (helpers first vs 4c1e first vs parallel)

## Deferred Ideas

None — discussion stayed within phase scope.
