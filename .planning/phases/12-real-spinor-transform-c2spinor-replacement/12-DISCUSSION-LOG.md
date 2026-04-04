# Phase 12: Real Spinor Transform (c2spinor Replacement) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-04
**Phase:** 12-real-spinor-transform-c2spinor-replacement
**Areas discussed:** Coefficient source strategy, Transform variant differentiation, Oracle testing approach, kappa dispatch logic

---

## Coefficient Source Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Extract from libcint c2spinor.c | Hardcode coefficient tables directly from upstream libcint's g_c2s_* arrays, same approach as c2s.rs. Guarantees oracle parity by construction. | ✓ |
| Compute analytically at build time | Use Wigner 3j / Clebsch-Gordan formula in build.rs to generate coefficients. More principled but introduces a second source of truth vs libcint. | |
| You decide | Claude picks the approach during planning. | |

**User's choice:** Extract from libcint c2spinor.c
**Notes:** None

### Follow-up: File Layout

| Option | Description | Selected |
|--------|-------------|----------|
| Separate c2spinor_coeffs.rs | Keeps large coefficient tables out of transform logic. Mirrors ROADMAP success criterion naming. | ✓ |
| Inline in c2spinor.rs | Single file, simpler module tree. c2s.rs puts coefficients inline at ~300 lines for l=0..4. | |
| You decide | Claude picks based on table size. | |

**User's choice:** Separate c2spinor_coeffs.rs
**Notes:** None

---

## Transform Variant Differentiation

| Option | Description | Selected |
|--------|-------------|----------|
| Match libcint's 4 distinct code paths | Each variant gets own coefficient application logic matching upstream c2spinor.c. ket vs iket differs by conjugation sign; sf vs si differs by CG coupling matrix. | ✓ |
| Shared core with flag dispatch | Single transform function with enum parameter selecting sign flips and matrix choice. Less duplication, same results. | |
| You decide | Claude picks internal factoring. | |

**User's choice:** Match libcint's 4 distinct code paths
**Notes:** None

### Follow-up: Buffer Layout

| Option | Description | Selected |
|--------|-------------|----------|
| Keep interleaved layout | Matches libcint's output format and existing staging contract. Oracle comparison works directly on flat buffer. | ✓ |
| Split to [re...][im...] | Separate real and imaginary arrays. Cleaner but requires layout conversion before oracle comparison. | |
| You decide | Claude picks based on libcint output. | |

**User's choice:** Keep interleaved layout
**Notes:** None

---

## Oracle Testing Approach

| Option | Description | Selected |
|--------|-------------|----------|
| 1e first, then all families | Land 1e spinor oracle parity first (overlap, kinetic, nuclear). Then extend to 2e, 2c2e, 3c1e, 3c2e. Isolates transform correctness from kernel complexity. | ✓ |
| All families in one pass | Implement transform + test all families at once. Faster if correct on first try, harder to debug. | |
| Per-family incremental | Separate oracle gate for each family. Most granular but more CI overhead. | |

**User's choice:** 1e first, then all families
**Notes:** None

### Follow-up: Test File Location

| Option | Description | Selected |
|--------|-------------|----------|
| Expand oracle_gate_closure.rs | Add spinor representation to existing fixture generation loop. Consistent with cart/sph pattern. | ✓ |
| New spinor_oracle.rs test file | Dedicated file for spinor oracle parity. Cleaner separation but diverges from single-gate pattern. | |
| You decide | Claude picks based on test file size. | |

**User's choice:** Expand oracle_gate_closure.rs
**Notes:** None

---

## kappa Dispatch Logic

| Option | Description | Selected |
|--------|-------------|----------|
| kappa selects CG submatrix | kappa determines which rows of CG coupling matrix to apply. kappa<0 → j=l+1/2, kappa>0 → j=l-1/2, kappa=0 → both. Matches libcint exactly. | ✓ |
| kappa selects separate coefficient tables | Maintain separate arrays for j=l+1/2 and j=l-1/2. kappa picks table. More memory but simpler indexing. | |
| You decide | Claude determines routing based on libcint source analysis. | |

**User's choice:** kappa selects CG submatrix
**Notes:** None

### Follow-up: Stub Test Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Delete and replace | Remove amplitude-averaging tests entirely. Replace with value-correctness tests. | ✓ |
| Keep alongside | Rename old tests to *_legacy, add new value-correctness tests alongside. | |

**User's choice:** Delete and replace
**Notes:** None

---

## Claude's Discretion

- Internal factoring of coefficient application loops within each variant
- Oracle fixture molecule/shell choices for spinor tests
- Order of variant implementation within each plan
- Exact plan boundaries between 1e and multi-center family coverage

## Deferred Ideas

None — discussion stayed within phase scope.
