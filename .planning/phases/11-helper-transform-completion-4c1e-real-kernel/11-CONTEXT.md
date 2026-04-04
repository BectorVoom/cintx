# Phase 11: Helper/Transform Completion & 4c1e Real Kernel - Context

**Gathered:** 2026-04-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Every helper, transform, and wrapper symbol in the manifest is oracle-wired and returns libcint-compatible values at atol=1e-12. The 4c1e stub is replaced with a real Rys quadrature kernel within the Validated4C1E envelope (cart/sph, scalar, max(l)<=4). Spinor 4c1e is unconditionally UnsupportedApi.

</domain>

<decisions>
## Implementation Decisions

### Tolerance strategy
- **D-01:** Tighten ALL per-family tolerances to atol=1e-12 in this phase, not just new symbols. This front-loads Phase 15's unification work. Replace the per-family constants in `compare.rs` (TOL_1E_ATOL, TOL_2E_ATOL, TOL_2C2E_3C2E_ATOL, TOL_3C1E_ATOL, TOL_4C1E_ATOL) with a single unified constant.
- **D-02:** CINTgto_norm (the only float-returning helper) uses float atol=1e-12. All other helpers (count, offset, norm-count) use exact integer equality. If a family fails at 1e-12, the kernel is buggy and must be fixed — tolerance is immutable.

### 4c1e kernel approach
- **D-03:** Adapt existing 2e Rys quadrature infrastructure (rys_roots, rys_weights, VRR, HRR) for the 4-center 1-electron operator. Do not create a parallel code path; reuse the patterns from `two_electron.rs` and `center_3c1e.rs` with 4-center routing modifications.
- **D-04:** `int4c1e_via_2e_trace` workaround lives in a new `cintx-compat::workaround` module. It calls eval_raw with a 2e symbol, then traces/contracts the result to produce 4c1e-equivalent output. Clean separation from the real kernel in `cintx-cubecl`.
- **D-05:** Validated4C1E envelope stays as-is: cart/sph representation, scalar component_rank, max(l)<=4. Spinor 4c1e returns UnsupportedApi unconditionally — classifier checks representation before angular momentum (v1.2 roadmap decision).

### Oracle coverage gaps
- **D-06:** Use manifest-driven gap analysis to identify missing symbols. Query compiled_manifest.lock.json for all helper_kind entries (helper, transform, legacy-wrapper, optimizer). Diff against IMPLEMENTED_*_SYMBOLS arrays in `compare.rs` to produce an exact gap list.
- **D-07:** Legacy wrappers use eval-based comparison: call each wrapper via eval_raw with test fixtures, compare output buffers against vendored libcint at atol=1e-12. Same approach as base family oracle tests.

### CI gate wiring
- **D-08:** Extend the existing `helper_legacy_parity_gate` from Phase 4 in-place to cover newly oracle-wired symbols. No new CI jobs — expand what the existing gate tests across all four feature profiles.
- **D-09:** 4c1e oracle parity runs inside the existing `oracle_parity_gate` when with-4c1e profile is active. Already profile-gated, no new CI jobs needed.

### Claude's Discretion
- Exact Rys quadrature adaptation details for 4c1e (center routing, pair data construction)
- Oracle fixture molecule/shell choices for new symbols
- Internal module organization within the workaround module
- Order of implementation (helpers first vs 4c1e first vs parallel)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Design & requirements
- `docs/design/cintx_detailed_design.md` — Master design document defining API surface, families, and compatibility contracts
- `.planning/REQUIREMENTS.md` — HELP-01 through HELP-04 and 4C1E-01 through 4C1E-04 requirement definitions

### Oracle & comparison infrastructure
- `crates/cintx-oracle/src/compare.rs` — Per-family tolerance constants (to be unified), IMPLEMENTED_*_SYMBOLS arrays, FamilyTolerance struct, oracle comparison logic
- `crates/cintx-oracle/src/fixtures.rs` — Oracle fixture generation, profile representation matrix, artifact writing
- `crates/cintx-oracle/tests/oracle_gate_closure.rs` — Gate closure test that must pass for all families

### Helper/transform/wrapper implementation
- `crates/cintx-compat/src/helpers.rs` — 17 implemented helper symbols (CINTlen_cart, CINTcgto_*, CINTtot_*, CINTshells_*_offset, CINTgto_norm)
- `crates/cintx-compat/src/transform.rs` — Transform wrappers delegating to c2s/c2spinor
- `crates/cintx-compat/src/legacy.rs` — Macro-generated legacy wrapper functions (all_cint1e_wrappers, etc.)
- `crates/cintx-compat/src/optimizer.rs` — Optimizer lifecycle symbols

### 4c1e kernel
- `crates/cintx-cubecl/src/kernels/center_4c1e.rs` — Current stub with Validated4C1E envelope validation
- `crates/cintx-cubecl/src/kernels/two_electron.rs` — 2e Rys quadrature kernel (reuse base)
- `crates/cintx-cubecl/src/kernels/center_3c1e.rs` — 3c1e kernel (reuse patterns)
- `crates/cintx-cubecl/src/kernels/mod.rs` — Family registry and dispatch

### Manifest & resolver
- `crates/cintx-ops/src/generated/api_manifest.rs` — Compiled manifest with all symbol entries
- `crates/cintx-ops/src/resolver.rs` — HelperKind enum, symbol lookup, manifest queries

### CI
- `.github/workflows/` — Existing CI gates (helper_legacy_parity_gate, oracle_parity_gate) to extend

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `two_electron.rs` Rys quadrature kernel: Full 2e implementation with rys_roots/rys_weights, VRR/HRR, pair data — primary reuse target for 4c1e kernel
- `center_3c1e.rs`: 3-center 1e kernel showing how to adapt multi-center routing for 1e operator
- `compare.rs` oracle infrastructure: FamilyTolerance struct, profile-scoped comparison, artifact generation — extend for new symbols
- `legacy.rs` macro system: `all_cint1e_wrappers!` / `all_cint2e_wrappers!` macros generate cart/sph/spinor wrapper triples — pattern for adding missing wrappers
- `fixtures.rs`: Profile-aware fixture generation already handles four feature profiles

### Established Patterns
- `#[cube(launch)]` kernels with host wrappers for testing (Phase 8 pattern)
- `ensure_validated_*` preflight checks before kernel launch (center_4c1e.rs)
- `IMPLEMENTED_*_SYMBOLS` arrays as oracle coverage tracking
- Eval-based oracle comparison: call eval_raw, compare buffers against vendored libcint

### Integration Points
- `kernels/mod.rs` `resolve_family_name()` / `supports_canonical_family()` — 4c1e already registered, needs real launch fn
- `compare.rs` tolerance constants — replace per-family with unified atol=1e-12
- `compare.rs` IMPLEMENTED_*_SYMBOLS — expand to cover full manifest
- CI workflow YAML — extend existing gate jobs, no new jobs

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 11-helper-transform-completion-4c1e-real-kernel*
*Context gathered: 2026-04-04*
