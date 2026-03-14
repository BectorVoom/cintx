# Phase 3: Verification and Compatibility Governance - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning
**Mode:** Auto (`$gsd-discuss-phase 3 --auto`)

<domain>
## Phase Boundary

Phase 3 proves and governs compatibility claims for the existing CPU/stable-family surface. The scope is helper parity, manifest-backed coverage claims, and CI gates that block regressions.

In scope:
- Helper/transform parity required for migration workflows
- Manifest governance for supported API and profile coverage claims
- CI/release gates for manifest drift and oracle/regression failures
- Regression protection for optimizer parity, spinor/layout semantics, and OOM/error paths

Out of scope:
- New execution backends (GPU) and dispatch heuristics
- Optional C ABI shim rollout
- New optional family expansion beyond already-scoped support envelopes

</domain>

<decisions>
## Implementation Decisions

### Helper And Transform Parity Scope
- Phase 3 includes parity for AO counts, shell offsets, normalization, and cart/sph/spinor transform behavior required by migration workflows (COMP-02).
- Helper parity is validated across the Phase 2 stable matrix envelopes so parity evidence aligns with currently supported family/operator/representation rows.
- Helper/transform coverage is tracked as first-class compatibility inventory, not as informal docs-only claims.

### Manifest Coverage Governance
- Compatibility coverage claims are manifest-backed and machine-auditable (COMP-03).
- The compiled-manifest lock is treated as the source of truth for release coverage assertions.
- Support profile set is fixed to `{base, with-f12, with-4c1e, with-f12+with-4c1e}` for lock generation and CI checks; GTG remains out of scope for this governance phase.
- Any symbol/profile drift requires an explicit lock update flow instead of silent acceptance.

### CI Gate Policy For Compatibility Claims
- CI gates are blocking for manifest-lock drift, oracle-regression failures, and helper parity failures (VERI-01, VERI-02).
- PR checks should catch drift/regression early; release checks enforce full profile matrix confidence before claims are published.
- Compatibility claims are allowed only when gate evidence is green across the required profile set.

### Regression Guarantees Beyond Numeric Oracle
- RAW-04 optimizer on/off numerical equivalence is enforced as an explicit regression gate, not a one-time test.
- VERI-03 layout and failure semantics protection must include spinor/complex layout invariants and deterministic OOM/error behavior.
- Existing no-partial-write and typed-error contracts from prior phases are treated as non-regression constraints during Phase 3 gate design.

### Claude's Discretion
- Exact module/file layout for manifest tooling and generated artifacts
- CI job naming, matrix decomposition, and caching strategy
- Precise partitioning between fast PR checks and exhaustive release/nightly compatibility checks
- Tolerance-table implementation details, provided requirement-level acceptance criteria remain satisfied

</decisions>

<specifics>
## Specific Ideas

- Keep the current stable-family matrix (including explicit `3c1e` spinor handling) as the base compatibility domain for parity/gate automation.
- Tie helper parity evidence to the same matrix/test-fixture model already used by phase-2 oracle and safe/raw equivalence checks.
- Treat compatibility governance as release-contract infrastructure: claims must be reproducible from lock + CI artifacts, not narrative docs.
- Prefer deterministic, profile-aware audits over ad-hoc symbol counting scripts.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tests/common/phase2_fixtures.rs`: Stable matrix cases and shared fixtures suitable for phase-3 helper/optimizer parity expansion.
- `tests/common/oracle_runner.rs`: Existing oracle tolerance utilities and comparison helpers for regression gates.
- `docs/phase2-support-matrix.md`: Baseline support-envelope document to extend into phase-3 governance evidence.
- `src/runtime/backend/cpu/router.rs`: Canonical route envelope mapping for supported/out-of-phase behavior checks.
- `src/runtime/raw/{query,evaluate,validator,views}.rs`: Established raw contract and failure-semantics boundary to protect via phase-3 regressions.
- `src/diagnostics/report.rs`: Existing diagnostics contract hooks useful for failure-evidence assertions.

### Established Patterns
- Shared fixtures + table-driven matrix tests are already the dominant validation style for compatibility behavior.
- Safe/raw parity and oracle comparisons already use deterministic helper utilities, making them natural anchors for phase-3 gates.
- Typed error variants and explicit diagnostics fields are already enforced in tests and should remain gate-level invariants.

### Integration Points
- Add phase-3 governance tests under `tests/phase3_*.rs` reusing `tests/common/*` helpers.
- Extend docs in `docs/` with manifest/governance specification tied to requirement IDs (COMP-02/03/04, RAW-04, VERI-01/02/03).
- Introduce manifest audit artifacts and lock checks in repository tooling, then wire CI under `.github/workflows/` for blocking gates.
- Maintain alignment between support-matrix docs, route envelope logic, and CI profile matrix inputs.

</code_context>

<deferred>
## Deferred Ideas

- GPU backend dispatch and fallback observability gates (Phase 4: EXEC-02/EXEC-03)
- Optional C ABI shim coverage and migration verification (Phase 4: ABIC-01)
- Expansion of optional-family envelopes beyond current support matrix (Phase 4: OPTF-01/OPTF-02 and later roadmap items)
- Any GTG enablement work (explicitly roadmap-only, out of current support matrix governance)

</deferred>

---

*Phase: 03-verification-and-compatibility-governance*
*Context gathered: 2026-03-14*
