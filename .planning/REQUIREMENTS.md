# Requirements: Rust Crate Test Governance

**Defined:** 2026-03-21
**Core Value:** Users can turn a Rust crate testing request into an auditable verification plan that chooses the right tools, sets the right gates, and states residual risk clearly.

## v1 Requirements

### Scoping

- [ ] **SCOP-01**: User can identify the Rust crate or module that is in scope for verification.
- [ ] **SCOP-02**: User can name the governing specification sources, public APIs, invariants, error contracts, feature flags, and side effects that define the review scope.

### Classification

- [ ] **CLAS-01**: User can classify whether the target crate includes public API docs or doctests, compile-time usage constraints, stateful workflows, unsafe code, concurrency or atomics, hostile-input parsing, multiple feature flags, and high-value invariants.
- [ ] **CLAS-02**: User can record why each detected trait changes the required verification tools or CI gates.

### Tool Selection

- [ ] **TOOL-01**: User can apply the mandatory baseline toolset (`cargo test`, `proptest`, `cargo-mutants`, `cargo-hack`, `cargo-llvm-cov`, doctests, and compile-fail tests where applicable) unless scope is explicitly narrowed.
- [ ] **TOOL-02**: User can require conditional tools (`proptest-state-machine`, `Kani`, `Miri`, `loom`, and `cargo-fuzz`) when the target crate traits make them applicable.
- [ ] **TOOL-03**: User can produce a tool-selection decision with explicit rationale, including why non-applicable tools were excluded.
- [ ] **TOOL-04**: User can define checks that expose fake or shallow implementations instead of accepting narrow happy-path tests.

### CI Gates

- [ ] **GATE-01**: User can define explicit PR CI gate conditions, including what must pass, what is blocked, and what can be waived.
- [ ] **GATE-02**: User can define explicit nightly CI gate conditions for heavier or slower verification work.
- [ ] **GATE-03**: User can define explicit release gate conditions that must pass before strong quality or compatibility claims are published.
- [ ] **GATE-04**: User can document waiver owner, rationale, expiry, and revalidation trigger for any blocked or deferred gate.

### Reporting

- [ ] **REPT-01**: User can produce an auditable report that maps each recommendation or change to a specification item, test or tool, gate condition, and residual risk.
- [ ] **REPT-02**: User can state verified scope, not yet verified scope, blocked items, and waiver status without making unqualified assurance claims.
- [ ] **REPT-03**: User can deliver a testing gap analysis that distinguishes verified scope from unverified scope and names residual risks.

### Repository Changes

- [ ] **CHNG-01**: User can request concrete repository updates and receive consistent policy, CI, and template changes aligned with the governance rules.

## v2 Requirements

### Future Extensions

- **AUTO-01**: User can generate machine-readable evidence bundles and publish them automatically to downstream audit systems.
- **COST-01**: User can model PR, nightly, and release gate cost envelopes before enabling expensive verification lanes.
- **BENCH-01**: User can compare the governance profile against external frameworks or competitor policies with cited evidence.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Non-Rust crate or application QA governance | This project is specifically for Rust crate verification policy and tool selection |
| Single score or overall quality badge | Encourages false assurance and hides unverified scope |
| Coverage-only pass/fail decisions | Coverage is supporting evidence, not proof of correctness or conformance |
| One-size-fits-all toolchain with no applicability rationale | Ignores crate-specific risk traits and breaks governance intent |
| Unqualified claims such as "fully tested" or "all good" | The project requires explicit verified scope, unverified scope, and residual risk reporting |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| SCOP-01 | TBD | Pending |
| SCOP-02 | TBD | Pending |
| CLAS-01 | TBD | Pending |
| CLAS-02 | TBD | Pending |
| TOOL-01 | TBD | Pending |
| TOOL-02 | TBD | Pending |
| TOOL-03 | TBD | Pending |
| TOOL-04 | TBD | Pending |
| GATE-01 | TBD | Pending |
| GATE-02 | TBD | Pending |
| GATE-03 | TBD | Pending |
| GATE-04 | TBD | Pending |
| REPT-01 | TBD | Pending |
| REPT-02 | TBD | Pending |
| REPT-03 | TBD | Pending |
| CHNG-01 | TBD | Pending |

**Coverage:**
- v1 requirements: 16 total
- Mapped to phases: 0
- Unmapped: 16 WARNING

---
*Requirements defined: 2026-03-21*
*Last updated: 2026-03-21 after initial definition*
