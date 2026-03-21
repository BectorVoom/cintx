# Rust Crate Test Governance

## What This Is

Rust Crate Test Governance is a project for turning Rust crate testing requests into operational verification strategies. It defines how engineers or agents classify a crate, select the right verification tools, set CI gates, and report verified versus unverified scope without overstating assurance.

## Core Value

Users can turn a Rust crate testing request into an auditable verification plan that chooses the right tools, sets the right gates, and states residual risk clearly.

## Requirements

### Validated

(None yet -- ship to validate)

### Active

- [ ] Users can classify a Rust crate by API surface, invariants, unsafe usage, concurrency, parser exposure, and feature-flag complexity.
- [ ] Users can derive mandatory and conditional verification tooling from crate traits with explicit rationale for each tool choice.
- [ ] Users can define separate PR, nightly, and release CI gates, including waiver and expiration handling.
- [ ] Users can produce reports that map specification items to tests, tools, gate conditions, and residual risks.
- [ ] Users can distinguish verified scope from unverified scope and block unsupported assurance claims.

### Out of Scope

- Generic non-Rust product QA policy -- this project is specifically for Rust crate verification governance.
- Tool recommendations without applicability rationale -- every tool must be justified against crate characteristics.
- Coverage-only or passing-tests-only quality claims -- these are explicitly insufficient under the governance model.
- Unqualified completeness claims such as "fully tested" or "all good" -- the project requires explicit residual-risk reporting.

## Context

- Source brief: `test/rust_crate_guideline.md`
- The project focuses on operational testing governance for Rust crates, not on implementing a specific application feature set.
- Expected outputs include strategy reviews, tool-selection rationale, CI gate recommendations, testing gap analysis, auditable reports, and concrete policy/CI/template updates.
- The guideline defines a mandatory baseline around `cargo test`, `proptest`, `cargo-mutants`, `cargo-hack`, `cargo-llvm-cov`, doctests, and compile-fail tests when compile-time contracts exist.
- Conditional tooling is required when the crate includes relevant risk classes such as unsafe code, concurrency, hostile-input parsing, stateful workflows, or high-value invariants.

## Constraints

- **Language Scope**: Rust crates only -- the policy and tooling matrix are intentionally Rust-specific.
- **Assurance Model**: Verified versus unverified scope must be stated explicitly -- unsupported confidence claims are not allowed.
- **Tool Selection**: Tooling must follow crate classification -- adding or removing tools requires a stated applicability rationale.
- **CI Structure**: PR, nightly, and release gates must remain separate -- gate purpose and cost differ by stage.
- **Reporting**: Residual risks, waivers, and blocked areas must be preserved in outputs -- silent omissions are not acceptable.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Treat test governance as specification-driven verification rather than generic test advice | The brief requires every recommendation to map to specification items, gates, and residual risk | Pending |
| Require a mandatory Rust testing baseline unless scope is explicitly narrowed | The brief names a default minimum toolset that should not be silently skipped | Pending |
| Gate conditional tools on crate traits instead of enabling everything by default | Tool cost and relevance depend on unsafe code, concurrency, parser surfaces, and similar signals | Pending |
| Separate PR, nightly, and release obligations | The brief requires auditable gate conditions and different assurance levels by workflow stage | Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `$gsd-transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `$gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check -- still the right priority?
3. Audit Out of Scope -- reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-21 after initialization*
