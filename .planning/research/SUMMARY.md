# Project Research Summary

**Project:** Rust Crate Test Governance
**Domain:** Rust crate test-governance policy, CI gates, and verification reporting
**Researched:** 2026-03-21
**Confidence:** MEDIUM

## Executive Summary

This project defines a Rust-specific test-governance system that turns a crate’s characteristics into an auditable verification plan. The research converges on a risk-trait classification step that drives tool selection, CI gate tiering (PR/nightly/release), and reporting that explicitly separates verified from unverified scope. Experts build this as policy-as-data plus a deterministic decision pipeline, so governance remains auditable and adaptable without rewriting logic.

The recommended approach is to establish the mandatory baseline toolset, then layer conditional tools based on risk traits (unsafe, concurrency, parsing, compile-time contracts). Separate CI gates prevent expensive verification from collapsing into PR-only checks, while reports must surface residual risks and avoid coverage-only claims. Tool versions should be pinned and governance updates required for tool upgrades to preserve reproducibility.

Key risks are misclassification (skipping required tools), gate collapse (all checks in one tier), and misleading assurance language. Mitigation is explicit classification artifacts, tiered gate policy, and enforced report templates that require verified/unverified scope plus tool limitation notes.

## Key Findings

### Recommended Stack

The stack is centered on stable Rust plus a pinned nightly for verification tooling, with GitHub Actions for CI orchestration and RustSec tooling for dependency governance. The baseline includes `cargo test`, `proptest`, `cargo-mutants`, `cargo-hack`, and `cargo-llvm-cov`, with conditional tools for unsafe (`miri`), concurrency (`loom`), parsers (`cargo-fuzz`), and compile-fail contracts (`trybuild`/`ui_test`). Tool version pinning is required to prevent drift in results.

**Core technologies:**
- Rust toolchain (stable + pinned nightly) — baseline testing plus nightly-only verification like Miri.
- GitHub Actions + rust-toolchain action — CI orchestration with explicit toolchain control.
- RustSec enforcement (`cargo-audit`, `cargo-deny`) — dependency and license policy gates.

### Expected Features

**Must have (table stakes):**
- Crate classification by risk traits — drives governance decisions.
- Mandatory baseline tool enforcement + conditional tool selection — defines verification scope.
- PR/nightly/release gate definitions — operationalizes governance stages.
- Verified vs unverified scope reporting + residual risks — prevents false assurance.

**Should have (competitive):**
- Waiver lifecycle management — avoids permanent risk debt.
- Spec-to-test/tool traceability map — auditability for compliance contexts.
- Anti-fake-implementation checks — elevate mutation/property testing outcomes.

**Defer (v2+):**
- Gate cost modeling and optimization — valuable at scale but not required for v1.
- Automated report integrations — add after workflow stabilizes.

### Architecture Approach

Architecture should treat policy as data, with a deterministic pipeline: classification engine → tool selection → gate planning → report building. Major components are policy assets (baseline and applicability matrix), domain models (traits/tools/gates), analysis logic (classifier/selector/planner), reporting (verified vs unverified scope + spec mapping), and CI integration (workflow templates + artifact publishing).

**Major components:**
1. Policy assets — baseline requirements, applicability matrix, reporting language.
2. Classification and selection — risk traits to tools with rationale.
3. Gate planning + reporting — PR/nightly/release rules and evidence outputs.

### Critical Pitfalls

1. **Misclassifying the crate** — require a signed-off classification artifact that maps traits to required tools.
2. **Collapsing CI gates** — define distinct PR/nightly/release workflows with cost-appropriate checks.
3. **Coverage/passing tests as proof** — enforce verified vs unverified scope and tool limitations in reports.
4. **Mutation testing skew** — require hermetic, deterministic tests and document scope.
5. **Fuzzing without artifacts** — bound runtime and upload crash artifacts in CI.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Policy Baseline + Classification
**Rationale:** All downstream decisions depend on correct crate classification and baseline policy.
**Delivers:** Policy assets, classification rubric, and mandatory baseline toolset.
**Addresses:** Crate classification, baseline enforcement, conditional tool selection.
**Avoids:** Misclassification pitfall.

### Phase 2: Gate Planning + Tooling Integration
**Rationale:** Gate separation and tool configuration are required to operationalize policy.
**Delivers:** PR/nightly/release gate schema, CI workflow templates, pinned tool versions.
**Uses:** GitHub Actions, cargo toolchain, nightly pinning, RustSec tools.
**Implements:** Gate planner + integration components.

### Phase 3: Reporting + Evidence Mapping
**Rationale:** Governance requires explicit verified/unverified reporting and evidence traceability.
**Delivers:** Report templates, verified/unverified scope sections, residual-risk capture.
**Implements:** Reporting pipeline and spec-to-evidence mapping.

### Phase Ordering Rationale

- Classification must precede tool selection, which must precede gate design and reporting.
- Architecture separates policy, analysis, and reporting; phases map to those component boundaries.
- Separating gate design from reporting reduces the risk of coverage-only or “all good” claims.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2:** CI integration details for tools like `cargo-mutants`, `cargo-fuzz`, `miri`, and `loom` (workflow specifics and runtime controls).
- **Phase 3:** Evidence schema and report phrasing requirements for audited environments.

Phases with standard patterns (skip research-phase):
- **Phase 1:** Policy baseline and classification (well-defined in existing guidance and research).

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM | Tooling choices are well-established, but versions and integration details need validation. |
| Features | MEDIUM | Strong alignment with project brief; differentiation features need user validation. |
| Architecture | MEDIUM | Patterns are standard; implementation complexity depends on reporting scope. |
| Pitfalls | MEDIUM | Derived from known tooling limitations; mitigation is clear but requires enforcement. |

**Overall confidence:** MEDIUM

### Gaps to Address

- CI workflow specifics for nightly tools and fuzzing runtime controls — confirm expected runtimes and artifact retention strategy.
- Report schema fidelity — validate required language with stakeholders to avoid compliance gaps.

## Sources

### Primary (HIGH confidence)
- .planning/PROJECT.md — project constraints and baseline requirements.
- test/rust_crate_guideline.md — domain brief and mandatory baseline.

### Secondary (MEDIUM confidence)
- .planning/research/STACK.md — toolchain recommendations and version pinning.
- .planning/research/FEATURES.md — feature expectations and priorities.
- .planning/research/ARCHITECTURE.md — architecture patterns and component boundaries.
- .planning/research/PITFALLS.md — risk analysis and mitigation strategies.

---
*Research completed: 2026-03-21*
*Ready for roadmap: yes*
