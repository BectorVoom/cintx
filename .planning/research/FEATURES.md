# Feature Research

**Domain:** Rust crate test governance (policy + tooling selection + CI gates)
**Researched:** 2026-03-21
**Confidence:** MEDIUM

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Crate classification by risk traits (unsafe, concurrency, parser exposure, feature flags, invariants) | Governance must be risk-aware and crate-specific | MEDIUM | Drives tool applicability and gate choices |
| Mandatory baseline toolset enforcement with rationale | Rust testing governance implies a minimum verification bar | MEDIUM | Baseline: `cargo test`, `proptest`, `cargo-mutants`, `cargo-hack`, `cargo-llvm-cov`, doctests, compile-fail where applicable |
| Conditional tool selection based on classification | Users expect “why this tool” to be explicit | MEDIUM | E.g., `loom` for concurrency, `Miri` for unsafe, `cargo-fuzz` for parsers |
| CI gate definitions for PR/nightly/release | Governance implies auditable gates by stage | MEDIUM | Gate scope, required tools, and waiver rules differ by stage |
| Verified vs unverified scope reporting | Avoids false assurance | MEDIUM | Explicitly separate verified scope, blocked, waived, and unverified areas |
| Residual risk capture | Required for governance clarity | LOW | Must persist in outputs, not hidden |

### Differentiators (Competitive Advantage)

Features that set the product apart. Not required, but valuable.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Spec-to-test/tool traceability map | Auditable alignment between spec, tests, and gates | HIGH | Links spec items to tools, tests, and gates |
| Waiver lifecycle management (expiry, owner, rationale) | Prevents permanent risk debt | MEDIUM | Includes expiration and revalidation triggers |
| Anti-fake-implementation checks as first-class outcomes | Detects shallow or hardcoded implementations | HIGH | Requires mutation testing, negative tests, and property tests alignment |
| Gate cost modeling (time/resource budgets by stage) | Balances rigor with CI cost | MEDIUM | Explicit PR vs nightly vs release cost tradeoffs |
| Reusable report templates with required phrasing | Reduces audit friction, enforces precise language | LOW | Standardizes "Verified in scope" / "Not yet verified" statements |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Single score or “overall quality” badge | Easy status reporting | Encourages false assurance and hides unverified scope | Verified vs unverified scope report with residual risks |
| Coverage-only gating | Fast pass/fail signal | Coverage is insufficient and can be gamed | Coverage as supporting evidence with mutation/property tests |
| One-size-fits-all toolchain | Simplifies adoption | Ignores crate-specific risk traits | Classification-driven tool applicability matrix |
| “All good” or “fully tested” claims | Appealing certainty | Violates governance integrity | Required phrasing with explicit gaps and risks |

## Feature Dependencies

```
Crate Classification
    └──requires──> Tool Applicability Decision
                       └──requires──> CI Gate Definitions
                                              └──requires──> Verified/Unverified Reporting

Spec Identification ──enhances──> Spec-to-Test Traceability

Waiver Tracking ──requires──> CI Gate Definitions

Anti-Fake-Implementation Checks ──requires──> Mutation + Property Test Baseline
```

### Dependency Notes

- **Crate Classification requires Tool Applicability Decision:** classification drives which tools are mandatory vs conditional.
- **Tool Applicability Decision requires CI Gate Definitions:** gates are built from the selected toolset per stage.
- **CI Gate Definitions require Verified/Unverified Reporting:** reporting must reflect what was gated and what was not.
- **Spec Identification enhances Spec-to-Test Traceability:** traceability only works if a spec source is explicitly named.
- **Waiver Tracking requires CI Gate Definitions:** waivers are defined against a specific gate and stage.
- **Anti-Fake-Implementation Checks require Mutation + Property Test Baseline:** these checks depend on robust test varieties.

## MVP Definition

### Launch With (v1)

Minimum viable product — what's needed to validate the concept.

- [ ] Crate classification by risk traits — core input to all governance decisions
- [ ] Mandatory baseline tool enforcement + conditional tool selection — defines verification scope
- [ ] PR/nightly/release gate definitions — operationalizes governance
- [ ] Verified vs unverified scope reporting + residual risks — prevents false assurance

### Add After Validation (v1.x)

Features to add once core is working.

- [ ] Waiver lifecycle management — add when multiple waivers are accumulating
- [ ] Spec-to-test/tool traceability map — add when audits or compliance pressure appear

### Future Consideration (v2+)

Features to defer until product-market fit is established.

- [ ] Gate cost modeling and optimization — valuable at scale but not required initially
- [ ] Automated report generation integrations — add after workflow stabilizes

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Crate classification by risk traits | HIGH | MEDIUM | P1 |
| Mandatory baseline tool enforcement | HIGH | MEDIUM | P1 |
| Conditional tool selection | HIGH | MEDIUM | P1 |
| CI gate definitions (PR/nightly/release) | HIGH | MEDIUM | P1 |
| Verified vs unverified scope reporting | HIGH | MEDIUM | P1 |
| Residual risk capture | HIGH | LOW | P1 |
| Waiver lifecycle management | MEDIUM | MEDIUM | P2 |
| Spec-to-test/tool traceability | MEDIUM | HIGH | P2 |
| Anti-fake-implementation checks | MEDIUM | HIGH | P2 |
| Gate cost modeling | LOW | MEDIUM | P3 |

## Competitor Feature Analysis

| Feature | Competitor A | Competitor B | Our Approach |
|---------|--------------|--------------|--------------|
| Baseline Rust toolset | Generic QA policies | Rust testing checklists | Governance baseline with applicability rationale |
| Stage-separated CI gates | Ad-hoc CI configs | “All tests on PR” | Explicit PR/nightly/release gate definitions |
| Verified vs unverified reporting | Rare | Rare | Mandatory, structured reporting |

## Sources

- `test/rust_crate_guideline.md`
- `.planning/PROJECT.md`

---
*Feature research for: Rust crate test governance*
*Researched: 2026-03-21*
