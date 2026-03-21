# Architecture Research

**Domain:** Rust crate test-governance system
**Researched:** 2026-03-21
**Confidence:** MEDIUM

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 Policy & Governance Layer                                │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌────────────────────┐  ┌──────────────────────────┐ │
│  │ Policy Assets│  │ Tool Applicability │  │ CI Gate Definitions      │ │
│  │ (guidelines, │  │ Matrix & Rules     │  │ (PR/Nightly/Release)     │ │
│  │ templates)   │  └─────────┬──────────┘  └──────────┬───────────────┘ │
│  └──────┬───────┘            │                       │                  │
├─────────┴────────────────────┴───────────────────────┴──────────────────┤
│                 Analysis & Orchestration Layer                            │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ Classification Engine → Tool Selection → Gate Plan → Report Builder │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────┤
│                 Execution & Evidence Layer                                │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌─────────────────────────┐ │
│  │ CI Jobs  │  │ Test Tools│  │ Evidence │  │ Artifact Store (reports)│ │
│  │ (GitHub/ │  │ (cargo*,  │  │ Collector│  │                         │ │
│  │ CI)      │  │ miri, etc)│  │           │  │                         │ │
│  └──────────┘  └───────────┘  └──────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| Policy assets | Define mandatory baseline, conditional tools, reporting language, and prohibited claims | Markdown policy docs, templates, checklists |
| Classification engine | Map crate traits (unsafe, concurrency, parsers, feature flags) to risk classes | Rule-based classifier with inputs from crate metadata and spec |
| Tool selection | Select mandatory + conditional tools with rationale | Rule table lookup + decision log |
| Gate planner | Produce PR/Nightly/Release gate requirements + waivers | CI gate schema and checklist |
| Execution layer | Run selected tools and collect outputs | CI workflows, cargo tool invocations |
| Evidence & reporting | Aggregate results, map to spec items, state verified/unverified scope | Report templates + structured artifact outputs |

## Recommended Project Structure

```
src/
├── policy/                 # Policy assets and templates
│   ├── baseline.md         # Mandatory baseline requirements
│   ├── applicability.md    # Tool applicability matrix
│   └── report-template.md  # Reporting structure
├── domain/                 # Core domain models
│   ├── classification.rs   # Crate traits and risk classes
│   ├── tools.rs            # Tool catalog + applicability rules
│   └── gates.rs            # PR/Nightly/Release gate definitions
├── analysis/               # Decision logic
│   ├── classifier.rs       # Rule evaluation
│   ├── selector.rs         # Tool selection with rationale
│   └── planner.rs          # Gate plan construction
├── reporting/              # Report generation
│   ├── renderer.rs         # Markdown/JSON report builder
│   └── mapping.rs          # Spec-to-test mapping
├── integration/            # CI integration helpers
│   ├── github.rs           # GitHub Actions emitters
│   └── artifacts.rs        # Artifact paths and storage conventions
└── cli/                    # CLI entrypoints (if needed)
    └── main.rs
```

### Structure Rationale

- **policy/**: Keeps non-code governance artifacts close to the system, since they are authoritative inputs.
- **domain/** + **analysis/**: Clean boundary between data models and decision logic; makes policy changes auditable.
- **reporting/**: Ensures the verified/unverified scope split is consistently expressed across outputs.
- **integration/**: CI provider specifics are isolated from policy logic to avoid vendor lock-in.

## Architectural Patterns

### Pattern 1: Policy-as-Data (Decision Tables)

**What:** Encode tool applicability and gate requirements in structured tables or rules.
**When to use:** When policy must be auditable and changeable without rewiring code.
**Trade-offs:** More upfront structuring; fewer ad-hoc decisions later.

**Example:**
```rust
// Pseudocode-style model
struct ToolRule {
    applies_if: Vec<CrateTrait>,
    tool: ToolId,
    rationale: &'static str,
}
```

### Pattern 2: Spec-to-Evidence Mapping

**What:** Every spec item maps to tests/tools and resulting evidence.
**When to use:** Always — required to avoid unsupported assurance claims.
**Trade-offs:** Additional mapping work; higher assurance clarity.

**Example:**
```rust
struct EvidenceMap {
    spec_item: String,
    tool: ToolId,
    gate: GateTier,
    status: EvidenceStatus,
}
```

### Pattern 3: Gate Tiering (PR/Nightly/Release)

**What:** Separate gates by CI tier with explicit rationale and waiver rules.
**When to use:** Always — requested in project constraints.
**Trade-offs:** Extra CI config; avoids conflating fast feedback with deep verification.

## Data Flow

### Request Flow

```
Testing Request
    ↓
Scope + Spec Inputs
    ↓
Classification Engine
    ↓
Tool Selection (mandatory + conditional)
    ↓
Gate Plan (PR/Nightly/Release + waivers)
    ↓
CI Execution + Evidence Collection
    ↓
Report Generation (verified vs unverified scope)
```

### Key Data Flows

1. **Policy → Decision:** Policy assets feed classification and selection logic.
2. **Decision → CI:** Gate plan configures CI jobs and required tools.
3. **CI → Report:** Tool outputs become evidence, mapped back to spec items.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 0-10 crates | Single repo + static policy docs; manual report generation |
| 10-100 crates | Centralized policy + reusable gate templates; automated reporting |
| 100+ crates | Policy registry + automated classification + standardized artifact schema |

### Scaling Priorities

1. **First bottleneck:** Manual report assembly → automate evidence collection and mapping.
2. **Second bottleneck:** Divergent CI configs → normalize with shared gate templates.

## Anti-Patterns

### Anti-Pattern 1: Coverage-Only Governance

**What people do:** Treat coverage or green tests as sufficient.
**Why it's wrong:** Violates governance rule against unsupported assurance.
**Do this instead:** Map spec items to tools + evidence and report residual risk.

### Anti-Pattern 2: Single-Gate CI

**What people do:** Run all tools on PR or only on nightly.
**Why it's wrong:** Blurs assurance tiers and breaks explicit gate requirements.
**Do this instead:** Keep PR/Nightly/Release gates separate with rationale.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| GitHub Actions (or CI) | Generated workflow fragments or templates | Keep gate tiers separate |
| Artifact storage | Publish reports + evidence logs | Must preserve waived/blocked info |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| policy ↔ analysis | Rule tables and metadata | Must be auditable |
| analysis ↔ integration | Gate plan output | Deterministic, versioned |
| integration ↔ reporting | Evidence bundle | Consistent schema for reports |

## Suggested Build Order

1. **Policy assets + domain models** (baseline, applicability matrix, gate schema).
2. **Classification + tool selection logic** (rule engine + rationale output).
3. **Gate planner** (PR/Nightly/Release tiering and waiver handling).
4. **Reporting pipeline** (spec-to-evidence mapping, verified/unverified outputs).
5. **CI integration** (workflow generation and artifact publishing).

## Sources

- `test/rust_crate_guideline.md`
- `.planning/PROJECT.md`

---
*Architecture research for: Rust crate test-governance system*
*Researched: 2026-03-21*
