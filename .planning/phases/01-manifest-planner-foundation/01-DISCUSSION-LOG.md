# Phase 1: Manifest & Planner Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md - this log preserves the alternatives considered.

**Date:** 2026-03-21
**Phase:** 01-manifest-planner-foundation
**Areas discussed:** Manifest source of truth, Typed foundation boundary, Planner and workspace contract, Phase 1 delivery boundary

---

## Manifest source of truth

### Q1: Where should the canonical compiled manifest lock live?

| Option | Description | Selected |
|--------|-------------|----------|
| `crates/cintx-ops/generated/compiled_manifest.lock.json` | Matches the design doc and keeps codegen inputs collocated with generated resolver artifacts | x |
| `compiled_manifest.lock.json` at repo root | Preserve the current location as the long-term source of truth | |
| Dual-source arrangement | Keep both locations authoritative for now | |

**User's choice:** [auto] `crates/cintx-ops/generated/compiled_manifest.lock.json`
**Notes:** Recommended default because the design doc points there; the current root file should be reconciled during planning rather than treated as a permanent second authority.

### Q2: What support matrix should Phase 1 encode into the lock metadata?

| Option | Description | Selected |
|--------|-------------|----------|
| `base`, `with-f12`, `with-4c1e`, `with-f12+with-4c1e`; exclude GTG | Matches the detailed design and release gate | x |
| `base` only for now | Delay optional-family metadata until later phases | |
| Include GTG as a future placeholder | Track GTG in the manifest early | |

**User's choice:** [auto] approved four-profile matrix with GTG excluded
**Notes:** Recommended default because the roadmap and design already lock this boundary.

### Q3: When is lock regeneration allowed?

| Option | Description | Selected |
|--------|-------------|----------|
| Only when upstream version, support matrix, or schema changes; CI otherwise diffs generated output | Keeps coverage auditable and stable | x |
| Regenerate on every build | Simplifies local development but weakens auditability | |
| Regenerate only at release time | Delays drift detection | |

**User's choice:** [auto] regenerate only on explicit version/schema/support-matrix changes
**Notes:** Recommended default because the lock is a release gate, not a cache.

### Q4: How should helper, legacy, and optional families be tracked?

| Option | Description | Selected |
|--------|-------------|----------|
| Put them in the same manifest and resolver system with stability metadata | Single source of truth for coverage and gating | x |
| Track them outside the main manifest | Separate inventories for helpers/legacy APIs | |
| Defer helper/optional metadata until later phases | Keep the Phase 1 manifest minimal | |

**User's choice:** [auto] single manifest/resolver system with stability metadata
**Notes:** Recommended default because release gates and downstream planners need one catalog.

---

## Typed foundation boundary

### Q1: Which typed models must Phase 1 establish?

| Option | Description | Selected |
|--------|-------------|----------|
| `Atom`, `Shell`, `BasisSet`, `EnvParams`, `OperatorId`, `ExecutionPlan`, and output tensor metadata | Full typed foundation for downstream layers | x |
| Only atoms, shells, basis, and operator IDs | Minimal start; defer planning/output types | |
| Internal-only structs with no stable typed surface yet | Keep Phase 1 implementation-only | |

**User's choice:** [auto] full typed foundation set
**Notes:** Recommended default because these types appear repeatedly in the design's public and internal signatures.

### Q2: Where does raw array and symbol-string handling belong?

| Option | Description | Selected |
|--------|-------------|----------|
| Confine it to compat/resolver boundaries; keep it out of the safe API | Preserves typed-safe contracts | x |
| Allow mixed raw and typed inputs in early safe APIs | Faster bootstrap but leaks C semantics upward | |
| Delay the separation until compat phase | Postpone the boundary decision | |

**User's choice:** [auto] keep raw handling out of the safe API
**Notes:** Recommended default because Phase 1 exists to lock the typed contract before higher layers grow around it.

### Q3: What ownership model should the foundation prefer?

| Option | Description | Selected |
|--------|-------------|----------|
| Immutable shared domain objects with plans borrowing from them | Matches the design and avoids unnecessary cloning | x |
| Clone all inputs into every execution plan | Simpler ownership, higher memory churn | |
| Keep most foundation objects mutable for now | Delays contract clarity | |

**User's choice:** [auto] immutable shared domain objects with borrowed plan views
**Notes:** Recommended default because the design explicitly calls out `ExecutionPlan` borrowing core inputs.

### Q4: When should layout and shape validation start?

| Option | Description | Selected |
|--------|-------------|----------|
| In Phase 1, alongside typed foundations and planner contracts | Makes later phases build on explicit invariants | x |
| Only once real execution exists | Risks pushing invalid assumptions into multiple layers | |
| Only in the raw compat layer | Leaves the safe path underspecified | |

**User's choice:** [auto] validate invariants in Phase 1
**Notes:** Recommended default because later planner/backend work depends on these contracts already being explicit.

---

## Planner and workspace contract

### Q1: How should the safe API expose planning versus execution?

| Option | Description | Selected |
|--------|-------------|----------|
| Split `query_workspace()` and `evaluate()` from the start | Makes workspace planning explicit and testable | x |
| Single `evaluate()` entrypoint only | Simpler initial API but hides planning contract | |
| Add `query_workspace()` after the backend lands | Delays a core contract | |

**User's choice:** [auto] split `query_workspace()` and `evaluate()`
**Notes:** Recommended default because the design repeatedly treats workspace querying as canonical.

### Q2: What memory policy belongs in the foundation?

| Option | Description | Selected |
|--------|-------------|----------|
| Planner estimates workspace, honors `memory_limit_bytes`, chunks deterministically, and forbids partial writes | Establishes the no-surprises contract early | x |
| Add memory limits only after backend execution exists | Defers a core non-functional requirement | |
| Allow partial writes under pressure for early progress | Conflicts with the design's stop policy | |

**User's choice:** [auto] establish memory-limit and no-partial-write policy now
**Notes:** Recommended default because OOM-safe behavior is a primary project goal, not a polish task.

### Q3: How should unsupported or invalid requests surface?

| Option | Description | Selected |
|--------|-------------|----------|
| Typed errors such as `UnsupportedApi`, invalid layout, and invalid dims | Clear contracts for planners, tests, and callers | x |
| Generic backend failures for all unsupported cases | Simpler implementation, weaker diagnostics | |
| Silent fallback whenever possible | Risks surprising behavior and parity drift | |

**User's choice:** [auto] typed errors
**Notes:** Recommended default because later phases and release gates depend on distinct failure categories.

### Q4: How much observability belongs in Phase 1?

| Option | Description | Selected |
|--------|-------------|----------|
| Add tracing hooks for planner decisions, chunking, fallback, transfer, and OOM now | Builds diagnostics into the contract early | x |
| Minimal logging now, tracing later | Delays visibility into planner behavior | |
| No diagnostics until the verification phase | Too late for foundation feedback loops | |

**User's choice:** [auto] tracing hooks are part of the foundation
**Notes:** Recommended default because the design treats observability as a first-class requirement.

---

## Phase 1 delivery boundary

### Q1: What should happen to the current crate split?

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse the existing crate split and fill the current stubs in place | Preserves the repository boundary model already aligned to the design | x |
| Collapse into one crate before implementation | Simplifies bootstrapping but discards the planned boundaries | |
| Reorganize the repository before coding | Adds churn ahead of the actual phase goal | |

**User's choice:** [auto] reuse the existing crate split
**Notes:** Recommended default because the current skeleton already mirrors the planned architecture.

### Q2: Where does Phase 1 stop?

| Option | Description | Selected |
|--------|-------------|----------|
| Stop at typed models, manifest/resolver generation, validator/planner scaffolding, and workspace policy | Leaves execution and public facades for later phases | x |
| Also implement compat raw writers in Phase 1 | Pulls Phase 2 work forward | |
| Also expose the safe Rust facade in Phase 1 | Pulls Phase 3 work forward | |

**User's choice:** [auto] stop at foundation contracts and scaffolding
**Notes:** Recommended default because the roadmap already separates execution/compat and public surfaces into later phases.

### Q3: How should the resolver identify operations?

| Option | Description | Selected |
|--------|-------------|----------|
| Family/operator/representation metadata drives resolution; raw symbol strings stay internal | Keeps resolution declarative and typed | x |
| Public APIs resolve only by raw symbol strings | Leaks low-level naming into the public contract | |
| Hard-code a temporary enum list before reading the manifest | Creates a second source of truth | |

**User's choice:** [auto] metadata-driven resolution
**Notes:** Recommended default because the manifest is supposed to be the canonical catalog.

### Q4: What happens to GTG and optional-family runtime support in this phase?

| Option | Description | Selected |
|--------|-------------|----------|
| Encode stability metadata now, keep GTG out, and defer runtime optional-family execution to later phases | Locks policy without over-scoping the phase | x |
| Add GTG placeholders to public feature flags | Conflicts with the design's out-of-scope rule | |
| Ignore stability metadata until optional families are implemented | Leaves the foundation incomplete | |

**User's choice:** [auto] encode stability metadata now, keep GTG out, defer runtime optional families
**Notes:** Recommended default because later phases depend on stability metadata already existing in the catalog.

## the agent's Discretion

- Exact internal trait shapes and helper types.
- The migration path for the existing root manifest lock.
- Code generation format details for generated tables and resolver helpers.
- Placeholder depth required to keep Phase 1 compile-ready before Phase 2.

## Deferred Ideas

None.
