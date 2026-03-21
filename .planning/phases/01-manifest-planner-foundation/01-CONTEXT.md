# Phase 1: Manifest & Planner Foundation - Context

**Gathered:** 2026-03-21
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase establishes the typed domain structures, manifest lock/governance, resolver, and planner/workspace foundations that every later layer consumes. It does not implement CubeCL execution, raw compat writers, the safe facade, or the optional C ABI; it makes those later phases possible by locking the canonical data model and planning contract now.

</domain>

<decisions>
## Implementation Decisions

### Manifest source of truth
- **D-01:** [auto] The canonical compiled manifest lock for downstream generation and gating is `crates/cintx-ops/generated/compiled_manifest.lock.json`; the current root `compiled_manifest.lock.json` is treated as existing input that Phase 1 must reconcile or relocate.
- **D-02:** [auto] The lock covers exactly the approved support matrix `base`, `with-f12`, `with-4c1e`, and `with-f12+with-4c1e`; GTG remains excluded.
- **D-03:** [auto] Normal builds validate generated manifest output against the lock; intentional lock regeneration is reserved for upstream-version, feature-matrix, or schema changes.
- **D-04:** [auto] Helper, legacy, and optional-family metadata remain part of the same manifest/resolver system rather than separate ad hoc registries.

### Typed foundation boundary
- **D-05:** [auto] Phase 1 defines explicit typed foundations for `Atom`, `Shell`, `BasisSet`, `EnvParams`, `OperatorId`, `ExecutionPlan`, and output tensor metadata.
- **D-06:** [auto] Raw offset-array and symbol-string handling stay outside the safe API and remain confined to compat/resolver boundaries.
- **D-07:** [auto] Typed domain objects are immutable/shared by default, and plans may borrow them rather than duplicating large structures.
- **D-08:** [auto] Validation scaffolding for shapes, layouts, and basic invariants belongs in the foundation phase, not as a late cleanup task.

### Planner and workspace contract
- **D-09:** [auto] The safe contract adopts `query_workspace()` and `evaluate()` as separate operations from the beginning.
- **D-10:** [auto] Planner foundations must estimate workspace, honor `memory_limit_bytes`, and define deterministic chunking/no-partial-write behavior before backend execution broadens.
- **D-11:** [auto] Unsupported families, invalid layouts, and envelope violations surface as typed errors such as `UnsupportedApi`, invalid layout, or invalid dims errors.
- **D-12:** [auto] Tracing hooks for planner decisions, chunking, fallback, transfer, and OOM behavior are part of the foundation scope.

### Phase 1 delivery boundary
- **D-13:** [auto] Reuse the existing crate split (`cintx-core`, `cintx-ops`, `cintx-runtime`) and fill the current stub landing zones instead of reorganizing the repository first.
- **D-14:** [auto] This phase stops at typed models, manifest generation/resolution, validator/planner scaffolding, and workspace policy; execution kernels, compat writers, and public facades stay in later phases.
- **D-15:** [auto] Resolver lookups key off family/operator/representation metadata rather than exposing raw exported symbol names in public-facing APIs.
- **D-16:** [auto] Stability metadata is encoded in the manifest now, even though optional-family runtime support is deferred to Phase 3 and GTG remains out of scope.

### the agent's Discretion
- Exact Rust type names and trait signatures inside the existing module skeletons.
- The migration strategy from the current root `compiled_manifest.lock.json` to the generated path in `cintx-ops`.
- The exact generated Rust table format for manifest/resolver code.
- The minimum compile-ready placeholder depth needed in stubs before Phase 2 planning.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project scope and phase boundary
- `.planning/PROJECT.md` - project intent, non-negotiable constraints, and core value.
- `.planning/REQUIREMENTS.md` - `BASE-01`, `BASE-02`, and `BASE-03` definitions plus downstream dependencies.
- `.planning/ROADMAP.md` - fixed Phase 1 scope, dependency order, and success criteria.
- `.planning/STATE.md` - current workflow position and phase focus.

### Detailed design authority
- `docs/design/cintx_detailed_design.md` Section 1.5 - compatibility policy and safe/raw API split.
- `docs/design/cintx_detailed_design.md` Section 1.6 - Rust-native type design policy.
- `docs/design/cintx_detailed_design.md` Section 3.2 - API inventory scope and manifest authority.
- `docs/design/cintx_detailed_design.md` Section 3.3 - manifest schema, lock governance, and support-matrix rules.
- `docs/design/cintx_detailed_design.md` Section 4.2 - layer decomposition across core, ops, runtime, compat, facade, and oracle.
- `docs/design/cintx_detailed_design.md` Section 4.6 - data flow from API selection through planning and execution.
- `docs/design/cintx_detailed_design.md` Section 5.4 - safe API signatures, including `query_workspace()` and evaluation boundaries.
- `docs/design/cintx_detailed_design.md` Section 5.5 - raw compatibility API signatures and sentinel-contract expectations.
- `docs/design/cintx_detailed_design.md` Section 6.1 - typed domain models to establish in Phase 1.
- `docs/design/cintx_detailed_design.md` Section 6.4 - internal DTO expectations for planning/execution.
- `docs/design/cintx_detailed_design.md` Section 7.1 - safe API call flow.
- `docs/design/cintx_detailed_design.md` Section 7.3 - input validation responsibilities.
- `docs/design/cintx_detailed_design.md` Section 7.4 - buffer allocation policy.
- `docs/design/cintx_detailed_design.md` Section 7.5 - OOM-safe stop behavior.
- `docs/design/cintx_detailed_design.md` Section 10.1 - feature matrix and GTG exclusion.
- `docs/design/cintx_detailed_design.md` Section 14.1 - release gates tied to manifest/oracle governance.

### Manifest inputs and current repository state
- `compiled_manifest.lock.json` - current compiled-symbol lock snapshot and approved profile list.
- `docs/design/api_manifest.csv` - design-time manifest snapshot and scope reference.
- `README.md` - intended crate layout and module responsibilities.
- `crates/cintx-core/src/lib.rs` - current core-type module boundary.
- `crates/cintx-ops/build.rs` - current manifest codegen hook location.
- `crates/cintx-ops/src/resolver.rs` - resolver landing zone.
- `crates/cintx-ops/src/generated/api_manifest.rs` - generated-manifest landing zone.
- `crates/cintx-runtime/src/planner.rs` - planner landing zone.
- `crates/cintx-runtime/src/validator.rs` - validation landing zone.
- `crates/cintx-runtime/src/workspace.rs` - workspace and memory-policy landing zone.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `crates/cintx-core/src/*.rs`: existing stub modules already line up with the typed-domain split required by Phase 1.
- `crates/cintx-ops/src/generated/` and `crates/cintx-ops/src/resolver.rs`: natural landing zones for generated manifest tables and family-resolution logic.
- `crates/cintx-runtime/src/planner.rs`, `validator.rs`, and `workspace.rs`: existing scaffold for the exact planning/validation/workspace boundary Phase 1 must establish.
- `compiled_manifest.lock.json`: current symbol-lock artifact that can seed or validate the generated canonical lock.

### Established Patterns
- The repository is already organized by boundary rather than by vertical feature, which matches the design doc and should be preserved.
- Most implementation files for this phase are stubs, so there is little legacy behavior to preserve beyond the current crate split.
- Generated artifacts already have a dedicated namespace under `cintx-ops`, which favors build-time code generation over hard-coded tables.

### Integration Points
- `cintx-core` types will feed both the manifest resolver in `cintx-ops` and the planning logic in `cintx-runtime`.
- Manifest generation must connect `compiled_manifest.lock.json`, `docs/design/api_manifest.csv`, and `cintx-ops/build.rs`.
- `xtask` manifest-audit and later oracle tooling will consume the same canonical lock produced in Phase 1.

</code_context>

<specifics>
## Specific Ideas

- Keep Phase 1 anchored to the current crate boundaries rather than spending time on repository reshaping.
- Favor generated manifest tables and resolver metadata over manually curated enums or symbol lists.
- Treat the current root `compiled_manifest.lock.json` as a migration concern to resolve explicitly during planning, not as an informal duplicate source of truth.

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within phase scope.

</deferred>

---
*Phase: 01-manifest-planner-foundation*
*Context gathered: 2026-03-21*
