# Phase 1: Manifest & Planner Foundation - Research

**Researched:** 2026-03-21  
**Domain:** Manifest inventory + planner/workspace foundation  
**Confidence:** MEDIUM

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
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

### Deferred Ideas (OUT OF SCOPE)
## Deferred Ideas
None - discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BASE-01 | Rust caller can model atoms, shells, basis sets, environment parameters, operators, and tensor layouts through explicit typed domain structures. | `docs/design/cintx_detailed_design.md` Section 6 shows the shape of `Atom`, `Shell`, `BasisSet`, `EnvParams`, `OperatorId`, `ExecutionPlan`, and tensor metadata and points to `crates/cintx-core` as the landing module for those definitions. |
| BASE-02 | Maintainer can generate and lock a manifest-backed API inventory that classifies stable, optional, and unstable-source families across the supported feature matrix. | `docs/design/cintx_detailed_design.md` Sections 3.2–3.3.1 describe the manifest schema, generation procedure, support matrix, and release-gate behavior, and `compiled_manifest.lock.json` is the current canonical output with `profile_scope`, `entries`, and `stability` annotations. |
| BASE-03 | Rust caller can resolve supported integral families and representations through a manifest-aware registry without relying on raw symbol names. | `docs/design/cintx_detailed_design.md` Sections 4.5–5.2 explain how the manifest resolver identifies API families, which enables the runtime planner/scheduler in `crates/cintx-runtime` to select kernels via metadata instead of symbol strings. |
</phase_requirements>

## Summary

Phase 1 delivers the typed domain, manifest, and planner foundation that later layers rely on: `docs/design/cintx_detailed_design.md` Section 6 mandates explicit `Atom`, `Shell`, `BasisSet`, `EnvParams`, `OperatorId`, `ExecutionPlan`, and tensor-layout models, all intended to be shared immutably via `Arc` and exposed through `crates/cintx-core`. Those models must power both the safe API and any manifest-driven resolver so that downstream code never reinterprets raw pointer arrays.

The manifest pipeline centers on `compiled_manifest.lock.json` as the canonical union of the `{base, with-f12, with-4c1e, with-f12+with-4c1e}` support matrix, and Section 3.3.1 spells out the schema fields (`family_name`, `symbol_name`, `feature_flag`, `stability`, `compiled_in_profiles`, etc.), the generation steps (nm-normalization + classification), and the release-gate policy (zero-diff guard + approval workflow). The resolver in `cintx-ops`/`cintx-runtime` must consume those entries to classify stability and provide lookup metadata rather than exposing raw symbol matches, which leaves the manifest lock as the only truth for BASE-02/BASE-03.

Planner/workspace scaffolding rounds out the contract: Sections 4.6 and 7.* describe how `runtime::planner` intercepts safe/raw calls, validates shell tuples/`dims`, estimates workspace via `WorkspaceAllocator`, applies `memory_limit_bytes`, and uses a `ChunkPlanner` so that `query_workspace()` and `evaluate()` expose deterministic chunking, traced decisions, and typed errors (`MemoryLimitExceeded`, `UnsupportedApi`, etc.) before CubeCL kernels run. That pipeline also ensures OOM-safe stops, tracing hooks, and chunking metrics before handing work to the backend.

**Primary recommendation:** Build the typed manifest/resolver tables in `cintx-ops`, keep `compiled_manifest.lock.json` as the canonical artefact, and wire `cintx-runtime`’s validator/planner/workspace stack to consume them so workspaces, chunking, and memory-limit enforcement are established before backend kernels land.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Rust toolchain | 1.94.0 (target pin) | Reproducible compilation and manifest/oracle gating | `CLAUDE.md` and `AGENTS.md` prescribe pinning the stable 1.94.0 toolchain (stable is current as of 2026‑03‑05) so every phase runs under the same resolver and avoids non-determinism. |
| cubecl | 0.9.0 | Shared GPU backend for all integral kernels | `AGENTS.md` and the design (Section 4.2) make CubeCL the only compute driver so runtime traits can remain backend‑agnostic while leveraging CubeCL’s kernels. |
| thiserror | 2.0.18 | Typed public error surface | Design Sections 4.6/6.6 mandate public `Result<T, cintxRsError>` enums built with `thiserror` so library consumers see structured errors instead of `anyhow`. |
| anyhow | 1.0.102 | xtask/benchmark/oracle/capi error handling | The project insists that binary/xtask/oracle layers use `anyhow` for contextual errors while the library keeps its `thiserror` contract (Sections 0.3–0.4). |
| tracing | 0.1.41 | Structured spans for planner decisions, chunking, fallback reasons, and OOM reporting | Tracing is the only permitted instrumentation in Sections 4.6, 7.4, and 7.6; planners, chunkers, and backend transfers emit spans for diagnostics and gates. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| rayon | 1.11.0 | Host-side chunk preparation and parallel staging | Section 7.9 encourages Rayon for parallel chunk prep so planners can fill command queues without replicating state. |
| smallvec | 1.15.1 | Small fixed-size collections for shell tuples, strides, component lists | Section 6.5 and the `TensorLayout` design favor `SmallVec` to limit heap churn while representing small, variable-length metadata. |
| num-complex | 0.4.6 | Complex cart/sph/spinor output tensors in the safe API | Section 6.6 highlights complex interleaving, and `num-complex` backs the typed safe output tensor elements. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| cubecl | Another GPU backend or CPU compute provider | Only if CubeCL becomes a correctness or maintainability blocker—CubeCL is currently the canonical backend per Section 4.2 and the oracle harness depends on its kernels. |
| thiserror for the public API | anyhow everywhere | `thiserror` keeps the library surface typed while `anyhow` is reserved for xtask/oracle layers (Sections 0.3–0.4); swapping them would leak binary semantics to users. |
| Exact toolchain pin | Floating `stable` | Pinning to 1.94.0 keeps manifests/oracles deterministic; using floating stable risks subtle manifest/feature-resolution drift across CI runs. |

**Installation:**
```bash
cargo add cubecl@0.9.0 thiserror@2.0.18 anyhow@1.0.102 tracing@0.1.41 rayon@1.11.0 smallvec@1.15.1 num-complex@0.4.6
```

**Version verification:** `cargo metadata --format-version 1 --no-deps` and the existing `Cargo.toml`/`Cargo.lock` entries (2026-03-21) confirm `cubecl 0.9.0`, `thiserror 2.0.18`, `anyhow 1.0.102`, `tracing 0.1.41`, `rayon 1.11.0`, `smallvec 1.15.1`, `num-complex 0.4.6`, and the overall dependency graph.

## Architecture Patterns

### Recommended Project Structure
```text
crates/
├── cintx-core       # Immutable domain models (Atom/Shell/BasisSet/Representation/OperatorId)
├── cintx-ops        # Generated manifest + resolver metadata / build.rs hook
├── cintx-runtime    # Validator, planner, scheduler, workspace, tracing layers
├── cintx-cubecl     # CubeCL kernels, transfers, fallback strategy
├── cintx-compat     # Raw/r FFI compat helpers and legacy wrappers
├── cintx-capi       # Optional C ABI shim that wraps compat
├── cintx-rs         # Safe Rust facade builders and entry points
├── cintx-oracle     # Vendored libcint comparison harness and identity tests
└── xtask            # Manifest/audit automation, benchmark drivers, manifest generation logic
```
This crate layout mirrors the design (Sections 4.2–4.5) and keeps each responsibility isolated while sharing `core` domain definitions across the manifest resolver and planner.

### Pattern 1: Manifest-driven resolution
**What:** `cintx-ops` builds a manifest whose entries (§3.3) know family/operator/representation/stability/feature flag/profile membership, and `ops::resolver` provides a lookup table that returns `OperatorDescriptor` metadata instead of raw symbol strings.
**When to use:** Every safe or compat entry point must resolve an `OperatorId` through this manifest so BASE-02/BASE-03 never expose symbol names or unauthorized families.
**Example:**
```rust
// Source: docs/design/cintx_detailed_design.md Section 3.3
pub struct ManifestEntry {
    pub family_name: &'static str,
    pub symbol_name: &'static str,
    pub representation: Representation,
    pub stability: Stability,
    pub feature_flag: FeatureFlag,
    pub profiles: &'static [&'static str],
    pub component_rank: ComponentRank,
}
```

### Pattern 2: Planner & workspace pipeline
**What:** Safe/raw entry points feed validated `ShellTuple`, `Representation`, and `OperatorDescriptor` metadata into `runtime::planner`, which estimates workspace (Section 7.4), enforces `memory_limit_bytes`, and emits `ChunkPlanner` decisions and tracing spans before calling the backend.
**When to use:** Every `query_workspace`/`evaluate` call; `planner` also determines whether `UnsupportedApi` or `MemoryLimitExceeded` should be returned when the requested combination falls outside validated envelopes (Section 3.11).
**Example flow:**
Safe API → validator → manifest resolver → planner → workspace estimator → chunker → backend executor (CubeCL) → writer.

### Anti-Patterns to Avoid
- **Anti-pattern:** Hand-coding symbol strings to pick kernels. **Why it's bad:** It bypasses manifest stability metadata and makes it impossible to gate optional/unstable families; use manifest-derived `OperatorId` tables instead (Sections 4.6 & 5.2).
- **Anti-pattern:** Chunking inside the backend or after CubeCL submission. **Why it's bad:** It violates the `memory_limit_bytes` contract and allows partial writes; chunking must happen inside the planner/workspace layer with explicit typed errors (Section 7.5).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| API inventory drift | Manual symbol strings, per-family diffs, or hand-curated lists | Generated manifest/resolver tables plus CI manifest-audit hooks (`xtask manifest-audit`) that compare reconstructions against `compiled_manifest.lock.json` | Section 3.3.1 mandates a single canonical lock, diff gating, and release gates, so manual lists risk mismatched coverage and CI failures. |
| Operator selection | String-based matches at runtime that expose symbol names to the safe API | Generated `OperatorId` + `OperatorDescriptor` metadata from the manifest resolver | Section 4.6 says lookup must be metadata-driven to keep raw arrays in compat and to honor stability/feature flags. |
| Memory-limit enforcement | Let the backend allocate freely or embed chunking ad hoc | `WorkspaceAllocator`, `WorkspaceEstimator`, and `ChunkPlanner` that evaluate work sizes before backend execution | Sections 4.9 and 7.4–7.5 require deterministic chunking and typed `MemoryLimitExceeded` errors to avoid partial writes and ensure OOM-safe stop. |

**Key insight:** Custom symbol tables or ad hoc chunking bypass the manifest’s support matrix and the workspace policies that define safe, typed behavior; reuse the generated metadata so the planned contracts are enforced everywhere.

## Common Pitfalls

### Pitfall 1: Manifest drift regenerates the lock accidentally
**What goes wrong:** Developers rebuild the manifest on every run, diff the new JSON, and trigger CI failures or release gate blockers.  
**Why it happens:** The lock’s generation is complex (nm-normalization, feature-flag union, classification), and automation sometimes reruns it by default.  
**How to avoid:** Treat `crates/cintx-ops/generated/compiled_manifest.lock.json` as the master artifact, regenerate only when upstream version/feature matrix/schema changes, and let `xtask manifest-audit` enforce zero diffs per Section 3.3.1.  
**Warning signs:** `cargo build` rewrites `compiled_manifest.lock.json`, CI reports manifest diff, or generated manifest entries change without a manifest schema change.

### Pitfall 2: lax `dims`/layout validation
**What goes wrong:** Compat or C ABI callers pass arbitrary `dims`, leading to partial writes or inconsistent `not0` output.  
**Why it happens:** The Safe API avoids `dims`, but compat layers may mimic upstream’s lax contracts without reusing the manifest metadata.  
**How to avoid:** Validate `dims` strictly using the manifest’s `component_rank`/representation lengths (Section 3.6.1) and reject overrides smaller/larger than natural shapes; never perform partial writes — return `InvalidDims`.  
**Warning signs:** `not0` reports true but `out == NULL` returns unexpected workspace, or `dims` validation returns success despite mismatched lengths.

### Pitfall 3: Ignoring `memory_limit_bytes`/chunking early
**What goes wrong:** Large evaluations allocate beyond the user limit, trigger backend OOMs, or produce partial writes.  
**Why it happens:** Planners that skip `WorkspaceEstimator`/`ChunkPlanner` let CubeCL or the backend guess chunking and do not honor the safe API contract from Section 7.5.  
**How to avoid:** Implement `ChunkPlanner` inside `cintx-runtime` so it splits work before backend execution, honoring `memory_limit_bytes` and returning `MemoryLimitExceeded` as specified.  
**Warning signs:** Backend logs show allocations exceeding limits, users see partial writes flagged as successes, or tracing lacks chunk decision spans.

## Code Examples

### Safe API signatures
```rust
pub fn query_workspace(
    op: OperatorId,
    rep: Representation,
    basis: &BasisSet,
    shells: ShellTuple,
    opts: &ExecutionOptions,
) -> Result<WorkspaceQuery, cintxRsError>;

pub fn evaluate_into<T: OutputElement>(
    op: OperatorId,
    rep: Representation,
    basis: &BasisSet,
    shells: ShellTuple,
    params: &OperatorParams,
    opts: &ExecutionOptions,
    out: &mut OutputTensorMut<'_, T>,
) -> Result<ExecutionStats, cintxRsError>;
```
Source: `docs/design/cintx_detailed_design.md` Section 5.4.

### Backend executor trait
```rust
pub trait BackendExecutor {
    fn supports(&self, plan: &ExecutionPlan) -> bool;
    fn query_workspace(&self, plan: &ExecutionPlan) -> Result<WorkspaceBytes, cintxRsError>;
    fn execute(&self, plan: &ExecutionPlan, io: &mut ExecutionIo<'_>) -> Result<ExecutionStats, cintxRsError>;
}
```
Source: `docs/design/cintx_detailed_design.md` Section 5.2.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hand-curated symbol tables or manual symbol-name lists | Auto-generated manifest lock + resolver metadata + `xtask manifest-audit` release gate | 2026-03-21 design revision (Section 3.3.1 & Section 9.2) | Deterministic API inventory, automatic release gating, and binary-level diff detection prevent unnoticed coverage gaps. |
| Backend-only memory-limit heuristics with best-effort partial writes | Planner-driven workspace estimation + chunking + typed `MemoryLimitExceeded` ( Sections 4.9 & 7.5 ) | 2026-03-21 design revision (Section 7.5) | Guarantees safe stops, predictable chunk decisions, and consistent tracing before CubeCL runs. |

**Deprecated/outdated:**
- GTG-based API surface: Section 3.11.3 explicitly keeps GTG out of GA/optional/unstable support matrix; do not add `with-gtg` or expose GTG entry points until independent implementation, oracle tests, and manifest additions exist.
- Partial writes on allocation failure: Section 7.5 forbids them; always return typed errors instead of noisy partial output.

## Open Questions

1. **What should the generated manifest/resolver tables look like in Rust (struct layout, builder, metadata organization)?**
   - What we know: Section 3.3 describes required fields and Section 4.6 calls for metadata-driven resolution.
   - What's unclear: The exact Rust struct/trait design that populates `OperatorId`, `OperatorDescriptor`, and lookup indices is still undefined.
   - Recommendation: Decide on a manifest entry struct, codegen output format, and resolver API before implementing the planner so BASE-02/BASE-03 have precise contracts.

2. **How do we migrate from the root `compiled_manifest.lock.json` to `crates/cintx-ops/generated/compiled_manifest.lock.json` without breaking references?**
   - What we know: D-01 labels the new path as canonical and D-02 states the lock covers the support matrix.
   - What's unclear: Whether the root lock is copied, renamed, or replaced and how existing scripts/CI will find it.
   - Recommendation: Treat the root lock as input, migrate references in `xtask`, CI, and crates that consume the lock, and ensure the new location is the only one verified during manifest generation once the migration plan is settled.

3. **What heuristics should `ChunkPlanner`/workspace estimator adopt for chunk size, fallback reasons, and `MemoryLimitExceeded` detection?**
   - What we know: Sections 4.9 and 7.4 outline the need for chunking, fallible allocation, and deterministic chunk counts.
   - What's unclear: Concrete chunk-size heuristics, when to keep chunking versus failing, and how to record fallback/reason spans in tracing.
   - Recommendation: Define chunk thresholds, fallback criteria, and tracing fields before wiring the planner so the first implementation honors the safe-stop contract.

## Sources

### Primary (HIGH confidence)
- `docs/design/cintx_detailed_design.md` — Sections 3.1–9 frame the manifest schema, resolve/workspace flow, typed domain models, planner contracts, and tracing/validation requirements that Phase 1 must implement.
- `compiled_manifest.lock.json` — Current canonical lock with `profile_scope`, `entries`, and per-symbol `stability` metadata; a concrete example of the output Phase 1 must produce and consume.

### Secondary (MEDIUM confidence)
- `Cargo.toml` / `Cargo.lock` (2026-03-21) — Current dependency versions for `cubecl`, `thiserror`, `tracing`, `rayon`, `smallvec`, `num-complex`, etc., which confirm the stack choices and installation commands.
- `CLAUDE.md` / `AGENTS.md` — Project-level constraints (cubecl-only backend, thiserror vs anyhow, pinned toolchain, manifest governance) that influence stack selections and release gating.

### Tertiary (LOW confidence)
- None — all assertions rely on authoritative design docs and repository artifacts.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — dependency versions are sourced directly from `Cargo.toml`/`Cargo.lock` and the stack doc.
- Architecture: MEDIUM — the design document is authoritative, but runtime modules are still stubs, so implementation details remain to be verified.
- Pitfalls: MEDIUM — the risks come from documented contracts, yet their mitigation must be proven during implementation.
**Research date:** 2026-03-21  
**Valid until:** 2026-04-20
