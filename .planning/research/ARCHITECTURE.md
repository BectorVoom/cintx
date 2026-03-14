# Architecture Research

**Domain:** Rust-first scientific compute library (libcint-compatible integral engine)
**Researched:** 2026-03-14
**Confidence:** HIGH

## Standard Architecture

### System Overview

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ API Layer                                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Safe Rust API (facade)   Raw Compat API (atm/bas/env)   Optional C ABI     │
└───────────────┬───────────────────────────────┬─────────────────────────────┘
                │                               │
┌───────────────┴───────────────────────────────┴─────────────────────────────┐
│ Validation + Resolution Layer                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  Input Validator   Manifest Resolver   Feature/Stability Gate Classifier      │
└───────────────┬───────────────────────────────┬─────────────────────────────┘
                │                               │
┌───────────────┴───────────────────────────────┴─────────────────────────────┐
│ Runtime Layer                                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Planner   Scheduler/Chunker   Workspace Estimator   Dispatch/Fallback Logic  │
└───────────────┬───────────────────────────────┬─────────────────────────────┘
                │                               │
┌───────────────┴───────────────┐   ┌───────────┴────────────────────────────┐
│ CPU Backend (reference)        │   │ CubeCL Backend (optional GPU)          │
├────────────────────────────────┤   ├────────────────────────────────────────┤
│ Kernels + transforms + writer  │   │ Kernels + transfers + device caches    │
└───────────────┬────────────────┘   └───────────┬────────────────────────────┘
                │                                │
┌───────────────┴────────────────────────────────┴─────────────────────────────┐
│ Verification/Gate Layer                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ Oracle compare (vendored libcint) + manifest audit + feature-matrix CI gates │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| `facade` | Safe typed API (`BasisSet`, `OperatorId`, `ExecutionOptions`) | Thin public crate delegating to runtime |
| `compat` | libcint-style raw compatibility (`atm/bas/env`, `shls`, `dims`, cache/query behavior) | Unsafe boundary + strict validator + shared runtime |
| `capi` | Optional `extern "C"` ABI shim for migration | `feature = "capi"` crate layered over `compat` |
| `core` | Domain model, tensor layout/view types, typed errors | Dependency-free core crate |
| `ops` | Generated operator manifest and metadata tables | Generated code + lockfile-backed manifest resolver |
| `runtime` | Validation, planning, chunking, workspace allocation, dispatch | Shared orchestration crate used by safe+compat APIs |
| `cpu` | Reference numerical backend and canonical behavior | Kernel modules + transforms + direct writer |
| `cubecl` | Optional GPU acceleration backend | CubeCL executors, transfer planner, resident caches |
| `oracle/dev` | Result-compatibility gates and regression checks | Vendored libcint build/bindgen + comparison harness |

## Recommended Project Structure

```text
.
├── crates/
│   ├── libcint-rs/        # Safe facade exports
│   ├── libcint-core/      # Domain types, errors, tensor views, traits
│   ├── libcint-ops/       # Generated API manifest + resolver
│   ├── libcint-runtime/   # Validation, planning, dispatch, workspace
│   ├── libcint-cpu/       # CPU kernels/transforms/executor
│   ├── libcint-cubecl/    # CubeCL kernels/transfers/executor (feature-gated)
│   ├── libcint-compat/    # Raw compat + helper/legacy APIs
│   ├── libcint-capi/      # Optional C ABI shim (feature-gated)
│   └── libcint-oracle/    # Oracle comparison harness (dev/CI)
├── xtask/                 # manifest audit, oracle tooling, benchmark/report glue
├── docs/
│   ├── design/            # detailed design + diagrams
│   └── compatibility.md   # support matrix and gate status
├── libcint-master/        # vendored upstream oracle/reference source
└── .planning/             # planning and research artifacts
```

### Structure Rationale

- **`libcint-core` + `libcint-ops` split:** keeps domain model stable while operator inventory evolves via generated manifests.
- **Single shared `libcint-runtime`:** ensures safe API and raw compat API cannot drift in execution semantics.
- **Backend crates isolated from API crates:** CPU reference correctness stays independent from optional CubeCL acceleration.
- **`libcint-compat` and `libcint-capi` separated:** unsafe raw contracts and C ABI concerns stay outside safe facade.
- **`libcint-oracle` + `xtask` isolated:** verification gates remain strong without polluting production runtime dependencies.

## Build Order

1. **`libcint-core`**: establish typed domain model, tensor/layout contracts, and public error taxonomy.
2. **`libcint-ops`**: generate manifest schema/resolver and pin `compiled_manifest.lock.json` workflow.
3. **`libcint-runtime`**: implement validator, planner, scheduler, workspace allocator, and dispatch traits.
4. **`libcint-cpu`**: deliver reference executor for required stable families; wire direct writer/transform paths.
5. **`libcint-compat` + `libcint-rs`**: expose raw and safe APIs on the same runtime path; enforce `dims`/buffer contracts.
6. **`libcint-oracle` + `xtask` gates**: add oracle comparison, manifest-audit checks, helper parity, and CI release gates.
7. **`libcint-capi` (optional)**: add C ABI shim only after compat layer stabilizes and error-report API is defined.
8. **`libcint-cubecl` (optional GPU)**: add CubeCL executor with deterministic CPU fallback and CPU/GPU consistency gates.
9. **Feature-gated families**: add `with-f12` (sph-only), `with-4c1e` (Validated4C1E envelope), then `unstable-source-api`.

## Architectural Patterns

### Pattern 1: Layered API Surfaces on Shared Runtime

**What:** Safe Rust API, raw compatibility API, and optional C ABI all funnel into one runtime pipeline.
**When to use:** Always; this prevents semantic drift and duplicated numerical logic.
**Trade-offs:** Strong consistency and easier testing, but runtime abstractions must be carefully designed for both typed and raw call sites.

**Example:**
```rust
pub fn evaluate_safe(req: SafeRequest) -> Result<Stats, LibcintRsError> {
    let plan = runtime::planner::build_from_safe(req)?;
    runtime::execute(plan)
}

pub unsafe fn eval_raw(req: RawRequest<'_>) -> Result<RawEvalSummary, LibcintRsError> {
    let plan = runtime::planner::build_from_raw(req)?;
    runtime::execute(plan).map(RawEvalSummary::from)
}
```

### Pattern 2: Manifest-Driven Capability and Stability Gates

**What:** Route operators through generated metadata (`stable | optional | unstable_source`, feature flags, representation support).
**When to use:** For every operator lookup, CI audit, and release gate.
**Trade-offs:** More codegen/tooling overhead, but it makes "full coverage" claims auditable and protects against silent API drift.

**Example:**
```rust
match manifest.resolve(api_name, features)? {
    OperatorMeta { stability: Stability::Stable, .. } => proceed(),
    OperatorMeta { stability: Stability::Optional, feature, .. } if feature.enabled() => proceed(),
    meta => Err(LibcintRsError::UnsupportedApi { api: meta.name, reason: "feature/stability gate" }),
}
```

### Pattern 3: Backend Strategy with Deterministic CPU Fallback

**What:** Planner decides CPU vs CubeCL based on support and cost; unsupported/unstable paths fall back to CPU or return typed rejection.
**When to use:** Every execution plan with GPU enabled.
**Trade-offs:** Predictable correctness and easier debugging, but heuristic tuning is required to avoid GPU overhead regressions.

## Data Flow

### Request Flow

```text
[Safe API | Raw Compat | Optional C ABI]
              ↓
 [Input Validation + Manifest Resolve]
              ↓
[Plan shape/workspace/backend/chunking]
              ↓
      [Dispatch Decision]
        ↓             ↓
 [CPU Executor]   [CubeCL Executor]
        ↓             ↓
      [Writer / Transform / Output View]
              ↓
    [Result + typed error + tracing stats]
```

### State Management

```text
[Basis/Operator Metadata Cache]      [Optimizer Cache]
               ↓                           ↓
        [Planner + Scheduler] ←→ [Workspace Pool]
               ↓                           ↓
         [Execution Plan] → [Backend Executor]
               ↓
    [ExecutionStats + fallback reason + bytes]
```

### Key Data Flows

1. **Primary evaluation flow:** caller inputs are validated, normalized into an `ExecutionPlan`, executed on CPU/CubeCL, and written to compat-correct output layout.
2. **Oracle gate flow:** the same raw inputs run through project API and vendored libcint oracle; diffs/tolerances are enforced in CI.
3. **Feature/stability flow:** `with-f12`, `with-4c1e`, and `unstable-source-api` gates are checked at resolve/plan time before backend dispatch.
4. **OOM-safe flow:** workspace estimate compares against memory limits, then chunking is attempted; irreducible over-limit requests stop with typed errors.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| Small workloads (single calls to small batches) | CPU-only fast path; avoid GPU startup overhead; minimal caching. |
| Medium workloads (repeated batch workloads) | Reuse optimizer/workspace caches; enable Rayon chunk parallelism; tune chunk size by memory limit. |
| Large workloads (high-throughput homogeneous batches) | Enable CubeCL backend and device-resident caches; tune crossover heuristics and transfer plans; keep CPU fallback deterministic. |

### Scaling Priorities

1. **First bottleneck: workspace pressure/OOM risk.** Fix with allocator wrappers, up-front estimates, and chunk planner enforcement.
2. **Second bottleneck: backend crossover mistakes.** Fix with benchmarked dispatch thresholds and tracing of fallback/dispatch reasons.

## Anti-Patterns

### Anti-Pattern 1: Mixing Raw Pointer Contracts into Safe API

**What people do:** expose `atm/bas/env` offsets and `dims` sentinel behavior directly in "safe" interfaces.
**Why it's wrong:** defeats Rust type-safety goals and spreads unsafe layout assumptions across the codebase.
**Do this instead:** keep raw contracts in `compat`/`capi`; safe API uses typed `BasisSet`, `ShellTuple`, and tensor views.

### Anti-Pattern 2: Declaring Parity Without Manifest + Oracle Gates

**What people do:** mark APIs as supported based on header/source inspection or ad hoc tests only.
**Why it's wrong:** source-only/feature-gated symbols drift silently and compatibility claims become non-auditable.
**Do this instead:** require compiled-manifest lock diff checks, oracle comparisons, helper parity checks, and feature-matrix CI gates.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Vendored upstream `libcint` | Local build + bindgen via `libcint-oracle` | Authoritative oracle for result parity and helper API comparison. |
| CubeCL runtime | Optional backend crate (`feature = "gpu"`) | GPU acceleration only; CPU remains reference/fallback backend. |
| CI runners (feature matrix) | `xtask`-driven manifest/oracle/benchmark jobs | Release gates enforce manifest lock, oracle parity, 4c1e envelope, and F12 sph-only constraints. |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `facade` ↔ `runtime` | Typed plan/input structs | Safe API never touches raw pointer layout internals. |
| `compat` ↔ `runtime` | Validated raw-view DTOs | Only validated raw data may cross into planner. |
| `runtime` ↔ `cpu/cubecl` | `BackendExecutor` trait | Standard `supports/query_workspace/execute` contract. |
| `compat` ↔ `capi` | Thin FFI wrappers + error translation | C ABI is optional and isolated behind feature gate. |
| `oracle/xtask` ↔ `ops` | Manifest lock + resolver metadata | Gate coverage is generated from operator inventory, not hand-maintained lists. |

## Sources

- `.planning/PROJECT.md`
- `docs/libcint_detailed_design_resolved_en.md` (sections 1-5, 7, 10, 12-16)
- `.planning/codebase/STRUCTURE.md`
- `.planning/codebase/ARCHITECTURE.md`

---
*Architecture research for: libcint-rs (Rust-compatible libcint redesign)*
*Researched: 2026-03-14*
