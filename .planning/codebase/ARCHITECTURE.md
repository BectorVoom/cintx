# Architecture

**Analysis Date:** 2026-03-21

## Pattern Overview

**Overall:** Layered Rust library architecture with dual API surfaces (`safe` + `raw`) converging on a shared runtime pipeline and policy-driven CPU route resolution.

**Key Characteristics:**
- Domain contracts are isolated from execution mechanics (`src/contracts/*.rs` vs `src/runtime/*.rs`).
- Public API layers are thin facades over runtime orchestration (`src/api/safe.rs`, `src/api/raw.rs`).
- Route selection is manifest-backed and policy-gated before kernel execution (`src/runtime/backend/cpu/router.rs`, `src/runtime/policy.rs`).

## Layers

**Domain Contract Layer:**
- Purpose: Define validated chemistry/domain inputs and canonical operator/representation enums.
- Location: `src/contracts/`
- Contains: `Atom`, `Shell`, `BasisSet`, `Operator`, `IntegralFamily`, `Representation` (`src/contracts/atom.rs`, `src/contracts/shell.rs`, `src/contracts/basis.rs`, `src/contracts/operator.rs`, `src/contracts/representation.rs`)
- Depends on: Error taxonomy in `src/errors/libcint_error.rs`
- Used by: API and runtime modules (`src/api/*.rs`, `src/runtime/*.rs`), integration tests (`tests/*.rs`)

**API Facade Layer:**
- Purpose: Expose ergonomic safe/raw entry points and attach diagnostics envelopes.
- Location: `src/api/` and crate exports in `src/lib.rs`
- Contains: Safe workspace/evaluation APIs (`src/api/safe.rs`) and raw-compat APIs (`src/api/raw.rs`)
- Depends on: Runtime orchestration (`src/runtime/mod.rs`), diagnostics (`src/diagnostics/report.rs`)
- Used by: Downstream crate consumers and integration tests (`tests/phase1_workspace_query.rs`, `tests/phase2_raw_query_execute.rs`)

**Runtime Orchestration Layer:**
- Purpose: Validate requests, plan shapes/layout, estimate memory, and execute dispatch.
- Location: `src/runtime/`
- Contains: Validation (`src/runtime/validator.rs`), planning (`src/runtime/planner.rs`), layout (`src/runtime/layout.rs`), memory planning (`src/runtime/memory/chunking.rs`), execution (`src/runtime/executor.rs`), workspace query (`src/runtime/workspace_query.rs`)
- Depends on: Contracts, backend CPU routing/execution, and policy enforcement (`src/runtime/backend/cpu/mod.rs`, `src/runtime/policy.rs`)
- Used by: `src/api/safe.rs`, `src/api/raw.rs`, raw-compat modules under `src/runtime/raw/`

**Raw Compatibility Layer:**
- Purpose: Validate libcint C-style buffers and enforce query/execute contract parity.
- Location: `src/runtime/raw/`
- Contains: Raw request validation and views (`src/runtime/raw/validator.rs`, `src/runtime/raw/views.rs`), compat workspace query (`src/runtime/raw/query.rs`), compat execution (`src/runtime/raw/evaluate.rs`)
- Depends on: Runtime executor/memory policy and CPU backend route queries (`src/runtime/executor.rs`, `src/runtime/backend/cpu/mod.rs`)
- Used by: Raw API facade (`src/api/raw.rs`) and raw-focused tests (`tests/phase2_raw_contracts.rs`, `tests/phase2_raw_failure_semantics.rs`)

**CPU Backend Layer:**
- Purpose: Resolve route keys to callable targets and bridge safe/raw requests to libcint kernels.
- Location: `src/runtime/backend/cpu/`
- Contains: Route manifest and resolution (`src/runtime/backend/cpu/router.rs`), FFI symbols and extern declarations (`src/runtime/backend/cpu/ffi.rs`), kernel dispatch bridges (`src/runtime/backend/cpu/mod.rs`), Rust-native 1e wrappers (`src/runtime/backend/cpu/overlap_cartesian.rs`)
- Depends on: Runtime execution requests and raw views (`src/runtime/execution_plan.rs`, `src/runtime/raw/views.rs`)
- Used by: Runtime query/evaluate paths (`src/runtime/workspace_query.rs`, `src/runtime/executor.rs`, `src/runtime/raw/*.rs`)

**Manifest Governance Layer:**
- Purpose: Canonicalize and audit compatibility route manifests against compiled lock policy.
- Location: `src/manifest/` with CLI entry at `src/bin/manifest_audit.rs`
- Contains: Canonicalization helpers (`src/manifest/canonicalize.rs`), lock schema/governance (`src/manifest/lock.rs`), generated-vs-committed lock audit (`src/manifest/compiled.rs`)
- Depends on: CPU route manifest metadata (`src/runtime/backend/cpu/router.rs`, `src/runtime/backend/cpu/route_coverage_manifest.lock.json`, `compiled_manifest.lock.json`)
- Used by: Test governance gates (`tests/phase3_manifest_governance.rs`, `tests/phase3_compiled_manifest_audit.rs`) and CI workflows (`.github/workflows/compat-governance-pr.yml`, `.github/workflows/compat-governance-release.yml`)

## Data Flow

**Safe Query + Evaluate Flow:**

1. `src/api/safe.rs` receives typed `BasisSet` + `Operator` + `Representation` + shell tuple.
2. Input and shape checks run in `src/runtime/validator.rs` and `src/runtime/planner.rs`.
3. Route resolution runs via `resolve_safe_route` in `src/runtime/backend/cpu/router.rs`.
4. Memory/layout are computed in `src/runtime/memory/chunking.rs` and `src/runtime/layout.rs`.
5. `src/runtime/executor.rs` executes specialized CPU routes first (`execute_safe_specialized_route`), else deterministic fallback writers.
6. Diagnostics and failures are wrapped as `QueryResult<T>` using `src/diagnostics/report.rs`.

**Raw Compat Query + Execute Flow:**

1. `src/api/raw.rs` forwards raw buffers to `src/runtime/raw/query.rs` or `src/runtime/raw/evaluate.rs`.
2. Buffer/table validation runs through `RawAtmView`/`RawBasView`/`RawEnvView` in `src/runtime/raw/views.rs` and `validate_raw_contract` in `src/runtime/raw/validator.rs`.
3. Query path computes required bytes/chunking/cache using route-aware calls in `src/runtime/backend/cpu/mod.rs`.
4. Execute path enforces query/execute contract equality in `validate_query_then_execute_contract` (`src/runtime/raw/evaluate.rs`).
5. Route dispatch executes backend-specialized kernels (`execute_raw_specialized_route`) or chunked fallback synthesis.

**Manifest Governance Flow:**

1. Route policy source is the static route table in `src/runtime/backend/cpu/router.rs`.
2. Snapshot metadata is stored in `src/runtime/backend/cpu/route_coverage_manifest.lock.json`.
3. Compiled lock synthesis/audit is performed in `src/manifest/compiled.rs`.
4. CLI guard `src/bin/manifest_audit.rs` enforces drift policy against `compiled_manifest.lock.json`.

**State Management:**
- Runtime operations are request-scoped and mostly immutable (`ExecutionRequest`, `PlannedExecution`, `WorkspaceQuery` in `src/runtime/*.rs`).
- Global/shared state is static policy data (`ROUTE_COVERAGE_MANIFEST` in `src/runtime/backend/cpu/router.rs`) and embedded JSON lock snapshots (`include_str!` usage in `src/manifest/compiled.rs`).

## Key Abstractions

**Execution Request Contract:**
- Purpose: Canonical execution input shared across safe/raw routes.
- Examples: `ExecutionRequest`, `ExecutionDispatch`, `ExecutionMemoryOptions` in `src/runtime/execution_plan.rs`
- Pattern: Build from API inputs (`from_safe`/`from_raw`), pass unchanged through planner/router/executor.

**Route Manifest Entry:**
- Purpose: Declarative mapping from `{family, operator, representation}` to backend kernels and policy constraints.
- Examples: `CpuRouteManifestEntry`, `ResolvedCpuRoute` in `src/runtime/backend/cpu/router.rs`
- Pattern: Manifest-first route resolution; route IDs and parity gates tie runtime behavior to tests.

**Raw Contract Views:**
- Purpose: Prevent invalid C-buffer interpretation before execution.
- Examples: `RawAtmView`, `RawBasView`, `RawEnvView`, `CompatDims` in `src/runtime/raw/views.rs`
- Pattern: Typed view wrappers convert raw slices into validated metadata before planning/routing.

**Workspace + Memory Policy:**
- Purpose: Compute required bytes, working set, chunking, and allocation safety.
- Examples: `WorkspaceQuery` in `src/runtime/workspace_query.rs`, `MemoryPlan` in `src/runtime/memory/chunking.rs`
- Pattern: Validation then deterministic byte accounting with hard failure on overflow/limit mismatch.

**Governance Lock Model:**
- Purpose: Enforce canonical route/profile/stability inventory.
- Examples: `CompiledManifestLock`, `ManifestLockEntry`, `ManifestProfile` in `src/manifest/lock.rs`
- Pattern: Canonicalization + schema invariant checks + explicit drift approval path.

## Entry Points

**Library Crate Surface:**
- Location: `src/lib.rs`
- Triggers: Any consumer importing `cintx`
- Responsibilities: Re-export API modules, runtime types, route metadata, and governance utilities.

**Raw/Safe Runtime Entrypoints:**
- Location: `src/api/safe.rs`, `src/api/raw.rs`
- Triggers: Safe typed API calls or raw compatibility calls.
- Responsibilities: Forward into shared runtime and decorate failures with diagnostics context.

**Manifest Audit CLI:**
- Location: `src/bin/manifest_audit.rs`
- Triggers: `cargo run --bin manifest_audit -- check|generate`
- Responsibilities: Generate or audit `compiled_manifest.lock.json` against runtime route policy.

**Build Script:**
- Location: `build.rs`
- Triggers: Cargo build lifecycle.
- Responsibilities: Validate vendored libcint source presence, generate headers, compile static archives (`cint_phase2_cpu`, `cint`).

**Default Binary Stub:**
- Location: `src/main.rs`
- Triggers: `cargo run` without `--bin`.
- Responsibilities: Placeholder hello-world binary; production behavior lives in library APIs and `manifest_audit` bin.

## Error Handling

**Strategy:** Explicit `Result` propagation with structured domain/runtime errors and diagnostics-rich API failures.

**Patterns:**
- Core modules return `Result<_, LibcintRsError>` and fail early on invalid shape/layout/overflow (`src/runtime/planner.rs`, `src/runtime/memory/chunking.rs`, `src/runtime/raw/validator.rs`).
- API-facing query/evaluate calls wrap errors with stage metadata using `QueryDiagnostics::record_failure` (`src/diagnostics/report.rs`, `src/api/safe.rs`, `src/runtime/raw/query.rs`).
- Backend execution issues normalize into `LibcintRsError::BackendFailure` (`src/runtime/backend/cpu/mod.rs`, `src/runtime/backend/cpu/router.rs`).

## Cross-Cutting Concerns

**Logging:** `tracing` is used for structured success/failure telemetry in `src/diagnostics/report.rs`.
**Validation:** Input and policy validation is multi-stage across `src/contracts/*.rs`, `src/runtime/validator.rs`, `src/runtime/raw/validator.rs`, and `src/runtime/policy.rs`.
**Authentication:** Not applicable in current codebase (library/runtime code only; no identity or credential subsystem detected).

---

*Architecture analysis: 2026-03-21*
