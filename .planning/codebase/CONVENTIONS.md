# Coding Conventions

**Analysis Date:** 2026-03-21

## Naming Patterns

**Files:**
- Use `snake_case` filenames for Rust modules and test suites, matching patterns in `src/runtime/output_writer.rs`, `src/runtime/raw/query.rs`, and `tests/phase2_memory_contracts.rs`.
- Use phase-scoped test names (`phase1_`, `phase2_`, `phase3_`) for milestone suites, as in `tests/phase1_typed_contracts.rs`, `tests/phase2_cpu_execution_matrix.rs`, and `tests/phase3_route_audit.rs`.
- Use capability-specific suffixes for parity suites (`*_wrapper_parity.rs`) as in `tests/one_e_overlap_cartesian_wrapper_parity.rs` and `tests/two_e_wrapper_parity.rs`.

**Functions:**
- Use `snake_case` function names for API, runtime, and tests, as in `query_workspace_compat_with_sentinels` (`src/api/raw.rs`), `validate_query_then_execute_contract` (`src/runtime/raw/evaluate.rs`), and `stable_phase2_matrix` (`tests/common/phase2_fixtures.rs`).
- Use verb-first names for operations (`query_*`, `evaluate_*`, `validate_*`, `resolve_*`) in `src/runtime/workspace_query.rs`, `src/runtime/executor.rs`, and `src/runtime/backend/cpu/router.rs`.

**Variables:**
- Use `snake_case` locals and fields (`required_bytes`, `memory_limit_bytes`, `shell_tuple`) consistently in `src/runtime/validator.rs`, `src/runtime/raw/query.rs`, and `src/diagnostics/report.rs`.
- Use `UPPER_SNAKE_CASE` for constants, including API labels and tolerances (`RAW_COMPAT_QUERY_API` in `src/runtime/raw/query.rs`, `DEFAULT_ALIGNMENT_BYTES` in `src/runtime/memory/chunking.rs`, `ABS_TOLERANCE` in `tests/phase2_safe_raw_equivalence.rs`).

**Types:**
- Use `PascalCase` for structs/enums (`WorkspaceQueryOptions`, `RawCompatWorkspace`, `LibcintRsError`) in `src/runtime/validator.rs`, `src/runtime/raw/query.rs`, and `src/errors/libcint_error.rs`.
- Keep enum variants as `PascalCase` with domain meaning (`UnsupportedApi`, `MemoryLimitExceeded`, `CpuReference`) in `src/errors/libcint_error.rs` and `src/runtime/execution_plan.rs`.

## Code Style

**Formatting:**
- Use standard Rust formatting style (4-space indentation, trailing commas, wrapped match arms) consistent with `src/runtime/executor.rs` and `src/runtime/backend/cpu/ffi.rs`.
- No repository-level formatter config is present (`rustfmt.toml`/`.rustfmt.toml` not detected at repo root), so keep rustfmt-default output style.

**Linting:**
- Enforce unsafe discipline with crate-level `#![deny(unsafe_op_in_unsafe_fn)]` in `src/lib.rs`.
- Use focused lint exceptions only where interop signatures require them, as in `#[allow(clippy::too_many_arguments)]` at `src/runtime/executor.rs`, `src/runtime/raw/evaluate.rs`, and `src/runtime/backend/cpu/ffi.rs`.
- No repo-level `clippy.toml`/`.clippy.toml` is detected; keep lint behavior aligned with code-level attributes and Rust defaults.

## Import Organization

**Order:**
1. `core`/`std` imports first for language/runtime primitives (`src/api/raw.rs`, `src/runtime/raw/views.rs`, `src/runtime/backend/cpu/mod.rs`).
2. `crate::...` imports next for cross-module dependencies (`src/runtime/workspace_query.rs`, `src/runtime/planner.rs`).
3. `super::...` imports last for same-module layering (`src/runtime/output_writer.rs`, `src/runtime/raw/query.rs`).

**Path Aliases:**
- Use explicit Rust module paths (`crate::`, `super::`) instead of alias systems in source files like `src/runtime/mod.rs` and `src/contracts/mod.rs`.
- In integration tests, import the crate as `cintx::...` directly (`tests/phase2_raw_query_execute.rs`, `tests/phase3_manifest_governance.rs`).

## Error Handling

**Patterns:**
- Use typed domain errors with `LibcintRsError` variants from `src/errors/libcint_error.rs`; avoid string-only error channels.
- Return `Result<T, LibcintRsError>` from internal runtime/helpers (for example `src/runtime/validator.rs`, `src/runtime/memory/chunking.rs`).
- Wrap query/evaluate surface failures in diagnostics-rich `QueryResult<T> = Result<T, Box<QueryError>>` from `src/diagnostics/report.rs`.
- Convert internal failures with stage labels via `.map_err(|error| diagnostics.clone().record_failure(...))` as in `src/runtime/workspace_query.rs`, `src/api/safe.rs`, and `src/runtime/raw/evaluate.rs`.
- In tests, validate exact failure classes and fields with `matches!` (for example `tests/phase1_error_taxonomy.rs`, `tests/phase2_raw_failure_semantics.rs`).

## Logging

**Framework:** `tracing`

**Patterns:**
- Emit structured failure logs with `tracing::error!` from `QueryDiagnostics::record_failure` in `src/diagnostics/report.rs`.
- Emit structured success logs with `tracing::debug!` from `QueryDiagnostics::record_success` in `src/diagnostics/report.rs`.
- Keep logging centralized in diagnostics; core runtime modules (`src/runtime/executor.rs`, `src/runtime/planner.rs`) primarily propagate typed errors upward.

## Comments

**When to Comment:**
- Add targeted comments for non-obvious invariants and policy constraints, as in:
  - `src/runtime/execution_plan.rs` ("Keep request memory options aligned with query-time canonicalization.")
  - `src/runtime/planner.rs` (safe shell model `nctr=1` note)
  - `src/runtime/backend/cpu/ffi.rs` (linked symbol fallback rationale)
- Use short test comments to clarify fixture slot semantics, for example in `tests/common/phase3_helper_cases.rs`.

**JSDoc/TSDoc:**
- Not applicable for this Rust codebase.
- Rustdoc-style `///` API comments are not detected across sampled `src/` and `tests/`; current documentation is primarily in `docs/*.md` (for example `docs/cintx_detailed_test_design_en.md`).

## Function Design

**Size:** Keep orchestration functions split by responsibility (validation, planning, execution, diagnostics), following `src/runtime/workspace_query.rs` and `src/runtime/raw/query.rs`.

**Parameters:** Prefer explicit, typed argument lists (`&BasisSet`, `Operator`, `Representation`, slices/options) instead of implicit globals, as in `src/api/safe.rs` and `src/runtime/executor.rs`.

**Return Values:** Return computed metadata and payload explicitly (`EvaluationMetadata`, `EvaluationTensor`, `RawCompatWorkspace`) rather than mutating global state (`src/runtime/executor.rs`, `src/runtime/raw/query.rs`).

## Module Design

**Exports:** Use module barrels with `pub mod` + `pub use` to define API surfaces:
- `src/contracts/mod.rs`
- `src/runtime/mod.rs`
- `src/runtime/raw/mod.rs`
- `src/lib.rs`

**Barrel Files:** Keep `mod.rs` as aggregation points and avoid deep import coupling; tests use explicit local helper modules with `#[path = "common/..."]` (`tests/phase2_safe_raw_equivalence.rs`, `tests/phase3_helper_transform_parity.rs`).

---

*Convention analysis: 2026-03-21*
