# Codebase Structure

**Analysis Date:** 2026-03-21

## Directory Layout

```text
[project-root]/
├── src/                              # Library code (API, runtime, contracts, governance)
│   ├── api/                          # Public safe/raw API facades
│   ├── contracts/                    # Typed domain models and validation
│   ├── runtime/                      # Planning, validation, memory, backend dispatch
│   │   └── backend/cpu/              # Route manifest, FFI bridge, specialized wrappers
│   ├── manifest/                     # Compiled-lock canonicalization and audits
│   ├── diagnostics/                  # Query diagnostics envelope
│   ├── errors/                       # Shared error enum
│   ├── bin/                          # Operational CLI binaries (`manifest_audit`)
│   ├── lib.rs                        # Crate export surface
│   └── main.rs                       # Placeholder default binary
├── tests/                            # Integration and governance test suites
│   └── common/                       # Shared test fixtures/oracle helpers
├── docs/                             # Design and route-manifest specs
├── .github/workflows/                # Governance-oriented CI workflows
├── libcint-master/                   # Vendored upstream libcint source tree
├── compiled_manifest.lock.json       # Committed compiled route-profile lock snapshot
├── build.rs                          # Native build orchestration for libcint static archives
├── Cargo.toml                        # Crate manifest
├── .planning/                        # Planning/workflow artifacts used by GSD process
└── target/                           # Cargo build artifacts (generated)
```

## Directory Purposes

**`src/api`:**
- Purpose: Public API boundary for safe and raw callers.
- Contains: `safe` facade (`src/api/safe.rs`) and raw-compat facade (`src/api/raw.rs`).
- Key files: `src/api/safe.rs`, `src/api/raw.rs`, `src/api/mod.rs`

**`src/contracts`:**
- Purpose: Validate and hold typed basis/operator/representation data.
- Contains: Atom/shell/basis domain structs plus operator-family compatibility checks.
- Key files: `src/contracts/atom.rs`, `src/contracts/shell.rs`, `src/contracts/basis.rs`, `src/contracts/operator.rs`

**`src/runtime`:**
- Purpose: Internal execution pipeline from validation to backend dispatch.
- Contains: Query validator/planner/layout/executor/memory modules plus raw compatibility submodules.
- Key files: `src/runtime/validator.rs`, `src/runtime/planner.rs`, `src/runtime/executor.rs`, `src/runtime/workspace_query.rs`, `src/runtime/raw/query.rs`, `src/runtime/raw/evaluate.rs`

**`src/runtime/backend/cpu`:**
- Purpose: CPU route policy, kernel binding, and route-specific execution helpers.
- Contains: Route manifest table/resolver, FFI symbol map, wrapper-backed 1e implementations.
- Key files: `src/runtime/backend/cpu/router.rs`, `src/runtime/backend/cpu/mod.rs`, `src/runtime/backend/cpu/ffi.rs`, `src/runtime/backend/cpu/overlap_cartesian.rs`, `src/runtime/backend/cpu/route_coverage_manifest.lock.json`

**`src/manifest`:**
- Purpose: Governance model for canonical compiled route locks and drift auditing.
- Contains: Canonicalization helpers, manifest lock schema, route->lock synthesis.
- Key files: `src/manifest/canonicalize.rs`, `src/manifest/lock.rs`, `src/manifest/compiled.rs`

**`src/diagnostics` and `src/errors`:**
- Purpose: Cross-cutting failure and diagnostics contracts.
- Contains: `QueryDiagnostics`/`QueryError` envelope and `LibcintRsError` enum taxonomy.
- Key files: `src/diagnostics/report.rs`, `src/errors/libcint_error.rs`

**`src/bin`:**
- Purpose: Operational binary tools.
- Contains: Manifest lock audit/generation CLI.
- Key files: `src/bin/manifest_audit.rs`

**`tests`:**
- Purpose: Integration-level behavioral, parity, and governance checks.
- Contains: Phase-prefixed suites and wrapper parity tests.
- Key files: `tests/phase2_raw_query_execute.rs`, `tests/phase3_manifest_governance.rs`, `tests/phase3_regression_gates.rs`, `tests/one_e_overlap_cartesian_wrapper_parity.rs`

**`tests/common`:**
- Purpose: Shared fixtures and oracle logic reused across integration suites.
- Contains: Stable basis/raw layouts and deterministic oracle assertions.
- Key files: `tests/common/phase2_fixtures.rs`, `tests/common/oracle_runner.rs`, `tests/common/phase3_helper_cases.rs`

**`docs`:**
- Purpose: Design references and governance specification docs.
- Contains: Route-coverage manifest spec and detailed design notes.
- Key files: `docs/libcint_route_coverage_manifest_spec.md`, `docs/libcint_detailed_design_resolved_en_routing_amended.md`

**`libcint-master`:**
- Purpose: Vendored upstream C source used by `build.rs` for static linking.
- Contains: Upstream `src/`, `include/`, tests, scripts, and docs from libcint.
- Key files: `libcint-master/src/*.c`, `libcint-master/include/cint.h.in`

## Key File Locations

**Entry Points:**
- `src/lib.rs`: Primary crate entry and re-export surface for APIs/runtime/governance.
- `src/bin/manifest_audit.rs`: CLI entry for manifest generation/audit commands.
- `src/main.rs`: Default placeholder binary.
- `build.rs`: Cargo build entry for native libcint compilation/linking.

**Configuration:**
- `Cargo.toml`: Dependency and package configuration.
- `.github/workflows/compat-governance-pr.yml`: PR governance gates.
- `.github/workflows/compat-governance-release.yml`: Release/tag governance gates.
- `compiled_manifest.lock.json`: Committed compiled manifest lock consumed by audits.
- `src/runtime/backend/cpu/route_coverage_manifest.lock.json`: Route coverage lock snapshot embedded by runtime governance code.

**Core Logic:**
- `src/runtime/planner.rs`: Shape planning and dimensional contract computation.
- `src/runtime/executor.rs`: Safe execution orchestration, chunked writes, fallback behavior.
- `src/runtime/backend/cpu/router.rs`: Route resolution policy and manifest-backed dispatch target selection.
- `src/runtime/backend/cpu/mod.rs`: Kernel bridge functions for safe/raw execution.
- `src/runtime/raw/validator.rs`: Raw C-layout validation for compat API.
- `src/manifest/compiled.rs`: Route manifest to compiled-lock governance synthesis.

**Testing:**
- `tests/*.rs`: Integration suites by phase/parity topic.
- `tests/common/*.rs`: Shared fixtures and oracle helpers.

## Naming Conventions

**Files:**
- Rust source files use `snake_case.rs` (`src/runtime/workspace_query.rs`, `src/runtime/output_writer.rs`).
- Module roots use `mod.rs` (`src/runtime/mod.rs`, `src/contracts/mod.rs`).
- Integration tests use descriptive `snake_case` names with phase or route intent (`tests/phase2_cpu_backend_routing.rs`, `tests/two_e_wrapper_parity.rs`).

**Directories:**
- Source directories use lowercase domain nouns (`src/api`, `src/runtime`, `src/manifest`, `tests/common`).
- Backend specialization path uses nested lowercase taxonomy (`src/runtime/backend/cpu`).

## Where to Add New Code

**New Feature:**
- Primary code: Add API surface in `src/api/safe.rs` and/or `src/api/raw.rs`, then implement runtime behavior in `src/runtime/` (typically `planner.rs`, `validator.rs`, `workspace_query.rs`, `executor.rs`).
- Tests: Add integration coverage in `tests/` using phase-aligned naming, and shared fixtures in `tests/common/` only when reused across suites.

**New Component/Module:**
- Implementation: Place domain types in `src/contracts/`, execution internals in `src/runtime/`, backend dispatch logic in `src/runtime/backend/cpu/`, and governance rules in `src/manifest/`.

**Utilities:**
- Shared runtime helpers: `src/runtime/helpers.rs` (AO counts/layout, normalization, deterministic transforms).
- Raw buffer parsing helpers: `src/runtime/raw/views.rs`.
- Shared test helpers: `tests/common/`.

## Special Directories

**`target/`:**
- Purpose: Cargo build outputs and incremental artifacts.
- Generated: Yes
- Committed: No (ignored by `.gitignore`)

**`libcint-master/`:**
- Purpose: Vendored upstream libcint code compiled by `build.rs`.
- Generated: No (checked-in vendor source)
- Committed: Yes

**`.planning/`:**
- Purpose: GSD planning/state artifacts (`PROJECT.md`, `ROADMAP.md`, phase plans, codebase maps).
- Generated: No (workflow-maintained project artifacts)
- Committed: Yes

**`src/runtime/backend/cpu/route_coverage_manifest.lock.json`:**
- Purpose: Route policy lock snapshot consumed by runtime governance.
- Generated: No (tracked source artifact)
- Committed: Yes

---

*Structure analysis: 2026-03-21*
