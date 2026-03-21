# Testing Patterns

**Analysis Date:** 2026-03-21

## Test Framework

**Runner:**
- Rust built-in `cargo test` harness (integration-first).
- Config: no dedicated test-runner config file is detected (`jest.config.*`, `vitest.config.*`, `nextest` config not present at repo root).

**Assertion Library:**
- Standard Rust assertions/macros: `assert!`, `assert_eq!`, `assert_ne!`, `matches!`, and explicit `panic!` context strings across `tests/phase1_typed_contracts.rs`, `tests/phase2_raw_contracts.rs`, and `tests/phase3_regression_gates.rs`.

**Run Commands:**
```bash
cargo test --workspace                                                      # Run all integration tests
cargo test --workspace --test phase3_helper_transform_parity               # Run a single suite
cargo test --workspace --test phase3_regression_gates requirement_traceability_gate  # Run one gate test
```
- CI governance jobs also run `cargo run --bin manifest_audit -- check` in `.github/workflows/compat-governance-pr.yml` and `.github/workflows/compat-governance-release.yml`.

## Test File Organization

**Location:**
- Integration tests live in top-level `tests/`.
- Shared fixtures/helpers live in `tests/common/`.
- Source files in `src/` currently do not use `#[cfg(test)]` unit-test modules.

**Naming:**
- Use `snake_case` test filenames.
- Use phase-scoped names for governance progression (`tests/phase1_*`, `tests/phase2_*`, `tests/phase3_*`).
- Use capability-focused parity suites (`tests/*_wrapper_parity.rs`) for wrapper equivalence.

**Structure:**
```text
tests/
├── common/
│   ├── phase2_fixtures.rs
│   ├── oracle_runner.rs
│   └── phase3_helper_cases.rs
├── phase1_*.rs
├── phase2_*.rs
├── phase3_*.rs
└── *_wrapper_parity.rs
```

## Test Structure

**Suite Organization:**
```rust
#[path = "common/phase2_fixtures.rs"]
mod phase2_fixtures;

#[test]
fn stable_family_safe_raw_numeric_and_layout_equivalence() {
    let basis = phase2_fixtures::stable_safe_basis();
    // setup -> exercise -> assert
}
```
- Pattern appears in `tests/phase2_safe_raw_equivalence.rs`, `tests/phase2_cpu_execution_matrix.rs`, and `tests/phase3_helper_transform_parity.rs`.

**Patterns:**
- Setup pattern: fixture factories (`stable_safe_basis`, `stable_raw_layout`, `phase2_cpu_options`) from `tests/common/phase2_fixtures.rs`.
- Teardown pattern: no explicit teardown; tests allocate local vectors/slices and rely on scope drop.
- Assertion pattern: typed error matching (`matches!`) and tolerance-based numeric checks (`assert_within_tolerance`) in `tests/common/oracle_runner.rs`.

## Mocking

**Framework:** None detected.

**Patterns:**
```rust
let failure = raw::evaluate_compat(...).expect_err("raw execute must fail for undersized output");
assert!(matches!(failure.error, LibcintRsError::InvalidLayout { item: "out_length", .. }));
```
- Pattern used in `tests/phase2_raw_failure_semantics.rs` and `tests/phase2_raw_query_execute.rs`.

```rust
let optimizer_ptr = NonNull::new(optimizer_handle.as_ptr() as *mut c_void).expect("optimizer pointer");
```
- Used in `tests/two_e_wrapper_parity.rs` and `tests/phase3_optimizer_equivalence.rs` to exercise real optimizer pointer paths instead of mocking.

**What to Mock:**
- Current suites prefer real integrations and deterministic oracles; no mocking layer is established.

**What NOT to Mock:**
- Do not bypass route resolution, raw validator, or safe/raw API surfaces; parity suites exercise production paths directly (`tests/phase2_cpu_backend_routing.rs`, `tests/phase3_route_audit.rs`).

## Fixtures and Factories

**Test Data:**
```rust
pub fn stable_raw_layout() -> (Vec<i32>, Vec<i32>, Vec<f64>) { ... }
pub fn stable_safe_basis() -> BasisSet { ... }
pub fn stable_phase2_matrix() -> Vec<StableMatrixCase> { ... }
```
- Defined in `tests/common/phase2_fixtures.rs`.

```rust
pub fn oracle_expected_scalars_with_wrapper_override(...) -> Result<Vec<f64>, LibcintRsError> { ... }
```
- Defined in `tests/common/oracle_runner.rs` for deterministic route- and wrapper-based oracle comparison.

**Location:**
- Shared fixture/oracle modules: `tests/common/phase2_fixtures.rs`, `tests/common/oracle_runner.rs`, `tests/common/phase3_helper_cases.rs`.
- Suites import helpers via `#[path = "common/..."] mod ...;` (for example `tests/phase2_safe_raw_equivalence.rs`).

## Coverage

**Requirements:** No numeric line/branch coverage threshold is enforced in active workflows under `.github/workflows/`.
- Current CI gates are targeted suite runs in `compat-governance-pr.yml` and `compat-governance-release.yml`.
- `docs/cintx_detailed_test_design_en.md` documents stronger future tool expectations (`cargo-llvm-cov`, `proptest`, `cargo-hack`), but those tools are not currently wired in `Cargo.toml` or CI workflow steps.

**View Coverage:**
```bash
Not detected in active CI workflows (no coverage command is currently executed)
```

## Test Types

**Unit Tests:**
- Unit-style checks are implemented as integration tests against public/internal APIs (for example `tests/phase1_typed_contracts.rs`, `tests/phase1_error_taxonomy.rs`).

**Integration Tests:**
- Dominant style. Cross-layer checks validate safe/raw parity, route policy, memory contracts, governance locks, and wrapper parity:
  - `tests/phase2_safe_raw_equivalence.rs`
  - `tests/phase2_cpu_execution_matrix.rs`
  - `tests/phase3_route_audit.rs`
  - `tests/*_wrapper_parity.rs`

**E2E Tests:**
- Not used as a separate framework; no browser/service E2E harness detected.

## Common Patterns

**Async Testing:**
```rust
Not used (no async test attributes such as #[tokio::test] detected)
```

**Error Testing:**
```rust
let failure = cintx::raw::query_workspace(...).expect_err("mismatched dims should fail");
assert!(matches!(failure.error, LibcintRsError::DimsBufferMismatch { .. }));
assert_eq!(failure.diagnostics.api, "raw.query_workspace");
```
- Pattern used repeatedly in `tests/phase1_workspace_query.rs`, `tests/phase2_raw_contracts.rs`, and `tests/phase2_memory_contracts.rs`.

---

*Testing analysis: 2026-03-21*
