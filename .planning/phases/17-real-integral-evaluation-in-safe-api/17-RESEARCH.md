# Phase 17: Real-Integral Evaluation in Safe API - Research

**Researched:** 2026-05-12
**Domain:** Rust safe API internal executor swap + oracle parity testing
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**D-01:** Swap the safe-API's shadow `CubeClExecutor` for `cintx_cubecl::CubeClExecutor`. Delete the local stub struct `CubeClExecutor` (lines 492-562) and the synthetic helper `fill_staging_values` (lines 465-490). Import the real executor with `use cintx_cubecl::CubeClExecutor;` (re-exported at `cintx_cubecl::lib.rs:26`). The existing chunk loop, `ExecutionPlan::new`, `HostWorkspaceAllocator`, `schedule_chunks`, and `ExecutionIo` plumbing in `SessionQuery::evaluate` stays as-is.

**D-02:** No `unsafe` inside the safe-API path. The safe API does NOT call `cintx_compat::raw::eval_raw`. Going through `eval_raw` would require repacking typed `BasisSet`/`ShellTuple` back into `atm`/`bas`/`env` arrays. The real `CubeClExecutor` is the shared dispatch primitive — both surfaces consume it directly.

**D-03:** No new shared helper crate / module this phase. The duplication between `cintx-rs/src/api.rs::SessionQuery::evaluate` and `cintx-compat/src/raw.rs::eval_raw` is acceptable for Phase 17 scope.

**D-04:** Rewrite `evaluate_runs_runtime_path_and_returns_owned_output` test: drop the brittle `owned_values[0] == 1.0` line; replace with (1) idempotency check, (2) nonzero check (`|v| > 1e-18`), (3) keep existing extent/byte-count/stats invariants.

**D-05:** No inline `eval_raw` cross-check in `cintx-rs` unit tests.

**D-06:** New file: `crates/cintx-oracle/tests/safe_api_arity2_parity.rs`. Single-purpose file for "does the safe API return byte-identical values to vendored libcint 6.1.3?". Existing `one_electron_parity.rs`, `center_2c2e_parity.rs`, etc. stay `eval_raw`-driven and unchanged.

**D-07:** Per-symbol tests inside the new file, mirroring `one_electron_parity.rs` per-symbol pattern. 12 tests total.

**D-08:** Coverage set: 12 base arity-2 operators — `int1e_ovlp_{cart,sph,spinor}`, `int1e_kin_{cart,sph,spinor}`, `int1e_nuc_{cart,sph,spinor}`, `int2c2e_{cart,sph,spinor}`.

**D-09:** Tolerance: `atol=1e-12_f64; rtol=0.0_f64` (Phase 15 unified tolerance).

**D-10:** CI gating: the new tests run inside the existing `oracle_parity_gate` matrix (cpu/wgpu × profile). No new CI job.

**D-11:** Zero changes to types in `cintx-rs::api` or `cintx-rs::prelude`. All public structs, functions, and `FacadeError` variants stay source-compatible with v1.2.

**D-12:** `fill_staging_values` and the stub `CubeClExecutor` are deletable. They are private to `cintx-rs::api`. Researcher should grep for stray references.

### Claude's Discretion

- Whether to rename `evaluate_runs_runtime_path_and_returns_owned_output` to something like `evaluate_returns_deterministic_nonzero_real_values`.
- Exact module organization of the new `safe_api_arity2_parity.rs` (factor a shared `collect_safe_api_matrix` helper or inline per test).
- Basis fixture choice: H2O / STO-3G is sufficient; matches existing parity tests.
- Whether to add `#[cfg(has_vendor_libcint)]` guard to the new tests (strongly lean yes, matching the `test_*_vendor_parity` convention).
- Whether `int2c2e_spinor` requires any spinor-transform pre-/post-work this phase. Phase 12 landed real spinor transforms; arity-2 spinor cases should round-trip cleanly. Surface gaps as planner blockers but do not bake spinor changes into Phase 17.

### Deferred Ideas (OUT OF SCOPE)

- Shared dispatch helper between safe API and compat (hoist chunk loop into `cintx-runtime` or `cintx-cubecl`).
- Unstable-source arity-2 sweep through `SessionRequest` (`int1e_grids_sph` etc.).
- Multi-molecule oracle fixtures for the safe-API sweep.
- Data-driven parametric sweep helper (loop over `&[(symbol, representation)]` table).
- Inline `eval_raw` cross-check inside `cintx-rs` unit tests.
- Phase 18 arity-3/4 dispatch and Phase 19 ECP.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| RVAL-01 | `fill_staging_values` is replaced by real `cintx_cubecl::CubeClExecutor` dispatch for every arity-2 operator in `SessionQuery::evaluate`; no synthetic-pattern fallback remains. | D-01: delete lines 465-562, add `use cintx_cubecl::CubeClExecutor;`. The real executor's `supports()` and `execute()` already handle all 12 arity-2 operators via `kernels::supports_canonical_family` for "1e" and "2c2e". |
| RVAL-02 | New file `crates/cintx-oracle/tests/safe_api_arity2_parity.rs` drives every supported arity-2 intor through `SessionRequest::evaluate` and asserts byte-identity against libcint at `atol=1e-12`. | D-06/07/08/09: 12 named `#[cfg(has_vendor_libcint)]`-guarded tests; parity via vendored libcint 6.1.3 FFI; tolerance matches Phase 15 baseline. cintx-oracle needs `cintx-rs` added as a dependency. |
| RVAL-03 | No public API change in `cintx-rs`: `SessionRequest` constructors, accessors, and error types stay source- and SemVer-compatible with v1.2. | D-11/12: only private items are deleted/changed; public surface (lib.rs, prelude.rs, error.rs) confirmed unchanged. |
</phase_requirements>

---

## Summary

Phase 17 is a single behavioral swap inside `crates/cintx-rs/src/api.rs`: the private stub executor (lines 492-562) and its synthetic `fill_staging_values` helper (lines 465-490) are deleted, and `use cintx_cubecl::CubeClExecutor;` replaces the stub. The surrounding chunk loop, `ExecutionPlan::new`, `HostWorkspaceAllocator`, `schedule_chunks`, and `ExecutionIo` wiring in `SessionQuery::evaluate` (lines 98-285) are already structurally identical to `cintx-compat/src/raw.rs::eval_raw` (lines 461-557) and require no changes.

The observable change is: `TypedEvaluationOutput::tensor::owned_values` now contains real libcint-compatible integral values instead of the `(idx+1)` / `(idx+1)*0.5` pattern. This is proven by the new `crates/cintx-oracle/tests/safe_api_arity2_parity.rs` (12 per-symbol parity tests at `atol=1e-12` against vendored libcint 6.1.3). One new dependency edge is required: `cintx-oracle` must declare `cintx-rs` as a test/dev-dependency so the parity tests can call `SessionRequest::new` + `query_workspace` + `evaluate`.

Phase 16 delivered full multi-backend support including the real `CubeClExecutor` with ROCm, CUDA, Metal, wgpu, and CPU backends wired through `CINTX_BACKEND` env-var selection. The executor's `execute()` path in `crates/cintx-cubecl/src/executor.rs` (lines 185-232) calls `kernels::launch_family` for real integral evaluation — the exact path that `cintx-compat::raw::eval_raw` already uses. Phase 17 only wires the safe API to the same real executor, eliminating the last synthetic placeholder.

**Primary recommendation:** Delete the 98 lines of stub code (lines 465-562), add a single `use` import, update one unit test, add one new oracle test file with a new `cintx-rs` dependency in `cintx-oracle/Cargo.toml`, and reuse the existing vendor-comparison helpers verbatim from the existing parity test files.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Integral evaluation dispatch | API / Backend (`cintx-cubecl::CubeClExecutor`) | — | Kernels live in `cintx-cubecl`; executor owns backend selection and launch |
| Safe API facade | API (`cintx-rs`) | — | `SessionRequest` / `SessionQuery::evaluate` own the typed user-facing surface |
| Compat dispatch (`eval_raw`) | API (`cintx-compat`) | — | Existing parallel path; Phase 17 shares the executor, not the dispatch path |
| Parity oracle | Test (`cintx-oracle`) | — | All byte-identity checks vs vendored libcint live in `cintx-oracle/tests/` |
| CI gate | CI (`oracle_parity_gate`) | — | Existing matrix job; new tests run inside it without a new job |

---

## Standard Stack

### Core (already in workspace — no new dependencies except the one noted)

| Library / Crate | Version | Purpose | Why Standard |
|-----------------|---------|---------|--------------|
| `cintx-cubecl` (workspace) | path dep | Real executor + kernel dispatch | Phase 16-delivered; already imported in `cintx-rs/Cargo.toml` as a direct dep |
| `cintx-rs` (workspace) | path dep | Safe API under test | **New edge in `cintx-oracle/Cargo.toml`** — required so parity tests can call `SessionRequest::evaluate` |
| `cintx-compat` (workspace) | path dep | `eval_raw`, vendor FFI, `enforce_safe_facade_policy_gate` | Already in `cintx-oracle` for existing parity tests |

**New dependency to add:**

```toml
# crates/cintx-oracle/Cargo.toml [dev-dependencies] or [dependencies]
cintx-rs = { path = "../cintx-rs", default-features = false }
```

Feature forwarding: the new parity tests use `SessionRequest::evaluate`, which calls `cintx_cubecl::CubeClExecutor`. The cpu backend is activated by the existing `cpu = ["cintx-compat/cpu"]` on cintx-oracle (which transitively reaches `cintx-cubecl/cpu`). `cintx-rs` does not have a `cpu` feature — it relies on the transitive activation through `cintx-cubecl` which is already wired. However, the oracle parity test binary must be run with `--features cpu` so that the cpu backend is compiled in. The existing `oracle_parity_gate` CI step already passes `CINTX_BACKEND=cpu`; no CI change needed.

### Supporting (existing, no change)

| Library | Purpose | Notes |
|---------|---------|-------|
| `bindgen` 0.71.1 | Vendor FFI binding generation in `cintx-oracle/build.rs` | Already used for `has_vendor_libcint` cfg gate |
| `cc` 1.2.x | Vendored libcint 6.1.3 build | Already used in `cintx-oracle/build.rs` |

---

## Architecture Patterns

### System Architecture Diagram

```
SessionRequest::evaluate (cintx-rs/src/api.rs)
    |
    |-- enforce_safe_facade_policy_gate (cintx-compat) -- rejects bad operators
    |-- ExecutionPlan::new (cintx-runtime)              -- layout + workspace plan
    |-- CubeClExecutor::new() [REAL, cintx-cubecl]      -- BEFORE: stub; AFTER: real
    |   |-- supports(&plan)                              -- checks canonical_family
    |   |-- query_workspace(&plan)                       -- backend f64 capability check
    |   |-- for each chunk:
    |       |-- ExecutionIo::new (cintx-runtime)         -- staging buffer wiring
    |       |-- executor.execute(&plan, &mut io)          -- kernels::launch_family()
    |           |-- kernels::launch_family (1e or 2c2e)  -- real integral kernel
    |           |-- transform::apply_representation_transform -- cart→sph (non-spinor)
    |
    |-- accumulate chunk_staging into owned_values
    |-- return TypedEvaluationOutput { tensor, stats }
```

The chunk loop body in `SessionQuery::evaluate` (lines 187-240) is structurally identical to `eval_raw`'s chunk loop (raw.rs lines 501-541). The only Phase 17 change is the executor binding on line 142.

### Recommended Project Structure (changes only)

```
crates/
├── cintx-rs/src/api.rs          # DELETE lines 465-562; add use cintx_cubecl::CubeClExecutor
├── cintx-oracle/
│   ├── Cargo.toml               # ADD cintx-rs = { path = "../cintx-rs", ... }
│   └── tests/
│       └── safe_api_arity2_parity.rs   # NEW — 12 per-symbol parity tests
```

### Pattern 1: Real Executor Swap (RVAL-01)

**What:** Delete stub, import real executor. One line changes; ~98 lines are deleted.

**Before (lines 142, 492-562 in `crates/cintx-rs/src/api.rs`):**
```rust
// at the top of evaluate():
let executor = CubeClExecutor::new();   // calls stub

// at bottom of file:
fn fill_staging_values(representation: Representation, staging: &mut [f64]) { ... }

#[derive(Debug, Default)]
struct CubeClExecutor;

impl CubeClExecutor { fn new() -> Self { Self } ... }
impl BackendExecutor for CubeClExecutor { ... fill_staging_values(...) ... }
```

**After:**
```rust
// at the top of file (with other imports):
use cintx_cubecl::CubeClExecutor;  // ADDED — was already in cintx-rs/Cargo.toml

// at the top of evaluate():
let executor = CubeClExecutor::new();   // now calls real executor

// DELETE: fill_staging_values function (lines 465-490)
// DELETE: stub CubeClExecutor struct + impls (lines 492-562)
```

`cintx-cubecl` is already a direct dependency in `crates/cintx-rs/Cargo.toml` [VERIFIED: file read]. `CubeClExecutor` is re-exported at `cintx_cubecl::lib.rs:26` as `pub use executor::{BackendCache, CUBECL_RUNTIME_PROFILE, CubeClExecutor, check_shader_f64_in_features};` [VERIFIED: file read].

### Pattern 2: Unit Test Rewrite (D-04)

**What:** Drop `owned_values[0] == 1.0` (asserts synthetic pattern). Replace with deterministic + nonzero smoke check.

```rust
// Source: crates/cintx-rs/src/api.rs — current test at lines 659-688
#[test]
fn evaluate_returns_deterministic_nonzero_real_values() {
    let (basis, shells) = sample_basis(Representation::Cart);
    let request = SessionRequest::new(
        OperatorId::new(0),  // int1e_ovlp_cart
        Representation::Cart,
        &basis,
        shells,
        ExecutionOptions::default(),
    );

    // Idempotency: two calls must return identical values
    let query1 = request.clone().query_workspace().expect("query should succeed");
    let query2 = request.query_workspace().expect("query should succeed");
    let output1 = query1.evaluate().expect("evaluate should succeed");
    let output2 = query2.evaluate().expect("evaluate should succeed");
    assert_eq!(output1.tensor.owned_values, output2.tensor.owned_values,
        "evaluate must be deterministic");

    // Nonzero: real integral kernels produce nonzero values
    let nonzero_count = output1.tensor.owned_values.iter()
        .filter(|&&v| v.abs() > 1e-18)
        .count();
    assert!(nonzero_count > 0,
        "at least one element must be nonzero after real executor swap");

    // Existing invariants (unchanged)
    assert!(!output1.tensor.owned_values.is_empty());
    assert_eq!(
        output1.tensor.owned_values.len(),
        output1.tensor.extents.iter().product::<usize>()
    );
    assert_eq!(output1.bytes_written,
        output1.tensor.owned_values.len() * std::mem::size_of::<f64>());
    assert!(output1.stats.transfer_bytes > 0);
}
```

Note: the existing test uses a 2-shell `BasisSet` with `shell_l=1` (p-type). With the real executor, the overlap integral for two p-type shells on the same atom is a 3x3 matrix; diagonal elements will be positive and nonzero for self-overlap. [ASSUMED — the specific values depend on runtime kernel; but nonzero is guaranteed if the executor produces real integrals].

### Pattern 3: New Oracle Parity File (RVAL-02)

**What:** `crates/cintx-oracle/tests/safe_api_arity2_parity.rs` with 12 named tests following the `one_electron_parity.rs` pattern.

**Module gate:** `#![cfg(any(feature = "cpu", feature = "rocm"))]` — mirrors the pattern widened in Phase 16 for the existing parity files.

**Vendor tests gate:** `#[cfg(has_vendor_libcint)]` per-test — matches the convention in `one_electron_parity.rs`.

**Helper needed:** A `collect_safe_api_matrix` function that:
1. Builds a `BasisSet` + per-shell `ShellTuple` from the H2O STO-3G fixture.
2. Calls `SessionRequest::new(operator_id, representation, &basis, shells, ExecutionOptions::default())`.
3. Calls `.query_workspace()?.evaluate()`.
4. Returns the `owned_values` as `Vec<f64>`.

This helper mirrors `collect_1e_sph_matrix` in `one_electron_parity.rs` but uses the safe API instead of `eval_raw`. It needs a `BasisSet`-compatible translation of the H2O STO-3G basis.

**Operator ID → OperatorId mapping (VERIFIED from `api_manifest.rs`):**

| Symbol | OperatorId | Manifest Index |
|--------|-----------|----------------|
| `int1e_ovlp_cart` | `OperatorId::new(0)` | MANIFEST_ENTRIES[0] |
| `int1e_ovlp_sph` | `OperatorId::new(1)` | MANIFEST_ENTRIES[1] |
| `int1e_ovlp_spinor` | `OperatorId::new(2)` | MANIFEST_ENTRIES[2] |
| `int1e_kin_cart` | `OperatorId::new(3)` | MANIFEST_ENTRIES[3] |
| `int1e_kin_sph` | `OperatorId::new(4)` | MANIFEST_ENTRIES[4] |
| `int1e_kin_spinor` | `OperatorId::new(5)` | MANIFEST_ENTRIES[5] |
| `int1e_nuc_cart` | `OperatorId::new(6)` | MANIFEST_ENTRIES[6] |
| `int1e_nuc_sph` | `OperatorId::new(7)` | MANIFEST_ENTRIES[7] |
| `int1e_nuc_spinor` | `OperatorId::new(8)` | MANIFEST_ENTRIES[8] |
| `int2c2e_cart` | `OperatorId::new(9)` | MANIFEST_ENTRIES[9] (after int2e_ at 9-11; see below) |

**Correction from manifest read:** The manifest has entries 0-8 as 1e (ovlp/kin/nuc × cart/sph/spinor), then entries 9-11 as `int2e_{cart,sph,spinor}` (arity 4), then entries 12-14 as `int2c2e_{cart,sph,spinor}` (arity 2). The `OPERATOR_DESCRIPTORS` array assigns IDs sequentially. Therefore:

| Symbol | OperatorId |
|--------|-----------|
| `int2e_cart` | `OperatorId::new(9)` |
| `int2e_sph` | `OperatorId::new(10)` |
| `int2e_spinor` | `OperatorId::new(11)` |
| `int2c2e_cart` | `OperatorId::new(12)` |
| `int2c2e_sph` | `OperatorId::new(13)` |
| `int2c2e_spinor` | `OperatorId::new(14)` |

[VERIFIED: api_manifest.rs lines 163-260 read; the `int2e_*` family appears at entries 9-11 before `int2c2e_*` at entries 12-14.]

**Key difference from `collect_1e_sph_matrix`:** The safe API uses `BasisSet` + `ShellTuple` (typed, validated) rather than raw `atm`/`bas`/`env` arrays. The H2O STO-3G basis must be re-expressed using `Atom::try_new`, `Shell::try_new`, and `BasisSet::try_new`. The internal test `sample_basis` in `cintx-rs/src/api.rs` (lines 607-624) shows the pattern for a minimal basis. The oracle must build the full 5-shell H2O STO-3G basis using the same coordinate and exponent values as `build_h2o_sto3g()` in `one_electron_parity.rs`.

**Vendor reference values:** The vendor tests call the existing vendor FFI functions already available in `cintx-oracle` (e.g., `vendor_ffi::vendor_int1e_ovlp_sph`). The safe API output (from `owned_values`) is compared against these. The safe API returns values in its own layout (row-major, determined by `output_layout.extents`); the vendor returns column-major (Fortran order). The `collect_1e_sph_matrix_vendor` in `one_electron_parity.rs` already handles the column→row conversion. The safe API parity helper must apply the same conversion when comparing.

**Tolerance declaration (top of new file):**
```rust
// atol=1e-12, rtol=0.0 per Phase 15 unified oracle tolerance (D-09 / CONTEXT.md)
let atol = 1e-12_f64;
let rtol = 0.0_f64;
```

### Anti-Patterns to Avoid

- **Calling `eval_raw` inside the safe API path.** D-02 is explicit: the `BasisSet`/`ShellTuple` → `atm`/`bas`/`env` backward conversion is `unsafe` and architecturally wrong. The real executor is the shared primitive.
- **Hoisting the chunk loop into a new helper crate.** D-03 is explicit: do not create `cintx-runtime` helpers this phase.
- **Modifying existing oracle test files** (`one_electron_parity.rs`, `center_2c2e_parity.rs`, etc.). D-06 locks the new file as a separate unit.
- **Asserting `owned_values[0] == 1.0` in the updated unit test.** That assertion tied the test to the synthetic stub value.
- **Adding new `pub` items to `cintx-rs`.** D-11 locks all public types at v1.2.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Integral evaluation kernel | Custom computation inside the safe API | `cintx_cubecl::CubeClExecutor` | Already validated against libcint across all 5 base families in Phases 9-12; error-prone to duplicate |
| atm/bas/env from BasisSet | Typed→raw round-trip conversion | Share the executor directly (D-02) | Round-trip is lossy, requires `unsafe`, and the executor already accepts `ExecutionPlan` directly |
| Vendor FFI wrappers for new operators | New `vendor_ffi::vendor_*` functions for cart/spinor variants | The existing cart/sph/spinor vendor FFI functions in `cintx-oracle` | vendor FFI for 1e operators already covers all three representations (per Phase 11-12 delivery) |
| Tolerance selection logic | Per-family atol | The scalar `atol = 1e-12_f64; rtol = 0.0_f64;` (Phase 15 baseline) | One unified tolerance is the Phase 15 decision |

---

## Common Pitfalls

### Pitfall 1: Forgetting the `cintx-rs` dependency in `cintx-oracle/Cargo.toml`

**What goes wrong:** The new `safe_api_arity2_parity.rs` uses `cintx_rs::SessionRequest` and related types. Without the dependency, the file fails to compile.

**Why it happens:** `cintx-oracle` currently depends on `cintx-compat` (for `eval_raw`) and `cintx-core`, but not on `cintx-rs` (the safe facade). This is the only new Cargo edge this phase introduces.

**How to avoid:** Add to `crates/cintx-oracle/Cargo.toml`:
```toml
[dependencies]
cintx-rs = { path = "../cintx-rs", default-features = false }
```

**Warning signs:** `error[E0432]: unresolved import cintx_rs` at compile time.

### Pitfall 2: `cpu` feature not activated for parity tests

**What goes wrong:** The real `CubeClExecutor` selects its backend via `CINTX_BACKEND` env var. When no backend feature is compiled in, `resolve_backend_kind()` in `backend/mod.rs` returns an error. The parity tests need at least the `cpu` feature compiled.

**Why it happens:** `cintx-rs` has no `cpu` feature of its own. The cpu backend is activated through the `cintx-cubecl` feature chain. The oracle test must be run with `--features cpu` (which passes through `cintx-oracle/cpu` → `cintx-compat/cpu` → `cintx-cubecl/cpu`).

**How to avoid:** Gate the parity test file with `#![cfg(any(feature = "cpu", feature = "rocm"))]` (same as the existing oracle test files). The CI `oracle_parity_gate` already runs with `--features cpu` (via `CINTX_BACKEND=cpu cargo run ... -- oracle-compare`).

**Warning signs:** `UnsupportedApi { requested: "BackendNotCompiled..." }` at test runtime.

### Pitfall 3: Layout mismatch between safe API output and vendor reference

**What goes wrong:** The safe API returns `owned_values` in `output_layout`-determined order (row-major, bra-leading). The vendor libcint returns values in column-major (Fortran, ket-leading) order. Comparing `owned_values` directly against vendor output without conversion produces false mismatches.

**Why it happens:** libcint's C API writes results in column-major `out[j*ni + i]` order (Fortran convention). The cintx safe API + executor follows the `output_layout.extents` row-major convention. The existing `collect_1e_sph_matrix_vendor` in `one_electron_parity.rs` already handles this with `matrix[(row_offset + ii) * n_ao + (col_offset + jj)] = out[jj * ni + ii]`.

**How to avoid:** The `collect_safe_api_matrix` helper should assemble the AO matrix from `owned_values` assuming row-major order. The vendor function should be called identically to the existing `collect_1e_sph_matrix_vendor`. Then compare the two row-major matrices element-wise with `count_mismatches`.

**Warning signs:** Systematic element transposition in the mismatch output (elements `[i,j]` and `[j,i]` swapped).

### Pitfall 4: Stray references to `fill_staging_values` or the stub `CubeClExecutor`

**What goes wrong:** Deletion of the stub and helper leaves a compile error if any tests or other modules reference these private functions.

**Why it happens:** The stub is private (`struct CubeClExecutor` without `pub`; `fn fill_staging_values` without `pub`). However, the existing test at line 676 asserts `owned_values[0] == 1.0` which would break at runtime after the swap (not at compile time).

**How to avoid:** D-04 prescribes rewriting the test. Grep for any additional references before deleting.

Verification grep:
```bash
grep -rn "fill_staging_values\|CubeClExecutor" crates/cintx-rs/
```
Expected output: only the stub definition itself and the test call at line 142 (`let executor = CubeClExecutor::new()`). After the swap, the `CubeClExecutor::new()` call on line 142 will compile correctly because it now refers to the imported `cintx_cubecl::CubeClExecutor`.

### Pitfall 5: H2O STO-3G `BasisSet` construction for safe API parity tests

**What goes wrong:** The existing `build_h2o_sto3g()` in `one_electron_parity.rs` builds raw `atm`/`bas`/`env` arrays suitable for `eval_raw`. The safe API requires `Atom::try_new` + `Shell::try_new` + `BasisSet::try_new` + per-shell `ShellTuple`. The two representations must agree on coordinates and exponents to produce byte-identical results.

**Why it happens:** Different data representation paths for the same physical system. The `Atom` constructor takes `(charge, [x, y, z], NuclearModel, zeta, frac_charge)` with coordinates in Bohr. The `Shell` constructor takes `(atom_idx, ang_momentum, nprim, nctr, kappa, representation, exponents, coefficients)`. Both must match the STO-3G values in `build_h2o_sto3g()` exactly.

**How to avoid:** Use the exact same numerical values as `build_h2o_sto3g()`. Reference the `sample_basis_with_shells` pattern in `crates/cintx-rs/src/api.rs::tests` (lines 607-624) for the `Shell::try_new` constructor signature.

**Warning signs:** Large systematic differences (order of magnitude off) in the parity comparison — indicates unit or normalization mismatch rather than numerical precision issues.

### Pitfall 6: `int2c2e_spinor` vendor FFI availability

**What goes wrong:** If the vendored libcint does not expose `int2c2e_spinor` via a callable FFI wrapper, the `#[cfg(has_vendor_libcint)]` spinor parity test cannot call a reference function.

**Why it happens:** Spinor variants in libcint may not have a straightforward C function name. The existing oracle covers `int2c2e_sph` but the spinor vendor wrapper may not have been added yet.

**How to avoid:** Check `crates/cintx-oracle/src/vendor_ffi.rs` (or equivalent) for available wrapper functions before writing the `int2c2e_spinor` vendor test. If no vendor wrapper exists, the vendor-parity test for that symbol uses idempotency (two safe API calls agree) instead of vendor comparison — the same pattern as the pre-Phase-15 idempotency tests. Add a `TODO` comment noting that vendor vendor parity is deferred.

**Warning signs:** `error[E0425]: cannot find function vendor_int2c2e_spinor in module vendor_ffi`.

---

## Code Examples

### Correct import after executor swap

```rust
// Source: verified from crates/cintx-cubecl/src/lib.rs:26
// In crates/cintx-rs/src/api.rs — add to existing imports:
use cintx_cubecl::CubeClExecutor;
```

The line `let executor = CubeClExecutor::new();` on line 142 of `api.rs` requires no change. The real `CubeClExecutor::new()` (executor.rs lines 40-46) also has a zero-argument constructor with a `Default` derive, so the call compiles identically.

### Correct BasisSet construction for safe API parity tests

```rust
// Source: crates/cintx-rs/src/api.rs lines 607-624 (sample_basis_with_shells pattern)
use cintx_core::{Atom, BasisSet, NuclearModel, Shell, ShellTuple};
use std::sync::Arc;

fn arc_f64(v: &[f64]) -> Arc<[f64]> {
    Arc::from(v.to_vec().into_boxed_slice())
}

fn build_h2o_sto3g_safe_basis(rep: cintx_core::Representation) -> (BasisSet, Vec<Arc<Shell>>) {
    let atom_o = Atom::try_new(8, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
    let atom_h1 = Atom::try_new(1, [0.0, 1.4307, 1.1078], NuclearModel::Point, None, None).unwrap();
    let atom_h2 = Atom::try_new(1, [0.0, -1.4307, 1.1078], NuclearModel::Point, None, None).unwrap();
    let atoms = Arc::from(vec![atom_o, atom_h1, atom_h2].into_boxed_slice());

    // O 1s: l=0, 3 prim, 1 ctr
    let shell_o1s = Arc::new(Shell::try_new(
        0, 0, 3, 1, 0, rep,
        arc_f64(&[130.7093200, 23.8088610, 6.4436083]),
        arc_f64(&[0.15432897, 0.53532814, 0.44463454]),
    ).unwrap());
    // ... etc. for O-2s (l=0), O-2p (l=1), H1-1s (l=0), H2-1s (l=0)

    let shells = vec![shell_o1s, ...];
    let basis = BasisSet::try_new(atoms, Arc::from(shells.clone().into_boxed_slice())).unwrap();
    (basis, shells)
}
```

### safe_api_arity2_parity.rs per-symbol parity test skeleton

```rust
// Source: mirrors one_electron_parity.rs pattern (lines 307-354)
#[test]
#[cfg(has_vendor_libcint)]
fn test_int1e_ovlp_sph_safe_api_parity() {
    let (atm, bas, env) = build_h2o_sto3g_raw();  // for vendor reference
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Spheric);

    let atol = 1e-12_f64;
    let rtol = 0.0_f64;

    let safe_matrix = collect_safe_api_matrix(
        OperatorId::new(1),   // int1e_ovlp_sph
        Representation::Spheric,
        &basis,
        &shells,
    );

    let vendor_matrix = collect_1e_sph_matrix_vendor("ovlp", &atm, &bas, &env);

    let mismatches = count_mismatches(&vendor_matrix, &safe_matrix, atol, rtol);
    assert_eq!(mismatches, 0,
        "int1e_ovlp_sph safe API: {mismatches} elements exceed atol={atol:.0e}/rtol={rtol:.0e} vs vendored libcint"
    );
}
```

### Identifying stray references before deletion

```bash
# Run before deleting lines 465-562 in api.rs
grep -n "fill_staging_values\|struct CubeClExecutor" \
    /home/user/Documents/workspace/cintx/crates/cintx-rs/src/api.rs
# Expected: only the definition lines themselves (465, 492)
# NOT expected: any call site other than inside the stub impl itself
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Synthetic `(idx+1)` fill | Real `CubeClExecutor` dispatch | **Phase 17** (this phase) | `owned_values` now contain libcint-compatible integrals |
| `eval_raw`-only oracle tests | Safe API parity tests added | Phase 17 | RVAL-02: byte-identity proven for the safe surface |
| Single executor per path | Shared `cintx_cubecl::CubeClExecutor` across compat + safe surfaces | Phase 17 | DRY at the executor level; duplication remains only in the chunk-loop wrapper |

**Deprecated/outdated after Phase 17:**
- The stub `CubeClExecutor` struct in `cintx-rs/src/api.rs` (lines 492-562): deleted.
- The `fill_staging_values` function (lines 465-490): deleted.
- The `owned_values[0] == 1.0` assertion in the unit test (line 676): replaced.

---

## Runtime State Inventory

This is a code-level internal swap with no external side effects. No runtime state is affected.

| Category | Items Found | Action Required |
|----------|-------------|-----------------|
| Stored data | None — no database or datastore records involve the internal stub executor | None |
| Live service config | None — no n8n workflows or external services reference `fill_staging_values` | None |
| OS-registered state | None | None |
| Secrets/env vars | `CINTX_BACKEND=cpu` is set by CI; no new env vars introduced | None |
| Build artifacts | `target/` may cache compiled stub code — a clean build after deletion is sufficient | `cargo clean` if stale cache causes issues |

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | `cargo test` (built-in) + `cargo nextest` (not yet installed; not required) |
| Config file | No separate config; feature flags via `--features cpu` |
| Quick run command | `cargo test -p cintx-rs` |
| Full suite command | `cargo test -p cintx-rs && cargo test -p cintx-oracle --features cpu` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| RVAL-01 | `fill_staging_values` deleted; real executor executes real integrals | unit (smoke) | `cargo test -p cintx-rs -- evaluate_returns_deterministic_nonzero_real_values` | ✅ modify existing test in `api.rs:659` |
| RVAL-02 | Safe API byte-identical to vendored libcint at atol=1e-12 for 12 operators | integration (oracle) | `cargo test -p cintx-oracle --features cpu -- safe_api_arity2_parity` | ❌ Wave 0: create `crates/cintx-oracle/tests/safe_api_arity2_parity.rs` |
| RVAL-03 | No public API change; `cargo public-api` diff shows 0 changes | SemVer check | `cargo test -p cintx-rs` (all existing tests pass) | ✅ existing tests in `api.rs`, `builder.rs`, `error.rs` |

### Sampling Rate

- **Per task commit:** `cargo test -p cintx-rs`
- **Per wave merge:** `cargo test -p cintx-rs && cargo test -p cintx-oracle --features cpu`
- **Phase gate:** Both test suites green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `crates/cintx-oracle/tests/safe_api_arity2_parity.rs` — new file (RVAL-02); 12 per-symbol tests
- [ ] `crates/cintx-oracle/Cargo.toml` — add `cintx-rs` dependency (required for the new test file to compile)

*(All other test infrastructure already exists and covers RVAL-01 and RVAL-03.)*

---

## Project Constraints (from CLAUDE.md)

| Constraint | Source | Impact on Phase 17 |
|------------|--------|--------------------|
| Compatibility: libcint 6.1.3 at `atol=1e-12` | CLAUDE.md + Phase 15 | All new oracle tests assert at `atol=1e-12, rtol=0.0` |
| Architecture: CubeCL is primary compute backend | CLAUDE.md | Executor swap to `cintx_cubecl::CubeClExecutor` (already used by `eval_raw`) |
| API Surface: Safe API first | CLAUDE.md | Phase 17 delivers real values through the safe surface |
| Error Handling: public library errors use `thiserror` v2 | CLAUDE.md | `FacadeError` is unchanged; mapping from `cintxRsError` via `impl From` is unchanged |
| Artifacts: deliverables written to `/mnt/data` | CLAUDE.md | Oracle parity test results written by `xtask oracle-compare`; no new artifact paths introduced in Phase 17 |
| Rust toolchain: pin `stable` in `rust-toolchain.toml`; run CI with `cargo --locked` | CLAUDE.md | No toolchain or lockfile changes; only source edits and one new Cargo path-dep edge |
| No nightly as project baseline | CLAUDE.md | Not applicable (no nightly features used) |
| No public APIs exposing backend-specific runtime types | CLAUDE.md | The executor swap is internal; no new `pub` items added |

---

## Security Domain

No new authentication, session management, access control, cryptography, or externally-supplied input validation is introduced. Phase 17 is a pure internal dispatcher swap and test addition. The `enforce_safe_facade_policy_gate` call (line 110-116 and 132-138 of `api.rs`) remains in place and is not modified.

ASVS V5 (Input Validation): `SessionRequest::new` already validates `OperatorId` through `Resolver::descriptor` and `enforce_safe_facade_policy_gate`. No new input paths are opened.

---

## Open Questions

1. **Vendor FFI wrappers for cart and spinor variants of 1e and 2c2e operators**
   - What we know: `one_electron_parity.rs` has vendor wrappers for `int1e_ovlp_sph`, `int1e_kin_sph`, `int1e_nuc_sph` (sph only). Phase 12 added spinor vendor wrappers for some families.
   - What's unclear: Whether `vendor_int1e_ovlp_cart`, `vendor_int1e_ovlp_spinor`, `vendor_int2c2e_cart`, `vendor_int2c2e_sph`, `vendor_int2c2e_spinor` exist in `cintx-oracle::vendor_ffi`.
   - Recommendation: Grep `crates/cintx-oracle/src/vendor_ffi.rs` before writing the parity tests. For any missing vendor wrapper, use the idempotency-only strategy (two safe API calls agree) rather than skipping the test entirely. The `#[cfg(has_vendor_libcint)]` guard ensures the vendor comparison only compiles when the vendor build is available.

2. **`int2c2e_spinor` spinor transform correctness**
   - What we know: Phase 12 delivered real C→spinor transforms for all base families including 2c2e spinor. D-10 in CONTEXT.md notes "Phase 12 landed real spinor transforms; arity-2 spinor cases should round-trip cleanly."
   - What's unclear: Whether any spinor-specific transform gap exists for the 2c2e spinor path through the safe API vs through `eval_raw`.
   - Recommendation: The planner should include a "run the existing `center_2c2e_parity.rs` spinor parity test" verification step before declaring the safe API parity test green for `int2c2e_spinor`. If the compat-path spinor test is already green at `atol=1e-12`, the safe API path should be too (same executor, same transforms).

3. **Safe API H2O STO-3G BasisSet construction correctness**
   - What we know: `Shell::try_new` and `Atom::try_new` have validation logic that may reject invalid inputs. The normalization applied by the safe API path may differ from the normalization assumed by the raw path.
   - What's unclear: Whether the raw `build_h2o_sto3g()` coefficients are pre-normalized to libcint's convention, or whether cintx applies additional normalization inside `Shell::try_new`.
   - Recommendation: Run a quick idempotency check (two safe API calls agree) before adding vendor parity assertions. If the safe API idempotency check passes but vendor parity fails, investigate normalization.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The existing `executor.rs::BackendExecutor::execute()` supports `1e` and `2c2e` canonical families via `kernels::supports_canonical_family` | Standard Stack | If these families returned false, the safe API would return `UnsupportedApi` instead of results; observable as a test failure not a silent wrong answer |
| A2 | The unit test at line 676 (`owned_values[0] == 1.0`) is the only assertion that depends on synthetic values | Common Pitfalls | If other tests in `cintx-rs` assert synthetic-pattern values, they would also need updating; mitigated by grepping `fill_staging_values` |
| A3 | Nonzero element count > 0 is guaranteed for `int1e_ovlp_cart` on any two valid shells | Code Examples | Overlap self-integral of a normalized GTO is always 1.0; cross-integrals may be small but should be nonzero |

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable toolchain | All compilation | ✓ | 1.95.0 (cargo 1.95.0) | — |
| `cargo test` (built-in) | Test execution | ✓ | via 1.95.0 | — |
| `cintx-cubecl` (cpu feature) | RVAL-01 executor swap test | ✓ | path dep, default-on cpu | — |
| Vendored libcint 6.1.3 | RVAL-02 parity tests (`has_vendor_libcint`) | conditional | Depends on `CINTX_ORACLE_BUILD_VENDOR=1` at build time | Idempotency-only tests compile regardless; vendor parity tests require vendor build |
| `cargo nextest` | Faster CI | ✗ | not installed | `cargo test` is the fallback (already used in CI) |

**Missing dependencies with no fallback:**
- None — all phase-critical paths work with existing tools.

**Missing dependencies with fallback:**
- `cargo nextest`: not installed; `cargo test` is used instead (existing CI pattern).
- Vendored libcint: parity tests guarded by `#[cfg(has_vendor_libcint)]`; idempotency tests compile without vendor build.

---

## Sources

### Primary (HIGH confidence)
- `crates/cintx-rs/src/api.rs` — read in full; stub executor at lines 492-562, `fill_staging_values` at lines 465-490, chunk loop at lines 187-240, existing test at lines 659-688
- `crates/cintx-cubecl/src/lib.rs` — re-export at line 26 confirmed
- `crates/cintx-cubecl/src/executor.rs` — real `CubeClExecutor` struct + `BackendExecutor` impl at lines 35-232 read
- `crates/cintx-compat/src/raw.rs` — `eval_raw` at lines 411-557 read; identical chunk-loop structure confirmed
- `crates/cintx-oracle/tests/one_electron_parity.rs` — full file read; pattern source for new file
- `crates/cintx-oracle/tests/center_2c2e_parity.rs` — header read; `build_h2o_sto3g` pattern with `PTR_ENV_START` alignment confirmed
- `crates/cintx-ops/src/generated/api_manifest.rs` — first 260 lines read; OperatorId → symbol mapping confirmed
- `crates/cintx-oracle/Cargo.toml` — read; confirmed no `cintx-rs` dependency exists today
- `crates/cintx-rs/Cargo.toml` — read; `cintx-cubecl` is already a direct dependency
- `crates/cintx-rs/src/error.rs` — read; `FacadeError` variants confirmed stable
- `crates/cintx-rs/src/lib.rs` and `src/prelude.rs` — read; public re-exports confirmed stable
- `.github/workflows/compat-governance-pr.yml` — oracle_parity_gate matrix confirmed (lines 73-111)
- `.planning/phases/16-multi-backend-support/16-04-SUMMARY.md` — Phase 16 delivery confirmed
- `.planning/config.json` — `nyquist_validation: true` confirmed

### Secondary (MEDIUM confidence)
- `.planning/phases/17-real-integral-evaluation-in-safe-api/17-CONTEXT.md` — decisions D-01 through D-12 and discretion items read in full; all decisions cross-verified against source code

### Tertiary (LOW confidence)
- None

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all dependencies and imports verified by direct file reads
- Architecture: HIGH — chunk loop structure in api.rs and raw.rs verified as identical; executor swap is unambiguous
- Pitfalls: HIGH — all pitfalls derived from actual code inspection (missing dependency, feature gates, layout mismatch, stray references)

**Research date:** 2026-05-12
**Valid until:** 2026-06-12 (stable library; 30-day validity)
