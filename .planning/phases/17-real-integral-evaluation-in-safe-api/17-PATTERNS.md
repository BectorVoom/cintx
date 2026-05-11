# Phase 17: Real-Integral Evaluation in Safe API - Pattern Map

**Mapped:** 2026-05-12
**Files analyzed:** 4 (3 modified, 1 new)
**Analogs found:** 4 / 4

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/cintx-rs/src/api.rs` (delete stub, add import) | production-source | request-response | `crates/cintx-compat/src/raw.rs` (eval_raw executor binding) | exact |
| `crates/cintx-rs/src/api.rs` (rewrite unit test) | test | request-response | `crates/cintx-oracle/tests/one_electron_parity.rs` (idempotency + nonzero pattern) | role-match |
| `crates/cintx-oracle/tests/safe_api_arity2_parity.rs` | test | request-response | `crates/cintx-oracle/tests/one_electron_parity.rs` (per-symbol parity pattern) | exact |
| `crates/cintx-oracle/Cargo.toml` (add cintx-rs dep) | config | N/A | `crates/cintx-rs/Cargo.toml` (existing path-dep declarations) | exact |

---

## Pattern Assignments

### `crates/cintx-rs/src/api.rs` — executor swap (production-source, request-response)

**Analog:** `crates/cintx-compat/src/raw.rs`

**Imports pattern** (`raw.rs` lines 1-13 — the real executor import that `api.rs` must mirror):
```rust
use cintx_cubecl::{CUBECL_RUNTIME_PROFILE, CubeClExecutor};
use cintx_runtime::{
    BackendExecutor, ExecutionIo, ExecutionOptions, ExecutionPlan,
    HostWorkspaceAllocator, WorkspaceAllocator, WorkspaceQuery, query_workspace, schedule_chunks,
};
```

The import to ADD to `crates/cintx-rs/src/api.rs` (top-of-file import block) is:
```rust
use cintx_cubecl::CubeClExecutor;
```
`cintx-cubecl` is already a direct dependency in `crates/cintx-rs/Cargo.toml` line 14 — no Cargo.toml change needed for the executor swap.

**Real executor binding pattern** (`raw.rs` lines 461-493 — the executor construction + supports + query_workspace guard):
```rust
let executor = CubeClExecutor::new();
let mut allocator = HostWorkspaceAllocator::default();

// ... staging allocation ...

if !executor.supports(&plan) {
    return Err(cintxRsError::UnsupportedApi {
        requested: format!(
            "{}/{}/{}",
            plan.descriptor.family(),
            plan.descriptor.operator_name(),
            plan.representation
        ),
    });
}

let backend_workspace = executor.query_workspace(&plan)?.get();
if backend_workspace > plan.workspace.bytes {
    return Err(cintxRsError::MemoryLimitExceeded {
        requested: backend_workspace,
        limit: plan.workspace.bytes,
    });
}
```

This is structurally identical to the existing `api.rs` lines 142-163. The only change is that after adding `use cintx_cubecl::CubeClExecutor;`, the `let executor = CubeClExecutor::new();` call on line 142 resolves to the real executor rather than the local stub.

**Core chunk-loop pattern** (`raw.rs` lines 495-534 — executor.execute per chunk):
```rust
let schedule = schedule_chunks(&plan.workspace);
let total_units = plan.workspace.work_units.max(1);

for chunk in schedule.chunks() {
    let start = chunk.work_unit_start.min(total_units);
    let end = chunk
        .work_unit_start
        .saturating_add(chunk.work_unit_count)
        .min(total_units);
    let prefix = staging_elements.saturating_mul(start) / total_units;
    let suffix = staging_elements.saturating_mul(end) / total_units;
    let chunk_len = suffix.saturating_sub(prefix).max(1);

    // ... chunk_staging allocation ...
    let mut workspace = allocator.try_alloc(chunk.bytes, plan.workspace.alignment)?;
    {
        let mut io =
            ExecutionIo::new(chunk, &mut chunk_staging, &mut workspace, plan.dispatch)?;
        io.ensure_output_contract()?;
        let chunk_stats = executor.execute(&plan, &mut io)?;
        total_not0 = total_not0.saturating_add(chunk_stats.not0.max(0));
        total_transfer_bytes = total_transfer_bytes.saturating_add(io.transfer_bytes());
    }
    allocator.release(workspace);
    // ... copy chunk_staging into accumulator ...
}
```

The chunk loop in `api.rs` lines 187-240 is already structurally identical — do not modify it. Only the executor binding at line 142 changes.

**Code to DELETE** (`api.rs` lines 465-562 — the synthetic stub):
```rust
// DELETE this entire block:
fn fill_staging_values(representation: Representation, staging: &mut [f64]) { ... }

#[derive(Debug, Default)]
struct CubeClExecutor;

impl CubeClExecutor { fn new() -> Self { Self } ... }
impl BackendExecutor for CubeClExecutor { ... fill_staging_values(...) ... }
```

The `fill_staging_values` function (lines 465-490) and the stub `CubeClExecutor` struct + impls (lines 492-562) are deleted in their entirety. Neither is `pub`; no downstream crate references them.

**Anti-patterns to avoid:**
- Do NOT call `cintx_compat::raw::eval_raw` from inside `SessionQuery::evaluate` (D-02: requires `unsafe` typed→raw round-trip).
- Do NOT hoist the chunk loop into a new shared helper crate (D-03: out of scope).
- Do NOT add any new `pub` items to `api.rs` or `prelude.rs` (D-11: public API stays v1.2-compatible).
- Do NOT retain `fill_staging_values` in any form (it is the deliverable being removed).

**Pre-deletion verification grep** (run before deleting):
```bash
grep -n "fill_staging_values\|struct CubeClExecutor" \
    /home/user/Documents/workspace/cintx/crates/cintx-rs/src/api.rs
```
Expected: only definition lines 465 and 492. No call sites other than inside the stub impl itself.

---

### `crates/cintx-rs/src/api.rs` — unit test rewrite (test, request-response)

**Analog:** `crates/cintx-oracle/tests/one_electron_parity.rs` (idempotency + nonzero pattern, lines 307-354)

**Existing test to rewrite** (`api.rs` lines 659-688):
```rust
// CURRENT — brittle, asserts synthetic value:
#[test]
fn evaluate_runs_runtime_path_and_returns_owned_output() {
    // ...
    assert_eq!(output.tensor.owned_values[0], 1.0);  // DELETE THIS LINE
    // ...
}
```

**Target pattern** — idempotency + nonzero smoke (mirrors `one_electron_parity.rs` lines 307-354):
```rust
#[test]
fn evaluate_returns_deterministic_nonzero_real_values() {
    let (basis, shells) = sample_basis(Representation::Cart);
    let request = SessionRequest::new(
        OperatorId::new(0),
        Representation::Cart,
        &basis,
        shells,
        ExecutionOptions::default(),
    );

    // Idempotency: two calls with the same request must return identical values
    let query1 = request.clone().query_workspace().expect("query should succeed");
    let query2 = request.query_workspace().expect("query should succeed");
    let output1 = query1.evaluate().expect("evaluate should succeed");
    let output2 = query2.evaluate().expect("evaluate should succeed");
    assert_eq!(
        output1.tensor.owned_values, output2.tensor.owned_values,
        "evaluate must be deterministic"
    );

    // Nonzero: real executor must produce at least one nonzero element
    let nonzero_count = output1.tensor.owned_values.iter()
        .filter(|&&v| v.abs() > 1e-18)
        .count();
    assert!(
        nonzero_count > 0,
        "at least one element must be nonzero after real executor swap"
    );

    // Existing invariants (unchanged):
    assert!(!output1.tensor.owned_values.is_empty());
    assert_eq!(
        output1.tensor.owned_values.len(),
        output1.tensor.extents.iter().product::<usize>()
    );
    assert_eq!(
        output1.bytes_written,
        output1.tensor.owned_values.len() * std::mem::size_of::<f64>()
    );
    assert!(output1.stats.transfer_bytes > 0);
    assert_eq!(output1.workspace_bytes, query1_workspace_bytes);  // query1_workspace_bytes saved before consume
    assert_eq!(output1.chunk_count, query1_chunk_count);
}
```

The `sample_basis` helper at `api.rs` lines 626-628 produces a 2-shell BasisSet with `shell_l=1` (p-type) — sufficient for `int1e_ovlp_cart` (OperatorId::new(0)). Keep using it unchanged.

**Anti-patterns to avoid:**
- Do NOT assert `owned_values[0] == 1.0` (that assertion existed solely to detect the synthetic `idx+1` fill).
- Do NOT add `cintx-compat` as a test dependency inside `cintx-rs` (D-05: no inline `eval_raw` cross-check).

---

### `crates/cintx-oracle/tests/safe_api_arity2_parity.rs` (new test file)

**Analog:** `crates/cintx-oracle/tests/one_electron_parity.rs` (entire file — direct pattern source)

**Module-level gate** (matches `one_electron_parity.rs` line 35 and `center_2c2e_parity.rs` line 26):
```rust
#![cfg(any(feature = "cpu", feature = "rocm"))]
```

**Imports block** (mirrors `one_electron_parity.rs` lines 37-40, plus safe-API additions):
```rust
use cintx_compat::raw::{
    ATM_SLOTS, ANG_OF, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF,
    PTR_COEFF, PTR_COORD, PTR_EXP, PTR_ENV_START, PTR_ZETA, NUC_MOD_OF, POINT_NUC,
};
use cintx_core::{OperatorId, Representation};
use cintx_rs::{SessionRequest};
use cintx_runtime::ExecutionOptions;
```

Note: import `PTR_ENV_START` from `cintx_compat::raw` — required because the safe-API parity test must also build the raw `atm`/`bas`/`env` arrays for the vendor reference path, and the `center_2c2e_parity.rs` pattern (line 30) shows that 2c2e tests must use the `PTR_ENV_START=20` env alignment.

**Tolerance declaration** (at top of each test or as file-level constants, per D-09):
```rust
let atol = 1e-12_f64;
let rtol = 0.0_f64;
```

**H2O STO-3G raw fixture** — copy `build_h2o_sto3g()` verbatim from `center_2c2e_parity.rs` lines 37-175 (the version that uses `PTR_ENV_START`), not from `one_electron_parity.rs` (which omits `PTR_ENV_START` alignment and works only for 1e). The 2c2e operators require the `PTR_ENV_START=20` env alignment or range-separated Coulomb kernels produce wrong values. Using the `center_2c2e_parity.rs` version is safe for all 12 operators.

**Safe BasisSet construction helper** (mirrors `api.rs` tests lines 603-624, adapted to H2O STO-3G):
```rust
// Source: api.rs lines 603-605, 607-623
fn arc_f64(values: &[f64]) -> std::sync::Arc<[f64]> {
    std::sync::Arc::from(values.to_vec().into_boxed_slice())
}

fn build_h2o_sto3g_safe_basis(rep: Representation) -> (cintx_core::BasisSet, Vec<std::sync::Arc<cintx_core::Shell>>) {
    use cintx_core::{Atom, BasisSet, NuclearModel, Shell};
    use std::sync::Arc;

    let atom_o  = Atom::try_new(8, [0.0, 0.0, 0.0],        NuclearModel::Point, None, None).unwrap();
    let atom_h1 = Atom::try_new(1, [0.0, 1.4307, 1.1078],  NuclearModel::Point, None, None).unwrap();
    let atom_h2 = Atom::try_new(1, [0.0, -1.4307, 1.1078], NuclearModel::Point, None, None).unwrap();
    let atoms = Arc::from(vec![atom_o, atom_h1, atom_h2].into_boxed_slice());

    // STO-3G values must match build_h2o_sto3g() exactly (same source: Hehre/Stewart/Pople 1969)
    let shell_o1s = Arc::new(Shell::try_new(
        0, 0, 3, 1, 0, rep,
        arc_f64(&[130.7093200, 23.8088610, 6.4436083]),
        arc_f64(&[0.15432897, 0.53532814, 0.44463454]),
    ).unwrap());
    let shell_o2s = Arc::new(Shell::try_new(
        0, 0, 3, 1, 0, rep,
        arc_f64(&[5.0331513, 1.1695961, 0.3803890]),
        arc_f64(&[-0.09996723, 0.39951283, 0.70011547]),
    ).unwrap());
    let shell_o2p = Arc::new(Shell::try_new(
        0, 1, 3, 1, 0, rep,
        arc_f64(&[5.0331513, 1.1695961, 0.3803890]),
        arc_f64(&[0.15591627, 0.60768372, 0.39195739]),
    ).unwrap());
    let shell_h1_1s = Arc::new(Shell::try_new(
        1, 0, 3, 1, 0, rep,
        arc_f64(&[3.4252509, 0.6239137, 0.1688554]),
        arc_f64(&[0.15432897, 0.53532814, 0.44463454]),
    ).unwrap());
    let shell_h2_1s = Arc::new(Shell::try_new(
        2, 0, 3, 1, 0, rep,
        arc_f64(&[3.4252509, 0.6239137, 0.1688554]),
        arc_f64(&[0.15432897, 0.53532814, 0.44463454]),
    ).unwrap());

    let shells = vec![shell_o1s, shell_o2s, shell_o2p, shell_h1_1s, shell_h2_1s];
    let basis = BasisSet::try_new(atoms, Arc::from(shells.clone().into_boxed_slice())).unwrap();
    (basis, shells)
}
```

**`collect_safe_api_matrix` helper** (new, mirrors `collect_1e_sph_matrix` from `one_electron_parity.rs` lines 214-263 but uses `SessionRequest::evaluate`):
```rust
fn collect_safe_api_matrix(
    operator_id: OperatorId,
    rep: Representation,
    basis: &cintx_core::BasisSet,
    shells: &[std::sync::Arc<cintx_core::Shell>],
) -> Vec<f64> {
    use cintx_core::ShellTuple;
    let shell_tuple = ShellTuple::try_from_iter(shells.to_vec()).unwrap();
    let request = SessionRequest::new(
        operator_id,
        rep,
        basis,
        shell_tuple,
        ExecutionOptions::default(),
    );
    let query = request.query_workspace().expect("query_workspace should succeed");
    let output = query.evaluate().expect("evaluate should succeed");
    output.tensor.owned_values
}
```

**count_mismatches helper** — copy verbatim from `one_electron_parity.rs` lines 271-292:
```rust
fn count_mismatches(reference: &[f64], observed: &[f64], atol: f64, rtol: f64) -> usize {
    assert_eq!(reference.len(), observed.len(), "output length mismatch: {} vs {}", ...);
    let mut mismatches = 0usize;
    for (i, (&ref_val, &obs_val)) in reference.iter().zip(observed.iter()).enumerate() {
        let diff = (obs_val - ref_val).abs();
        let threshold = atol + rtol * ref_val.abs();
        if diff > threshold {
            mismatches += 1;
            eprintln!("  MISMATCH at index {i}: reference={ref_val:.15e}, observed={obs_val:.15e}, diff={diff:.3e}, threshold={threshold:.3e}");
        }
    }
    mismatches
}
```

**Per-symbol vendor-parity test pattern** (mirrors `one_electron_parity.rs` lines 541-573 — the `_vendor_parity` pattern):
```rust
#[test]
#[cfg(has_vendor_libcint)]
fn test_int1e_ovlp_sph_safe_api_parity() {
    let (atm, bas, env) = build_h2o_sto3g();  // raw arrays for vendor FFI reference
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Spheric);
    let atol = 1e-12_f64;
    let rtol = 0.0_f64;

    let safe_matrix = collect_safe_api_matrix(
        OperatorId::new(1),  // int1e_ovlp_sph
        Representation::Spheric,
        &basis,
        &shells,
    );
    let vendor_matrix = collect_1e_sph_matrix_vendor("ovlp", &atm, &bas, &env);

    let mismatches = count_mismatches(&vendor_matrix, &safe_matrix, atol, rtol);
    assert_eq!(
        mismatches, 0,
        "int1e_ovlp_sph safe API: {mismatches} elements exceed atol={atol:.0e}/rtol={rtol:.0e} vs vendored libcint"
    );
}
```

**`collect_1e_sph_matrix_vendor` / `collect_cart_matrix_vendor` helpers** — copy from `one_electron_parity.rs` lines 477-534 for the sph 1e vendor helper. For cart operators, use `vendor_int1e_ovlp_cart`, `vendor_int1e_kin_cart`, `vendor_int1e_nuc_cart` from `cintx-oracle::vendor_ffi` (confirmed present at `src/vendor_ffi.rs` lines 301, 327, 353). For `int2c2e_{sph,cart}`, use `vendor_int2c2e_sph` (line 142) and `vendor_int2c2e_cart` (line 412). Cart AO count uses `vendor_CINTcgto_cart` (line 535) instead of `nsph`.

**Spinor availability:** No spinor vendor FFI wrappers exist for `int1e_*_spinor` or `int2c2e_spinor` in `src/vendor_ffi.rs`. For the 6 spinor tests, use the idempotency-only strategy (two safe API calls agree) rather than vendor comparison. Add a `// TODO: add vendor vendor_int1e_*_spinor wrappers when available` comment inside the `#[cfg(has_vendor_libcint)]` guard or use a separate non-vendor idempotency path for the spinor cases.

**OperatorId mapping** (from `api_manifest.rs` — VERIFIED):
```
int1e_ovlp_cart     OperatorId::new(0)
int1e_ovlp_sph      OperatorId::new(1)
int1e_ovlp_spinor   OperatorId::new(2)
int1e_kin_cart      OperatorId::new(3)
int1e_kin_sph       OperatorId::new(4)
int1e_kin_spinor    OperatorId::new(5)
int1e_nuc_cart      OperatorId::new(6)
int1e_nuc_sph       OperatorId::new(7)
int1e_nuc_spinor    OperatorId::new(8)
int2c2e_cart        OperatorId::new(12)
int2c2e_sph         OperatorId::new(13)
int2c2e_spinor      OperatorId::new(14)
```

Note: `int2e_{cart,sph,spinor}` occupy IDs 9-11 (arity-4, not in Phase 17 scope).

**Anti-patterns to avoid:**
- Do NOT modify `one_electron_parity.rs`, `center_2c2e_parity.rs`, or any existing oracle test file (D-06).
- Do NOT call `eval_raw` from `collect_safe_api_matrix` (the safe API path is the system under test).
- Do NOT use a single parametric loop over the 12 operators — 12 named `#[test]` functions are required for per-symbol CI failure messages (D-07).
- Do NOT compare `owned_values` directly against vendor output without applying the column-major→row-major conversion for vendor FFI results (Pitfall 3: vendor returns Fortran order `out[j*ni + i]`; convert to row-major before comparison using the same loop as `collect_1e_sph_matrix_vendor`).
- Do NOT use the `one_electron_parity.rs` version of `build_h2o_sto3g()` (which omits `PTR_ENV_START`) for 2c2e tests — use the `center_2c2e_parity.rs` version that initializes `env = vec![0.0_f64; PTR_ENV_START]` first.

---

### `crates/cintx-oracle/Cargo.toml` (add cintx-rs dependency)

**Analog:** `crates/cintx-rs/Cargo.toml` lines 9-14 (path-dep declaration pattern)

**Current state** (`cintx-oracle/Cargo.toml` lines 23-28 — no cintx-rs):
```toml
[dependencies]
anyhow = "1.0.102"
cintx-compat = { path = "../cintx-compat" }
cintx-core = { path = "../cintx-core" }
cintx-ops = { path = "../cintx-ops" }
serde_json = "1.0.145"
```

**Change to make** — add one line to `[dependencies]`:
```toml
cintx-rs = { path = "../cintx-rs", default-features = false }
```

`default-features = false` matches the pattern in `cintx-rs/Cargo.toml` line 12 where `cintx-compat` is declared with `default-features = false`. The `cpu` backend is transitively activated through `cintx-oracle/cpu` → `cintx-compat/cpu` → `cintx-cubecl/cpu` — `cintx-rs` does not need its own `cpu` feature because it relies on `cintx-cubecl` which is already in the transitive chain.

**Feature forwarding note:** No new feature entries are needed in `[features]` — the existing `cpu = ["cintx-compat/cpu"]` chain already reaches `cintx-cubecl/cpu` which is what `CubeClExecutor` needs.

---

## Shared Patterns

### Module gate for oracle test files
**Source:** `crates/cintx-oracle/tests/one_electron_parity.rs` line 35 and `center_2c2e_parity.rs` line 26
**Apply to:** `safe_api_arity2_parity.rs` (new file)
```rust
#![cfg(any(feature = "cpu", feature = "rocm"))]
```
This ensures the file compiles only when at least one backend feature is active. The CI `oracle_parity_gate` runs with `--features cpu`.

### Vendor libcint cfg guard
**Source:** `crates/cintx-oracle/tests/one_electron_parity.rs` lines 476, 542, 580
**Apply to:** All 12 per-symbol tests in `safe_api_arity2_parity.rs` that call vendor FFI
```rust
#[cfg(has_vendor_libcint)]
```
Place on each `#[test]` fn that calls a `vendor_ffi::vendor_*` function. Tests without vendor comparison (spinor tests using idempotency-only strategy) do NOT need this guard — they should compile and run even without the vendor build.

### Column-major → row-major conversion for vendor FFI output
**Source:** `crates/cintx-oracle/tests/one_electron_parity.rs` lines 520-526
**Apply to:** All `collect_*_matrix_vendor` helpers in `safe_api_arity2_parity.rs`
```rust
// libcint 1e output is column-major (Fortran order): out[j*ni + i]
// Convert to row-major for our matrix layout
for ii in 0..ni {
    for jj in 0..nj {
        matrix[(row_offset + ii) * n_ao + (col_offset + jj)] = out[jj * ni + ii];
    }
}
```

### Phase 15 unified tolerance
**Source:** CONTEXT.md D-09 / RESEARCH.md Pattern 3
**Apply to:** All 12 per-symbol tests in `safe_api_arity2_parity.rs`
```rust
let atol = 1e-12_f64;
let rtol = 0.0_f64;
```
Do NOT copy the `atol = 1e-11_f64; rtol = 1e-9_f64;` values from `one_electron_parity.rs` (those predate Phase 15 unification). The new file adopts the tighter unified tolerance from day one.

### path-dep with default-features = false
**Source:** `crates/cintx-rs/Cargo.toml` line 12
**Apply to:** `crates/cintx-oracle/Cargo.toml` new dependency entry
```toml
cintx-rs = { path = "../cintx-rs", default-features = false }
```

---

## No Analog Found

None. All four file changes have direct analogs in the codebase.

---

## Metadata

**Analog search scope:**
- `crates/cintx-rs/src/api.rs` (production source + tests)
- `crates/cintx-compat/src/raw.rs` (real executor dispatch pattern)
- `crates/cintx-oracle/tests/one_electron_parity.rs` (per-symbol parity test pattern)
- `crates/cintx-oracle/tests/center_2c2e_parity.rs` (2c2e + PTR_ENV_START pattern)
- `crates/cintx-oracle/src/vendor_ffi.rs` (vendor FFI availability audit)
- `crates/cintx-oracle/Cargo.toml` (current dependency declarations)
- `crates/cintx-rs/Cargo.toml` (path-dep declaration pattern)

**Files scanned:** 7 source files read in full or targeted sections
**Pattern extraction date:** 2026-05-12
