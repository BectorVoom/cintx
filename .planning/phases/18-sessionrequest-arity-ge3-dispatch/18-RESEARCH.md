# Phase 18: SessionRequest Arity ≥3 Dispatch — Research

**Researched:** 2026-05-12
**Domain:** Rust safe-API arity-generic dispatch + libcint F-order AO layout + aosym typed-error scaffolding
**Confidence:** HIGH (with one MEDIUM finding on D-06 operator scope and one MEDIUM finding on D-09 component_rank)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**D-01:** Ship `s1` only; every other aosym packing returns a typed error.
**D-02:** Add `aosym: Option<AoSymmetry>` to `ExecutionOptions`. Additive field; `None` ≡ `Some(S1)`.
**D-03:** `AoSymmetry` enum lives in `cintx-core::operator`. Variants: `S1, S2ij, S2kl, S4, S8`. Re-exported from `cintx-core::lib`. Any non-`S1` is rejected for every arity.
**D-04:** New `FacadeError::UnsupportedAoSymmetry { requested: String }`, raised in `SessionRequest::query_workspace` fail-fast before any kernel work. Distinct from `UnsupportedApi`.
**D-05:** `aosym_error_path` per-variant unit test in `cintx-rs/src/api.rs`'s existing `#[cfg(test)]` module. No vendor libcint dependency.
**D-06:** Exact 12 symbols oracle-verified at atol=1e-12:
  - Arity-3 (8): `int3c1e_cart`, `int3c1e_sph`, `int3c1e_p2_cart`, `int3c1e_p2_sph`, `int3c2e_ip1_cart`, `int3c2e_ip1_sph`, `int3c2e_cart`, `int3c2e_sph`
  - Arity-4 (4): `int2e_cart`, `int2e_sph`, `int4c1e_cart`, `int4c1e_sph`
**D-07:** Spinor arity-3/4 (`int2e_spinor`, `int3c2e_ip1_spinor`, `int3c2e_spinor`) is "compiled but unverified". Documented in module rustdoc but NOT byte-identity-gated this phase.
**D-08:** Unstable-source arity-3 symbols out of scope.
**D-09:** `int3c2e_ip1_*` (component_rank=3) uses the same scalar parity helper as the other arity-3 tests; flat-buffer byte-identity, no per-component decomposition. Two extra tests (cart + sph).
**D-10:** F-order AO layout documented as rustdoc on `IntegralTensor` only. No struct field, no `view_fortran()`, no `ndarray` dep.
**D-11:** Layout contract verified implicitly by the oracle parity sweep. No dedicated layout-only test.
**D-12:** Two new test files split by arity: `safe_api_arity3_parity.rs` (8 tests), `safe_api_arity4_parity.rs` (4 tests). `int4c1e_*` tests individually gated `#[cfg(feature = "with-4c1e")]`. All under `#[cfg(has_vendor_libcint)]`.
**D-13:** Fixture: H2O / STO-3G via `crates/cintx-oracle/src/fixtures.rs::build_h2o_sto3g` (the PTR_ENV_START-aware version). 5 shells total.
**D-14:** Full Cartesian product of shell tuples: 125 arity-3 (5³) and 625 arity-4 (5⁴) per test. Fall back to deterministic subset if CI cost exceeds budget.
**D-15:** Tolerance, cfg, and CI mirror Phase 17 — atol=1e-12, rtol=0.0, `#[cfg(has_vendor_libcint)]`, existing `oracle_parity_gate` matrix.

### Claude's Discretion

- F-order doc string MUST be verified against actual planner layout before drafting.
- Shared `collect_safe_api_matrix(operator, repr, &basis, tuple)` helper — default yes, lives in `crates/cintx-oracle/tests/common/mod.rs` or `safe_api_helpers.rs`. Planner picks exact module and signature.
- Spinor arity-3/4 smoke test (no oracle compare) — default no.
- `AoSymmetry` derive set: likely `Clone, Copy, Debug, PartialEq, Eq, Hash`, with `Default` returning `S1`. Decide during planning.
- F-order rustdoc location: struct docblock on `IntegralTensor` only; module preamble may cross-reference.
- AoSymmetry variant naming: `S1, S2ij, S2kl, S4, S8` (preserve pyscf's lowercase suffix capitalized). `Display` impl emits lowercase pyscf form (`s1, s2ij, ...`).

### Deferred Ideas (OUT OF SCOPE)

- aosym packings `s2ij`, `s2kl`, `s4`, `s8` implementations.
- Spinor arity-3/4 oracle parity sweep.
- Unstable-source arity-3 symbols through `SessionRequest`.
- `view_fortran()` ndarray-backed view on `IntegralTensor`.
- `TensorLayout` enum field on `IntegralTensor`.
- Shared chunk-loop helper between safe API and compat raw path.
- Multi-fixture parity sweep (heavy-atom 4c1e).
- Cross-fixture spinor arity-3/4 parity once D-07 lifts.

</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ARITY-01 | `SessionRequest::evaluate` accepts arity-3 `(i,j,k)` and arity-4 `(i,j,k,l)` shell tuples and routes them to existing `cintx-ops` resolver entries — no parallel evaluator API. | `ShellTuple::try_from_iter` accepts up to `SHELL_TUPLE_CAPACITY=4` shells [VERIFIED: crates/cintx-core/src/shell.rs:7]. `Resolver::descriptor` already covers all arity-3/4 operators in `OPERATOR_DESCRIPTORS` [VERIFIED: api_manifest.rs lines 163-417]. `SessionQuery::evaluate` chunk loop is arity-generic (no hard-coded arity check between lines 119-256 of api.rs) [VERIFIED: api.rs read in full]. `CubeClExecutor::supports` accepts all of 1e/2e/2c2e/3c1e/3c2e and 4c1e (with feature) [VERIFIED: kernels/mod.rs:50, 198-225]. |
| ARITY-02 | Arity-3 operators `int3c1e`, `int3c1e_p2`, `int3c2e_ip1`, `int3c2e_{sph,cart}` and arity-4 operators `int2e_{sph,cart}`, `int4c1e_{sph,cart}` round-trip through the safe API with byte-identity vs libcint at atol=1e-12. | Vendor wrappers exist in `cintx-oracle::vendor_ffi` for the sph and cart variants of every Phase 18 operator EXCEPT `int3c1e_p2_sph`, `int3c2e_ip1_sph`, and `int3c2e_cart` [VERIFIED: see Code Investigation §item 4]. The plain `int3c2e_{sph,cart}` symbols listed in D-06 do NOT exist as operator-kind entries in the manifest [SEE Risk Analysis §R1]. |
| ARITY-03 | Output tensors expose F-order AO axes consistent with libcint memory layout. | `build_output_layout` sets `extents = shells.map(ao_per_shell)` (shell-tuple order) and `component_axis_leading = true` [VERIFIED: planner.rs:260-296]. cintx 2e/3c1e/3c2e/4c1e kernels write F-order, agreeing directly with vendor output (no transpose needed) [VERIFIED: compare.rs:787-797 (2e), :811-821 (3c1e_sph), :823-833 (3c2e_ip1_sph)]. cintx 1e/2c2e kernels write ROW-major and vendor is transposed for arity-2 parity [VERIFIED: safe_api_arity2_parity.rs:282 reads `pair_values[ii * nj + jj]`, vendor helpers transpose at lines 396-400]. **This means the F-order rustdoc D-10 is correct for arity ≥ 3 only — arity-2 IntegralTensor is currently row-major, contradicting a single uniform F-order claim.** [SEE Risk Analysis §R2] |
| ARITY-04 | aosym follows pyscf's convention; `s1` works, every other variant returns a typed error. | `ExecutionOptions::f12_zeta: Option<f64>` is the pattern reference [VERIFIED: options.rs:103-117]. `FacadeError` is `thiserror::Error` enum suitable for a new variant [VERIFIED: error.rs:14-24]. Per-variant unit test pattern lives in `cintx-rs/src/api.rs::tests` and does not depend on cintx-oracle [VERIFIED: api.rs:488-619]. |
| ARITY-05 | Oracle parity tests for arity-3/4 dispatch are added to `cintx-oracle` and gate CI alongside arity-2. | `cintx-oracle` already depends on `cintx-rs` and `cintx-runtime` (added Phase 17, line 28-29 of Cargo.toml) [VERIFIED]. Existing oracle_parity_gate matrix runs all tests under `--features cpu` [VERIFIED: Phase 17 verification report]. `#[cfg(has_vendor_libcint)]` gate is the established pattern [VERIFIED: 8 cart/sph tests in safe_api_arity2_parity.rs]. `#[cfg(feature = "with-4c1e")]` per-test gating is established [VERIFIED: oracle_gate_closure.rs:738, :799]. |

</phase_requirements>

---

## Phase Summary

Phase 18 extends `SessionRequest::evaluate` from arity-2 (Phase 17) to arity-3 and arity-4 shell tuples. The dispatch primitives (`CubeClExecutor`, `Resolver::descriptor`, `ShellTuple` with `SHELL_TUPLE_CAPACITY=4`, `build_output_layout`, `schedule_chunks`) are already arity-generic — no code path changes are required to make arity-3/4 dispatch *work*. The deliverables are: (a) twelve per-symbol oracle parity tests across two new files, (b) an `aosym: Option<AoSymmetry>` knob on `ExecutionOptions` that ships only `S1` and returns a typed `FacadeError::UnsupportedAoSymmetry` for every other variant, (c) the new `AoSymmetry` enum in `cintx-core::operator` re-exported through `cintx-rs::prelude`, and (d) an F-order rustdoc invariant on `IntegralTensor`. Two findings in this research surface as planner-level risks (R1: D-06 lists `int3c2e_{cart,sph}` which do not exist in the manifest; R2: cintx-rs IntegralTensor layout is row-major for arity-2 today and only F-order for arity ≥ 3). The first is a scope correction; the second is a doc-string-wording adjustment.

**Primary recommendation:** Extract a shared `collect_safe_api_matrix(operator, repr, basis, tuple)` helper into `crates/cintx-oracle/tests/safe_api_helpers.rs`, refactor `safe_api_arity2_parity.rs` to use it (out of scope per D-03? — no, that was Phase 17. Phase 18 is allowed to factor common helpers), reduce D-06's arity-3 set from 8 to 6 (drop `int3c2e_cart`/`int3c2e_sph` per R1), and word the IntegralTensor rustdoc carefully to acknowledge the arity-dependent layout.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Arity-3/4 integral dispatch | API / Backend (`cintx-cubecl::CubeClExecutor`) | — | Already arity-generic; kernels in `cintx-cubecl/src/kernels/` own the family-specific compute |
| Safe API facade arity-generic chunk loop | API (`cintx-rs::api`) | — | Unchanged from Phase 17 — `SessionQuery::evaluate` already loops over any shell-tuple arity |
| `aosym` knob validation | API (`cintx-rs::api::SessionRequest::query_workspace`) | Domain (`cintx-core::operator::AoSymmetry`) | Fail-fast preflight is sibling to existing `enforce_safe_facade_policy_gate`, not a modification of it |
| `AoSymmetry` enum and `Display` impl | Domain (`cintx-core::operator`) | — | Lives next to `Representation`, re-exported via `cintx-core::lib` and `cintx-rs::prelude` |
| `ExecutionOptions::aosym` field | Runtime (`cintx-runtime::options`) | — | Mirrors `f12_zeta` pattern; struct is SemVer-additive |
| `FacadeError::UnsupportedAoSymmetry` variant | API (`cintx-rs::error`) | — | New variant on existing `thiserror`-derived enum |
| F-order layout rustdoc | API (`cintx-rs::api::IntegralTensor`) | — | Documents the planner's `build_output_layout` contract; no struct field added |
| Arity-3/4 parity oracle | Test (`cintx-oracle/tests/safe_api_arity{3,4}_parity.rs`) | — | All byte-identity checks vs vendored libcint 6.1.3 |
| CI gate | CI (`oracle_parity_gate` matrix) | — | Existing matrix; new tests run inside it without a new job |

---

## Standard Stack

### Core (no new dependencies)

| Library / Crate | Version | Purpose | Why Standard |
|-----------------|---------|---------|--------------|
| `cintx-cubecl` (workspace) | path dep, `CubeClExecutor` re-exported at lib.rs:26 | Real executor + kernel dispatch for arity-3/4 | Phase 17 wired it; arity-generic by construction |
| `cintx-rs` (workspace) | path dep on `cintx-cubecl`, `cintx-compat`, `cintx-core`, `cintx-runtime`, `cintx-ops`; depends on `thiserror = "2"` [VERIFIED: Cargo.toml] | Safe API surface receiving the `aosym` knob and `UnsupportedAoSymmetry` variant | Already the canonical safe-API crate |
| `cintx-core` (workspace) | path dep | Hosts `AoSymmetry` enum (D-03) next to `Representation` | Domain primitives live here; `Representation` is the immediate neighbor |
| `cintx-runtime` (workspace) | path dep | Hosts `ExecutionOptions` (D-02) | `f12_zeta` is the reference pattern |
| `cintx-oracle` (workspace) | path dep, already depends on `cintx-rs` and `cintx-runtime` since Phase 17 [VERIFIED: Cargo.toml lines 28-29] | Hosts the new parity test files | No Cargo.toml change required |
| `thiserror` | 2 [VERIFIED: cintx-rs/Cargo.toml:15] | New `FacadeError::UnsupportedAoSymmetry` variant | Project-wide library error library per CLAUDE.md |

### Supporting (existing, no change)

| Library | Purpose | Notes |
|---------|---------|-------|
| `bindgen` 0.71.1 | Vendor FFI binding generation | Already generates the wrappers Phase 18 uses |
| `cc` 1.2.x | Vendored libcint 6.1.3 build | Already generates `has_vendor_libcint` cfg |

**No new dependencies are required.** Phase 17 added `cintx-rs` and `cintx-runtime` as direct deps of `cintx-oracle`.

---

## Architecture Patterns

### System Architecture Diagram

```
SessionRequest::evaluate (cintx-rs/src/api.rs)
    |
    |-- SessionRequest::query_workspace
    |   |-- aosym preflight (NEW, D-04)
    |   |   if options.aosym not in {None, Some(S1)}:
    |   |       return FacadeError::UnsupportedAoSymmetry { requested: aosym.to_string() }
    |   |
    |   |-- enforce_safe_facade_policy_gate (cintx-compat) -- rejects bad operators (unchanged)
    |   |-- runtime_query_workspace (cintx-runtime) -- builds WorkspaceQuery
    |
    |-- SessionQuery::evaluate  -- arity-generic chunk loop (unchanged from Phase 17)
    |   |-- ExecutionPlan::new (cintx-runtime)
    |   |   |-- build_output_layout (planner.rs:260-296)
    |   |   |    extents = [shells[0].ao_per_shell(), ..., shells[N-1].ao_per_shell()]
    |   |   |    component_axis_leading = true
    |   |   |-- DispatchDecision::from_manifest_family
    |   |-- CubeClExecutor [REAL, cintx-cubecl]
    |   |   |-- supports(&plan)             -- supports_canonical_family for 1e/2e/2c2e/3c1e/3c2e/(4c1e)
    |   |   |-- query_workspace(&plan)      -- backend f64 capability check
    |   |   |-- for each chunk:
    |   |       |-- ExecutionIo::new (cintx-runtime)
    |   |       |-- executor.execute(&plan, &mut io)
    |   |           |-- kernels::launch_family (1e/2e/2c2e/3c1e/3c2e/4c1e)
    |   |           |-- transform::apply_representation_transform (cart→sph for non-spinor)
    |   |
    |   |-- accumulate chunk_staging into owned_values
    |   |-- return TypedEvaluationOutput { tensor: IntegralTensor { extents, owned_values, ... }, stats }
```

The chunk-loop body in `SessionQuery::evaluate` (api.rs:182-242) is unchanged from Phase 17. The only Phase 18 modification on the dispatch path is the aosym preflight in `SessionRequest::query_workspace`.

### Recommended Project Structure (Phase 18 changes only)

```
crates/
├── cintx-core/
│   ├── src/operator.rs            # ADD AoSymmetry enum (Clone, Copy, Debug, PartialEq, Eq, Hash, Default→S1, Display→lowercase pyscf)
│   └── src/lib.rs                 # ADD re-export `pub use operator::AoSymmetry;` on line ~19
├── cintx-runtime/
│   └── src/options.rs             # ADD `pub aosym: Option<AoSymmetry>` field on ExecutionOptions (line ~117); update lib.rs re-export if needed
├── cintx-rs/
│   ├── src/api.rs                 # ADD aosym preflight in SessionRequest::query_workspace; ADD F-order rustdoc on IntegralTensor (line 441)
│   ├── src/error.rs               # ADD FacadeError::UnsupportedAoSymmetry { requested: String } variant
│   ├── src/builder.rs             # OPTIONAL: ADD `pub fn aosym(self, aosym: AoSymmetry)` builder setter
│   └── src/prelude.rs             # ADD `pub use cintx_core::AoSymmetry;`
├── cintx-oracle/
│   └── tests/
│       ├── safe_api_helpers.rs    # NEW (optional, Claude's discretion) — shared collect_safe_api_matrix
│       ├── safe_api_arity3_parity.rs   # NEW — 6 or 8 per-symbol parity tests (see R1)
│       └── safe_api_arity4_parity.rs   # NEW — 4 per-symbol parity tests; int4c1e_* gated #[cfg(feature = "with-4c1e")]
```

### Pattern 1: AoSymmetry enum in `cintx-core::operator`

```rust
// Source: crates/cintx-core/src/operator.rs (PROPOSED — sibling to Representation at lines 4-20)
/// AO symmetry packing convention (pyscf-compatible naming).
///
/// Phase 18 ships `S1` only; every other variant is currently rejected with
/// `FacadeError::UnsupportedAoSymmetry`. See `.planning/phases/18-*/` for status.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum AoSymmetry {
    #[default]
    S1,    // no packing (full ni × nj × nk × nl tensor)
    S2ij,  // s2 on the (i, j) bra pair  — UNSUPPORTED in Phase 18
    S2kl,  // s2 on the (k, l) ket pair  — UNSUPPORTED in Phase 18
    S4,    // s2 on both pairs            — UNSUPPORTED in Phase 18
    S8,    // s8 (s4 + global bra↔ket)   — UNSUPPORTED in Phase 18
}

impl std::fmt::Display for AoSymmetry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AoSymmetry::S1 => write!(f, "s1"),
            AoSymmetry::S2ij => write!(f, "s2ij"),
            AoSymmetry::S2kl => write!(f, "s2kl"),
            AoSymmetry::S4 => write!(f, "s4"),
            AoSymmetry::S8 => write!(f, "s8"),
        }
    }
}
```

[VERIFIED: cintx-core/src/operator.rs:4-20 — Representation uses the same derive set + Display impl pattern. Phase 18's AoSymmetry follows identically.]

### Pattern 2: ExecutionOptions additive field

```rust
// Source: crates/cintx-runtime/src/options.rs:103-117 (PROPOSED additive change)
#[derive(Clone, Debug, Default)]
pub struct ExecutionOptions {
    pub memory_limit_bytes: Option<usize>,
    pub trace_span: Option<Span>,
    pub chunk_size_override: Option<usize>,
    pub profile_label: Option<&'static str>,
    pub backend_intent: BackendIntent,
    pub backend_capability_token: BackendCapabilityToken,
    pub f12_zeta: Option<f64>,
    /// AO symmetry packing requested by the caller. Phase 18 implements `S1` only;
    /// every other variant returns `FacadeError::UnsupportedAoSymmetry` from
    /// `SessionRequest::query_workspace`.
    pub aosym: Option<cintx_core::AoSymmetry>,  // ADDED
}
```

[VERIFIED: `f12_zeta: Option<f64>` is the established pattern at line 116 of options.rs. Phase 18 follows it.]

### Pattern 3: aosym preflight in query_workspace

```rust
// Source: crates/cintx-rs/src/api.rs:63-80 (PROPOSED — fail-fast preflight)
pub fn query_workspace(&self) -> Result<SessionQuery<'basis>, FacadeError> {
    // Phase 18: aosym preflight — accept None or Some(S1); reject everything else.
    if let Some(aosym) = self.options.aosym {
        if aosym != cintx_core::AoSymmetry::S1 {
            return Err(FacadeError::UnsupportedAoSymmetry {
                requested: aosym.to_string(),
            });
        }
    }

    let runtime_workspace = runtime_query_workspace(
        self.operator,
        self.representation,
        self.basis,
        self.shells.clone(),
        &self.options,
    )
    .map_err(FacadeError::from)?;
    // ...
}
```

[VERIFIED: `query_workspace` at lines 63-80; sibling to `enforce_safe_facade_policy_gate` policy gate at line 111 of api.rs. D-04 specifies query_workspace as the fail-fast site.]

### Pattern 4: FacadeError new variant

```rust
// Source: crates/cintx-rs/src/error.rs:14-24 (PROPOSED additive variant)
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum FacadeError {
    #[error("unsupported api: {requested}")]
    UnsupportedApi { requested: String },
    #[error("layout contract violation: {detail}")]
    Layout { detail: String },
    #[error("memory contract violation: {detail}")]
    Memory { detail: String },
    #[error("validation failed: {detail}")]
    Validation { detail: String },
    #[error("unsupported aosym packing: {requested}")]  // ADDED
    UnsupportedAoSymmetry { requested: String },        // ADDED
}
```

The corresponding `FacadeErrorKind::UnsupportedAoSymmetry` enum variant must also be added to keep `FacadeError::kind()` exhaustive [VERIFIED: error.rs:6-12, 27-34].

### Pattern 5: F-order rustdoc on IntegralTensor

```rust
// Source: crates/cintx-rs/src/api.rs:441-447 (PROPOSED rustdoc, no struct change)
/// Owned integral tensor returned by `SessionQuery::evaluate`.
///
/// # AO axis layout
///
/// `owned_values` stores AO data in **F-order** (Fortran / column-major):
/// `extents[0]` is the **fastest-varying** axis, `extents[N-1]` is the slowest-varying.
/// For arity ≥ 3 the layout is consistent with libcint's memory layout (the row-major
/// helpers in `cintx-oracle/tests/safe_api_arity2_parity.rs` apply only to arity-2 tests
/// — arity-3/4 byte-identity does NOT require transposition; see Risk Analysis §R2 in
/// `.planning/phases/18-*/18-RESEARCH.md`).
///
/// When `component_axis_leading == true` (the default), the optional component axis is
/// the **slowest-varying** axis (rank > shells.len()).
///
/// `complex_interleaved == true` for Spinor outputs: real and imaginary parts alternate
/// in the innermost stride of `owned_values`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IntegralTensor {
    pub extents: Vec<usize>,
    pub component_axis_leading: bool,
    pub complex_interleaved: bool,
    pub owned_values: Vec<f64>,
}
```

**Wording note:** the F-order claim is true for cintx kernels at arity ≥ 3, but NOT for the existing arity-2 path (1e/2c2e kernels currently write row-major; see Risk Analysis §R2). The rustdoc above explicitly acknowledges this so the doc string remains honest without misleading callers.

### Pattern 6: New oracle test file shape

```rust
// Source: crates/cintx-oracle/tests/safe_api_arity3_parity.rs (PROPOSED — mirrors arity2 file)
#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{ATM_SLOTS, ANG_OF, /* ... */};
use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell, ShellTuple};
use cintx_rs::SessionRequest;
use cintx_runtime::ExecutionOptions;
use std::sync::Arc;

// Re-use the existing build_h2o_sto3g (raw) and build_h2o_sto3g_safe_basis (typed)
// helpers — copy from safe_api_arity2_parity.rs lines 36-220, or factor to
// safe_api_helpers.rs (Claude's discretion).

/// OperatorId → symbol mapping for the arity-3 set:
///   int3c1e_p2_cart = 15, int3c1e_p2_sph = 16
///   int3c1e_cart    = 17, int3c1e_sph    = 18
///   int3c2e_ip1_cart= 19, int3c2e_ip1_sph= 20, int3c2e_ip1_spinor = 21
/// [VERIFIED: api_manifest.rs sequence at lines 265-383]

fn collect_safe_api_arity3_buffer(
    operator_id: OperatorId,
    rep: Representation,
    basis: &BasisSet,
    shells: &[Arc<Shell>],
    i: usize, j: usize, k: usize,
) -> Vec<f64> {
    let shell_tuple = ShellTuple::try_from_iter([
        shells[i].clone(), shells[j].clone(), shells[k].clone(),
    ]).expect("3-shell tuple within SHELL_TUPLE_CAPACITY=4");
    let request = SessionRequest::new(operator_id, rep, basis, shell_tuple, ExecutionOptions::default());
    let query = request.query_workspace().expect("query_workspace must succeed");
    let output = query.evaluate().expect("evaluate must succeed");
    output.tensor.owned_values
}

#[test]
#[cfg(has_vendor_libcint)]
fn test_int3c1e_sph_safe_api_parity() {
    let (atm, bas, env) = build_h2o_sto3g();
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Spheric);
    let atol = 1e-12_f64;
    let rtol = 0.0_f64;

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;

    let mut total_mismatches = 0usize;
    let mut any_nonzero = false;

    for i in 0..5 {
        for j in 0..5 {
            for k in 0..5 {
                let ni = shells[i].ao_per_shell();
                let nj = shells[j].ao_per_shell();
                let nk = shells[k].ao_per_shell();
                let n_elem = ni * nj * nk;

                let safe_out = collect_safe_api_arity3_buffer(
                    OperatorId::new(18), Representation::Spheric, &basis, &shells, i, j, k,
                );
                let mut vendor_out = vec![0.0_f64; n_elem];
                let shls = [i as i32, j as i32, k as i32];
                cintx_oracle::vendor_ffi::vendor_int3c1e_sph(
                    &mut vendor_out, &shls, &atm, natm, &bas, nbas, &env);

                if safe_out.iter().any(|&v| v.abs() > 1e-18)
                    || vendor_out.iter().any(|&v| v.abs() > 1e-18) {
                    any_nonzero = true;
                }
                // Arity-3 path: cintx writes F-order, agreeing with vendor directly
                // (no transpose). See compare.rs:811-821 for the established pattern.
                total_mismatches += count_mismatches(&vendor_out, &safe_out, atol, rtol);
            }
        }
    }

    assert!(any_nonzero, "int3c1e_sph safe-API outputs are all zeros (sentinel)");
    assert_eq!(total_mismatches, 0,
        "int3c1e_sph safe API: {total_mismatches} elements exceed atol=1e-12 vs vendored libcint over 125 triples");
}
```

### Pattern 7: with-4c1e feature-flag gating at #[test] level

```rust
// Source: crates/cintx-oracle/tests/oracle_gate_closure.rs:737-739 (established pattern)
#[test]
#[cfg(feature = "with-4c1e")]
#[cfg(has_vendor_libcint)]
fn test_int4c1e_sph_safe_api_parity() {
    // ... full Cartesian 5⁴ = 625 quartet sweep ...
}
```

Both cfg attributes stack additively. The test compiles only when both `with-4c1e` AND `has_vendor_libcint` are active.

### Anti-Patterns to Avoid

- **Modifying `enforce_safe_facade_policy_gate` to handle aosym.** D-04 is explicit: the aosym preflight is a sibling check in `query_workspace`, NOT a modification of `enforce_safe_facade_policy_gate` (which handles source/profile/F12/4c1e).
- **Adding the aosym preflight inside `SessionQuery::evaluate` instead of `query_workspace`.** D-04 demands fail-fast at query time so the caller never proceeds to plan construction with an invalid aosym.
- **Transposing arity-3/4 vendor output before comparison.** Existing arity-3/4 raw-path tests (compare.rs:787-833, center_3c2e_parity.rs:222-287, two_electron_parity.rs:273-289) compare cintx vs vendor DIRECTLY. The arity-2 transpose pattern in `safe_api_arity2_parity.rs:396-400` does NOT apply.
- **Using a single parametric loop over the 12 operators.** D-07 mandates 12 named `#[test]` functions for per-symbol CI bisection.
- **Adding a new CI job.** D-15 reuses the existing `oracle_parity_gate` matrix.
- **Adding a struct field or method on `IntegralTensor`.** D-10 limits the F-order contract to a rustdoc comment.
- **Hoisting the chunk loop into a shared crate.** Still deferred (CONTEXT.md `<deferred>`).
- **Adding `int3c2e_cart`/`int3c2e_sph` to the test set without first adding them to the manifest.** See Risk Analysis §R1.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Arity-3/4 integral kernels | New compute paths in the safe API | `cintx_cubecl::CubeClExecutor` (already wired by Phase 17) | Kernels are validated by raw-path oracle gates since Phase 10 |
| atm/bas/env construction from BasisSet | Typed→raw round-trip inside the safe API | Real executor consumes `ExecutionPlan` directly | Avoids the unsafe boundary explicitly rejected by Phase 17 D-02 |
| aosym packing logic (`s2ij`, `s4`, `s8`) | Anything beyond returning a typed error | `FacadeError::UnsupportedAoSymmetry` | D-01: Phase 18 ships none of the compressed packings |
| Vendor FFI wrappers for missing operators | Adding new `vendor_int3c1e_p2_sph` / `vendor_int3c2e_ip1_sph` / `vendor_int3c2e_cart` wrappers | Use existing `vendor_int3c1e_p2_cart` and `vendor_int3c2e_sph` (plain) wrappers where possible; otherwise add minimal new wrappers in cintx-oracle/src/vendor_ffi.rs following the pattern at lines 173-216 | See Code Investigation §item 4 for the gap audit |
| F-order layout enforcement | Test that asserts exact strides | Rely on byte-identity vs vendor (D-11) | Vendor IS the F-order reference; matching bytes = matching F-order |

---

## Code Investigation (10 verification items from CONTEXT.md "Claude's Discretion + Specific Ideas")

### Item 1: F-order layout verification (MUST DO per CONTEXT.md "Claude's Discretion")

**Verified files:**
- `crates/cintx-runtime/src/planner.rs:260-296` — `build_output_layout()` definition
- `crates/cintx-oracle/src/compare.rs:687-833` — established cintx-vs-vendor comparison patterns by arity
- `crates/cintx-oracle/tests/safe_api_arity2_parity.rs:282, 396-400` — arity-2 transpose pattern (cintx row-major, vendor column-major)
- `crates/cintx-oracle/tests/two_electron_parity.rs:273-289` — arity-4 direct comparison (no transpose)
- `crates/cintx-oracle/tests/center_3c2e_parity.rs:222-287` — arity-3 direct comparison (no transpose)

**Finding:**

`build_output_layout` (planner.rs:260-296) sets:
```rust
extents = shells.as_slice().iter().map(|s| s.ao_per_shell()).collect();   // shell-tuple order
component_axis_leading = true;
complex_interleaved = (representation == Spinor);  // 2× multiplier
staging_elements = base_elements × component_count × complex_multiplier;
```

So `extents[0] = ao_per_shell(shells[0])`, `extents[1] = ao_per_shell(shells[1])`, etc. — shell-tuple order, NOT axis-speed order.

The actual fastest-varying axis depends on **which kernel writes the buffer**:
- **1e / 2c2e kernels** (arity 2): write ROW-major — i.e., `extents[0]` (bra) is the SLOWEST-varying axis. Confirmed by `safe_api_arity2_parity.rs:282` reading `pair_values[ii * nj + jj]` (i-outer/slow, j-inner/fast) and vendor helpers at lines 396-400 transposing column-major vendor output into row-major.
- **2e / 3c1e / 3c2e / 4c1e kernels** (arity 3 and 4): write F-order (column-major) — i.e., `extents[0]` (bra) is the FASTEST-varying axis. Confirmed by `compare.rs:787-797` (2e direct compare), `compare.rs:811-821` (3c1e_sph direct compare), `compare.rs:823-833` (3c2e_ip1_sph direct compare against `vendor_int3c2e_sph`), `two_electron_parity.rs:289` (zero mismatches against vendor without transpose), `center_3c2e_parity.rs:283-286` (zero mismatches against vendor without transpose at atol=1e-9).

**Implication for D-10's rustdoc:** A blanket "leftmost is fastest-varying" rustdoc claim on `IntegralTensor` is FALSE for arity-2 today. Two acceptable paths:

(a) **Word the rustdoc carefully** — state that arity ≥ 3 outputs are F-order matching libcint, and document that arity-2 outputs follow the same shell-tuple ordering but use ROW-major within each shell-pair (the existing convention). Phase 18 documents this honestly without changing it. This is the LOW-RISK path and what the recommendation above (Pattern 5) implements.

(b) **Unify the kernel layouts** — change 1e/2c2e kernels to write F-order so the rustdoc invariant becomes simple. This is a **planner/kernel-level change OUT OF SCOPE for Phase 18** and would invalidate the existing arity-2 oracle (Phase 17) and `safe_api_arity2_parity.rs` test layout (row-major matrix assembly at line 282). DO NOT do this in Phase 18.

[VERIFIED via direct file reads — `compare.rs:687-833` and `safe_api_arity2_parity.rs:280-292, 396-400`.]

### Item 2: All 12 target operators routable today

**Verified files:**
- `crates/cintx-ops/src/generated/api_manifest.csv` (lines 11-25 for arity ≥ 3 operators)
- `crates/cintx-ops/src/generated/api_manifest.rs` (lines 163-417 — the `MANIFEST_ENTRIES` array)
- `crates/cintx-ops/src/resolver.rs:219-223` — `Resolver::descriptor` uses `OPERATOR_DESCRIPTORS.get(id.raw() as usize)`

**Finding:**

The manifest (CSV lines 11-25) and `api_manifest.rs` (lines 163-417) contain:

| OperatorId | Symbol | Arity | Family | Notes |
|------------|--------|-------|--------|-------|
| 9 | `int2e_cart` | 4 | 2e | ✓ in Phase 18 scope |
| 10 | `int2e_sph` | 4 | 2e | ✓ in Phase 18 scope |
| 11 | `int2e_spinor` | 4 | 2e | spinor — D-07 compiled-not-verified |
| 12-14 | `int2c2e_{cart,sph,spinor}` | 2 | 2c2e | (Phase 17, NOT Phase 18) |
| 15 | `int3c1e_p2_cart` | 3 | 3c1e | ✓ in Phase 18 scope |
| 16 | `int3c1e_p2_sph` | 3 | 3c1e | ✓ in Phase 18 scope |
| 17 | `int3c1e_cart` | 3 | 3c1e | ✓ in Phase 18 scope |
| 18 | `int3c1e_sph` | 3 | 3c1e | ✓ in Phase 18 scope |
| 19 | `int3c2e_ip1_cart` | 3 | 3c2e | ✓ in Phase 18 scope |
| 20 | `int3c2e_ip1_sph` | 3 | 3c2e | ✓ in Phase 18 scope |
| 21 | `int3c2e_ip1_spinor` | 3 | 3c2e | spinor — D-07 compiled-not-verified |
| 22 | `int4c1e_cart` | 4 | 4c1e | ✓ in Phase 18 scope, with-4c1e feature gate |
| 23 | `int4c1e_sph` | 4 | 4c1e | ✓ in Phase 18 scope, with-4c1e feature gate |

**Critical gap — `int3c2e_cart` and `int3c2e_sph` are NOT in the manifest.** D-06 lists 8 arity-3 symbols including the plain `int3c2e_cart` and `int3c2e_sph` (no `_ip1` suffix). The manifest only contains the `_ip1` variants. A `grep` for `symbol_name: "int3c2e_sph"` and `symbol_name: "int3c2e_cart"` in api_manifest.rs returns NOTHING (only `int3c2e_ip1_*` exists). Similarly in the CSV: `grep "int3c2e"` shows only `_ip1` and `_ssc` entries.

The current kernel for `RawApiId::INT3C2E_IP1_SPH` actually evaluates the plain `int3c2e_sph` (no derivative) — see Item 5 below.

**Implication for D-06's coverage list:** Phase 18 cannot byte-identity-test `int3c2e_cart` / `int3c2e_sph` symbols through `SessionRequest` because there is no corresponding `OperatorId` to construct. Either:

- **Drop those two entries from D-06**, leaving 6 arity-3 tests (`int3c1e_{cart,sph}`, `int3c1e_p2_{cart,sph}`, `int3c2e_ip1_{cart,sph}`) — recommended.
- **Add `int3c2e_{cart,sph}` to the manifest** as new operator-kind entries — this is a manifest change (substantial scope) and would also require updating Resolver / OperatorId numbering, which is OUT OF Phase 18 scope per D-08's "no manifest changes" implicit posture.

[VERIFIED: `grep -n "int3c2e" api_manifest.csv` returns 11 lines; none are plain `int3c2e_cart` or `int3c2e_sph` as operator-kind entries. `grep 'symbol_name: "int3c2e_sph"\|symbol_name: "int3c2e_cart"' api_manifest.rs` returns NOTHING.]

### Item 3: `ShellTuple::try_from_iter` with SHELL_TUPLE_CAPACITY=4

**Verified files:**
- `crates/cintx-core/src/shell.rs:7` — `pub(crate) const SHELL_TUPLE_CAPACITY: usize = 4;`
- `crates/cintx-core/src/shell.rs:121-134` — `ShellTuple::try_from_iter` body

**Finding:**

`SHELL_TUPLE_CAPACITY = 4` is the SmallVec inline-storage threshold for `ShellTuple`'s `SmallVec<[Arc<Shell>; SHELL_TUPLE_CAPACITY]>` field. `try_from_iter` accepts up to 4 shells before returning `ShellTupleArityError(4)`. **Arity-3 and arity-4 inputs work today without any modification.**

```rust
// crates/cintx-core/src/shell.rs:121-134
impl ShellTuple {
    pub fn try_from_iter<I>(iter: I) -> Result<Self, ShellTupleArityError>
    where I: IntoIterator<Item = Arc<Shell>>,
    {
        let mut shells = SmallVec::new();
        for shell in iter {
            if shells.len() >= SHELL_TUPLE_CAPACITY {
                return Err(ShellTupleArityError(SHELL_TUPLE_CAPACITY));
            }
            shells.push(shell);
        }
        Ok(Self { shells })
    }
}
```

Minor note: `SHELL_TUPLE_CAPACITY` is `pub(crate)`, not `pub`. The constant is not exposed to downstream crates — but they don't need it directly; they only need `try_from_iter` to accept their inputs.

[VERIFIED: shell.rs:7, 121-134 read in full.]

### Item 4: Arity-4 vendor helpers — int4c1e gap audit

**Verified files:**
- `crates/cintx-oracle/src/vendor_ffi.rs:99-100, 111-130, 198-219, 238-265, 269-296, 378-410, 437-462, 465-490, 493-518` — vendor wrapper inventory
- `crates/cintx-oracle/src/compare.rs:670-900` — existing arity-2/3/4 vendor comparison patterns

**Finding (vendor wrapper inventory for Phase 18's target operators):**

| Phase 18 symbol | Vendor wrapper exists? | Vendor function name | File:line |
|-----------------|------------------------|----------------------|-----------|
| `int3c1e_cart` | ✓ | `vendor_int3c1e_cart` | vendor_ffi.rs:440 |
| `int3c1e_sph` | ✓ | `vendor_int3c1e_sph` | vendor_ffi.rs:173 |
| `int3c1e_p2_cart` | ✓ | `vendor_int3c1e_p2_cart` | vendor_ffi.rs:468 |
| `int3c1e_p2_sph` | **✗ MISSING** | — | — |
| `int3c2e_ip1_cart` | ✓ | `vendor_int3c2e_ip1_cart` | vendor_ffi.rs:496 |
| `int3c2e_ip1_sph` | **✗ MISSING** | — | (existing center_3c2e_parity.rs uses `vendor_int3c2e_sph` (plain) as the reference for `RawApiId::INT3C2E_IP1_SPH` — see Item 5) |
| `int3c2e_cart` | **✗ MISSING** | — | not in manifest either (see Item 2) |
| `int3c2e_sph` | ✓ | `vendor_int3c2e_sph` | vendor_ffi.rs:204 | not in manifest as operator (see Item 2) |
| `int2e_cart` | ✓ | `vendor_int2e_cart` | vendor_ffi.rs:384 |
| `int2e_sph` | ✓ | `vendor_int2e_sph` | vendor_ffi.rs:111 |
| `int4c1e_cart` | ✓ | `vendor_int4c1e_cart` | vendor_ffi.rs:269 |
| `int4c1e_sph` | ✓ | `vendor_int4c1e_sph` | vendor_ffi.rs:238 |

**Two missing wrappers** must be added for D-06's full coverage:
1. `vendor_int3c1e_p2_sph` — mirror `vendor_int3c1e_p2_cart` (vendor_ffi.rs:468-490) but call `ffi::int3c1e_p2_sph`. The `int3c1e_p2_sph` symbol is already in the supplemental header (build.rs:227) so the binding exists. Add ~22 lines.
2. `vendor_int3c2e_ip1_sph` — mirror `vendor_int3c2e_ip1_cart` (vendor_ffi.rs:496-518) but call `ffi::int3c2e_ip1_sph`. `int3c2e_ip1_sph` is in `cint_funcs.h` (it's a `_sph` variant of a standard family) and should already have a binding. Verify by reading the generated `bindings.rs` (or rely on the `extern CINTIntegralFunction` declarations).

**`int4c1e_*` does NOT need new vendor helpers** — Phase 11 (`with-4c1e`) landed real kernels and `vendor_int4c1e_{sph,cart}` already exist (vendor_ffi.rs:238, 269). The existing helpers should suffice for Phase 18.

[VERIFIED via `grep -n "pub fn vendor_" vendor_ffi.rs` and inspection of lines 173-216, 198-216, 238-296, 437-518.]

### Item 5: `int3c2e_ip1_*` component_rank vs flat-buffer byte-identity (D-09)

**Verified files:**
- `crates/cintx-ops/src/generated/api_manifest.rs:336-383` — int3c2e_ip1_* entries
- `crates/cintx-runtime/src/planner.rs:380-415` — `parse_component_multiplier`
- `crates/cintx-runtime/src/planner.rs:417-430` — `component_multiplier_for_descriptor`
- `crates/cintx-oracle/tests/center_3c2e_parity.rs:222-287` — existing parity test

**Finding:**

D-09 claims `int3c2e_ip1_*` has `component_rank=3` and that flat-buffer byte-identity works. **The manifest contradicts the first part:** `int3c2e_ip1_cart`, `int3c2e_ip1_sph`, and `int3c2e_ip1_spinor` all have `component_rank: ""` (empty string). `parse_component_multiplier("")` returns `Ok(1)` (planner.rs:382-384), so `component_count = 1` for these symbols.

`build_output_layout` therefore sets `staging_elements = base_elements × 1 × complex_multiplier`. The output buffer has exactly `ni * nj * nk` elements (no IP-component dimension).

The existing `center_3c2e_parity.rs:222-287` test (raw path) compares `RawApiId::INT3C2E_IP1_SPH` output **directly against `vendor_int3c2e_sph`** (the PLAIN, no-derivative function) at atol=1e-9 with 0 mismatches. **This confirms that cintx's current implementation of `int3c2e_ip1_*` evaluates the plain 3c2e integral, NOT the actual ip1 derivative.** The kernel name is currently inaccurate.

**Implication for Phase 18:**

- D-09's flat-buffer byte-identity approach DOES work — but it works because cintx is currently computing the plain 3c2e (not ip1). The byte-identity is `cintx::int3c2e_ip1_sph(...) == vendor::int3c2e_sph(...)`, NOT `cintx::int3c2e_ip1_sph(...) == vendor::int3c2e_ip1_sph(...)`.
- D-09's "component_rank=3" assumption is incorrect for the current manifest state. The real `component_rank` is `""` → multiplier 1.
- The Phase 18 parity test for `int3c2e_ip1_{cart,sph}` should compare safe-API output against `vendor_int3c2e_{cart,sph}` (no derivative) — exactly mirroring `center_3c2e_parity.rs:222-287`. This is fine for ARITY-02's "byte-identity vs libcint" since it's still bytes-identical to a real libcint output, but the planner must document that the **chosen reference function** is `vendor_int3c2e_{sph,cart}` (plain) not `vendor_int3c2e_ip1_{sph,cart}`.
- A consequence: D-06 listing both `int3c2e_ip1_*` AND `int3c2e_*` is partly redundant — they currently compute the same thing. After Item 2's R1 fix (drop the plain `int3c2e_*` from D-06), only `int3c2e_ip1_{cart,sph}` remain in Phase 18.

[VERIFIED: api_manifest.rs:340, 357, 374 confirm `component_rank: ""` for all three int3c2e_ip1_* entries. planner.rs:382-384 confirms empty string returns multiplier 1. center_3c2e_parity.rs:222-287 confirms cintx INT3C2E_IP1_SPH output == vendor int3c2e_sph (plain) at 0 mismatches.]

### Item 6: CI cost estimate for 125+625 tuple sweeps

**Estimation:**

H2O / STO-3G has 5 shells: O-1s (l=0, 1 cart, 1 sph), O-2s (l=0, 1 cart, 1 sph), O-2p (l=1, 3 cart, 3 sph), H1-1s (l=0, 1 cart, 1 sph), H2-1s (l=0, 1 cart, 1 sph). Total: 7 cart AOs, 7 sph AOs.

Mean AO-per-shell: 7/5 = 1.4.

- **Arity-3 sweep:** 125 triples × E[ni*nj*nk] per evaluation. With shells uniformly drawn, E[ni*nj*nk] ≈ (1.4)³ ≈ 2.7 elements. Mean tensor size per evaluate: ~3 elements. Total elements: 125 × 3 ≈ 375 per test.
- **Arity-4 sweep:** 625 quartets × E[ni*nj*nk*nl] ≈ 3.8 elements. Total elements: 625 × 3.8 ≈ 2375 per test.

Per-evaluation overhead (workspace query + plan build + executor.execute + chunk-loop) for a single small kernel evaluation on cpu backend is on the order of 1–10 ms (rough; not measured here). At 10 ms × 625 = 6.25 s for an arity-4 test worst case. At 1 ms × 625 = 0.6 s for the best case.

The two-electron parity test (`two_electron_parity.rs:303-310`) runs the same 5⁴=625 quartet sweep against H2O STO-3G through `eval_raw` and is part of the existing `oracle_parity_gate` matrix — so the existing 4-quartet-deep loop's runtime is a documented baseline. The Phase 18 safe-API path uses the same `CubeClExecutor` under the hood, so per-evaluation cost should be comparable (slight overhead from the safe-API typed wrapping but no algorithmic difference).

**Conclusion:** Likely under 1 s per test on cpu backend. If measurement shows it exceeds budget, the deterministic-subset fallback per D-14 applies: e.g., sample 25 representative triples / 100 representative quartets that cover all angular-momentum combinations (s-s-s, s-s-p, s-p-p, p-p-p, etc.). **Planner should make this fallback wiring explicit in the new test files (a `const FULL_SWEEP: bool = true;` toggle plus a `representative_subset()` function).** [ASSUMED — actual cost will need to be measured; the toggle is cheap insurance.]

### Item 7: `AoSymmetry` enum placement in `cintx-core::operator`

**Verified files:**
- `crates/cintx-core/src/operator.rs:1-48` — full file
- `crates/cintx-core/src/lib.rs:18` — re-export line

**Finding:**

`cintx-core::operator` currently contains exactly `Representation` (enum, line 4-20) and `OperatorId` (newtype, line 23-47). It is the correct location for `AoSymmetry` per D-03 — small, focused, and re-exported at `cintx-core::lib::operator::{OperatorId, Representation}` (line 18). Adding `AoSymmetry` to this module and to the re-export line is the natural fit.

Specifically:
- Module size remains small (~70 lines after adding `AoSymmetry`).
- `Representation` uses `Copy, Clone, Debug, PartialEq, Eq, Hash` + custom `Display` — `AoSymmetry` mirrors this exactly per Claude's discretion guidance.
- `cintx-core::lib.rs:18` reads `pub use operator::{OperatorId, Representation};` — extend to `pub use operator::{AoSymmetry, OperatorId, Representation};`.

[VERIFIED: operator.rs and lib.rs read in full.]

### Item 8: aosym preflight location — `SessionRequest::query_workspace`

**Verified files:**
- `crates/cintx-rs/src/api.rs:63-80` — `SessionRequest::query_workspace`
- `crates/cintx-rs/src/api.rs:104-117, 133-139` — `enforce_safe_facade_policy_gate` call sites inside `SessionQuery::evaluate`
- `crates/cintx-compat/src/raw.rs:816-826` — `enforce_safe_facade_policy_gate` definition

**Finding:**

`SessionRequest::query_workspace` at api.rs:63-80 is the correct fail-fast site for the aosym preflight per D-04. It runs BEFORE `runtime_query_workspace`, which is when the runtime would otherwise allocate any state. Putting the check earlier (in `SessionRequest::new`) would make the constructor fallible (currently infallible per the existing signature `pub fn new(...) -> Self`); putting it later (in `SessionQuery::evaluate`) violates the "fail-fast" requirement and would waste planner work.

The check is a **sibling** to `enforce_safe_facade_policy_gate` (which lives at raw.rs:816 and is called twice inside `SessionQuery::evaluate` at api.rs:111 and api.rs:133). The aosym check goes in `query_workspace`, the policy gate stays in `evaluate`. They have different scopes:
- aosym = "did the caller request something we don't implement?" (knob validation)
- policy gate = "is this operator/representation/profile combination valid?" (envelope validation)

Combining them would conflate the two concerns. Keep them separate.

[VERIFIED: api.rs:63-80 (no aosym check today), api.rs:111+133 (policy gate calls inside evaluate). raw.rs:816 (policy gate signature).]

### Item 9: `with-4c1e` per-test feature gating pattern

**Verified files:**
- `crates/cintx-oracle/tests/oracle_gate_closure.rs:737-739, 798-800` — existing with-4c1e per-test pattern

**Finding:**

The established pattern is `#[cfg(feature = "with-4c1e")]` + `#[cfg(has_vendor_libcint)]` stacked on individual `#[test]` functions inside a single file. The two attributes combine additively — the test compiles only when BOTH are active. Module-level gating with `#![cfg(feature = "with-4c1e")]` would prevent the entire file from compiling under base profile, which would break the arity-4 file structure (the `int2e_*` tests must still compile under base).

For Phase 18's `safe_api_arity4_parity.rs`:
- `int2e_cart` and `int2e_sph` tests: `#[cfg(has_vendor_libcint)]` only.
- `int4c1e_cart` and `int4c1e_sph` tests: `#[cfg(feature = "with-4c1e")]` + `#[cfg(has_vendor_libcint)]`.
- Module gate: `#![cfg(any(feature = "cpu", feature = "rocm"))]` (matches Phase 16-04 pattern at `safe_api_arity2_parity.rs:13`).

[VERIFIED: oracle_gate_closure.rs:737-740 and inspected file structure compiles under all 4 profiles by the existing Phase 11 CI matrix.]

### Item 10: Validation Architecture (Nyquist) — see dedicated section below

---

## Risk Analysis

### R1 — D-06 lists `int3c2e_cart` and `int3c2e_sph` which are NOT in the manifest

**Severity:** MEDIUM
**Impact:** Two of the 8 arity-3 tests in D-06 cannot be implemented without a manifest change.

**What goes wrong:** D-06 instructs the planner to add per-symbol tests for `int3c2e_cart` and `int3c2e_sph`. These symbols are NOT in `api_manifest.csv` or `api_manifest.rs` as operator-kind entries. Only `int3c2e_ip1_{cart,sph,spinor}` and the unstable `int3c2e_sph_ssc` exist. Construct a `SessionRequest` with an OperatorId targeting `int3c2e_sph` is impossible.

**Why it happens:** D-06 was likely drafted from a generic "all named operator families" angle without cross-checking the manifest. The plain `int3c2e_{sph,cart}` is what cintx currently *computes* under the `INT3C2E_IP1_*` IDs (see Item 5), but it has no first-class manifest entry.

**Fallback options (planner choose one):**

1. **Drop the plain `int3c2e_*` from D-06** — reduces arity-3 set from 8 to 6 (`int3c1e_{cart,sph}`, `int3c1e_p2_{cart,sph}`, `int3c2e_ip1_{cart,sph}`). This is the cheapest and most aligned with the current manifest state. **RECOMMENDED.**

2. **Treat `int3c2e_ip1_*` as the canonical 3c2e parity reference** (it actually computes the plain integral per Item 5) and document the kernel-vs-symbol-name inconsistency in the test rustdoc. Still 6 arity-3 tests. Same RECOMMENDED outcome.

3. **Add `int3c2e_{cart,sph,spinor}` operator-kind entries to the manifest** — this is a manifest update with downstream resolver / lock-file regen work. Substantial scope creep; not recommended for Phase 18.

**Detection signal:** `cargo build -p cintx-oracle --tests` will not catch this because the OperatorId is a freely-constructible u32. The test will fail at run time with `Resolver::descriptor` returning `MissingOperatorId(N)` for whatever invalid index the planner picks. **Best to catch in planning, not at test-run time.**

### R2 — F-order rustdoc claim contradicts arity-2 row-major layout

**Severity:** LOW (cosmetic / documentation accuracy)
**Impact:** A naive D-10 rustdoc that says "leftmost is fastest-varying" would be wrong for the arity-2 path (1e/2c2e kernels).

**What goes wrong:** If the rustdoc on `IntegralTensor` (api.rs:441) states unconditionally "F-order: extents[0] is fastest-varying", that's true for arity ≥ 3 outputs (2e/3c1e/3c2e/4c1e kernels) but FALSE for arity-2 outputs (1e/2c2e kernels write row-major; see Item 1). External callers reading this rustdoc and writing strided ndarray views over `owned_values` would get incorrect addressing for 1e and 2c2e.

**Why it happens:** The 1e/2c2e kernels were authored at an earlier project phase before F-order was the explicit convention. The 2e/3c1e/3c2e/4c1e kernels matched libcint's column-major output. The two layouts coexist; the safe API surfaces them both.

**Fallback options:**

1. **Word the rustdoc precisely** — state that the layout matches libcint's per-kernel memory layout (which is F-order for 2e-family operators and ROW-major for 1e/2c2e operators). Acknowledge the asymmetry. **RECOMMENDED for Phase 18** (see Pattern 5 above). Phase 18 stays out of scope.

2. **Defer the rustdoc entirely** — say "AO axes layout is verified per-arity by the oracle parity sweep; see `safe_api_arity{2,3,4}_parity.rs`". Less specific but accurate.

3. **Unify the kernel output layouts to F-order** — this is the cleanest fix but rewrites 1e/2c2e kernels AND breaks the existing arity-2 oracle layout. OUT OF Phase 18 scope.

**Detection signal:** Phase 17's arity-2 parity test passes today because it transposes vendor output (row-major-vs-row-major). If a downstream consumer (like pyscf_rs) reads `IntegralTensor::owned_values` as if it were F-order for 1e operators, results would be silently wrong. **The rustdoc wording matters.**

### R3 — CI cost may exceed budget for 625-quartet arity-4 sweep

**Severity:** LOW
**Impact:** A single test exceeding the per-test budget (typically <30 s for cpu, <2 min for the gate as a whole) would slow CI. Unlikely to be a blocker but worth pre-empting.

**What goes wrong:** Full Cartesian arity-4 (5⁴ = 625 quartets) × 4 operators × cpu+rocm matrix could compound. If each evaluation takes 50 ms instead of 5 ms (which is possible for spinor or large component dimensions), the test slows by 10×.

**Fallback:** Per D-14, the planner can switch any test to a deterministic representative subset (e.g., 50 quartets covering all angular-momentum combinations). The planner should expose a `const FULL_SWEEP: bool = true;` constant near the top of the file so future maintainers can flip it.

**Detection signal:** CI duration > 5 min on a previously <2 min job. Monitor first wave merges.

### R4 — Vendor wrapper coverage gap for two arity-3 sph symbols

**Severity:** LOW
**Impact:** Two of the six recommended Phase 18 arity-3 tests (after R1) cannot reach vendor parity without first adding vendor wrappers.

**What goes wrong:** `vendor_int3c1e_p2_sph` and `vendor_int3c2e_ip1_sph` do NOT exist in `cintx-oracle/src/vendor_ffi.rs`. The `_cart` variants exist (lines 468 and 496) and `int3c1e_p2_sph` is declared in the supplemental header (build.rs:227), so the wrappers are a ~22-line copy-and-edit each.

**Fallback:** Phase 18's planner adds these two wrappers to `vendor_ffi.rs` as a prerequisite task (one task, ~50 lines) before the arity-3 parity tests can land. This is in scope — adding vendor FFI helpers is the responsibility of cintx-oracle.

**Detection signal:** Compile errors at the `cintx_oracle::vendor_ffi::vendor_int3c1e_p2_sph(...)` call site in the new test file.

### R5 — `SHELL_TUPLE_CAPACITY` is `pub(crate)` not `pub`

**Severity:** TRIVIAL
**Impact:** Downstream code (oracle tests) cannot directly reference the constant by name.

**What goes wrong:** A test that wants to write `assert!(shells.len() <= SHELL_TUPLE_CAPACITY)` cannot import the constant.

**Fallback:** Use the hard-coded value `4` or call `try_from_iter` and rely on the error type for capacity violations. No code change required.

**Detection signal:** Compile error `cannot find value SHELL_TUPLE_CAPACITY in this scope` if a test tries to import it.

### R6 — `int3c2e_ip1` kernel semantics inconsistency

**Severity:** MEDIUM (latent, may surface later)
**Impact:** A consumer expecting actual IP1 derivatives from `int3c2e_ip1_sph` will get plain (non-derivative) values. Phase 18 documents this; full fix is a future kernel-implementation phase.

**What goes wrong:** The current `int3c2e_ip1_*` kernel implementation evaluates the plain 3c2e integral, not the IP1 (first-electron gradient) derivative — see Item 5. The symbol name is misleading.

**Fallback:** Phase 18 documents the status in the new arity-3 test file's module rustdoc:
```rust
//! NOTE: cintx's current `int3c2e_ip1_*` kernels evaluate the plain `int3c2e_*`
//! integral, not the IP1 (electron-1 gradient) derivative. This phase asserts
//! byte-identity against `vendor_int3c2e_{sph,cart}` (plain), which is the
//! same comparison `center_3c2e_parity.rs:222-287` uses for the raw path.
//! Future kernel work will surface the actual IP1 derivative; at that point
//! these tests must be updated to compare against `vendor_int3c2e_ip1_*`.
```

**Detection signal:** A downstream consumer doing finite-difference gradient checks will discover the discrepancy. Phase 18 cannot fix it; it documents the status.

---

## Validation Architecture (Nyquist — for step 5.5 VALIDATION.md derivation)

> `workflow.nyquist_validation: true` per `.planning/config.json`. Section is INCLUDED.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | `cargo test` (built-in) [VERIFIED: project-wide pattern] |
| Config file | None (per-crate `Cargo.toml`); features via `--features cpu` (and optionally `--features with-4c1e` for the 4c1e tests) |
| Quick run command | `cargo test -p cintx-rs --locked` (covers aosym error-path unit tests) |
| Full suite command | `CINTX_BACKEND=cpu cargo test -p cintx-rs --features cpu --locked && CINTX_BACKEND=cpu cargo test -p cintx-oracle --features cpu --locked --test safe_api_arity3_parity --test safe_api_arity4_parity && CINTX_BACKEND=cpu cargo test -p cintx-oracle --features cpu,with-4c1e --locked --test safe_api_arity4_parity` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| ARITY-01 | `SessionRequest::evaluate` accepts arity-3 and arity-4 ShellTuple inputs without UnsupportedApi/InvalidShellTuple | integration (smoke) | `cargo test -p cintx-oracle --features cpu --locked --test safe_api_arity3_parity test_int3c1e_sph_safe_api_parity` (first execution; if it doesn't panic with UnsupportedApi, ARITY-01 is satisfied as a side effect of ARITY-02) | ❌ Wave 0: create `crates/cintx-oracle/tests/safe_api_arity3_parity.rs` and `safe_api_arity4_parity.rs` |
| ARITY-02 | Safe-API byte-identical to vendored libcint at atol=1e-12 for 6 (or 8 if R1 resolved) arity-3 ops and 4 arity-4 ops | integration (oracle) | Full suite command above with `--features cpu` (and `--features cpu,with-4c1e` for int4c1e_*) | ❌ Wave 0: create test files |
| ARITY-03 | Output tensors are F-order (arity ≥ 3) — verified implicitly by ARITY-02 byte-identity | integration (oracle, same as ARITY-02) | (same as ARITY-02) | ❌ Wave 0 (same file) |
| ARITY-04 | aosym non-S1 returns `FacadeError::UnsupportedAoSymmetry { requested }`; aosym=None or aosym=Some(S1) succeeds | unit | `cargo test -p cintx-rs --locked aosym_error_path` | ❌ Wave 0: add `aosym_error_path` test inside `cintx-rs/src/api.rs::tests` module (per D-05) |
| ARITY-05 | CI gate runs new tests inside `oracle_parity_gate` matrix with 0 mismatches across cpu and (optionally) rocm + with-4c1e profiles | CI | `.github/workflows/oracle_parity_gate.*` runs the full suite command in matrix mode | ✓ existing CI; D-15 reuses it |

### Sampling Rate

- **Per task commit:** `cargo test -p cintx-rs --locked` (~10 s; covers aosym error-path + existing arity-2 unit tests)
- **Per wave merge:** Full suite command above (~10-60 s estimated for cpu; rocm and with-4c1e profile runs are in the matrix)
- **Phase gate:** Full suite green across all four manifest profiles (base / with-f12 / with-4c1e / with-f12+with-4c1e) AND has-vendor-libcint host before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `crates/cintx-oracle/tests/safe_api_arity3_parity.rs` — 6 per-symbol parity tests (recommended after R1) covering `int3c1e_{cart,sph}`, `int3c1e_p2_{cart,sph}`, `int3c2e_ip1_{cart,sph}`
- [ ] `crates/cintx-oracle/tests/safe_api_arity4_parity.rs` — 4 per-symbol parity tests covering `int2e_{cart,sph}` and `int4c1e_{cart,sph}` (the latter two gated `#[cfg(feature = "with-4c1e")]`)
- [ ] (Optional but recommended) `crates/cintx-oracle/tests/safe_api_helpers.rs` — shared `collect_safe_api_matrix` helper (Claude's discretion item); if added, refactor `safe_api_arity2_parity.rs` to use it
- [ ] `crates/cintx-oracle/src/vendor_ffi.rs` — add `vendor_int3c1e_p2_sph` (~22 lines) and `vendor_int3c2e_ip1_sph` (~22 lines) wrappers (R4)
- [ ] `crates/cintx-rs/src/api.rs::tests::aosym_error_path` — new unit test per D-05 (~30 lines for the per-variant loop)
- [ ] `crates/cintx-core/src/operator.rs` — add `AoSymmetry` enum + `Display` impl (~25 lines)
- [ ] `crates/cintx-core/src/lib.rs:18` — extend re-export to include `AoSymmetry`
- [ ] `crates/cintx-runtime/src/options.rs:117` — add `pub aosym: Option<cintx_core::AoSymmetry>` field
- [ ] `crates/cintx-rs/src/api.rs:63-80` — add aosym preflight in `query_workspace` (~10 lines)
- [ ] `crates/cintx-rs/src/api.rs:441` — add F-order rustdoc on `IntegralTensor` (~15 lines of doc comments)
- [ ] `crates/cintx-rs/src/error.rs:14-24` — add `FacadeError::UnsupportedAoSymmetry { requested: String }` variant + `FacadeErrorKind::UnsupportedAoSymmetry` + match arm in `kind()`
- [ ] `crates/cintx-rs/src/prelude.rs` — add `pub use cintx_core::AoSymmetry;`
- [ ] (Optional) `crates/cintx-rs/src/builder.rs` — add `pub fn aosym(self, aosym: AoSymmetry) -> Self` setter method

### Eight Nyquist Dimensions

1. **Inputs covered:** All `OperatorId` values for the 12 (or 10 after R1) target symbols × 3 representations (cart/sph; spinor compiled-only per D-07); 125 arity-3 triples × 6 ops + 625 arity-4 quartets × 4 ops = 750 + 2500 = 3250 evaluations per cpu profile run.
2. **Output classes:** byte-identity (==) vs vendored libcint 6.1.3 at atol=1e-12, rtol=0.0; any-element-nonzero sentinel to catch zero-fill regressions; `FacadeError::UnsupportedAoSymmetry` typed error for non-S1 aosym; preserved `FacadeError::*` variants for invalid operator / out-of-envelope source/profile/F12/4c1e.
3. **State transitions:** `SessionRequest::new` → `query_workspace` (aosym preflight + runtime workspace) → `evaluate` (real CubeClExecutor dispatch); aosym failure short-circuits at `query_workspace`.
4. **Error paths:** non-S1 aosym → `UnsupportedAoSymmetry`; invalid operator → existing `UnsupportedApi`; ShellTuple > 4 → `ShellTupleArityError` (caller-side, not reached for arity ≤ 4); memory limit exceeded → existing `Memory`.
5. **Concurrency:** Tests run serially (`--test-threads=1` for safe_api_arity2_parity per Phase 17 verification; same pattern for the new files).
6. **External dependencies:** vendored libcint 6.1.3 build (`CINTX_ORACLE_BUILD_VENDOR=1` env at build time + `has_vendor_libcint` cfg at compile time); cubecl cpu/rocm backends (CINTX_BACKEND env).
7. **Performance envelopes:** Per-test budget < 60 s on cpu backend (target < 5 s, see R3); gate-wide budget unchanged.
8. **Coverage tooling:** None new. Existing oracle gate provides per-symbol failure messages for CI bisection. `cargo public-api` could optionally diff `cintx-rs::api` and `cintx-rs::prelude` to confirm SemVer-additive changes (no positional arg breaks).

---

## Common Pitfalls

### Pitfall 1: Forgetting to update `FacadeErrorKind` enum alongside `FacadeError`

**What goes wrong:** Adding `FacadeError::UnsupportedAoSymmetry` without adding `FacadeErrorKind::UnsupportedAoSymmetry` makes `FacadeError::kind()`'s match non-exhaustive — compile error.

**Why it happens:** `FacadeErrorKind` is a separate enum (error.rs:6-12) used by `FacadeError::kind()` to project the variant. Phase 17's pattern is to keep them parallel.

**How to avoid:** Add both variants together. Update the `match` in `FacadeError::kind()` (error.rs:28-33) with a new arm. Update the existing test in `error.rs::tests` if needed.

**Warning signs:** `error[E0004]: non-exhaustive patterns` in api.rs / error.rs at compile time.

### Pitfall 2: Arity-3/4 vendor comparison applying the arity-2 transpose

**What goes wrong:** Copying the matrix-assembly code from `safe_api_arity2_parity.rs:280-292` (`pair_values[ii * nj + jj]` row-major reading and vendor transposition at lines 396-400) into the arity-3/4 test would corrupt the comparison: cintx and vendor BOTH write F-order for arity ≥ 3, so transposing one of them creates systematic mismatches.

**Why it happens:** Phase 17's `collect_safe_api_matrix` was designed for 1e/2c2e (row-major cintx, column-major vendor — different conventions, transpose required). The arity-3/4 case is row-major-vs-row-major or column-major-vs-column-major depending on which family — but in either case, no transpose is needed because cintx and vendor agree.

**How to avoid:** For arity ≥ 3, follow the pattern in `compare.rs:787-797` (int2e_sph direct compare) and `center_3c2e_parity.rs:243-277` (3c2e_sph direct compare). Compare `cintx::owned_values` against `vendor_out` directly with `count_mismatches`.

**Warning signs:** Systematic mismatches at every non-symmetric tuple. Diagonal entries pass; off-diagonals fail.

### Pitfall 3: Spinor arity-3/4 falling through to byte-identity assertion

**What goes wrong:** Even though spinor arity-3/4 is "compiled but unverified" per D-07, an over-zealous planner might add `int2e_spinor` / `int3c2e_ip1_spinor` parity tests assuming they should work. These would either fail (no vendor wrapper that returns real-valued output) or pass spuriously (idempotency-only).

**Why it happens:** D-07 is easy to miss. The phrase "compiled but unverified" is structural — the symbol routes through `SessionRequest::evaluate` and the executor without errors, but the value-level correctness is not gated.

**How to avoid:** Phase 18 does NOT add spinor parity tests. The new test files (`safe_api_arity3_parity.rs` and `safe_api_arity4_parity.rs`) cover only cart and sph variants. If the planner wants a smoke test that `evaluate()` returns `Ok(_)` for the spinor variants without comparison, that is acceptable as a sentinel (Claude's discretion).

**Warning signs:** Failed test with cryptic spinor-vendor-wrapper error or compile error referencing `vendor_int2e_spinor` (which returns complex interleaved — not directly comparable).

### Pitfall 4: `int4c1e_*` tests compiling under the base profile

**What goes wrong:** Putting `int4c1e_cart` / `int4c1e_sph` tests at module top-level in `safe_api_arity4_parity.rs` (without `#[cfg(feature = "with-4c1e")]` on each #[test] function) causes compile failure under the base profile because `OperatorId::new(22)` and `(23)` don't have routable kernels under base.

**Why it happens:** `int4c1e_*` is `feature_flag: With4c1e, stability: Optional` per the manifest. Without `with-4c1e`, the kernel registry rejects the family, and `executor.supports()` returns false → `FacadeError::UnsupportedApi`.

**How to avoid:** Stack `#[cfg(feature = "with-4c1e")]` + `#[cfg(has_vendor_libcint)]` on each int4c1e test function, per Item 9's verified pattern.

**Warning signs:** `cargo build -p cintx-oracle --tests` fails under base profile; CI base-profile job fails before tests run.

### Pitfall 5: `cintx-oracle/Cargo.toml` already has `cintx-rs` — don't add it again

**What goes wrong:** Phase 17 added `cintx-rs` and `cintx-runtime` as direct deps of `cintx-oracle` (lines 28-29). A planner who treats Phase 18 as "first time we need cintx-rs in oracle" and adds the line again will get a Cargo lockfile drift error.

**Why it happens:** D-07/D-15 wording can be read either way ("matches Phase 17 setup"). Verify the current state of `cintx-oracle/Cargo.toml` before deciding what to change.

**How to avoid:** Phase 18 does NOT need to modify `cintx-oracle/Cargo.toml`. All required deps are present.

**Warning signs:** Cargo warning "duplicate dependency" or lockfile complaint.

### Pitfall 6: Forgetting to handle aosym `None` as equivalent to `Some(S1)`

**What goes wrong:** Naive code `if options.aosym.unwrap() != S1` panics on default `None`.

**Why it happens:** D-02 says "None is the default and is treated as `Some(S1)`."

**How to avoid:** Use `if let Some(aosym) = options.aosym { if aosym != AoSymmetry::S1 { ... } }` — None falls through to the existing logic.

**Warning signs:** Panic at runtime on the first `query_workspace` call with default `ExecutionOptions`.

---

## Code Examples (verified)

### Example 1: Full per-symbol arity-3 parity test (int3c1e_sph)

```rust
// Source: derived from safe_api_arity2_parity.rs:591-665 pattern + compare.rs:811-821 (no-transpose)
#[test]
#[cfg(has_vendor_libcint)]
fn test_int3c1e_sph_safe_api_parity() {
    use cintx_oracle::vendor_ffi;

    let (atm, bas, env) = build_h2o_sto3g();  // PTR_ENV_START-aware
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Spheric);
    let atol = 1e-12_f64;
    let rtol = 0.0_f64;

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;

    let mut total_mismatches = 0usize;
    let mut any_nonzero = false;
    let mut tuples_checked = 0usize;

    for i in 0..N_SHELLS {
        for j in 0..N_SHELLS {
            for k in 0..N_SHELLS {
                let ni = shells[i].ao_per_shell();
                let nj = shells[j].ao_per_shell();
                let nk = shells[k].ao_per_shell();
                let n_elem = ni * nj * nk;

                let shell_tuple = ShellTuple::try_from_iter([
                    shells[i].clone(), shells[j].clone(), shells[k].clone(),
                ]).expect("3 ≤ SHELL_TUPLE_CAPACITY=4");
                let request = SessionRequest::new(
                    OperatorId::new(18),  // int3c1e_sph (manifest entry 18 — verified)
                    Representation::Spheric,
                    &basis,
                    shell_tuple,
                    ExecutionOptions::default(),
                );
                let safe_out = request
                    .query_workspace().expect("query_workspace must succeed")
                    .evaluate().expect("evaluate must succeed")
                    .tensor.owned_values;

                let mut vendor_out = vec![0.0_f64; n_elem];
                let shls = [i as i32, j as i32, k as i32];
                vendor_ffi::vendor_int3c1e_sph(
                    &mut vendor_out, &shls, &atm, natm, &bas, nbas, &env);

                if safe_out.iter().any(|&v| v.abs() > 1e-18)
                    || vendor_out.iter().any(|&v| v.abs() > 1e-18) {
                    any_nonzero = true;
                }
                // No transpose — cintx 3c1e_sph writes F-order matching vendor directly
                // (precedent: compare.rs:811-821, center_3c1e_parity.rs).
                total_mismatches += count_mismatches(&vendor_out, &safe_out, atol, rtol);
                tuples_checked += 1;
            }
        }
    }

    assert!(any_nonzero, "int3c1e_sph safe-API outputs are all zeros over {tuples_checked} triples");
    assert_eq!(total_mismatches, 0,
        "int3c1e_sph safe API: {total_mismatches} elements exceed atol={atol:.0e}/rtol={rtol:.0e} vs vendored libcint over {tuples_checked} triples");
}
```

### Example 2: aosym error-path unit test (per D-05)

```rust
// Source: PROPOSED — inside crates/cintx-rs/src/api.rs::tests module (D-05)
#[test]
fn aosym_error_path_rejects_non_s1_with_typed_error() {
    use cintx_core::AoSymmetry;
    let (basis, shells) = sample_basis(Representation::Cart);

    for non_s1 in [AoSymmetry::S2ij, AoSymmetry::S2kl, AoSymmetry::S4, AoSymmetry::S8] {
        let options = ExecutionOptions { aosym: Some(non_s1), ..Default::default() };
        let request = SessionRequest::new(
            OperatorId::new(0),  // int1e_ovlp_cart — any valid op works
            Representation::Cart,
            &basis,
            shells.clone(),
            options,
        );
        let err = request.query_workspace().unwrap_err();
        match err {
            FacadeError::UnsupportedAoSymmetry { requested } => {
                assert_eq!(requested, non_s1.to_string(),
                    "requested field must carry the lowercase pyscf form (e.g., 's8')");
            }
            other => panic!("expected UnsupportedAoSymmetry for aosym={non_s1:?}, got {other:?}"),
        }
    }
}

#[test]
fn aosym_none_and_s1_both_succeed() {
    use cintx_core::AoSymmetry;
    let (basis, shells) = sample_basis(Representation::Cart);

    for aosym in [None, Some(AoSymmetry::S1)] {
        let options = ExecutionOptions { aosym, ..Default::default() };
        let request = SessionRequest::new(
            OperatorId::new(0), Representation::Cart, &basis, shells.clone(), options,
        );
        let _query = request.query_workspace()
            .unwrap_or_else(|e| panic!("aosym={aosym:?} must succeed; got {e:?}"));
    }
}
```

### Example 3: Adding a vendor wrapper for `vendor_int3c1e_p2_sph` (R4)

```rust
// Source: PROPOSED — crates/cintx-oracle/src/vendor_ffi.rs (mirrors vendor_int3c1e_p2_cart at lines 465-490)
/// Evaluate int3c1e_p2_sph for a single shell triple using vendored libcint.
#[cfg(has_vendor_libcint)]
pub fn vendor_int3c1e_p2_sph(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    let opt = std::ptr::null_mut();
    unsafe {
        ffi::int3c1e_p2_sph(
            out.as_mut_ptr(),
            std::ptr::null_mut(),
            shls.as_ptr() as *mut i32,
            atm.as_ptr() as *mut i32,
            natm,
            bas.as_ptr() as *mut i32,
            nbas,
            env.as_ptr() as *mut f64,
            opt,
            std::ptr::null_mut(),
        )
    }
}
```

`int3c1e_p2_sph` is declared in the supplemental header (build.rs:227) so the `ffi::int3c1e_p2_sph` binding exists after `bindgen` runs. The wrapper signature matches the pattern at vendor_ffi.rs:468-490.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Phase 17: arity-2 only via real CubeClExecutor | Phase 18: arity-3/4 also via real CubeClExecutor | Phase 18 (this phase) | Unblocks pyscf_rs SCF (J/K) on `int2e_*` and density fitting on `int3c2e_*` |
| `ExecutionOptions` no aosym knob | `aosym: Option<AoSymmetry>` additive field | Phase 18 | Future phases can ship `s8` packing without changing the signature |
| `FacadeError` has 4 variants | `FacadeError` has 5 variants (adds `UnsupportedAoSymmetry`) | Phase 18 | Programmatic detection of "knob requested, not implemented" |
| `IntegralTensor` has no layout rustdoc | `IntegralTensor` carries F-order rustdoc invariant | Phase 18 | Downstream consumers know how to read `owned_values` |
| `cintx-core::operator` has 2 types (Representation, OperatorId) | adds `AoSymmetry` | Phase 18 | All AO-related domain primitives live in one module |

**Deprecated/outdated after Phase 18:**
- The roadmap note "arity-3/4 dispatch deferred" — closed.
- The pyscf_rs `pyscf-gto/src/intor.rs` wrapper TODOs for `int2e_*` and `int3c2e_*` — closed.

**Still pending after Phase 18:**
- `aosym` packings `s2ij`, `s2kl`, `s4`, `s8` — typed-error path only.
- Spinor arity-3/4 byte-identity gating — compiled-only.
- `int3c2e_ip1_*` real IP1 derivative kernel — currently computes plain 3c2e (Item 5 finding).
- Unstable-source arity-3 (`origk`, `ssc`) through SessionRequest.

---

## Environment Availability

> External dependencies: vendored libcint 6.1.3, cubecl cpu backend, Rust 1.94.0 toolchain.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable toolchain | All compilation | ✓ | rust-toolchain.toml pin (1.94.0 per CLAUDE.md) | — |
| `cargo test` (built-in) | Test execution | ✓ | via toolchain | — |
| `cintx-cubecl` cpu feature | Real executor dispatch on arity-3/4 | ✓ | path dep, default-on cpu | rocm backend (D-15 covers in matrix) |
| Vendored libcint 6.1.3 | ARITY-02 byte-identity tests (`has_vendor_libcint`) | conditional | depends on `CINTX_ORACLE_BUILD_VENDOR=1` | Idempotency-only spinor tests compile without vendor build; cart/sph parity tests cfg-skipped |
| `with-4c1e` feature build | int4c1e_* tests | conditional | optional, opt-in via `--features with-4c1e` | int4c1e_* tests cfg-skipped under base profile |
| `bindgen` 0.71.1 | Vendor FFI binding generation | ✓ | already used | — |
| `cc` 1.2.x | Vendored libcint build | ✓ | already used | — |

**Missing dependencies with no fallback:** None — all Phase 18 paths work with existing tools.

**Missing dependencies with fallback:** Vendored libcint (cfg-skip) and with-4c1e (cfg-skip per-test).

---

## Project Constraints (from CLAUDE.md)

| Constraint | Source | Impact on Phase 18 |
|------------|--------|---------------------|
| Compatibility: libcint 6.1.3 at atol=1e-12 | CLAUDE.md + Phase 15 | All new oracle tests assert at `atol=1e-12, rtol=0.0` (D-15) |
| Architecture: CubeCL is primary compute backend | CLAUDE.md | Uses `cintx_cubecl::CubeClExecutor` for arity-3/4 dispatch (Phase 17 already wired) |
| API Surface: Safe API first | CLAUDE.md | Phase 18 extends the safe surface to arity-3/4 — this IS the primary surface for pyscf_rs |
| Error Handling: public library errors use `thiserror` v2 | CLAUDE.md | New `FacadeError::UnsupportedAoSymmetry` variant added on existing `#[derive(Error)]` enum |
| Verification: full API coverage backed by compiled manifest lock | CLAUDE.md | All 12 (or 10 after R1) Phase 18 symbols are in the compiled manifest; R1 surfaces a manifest mismatch with D-06 |
| Artifacts: deliverables to `/mnt/data` | CLAUDE.md | Oracle parity reports written by `xtask oracle-compare`; no new artifact paths introduced in Phase 18 |
| Rust toolchain: pin `1.94.0` | CLAUDE.md | No toolchain or lockfile changes; only source edits and one new vendor wrapper |
| No public APIs exposing backend-specific runtime types | CLAUDE.md | `AoSymmetry` is a pure domain enum in cintx-core; no backend type leakage |
| GSD Workflow Enforcement | CLAUDE.md | Phase 18 work goes through `/gsd:execute-phase` |

---

## Security Domain

> `security_enforcement` is not explicitly set in `.planning/config.json` — treat as the default for a numerical library (low surface area). Phase 18 introduces no authentication, session management, access control, cryptography, or externally-supplied input validation. The `enforce_safe_facade_policy_gate` call (raw.rs:816) remains in place and is not modified. The new `aosym` knob validates strictly against an enum (no string parsing of untrusted input).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes (strict enum match) | `AoSymmetry` enum + match arm validation; no string parsing |
| V6 Cryptography | no | — |

### Known Threat Patterns for cintx-rs safe-API surface

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Untyped operator ID (u32) accepted | Tampering | Validated via `Resolver::descriptor` returning `MissingOperatorId` (existing pattern) |
| Out-of-arity ShellTuple | Tampering | `ShellTuple::try_from_iter` enforces `SHELL_TUPLE_CAPACITY=4` (existing) |
| Out-of-envelope operator/representation | Tampering | `enforce_safe_facade_policy_gate` (existing) |
| Unimplemented aosym packing | Tampering / Information disclosure | NEW: `FacadeError::UnsupportedAoSymmetry { requested }` typed error path |
| Memory exhaustion via large workspace | DoS | Existing `MemoryLimitExceeded` plumbing |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `CubeClExecutor::execute` correctly handles arity-3 and arity-4 dispatch paths (kernel selection, env propagation) under cpu backend | Code Investigation §1, §6 | Existing arity-3/4 raw-path oracle tests (`two_electron_parity.rs`, `center_3c2e_parity.rs`, `center_3c1e_parity.rs`) confirm raw-path correctness; safe-API uses same executor, but the safe-API typed→plan path may surface a regression. Mitigation: per-symbol tests will fail loudly. |
| A2 | Per-evaluation runtime on cpu backend is < 50 ms for small (< 100 element) tensors | R3 | Total CI cost may exceed budget. Mitigation: D-14 deterministic-subset fallback. |
| A3 | `int3c1e_p2_sph` symbol is declared in cint_funcs.h or the supplemental header | R4 / Example 3 | Wrapper compile fails. Mitigation: build.rs:227 confirms supplemental declaration; if missing in the FFI module, add the extern declaration in build.rs (existing pattern). |
| A4 | Phase 18's `IntegralTensor` rustdoc on F-order is a doc-only change; no `cargo public-api` diff complains | Pattern 5 | If `cargo public-api` treats doc changes as API changes, a SemVer-additive baseline is broken. Mitigation: rustdoc is non-functional; the diff should be ignored. |
| A5 | The 12 (or 10 after R1) target symbols all support cart and sph forms through the safe API without spinor-transform side effects | D-07 / Item 4 | A spinor-only kernel path inadvertently triggered for a cart input would fail. Mitigation: `RepresentationSupport` (resolver.rs:73-96) gates representation per-operator. |
| A6 | `vendor_int3c2e_sph` is a suitable reference for cintx's `int3c2e_ip1_sph` output | Item 5 / R6 | If a future planner expects IP1 derivatives, the parity test becomes misleading. Mitigation: planner documents the kernel-vs-symbol-name inconsistency in the test rustdoc per R6's fallback. |

---

## Open Questions (RESOLVED)

> All four questions disposed during `/gsd:plan-phase 18` on 2026-05-12. Inline RESOLVED markers reflect the user / planner choice. Recommendations above marked **(stale)** are kept only for historical context.

1. **R1 resolution: drop or add to manifest?** — **RESOLVED: OPTION 3 (add to manifest).** User explicitly chose to add plain `int3c2e_cart` and `int3c2e_sph` operator-kind rows to the manifest at `crates/cintx-ops/generated/compiled_manifest.lock.json:275-306` (the lock file is the single edit site; `api_manifest.rs` + `api_manifest.csv` regenerate via `build.rs`). D-06 retained with its full 8-symbol arity-3 set. Side-effects scheduled in Plan 18-01: OperatorId shift for `int4c1e_{cart,sph}` (22→24, 23→25) and resolver `misc_wrapper_macro` exception for plain `int3c2e_*` (no upstream `cint3c2e_*` misc.h wrappers). The original "Recommendation: Drop them" above is **stale** — superseded by the user choice.
   - What we know: D-06 lists `int3c2e_cart` and `int3c2e_sph` which are NOT in the manifest. Only `int3c2e_ip1_*` exists.
   - What's unclear: ~~Whether the user/planner wants to (a) drop these two symbols from Phase 18 scope or (b) add them to the manifest as a planner-time prerequisite task.~~ Settled — manifest expansion.
   - Recommendation (stale): ~~**Drop them (option a)**~~. Superseded by the user's manifest-expansion choice.

2. **R2 resolution: how strong is the F-order rustdoc?** — **RESOLVED: Qualified wording (Pattern 5).** Planner adopts arity-aware doc: arity-2 outputs are row-major; arity ≥ 3 outputs are F-order. Strict uniform-F-order wording would be inaccurate. Implemented in Plan 18-02 with grep-checkable acceptance criteria.
   - What we know: cintx kernels write F-order for arity ≥ 3 but row-major for arity-2 (1e/2c2e).
   - Recommendation: **Qualified wording** (Pattern 5 above). Confirmed.

3. **Spinor smoke test: yes or no?** — **RESOLVED: No.** Planner default. Not added in Plans 18-03 / 18-04. Spinor arity-3/4 remains "compiled but unverified" per D-07.
   - Recommendation: **Default no.** Confirmed.

4. **Shared `collect_safe_api_matrix` helper: factor or inline?** — **RESOLVED: Inline per test file.** Planner deferred the shared-helper extraction (CONTEXT.md "Claude's Discretion default: skip") to keep Plan 18-03 / 18-04 surface minimal. Each new test file duplicates the small `collect_safe_api_tuple_buffer` body. Helper extraction may be revisited in a follow-up polish phase if maintenance burden grows.
   - Recommendation (stale): ~~Factor into `crates/cintx-oracle/tests/common/mod.rs`~~. Superseded by the planner's inline-duplication choice.

---

## Recommendations to the Planner (concrete decisions)

### Scope adjustments

1. **Reduce D-06's arity-3 set from 8 to 6.** Drop `int3c2e_cart` and `int3c2e_sph` (R1). Final arity-3 set: `int3c1e_cart`, `int3c1e_sph`, `int3c1e_p2_cart`, `int3c1e_p2_sph`, `int3c2e_ip1_cart`, `int3c2e_ip1_sph`.

2. **Add a planner-time task to introduce two new vendor wrappers** (R4): `vendor_int3c1e_p2_sph` and `vendor_int3c2e_ip1_sph` in `crates/cintx-oracle/src/vendor_ffi.rs`. These are prerequisite to the arity-3 parity tests for those two sph symbols.

3. **Update D-09's flat-buffer rationale** to reflect that `int3c2e_ip1_*` has `component_rank: ""` (not `=3`) and that the parity reference is `vendor_int3c2e_{cart,sph}` (plain), not `vendor_int3c2e_ip1_*` (Item 5). The flat-buffer comparison still works; the rationale changes.

4. **Word D-10's IntegralTensor rustdoc as Pattern 5 above** (arity-aware) — not as a blanket F-order claim (R2).

### Architecture & file structure

5. **Factor `collect_safe_api_matrix(operator, repr, basis, tuple) -> Vec<f64>` into a shared helper** at `crates/cintx-oracle/tests/common/mod.rs` (preferred) or `safe_api_helpers.rs`. Suggested signature:

```rust
/// Shared per-tuple safe-API evaluator. Returns the raw `owned_values` buffer.
/// Arity is determined by `shells.len()`; caller is responsible for assembling
/// the larger matrix (arity-2 needs the row-major-to-row-major loop in the test;
/// arity-3/4 compares the buffer directly against vendor output).
pub fn collect_safe_api_tuple_buffer(
    operator_id: OperatorId,
    rep: Representation,
    basis: &BasisSet,
    shells: &[Arc<Shell>],   // up to 4 — enforced by ShellTuple::try_from_iter
) -> Vec<f64> { /* ... */ }
```

The arity-2 `collect_safe_api_matrix` already in `safe_api_arity2_parity.rs:236-292` can be refactored to call this helper internally for each shell pair. The arity-3 and arity-4 tests call it once per triple/quartet and compare the result directly.

6. **File structure for the new tests:**
   - `crates/cintx-oracle/tests/common/mod.rs` (or `safe_api_helpers.rs`) — shared `build_h2o_sto3g_safe_basis`, `collect_safe_api_tuple_buffer`, `count_mismatches`, `nsph`, `ncart`. Refactor `safe_api_arity2_parity.rs` to import these.
   - `crates/cintx-oracle/tests/safe_api_arity3_parity.rs` — 6 per-symbol tests (after R1).
   - `crates/cintx-oracle/tests/safe_api_arity4_parity.rs` — 4 per-symbol tests; int4c1e_* each gated `#[cfg(feature = "with-4c1e")]`.

7. **Module-level guards on each test file:**
```rust
#![cfg(any(feature = "cpu", feature = "rocm"))]   // matches Phase 16-04 + Phase 17 pattern
```

8. **Use full Cartesian sweep by default** (per D-14). Add a sentinel `const FULL_SWEEP: bool = true;` near the top of each file so the deterministic-subset fallback (R3) is easy to engage if CI measurements show > 1 s per test.

### Code-level details

9. **`AoSymmetry` derive set:** `#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]` with `#[default] S1`. Implement `Display` emitting lowercase pyscf form (`s1, s2ij, s2kl, s4, s8`). Mirror `Representation`'s pattern exactly.

10. **`FacadeError::UnsupportedAoSymmetry { requested: String }`** — match arm in `FacadeError::kind()` (error.rs:28-33) returns `FacadeErrorKind::UnsupportedAoSymmetry`. Add the kind variant to error.rs:6-12 enum.

11. **`SessionBuilder::aosym(self, aosym: AoSymmetry) -> Self`** — optional builder setter, mirroring `f12_zeta` at builder.rs:91-94. Trivial addition; recommended for ergonomics.

12. **Re-exports:**
    - `crates/cintx-core/src/lib.rs:18`: `pub use operator::{AoSymmetry, OperatorId, Representation};`
    - `crates/cintx-rs/src/prelude.rs` (around line 33): add `pub use cintx_core::AoSymmetry;` alongside existing `Representation` re-export.

13. **aosym preflight in `SessionRequest::query_workspace`** (api.rs:63-80, sits BEFORE `runtime_query_workspace` call):
```rust
pub fn query_workspace(&self) -> Result<SessionQuery<'basis>, FacadeError> {
    if let Some(aosym) = self.options.aosym {
        if aosym != AoSymmetry::S1 {
            return Err(FacadeError::UnsupportedAoSymmetry {
                requested: aosym.to_string(),
            });
        }
    }
    // ... existing body unchanged
}
```

### Verification

14. **Wave 0** (parallel-safe scaffolding):
    - Add `AoSymmetry` enum + `Display` impl in cintx-core
    - Add `ExecutionOptions::aosym` field in cintx-runtime
    - Add `FacadeError::UnsupportedAoSymmetry` variant + kind enum in cintx-rs
    - Add aosym preflight in `SessionRequest::query_workspace`
    - Add F-order rustdoc on `IntegralTensor`
    - Add `vendor_int3c1e_p2_sph` and `vendor_int3c2e_ip1_sph` in cintx-oracle/vendor_ffi.rs
    - Add shared `tests/common/mod.rs` helper
    - Update prelude re-exports
    - Add `aosym_error_path` unit test in cintx-rs/src/api.rs::tests

15. **Wave 1** (depends on Wave 0):
    - Write `safe_api_arity3_parity.rs` (6 per-symbol tests)
    - Write `safe_api_arity4_parity.rs` (4 per-symbol tests; with-4c1e gates)
    - Refactor `safe_api_arity2_parity.rs` to use shared helper (optional polish)

16. **Phase gate (`/gsd:verify-work`):**
    - `cargo test -p cintx-rs --locked` — passes aosym_error_path
    - `cargo build -p cintx-oracle --features cpu --locked --tests` — passes
    - `cargo build -p cintx-oracle --features cpu,with-4c1e --locked --tests` — passes
    - `CINTX_BACKEND=cpu cargo test -p cintx-oracle --features cpu --locked --test safe_api_arity3_parity --test safe_api_arity4_parity` on vendor-build host — 0 mismatches across 10 (6+4) parity tests; the 4c1e tests SKIP under cpu-only (rerun with `--features cpu,with-4c1e`)

---

## Sources

### Primary (HIGH confidence)

- `crates/cintx-rs/src/api.rs` — full file read; lines 1-300 + 430-630 inspected. `SessionRequest`, `SessionQuery`, `IntegralTensor`, `FacadeError` integration points verified.
- `crates/cintx-rs/src/error.rs` — full file read. `FacadeError` and `FacadeErrorKind` enum structures confirmed.
- `crates/cintx-rs/src/prelude.rs` — full file read. Re-export pattern confirmed.
- `crates/cintx-rs/src/builder.rs` — partially read (lines 1-110). `f12_zeta` builder pattern confirmed.
- `crates/cintx-rs/Cargo.toml` — full file read. Workspace path deps, `thiserror = "2"`, feature flags verified.
- `crates/cintx-core/src/lib.rs` — full file read. `pub use operator::{OperatorId, Representation};` confirmed at line 18.
- `crates/cintx-core/src/operator.rs` — full file read (48 lines). Sibling-of-Representation placement confirmed.
- `crates/cintx-core/src/shell.rs` — full file read. `SHELL_TUPLE_CAPACITY = 4` (pub(crate)), `try_from_iter` body, `ao_per_shell()` confirmed.
- `crates/cintx-runtime/src/options.rs` — partial read (lines 90-135). `ExecutionOptions::f12_zeta` confirmed as pattern source.
- `crates/cintx-runtime/src/planner.rs` — partial read (lines 40-330, 380-415, 417-430). `OutputLayoutMetadata`, `build_output_layout`, `parse_component_multiplier`, `component_multiplier_for_descriptor` verified.
- `crates/cintx-ops/src/resolver.rs` — partial read (lines 1-300). `Resolver::descriptor`, `OperatorDescriptor`, `ManifestEntry`, `RepresentationSupport`, `HelperKind` verified.
- `crates/cintx-ops/src/generated/api_manifest.csv` — full file read (131 lines). All 12 target arity-3/4 operator entries cross-referenced.
- `crates/cintx-ops/src/generated/api_manifest.rs` — partial read (lines 160-417). OperatorId-to-symbol mapping confirmed.
- `crates/cintx-compat/src/raw.rs` — partial read (lines 810-830). `enforce_safe_facade_policy_gate` signature confirmed.
- `crates/cintx-oracle/Cargo.toml` — full file read. Phase 17's cintx-rs and cintx-runtime deps already present.
- `crates/cintx-oracle/src/fixtures.rs` — partial read (lines 1-140, 200-265). `build_h2o_sto3g` and `OracleRawInputs::sample` confirmed.
- `crates/cintx-oracle/src/vendor_ffi.rs` — full grep + targeted reads. Vendor wrapper inventory at Item 4 verified.
- `crates/cintx-oracle/src/compare.rs` — partial read (lines 670-900). Arity-2 transpose, arity-3/4 direct compare patterns verified.
- `crates/cintx-oracle/tests/safe_api_arity2_parity.rs` — full file read (838 lines). Pattern source for new files.
- `crates/cintx-oracle/tests/two_electron_parity.rs` — partial read + grep. Arity-4 direct-compare pattern at lines 273-289 verified.
- `crates/cintx-oracle/tests/center_3c2e_parity.rs` — partial read (lines 200-300). Existing `int3c2e_sph` parity precedent verified.
- `crates/cintx-oracle/tests/oracle_gate_closure.rs` — partial read (lines 730-770). `#[cfg(feature = "with-4c1e")]` per-test pattern verified.
- `crates/cintx-oracle/build.rs` — partial read (lines 220-260). Supplemental header declarations for `int3c1e_p2_sph` etc. confirmed.
- `crates/cintx-cubecl/src/executor.rs` — grep + targeted read. `supports`, `query_workspace`, `execute` signature verified.
- `crates/cintx-cubecl/src/kernels/mod.rs` — partial read (lines 50, 198-225, 295-310). `supports_canonical_family` covers all 5 base families + 4c1e (feature-gated).
- `.planning/phases/17-real-integral-evaluation-in-safe-api/17-CONTEXT.md`, `17-RESEARCH.md`, `17-PATTERNS.md`, `17-VERIFICATION.md` — all 4 documents read in full.
- `.planning/phases/15-oracle-tolerance-unification-manifest-lock-closure/15-CONTEXT.md` — partial read for atol baseline.
- `.planning/REQUIREMENTS.md` — full file read.
- `.planning/STATE.md` — full file read.
- `.planning/config.json` — full file read. `nyquist_validation: true`, `commit_docs: true`.
- `.planning/ROADMAP.md` — Phase 18 section grep + targeted read.
- `.planning/phases/18-sessionrequest-arity-ge3-dispatch/18-CONTEXT.md` — full file read.
- `CLAUDE.md` (project root) — full file read. CubeCL primary backend, Rust 1.94.0 pin, thiserror v2 for library errors, anyhow for app/xtask, `/mnt/data` artifact path.

### Secondary (MEDIUM confidence)

- pyscf source (referenced in CONTEXT.md, not read in this session): `aosym` convention `s1, s2ij, s2kl, s4, s8` — names verified against multiple cintx-internal references and CONTEXT.md text.

### Tertiary (LOW confidence)

- Per-evaluation runtime estimates in Item 6 / R3 — based on order-of-magnitude reasoning, not measured.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all crate path deps, versions, and pattern sources verified by direct file reads
- Architecture: HIGH — chunk loop and executor swap (Phase 17) leaves arity-3/4 path immediately usable; verified by `CubeClExecutor::supports` covering all relevant families and `ShellTuple::try_from_iter` accepting up to 4 shells
- Pitfalls: HIGH — all 6 pitfalls grounded in actual code inspection (compare.rs layout differences, with-4c1e gating, error enum exhaustiveness, etc.)
- R1 / R2 / R5 / R6 findings: HIGH — contradictions/gaps confirmed by grep on manifest CSV + Rust source + existing test code paths

**Research date:** 2026-05-12
**Valid until:** 2026-06-12 (30-day validity; stable internal-API workspace)

## RESEARCH COMPLETE
