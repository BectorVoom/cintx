# Phase 18: SessionRequest Arity ≥3 Dispatch — Pattern Map

**Mapped:** 2026-05-12
**Files analyzed:** 14 (2 new tests, 1 optional new helper, 11 modified)
**Analogs found:** 14 / 14
**User R1 resolution:** **OPTION 3** — *Add plain `int3c2e_cart` and `int3c2e_sph` operator-kind rows to the manifest.* The arity-3 oracle set stays at 8 symbols (not 6). The non-trivial pattern map for this resolution is captured in §"Manifest-row additions" below.

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/cintx-oracle/tests/safe_api_arity3_parity.rs` (NEW) | integration test | request-response (per-tuple buffer compare) | `crates/cintx-oracle/tests/safe_api_arity2_parity.rs` + `tests/center_3c2e_parity.rs:222-287` | exact |
| `crates/cintx-oracle/tests/safe_api_arity4_parity.rs` (NEW) | integration test | request-response (per-tuple buffer compare) | `crates/cintx-oracle/tests/safe_api_arity2_parity.rs` + `crates/cintx-oracle/src/compare.rs:787-797` | exact |
| `crates/cintx-oracle/tests/safe_api_helpers.rs` (NEW, OPTIONAL) | test helper module | utility | `crates/cintx-oracle/tests/safe_api_arity2_parity.rs:236-292` (`collect_safe_api_matrix` body) | exact |
| `crates/cintx-oracle/src/vendor_ffi.rs` (MODIFY: add 2 wrappers) | FFI shim | request-response | `vendor_ffi.rs:465-490` (`vendor_int3c1e_p2_cart`) + `vendor_ffi.rs:493-518` (`vendor_int3c2e_ip1_cart`) | exact |
| `crates/cintx-core/src/operator.rs` (MODIFY: add `AoSymmetry`) | domain enum | N/A | `crates/cintx-core/src/operator.rs:4-20` (`Representation` enum + `Display`) | exact |
| `crates/cintx-core/src/lib.rs` (MODIFY: re-export) | crate re-export | N/A | `crates/cintx-core/src/lib.rs:18` (existing `pub use operator::{OperatorId, Representation}`) | exact |
| `crates/cintx-runtime/src/options.rs` (MODIFY: add `aosym` field) | config struct | N/A | `crates/cintx-runtime/src/options.rs:103-117` (`f12_zeta` field) | exact |
| `crates/cintx-rs/src/api.rs` (MODIFY: preflight + rustdoc + `aosym_error_path` test) | safe API surface | request-response | `crates/cintx-rs/src/api.rs:63-80` (`query_workspace`) + `crates/cintx-rs/src/api.rs:128-131` (`f12_zeta` propagation) | exact |
| `crates/cintx-rs/src/error.rs` (MODIFY: new variant + kind) | error enum | N/A | `crates/cintx-rs/src/error.rs:14-34` (existing `FacadeError` + `kind()` match) | exact |
| `crates/cintx-rs/src/prelude.rs` (MODIFY: re-export `AoSymmetry`) | crate re-export | N/A | `crates/cintx-rs/src/prelude.rs:33` (`pub use cintx_core::Representation`) | exact |
| `crates/cintx-rs/src/builder.rs` (MODIFY, OPTIONAL: `.aosym(…)`) | builder method | N/A | `crates/cintx-rs/src/builder.rs:87-94` (`.f12_zeta(…)`) | exact |
| `crates/cintx-ops/generated/compiled_manifest.lock.json` (MODIFY: +2 rows) | generated source-of-truth | N/A | `compiled_manifest.lock.json:275-306` (`int3c1e_cart`/`int3c1e_sph` rows — also stable arity-3, no feature flag) | exact |
| `crates/cintx-ops/src/generated/api_manifest.rs` (REGEN by build.rs) | generated Rust | N/A | `api_manifest.rs:300-332` (`int3c1e_cart`/`int3c1e_sph` `ManifestEntry` blocks) + `api_manifest.rs:2221+` (`OPERATOR_DESCRIPTORS` table) | exact (auto-regenerated) |
| `crates/cintx-ops/src/generated/api_manifest.csv` (REGEN by build.rs) | generated CSV | N/A | `api_manifest.csv:19-20` (`int3c1e_cart` / `int3c1e_sph` rows) | exact (auto-regenerated) |
| `crates/cintx-ops/src/resolver.rs` (MODIFY: misc-wrapper macro arm) | symbol routing | N/A | `crates/cintx-ops/src/resolver.rs:314-322` (`misc_wrapper_macro` match) | role-match |

**Note on regenerated files:** `api_manifest.rs` and `api_manifest.csv` are produced by `crates/cintx-ops/build.rs` from the lock file. Plans must edit *only* the lock file; the generator emits the rest at build time.

---

## Manifest-Row Additions (R1 Resolution Pattern — non-trivial)

Adding plain `int3c2e_cart` (OperatorId 21) and `int3c2e_sph` (OperatorId 22) is the **non-trivial** pattern map per the user's instruction. The lock file is the single source of truth; everything else either auto-regenerates or follows a routing rule.

### Step 1: Lock file (single edit site)

**Analog:** `crates/cintx-ops/generated/compiled_manifest.lock.json:275-306` (the two `int3c1e_cart` and `int3c1e_sph` entries — *plain* operator rows in an arity-3 base-profile family, exactly the shape needed).

**Existing pattern at lines 275-306** (verified — `int3c1e_cart` shown):
```json
{
  "id": {
    "family": "3c1e",
    "operator": "overlap",
    "representation": "cart",
    "symbol": "int3c1e_cart"
  },
  "oracle_covered": true,
  "profiles": [
    "base",
    "with-f12",
    "with-4c1e",
    "with-f12+with-4c1e"
  ],
  "stability": "stable"
}
```

**Rows to ADD** (insert AFTER the existing `int3c2e_ip1_spinor` entry at line 354, BEFORE the `int4c1e_cart` entry at line 355 — preserves family ordering 3c1e → 3c2e → 4c1e):

```json
{
  "id": {
    "family": "3c2e",
    "operator": "electron-repulsion",
    "representation": "cart",
    "symbol": "int3c2e_cart"
  },
  "oracle_covered": true,
  "profiles": [
    "base",
    "with-f12",
    "with-4c1e",
    "with-f12+with-4c1e"
  ],
  "stability": "stable"
},
{
  "id": {
    "family": "3c2e",
    "operator": "electron-repulsion",
    "representation": "sph",
    "symbol": "int3c2e_sph"
  },
  "oracle_covered": true,
  "profiles": [
    "base",
    "with-f12",
    "with-4c1e",
    "with-f12+with-4c1e"
  ],
  "stability": "stable"
}
```

**Pattern-key copy mapping:**
- `family: "3c2e"` — same as existing `int3c2e_ip1_*` siblings; routes through `kernels::resolve_family_name("3c2e") → launch_center_3c2e` (verified `crates/cintx-cubecl/src/kernels/mod.rs:31`).
- `operator: "electron-repulsion"` — same `operator_name` as `int3c2e_ip1_*` (verified `api_manifest.rs:335, 352, 369`). This keeps `Resolver::resolve("3c2e", "electron-repulsion", Representation::Cart)` deterministic; it will return the first cart match. The existing `int3c2e_ip1_cart` row also has `operator_name: "electron-repulsion"` — see "Resolver disambiguation note" below.
- `stability: "stable"`, `profiles: [base|with-f12|with-4c1e|with-f12+with-4c1e]` — same as `int3c1e_{cart,sph}` (no feature flag, stable in every profile).
- `oracle_covered: true` — these symbols will be Phase 18 parity-gated, so coverage is true on land.

**Position in the lock-file entries array:** Insert as **entries[21]** and **entries[22]** (between `int3c2e_ip1_spinor` at index 20 and `int4c1e_cart` at index 21 in the *current* lock). This will shift all downstream OperatorIds by +2.

### Step 2: Side-effects of OperatorId shift

The build.rs generator (`crates/cintx-ops/build.rs:151-163`) emits `OPERATOR_DESCRIPTORS` with `OperatorId::new(idx)` matching the array position. After inserting at index 21+22, the **post-shift OperatorId table for the 12 Phase 18 symbols** becomes:

```
int3c1e_p2_cart       OperatorId::new(15)   // unchanged
int3c1e_p2_sph        OperatorId::new(16)   // unchanged
int3c1e_cart          OperatorId::new(17)   // unchanged
int3c1e_sph           OperatorId::new(18)   // unchanged
int3c2e_ip1_cart      OperatorId::new(19)   // unchanged
int3c2e_ip1_sph       OperatorId::new(20)   // unchanged
int3c2e_ip1_spinor    OperatorId::new(21)   // unchanged
int3c2e_cart          OperatorId::new(22)   // NEW
int3c2e_sph           OperatorId::new(23)   // NEW
int4c1e_cart          OperatorId::new(24)   // was 22, +2
int4c1e_sph           OperatorId::new(25)   // was 23, +2
int2e_cart            OperatorId::new(9)    // unchanged
int2e_sph             OperatorId::new(10)   // unchanged
```

**Anti-pattern:** Do NOT preserve the old `int4c1e_*` IDs (22, 23) by inserting elsewhere. The lock-file ordering must follow family grouping for `manifest_audit` consistency. The planner's affected-test list must update any place that hard-codes `OperatorId::new(22)` or `OperatorId::new(23)` for 4c1e:

```bash
# Pre-implementation grep (executor MUST run this and update each hit):
grep -rn "OperatorId::new(22)\|OperatorId::new(23)\|INT4C1E_CART_OPERATOR_ID.*22\|INT4C1E_SPH_OPERATOR_ID.*23\|operator_id.*22\|operator_id.*23" \
    /home/user/Documents/workspace/cintx/crates/ \
    /home/user/Documents/workspace/cintx/xtask/
```

**Known hard-coded sites to update** (from CONTEXT.md anti-patterns + existing source greps):
- `crates/cintx-rs/src/api.rs:501` — `const INT4C1E_CART_OPERATOR_ID: u32 = 22;` → `24`.

### Step 3: Resolver — misc.h legacy-wrapper symbol-classification arm

**Analog:** `crates/cintx-ops/src/resolver.rs:314-322` (`misc_wrapper_macro` match) — the function the `legacy_wrapper_manifest_matches_misc` test (resolver.rs:391-417) uses to validate operator↔legacy parity.

**Current pattern (lines 314-322):**
```rust
fn misc_wrapper_macro(base_symbol: &str) -> Option<MiscWrapperMacro> {
    match base_symbol {
        "int1e_ovlp" | "int1e_nuc" | "int2e" | "int2c2e" | "int3c1e" | "int3c1e_p2" | "int3c2e_ip1" => {
            Some(MiscWrapperMacro::AllCint)
        }
        "int1e_kin" => Some(MiscWrapperMacro::AllCint1e),
        _ => None,
    }
}
```

**The challenge:** This match arm checks `base_symbol_from_operator` output (resolver.rs:301-312) which strips `_cart`/`_sph`/`_spinor`. After Step 1, `int3c2e_cart` and `int3c2e_sph` will produce `base_symbol = "int3c2e"`. The current match returns `None` for `"int3c2e"`, and `legacy_wrapper_manifest_matches_misc` panics with `missing misc.h wrapper macro classification for int3c2e`.

**Two options for the planner (choose ONE and document in PLAN.md):**

**Option A — Treat `int3c2e` as having NO misc.h legacy wrappers.** Change the resolver test (`base_symbol_from_operator` filter or `misc_wrapper_macro` to return `Some(NoLegacy)`) to make `int3c2e` an *explicit exception*. The plain `cint3c2e_*` wrappers do NOT exist in libcint 6.1.3 upstream `src/misc.h` (only `cint3c2e_ip1_*` exists), so this matches reality. **Recommended.**

**Sketch:**
```rust
// crates/cintx-ops/src/resolver.rs:314-322 (PROPOSED edit)
fn misc_wrapper_macro(base_symbol: &str) -> Option<MiscWrapperMacro> {
    match base_symbol {
        "int1e_ovlp" | "int1e_nuc" | "int2e" | "int2c2e" | "int3c1e" | "int3c1e_p2" | "int3c2e_ip1" => {
            Some(MiscWrapperMacro::AllCint)
        }
        "int1e_kin" => Some(MiscWrapperMacro::AllCint1e),
        // int3c2e (plain) has no misc.h wrapper in libcint 6.1.3 upstream — distinct from int3c2e_ip1.
        // Returning None here keeps `legacy_wrapper_manifest_matches_misc` test green without
        // requiring synthetic legacy entries.
        "int3c2e" => None,
        _ => None,
    }
}
```

Then update the test (`legacy_wrapper_manifest_matches_misc` body at lines 391-417) to tolerate `None`:
```rust
// crates/cintx-ops/src/resolver.rs:401-404 (PROPOSED edit)
for base_symbol in base_symbols {
    let Some(macro_kind) = misc_wrapper_macro(&base_symbol) else {
        continue;  // base symbols without misc.h wrappers (e.g. int3c2e plain) do not contribute
    };
    expected.extend(expected_legacy_wrapper_symbols(&base_symbol, macro_kind));
}
```

This is a **minimal targeted edit**: one new arm in `misc_wrapper_macro`, one early-continue in the test, no new legacy entries needed.

**Option B — Add six new `cint3c2e_*` legacy rows.** Synthesize `cint3c2e_cart`, `cint3c2e_sph`, `cint3c2e`, `cint3c2e_cart_optimizer`, `cint3c2e_sph_optimizer`, `cint3c2e_optimizer` entries in the lock file. This matches the `expected_legacy_wrapper_symbols` shape (resolver.rs:324-339). **Rejected** because these wrappers do NOT exist in vendored libcint 6.1.3 `src/misc.h` — the resolver test would pass, but build-time linkage to a non-existent C symbol would fail downstream (compat raw path). Use Option A.

### Step 4: Re-build the generated files

```
cargo clean -p cintx-ops
cargo build -p cintx-ops --locked
```

`build.rs` regenerates `src/generated/api_manifest.rs` (entries 21, 22 inserted) and `src/generated/api_manifest.csv` (two new lines after row 22). The Rust block format that will be emitted (auto, no manual edit):

```rust
// AUTO-GENERATED entry pattern (verified format at api_manifest.rs:316-332)
ManifestEntry {
    family_name: "3c2e",
    operator_name: "electron-repulsion",
    symbol_name: "int3c2e_cart",
    category: "3c2e",
    arity: 3,
    forms: &["cart"],
    component_rank: "",
    feature_flag: FeatureFlag::None,
    stability: Stability::Stable,
    declared_in: "unknown",
    compiled_in_profiles: &["base", "with-f12", "with-4c1e", "with-f12+with-4c1e"],
    oracle_covered: true,
    helper_kind: HelperKind::Operator,
    canonical_family: "3c2e",
    representation: RepresentationSupport::new(true, false, false),
},
ManifestEntry {
    family_name: "3c2e",
    operator_name: "electron-repulsion",
    symbol_name: "int3c2e_sph",
    category: "3c2e",
    arity: 3,
    forms: &["sph"],
    component_rank: "",
    feature_flag: FeatureFlag::None,
    stability: Stability::Stable,
    declared_in: "unknown",
    compiled_in_profiles: &["base", "with-f12", "with-4c1e", "with-f12+with-4c1e"],
    oracle_covered: true,
    helper_kind: HelperKind::Operator,
    canonical_family: "3c2e",
    representation: RepresentationSupport::new(false, true, false),
},
```

The kernel `launch_center_3c2e` already produces the *plain* 3c2e integral output (verified by `compare.rs:823-833` and the Item 5 RESEARCH finding) — adding the manifest rows is therefore a label/routing change only; no new kernel code.

### Resolver disambiguation note

`Resolver::resolve("3c2e", "electron-repulsion", Representation::Cart)` (resolver.rs:262-287) returns the **first** match by `(family, operator)` whose `RepresentationSupport` covers Cart. After Step 1, the cart matches for `("3c2e", "electron-repulsion")` will be `int3c2e_ip1_cart` (id 19) *and* `int3c2e_cart` (id 22). The iteration order is `MANIFEST_ENTRIES` array order, so `Resolver::resolve` will return `int3c2e_ip1_cart` (id 19) first — preserving Phase 17 behavior for all existing call sites. The new plain `int3c2e_{cart,sph}` symbols are reachable **only via `OperatorId::new(22)` / `OperatorId::new(23)`** through `Resolver::descriptor` or `Resolver::descriptor_by_symbol("int3c2e_cart")` — exactly what the Phase 18 oracle tests use. Verified by reading `crates/cintx-ops/src/resolver.rs:262-287`.

---

## Pattern Assignments

### `crates/cintx-oracle/tests/safe_api_arity3_parity.rs` (test, request-response)

**Analog:** `crates/cintx-oracle/tests/safe_api_arity2_parity.rs` (file-structure, fixture, per-symbol tests) + `crates/cintx-oracle/tests/center_3c2e_parity.rs:222-287` (no-transpose 3-shell Cartesian sweep + nonzero sentinel) + `crates/cintx-oracle/src/compare.rs:811-833` (direct cintx-vs-vendor buffer compare for `int3c1e_sph`/`int3c2e_*_sph`).

**Module gate** (verbatim from `safe_api_arity2_parity.rs:13`):
```rust
#![cfg(any(feature = "cpu", feature = "rocm"))]
```

**Imports block** (mirrors `safe_api_arity2_parity.rs:15-22`):
```rust
use cintx_compat::raw::{
    ATM_SLOTS, ANG_OF, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF,
    NUC_MOD_OF, POINT_NUC, PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA,
};
use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell, ShellTuple};
use cintx_rs::SessionRequest;
use cintx_runtime::ExecutionOptions;
use std::sync::Arc;
```

**Fixture helpers** — copy verbatim from `safe_api_arity2_parity.rs:34-167` (`N_SHELLS`, `build_h2o_sto3g()` — the PTR_ENV_START-aware raw arrays) and `:169-228` (`arc_f64`, `build_h2o_sto3g_safe_basis`). OR factor into `safe_api_helpers.rs` (see optional module below) and `mod common;`/`use common::*;` from both arity-3 and arity-4 files.

**Tolerance constants** (file-top, per Phase 15 unified atol from CONTEXT.md D-15):
```rust
const ATOL: f64 = 1e-12;
const RTOL: f64 = 0.0;
```

**Per-tuple safe-API evaluator** — the *new* arity-generic helper. Different from arity-2's `collect_safe_api_matrix` (which assembles a full ni×nj matrix and applies the row-major→row-major transpose at lines 280-292). Arity-3/4 needs the raw `owned_values` buffer with NO transpose because cintx 2e/3c1e/3c2e/4c1e kernels write F-order matching vendor directly (verified by `compare.rs:787-797` for 2e, `:811-821` for 3c1e_sph, `:823-833` for 3c2e_ip1_sph):

```rust
/// Arity-agnostic per-tuple buffer collector. `shells` must have length 2-4.
/// Returns the safe-API `owned_values` buffer for direct comparison against
/// vendor output — NO transpose required for arity ≥ 3 (compare.rs:787-797).
fn collect_safe_api_tuple_buffer(
    operator_id: OperatorId,
    rep: Representation,
    basis: &BasisSet,
    tuple_shells: &[Arc<Shell>],
) -> Vec<f64> {
    let shell_tuple = ShellTuple::try_from_iter(tuple_shells.iter().cloned())
        .expect("tuple within SHELL_TUPLE_CAPACITY=4");
    let request = SessionRequest::new(
        operator_id,
        rep,
        basis,
        shell_tuple,
        ExecutionOptions::default(),
    );
    let query = request
        .query_workspace()
        .expect("query_workspace must succeed for a valid safe-API request");
    let output = query
        .evaluate()
        .expect("evaluate must succeed for arity-3 dispatch");
    output.tensor.owned_values
}
```

**`count_mismatches`** — copy verbatim from `safe_api_arity2_parity.rs:300-321`.

**Per-symbol test pattern** (combines `safe_api_arity2_parity.rs:614-635` per-symbol scaffold with `center_3c2e_parity.rs:222-287` triple-sweep + nonzero sentinel):

```rust
#[test]
#[cfg(has_vendor_libcint)]
fn test_int3c1e_sph_safe_api_parity() {
    let (atm, bas, env) = build_h2o_sto3g();
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Spheric);

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

                let safe_out = collect_safe_api_tuple_buffer(
                    OperatorId::new(18),  // int3c1e_sph (post-shift unchanged, see R1 §Step 2)
                    Representation::Spheric,
                    &basis,
                    &[shells[i].clone(), shells[j].clone(), shells[k].clone()],
                );

                let mut vendor_out = vec![0.0_f64; n_elem];
                let shls = [i as i32, j as i32, k as i32];
                cintx_oracle::vendor_ffi::vendor_int3c1e_sph(
                    &mut vendor_out, &shls, &atm, natm, &bas, nbas, &env);

                if safe_out.iter().any(|&v| v.abs() > 1e-18)
                    || vendor_out.iter().any(|&v| v.abs() > 1e-18) {
                    any_nonzero = true;
                }

                // Arity-3: cintx and vendor agree in F-order — no transpose.
                // Precedent: compare.rs:811-821, center_3c2e_parity.rs:243-277.
                total_mismatches += count_mismatches(&vendor_out, &safe_out, ATOL, RTOL);
                tuples_checked += 1;
            }
        }
    }

    assert!(any_nonzero,
        "int3c1e_sph safe-API outputs are all zeros over {tuples_checked} triples");
    assert_eq!(total_mismatches, 0,
        "int3c1e_sph safe API: {total_mismatches} elements exceed atol={ATOL:.0e}/rtol={RTOL:.0e} \
         vs vendored libcint over {tuples_checked} triples");
}
```

**OperatorId mapping (post-R1 shift)** — used by the 8 arity-3 tests:
```
int3c1e_p2_cart       OperatorId::new(15)
int3c1e_p2_sph        OperatorId::new(16)
int3c1e_cart          OperatorId::new(17)
int3c1e_sph           OperatorId::new(18)
int3c2e_ip1_cart      OperatorId::new(19)
int3c2e_ip1_sph       OperatorId::new(20)
int3c2e_cart          OperatorId::new(22)   // NEW
int3c2e_sph           OperatorId::new(23)   // NEW
```

**Vendor function per symbol** (verified vendor_ffi.rs inventory from RESEARCH Item 4):
| Symbol | Vendor wrapper | Notes |
|--------|----------------|-------|
| `int3c1e_cart` | `vendor_int3c1e_cart` (vendor_ffi.rs:440) | exists |
| `int3c1e_sph` | `vendor_int3c1e_sph` (vendor_ffi.rs:173) | exists |
| `int3c1e_p2_cart` | `vendor_int3c1e_p2_cart` (vendor_ffi.rs:468) | exists |
| `int3c1e_p2_sph` | `vendor_int3c1e_p2_sph` | **NEW — add per vendor_ffi.rs pattern below** |
| `int3c2e_ip1_cart` | `vendor_int3c2e_ip1_cart` (vendor_ffi.rs:496) | exists |
| `int3c2e_ip1_sph` | `vendor_int3c2e_ip1_sph` | **NEW — add per vendor_ffi.rs pattern below** |
| `int3c2e_cart` | `vendor_int3c2e_cart` | **NEW — must also add** (since R1 resolution promotes plain int3c2e) |
| `int3c2e_sph` | `vendor_int3c2e_sph` (vendor_ffi.rs:204) | exists |

**Anti-patterns** (from RESEARCH Pitfall 2):
- Do NOT apply the arity-2 `pair_values[ii * nj + jj]` row-major assembly here. Arity-3 cintx kernels write F-order; cintx and vendor agree byte-for-byte without transpose.
- Do NOT use a single parametric loop over the 8 operators (per CONTEXT.md D-07 / D-14 anti-pattern: per-symbol `#[test]` functions are required for CI bisection).

---

### `crates/cintx-oracle/tests/safe_api_arity4_parity.rs` (test, request-response)

**Analog:** `crates/cintx-oracle/tests/safe_api_arity2_parity.rs` (file structure) + `crates/cintx-oracle/tests/two_electron_parity.rs:273-289` (5⁴ quartet sweep, no transpose) + `crates/cintx-oracle/tests/oracle_gate_closure.rs:737-739` (`#[cfg(feature = "with-4c1e")]` per-test gate).

**Imports + fixture + tolerance constants** — identical to `safe_api_arity3_parity.rs` (above).

**Per-symbol quartet sweep pattern** (mirrors arity-3 with one extra `for l in 0..N_SHELLS` loop):

```rust
#[test]
#[cfg(has_vendor_libcint)]
fn test_int2e_sph_safe_api_parity() {
    let (atm, bas, env) = build_h2o_sto3g();
    let (basis, shells) = build_h2o_sto3g_safe_basis(Representation::Spheric);

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;

    let mut total_mismatches = 0usize;
    let mut any_nonzero = false;
    let mut tuples_checked = 0usize;

    for i in 0..N_SHELLS {
        for j in 0..N_SHELLS {
            for k in 0..N_SHELLS {
                for l in 0..N_SHELLS {
                    let ni = shells[i].ao_per_shell();
                    let nj = shells[j].ao_per_shell();
                    let nk = shells[k].ao_per_shell();
                    let nl = shells[l].ao_per_shell();
                    let n_elem = ni * nj * nk * nl;

                    let safe_out = collect_safe_api_tuple_buffer(
                        OperatorId::new(10),  // int2e_sph (unchanged)
                        Representation::Spheric,
                        &basis,
                        &[shells[i].clone(), shells[j].clone(),
                          shells[k].clone(), shells[l].clone()],
                    );

                    let mut vendor_out = vec![0.0_f64; n_elem];
                    let shls = [i as i32, j as i32, k as i32, l as i32];
                    cintx_oracle::vendor_ffi::vendor_int2e_sph(
                        &mut vendor_out, &shls, &atm, natm, &bas, nbas, &env);

                    if safe_out.iter().any(|&v| v.abs() > 1e-18)
                        || vendor_out.iter().any(|&v| v.abs() > 1e-18) {
                        any_nonzero = true;
                    }
                    total_mismatches += count_mismatches(&vendor_out, &safe_out, ATOL, RTOL);
                    tuples_checked += 1;
                }
            }
        }
    }

    assert!(any_nonzero,
        "int2e_sph safe-API outputs are all zeros over {tuples_checked} quartets");
    assert_eq!(total_mismatches, 0,
        "int2e_sph safe API: {total_mismatches} elements exceed atol={ATOL:.0e} \
         over {tuples_checked} quartets");
}
```

**OperatorId mapping (post-R1 shift)** — used by the 4 arity-4 tests:
```
int2e_cart            OperatorId::new(9)    // unchanged
int2e_sph             OperatorId::new(10)   // unchanged
int4c1e_cart          OperatorId::new(24)   // was 22, +2 from R1
int4c1e_sph           OperatorId::new(25)   // was 23, +2 from R1
```

**`with-4c1e` per-test gating** (mirrors `oracle_gate_closure.rs:737-739` verbatim — verified pattern):

```rust
#[test]
#[cfg(feature = "with-4c1e")]
#[cfg(has_vendor_libcint)]
fn test_int4c1e_sph_safe_api_parity() {
    // ... same body as test_int2e_sph_safe_api_parity, with:
    //   OperatorId::new(25)  // int4c1e_sph
    //   vendor_ffi::vendor_int4c1e_sph(...)
    // ...
}
```

**Both `#[cfg(...)]` attributes stack additively** — the test compiles only when both `feature = "with-4c1e"` AND `has_vendor_libcint` are active. Module-level `#![cfg(feature = "with-4c1e")]` would break the `int2e_*` tests under the base profile (RESEARCH Pitfall 4) — do NOT use module-level gating.

**Vendor function per symbol** (verified):
| Symbol | Vendor wrapper |
|--------|----------------|
| `int2e_cart` | `vendor_int2e_cart` (vendor_ffi.rs:384) |
| `int2e_sph` | `vendor_int2e_sph` (vendor_ffi.rs:111) |
| `int4c1e_cart` | `vendor_int4c1e_cart` (vendor_ffi.rs:269) |
| `int4c1e_sph` | `vendor_int4c1e_sph` (vendor_ffi.rs:238) |

---

### `crates/cintx-oracle/tests/safe_api_helpers.rs` (NEW, OPTIONAL — Claude's discretion item)

**Analog:** `crates/cintx-oracle/tests/safe_api_arity2_parity.rs:34-321` (the 5 helpers at top of the file are the natural extraction target).

**Decision (per CONTEXT.md "Claude's discretion" + RESEARCH §"Recommendations to the Planner" item 5):** Factor *yes* — 12 new tests across 2 files share enough fixture and comparison code to justify a shared module. Module path: `crates/cintx-oracle/tests/safe_api_helpers.rs` (single-file flat layout; not `common/mod.rs` so other oracle tests are unaffected).

**Items to extract** (copy from `safe_api_arity2_parity.rs`):
- `N_SHELLS` constant (line 34)
- `build_h2o_sto3g()` raw-array fixture (lines 36-167)
- `arc_f64`, `build_h2o_sto3g_safe_basis()` typed-basis fixture (lines 169-228)
- `count_mismatches` (lines 300-321)
- `nsph`, `ncart` (lines 328-335)
- **NEW** `collect_safe_api_tuple_buffer` (see arity-3 pattern above)
- **KEEP private to arity-2** `collect_safe_api_matrix` (the row-major matrix assembler at lines 236-292 stays in the arity-2 file because arity-3/4 use the buffer-direct path)

**Wiring from test files** — Cargo loads `tests/*.rs` as separate binaries; bringing a sibling test file into scope requires either `mod safe_api_helpers;` at the top of each test file or moving the helper to a `tests/common/mod.rs` (Rust convention). Either works; the planner picks one and applies consistently. The simplest mechanical change is:

```rust
// At top of safe_api_arity3_parity.rs (and safe_api_arity4_parity.rs):
mod safe_api_helpers;
use safe_api_helpers::*;
```

Cargo treats `tests/safe_api_helpers.rs` as a test binary by default; adding `harness = false` is NOT needed for the `mod` declaration pattern, but rustc will emit "unused crate-level attribute" warnings unless the file's `#[cfg(any(feature = "cpu", feature = "rocm"))]` is removed (module-level cfg on a `mod`-included file is silently ignored). **Cleaner alternative:** use the `tests/common/mod.rs` Cargo convention — `cargo test` does NOT compile `tests/common/mod.rs` as its own binary, so `pub fn` items don't trigger dead-code warnings.

The Phase 17 file (`safe_api_arity2_parity.rs`) currently keeps everything inline. If the planner prefers minimal scope, **skip this extraction in Phase 18** and let `safe_api_arity3_parity.rs` and `safe_api_arity4_parity.rs` duplicate the ~190-line fixture block. **Default: skip extraction.** The marginal token cost is bearable; the wiring complexity is real.

---

### `crates/cintx-oracle/src/vendor_ffi.rs` (MODIFY — add 3 wrappers)

**Analog:** `crates/cintx-oracle/src/vendor_ffi.rs:465-490` (`vendor_int3c1e_p2_cart`) and `:493-518` (`vendor_int3c2e_ip1_cart`) — the closest existing arity-3 sph wrappers to mirror.

**Wrappers to add (3 total):**

1. **`vendor_int3c1e_p2_sph`** (R4 from RESEARCH; mirror lines 465-490 with `sph` suffix):
```rust
/// Evaluate int3c1e_p2_sph for a single shell triple using vendored libcint.
///
/// `shls` is `[i, j, k]` — three shell indices (3-center 1-electron integral, p2 variant, spherical).
pub fn vendor_int3c1e_p2_sph(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c1e_p2_sph(
            out.as_mut_ptr(),
            ptr::null_mut(),
            shls.as_ptr() as *mut i32,
            atm.as_ptr() as *mut i32,
            natm,
            bas.as_ptr() as *mut i32,
            nbas,
            env.as_ptr() as *mut f64,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    }
}
```

2. **`vendor_int3c2e_ip1_sph`** (R4; mirror lines 493-518 with `sph` suffix):
```rust
/// Evaluate int3c2e_ip1_sph for a single shell triple using vendored libcint.
///
/// `shls` is `[i, j, k]` — three shell indices (3-center 2-electron integral, ip1 variant, spherical).
pub fn vendor_int3c2e_ip1_sph(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c2e_ip1_sph(
            out.as_mut_ptr(),
            ptr::null_mut(),
            shls.as_ptr() as *mut i32,
            atm.as_ptr() as *mut i32,
            natm,
            bas.as_ptr() as *mut i32,
            nbas,
            env.as_ptr() as *mut f64,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    }
}
```

3. **`vendor_int3c2e_cart`** (R1 promotion — plain 3c2e cart variant; mirror `vendor_int3c2e_sph` at lines 198-227 with `cart` suffix):
```rust
/// Evaluate int3c2e_cart for a single shell triple using vendored libcint.
///
/// `shls` is `[i, j, k]` — three shell indices (3-center 2-electron integral, Cartesian).
pub fn vendor_int3c2e_cart(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c2e_cart(
            out.as_mut_ptr(),
            ptr::null_mut(),
            shls.as_ptr() as *mut i32,
            atm.as_ptr() as *mut i32,
            natm,
            bas.as_ptr() as *mut i32,
            nbas,
            env.as_ptr() as *mut f64,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    }
}
```

**Binding verification (executor MUST run before adding):**
```bash
grep -n "int3c1e_p2_sph\|int3c2e_ip1_sph\|int3c2e_cart" \
    /home/user/Documents/workspace/cintx/crates/cintx-oracle/build.rs
```
RESEARCH A3 confirms `int3c1e_p2_sph` is in the supplemental header (build.rs:227). `int3c2e_ip1_sph` and `int3c2e_cart` are stable `_sph`/`_cart` variants and should be in `cint_funcs.h` already (auto-bound via bindgen). If `ffi::int3c2e_cart` is missing, add it to the supplemental header per the existing pattern at build.rs:220-260.

All three wrappers must be `#[cfg(has_vendor_libcint)]`-gated to match the surrounding file convention. Verified: the file already declares `#[cfg(has_vendor_libcint)]` near top; new wrappers inherit the file-level guard if added inside the same `cfg`-block. Otherwise, individually annotate each.

---

### `crates/cintx-core/src/operator.rs` (MODIFY — add `AoSymmetry`)

**Analog:** `crates/cintx-core/src/operator.rs:4-20` — the `Representation` enum + `Display` impl, the IMMEDIATE neighbor.

**Existing pattern (lines 4-20, verbatim verified):**
```rust
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Representation {
    Cart,
    Spheric,
    Spinor,
}

impl fmt::Display for Representation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Representation::Cart => write!(f, "Cart"),
            Representation::Spheric => write!(f, "Spheric"),
            Representation::Spinor => write!(f, "Spinor"),
        }
    }
}
```

**Code to ADD** (place AFTER the `Representation` block, BEFORE the `OperatorId` block at line 22):
```rust
/// AO symmetry packing convention (pyscf-compatible naming).
///
/// Phase 18 ships `S1` only; every other variant returns
/// `FacadeError::UnsupportedAoSymmetry` from `SessionRequest::query_workspace`.
/// `Display` emits the lowercase pyscf form (`s1`, `s2ij`, `s2kl`, `s4`, `s8`)
/// so error messages and downstream interop read directly.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum AoSymmetry {
    #[default]
    S1,
    S2ij,
    S2kl,
    S4,
    S8,
}

impl fmt::Display for AoSymmetry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

**Note** on `#[default]`: the existing `Representation` does NOT derive `Default`. Per CONTEXT.md "Claude's discretion" + RESEARCH §"Recommendations to the Planner" item 9, `AoSymmetry::default() == S1` is the intended behavior — `#[default]` attribute on the `S1` variant is required for the `derive(Default)`.

---

### `crates/cintx-core/src/lib.rs` (MODIFY — extend re-export)

**Analog:** `crates/cintx-core/src/lib.rs:18` (verbatim verified):
```rust
pub use operator::{OperatorId, Representation};
```

**Change** — extend to include `AoSymmetry`:
```rust
pub use operator::{AoSymmetry, OperatorId, Representation};
```

Alphabetical order preserved.

---

### `crates/cintx-runtime/src/options.rs` (MODIFY — add `aosym` field)

**Analog:** `crates/cintx-runtime/src/options.rs:113-117` — the `f12_zeta` field declaration.

**Existing pattern (lines 103-117 verbatim verified):**
```rust
#[derive(Clone, Debug, Default)]
pub struct ExecutionOptions {
    pub memory_limit_bytes: Option<usize>,
    pub trace_span: Option<Span>,
    pub chunk_size_override: Option<usize>,
    pub profile_label: Option<&'static str>,
    pub backend_intent: BackendIntent,
    pub backend_capability_token: BackendCapabilityToken,
    /// F12/STG/YP zeta parameter. When set, populates `operator_env_params.f12_zeta`
    /// on the `ExecutionPlan` for F12-family operators.
    /// Must be non-zero for F12 calls (validated by `validate_f12_env_params` before kernel launch).
    pub f12_zeta: Option<f64>,
}
```

**Change** — add `aosym` AFTER `f12_zeta` (preserves field order; SemVer-additive):
```rust
    pub f12_zeta: Option<f64>,
    /// AO symmetry packing requested by the caller. Phase 18 implements `S1` only;
    /// every other variant returns `FacadeError::UnsupportedAoSymmetry` from
    /// `SessionRequest::query_workspace`. `None` is the default and is treated as
    /// `Some(AoSymmetry::S1)`.
    pub aosym: Option<cintx_core::AoSymmetry>,
}
```

`cintx_core::AoSymmetry` is the fully-qualified path — `cintx-runtime` already depends on `cintx-core` (verified via existing `use cintx_core::*` patterns in the same file). No new dep.

---

### `crates/cintx-rs/src/api.rs` (MODIFY — preflight + F-order rustdoc + `aosym_error_path` test)

**Three distinct edits, each with a distinct analog.**

#### Edit A: aosym preflight in `query_workspace`

**Analog:** `crates/cintx-rs/src/api.rs:63-80` — current `query_workspace` body (verbatim verified).

**Existing pattern (lines 63-80):**
```rust
pub fn query_workspace(&self) -> Result<SessionQuery<'basis>, FacadeError> {
    let runtime_workspace = runtime_query_workspace(
        self.operator,
        self.representation,
        self.basis,
        self.shells.clone(),
        &self.options,
    )
    .map_err(FacadeError::from)?;
    // ... rest of body
}
```

**Change** — insert the preflight BEFORE `runtime_query_workspace` (per CONTEXT.md D-04: fail-fast before any kernel/runtime work):
```rust
pub fn query_workspace(&self) -> Result<SessionQuery<'basis>, FacadeError> {
    // Phase 18 D-04: aosym preflight — only S1 (and None ≡ S1) is implemented.
    // Non-S1 packings return a typed FacadeError::UnsupportedAoSymmetry so callers
    // can pattern-match programmatically. Fails fast before any runtime work.
    if let Some(aosym) = self.options.aosym {
        if aosym != cintx_core::AoSymmetry::S1 {
            return Err(FacadeError::UnsupportedAoSymmetry {
                requested: aosym.to_string(),
            });
        }
    }

    let runtime_workspace = runtime_query_workspace(/* ... existing args ... */)
        .map_err(FacadeError::from)?;
    // ... rest of body unchanged
}
```

**Anti-pattern (RESEARCH Pitfall 6):** Naive `if self.options.aosym.unwrap() != S1` panics on default `None`. Always use `if let Some(aosym) = ...`.

#### Edit B: F-order rustdoc on `IntegralTensor`

**Analog:** `crates/cintx-rs/src/api.rs:441-447` (verbatim verified — the bare struct definition with no doc).

**Existing pattern (lines 441-447):**
```rust
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IntegralTensor {
    pub extents: Vec<usize>,
    pub component_axis_leading: bool,
    pub complex_interleaved: bool,
    pub owned_values: Vec<f64>,
}
```

**Change** — prepend the rustdoc block (per CONTEXT.md D-10 + RESEARCH R2 honest wording — arity-2 is row-major, arity ≥ 3 is F-order):
```rust
/// Owned integral tensor returned by `SessionQuery::evaluate`.
///
/// # Memory layout
///
/// `owned_values` is a dense `Vec<f64>` storing `extents.iter().product()` real
/// values (or 2× that for `Spinor` outputs with `complex_interleaved == true`,
/// where real and imaginary parts alternate in the innermost stride).
///
/// **AO axis layout:** `extents` lists AO-axis sizes in **shell-tuple order**:
/// `extents[0] = ao_per_shell(shells[0])`, `extents[1] = ao_per_shell(shells[1])`,
/// etc. The per-kernel index ordering inside `owned_values` matches libcint's
/// memory layout for that family:
///
/// - **Arity ≥ 3** (`int2e_*`, `int3c1e_*`, `int3c2e_*`, `int4c1e_*`): **F-order**
///   (Fortran / column-major) — `extents[0]` is the fastest-varying axis.
///   Byte-identical to vendor libcint output without transposition.
/// - **Arity 2** (`int1e_*`, `int2c2e_*`): row-major within each shell-pair
///   block — `extents[0]` is the slowest-varying axis. The arity-2 oracle parity
///   tests apply the column-major→row-major conversion to vendor output before
///   comparison (see `crates/cintx-oracle/tests/safe_api_arity2_parity.rs:280-292`).
///
/// When `component_axis_leading == true` (the planner default), an optional
/// component axis (e.g., for IP/derivative operators) is the slowest-varying
/// axis — placed beyond `extents.len()` shell-tuple axes.
///
/// The arity-aware layout is verified implicitly by the oracle parity sweep
/// (`crates/cintx-oracle/tests/safe_api_arity{2,3,4}_parity.rs`). If the layout
/// silently drifts, the first parity test fails.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IntegralTensor {
    pub extents: Vec<usize>,
    pub component_axis_leading: bool,
    pub complex_interleaved: bool,
    pub owned_values: Vec<f64>,
}
```

Per CONTEXT.md "Claude's Discretion": docstring on struct only; module preamble may cross-reference but is not required.

#### Edit C: `aosym_error_path` unit test

**Analog:** `crates/cintx-rs/src/api.rs:488-528` (existing `tests` module — `sample_basis_with_shells` helper at line 511 is reusable) + RESEARCH §Example 2 (the spec for the test body).

**Test body to ADD** (place inside the existing `#[cfg(test)] mod tests` block):
```rust
#[test]
fn aosym_error_path_rejects_non_s1_with_typed_error() {
    use cintx_core::AoSymmetry;
    let (basis, shells) = sample_basis_with_shells(Representation::Cart, &[0, 0]);

    for non_s1 in [AoSymmetry::S2ij, AoSymmetry::S2kl, AoSymmetry::S4, AoSymmetry::S8] {
        let options = ExecutionOptions {
            aosym: Some(non_s1),
            ..Default::default()
        };
        let request = SessionRequest::new(
            OperatorId::new(0),       // int1e_ovlp_cart — any valid op works
            Representation::Cart,
            &basis,
            shells.clone(),
            options,
        );
        let err = request
            .query_workspace()
            .expect_err("non-S1 aosym must return UnsupportedAoSymmetry");
        match err {
            FacadeError::UnsupportedAoSymmetry { requested } => {
                assert_eq!(
                    requested,
                    non_s1.to_string(),
                    "requested field must carry the lowercase pyscf form"
                );
            }
            other => panic!(
                "expected UnsupportedAoSymmetry for aosym={non_s1:?}, got {other:?}"
            ),
        }
    }
}

#[test]
fn aosym_none_and_s1_both_succeed_through_query_workspace() {
    use cintx_core::AoSymmetry;
    let (basis, shells) = sample_basis_with_shells(Representation::Cart, &[0, 0]);

    for aosym in [None, Some(AoSymmetry::S1)] {
        let options = ExecutionOptions { aosym, ..Default::default() };
        let request = SessionRequest::new(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells.clone(),
            options,
        );
        request
            .query_workspace()
            .unwrap_or_else(|e| panic!("aosym={aosym:?} must succeed; got {e:?}"));
    }
}
```

No new imports beyond `use cintx_core::AoSymmetry;` and the existing `super::*` / `crate::error::FacadeError` already in scope. No vendor libcint dependency — these are pure unit tests (CONTEXT.md D-05).

---

### `crates/cintx-rs/src/error.rs` (MODIFY — add `UnsupportedAoSymmetry` variant + kind)

**Analog:** `crates/cintx-rs/src/error.rs:6-34` — the existing `FacadeErrorKind`, `FacadeError`, and `FacadeError::kind()` block (verbatim verified).

**Existing pattern:**
```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FacadeErrorKind {
    UnsupportedApi,
    Layout,
    Memory,
    Validation,
}

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
}

impl FacadeError {
    pub const fn kind(&self) -> FacadeErrorKind {
        match self {
            Self::UnsupportedApi { .. } => FacadeErrorKind::UnsupportedApi,
            Self::Layout { .. } => FacadeErrorKind::Layout,
            Self::Memory { .. } => FacadeErrorKind::Memory,
            Self::Validation { .. } => FacadeErrorKind::Validation,
        }
    }
    /* ... */
}
```

**Change** — add ONE variant to each enum AND one match arm (RESEARCH Pitfall 1: missing the `kind()` arm causes non-exhaustive match compile error):

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FacadeErrorKind {
    UnsupportedApi,
    Layout,
    Memory,
    Validation,
    UnsupportedAoSymmetry,   // ADDED — append at end to keep ordinals stable
}

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
    #[error("unsupported aosym packing: {requested}")]            // ADDED
    UnsupportedAoSymmetry { requested: String },                  // ADDED
}

impl FacadeError {
    pub const fn kind(&self) -> FacadeErrorKind {
        match self {
            Self::UnsupportedApi { .. } => FacadeErrorKind::UnsupportedApi,
            Self::Layout { .. } => FacadeErrorKind::Layout,
            Self::Memory { .. } => FacadeErrorKind::Memory,
            Self::Validation { .. } => FacadeErrorKind::Validation,
            Self::UnsupportedAoSymmetry { .. } => FacadeErrorKind::UnsupportedAoSymmetry,  // ADDED
        }
    }
    /* ... existing unsupported_representation unchanged ... */
}
```

`From<cintxRsError>` impl (lines 45-108) does NOT need a new arm — `UnsupportedAoSymmetry` is raised exclusively from `SessionRequest::query_workspace` after the new preflight, not from anywhere in `cintxRsError`.

---

### `crates/cintx-rs/src/prelude.rs` (MODIFY — re-export `AoSymmetry`)

**Analog:** `crates/cintx-rs/src/prelude.rs:31-34` (verbatim verified):
```rust
pub use cintx_core::BasisSet;
pub use cintx_core::OperatorId;
pub use cintx_core::Representation;
pub use cintx_core::ShellTuple;
```

**Change** — add ONE re-export line (alphabetical order: `AoSymmetry` goes BEFORE `BasisSet`):
```rust
pub use cintx_core::AoSymmetry;     // ADDED
pub use cintx_core::BasisSet;
pub use cintx_core::OperatorId;
pub use cintx_core::Representation;
pub use cintx_core::ShellTuple;
```

---

### `crates/cintx-rs/src/builder.rs` (MODIFY, OPTIONAL — `.aosym()` setter)

**Analog:** `crates/cintx-rs/src/builder.rs:87-94` — the existing `f12_zeta` setter (verbatim verified):
```rust
/// Set the F12/STG/YP zeta parameter.
///
/// When set, `operator_env_params.f12_zeta` is populated on the `ExecutionPlan`
/// for F12-family operators. Must be non-zero for F12 calls.
pub fn f12_zeta(mut self, zeta: f64) -> Self {
    self.options.f12_zeta = Some(zeta);
    self
}
```

**Change (optional)** — add a sibling setter:
```rust
/// Set the AO symmetry packing requested by the caller.
///
/// Phase 18 implements `AoSymmetry::S1` only; every other variant returns
/// `FacadeError::UnsupportedAoSymmetry` from `SessionRequest::query_workspace`.
pub fn aosym(mut self, aosym: cintx_core::AoSymmetry) -> Self {
    self.options.aosym = Some(aosym);
    self
}
```

Per CONTEXT.md "Claude's Discretion" + RESEARCH §"Recommendations" item 11 — recommended for ergonomic parity with `f12_zeta`, but skippable if the planner wants minimal surface.

---

## Shared Patterns

### Module-level test cfg gate
**Source:** `crates/cintx-oracle/tests/safe_api_arity2_parity.rs:13`
**Apply to:** Both new arity-3 / arity-4 oracle test files
```rust
#![cfg(any(feature = "cpu", feature = "rocm"))]
```

### Per-symbol parity test cfg stack
**Source:** `crates/cintx-oracle/tests/oracle_gate_closure.rs:737-739`
**Apply to:** Each `int4c1e_*` `#[test]` function in `safe_api_arity4_parity.rs`
```rust
#[test]
#[cfg(feature = "with-4c1e")]
#[cfg(has_vendor_libcint)]
fn test_int4c1e_sph_safe_api_parity() { /* ... */ }
```

For the 10 non-4c1e tests (8 arity-3 + 2 arity-4 `int2e_*`), use the same pattern minus `#[cfg(feature = "with-4c1e")]`:
```rust
#[test]
#[cfg(has_vendor_libcint)]
fn test_int3c1e_sph_safe_api_parity() { /* ... */ }
```

### Tolerance constants (Phase 15 unified)
**Source:** `crates/cintx-oracle/tests/safe_api_arity2_parity.rs:596-597`
**Apply to:** Both new test files (file-top constants per RESEARCH §"Pattern 6")
```rust
const ATOL: f64 = 1e-12;
const RTOL: f64 = 0.0;
```

### Direct cintx-vs-vendor compare (no transpose) for arity ≥ 3
**Source:** `crates/cintx-oracle/src/compare.rs:787-797` (`int2e_sph`), `:811-821` (`int3c1e_sph`), `:823-833` (`int3c2e_ip1_sph`); test-side precedent: `crates/cintx-oracle/tests/center_3c2e_parity.rs:243-277`, `crates/cintx-oracle/tests/two_electron_parity.rs:273-289`
**Apply to:** Every `#[test]` in both new test files
```rust
// Arity ≥ 3: NO transpose. cintx and vendor both write F-order, agreeing byte-for-byte.
total_mismatches += count_mismatches(&vendor_out, &safe_out, ATOL, RTOL);
```

### Per-symbol nonzero sentinel
**Source:** `crates/cintx-oracle/tests/center_3c2e_parity.rs:271-275, 282`
**Apply to:** Every `#[test]` in both new test files
```rust
if safe_out.iter().any(|&v| v.abs() > 1e-18)
    || vendor_out.iter().any(|&v| v.abs() > 1e-18) {
    any_nonzero = true;
}
// ... after the sweep:
assert!(any_nonzero, "<symbol> outputs are all zeros over {tuples_checked} tuples");
```

### Error variant kind() exhaustiveness
**Source:** `crates/cintx-rs/src/error.rs:26-34`
**Apply to:** `FacadeError::UnsupportedAoSymmetry` addition
- New variant in `FacadeError`.
- Corresponding new variant in `FacadeErrorKind`.
- New arm in `FacadeError::kind()` match.
- All three edits made in the same commit to avoid compile breakage.

### Re-export alphabetical ordering
**Source:** `crates/cintx-core/src/lib.rs:18` (`OperatorId, Representation`); `crates/cintx-rs/src/prelude.rs:31-34`
**Apply to:** `AoSymmetry` re-exports in both crates — alphabetical order keeps the diff minimal.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| (none) | — | — | All 14 Phase 18 files have direct or near-exact analogs in the codebase. Adding `AoSymmetry` is the closest to "new from scratch" but mirrors `Representation` line-for-line. Adding manifest rows for plain `int3c2e_{cart,sph}` mirrors `int3c1e_{cart,sph}` exactly in the lock-file JSON shape; only the resolver test-arm change in `misc_wrapper_macro` lacks a precise prior precedent (closest is the existing match arms at resolver.rs:314-322 — a small additive edit). |

---

## Pre-Implementation Verification Greps (executor MUST run)

```bash
# 1. Verify the 2 missing vendor wrappers (R4) and 1 R1 wrapper are absent before adding:
grep -n "vendor_int3c1e_p2_sph\|vendor_int3c2e_ip1_sph\|vendor_int3c2e_cart\b" \
    /home/user/Documents/workspace/cintx/crates/cintx-oracle/src/vendor_ffi.rs
# Expected: zero matches for vendor_int3c1e_p2_sph and vendor_int3c2e_ip1_sph
# (vendor_int3c2e_sph at line 198 should NOT match the `\b` boundary for vendor_int3c2e_cart)

# 2. Verify FFI binding availability for the new wrappers (RESEARCH A3):
grep -n "int3c1e_p2_sph\|int3c2e_ip1_sph\|int3c2e_cart" \
    /home/user/Documents/workspace/cintx/crates/cintx-oracle/build.rs

# 3. Find all hard-coded post-R1 OperatorId references to update (Step 2 of Manifest-Row Additions):
grep -rn "OperatorId::new(22)\|OperatorId::new(23)\|INT4C1E_CART_OPERATOR_ID\|INT4C1E_SPH_OPERATOR_ID" \
    /home/user/Documents/workspace/cintx/crates/ \
    /home/user/Documents/workspace/cintx/xtask/

# 4. Verify the resolver test currently expects misc_wrapper_macro to cover every base symbol:
grep -n "missing misc.h wrapper macro classification" \
    /home/user/Documents/workspace/cintx/crates/cintx-ops/src/resolver.rs

# 5. Confirm cintx-oracle/Cargo.toml already has cintx-rs + cintx-runtime (Phase 17 added these — Pitfall 5):
grep -n "cintx-rs\|cintx-runtime" \
    /home/user/Documents/workspace/cintx/crates/cintx-oracle/Cargo.toml

# 6. Confirm sample_basis_with_shells exists (used by aosym_error_path tests):
grep -n "fn sample_basis_with_shells" \
    /home/user/Documents/workspace/cintx/crates/cintx-rs/src/api.rs
```

---

## Metadata

**Analog search scope:**
- `crates/cintx-rs/src/` (api.rs, error.rs, prelude.rs, builder.rs)
- `crates/cintx-core/src/` (operator.rs, lib.rs)
- `crates/cintx-runtime/src/` (options.rs)
- `crates/cintx-ops/src/` (resolver.rs, generated/api_manifest.rs, generated/api_manifest.csv)
- `crates/cintx-ops/generated/` (compiled_manifest.lock.json)
- `crates/cintx-ops/build.rs`
- `crates/cintx-cubecl/src/kernels/` (mod.rs, center_3c2e.rs — routing verification)
- `crates/cintx-oracle/src/` (vendor_ffi.rs, compare.rs, fixtures.rs)
- `crates/cintx-oracle/tests/` (safe_api_arity2_parity.rs, one_electron_parity.rs, center_3c2e_parity.rs, two_electron_parity.rs, oracle_gate_closure.rs)
- `xtask/src/` (manifest_audit.rs, oracle_covered_update.rs)
- `.planning/phases/17-real-integral-evaluation-in-safe-api/17-PATTERNS.md` (Phase 17 predecessor pattern map)

**Files scanned:** ~30
**Strong-match analogs found:** 14 / 14 (all 14 Phase 18 files have at least one role-match analog; 13 have exact-match)
**Pattern extraction date:** 2026-05-12
