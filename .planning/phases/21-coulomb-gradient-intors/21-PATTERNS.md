# Phase 21: Plain-Coulomb Gradient Integral Families (`ip1`/`iprinv`) - Pattern Map

**Mapped:** 2026-05-26
**Files analyzed:** 9 new/modified files
**Analogs found:** 9 / 9

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `crates/cintx-runtime/src/planner.rs` | env-param plumbing | request-response | self (add `rinv_orig` to `OperatorEnvParams`) | exact – same 4-step pattern as `f12_zeta`/`grids_params` |
| `crates/cintx-runtime/src/validator.rs` | validation gate | request-response | self (add `validate_rinv_orig_env_params`) | exact – mirrors `validate_f12_env_params` |
| `crates/cintx-compat/src/raw.rs` | raw-compat surface | request-response | self (add `PTR_RINV_ORIG` const + `eval_raw` read block) | exact – mirrors `PTR_F12_ZETA` read block |
| `crates/cintx-ops/generated/compiled_manifest.lock.json` | manifest registration | config | `int1e_ecp_ipnuc_cart/sph` entries (carry `"component_rank":"3"`) | exact |
| `crates/cintx-cubecl/src/kernels/one_electron.rs` | kernel math (1e grads) | CRUD | self dispatcher at line 486 + `contract_kinetic` nabla at line 208 | exact |
| `crates/cintx-cubecl/src/kernels/two_electron.rs` | kernel math (2e grad) | CRUD | `launch_two_electron_typed` + `build_2e_shape` + `fill_g_tensor_2e` | exact |
| `crates/cintx-cubecl/src/kernels/center_3c2e.rs` | kernel math (3c2e repair) | CRUD | `launch_center_3c2e_typed` (the operator-blind stub to replace) | exact |
| `crates/cintx-cubecl/src/kernels/ecp.rs` | kernel math (ECP grad) | CRUD | `launch_ecp` / `deriv1_cart_pair` (`ipnuc` driver) | exact |
| `crates/cintx-compat/src/legacy.rs` | legacy wrapper registration | config | `all_cint_wrappers!(cint3c2e_ip1_cart, …)` block at line 227 + `LEGACY_WRAPPER_SYMBOLS` at line 239 | exact |
| `crates/cintx-capi/src/shim.rs` | C-ABI shim | request-response | existing `CintxRawApi` `#[repr(i32)]` enum + `from_i32` + `raw_id` at lines 9–116 | exact |
| `crates/cintx-oracle/src/vendor_ffi.rs` | oracle FFI | request-response | `vendor_int1e_ovlp_sph` / `vendor_int3c2e_ip1_cart` patterns | exact |
| `crates/cintx-oracle/tests/*_parity.rs` (6 new files) | oracle tests | CRUD | `safe_api_ecp_parity.rs` test structure (gradient, multi-component) | exact |

---

## Pattern Assignments

### 21-01: `crates/cintx-runtime/src/planner.rs` — add `rinv_orig` to `OperatorEnvParams`

**Analog:** self, lines 27–50 (`GridsEnvParams` / `OperatorEnvParams`)

**Step 1 — new field on `OperatorEnvParams`** (analog: `f12_zeta: Option<f64>` at line 46):
```rust
/// PTR_RINV_ORIG value (env[4..6]) for iprinv/ECPscalar_iprinv kernels.
/// Must be present when operator_name contains "iprinv".
/// Absent (None) is allowed for all other operators.
pub rinv_orig: Option<[f64; 3]>,
```

The full struct after the addition mirrors lines 42–50 of `planner.rs`:
```rust
#[derive(Clone, Debug, Default)]
pub struct OperatorEnvParams {
    pub f12_zeta: Option<f64>,
    pub grids_params: Option<GridsEnvParams>,
    pub rinv_orig: Option<[f64; 3]>,   // ← new
}
```

**Step 2 — safe-API setter** (by analogy with `ExecutionOptions`; no existing exact line, but follows the same optional-field convention):
```rust
/// Set the rinv origin for iprinv operators (env[4..6] in the raw API).
pub fn with_rinv_origin(mut self, origin: [f64; 3]) -> Self {
    self.operator_env_params.rinv_orig = Some(origin);
    self
}
```

---

### 21-01: `crates/cintx-runtime/src/validator.rs` — add `validate_rinv_orig_env_params`

**Analog:** `validate_f12_env_params` (lines 147–171) — exact template:

```rust
/// Returns `InvalidEnvParam` if `rinv_orig` is `None` for an iprinv-family operator.
/// Called before kernel launch so we surface a typed error before kernel entry.
pub fn validate_rinv_orig_env_params(
    operator_name: &str,
    params: &OperatorEnvParams,
) -> Result<(), cintxRsError> {
    if operator_name.contains("iprinv") {
        match params.rinv_orig {
            None => {
                return Err(cintxRsError::InvalidEnvParam {
                    param: "PTR_RINV_ORIG",
                    reason: "env[4..6] (PTR_RINV_ORIG) must be set for iprinv operators"
                        .to_owned(),
                });
            }
            _ => {}
        }
    }
    Ok(())
}
```

**Unit test pattern** (analog: validator.rs lines 281–310):
```rust
#[test]
fn validate_rinv_orig_rejects_none_for_iprinv() {
    let params = OperatorEnvParams::default(); // rinv_orig: None
    let err = validate_rinv_orig_env_params("iprinv", &params).unwrap_err();
    assert!(matches!(err, cintxRsError::InvalidEnvParam { param, .. } if param == "PTR_RINV_ORIG"));
}

#[test]
fn validate_rinv_orig_accepts_non_iprinv() {
    let params = OperatorEnvParams::default();
    validate_rinv_orig_env_params("overlap", &params).expect("non-iprinv must not be gated");
}
```

---

### 21-01: `crates/cintx-compat/src/raw.rs` — add `PTR_RINV_ORIG` const + `eval_raw` read block

**Step 1 — constant** (analog: `PTR_F12_ZETA` at line 48):
```rust
/// Index range in the libcint env array for the rinv origin (x, y, z).
///
/// libcint defines `PTR_RINV_ORIG = 4` (three consecutive slots 4, 5, 6).
/// Raw callers set env[4..7] = [x, y, z] before calling iprinv integrals.
pub const PTR_RINV_ORIG: usize = 4;
```

**Step 2 — `RawApiId` constants** (analog: `INT1E_ECP_IPNUC_CART/SPH` at lines 150–151):
```rust
pub const INT1E_IPOVLP_CART: Self = Self::Symbol("int1e_ipovlp_cart");
pub const INT1E_IPOVLP_SPH:  Self = Self::Symbol("int1e_ipovlp_sph");
pub const INT1E_IPOVLP_SPINOR: Self = Self::Symbol("int1e_ipovlp_spinor");

pub const INT1E_IPKIN_CART: Self = Self::Symbol("int1e_ipkin_cart");
pub const INT1E_IPKIN_SPH:  Self = Self::Symbol("int1e_ipkin_sph");
pub const INT1E_IPKIN_SPINOR: Self = Self::Symbol("int1e_ipkin_spinor");

pub const INT1E_IPNUC_CART: Self = Self::Symbol("int1e_ipnuc_cart");
pub const INT1E_IPNUC_SPH:  Self = Self::Symbol("int1e_ipnuc_sph");
pub const INT1E_IPNUC_SPINOR: Self = Self::Symbol("int1e_ipnuc_spinor");

pub const INT1E_IPRINV_CART: Self = Self::Symbol("int1e_iprinv_cart");
pub const INT1E_IPRINV_SPH:  Self = Self::Symbol("int1e_iprinv_sph");
pub const INT1E_IPRINV_SPINOR: Self = Self::Symbol("int1e_iprinv_spinor");

pub const INT2E_IP1_CART: Self = Self::Symbol("int2e_ip1_cart");
pub const INT2E_IP1_SPH:  Self = Self::Symbol("int2e_ip1_sph");
pub const INT2E_IP1_SPINOR: Self = Self::Symbol("int2e_ip1_spinor");

pub const INT1E_ECP_IPRINV_CART: Self = Self::Symbol("int1e_ecp_iprinv_cart");
pub const INT1E_ECP_IPRINV_SPH:  Self = Self::Symbol("int1e_ecp_iprinv_sph");
```

**Step 3 — `eval_raw` read block** (analog: f12_zeta block at lines 555–563):
```rust
// Extract rinv_orig from env[PTR_RINV_ORIG..PTR_RINV_ORIG+3] for iprinv operators.
// Raw callers must set env[4..7] = [x, y, z] before calling any iprinv integral.
if is_iprinv_family_symbol(plan.descriptor.operator_symbol()) {
    if env.len() >= PTR_RINV_ORIG + 3 {
        let x = env[PTR_RINV_ORIG];
        let y = env[PTR_RINV_ORIG + 1];
        let z = env[PTR_RINV_ORIG + 2];
        plan.operator_env_params.rinv_orig = Some([x, y, z]);
    }
    cintx_runtime::validator::validate_rinv_orig_env_params(
        plan.descriptor.operator_name(),
        &plan.operator_env_params,
    )?;
}
```

**Helper predicate** (analog: `is_f12_family_symbol` which the existing code already uses):
```rust
fn is_iprinv_family_symbol(symbol: &str) -> bool {
    symbol.contains("iprinv")
}
```

---

### 21-02: `crates/cintx-ops/generated/compiled_manifest.lock.json` — manifest registration

**Tell for a gradient family** — `"component_rank":"3"` present.
**Proof of the `int3c2e_ip1` bug** — current entries at lines 308–354 carry `"operator":"electron-repulsion"` and NO `"component_rank"` field at all (it is absent from the top-level entry object), whereas `int1e_ecp_ipnuc_cart` (lines 507–538) has `"component_rank":"3"` and `"operator":"ecp_ipnuc"`.

**Template for a new 1e gradient entry** — copy `int1e_ecp_ipnuc_cart` (lines 507–538), replacing family/operator/symbol/arity:
```json
{
  "arity": 2,
  "canonical_family": "1e",
  "category": "1e",
  "compiled_in_profiles": [
    "base",
    "with-f12",
    "with-4c1e",
    "with-f12+with-4c1e"
  ],
  "component_rank": "3",
  "declared_in": "unknown",
  "feature_flag": "none",
  "forms": ["cart"],
  "helper_kind": "operator",
  "id": {
    "family": "1e",
    "operator": "ipovlp",
    "representation": "cart",
    "symbol": "int1e_ipovlp_cart"
  },
  "oracle_covered": true,
  "profiles": ["base", "with-f12", "with-4c1e", "with-f12+with-4c1e"],
  "stability": "stable"
}
```
Repeat for `sph` and `spinor` representations; adjust `operator` and `symbol` for each of the 6 families.

**Template for the `int3c2e_ip1` correction** — the three existing entries (lines 307–354) need `"component_rank":"3"` added and `"operator"` changed from `"electron-repulsion"` to `"ip1"` (matching the libcint symbol). Also add the top-level `"arity"`, `"canonical_family"`, `"category"`, `"component_rank":"3"`, `"declared_in"`, `"feature_flag"`, `"forms"`, `"helper_kind"` fields that are currently absent from those entries. Use the `int1e_ecp_ipnuc` JSON shape as the template but with `"arity":3`, `"canonical_family":"3c2e"`, `"category":"3c2e"`.

**`cint*` legacy symbols** — also add helper-kind `"optimizer"` and `"legacy"` sibling entries for each new operator symbol, following the existing `int3c2e_ip1_cart_optimizer` / `cint3c2e_ip1_cart` entries that are already in the file but currently wired to the wrong (scalar) operator.

---

### 21-03 / 21-04: `crates/cintx-cubecl/src/kernels/one_electron.rs` — 1e gradient branches

**Analog (dispatcher):** lines 485–495:
```rust
let op_name = plan.descriptor.operator_name();
let is_overlap  = op_name == "overlap";
let is_kinetic  = op_name == "kinetic";
let is_nuclear  = op_name == "nuclear-attraction";

if !is_overlap && !is_kinetic && !is_nuclear {
    return Err(cintxRsError::UnsupportedApi {
        requested: format!("1e operator '{}' is not supported", op_name),
    });
}
```

**New dispatcher extension** — add gradient branches and thread `rinv_orig`:
```rust
let is_ipovlp  = op_name == "ipovlp";
let is_ipkin   = op_name == "ipkin";
let is_ipnuc   = op_name == "ipnuc";
let is_iprinv  = op_name == "iprinv";

if !is_overlap && !is_kinetic && !is_nuclear
    && !is_ipovlp && !is_ipkin && !is_ipnuc && !is_iprinv
{
    return Err(cintxRsError::UnsupportedApi {
        requested: format!("1e operator '{}' is not supported", op_name),
    });
}
```

**Analog (nabla pattern for kinetic):** `contract_kinetic` at line 208:
```rust
fn contract_kinetic(g: &[f64], li: u8, lj: u8, nmax: u32, aj: f64) -> Vec<f64> {
    // ...
    // CINTnabla1j_1e: derivative on j channel (ket).
    // g was built with lj+2 HRR j-levels (nmax = li + lj + 2).
}
```

For `ipovlp` / `ipkin` / `ipnuc` / `iprinv` the nabla is `∂/∂Ai` (bra derivative), so the `nabla1i_2e` function from `f12.rs:590` is the direct pattern. The 1e G-tensor does not use the 4-index `F12Shape` directly — adapt by passing a 1e-equivalent shape struct or inline the same loop logic for a 2-index G-tensor, following the same `di`, `dj`, `axis` triple-axis loop structure.

**`ipnuc` atom-loop / `iprinv` single-origin split** — implement after the nabla contraction as a per-primitive kernel parameter:
- `ipnuc`: outer loop `for atom_c in 0..natm`, accumulate `∑_C (-Z_C) * gout_ip1(g_C)` where `g_C` uses `rc = atoms[atom_c].coord_bohr`.
- `iprinv`: single origin from `plan.operator_env_params.rinv_orig.unwrap()`, no `Z_C` charge factor (factor = `1.0`).

---

### 21-05: `crates/cintx-cubecl/src/kernels/two_electron.rs` — `int2e_ip1` gradient path

**Analog (launcher skeleton):** `launch_two_electron_typed` at lines 563–624:
```rust
fn launch_two_electron_typed<F: CintFloat>(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [F],
) -> Result<ExecutionStats, cintxRsError> {
    // family guard ...
    let _ = backend;
    let shells = plan.shells.as_slice();
    // shell/li/lj/lk/ll extraction ...
    let shape = build_2e_shape(li as usize, lj as usize, lk as usize, ll as usize);
    // ...
}
```

**New gradient path** — check `plan.descriptor.operator_name() == "ip1"` inside `launch_two_electron_typed`, then:

```rust
// Gradient path: nroots from li+1, not li (D-06: build_2e_shape(li+1, lj, lk, ll))
let grad_shape = build_2e_shape(li as usize + 1, lj as usize, lk as usize, ll as usize);
if grad_shape.nroots > 5 {
    return Err(cintxRsError::UnsupportedApi {
        requested: format!("unsupported_nrys_roots:{}", grad_shape.nroots),
    });
}
```

Then fill the G-tensor via `fill_g_tensor_2e` (line 351 signature) with `grad_shape`, call `gout_ip1` from `f12.rs:727` verbatim (imported from the same crate), then write 3-component leading output: `staging[comp * block_len + n] = value`.

**F12Shape / TwoEShape bridge** — `gout_ip1` takes a `&F12Shape`. Construct a `F12Shape` from the `TwoEShape` fields:
```rust
// F12Shape is defined in f12.rs with the same strides.
// TwoEShape fields di/dj/dk/dl/nroots/g_size map 1:1 to F12Shape.
let f12_shape = F12Shape {
    nroots: grad_shape.nroots,
    di: grad_shape.di,
    dj: grad_shape.dj,
    dk: grad_shape.dk,
    dl: grad_shape.dl,
    g_size: grad_shape.g_size,
};
```
(If `F12Shape` is not pub-exported from `f12.rs`, either re-export it or inline the `gout_ip1` logic with the `TwoEShape` strides directly — the computation is identical.)

---

### 21-06: `crates/cintx-cubecl/src/kernels/center_3c2e.rs` — `int3c2e_ip1` real kernel

**Analog (the stub to replace):** `launch_center_3c2e_typed` at lines 315–480. The repair inserts an operator-symbol branch at the top:

```rust
let op_name = plan.descriptor.operator_name();
let is_ip1  = op_name == "ip1";
// existing scalar path continues for op_name == "electron-repulsion"
```

For the `ip1` branch, the pattern is identical to the scalar path but:
1. Use `build_2e_shape(li as usize + 1, lj as usize, 0, lk as usize)` (the 3c2e kl-mapping: phantom `lk_ceil=0`, real k mapped to `ll` slot — see file header comment lines 6–12).
2. Call `gout_ip1` after `fill_g_tensor_3c2e` instead of `contract_3c2e`.
3. Output shape `[3 * nci * ncj * nck]` (component-leading, same F-order as `int2e_ip1`).

The 3c2e pitfall mapping (lines 6–12 of the file) is the key comment to preserve:
```
// 2e "ij side"  <- real (i, j)
// 2e "kl side"  <- real k mapped into the 2e `ll` slot
// 2e `lk` slot is a phantom s-function (lk_ceil = 0, ak = 0)
```

---

### 21-07: `crates/cintx-cubecl/src/kernels/ecp.rs` — `ECPscalar_iprinv`

**Analog (dispatcher):** `launch_ecp` lines 1369–1378:
```rust
let operator_name = plan.descriptor.operator_name();
let is_gradient = match operator_name {
    "ecp"        => false,
    "ecp_ipnuc"  => true,
    other => {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("unknown ecp operator name: {other}"),
        });
    }
};
```

**Extension** — add `"ecp_iprinv"` arm:
```rust
let is_gradient = match operator_name {
    "ecp"        => false,
    "ecp_ipnuc"  => true,
    "ecp_iprinv" => true,   // ← new
    other => { /* UnsupportedApi */ }
};
// Track which gradient variant:
let is_iprinv = operator_name == "ecp_iprinv";
```

**`deriv1_cart_pair` analog (lines 1181–1292):** The new `iprinv` path is a **single-origin** version. The `ipnuc` driver (alias at line 1295–1312) loops over all ECP shells for all atoms, accumulating all contributions. For `iprinv`, select only the ECP shell(s) for `atom_c == rinv_atom_index` where `rinv_atom_index` is found by matching `plan.operator_env_params.rinv_orig` coordinates against `atoms[c].coord_bohr`.

The `lc` parameter in `deriv1_cart_pair` selects the ECP channel (Type-1: `lc = -1`; Type-2: `lc = l`). This logic is unchanged; the change is only which atom's ECP contribution is included.

**`Y_ADDR`/`Z_ADDR`/`CART_POW_*` tables** — the salvaged Phase 19 tables have a `[usize;135]` sizing bug to fix to `[usize;120]` on reuse. These tables are already present in `ecp.rs`; do not copy them again — just confirm the corrected size before calling `deriv1_cart_pair` from the new `iprinv` branch.

---

### 21-02 (legacy): `crates/cintx-compat/src/legacy.rs` — legacy wrapper registration

**Analog:** `all_cint_wrappers!(cint3c2e_ip1_cart, …)` block at lines 227–237 and `LEGACY_WRAPPER_SYMBOLS` at lines 239–285.

**New wrapper blocks** (one per new family, follow the same macro invocation pattern):
```rust
all_cint_wrappers!(
    cint1e_ipovlp_cart,
    cint1e_ipovlp_sph,
    cint1e_ipovlp,
    cint1e_ipovlp_cart_optimizer,
    cint1e_ipovlp_sph_optimizer,
    cint1e_ipovlp_optimizer,
    RawApiId::INT1E_IPOVLP_CART,
    RawApiId::INT1E_IPOVLP_SPH,
    RawApiId::INT1E_IPOVLP_SPINOR
);
// Repeat for ipkin, ipnuc, iprinv, int2e_ip1, ECPscalar_iprinv
```

**`LEGACY_WRAPPER_SYMBOLS` additions** — append the 6 symbol triples (cart/sph/base + 3 optimizer variants each) for all new families.

**`misc_wrapper_macro` test guard** (line 312) — extend the match arm:
```rust
fn misc_wrapper_macro(base_symbol: &str) -> Option<MiscWrapperMacro> {
    match base_symbol {
        "int1e_ovlp" | "int1e_nuc" | "int2e" | "int2c2e"
        | "int3c1e" | "int3c1e_p2" | "int3c2e_ip1"
        | "int1e_ipovlp" | "int1e_ipkin" | "int1e_ipnuc" | "int1e_iprinv"
        | "int2e_ip1" | "int1e_ecp_iprinv"                    // ← new
        => Some(MiscWrapperMacro::AllCint),
        "int1e_kin" => Some(MiscWrapperMacro::AllCint1e),
        _ => None,
    }
}
```

---

### 21-02 (C-ABI): `crates/cintx-capi/src/shim.rs` — `CintxRawApi` variants

**Analog:** existing enum variants at lines 9–33, `from_i32` at lines 62–88, `raw_id` at lines 91–116.

**New variants** — append after `Int4c1eSph = 22` (next discriminants start at 23):
```rust
Int1eIpovlpCart   = 23,
Int1eIpovlpSph    = 24,
Int1eIpovlpSpinor = 25,
Int1eIpkinCart    = 26,
Int1eIpkinSph     = 27,
Int1eIpkinSpinor  = 28,
Int1eIpnucCart    = 29,
Int1eIpnucSph     = 30,
Int1eIpnucSpinor  = 31,
Int1eIprinvCart   = 32,
Int1eIprinvSph    = 33,
Int1eIprinvSpinor = 34,
Int2eIp1Cart      = 35,
Int2eIp1Sph       = 36,
Int2eIp1Spinor    = 37,
Int1eEcpIprinvCart = 38,
Int1eEcpIprinvSph  = 39,
```

**`from_i32` extension** (analog: lines 62–88 — one arm per variant):
```rust
23 => Some(Self::Int1eIpovlpCart),
24 => Some(Self::Int1eIpovlpSph),
// ...
```

**`raw_id` extension** (analog: lines 91–116):
```rust
Self::Int1eIpovlpCart   => RawApiId::INT1E_IPOVLP_CART,
Self::Int1eIpovlpSph    => RawApiId::INT1E_IPOVLP_SPH,
// ...
```

---

### Oracle FFI: `crates/cintx-oracle/src/vendor_ffi.rs` — new `vendor_*` wrappers

**Analog:** `vendor_int1e_ovlp_sph` (lines 21–44) for 1e families; `vendor_int3c2e_ip1_cart` (lines 552–574) for the 3c2e ip1 repair.

**Template for 1e gradient wrappers** (2-element `shls`):
```rust
/// Evaluate int1e_ipovlp_sph for a single shell pair using vendored libcint.
/// `out` must be pre-allocated with 3 * ni * nj elements (3 components).
pub fn vendor_int1e_ipovlp_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipovlp_sph(
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
Repeat for `cart` variant and for `ipkin`, `ipnuc`, `iprinv` families; all share the same `[i32; 2]` `shls` signature.

**Template for `int2e_ip1`** (4-element `shls`, analog: `vendor_int2e_sph` at lines 111–134):
```rust
pub fn vendor_int2e_ip1_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    // same signature as vendor_int2e_sph
) -> i32 { /* ffi::int2e_ip1_sph(...) */ }
```

**Note on `iprinv` env setup** — the test caller must populate `env[PTR_RINV_ORIG..PTR_RINV_ORIG+3]` before calling `vendor_int1e_iprinv_sph`, matching the raw-compat path. Document this in the wrapper doc comment.

---

### Oracle tests: `crates/cintx-oracle/tests/*_parity.rs` (6 new files)

**Analog:** `safe_api_ecp_parity.rs` (gradient family, `component_rank:3`, Cu/LANL2DZ) for ECP tests; `one_electron_parity.rs` (scalar, H2O STO-3G) for the non-ECP 1e tests.

**Test file header template** (follow `one_electron_parity.rs` lines 1–41):
```rust
//! Oracle parity tests for int1e_ipovlp_{cart,sph}: H2O STO-3G.
//!
//! Validates 3-component gradient output vs vendored libcint 6.1.3 at atol=1e-12.

#![cfg(any(feature = "cpu", feature = "rocm"))]

const ATOL: f64 = 1e-12;
const RTOL: f64 = 0.0;
```

**Gradient-matrix collector** — the existing `collect_1e_sph_matrix` helper (one_electron_parity.rs line 214) returns a scalar `n_ao × n_ao` matrix. For gradient tests, the output per shell pair is `3 × ni × nj` (3 components). Adapt to collect a `3 × n_ao × n_ao` matrix:
```rust
fn collect_1e_grad_sph_matrix(
    api_id: RawApiId,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> Vec<f64> {
    // out per (si, sj): 3 * ni * nj elements, component-leading
    // aggregate into [3 * n_ao * n_ao]
}
```

**Byte-identity assertion** (analog: `count_mismatches` at line 271 + assertion at line 307):
```rust
#[test]
#[cfg(has_vendor_libcint)]
fn test_int1e_ipovlp_sph_h2o_sto3g_parity() {
    let (atm, bas, env) = build_h2o_sto3g();
    let vendor = collect_vendor_1e_grad_sph_matrix(
        vendor_ffi::vendor_int1e_ipovlp_sph, &atm, &bas, &env
    );
    let cintx = collect_1e_grad_sph_matrix(RawApiId::INT1E_IPOVLP_SPH, &atm, &bas, &env);
    let mismatches = count_mismatches(&vendor, &cintx, ATOL, RTOL);
    assert_eq!(mismatches, 0, "…");
}
```

**ECP `iprinv` tests** — use the `build_cu_lanl2dz` fixture (already in oracle fixtures) and follow `safe_api_ecp_parity.rs` lines 550–588 exactly, substituting `iprinv` for `ipnuc` and setting `env[PTR_RINV_ORIG..PTR_RINV_ORIG+3]` to the coordinates of each nucleus in turn.

---

## Shared Patterns

### Gradient mathematics — `gout_ip1` + `nabla1i/j/k_2e`
**Source:** `crates/cintx-cubecl/src/kernels/f12.rs:590–785`
**Apply to:** `one_electron.rs` (1e grads), `two_electron.rs` (`int2e_ip1`), `center_3c2e.rs` (`int3c2e_ip1` repair), `ecp.rs` (`ECPscalar_iprinv` via `deriv1_cart_pair`)

Key signatures (reuse verbatim — zero F12 logic):
```rust
fn nabla1i_2e(f: &mut [f64], g: &[f64], li: usize, lj: usize, lk: usize, ll: usize, ai: f64, shape: &F12Shape)
fn nabla1j_2e(f: &mut [f64], g: &[f64], li: usize, lj: usize, lk: usize, ll: usize, aj: f64, shape: &F12Shape)
fn nabla1k_2e(f: &mut [f64], g: &[f64], li: usize, lj: usize, lk: usize, ll: usize, ak: f64, shape: &F12Shape)
fn gout_ip1(g: &[f64], shape: &F12Shape, li: usize, lj: usize, lk: usize, ll: usize, ai: f64) -> Vec<f64>
```

`gout_ip1` allocates and returns a `Vec<f64>` of length `3 * nfi * nfj * nfk * nfl` with interleaved layout `out[n*3+comp]`. For the `int2e_ip1` component-leading F-order contract (D-06), transpose after the `gout_ip1` call: `staging[comp * block_len + n] = gout[n * 3 + comp]`.

### `PTR_RINV_ORIG` env-slot 4-step template
**Source:** `f12_zeta` in `planner.rs:44`, `validator.rs:147–171`, `raw.rs:555–563`
**Apply to:** `planner.rs` (add field), `validator.rs` (add validation fn), `raw.rs` (add const + read block)

The 4 steps are:
1. Add `rinv_orig: Option<[f64;3]>` to `OperatorEnvParams` (planner.rs)
2. Add `validate_rinv_orig_env_params` (validator.rs) — reject `None` for `iprinv` operators
3. Add `PTR_RINV_ORIG: usize = 4` const and read block in `eval_raw` (raw.rs)
4. Thread `plan.operator_env_params.rinv_orig` into the kernel (one_electron.rs, ecp.rs)

### `component_rank:"3"` manifest tell
**Source:** `crates/cintx-ops/generated/compiled_manifest.lock.json` lines 507–538 (`int1e_ecp_ipnuc_cart`)
**Apply to:** all 6 new families + `int3c2e_ip1` correction entries

The runtime planner at `cintx-runtime/src/planner.rs:395,432` reads `component_rank` to auto-allocate `3 × ni × nj[× nk × nl]` staging. A missing or empty `"component_rank"` field means the planner never allocates 3-component staging — this is the root cause of the `int3c2e_ip1` stub bug.

### Family guard + operator-symbol dispatch
**Source:** `launch_ecp` lines 1357–1378; `launch_center_3c2e_typed` lines 321–329
**Apply to:** all new kernel branches

Every launcher verifies `specialization.canonical_family()` matches the expected family string before any shell access, then branches on `plan.descriptor.operator_name()`.

### Output component-leading F-order layout
**Source:** `deriv1_cart_pair` lines 1270–1284 (ECP: `gctr[g]`, `gctr[dij+g]`, `gctr[2*dij+g]`)
**Apply to:** all 3-component gradient kernels

Layout is `[comp * block_len + n]` where `comp ∈ {0,1,2}` and `block_len = ni * nj [* nk * nl]`. This matches pyscf-gto `layout_table.rs` component-leading F-order (Risk R3). Verify against vendor in the oracle test.

### Error type conventions
**Source:** `cintxRsError::UnsupportedApi`, `cintxRsError::InvalidEnvParam`, `cintxRsError::ChunkPlanFailed`
- `UnsupportedApi` — unsupported operator name, unsupported nroots, spinor variants (D-03)
- `InvalidEnvParam` — missing/invalid `PTR_RINV_ORIG` (validation gate)
- `ChunkPlanFailed` — family mismatch guard at launcher entry

---

## No Analog Found

All files have close analogs in the codebase. No file requires falling back to RESEARCH.md patterns exclusively.

---

## Metadata

**Analog search scope:** `crates/cintx-cubecl/src/kernels/`, `crates/cintx-runtime/src/`, `crates/cintx-compat/src/`, `crates/cintx-capi/src/`, `crates/cintx-oracle/src/`, `crates/cintx-oracle/tests/`, `crates/cintx-ops/generated/`
**Files scanned:** 13 source files + 1 JSON manifest
**Pattern extraction date:** 2026-05-26
