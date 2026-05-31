# Phase 25: Group 2 — Hessian & Higher-Order Derivatives - Pattern Map

**Mapped:** 2026-05-30
**Files analyzed:** 11 distinct file targets (2 foundations + the 5-step registration recipe ×4 clusters)
**Analogs found:** 11 / 11 (every new/modified file has a strong in-repo analog — this is a source-derivation phase, not a greenfield one)

> **Verify-before-edit:** all line numbers below were read from current source this session, but the codebase churns. Re-`grep` the guard sites (FND-06) and the `rys.rs` panic block at plan time. The libcint-side derivations (gout order, `ng[]`, `component_rank`) come from RESEARCH.md and are frozen at 6.1.3.0.

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/cintx-cubecl/src/math/rys.rs` (FND-02 host port) | utility (numerics) | transform | Phase 19 ECP K-Taylor host-first port (`rys.rs` low-n polyfit branch is in-file) | role-match (host-first numeric port precedent) |
| `crates/cintx-cubecl/src/executor.rs` (FND-02 gate) | middleware (validator gate) | request-response | the existing `validated_4c1e_error` l-gate `:135-142` | exact (same file, same gate idiom) |
| `crates/cintx-runtime/src/planner.rs` (FND-06 assertion + OOM test) | service (chunk planner) | batch / OOM-safe-stop | `try_alloc_staging` `:341` + `staging_elements_for_chunk` `:321` + OOM test template `:1000` | exact (same boundary fn) |
| kernel scatter-guard strips (6 files, 19 sites) (FND-06) | kernel | file-I/O (staging scatter) | `one_electron.rs:6545` guard idiom (representative) | exact (identical idiom every site) |
| `crates/cintx-ops/generated/compiled_manifest.lock.json` (lock entries, all clusters) | config (manifest) | CRUD | `int1e_ipovlpip_*` rank-9 entries `:4051-4145` (clone, fix rank) | exact (rank-9) / role-match (rank-27/81) |
| `crates/cintx-compat/src/raw.rs` `RawApiId` consts + `eval_raw` (all clusters) | route (raw dispatch) | request-response | `INT1E_IPOVLPIP_*` consts `:279-290` + `eval_raw` `:704` | exact |
| `crates/cintx-cubecl/src/kernels/one_electron.rs` (HESS-01 launcher) | kernel (1e engine) | transform | `is_rank9_both` dispatch `:5987-6512` (ipovlpip/ipkinip/ipnucip) | exact (same engine, +2 bra) |
| `crates/cintx-cubecl/src/kernels/two_electron.rs` (HESS-02 launcher) | kernel (2e ERI engine) | transform | `int2e_ip1` host path `fill_g_tensor_2e` `:1459,1516,1765` + nroots guard `:2065` | role-match (gradient → Hessian) |
| `crates/cintx-cubecl/src/kernels/center_2c2e.rs` + `center_3c2e.rs` (HESS-03) | kernel (multi-center) | transform | their existing `ip1`/`ip2` launchers + scatter guard sites | role-match |
| `crates/cintx-cubecl/src/kernels/unstable/` 2e stubs (HESS-02 D-07 re-home) | kernel (delete + re-home) | transform | lock entries `:3325-3390` (`int2e_ipip1_sph`/`ipvip1_sph` to delete) | exact (delete target) |
| `crates/cintx-oracle/build.rs` allowlist + sources (all clusters) | config (build) | CRUD | `allowlist_function` regex `:358` + `.file()` list `:192-249` + suppl-header `:265-345` | exact |
| `crates/cintx-oracle/src/vendor_ffi.rs` safe wrappers (all clusters) | utility (FFI shim) | request-response | `vendor_int1e_ipnucip_sph/cart` `:691-735` | exact |
| `crates/cintx-oracle/tests/*_parity.rs` (per-cluster) | test | request-response | `tests/one_electron_grad_both_parity.rs` (rank-9, non-square, atol=1e-12) | exact (rank-9) / role-match (FND-02 sweep is new shape) |

---

## Pattern Assignments

### FND-02 — `crates/cintx-cubecl/src/math/rys.rs` (utility, transform)

**Analog:** the in-file low-n dispatcher + Phase 19 ECP K-Taylor host-first precedent (D-01).

**Panic site to replace** (`rys.rs:3244-3257`):
```rust
/// Unified host-side Rys quadrature dispatcher for nroots=1..5.
/// Panics if nroots > 5 (Wheeler fallback deferred to Phase 10).   // <- stale comment, update (D-state-of-the-art)
fn rys_roots_host_f64(nroots: usize, x: f64) -> (Vec<f64>, Vec<f64>) {
    match nroots {
        1 => { let (u, w) = rys_root1_host(x); (vec![u], vec![w]) }
        2 => { let (u, w) = rys_root2_host(x); (u.to_vec(), w.to_vec()) }
        3 => { let (u, w) = rys_root3_host(x); (u.to_vec(), w.to_vec()) }
        4 => { let (u, w) = rys_root4_host(x); (u.to_vec(), w.to_vec()) }
        5 => { let (u, w) = rys_root5_host(x); (u.to_vec(), w.to_vec()) }
        _ => panic!("rys_roots_host: nroots={nroots} > 5 not supported"),  // <- REPLACE with Wheeler dispatch
    }
}
```

**`CintFloat`-generic wrapper to preserve** (`rys.rs:3235-3242`) — the new `nroots≥6` path plugs into `rys_roots_host_f64`; the generic wrapper above it stays unchanged:
```rust
pub fn rys_roots_host<F: CintFloat>(nroots: usize, x: F) -> (Vec<F>, Vec<F>) {
    let x_f64 = x.to_f64().expect("CintFloat is f32|f64; to_f64 is total");
    let (roots, weights) = rys_roots_host_f64(nroots, x_f64);
    // ... map back to F
}
```

**Pattern to copy (from RESEARCH.md §FND-02, libcint `rys_roots.c:97-114`):** port the `CINTrys_roots` per-nroots dispatch verbatim, host-side, `lower==0` only (SR path → unimplemented stub). Chain: `CINTrys_jacobi` → `flocke_jacobi_moments` → `wheeler_recursion` → `_CINTdiagonalize` (the `eigh.c` `#else` vendored MRRR) → root transform `roots[i]/(1-roots[i])`, `weights[i]=c0[i*n]²·mu0`. Cap validated sweep at nroots 12 (quad disabled). Replicate `c99_sqrtl`/`c99_expl` for nroots≥8. Constant tables (`JACOBI_*`, `POLY_*`) via an xtask `gen-rys-tables` `--check` drift-gate (P19 `roots_xw_data.rs` precedent — **do not hand-transcribe**).

---

### FND-02 — `crates/cintx-cubecl/src/executor.rs` (middleware gate, request-response)

**Analog:** the in-file `validated_4c1e_error` l-gate (`:135-142`):
```rust
// Validated4C1E requires max(l)<=4.
if plan
    .shells
    .as_slice()
    .iter()
    .any(|shell| shell.ang_momentum > 4)
{
    return Err(validated_4c1e_error("max(l)>4"));
}
```
**PITFALL (RESEARCH Pitfall 2):** this is the **Validated4C1E** validator, NOT the global nroots gate. FND-02's real gate work is (a) the `rys.rs` panic replacement above, and (b) raising the launcher `nroots > MAX_DEVICE_NROOTS`/`> 5` guards to route nroots≥6 to the **host** `fill_g_tensor_2e` path (see two_electron.rs assignment). Extend this 4c1e l-gate to the validated ceiling (g/h/i) only as D-02's forward-looking foundation — validate it on the nroots sweep, not the family parity tests (D-03).

---

### FND-06 — `crates/cintx-runtime/src/planner.rs` (service, OOM-safe-stop)

**Analog:** the staging-allocation boundary (`:321-351`) — the single D-04 assertion site:
```rust
fn try_alloc_staging(elements: usize) -> Result<Vec<f64>, cintxRsError> {
    let bytes = elements
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or(cintxRsError::HostAllocationFailed { bytes: usize::MAX })?;
    let mut staging = Vec::new();
    staging
        .try_reserve_exact(elements)
        .map_err(|_| cintxRsError::HostAllocationFailed { bytes })?;
    staging.resize(elements, 0.0);
    Ok(staging)   // <- D-04: add upfront `staging.len() >= required` assertion here / at call site,
                  //    emit cintxRsError::BufferTooSmall { required, provided } (variant exists, error.rs:66)
}
```
`staging_elements_for_chunk` (`:321`) computes the per-chunk size from `plan.output_layout.staging_elements` (which `parse_component_multiplier`/`component_multiplier_for_descriptor` already sized by `component_rank`). **The single assertion (D-04) proves `staging.len() >= component_multiplier · per_component_elements` once; all 19 per-element scatter guards then become unconditional.**

**OOM test template** (`planner.rs:1000`, `try_alloc_staging_oom_safe_and_f32_lane_count_adequate`) — clone for D-05 `rank81_oom_no_partial_write`: set a memory limit below rank-81 staging, drive `int1e_ipipipiprinv` or `int2e_ipip1ipip2`, assert `Err(BufferTooSmall|ChunkPlanFailed)` AND the output buffer is byte-for-byte untouched.

---

### FND-06 — kernel scatter-guard strip (19 sites across 6 files) (kernel, staging scatter)

**Analog (representative idiom, `one_electron.rs:6544-6547`):**
```rust
let dst = staging_comp_base + ii + jj * ni_sph;
if dst < staging.len() {                    // <- STRIP after the upfront assertion (D-04)
    staging[dst] = F::from_f64_lossy(sph_tmp[mj * nsi + mi]);
}
```
→ becomes unconditional `staging[dst] = F::from_f64_lossy(...)`.

**The 19 sites (re-grep `if dst < staging.len()` per file before editing — D-04, RESEARCH §FND-06):**
| File | Sites |
|------|-------|
| `kernels/one_electron.rs` | 6545, 6569, 6736, 6760, 6973, 7028 |
| `kernels/two_electron.rs` | 1600, 1641, 1845, 1886, 2173, 2231 |
| `kernels/center_3c2e.rs` | 2525, 2559, 2767, 2801 |
| `kernels/center_2c2e.rs` | 736, 761 |
| `kernels/f12.rs` | 1784 |
| `kernels/unstable/grids.rs` | 1521 |

---

### Registration recipe (D-08) — applies to Clusters A/B/C/D

#### Step 1 — `crates/cintx-ops/generated/compiled_manifest.lock.json` (config, CRUD)

**Analog: the `int1e_ipovlpip_*` rank-9 entry (`:4051-4082`, cart shown)** — clone per family, fix `component_rank`:
```json
{
  "arity": 2,
  "canonical_family": "1e",
  "category": "1e",
  "compiled_in_profiles": ["base","with-f12","with-4c1e","with-f12+with-4c1e"],
  "component_rank": "9",            // <- 9 (ipip*), 27 (deriv3), 81 (deriv4) — D-10, derive from ng[] last elem
  "declared_in": "unknown",
  "feature_flag": "none",
  "forms": ["cart"],
  "helper_kind": "operator",
  "id": {
    "family": "1e",
    "operator": "ipovlpip",         // <- per-family operator name
    "representation": "cart",
    "symbol": "int1e_ipovlpip_cart"
  },
  "oracle_covered": true,           // <- flip true AFTER parity passes
  "profiles": ["base","with-f12","with-4c1e","with-f12+with-4c1e"],
  "stability": "stable"
}
```
Three entries per family (cart/sph/spinor); the **spinor** entry sets `oracle_covered: false` (`:4138`) and the kernel returns `UnsupportedApi` (D-11). After editing, `cargo build -p cintx-ops` regenerates `api_manifest.{rs,csv}`; **lock edits auto-sync `manifest-audit`** (no fixtures list to touch — memory `project_ipovlpip_rank9_kernel`).

**HESS-02 D-07 DELETE targets** (`:3325-3390`) — the two unstable sph-only stubs (`canonical_family: "unstable::source::2e"`, `component_rank: ""`, `oracle_covered: false`):
```json
{ "id": { "family": "unstable::source::2e", "operator": "ipip1",  "symbol": "int2e_ipip1_sph"  }, ... }  // :3325-3357 DELETE
{ "id": { "family": "unstable::source::2e", "operator": "ipvip1", "symbol": "int2e_ipvip1_sph" }, ... }  // :3358-3390 DELETE
```
Replace with stable cart+sph entries (`component_rank: "9"`, a stable family e.g. `"2e"`), and register `int2e_ip1ip2` (rank 9) + `int2e_ipip1ipip2` (**rank 81** — 4th-order 2e) fresh. One canonical entry per symbol, no alias (D-07).

#### Step 2 — `crates/cintx-compat/src/raw.rs` `RawApiId` consts (route)

**Analog: the Phase-23 both-side rank-9 consts (`:279-290`):**
```rust
// Phase 23 both-side rank-9 1e families (spinor returns UnsupportedApi, D-06).
pub const INT1E_IPOVLPIP_CART: Self = Self::Symbol("int1e_ipovlpip_cart");
pub const INT1E_IPOVLPIP_SPH: Self  = Self::Symbol("int1e_ipovlpip_sph");
pub const INT1E_IPOVLPIP_SPINOR: Self = Self::Symbol("int1e_ipovlpip_spinor");
```
Add `INT1E_IPIPOVLP_*`, `INT1E_IPIPNUC_*`, `INT1E_IPIPKIN_*`, `INT1E_IPIPRINV_*` (Cluster A); `INT2E_IPIP1_*`, `INT2E_IPVIP1_*`, `INT2E_IP1IP2_*`, `INT2E_IPIP1IPIP2_*` (Cluster B); `INT2C2E_IPIP1_*`, `INT3C2E_IPIP1_*`, `INT3C2E_IPIP2_*` (Cluster C); `INT1E_IPIPIPNUC_*`, `INT1E_IPIPIPIPRINV_*` + deriv3/deriv4 siblings (Cluster D, roster per RESEARCH §HESS-04).

**`eval_raw` dispatch (`raw.rs:704`)** — operator-family env reads are guarded by `is_*_family_symbol` helpers. Hessian rinv families (`ipiprinv`, `ipipiprinv`) reuse the existing `is_iprinv_family_symbol` rinv-origin read (`:761-772`); nuc families need no extra env. No new dispatch arm needed unless a new env slot is introduced — the executor selects the kernel by `operator_name()`.

#### Step 4 — `crates/cintx-oracle/build.rs` (config, CRUD)

**Allowlist regex (`:358`)** — append the new cart/sph symbols (the existing line already has `int1e_ipovlpip_sph|int1e_ipovlpip_cart|...|int1e_ipnucip_cart`). Add `int1e_ipipovlp_{sph,cart}`, `int1e_ipipnuc_{sph,cart}`, `int1e_ipipkin_{sph,cart}`, `int1e_ipiprinv_{sph,cart}`, `int2e_ipip1_{sph,cart}`, `int2e_ipvip1_{sph,cart}`, `int2e_ip1ip2_{sph,cart}`, `int2e_ipip1ipip2_{sph,cart}`, `int2c2e_ipip1_{sph,cart}`, `int3c2e_ipip1_{sph,cart}`, `int3c2e_ipip2_{sph,cart}`, and the deriv3/deriv4 symbols.

**Source `.file()` list (`:192-249`)** — `hess.c` (`:237`) and `int3c2e.c` (`:224`) are **ALREADY in the build** → HESS-01/02/03 need ONLY the allowlist regex extended (their C already compiles, RESEARCH key finding 1). **HESS-04 needs BOTH** `.file(libcint_root.join("src/autocode/deriv3.c"))` + `deriv4.c` added here AND the allowlist.

**Suppl-header `extern` decls (`:265-345`)** — add `extern CINTIntegralFunction int1e_ipip*_{sph,cart};` etc. for any symbol NOT in `cint_funcs.h` (follow the `int2e_stg_ipip1_sph` / `int1e_grids_ipip_sph` precedent in that block). Add `deriv3.c`/`deriv4.c` to the `rerun-if-changed` list (`:51-83`).

#### Step 4 — `crates/cintx-oracle/src/vendor_ffi.rs` safe wrappers (utility, FFI)

**Analog: `vendor_int1e_ipnucip_sph` (`:691-715`):**
```rust
pub fn vendor_int1e_ipnucip_sph(
    out: &mut [f64], shls: &[i32; 2], atm: &[i32], natm: i32,
    bas: &[i32], nbas: i32, env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipnucip_sph(
            out.as_mut_ptr(), ptr::null_mut(),
            shls.as_ptr() as *mut i32,
            atm.as_ptr() as *mut i32, natm,
            bas.as_ptr() as *mut i32, nbas,
            env.as_ptr() as *mut f64,
            ptr::null_mut(), ptr::null_mut(),
        )
    }
}
```
Clone one `_sph` + one `_cart` wrapper per new family, swapping the `ffi::` symbol. (3-center families take `&[i32; 3]` shls — follow the existing `int3c2e_ip1` wrapper shape.)

#### Step 5 — `crates/cintx-oracle/tests/*_parity.rs` (test)

**Analog: `tests/one_electron_grad_both_parity.rs`** — the rank-9 non-square parity template:
```rust
#![cfg(any(feature = "cpu", feature = "rocm"))]
const ATOL: f64 = 1e-12;
const NCOMP: usize = 9;                       // <- 9 / 27 / 81 per cluster

#[cfg(has_vendor_libcint)]                    // <- vendor gate (D-12)
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipnucip_h2o_sto3g_parity() {
    // eval_raw(RawApiId::INT1E_..., ...) compared to vendor_ffi::vendor_int1e_..._{sph,cart}
    // assert_eq!(count_mismatches(reference, observed, ATOL, RTOL), 0, "...");
}
```
**D-09 gate: every parity test MUST use a NON-SQUARE bra×ket block (p×d), and deriv4 dual-headroom families MUST use distinct bra≠ket l** so a transpose/truncation cannot pass (memory `project_1e_gpu_port_scalar_only`). New test files per RESEARCH §Validation: `tests/rys_nroots_sweep_parity.rs` (FND-02 nroots 6..12 vs vendor `CINTrys_roots` — **new shape, no direct analog; highest-priority Wave 0**), `tests/hess1e_ipip_parity.rs`, `tests/hess2e_parity.rs`, `tests/hess_multicenter_ipip_parity.rs`, `tests/deriv34_parity.rs`.

---

### HESS-01 launcher — `crates/cintx-cubecl/src/kernels/one_electron.rs` (kernel, transform)

**Analog: the `is_rank9_both` dispatch (`:5987-6512`)** — `ipovlpip`/`ipkinip`/`ipnucip` are the both-side (∇bra×∇ket) rank-9 families; HESS-01 `ipip*` are the **bra-only** rank-9 families (∇²bra, `ng[]={2,0,0,0,2,1,0,9}`). Copy the operator-name ladder idiom and the cart→sph/cart staging scatter:
```rust
let op_name = plan.descriptor.operator_name();
let is_ipovlpip = op_name == "ipovlpip";
let is_ipkinip  = op_name == "ipkinip";
let is_ipnucip  = op_name == "ipnucip";
let is_rank9_both = is_ipovlpip || is_ipkinip || is_ipnucip;   // <- add is_ipipovlp/ipipnuc/ipipkin/ipiprinv
```
**Routing (RESEARCH §HESS-01):** `int1e_ipipovlp`/`ipipkin` reuse the **overlap-derivative engine** (no Rys, `run_1e_grad_*_on_backend` analogs); `int1e_ipipnuc`/`ipiprinv` ride the **nuclear/Rys 1e path** — these are the in-phase FND-02 consumers. The nroots fail-closed guard (`:6479-6487`, `nuc_nroots_both` for ipnucip) must route nroots≥6 to the host path (FND-02) instead of `UnsupportedApi`. **Copy each family's gout `s→gout` permutation verbatim** from its own `hess.c` block (RESEARCH §Family Reference, `hess.c:548-557` for `int1e_ipipnuc`) — D-09.

### HESS-02 launcher — `crates/cintx-cubecl/src/kernels/two_electron.rs` (kernel, transform)

**Analog: the `int2e_ip1` host path (`:1459-1765`)** — gradient families route through host `fill_g_tensor_2e` (`:1516,1765`), NOT the device comptime kernel (capped at `MAX_DEVICE_NROOTS=5`, `:31`). Hessian = gradient engine applied twice (RESEARCH "Don't Hand-Roll": `G2E_D_I` on `g1→g3`). The nroots fail-closed guard:
```rust
if grad_shape.nroots > 5 {                    // :1459, :1711
    return Err(cintxRsError::UnsupportedApi { ... });
}
```
**FND-02 dependency (assumption A2):** confirm the Hessian launchers route nroots≥6 to `fill_g_tensor_2e` (host), not the device kernel — d-quartet Hessian elevation is the in-phase FND-02 trigger (D-03). HESS-02 `int2e_ipip1ipip2` is **rank 81** (`ng[]={2,0,2,0,4,1,1,81}`), the others rank 9.

### HESS-03 launcher — `center_2c2e.rs` + `center_3c2e.rs` (kernel, transform)

**Analog:** the existing `int2c2e_ip1`/`int3c2e_ip1`/`int3c2e_ip2` launchers in those files (Phase 21/24). `int3c2e_ipip2` raises **ket** headroom (`ng[]={0,0,2,0,2,1,1,9}`, third tuple element `k_inc=2` — D-09 ket-side). Reuse the existing nroots/scatter machinery; the FND-06 guard strip touches `center_2c2e.rs:736,761` and `center_3c2e.rs:2525,2559,2767,2801`.

---

## Shared Patterns

### Registration recipe (D-08) — applies to every family in every cluster
**Source:** `.planning/phases/23-…/23-CONTEXT.md` D-11 + memory `project_ipovlpip_rank9_kernel`.
**Apply to:** all new families.
5 steps: (1) lock entry (clone `int1e_ipovlpip_*`, fix `component_rank`) → `cargo build -p cintx-ops`; (2) `RawApiId` consts (`raw.rs:279` idiom); (3) launcher dispatch on `operator_name()`; (4) vendor FFI (allowlist regex `build.rs:358` + `vendor_ffi.rs:691` wrapper + confirm `.c` in `:192-249` source list); (5) `vendor_*` parity test. Lock edits auto-sync `manifest-audit` — **no separate fixtures list**.

### Vendor gate (D-12) — applies to every parity test
**Source:** `tests/one_electron_grad_both_parity.rs:9,188,305-306` + memory `reference_oracle_vendor_parity_invocation`.
**Apply to:** all `*_parity.rs` tests.
```rust
#![cfg(any(feature = "cpu", feature = "rocm"))]   // file-level
#[cfg(has_vendor_libcint)]                        // vendor parity (CINTX_ORACLE_BUILD_VENDOR=1)
#[cfg(feature = "cpu")]
#[test]
fn test_..._parity() { /* atol=1e-12, count_mismatches == 0 */ }
```
**Without BOTH `--features cpu` AND `CINTX_ORACLE_BUILD_VENDOR=1`, parity SILENTLY SKIPS** — only the `#[cfg(feature="cpu")]` determinism tests run.

### Component-rank-truncation hard rule (D-10) — applies to every lock entry
**Source:** memory `project_unstable_derivative_ports` (the 260530-9ay root-cause).
**Apply to:** every `component_rank` field.
A rank set too LOW silently TRUNCATES trailing components. `component_rank` MUST equal the `ng[]` last element (9 / 27 / 81), derived from libcint source, gated with a non-square block.

### Transpose discipline (D-09) — applies to every kernel + every test
**Source:** memory `project_1e_gpu_port_scalar_only`.
**Apply to:** every gout emit + every parity fixture.
Raise headroom on the **ket** (`ng[]`) per family (deriv4: bra+2 AND ket+2); copy the gout `s→gout` permutation verbatim per family; gate with a NON-SQUARE bra×ket block (p×d). A square block hides a transpose.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `crates/cintx-cubecl/src/math/rys.rs` n>5 Wheeler/MRRR body | utility | transform | No existing tridiagonal-eigensolver or Wheeler-recursion code in-repo. The ANALOG is *methodological* (Phase 19 ECP K-Taylor host-first port + table-blob drift-gate), not structural — the body is a fresh verbatim port of `eigh.c` `#else` + `rys_wheeler.c`. **Planner uses RESEARCH §FND-02 derivation, not an in-repo code analog.** This is the phase long-pole (Plan-1 MRRR-vs-QL spike, RESEARCH Open Question 2). |
| `tests/rys_nroots_sweep_parity.rs` | test | request-response | New test SHAPE (sweeps `vendor_CINTrys_roots(6..12)` directly, not a family integral). Reuse the gate cfg idiom from `one_electron_grad_both_parity.rs` but the body is new. |
| xtask `gen-rys-tables` subcommand | utility (codegen) | batch | No in-repo analog in this read; precedent is Phase 19's `roots_xw_data.rs`/`_sph_ine_tab` table-gen (memory) — planner should locate the existing P19 xtask table generator and clone its `--check` drift-gate. |

---

## Metadata

**Analog search scope:** `crates/cintx-ops/generated/`, `crates/cintx-compat/src/`, `crates/cintx-cubecl/src/{math,kernels,executor}`, `crates/cintx-runtime/src/`, `crates/cintx-oracle/{build.rs,src,tests}`.
**Files scanned:** 10 (manifest lock, raw.rs, one_electron.rs, two_electron.rs, rys.rs, executor.rs, planner.rs, build.rs, vendor_ffi.rs, one_electron_grad_both_parity.rs).
**Pattern extraction date:** 2026-05-30
