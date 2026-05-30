# Phase 22: Gauge-Origin Env Slot (Gap A — `PTR_COMMON_ORIG`) - Pattern Map

**Mapped:** 2026-05-29
**Files analyzed:** 6 (5 modified + 1 fixture/harness)
**Analogs found:** 6 / 6 (every file has the exact Phase-21 `PTR_RINV_ORIG` precedent)

> **D-01 DIVERGENCE — READ FIRST.** The validator is the ONE place where `common_orig`
> must NOT clone the rinv precedent verbatim. `validate_rinv_orig_env_params` rejects
> `None` (presence check) because rinv has no sensible default origin. Gauge-origin
> DEFAULTS `None`→`[0,0,0]` (libcint reads unset env as zero), so the FND-01 gate is a
> **finiteness check on `Some(...)`**, NOT a presence rejection. There is also NO
> operator-name `.contains()` predicate yet (D-02 — operator-agnostic in this phase).
> Every OTHER file (field, const, env-read, setter, options) is a faithful field-for-field clone.

## File Classification

| Modified/New File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/cintx-runtime/src/planner.rs` | model (plan struct) | transform | `OperatorEnvParams.rinv_orig` :50-53 | exact |
| `crates/cintx-compat/src/raw.rs` (const) | config | — | `PTR_RINV_ORIG` const+doc :43-49 | exact |
| `crates/cintx-compat/src/raw.rs` (`eval_raw` read) | utility (env reader) | transform | rinv read+validate block :599-615 | exact |
| `crates/cintx-runtime/src/validator.rs` | utility (validator) | transform | `validate_rinv_orig_env_params` :173-199 + tests :385-426 | **role-match (D-01 semantic divergence)** |
| `crates/cintx-runtime/src/options.rs` | config | — | `ExecutionOptions.rinv_orig` field+doc :118-122 | exact |
| `crates/cintx-rs/src/api.rs` (options→plan) | utility (propagation) | transform | rinv propagation :193-196 | exact |
| `crates/cintx-rs/src/builder.rs` | utility (setter) | — | `with_rinv_origin` :96-105 | exact |
| `crates/cintx-oracle/src/fixtures.rs` (+ `vendor_ffi.rs`) | test fixture / harness | file-I/O | `build_h2o_sto3g_f12` :141-146 / `vendor_int1e_iprinv_sph` :596-628 | role-match |

**Note:** CONTEXT names only 5 modified files, but the rinv plumbing has a 6th wiring
site the planner must not miss — `crates/cintx-rs/src/api.rs:193-196` propagates
`ExecutionOptions.rinv_orig` → `plan.operator_env_params.rinv_orig` on the safe-API path.
`common_orig` needs the identical propagation, or the `with_common_origin` setter never
reaches the plan.

## Pattern Assignments

### `crates/cintx-runtime/src/planner.rs` (model, transform)

**Analog:** `OperatorEnvParams.rinv_orig`, lines 50-53.

**Field to clone** (add alongside `rinv_orig` inside `struct OperatorEnvParams`):
```rust
    /// PTR_RINV_ORIG value (env[4..6]) for iprinv/ECPscalar_iprinv kernels.
    /// Must be present when operator_name contains "iprinv".
    /// Absent (None) is allowed for all other operators.
    pub rinv_orig: Option<[f64; 3]>,
```
New field doc must reflect D-01 semantics — e.g.:
```rust
    /// PTR_COMMON_ORIG value (env[1..3]) — gauge origin for moment/GIAO families.
    /// None defaults to [0,0,0] (libcint reads unset env as zero); consumers use
    /// `common_orig.unwrap_or([0.0; 3])`. Validator checks finiteness only (D-01).
    pub common_orig: Option<[f64; 3]>,
```
`OperatorEnvParams` derives `Default` (line 42), so the new `Option` field defaults to
`None` automatically — no other construction sites change.

---

### `crates/cintx-compat/src/raw.rs` — `PTR_COMMON_ORIG` const (config)

**Analog:** `PTR_RINV_ORIG` const + doc, lines 43-49. The env-slot map comment at line 34
already documents `PTR_COMMON_ORIG = 1..3`.

**Const to clone** (place in the same const block, just above `PTR_RINV_ORIG`):
```rust
/// Index range in the libcint env array for the rinv origin (x, y, z).
///
/// libcint defines `PTR_RINV_ORIG = 4` (three consecutive slots 4, 5, 6).
/// Raw callers set `env[4..7] = [x, y, z]` (in Bohr) before calling any iprinv
/// integral. This constant is the start index; the full origin occupies slots
/// `PTR_RINV_ORIG`, `PTR_RINV_ORIG + 1`, `PTR_RINV_ORIG + 2`.
pub const PTR_RINV_ORIG: usize = 4;
```
New const: `pub const PTR_COMMON_ORIG: usize = 1;` with a doc adapted to slots 1,2,3
(gauge origin; default `[0,0,0]`).

---

### `crates/cintx-compat/src/raw.rs` — `eval_raw` env-read block (utility, transform)

**Analog:** rinv read+validate block, lines 599-615.

**Block to adapt** (note the operator-name guard and presence-validate — BOTH change for D-01/D-02):
```rust
    // Phase 21-01: Extract rinv_orig from env[PTR_RINV_ORIG..PTR_RINV_ORIG+3] for iprinv operators.
    // Raw callers must set env[4..7] = [x, y, z] (in Bohr) before calling any iprinv integral.
    // Guard with env.len() >= PTR_RINV_ORIG + 3 so a too-short env never indexes out of bounds
    // (T-21-01-01); if the origin is still None after the read, validate_rinv_orig_env_params
    // returns a typed InvalidEnvParam BEFORE kernel entry — no garbage-origin evaluation (T-21-01-02).
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

**HOW THE COMMON_ORIG VERSION DIFFERS (D-01 + D-02):**
- **NO operator-name guard** (`if is_iprinv_family_symbol(...)`). D-02: the slot is
  operator-agnostic this phase — read `env[1..3]` unconditionally (no family predicate
  exists to dispatch on yet). Keep ONLY the bounds guard:
  `if env.len() >= PTR_COMMON_ORIG + 3 { ... read into common_orig ... }`.
- Keep the same out-of-bounds bounds guard pattern (`env.len() >= PTR_COMMON_ORIG + 3`).
- Call the finiteness validator (see below) unconditionally after the read — it is a
  no-op when `common_orig` is `None` (D-01 default-is-zero), and only fires on
  non-finite `Some(...)`.

---

### `crates/cintx-runtime/src/validator.rs` (utility, transform) — **D-01 DIVERGENCE**

**Analog (STRUCTURE ONLY):** `validate_rinv_orig_env_params`, lines 173-199.

**Precedent source (DO NOT clone the `None => Err` arm):**
```rust
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

**What the gauge-origin validator MUST do instead (D-01 + D-02):**
- Keep the signature/error shape: `fn validate_common_orig_env_params(operator_name: &str, params: &OperatorEnvParams) -> Result<(), cintxRsError>` returning `cintxRsError::InvalidEnvParam { param: "PTR_COMMON_ORIG", reason }`. (`operator_name` may be unused/`_` per D-02; keep it for signature parity with the validator family — implementer's discretion per CONTEXT line 47.)
- **DROP** the `operator_name.contains(...)` predicate (D-02: operator-agnostic — no dispatchable operator exists yet; a name-list would be dead).
- **DROP** the `None => Err` arm. `None` is VALID (defaults to `[0,0,0]`).
- **ADD** a finiteness check on the `Some([x,y,z])` arm: if any component is not finite
  (`!v.is_finite()` — catches `NaN`/`±inf`), return `InvalidEnvParam`. `Some(finite)` and
  `None` both pass.

**Test patterns to clone/adapt:** analog tests at lines 385-426 (`rinv_orig_default_is_none`,
`validate_rinv_orig_rejects_none_for_iprinv`, `validate_rinv_orig_accepts_non_iprinv`,
`validate_rinv_orig_accepts_some`). The `OperatorEnvParams` struct-update test pattern:
```rust
    #[test]
    fn validate_rinv_orig_accepts_some() {
        let params = OperatorEnvParams {
            rinv_orig: Some([0.0, 0.0, 1.4]),
            ..OperatorEnvParams::default()
        };
        validate_rinv_orig_env_params("iprinv", &params)
            .expect("iprinv with rinv_orig=Some(...) must pass");
    }
```
Required D-01 unit tests (per CONTEXT line 34):
- `common_orig_default_is_none` — `OperatorEnvParams::default().common_orig.is_none()`.
- `validate_common_orig_accepts_none` — `None` → `Ok(())` (defaults to zero, no error).
- `validate_common_orig_accepts_some_finite` — `Some([0.5, -1.2, 0.0])` → `Ok(())`.
- `validate_common_orig_rejects_non_finite` — `Some([f64::NAN, 0.0, 0.0])` (and an `inf`
  case) → `Err(InvalidEnvParam { param == "PTR_COMMON_ORIG", .. })`.

Test module imports already present at validator.rs:202-207 (`use super::*;`, `OperatorEnvParams` in scope).

---

### `crates/cintx-runtime/src/options.rs` (config)

**Analog:** `ExecutionOptions.rinv_orig` field + doc, lines 118-122.

**Field to clone** (add alongside `rinv_orig`):
```rust
    /// Rinv origin for iprinv operators (env[4..6] in the raw API).
    /// When set, populates `operator_env_params.rinv_orig` on the `ExecutionPlan`.
    /// Must be present for any operator whose name contains "iprinv"
    /// (validated by `validate_rinv_orig_env_params` before kernel launch).
    pub rinv_orig: Option<[f64; 3]>,
```
New field `common_orig: Option<[f64; 3]>` — doc reflects D-01 (None defaults to `[0,0,0]`;
finiteness-validated, not presence-validated). Confirm `ExecutionOptions` derives `Default`
so the new field needs no manual default (the surrounding fields are all defaulted).

---

### `crates/cintx-rs/src/api.rs` (utility, transform) — **6th wiring site, not in CONTEXT file list**

**Analog:** rinv options→plan propagation, lines 193-196.

**Block to clone** (add immediately after the rinv block):
```rust
        // Propagate rinv_orig from ExecutionOptions to operator_env_params (safe API path, Plan 21-01).
        if let Some(origin) = self.request.options().rinv_orig {
            plan.operator_env_params.rinv_orig = Some(origin);
        }
```
The `common_orig` clone is a verbatim field swap (no D-01 divergence here — propagation is
pure assignment; finiteness is enforced by the validator, not this site). Without this the
safe-API setter is inert.

---

### `crates/cintx-rs/src/builder.rs` (utility, setter)

**Analog:** `with_rinv_origin`, lines 96-105.

**Setter to clone:**
```rust
    /// Set the rinv origin for iprinv operators (env[4..6] in the raw API).
    ///
    /// When set, `operator_env_params.rinv_orig` is populated on the `ExecutionPlan`
    /// for any operator whose name contains `"iprinv"`. Required for `int1e_iprinv`
    /// and `ECPscalar_iprinv` integrals; validated by `validate_rinv_orig_env_params`
    /// before kernel launch.
    pub fn with_rinv_origin(mut self, origin: [f64; 3]) -> Self {
        self.options.rinv_orig = Some(origin);
        self
    }
```
New setter `with_common_origin(mut self, origin: [f64; 3]) -> Self { self.options.common_orig = Some(origin); self }`
with a doc reflecting D-01 (gauge origin; defaults to `[0,0,0]` when unset; finiteness-validated).

---

### `crates/cintx-oracle/src/fixtures.rs` (+ `vendor_ffi.rs`) (test fixture / harness, file-I/O)

**Analog (fixture builder):** `build_h2o_sto3g_f12`, lines 141-146 — a thin wrapper that
takes the base H2O fixture and sets one global env slot.
```rust
pub fn build_h2o_sto3g_f12(zeta: f64) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let (atm, bas, mut env) = build_h2o_sto3g();
    // PTR_F12_ZETA = 9 — within the PTR_ENV_START global params block.
    env[PTR_F12_ZETA] = zeta;
    (atm, bas, env)
}
```
**Clone as** `build_h2o_sto3g_common_orig(origin: [f64; 3])` (or fixed non-zero origin):
start from `build_h2o_sto3g()`, then set `env[PTR_COMMON_ORIG..PTR_COMMON_ORIG+3] = origin`.
**The fixture's whole point is NON-ZERO `env[1..3]`** (CONTEXT line 103) — a zero origin is
indistinguishable from the default and proves nothing. Use a non-trivial origin (e.g.
`[0.5, -0.3, 0.8]`). `PTR_COMMON_ORIG` must be imported from `cintx_compat::raw` (the
`use cintx_compat::raw::{...}` block at fixtures.rs:3-6 already imports `PTR_F12_ZETA`,
`PTR_ENV_START`, etc. — add `PTR_COMMON_ORIG`).

**Analog (vendor harness):** `vendor_int1e_iprinv_sph`, lines 596-628 — note the doc
contract at :626-628: *"the caller MUST set `env[PTR_RINV_ORIG..PTR_RINV_ORIG+3]` before
calling."* The gauge-origin vendor reference (when MOM/GIAO land) follows the same shape:
caller sets `env[1..3]` before the FFI call; the env-array already carries the origin.

**D-03 / Claude's-discretion (CONTEXT line 48):** Phase 22 builds the fixture + harness
scaffolding as DATA infrastructure only — NO real byte-identity parity (no consuming
kernel exists). The vendor harness may be a no-op-now stub OR a fully-wired `vendor_*`
call that Phases 24/26 point a kernel at. Existing `vendor_ffi.rs` wrappers (e.g.
`vendor_int1e_ovlp_sph` :21, `vendor_int1e_iprinv_sph` :596) are the FFI-wrapper template
if a wired stub is chosen. The fixture-vs-vendor double-gate (`--features cpu` +
`CINTX_ORACLE_BUILD_VENDOR=1`, `#[cfg(has_vendor_libcint)]`) applies to any real parity
test added (CONTEXT line 70) — but per D-03 no such gated parity test is required to pass
this phase, only round-trip + validator unit tests.

---

## Shared Patterns

### Env-slot const block
**Source:** `crates/cintx-compat/src/raw.rs:31-60` (the `PTR_ENV_START`/`PTR_RINV_ORIG`/`PTR_F12_ZETA` block).
**Apply to:** the new `PTR_COMMON_ORIG = 1` const. The env-slot map comment at raw.rs:34
already documents `PTR_COMMON_ORIG = 1..3` — the const just makes the documented index real.

### Out-of-bounds env-read guard
**Source:** `crates/cintx-compat/src/raw.rs:605` — `if env.len() >= PTR_RINV_ORIG + 3 { ... }`.
**Apply to:** the `common_orig` env-read. Always guard `env.len() >= PTR_COMMON_ORIG + 3`
before indexing `env[PTR_COMMON_ORIG..]`.

### Validator signature + error shape
**Source:** `crates/cintx-runtime/src/validator.rs` — all validators take
`(operator_name: &str, params: &OperatorEnvParams) -> Result<(), cintxRsError>` and return
`cintxRsError::InvalidEnvParam { param: &'static str, reason: String }`.
**Apply to:** `validate_common_orig_env_params`. Same signature/error shape; ONLY the
predicate+match body diverges (D-01 finiteness, not presence; D-02 no operator-name match).

### Options→plan propagation (safe API)
**Source:** `crates/cintx-rs/src/api.rs:189-196` — `if let Some(x) = self.request.options().FIELD { plan.operator_env_params.FIELD = Some(x); }`.
**Apply to:** `common_orig`. Add a block mirroring the rinv one immediately after it.

### Setter on the builder
**Source:** `crates/cintx-rs/src/builder.rs:91-105` (`f12_zeta`, `with_rinv_origin`).
**Apply to:** `with_common_origin`. `mut self → set self.options.FIELD = Some(...) → self`.

### Single-slot fixture wrapper
**Source:** `crates/cintx-oracle/src/fixtures.rs:141-146` (`build_h2o_sto3g_f12`).
**Apply to:** the non-zero gauge-origin fixture — wrap `build_h2o_sto3g()` and set one
global env slot (`env[1..3]`).

## No Analog Found

None. Every file has a direct Phase-21 `PTR_RINV_ORIG` (or Phase-21/19 env-slot) precedent.
The only "new" behavior is the D-01 finiteness semantics in the validator, which is a
deliberate body-level divergence from an existing structural template — not an unprecedented file.

## Metadata

**Analog search scope:** `crates/cintx-runtime/src/{planner,validator,options}.rs`,
`crates/cintx-compat/src/raw.rs`, `crates/cintx-rs/src/{builder,api}.rs`,
`crates/cintx-oracle/src/{fixtures,vendor_ffi}.rs`, `crates/cintx-oracle/tests/center_2c2e_parity.rs`.
**Files scanned:** 9
**Pattern extraction date:** 2026-05-29
