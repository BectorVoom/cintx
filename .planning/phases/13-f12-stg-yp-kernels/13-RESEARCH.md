# Phase 13: F12/STG/YP Kernels - Research

**Researched:** 2026-04-05
**Domain:** Slater-type geminal and Yukawa potential two-electron integral kernels; STG Clenshaw/DCT root algorithm; PTR_F12_ZETA env plumbing; oracle parity under `with-f12` profile
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**D-01:** When `env[9]` (PTR_F12_ZETA) is 0.0 for an F12/STG/YP symbol, the validator rejects with a typed `InvalidEnvParam` error before kernel launch. No silent fallback to plain Coulomb. Fail-closed, consistent with existing `UnsupportedApi` pattern.

**D-02:** The validation check lives in the `ExecutionPlan` validator (cintx-runtime), not in the kernel itself. The kernel can assume zeta > 0.

**D-03:** Each of the 10 with-f12 sph symbols gets its own kernel entry point in `kernels/f12.rs`. The 5 STG variants (base, ip1, ipip1, ipvip1, ip1ip2) and 5 YP variants are separate launch functions. No shared-core-with-flags abstraction.

**D-04:** STG and YP base kernels have fundamentally different ibase/kbase routing (per roadmap SC1). Derivative variants within each operator type share the same root-finding but differ in angular momentum increments matching libcint's `cint2e_f12.c` `CINTEnvVars` setup.

**D-05:** Port `CINTstg_roots` from `libcint-master/src/stg_roots.c` into a new `math/stg.rs` module in cintx-cubecl. Embed `DATA_X` and `DATA_W` tables from `roots_xw.dat` as `static [f64; N]` arrays with the same offset formula: `nroots * 196 * (iu + it * 10)`.

**D-06:** STG root computation is host-side only (like `rys_roots_host`). Results are uploaded to device as kernel arguments. No device-side Clenshaw recurrence.

**D-07:** The `COS_14_14` cosine table and Clenshaw recurrence helpers (`_clenshaw_dc`, `_matmul_14_14`, `_clenshaw_d1`) are ported as host-side Rust functions in `math/stg.rs`. The `t = min(t, 19682.99)` clamp from `CINTstg_roots` is replicated exactly.

**D-08:** Host wrapper function `stg_roots_host(nroots, ta, ua)` follows the established `rys_roots_host` pattern with `_host()` suffix for test accessibility.

**D-09:** Oracle parity fixtures reuse the existing H2O/STO-3G molecule and basis set, consistent with all prior oracle tests. Same PTR_ENV_START-aligned env layout.

**D-10:** All 10 with-f12 sph symbols tested against vendored libcint at atol=1e-12. Oracle confirms cart and spinor symbol counts for the with-f12 profile are zero (existing sph-only enforcement).

**D-11:** `ExecutionPlan` gains an `operator_env_params` field (or similar struct like `F12KernelParams`) carrying `PTR_F12_ZETA` value extracted from `env[9]` during plan construction when the operator family is with-f12.

**D-12:** The kernel dispatch in `kernels/mod.rs` gains an `"f12"` arm in `resolve_family_name()` routing to `f12.rs` launch functions. STG vs YP selection is based on the resolved symbol name.

### Claude's Discretion

- Internal factoring of Clenshaw recurrence helpers within `math/stg.rs`
- Exact `OperatorEnvParams` struct layout and naming
- Order of the 10 symbol implementations within plans
- Whether STG/YP share a common pdata setup before diverging at root computation
- Test molecule zeta value choice (non-zero, physically reasonable)

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| F12-01 | STG (Slater-type geminal) kernel implements modified Rys quadrature with tabulated polynomial roots matching libcint | `CINTstg_roots` algorithm fully analyzed in `stg_roots.c`; Clenshaw/DCT port strategy confirmed; t-clamp at 19682.99 identified |
| F12-02 | YP (Yukawa potential) kernel implements correct routing distinct from STG path | `g2e_f12.c` YP vs STG weight normalization difference documented; ibase/kbase routing divergence in `CINTinit_int2e_yp_EnvVars` identified |
| F12-03 | All 10 with-f12 sph symbols pass oracle parity against libcint at atol=1e-12 | Oracle infrastructure reuse confirmed; `build_h2o_sto3g` fixture extended with env[9]=zeta; vendor_ffi bindings already exist for F12 symbols |
| F12-04 | PTR_F12_ZETA (env[9]) is correctly plumbed through ExecutionPlan to kernel launchers | `ExecutionPlan` extension point identified in `planner.rs`; env[9] slot confirmed in `raw.rs` comment at PTR_F12_ZETA=9 |
| F12-05 | Oracle fixtures validate that zeta=0 is rejected or produces Coulomb-equivalent results explicitly | Validator extension point confirmed in `validator.rs`; `InvalidEnvParam` variant must be added to `cintxRsError` |
</phase_requirements>

---

## Summary

Phase 13 implements the Slater-type geminal (STG) and Yukawa potential (YP) two-electron integral kernels, covering all 10 `with-f12` sph symbols. The physics diverges from plain Rys quadrature only at root-finding: STG and YP call `CINTstg_roots` (a Clenshaw/DCT algorithm over a precomputed 2D grid of Chebyshev coefficients embedded from `roots_xw.dat`) instead of `CINTrys_roots`. After root-finding, the integral contraction pipeline (pdata, VRR fill, HRR 4D transfer, Cartesian contraction, cart-to-sph) is identical to the plain 2e kernel already implemented in `two_electron.rs`.

The primary work is: (1) porting `CINTstg_roots` to Rust in a new `math/stg.rs` module with the `roots_xw.dat` table embedded as static arrays; (2) creating `kernels/f12.rs` with separate STG and YP entry points that differ in weight post-processing (STG: `w *= (1-u) * 2*ua/zeta`; YP: `w *= u`); (3) extending `ExecutionPlan` with an `operator_env_params` field carrying the extracted `PTR_F12_ZETA` value; (4) adding a `InvalidEnvParam` error variant and a pre-launch validator check for `zeta==0.0`; and (5) extending the oracle gate closure test for the `with-f12` profile.

A critical implementation detail: the manifest has `canonical_family: "2e"` for all 10 F12/STG/YP symbols. This means D-12's `"f12"` arm in `resolve_family_name()` requires changing the manifest `canonical_family` field from `"2e"` to `"f12"` for those 10 entries, or alternatively routing within the existing 2e launch by consulting `plan.descriptor.entry.operator_name`. The cleanest approach is a manifest canonical_family change to `"f12"` for the 10 F12 entries, which cleanly separates dispatch.

**Primary recommendation:** Port `CINTstg_roots` exactly (preserving t-clamp, Clenshaw structure, roots_xw.dat indexing), create dedicated `kernels/f12.rs` with 10 separate entry points, update manifest canonical_family, extend `ExecutionPlan`, add `InvalidEnvParam` validation, and extend oracle fixtures with env[9] zeta plumbing.

---

## Standard Stack

### Core (unchanged from project baseline)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Rust stable | 1.94.0 (pinned in rust-toolchain.toml) | Compiler | Reproducible build per CLAUDE.md |
| `cubecl` | 0.9.x | Compute backend | Architecture mandate; CPU path for oracle CI |
| `thiserror` | 2.0.18 | Typed error surface | Library-facing error enum extension (`InvalidEnvParam`) |
| `anyhow` | 1.0.102 | Oracle harness errors | Matches existing oracle test error handling |

### New in Phase 13 (no new external crates)

All math is ported from vendored C source. No new crate additions.

| Addition | Location | Purpose |
|----------|----------|---------|
| `math/stg.rs` (new file) | `cintx-cubecl` | Host-side `CINTstg_roots` port: Clenshaw/DCT, COS_14_14 table, roots_xw.dat static arrays |
| `kernels/f12.rs` (new file) | `cintx-cubecl` | 10 separate launch entry points for STG/YP sph variants |
| `InvalidEnvParam` variant | `cintx-core/src/error.rs` | Typed rejection for `env[9]==0.0` on F12 calls |
| `operator_env_params` field | `cintx-runtime/src/planner.rs` `ExecutionPlan` | Carries extracted `PTR_F12_ZETA` to kernel launchers |

**Installation:** No new packages. All changes are internal.

**Version verification:** Not applicable — no new external dependencies.

---

## Architecture Patterns

### Existing Pattern: Host-Side Math with `_host()` Wrapper

Every math module follows the established convention from Phase 8:

```rust
// Confirmed pattern from crates/cintx-cubecl/src/math/rys.rs
// Each function has a _host() counterpart callable without GPU context.
pub fn rys_roots_host(nroots: usize, x: f64) -> (Vec<f64>, Vec<f64>) { ... }
```

`stg_roots_host(nroots: usize, ta: f64, ua: f64) -> (Vec<f64>, Vec<f64>)` follows this exact signature convention.

### Pattern: F12 Dispatch Architecture (CRITICAL: manifest canonical_family mismatch)

**Current manifest state (confirmed by direct inspection of `api_manifest.rs` lines 1710-1883):**
All 10 F12/STG/YP symbols have `canonical_family: "2e"`. This routes them to `launch_two_electron` via the existing `"2e"` arm.

**D-12 requires adding an `"f12"` arm.** For this arm to fire, the manifest `canonical_family` must be changed from `"2e"` to `"f12"` for the 10 F12 entries. This is the cleanest separation. The alternative (routing within `launch_two_electron` by checking `plan.descriptor.entry.operator_name`) would require passing operator identity through to the root-finding call, which is functionally equivalent but architecturally messier.

**Recommended approach (Claude's discretion — justified):**
Change `canonical_family: "2e"` to `canonical_family: "f12"` for all 10 STG/YP manifest entries. Add `"f12"` arm to `resolve_family_name()`. This cleanly separates F12 dispatch.

```rust
// crates/cintx-cubecl/src/kernels/mod.rs
fn resolve_family_name(canonical_family: &str) -> Option<FamilyLaunchFn> {
    match canonical_family {
        "1e" => Some(one_electron::launch_one_electron as FamilyLaunchFn),
        "2e" => Some(two_electron::launch_two_electron as FamilyLaunchFn),
        "2c2e" => Some(center_2c2e::launch_center_2c2e as FamilyLaunchFn),
        "3c1e" => Some(center_3c1e::launch_center_3c1e as FamilyLaunchFn),
        "3c2e" => Some(center_3c2e::launch_center_3c2e as FamilyLaunchFn),
        #[cfg(feature = "with-4c1e")]
        "4c1e" => Some(center_4c1e::launch_center_4c1e as FamilyLaunchFn),
        #[cfg(feature = "with-f12")]
        "f12" => Some(f12::launch_f12 as FamilyLaunchFn),
        _ => None,
    }
}
```

`f12::launch_f12` dispatches to STG or YP entry points based on `plan.descriptor.entry.operator_name.starts_with("stg")` vs `starts_with("yp")`.

### Pattern: STG vs YP Root Post-Processing (CRITICAL correctness difference)

From `g2e_f12.c` direct inspection (HIGH confidence):

**STG weight post-processing (lines 290-296):**
```c
double ua2 = 2. * ua / zeta;
for (irys = 0; irys < nroots; irys++) {
    w[irys] *= (1 - u[irys]) * ua2;   // STG-specific
    u[irys] = u[irys] / (1 - u[irys]);
}
```

**YP weight post-processing (lines 197-200):**
```c
for (irys = 0; irys < nroots; irys++) {
    w[irys] *= u[irys];                // YP-specific (different from STG)
    u[irys] = u[irys] / (1 - u[irys]);
}
```

The `u[irys] -> u[irys] / (1 - u[irys])` transformation is identical; only the weight factor differs. This is the ibase/kbase routing divergence mentioned in D-04 — the post-root transformation changes the effective Rys roots used in VRR, which changes which `f_g0_2d4d` branch is correct for each operator type.

After root post-processing, both operators use the identical VRR + HRR pipeline (same `b00`, `b10`, `b01`, `c00x/y/z`, `c0px/y/z` computation).

### Pattern: STG Root Algorithm (`CINTstg_roots`)

From `stg_roots.c` direct inspection (HIGH confidence):

```
Input: nroots, ta (=a0*r^2), ua (=0.25*zeta^2/a0)
Preprocessing:
  t = min(ta, 19682.99)  // THE EXACT CLAMP — must be replicated
  if t > 1.0: tt = log(t) * 0.9102392266268373 + 1.0
  else:        tt = sqrt(t)
  uu = log10(ua)
  it = floor(tt); tt = 2*(tt - it) - 1    // normalized in [-1, 1]
  iu = floor(uu + 7)                       // 0 <= iu <= 10
  uu = 2*(uu - (iu-7)) - 1                // normalized in [-1, 1]

Lookup: offset = nroots * 196 * (iu + it * 10)
        DATA_X[offset..], DATA_W[offset..]  — tables from roots_xw.dat

Algorithm:
  _clenshaw_dc(im, DATA_X+offset, uu, nroots)   // 2D DCT over u-axis
  _matmul_14_14(imc, im, nroots)                 // cos-DCT basis change
  _clenshaw_d1(rr, imc, tt, nroots)              // 1D eval over t-axis

  (same for weights using DATA_W)
  weights /= sqrt(ua)
```

The `roots_xw.dat` file is 3,567,230 lines and provides `DATA_X` and `DATA_W` as static C arrays. It must be embedded in Rust as `static` arrays. The file is included in the vendored source at `libcint-master/src/roots_xw.dat`.

### Pattern: `ExecutionPlan` Extension for PTR_F12_ZETA

From `planner.rs` direct inspection, `ExecutionPlan<'a>` currently has:
```rust
pub struct ExecutionPlan<'a> {
    pub basis: &'a BasisSet,
    pub descriptor: &'a OperatorDescriptor,
    pub representation: Representation,
    pub shells: ValidatedShellTuple,
    pub workspace: &'a WorkspaceQuery,
    pub dispatch: DispatchDecision,
    pub component_count: usize,
    pub output_layout: OutputLayoutMetadata,
}
```

Extension for F12:
```rust
pub struct ExecutionPlan<'a> {
    // ... existing fields ...
    pub operator_env_params: Option<OperatorEnvParams>,
}

#[derive(Clone, Debug)]
pub struct OperatorEnvParams {
    pub f12_zeta: Option<f64>,  // env[PTR_F12_ZETA] when family is "f12"
}
```

The `PTR_F12_ZETA = 9` env slot is confirmed in `cintx-compat/src/raw.rs` (comment at line 35). However, `BasisSet` does not carry a raw `env` array. The `env[9]` value must be passed through the caller API. **This is a design gap that the planner must address**: the safe API must expose a way to set `zeta` for F12 calls.

The simplest approach consistent with D-11: add `f12_zeta: Option<f64>` to `ExecutionOptions` (or `query_workspace` parameter) so the planner can populate `operator_env_params`. The validator reads this value to enforce D-01.

### Pattern: Derivative Variants and Angular Momentum Increments

From `cint2e_f12.c` direct inspection (HIGH confidence):

| Variant suffix | `ng[]` array | Notes |
|---------------|-------------|-------|
| base | `{0,0,0,0, 0,1,1,1}` | IINC=JINC=KINC=LINC=0 |
| ip1 | `{1,0,0,0, 1,1,1,3}` | IINC=1, ncomp=3 (gradient) |
| ipip1 | `{2,0,0,0, 2,1,1,9}` | IINC=2, ncomp=9 (Hessian-like) |
| ipvip1 | `{1,1,0,0, 2,1,1,9}` | IINC=1, JINC=1, ncomp=9 |
| ip1ip2 | `{1,0,1,0, 2,1,1,9}` | IINC=1, KINC=1, ncomp=9 |

The `ng[]` values drive `li_ceil = li + IINC`, `lj_ceil = lj + JINC` etc., which changes `nroots` and all downstream g-tensor dimensions. Derivative variants require more Rys roots than the base variant.

### Recommended Project Structure

No structural changes to existing crates. New files:

```
crates/cintx-cubecl/src/
├── math/
│   ├── mod.rs           (add pub mod stg;)
│   ├── stg.rs           (NEW: CINTstg_roots port + roots_xw.dat embed)
│   └── ...existing...
├── kernels/
│   ├── mod.rs           (add "f12" arm, add pub mod f12)
│   ├── f12.rs           (NEW: 10 entry points, stg/yp dispatch)
│   └── ...existing...
crates/cintx-runtime/src/
├── planner.rs           (ExecutionPlan + OperatorEnvParams)
├── validator.rs         (add zeta==0.0 check for f12 family)
crates/cintx-core/src/
├── error.rs             (add InvalidEnvParam variant)
crates/cintx-ops/src/generated/
├── api_manifest.rs      (change canonical_family "2e"->"f12" for 10 F12 entries)
crates/cintx-oracle/
├── tests/oracle_gate_closure.rs  (extend with with-f12 profile section)
└── src/fixtures.rs              (extend F12 fixture env setup with zeta)
```

### Anti-Patterns to Avoid

- **Shared STG/YP core with runtime zeta flag:** D-04 and D-03 explicitly prohibit this. YP and STG have different weight post-processing that is not a runtime flag — it is a different operator.
- **Calling `CINTrys_roots` when zeta==0:** The validator must reject before kernel launch (D-01, D-02). The kernel must never be reached with zeta==0.
- **Float equality test `zeta == 0.0`:** Use `zeta == 0.0_f64` (exact IEEE test). The validator check is: "if the caller set `env[9] == 0.0` (never written a value), reject". This is the correct fail-closed behavior.
- **Changing the HRR/VRR pipeline:** After root post-processing, the pipeline is identical to `two_electron.rs`. Do not duplicate the g-tensor fill logic — factor it out or call the same function.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| STG root Chebyshev evaluation | Custom polynomial solver | Port `CINTstg_roots` from `stg_roots.c` verbatim | Algorithm is highly tuned; any deviation breaks oracle parity |
| roots_xw.dat data | Re-derive tables | Embed `DATA_X`/`DATA_W` from vendored `roots_xw.dat` as static arrays | Tables require multiprecision arithmetic to generate; vendored source is the canonical reference |
| YP weight normalization | Guess or derive from theory | Copy lines 197-200 of `g2e_f12.c` exactly | The `w *= u[irys]` post-processing is non-obvious from first principles |
| VRR/HRR pipeline | New implementation | Reuse `vrr_fill_axis`, `hrr_lj2d_4d`, `hrr_ik2d_4d` etc. from `two_electron.rs` | Pipeline is identical to plain 2e after root post-processing |
| H2O oracle fixture | New test molecule | Reuse `build_h2o_sto3g()` from `oracle_gate_closure.rs` | Established pattern (D-09); only addition is `env[9] = zeta` |

**Key insight:** The STG/YP kernels are structurally identical to plain 2e except at root computation. The only new math is `CINTstg_roots`, which is a table lookup + Clenshaw recurrence. Do not redesign what works.

---

## Common Pitfalls

### Pitfall 1: roots_xw.dat Embedding Strategy

**What goes wrong:** `roots_xw.dat` is 3.5 million lines (large static array in C). Direct `include!` of a Rust file that size may hit compile-time limits or cause very slow compilation.

**Why it happens:** The C approach uses `#include "roots_xw.dat"` which is a single compilation unit. Rust does not have an equivalent preprocessor include for data.

**How to avoid:** Convert `roots_xw.dat` to a Rust `static [f64; N]` array during a build script (`build.rs`) or via a one-time generation script. Alternatively, use `include!` macro with a generated `.rs` file. The generated file should be checked into the repository under `cintx-cubecl/src/math/roots_xw_data.rs`. This avoids runtime file I/O and preserves the "embedded constant data" property.

**Warning signs:** Build times exceed 60 seconds for `cintx-cubecl`; compiler OOM during codegen.

### Pitfall 2: t-Clamp Exact Replication

**What goes wrong:** Implementing `t.min(19682.99)` as `t.min(19683.0)` or `t.clamp(0.0, 19682.99)` produces results that differ from upstream for large `ta` inputs.

**Why it happens:** `19682.99` is a specific float literal in `stg_roots.c` that defines the domain boundary. Using a different value changes which table segment is accessed.

**How to avoid:** Use `const T_MAX: f64 = 19682.99_f64;` with the exact value copied from `stg_roots.c` line 416: `if (t > 19682.99) t = 19682.99;`.

**Warning signs:** Oracle mismatches for shell pairs with large exponent products (high-angular-momentum shells or tightly contracted basis sets).

### Pitfall 3: canonical_family Mismatch

**What goes wrong:** If manifest `canonical_family` is left as `"2e"` and a `"f12"` arm is added to `resolve_family_name()`, the `"f12"` arm never fires — F12 symbols route to `launch_two_electron` and execute as plain Rys without STG root computation.

**Why it happens:** The dispatch uses `plan.descriptor.entry.canonical_family`. All 10 STG/YP manifest entries have `canonical_family: "2e"` (verified by direct inspection at lines 1724, 1741, 1758, 1775, 1792, 1809, 1826, 1843, 1860, 1877 of `api_manifest.rs`).

**How to avoid:** Change `canonical_family: "2e"` to `canonical_family: "f12"` for all 10 F12/STG/YP entries in `api_manifest.rs`. Verify with a test that `resolve_family_name("f12")` returns the F12 launch function and that `supports_canonical_family("f12")` returns `true` under `with-f12` feature.

**Warning signs:** `launch_two_electron` is called for `int2e_stg_sph`; oracle results for F12 symbols match plain Coulomb exactly (which would be wrong).

### Pitfall 4: zeta Plumbing Gap in Safe API

**What goes wrong:** `ExecutionPlan::new` needs `env[9]` to populate `operator_env_params.f12_zeta`, but `BasisSet` does not carry a raw env array. Neither does `ExecutionOptions`. There is no current pathway for the safe API caller to supply `zeta`.

**Why it happens:** The existing API is designed for standard 2e/1e integrals where no operator-specific float parameters beyond shell geometry are needed.

**How to avoid:** Add `f12_zeta: Option<f64>` to `ExecutionOptions` (or a new `OperatorParams` wrapper). The planner extracts this from options when `canonical_family == "f12"` and validates `zeta != 0.0`. The raw compat path reads `env[PTR_F12_ZETA]` = `env[9]` directly.

**Warning signs:** `operator_env_params` is always `None` because there is no source to populate it from; validator never triggers for zeta=0 tests.

### Pitfall 5: YP vs STG ibase/kbase Routing

**What goes wrong:** Both operators use `CINTinit_int2e_yp_EnvVars` for their `EnvVars` setup (STG calls yp init then overrides `f_g0_2e`). However the `f_g0_2d4d` function pointer selection (which HRR branch) is determined by the same ibase/kbase logic. The only difference is `f_g0_2e` (the root computation function). If STG and YP are implemented as separate entry points (D-03) with the same pdata/VRR/HRR pipeline, both correctly use the shared ibase/kbase routing — the PITFALL only arises if someone tries to share the root computation in a single entry point with a flag.

**How to avoid:** Per D-03, use separate entry points. Both STG and YP kernel functions call `stg_roots_host` to get roots/weights, then apply different post-processing (STG: `w *= (1-u) * 2*ua/zeta`; YP: `w *= u`), then feed into the shared VRR/HRR infrastructure.

### Pitfall 6: Derivative Variant nroots

**What goes wrong:** Derivative variants increase `nroots` via angular momentum increments (`IINC`, `JINC` etc.). Using the base variant's nroots for a derivative variant produces wrong results (or an out-of-bounds access into the Rys table).

**Why it happens:** `nroots = ceil((li_ceil + lj_ceil + lk_ceil + ll_ceil + 3) / 2)`. For ip1, `li_ceil = li + 1`, so nroots is larger than base.

**How to avoid:** Compute nroots from the derivative-adjusted angular momenta `li + IINC`, `lj + JINC`, `lk + KINC`, `ll + LINC`. The derivative variant entry points must apply the correct increments before calling `build_2e_shape` (or its F12 equivalent).

### Pitfall 7: Oracle env Layout for F12

**What goes wrong:** The oracle fixture calls `build_h2o_sto3g()` which produces `env` with PTR_ENV_START alignment but does not set `env[9]` (PTR_F12_ZETA). When the cintx kernel reads `operator_env_params.f12_zeta`, it gets `None` (or 0.0), and the validator rejects or the kernel produces wrong output.

**Why it happens:** The existing `build_h2o_sto3g()` function was written for base families; no F12 tests existed.

**How to avoid:** Create a `build_h2o_sto3g_f12(zeta: f64)` variant (or extend `build_h2o_sto3g` with an option) that sets `env[9] = zeta` in the PTR_ENV_START-aligned env. The vendored libcint call also reads `env[9]` directly, so both sides of the oracle comparison receive the same zeta value.

---

## Code Examples

Verified patterns from official sources (direct codebase inspection):

### STG Root Function Signature (from stg_roots.c)
```rust
// Source: libcint-master/src/stg_roots.c CINTstg_roots
// ta = a0 * r^2 (r is ij-kl center distance)
// ua = 0.25 * zeta^2 / a0
pub fn stg_roots_host(nroots: usize, ta: f64, ua: f64) -> (Vec<f64>, Vec<f64>) {
    // rr = roots (u values), ww = weights
    // Returns (rr, ww) analogous to rys_roots_host
    let t = ta.min(19682.99_f64);  // EXACT clamp from stg_roots.c line 416
    // ... Clenshaw/DCT pipeline ...
}
```

### STG vs YP Weight Post-Processing (from g2e_f12.c)
```rust
// STG (lines 290-296 of g2e_f12.c)
let ua2 = 2.0 * ua / zeta;
for irys in 0..nroots {
    weights[irys] *= (1.0 - roots[irys]) * ua2;
    roots[irys] = roots[irys] / (1.0 - roots[irys]);
}

// YP (lines 197-200 of g2e_f12.c)
for irys in 0..nroots {
    weights[irys] *= roots[irys];
    roots[irys] = roots[irys] / (1.0 - roots[irys]);
}
```

### oracle_gate_closure.rs F12 env setup (from existing build_h2o_sto3g pattern)
```rust
// Extend existing fixture for F12 oracle calls
fn build_h2o_sto3g_f12(zeta: f64) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let (atm, bas, mut env) = build_h2o_sto3g();
    // PTR_F12_ZETA = 9 (env slot reserved in PTR_ENV_START range)
    env[9] = zeta;  // env[0..20] is the global param block
    (atm, bas, env)
}
```

### ExecutionPlan Extension (from planner.rs current struct)
```rust
// crates/cintx-runtime/src/planner.rs
#[derive(Clone, Debug)]
pub struct OperatorEnvParams {
    pub f12_zeta: f64,  // extracted from caller-supplied zeta parameter
}

#[derive(Clone, Debug)]
pub struct ExecutionPlan<'a> {
    pub basis: &'a BasisSet,
    pub descriptor: &'a OperatorDescriptor,
    pub representation: Representation,
    pub shells: ValidatedShellTuple,
    pub workspace: &'a WorkspaceQuery,
    pub dispatch: DispatchDecision,
    pub component_count: usize,
    pub output_layout: OutputLayoutMetadata,
    pub operator_env_params: Option<OperatorEnvParams>,  // NEW
}
```

### InvalidEnvParam Variant (from error.rs current enum)
```rust
// crates/cintx-core/src/error.rs — add to cintxRsError
#[error("invalid env parameter {param}: {detail}")]
InvalidEnvParam {
    param: &'static str,
    detail: String,
},
```

### Validator PTR_F12_ZETA Check (from validator.rs pattern)
```rust
// crates/cintx-runtime/src/validator.rs
// Called during ExecutionPlan construction when canonical_family == "f12"
pub fn validate_f12_zeta(zeta: f64) -> Result<(), cintxRsError> {
    if zeta == 0.0_f64 {
        return Err(cintxRsError::InvalidEnvParam {
            param: "PTR_F12_ZETA",
            detail: "zeta must be non-zero for F12/STG/YP integrals".to_owned(),
        });
    }
    Ok(())
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| F12 symbols routed to 2e launch (wrong, no-op) | Separate `f12.rs` with STG roots | Phase 13 | Oracle parity for with-f12 profile |
| No env param validation | `InvalidEnvParam` typed error for zeta=0 | Phase 13 | Fail-closed F12 calls |
| roots_xw.dat as runtime file | Embedded static arrays | Phase 13 | Hermetic build, no I/O |

**Deprecated/outdated:**
- `canonical_family: "2e"` for STG/YP manifest entries: will change to `"f12"` in this phase.

---

## Open Questions

1. **roots_xw.dat embedding strategy**
   - What we know: File is 3.5M lines; contains `DATA_X[]` and `DATA_W[]` as C static double arrays; must become Rust `static [f64; N]` arrays.
   - What's unclear: Whether a `build.rs` code-generation step is required vs a checked-in pre-generated `.rs` file. Large static arrays can cause slow compile times.
   - Recommendation: Use a `build.rs` script that parses `roots_xw.dat` and generates `roots_xw_data.rs`. Check the generated file in to avoid build-time dependency on the C source file being present. Plan 1 should establish the embedding strategy and compile-time impact before writing any kernel code.

2. **Safe API zeta plumbing mechanism**
   - What we know: `ExecutionOptions` doesn't have a zeta field; `BasisSet` doesn't carry raw env; raw compat path reads `env[9]` directly.
   - What's unclear: Whether `ExecutionOptions` is the right place for `f12_zeta: Option<f64>`, or whether a separate `OperatorParams` wrapper is cleaner.
   - Recommendation: Add `f12_zeta: Option<f64>` to `ExecutionOptions` (simplest, consistent with existing option extension pattern). The validator reads this during `ExecutionPlan::new`.

3. **roots_xw.dat static array size**
   - What we know: The offset formula is `nroots * 196 * (iu + it * 10)` with `nroots` up to 14 and `it` up to ~196 (log-t domain), `iu` 0..10. Total entries: sum over nroots 1..14 of `nroots*(nroots-1)/2 * 19600`.
   - What's unclear: The exact array length — needs to be computed by inspecting the DATA_X/DATA_W declarations in `roots_xw.dat`.
   - Recommendation: Check the final C array size declaration in `roots_xw.dat` and record it before writing the embedding code.

---

## Environment Availability

This phase is code-only within the existing workspace. No new external tools required.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Vendored libcint | Oracle parity gate (with-f12 feature) | Already present | 6.1.3 | — |
| `cargo test --features cpu,with-f12` | Oracle gate CI | Already established | Rust 1.94.0 | — |
| `roots_xw.dat` | STG root table embedding | Present at `libcint-master/src/roots_xw.dat` | — | — |

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in test + cargo nextest |
| Config file | `rust-toolchain.toml` (pins Rust 1.94.0) |
| Quick run command | `cargo test --features cpu,with-f12 -p cintx-cubecl -- math::stg` |
| Full suite command | `cargo test --features cpu,with-f12 -p cintx-oracle -- oracle_gate` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| F12-01 | STG roots match libcint at atol=1e-12 for sample (ta, ua) | unit | `cargo test --features cpu,with-f12 -p cintx-cubecl -- stg_roots_host` | Wave 0 |
| F12-01 | t-clamp at 19682.99 replicated exactly | unit | `cargo test --features cpu,with-f12 -p cintx-cubecl -- stg_roots_t_clamp` | Wave 0 |
| F12-02 | YP vs STG produce distinct non-equal outputs for same inputs | unit | `cargo test --features cpu,with-f12 -p cintx-cubecl -- f12_stg_yp_differ` | Wave 0 |
| F12-03 | All 10 with-f12 sph symbols pass oracle parity at atol=1e-12 | oracle | `cargo test --features cpu,with-f12 -p cintx-oracle -- oracle_gate_f12` | Wave 0 |
| F12-03 | Cart and spinor symbol counts for with-f12 profile are zero | oracle | `cargo test --features cpu,with-f12 -p cintx-oracle -- f12_sph_only_enforcement` | Wave 0 |
| F12-04 | ExecutionPlan carries f12_zeta when family is f12 | unit | `cargo test -p cintx-runtime -- execution_plan_f12_zeta` | Wave 0 |
| F12-05 | zeta=0 call returns InvalidEnvParam | unit | `cargo test -p cintx-runtime -- f12_zeta_zero_rejected` | Wave 0 |
| F12-05 | Oracle fixture with zeta=0 triggers validator rejection | integration | `cargo test --features cpu,with-f12 -p cintx-oracle -- f12_zeta_zero_fixture` | Wave 0 |

### Sampling Rate

- **Per task commit:** `cargo test --features cpu,with-f12 -p cintx-cubecl -x`
- **Per wave merge:** `cargo test --features cpu,with-f12 --workspace`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `crates/cintx-cubecl/src/math/stg.rs` — unit tests for `stg_roots_host` (F12-01)
- [ ] `crates/cintx-cubecl/src/kernels/f12.rs` — launch function tests (F12-01, F12-02)
- [ ] `crates/cintx-runtime/tests/f12_plan_tests.rs` — ExecutionPlan zeta plumbing tests (F12-04, F12-05)
- [ ] `crates/cintx-oracle/tests/oracle_gate_closure.rs` — extend with `#[cfg(feature="with-f12")]` F12 section (F12-03)

---

## Project Constraints (from CLAUDE.md)

| Directive | Impact on Phase 13 |
|-----------|-------------------|
| Error handling: `thiserror` for public library errors | `InvalidEnvParam` variant must use `#[error(...)]` from `thiserror` in `cintx-core/src/error.rs` |
| Architecture: CubeCL as primary compute backend | STG root computation is host-side (D-06); kernels use same dispatch pattern as two_electron.rs |
| API Surface: Safe Rust API first | F12 zeta plumbing must flow through `ExecutionOptions` safe API, not raw env-reading |
| Compatibility: libcint 6.1.3 result compatibility | t-clamp, Clenshaw algorithm, weight normalization must match upstream verbatim |
| Artifacts to `/mnt/data` | Oracle parity artifacts for with-f12 profile must target `/mnt/data` with fallback |
| GSD workflow enforcement | All file changes must go through GSD workflow (`/gsd:execute-phase`) |

---

## Sources

### Primary (HIGH confidence — direct code inspection)

- `libcint-master/src/stg_roots.c` — `CINTstg_roots` algorithm: t-clamp=19682.99, Clenshaw/DCT over roots_xw.dat tables, `COS_14_14` constant array (196 values), `_clenshaw_dc`, `_matmul_14_14`, `_clenshaw_d1` helper functions
- `libcint-master/src/g2e_f12.c` — `CINTg0_2e_stg` lines 257-351 (STG weight post-processing); `CINTg0_2e_yp` lines 161-254 (YP weight post-processing); `CINTinit_int2e_yp_EnvVars` lines 21-145 (shared ibase/kbase routing)
- `libcint-master/src/cint2e_f12.c` — all 10 with-f12 symbol definitions; `ng[]` arrays confirming IINC/JINC/KINC/LINC per derivative variant; `CINTall_2e_stg_optimizer` calls
- `crates/cintx-ops/src/generated/api_manifest.rs` lines 1710-1883 — all 10 F12/STG/YP entries with `canonical_family: "2e"` (critical dispatch issue), `oracle_covered: false`, `feature_flag: FeatureFlag::WithF12`
- `crates/cintx-cubecl/src/kernels/mod.rs` — `resolve_family_name()` dispatch; `"f12"` arm absent; pattern for `#[cfg(feature = "with-4c1e")]` to mirror
- `crates/cintx-cubecl/src/kernels/two_electron.rs` — `build_2e_shape`, `vrr_fill_axis`, `hrr_lj2d_4d`/`hrr_kj2d_4d`/`hrr_il2d_4d`/`hrr_ik2d_4d`, `contract_2e_cart`, `launch_two_electron` — all reusable for F12 post-root-computation pipeline
- `crates/cintx-runtime/src/planner.rs` — `ExecutionPlan<'a>` struct (8 fields); extension point for `operator_env_params`
- `crates/cintx-runtime/src/validator.rs` — `validate_shell_tuple` pattern; no F12-specific validation present
- `crates/cintx-core/src/error.rs` — `cintxRsError` enum; no `InvalidEnvParam` variant present (must add)
- `crates/cintx-compat/src/raw.rs` line 35 — `PTR_F12_ZETA = 9` confirmed in comment
- `crates/cintx-oracle/tests/oracle_gate_closure.rs` — `build_h2o_sto3g()`, comparison helpers, existing family gate pattern to extend
- `libcint-master/src/roots_xw.dat` — 3,567,230 lines; `DATA_X[]` and `DATA_W[]` static double arrays

### Secondary (MEDIUM confidence)

- `.planning/research/STACK.md` lines 19-57 — F12 stack analysis confirming no new external crates needed; roots_xw.dat embedding strategy; PTR_F12_ZETA plumbing approach
- `.planning/research/SUMMARY.md` — F12 phase design; critical pitfalls 3 (zeta=0), 5 (YP/STG routing), 6 (t-clamp)

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new external crates; all additions are internal modules following established patterns
- Architecture: HIGH — dispatch issue (canonical_family mismatch) identified from direct code inspection; resolution path clear
- Pitfalls: HIGH — all 7 pitfalls derived from direct C source and Rust code inspection, not inference
- roots_xw.dat size: MEDIUM — exact array size not computed; impacts build script design

**Research date:** 2026-04-05
**Valid until:** 2026-05-05 (stable domain; roots_xw.dat and libcint C source won't change)
