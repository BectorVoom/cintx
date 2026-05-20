# Phase 20: Generic Float Precision (f64/f32 Switch) - Pattern Map

**Mapped:** 2026-05-20
**Files analyzed:** 17 new/modified across 6 crates (3 GENERICIZE crates, ECP host-only + ABI + tooling FROZEN)
**Analogs found:** 15 / 17 (2 new files have no in-repo analog — pattern templated from research)

> **Classification axis for this phase is GENERICIZE vs FREEZE**, not the usual
> role/data-flow. Every file below carries an explicit disposition:
> - **GENERICIZE** — thread `F: Float` (device `#[cube]`) or `F: CintFloat` (host/API). f64 is the default monomorphization, byte-identical to today.
> - **FREEZE** — stays concrete `f64`/`double`. Touch only at the kernel/staging conversion boundary. Includes ABI (`cintx-compat`/`cintx-capi`), oracle/xtask/test tooling, and ECP host-only paths.
>
> Per D-11 the executor performs every edit with serena MCP symbol tools
> (`find_symbol` / `find_referencing_symbols` / `replace_symbol_body` /
> `insert_before/after_symbol`) — never blind text replacement. FREEZE sites are
> identified by symbol and skipped.

---

## File Classification

| New/Modified File | Disposition | Role / Data Flow | Closest Analog | Match Quality |
|-------------------|-------------|------------------|----------------|---------------|
| `crates/cintx-core/src/precision.rs` | NEW (scaffolding) | model / type-def | `crates/cintx-cubecl/src/math/boys.rs` (const+trait conventions) | no-analog (new sealed trait) |
| `crates/cintx-cubecl/src/math/boys.rs` | GENERICIZE | utility (shared math) / transform | self (concrete→generic) | exact (self-refactor) |
| `crates/cintx-cubecl/src/math/obara_saika.rs` | GENERICIZE | utility (shared math) / transform | `math/boys.rs` (generic pattern) | exact |
| `crates/cintx-cubecl/src/math/rys.rs` | GENERICIZE | utility (shared math) / transform | `math/boys.rs` (generic pattern) | exact |
| `crates/cintx-cubecl/src/math/stg.rs` | GENERICIZE | utility (shared math) / transform | `math/boys.rs` (generic pattern) | exact |
| `crates/cintx-cubecl/src/math/pdata.rs` | GENERICIZE | utility (primitive-pair data) / transform | `math/boys.rs` (generic pattern) | role-match |
| `crates/cintx-cubecl/src/kernels/*.rs` (1e, 2e, 2c2e, 3c1e, 3c2e, 4c1e, f12, center_4c1e) | GENERICIZE | kernel-launcher / batch | `kernels/one_electron.rs` (launcher pattern) | exact |
| `crates/cintx-cubecl/src/transform/c2s.rs`, `transform/c2spinor.rs` | GENERICIZE | transform / transform | `math/boys.rs` (generic pattern) | role-match |
| `crates/cintx-cubecl/src/executor.rs` | GENERICIZE (capability branch only) | executor / request-response | self (`check_f64_capability`) | exact (self-refactor) |
| `crates/cintx-runtime/src/dispatch.rs` | GENERICIZE (staging cast boundary) | dispatch / streaming-buffer | self (`ExecutionIo`) | exact |
| `crates/cintx-runtime/src/planner.rs` | GENERICIZE (alloc by byte size) | planner / batch | self (`try_alloc_staging`) | exact |
| `crates/cintx-rs/src/api.rs` | GENERICIZE | controller (safe facade) / request-response | self (`evaluate` / `TypedEvaluationOutput`) | exact (self-refactor) |
| `crates/cintx-oracle/src/compare.rs` | FREEZE existing + ADD f32 profile | test-harness / batch | self (`tolerance_for_family` / `FamilyTolerance`) | exact |
| `crates/cintx-oracle/tests/f32_parity.rs` | NEW | test / batch | `crates/cintx-oracle/tests/safe_api_arity2_parity.rs` (existing parity test) | role-match |
| `crates/cintx-cubecl/src/math/{ecp_k_taylor,bessel,radial_quadrature}.rs`, `kernels/ecp.rs` | **FREEZE** (ECP host-only) | utility (host-only ECP) | n/a — stays f64 | n/a |
| `crates/cintx-compat/src/{raw,transform,lib}.rs` (`eval_raw`, env/atm/bas) | **FREEZE** (ABI) | compat ABI / request-response | n/a — libcint `double` ABI | n/a |
| `crates/cintx-capi/src/lib.rs` (C shim) | **FREEZE** (C ABI) | capi / request-response | n/a — D-07 f64-only | n/a |

---

## Pattern Assignments

### `crates/cintx-core/src/precision.rs` (NEW — sealed `CintFloat` trait + `PrecisionKind`)

**Analog:** No in-repo sealed-trait analog. Const/comment conventions copy from `crates/cintx-cubecl/src/math/boys.rs:25-26`. Trait body templated from RESEARCH.md Pattern 3.

**Why new:** Host `_host()` math needs `num_traits::Float` ops that CubeCL's device-only `Float` does not expose on the host. A sealed trait both bridges the two and forbids f16/bf16 monomorphization (D-01: only f64 and f32).

**Trait to create:**
```rust
mod sealed { pub trait Sealed {} }

/// The two concrete float precisions cintx supports (f64 default, f32 opt-in).
/// Bounds host-side math (`num_traits::Float`) and seals to f64|f32 only.
pub trait CintFloat:
    Copy + Send + Sync + 'static
    + num_traits::Float
    + num_traits::FromPrimitive
    + sealed::Sealed
{
    /// Inject an f64 const-table value at the host boundary (lossy for f32).
    fn from_f64_lossy(x: f64) -> Self;
}

impl sealed::Sealed for f64 {}
impl CintFloat for f64 { fn from_f64_lossy(x: f64) -> Self { x } }

impl sealed::Sealed for f32 {}
impl CintFloat for f32 { fn from_f64_lossy(x: f64) -> Self { x as f32 } }

/// Runtime precision tag for enum dispatch (keeps `BackendExecutor` object-safe).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrecisionKind { #[default] F64, F32 }

impl PrecisionKind {
    pub const fn element_size(self) -> usize {
        match self { Self::F64 => 8, Self::F32 => 4 }
    }
}
```

**before→after generic shape:** n/a (new). Device kernels still bound `F: cubecl::prelude::Float`; only host wrappers and the public `evaluate::<F>` bound on `F: CintFloat`.

**Wave:** 0 (scaffolding, no behavior change). Add `num-traits` as a direct dep in `crates/cintx-core/Cargo.toml` (already transitive).

---

### `crates/cintx-cubecl/src/math/boys.rs` (GENERICIZE — the reference refactor)

**Analog:** itself. This is the proven concrete-f64 file whose generic shape every other math/kernel file copies.

**Const tables stay f64 (FROZEN values, cast at boundary)** — `boys.rs:25-26,36`:
```rust
pub const SQRTPIE4: f64 = 0.886226925452758013649083741670572591398774728061193564106903894926;
pub const TURNOVER_POINT: [f64; 40] = [ /* fmt.c-exact values */ ];
```
> Do NOT make these `[F; 40]` — Rust forbids generic const arrays, and `F::new(lit)` takes an `f32` and would drop f64 precision (Pitfall 5). Keep `f64`; the host wrapper / launcher injects each entry via `<F as CintFloat>::from_f64_lossy(SQRTPIE4)`.

**Host wrapper — before→after** (`boys.rs:86`):
```rust
// BEFORE
pub fn boys_gamma_inc_host(t: f64, m: u32) -> Vec<f64> {
    let turnover = TURNOVER_POINT[m as usize];
    // ... boys_gamma_inc_impl(&mut f, t, m, turnover)
}
// AFTER  (host bound = CintFloat, NOT cubecl Float)
pub fn boys_gamma_inc_host<F: CintFloat>(t: F, m: u32) -> Vec<F> {
    let turnover = F::from_f64_lossy(TURNOVER_POINT[m as usize]);
    // ... boys_gamma_inc_impl::<F>(&mut f, t, m, turnover)
}
```

**Host C-erf linkage is FROZEN to f64** (`boys.rs:150-153`):
```rust
pub fn erf_host(x: f64) -> f64 {
    unsafe extern "C" { fn erf(x: f64) -> f64; }   // libm double erf — keep f64
    unsafe { erf(x) }
}
```
> For an `F` host path, wrap as `F::from_f64_lossy(erf_host(x.to_f64().unwrap()))`. The `extern "C" erf` symbol itself stays `f64` (host-only test/reference math, not the device path).

**Device `#[cube]` — before→after** (`boys.rs:173`, intrinsics at `:206-207`):
```rust
// BEFORE
#[cube]
pub fn boys_gamma_inc(f: &mut Array<f64>, t: f64, m: u32, turnover: f64) {
    // ...
    let erf_val = boys_erf_approx(tt);
    f[0usize] = erf_val * (SQRTPIE4 / tt);   // SQRTPIE4 is f64 const
}
// AFTER  (device bound = cubecl::prelude::Float; SQRTPIE4 now a passed param)
#[cube]
pub fn boys_gamma_inc<F: Float>(
    f: &mut Array<F>, t: F, m: u32, turnover: F, sqrtpie4: F,
) {
    // ...
    let erf_val = boys_erf_approx::<F>(tt);   // or F::erf(tt) — Erf is on Float for f32+f64
    f[0usize] = erf_val * (sqrtpie4 / tt);
}
```

**Intrinsic substitution map** (apply across every `#[cube]` body):
| Concrete (now) | Generic (after) | Verified for f32+f64? |
|----------------|-----------------|------------------------|
| `f64::exp(x)` | `F::exp(x)` | yes (manual `impl Exp for f32` + macro for f64) |
| `f64::sqrt(x)` | `F::sqrt(x)` | yes |
| `f64::erf(x)` / `boys_erf_approx(x)` | `F::erf(x)` / `boys_erf_approx::<F>(x)` | yes (`Erf` on `Float`) |
| natural log | `F::ln(x)` (NOT `F::log` — that is 2-arg) | yes (Phase 19 D-02) |
| `0.5f64` literal | `F::new(0.5)` (small/exact) or passed param (precision-critical const) | — |
| `Array<f64>` | `Array<F>`, indexed `[k as usize]` | — |

> Preserve CubeCL form: loop counters stay `u32`, function-call syntax `F::exp(x)` not `.exp()`, `Array` indexed via `as usize` (Established Patterns, CONTEXT §code_context).

**Wave:** 1. This file first; `obara_saika.rs`, `rys.rs`, `stg.rs`, `pdata.rs` follow the identical shape.

---

### `crates/cintx-cubecl/src/math/{rys,obara_saika,stg,pdata}.rs` (GENERICIZE)

**Analog:** `math/boys.rs` (generic shape above). `rys.rs` is the largest (~1,140 f64 lines: Horner/Clenshaw polynomials) — every f64 literal becomes `F` via `F::new(..)` for exact-small constants or a passed param for precision-critical ones. `transform/c2s.rs` and `transform/c2spinor.rs` follow the same shape; their CG/c2s coefficients are exact rationals — keep the table f64 and cast to `F` for the output accumulation (`Complex<F>` for spinor, see api.rs section).

**Wave:** 1 (math), then transforms with the kernels in Wave 2.

---

### `crates/cintx-cubecl/src/kernels/*.rs` (GENERICIZE — launchers)

**Analog:** `kernels/one_electron.rs` (launcher pattern). Each `launch_<family>()` gains `<F: Float>`, allocates generic device arrays, looks up f64 const tables and passes each entry as an `F` scalar param (`F::from_f64_lossy(TURNOVER_POINT[m])`), and calls the genericized math with `::<F>`.

**before→after shape:**
```rust
// BEFORE: pub fn launch_1e(io: &mut ExecutionIo, plan: &ExecutionPlan) -> Result<..>
// AFTER:  pub fn launch_1e<F: Float>(io: &mut ExecutionIo, plan: &ExecutionPlan) -> Result<..>
//   - device Array<f64> -> Array<F>
//   - boys_gamma_inc(..)  -> boys_gamma_inc::<F>(.., sqrtpie4)
//   - staging slice cast to &mut [F] via bytemuck (see dispatch.rs section)
```

**Wave:** 2.

---

### `crates/cintx-cubecl/src/executor.rs` (GENERICIZE — capability branch only)

**Analog:** itself. The only change is making the SHADER_F64 gate skip the f32 path (D-10). Body stays concrete; precision arrives via `PrecisionKind` on the plan (avoids making `BackendExecutor` generic, which would break `&dyn` object safety — Pitfall 3).

**FROZEN factored check (do not touch)** — `executor.rs:160-167`:
```rust
pub fn check_shader_f64_in_features(features: &[String]) -> Result<(), cintxRsError> {
    if !features.iter().any(|f| f == "SHADER_F64") {
        return Err(cintxRsError::UnsupportedApi {
            requested: "wgpu-capability:missing_shader_f64".to_owned(),
        });
    }
    Ok(())
}
```

**before→after** — `executor.rs:73-94` (note `_plan` becomes used):
```rust
// BEFORE
fn check_f64_capability(&self, backend: &ResolvedBackend, _plan: &ExecutionPlan<'_>)
    -> Result<(), cintxRsError> {
    match backend {
        ResolvedBackend::Wgpu(_, _) => check_shader_f64_in_features(backend.wgpu_features()),
        ResolvedBackend::Metal(_, _) => check_shader_f64_in_features(backend.wgpu_features()),
        // Cpu/Cuda/Rocm => Ok(())
    }
}
// AFTER  (add a precision-aware wrapper; keep f64 arm byte-identical)
fn check_capability(&self, backend: &ResolvedBackend, plan: &ExecutionPlan<'_>)
    -> Result<(), cintxRsError> {
    if plan.precision == PrecisionKind::F32 {
        return Ok(()); // f32 is WebGPU-baseline universal — no SHADER_F64 gate (D-10)
    }
    self.check_f64_capability(backend, plan) // unchanged f64 path
}
```
Update the two call sites at `executor.rs:180` and `:191` (`query_workspace`, execute) from `check_f64_capability` to `check_capability`. The existing `check_shader_f64_in_features` unit tests (`executor.rs:303-334`) stay green unchanged (PREC-04).

**Wave:** 3.

---

### `crates/cintx-runtime/src/dispatch.rs` (GENERICIZE — staging via bytemuck cast)

**Analog:** itself (`ExecutionIo`). RESEARCH Pattern 5 recommends NOT threading `<F>` into `ExecutionIo`/`BackendExecutor` (object-safety). Keep `staging_output() -> &mut [f64]`; the kernel launcher reinterprets it as `&mut [F]` with `bytemuck::cast_slice_mut` (both f32/f64 are `Pod`, zero-copy).

**FROZEN struct/method shape (keep f64 storage)** — `dispatch.rs:104-138`:
```rust
pub struct ExecutionIo<'a> {
    staging_output: &'a mut [f64],   // stays f64-typed byte buffer
    // ...
}
impl<'a> ExecutionIo<'a> {
    pub fn staging_output(&mut self) -> &mut [f64] { self.staging_output }
}
```

**Cast at the launcher boundary (the only new code, in `kernels/*.rs`):**
```rust
let staging_f: &mut [F] = bytemuck::cast_slice_mut(io.staging_output());
// write F values; planner already sized the byte buffer for size_of::<F>()
```

**Wave:** 3.

---

### `crates/cintx-runtime/src/planner.rs` (GENERICIZE — allocate by byte size, not element type)

**Analog:** itself (`try_alloc_staging`). The buffer stays `Vec<f64>` byte storage but is sized for `F`: element count scaled by `PrecisionKind::element_size()` so the f32 path reserves the right number of bytes for the bytemuck reinterpretation.

**FROZEN fallible-alloc contract (keep, per CLAUDE.md OOM-safe stop)** — `planner.rs:318-328`:
```rust
fn try_alloc_staging(elements: usize) -> Result<Vec<f64>, cintxRsError> {
    let bytes = elements
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or(cintxRsError::HostAllocationFailed { bytes: usize::MAX })?;
    let mut staging = Vec::new();
    staging.try_reserve_exact(elements)            // fallible alloc — NO best-effort partial write
        .map_err(|_| cintxRsError::HostAllocationFailed { bytes })?;
    staging.resize(elements, 0.0);
    Ok(staging)
}
```
**before→after:** thread the chunk's `PrecisionKind` so byte sizing is `elements * precision.element_size()` for the f32 buffer reinterpret. Keep the `try_reserve_exact` + typed `HostAllocationFailed` failure mode — never a partial write.

**Wave:** 3.

---

### `crates/cintx-rs/src/api.rs` (GENERICIZE — method-level generic public surface)

**Analog:** itself. D-03/D-04: `evaluate()` keeps its exact signature and delegates to `evaluate::<f64>()`; output types gain `F = f64` default so existing call sites compile unchanged.

**before→after — entry point** (`api.rs:125`):
```rust
// BEFORE
pub fn evaluate(self) -> Result<TypedEvaluationOutput, FacadeError> { /* full body */ }
// AFTER  (existing body moves into the generic; f64 shim delegates)
pub fn evaluate<F: CintFloat>(self) -> Result<TypedEvaluationOutput<F>, FacadeError> {
    /* existing body; owned_values typed Vec<F>; staging cast via bytemuck */
}
// preserve the unparameterized call site (D-12): a thin shim or default type param
pub fn evaluate_f64(self) -> Result<TypedEvaluationOutput<f64>, FacadeError> { self.evaluate::<f64>() }
```
> The `owned_values.resize(staging_elements, 0.0f64)` at `api.rs:206` becomes `F::zero()`; the `size_of::<f64>()` at `api.rs:678` (test) becomes `size_of::<F>()`.

**before→after — output structs** (`api.rs:497-511`):
```rust
// BEFORE
pub struct IntegralTensor { /* ... */ pub owned_values: Vec<f64> }
pub struct TypedEvaluationOutput { pub tensor: IntegralTensor, /* ... */ }
// AFTER  (F = f64 default keeps every existing reference compiling)
pub struct IntegralTensor<F = f64> {
    pub extents: Vec<usize>,
    pub component_axis_leading: bool,
    pub complex_interleaved: bool,
    pub owned_values: Vec<F>,
}
pub struct TypedEvaluationOutput<F = f64> {
    pub tensor: IntegralTensor<F>,
    pub stats: EvaluationStats,
    pub workspace_bytes: usize,
    pub chunk_count: usize,
    pub bytes_written: usize,
}
```
**Spinor/complex:** propagate as `num_complex::Complex<F>` (already generic, in workspace). Thread through the `complex_interleaved` accumulation in `transform/c2spinor.rs`.

**Wave:** 4.

---

### `crates/cintx-oracle/src/compare.rs` (FREEZE existing + ADD parallel f32 profile)

**Analog:** itself. The existing f64 gate is FROZEN (D-08: byte-identity, atol=1e-12). The f32 gate is a *parallel* profile that copies the `FamilyTolerance` / `tolerance_for_family` shape at looser tolerance. Oracle reference values stay `f64` (libcint is f64-only); the f32 comparison casts cintx `f32` output up to `f64` before differencing.

**FROZEN existing shape (do NOT genericize — reference buffers stay f64)** — `compare.rs:20,67-72,126-149`:
```rust
const UNIFIED_ATOL: f64 = 1e-12;     // f64 byte-identity floor — UNCHANGED

pub struct FamilyTolerance {
    pub family: &'static str,
    pub atol: f64,
    pub rtol: f64,
    pub zero_threshold: f64,
}
pub fn tolerance_for_family(family: &str) -> FamilyTolerance {
    // match arms documentation-only; all families -> UNIFIED_ATOL/UNIFIED_RTOL
    FamilyTolerance { family: static_family, atol: UNIFIED_ATOL, rtol: UNIFIED_RTOL, zero_threshold: ZERO_THRESHOLD }
}
```

**ADD (new, parallel — copy the shape above):**
```rust
pub const F32_UNIFIED_RTOL: f64 = 1e-4;   // conservative start; tighten per family empirically
pub const F32_UNIFIED_ATOL: f64 = 1e-7;

/// Parallel to `tolerance_for_family` but for the f32 gate (D-09).
/// Per-family floors are derived empirically (~1e-5..1e-4) then frozen here.
pub fn f32_tolerance_for_family(family: &str) -> FamilyTolerance {
    // start catch-all; tighten per family after the empirical sweep
    FamilyTolerance { family: /* static */, atol: F32_UNIFIED_ATOL, rtol: F32_UNIFIED_RTOL, zero_threshold: ZERO_THRESHOLD }
}
```
> The diff routine reuses `diff_summary` (`compare.rs:152`, `abs_error <= atol + rtol*abs_ref`) with `f32_out as f64` vs the `f64` libcint reference. Existing four f64 profiles are unchanged.

**Wave:** 5.

---

### `crates/cintx-oracle/tests/f32_parity.rs` (NEW — separate f32 oracle gate)

**Analog:** `crates/cintx-oracle/tests/safe_api_arity2_parity.rs` (and `arity3`/`arity4` siblings) — the existing safe-API parity test structure. Copy its fixture-sweep + artifact-write shape, swap `evaluate()` → `evaluate::<f32>()`, and assert against `f32_tolerance_for_family` rather than `tolerance_for_family`.

**Empirical floor procedure (RESEARCH §Per-Family f32 Tolerance):** run `evaluate::<f32>()` over all fixtures, record `max(abs(f32_out as f64 - libcint_ref)/abs(libcint_ref))` per family, set `f32_rtol = 10 * max_rel_error` rounded up. Gate is advisory first, promoted to required once floors stabilize. Does NOT replace the f64 gate (parallel CI job `oracle_parity_gate_f32`).

**Wave:** 5.

---

## FREEZE Inventory (explicit — stay f64 / `double`)

These are identified by symbol and SKIPPED by serena. Editing any of them is a regression.

| Frozen target | File(s) | f64 lines | Reason |
|---------------|---------|-----------|--------|
| ECP host-only math | `cintx-cubecl/src/math/ecp_k_taylor.rs`, `math/bessel.rs`, `math/radial_quadrature.rs`, `kernels/ecp.rs` | ~350 | ECP K-Taylor tables are byte-locked f64 blobs (`ecp_k_taylor_in.bin`, `..._order7.bin`) with a drift-gate; ECP math is host-only; PySCF reference is f64-only (Pitfall 6, A2) |
| Raw compat ABI | `cintx-compat/src/raw.rs`, `transform.rs`, `lib.rs` (`eval_raw`, env/atm/bas arrays) | 101 | libcint `double` ABI untouched (D-06); precision conversion happens at the kernel/staging boundary, not here |
| C ABI shim | `cintx-capi/src/lib.rs` | 5 | D-07: f64-only this milestone; no precision-suffixed C entry points |
| `cintx-core` domain types | `Shell.exponents: Arc<[f64]>`, `Atom.coord_bohr: [f64; 3]` | ~50 | Domain inputs stay f64; only the *output* (`TypedEvaluationOutput`) is parameterized |
| Oracle reference buffers | `cintx-oracle/src/compare.rs` f64 reference/diff buffers, `lib.rs` | 281 | Oracle compares against f64 libcint reference; f32 gate casts cintx output up to f64 (the reference side stays f64) |
| Const tables (values) | `boys.rs` `SQRTPIE4`/`TURNOVER_POINT`, `roots_xw_data.rs`, c2s/CG coefficient tables | — | Stay `[f64; N]` definitions (no generic const arrays); cast to `F` only at the host/launcher boundary |
| Host `extern "C" erf` | `boys.rs:150-153 erf_host` | — | libm `double erf` linkage; host-only reference math |
| `f12_zeta: Option<f64>` | `OperatorEnvParams` / `ExecutionOptions` | — | Physics env param; stays f64, cast at kernel boundary (Open Q3) |
| xtask / bench / tooling | `xtask/`, criterion benches, oracle harness glue | — | `anyhow`-boundary tooling; not on the compute path (CLAUDE.md split) |

---

## Shared Patterns

### Sealed-trait precision bound
**Source:** `crates/cintx-core/src/precision.rs` (NEW), templated from RESEARCH Pattern 3.
**Apply to:** every host wrapper (`*_host::<F: CintFloat>`) and the public `evaluate::<F: CintFloat>`. Device `#[cube]` fns instead bound `F: cubecl::prelude::Float` (CubeCL expansion requires its own trait). Never `where F: cubecl::prelude::Float + num_traits::Float` in the public signature (leaky bound).

### Const-table injection (FROZEN tables, cast at boundary)
**Source:** `crates/cintx-cubecl/src/math/boys.rs:25-26,36` (tables) + Phase 8 D-01 "pass `TURNOVER_POINT[m]` as a scalar param".
**Apply to:** every `#[cube]` kernel and launcher that reads a libcint-exact const. Host reads f64 const → `F::from_f64_lossy(..)` → passes as `F` scalar param. Never `const TABLE: [F; N]`, never `F::new(f64_literal)` for precision-critical constants.

### Generic intrinsic substitution
**Source:** `crates/cintx-cubecl/src/math/boys.rs:206-207` (`boys_erf_approx`, `SQRTPIE4 / tt`).
**Apply to:** every `#[cube]` body — `f64::exp/sqrt/erf → F::exp/sqrt/erf`, natural log → `F::ln` (not `F::log`), `Array<f64> → Array<F>`. Verified available for both f32 and f64 in cubecl-core 0.10.0.

### PrecisionKind enum dispatch (object-safety preserve)
**Source:** `crates/cintx-core/src/precision.rs` `PrecisionKind` + `crates/cintx-cubecl/src/executor.rs:73-94`.
**Apply to:** executor capability branch and runtime staging size. NEVER make `BackendExecutor::execute` generic (`&dyn BackendExecutor` would break — Pitfall 3). Carry precision as a plain `PrecisionKind` field on `ExecutionPlan` and `match` internally.

### Fallible-alloc / OOM-safe stop (FROZEN contract)
**Source:** `crates/cintx-runtime/src/planner.rs:318-328` (`try_reserve_exact` + typed `HostAllocationFailed`).
**Apply to:** all staging allocation. Per CLAUDE.md: fallible allocation + typed failure + no partial writes. The f32 path only changes the byte sizing (`* precision.element_size()`), not the failure contract.

### Parallel oracle profile (FREEZE f64 gate, ADD f32 gate)
**Source:** `crates/cintx-oracle/src/compare.rs:20,67-72,126-149` (`UNIFIED_ATOL`, `FamilyTolerance`, `tolerance_for_family`).
**Apply to:** the new `f32_tolerance_for_family` + `tests/f32_parity.rs`. f64 gate stays byte-identical (D-08); f32 is a separate, looser, parallel CI job (D-09).

### Serena symbol-aware edit (D-11, all waves)
**Source:** RESEARCH §Serena-Driven Refactor Sequencing.
**Apply to:** every edit. `find_symbol` → `find_referencing_symbols` → `replace_symbol_body` / `insert_*_symbol`. Scope queries to symbol names (function/type), never to the primitive `f64`. Call `check_onboarding_performed` / `initial_instructions` before Wave 1. FREEZE inventory above lists every symbol to skip (Pitfall 4).

---

## No Analog Found

| File | Disposition | Reason |
|------|-------------|--------|
| `crates/cintx-core/src/precision.rs` | NEW | No sealed-trait or `PrecisionKind` precedent in the repo; trait body templated from RESEARCH Pattern 3 (const/comment conventions copied from `boys.rs`). |
| `crates/cintx-oracle/tests/f32_parity.rs` | NEW | No f32 parity test exists; structure copied from the existing `safe_api_arity{2,3,4}_parity.rs` f64 gate, swapping in `evaluate::<f32>()` and `f32_tolerance_for_family`. |

---

## Metadata

**Analog search scope:** `crates/cintx-cubecl/src/{math,kernels,transform}/`, `crates/cintx-rs/src/api.rs`, `crates/cintx-cubecl/src/executor.rs`, `crates/cintx-runtime/src/{dispatch,planner}.rs`, `crates/cintx-oracle/src/{compare,lib}.rs`, `crates/cintx-oracle/tests/`.
**Files scanned:** boys.rs, api.rs, executor.rs, dispatch.rs, planner.rs, compare.rs (+ symbol-level greps across 8 crates).
**Refactor method (locked):** serena MCP symbol-aware tools only (D-11). Bottom-up waves 0→5; f64 path stays green at every wave boundary by threading `<f64>`.
**Pattern extraction date:** 2026-05-20

---

## PATTERN MAPPING COMPLETE

**Phase:** 20 - Generic Float Precision (f64/f32 Switch)
**Files classified:** 17 (GENERICIZE: 12 compute-path; FREEZE: 5 ABI/ECP/tooling clusters)
**Analogs found:** 15 / 17

### Coverage
- Files with exact (self-refactor) analog: 9
- Files with role-match analog: 6
- Files with no analog (new): 2

### Key Patterns Identified
- Const tables stay `[f64; N]` (FROZEN values); cast to `F` only at the host/launcher boundary via `CintFloat::from_f64_lossy` — never generic const arrays, never `F::new(f64_literal)`.
- Device kernels bound `F: cubecl::prelude::Float` (intrinsics `F::exp/sqrt/erf/ln` verified for f32+f64 in cubecl 0.10.0); host wrappers and public `evaluate::<F>` bound `F: CintFloat` (sealed to f32|f64).
- `PrecisionKind` enum dispatch keeps `BackendExecutor` object-safe; SHADER_F64 gate skipped only for the f32 path (D-10); fallible OOM-safe staging contract unchanged, only byte-sized for `F`.
- f64 oracle gate FROZEN byte-identical (D-08); a parallel `f32_tolerance_for_family` + `tests/f32_parity.rs` adds the looser ~1e-4 rtol gate (D-09). ABI (`cintx-compat`/`cintx-capi`), ECP host-only math, and xtask/tooling all stay f64.

### File Created
`/home/user/Documents/workspace/cintx/.planning/phases/20-precision-generic-f64-f32-switch/20-PATTERNS.md`

### Ready for Planning
Pattern mapping complete. Planner can reference each analog file + line range and the explicit GENERICIZE/FREEZE disposition when authoring PLAN.md files.
