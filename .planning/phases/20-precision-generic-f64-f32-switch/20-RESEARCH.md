# Phase 20: Generic Float Precision (f64/f32 Switch) - Research

**Researched:** 2026-05-20
**Domain:** Rust generic float type parameterization over a cross-cutting ~3,396-site codebase; CubeCL `Float` trait surface; wgpu f32 shader capability; serena-MCP-driven symbol-aware refactor sequencing; per-family single-precision oracle tolerance model.
**Confidence:** HIGH (core findings verified against installed crate source and wgpu docs); MEDIUM for monomorphization compile-time estimates and f32 tolerance floors (empirical, must be measured)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Generic over `F: Float` (NOT a Cargo feature, NOT a runtime enum dispatch). Two concrete monomorphizations: `f64` (default) and `f32`.
- **D-02:** CubeCL kernels become `#[cube] fn ...<F: Float>(...)`. Concrete `f64::exp/sqrt/erf` calls become `F`-generic intrinsics; f64 const tables (e.g. `SQRTPIE4`, `TURNOVER_POINT: [f64; 40]`) cast to `F`.
- **D-03:** Method-level generic: `SessionRequest<'basis>` setup stays monomorphic; `evaluate::<F>()` is generic returning `TypedEvaluationOutput<F>`; `evaluate()` delegates to `evaluate::<f64>()` so **every existing call site compiles unchanged**.
- **D-04:** `TypedEvaluationOutput` becomes generic with `owned_values: Vec<F>` (default `F = f64`). Spinor/complex outputs propagate as `Complex<F>` via `num-complex`.
- **D-05:** Full compute path parameterizes: kernels + shared math + staging buffers + safe-API outputs.
- **D-06:** Raw compat `env`/`atm`/`bas` arrays **stay f64** — libcint ABI untouched. Precision conversion happens at the kernel/staging boundary.
- **D-07:** C ABI shim (`cintx-capi`) stays f64-only.
- **D-08:** f64 path keeps strict byte-identity against libcint (existing per-family atol ~1e-12). Zero regression.
- **D-09:** f32 gets a **separate oracle gate** at a realistic single-precision tolerance (~1e-4 rtol; exact per-family floors empirical). Verified against libcint — not byte-identical.
- **D-10:** f32 unlocks the wgpu backend on adapters lacking `SHADER_F64`. The f32 path must NOT gate on `SHADER_F64`.
- **D-11:** Refactor MUST be performed using the serena MCP server's symbol-aware tools (`find_symbol`, `find_referencing_symbols`, `rename_symbol`, `replace_symbol_body`, `insert_before/after_symbol`), NOT blind text replacement.
- **D-12:** f64 stays the default everywhere. No existing public signature breaks.

### Claude's Discretion

- Exact per-family f32 tolerance floors (empirical, research-driven).
- Internal helper genericization order and any intermediate type-alias scaffolding.
- Whether to introduce a sealed `Scalar`/`CintFloat` super-trait bridging device-side CubeCL `Float` and host-side `num_traits::Float`.

### Deferred Ideas (OUT OF SCOPE)

- C-ABI f32 variants (`cintx-capi` stays f64-only this milestone).
- Raw compat `env`/`atm`/`bas` precision parameterization (kept f64).
- Runtime / mixed-precision per-call dispatch (chose static generic D-01).
- Other precisions (f16/bf16, extended) — explicitly out of scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PREC-01 | Generic `F: Float` threaded through full compute path (kernels, shared math, staging buffers, safe-API outputs); f64 monomorphization preserved; const tables cast to `F`. | CubeCL `Float` trait verified to include `Exp`, `Sqrt`, `Erf` for both f32 and f64; const-cast pattern identified. |
| PREC-02 | `evaluate::<F>()` method-level generic; `evaluate()` delegates to f64; `TypedEvaluationOutput<F>` with `Vec<F>`; `Complex<F>` for spinor; all existing call sites unchanged. | API shape in `api.rs:125` + `IntegralTensor` struct confirmed; `num-complex` 0.4.6 already in workspace. |
| PREC-03 | Raw compat `env`/`atm`/`bas` and C ABI shim stay f64; conversion at kernel/staging boundary. | ABI-freeze crate inventory complete: `cintx-compat` (101 f64 lines), `cintx-capi` (5 f64 lines) — all stay frozen. |
| PREC-04 | f64 path byte-identity unchanged; all existing oracle gates, manifest locks, tests pass. | f64 path compiles as monomorphization of generic — zero behavioral change. SHADER_F64 gate stays for f64. |
| PREC-05 | Separate f32 oracle gate at ~1e-4 rtol per-family; verified against libcint. | Oracle model from Phase 15 identified; `tolerance_for_family` extension pattern researched; empirical floor approach documented. |
| PREC-06 | f32 path does NOT gate on `SHADER_F64`; unlocks adapters without SHADER_F64. | wgpu f32 confirmed WebGPU-baseline universal (no feature flag). `check_f64_capability` bypass pattern identified. |
| PREC-07 | Refactor via serena MCP symbol-aware tools, not blind text replacement. | Serena MCP tool inventory documented; bottom-up refactor sequencing strategy provided. |
</phase_requirements>

---

## Summary

Phase 20 is a milestone-sized cross-cutting type-parameter refactor that threads a generic `F: Float` through the full compute path of cintx — ~3,607 `f64` occurrences across 8 crates, with roughly 2,521 in `cintx-cubecl` (the compute engine) and the rest split among `cintx-oracle` (harness), `cintx-compat` (ABI/env, frozen), `cintx-core` (domain types), and `cintx-runtime` (staging buffers). The primary motivation is unlocking the wgpu backend on GPUs that lack `SHADER_F64` capability — f32 is the WebGPU baseline and requires no feature flag, confirmed against wgpu 29.0.3 docs.

The two highest-priority research flags from `20-CONTEXT.md` are resolved: (1) the CubeCL 0.10.0 `Float` trait **does** include `Exp`, `Sqrt`, and `Erf` for both `f32` and `f64`, meaning all transcendentals the Boys/Rys/Obara-Saika kernels require are available as generic intrinsics; (2) f32 shader support is **universal** on all wgpu adapters (WebGPU baseline) — no capability check is needed, and the existing `check_shader_f64_in_features` call must simply be bypassed for the f32 path. One important subtlety: the `impl_unary_func!(Exp, ..., f32, f64)` macro line has `// f32` commented out, but a **manual `impl Exp for f32`** block exists (unary.rs:63), and `Erf`/`Sqrt` include `f32` normally. This means `f32::exp()` inside `#[cube]` functions WORKS via the blanket `impl<T: Exp + CubePrimitive> ExpExpand for NativeExpand<T>`.

The recommended sealed `CintFloat` super-trait bridging device-side CubeCL `Float` and host-side `num_traits::Float` (for `boys_gamma_inc_host` and other host-only math) is advisable and is fully specified below.

**Primary recommendation:** Use a bottom-up, wave-based refactor driven exclusively by serena MCP symbol-aware tools. Start with shared-math leaves (`boys.rs`, `rys.rs`, `obara_saika.rs`), then kernel launchers, then staging/dispatch, then safe API surface. Freeze `cintx-compat`, `cintx-capi`, and raw env/atm/bas ABI sites explicitly. Keep every commit green on the f64 path by threading `<f64>` at wave boundaries until the full chain is wired.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Generic `F` type parameter definition / sealed trait | `cintx-core` or `cintx-cubecl` | — | Must be in a crate that both `cintx-cubecl` (device) and `cintx-rs` (host) can depend on |
| `#[cube]` kernel genericization (`F: Float`) | `cintx-cubecl` (device kernels) | — | All CubeCL math and kernel launchers live here |
| Staging buffer precision (`&mut [F]`) | `cintx-runtime` (dispatch layer) | `cintx-cubecl` (allocator) | `ExecutionIo::staging_output()` returns `&mut [F]`; planner allocates `Vec<F>` |
| `evaluate::<F>()` public API | `cintx-rs` (safe facade) | — | Method-level generic on `SessionQuery::evaluate` |
| `TypedEvaluationOutput<F>` / `Vec<F>` | `cintx-rs` (safe facade) | — | Public output type parameterized by `F` |
| f32 oracle gate at loose tolerance | `cintx-oracle` | — | Parallel profile to Phase 15 per-family tolerance model |
| ABI freeze (env/atm/bas stay f64) | `cintx-compat` | `cintx-capi` | Libcint ABI is untouched; conversion at staging boundary |
| SHADER_F64 capability bypass for f32 | `cintx-cubecl` executor | — | `check_f64_capability` must branch on precision |
| Serena-driven symbol refactor | executor (serena MCP) | — | D-11: no blind text replacement |

---

## Standard Stack

### Core (all already in workspace)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cubecl` | 0.10.0 (pinned) | CubeCL `Float` trait and `#[cube]` macro | Verified in Cargo.lock; `Float` trait covers all needed transcendentals |
| `num-complex` | 0.4.6 | Generic `Complex<F>` for spinor outputs | Already in Cargo.lock; `Complex<F>` is generic by design |
| `num-traits` | 0.2.19 | Host-side float operations in `_host()` functions | Already transitive dep via cubecl; needed for `CintFloat` super-trait |
| `thiserror` | 2.0.18 | Library error types | Per CLAUDE.md |
| `bytemuck` | 1.x | Const-table byte casting | Already a direct dep of `cintx-cubecl` |

**No new package dependencies are required for this phase.** All needed crates are already in the workspace.

### Supporting (discretionary)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `approx` | 0.5 | Approximate float comparisons in tests | Already in `cintx-cubecl` dev-deps; use for f32 oracle tolerance assertions |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `F: cubecl::prelude::Float` in `#[cube]` kernels | A separate `num_traits::Float` bound | CubeCL's `Float` is required for device kernels; `num_traits::Float` is for host-only paths |
| A sealed `CintFloat` trait | Direct `where F: Float + num_traits::Float` | Sealed trait is cleaner for enforcing only f32/f64; avoids accidentally constructing f16 paths |

**Installation:** No new packages to install — all dependencies are already resolved.

---

## Package Legitimacy Audit

No new external packages are installed in this phase. All required crates (`cubecl`, `num-complex`, `num-traits`, `bytemuck`, `approx`) are already in `Cargo.lock`.

| Package | Registry | Status | Disposition |
|---------|----------|--------|-------------|
| `num-traits` 0.2.19 | crates.io | [OK] slopcheck verified | Approved — already transitive dep |
| `num-complex` 0.4.6 | crates.io | Already in workspace | Approved — existing dep |
| `cubecl` 0.10.0 | crates.io | Pinned per CLAUDE.md | Approved — locked |
| `bytemuck` 1.x | crates.io | Already in workspace | Approved — existing dep |

**Packages removed due to slopcheck:** None.
**Packages flagged as suspicious:** None.

---

## Architecture Patterns

### System Architecture Diagram

```
[SessionRequest::evaluate::<F>()]      <-- method-level generic (D-03)
        |
        v
[TypedEvaluationOutput<F>]             <-- Vec<F> output (D-04)
        |
        v
[CubeClExecutor::execute()::<F>]       <-- backend executor generic or dispatch
        |
        +-- [check_f64_capability] BYPASS for f32 path (D-10)
        |
        v
[ExecutionIo::staging_output::<F>]     <-- &mut [F] staging buffer
        |
        v
[kernels::launch_family::<F>()]        <-- per-family kernel launcher generic
        |
        v
[#[cube] fn kernel<F: Float>(...)]]    <-- CubeCL generic kernel
        |
        +-- [F::exp(x)]                <-- generic intrinsic (replaces f64::exp)
        +-- [F::sqrt(x)]               <-- generic intrinsic
        +-- [boys_erf_approx::<F>(x)]  <-- generic erf (or F::erf(x))
        +-- [TURNOVER_POINT[m] as F]   <-- const table cast
        |
        v
[host f64 env/atm/bas]                 <-- ABI boundary FROZEN (D-06)
        |
        v (conversion at staging boundary: f64 env -> F buffers)
[libcint-compatible output]
```

### Recommended Project Structure (changes only)

```
crates/cintx-core/src/
├── precision.rs          # NEW: CintFloat sealed trait (if adopted)
crates/cintx-cubecl/src/
├── math/
│   ├── boys.rs           # Genericize boys_gamma_inc, boys_erf_approx -> F: Float
│   ├── rys.rs            # Genericize rys_root1..5, clenshaw_d1 -> F: Float
│   ├── obara_saika.rs    # Genericize vrr_step -> F: Float
│   ├── stg.rs            # Genericize F12/STG/YP math -> F: Float
│   └── bessel.rs         # Genericize Bessel kernels -> F: Float (or freeze: host-only ECP)
├── kernels/
│   ├── one_electron.rs   # launch_1e::<F>()
│   ├── two_electron.rs   # launch_2e::<F>()
│   └── ...               # Per-family launchers generic
├── executor.rs           # check_f64_capability -> check_capability::<F>()
crates/cintx-runtime/src/
├── dispatch.rs           # ExecutionIo::staging_output -> &mut [F]
├── planner.rs            # try_alloc_staging -> Vec<F>; staging_elements_for_chunk
crates/cintx-rs/src/
├── api.rs                # evaluate::<F>(), TypedEvaluationOutput<F>
crates/cintx-oracle/src/
├── compare.rs            # f32 tolerance profile, tolerance_for_family_f32()
└── tests/
    └── f32_parity.rs     # NEW: f32 oracle gate at ~1e-4 rtol
```

### Pattern 1: CubeCL Generic Float Intrinsics

**What:** Replace concrete `f64::exp(x)`, `f64::sqrt(x)`, `f64::erf(x)` calls inside `#[cube]` functions with `F`-generic calls. Replace `f64` in `Array<f64>` with `Array<F>`.

**When to use:** Every `#[cube]` function in `cintx-cubecl/src/math/` and `kernels/`.

**Verified trait surface (cubecl-core 0.10.0):**
- `Exp`: `impl Exp for f64` (via macro) + `impl Exp for f32` (manual block, unary.rs:63) → `f32::exp(x)` works in `#[cube]`
- `Sqrt`: `impl_unary_func!(Sqrt, sqrt, ..., f32, f64)` → both types
- `Erf`: `impl_unary_func!(Erf, erf, ..., f32, f64)` → both types
- `Log` (natural log, call as `F::ln(x)`): implemented for both f32 and f64

**Important CubeCL-0.10.0 naming note** (established in Phase 19 D-02): natural log inside `#[cube]` is `F::ln(x)`, NOT `F::log(x)`. The `log` name refers to the 2-argument version.

```rust
// Source: Verified against cubecl-core-0.10.0/src/frontend/element/float.rs
// and cubecl-core-0.10.0/src/frontend/operation/unary.rs

// BEFORE (f64-concrete):
#[cube]
pub fn boys_gamma_inc(f: &mut Array<f64>, t: f64, m: u32, turnover: f64) { ... }

// AFTER (F-generic):
#[cube]
pub fn boys_gamma_inc<F: Float>(f: &mut Array<F>, t: F, m: u32, turnover: F) {
    // f64::exp(-t)  becomes:
    let e = F::exp(-t);
    // f64::sqrt(t)  becomes:
    let tt = F::sqrt(t);
    // boys_erf_approx(tt) — already uses F inside since it's #[cube]
    let erf_val = boys_erf_approx::<F>(tt);
    // 0.5f64 literal: F::new(0.5) or just cast via comptime
    let half = F::new(0.5);
    // SQRTPIE4 / tt: pass as parameter from host, typed as F
    // ...
}
```

### Pattern 2: Const-Table Casting

**What:** Const tables like `TURNOVER_POINT: [f64; 40]` and `SQRTPIE4: f64` are defined at module level as `f64` (the libcint-exact values). Inside `#[cube]` functions they must be cast to `F`.

**Resolution:** Tables remain `f64` in the static definition. The host wrapper passes each entry as `F` by casting at the host-to-device boundary via `F::new(val as f32)` (CubeCL's `F::new` takes an `f32` literal, so for double-precision accuracy, use a parameter approach):

```rust
// APPROACH: pass the constant as a parameter (already established in Phase 8 D-01)
// The host wrapper looks up TURNOVER_POINT[m] as f64 and passes it as F via as-cast.

// Host wrapper (non-cube, runs on host):
pub fn boys_gamma_inc_host<F>(t: F, m: u32) -> Vec<F>
where
    F: CintFloat,  // see CintFloat sealed trait below
{
    let turnover = TURNOVER_POINT[m as usize];  // f64 lookup
    let turnover_f = <F as CintFloat>::from_f64(turnover);
    let mut f = vec![F::zero(); (m + 1) as usize];
    boys_gamma_inc_impl::<F>(&mut f, t, m, turnover_f);
    f
}
```

**For device kernels** (`#[cube]` functions): constants are passed as parameters from the kernel launcher (the Phase 8 established pattern — TURNOVER_POINT[m] passed as `turnover: f64` scalar parameter). Generalizing: the launcher reads the f64 table, casts to `F` (host-side cast is sound since f32 has ~7 decimal digits and the turnover points are 6 digits), and passes as `turnover: F`.

**Casting soundness for f32:** The turnover thresholds (e.g. `0.866025403784`, `1.295010032056`) have 12 significant digits. Casting to f32 introduces ~2e-7 relative error, but these are branch thresholds — an off-by-one-ULP branch miss causes at most a path switch between power-series and erfc, both of which converge. This is acceptable for the f32 tolerance envelope (~1e-4 rtol). [ASSUMED — empirical verification required]

### Pattern 3: CintFloat Sealed Super-Trait (RECOMMENDED)

**What:** A sealed trait that bridges device-side CubeCL `Float` and host-side `num_traits::Float`, plus the conversion utilities needed by `_host()` functions.

**Recommendation: YES, introduce `CintFloat`.** The host math functions (`boys_gamma_inc_host`, `boys_gamma_inc_impl`, and future `rys_root1_host`-equivalents) need host-side float ops (basic arithmetic, comparison) but cannot use CubeCL's `Float` trait (no `#[cube]` expansion context). The `num_traits::Float` trait provides `exp()`, `sqrt()`, `zero()`, `one()`, `from_f64()` etc. for host use.

**Sealed trait design:**

```rust
// Source: [ASSUMED] pattern; will be defined in cintx-core or cintx-cubecl

/// Private sealing module — prevents external implementations.
mod sealed { pub trait Sealed {} }

/// CintFloat: the two concrete float types supported by cintx (f64 and f32).
/// - For device kernels: also bounds `cubecl::prelude::Float`.
/// - For host-side math: bounds `num_traits::Float`.
/// Only `f64` and `f32` implement this trait.
pub trait CintFloat:
    num_traits::Float    // host: exp, sqrt, from_f64, zero, one
    + sealed::Sealed
    + Copy
    + Send
    + Sync
    + 'static
{
    /// Convert from f64 at the host level (for const-table injection).
    fn from_f64_lossy(x: f64) -> Self;
}

impl sealed::Sealed for f64 {}
impl CintFloat for f64 {
    fn from_f64_lossy(x: f64) -> Self { x }
}

impl sealed::Sealed for f32 {}
impl CintFloat for f32 {
    fn from_f64_lossy(x: f64) -> Self { x as f32 }
}
```

**Where to define it:** `cintx-core/src/precision.rs` (or `cintx-cubecl/src/precision.rs` if the `num_traits` dep is not already in `cintx-core`). `cintx-core` currently only uses `f64` literals in `atom.rs` / `shell.rs` — adding `num-traits` as a direct dep of `cintx-core` is acceptable since it is already a transitive dep (via cubecl in the workspace).

**Kernel-level bound:** `#[cube] fn ...<F: cubecl::prelude::Float>(...)` — device kernels use CubeCL's `Float` directly, not `CintFloat`, since CubeCL's expand machinery requires its own trait. The `CintFloat` bound is for host-side wrappers and the public API type parameter.

**Alternative (without CintFloat):** Bound as `where F: cubecl::prelude::Float + num_traits::Float`. This avoids a new sealed trait but leaks both trait names into the public API. Given the milestone scale, the sealed trait is the cleaner boundary.

### Pattern 4: Executor Capability Branch for f32

**What:** `check_f64_capability` in `executor.rs` must be bypassed for the f32 path.

**Current code** (executor.rs:73–93):
```rust
fn check_f64_capability(&self, backend: &ResolvedBackend, _plan: ...) -> Result<(), cintxRsError> {
    match backend {
        ResolvedBackend::Cpu(_) => Ok(()),
        ResolvedBackend::Wgpu(_, _) => check_shader_f64_in_features(backend.wgpu_features()),
        // ...
    }
}
```

**After generic dispatch, the plan (or a precision tag) must carry the requested precision type.** The simplest approach (Claude's discretion): add a `precision: PrecisionKind` field to `ExecutionPlan` (or thread it as a second type parameter), and in `execute()`/`query_workspace()` skip the f64 capability check when `precision == F32`. Since `ExecutionPlan` is already a struct (not generic), a `PrecisionKind { F64, F32 }` enum field avoids making the plan generic:

```rust
// In cintx-runtime, new enum (no new public API surface)
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum PrecisionKind { #[default] F64, F32 }

// In executor:
fn check_capability(&self, backend: &ResolvedBackend, plan: &ExecutionPlan) -> Result<(), ...> {
    if plan.precision == PrecisionKind::F32 {
        return Ok(()); // f32 is universally supported — no SHADER_F64 needed
    }
    // existing f64 check:
    match backend { ... check_shader_f64_in_features(...) ... }
}
```

**Why this is correct:** f32 (FloatKind::F32) is unconditionally registered in the wgsl backend with `TypeUsage::all()` (verified in cubecl-wgpu-0.10.0/src/backend/wgsl.rs:78). SHADER_F64 gates only `FloatKind::F64` registration. f32 compute shader support is the WebGPU baseline — confirmed via wgpu 29.0.3 docs. [VERIFIED: wgpu docs.rs/wgpu/29.0.3]

### Pattern 5: Staging Buffer Genericization

**What:** `ExecutionIo::staging_output()` currently returns `&mut [f64]`. After genericization it must return `&mut [F]`. The allocator in `planner.rs::try_alloc_staging()` returns `Vec<f64>`.

**Challenge:** `ExecutionIo` is not currently generic. Making it generic (`ExecutionIo<'a, F>`) propagates the type parameter into `BackendExecutor` trait, kernel launchers, and the planner. This is a significant but mechanical change.

**Recommended approach:** Thread `F` as a type parameter on `ExecutionIo<'a, F>` and `BackendExecutor` trait. The `staging_output()` method returns `&mut [F]`. Kernel launchers receive typed staging slices. Since `BackendExecutor` is used as `&dyn BackendExecutor` (07-CONTEXT.md D-02), making it generic would break object safety — instead, use a concrete enum dispatch through `CubeClExecutor::execute_typed::<F>()` called from a non-generic wrapper, or make `execute` take a `PrecisionKind` enum and internally downcast the staging slice.

**Simpler alternative:** Keep `staging_output() -> &mut [f64]` internally but have the kernel launcher cast to `&mut [F]` via `bytemuck::cast_slice_mut` (zero-copy reinterpretation, safe when `F: bytemuck::Pod`). Both `f32` and `f64` are `Pod`. This avoids threading `F` into `ExecutionIo` and `BackendExecutor` entirely.

```rust
// In kernel launcher:
let staging_f: &mut [F] = bytemuck::cast_slice_mut(io.staging_output());
// write F values directly
```

**Recommendation:** The bytemuck cast approach is simpler for this refactor scope and avoids a 2nd wave of type-parameter threading into runtime traits. The planner still allocates in bytes (`size_of::<F>() * elements`), which it can compute at runtime via `PrecisionKind::element_size()`.

### Anti-Patterns to Avoid

- **Blind `sed -i 's/f64/F/g'`:** Corrupts deliberately-f64 ABI sites (env/atm/bas), comment text, and const table values. D-11 requires serena MCP tools.
- **Making `BackendExecutor` generic on `F`:** Breaks `&dyn BackendExecutor` object safety. Use a `PrecisionKind` enum dispatch instead.
- **`const TURNOVER_POINT: [F; 40]` with a generic type:** Rust does not allow generic const arrays. Keep the table as `[f64; 40]` and cast at the host boundary.
- **Calling `f64::erf()` on host expecting C `erf` linkage:** The existing `erf_host()` function in `boys.rs` uses `unsafe extern "C" { fn erf(x: f64) -> f64; }`. For f32 host path, use `num_traits::Float::asin()` etc., or just use `f32::from(erf_host(f64::from(x)))` — acceptable for host-only test code.
- **Using `F::new(val)` for precision-sensitive const tables:** `F::new()` takes `f32` which loses significant bits for `f64` constants. Use the `from_f64_lossy` pattern in `CintFloat` instead.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Host-side float math for `_host()` functions | Custom trait impls | `num_traits::Float` | Already transitive dep; covers exp, sqrt, from_f64, zero, one |
| Generic complex output types | `(F, F)` tuple pair | `num_complex::Complex<F>` | Already in workspace; generic by design; spinor layout already uses it |
| Erf approximation in `#[cube]` | Improve existing `boys_erf_approx` | Keep as-is for f64; use `F::erf(x)` intrinsic (Erf is on Float trait) | CubeCL 0.10.0's `Erf` is implemented for both f32 and f64; the intrinsic uses backend-native erf for both |
| f32 tolerance floors | Empirical guess | Run oracle at f32 and measure per-family max rel error | f32 ULP errors depend on input distribution; must be measured against libcint reference |

**Key insight:** CubeCL 0.10.0 provides a complete transcendental function set for both `f32` and `f64` — `exp`, `ln`, `sqrt`, `erf`, `sin`, `cos`, and more. No custom math approximations are needed for the generic kernel path.

---

## RESEARCH FLAG RESOLUTION: CubeCL Float Trait Surface

**Confirmed [VERIFIED: cubecl-core-0.10.0/src/frontend/element/float.rs + unary.rs]:**

The `cubecl::prelude::Float` trait in version 0.10.0 is a supertrait over:
- `Exp` — natural exponentiation (`exp`) — implemented for `f32` (manual impl, unary.rs:63) AND `f64` (macro)
- `Log` — natural log (`ln`) — implemented for both `f32` and `f64`
- `Sqrt` — square root — implemented for both `f32` and `f64`
- `Erf` — error function — implemented for both `f32` and `f64`

All transcendentals the Boys/Rys/Obara-Saika kernels use are on the `Float` trait for both concrete types. The kernel genericization is mechanically straightforward.

**`Exp` for f32 — the commented-out macro entry:**
The `impl_unary_func!` macro call has `// f32` commented out (confirmed in unary.rs), but a **separate manual `impl Exp for f32 {}`** block exists at line 63. The blanket `impl<T: Exp + CubePrimitive> ExpExpand for NativeExpand<T>` then applies to `f32` since `f32: Exp` and `f32: CubePrimitive`. Thus `f32::exp(x)` inside `#[cube]` compiles and emits via `Arithmetic::Exp → wgsl: exp(x)` (a WGSL built-in for f32). [VERIFIED: cubecl-core-0.10.0/src/frontend/operation/unary.rs]

**Const-table casting soundness:**
`TURNOVER_POINT: [f64; 40]` values have 12 significant digits. Casting to `f32` (`x as f32`) introduces ~1e-7 relative error in the threshold. Since TURNOVER_POINT is a branch threshold (not a numerically critical accumulator), this cast is sound: an off-by-epsilon branch switch between power-series and erfc branches produces equivalent results at f32 precision. `SQRTPIE4` similarly: the f32 representation is accurate to ~7 digits, matching the f32 result precision. [ASSUMED — empirical confirmation recommended]

---

## RESEARCH FLAG RESOLUTION: wgpu f32 Shader Capability

**Confirmed [VERIFIED: wgpu docs.rs/wgpu/29.0.3, cubecl-wgpu-0.10.0/src/backend/wgsl.rs]:**

- **f32 is WebGPU-baseline universal.** The wgsl backend registers `FloatKind::F32` with `TypeUsage::all()` unconditionally (no feature flag). This means any wgpu adapter — including those lacking `SHADER_F64` — supports f32 compute shaders.
- **SHADER_F64 is native-only and Vulkan-specific.** It is explicitly marked "native only" in wgpu docs. Many adapters (integrated GPUs, mobile GPUs, WebGPU backends) do not support it.
- **The f32 path genuinely unlocks non-SHADER_F64 adapters.** The existing `check_shader_f64_in_features` / `check_f64_capability` logic in `executor.rs` only gates f64 execution. The f32 path must bypass this check entirely.
- **The exact bypass:** In `executor.rs`, `check_f64_capability()` must be guarded by a `PrecisionKind` (or equivalent) check that skips the SHADER_F64 test when executing f32 kernels.

---

## f64-Site Surface Map (Crate-by-Crate)

### Crates to GENERICIZE (compute path)

| Crate | ~f64 Lines | Key Modules | Action |
|-------|-----------|-------------|--------|
| `cintx-cubecl` | 2,521 | `math/boys.rs`, `math/rys.rs`, `math/obara_saika.rs`, `math/stg.rs`, `kernels/*.rs`, `transform/c2s.rs`, `transform/c2spinor.rs` | Full generic `F: Float` threading |
| `cintx-runtime` | 23 | `dispatch.rs` (`staging_output: &mut [f64]`), `planner.rs` (`try_alloc_staging`, `staging_elements_for_chunk`), `dispatch.rs` `ExecutionIo` | Generic staging allocation + bytemuck cast boundary |
| `cintx-rs` | 19 | `api.rs` (`evaluate()`, `TypedEvaluationOutput`, `IntegralTensor::owned_values`) | Method-level generic + generic output struct |

**Breakdown within `cintx-cubecl` by file:**

| File | f64 Lines | Genericize? | Notes |
|------|-----------|------------|-------|
| `math/rys.rs` | 1,140 | YES | Largest file; Horner polynomials, Clenshaw; all f64 literals become F |
| `kernels/unstable.rs` | 231 | YES | unstable-source families |
| `transform/c2spinor.rs` | 204 | PARTIAL | CG coefficients are pure rational numbers compiled from f64 tables; generic over F for the output accumulation |
| `math/radial_quadrature.rs` | 121 | PARTIAL/HOST-ONLY | ECP-specific; mostly host-only; may leave f64 |
| `kernels/f12.rs` | 118 | YES | STG/YP kernel |
| `kernels/ecp.rs` | 112 | DEFER/HOST | ECP uses host-only K-Taylor tables; can stay f64 for Phase 20 |
| `math/ecp_k_taylor.rs` | 66 | DEFER | Host-only ECP math; leave f64 |
| `math/bessel.rs` | 51 | DEFER | ECP-specific host-only; leave f64 |
| `math/boys.rs` | 42 | YES | Core math; pattern established |
| `kernels/one_electron.rs` | 41 | YES | |
| `kernels/center_4c1e.rs` | 38 | YES | |
| `transform/c2s.rs` | 37 | YES | c2s coefficients are rational |
| `kernels/two_electron.rs` | 36 | YES | |
| `executor.rs` | 35 | PARTIAL | Capability branch + staging cast |
| `math/stg.rs` | 34 | YES | F12/STG math |
| `kernels/center_3c1e.rs` | 34 | YES | |
| `kernels/center_3c2e.rs` | 33 | YES | |
| `math/pdata.rs` | 32 | YES | Primitive pair data |
| `kernels/center_2c2e.rs` | 29 | YES | |
| `math/roots_xw_data.rs` | ~5 | PARTIAL | Binary table stays f64; consumed by STG kernels via parameter |

**ECP exception:** `math/ecp_k_taylor.rs`, `math/bessel.rs`, `math/radial_quadrature.rs`, `kernels/ecp.rs` are ECP-specific host-only paths. The K-Taylor tables (`ecp_k_taylor_in.bin`, `ecp_k_taylor_order7.bin`) are f64 binary blobs with a drift-gate. Genericizing ECP for f32 in Phase 20 is optional — f32 ECP results at the oracle level are undefined by PySCF (reference is f64-only). Recommendation: leave ECP as f64-only for Phase 20, document as a future extension.

### Crates to FREEZE (ABI boundary — stay f64)

| Crate | f64 Lines | Reason to Freeze |
|-------|-----------|-----------------|
| `cintx-compat` | 101 | `env`/`atm`/`bas` raw ABI: `eval_raw()`, `raw.rs`, `transform.rs` — D-06 locked |
| `cintx-capi` | 5 | C ABI shim — D-07 locked |
| `cintx-core` | 50 | `Shell.exponents: Arc<[f64]>`, `Atom.coord_bohr: [f64; 3]` — domain types; stay f64; only output is parameterized |
| `cintx-oracle` | 281 | Oracle operates on `f64` reference values from libcint; f32 gate compares `f32` cintx output vs. `f64` libcint output, converting for comparison |
| `cintx-ops` | 0 | No f64 in manifest/resolver |

---

## Recommended Sealed Super-Trait: CintFloat

**Decision: YES, introduce it.** Rationale:

1. Host-side `_host()` functions (e.g. `boys_gamma_inc_host`, `boys_gamma_inc_impl`) need `exp`, `sqrt`, basic arithmetic — these come from `num_traits::Float`, not CubeCL's `Float`.
2. Without a sealed trait, the public API signature `evaluate::<F>()` would expose `where F: cubecl::prelude::Float + num_traits::Float` — a leaky bound that locks callers into both trait namespaces and makes future expansion (or backend changes) harder.
3. The sealed trait enforces that only `f32` and `f64` monomorphize — no accidental f16/bf16 instantiation.
4. `num-traits` 0.2.19 is already a transitive dep (confirmed in Cargo.lock).

**Placement:** `cintx-core/src/precision.rs` (exported from `cintx-core` crate root). Both `cintx-cubecl` and `cintx-rs` already depend on `cintx-core`.

**Bounds sketch:**

```rust
// cintx-core/src/precision.rs
mod sealed { pub trait Sealed {} }

pub trait CintFloat:
    Copy + Send + Sync + 'static
    + num_traits::Float
    + num_traits::FromPrimitive
    + sealed::Sealed
{
    fn from_f64_lossy(x: f64) -> Self;
}

impl sealed::Sealed for f64 {}
impl CintFloat for f64 { fn from_f64_lossy(x: f64) -> Self { x } }

impl sealed::Sealed for f32 {}
impl CintFloat for f32 { fn from_f64_lossy(x: f64) -> Self { x as f32 } }
```

**Device kernel bound stays as `F: cubecl::prelude::Float`** — CubeCL's Float is required for `#[cube]` expansion machinery. The `CintFloat` bound is on the public API (`evaluate::<F>`) and host-side wrappers.

---

## Per-Family f32 Tolerance Floor Strategy

**Mirror of Phase 15 model but at single-precision tolerance.**

### Approach
Phase 15 unified all families at `UNIFIED_ATOL = 1e-12`. For f32, the theoretical floor is ~1e-7 relative (f32 epsilon = 1.2e-7). Accumulated error through multiple polynomial evaluations, recurrences, and Rys quadrature will inflate this. The expected floor across integral families is ~1e-4 to 1e-6 rtol.

**Empirical derivation procedure:**
1. Add `compare_f32_profile_parity()` alongside `compare_profile_parity()` in `cintx-oracle/src/compare.rs`.
2. Run `evaluate::<f32>()` for all oracle fixture symbols; collect `f32` output.
3. Compare against `f64` libcint reference: `abs(f32_out as f64 - libcint_ref) / abs(libcint_ref)`.
4. Record `max_rel_error` per family.
5. Set per-family `f32_rtol = max_rel_error * 10.0` (10x safety margin) rounded up to a round number.

**Expected floors** [ASSUMED — must be empirically measured]:
- `1e`, `2c2e`: ~1e-5 to 1e-6 (simple polynomial)
- `2e`, `3c1e`, `3c2e`: ~1e-4 (Rys quadrature accumulates more error)
- `4c1e`, `f12`: ~1e-4 (same Rys quadrature, double-precision reference)

**Oracle structure:**
```rust
// cintx-oracle/src/compare.rs — new section
pub const F32_UNIFIED_RTOL: f64 = 1e-4;  // initial conservative floor; tighten per family

pub struct F32FamilyTolerance {
    pub family: &'static str,
    pub rtol: f64,  // relative tolerance vs. f64 libcint reference
    pub atol: f64,  // absolute tolerance for near-zero elements
}

pub fn f32_tolerance_for_family(family: &str) -> F32FamilyTolerance {
    // Start with catch-all, tighten per family after empirical run
    F32FamilyTolerance { family: ..., rtol: F32_UNIFIED_RTOL, atol: 1e-7 }
}
```

**Separate oracle gate:** The f32 oracle gate runs as a parallel CI job (not replacing the f64 gate), using a dedicated test file `cintx-oracle/tests/f32_parity.rs`. The existing four-profile f64 gate is unchanged.

---

## Serena-Driven Refactor Sequencing

**D-11 requires serena MCP tools.** The executor must call `check_onboarding_performed` / `initial_instructions` first.

### Rationale for Bottom-Up Order

With ~3,607 f64 sites across the codebase, the safest strategy is bottom-up: genericize the leaves first (math functions used by kernels), then the kernels themselves, then the staging/dispatch layer, then the public API. At each wave boundary, `<f64>` is threaded explicitly so the f64 path stays green.

### Refactor Wave Order

**Wave 0: Scaffolding (no behavior change)**
1. Define `CintFloat` sealed trait in `cintx-core/src/precision.rs`.
2. Add `PrecisionKind { F64, F32 }` enum to `cintx-runtime`.
3. Add `precision: PrecisionKind` field to `ExecutionPlan` (default `F64`).
4. Add `num-traits` as direct dep of `cintx-core` (already transitive, just make explicit).
5. CI: `cargo check --all-features` green.

**Wave 1: Shared Math Leaves** (cintx-cubecl/src/math/)
Genericize in order (each depends on the previous):
1. `boys.rs` — `boys_gamma_inc::<F>`, `boys_erf_approx::<F>`, `boys_gamma_inc_host::<F>`
2. `obara_saika.rs` — VRR/HRR steps
3. `rys.rs` — `rys_root1::<F>` through `rys_root5::<F>`, `clenshaw_d1::<F>`
4. `stg.rs` — F12/STG math
5. `pdata.rs` — primitive pair data

Each function: serena `find_symbol` → `replace_symbol_body` to add `<F: Float>` generic + replace `f64` type annotations + replace `f64::` calls with `F::` calls. Verify: `cargo test -p cintx-cubecl --features cpu` with `CINTX_BACKEND=cpu` (uses `<f64>` paths). Math host tests call `_host::<f64>()`.

**Wave 2: Kernel Launchers** (cintx-cubecl/src/kernels/)
Genericize each kernel launcher: `launch_1e::<F>()`, `launch_2e::<F>()`, `launch_2c2e::<F>()`, `launch_3c1e::<F>()`, `launch_3c2e::<F>()`, `launch_4c1e::<F>()`, `launch_f12::<F>()`.

Each launcher: serena `find_symbol` → `replace_symbol_body` to add `F: Float` + generic array allocations + call genericized math functions with `<F>`.

**Wave 3: Executor + Staging** (cintx-cubecl/src/executor.rs, cintx-runtime/src/)
1. Add `check_capability()` replacing `check_f64_capability()` with `PrecisionKind` branch.
2. Add bytemuck cast in kernel launch path: `staging_output()` stays `&mut [f64]`; inside `launch_family`, cast to `&mut [F]` via `bytemuck::cast_slice_mut`.
3. `planner.rs::try_alloc_staging` stays `Vec<f64>` but allocates `size_of::<F>() * elements` bytes (needs `element_size` from `PrecisionKind`).

**Wave 4: Safe API Surface** (cintx-rs/src/api.rs)
1. `evaluate::<F>()` method-level generic.
2. `TypedEvaluationOutput<F = f64>` with default.
3. `IntegralTensor::owned_values: Vec<F>`.
4. `evaluate()` delegates to `evaluate::<f64>()`.
5. Spinor path: `Complex<F>` — `num-complex` is already generic.

**Wave 5: f32 Oracle Gate** (cintx-oracle/)
1. Add `F32FamilyTolerance` and `f32_tolerance_for_family()` in `compare.rs`.
2. Add `f32_parity.rs` test file.
3. Empirically measure per-family rtol floors.
4. Add CI job `oracle_parity_gate_f32` (advisory initially; promote to required after empirical floors stabilize).

### Serena Tool Usage per Wave

At each symbol:
```
1. find_symbol("boys_gamma_inc") → locate definition
2. find_referencing_symbols("boys_gamma_inc") → map call sites
3. replace_symbol_body("boys_gamma_inc", "<new generic body>")
4. For each call site: replace_symbol_body or insert_before/after_symbol to thread <F>
```

**Critical:** Deliberately-f64 sites (env/atm/bas in `cintx-compat`, `f64` in `compare.rs` reference buffers, ECP tables) must be identified with `find_symbol` and skipped. The serena tool's symbol-level granularity prevents accidentally modifying frozen sites — unlike grep+sed.

---

## Common Pitfalls

### Pitfall 1: `Exp` for f32 — the commented-out macro line
**What goes wrong:** Developer sees `// f32,` commented out in `impl_unary_func!(Exp, ...)` and concludes `f32::exp()` does not work in `#[cube]`. Removes the `Exp` supertrait bound from `F` or adds a polyfill.
**Why it happens:** Misleading code — the manual `impl Exp for f32` block below the macro provides the implementation.
**How to avoid:** The manual `impl Exp for f32` at unary.rs:63 satisfies `f32: Exp`. The blanket `impl<T: Exp + CubePrimitive> ExpExpand for NativeExpand<T>` handles the expansion. Do NOT add a custom exp polyfill.
**Warning signs:** Compile error "the trait `Exp` is not implemented for `f32`" — this should NOT happen; if it does, check that `cubecl::prelude::*` is imported.

### Pitfall 2: CubeCL prelude shadowing in test modules
**What goes wrong:** Tests in `cintx-cubecl` that `use cubecl::prelude::*;` cannot call `f64::exp(x)` as a standard method — it resolves to the CubeCL Cube intrinsic and panics with "Unexpanded Cube functions should not be called".
**Why it happens:** Phase 19 D-02 (REVISED): CubeCL prelude shadows host f64 methods inside `#[cfg(test)]` modules.
**How to avoid:** Use precomputed reference literals in tests, or compute references outside the prelude scope. The established pattern: test bodies do NOT call transcendentals inside cubecl-prelude-imported scopes.
**Warning signs:** Panic at test runtime with "Unexpanded Cube functions should not be called".

### Pitfall 3: Making BackendExecutor generic on F breaks object safety
**What goes wrong:** Threading `<F>` into `BackendExecutor` trait method `execute::<F>()` makes it non-object-safe. `&dyn BackendExecutor` in `planner::evaluate()` then fails to compile.
**Why it happens:** Generic methods on trait objects violate object safety rules.
**How to avoid:** Use `PrecisionKind` enum dispatch internally. The kernel launcher calls a precision-dispatched function internally rather than making the trait generic.
**Warning signs:** Compile error "the trait `BackendExecutor` cannot be made into an object".

### Pitfall 4: Corrupting env/atm/bas f64 sites with serena bulk replace
**What goes wrong:** A `find_referencing_symbols("f64")` call in serena accidentally includes env/atm/bas array types in `cintx-compat/src/raw.rs`.
**Why it happens:** Over-broad symbol search.
**How to avoid:** Scope serena queries to specific symbols (function names, type names) not to primitive type names. Identify frozen crates explicitly before each wave.
**Warning signs:** `cargo check -p cintx-compat` fails after the refactor with type errors.

### Pitfall 5: SQRTPIE4 / PIE4 precision loss in generic cast
**What goes wrong:** `SQRTPIE4 as F` in the `#[cube]` kernel loses 45 ULPs of f64 precision when instantiated as f32. This is acceptable for f32 but must not be passed to f64 kernels via a lossy intermediate.
**Why it happens:** `F::new(0.886...)` takes an `f32` literal, losing bits.
**How to avoid:** Pass constants as `F` parameters from the host wrapper (Phase 8 pattern). The host wrapper accesses the `f64` const and converts via `CintFloat::from_f64_lossy(SQRTPIE4)`.
**Warning signs:** f64 oracle parity regressions in Boys function after genericization.

### Pitfall 6: ECP binary tables misidentified as compute-path f64
**What goes wrong:** `ecp_k_taylor_in.bin` and `ecp_k_taylor_order7.bin` are incorrectly included in the genericization scope, causing ECP to emit f32 results at incorrect accuracy.
**Why it happens:** ECP K-Taylor tables are byte-locked f64 blobs; ECP math is host-only.
**How to avoid:** ECP paths stay f64-only in Phase 20 (explicit exception). Mark `kernels/ecp.rs`, `math/ecp_k_taylor.rs`, `math/bessel.rs`, `math/radial_quadrature.rs` as FROZEN.
**Warning signs:** ECP oracle parity tests fail at atol=1e-12 after Phase 20.

### Pitfall 7: Monomorphization bloat and duplicate kernel compilation
**What goes wrong:** Both `<f64>` and `<f32>` kernel monomorphizations are compiled into the artifact, doubling compile time for the kernel-heavy `cintx-cubecl` crate (~1,683-line rys.rs alone).
**Why it happens:** Generic monomorphization is eager in Rust.
**How to avoid:** Accept the compile-time cost (it's a one-time cost, not a hot path). Use CI build time monitoring. The CubeCL JIT kernel cache handles the device side separately per precision.
**Warning signs:** `cargo build` time doubles for `cintx-cubecl`; this is expected.

---

## Code Examples

### Boys Function — Generic `#[cube]` Pattern

```rust
// Verified pattern: cubecl-core-0.10.0/src/frontend/element/float.rs shows
// Float: Exp + Sqrt + Erf + Log + ... — all operations available as F::op(x)

use cubecl::prelude::*;

// The SQRTPIE4 constant is passed as a parameter from the host
// (per Phase 8 D-01: "Pass TURNOVER_POINT[m] as scalar parameter")
#[cube]
pub fn boys_gamma_inc<F: Float>(
    f: &mut Array<F>,
    t: F,
    m: u32,
    turnover: F,
    sqrtpie4: F,    // host passes F::from_f64_lossy(SQRTPIE4)
) {
    if t == F::new(0.0) {
        f[0usize] = F::new(1.0);
        let mut k: u32 = 1;
        while k <= m {
            f[k as usize] = F::new(1.0) / F::new((2u32 * k + 1u32) as f32);
            k += 1;
        }
    } else if t < turnover {
        let b = m as F + F::new(0.5);
        let e = F::new(0.5) * F::exp(-t);  // F::exp works for both f32 and f64
        // ...
    } else {
        let tt = F::sqrt(t);
        let erf_val = F::erf(tt);           // F::erf — on Float trait for both f32/f64
        f[0usize] = erf_val * (sqrtpie4 / tt);
        let e = F::exp(-t);
        // ...
    }
}
```

### Executor Capability Bypass for f32

```rust
// cintx-cubecl/src/executor.rs
fn check_capability(
    &self,
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
) -> Result<(), cintxRsError> {
    // f32 is WebGPU-baseline — no SHADER_F64 check needed
    if plan.precision == PrecisionKind::F32 {
        return Ok(());
    }
    // f64 path: existing SHADER_F64 gate
    self.check_f64_capability(backend, plan)
}
```

### Safe API Method-Level Generic

```rust
// cintx-rs/src/api.rs (schematic)
impl<'basis> SessionQuery<'basis> {
    /// Evaluate integrals in precision F (default: f64, byte-identity).
    pub fn evaluate<F: CintFloat>(self) -> Result<TypedEvaluationOutput<F>, FacadeError>
    where
        // CintFloat seals to f32 | f64 only
    {
        // ... existing evaluate logic, with staging cast to &mut [F] via bytemuck
    }

    /// Evaluate at f64 precision — existing behavior, unchanged.
    pub fn evaluate(self) -> Result<TypedEvaluationOutput<f64>, FacadeError> {
        // [ASSUMED] Default: could also be a blanket impl if needed
        self.evaluate::<f64>()
    }
}

// TypedEvaluationOutput becomes:
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TypedEvaluationOutput<F = f64> {
    pub tensor: IntegralTensor<F>,
    pub stats: EvaluationStats,
    pub workspace_bytes: usize,
    pub chunk_count: usize,
    pub bytes_written: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct IntegralTensor<F = f64> {
    pub extents: Vec<usize>,
    pub component_axis_leading: bool,
    pub complex_interleaved: bool,
    pub owned_values: Vec<F>,
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| f64-only CubeCL kernels (07-CONTEXT.md D-09) | Generic `F: Float` with f64 default | Phase 20 | Unlocks f32 path on non-SHADER_F64 GPUs |
| Phase 15 per-family atol=1e-12 (f64 unified) | f64 gate: unchanged; new f32 gate at ~1e-4 rtol | Phase 20 | Parallel oracle profile for single-precision |
| `check_shader_f64_in_features` for all wgpu eval | Bypassed for f32; retained for f64 | Phase 20 | f32 path is universally available |

**Deprecated in Phase 20:**
- `TypedEvaluationOutput.tensor.owned_values: Vec<f64>` — becomes `Vec<F>` with `F = f64` default (SemVer compatible)
- `boys_gamma_inc(f: &mut Array<f64>, ...)` → `boys_gamma_inc::<F>(f: &mut Array<F>, ...)`

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Casting `TURNOVER_POINT[m]` thresholds from f64 to f32 is sound (acceptable branch-selection error) | Pattern 2 | Off-by-one branch in Boys function for f32; compensated by the ~1e-4 rtol oracle tolerance; low risk |
| A2 | ECP K-Taylor computation is host-only and can be left as f64-only in Phase 20 | f64-Site Map | If ECP is exercised via `evaluate::<f32>()`, it will implicitly use f64 K-Taylor and then cast down — may need boundary cast layer |
| A3 | f32 per-family rtol floors will be ~1e-4 (expected); initial oracle gate set at 1e-4 | f32 Tolerance Floor | If actual error exceeds 1e-4 for some family, the oracle gate fails and floors must be re-derived |
| A4 | `impl Exp for f32 {}` (manual block) + blanket `ExpExpand for NativeExpand<T>` together make `f32::exp(x)` callable inside `#[cube]` | CubeCL Float research | If the manual impl doesn't wire `ExpExpand`, `f32::exp` fails to compile in `#[cube]` — verify with `cargo check` before Wave 1 |
| A5 | Bytemuck cast `cast_slice_mut::<f64, f32>` (or vice versa) across the staging buffer boundary is zero-cost and sound | Pattern 5 | If bytemuck rejects the cast (alignment or size mismatch), staging must be separate Vec<F> — increases runtime overhead slightly |
| A6 | `CintFloat` sealed trait placed in `cintx-core` doesn't create circular dep issues | CintFloat section | If `cintx-cubecl` needs to re-export `CintFloat`, the public bound may need to move |

---

## Open Questions (RESOLVED)

> All three questions are answered inline below (see each **Recommendation:**) and the
> chosen answers are implemented in the Phase 20 plans: Q1 → 20-04/05/06 (`PrecisionKind`
> enum field + internal `execute()` match), Q2 → 20-04 T2 (genericize `c2spinor.rs`
> accumulation with an f32-rtol docs warning), Q3 → 20-05 T2 (`f12_zeta` stays
> `Option<f64>`, cast at the kernel boundary).

1. **`BackendExecutor` trait generics vs `PrecisionKind` dispatch**
   - What we know: `BackendExecutor` is used as `&dyn BackendExecutor` (non-generic); making it generic breaks object safety.
   - What's unclear: Whether to make the executor's `execute()` method take a `PrecisionKind` arg and dispatch internally (enum match), or whether to use two separate `execute_f64()` / `execute_f32()` methods.
   - Recommendation: `PrecisionKind` enum field on `ExecutionPlan` + internal match in `execute()`. The bytemuck cast approach avoids the need for two method variants.

2. **`c2spinor.rs` (204 lines) — genericize or freeze?**
   - What we know: CG coefficients are rational numbers (exact in f64). The spinor transform accumulation is where f32 error would manifest.
   - What's unclear: Whether f32 spinor integrals are useful to any caller (spinor integrals are already precision-sensitive).
   - Recommendation: Genericize the accumulation but test carefully; emit a warning in docs that spinor f32 results have ~1e-4 rtol.

3. **`f12_zeta: Option<f64>` in `OperatorEnvParams`**
   - What we know: The STG/YP kernel uses `f12_zeta` which is f64 in `ExecutionOptions`. F12 zeta is a physics parameter; the integral values will be in `F`.
   - What's unclear: Whether `f12_zeta` should become `Option<F>` or stay `Option<f64>` (cast at boundary).
   - Recommendation: Stay `Option<f64>`; cast at the kernel boundary (D-06 pattern: env stays f64).

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain (stable 1.94.0) | Phase compilation | ✓ | pinned per CLAUDE.md | — |
| `cargo nextest` | Test execution | Not checked | — | `cargo test` |
| serena MCP server | D-11 refactor (required) | In environment (listed in MCP context) | — | D-11 forbids fallback to blind text replace |
| CubeCL 0.10.0 (wgpu feature) | f32 shader execution | ✓ (pinned) | 0.10.0 | CPU backend for oracle |
| wgpu-capable GPU | f32 path integration test | Unknown | — | CPU backend for oracle; f32 path tested on CPU |

---

## Validation Architecture

Nyquist validation is ENABLED (config.json `workflow.nyquist_validation: true`).

### Test Framework

| Property | Value |
|----------|-------|
| Framework | `cargo test` + `cargo nextest` (if available) |
| Config file | `rust-toolchain.toml` (pinned 1.94.0) |
| Quick run command | `CINTX_BACKEND=cpu cargo test -p cintx-cubecl --features cpu` |
| Full suite command | `CINTX_BACKEND=cpu cargo test --workspace --features cpu` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PREC-01 | `#[cube] fn ...<F: Float>` compiles for both f32 and f64 | Unit compile check | `cargo check -p cintx-cubecl --features cpu` | ❌ Wave 1 |
| PREC-01 | `boys_gamma_inc::<f32>()` host result finite and non-zero | Unit | `cargo test -p cintx-cubecl boys_f32 --features cpu` | ❌ Wave 1 |
| PREC-02 | `evaluate::<f64>()` byte-identical to existing oracle | Unit regression | `cargo test -p cintx-oracle --features cpu` | ✅ existing |
| PREC-02 | `evaluate()` (unparameterized) compiles unchanged | Compile smoke | `cargo check -p cintx-rs` | ✅ existing |
| PREC-02 | `evaluate::<f32>()` compiles and returns `Vec<f32>` | Unit | `cargo test -p cintx-rs f32_evaluate --features cpu` | ❌ Wave 4 |
| PREC-03 | `eval_raw()` still accepts `&[f64]` env/atm/bas unchanged | Regression | `cargo test -p cintx-compat --features cpu` | ✅ existing |
| PREC-04 | All four f64 oracle profiles pass at atol=1e-12 | Integration | `cargo test -p cintx-oracle --features cpu` | ✅ existing |
| PREC-05 | f32 oracle gate passes at ~1e-4 rtol for all families | Integration | `cargo test -p cintx-oracle f32_parity --features cpu` | ❌ Wave 5 |
| PREC-06 | `evaluate::<f32>()` succeeds on CPU backend without SHADER_F64 | Smoke | `CINTX_BACKEND=cpu cargo test f32_smoke` | ❌ Wave 3 |
| PREC-07 | Serena `check_onboarding_performed` passes at wave start | Process gate | Serena MCP call | N/A |

### Sampling Rate

- **Per task commit:** `CINTX_BACKEND=cpu cargo check --workspace --features cpu`
- **Per wave merge:** `CINTX_BACKEND=cpu cargo test --workspace --features cpu 2>&1 | tail -20`
- **Phase gate:** Full f64 oracle suite (four profiles) green; f32 oracle gate advisory-green

### Wave 0 Gaps

- [ ] `cintx-core/src/precision.rs` — `CintFloat` sealed trait + `PrecisionKind` enum
- [ ] `cintx-runtime/src/precision.rs` or inline — `PrecisionKind` field on `ExecutionPlan`
- [ ] Confirm: `num-traits` direct dep in `cintx-core/Cargo.toml` (currently transitive only)
- [ ] Serena onboarding check: call `check_onboarding_performed` before Wave 1

---

## Security Domain

`security_enforcement` not explicitly set to `false` in config.json — section included.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | Phase is internal type refactor |
| V3 Session Management | No | No session state changes |
| V4 Access Control | No | No access model changes |
| V5 Input Validation | Partial | `CintFloat::from_f64_lossy` cast must not accept NaN/Inf from env arrays; existing validation unchanged |
| V6 Cryptography | No | No cryptographic operations |

### Known Threat Patterns for Generic Float Parameterization

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| f32 result silently treated as f64 by caller | Tampering | `TypedEvaluationOutput<F>` type system prevents confusion; `evaluate::<f32>()` returns `Vec<f32>` not `Vec<f64>` |
| Buffer aliasing via `bytemuck::cast_slice_mut` | Elevation of privilege | `bytemuck::Pod` bound is required; both `f32` and `f64` are Pod; cast is zero-copy and safe |
| f32 oracle false positive (too-loose tolerance) | Information disclosure | Empirical tolerance derivation with 10x safety margin; advisory gate initially |

---

## Sources

### Primary (HIGH confidence)

- `cubecl-core-0.10.0/src/frontend/element/float.rs` — `Float` trait definition confirming `Exp`, `Sqrt`, `Erf` supertraits [VERIFIED: installed registry source]
- `cubecl-core-0.10.0/src/frontend/operation/unary.rs` — `impl_unary_func!` calls confirming f32/f64 support per op; manual `impl Exp for f32` at line 63 [VERIFIED: installed registry source]
- `cubecl-core-0.10.0/src/frontend/polyfills.rs` — `erf<F: Float>` polyfill using `Float::exp` [VERIFIED: installed registry source]
- `cubecl-wgpu-0.10.0/src/backend/wgsl.rs` lines 70-90 — unconditional f32 TypeUsage registration; SHADER_F64 gating only for f64 [VERIFIED: installed registry source]
- `wgpu 29.0.3` Features docs — SHADER_F64 is native-only; f32 is WebGPU-baseline [VERIFIED: docs.rs/wgpu/29.0.3]
- `Cargo.lock` — cubecl 0.10.0 checksum; num-traits 0.2.19; num-complex 0.4.6 [VERIFIED: project Cargo.lock]
- `cintx-cubecl/src/executor.rs` — `check_shader_f64_in_features`, `check_f64_capability` [VERIFIED: project source]
- `cintx-cubecl/src/math/boys.rs` — `TURNOVER_POINT: [f64; 40]`, `boys_erf_approx` pattern [VERIFIED: project source]
- `cintx-rs/src/api.rs:125,501` — `evaluate()`, `TypedEvaluationOutput`, `IntegralTensor::owned_values: Vec<f64>` [VERIFIED: project source]
- `20-CONTEXT.md` D-01..D-12 — milestone-level decisions [VERIFIED: project planning artifact]

### Secondary (MEDIUM confidence)

- Phase 8 D-01 (STATE.md accumulated context) — "Pass TURNOVER_POINT[m] as scalar parameter to avoid runtime const array indexing" — established pattern for const table injection [CITED: .planning/STATE.md]
- Phase 19 D-02 (REVISED) (STATE.md) — "CubeCL prelude shadows host f64 methods inside `#[cfg(test)]` modules" [CITED: .planning/STATE.md]

### Tertiary (LOW confidence — verify before acting)

- Expected f32 rtol floors (~1e-4 to 1e-6 per family) — training-data estimate; must be empirically derived [ASSUMED]
- `bytemuck::cast_slice_mut` applicability for the staging buffer cast — requires `F: Pod`; both f32 and f64 are Pod; functional but not tested in this codebase [ASSUMED]

---

## Metadata

**Confidence breakdown:**
- CubeCL Float trait surface: HIGH — verified against installed crate source
- wgpu f32 universality: HIGH — verified against wgpu 29.0.3 docs
- f64-site categorization by crate: HIGH — grep counts from project source
- Architecture patterns / refactor sequencing: MEDIUM — design reasoning, not tested
- f32 tolerance floors: LOW — must be measured empirically

**Research date:** 2026-05-20
**Valid until:** 2026-06-20 (30 days; CubeCL 0.10.0 is pinned so Float trait surface is stable)
