# Technology Stack

**Project:** cintx
**Researched:** 2026-04-02 (v1.1 update — CubeCL direct client API, multi-backend, real kernels, oracle parity)
**Prior research:** 2026-03-21 (v1.0 foundational stack — preserved below)

---

## v1.1 Stack Additions and Changes

This section covers only what is NEW for v1.1. The v1.0 baseline stack below remains valid and unchanged.

### CubeCL Client API — What Changes in cintx-cubecl

The executor rewrite removes `RecordingExecutor` and replaces it with direct CubeCL client API calls. The client API pattern is stable in 0.9.0 and confirmed across multiple project reference files (`docs/manual/Cubecl/Cubecl_vector.md`, `docs/manual/Cubecl/Cubecl_shared_memory.md`, `docs/manual/Cubecl/Cubecl_multi_compute.md`).

**Confirmed wgpu pattern (HIGH confidence — from project reference files):**

```rust
use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

// Kernel definition
#[cube(launch)]
fn my_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let tid = ABSOLUTE_POS;
    if tid < input.len() {
        output[tid] = input[tid] * 2.0;
    }
}

// Launch site
let device = WgpuDevice::default();
let client = WgpuRuntime::client(&device);

let input_buffer  = client.create(bytemuck::cast_slice(&input_data));
let output_buffer = client.empty(output_size_bytes);

my_kernel::launch::<WgpuRuntime>(
    &client,
    CubeCount::Static(1, 1, 1),
    CubeDim::new(n as u32, 1, 1),
    unsafe { ArrayArg::from_raw_parts::<f32>(&input_buffer, n, 1) },
    unsafe { ArrayArg::from_raw_parts::<f32>(&output_buffer, n, 1) },
);

let result_bytes = client.read(vec![output_buffer.binding()]);
let result: &[f32] = bytemuck::cast_slice(&result_bytes[0]);
```

Key points:
- `client.create(bytes)` — uploads host bytes to a device buffer handle.
- `client.empty(size_bytes)` — allocates an uninitialized device buffer.
- `client.read(vec![handle.binding()])` — reads back; returns `Vec<Vec<u8>>`, one entry per binding.
- `ArrayArg::from_raw_parts::<T>(&handle, len, vectorization_factor)` — must be `unsafe`.
- `#[cube(launch)]` — the standard macro; `launch_unchecked` variant skips bounds checking at the cost of safety.
- The third argument to `ArrayArg::from_raw_parts` is the vectorization factor — use `1` for scalar access.

**Alternate initialization pattern for CPU backend (HIGH confidence — from `docs/manual/Cubecl/cubecl_reduce_sum.md`):**

```rust
use cubecl_cpu::{CpuDevice, CpuRuntime, RuntimeOptions};
use cubecl_runtime::client::ComputeClient;

let device = CpuDevice::default();
let client: ComputeClient<_> = ComputeClient::load(&device);
// Same create/empty/read API as wgpu client
```

Both `WgpuRuntime::client(&device)` and `ComputeClient::load(&device)` are valid initialization paths. The `Runtime::client(&device)` form is idiomatic for the wgpu path; `ComputeClient::load` is the lower-level form used in the reduce and matmul examples. Either works — prefer `Runtime::client(&device)` for consistency with the project's existing `runtime_bootstrap.rs` pattern.

### Critical CubeCL Kernel Constraint: No Plain Rust Calls Inside `#[cube]`

**This is the most important implementation constraint for kernel code (HIGH confidence — from `docs/manual/Cubecl/cubecl_error_solution_guide/`):**

Functions called from inside a `#[cube]` function MUST themselves be annotated with `#[cube]`. Calling a plain Rust function from inside a `#[cube]` function produces a compile error:

```
failed to resolve: function my_fn is not a crate or module (E0433)
```

This means:
- The Boys function evaluation inside kernels must be written as `#[cube]` functions.
- Helper functions for Rys quadrature, Obara-Saika recurrences, and coordinate transformations must all carry `#[cube]` (or `#[cube(expand)]` for pure computation).
- Standard library math functions (`f64::sqrt`, `f64::erf`, etc.) are NOT callable inside `#[cube]`. Use CubeCL's built-in Float primitives instead.
- `libm`, `std::f64`, and any plain-Rust math helpers are host-side only.

**Correct pattern for helper functions:**

```rust
// Wrong: will fail with E0433
fn boys_f0(t: f32) -> f32 { ... }

#[cube(launch)]
fn kernel(t: &Array<f32>, out: &mut Array<f32>) {
    out[ABSOLUTE_POS] = boys_f0(t[ABSOLUTE_POS]);  // ERROR
}

// Correct: helper must be #[cube]
#[cube]
fn boys_f0(t: f32) -> f32 {
    let sqrt_t = f32::sqrt(t);  // CubeCL built-in, not std::f32::sqrt
    f32::erf(sqrt_t) / (2.0_f32 * sqrt_t)
}

#[cube(launch)]
fn kernel(t: &Array<f32>, out: &mut Array<f32>) {
    out[ABSOLUTE_POS] = boys_f0(t[ABSOLUTE_POS]);  // OK
}
```

Consequence: every Boys function variant, every Rys root/weight lookup, and every recurrence step needs to be a `#[cube]` function. This shapes the entire kernel module structure.

### New Dependency: `bytemuck` in `cintx-cubecl`

`bytemuck` is required for `bytemuck::cast_slice` to convert `&[f64]` / `&[f32]` host slices to `&[u8]` for `client.create()` and back from `client.read()` output.

| Library | Version | Scope | Why |
|---------|---------|-------|-----|
| `bytemuck` | `1.25.0` (already in `Cargo.lock` as transitive dep from cubecl/wgpu) | `cintx-cubecl` direct dependency | `client.create()` requires `&[u8]`; `bytemuck::cast_slice` is the safe, zero-copy conversion. No new crate resolution needed — just promote to direct dep. |

**Add to `crates/cintx-cubecl/Cargo.toml`:**
```toml
bytemuck = "1"
```

No additional features needed for `f32`/`f64` slices — both already implement `bytemuck::Pod`.

### CubeCL CPU Backend — Enable for Testing

The CPU backend (`cubecl-cpu`) is already present in `Cargo.lock` as a transitive dependency of `cubecl = "0.9.0"`. It is NOT currently enabled as a feature in `cintx-cubecl`.

**Feature activation (HIGH confidence — verified via `crates.io/api/v1/crates/cubecl/0.9.0` feature list):**

The `cubecl` crate exposes a `"cpu"` feature flag that activates `cubecl-cpu`. Enabling it gives access to `CpuRuntime` and `CpuDevice`:

```rust
use cubecl::cpu::{CpuDevice, CpuRuntime};

let device = CpuDevice::default();
let client = CpuRuntime::client(&device);
// Same client API as WgpuRuntime — client.create(), client.empty(), client.read()
```

**API contract (HIGH confidence):**
- `CpuRuntime` is a zero-sized marker struct implementing the `Runtime` trait.
- `CpuRuntime::client(&device)` returns `ComputeClient<CpuRuntime>`.
- The `ComputeClient` API is identical between wgpu and cpu backends — same `create`, `empty`, `read` methods.
- The same `#[cube(launch)]` kernel compiles and runs on both backends with no kernel code changes.

**Add to `crates/cintx-cubecl/Cargo.toml` features:**
```toml
[features]
default = []
with-f12 = []
with-4c1e = []
cpu = ["cubecl/cpu"]
```

Use the `cpu` feature in `dev-dependencies` tests and `#[cfg(test)]` blocks where a real GPU is not required. CI without a GPU adapter can gate oracle parity tests under `--features cpu`.

**Why CPU backend, not a fake stub:** The CPU backend runs the same `#[cube(launch)]` kernels as wgpu. This means kernel correctness tests (oracle parity, numerical accuracy) do not require a physical GPU — the CPU backend produces the same numerical outputs. This is the primary testing path for kernel development and oracle comparison.

### Multi-Backend Runtime Configuration

The v1.1 goal is configurable backend switching. The approach is generic dispatch over `R: Runtime`.

**Pattern (MEDIUM confidence — derived from CubeCL examples and existing `runtime_bootstrap.rs`):**

The existing `BackendIntent` / `BackendKind` types in `cintx-runtime` already provide a selector string. The rewritten executor should:

1. Branch on `BackendKind` to select `WgpuRuntime` vs `CpuRuntime`.
2. Pass the selected client to a generic kernel launch helper parameterized on `R: Runtime`.
3. Do NOT expose `WgpuRuntime` or `CpuRuntime` as public types — keep them behind the `BackendExecutor` trait boundary that already exists in `cintx-runtime`.

No new crate additions are needed. The dispatcher stays within `cintx-cubecl`. Future backends (CUDA, ROCm, Metal) are added by extending the `BackendKind` enum and branching in the dispatcher.

**Dependency change:** Add `cubecl/cuda` and `cubecl/hip` feature flags to the `cintx-cubecl` feature table as placeholders, but do NOT enable them in CI until hardware is available. Do not add `cubecl-cuda` or `cubecl-hip` as direct `[dependencies]` — use the feature flag forwarding pattern only.

### Integral Kernel Math: Boys Function and Rys Quadrature

**Finding: No viable standalone Rust crate exists for production Boys function / Rys quadrature evaluation (HIGH confidence).**

Investigation:
- `boys` crate on crates.io: depends on GNU Scientific Library via `rgsl` — a non-trivial C dependency. NOT acceptable: adds a GSL build dependency to the kernel path, contradicts the workspace's vendored-build-hermetic strategy, and the crate is experimental (version 0.1.0, no production usage evidence).
- `gauss-quad`: general numerical quadrature for real-valued functions. Not specialized for molecular integrals, and cannot replace Boys function or Rys quadrature.
- No Rys quadrature Rust crate found on crates.io.
- The upstream `libcint-master/src/fmt.c` (Boys function / incomplete gamma), `rys_roots.c`, and `rys_wheeler.c` are the reference implementations vendored in the project already.

**Conclusion: Implement Boys function and Rys quadrature in-house within `cintx-cubecl` kernel modules.**

Required math primitives and their approach:

| Math Primitive | Implementation Approach | CubeCL Constraint |
|---|---|---|
| Boys function `F_0(T)` for small T | `erf(sqrt(T)) / (2*sqrt(T))` — uses CubeCL built-in Float | Must be `#[cube]` fn; use `f32::erf` / `f32::sqrt` |
| Boys function `F_n(T)` recurrence | Upward recurrence from `F_0`; downward from large-T asymptotic | Must be `#[cube]` fn — no plain Rust helpers allowed |
| Boys function large-T asymptotic | `(2n-1)!! / 2^(n+1) * sqrt(pi/T^(2n+1))` | Must be `#[cube]` fn |
| Rys roots/weights (degree ≤ 4) | Precomputed polynomial fits (from `libcint-master/src/polyfits.c`) | Coefficient arrays as kernel `comptime` or passed as `Array` |
| Obara-Saika recurrence (1e overlap/kinetic) | Pure arithmetic recurrence — reference: `g1e.c` in vendored source | Must be `#[cube]` fn |
| Nuclear attraction auxiliary integrals | Boys function + Hermite Coulomb integrals | As above |

**Do not add `libm` as a direct `cintx-cubecl` kernel dependency.** `libm` is host-side only. CubeCL kernels execute on GPU/CPU via the `#[cube]` IR and must use CubeCL's built-in Float methods (`f32::erf`, `f32::sqrt`, etc.).

Use `libm` only in CPU-side test helpers, oracle cross-check code, and `dev-dependencies`. It is already in `Cargo.lock` version `0.2.16` as a transitive dep — promote to direct `dev-dependency` if test helpers need it.

### f64 Precision in CubeCL Kernels

**Important constraint (MEDIUM confidence — requires hardware validation):**

CubeCL's wgpu backend maps to WebGPU/WGSL. The `SHADER_F64` wgpu feature (double precision in compute shaders) is optional and not universally supported across hardware. The existing `runtime_bootstrap.rs` already collects `SHADER_F64` in `collect_feature_names()`.

Consequences:
1. Before dispatching f64 kernels via wgpu, the executor must check `SHADER_F64` in the capability snapshot.
2. For oracle parity (which requires f64 to match libcint's `double` outputs), run oracle comparison under the CPU backend (`--features cpu`), not wgpu. The CPU backend compiles kernels to native code with full f64.
3. wgpu f64 is a "bonus" precision path — do not make oracle parity dependent on it.

**Recommended approach for oracle parity tests:** Use `#[cfg(feature = "cpu")]` to gate oracle comparison tests. Do not require a GPU for oracle parity CI.

---

## v1.0 Baseline Stack (Unchanged)

### Core Platform

| Technology | Version guidance | Purpose | Why recommended |
|------------|------------------|---------|-----------------|
| Rust toolchain | Pin `1.94.0` in `rust-toolchain.toml` | Reproducible compiler behavior across local dev and CI | Rust 1.94.0 is the current stable release as of 2026-03-05, and pinning an exact toolchain keeps oracle and manifest results reproducible. |
| Cargo lockfile | Commit `Cargo.lock`; run CI with `cargo --locked` | Deterministic dependency graph | Oracle comparisons and manifest audits are only credible if every runner uses the same resolved graph. |
| Cargo resolver | Use edition-2024 default `resolver = "3"` | Predictable feature resolution in a multi-crate workspace | Resolver 3 is the 2024-edition default. |
| Multi-crate workspace | `core`, `ops`, `runtime`, `cubecl`, `compat`, `capi`, `oracle`, `xtask` | Isolate domain types, execution, compat, verification, and tooling | Hard architectural boundaries between typed API, compat contracts, backend execution, and release gating. |

### Core Libraries

| Library | Version guidance | Purpose | Notes |
|---------|------------------|---------|-------|
| `cubecl` | `0.9.0` (keep locked) | Shared GPU+CPU compute backend | Keep public API backend-agnostic. |
| `cubecl-wgpu` | `0.9.0` | wgpu backend for cubecl | Direct dep in `cintx-cubecl`. |
| `cubecl-runtime` | `0.9.0` | Runtime traits | Direct dep in `cintx-cubecl`. |
| `thiserror` | `2.0.18` | Public typed error surface | Library-facing error enums. |
| `anyhow` | `1.0.102` | App-boundary errors | xtask, benchmarks, oracle tooling. |
| `tracing` | `0.1.x` | Structured spans and diagnostics | Planner, chunking, transfer, fallback. |
| `bindgen` | `0.71.1` | Oracle/header binding generation | Upgrade deliberately. |
| `cc` | `1.2.x` | Vendored upstream libcint build | Oracle harness hermetic. |
| `wgpu` | `26.0.1` | Capability snapshot and adapter enumeration | Used in `runtime_bootstrap.rs`. |

### Supporting Libraries

| Library | Use | Why |
|---------|-----|-----|
| `smallvec` | Small fixed-ish collections | Cuts heap churn in hot control-plane paths. |
| `num-complex` | Safe API complex/spinor outputs | In `Cargo.lock` 0.4.6 as transitive dep. |
| `approx`, `proptest`, `criterion` | Verification and benchmarking | Oracle comparison, property testing, perf baselines. |

---

## Complete Dependency Delta for v1.1

Changes needed to `crates/cintx-cubecl/Cargo.toml`:

```toml
[features]
default = []
with-f12 = []
with-4c1e = []
cpu = ["cubecl/cpu"]           # NEW: enables CpuRuntime for test-time kernel execution
cuda = ["cubecl/cuda"]         # NEW: placeholder, not enabled in CI
hip = ["cubecl/hip"]           # NEW: placeholder, not enabled in CI

[dependencies]
# ... existing deps unchanged ...
bytemuck = "1"                 # NEW: cast_slice for client.create()/client.read()
```

No new crates need to be added to the workspace `Cargo.toml`.

No integral math crates from crates.io. All Boys function / Rys quadrature math is implemented in-house in `cintx-cubecl` kernel modules as `#[cube]` functions.

---

## Alternatives Considered (v1.1)

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| In-house Boys function as `#[cube]` fns using CubeCL built-in Float | `boys` crate (crates.io) | Depends on GNU Scientific Library via `rgsl`; experimental (0.1.0); incompatible with `#[cube]` IR constraint. |
| `bytemuck::cast_slice` for buffer I/O | Manual `unsafe` pointer casting | `bytemuck` is already in the lockfile, provides safe Pod trait bounds, and is the pattern shown in all CubeCL examples. |
| CPU backend via `cubecl/cpu` feature for oracle parity tests | Mock/stub backend | CPU backend runs real `#[cube]` kernels, so oracle parity results are trustworthy rather than being tested against a fake. |
| f64 oracle parity via CPU backend | f64 via wgpu `SHADER_F64` | `SHADER_F64` is not universally supported; CPU backend provides f64 without hardware dependency. |
| All Boys/recurrence helpers as `#[cube]` fns | Plain Rust helpers called from `#[cube]` | CubeCL macro transformation disallows calling non-`#[cube]` functions from inside `#[cube]` (error E0433). |

---

## Sources

### Verified (HIGH confidence)
- CubeCL 0.9.0 feature list (`cpu`, `wgpu`, `cuda`, `hip` flags): `curl https://crates.io/api/v1/crates/cubecl/0.9.0` + feature map
- CubeCL client API pattern (`client.create`, `client.empty`, `client.read`, `ArrayArg::from_raw_parts`): `docs/manual/Cubecl/Cubecl_vector.md`, `docs/manual/Cubecl/Cubecl_shared_memory.md`, `docs/manual/Cubecl/Cubecl_multi_compute.md` (project reference files)
- `CpuRuntime`/`CpuDevice` client API: `docs/manual/Cubecl/cubecl_reduce_sum.md` (project reference file)
- `#[cube]` function constraint (no plain Rust calls): `docs/manual/Cubecl/cubecl_error_solution_guide/` (project reference file)
- `cubecl-cpu 0.9.0` in `Cargo.lock`: local evidence (`name = "cubecl-cpu"`, `version = "0.9.0"`)
- `bytemuck 1.25.0` in `Cargo.lock`: local evidence; crates.io API confirmed 1.25.0 as latest stable
- `libm 0.2.16` in `Cargo.lock`: local evidence (transitive dep)
- libcint Boys function reference (`fmt.c`, `rys_roots.c`, `rys_wheeler.c`): `libcint-master/src/` in workspace

### Verified (MEDIUM confidence)
- `CubeCL multi-backend generic dispatch pattern` `launch::<R: Runtime>`: verified in matmul example (`docs/manual/Cubecl/cubecl_matmul_gemm_example.md`) and multi-compute example
- f64 / `SHADER_F64` limitation in wgpu compute: inferred from `runtime_bootstrap.rs` `collect_feature_names()` collecting `SHADER_F64` and WebGPU spec; requires empirical validation on target hardware

### Not found (integral math crates)
- No production Rust crate for Boys function, Rys quadrature, or Obara-Saika recurrence exists on crates.io as of 2026-04-02. The `boys` crate at https://crates.io/crates/boys depends on GSL and is experimental. In-house implementation is the correct path.

---
*Stack research for: cintx v1.1 — CubeCL direct client API, multi-backend, real kernel compute, oracle parity*
*Researched: 2026-04-02*
