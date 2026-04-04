# Technology Stack

**Project:** cintx
**Researched:** 2026-04-04 (v1.2 update — F12/STG/YP kernels, spinor algebra, 4c1e extended, atol=1e-12)
**Prior research:** 2026-04-02 (v1.1 — CubeCL direct client API, multi-backend, real kernels, oracle parity)

---

## v1.2 Stack Additions and Changes

This section covers only what is NEW for v1.2. The v1.0 and v1.1 baseline stacks below remain valid and unchanged.

### Summary: No New External Crates Required

The v1.2 milestone adds no new crates to the workspace. All new math (STG roots, YP correlation factor, spinor Clebsch-Gordan coefficients) follows the same in-house `#[cube]` / host-side port strategy used in v1.1 for Boys, Rys, and Obara-Saika. One transitive dependency needs promotion to a direct dependency (`num-complex`), and one math module needs new sub-modules. Everything else is code changes within existing crates.

---

### F12/STG/YP Kernel Math — No New Crates, New Math Sub-Modules

**Finding (HIGH confidence — from `libcint-master/src/` direct inspection):**

The three F12 correlation factor variants used in the with-f12 family are:

| Variant | Operator | Correlation factor | Root-finding strategy |
|---------|----------|-------------------|----------------------|
| STG | `int2e_stg` | Slater-type geminal `exp(-zeta*r12)` | `CINTstg_roots` via Clenshaw/DCT over precomputed `roots_xw.dat` table |
| YP | `int2e_yp` | Yukawa potential `exp(-zeta*r12)/r12` | `CINTstg_roots` with same table; modified integral weight normalization |
| F12 | `int2e_f12` | `exp(-zeta*r12^2)` (Gaussian geminal) | Standard Rys quadrature (already implemented) |

**STG/YP root algorithm** (`libcint-master/src/stg_roots.c`):
- `CINTstg_roots(nroots, ta, ua, rr, ww)` evaluates Clenshaw recurrence over a 2D Chebyshev expansion grid indexed by `(ta, ua)`.
- The grid data lives in `roots_xw.dat` (binary table of doubles — 19600 entries per `nroots*(nroots-1)/2` block, up to 14 roots).
- `ta` and `ua` are the exponential/length arguments computed from the ij/kl Gaussian exponents and the STG exponent `zeta` (`PTR_F12_ZETA` = env[9]).
- The algorithm uses: `_clenshaw_dc` (1D Chebyshev), `_matmul_14_14` (cos-DCT basis change with `COS_14_14` constant table), `_clenshaw_d1` (Chebyshev evaluation of the result coefficients).

**What this means for cintx-cubecl:**

Add a new `math/stg.rs` module in `cintx-cubecl` that ports `CINTstg_roots`. This is a host-side function (like `rys_roots_host` already is). The Clenshaw evaluation is pure floating-point arithmetic with no external dependencies. The `COS_14_14` constant array (14x14 cosine values) and the approach of embedding root-table lookups are already established by the `TURNOVER_POINT` pattern in `boys.rs`.

**STG root table (`roots_xw.dat`):**

The `stg_roots.c` file includes `roots_xw.dat` via `#include "roots_xw.dat"` — the table is embedded as static C data, not a file read at runtime. The port to Rust must embed the same coefficients as a Rust `static` array. The table is large (libcint ships it as a generated file). The implementation approach is:

1. Copy the `DATA_X` and `DATA_W` table contents from the compiled libcint vendored source.
2. Embed as `static` `[f64; N]` arrays in `math/stg.rs`.
3. Index with the same offset formula: `nroots * 196 * (iu + it * 10)`.

**`PTR_F12_ZETA` parameter plumbing:**

`zeta` = `env[PTR_F12_ZETA]` = `env[9]`. The `ExecutionPlan` does not currently carry operator-specific float parameters. The planner already reads from the `BasisSet`'s env array for nuclear charges. The cleanest approach is:

- Add an `operator_env_params: Option<OperatorEnvParams>` field to `ExecutionPlan` (or a new `F12KernelParams` struct).
- Populate from `env[PTR_F12_ZETA]` during `ExecutionPlan::new` when the operator family is `with-f12`.
- Pass into the kernel launch function alongside the existing plan.

No new crate is needed. This is a struct extension in `cintx-runtime`.

---

### Spinor Algebra Infrastructure — Promote `num-complex` to Direct Dep, Replace Stub

**Finding (HIGH confidence — from `libcint-master/src/cart2sph.c` lines 809-4978 and `cintx-compat/src/transform.rs`):**

The current spinor transform in `cintx-cubecl/src/transform/c2spinor.rs` is a stub. It performs:
```
amplitude = (pair[0].abs() + pair[1].abs()) * 0.5
```
This is obviously wrong for oracle parity — it averages magnitudes instead of applying the 2j-spinor Clebsch-Gordan transformation.

**What libcint actually does (`cart2sph.c`):**

Spinor output requires `CINTc2s_ket_spinor_sf1` / `CINTc2s_ket_spinor_si1`. These functions apply:
1. A Clebsch-Gordan coupling from Cartesian GTO components to 2-spinor components `(alpha, beta)`.
2. Separate `real` and `imaginary` coefficient tables for the `j = l+1/2` (kappa < 0) and `j = l-1/2` (kappa > 0) cases.
3. The coefficient tables are extracted from `cart2sph.c` lines 809-2172 (real part, `c2s_cart2spinor_r`) and lines 2174-3535 (imaginary part, `c2s_cart2spinor_i`).

The spinor output buffer layout is interleaved complex doubles: `[re0, im0, re1, im1, ...]` for each spinor component. This interleaved layout is already correctly identified in the existing stub and in `cintx-core/src/tensor.rs` (`complex_interleaved: bool`).

**`num-complex` as direct dependency:**

`num-complex 0.4.6` is already in `Cargo.lock` as a transitive dependency (pulled in via `cubecl`). For the spinor transform and typed complex output APIs, promote it to a direct dependency in:
- `cintx-core` — for `Complex<f64>` in output layout types and the public safe API spinor output views.
- `cintx-cubecl` — for staging buffer construction when spinor output is requested.

No feature flags needed for `num-complex 0.4.6` — the base crate provides `Complex<f64>` and basic arithmetic. Do NOT enable the `"serde"` feature unless serialization of complex values becomes an explicit requirement.

**Clebsch-Gordan coefficient tables:**

Same strategy as `cart2sph.c` for `c2s.rs`: extract the coefficient tables for `j = l+1/2` (sf) and `j = l-1/2` (si) cases from `cart2sph.c` and embed them as Rust `static` arrays in a new `transform/c2spinor_coeffs.rs` module. The tables are large but finite — libcint ships them for l=0..8 (sufficient for all standard GTOs up to k-functions).

**What needs to be rewritten:**

| File | Change |
|------|--------|
| `crates/cintx-cubecl/src/transform/c2spinor.rs` | Replace placeholder `cart_to_spinor_interleaved_staging` with correct Clebsch-Gordan matrix apply per (l, kappa) |
| `crates/cintx-cubecl/src/transform/` | Add `c2spinor_coeffs.rs` with real/imaginary coefficient tables |
| `crates/cintx-compat/src/transform.rs` | Fix `CINTc2s_ket_spinor_sf1`, `CINTc2s_ket_spinor_si1`, etc. to pass (l, kappa) through to the kernel |
| `crates/cintx-core/Cargo.toml` | Add `num-complex = "0.4.6"` as direct dep |
| `crates/cintx-cubecl/Cargo.toml` | Add `num-complex = "0.4.6"` as direct dep |

**`kappa` parameter threading:**

The existing `CINTc2s_ket_spinor_sf1` signature already accepts `kappa: i32` — it just ignores it. The real implementation needs `kappa` to select which coefficient table to use. No API signature change is required, only internal implementation.

---

### 4c1e Extended Envelope — No New Math Deps, Kernel Completion Required

**Finding (HIGH confidence — from `libcint-master/src/g4c1e.c` and `crates/cintx-cubecl/src/kernels/center_4c1e.rs`):**

The `center_4c1e.rs` kernel is a stub that returns zeros and returns an error for anything outside a narrow validated envelope. The 4c1e integral uses Rys quadrature (same `rys_roots_host` already in `math/rys.rs`) and the same Obara-Saika recurrence as the 2e kernel. No new math primitives are needed — Rys and Obara-Saika already exist.

The tolerance gap (`TOL_4C1E_ATOL = 1e-6` vs target `1e-12`) is not a precision problem inherent to f64 — it reflects the stub returning zeros. Once the real kernel produces correct outputs, the tighter tolerance is achievable.

**Dependency delta for 4c1e:** None. Use existing `rys_roots_host`, `compute_pdata_host`, and Obara-Saika recurrences from `math/`.

---

### Oracle Tolerance Unification to atol=1e-12 — Code Change Only

**Finding (HIGH confidence — from `crates/cintx-oracle/src/compare.rs` direct inspection):**

Current per-family tolerances:

| Family | Current atol | Current rtol | Target atol |
|--------|-------------|-------------|-------------|
| 1e | 1e-11 | 1e-9 | 1e-12 |
| 2e | 1e-12 | 1e-10 | 1e-12 (already at target) |
| 2c2e / 3c2e | 1e-9 | 1e-7 | 1e-12 |
| 3c1e | 1e-7 | 1e-5 | 1e-12 |
| 4c1e | 1e-6 | 1e-5 | 1e-12 |
| with-f12 (new) | not yet registered | — | 1e-12 |

The tolerance constants are in `compare.rs` lines 21-31. Moving them all to `1e-12` atol is a pure constant change — no new crates, no interface change. The constraint from PROJECT.md is explicit: "atol=1e-12 for ALL families (immutable unless explicitly approved spec update)."

The families for which tighter tolerance currently fails (3c1e atol=1e-7, 4c1e atol=1e-6) will fail oracle gates until the corresponding kernels are correctly implemented. The tolerance change and the kernel completion must be coordinated — tighten the tolerance constants only after the kernel produces correct f64 outputs, or accept failing CI gates as the signal that the kernel is incomplete.

**Dependency delta for tolerance unification:** None. Code-only change in `compare.rs`.

---

### New Oracle Coverage: with-f12 and Full 4c1e Family

**Finding (HIGH confidence — from `compare.rs` `tolerance_for_family` match + api_manifest):**

The `tolerance_for_family` match in `compare.rs` currently has no arm for `"with-f12"` or `"f12"`. Adding F12/STG/YP oracle coverage requires:

1. Add `"f12"` (or `"with-f12"`) arm to `tolerance_for_family` returning atol=1e-12.
2. Extend oracle fixtures to include molecule+basis sets that exercise STG/YP kernels.
3. Extend `IMPLEMENTED_TRANSFORM_SYMBOLS` if new transform variants are added.

No new crates. All within `cintx-oracle`.

---

### Summary Dependency Table for v1.2

| Crate | Change | Why |
|-------|--------|-----|
| `cintx-core` | Add `num-complex = "0.4.6"` as direct dep | Typed complex output for spinor safe API |
| `cintx-cubecl` | Add `num-complex = "0.4.6"` as direct dep | Staging buffer for spinor interleaved output |
| `cintx-cubecl` | Add `math/stg.rs` sub-module | STG/YP root evaluation (Clenshaw/DCT port) |
| `cintx-cubecl` | Add `transform/c2spinor_coeffs.rs` sub-module | Real/imaginary Clebsch-Gordan tables |
| `cintx-runtime` | Extend `ExecutionPlan` with `operator_env_params` | Carry `PTR_F12_ZETA` to kernel launch |
| `cintx-oracle` | Add `"f12"` arm to `tolerance_for_family` | Oracle coverage for with-f12 family |
| `cintx-oracle` | Update tolerance constants toward 1e-12 | Unify atol across all families |

**No new workspace members. No new crate additions to `Cargo.toml`.**

---

## Alternatives Considered (v1.2)

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| In-house STG root port (embed `roots_xw.dat` as static arrays) | External libslater/GSL Bessel quadrature crate | No production Rust crate exists; embedding is the same strategy used for Rys roots. The source (libslater port in libcint) is already vendored. |
| `num-complex 0.4.6` (already in lockfile) | Raw interleaved `[f64]` throughout | `num-complex` gives zero-cost `Complex<f64>` with clean arithmetic; avoids index arithmetic bugs in the spinor output path. Already in lockfile — promotion costs nothing. |
| In-house Clebsch-Gordan tables ported from `cart2sph.c` | External quantum-chem CG coefficient crate | No mature Rust crate exists for the specific normalized 2j-spinor coefficients libcint uses; the tables are finite and already in the vendored source. |
| Tight atol=1e-12 with CPU backend for oracle parity | Looser per-family tolerances indefinitely | PROJECT.md states the target tolerance is immutable; looser tolerances mask real kernel precision gaps. |
| `operator_env_params` struct field on `ExecutionPlan` | Pass zeta as a separate argument to every kernel fn | The plan struct is already the single context object passed to kernel launchers; adding a field maintains the existing calling convention cleanly. |

---

## What NOT to Add (v1.2)

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Any external numeric/algebraic crate for STG roots or spinor CG | None exists with the right specialized behavior; all would add build complexity without matching libcint's exact algorithm | In-house ports from vendored `libcint-master/src/` |
| `ndarray` or `nalgebra` for the CG coefficient matrix multiply | The coefficient tables are small, fixed-size, and known at compile time; a general matrix library adds compile time and complexity for no benefit | Hardcoded `static` arrays + manual matrix-vector multiply (same as `c2s.rs` pattern) |
| `num-complex` `"serde"` feature | Not needed for kernel math or oracle comparison | Leave feature set at default (base `Complex<f64>` arithmetic only) |
| Promoting `cubecl/cuda` or `cubecl/hip` to active deps | Hardware not validated; premature | Keep as placeholder feature flags only, not enabled in CI |
| Bumping `cubecl` from 0.9.x | No correctness or API blocker identified | Pin to 0.9.0 until a specific upstream fix requires upgrading |

---

## v1.1 Stack (Unchanged, for Reference)

### CubeCL Client API in cintx-cubecl

Direct client API pattern confirmed from project reference files:
- `client.create(bytes)` / `client.empty(size_bytes)` / `client.read(vec![handle.binding()])`
- `ArrayArg::from_raw_parts::<T>(&handle, len, 1)` (unsafe, vectorization factor = 1)
- All helpers called from `#[cube]` must themselves be `#[cube]`; plain Rust helper calls compile with error E0433
- CPU backend via `cubecl/cpu` feature for oracle parity CI without GPU

### New Dep Added in v1.1

| Library | Version | Crate | Why |
|---------|---------|-------|-----|
| `bytemuck` | `1` | `cintx-cubecl` | `client.create()` requires `&[u8]`; `cast_slice` converts `&[f64]` safely |

---

## v1.0 Baseline Stack (Unchanged)

### Core Platform

| Technology | Version guidance | Purpose | Why recommended |
|------------|------------------|---------|-----------------|
| Rust toolchain | Pin `1.94.0` in `rust-toolchain.toml` | Reproducible compiler behavior | Pinning keeps oracle and manifest results reproducible. |
| Cargo lockfile | Commit `Cargo.lock`; run CI with `cargo --locked` | Deterministic dependency graph | Oracle comparisons are only credible with the same resolved graph. |
| Cargo resolver | `resolver = "3"` (edition-2024 default) | Predictable feature resolution | Resolver 3 is the 2024-edition default. |
| Multi-crate workspace | `core`, `ops`, `runtime`, `cubecl`, `compat`, `capi`, `oracle`, `xtask` | Isolate domain types, execution, compat, verification, tooling | Hard architectural boundaries between API, compat contracts, backend execution, and release gating. |

### Core Libraries

| Library | Version guidance | Purpose | Notes |
|---------|------------------|---------|-------|
| `cubecl` | `0.9.0` (keep locked) | GPU+CPU compute backend | Backend-agnostic public API. |
| `cubecl-wgpu` | `0.9.0` | wgpu backend | Direct dep in `cintx-cubecl`. |
| `cubecl-runtime` | `0.9.0` | Runtime traits | Direct dep in `cintx-cubecl`. |
| `thiserror` | `2.0.18` | Public typed errors | Library-facing error enums. |
| `anyhow` | `1.0.102` | App-boundary errors | xtask, benchmarks, oracle tooling. |
| `tracing` | `0.1.x` | Structured spans | Planner, chunking, transfer, fallback. |
| `bindgen` | `0.71.1` | Oracle binding generation | Upgrade deliberately. |
| `cc` | `1.2.x` | Vendored libcint build | Oracle harness hermetic. |
| `wgpu` | `26.0.1` | Capability snapshot | `runtime_bootstrap.rs`. |
| `smallvec` | `1.13` | Small fixed collections | Control-plane hot paths. |
| `approx` | `0.5.1` | Float equality in tests | Dev dep in `cintx-cubecl`. |
| `criterion` | `0.8.2` | Benchmark harness | Dev dep in root crate. |

---

## Sources

### Verified (HIGH confidence — code inspection)
- STG root algorithm: `libcint-master/src/stg_roots.c` — Clenshaw/DCT over `roots_xw.dat`, no external dep
- F12 kernel structure: `libcint-master/src/g2e_f12.c` — shares Rys quadrature with 2e; adds `PTR_F12_ZETA = env[9]`
- Spinor transform stub: `crates/cintx-cubecl/src/transform/c2spinor.rs` — amplitude averaging is not a correct CG transform
- Real spinor CG tables: `libcint-master/src/cart2sph.c` lines 809-3535 — separate real/imaginary tables for sf (j=l+1/2) and si (j=l-1/2) variants
- `num-complex 0.4.6` already in `Cargo.lock`: local evidence
- Current oracle tolerances: `crates/cintx-oracle/src/compare.rs` lines 21-31 — per-family values documented above
- 4c1e stub: `crates/cintx-cubecl/src/kernels/center_4c1e.rs` — returns zeros, no new math needed
- No `"f12"` arm in `tolerance_for_family`: `compare.rs` match expression
- `PTR_F12_ZETA = 9`: `libcint-master/include/cint.h.in` line 40

### Verified (MEDIUM confidence — ecosystem survey)
- No external Rust crate for STG roots, YP correlation factor evaluation, or 2j-spinor CG coefficients exists on crates.io as of 2026-04-04

---
*Stack research for: cintx v1.2 — F12/STG/YP kernels, spinor CG algebra, 4c1e completion, atol=1e-12 unification*
*Researched: 2026-04-04*
