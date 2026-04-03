# Feature Landscape

**Domain:** CubeCL direct client API, real GPU integral kernels, oracle parity — cintx v1.1
**Researched:** 2026-04-02
**Scope:** Subsequent milestone — what is NEW in v1.1. The three-layer API surface, manifest-driven resolver,
planner/dispatcher, stub executor, staging/fingerprint propagation, and CI gates are all already built.

---

## Context: What Is Already Built (Do Not Reimplement)

The v1.0 deliverables are complete and locked behind passing CI:

- Three-layer API surface (safe Rust API, raw compat, C ABI shim)
- Manifest-driven resolver and feature-gated family registry
- Runtime planner with chunking, memory-limit policy, and typed OOM errors
- CubeCL executor scaffolding with `TransferPlan`, `SpecializationKey`,
  `DeviceResidentCache`, `bootstrap_wgpu_runtime` capability check
- All five family `launch_*` functions as **zero-producing stubs** — they call
  `transfer.stage_output_buffer()` which returns a zero-filled staging buffer
- Staging / fingerprint propagation wired end to end
- Oracle harness with `compare.rs` tolerance tables (already calibrated per family)
- CI governance gates that pass today by accepting zero outputs (oracle comparisons
  are not yet gated on numerical values)

v1.1 replaces the stub bodies with real math. Everything above is infrastructure
that the real kernels must plug into, not work to re-do.

---

## Table Stakes for v1.1

These are features that must ship for v1.1 to be considered complete.
Missing any one of them means "kernels produce zeros" remains the state.

### 1. CubeCL Direct Client API Rewrite

**What:** Replace `RecordingExecutor` (wrapper abstraction) with direct
`WgpuRuntime::client()` usage: `client.create()` for device buffer allocation,
`client.read_one()` for result readback, `ArrayArg::from_raw_parts()` for kernel
argument passing, and `#[cube(launch)]` / `launch_unchecked` for dispatch.

**Why required:** The current executor wrapper prevents real kernel control over
buffer lifetime, upload timing, and readback. Real kernels need to own their
GPU buffer handles, pass them directly as `ArrayArg`, and read back per-chunk.
This is the prerequisite gate for all real kernel work.

**Key mechanics (HIGH confidence — verified from CubeCL docs and examples):**
```rust
let client = WgpuRuntime::client(&device);
let input_gpu = client.create(f64::as_bytes(&host_data));
let output_gpu = client.create(f64::as_bytes(&zeros));
unsafe {
    my_kernel::launch::<WgpuRuntime>(
        &client,
        CubeCount::Static(grid_x, grid_y, 1),
        CubeDim::new(threads_x, threads_y, 1),
        ArrayArg::from_raw_parts::<f64>(&input_gpu, len, 1),
        ArrayArg::from_raw_parts::<f64>(&output_gpu, len, 1),
    )
}
let result = client.read_one(output_gpu);
```

**Dependencies:** Existing `TransferPlan`, `SpecializationKey`, `ExecutionStats`
structures. The `FamilyLaunchFn` signature stays the same; only the internals change.

**Complexity:** Medium. One surgical rewrite of `executor.rs` + `transfer.rs` internals.
No API surface change for callers.

### 2. Configurable Multi-Backend Support

**What:** Backend selection configurable at runtime: `wgpu` (GPU, default) and
`cpu` (CubeCL CPU runtime, for testing and CI without GPU). Extensible to
`cuda`, `rocm`, `metal` by adding a Runtime type parameter.

**Why required:** CI runners may not have GPU hardware. The CPU backend allows
oracle comparisons to run deterministically in CI without GPU. The design
document explicitly requires this as a v1.1 deliverable (PROJECT.md line 39).

**Key mechanics:** `DevicePreference` enum in `ExecutionOptions` already exists.
Planner's `backend_intent` field drives the bootstrap. Adding a CPU runtime path
means `bootstrap_cpu_runtime()` alongside the existing `bootstrap_wgpu_runtime()`,
with the same `BackendCapabilityReport` shape.

**Dependencies:** `runtime_bootstrap.rs`, `capability.rs`, existing
`ExecutionOptions.device` field.

**Complexity:** Low-Medium. Primarily wiring the CPU runtime bootstrap and making
kernel launch generic over `R: Runtime`.

### 3. One-Electron (1e) Real Kernel — Overlap and Kinetic Energy

**What:** Implement real GPU integral kernels for the 1e family covering the
overlap and kinetic energy operators (the two most common 1e integrals).

**Algorithm (MEDIUM confidence — multiple research sources agree):**
The overlap of two contracted Gaussian shells uses the Obara-Saika horizontal
recurrence relation (HRR) or the equivalent McMurchie-Davidson E-coefficient approach.
For s and p functions, closed-form expressions suffice. For higher angular momentum,
the recurrence expands products of Gaussians centered at different atoms.

The overlap primitive integral is:
```
S_ij = product over x,y,z of:
  E^0(i_x, j_x) * sqrt(pi/p)
  where p = alpha_i + alpha_j,
  E^0 is the McMurchie-Davidson E-coefficient at zero order
```

For kinetic energy:
```
T_ij = alpha_j*(2*l_j+3)*S(i,j) - 2*alpha_j^2*S(i,j+2) - 0.5*l_j*(l_j-1)*S(i,j-2)
```

**GPU parallelism:** One thread per shell pair (i,j). For a basis with N shells,
grid is N×N. Each thread loops over primitives in both shells.

**libcint compatibility note:** libcint uses the DRK (Dupuis-Rys-King) framework
internally, not Obara-Saika, but for cintx the algorithm does not need to match —
only the output values. For 1e integrals, any standard algorithm (OS, HRR, MD)
produces the same Cartesian integrals before the cart-to-sph transform.

**Cart-to-sph transform dependency:** After Cartesian primitives are accumulated,
a transformation matrix multiply is required for spherical representation. The
transform coefficients are fixed per angular momentum (from Schlegel-Frisch tables,
already present in `libcint-master/src/cart2sph.c`). These must be GPU-resident.

**Dependencies:** Real primitive Gaussian evaluation, prim-to-contraction
accumulation, staging buffer from `TransferPlan`, cart-to-sph transform matrix
in `DeviceResidentCache`.

**Complexity:** Medium. The math is well-characterized; the challenge is the
contraction loop and correct output indexing for all three representations
(cart, sph, spinor).

### 4. Two-Electron Repulsion Integral (2e) Real Kernel — ERI Core

**What:** Implement real GPU kernels for the 2e family: four-center two-electron
repulsion integrals (ERIs), the most computationally expensive family.

**Algorithm (HIGH confidence — multiple authoritative sources):**
libcint uses the DRK/Rys quadrature approach. For a GPU reimplementation,
three algorithms are viable; Rys quadrature is the preferred choice for GPU
due to its small memory footprint and data locality (confirmed by GPU4PySCF,
Rys paper, and BRUSH algorithm analysis).

Rys quadrature decomposes a 6D ERI into:
```
(ij|kl) = sum over N Rys roots t_n, weights w_n:
  product of 1D integrals in x, y, z at quadrature point t_n
  where N = floor((l_i + l_j + l_k + l_l)/2) + 1
```

Each 1D integral is computed via a simple recurrence on the Rys
polynomial at the given root. The Boys function `F_m(x)` provides the
base case for the Coulomb interaction; it can be computed via either
a precomputed table (gridded Taylor expansion) or the recursive downward
formula.

**Boys function on GPU (HIGH confidence):**
The gridded Taylor expansion approach (precomputed on CPU, uploaded once
to `DeviceResidentCache`) is the standard GPU strategy. For the input
argument range 0 <= x < X_max, a table of function values and derivatives
at equally spaced grid points enables fast GPU interpolation. For large x,
the asymptotic expansion applies directly. This avoids branch divergence
compared to conditional evaluation paths. (Tsuji 2025 GPU Boys function
paper confirms this pattern.)

**Angular momentum support:** For cart/sph with max(l) <= 4 (g functions),
the algorithm stays tractable on current GPU hardware. GPU4PySCF (g-function
support confirmed) and TeraChem (f-function) confirm this range. Above h
functions, register pressure becomes critical.

**GPU thread structure:**
- One CUDA/WGSL workgroup per shell quartet (i,j,k,l)
- Inner loop over primitive pairs (ip,jp,kp,lp)
- Each thread handles a subset of output Cartesian components

**libcint parity note:** The DRK recurrence relations in libcint and the Rys
quadrature approach produce analytically identical results for the same basis.
The oracle gate (TOL_2E_ATOL=1e-12, TOL_2E_RTOL=1e-10) reflects the expected
floating-point round-off from using double precision throughout.

**Dependencies:** Boys function table in `DeviceResidentCache`, Rys root/weight
tables, contraction loop, staging buffer, cart-to-sph transform.

**Complexity:** High. The 2e kernel is the most complex piece in this milestone.
The Rys root computation, Boys function table management, primitive loop nesting,
and four-index output ordering all interact. This is the kernel most likely to
require a dedicated research phase during implementation.

### 5. Two-Center Two-Electron Integral (2c2e) Real Kernel

**What:** Two-center ERIs (density fitting auxiliary integrals). Two-shell input,
electron-repulsion operator. Used extensively in resolution-of-identity (RI) methods.

**Algorithm:** The 2c2e is a simplified ERI where shells i and j coincide at bra
and ket: `(i|j) = (ii|jj)` with the middle integration removed. This reduces the
Rys quadrature root count and simplifies the 1D recurrence. The Boys function
argument simplifies to `x = p*q/(p+q) * R_ij^2` with only two primitive exponent
pairs.

**libcint parity tolerance:** Oracle uses TOL_2C2E_3C2E_ATOL=1e-9, TOL_2C2E_3C2E_RTOL=1e-7.
The looser tolerance versus 2e reflects accumulated float32 round-off in the
range-separated and attenuated variants.

**Complexity:** Medium-Low. The 2c2e shares the Boys function infrastructure with
2e, but with fewer indices and simpler recurrence. Implement after the 2e kernel
is working.

**Dependencies:** 2e kernel infrastructure (Boys function table, Rys roots).

### 6. Three-Center One-Electron Integral (3c1e) Real Kernel

**What:** Three-center integrals with one-electron nuclear attraction operators
(used in RI-MP2, COSX, and similar methods).

**Algorithm:** 3c1e integrals involve three shells and the nuclear attraction
operator `1/|r - R_C|`. This is a 1e nuclear attraction problem (Boys function
required via `F_m(x)` with three-center geometry) generalized to three shells.

The recurrence follows the standard 1e nuclear attraction path:
```
g_{000}^m = Boys_m(x) * prefactor
g_{ijk}^0 = [horizontal recurrence on angular momenta i, j, k]
```

**libcint parity tolerance:** Oracle uses TOL_3C1E_ATOL=1e-7, TOL_3C1E_RTOL=1e-5.
The loose tolerance (relative to 2e) reflects that the 3c1e integrals include
operators (like `nabla`, `r`, `r^2`) with larger absolute magnitudes.

**Complexity:** Medium. Shares Boys function infrastructure with 2e.
The three-center geometry adds a third shell loop and additional recurrence steps.

### 7. Three-Center Two-Electron Integral (3c2e) Real Kernel

**What:** Three-center ERIs (density fitting integrals of the form `(i|jk)`).
Critical for resolution-of-identity (RI-J, RI-K) approximations.

**Algorithm:** The 3c2e is an ERI with one bra auxiliary index and two ket
indices: `(P|ij)` where P is an auxiliary function. Rys quadrature applies with
N = floor((l_P + l_i + l_j)/2) + 1 roots.

Memory-efficient recursive evaluation for 3-center integrals is well-studied
(Barca et al. 2022 arxiv:2210.03192). The key saving over full 4-center ERIs is
that only three primitive loops are needed (not four), and the Rys root count
is smaller.

**libcint parity tolerance:** Same as 2c2e — TOL_2C2E_3C2E_ATOL=1e-9,
TOL_2C2E_3C2E_RTOL=1e-7.

**Complexity:** Medium-High. Builds on 2e kernel infrastructure.
More complex than 2c2e due to three-center geometry but simpler than full 4c ERI.

### 8. Gaussian Primitive Evaluation and Contraction Infrastructure

**What:** Shared GPU-side infrastructure for:
- Evaluating primitive Gaussian products: prefactor `exp(-alpha*r^2)` terms
- Computing the Gaussian product center P = (alpha_i * R_i + alpha_j * R_j) / (alpha_i + alpha_j)
- Pair data: `p = alpha_i + alpha_j`, `mu = alpha_i * alpha_j / p`, `R_ij`, `R_PQ`
- Contraction accumulation: summing over primitives with contraction coefficients
- Exponential cutoff screening: skip shell pairs where `pdata.cceij > expcutoff`

**Why a separate table stakes item:** Every kernel depends on this layer.
Getting the pair data computation, primitive loop, and contraction accumulation
correct and matching libcint's conventions (coefficient normalization, common_fac_sp
factors for s/p shells) is required before any individual kernel produces
correct numbers.

**libcint conventions (HIGH confidence — from source analysis):**
- `common_factor = CINTcommon_fac_sp(l_i) * CINTcommon_fac_sp(l_j) * operator_factor`
- Coefficients are pre-normalized in `env[]` by libcint's optimizer
- Non-zero coefficient index lists (`non0ctr`, `non0idx`) accelerate the
  contraction step for sparse contractions

**Complexity:** Medium. This is shared scaffolding that must be GPU-resident
(in `DeviceResidentCache` or passed as kernel arguments via ArrayArg).

### 9. Cart-to-Sph Transform on GPU

**What:** Post-kernel transformation from Cartesian output to real-spherical
harmonic output. Required by any caller requesting `Representation::Spheric`.

**Algorithm:** The transform is a fixed sparse matrix multiply:
`sph[m] = sum over cart components c: T[m,c] * cart[c]`

The transformation matrix coefficients are tabulated by angular momentum
(Schlegel-Frisch 1995, already hardcoded in `libcint-master/src/cart2sph.c`
as `g_trans_cart2sph[]`). For l=0,1 the transform is trivial (identity or
simple factors). For l=2 (d), l=3 (f), l=4 (g) the matrix has increasing
density.

**GPU strategy:** Upload the fixed transform matrix per angular momentum
into `DeviceResidentCache` at initialization. Apply as a device-side
matrix multiply after the Cartesian kernel completes. For small l, inline
the multiply inside the kernel to avoid a second kernel dispatch.

**Existing transform code:** `crates/cintx-cubecl/src/transform/` already
exists as a directory in the project. Its current state needs to be checked —
it may already have stub transform logic.

**Complexity:** Low-Medium. The math is straightforward. The challenge is
correct output shape handling (comp × n_sph_i × n_sph_j vs comp × n_cart_i × n_cart_j)
and ensuring the transform is applied before the staging buffer write.

**Dependencies:** `DeviceResidentCache` for transform coefficient storage,
output buffer dimensions from `TransferPlan`.

### 10. Oracle Parity Validation

**What:** After real kernels are implemented, the oracle CI gate must pass.
This means:
- The oracle harness (`crates/cintx-oracle/`) runs actual libcint 6.1.3
  (vendored in `libcint-master/`) for the same inputs
- `compare.rs` applies the family-specific tolerances (see table below)
- All stable family symbols must be within tolerance
- The `Phase2ParityReport` must have `mismatch_count == 0`

**Tolerance table (HIGH confidence — from `compare.rs` source, verified above):**

| Family | atol | rtol | Zero threshold |
|--------|------|------|----------------|
| 1e | 1e-11 | 1e-9 | 1e-18 |
| 2e | 1e-12 | 1e-10 | 1e-18 |
| 2c2e | 1e-9 | 1e-7 | 1e-18 |
| 3c2e | 1e-9 | 1e-7 | 1e-18 |
| 3c1e | 1e-7 | 1e-5 | 1e-18 |
| 4c1e | 1e-6 | 1e-5 | 1e-18 |

**Comparison logic:** `diff_summary()` in `compare.rs` uses:
```
abs_error = |observed - reference|
rel_error = abs_error / |reference|   (skipped when |reference| < zero_threshold)
pass = (abs_error <= atol) AND (rel_error <= rtol)
```

**What triggers zero_threshold:** When `|reference| < 1e-18`, the element is
treated as structurally zero and only the absolute error is checked. This prevents
division by near-zero from polluting the relative error metric.

**Complexity:** Low (the infrastructure is already built). The oracle gate
itself is not new work — making the kernels produce correct values is the work.

---

## Differentiators (Add Value, Not Required for Oracle Parity)

### D1. Exponential Screening (Pair Pre-filtering)

**What:** Skip shell pairs where the Schwarz upper bound or the libcint
`pdata.cceij > expcutoff` criterion predicts the integral is below the screening
threshold. Reduces the number of kernel dispatches for diffuse basis sets.

**Why differentiator:** Oracle parity does not depend on screening (screening
is a performance optimization). The library can pass all oracle tests without
it. Correct screening implementation requires matching libcint's `PairData`
construction, which adds complexity without changing correctness.

**When to add:** After oracle parity is confirmed on unscreened kernels.

### D2. Optimizer Cache (Non-zero Contraction Index Lists)

**What:** libcint's `CINTOpt` pre-computes sparse contraction indices
(`non0ctr`, `non0idx`) that skip zero coefficients in contracted shells.
For highly contracted basis sets (cc-pVTZ, def2-TZVP), this reduces the
contraction loop count significantly.

**Why differentiator:** Correctness does not depend on it — a dense contraction
loop produces identical results. This is purely a performance feature.

**When to add:** After oracle parity is confirmed.

### D3. Batched Shell-Quartet Launch

**What:** Launch multiple shell quartets in a single kernel dispatch using
a queue-sorted list, rather than one kernel per shell quartet. Reduces
kernel launch overhead for large basis sets.

**Why differentiator:** Each individual shell-quartet kernel is correct
independently. Batching is a throughput optimization.

### D4. Spinor Representation Kernels

**What:** Complex spinor output for 1e and 2e families using the eight
cart-to-spinor transform variants (already partially scaffolded in
`crates/cintx-cubecl/src/transform/`).

**Why differentiator:** Spinor integrals are required for 4-component
relativistic calculations but are not needed for the common non-relativistic
use cases. The oracle harness has `CINTcgtos_spinor` and spinor helper coverage,
but spinor integral output is the most complex transform path.

**When to add:** After cart and sph parity is confirmed for the core families.

### D5. F12/STG/YP Source-Only Family Kernels

**What:** Real kernels for the `with-f12` feature: `int2e_stg_sph` and
`int2e_yp_sph` operators. These use range-separated Coulomb operators
modified by Slater-type or Yukawa potentials.

**Why differentiator:** The `with-f12` profile is an optional feature gate.
The base profile (1e, 2e, 2c2e, 3c1e, 3c2e) is sufficient for v1.1 parity.
F12 adds the `with-f12` profile to the oracle gate.

**Note on tolerance:** The Libcint 6 paper notes relative errors for range-
separated attenuated Coulomb interactions can reach 1e-10, slightly looser
than regular ERIs (confirmed by the existing 2C2E/3C2E tolerance of 1e-7
in `compare.rs`).

---

## Anti-Features (Explicitly Exclude from v1.1)

### X1. GTG Family Kernels

**What:** "Gaussian type geminal" integrals. Upstream marks these as having
known bugs (`CMakeLists.txt:106-109`). The design explicitly keeps GTG
out of GA, optional, and unstable categories.

**Decision:** No GPU kernel, no feature flag, no manifest entry. The
`resolve_family_name` in `kernels/mod.rs` must never match `"gtg"`.

### X2. Bitwise-Identical Libcint Internals

**What:** Reproducing libcint's specific DRK 2D-integral intermediate
scratch layout, its exact cache-stack memory management (`MALLOC_INSTACK`),
or its SIMD-packed register patterns.

**Decision:** Target result compatibility only (L1 in the design). Internal
implementation may differ freely as long as the oracle gate passes.
Using Rys quadrature instead of DRK is explicitly allowed.

### X3. Asynchronous Public API

**What:** `async fn evaluate(...)` or any futures-based execution path.

**Decision:** All public APIs remain synchronous. GPU execution may use
internal command queues but the caller always blocks.

### X4. 4c1e Beyond the Validated Envelope

**What:** 4c1e with spinor representation, max(l) > 4, non-natural dims,
or inputs that fail the identity test
`int4c1e_sph == (-1/4π) * trace(int2e_ipip1 + 2*int2e_ipvip1 + permuted)`.

**Decision:** The planner's `Validated4C1E` classifier must reject inputs
outside the bug envelope with `UnsupportedApi`. The workaround path
`compat::workaround::int4c1e_via_2e_trace` is the sanctioned fallback.

### X5. Host-CPU Integral Computation

**What:** Pure CPU recurrence-relation implementations of the integral
families (outside of CubeCL's CPU runtime backend).

**Decision:** The design constraint is "CubeCL is the primary compute
backend; host CPU work stays limited to planning, validation, marshaling,
and test/oracle glue." The CubeCL CPU runtime backend (for testing) is
acceptable; a separate CPU integral library path is not.

---

## Feature Dependencies

```
CubeCL Direct Client API (TS1)
  → Gaussian Primitive Infrastructure (TS8)
    → 1e Real Kernel (TS3)
    → Boys Function Table in DeviceResidentCache
      → 2e Real Kernel (TS4)  [highest complexity]
      → 2c2e Real Kernel (TS5) [shares Boys infra with 2e]
      → 3c1e Real Kernel (TS6) [shares Boys infra]
      → 3c2e Real Kernel (TS7) [shares 2e/Boys infra]
  → Cart-to-Sph Transform (TS9)
    → required by all kernels when Representation::Spheric requested

Configurable Backend (TS2) → required for CI without GPU hardware

Oracle Parity Gate (TS10) → passes only after TS3-TS7 + TS8 + TS9 are correct
```

---

## Integral Math Approach Comparison

The question "which recurrence relation algorithm" is the most important
algorithmic decision for v1.1. Three algorithms are in use by the GPU
quantum chemistry community:

| Algorithm | Memory footprint | GPU suitability | Used by |
|-----------|-----------------|-----------------|---------|
| Rys quadrature | Small (register-friendly) | High | GPU4PySCF, cintx |
| McMurchie-Davidson | Medium | Medium-High | TeraChem, GAMESS-GPU |
| Obara-Saika / Head-Gordon-Pople | Large (workspace grows with l) | Lower | libmint, NWChem-GPU |

**Recommendation for cintx v1.1:** Use Rys quadrature as the primary 2e/2c2e/3c2e
algorithm and McMurchie-Davidson E-coefficient recurrence for 1e/3c1e.

Rationale:
- Rys quadrature: small memory footprint matches CubeCL/wgpu's constrained
  per-thread shared memory. The Boys function table fits comfortably in
  `DeviceResidentCache`. GPU4PySCF uses it successfully for up-to-g-function
  integrals with τ=1e-10 accuracy.
- McMurchie-Davidson for 1e: the E-coefficient tables are small (1e integrals
  have no electron repulsion) and the closed-form overlap/kinetic/nuclear
  attraction expressions are compact.
- Obara-Saika is the algorithm libcint started with but moved away from for
  its code generator (DRK is used instead). For GPU use, OS has larger
  intermediate buffers that increase register pressure.

**Note on libcint's DRK algorithm:** libcint uses the Dupuis-Rys-King
two-dimensional integral framework, which is mathematically equivalent to Rys
quadrature. The cintx GPU implementation does not need to match the DRK
intermediate scratch layout — only the final contracted integral values must
agree within the oracle tolerance. (MEDIUM confidence — confirmed from libcint
paper and source analysis, but the exact floating-point equivalence path between
DRK intermediates and Rys outputs requires careful verification during implementation.)

---

## MVP Recommendation for v1.1

**Phase ordering for shipping oracle parity:**

1. CubeCL direct client API rewrite (TS1) + configurable backend (TS2)
   — unblocks all kernel work on both GPU and CPU runtimes

2. Gaussian primitive infrastructure (TS8) + Boys function table (part of TS4)
   — shared foundation for all families

3. 1e real kernel (TS3) + cart-to-sph transform (TS9)
   — simplest integral family; validates the end-to-end pipeline on a tractable problem

4. 2e real kernel (TS4)
   — most complex; implement after 1e proves the pipeline correct

5. 2c2e (TS5), 3c1e (TS6), 3c2e (TS7) real kernels
   — build on Boys/Rys infrastructure from TS4; can proceed in parallel after TS4

6. Oracle parity gate (TS10)
   — run after each kernel family, not only at the end

**Defer to post-v1.1:**
- Exponential screening (D1)
- Optimizer cache contraction index lists (D2)
- Batched shell-quartet dispatch (D3)
- Spinor representation kernels (D4)
- F12/STG/YP kernels (D5)

---

## Sources

- libcint paper (DRK algorithm, cart2sph, tolerance): https://ar5iv.labs.arxiv.org/html/1412.0649
- libcint 6 updates (SIMD, API): https://pubmed.ncbi.nlm.nih.gov/38748029/
- GPU4PySCF Rys quadrature GPU implementation: https://arxiv.org/html/2407.09700v1
- TeraChem f-function McMurchie-Davidson GPU: https://arxiv.org/html/2406.14920v1
- GPU Boys function evaluation (gridded Taylor, 2025): https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.8328
- 3-center two-electron GPU (efficient GPU impl): https://www.researchgate.net/publication/396374573_Efficient_GPU_Implementations_of_Three-Center_Two-Electron_Repulsion_Integrals
- CubeCL architecture and client API: https://www.thomasantony.com/posts/202512281621-gpu-agnostic-programming-using-cubecl/
- CubeCL GitHub: https://github.com/tracel-ai/cubecl
- McMurchie-Davidson GPU ERI (2025): https://www.mdpi.com/2076-3417/15/5/2572
- Project design document: /home/chemtech/workspace/cintx/docs/design/cintx_detailed_design.md (sections 3.6, 3.7, 3.11, 4.6)
- Oracle tolerance table: /home/chemtech/workspace/cintx/crates/cintx-oracle/src/compare.rs (lines 21-31)
- Existing kernel stubs: /home/chemtech/workspace/cintx/crates/cintx-cubecl/src/kernels/
- libcint 1e source loop: /home/chemtech/workspace/cintx/libcint-master/src/cint1e.c
- libcint cart2sph coefficients: /home/chemtech/workspace/cintx/libcint-master/src/cart2sph.c
- libcint test tolerances: /home/chemtech/workspace/cintx/libcint-master/testsuite/test_int1e.py (line 57: thr=1e-9)
