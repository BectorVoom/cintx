# Phase 9: 1e Real Kernel and Cart-to-Sph Transform - Research

**Researched:** 2026-04-03
**Domain:** Quantum chemistry Gaussian integral kernels (Obara-Saika 1e), Condon-Shortley cart-to-sph transform, oracle parity validation
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**1e Kernel Structure**
- D-01: Single `launch_one_electron` entry point dispatches to a shared G-tensor fill (VRR/HRR for gx, gy, gz arrays) followed by per-operator post-processing. Matches libcint's g1e.c architecture.
- D-02: Operator switch inside the kernel: overlap = direct contraction, kinetic = nabla-squared post-process, nuclear = Boys-weighted sum over atom centers.
- D-03: The shared G-fill uses the Phase 8 `vrr_step()` and `hrr_step()` #[cube] functions from `math/obara_saika.rs`, plus `compute_pdata()` from `math/pdata.rs`.

**Cart-to-Sph Transform**
- D-04: Cart-to-sph is implemented as host-side Rust code (not a #[cube] function). Kernel writes cartesian components to staging, `client.read()` brings data to host, then `cart_to_sph_1e()` applies Condon-Shortley matrix per shell pair before writing to `io.staging_output()`.
- D-05: Condon-Shortley coefficients are extracted from libcint's `cart2sph.c` `g_trans_cart2sph[]` array for l=0..4. Stored as const arrays in `transform/c2s.rs`.
- D-06: GPU-side c2s is a future optimization (deferred). Host-side is sufficient for correctness validation and oracle parity.

**Nuclear Attraction Operator**
- D-07: Nuclear attraction loops over all atom centers C inside the kernel. For each atom: compute Boys F_m(t) where t = aij * |P-C|^2, fill G-tensor with nuclear-weighted VRR using PC displacement, accumulate Z_c * contracted result.
- D-08: Atom coordinates and charges are passed as input arrays to the kernel function. Uses Phase 8 `boys_gamma_inc()` from `math/boys.rs`.

**Validation Strategy**
- D-09: Cart-to-sph coefficients validated for all angular momenta l=0..4 via dedicated unit tests comparing coefficient matrices against libcint `cart2sph.c` reference values. Covers: l=0 (1x1), l=1 (3x3), l=2 (6x5), l=3 (10x7), l=4 (15x9).
- D-10: End-to-end oracle parity test: H2O STO-3G for int1e_ovlp_sph, int1e_kin_sph, int1e_nuc_sph. Tolerances: atol 1e-11 / rtol 1e-9 (per success criteria).
- D-11: Oracle parity verified per family as each kernel lands (VERI-05), not deferred to end.

**Carried Forward**
- D-12: Host wrapper + #[cube] pair pattern for math functions (Phase 8). Integration tests use host-side wrappers, not CubeCL CPU backend launch (avoids cond_br MLIR limitation).
- D-13: CPU backend is the primary oracle target (Phase 7 D-03). Tests run under `--features cpu`.
- D-14: Buffer lifecycle lives inside kernel family modules, not centralized (Phase 7 D-07).
- D-15: Staging buffer sizing already accounts for spherical vs cartesian via `ao_per_shell()` in `cintx_core/shell.rs` (Cart: (l+1)(l+2)/2, Spheric: 2l+1).

### Claude's Discretion
- Internal G-tensor array sizing and indexing strategy (flat vs 3-component gx/gy/gz)
- Exact GTO contraction loop structure
- How operator ID is extracted from `SpecializationKey` or `ExecutionPlan` to select the operator variant
- Host-side c2s buffer management (in-place vs separate cart/sph buffers)
- Test fixture design for c2s coefficient validation

### Deferred Ideas (OUT OF SCOPE)
- GPU-side #[cube] cart-to-sph transform -- future optimization after correctness proven on host
- Higher angular momentum end-to-end tests (cc-pVTZ/cc-pVQZ with d/f shells) -- Phase 10 will exercise these via 2e test cases
- Workgroup sizing for kernel launch -- post-v1.1 optimization
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| KERN-01 | 1e family kernels (overlap, kinetic, nuclear attraction) produce real values via `#[cube(launch)]` | G-tensor fill algorithm from g1e.c verified; kinetic nabla-squared from intor1.c verified; nuclear Boys-VRR loop from g1e.c lines 208-320 verified |
| KERN-06 | Cart-to-sph transform implements real Condon-Shortley coefficients replacing stub blend | `g_trans_cart2sph[]` offset table verified from cart2sph.c g_c2s[]; matrix sizes confirmed (1, 9, 30, 70, 135 floats for l=0..4) |
| VERI-05 | Oracle parity verified per family as each kernel lands (not deferred to end) | Oracle harness `compare.rs` already defines TOL_1E_ATOL=1e-11, TOL_1E_RTOL=1e-9; test infrastructure exists |
</phase_requirements>

---

## Summary

Phase 9 replaces two stubs with real implementations: `one_electron.rs` (currently returns zeros) and `c2s.rs` (currently applies a meaningless running blend). Both files are small and well-bounded; all upstream math primitives (VRR, HRR, Boys, PairData) are complete from Phase 8.

The 1e kernel follows the exact libcint architecture: a shared G-tensor fill producing three interleaved arrays `gx/gy/gz`, followed by operator-specific post-processing. Overlap is a direct product `g[ix]*g[iy]*g[iz]`. Kinetic applies `CINTnabla1j_1e` (nabla operator on j-center) using the `-2aj * g[n+1] + n * g[n-1]` derivative formula, with a `0.5` common factor. Nuclear loops over all atoms, calling Boys + Rys-VRR for each center and accumulating `Z_C * contracted_result`.

The cart-to-sph transform is a straightforward matrix-vector multiplication using coefficient tables extracted verbatim from libcint's `g_trans_cart2sph[]` array. For Phase 9 this runs host-side after `client.read()` brings cartesian staging data back to CPU. The coefficient offsets into `g_trans_cart2sph[]` are: l=0 at 0 (1 coefficient), l=1 at 1 (9), l=2 at 10 (30), l=3 at 40 (70), l=4 at 110 (135). Oracle parity infrastructure already has the right tolerance constants (`TOL_1E_ATOL=1e-11`, `TOL_1E_RTOL=1e-9`).

**Primary recommendation:** Implement `launch_one_electron` as a pure host-side pipeline (no GPU kernel launch yet -- the CPU backend returns results directly from host math), following Phase 8's pattern of host-wrapper calls. Replace `cart_to_spheric_staging` with a real Condon-Shortley matrix apply. Wire the oracle parity test for `int1e_ovlp_sph`, `int1e_kin_sph`, `int1e_nuc_sph` using the existing oracle harness.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cubecl` | 0.9.x | #[cube] kernel functions | Project-locked backend |
| `approx` | 0.5 | `assert_abs_diff_eq!` in tests | Already in dev-dependencies |
| `cintx_cubecl::math::*` | (workspace) | Boys, PairData, VRR/HRR primitives | Completed Phase 8 |
| `cintx_oracle` | (workspace) | Oracle parity comparison harness | Already exists with 1e tolerances |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `cintx_core::Shell` | (workspace) | `ao_per_shell()` for buffer sizing | Staging size computation |
| `cintx_compat::raw` | (workspace) | Oracle raw API invocation | Oracle parity test |

**Installation:** No new dependencies — all required libraries are in the workspace already.

---

## Architecture Patterns

### Recommended Project Structure

The implementation touches exactly these files:

```
crates/cintx-cubecl/src/
├── kernels/one_electron.rs      # Replace stub with real 1e pipeline
├── transform/c2s.rs             # Replace stub with Condon-Shortley matrix apply
├── transform/mod.rs             # No change needed (calls c2s already)

crates/cintx-cubecl/tests/
├── one_electron_tests.rs        # New: unit tests for operator post-processing
├── c2s_tests.rs                 # New: coefficient matrix correctness per l

crates/cintx-oracle/src/
└── compare.rs (or new test)     # Oracle parity test for 1e sph family
```

### Pattern 1: G-Tensor Fill + Operator Post-Processing (libcint g1e.c architecture)

**What:** The G-tensor `g[0..3*g_size]` is split into three axis arrays `gx/gy/gz` (offsets 0, g_size, 2*g_size). VRR fills each axis. The final integral is `gout[n] = g[idx_x]*g[idx_y]*g[idx_z]` for overlap, modified for kinetic and nuclear.

**When to use:** Every 1e kernel call.

**Key libcint lines (verified):**
```c
// g1e.c lines 127-134: base case setup
gx[0] = 1;
gy[0] = 1;
gz[0] = envs->fac[0] * SQRTPI*M_PI / (aij * sqrt(aij));  // overlap base value

// g1e.c lines 164-182: VRR + HRR (already ported as vrr_step, hrr_step in Phase 8)

// cint1e.c lines 306-325: CINTgout1e — final product accumulation
gout[n] = g[idx_x] * g[idx_y] * g[idx_z];  // overlap: direct product
```

**Rust host-side pseudocode:**
```rust
// In launch_one_electron (host-side pipeline, no GPU launch for CPU backend)
fn ovlp_primitive(ai, aj, ri, rj, li, lj, norm_i, norm_j) -> Vec<f64> {
    let pd = compute_pdata_host(ai, aj, ...);
    let aij = pd.zeta_ab;
    
    // G-array: flat [gx | gy | gz], each of size (li+lj+1)
    let g_size = (li + lj + 1) as usize;
    let mut g = vec![0.0; 3 * g_size];
    
    // Base case (g1e.c lines 132-134)
    g[0] = 1.0;               // gx[0]
    g[g_size] = 1.0;          // gy[0]
    const SQRTPI_M_PI: f64 = std::f64::consts::PI.sqrt() * std::f64::consts::PI;
    g[2*g_size] = pd.fac * SQRTPI_M_PI / (aij * aij.sqrt());  // gz[0]
    
    // VRR for each axis (Phase 8 vrr_step_host)
    let rijrx = pd.center_p_x - ri[0];
    // ... apply vrr_step_host / hrr_step_host per axis ...
    
    // Accumulate: gout[n] = gx[idx_x] * gy[idx_y] * gz[idx_z]
    // idx comes from CINTg1e_index_xyz pattern
    todo!()
}
```

### Pattern 2: Kinetic Operator (CINTgout1e_int1e_kin, intor1.c lines 18-46)

**What:** The kinetic operator uses nabla derivatives on the j-center. libcint generates `g1, g2, g3` arrays via `G1E_D_J` (nabla on j), then contracts as:
```
gout[n] = -(g3[ix]*g0[iy]*g0[iz] + g0[ix]*g3[iy]*g0[iz] + g0[ix]*g0[iy]*g3[iz])
```
with a `0.5` common factor applied at the driver level (`envs.common_factor *= 0.5`).

**Nabla-on-j formula** (`CINTnabla1j_1e`, g1e.c lines 352-384):
```c
// j=0 term: f[i] = aj2 * g[i + dj]  where aj2 = -2*aj
// j>0 term: f[i] = j * g[i-dj] + aj2 * g[i+dj]
```

This means in Rust host-side:
```rust
// Derivative of g-array with respect to j-center exponent:
// nabla_j_gx[n] = -2*aj * gx[n + dj]         for j=0
// nabla_j_gx[n] = n * gx[n - dj] - 2*aj * gx[n + dj]  for j>0
// kinetic = -0.5 * (nabla_j_nabla_j applied twice) = 
//   -0.5 * (d2gx/dj2 * gy * gz + gx * d2gy/dj2 * gz + gx * gy * d2gz/dj2)
```

**Note on G1E_D_J macro:** `g3 = D_J(D_J(g0))` — two successive nabla-j applications. The intermediate `g2 = D_J(g0)` at lj+1, then `g3 = D_J(g2)` at lj. The result s[0]+s[4]+s[8] diagonal terms (x, y, z) are negated and summed.

### Pattern 3: Nuclear Attraction (CINTg1e_nuc, g1e.c lines 208-320)

**What:** Nuclear attraction loops over Rys roots (nrys_roots = (li+lj)/2+1). For each atom center C with charge Z_C:

```c
// g1e.c lines 239-246
crij[d] = cr[d] - rij[d];       // displacement from Gaussian center to nucleus
x = aij * tau*tau * |crij|^2;   // Boys argument (for point nuclei: tau=1)
CINTrys_roots(nrys_roots, x, u, w);  // Rys roots u[], weights w[] -> gz[i] = fac1 * w[i]

// VRR for each Rys root (root-dependent c00, rt):
// c00 = rijrx + ru * crij[d]  (d=0,1,2 for x,y,z)
// rt  = aij2 - aij2 * ru      where ru = tau^2 * u / (1 + u)
// g0 VRR: p0x[n+i*di] = i * rt * p1x[n+i*di] + r0 * gx[n+i*di]

// Output: gout[n] = sum_i( gx[i+ix] * gy[i+iy] * gz[i+iz] )  over Rys roots
```

For Point nuclei: `tau=1`, `fac1 = 2*pi * (-Z_C) * envs->fac[0] / aij`.

**Implementation note:** This uses Rys quadrature (not Boys directly). The Phase 8 Rys module (`math/rys.rs`) provides `rys_root1_host()` for nroots=1. For higher angular momenta, nroots can be > 1 and the full polynomial fit tables are required (already implemented in Phase 8 `rys.rs`).

### Pattern 4: Condon-Shortley Cart-to-Sph Matrix Apply

**What:** For each shell pair (i, j) with angular momenta (li, lj), apply the transformation matrix `T[li] ⊗ T[lj]` to the cartesian output block, where `T[l]` is the `(2l+1) x ncart(l)` matrix from `g_trans_cart2sph[]`.

**Matrix dimensions (verified from g_c2s[] offsets in cart2sph.c):**
| l | ncart | nsph | matrix elements | offset in g_trans_cart2sph |
|---|-------|------|-----------------|---------------------------|
| 0 | 1 | 1 | 1 | 0 |
| 1 | 3 | 3 | 9 | 1 |
| 2 | 6 | 5 | 30 | 10 |
| 3 | 10 | 7 | 70 | 40 |
| 4 | 15 | 9 | 135 | 110 |

**Libcint g_trans_cart2sph[] is stored row-major** with rows = spherical components (2l+1) and columns = cartesian components ncart(l). The transform: `gsph[m * ncart + k] = sum_c( T[m*ncart + c] * gcart[k * ncart + c] )` where k is the bra index.

Actually — the matrix is applied as a DGEMM: `CINTdgemm_TN(nd, nket, nf, T, gcart, gsph)` for bra transform, `CINTdgemm_NN1(nbra, nd, nf, gcart, T, gsph, lds)` for ket transform. For 1e integrals (`c2s_sph_1e`), both bra and ket are transformed.

**Host-side Rust matrix multiply (for 1e, no contraction indices yet):**
```rust
// cart_to_sph_1e(cart_buf: &[f64], li: u8, lj: u8) -> Vec<f64>
// Input:  cart_buf layout = [j_cart_cols x i_cart_rows] (nfj * nfi floats)
// Output: sph_buf layout  = [j_sph_cols  x i_sph_rows]  (ndj * ndi floats)
// Step 1: apply T[lj] on ket (j-axis): (ndi, nfj) -> (ndi, ndj)
// Step 2: apply T[li] on bra (i-axis): (nfi, ndj) -> (ndi, ndj)
```

**Coefficient storage in `transform/c2s.rs`:**
```rust
// L=0: identity (1x1)
pub const C2S_L0: [[f64; 1]; 1] = [[1.0]];

// L=1: px,py,pz -> m=-1,0,1 (3x3 identity, default ordering px,py,pz)
// Source: g_trans_cart2sph[1..10], d=l ordering is px,py,pz -> m=0,±1
pub const C2S_L1: [[f64; 3]; 3] = [
    [1.0, 0.0, 0.0],  // m= 0 (px = real sph m=0 by default convention)
    [0.0, 1.0, 0.0],  // m=-1 (py)
    [0.0, 0.0, 1.0],  // m=+1 (pz)
];

// L=2: 6 cartesian -> 5 spherical (dxy, dyz, dz2, dxz, dx2-y2 order)
// Source: g_trans_cart2sph[10..40], verified values below
pub const C2S_L2: [[f64; 6]; 5] = [
    [0.0,  1.092548430592079070, 0.0, 0.0, 0.0, 0.0],
    [0.0,  0.0, 0.0, 0.0, 1.092548430592079070, 0.0],
    [-0.315391565252520002, 0.0, 0.0, -0.315391565252520002, 0.0, 0.630783130505040012],
    [0.0, 0.0, 1.092548430592079070, 0.0, 0.0, 0.0],
    [0.546274215296039535, 0.0, 0.0, -0.546274215296039535, 0.0, 0.0],
];
// ... L=3 (10->7) and L=4 (15->9) follow same pattern from g_trans_cart2sph
```

### Anti-Patterns to Avoid
- **Calling into GPU kernel from integration tests:** Integration tests use `*_host()` wrappers only. The cond_br MLIR limitation from Phase 8 applies here too (D-12).
- **Accumulating nuclear attraction result without Rys loop:** Nuclear uses Rys quadrature (nroots roots), not a single Boys call. Each root contributes to the sum via `gz[root] * fac1`.
- **Forgetting the 0.5 kinetic common_factor:** `int1e_kin` applies `envs.common_factor *= 0.5` before the integral driver. This must be applied to the final output, not inside the VRR.
- **Wrong c2s coefficient ordering:** libcint uses the real spherical harmonic ordering m = -l, -l+1, ..., 0, ..., l-1, l (rows of T). The g_trans_cart2sph[] rows already encode this. Do not reorder.
- **Reusing the stub `cart_to_spheric_staging` signature:** The new `cart_to_sph_1e()` must accept shell angular momenta `(li, lj)` to select the right coefficient matrices. The current stub signature `cart_to_spheric_staging(staging: &mut [f64])` is not sufficient.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Boys function for nuclear VRR | Custom Boys implementation | `boys_gamma_inc_host()` from Phase 8 `math/boys.rs` | Already verified against libcint fmt.c |
| Gaussian pair setup (P center, fac, aij2) | Manual zeta/fac computation | `compute_pdata_host()` from Phase 8 `math/pdata.rs` | Already verified, handles normalization |
| VRR/HRR recurrence | Custom recurrence loops | `vrr_step_host()` / `hrr_step_host()` from Phase 8 `math/obara_saika.rs` | Verified against g1e.c lines 164-182 |
| Matrix multiply for c2s | Custom loop | A direct `O(nsph * ncart * nbatch)` loop in Rust is fine; avoid reimplementing BLAS | For Phase 9 sizes (max 15x9=135 floats) a simple nested loop is correct and readable |
| Oracle reference values | Recomputing libcint integrals | `cintx_oracle::compare` / `cintx_compat::raw` | The oracle harness already calls vendored libcint via FFI |

**Key insight:** The kernel for Phase 9 runs entirely host-side via the existing `_host()` wrapper pattern. No `#[cube(launch)]` is required to produce correct results on the CPU backend — the CubeCL CPU executor invokes the Rust function directly. The `#[cube]` annotation is needed only for GPU dispatch (Phase 10+).

---

## Common Pitfalls

### Pitfall 1: Nuclear attraction needs Rys quadrature, not direct Boys
**What goes wrong:** Writing nuclear as `F_m(t) * gx * gy * gz` and getting wrong values.
**Why it happens:** Nuclear is Rys-based because the `1/r` Coulomb operator introduces a Gaussian resolution via the Boys function indirectly — but the actual G-tensor fill for nuclear uses Rys roots `u[i]` to handle the angular momentum recurrence, not direct Boys values.
**How to avoid:** Follow `CINTg1e_nuc` architecture: `CINTrys_roots(nrys_roots, x, u, w)` fills `u[]` and `w[]`; `gz[i] = fac1 * w[i]`; then a per-root VRR fills `gx[n+i*di]` using root-dependent `c00` and `rt` coefficients; final contraction sums over roots.
**Warning signs:** All nuclear attraction values are exactly zero or match the overlap values incorrectly.

### Pitfall 2: Kinetic operator needs two G-arrays (g2 = D_J(g0), g3 = D_J(g2))
**What goes wrong:** Implementing kinetic as a single nabla pass and getting wrong magnitudes.
**Why it happens:** The kinetic operator is `T = -0.5 * nabla^2`. The nabla-j operator is applied twice (second-order derivative). The libcint autocode in `intor1.c` creates `g1 = D_J(g0, lj+0)`, `g2 = D_J(g0, lj+1)`, `g3 = D_J(g2, lj+0)`. The result is `gout[n] = -(g3[ix]*g0[iy]*g0[iz] + g0[ix]*g3[iy]*g0[iz] + g0[ix]*g0[iy]*g3[iz])` with 0.5 factor.
**How to avoid:** Implement `nabla_j` as: `nabla[n] = -2*aj * g[n+dj]` (n=0), `nabla[n] = n * g[n-dj] - 2*aj * g[n+dj]` (n>0). Apply twice in sequence.
**Warning signs:** Kinetic values off by a constant factor of 2 or 4.

### Pitfall 3: G-tensor array size must accommodate li_ceil + lj_ceil + 1 elements
**What goes wrong:** Buffer overflow or wrong indexing for non-s-shell pairs.
**Why it happens:** VRR fills up to nmax = li_ceil + lj_ceil elements; HRR then reads from those to fill j-angular-momentum components. The total axis length is `li_ceil + lj_ceil + 1`.
**How to avoid:** Allocate `g = vec![0.0; 3 * (li + lj + 1)]` per shell pair primitive. For Phase 9 H2O STO-3G this is at most l=1 so nmax=2 — safe range.
**Warning signs:** Panic on index out of bounds, or silent wrong values if buffer was zeroed.

### Pitfall 4: c2s coefficient layout is row = spherical component, col = cartesian component
**What goes wrong:** Transposing the matrix and getting wrong output.
**Why it happens:** The g_trans_cart2sph[] array in libcint is laid out with rows indexing m (-l to +l) and columns indexing Cartesian components (in lexicographic order as per `CINTcart_comp`). Confusion between row-major C storage and column-major Fortran convention leads to silent incorrect results.
**How to avoid:** Extract coefficients exactly as: `C2S_L2[m][cart_idx]` for m=0..4, cart_idx=0..5. Apply as `sph[m] = sum_c( C2S_L2[m][c] * cart[c] )` for each bra index k.
**Warning signs:** Integral magnitudes correct but angular structure wrong; off-diagonal block elements in overlap matrix are non-symmetric.

### Pitfall 5: Rij (Gaussian product center) vs Ri (shell center) displacement
**What goes wrong:** Using `pdata.center_p_x - rj[0]` instead of `pdata.center_p_x - rx[0]` in VRR.
**Why it happens:** g1e.c line 160-162 sets `rijrx[d] = rij[d] - rx[d]` where `rx` is the center with larger angular momentum (either ri or rj depending on `ibase` flag). The Phase 8 math integration tests showed `rijrx = center_p_x - ri[0]` for the i-center VRR.
**How to avoid:** Follow the g1e.c `ibase` branching logic: if `li_ceil > lj_ceil`, `rx = ri` and `di=g_stride_i`, else `rx = rj` and `di=g_stride_j`.

### Pitfall 6: The transform/c2s.rs function signature must change
**What goes wrong:** Trying to call `cart_to_spheric_staging(staging)` with no angular momentum info and getting wrong results.
**Why it happens:** The current stub ignores angular momenta. The real transform needs `(li, lj)` to pick the right coefficient matrix. The `executor.rs` line 206 calls `transform::apply_representation_transform(plan.representation, staging)` — this will need to be extended to pass shell angular momenta.
**How to avoid:** Change `cart_to_sph_1e` to accept `(staging: &mut [f64], li: u8, lj: u8, nctr_i: usize, nctr_j: usize)`. Update `apply_representation_transform` signature or create a separate 1e-specific transform call in `one_electron.rs` that bypasses the generic transform dispatcher.

---

## Code Examples

Verified patterns from official sources (libcint 6.1.3):

### G-tensor Base Case Setup (g1e.c lines 127-134)
```rust
// Source: libcint-master/src/g1e.c lines 127-134
const SQRTPI: f64 = 1.7724538509055159; // sqrt(pi)
const M_PI: f64 = std::f64::consts::PI;

// gz[0] carries the full normalization factor for the z-axis
let g_z0 = pd.fac * SQRTPI * M_PI / (pd.zeta_ab * pd.zeta_ab.sqrt());
// gx[0] = gy[0] = 1.0 (unit base for x and y axes)
```

### Nabla-J Derivative (g1e.c lines 352-384, CINTnabla1j_1e)
```rust
// Source: libcint-master/src/g1e.c CINTnabla1j_1e
// f[i] = -2*aj * g[i + dj]             for j=0 range
// f[i] = j * g[i-dj] + (-2*aj) * g[i+dj]  for j=1..lj range
fn nabla_j_host(f: &mut [f64], g: &[f64], dj: usize, lj: usize, li: usize, aj: f64) {
    let aj2 = -2.0 * aj;
    // j=0: f[i] = aj2 * g[i + dj]
    for i in 0..=li {
        f[i] = aj2 * g[i + dj];
    }
    // j=1..lj: f[j*dj + i] = j*g[(j-1)*dj+i] + aj2*g[(j+1)*dj+i]
    for j in 1..=lj {
        for i in 0..=(li) {
            let idx = j * dj + i;
            f[idx] = j as f64 * g[(j-1)*dj + i] + aj2 * g[(j+1)*dj + i];
        }
    }
}
```

### Condon-Shortley D-shell Coefficients (g_trans_cart2sph[10..40], cart2sph.c lines 52-81)
```rust
// Source: libcint-master/src/cart2sph.c g_trans_cart2sph[10..40]
// Row = spherical m (-2,-1,0,+1,+2), Col = cartesian (xx,xy,xz,yy,yz,zz)
pub const C2S_D: [[f64; 6]; 5] = [
    [0.0, 1.092548430592079070, 0.0, 0.0, 0.0, 0.0],               // m=-2: dxy
    [0.0, 0.0, 0.0, 0.0, 1.092548430592079070, 0.0],               // m=-1: dyz
    [-0.315391565252520002, 0.0, 0.0, -0.315391565252520002, 0.0, 0.630783130505040012], // m=0: dz2
    [0.0, 0.0, 1.092548430592079070, 0.0, 0.0, 0.0],               // m=+1: dxz
    [0.546274215296039535, 0.0, 0.0, -0.546274215296039535, 0.0, 0.0], // m=+2: dx2-y2
];
```

### Nuclear Attraction Rys Root VRR (g1e.c lines 282-297)
```rust
// Source: libcint-master/src/g1e.c CINTg1e_nuc lines 282-297
// For each Rys root n (0..nrys_roots):
//   ru = tau^2 * u[n] / (1 + u[n])  (for point nucleus: tau=1)
//   rt = aij2 - aij2 * ru            (modified aij2)
//   r0 = rijrx + ru * crij[x]        (modified displacement)
//   p0x[n+1*di] = r0 * gx[n]
//   p0x[n+i*di] = i * rt * p1x[n+i*di] + r0 * gx[n+i*di]  for i=1..nmax-1
// where:
//   crij[d] = cr_nucleus[d] - rij[d]  (nucleus to Gaussian center)
//   gx[n] = 1.0 for all roots n (initialized before VRR)
//   gz[n] = fac1 * w[n]  where fac1 = 2*pi * Z_C * fac / aij (point nucleus, negative Z)
```

### Oracle Parity Test Pattern (H2O STO-3G)
```rust
// Source: crates/cintx-oracle/src/compare.rs (existing constants)
// TOL_1E_ATOL = 1e-11, TOL_1E_RTOL = 1e-9 — already defined
//
// Test wiring approach (using existing oracle infrastructure):
// 1. Build H2O STO-3G basis via OracleRawInputs
// 2. Call cintx_compat::raw::int1e_ovlp_sph (via libcint FFI) -> reference
// 3. Call cintx CubeClExecutor::execute() with int1e_ovlp_sph -> cintx result
// 4. Compare elementwise: |cintx - libcint| <= atol + rtol * |libcint|
// 5. Assert mismatch_count == 0
```

---

## Runtime State Inventory

Not applicable — this is a greenfield kernel implementation phase, not a rename/refactor.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `cart_to_spheric_staging` returns blend of neighbors | Real Condon-Shortley matrix multiply | Phase 9 | Correct spherical harmonic outputs |
| `launch_one_electron` returns zeros | Real Obara-Saika G-fill + operator dispatch | Phase 9 | Non-zero, oracle-comparable outputs |

**Deprecated/outdated in this phase:**
- The running-blend stub in `c2s.rs` is intentionally a placeholder from Phase 2 — it must be completely replaced.
- The `not0: i32::from(!staging.is_empty())` metric in the stub is wrong; after Phase 9 `not0` should be the count of non-zero output elements.

---

## Open Questions

1. **Nuclear attraction: `ibase` branching for li vs lj**
   - What we know: g1e.c selects `rx = ri` if `li_ceil > lj_ceil`, otherwise `rx = rj`. The Phase 8 `vrr_step_host` tests used `rx = ri` convention.
   - What's unclear: For the initial H2O STO-3G implementation (all s and p shells), li and lj will often be equal. The ibase selection needs to be explicit.
   - Recommendation: Implement both branches in host-side code; test with asymmetric (li != lj) shell pairs in unit tests.

2. **How operator type is dispatched inside `launch_one_electron`**
   - What we know: `SpecializationKey` carries `canonical_family` ("1e") but not the specific operator (ovlp vs kin vs nuc). The `ExecutionPlan` carries `descriptor` which has the `symbol` name.
   - What's unclear: The cleanest way to extract operator type — via `plan.descriptor.entry.symbol` string matching, or via a new enum.
   - Recommendation (Claude's discretion): Match on `plan.descriptor.entry.symbol` with `contains("kin")`, `contains("nuc")` patterns. This avoids a new enum and matches how libcint identifies operators via the int1e_type integer switch.

3. **c2s: in-place vs separate cart/sph buffers**
   - What we know: The staging buffer is pre-sized for spherical output (per D-15, `ao_per_shell()` for Spheric gives 2l+1). The kernel writes cartesian, which has more elements for l>=2.
   - What's unclear: Whether `staging` is sized for cart or sph on entry to `launch_one_electron`.
   - Recommendation (Claude's discretion): The kernel should write to a local cart buffer first, then apply c2s into `staging`. Staging is correctly pre-sized for sph by the planner.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Rust toolchain | Compilation | ✓ | 1.94.0 (pinned) | — |
| CubeCL CPU feature | Test execution (`--features cpu`) | ✓ | 0.9.x | — |
| `cargo test` / `cargo nextest` | Test runner | cargo test ✓, nextest not found | stable | Use `cargo test` |
| vendored libcint (oracle) | Oracle parity tests | ✓ | 6.1.3 (in crates/cintx-oracle/build.rs) | — |
| `approx` crate | Unit test assertions | ✓ | 0.5 | — |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:** `cargo nextest` not installed; use `cargo test` instead.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | cargo test (built-in) |
| Config file | none (uses workspace Cargo.toml) |
| Quick run command | `cargo test -p cintx-cubecl --features cpu -- c2s` |
| Full suite command | `cargo test -p cintx-cubecl --features cpu && cargo test -p cintx-oracle --features cpu` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| KERN-01 | Overlap kernel produces non-zero output for s-s H pair | unit | `cargo test -p cintx-cubecl --features cpu -- ovlp` | ❌ Wave 0 |
| KERN-01 | Kinetic kernel value matches nabla-squared formula | unit | `cargo test -p cintx-cubecl --features cpu -- kinetic` | ❌ Wave 0 |
| KERN-01 | Nuclear attraction kernel value non-zero, Boys-based | unit | `cargo test -p cintx-cubecl --features cpu -- nuclear` | ❌ Wave 0 |
| KERN-06 | C2S matrix for l=0..4 matches libcint coefficients | unit | `cargo test -p cintx-cubecl -- c2s_coeff` | ❌ Wave 0 |
| KERN-06 | D-shell (l=2) transform maps 6 cart -> 5 sph correctly | unit | `cargo test -p cintx-cubecl -- c2s_d_shell` | ❌ Wave 0 |
| VERI-05 | int1e_ovlp_sph H2O STO-3G matches libcint atol=1e-11 | oracle | `cargo test -p cintx-oracle --features cpu -- ovlp_parity` | ❌ Wave 0 |
| VERI-05 | int1e_kin_sph H2O STO-3G matches libcint atol=1e-11 | oracle | `cargo test -p cintx-oracle --features cpu -- kin_parity` | ❌ Wave 0 |
| VERI-05 | int1e_nuc_sph H2O STO-3G matches libcint atol=1e-11 | oracle | `cargo test -p cintx-oracle --features cpu -- nuc_parity` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p cintx-cubecl --features cpu 2>&1 | tail -5`
- **Per wave merge:** `cargo test -p cintx-cubecl --features cpu && cargo test -p cintx-oracle --features cpu`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `crates/cintx-cubecl/tests/one_electron_tests.rs` — covers KERN-01 (operator post-processing)
- [ ] `crates/cintx-cubecl/tests/c2s_tests.rs` — covers KERN-06 (coefficient matrix correctness)
- [ ] Oracle parity test for 1e sph family in `crates/cintx-oracle/` — covers VERI-05

---

## Sources

### Primary (HIGH confidence)
- `libcint-master/src/g1e.c` — G-tensor fill algorithm (verified lines 125-320)
- `libcint-master/src/cint1e.c` — make_g1e_gout operator dispatch (lines 284-304), int1e_ovlp/nuc drivers (lines 360-436)
- `libcint-master/src/autocode/intor1.c` — CINTgout1e_int1e_kin, nabla-squared formula (lines 18-46)
- `libcint-master/src/cart2sph.c` — g_trans_cart2sph[] coefficients (lines 21-308), g_c2s[] offset table (lines 3561-3578), a_ket_cart2spheric / a_bra_cart2spheric DGEMM pattern (lines 3581-3596)
- `crates/cintx-cubecl/src/math/obara_saika.rs` — vrr_step_host, hrr_step_host (Phase 8, verified)
- `crates/cintx-cubecl/src/math/pdata.rs` — compute_pdata_host (Phase 8, verified)
- `crates/cintx-cubecl/src/math/boys.rs` — boys_gamma_inc_host (Phase 8, verified)
- `crates/cintx-oracle/src/compare.rs` — TOL_1E_ATOL, TOL_1E_RTOL constants (lines 21-22)

### Secondary (MEDIUM confidence)
- `crates/cintx-cubecl/tests/math_integration_tests.rs` — established integration test pattern for host-wrapper-only approach

---

## Project Constraints (from CLAUDE.md)

- **Compatibility:** Target upstream libcint 6.1.3 result compatibility. Integral values must match within atol 1e-11 / rtol 1e-9 for the H2O STO-3G oracle gate.
- **Architecture:** CubeCL is the primary compute backend. For Phase 9 CPU path, host-side `*_host()` wrappers are used; `#[cube(launch)]` is reserved for GPU paths.
- **Error Handling:** Public library errors use `thiserror` v2 (`cintxRsError`). No `anyhow` in the kernel crate.
- **API Surface:** Changes confined to `cintx-cubecl` (kernel + transform) and `cintx-oracle` (parity test). No changes to `cintx-core`, `cintx-runtime`, or `cintx-compat`.
- **Verification:** Oracle parity CI gate must pass with `mismatch_count == 0` under `--features cpu`.
- **Artifacts:** Parity report artifacts must be written to `/mnt/data` (see `REQUIRED_REPORT_ARTIFACT` in `fixtures.rs`).
- **GSD Workflow:** All file changes must flow through GSD phase execution workflow.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries are workspace-pinned and verified
- Architecture: HIGH — libcint source code read directly, Phase 8 patterns verified in tests
- Pitfalls: HIGH — derived from direct source-code analysis of libcint reference implementation
- Coefficient values: HIGH — extracted directly from `cart2sph.c` `g_trans_cart2sph[]` array

**Research date:** 2026-04-03
**Valid until:** 2026-05-03 (libcint 6.1.3 is pinned vendor copy; coefficients don't change)
