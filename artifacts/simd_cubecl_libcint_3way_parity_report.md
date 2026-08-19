# 3-Way Parity Report: SIMD-Kernel vs. CubeCL-Kernel vs. libcint

## 1. Objective and Scope
- **Objective**: Conduct a 3-way oracle parity comparison across `simd-kernel` (`cintx-simd`), `cubecl-kernel` (`cintx-cubecl` / `cintx-compat`), and reference `libcint` (vendored 6.1.3 C library).
- **Scope**:
  - 1-Electron Overlap (`int1e_ovlp_cart`)
  - 1-Electron Kinetic Energy (`int1e_kin_cart`)
  - 1-Electron Nuclear Attraction (`int1e_nuc_cart`)
  - 2-Center 2-Electron Coulomb Repulsion (`int2c2e_cart`)
  - 3-Center 1-Electron Product Overlap (`int3c1e_cart`)
  - 3-Center 2-Electron Coulomb Repulsion (`int3c2e_cart`)
  - 4-Center 2-Electron Coulomb Repulsion (`int2e_cart`)
- **Dataset**: Real molecular basis data: $\text{H}_2\text{O}$ STO-3G (Oxygen $1s, 2s, 2p$, Hydrogen $1s$ centers) with multi-primitive contractions.

---

## 2. Specification Sources & Tools Applied

- **Specification Source**:
  - Upstream `libcint` 6.1.3 reference implementation (`g1e.c`, `g2e.c`, `g2c2e.c`, `g3c1e.c`, `g3c2e.c`, `cart2sph.c`).
  - cintx design and oracle tolerance specification (D-06).
- **Selected Mandatory Tools**:
  - `cargo test --test simd_cubecl_libcint_3way_parity -p cintx-oracle --features cpu` with `CINTX_ORACLE_BUILD_VENDOR=1`.
  - `cargo test -p cintx-simd`.
  - `approx` for numerical tolerance assertions (`assert_relative_eq!`).
- **Conditional Tools & Rationale**:
  - `cintx-oracle::vendor_ffi`: Direct FFI binding to vendored `libcint 6.1.3` compiled via `cc`.
  - `cintx-compat::raw::eval_raw`: Raw symbol dispatch driving `cintx-cubecl` GPU/CPU kernels.
  - `cintx-simd`: Direct generic SIMD evaluation using `wide` and `rmath`.

---

## 3. Results Summary

| Integral Family | Shell Combinations | Output Tensor Channels | Parity Result | Tolerance | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`int1e_ovlp_cart`** | All $\text{H}_2\text{O}$ pairs ($5 \times 5 = 25$) | $s-s, s-p, p-s, p-p$ | `simd == cubecl == libcint` | atol $10^{-12}$ | **PASS** |
| **`int1e_kin_cart`** | All $\text{H}_2\text{O}$ pairs ($5 \times 5 = 25$) | $s-s, s-p, p-s, p-p$ | `simd == cubecl == libcint` | atol $10^{-12}$ | **PASS** |
| **`int1e_nuc_cart`** | All $\text{H}_2\text{O}$ pairs ($5 \times 5 = 25$) | 3 nuclear centers (O, H1, H2) | `simd == cubecl == libcint` | atol $10^{-10}$ | **PASS** |
| **`int2c2e_cart`** | All $\text{H}_2\text{O}$ pairs ($5 \times 5 = 25$) | $(s\vert s), (s\vert p), (p\vert s), (p\vert p)$ | `simd == cubecl == libcint` | atol $10^{-10}$ | **PASS** |
| **`int3c1e_cart`** | Representative triples ($4 \times 4 \times 4 = 64$) | $\langle i \vert O_k \vert j \rangle$ | `simd == cubecl == libcint` | atol $10^{-10}$ | **PASS** |
| **`int3c2e_cart`** | Representative triples ($4 \times 4 \times 4 = 64$) | $(ij\vert k)$ with $s, p$ shells | `simd == cubecl == libcint` | atol $10^{-9}$ | **PASS** |
| **`int2e_cart`** | Quartets ($s, s, s, s$, $s, p, s, s$, $p, p, s, s$, mixed) | $(ij\vert kl)$ 4-center ERIs | `simd == cubecl == libcint` | atol $10^{-10}$ | **PASS** |

---

## 4. Statements of Assurance & Residual Risk

- **Verified in scope:**
  - Bit-exact / high-precision numerical parity between `cintx-simd` (with `rmath` and `wide`), `cintx-cubecl` (CubeCL CPU backend), and upstream `libcint` (6.1.3 C oracle) across 1e, 2c2e, 3c1e, 3c2e, and 2e Cartesian integral families on real contracted multi-center STO-3G molecular geometries.
- **Not yet verified:**
  - High angular momentum shells ($l \ge 4$, e.g., $g, h, i$ shells) which require higher order Rys roots ($N_{roots} > 4$).
  - Spinor 4-component relativistic transforms for 3c2e/2e on SIMD backend.
