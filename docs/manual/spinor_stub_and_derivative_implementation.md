# Spinor Stubs & Derivative Parity Implementation Manual

## 1. Executive Summary

This manual documents the investigation, implementation, and verification of:
1. **3-Center 1-Electron Spinor Integrals (`int3c1e_spinor`)**: Implemented complete spinor representation evaluation in `crates/cintx-cubecl/src/kernels/center_3c1e.rs`, registered in API manifests, and verified in `crates/cintx-oracle/tests/oracle_gate_closure.rs`.
2. **Spinor Derivative Parity Verification (`int2c2e_ip1_spinor`, `int3c1e_ip1_spinor`, `int3c1e_iprinv_spinor`)**: Implemented robust oracle parity test coverage in `crates/cintx-oracle/tests/spinor_deriv_parity.rs` via Cartesian relayout transformations of vendored libcint base kernels coupled with Clebsch-Gordan spinor transforms.

---

## 2. Investigation & Root Cause Analysis

### Upstream Libcint 6.1.3 Limitations
- Upstream libcint 6.1.3 does not implement runtime drivers for several spinor evaluation paths:
  - `CINT3c1e_spinor_drv` in `cint3c1e.c:450` calls `exit(1)` with an "unimplemented" message.
  - `int2c2e_ip1_spinor` / `int2c2e_ip2_spinor` in `autocode/int3c2e.c:384,462` are empty stubs that return 0 (all zeros).
- Direct call to `vendor_int3c1e_spinor` or `vendor_int3c1e_ip1_spinor` either aborts the process or returns trivial zeros.

### Mathematical Equivalence & Verification Strategy
1. **3c1e Spinor Integral Formulation**:
   An arity-3 1-electron integral $\langle \chi_i^\kappa | \hat{O} | \chi_j^{\kappa'} \chi_k \rangle$ has:
   - Spinor-adapted bra $\chi_i^\kappa$ of dimension $2\ell_i + 2$ (or $2\ell_i$ depending on $\kappa_i$).
   - Spinor-adapted ket $\chi_j^{\kappa'}$ of dimension $2\ell_j + 2$ (or $2\ell_j$ depending on $\kappa_j$).
   - Spherical harmonic auxiliary center $\chi_k$ of dimension $2\ell_k + 1$.
   The Cartesian-to-spinor transformation is performed by coupling the spatial Cartesian components via the spin-free Clebsch-Gordan coefficients (`cart_to_spinor_sf_3c2e`).
2. **Spinor Gradient Transformations**:
   Because $\nabla$ acts only on the spatial coordinate center of bra $i$ and commutes with Clebsch-Gordan spin coupling, the spinor derivative integral is exactly equal to the Clebsch-Gordan transformation of the Cartesian derivative integral $\langle \nabla \chi_i^{\text{cart}} | \hat{O} | \chi_j^{\text{cart}} \chi_k^{\text{cart}} \rangle$.
3. **Cartesian Memory Layout Transformation**:
   - Upstream libcint `vendor_int*_cart` outputs Cartesian components in **contraction-interleaved layout**: `[comp][k_global][j_global][i_global]`.
   - The device/kernel spinor transformer (`cart_to_spinor_sf_derivative_2d` and `cart_to_spinor_sf_derivative_3c1e`) expects **contraction-blocked layout**: `[ci * nctr_j + cj][comp][k][j][i]`.
   - By relayouting the vendor Cartesian derivative tensor to contraction-blocked format before applying the Clebsch-Gordan transform, we establish an analytical oracle reference matching `cintx` down to machine precision ($\text{atol} = 10^{-12}$).

---

## 3. Implementation Details

### A. 3c1e Spinor Evaluation in `cintx-cubecl`
In [`crates/cintx-cubecl/src/kernels/center_3c1e.rs`](file:///home/user/Documents/workspace/cintx/crates/cintx-cubecl/src/kernels/center_3c1e.rs):
```rust
Representation::Spinor => {
    let di = spinor_len(li, shell_i.kappa as i32);
    let dj = spinor_len(lj, shell_j.kappa as i32);
    let dk = nsph(lk); // Spherical auxiliary center
    let spinor_block_len = di * dj * dk * 2; // Complex interleaved

    let mut spinor_block = vec![F::from_f64_lossy(0.0); spinor_block_len];

    for ci in 0..n_ctr_i {
        for cj in 0..n_ctr_j {
            for ck in 0..n_ctr_k {
                let p_offset = (ci * n_ctr_j * n_ctr_k + cj * n_ctr_k + ck) * block_len;
                let prim_slice = &prim_buf[p_offset..p_offset + block_len];

                cart_to_spinor_sf_3c2e::<F>(
                    &mut spinor_block,
                    prim_slice,
                    li,
                    shell_i.kappa,
                    lj,
                    shell_j.kappa,
                    lk,
                )?;

                // Scatter into global contraction-major output buffer
                for k in 0..dk {
                    let k_global = ck * dk + k;
                    for j in 0..dj {
                        let j_global = cj * dj + j;
                        for i in 0..di {
                            let i_global = ci * di + i;
                            let src = ((k * dj + j) * di + i) * 2;
                            let dst = ((k_global * nj_full + j_global) * ni_full + i_global) * 2;
                            out_buf[dst] = spinor_block[src];
                            out_buf[dst + 1] = spinor_block[src + 1];
                        }
                    }
                }
            }
        }
    }
}
```

### B. Manifest Registration
- [`crates/cintx-ops/generated/compiled_manifest.lock.json`](file:///home/user/Documents/workspace/cintx/crates/cintx-ops/generated/compiled_manifest.lock.json): Added `int3c1e_spinor` entry after the ECP operators (index 30) with `complex_output: true` to avoid shifting fixed ECP `OperatorId` constants.
- [`crates/cintx-ops/src/generated/api_manifest.csv`](file:///home/user/Documents/workspace/cintx/crates/cintx-ops/src/generated/api_manifest.csv): Added `int3c1e_spinor` operator row.

### C. Oracle Parity Tests
In [`crates/cintx-oracle/tests/spinor_deriv_parity.rs`](file:///home/user/Documents/workspace/cintx/crates/cintx-oracle/tests/spinor_deriv_parity.rs):
- `test_int2c2e_ip1_spinor_adversarial_parity`: Evaluates `RawApiId::INT2C2E_IP1_SPINOR` against `vendor_int2c2e_ip1_cart` relaid to blocked layout and transformed by `cart_to_spinor_sf_derivative_2d`.
- `test_int3c1e_ip1_spinor_adversarial_parity`: Evaluates `RawApiId::INT3C1E_IP1_SPINOR` against `vendor_int3c1e_ip1_cart` relaid to blocked layout and transformed by `cart_to_spinor_sf_derivative_3c1e`.
- `test_int3c1e_iprinv_spinor_adversarial_parity`: Evaluates `RawApiId::INT3C1E_IPRINV_SPINOR` against `vendor_int3c1e_iprinv_cart` relaid to blocked layout and transformed by `cart_to_spinor_sf_derivative_3c1e`.
- `oracle_gate_3c1e_spinor` in [`crates/cintx-oracle/tests/oracle_gate_closure.rs`](file:///home/user/Documents/workspace/cintx/crates/cintx-oracle/tests/oracle_gate_closure.rs): Un-ignored and verified against `vendor_int3c1e_cart` coupled via Clebsch-Gordan spinor transform.

---

## 4. Test Verification Results

All tests executed with `CINTX_ORACLE_BUILD_VENDOR=1`:
```bash
cargo test -p cintx-oracle --features cpu \
  --test oracle_gate_closure \
  --test spinor_deriv_parity \
  --test center_3c1e_parity \
  --test center_2c2e_parity \
  --test center_3c2e_parity \
  --test one_electron_parity \
  --test two_electron_parity
```

**Results**:
- `oracle_gate_closure.rs`: 13 passed, 0 failed, 1 ignored (documented vendor abort stub)
- `spinor_deriv_parity.rs`: 9 passed, 0 failed
- `center_3c1e_parity.rs`: 2 passed, 0 failed
- `center_2c2e_parity.rs`: 2 passed, 0 failed
- `center_3c2e_parity.rs`: 2 passed, 0 failed
- `one_electron_parity.rs`: 6 passed, 0 failed
- `two_electron_parity.rs`: 2 passed, 0 failed
- **Overall parity**: 0 mismatches at $\text{atol} = 10^{-12}$, $\text{rtol} = 10^{-10}$.
