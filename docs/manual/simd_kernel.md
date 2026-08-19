# SIMD-Vectorized Molecular Integral Kernel Manual

## 1. Overview & Architecture

- **Target SIMD Crate**: `wide` (v1.6.1)
- **Vector Math Crate**: `rmath` (via `/home/user/Documents/workspace/rmath`)
- **Code Graph Exploration**: `codegraph` MCP server
- **Workspace Crate**: [`crates/cintx-simd`](file:///home/user/Documents/workspace/cintx/crates/cintx-simd)

### Implemented Kernel Families
1. **1-Electron Integrals** ([`SimdOneElectronKernel`](file:///home/user/Documents/workspace/cintx/crates/cintx-simd/src/kernels/one_electron.rs)):
   - `eval_ovlp`: 2-center 1-electron overlap integrals $\langle i | j \rangle$.
   - `eval_kin`: 2-center 1-electron kinetic energy integrals $\langle i | -\frac{1}{2}\nabla^2 | j \rangle$.
   - `eval_nuc`: 2-center nuclear attraction integrals with point-charge Coulomb potentials $\langle i | \sum_A \frac{-Z_A}{|\mathbf{r} - \mathbf{R}_A|} | j \rangle$.
2. **2-Center 2-Electron Integrals** ([`SimdCenter2c2eKernel`](file:///home/user/Documents/workspace/cintx/crates/cintx-simd/src/kernels/center_2c2e.rs)):
   - `eval`: $(i|k)$ electron repulsion integrals with 2D Rys VRR grid evaluation.
3. **3-Center 1-Electron Integrals** ([`SimdCenter3c1eKernel`](file:///home/user/Documents/workspace/cintx/crates/cintx-simd/src/kernels/center_3c1e.rs)):
   - `eval`: 3-center 1-electron Gaussian product overlap integrals $\langle i | O_k | j \rangle = \int \phi_i(\mathbf{r}) \phi_j(\mathbf{r}) \phi_k(\mathbf{r}) d\mathbf{r}$.
4. **3-Center 2-Electron Integrals** ([`SimdCenter3c2eKernel`](file:///home/user/Documents/workspace/cintx/crates/cintx-simd/src/kernels/center_3c2e.rs)):
   - `eval`: 3-center 2-electron Coulomb integrals $(ij|k) = \iint \phi_i(\mathbf{r}_1)\phi_j(\mathbf{r}_1) \frac{1}{|\mathbf{r}_1 - \mathbf{r}_2|} \phi_k(\mathbf{r}_2) d\mathbf{r}_1 d\mathbf{r}_2$.
5. **4-Center 1-Electron Integrals** ([`SimdCenter4c1eKernel`](file:///home/user/Documents/workspace/cintx/crates/cintx-simd/src/kernels/center_4c1e.rs)):
   - `eval`: 4-center 1-electron product overlap integrals $(ij|kl)_{1e} = \int \phi_i(\mathbf{r})\phi_j(\mathbf{r})\phi_k(\mathbf{r})\phi_l(\mathbf{r}) d\mathbf{r}$.
6. **4-Center 2-Electron Integrals** ([`SimdTwoElectronKernel`](file:///home/user/Documents/workspace/cintx/crates/cintx-simd/src/kernels/two_electron.rs)):
   - `eval`: 4-center 2-electron Coulomb integrals $(ij|kl) = \iint \phi_i(\mathbf{r}_1)\phi_j(\mathbf{r}_1) \frac{1}{|\mathbf{r}_1 - \mathbf{r}_2|} \phi_k(\mathbf{r}_2)\phi_l(\mathbf{r}_2) d\mathbf{r}_1 d\mathbf{r}_2$.

---

## 2. SIMD Acceleration & `rmath` Integration

All kernels are fully generic over `V: SimdFloat` where `V` is any of:
- `wide::f64x4`, `wide::f64x2`, `f64` (double precision)
- `wide::f32x4`, `wide::f32x8`, `f32` (single precision)

All mathematical functions are provided by [`rmath`](file:///home/user/Documents/workspace/rmath):
- `rmath::exp`, `rmath::sqrt`, `rmath::ln`, `rmath::erf`, `rmath::erfc`, `rmath::sin`, `rmath::cos`, `rmath::lgamma`, `rmath::pow`, `rmath::j0`, `rmath::exp10`

---

## 3. Mathematical Recurrence Details

### 3.1 3-Center 2-Electron Integrals ($(ij|k)$)
- **Gaussian Bra Pair**: $\zeta = \alpha_i + \alpha_j$, $\mathbf{P} = (\alpha_i \mathbf{R}_i + \alpha_j \mathbf{R}_j) / \zeta$.
- **Gaussian Ket Center**: $\mathbf{R}_k$, exponent $\alpha_k$.
- **Rys Parameter**: $x_{rys} = \frac{\zeta \alpha_k}{\zeta + \alpha_k} |\mathbf{P} - \mathbf{R}_k|^2$.
- **2D VRR**: Evaluates coupled $G(n, m)$ grid for combined $n \in 0..=(l_i + l_j)$ and ket $m \in 0..=l_k$.
- **1D HRR**: Transfers $l_j$ angular momentum units along bra $j$ with displacement $\mathbf{R}_i - \mathbf{R}_j$.

### 3.2 3-Center 1-Electron Integrals ($(i|O_k|j)$)
- **Gaussian Center**: $\mathbf{W} = (\alpha_i \mathbf{R}_i + \alpha_j \mathbf{R}_j + \alpha_k \mathbf{R}_k) / (\alpha_i + \alpha_j + \alpha_k)$.
- **Overlap Prefactor**: $\left(\frac{\pi}{\zeta}\right)^{3/2} \exp\left(-\frac{\alpha_i \alpha_j R_{ij}^2 + \alpha_i \alpha_k R_{ik}^2 + \alpha_j \alpha_k R_{jk}^2}{\zeta}\right)$.
- **1D VRR**: $g[0] = 1, g[1] = \mathbf{W} - \mathbf{R}_i, g[n+1] = (\mathbf{W} - \mathbf{R}_i) g[n] + \frac{n}{2\zeta} g[n-1]$ for $n \in 0..(l_i + l_j + l_k)$.
- **2-Stage HRR**:
  1. Transfer $(j+k)$ from $i$ using $(\mathbf{R}_i - \mathbf{R}_j)$.
  2. Transfer $k$ from $j$ using $(\mathbf{R}_j - \mathbf{R}_k)$.

### 3.3 4-Center 1-Electron Integrals ($(ij|kl)_{1e}$)
- **Gaussian Center**: $\mathbf{W} = (\alpha_i \mathbf{R}_i + \alpha_j \mathbf{R}_j + \alpha_k \mathbf{R}_k + \alpha_l \mathbf{R}_l) / \zeta_{ijkl}$.
- **1D VRR**: $g[n+1] = (\mathbf{W} - \mathbf{R}_i) g[n] + \frac{n}{2\zeta} g[n-1]$ for $n \in 0..(l_i + l_j + l_k + l_l)$.
- **3-Stage HRR**:
  1. Transfer $(j+k+l)$ from $i$ using $(\mathbf{R}_i - \mathbf{R}_j)$.
  2. Transfer $(k+l)$ from $j$ using $(\mathbf{R}_j - \mathbf{R}_k)$.
  3. Transfer $l$ from $k$ using $(\mathbf{R}_k - \mathbf{R}_l)$.

---

## 4. Verification and Test Suites

Run test commands:
```bash
cargo test -p cintx-simd
cargo clippy -p cintx-simd
```

### Test Suites (17/17 Passing)
- `test_ovlp_ss_analytical_and_simd_match`: PASS
- `test_kin_ss_analytical_and_simd_match`: PASS
- `test_nuc_ss_scalar_and_simd_match`: PASS
- `test_sp_angular_momentum_parity`: PASS
- `test_pp_contracted_multiprimitive_simd_parity`: PASS
- `test_dd_higher_angular_momentum_parity`: PASS
- `test_f32_precision_simd_kernel`: PASS
- `test_rys_root1_and_root2_identities`: PASS
- `test_center_2c2e_ss_simd_match`: PASS
- `test_center_2c2e_pp_contracted_simd_parity`: PASS
- `test_center_3c2e_sss_simd_match`: PASS
- `test_center_3c2e_pss_angular_momentum_parity`: PASS
- `test_center_3c1e_sss_analytical_and_simd_match`: PASS
- `test_center_4c1e_ssss_analytical_and_simd_match`: PASS
- `test_two_electron_ssss_simd_match`: PASS
- `test_two_electron_psss_angular_momentum_parity`: PASS
- `test_rmath_vectorized_math_apis`: PASS
