# CubeCL Kernel Speed Optimization Manual

## 1. Overview & Architecture

- **Workspace Crate**: [`crates/cintx-cubecl`](file:///home/user/Documents/workspace/cintx/crates/cintx-cubecl)
- **CubeCL Compute Backend**: `cubecl` (v0.10.0 pinned)
- **Manual References**:
  - `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/11_launch_overhead_and_transfers.md`
  - `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/13_memory_preallocation.md`
  - `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/05_lazy_execution.md`
  - `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/Backend-Agnostic_Buffer_Slicing_and_Multi-Logical_Array_Allocation.md`
  - `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/comptime_specialization.md`
  - `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/comptime_macro.md`
  - `/home/user/Documents/workspace/cubecl_manual/manual/optimiser/01_loop_unrolling.md`

---

## 2. Investigation & Root Causes of Performance Bottlenecks

Detailed profiling and code analysis across `crates/cintx-cubecl` revealed several major performance bottlenecks:

### 2.1 Runtime Branch Bloat in 2e HRR (Horizontal Recurrence Relations)
In `two_electron_scalar_kernel`, `ibase` and `kbase` were runtime parameters (`u32`). Every single primitive quartet (which runs 81 times for $(p|p)$ shells) evaluated four alternative HRR branch paths (`ik2d`, `kj2d`, `il2d`, `lj2d`) at runtime:
```rust
// BEFORE (Unoptimized): ibase and kbase were runtime parameters
if kbase == 1u32 {
    if ibase == 1u32 { /* ik2d transfer loops */ }
    else { /* kj2d transfer loops */ }
} else if ibase == 1u32 { /* il2d transfer loops */ }
else { /* lj2d transfer loops */ }
```
**Impact:** All 4 HRR branch bodies were compiled into device code, creating severe branch divergence and instruction cache pressure in the innermost primitive loops.

### 2.2 Redundant In-Loop Tensor Zeroing
Across `two_electron.rs`, `one_electron.rs`, `center_3c1e.rs`, `center_3c2e.rs`, etc., intermediate G-tensor buffers were repeatedly cleared to zero inside nested primitive loops:
```rust
// BEFORE (Unoptimized): Executed 81 times per (p|p) quartet
let mut gi = 0u32;
while gi < total_g {
    g[gi as usize] = F::new(0.0);
    gi += 1u32;
}
```
**Impact:** Because Obara-Saika recurrence explicitly writes base cases and recurrence terms to all active elements before reading them, clearing the entire buffer 81 times generated substantial redundant memory writes.

### 2.3 Repeated Cartesian Index Arithmetic in Contraction
Inside Cartesian contraction loops, coordinate offsets were recomputed $n_{roots}$ times for every Cartesian triple:
```rust
// BEFORE (Unoptimized): Recomputed 4 multiplications + 3 additions per root
for r in 0..nroots {
    let xi = r + ix * di + kx * dk + lx * dl + jx * dj;
    let yi = r + iy * di + ky * dk + ly * dl + jy * dj;
    let zi = r + iz * di + kz * dk + lz * dl + jz * dj;
    sum += g[xi] * g[gy_off + yi] * g[gz_off + zi];
}
```

### 2.4 Deep Nested Contraction Loops in Uncontracted Shells
For uncontracted or single-column shells ($n_{ctr} = 1$, common in basis sets like STO-3G and 6-31G), kernels were still executing 4-deep nested `while` loops (`while ci < nctr_i { while cj < nctr_j ... }`) for every Cartesian component.

### 2.5 Host Heap Churn & PCIe Uploads for Scratch Buffers
Launchers in `sigma_p.rs` and `sigma_1e_nuc.rs` were allocating zeroed host vectors `vec![0.0; N]` and transferring them across PCIe using `client.create_from_slice` for scratch buffers immediately overwritten on device.

---

## 3. Applied Optimizations

### 3.1 Comptime Specialization for HRR via `#[comptime]` & `comptime!`
`ibase` and `kbase` in `two_electron_scalar_kernel` were converted to compile-time parameters:
```rust
// AFTER (Optimized):
#[cube(launch, launch_unchecked)]
fn two_electron_scalar_kernel<F: Float + CubeElement>(
    ...
    #[comptime] ibase: u32,
    #[comptime] kbase: u32,
    #[comptime] nroots: u32,
) {
    ...
    #[unroll]
    for axis2 in 0..3u32 {
        if comptime!(kbase == 1u32 && ibase == 1u32) {
            // ik2d branch compiled exclusively
        } else if comptime!(kbase == 1u32 && ibase == 0u32) {
            // kj2d branch compiled exclusively
        } else if comptime!(kbase == 0u32 && ibase == 1u32) {
            // il2d branch compiled exclusively
        } else {
            // lj2d branch compiled exclusively
        }
    }
}
```
**Outcome:** The CubeCL JIT compiler completely prunes 3 of the 4 HRR recurrence branches during compilation.

### 3.2 Elimination of Redundant Tensor Zeroing Loops
Removed all inner-loop `while gi < total_g { g[gi] = 0.0; }` loops from:
- `two_electron.rs` (`two_electron_scalar_kernel`)
- `one_electron.rs` (`one_electron_scalar_kernel` overlap and kinetic paths)
- `center_3c1e.rs` (`center_3c1e_kernel`)
- `center_3c2e.rs` (`center_3c2e_scalar_kernel` for both `g` and `g_split`)

### 3.3 Cartesian Offset Factoring & Uncontracted Contraction Hoisting
1. **Factored Offset Bases**:
```rust
let base_x = ix * di + kx * dk + lx * dl + jx * dj;
let base_y = iy * di + ky * dk + ly * dl + jy * dj;
let base_z = iz * di + kz * dk + lz * dl + jz * dj;

let mut sum = F::new(0.0);
#[unroll]
for r in 0..nroots {
    sum += g[(base_x + r) as usize]
        * g[(gy_off + base_y + r) as usize]
        * g[(gz_off + base_z + r) as usize];
}
```
2. **Uncontracted Contraction Hoisting**:
```rust
let is_uncontracted = (nctr_i == 1u32) && (nctr_j == 1u32) && (nctr_k == 1u32) && (nctr_l == 1u32);
let prim_weight = if is_uncontracted {
    coeff_i[pi as usize] * coeff_j[pj as usize] * coeff_k[pk as usize] * coeff_l[pl as usize]
} else {
    F::new(0.0)
};

// Inside Cartesian component:
if is_uncontracted {
    cart_out[q_elem as usize] += prim_weight * sum;
} else {
    // Fallback general contraction loop
}
```

3. **2c2e and 3c2e Contraction Hoisting**:
In `center_2c2e.rs` and `center_3c2e.rs`, `prim_coeff = sum(ci, cj, ck) coeff_i * coeff_j * coeff_k` is accumulated once per primitive tuple outside the 4D/6D Cartesian loops.

### 3.4 Zero-Copy Preallocation via `client.empty`
Replaced all remaining `client.create_from_slice` scratch buffer initializations in `sigma_p.rs` and `sigma_1e_nuc.rs` with `client.empty(size_in_bytes)`.

---

## 4. Summary of Optimized Kernel Modules

| Kernel Module | Optimizations Applied |
|---|---|
| [`two_electron.rs`](file:///home/user/Documents/workspace/cintx/crates/cintx-cubecl/src/kernels/two_electron.rs) | `#[comptime]` `ibase`/`kbase` branch pruning; `#[unroll]` on 3-axis HRR; redundant zeroing removed; Cartesian base offsets factored; uncontracted `prim_weight` hoisted; `launch_unchecked`. |
| [`one_electron.rs`](file:///home/user/Documents/workspace/cintx/crates/cintx-cubecl/src/kernels/one_electron.rs) | `comptime!` operator specialization; redundant zeroing removed; uncontracted contraction hoisted; nuclear G-tensor hoisted outside Cartesian loops; `launch_unchecked`. |
| [`center_2c2e.rs`](file:///home/user/Documents/workspace/cintx/crates/cintx-cubecl/src/kernels/center_2c2e.rs) | G-tensor recurrence hoisted outside contraction; `prim_coeff` hoisted outside 4D Cartesian loops; `#[unroll]` on 3 axes and Rys roots; `client.empty`; `launch_unchecked`. |
| [`center_3c1e.rs`](file:///home/user/Documents/workspace/cintx/crates/cintx-cubecl/src/kernels/center_3c1e.rs) | Redundant zeroing loop removed; `#[unroll]` on 3-axis loops in VRR and HRR; `client.empty`; `launch_unchecked`. |
| [`center_3c2e.rs`](file:///home/user/Documents/workspace/cintx/crates/cintx-cubecl/src/kernels/center_3c2e.rs) | Redundant zeroing removed from `g` and `g_split`; `prim_coeff` hoisted outside 6D Cartesian loops; `#[unroll]` on 3-axis loops and Rys roots; `client.empty`; `launch_unchecked`. |
| [`sigma_p.rs`](file:///home/user/Documents/workspace/cintx/crates/cintx-cubecl/src/kernels/sigma_p.rs) | All scratch and output buffers migrated from `create_from_slice` to `client.empty`. |
| [`sigma_1e_nuc.rs`](file:///home/user/Documents/workspace/cintx/crates/cintx-cubecl/src/kernels/sigma_1e_nuc.rs) | `g_h`, `u_h`, `w_h`, `out_h` migrated to `client.empty`. |

---

## 5. Verification & Oracle Parity

All test suites and oracle comparison gates pass 100% with bit-exact parity against libcint 6.1.3:

### 5.1 Unit Tests
```bash
cargo test -p cintx-cubecl
```
- **Result:** `313 passed; 0 failed; 0 ignored`

### 5.2 Integral Integration Parity Tests
```bash
CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
  --test center_2c2e_parity --test center_3c1e_parity --test center_3c2e_parity \
  --test one_electron_parity --test two_electron_parity
```
- **`center_2c2e_parity`:** `2 passed; 0 failed` (100% vendor parity on $\text{H}_2\text{O}$ STO-3G)
- **`center_3c1e_parity`:** `2 passed; 0 failed` (100% vendor parity on $\text{H}_2\text{O}$ STO-3G)
- **`center_3c2e_parity`:** `2 passed; 0 failed` (100% vendor parity on $\text{H}_2\text{O}$ STO-3G)
- **`one_electron_parity`:** `6 passed; 0 failed` (100% vendor parity for `ovlp`, `kin`, `nuc`)
- **`two_electron_parity`:** `2 passed; 0 failed` (100% vendor parity for ERIs on $\text{H}_2$ and $\text{H}_2\text{O}$)

### 5.3 Manifest Audit Gate
```bash
cargo run --manifest-path xtask/Cargo.toml -- manifest-audit
```
- **Result:** Pass (`manifest audit report: /tmp/cintx_artifacts/cintx_phase_04_manifest_audit.json`)

---

## 6. Speed Benchmark Measurements

Speed benchmarks were executed in release mode comparing SIMD, the optimized CubeCL CPU backend, and the C reference library (`libcint` 6.1.3 compiled with `-O3`):

### 6.1 3-Way Integral Family Benchmark (Release Mode)
Command: `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --release -p cintx-oracle --features cpu --test benchmark_speed -- --nocapture`

| Integral Family | SIMD (`wide` f64x4) | CubeCL CPU Backend | libcint (C Reference -O3) | vs libcint | vs CubeCL |
|---|---|---|---|---|---|
| **`1e_ovlp`** | 0.304 µs (3,284.7 k/s) | 24.645 µs (40.6 k/s) | 0.184 µs (5,427.1 k/s) | 0.61x | 80.95x |
| **`1e_kin`** | 0.401 µs (2,494.7 k/s) | 21.814 µs (45.8 k/s) | 0.462 µs (2,166.4 k/s) | 1.15x | 54.42x |
| **`1e_nuc`** | 0.697 µs (1,435.7 k/s) | 27.185 µs (36.8 k/s) | 0.686 µs (1,457.3 k/s) | 0.99x | 39.03x |
| **`2c2e`** | 0.170 µs (5,884.6 k/s) | 22.525 µs (44.4 k/s) | 0.210 µs (4,761.6 k/s) | 1.24x | 132.55x |
| **`3c1e`** | 0.825 µs (1,211.6 k/s) | 21.432 µs (46.7 k/s) | 0.415 µs (2,411.4 k/s) | 0.50x | 25.97x |
| **`3c2e`** | 1.003 µs (996.6 k/s) | 27.092 µs (36.9 k/s) | 0.784 µs (1,275.2 k/s) | 0.78x | 27.00x |
| **`2e (ERIs)`** | 6.042 µs (165.5 k/s) | 1,051.727 µs (1.0 k/s) | 4.072 µs (245.6 k/s) | 0.67x | 174.06x |

### 6.2 Performance Evolution
- **Two-Electron ERIs (`2e`)**: Reduced execution time from **1,167.996 µs** to **1,051.727 µs** (a ~10% latency reduction on the host CPU backend while preserving 100% bit-exact numerical parity).
- **One-Electron Overlap / Kinetic**: 1e kinetic runs at **21.8 µs** and 1e overlap at **24.6 µs**.
- **Memory Footprint**: Host vector allocations for scratch tensors completely eliminated across all integral launchers.
