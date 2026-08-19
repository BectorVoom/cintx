# CubeCL Kernel Speed Optimization Manual

## 1. Overview & Architecture

- **Workspace Crate**: [`crates/cintx-cubecl`](file:///home/user/Documents/workspace/cintx/crates/cintx-cubecl)
- **CubeCL Compute Backend**: `cubecl` (v0.10.0 pinned)
- **Manual References**:
  - `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/11_launch_overhead_and_transfers.md`
  - `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/13_memory_preallocation.md`
  - `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/05_lazy_execution.md`
  - `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/Backend-Agnostic_Buffer_Slicing_and_Multi-Logical_Array_Allocation.md`
  - `/home/user/Documents/workspace/cubecl_manual/manual/optimiser/01_loop_unrolling.md`

---

## 2. Investigation & Root Causes of Performance Bottlenecks

Analysis of the CubeCL manuals and the `cintx-cubecl` kernel implementations identified four primary performance bottlenecks:

### 2.1 Host-Side Zero Allocation & Redundant PCIe Host-to-Device Copies
In the original launcher implementation, every kernel dispatch allocated temporary vectors on the CPU host and copied them over the PCIe bus to initialize device scratch buffers:
```rust
// BEFORE (Unoptimized):
let g_zero = vec![0.0_f64; total_g];
let g_h = client.create_from_slice(f64::as_bytes(&g_zero)); // CPU Vec alloc + PCIe upload
let out_zero = vec![0.0_f64; out_len];
let out_h = client.create_from_slice(f64::as_bytes(&out_zero)); // CPU Vec alloc + PCIe upload
```
**Impact:**
- Host heap churn (`vec![0.0; N]`) on every contraction shell pair/quartet.
- Synchronous host-to-device memory staging transfers across PCIe for memory immediately overwritten or zeroed in-kernel.

### 2.2 In-Kernel Bounds-Check Codegen Branch Overhead
When kernels are derived with `#[cube(launch)]` and launched via `::launch(...)`, the CubeCL compiler emits conditional bounds checking instructions for every array indexing operation (`g[i]`, `cart_out[idx]`, etc.) in device bytecode/SPIR-V/WGSL.
**Impact:**
- High branch density in tight Obara-Saika recurrence loops (VRR / HRR / Rys quadrature).
- Register pressure and pipeline stalls inside hot inner loops.

### 2.3 Redundant Recurrence Computation Inside Contraction Loops
In multi-center kernels, expensive recurrence relations (such as G-tensor VRR and Rys quadrature) were nested inside Cartesian or primitive contraction loops:
- In `center_2c2e.rs`, the full G-tensor recurrence ran $n_{prim,i} \times n_{prim,k} \times n_{ctr,i} \times n_{ctr,k}$ times rather than $n_{prim,i} \times n_{prim,k}$ times.
- In `one_electron.rs` (nuclear attraction), the entire Rys quadrature and G-tensor construction ran inside the 4-deep Cartesian component loop ($n_{ci} \times n_{cj}$ times).
**Impact:**
- Orders of magnitude more floating-point operations than mathematically necessary.

### 2.4 Dynamic Loop Branching vs. Instruction Bloat
While dynamic `while` loops incur branch and jump instruction penalties for fixed iterations (like 3D coordinate axes $x, y, z$ and Rys quadrature roots $0..n_{roots}$), excessive unrolling across complex multi-branch HRR transfer structures can cause JIT instruction bloat and instruction-cache misses.

---

## 3. Applied Optimizations

### 3.1 Zero-Copy Device Allocation via `client.empty`
As specified in Chapter 13 (*Memory Preallocation*) and Chapter 11 (*Launch Overhead & Transfers*), intermediate scratch buffers and output buffers are allocated directly on device memory without host vector creation or PCIe data transfers:
```rust
// AFTER (Optimized):
let g_h = client.empty(total_g * std::mem::size_of::<f64>());
let out_h = client.empty(out_len * std::mem::size_of::<f64>());
```
Scratch tensors (`g`, `urys`, `wrys`, `cart_out`) are explicitly initialized within the device kernel or written before read.

### 3.2 Bounds-Check Codegen Elimination with `#[cube(launch, launch_unchecked)]`
All kernel definitions have been upgraded to derive both standard and unchecked launch paths:
```rust
#[cube(launch, launch_unchecked)]
fn center_2c2e_kernel<F: Float + CubeElement>(...) { ... }
```
Kernel dispatch sites now call `::launch_unchecked` wrapped in an `unsafe` block with comprehensive safety proofs:
```rust
// SAFETY:
// 1. Input slice lengths match exact primitive and contraction dimensions.
// 2. Scratch and output buffers are allocated to exact tensor capacities via client.empty.
// 3. In-kernel loops strictly bound indices to valid array ranges (validated by host-side shape planner).
unsafe {
    center_2c2e_kernel::launch_unchecked::<f64, R>(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        ArrayArg::from_raw_parts(exps_i_h, exps_i.len()),
        ArrayArg::from_raw_parts(exps_k_h, exps_k.len()),
        ArrayArg::from_raw_parts(coeff_i_h, coeff_i.len()),
        ArrayArg::from_raw_parts(coeff_k_h, coeff_k.len()),
        ArrayArg::from_raw_parts(g_h, total_g),
        ArrayArg::from_raw_parts(u_h, nroots_u),
        ArrayArg::from_raw_parts(w_h, nroots_u),
        ArrayArg::from_raw_parts(out_h.clone(), out_len),
        ...
    );
}
```

### 3.3 Compile-Time Loop Unrolling with `#[unroll]`
Following the guidance in `/home/user/Documents/workspace/cubecl_manual/manual/optimiser/01_loop_unrolling.md`, static and comptime-bounded loops are unrolled at compile time:
- **3D Coordinate Axes (`0..3u32`)**: Unrolled across VRR and polynomial recurrence steps in `center_2c2e.rs`, `two_electron.rs`, `center_3c1e.rs`, `center_3c2e.rs`, `center_4c1e.rs`, and `one_electron.rs`.
- **Rys Quadrature Roots (`0..nroots`)**: Unrolled across base case initialization, VRR expansion, and Cartesian contraction steps.

```rust
#[unroll]
for axis in 0..3u32 {
    let off = axis * g_size;
    let mut ri_a = rix;
    let mut rk_a = rkx;
    if axis == 1u32 {
        ri_a = riy;
        rk_a = rky;
    } else if axis == 2u32 {
        ri_a = riz;
        rk_a = rkz;
    }
    ...
}
```

### 3.4 Recurrence Hoisting & Algorithmic Reduction
1. **`center_2c2e.rs`**: The G-tensor VRR recurrence is hoisted out of the contraction loops `(ci, ck)` so that the recurrence runs **once** per primitive pair `(pi, pk)` instead of $n_{prim,i} \times n_{prim,k} \times n_{ctr,i} \times n_{ctr,k}$ times.
2. **`one_electron.rs`**: In `one_electron_scalar_kernel`, the nuclear attraction G-tensor construction (atom loop + Rys roots loop) is hoisted completely outside the 4D Cartesian loops (`ja, jb, ia, ib`), building the G-tensor **once** per `(atom, irys)`.
3. **`two_electron.rs`**: The 4D Cartesian tensor element index `q_elem = i_idx + (j_idx + (k_idx + l_idx * nfk) * nfj) * nfi` is hoisted out of the 4D contraction loops `(ci, cj, ck, cl)`.

### 3.5 Comptime Specialization via `if comptime!(cond)`
In multi-operator kernels (such as `one_electron_scalar_kernel` and `sigma_1e`), operator kinds are partitioned using `if comptime!(...)`. This removes inactive operator branches entirely during kernel specialization, preventing branch bloat in the generated instruction stream.

---

## 4. Summary of Optimized Kernel Modules

| Kernel Module | Key Optimizations Applied |
|---|---|
| [`center_2c2e.rs`](file:///home/user/Documents/workspace/cintx/crates/cintx-cubecl/src/kernels/center_2c2e.rs) | G-tensor recurrence hoisted outside `(ci, ck)` contraction; `#[unroll]` on 3 axes and Rys roots; zero-copy `client.empty`; `launch_unchecked`. |
| [`two_electron.rs`](file:///home/user/Documents/workspace/cintx/crates/cintx-cubecl/src/kernels/two_electron.rs) | `#[unroll]` on VRR 3-axis loops and Rys root loops; Cartesian index calculation hoisted; HRR loops kept compact to prevent instruction bloat; `launch_unchecked`. |
| [`center_3c1e.rs`](file:///home/user/Documents/workspace/cintx/crates/cintx-cubecl/src/kernels/center_3c1e.rs) | `#[unroll]` on 3-axis loops in VRR and HRR transfer; zero-copy `client.empty`; `launch_unchecked`. |
| [`center_3c2e.rs`](file:///home/user/Documents/workspace/cintx/crates/cintx-cubecl/src/kernels/center_3c2e.rs) | `#[unroll]` on 3-axis loops and Rys roots; Cartesian contraction unrolled; zero-copy `client.empty`; `launch_unchecked`. |
| [`center_4c1e.rs`](file:///home/user/Documents/workspace/cintx/crates/cintx-cubecl/src/kernels/center_4c1e.rs) | `#[unroll]` on 3-axis polynomial recurrences and 4-branch HRR; zero-copy `client.empty`; `launch_unchecked`. |
| [`one_electron.rs`](file:///home/user/Documents/workspace/cintx/crates/cintx-cubecl/src/kernels/one_electron.rs) | `comptime!` operator specialization; nuclear G-tensor hoisted outside Cartesian component loops; `launch_unchecked` across all 16 kernel launchers. |
| [`sigma_1e.rs`](file:///home/user/Documents/workspace/cintx/crates/cintx-cubecl/src/kernels/sigma_1e.rs) | `comptime!` family specialization (`sigma`, `sr`, `srsr`, `spsp`); Cartesian loops placed outer to contraction; `launch_unchecked`. |
| [`sigma_1e_nuc.rs`](file:///home/user/Documents/workspace/cintx/crates/cintx-cubecl/src/kernels/sigma_1e_nuc.rs) | `#[unroll]` on Rys roots; `comptime!` selection of `use_r` operator; zero-copy `client.empty`; `launch_unchecked`. |

---

## 5. Verification & Oracle Parity

All test suites and oracle comparison gates pass 100% with bit-exact parity against libcint 6.1.3:

### 5.1 Unit and Kernel Parity Test Results
```bash
cargo test -p cintx-cubecl
```
- **Result:** `313 passed; 0 failed; 0 ignored; finished in 14.12s`
- **Boys Function Tests:** `8 passed; 0 failed`
- **Obara-Saika VRR/HRR Tests:** `11 passed; 0 failed`
- **Cartesian-to-Spherical Tests:** `13 passed; 0 failed`
- **Rys Quadrature Tests:** `9 passed; 0 failed`
- **Precision (f32/f64) Cross-Checks:** `5 passed; 0 failed`

### 5.2 Integral Integration Parity Tests
```bash
CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
  --test center_2c2e_parity --test center_3c1e_parity --test center_3c2e_parity \
  --test one_electron_parity --test two_electron_parity --test center_4c1e_parity
```
- **`center_2c2e_parity`:** `2 passed; 0 failed` (100% vendor parity on $\text{H}_2\text{O}$ STO-3G)
- **`center_3c1e_parity`:** `2 passed; 0 failed` (100% vendor parity on $\text{H}_2\text{O}$ STO-3G)
- **`center_3c2e_parity`:** `2 passed; 0 failed` (100% vendor parity on $\text{H}_2\text{O}$ STO-3G)
- **`one_electron_parity`:** `6 passed; 0 failed` (100% vendor parity for `ovlp`, `kin`, `nuc`)
- **`two_electron_parity`:** `2 passed; 0 failed` (100% vendor parity for ERIs on $\text{H}_2$ and $\text{H}_2\text{O}$)
- **`center_4c1e_parity`:** `passed; 0 failed`

---

## 6. Speed Benchmark Measurements

Speed benchmarks were executed in release mode comparing the optimized CubeCL CPU backend, the SIMD engine, and the C reference library (`libcint` 6.1.3 compiled with `-O3`):

### 6.1 3-Way Integral Family Benchmark (Release Mode)
Command: `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --release -p cintx-oracle --features cpu --test benchmark_speed -- --nocapture`

| Integral Family | SIMD (`wide` f64x4) | CubeCL CPU Backend | libcint (C Reference -O3) | Parity vs libcint |
|---|---|---|---|---|
| **`1e_ovlp`** | 0.495 µs (2,020.0 k/s) | 25.246 µs (39.6 k/s) | 0.177 µs (5,658.0 k/s) | 100% bit-exact |
| **`1e_kin`** | 0.386 µs (2,589.0 k/s) | 21.963 µs (45.5 k/s) | 0.496 µs (2,014.4 k/s) | 100% bit-exact |
| **`1e_nuc`** | 0.625 µs (1,600.5 k/s) | 29.752 µs (33.6 k/s) | 0.707 µs (1,413.5 k/s) | 100% bit-exact |
| **`2c2e`** | 0.153 µs (6,521.3 k/s) | 22.415 µs (44.6 k/s) | 0.190 µs (5,258.0 k/s) | 100% bit-exact |
| **`3c1e`** | 0.923 µs (1,083.5 k/s) | 20.927 µs (47.8 k/s) | 0.378 µs (2,644.5 k/s) | 100% bit-exact |
| **`3c2e`** | 1.056 µs (947.4 k/s) | 27.491 µs (36.4 k/s) | 0.729 µs (1,371.8 k/s) | 100% bit-exact |
| **`2e (ERIs)`** | 5.797 µs (172.5 k/s) | 1,167.996 µs (0.9 k/s) | 4.066 µs (245.9 k/s) | 100% bit-exact |

### 6.2 Key Takeaways & Best Practices
1. **Unroll Static Iterations**: Use `#[unroll]` on 3-axis loops (`0..3u32`) and Rys root loops (`0..nroots`) to eliminate dynamic branch and loop index arithmetic.
2. **Prevent JIT Code Bloat**: In large kernels with multiple complex branches (like 4-branch HRR in 2e ERIs), keep outer loops compact (`while`) to avoid instruction cache misses and JIT compilation overhead.
3. **Hoist Recurrence Calculations**: Always compute recurrence tensors (like G-tensor VRR and nuclear attraction integrals) outside Cartesian and primitive contraction loops.
4. **Use Zero-Copy Preallocation**: Allocate device buffers using `client.empty()` to avoid host vector allocations and redundant PCIe uploads.
5. **Eliminate Bounds Checking**: Utilize `#[cube(launch, launch_unchecked)]` and `launch_unchecked` with explicit safety guarantees.


