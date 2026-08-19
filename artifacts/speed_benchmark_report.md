# 3-Way Performance Benchmark: SIMD-Kernel vs. CubeCL-Kernel vs. libcint

## 1. Benchmark Setup & Environment
- **Compilation Mode**: `rustc` release profile (`opt-level = 3`, LTO, target-cpu native).
- **Dataset**: $\text{H}_2\text{O}$ STO-3G contracted molecular basis set.
- **Hardware Architecture**: x86_64 Linux, SIMD vector registers enabled via `wide` (AVX2/FMA/SSE2).
- **Execution Engines**:
  1. `simd-kernel` (`cintx-simd`): Native vectorized SIMD kernel with `wide::f64x4` + `rmath`.
  2. `cubecl-kernel` (`cintx-cubecl`): CubeCL host CPU compute kernel via `eval_raw`.
  3. `libcint` (`vendor_ffi`): Upstream C library 6.1.3 compiled with `cc -O3`.

---

## 2. Benchmark Results Table

| Integral Family | SIMD Latency (Throughput) | CubeCL Latency (Throughput) | libcint Latency (Throughput) | Speedup vs. libcint | Speedup vs. CubeCL |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`1e_ovlp`** | **0.283 µs** (3,538.5 k/s) | 29.746 µs (33.6 k/s) | 0.186 µs (5,374.0 k/s) | 0.66x | **105.26x** |
| **`1e_kin`** | **0.387 µs** (2,582.6 k/s) | 28.751 µs (34.8 k/s) | 0.504 µs (1,982.5 k/s) | **1.30x (Faster)** | **74.25x** |
| **`1e_nuc`** | **0.693 µs** (1,442.8 k/s) | 36.148 µs (27.7 k/s) | 0.681 µs (1,467.6 k/s) | **0.98x (Parity)** | **52.15x** |
| **`2c2e`** | **0.172 µs** (5,806.9 k/s) | 26.066 µs (38.4 k/s) | 0.201 µs (4,984.6 k/s) | **1.16x (Faster)** | **151.36x** |
| **`3c1e`** | **0.836 µs** (1,196.5 k/s) | 25.952 µs (38.5 k/s) | 0.422 µs (2,367.4 k/s) | 0.51x | **31.05x** |
| **`3c2e`** | **1.101 µs** (908.3 k/s) | 34.006 µs (29.4 k/s) | 0.837 µs (1,194.7 k/s) | 0.76x | **30.89x** |
| **`2e (ERIs)`** | **6.732 µs** (148.5 k/s) | 1,124.077 µs (0.9 k/s) | 5.802 µs (172.3 k/s) | 0.86x | **166.96x** |

---

## 3. Analysis & Key Insights

1. **SIMD vs. libcint (C)**:
   - **`1e_kin` (Kinetic)**: SIMD is **30% faster** than libcint due to vectorized 2nd-derivative accumulation on registers.
   - **`2c2e` (2-Center 2-Electron)**: SIMD is **16% faster** than libcint, achieving **5.81 Million integrals/sec**.
   - **`1e_nuc` (Nuclear Attraction)**: SIMD reaches **near-identical speed** (0.98x) with libcint across 3 nuclear Coulomb centers.
   - **`2e` (4-Center ERIs)**: SIMD executes in **6.7 µs per quartet** (86% of libcint's highly tuned C implementation), processing **148,500 quartets/sec** purely in safe Rust.

2. **SIMD vs. CubeCL CPU Engine**:
   - `simd-kernel` is **30x to 167x faster** than running individual shell pairs on the CubeCL CPU backend.
   - This occurs because CubeCL's host CPU engine involves tensor marshaling, memory allocation, and client-server dispatch overhead for small single-shell-pair workloads, whereas `cintx-simd` runs stack-allocated, register-packed SIMD instructions directly in CPU L1 cache.
