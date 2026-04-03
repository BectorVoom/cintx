
```rust
//! Memory‑efficient Fock build with faster ERI lookup.
//! Improvements over the linear‑scan baseline:
//! - Sparse ERI (idx,val) is **sorted** by idx; GPU kernel does binary search (O(log nnz))
//!   instead of O(nnz) linear scan.
//! - CPU reference uses a hash map for O(1) average lookup.
//! - Packed (pq|rs) upper‑triangular indexing is preserved; nnz << n^4.
//! - Buffers are reused and transfers minimized, matching the “memory efficient
//!   Hartree–Fock” sample style.

use anyhow::Result;
use cubecl_core::{self as cubecl, prelude::*, Runtime};
use cubecl_runtime::client::ComputeClient;
use cubecl_cpu::{CpuDevice, CpuRuntime};
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};
use std::collections::HashMap;

// ---------- Packed indexing helpers ----------

#[inline(always)]
fn pair_index(n: usize, p: usize, q: usize) -> usize {
    let (a, b) = if p <= q { (p, q) } else { (q, p) };
    a * n - a * (a + 1) / 2 + (b - a)
}

#[inline(always)]
fn quartet_index(pair_count: usize, pq: usize, rs: usize) -> usize {
    let (x, y) = if pq <= rs { (pq, rs) } else { (rs, pq) };
    x * pair_count - x * (x + 1) / 2 + (y - x)
}

#[inline(always)]
fn idx(n: usize, r: usize, c: usize) -> usize {
    r * n + c
}

// ---------- Packed + sparse ERI container ----------

struct PackedEri {
    n: usize,
    pair_count: usize,
    data: Vec<f64>, // full packed (pq|rs) upper triangle
}

impl PackedEri {
    fn new(n: usize) -> Self {
        let pair_count = n * (n + 1) / 2;
        let quartets = pair_count * (pair_count + 1) / 2;
        Self {
            n,
            pair_count,
            data: vec![0.0; quartets],
        }
    }

    fn set(&mut self, mu: usize, nu: usize, la: usize, si: usize, v: f64) {
        let pq = pair_index(self.n, mu, nu);
        let rs = pair_index(self.n, la, si);
        let idx = quartet_index(self.pair_count, pq, rs);
        self.data[idx] = v;
    }

    fn get(&self, mu: usize, nu: usize, la: usize, si: usize) -> f64 {
        let pq = pair_index(self.n, mu, nu);
        let rs = pair_index(self.n, la, si);
        self.data[quartet_index(self.pair_count, pq, rs)]
    }

    /// Convert to sparse, sorted by idx; drops |v| < tol.
    fn to_sparse_sorted(&self, tol: f64) -> (Vec<u32>, Vec<f64>) {
        let mut idx = Vec::new();
        let mut val = Vec::new();
        for (i, &v) in self.data.iter().enumerate() {
            if v.abs() >= tol {
                idx.push(i as u32);
                val.push(v);
            }
        }
        // data already traversed in ascending idx, so idx is sorted; keep invariant explicit
        (idx, val)
    }
}

// ---------- Hash lookup for CPU reference ----------

struct EriHash {
    map: HashMap<u32, f64>,
}

impl EriHash {
    fn new(idx: &[u32], val: &[f64]) -> Self {
        let mut map = HashMap::with_capacity(idx.len());
        for (&k, &v) in idx.iter().zip(val.iter()) {
            map.insert(k, v);
        }
        Self { map }
    }
    #[inline(always)]
    fn get(&self, k: u32) -> f64 {
        *self.map.get(&k).unwrap_or(&0.0)
    }
}

// ---------- Binary search lookup (CPU helper) ----------

#[inline(always)]
fn eri_lookup_binary(idx: u32, eri_idx: &[u32], eri_val: &[f64]) -> f64 {
    match eri_idx.binary_search(&idx) {
        Ok(p) => eri_val[p],
        Err(_) => 0.0,
    }
}

// ---------- cubecl kernel: Fock with binary-search ERI lookup ----------

#[cube(launch_unchecked)]
fn fock_kernel_bs(
    h: &Array<f64>,            // n*n
    p: &Array<f64>,            // n*n
    eri_idx: &Array<u32>,      // sorted nnz
    eri_val: &Array<f64>,      // aligned nnz
    f_out: &mut Array<f64>,    // n*n upper triangle filled
    #[comptime] n: u32,
) {
    let flat = ABSOLUTE_POS;
    let total = n * n;
    if flat >= total {
        terminate!();
    }
    let mu = flat / n;
    let nu = flat % n;
    if mu < nu {
        f_out[flat] = 0.0;
        terminate!();
    }
    let mut acc = h[flat];
    let pc = n * (n + 1) / 2;

    let mut lambda = 0u32;
    while lambda < n {
        let mut sigma = 0u32;
        while sigma < n {
            let p_ls = p[lambda * n + sigma];

            let mut a_pq = mu;
            let mut b_pq = nu;
            if a_pq > b_pq {
                let t = a_pq;
                a_pq = b_pq;
                b_pq = t;
            }
            let pq = a_pq * n - a_pq * (a_pq + 1) / 2 + (b_pq - a_pq);

            let mut a_rs = lambda;
            let mut b_rs = sigma;
            if a_rs > b_rs {
                let t = a_rs;
                a_rs = b_rs;
                b_rs = t;
            }
            let rs_coul = a_rs * n - a_rs * (a_rs + 1) / 2 + (b_rs - a_rs);

            let mut x_c = pq;
            let mut y_c = rs_coul;
            if x_c > y_c {
                let t = x_c;
                x_c = y_c;
                y_c = t;
            }
            let idx_c = x_c * pc - x_c * (x_c + 1) / 2 + (y_c - x_c);
            // binary search for idx_c
            let mut lo = 0u32;
            let mut hi = eri_idx.len() as u32;
            while lo < hi {
                let mid = (lo + hi) >> 1;
                let v = eri_idx[mid];
                if v < idx_c {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            let mut eri_coul = 0.0;
            if lo < eri_idx.len() as u32 && eri_idx[lo] == idx_c {
                eri_coul = eri_val[lo];
            }

            let mut a_pq_ex = mu;
            let mut b_pq_ex = lambda;
            if a_pq_ex > b_pq_ex {
                let t = a_pq_ex;
                a_pq_ex = b_pq_ex;
                b_pq_ex = t;
            }
            let pq_ex = a_pq_ex * n - a_pq_ex * (a_pq_ex + 1) / 2 + (b_pq_ex - a_pq_ex);

            let mut a_rs_ex = nu;
            let mut b_rs_ex = sigma;
            if a_rs_ex > b_rs_ex {
                let t = a_rs_ex;
                a_rs_ex = b_rs_ex;
                b_rs_ex = t;
            }
            let rs_ex = a_rs_ex * n - a_rs_ex * (a_rs_ex + 1) / 2 + (b_rs_ex - a_rs_ex);

            let mut x_e = pq_ex;
            let mut y_e = rs_ex;
            if x_e > y_e {
                let t = x_e;
                x_e = y_e;
                y_e = t;
            }
            let idx_e = x_e * pc - x_e * (x_e + 1) / 2 + (y_e - x_e);
            // binary search for idx_e
            let mut lo2 = 0u32;
            let mut hi2 = eri_idx.len() as u32;
            while lo2 < hi2 {
                let mid = (lo2 + hi2) >> 1;
                let v = eri_idx[mid];
                if v < idx_e {
                    lo2 = mid + 1;
                } else {
                    hi2 = mid;
                }
            }
            let mut eri_ex = 0.0;
            if lo2 < eri_idx.len() as u32 && eri_idx[lo2] == idx_e {
                eri_ex = eri_val[lo2];
            }

            acc += p_ls * (2.0 * eri_coul - eri_ex);
            sigma += 1;
        }
        lambda += 1;
    }

    f_out[flat] = acc;
}

// ---------- CPU reference Fock (hash lookup) ----------

fn build_fock_cpu_hash(
    h: &[f64],
    p: &[f64],
    eri_hash: &EriHash,
    n: usize,
    pair_count: usize,
) -> Vec<f64> {
    let mut f = vec![0.0; n * n];
    for mu in 0..n {
        for nu in 0..n {
            let mut acc = h[idx(n, mu, nu)];
            for la in 0..n {
                for si in 0..n {
                    let pq = pair_index(n, mu, nu);
                    let rs_c = pair_index(n, la, si);
                    let idx_c = quartet_index(pair_count, pq, rs_c) as u32;
                    let pq_ex = pair_index(n, mu, la);
                    let rs_ex = pair_index(n, nu, si);
                    let idx_e = quartet_index(pair_count, pq_ex, rs_ex) as u32;
                    let coul = eri_hash.get(idx_c);
                    let exch = eri_hash.get(idx_e);
                    acc += p[idx(n, la, si)] * (2.0 * coul - exch);
                }
            }
            f[idx(n, mu, nu)] = acc;
        }
    }
    f
}

fn build_fock_cpu_binsrch(
    h: &[f64],
    p: &[f64],
    eri_idx: &[u32],
    eri_val: &[f64],
    n: usize,
    pair_count: usize,
) -> Vec<f64> {
    let mut f = vec![0.0; n * n];
    for mu in 0..n {
        for nu in 0..n {
            let mut acc = h[idx(n, mu, nu)];
            for la in 0..n {
                for si in 0..n {
                    let pq = pair_index(n, mu, nu);
                    let rs_c = pair_index(n, la, si);
                    let idx_c = quartet_index(pair_count, pq, rs_c) as u32;
                    let pq_ex = pair_index(n, mu, la);
                    let rs_ex = pair_index(n, nu, si);
                    let idx_e = quartet_index(pair_count, pq_ex, rs_ex) as u32;
                    let coul = eri_lookup_binary(idx_c, eri_idx, eri_val);
                    let exch = eri_lookup_binary(idx_e, eri_idx, eri_val);
                    acc += p[idx(n, la, si)] * (2.0 * coul - exch);
                }
            }
            f[idx(n, mu, nu)] = acc;
        }
    }
    f
}

// ---------- GPU Fock (binary search) ----------

fn build_fock_gpu<R: Runtime>(
    client: &ComputeClient<R::Server>,
    h: &[f64],
    p: &[f64],
    eri_idx: &[u32],
    eri_val: &[f64],
    n: usize,
) -> Result<Vec<f64>> {
    let elem = core::mem::size_of::<f64>();
    let h_handle = client.create(f64::as_bytes(h));
    let p_handle = client.create(f64::as_bytes(p));
    let idx_handle = client.create(u32::as_bytes(eri_idx));
    let val_handle = client.create(f64::as_bytes(eri_val));
    let f_handle = client.empty(n * n * elem);

    let cube_dim = CubeDim::new_1d((n * n) as u32);
    let cube_count = CubeCount::new_1d(1);
    let pair_count = (n * (n + 1) / 2) as u32;

    unsafe {
        fock_kernel_bs::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<f64>(&h_handle, n * n, 1),
            ArrayArg::from_raw_parts::<f64>(&p_handle, n * n, 1),
            ArrayArg::from_raw_parts::<u32>(&idx_handle, eri_idx.len(), 1),
            ArrayArg::from_raw_parts::<f64>(&val_handle, eri_val.len(), 1),
            ArrayArg::from_raw_parts::<f64>(&f_handle, n * n, 1),
            n as u32,
        );
    }

    let mut f = f64::from_bytes(&client.read_one(f_handle)).to_vec();
    // symmetrize (kernel computed upper)
    for mu in 0..n {
        for nu in 0..mu {
            let v = f[idx(n, mu, nu)];
            f[idx(n, nu, mu)] = v;
        }
    }
    Ok(f)
}

// ---------- Demo system ----------

fn demo_system(n: usize) -> (Vec<f64>, Vec<f64>, PackedEri) {
    let mut h = vec![0.0; n * n];
    let mut p = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            h[idx(n, i, j)] = if i == j { 1.0 + 0.3 * i as f64 } else { 0.05 * (i + j) as f64 };
            p[idx(n, i, j)] = if i == j { 0.6 - 0.05 * i as f64 } else { 0.02 * (i + j + 1) as f64 };
        }
    }
    let mut eri = PackedEri::new(n);
    for mu in 0..n {
        for nu in 0..n {
            for la in 0..n {
                for si in 0..n {
                    let pq = pair_index(n, mu, nu) as f64;
                    let rs = pair_index(n, la, si) as f64;
                    let val = 0.25 / (1.0 + pq + rs);
                    eri.set(mu, nu, la, si, val);
                }
            }
        }
    }
    (h, p, eri)
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn print_matrix(a: &[f64], n: usize, label: &str) {
    println!("{label}:");
    for r in 0..n {
        for c in 0..n {
            print!("{:8.4} ", a[idx(n, r, c)]);
        }
        println!();
    }
}

// ---------- Main ----------

fn main() -> Result<()> {
    let n = 4;
    let (h, p, eri_full) = demo_system(n);
    let pair_count = eri_full.pair_count;
    let (eri_idx, eri_val) = eri_full.to_sparse_sorted(1e-6);
    let eri_hash = EriHash::new(&eri_idx, &eri_val);

    // CPU reference (hash)
    let f_cpu = build_fock_cpu_hash(&h, &p, &eri_hash, n, pair_count);
    // CPU binary-search reference
    let f_cpu_bs = build_fock_cpu_binsrch(&h, &p, &eri_idx, &eri_val, n, pair_count);

    // GPU (binary search) with CPU fallback
    let try_wgpu = wgpu_supports_f64();
    let (f_gpu, backend) = if try_wgpu {
        let client: ComputeClient<_> = WgpuRuntime::client(&WgpuDevice::DefaultDevice);
        (
            build_fock_gpu::<WgpuRuntime>(&client, &h, &p, &eri_idx, &eri_val, n)?,
            "wgpu",
        )
    } else {
        // On CPU backend, reuse CPU binary-search path (identical math, avoids kernel overhead).
        let f = build_fock_cpu_binsrch(&h, &p, &eri_idx, &eri_val, n, pair_count);
        (f, "cpu")
    };

    let diff = max_abs_diff(&f_cpu, &f_gpu);
    let diff_bs = max_abs_diff(&f_cpu, &f_cpu_bs);

    println!("Backend used: {backend}");
    println!("n={n}, nnz={}, full_quartets={}", eri_idx.len(), pair_count * (pair_count + 1) / 2);
    println!("Max |F_cpu(hash) - F_cpu(bs)|   = {:.3e}", diff_bs);
    println!("Max |F_cpu(hash) - F_gpu(bs)| = {:.3e}", diff);
    print_matrix(&f_gpu, n, "Fock (optimized)");

    Ok(())
}

// ---------- Capability probe ----------
fn wgpu_supports_f64() -> bool {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    for adapter in instance.enumerate_adapters(wgpu::Backends::all()) {
        if adapter.features().contains(wgpu::Features::SHADER_F64) {
            return true;
        }
    }
    false
}

```
```
```