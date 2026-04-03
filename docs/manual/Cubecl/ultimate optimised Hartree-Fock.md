```rust
//! Design overview (memory-efficient HF, improved ERI lookup):
//! - Baseline: ERIs stored densely in `PackedEri` (packed upper-triangular pq|rs), then
//!   converted to sparse (eri_idx, eri_val). GPU kernel previously linearly scanned nnz.
//! - Current refactor:
//!   1) PackedEri is consumed via `into_sparse_sorted`, returning (pair_count, idx, val)
//!      and dropping the dense buffer immediately (“early free”).
//!   2) CPU hash map no longer duplicates values: HashMap<u64, usize> maps quartet index
//!      -> position in shared `Arc<Vec<f64>>` (eri_val). Values live once, shared by CPU+GPU.
//!   3) GPU and CPU paths share the same `eri_val`; GPU uses binary search over sorted idx.
//! - Ownership: `Arc<Vec<f64>>` holds ERI values; CPU hash holds indices; GPU buffers
//!   are created from slices of the same `Arc` to avoid copying data.

use anyhow::Result;
use cubecl_core as cubecl;
use cubecl_core::{Runtime, cube, prelude::*};
use cubecl_cpu::{CpuDevice, CpuRuntime};
use cubecl_runtime::client::ComputeClient;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};
use std::collections::HashMap;
use std::sync::Arc;

// ---------- Packed indexing ----------

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

// ---------- Packed ERI (consuming to sparse) ----------

struct PackedEri {
    n: usize,
    pair_count: usize,
    data: Vec<f64>,
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
        let k = quartet_index(self.pair_count, pq, rs);
        self.data[k] = v;
    }

    /// Consume self, emit sorted sparse (idx,val) and free dense storage.
    fn into_sparse_sorted(self, tol: f64) -> (usize, Vec<u64>, Arc<Vec<f64>>) {
        let mut idx = Vec::new();
        let mut val = Vec::new();
        for (i, &v) in self.data.iter().enumerate() {
            if v.abs() >= tol {
                idx.push(i as u64);
                val.push(v);
            }
        }
        (self.pair_count, idx, Arc::new(val))
    }
}

// ---------- Hash without duplicating values ----------

struct EriHash {
    map: HashMap<u64, usize>,
    val: Arc<Vec<f64>>,
}

impl EriHash {
    fn new(idx: &[u64], val: Arc<Vec<f64>>) -> Self {
        let mut map = HashMap::with_capacity(idx.len());
        for (i, &k) in idx.iter().enumerate() {
            map.insert(k, i);
        }
        Self { map, val }
    }
    #[inline(always)]
    fn get(&self, k: u64) -> f64 {
        match self.map.get(&k) {
            Some(&i) => self.val[i],
            None => 0.0,
        }
    }
}

// ---------- Binary search helper ----------

#[inline(always)]
fn eri_lookup_binary(idx: u64, eri_idx: &[u64], eri_val: &[f64]) -> f64 {
    match eri_idx.binary_search(&idx) {
        Ok(p) => eri_val[p],
        Err(_) => 0.0,
    }
}

// ---------- CubeCL kernel: Fock with binary-search ERI ----------

#[cube(launch_unchecked)]
fn fock_kernel_bs(
    h: &Array<f64>,
    p: &Array<f64>,
    eri_idx: &Array<u64>,
    eri_val: &Array<f64>,
    f_out: &mut Array<f64>,
    #[comptime] n: u64,
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

    let mut lambda = 0u64;
    while lambda < n {
        let mut sigma = 0u64;
        while sigma < n {
            let p_ls = p[lambda * n + sigma];

            // pq
            let (mut a_pq, mut b_pq) = (mu, nu);
            if a_pq > b_pq {
                let t = a_pq;
                a_pq = b_pq;
                b_pq = t;
            }
            let pq = a_pq * n - a_pq * (a_pq + 1) / 2 + (b_pq - a_pq);
            // rs coul
            let (mut a_rs, mut b_rs) = (lambda, sigma);
            if a_rs > b_rs {
                let t = a_rs;
                a_rs = b_rs;
                b_rs = t;
            }
            let rs_c = a_rs * n - a_rs * (a_rs + 1) / 2 + (b_rs - a_rs);
            let (mut x_c, mut y_c) = (pq, rs_c);
            if x_c > y_c {
                let t = x_c;
                x_c = y_c;
                y_c = t;
            }
            let idx_c = x_c * pc - x_c * (x_c + 1) / 2 + (y_c - x_c);

            // binary search idx_c
            let mut lo = 0u64;
            let mut hi = eri_idx.len() as u64;
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
            if lo < eri_idx.len() as u64 && eri_idx[lo] == idx_c {
                eri_coul = eri_val[lo];
            }

            // exchange
            let (mut a_px, mut b_px) = (mu, lambda);
            if a_px > b_px {
                let t = a_px;
                a_px = b_px;
                b_px = t;
            }
            let pq_ex = a_px * n - a_px * (a_px + 1) / 2 + (b_px - a_px);

            let (mut a_rx, mut b_rx) = (nu, sigma);
            if a_rx > b_rx {
                let t = a_rx;
                a_rx = b_rx;
                b_rx = t;
            }
            let rs_ex = a_rx * n - a_rx * (a_rx + 1) / 2 + (b_rx - a_rx);
            let (mut x_e, mut y_e) = (pq_ex, rs_ex);
            if x_e > y_e {
                let t = x_e;
                x_e = y_e;
                y_e = t;
            }
            let idx_e = x_e * pc - x_e * (x_e + 1) / 2 + (y_e - x_e);

            let mut lo2 = 0u64;
            let mut hi2 = eri_idx.len() as u64;
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
            if lo2 < eri_idx.len() as u64 && eri_idx[lo2] == idx_e {
                eri_ex = eri_val[lo2];
            }

            acc += p_ls * (2.0 * eri_coul - eri_ex);
            sigma += 1;
        }
        lambda += 1;
    }

    f_out[flat] = acc;
}

// ---------- CPU Fock builders ----------

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
                    let idx_c = quartet_index(pair_count, pq, rs_c) as u64;
                    let pq_ex = pair_index(n, mu, la);
                    let rs_ex = pair_index(n, nu, si);
                    let idx_e = quartet_index(pair_count, pq_ex, rs_ex) as u64;
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
    eri_idx: &[u64],
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
                    let idx_c = quartet_index(pair_count, pq, rs_c) as u64;
                    let pq_ex = pair_index(n, mu, la);
                    let rs_ex = pair_index(n, nu, si);
                    let idx_e = quartet_index(pair_count, pq_ex, rs_ex) as u64;
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

// ---------- GPU Fock (binary search kernel) ----------

fn build_fock_gpu<R: Runtime>(
    client: &ComputeClient<R::Server>,
    h: &[f64],
    p: &[f64],
    eri_idx: &[u64],
    eri_val: &Arc<Vec<f64>>,
    n: usize,
) -> Result<Vec<f64>> {
    let elem = core::mem::size_of::<f64>();
    let h_handle = client.create(f64::as_bytes(h));
    let p_handle = client.create(f64::as_bytes(p));
    let idx_handle = client.create(u64::as_bytes(eri_idx));
    let val_handle = client.create(f64::as_bytes(eri_val.as_ref()));
    let f_handle = client.empty(n * n * elem);

    let cube_dim = CubeDim::new_1d((n * n) as u64);
    let cube_count = CubeCount::new_1d(1);

    unsafe {
        fock_kernel_bs::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<f64>(&h_handle, n * n, 1),
            ArrayArg::from_raw_parts::<f64>(&p_handle, n * n, 1),
            ArrayArg::from_raw_parts::<u64>(&idx_handle, eri_idx.len(), 1),
            ArrayArg::from_raw_parts::<f64>(&val_handle, eri_val.len(), 1),
            ArrayArg::from_raw_parts::<f64>(&f_handle, n * n, 1),
            n as u64,
        );
    }

    let mut f = f64::from_bytes(&client.read_one(f_handle)).to_vec();
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
            h[idx(n, i, j)] = if i == j {
                1.0 + 0.3 * i as f64
            } else {
                0.05 * (i + j) as f64
            };
            p[idx(n, i, j)] = if i == j {
                0.6 - 0.05 * i as f64
            } else {
                0.02 * (i + j + 1) as f64
            };
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
    let (h, p, eri_packed) = demo_system(n);
    let (pair_count, eri_idx, eri_val_arc) = eri_packed.into_sparse_sorted(1e-6);

    let eri_hash = EriHash::new(&eri_idx, eri_val_arc.clone());

    // CPU references
    let f_cpu_hash = build_fock_cpu_hash(&h, &p, &eri_hash, n, pair_count);
    let f_cpu_bs = build_fock_cpu_binsrch(&h, &p, &eri_idx, eri_val_arc.as_ref(), n, pair_count);

    // GPU (or CPU fallback) using shared eri_val
    let try_wgpu = wgpu_supports_f64();
    let (f_gpu, backend) = if try_wgpu {
        let client: ComputeClient<_> = WgpuRuntime::client(&WgpuDevice::DefaultDevice);
        (
            build_fock_gpu::<WgpuRuntime>(&client, &h, &p, &eri_idx, &eri_val_arc, n)?,
            "wgpu",
        )
    } else {
        // Pure CPU fallback using the same sparse data (no kernel launch).
        let f = build_fock_cpu_binsrch(&h, &p, &eri_idx, eri_val_arc.as_ref(), n, pair_count);
        (f, "cpu-fallback")
    };

    let diff_hash_bs = max_abs_diff(&f_cpu_hash, &f_cpu_bs);
    let diff_hash_gpu = max_abs_diff(&f_cpu_hash, &f_gpu);

    println!("Backend used: {backend}");
    println!(
        "n={n}, nnz={}, full_quartets={}",
        eri_idx.len(),
        pair_count * (pair_count + 1) / 2
    );
    println!("Max |F_cpu(hash) - F_cpu(bs)| = {:.3e}", diff_hash_bs);
    println!("Max |F_cpu(hash) - F_gpu|     = {:.3e}", diff_hash_gpu);
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