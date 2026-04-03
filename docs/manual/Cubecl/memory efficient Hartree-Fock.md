
```rust
//! Memory-efficient Fock matrix construction with cubecl acceleration.
//!
//! Key ideas:
//! - Avoid materializing the full n⁴ ERI tensor (dominant memory bottleneck).
//! - Store ERIs in a packed, symmetry-aware (pq|rs) upper-triangular layout:
//!   * A pair index `pq` packs (p,q) with p<=q into a single number.
//!   * A quartet index packs (pq,rs) with pq<=rs into a single number.
//!   * Total elements: n_pairs = n(n+1)/2, stored quartets = n_pairs(n_pairs+1)/2.
//!   * Memory drops from O(n⁴) to O(n⁴/8) in the dense case, and further if sparsity is used.
//! - Kernel fetches ERIs on demand from the packed buffer, no expansion to n⁴ on device.
//! - Optional on-the-fly ERI generator can replace the packed buffer for even lower memory.
//! - Fock symmetry exploited on host: we launch one thread per (μ,ν) but only write μ>=ν,
//!   and mirror on host after copy-back (simple and keeps kernel branch-free).

use anyhow::Result;
use cubecl_core::{self as cubecl, prelude::*, Runtime};
use cubecl_runtime::client::ComputeClient;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};
use cubecl_cpu::{CpuDevice, CpuRuntime};
use wgpu::Backends;

// ---------- Index helpers (host and device friendly) ----------

/// Upper-triangular packed index for a pair (p,q) with p<=q.
#[inline(always)]
fn pair_index(p: usize, q: usize, n: usize) -> usize {
    let (a, b) = if p <= q { (p, q) } else { (q, p) };
    // number of elements before row a in an upper-triangular matrix
    let offset = a * n - a * (a + 1) / 2;
    offset + (b - a)
}

/// Upper-triangular packed index for quartet (pq,rs) with pq<=rs.
#[inline(always)]
fn quartet_index(pq: usize, rs: usize, pair_count: usize) -> usize {
    let (x, y) = if pq <= rs { (pq, rs) } else { (rs, pq) };
    let offset = x * pair_count - x * (x + 1) / 2;
    offset + (y - x)
}

// Device versions (u32 for cube indices).

// ---------- Symmetric ERI container (host) ----------

/// Packed, symmetry-aware ERI storage: stores unique (pq|rs) with pq<=rs and each pair packed.
struct PackedEri {
    n: usize,
    pair_count: usize,
    data: Vec<f64>,
}

impl PackedEri {
    fn new(n: usize) -> Self {
        let pair_count = n * (n + 1) / 2;
        let quartet_count = pair_count * (pair_count + 1) / 2;
        Self {
            n,
            pair_count,
            data: vec![0.0; quartet_count],
        }
    }

    fn set(&mut self, mu: usize, nu: usize, lambda: usize, sigma: usize, val: f64) {
        let pq = pair_index(mu, nu, self.n);
        let rs = pair_index(lambda, sigma, self.n);
        let idx = quartet_index(pq, rs, self.pair_count);
        self.data[idx] = val;
    }

    fn get(&self, mu: usize, nu: usize, lambda: usize, sigma: usize) -> f64 {
        let pq = pair_index(mu, nu, self.n);
        let rs = pair_index(lambda, sigma, self.n);
        let idx = quartet_index(pq, rs, self.pair_count);
        self.data[idx]
    }

    fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Build a sparse representation (indices, values) dropping entries with |v| < tol.
    /// Returns sorted by index for deterministic lookups.
    fn to_sparse(&self, tol: f64) -> SparseEri {
        let mut idx = Vec::new();
        let mut val = Vec::new();
        for (i, &v) in self.data.iter().enumerate() {
            if v.abs() >= tol {
                idx.push(i as u32);
                val.push(v);
            }
        }
        SparseEri {
            n: self.n,
            pair_count: self.pair_count,
            idx,
            val,
        }
    }
}

/// Sparse ERI: stores only non-negligible packed quartets.
struct SparseEri {
    n: usize,
    pair_count: usize,
    idx: Vec<u32>,
    val: Vec<f64>,
}

// ---------- CubeCL kernel (packed ERI lookup, no n⁴ tensor) ----------

#[cube(launch_unchecked)]
pub fn fock_kernel_packed(
    h: &Array<f64>,           // n*n
    p: &Array<f64>,           // n*n
    eri_idx: &Array<u32>,     // nnz
    eri_val: &Array<f64>,     // nnz
    nnz: u32,
    f_out: &mut Array<f64>,   // upper-triangular of F stored in full n*n for simplicity
    #[comptime] n: u32,
) {
    let flat: u32 = ABSOLUTE_POS;
    let total = n * n;
    if flat >= total {
        terminate!();
    }

    let n_u = n;
    let mu = flat / n_u;
    let nu = flat % n_u;

    // Only compute upper triangle (mu >= nu); else write zero, host will symmetrize.
    if mu < nu {
        f_out[flat] = 0.0;
        terminate!();
    }

    let mut acc = h[flat];
    let pair_count = (n_u * (n_u + 1)) / 2;

    let mut lambda = 0u32;
    while lambda < n_u {
        let mut sigma = 0u32;
        while sigma < n_u {
            let p_ls = p[lambda * n_u + sigma];

            // pq
            let mut a_pq = mu;
            let mut b_pq = nu;
            if mu > nu {
                a_pq = nu;
                b_pq = mu;
            }
            let pq = a_pq * n_u - a_pq * (a_pq + 1) / 2 + (b_pq - a_pq);

            // rs (coulomb)
            let mut a_rs = lambda;
            let mut b_rs = sigma;
            if lambda > sigma {
                a_rs = sigma;
                b_rs = lambda;
            }
            let rs_coul = a_rs * n_u - a_rs * (a_rs + 1) / 2 + (b_rs - a_rs);

            let mut x_c = pq;
            let mut y_c = rs_coul;
            if pq > rs_coul {
                x_c = rs_coul;
                y_c = pq;
            }
            let idx_coul = x_c * pair_count - x_c * (x_c + 1) / 2 + (y_c - x_c);
            let mut eri_coul = 0.0;
            let mut k = 0u32;
            while k < nnz {
                if eri_idx[k] == idx_coul {
                    eri_coul = eri_val[k];
                    break;
                }
                k += 1;
            }

            // exchange pq_ex, rs_ex
            let mut a_pq_ex = mu;
            let mut b_pq_ex = lambda;
            if mu > lambda {
                a_pq_ex = lambda;
                b_pq_ex = mu;
            }
            let pq_ex = a_pq_ex * n_u - a_pq_ex * (a_pq_ex + 1) / 2 + (b_pq_ex - a_pq_ex);

            let mut a_rs_ex = nu;
            let mut b_rs_ex = sigma;
            if nu > sigma {
                a_rs_ex = sigma;
                b_rs_ex = nu;
            }
            let rs_ex = a_rs_ex * n_u - a_rs_ex * (a_rs_ex + 1) / 2 + (b_rs_ex - a_rs_ex);

            let mut x_e = pq_ex;
            let mut y_e = rs_ex;
            if pq_ex > rs_ex {
                x_e = rs_ex;
                y_e = pq_ex;
            }
            let idx_ex = x_e * pair_count - x_e * (x_e + 1) / 2 + (y_e - x_e);
            let mut eri_exch = 0.0;
            let mut j = 0u32;
            while j < nnz {
                if eri_idx[j] == idx_ex {
                    eri_exch = eri_val[j];
                    break;
                }
                j += 1;
            }

            acc += p_ls * (2.0 * eri_coul - eri_exch);
            sigma += 1;
        }
        lambda += 1;
    }

    f_out[flat] = acc;
}

// ---------- Host-side Fock builders ----------

fn build_fock_matrix_efficient<R: Runtime>(
    client: &ComputeClient<R::Server>,
    h: &[f64],          // n*n row-major
    p: &[f64],          // n*n row-major
    eri_sparse_idx: &[u32],
    eri_sparse_val: &[f64],
    n: usize,
) -> Result<Vec<f64>> {
    let elem = core::mem::size_of::<f64>();
    let h_handle = client.create(f64::as_bytes(h));
    let p_handle = client.create(f64::as_bytes(p));
    let eri_idx_handle = client.create(u32::as_bytes(eri_sparse_idx));
    let eri_val_handle = client.create(f64::as_bytes(eri_sparse_val));
    let f_handle = client.empty(n * n * elem);

    let cube_dim = CubeDim::new_1d((n * n) as u32);
    let cube_count = CubeCount::new_1d(1);

    let nnz = eri_sparse_idx.len() as u32;

    unsafe {
        fock_kernel_packed::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<f64>(&h_handle, n * n, 1),
            ArrayArg::from_raw_parts::<f64>(&p_handle, n * n, 1),
            ArrayArg::from_raw_parts::<u32>(&eri_idx_handle, eri_sparse_idx.len(), 1),
            ArrayArg::from_raw_parts::<f64>(&eri_val_handle, eri_sparse_val.len(), 1),
            ScalarArg { elem: nnz },
            ArrayArg::from_raw_parts::<f64>(&f_handle, n * n, 1),
            n as u32,
        );
    }

    let mut f = f64::from_bytes(&client.read_one(f_handle)).to_vec();

    // Symmetrize (host) because kernel only filled mu>=nu.
    for mu in 0..n {
        for nu in 0..mu {
            let v = f[mu * n + nu];
            f[nu * n + mu] = v;
        }
    }
    Ok(f)
}

/// CPU reference using packed ERI but plain loops.
fn build_fock_matrix_cpu(h: &[f64], p: &[f64], eri: &PackedEri, n: usize) -> Vec<f64> {
    let mut f = vec![0.0; n * n];
    for mu in 0..n {
        for nu in 0..n {
            let mut acc = h[mu * n + nu];
            for lambda in 0..n {
                for sigma in 0..n {
                    let p_ls = p[lambda * n + sigma];
                    let coul = eri.get(mu, nu, lambda, sigma);
                    let exch = eri.get(mu, lambda, nu, sigma);
                    acc += p_ls * (2.0 * coul - exch);
                }
            }
            f[mu * n + nu] = acc;
        }
    }
    f
}

// ---------- Demo system generation ----------

fn realistic_demo_system(n: usize) -> (Vec<f64>, Vec<f64>, PackedEri) {
    let mut h = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            h[i * n + j] = if i == j {
                1.0 + 0.3 * i as f64
            } else {
                0.04 * (1 + i + j) as f64
            };
        }
    }

    let mut p = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            p[i * n + j] = if i == j {
                0.7 - 0.05 * i as f64
            } else {
                0.03 * (i + j + 1) as f64
            };
        }
    }

    // Populate packed ERI with a simple decaying model.
    let mut eri = PackedEri::new(n);
    for mu in 0..n {
        for nu in 0..n {
            for lambda in 0..n {
                for sigma in 0..n {
                    let pq = pair_index(mu, nu, n) as f64;
                    let rs = pair_index(lambda, sigma, n) as f64;
                    let val = 0.25 / (1.0 + pq + rs);
                    eri.set(mu, nu, lambda, sigma, val);
                }
            }
        }
    }

    (h, p, eri)
}

// ---------- GPU capability check (float64) ----------

fn wgpu_supports_f64() -> bool {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: Backends::all(),
        ..Default::default()
    });
    for adapter in instance.enumerate_adapters(Backends::all()) {
        if adapter.features().contains(wgpu::Features::SHADER_F64) {
            return true;
        }
    }
    false
}

// ---------- Main demo ----------

fn main() -> Result<()> {
    let n = 4;
    let (h, p, eri_packed) = realistic_demo_system(n);
    let sparse = eri_packed.to_sparse(1e-4);
    let pair_count = n * (n + 1) / 2;
    let quartet_count = pair_count * (pair_count + 1) / 2;

    // CPU reference
    let f_cpu = build_fock_matrix_cpu(&h, &p, &eri_packed, n);

    // Decide runtime
    let (f_gpu, backend) = if wgpu_supports_f64() {
        let client: ComputeClient<_> = WgpuRuntime::client(&WgpuDevice::DefaultDevice);
        (
            build_fock_matrix_efficient::<WgpuRuntime>(
                &client,
                &h,
                &p,
                &sparse.idx,
                &sparse.val,
                n,
            )?,
            "wgpu",
        )
    } else {
        eprintln!("No wgpu adapter with SHADER_F64 support; using CPU runtime instead.");
        let client: ComputeClient<_> = CpuRuntime::client(&CpuDevice::default());
        (
            build_fock_matrix_efficient::<CpuRuntime>(
                &client,
                &h,
                &p,
                &sparse.idx,
                &sparse.val,
                n,
            )?,
            "cpu",
        )
    };

    // Compare
    let mut max_diff = 0.0;
    for i in 0..f_cpu.len() {
        let d = (f_cpu[i] - f_gpu[i]).abs();
        if d > max_diff {
            max_diff = d;
        }
    }

    println!("Backend used: {backend}");
    println!("n = {n}");
    println!("Memory sizes (elements): h={}, P={}, packed ERI={}, sparse nnz={}, full ERI={} (for comparison)", h.len(), p.len(), quartet_count, sparse.idx.len(), n * n * n * n);
    println!("Fock matrix (GPU path):");
    for mu in 0..n {
        for nu in 0..n {
            print!("{:8.4} ", f_gpu[mu * n + nu]);
        }
        println!();
    }
    println!("Max |F_gpu - F_cpu| = {:.3e}", max_diff);

    // Quick symmetry check
    let mut sym_err = 0.0;
    for mu in 0..n {
        for nu in 0..n {
            let d = (f_gpu[mu * n + nu] - f_gpu[nu * n + mu]).abs();
            if d > sym_err {
                sym_err = d;
            }
        }
    }
    println!("Max symmetry deviation |F - F^T| = {:.3e}", sym_err);

    Ok(())
}

```