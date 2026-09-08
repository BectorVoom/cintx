//! Root-axis vector math for the Rys 2D recurrence.
//!
//! Every G-tensor index the 2e pipeline forms is `base + r`, where `r` is the
//! Rys root and `base` is built from `di`/`dk`/`dl`/`dj` — all of which
//! `build_2e_shape` defines as multiples of `nroots`. The root axis is
//! therefore the contiguous, aligned, innermost axis of every G slab, of
//! exactly `nroots` elements, and the VRR never crosses it: root `r`'s
//! recurrence reads and writes only `off + r + n·dn + m·dm`.
//!
//! [`vrr_fill_axis_roots`] runs all `nroots` of those recurrences as one CubeCL
//! `Vector` chain instead of `nroots` scalar ones. What that buys is not wider
//! loads — the loads are the same — but the *dependency chain*: the VRR is
//! latency-bound, one division and two serial two-term recurrences per root,
//! and folding the roots into lanes collapses `nroots` chains into one.
//! Measured on the CPU runtime by `vrr_bench_vector_vs_scalar`, at the
//! `(nroots, nmax, mmax)` shapes a GTH-MOLOPT basis produces: **1.7x–3.1x**.
//!
//! The HRR transfer and the contraction's `Σ_r gx·gy·gz` were measured the same
//! way and are deliberately *not* here: both are gather-bound rather than
//! latency-bound, and both came out at 0.6–1.1x — no better than the scalar
//! loops they would have replaced.
//!
//! # Why the results do not move
//!
//! A `Vector` op is elementwise: lane `r` of `a * b` is `a[r] * b[r]`, the same
//! multiply the scalar loop performed, on the same operands, in the same order.
//! Nothing is reassociated and no fused multiply-add is introduced, so the G
//! tensor comes out bit-for-bit what it was. `root_vec_matches_scalar_bit_for_bit`
//! is the unit-level gate and `CINTX_GTH_COMPARE` the end-to-end one.
//!
//! # The width
//!
//! The width is a *dynamic* size: it is registered into the kernel's expansion
//! scope from a comptime argument rather than carried as a Rust const generic.
//! That matters because `nroots` reaches a kernel as a comptime `u32` — one JIT
//! specialization per value, one Rust instantiation — so a `Const<N>` would
//! force five Rust monomorphizations of an 11 000-line kernel through every
//! launcher. A kernel opts in by declaring an `N: Size` generic plus a
//! `#[define(N)]` comptime argument carrying `nroots`; the macro strips the
//! generic from the generated `launch` signature, so call sites are unchanged
//! apart from that one extra argument.
//!
//! Widths of 3 and 5 are as ordinary here as 2 and 4: `VectorSize` is a plain
//! `usize` in the IR, and the CPU, HIP and CUDA runtimes all report
//! `max_vector_size: VectorSize::MAX`. The one shape to avoid is binding a
//! buffer as `Array<Vector<F, N>>` at an odd `N` — that was tried and returns
//! wrong values at `N = 3` on the CPU runtime. Nothing here does: the slab
//! stays scalar-typed and the lanes are gathered and scattered explicitly, so
//! a run needs only to be contiguous. See [`roots_load`] for why that is not a
//! detail — the slab strides are padded to 8 and are *not* `nroots`-aligned.
//!
//! # Two orderings
//!
//! libcint's `CINTg0_2e_2d` appears in this crate in two loop orderings, and
//! they are not interchangeable — the mixed `b00` recurrence raises a different
//! index and reads a different neighbour in each:
//!
//! - [`vrr_fill_axis_roots`] raises the **bra** index last,
//!   `g(n+1,m) = c00·g(n,m) + n·b10·g(n-1,m) + m·b00·g(n,m-1)`. This is what
//!   `two_electron` and the 3c2e `ip1`/`ip2` derivative kernels inline.
//! - [`vrr_fill_axis_roots_ket`] raises the **ket** index last,
//!   `g(n,m+1) = c0p·g(n,m) + m·b01·g(n,m-1) + n·b00·g(n-1,m)`. This is what
//!   the 3c2e base kernel and `center_2c2e` inline.

use cubecl::prelude::*;

/// Read the `width`-wide root run at `base` as one vector.
///
/// The lanes are gathered element by element, so `base` needs only to start a
/// *contiguous* run — no alignment. That is deliberate rather than lazy: the
/// per-slot slab stride pads to 8 `f64` (`g_slab_stride`, `three_c2e_slab_stride`),
/// which is not a multiple of an odd `nroots`, so `slot * g_stride` is not
/// `nroots`-aligned and a reinterpreting vector load would read across run
/// boundaries for slots past the first. The gather cannot; and the win here is
/// the dependency chain, not the load width.
#[cube]
pub fn roots_load<F: Float, N: Size>(
    g: &Slice<F, ReadWrite>,
    base: u32,
    #[comptime] width: usize,
) -> Vector<F, N> {
    let mut run = Vector::<F, N>::empty();
    #[unroll]
    for r in 0..comptime!(width as u32) {
        run[r as usize] = g[(base + r) as usize];
    }
    run
}

/// Write `value` over the `width`-wide root run at `base`.
#[cube]
pub fn roots_store<F: Float, N: Size>(
    g: &mut Slice<F, ReadWrite>,
    base: u32,
    value: Vector<F, N>,
    #[comptime] width: usize,
) {
    #[unroll]
    for r in 0..comptime!(width as u32) {
        g[(base + r) as usize] = value[r as usize];
    }
}

/// The 2D VRR fill for one axis, every Rys root at once — **bra-raising**
/// ordering (see the module note on the two orderings).
///
/// A transcription of the scalar `vrr_fill_axis` body — libcint's
/// `CINTg0_2e_2d` (`g2e.c:135-192`) — with the root index folded into the
/// vector lanes. `off` is the axis slice's base, `dn`/`dm` are `g2d_ijmax` and
/// `g2d_klmax`, and `seed` is what the slice's `(0,0)` entry starts at: one for
/// the `gx`/`gy` axes, `wrys · fac1` for `gz`.
///
/// Every statement is the scalar one with `g[off + root + X]` replaced by the
/// run at `off + X`, so the four loops, their bounds and their order are what
/// they were.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn vrr_fill_axis_roots<F: Float, N: Size>(
    g: &mut Slice<F, ReadWrite>,
    off: u32,
    nmax: u32,
    mmax: u32,
    dn: u32,
    dm: u32,
    c00: Vector<F, N>,
    c0p: Vector<F, N>,
    b00: Vector<F, N>,
    b10: Vector<F, N>,
    b01: Vector<F, N>,
    seed: Vector<F, N>,
    #[comptime] width: usize,
) {
    roots_store::<F, N>(g, off, seed, width);

    if nmax > 0u32 {
        let mut s0 = seed;
        let mut s1 = c00 * s0;
        roots_store::<F, N>(g, off + dn, s1, width);
        let mut n = 1u32;
        while n < nmax {
            let nf: F = F::cast_from(n);
            let s2 = c00 * s1 + Vector::<F, N>::new(nf) * b10 * s0;
            roots_store::<F, N>(g, off + (n + 1u32) * dn, s2, width);
            s0 = s1;
            s1 = s2;
            n += 1u32;
        }
    }

    if mmax > 0u32 {
        let mut s0 = seed;
        let mut s1 = c0p * s0;
        roots_store::<F, N>(g, off + dm, s1, width);
        let mut m = 1u32;
        while m < mmax {
            let mf: F = F::cast_from(m);
            let s2 = c0p * s1 + Vector::<F, N>::new(mf) * b01 * s0;
            roots_store::<F, N>(g, off + (m + 1u32) * dm, s2, width);
            s0 = s1;
            s1 = s2;
            m += 1u32;
        }

        if nmax > 0u32 {
            let mut s0n = roots_load::<F, N>(g, off + dn, width);
            let mut s1n = c0p * s0n + b00 * roots_load::<F, N>(g, off, width);
            roots_store::<F, N>(g, off + dn + dm, s1n, width);
            let mut m2 = 1u32;
            while m2 < mmax {
                let m2f: F = F::cast_from(m2);
                let s2n = c0p * s1n
                    + Vector::<F, N>::new(m2f) * b01 * s0n
                    + b00 * roots_load::<F, N>(g, off + m2 * dm, width);
                roots_store::<F, N>(g, off + dn + (m2 + 1u32) * dm, s2n, width);
                s0n = s1n;
                s1n = s2n;
                m2 += 1u32;
            }
        }
    }

    if nmax > 0u32 {
        let mut m3 = 1u32;
        while m3 <= mmax {
            let jbase = off + m3 * dm;
            let m3f: F = F::cast_from(m3);
            let m3v = Vector::<F, N>::new(m3f);
            let mut s0 = roots_load::<F, N>(g, jbase, width);
            let mut s1 = roots_load::<F, N>(g, jbase + dn, width);
            let mut n2 = 1u32;
            while n2 < nmax {
                let n2f: F = F::cast_from(n2);
                let s2 = c00 * s1
                    + Vector::<F, N>::new(n2f) * b10 * s0
                    + m3v * b00 * roots_load::<F, N>(g, jbase + n2 * dn - dm, width);
                roots_store::<F, N>(g, jbase + (n2 + 1u32) * dn, s2, width);
                s0 = s1;
                s1 = s2;
                n2 += 1u32;
            }
            m3 += 1u32;
        }
    }
}

/// The 2D VRR fill for one axis, every Rys root at once — **ket-raising**
/// ordering (see the module note on the two orderings).
///
/// The form `center_2c2e` and the 3c2e base kernel inline: the two pure ladders
/// as above, then the mixed recurrence walked `n` outer / `m` inner, raising the
/// ket index and reading the `n-1` neighbour through `b00`.
///
/// `off` is the axis slice's base and `seed` its `(0,0)` entry: one for the
/// `gx`/`gy` axes, `wrys · fac1` for `gz`. The callers wrote those three seeds
/// in a separate pass over the roots before entering the axis loop; writing each
/// one here instead only reorders stores into disjoint axis slices.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn vrr_fill_axis_roots_ket<F: Float, N: Size>(
    g: &mut Slice<F, ReadWrite>,
    off: u32,
    nmax: u32,
    mmax: u32,
    dn: u32,
    dm: u32,
    c00: Vector<F, N>,
    c0p: Vector<F, N>,
    b00: Vector<F, N>,
    b10: Vector<F, N>,
    b01: Vector<F, N>,
    seed: Vector<F, N>,
    #[comptime] width: usize,
) {
    roots_store::<F, N>(g, off, seed, width);

    if nmax >= 1u32 {
        let mut s_prev = seed;
        let mut s1 = c00 * s_prev;
        roots_store::<F, N>(g, off + dn, s1, width);
        let mut n = 1u32;
        while n < nmax {
            let nf: F = F::cast_from(n);
            let s2 = c00 * s1 + Vector::<F, N>::new(nf) * b10 * s_prev;
            roots_store::<F, N>(g, off + (n + 1u32) * dn, s2, width);
            s_prev = s1;
            s1 = s2;
            n += 1u32;
        }
    }

    if mmax >= 1u32 {
        let mut s_prev = seed;
        let mut s1 = c0p * s_prev;
        roots_store::<F, N>(g, off + dm, s1, width);
        let mut m = 1u32;
        while m < mmax {
            let mf: F = F::cast_from(m);
            let s2 = c0p * s1 + Vector::<F, N>::new(mf) * b01 * s_prev;
            roots_store::<F, N>(g, off + (m + 1u32) * dm, s2, width);
            s_prev = s1;
            s1 = s2;
            m += 1u32;
        }

        // n > 0 ladders over m, with the b00 cross term.
        if nmax >= 1u32 {
            let mut n = 1u32;
            while n <= nmax {
                let i_off = off + n * dn;
                let nf: F = F::cast_from(n);
                let nv = Vector::<F, N>::new(nf);
                let s0_k0 = roots_load::<F, N>(g, i_off, width);
                let prev_i_k0 = roots_load::<F, N>(g, off + (n - 1u32) * dn, width);
                let mut s1 = c0p * s0_k0 + nv * b00 * prev_i_k0;
                roots_store::<F, N>(g, i_off + dm, s1, width);
                let mut s_prev = s0_k0;
                let mut m = 1u32;
                while m < mmax {
                    let mf: F = F::cast_from(m);
                    let prev_i_km = roots_load::<F, N>(g, off + (n - 1u32) * dn + m * dm, width);
                    let s2 =
                        c0p * s1 + Vector::<F, N>::new(mf) * b01 * s_prev + nv * b00 * prev_i_km;
                    roots_store::<F, N>(g, i_off + (m + 1u32) * dm, s2, width);
                    s_prev = s1;
                    s1 = s2;
                    m += 1u32;
                }
                n += 1u32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_client() -> cubecl::client::ComputeClient<cubecl::cpu::CpuRuntime> {
        cubecl::cpu::CpuRuntime::client(&Default::default())
    }

    /// Both arms of the VRR in one launch, over separate slabs. The scalar arm
    /// is the loop `two_electron_scalar_kernel` inlines today, transcribed.
    #[cube(launch_unchecked)]
    #[allow(clippy::too_many_arguments)]
    fn vrr_probe_kernel<F: Float + CubeElement, N: Size>(
        urys: &mut Array<F>,
        wrys: &mut Array<F>,
        g_scalar: &mut Array<F>,
        g_vector: &mut Array<F>,
        aij: F,
        akl: F,
        a0: F,
        a1: F,
        xkl: F,
        rijrx: F,
        rklrx: F,
        fac1: F,
        nmax: u32,
        mmax: u32,
        dn: u32,
        dm: u32,
        reps: u32,
        #[define(N)]
        #[comptime]
        width: usize,
        #[comptime] nroots: u32,
        #[comptime] vectorized: u32,
        // `0` = bra-raising (`vrr_fill_axis_roots`), `1` = ket-raising
        // (`vrr_fill_axis_roots_ket`).
        #[comptime] ordering: u32,
    ) {
        let half = F::new(0.5_f32);
        let two = F::new(2.0_f32);

        let mut rep = 0u32;
        while rep < reps {
            if comptime!(vectorized == 1u32) {
                let u = urys.to_slice_mut();
                let w = wrys.to_slice_mut();
                let mut g = g_vector.to_slice_mut();

                let u2 = Vector::<F, N>::new(a0) * roots_load::<F, N>(&u, 0u32, width);
                let tmp4 = Vector::<F, N>::new(half)
                    / (u2 * Vector::<F, N>::new(aij + akl) + Vector::<F, N>::new(a1));
                let tmp5 = u2 * tmp4;
                let tmp1 = Vector::<F, N>::new(two) * tmp5;
                let tmp2 = tmp1 * Vector::<F, N>::new(akl);
                let tmp3 = tmp1 * Vector::<F, N>::new(aij);
                let b00 = tmp5;
                let b10 = tmp5 + tmp4 * Vector::<F, N>::new(akl);
                let b01 = tmp5 + tmp4 * Vector::<F, N>::new(aij);
                let c00 = Vector::<F, N>::new(rijrx) - tmp2 * Vector::<F, N>::new(xkl);
                let c0p = Vector::<F, N>::new(rklrx) + tmp3 * Vector::<F, N>::new(xkl);
                let seed = roots_load::<F, N>(&w, 0u32, width) * Vector::<F, N>::new(fac1);

                if comptime!(ordering == 0u32) {
                    vrr_fill_axis_roots::<F, N>(
                        &mut g, 0u32, nmax, mmax, dn, dm, c00, c0p, b00, b10, b01, seed, width,
                    );
                } else {
                    vrr_fill_axis_roots_ket::<F, N>(
                        &mut g, 0u32, nmax, mmax, dn, dm, c00, c0p, b00, b10, b01, seed, width,
                    );
                }
            } else {
                #[unroll]
                for irys in 0..nroots {
                    let u2 = a0 * urys[irys as usize];
                    let tmp4 = half / (u2 * (aij + akl) + a1);
                    let tmp5 = u2 * tmp4;
                    let tmp1 = two * tmp5;
                    let tmp2 = tmp1 * akl;
                    let tmp3 = tmp1 * aij;
                    let b00 = tmp5;
                    let b10 = tmp5 + tmp4 * akl;
                    let b01 = tmp5 + tmp4 * aij;
                    let c00 = rijrx - tmp2 * xkl;
                    let c0p = rklrx + tmp3 * xkl;

                    let root = irys;
                    let seed = wrys[root as usize] * fac1;
                    g_scalar[(root) as usize] = seed;

                    if comptime!(ordering == 1u32) {
                        // The `center_2c2e` / 3c2e-base transcription.
                        if nmax >= 1u32 {
                            let mut s_prev = g_scalar[(root) as usize];
                            let mut s1 = c00 * s_prev;
                            g_scalar[(root + dn) as usize] = s1;
                            let mut n = 1u32;
                            while n < nmax {
                                let s2 = c00 * s1 + F::cast_from(n) * b10 * s_prev;
                                g_scalar[(root + (n + 1u32) * dn) as usize] = s2;
                                s_prev = s1;
                                s1 = s2;
                                n += 1u32;
                            }
                        }

                        if mmax >= 1u32 {
                            let mut s_prev = g_scalar[(root) as usize];
                            let mut s1 = c0p * s_prev;
                            g_scalar[(root + dm) as usize] = s1;
                            let mut m = 1u32;
                            while m < mmax {
                                let s2 = c0p * s1 + F::cast_from(m) * b01 * s_prev;
                                g_scalar[(root + (m + 1u32) * dm) as usize] = s2;
                                s_prev = s1;
                                s1 = s2;
                                m += 1u32;
                            }

                            if nmax >= 1u32 {
                                let mut n = 1u32;
                                while n <= nmax {
                                    let i_off = root + n * dn;
                                    let s0_k0 = g_scalar[(i_off) as usize];
                                    let prev_i_k0 = g_scalar[(root + (n - 1u32) * dn) as usize];
                                    let mut s1 = c0p * s0_k0 + F::cast_from(n) * b00 * prev_i_k0;
                                    g_scalar[(i_off + dm) as usize] = s1;
                                    let mut s_prev = s0_k0;
                                    let mut m = 1u32;
                                    while m < mmax {
                                        let prev_i_km =
                                            g_scalar[(root + (n - 1u32) * dn + m * dm) as usize];
                                        let s2 = c0p * s1
                                            + F::cast_from(m) * b01 * s_prev
                                            + F::cast_from(n) * b00 * prev_i_km;
                                        g_scalar[(i_off + (m + 1u32) * dm) as usize] = s2;
                                        s_prev = s1;
                                        s1 = s2;
                                        m += 1u32;
                                    }
                                    n += 1u32;
                                }
                            }
                        }
                    } else {
                        if nmax > 0u32 {
                            let mut s0 = g_scalar[(root) as usize];
                            let mut s1 = c00 * s0;
                            g_scalar[(root + dn) as usize] = s1;
                            let mut n = 1u32;
                            while n < nmax {
                                let s2 = c00 * s1 + F::cast_from(n) * b10 * s0;
                                g_scalar[(root + (n + 1u32) * dn) as usize] = s2;
                                s0 = s1;
                                s1 = s2;
                                n += 1u32;
                            }
                        }

                        if mmax > 0u32 {
                            let mut s0 = g_scalar[(root) as usize];
                            let mut s1 = c0p * s0;
                            g_scalar[(root + dm) as usize] = s1;
                            let mut m = 1u32;
                            while m < mmax {
                                let s2 = c0p * s1 + F::cast_from(m) * b01 * s0;
                                g_scalar[(root + (m + 1u32) * dm) as usize] = s2;
                                s0 = s1;
                                s1 = s2;
                                m += 1u32;
                            }

                            if nmax > 0u32 {
                                let mut s0n = g_scalar[(root + dn) as usize];
                                let mut s1n = c0p * s0n + b00 * g_scalar[(root) as usize];
                                g_scalar[(root + dn + dm) as usize] = s1n;
                                let mut m2 = 1u32;
                                while m2 < mmax {
                                    let s2n = c0p * s1n
                                        + F::cast_from(m2) * b01 * s0n
                                        + b00 * g_scalar[(root + m2 * dm) as usize];
                                    g_scalar[(root + dn + (m2 + 1u32) * dm) as usize] = s2n;
                                    s0n = s1n;
                                    s1n = s2n;
                                    m2 += 1u32;
                                }
                            }
                        }

                        if nmax > 0u32 {
                            let mut m3 = 1u32;
                            while m3 <= mmax {
                                let jbase = m3 * dm + root;
                                let mut s0 = g_scalar[(jbase) as usize];
                                let mut s1 = g_scalar[(jbase + dn) as usize];
                                let mut n2 = 1u32;
                                while n2 < nmax {
                                    let s2 = c00 * s1
                                        + F::cast_from(n2) * b10 * s0
                                        + F::cast_from(m3)
                                            * b00
                                            * g_scalar[(jbase + n2 * dn - dm) as usize];
                                    g_scalar[(jbase + (n2 + 1u32) * dn) as usize] = s2;
                                    s0 = s1;
                                    s1 = s2;
                                    n2 += 1u32;
                                }
                                m3 += 1u32;
                            }
                        }
                    }
                }
            }
            rep += 1u32;
        }
    }

    /// `(nroots, nmax, mmax)` as `build_2e_shape` pairs them: `nroots` is
    /// `(li+lj+lk+ll)/2 + 1`, `nmax = li+lj` and `mmax = lk+ll`. The rows span
    /// the GTH-MOLOPT classes, including the odd widths 3 and 5.
    const VRR_SHAPES: [(u32, u32, u32); 5] =
        [(1, 0, 0), (2, 2, 1), (3, 2, 2), (4, 3, 3), (5, 4, 4)];

    type VrrRow = ((u32, u32, u32), Vec<f64>, Vec<f64>, [f64; 2]);

    /// Run both arms once per shape: `(shape, scalar slab, vector slab, best ms)`.
    fn run_vrr_arms(reps: u32, rounds: usize, ordering: u32) -> Vec<VrrRow> {
        let client = cpu_client();
        let mut rows = Vec::new();

        for (nroots, nmax, mmax) in VRR_SHAPES {
            let n = nroots as usize;
            let dn = nroots;
            let dm = nroots * (nmax + 1);
            let g_len = (nroots * (nmax + 1) * (mmax + 1)) as usize;
            let urys: Vec<f64> = (0..n).map(|i| 0.21 + 0.37 * i as f64).collect();
            let wrys: Vec<f64> = (0..n).map(|i| 0.11 + 0.13 * i as f64).collect();

            let urys_h = client.create_from_slice(f64::as_bytes(&urys));
            let wrys_h = client.create_from_slice(f64::as_bytes(&wrys));

            let mut finals: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
            let mut millis = [0.0_f64; 2];
            for (arm, vectorized) in [0u32, 1].iter().enumerate() {
                let mut best = f64::INFINITY;
                for _round in 0..rounds {
                    let zeros = vec![0.0_f64; g_len];
                    let gs_h = client.create_from_slice(f64::as_bytes(&zeros));
                    let gv_h = client.create_from_slice(f64::as_bytes(&zeros));
                    let start = std::time::Instant::now();
                    unsafe {
                        vrr_probe_kernel::launch_unchecked::<f64, cubecl::cpu::CpuRuntime>(
                            &client,
                            crate::plane::single_cube_count(),
                            CubeDim::new_1d(1),
                            ArrayArg::from_raw_parts(urys_h.clone(), n),
                            ArrayArg::from_raw_parts(wrys_h.clone(), n),
                            ArrayArg::from_raw_parts(gs_h.clone(), g_len),
                            ArrayArg::from_raw_parts(gv_h.clone(), g_len),
                            1.7_f64,
                            2.3_f64,
                            0.61_f64,
                            0.9_f64,
                            0.41_f64,
                            0.27_f64,
                            0.33_f64,
                            1.19_f64,
                            nmax,
                            mmax,
                            dn,
                            dm,
                            reps,
                            n,
                            nroots,
                            *vectorized,
                            ordering,
                        );
                    }
                    let handle = if *vectorized == 1u32 { gv_h } else { gs_h };
                    let raw = client.read_one_unchecked(handle);
                    best = best.min(start.elapsed().as_secs_f64() * 1e3);
                    finals[arm] = f64::from_bytes(&raw).to_vec();
                }
                millis[arm] = best;
            }

            let [scalar, vector] = finals;
            rows.push(((nroots, nmax, mmax), scalar, vector, millis));
        }
        rows
    }

    /// The vector VRR must reproduce the scalar one bit for bit at every width
    /// the device 2e path compiles for — the odd 3 and 5 included.
    #[test]
    fn root_vec_matches_scalar_bit_for_bit() {
        for ordering in [0u32, 1] {
            for ((nroots, nmax, mmax), scalar, vector, _) in run_vrr_arms(1, 1, ordering) {
                assert_eq!(scalar.len(), vector.len());
                for (i, (a, b)) in scalar.iter().zip(vector.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "ordering={ordering} nroots={nroots} nmax={nmax} mmax={mmax}, \
                     g[{i}]: scalar {a} vs vector {b}"
                    );
                }
            }
        }
    }

    /// `cargo test -p cintx-cubecl --release --lib vrr_bench -- --ignored --nocapture`
    #[test]
    #[ignore = "timing probe; run explicitly in release"]
    fn vrr_bench_vector_vs_scalar() {
        for ordering in [0u32, 1] {
            println!(
                "-- {} ordering",
                if ordering == 0 {
                    "bra-raising"
                } else {
                    "ket-raising"
                }
            );
            for ((nroots, nmax, mmax), scalar, vector, millis) in run_vrr_arms(200_000, 5, ordering)
            {
                for (i, (a, b)) in scalar.iter().zip(vector.iter()).enumerate() {
                    assert_eq!(a.to_bits(), b.to_bits(), "nroots={nroots}, g[{i}]");
                }
                println!(
                    "nroots={nroots} nmax={nmax} mmax={mmax}: scalar {:.1} ms, vector {:.1} ms  ({:.2}x)",
                    millis[0],
                    millis[1],
                    millis[0] / millis[1]
                );
            }
        }
    }
}
