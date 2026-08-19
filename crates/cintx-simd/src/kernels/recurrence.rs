use crate::vector::SimdFloat;

/// Compute 1D Vertical Recurrence Relation (VRR) for overlap along one Cartesian axis.
///
/// Builds levels `0..=nmax` into `g[base + n]`.
/// Level 0 `g[base]` must already contain the base factor.
#[inline(always)]
pub fn vrr_1e_axis<V: SimdFloat>(g: &mut [V], base: usize, rijrx: V, aij2: V, nmax: usize) {
    if nmax >= 1 {
        g[base + 1] = rijrx * g[base];
        for n in 1..nmax {
            g[base + n + 1] = V::from_usize(n) * aij2 * g[base + n - 1] + rijrx * g[base + n];
        }
    }
}

/// Compute 1D Vertical Recurrence Relation (VRR) for nuclear attraction along one axis.
///
/// Builds levels `0..=nmax` into `g[base + n]` using root displacement `c00` and $r_t = \frac{1}{2\zeta}(1 - \tau)$.
#[inline(always)]
pub fn vrr_nuc_axis<V: SimdFloat>(g: &mut [V], base: usize, c00: V, rt: V, nmax: usize) {
    if nmax >= 1 {
        g[base + 1] = c00 * g[base];
        for n in 1..nmax {
            g[base + n + 1] = V::from_usize(n) * rt * g[base + n - 1] + c00 * g[base + n];
        }
    }
}

/// Compute 1D Horizontal Recurrence Relation (HRR) to transfer angular momentum to the ket center.
///
/// Transforms `g` from bra-only angular momentum `0..=li_max` to $(j, i)$ levels with $j \in 0..=lj$.
#[inline(always)]
pub fn hrr_1e_axis<V: SimdFloat>(
    g: &mut [V],
    base: usize,
    rirj: V,
    dj: usize,
    li_max: usize,
    lj: usize,
) {
    for j in 1..=lj {
        let i_max = li_max - j;
        for i in 0..=i_max {
            let idx_out = base + j * dj + i;
            let idx_hi = base + (j - 1) * dj + (i + 1);
            let idx_lo = base + (j - 1) * dj + i;
            g[idx_out] = g[idx_hi] + rirj * g[idx_lo];
        }
    }
}

/// Second ket-derivative $D_j^2(g_0)[j, i]$ on one axis for the kinetic energy operator.
///
/// $g_3 = 4 \alpha_j^2 g_0[j+2] - 2 \alpha_j (2j + 1) g_0[j] + j(j-1) g_0[j-2]$.
#[inline(always)]
pub fn kin_d2_axis<V: SimdFloat>(
    g: &[V],
    base: usize,
    nx: usize,
    dj: usize,
    jx: usize,
    aj: V,
) -> V {
    let g_hi = g[base + nx + 2 * dj];
    let v0 = g[base + nx];
    let mut lo = V::splat(V::Scalar::default());
    if jx >= 2 {
        lo = g[base + nx - 2 * dj];
    }
    let jxf = jx as f64;
    let four = V::from_f64(4.0);
    let two = V::from_f64(2.0);
    let one = V::from_f64(1.0);

    four * aj * aj * g_hi - two * aj * (two * V::from_f64(jxf) + one) * v0
        + V::from_f64(jxf * (jxf - 1.0)) * lo
}

/// 2D Vertical Recurrence Relation (VRR) along one axis for 2-electron integrals (2c2e, 3c2e, 2e).
///
/// Generates the full 2D angular momentum grid `(n in 0..=nmax, m in 0..=mmax)` for a Rys root.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn vrr_2e_2d_axis<V: SimdFloat>(
    g: &mut [V],
    base: usize,
    irys: usize,
    nmax: usize,
    mmax: usize,
    dn: usize,
    dm: usize,
    c00: V,
    c0p: V,
    b10: V,
    b01: V,
    b00: V,
) {
    if nmax > 0 {
        let mut s_prev = g[base + irys];
        let mut s1 = c00 * s_prev;
        g[base + irys + dn] = s1;
        for n in 1..nmax {
            let s2 = c00 * s1 + V::from_usize(n) * b10 * s_prev;
            g[base + irys + (n + 1) * dn] = s2;
            s_prev = s1;
            s1 = s2;
        }
    }

    if mmax > 0 {
        let mut s_prev = g[base + irys];
        let mut s1 = c0p * s_prev;
        g[base + irys + dm] = s1;
        for m in 1..mmax {
            let s2 = c0p * s1 + V::from_usize(m) * b01 * s_prev;
            g[base + irys + (m + 1) * dm] = s2;
            s_prev = s1;
            s1 = s2;
        }

        if nmax > 0 {
            for n in 1..=nmax {
                let i_off = irys + n * dn;
                let s0_k0 = g[base + i_off];
                let prev_i_k0 = g[base + irys + (n - 1) * dn];
                let mut s1 = c0p * s0_k0 + V::from_usize(n) * b00 * prev_i_k0;
                g[base + i_off + dm] = s1;
                let mut s_prev = s0_k0;
                for m in 1..mmax {
                    let prev_i_km = g[base + irys + (n - 1) * dn + m * dm];
                    let s2 = c0p * s1
                        + V::from_usize(m) * b01 * s_prev
                        + V::from_usize(n) * b00 * prev_i_km;
                    g[base + i_off + (m + 1) * dm] = s2;
                    s_prev = s1;
                    s1 = s2;
                }
            }
        }
    }
}
