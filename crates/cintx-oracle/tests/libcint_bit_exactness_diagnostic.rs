//! Bit-exactness diagnostic: cintx `eval_raw` vs vendored libcint 6.1.3.
//!
//! The per-family parity gates assert an absolute/relative tolerance. This
//! reports the stronger property the gates do not: how many output elements
//! are *bit-identical* to the vendor, and for the rest, the ULP distance. It
//! is a measurement, not a gate — it prints a table and never fails, so the
//! remaining gap can be attributed family by family.

#![cfg(feature = "cpu")]
#![cfg(has_vendor_libcint)]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};
use cintx_oracle::vendor_ffi;

/// H2O/STO-3G, the fixture the 3-way parity test uses.
fn build_h2o_sto3g() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let o_coord = [0.0_f64, 0.0, 0.0];
    let h1_coord = [0.0_f64, 1.4307, 1.1078];
    let h2_coord = [0.0_f64, -1.4307, 1.1078];

    let o_1s_exp = [130.7093200_f64, 23.8088610, 6.4436083];
    let o_1s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];
    let o_2s_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let o_2s_coeff = [-0.09996723_f64, 0.39951283, 0.70011547];
    let o_2p_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let o_2p_coeff = [0.15591627_f64, 0.60768372, 0.39195739];
    let h_1s_exp = [3.4252509_f64, 0.6239137, 0.1688554];
    let h_1s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];

    let mut env = vec![0.0_f64; PTR_ENV_START];
    let o_coord_ptr = env.len() as i32;
    env.extend_from_slice(&o_coord);
    let h1_coord_ptr = env.len() as i32;
    env.extend_from_slice(&h1_coord);
    let h2_coord_ptr = env.len() as i32;
    env.extend_from_slice(&h2_coord);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    let o1s_exp_ptr = env.len() as i32;
    env.extend_from_slice(&o_1s_exp);
    let o1s_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&o_1s_coeff);
    let o2s_exp_ptr = env.len() as i32;
    env.extend_from_slice(&o_2s_exp);
    let o2s_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&o_2s_coeff);
    let o2p_exp_ptr = env.len() as i32;
    env.extend_from_slice(&o_2p_exp);
    let o2p_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&o_2p_coeff);
    let h1s_exp_ptr = env.len() as i32;
    env.extend_from_slice(&h_1s_exp);
    let h1s_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&h_1s_coeff);

    let mut atm = vec![0_i32; 3 * ATM_SLOTS];
    for (a, ptr) in [o_coord_ptr, h1_coord_ptr, h2_coord_ptr].iter().enumerate() {
        atm[a * ATM_SLOTS + PTR_COORD] = *ptr;
        atm[a * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
        atm[a * ATM_SLOTS + PTR_ZETA] = zeta_ptr;
    }
    atm[CHARGE_OF] = 8;
    atm[ATM_SLOTS + CHARGE_OF] = 1;
    atm[2 * ATM_SLOTS + CHARGE_OF] = 1;

    let mut bas = vec![0_i32; 5 * BAS_SLOTS];
    let shells: [(i32, i32, i32, i32); 5] = [
        (0, 0, o1s_exp_ptr, o1s_coeff_ptr),
        (0, 0, o2s_exp_ptr, o2s_coeff_ptr),
        (0, 1, o2p_exp_ptr, o2p_coeff_ptr),
        (1, 0, h1s_exp_ptr, h1s_coeff_ptr),
        (2, 0, h1s_exp_ptr, h1s_coeff_ptr),
    ];
    for (s, (atom, l, exp_ptr, coeff_ptr)) in shells.iter().enumerate() {
        bas[s * BAS_SLOTS + ATOM_OF] = *atom;
        bas[s * BAS_SLOTS + ANG_OF] = *l;
        bas[s * BAS_SLOTS + NPRIM_OF] = 3;
        bas[s * BAS_SLOTS + NCTR_OF] = 1;
        bas[s * BAS_SLOTS + PTR_EXP] = *exp_ptr;
        bas[s * BAS_SLOTS + PTR_COEFF] = *coeff_ptr;
    }

    (atm, bas, env)
}

fn ncart(l: i32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}

fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}

fn shell_l(s: usize, bas: &[i32]) -> i32 {
    bas[s * BAS_SLOTS + ANG_OF]
}

/// Distance in representable f64 steps. `f64::MAX` when the two are not
/// comparable that way (opposite signs, or a non-finite operand).
fn ulp_distance(a: f64, b: f64) -> f64 {
    if a == b {
        return 0.0;
    }
    if !a.is_finite() || !b.is_finite() {
        return f64::MAX;
    }
    // Map to a monotone integer ordering over f64, then subtract.
    let key = |x: f64| -> i64 {
        let bits = x.to_bits() as i64;
        if bits < 0 { i64::MIN - bits } else { bits }
    };
    (key(a) - key(b)).unsigned_abs() as f64
}

#[derive(Default)]
struct Tally {
    elements: usize,
    identical: usize,
    max_ulp: f64,
    max_abs: f64,
    worst: Option<(String, f64, f64)>,
    /// Worst by absolute difference. The ULP-worst element is almost always a
    /// catastrophic cancellation around 1e-19, where the metric is meaningless;
    /// this one says where the arithmetic actually diverges.
    worst_abs: Option<(String, f64, f64)>,
    /// Elements that are numerically equal but differ in the sign of zero.
    signed_zero: usize,
    /// The first few mismatching labels — which shell tuples actually diverge.
    labels: Vec<String>,
}

impl Tally {
    fn add(&mut self, label: &str, cintx: &[f64], vendor: &[f64]) {
        assert_eq!(cintx.len(), vendor.len(), "{label}: length mismatch");
        for (idx, (&c, &v)) in cintx.iter().zip(vendor).enumerate() {
            self.elements += 1;
            if c.to_bits() == v.to_bits() {
                self.identical += 1;
                continue;
            }
            if c == v {
                // Numerically equal but different bits: a signed zero.
                self.signed_zero += 1;
                if self.signed_zero <= 2 {
                    println!(
                        "    signed-zero {label}[{idx}]: cintx={} vendor={}",
                        if c.is_sign_negative() { "-0.0" } else { "+0.0" },
                        if v.is_sign_negative() { "-0.0" } else { "+0.0" },
                    );
                }
            }
            if self.labels.len() < 6 && !self.labels.iter().any(|l| l == label) {
                self.labels.push(label.to_string());
            }
            let ulp = ulp_distance(c, v);
            let abs = (c - v).abs();
            if ulp > self.max_ulp {
                self.max_ulp = ulp;
                self.worst = Some((format!("{label}[{idx}]"), c, v));
            }
            if abs > self.max_abs {
                self.max_abs = abs;
                self.worst_abs = Some((format!("{label}[{idx}]"), c, v));
            }
        }
    }

    fn report(&self, family: &str) {
        let pct = if self.elements == 0 {
            100.0
        } else {
            100.0 * self.identical as f64 / self.elements as f64
        };
        println!(
            "{family:<22} {:>7} elems  {:>7} bit-identical ({pct:6.2}%)  max_ulp={:<10} max_abs={:.3e}",
            self.elements, self.identical, self.max_ulp, self.max_abs
        );
        if self.signed_zero > 0 {
            println!("    (of which {} are signed-zero only)", self.signed_zero);
        }
        if !self.labels.is_empty() {
            println!("    first mismatching tuples: {:?}", self.labels);
        }
        if let Some((where_, c, v)) = &self.worst_abs {
            println!("    worst |diff| at {where_}: cintx={c:.17e} vendor={v:.17e}");
        }
    }
}

fn eval(api: RawApiId, out: &mut [f64], shls: &[i32], atm: &[i32], bas: &[i32], env: &[f64]) {
    unsafe {
        eval_raw(api, Some(out), None, shls, atm, bas, env, None, None)
            .unwrap_or_else(|e| panic!("eval_raw {api:?} on {shls:?}: {e:?}"));
    }
}

#[test]
fn report_bit_exactness_against_vendor() {
    let (atm, bas, env) = build_h2o_sto3g();
    let natm = 3_i32;
    let nbas = 5_i32;
    let n = nbas as usize;

    println!("\n=== cintx vs vendored libcint 6.1.3 — bit-exactness (H2O/STO-3G) ===");

    // ── 1e, cart and sph ────────────────────────────────────────────────────
    for (name, cart_api, sph_api, cart_fn, sph_fn) in [
        (
            "int1e_ovlp",
            RawApiId::INT1E_OVLP_CART,
            RawApiId::INT1E_OVLP_SPH,
            vendor_ffi::vendor_int1e_ovlp_cart
                as fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
            vendor_ffi::vendor_int1e_ovlp_sph
                as fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
        ),
        (
            "int1e_kin",
            RawApiId::INT1E_KIN_CART,
            RawApiId::INT1E_KIN_SPH,
            vendor_ffi::vendor_int1e_kin_cart,
            vendor_ffi::vendor_int1e_kin_sph,
        ),
        (
            "int1e_nuc",
            RawApiId::INT1E_NUC_CART,
            RawApiId::INT1E_NUC_SPH,
            vendor_ffi::vendor_int1e_nuc_cart,
            vendor_ffi::vendor_int1e_nuc_sph,
        ),
    ] {
        let mut cart = Tally::default();
        let mut sph = Tally::default();
        for si in 0..n {
            for sj in 0..n {
                let (li, lj) = (shell_l(si, &bas), shell_l(sj, &bas));
                let shls = [si as i32, sj as i32];

                let len_c = ncart(li) * ncart(lj);
                let (mut a, mut b) = (vec![0.0; len_c], vec![0.0; len_c]);
                eval(cart_api, &mut a, &shls, &atm, &bas, &env);
                let _ = cart_fn(&mut b, &shls, &atm, natm, &bas, nbas, &env);
                cart.add(&format!("{si},{sj}"), &a, &b);

                let len_s = nsph(li) * nsph(lj);
                let (mut a, mut b) = (vec![0.0; len_s], vec![0.0; len_s]);
                eval(sph_api, &mut a, &shls, &atm, &bas, &env);
                let _ = sph_fn(&mut b, &shls, &atm, natm, &bas, nbas, &env);
                sph.add(&format!("{si},{sj}"), &a, &b);
            }
        }
        cart.report(&format!("{name}_cart"));
        sph.report(&format!("{name}_sph"));
    }

    // ── 2c2e ────────────────────────────────────────────────────────────────
    {
        let mut cart = Tally::default();
        let mut sph = Tally::default();
        for si in 0..n {
            for sk in 0..n {
                let (li, lk) = (shell_l(si, &bas), shell_l(sk, &bas));
                let shls = [si as i32, sk as i32];

                let len_c = ncart(li) * ncart(lk);
                let (mut a, mut b) = (vec![0.0; len_c], vec![0.0; len_c]);
                eval(RawApiId::INT2C2E_CART, &mut a, &shls, &atm, &bas, &env);
                let _ =
                    vendor_ffi::vendor_int2c2e_cart(&mut b, &shls, &atm, natm, &bas, nbas, &env);
                cart.add(&format!("{si},{sk}"), &a, &b);

                let len_s = nsph(li) * nsph(lk);
                let (mut a, mut b) = (vec![0.0; len_s], vec![0.0; len_s]);
                eval(RawApiId::INT2C2E_SPH, &mut a, &shls, &atm, &bas, &env);
                let _ = vendor_ffi::vendor_int2c2e_sph(&mut b, &shls, &atm, natm, &bas, nbas, &env);
                sph.add(&format!("{si},{sk}"), &a, &b);
            }
        }
        cart.report("int2c2e_cart");
        sph.report("int2c2e_sph");
    }

    // ── 3c1e / 3c2e ─────────────────────────────────────────────────────────
    {
        let mut c1e = Tally::default();
        let mut c2e_cart = Tally::default();
        let mut c2e_sph = Tally::default();
        for si in 0..n {
            for sj in 0..n {
                for sk in 0..n {
                    let (li, lj, lk) = (shell_l(si, &bas), shell_l(sj, &bas), shell_l(sk, &bas));
                    let shls = [si as i32, sj as i32, sk as i32];
                    let label = format!("{si},{sj},{sk}");

                    let len_c = ncart(li) * ncart(lj) * ncart(lk);
                    let (mut a, mut b) = (vec![0.0; len_c], vec![0.0; len_c]);
                    eval(RawApiId::INT3C1E_CART, &mut a, &shls, &atm, &bas, &env);
                    let _ = vendor_ffi::vendor_int3c1e_cart(
                        &mut b, &shls, &atm, natm, &bas, nbas, &env,
                    );
                    c1e.add(&label, &a, &b);

                    let (mut a, mut b) = (vec![0.0; len_c], vec![0.0; len_c]);
                    eval(RawApiId::INT3C2E_CART, &mut a, &shls, &atm, &bas, &env);
                    let _ = vendor_ffi::vendor_int3c2e_cart(
                        &mut b, &shls, &atm, natm, &bas, nbas, &env,
                    );
                    c2e_cart.add(&label, &a, &b);

                    let len_s = nsph(li) * nsph(lj) * nsph(lk);
                    let (mut a, mut b) = (vec![0.0; len_s], vec![0.0; len_s]);
                    eval(RawApiId::INT3C2E_SPH, &mut a, &shls, &atm, &bas, &env);
                    let _ =
                        vendor_ffi::vendor_int3c2e_sph(&mut b, &shls, &atm, natm, &bas, nbas, &env);
                    c2e_sph.add(&label, &a, &b);
                }
            }
        }
        c1e.report("int3c1e_cart");
        c2e_cart.report("int3c2e_cart");
        c2e_sph.report("int3c2e_sph");
    }

    // ── 2e, every quartet ───────────────────────────────────────────────────
    {
        let mut cart = Tally::default();
        let mut sph = Tally::default();
        // `CINTg0_2e` dispatches to a hand-unrolled `_g0_2d4d_XXXX` whenever
        // `rys_order = (li+lj+lk+ll)/2 + 1 <= 2`, i.e. the l-sum is at most 3.
        // Its algebra is a different grouping of the same recurrence, so split
        // the tally on that boundary to see what the general path still owes.
        let mut unrolled = Tally::default();
        let mut general = Tally::default();
        for si in 0..n {
            for sj in 0..n {
                for sk in 0..n {
                    for sl in 0..n {
                        let ls = [
                            shell_l(si, &bas),
                            shell_l(sj, &bas),
                            shell_l(sk, &bas),
                            shell_l(sl, &bas),
                        ];
                        let shls = [si as i32, sj as i32, sk as i32, sl as i32];
                        let label = format!("{si},{sj},{sk},{sl}");

                        let len_c: usize = ls.iter().map(|&l| ncart(l)).product();
                        let (mut a, mut b) = (vec![0.0; len_c], vec![0.0; len_c]);
                        eval(RawApiId::INT2E_CART, &mut a, &shls, &atm, &bas, &env);
                        let _ = vendor_ffi::vendor_int2e_cart(
                            &mut b, &shls, &atm, natm, &bas, nbas, &env,
                        );
                        cart.add(&label, &a, &b);
                        if ls.iter().sum::<i32>() <= 3 {
                            unrolled.add(&label, &a, &b);
                        } else {
                            general.add(&label, &a, &b);
                        }

                        let len_s: usize = ls.iter().map(|&l| nsph(l)).product();
                        let (mut a, mut b) = (vec![0.0; len_s], vec![0.0; len_s]);
                        eval(RawApiId::INT2E_SPH, &mut a, &shls, &atm, &bas, &env);
                        let _ = vendor_ffi::vendor_int2e_sph(
                            &mut b, &shls, &atm, natm, &bas, nbas, &env,
                        );
                        sph.add(&label, &a, &b);
                    }
                }
            }
        }
        cart.report("int2e_cart");
        sph.report("int2e_sph");
        unrolled.report("  int2e l-sum<=3 (libcint unrolled)");
        general.report("  int2e l-sum>3  (libcint general)");
    }
}

/// Are the fixed-order Rys roots/weights themselves bit-identical?
///
/// Everything downstream of `CINTrys_roots` is a recurrence seeded by these, so
/// a single-ULP difference here caps how exact any Rys family can be. Orders
/// 1..=5 are the polynomial-fit solvers; 6..=12 are libcint's `long double`
/// path, which `rys_nroots_sweep_parity` documents as unreachable in portable
/// Rust, so they are reported but not expected to match.
#[test]
fn report_rys_root_bit_exactness() {
    use cintx_cubecl::math::rys;

    // Split the grid: libcint takes a table-driven early exit below
    // `SMALLX_LIMIT = 3e-7` and above `35 + nroots*5`, and only the band
    // between them reaches `rys_root2..5`. Reporting them together hides which
    // of the two is diverging.
    const XS_MID: &[f64] = &[
        0.001, 0.05, 0.3, 0.5, 1.0, 3.0, 5.0, 8.0, 11.0, 15.0, 20.0, 30.0,
    ];
    const XS_SMALL: &[f64] = &[1e-9, 1e-8, 1e-7];
    const XS_LARGE: &[f64] = &[70.0, 100.0, 500.0];

    println!("\n=== Rys roots/weights: cintx host vs vendored CINTrys_roots ===");
    for (band, xs) in [
        ("mid", XS_MID),
        ("small-x", XS_SMALL),
        ("large-x", XS_LARGE),
    ] {
        for nroots in 1..=5_usize {
            let mut tally = Tally::default();
            for &x in xs {
                let (u, w) = rys::rys_roots_host::<f64>(nroots, x);
                let (vu, vw) = vendor_ffi::vendor_CINTrys_roots(nroots as i32, x);
                tally.add(&format!("n{nroots} x={x} u"), &u[..nroots], &vu[..nroots]);
                tally.add(&format!("n{nroots} x={x} w"), &w[..nroots], &vw[..nroots]);
            }
            tally.report(&format!("rys {band} nroots={nroots}"));
        }
    }
}

/// Where does the last ULP on `(ss|ss)` come from?
///
/// After the G-tensor work, `int2e`'s worst remaining case is shells
/// `(3,3|1,3)` — all `s`, so `nmax = mmax = 0` and there is no recurrence at
/// all. The whole integral is
/// `sum over primitive quartets of w[0] * fac1`. This reproduces `CINT2e_loop`
/// for exactly that quartet in host `f64`, taking the roots from the *vendor's*
/// own `CINTrys_roots`, so the only things under test are the factor chain, the
/// pair data and the summation order. If this matches the vendor bit-for-bit
/// the recipe is right and the divergence is inside the device kernel; if it
/// does not, the recipe is what is wrong.
#[test]
fn locate_the_last_ulp_on_ss_ss() {
    let (atm, bas, env) = build_h2o_sto3g();
    let shls = [3_i32, 3, 1, 3];

    let cfs0 = 0.282094791773878143_f64;
    let sqrtpi = 1.7724538509055160272981674833411451_f64;
    let pi = std::f64::consts::PI;
    // `g2e.c:54-56`, chained left to right.
    let common_factor = (pi * pi * pi) * 2.0 / sqrtpi * cfs0 * cfs0 * cfs0 * cfs0;

    let shell = |s: usize| -> (Vec<f64>, Vec<f64>, [f64; 3]) {
        let nprim = bas[s * BAS_SLOTS + NPRIM_OF] as usize;
        let pe = bas[s * BAS_SLOTS + PTR_EXP] as usize;
        let pc = bas[s * BAS_SLOTS + PTR_COEFF] as usize;
        let a = bas[s * BAS_SLOTS + ATOM_OF] as usize;
        let pr = atm[a * ATM_SLOTS + PTR_COORD] as usize;
        (
            env[pe..pe + nprim].to_vec(),
            env[pc..pc + nprim].to_vec(),
            [env[pr], env[pr + 1], env[pr + 2]],
        )
    };
    let (ai, ci, ri) = shell(shls[0] as usize);
    let (aj, cj, rj) = shell(shls[1] as usize);
    let (ak, ck, rk) = shell(shls[2] as usize);
    let (al, cl, rl) = shell(shls[3] as usize);

    let sq = |a: [f64; 3], b: [f64; 3]| {
        let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    };
    let rr_ij = sq(ri, rj);
    let rr_kl = sq(rk, rl);

    // `CINTset_pairdata` (optimizer.c:320-333).
    let pair = |a1: f64, a2: f64, r1: [f64; 3], r2: [f64; 3], rr: f64| {
        let inv = 1.0 / (a1 + a2);
        let e = rr * a1 * a2 * inv;
        let w2 = a2 * inv;
        (
            (-e).exp(),
            [
                r1[0] + w2 * (r2[0] - r1[0]),
                r1[1] + w2 * (r2[1] - r1[1]),
                r1[2] + w2 * (r2[2] - r1[2]),
            ],
        )
    };

    // `CINT2e_loop` (cint2e.c:72-119): lp, kp, jp, ip.
    let mut gout = 0.0_f64;
    let mut first = true;
    for lp in 0..al.len() {
        let fac1l = common_factor * cl[lp];
        for kp in 0..ak.len() {
            // `CINT2e_loop_nopt` (`cint2e.c:202-213`) does NOT use
            // `CINTset_pairdata` for the ket: it forms it inline, with a true
            // division and the weighted-sum centre.
            let akl = ak[kp] + al[lp];
            let ekl0 = rr_kl * ak[kp] * al[lp] / akl;
            let rkl = [
                (ak[kp] * rk[0] + al[lp] * rl[0]) / akl,
                (ak[kp] * rk[1] + al[lp] * rl[1]) / akl,
                (ak[kp] * rk[2] + al[lp] * rl[2]) / akl,
            ];
            let expkl = (-ekl0).exp();
            let fac1k = fac1l * ck[kp];
            for jp in 0..aj.len() {
                let fac1j = fac1k * cj[jp];
                for ip in 0..ai.len() {
                    let (expij, rij) = pair(ai[ip], aj[jp], ri, rj, rr_ij);
                    // `expijkl = pdata_ij->eij * ekl` then
                    // `fac1i = fac1j*ci[ip]*expijkl` (`cint2e.c:238-240`) — the
                    // two exponentials multiply each other first.
                    let expijkl = expij * expkl;
                    let fac1i = fac1j * ci[ip] * expijkl;
                    let aij = ai[ip] + aj[jp];
                    let a1 = aij * akl;
                    let a0 = a1 / (aij + akl);
                    // `g2e.c:17`
                    let fac1 = (a0 / (a1 * a1 * a1)).sqrt() * fac1i;
                    let x = a0 * sq(rij, rkl);
                    let (_u, w) = vendor_ffi::vendor_CINTrys_roots(1, x);
                    let term = w[0] * fac1;
                    // `gempty`: the first primitive quartet assigns.
                    if first {
                        gout = term;
                        first = false;
                    } else {
                        gout += term;
                    }
                }
            }
        }
    }

    let mut vendor = vec![0.0_f64; 1];
    let _ = vendor_ffi::vendor_int2e_cart(&mut vendor, &shls, &atm, 3, &bas, 5, &env);
    let mut cintx = vec![0.0_f64; 1];
    eval(RawApiId::INT2E_CART, &mut cintx, &shls, &atm, &bas, &env);

    println!("\n=== (ss|ss) at shells {shls:?} ===");
    println!(
        "  hand-rolled libcint recipe : {gout:.17e}  bits={:016x}",
        gout.to_bits()
    );
    println!(
        "  vendored libcint           : {:.17e}  bits={:016x}",
        vendor[0],
        vendor[0].to_bits()
    );
    println!(
        "  cintx                      : {:.17e}  bits={:016x}",
        cintx[0],
        cintx[0].to_bits()
    );
    println!(
        "  recipe==vendor: {}   cintx==vendor: {}",
        gout.to_bits() == vendor[0].to_bits(),
        cintx[0].to_bits() == vendor[0].to_bits()
    );

    // Both are gates now, and they pin the two things the investigation found:
    // that the oracle's `opt == NULL` call takes `CINT2e_loop_nopt`, and that
    // its ket pair data and `expijkl` grouping are what cintx must reproduce.
    assert_eq!(
        gout.to_bits(),
        vendor[0].to_bits(),
        "the hand-rolled `CINT2e_loop_nopt` recipe no longer reproduces the vendor"
    );
    assert_eq!(
        cintx[0].to_bits(),
        vendor[0].to_bits(),
        "cintx's (ss|ss) is no longer bit-identical to the vendor"
    );
}
