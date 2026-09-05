//! Primitive-pair data and libcint's `expcutoff` screen
//! (`def2_speed_memory_optimization_plan.md` S1).
//!
//! # What this replaces
//!
//! The batched 2e kernel used to walk `nprim_i · nprim_j · nprim_k · nprim_l`
//! primitive quartets per shell quartet, forming the ket's product centre and
//! overlap exponential inside the bra loop — so `rkl` and `exp(-mu_kl R_kl^2)`
//! were recomputed `nprim_i · nprim_j` times for every ket primitive pair.
//!
//! libcint does neither. `CINTset_pairdata` (`optimizer.c:288`) forms each bra
//! pair's `(rij, exp(-eij), cceij)` **once per shell pair**, and
//! `CINT2e_loop_nopt` (`cint2e.c:192`) runs the ket pair as the *outer* loop so
//! its data is formed once per ket primitive pair. On top of that it drops any
//! primitive pair or quartet whose estimated contribution is below
//! `exp(-expcutoff)`:
//!
//! - pair level, in `CINTset_pairdata`: keep only `cceij < expcutoff`;
//! - ket level, `cint2e.c:205`: skip the whole ket pair when `ccekl > expcutoff`;
//! - quartet level, `cint2e.c:232`: skip when `cceij > expcutoff - ccekl`.
//!
//! Those terms are therefore **absent from the vendored reference cintx is
//! compared against**. Computing them cost time and moved cintx away from the
//! vendor, not towards it.
//!
//! # The estimate, transcribed
//!
//! `cceij` is a log-scale bound on one primitive pair's contribution:
//!
//! ```text
//! log_rr_ij = 1.7 - 1.5 * ln(a_i[last] + a_j[last])
//!           + (li + lj) * ln(sqrt(rr_ij) + 1)          [when li + lj > 0]
//! aij       = 1 / (a_i[p] + a_j[q])
//! eij       = rr_ij * a_i[p] * a_j[q] * aij
//! cceij     = eij - log_rr_ij - ln(max_c |c_i[p]|) - ln(max_c |c_j[q]|)
//! ```
//!
//! Every expression is transcribed operation for operation from the vendor,
//! including the association in `eij` (`rr * ai * aj * aij`, with `aij` the
//! reciprocal formed once) and the `rij = ri + wj * (rj - ri)` product centre —
//! the same two choices whose 1e counterparts were worth 2000-4400x at high
//! angular momentum. `approx_log` is `#define approx_log log` in
//! `optimizer.h:76`, so it is `f64::ln` here and not an approximation.
//!
//! The default `expcutoff` is libcint's `EXPCUTOFF = 60`
//! (`src/cint_config.h.in:27`), i.e. a threshold of `exp(-60) ~= 9e-27` on the
//! estimate — far below the project's `1e-12` oracle tolerance, which is why
//! adopting it moves results by less than the tolerance while removing work.
//!
//! # Disabling it
//!
//! [`PairTableOptions::unscreened`] sets the cutoff to `+inf`, which keeps every
//! primitive pair and makes both kernel-side tests unconditionally true. That is
//! the A/B reference the S1 gate compares against: the same kernel, the same
//! loop order, every primitive quartet evaluated.

use super::two_electron::BatchShell;
use cintx_runtime::ExecutionPlan;

/// libcint's default `expcutoff` for two-electron integrals.
///
/// `EXPCUTOFF` in `src/cint_config.h.in:27`, applied by `g2e.c:57` when
/// `env[PTR_EXPCUTOFF]` is zero — which is what every cintx caller passes today,
/// since the raw API's `env` comes from `cintx-basis` and never sets slot 0.
pub const LIBCINT_EXPCUTOFF: f64 = 60.0;

/// `f64` slots per surviving primitive pair in [`PairTable::data`].
///
/// `[rij_x, rij_y, rij_z, eij, cceij]` — the product centre, the overlap
/// exponential libcint calls `pdata->eij`, and the log-scale estimate the
/// quartet-level cutoff compares.
pub const PAIR_DATA_STRIDE: usize = 5;

/// `u32` slots per surviving primitive pair in [`PairTable::index`].
///
/// `[p_bra, p_ket]` — the primitive indices, needed to reach the contraction
/// coefficients and the exponents the kernel still forms `aij` from.
pub const PAIR_INDEX_STRIDE: usize = 2;

/// How a [`PairTable`] screens.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairTableOptions {
    /// The `expcutoff` threshold, in the log-scale units `cceij` is measured in.
    pub expcutoff: f64,
}

impl Default for PairTableOptions {
    /// libcint's own default.
    fn default() -> Self {
        Self {
            expcutoff: LIBCINT_EXPCUTOFF,
        }
    }
}

impl PairTableOptions {
    /// Keep every primitive pair: the A/B reference for the S1 gate.
    #[must_use]
    pub fn unscreened() -> Self {
        Self {
            expcutoff: f64::INFINITY,
        }
    }

    /// Is this the unscreened setting?
    #[must_use]
    pub fn is_unscreened(&self) -> bool {
        self.expcutoff.is_infinite() && self.expcutoff.is_sign_positive()
    }
}

/// Per-primitive `ln(max_c |c|)` — libcint's `CINTOpt_log_max_pgto_coeff`
/// (`optimizer.c:248`), read out of cintx's primitive-major coefficient
/// layout (`coefficients[p * nctr + c]`). `ln(0)` is `-inf`, which makes a
/// primitive's `cceij` `+inf` and drops it; that mirrors the vendor, and is
/// correct — a primitive with zero coefficient in every contraction
/// contributes nothing.
pub(crate) fn log_max_abs_coeff(nprim: usize, nctr: usize, coefficients: &[f64]) -> Vec<f64> {
    (0..nprim)
        .map(|p| {
            let maxc = (0..nctr)
                .map(|c| coefficients[p * nctr + c].abs())
                .fold(0.0_f64, f64::max);
            maxc.ln()
        })
        .collect()
}

/// The `expcutoff` screen for one shell pair (`CINTset_pairdata`,
/// `optimizer.c:288-341`), computed directly from a bra/ket shell's raw
/// fields rather than through a whole-basis [`PairTable`].
///
/// [`PairTable::push_shell_pair`] is the primary consumer — a whole basis,
/// one shell pair at a time. The derivative and property 2e launchers in
/// [`crate::kernels::two_electron`] (`int2e_ip1`, `_ip2`, `_hess2e`,
/// `_gauge2e`, `_giao2e`) are the other one: their primitive loop walks one
/// shell quartet directly and has no `[BatchShell]` slice to build a table
/// from, so they use this same formula without a table around it — S1's
/// follow-up, so a caller-supplied `env[PTR_EXPCUTOFF]` screens those
/// operators too, not only the plain-Coulomb scalar/batched path.
pub(crate) struct PairScreen {
    pub(crate) rr: f64,
    log_rr: f64,
}

impl PairScreen {
    /// # Panics
    /// Indexes `bra_exponents[bra_nprim - 1]` / `ket_exponents[ket_nprim -
    /// 1]`, so panics if either `nprim` is `0`. Every shell reaching a
    /// dispatched quartet has `nprim >= 1` (`cintx_core::Shell::try_new`
    /// rejects `0` at construction), so this is an invariant of the caller,
    /// not a validated input.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        bra_l: u8,
        bra_nprim: usize,
        bra_exponents: &[f64],
        bra_center: [f64; 3],
        ket_l: u8,
        ket_nprim: usize,
        ket_exponents: &[f64],
        ket_center: [f64; 3],
    ) -> Self {
        let d = [
            bra_center[0] - ket_center[0],
            bra_center[1] - ket_center[1],
            bra_center[2] - ket_center[2],
        ];
        let rr = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];

        // `aij` from the *last* primitive of each shell — the most diffuse pair,
        // and so the loosest bound. Verbatim `optimizer.c:301`.
        let a_last = bra_exponents[bra_nprim - 1] + ket_exponents[ket_nprim - 1];
        let mut log_rr = 1.7 - 1.5 * a_last.ln();
        let lij = u32::from(bra_l) + u32::from(ket_l);
        if lij > 0 {
            // The `omega >= 0` arm: every caller of this screen is plain
            // Coulomb, so the range-separated `theta * r_guess` term
            // (`optimizer.c:311`) does not apply.
            log_rr += f64::from(lij) * (rr.sqrt() + 1.0).ln();
        }
        Self { rr, log_rr }
    }

    /// `cceij` for bra primitive `p` (exponent `ap`, `ln(max_c |c|)`
    /// `log_maxc_p`) and ket primitive `q` (exponent `aq`, `log_maxc_q`).
    pub(crate) fn cceij(&self, ap: f64, aq: f64, log_maxc_p: f64, log_maxc_q: f64) -> f64 {
        let aij = 1.0 / (ap + aq);
        let eij = self.rr * ap * aq * aij;
        eij - self.log_rr - log_maxc_p - log_maxc_q
    }
}

/// The `expcutoff` screening state for one shell quartet, shared by the
/// derivative and property 2e launchers in [`crate::kernels::two_electron`]
/// (`int2e_ip1`, `_ip2`, `_hess2e`, `_gauge2e`, `_giao2e`): each walks one
/// shell quartet directly (from a [`ValidatedShellTuple`](cintx_runtime::ValidatedShellTuple),
/// not a `[BatchShell]`) and has no table to build, so they all build this
/// same [`PairScreen`] pair (S1's follow-up) instead of duplicating it.
pub(crate) struct QuartetExpScreen {
    pub(crate) expcutoff: f64,
    pub(crate) bra_screen: PairScreen,
    pub(crate) ket_screen: PairScreen,
    pub(crate) log_maxc_i: Vec<f64>,
    pub(crate) log_maxc_j: Vec<f64>,
    pub(crate) log_maxc_k: Vec<f64>,
    pub(crate) log_maxc_l: Vec<f64>,
}

impl QuartetExpScreen {
    /// `nprim[n]`/`nctr[n]`/`exponents[n]`/`coefficients[n]` describe the same
    /// shell as `ls[n]`/`centers[n]`, in bra-then-ket `(i, j, k, l)` order.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        plan: &ExecutionPlan<'_>,
        ls: [u8; 4],
        nprim: [usize; 4],
        nctr: [usize; 4],
        exponents: [&[f64]; 4],
        coefficients: [&[f64]; 4],
        centers: [[f64; 3]; 4],
    ) -> Self {
        let [li, lj, lk, ll] = ls;
        let [n_prim_i, n_prim_j, n_prim_k, n_prim_l] = nprim;
        let [n_ctr_i, n_ctr_j, n_ctr_k, n_ctr_l] = nctr;
        let [exp_i, exp_j, exp_k, exp_l] = exponents;
        let [coef_i, coef_j, coef_k, coef_l] = coefficients;
        let [ri, rj, rk, rl] = centers;

        let expcutoff = plan
            .operator_env_params
            .expcutoff
            .unwrap_or(LIBCINT_EXPCUTOFF);
        let bra_screen = PairScreen::new(li, n_prim_i, exp_i, ri, lj, n_prim_j, exp_j, rj);
        let ket_screen = PairScreen::new(lk, n_prim_k, exp_k, rk, ll, n_prim_l, exp_l, rl);
        let log_maxc_i = log_max_abs_coeff(n_prim_i, n_ctr_i, coef_i);
        let log_maxc_j = log_max_abs_coeff(n_prim_j, n_ctr_j, coef_j);
        let log_maxc_k = log_max_abs_coeff(n_prim_k, n_ctr_k, coef_k);
        let log_maxc_l = log_max_abs_coeff(n_prim_l, n_ctr_l, coef_l);

        Self {
            expcutoff,
            bra_screen,
            ket_screen,
            log_maxc_i,
            log_maxc_j,
            log_maxc_k,
            log_maxc_l,
        }
    }
}

/// Primitive-pair data for every ordered shell pair of a basis.
///
/// Ordered rather than canonical because a quartet's bra is `(i, j)` and its ket
/// is `(k, l)` in the caller's order, and `rij` is not symmetric under the swap
/// — `ri + wj * (rj - ri)` and `rj + wi * (ri - rj)` are the same point in exact
/// arithmetic and not the same `f64`. Storing both orders costs `nbas^2` rows
/// where `nbas(nbas+1)/2` would do, and buys an exact match with whichever order
/// libcint was handed.
#[derive(Clone, Debug, Default)]
pub struct PairTable {
    /// [`PAIR_DATA_STRIDE`] `f64` per surviving primitive pair.
    pub data: Vec<f64>,
    /// [`PAIR_INDEX_STRIDE`] `u32` per surviving primitive pair.
    pub index: Vec<u32>,
    /// Prefix sums over ordered shell pairs: `nbas * nbas + 1` entries, so pair
    /// `(i, j)`'s rows are `offset[i * nbas + j] .. offset[i * nbas + j + 1]`.
    pub offset: Vec<u32>,
    /// Shell count the offsets are indexed by.
    pub nbas: u32,
    /// The threshold this table was compacted at.
    pub expcutoff: f64,
    /// Primitive pairs the pair-level screen dropped, summed over shell pairs.
    pub pairs_dropped: u64,
    /// Primitive pairs retained, summed over shell pairs.
    pub pairs_kept: u64,
}

impl PairTable {
    /// Build the table for every ordered shell pair of `shells`.
    ///
    /// # Panics
    /// Panics if `shells` has more than `u32::MAX` entries, which no basis this
    /// library can represent reaches.
    #[must_use]
    pub fn build(shells: &[BatchShell], options: PairTableOptions) -> Self {
        let nbas = shells.len();
        assert!(u32::try_from(nbas).is_ok(), "shell count exceeds u32");

        let log_maxc: Vec<Vec<f64>> = shells
            .iter()
            .map(|shell| {
                log_max_abs_coeff(
                    shell.nprim as usize,
                    shell.nctr as usize,
                    &shell.coefficients,
                )
            })
            .collect();

        let mut table = Self {
            offset: Vec::with_capacity(nbas * nbas + 1),
            nbas: nbas as u32,
            expcutoff: options.expcutoff,
            ..Self::default()
        };
        table.offset.push(0);

        for (bra_index, bra) in shells.iter().enumerate() {
            for (ket_index, ket) in shells.iter().enumerate() {
                table.push_shell_pair(bra, ket, bra_index, ket_index, &log_maxc, options);
                table
                    .offset
                    .push((table.index.len() / PAIR_INDEX_STRIDE) as u32);
            }
        }
        table
    }

    /// Append one ordered shell pair's surviving primitive pairs.
    ///
    /// Transcribes `CINTset_pairdata` (`optimizer.c:288-341`), including its
    /// iteration order — `for jp { for ip { } }`, so the ket primitive is the
    /// outer index — because the kernel walks these rows in the order they are
    /// written and the accumulation order is what libcint's rounding is.
    ///
    /// A shell with `nprim == 0` contributes no pairs; returning early avoids
    /// the `nprim - 1` underflow the vendor's "last primitive" bound would
    /// otherwise hit (empty shells are rejected upstream by
    /// `cintx_core::Shell::try_new`, but `BatchShell` is public and unvalidated,
    /// so this stays a typed no-op rather than a panic for callers who bypass
    /// that layer).
    fn push_shell_pair(
        &mut self,
        bra: &BatchShell,
        ket: &BatchShell,
        bra_index: usize,
        ket_index: usize,
        log_maxc: &[Vec<f64>],
        options: PairTableOptions,
    ) {
        if bra.nprim == 0 || ket.nprim == 0 {
            return;
        }
        let (log_maxc_bra, log_maxc_ket) = (&log_maxc[bra_index], &log_maxc[ket_index]);
        let screen = PairScreen::new(
            bra.l,
            bra.nprim as usize,
            &bra.exponents,
            bra.center,
            ket.l,
            ket.nprim as usize,
            &ket.exponents,
            ket.center,
        );

        for (q, &aq) in ket.exponents[..ket.nprim as usize].iter().enumerate() {
            for (p, &ap) in bra.exponents[..bra.nprim as usize].iter().enumerate() {
                let cceij = screen.cceij(ap, aq, log_maxc_bra[p], log_maxc_ket[q]);
                if !(cceij < options.expcutoff) {
                    self.pairs_dropped += 1;
                    continue;
                }
                let aij = 1.0 / (ap + aq);
                let eij = screen.rr * ap * aq * aij;
                let wq = aq * aij;
                self.data.extend_from_slice(&[
                    bra.center[0] + wq * (ket.center[0] - bra.center[0]),
                    bra.center[1] + wq * (ket.center[1] - bra.center[1]),
                    bra.center[2] + wq * (ket.center[2] - bra.center[2]),
                    (-eij).exp(),
                    cceij,
                ]);
                self.index.push(p as u32);
                self.index.push(q as u32);
                self.pairs_kept += 1;
            }
        }
    }

    /// Surviving primitive pairs of ordered shell pair `(i, j)`.
    #[must_use]
    pub fn pair_count(&self, i: u32, j: u32) -> u32 {
        let slot = (i * self.nbas + j) as usize;
        self.offset[slot + 1] - self.offset[slot]
    }

    /// Bytes this table costs to upload.
    #[must_use]
    pub fn upload_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f64>()
            + (self.index.len() + self.offset.len()) * std::mem::size_of::<u32>()
    }

    /// Primitive quartets a work list still evaluates under this table.
    ///
    /// The pair-level product only: the quartet-level `cceij > expcutoff -
    /// ccekl` test is a function of two rows and is applied in the kernel, so
    /// this is an upper bound on what runs and an exact count of what is
    /// dispatched. Reported as `primitive_quartets_evaluated` (M6).
    #[must_use]
    pub fn primitive_quartets_in(&self, quartets: &[[u32; 4]]) -> u64 {
        quartets
            .iter()
            .map(|q| {
                u64::from(self.pair_count(q[0], q[1])) * u64::from(self.pair_count(q[2], q[3]))
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shell(l: u8, exps: &[f64], coeffs: &[f64], center: [f64; 3]) -> BatchShell {
        BatchShell {
            l,
            nprim: exps.len() as u32,
            nctr: (coeffs.len() / exps.len()) as u32,
            exponents: exps.to_vec(),
            coefficients: coeffs.to_vec(),
            center,
        }
    }

    fn two_s_shells() -> Vec<BatchShell> {
        vec![
            shell(0, &[13.0, 1.96, 0.44], &[0.03, 0.23, 0.81], [0.0, 0.0, 0.0]),
            shell(0, &[0.12], &[1.0], [0.0, 0.0, 1.4]),
        ]
    }

    /// The unscreened table must keep exactly `nprim_i * nprim_j` rows for every
    /// ordered pair — the property the S1 A/B rests on.
    #[test]
    fn unscreened_keeps_every_primitive_pair() {
        let shells = two_s_shells();
        let table = PairTable::build(&shells, PairTableOptions::unscreened());
        assert_eq!(table.pairs_dropped, 0);
        for i in 0..2u32 {
            for j in 0..2u32 {
                let expected = shells[i as usize].nprim * shells[j as usize].nprim;
                assert_eq!(table.pair_count(i, j), expected, "pair ({i},{j})");
            }
        }
        assert_eq!(table.offset.len(), 5);
        assert_eq!(
            table.data.len(),
            table.pairs_kept as usize * PAIR_DATA_STRIDE
        );
        assert_eq!(
            table.index.len(),
            table.pairs_kept as usize * PAIR_INDEX_STRIDE
        );
    }

    /// The product centre must be libcint's `ri + wj * (rj - ri)`, not the
    /// algebraically equal `(ai*ri + aj*rj)/aij` — the distinction the 1e fix
    /// on 2026-09-03 was worth thousands of ulps for at high `l`.
    #[test]
    fn product_centre_uses_the_vendor_association() {
        let shells = two_s_shells();
        let table = PairTable::build(&shells, PairTableOptions::unscreened());
        // Pair (1, 0): one bra primitive, three ket primitives.
        let start = table.offset[(1 * table.nbas) as usize] as usize;
        let (ap, aq) = (shells[1].exponents[0], shells[0].exponents[0]);
        let aij = 1.0 / (ap + aq);
        let wq = aq * aij;
        let expected = shells[1].center[2] + wq * (shells[0].center[2] - shells[1].center[2]);
        assert_eq!(table.data[start * PAIR_DATA_STRIDE + 2], expected);
    }

    /// A distant, diffuse pair falls off the cutoff; a tight on-centre one does
    /// not. This is the screen doing its job, not merely being present.
    #[test]
    fn cutoff_drops_distant_diffuse_pairs() {
        let far = vec![
            shell(0, &[0.05], &[1.0], [0.0, 0.0, 0.0]),
            shell(0, &[0.05], &[1.0], [0.0, 0.0, 400.0]),
        ];
        let screened = PairTable::build(&far, PairTableOptions::default());
        assert_eq!(screened.pair_count(0, 1), 0, "distant pair must be dropped");
        assert!(screened.pairs_dropped > 0);

        let near = two_s_shells();
        let kept = PairTable::build(&near, PairTableOptions::default());
        assert!(
            kept.pair_count(0, 0) > 0,
            "an on-centre tight pair must survive"
        );
    }

    /// A primitive whose coefficient is zero in every contraction contributes
    /// nothing, and `ln(0) = -inf` makes `cceij` `+inf`, which drops it. Worth
    /// pinning because it is the one input that reaches the cutoff as a
    /// non-finite value.
    #[test]
    fn zero_coefficient_primitive_is_dropped() {
        let shells = vec![
            shell(0, &[1.0, 2.0], &[1.0, 0.0], [0.0, 0.0, 0.0]),
            shell(0, &[1.0], &[1.0], [0.0, 0.0, 0.0]),
        ];
        let table = PairTable::build(&shells, PairTableOptions::default());
        assert_eq!(
            table.pair_count(0, 1),
            1,
            "only the non-zero-coefficient primitive survives"
        );
    }

    /// `primitive_quartets_in` must equal the product of the two pair counts —
    /// the number M6 reports as evaluated.
    #[test]
    fn primitive_quartet_count_is_the_pair_product() {
        let shells = two_s_shells();
        let table = PairTable::build(&shells, PairTableOptions::unscreened());
        assert_eq!(table.primitive_quartets_in(&[[0, 0, 1, 1]]), 9);
        assert_eq!(table.primitive_quartets_in(&[[0, 1, 0, 1]]), 9);
        assert_eq!(table.primitive_quartets_in(&[[1, 1, 1, 1]]), 1);
    }

    /// A `nprim == 0` `BatchShell` (unreachable through `cintx_core::Shell`, but
    /// `BatchShell` is a public, unvalidated cubecl-crate type) must not panic —
    /// it contributes zero pairs rather than underflowing the "last primitive"
    /// index.
    #[test]
    fn zero_nprim_shell_contributes_no_pairs_and_does_not_panic() {
        let shells = vec![
            BatchShell {
                l: 0,
                nprim: 0,
                nctr: 0,
                exponents: Vec::new(),
                coefficients: Vec::new(),
                center: [0.0, 0.0, 0.0],
            },
            shell(0, &[1.0], &[1.0], [0.0, 0.0, 1.4]),
        ];
        let table = PairTable::build(&shells, PairTableOptions::unscreened());
        assert_eq!(table.pair_count(0, 0), 0);
        assert_eq!(table.pair_count(0, 1), 0);
        assert_eq!(table.pair_count(1, 0), 0);
        assert_eq!(table.pair_count(1, 1), 1);
    }
}
