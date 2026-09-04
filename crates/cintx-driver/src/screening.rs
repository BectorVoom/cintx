//! Cauchy–Schwarz (Schwarz) prescreening.
//!
//! For a positive-definite two-electron operator,
//!
//! ```text
//! |(ij|kl)| <= sqrt((ij|ij)) * sqrt((kl|kl)) = Q_ij * Q_kl
//! ```
//!
//! so a quartet whose `Q_ij * Q_kl` falls below the tolerance cannot contribute
//! above it and may be skipped. On a real def2-TZVP molecule this removes the
//! large majority of quartets.
//!
//! Screening is an **algorithmic** win, not a kernel win. Any benchmark that
//! applies it to cintx must apply it to the reference implementation too, or
//! report the two separately — see `docs` on [`ScreeningReport`].

use crate::basis_view::BasisView;
use crate::error::DriverError;
use crate::worklist::{ShellPair, ShellQuartet};

/// Per-shell-pair Schwarz bounds `Q_ij`, indexed by compound index.
#[derive(Clone, Debug)]
pub struct SchwarzTable {
    values: Vec<f64>,
    /// Largest `Q` in the table; useful for reporting and for an early bail.
    max_q: f64,
}

impl SchwarzTable {
    /// Build a table from bounds already computed elsewhere (S6).
    ///
    /// `values` must be in compound-index order — the order
    /// [`crate::enumerate_pairs`] produces, `i` ascending and `j` in `0..=i` —
    /// so that [`Self::get`] finds each pair where it left it.
    ///
    /// This exists so a caller can build the bounds through the batched 2e
    /// surface (`cintx_cubecl::schwarz_bounds`) instead of one diagonal quartet
    /// at a time through [`DiagonalEvaluator`], and still screen with the same
    /// code the benchmark screens with. A production caller has no vendored
    /// libcint to hand [`build_schwarz_table`], and building the table through
    /// the per-tuple path would cost more than the screening saves.
    ///
    /// # Panics
    /// If `values`' length is not a triangular number `n(n+1)/2` — the shape a
    /// compound-index-ordered pair list always has. This is the shape a caller
    /// gets wrong most often: `cintx_cubecl::schwarz_bounds` returns a *square*
    /// `nbas*nbas` matrix (every `(i,j)` and its mirror `(j,i)`, so `get` can
    /// index it directly), not this table's packed triangular layout — passing
    /// it here unreindexed silently reads the wrong entries instead of failing.
    /// Reindex it first, or use [`Self::from_square_matrix`], which does that
    /// reindexing for you.
    #[must_use]
    pub fn from_pair_values(values: impl IntoIterator<Item = f64>) -> Self {
        let values: Vec<f64> = values.into_iter().collect();
        assert!(
            is_triangular_number(values.len()),
            "SchwarzTable::from_pair_values: {} values is not a triangular number \
             n(n+1)/2 — did you pass a square nbas*nbas matrix (e.g. from \
             cintx_cubecl::schwarz_bounds) without reindexing it? Use \
             SchwarzTable::from_square_matrix for that shape instead.",
            values.len()
        );
        let max_q = values.iter().copied().fold(0.0_f64, f64::max);
        Self { values, max_q }
    }

    /// Build a table from a square `nbas*nbas` bounds matrix (S6).
    ///
    /// This is the shape `cintx_cubecl::schwarz_bounds` returns: `bounds[i *
    /// nbas + j]` and its mirror `bounds[j * nbas + i]` both hold `Q_ij`. This
    /// constructor reindexes it into the packed compound-index order
    /// [`Self::from_pair_values`] expects, so a caller of `schwarz_bounds`
    /// never has to get that reindexing right by hand.
    ///
    /// # Panics
    /// If `bounds.len() != nbas * nbas`.
    #[must_use]
    pub fn from_square_matrix(bounds: &[f64], nbas: usize) -> Self {
        assert_eq!(
            bounds.len(),
            nbas * nbas,
            "SchwarzTable::from_square_matrix: expected a {nbas}x{nbas} matrix \
             ({} values), got {}",
            nbas * nbas,
            bounds.len()
        );
        Self::from_pair_values(
            (0..nbas as u32)
                .flat_map(|i| (0..=i).map(move |j| (i, j)))
                .map(move |(i, j)| bounds[i as usize * nbas + j as usize]),
        )
    }

    /// `Q_ij` for a canonical pair.
    #[must_use]
    pub fn get(&self, pair: ShellPair) -> f64 {
        self.values
            .get(pair.compound_index() as usize)
            .copied()
            .unwrap_or(0.0)
    }

    #[must_use]
    pub fn max(&self) -> f64 {
        self.max_q
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Is `len` a triangular number `n(n+1)/2` for some `n >= 0`?
///
/// The shape a compound-index-ordered pair list (`nbas` shells, `i` ascending,
/// `j` in `0..=i`) always has, and so the one thing that catches a caller
/// handing [`SchwarzTable::from_pair_values`] a square matrix by mistake.
fn is_triangular_number(len: usize) -> bool {
    // n(n+1)/2 == len  <=>  n == (isqrt(8*len + 1) - 1) / 2, and that root must
    // be an exact match (8*len+1 a perfect square) with an odd root (it always
    // is, when 8*len+1 — odd — is a perfect square at all).
    let discriminant = 8_u128 * len as u128 + 1;
    let root = discriminant.isqrt();
    root * root == discriminant && {
        let n = (root - 1) / 2;
        n * (n + 1) / 2 == len as u128
    }
}

/// Evaluates the diagonal `(ij|ij)` block for one pair.
///
/// Injected rather than hardcoded so the table can be built from cintx or from
/// a reference implementation with identical downstream behaviour — which is
/// what makes a like-for-like benchmark possible.
pub trait DiagonalEvaluator {
    /// Fill `out` with the `(ij|ij)` block. `out.len()` is the block size.
    ///
    /// # Errors
    /// Returns [`DriverError`] if the underlying evaluation fails.
    fn eval_diagonal(&mut self, pair: ShellPair, out: &mut [f64]) -> Result<(), DriverError>;
}

/// Build the Schwarz table for every canonical pair.
///
/// # Errors
/// Propagates any evaluator failure.
pub fn build_schwarz_table<E: DiagonalEvaluator>(
    basis: &BasisView<'_>,
    pairs: &[ShellPair],
    evaluator: &mut E,
) -> Result<SchwarzTable, DriverError> {
    let mut values = vec![0.0_f64; pairs.len()];
    let mut scratch: Vec<f64> = Vec::new();
    let mut max_q = 0.0_f64;

    for pair in pairs {
        let ni = basis.nsph(pair.i as usize);
        let nj = basis.nsph(pair.j as usize);
        let block = ni * nj * ni * nj;
        if scratch.len() < block {
            scratch.resize(block, 0.0);
        }
        let out = &mut scratch[..block];
        out.fill(0.0);
        evaluator.eval_diagonal(*pair, out)?;

        // Q_ij = sqrt(max_{ab} |(ab|ab)|) over the block's own diagonal.
        // Using the block maximum (rather than a norm) keeps the bound valid
        // element-wise, which is what the per-quartet skip decision needs.
        let mut peak = 0.0_f64;
        for a in 0..(ni * nj) {
            let diagonal = out[a * (ni * nj) + a].abs();
            if diagonal > peak {
                peak = diagonal;
            }
        }
        let q = peak.sqrt();
        max_q = max_q.max(q);
        values[pair.compound_index() as usize] = q;
    }

    Ok(SchwarzTable { values, max_q })
}

/// Outcome of applying the screen to a quartet list.
#[derive(Clone, Debug, PartialEq)]
pub struct ScreeningReport {
    pub total: usize,
    pub kept: usize,
    pub tolerance: f64,
}

impl ScreeningReport {
    #[must_use]
    pub fn skipped(&self) -> usize {
        self.total - self.kept
    }

    #[must_use]
    pub fn kept_fraction(&self) -> f64 {
        if self.total == 0 {
            return 1.0;
        }
        self.kept as f64 / self.total as f64
    }
}

/// Filter a quartet list by the Schwarz bound.
///
/// A `tolerance` of `0.0` keeps everything, which is the identity the
/// correctness gate relies on: screened output at `tolerance = 0` must equal
/// unscreened output exactly.
#[must_use]
pub fn screen_quartets(
    quartets: &[ShellQuartet],
    table: &SchwarzTable,
    tolerance: f64,
) -> (Vec<ShellQuartet>, ScreeningReport) {
    let total = quartets.len();
    let kept: Vec<ShellQuartet> = quartets
        .iter()
        .copied()
        .filter(|q| {
            if tolerance <= 0.0 {
                return true;
            }
            let bra = table.get(ShellPair { i: q.i, j: q.j });
            let ket = table.get(ShellPair { i: q.k, j: q.l });
            bra * ket > tolerance
        })
        .collect();
    let report = ScreeningReport {
        total,
        kept: kept.len(),
        tolerance,
    };
    (kept, report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(values: Vec<f64>) -> SchwarzTable {
        let max_q = values.iter().copied().fold(0.0_f64, f64::max);
        SchwarzTable { values, max_q }
    }

    #[test]
    fn triangular_number_check_matches_pair_counts() {
        for nbas in 0..12_usize {
            let len = nbas * (nbas + 1) / 2;
            assert!(is_triangular_number(len), "n={nbas} len={len}");
        }
        for len in [2, 4, 5, 8, 9, 11, 13, 14] {
            assert!(!is_triangular_number(len), "len={len} is not triangular");
        }
    }

    /// A caller handing `from_pair_values` a square `nbas*nbas` matrix
    /// unreindexed must be refused, not silently misread.
    #[test]
    #[should_panic(expected = "not a triangular number")]
    fn from_pair_values_rejects_a_square_matrix() {
        let nbas = 3;
        let _ = SchwarzTable::from_pair_values(vec![0.0; nbas * nbas]);
    }

    /// `from_square_matrix` reindexes the square layout `schwarz_bounds`
    /// produces into the same packed table `from_pair_values` would, by hand,
    /// for the compound-index-ordered pair list.
    #[test]
    fn from_square_matrix_matches_hand_reindexed_pair_values() {
        let nbas = 4_usize;
        // A square, symmetric matrix in the shape `schwarz_bounds` returns.
        let mut bounds = vec![0.0_f64; nbas * nbas];
        for i in 0..nbas {
            for j in 0..nbas {
                let q = (i.max(j) * 10 + i.min(j)) as f64;
                bounds[i * nbas + j] = q;
            }
        }

        let expected = SchwarzTable::from_pair_values((0..nbas as u32).flat_map(|i| {
            let bounds = &bounds;
            (0..=i).map(move |j| bounds[i as usize * nbas + j as usize])
        }));
        let got = SchwarzTable::from_square_matrix(&bounds, nbas);

        assert_eq!(got.len(), expected.len());
        for i in 0..nbas as u32 {
            for j in 0..=i {
                let pair = ShellPair { i, j };
                assert_eq!(got.get(pair), expected.get(pair), "pair ({i},{j})");
            }
        }
    }

    #[test]
    #[should_panic(expected = "expected a 3x3 matrix")]
    fn from_square_matrix_rejects_a_mismatched_length() {
        let _ = SchwarzTable::from_square_matrix(&[0.0; 5], 3);
    }

    /// Tolerance 0 must be the identity — the property the correctness gate
    /// uses to prove screening never changes results, only cost.
    #[test]
    fn zero_tolerance_keeps_everything() {
        let quartets = vec![
            ShellQuartet {
                i: 0,
                j: 0,
                k: 0,
                l: 0,
            },
            ShellQuartet {
                i: 1,
                j: 0,
                k: 1,
                l: 0,
            },
        ];
        let t = table(vec![1e-30, 1e-30, 1e-30]);
        let (kept, report) = screen_quartets(&quartets, &t, 0.0);
        assert_eq!(kept.len(), quartets.len());
        assert_eq!(report.skipped(), 0);
    }

    #[test]
    fn screens_products_below_tolerance() {
        // pairs: (0,0)->idx0, (1,0)->idx1, (1,1)->idx2
        let t = table(vec![1.0, 1e-8, 1.0]);
        let quartets = vec![
            ShellQuartet {
                i: 0,
                j: 0,
                k: 0,
                l: 0,
            }, // 1.0 * 1.0 = 1
            ShellQuartet {
                i: 1,
                j: 0,
                k: 1,
                l: 0,
            }, // 1e-8 * 1e-8 = 1e-16
        ];
        let (kept, report) = screen_quartets(&quartets, &t, 1e-10);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0], quartets[0]);
        assert_eq!(report.kept, 1);
        assert!((report.kept_fraction() - 0.5).abs() < 1e-12);
    }

    /// The bound must never discard a quartet whose true value exceeds the
    /// tolerance: Q_ij * Q_kl >= |(ij|kl)| is the whole justification.
    #[test]
    fn bound_is_conservative_for_equal_pairs() {
        // For the diagonal quartet (ij|ij), the bound is exact: Q^2 = (ij|ij).
        let t = table(vec![2.0]);
        let q = ShellQuartet {
            i: 0,
            j: 0,
            k: 0,
            l: 0,
        };
        let (kept, _) = screen_quartets(&[q], &t, 3.999);
        assert_eq!(kept.len(), 1, "an exact-bound quartet must survive");
        let (dropped, _) = screen_quartets(&[q], &t, 4.001);
        assert_eq!(dropped.len(), 0);
    }
}
