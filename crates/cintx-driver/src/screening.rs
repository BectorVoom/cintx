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
