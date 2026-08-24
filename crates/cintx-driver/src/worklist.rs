//! Shell-pair and shell-quartet work-list enumeration.
//!
//! The whole point of the batched driver is that the unit of work is the
//! *list*, not one tuple: a def2-SVP water molecule offers ~3.1 k independent
//! quartets and a def2-TZVP one ~18 k, and dispatching them one at a time
//! throws that parallelism away.
//!
//! Enumeration uses the standard 8-fold permutational symmetry of a real
//! `(ij|kl)` integral:
//!
//! ```text
//! (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk) = (kl|ij) = (lk|ij) = (kl|ji) = (lk|ji)
//! ```
//!
//! so only `i >= j`, `k >= l`, and `ij >= kl` (with `ij = i(i+1)/2 + j`) are
//! generated.

use crate::basis_view::BasisView;

/// One canonical shell pair with `i >= j`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShellPair {
    pub i: u32,
    pub j: u32,
}

impl ShellPair {
    /// Compound index `i(i+1)/2 + j`, the canonical ordering key for pairs.
    #[must_use]
    pub fn compound_index(self) -> u64 {
        let i = u64::from(self.i);
        u64::from(self.j) + i * (i + 1) / 2
    }
}

/// One canonical shell quartet under 8-fold symmetry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShellQuartet {
    pub i: u32,
    pub j: u32,
    pub k: u32,
    pub l: u32,
}

impl ShellQuartet {
    #[must_use]
    pub fn shls(self) -> [i32; 4] {
        [self.i as i32, self.j as i32, self.k as i32, self.l as i32]
    }

    /// Permutational multiplicity: how many of the 8 equivalent orderings this
    /// canonical representative stands for. A Fock build must weight by this.
    #[must_use]
    pub fn degeneracy(self) -> u32 {
        let bra = if self.i == self.j { 1 } else { 2 };
        let ket = if self.k == self.l { 1 } else { 2 };
        let swap = if (self.i, self.j) == (self.k, self.l) {
            1
        } else {
            2
        };
        bra * ket * swap
    }
}

/// Every canonical shell pair (`i >= j`) of a basis.
#[must_use]
pub fn enumerate_pairs(basis: &BasisView<'_>) -> Vec<ShellPair> {
    let nbas = basis.nbas() as u32;
    let mut pairs = Vec::with_capacity((nbas as usize * (nbas as usize + 1)) / 2);
    for i in 0..nbas {
        for j in 0..=i {
            pairs.push(ShellPair { i, j });
        }
    }
    pairs
}

/// Every canonical shell quartet under 8-fold symmetry.
///
/// `pairs` must be the canonical pair list in compound-index order, as returned
/// by [`enumerate_pairs`].
#[must_use]
pub fn enumerate_quartets(pairs: &[ShellPair]) -> Vec<ShellQuartet> {
    let mut quartets = Vec::with_capacity(pairs.len() * (pairs.len() + 1) / 2);
    for (bra_index, bra) in pairs.iter().enumerate() {
        for ket in &pairs[..=bra_index] {
            quartets.push(ShellQuartet {
                i: bra.i,
                j: bra.j,
                k: ket.i,
                l: ket.j,
            });
        }
    }
    quartets
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compound indices must be dense and strictly increasing in canonical
    /// pair order — the property the quartet `ij >= kl` filter relies on.
    #[test]
    fn compound_index_is_dense_and_monotonic() {
        let mut expected = 0_u64;
        for i in 0..8_u32 {
            for j in 0..=i {
                assert_eq!(ShellPair { i, j }.compound_index(), expected);
                expected += 1;
            }
        }
    }

    /// n(n+1)/2 pairs, and P(P+1)/2 quartets over those pairs.
    #[test]
    fn quartet_count_matches_eightfold_symmetry() {
        for nbas in 1..12_usize {
            let pairs: Vec<ShellPair> = (0..nbas as u32)
                .flat_map(|i| (0..=i).map(move |j| ShellPair { i, j }))
                .collect();
            assert_eq!(pairs.len(), nbas * (nbas + 1) / 2);
            let quartets = enumerate_quartets(&pairs);
            let p = pairs.len();
            assert_eq!(quartets.len(), p * (p + 1) / 2);
        }
    }

    /// Every canonical quartet must satisfy i>=j, k>=l, ij>=kl, and the
    /// canonical set must cover each 8-fold orbit exactly once.
    #[test]
    fn canonical_quartets_cover_each_orbit_once() {
        let nbas = 5_u32;
        let pairs: Vec<ShellPair> = (0..nbas)
            .flat_map(|i| (0..=i).map(move |j| ShellPair { i, j }))
            .collect();
        let quartets = enumerate_quartets(&pairs);

        let canonical = |i: u32, j: u32, k: u32, l: u32| {
            let (i, j) = if i >= j { (i, j) } else { (j, i) };
            let (k, l) = if k >= l { (k, l) } else { (l, k) };
            let bra = ShellPair { i, j }.compound_index();
            let ket = ShellPair { i: k, j: l }.compound_index();
            if bra >= ket {
                (i, j, k, l)
            } else {
                (k, l, i, j)
            }
        };

        let mut seen = std::collections::HashSet::new();
        for q in &quartets {
            assert!(q.i >= q.j && q.k >= q.l);
            assert!(
                ShellPair { i: q.i, j: q.j }.compound_index()
                    >= ShellPair { i: q.k, j: q.l }.compound_index()
            );
            assert!(
                seen.insert((q.i, q.j, q.k, q.l)),
                "duplicate canonical quartet"
            );
        }

        // Every one of the nbas^4 orderings must map into the canonical set.
        for i in 0..nbas {
            for j in 0..nbas {
                for k in 0..nbas {
                    for l in 0..nbas {
                        assert!(
                            seen.contains(&canonical(i, j, k, l)),
                            "orbit ({i},{j},{k},{l}) is not represented"
                        );
                    }
                }
            }
        }
    }

    /// Degeneracies must sum to the full nbas^4 ordering count — the check that
    /// catches an off-by-one in the multiplicity rule.
    #[test]
    fn degeneracies_sum_to_full_ordering_count() {
        for nbas in 1..7_u32 {
            let pairs: Vec<ShellPair> = (0..nbas)
                .flat_map(|i| (0..=i).map(move |j| ShellPair { i, j }))
                .collect();
            let total: u64 = enumerate_quartets(&pairs)
                .iter()
                .map(|q| u64::from(q.degeneracy()))
                .sum();
            let n = u64::from(nbas);
            assert_eq!(total, n * n * n * n, "nbas={nbas}");
        }
    }
}
