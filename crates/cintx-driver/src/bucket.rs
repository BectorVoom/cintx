//! Bucketing quartets into launch classes.
//!
//! The bucketing key is the **angular-momentum quartet only**, deliberately not
//! `(l, nprim, nctr)`. Keying on primitive counts would shatter a def2 basis
//! into hundreds of near-empty buckets — def2-TZVP oxygen alone has 11 distinct
//! shell signatures, so `11^4` combinations are reachable — and each tiny
//! bucket would pay a full launch. Instead the primitive counts ride along as
//! per-quartet data with dynamic loop bounds, and quartets are sorted within a
//! bucket by total primitive work so neighbouring work-items in a plane have
//! similar trip counts.
//!
//! That trades a little intra-plane divergence for roughly an order of
//! magnitude fewer launches, which is the right side of the trade when a launch
//! costs ~25 us and a primitive-loop iteration costs nanoseconds.

use crate::basis_view::BasisView;
use crate::worklist::ShellQuartet;
use std::collections::BTreeMap;

/// Launch class: the angular-momentum quartet, plus the derived Rys order that
/// decides device eligibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LaunchClass {
    pub angular_momenta: [u8; 4],
    pub nroots: u32,
}

impl LaunchClass {
    #[must_use]
    pub fn of(basis: &BasisView<'_>, quartet: ShellQuartet) -> Self {
        let angular_momenta = [
            basis.ang_momentum(quartet.i as usize),
            basis.ang_momentum(quartet.j as usize),
            basis.ang_momentum(quartet.k as usize),
            basis.ang_momentum(quartet.l as usize),
        ];
        let nroots = u32::from(
            angular_momenta[0] + angular_momenta[1] + angular_momenta[2] + angular_momenta[3],
        ) / 2
            + 1;
        Self {
            angular_momenta,
            nroots,
        }
    }

    /// `g_size = nroots * dli * dlk * dll * dlj` — the libcint 2e G-tensor
    /// element count, a verbatim mirror of `build_2e_shape`. The kernel needs
    /// `3 * g_size` doubles (x/y/z), which is what decides the launch tier.
    #[must_use]
    pub fn g_size(self) -> usize {
        let [li, lj, lk, ll] = self.angular_momenta.map(usize::from);
        let (dli, dlj) = if li > lj {
            (li + lj + 1, lj + 1)
        } else {
            (li + 1, li + lj + 1)
        };
        let (dlk, dll) = if lk > ll {
            (lk + ll + 1, ll + 1)
        } else {
            (lk + 1, lk + ll + 1)
        };
        self.nroots as usize * dli * dlk * dll * dlj
    }

    /// Bytes of G-tensor scratch one work-item needs.
    #[must_use]
    pub fn g_tensor_bytes(self) -> usize {
        3 * self.g_size() * std::mem::size_of::<f64>()
    }

    /// The launch tier this class belongs to, from its G-tensor footprint.
    #[must_use]
    pub fn tier(self, shared_memory_bytes: usize) -> LaunchTier {
        let bytes = self.g_tensor_bytes();
        if bytes <= LaunchTier::PRIVATE_BYTE_CEILING {
            LaunchTier::ThreadPerQuartet
        } else if bytes <= shared_memory_bytes {
            LaunchTier::CubePerQuartetShared
        } else {
            LaunchTier::CubePerQuartetGlobal
        }
    }
}

/// How a class must be launched, decided by its G-tensor footprint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaunchTier {
    /// G-tensor fits in per-work-item private storage; grid-stride over
    /// quartets. Covers (ss|ss) through (pp|pp).
    ThreadPerQuartet,
    /// One cube per quartet, G-tensor in shared memory, plane cooperating over
    /// the recursion index space. Covers up to (dd|dd) — all of def2-SVP.
    CubePerQuartetShared,
    /// One cube per quartet, G-tensor in a global scratch slab. Needed for the
    /// f and g quartets def2-TZVP introduces; bandwidth-bound.
    CubePerQuartetGlobal,
}

impl LaunchTier {
    /// Ceiling for treating the G-tensor as private per-work-item storage.
    /// Above this a thread-per-quartet launch would need more registers/local
    /// memory than any backend gives a single work-item.
    pub const PRIVATE_BYTE_CEILING: usize = 4 * 1024;
}

/// One bucket of quartets sharing a launch class.
#[derive(Clone, Debug)]
pub struct Bucket {
    pub class: LaunchClass,
    pub quartets: Vec<ShellQuartet>,
}

impl Bucket {
    #[must_use]
    pub fn len(&self) -> usize {
        self.quartets.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.quartets.is_empty()
    }
}

/// Total primitive-quartet trip count for one shell quartet — the sort key that
/// keeps divergence low inside a plane.
#[must_use]
pub fn primitive_work(basis: &BasisView<'_>, quartet: ShellQuartet) -> u64 {
    u64::from(basis.nprim(quartet.i as usize))
        * u64::from(basis.nprim(quartet.j as usize))
        * u64::from(basis.nprim(quartet.k as usize))
        * u64::from(basis.nprim(quartet.l as usize))
}

/// Group quartets by launch class and sort each bucket by primitive work.
#[must_use]
pub fn bucket_quartets(basis: &BasisView<'_>, quartets: &[ShellQuartet]) -> Vec<Bucket> {
    let mut grouped: BTreeMap<LaunchClass, Vec<ShellQuartet>> = BTreeMap::new();
    for &quartet in quartets {
        grouped
            .entry(LaunchClass::of(basis, quartet))
            .or_default()
            .push(quartet);
    }

    grouped
        .into_iter()
        .map(|(class, mut quartets)| {
            quartets.sort_by_key(|&q| primitive_work(basis, q));
            Bucket { class, quartets }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn class(l: [u8; 4]) -> LaunchClass {
        let nroots = u32::from(l[0] + l[1] + l[2] + l[3]) / 2 + 1;
        LaunchClass {
            angular_momenta: l,
            nroots,
        }
    }

    /// G-tensor sizes must reproduce `build_2e_shape` exactly; these are the
    /// numbers the tier decision and the shared-memory budget depend on.
    #[test]
    fn g_size_matches_libcint_shape_formula() {
        assert_eq!(class([0, 0, 0, 0]).g_size(), 1);
        assert_eq!(class([1, 1, 1, 1]).g_size(), 108);
        assert_eq!(class([2, 2, 1, 1]).g_size(), 360);
        assert_eq!(class([2, 2, 2, 2]).g_size(), 1125);
        assert_eq!(class([3, 3, 3, 3]).g_size(), 5488);
    }

    /// The def2 envelope boundaries, pinned: def2-SVP's worst quartet sits at
    /// Rys order 5 and fits shared memory; def2-TZVP's f quartet needs order 7
    /// and does not.
    #[test]
    fn def2_envelope_boundaries_are_where_expected() {
        let svp_worst = class([2, 2, 2, 2]);
        assert_eq!(svp_worst.nroots, 5);
        assert_eq!(svp_worst.g_tensor_bytes(), 27_000);
        assert_eq!(
            svp_worst.tier(48 * 1024),
            LaunchTier::CubePerQuartetShared,
            "def2-SVP must fit the shared-memory tier"
        );

        let tzvp_worst = class([3, 3, 3, 3]);
        assert_eq!(tzvp_worst.nroots, 7);
        assert_eq!(tzvp_worst.g_tensor_bytes(), 131_712);
        assert_eq!(
            tzvp_worst.tier(48 * 1024),
            LaunchTier::CubePerQuartetGlobal,
            "def2-TZVP f quartets exceed shared memory"
        );

        // Transition-metal def2-TZVP reaches g functions.
        assert_eq!(class([4, 4, 4, 4]).nroots, 9);
    }

    #[test]
    fn low_l_classes_use_the_private_tier() {
        assert_eq!(
            class([0, 0, 0, 0]).tier(48 * 1024),
            LaunchTier::ThreadPerQuartet
        );
        assert_eq!(
            class([1, 1, 1, 1]).tier(48 * 1024),
            LaunchTier::ThreadPerQuartet
        );
    }
}
