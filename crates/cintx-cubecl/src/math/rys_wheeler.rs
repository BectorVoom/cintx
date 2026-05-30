//! Host Wheeler/Jacobi Rys root+weight engine for `nroots >= 6` (Phase 25 FND-02).
//!
//! Phase 25 Task 0 scaffold: this module currently exposes only the dispatch entry
//! `rys_roots_host_wheeler`, which is wired into `rys::rys_roots_host_f64`'s `nroots > 5`
//! arm so the host dispatcher no longer panics outright. The real verbatim port of
//! libcint's Flocke modified-moments -> Wheeler recursion -> vendored-MRRR eigensolve
//! -> root transform lands in Task 1b (consuming `eigh::cint_diagonalize` from Task 1a).

/// Compute the `nroots` Rys roots and weights for `nroots >= 6` host-side.
///
/// Task 0 scaffold: unimplemented. Task 1b replaces this with the verbatim
/// Flocke/Wheeler port. Returns `(roots, weights)` each of length `nroots`.
pub fn rys_roots_host_wheeler(nroots: usize, _x: f64) -> (Vec<f64>, Vec<f64>) {
    unimplemented!(
        "rys_roots_host_wheeler(nroots={nroots}): host Wheeler nroots>=6 engine \
         lands in Phase 25 Task 1b"
    )
}
