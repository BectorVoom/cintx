//! Kernel launch functions for unstable-source API families.
//!
//! Phase 14 — all five unstable-source families implemented:
//!   - grids: grid-point integrals with NGRIDS env parameter (cint1e_grids.c)
//!   - breit: Breit spinor-only 2e integrals (breit.c)
//!   - origi: origin-displaced r^n one-electron integrals (cint1e_a.c)
//!   - origk: origin-k-displaced 3c1e integrals (cint3c1e_a.c)
//!   - ssc: spin-spin contact 3c2e integral (cint3c2e.c)
//!
//! This module was split from the original single-file `unstable.rs` into a
//! per-family directory module. The split is behavior-preserving (move-only):
//! cross-family helpers live in `shared`, and each family's launch fn plus its
//! private helpers live in its own submodule. The five `launch_*` entry points
//! are re-exported below so existing callers (`kernels::mod::resolve_family_name`)
//! continue to resolve `unstable::launch_{origi,grids,breit,origk,ssc}` unchanged.

mod breit;
mod grids;
mod origi;
mod origk;
mod shared;
mod ssc;

pub use breit::launch_breit;
pub use grids::launch_grids;
pub use origi::launch_origi;
pub use origk::launch_origk;
pub use ssc::launch_ssc;
