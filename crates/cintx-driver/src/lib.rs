//! Batched shell-quartet driver for cintx.
//!
//! The per-shell-tuple entry points (`eval_raw`, `SessionRequest`) mirror
//! libcint's API and are the right shape for compatibility — but they are the
//! wrong shape for throughput. Each call pays a full kernel launch, a fresh
//! upload of the shell's exponents and coefficients, and a blocking readback,
//! which is why the CubeCL path currently measures ~1124 us per ERI quartet
//! against libcint's ~5.8 us.
//!
//! This crate changes the unit of work from one quartet to a *list* of them:
//!
//! 1. [`worklist`] enumerates canonical quartets under 8-fold symmetry;
//! 2. [`screening`] drops quartets that cannot contribute (Cauchy-Schwarz);
//! 3. [`bucket`] groups the survivors into launch classes and sorts each bucket
//!    so neighbouring work-items do similar amounts of work;
//! 4. [`execute`] runs the buckets and reports auditable statistics.
//!
//! The execution backend is behind [`execute::QuartetEvaluator`] so the same
//! work-list can be driven through cintx or through a reference implementation.
//! That is what makes a like-for-like speed comparison possible: screening is an
//! algorithmic win and must be applied to both sides, or reported separately.

pub mod basis_view;
pub mod bucket;
pub mod error;
pub mod execute;
pub mod screening;
pub mod worklist;

pub use basis_view::BasisView;
pub use bucket::{Bucket, LaunchClass, LaunchTier, bucket_quartets, primitive_work};
pub use error::DriverError;
pub use execute::{BatchOutput, BatchStats, QuartetEvaluator, run_buckets};
pub use screening::{
    DiagonalEvaluator, SchwarzTable, ScreeningReport, build_schwarz_table, screen_quartets,
};
pub use worklist::{ShellPair, ShellQuartet, enumerate_pairs, enumerate_quartets};
