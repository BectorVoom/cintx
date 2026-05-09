//! `cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle [--profile <p>]`
//!
//! Operator-driven wrapper for the ROCm full base-family oracle suite
//! (Phase 16-04 / D-15). Spawns:
//!
//! ```text
//! CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm \
//!   cargo test -p cintx-oracle --features rocm[,<profile>] -- --ignored
//! ```
//!
//! The double env-var + `#[ignore]` + `#[cfg(feature = "rocm")]` triple-gate
//! ensures the rocm oracle suite is impossible to run accidentally. D-15 is
//! explicit that this is **not** wired into CI — the dev host (Linux + AMD
//! ROCm 7.x) is the only place it runs. See
//! `.planning/seeds/gpu-ci-runners.md` for the deferred CI follow-up.
//!
//! Profile semantics mirror `oracle_compare`'s `--profile` allowlist:
//! - `base` (default): activates only the `rocm` feature.
//! - `with-f12`, `with-4c1e`, `with-f12+with-4c1e`: append the corresponding
//!   feature(s); rocm is always on. Note that the rocm tests currently only
//!   cover the base family (7 tests across 5 files); higher profiles compile
//!   and run cleanly but exercise no additional rocm-specific tests.

use anyhow::{Context, Result};
use std::process::Command;

/// Run the ROCm base-family oracle suite under the given profile.
///
/// `profile` defaults to `"base"` when `None`. Returns an error if the
/// `cargo test` invocation exits non-zero.
pub fn run_rocm_oracle(profile: Option<&str>) -> Result<()> {
    let profile = profile.unwrap_or("base");

    // Build the feature spec. `base` is just `rocm`; other profiles append
    // the matching feature flag (mirrors oracle_compare's `--profiles`).
    let features = if profile == "base" {
        "rocm".to_owned()
    } else if profile == "with-f12+with-4c1e" {
        "rocm,with-f12,with-4c1e".to_owned()
    } else {
        format!("rocm,{profile}")
    };

    let status = Command::new("cargo")
        .env("CINTX_ROCM_ORACLE", "1")
        .env("CINTX_BACKEND", "rocm")
        .args([
            "test",
            "-p",
            "cintx-oracle",
            "--features",
            &features,
            "--",
            "--ignored",
        ])
        .status()
        .context("spawn cargo test for rocm-oracle")?;

    if !status.success() {
        anyhow::bail!("rocm-oracle suite failed for profile `{profile}`");
    }
    println!("rocm-oracle suite passed for profile `{profile}`");
    Ok(())
}
