# Deferred Items

## 2026-03-28 — Plan 04 execution

- `cargo fmt --all` failed due a pre-existing workspace issue: `crates/cintx-rs/src/error.rs` was missing when rustfmt resolved module paths. Scope-limited formatting (`rustfmt crates/cintx-capi/src/*.rs`) was used for this plan.
- Multiple unrelated files were already being modified concurrently in the worktree by other parallel executors; this plan intentionally staged only `crates/cintx-capi/src/{errors.rs,shim.rs,lib.rs}`.
