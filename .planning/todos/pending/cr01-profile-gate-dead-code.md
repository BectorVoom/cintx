---
id: cr01-profile-gate-dead-code
created: {
  timestamp: 2026-05-30T04:02:28.028Z
}
source: 23-REVIEW.md (CR-01)
severity: warning
resolves_phase: null
---

# Pre-existing: unreachable unstable-source profile-membership check

`validate_profile_and_source_gate` in `crates/cintx-compat/src/raw.rs` has two
`is_source_only()` blocks. The first (raw.rs:997-1007) returns `Ok(())` early when
`unstable_source_api_enabled()`, making the second block's
`is_compiled_in_profile("unstable-source")` check (raw.rs:1023) unreachable. A
source-only operator that is NOT compiled into the unstable-source profile would be
silently accepted instead of rejected with `UnsupportedApi`.

PRE-EXISTING (introduced in commit 319d055, present at phase-23 base 3d4a714) — NOT a
phase-23 regression. Does not affect phase-23 DRV1 families (none are source-only).
Fix: drop the dead early-return, or fold the profile-membership check into the first block.
