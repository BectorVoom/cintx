---
id: cr01-profile-gate-dead-code
created: 2026-05-30T04:02:28.028Z
source: 23-REVIEW.md (CR-01)
severity: warning
resolves_phase: null
status: resolved
resolved: 2026-05-30
resolved_by: quick task 260530-j62
---

# RESOLVED: unreachable unstable-source profile-membership check

`validate_profile_and_source_gate` in `crates/cintx-compat/src/raw.rs` had two
`is_source_only()` blocks. The first returned `Ok(())` early when
`unstable_source_api_enabled()`, making the second block's
`is_compiled_in_profile("unstable-source")` check unreachable. A source-only operator
compiled into no available profile was silently accepted instead of rejected with
`UnsupportedApi`. Pre-existing (commit 319d055).

## Fix (quick task 260530-j62)

Consolidated to ONE reachable `is_source_only()` block. The code-review's literal
suggestion (just check `is_compiled_in_profile("unstable-source")`) was WRONG and would
have regressed two entries: `active_manifest_profile()` only ever returns
base/with-f12/with-4c1e/with-f12+with-4c1e (NEVER "unstable-source"), and the manifest
has two source-only `unstable::source::2e` symbols compiled in the BASE profiles, plus
origi/grids/breit/origk/ssc in the `unstable-source` profile. Correct rule: after the
feature gate, accept iff the symbol is compiled in the ACTIVE profile OR the
`unstable-source` profile; reject (UnsupportedApi) when in NEITHER.

Verified: cintx-compat 43, cubecl --lib 280, and `unstable_source_parity` 23/23 under
`--features cpu,unstable-source-api` + `CINTX_ORACLE_BUILD_VENDOR=1` (every real source
family resolves through the now-reachable gate — no false rejection).
