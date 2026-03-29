---
phase: 04-verification-release-automation
plan: 05
type: execute
wave: 5
depends_on:
  - 04
files_modified:
  - crates/cintx-oracle/src/lib.rs
autonomous: true
requirements:
  - VERI-01
  - VERI-02
  - VERI-03
  - VERI-04
gap_closure: true
must_haves:
  truths:
    - "Oracle crate lib surface is substantive (>=20 lines) and explicitly re-exports profile-aware Phase 4 entry points used by verification gates."
    - "Profile-aware fixture/parity exports remain explicit and importable from the crate root for CI and xtask consumers."
  artifacts:
    - path: crates/cintx-oracle/src/lib.rs
      provides: "Non-stub oracle crate export hub with explicit profile-aware re-exports."
      min_lines: 20
  key_links:
    - from: crates/cintx-oracle/src/lib.rs
      to: crates/cintx-oracle/src/fixtures.rs
      via: "crate root re-exports required profile fixture builders/constants used by Phase 4 tooling."
      pattern: "build_profile_representation_matrix|build_required_profile_matrices|PHASE4_APPROVED_PROFILES|PHASE4_ORACLE_FAMILIES"
    - from: crates/cintx-oracle/src/lib.rs
      to: crates/cintx-oracle/src/compare.rs
      via: "crate root re-exports profile-aware parity/tolerance/helper verification APIs."
      pattern: "generate_profile_parity_report|generate_phase2_parity_report|verify_helper_surface_coverage|tolerance_for_family"
---

<objective>
Close the remaining Phase 04 verification gap by fixing the `crates/cintx-oracle/src/lib.rs` artifact substance gate.
Purpose: Resolve the 04-VERIFICATION failure (`17 < 20` lines) while preserving explicit profile-aware exports required by Decisions D-01, D-03, D-04, and D-14.
Output: Expanded crate-root oracle exports that remain explicit, profile-aware, and gate-consumable.
</objective>

<execution_context>
@/home/chemtech/.codex/get-shit-done/workflows/execute-plan.md
@/home/chemtech/.codex/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/REQUIREMENTS.md
@.planning/STATE.md
@.planning/phases/04-verification-release-automation/04-CONTEXT.md
@.planning/phases/04-verification-release-automation/04-VERIFICATION.md
@.planning/phases/04-verification-release-automation/01-PLAN-SUMMARY.md
@AGENTS.md
@crates/cintx-oracle/src/lib.rs
@crates/cintx-oracle/src/fixtures.rs
@crates/cintx-oracle/src/compare.rs
<interfaces>
From `crates/cintx-oracle/src/fixtures.rs`:
```rust
pub const PHASE4_APPROVED_PROFILES: &[&str];
pub const PHASE4_ORACLE_FAMILIES: &[&str];
pub fn build_profile_representation_matrix(
    inputs: &OracleRawInputs,
    profile: &str,
    include_unstable_source: bool,
) -> Result<Vec<OracleFixture>>;
pub fn build_required_profile_matrices(
    inputs: &OracleRawInputs,
) -> Result<Vec<ProfileRepresentationMatrix>>;
```

From `crates/cintx-oracle/src/compare.rs`:
```rust
pub fn tolerance_for_family(family: &str) -> Result<FamilyTolerance>;
pub fn verify_helper_surface_coverage(inputs: &OracleRawInputs) -> Result<()>;
pub fn generate_profile_parity_report(
    inputs: &OracleRawInputs,
    profile: &str,
    include_unstable_source: bool,
) -> Result<Phase2ParityReport>;
pub fn generate_phase2_parity_report(inputs: &OracleRawInputs) -> Result<Phase2ParityReport>;
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Raise `cintx-oracle` crate-root export surface above the substance floor without changing gate semantics</name>
  <files>crates/cintx-oracle/src/lib.rs</files>
  <read_first>crates/cintx-oracle/src/lib.rs, crates/cintx-oracle/src/fixtures.rs, crates/cintx-oracle/src/compare.rs, .planning/phases/04-verification-release-automation/04-VERIFICATION.md, .planning/phases/04-verification-release-automation/04-CONTEXT.md</read_first>
  <action>
Expand `crates/cintx-oracle/src/lib.rs` from 17 lines to at least 20 lines by adding explicit crate-root `pub use` exports for the existing profile-aware APIs and constants. Keep `pub mod compare;` and `pub mod fixtures;` intact. Add one grouped `pub use fixtures::{...}` that includes `build_profile_representation_matrix`, `build_required_profile_matrices`, `PHASE4_APPROVED_PROFILES`, and `PHASE4_ORACLE_FAMILIES`. Add one grouped `pub use compare::{...}` that includes `generate_profile_parity_report`, `generate_phase2_parity_report`, `verify_helper_surface_coverage`, `tolerance_for_family`, `Phase2ParityReport`, and `FamilyTolerance`. Preserve the existing compile-edge test semantics; adjust imports only if required by the new exports.
  </action>
  <acceptance_criteria>
    - `wc -l crates/cintx-oracle/src/lib.rs | awk '{print $1}'` returns `>= 20`
    - `rg -n "pub use fixtures::\\{.*build_profile_representation_matrix.*build_required_profile_matrices.*PHASE4_APPROVED_PROFILES.*PHASE4_ORACLE_FAMILIES" crates/cintx-oracle/src/lib.rs`
    - `rg -n "pub use compare::\\{.*generate_profile_parity_report.*generate_phase2_parity_report.*verify_helper_surface_coverage.*tolerance_for_family.*Phase2ParityReport.*FamilyTolerance" crates/cintx-oracle/src/lib.rs`
    - `rg -n "pub mod compare;|pub mod fixtures;" crates/cintx-oracle/src/lib.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-oracle --lib exports_and_compat_raw_edge_compile -- --exact</automated>
  </verify>
  <done>`crates/cintx-oracle/src/lib.rs` meets the min-lines gate and exports the explicit profile-aware API surface needed by Phase 4 verification tooling.</done>
</task>

</tasks>

<verification>
Confirm `crates/cintx-oracle/src/lib.rs` line count is at least 20, crate-root profile-aware exports are explicit, and oracle crate library tests still pass.
</verification>

<success_criteria>
The lone 04-VERIFICATION gap is closed: `crates/cintx-oracle/src/lib.rs` is no longer flagged as a stub and continues to provide explicit profile-aware exports for Phase 4 gate consumers.
</success_criteria>

<output>
After completion, create `.planning/phases/04-verification-release-automation/05-PLAN-SUMMARY.md`
</output>
