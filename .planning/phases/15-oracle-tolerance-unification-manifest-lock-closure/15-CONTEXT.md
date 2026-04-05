# Phase 15: Oracle Tolerance Unification & Manifest Lock Closure - Context

**Gathered:** 2026-04-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Audit every family's oracle parity at the unified atol=1e-12 threshold, close the oracle_covered gap in the manifest lock (98 of 130 entries currently uncovered), regenerate the four-profile manifest lock after oracle confirmation, and make all four CI profile gates pass with mismatch_count==0. No new kernel families are added; this phase verifies and certifies what already exists.

</domain>

<decisions>
## Implementation Decisions

### oracle_covered gap closure
- **D-01:** The oracle parity comparison (`compare.rs`) marks `oracle_covered=true` on each manifest entry that passes at atol=1e-12. Coverage is objective — only entries that actually passed get the flag.
- **D-02:** `oracle_covered` persists in the committed `compiled_manifest.lock.json`. CI verifies the claim matches actual parity results. Drift between the flag and real oracle status = CI failure.
- **D-03:** Any stable entry that fails oracle parity at 1e-12 is treated as a kernel bug to be fixed, not a tolerance to be loosened (per ORAC-01). Block until fixed.

### Family tolerance map completeness
- **D-04:** Replace the explicit `tolerance_for_family()` match arms with a catch-all default returning `UNIFIED_ATOL`. The match arms become documentation, not gatekeeping. New families never cause "missing tolerance" errors.
- **D-05:** Replace `PHASE4_ORACLE_FAMILIES` hardcoded list with manifest-driven oracle eligibility — derive oracle-eligible families from the manifest lock itself (any entry with `stability: stable` or `stability: optional`). No family allow-list to maintain.

### Manifest lock regeneration ordering
- **D-06:** Keep the manifest lock as a single file (`compiled_manifest.lock.json`) with all profiles. Regeneration is atomic — one xtask command regenerates the whole lock after oracle passes.
- **D-07:** Unstable-source profile stays separate per Phase 14 D-02. Regeneration covers the four standard profiles (base, with-f12, with-4c1e, with-f12+with-4c1e); unstable-source handled by nightly CI only.
- **D-08:** Regeneration happens AFTER oracle parity is confirmed, not before (per ROADMAP SC3).

### CI gate four-profile pass
- **D-09:** Use GitHub Actions matrix strategy over the four profiles. Each profile runs as a parallel job. All must pass for the gate to succeed.
- **D-10:** The manifest-audit CI gate validates both: (1) no lock drift AND (2) every `stability: stable` entry has `oracle_covered=true`. A single uncovered stable entry fails the gate.

### Claude's Discretion
- Internal ordering of oracle audit across families/profiles
- Whether to run tolerance audit as a standalone xtask or integrate into existing parity commands
- How to structure the oracle_covered write-back into the manifest lock (post-run xtask update vs inline during comparison)
- Exact matrix job naming and artifact handling in CI workflow

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Design & requirements
- `docs/design/cintx_detailed_design.md` -- Master design document; tolerance contracts and API coverage definitions
- `.planning/REQUIREMENTS.md` -- ORAC-01 through ORAC-04 requirement definitions
- `.planning/ROADMAP.md` -- Phase 15 goal and success criteria (lines 99-108)

### Oracle comparison infrastructure
- `crates/cintx-oracle/src/compare.rs` -- UNIFIED_ATOL/RTOL constants (line 21-22), tolerance_for_family() (line 127), oracle parity comparison logic, legacy wrapper comparison
- `crates/cintx-oracle/src/fixtures.rs` -- PHASE4_ORACLE_FAMILIES (line 156), is_phase4_oracle_family(), build_profile_representation_matrix(), fixture generation and profile-scoped APIs
- `crates/cintx-oracle/tests/oracle_gate_closure.rs` -- Gate closure test entry point

### Manifest lock
- `crates/cintx-ops/generated/compiled_manifest.lock.json` -- Current lock with 130 entries, 5 profiles, oracle_covered metadata field
- `crates/cintx-ops/build.rs` -- Manifest generator that produces the lock

### CI workflows
- `.github/workflows/compat-governance-pr.yml` -- Existing oracle_parity_gate and manifest-audit jobs
- `.github/workflows/compat-governance-release.yml` -- Release workflow gates

### Prior phase context
- `.planning/phases/14-unstable-source-api-families/14-CONTEXT.md` -- D-02: unstable-source profile standalone policy; D-11: nightly CI advisory-only
- `.planning/phases/13-f12-stg-yp-kernels/13-CONTEXT.md` -- F12 family oracle parity at atol=1e-12 confirmed; OperatorEnvParams pattern
- `.planning/phases/04-verification-release-automation/04-CONTEXT.md` -- CI gate architecture, artifact paths, profile-scoped verification

### Oracle test files (all families)
- `crates/cintx-oracle/tests/one_electron_parity.rs` -- 1e family oracle tests
- `crates/cintx-oracle/tests/two_electron_parity.rs` -- 2e family oracle tests
- `crates/cintx-oracle/tests/center_2c2e_parity.rs` -- 2c2e family oracle tests
- `crates/cintx-oracle/tests/center_3c1e_parity.rs` -- 3c1e family oracle tests
- `crates/cintx-oracle/tests/center_3c2e_parity.rs` -- 3c2e family oracle tests
- `crates/cintx-oracle/tests/f12_oracle_parity.rs` -- F12/STG/YP family oracle tests
- `crates/cintx-oracle/tests/unstable_source_parity.rs` -- Unstable-source family oracle tests

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `compare.rs` already has `UNIFIED_ATOL = 1e-12` and `UNIFIED_RTOL = 1e-12` -- tolerance constants are already unified; the gap is in family coverage tracking, not tolerance values
- `tolerance_for_family()` returns the same `FamilyTolerance` for all matched families -- refactoring to catch-all is straightforward
- `build_profile_representation_matrix()` already takes `include_unstable_source` flag and handles profile-scoped fixture generation
- `is_phase4_oracle_family()` already handles `unstable::source::` prefix via `starts_with` check -- pattern to generalize
- Oracle parity reports already include tolerance metadata in JSON artifacts (`tolerance_table` with `unified_atol`)

### Established Patterns
- Profile-scoped oracle runs: `compare_profile_parity()` takes profile name and generates per-profile JSON reports
- Artifact persistence: JSON reports written to `/mnt/data` with `CINTX_ARTIFACT_DIR` fallback (Phase 4 convention)
- Legacy wrapper comparison: separate from operator parity, uses `UNIFIED_ATOL` directly (line 703)
- CI required profiles defined as `CINTX_REQUIRED_PROFILES` env variable in workflow

### Integration Points
- `compare.rs` `tolerance_for_family()` -- replace explicit match with catch-all
- `fixtures.rs` `PHASE4_ORACLE_FAMILIES` -- replace with manifest-driven derivation
- `compiled_manifest.lock.json` -- regenerate with oracle_covered flags after parity confirmed
- `compat-governance-pr.yml` -- add matrix strategy for four-profile oracle gate, add oracle_covered completeness check to manifest-audit

</code_context>

<specifics>
## Specific Ideas

No specific requirements -- open to standard approaches.

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope.

</deferred>

---

*Phase: 15-oracle-tolerance-unification-manifest-lock-closure*
*Context gathered: 2026-04-06*
