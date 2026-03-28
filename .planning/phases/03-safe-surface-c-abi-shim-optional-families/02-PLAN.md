---
phase: 03-safe-surface-c-abi-shim-optional-families
plan: 02
type: execute
wave: 2
depends_on:
  - 01
files_modified:
  - crates/cintx-ops/generated/compiled_manifest.lock.json
  - crates/cintx-ops/src/generated/api_manifest.rs
  - crates/cintx-ops/src/generated/api_manifest.csv
  - crates/cintx-ops/src/resolver.rs
  - crates/cintx-compat/src/raw.rs
  - crates/cintx-cubecl/src/executor.rs
  - crates/cintx-cubecl/src/kernels/mod.rs
  - crates/cintx-cubecl/src/kernels/center_4c1e.rs
autonomous: true
requirements:
  - OPT-01
  - OPT-02
  - OPT-03
must_haves:
  truths:
    - "With `with-f12` enabled, only the validated sph F12/STG/YP envelope resolves; cart/spinor requests fail with explicit `UnsupportedApi` reasons."
    - "With `with-4c1e` enabled, only `Validated4C1E` inputs execute; out-of-envelope inputs are rejected before backend launch with explicit `UnsupportedApi` reasons."
    - "When `unstable-source-api` is disabled, source-only symbols are not considered available and indirect requests fail explicitly."
    - "Manifest and resolver metadata remain the single source of truth for optional/unstable family support decisions."
  artifacts:
    - path: crates/cintx-ops/generated/compiled_manifest.lock.json
      provides: "Feature/profile inventory including sph-only F12/STG/YP and unstable-source entries."
      min_lines: 300
    - path: crates/cintx-ops/src/resolver.rs
      provides: "Profile-aware resolver checks and helper functions for optional/unstable gating."
      min_lines: 220
    - path: crates/cintx-compat/src/raw.rs
      provides: "Runtime envelope validators and profile-gated optional/source-only request rejection paths."
      min_lines: 900
    - path: crates/cintx-cubecl/src/executor.rs
      provides: "Validated4C1E enforcement and feature-aware 4c1e backend acceptance/rejection paths."
      min_lines: 260
  key_links:
    - from: crates/cintx-ops/generated/compiled_manifest.lock.json
      to: crates/cintx-ops/src/generated/api_manifest.rs
      via: "Generated manifest tables encode optional/unstable profiles used by resolver and compat gates."
      pattern: "with-f12|with-4c1e|unstable"
    - from: crates/cintx-ops/src/resolver.rs
      to: crates/cintx-compat/src/raw.rs
      via: "Compat resolution uses resolver metadata to enforce compile/profile availability and optional envelopes."
      pattern: "compiled_in_profiles|feature_flag|HelperKind::SourceOnly"
    - from: crates/cintx-compat/src/raw.rs
      to: crates/cintx-cubecl/src/executor.rs
      via: "Validated4C1E classifier constrains backend dispatch and preserves fail-closed behavior."
      pattern: "Validated4C1E|outside"
---

<objective>
Implement manifest-driven optional and unstable family gates with runtime envelope validation for F12/STG/YP and 4c1e.
Purpose: Satisfy Phase 3 optional-family contracts without violating fail-closed behavior or stable-surface boundaries.
Output: Updated manifest/resolver metadata plus compat/CubeCL enforcement and tests for `with-f12`, `with-4c1e`, and `unstable-source-api`.
</objective>

<execution_context>
@/home/chemtech/.codex/get-shit-done/workflows/execute-plan.md
@/home/chemtech/.codex/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/REQUIREMENTS.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/03-safe-surface-c-abi-shim-optional-families/03-CONTEXT.md
@.planning/phases/03-safe-surface-c-abi-shim-optional-families/03-RESEARCH.md
@AGENTS.md
@docs/design/cintx_detailed_design.md §3.11.1-3.11.2
@crates/cintx-ops/generated/compiled_manifest.lock.json
@crates/cintx-ops/src/resolver.rs
@crates/cintx-compat/src/raw.rs
@crates/cintx-cubecl/src/executor.rs
@crates/cintx-cubecl/src/kernels/mod.rs
<interfaces>
From `crates/cintx-ops/src/resolver.rs`:
```rust
pub struct ManifestEntry {
    pub feature_flag: FeatureFlag,
    pub compiled_in_profiles: &'static [&'static str],
    pub helper_kind: HelperKind,
    pub canonical_family: &'static str,
}

pub fn descriptor_by_symbol(symbol: &str) -> Result<&'static OperatorDescriptor, ResolverError>;
```

From `crates/cintx-compat/src/raw.rs`:
```rust
pub unsafe fn query_workspace_raw(...) -> Result<WorkspaceQuery, cintxRsError>;
pub unsafe fn eval_raw(...) -> Result<RawEvalSummary, cintxRsError>;
```

From `docs/design/cintx_detailed_design.md` §3.11.2:
```text
Validated4C1E = cart/sph representation, scalar int4c1e, dims natural, max(l)<=4, CubeCL path.
Outside envelope => UnsupportedApi("outside Validated4C1E").
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Expand manifest/resolver metadata for optional and unstable-source families</name>
  <files>crates/cintx-ops/generated/compiled_manifest.lock.json, crates/cintx-ops/src/generated/api_manifest.rs, crates/cintx-ops/src/generated/api_manifest.csv, crates/cintx-ops/src/resolver.rs</files>
  <read_first>crates/cintx-ops/generated/compiled_manifest.lock.json, crates/cintx-ops/src/generated/api_manifest.rs, crates/cintx-ops/src/generated/api_manifest.csv, crates/cintx-ops/src/resolver.rs, crates/cintx-ops/build.rs, docs/design/cintx_detailed_design.md §3.11.1, §10.1, .planning/phases/03-safe-surface-c-abi-shim-optional-families/03-CONTEXT.md</read_first>
  <action>
Update the canonical manifest lock and generated tables so optional and unstable families are represented explicitly and auditable. Add the sph-only F12/STG/YP operator set with `feature_flag = with-f12`, `stability = optional`, and profile membership only in `with-f12` and `with-f12+with-4c1e` (per D-09 and D-10). Ensure cart/spinor variants for those families are absent from compiled symbol inventory and enforce this in tests. Add source-only entries using `feature_flag = unstable_source` and `helper_kind = source` under explicit unstable namespaces (per D-13). Extend resolver helpers to expose profile-aware checks (`compiled_in_profiles`) and source-only identification so downstream callers can reject unavailable requests deterministically (per D-12 and D-16). Include documentation comments/tests stating promotion from unstable to stable remains evidence-gated by manifest/oracle/release checks and explicit maintainer decision (per D-15).
  </action>
  <acceptance_criteria>
    - `rg -n "with-f12|unstable_source|source" crates/cintx-ops/generated/compiled_manifest.lock.json`
    - `rg -n "int2e_(stg|yp).*_sph" crates/cintx-ops/src/generated/api_manifest.csv`
    - `! rg -n "int2e_(stg|yp).*(cart|spinor)" crates/cintx-ops/src/generated/api_manifest.csv`
    - `rg -n "compiled_in_profiles|feature_flag|SourceOnly" crates/cintx-ops/src/resolver.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-ops --lib</automated>
  </verify>
  <done>Manifest and resolver metadata fully encode optional/unstable family scope and profile availability, including sph-only F12/STG/YP and explicit source-only entries.</done>
</task>

<task type="auto">
  <name>Task 2: Enforce optional-family and unstable-source envelopes in compat and CubeCL execution</name>
  <files>crates/cintx-compat/src/raw.rs, crates/cintx-cubecl/src/executor.rs, crates/cintx-cubecl/src/kernels/mod.rs, crates/cintx-cubecl/src/kernels/center_4c1e.rs</files>
  <read_first>crates/cintx-compat/src/raw.rs, crates/cintx-cubecl/src/executor.rs, crates/cintx-cubecl/src/kernels/mod.rs, crates/cintx-cubecl/src/kernels/center_4c1e.rs, crates/cintx-ops/src/resolver.rs, docs/design/cintx_detailed_design.md §3.11.1-3.11.2, .planning/phases/03-safe-surface-c-abi-shim-optional-families/03-RESEARCH.md, docs/rust_crate_test_guideline.md</read_first>
  <action>
Implement dual-gating and runtime envelope checks in the raw/execute path. In `raw.rs`, derive active feature profile and reject any descriptor whose `compiled_in_profiles` does not include the active profile (per D-09 and D-12). Add explicit `validate_f12_envelope` logic requiring sph representation and natural dims; reject out-of-envelope requests with `UnsupportedApi` reason text that names the with-f12 sph envelope (per D-10). Add explicit `validate_4c1e_envelope` logic requiring `with-4c1e`, cart/sph representation, scalar component rank, natural dims, `max(l)<=4`, and CPU CubeCL execution; reject all others with `UnsupportedApi` reason `"outside Validated4C1E"` (per D-11). In CubeCL, wire `center_4c1e` launch only for validated inputs and keep fail-closed no-partial-write ownership guarantees unchanged from Phase 2 (per D-18). Ensure source-only requests fail explicitly when `unstable-source-api` is disabled (per D-16), while enabling them behind the unstable feature gate (per D-13).
  </action>
  <acceptance_criteria>
    - `rg -n "validate_f12|with-f12|sph envelope" crates/cintx-compat/src/raw.rs`
    - `rg -n "Validated4C1E|outside Validated4C1E|max\\(l\\)|with-4c1e" crates/cintx-compat/src/raw.rs crates/cintx-cubecl/src/executor.rs`
    - `rg -n "cfg\\(feature = \"with-4c1e\"\\)|center_4c1e" crates/cintx-cubecl/src/kernels/mod.rs crates/cintx-cubecl/src/kernels/center_4c1e.rs`
    - `rg -n "unstable-source-api|SourceOnly|UnsupportedApi" crates/cintx-compat/src/raw.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-compat --lib && cargo test -p cintx-compat --lib --features with-f12 && cargo test -p cintx-compat --lib --features with-4c1e && cargo test -p cintx-cubecl --lib --features with-4c1e</automated>
  </verify>
  <done>Optional families and unstable-source symbols are now controlled by compile-time features plus runtime envelope checks, with explicit `UnsupportedApi` failures outside validated scope.</done>
</task>

</tasks>

<verification>
Run `cintx-ops` tests for manifest/resolver consistency, then run compat/CubeCL feature-matrix tests to confirm envelope checks and explicit rejection paths.
</verification>

<success_criteria>
The system enforces `with-f12` sph-only behavior, strict `Validated4C1E` boundaries for `with-4c1e`, and explicit unstable-source gating with deterministic `UnsupportedApi` failures outside enabled envelopes.
</success_criteria>

<output>
After completion, create `.planning/phases/03-safe-surface-c-abi-shim-optional-families/02-PLAN-SUMMARY.md`
</output>
