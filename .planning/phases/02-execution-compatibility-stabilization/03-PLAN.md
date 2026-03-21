---
phase: 02-execution-compatibility-stabilization
plan: 03
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/cintx-core/src/error.rs
  - crates/cintx-core/src/lib.rs
  - crates/cintx-ops/build.rs
  - crates/cintx-ops/generated/compiled_manifest.lock.json
  - crates/cintx-ops/src/generated/api_manifest.rs
  - crates/cintx-ops/src/generated/api_manifest.csv
  - crates/cintx-ops/src/resolver.rs
autonomous: true
requirements:
  - COMP-03
  - COMP-05
must_haves:
  truths:
    - "The canonical manifest now includes the helper, transform, optimizer-lifecycle, and legacy-wrapper APIs that Phase 2 claims to support, so coverage is still mechanically auditable from one source of truth."
    - "Raw compat validation failures use typed `cintxRsError` variants for layout, env-offset, and buffer-size faults instead of generic planner strings or silent truncation."
    - "Base-only Phase 2 scope stays explicit: helper and legacy coverage is added for the upstream base surface, while `4c1e`, F12/STG/YP, and GTG remain outside this phase."
  artifacts:
    - path: crates/cintx-ops/generated/compiled_manifest.lock.json
      provides: "Canonical manifest entries for operator plus helper/transform/optimizer/legacy symbols."
      min_lines: 150
    - path: crates/cintx-ops/src/resolver.rs
      provides: "Metadata-aware lookup helpers that can distinguish operator, helper, transform, optimizer, and legacy entries."
      min_lines: 120
    - path: crates/cintx-core/src/error.rs
      provides: "Typed raw-compat failure variants for layout, env-offset, and output-size contract violations."
      min_lines: 60
  key_links:
    - from: crates/cintx-ops/build.rs
      to: crates/cintx-ops/generated/compiled_manifest.lock.json
      via: "Generates manifest tables from the canonical lock, including helper and legacy metadata."
      pattern: "helper_kind"
    - from: crates/cintx-ops/src/resolver.rs
      to: crates/cintx-core/src/error.rs
      via: "Downstream compat/runtime callers rely on manifest classification plus typed errors to reject unsupported or malformed calls."
      pattern: "UnsupportedApi"
---

<objective>
Make the manifest and public error surface honest about the Phase 2 compatibility scope before backend and raw-call implementation begins.
Purpose: Land the helper/legacy source-of-truth coverage and the missing typed failure variants that all later compat/runtime code will depend on.
Output: Expanded canonical manifest metadata and a completed `cintxRsError` raw-validation contract.
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
@.planning/phases/01-manifest-planner-foundation/01-SUMMARY.md
@.planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md
@AGENTS.md
@docs/design/cintx_detailed_design.md
@libcint-master/include/cint.h.in
@libcint-master/src/misc.h
@crates/cintx-ops/build.rs
@crates/cintx-ops/src/resolver.rs
@crates/cintx-core/src/error.rs
</context>

<tasks>

<task type="auto">
  <name>Task 1: Extend the canonical manifest to cover the helper and legacy Phase 2 surface</name>
  <files>crates/cintx-ops/generated/compiled_manifest.lock.json, crates/cintx-ops/build.rs, crates/cintx-ops/src/generated/api_manifest.rs, crates/cintx-ops/src/generated/api_manifest.csv, crates/cintx-ops/src/resolver.rs</files>
  <read_first>crates/cintx-ops/generated/compiled_manifest.lock.json, crates/cintx-ops/build.rs, crates/cintx-ops/src/resolver.rs, .planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md, docs/design/cintx_detailed_design.md §3.3 and Appendix A, libcint-master/include/cint.h.in:227-290, libcint-master/src/misc.h:34-76</read_first>
  <action>
Extend the canonical manifest and code generation pipeline so the lock includes the upstream base-scope helper, transform, optimizer-lifecycle, and legacy-wrapper symbols listed in `include/cint.h.in:227-290` and `src/misc.h:34-76`. Add entries for the helper counts/offsets/norms, the Cartesian-to-spherical and Cartesian-to-spinor transform helpers, the optimizer lifecycle symbols (`CINTinit_2e_optimizer`, `CINTinit_optimizer`, `CINTdel_2e_optimizer`, `CINTdel_optimizer`), and the legacy `cint2e_cart`, `cint2e_sph`, `cint2e`, and matching optimizer wrappers. In the lock and generated tables, set `category` to `helper` or `legacy`, set `helper_kind` to one of `Helper`, `Transform`, `Optimizer`, or `Legacy`, and normalize `canonical_family` so wrapper/optimizer rows still map back to the underlying base family. Keep the additions Phase-2-scoped: do not add `4c1e`, F12/STG/YP, or GTG helper/wrapper rows here. Update `Resolver` so lookup helpers can filter by `helper_kind` and resolve helper/legacy symbols without hard-coded string tables outside the generated manifest.
  </action>
  <acceptance_criteria>
    - `rg -n "HelperKind::Helper|HelperKind::Transform|HelperKind::Optimizer|HelperKind::Legacy" crates/cintx-ops/src/generated/api_manifest.rs`
    - `rg -n "CINTinit_optimizer|CINTdel_optimizer|cint2e_cart|CINTc2s_bra_sph" crates/cintx-ops/src/generated/api_manifest.csv`
    - `rg -n "helper_kind" crates/cintx-ops/build.rs`
    - `rg -n "filter.*helper_kind|helpers_by_kind|entries_by_kind" crates/cintx-ops/src/resolver.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-ops --lib</automated>
  </verify>
  <done>The canonical manifest and generated resolver metadata now enumerate the helper/transform/optimizer/legacy APIs that COMP-03 depends on, without broadening Phase 2 beyond the locked base-family scope.</done>
</task>

<task type="auto">
  <name>Task 2: Add the missing typed raw-validation and buffer-contract errors</name>
  <files>crates/cintx-core/src/error.rs, crates/cintx-core/src/lib.rs</files>
  <read_first>crates/cintx-core/src/error.rs, crates/cintx-core/src/lib.rs, .planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md, docs/design/cintx_detailed_design.md §7.3-7.5, AGENTS.md</read_first>
  <action>
Expand `cintxRsError` so raw compat callers can receive typed layout and output-contract failures instead of planner-detail strings. Add exactly these public variants: `InvalidAtmLayout { slot_width: usize, provided: usize }`, `InvalidBasLayout { slot_width: usize, provided: usize }`, `InvalidEnvOffset { slot: &'static str, offset: usize, env_len: usize }`, and `BufferTooSmall { required: usize, provided: usize }`. Keep the existing `InvalidDims`, `UnsupportedApi`, `MemoryLimitExceeded`, `HostAllocationFailed`, and `DeviceOutOfMemory` variants intact. Update `lib.rs` re-exports if needed so downstream crates can use the expanded enum directly, and add focused unit tests in `error.rs` that assert the new variants format and match predictably.
  </action>
  <acceptance_criteria>
    - `rg -n "InvalidAtmLayout" crates/cintx-core/src/error.rs`
    - `rg -n "InvalidBasLayout" crates/cintx-core/src/error.rs`
    - `rg -n "InvalidEnvOffset" crates/cintx-core/src/error.rs`
    - `rg -n "BufferTooSmall" crates/cintx-core/src/error.rs`
    - `rg -n "pub use error::\\{CoreError, cintxRsError\\}" crates/cintx-core/src/lib.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-core --lib</automated>
  </verify>
  <done>The public library error contract now has typed raw-layout, env-offset, and output-size failures, satisfying the no-silent-truncation and explicit-validation part of COMP-05.</done>
</task>

</tasks>

<verification>
Regenerate the manifest tables from the canonical lock, then run the `cintx-ops` and `cintx-core` library test suites to confirm helper metadata and typed errors both compile and match tests.
</verification>

<success_criteria>
Helper/transform/optimizer/legacy symbols are present in the canonical manifest and generated tables, and the public error enum exposes explicit raw-layout and buffer-size failures for later compat work.
</success_criteria>

<output>
After completion, create `.planning/phases/02-execution-compatibility-stabilization/03-PLAN-SUMMARY.md`
</output>
