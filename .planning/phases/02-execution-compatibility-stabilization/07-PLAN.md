---
phase: 02-execution-compatibility-stabilization
plan: 07
type: execute
wave: 5
depends_on:
  - 03
  - 05
  - 06
files_modified:
  - crates/cintx-compat/src/helpers.rs
  - crates/cintx-compat/src/transform.rs
  - crates/cintx-compat/src/optimizer.rs
  - crates/cintx-compat/src/legacy.rs
  - crates/cintx-compat/src/lib.rs
  - crates/cintx-oracle/build.rs
  - crates/cintx-oracle/src/lib.rs
  - crates/cintx-oracle/src/fixtures.rs
  - crates/cintx-oracle/src/compare.rs
autonomous: true
requirements:
  - COMP-03
  - EXEC-05
must_haves:
  truths:
    - "Compat callers can use the helper, transform, optimizer-lifecycle, and legacy-wrapper APIs that Phase 2 claims to preserve from upstream libcint."
    - "Optimized and non-optimized execution paths share the same writer/layout contract and produce numerically equivalent results within the accepted tolerance envelope."
    - "The oracle harness can compare the Phase 2 base-family compat surface against vendored upstream libcint, including helper parity and optimizer-on/off equivalence."
  artifacts:
    - path: crates/cintx-compat/src/helpers.rs
      provides: "The upstream helper/count/offset/norm APIs for the Phase 2 base scope."
      min_lines: 120
    - path: crates/cintx-compat/src/optimizer.rs
      provides: "Immutable optimizer handle lifecycle and raw compat optimizer entry points."
      min_lines: 80
    - path: crates/cintx-compat/src/legacy.rs
      provides: "Thin `cint2e_*` wrapper forwards into the shared raw compat pipeline."
      min_lines: 80
    - path: crates/cintx-oracle/src/compare.rs
      provides: "Oracle comparison routines for helper parity and optimizer equivalence."
      min_lines: 120
  key_links:
    - from: crates/cintx-compat/src/legacy.rs
      to: crates/cintx-compat/src/raw.rs
      via: "Legacy wrappers forward into the same raw path instead of duplicating dims/output math."
      pattern: "eval_raw|query_workspace_raw"
    - from: crates/cintx-compat/src/optimizer.rs
      to: crates/cintx-oracle/src/compare.rs
      via: "Optimizer handles are verified by on/off parity checks against oracle comparisons."
      pattern: "optimizer"
    - from: crates/cintx-oracle/src/fixtures.rs
      to: crates/cintx-ops/generated/compiled_manifest.lock.json
      via: "Oracle fixtures derive the Phase 2 comparison set from the canonical manifest."
      pattern: "compiled_manifest"
---

<objective>
Finish the Phase 2 compatibility surface by adding helpers, transforms, optimizer/legacy wrappers, and oracle parity coverage.
Purpose: Close COMP-03 and EXEC-05 with concrete APIs and verification so the phase claims are backed by helper coverage and optimizer-equivalence tests, not just execution plumbing.
Output: Helper/transform/optimizer/legacy compat APIs plus a base-family oracle harness for parity and optimizer-on/off checks.
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
@.planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md
@AGENTS.md
@docs/design/cintx_detailed_design.md
@libcint-master/include/cint.h.in:227-290
@libcint-master/src/misc.h:24-76
@crates/cintx-compat/src/raw.rs
@crates/cintx-cubecl/src/transform/mod.rs
@crates/cintx-ops/generated/compiled_manifest.lock.json
</context>

<tasks>

<task type="auto">
  <name>Task 1: Implement the Phase 2 helper and transform compat APIs</name>
  <files>crates/cintx-compat/src/helpers.rs, crates/cintx-compat/src/transform.rs, crates/cintx-compat/src/lib.rs</files>
  <read_first>crates/cintx-compat/src/helpers.rs, crates/cintx-compat/src/transform.rs, crates/cintx-compat/src/lib.rs, libcint-master/include/cint.h.in:227-250, libcint-master/include/cint.h.in:283-290, docs/design/cintx_detailed_design.md Appendix A, .planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md</read_first>
  <action>
Implement the exact upstream helper and transform APIs that are in Phase 2 scope. In `helpers.rs`, add `CINTlen_cart`, `CINTlen_spinor`, `CINTcgtos_cart`, `CINTcgtos_spheric`, `CINTcgtos_spinor`, `CINTcgto_cart`, `CINTcgto_spheric`, `CINTcgto_spinor`, `CINTtot_pgto_spheric`, `CINTtot_pgto_spinor`, `CINTtot_cgto_cart`, `CINTtot_cgto_spheric`, `CINTtot_cgto_spinor`, `CINTshells_cart_offset`, `CINTshells_spheric_offset`, `CINTshells_spinor_offset`, and `CINTgto_norm`. In `transform.rs`, add the helper transform entry points `CINTc2s_bra_sph`, `CINTc2s_ket_sph`, `CINTc2s_ket_sph1`, `CINTc2s_ket_spinor_sf1`, `CINTc2s_iket_spinor_sf1`, `CINTc2s_ket_spinor_si1`, and `CINTc2s_iket_spinor_si1`. Use shared basis metadata and the CubeCL transform helpers where applicable, and keep every helper visible through `lib.rs`. Do not introduce manual wrapper lists outside the canonical manifest.
  </action>
  <acceptance_criteria>
    - `rg -n "CINTlen_cart|CINTlen_spinor|CINTcgtos_cart|CINTgto_norm" crates/cintx-compat/src/helpers.rs`
    - `rg -n "CINTc2s_bra_sph|CINTc2s_ket_sph1|CINTc2s_ket_spinor_sf1" crates/cintx-compat/src/transform.rs`
    - `rg -n "pub mod helpers;|pub mod transform;" crates/cintx-compat/src/lib.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-compat --lib</automated>
  </verify>
  <done>The helper and transform compat surface now matches the documented Phase 2 upstream subset, satisfying the helper half of COMP-03.</done>
</task>

<task type="auto">
  <name>Task 2: Implement immutable optimizer handles and thin legacy wrapper forwards</name>
  <files>crates/cintx-compat/src/optimizer.rs, crates/cintx-compat/src/legacy.rs, crates/cintx-compat/src/lib.rs</files>
  <read_first>crates/cintx-compat/src/optimizer.rs, crates/cintx-compat/src/legacy.rs, crates/cintx-compat/src/raw.rs, libcint-master/include/cint.h.in:256-278, libcint-master/src/misc.h:34-76, docs/design/cintx_detailed_design.md §7.8, .planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md</read_first>
  <action>
In `optimizer.rs`, implement an immutable `RawOptimizerHandle` backed by cached planner/backend metadata and expose the upstream lifecycle entry points `CINTinit_2e_optimizer`, `CINTinit_optimizer`, `CINTdel_2e_optimizer`, and `CINTdel_optimizer`. In `legacy.rs`, implement the thin wrapper forwards `cint2e_cart`, `cint2e_cart_optimizer`, `cint2e_sph`, `cint2e_sph_optimizer`, `cint2e`, and `cint2e_optimizer` so they call the shared raw compat pipeline instead of duplicating dims math, layout writers, or backend dispatch. Keep the optimized and non-optimized paths on the same output writer and `RawEvalSummary` contract, and export the new APIs from `lib.rs`.
  </action>
  <acceptance_criteria>
    - `rg -n "struct RawOptimizerHandle" crates/cintx-compat/src/optimizer.rs`
    - `rg -n "CINTinit_2e_optimizer|CINTdel_optimizer" crates/cintx-compat/src/optimizer.rs`
    - `rg -n "cint2e_cart|cint2e_sph|cint2e_optimizer" crates/cintx-compat/src/legacy.rs`
    - `rg -n "eval_raw|query_workspace_raw" crates/cintx-compat/src/legacy.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-compat --lib</automated>
  </verify>
  <done>The optimizer lifecycle and legacy wrappers now exist as thin forwards over the shared raw path, completing the wrapper half of COMP-03 while preserving the shared output contract needed for EXEC-05.</done>
</task>

<task type="auto">
  <name>Task 3: Build the base-family oracle harness for helper parity and optimizer equivalence</name>
  <files>crates/cintx-oracle/build.rs, crates/cintx-oracle/src/lib.rs, crates/cintx-oracle/src/fixtures.rs, crates/cintx-oracle/src/compare.rs</files>
  <read_first>crates/cintx-oracle/build.rs, crates/cintx-oracle/src/lib.rs, crates/cintx-oracle/src/fixtures.rs, crates/cintx-oracle/src/compare.rs, crates/cintx-ops/generated/compiled_manifest.lock.json, libcint-master/include/cint_funcs.h, docs/design/cintx_detailed_design.md §13.4 and §14.1, .planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md</read_first>
  <action>
Implement the oracle harness for the Phase 2 base families. In `build.rs`, wire the vendored upstream libcint build and bindgen/header setup needed for oracle comparison. In `fixtures.rs`, derive the comparison target set from the canonical manifest, but keep it limited to the Phase 2 base families and helper/legacy scope. In `compare.rs`, implement comparison routines for helper parity (counts, offsets, norms, transforms) and optimizer-on/off equivalence for `1e`, `2e`, `2c2e`, `3c1e`, and `3c2e`. Add tests that fail when helper coverage drifts from the manifest or when optimized vs non-optimized outputs exceed the accepted tolerance envelope. Export the harness through `lib.rs`.
  </action>
  <acceptance_criteria>
    - `rg -n "compiled_manifest|manifest" crates/cintx-oracle/src/fixtures.rs`
    - `rg -n "optimizer|tolerance|helper" crates/cintx-oracle/src/compare.rs`
    - `rg -n "cc::Build|bindgen" crates/cintx-oracle/build.rs`
    - `rg -n "pub mod compare;|pub mod fixtures;" crates/cintx-oracle/src/lib.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-oracle --lib && cargo test -p cintx-compat --lib</automated>
  </verify>
  <done>The Phase 2 compatibility claims are now backed by an oracle harness that checks helper parity and optimizer-on/off equivalence against vendored upstream libcint, satisfying EXEC-05.</done>
</task>

</tasks>

<verification>
Run the compat and oracle library tests together; the result should prove that the helper/wrapper surface exists and that optimizer parity is checked against vendored upstream behavior.
</verification>

<success_criteria>
All Phase 2 helper, transform, optimizer, and legacy compat APIs exist, and oracle-backed tests verify helper parity plus optimized/non-optimized result equivalence for the base family set.
</success_criteria>

<output>
After completion, create `.planning/phases/02-execution-compatibility-stabilization/07-PLAN-SUMMARY.md`
</output>
