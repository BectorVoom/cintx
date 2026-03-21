---
phase: 02-execution-compatibility-stabilization
plan: 07
type: execute
wave: 6
depends_on:
  - 03
  - 06
  - 08
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
    - "Optimized and non-optimized execution paths share the same compat writer/layout contract introduced earlier and produce numerically equivalent caller-visible cart/spheric/spinor flat buffers within the accepted family-specific tolerance envelope, including interleaved spinor doubles."
    - "The oracle harness reaches the Phase 2 compat APIs directly and compares evaluated outputs against vendored upstream libcint for `1e`, `2e`, `2c2e`, `3c1e`, and `3c2e` across a manifest-derived cart/spheric/spinor fixture matrix, while emitting auditable `/mnt/data` parity artifacts for helper coverage, layout semantics, and optimizer-on/off equivalence."
  artifacts:
    - path: crates/cintx-compat/src/helpers.rs
      provides: "The upstream helper/count/offset/norm APIs for the Phase 2 base scope."
      min_lines: 120
    - path: crates/cintx-compat/src/optimizer.rs
      provides: "Immutable optimizer handle lifecycle and raw compat optimizer entry points."
      min_lines: 80
    - path: crates/cintx-compat/src/legacy.rs
      provides: "Thin misc.h-derived `cNAME*` wrapper forwards for the in-scope base families into the shared raw compat pipeline."
      min_lines: 80
    - path: crates/cintx-oracle/src/compare.rs
      provides: "Oracle comparison routines for helper parity, family-output vs upstream comparisons, and optimizer equivalence."
      min_lines: 120
    - path: /mnt/data/cintx_phase_02_manifest_representation_matrix.json
      provides: "Pretty-printed manifest-derived cart/spheric/spinor fixture coverage with final compat flat-buffer layout metadata for the Phase 2 family set."
      min_lines: 20
    - path: /mnt/data/cintx_phase_02_compat_parity_report.json
      provides: "Pretty-printed oracle parity results covering helper parity, evaluated-output parity, optimizer equivalence, and spinor interleaving/layout assertions."
      min_lines: 40
  key_links:
    - from: crates/cintx-compat/src/legacy.rs
      to: crates/cintx-compat/src/raw.rs
      via: "Legacy wrappers forward into the same raw path instead of duplicating dims/output math."
      pattern: "eval_raw|query_workspace_raw"
    - from: crates/cintx-oracle/src/compare.rs
      to: crates/cintx-compat/src/lib.rs
      via: "Oracle comparisons import helper, raw, optimizer, and legacy compat APIs instead of calling runtime or CubeCL directly, then compare their evaluated outputs against vendored upstream libcint."
      pattern: "cintx_compat"
    - from: crates/cintx-oracle/src/fixtures.rs
      to: crates/cintx-ops/generated/compiled_manifest.lock.json
      via: "Oracle fixtures derive the Phase 2 comparison set from the canonical manifest."
      pattern: "compiled_manifest"
    - from: crates/cintx-oracle/src/compare.rs
      to: crates/cintx-compat/src/layout.rs
      via: "Oracle comparisons assert final caller-visible compat flat-buffer semantics, including cart/spheric axis ordering and interleaved spinor doubles, before diffing against upstream."
      pattern: "layout|flat|interleav"
---

<objective>
Finish the Phase 2 compatibility surface by adding helpers, transforms, optimizer/legacy wrappers, and oracle parity coverage.
Purpose: Close COMP-03 and EXEC-05 with concrete APIs and verification so the phase claims are backed by helper coverage and optimizer-equivalence tests, not just execution plumbing.
Output: Helper/transform/optimizer/legacy compat APIs plus a base-family oracle harness, a manifest-derived cart/spheric/spinor fixture matrix, and `/mnt/data` parity artifacts for evaluated-output and optimizer-on/off checks.
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
@crates/cintx-compat/src/optimizer.rs
@crates/cintx-compat/src/layout.rs
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
  <name>Task 2: Implement immutable optimizer handles and the full misc.h-derived legacy wrapper forwards</name>
  <files>crates/cintx-compat/src/optimizer.rs, crates/cintx-compat/src/legacy.rs, crates/cintx-compat/src/lib.rs</files>
  <read_first>crates/cintx-compat/src/optimizer.rs, crates/cintx-compat/src/legacy.rs, crates/cintx-compat/src/raw.rs, crates/cintx-ops/generated/compiled_manifest.lock.json, libcint-master/include/cint.h.in:256-278, libcint-master/src/misc.h:34-76, docs/design/cintx_detailed_design.md §3.4.1, §7.8, and Appendix C, .planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md</read_first>
  <action>
Extend the `RawOptimizerHandle` contract introduced in Plan 06 into an immutable handle backed by cached planner/backend metadata and expose the upstream lifecycle entry points `CINTinit_2e_optimizer`, `CINTinit_optimizer`, `CINTdel_2e_optimizer`, and `CINTdel_optimizer`. In `legacy.rs`, implement or generate the full Phase 2 `cNAME*` wrapper surface from manifest metadata or a small wrapper macro layer that mirrors `src/misc.h`; do not hand-author a one-off list and do not stop at `cint2e_*`. Follow the upstream split exactly: `ALL_CINT1E` families contribute `cNAME_cart`, `cNAME_sph`, and `cNAME`, while `ALL_CINT` families contribute those three plus `cNAME_cart_optimizer`, `cNAME_sph_optimizer`, and `cNAME_optimizer`. That means the in-scope base-family surface must cover the `1e`, `2e`, `2c2e`, `3c1e`, and `3c2e` manifest families that expand through those macros, with optimizer variants only where the macro defines them. Make every wrapper a thin forward into the shared raw compat pipeline instead of duplicating dims math, layout writers, or backend dispatch. Keep the optimized and non-optimized paths on the same compat-owned output writer and `RawEvalSummary` contract, export the new APIs from `lib.rs`, and add a regression test in `legacy.rs` that derives the expected wrapper set from the base-family manifest plus the `misc.h` macro rules and fails on missing or extra wrappers.
  </action>
  <acceptance_criteria>
    - `rg -n "struct RawOptimizerHandle" crates/cintx-compat/src/optimizer.rs`
    - `rg -n "CINTinit_2e_optimizer|CINTdel_optimizer" crates/cintx-compat/src/optimizer.rs`
    - `rg -n "cint1e_ovlp_cart|cint1e_nuc|cint2e_cart_optimizer|cint2c2e_optimizer|cint3c1e_sph_optimizer|cint3c2e_optimizer" crates/cintx-compat/src/legacy.rs`
    - `rg -n "eval_raw|query_workspace_raw" crates/cintx-compat/src/legacy.rs`
    - `cargo test -p cintx-compat --lib legacy_wrapper_surface_matches_misc -- --exact`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-compat --lib legacy_wrapper_surface_matches_misc -- --exact</automated>
  </verify>
  <done>The optimizer lifecycle and misc.h-derived legacy wrapper surface now exist as thin forwards over the shared raw path, and the wrapper-coverage regression test proves the in-scope base-family `cNAME*` set is complete while preserving the shared output contract needed for EXEC-05.</done>
</task>

<task type="auto">
  <name>Task 3: Build the base-family oracle harness for evaluated-output comparison, helper parity, and optimizer equivalence</name>
  <files>crates/cintx-oracle/build.rs, crates/cintx-oracle/src/lib.rs, crates/cintx-oracle/src/fixtures.rs, crates/cintx-oracle/src/compare.rs</files>
  <read_first>crates/cintx-oracle/build.rs, crates/cintx-oracle/src/lib.rs, crates/cintx-oracle/src/fixtures.rs, crates/cintx-oracle/src/compare.rs, crates/cintx-ops/generated/compiled_manifest.lock.json, libcint-master/include/cint_funcs.h, docs/design/cintx_detailed_design.md §13.4 and §14.1, .planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md</read_first>
  <action>
Implement the oracle harness for the Phase 2 base families. In `build.rs`, wire the vendored upstream libcint build and bindgen/header setup needed for oracle comparison. In `fixtures.rs`, derive the comparison target set from the canonical manifest, keep it limited to the Phase 2 base families and helper/legacy scope, and materialize a pretty-printed `/mnt/data/cintx_phase_02_manifest_representation_matrix.json` artifact that records every supported cart/spheric/spinor family-representation fixture chosen from the manifest plus the final compat flat-buffer metadata each fixture expects (arity-owned `dims`, component count, and whether spinor output is interleaved doubles). In `compare.rs`, implement comparison routines by importing and calling the public `cintx-compat` helper, raw, optimizer, and legacy APIs; do not call `cintx-runtime` or `cintx-cubecl` directly from oracle code. The oracle must explicitly compare evaluated compat outputs against vendored upstream libcint for `1e`, `2e`, `2c2e`, `3c1e`, and `3c2e` using identical `atm`/`bas`/`env`/`shls`/`dims` fixtures and the family-specific tolerance criteria from `docs/design/cintx_detailed_design.md` §13.8 plus §13.9: basic `1e` uses `atol=1e-11`, `rtol=1e-9`; plain/low-derivative `2e` uses `atol=1e-12`, `rtol=1e-10`; `2c2e` and `3c2e` use `atol=1e-9`, `rtol=1e-7`; high-order `3c1e` uses `atol=1e-7`, `rtol=1e-5`; and when `abs(ref) < 1e-18`, compare by absolute error only. Make the compare assertions explicit about final compat flat-buffer layout semantics: cart and spheric outputs must be diffed in the caller-visible ordering produced by `cintx-compat::layout`, and spinor outputs must be checked as interleaved doubles in that same final flat-buffer contract rather than as an internal complex staging type. Cover helper parity (counts, offsets, norms, transforms) and optimizer-on/off equivalence for the same family set as additional gates. Emit a pretty-printed `/mnt/data/cintx_phase_02_compat_parity_report.json` artifact capturing manifest coverage, tolerance parameters, helper parity verdicts, final flat-buffer layout assertions, and optimizer equivalence results. Add tests that fail when helper coverage drifts from the manifest, when the cart/spheric/spinor representation matrix no longer matches manifest fixtures, when evaluated-output diffs vs upstream exceed tolerance, or when optimized vs non-optimized outputs diverge beyond the same family tolerance envelope. Export the harness through `lib.rs`.
  </action>
  <acceptance_criteria>
    - `rg -n "compiled_manifest|manifest" crates/cintx-oracle/src/fixtures.rs`
    - `rg -n "cintx_compat" crates/cintx-oracle/src/compare.rs`
    - `rg -n "1e-11|1e-12|1e-9|1e-7|1e-5|1e-18" crates/cintx-oracle/src/compare.rs`
    - `rg -n "1e|2e|2c2e|3c1e|3c2e" crates/cintx-oracle/src/compare.rs crates/cintx-oracle/src/fixtures.rs`
    - `rg -n "optimizer|tolerance|helper|upstream|atol|rtol|zero_threshold" crates/cintx-oracle/src/compare.rs`
    - `rg -n "/mnt/data/cintx_phase_02_manifest_representation_matrix\\.json|/mnt/data/cintx_phase_02_compat_parity_report\\.json" crates/cintx-oracle/src/fixtures.rs crates/cintx-oracle/src/compare.rs`
    - `rg -n "cart|spheric|spinor|representation_matrix|flat-buffer|interleav" crates/cintx-oracle/src/fixtures.rs crates/cintx-oracle/src/compare.rs`
    - `rg -n "cc::Build|bindgen" crates/cintx-oracle/build.rs`
    - `rg -n "pub mod compare;|pub mod fixtures;" crates/cintx-oracle/src/lib.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-oracle --lib && cargo test -p cintx-compat --lib && test -f /mnt/data/cintx_phase_02_manifest_representation_matrix.json && test -f /mnt/data/cintx_phase_02_compat_parity_report.json</automated>
  </verify>
  <done>The Phase 2 compatibility claims are now backed by an oracle harness that compares evaluated outputs against vendored upstream libcint for `1e`/`2e`/`2c2e`/`3c1e`/`3c2e`, explicitly checks final compat cart/spheric/spinor flat-buffer semantics including spinor interleaving, emits the required `/mnt/data` coverage/parity artifacts, and also checks helper parity plus optimizer-on/off equivalence, satisfying EXEC-05.</done>
</task>

</tasks>

<verification>
Run the compat misc.h-derived wrapper coverage test and the compat/oracle library tests together; the result should prove that the helper/wrapper surface exists, wrapper coverage matches the base-family macro expansion from `src/misc.h`, evaluated outputs plus optimizer parity are checked against vendored upstream behavior under the explicit family tolerance table, and the `/mnt/data/cintx_phase_02_manifest_representation_matrix.json` plus `/mnt/data/cintx_phase_02_compat_parity_report.json` artifacts are produced.
</verification>

<success_criteria>
All Phase 2 helper, transform, optimizer, and legacy compat APIs exist, the misc.h-derived wrapper-coverage test proves the full in-scope base-family `cNAME*` surface is present, oracle-backed tests verify helper parity, manifest-derived cart/spheric/spinor representation coverage, evaluated-output-vs-upstream parity, and optimized/non-optimized result equivalence for the base family set with explicit tolerance criteria, and the required `/mnt/data` oracle artifacts exist.
</success_criteria>

<output>
After completion, create `.planning/phases/02-execution-compatibility-stabilization/07-PLAN-SUMMARY.md`, `/mnt/data/cintx_phase_02_manifest_representation_matrix.json`, and `/mnt/data/cintx_phase_02_compat_parity_report.json`
</output>
