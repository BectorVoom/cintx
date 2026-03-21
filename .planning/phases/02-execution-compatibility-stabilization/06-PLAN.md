---
phase: 02-execution-compatibility-stabilization
plan: 06
type: execute
wave: 4
depends_on:
  - 02
  - 03
  - 04
  - 05
files_modified:
  - crates/cintx-compat/src/lib.rs
  - crates/cintx-compat/src/raw.rs
  - crates/cintx-compat/src/layout.rs
autonomous: true
requirements:
  - COMP-01
  - COMP-02
  - COMP-05
  - EXEC-02
  - EXEC-03
  - EXEC-04
must_haves:
  truths:
    - "Compat callers can pass `atm`, `bas`, `env`, `shls`, `dims`, `opt`, and `cache` into a raw API entry and reach the shared runtime/CubeCL backend."
    - "Compat callers can query output/workspace requirements without writing outputs by using the canonical `query_workspace_raw` path or by passing `out == NULL`."
    - "Invalid dims, undersized output/cache buffers, malformed raw layouts, and impossible memory limits fail before any caller-visible write occurs."
  artifacts:
    - path: crates/cintx-compat/src/raw.rs
      provides: "Raw view validation, API-to-manifest mapping, and the `query_workspace_raw` / `eval_raw` entry points."
      min_lines: 180
    - path: crates/cintx-compat/src/layout.rs
      provides: "`CompatDims`, required-element calculation, and flat output writer rules for cart/sph/spinor buffers."
      min_lines: 120
    - path: crates/cintx-compat/src/lib.rs
      provides: "Public raw-compat exports for Phase 2 callers."
      min_lines: 30
  key_links:
    - from: crates/cintx-compat/src/raw.rs
      to: crates/cintx-runtime/src/planner.rs
      via: "Validated raw calls bridge into `query_workspace()` and backend `evaluate()` instead of implementing a second planner."
      pattern: "query_workspace|evaluate"
    - from: crates/cintx-compat/src/raw.rs
      to: crates/cintx-compat/src/layout.rs
      via: "Raw calls compute required sizes and write flat outputs through the shared layout contract."
      pattern: "required_elems_from_dims|CompatDims"
---

<objective>
Implement the raw compat query/evaluate pipeline that maps libcint-style arrays and sentinel arguments onto the shared runtime and CubeCL backend.
Purpose: Deliver the actual `atm/bas/env/shls/dims/opt/cache` call surface for Phase 2 while enforcing the no-partial-write contract and split workspace-query semantics.
Output: Validated raw views, `CompatDims`, `query_workspace_raw`, `eval_raw`, and compat regression tests for layout/buffer/memory-limit failures.
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
@libcint-master/include/cint.h.in
@crates/cintx-core/src/error.rs
@crates/cintx-runtime/src/planner.rs
@crates/cintx-runtime/src/dispatch.rs
@crates/cintx-ops/src/resolver.rs
<interfaces>
From `docs/design/cintx_detailed_design.md` §5.5:
```rust
pub unsafe fn query_workspace_raw(
    api: RawApiId,
    dims: Option<&[i32]>,
    shls: &[i32],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    opt: Option<&RawOptimizerHandle>,
) -> Result<WorkspaceQuery, cintxRsError>;

pub unsafe fn eval_raw(
    api: RawApiId,
    out: Option<&mut [f64]>,
    dims: Option<&[i32]>,
    shls: &[i32],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    opt: Option<&RawOptimizerHandle>,
    cache: Option<&mut [f64]>,
) -> Result<RawEvalSummary, cintxRsError>;
```

From `crates/cintx-runtime/src/planner.rs`:
```rust
pub fn query_workspace(
    op: OperatorId,
    rep: Representation,
    basis: &BasisSet,
    shells: ShellTuple,
    opts: &ExecutionOptions,
) -> Result<WorkspaceQuery, cintxRsError>;
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Implement raw views, slot validation, and the shared dims/output contract</name>
  <files>crates/cintx-compat/src/lib.rs, crates/cintx-compat/src/raw.rs, crates/cintx-compat/src/layout.rs</files>
  <read_first>crates/cintx-compat/src/lib.rs, crates/cintx-compat/src/raw.rs, crates/cintx-compat/src/layout.rs, libcint-master/include/cint.h.in:46-70, libcint-master/include/cint.h.in:227-290, docs/design/cintx_detailed_design.md §3.6.1, §6.2, and §7.2-7.3, .planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md</read_first>
  <action>
In `raw.rs`, define the exact raw slot constants from upstream (`ATM_SLOTS = 6`, `BAS_SLOTS = 8`, `PTR_COORD`, `PTR_ZETA`, `PTR_FRAC_CHARGE`, `PTR_EXP`, `PTR_COEFF`) and use them to implement `RawAtmView`, `RawBasView`, and `RawEnvView`. Validate `atm.len() % ATM_SLOTS == 0`, `bas.len() % BAS_SLOTS == 0`, every pointer slot stays within `env.len()`, and `NPRIM_OF > 0` plus `NCTR_OF > 0`. In `layout.rs`, define `CompatDims`, `required_elems_from_dims(arity, component_count, dims, complex_interleaved)`, and flat writer helpers that treat the component axis as manifest-derived and never part of `dims`. Accept only exact `dims.len() == arity`; reject both undersized and oversized overrides with `InvalidDims`, and return `BufferTooSmall` when the caller-provided `out` or `cache` slice is shorter than the computed requirement. Re-export the raw/layout types from `lib.rs`.
  </action>
  <acceptance_criteria>
    - `rg -n "ATM_SLOTS\\s*=\\s*6|BAS_SLOTS\\s*=\\s*8" crates/cintx-compat/src/raw.rs`
    - `rg -n "struct RawAtmView|struct RawBasView|struct RawEnvView" crates/cintx-compat/src/raw.rs`
    - `rg -n "struct CompatDims|required_elems_from_dims" crates/cintx-compat/src/layout.rs`
    - `rg -n "BufferTooSmall|InvalidDims" crates/cintx-compat/src/layout.rs crates/cintx-compat/src/raw.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-compat --lib</automated>
  </verify>
  <done>The compat layer now has a single authoritative raw-layout and dims contract, satisfying the input-validation half of COMP-01, COMP-02, and COMP-05.</done>
</task>

<task type="auto">
  <name>Task 2: Bridge raw compat calls into runtime query/evaluate with exact sentinel behavior</name>
  <files>crates/cintx-compat/src/raw.rs, crates/cintx-compat/src/layout.rs</files>
  <read_first>crates/cintx-compat/src/raw.rs, crates/cintx-compat/src/layout.rs, crates/cintx-runtime/src/planner.rs, crates/cintx-runtime/src/dispatch.rs, crates/cintx-ops/src/resolver.rs, docs/design/cintx_detailed_design.md §5.5 and §7.2-7.5, .planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md</read_first>
  <action>
Implement `RawApiId`, `RawEvalSummary`, `query_workspace_raw`, and `eval_raw` in `raw.rs`. Map each raw API identifier to manifest family/operator/representation metadata through `Resolver`, validate `shls` arity against the resolved descriptor, and convert the validated raw arrays into the typed `BasisSet` and `ShellTuple` values that `cintx-runtime` expects. Enforce the exact sentinel rules from the design: `dims == None` means natural shape, `out == None` returns workspace/output requirements without writing, `cache == None` allocates through the fallible workspace path, and `opt` is forwarded when present but cannot change output layout. Use `query_workspace()` as the canonical estimator, then call backend `evaluate()` only after layout validation, output-size checks, cache checks, and workspace allocation all succeed. Return `RawEvalSummary { not0, bytes_written, workspace_bytes }`, and keep every early failure path side-effect-free on the caller's `out` slice.
  </action>
  <acceptance_criteria>
    - `rg -n "enum RawApiId" crates/cintx-compat/src/raw.rs`
    - `rg -n "struct RawEvalSummary" crates/cintx-compat/src/raw.rs`
    - `rg -n "unsafe fn query_workspace_raw" crates/cintx-compat/src/raw.rs`
    - `rg -n "unsafe fn eval_raw" crates/cintx-compat/src/raw.rs`
    - `rg -n "out\\.is_none|cache\\.is_none|dims" crates/cintx-compat/src/raw.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-compat --lib</automated>
  </verify>
  <done>Compat callers can now query workspace or execute raw APIs through the shared planner/backend path with the correct `out == NULL`, `cache == NULL`, and `dims == NULL` semantics required by COMP-02 and EXEC-02.</done>
</task>

<task type="auto">
  <name>Task 3: Add raw compat regression tests for layout, buffer, and no-partial-write behavior</name>
  <files>crates/cintx-compat/src/raw.rs, crates/cintx-compat/src/layout.rs</files>
  <read_first>crates/cintx-compat/src/raw.rs, crates/cintx-compat/src/layout.rs, .planning/phases/02-execution-compatibility-stabilization/02-RESEARCH.md, docs/design/cintx_detailed_design.md §7.3-7.5 and §11.1</read_first>
  <action>
Add focused tests inside `raw.rs` and `layout.rs` that cover: malformed `atm`/`bas` slot widths; bad `PTR_*` env offsets; invalid `dims` length for each arity; undersized output buffers; `query_workspace_raw` vs `eval_raw(out=None)` agreement; `memory_limit_bytes` chunking that either succeeds without partial writes or fails with `MemoryLimitExceeded`; and a sentinel test proving a rejected call leaves the caller's output slice unchanged. Keep the tests Phase-2-scoped to `1e`, `2e`, `2c2e`, `3c1e`, and `3c2e`.
  </action>
  <acceptance_criteria>
    - `rg -n "output slice unchanged|partial write|MemoryLimitExceeded" crates/cintx-compat/src/raw.rs crates/cintx-compat/src/layout.rs`
    - `rg -n "query_workspace_raw.*eval_raw|eval_raw.*query_workspace_raw" crates/cintx-compat/src/raw.rs`
    - `rg -n "InvalidAtmLayout|InvalidBasLayout|InvalidEnvOffset|BufferTooSmall" crates/cintx-compat/src/raw.rs crates/cintx-compat/src/layout.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-compat --lib</automated>
  </verify>
  <done>Compat regression coverage now proves raw layout failures, buffer-size failures, and impossible memory limits are explicit and side-effect-free, satisfying the remaining COMP-05 and EXEC-03 contract checks for this phase.</done>
</task>

</tasks>

<verification>
Run the compat library tests after all three tasks; the suite should cover raw validation, workspace-query semantics, and no-partial-write failure cases for the Phase 2 base families.
</verification>

<success_criteria>
`query_workspace_raw` and `eval_raw` exist, raw inputs are validated against upstream slot rules and dims semantics, and failure cases are tested to prove no partial writes occur.
</success_criteria>

<output>
After completion, create `.planning/phases/02-execution-compatibility-stabilization/06-PLAN-SUMMARY.md`
</output>
