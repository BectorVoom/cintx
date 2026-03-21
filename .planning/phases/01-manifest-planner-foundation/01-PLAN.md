---
phase: 01-manifest-planner-foundation
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/cintx-core/src/atom.rs
  - crates/cintx-core/src/basis.rs
  - crates/cintx-core/src/env.rs
  - crates/cintx-core/src/operator.rs
  - crates/cintx-core/src/shell.rs
  - crates/cintx-core/src/tensor.rs
  - crates/cintx-core/src/lib.rs
  - crates/cintx-ops/build.rs
  - crates/cintx-ops/src/resolver.rs
  - crates/cintx-ops/src/generated/api_manifest.rs
  - compiled_manifest.lock.json
  - crates/cintx-ops/src/lib.rs
autonomous: true
requirements:
  - BASE-01
  - BASE-02
  - BASE-03
must_haves:
  truths:
    - "Rust callers can instantiate typed `Atom`, `Shell`, `BasisSet`, `EnvParams`, `Representation`, `OperatorId`, and tensor metadata that follow docs/design/cintx_detailed_design.md §6 and remain immutable/shared via `Arc`, satisfying BASE-01."
    - "The canonical manifest lives at `crates/cintx-ops/generated/compiled_manifest.lock.json`, records every family in the {base, with-f12, with-4c1e, with-f12+with-4c1e} support matrix, and the build script verifies the four-profile diff before emitting resolver tables, satisfying BASE-02."
    - "Resolver data structures expose every manifest field (`family_name`, `symbol_name`, `category`, `arity`, `representation`, `feature_flag`, `stability`, `helper_kind`, `compiled_in_profiles`, `oracle_covered`, `canonical_family`) so runtime selects kernels without referencing raw symbol names, satisfying BASE-03."
  artifacts:
    - path: crates/cintx-core/src/atom.rs
      provides: "Typed atom/ shell metadata plus shared tensors that the public API will consume."
      min_lines: 40
    - path: crates/cintx-ops/src/generated/api_manifest.rs
      provides: "Generated `ManifestEntry` table with canonical metadata for every approved profile."
      min_lines: 120
    - path: crates/cintx-ops/src/resolver.rs
      provides: "Resolver helpers that index `ManifestEntry` rows into `OperatorDescriptor`/`OperatorId` without leaking symbol strings."
      min_lines: 80
  key_links:
    - from: crates/cintx-ops/src/resolver.rs
      to: crates/cintx-ops/generated/compiled_manifest.lock.json
      via: "Loads `ManifestEntry` records and attaches metadata to each `OperatorDescriptor`."
      pattern: "ManifestEntry"
    - from: crates/cintx-ops/src/generated/api_manifest.rs
      to: compiled_manifest.lock.json
      via: "Code-generated table that mirrors the canonical lock."
      pattern: "include_str!(.*compiled_manifest.lock.json)"
---

<objective>
Create the typed domain primitives and canonical manifest/resolver pipeline that Phase 1 users and downstream runtime can build against.
Purpose: Lock in BASE-01 (typed inputs), BASE-02 (manifest lock), and BASE-03 (manifest-aware registry) before planner execution logic consumes them.
Output: Domain structs in `cintx-core`, a moved canonical lock plus generation step, and a resolver that maps manifest metadata to `OperatorId`.
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
@docs/design/cintx_detailed_design.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Implement typed domain primitives in `cintx-core`</name>
  <files>crates/cintx-core/src/atom.rs, crates/cintx-core/src/basis.rs, crates/cintx-core/src/env.rs, crates/cintx-core/src/operator.rs, crates/cintx-core/src/shell.rs, crates/cintx-core/src/tensor.rs, crates/cintx-core/src/lib.rs</files>
  <read_first>crates/cintx-core/src/lib.rs, docs/design/cintx_detailed_design.md §6.1-6.9, AGENTS.md (stack/table constraints)</read_first>
  <action>
Define `Atom`, `Shell`, `BasisSet`, `EnvParams`, `Representation`, `OperatorId`, `ShellTuple`, `TensorShape`, `TensorLayout`, and supporting helpers so the safe API only exposes typed, immutable structures per D-05..D-08. In `atom.rs`, implement `pub struct Atom { atomic_number: u16, coord_bohr: [f64; 3], nuclear_model: NuclearModel, zeta: Option<f64>, fractional_charge: Option<f64> }` plus a `NuclearModel` enum backed by `#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]`. In `shell.rs`, define `Shell` with the fields `atom_index`, `ang_momentum`, `nprim`, `nctr`, `kappa`, and `Arc<[f64]>` for `exponents`/`coefficients`, plus a `ShellTuple` wrapper over `smallvec::SmallVec<[Arc<Shell>; 4]>` for arity safety. In `basis.rs`, expose `BasisSet` that owns a shared `Arc<[Arc<Shell>]>`, references the owning atoms, and caches derived `BasisMeta` (offsets/ao counts) without touching raw arrays. In `env.rs`, make `EnvParams` hold an `Arc<[f64]>` environment vector, an optional `EnvUnits` enum, and companion bounds-check helpers so we do not expose raw offsets (per D-06). In `operator.rs`, define the `Representation` enum (`Cart`, `Spheric`, `Spinor`) and a lean `OperatorId(u32)` newtype with accessor helpers. In `tensor.rs`, model `TensorShape` (`batch`, `comp`, `extents: SmallVec<[usize; 4]>`, `complex_interleaved`) and `TensorLayout` (`strides: SmallVec<[usize; 6]>`, `column_major_compat`, `comp_is_leading`) so clients can reason about strides/shapes. Re-export the modules from `lib.rs`.
  </action>
  <acceptance_criteria>
    - `rg -n "pub struct Atom" crates/cintx-core/src/atom.rs`
    - `rg -n "pub struct Shell" crates/cintx-core/src/shell.rs`
    - `rg -n "ShellTuple" crates/cintx-core/src/shell.rs`
    - `rg -n "pub struct BasisSet" crates/cintx-core/src/basis.rs`
    - `rg -n "pub struct EnvParams" crates/cintx-core/src/env.rs`
    - `rg -n "pub enum Representation" crates/cintx-core/src/operator.rs`
    - `rg -n "pub struct OperatorId" crates/cintx-core/src/operator.rs`
    - `rg -n "pub struct TensorShape" crates/cintx-core/src/tensor.rs`
    - `rg -n "pub struct TensorLayout" crates/cintx-core/src/tensor.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-core --lib</automated>
  </verify>
  <done>BASE-01 is satisfied: the safe API exposes fully typed, `Arc`-backed domain objects, raw layouts are confined to compat, and tensor metadata captures shape/stride details per docs/design §6.</done>
</task>

<task type="auto">
  <name>Task 2: Move canonical manifest & build the resolver table</name>
  <files>compiled_manifest.lock.json, crates/cintx-ops/build.rs, crates/cintx-ops/src/resolver.rs, crates/cintx-ops/src/generated/api_manifest.rs, crates/cintx-ops/src/lib.rs</files>
  <read_first>compiled_manifest.lock.json, docs/design/cintx_detailed_design.md §3.1-3.4, §4.6, crates/cintx-ops/build.rs, crates/cintx-ops/src/generated/api_manifest.rs, crates/cintx-ops/src/resolver.rs</read_first>
  <action>
Per D-01..D-04 & D-15, migrate the existing root lock into `crates/cintx-ops/generated/compiled_manifest.lock.json`, keep the new path as the canonical artifact, and update `build.rs` to load that file, assert the observed union exactly contains {base, with-f12, with-4c1e, with-f12+with-4c1e}, and regenerate `generated/api_manifest.rs` whenever the schema or profiles change. Implement `ManifestEntry` in `resolver.rs` with every column from docs/design §3.3 (`family_name`, `symbol_name`, `category`, `arity`, `forms`, `component_rank`, `feature_flag`, `stability`, `declared_in`, `compiled_in_profiles`, `oracle_covered`, `helper_kind`, `canonical_family`) plus derived `Representation` support. Use the build script to emit a `pub const MANIFEST_ENTRIES: &[ManifestEntry]` array so resolver helpers can map entry index → `OperatorId`/`OperatorDescriptor` without raw symbol strings; include `feature_flag`/`stability` in `OperatorDescriptor` and expose a `Resolver::descriptor(id: OperatorId) -> &'static OperatorDescriptor`. Keep helper/legacy metadata with the same manifest, and make the resolver report an error when a symbol is missing from the canonical lock so BASE-03 is enforced.
  </action>
  <acceptance_criteria>
    - `rg -n "ManifestEntry" crates/cintx-ops/src/resolver.rs`
    - `rg -n "compiled_manifest.lock.json" crates/cintx-ops/build.rs`
    - `rg -n "support_matrix" compiled_manifest.lock.json`
    - `rg -n "MANIFEST_ENTRIES" crates/cintx-ops/src/generated/api_manifest.rs`
    - `rg -n "Resolver::descriptor" crates/cintx-ops/src/resolver.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-ops --lib</automated>
  </verify>
  <done>BASE-02 and BASE-03 are satisfied: the canonical lock now lives under `cintx-ops/generated`, the build script keeps it in sync with the four-profile matrix, and the resolver exposes metadata-driven `OperatorDescriptor` lookup without leaking raw symbol names.</done>
</task>

</tasks>

<verification>
Wave 1 builders can run `cargo test -p cintx-core --lib` and `cargo test -p cintx-ops --lib` to ensure the domain & resolver compile.
</verification>

<success_criteria>
All tasks compile, the canonical lock file is relocated, and the resolver can map manifest rows to `OperatorId`/`OperatorDescriptor` without touching raw symbol names.
</success_criteria>

<output>
After completion, create `.planning/phases/01-manifest-planner-foundation/01-PLAN-SUMMARY.md`
</output>
