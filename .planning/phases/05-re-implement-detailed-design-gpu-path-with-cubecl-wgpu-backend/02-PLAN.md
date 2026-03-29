---
phase: 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend
plan: 02
type: execute
wave: 2
depends_on:
  - 01
files_modified:
  - crates/cintx-cubecl/Cargo.toml
  - crates/cintx-cubecl/src/capability.rs
  - crates/cintx-cubecl/src/runtime_bootstrap.rs
  - crates/cintx-cubecl/src/lib.rs
autonomous: true
requirements:
  - EXEC-02
  - COMP-05
  - VERI-04
must_haves:
  truths:
    - "CubeCL backend bootstrap auto-selects wgpu adapters or fails with explicit typed reasons."
    - "Capability metadata is captured as reproducible control-plane data and can be hashed into a runtime token."
    - "No hidden CPU compute fallback path is introduced during backend setup."
  artifacts:
    - path: crates/cintx-cubecl/src/capability.rs
      provides: "Capability snapshot/reason taxonomy for wgpu preflight."
      min_lines: 140
    - path: crates/cintx-cubecl/src/runtime_bootstrap.rs
      provides: "WGPU selector parsing and fail-closed adapter bootstrap helpers."
      min_lines: 180
    - path: crates/cintx-cubecl/Cargo.toml
      provides: "Explicit cubecl-wgpu runtime dependency wiring for phase-5 execution path."
      min_lines: 20
  key_links:
    - from: crates/cintx-runtime/src/options.rs
      to: crates/cintx-cubecl/src/runtime_bootstrap.rs
      via: "ExecutionOptions.backend_intent drives adapter-selection policy."
      pattern: "BackendIntent|selector|bootstrap_wgpu_runtime"
    - from: crates/cintx-cubecl/src/runtime_bootstrap.rs
      to: crates/cintx-cubecl/src/capability.rs
      via: "Bootstrap emits capability snapshot + fingerprint token used for drift checks and diagnostics."
      pattern: "WgpuCapabilitySnapshot|capability_fingerprint"
---

<objective>
Add typed wgpu runtime bootstrap and capability snapshot modules for the CubeCL backend.
Purpose: Implement D-01 through D-04 fail-closed backend selection before executor rewrite work.
Output: New cubecl capability/bootstrap modules, dependency wiring, and focused preflight tests.
</objective>

<execution_context>
@/home/chemtech/.codex/get-shit-done/workflows/execute-plan.md
@/home/chemtech/.codex/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/05-CONTEXT.md
@.planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/05-RESEARCH.md
@AGENTS.md
@docs/manual/Cubecl/Cubecl_vector.md
@docs/manual/Cubecl/cubecl_matmul_gemm_example.md
@docs/manual/Cubecl/cubecl_reduce_sum.md
@docs/manual/Cubecl/cubecl_error_solution_guide/mismatched types.md
@crates/cintx-cubecl/Cargo.toml
@crates/cintx-cubecl/src/lib.rs
@crates/cintx-cubecl/src/executor.rs
@crates/cintx-cubecl/src/resident_cache.rs
<interfaces>
From `crates/cintx-cubecl/src/lib.rs`:
```rust
pub use executor::{CUBECL_RUNTIME_PROFILE, CubeClExecutor};
pub use resident_cache::{DeviceResidentCache, ResidentCache};
```

From `crates/cintx-runtime/src/options.rs` (Plan 01 output):
```rust
pub struct BackendIntent {
    pub backend: BackendKind,
    pub selector: String,
}
pub struct BackendCapabilityToken {
    pub adapter_name: String,
    pub backend_api: String,
    pub capability_fingerprint: u64,
}
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Define CubeCL capability snapshot and reason taxonomy contracts</name>
  <files>crates/cintx-cubecl/src/capability.rs, crates/cintx-cubecl/src/lib.rs</files>
  <read_first>crates/cintx-cubecl/src/lib.rs, crates/cintx-cubecl/src/executor.rs, .planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/05-CONTEXT.md, docs/design/cintx_detailed_design.md, docs/manual/Cubecl/cubecl_matmul_gemm_example.md</read_first>
  <behavior>
    - Test 1: Capability fingerprint remains stable for identical snapshot input.
    - Test 2: Capability fingerprint changes when adapter/features/limits fields change.
    - Test 3: Unsupported reason taxonomy formats explicit reason classes for missing adapter/feature/limit.
  </behavior>
  <action>
Create `capability.rs` with concrete types `WgpuCapabilitySnapshot`, `CapabilityReason`, and `WgpuPreflightReport`. Include helper `capability_fingerprint(snapshot: &WgpuCapabilitySnapshot) -> u64` using deterministic hashing of adapter name, backend API, feature bits, and normalized limits map. Define reason variants matching D-02 and D-12 (`MissingAdapter`, `MissingFeature`, `LimitTooLow`, `FamilyUnsupported`, `RepresentationUnsupported`) and a formatter that emits reason-prefixed strings (`missing_adapter`, `missing_feature:<name>`, `limit_too_low:<name>`). Export these types from `lib.rs`.
  </action>
  <acceptance_criteria>
    - `rg -n "struct WgpuCapabilitySnapshot|enum CapabilityReason|struct WgpuPreflightReport|fn capability_fingerprint" crates/cintx-cubecl/src/capability.rs`
    - `rg -n "MissingAdapter|MissingFeature|LimitTooLow|FamilyUnsupported|RepresentationUnsupported" crates/cintx-cubecl/src/capability.rs`
    - `rg -n "pub use capability::|WgpuCapabilitySnapshot|CapabilityReason" crates/cintx-cubecl/src/lib.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-cubecl capability::tests::capability_fingerprint_is_deterministic -- --exact && cargo test -p cintx-cubecl capability::tests::capability_fingerprint_changes_when_snapshot_changes -- --exact</automated>
  </verify>
  <done>CubeCL crate has reusable capability snapshot/taxonomy contracts with deterministic fingerprint behavior.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Implement wgpu bootstrap preflight and dependency wiring</name>
  <files>crates/cintx-cubecl/Cargo.toml, crates/cintx-cubecl/src/runtime_bootstrap.rs, crates/cintx-cubecl/src/lib.rs</files>
  <read_first>crates/cintx-cubecl/Cargo.toml, crates/cintx-cubecl/src/lib.rs, crates/cintx-cubecl/src/executor.rs, docs/manual/Cubecl/Cubecl_vector.md, docs/manual/Cubecl/cubecl_matmul_gemm_example.md, .planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/05-RESEARCH.md</read_first>
  <behavior>
    - Test 1: Selector parser accepts `auto`, `default`, `discrete:0`, and `integrated:0`.
    - Test 2: Invalid selector yields typed `UnsupportedApi` with `missing_adapter` reason text.
    - Test 3: Preflight report returns capability token fields required by runtime drift enforcement.
  </behavior>
  <action>
Add explicit crate dependencies in `crates/cintx-cubecl/Cargo.toml`: `cubecl-wgpu = "0.9.0"` and `cubecl-runtime = "0.9.0"`. Create `runtime_bootstrap.rs` exposing `bootstrap_wgpu_runtime(intent: &cintx_runtime::options::BackendIntent) -> Result<WgpuPreflightReport, cintxRsError>`. Implement concrete selector parsing (`auto|default|discrete:N|integrated:N`) and adapter preflight with fail-closed mapping to `cintxRsError::UnsupportedApi { requested: format!("wgpu-capability:{reason}") }` per D-01/D-02. Fill `WgpuCapabilitySnapshot` from adapter info/features/limits and populate `BackendCapabilityToken` values used by runtime contract checks (D-03/D-04). Export bootstrap module in `lib.rs`.
  </action>
  <acceptance_criteria>
    - `rg -n "cubecl-wgpu\\s*=\\s*\"0\\.9\\.0\"|cubecl-runtime\\s*=\\s*\"0\\.9\\.0\"" crates/cintx-cubecl/Cargo.toml`
    - `rg -n "fn bootstrap_wgpu_runtime|discrete:|integrated:|missing_adapter|wgpu-capability" crates/cintx-cubecl/src/runtime_bootstrap.rs`
    - `rg -n "WgpuPreflightReport|BackendCapabilityToken|capability_fingerprint" crates/cintx-cubecl/src/runtime_bootstrap.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-cubecl runtime_bootstrap::tests::selector_parser_accepts_auto_default_discrete_integrated -- --exact && cargo test -p cintx-cubecl runtime_bootstrap::tests::invalid_selector_returns_typed_missing_adapter_error -- --exact</automated>
  </verify>
  <done>CubeCL has explicit wgpu bootstrap helpers that auto-select adapters or fail closed with typed capability reasons and tokenized metadata.</done>
</task>

</tasks>

<verification>
Build and run cubecl bootstrap unit tests; confirm capability snapshot/token fields are emitted and selector parsing is deterministic.
</verification>

<success_criteria>
WGPU backend preflight can be called by executor code without hidden fallback logic, and all capability failures map to explicit typed reasons.
</success_criteria>

<output>
After completion, create `.planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/02-PLAN-SUMMARY.md`
</output>
