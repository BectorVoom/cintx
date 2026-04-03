# Phase 5: Re-implement detailed-design GPU path with CubeCL (wgpu backend) - Context

**Gathered:** 2026-03-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 5 re-implements the GPU execution path so supported evaluations run through a real CubeCL + wgpu backend path end-to-end (planner, dispatch, workspace, execution, and validation integration), while CPU stays limited to control-plane work.

This phase clarifies implementation policy within that scope. It does not add new product capabilities beyond the roadmap item.

</domain>

<decisions>
## Implementation Decisions

### Runtime backend policy
- **D-01:** Backend selection auto-selects among wgpu-capable adapters, then fails closed when no valid adapter/capability is available; no CPU substitute compute fallback.
- **D-02:** Capability gaps (for example required shader/device features) must return explicit typed failures rather than silent runtime substitution.
- **D-03:** Backend/device intent is control-plane metadata carried via runtime options/plumbing (not hidden executor-only policy).
- **D-04:** Runtime diagnostics and verification artifacts must include backend/adapter capability context for reproducibility.

### Planner/dispatch/execution integration strictness
- **D-05:** Phase cutline is end-to-end real CubeCL path; placeholder/synthetic execution behavior in compute path must be removed.
- **D-06:** Preserve strict ownership contract: backend output remains staging-only, compat retains final caller-visible flat writes.
- **D-07:** Chunking remains CPU control-plane only, but each chunk still executes through CubeCL compute path.
- **D-08:** Query/evaluate backend policy contract is locked; policy drift between query and evaluate must fail with typed errors.

### Unsupported-scope policy
- **D-09:** Out-of-envelope or unimplemented requests fail explicitly with typed unsupported/capability errors; no hidden fallback masking.
- **D-10:** Unsupported scope must be visible both at runtime and in artifactized reporting (matrix/report format) for verification audits.
- **D-11:** Validated4C1E policy remains strict-envelope, but backend requirement shifts from cpu-profile gate to explicit wgpu capability gating.
- **D-12:** Unimplemented family/representation paths must return specific unsupported reason taxonomy, not generic errors.

### Validation and regression gates
- **D-13:** Verification must be layered across runtime + cubecl + compat (not single-layer crate-local tests only).
- **D-14:** CI uses capability-aware required gates for wgpu regression checks (explicit skip metadata only when capability truly absent).
- **D-15:** Add explicit anti-pseudo regression assertions so synthetic execution substitutions cannot silently return.
- **D-16:** Unsupported behavior tests must assert both reason taxonomy and reporting artifact presence.

### the agent's Discretion
- Concrete Rust type/field names for backend-selection control-plane metadata and diagnostics payloads.
- Exact test and artifact file naming as long as D-13 through D-16 are satisfied.
- Exact location of helper functions used to preflight device capabilities.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope and project policy
- `.planning/ROADMAP.md` — Phase 5 scope and dependency boundary.
- `.planning/PROJECT.md` — core constraints (CubeCL primary backend, CPU control-plane only).
- `.planning/REQUIREMENTS.md` — v1 compatibility/execution/verification contracts already achieved and must not regress.
- `.planning/STATE.md` — prior phase decisions carried forward.
- `AGENTS.md` — mandatory project instructions, including CubeCL manual and test workflow constraints.

### Design authority for GPU path and fail-closed behavior
- `docs/design/cintx_detailed_design.md` — execution model and CubeCL-first architecture.
- `docs/design/cintx_detailed_design.md` — section "0.5 Design Decisions Finalized in This Revision" (Validated4C1E and fail-closed policy).
- `docs/design/cintx_detailed_design.md` — section "2.4 Performance Requirements" (compute path stays CubeCL).
- `docs/design/cintx_detailed_design.md` — section "12.5 CubeCL Execution Planning" (planner/execution policy).

### Mandatory CubeCL implementation references
- `docs/manual/Cubecl/Cubecl_vector.md` — wgpu runtime initialization and launch pattern example.
- `docs/manual/Cubecl/cubecl_matmul_gemm_example.md` — CubeCL wgpu setup and launch configuration pattern.
- `docs/manual/Cubecl/cubecl_reduce_sum.md` — reduction strategy usage and launch-policy examples.
- `docs/manual/Cubecl/cubecl_error_solution_guide/mismatched types.md` — CubeCL kernel coding constraints for compile-stable kernels.
- `docs/cubecl_error_guideline.md` — required troubleshooting/reporting process for CubeCL build/runtime failures.

### Existing execution-path integration points
- `crates/cintx-runtime/src/planner.rs` — query/evaluate contract, drift checks, and executor dispatch loop.
- `crates/cintx-runtime/src/dispatch.rs` — output ownership contract and backend executor interface.
- `crates/cintx-runtime/src/workspace.rs` — deterministic chunk planning and fallback reason model.
- `crates/cintx-cubecl/src/executor.rs` — backend execution entrypoint and current runtime-profile policy.
- `crates/cintx-cubecl/src/transfer.rs` — transfer staging contract and ownership checks.
- `crates/cintx-cubecl/src/kernels/mod.rs` — family registry and unsupported-family handling.
- `crates/cintx-compat/src/raw.rs` — compat query/eval path and safe-facade policy gate behavior.

### Testing constraints
- `docs/rust_crate_test_guideline.md` — mandatory test design guidance before creating/updating tests.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `cintx-runtime` already enforces query/evaluate contract and dispatch ownership checks (`planner.rs`, `dispatch.rs`).
- `cintx-cubecl` already has family registry, transfer planning, specialization hooks, and executor wiring (`kernels/mod.rs`, `transfer.rs`, `executor.rs`).
- `cintx-compat` raw path already routes through shared runtime planner/evaluate with typed errors (`raw.rs`).
- `cintx-runtime` workspace chunk planner already provides deterministic chunk metadata and fallback reasons (`workspace.rs`).

### Established Patterns
- Unsupported scope is represented with typed `UnsupportedApi` errors and explicit reason strings.
- Planner and runtime enforce `BackendStagingOnly -> CompatFinalWrite` ownership contract.
- Memory/plan drift handling is fail-closed with typed errors, not silent replan.
- Profile/memory-limit hints are carried through `ExecutionOptions` and compat optimizer hints.

### Integration Points
- Replace current cpu-profile lock and placeholder execution behavior in `crates/cintx-cubecl/src/executor.rs` with real wgpu-targeted policy.
- Ensure compat raw and safe-facade policy gating in `crates/cintx-compat/src/raw.rs` stays aligned with runtime/backend behavior.
- Keep planner/dispatch/workspace checks in `cintx-runtime` as contract source, while extending diagnostics for backend/device context.
- Extend tests around runtime, cubecl, and compat boundaries to catch pseudo execution and fallback masking regressions.

</code_context>

<specifics>
## Specific Ideas

- User intent is to replace non-CubeCL GPU logic, pseudo implementation, and CPU-side substitute execution with a real CubeCL path using wgpu backend.
- Remaining unsupported scope must be explicitly reported rather than masked by fallback behavior.
- Regression tests should prevent silent return of placeholder execution behavior.

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within Phase 5 scope.

</deferred>

---

*Phase: 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend*
*Context gathered: 2026-03-29*
