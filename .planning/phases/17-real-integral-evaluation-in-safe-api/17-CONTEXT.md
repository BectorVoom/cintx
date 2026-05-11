# Phase 17: Real-Integral Evaluation in Safe API - Context

**Gathered:** 2026-05-12
**Status:** Ready for planning

<domain>
## Phase Boundary

`SessionRequest::evaluate` returns real libcint-compatible values for every arity-2 intor the safe API routes today. The synthetic `(idx + 1)` / `((idx + 1) * 0.5)` placeholder in `crates/cintx-rs/src/api.rs::fill_staging_values` disappears, replaced with the same `cintx_cubecl::CubeClExecutor` dispatch that `cintx-compat::raw::eval_raw` already drives. No public API change: `SessionRequest`, `SessionQuery`, `TypedEvaluationOutput`, `IntegralTensor`, and `FacadeError` stay source- and SemVer-compatible with v1.2. Verification is byte-identity vs vendored libcint 6.1.3 at the unified Phase 15 tolerance (`atol=1e-12`) for the 12 base arity-2 operators (1e × {ovlp, kin, nuc} × {cart, sph, spinor} + 2c2e × {cart, sph, spinor}).

</domain>

<decisions>
## Implementation Decisions

### Dispatch mechanism
- **D-01:** **Swap the safe-API's shadow `CubeClExecutor` for `cintx_cubecl::CubeClExecutor`.** Delete the local stub struct `CubeClExecutor` (`crates/cintx-rs/src/api.rs:492-562`) and the synthetic helper `fill_staging_values` (`crates/cintx-rs/src/api.rs:465-490`). Import the real executor with `use cintx_cubecl::CubeClExecutor;` (it's already re-exported at `cintx_cubecl::lib.rs:26`). The existing chunk loop, `ExecutionPlan::new`, `HostWorkspaceAllocator`, `schedule_chunks`, and `ExecutionIo` plumbing in `SessionQuery::evaluate` stay as-is — they already match what compat does.
- **D-02:** **No `unsafe` inside the safe-API path.** The safe API does NOT call `cintx_compat::raw::eval_raw`. Going through `eval_raw` would require repacking typed `BasisSet` / `ShellTuple` back into `atm`/`bas`/`env` arrays (an `unsafe` boundary that runs the typed→raw round-trip backwards). The real `CubeClExecutor` is the shared dispatch primitive — both surfaces consume it directly. ROADMAP's "or an equivalent compat dispatch" wording is satisfied by sharing the executor, not by funneling through the raw FFI shape.
- **D-03:** **No new shared helper crate / module this phase.** Do NOT hoist the chunk loop into a shared `cintx-runtime` or `cintx-cubecl` helper. The current duplication between `cintx-rs/src/api.rs::SessionQuery::evaluate` and `cintx-compat/src/raw.rs::eval_raw` is acceptable for Phase 17 (Task 3 is explicitly scoped as the smallest of the three v1.3 tasks). A unifying helper is a deferred follow-up — see `<deferred>`.

### Test handling (existing synthetic-value assertion)
- **D-04:** **Rewrite `evaluate_runs_runtime_path_and_returns_owned_output`** (`crates/cintx-rs/src/api.rs:659-688`). Drop the brittle `owned_values[0] == 1.0` line (an assertion of the synthetic pattern). Replace with a deterministic + nonzero smoke check that does NOT depend on the oracle harness:
  1. Call `evaluate()` twice with the same request; assert element-wise equality of `owned_values` (idempotency).
  2. Assert at least one element of `owned_values` is non-zero (`|v| > 1e-18`) so a regression to a zero-fill stub fails the test.
  3. Keep the existing extent/byte-count invariants (`owned_values.len() == extents.iter().product()`, `bytes_written == owned_values.len() * size_of::<f64>()`, `transfer_bytes > 0`, workspace/chunk count match query).
- **D-05:** **No inline `eval_raw` cross-check in `cintx-rs` unit tests.** The cintx-rs crate stays free of compat-side dependencies for its own unit smoke (the deterministic+nonzero check is enough at the unit level). Byte-identity vs libcint lives entirely in the new `cintx-oracle` parity test (D-06).

### Oracle test shape
- **D-06:** **New file: `crates/cintx-oracle/tests/safe_api_arity2_parity.rs`.** Single-purpose file focused on "does the safe API return byte-identical values to vendored libcint 6.1.3?". The existing `one_electron_parity.rs`, `center_2c2e_parity.rs`, etc. stay `eval_raw`-driven — Phase 17 does NOT modify them. This separation keeps "safe-API correctness" diagnosable from "compat correctness" if either regresses.
- **D-07:** **Per-symbol tests inside the new file**, mirroring the existing `one_electron_parity.rs` per-symbol pattern (`test_int1e_ovlp_sph_h2o_sto3g_parity`, `test_int1e_kin_sph_h2o_sto3g_vendor_parity`, …). 12 tests total (D-08 coverage). One molecule + basis (H2O / STO-3G) is enough to start; per-symbol naming makes per-operator regressions trivially diagnosable in CI output. Data-driven parametric sweeps are deferred.
- **D-08:** **Coverage set: 12 base arity-2 operators.**
  - `int1e_ovlp_{cart, sph, spinor}`
  - `int1e_kin_{cart, sph, spinor}`
  - `int1e_nuc_{cart, sph, spinor}`
  - `int2c2e_{cart, sph, spinor}`

  This matches "every arity-2 intor the safe API currently routes" under default features. Unstable-source arity-2 symbols (e.g., `int1e_grids_sph`) stay behind `#[cfg(feature = "unstable-source-api")]` and are NOT in this sweep — they get their own follow-up if/when the safe-API path adopts them. F12 family is arity-4 (not arity-2); with-4c1e is arity-4. Neither contributes operators to Phase 17.
- **D-09:** **Tolerance: `atol=1e-12, rtol=0.0`** (the unified Phase 15 oracle tolerance — `.planning/phases/15-oracle-tolerance-unification-manifest-lock-closure/15-CONTEXT.md`). The existing 1e tests use `atol=1e-11/rtol=1e-9` because they predate Phase 15 unification; the new safe-API tests align to the unified tolerance from day one. If a vendored-libcint comparison fails at 1e-12 but passes at 1e-11, that's a real divergence — investigate before relaxing.
- **D-10:** **CI gating: required.** The new tests run as part of the existing `oracle_parity_gate` matrix (it's already cpu/wgpu × profile matrix). No new CI job; reuse the existing gate. Per-symbol failures appear as individual `test_*` lines so CI bisection is straightforward.

### Public API stability (SC #3 / RVAL-03)
- **D-11:** **Zero changes to types in `cintx-rs::api` or `cintx-rs::prelude`.** Public structs (`SessionRequest`, `SessionQuery`, `WorkspacePlan`, `TypedEvaluationOutput`, `IntegralTensor`, `EvaluationStats`), public functions (`SessionRequest::new`, `query_workspace`, `SessionQuery::evaluate`), and `FacadeError` variants stay source-compatible with v1.2. No new `pub` items added by this phase — the change is purely internal. Researcher must confirm this on the v1.2 baseline (e.g., via `cargo public-api` diff or manifest comparison).
- **D-12:** **`fill_staging_values` and `CubeClExecutor` (the stub) are deletable.** They are private to `cintx-rs::api` (no `pub`). Deletion does not affect any downstream crate. Researcher should grep for stray references in tests / benches / examples; if any exist they convert to the real executor.

### Claude's Discretion
- Whether to leave the old test name `evaluate_runs_runtime_path_and_returns_owned_output` or rename it to something more descriptive (e.g., `evaluate_returns_deterministic_nonzero_real_values`). Lean toward renaming for honesty about what's asserted now.
- Exact module organization of the new `safe_api_arity2_parity.rs` (whether to factor a shared helper for the SessionRequest → matrix collection, or inline per-test). Pattern-match `one_electron_parity.rs`'s existing helpers (`collect_1e_sph_matrix`, `collect_1e_sph_matrix_vendor`, `count_mismatches`) — likely a `collect_safe_api_matrix(operator, representation, &basis, &shells)` helper.
- Basis fixture choice for the parity sweep. H2O / STO-3G is sufficient and matches the existing parity tests; researcher can confirm whether a second molecule (e.g., a heavier-atom case) adds coverage without ballooning CI time. Default: H2O / STO-3G only for Phase 17.
- Whether to add a #[cfg(has_vendor_libcint)] guard pattern to the new tests (matching the existing `test_*_vendor_parity` convention) so the file still compiles on systems without the vendored libcint build artifact. Strongly lean yes — preserves CI portability.
- Whether `int2c2e_spinor` requires any spinor-transform pre-/post-work this phase. Phase 12 landed real spinor transforms; arity-2 spinor cases should round-trip cleanly. If the researcher finds a gap, surface it as a planner-time blocker — but do NOT bake spinor transform changes into Phase 17 plans.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase intent and scope
- `.planning/ROADMAP.md` § "Phase 17: Real-Integral Evaluation in Safe API" — locks goal, success criteria 1–3 (RVAL-01/02/03), and downstream-impact note for pyscf_rs `pyscf-gto/src/intor.rs`.
- `.planning/PROJECT.md` § Constraints — CubeCL is the primary compute backend; safe API is the first-priority surface.
- `.planning/notes/pyscf-rs-as-cintx-consumer.md` — downstream consumer context (issue #11). Two independent oracle gates exist: cintx-oracle (this repo) and pyscf_rs `tests/oracle/`. Phase 17 makes both green for arity-2.

### Phase 15 tolerance baseline (required for D-09)
- `.planning/phases/15-oracle-tolerance-unification-manifest-lock-closure/15-CONTEXT.md` — unified oracle tolerance is `atol=1e-12` with the four-profile manifest lock. New tests adopt this tolerance directly.

### Existing safe-API surface (the code being changed)
- `crates/cintx-rs/src/api.rs` — host of `SessionRequest`, `SessionQuery`, `evaluate()` chunk loop, the shadow `CubeClExecutor` stub (lines 492-562), and `fill_staging_values` (lines 465-490) that this phase deletes.
- `crates/cintx-rs/src/lib.rs` and `crates/cintx-rs/src/prelude.rs` — confirm public re-exports stay unchanged (D-11).
- `crates/cintx-rs/src/error.rs` — `FacadeError` variants stay unchanged.

### Real dispatch reference (the path being mirrored)
- `crates/cintx-compat/src/raw.rs` lines 6, 411-557 — `eval_raw` uses `cintx_cubecl::CubeClExecutor` with `HostWorkspaceAllocator`, `schedule_chunks`, `ExecutionIo` — the exact pattern Phase 17's safe API adopts.
- `crates/cintx-cubecl/src/lib.rs` line 26 — `pub use executor::{BackendCache, CUBECL_RUNTIME_PROFILE, CubeClExecutor, check_shader_f64_in_features};` — the import target.
- `crates/cintx-cubecl/src/executor.rs` lines 35-200 — the real `CubeClExecutor` struct and its `BackendExecutor` impl. No safe-API changes needed here.

### Oracle test pattern reference
- `crates/cintx-oracle/tests/one_electron_parity.rs` — pattern source for the new `safe_api_arity2_parity.rs`. Reuse the `count_mismatches` helper, the `build_h2o_sto3g` fixture, the per-symbol test naming, and the `#[cfg(has_vendor_libcint)]` guard convention.
- `crates/cintx-oracle/tests/center_2c2e_parity.rs` — adjacent pattern reference for 2c2e parity helpers (if any are reusable).
- `.github/workflows/` `oracle_parity_gate` — existing CI gate; the new tests run inside it without a new job (D-10).

### Manifest (arity-2 enumeration source)
- `crates/cintx-ops/src/generated/api_manifest.csv` — authoritative list of arity-2 operators by family. The 12 base operators (D-08) are derivable from `arity == 2 AND stability == "stable" AND helper_kind == "operator" AND canonical_family IN {1e, 2c2e}`.

### Downstream consumer surface (for verification context)
- pyscf_rs `crates/pyscf-gto/src/intor.rs` (private repo; sibling path-dep) — primary downstream consumer of `SessionRequest`. Phase 17 unblocks every arity-2 intor in this file on land. Researcher does NOT need read access; the consumer-side oracle gate (pyscf_rs `tests/oracle/`) is the safety net.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`cintx_cubecl::CubeClExecutor`** (re-exported at `crates/cintx-cubecl/src/lib.rs:26`) — the real executor that drives kernels. Already consumed by `cintx-compat::raw::eval_raw`. Safe API imports and uses it directly under D-01.
- **`SessionQuery::evaluate` chunk loop** (`crates/cintx-rs/src/api.rs:181-256`) — `schedule_chunks`, per-chunk allocator dance, `ExecutionIo::new`, and the copy-into-accumulator step are correct as-is. Only the `CubeClExecutor` binding inside the loop changes.
- **`enforce_safe_facade_policy_gate`** (`crates/cintx-compat/src/raw.rs:816`) — already validates source/profile/F12/4c1e envelopes; Phase 17 reuses it unchanged.
- **Oracle test helpers** (`crates/cintx-oracle/tests/one_electron_parity.rs`): `count_mismatches`, `build_h2o_sto3g`, `collect_1e_sph_matrix_vendor`. Reusable for the new sweep; D-07 leans toward a sibling `collect_safe_api_matrix` helper.
- **`HostWorkspaceAllocator`, `schedule_chunks`, `ExecutionIo`, `ExecutionPlan`** (all in `cintx-runtime`) — already wired into both surfaces; no change.

### Established Patterns
- **Per-symbol parity tests** (`one_electron_parity.rs`): one function per operator × representation, `#[cfg(has_vendor_libcint)]` for vendor-comparison tests, fixed H2O / STO-3G fixture, `assert_eq!(mismatches, 0, ...)` style. Phase 17 mirrors this exactly.
- **Tolerance literals at top of test file** (`atol = 1e-11_f64; rtol = 1e-9_f64;` today). New file declares `atol = 1e-12_f64; rtol = 0.0_f64;` to match Phase 15 unified tolerance.
- **Compat-and-safe-API duplicate chunk loops** — currently both `cintx-rs::SessionQuery::evaluate` and `cintx-compat::raw::eval_raw` carry their own chunk-loop body. Phase 17 keeps this duplication (D-03); unification deferred.
- **Internal-only private types** — the shadow `CubeClExecutor` and `fill_staging_values` are not `pub` and not re-exported; safe to delete (D-12).

### Integration Points
- `crates/cintx-rs/src/api.rs::SessionQuery::evaluate` — the single function this phase modifies. Imports change (`use cintx_cubecl::CubeClExecutor;`), the local `CubeClExecutor` struct + impl + `fill_staging_values` get deleted, and the existing `let executor = CubeClExecutor::new();` line keeps working (the new type also has `::new()`).
- `crates/cintx-rs/src/api.rs::evaluate_runs_runtime_path_and_returns_owned_output` (test) — modify per D-04.
- `crates/cintx-oracle/tests/safe_api_arity2_parity.rs` (new) — 12 per-symbol tests under D-06/D-07/D-08, `#[cfg(has_vendor_libcint)]` guarded.
- No changes required in `cintx-runtime`, `cintx-cubecl`, `cintx-ops`, `cintx-compat`, or `cintx-capi`.

</code_context>

<specifics>
## Specific Ideas

- The safe API and compat surfaces currently each own a copy of the chunk-loop body. That duplication is **intentional for Phase 17 scope**; Phase 17 must not refactor it. Anyone tempted to extract a shared helper is doing Phase 17.5 work — push it to a follow-up.
- The deletion of `fill_staging_values` is a one-line behavioral fix from the user's perspective ("safe API now returns real values"), but it's the SC #1 deliverable — make sure the planner doesn't bury it in a larger refactor.
- The 12-operator parity sweep is the visible-from-CI evidence that SC #1 + SC #2 landed together. Keep it as 12 named test functions, not a single parametric loop — per-symbol failure messages are what reviewers want when oracle regresses.
- pyscf_rs's `pyscf-gto/src/intor.rs` is unblocked on land. The pyscf_rs-side oracle (`tests/oracle/`) is independent and is the second gate; Phase 17 doesn't need to coordinate with it explicitly, but the user has access to that repo and will run it after merge.

</specifics>

<deferred>
## Deferred Ideas

- **Shared dispatch helper between safe API and compat.** Hoist the chunk loop (`ExecutionPlan + CubeClExecutor.execute` per chunk) into a single helper in `cintx-runtime` or `cintx-cubecl`. Both surfaces then call it. Cleanest long-term layering but explicitly out of scope for Phase 17 (D-03). Candidate for v1.3 polish phase or v1.4.
- **Unstable-source arity-2 sweep through `SessionRequest`.** `int1e_grids_sph` and related arity-2 symbols behind `unstable-source-api`. Adds CI surface and exercises the source-only safe-API path. Defer until pyscf_rs (or another consumer) actually drives the path.
- **Multi-molecule oracle fixtures for the safe-API sweep.** H2O / STO-3G is enough to prove parity. Heavier-atom cases (e.g., for `int2c2e_spinor`) could surface edge cases. Add when CI budget allows or a regression motivates it.
- **Data-driven parametric sweep helper** that loops over a `&[(symbol, representation)]` table instead of 12 named tests. Better ergonomics if the arity-2 set grows; not worth the indirection while the set is fixed at 12.
- **Inline `eval_raw` cross-check inside `cintx-rs` unit tests.** Decided against (D-05) — keeps cintx-rs free of compat-side test deps. If a unit-level byte-identity smoke ever becomes useful, revisit.
- **Phase 18 arity-3/4 dispatch and Phase 19 ECP** (issue #11 Task 2/Task 1). Already roadmap'd as separate phases; Phase 17 unblocks them but does not anticipate their decisions.

</deferred>

---

*Phase: 17-real-integral-evaluation-in-safe-api*
*Context gathered: 2026-05-12*
