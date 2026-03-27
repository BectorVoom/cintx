# Phase 3: Safe Surface, C ABI Shim & Optional Families - Context

**Gathered:** 2026-03-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 3 delivers the public safe Rust facade (`cintx-rs`), an optional migration-focused C ABI shim (`cintx-capi`), and optional-family exposure gates (`with-f12`, `with-4c1e`, `unstable-source-api`) on top of the already-stable runtime/compat foundation from Phase 2.

This phase clarifies *how* those surfaces behave. It does not add new integral capabilities beyond the Phase 3 roadmap scope.

</domain>

<decisions>
## Implementation Decisions

### Safe API Contract
- **D-01:** The safe facade uses a typed session/request object so `query_workspace()` and `evaluate()` remain explicitly connected.
- **D-02:** `evaluate()` returns owned typed outputs by default (no caller-managed raw output buffers in safe mode).
- **D-03:** `query_workspace()` returns structured planning metadata (bytes/chunks plus execution token contract), not only a scalar byte count.
- **D-04:** Safe API errors are exposed through a stable facade-level typed enum that preserves core categories (`UnsupportedApi`, layout, memory, validation).

### C ABI Shim Contract
- **D-05:** C ABI status model is `0` on success and nonzero typed failure codes on error.
- **D-06:** Error details are thread-local and retrieved via copy-out APIs (caller-owned buffers), not global state.
- **D-07:** Phase 3 C ABI surface is a thin compat-style wrapper layer for migration parity, not a separate opaque-handle API.
- **D-08:** Failures are fail-closed with no partial writes; status + thread-local error report are the only failure outputs.

### Optional Family Gating
- **D-09:** Optional-family behavior is enforced by both compile-time features and runtime envelope validation.
- **D-10:** `with-f12` enables only the validated sph envelope; out-of-envelope requests fail with explicit `UnsupportedApi` reason text.
- **D-11:** `with-4c1e` is strict-envelope only; requests outside validated bounds are explicitly rejected.
- **D-12:** Manifest/resolver metadata is the single source of truth for optional-family support decisions.

### Unstable Source API Boundary
- **D-13:** Source-only APIs live in explicitly unstable namespaces when `unstable-source-api` is enabled; stable namespaces remain unchanged.
- **D-14:** C ABI remains stable-surface only in Phase 3 (no unstable source-only C exports yet).
- **D-15:** Promotion from unstable to stable requires manifest/oracle/release-gate evidence plus explicit maintainer decision.
- **D-16:** When `unstable-source-api` is disabled, unstable symbols are not compiled; indirect requests fail explicitly with `UnsupportedApi`.

### Carried Forward from Prior Phases
- **D-17:** Preserve the Phase 1 split between `query_workspace()` and `evaluate()` (already locked in 01-CONTEXT and ROADMAP Phase 3 criteria).
- **D-18:** Preserve Phase 2 fail-closed execution/no-partial-write behavior and backend-neutral runtime ownership contract.

### the agent's Discretion
- Exact Rust type names, module layout, and builder ergonomics inside `cintx-rs`.
- Concrete integer code assignments for C ABI status taxonomy.
- Exact `last_error` struct fields/string format and copy helper naming.
- How plan tasks partition work across `cintx-rs`, `cintx-capi`, `cintx-compat`, and tests/docs.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Scope and Requirements
- `.planning/ROADMAP.md` — Phase 3 goal, dependencies, requirement IDs, and success criteria.
- `.planning/REQUIREMENTS.md` — `EXEC-01`, `COMP-04`, `OPT-01`, `OPT-02`, `OPT-03` requirement contracts.
- `.planning/PROJECT.md` — three-layer API ordering and compatibility-first constraints.
- `.planning/STATE.md` — continuity from completed Phase 2 and current milestone position.

### Design Authority
- `docs/design/cintx_detailed_design.md` §1.5, §1.6 — compatibility policy and Rust-native API design constraints.
- `docs/design/cintx_detailed_design.md` §3.2, §3.3, §3.6.1 — manifest authority, support matrix, and `dims` contract boundaries.
- `docs/design/cintx_detailed_design.md` §5.4 — safe API contract (`query_workspace`/`evaluate` split).
- `docs/design/cintx_detailed_design.md` §5.6 — C ABI shim principles.
- `docs/design/cintx_detailed_design.md` §10.2 — feature flags (`with-f12`, `with-4c1e`, `unstable-source-api`).
- `docs/design/cintx_detailed_design.md` §11.4 — thread-local C ABI error report expectations.
- `docs/design/cintx_detailed_design.md` §14.1 — release gates for optional/unstable promotion.

### Existing Code Contracts
- `crates/cintx-runtime/src/planner.rs` — existing `query_workspace`/`evaluate` contract and fail-closed execution checks.
- `crates/cintx-runtime/src/lib.rs` — runtime public exports consumed by upper layers.
- `crates/cintx-compat/src/raw.rs` — compat raw API shape and current validation/error paths.
- `crates/cintx-core/src/error.rs` — canonical typed error categories to map into facade/C ABI surfaces.
- `crates/cintx-ops/generated/compiled_manifest.lock.json` — support-matrix source for optional-family exposure.
- `crates/cintx-rs/src/api.rs` — safe facade landing zone (currently stub).
- `crates/cintx-capi/src/shim.rs` — C ABI shim landing zone (currently stub).
- `crates/cintx-capi/src/errors.rs` — C ABI error-report landing zone (currently stub).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `cintx-runtime` already provides `query_workspace()` and `evaluate()` separation with explicit planning-match validation.
- `cintx-compat` already encodes raw libcint layout handling, resolver wiring, and fail-closed semantics.
- `cintx-core` already defines typed error variants suitable for stable mapping into higher layers.
- Manifest metadata already distinguishes feature profiles and optional families.

### Established Patterns
- Compatibility enforcement is manifest-driven (`Resolver`, descriptor metadata) rather than hardcoded symbol lists.
- Runtime policy is deterministic and explicit about unsupported APIs and memory/ownership violations.
- Public-facing errors are typed; silent truncation/partial writes are explicitly rejected.

### Integration Points
- `cintx-rs` should wrap `cintx-runtime` contracts without reintroducing sentinel/raw pointer patterns.
- `cintx-capi` should wrap compat/runtime behavior and layer C status + TLS last-error reporting.
- Optional-family and unstable exposure should be driven by manifest features plus Cargo feature gates.

</code_context>

<specifics>
## Specific Ideas

- No external product/style references were requested.
- Decisions prioritize explicit, auditable contracts over convenience fallbacks.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within Phase 3 scope.

</deferred>

---

*Phase: 03-safe-surface-c-abi-shim-optional-families*
*Context gathered: 2026-03-28*
