# Phase 3: Safe Surface, C ABI Shim & Optional Families - Research

**Researched:** 2026-03-28  
**Domain:** Safe Rust facade, C ABI shim, and optional-family feature/runtime gating  
**Confidence:** MEDIUM

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

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

### Claude's Discretion
- Exact Rust type names, module layout, and builder ergonomics inside `cintx-rs`.
- Concrete integer code assignments for C ABI status taxonomy.
- Exact `last_error` struct fields/string format and copy helper naming.
- How plan tasks partition work across `cintx-rs`, `cintx-capi`, `cintx-compat`, and tests/docs.

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within Phase 3 scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EXEC-01 | Rust caller can query workspace needs separately from evaluation through the safe API. | `cintx-runtime::query_workspace` + `ExecutionPlan::new` contract reuse; typed session pattern in Architecture Patterns. |
| COMP-04 | C integrator can enable an optional C ABI shim that returns integer status codes and exposes thread-local last-error details. | C ABI status/TLS pattern, panic/FFI boundary guidance, and copy-out error API recommendations. |
| OPT-01 | Caller can enable sph-only F12, STG, and YP families behind `with-f12`, and unsupported representations fail explicitly. | Dual gating pattern (compile feature + runtime envelope) + manifest-driven availability checks. |
| OPT-02 | Caller can enable 4c1e behind `with-4c1e` only within the validated bug envelope, and out-of-envelope cases fail explicitly. | Runtime envelope validator pattern + explicit `UnsupportedApi` failure path requirements. |
| OPT-03 | Maintainer can expose approved source-only families behind `unstable-source-api` without changing the stable GA surface. | Unstable namespace boundary pattern and manifest/feature contract checks. |
</phase_requirements>

## Summary

Phase 3 should be implemented as a strict layering phase, not a new execution engine phase. The safest path is to keep execution behavior in `cintx-runtime`/`cintx-compat` and add two thin surfaces above it: a typed safe facade in `cintx-rs` and a status-code C shim in `cintx-capi`.

Current code evidence shows the runtime/compat contract is already strong: `query_workspace()` and `evaluate()` are split, `ExecutionPlan::new` rejects query/evaluate drift, and failure paths are fail-closed with no partial writes. The Phase 3 risk is mostly surface contract drift: accidentally reintroducing raw/dims semantics into the safe API, global error state in C ABI, or optional-family checks that bypass manifest authority.

Optional-family readiness is partially present but incomplete. The manifest lock has `4c1e` operator entries (cart/sph) and no current F12/STG/YP or source-only entries. Phase planning must therefore include explicit manifest/profile and resolver updates before safe/capi exposure tasks, not after.

**Primary recommendation:** Plan Phase 3 as three ordered waves: (1) feature/manifest gating foundations, (2) safe facade/session API, (3) C ABI status+TLS shim, with compatibility/error contract tests at each wave boundary.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cintx-runtime` | workspace `0.1.0` | Canonical `query_workspace` + `evaluate` execution contract | Already enforces query/evaluate token consistency and fail-closed behavior. |
| `cintx-compat` | workspace `0.1.0` | Raw compat behavior and output/dims/cache contract | Already contains validated raw layout conversion (`CompatDims`) and no-partial-write semantics. |
| `cintx-ops` | workspace `0.1.0` | Manifest/resolver authority for symbol/family support | Existing resolver and generated manifest are the phase-mandated source of truth. |
| `cintx-core` + `thiserror` | `thiserror` `2.0.18` | Stable typed error categories for safe facade and C status mapping | Aligns with project error policy and existing `cintxRsError` categories. |
| `std::thread_local!` | std `1.94.x` docs | Per-thread C ABI error storage | Matches thread-local `last_error` decision and avoids global mutable state races. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `tracing` | `0.1.44` | Spans for query/evaluate/shim diagnostics | Keep parity with existing runtime diagnostics and failure telemetry. |
| `smallvec` | `1.15.1` | Compact dims/layout containers | Reuse compat layout semantics without new allocation-heavy paths. |
| `libcint` | `0.2.2` | Upstream feature basis (`with_f12`, `with_4c1e`) | Needed to map workspace feature flags to upstream build flags correctly. |
| `std::panic::catch_unwind` | std `1.94.1` docs | Convert panics at ABI boundary to status+last_error | Use in C ABI wrappers when graceful failure reporting is desired. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Thin C ABI over compat/runtime | Opaque handle-first C API redesign | Breaks locked migration-focused parity scope (D-07). |
| Manifest-driven optional gating | Hardcoded symbol allowlists | Drifts from compiled lock authority and increases maintenance risk. |
| Thread-local `last_error` | Process-global static error buffer | Racy under multithreaded C callers and violates D-06. |

**Version verification (2026-03-28):**
- `anyhow` `1.0.102` (published 2026-02-20)
- `thiserror` `2.0.18` (published 2026-01-18)
- `tracing` `0.1.44` (published 2025-12-18)
- `smallvec` latest stable `1.15.1` (2.0 line is pre-release)
- `cubecl` latest stable `0.9.0` (0.10 line is pre-release)
- `libcint` `0.2.2` with features including `with_f12` and `with_4c1e`

## Architecture Patterns

### Recommended Project Structure
```
crates/
├── cintx-rs/
│   └── src/
│       ├── api.rs        # Safe session/query/evaluate facade
│       ├── builder.rs    # Typed request/session builders
│       └── prelude.rs    # Stable re-exports only
├── cintx-capi/
│   └── src/
│       ├── shim.rs       # extern "C" wrappers returning status
│       ├── errors.rs     # TLS last_error state + copy-out helpers
│       └── lib.rs        # feature-gated export surface
└── shared existing crates (`core`, `ops`, `runtime`, `compat`) remain execution authority
```

### Pattern 1: Typed Safe Session Tied to Query Token
**What:** Build a facade-level session/request object that stores the validated workspace query and prevents evaluate-time drift.
**When to use:** Every safe API call path implementing `EXEC-01`.
**Example:**
```rust
// Source: crates/cintx-runtime/src/planner.rs + Phase 3 decisions D-01..D-04
pub struct SafeSession {
    op: OperatorId,
    rep: Representation,
    basis: BasisSet,
    shells: ShellTuple,
    query: WorkspaceQuery,
}

impl SafeSession {
    pub fn query_workspace(...) -> Result<Self, FacadeError> { /* wrap runtime::query_workspace */ }
    pub fn evaluate(self, opts: &ExecutionOptions) -> Result<OwnedTensor<f64>, FacadeError> {
        // Build ExecutionPlan from stored query; never accept ad-hoc dims/out buffers.
        /* ... */
    }
}
```

### Pattern 2: C ABI Status + Thread-Local Error Report
**What:** Wrap compat/safe calls in `extern "C"` fns that return `0` or nonzero status, while writing structured error details into TLS.
**When to use:** All `cintx-capi` exported functions for `COMP-04`.
**Example:**
```rust
// Source: https://doc.rust-lang.org/std/macro.thread_local.html
// Source: https://doc.rust-lang.org/std/panic/fn.catch_unwind.html
thread_local! {
    static LAST_ERROR: std::cell::RefCell<Option<CapiErrorReport>> = const { std::cell::RefCell::new(None) };
}

#[no_mangle]
pub unsafe extern "C" fn cintx_rs_eval_raw(...) -> i32 {
    let result = std::panic::catch_unwind(|| compat_eval_impl(...));
    match result {
        Ok(Ok(_)) => 0,
        Ok(Err(err)) => { store_tls_error(err); map_error_to_status(err) }
        Err(_) => { store_tls_panic(); STATUS_PANIC }
    }
}
```

### Pattern 3: Dual Gating for Optional Families
**What:** Require both compile-time feature activation and runtime envelope validation against manifest metadata.
**When to use:** `with-f12`, `with-4c1e`, `unstable-source-api` pathways.
**Example:**
```rust
// Source: crates/cintx-ops/src/resolver.rs + docs/design/cintx_detailed_design.md §3.11
fn ensure_optional_allowed(desc: &OperatorDescriptor, req: &RequestMeta) -> Result<(), cintxRsError> {
    match desc.entry.canonical_family {
        "4c1e" => validate_4c1e_envelope(req), // cart/sph, scalar, l<=4, etc.
        _ => Ok(()),
    }?;

    if !desc.entry.compiled_in_profiles.contains(&req.profile_name) {
        return Err(cintxRsError::UnsupportedApi { requested: req.symbol.clone() });
    }
    Ok(())
}
```

### Anti-Patterns to Avoid
- **Reintroducing safe-path raw semantics:** Do not expose `dims`, raw output buffers, or nullable sentinel contracts in `cintx-rs`.
- **Global mutable C error singleton:** Use TLS, not shared statics.
- **Feature-only gating without runtime envelope checks:** Cargo feature gates alone do not satisfy OPT-01/OPT-02.
- **Bypassing manifest/resolver:** Never hardcode optional support tables in facade/shim code.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Workspace/output planning contract | New ad-hoc planner in `cintx-rs` | `cintx-runtime::query_workspace` + `ExecutionPlan::new` | Existing contract already enforces no drift and fail-closed semantics. |
| Compat/C layout math | Custom dims/stride formulas in C shim | `cintx-compat::CompatDims` and `required_elems_from_dims` | Existing logic already validates arity and rejects partial writes. |
| Optional support matrix | Manual symbol allowlist | `cintx-ops` generated manifest + resolver | Manifest lock is canonical and already profile-scoped. |
| C error transport | Heap/global C string cache | TLS report + copy-out APIs | Avoids cross-thread corruption and ownership leaks. |

**Key insight:** Phase 3 should compose existing verified contracts, not duplicate them behind new public surfaces.

## Common Pitfalls

### Pitfall 1: Feature Name Drift Across Layers
**What goes wrong:** Cargo features in Rust code (`with-f12`) and upstream crate features (`with_f12`) are mixed incorrectly.
**Why it happens:** Hyphen/underscore naming differs between workspace gating and dependency feature names.
**How to avoid:** Define explicit mapping in one place (workspace features -> upstream features), and test each profile.
**Warning signs:** Feature compiles but manifest/profile checks or resolver gating behave unexpectedly.

### Pitfall 2: Panic Behavior at `extern "C"` Boundary
**What goes wrong:** Panics escape ABI wrappers or are reported inconsistently.
**Why it happens:** Missing boundary handling in shim wrappers.
**How to avoid:** Wrap shim internals with `catch_unwind`, map to typed status, and write TLS error reports.
**Warning signs:** Process aborts in C integration tests without TLS error context.

### Pitfall 3: Partial Writes on Error Paths
**What goes wrong:** Output buffers are mutated before full validation/allocation succeeds.
**Why it happens:** Copy/write performed before all contract checks pass.
**How to avoid:** Keep backend staging + final compat write ordering; only write caller outputs after successful execution.
**Warning signs:** Output buffer changes after `MemoryLimitExceeded`/`UnsupportedApi`.

### Pitfall 4: Assuming Optional Families Already Exist in Manifest
**What goes wrong:** Planning skips manifest generation updates and implements facade/shim APIs that cannot resolve symbols.
**Why it happens:** Current lock includes `4c1e` but no F12/STG/YP or source-only entries.
**How to avoid:** Add explicit manifest/profile tasks first, then expose APIs.
**Warning signs:** Resolver missing-symbol errors for optional targets in enabled builds.

## Code Examples

Verified patterns from existing code and official references:

### Query/Evaluate Contract Reuse
```rust
// Source: crates/cintx-runtime/src/planner.rs
let query = query_workspace(op, rep, &basis, shells.clone(), &opts)?;
let plan = ExecutionPlan::new(op, rep, &basis, shells, &query)?;
let stats = evaluate(plan, &opts, &mut allocator, &executor)?;
```

### Fail-Closed Optional Rejection
```rust
// Source: crates/cintx-cubecl/src/executor.rs
if canonical_family == "4c1e" {
    return Err(cintxRsError::UnsupportedApi {
        requested: "4c1e remains out of scope for Phase 2 CubeCL executor".to_owned(),
    });
}
```

### TLS Error Slot Declaration
```rust
// Source: https://doc.rust-lang.org/std/macro.thread_local.html
thread_local! {
    static LAST_ERROR: std::cell::RefCell<Option<CapiErrorReport>> = const {
        std::cell::RefCell::new(None)
    };
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Raw-only externally visible contract | Safe facade + raw compat + optional C ABI layering | Locked in design and roadmap for Phase 3 | Keeps migration parity while improving Rust ergonomics. |
| Feature gates treated as build-only toggles | Build-time gates + runtime envelope validation | Design finalized for optional families | Prevents invalid optional calls from reaching backend execution. |
| 4c1e treated as entirely unavailable in executor | 4c1e gated for controlled Phase 3 envelope | Phase 3 requirement OPT-02 | Enables incremental support without broad correctness risk. |

**Deprecated/outdated:**
- Global singleton last-error model for C callers.
- Hardcoded optional symbol tables disconnected from manifest lock.
- Safe API designs that expose compat-only `dims` behavior.

## Open Questions

1. **What exact numeric status taxonomy should map `cintxRsError` to C?**
   - What we know: Status must be `0` success + nonzero typed failures (D-05), with TLS details (D-06).
   - What's unclear: Final integer code assignments and ABI stability policy.
   - Recommendation: Reserve stable code ranges per error class and freeze in `cintx-capi` docs/tests before implementation.

2. **How should `with-f12`/`unstable-source-api` be populated in the current manifest lock?**
   - What we know: Current lock has no F12/STG/YP symbols and no source-only entries; all feature flags are currently `"none"`.
   - What's unclear: Whether upstream discovery/generation currently excludes these, or they are intentionally deferred.
   - Recommendation: Add a dedicated pre-wave task to regenerate/validate manifest entries under profile builds before facade/shim exposure.

3. **Should Phase 3 add `cintx-rs` and `cintx-capi` to workspace members immediately or per-wave?**
   - What we know: Root workspace members currently exclude both crates.
   - What's unclear: Whether maintainers want staged visibility or immediate full-workspace compilation.
   - Recommendation: Add both crates early with minimal compile targets to catch cross-crate feature drift continuously.

## Sources

### Primary (HIGH confidence)
- `docs/design/cintx_detailed_design.md` - Safe API signatures, C ABI principles, optional-family and unstable boundary rules.
- `crates/cintx-runtime/src/planner.rs` - Existing query/evaluate token contract and fail-closed runtime behavior.
- `crates/cintx-compat/src/raw.rs` and `crates/cintx-compat/src/layout.rs` - Raw contract, dims validation, no-partial-write path.
- `crates/cintx-ops/generated/compiled_manifest.lock.json` - Current profile/family inventory and optional coverage reality.
- `crates/cintx-ops/src/resolver.rs` and `crates/cintx-ops/build.rs` - Manifest feature/profile mapping and resolver authority.
- https://doc.rust-lang.org/cargo/reference/features.html - Feature naming and resolver behavior.
- https://doc.rust-lang.org/std/macro.thread_local.html - TLS key pattern for thread-local error state.
- https://doc.rust-lang.org/std/panic/fn.catch_unwind.html - Panic boundary handling guidance for foreign callers.
- https://doc.rust-lang.org/reference/items/external-blocks.html - Explicit ABI declaration guidance for `extern`.

### Secondary (MEDIUM confidence)
- https://crates.io/api/v1/crates/anyhow
- https://crates.io/api/v1/crates/thiserror
- https://crates.io/api/v1/crates/tracing
- https://crates.io/api/v1/crates/smallvec
- https://crates.io/api/v1/crates/cubecl
- https://crates.io/api/v1/crates/libcint

### Tertiary (LOW confidence)
- None.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Based on current workspace crates plus verified registry metadata.
- Architecture: HIGH - Directly grounded in locked Phase 3 decisions and existing runtime/compat contracts.
- Pitfalls: MEDIUM - Strong local evidence, but envelope details for F12/source-only still require manifest-generation confirmation.

**Research date:** 2026-03-28  
**Valid until:** 2026-04-27
