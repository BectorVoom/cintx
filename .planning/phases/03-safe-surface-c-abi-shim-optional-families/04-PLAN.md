---
phase: 03-safe-surface-c-abi-shim-optional-families
plan: 04
type: execute
wave: 3
depends_on:
  - 01
  - 02
files_modified:
  - crates/cintx-capi/src/lib.rs
  - crates/cintx-capi/src/errors.rs
  - crates/cintx-capi/src/shim.rs
autonomous: true
requirements:
  - COMP-04
must_haves:
  truths:
    - "C callers can invoke optional shim entry points and receive `0` on success or nonzero typed status codes on failure."
    - "Error details are stored thread-locally and retrievable through copy-out APIs without shared global mutable error state."
    - "C shim functions are thin compat-style wrappers that preserve raw compat contracts and fail-closed no-partial-write behavior."
    - "The C ABI surface remains stable-only in Phase 3 and does not expose unstable source-only symbols."
  artifacts:
    - path: crates/cintx-capi/src/errors.rs
      provides: "Status-code taxonomy, TLS error report storage, and copy-out accessor APIs."
      min_lines: 150
    - path: crates/cintx-capi/src/shim.rs
      provides: "extern \"C\" compat-style wrappers for query/eval and error mapping."
      min_lines: 220
    - path: crates/cintx-capi/src/lib.rs
      provides: "Public C ABI export surface with stable-only module boundaries."
      min_lines: 30
  key_links:
    - from: crates/cintx-capi/src/shim.rs
      to: crates/cintx-compat/src/raw.rs
      via: "C shim wrappers call compat raw APIs directly, preserving migration-focused parity and avoiding a parallel implementation."
      pattern: "query_workspace_raw|eval_raw"
    - from: crates/cintx-capi/src/shim.rs
      to: crates/cintx-capi/src/errors.rs
      via: "All failures map to typed status codes and write thread-local reports used by copy-out accessors."
      pattern: "set_last_error|status"
    - from: crates/cintx-capi/src/lib.rs
      to: crates/cintx-capi/src/shim.rs
      via: "Stable-only exports include shim/error modules while excluding unstable-source C symbols."
      pattern: "pub mod errors|pub mod shim"
---

<objective>
Implement the optional C ABI shim with integer status returns and thread-local last-error reporting over compat raw paths.
Purpose: Provide migration/interoperability support for C callers without weakening typed failure semantics or memory-safety guarantees.
Output: C ABI status/error subsystem, compat-style extern wrappers, and focused shim tests.
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
@.planning/phases/03-safe-surface-c-abi-shim-optional-families/03-CONTEXT.md
@.planning/phases/03-safe-surface-c-abi-shim-optional-families/03-RESEARCH.md
@AGENTS.md
@docs/design/cintx_detailed_design.md §5.6, §11.4
@crates/cintx-compat/src/raw.rs
@crates/cintx-core/src/error.rs
@crates/cintx-capi/src/lib.rs
@crates/cintx-capi/src/errors.rs
@crates/cintx-capi/src/shim.rs
<interfaces>
From `docs/design/cintx_detailed_design.md` §11.4:
```c
int cintrs_last_error_code(void);
const char* cintrs_last_error_message(void);
void cintrs_clear_last_error(void);
```

From `crates/cintx-compat/src/raw.rs`:
```rust
pub unsafe fn query_workspace_raw(...) -> Result<WorkspaceQuery, cintxRsError>;
pub unsafe fn eval_raw(...) -> Result<RawEvalSummary, cintxRsError>;
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Build C status taxonomy and thread-local last-error copy-out APIs</name>
  <files>crates/cintx-capi/src/errors.rs, crates/cintx-capi/src/lib.rs</files>
  <read_first>crates/cintx-capi/src/errors.rs, crates/cintx-capi/src/lib.rs, crates/cintx-core/src/error.rs, docs/design/cintx_detailed_design.md §11.1, §11.4, .planning/phases/03-safe-surface-c-abi-shim-optional-families/03-CONTEXT.md</read_first>
  <action>
Implement the C ABI error subsystem in `errors.rs` with a stable integer status taxonomy where success is `0` and all failures are nonzero typed codes (per D-05). Add thread-local storage for the last error report and provide copy-out accessor functions (code + message + clear/reset) that operate on caller-owned buffers (per D-06). Map `cintxRsError` categories into stable status codes and include enough report fields to support diagnostics (`api/family/representation/message`). Keep the implementation thread-safe by using TLS only; do not introduce a global singleton error slot. Export these APIs through `lib.rs` while keeping C ABI exports stable-only (no unstable-source export additions in this phase, per D-14).
  </action>
  <acceptance_criteria>
    - `rg -n "repr\\(i32\\)|STATUS|Success = 0|= 0" crates/cintx-capi/src/errors.rs`
    - `rg -n "thread_local!|LAST_ERROR|RefCell" crates/cintx-capi/src/errors.rs`
    - `rg -n "last_error|clear|copy" crates/cintx-capi/src/errors.rs`
    - `rg -n "pub mod errors|pub mod shim" crates/cintx-capi/src/lib.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-capi --lib errors::tests:: -- --nocapture</automated>
  </verify>
  <done>C ABI now has deterministic status codes and thread-local copy-out error reporting suitable for multi-threaded C callers.</done>
</task>

<task type="auto">
  <name>Task 2: Implement thin compat-style extern wrappers with fail-closed status + TLS reporting</name>
  <files>crates/cintx-capi/src/shim.rs, crates/cintx-capi/src/lib.rs</files>
  <read_first>crates/cintx-capi/src/shim.rs, crates/cintx-capi/src/lib.rs, crates/cintx-capi/src/errors.rs, crates/cintx-compat/src/raw.rs, docs/design/cintx_detailed_design.md §5.6, docs/rust_crate_test_guideline.md</read_first>
  <action>
Implement `extern "C"` shim entry points as thin wrappers over compat raw APIs (per D-07), wrapping each call with panic boundaries and error translation to status codes. On success, return `0` and clear TLS error; on failure or panic, set TLS error details and return nonzero status (per D-05 and D-06). Keep wrapper behavior fail-closed: do not write partial outputs in shim code; delegate all output writes to compat raw paths and return status + TLS error only on failure (per D-08 and D-18). Explicitly keep the C ABI export list stable-only for this phase by avoiding unstable-source wrapper exports (per D-14). Add unit tests covering success, invalid-input failure, panic mapping, and thread-local error isolation.
  </action>
  <acceptance_criteria>
    - `rg -n "extern \"C\"|no_mangle|unsafe" crates/cintx-capi/src/shim.rs`
    - `rg -n "query_workspace_raw|eval_raw" crates/cintx-capi/src/shim.rs`
    - `rg -n "catch_unwind|set_last_error|clear_last_error" crates/cintx-capi/src/shim.rs`
    - `rg -n "status|UnsupportedApi|BufferTooSmall|MemoryLimitExceeded" crates/cintx-capi/src/shim.rs`
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p cintx-capi --lib</automated>
  </verify>
  <done>C callers can invoke shim functions with deterministic integer statuses and TLS diagnostics while preserving compat fail-closed semantics.</done>
</task>

</tasks>

<verification>
Run `cargo test -p cintx-capi --lib` to validate status mapping, TLS behavior, panic handling, and compat-wrapper integration.
</verification>

<success_criteria>
The optional C ABI shim is operational and standards-compliant for migration: integer statuses, thread-local last-error copy-out APIs, and fail-closed wrapper behavior over compat paths.
</success_criteria>

<output>
After completion, create `.planning/phases/03-safe-surface-c-abi-shim-optional-families/04-PLAN-SUMMARY.md`
</output>
