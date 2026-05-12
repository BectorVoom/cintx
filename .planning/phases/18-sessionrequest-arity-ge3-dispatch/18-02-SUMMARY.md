---
phase: 18-sessionrequest-arity-ge3-dispatch
plan: 02
subsystem: cintx-rs
tags: [safe-api, aosym, preflight, f-order-rustdoc, facade-error, operator-id-shift, wave-1]

# Dependency graph
requires:
  - phase: 18-sessionrequest-arity-ge3-dispatch
    plan: 01
    provides: "AoSymmetry enum + ExecutionOptions::aosym field + post-shift OperatorIds; INT4C1E_CART_OPERATOR_ID already at 24 via Plan 18-01 commit f601635"
  - phase: 17-real-integral-evaluation-in-safe-api
    provides: "SessionRequest::query_workspace + IntegralTensor + FacadeError + test-module sample_basis_with_shells helper"
provides:
  - "FacadeError::UnsupportedAoSymmetry { requested: String } variant (#[error('unsupported aosym packing: {requested}')]) — typed, pattern-matchable rejection of non-S1 aosym from SessionRequest::query_workspace"
  - "FacadeErrorKind::UnsupportedAoSymmetry variant + matching FacadeError::kind() arm (additive, ordinals preserved)"
  - "SessionRequest::query_workspace fail-fast aosym preflight — returns FacadeError::UnsupportedAoSymmetry for every non-S1 ExecutionOptions::aosym; runs before any runtime_query_workspace call"
  - "IntegralTensor arity-aware F-order rustdoc — documents F-order for arity >= 3 outputs and row-major for arity-2 outputs (RESEARCH R2 honest wording)"
  - "Two new unit tests in crates/cintx-rs/src/api.rs::tests: aosym_error_path_rejects_non_s1_with_typed_error + aosym_none_and_s1_both_succeed_through_query_workspace (no vendor dep)"
affects:
  - "Plan 18-03 (arity-3 oracle parity) — tests build SessionRequest with default ExecutionOptions; the preflight is a no-op for None/S1 so byte-identity path is unaffected"
  - "Plan 18-04 (arity-4 oracle parity) — same; int4c1e_* tests dispatch via the (already-shifted) OperatorId 24/25"
  - "Downstream consumer pyscf_rs (pyscf-gto/src/intor.rs) — gains the typed FacadeError::UnsupportedAoSymmetry pattern-match contract for the four non-S1 packings"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Fail-fast safe-API preflight using `if let Some(_) =` (never naive .unwrap()) — anti-pattern from RESEARCH §Pitfall 6"
    - "Three-edit error-variant additive-only protocol (variant in enum + variant in kind enum + match arm) in a single commit to avoid non-exhaustive-match compile breakage"
    - "Arity-aware rustdoc honesty — rather than a uniform F-order claim, the docstring distinguishes arity >= 3 (F-order, no transpose) from arity-2 (row-major within shell-pair block, vendor transposed during parity)"

key-files:
  created:
    - ".planning/phases/18-sessionrequest-arity-ge3-dispatch/18-02-SUMMARY.md"
  modified:
    - "crates/cintx-rs/src/error.rs - +1 FacadeErrorKind variant, +1 FacadeError variant, +1 kind() match arm"
    - "crates/cintx-rs/src/api.rs - +9 lines aosym preflight at query_workspace head; +29 lines F-order rustdoc on IntegralTensor; +57 lines for two aosym unit tests"

key-decisions:
  - "Edit C (INT4C1E_CART_OPERATOR_ID 22 -> 24) was already applied in Plan 18-01 wave 0 (commit f601635). Verified via grep -F 'INT4C1E_CART_OPERATOR_ID: u32 = 24' crates/cintx-rs/src/api.rs at line 501 BEFORE Task 2 work began. Task 2 only needs to confirm the constant is at 24 (it is) — no edit applied. This is a benign deviation, documented under Deviations below."
  - "Preflight uses fully-qualified `cintx_core::AoSymmetry::S1` in api.rs body (no file-scope `use`) to keep the rustdoc neighborhood minimal; the test functions use `use cintx_core::AoSymmetry;` inside the function body per the patterns from PATTERNS.md §api.rs Edit C."
  - "Anti-pattern from RESEARCH §Pitfall 6 enforced: `if let Some(aosym) = self.options.aosym` rather than `self.options.aosym.unwrap()` — verified by `! grep -F '.unwrap() != cintx_core::AoSymmetry::S1' crates/cintx-rs/src/api.rs`."

requirements-completed: [ARITY-01, ARITY-03, ARITY-04]

# Metrics
duration: ~12min
completed: 2026-05-12
---

# Phase 18 Plan 02: Safe-API Surface Wiring Summary

**Wave-1 safe-API surface delivery: aosym preflight in `SessionRequest::query_workspace`, arity-aware F-order rustdoc on `IntegralTensor`, and the typed `FacadeError::UnsupportedAoSymmetry { requested: String }` variant (with kind() arm) — plus two vendor-free unit tests that exercise both error and success paths.**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-05-12 (post-Wave-0 merge)
- **Tasks:** 2
- **Files modified:** 2 hand-edited (cintx-rs/src/error.rs, cintx-rs/src/api.rs)

## Accomplishments

### Task 1 — `crates/cintx-rs/src/error.rs` (`a501776`)

- Appended `UnsupportedAoSymmetry` to the `FacadeErrorKind` enum (after `Validation`) — ordinals of existing variants preserved.
- Appended `UnsupportedAoSymmetry { requested: String }` variant to `FacadeError` with `#[error("unsupported aosym packing: {requested}")]` thiserror attribute. Message format follows the existing `{requested}` pattern from `UnsupportedApi`.
- Added the matching `Self::UnsupportedAoSymmetry { .. } => FacadeErrorKind::UnsupportedAoSymmetry` arm to `FacadeError::kind()` to keep the const-fn exhaustive. All three edits in one commit so the file stays compilable.
- `From<cintxRsError>` impl deliberately untouched per `RESEARCH.md §error.rs` — `UnsupportedAoSymmetry` is raised exclusively from `SessionRequest::query_workspace`'s new preflight, never via the runtime-error conversion path.

### Task 2 — `crates/cintx-rs/src/api.rs` (`bcb9325`)

- **Edit A — aosym preflight in `query_workspace`** (lines 63–72 after edit): inserted a fail-fast `if let Some(aosym) = self.options.aosym { if aosym != cintx_core::AoSymmetry::S1 { return Err(FacadeError::UnsupportedAoSymmetry { requested: aosym.to_string() }); } }` block BEFORE the existing `runtime_query_workspace` call. The `aosym.to_string()` uses the `Display` impl added by Plan 18-01 which emits the lowercase pyscf form (`s2ij`, `s2kl`, `s4`, `s8`).
- **Edit B — F-order rustdoc on `IntegralTensor`** (starting at line 441 before edit / line 452 after edit, ~28 lines of rustdoc): prepended a `# Memory layout` rustdoc block honoring `CONTEXT.md D-10` and `RESEARCH.md R2`:
  - Arity ≥ 3 (`int2e_*`, `int3c1e_*`, `int3c2e_*`, `int4c1e_*`) — F-order, byte-identical to vendor libcint without transposition (verified by the Phase 18 oracle parity sweep in Plans 03/04).
  - Arity 2 (`int1e_*`, `int2c2e_*`) — row-major within each shell-pair block; vendor output is transposed during arity-2 parity comparison.
  - `component_axis_leading == true` behavior documented.
  - Cross-reference to the oracle parity sweep file paths as the implicit verifier.
- **Edit C — `INT4C1E_CART_OPERATOR_ID = 24`** — verified-only. Plan 18-01 Wave 0 (`f601635`) had already propagated the +2 shift to all three hard-coded constants in `crates/cintx-rs/src/api.rs`, including `INT4C1E_CART_OPERATOR_ID: u32 = 24` at line 501. Pre-Task-2 grep confirmed the constant is at 24 and the old value 22 is absent. No edit applied.
- **Edit D — two aosym unit tests** appended to `#[cfg(test)] mod tests` (after `unsupported_unstable_requests_map_to_unsupported_api`):
  - `aosym_error_path_rejects_non_s1_with_typed_error` — exercises all four non-S1 variants (`S2ij`, `S2kl`, `S4`, `S8`) with `OperatorId::new(0)` (`int1e_ovlp_cart` — any valid operator works because preflight runs before operator-specific routing). Asserts `FacadeError::UnsupportedAoSymmetry { requested }` carries the lowercase pyscf form of each variant (verified via `non_s1.to_string()`).
  - `aosym_none_and_s1_both_succeed_through_query_workspace` — asserts both `None` and `Some(AoSymmetry::S1)` reach `query_workspace`'s normal return path. No vendor libcint dependency in either test.

## Insertion Locations (per `<output>` instruction)

Per the plan's `<output>` spec: Plans 03/04 can cross-reference these exact post-Task-2 line numbers in `crates/cintx-rs/src/api.rs`:

| Element | Line | Anchor |
|---------|------|--------|
| aosym preflight block | starts at line 64 | `// Phase 18 D-04: aosym preflight — only S1 (and None ≡ S1) is implemented.` |
| `IntegralTensor` rustdoc (`# Memory layout`) | starts at line 452 | `/// Owned integral tensor returned by \`SessionQuery::evaluate\`.` |

## Task Commits

Each task was committed atomically on `worktree-agent-a9f5600ad5e4eb513`:

1. **Task 1: FacadeError::UnsupportedAoSymmetry variant + FacadeErrorKind variant + kind() arm** — `a501776` (feat)
2. **Task 2: aosym preflight + F-order rustdoc + aosym unit tests in api.rs** — `bcb9325` (feat)

## Verification Output

### Feature-matrix builds (all exit 0)

```
$ cargo build -p cintx-rs --locked                              # default
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.12s
$ cargo build -p cintx-rs --features with-f12 --locked
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.13s
$ cargo build -p cintx-rs --features with-4c1e --locked
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.12s
$ cargo build -p cintx-rs --features unstable-source-api --locked
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.11s
```

### Test results (13 tests, all green)

```
$ cargo test -p cintx-rs --locked
running 13 tests
test api::tests::aosym_error_path_rejects_non_s1_with_typed_error ... ok
test api::tests::aosym_none_and_s1_both_succeed_through_query_workspace ... ok
test api::tests::unsupported_unstable_requests_map_to_unsupported_api ... ok
test api::tests::query_workspace_returns_structured_contract_metadata ... ok
test api::tests::evaluate_rejects_source_only_symbols_via_compat_policy_gate ... ok
test api::tests::query_evaluate_contract_drift_is_detected_before_execution ... ok
test api::tests::evaluate_returns_deterministic_nonzero_real_values ... ok
test builder::tests::builder_f12_zeta_propagates_into_options ... ok
test builder::tests::builder_clear_helpers_remove_optional_overrides ... ok
test builder::tests::builder_from_request_rebuilds_without_mutating_original_contract ... ok
test builder::tests::builder_propagates_option_composition_into_request ... ok
test error::tests::invalid_dims_maps_to_layout_kind ... ok
test error::tests::memory_limit_maps_to_memory_kind ... ok
test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

### Targeted aosym test execution

```
$ cargo test -p cintx-rs --locked api::tests::aosym_error_path_rejects_non_s1_with_typed_error -- --exact
test api::tests::aosym_error_path_rejects_non_s1_with_typed_error ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out

$ cargo test -p cintx-rs --locked api::tests::aosym_none_and_s1_both_succeed_through_query_workspace -- --exact
test api::tests::aosym_none_and_s1_both_succeed_through_query_workspace ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out
```

(Note: the plan's verification block uses unqualified test names `aosym_error_path_rejects_non_s1_with_typed_error`; cargo's `--exact` requires the fully qualified path `api::tests::...`. The intent — each test runs in isolation and passes — is satisfied above.)

### Acceptance greps

```
$ grep -F 'UnsupportedAoSymmetry { requested: String }' crates/cintx-rs/src/error.rs
    UnsupportedAoSymmetry { requested: String },
$ grep -F '"unsupported aosym packing: {requested}"' crates/cintx-rs/src/error.rs
    #[error("unsupported aosym packing: {requested}")]
$ grep -F 'UnsupportedAoSymmetry,' crates/cintx-rs/src/error.rs
    UnsupportedAoSymmetry,
$ grep -F 'Self::UnsupportedAoSymmetry { .. } => FacadeErrorKind::UnsupportedAoSymmetry' crates/cintx-rs/src/error.rs
            Self::UnsupportedAoSymmetry { .. } => FacadeErrorKind::UnsupportedAoSymmetry,
$ grep -v '^//' crates/cintx-rs/src/error.rs | grep -cE 'UnsupportedAoSymmetry'
3

$ grep -F 'return Err(FacadeError::UnsupportedAoSymmetry' crates/cintx-rs/src/api.rs
                return Err(FacadeError::UnsupportedAoSymmetry {
$ grep -F 'cintx_core::AoSymmetry::S1' crates/cintx-rs/src/api.rs
            if aosym != cintx_core::AoSymmetry::S1 {
$ grep -F 'if let Some(aosym) = self.options.aosym' crates/cintx-rs/src/api.rs
        if let Some(aosym) = self.options.aosym {
$ grep -F '# Memory layout' crates/cintx-rs/src/api.rs
/// # Memory layout
$ grep -F 'INT4C1E_CART_OPERATOR_ID: u32 = 24' crates/cintx-rs/src/api.rs
    const INT4C1E_CART_OPERATOR_ID: u32 = 24;
$ ! grep -F 'INT4C1E_CART_OPERATOR_ID: u32 = 22' crates/cintx-rs/src/api.rs
(no match — old value absent)
$ grep -F 'fn aosym_error_path_rejects_non_s1_with_typed_error' crates/cintx-rs/src/api.rs
    fn aosym_error_path_rejects_non_s1_with_typed_error() {
$ grep -F 'fn aosym_none_and_s1_both_succeed_through_query_workspace' crates/cintx-rs/src/api.rs
    fn aosym_none_and_s1_both_succeed_through_query_workspace() {
```

### Anti-pattern negative greps (must NOT match)

```
$ ! grep -F 'has_vendor_libcint' crates/cintx-rs/src/api.rs    # (no vendor cfg crept in)
$ ! grep -F '.unwrap() != cintx_core::AoSymmetry::S1' crates/cintx-rs/src/api.rs    # (no naive unwrap)
```

### Additive-only public-API check (staged-diff grep)

```
$ git diff --staged -- crates/cintx-rs/src/api.rs crates/cintx-rs/src/error.rs | grep -E '^\-pub '
(no output — no `pub` items removed or renamed)
$ git diff --staged -- crates/cintx-rs/src/error.rs | grep -E '^\-\s+UnsupportedApi|^\-\s+Layout|^\-\s+Memory|^\-\s+Validation'
(no output — no existing variants reordered or removed)
$ git diff --staged -- crates/cintx-rs/src/api.rs | grep -cE '^\+\s*#\[test\]'
2    # (exactly the two new aosym tests)
```

## Files Created/Modified

**Created:**
- `.planning/phases/18-sessionrequest-arity-ge3-dispatch/18-02-SUMMARY.md` (this file)

**Modified (2 files):**
- `crates/cintx-rs/src/error.rs` — +4 lines (1 FacadeErrorKind variant + 2 FacadeError variant lines + 1 kind() arm).
- `crates/cintx-rs/src/api.rs` — +95 lines net (9 preflight body, 29 IntegralTensor rustdoc, 57 two new tests; INT4C1E_CART_OPERATOR_ID already at 24 from Plan 18-01).

## Decisions Made

- **Followed `CONTEXT.md` D-04, D-05, D-10, D-11 verbatim.** Preflight is a sibling to the existing `enforce_safe_facade_policy_gate`, not a replacement. F-order rustdoc lives on the struct only (no module preamble cross-reference per Claude's discretion default).
- **Followed `PATTERNS.md` §api.rs Edit A, Edit B, Edit D verbatim.** Edit C reduced to a verification-only step because Plan 18-01 had already applied the +2 shift to `INT4C1E_CART_OPERATOR_ID`.
- **Followed `PATTERNS.md` §error.rs all three sub-edits verbatim.** Three additive changes in one commit to avoid the non-exhaustive-match compile break (RESEARCH Pitfall 1 mitigated).
- **Did NOT add `INT4C1E_SPH_OPERATOR_ID` sibling constant.** PATTERNS.md §Step 2 explicit guidance: "DO NOT add unless one already exists" — verified via `grep -F 'INT4C1E_SPH_OPERATOR_ID' crates/cintx-rs/src/api.rs` returned no matches before Task 2, so left absent.
- **Test bodies use `OperatorId::new(0)` (int1e_ovlp_cart) as the dispatch target.** Per CONTEXT.md D-05 + PATTERNS.md §api.rs Edit C: any valid arity-2 operator works because the preflight runs BEFORE any operator-specific routing — the test is exercising the new preflight path, not a kernel.
- **No vendor libcint dependency in the new tests.** Per CONTEXT.md D-05 (`pure unit tests, no has_vendor_libcint cfg`) and verified via `! grep -F 'has_vendor_libcint' crates/cintx-rs/src/api.rs`.
- **`From<cintxRsError>` impl untouched.** Per `RESEARCH.md §error.rs` — `UnsupportedAoSymmetry` is raised exclusively from `SessionRequest::query_workspace`'s new preflight, not from any runtime-error conversion site.

## Deviations from Plan

### Edit C reduced to verification-only (benign — pre-condition already satisfied by Plan 18-01)

**1. [Benign — pre-condition satisfied] Edit C of Task 2 (`INT4C1E_CART_OPERATOR_ID: u32 = 22 → 24`) required no edit**

- **Found during:** Task 2 pre-implementation read of `crates/cintx-rs/src/api.rs`.
- **Issue:** The plan's Task 2 Edit C states "Bump `INT4C1E_CART_OPERATOR_ID` constant from `22` to `24`". Pre-edit `grep -n 'INT4C1E_CART_OPERATOR_ID' crates/cintx-rs/src/api.rs` returned `501: const INT4C1E_CART_OPERATOR_ID: u32 = 24;` — the constant was already at 24, having been updated by Plan 18-01 Wave 0 (commit `f601635`, Task 2 Deviation 2). The orchestrator's spawn context explicitly flagged this: "Wave 0 also touched two additional OperatorId constants ... already updated by 18-01. Your job is just the `INT4C1E_CART_OPERATOR_ID: u32 = 22 → 24` shift from your plan." The wording is ambiguous; the actual constant was *already* shifted, so the edit reduced to a verification-only no-op.
- **Fix applied:** Verified the constant is at 24 via `grep -F 'INT4C1E_CART_OPERATOR_ID: u32 = 24'` (matched) and `! grep -F 'INT4C1E_CART_OPERATOR_ID: u32 = 22'` (no match). All Task 2 acceptance criteria for Edit C pass without an edit. The Task 2 commit body documents this explicitly.
- **Files modified:** none (verification-only).
- **Impact on plan:** Zero functional impact. All `<verify>` and `<acceptance_criteria>` greps depend on the post-state (`= 24`), which was already true. The plan's success criterion "ARITY-01: ... constant bump is the ARITY-01 dispatch-routing fixup" is satisfied because the post-state is what was wanted; the work was done in the wave-0 commit rather than this plan's commit.

**Auto-fixed Issues:** None.

**Auth gates:** None — purely internal Rust library changes.

**Total deviations:** 1 benign (no work needed — pre-condition satisfied by upstream plan).

## Issues Encountered

- The plan's `<verify>` block uses unqualified test names with `--exact`: `cargo test -p cintx-rs --locked aosym_error_path_rejects_non_s1_with_typed_error -- --exact`. Cargo's `--exact` filter requires the fully qualified path including the parent module: `api::tests::aosym_error_path_rejects_non_s1_with_typed_error`. With the unqualified name, `--exact` reports `0 passed; 0 failed; 0 ignored; 0 measured; 13 filtered out` — a *false* indication of "no tests run". Both tests do pass when invoked with the fully qualified path. This is a wording artifact in the plan's verify block, not a bug in the implementation. Documented above under "Targeted aosym test execution".
- Two pre-existing `cintx-cubecl` warnings surface during feature-matrix builds (`warning: cintx-cubecl (lib) generated 8 warnings`). Out of scope per the executor scope-boundary rule. Not logged to deferred-items because they are not new and not caused by Plan 18-02.

## User Setup Required

None — purely internal changes to the public Rust library. No external service configuration, no environment variables, no migrations.

## Next Phase Readiness

- **Plan 18-03 (arity-3 oracle parity sweep):** Can build `SessionRequest` instances with default `ExecutionOptions` (which has `aosym: None`); the new preflight is a no-op for `None`/`Some(S1)` so byte-identity comparison against vendor libcint is unaffected. Tests can use `OperatorId::new(15..=18)` for the existing `int3c1e_*` symbols, `OperatorId::new(19..=20)` for `int3c2e_ip1_*`, and `OperatorId::new(22..=23)` for the new plain `int3c2e_*` rows (all unchanged after Wave 1).
- **Plan 18-04 (arity-4 oracle parity sweep):** Can use `OperatorId::new(9)`/`OperatorId::new(10)` for `int2e_*` (unchanged) and `OperatorId::new(24)`/`OperatorId::new(25)` for the (Plan 18-01-shifted) `int4c1e_*` symbols. The `cintx-rs::api` test that exercises `INT4C1E_CART_OPERATOR_ID` (`evaluate_rejects_out_of_envelope_validated4c1e_requests`) is verified passing under the `with-4c1e` feature.
- **pyscf_rs consumer:** Can now pattern-match `FacadeError::UnsupportedAoSymmetry { requested }` for the four non-S1 packings, satisfying ROADMAP SC#4's "where supported, or returns a typed error" contract.

## Threat Flags

None — no new threat surface introduced. All edits stay inside the established in-process integral evaluation envelope. The new preflight is a defensive validator (Rule 2 family — correctness requirement) that fails fast before any allocation or kernel dispatch; it REDUCES, not increases, the DoS attack surface (`<threat_model>` T-18-02-D accept disposition is satisfied). The `requested: String` field carries only the lowercase pyscf packing name from the public `Display` impl — no secret or PII (T-18-02-I accept).

## Self-Check: PASSED

Verified post-write:

- `crates/cintx-rs/src/error.rs` FOUND — contains `UnsupportedAoSymmetry { requested: String }` (FacadeError), `UnsupportedAoSymmetry,` (FacadeErrorKind), and the matching `kind()` arm; 3 non-comment occurrences confirmed.
- `crates/cintx-rs/src/api.rs` FOUND — contains the preflight `return Err(FacadeError::UnsupportedAoSymmetry`, `cintx_core::AoSymmetry::S1`, the `if let Some(aosym) = self.options.aosym` safe destructure, the `# Memory layout` rustdoc anchor with both `Arity >= 3` and `Arity 2` rows, `INT4C1E_CART_OPERATOR_ID: u32 = 24`, and both new test function definitions.
- Commit `a501776` FOUND (Task 1, error.rs).
- Commit `bcb9325` FOUND (Task 2, api.rs).
- `cargo build -p cintx-rs --locked` exits 0.
- `cargo build -p cintx-rs --features with-f12 --locked` exits 0.
- `cargo build -p cintx-rs --features with-4c1e --locked` exits 0.
- `cargo build -p cintx-rs --features unstable-source-api --locked` exits 0.
- `cargo test -p cintx-rs --locked` exits 0 — 13 tests pass (11 existing + 2 new aosym tests).
- Both new aosym tests pass individually under `--exact` with their fully qualified path.
- Anti-pattern negative greps pass: no `has_vendor_libcint` in api.rs, no naive `.unwrap()` against `S1`.
- Additive-only check: `git diff --staged | grep '^\-pub '` produces no output across both modified files.
- Delta in `#[test]` markers in api.rs: +2 (exactly the two new aosym tests).

---
*Phase: 18-sessionrequest-arity-ge3-dispatch*
*Completed: 2026-05-12*
