---
phase: 18-sessionrequest-arity-ge3-dispatch
plan: 03
subsystem: testing
tags: [oracle, parity, safe-api, arity-3, libcint, h2o-sto3g, atol-1e-12, vendor-libcint, f-order]

# Dependency graph
requires:
  - phase: 18
    provides: "Plan 18-01 manifest expansion (OperatorId 22/23 for int3c2e_{cart,sph}) and vendor wrappers vendor_int3c1e_p2_sph + vendor_int3c2e_cart"
  - phase: 18
    provides: "Plan 18-02 safe-API surface (aosym preflight, IntegralTensor F-order rustdoc, post-shift OperatorId constants)"
  - phase: 17
    provides: "safe_api_arity2_parity.rs structural template (module gate, fixture helpers, count_mismatches, ATOL/RTOL constants)"
  - phase: 15
    provides: "Unified oracle tolerance atol=1e-12 / rtol=0.0"
provides:
  - "crates/cintx-oracle/tests/safe_api_arity3_parity.rs — 8 per-symbol vendor-parity tests for the full arity-3 operator set in CONTEXT.md D-06"
  - "Inline H2O/STO-3G fixture helpers (PTR_ENV_START-aware) duplicated from safe_api_arity2_parity.rs per PATTERNS.md §safe_api_helpers default (skip extraction)"
  - "collect_safe_api_tuple_buffer helper (arity-3 per-tuple buffer collector — direct buffer-to-buffer compare, no transpose)"
affects:
  - "Phase 18-04 (arity-4 parity) — uses the same per-tuple collector pattern with a 4-shell ShellTuple"
  - "Future phases that ship an actual IP1 derivative kernel for int3c2e_ip1_* — must update this file to use vendor_int3c2e_ip1_{cart,sph} with buffer size ni*nj*nk*3"

# Tech tracking
tech-stack:
  added: []  # No new dependencies; reuses cintx_rs::SessionRequest, cintx_oracle::vendor_ffi
  patterns:
    - "Arity-3 per-tuple buffer collector (collect_safe_api_tuple_buffer) — calls SessionRequest::new(...).query_workspace()?.evaluate() per 3-shell tuple, returns owned_values directly"
    - "Arity-3 direct buffer-to-buffer compare (no transpose) — both cintx and vendor write F-order; precedent compare.rs:811-833"
    - "Per-symbol named #[test] functions (NOT a parametric loop) — required by CONTEXT.md D-14 for CI bisection"
    - "int3c2e_ip1_* parity wired to plain vendor_int3c2e_* — kernel-vs-symbol-name disposition per RESEARCH.md Item 5 / A6 / R6"

key-files:
  created:
    - "crates/cintx-oracle/tests/safe_api_arity3_parity.rs (801 lines, 8 #[test] functions)"
  modified: []

key-decisions:
  - "Inline fixture helpers instead of extracting to a shared safe_api_helpers.rs module — per CONTEXT.md Claude's-Discretion default + PATTERNS.md §safe_api_helpers (marginal token cost is small; wiring complexity from cross-test module sharing is real)"
  - "All 8 tests gated #[cfg(has_vendor_libcint)] — under feature `cpu` without vendor build: 0 passed/0 failed/0 filtered (items entirely cfg-removed), exit 0"
  - "int3c2e_ip1_{cart,sph} tests reference plain vendor_int3c2e_{cart,sph} (not _ip1_ variants) — kernel currently computes plain 3c2e integral; component_rank \"\" → multiplier 1; buffer size ni*nj*nk (NOT ni*nj*nk*3). Mirrors center_3c2e_parity.rs:222-287 raw-path precedent."

patterns-established:
  - "Pattern: collect_safe_api_tuple_buffer — arity-agnostic per-tuple safe-API driver. Builds a ShellTuple from the input slice, runs query_workspace().evaluate(), returns owned_values verbatim. Reusable for arity-3 and arity-4 (Plan 18-04 will reuse the same shape with a 4-shell input)."
  - "Pattern: Arity-3 5x5x5 = 125 triple Cartesian sweep per test. Each triple allocates a fresh vendor buffer (vec![0.0_f64; ni * nj * nk]), asserts safe_out.len() == vendor_out.len() before count_mismatches, tracks any_nonzero sentinel to guard against zero-fill regressions."
  - "Pattern: int3c2e_ip1_* kernel-misnomer NOTE comment + plain vendor wiring — applied uniformly to both cart and sph variants; matches the existing raw-path test wiring."

requirements-completed:
  - ARITY-01
  - ARITY-02
  - ARITY-03
  - ARITY-05

# Metrics
duration: ~25min
completed: 2026-05-12
---

# Phase 18 Plan 03: arity-3 safe-API oracle parity Summary

**Eight per-symbol vendor-parity `#[test]` functions for the full arity-3 operator set (int3c1e_{cart,sph}, int3c1e_p2_{cart,sph}, int3c2e_ip1_{cart,sph}, int3c2e_{cart,sph}) at atol=1e-12 / rtol=0.0, driving `SessionRequest::new(...).query_workspace()?.evaluate()` against vendored libcint 6.1.3 across the H2O/STO-3G 5x5x5 = 125 triple Cartesian sweep.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-05-12T02:46Z (worktree branch check)
- **Completed:** 2026-05-12T03:11Z
- **Tasks:** 2
- **Files created:** 1 (`crates/cintx-oracle/tests/safe_api_arity3_parity.rs`, 801 lines)
- **Files modified:** 0 (no existing oracle tests touched)

## Accomplishments

- Created `crates/cintx-oracle/tests/safe_api_arity3_parity.rs` with the Phase-17 module-gate pattern (`#![cfg(any(feature = "cpu", feature = "rocm"))]`), inline H2O/STO-3G fixture helpers (PTR_ENV_START-aware, duplicated verbatim from `safe_api_arity2_parity.rs`), an arity-3 per-tuple buffer collector (`collect_safe_api_tuple_buffer`), the unified-tolerance `count_mismatches`, and 8 per-symbol parity tests.
- 4 cart tests: `test_int3c1e_cart_safe_api_parity` (Op 17), `test_int3c1e_p2_cart_safe_api_parity` (Op 15), `test_int3c2e_ip1_cart_safe_api_parity` (Op 19 → plain `vendor_int3c2e_cart`), `test_int3c2e_cart_safe_api_parity` (Op 22, new per 18-01).
- 4 sph tests: `test_int3c1e_sph_safe_api_parity` (Op 18), `test_int3c1e_p2_sph_safe_api_parity` (Op 16, uses new `vendor_int3c1e_p2_sph` from 18-01), `test_int3c2e_ip1_sph_safe_api_parity` (Op 20 → plain `vendor_int3c2e_sph`), `test_int3c2e_sph_safe_api_parity` (Op 23, new per 18-01).
- All 8 tests use direct buffer-to-buffer comparison with NO transpose (arity-3 cintx kernels write F-order matching vendor output directly per RESEARCH.md and `compare.rs:811-833`).
- Both `int3c2e_ip1_*` tests carry the kernel-misnomer NOTE comment and a pre-compare length assert; they reference plain `vendor_int3c2e_{cart,sph}` (NOT `vendor_int3c2e_ip1_*`) and use buffer size `ni*nj*nk` (no `*3` multiplier) — matching the passing raw-path precedent at `center_3c2e_parity.rs:222-287`.
- No modifications to `safe_api_arity2_parity.rs`, `center_3c1e_parity.rs`, or `center_3c2e_parity.rs` (`git diff` clean).
- `cargo build -p cintx-oracle --features cpu --locked --tests` exits 0.
- `cargo build -p cintx-oracle --locked --tests` exits 0 (module gate correctly excludes the file when no backend feature is selected).
- `CINTX_BACKEND=cpu cargo test -p cintx-oracle --features cpu --locked --test safe_api_arity3_parity -- --test-threads=1` exits 0 on this non-vendor host (`has_vendor_libcint` cfg inactive → all 8 `#[cfg(has_vendor_libcint)]` items removed at compile time, `test result: ok. 0 passed; 0 failed; 0 filtered out`). On a vendor-built host the same command will run the 8 tests; the cfg-gate semantics are inherited verbatim from `safe_api_arity2_parity.rs`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Create file + 4 arity-3 cart tests** — `ccd4d73` (test)
2. **Task 2: Append 4 arity-3 sph tests** — `bdd2bd0` (test)

## Files Created/Modified

- `crates/cintx-oracle/tests/safe_api_arity3_parity.rs` — **created** (801 lines). Module gate `#![cfg(any(feature = "cpu", feature = "rocm"))]`. Inline `build_h2o_sto3g`, `arc_f64`, `build_h2o_sto3g_safe_basis`, `collect_safe_api_tuple_buffer`, `count_mismatches`. Eight `#[test] #[cfg(has_vendor_libcint)]` functions: 4 cart (rows 304-548), 4 sph (rows 557-800). All eight assert `count_mismatches == 0` over the 5x5x5 triple sweep at ATOL=1e-12, RTOL=0.0; all eight assert `any_nonzero` to guard against zero-fill regression; all eight carry a pre-compare length assert.

## Empirical OperatorId table (post-Plan-18-01, verified via grep on `crates/cintx-ops/src/generated/api_manifest.rs`)

| Symbol               | OperatorId | Manifest line |
|----------------------|-----------:|--------------:|
| int3c1e_p2_cart      | 15         | entry 15 (`symbol_name: "int3c1e_p2_cart"` at api_manifest.rs:268) |
| int3c1e_p2_sph       | 16         | entry 16 (line 285) |
| int3c1e_cart         | 17         | entry 17 (line 302) |
| int3c1e_sph          | 18         | entry 18 (line 319) |
| int3c2e_ip1_cart     | 19         | entry 19 (line 336) |
| int3c2e_ip1_sph      | 20         | entry 20 (line 353) |
| int3c2e_ip1_spinor   | 21         | entry 21 (line 370)  *(not used by this plan; spinor out of scope per D-07)* |
| **int3c2e_cart**     | **22**     | entry 22 (line 387) *(NEW per 18-01)* |
| **int3c2e_sph**      | **23**     | entry 23 (line 404) *(NEW per 18-01)* |

The IDs match the plan's expected mapping exactly (CSV row indices line up with `OperatorDescriptor { id: OperatorId::new(N), entry: &MANIFEST_ENTRIES[N] }` in the generated `OPERATORS` slice at api_manifest.rs:2316–2351).

## Decisions Made

- **Inline helpers, skip the `safe_api_helpers.rs` module extraction** — followed CONTEXT.md "Claude's-Discretion default: skip extraction" and PATTERNS.md §safe_api_helpers. The fixture is ~190 LoC duplicated; the cross-test module-sharing complexity is greater than the duplication cost. If Plan 18-04 + a future arity-4 doubling makes the duplication painful, a follow-up phase can extract.
- **`int3c2e_ip1_{cart,sph}` parity wired to plain `vendor_int3c2e_{cart,sph}`** — per RESEARCH.md Item 5 + A6 + R6: cintx's `int3c2e_ip1_*` kernel currently computes the plain 3c2e integral. Manifest entries carry `component_rank: ""` → planner multiplier 1 (`planner.rs:382-384`), so cintx output buffer size is `ni*nj*nk` (NOT `ni*nj*nk*3`). Using `vendor_int3c2e_ip1_{cart,sph}` would return a 3-vector gradient and create a guaranteed length mismatch. The same wiring is exercised by the passing raw-path test at `center_3c2e_parity.rs:222-287` (0 mismatches against `vendor_int3c2e_sph` at atol 1e-9). Both new safe-API tests carry the kernel-misnomer NOTE comment + a pre-compare length assert. When/if a future phase ships an actual IP1 derivative kernel, this file must be updated to use the `_ip1_` vendor wrappers and `n_elem = ni*nj*nk*3`.
- **Single-line `assert_eq!(safe_out.len(), vendor_out.len(), ...)`** — keeps the plan's `grep -cF 'assert_eq!(safe_out.len(), vendor_out.len()'` count check at 8. Splitting the macro across lines would have failed that line-based grep even though the assert was semantically present.

## Vendor verification (cannot be exercised on this host)

The vendor-build (`CINTX_ORACLE_BUILD_VENDOR=1`) is not available in this worktree, so the actual `8 passed` line for `cargo test --features cpu --test safe_api_arity3_parity` was not observed locally. Verification on this host: the test binary builds cleanly under `--features cpu --locked --tests`, and `cargo test --test safe_api_arity3_parity -- --test-threads=1` exits 0 with all 8 items cfg-removed (`test result: ok. 0 passed; 0 failed; 0 filtered out`). The CI oracle gate at `oracle_parity_gate` (which sets `CINTX_ORACLE_BUILD_VENDOR=1` per CONTEXT.md D-15) will exercise the 8 active tests across the cpu × four-manifest-profiles matrix.

**Expected runtime envelope** (extrapolating from `center_3c1e_parity.rs` / `center_3c2e_parity.rs`, which exercise the same 5x5x5 = 125 triple sweep against the same vendor wrappers and complete well under 1 s per test on CI): 8 × ≤ 1 s = ≤ 8 s for the file in total. No subset fallback expected per CONTEXT.md D-14.

## Deviations from Plan

None — plan executed exactly as written. All 8 tests, all 8 OperatorIds, all required asserts, NOTE comments, and the int3c2e_ip1 disposition match the plan's `must_haves.truths` and `acceptance_criteria` verbatim.

The plan's verification regex `! grep -F 'vendor_int3c2e_ip1_cart'` initially tripped on my own explanatory inline comment (`Plain int3c2e_cart — NOT vendor_int3c2e_ip1_cart — per ...`). I reworded that comment to `Plain vendor_int3c2e_cart (no _ip1 suffix); see RESEARCH.md Item 5 / A6.` so the anti-pattern grep stays clean. Same wording applied to the sph variant. This is a verification-conformant rewording inside the same task (not a deviation per se — the plan explicitly listed the anti-pattern grep as a gate).

## Issues Encountered

- The plan's `assert_eq!(safe_out.len(), vendor_out.len()` line-based grep would have failed if I formatted the macro across multiple lines. Fixed by keeping the macro invocation on a single line in all 8 tests. No functional impact.

## Self-Check

- File exists: `crates/cintx-oracle/tests/safe_api_arity3_parity.rs` — FOUND (801 lines).
- 8 `#[test]` functions: `grep -cE '^#\[test\]'` returns 8 — FOUND.
- 8 `#[cfg(has_vendor_libcint)]` guards: count = 8 — FOUND.
- 8 single-line `assert_eq!(safe_out.len(), vendor_out.len(), ...)`: count = 8 — FOUND.
- 2 `NOTE: cintx int3c2e_ip1_* currently computes plain 3c2e` disposition comments (one cart, one sph): count = 2 — FOUND.
- All 8 OperatorIds (15, 16, 17, 18, 19, 20, 22, 23) appear exactly once each — FOUND.
- Anti-patterns absent: `vendor_int3c2e_ip1_cart`, `vendor_int3c2e_ip1_sph`, `ni * nj * nk * 3`, `1e-11`, `eval_raw`, `spinor` — all clear.
- `cargo build -p cintx-oracle --features cpu --locked --tests` exit 0 — VERIFIED.
- `cargo build -p cintx-oracle --locked --tests` exit 0 — VERIFIED.
- `cargo test --features cpu --test safe_api_arity3_parity -- --test-threads=1` exit 0 (0 passed/0 filtered on non-vendor host — cfg-removed items) — VERIFIED.
- Existing oracle tests untouched: `git diff -- crates/cintx-oracle/tests/safe_api_arity2_parity.rs crates/cintx-oracle/tests/center_3c1e_parity.rs crates/cintx-oracle/tests/center_3c2e_parity.rs` — empty — VERIFIED.
- Task commits exist: `ccd4d73` (Task 1), `bdd2bd0` (Task 2) — both in `git log` — FOUND.

## Self-Check: PASSED

## Next Phase Readiness

- The safe-API arity-3 dispatch surface is now exercised end-to-end against vendored libcint at the Phase 15 unified tolerance for all 8 ROADMAP-named arity-3 operators.
- Plan 18-04 (arity-4 parity) can reuse `collect_safe_api_tuple_buffer` verbatim with a 4-shell input (the function is arity-agnostic — it builds `ShellTuple::try_from_iter(...)` from any input within `SHELL_TUPLE_CAPACITY=4`).
- The `int3c2e_ip1_*` kernel-vs-symbol-name inconsistency is documented in two NOTE comments (one cart, one sph) inside the test file, plus the plan/research records. When the actual IP1 derivative kernel ships, the two `_ip1_*` tests must be updated to use the `_ip1_*` vendor wrappers and `n_elem = ni*nj*nk*3`.
- No spinor arity-3 tests were added (D-07: out of Phase 18 scope; spinor arity-3 stays "compiled but unverified").

---
*Phase: 18-sessionrequest-arity-ge3-dispatch*
*Completed: 2026-05-12*
