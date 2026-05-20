---
phase: 19-int1e-ecp-type1-type2-evaluator
plan: 08
subsystem: testing
tags: [ecp, oracle, libecpint, ffi, cross-check, build-cfg, non-blocking]

# Dependency graph
requires:
  - phase: 19-int1e-ecp-type1-type2-evaluator (19-06)
    provides: byte-identity scalar ECP kernel + Cu/LANL2DZ fixture + safe-API ECP collector pattern
provides:
  - "Optional, non-blocking libecpint secondary cross-check oracle behind has_libecpint_oracle (CINTX_LIBECPINT_ORACLE=1)"
  - "libecpint extern \"C\" FFI shim (libecpint_ffi.rs) gated #![cfg(has_libecpint_oracle)]"
  - "Env-gated #[ignore] cross-check parity test file at informational atol=1e-9"
  - "Provenance + tolerance-rationale note (.planning/notes/ecp-libecpint-crosscheck.md)"
affects: [milestone-closure, roadmap-sc4, ecp-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Best-effort opt-in cc::Build branch that warns + skips (no build failure) when the optional dependency is absent"
    - "Build cfg emitted only after the optional shim compiles AND the optional lib is linkable"
    - "Informational (non-asserting) cross-check at a loose tolerance vs the strict primary byte-identity gate"

key-files:
  created:
    - crates/cintx-oracle/src/libecpint_ffi.rs
    - crates/cintx-oracle/tests/ecp_libecpint_crosscheck_parity.rs
    - .planning/notes/ecp-libecpint-crosscheck.md
  modified:
    - crates/cintx-oracle/build.rs
    - crates/cintx-oracle/src/lib.rs

key-decisions:
  - "libecpint not vendored: ship opt-in scaffolding (build cfg + FFI shim + env-gated #[ignore] test + note); operator supplies CINTX_LIBECPINT_DIR + extern \"C\" shim to activate"
  - "has_libecpint_oracle is emitted ONLY after the operator shim compiles and libecpint is linkable; a set env var alone with no library produces a cargo:warning and no cfg (default build untouched)"
  - "Cross-check tolerance is informational atol=1e-9, rtol=0.0; mismatches are logged as drift, never asserted (D-02 non-blocking)"

patterns-established:
  - "Pattern: opt-in optional-oracle gate = rerun-if-env-changed + rustc-check-cfg + best-effort discovery branch + #![cfg(...)] FFI module + (#[ignore] AND #[cfg(...)]) tests"
  - "Pattern: optional C++ FFI via operator-supplied extern \"C\" shim compiled with cc::Build cpp(true).flag_if_supported(-std=c++17)"

requirements-completed: [ECP-04]

# Metrics
duration: 5min
completed: 2026-05-20
---

# Phase 19 Plan 08: Optional libecpint Secondary Cross-Check Oracle Summary

**Non-blocking libecpint (Shaw & Hill, JCP 147 074108, 2017, MIT) cross-check behind a `has_libecpint_oracle` build cfg emitted only on `CINTX_LIBECPINT_ORACLE=1`, with an `extern "C"` FFI shim and an env-gated `#[ignore]` parity test at informational atol=1e-9 — the default oracle build is byte-for-byte unchanged.**

## Performance

- **Duration:** 5 min
- **Started:** 2026-05-20T12:32:35Z
- **Completed:** 2026-05-20T12:37:35Z
- **Tasks:** 2
- **Files modified:** 5 (3 created, 2 modified)

## Accomplishments

- Added the optional `has_libecpint_oracle` build cfg in `crates/cintx-oracle/build.rs`, emitted ONLY when `CINTX_LIBECPINT_ORACLE=1` and libecpint + an operator-supplied `extern "C"` shim are reachable — mirroring the Phase 16 ROCm `CINTX_ROCM_ORACLE=1` opt-in precedent. The branch is best-effort: when the gate is set but libecpint is absent it logs a `cargo:warning` and continues WITHOUT emitting the cfg, so the build never fails on a host that lacks libecpint.
- Created `crates/cintx-oracle/src/libecpint_ffi.rs` (`#![cfg(has_libecpint_oracle)]`) with `extern "C"` shim wrappers (`libecpint_ecp_cart`/`libecpint_ecp_sph`) following the `vendor_ffi.rs` slice-taking wrapper shape, registered in `lib.rs` behind the same cfg.
- Created `crates/cintx-oracle/tests/ecp_libecpint_crosscheck_parity.rs` with two tests (`test_int1e_ecp_{cart,sph}_libecpint_crosscheck`), BOTH `#[ignore]` AND `#[cfg(has_libecpint_oracle)]`, at informational `CROSSCHECK_ATOL=1e-9, RTOL=0.0`; mismatches are logged as cross-implementation drift, never asserted.
- Wrote the provenance + tolerance-rationale note `.planning/notes/ecp-libecpint-crosscheck.md` documenting the MIT/JCP-2017 citation, the "different recurrence + quadrature conventions" rationale for the loose tolerance, and the exact operator activation steps.
- VERIFIED no regression: `cargo build -p cintx-oracle --features cpu --locked` succeeds unchanged with the env var unset, the cross-check test target compiles to **0 tests** in the default build (cfg'd out), and the existing `safe_api_ecp_parity` suite still passes.

## libecpint availability this session (live run vs deferred)

libecpint was **NOT available on the dev host this session** — no system package, no `pkg-config` entry, not vendored under `vendor/` (only `vendor/pyscf-nr-ecp/` exists). Per the plan's "optional, non-blocking" framing and the executor critical-constraint, a large external C++ dependency was **not** vendored on the executor's own initiative. Instead the **opt-in mechanism** landed (build cfg + FFI shim signatures + env-gated `#[ignore]` test + provenance note) with the default build verified unchanged. The **live cross-check run is deferred to a host with libecpint installed**; the harness is in place per D-02.

- **Observed cross-implementation |diff| this session:** not measured (libecpint unavailable). To be recorded when run on an equipped host.
- **Normalization/convention adapter:** documented in the test file's `collect_ecp_matrix_libecpint` helper — it assumes the operator shim normalizes libecpint output to libcint column-major `out[j*ni + i]` block layout (the same convention the PySCF vendor collector consumes); that helper is the single place a convention adapter is applied. Because the cross-check is informational, residual convention mismatches surviving within `atol=1e-9` are reported as drift, not corrected.
- **What an operator must provide to activate `CINTX_LIBECPINT_ORACLE=1`:** (1) a libecpint install/build root via `CINTX_LIBECPINT_DIR` (or a vendored `vendor/libecpint/`), (2) an `extern "C"` shim `.cpp` via `CINTX_LIBECPINT_SHIM` exporting `cintx_libecpint_ecp_{cart,sph}`, optionally (3) extra include roots via `CINTX_LIBECPINT_INCLUDE`. Full steps in `.planning/notes/ecp-libecpint-crosscheck.md`.

## Task Commits

Each task was committed atomically via plain git on `main`:

1. **Task 1: Optional libecpint build cfg + extern "C" FFI shim + provenance note** - `2413bc1` (feat)
2. **Task 2: Env-gated #[ignore] libecpint cross-check parity test at informational tolerance** - `3e4a828` (test)

## Files Created/Modified

- `crates/cintx-oracle/build.rs` (modified) - Registers `rustc-check-cfg(has_libecpint_oracle)` + `rerun-if-env-changed=CINTX_LIBECPINT_ORACLE`; adds the best-effort `try_build_libecpint` branch gated on the env var.
- `crates/cintx-oracle/src/lib.rs` (modified) - Registers `pub mod libecpint_ffi` behind `#[cfg(has_libecpint_oracle)]`.
- `crates/cintx-oracle/src/libecpint_ffi.rs` (created) - `extern "C"` shim wrappers `libecpint_ecp_{cart,sph}` over the operator-supplied C surface, gated `#![cfg(has_libecpint_oracle)]`.
- `crates/cintx-oracle/tests/ecp_libecpint_crosscheck_parity.rs` (created) - Two `#[ignore]` + `#[cfg(has_libecpint_oracle)]` informational cross-check tests at atol=1e-9.
- `.planning/notes/ecp-libecpint-crosscheck.md` (created) - Provenance, tolerance rationale, operator activation steps.

## Decisions Made

- **libecpint not vendored.** Shipped the opt-in scaffolding only; documented activation requirements. Rationale: the plan's must-haves are about the OPT-IN MECHANISM and no-regression guarantee, not a passing cross-check in this environment, and the critical-constraint forbids vendoring a large external C++ dependency on the executor's own initiative.
- **cfg emitted only after the shim compiles AND libecpint is linkable.** A bare `CINTX_LIBECPINT_ORACLE=1` with no library produces a `cargo:warning` and no cfg. This guarantees the cross-check tests stay compiled-out and the build never fails on a host lacking libecpint.
- **Informational, non-asserting cross-check.** atol=1e-9 / rtol=0.0; drift is logged, never asserted — the strict atol=1e-12 PySCF `nr_ecp` gate (19-06/19-07) remains the only blocking gate.

## Deviations from Plan

None - plan executed exactly as written. The plan explicitly anticipated the "libecpint not obtainable this session" path (Task 1 Step 4) and instructed shipping the FFI shim + build branch behind the cfg with the live run deferred — which is what was done.

## Issues Encountered

None. libecpint's absence on the host was an anticipated path, handled by the plan's documented deferral framing.

## User Setup Required

None for the default build / CI. To activate the OPTIONAL cross-check on an equipped host, an operator must provide libecpint + an `extern "C"` shim and set `CINTX_LIBECPINT_ORACLE=1` (plus `CINTX_LIBECPINT_DIR`/`CINTX_LIBECPINT_SHIM`) — see `.planning/notes/ecp-libecpint-crosscheck.md`. This is opt-in and never required by CI (D-02 non-blocking).

## Next Phase Readiness

- ROADMAP SC#4's secondary-oracle clause is satisfied with a non-blocking, env-gated libecpint cross-check (D-02 REVISED). Phase 19's optional close is complete.
- Default oracle build and CI gates are unaffected; the primary byte-identity gate remains the PySCF `nr_ecp` atol=1e-12 parity from 19-06/19-07.
- No blockers. Live libecpint cross-check measurement remains an optional follow-up on a host with libecpint installed.

## Threat Surface Scan

No new security-relevant surface beyond the plan's `<threat_model>`. The single new FFI boundary (libecpint `extern "C"` shim) is exactly threat T-19-26, mitigated as planned: gated behind `has_libecpint_oracle` (off by default), takes Rust slices, bounds the out buffer, and is only exercised under explicit opt-in. T-19-27 (default-build regression) mitigated and verified. T-19-25 (informational tolerance masking a bug) accepted by design.

## Self-Check: PASSED

- FOUND: `crates/cintx-oracle/src/libecpint_ffi.rs`
- FOUND: `crates/cintx-oracle/tests/ecp_libecpint_crosscheck_parity.rs`
- FOUND: `.planning/notes/ecp-libecpint-crosscheck.md`
- FOUND: `.planning/phases/19-int1e-ecp-type1-type2-evaluator/19-08-SUMMARY.md`
- FOUND commit: `2413bc1` (Task 1)
- FOUND commit: `3e4a828` (Task 2)
