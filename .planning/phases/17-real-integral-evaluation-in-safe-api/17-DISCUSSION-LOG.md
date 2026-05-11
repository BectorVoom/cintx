# Phase 17 — Discussion Log

**Date:** 2026-05-12
**Phase:** 17 — Real-Integral Evaluation in Safe API (issue #11 Task 3)
**Mode:** default (no flags)

This log is for human reference only (audits, retrospectives). It is NOT consumed by downstream agents — they read `17-CONTEXT.md`.

---

## Phase scope confirmed before discussion

Read into context:
- `.planning/ROADMAP.md` Phase 17 entry (goal, SC1-3, RVAL-01/02/03, dependency on Phase 16).
- `.planning/STATE.md` — Phase 16 complete, HUMAN-UAT pending; we're entering v1.3 work.
- `.planning/PROJECT.md` — v1.2 complete; v1.3 goal is safe-API closure for pyscf_rs consumer.
- `.planning/notes/pyscf-rs-as-cintx-consumer.md` — downstream consumer is pyscf_rs `pyscf-gto/src/intor.rs`; two oracle gates (this repo + pyscf_rs `tests/oracle/`).
- `.planning/phases/16-multi-backend-support/16-CONTEXT.md` — most recent prior CONTEXT for tone/structure reference.

Scouted code:
- `crates/cintx-rs/src/api.rs` — the file being changed. Located the synthetic placeholder (`fill_staging_values`, lines 465-490) and the shadow `CubeClExecutor` stub (lines 492-562).
- `crates/cintx-compat/src/raw.rs` — confirmed `eval_raw` (line 411) uses `cintx_cubecl::CubeClExecutor` (imported at line 6, constructed at line 461). Phase 17 mirrors this pattern.
- `crates/cintx-cubecl/src/lib.rs:26` — confirmed real `CubeClExecutor` is re-exported.
- `crates/cintx-oracle/tests/one_electron_parity.rs` — confirmed existing per-symbol parity test pattern. Currently `atol=1e-11/rtol=1e-9` (predates Phase 15 unified `atol=1e-12`).
- `crates/cintx-ops/src/generated/api_manifest.csv` — enumerated arity-2 base operators: 12 total (1e × {ovlp,kin,nuc} × {cart,sph,spinor} + 2c2e × {cart,sph,spinor}).

---

## Areas selected for discussion

User selected all 4 proposed gray areas (multiSelect):

1. ☑ Dispatch mechanism
2. ☑ Existing synthetic-value test
3. ☑ Oracle test shape
4. ☑ Arity-2 coverage scope

No areas redirected as scope creep.

---

## Area 1 — Dispatch mechanism

**Question:** How should the safe API get real values?

**Options presented:**
- Swap to `cintx_cubecl::CubeClExecutor` (delete the shadow stub and `fill_staging_values`).
- Route through `unsafe cintx_compat::raw::eval_raw` (repack typed BasisSet/ShellTuple → atm/bas/env arrays).
- Extract a shared dispatch helper (hoist chunk loop into `cintx-runtime` or `cintx-cubecl`).

**User selection:** Swap to `cintx_cubecl::CubeClExecutor`.

**Why this matters:** Determines whether Phase 17 stays a minimal internal swap (yes) or grows into a refactor. Also determines whether the safe API takes an `unsafe` boundary internally (no).

**Locked in:** D-01, D-02, D-03 (the shared-helper option becomes a deferred follow-up).

---

## Area 2 — Existing synthetic-value test

**Question:** `crates/cintx-rs/src/api.rs:659 evaluate_runs_runtime_path_and_returns_owned_output` (line 676 asserts `owned_values[0] == 1.0` — the synthetic placeholder) — how do we handle it once real values land?

**Options presented:**
- Rewrite as deterministic+nonzero check (idempotent, at least one non-zero, extent/byte invariants).
- Cross-check vs `eval_raw` inline (couples cintx-rs unit tests to compat).
- Delete the existing test (rely entirely on new oracle tests).

**User selection:** Rewrite as deterministic+nonzero check.

**Why this matters:** Without this decision, planning either misses that the existing test breaks, or over-couples cintx-rs to compat, or loses unit-level coverage entirely.

**Locked in:** D-04, D-05.

---

## Area 3 — Oracle test shape

**Question:** SC #2 says "extended or sibling test added" — where do the new SessionRequest parity tests live and how are they organized?

**Options presented:**
- New file: `crates/cintx-oracle/tests/safe_api_arity2_parity.rs` (single-purpose, focused on safe-API byte parity).
- Extend existing per-family files (siblings co-located with `eval_raw` tests).
- Data-driven sweep (one parametric test over a const table).

**User selection:** New file: `safe_api_arity2_parity.rs`.

**Why this matters:** Determines diagnosability. Per-symbol named tests in a focused file make safe-API regressions trivially distinguishable from compat regressions.

**Locked in:** D-06, D-07, D-10 (CI runs new file inside the existing `oracle_parity_gate` matrix — no new CI job).

---

## Area 4 — Arity-2 coverage scope

**Question:** Which operators must the new parity sweep cover?

**Options presented:**
- Base 1e + 2c2e × 3 reps (12 ops): the full arity-2 base set.
- Base + unstable-source arity-2 (feature-gated).
- Sph-only minimum for v1.3 unblock (4 ops).

**User selection:** Base 1e + 2c2e × 3 reps (12 ops).

**Why this matters:** Defines the exact verification surface for SC #2. F12/with-4c1e are not arity-2, so they don't contribute. Unstable-source is gated and gets its own follow-up.

**Locked in:** D-08, D-09 (tolerance: `atol=1e-12, rtol=0.0` per Phase 15 unification).

---

## Decisions left to Claude's discretion (not asked)

- Renaming `evaluate_runs_runtime_path_and_returns_owned_output` to be honest about what's asserted now.
- Exact factoring of the test helper inside `safe_api_arity2_parity.rs`.
- Basis fixture choice (default H2O / STO-3G mirroring existing parity tests).
- `#[cfg(has_vendor_libcint)]` guard on the new tests (strongly leaning yes).
- Whether `int2c2e_spinor` needs any extra plumbing — researcher confirms during plan-phase.

---

## Deferred ideas captured

- Shared dispatch helper across safe API and compat (Phase 17.5 / v1.3 polish).
- Unstable-source arity-2 sweep through SessionRequest (when a consumer drives it).
- Multi-molecule oracle fixtures for the safe-API sweep.
- Data-driven parametric sweep helper.
- Inline `eval_raw` cross-check in cintx-rs unit tests.

Phase 18 (arity-3/4 dispatch) and Phase 19 (ECP) are already roadmap'd as separate phases — not in scope.

---

*Discussion log written: 2026-05-12*
