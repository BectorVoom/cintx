---
phase: 21-coulomb-gradient-intors
plan: "08"
subsystem: manifest/verification/consumer-handoff
tags: [oracle-covered, manifest-audit, R3-F-order, vendor-parity, pyscf-rs-handoff, GRAD-10, phase-close-out]

# Dependency graph
requires:
  - phase: 21 (plans 01-07)
    provides: "the 6 gradient family kernels + int3c2e_ip1 derivative + per-family vendor parity gates + the env-slot/manifest/surface plumbing"
provides:
  - "oracle_covered=true for all 6 gradient families (cart+sph) + int3c2e_ip1 in the manifest lock"
  - "regenerated api_manifest.{rs,csv}"
  - "green manifest-audit (status ok, no mismatch)"
  - "the R3 component-leading [3,...] F-order validation conclusion"
  - "the pyscf_rs Phase 7 gradient un-gate hand-off note"
affects: [pyscf-rs-consumer, ci-oracle-gate, phase-21-completion]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Flip oracle_covered=true ONLY after the family's vendor parity gate is green (threat T-21-08-02; mitigation order preserved)"
    - "Element-for-element vendor byte-identity IS the component-leading [3,...] F-order layout validation (R3) — no separate layout assertion"
    - "oracle-covered-update stamps helper/transform/optimizer/legacy siblings; operator rows edited in the lock directly (Phase 15/19 precedent)"

key-files:
  created:
    - .planning/notes/phase-21-pyscf-rs-handoff.md
  modified:
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-ops/src/generated/api_manifest.csv

key-decisions:
  - "Ran the full vendor-gated gradient oracle suite (CINTX_ORACLE_BUILD_VENDOR=1 + --features cpu) BEFORE flipping oracle_covered, so the flip follows a green parity gate (T-21-08-02 mitigation)"
  - "Reverted the 6 gradient SPINOR operator rows back to oracle_covered=false after oracle-covered-update stamped them true (Rule 1 fix): the parity report records spinor gradients as skipped-not-verified (UnsupportedApi, R5/D-03), so a true stamp would be a false verification claim — matches the 21-05/21-06 precedent and the orchestrator's explicit 'spinor stays false per R5' directive"
  - "xtask is its own workspace (own [workspace] block + own Cargo.lock); invoked via --manifest-path xtask/Cargo.toml (cargo run -p xtask fails from the worktree root)"
  - "ROADMAP/STATE/REQUIREMENTS finalization is orchestrator-owned (verify->phase.complete flow) and was intentionally NOT edited here, per the orchestrator scope override"

requirements-completed: [GRAD-10]

# Metrics
metrics:
  duration_min: 22
  tasks: 2
  files_changed: 4
  completed: "2026-05-26"
---

# Phase 21 Plan 08: Phase Close-Out — R3 Layout Validation, Manifest Coverage, pyscf_rs Hand-off Summary

Validated the component-leading `[3, …]` F-order layout against vendored libcint
6.1.3 (Risk R3), flipped `oracle_covered=true` for all six plain-Coulomb
gradient families + `int3c2e_ip1` (cart+sph) after a green full vendor-gated
oracle suite at atol=1e-12, kept manifest-audit green, and wrote the pyscf_rs
Phase 7 gradient un-gate hand-off note. Tracking-doc finalization is left to the
orchestrator.

## What Shipped

### Task 1 — R3 validation + oracle_covered flips + full oracle suite (commit `10f985e`)

- **R3 F-order validation (PART A).** Re-ran the full vendor-gated gradient
  oracle suite (double-gated: `CINTX_ORACLE_BUILD_VENDOR=1` + `--features cpu`).
  The element-for-element byte-identity comparison vs libcint's own
  `int2e_ip1` / `int3c2e_ip1` (and the 1e/ecp gradient references) IS the
  component-leading `[3, …]` F-order gate — a layout drift would produce nonzero
  `count_mismatches`. All parity assertions executed (not skipped) and reported
  0 mismatches at atol=1e-12. The 21-05/21-06 SUMMARYs document the transpose
  strides (`int2e_ip1` → `[3, nl, nk, nj, ni]`; `int3c2e_ip1` → `[3, nk, nj, ni]`)
  matching the pyscf-gto `layout_table.rs` component-leading convention. **R3 validated.**
- **oracle_covered flips (PART B).** Flipped the 10 remaining gradient operator
  rows (`int1e_ipovlp/ipkin/ipnuc/iprinv` + `int2e_ip1`, cart+sph) from
  `false` → `true`. `int3c2e_ip1` (cart+sph) and `int1e_ecp_iprinv` (cart+sph)
  were already `true` from 21-06/21-07. Ran `oracle-covered-update` to stamp the
  helper/optimizer/legacy siblings, then `cargo build -p cintx-ops` to
  regenerate `api_manifest.{rs,csv}`.
- **manifest-audit (PART C).** `manifest-audit` exits 0 with `"status": "ok"`,
  `"has_mismatch": false`. The lock stays well-formed (`python3 -m json.tool`
  exits 0). `CINTX_BACKEND=cpu cargo check --workspace --features cpu` exits 0.

### Task 2 — pyscf_rs hand-off note (commit `859a0a2`)

- Created `.planning/notes/phase-21-pyscf-rs-handoff.md` recording: the
  byte-identity status table for all 6 families + `int3c2e_ip1`; the Phase 7
  `workflow_dispatch` arms that un-gate (RHF/UHF/RKS/UKS/MP2/CCSD + CPHF +
  geomopt, riding `grad.verify_fd` ≤1e-6 Ha/Bohr + upstream-PySCF ≤1e-7 Ha/Bohr
  with zero pyscf_rs rework); the R3 layout conclusion; the `int3c2e_ip1` Risk-R1
  re-gating history (operator-blind scalar stub → real derivative + oracle
  reference flip; `pyscf-grad/src/hooks.rs:24` now gets a correct derivative);
  the R2 high-l/nroots>5 l-ceiling caveat; the R6 safe-API-vs-raw-path note; and
  the R5 spinor-UnsupportedApi caveat.

## Verification Results

| Gate | Command | Result |
|------|---------|--------|
| Full vendor gradient oracle suite | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test two_electron_ip1_parity --test center_3c2e_parity --test one_electron_grad_parity --test one_electron_nuc_grad_parity --test ecp_iprinv_parity` | GREEN — assertions executed, 0 mismatches @ atol=1e-12 |
| `two_electron_ip1_parity` | (above) | 3/3 (incl. `oracle_parity_int2e_ip1_{cart,sph}_spd`) |
| `center_3c2e_parity` | (above) | 2/2 (incl. `test_center_3c2e_sph_h2o_sto3g_vendor_parity`) |
| `one_electron_grad_parity` | (above) | 8/8 (incl. `test_int1e_ip{ovlp,kin}_{cart,sph}_h2o_sto3g_parity`) |
| `one_electron_nuc_grad_parity` | (above) | 6/6 (incl. `test_int1e_ip{nuc,rinv}_{cart,sph}_h2o_sto3g_parity`) |
| `ecp_iprinv_parity` | (above) | 3/3 (incl. `test_ECPscalar_iprinv_{cart,sph}_cu_lanl2dz_parity`) |
| Manifest JSON | `python3 -m json.tool …lock.json` | exits 0 (well-formed) |
| Manifest audit | `cargo run --manifest-path xtask/Cargo.toml -- manifest-audit` | exits 0, `"status":"ok"`, `has_mismatch:false` |
| Workspace | `CINTX_BACKEND=cpu cargo check --workspace --features cpu` | exits 0 |

### Final oracle_covered state (gradient families)

| Family | cart | sph | spinor |
|--------|------|-----|--------|
| `int1e_ipovlp` | true | true | false (R5) |
| `int1e_ipkin` | true | true | false (R5) |
| `int1e_ipnuc` | true | true | false (R5) |
| `int1e_iprinv` | true | true | false (R5) |
| `int2e_ip1` | true | true | false (R5) |
| `int3c2e_ip1` | true | true | false (R5) |
| `int1e_ecp_iprinv` | true | true | false (R5) |

Total `oracle_covered: true` entries in the lock: 145 (operators with green
parity + all helper/transform/optimizer/legacy siblings).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `oracle-covered-update` falsely stamped the 6 gradient spinor operator rows true**
- **Found during:** Task 1 PART B verification (after running `oracle-covered-update`).
- **Issue:** `oracle-covered-update` inserts every symbol present in the
  base-profile parity report into `covered_symbols`. The parity report
  (`compare.rs:1102-1135`) records spinor gradients (`component_count==3` +
  `representation=="spinor"`) as a **skipped-but-passing** fixture
  (`"skipped": "spinor gradient transform unsupported by design (R5/D-03)"`),
  so the stamper flipped `int1e_ipovlp_spinor`, `int1e_ipkin_spinor`,
  `int1e_ipnuc_spinor`, `int1e_iprinv_spinor`, `int2e_ip1_spinor`, and
  `int3c2e_ip1_spinor` to `true`. These kernels return `UnsupportedApi` — they
  are not verified, so a `true` stamp is a false verification claim
  (threat T-21-08-02) and contradicts the explicit R5/D-03 contract and the
  21-05/21-06 precedent.
- **Fix:** Reverted those 6 spinor operator rows back to `oracle_covered=false`
  with a targeted Python edit (minimal diff). `int1e_ecp_iprinv_spinor` was
  never stamped (ecp is a dedicated-oracle family excluded from the base parity
  matrix) and stays `false` as before. All cart+sph rows remain `true`.
- **Files modified:** `crates/cintx-ops/generated/compiled_manifest.lock.json`
- **Verification:** post-revert check confirms 12 gradient cart+sph operator
  rows + 2 ecp cart+sph rows `true`, all 7 spinor rows `false`; `manifest-audit`
  exits 0; lock well-formed.
- **Commit:** `10f985e`

### Process note (handoff-note path drift, resolved)

The Write tool initially created `.planning/notes/phase-21-pyscf-rs-handoff.md`
in the MAIN repo (`/home/user/Documents/workspace/cintx/.planning/...`) instead
of the worktree, despite the working directory being the worktree (#3099
absolute-path resolution drift). Caught immediately (the worktree-relative grep
failed). Recovery: copied the file into the worktree-absolute path and removed
it from the main repo; the main repo working tree has no stray planning-note
state from this plan. The note was then staged and committed inside the worktree.

## Orchestrator-Owned Finalization (NOT done here)

Per the orchestrator scope override, the following are **intentionally not
edited** in this plan and are finalized by the orchestrator's standard
verify → `phase.complete` flow after this executor returns (worktree mode also
auto-skips STATE/ROADMAP edits):

- `.planning/ROADMAP.md` — Phase 21 → `8/8 | Complete` + the 8 plan checkboxes.
- `.planning/STATE.md` — Current Position + Accumulated Context decisions +
  milestone counters.
- `.planning/REQUIREMENTS.md` — GRAD-01..GRAD-10 `- [ ]` → `- [x]` + Traceability
  rows Pending → Complete + Coverage summary.

## Threat Model Dispositions

| Threat ID | Disposition | Outcome |
|-----------|-------------|---------|
| T-21-08-01 (R3 F-order layout drift) | mitigate | CLOSED — the full vendor byte-identity suite re-ran green (0 mismatches @ atol=1e-12); the layout conclusion is recorded in the hand-off note for the consumer. |
| T-21-08-02 (premature/false oracle_covered flip) | mitigate | CLOSED — operator flips followed a green parity gate; the spurious spinor stamps (skipped/UnsupportedApi) were reverted to `false`; `manifest-audit` re-checked green. |
| T-21-08-03 (int3c2e_ip1 R1 history lost) | mitigate | CLOSED — the hand-off note records the full stub → real-derivative + oracle-flip history; the consumer knows pyscf-grad now gets a correct derivative. |
| T-21-08-04 (historical phase docs edited) | accept | Honoured — only the live note was created; no historical 21-CONTEXT/RESEARCH/PATTERNS or prior phase docs touched. ROADMAP/STATE/REQUIREMENTS left to the orchestrator. |
| T-21-08-SC (package installs) | accept | No new external packages; only manifest edits + a new planning note. |

## Known Stubs

None introduced. The spinor gradient rows are `oracle_covered=false`
(register-but-`UnsupportedApi`, R5/D-03) — a documented intentional state, not a
stub blocking the plan goal; the un-gating consumer arms need only cart+sph.

## Self-Check: PASSED

- Created file present: `.planning/notes/phase-21-pyscf-rs-handoff.md` (worktree), `21-08-SUMMARY.md`.
- Modified files present: `compiled_manifest.lock.json`, `api_manifest.rs`, `api_manifest.csv`.
- Commits present: `10f985e` (feat — manifest flips), `859a0a2` (docs — hand-off note).
- manifest-audit exits 0; full vendor oracle suite green @ atol=1e-12.

---
*Phase: 21-coulomb-gradient-intors*
*Completed: 2026-05-26*
