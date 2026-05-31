---
phase: 26-group-5-spin-free-giao-nmr-integrals-complex
plan: 08
subsystem: kernels
tags: [giao, code-quality, wr-04, wr-05, in-02, in-03, hardening, gap-closure]

# Dependency graph
requires:
  - phase: 26-group-5-spin-free-giao-nmr-integrals-complex
    plan: 05
    provides: "int1e_a01gp byte-identical via restored 0.5 common factor; a01gp guard removed; GIAO-01 fully closed (all 11 1e families parity-green)"
provides:
  - "GIAO not0 counts the imaginary half of the interleaved [re=0, im=v] buffer only (one_electron.rs + two_electron.rs) — matches libcint real double* semantics (WR-04)"
  - "is_rinv_center is data-driven from an explicit per-family bool on the giao_nuc_op dispatch tuple (mirrors moment is_origj), not from the op_kind>=2 ordinal coupling (IN-02)"
  - "inert comptime complex_output hint removed from the moment/1e device path — no dead arg threaded through three signatures (WR-05)"
  - "GIAO per-engine VRR headroom (overlap nmax=li+lj+3, nuclear nmax=li+lj+5, nroots=nmax/2+1) lives in shared const fns consumed by host guard + host nroots + both host-side device sizings (IN-03)"
affects: [giao, complex-output, 26-verification-warnings]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "GIAO complex-interleaved not0 counts the imaginary half only (chunks_exact(2).filter(c[1])) — the re half is dense-zero, so the numeric result is unchanged today, but the semantics now correctly mirror libcint's real double* output convention"
    - "Per-family selector flags ride the dispatch tuple (is_rinv_center alongside op_kind/rank) rather than being re-derived from dispatch ordinals — same anti-coupling pattern as moment is_origj and the OperatorId-shift project memory"
    - "GIAO headroom centralized in const fns (giao_ovlp_nmax/giao_nuc_nmax/giao_nuc_nroots) consumed by host Rust; the #[cube] kernel bodies keep inline nmax arithmetic because CubeCL forbids plain-fn calls inside #[cube] (D-08), but the host envelope they read is governed by the const fns (D-13 anti-truncation)"

key-files:
  created:
    - .planning/phases/26-group-5-spin-free-giao-nmr-integrals-complex/26-08-SUMMARY.md
  modified:
    - crates/cintx-cubecl/src/kernels/one_electron.rs
    - crates/cintx-cubecl/src/kernels/two_electron.rs

key-decisions:
  - "WR-05 path: REMOVED the inert comptime complex_output hint (not annotated). Removal was clean and bounded — the param was pure passthrough through exactly three signatures (one_electron_moment_kernel #[cube], run_1e_moment_device, run_1e_moment_on_backend) plus one call site, with the only consumer being a dead `let _is_complex_out = comptime!(...)` bind. No on-device math used it. The plan's annotation fallback was unnecessary; the future GIAO-on-device path (Phase 30 GIAO×σ) will add its own output-convention plumbing when it lands."
  - "IN-03 scope: applied giao_nuc_nroots only to the GIAO nuclear arm (one_electron.rs:8825). The structurally-identical (li+lj+5)/2+1 nroots ceiling in the deriv3/deriv4 HOST nuclear path (one_electron.rs:9485, Phase 25 HESS-04) was LEFT UNTOUCHED — it is a different family engine (host-routed Hessian, HOST_RYS_NROOTS_CEILING_1E) that coincidentally shares the +5 headroom literal; reusing a giao_-prefixed const fn there would be a semantic mislabel."
  - "IN-03 #[cube] bodies: the device kernel bodies (one_electron_giao_ovlp_kernel ~2795, one_electron_giao_nuc_kernel ~3267) keep their inline `nmax = li+lj+{3,5}u32` arithmetic because D-08 forbids calling a plain const fn inside #[cube]. The const fns govern the HOST-side device buffer sizing (run_1e_giao_ovlp_device:3100, run_1e_giao_nuc_device:3729) which is the allocation envelope the kernel reads — that is the drift surface D-13 cares about, and it is now single-sourced."

patterns-established:
  - "When centralizing constants shared between host Rust and #[cube] device code, the const fn drives the HOST-side buffer sizing (the allocation envelope) — the #[cube] body keeps inline arithmetic but is guaranteed to fit because the host guard and host sizing both derive from the same const fn."

requirements-completed: []

# Metrics
duration: 25min
completed: 2026-05-31
---

# Phase 26 Plan 08: GIAO Kernel-Quality Hardening (WR-04 / WR-05 / IN-02 / IN-03) Summary

**Closed the four GIAO kernel-quality warnings flagged in 26-VERIFICATION/26-REVIEW: GIAO `not0` now counts only the imaginary half of the interleaved `[re=0, im=v]` buffer in both kernel files (WR-04); `is_rinv_center` is data-driven from an explicit per-family bool on the `giao_nuc_op` dispatch tuple instead of the `op_kind>=2` ordinal coupling (IN-02); the inert comptime `complex_output` hint was removed from the moment/1e device path (WR-05); and the GIAO per-engine VRR headroom lives in shared `const fn`s consumed by the host guard, the host nuclear nroots ceiling, and both host-side device buffer sizings (IN-03). All changes are behavior-neutral: GIAO 1e parity stays 11/11 and 2e parity 4/4 byte-identical to libcint 6.1.3; no a01gp guard was reintroduced.**

## Performance

- **Duration:** ~25 min
- **Tasks:** 3
- **Files modified:** 2 (1 created besides this SUMMARY)

## Accomplishments

- **Task 1 — WR-04 imaginary-half not0 (`one_electron.rs` + `two_electron.rs`):** Changed both GIAO `not0` computations (`write_giao_complex_staging` ~8500; `launch_two_electron_giao2e` ~2528) from `staging.iter().filter(v.abs())` to `staging.chunks_exact(2).filter(|c| c[1].abs())`, counting only the imaginary component of the interleaved `[re=0, im=v]` buffer. The numeric result is unchanged today (the re half is dense-zero), but the count now correctly mirrors libcint's real `double*` output semantics. Verified `grep -c chunks_exact(2)` = 2 (one_electron) / 1 (two_electron); `cargo test -p cintx-cubecl --features cpu` green; vendor-gated GIAO 1e parity 11/11 and 2e parity 4/4. Committed `5a472d2`.
- **Task 2 — IN-02 explicit is_rinv_center + WR-05 inert-hint removal (`one_electron.rs`):** (IN-02) Extended `giao_nuc_op` from `Option<(u32, u32)>` to `Option<(u32, u32, bool)>` carrying a per-family `is_rinv_center` bool, mirroring the moment path's `is_origj` precedent. Updated the destructuring `if let Some((op_kind, rank, is_rinv_center)) = giao_nuc_op` and removed `let is_rinv_center = op_kind >= 2;`. The `op_kind >= 2` literal is now absent from the file (grep count 0). (WR-05) Removed the inert `#[comptime] complex_output: u32` param and its dead `let _is_complex_out = comptime!(...)` bind from `one_electron_moment_kernel`, plus the passthrough through `run_1e_moment_device` and `run_1e_moment_on_backend` (5 backend arms) and the `complex_output_hint` plumbing at the call site (`grep -c 'complex_output: u32'` = 0). Verified build + cubecl tests green; GIAO 1e parity 11/11 and ALL moment parity families (r/low/high/nontensor/genctr = 20 tests) green — WR-05 is behavior-neutral. Committed `b5487dc`.
- **Task 3 — IN-03 shared GIAO headroom const fns (`one_electron.rs`):** Added `const fn giao_ovlp_nmax(li,lj) -> li+lj+3`, `giao_nuc_nmax(li,lj) -> li+lj+5`, and `giao_nuc_nroots(li,lj) -> giao_nuc_nmax/2+1` next to the `moment_params` precedent. Wired all four host sites: the host overlap fail-closed guard (8745 `giao_ovlp_nmax(..) > 8`), the host nuclear Rys-nroots ceiling (8825 `giao_nuc_nroots(..)`), and both host-side device buffer sizings (`run_1e_giao_ovlp_device:3100`, `run_1e_giao_nuc_device:3729`). The VRR envelope is now single-sourced; the host guard and the device allocation cannot drift (D-13). Verified `grep -c 'const fn giao_'` = 3; headroom-fn occurrence count 10; build green; GIAO 1e parity 11/11 unchanged. Committed `718ebb0`.

## Task Commits

1. **Task 1: WR-04 imaginary-half not0 (both kernel files)** — `5a472d2` (fix)
2. **Task 2: IN-02 explicit is_rinv_center + WR-05 remove inert hint** — `b5487dc` (refactor)
3. **Task 3: IN-03 shared GIAO headroom const fns** — `718ebb0` (refactor)

## Deviations from Plan

### Decisions (documented per plan request)

**1. [Decision] WR-05 resolved by REMOVAL, not annotation**
- **Found during:** Task 2
- **Issue:** The plan preferred removal but offered an annotation fallback "if removing it ripples through too many signatures."
- **Resolution:** Removal was clean. The `complex_output` param was pure passthrough through exactly three signatures with a single dead consumer (`let _is_complex_out`). Removed it everywhere. No annotation fallback needed.
- **Files modified:** `crates/cintx-cubecl/src/kernels/one_electron.rs`
- **Commit:** `b5487dc`

**2. [Scope clarification] IN-03 const fn NOT applied to the deriv34 host nuclear path**
- **Found during:** Task 3
- **Issue:** A structurally-identical `(li+lj+5)/2+1` nroots ceiling exists at `one_electron.rs:9485` in the Phase-25 HESS-04 deriv3/deriv4 HOST nuclear path. It coincidentally shares the `+5` headroom literal.
- **Resolution:** Left it untouched — it is a different family engine (`HOST_RYS_NROOTS_CEILING_1E`, host-routed Hessian), not a GIAO engine. Applying a `giao_`-named const fn there would be a semantic mislabel and out of IN-03's GIAO scope.
- **Files modified:** none (intentional no-op)
- **Commit:** n/a

**No auto-fixed bugs (Rules 1-3): the plan executed as written; no broken behavior, missing critical functionality, or blocking issues were encountered.**

---

**Total deviations:** 2 (1 WR-05 path decision, 1 IN-03 scope clarification). No scope creep, no architectural changes.

## Threat Model Coverage

- **T-26-13 (Tampering — silent truncation, host guard vs device sizing drift):** mitigated. Task 3 routes the host fail-closed guard, the host nuclear nroots ceiling, AND both host-side device buffer sizings through the same `giao_ovlp_nmax`/`giao_nuc_nmax`/`giao_nuc_nroots` const fns. A one-sided edit can no longer introduce a D-13 truncation; the envelope is single-sourced.
- **T-26-14 (Tampering — positional coupling, is_rinv_center op_kind>=2):** mitigated. Task 2 makes the nuclear-model selection (type-2 atom-sum -Z vs type-1 single rinv center +1) a per-family bool on the dispatch tuple. Adding or reordering a GIAO nuclear family can no longer silently re-point the nuclear-model branch via an ordinal threshold — same anti-coupling class as the OperatorId-shift project memory.

## Threat Flags

None. No new trust boundaries, network endpoints, auth paths, file access, or schema changes. All three edits are internal code-quality refactors on existing host control-plane code; the `caller → eval_raw` numeric surface and the staging-buffer contract are byte-for-byte unchanged (parity stays green).

## Known Stubs

None. All four warnings (WR-04, WR-05, IN-02, IN-03) are fully closed with concrete code, not placeholders. No a01gp guard was reintroduced (`grep -c 'op_name == "a01gp"'` = 0); a01gp continues to ride the normal nuclear-engine path with byte-identity from 26-05.

## Self-Check: PASSED

- FOUND: crates/cintx-cubecl/src/kernels/one_electron.rs (WR-04 chunks_exact not0; IN-02 is_rinv_center tuple bool, op_kind>=2 gone; WR-05 complex_output removed; IN-03 three giao_ const fns + four host call sites)
- FOUND: crates/cintx-cubecl/src/kernels/two_electron.rs (WR-04 chunks_exact not0 in launch_two_electron_giao2e)
- FOUND: .planning/phases/26-group-5-spin-free-giao-nmr-integrals-complex/26-08-SUMMARY.md
- FOUND commit 5a472d2 (Task 1 — fix: WR-04 imaginary-half not0)
- FOUND commit b5487dc (Task 2 — refactor: IN-02 is_rinv_center + WR-05 hint removal)
- FOUND commit 718ebb0 (Task 3 — refactor: IN-03 shared headroom const fns)
- VERIFIED: grep -c 'chunks_exact(2)' = 2 (one_electron) / 1 (two_electron)
- VERIFIED: grep -c 'op_kind >= 2' one_electron.rs = 0; giao_nuc_op type Option<(u32, u32, bool)>
- VERIFIED: grep -c 'complex_output: u32' one_electron.rs = 0
- VERIFIED: grep -c 'const fn giao_' one_electron.rs = 3; headroom-fn occurrences = 10
- VERIFIED: grep -c 'op_name == "a01gp"' one_electron.rs = 0 (no guard reintroduced)
- VERIFIED: cargo build/test -p cintx-cubecl --features cpu green; vendor-gated GIAO 1e parity 11/11, GIAO 2e parity 4/4, all moment parity (20 tests) green
- VERIFIED: .planning/STATE.md and .planning/ROADMAP.md untouched (git status clean)
