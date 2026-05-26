# Phase 21 Deferred Items

Out-of-scope discoveries logged during execution. NOT fixed in the discovering plan.

## From 21-06 (int3c2e_ip1 derivative kernel)

### D-21-06-A: `xtask manifest-audit --check-lock` oracle-coverage gate fails (PRE-EXISTING, phase-wide)

- **Discovered during:** 21-06 oracle_covered flip verification.
- **Status:** Pre-existing — NOT caused by 21-06. 21-06 actually REDUCES the
  uncovered count (flips `int3c2e_ip1_cart` + `int3c2e_ip1_sph` from false→true).
- **Detail:** `manifest-audit --check-lock` reports 37 `uncovered_stable_entries`,
  i.e. stable manifest rows with `oracle_covered: false`. These are the Phase 21
  gradient operators shipped by earlier waves (21-01..05) whose `oracle_covered`
  flags have not yet been flipped: `int2e_ip1_{cart,sph,spinor}`,
  `int1e_ipovlp_*`, `int1e_ipkin_*`, `int1e_ipnuc_*`, `int1e_iprinv_*`,
  `int1e_ecp_iprinv_*`, and their `cint*` legacy wrappers, plus the R5-exempt
  `int3c2e_ip1_spinor`.
- **Why deferred:** This is a phase-completion reconciliation task. The
  `--check-lock` audit is not a per-plan merge blocker during an in-progress
  phase; it is reconciled once every gradient operator has oracle coverage (or a
  documented exemption). 21-06 only owns the int3c2e_ip1 family.
- **Spinor exemption note:** `int3c2e_ip1_spinor` stays `oracle_covered: false`
  intentionally — gradient spinor is `UnsupportedApi` (R5 / D-03), exactly
  mirroring the established `int2e_ip1_spinor` precedent from 21-05.
- **Also pre-existing:** `profile_scope_mismatch.lock_approved_extra:
  ['unstable-source']` — unrelated to this plan.

### D-21-06-B: pre-existing unused-import warnings (out of scope)

- `crates/cintx-cubecl/src/kernels/f12.rs:1686,1743,1793` — unused `OperatorId`
  import in `#[cfg(test)]` modules.
- `crates/cintx-cubecl/src/kernels/unstable.rs:2630,2756-2758,2916,2920` —
  unused variables (`dlj`, `gx`, `gy`, `gz`, `nmax`, `dj`).
- Not in 21-06's touched files; left untouched per the scope boundary.
