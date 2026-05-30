---
phase: 24-group-3-position-multipole-moment-integrals
plan: 02
subsystem: kernels
tags: [cubecl, kernel, moment, gauge-origin, manifest, parity, multipole]

# Dependency graph
requires:
  - phase: 24-group-3-position-multipole-moment-integrals
    plan: 01
    provides: 36 vendor FFI wrappers + rank-parameterized vendor_parity + 4 RED moment parity scaffolds
  - phase: 22-gauge-origin-env-slot-gap-a-ptr-common-orig
    provides: PTR_COMMON_ORIG live read in eval_raw + build_h2o_sto3g_common_orig fixture
provides:
  - 42 manifest entries for the 14 Cluster-A families (r/rr/rrr/rrrr/r2/r4/z/zz + 6 _origj) x {cart,sph,spinor}
  - RawApiId consts for all Cluster-A families (INT1E_{R,RR,RRR,RRRR,R2,R4,Z,ZZ}_* + _ORIGJ_*)
  - one_electron_moment_kernel — ONE parameterized #[cube] kernel for all Cluster-A families
  - cross_center_non_square_shell_pair + vendor_parity_at scaffold helpers (_origj non-triviality fix)
affects: [24-03, 24-04, 24-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Parameterized moment #[cube] kernel: build overlap G-tensor to lj+moment_order ket levels (ng[1] headroom), then per-axis moment ladder m_p = Sum_t C(p,t) drj^(p-t) overlap[jx+t] (closed-form of libcint CINTx1j_1e); emit verbatim gout via canonical base-3 digit decomposition"
    - "Origin-source branch (D-02) realized PURELY host-side: launcher passes drj = rj - origin (origin = common_orig for base, rj for _origj so drj=0 = libcint G1E_R_J pointer-shift)"
    - "_origj cross-center parity gate: _origj measures position relative to the ket center, so same-center even-moment blocks are identically zero (vendor included); use a cross-center non-square ket (H1-1s x O-2p) to keep the comparison substantive AND non-square"

key-files:
  created: []
  modified:
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-ops/src/generated/api_manifest.csv
    - crates/cintx-compat/src/raw.rs
    - crates/cintx-cubecl/src/kernels/one_electron.rs
    - crates/cintx-oracle/tests/moment_common.rs
    - crates/cintx-oracle/tests/moment_r_parity.rs
    - crates/cintx-oracle/tests/moment_low_parity.rs
    - crates/cintx-oracle/tests/moment_high_parity.rs

key-decisions:
  - "Per-axis moment value uses the closed-form binomial m_p = Sum_{t=0..p} C(p,t) drj^(p-t) overlap[jx+t] — the analytic equivalent of libcint's repeated CINTx1j_1e (g1e.c:453). Verified byte-identical against vendor for every family, so it reproduces the verbatim gout order without hand-transcribing 81 s[k] lines"
  - "Tensor families (rr/rrr/rrrr) emit components via canonical base-3 digit decomposition (k -> per-axis powers). Confirmed against libcint source that the gout index = canonical nested ordering (s[k] g-block power per axis matches the digit split); r2/r4 are the documented trace contractions of the rr/rrrr s-lists"
  - "_origj = drj=0 branch (origin = ket center rj). libcint's G1E_R_J is a pure ket+1 pointer shift, which is exactly the moment ladder with drj=0"
  - "[Rule 3] _origj families need a CROSS-center non-square block; the 24-01 scaffold's hardcoded same-center (0,2) gives identically-zero _origj even-moment integrals (vendor too). Added cross_center_non_square_shell_pair (3,2) + vendor_parity_at"

patterns-established:
  - "One #[cube] moment kernel with #[comptime] (op_mode, moment_order, rank) selected via a host match in run_1e_moment_device; CubeCL cannot pass comptime args dynamically so the 8 valid Cluster-A combinations are enumerated"
  - "moment_origj_parity_test! macro variant routes _origj tests through the cross-center block while base families stay on (0,2)"

requirements-completed: [MOM-01, MOM-02, MOM-03]

# Metrics
duration: 70min
completed: 2026-05-30
---

# Phase 24 Plan 02: Cluster A Moment Kernel Summary

**One parameterized `#[cube]` moment kernel banks MOM-01/02/03 — all 14 overlap-derived position-tensor families (r/rr/rrr/rrrr/r2/r4/z/zz + 6 _origj) match vendored libcint 6.1.3 at atol=1e-12 (cart+sph) on a non-square block, walking the rank 1→81 ramp through a single kernel with a host-side origin-source branch (common gauge origin vs ket center).**

## Performance

- **Duration:** ~70 min
- **Completed:** 2026-05-30
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments
- Registered 42 manifest entries (14 families × {cart,sph,spinor}) with EXACT component_rank (r=3, rr=9, rrr=27, rrrr=81, r2/r4/z/zz=1; _origj mirrors base); rrr/rrrr `_origj` correctly ABSENT (OQ-3); spinor forms `oracle_covered=false` → UnsupportedApi (D-09). `cargo build -p cintx-ops` auto-regenerates `api_manifest.{rs,csv}` and manifest-audit derives both sides from the lock.
- Added RawApiId consts for every Cluster-A family + `_origj` variant, string-exact to the manifest symbols.
- Implemented ONE parameterized `#[cube] one_electron_moment_kernel`: builds the overlap G-tensor to `lj + moment_order` KET levels (headroom on `ng[1]`, NOT the bra — D-07), computes per-axis moment-power ladders via the closed-form binomial of libcint's repeated `CINTx1j_1e`, and emits components in the verbatim libcint gout order (canonical base-3 nesting for tensors; documented trace contractions for r2/r4; single z-component for z/zz).
- Realized the origin-source branch (D-02) entirely host-side: the launcher passes `drj = rj - origin` with `origin = env[PTR_COMMON_ORIG]` (base) or the ket center `rj` (`_origj`, so `drj = 0`, matching libcint's `G1E_R_J` pointer shift). No new env-read code — `common_orig` is already read live in `eval_raw` (D-06).
- All 14 vendor parity tests GREEN: `moment_r_parity` 2/2, `moment_low_parity` 8/8, `moment_high_parity` 4/4, byte-identical to vendored libcint 6.1.3 at atol=1e-12 (cart+sph), under the `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1` double gate. Non-square block enforced (transpose gate). No regression: cubecl `--lib` 280/280, compat `--lib` 43/43.

## Task Commits

1. **Task 1 — r/z/zz (+_origj) via parameterized moment kernel** — `af88c1e` (feat)
2. **Task 2 — rr/rrr/rrrr/r2/r4 (+rr/r2/r4 _origj) high-rank + trace contractions** — `01c4ab8` (feat)

## Decisions Made
- **Closed-form moment ladder instead of hand-transcribing s[k]:** the per-axis moment value is `m_p[jx] = Σ_{t=0..p} C(p,t)·drj^(p-t)·overlap[jx+t]`, the analytic equivalent of libcint's repeated `CINTx1j_1e` (g1e.c:453). For tensor families the gout `s[k]` factorizes as (per-axis power product) and the component index is the canonical base-3 nesting (verified against intor1.c). This reproduces the verbatim gout order byte-for-byte (proven by atol=1e-12 vendor parity) without transcribing the rank-81 s-list character-for-character — the parity gate is the proof of faithfulness.
- **`_origj` = `drj=0`:** libcint's `G1E_R_J(f,g) = (f = g + g_stride_j)` is a pure ket-level+1 pointer shift, which is exactly the moment ladder with `drj=0`. So the SAME kernel serves base and `_origj` — only the host-passed `drj` differs (D-02 "kernel-side coordinate choice").

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking scaffold issue] `_origj` parity tests need a cross-center non-square block**
- **Found during:** Task 2 (running moment_low_parity / moment_high_parity)
- **Issue:** The 24-01 scaffold hardcoded `non_square_shell_pair() = (0,2)` (O-1s × O-2p, SAME center O) for every family. `_origj` families measure position relative to the KET basis center, so on a same-center block every even-moment `_origj` integral is identically zero — confirmed the VENDOR also returns all-zero there. The cintx output was therefore correct (matching vendor's zero), but `assert_any_nonzero` tripped, blocking the parity assert.
- **Fix:** Added `cross_center_non_square_shell_pair()` → `(3,2)` (H1-1s × O-2p: non-square AND cross-center) and a `vendor_parity_at(rank, shls_pair, ...)` helper; routed all `_origj` tests through it via a `moment_origj_parity_test!` macro variant. Base families stay on the original `(0,2)` block. The block remains strictly non-square, so the D-07 transpose gate is preserved.
- **Files modified:** crates/cintx-oracle/tests/moment_common.rs, moment_r_parity.rs, moment_low_parity.rs, moment_high_parity.rs
- **Verification:** all 14 `_origj` + base parity tests now GREEN at atol=1e-12; verified the cross-center pair gives genuinely-nonzero vendor output for rr_origj/zz_origj.
- **Committed in:** `01c4ab8` (Task 2 commit)

**Total deviations:** 1 auto-fixed (1 blocking scaffold issue). No architectural changes. No kernel math defect — the kernel was correct from the start; the deviation was purely a test-fixture shell-pair selection.

## Threat Surface
No new trust boundaries introduced. T-24-02-01 (non-finite gauge origin) is mitigated by the pre-existing `validate_common_orig_env_params` + bounds-guarded env read (no new validation gap — Cluster A is the first consumer of an already-validated slot). T-24-02-02 (component-rank truncation) is mitigated by exact per-family component_rank + the per-component non-square parity test catching any missing/permuted component. T-24-02-03 (rank-81 OOM) is accepted/deferred to Phase 25 (D-03); the gate corpus does not stress allocation.

## Known Stubs
None. All 14 Cluster-A families are fully wired (manifest + RawApiId + kernel + vendor parity), no placeholders. Spinor forms are intentional `UnsupportedApi` (D-09), registered for surface completeness; resolved when a relativistic consumer needs them (carry-forward, not a stub).

## Self-Check: PASSED

- Created files: none (all modifications to existing files).
- Modified files exist: crates/cintx-cubecl/src/kernels/one_electron.rs, crates/cintx-compat/src/raw.rs, crates/cintx-ops/generated/compiled_manifest.lock.json (all confirmed present).
- Commits present in git history: `af88c1e`, `01c4ab8` (both FOUND).
- Parity gate: moment_r_parity 2/2, moment_low_parity 8/8, moment_high_parity 4/4 GREEN under the vendor double-gate.

---
*Phase: 24-group-3-position-multipole-moment-integrals*
*Completed: 2026-05-30*
