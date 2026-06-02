---
phase: 29-group-4-relativistic-spin-operator-integrals-spinor
plan: 03
subsystem: transform
tags: [spinor, c2spinor, 2e, si_2e, sf_2e, pauli-sigma-mix, zcopy_iklj, kappa-fixture, libcint, relativistic, wave-2]

# Dependency graph
requires:
  - phase: 29 (plan 02)
    provides: 7 1e Group-4 σ launcher arms wired byte-identical, int1e_sigma rank=3, sigma_1e/sigma_1e_nuc engines
  - phase: 29 (plan 01)
    provides: cart_to_spinor_si_2di 1e imaginary-ket transform, 7 1e Group-4 manifest rows
  - phase: 28-gap-b2-c2s-si-sigma-p
    provides: cart_to_spinor_si_2d + cart_to_spinor_sf_2d, build_kappa_spinor_fixture, build_heavy_atom_spinor_fixture
  - phase: 12-real-spinor-transform
    provides: cart_to_spinor_sf_4d 2e two-stage skeleton, apply_2d_spinor_zf, spinor_len, interleaved-complex layout
provides:
  - Full 2e cart→spinor si/sf transform suite (cart_to_spinor_si_2e1/2e1i/2e2/2e2i + sf_2e1/sf_2e2) in c2spinor.rs
  - apply_2d_spinor_zi — the electron-2 si σ-mix (2×2 Pauli σ·n expansion on four complex gx/gy/gz/g1 blocks), transcribed verbatim from libcint cart2sph.c:4118-4186
  - build_kappa_spinor_2e_fixture (D-02) — 4 spinor shells, non-square (2,6,2,4), GT/LT kappa mix, nctr>1
affects: [29-04, 29-group-4-wave-3, 30-giao-sigma, 31-breit-gaunt]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "2e si transform suite = per-electron split of the cart_to_spinor_sf_4d two-stage skeleton: electron-1 (cart→opij <ik|lj>) loops (l,k) cart slices through the 1e si_2d/si_2di/sf_2d transforms (which own the KET→BRA transpose); electron-2 (opij→fijkl) extracts per-(i_sp,j_sp) complex [nck*ncl] slices, folds via apply_2d_spinor_zi/zf, zcopy_iklj-stores"
    - "electron-2 σ-mix (apply_2d_spinor_zi) consumes FOUR complex blocks (vs zf's one) and applies the verbatim 2×2 Pauli σ·n expansion [[1+iz, y+ix],[-y+ix, 1-iz]] before the bra1 fold; the ket1 step is identical to the zf path"
    - "imaginary-ket 2e variants (2e1i/2e2i) i-rotate (re,im)→(-im,re) at the a_ket store boundary, mirroring cart_to_spinor_iket_si vs cart_to_spinor_si"

key-files:
  created: []
  modified:
    - crates/cintx-cubecl/src/transform/c2spinor.rs
    - crates/cintx-oracle/src/fixtures.rs

key-decisions:
  - "Built the 6-fn suite as per-electron functions (electron-1 producing opij, electron-2 consuming it), NOT a single fused 4d fn — matches the libcint c2s_si_2e1/2e2 driver split and lets Wave-3 launchers pair electron-1×electron-2 transforms per family (spsp1=si_2e1+sf_2e2, spsp1spsp2=si_2e1+si_2e2, ssp1ssp2=si_2e1i+si_2e2i)"
  - "apply_2d_spinor_zi transcribed verbatim from cart2sph.c:4118-4186 (v11=g1R-gzI / v11I=g1I+gzR / v12=gyR-gxI / v21=-gyR-gxI / v22=g1R+gzI, accumulated caR*v11R+caI*v11I+cbR*v21R+cbI*v21I etc.); the ket1 step REUSES the existing apply_ket1_block_all_kappa (identical to the zf path — σ-mix lives entirely in bra1)"
  - "sf_2e1/sf_2e2 are exact extractions of cart_to_spinor_sf_4d's two stages — proven by a unit test asserting split==fused byte-for-byte on a non-square p/d/s/p GT/LT quartet"
  - "build_kappa_spinor_2e_fixture: i p kappa=+1 LT (di=2) + j d kappa=−1 GT (dj=6) + k s kappa=−1 GT (dk=2) + l p kappa=−1 GT (dl=4) → dims (2,6,2,4) all distinct (non-square), GT/LT mix, shell-i nctr=2 column-major coeff. Distinct centers for nonzero 2e blocks. Element/kappa = Claude's discretion per D-02"
  - "This plan delivers compiling structural code ONLY — byte-identity of the 2e suite is the 29-04 [BLOCKING] micro-test (D-03). No families wired, no oracle_covered flips, no manifest edits here"

patterns-established:
  - "Pattern 1: split-electron 2e spinor transform — clone the fused sf_4d skeleton into reusable per-electron fns so Wave-3 can compose electron-1×electron-2 transform pairs per family"
  - "Pattern 2: the σ-mix is bra1-only — apply_2d_spinor_zi differs from apply_2d_spinor_zf solely in the bra1 fold (4 blocks + Pauli expansion vs 1 scalar block); ket1 is shared"

requirements-completed: [REL-03, REL-04]

# Metrics
duration: 35min
completed: 2026-05-31
---

# Phase 29 Plan 03: Wave-2 2e Transform Foundation Summary

**Built the full 2e cart→spinor transform suite — `cart_to_spinor_si_2e1/2e1i/2e2/2e2i` + `sf_2e1/sf_2e2` — by splitting the existing `cart_to_spinor_sf_4d` two-stage skeleton into reusable per-electron functions, transcribed the genuinely-novel electron-2 si σ-mix `apply_2d_spinor_zi` (the 2×2 Pauli σ·n expansion on four complex blocks) verbatim from libcint cart2sph.c:4118-4186, and added the D-02 `build_kappa_spinor_2e_fixture` (4 spinor shells, non-square (2,6,2,4), GT/LT kappa mix, nctr>1). All structural code compiles; the 2e suite's byte-identity gate is the 29-04 [BLOCKING] micro-test.**

## Performance

- **Duration:** ~35 min
- **Completed:** 2026-05-31
- **Tasks:** 3
- **Files modified:** 2 (0 created, 2 modified) — +840 lines

## Accomplishments

- **Task 1 — `apply_2d_spinor_zi` (the D-03-risk piece):** Added the si sibling of `apply_2d_spinor_zf`. Transcribed `a_bra1_cart2spinor_zi` (cart2sph.c:4118-4186) verbatim: the σ·n 2×2 Pauli expansion `[[1+iz, y+ix], [-y+ix, 1-iz]]` over the four complex `gx/gy/gz/g1` blocks — `v11R=g1R-gzI; v11I=g1I+gzR; v12R=gyR-gxI; v12I=gyI+gxR; v21R=-gyR-gxI; v21I=-gyI+gxR; v22R=g1R+gzI; v22I=g1I-gzR;` — accumulated into α/β half-blocks with the four `caR*v11R + caI*v11I + cbR*v21R + cbI*v21I` (and α-I, β-R, β-I) lines. Built `apply_bra1_zi_block_all_kappa` + `apply_bra1_zi_block` as the kappa-dispatching fold; the ket1 step reuses the existing `apply_ket1_block_all_kappa` (identical to the zf path — the σ-mix lives entirely in bra1). The `apply_2d_spinor_zf` sf sibling is untouched.
- **Task 2 — 6-fn 2e si/sf suite (D-01):**
  - **Electron-1** (`si_2e1`, `si_2e1i`, `sf_2e1`): cart → `opij` (`<ik|lj>` ordered). Loops `(l_cart, k_cart)`, slices the `nci*ncj` kl-block, calls the 1e 2D transform per slice. si variants fold the 4 σ-tensor cart blocks via `cart_to_spinor_si_2d` (real ket) / `cart_to_spinor_si_2di` (imaginary ket) — both own the KET→BRA transpose internally; sf uses `cart_to_spinor_sf_2d` (= sf_4d Step 1). Shared `cart_to_spinor_si_2e1_impl` with an `imaginary_ket` flag.
  - **Electron-2** (`si_2e2`, `si_2e2i`, `sf_2e2`): `opij` → `staging` via `zcopy_iklj` store `staging[(((l_sp*dk+k_sp)*dj+j_sp)*di+i_sp)*2]`. For each `(j_sp, i_sp)` extracts the complex `[nck*ncl]` slice(s), folds via `apply_2d_spinor_zi` (si, 4 blocks) / `apply_2d_spinor_zf` (sf, 1 block), then stores. Imaginary variant i-rotates `(re,im)→(-im,re)` at the store. Shared `cart_to_spinor_2e2_impl` with an `Electron2Kind` selector + `imaginary_ket` flag.
  - **Guards** cloned from sf_4d: `expected_cart`/`opij block_len` → `ChunkPlanFailed`; `required = di*dj*dk*dl*2` → `BufferTooSmall`; NO writes before guards pass (T-29-05 mitigated). ALL sizing via `spinor_len` — no literal `4l+2` in any new fn.
  - **Tests:** `sf_2e_split_matches_fused_4d` proves `sf_2e1+sf_2e2 == cart_to_spinor_sf_4d` byte-for-byte on a non-square p/d/s/p GT/LT quartet; `si_2e_suite_nonsquare_nonzero_finite` runs the full si path on a non-square GT/LT quartet (finite, nonzero); `si_2e2i_is_i_rotation_of_si_2e2` confirms the imaginary variant is exactly multiply-by-i. 45/45 c2spinor lib tests green.
- **Task 3 — `build_kappa_spinor_2e_fixture` (D-02):** 4 spinor shells forming a 2-electron quartet `(i,j,k,l)` on 4 distinct centers (nonzero 2e blocks): i p kappa=+1 (LT, di=2, **nctr=2** column-major coeff), j d kappa=−1 (GT, dj=6), k s kappa=−1 (GT, dk=2), l p kappa=−1 (GT, dl=4). Spinor dims (2,6,2,4) all distinct → **non-square**; GT/LT mix exercises both `spinor_len` branches (2l and 2l+2), not just 4l+2; shell-i nctr>1 catches the coeff transpose. A unit test asserts every D-02 hard constraint; `build_heavy_atom_spinor_fixture` (Phase-28, already present) is the secondary realism cross-check, asserted well-formed/finite.

## Task Commits

1. **Task 1: apply_2d_spinor_zi (2×2 Pauli σ-mix)** — `5893840` (feat)
2. **Task 2: 2e si/sf transform suite (D-01)** — `c94332d` (feat)
3. **Task 3: build_kappa_spinor_2e_fixture (D-02)** — `c0d2090` (feat)

## Files Created/Modified

- `crates/cintx-cubecl/src/transform/c2spinor.rs` — `apply_2d_spinor_zi` + bra1_zi helpers; the 6-fn 2e si/sf suite (`cart_to_spinor_si_2e1/2e1i/2e2/2e2i` + `sf_2e1/sf_2e2`) + shared `_impl`/`store_2e2_block` helpers; 3 new unit tests
- `crates/cintx-oracle/src/fixtures.rs` — `build_kappa_spinor_2e_fixture` (D-02) + 2 D-02 constraint tests

## Decisions Made

- **Per-electron split, not a fused 4d fn.** libcint's drivers split `c2s_si_2e1` (electron 1) and `c2s_si_2e2` (electron 2), and Wave-3 families pair them differently (e.g. `spsp1`=si_2e1+sf_2e2 vs `spsp1spsp2`=si_2e1+si_2e2 vs `ssp1ssp2`=si_2e1i+si_2e2i per RESEARCH §2e map). Building 6 composable per-electron fns (rather than one monolith) matches the driver structure and lets each Wave-3 launcher select its electron-1×electron-2 transform pair.
- **The σ-mix is bra1-only.** `apply_2d_spinor_zi` differs from `apply_2d_spinor_zf` solely in the bra1 fold — four complex blocks + the Pauli expansion vs one scalar block. The `a_ket1_cart2spinor` step is byte-identical, so the existing `apply_ket1_block_all_kappa` is reused verbatim. This minimizes the novel surface to exactly the 2×2 Pauli math the 29-04 micro-test must pin.
- **sf_2e1/sf_2e2 are extractions, proven equal to the fused path.** Rather than re-derive, the sf per-electron fns reuse the exact bodies of `cart_to_spinor_sf_4d`'s two stages; the `sf_2e_split_matches_fused_4d` test guarantees byte-equality so any future refactor of either path is caught.

## Deviations from Plan

None requiring a Rule-4 stop. The plan was executed as written; line-number references in the plan (`sf_4d` at L1199, `apply_2d_spinor_zf` at L1353, `si_2d` at L673) were stale — the live code is at L1235/L1371/L640 — but the functions and structure matched the plan exactly, so this was a no-op navigation adjustment, not a deviation.

## Issues Encountered / Deferred

- **Pre-existing out-of-scope test failure** (NOT a regression): `cintx-oracle` lib test `fixtures::tests::unstable_source_fixtures_require_opt_in` fails under `--features cpu`. Verified by reverting the 29-03 `fixtures.rs` change and re-running on a clean tree — it fails identically (the test needs the `unstable-source-api` feature, which the plan's `--features cpu` verification profile does not enable). Logged to `deferred-items.md`; out of scope per the SCOPE BOUNDARY rule (not caused by, nor fixable within, this plan's changes).

## Known Stubs

None. All 6 transform fns + `apply_2d_spinor_zi` + the fixture are fully implemented and exercised by unit tests. The Wave-3 launcher wiring, vendor FFI, manifest rows, and `oracle_covered` flips are explicitly OUT of scope for this plan (29-06 wires families; 29-04 proves the transforms) — this is the planned Wave-2 foundation state, not an incomplete deliverable. Byte-identity of the 2e suite is the 29-04 [BLOCKING] micro-test (D-03), as the plan's `<verification>` note states.

## Threat Flags

None. The two registered threats are addressed: T-29-05 (2e transform output buffer Tampering/DoS, `mitigate`) — every new transform applies the sf_4d-cloned guards (`ChunkPlanFailed` on cart/opij undersize, `BufferTooSmall` on staging undersize) before any write (OOM-safe stop, no partial writes); T-29-06 (wrong Pauli expansion/stride, `accept`) — transcribed verbatim from cart2sph.c:4118-4186 and gated by the 29-04 micro-test at atol=1e-12 before any family wires on. No new network/auth/file-access surface (this is host-side numerical transform code only).

## Next Phase Readiness

- **Wave-2 foundation delivered** — REL-03/04 transforms + D-02 fixture exist, compile, size via spinor_len, own the KET→BRA transpose, and wire the σ-mix. The 6 fns are public and composable for Wave-3 launcher pairing.
- **29-04 (next, [BLOCKING]):** drive the thinnest 2e si family `int2e_spsp1_spinor` (si_2e1 + sf_2e2) against `vendor_int2e_spsp1_spinor` on `build_kappa_spinor_2e_fixture` at atol=1e-12 — the D-03 gate that pins the 2e layout before any Wave-3 family wires onto it.
- No blockers introduced by this plan.

## Self-Check: PASSED

- Modified files exist on disk: `crates/cintx-cubecl/src/transform/c2spinor.rs`, `crates/cintx-oracle/src/fixtures.rs` (both tracked, both built green under `--features cpu`).
- All 3 task commits present in git history: `5893840`, `c94332d`, `c0d2090`.
- All 6 suite fns + `apply_2d_spinor_zi` + `build_kappa_spinor_2e_fixture` grep-confirmed present.
- `cargo build -p cintx-cubecl --features cpu` and `cargo build -p cintx-oracle --features cpu` exit 0.
- `cargo test -p cintx-cubecl --features cpu --lib c2spinor` = 45 passed / 0 failed.

---
*Phase: 29-group-4-relativistic-spin-operator-integrals-spinor*
*Completed: 2026-05-31*
