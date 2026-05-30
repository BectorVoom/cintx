---
phase: quick
plan: 260530-iiq
subsystem: api
tags: [libcint, int3c1e, general-contraction, nctr, cubecl, oracle-parity, raw-api]

# Dependency graph
requires:
  - phase: "23 (DRV1-03)"
    provides: "int3c1e scalar + ip1 + iprinv launchers and vendor FFI wrappers"
provides:
  - "general-contraction (nctr>1) support for int3c1e scalar + ip1 + iprinv (cart + sph)"
  - "column-major env coefficient transpose at the raw eval_raw boundary (fixes all families' latent nctr>1 raw path)"
  - "first raw (eval_raw / env-based) nctr>1 vendor byte-identity parity test in the suite"
affects: [int3c1e, raw-api, general-contraction, future nctr>1 family work]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "nctr-blocked interleaved output: single dense [di*nctr_i, dj*nctr_j, dk*nctr_k] array, contraction-MAJOR per axis (i_global = ci*nblk_i + i_idx), matching libcint c2s_{cart,sph}_3c2e1"
    - "gradient nctr layout: 3 components outermost (component-leading), interleaved nctr/angular index within each component"
    - "libcint env coefficient block is COLUMN-MAJOR (env[ci*nprim+ip]); cintx Shells are ROW-MAJOR — transpose at the env->Shell boundary"

key-files:
  created:
    - "crates/cintx-oracle/tests/int3c1e_genctr_parity.rs"
  modified:
    - "crates/cintx-cubecl/src/kernels/center_3c1e.rs"
    - "crates/cintx-compat/src/raw.rs"
    - ".planning/todos/completed/wr03-3c1e-grad-nctr-gt1.md (moved from pending)"

key-decisions:
  - "Root cause is the column-major env coeff layout, NOT just output block placement — fixed at the raw boundary so ALL families' raw nctr>1 path is correct, not just 3c1e"
  - "Scalar path shared the same nctr>1 limitation as the gradient launchers and was fixed in the same change"
  - "Device #[cube] kernels left UNCHANGED — fix is host-side block placement + raw coeff transpose only"

patterns-established:
  - "Pattern: confirm libcint nctr>1 block ordering EMPIRICALLY against the vendor before committing the kernel fix (the plan's guessed offset formula was wrong)"
  - "Pattern: a single nctr-blocked interleaved output buffer (contraction-MAJOR) replaces the old accumulate-into-one-block approach"

requirements-completed: []

# Metrics
duration: ~55min
completed: 2026-05-30
---

# Phase quick Plan 260530-iiq: int3c1e General-Contraction (nctr>1) Support Summary

**General-contraction (nctr>1) support for int3c1e scalar + ip1 + iprinv, byte-identical to libcint 6.1.3 (cart+sph) on a non-square fixture — root-caused to a column-major env coefficient transpose bug at the raw API boundary.**

## Performance

- **Duration:** ~55 min
- **Started:** 2026-05-30 (worktree reset to plan base f3fc0b0)
- **Completed:** 2026-05-30
- **Tasks:** 3
- **Files modified:** 3 (+1 todo moved)

## Accomplishments
- Scalar `launch_center_3c1e_typed` now emits a single dense interleaved `[di*nctr_i, dj*nctr_j, dk*nctr_k]` output (contraction-MAJOR per axis) instead of accumulating all `(ci,cj,ck)` columns into one block.
- Both gradient launchers (`launch_center_3c1e_ip1`, `launch_center_3c1e_iprinv`) now contract every `(ck,cj,ci)` column with its own coefficient column and scatter component-leading blocks into a single interleaved output via the new `scatter_3c1e_grad_block` helper.
- Root-caused the real defect: the libcint env coefficient block is COLUMN-MAJOR (`env[ci*nprim+ip]`) but cintx Shells are ROW-MAJOR; the raw env→Shell parse copied verbatim, transposing nctr>1 coefficients for every family. Fixed at `crates/cintx-compat/src/raw.rs`.
- Added the suite's first raw (env-based) nctr>1 vendor byte-identity parity test, covering scalar + ip1 + iprinv on a non-square `i=p(nctr=2), j=d, k=s` fixture.

## Task Commits

1. **Task 1: nctr>1 vendor parity test (RED) + confirm libcint block ordering** - `e404331` (test)
2. **Task 2: nctr-blocked scalar int3c1e + column-major env coeff transpose** - `9778669` (fix)
3. **Task 3: nctr-blocked component-leading gradient launchers + close todo** - `ed42934` (fix)

## Files Created/Modified
- `crates/cintx-oracle/tests/int3c1e_genctr_parity.rs` - New nctr>1 vendor parity test (scalar + ip1 + iprinv, cart + sph), double-gated `#[cfg(all(has_vendor_libcint, feature = "cpu"))]`, with the confirmed libcint block-ordering documented in the header.
- `crates/cintx-cubecl/src/kernels/center_3c1e.rs` - Scalar + both gradient launchers emit nctr-blocked interleaved output; new `scatter_3c1e_grad_block` helper; removed superseded `write_3c1e_grad_staging`.
- `crates/cintx-compat/src/raw.rs` - Transpose column-major env coefficients → row-major Shell at the eval_raw boundary.
- `.planning/todos/completed/wr03-3c1e-grad-nctr-gt1.md` - Moved from pending; body updated with the fix and the scalar-path finding.

## Decisions Made
- The plan and the original WR-03 review framed this as a gradient-only output-placement bug. Empirical confirmation against the vendor (mandated by Task 1) revealed the deeper root cause: the raw env coefficient layout is column-major and cintx interpreted it row-major. The correct, minimal fix transposes at the env→Shell boundary so the fix is consistent across all families and keeps the launchers' existing row-major assumption intact.
- Confirmed the libcint nctr>1 block ordering (contraction-MAJOR, interleaved per axis) from `CINT3c1e_drv` + `c2s_{cart,sph}_3c2e1` and verified element-by-element against the vendor — the plan's guessed `((ck*nctr_j+cj)*nctr_i+ci)*block_len` formula was NOT correct and was not used.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Column-major env coefficient transpose at the raw boundary**
- **Found during:** Task 2 (empirical block-ordering confirmation from Task 1)
- **Issue:** The plan scoped the fix to `center_3c1e.rs` host-side block placement. But the cart output was still wrong (close-but-not-equal at all indices) after the block-placement fix. Element-by-element vendor comparison proved the real defect: libcint stores env contraction coefficients COLUMN-MAJOR (`env[ci*nprim+ip]`, per `CINTprim_to_ctr_0`), while cintx Shells / all launchers are ROW-MAJOR (`coefficients[ip*nctr+ci]`). The raw `eval_raw` env→Shell parse copied the block verbatim, transposing nctr>1 coefficients — a latent bug affecting every family's raw nctr>1 path (no raw nctr>1 parity test existed before this plan).
- **Fix:** Transpose column-major → row-major when constructing `Shell.coefficients` in `crates/cintx-compat/src/raw.rs`. Identity for nctr==1.
- **Files modified:** `crates/cintx-compat/src/raw.rs`
- **Verification:** Scalar genctr parity went from 36/36 mismatches to 0; nctr==1 byte-identity preserved (`int3c1e_ip_parity` 5/5, `center_3c2e_parity` 2/2, `one_electron_grad_both_parity` 6/6, `cintx-compat --lib` 43/43).
- **Committed in:** `9778669` (Task 2 commit)

**2. [Rule 1 - Bug] Scalar path also lacked nctr>1 support (not just gradients)**
- **Found during:** Task 2
- **Issue:** The WR-03 review flagged only the gradient launchers, but `launch_center_3c1e_typed` accumulated every `(ci,cj,ck)` triple into ONE block (`*dst += src`), merging contraction columns the same way.
- **Fix:** Scalar launcher rewritten to write each column triple's block to its own contraction-MAJOR interleaved offset.
- **Files modified:** `crates/cintx-cubecl/src/kernels/center_3c1e.rs`
- **Verification:** scalar `int3c1e_genctr_parity` 0 mismatches (cart+sph).
- **Committed in:** `9778669` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 bugs — root-cause and scope corrections).
**Impact on plan:** Both fixes were necessary for correctness and are required to reach vendor byte-identity. The raw-boundary transpose is slightly broader than the plan's stated single-file scope, but it is the true root cause and is the minimal correct fix; it preserves nctr==1 byte-identity exactly. No scope creep beyond what correctness required.

## Issues Encountered
- The plan's preview-guessed block-offset formula was incorrect; resolved by deriving the ordering directly from libcint's `c2s_{cart,sph}_3c2e1` and confirming element-by-element against the vendor (exactly the empirical confirmation Task 1 mandated).
- Pre-existing unrelated baseline noise on this branch (e.g., `cintx-oracle --lib compare::tests`, `test_f32_int3c2e_sph_parity`) was NOT touched per the plan's note and memory.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- int3c1e now has full general-contraction support across scalar + both gradient operators, vendor byte-identical (cart+sph).
- The raw env→Shell coefficient transpose now makes the eval_raw nctr>1 path correct for ALL families; consider adding nctr>1 raw parity tests for other families (one_electron, 3c2e) to lock in this previously-untested surface.
- Device `#[cube]` kernels remain unchanged, so the on-device GPU paths are unaffected.

---
*Phase: quick 260530-iiq*
*Completed: 2026-05-30*

## Self-Check: PASSED

- Commits e404331, 9778669, ed42934 all present in git history.
- All created/modified files exist on disk; pending todo removed, completed todo present.
- int3c1e_genctr_parity 6/6 ok under the double gate (scalar+ip1+iprinv, cart+sph, 0 mismatches at atol=1e-12).
- No nctr==1 regression: int3c1e_ip_parity 5/5, center_3c2e_parity 2/2, one_electron_grad_both_parity 6/6, cintx-cubecl --lib 280, cintx-compat --lib 43.
