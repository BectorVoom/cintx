---
phase: 26-group-5-spin-free-giao-nmr-integrals-complex
reviewed: 2026-05-31T00:00:00Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - crates/cintx-cubecl/src/kernels/one_electron.rs
  - crates/cintx-cubecl/src/kernels/two_electron.rs
  - crates/cintx-runtime/src/planner.rs
  - crates/cintx-oracle/tests/giao_1e_parity.rs
  - crates/cintx-oracle/tests/giao_2e_parity.rs
findings:
  critical: 0
  warning: 3
  info: 2
  total: 5
status: issues_found
---

# Phase 26: Code Review Report

**Reviewed:** 2026-05-31
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

## Summary

Scope: the phase-26 GIAO gap-closure diff against `8d64fd1` (plans 26-04..26-08) —
the `int1e_a01gp` 0.5 common-factor fix + guard removal, the GIAO `not0`
imaginary-half counting, the explicit per-family `is_rinv_center` dispatch bool, the
shared GIAO headroom `const fn`s, the removal of the inert moment `complex_output`
comptime arg, the full-block chunk staging in `planner.rs`, and the
`giao_2e_parity` → `moment_common` refactor.

The core numeric fixes are sound and well-justified:

- The **a01gp 0.5 factor** (`op_kind == 3`, one_electron.rs:3289) is correct and matches
  the libcint `envs.common_factor *= 0.5` (intor1.c:551/572). The `fam_factor` chain
  still leaves `ia01p` (`op_kind == 2`) at the default `1.0`, which is intentional
  (rank-3 ia01p passes parity) — not a gap.
- The **`is_rinv_center` enumeration** is byte-equivalent to the old `op_kind >= 2`
  threshold (gnuc/ignuc=false; ia01p/a01gp/cg_a11part/giao_a11part=true) and is genuinely
  more robust against table reordering. Verified row-by-row.
- The **`const fn` headroom** (`giao_ovlp_nmax = li+lj+3`, `giao_nuc_nmax = li+lj+5`,
  `giao_nuc_nroots = nmax/2+1`) exactly reproduces the prior inline arithmetic, is used at
  all three host sites (buffer sizing lines 3100/3729 + guard 8745 + nroots 8825), and
  matches the inline `#[cube]` kernel bodies at lines 2795 (`+3`) and 3269 (`+5`). No drift.
- The **`complex_output` comptime-arg removal** is clean: no dangling references remain
  across `run_1e_moment_device`, the macro, the 5-arm `run_1e_moment_on_backend`, or the
  launcher call site.
- The **`moment_common` refactor** preserves gate semantics: for `RTOL = 0.0` both the
  old per-`ref`-divisor mismatch test and the shared `atol + rtol*|ref|` test reduce to
  `diff <= atol`. The shared `count_mismatches`/`assert_any_nonzero`/`ncart`/`nsph`
  helpers exist with matching signatures (moment_common.rs:152/170/46/51).

Three WARNINGs concern second-order correctness/contract issues that the new code
introduces or now relies on but does not verify, plus two INFO items.

## Warnings

### WR-01: Chunked complex/GIAO families over-count `not0` by a factor of `chunk_count`

**File:** `crates/cintx-runtime/src/planner.rs:280` (with `crates/cintx-cubecl/src/kernels/one_electron.rs:8519` and `crates/cintx-cubecl/src/kernels/two_electron.rs:2528`)
**Issue:** The gap-closure `staging_elements_for_chunk` now hands the **full**
interleaved block to *every* chunk for `complex_interleaved` families (correct — fixes the
BufferTooSmall regression). But the monolithic GIAO launchers
(`write_giao_complex_staging`, `launch_two_electron_giao2e`) compute `not0` over the
**entire full block** and return it once per chunk, and `evaluate` accumulates it via
`metrics.observe_not0(io.not0())` at planner.rs:280 (`RunMetrics::observe_not0` does a
`saturating_add`). So when `memory_limit_bytes` forces `chunk_count = N`, the reported
`not0` becomes `N × (true full-block not0)`. Before this fix the family was inoperable
under chunking (returned `BufferTooSmall`), so the over-count never surfaced; it is newly
reachable now. `not0` is a libcint-compatibility signal (non-zero/early-exit work report),
so an N× inflation is an observable contract value that is now wrong for any chunked
complex family. The new test `evaluate_giao_complex_family_survives_memory_chunking`
asserts only `chunk_count > 1` and `fallback_reason`, never `not0`, so it does not catch
this.
**Fix:** Compute the full-block `not0` exactly once for `complex_interleaved` families
(e.g. only on the first chunk, or set the final `not0` to the single full-block value
rather than summing per chunk), and add a chunked-vs-single assertion:
```rust
assert_eq!(chunk_stats.not0, baseline_stats.not0,
    "chunked complex family must not inflate not0 by chunk_count");
```

### WR-02: `staging_elements_for_chunk` doc claims an upfront fail-closed guard the workspace estimate does not provide for complex families

**File:** `crates/cintx-runtime/src/planner.rs:351-352` (claim) vs `crates/cintx-runtime/src/planner.rs:426-433` (estimate)
**Issue:** The new doc comment states: *"If the full block cannot fit, the upfront
workspace check fails closed with a typed `MemoryLimitExceeded` (no partial write)."* That
guarantee is not real. `estimate_workspace_request` (lines 426-433) sizes
`output_bytes = output_elements * component_multiplier * size_of::<f64>()` and never
applies the `complex_output` 2× multiplier — only `build_output_layout` (lines 311-318)
doubles `staging_elements`. So `required_bytes`, the `memory_limit_bytes` check, and the
chunk plan all **undercount complex families by 2×**, while the per-chunk staging
allocation (`try_alloc_staging(required_elements)`, line 265) requests the full doubled
block independently of the limit. The doubled staging is therefore not gated by the
upfront workspace check the comment cites. (No memory-unsafety: staging is fallibly
allocated via `try_reserve_exact`, so OOM yields a typed `HostAllocationFailed`, not UB —
but it is not the cited `MemoryLimitExceeded` path, and `memory_limit_bytes` does not
actually bound the complex staging.)
**Fix:** Either fold the complex multiplier into `estimate_workspace_request` so the
memory-limit check and chunk plan reflect the true doubled footprint, or correct the doc
comment to state that the complex doubling is bounded by fallible host allocation
(`HostAllocationFailed`), not by the `memory_limit_bytes` / `MemoryLimitExceeded` upfront
check.

### WR-03: WR-01 chunking test asserts survival but not output equality, and its docstring overclaims

**File:** `crates/cintx-runtime/src/planner.rs:1308-1309, 1395-1407`
**Issue:** The test docstring (lines 1308-1309) says the chunked run *"matches the
single-chunk run"*, but the test never compares any output value or `not0` between
`baseline_stats` and `chunk_stats` — it only checks `chunk_count > 1` and
`fallback_reason == Some("memory_limit")`. Combined with the `MockBackend` (which writes
only `staging[0] = 1.0`), the test proves "no `BufferTooSmall`" but provides **zero**
evidence that a monolithic full-block writer driven once per chunk yields a correct,
non-duplicated result — exactly the regression class WR-01 describes.
**Fix:** Add value/`not0` parity assertions between the baseline (single-chunk) and chunked
runs, or soften the docstring to "survives chunking without `BufferTooSmall`" and note
that output-equality under chunking is covered by the compat `eval_raw` scatter tests, not
here.

## Info

### IN-01: `assert_any_nonzero` deduplicated in 2e but left as a local copy in 1e

**File:** `crates/cintx-oracle/tests/giao_1e_parity.rs:96-101`
**Issue:** The WR-03 dedup pulled `ATOL`/`RTOL`/`count_mismatches`/`ncart`/`nsph`/
`assert_any_nonzero` into `moment_common`, and `giao_2e_parity.rs` now imports the shared
`assert_any_nonzero` (moment_common.rs:170). `giao_1e_parity.rs` still keeps a **local**
copy (identical `1e-14` threshold) instead of importing the shared one, so the same
"single source of truth" rationale the 2e refactor cites is only half applied. Harmless
(thresholds match) but a latent drift point.
**Fix:** Drop the local `assert_any_nonzero` in `giao_1e_parity.rs` and add it to the
`use super::moment_common::{...}` import, matching `giao_2e_parity.rs`.

### IN-02: a01gp `fam_factor` rationale comment carries the superseded (wrong) diagnosis

**File:** `crates/cintx-cubecl/src/kernels/one_electron.rs:3290-3294`
**Issue:** The comment labels the missing-0.5 scale fix as *"the 26-02 ket-derivative
double-count"*, but the actual root cause is a missing family-level
`common_factor *= 0.5` (a uniform scale), not a ket-derivative double-count in the
`g2 = D_J + D_I` tensor — the latter was the earlier (incorrect) 26-02 hypothesis quoted
verbatim from the now-removed deferral note (and the removed-note block at
giao_1e_parity.rs:289-297 itself acknowledges the gout already matched verbatim). Carrying
the superseded diagnosis inline risks future readers re-chasing the wrong tensor path.
**Fix:** Reword to: "uniform ~2× from a missing family `common_factor *= 0.5`
(intor1.c:551/572); the s-table/gout already matched verbatim — the 26-02
'ket-derivative double-count' hypothesis was wrong."

---

_Reviewed: 2026-05-31_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
