---
phase: 26-group-5-spin-free-giao-nmr-integrals-complex
reviewed: 2026-05-31T00:00:00Z
depth: standard
files_reviewed: 13
files_reviewed_list:
  - crates/cintx-compat/src/raw.rs
  - crates/cintx-cubecl/src/kernels/f12.rs
  - crates/cintx-cubecl/src/kernels/one_electron.rs
  - crates/cintx-cubecl/src/kernels/two_electron.rs
  - crates/cintx-ops/build.rs
  - crates/cintx-ops/src/resolver.rs
  - crates/cintx-oracle/build.rs
  - crates/cintx-oracle/src/compare.rs
  - crates/cintx-oracle/src/vendor_ffi.rs
  - crates/cintx-oracle/tests/giao_1e_parity.rs
  - crates/cintx-oracle/tests/giao_2e_parity.rs
  - crates/cintx-oracle/tests/giao_complex_roundtrip.rs
  - crates/cintx-runtime/src/planner.rs
findings:
  critical: 1
  warning: 5
  info: 3
  total: 9
status: issues_found
---

# Phase 26: Code Review Report

**Reviewed:** 2026-05-31
**Depth:** standard
**Files Reviewed:** 13
**Status:** issues_found

## Summary

Phase 26 adds the spin-free GIAO/NMR families (GIAO-01 1e: 11 families; GIAO-02 2e: g1/ig1/gg1/g1g2) on top of the FND-03 complex-output foundation. The complex plumbing — manifest `complex_output` flag → planner `build_output_layout` 2× sizing + `complex_interleaved` → `CompatDims` → `complex_values()` view — is wired consistently and end-to-end; the `[re=0, im=v]` materialization writers (`write_giao_complex_staging`, `launch_two_electron_giao2e`) both fail closed with `BufferTooSmall` rather than partial-write. Manifest registration is internally consistent (verified against the lock: 347 entries, `int1e_ovlp_spinor` still at positional id 2, all GIAO rows `complex_output=true`, spinor rows `oracle_covered=false`). The gout transcriptions in `f12.rs` (`gout_g1`/`gout_ig1`/`gout_gg1`/`gout_g1g2`) and the device kernels mirror the libcint autocode and the 10 covered families have vendor byte-identity parity tests.

The dominant concern is the `int1e_a01gp` family: it is a fully *dispatchable* public API (`eval_raw` op_kind 3, manifest-registered, kernel'd, vendor-wrapped) that returns *known-incorrect* results (~2× on a subset of components) with NO runtime guard — a silent-wrong-answer hazard in a project whose core value is byte-identity parity. The remaining findings are robustness/quality issues: GIAO families are unusable under memory-limit chunking (fail-closed, not corrupting), and some test-helper duplication.

## Critical Issues

### CR-01: `int1e_a01gp` is dispatchable but returns known-incorrect results with no guard

**File:** `crates/cintx-cubecl/src/kernels/one_electron.rs:8608`, `crates/cintx-compat/src/raw.rs:240-242`

**Issue:** `int1e_a01gp_{cart,sph}` is registered in the manifest, declared as a `RawApiId`, mapped in the `giao_nuc_op` dispatch table (`"a01gp" => Some((3, 9))`), and has a live `#[cube]` kernel arm (op_kind 3) plus a vendor FFI wrapper. The parity test for it is `#[ignore]`d with a comment documenting that the rank-9 27-s table is byte-identical only on component 0 and is "~2× on a subset of ket-varying elements of components 1..8" (`giao_1e_parity.rs:209-228`), and the manifest correctly carries `oracle_covered=false`. However, *nothing prevents a caller from invoking it*: `eval_raw(RawApiId::INT1E_A01GP_CART, ...)` resolves a valid descriptor, builds a plan, dispatches to op_kind 3, and returns a silently-wrong complex buffer. For a library whose stated core value is "libcint-compatible results … byte-identity as the primary goal," shipping a public, dispatchable entry point that emits incorrect numbers with no error is the highest-severity defect class (silent data incorrectness).

**Fix:** Gate the known-broken family behind a typed fail-closed stop until the ket-derivative double-count is fixed, so callers get an explicit `UnsupportedApi`/`NotYetImplemented` instead of wrong numbers. For example, in the `giao_nuc_op` dispatch (or at the top of the op_kind-3 arm):
```rust
if op_name == "a01gp" {
    return Err(cintxRsError::UnsupportedApi {
        requested: format!(
            "int1e_a01gp is registered but not yet correct (rank-9 ket-derivative \
             double-count; tracked, oracle_covered=false) — refusing to return \
             non-parity output for {op_name}"
        ),
    });
}
```
This preserves the "registered for surface completeness" goal while honoring the no-silent-wrong-output contract. Remove the guard when the kernel passes vendor parity and `oracle_covered` flips to true.

## Warnings

### WR-01: GIAO complex families fail closed under memory-limit chunking (unusable, not corrupting)

**File:** `crates/cintx-runtime/src/planner.rs:332-350` (`staging_elements_for_chunk`) vs `crates/cintx-cubecl/src/kernels/one_electron.rs:8424-8430` and `crates/cintx-cubecl/src/kernels/two_electron.rs:2425-2431`

**Issue:** The GIAO launchers are monolithic whole-block writers: `write_giao_complex_staging` and `launch_two_electron_giao2e` both require a staging buffer of the *full* `2 * real_total` and reject anything smaller with `BufferTooSmall`. In the safe-API `evaluate` path, `staging_elements_for_chunk` slices `output_layout.staging_elements` by `(suffix - prefix)` per chunk. With a default (no) memory limit there is a single chunk and `suffix - prefix == staging_elements`, so the round-trip test passes. But under any `memory_limit_bytes` that forces `chunk_count > 1`, the per-chunk staging is smaller than `2 * real_total`, so every GIAO call fails closed with `BufferTooSmall`. This is the documented FND-06 monolithic-writer hazard (project memory: "family kernels are MONOLITHIC whole-block writers → per-chunk staging must be FULL-block sized"). It is *safe* (no partial write / no corruption) but renders the entire GIAO family inoperable whenever memory chunking engages, which is a real availability regression vs. the raw path (`eval_raw` correctly allocates a full-block `chunk_staging` of `staging_elements` per chunk — see `raw.rs:1061-1070`).

**Fix:** Make the safe-API `evaluate` path size GIAO (and other monolithic complex/whole-block writers') chunk staging to the full `plan.output_layout.staging_elements`, mirroring `eval_raw`'s `chunk_staging` allocation, OR mark these families as non-chunkable in the planner so a memory limit that would split them produces a single typed `MemoryLimitExceeded` up front rather than a per-chunk `BufferTooSmall`. At minimum, add a runtime test driving a GIAO family through `SessionRequest::evaluate` with a `memory_limit_bytes` small enough to force `chunk_count > 1` and assert the intended behavior.

### WR-02: a01gp deferral conflates "oracle coverage" with "kernel correctness"

**File:** `crates/cintx-oracle/tests/giao_1e_parity.rs:209-228`

**Issue:** The `#[ignore]` comment frames a01gp as "tracked in SUMMARY Known Stubs" and the prompt frames it as "deferred, tracked." But the family is not merely *untested* — it is *demonstrably wrong* (the comment itself states the rank-9 table is ~2× on a subset of components). Marking `oracle_covered=false` removes it from coverage gates but does not remove it from the dispatch surface, so the manifest flag silently understates a correctness defect rather than an absence of evidence. This is the metadata side of CR-01.

**Fix:** Pair the `oracle_covered=false` flag with the CR-01 runtime guard so the manifest's "not covered" honestly implies "not callable for results," not "callable but wrong." Alternatively, drop the a01gp dispatch arm entirely (keep only the symbol registration) until the kernel is correct.

### WR-03: `giao_2e_parity.rs` duplicates comparison helpers instead of reusing `moment_common`

**File:** `crates/cintx-oracle/tests/giao_2e_parity.rs:42-153`

**Issue:** `ncart`, `nsph`, `matches_with_tol`, `count_mismatches`, `assert_any_nonzero`, and the `ATOL`/`RTOL` consts are re-implemented locally, whereas the sibling `giao_1e_parity.rs:30-33` imports the same helpers from `moment_common`. Duplicated tolerance/mismatch logic drifts: the two files could silently diverge on `RTOL` or the mismatch-printing threshold, weakening the parity gate. (The 1e file's `count_mismatches` comes from `moment_common`; the 2e file's local copy hard-codes `ATOL`/`RTOL` inside the function instead of taking them as parameters, so a future tolerance change to one path won't track the other.)

**Fix:** Import the shared helpers from `moment_common` (add a `#[path = "moment_common.rs"] mod moment_common;` as in `giao_1e_parity.rs`) and delete the local copies, or factor a `giao_common` module shared by both GIAO parity files.

### WR-04: GIAO `not0` counts the always-zero real (re) half of the interleaved buffer

**File:** `crates/cintx-cubecl/src/kernels/one_electron.rs:8493-8497`, `crates/cintx-cubecl/src/kernels/two_electron.rs:2526-2528`

**Issue:** `not0` is computed by filtering the *entire interleaved* staging (`staging.iter().filter(|v| v.abs() > threshold)`). Because the re half is deliberately zeroed, the count reflects only the imaginary entries — which is *numerically* fine — but the reported `not0` for a GIAO family is therefore the count of non-zero imaginary values, not a count over a contiguous real block, and is not directly comparable to the `not0` libcint would report for the same call (libcint counts over its real `double*`). Any downstream consumer treating `not0` as a libcint-comparable nonzero count for these families will see a value that happens to match only because the imaginary half is dense; if a future GIAO family had structural zeros in the imaginary half, the semantics would be subtly off. This is a contract-clarity issue, not a wrong-result issue.

**Fix:** Count `not0` over the imaginary half only (`staging.chunks_exact(2).filter(|c| c[1].abs() > threshold)`) so the reported nonzero count matches the libcint real-output semantics, and document that GIAO `not0` is an imaginary-component count.

### WR-05: Unused/inert comptime `complex_output` hint threaded through the moment/1e device path

**File:** `crates/cintx-cubecl/src/kernels/one_electron.rs:7034-7046`, `8913-8918`

**Issue:** The comptime `complex_output` hint is plumbed into the moment/1e device kernel (`let _is_complex_out = comptime!(complex_output == 1u32);`) but is explicitly inert ("stays inert on-device today"). The GIAO families do NOT use this device hint — they go through the separate `one_electron_giao_ovlp_kernel`/`one_electron_giao_nuc_kernel` + host `write_giao_complex_staging` path, and the complex interleaving is a pure host concern. The threaded hint is dead plumbing (an `_`-prefixed bind that is never read) carried through the launcher signature and call sites, adding surface area with no current consumer.

**Fix:** Either remove the inert hint from the moment device path until a real GIAO-on-device path needs it, or add a short `// reserved for <ticket>` note pointing at the concrete future consumer so it is not mistaken for live behavior.

## Info

### IN-01: `giao_2e_parity.rs` lacks the shared-fixture comment cross-references used elsewhere

**File:** `crates/cintx-oracle/tests/giao_2e_parity.rs:52-54`

**Issue:** `cross_center_non_square_quartet()` hard-codes `[3, 2, 3, 2]` with an inline comment naming the shells, but unlike the 1e file it does not source the pair from a shared `moment_common` helper. If the `build_h2o_sto3g_common_orig` shell ordering ever changes, this literal quartet silently selects different shells. Low risk (the fixture is stable), but a named helper would be self-documenting and refactor-safe.

**Fix:** Move the quartet selection into `moment_common` (or assert the selected shell angular momenta match the expected `[0,1,0,1]` before use).

### IN-02: `gnuc`/`ignuc` rinv-center exclusion relies on an implicit `op_kind >= 2` convention

**File:** `crates/cintx-cubecl/src/kernels/one_electron.rs:8813`

**Issue:** The single-center-vs-atom-sum split is encoded as `let is_rinv_center = op_kind >= 2;` with the family→op_kind mapping living separately in `giao_nuc_op` (`raw.rs` reads `is_giao_rinv_center_symbol` for env extraction). The `op_kind >= 2` magic boundary couples the kernel to the exact numbering of the dispatch table — re-numbering `giao_nuc_op` (e.g., inserting a new rank-3 family at slot 2) would silently flip gnuc/ignuc onto the rinv-center path. This is the same positional-coupling class flagged in project memory ("new family registration shifts ids → breaks hardcoded consts"), here applied to op_kind ordinals.

**Fix:** Derive `is_rinv_center` from an explicit per-family property (e.g., a field on the dispatch tuple, matching the `is_origj` precedent the moment path already uses at `one_electron.rs:8654-8661`) rather than from the ordinal value of `op_kind`.

### IN-03: Magic angular-momentum ceilings duplicated across host guard and device sizing

**File:** `crates/cintx-cubecl/src/kernels/one_electron.rs:8710` (`li + lj + 3 > 8`), `8783` (`(li + lj + 5) / 2 + 1`), `3098`/`3718` (`nmax = li + lj + 3` / `+ 5`)

**Issue:** The GIAO headroom constants (`+3` for the overlap engine, `+5` for the nuclear engine, the `<= 8` VRR envelope) appear as bare literals in both the host fail-closed guard and the device `run_*_device` sizing, with the relationship between them ("nmax = li+lj+5; nroots = nmax/2+1") spelled out only in comments. A change to one site that is not mirrored in the other would either over-allocate or (worse) under-size the device G-tensor without the host guard catching it. The moment path already centralizes this via a shared `moment_params` const fn (cited at `one_electron.rs:8627-8661`); the GIAO path does not.

**Fix:** Hoist the per-engine headroom into shared `const fn`s (mirroring `moment_params`) consumed by both the host guard and the device sizing so the envelope cannot drift between the two.

---

_Reviewed: 2026-05-31_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
