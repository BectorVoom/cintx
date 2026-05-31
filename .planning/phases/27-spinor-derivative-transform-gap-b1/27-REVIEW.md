---
phase: 27-spinor-derivative-transform-gap-b1
reviewed: 2026-05-31T00:00:00Z
depth: standard
files_reviewed: 13
files_reviewed_list:
  - crates/cintx-cubecl/src/kernels/center_2c2e.rs
  - crates/cintx-cubecl/src/kernels/center_3c1e.rs
  - crates/cintx-cubecl/src/kernels/center_3c2e.rs
  - crates/cintx-cubecl/src/kernels/one_electron.rs
  - crates/cintx-cubecl/src/kernels/two_electron.rs
  - crates/cintx-cubecl/src/transform/c2spinor.rs
  - crates/cintx-oracle/build.rs
  - crates/cintx-oracle/src/compare.rs
  - crates/cintx-oracle/src/fixtures.rs
  - crates/cintx-oracle/src/vendor_ffi.rs
  - crates/cintx-oracle/tests/giao_1e_parity.rs
  - crates/cintx-oracle/tests/spinor_deriv_parity.rs
  - xtask/src/oracle_covered_update.rs
findings:
  critical: 0
  warning: 4
  info: 3
  total: 7
status: issues_found
---

# Phase 27: Code Review Report

**Reviewed:** 2026-05-31
**Depth:** standard
**Files Reviewed:** 13
**Status:** issues_found

## Summary

Reviewed the phase-27 spinor-derivative transform diff (base `7ab2bf9^`): the three new
derivative wrappers in `c2spinor.rs` (`cart_to_spinor_sf_derivative_2d` / `_3c2e` / `_3c1e`
delegating to `_3c_impl`), the launcher rewiring across `one_electron.rs`, `center_2c2e.rs`,
`center_3c2e.rs`, and `center_3c1e.rs`, the spherical-aux-k sizing reconciliation in
`fixtures.rs`/`vendor_ffi.rs`, the GIAO `a01gp` 0.5-factor fix, and the coverage-flip /
no-silent-skip machinery in `spinor_deriv_parity.rs` + `oracle_covered_update.rs`.

The core transform math is sound. I traced the buffer layouts end-to-end and found NO
correctness blockers in the wrappers or launchers:

- The KET→BRA transpose is correctly centralized in `cart_to_spinor_sf_derivative_2d`
  (and inside `cart_to_spinor_sf_3c2e` for the arity-3 path); no launcher re-owns it.
- The contraction-major-outer / component / ket-major-bra-fastest cart layout is consistent
  between every producer (`one_electron.rs` kernels, `center_2c2e.rs` fill, `center_3c1e.rs`
  `relayout_3c1e_grad_to_blocked`, `center_3c2e.rs` device cart) and the wrapper's
  `src_base = (ci*nctr_j+cj)*total_len + comp*block_len` / `block[jc*nci+ic]` reader.
- Aux-k is sized SPHERICAL `nsph(lk)` (2lk+1) in the wrapper, the fixtures collector, and
  the vendor collector — matching the documented `CINT3c2e_spinor_drv is_ssc=0` correction.
  The 360-vs-720 invariant is regression-anchored.
- All fail-closed size checks run BEFORE any write (the `_too_small_fails_closed` tests prove
  the sentinel survives), and the `nctr_k > 1` aux-k contraction case is rejected in every
  arity-3 launcher path rather than silently mis-folded.
- The `int2c2e_ip*` / `int3c1e_*` vendor-stub arms (which `return 0` or `exit(1)` in libcint
  6.1.3) are correctly held `oracle_covered=false` in the lock and their parity tests are
  `#[ignore]`'d with explicit "DO NOT run with --include-ignored" warnings.

The findings below are coverage-claim integrity issues (the headline WR-01) plus
documentation/consistency defects. No data-loss, injection, or crash defects were found.

## Warnings

### WR-01: `oracle_covered=true` stamped on ~14 spinor-derivative families with no executed per-symbol vendor byte-identity comparison

**File:** `crates/cintx-oracle/tests/spinor_deriv_parity.rs:438-464` (FLIPPED list); `crates/cintx-ops/generated/compiled_manifest.lock.json`; `crates/cintx-oracle/src/vendor_ffi.rs:4462-4577`

**Issue:** The `FLIPPED` set flips 20 families to `oracle_covered=true`, and the lock confirms
them true. But the actual *executed* (non-`#[ignore]`'d) vendor byte-identity tests in this
file cover only a representative subset: `int1e_ipovlp` (rank-3 sf_2d), `int1e_ipovlpip`
(rank-9 sf_2d), `int1e_ipipipiprinv` (rank-81 sf_2d), and `int3c2e_ip1` (rank-3 sf_3c2e).
The remaining flipped families have **neither a `vendor_*` FFI wrapper nor a backing
vendor comparison**:

- `int3c2e_ip2_spinor` — stamped covered, yet has NO `vendor_int3c2e_ip2_spinor` wrapper, NO
  allowlist entry in `build.rs` (only `int3c2e_ip1_spinor` was added), and NO test. The only
  reference is its name in the `FLIPPED` array.
- rank-9 sf_2d: `int1e_ipkinip`, `int1e_ipnucip`, `int1e_ipipovlp`, `int1e_ipipnuc`,
  `int1e_ipipkin`, `int1e_ipiprinv` — covered, no per-symbol vendor wrapper/test.
- rank-27 sf_2d: `int1e_ipipnucip`, `int1e_ipiprinvip`, `int1e_ipipipnuc`, `int1e_ipipiprinv`
  — entire rank tier covered with ZERO executed vendor comparison.
- rank-81 sf_2d: `int1e_ipiprinvipip`, `int1e_ipipiprinvip` — covered, no per-symbol test.

This is the precise anti-pattern recorded as threat T-21-08-02 ("must NOT be stamped
oracle_covered=true ... false verification claim"). The phase's stated rationale is "one
representative byte-identity test per (transform-path × rank-tier) then flip the tier," which
is a defensible *design* choice — but it makes `oracle_covered=true` mean "a sibling sharing
this wrapper was byte-identity-checked," not the literal "this symbol has a vendor reference."
A downstream consumer reading the flag at face value is misled, and a divergence introduced in
one tier member (e.g. an `ipnucip`-specific gout or origin path) would pass coverage silently.

**Fix:** Either (a) add executed (non-ignored) vendor parity tests + `vendor_*` wrappers +
`build.rs` allowlist entries for every symbol flipped to `oracle_covered=true`, or
(b) introduce a distinct manifest flag (e.g. `oracle_covered_via_shared_wrapper`) so the
literal per-symbol byte-identity claim is not overstated. At minimum, `int3c2e_ip2_spinor`
should not read `oracle_covered=true` while it has no wrapper, no allowlisted vendor symbol,
and no test of any kind:
```rust
// build.rs allowlist — int3c2e_ip2_spinor is referenced as covered but never allowlisted:
// add `int3c2e_ip2_spinor` and a vendor_int3c2e_ip2_spinor wrapper + non-ignored parity test,
// or move it to DEFERRED until one exists.
```

### WR-02: Docstrings claim `nsph(lk) = (2lk+1)*nctr_k` but `nsph` returns only `2lk+1`

**File:** `crates/cintx-cubecl/src/transform/c2spinor.rs:1454,1474` (derivative_3c2e / _3c1e docstrings); echoed in `crates/cintx-oracle/src/vendor_ffi.rs:4452,4544,4566` and several test comments

**Issue:** Multiple docstrings state the aux-k axis length is `nsph(lk) = (2lk+1)*nctr_k`, but
`c2s::nsph(l)` is defined as exactly `2*l + 1` with no contraction factor (`c2s.rs:22`). The
arity-3 wrapper uses bare `nsph(lk)` for `nsk` and the launchers reject `nctr_k > 1`, so the
runtime is correct for the only supported case (`nctr_k == 1`). But the formula in the docs is
dimensionally wrong: the contracted axis length would be `nsph(lk) * nctr_k`, which the
wrapper does NOT compute. A future maintainer who lifts the `nctr_k > 1` guard trusting this
doc would size the aux-k axis incorrectly.

**Fix:** Correct the docstrings to `nsph(lk) = 2lk+1` (single spherical axis; aux-k
contraction `nctr_k > 1` is rejected upstream), e.g.:
```rust
/// AUX-K IS SPHERICAL `nsph(lk) = 2lk+1` (a single spherical axis; the launchers reject
/// `nctr_k > 1`, so the contracted aux-k length `nsph(lk)*nctr_k` is never materialized here).
```

### WR-03: `grad_stats` `not0` counts re+im on interleaved-complex spinor output (inconsistent with the WR-04 GIAO fix applied this same phase)

**File:** `crates/cintx-cubecl/src/kernels/center_3c1e.rs:981`

**Issue:** `grad_stats` computes `not0 = staging.iter().filter(|v| v.abs() > thr).count()`. For
the new Spinor path the staging buffer is interleaved-complex `[re, im, re, im, ...]`, so this
counts both real and imaginary lanes — the exact double-count that WR-04 corrected in the GIAO
paths (`two_electron.rs:2528-2533` and `one_electron.rs:8517-8521`) this same phase by switching
to `chunks_exact(2).filter(|c| c[1].abs() > thr)`. `grad_stats` was left on the old per-element
count, so for spinor 3c1e gradients `not0` is inconsistent with the GIAO convention adopted
elsewhere. `not0` is a stats/diagnostic field (not a parity gate), so this is not a correctness
blocker — but it is an internally inconsistent metric introduced by the same diff that fixed the
identical issue two files over.

**Fix:** Make `grad_stats` honor the interleaved-complex layout for the spinor case (or count
imaginary lanes only, matching the GIAO convention), e.g. branch on
`plan.representation == Representation::Spinor` to use `chunks_exact(2)`.

### WR-04: `derivative_3c2e_rank3_layout` test asserts non-zero output but never validates numeric correctness against the inline replay (unlike the rank-3 2d test)

**File:** `crates/cintx-cubecl/src/transform/c2spinor.rs:2192-2230` (`derivative_3c2e_rank3_layout`)

**Issue:** The arity-2 `derivative_2d_rank3_matches_inline` test (c2spinor.rs:2080) properly
pins correctness by replaying the inline transform and asserting byte equality per element.
The arity-3 analogue `derivative_3c2e_rank3_layout` only asserts the total is 360 and that each
component slice is "not all-zero" — it never checks the folded *values* against an independent
replay of `cart_to_spinor_sf_3c2e` per (comp,k). A scatter-index bug in
`cart_to_spinor_sf_derivative_3c_impl` (e.g. the `mk * ni_full * nj_full` k-stride or the
`j_global * ni_full + i_global` placement) that still produced non-zero output in every
component would pass this test. The arity-3 scatter is the most index-dense new code in the
diff and is the least directly verified at the unit level (it leans entirely on the vendor
parity test for `int3c2e_ip1`, which is square-block-friendly for nctr=1).

**Fix:** Mirror the 2d test: build the expected buffer by calling `cart_to_spinor_sf_3c2e`
per `(comp)` block and scattering into the contraction-major position with the same k-stride,
then `check_close` every element of `got` against `expected`.

## Info

### IN-01: `relayout_3c1e_grad_to_blocked` hard-codes `ncomp = 3`, coupling it to the gradient rank

**File:** `crates/cintx-cubecl/src/kernels/center_3c1e.rs:1149`

**Issue:** `relayout_3c1e_grad_to_blocked` sets `let ncomp = 3usize;` internally and is called
only from the rank-3 `ip1`/`iprinv` paths, so it is correct today. But the magic `3` is
implicit; if a higher-order 3c1e spinor derivative (rank-9+) is ever wired through this helper
it will silently truncate to 3 components. The arity-2 wrapper takes `ncomp` as a parameter for
exactly this reason.

**Fix:** Take `ncomp` as a parameter (passing `3` from the two current callers) so the helper
cannot silently truncate a future higher-rank family.

### IN-02: Several launcher comments assert vendor source line numbers without an in-repo cross-check

**File:** `crates/cintx-cubecl/src/transform/c2spinor.rs:1444` ("cint3c2e.c:631-636"); `crates/cintx-cubecl/src/kernels/one_electron.rs:3290` ("intor1.c:551,572")

**Issue:** Multiple comments cite exact upstream libcint line numbers as the authority for the
spherical-aux-k decision and the a01gp 0.5 factor. These are load-bearing claims (they justify
removing fail-closed guards), but nothing in the diff pins them to the vendored source, so they
will silently rot if the vendored libcint version moves. This is documentation hygiene, not a
defect.

**Fix:** Optionally reference the vendored path/tag alongside the line number, or add a
test/assert that reads the documented constant from the vendored header where feasible.

### IN-03: `to_j_fastest` negative-control helper is correct but relies on `ni*nj` index range staying in bounds without an explicit assert

**File:** `crates/cintx-oracle/tests/spinor_deriv_parity.rs:377-391`

**Issue:** `to_j_fastest` writes `dst = comp_off + (i*nj + j)*2`. For a non-square block this
index range coincides with the `ni*nj` block extent (max `(ni-1)*nj + (nj-1) = ni*nj - 1`), so
it is in-bounds — but this is only true because the source and destination logical extents are
equal `ni*nj`; the helper transposes the *index interpretation* without resizing. It is correct
as written; an explicit comment or `debug_assert` that the swapped buffer length equals the
source would make the in-bounds reasoning obvious to a future editor.

**Fix:** Add a one-line comment noting that `(i*nj+j)` and `(j*ni+i)` both range over
`0..ni*nj`, so the reindex stays in-bounds for any (ni, nj).

---

_Reviewed: 2026-05-31_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
