---
phase: 260529-mfh
plan: 01
type: tdd
wave: 1
depends_on: []
files_modified:
  - crates/cintx-compat/src/helpers.rs
  - crates/cintx-oracle/tests/cintgto_norm_parity.rs
autonomous: true
requirements: [HELP-02]
must_haves:
  truths:
    - "CINTgto_norm(n,a) matches libcint 6.1.3 misc.c::CINTgto_norm within atol=1e-12 for n in 0..5, a in {0.5,1.0,2.5}"
    - "The cintx-compat lib unit test pins at least one exact expected value (no vendor build required to catch regression)"
    - "A vendor parity test exists, gated like the other oracle vendor tests, asserting 0 mismatches over the (n,a) grid"
    - "The full vendor oracle gate no longer reports a CINTgto_norm mismatch"
    - "Defensive guard preserved: invalid input (non-finite a, a<=0, n<0) returns 0.0 with no panic/NaN/Inf"
  artifacts:
    - path: "crates/cintx-compat/src/helpers.rs"
      provides: "Corrected CINTgto_norm using libcint's closed-form normalization"
      contains: "pub fn CINTgto_norm"
    - path: "crates/cintx-oracle/tests/cintgto_norm_parity.rs"
      provides: "Vendor parity test for CINTgto_norm"
      contains: "vendor_CINTgto_norm"
  key_links:
    - from: "crates/cintx-oracle/tests/cintgto_norm_parity.rs"
      to: "cintx_compat::helpers::CINTgto_norm"
      via: "direct call compared against vendor_ffi::vendor_CINTgto_norm"
      pattern: "CINTgto_norm"
---

<objective>
Fix `cintx_compat::helpers::CINTgto_norm` so it matches libcint 6.1.3
`misc.c::CINTgto_norm` exactly. The current implementation is wrong: it places
`(2a)^(n+1.5)` in the DENOMINATOR (values shrink as `a` grows — the opposite of
correct) and uses `(2n-1)!!` instead of the correct gamma/factorial terms.
Result: all 15 (l,a) combinations in the oracle gate mismatch (e.g. (0,0.5)
cintx=1.331 vs vendor=1.502; (4,2.5) cintx=0.163 vs vendor=16.34).

Purpose: Restore libcint result compatibility for the GTO normalization helper
(HELP-02), unblocking the helper/transform parity check in the vendor oracle gate.
Output: Corrected `CINTgto_norm`, a strengthened lib unit assertion, a new
vendor parity test, and an honest report of the next gate blocker.
</objective>

<execution_context>
@/home/user/Documents/workspace/cintx/.claude/get-shit-done/workflows/execute-plan.md
@/home/user/Documents/workspace/cintx/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@./CLAUDE.md

<root_cause>
Ground truth — libcint `libcint-master/src/misc.c` (lines 72-92):
```c
static double _gaussian_int(FINT n, double alpha) {
    double n1 = (n + 1) * .5;
    return exp(lgamma(n1)) / (2. * pow(alpha, n1));
}
double CINTgto_norm(FINT n, double a) {
    return 1. / sqrt(_gaussian_int(n*2+2, 2*a));
}
```
Documented closed form (misc.c comment, Schlegel & Frisch IJQC 54(1995) 83-87):
  norm = sqrt( 2^(2n+3) * (n+1)! * (2a)^(n+1.5) / ((2n+2)! * sqrt(pi)) )

cintx CURRENT (crates/cintx-compat/src/helpers.rs ~line 225-242) is WRONG:
  sqrt( (2n-1)!! * sqrt(pi) / (2a)^(n+1.5) )   <- (2a) in denominator, wrong term.
</root_cause>

<interfaces>
From crates/cintx-compat/src/helpers.rs:
```rust
pub fn CINTgto_norm(n: i32, a: f64) -> f64;  // current WRONG body to replace
// PRESERVE this guard verbatim at the top:
//   if !a.is_finite() || a <= 0.0 || n < 0 { return 0.0; }
```
From crates/cintx-oracle/src/vendor_ffi.rs (line 1010):
```rust
pub fn vendor_CINTgto_norm(n: i32, a: f64) -> f64;  // calls ffi::CINTgto_norm
```
Vendor-test gating pattern (from tests/one_electron_scalar_spinor_parity.rs):
```rust
#![cfg(any(feature = "cpu", feature = "rocm"))]
// non-vendor smoke test:
#[cfg(feature = "cpu")] #[test] fn ... { }
// vendor parity test:
#[cfg(has_vendor_libcint)] #[cfg(feature = "cpu")] #[test] fn ... { }
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: RED — pin libcint-correct expected values in unit + vendor parity tests</name>
  <files>crates/cintx-compat/src/helpers.rs, crates/cintx-oracle/tests/cintgto_norm_parity.rs</files>
  <behavior>
    - Lib unit test (helpers.rs `tests` module): replace the weak smoke assert
      `assert!(CINTgto_norm(1, 0.5) > 0.0)` (~line 263) with at least one EXACT
      hardcoded expected value asserted within 1e-12. Use these libcint-correct
      values (verified against the closed form):
        * CINTgto_norm(0, 0.5) ≈ 1.5022113490956049  (assert_abs_diff < 1e-12)
        * CINTgto_norm(4, 2.5) ≈ 16.34  (verify exact value at impl time; assert < 1e-9
          if you compute the literal independently, or recompute the precise constant)
      Recommended: assert the n=0,a=0.5 value (1.5022113490956049) exactly and
      add a self-consistency check that norm GROWS with `a` (current bug makes it
      shrink): `assert!(CINTgto_norm(2, 2.5) > CINTgto_norm(2, 0.5))`.
      Also assert the guard: `assert_eq!(CINTgto_norm(-1, 0.5), 0.0)` and
      `assert_eq!(CINTgto_norm(1, 0.0), 0.0)`.
    - New vendor parity test file crates/cintx-oracle/tests/cintgto_norm_parity.rs:
        * `#![cfg(any(feature = "cpu", feature = "rocm"))]` at top.
        * One vendor parity test gated `#[cfg(has_vendor_libcint)] #[cfg(feature = "cpu")] #[test]`.
        * Loop n in 0..5, a in {0.5, 1.0, 2.5, 0.75, 5.0} (grid + extras), compare
          `cintx_compat::helpers::CINTgto_norm(n,a)` vs
          `cintx_oracle::vendor_ffi::vendor_CINTgto_norm(n,a)`, accumulate a
          `mismatches` counter when `(c - v).abs() > 1e-12`, then
          `assert_eq!(mismatches, 0, ...)` reporting each (n,a,cintx,vendor,diff).
        * Add a non-vendor `#[cfg(feature = "cpu")] #[test]` smoke test asserting
          CINTgto_norm(0,0.5) is finite and ~1.5022113490956049 so the file is not
          a pure no-op without the vendor build.
  </behavior>
  <action>
    Write the tests FIRST. Do NOT touch the CINTgto_norm body in this task. The
    strengthened lib unit test and the vendor parity test MUST fail against the
    current (wrong) implementation. Confirm RED:
      - `cargo test -p cintx-compat --lib` fails on the new exact-value assertion.
    Model the vendor test's gating/imports on
    crates/cintx-oracle/tests/one_electron_scalar_spinor_parity.rs (see <interfaces>).
    Import via `use cintx_oracle::vendor_ffi;` and call
    `cintx_compat::helpers::CINTgto_norm`. Do NOT modify compare.rs.
    Commit (code only): `test(260529-mfh): pin libcint-correct CINTgto_norm expected values`
  </action>
  <verify>
    <automated>cargo test -p cintx-compat --lib 2>&1 | grep -q "FAILED\|test result: FAILED" && echo "RED confirmed (unit test fails as expected)"</automated>
  </verify>
  <done>Lib unit test asserts ≥1 exact libcint value + guard cases; vendor parity test file exists and is correctly gated; `cargo test -p cintx-compat --lib` FAILS on the new assertion (RED).</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: GREEN — replace CINTgto_norm body with libcint closed form, then run full gate</name>
  <files>crates/cintx-compat/src/helpers.rs</files>
  <behavior>
    - After the fix, the unit test from Task 1 passes (n=0,a=0.5 → 1.5022113490956049
      within 1e-12; norm grows with `a`; guard returns 0.0).
    - Vendor parity test reports 0 mismatches over the (n,a) grid.
  </behavior>
  <action>
    Replace the body of `CINTgto_norm` (crates/cintx-compat/src/helpers.rs ~225-242)
    with libcint's exact normalization. PRESERVE the guard verbatim:
      `if !a.is_finite() || a <= 0.0 || n < 0 { return 0.0; }`
    Use OPTION (a) — the documented closed form, dependency-free, exact for integer n
    (RECOMMENDED; no Cargo.toml change). Implement:
      norm = sqrt( 2^(2n+3) * (n+1)! * (2a)^(n+1.5) / ((2n+2)! * sqrt(pi)) )
    Compute (n+1)! and (2n+2)! as f64 products (exact in f64 for n<=~8; the oracle
    only tests n in 0..5). Use `2.0f64.powi(2*n + 3)`, `(2.0*a).powf(n as f64 + 1.5)`,
    `std::f64::consts::PI.sqrt()`. Add a one-line comment citing misc.c + the
    Schlegel & Frisch reference and noting this replaces the prior inverted formula.
    Justify in the SUMMARY why (a) over (b): no new dep, exact for the tested integer
    range, byte-stable. (If a numerical reason forces option (b) — `1.0 /
    gaussian_int(2*n+2, 2.0*a).sqrt()` with an lgamma — `libm` may be added as a
    DIRECT dep of cintx-compat; only then edit crates/cintx-compat/Cargo.toml.
    Default expectation: NO Cargo.toml change.)
    Do NOT modify crates/cintx-oracle/src/compare.rs.

    Then run, in order, reporting each verbatim:
      1. `cargo test -p cintx-compat --lib`  (MUST be green)
      2. Vendor parity test under both gates:
         `CINTX_BACKEND=cpu CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --locked --test cintgto_norm_parity` (MUST be green; vendor build is slow — allow generous time)
      3. FINAL MANDATORY full vendor oracle gate (verbatim):
         `CINTX_BACKEND=cpu CINTX_ORACLE_BUILD_VENDOR=1 cargo run --locked --manifest-path xtask/Cargo.toml -- oracle-compare --profiles "base,with-f12,with-4c1e,with-f12+with-4c1e" --include-unstable-source false`
    Confirm the CINTgto_norm mismatch is GONE. Report what the gate hits NEXT —
    it should advance to the CINTc2s_bra_sph helper check (known pre-existing,
    SEPARATE task) or further. Report the next blocker VERBATIM. Do NOT fix
    CINTc2s_bra_sph or anything else here. Never fabricate a pass — if vendor build
    fails to compile/link, report that honestly instead of claiming the gate passed.
    Commit (code only, NOT docs): `fix(260529-mfh): CINTgto_norm matches libcint misc.c closed form`
  </action>
  <verify>
    <automated>cargo test -p cintx-compat --lib 2>&1 | grep -q "test result: ok" && echo "GREEN: cintx-compat lib tests pass"</automated>
  </verify>
  <done>`cargo test -p cintx-compat --lib` green; vendor parity test (both gates) green with 0 mismatches; full oracle gate run verbatim with the CINTgto_norm mismatch GONE and the NEXT blocker reported verbatim (no fabrication); CINTgto_norm body uses libcint's closed form with guard preserved; no changes to compare.rs.</done>
</task>

</tasks>

<verification>
- `cargo test -p cintx-compat --lib` passes (pins libcint-correct value without vendor build).
- `CINTX_BACKEND=cpu CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --locked --test cintgto_norm_parity` passes with 0 mismatches.
- Full oracle gate run verbatim; CINTgto_norm mismatch absent; next blocker reported verbatim.
- No edits to crates/cintx-oracle/src/compare.rs. Cargo.toml unchanged unless option (b) chosen.
- Defensive guard preserved (no panic/NaN/Inf on invalid input).
</verification>

<success_criteria>
- CINTgto_norm matches libcint 6.1.3 within atol=1e-12 over the full (n,a) grid.
- TDD order honored: tests pinned RED first, then formula fix turns them GREEN.
- Two atomic code commits (test, then fix); docs not committed in this task.
- Honest report of the gate's next blocker (expected: CINTc2s_bra_sph or beyond), left UNFIXED.
</success_criteria>

<output>
After completion, create `.planning/quick/260529-mfh-fix-cintgto-norm-to-match-libcint-misc-c/260529-mfh-SUMMARY.md`
</output>
