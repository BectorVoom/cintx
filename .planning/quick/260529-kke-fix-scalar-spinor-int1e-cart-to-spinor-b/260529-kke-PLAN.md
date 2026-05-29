---
phase: quick-260529-kke
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/cintx-oracle/tests/one_electron_scalar_spinor_parity.rs
  - crates/cintx-cubecl/src/kernels/one_electron.rs
autonomous: true
requirements: [QUICK-260529-kke]
must_haves:
  truths:
    - "A vendor parity test exists that drives an asymmetric p+ x p+ cross block (two DISTINCT p shells on different centers, nci=ncj=3, i!=j) for int1e_{ovlp,kin,nuc}_spinor."
    - "Before the fix, that test FAILS (mismatch_count > 0) — proving the orientation bug is real and the fixture actually exercises it."
    - "After the transpose fix in the scalar spinor arm, the test PASSES (0 mismatches vs libcint 6.1.3) for all three operators."
    - "The scalar spinor arm transposes its single nci*ncj Cartesian block ket-major -> bra-major before cart_to_spinor_sf_2d, mirroring the gradient arm (260529-jtd)."
    - "cart_to_spinor_sf_2d itself is unchanged (the gradient path depends on its bra-major contract)."
    - "Pre-existing passing tests still pass: scalar-spinor idempotency, sph/cart scalar parity, and the jtd gradient spinor parity."
  artifacts:
    - path: "crates/cintx-oracle/tests/one_electron_scalar_spinor_parity.rs"
      provides: "Asymmetric two-p-shell vendor parity test for int1e_{ovlp,kin,nuc}_spinor (double-gated cpu + has_vendor_libcint)."
      min_lines: 200
    - path: "crates/cintx-cubecl/src/kernels/one_electron.rs"
      provides: "Transpose of the scalar spinor Cartesian block to bra-major before cart_to_spinor_sf_2d."
      contains: "block_bra_major"
  key_links:
    - from: "crates/cintx-cubecl/src/kernels/one_electron.rs (scalar Spinor arm ~line 2906)"
      to: "cart_to_spinor_sf_2d (c2spinor.rs apply_bra_block, bra-major reader)"
      via: "ket-major -> bra-major transpose of the single nci*ncj cart_blocks block"
      pattern: "block_bra_major\\[ic \\* ncj \\+ jc\\] = .*\\[jc \\* nci \\+ ic\\]"
    - from: "crates/cintx-oracle/tests/one_electron_scalar_spinor_parity.rs"
      to: "vendor_ffi::vendor_int1e_{ovlp,kin,nuc}_spinor / RawApiId::INT1E_{OVLP,KIN,NUC}_SPINOR"
      via: "count_mismatches over the full asymmetric spinor matrix, asserted == 0"
      pattern: "vendor_int1e_(ovlp|kin|nuc)_spinor"
---

<objective>
Fix the SCALAR spinor int1e cart->spinor block-orientation bug and prove it with vendor
parity on ASYMMETRIC spinor shells.

The scalar spinor 1e arm (Representation::Spinor in one_electron.rs ~line 2906) feeds a
ket-major / bra-fastest Cartesian block (`block[cj*nci + ci]`, produced by the device
scalar kernel ported in 260529-imi) into `cart_to_spinor_sf_2d`, which reads bra-major /
ket-fastest (`cart[bra*ncj + ket]`, see c2spinor.rs apply_bra_block line ~693). This is the
exact mirror of the GRADIENT-path bug fixed in quick task 260529-jtd (commit e1dae40), which
transposed each per-component block to bra-major before the transform. The scalar arm was
never fixed because its existing tests are idempotency-only and could not catch it.

The orientation only matters when BOTH nci>1 AND ncj>1 for two DISTINCT shells. Every
existing scalar-spinor fixture (H2O/STO-3G) has at most a single p shell, so every cross
block has an s side (nci==1 or ncj==1) and is transpose-invariant — the bug stays hidden.

Purpose: Make int1e_{ovlp,kin,nuc}_spinor byte-parity-correct vs libcint 6.1.3 for
asymmetric p+ blocks, with a regression test that fails on the bug and passes on the fix.

Output:
- New vendor parity test driving an asymmetric two-distinct-p-shell fixture.
- One transpose in the scalar spinor arm (mirror of jtd's gradient transpose).
- Honest before/after mismatch counts vs vendored libcint.
</objective>

<execution_context>
@/home/user/Documents/workspace/cintx/.claude/get-shit-done/workflows/execute-plan.md
@/home/user/Documents/workspace/cintx/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@./CLAUDE.md

# The fix site (scalar Spinor arm) — currently passes cart_blocks WITHOUT transpose:
@crates/cintx-cubecl/src/kernels/one_electron.rs

# The bra-major reader (DO NOT CHANGE — gradient path depends on it):
@crates/cintx-cubecl/src/transform/c2spinor.rs

# The mirror to copy: gradient spinor vendor parity test (model the new scalar test on it):
@crates/cintx-oracle/tests/one_electron_grad_spinor_parity.rs

<interfaces>
<!-- Contracts the executor needs. Already verified present — no exploration required. -->

Scalar spinor vendor FFI wrappers ALREADY EXIST (crates/cintx-oracle/src/vendor_ffi.rs:1309-1393):
```rust
pub fn vendor_int1e_ovlp_spinor(out: &mut [f64], shls: &[i32;2], atm: &[i32], natm: i32, bas: &[i32], nbas: i32, env: &[f64]) -> i32;
pub fn vendor_int1e_kin_spinor (out: &mut [f64], shls: &[i32;2], atm: &[i32], natm: i32, bas: &[i32], nbas: i32, env: &[f64]) -> i32;
pub fn vendor_int1e_nuc_spinor (out: &mut [f64], shls: &[i32;2], atm: &[i32], natm: i32, bas: &[i32], nbas: i32, env: &[f64]) -> i32;
```
Output layout per shell pair: ni_sp * nj_sp complex = ni_sp*nj_sp*2 f64 (interleaved re/im,
column-major bra-fastest within the block). NO new FFI wrappers, NO build.rs allowlist
changes needed — `int1e_ovlp_spinor|int1e_kin_spinor|int1e_nuc_spinor` are already in the
bindgen allowlist (build.rs:358) and the RawApiId map (compare.rs:294-300).

RawApiId variants (crates/cintx-compat/src/raw.rs:130-138):
```rust
RawApiId::INT1E_OVLP_SPINOR  // "int1e_ovlp_spinor"
RawApiId::INT1E_KIN_SPINOR   // "int1e_kin_spinor"
RawApiId::INT1E_NUC_SPINOR   // "int1e_nuc_spinor"
```
eval_raw (cintx_compat::raw::eval_raw) — same signature used by the gradient test:
```rust
unsafe { eval_raw(api_id, Some(&mut out), None, &shls, atm, bas, env, None, None) }
```

Spinor block length per shell (kappa==0): ni_sp = 4*l + 2  (s=2, p=6).
For two p shells (l=1), each block is 6x6 complex; the cross block (i!=j) is the asymmetric
p+ x p+ block that exposes the bug (Cartesian nci=ncj=3 > 1, distinct shells).

The fix site — scalar Spinor arm (one_electron.rs ~line 2892-2906), CURRENT (buggy):
```rust
Representation::Spinor => {
    if n_ctr_i != 1 || n_ctr_j != 1 { return Err(... nctr>1 ...); }
    let kappa_i = shell_i.kappa;
    let kappa_j = shell_j.kappa;
    cart_to_spinor_sf_2d::<F>(staging, &cart_blocks, li, kappa_i, lj, kappa_j)?;  // <-- ket-major fed to bra-major reader
}
```

The mirror to copy — gradient arm transpose (one_electron.rs ~line 2712-2717, from jtd):
```rust
let mut block_bra_major = vec![0.0f64; block_len];
for ic in 0..nci {
    for jc in 0..ncj {
        block_bra_major[ic * ncj + jc] = block[jc * nci + ic];
    }
}
cart_to_spinor_sf_2d::<F>(&mut staging[..], &block_bra_major, li, shell_i.kappa, lj, shell_j.kappa)?;
```
(`nci`, `ncj`, `block_len = nci*ncj` are already in scope in the scalar arm; cart_blocks is
exactly one nci*ncj block here because nctr=1 is enforced just above.)
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add failing asymmetric vendor parity test for scalar int1e_{ovlp,kin,nuc}_spinor</name>
  <files>crates/cintx-oracle/tests/one_electron_scalar_spinor_parity.rs</files>
  <action>
Create a new vendor parity test modeled on
`crates/cintx-oracle/tests/one_electron_grad_spinor_parity.rs`, but for the SCALAR
(non-gradient) spinor operators and WITHOUT the 3-component dimension. Implements the proof
side of this quick task. DO NOT touch any production code in this task — write the test, run
it against the CURRENT (buggy) code, and confirm it FAILS.

1. File header `#![cfg(any(feature = "cpu", feature = "rocm"))]` and the same imports as the
   gradient test (ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, KAPPA_OF, NCTR_OF,
   NPRIM_OF, NUC_MOD_OF, POINT_NUC, PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA,
   RawApiId, eval_raw). ATOL=1e-12, RTOL=0.0.

2. Build an ASYMMETRIC fixture `build_two_p_spinor()` that GUARANTEES a p+ x p+ cross block
   with nci=ncj=3 and i!=j. Do NOT use H2O/STO-3G (single p shell -> every cross block has an
   s side). Construct a minimal self-contained slab (PTR_ENV_START-aligned, mirroring
   build_h2o_sto3g_spinor) with:
     - 2 atoms at DIFFERENT coordinates (e.g. A at [0,0,0], B at [0.0, 1.4307, 1.1078]).
     - 2 DISTINCT p shells (ANG_OF=1, KAPPA_OF=0, NCTR_OF=1, NPRIM_OF=3), each on a different
       atom AND with DIFFERENT exponents/coefficients (e.g. shell 0 = the O_2p triple
       {5.0331513, 1.1695961, 0.3803890} / {0.15591627,0.60768372,0.39195739} on atom 0;
       shell 1 = a scaled/distinct p set, e.g. exps {3.4252509, 0.6239137, 0.1688554} with a
       valid normalized coeff triple, on atom 1). Both atoms POINT_NUC with a shared zeta=0
       slot. Distinct exponents are REQUIRED so the (0,1) and (1,0) cross blocks are
       value-asymmetric — not just an idempotent same-shell block.
     - N_SHELLS=2, N_ATOMS=2.

3. Spinor length helper: `spinor_len_kappa0(l) = (4*l + 2) as usize` (copy from gradient
   test). For two p shells: each ni_sp = nj_sp = 6, total n_sp = 12.

4. cintx collector `collect_cintx_spinor(api_id, atm, bas, env) -> Vec<f64>` of shape
   [n_sp * n_sp * 2] (interleaved complex, column-major / bra-fastest stitch). This is the
   gradient collector with the `comp` loop REMOVED. Per shell pair: n_elem = ni*nj*2,
   eval_raw into `out`, then stitch column-major: dst = (col*n_sp + row)*2, src = (jj*ni+ii)*2.

5. Vendor collector `collect_vendor_spinor<F>(vendor_fn, atm, bas, env) -> Vec<f64>` gated on
   `#[cfg(has_vendor_libcint)]`, same shape, calling `vendor_fn(&mut out, &shls, atm, natm,
   bas, nbas, env)` (signature: Fn(&mut [f64], &[i32;2], &[i32], i32, &[i32], i32, &[f64]) -> i32).

6. count_mismatches and assert_any_nonzero helpers: copy verbatim from the gradient test.

7. A non-vendor smoke test (cfg(feature="cpu")) per operator: collect cintx, assert
   len == n_sp*n_sp*2 == 12*12*2 == 288, assert_any_nonzero. Add an EXPLICIT assertion that
   the (0,1) cross block has nci>1 && ncj>1 by construction (assert both shells have
   ANG_OF==1 and the two shells are distinct atoms) so the fixture's bug-exercising property
   is self-documenting and can't silently regress.

8. Three vendor parity tests gated `#[cfg(has_vendor_libcint)] #[cfg(feature="cpu")]`:
   `test_int1e_ovlp_spinor_asym_parity`, `test_int1e_kin_spinor_asym_parity`,
   `test_int1e_nuc_spinor_asym_parity`. Each: build fixture, collect vendor + cintx,
   assert_any_nonzero on both, `assert_eq!(count_mismatches(&vendor,&cintx,ATOL,RTOL), 0, ...)`.
   Use RawApiId::INT1E_{OVLP,KIN,NUC}_SPINOR and vendor_ffi::vendor_int1e_{ovlp,kin,nuc}_spinor.

CONFIRM THE BUG: run with vendor enabled (slow, allow minutes):
  `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test one_electron_scalar_spinor_parity -- --nocapture`
EXPECT: the three *_asym_parity tests FAIL with mismatch_count > 0 (the cross-block re/im
elements differ). The smoke tests pass. Record the observed mismatch counts.

IF THE PARITY TESTS PASS BEFORE THE FIX: the fixture is wrong — it has no genuine asymmetric
p+ x p+ cross block (e.g. shells share exponents/center and are accidentally symmetric, or
one side collapsed to s). Fix the fixture (force distinct exponents AND distinct centers,
both l==1) until the test fails. DO NOT proceed to Task 2 or declare success until the test
fails on the current code — a non-failing test does not prove the fix.
  </action>
  <verify>
    <automated>CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test one_electron_scalar_spinor_parity -- --nocapture 2>&1 | tee /tmp/kke_before.log; grep -E "mismatches vs vendored|FAILED|test result" /tmp/kke_before.log</automated>
  </verify>
  <done>
File compiles. The three *_asym_parity tests FAIL with mismatch_count > 0 on the CURRENT
(unfixed) code, proving the bug is real and the fixture exercises an asymmetric p+ x p+ block.
The smoke tests pass. Observed before-fix mismatch counts recorded in the run log.
  </done>
</task>

<task type="auto">
  <name>Task 2: Transpose the scalar spinor cart block to bra-major; confirm 0 mismatches</name>
  <files>crates/cintx-cubecl/src/kernels/one_electron.rs</files>
  <action>
Apply the one-block fix in the `Representation::Spinor =>` arm of the scalar 1e staging
(one_electron.rs ~line 2892-2907), mirroring the gradient arm transpose added in 260529-jtd
(~line 2712-2717). DO NOT modify `cart_to_spinor_sf_2d` or apply_bra_block in c2spinor.rs —
the gradient path now depends on its bra-major contract.

After the `n_ctr_i != 1 || n_ctr_j != 1` guard and the `kappa_i/kappa_j` lets, replace the
direct `cart_to_spinor_sf_2d::<F>(staging, &cart_blocks, ...)` call with a transposed copy:

```rust
// cart_blocks is exactly one nci*ncj block here (nctr=1 enforced above). The device
// scalar kernel emits it ket-major / bra-fastest (block[cj*nci + ci]), but
// cart_to_spinor_sf_2d reads bra-major / ket-fastest (cart[bra*ncj + ket], see
// c2spinor.rs apply_bra_block: cart[n*ncj + j]). Transpose to bra-major before the
// spin-free cart->spinor transform so the bra/ket coefficient roles line up with libcint
// c2s_sf_1e — identical to the GRADIENT arm fix (260529-jtd). For square symmetric blocks
// (an s side, or a same-shell symmetric block) this is a no-op, which is why asymmetric
// p+ x p+ cross blocks are the ones that surface the orientation.
let mut cart_bra_major = vec![0.0f64; nci * ncj];
for ic in 0..nci {
    for jc in 0..ncj {
        cart_bra_major[ic * ncj + jc] = cart_blocks[jc * nci + ic];
    }
}
cart_to_spinor_sf_2d::<F>(staging, &cart_bra_major, li, kappa_i, lj, kappa_j)?;
```

Confirm `nci`, `ncj`, `block_len` (== nci*ncj) are in scope in this arm (they are used by the
Cart arm above). Use `nci`/`ncj` directly; if a `block_len` local exists, `nci*ncj` and
`block_len` are equal here. Keep the sp_scale normalization (applied to cart_blocks earlier)
untouched — the transpose reads the already-scaled cart_blocks.

Then RUN THE FULL PROOF (vendor build is slow — allow minutes; never fabricate counts):

1. The new parity test now PASSES (0 mismatches):
   `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test one_electron_scalar_spinor_parity -- --nocapture`
   EXPECT: all three *_asym_parity tests pass, "0 mismatches vs vendored libcint" for
   int1e_{ovlp,kin,nuc}_spinor.

2. Regression guard — the jtd gradient spinor parity still passes:
   `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test one_electron_grad_spinor_parity`

3. Regression guard — scalar-spinor idempotency + arity2 parity still pass:
   `cargo test -p cintx-oracle --features cpu --test safe_api_arity2_parity`

4. cubecl crate unit tests (the in-crate spinor + scalar tests) still pass:
   `cargo test -p cintx-cubecl --features cpu`

Report honest before (Task 1) -> after mismatch counts for each operator.
  </action>
  <verify>
    <automated>CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test one_electron_scalar_spinor_parity -- --nocapture 2>&1 | tee /tmp/kke_after.log; grep -cE "0 mismatches vs vendored" /tmp/kke_after.log; grep -E "test result" /tmp/kke_after.log</automated>
    <automated>CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test one_electron_grad_spinor_parity 2>&1 | grep -E "test result"</automated>
    <automated>cargo test -p cintx-oracle --features cpu --test safe_api_arity2_parity 2>&1 | grep -E "test result"</automated>
  </verify>
  <done>
The scalar spinor arm transposes its single nci*ncj cart block to bra-major before
cart_to_spinor_sf_2d. All three new *_asym_parity tests PASS with 0 mismatches vs libcint
6.1.3 for int1e_{ovlp,kin,nuc}_spinor. cart_to_spinor_sf_2d is unchanged. The jtd gradient
spinor parity, scalar-spinor idempotency, and arity2 parity tests all still pass. cintx-cubecl
unit tests pass. Honest before/after mismatch counts reported.
  </done>
</task>

</tasks>

<verification>
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test one_electron_scalar_spinor_parity`
  -> 3 *_asym_parity tests pass (0 mismatches), smoke tests pass.
- Before-fix run of the same command FAILED (mismatch_count > 0) — proves the bug + fixture.
- jtd gradient spinor parity unchanged (still 0 mismatches).
- safe_api_arity2_parity (scalar-spinor idempotency) unchanged.
- cintx-cubecl --features cpu unit tests pass.
- Diff in one_electron.rs is confined to the Representation::Spinor scalar arm; c2spinor.rs
  untouched.
</verification>

<success_criteria>
- int1e_{ovlp,kin,nuc}_spinor are parity-clean vs libcint 6.1.3 on an ASYMMETRIC p+ x p+
  cross block (0 mismatches at atol=1e-12).
- The regression test fails before the fix and passes after — the proof is honest and the
  fixture provably exercises nci>1 && ncj>1 && i!=j.
- The fix mirrors the gradient path (260529-jtd): caller-side transpose, no change to the
  bra-major reader.
- No pre-existing tests regress.
</success_criteria>

<output>
After completion, create
`.planning/quick/260529-kke-fix-scalar-spinor-int1e-cart-to-spinor-b/260529-kke-SUMMARY.md`
with: the fix diff location, the before/after mismatch counts for all three operators, the
exact vendor cargo/env invocations used, and confirmation that c2spinor.rs was not modified.
</output>
