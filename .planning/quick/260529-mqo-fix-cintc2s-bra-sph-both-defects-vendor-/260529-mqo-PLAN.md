---
phase: 260529-mqo
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/cintx-cubecl/src/transform/c2s.rs
  - crates/cintx-compat/src/transform.rs
  - crates/cintx-oracle/src/vendor_ffi.rs
  - crates/cintx-oracle/tests/cintc2s_bra_sph_parity.rs
autonomous: true
requirements: [HELP-02]

must_haves:
  truths:
    - "cintx CINTc2s_bra_sph applies the real per-l c2s coefficient transform (d/f/g write spheric values, not raw cart)"
    - "cintx CINTc2s_bra_sph remains identity for l=0 and l=1 (internal CINTc2s_ket_sph/_sph1 callers unchanged)"
    - "vendor_CINTc2s_bra_sph copies the libcint RETURNED pointer into sph (handles l<2 return-gcart aliasing)"
    - "cintx and vendor CINTc2s_bra_sph agree at atol=1e-12 for l in 0..=4"
    - "cargo test -p cintx-compat --lib is green"
    - "full vendor oracle gate no longer reports the CINTc2s_bra_sph helper-parity mismatch"
  artifacts:
    - path: "crates/cintx-compat/src/transform.rs"
      provides: "Real per-l bra cart->sph transform in CINTc2s_bra_sph"
      contains: "c2s_coeff"
    - path: "crates/cintx-oracle/src/vendor_ffi.rs"
      provides: "vendor_CINTc2s_bra_sph that copies the returned pointer into sph"
      contains: "copy"
    - path: "crates/cintx-oracle/tests/cintc2s_bra_sph_parity.rs"
      provides: "Vendor parity test over l in 0..=4"
      contains: "vendor_CINTc2s_bra_sph"
  key_links:
    - from: "crates/cintx-compat/src/transform.rs"
      to: "cintx_cubecl::transform::c2s::c2s_coeff"
      via: "per-ket per-sph-row accumulation"
      pattern: "c2s_coeff"
    - from: "crates/cintx-oracle/src/vendor_ffi.rs"
      to: "ffi::CINTc2s_bra_sph return pointer"
      via: "std::ptr::copy from returned ptr into sph"
      pattern: "ptr::copy"
---

<objective>
Fix BOTH root-caused defects in `CINTc2s_bra_sph` so cintx matches libcint 6.1.3 for all l, then prove it
with a lib unit test and a vendor parity test, then re-run the full vendor oracle gate.

Defect A (oracle FFI wrapper): `vendor_CINTc2s_bra_sph` discards the `*mut f64` returned by libcint. For
l<2, libcint returns `gcart` WITHOUT writing `sph`, so the harness reads zero-init -> spurious vendor=0.
Defect B (cintx helper): `CINTc2s_bra_sph` copies cart through and calls the no-op
`cart_to_spheric_staging` -> it is the identity for ALL l, which is WRONG for l>=2 (d/f/g must apply the
c2s coefficients).

Purpose: clear the LAST known helper/transform parity blocker under the vendor gate.
Output: two production fixes, one extended lib test, one new vendor parity test, and an honest report of
what the full vendor gate reaches next.
</objective>

<execution_context>
@/home/user/Documents/workspace/cintx/.claude/get-shit-done/workflows/execute-plan.md
@/home/user/Documents/workspace/cintx/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@./CLAUDE.md
@.planning/STATE.md

<interfaces>
<!-- From crates/cintx-cubecl/src/transform/c2s.rs (in scope of this plan). -->
<!-- These are the contracts the executor uses directly. NO codebase exploration needed. -->

```rust
// c2s.rs — dimension helpers (already pub)
pub fn ncart(l: u8) -> usize;   // (l+1)(l+2)/2
pub fn nsph(l: u8) -> usize;    // 2l+1

// c2s.rs — coefficient accessor: CURRENTLY PRIVATE `fn c2s_coeff(...)`.
// Task 2 makes it `pub fn`. Layout: rows = spheric m (0-based, m=-l..+l),
// cols = cartesian (libcint ordering). l=0 and l=1 tables are the IDENTITY
// (C2S_L0=[[1.0]], C2S_L1 = 3x3 identity, non-PYPZPX). Returns 0.0 for l>4.
pub fn c2s_coeff(l: u8, m_row: usize, cart_col: usize) -> f64;

// c2s.rs — reference bra-then-ket transform pattern (DO NOT call; it does a 2-axis
// transform). The single-axis bra accumulation for CINTc2s_bra_sph is:
//   sph[k*nsph + m] = Σ_c c2s_coeff(l, m, c) * cart[k*ncart + c]
pub fn cart_to_sph_1e<F: CintFloat>(cart_buf: &[F], sph_buf: &mut [F], li: u8, lj: u8);

// c2s.rs — DO NOT delete. Still called by transform/mod.rs:16 for Representation::Spheric.
pub fn cart_to_spheric_staging(staging: &mut [f64]) -> Result<(), cintxRsError>;
```

```rust
// crates/cintx-compat/src/transform.rs — current signatures (preserve these exactly)
pub fn CINTc2s_bra_sph(sph: &mut [f64], nket: i32, cart: &[f64], l: i32) -> Result<(), cintxRsError>;
// Internal callers that pass l=0 (must keep working — identity path):
//   CINTc2s_ket_sph(sph, nket, cart, l)        -> CINTc2s_bra_sph(sph, 0, cart, 0)
//   CINTc2s_ket_sph1(sph, cart, lds, ldc, l)   -> CINTc2s_bra_sph(sph, 0, cart, 0)
fn copy_cart_into_target(target: &mut [f64], cart: &[f64]) -> Result<(), cintxRsError>;
```

```rust
// crates/cintx-oracle/src/vendor_ffi.rs — current wrapper (Defect A).
// The bindgen extern `ffi::CINTc2s_bra_sph` RETURNS *mut f64 (gsph for l>=2, gcart for l<2).
pub fn vendor_CINTc2s_bra_sph(sph: &mut [f64], nket: i32, cart: &[f64], l: i32);
```

<!-- libcint 6.1.3 ground truth (libcint-master/src/cart2sph.c), confirmed by source read:
  CINTc2s_bra_sph dispatches on l. Oracle build does NOT define PYPZPX.
  l=0 s_bra_cart2spheric: `return gcart;` (does NOT write gsph).
  l=1 p_bra_cart2spheric (non-PYPZPX): `return gcart;` (does NOT write gsph) — px,py,pz identity.
  l>=2 d/f/g/a_bra_cart2spheric: apply coeffs, write gsph ket-blocked (per ket: write nsph=2l+1,
    advance gsph+=nsph, gcart+=ncart), `return gsph`.
  => For l<2 the returned pointer aliases the `cart` INPUT, NOT `sph`. -->
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add RED tests — lib d-transform unit test + vendor parity test</name>
  <files>crates/cintx-compat/src/transform.rs, crates/cintx-oracle/tests/cintc2s_bra_sph_parity.rs</files>
  <behavior>
    LIB unit tests (crates/cintx-compat/src/transform.rs `mod tests`):
    - REPLACE the existing dimensionally-wrong `spherical_transform_entry_points_work` test (~line 242-250;
      it calls CINTc2s_bra_sph with l=1, nket=1 and 4-element buffers — but p-shell is 3 cart / 3 sph) with:
      (a) `bra_sph_l1_identity`: l=1, nket=1, cart=[1.0,2.0,3.0] (3=ncart(1)), sph len 3 (=nsph(1)).
          Assert sph == [1.0,2.0,3.0] (libcint p_bra non-PYPZPX is identity).
      (b) `bra_sph_l0_identity`: l=0, nket=1, cart=[7.0], sph len 1 -> assert sph==[7.0]. Pins the path
          internal CINTc2s_ket_sph/_sph1 depend on (they pass l=0).
      (c) `bra_sph_l2_d_transform` (the RED one): l=2, nket=1, cart=[1.0,2.0,3.0,4.0,5.0,6.0] (6=ncart(2)),
          sph len 5 (=nsph(2)). Compute expected from the C2S_L2 matrix in c2s.rs
          (rows m=-2..+2; cols xx,xy,xz,yy,yz,zz):
            m=-2 (dxy):  1.092548430592079070 * cart[1]
            m=-1 (dyz):  1.092548430592079070 * cart[4]
            m= 0 (dz2): -0.315391565252520002*cart[0] -0.315391565252520002*cart[3] +0.630783130505040012*cart[5]
            m=+1 (dxz):  1.092548430592079070 * cart[2]
            m=+2 (dx2y2):0.546274215296039535*cart[0] -0.546274215296039535*cart[3]
          Assert |sph[m]-expected[m]| < 1e-12 for each m. RED against the current stub (returns raw cart).
      (d) `bra_sph_l2_nket2_blocking`: l=2, nket=2, cart = 12 values (two ket blocks of 6), sph len 10
          (=2*nsph(2)). Assert sph[0..5] and sph[5..10] each equal the d-transform of their own cart block.
          Verifies the ket-blocked layout (sph row fastest, advance per ket).
    - The l=0 and l=1 identity tests are expected to PASS even before the fix (stub is identity there) —
      intentional: they pin behavior internal callers rely on.

    VENDOR parity test (NEW file crates/cintx-oracle/tests/cintc2s_bra_sph_parity.rs):
    - Model EXACTLY on crates/cintx-oracle/tests/cintgto_norm_parity.rs:
        `#![cfg(any(feature = "cpu", feature = "rocm"))]`
        `const ATOL: f64 = 1e-12;` (with `#[allow(dead_code)]`).
        Non-vendor smoke `#[cfg(feature = "cpu")] fn cintc2s_bra_sph_smoke()`: call cintx
          `cintx_compat::transform::CINTc2s_bra_sph` for l=2 nket=1 with the same cart=[1.0..6.0] input,
          assert output finite AND equals the d-transform expected values from lib test (c). Keeps the file
          from being a no-op without the vendor build.
        Vendor `#[cfg(has_vendor_libcint)] #[cfg(feature = "cpu")] fn cintc2s_bra_sph_matches_vendor()`.
    - Vendor test body: for l in 0..=4 and nket in [1,2]: build deterministic cart of len nket*ncart(l)
      (value = (idx+1) as f64 * 0.1). Allocate cintx_out and vendor_out, each len nket*nsph(l), zero-init.
      Call cintx_compat::transform::CINTc2s_bra_sph(&mut cintx_out, nket, &cart, l).unwrap() and
      vendor_ffi::vendor_CINTc2s_bra_sph(&mut vendor_out, nket, &cart, l). Compare element-wise at ATOL,
      accumulate `mismatches` + report string (l, nket, idx, cintx, vendor, diff). assert_eq!(mismatches, 0, ...).
      Use ncart/nsph from cintx_cubecl::transform::c2s. RED until BOTH fixes land.
    - It is an integration test under crates/cintx-oracle/tests/ — no Cargo.toml change needed.
  </behavior>
  <action>
    Write the two test sets above. DO NOT touch production logic in this task — these MUST be RED:
    the lib d-transform test fails (stub returns raw cart); the vendor parity test fails under the vendor
    build (Defects A and B). Use `.unwrap()` on Results like existing tests (preserve typed-error contract).
  </action>
  <verify>
    <automated>cargo test -p cintx-compat --lib bra_sph 2>&1 | grep -E "bra_sph_l2_d_transform|test result"; echo "--- expect bra_sph_l2_* FAILING (RED), l0/l1 passing ---"; cargo test -p cintx-oracle --no-run 2>&1 | tail -3</automated>
  </verify>
  <done>bra_sph_l0/l1 identity tests PASS; bra_sph_l2_d_transform and bra_sph_l2_nket2_blocking FAIL (RED); new cintc2s_bra_sph_parity.rs compiles and is listed by cintx-oracle test harness.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Fix both defects, go GREEN, run full vendor gate</name>
  <files>crates/cintx-cubecl/src/transform/c2s.rs, crates/cintx-compat/src/transform.rs, crates/cintx-oracle/src/vendor_ffi.rs</files>
  <behavior>
    After this task: lib bra_sph_l2 tests GREEN, vendor parity test GREEN (0 mismatches), full vendor gate
    no longer reports the CINTc2s_bra_sph mismatch.
  </behavior>
  <action>
    1. crates/cintx-cubecl/src/transform/c2s.rs — make the coefficient accessor public:
       change `fn c2s_coeff(l: u8, m_row: usize, cart_col: usize) -> f64` (~line 181) to `pub fn c2s_coeff(...)`.
       Do NOT change the no-op `cart_to_spheric_staging` (transform/mod.rs:16 still calls it). Leave the
       in-module callers and the duplicate ecp.rs `c2s_coeff` (separate fn) untouched.

    2. crates/cintx-compat/src/transform.rs — implement the real per-l bra transform in `CINTc2s_bra_sph`
       (replace the `copy_cart_into_target` + `cart_to_spheric_staging` body). Mirror libcint's
       *_bra_cart2spheric (ket-blocked, sph-row fastest):
         - Let `lu = l as u8`; nc = c2s::ncart(lu); ns = c2s::nsph(lu); let nk = nket.max(0) as usize.
         - Validate buffers with the typed error (NO panic): require cart.len() >= nk*nc and sph.len() >= nk*ns,
           else return `cintxRsError::BufferTooSmall { required, provided }` (same pattern as copy_cart_into_target).
         - For k in 0..nk, for m in 0..ns:
             sph[k*ns + m] = Σ_{c in 0..nc} c2s::c2s_coeff(lu, m, c) * cart[k*nc + c]
         - Import c2s_coeff via the existing `use cintx_cubecl::transform::{c2s, ...}` (add `c2s::c2s_coeff` or
           call `c2s::c2s_coeff(...)` fully qualified). Keep `ncart` import; add `nsph` import as needed.
       For l=0 and l=1 the coeff table is identity, so internal CINTc2s_ket_sph/_sph1 (l=0) are preserved.
       NOTE on l>4: c2s_coeff returns 0.0 for l>4 (would zero the output). The vendor gate only exercises
       l in 0..=4 here; do not add l>4 support — but if you want a guard, return BufferTooSmall is wrong;
       instead leave as-is (matches "returns 0.0" accessor contract) and note it in the summary. Do NOT
       silently mis-handle — just document.

    3. crates/cintx-oracle/src/vendor_ffi.rs — fix `vendor_CINTc2s_bra_sph` (~line 1021) to copy the
       RETURNED pointer into `sph`:
         let ret = ffi::CINTc2s_bra_sph(sph.as_mut_ptr(), nket, cart.as_ptr() as *mut f64, l);
         let n = (nket.max(0) as usize) * ((2 * l.max(0) + 1) as usize);   // nket * nsph(l)
         let n = n.min(sph.len());
         if !ret.is_null() && ret != sph.as_mut_ptr() {
             std::ptr::copy(ret, sph.as_mut_ptr(), n);   // memmove-safe; ret may alias `cart` (lives across call)
         }
       (For l>=2 libcint already wrote sph and returns sph -> the != guard skips the redundant copy. For l<2
       ret aliases the `cart` input, which outlives the call -> copy is sound.) Keep the wrapper signature
       (-> ()) unchanged. Update the doc comment to reflect the return-pointer copy.

    4. DO NOT modify crates/cintx-oracle/src/compare.rs — its CINTc2s_bra_sph comparison (~725-750) allocates
       vendor_out=n_sph and reads it after the wrapper call; fixing the wrapper populates it correctly.

    5. Run the gates (see <verify>). Then run the FULL VENDOR GATE verbatim (slow — allow generous time):
         CINTX_BACKEND=cpu CINTX_ORACLE_BUILD_VENDOR=1 cargo run --locked --manifest-path xtask/Cargo.toml -- oracle-compare --profiles "base,with-f12,with-4c1e,with-f12+with-4c1e" --include-unstable-source false
       Confirm the CINTc2s_bra_sph mismatch is GONE. This was the LAST known helper-parity blocker; report
       what the gate reaches NEXT — it should now clear the entire helper/transform block and advance into
       `verify_legacy_wrapper_parity` + numeric INTEGRAL parity (never exercised under vendor on this branch).
       Report HONESTLY whether all 4 profiles pass clean OR the next blocker verbatim. Do NOT fix
       newly-surfaced downstream issues — just note them. NEVER fabricate a pass.

    6. Commit CODE atomically (not docs): the 3 production files + 2 test files in one commit.
  </action>
  <verify>
    <automated>cargo test -p cintx-compat --lib 2>&1 | tail -3; echo "=== vendor parity (double-gated) ==="; CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu cintc2s_bra_sph 2>&1 | tail -6</automated>
  </verify>
  <done>`cargo test -p cintx-compat --lib` green; vendor parity test `cintc2s_bra_sph_matches_vendor` reports 0 mismatches; full vendor gate run completed and CINTc2s_bra_sph mismatch confirmed gone; next blocker (if any) reported verbatim; code committed (docs NOT committed).</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| cintx-oracle FFI -> vendored libcint C | Raw pointers cross into C; returned `*mut f64` may alias the `cart` input for l<2. |
| compat lib API -> caller buffers | Caller-provided `sph`/`cart` slices; lengths must be validated before indexed writes. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-mqo-01 | Tampering | `vendor_CINTc2s_bra_sph` `std::ptr::copy` from returned ptr | mitigate | Clamp n to `sph.len()` and null/alias-guard the source ptr; `ptr::copy` is memmove-safe for the cart-aliasing case. |
| T-mqo-02 | Denial of Service | `CINTc2s_bra_sph` indexed writes into `sph`/`cart` | mitigate | Validate `cart.len() >= nk*nc` and `sph.len() >= nk*ns`; return typed `BufferTooSmall`, never panic/OOB. |
| T-mqo-03 | Information Disclosure | reading uninitialized vendor_out (the original Defect A) | mitigate | Copy the libcint-returned buffer into `sph` so harness never reads zero-init for l<2. |
</threat_model>

<verification>
- `cargo test -p cintx-compat --lib` green (all bra_sph_l0/l1/l2 tests pass).
- Vendor parity `cintc2s_bra_sph_matches_vendor` reports 0 mismatches at atol=1e-12 for l in 0..=4, nket in {1,2}.
- Full vendor gate (4 profiles) run verbatim; CINTc2s_bra_sph mismatch gone; next reached stage reported honestly.
</verification>

<success_criteria>
- Defect A fixed (wrapper copies returned pointer) and Defect B fixed (real per-l d/f/g transform).
- compare.rs untouched; l=0/l=1 internal-caller behavior preserved; typed-error/no-panic contract intact.
- Code committed atomically; docs not committed.
- Honest report: all 4 vendor profiles pass clean OR the next verbatim blocker (noted, not fixed).
</success_criteria>

<output>
After completion, create `.planning/quick/260529-mqo-fix-cintc2s-bra-sph-both-defects-vendor-/260529-mqo-SUMMARY.md`
</output>
