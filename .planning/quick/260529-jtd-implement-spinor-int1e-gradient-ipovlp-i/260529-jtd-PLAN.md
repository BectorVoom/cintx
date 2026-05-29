---
phase: quick-260529-jtd
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/cintx-cubecl/src/kernels/one_electron.rs
  - crates/cintx-oracle/src/vendor_ffi.rs
  - crates/cintx-oracle/build.rs
  - crates/cintx-oracle/tests/one_electron_grad_spinor_parity.rs
autonomous: true
requirements: [R5, D-03]
must_haves:
  truths:
    - "int1e_ipovlp_spinor, int1e_ipkin_spinor, int1e_ipnuc_spinor, int1e_iprinv_spinor evaluate (no UnsupportedApi) for nctr=1 spinor shells"
    - "Each spinor-gradient output is a 3-component, component-leading, interleaved-complex buffer matching libcint c2s_sf_1e per component"
    - "nctr>1 spinor gradient still returns UnsupportedApi (general-contraction spinor remains unsupported, same as scalar spinor path)"
    - "nuclear spinor gradients still fail-closed when nroots>5 (li+lj>8)"
    - "Vendor parity (cintx vs libcint 6.1.3) on H2O/STO-3G reports a mismatch count for all four operators"
  artifacts:
    - path: "crates/cintx-cubecl/src/kernels/one_electron.rs"
      provides: "Spinor arm of the 1e gradient staging match implementing per-component cart_to_spinor_sf_2d; removal of the R5/D-03 rejection"
      contains: "cart_to_spinor_sf_2d"
    - path: "crates/cintx-oracle/src/vendor_ffi.rs"
      provides: "vendor_int1e_ip{ovlp,kin,nuc,rinv}_spinor FFI wrappers (interleaved-complex output)"
      contains: "vendor_int1e_ipovlp_spinor"
    - path: "crates/cintx-oracle/tests/one_electron_grad_spinor_parity.rs"
      provides: "Vendor parity test for the four spinor gradient operators + nctr>1 guard test"
  key_links:
    - from: "crates/cintx-cubecl/src/kernels/one_electron.rs (gradient staging match, Representation::Spinor arm)"
      to: "crates/cintx-cubecl/src/transform/c2spinor.rs::cart_to_spinor_sf_2d"
      via: "per-component call on each (ci,cj) cart block"
      pattern: "cart_to_spinor_sf_2d::<F>"
    - from: "crates/cintx-oracle/tests/one_electron_grad_spinor_parity.rs"
      to: "crates/cintx-oracle/src/vendor_ffi.rs::vendor_int1e_ipovlp_spinor"
      via: "vendor FFI call gated on has_vendor_libcint"
      pattern: "vendor_int1e_ip.*_spinor"
---

<objective>
Implement the four spinor int1e GRADIENT operators (int1e_{ipovlp,ipkin,ipnuc,iprinv}_spinor)
that are registered in the manifest but currently rejected with `UnsupportedApi` (Risk R5 / D-03).

This is NOT a device-kernel port. The 3-component Cartesian gradient compute is ALREADY on-device
(ported in quick task 260529-j7d via `run_1e_grad_bra_on_backend` / `run_1e_nuc_grad_on_backend`).
The deliverable is:
  (a) wiring the existing on-device Cartesian gradient through the EXISTING host-side spin-free
      cart→spinor transform `cart_to_spinor_sf_2d`, applied PER COMPONENT — exactly mirroring how
      the SCALAR spinor 1e path already works (one_electron.rs ~line 2857-2871); and
  (b) a vendor oracle parity test proving cintx matches libcint 6.1.3.

The cart→spinor transform STAYS HOST-SIDE by established project convention (every prior family
keeps c2s/c2spinor host-only; the scalar spinor 1e path is device-cart-compute → host transform).

Purpose: Close the last UnsupportedApi gap in the 1e gradient family so the spinor derivative
operators return correct libcint-compatible output instead of an error.
Output: Working spinor gradient path + a double-gated vendor parity test with an honest mismatch count.
</objective>

<execution_context>
@/home/user/Documents/workspace/cintx/.claude/get-shit-done/workflows/execute-plan.md
@/home/user/Documents/workspace/cintx/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@./CLAUDE.md

<interfaces>
<!-- Contracts the executor needs. Extracted from codebase — no exploration required. -->

cart_to_spinor_sf_2d (crates/cintx-cubecl/src/transform/c2spinor.rs ~line 531):
```rust
pub fn cart_to_spinor_sf_2d<F: CintFloat>(
    staging: &mut [F],   // must have >= di*dj*2 F elements; written interleaved complex
    cart: &[f64],        // single nci*ncj block; SAME layout the SCALAR spinor path passes
    li: u8, kappa_i: i16,
    lj: u8, kappa_j: i16,
) -> Result<(), cintxRsError>;
// Output: di*dj*2 F elements, column-major (j_spinor outer, i_spinor inner),
//   staging[(j*di + i)*2] = re, +1 = im. di = spinor_len(li, kappa_i), dj = spinor_len(lj, kappa_j).
```

spinor_len (crates/cintx-cubecl/src/transform/c2spinor.rs ~line 25):
```rust
pub fn spinor_len(l: u8, kappa: i32) -> usize {
    if kappa < 0 { 2*l as usize + 2 } else if kappa > 0 { 2*l as usize } else { 4*l as usize + 2 }
}
```

SCALAR spinor arm to mirror (one_electron.rs ~line 2857-2871):
```rust
Representation::Spinor => {
    if n_ctr_i != 1 || n_ctr_j != 1 {
        return Err(cintxRsError::UnsupportedApi {
            requested: "spinor 1e with general contraction (nctr>1)".to_owned(),
        });
    }
    let kappa_i = shell_i.kappa;
    let kappa_j = shell_j.kappa;
    cart_to_spinor_sf_2d::<F>(staging, &cart_blocks, li, kappa_i, lj, kappa_j)?;
}
```

Gradient cart layout (one_electron.rs ~line 2561-2566, already in scope at the staging match):
- `cart_3comp`: per (ci,cj) pair at offset `(ci*n_ctr_j+cj)*total_len`, then component `comp*block_len`,
  then element `cj_idx*nci + ci_idx`. `block_len = nci*ncj`, `total_len = 3*block_len`.
- Each per-component block (`&cart_3comp[base + comp*block_len .. + block_len]`) is layout-IDENTICAL
  to a SCALAR single block — VERIFIED: the Spheric gradient arm and the scalar Spheric arm both feed
  these blocks to `cart_to_sph_1e` identically. So each per-component block can be passed to
  `cart_to_spinor_sf_2d` exactly as the scalar single block is passed.

Existing vendor scalar spinor FFI wrapper to model (crates/cintx-oracle/src/vendor_ffi.rs ~line 1316):
```rust
pub fn vendor_int1e_ovlp_spinor(out: &mut [f64], shls: &[i32; 2], atm: &[i32], natm: i32,
    bas: &[i32], nbas: i32, env: &[f64]) -> i32 {
    unsafe { ffi::int1e_ovlp_spinor(out.as_mut_ptr(), ptr::null_mut(), shls.as_ptr() as *mut i32,
        atm.as_ptr() as *mut i32, natm, bas.as_ptr() as *mut i32, nbas,
        env.as_ptr() as *mut f64, ptr::null_mut(), ptr::null_mut()) }
}
```
Output buffer for the GRADIENT spinor wrappers: `3 * ni_sp * nj_sp * 2` f64 (3 components ×
interleaved-complex). ni_sp = CINTcgto_spinor(shls[0]), nj_sp = CINTcgto_spinor(shls[1]).

bindgen allowlist (crates/cintx-oracle/build.rs ~line 358): the four spinor gradient symbols
`int1e_ipovlp_spinor|int1e_ipkin_spinor|int1e_ipnuc_spinor|int1e_iprinv_spinor` are NOT yet in the
`.allowlist_function(...)` regex but ARE declared in libcint-master/include/cint_funcs.h (verified) —
so adding them to the allowlist regex is sufficient; no supplemental header needed.

Vendor double-gating (verified in build.rs + safe_api_arity2_parity.rs): real parity only runs when
BOTH `--features cpu` AND env `CINTX_ORACLE_BUILD_VENDOR=1` are set. The vendor matrix collectors and
parity asserts are guarded by `#[cfg(has_vendor_libcint)]` (set by build.rs only when the env var is
present) plus `#[cfg(feature = "cpu")]`. Without both, the test compiles but the vendor body is
cfg'd out (silent no-op).

cintx side comparison: use the RAW path `eval_raw(api_id, Some(&mut out), None, &shls, atm, bas, env,
None, None)` — for spinor it writes the interleaved-complex staging directly (matching vendor layout).
This is the SAME approach the sph/cart grad parity tests use (one_electron_grad_parity.rs ~line 169),
NOT the safe-API facade (which projects to real Vec<f64> and is why the scalar spinor tests are
idempotency-only). RawApiId variants exist: INT1E_IPOVLP_SPINOR, INT1E_IPKIN_SPINOR,
INT1E_IPNUC_SPINOR, INT1E_IPRINV_SPINOR (compare.rs ~line 333-342).

iprinv env layout (one_electron_nuc_grad_parity.rs): rinv origin lives at env[PTR_RINV_ORIG=4..6];
use a PTR_ENV_START=20-aligned fixture so atom coords (env[20..]) never clobber the rinv slot.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Replace the spinor-gradient rejection tests + add vendor FFI/bindgen wiring (RED + scaffolding)</name>
  <files>crates/cintx-oracle/src/vendor_ffi.rs, crates/cintx-oracle/build.rs, crates/cintx-oracle/tests/one_electron_grad_spinor_parity.rs, crates/cintx-cubecl/src/kernels/one_electron.rs</files>
  <behavior>
    - The four `test_ip*_spinor_returns_unsupported` tests (one_electron.rs ~line 4344, 4647, 4726,
      and any ipkin equivalent) currently assert the OLD R5/D-03 rejection. They must be REPLACED so
      that after Task 2 they assert the NEW behavior:
        * Test: dispatching int1e_ipovlp/ipkin/ipnuc/iprinv with Representation::Spinor and nctr=1
          returns Ok (no UnsupportedApi) and writes a 3-component interleaved-complex staging of the
          expected length (3 * di * dj * 2, di=dj=spinor_len for the s/s or s/p fixture used).
        * Test: nctr>1 spinor gradient STILL returns UnsupportedApi (general-contraction guard).
      At the end of THIS task these replacement tests are written but expected to FAIL (RED), because
      the implementation lands in Task 2. Confirm RED by running them and observing the
      UnsupportedApi/`unreachable!` failure.
    - New file crates/cintx-oracle/tests/one_electron_grad_spinor_parity.rs: a vendor parity test for
      all four operators modeled on one_electron_grad_parity.rs + one_electron_nuc_grad_parity.rs:
        * PTR_ENV_START=20-aligned H2O/STO-3G fixture (model the env-start fixture in
          one_electron_nuc_grad_parity.rs so the iprinv rinv-origin at env[4..6] is never clobbered).
          Set shell kappa=0 (both GT+LT blocks) for spinor; reuse the STO-3G primitive data.
        * cintx collector via `eval_raw(RawApiId::INT1E_IP*_SPINOR, Some(&mut out), ...)` producing
          `3 * ni_sp * nj_sp * 2` interleaved-complex per shell pair, stitched component-leading.
        * vendor collector via the new `vendor_int1e_ip*_spinor` wrappers, guarded
          `#[cfg(has_vendor_libcint)]` + `#[cfg(feature = "cpu")]`.
        * For iprinv: set the rinv origin via an `env_with_rinv_origin`-style helper (env[4..6]),
          sweeping at least one nucleus, exactly like one_electron_nuc_grad_parity.rs.
        * `count_mismatches(vendor, cintx, atol=1e-12, rtol=0.0)` and assert mismatches==0 per operator.
      This parity test will only meaningfully run after Task 2 lands + vendor is built (Task 3 runs it).
  </behavior>
  <action>
    1. In crates/cintx-oracle/src/vendor_ffi.rs, add four wrappers `vendor_int1e_ipovlp_spinor`,
       `vendor_int1e_ipkin_spinor`, `vendor_int1e_ipnuc_spinor`, `vendor_int1e_iprinv_spinor` modeled
       VERBATIM on `vendor_int1e_ovlp_spinor` (~line 1316) but calling `ffi::int1e_ip*_spinor`. These
       are gradient ops: document that `out` must be `3 * ni_sp * nj_sp * 2` f64 (3 components ×
       interleaved complex). Keep the existing `#[cfg(...)]`/feature gating consistent with the
       neighbouring scalar spinor wrappers.
    2. In crates/cintx-oracle/build.rs, add `int1e_ipovlp_spinor|int1e_ipkin_spinor|int1e_ipnuc_spinor|int1e_iprinv_spinor`
       to the `.allowlist_function(...)` regex (~line 358). The symbols are in cint_funcs.h (verified)
       so NO supplemental-header change is needed. Keep the build.rs vendor double-gate untouched.
    3. Write crates/cintx-oracle/tests/one_electron_grad_spinor_parity.rs per the <behavior> block.
       Capture the EXACT cfg gates (`#[cfg(has_vendor_libcint)]`, `#[cfg(feature = "cpu")]`) and the
       eval_raw signature from one_electron_grad_parity.rs. Use ATOL=1e-12, RTOL=0.0.
    4. In crates/cintx-cubecl/src/kernels/one_electron.rs, REPLACE the four
       `test_*_spinor_returns_unsupported` tests with the new-behavior tests described in <behavior>.
       Do NOT yet implement the kernel spinor arm — that is Task 2. Leave the rejection at ~line
       2509-2516 and the `unreachable!` arm at ~line 2689-2691 in place for now so the new tests are
       RED.
    5. Run the replaced cubecl tests to confirm RED.
  </action>
  <verify>
    <automated>cargo test -p cintx-cubecl --features cpu spinor_grad 2>&1 | tail -30 || true; cargo build -p cintx-oracle --features cpu 2>&1 | tail -15</automated>
  </verify>
  <done>Four vendor spinor-gradient FFI wrappers compile; bindgen allowlist updated; new parity test file compiles (vendor body cfg'd out without env var); replaced cubecl tests exist and FAIL (RED) against the still-present rejection.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Implement the spinor arm of the 1e gradient staging (GREEN)</name>
  <files>crates/cintx-cubecl/src/kernels/one_electron.rs</files>
  <behavior>
    - With the rejection removed and the spinor arm implemented, the four replaced cubecl tests from
      Task 1 now PASS:
        * nctr=1 spinor ipovlp/ipkin/ipnuc/iprinv return Ok and write 3 * di * dj * 2 interleaved-complex
          staging elements (di=spinor_len(li,kappa_i), dj=spinor_len(lj,kappa_j)).
        * nctr>1 spinor gradient still returns UnsupportedApi.
    - All pre-existing 1e gradient sph/cart parity + determinism tests, the scalar spinor path, and the
      nctr>1 scalar spinor guard remain PASSING (no regression).
  </behavior>
  <action>
    1. Remove the R5/D-03 spinor-gradient rejection block at one_electron.rs ~line 2509-2516
       (`if (is_ipovlp||is_ipkin||is_ipnuc||is_iprinv) && representation==Spinor → UnsupportedApi`).
    2. Replace the `Representation::Spinor => unreachable!("spinor gradient rejected earlier")` arm at
       ~line 2689-2691 with a real implementation that MIRRORS the scalar spinor arm (~line 2857-2871)
       and the Spheric gradient arm's per-component loop structure (~line 2635-2663):
         - Keep the SAME nctr>1 guard FIRST: `if n_ctr_i != 1 || n_ctr_j != 1 { return Err(UnsupportedApi
           { requested: "spinor 1e gradient with general contraction (nctr>1)".to_owned() }); }`.
         - Compute `di = spinor_len(li, shell_i.kappa as i32)`, `dj = spinor_len(lj, shell_j.kappa as i32)`,
           `spinor_block = di * dj * 2` (interleaved complex), mirroring how Spheric uses `sph_block`
           and Cart uses `cart_block` as the per-component staging stride.
         - For nctr=1 the cart pair base is `0` and `total_len = 3*block_len`. For each `comp in 0..3`:
           slice the per-component cart block `&cart_3comp[comp*block_len .. comp*block_len + block_len]`
           and call `cart_to_spinor_sf_2d::<F>(&mut staging[comp*spinor_block .. comp*spinor_block + spinor_block],
           &block, li, shell_i.kappa, lj, shell_j.kappa)?` — passing each per-component block EXACTLY as
           the scalar path passes its single block (same orientation; verified layout-identical).
         - Use `shell_i.kappa` / `shell_j.kappa` (type i16) directly as the scalar arm does.
       These are SPIN-FREE operators (no sigma): use `cart_to_spinor_sf_2d` (the c2s_sf_1e analogue),
       matching how libcint drives int1e_ip*_spinor through c2s_sf_1e — do NOT use a spin-included
       (c2s_si) transform.
    3. Do NOT modify run_1e_grad_bra_on_backend / run_1e_nuc_grad_on_backend or any #[cube] device
       kernel. Do NOT touch the sp-normalization scaling, the nroots>5 fail-closed guard, or the
       not0 sentinel (they already wrap the spinor arm correctly).
    4. The not0 sentinel and ExecutionStats return after the staging match already handle the spinor
       buffer because staging is sized by the planner for the spinor component_rank; confirm the
       staging length used by the sentinel covers `3 * spinor_block`.
  </action>
  <verify>
    <automated>cargo test -p cintx-cubecl --features cpu 2>&1 | tail -40</automated>
  </verify>
  <done>The four replaced spinor-gradient cubecl tests PASS; the nctr>1 spinor-gradient guard fires; all pre-existing cintx-cubecl tests (sph/cart gradient parity, scalar spinor, scalar nctr>1 guard) still PASS.</done>
</task>

<task type="auto">
  <name>Task 3: Build vendor + run the spinor-gradient parity test and report the honest mismatch count</name>
  <files>crates/cintx-oracle/tests/one_electron_grad_spinor_parity.rs</files>
  <action>
    1. Run the new vendor parity test with BOTH gates enabled (building the libcint vendor is slow —
       allow generous time):
         `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test one_electron_grad_spinor_parity -- --nocapture`
       Capture the exact env+cargo invocation from safe_api_arity2_parity.rs / one_electron_nuc_grad_parity.rs
       gating to ensure `has_vendor_libcint` is actually set (the vendor build must compile).
    2. Report the mismatch count vs libcint for EACH of the four operators (ipovlp/ipkin/ipnuc/iprinv)
       HONESTLY — do not fabricate. If any operator mismatches, capture the first few mismatching
       (index, vendor, cintx, abs/rel diff) tuples and diagnose (likely candidates: complex-interleave
       orientation, kappa GT/LT block ordering, per-component cart block transpose, or rinv-origin env
       slot). Fix in one_electron.rs (Task 2 arm) or the test fixture as appropriate and re-run.
    3. Also run the broader oracle gate once to confirm no regression in the existing 1e gradient
       sph/cart parity tests under vendor:
         `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test one_electron_grad_parity 2>&1 | tail -20`
       (Note: per project memory, `test_f32_int3c2e_sph_parity` and the lib-level
       `CINTshells_cart_offset[4]` failure are PRE-EXISTING baseline noise on this branch — do NOT
       attempt to fix them; just don't introduce NEW failures.)
  </action>
  <verify>
    <automated>CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test one_electron_grad_spinor_parity -- --nocapture 2>&1 | tail -40</automated>
  </verify>
  <done>The vendor parity test actually RAN under `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1` (vendor built, has_vendor_libcint active); the SUMMARY reports the mismatch count vs libcint 6.1.3 for all four spinor gradient operators (target: 0 mismatches at atol=1e-12); no new failures in one_electron_grad_parity.</done>
</task>

</tasks>

<verification>
- All four int1e_{ipovlp,ipkin,ipnuc,iprinv}_spinor evaluate without UnsupportedApi for nctr=1.
- nctr>1 spinor gradient still returns UnsupportedApi.
- nroots>5 nuclear spinor gradient still fails closed.
- Vendor parity test ran with both gates and reports an honest mismatch count (target 0).
- No regression in existing cubecl tests or one_electron_grad_parity sph/cart vendor tests.
</verification>

<success_criteria>
- `cargo test -p cintx-cubecl --features cpu` passes (incl. the four replaced spinor-gradient tests + nctr>1 guard).
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test one_electron_grad_spinor_parity` ran with vendor built; mismatch count vs libcint reported (target 0 at atol=1e-12).
- cart→spinor transform remains host-side; no device kernel modified.
- Existing passing tests preserved.
</success_criteria>

<output>
After completion, create `.planning/quick/260529-jtd-implement-spinor-int1e-gradient-ipovlp-i/260529-jtd-SUMMARY.md`
</output>
