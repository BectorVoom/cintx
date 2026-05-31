# Phase 27: Spinor-Derivative Transform (Gap B1) - Pattern Map

**Mapped:** 2026-05-31
**Files analyzed:** 8 (5 modified, 3 created/extended)
**Analogs found:** 8 / 8 (all in-repo; this phase is composition + verification, not new numerics)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/cintx-cubecl/src/transform/c2spinor.rs` (MODIFY: add `cart_to_spinor_sf_derivative_2d` / `_3c2e`) | transform/utility | transform (cart→spinor fold) | `cart_to_spinor_sf_2d` (L531), `cart_to_spinor_sf_3c2e` (L1281), inline rank-3 loop @ `one_electron.rs` L9919-9965 | exact (same file + proven inline) |
| `crates/cintx-cubecl/src/kernels/one_electron.rs` (MODIFY: rewire rank-3 inline; lift rank-9/27/81 rejections) | kernel launcher | request-response (eval_raw dispatch) | inline rank-3 arm @ L9919-9965 (passing) + scalar-spinor arm @ L10129 | exact (the byte-identity-proven block) |
| `crates/cintx-cubecl/src/kernels/center_3c2e.rs` (MODIFY: lift ip1/ip2 spinor rejections) | kernel launcher | request-response | scalar-spinor 3c2e arm @ L3276-3284 (calls `cart_to_spinor_sf_3c2e`) | exact (same call, no deriv loop) |
| `crates/cintx-cubecl/src/kernels/center_2c2e.rs` (MODIFY: lift ip1/ip2 spinor rejection @ L629) | kernel launcher | request-response | scalar-spinor 2c2e arm @ L1082-1086 (calls `cart_to_spinor_sf_2d`) | exact (same call, no deriv loop) |
| `crates/cintx-ops/generated/compiled_manifest.lock.json` (MODIFY: flip `oracle_covered`) | config/manifest | batch (coverage bookkeeping) | existing `oracle_covered: true` rows (e.g. L10) vs `false` rows (L3889) | exact (single-field flip) |
| `xtask/src/oracle_covered_update.rs` (MODIFY: lift R5/D-03 note @ L13-40) | utility/tooling | batch | the `if fixture.skipped { continue; }` note @ L31-39 | exact (in-place comment/logic update) |
| `crates/cintx-oracle/tests/spinor_deriv_parity.rs` (CREATE) | test | request-response (collect/stitch/compare) | `one_electron_grad_spinor_parity.rs` (collect/stitch/compare); `int3c1e_genctr_parity.rs` (no-silent-skip + double-gate) | exact (clone + extend) |
| `crates/cintx-oracle/src/fixtures.rs` (MODIFY: add D-08 adversarial fixture) | test/fixture | batch (fixture build) | `int3c1e_genctr_parity.rs::build_genctr_fixture` (L56) — already non-square p×d + nctr=2 | exact (extend to kappa=0 spinor) |
| `crates/cintx-oracle/src/vendor_ffi.rs` (MODIFY: add vendor FFI for rank-9/27/81 + 3c2e/2c2e/3c1e ip-spinor) | utility/FFI | request-response (C ABI) | `vendor_int1e_ipovlp_spinor` (L4193-4216) + spinor sizing helpers (L3713-3777) | exact (mechanical copy) |

## Pattern Assignments

### `crates/cintx-cubecl/src/transform/c2spinor.rs` — NEW `cart_to_spinor_sf_derivative_2d` / `_3c2e` (transform, cart→spinor fold)

**Analog (primary):** inline rank-3 loop in `one_electron.rs` L9919-9965 — this is the byte-identity-proven block the wrapper must reproduce exactly (D-06 moves the transpose INTO the wrapper).

**Analog (inner transform signature, sf_2d, c2spinor.rs L531-559):**
```rust
pub fn cart_to_spinor_sf_2d<F: CintFloat>(
    staging: &mut [F],
    cart: &[f64],
    li: u8, kappa_i: i16,
    lj: u8, kappa_j: i16,
) -> Result<(), cintxRsError> {
    let nci = ncart(li);
    let ncj = ncart(lj);
    let di = spinor_len(li, kappa_i as i32);   // spinor_len @ L25, handles kappa=0 → 4l+2
    let dj = spinor_len(lj, kappa_j as i32);
    if cart.len() < nci * ncj { /* ChunkPlanFailed{ from:"c2spinor_sf_2d", detail } */ }
    let required = di * dj * 2;
    if staging.len() < required { /* BufferTooSmall{ required, provided } */ }
    // ...3 steps: bra transform, ket transform, column-major interleaved write...
}
```

**Inner transform reads BRA-major** (the orientation landmine root, c2spinor.rs Step-3 write @ L603-609):
```rust
// staging[(j*di + i)*2] = re ; [(j*di+i)*2 + 1] = im   (column-major, j-spinor outer, i-spinor inner)
// apply_bra_block reads cart[n*ncj + j] → BRA-major. Device blocks are KET-major.
```

**Core pattern to copy (the inline rank-3 transpose+loop, one_electron.rs L9937-9964) — generalize `3` → `ncomp`:**
```rust
let di = spinor_len(li, shell_i.kappa as i32);
let dj = spinor_len(lj, shell_j.kappa as i32);
let spinor_block = di * dj * 2;                         // D-07 component-outer stride
for comp in 0..3usize {                                  // → 0..ncomp (3|9|27|81)
    let src_base = comp * block_len;                     // block_len = nci*ncj
    let block = &cart_3comp[src_base..src_base + block_len];
    let staging_comp_base = comp * spinor_block;
    // D-06: KET→BRA transpose owned HERE (move out of launcher into wrapper)
    let mut block_bra_major = vec![0.0f64; block_len];
    for ic in 0..nci { for jc in 0..ncj {
        block_bra_major[ic * ncj + jc] = block[jc * nci + ic];
    }}
    cart_to_spinor_sf_2d::<F>(
        &mut staging[staging_comp_base..staging_comp_base + spinor_block],
        &block_bra_major, li, shell_i.kappa, lj, shell_j.kappa)?;
}
```

**`_3c2e` sibling — inner transform already folds the k-axis (c2spinor.rs L1316-1343):** `cart_to_spinor_sf_3c2e` takes `cart[(ck*ncj+j)*nci+i]` (k-outer, i-inner), does cart→sph on k, then `cart_to_spinor_sf_2d` per k-slice with `staging_start = mk * di*dj*2`. The derivative wrapper loops this `ncomp` times with `comp_stride = di*dj*nsk*2`. **OPEN (D-11 spike, Assumption A1):** transpose granularity on the 3c2e path (per-(comp,k) `[ket][bra]` vs per-comp before k-fold) — do NOT finalize signature before the spike.

**Fail-closed size checks (FND-06, copy from sf_2d L544-559):** upfront `cart.len() < ncomp*block_len → ChunkPlanFailed`; `staging.len() < ncomp*spinor_block → BufferTooSmall`. **Never re-add `if dst < staging.len()` scatter guards** — family kernels are monolithic whole-block writers.

**nctr>1 (D-08 BLOCKER, Pitfall 4 / Assumption A3):** `sf_2d` has NO nctr param (`di`/`dj` from `(l,kappa)` only). The wrapper (or a contraction-major loop around it) must compose nctr per spike-005:
```
i_global = ci*di + ic   (ci=contraction 0..nctr_i, ic=angular 0..di) ;  ni_full = nctr_i*di
out[comp*(ni_full*nj_full) + (j_global*ni_full + i_global)]
```
Env coeff is COLUMN-major (`env[ci*nprim+ip]`), Shell row-major — transpose internally. Cross-reference `int3c1e_genctr_parity.rs::build_genctr_fixture` for the gc layout. This is the largest concrete addition and a likely plan-task boundary.

---

### `crates/cintx-cubecl/src/kernels/one_electron.rs` — rewire spinor arms (kernel launcher, request-response)

**Analog:** the existing rank-3 `Representation::Spinor` arm @ L9919-9965 (inside the staging-copy `match`). Replace its inline transpose+loop body with one call to `cart_to_spinor_sf_derivative_2d::<F>(staging, cart_3comp, 3, li, shell_i.kappa, lj, shell_j.kappa)?` and DROP the `n_ctr_i != 1 || n_ctr_j != 1` rejection @ L9925-9930 (nctr now handled in the wrapper, D-08).

**Rank-9/27/81 rejections are EARLY GUARDS, not match arms** — different replacement shape. They sit before the staging-copy match (grep map):
```
L8810-8814  rank-? : if plan.representation == Spinor → UnsupportedApi("spinor int1e_{op_name}")
L9016-9018  L9126-9128  L9210-9212  L9300-9302 (gradient)  L9453-9455 (3rd/4th-order)  L9592-9594 (Hessian)
```
Each guard must be lifted and the corresponding staging-copy match arm (currently `Representation::Spinor => unreachable!("spinor rejected above")` @ L9416, L9564, L9732) must call the derivative wrapper with the family's `ncomp`. **Verify `ncomp` == lock `component_rank` per family (Pitfall 2 / A5).**

**Import already present (L24):** `use crate::transform::c2spinor::{cart_to_spinor_sf_2d, spinor_len};` — add `cart_to_spinor_sf_derivative_2d` to it.

---

### `crates/cintx-cubecl/src/kernels/center_3c2e.rs` — lift ip1/ip2 spinor rejections (kernel launcher, request-response)

**Analog:** scalar-spinor 3c2e arm @ L3276-3284 (the established working call):
```rust
Representation::Spinor => {
    let kappa_i = shell_i_in.kappa;
    let kappa_j = shell_j_in.kappa;
    cart_to_spinor_sf_3c2e::<F>(staging, &cart_out, li_in, kappa_i, lj_in, kappa_j, lk)?;
}
```
The ip1/ip2 derivative arms (rejections @ L2394-2396, L2632-2634, L2868-2870) replace the reject with `cart_to_spinor_sf_derivative_3c2e::<F>(staging, &cart_out, ncomp, li, kappa_i, lj, kappa_j, lk)?` (ncomp=3 for ip1/ip2). The corresponding `unreachable!("spinor ... rejected above")` match arms (L2572, L2810, L3044) become the wrapper call. **Import @ L27:** `use crate::transform::c2spinor::cart_to_spinor_sf_3c2e;` — extend.

**int3c1e_ip1/iprinv (D-02, Claude's discretion / A2):** decide during the spike whether it shares `_3c2e` or needs a thin sibling — plan must allow the sibling fallback.

---

### `crates/cintx-cubecl/src/kernels/center_2c2e.rs` — lift ip1/ip2 spinor rejection (kernel launcher, request-response)

**Analog:** scalar-spinor 2c2e arm @ L1082-1086 (folds through `sf_2d`):
```rust
Representation::Spinor => {
    let kappa_i = shell_i.kappa;
    let kappa_k = shell_k.kappa;
    cart_to_spinor_sf_2d::<F>(staging, &cart_buf, li, kappa_i, lk, kappa_k)?;
}
```
The ip1/ip2 rejection @ L629-631 (and unreachable arms @ L773, L930) become `cart_to_spinor_sf_derivative_2d::<F>(staging, &cart_buf, 3, li, kappa_i, lk, kappa_k)?`. **Import @ L53:** `use crate::transform::c2spinor::cart_to_spinor_sf_2d;` — extend.

---

### `crates/cintx-ops/generated/compiled_manifest.lock.json` — flip `oracle_covered` (config/manifest, batch)

**Analog:** field exists on all 347 entries; in-scope rows currently `false` (e.g. `int1e_ipovlp_spinor` @ L3889, `int3c2e_ip1_spinor` @ L399-area). Flip `false → true` ONLY for in-scope families (D-01/D-02); leave `int2e_ip*_spinor` (D-03) and `int1e_ecp_iprinv_spinor` (D-04) `false`. Entry shape:
```json
{ "id": { "symbol": "int1e_ipovlp_spinor" }, "complex_output": true,
  "oracle_covered": false,  ← flip to true
  "component_rank": "3", "forms": ["spinor"], ... }
```
**Audit auto-syncs** (A4): both audit sides derive from the lock; no separate fixtures list. Run `cargo run -p xtask -- manifest-audit` after. **Verify `component_rank`** for each flipped family matches its rank tier (3/9/27/81) before relying on it for `ncomp` (Pitfall 2).

---

### `xtask/src/oracle_covered_update.rs` — lift R5/D-03 deferral note (utility, batch)

**Analog (the note to update, L31-39):**
```rust
// Skipped fixtures carry no numeric parity obligation (e.g. spinor gradients,
// UnsupportedApi by design per R5/D-03). ... must NOT be stamped oracle_covered=true
// (threat T-21-08-02).
if fixture.skipped { continue; }
```
Update the comment to reflect that sf-derivative spinor families are now covered (no longer skipped); KEEP the `if fixture.skipped { continue; }` guard (still correct for D-03/D-04 deferred families). Do NOT stamp any fixture that remains `skipped`.

---

### `crates/cintx-oracle/tests/spinor_deriv_parity.rs` — CREATE (test, request-response)

**Analog (collect/stitch/compare):** `one_electron_grad_spinor_parity.rs`:
- `collect_cintx_spinor_grad` (L183-214) — per-shell-pair `eval_raw` into `3*ni*nj*2`, `stitch_block` into full matrix. Generalize the hardcoded `3` to per-family `ncomp`.
- `stitch_block` (L219-242) — component-leading, column-major (bra fastest): `out[comp*ni*nj*2 + (jj*ni+ii)*2 + {0:re,1:im}]`.
- `collect_vendor_spinor_grad<F>` (L249-278) — `F: Fn(&mut [f64], &[i32;2], &[i32], i32, &[i32], i32, &[f64]) -> i32`.
- `count_mismatches` (L285+), `ATOL=RTOL` setup (L34-36).
- Vendor test gating (L367-385): `#[cfg(has_vendor_libcint)] #[cfg(feature = "cpu")]`, `assert_any_nonzero` on BOTH cintx and vendor, then `assert_eq!(count_mismatches(&vendor,&cintx,ATOL,RTOL), 0)`.

**Analog (no-silent-skip D-10 + double-gate structure):** `int3c1e_genctr_parity.rs` — `#![cfg(any(feature="cpu", feature="rocm"))]` file gate, then per-test `#[cfg(has_vendor_libcint)]`. The D-10 assertion must fail if the vendor arm did not execute (`running N>0 tests`, not a `skipped` fixture) AND assert flipped families read `oracle_covered=true`.

**D-09 test matrix:** `sf_2d` rank-3 (`INT1E_IPOVLP_SPINOR`), rank-9 (`INT1E_IPOVLPIP_SPINOR`), rank-27/81 (`int1e_ipip*` — Open Q3, pick a rank-81 with scalar coverage); `sf_3c2e` rank-3 (`INT3C2E_IP1_SPINOR`); 2c2e (`INT2C2E_IP1_SPINOR`). Each uses the D-08 fixture.

**Orientation negative control (device-block-layout.md):** add a `to_j_fastest` reindex and assert `mismatches(vendor, cintx_jf) > 0` to decisively pin i-fastest on the non-square block.

---

### `crates/cintx-oracle/src/fixtures.rs` — D-08 adversarial fixture (test/fixture, batch)

**Analog:** `int3c1e_genctr_parity.rs::build_genctr_fixture` (L56-129) — ALREADY non-square (p × d) + nctr=2 on the bra. Reuse verbatim as the base, then:
- Set `KAPPA_OF = 0` on every bas row (spinor shells, both GT+LT, `di = 4l+2`) — copy the `bas[s*BAS_SLOTS + KAPPA_OF] = 0` pattern from `one_electron_grad_spinor_parity.rs` L116-148.
- Keep the COLUMN-major env coeff layout note (L63-67): `env[ci*nprim+ip]`, column 0 then column 1.
- Keep displaced centers + `env[PTR_RINV_ORIG..+3]` for the iprinv path (L80-82).

One fixture covers all three landmines: non-square (orientation), nctr>1 (gc transpose), kappa=0 (both blocks + 4l+2 sizing).

---

### `crates/cintx-oracle/src/vendor_ffi.rs` — add ip-spinor vendor FFI (utility/FFI, request-response)

**Analog (the four rank-3 fns @ L4189-4291):** `vendor_int1e_ipovlp_spinor` (L4193), `_ipkin_` (L4221), `_ipnuc_` (L4249), `_iprinv_` (L4278). Mechanical shape:
```rust
pub fn vendor_int1e_ipovlp_spinor(
    out: &mut [f64], shls: &[i32; 2], atm: &[i32], natm: i32,
    bas: &[i32], nbas: i32, env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipovlp_spinor(
            out.as_mut_ptr(), ptr::null_mut(),
            shls.as_ptr() as *mut i32,
            atm.as_ptr() as *mut i32, natm,
            bas.as_ptr() as *mut i32, nbas,
            env.as_ptr() as *mut f64, ptr::null_mut(), ptr::null_mut(),
        )
    }
}
```
Add extern `ffi::` decls + safe wrappers for: rank-9/27/81 1e ip-spinor (per D-09 family choice), `int3c2e_ip1/ip2_spinor`, `int2c2e_ip1/ip2_spinor`, `int3c1e_ip1/iprinv_spinor`. **Out buffer sizing:** `ncomp * ni_sp * nj_sp * 2` (1e/2c2e) or `ncomp * ni_sp * nj_sp * nsk * 2` (3c2e). Use existing sizing helpers (L3729 `vendor_CINTcgto_spinor`, L3773 `vendor_CINTshells_spinor_offset`).

## Shared Patterns

### Spinor output layout contract (verify EVERY wrapper + test against this)
**Source:** `.claude/skills/spike-findings-cintx/references/spinor-layout.md`; matches `c2spinor.rs` L599-609 write + `stitch_block` L233.
**Apply to:** the wrapper, all D-09 tests, the vendor collector.
```
out[comp * (ni_sp*nj_sp) * 2 + (j*ni_sp + i)*2 + {0:re, 1:im}]
    comp        : slowest (rank 3/9/27/81)            comp_stride = ni_sp*nj_sp*2
    (j*ni_sp+i) : ket-major, i (bra) fastest          ni_sp = 4l+2 at kappa=0 (spinor_len)
    {re,im}     : fastest axis, per-element pair       buffer is ×2 the real length
```

### KET→BRA transpose (the orientation landmine — D-06, own it in ONE place)
**Source:** inline @ `one_electron.rs` L9951-9956; root @ `c2spinor.rs` `apply_bra_block` (reads `cart[n*ncj+j]`, BRA-major).
**Apply to:** every spinor derivative arm — but ONLY inside the wrapper, NEVER in a launcher.
```rust
// device blocks: block[jc*nci + ic] (KET-major) → bra_major[ic*ncj + jc]
for ic in 0..nci { for jc in 0..ncj { block_bra_major[ic*ncj + jc] = block[jc*nci + ic]; }}
```
Verify ONLY on a non-square (p×d) block — square blocks are transpose-symmetric and hide the bug.

### Fail-closed buffer sizing (FND-06 / V5)
**Source:** `c2spinor.rs` L544-559 (`ChunkPlanFailed` for cart, `BufferTooSmall` for staging).
**Apply to:** both new wrappers — upfront size check, then unconditional full-block write. NO `if dst < staging.len()` guards.

### nctr>1 contraction-major composition (spike-005)
**Source:** `.claude/skills/spike-findings-cintx/references/device-block-layout.md` "General contraction"; fixture precedent `int3c1e_genctr_parity.rs` L63-67.
**Apply to:** wrappers (D-08) — `i_global = ci*di + ic`, `ni_full = nctr_i*di`; env coeff COLUMN-major → transpose internally.

### Double-gated vendor parity (no-silent-skip — D-10 / Pitfall 3)
**Source:** `one_electron_grad_spinor_parity.rs` L367-368 gating; `oracle_covered_update.rs` L35 skip-semantics.
**Apply to:** all D-09 tests + the D-10 assertion. Real comparison requires BOTH `--features cpu` AND `CINTX_ORACLE_BUILD_VENDOR=1`. The D-10 test must fail (not skip) if the vendor arm compiled out.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| (none) | — | — | Every file has a strong in-repo analog. The two genuine residual unknowns are NOT missing analogs but empirical layout questions for the D-11 spike: (Q1) 3c2e derivative transpose granularity, and (Q2) exact nctr>1 spinor loop nesting / coeff-transpose location. Both are tracked in RESEARCH.md Open Questions + Assumptions A1/A3 and must be settled against hand-checked vendor values before the `_3c2e` wrapper signature and the nctr composition are finalized. |

## Metadata

**Analog search scope:** `crates/cintx-cubecl/src/transform/`, `crates/cintx-cubecl/src/kernels/`, `crates/cintx-oracle/{src,tests}/`, `crates/cintx-ops/generated/`, `xtask/src/`, `.claude/skills/spike-findings-cintx/references/`
**Files scanned:** c2spinor.rs, one_electron.rs, center_3c2e.rs, center_2c2e.rs, vendor_ffi.rs, one_electron_grad_spinor_parity.rs, int3c1e_genctr_parity.rs, compiled_manifest.lock.json, oracle_covered_update.rs, spinor-layout.md, device-block-layout.md
**Pattern extraction date:** 2026-05-31
