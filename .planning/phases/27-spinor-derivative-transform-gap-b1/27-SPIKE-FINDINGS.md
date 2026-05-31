# 27 SPIKE FINDINGS — D-11 spinor-derivative transform gap (Phase 27 Plan 01 Task 1)

Empirical resolution of the two MEDIUM-confidence residual unknowns BEFORE Plan 02 finalizes
wrapper signatures. Every claim below was probed against the REAL vendor (libcint 6.1.3)
**double-gated under `--features cpu` AND `CINTX_ORACLE_BUILD_VENDOR=1`** on a NON-SQUARE p×d
block with `nctr>1` and `kappa=0`. The throwaway probe was
`crates/cintx-oracle/tests/spike_d11_spinor_deriv_probe.rs`; it has been removed from shipped
source (see "Probe hygiene" below).

Run that produced this evidence:
```
CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
    --test spike_d11_spinor_deriv_probe -- --ignored --nocapture --test-threads=1
```

---

## ⚠ CORRECTION NOTICE (2026-05-31, re-spike reconciliation) — D2/D3 aux-k DISPROVEN

The original D2/D3 claim that the **auxiliary-k axis of arity-3 spinor derivative families is
SPINOR-sized (`CINTcgto_spinor = 4l+2`, total 720 for p×d×s kappa=0)** is **DISPROVEN against
libcint 6.1.3 source**. The probe's `720` was an artifact of a compat-dims **over-sizing bug** in
`fixtures.rs::ao_count_for_rep` (it applied `CINTcgto_spinor` to the aux-k shell too), NOT a real
vendor requirement.

**Ground truth (source-verified):** `CINT3c2e_spinor_drv` (cint3c2e.c:631-636) sizes the output
axes as:
```c
counts[0] = CINTcgto_spinor(shls[0]);          // bra i: SPINOR (4l+2)
counts[1] = CINTcgto_spinor(shls[1]);          // ket j: SPINOR (4l+2)
if (is_ssc) { counts[2] = nfk * x_ctr[2]; }    // ssc: cartesian
else        { counts[2] = (k_l*2+1) * x_ctr[2]; }   // aux k: SPHERICAL (2lk+1) * nctr_k
```
`int3c2e_ip1_spinor` / `int3c2e_ip2_spinor` (autocode/int3c2e.c:94/175) both call this with
`is_ssc=0` → **aux-k = `(2lk+1)*nctr_k` SPHERICAL**, never spinor. For p×d×s kappa=0 nctr_k=1 the
correct total is `3·6·10·1·2 = 360`, not 720. (Also confirmed: cart2sph.c:6014.)

**Critical consequence:** the existing inner transform `cart_to_spinor_sf_3c2e` already uses
`nsk = nsph(lk)` for the k-axis (c2spinor.rs L1293/L1308) and was **ALREADY CORRECT**. The
original D2 instruction to "reconcile `nsph(lk)` up to `CINTcgto_spinor(k)`" was the error and is
**WITHDRAWN**. The `cart_to_spinor_sf_derivative_3c2e` wrapper simply loops the already-correct
inner transform `ncomp` times with `comp_stride = nsph(lk)*di*dj*2`.

The D2/D3 text below is preserved for the disproof trail but is **superseded** wherever it asserts
spinor-sized aux-k. The committed Plan-01 scaffolding that encoded the wrong contract
(`fixtures.rs ao_count_for_rep`, `vendor_ffi.rs` 3c2e/3c1e aux-k buffers, `spinor_deriv_parity.rs`
`SK`/`shell_nsp_full` on the aux-k) is corrected in the re-plan (see STATE.md reconciliation
checklist).

---

## D1. sf_2d device derivative cart block layout — `[comp][ket][bra]` CONFIRMED

**Family probed:** `int1e_ipovlp_spinor` (rank 3 gradient), p × d (NON-SQUARE), `ni_sp=6`,
`nj_sp=10`, kappa=0, nctr=1.

**Vendor evidence (Probe 1):**
```
int1e_ipovlp_spinor  p×d  ni_sp=6 nj_sp=10 (NON-SQUARE)  rank=3  cb=120
eval_raw OK: not0=161 bytes_written=2880
VENDOR: mm(vendor,cintx)=0   mm(vendor,transposed_ij)=248
```

- The device-emitted derivative cart buffer is **`[comp][ket][bra]` component-outer**, with
  per-component stride `block_len = nci*ncj` (here `ncart(p)*ncart(d) = 3*6 = 18`).
- Spinor output is interleaved-complex `rank*ni_sp*nj_sp*2`, `comp_stride = ni_sp*nj_sp*2`,
  ket-major / bra-fastest around the complex interleave (matches the spike-006 contract,
  now extended to a genuinely NON-SQUARE block).
- `mm(vendor,cintx)=0` AND the i/j-transposed misread diverges in 248 elements
  (`mm(vendor,transposed_ij)=248 > 0`) → orientation is **decisively pinned** on the
  non-square block (the square-block H2O test was orientation-blind, as warned).
- **Device cart block is KET-major (`block[jc*nci+ic]`); `cart_to_spinor_sf_2d` reads
  BRA-major (`cart[n*ncj+j]`)** → the host **`nci*ncj` transpose per component is required**
  before the spin-free fold. This matches the existing inline rank-3 transform at
  `one_electron.rs` L9919-9965 (`block_bra_major[ic*ncj+jc] = block[jc*nci+ic]`).

**Plan 02 decision:** the new `cart_to_spinor_sf_derivative_2d` wrapper loops the proven
sf_2d fold `ncomp` times with `comp_stride = di*dj*2` on the staging side and
`block_len = nci*ncj` on the cart side, transposing KET→BRA per component first.

---

## D2. 3c2e derivative transpose granularity + ip1-vs-ip2 + aux-k sizing

**Family probed:** `int3c2e_ip1_spinor` / `int3c2e_ip2_spinor` (rank 3), p × d × s, kappa=0.

**Vendor evidence (Probe 2):**
```
p×d×s  ni_sp=6 nj_sp=10  nsk_sph=1 nk_sp(spinor)=2  rank=3  total=720
int3c2e_ip1_spinor eval_raw -> UnsupportedApi { requested: "spinor int3c2e_ip1 gradient" }
VENDOR ip1 vs ip2: mm(ip1,ip2)=348  (v1_nnz=348 v2_nnz=348)
```
> **CORRECTED:** `total=720` above used the over-sized aux-k (`nk_sp(spinor)=2`). The real vendor
> output uses `nsk_sph=1` → **`total = 3·6·10·1·2 = 360`**. See CORRECTION NOTICE.

### Transpose granularity — per-(comp,k) `[ket][bra]`
The 3c2e derivative buffer is **`[comp][k][ket][bra]`**: component slowest, then the k-axis,
then the per-(comp,k) `[ket][bra]` spinor sub-block (ket-major, bra-fastest, complex
interleaved). The transpose is applied **per-(comp,k) sub-block** of size `ni_sp*nj_sp*2`,
NOT per-comp before the k-fold. This mirrors `cart_to_spinor_sf_3c2e` (c2spinor.rs L1316-1343),
which already does cart→sph on k then `cart_to_spinor_sf_2d` per k-slice
(`staging_start = mk * di*dj*2`); the derivative wrapper loops that whole thing `ncomp` times
with `comp_stride = (k-extent)*di*dj*2`.

### ip2 differs from ip1 only in VALUES, not shape
`int3c2e_ip2_spinor` (∇ on the auxiliary/ket k-center) produces the **same buffer SHAPE**
`[comp][k][ket][bra]` as ip1 (`v1_nnz == v2_nnz == 348`), differing only in which center is
differentiated (`mm(ip1,ip2)=348`). **Same wrapper shape for ip1 and ip2** — one
`cart_to_spinor_sf_derivative_3c2e` serves both; the launcher picks the gradient gout
(nabla1i vs nabla1l) exactly as the cart/sph paths already do.

### Auxiliary-k axis is SPHERICAL (2lk+1)*nctr_k, NOT spinor — CORRECTED
> **SUPERSEDED — see CORRECTION NOTICE at top.** The original claim here ("aux-k is
> SPINOR-sized `CINTcgto_spinor = 4l+2 = 2`, total 720") is **DISPROVEN**. libcint's
> `CINT3c2e_spinor_drv` (cint3c2e.c:631-636) sizes the aux-k as **SPHERICAL `(2lk+1)*nctr_k`**
> (the `is_ssc=0` branch `counts[2] = (k_l*2+1)*x_ctr[2]`), while only bra i and ket j use
> `CINTcgto_spinor` (4l+2). For p×d×s kappa=0 nctr_k=1 the correct total is **360**, not 720;
> the `720`/`BufferTooSmall{required:720}` was caused by the `fixtures.rs ao_count_for_rep`
> over-sizing bug (it applied `CINTcgto_spinor` to the aux-k), not a real requirement.
>
> The inner transform `cart_to_spinor_sf_3c2e` already uses `nsk = nsph(lk)` (c2spinor.rs
> L1293/L1308) and was **ALREADY CORRECT** — no reconciliation up to spinor-k is needed or
> wanted. The derivative wrapper loops it `ncomp` times with `comp_stride = nsph(lk)*di*dj*2`.
> SHIP the wrapper with k-axis `= nsph(lk)`.

### Current support envelope
`INT3C2E_IP1_SPINOR` currently returns `UnsupportedApi { "spinor int3c2e_ip1 gradient" }`
(center_3c2e.rs reject arms). This is the gap Plan 03 fills.

---

## D3. int3c1e_ip1/iprinv — SIBLING decision + launcher FILE PATH

**Decision: int3c1e_ip1/iprinv need a THIN SIBLING fold, NOT the shared `_3c2e` wrapper.**

Rationale (host-side launcher, distinct buffer producer):
- The int3c1e launchers are **host-side** and live in
  **`crates/cintx-cubecl/src/kernels/center_3c1e.rs`**:
  - `launch_center_3c1e_ip1` — spinor reject at **L1006-1010**
    (`UnsupportedApi { "spinor int3c1e_ip1 gradient" }`).
  - `launch_center_3c1e_iprinv` — spinor reject at **L1130-1134**
    (`UnsupportedApi { "spinor int3c1e_iprinv gradient" }`); reads `env[PTR_RINV_ORIG]` (L1156).
  - **Plan 04 owns this file.**
- The int3c1e launcher produces its own **host-side derivative cart `out_buf`** (L1064-1110):
  component-leading, contraction-MAJOR, sized
  `3 * (n_ctr_i*nblk_i) * (n_ctr_j*nblk_j) * (n_ctr_k*nblk_k)` where `nblk_* = nsph` (spheric)
  or `ncart` (cart), scattered per-(ci,cj,ck) by `scatter_3c1e_grad_block` (L922). This buffer
  is **already in the same component-leading `[comp][k][j][i]`-fold family** as the device 3c2e
  cart, so the SAME spinor-fold math applies — but it is produced by a DIFFERENT code path
  (host scatter, not the device kernel + `cart_to_spinor_sf_3c2e` call). A thin sibling that
  takes the int3c1e host `out_buf` and applies the per-(comp,k) `[ket][bra]` fold (reusing the
  inner `cart_to_spinor_sf_2d`/`_3c2e` machinery) keeps the 3c2e wrapper's preconditions
  (device cart-out, `cart_to_spinor_sf_3c2e` k-fold) decoupled from the 3c1e host scatter.
- int3c1e is **arity-3** (i,j + aux k), same aux-k sizing rule as D2 applies (CORRECTED):
  the k-axis of the spinor output is **SPHERICAL `nsph(lk) = (2lk+1)*nctr_k`**, NOT
  `CINTcgto_spinor(k)`. (int3c1e_spinor sizes aux-k spherically exactly as int3c2e_spinor does.)

**Both int3c1e_ip1 and int3c1e_iprinv use the same sibling fold**; iprinv differs only in the
gout (Rys-driven, reads `env[PTR_RINV_ORIG]`) and MUST be tested with a NON-ZERO rinv origin
(the D-08 fixture sets `env[PTR_RINV_ORIG..+3]` non-zero) so the rinv-center path is actually
exercised, not a zero-origin shortcut.

---

## D4. nctr>1 spinor composition — contraction-MAJOR, COLUMN→ROW coeff transpose

**Family probed:** `int1e_ipovlp_spinor`, bra p `nctr=2` × ket d, kappa=0.

**Evidence (Probe 3):**
```
bra p nctr=2 (column-major env coeff [c0_p0,c0_p1,c0_p2,c1_p0,c1_p1,c1_p2])
nctr>1 spinor eval_raw -> UnsupportedApi { requested: "spinor 1e gradient with general contraction (nctr>1)" }
```

- The CURRENT 1e spinor gradient path **REJECTS `nctr>1`** (one_electron.rs ~L9924:
  `"spinor 1e gradient with general contraction (nctr>1)"`). This is part of the gap.
- **Composition Plan 02/03 must wire** (proven for the real-family path in spike-005, and the
  same convention libcint uses for spinor):
  - `i_global = ci*di + ic` (contraction-MAJOR within the axis), `ni_full = nctr_i*di`,
    `di = spinor_len(li, kappa_i)` (= `4l+2` at kappa=0).
  - `out[comp*(ni_full*nj_full)*2 + (j_global*ni_full + i_global)*2 + {0:re,1:im}]`.
  - The libcint **env coefficient block is COLUMN-major** (`env[ci*nprim+ip]`); the cintx
    `Shell` coeff is **ROW-major** (`coeff[ip*nctr+ci]`). Transpose COLUMN→ROW internally
    (the historical nctr-transpose bug class — see project memory
    "raw path transposed nctr>1 coefficients"). The D-08 fixture uses the column-major env
    layout `[c0_p0,c0_p1,c0_p2,c1_p0,c1_p1,c1_p2]` to force this transpose to be observable.

---

## Concrete decisions for Plan 02 (signatures) and Plans 03/04 (launchers)

1. **`cart_to_spinor_sf_derivative_2d`** (1e): loop the proven sf_2d fold `ncomp` times,
   `comp_stride = di*dj*2`, KET→BRA `nci*ncj` transpose per component. nctr>1 via
   `i_global = ci*di+ic` with COLUMN→ROW coeff transpose.
2. **`cart_to_spinor_sf_derivative_3c2e`** (3c2e ip1 AND ip2, same shape): loop the
   `cart_to_spinor_sf_3c2e` k-fold `ncomp` times, `comp_stride = nsph(lk)*di*dj*2`. **The
   k-axis output length is `nsph(lk) = (2lk+1)*nctr_k` (SPHERICAL), NOT `CINTcgto_spinor(k)`**
   (CORRECTED — see CORRECTION NOTICE). The existing `nsk = nsph(lk)` intermediate is ALREADY
   correct; do not change it. Correct total for p×d×s kappa=0 is **360**, not 720.
3. **int3c1e_ip1/iprinv: THIN SIBLING** (not shared `_3c2e`), applied to the host-side
   `out_buf` in **`crates/cintx-cubecl/src/kernels/center_3c1e.rs`** (reject sites L1006-1010
   and L1130-1134; Plan 04 owns this file). Same per-(comp,k) `[ket][bra]` fold; iprinv reads
   `env[PTR_RINV_ORIG]` and needs a non-zero origin in tests.
4. **Aux-k SPHERICAL sizing rule** (CORRECTED) applies to every arity-3 spinor derivative family
   (int3c2e_ip1/ip2, int3c1e_ip1/iprinv): output k-axis = `nsph(lk) = (2lk+1)*nctr_k`, NOT
   `CINTcgto_spinor(k)`. Only bra i and ket j are spinor-sized (4l+2). This is the
   re-plan-blocking correction: the committed Plan-01 scaffolding sized aux-k as spinor and must
   be fixed (`fixtures.rs ao_count_for_rep`, `vendor_ffi.rs`, `spinor_deriv_parity.rs`).

---

## Probe hygiene (acceptance criteria)

- The probe ran with **BOTH** gate flags: `--features cpu` AND `CINTX_ORACLE_BUILD_VENDOR=1`
  (the `has_vendor_libcint` cfg) — vendor comparison is REAL, not the silent-skip
  determinism-only path. `mm(vendor,cintx)=0` for the sf_2d path proves a real vendor link.
- The throwaway probe file `crates/cintx-oracle/tests/spike_d11_spinor_deriv_probe.rs` has
  been **removed from shipped source**; the temporary `probe_int3c2e_ip1_spinor` /
  `probe_int3c2e_ip2_spinor` wrappers added to `vendor_ffi.rs` and the two allowlist entries in
  `build.rs` were **reverted**. `git status` shows no stray probe artifact under `crates/`.
  Plan 02 adds the real, documented, allowlisted vendor wrappers.
