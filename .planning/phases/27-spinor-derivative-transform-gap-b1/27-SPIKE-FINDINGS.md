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

### Auxiliary-k axis is SPINOR-sized (CINTcgto_spinor), NOT nsph — IMPORTANT
libcint's `int3c2e_spinor` family sizes the aux k as a **spinor shell**:
`nk_sp = CINTcgto_spinor(k) = 4l+2 = 2` at kappa=0 (confirmed: cintx compat dims required
`720 = 3*6*10*2*2`; an `nsph(k)=1`-sized 360-buffer triggered `BufferTooSmall{required:720}`;
the existing scalar `vendor_ffi_3c2e_spinor_nonzero` test also sizes k with
`vendor_CINTcgto_spinor`). **The inner transform `cart_to_spinor_sf_3c2e` currently uses
`nsk = nsph(lk)` for the k-axis (c2spinor.rs L1293/L1308)** — this DIVERGES from libcint's
spinor-k output sizing. **Plan 02/03 reconciliation required:** the 3c2e spinor derivative
fold must produce a k-axis of length `CINTcgto_spinor(k)` (apply the cart→spinor fold on k too,
or extend the existing transform), so the output extents match libcint and the compat dims.
Do NOT ship a wrapper whose k-axis is `nsph(lk)` — it will under-size the buffer by the
spinor/sph k-ratio and fail vendor parity.

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
- int3c1e is **arity-3** (i,j + aux k), same aux-k spinor-vs-sph sizing caveat as D2 applies:
  the k-axis of the spinor output must be `CINTcgto_spinor(k)`.

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
   `cart_to_spinor_sf_3c2e` k-fold `ncomp` times, `comp_stride = (k-extent)*di*dj*2`. **The
   k-axis output length MUST be `CINTcgto_spinor(k)` (spinor), NOT `nsph(lk)`** — reconcile the
   existing `nsk = nsph(lk)` intermediate so the final spinor output matches libcint + compat
   dims (720, not 360, for p×d×s kappa=0).
3. **int3c1e_ip1/iprinv: THIN SIBLING** (not shared `_3c2e`), applied to the host-side
   `out_buf` in **`crates/cintx-cubecl/src/kernels/center_3c1e.rs`** (reject sites L1006-1010
   and L1130-1134; Plan 04 owns this file). Same per-(comp,k) `[ket][bra]` fold; iprinv reads
   `env[PTR_RINV_ORIG]` and needs a non-zero origin in tests.
4. **Aux-k spinor sizing caveat** applies to every arity-3 spinor derivative family
   (int3c2e_ip1/ip2, int3c1e_ip1/iprinv): output k-axis = `CINTcgto_spinor(k)`.

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
