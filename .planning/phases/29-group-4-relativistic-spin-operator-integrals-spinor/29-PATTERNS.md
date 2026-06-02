# Phase 29: Group 4 — Relativistic Spin-Operator Integrals (spinor) - Pattern Map

**Mapped:** 2026-05-31
**Files analyzed:** 11 (3 waves)
**Analogs found:** 11 / 11

> Every analog below was verified against the live code this session (line numbers
> re-confirmed). The codebase already contains a near-complete template for every
> Phase-29 deliverable — this phase is overwhelmingly "clone the Phase-28 analog
> and adapt"; the only genuinely new structural code is the 2e si transform suite
> (Wave 2), which clones the existing `cart_to_spinor_sf_4d` skeleton.

## File Classification

| New/Modified File | Wave | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|------|-----------|----------------|---------------|
| `crates/cintx-cubecl/src/transform/c2spinor.rs` — new `cart_to_spinor_si_2di` (1e imaginary-ket) | 1 | transform (host) | transform / fold | `cart_to_spinor_si_2d` (c2spinor.rs:673) bra + `cart_to_spinor_iket_si` (457) ket | exact (si_2d body, iket ket) |
| `crates/cintx-cubecl/src/kernels/one_electron.rs` — new 1e σ Spinor launcher arms (spsp/spnucsp/sprinvsp/srsr/srnucsr/sr/sigma) + flip `int1e_sp` | 1 | kernel launcher | request-response | the `is_sp` Spinor arm (one_electron.rs:10495-10595) + `launch_int1e_sp_spinor_pair` (sigma_p.rs:565) | exact |
| `crates/cintx-oracle/src/vendor_ffi.rs` — `vendor_int1e_{spsp,spnucsp,sprinvsp,srsr,srnucsr,sr,sigma}_spinor` | 1 | FFI shim | request-response | `vendor_int1e_sp_spinor` (vendor_ffi.rs:4139) | exact |
| `crates/cintx-oracle/tests/rel_1e_sigma_parity.rs` (NEW) | 1 | test | request-response | `si_transform_parity.rs` (full file) | exact |
| `crates/cintx-cubecl/src/transform/c2spinor.rs` — NEW 2e suite `cart_to_spinor_si_2e1/2e2(+i)` + `c2s_sf_2e1/2e2` | 2 | transform (host) | transform / fold | `cart_to_spinor_sf_4d` (c2spinor.rs:1199) skeleton + `cart_to_spinor_si_2d` (673) bra step | role+flow match (new layout) |
| `crates/cintx-oracle/src/fixtures.rs` — NEW `build_kappa_spinor_2e_fixture` | 2 | fixture builder | data | `build_adversarial_spinor_fixture` (fixtures.rs:209) + Phase-28 `build_kappa_spinor_fixture` | exact (extend to 4 shells) |
| `crates/cintx-oracle/tests/si_2e_transform_parity.rs` (NEW) | 2 | test (micro-gate) | request-response | `si_transform_parity.rs` | exact |
| `crates/cintx-cubecl/src/kernels/two_electron.rs` — new 2e σ Spinor launcher arms | 3 | kernel launcher | request-response | one_electron.rs `is_sp` arm + `test_device_matches_host_spsp` (center_4c1e.rs:1878) for the 2e σ·p gout | role match (new 2e gout) |
| `crates/cintx-oracle/build.rs` — add `gaunt1.c` + `dkb.c` `.file()` + allowlist | 3 | build config | build | existing `.file()` block (build.rs:217-265) + `allowlist_function` regex (build.rs:374) | exact |
| `crates/cintx-oracle/src/vendor_ffi.rs` — `vendor_int2e_{spsp1,srsr1,spsp1spsp2,srsr1srsr2,ssp1ssp2,sps1sps2,vsp1,spv1,…}_spinor` | 3 | FFI shim | request-response | `vendor_int1e_sp_spinor` (4139) + `vendor_int2e_spinor` (signature) | exact |
| `crates/cintx-oracle/tests/rel_2e_sigma_parity.rs` (NEW) | 3 | test | request-response | `si_transform_parity.rs` | exact |
| `crates/cintx-ops/{generated/compiled_manifest.lock.json, src/generated/api_manifest.rs}` — ~22 new rows | 1+3 | manifest | data | `int1e_sp_spinor` row (lock:10710 / api_manifest.rs:6256) | exact |

---

## Pattern Assignments

### Wave 1

#### `cart_to_spinor_si_2di` — new 1e imaginary-ket transform (c2spinor.rs)
**Role:** transform (host) · **Flow:** Pauli-σ bra fold + imaginary ket
**Analog:** `cart_to_spinor_si_2d` (c2spinor.rs:673) for the bra step + structure; `cart_to_spinor_iket_si` (c2spinor.rs:457) for the imaginary-ket coefficient handling.

`c2s_si_1ei` differs from `c2s_si_1e` ONLY in the ket step: it uses `a_iket_cart2spinor` (multiply-by-i) instead of `a_ket_cart2spinor`. So clone `cart_to_spinor_si_2d` verbatim and swap Stage-2 `apply_ket_transform` for the iket variant. Used by `int1e_sr` and `int1e_sigma` (RESEARCH §Per-Family Map).

**Clone these stages from si_2d (673):**
- Buffer guards (lines 689-717): `need = nci*ncj` per gc block → `ChunkPlanFailed`; `required = di*dj*2` → `BufferTooSmall`. **No writes before guards pass.**
- KET→BRA transpose (lines 719-735): the transform OWNS it — `dst[i*ncj+j] = src[j*nci+i]` for each of the four gc blocks. Device cart blocks are KET-major.
- Stage 1 bra σ-mix `apply_bra_si_block` (lines 737-757) — unchanged.
- **Stage 2 — the ONLY change:** swap `apply_ket_transform` → the imaginary-ket ket transform (the same coefficient path `cart_to_spinor_iket_si` uses internally at c2spinor.rs:489-514).
- Stage 3 column-major interleaved zcopy (lines 777-784): `staging[(j*di+i)*2] = re, +1 = im`.

**Sizing:** all `di`/`dj` from `spinor_len(l, kappa)` (c2spinor.rs:25) — kappa<0→2l+2 (GT), kappa>0→2l (LT), kappa==0→4l+2. NEVER hardcode `4l+2`.

#### 1e σ Spinor launcher arms (one_electron.rs)
**Role:** kernel launcher · **Flow:** request-response
**Analog:** the live `int1e_sp` Spinor arm at **one_electron.rs:10495-10595** (and the standalone `launch_int1e_sp_spinor_pair` at sigma_p.rs:565). Each new family adds an arm with the identical shape.

**Copy the arm body (10504-10595):**
```rust
// 1. extract n_prim_i/j, n_ctr_i/j, exps_i/j, coeff_i/j (10504-10515)
// 2. device σ·p assembler — run_sigma_p_on_backend(backend, tensor_rank, ...) (10518-10533)
// 3. s/p normalization scale: common_fac_sp(li)*common_fac_sp(lj) (10535-10541)
// 4. di = spinor_len(li, kappa_i); ni_sp = n_ctr_i*di; ...   (10545-10548)
// 5. FAIL-CLOSED STAGING GUARD (10550-10560) — required = ni_sp*nj_sp*2:
let staging_required = ni_sp * nj_sp * 2;
if staging.len() < staging_required {
    return Err(cintxRsError::BufferTooSmall { required: staging_required, provided: staging.len() });
}
// 6. per-(ci,cj) fold + scatter via cart_to_spinor_si_2d (10566-end) — NO launcher transpose
```

**Per-family adaptations:**
- `tensor_rank` arg to `run_sigma_p_on_backend`: `int1e_sp`/`sr`/`sigma` are σ-on-bra (rank 1, 4 gc blocks); `spsp`/`spnucsp`/`sprinvsp`/`srsr`/`srnucsr` are σ-on-both / σ·p² — confirm the gout component count + cart-block extent against RESEARCH §Per-Family Map and intor3.c gout (Open Q2: the 2e/both-side assembler may need a sibling — verify vs `test_device_matches_host_spsp`).
- Transform selection (RESEARCH map): `spsp`→**`cart_to_spinor_sf_2d`** (c2s_sf_1e — scalar, NOT si!); `spnucsp`/`sprinvsp`/`srsr`/`srnucsr`/`sp`→`cart_to_spinor_si_2d`; `sr`/`sigma`→the new `cart_to_spinor_si_2di`.
- **Flip `int1e_sp`:** the existing arm rejects non-Spinor (10496-10502); keep that, just flip `oracle_covered` in the manifest.
- **Each new inline arm needs its own fail-closed staging guard** (Phase-28 CR-01) — copy lines 10550-10560 verbatim into every arm.

#### `vendor_int1e_*_spinor` shims (vendor_ffi.rs)
**Analog:** `vendor_int1e_sp_spinor` (vendor_ffi.rs:4139-4162). Clone verbatim, change the `ffi::int1e_X_spinor` symbol:
```rust
pub fn vendor_int1e_spsp_spinor(out, shls: &[i32;2], atm, natm, bas, nbas, env) -> i32 {
    unsafe { ffi::int1e_spsp_spinor(out.as_mut_ptr(), ptr::null_mut(),
        shls.as_ptr() as *mut i32, atm.as_ptr() as *mut i32, natm,
        bas.as_ptr() as *mut i32, nbas, env.as_ptr() as *mut f64,
        ptr::null_mut(), ptr::null_mut()) }
}
```
`out` sized `ni_sp*nj_sp*2` via `vendor_CINTcgto_spinor` (component_rank=1 for ALL Group-4). **The FFI symbols must be added to the `allowlist_function(...)` regex at build.rs:374** (a `|`-joined list — `int1e_sp_spinor` is already there; append the new symbol names) or bindgen won't emit them.

#### `rel_1e_sigma_parity.rs` (NEW test)
**Analog:** `si_transform_parity.rs` (entire file, 379 lines). Clone its structure:
- `collect_cintx_*` + `collect_vendor_*` (lines 140-172) per family.
- `count_mismatches` (178-201) + `assert_any_nonzero` (204-210).
- `test_kappa_sizing_non_4l_plus_2` (219-237) — assert `spinor_len` GT/LT values.
- **PRIMARY GATE** `#[cfg(has_vendor_libcint)] #[cfg(feature = "cpu")]` byte-identity at `ATOL=1e-12` on `build_kappa_spinor_fixture` (271-293).
- **NO-SILENT-SKIP** `test_no_silent_skip` (327-363) — vendor arm MUST run + produce nonzero; assert manifest `oracle_covered` matches the expected post-flip state.
- Non-vendor smoke `#[cfg(all(feature = "cpu", not(has_vendor_libcint)))]` (369-378).

Wave 1 reuses Phase-28's `build_kappa_spinor_fixture` (p kappa=+1 LT nctr=2 × d kappa=−1 GT).

---

### Wave 2

#### 2e si/sf transform suite (c2spinor.rs) — the only genuinely new structural code
**Role:** transform (host) · **Flow:** two-stage electron-1-then-electron-2 fold + reorder
**Analog:** `cart_to_spinor_sf_4d` (c2spinor.rs:1199-1337) — the existing 2e two-stage skeleton — with the electron-1 bra step swapped from sf to the si σ-mix.

**Clone the sf_4d skeleton structure:**
- Sizing + guards (1213-1244): `di/dj/dk/dl` from `spinor_len`; `expected_cart = nci*ncj*nck*ncl` → `ChunkPlanFailed`; `required = di*dj*dk*dl*2` → `BufferTooSmall`.
- **Step 1** (1246-1262): loop `(l_cart, k_cart)`, slice each `nci*ncj` kl-block, call the 2D electron-1 transform per slice into `opij[nck*ncl*di*dj*2]`. For si this calls the **new `cart_to_spinor_si_2e1`** (clone `cart_to_spinor_si_2d` body — bra σ-mix + ordinary ket) instead of `cart_to_spinor_sf_2d`.
- **Step 2** (1264-1334): for each `(j_sp, i_sp)` extract the complex `[nck*ncl]` slice (1297-1304), apply the electron-2 transform `apply_2d_spinor_zf` (1309-1322), then the `zcopy_iklj` store `staging[(((l_sp*dk+k_sp)*dj+j_sp)*di+i_sp)*2]` (1326-1332).
- Index conventions: cart `cart[((l*nck+k)*ncj+j)*nci+i]` (i innermost, l outermost); opij `opij[((l*nck+k)*dj*di + j_sp*di + i_sp)*2]`.

**The genuinely novel piece (D-03 micro-test must pin this):** the electron-2 si bra step `a_bra1_cart2spinor_zi` (libcint cart2sph.c:4118-4186) applies the 2×2 Pauli σ-matrix to four COMPLEX gx/gy/gz/g1 blocks — `apply_2d_spinor_zf` (c2spinor.rs:1353) is the **sf** (scalar-only) analog; the **si** electron-2 step needs the full σ expansion:
```
v11 = 1+iz, v12 = y+ix, v21 = -y+ix, v22 = 1-iz   // σ·n 2×2
v11R = g1R - gzI;  v11I = g1I + gzR;  (etc. — RESEARCH §Pattern 3)
```
Build `apply_2d_spinor_zi` as the si sibling of `apply_2d_spinor_zf`. The `i`-variants (`2e1i`/`2e2i`) multiply by i (mirror `cart_to_spinor_iket_si` vs `cart_to_spinor_si`).

**Suite to build (per RESEARCH §D-01):** `cart_to_spinor_si_2e1`, `_si_2e2`, `_si_2e1i`, `_si_2e2i`, `_sf_2e1`, `_sf_2e2`. The `sf_2e*` partners are extractions of the existing sf_4d's two stages into reusable per-electron fns.

#### `build_kappa_spinor_2e_fixture` (fixtures.rs)
**Analog:** `build_adversarial_spinor_fixture` (fixtures.rs:209-300) — extend its env/atm/bas construction to 4 shells.
**D-02 hard constraints:** 4 spinor shells (2-electron config), non-square, genuine kappa≠0 GT/LT mix (`p kappa=+1` LT, `d kappa=−1` GT — copy the Phase-28 momenta), ≥1 shell `nctr>1` (copy the `p_coeff = [c0_p0,c0_p1,c0_p2, c1_p0,c1_p1,c1_p2]` COLUMN-major env layout from line 222, `bas[NCTR_OF]=2` from line 278).

#### `si_2e_transform_parity.rs` (NEW — Wave-2 gating micro-test, FIRST task of the wave)
**Analog:** `si_transform_parity.rs`. Drive the thinnest 2e si family `int2e_spsp1_spinor` (`c2s_si_2e1`+`c2s_sf_2e2`, intor4.c:85) against `vendor_int2e_spsp1_spinor`. Same gate/no-silent-skip structure. **MUST be GREEN before any Wave-3 family wires onto the transform** (D-03 mitigation).

---

### Wave 3

#### 2e σ Spinor launcher arms (two_electron.rs)
**Role:** kernel launcher · **Flow:** request-response
**Analogs:** the one_electron.rs `is_sp` arm (10495-10595) for the launch→device-assembler→host-transform→scatter shape; `center_4c1e.rs::test_device_matches_host_spsp` (1878) for the **2e σ·p-on-both-sides cart gout** (mine the host G-tensor builder). All 2e Spinor currently returns `UnsupportedApi` (two_electron.rs ~1450/1665/1717) — replace per family.

**Per-arm requirements (copy from the 1e arm):**
- Fail-closed staging guard `required = di*dj*dk*dl*2` BEFORE any write (Phase-28 CR-01).
- Transform pairing per RESEARCH §2e map: `spsp1`/`srsr1`/`vsp1`/`spv1`→`si_2e1`+`sf_2e2`; `spsp1spsp2`/`srsr1srsr2`/`vsp1spv2`/…→`si_2e1`+`si_2e2`; `ssp1ssp2`/`sps1sps2`/`ssp1sps2`/`sps1ssp2`→`si_2e1i`+`si_2e2i` (both imaginary).
- **Open Q2:** the 2e σ·p assembler is expected NEW code (not direct `sigma_p.rs` reuse) — plan it as a thin 2e sibling mining the spsp harness.

#### `build.rs` — add gaunt1.c + dkb.c (BLOCKING for REL-04)
**Analog:** the `.file()` block (build.rs:217-265) + the `allowlist_function` regex (build.rs:374).
```rust
// In the .file() chain (after intor4.c at line 229):
.file(libcint_root.join("src/autocode/gaunt1.c"))   // int2e_ssp1ssp2/sps1sps2/...
.file(libcint_root.join("src/autocode/dkb.c"))       // int2e_vsp1/spv1/...
```
Both use the same include set as intor4.c (no suppl-header — all symbols in cint_funcs.h). REL-03 (`intor4.c`) is ALREADY wired (line 229) — no build change for it. **Append all new `int2e_*_spinor` symbol names to the `allowlist_function` regex at line 374.**

#### `vendor_int2e_*_spinor` shims (vendor_ffi.rs)
**Analog:** `vendor_int1e_sp_spinor` (4139). 2e drivers take `&[i32;4]` shls. Clone the body, swap `ffi::int2e_X_spinor`, `out` sized `ni*nj*nk*nl*2`.

#### `rel_2e_sigma_parity.rs` (NEW test)
**Analog:** `si_transform_parity.rs` — same gate structure, on `build_kappa_spinor_2e_fixture`, per REL-03/04 family.

#### Manifest rows (compiled_manifest.lock.json + api_manifest.rs)
**Analog:** the `int1e_sp_spinor` row — lock at compiled_manifest.lock.json:10710-10742, generated Rust at api_manifest.rs:6256-6273.
```jsonc
{ "arity": 2, "canonical_family": "1e"/"2e", "category": "1e"/"2e",
  "complex_output": true, "component_rank": "1",   // ALL Group-4 = "1"
  "forms": ["spinor"], "helper_kind": "operator",
  "id": { "family":"1e","operator":"<op>","representation":"spinor","symbol":"int1e_<op>_spinor" },
  "oracle_covered": false, "stability": "stable" }
```
The lock is the source of truth (edits auto-sync both audit sides). `RepresentationSupport::new(false, false, true)` (sph=false, cart=false, spinor=true) — **spinor-only**, do NOT over-claim cart/sph (SC#5). Flip `oracle_covered=true` per family AFTER its parity test is green.

---

## Shared Patterns

### Spinor sizing — `spinor_len`
**Source:** c2spinor.rs:25. **Apply to:** EVERY new transform + launcher arm + fixture sizing.
```rust
pub fn spinor_len(l: u8, kappa: i32) -> usize {
    if kappa < 0 { 2*l as usize + 2 }      // GT j=l+1/2
    else if kappa > 0 { 2*l as usize }     // LT j=l-1/2
    else { 4*l as usize + 2 }              // both
}
```
NEVER hardcode `4l+2`. The kappa fixtures (D-02) ride the GT/LT-only path precisely to surface a hardcoded mistake.

### KET→BRA transpose owned inside the transform
**Source:** c2spinor.rs:719-735 (in si_2d). **Apply to:** every si/sf transform (1e + 2e).
Device cart blocks arrive KET-major `block[j*nci+i]`; the bra step reads BRA-major `block[i*ncj+j]`. Transpose each gc block inside the transform — NO launcher transpose. Latent on square blocks; the non-square fixtures surface omission.

### Fail-closed staging guard per inline Spinor arm
**Source:** one_electron.rs:10550-10560. **Apply to:** every new Spinor launcher arm (1e + 2e). `required = di*dj*2` (1e) / `di*dj*dk*dl*2` (2e). Return `BufferTooSmall` BEFORE any write (OOM-safe stop, no partial writes). Phase-28 CR-01 — inline arms bypass any `launch_*_pair` guard.

### NO-SILENT-SKIP parity assertion
**Source:** si_transform_parity.rs:327-363. **Apply to:** every new parity test. Under `has_vendor_libcint`, the vendor arm MUST execute + produce nonzero output (fail, not skip). Double-gate: `--features cpu` AND `CINTX_ORACLE_BUILD_VENDOR=1`.

### component_rank = "1"
**Source:** api_manifest.rs:6263 / lock:10721. **Apply to:** EVERY Group-4 row. The σ_x/σ_y/σ_z fold is internal to the c2s transform — it is NOT an output component axis. A wrong rank>1 mis-strides the interleaved output (component_rank-truncation landmine). Verify the existing `int1e_sp` row stays "1". (A2/Open Q1: empirically confirm `int1e_sigma` output length == `di*dj*2` in Wave-1 first task.)

### Interleaved-complex output layout
**Source:** c2spinor.rs:777-784 (si_2d Stage 3) + skill `references/spinor-layout.md`.
1e: `staging[(j*di+i)*2 + {0:re,1:im}]` (column-major, ket outer, bra inner).
2e: `staging[(((l_sp*dk+k_sp)*dj+j_sp)*di+i_sp)*2]` (i innermost, l outermost — `zcopy_iklj`).
Oracle compares the flat buffer directly.

### OperatorId by symbol name (positional shift landmine)
Adding ~22 rows re-points hardcoded `OperatorId::new(N)` / `_OPERATOR_ID: u32 = N` consts at a different family (api_manifest.rs:6275+ derives positional ids). After adding rows, re-grep `OperatorId::new(` and `_OPERATOR_ID` and resolve by symbol name. (project memory: `operator_id_shift_breaks_hardcoded_test_consts`)

---

## No Analog Found

None. Every Phase-29 deliverable has a concrete in-repo template. The 2e si transform suite (Wave 2) has a **structural** analog (`cart_to_spinor_sf_4d`) but its electron-2 si bra step (`apply_2d_spinor_zi` — the 2×2 Pauli expansion on complex blocks) is new math; transcribe it from libcint `cart2sph.c:4118-4186` and gate with the D-03 micro-test.

| Concern | Status |
|---------|--------|
| 2e electron-2 σ-mix (`a_bra1_cart2spinor_zi`) | New code; transcribe from cart2sph.c:4118; sf sibling exists (`apply_2d_spinor_zf` c2spinor.rs:1353). |
| 2e σ·p-on-both-sides device gout | Expected new (thin 2e sibling); mine `center_4c1e.rs::test_device_matches_host_spsp` (L1878). Open Q2. |

## Metadata

**Analog search scope:** `crates/cintx-cubecl/src/transform/`, `crates/cintx-cubecl/src/kernels/`, `crates/cintx-oracle/{src,tests}/`, `crates/cintx-ops/{src/generated,generated}/`, `libcint-master/src/`.
**Files scanned:** c2spinor.rs, sigma_p.rs, one_electron.rs, two_electron.rs, center_4c1e.rs, vendor_ffi.rs, fixtures.rs, si_transform_parity.rs, build.rs, compiled_manifest.lock.json, api_manifest.rs.
**Pattern extraction date:** 2026-05-31
