# Phase 28: Spin-Included `c2s_si` Transform + σ·p Module (Gap B2) - Pattern Map

**Mapped:** 2026-05-31
**Files analyzed:** 8 (5 new code locations, 3 modified)
**Analogs found:** 8 / 8 (all exact or role-match; codebase is ~80% reuse per RESEARCH §"Don't Hand-Roll")

> **THE LANDMINE (read first):** The new si bra step MUST NOT reuse `apply_si_block` (`c2spinor.rs:124-172`). That function transcribes `CINTc2s_ket_spinor_si1`'s sign convention, which differs from `a_bra_cart2spinor_si` (the function the 2D `c2s_si_1e` path actually uses) by sign on 3 of 4 cross/imaginary terms. See **Shared Pattern: SI Sign Convention** below. RESEARCH.md §Spike Target A and §Code Examples already verified this term-by-term against vendored C.

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/cintx-cubecl/src/transform/c2spinor.rs` :: NEW `cart_to_spinor_si_2d` | transform (host) | transform (post-kernel host marshaling) | `cart_to_spinor_sf_2d` (`c2spinor.rs:531`) | exact (structural template) |
| `crates/cintx-cubecl/src/transform/c2spinor.rs` :: NEW `apply_bra_si_block` | transform-helper (host) | transform | `apply_bra_block` (`c2spinor.rs:668`) structure + `apply_si_block` (`:124`) 4-cart shape — but NEW sign convention | role-match + TRAP |
| `crates/cintx-cubecl/src/kernels/sigma_p.rs` (NEW) :: σ·p `#[cube]` assembler | kernel (device `#[cube]`) | per-primitive device compute | `contract_grad_1e_bra` / `one_electron_grad_bra_kernel` nabla machinery (`one_electron.rs:7735`, `:7705`); `write_component_leading_staging` (`:8300`) | role-match (gout/nabla) |
| `crates/cintx-cubecl/src/kernels/one_electron.rs` :: MODIFIED Spinor dispatch arm | kernel-launcher (host glue) | dispatch | sf_2d dispatch arm (`one_electron.rs:10101-10131`) | exact |
| `crates/cintx-oracle/src/fixtures.rs` :: NEW `build_kappa_spinor_fixture` + heavy-atom | fixture (test glue) | data-build | `build_adversarial_spinor_fixture` (`fixtures.rs:209`) | exact (clone + set kappa≠0) |
| `crates/cintx-oracle/src/vendor_ffi.rs` :: NEW `vendor_int1e_sp_spinor` | FFI shim (test glue) | request-response (FFI) | `vendor_int1e_ovlp_spinor` (`vendor_ffi.rs:4101`) | exact |
| `crates/cintx-ops/generated/compiled_manifest.lock.json` + `api_manifest.rs` :: `int1e_sp_spinor` row | config/manifest | declarative | existing `*_spinor` ManifestEntry rows | role-match |
| `xtask/src/oracle_covered_update.rs` :: SC#4 skipped-fixture guard | config-guard (host glue) | guard | EXISTING guard at `:50` (`if fixture.skipped { continue; }`) | exact (already present — extend/assert) |
| `crates/cintx-oracle/tests/si_transform_parity.rs` (NEW) | test | integration (vendor) | `spinor_deriv_parity.rs` (D-10 no-silent-skip pattern, Phase 27) | role-match |

---

## Pattern Assignments

### `cart_to_spinor_si_2d` — `crates/cintx-cubecl/src/transform/c2spinor.rs` (transform, host)

**Analog:** `cart_to_spinor_sf_2d` (`c2spinor.rs:531-612`) — structural template, copy the 3-stage skeleton verbatim, swap ONLY the bra step.

**Signature + buffer-size guard pattern** (`c2spinor.rs:531-559`) — copy verbatim, but the si version takes FOUR cart blocks (`gc_x/gc_y/gc_z/gc_1`) instead of one `cart`:
```rust
pub fn cart_to_spinor_sf_2d<F: CintFloat>(
    staging: &mut [F], cart: &[f64],
    li: u8, kappa_i: i16, lj: u8, kappa_j: i16,
) -> Result<(), cintxRsError> {
    let nci = ncart(li);  let ncj = ncart(lj);
    let di = spinor_len(li, kappa_i as i32);    // <- ALWAYS spinor_len, NEVER 4l+2
    let dj = spinor_len(lj, kappa_j as i32);
    if cart.len() < nci * ncj { return Err(cintxRsError::ChunkPlanFailed { .. }); }
    let required = di * dj * 2;
    if staging.len() < required {
        return Err(cintxRsError::BufferTooSmall { required, provided: staging.len() });
    }
    // ... bra step → ket step → zcopy interleave
}
```

**3-stage skeleton** (`c2spinor.rs:561-611`) — si_2d keeps stages 2 (ket) and 3 (zcopy) IDENTICAL, replaces stage 1 (bra) with the σ-coupled `apply_bra_si_block`:
- **Stage 1 (bra)** — REPLACE `apply_bra_sf_block_all_kappa` with the new si bra step (4 cart inputs, `a_bra_cart2spinor_si` signs).
- **Stage 2 (ket)** — REUSE `apply_ket_transform` (`c2spinor.rs:592-597`) VERBATIM. `c2s_si_1e` uses the ORDINARY ket (no `_si` suffix, no symmetrization). [VERIFIED: cart2sph.c:4984 == c2s_sf_1e's :4900]
- **Stage 3 (zcopy)** — REUSE the column-major interleave (`c2spinor.rs:603-608`) VERBATIM:
```rust
for j in 0..dj {
    for i in 0..di {
        let out_idx = j * di + i;
        staging[out_idx * 2]     = F::from_f64_lossy(out_r[j * di + i]);
        staging[out_idx * 2 + 1] = F::from_f64_lossy(out_i[j * di + i]);
    }
}
```

**KET→BRA transpose — OWN IT INSIDE the transform** (mirror `c2spinor.rs:1342-1354`, the 3c2e fix). Device cart blocks are KET-major; the bra step reads BRA-major. Latent on square blocks → the D-05 fixture is non-square (p×d) precisely to surface this:
```rust
// ket-major sph_k[j*nci + i]  →  bra-major bra_major[i*ncj + j]
for j in 0..ncj {
    for i in 0..nci {
        bra_major[i * ncj + j] = cart_slice[j * nci + i];
    }
}
```
Note: the existing sf_2d Spinor dispatch arm (`one_electron.rs:10124-10128`) does this transpose in the LAUNCHER for the single-block scalar case. For si_2d, RESEARCH §Pitfall 4 / Phase-27 D-06 says own it INSIDE the transform. The planner should follow the `c2spinor.rs:1342-1354` (in-transform) pattern, applied per the four gc blocks.

---

### `apply_bra_si_block` — `crates/cintx-cubecl/src/transform/c2spinor.rs` (transform-helper, host) — THE NEW CODE

**Structural analog:** `apply_bra_block` (`c2spinor.rs:668-710`) for the loop/index/coeff-layout structure.
**Shape analog:** `apply_si_block` (`c2spinor.rs:124-172`) for the 4-cart-input signature (`cart_v1/vx/vy/vz`).
**Sign convention:** NEITHER — transcribe `a_bra_cart2spinor_si` (`cart2sph.c:3958-3961`) VERBATIM.

**Loop + coeff-index structure to copy from `apply_bra_block`** (`c2spinor.rs:683-697`):
```rust
for j in 0..ncj {                 // ket cart column
    for i in 0..nd {              // bra spinor row
        let mut sa_r = 0.0f64; let mut sa_i = 0.0f64;
        let mut sb_r = 0.0f64; let mut sb_i = 0.0f64;
        for n in 0..nci {         // bra cart index
            let v1 = cart[n * ncj + j];   // BRA-major read; here 4 blocks v1/vx/vy/vz
            let ca_r = coeff_r[i * 2 * nci + n];
            let ca_i = coeff_i[i * 2 * nci + n];
            let cb_r = coeff_r[i * 2 * nci + nci + n];
            let cb_i = coeff_i[i * 2 * nci + nci + n];
            // *** SIGNS GO HERE — see Shared Pattern: SI Sign Convention ***
        }
        // write alpha (upper) then beta (lower) spinor blocks
    }
}
```

**The accumulation (transcribe EXACTLY — `cart2sph.c:3958-3961`, RESEARCH Pattern 1 / §Spike Target A):**
```rust
sa_r +=  ca_r * v1 + ca_i * vz - cb_r * vy + cb_i * vx;
sa_i += -ca_i * v1 + ca_r * vz + cb_i * vy + cb_r * vx;
sb_r +=  cb_r * v1 - cb_i * vz + ca_r * vy + ca_i * vx;
sb_i += -cb_i * v1 - cb_r * vz - ca_i * vy + ca_r * vx;
```

> **CONTRAST with `apply_si_block` (`c2spinor.rs:161-164`) — DO NOT COPY THESE:**
> ```rust
> // WRONG for the 2D path (this is CINTc2s_ket_spinor_si1's convention):
> sa_re += ca_r * v1 - ca_i * vz + cb_r * vy - cb_i * vx;   // 3 sign flips vs bra-si
> sa_im += ca_i * v1 + ca_r * vz + cb_i * vy + cb_r * vx;
> sb_re += cb_r * v1 + cb_i * vz - ca_r * vy - ca_i * vx;
> sb_im += cb_i * v1 - cb_r * vz - ca_i * vy + ca_r * vx;
> ```
> Grep gate: `apply_bra_si_block` MUST contain `+ ca_i * vz`, `- cb_r * vy`, `+ cb_i * vx`; it MUST NOT contain `- ca_i * vz + cb_r * vy - cb_i * vx`.

**kappa dispatch:** Follow `apply_bra_sf_block_all_kappa` (`c2spinor.rs:624-657`) for the GT/LT/both block-pointer dispatch (kappa<0→GT, kappa>0→LT, kappa==0→LT-rows-then-GT-rows over-read), and `bra_coeff_refs` (`c2spinor.rs:714`) for the CG table slices.

---

### σ·p `#[cube]` assembler — `crates/cintx-cubecl/src/kernels/sigma_p.rs` (NEW) (kernel, device)

**Analog:** the bra-nabla + component-mixing machinery in `one_electron.rs` — host ref `contract_grad_1e_bra` (`:7735`), device `one_electron_grad_bra_kernel`, and the component-leading staging writer `write_component_leading_staging` (`:8300`).

**The g1 = nabla_i(g0) component-mix (`one_electron.rs:7722-7724`, host ref) — the σ·p assembler reuses THIS exact s[0..2] mixing:**
```rust
//   s[0] = g1x[jx,ix] * g0y[jy,iy] * g0z[jz,iz]   (∂/∂Ax)  → gc_x
//   s[1] = g0x[jx,ix] * g1y[jy,iy] * g0z[jz,iz]   (∂/∂Ay)  → gc_y
//   s[2] = g0x[jx,ix] * g0y[jy,iy] * g1z[jz,iz]   (∂/∂Az)  → gc_z
```
nabla formula per axis (`one_electron.rs:7753+`, faithful `CINTnabla1i_1e`):
```rust
let ai2 = -2.0 * ai;
g1[off + jbase] = ai2 * g[off + jbase + 1];               // ix == 0
g1[off + jbase + ix] = ix * g[..ix-1] + ai2 * g[..ix+1];  // ix >= 1
```

**The σ·p difference vs the gradient kernel:** `int1e_sp` packs the 3 nabla components into a 4-component `gout` with a ZERO scalar slot (`intor3.c:431-434`, RESEARCH §Spike Target E):
```rust
gout[n*4 + 0] = s[0];   // → gc_x
gout[n*4 + 1] = s[1];   // → gc_y
gout[n*4 + 2] = s[2];   // → gc_z
gout[n*4 + 3] = 0.0;    // → gc_1 (scalar slot, ZERO for int1e_sp)
```

**Output layout (Shared Pattern: gout→gc layout below):** emit component-leading/pre-blocked `gc[comp*(nf*ictr*jctr) + n]` following `write_component_leading_staging` (`one_electron.rs:8300`, signature takes `rank`, `block_len`, `cart_comp`, `staging`) — NOT interleaved `n*4+comp`. RESEARCH Open Q #1 recommends pre-blocked on-device (avoids a host transpose; matches existing component-leading convention).

**D-03 reusability:** parameterize by `(tensor_rank, which-slot-nonzero)` so `int1e_sigma` (rank 3, 12-component) and Phase-29 families reuse it. `int1e_sp` is `tensor_rank=1`, Pauli in x/y/z, scalar=0.

**CubeCL `#[cube]` rules (RESEARCH §Project Constraints + memory `reference_cubecl_authoring_manuals.md`):** no plain-fn calls, no if-expr, `F::exp`/`F::sqrt`, u32/i32 only, no continue/break. Internal f64, F-output (per `project_first_gpu_family_2c2e.md` template). Watch the CpuRuntime FP-env side effect (RESEARCH §Pitfall 5) on the atol=1e-12 gate.

---

### Spinor dispatch arm — `crates/cintx-cubecl/src/kernels/one_electron.rs` (MODIFIED) (launcher, host)

**Analog:** the existing `Representation::Spinor =>` arm at `one_electron.rs:10101-10131` (the sf_2d dispatch). The new `int1e_sp` Spinor arm mirrors this: call the σ·p assembler to produce the 4 gc blocks, then dispatch `cart_to_spinor_si_2d` instead of `cart_to_spinor_sf_2d`.

**Import site:** `one_electron.rs:24`:
```rust
use crate::transform::c2spinor::{cart_to_spinor_sf_2d, cart_to_spinor_sf_derivative_2d};
// ADD: cart_to_spinor_si_2d
```

**nctr>1 note:** the existing sf_2d arm REJECTS nctr>1 spinor (`:10106-10110`). The D-05 fixture keeps an nctr>1 p shell, so the si_2d path must HANDLE nctr>1 (do not copy the rejection) — see RESEARCH WR-03 (`project_raw_nctr_coeff_transpose.md`); raw.rs already transposes the env COLUMN-major coeff to Shell ROW-major.

---

### `build_kappa_spinor_fixture` — `crates/cintx-oracle/src/fixtures.rs` (NEW) (fixture, test glue)

**Analog:** `build_adversarial_spinor_fixture` (`fixtures.rs:209-300`) — clone the whole geometry (non-square p×d, nctr=2 p shell, NON-ZERO rinv origin), then set genuine kappa≠0:
```rust
bas[KAPPA_OF]              = ?;   // EXISTING fixture: 0  → CHANGE: p kappa=+1 (LT, di=2l=2)
bas[BAS_SLOTS + KAPPA_OF]  = ?;   // EXISTING fixture: 0  → CHANGE: d kappa=-1 (GT, dj=2l+2=6)
```
This makes it the FIRST cintx fixture exercising the non-`(4l+2)` sizing path (`spinor_len(1,+1)==2`, `spinor_len(2,-1)==6`; block 2×6, buffer `di*dj*2 = 24` f64). Keeps every Phase-27 landmine (non-square, nctr>1) AND adds the kappa axis (RESEARCH §Spike Target D).

**Second fixture (D-05 secondary):** small real heavy-atom 2c Dirac/dyall-style single-atom spinor basis (element/exponents are Claude's discretion per A3) as a synthetic-blind-spot cross-check. Non-primary gate.

---

### `vendor_int1e_sp_spinor` — `crates/cintx-oracle/src/vendor_ffi.rs` (NEW) (FFI shim, test glue)

**Analog:** `vendor_int1e_ovlp_spinor` (`vendor_ffi.rs:4101-4124`) — copy verbatim, swap `ffi::int1e_ovlp_spinor` → `ffi::int1e_sp_spinor`:
```rust
pub fn vendor_int1e_sp_spinor(
    out: &mut [f64], shls: &[i32; 2],
    atm: &[i32], natm: i32, bas: &[i32], nbas: i32, env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_sp_spinor(
            out.as_mut_ptr(), ptr::null_mut(), shls.as_ptr() as *mut i32,
            atm.as_ptr() as *mut i32, natm, bas.as_ptr() as *mut i32, nbas,
            env.as_ptr() as *mut f64, ptr::null_mut(), ptr::null_mut(),
        )
    }
}
```
**ALSO ADD the extern declaration** in the `ffi` block (no existing `int1e_sp_spinor` extern — `int1e_ovlp_spinor`/`int1e_ipovlp_spinor` are the existing externs to mimic). `out` sized `ni_sp * nj_sp * 2` f64 (interleaved re/im); size from `vendor_CINTcgto_spinor` (`vendor_ffi.rs:3729`).

---

### Manifest row — `compiled_manifest.lock.json` + `api_manifest.rs` (MODIFIED) (config)

**Analog:** existing `*_spinor` ManifestEntry rows. The lock is the source of truth; editing it auto-syncs both audit sides (no fixtures list to touch — memory `project_ipovlpip_rank9_kernel.md`).

- `int1e_sp_spinor` row: `oracle_covered = false` (D-01 — STAYS UnsupportedApi this phase), stability `stable`.
- `component_rank`: RESEARCH Open Q #3 — verify against an existing spinor family's convention; output is `di*dj` complex (tensor rank 1; ng[7]=1). The 4 gc blocks are `ncomp_e1`, NOT tensor components. Watch component_rank truncation (`project_unstable_derivative_ports.md`).
- **Pitfall 6:** adding the row re-points positional `OperatorId::new(N)` / `_OPERATOR_ID: u32 = N` test consts → re-grep and fix by SYMBOL name (`project_operator_id_shift_breaks_hardcoded_test_consts.md`).

---

### SC#4 guard — `xtask/src/oracle_covered_update.rs` (MODIFIED) (guard)

**Analog:** the guard ALREADY EXISTS at `:50` (`if fixture.skipped { continue; }`, threat T-21-08-02). This phase ASSERTS it covers σ families: after `oracle-covered-update`, `int1e_sp_spinor` must stay `oracle_covered=false`. The skipped-fixture continue (`:50-52`) is the mechanism. Extend the comment block (`:36-49`) to note Phase-28 σ families stay deferred to Phase 29.

---

### `si_transform_parity.rs` (NEW) — `crates/cintx-oracle/tests/` (test, integration)

**Analog:** `spinor_deriv_parity.rs` (Phase-27, the D-10 no-silent-skip pattern). This is the D-01 PROOF vehicle: drive si_2d + σ·p assembler through `int1e_sp`, compare the flat buffer to vendor `c2s_si_1e` at atol=1e-12 — WITHOUT flipping any manifest flag.

---

## Shared Patterns

### SI Sign Convention (THE LANDMINE) — applies to `apply_bra_si_block`
**Source (CORRECT, use this):** `libcint-master/src/cart2sph.c:3958-3961` (`a_bra_cart2spinor_si`)
```
saR += caR*v1 + caI*vz - cbR*vy + cbI*vx
saI +=-caI*v1 + caR*vz + cbI*vy + cbR*vx
sbR += cbR*v1 - cbI*vz + caR*vy + caI*vx
sbI +=-cbI*v1 - cbR*vz - caI*vy + caR*vx
```
**Source (WRONG, do NOT reuse):** `c2spinor.rs:161-164` (`apply_si_block`, == `CINTc2s_ket_spinor_si1` `cart2sph.c:6883`) — `+caR*v1 - caI*vz + cbR*vy - cbI*vx` (3 sign flips on the cross/imaginary terms). Both are internally-consistent libcint functions for DIFFERENT call paths; `apply_si_block` stays correct for the covered single-block helper surface and is NOT deprecated. [VERIFIED: RESEARCH §Spike Target A, §Code Examples]

### Ordinary Ket Step (do NOT symmetrize) — applies to si_2d stage 2
**Source:** REUSE `apply_ket_transform` (`c2spinor.rs:766`) / `apply_ket_block` (`c2spinor.rs:811`) VERBATIM. `c2s_si_1e` calls the ORDINARY `a_ket_cart2spinor` (`cart2sph.c:4984`), identical to `c2s_sf_1e`. No `apply_ket_si` exists in cintx OR libcint (grep gate). [VERIFIED: cart2sph.c:4984 vs :4900]

### Spinor Buffer Sizing — applies to si_2d, the assembler, fixtures, FFI
**Source:** ALWAYS `spinor_len(l, kappa)` (`c2spinor.rs:25`): kappa==0→4l+2, kappa<0→2l+2 (GT), kappa>0→2l (LT). ZERO literal `4*l+2` / `4l+2` anywhere in the si_2d path (grep gate). [VERIFIED: c2spinor.rs:25-33 == cart2sph.c:3537]

### gout→gc Layout (component-interleaved → component-blocked) — applies to assembler + dispatch
**Source:** libcint `CINTdmat_transpose` (`cint1e.c:157`) converts `gout[n*4+comp]` → `gctr[comp*(nf*nc)+n]`. cintx: emit pre-blocked on-device following `write_component_leading_staging` (`one_electron.rs:8300`), so the host transform reads `gc_x=block0, gc_y=block1, gc_z=block2, gc_1=block3`. For `int1e_sp`, `n_comp=4>1` so the transpose ALWAYS runs. This is a SEPARATE transpose from the KET→BRA orientation transpose owned inside `cart_to_spinor_si_2d`. [VERIFIED: cint1e.c:156-158]

### Error Handling — applies to si_2d
**Source:** `cart_to_spinor_sf_2d` (`c2spinor.rs:544-559`). Validate buffer sizes BEFORE any write: `ChunkPlanFailed` on undersized cart, `BufferTooSmall { required, provided }` on undersized staging. No partial writes on failure (CLAUDE.md OOM-safe stop contract). thiserror `cintxRsError` for the library; anyhow for the oracle harness.

### No-Silent-Skip Vendor Parity — applies to si_transform_parity.rs
**Source:** Phase-27 D-10 pattern (`spinor_deriv_parity.rs`). Vendor parity double-gated `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`; the test must FAIL (not skip) when the vendor arm is expected but did not run. [memory `reference_oracle_vendor_parity_invocation.md`]

---

## No Analog Found

None. Every file has a strong in-repo analog (RESEARCH confirms ~80% reuse). The only genuinely-novel logic is the `apply_bra_si_block` SIGN CONVENTION (which has a verified C source, just no correct Rust analog) and the σ·p 4-component packing (gradient nabla machinery + a zero scalar slot).

---

## Metadata

**Analog search scope:** `crates/cintx-cubecl/src/transform/`, `crates/cintx-cubecl/src/kernels/`, `crates/cintx-oracle/src/`, `xtask/src/`, `crates/cintx-ops/generated/`
**Files scanned (read or grepped):** `c2spinor.rs`, `one_electron.rs`, `fixtures.rs`, `vendor_ffi.rs`, `oracle_covered_update.rs`, `compiled_manifest.lock.json`, plus 28-CONTEXT.md / 28-RESEARCH.md
**Pattern extraction date:** 2026-05-31
