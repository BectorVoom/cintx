# Phase 28: Spin-Included `c2s_si` Transform + σ·p Module (Gap B2) - Research

**Researched:** 2026-05-31
**Domain:** Relativistic spinor cart→spinor transform (Pauli σ-coupling) + σ·p G-tensor device assembler; libcint 6.1.3 byte-identity
**Confidence:** HIGH (all spike targets verified against vendored C source `libcint-master/src/*.c`; existing Rust read directly)

> **This research IS the D-06 HARD-GATE spike.** Every byte-level claim below is `[VERIFIED: libcint-master/src/<file>:<line>]` against the vendored C, or `[VERIFIED: <rust file>:<line>]` against the existing cintx code. The `## Validation Architecture` section contains the per-target spike deliverables (C source location, exact ordering/strides, hand-derived numeric check). **The single most important finding is the sign-convention discrepancy in §Spike Target A — the existing `apply_si_block` does NOT match the function `c2s_si_1e` uses, and must not be reused verbatim.**

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01 (INFRASTRUCTURE-ONLY):** Phase 28 flips **ZERO** σ families to `oracle_covered=true`. It lands the si_2d transform + reusable σ·p assembler + kappa fixture, and proves FND-05 via a **dedicated transform/component-level byte-identity test** that drives the si_2d transform + σ·p assembler through the `int1e_sp` path (D-02) and compares the flat buffer to vendored libcint `c2s_si_1e` at atol=1e-12 — **not** by flipping a manifest coverage flag. **Every σ family — including the `int1e_sp` vehicle — stays `UnsupportedApi`** this phase; ALL σ flips → Phase 29. The `oracle-covered-update` guard (SC#4) must refuse to flip any σ family whose only fixture was `skipped`.
- **D-02 (validation vehicle = `int1e_sp`):** σ·p on the **bra only**. `c2s_si_1e` mixes the Pauli σ on the bra (`a_bra_cart2spinor_si` over `gc_x/gc_y/gc_z/gc_1`) and uses the **ordinary** `a_ket_cart2spinor` on the ket. Thinnest family exercising BOTH new pieces. `int1e_sp` is the proof *vehicle* only — NOT flipped to `oracle_covered` this phase. (Rejected: `int1e_spsp` drags ket-side σ·p in; `int1e_sigma` under-tests σ·p.)
- **D-03 (σ·p assembler = reusable generic `#[cube]`):** Standalone reusable generic `#[cube]` emitter (a dedicated σ/`gout_si` module) parameterized so Phase 29's whole σ-group (`sp`, `spsp`, `spnucsp`, `sprinvsp`, `sigma`, …) reuses it directly. Front-load the architecture. (Rejected: "minimal now, generalize in 29".)
- **D-04 (host/device split):** The **si transform** (`cart_to_spinor_si_2d`) is a **HOST** function in `c2spinor.rs`, mirroring `cart_to_spinor_sf_2d`. The **σ·p gout assembler** that emits `gc_x/gc_y/gc_z/gc_1` is a **DEVICE `#[cube]`** step, mirroring the existing nabla/gout gradient machinery.
- **D-05 (BOTH fixtures):** (1) **Primary gate — adversarial kappa≠0:** reuse Phase-27 D-08 geometry (non-square bra/ket e.g. p×d, ≥1 shell nctr>1) **but with genuine kappa≠0** (p kappa=+1 → LT-only `j=l−1/2`, d kappa=−1 → GT-only `j=l+1/2`) → exercises the non-`(4l+2)` spinor sizing path (`di = 2l` or `2l+2`). Added as `build_kappa_spinor_fixture` in `fixtures.rs` (alongside `build_adversarial_spinor_fixture` at `:209`). (2) **Realism cross-check — one small real heavy-atom 2c-basis case** (single-atom Dirac/dyall-style spinor basis). Secondary, non-primary gate.
- **D-06 (research spike = HARD GATE):** Full design spike before plan tasks finalized. Nail against hand-checked vendor values: (a) `a_bra_cart2spinor_si` 4-block stride/ordering; (b) bra-Pauli-mix / ket-ordinary split; (c) device→host buffer hand-off; (d) kappa≠0 GT/LT-only sizing through `spinor_len`. Do NOT shortcut it. **← This document discharges D-06.**

### Claude's Discretion
- Exact molecule/element + kappa assignments for the fixture (subject to D-05 hard constraints: non-square, nctr>1 somewhere, kappa≠0).
- Internal module naming/factoring for the reusable σ·p assembler and the si_2d transform.
- Plan boundaries (e.g. spike → transform+assembler → fixture+parity).
- Exact `int1e_sp` gout component ordering — resolve from libcint `intor3.c` during the spike. **← RESOLVED below, §Spike Target E.**

### Deferred Ideas (OUT OF SCOPE)
- **All Group-4 σ family `oracle_covered` flips → Phase 29** — including the `int1e_sp` vehicle itself. `int1e_spsp/spnucsp/sprinvsp/srsr/sr/srnucsr/sigma` and the 2e `spsp1/srsr1/ssp*/sps*/vsp*/spv*` reuse the Phase-28 si_2d transform + reusable σ·p module; their kernels and coverage flips land in 29.
- **The `iket_si` (`c2s_si_1ei`) 2D variant and the 2e si transforms** (`c2s_si_2e1/2e1i/2e2/2e2i`) — needed for GIAO×σ (Phase 30) and gauge/Breit–Gaunt 2e (Phase 31). The single-block `iket_si` already exists; the 2D/2e si drivers are OUT of Phase 28 scope.
- **GIAO×σ slice** (Phase 30) and **gauge/Breit–Gaunt 2e** (Phase 31).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FND-05 | Gap B2: the spin-included spinor transform `cart_to_spinor_si_2d` (4-block `gc_x/gc_y/gc_z/gc_1` input of libcint `c2s_si_1e`, `cart2sph.c:4947`) + companion σ·p G-tensor assembler, validated against a kappa-bearing relativistic oracle fixture at atol=1e-12. | §Spike Targets A–E supply the exact bra-mix formula (A), the bra-Pauli/ket-ordinary split (B), the device→host gout→gc transpose hand-off (C), the `spinor_len` GT/LT sizing (D), and the `int1e_sp` 4-component gout (E). §Validation Architecture maps FND-05 to a transform/component byte-identity test + a vendor parity test. |
</phase_requirements>

## Summary

Phase 28 adds the **2D spin-included cart→spinor transform** `cart_to_spinor_si_2d` (host fn in `c2spinor.rs`, the bra+ket analog of the existing `cart_to_spinor_sf_2d` at `c2spinor.rs:531`) plus a **reusable generic `#[cube]` σ·p G-tensor assembler** that emits the four `gc_x/gc_y/gc_z/gc_1` cartesian blocks consumed by the transform. The byte-authoritative reference is libcint's `c2s_si_1e` (`cart2sph.c:4947`), which calls `a_bra_cart2spinor_si` (`:3920`) on the bra and the **ordinary** `a_ket_cart2spinor` (`:4343`) on the ket — confirming the bra-Pauli-mix / ket-ordinary split (no ket symmetrization).

**The decisive spike finding:** libcint has **two distinct si sign conventions**, and the existing Rust `apply_si_block` (`c2spinor.rs:124-172`) implements the WRONG one for this phase. `apply_si_block` was ported from the single-block helper `CINTc2s_ket_spinor_si1` (`cart2sph.c:6839`, sign pattern `+caR*v1 - caI*vz + cbR*vy - cbI*vx`). But the 2D path `c2s_si_1e` uses `a_bra_cart2spinor_si` (`cart2sph.c:3958`, sign pattern `+caR*v1 + caI*vz - cbR*vy + cbI*vx`). **These are different functions with different signs.** `cart_to_spinor_si_2d`'s bra step MUST transcribe `a_bra_cart2spinor_si` verbatim — it cannot reuse `apply_si_block`. (The discrepancy is reconciled below; both are internally consistent libcint functions for different call paths.)

The second decisive finding is the **device→host buffer hand-off** (Target C): `CINT1e_loop` (`cint1e.c:157`) calls `CINTdmat_transpose(gctr, gctrj, nf*nc, n_comp)` which converts the gout emitter's **component-interleaved** layout (`gout[n*4 + comp]`, comp-fastest) into the **component-blocked** layout `c2s_si_1e` reads (`gc_x = gctr`, `gc_y = gctr + nf*nc`, …). The device σ·p assembler must therefore emit `gout[n*4 + {x,y,z,1}]` (or write the 4 blocks pre-transposed), and the host transform reads 4 contiguous `nf*ictr*jctr`-sized blocks. For `int1e_sp`, `ncomp_e1 = ng[5] = 4`, so the transpose **always** runs.

**Primary recommendation:** Implement `cart_to_spinor_si_2d` as a host fn that (1) owns the KET→BRA transpose internally (Phase 27 D-06 pattern, `c2spinor.rs:1342-1354`), (2) runs a bra step transcribing `a_bra_cart2spinor_si` (the `+caI*vz`/`-cbR*vy`/`+cbI*vx` sign convention — NOT `apply_si_block`), (3) runs the ordinary `a_ket_cart2spinor` ket step (reuse the existing `apply_ket_transform`/`apply_ket_block` from `sf_2d` verbatim — the ket side is identical between sf and si), (4) sizes all buffers from `spinor_len`. The σ·p assembler is a generic `#[cube]` that emits 4-component `gout` from `g0` (overlap base) and `g1 = nabla_i(g0)` per `CINTgout1e_int1e_sp` (`intor3.c:416`). Validate via a transform-level byte-identity test against hand-rolled vendor `c2s_si_1e` output (the proof must NOT flip any manifest flag — D-01).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| σ·p G-tensor assembly (emit `gc_x/gc_y/gc_z/gc_1` from `g0` + `nabla_i g0`) | **Device `#[cube]`** (cintx-cubecl kernels) | — | D-04: mirrors existing nabla/gout gradient machinery (`one_electron.rs` `nabla1i`/gout). The σ·p contraction is hot per-primitive device work. |
| Component-interleaved → component-blocked transpose (gout layout → gc_* layout) | **Host** (post-kernel marshaling) OR device write-ordering | Device | libcint does this in `CINT1e_loop` via `CINTdmat_transpose`. cintx can either emit pre-blocked from the device or transpose host-side; the existing staging-writer pattern (`write_component_leading_staging`, `one_electron.rs:8300`) already produces component-leading blocks. |
| `cart_to_spinor_si_2d` (bra Pauli-mix + ket ordinary + zcopy interleave) | **Host** (cintx-cubecl `transform/c2spinor.rs`) | — | D-04: transforms run on the contracted cart staging buffer post-kernel; this is CPU control-plane marshaling, not hot compute. Mirrors `cart_to_spinor_sf_2d` exactly. |
| KET→BRA orientation transpose | **Host** (inside `cart_to_spinor_si_2d`) | — | Phase 27 D-06: device cart blocks are KET-major; the transpose lives in the transform layer, never the launcher. |
| Kappa→GT/LT block dispatch + buffer sizing | **Host** (`spinor_len`, `c2spinor.rs:25`) | — | Pure control-plane sizing; already correct in `spinor_len`. |
| nctr>1 coefficient handling | **Host** (raw→Shell marshaling, already fixed) | — | libcint env coeff COLUMN-major vs cintx Shell ROW-major (memory WR-03); the fixture keeps an nctr>1 shell to regression-guard this. |
| Vendor parity oracle (FFI to libcint `int1e_sp_spinor`) | **Host** (cintx-oracle, test/glue) | — | Double-gated `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`. |

## Standard Stack

This is an internal-implementation phase in an established workspace — no new external dependencies. The relevant in-repo "stack":

### Core (in-repo, reuse verbatim)
| Module | Location | Purpose | Why Standard |
|--------|----------|---------|--------------|
| `cart_to_spinor_sf_2d` | `crates/cintx-cubecl/src/transform/c2spinor.rs:531` | Structural template for `cart_to_spinor_si_2d` (bra step → ket step → zcopy interleave) | The si_2d transform is the sf_2d transform with the σ-Pauli bra step swapped in; ket step + zcopy are identical. [VERIFIED: c2spinor.rs:531-612] |
| `apply_ket_transform` / `apply_ket_block` | `c2spinor.rs:766` / `:811` | Ordinary ket cart→spinor step (the `a_ket_cart2spinor` analog) | D-02/B: `c2s_si_1e` uses the ORDINARY ket transform — reuse these verbatim, no si-specific ket code. [VERIFIED: c2spinor.rs:766-805 matches cart2sph.c:4343] |
| `apply_bra_block` | `c2spinor.rs:668` | sf bra step (template for the new si bra step) | The si bra step has the SAME loop/index structure but the σ-coupled accumulation (4 cart inputs `v1/vx/vy/vz`, the `a_bra_cart2spinor_si` formula). [VERIFIED: c2spinor.rs:668-710] |
| `spinor_len` | `c2spinor.rs:25` | GT/LT/both spinor sizing | Already correct: kappa<0→2l+2, kappa>0→2l, kappa==0→4l+2. [VERIFIED: c2spinor.rs:25-33 == cart2sph.c:3537 `_len_spinor`] |
| `bra_coeff_refs` + CG tables | `c2spinor.rs:714` + `c2spinor_coeffs.rs` | `cart2j_gt_l*`/`cart2j_lt_l*` CG coupling coefficients | Same `g_c2s[l]` coefficients libcint uses; `a_bra_cart2spinor_si` and `a_ket_cart2spinor` both index them as `coeff[i*nf*2 + {0,nf} + n]`. [VERIFIED: cart2sph.c:3954-3957] |
| nabla1i `#[cube]` machinery | `one_electron.rs` (`nabla1i`-style, e.g. `:1864` bra-direction nabla helper) | Produces `g1 = D_i(g0)` for the σ·p assembler | `CINTgout1e_int1e_sp` builds `g1 = G1E_D_I(g0)` then forms `gc_{x,y,z}` from `g1`. [VERIFIED: intor3.c:422 `G1E_D_I(g1,g0,...)`] |
| `write_component_leading_staging` | `one_electron.rs:8300` | Component-leading, contraction-blocked staging writer | The existing pattern for laying out the per-component blocks; the σ·p assembler / transpose can follow it. [VERIFIED: one_electron.rs:8300] |

### Supporting (oracle / test glue — anyhow side per CLAUDE.md)
| Module | Location | Purpose |
|--------|----------|---------|
| `build_adversarial_spinor_fixture` | `crates/cintx-oracle/src/fixtures.rs:209` | Template for the new `build_kappa_spinor_fixture` (D-05). Non-square p×d, nctr=2 p shell, kappa=0. Clone + set kappa≠0. |
| `vendor_int1e_*_spinor` pattern | `crates/cintx-oracle/src/vendor_ffi.rs` (e.g. `vendor_int1e_ovlp_sph:21`) | Add `vendor_int1e_sp_spinor` FFI binding. |
| oracle `compare` | `crates/cintx-oracle/src/compare.rs` | atol=1e-12 flat-buffer comparison. |
| `run_oracle_covered_update` guard | `xtask/src/oracle_covered_update.rs:11` | SC#4: the `if fixture.skipped { continue; }` guard at `:50` already refuses to stamp skipped fixtures. |

**Alternatives Considered:** None — D-01..D-06 lock the architecture. The only genuine fork (reuse `apply_si_block` vs. write a new si bra step) is resolved decisively against reuse by §Spike Target A.

## Architecture Patterns

### System Architecture Diagram (data flow for `int1e_sp` validation vehicle)

```
                          DEVICE  (#[cube])                              HOST (c2spinor.rs)
  shell pair (i=p,j=d)
  primitives, coeffs
        │
        ▼
  ┌──────────────────┐   g0 = overlap base G-tensor (gx·gy·gz)
  │ σ·p ASSEMBLER     │   g1 = nabla_i(g0)            [G1E_D_I]
  │ (generic #[cube]) │   per cart n:
  │  emit 4-comp gout │     s[0]=g1x·g0y·g0z  (→ gc_x)
  │                   │     s[1]=g0x·g1y·g0z  (→ gc_y)
  │                   │     s[2]=g0x·g0y·g1z  (→ gc_z)
  │                   │     gc_1 = 0          (int1e_sp scalar slot)
  └────────┬──────────┘   layout: gout[n*4 + {x,y,z,1}]  (COMPONENT-INTERLEAVED)
           │
           │ contract over primitives → gctrj  (still component-interleaved)
           ▼
  ┌─────────────────────────────────────────────┐
  │ TRANSPOSE  (CINTdmat_transpose / host marshal)│  gout[n*4+c] → gctr[c*(nf*nc) + n]
  │  → 4 contiguous blocks gc_x|gc_y|gc_z|gc_1    │  (COMPONENT-BLOCKED, each nf*ictr*jctr)
  └────────┬────────────────────────────────────┘
           │  (cart blocks are KET-major)
           ▼
  ┌─────────────────────────────────────────────────────────┐  HOST cart_to_spinor_si_2d
  │ (0) KET→BRA transpose  (own it here, Phase-27 D-06)        │  sph_k[j*nci+i] → bra[i*ncj+j]
  │ (1) BRA step = a_bra_cart2spinor_si over gc_x/y/z/1        │  reads v1,vx,vy,vz; Pauli-σ mix
  │         saR += caR*v1 + caI*vz - cbR*vy + cbI*vx   ← NEW   │  → tmp1[a|b spinor blocks]
  │ (2) KET step = a_ket_cart2spinor (ORDINARY, reuse sf)      │  → tmp2[di × dj] complex
  │ (3) zcopy_ij interleave column-major                      │  staging[(j*di+i)*2] = re/im
  └────────┬──────────────────────────────────────────────────┘
           ▼
  interleaved-complex spinor output  [re0,im0,re1,im1,…]  column-major (j outer, i inner)
  size = di*dj*2  (di,dj from spinor_len; kappa≠0 → 2l or 2l+2, NOT 4l+2)
           │
           ▼
  oracle compare vs vendor int1e_sp_spinor  (c2s_si_1e)  @ atol=1e-12
```

### Recommended Module Structure
```
crates/cintx-cubecl/src/
├── transform/
│   └── c2spinor.rs           # ADD: cart_to_spinor_si_2d (HOST); apply_bra_si_block (NEW si sign convention)
└── kernels/
    ├── one_electron.rs       # WIRE: int1e_sp Spinor dispatch arm calls cart_to_spinor_si_2d
    └── sigma_p.rs (NEW)      # σ·p generic #[cube] assembler (gout_si module, D-03 reusable)
crates/cintx-ops/
├── src/generated/api_manifest.rs   # int1e_sp_spinor row (oracle_covered=false, UnsupportedApi)
└── generated/compiled_manifest.lock.json  # source of truth; lock edit auto-syncs both audit sides
crates/cintx-oracle/src/
├── fixtures.rs               # ADD: build_kappa_spinor_fixture + a heavy-atom fixture
├── vendor_ffi.rs             # ADD: vendor_int1e_sp_spinor
└── tests/
    └── (new) si_transform_parity.rs   # transform/component byte-identity test (D-01 proof, no flag flip)
```

### Pattern 1: The si_2d transform = sf_2d with σ-coupled bra step
**What:** `cart_to_spinor_si_2d` has the identical 3-stage skeleton as `cart_to_spinor_sf_2d` (bra → ket → zcopy). Only the bra step changes: it consumes FOUR cart blocks (`gc_1` scalar + `gc_x/gc_y/gc_z` Pauli) instead of one, and applies the `a_bra_cart2spinor_si` formula.
**When to use:** This is THE pattern for Phase 28's transform.
**Example (the NEW bra step — transcribe `a_bra_cart2spinor_si` VERBATIM):**
```rust
// Source: libcint-master/src/cart2sph.c:3958-3961 (a_bra_cart2spinor_si)
// CG coeff layout: caR/caI = coeff[i*nf*2 + n]; cbR/cbI = coeff[i*nf*2 + nf + n]
// (same indexing as the sf bra step apply_bra_block, c2spinor.rs:694-697)
for i in 0..nd {                       // bra spinor row
    for j in 0..ncj {                  // ket cart column
        let (mut sa_r, mut sa_i, mut sb_r, mut sb_i) = (0.0, 0.0, 0.0, 0.0);
        for n in 0..nf {               // bra cart index
            // gc_* are KET-major after transpose; index per the host buffer layout.
            let v1 = gc_1[n * ncj + j];
            let vx = gc_x[n * ncj + j];
            let vy = gc_y[n * ncj + j];
            let vz = gc_z[n * ncj + j];
            let ca_r = coeff_r[i * 2 * nf + n];
            let ca_i = coeff_i[i * 2 * nf + n];
            let cb_r = coeff_r[i * 2 * nf + nf + n];
            let cb_i = coeff_i[i * 2 * nf + nf + n];
            // *** a_bra_cart2spinor_si signs — NOT apply_si_block's signs ***
            sa_r +=  ca_r * v1 + ca_i * vz - cb_r * vy + cb_i * vx;
            sa_i += -ca_i * v1 + ca_r * vz + cb_i * vy + cb_r * vx;
            sb_r +=  cb_r * v1 - cb_i * vz + ca_r * vy + ca_i * vx;
            sb_i += -cb_i * v1 - cb_r * vz - ca_i * vy + ca_r * vx;
        }
        // alpha (upper) block then beta (lower) block, exactly like apply_bra_block
        // tmp1 layout: gspaR/gspaI = first nket*nd; gspbR/gspbI = +nket*nd (cart2sph.c:3928)
    }
}
```

### Pattern 2: σ·p generic `#[cube]` assembler (D-03)
**What:** A reusable device emitter producing the 4-component `gout`. For `int1e_sp`: `s[0..2]` from `g1 = nabla_i(g0)`, scalar slot zero.
**Example:**
```rust
// Source: libcint-master/src/autocode/intor3.c:416-440 (CINTgout1e_int1e_sp)
// g0 = ovlp base (gx*gy*gz product); g1 = G1E_D_I(g0) (bra-direction nabla, c2sph.c:322 CINTnabla1i_1e)
// 4 components per cart n, COMPONENT-INTERLEAVED: gout[n*4 + {0,1,2,3}]
s[0] = g1[ix]*g0[iy]*g0[iz];   // → gc_x
s[1] = g0[ix]*g1[iy]*g0[iz];   // → gc_y
s[2] = g0[ix]*g0[iy]*g1[iz];   // → gc_z
gout[n*4+0] = s[0];  gout[n*4+1] = s[1];  gout[n*4+2] = s[2];  gout[n*4+3] = 0.0;
```
Generalize: parameterize the 4-component emission so `int1e_sigma` (12-comp, `gout[n*12+...]`, 3 tensor × 4 gc) and `int1e_spsp` (ket-side σ·p, uses `c2s_sf_1e` actually — see §Open Questions) reuse the same gc-block packing.

### Anti-Patterns to Avoid
- **Reusing `apply_si_block` (`c2spinor.rs:124`) for the si_2d bra step.** It implements `CINTc2s_ket_spinor_si1`'s signs (`-caI*vz + cbR*vy - cbI*vx`), which DIFFER from `a_bra_cart2spinor_si`'s signs (`+caI*vz - cbR*vy + cbI*vx`). Wrong by sign on 3 of 4 terms. [VERIFIED: cart2sph.c:3958 vs :6883]
- **Symmetrizing the ket.** `c2s_si_1e` uses the ORDINARY `a_ket_cart2spinor` (`cart2sph.c:4984`), NOT a si ket. Reuse `apply_ket_transform` verbatim. [VERIFIED: cart2sph.c:4984]
- **Hardcoding `4l+2` spinor sizing.** kappa≠0 → `2l` (LT) or `2l+2` (GT). Always call `spinor_len`. [VERIFIED: cart2sph.c:3537]
- **Doing the KET→BRA transpose in the launcher.** It belongs INSIDE the transform (Phase 27 D-06). [VERIFIED: c2spinor.rs:1338-1339 comment]
- **Square-block test fixtures.** A square bra×ket block is transpose-symmetric and hides the orientation bug. Use non-square p×d. [VERIFIED: SKILL.md D-07; project memory `project_1e_gpu_port_scalar_only.md`]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Ordinary ket cart→spinor step | A new si-specific ket transform | `apply_ket_transform`/`apply_ket_block` (`c2spinor.rs:766/811`) | `c2s_si_1e` uses the IDENTICAL ordinary ket as `c2s_sf_1e`. [VERIFIED: cart2sph.c:4984 vs :4900] |
| Spinor sizing (GT/LT/both) | Inline `4l+2` / kappa branches | `spinor_len(l, kappa)` (`c2spinor.rs:25`) | Already byte-matches `_len_spinor`. |
| CG coupling coefficients | Recompute Clebsch-Gordan | `c2spinor_coeffs.rs` `cart2j_gt/lt_l*` tables | Same `g_c2s[l]` libcint uses. |
| Complex interleave output write | Custom complex packer | `zcopy_ij` pattern (`c2spinor.rs:603-608`) | column-major `staging[(j*di+i)*2]=re,+1=im`. [VERIFIED: cart2sph.c:4499 zcopy_ij] |
| nabla `g1 = D_i(g0)` | New derivative recurrence | Existing `nabla1i` `#[cube]` helpers (`one_electron.rs:1864`) | Faithful `CINTnabla1i_1e` port already on-device. |
| nctr>1 coeff transpose | Re-handle column/row-major | Already fixed in raw.rs (memory `project_raw_nctr_coeff_transpose.md`) | env COLUMN-major → Shell ROW-major handled. Fixture just regression-guards it. |

**Key insight:** Phase 28 is ~80% reuse. The genuinely new code is (1) the σ-coupled bra step `apply_bra_si_block` (the `a_bra_cart2spinor_si` formula), (2) the generic σ·p `#[cube]` gout assembler, (3) the gout→gc transpose hand-off, (4) two fixtures + one vendor FFI. Everything else composes from `sf_2d`.

## Common Pitfalls

### Pitfall 1: The two si sign conventions (THE landmine)
**What goes wrong:** Reusing `apply_si_block` makes the transform compile and even pass on some symmetric inputs, but it produces wrong values vs vendor `c2s_si_1e` at atol=1e-12.
**Why it happens:** Phase 12 ported the SINGLE-BLOCK helper `CINTc2s_ket_spinor_si1` into `apply_si_block`. That helper has a different sign convention than `a_bra_cart2spinor_si`, which is what the 2D `c2s_si_1e` path uses. Both are correct libcint functions for their respective call sites.
**How to avoid:** Write a NEW `apply_bra_si_block` transcribing `cart2sph.c:3958-3961` verbatim. Do not call `apply_si_block`.
**Warning signs:** Sign-flip pattern in the parity diff (correct magnitudes, wrong signs on imaginary/cross terms).

### Pitfall 2: gout layout (component-interleaved) vs gc layout (component-blocked)
**What goes wrong:** The device assembler writes `gout[n*4+comp]`, but `c2s_si_1e` reads `gc_x = gctr; gc_y = gctr + nf*nc`. Without the transpose, the four blocks are interleaved and the transform reads garbage.
**Why it happens:** libcint hides the transpose in `CINT1e_loop` (`CINTdmat_transpose`, `cint1e.c:157`) — easy to miss when reading only `c2s_si_1e`.
**How to avoid:** Either emit pre-blocked from the device (write directly to `comp*(nf*nc)+n`) or run a host transpose after contraction. For `int1e_sp` `ncomp_e1=4`, `ncomp_tensor=1` → `n_comp=4 > 1`, so the transpose ALWAYS runs. [VERIFIED: cint1e.c:156-158]
**Warning signs:** Output looks like a 4-way interleave of the right values.

### Pitfall 3: kappa≠0 non-square sizing
**What goes wrong:** Buffers sized `4l+2` overflow/underflow when kappa≠0 (di=2l or 2l+2).
**Why it happens:** Phase 27's kappa=0 fixture never exercised the GT/LT-only path.
**How to avoid:** Size every buffer from `spinor_len`; the D-05 primary fixture (p kappa=+1 LT-only, d kappa=−1 GT-only) is the gate.
**Warning signs:** Buffer-too-small errors only on the kappa fixture; passing on kappa=0.

### Pitfall 4: KET→BRA transpose orientation
**What goes wrong:** Device cart blocks are KET-major (`sph_k[j*nci+i]`); the bra step reads BRA-major (`cart[n*ncj+j]`). Skipping the transpose silently transposes the block.
**Why it happens:** Latent on square blocks (`nci==ncj`). [VERIFIED: c2spinor.rs:1335-1341]
**How to avoid:** Own the transpose in `cart_to_spinor_si_2d` (mirror `c2spinor.rs:1342-1354`).
**Warning signs:** Passes on square test cases, fails on the non-square p×d fixture.

### Pitfall 5: CubeCL CpuRuntime FP-environment side effect on flat-atol gates
**What goes wrong:** A `#[cube]` CpuRuntime launch perturbs SUBSEQUENT host f64 accumulation ~1e-11, which can trip the atol=1e-12 flat parity gate even with a bit-identical kernel.
**Why it happens:** Documented CubeCL 0.10.0 CpuRuntime behavior. [VERIFIED: project memory `project_cubecl_cpuruntime_fp_env_side_effect.md`]
**How to avoid:** Suspect this before chasing kernel numerics; the σ·p assembler is a new device kernel. Consider batching launches or keeping the suspect band host (escape hatch). `fma()` IS fused/bit-exact on CpuRuntime (memory `project_cubecl_cpuruntime_fma_fused.md`).
**Warning signs:** Parity off by ~1e-11 only when the device kernel runs in the same process before host accumulation.

### Pitfall 6: New-family registration shifts OperatorIds
**What goes wrong:** Adding `int1e_sp_spinor` to the manifest re-points hardcoded `OperatorId::new(N)` / `_OPERATOR_ID: u32 = N` test consts at a different family (→ InvalidShellTuple arity mismatch).
**Why it happens:** Positional manifest ordering. [VERIFIED: project memory `project_operator_id_shift_breaks_hardcoded_test_consts.md`]
**How to avoid:** Reference families by SYMBOL name, not positional int. Re-grep hardcoded id consts after adding the row.

## Code Examples

### Resolving the sign discrepancy (the spike's core artifact)
```
// libcint c2s_si_1e (cart2sph.c:4947) BRA step → a_bra_cart2spinor_si (cart2sph.c:3958):
//   saR += caR*v1 + caI*vz - cbR*vy + cbI*vx
//   saI +=-caI*v1 + caR*vz + cbI*vy + cbR*vx
//   sbR += cbR*v1 - cbI*vz + caR*vy + caI*vx
//   sbI +=-cbI*v1 - cbR*vz - caI*vy + caR*vx
//
// EXISTING Rust apply_si_block (c2spinor.rs:161) == CINTc2s_ket_spinor_si1 (cart2sph.c:6883):
//   sa_re += caR*v1 - caI*vz + cbR*vy - cbI*vx     ← DIFFERENT (3 sign flips)
//   sa_im += caI*v1 + caR*vz + cbI*vy + cbR*vx
//   sb_re += cbR*v1 + cbI*vz - caR*vy - caI*vx
//   sb_im += cbI*v1 - cbR*vz - caI*vy + caR*vx
//
// → cart_to_spinor_si_2d's bra step uses the FIRST (a_bra_cart2spinor_si) convention.
```

### Ordinary ket step (reuse, do not re-derive)
```c
// Source: libcint-master/src/cart2sph.c:4368-4400 (a_ket_cart2spinor)
// Complex CG multiply over nf2 = 2*nf ket-cart indices (alpha block then beta block):
//   gspR[j+i*nbra] += cR*gR - cI*gI
//   gspI[j+i*nbra] += cI*gR + cR*gI
// == cintx apply_ket_block (c2spinor.rs:811). Identical for sf and si.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single-block si only (`cart_to_spinor_si`, Phase 12) | Full 2D si transform `cart_to_spinor_si_2d` (Phase 28) | This phase | Enables σ-operator families (Groups 4/6, GIAO×σ). |
| σ families `UnsupportedApi` | Still `UnsupportedApi` after Phase 28 (D-01); flipped in Phase 29 | Phase 29 | Phase 28 proves infra via transform test, not flag flip. |

**Deprecated/outdated:** None. `apply_si_block` is NOT deprecated — it remains the correct port of `CINTc2s_ket_spinor_si1` for the covered single-block helper surface; it is simply the wrong function for the 2D path.

## Runtime State Inventory

Phase 28 is additive (new transform, new kernel, new fixtures, new manifest row) — not a rename/refactor. No stored data, live-service config, OS-registered state, secrets, or build artifacts carry a renamed string.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no datastore keys involved | None |
| Live service config | None | None |
| OS-registered state | None | None |
| Secrets/env vars | `CINTX_ORACLE_BUILD_VENDOR=1` gates vendor parity — existing, unchanged | None (reuse existing gate) |
| Build artifacts | Manifest lock `compiled_manifest.lock.json` regenerates; adding a row re-points positional OperatorIds (Pitfall 6) | Re-grep hardcoded `OperatorId::new(N)` test consts after adding `int1e_sp_spinor` row |

## Validation Architecture

> Consumed by the Nyquist VALIDATION.md gate. `nyquist_validation` is enabled (not `false` in config). Per-spike-target entries below each give: C source location, exact ordering/strides observed, and a hand-derived numeric check method for byte-for-byte verification.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + cargo nextest; oracle parity via vendored libcint FFI |
| Config file | none (cargo workspace) |
| Quick run command | `cargo test -p cintx-cubecl --lib transform::c2spinor` |
| Vendor parity command | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test si_transform_parity` |
| Full suite command | `cargo test --workspace` then `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FND-05 | `cart_to_spinor_si_2d` byte-matches vendor `c2s_si_1e` (transform-level, D-01 proof, NO flag flip) | integration (vendor) | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test si_transform_parity` | ❌ Wave 0 |
| FND-05 | si bra step matches `a_bra_cart2spinor_si` signs (unit, no device) | unit | `cargo test -p cintx-cubecl --lib apply_bra_si_block` | ❌ Wave 0 |
| FND-05 | σ·p `#[cube]` assembler device-vs-host (gout 4-component) | integration | `cargo test -p cintx-cubecl --features cpu sigma_p_device_matches_host` | ❌ Wave 0 |
| FND-05 | kappa≠0 sizing (GT/LT, 2l / 2l+2) on non-square nctr>1 fixture | integration (vendor) | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu kappa_spinor` | ❌ Wave 0 |
| SC#4 | `oracle-covered-update` refuses to flip σ families with skipped/absent fixtures | integration | `cargo run -p xtask -- oracle-covered-update` then assert `int1e_sp_spinor` stays `oracle_covered=false` | partial (guard at `oracle_covered_update.rs:50` exists) |
| no-silent-skip | vendor parity test FAILS (not skips) if double-gate env not set in CI parity job | unit | (Phase 27 D-10 assertion pattern) | ❌ Wave 0 |

---

### Spike Target A — `a_bra_cart2spinor_si` 4-block stride/ordering
**C source:** `libcint-master/src/cart2sph.c:3920-3968` (`a_bra_cart2spinor_si`); driver at `:4947-4992` (`c2s_si_1e`); block setup at `:4971-4974`.
**Observed (exact):**
- The four blocks are CONTIGUOUS in `gctr`, each of size `nf * i_ctr * j_ctr` where `nf = envs->nf = nfi*nfj` cart:
  ```c
  double *gc_x = gctr;                          // cart2sph.c:4971
  double *gc_y = gc_x + nf * i_ctr * j_ctr;     // :4972
  double *gc_z = gc_y + nf * i_ctr * j_ctr;     // :4973
  double *gc_1 = gc_z + nf * i_ctr * j_ctr;     // :4974
  ```
  **Block order: x, y, z, 1** (scalar last).
- Per contraction pair `(ic,jc)` each block advances by `nf` (`:4987-4990`).
- Within `a_bra_cart2spinor_si`: `nf = _len_cart[l]`, `nd = _len_spinor(kappa,l)`. The bra output `tmp1` is split alpha|beta: `gspaR=gspR`, `gspbR=gspR + nket*nd` (`:3926-3929`). Reads `v1=g1[j*nf+n]`, `vx=gx[j*nf+n]`, etc. (`:3950-3953`). CG coeff: `caR=coeffR[i*nf*2 + n]`, `cbR=coeffR[i*nf*2 + nf + n]` (`:3954-3957`).
- **Accumulation (the four equations, `:3958-3961`):**
  ```
  saR += caR*v1 + caI*vz - cbR*vy + cbI*vx
  saI +=-caI*v1 + caR*vz + cbI*vy + cbR*vx
  sbR += cbR*v1 - cbI*vz + caR*vy + caI*vx
  sbI +=-cbI*v1 - cbR*vz - caI*vy + caR*vx
  ```
**Hand-derived numeric check:** Pick l=1 (nf=3), kappa=−1 (GT, nd=4). Hand-fill `gc_1/gc_x/gc_y/gc_z` with 4 distinct cart vectors of length nf for a single ket column (nket=1). Pull `g_c2s[1].cart2j_gt_lR/I` from `c2spinor_coeffs.rs` (CJ_GT_L1_R/I). Compute the 4×(saR,saI,sbR,sbI) by hand using the equations above. Assert the Rust `apply_bra_si_block` output matches to 1e-14. **Cross-check by also running vendor `a_bra_cart2spinor_si` through a tiny FFI shim** (or compute the full `int1e_sp_spinor` and back out the bra contribution). Grep-checkable: the four sign patterns `+ ca_i * vz`, `- cb_r * vy`, `+ cb_i * vx` must appear in `apply_bra_si_block` (and the WRONG `- ca_i * vz` from `apply_si_block` must NOT).

### Spike Target B — bra-Pauli-mix / ket-ordinary split
**C source:** `libcint-master/src/cart2sph.c:4983-4984` (inside `c2s_si_1e`).
**Observed (exact, quoted):**
```c
a_bra_cart2spinor_si(tmp1R, tmp1I, gc_x, gc_y, gc_z, gc_1, nfj, i_kp, i_l);  // :4983  bra = Pauli-σ mix
a_ket_cart2spinor(tmp2R, tmp2I, tmp1R, tmp1I, di, j_kp, j_l);                // :4984  ket = ORDINARY
zcopy_ij(opij+ofj*jc+di*ic, tmp2R, tmp2I, ni, nj, di, dj);                   // :4985
```
- The ket call is `a_ket_cart2spinor` (`cart2sph.c:4343`), the SAME function `c2s_sf_1e` uses at `:4900`. **No `_si` suffix on the ket — the ket is NOT σ-mixed and NOT symmetrized.**
- `a_ket_cart2spinor` reads the bra-output `tmp1` over `nf2 = 2*nf` indices (alpha block n∈[0,nf), beta block n∈[nf,2nf)) with complex multiply `gspR += cR*gR - cI*gI; gspI += cI*gR + cR*gI` (`:4377-4378`).
**Hand-derived numeric check:** Confirm cintx `cart_to_spinor_si_2d` calls the EXISTING `apply_ket_transform` (`c2spinor.rs:766`) for stage 2 — grep that it does NOT introduce a `*_si` ket variant. Numeric: a p(kappa=−1)×p(kappa=−1) case; verify the ket transform output equals `apply_ket_transform` fed the si-bra `tmp1` (reuse the existing ket test harness). Grep-checkable: no symbol `apply_ket_si` / `a_ket_cart2spinor_si` exists in cintx or libcint (the latter genuinely has no such function).

### Spike Target C — device→host buffer hand-off (gout → gc blocks)
**C source:** `libcint-master/src/cint1e.c:148-159` (`CINT1e_loop`), `:157` (`CINTdmat_transpose`); `intor3.c:416-440` (gout emitter); `cint1e.c:269-271` (per-tensor c2s call).
**Observed (exact):**
- The gout emitter `CINTgout1e_int1e_sp` writes **component-interleaved**: `gout[n*4 + comp]` for comp∈{0,1,2,3}={x,y,z,1}, comp fastest (`intor3.c:431-434`).
- After primitive contraction (`PRIM2CTR0`), `CINT1e_loop` transposes: `CINTdmat_transpose(gctr, gctrj, nf*nc, n_comp)` where `n_comp = ncomp_e1 * ncomp_tensor` (`cint1e.c:54,157`). For `int1e_sp`: `ncomp_e1 = ng[5] = 4`, `ncomp_tensor = ng[7] = 1` → `n_comp = 4`. Transpose runs because `n_comp > 1`.
- Post-transpose layout is **component-blocked**: `gctr[comp*(nf*nc) + n]`, exactly the `gc_x|gc_y|gc_z|gc_1` slicing `c2s_si_1e` reads (`cart2sph.c:4971-4974`).
- `CINT1e_spinor_drv` then calls `c2s_si_1e` once per `ncomp_tensor` (=1) with `gctr` (`cint1e.c:269-271`).
- **ng vector for int1e_sp:** `{1, 0, 0, 0, 1, 4, 1, 1}` (`intor3.c:463`). `ng[5]=4` → 4 gc blocks; `ng[7]=1` → tensor rank 1. (Contrast `int1e_sigma`: `{0,0,0,0,0,4,1,3}` → ng[5]=4 gc blocks × ng[7]=3 tensor = 12-component gout. THIS is the "12-component Pauli gout" SC#2 refers to — it is the SIGMA family, not int1e_sp.)
**cintx mapping:** The σ·p `#[cube]` assembler emits `gout[n*4 + {x,y,z,1}]` (component-interleaved). The host must transpose to `gc[comp*(nf*ictr*jctr) + n]` (4 contiguous blocks) before `cart_to_spinor_si_2d`. cintx's existing `write_component_leading_staging` (`one_electron.rs:8300`) already produces component-leading blocks — follow that layout so the host transform reads `gc_x=block0, gc_y=block1, …`. The KET→BRA transpose (Target/Pitfall 4) is a SEPARATE, additional transpose owned inside `cart_to_spinor_si_2d`.
**Hand-derived numeric check:** Device-vs-host test: run the σ·p `#[cube]` assembler on a single primitive p-shell, capture the 4-block output, and compare against a host-side hand-roll of `s[0]=g1x·g0y·g0z` etc. for each cart n. Assert block boundaries land at `comp*(nf*ictr*jctr)`. Grep-checkable: the staging writer base offset must be `comp * (nf * ictr * jctr)`, not interleaved `n*4 + comp`, at the point the transform consumes it.

### Spike Target D — kappa≠0 GT/LT-only sizing through `spinor_len`
**C source:** `libcint-master/src/cart2sph.c:3537-3546` (`_len_spinor`); used in `c2s_si_1e` at `:4960-4961` (`di = _len_spinor(i_kp,i_l); dj = _len_spinor(j_kp,j_l)`); coeff-pointer dispatch in `a_bra_cart2spinor_si:3931-3937`.
**Observed (exact):**
```c
static FINT _len_spinor(FINT kappa, FINT l) {     // cart2sph.c:3537
    if (0 == kappa) return 4*l + 2;               // both blocks
    else if (kappa < 0) return 2*l + 2;           // GT  j=l+1/2
    else return 2*l;                              // LT  j=l-1/2
}
```
- kappa<0 → `coeffR = g_c2s[l].cart2j_gt_lR` (GT); else `cart2j_lt_lR` (LT) (`:3931-3937`). For kappa≠0 ONLY ONE block is computed (no over-read into the other table).
- cintx `spinor_len` (`c2spinor.rs:25-33`) is byte-identical.
- D-05 primary fixture: p shell kappa=+1 → LT, `di = 2*1 = 2`; d shell kappa=−1 → GT, `dj = 2*2+2 = 6`. Block is 2×6 (non-square), nctr=2 on the p shell. This is the FIRST cintx fixture to exercise kappa≠0 (Phase 27's was kappa=0 only).
**Hand-derived numeric check:** Assert `spinor_len(1,+1)==2`, `spinor_len(2,-1)==6`. Verify the staging buffer sizing is `di*dj*2 = 2*6*2 = 24` f64, NOT `(4*1+2)*(4*2+2)*2`. Run the kappa fixture through vendor `int1e_sp_spinor` and compare flat buffers @ atol=1e-12. Grep-checkable: every buffer alloc in `cart_to_spinor_si_2d` uses `spinor_len(...)`, with zero literal `4 * l + 2` / `4*l+2` in the si_2d path.

### Spike Target E — `int1e_sp` gout component ordering + driver wiring
**C source:** `libcint-master/src/autocode/intor3.c:416-468` (`CINTgout1e_int1e_sp`, `int1e_sp_spinor`).
**Observed (exact):**
- Driver: `int1e_sp_spinor` sets `ng[]={1,0,0,0,1,4,1,1}`, `envs.f_gout = &CINTgout1e_int1e_sp`, returns `CINT1e_spinor_drv(out, dims, &envs, cache, &c2s_si_1e, 0)` (`intor3.c:461-468`). The `0` is `int1e_type=0` → overlap base (`CINTg1e_ovlp`, `cint1e.c:289-291`).
- Gout (`intor3.c:416-440`):
  ```c
  G1E_D_I(g1, g0, envs->i_l+0, envs->j_l, 0);   // g1 = nabla_i(g0), :422
  for (n=0; n<nf; n++) {
    s[0] = g1[ix]*g0[iy]*g0[iz];                // x-component, :427
    s[1] = g0[ix]*g1[iy]*g0[iz];                // y-component, :428
    s[2] = g0[ix]*g0[iy]*g1[iz];                // z-component, :429
    gout[n*4+0] = s[0];  // → gc_x              // :431-434
    gout[n*4+1] = s[1];  // → gc_y
    gout[n*4+2] = s[2];  // → gc_z
    gout[n*4+3] = 0;     // → gc_1 (scalar, ZERO for int1e_sp)
  }
  ```
- **12→4 composition note (SC#2 reconciliation):** `int1e_sp` is a **4-component** gout (3 Pauli + 1 zero scalar). The "12-component Pauli gout emitter" in SC#2 is `int1e_sigma` (`intor3.c:18-54`, `gout[n*12+...]`, ng[7]=3 tensor × 4 gc), where the scalar slot carries `−s[0]` per tensor component and the off-diagonals are zero. The generic σ·p assembler (D-03) should be parameterized by (tensor_rank, which-slot-is-nonzero) so int1e_sp (rank 1, Pauli in x/y/z) and int1e_sigma (rank 3) both emit through it. `g1 = G1E_D_I = CINTnabla1i_1e` (`g1e.c:322`): `fx[ptr+i] = i*gx[ptr+i-1] + (-2*ai)*gx[ptr+i+1]` (`:345`).
**Hand-derived numeric check:** For a primitive s×s pair (l=0, nf=1, ai given), `g0[0]=ovlp`, `g1[0] = -2*ai*g0[1]`. The 4-component gout is `[g1x·g0y·g0z, g0x·g1y·g0z, g0x·g0y·g1z, 0]`. Hand-evaluate from the analytic primitive overlap recurrence; assert the device assembler matches to 1e-14. Then assert `gout[n*4+3]==0` for every n (int1e_sp scalar slot). Grep-checkable: the assembler writes exactly 4 components with slot 3 = 0.0 for the sp path.

### Sampling Rate
- **Per task commit:** `cargo test -p cintx-cubecl --lib transform::c2spinor` (host transform unit tests, fast).
- **Per wave merge:** `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test si_transform_parity` (vendor parity, the FND-05 gate).
- **Phase gate:** full workspace + vendor suite green; `cargo run -p xtask -- oracle-covered-update` confirms `int1e_sp_spinor` stays `oracle_covered=false` (SC#4); manifest-audit green.

### Wave 0 Gaps
- [ ] `crates/cintx-cubecl/src/transform/c2spinor.rs` — `apply_bra_si_block` (new si bra step) + unit test against `a_bra_cart2spinor_si` hand values — covers FND-05/Target A.
- [ ] `crates/cintx-cubecl/src/kernels/sigma_p.rs` (NEW) — generic σ·p `#[cube]` assembler + device-vs-host test — covers FND-05/Target C,E.
- [ ] `crates/cintx-oracle/src/fixtures.rs` — `build_kappa_spinor_fixture` (p kappa=+1, d kappa=−1, nctr=2, non-square) + heavy-atom fixture — covers Target D.
- [ ] `crates/cintx-oracle/src/vendor_ffi.rs` — `vendor_int1e_sp_spinor` FFI binding.
- [ ] `crates/cintx-oracle/tests/si_transform_parity.rs` (NEW) — transform/component byte-identity test + no-silent-skip assertion (Phase 27 D-10) — covers FND-05 D-01 proof.
- [ ] `crates/cintx-ops/generated/compiled_manifest.lock.json` — `int1e_sp_spinor` row (`oracle_covered=false`, stability stable, correct `component_rank`); re-grep hardcoded OperatorId consts (Pitfall 6).

## Security Domain

> `security_enforcement` is enabled (absent = enabled). This is an internal numerical library with no auth/session/network surface.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | n/a (library, no auth) |
| V3 Session Management | no | n/a |
| V4 Access Control | no | n/a |
| V5 Input Validation | yes | Buffer-size validation: `cart_to_spinor_si_2d` MUST return `cintxRsError::BufferTooSmall`/`ChunkPlanFailed` on undersized staging/cart (mirror `sf_2d` `c2spinor.rs:544-559`). No partial writes on failure (CLAUDE.md OOM-safe stop contract). |
| V6 Cryptography | no | n/a |

### Known Threat Patterns for Rust numerical / FFI stack
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds slice on undersized spinor buffer (kappa-mis-sized) | Tampering / DoS | `spinor_len`-derived sizing + explicit length checks returning typed `thiserror` errors; never `unsafe` indexing in the host transform. |
| Vendor FFI UB (libcint writes past `out`) | Tampering | Size `out` from `CINTcgto_spinor`-equivalent (`spinor_len`×nctr); the parity harness allocates the full vendor-expected size. |
| False verification claim (flipping `oracle_covered` on a skipped fixture, T-21-08-02) | Repudiation | SC#4 guard at `oracle_covered_update.rs:50` (`if fixture.skipped { continue; }`) + Phase 28 keeps every σ family `UnsupportedApi`. |
| Silent parity skip (double-gate env unset → determinism-only) | Repudiation | Phase 27 D-10 no-silent-skip assertion: the parity test FAILS (not skips) when `--features cpu`+`CINTX_ORACLE_BUILD_VENDOR=1` are expected but the vendor arm did not run. |

## Project Constraints (from CLAUDE.md)
- **CubeCL is the primary compute backend** — σ·p assembler is a `#[cube]` device kernel (D-04). Host work limited to planning/validation/marshaling/test glue → the si_2d transform is host (correct per D-04).
- **`thiserror` v2 for public library errors**, `anyhow` for CLI/xtask/oracle/bench glue — the transform returns `cintxRsError` (thiserror); the oracle harness uses anyhow.
- **libcint 6.1.3 result compatibility** — byte-identity at atol=1e-12 against vendored libcint is the gate.
- **No partial writes on allocation failure** — fallible alloc + typed failure (OOM-safe stop). The si_2d transform must validate buffer sizes before any write.
- **`cargo --locked` + pinned toolchain** — no new deps; pure in-repo code.
- **No public APIs expose backend-specific runtime types** — the σ·p assembler stays behind the kernel launcher; the transform takes plain slices.
- **New-family surface policy** (project memory `feature_new_family_surface_scope.md`): manifest + RawApiId + kernel + vendor-FFI + oracle ONLY. NO capi enum variants, NO legacy `cint*` wrappers.
- **Vendor parity double-gated** (memory `reference_oracle_vendor_parity_invocation.md`): `--features cpu` AND `CINTX_ORACLE_BUILD_VENDOR=1`; without both, parity silently skips → add no-silent-skip assertion.
- **CubeCL authoring rules** (memory `reference_cubecl_authoring_manuals.md`, `docs/manual/Cubecl/*.md`): no plain-fn calls, no if-expr, `F::exp`/`F::sqrt`, u32/i32 only, no continue/break — read before writing the σ·p `#[cube]` kernel.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The host transform can read the σ·p assembler's 4 blocks as `gc[comp*(nf*ictr*jctr)+n]` after a host transpose, matching `CINTdmat_transpose` semantics — i.e. cintx will replicate libcint's interleaved→blocked transpose either on-device (pre-blocked write) or host-side. | Target C / Arch Map | If cintx already emits component-leading blocks from the device (per `write_component_leading_staging`), no extra transpose is needed — the assembler just writes to `comp*block + n` directly. Planner should confirm which layout the σ·p `#[cube]` emits; both reach the same gc layout, but the transpose location differs. LOW risk (both work; affects only where the transpose lives). |
| A2 | `int1e_spsp` uses `c2s_sf_1e` (not `c2s_si_1e`) as its c2s in libcint 6.1.3, so the Phase-29 spsp ket-side σ·p is folded into the gout, not the transform. | Open Questions | Verified at `intor3.c:411` (`&c2s_sf_1e`). If the planner assumes spsp also uses si_2d, the Phase-29 reuse story changes. MEDIUM risk for Phase 29 planning (not Phase 28). |
| A3 | The heavy-atom realism fixture (D-05 secondary) can be built with a small synthetic Dirac-style 2c basis without needing an external basis-set file; exact element/exponents are Claude's discretion. | Fixtures | If a real published basis is required for "realism", more sourcing work. LOW risk (D-05 says "small real heavy-atom case"; a representative single-atom spinor basis suffices as a blind-spot cross-check). |

## Open Questions

1. **Where does cintx perform the gout-interleaved → gc-blocked transpose?**
   - What we know: libcint does it in `CINT1e_loop` via `CINTdmat_transpose` (`cint1e.c:157`); cintx has `write_component_leading_staging` that produces component-leading blocks directly.
   - What's unclear: whether the σ·p `#[cube]` assembler should write pre-blocked (`comp*block+n`) on-device, or emit interleaved + transpose host-side.
   - Recommendation: emit pre-blocked on-device (follow `write_component_leading_staging` layout); avoids a host transpose and matches the existing component-leading staging convention. Confirm during plan task 1.

2. **`int1e_spsp` c2s (relevant only for Phase 29 reuse design).**
   - What we know: `int1e_spsp_spinor` uses `&c2s_sf_1e` (`intor3.c:411`), NOT `c2s_si_1e` — the ket-side σ·p is baked into the gout (`G2E_D_J`/`G2E_D_I` in `CINTgout1e_int1e_spnucsp:480-482`), and the transform is the ordinary sf.
   - What's unclear: nothing for Phase 28; flagged so Phase 29 reuse of the σ·p module accounts for spsp using sf_2d + a richer gout.
   - Recommendation: design the generic σ·p assembler's gc-packing to be transform-agnostic (it just produces gc blocks; whether c2s is si_2d or sf_2d is the family's choice).

3. **`component_rank` for the `int1e_sp_spinor` manifest row.**
   - What we know: ng[7]=1 (tensor rank 1) for int1e_sp; the 4 gc blocks are `ncomp_e1`, not tensor components.
   - What's unclear: whether the manifest `component_rank` should be 1 (tensor) or 4 (gc blocks) — Pitfall: component_rank truncation (memory `project_unstable_derivative_ports.md`).
   - Recommendation: verify against an existing spinor family's row convention in the lock before stamping; the transform output is `di*dj` complex (rank 1 in the spinor output sense). Resolve in the manifest task.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain | all | ✓ | pinned 1.94.0 (`rust-toolchain.toml`) | — |
| `cubecl` | σ·p `#[cube]` assembler | ✓ | pinned 0.10.0 | CpuRuntime (test path) |
| vendored libcint 6.1.3 | vendor parity oracle | ✓ | `libcint-master/` in repo | — |
| `cc` (build libcint) | vendor FFI | ✓ (existing oracle build) | 1.2.x | parity skips → must NOT silently pass (D-10) |
| `CINTX_ORACLE_BUILD_VENDOR=1` + `--features cpu` | vendor byte-identity | conditional (CI/local env) | — | determinism-only run; add no-silent-skip assertion |

**Missing dependencies with no fallback:** None.
**Missing dependencies with fallback:** Vendor parity requires the double-gate; without it the test must FAIL (not skip) per Phase 27 D-10.

## Sources

### Primary (HIGH confidence — vendored C, byte-authoritative)
- `libcint-master/src/cart2sph.c` — `c2s_si_1e` (:4947), `a_bra_cart2spinor_si` (:3920), `a_ket_cart2spinor` (:4343), `a_iket_cart2spinor` (:4405), `zcopy_ij` (:4499), `_len_spinor` (:3537), `CINTc2s_ket_spinor_si1` (:6839), `c2s_sf_1e` (:4869)
- `libcint-master/src/autocode/intor3.c` — `CINTgout1e_int1e_sp` (:416), `int1e_sp_spinor` (:461), `int1e_sigma` gout (:18), `int1e_spsp_spinor` (:405)
- `libcint-master/src/cint1e.c` — `CINT1e_loop` + `CINTdmat_transpose` (:40-160), `CINT1e_spinor_drv` (:239), `make_g1e_gout` (:284)
- `libcint-master/src/g1e.c` — `CINTnabla1i_1e` (:322), `CINTnabla1j_1e` (:352)

### Primary (HIGH confidence — existing cintx code)
- `crates/cintx-cubecl/src/transform/c2spinor.rs` — `spinor_len` (:25), `apply_si_block` (:124), `cart_to_spinor_si` (:392), `cart_to_spinor_sf_2d` (:531), `apply_bra_block` (:668), `apply_ket_transform` (:766), KET→BRA transpose pattern (:1342)
- `crates/cintx-cubecl/src/kernels/one_electron.rs` — sf_2d dispatch (:24), nabla helpers (:1828/:1864), `write_component_leading_staging` (:8300)
- `crates/cintx-oracle/src/fixtures.rs` — `build_adversarial_spinor_fixture` (:209)
- `xtask/src/oracle_covered_update.rs` — skipped-fixture guard (:50)

### Secondary (project memory / skill — verified knowledge)
- `.claude/skills/spike-findings-cintx/SKILL.md` — spinor interleaved-complex layout, non-square-block requirement (D-07)
- Project memory: `project_cubecl_cpuruntime_fp_env_side_effect.md`, `project_cubecl_cpuruntime_fma_fused.md`, `project_operator_id_shift_breaks_hardcoded_test_consts.md`, `project_raw_nctr_coeff_transpose.md`, `feature_new_family_surface_scope.md`, `reference_oracle_vendor_parity_invocation.md`

## Metadata

**Confidence breakdown:**
- Standard stack (reuse map): HIGH — every reused symbol read directly in cintx source.
- Spike Targets A–E: HIGH — all verified against vendored C with exact file:line and quoted code.
- Sign-discrepancy finding (A): HIGH — `a_bra_cart2spinor_si:3958` vs `CINTc2s_ket_spinor_si1:6883` vs `apply_si_block:161` compared term-by-term.
- Device→host hand-off (C): HIGH — `CINTdmat_transpose` + ng vector confirmed; A1 notes only the (low-risk) choice of WHERE cintx transposes.
- Fixtures / heavy-atom realism: MEDIUM — D-05 constraints clear; exact element/basis is Claude's discretion (A3).

**Research date:** 2026-05-31
**Valid until:** Stable — vendored libcint 6.1.3 is pinned in-repo; findings do not expire while the vendor source and `c2spinor.rs` structure hold (~indefinite for the C ground truth; re-verify cintx line numbers if `c2spinor.rs` is refactored).
