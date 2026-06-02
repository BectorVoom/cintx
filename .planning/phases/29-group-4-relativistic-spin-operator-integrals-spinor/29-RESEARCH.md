# Phase 29: Group 4 — Relativistic Spin-Operator Integrals (spinor) - Research

**Researched:** 2026-05-31
**Domain:** Relativistic σ-operator spinor integrals; cart→spinor `c2s_si` transform layout; libcint 6.1.3 byte-identity transcription
**Confidence:** HIGH (all transform/gout/driver facts read directly from vendored libcint 6.1.3 source and current cintx code)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01 (FULL 2e si suite):** Phase 29 builds the complete 2e si-transform foundation: `c2s_si_2e1` + `c2s_si_2e2` (real), the imaginary `c2s_si_2e1i` + `c2s_si_2e2i`, AND the `c2s_sf_2e1`/`c2s_sf_2e2` partner transforms for the non-σ electron. Corrects the Phase-28 record that attributed these to Phases 30/31 — they are a Phase-29 deliverable; Phases 30/31 only reuse them.
- **D-02 (NEW 4-shell kappa 2e fixture):** Add `build_kappa_spinor_2e_fixture` to `fixtures.rs`: 4 spinor shells (2-electron config), non-square, genuine kappa≠0 GT/LT mix (stresses `2l`/`2l+2` sizing, not just `4l+2`), ≥1 shell with nctr>1. Small real heavy-atom case is a secondary realism cross-check, not the primary gate.
- **D-03 (NO separate spike; transcribe-from-libcint + vendor byte-identity gate):** No separate hard-gate design spike. All transforms and σ gouts transcribed directly from libcint, proven by the atol=1e-12 vendor byte-identity gate. Accepted risk: the `c2s_si_2e1/2e2` layout is genuinely new in cintx; a wrong layout/stride/sign surfaces only at the vendor gate. Structural mitigation (Claude's discretion): make a transform-level byte-identity micro-test the FIRST task of the 2e wave, before any 2e family wires onto it.
- **D-04 (sequential de-risk, 3 waves):** Wave 1 — 1e σ (register REL-01/02 + flip `int1e_sp`, on the existing Phase-28 `si_2d` + σ·p foundation). Wave 2 — 2e foundation (full 2e si suite + `build_kappa_spinor_2e_fixture` + 2e transform micro-test). Wave 3 — 2e families (register REL-03/04). Each wave gated (vendor parity green) before the next.

### Claude's Discretion
- Internal module naming/factoring for the new 2e si transforms and where they live in `c2spinor.rs`.
- Exact per-family gout component ordering for the families — resolve from `autocode/intor3.c`/`intor4.c` (RESOLVED below).
- Exact molecule/element + kappa assignments for `build_kappa_spinor_2e_fixture` (subject to D-02 hard constraints) and the heavy-atom cross-check.
- Precise plan boundaries inside each wave (e.g. Wave 1 may split 1e-σ-with-p vs `int1e_sigma` pure-σ).
- Whether the 2e transform micro-test compares to vendored `c2s_si_2e*` directly or via a thin driving family.

### Deferred Ideas (OUT OF SCOPE)
- **GIAO×σ slice (Phase 30):** `int1e_spg*`, `int1e_spgnucsp`, `*_sa10*`, 2e `int2e_cg_sa10*`/`giao_sa10*`. (NOTE: research found these GIAO×σ 2e drivers at intor4.c L636/899/990/1249 — they REUSE the Phase-29 transforms but are NOT Phase-29 deliverables.)
- **Gauge / Breit–Gaunt 2e (Phase 31):** `int2e_gauge_r1/r2_*`, Gaunt `ssp/sps`.
- **PARITY-01 full-parity gate (Phase 31).**
- **Reviewed Todos not folded:** `oracle-cart-offset-vendor-zero`, `rys-nroots-ge6-wheeler-fallback`.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description (REQUIREMENTS.md L109-112) | Research Support |
|----|---------------------------------------|------------------|
| REL-01 | `int1e_spsp`, `int1e_spnucsp`, `int1e_sprinvsp` match vendored libcint at atol=1e-12 (spinor) via the FND-05 `c2s_si` path | Transform map below: `spsp`→`c2s_sf_1e`, `spnucsp`/`sprinvsp`→`c2s_si_1e`. gout orderings + component_rank in §"Per-Family gout & Transform Map". All wire onto Phase-28 `cart_to_spinor_si_2d`/`sf_2d` + `sigma_p.rs`. |
| REL-02 | `int1e_srsr`, `int1e_sr`/`srnucsr`, `int1e_sigma`, `int1e_sp` match at atol=1e-12 (spinor) | `sp`/`srsr`/`srnucsr`→`c2s_si_1e`; `sr`/`sigma`→`c2s_si_1ei` (imaginary). `int1e_sp` flips from oracle_covered=false. gouts transcribed below. |
| REL-03 | `int2e_spsp1`, `int2e_srsr1` (+`spsp1spsp2`/`srsr1srsr2`) match at atol=1e-12 (spinor) | Drivers in `intor4.c` (ALREADY in oracle build): `spsp1`/`srsr1`→`c2s_si_2e1`+`c2s_sf_2e2`; `spsp1spsp2`/`srsr1srsr2`→`c2s_si_2e1`+`c2s_si_2e2`. Needs full 2e si suite (Wave 2). |
| REL-04 | `int2e_ssp1ssp2`, `int2e_sps1sps2`, `int2e_vsp1*`, `int2e_spv1*` match at atol=1e-12 (spinor) | **LANDMINE: ssp/sps drivers live in `gaunt1.c`, vsp/spv in `dkb.c` — NEITHER is in the oracle build.rs.** A build.rs change IS required (contradicts CONTEXT.md). `ssp/sps`→`c2s_si_2e1i`+`c2s_si_2e2i` (both imaginary); `vsp1`/`spv1`→`c2s_si_2e1`+`c2s_sf_2e2`; `*spv2`/`*vsp2` 2-sided→`c2s_si_2e1`+`c2s_si_2e2`. |
</phase_requirements>

## Summary

Phase 29 closes Group 4 in three sequential waves on top of the Phase-28 Gap-B2 σ foundation. The **single most important research finding** is that the per-family transform choice is NOT uniform: each family selects one of `c2s_si_1e` (real bra-σ-mix), `c2s_si_1ei` (imaginary-ket bra-σ-mix), or even `c2s_sf_1e` (plain spin-free) — read directly from each driver's `CINT1e_spinor_drv(..., &c2s_*, ...)` line in `intor3.c`. The naive assumption "all σ families use `c2s_si`" is **wrong**: `int1e_spsp_spinor` uses `c2s_sf_1e` (intor3.c:411), and `int1e_sigma`/`sr`/`srsp` use the IMAGINARY `c2s_si_1ei` (intor3.c:81/285/351). cintx already has the single-block `iket_si` and the 1e `si_2d` real transform; the 1e `si_2di` (imaginary-ket) variant is the one structurally-new 1e piece Wave 1 must add.

The 2e wave (Waves 2–3) is the larger lift. cintx currently routes ALL 2e spinor through `UnsupportedApi` (two_electron.rs) — there is no `cart_to_spinor_si_4d` analog at all. Wave 2 builds the full 2e si/sf transform suite as the host structural analogs of libcint's `c2s_si_2e1/2e2(+i)` and `c2s_sf_2e1/2e2`, exactly mirroring the existing `cart_to_spinor_sf_4d` two-stage (electron-1 bra/ket then electron-2 bra/ket + `zcopy_iklj` reorder) structure. The 2e electron-2 bra step uses a DIFFERENT helper (`a_bra1_cart2spinor_zi`/`_zf`) with a per-grid layout and an explicit 2×2 Pauli σ-matrix expansion — this is the genuinely novel layout the D-03 micro-test must pin first.

**Primary recommendation:** Build the per-family transform-selection map (§ below) as the authoritative wiring table; transcribe each 1e gout from `intor3.c` and each 2e gout from `intor4.c`/`gaunt1.c`/`dkb.c` verbatim; add `gaunt1.c` + `dkb.c` to the oracle `build.rs` `.file()` list (REQUIRED for REL-04, NOT covered by the existing `intor4.c` wiring); make the 2e transform micro-test against vendored `int2e_spsp1_spinor` (the thinnest `c2s_si_2e1`+`c2s_sf_2e2` family) the gating first task of Wave 2.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| σ·p / σ·r G-tensor gout assembly (the 4 `gc_*` blocks) | Device `#[cube]` (`kernels/sigma_p.rs`, `two_electron.rs`) | — | Phase-28 D-03/D-04: gout/nabla is device; reuse the generic σ·p assembler for 1e, build the 2e analog on the same pattern. |
| 1e cart→spinor si/sf 2D transform | Host (`transform/c2spinor.rs`) | — | Phase-28 D-04: transforms run on the contracted cart staging post-kernel. `cart_to_spinor_si_2d` already exists; add `si_2di` (imaginary ket). |
| 2e cart→spinor si/sf transform suite (`_2e1/2e2(+i)`) | Host (`transform/c2spinor.rs`) | — | New; structural analog of host `cart_to_spinor_sf_4d`. The si bra step + the electron-2 `zi`/`zf` bra step + `zcopy_iklj` reorder all run host-side. |
| KET→BRA transpose | Host, inside the transform | — | Phase-27 D-06 / Phase-28: device cart blocks are KET-major; own the transpose inside the transform so no launcher can omit it. |
| Manifest registration + oracle_covered flip | Host (`compiled_manifest.lock.json`, `xtask`) | — | New-family surface = manifest + RawApiId + kernel + vendor-FFI + oracle ONLY. |
| Vendor byte-identity reference | Host oracle (`cintx-oracle`, vendored libcint via FFI) | — | Double-gated `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`. |

## Standard Stack

This is internal cintx/libcint work — no new external dependencies. Confirmed against CLAUDE.md and `Cargo.toml`:

### Core
| Component | Version | Purpose | Why Standard |
|-----------|---------|---------|--------------|
| `cubecl` | 0.10.0 (pinned) | Device `#[cube]` σ·p/σ·r gout assembler (1e + 2e) | CLAUDE.md: CubeCL is the primary compute backend. [VERIFIED: Cargo.toml / CLAUDE.md] |
| `thiserror` | 2.0.18 | Library error surface (`cintxRsError::BufferTooSmall`, `ChunkPlanFailed`, `UnsupportedApi`) | CLAUDE.md: public lib errors use thiserror v2. [VERIFIED: CLAUDE.md] |
| `anyhow` | 1.0.102 | Oracle harness / xtask / test glue errors | CLAUDE.md: app-boundary tooling uses anyhow. [VERIFIED: CLAUDE.md] |
| `num-complex` | workspace | Interleaved complex spinor output | Phase-12 staging convention. [ASSUMED — not re-verified this session; carry-over] |
| vendored libcint | 6.1.3 | Byte-identity oracle reference | CLAUDE.md compatibility target. [VERIFIED: build.rs cint.h `@cint_VERSION@`→"6.1.3"] |

### Supporting
| Component | Purpose | When to Use |
|-----------|---------|-------------|
| `cintx-oracle/build.rs` | Compile vendored libcint C sources behind `CINTX_ORACLE_BUILD_VENDOR=1` | MUST add `src/autocode/gaunt1.c` + `src/autocode/dkb.c` for REL-04. |
| `bindgen` (probe) + suppl-header | Generate/declare vendor FFI symbols | All Group-4 symbols are ALREADY in `cint_funcs.h` → NO suppl-header extern decls needed; only `.file()` additions + the `vendor_*` Rust wrappers. |

**Version verification:** Versions taken from CLAUDE.md (authoritative project guidance, dated 2026-05-09) and `Cargo.toml`. No registry re-query needed — this phase adds no new crates.

## Architecture Patterns

### System Data Flow

```
                  raw atm/bas/env (KAPPA_OF set on spinor shells)
                              │
                              ▼
                   ┌──────────────────────┐
                   │  kernel launcher      │  one_electron.rs (1e) / two_electron.rs (2e)
                   │  detects Representation::Spinor → routes to σ path
                   └──────────┬───────────┘
                              ▼
        ┌─────────────────────────────────────────────┐
        │  DEVICE #[cube] σ·p / σ·r gout assembler      │  sigma_p.rs (1e), new 2e analog
        │  emits 4 contiguous cart blocks per electron: │
        │    gc_x | gc_y | gc_z | gc_1                  │  (KET-major, contraction-blocked)
        └─────────────────────┬─────────────────────────┘
                              ▼  device → host hand-off (contracted cart staging)
        ┌─────────────────────────────────────────────┐
        │  HOST cart→spinor transform (c2spinor.rs)     │
        │  per-family selects ONE of:                   │
        │   1e:  c2s_si_1e | c2s_si_1ei | c2s_sf_1e     │
        │   2e:  (c2s_si_2e1|c2s_si_2e1i|c2s_sf_2e1)    │  electron 1
        │        × (c2s_si_2e2|c2s_si_2e2i|c2s_sf_2e2)  │  electron 2
        │  owns KET→BRA transpose; sizing from spinor_len│
        └─────────────────────┬─────────────────────────┘
                              ▼
              interleaved-complex spinor buffer
              out[comp*(ni_sp*nj_sp)*2 + (j*ni_sp+i)*2 + {re,im}]
                              │
                              ▼  (vendor gate, atol=1e-12)
              vendored libcint int{1e,2e}_*_spinor  (byte-identity)
```

### Pattern 1: Per-family transform selection (the authoritative wiring)
**What:** Every spinor driver chooses its bra/ket transform explicitly via its `CINT*_spinor_drv(..., &c2s_X, ...)` call. There is no uniform rule.
**When to use:** Building the manifest row + launcher arm for each family.
**Source:** `libcint-master/src/autocode/intor3.c` (1e), `intor4.c`/`gaunt1.c`/`dkb.c` (2e), read this session.

### Pattern 2: 2e two-stage host transform (analog of `cart_to_spinor_sf_4d`)
**What:** electron-1 transform (bra σ-mix + ordinary ket) producing `opij` ordered `<ik|lj>`, then electron-2 transform (`a_bra1_*` + `a_ket1_*`) + `zcopy_iklj` reorder into `fijkl`.
**Source:**
```c
// c2s_si_2e1 (cart2sph.c:5592) — electron 1, real
a_bra_cart2spinor_si(tmp1R, tmp1I, gc_x, gc_y, gc_z, gc_1, d_j, i_kp, i_l);
a_ket_cart2spinor(opij, opij+no, tmp1R, tmp1I, d_i, j_kp, j_l);
//   gc_x/gc_y/gc_z/gc_1 are 4 contiguous blocks each nf*i_ctr*j_ctr*k_ctr*l_ctr long;
//   loop over i_ctr*j_ctr*k_ctr*l_ctr, advancing each gc by nf and opij by no*OF_CMPLX.

// c2s_si_2e2 (cart2sph.c:5687) — electron 2, real
a_bra1_cart2spinor_zi(tmp1R, tmp1I, ox, oy, oz, o1, di, nfl*dj, k_kp, k_l);
a_ket1_cart2spinor(tmp2R, tmp2I, tmp1R, tmp1I, di*dk, dj, l_kp, l_l);
zcopy_iklj(pfijkl, tmp2R, tmp2I, ni, nj, nk, nl, di, dj, dk, dl);
//   ox/oy/oz/o1 are 4 contiguous blocks each nop*OF_CMPLX*i_ctr*j_ctr*k_ctr*l_ctr long.
```

### Pattern 3: The electron-2 σ-mix (`a_bra1_cart2spinor_zi`) — the genuinely novel layout
**What:** electron 2's σ block applies the 2×2 Pauli matrix `[[1+iz, y+ix],[-y+ix, 1-iz]]` to the four complex `gx/gy/gz/g1` blocks (each split R/I across `nket*nf*ngrids`), accumulating into α/β half-blocks. This differs from the 1e `a_bra_cart2spinor_si` (which folds real cart blocks). The D-03 micro-test must pin THIS.
**Source:** `cart2sph.c:4118-4186` (read this session):
```c
// v11 = 1+iz, v12 = y+ix, v21 = -y+ix, v22 = 1-iz  (the σ·n 2×2 expansion)
v11R = g1R - gzI;  v11I = g1I + gzR;
v12R = gyR - gxI;  v12I = gyI + gxR;
v21R = -gyR - gxI; v21I = -gyI + gxR;
v22R = g1R + gzI;  v22I = g1I - gzR;
gspaR += caR*v11R + caI*v11I + cbR*v21R + cbI*v21I;  // and α-I, β-R, β-I analogously
```
`a_bra1_cart2spinor_zf` (cart2sph.c:4188, the sf partner used by `c2s_sf_2e2`) is the spin-free version: only `g1R/g1I` (the scalar block), no σ mix.

### Anti-Patterns to Avoid
- **Assuming a uniform `c2s_si` for all σ families** — `spsp` uses `c2s_sf_1e`; `sigma`/`sr` use `c2s_si_1ei`. Read each driver line.
- **Assuming `intor4.c` covers REL-04** — ssp/sps are in `gaunt1.c`, vsp/spv in `dkb.c`; both must be added to build.rs.
- **Hardcoding `4l+2` spinor sizing** — kappa≠0 needs `2l` (LT) / `2l+2` (GT) from `spinor_len`.
- **Testing on a square spinor block** — orientation/transpose bugs hide; use non-square (D-02).
- **Re-adding a `if dst < staging.len()` per-element guard inside the monolithic block writer** — Phase-25 lesson: family kernels are whole-block writers; per-chunk staging must be FULL-block sized.

## Per-Family gout & Transform Map (RESOLVES CONTEXT.md research items 1, 2, 3, 4)

### 1e families (REL-01/02) — `intor3.c`, transcribed this session

| Family | gout rank (`gout[n*R+..]`) | gout components | Transform (`CINT1e_spinor_drv` arg) | `ng[]` | component_rank for lock |
|--------|---------------------------|------------------|-------------------------------------|--------|--------------------------|
| `int1e_sp` (L416) | 4 | `+s0, +s1, +s2, 0` (∇ on bra: x,y,z,scalar=0) | `c2s_si_1e` (L467) | `{1,0,0,0,1,4,1,1}` | **1** (existing lock row) |
| `int1e_spsp` (L356) | 1 | `+s0+s4+s8` (scalar: ∇²) | **`c2s_sf_1e`** (L411) | `{1,1,0,0,2,1,1,1}` | **1** |
| `int1e_spnucsp` (L472) | 4 | `+s5-s7, +s6-s2, +s1-s3, +s0+s4+s8` | `c2s_si_1e` (L537, nuc=2 centers) | `{1,1,0,0,2,4,0,1}` | **1** |
| `int1e_sprinvsp` (L542) | 4 | same ordering as spnucsp | `c2s_si_1e` (L607, rinv=1 center) | `{1,1,0,0,2,4,0,1}` | **1** |
| `int1e_srsr` (L168) | 4 | `+s5-s7, +s6-s2, +s1-s3, +s0+s4+s8` | `c2s_si_1e` (L229) | `{1,1,0,0,2,4,1,1}` | **1** |
| `int1e_srnucsr` (L612) | 4 | same ordering as srsr | `c2s_si_1e` (L677) | `{1,1,0,0,2,4,0,1}` | **1** |
| `int1e_sr` (L234) | 4 | `-s0, -s1, -s2, 0` (σ·r on bra) | **`c2s_si_1ei`** (L285) | `{1,0,0,0,1,4,1,1}` | **1** |
| `int1e_sigma` (L18) | 12 (`gout[n*12+..]`) | `[-s,0,0,0, 0,-s,0,0, 0,0,-s,0]` (3 Pauli diag blocks ×4) | **`c2s_si_1ei`** (L81) | `{0,0,0,0,0,4,1,3}` → ng[7]=3 | **needs verify: likely 1** (the c2s_si transform expands the 3 σ-components into the spinor coupling; the rank-3 ng[7] is the σ-component count folded by the transform, NOT a free output component axis) |

> **gout component meaning:** the `c2s_si_1e`/`c2s_si_1ei` transform consumes the 4 cart blocks `gc_x,gc_y,gc_z,gc_1` (the σ_x,σ_y,σ_z,scalar G-tensor) in that order. So the 4-rank gouts above ARE the (x,y,z,1) blocks the existing `cart_to_spinor_si_2d(gc_x,gc_y,gc_z,gc_1,...)` consumes. `int1e_spsp` is rank-1 because it routes through `c2s_sf_1e` (scalar only — the two σ·p contract to ∇²·scalar). [VERIFIED: intor3.c read this session]

> **`c2s_si_1e` (real) vs `c2s_si_1ei` (imaginary ket):** `c2s_si_1e` uses `a_ket_cart2spinor` (ordinary); `c2s_si_1ei` uses `a_iket_cart2spinor` (the imaginary-ket variant — cintx already has the single-block `cart_to_spinor_iket_si` at c2spinor.rs:457). Wave 1's only structurally-new 1e transform is the 2D `si_2di` (imaginary-ket bra-σ-mix) for `sr`/`sigma`. [VERIFIED: cart2sph.c c2s_si_1e/1ei + cintx c2spinor.rs]

### 2e families — driver transform pairings (electron 1 × electron 2)

| Family | Source file | Transform pairing | component_rank | In oracle build.rs? |
|--------|-------------|-------------------|----------------|----------------------|
| `int2e_spsp1` | intor4.c:85 | `c2s_si_2e1` + `c2s_sf_2e2` | gout rank 4 → lock **1** | ✅ intor4.c wired |
| `int2e_srsr1` | intor4.c:349 | `c2s_si_2e1` + `c2s_sf_2e2` | gout rank 4 → **1** | ✅ |
| `int2e_spsp1spsp2` | intor4.c:277 | `c2s_si_2e1` + `c2s_si_2e2` | **1** | ✅ |
| `int2e_srsr1srsr2` | intor4.c:541 | `c2s_si_2e1` + `c2s_si_2e2` | **1** | ✅ |
| `int2e_ssp1ssp2` | **gaunt1.c:111** | `c2s_si_2e1i` + `c2s_si_2e2i` | **1** | ❌ **gaunt1.c NOT wired** |
| `int2e_ssp1sps2` | gaunt1.c:209 | `c2s_si_2e1i` + `c2s_si_2e2i` | **1** | ❌ |
| `int2e_sps1ssp2` | gaunt1.c:307 | `c2s_si_2e1i` + `c2s_si_2e2i` | **1** | ❌ |
| `int2e_sps1sps2` | gaunt1.c:405 | `c2s_si_2e1i` + `c2s_si_2e2i` | **1** | ❌ |
| `int2e_spv1` | **dkb.c:249** | `c2s_si_2e1` + `c2s_sf_2e2` | **1** | ❌ **dkb.c NOT wired** |
| `int2e_vsp1` | dkb.c:333 | `c2s_si_2e1` + `c2s_sf_2e2` | **1** | ❌ |
| `int2e_spv1spv2` | dkb.c:501 | `c2s_si_2e1` + `c2s_si_2e2` | **1** | ❌ |
| `int2e_vsp1spv2` | dkb.c:597 | `c2s_si_2e1` + `c2s_si_2e2` | **1** | ❌ |
| `int2e_spv1vsp2` | dkb.c:693 | `c2s_si_2e1` + `c2s_si_2e2` | **1** | ❌ |
| `int2e_vsp1vsp2` | dkb.c:789 | `c2s_si_2e1` + `c2s_si_2e2` | **1** | ❌ |
| `int2e_spv1spsp2` | dkb.c:911 | `c2s_si_2e1` + `c2s_si_2e2` | **1** | ❌ |
| `int2e_vsp1spsp2` | dkb.c:1033 | `c2s_si_2e1` + `c2s_si_2e2` | **1** | ❌ |

> **`int2e_spsp1` gout (intor4.c:19, rank 4):** `+s5-s7, +s6-s2, +s1-s3, +s0+s4+s8` — identical structure to `int1e_spnucsp` (the σ·p G-tensor → x,y,z,scalar blocks). [VERIFIED: intor4.c read]
> **CONTEXT.md item-3 correction:** the intor4.c L636/899/990/1249/1539/1700/2197 "i-variant arms" cited in CONTEXT.md are GIAO×σ families (`int2e_cg_sa10sp1`, `int2e_giao_sa10sp1`, `int2e_spgsp1`, `int2e_g1`) — **Phase-30 deferred**, NOT REL-04. The real REL-04 ssp/sps/vsp/spv drivers are in gaunt1.c/dkb.c (above). [VERIFIED: intor4.c symbol names at those lines, read this session]

### 4. component_rank guard (RESOLVES research item 4)
Every Group-4 family produces a single complex spinor matrix per shell pair/quartet → **component_rank = "1"** in the lock for ALL of them (the σ-component axis is folded INTO the spinor coupling by the transform; it is not an output component axis like a gradient's rank-3). The existing `int1e_sp_spinor` lock row is already `component_rank: "1"` (lock L10721). The σ_x/σ_y/σ_z fold is internal to `c2s_si`; do NOT set component_rank to 3 or 4 for these. The `int1e_sigma` ng[7]=3 is the number of σ G-tensor components the transform consumes, not the output rank. **Action:** set `component_rank: "1"` on every new row; verify the existing sp row stays "1". This guards the component_rank-truncation landmine (a wrong rank>1 would mis-stride the interleaved output).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Single-block si/iket-si coupling | New CG-coupling math | `cart_to_spinor_si` / `cart_to_spinor_iket_si` (c2spinor.rs:392/457) | Phase-12 proven, byte-identity covered. |
| 1e si bra+ket driver | New transform | `cart_to_spinor_si_2d` (c2spinor.rs:673) | Phase-28 proven at atol=1e-12. |
| σ·p G-tensor (1e) | New gout | `kernels/sigma_p.rs` generic `#[cube]` assembler | Phase-28 D-03 front-loaded for exactly this reuse. |
| 2e two-stage transform skeleton | From-scratch | Mirror `cart_to_spinor_sf_4d` (c2spinor.rs:1199) structure | Same electron-1-then-electron-2 + `zcopy_iklj` shape. |
| `_len_spinor` GT/LT/both sizing | Hardcoded `4l+2` | `spinor_len(l, kappa)` (c2spinor.rs:25) | Handles kappa≠0 `2l`/`2l+2`. |
| spsp device/host harness | New | mine `center_4c1e.rs::test_device_matches_host_spsp` (L1878) | Existing σ·p-on-both-sides pattern. |
| Vendor FFI for Group-4 symbols | New extern decls | symbols ALREADY in `cint_funcs.h`; add only `vendor_*` Rust wrappers + `.file()` | confirmed via grep of cint_funcs.h this session. |

**Key insight:** Wave 1 reuses Phase-28 infrastructure almost entirely; its only new transform is the 1e `si_2di` (imaginary-ket) variant for `sr`/`sigma`. Wave 2 is where the real new code lives — the 2e transform suite, built by mirroring the existing host `sf_4d` skeleton with the si bra step swapped in.

## Common Pitfalls

### Pitfall 1: REL-04 source files not in oracle build (BLOCKING)
**What goes wrong:** REL-04 vendor parity silently has no reference symbol to link against → linker error or, worse, a `vendor_*` shim that can't be implemented.
**Why:** CONTEXT.md states "intor4.c is ALREADY wired… only vendor FFI shims needed, NOT a build change." Research found ssp/sps in `gaunt1.c` and vsp/spv in `dkb.c`, **neither in build.rs** (grep returned nothing).
**How to avoid:** Add `"src/autocode/gaunt1.c"` and `"src/autocode/dkb.c"` to the `.file()`/rerun list in `crates/cintx-oracle/build.rs`. Both use the standard include set (same as intor4.c) — no extra deps. REL-03 (`intor4.c`) genuinely needs no build change.
**Warning signs:** Undefined-symbol link errors for `int2e_ssp1ssp2_spinor` / `int2e_vsp1_spinor` when `CINTX_ORACLE_BUILD_VENDOR=1`.

### Pitfall 2: Wrong transform per family
**What goes wrong:** Using `c2s_si_1e` for `int1e_spsp` (actually `c2s_sf_1e`) or the real variant for `sigma`/`sr` (actually `c2s_si_1ei`) → off-by-conjugate / wrong-block parity failure at the gate.
**How to avoid:** Use the Per-Family Transform Map above as the single source of truth; cross-check each launcher arm against the `CINT*_spinor_drv` line.
**Warning signs:** Real part matches, imaginary part is sign-flipped (si vs sii) — or vice-versa.

### Pitfall 3: kappa≠0 spinor sizing
**What goes wrong:** Hardcoding `4l+2` over-/under-sizes for kappa≠0 shells (the D-02 fixture is GT/LT mix).
**How to avoid:** All sizing via `spinor_len` (`_len_spinor`: kappa=0→4l+2, kappa<0→2l+2 GT, kappa>0→2l LT). [VERIFIED: cart2sph.c:3537]
**Warning signs:** BufferTooSmall or truncated tail on the kappa fixture, fine on a kappa=0 fixture.

### Pitfall 4: KET→BRA transpose omission
**What goes wrong:** Device cart blocks are KET-major; the bra step reads BRA-major. A launcher/transform that skips the transpose silently passes on square blocks (transpose-symmetric) and fails on non-square.
**How to avoid:** Own the transpose INSIDE the transform (Phase-27 D-06; `cart_to_spinor_si_2d` already does this at c2spinor.rs:719). Replicate for the 2e transforms. D-02's non-square fixture surfaces it.

### Pitfall 5: OperatorId positional shift
**What goes wrong:** Adding ~16 new manifest rows re-points hardcoded `OperatorId::new(N)` / `_OPERATOR_ID: u32 = N` test consts at a different family → InvalidShellTuple arity mismatch.
**How to avoid:** Resolve OperatorIds by SYMBOL NAME; re-grep `OperatorId::new(` and `_OPERATOR_ID` after adding rows. [project memory: operator_id_shift_breaks_hardcoded_test_consts]

### Pitfall 6: Each inline Spinor dispatch arm needs its own staging guard
**What goes wrong:** A new Spinor arm that scatters directly bypasses the `launch_*_pair` BufferTooSmall guard (Phase-28 CR-01).
**How to avoid:** Add a fail-closed staging guard (`required = di*dj*2`, or `di*dj*dk*dl*2` for 2e) to every new inline arm before any write.

### Pitfall 7: CubeCL CpuRuntime FP-environment side effect
**What goes wrong:** A `#[cube]` launch perturbs subsequent host f64 accumulation ~1e-11 → trips the flat atol=1e-12 gate even with bit-identical kernel math.
**How to avoid:** Suspect this BEFORE chasing kernel numerics; mitigate by batching launches or keeping the band host. [project memory: cubecl_cpuruntime_fp_env_side_effect]

### Pitfall 8: nctr>1 column/row-major coeff transpose
**What goes wrong:** libcint env coeff block is column-major (`env[ci*nprim+ip]`); cintx Shell is row-major — verbatim copy transposes nctr>1 coeffs (latent until an nctr>1 fixture).
**How to avoid:** D-02 keeps ≥1 nctr>1 shell; `si_2d` already handles nctr>1 (verified Phase 28). For the 2e transform, confirm the contraction loop (`i_ctr*j_ctr*k_ctr*l_ctr`) matches libcint's. [project memory: raw_nctr_coeff_transpose]

## Code Examples

### `_len_spinor` (sizing — match exactly)
```c
// Source: libcint-master/src/cart2sph.c:3537
static FINT _len_spinor(FINT kappa, FINT l) {
    if (0 == kappa)      return 4 * l + 2;   // both GT+LT
    else if (kappa < 0)  return 2 * l + 2;   // GT  j=l+1/2
    else                 return 2 * l;       // LT  j=l-1/2
}
```
cintx equivalent: `spinor_len(l, kappa)` (c2spinor.rs:25).

### `a_bra_cart2spinor_si` (1e bra σ-mix — already in cintx, the reference)
```c
// Source: cart2sph.c:3920 — the 4-cart-block (gx,gy,gz,g1) σ fold
saR += caR*v1 + caI*vz - cbR*vy + cbI*vx;   // α real
saI += -caI*v1 + caR*vz + cbI*vy + cbR*vx;  // α imag
sbR += cbR*v1 - cbI*vz + caR*vy + caI*vx;   // β real
sbI += -cbI*v1 - cbR*vz - caI*vy + caR*vx;  // β imag
```

### `int1e_spsp_spinor` driver — note the SF transform
```c
// Source: intor3.c:405-412
envs.f_gout = &CINTgout1e_int1e_spsp;        // gout rank 1: +s0+s4+s8
return CINT1e_spinor_drv(out, dims, &envs, cache, &c2s_sf_1e, 0);  // NOT si!
```

### `int2e_spsp1_spinor` driver — the thinnest 2e si family (micro-test target)
```c
// Source: intor4.c:79-86
envs.f_gout = &CINTgout2e_int2e_spsp1;       // gout rank 4: σ·p G-tensor on bra
return CINT2e_spinor_drv(out, dims, &envs, opt, cache, &c2s_si_2e1, &c2s_sf_2e2);
```

## Runtime State Inventory

This is a code/manifest/build phase (register families, add transforms, add fixture, add vendor shims). No stored data, live-service config, OS-registered state, or secrets are touched.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no databases or persisted keys involved. | none |
| Live service config | None — no external services. | none |
| OS-registered state | None — no scheduled tasks / daemons. | none |
| Secrets/env vars | `CINTX_ORACLE_BUILD_VENDOR` and `--features cpu` are the existing double-gate flags (no new ones). | none (use existing) |
| Build artifacts | Adding `gaunt1.c`/`dkb.c` to `cintx-oracle/build.rs` changes the vendor cc build; the manifest lock auto-syncs both audit sides on row edits (no separate fixtures list). | rebuild vendor lib (automatic via build.rs); regenerate manifest from lock. |

## Common Pitfalls — already covered above.

## State of the Art

| Old (CONTEXT.md / Phase-28 assumption) | Current (verified this session) | Impact |
|------------------------|---------------------------------|--------|
| "intor4.c already wired → REL-04 needs no build change" | ssp/sps in gaunt1.c, vsp/spv in dkb.c — NOT in build.rs | Wave 3 MUST add 2 source files to build.rs. |
| "all σ families use c2s_si" | spsp→c2s_sf_1e; sigma/sr→c2s_si_1ei | Per-family transform map is mandatory. |
| CONTEXT.md item-3 cites intor4.c L636/899/… as REL-04 arms | those are GIAO×σ (Phase-30) families | Don't transcribe those for REL-04; use gaunt1.c/dkb.c. |
| Phase-28 record: 2e si transforms → Phase 30/31 | corrected by D-01: they are Phase-29 | (already locked in CONTEXT) |

## Validation Architecture

> nyquist_validation is enabled (config.json `workflow.nyquist_validation: true`).

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + `#[cfg(has_vendor_libcint)]` double-gate; oracle integration tests under `crates/cintx-oracle/tests/`. `cargo nextest` available (CLAUDE.md dev tool). |
| Config file | none (cargo test harness); gate cfg emitted by `cintx-oracle/build.rs` |
| Quick run command | `cargo test -p cintx-cubecl --features cpu --lib c2spinor` (transform unit tests, fast) |
| Full suite command | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test '*spinor*'` |

> **Double-gate (project memory `reference_oracle_vendor_parity_invocation`):** real parity requires BOTH `--features cpu` AND `CINTX_ORACLE_BUILD_VENDOR=1`. Without both, vendor bodies compile out and parity silently SKIPS (determinism-only). Every new parity test MUST carry the Phase-27 D-10 NO-SILENT-SKIP assertion that FAILS (not skips) if the vendor arm did not run when the double-gate was present.

### Phase Requirements → Test Map
| Req | Behavior | Test Type | Automated Command | File Exists? |
|-----|----------|-----------|-------------------|--------------|
| REL-01 `spsp` | byte-identity vs `int1e_spsp_spinor` (c2s_sf_1e path) | vendor parity | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test rel_1e_sigma_parity test_spsp` | ❌ Wave 1 |
| REL-01 `spnucsp`/`sprinvsp` | byte-identity (c2s_si_1e, 2-/1-center) | vendor parity | `…--test rel_1e_sigma_parity test_spnucsp test_sprinvsp` | ❌ Wave 1 |
| REL-02 `sp` | flip + byte-identity (c2s_si_1e) | vendor parity | `…--test rel_1e_sigma_parity test_sp` | ❌ Wave 1 (reuses Phase-28 path) |
| REL-02 `srsr`/`srnucsr` | byte-identity (c2s_si_1e) | vendor parity | `…test_srsr test_srnucsr` | ❌ Wave 1 |
| REL-02 `sr`/`sigma` | byte-identity (c2s_si_1ei — imaginary ket) | vendor parity | `…test_sr test_sigma` | ❌ Wave 1 (needs new `si_2di`) |
| Wave-2 transform | `c2s_si_2e1/2e2(+i)` + `c2s_sf_2e1/2e2` byte-identity (the D-03 gating micro-test, FIRST task) | transform parity | `…--test si_2e_transform_parity` (via thinnest family `int2e_spsp1_spinor` = `c2s_si_2e1`+`c2s_sf_2e2`) | ❌ Wave 2 |
| REL-03 `spsp1`/`srsr1` | byte-identity (c2s_si_2e1+c2s_sf_2e2) | vendor parity | `…--test rel_2e_sigma_parity test_spsp1 test_srsr1` | ❌ Wave 3 |
| REL-03 `spsp1spsp2`/`srsr1srsr2` | byte-identity (c2s_si_2e1+c2s_si_2e2) | vendor parity | `…test_spsp1spsp2 test_srsr1srsr2` | ❌ Wave 3 |
| REL-04 `ssp1ssp2`/`sps1sps2` | byte-identity (c2s_si_2e1i+c2s_si_2e2i) — needs gaunt1.c in build | vendor parity | `…test_ssp1ssp2 test_sps1sps2` | ❌ Wave 3 (+build.rs) |
| REL-04 `vsp1`/`spv1` (+2-sided) | byte-identity (c2s_si_2e1+c2s_sf_2e2 / +c2s_si_2e2) — needs dkb.c in build | vendor parity | `…test_vsp1 test_spv1` | ❌ Wave 3 (+build.rs) |

### Sampling Rate (coverage strategy)
- **Primary gate fixture — `build_kappa_spinor_2e_fixture` (D-02):** 4 spinor shells, NON-SQUARE (defeats transpose symmetry), GENUINE kappa≠0 GT/LT mix (stresses `2l`/`2l+2` sizing, not just `4l+2`), ≥1 shell nctr>1 (catches coeff transpose). This is the byte-identity gate for every REL-03/04 family. For 1e (Wave 1), reuse Phase-28's `build_kappa_spinor_fixture` (p kappa=+1 LT × d kappa=−1 GT, nctr=2).
- **Realism cross-check — `build_heavy_atom_spinor_fixture` (secondary):** asserted finite, NOT the primary gate; guards against synthetic-fixture blind spots.
- **Per task commit:** `cargo test -p cintx-cubecl --features cpu --lib c2spinor` (transform unit) + `cargo clippy`.
- **Per wave merge:** full vendor parity suite for that wave's families under the double-gate; manifest-audit green.
- **Phase gate:** all REL-01..04 families `oracle_covered=true` **spinor-only** (SC#5: do not over-claim cart/sph σ intermediates); NO-SILENT-SKIP assertions pass; `manifest-audit` green; `cargo test --workspace` green. The Wave-2 transform micro-test must be GREEN before any Wave-3 family wires onto the transform (D-03 mitigation).

### "Covered, non-skipped" per family
A Group-4 family is legitimately `oracle_covered=true` only when: (1) its `vendor_*` shim links against a real libcint 6.1.3 driver (so for REL-04, gaunt1.c/dkb.c MUST be in build.rs), (2) its parity test runs on a kappa-bearing fixture with N>0 byte-comparisons under BOTH gate flags, (3) the NO-SILENT-SKIP assertion confirms the vendor arm executed (not `skipped`), and (4) atol=1e-12 byte-identity holds. The `xtask oracle_covered_update` SC#4 guard must refuse to flip any family whose only fixture was `skipped`.

### Wave 0 Gaps
- [ ] `crates/cintx-oracle/tests/rel_1e_sigma_parity.rs` — REL-01/02 (new)
- [ ] `crates/cintx-oracle/tests/si_2e_transform_parity.rs` — Wave-2 gating micro-test (new)
- [ ] `crates/cintx-oracle/tests/rel_2e_sigma_parity.rs` — REL-03/04 (new)
- [ ] `build_kappa_spinor_2e_fixture` in `crates/cintx-oracle/src/fixtures.rs` (new, D-02)
- [ ] `vendor_int1e_{spsp,spnucsp,sprinvsp,srsr,srnucsr,sr,sigma}_spinor` shims (symbols in cint_funcs.h; add wrappers)
- [ ] `vendor_int2e_{spsp1,srsr1,spsp1spsp2,srsr1srsr2,ssp1ssp2,sps1sps2,vsp1,spv1,…}_spinor` shims
- [ ] `gaunt1.c` + `dkb.c` added to `cintx-oracle/build.rs` (REL-04 — BLOCKING)
- [ ] new 1e `cart_to_spinor_si_2di` (imaginary-ket) in c2spinor.rs for `sr`/`sigma`
- [ ] 2e transform suite (`c2s_si_2e1/2e2(+i)` + `c2s_sf_2e1/2e2`) in c2spinor.rs

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `num-complex` version unchanged from prior phases | Standard Stack | Low — no new complex API used; carry-over. |
| A2 | `int1e_sigma` lock component_rank should be "1" (not 3) — the σ-component axis is folded by the transform, mirroring the existing sp row | Per-Family Map item 4 | Medium — if libcint emits sigma as a rank-3 spinor output (3 separate σ matrices), the row needs rank 3 and the launcher a component loop. MUST confirm against `vendor_int1e_sigma_spinor` output shape during Wave-1 first task (compare `CINTcgto_spinor`-sized output count). The gout is rank-12 (`gout[n*12+..]`, 3 σ blocks ×4 cart-block components) which the `c2s_si_1ei` transform consumes — this strongly implies a single fused spinor output, but verify the buffer length empirically. |
| A3 | gaunt1.c/dkb.c compile cleanly with the existing include set + cc flags (same includes as intor4.c) | Pitfall 1 | Low — includes verified identical this session; risk is only an unexpected missing symbol, caught at link time in Wave 3. |

## Open Questions (RESOLVED)

1. **`int1e_sigma` output rank (component_rank 1 vs 3)**
   - What we know: gout is `gout[n*12+..]` (12 slots = 3 σ-components × 4 cart-blocks); driver routes through `c2s_si_1ei`; `ng[7]=3`.
   - What's unclear: whether the final spinor buffer is a single `di*dj*2` complex matrix (rank 1) or three stacked σ-matrices (rank 3).
   - Recommendation: Wave-1 first task — call `vendor_int1e_sigma_spinor` on the kappa fixture, measure output length vs `di*dj*2`. Set component_rank from the measured shape. (A2.)
   - **RESOLVED:** `component_rank = "1"`. `int1e_sigma_spinor` (intor3.c:75-82) makes a SINGLE `CINT1e_spinor_drv(..., &c2s_si_1ei, 0)` call producing one fused complex spinor matrix; the `ng[7]=3` / rank-12 gout is the σ-component count the `c2s_si_1ei` transform CONSUMES, not a free output axis — mirroring the existing `int1e_sp` lock row (rank "1"). Belt-and-suspenders: 29-02 T1 still empirically measures the vendor output length vs `di*dj*2` and branches if it disagrees.

2. **2e σ·p device assembler reuse vs new**
   - What we know: `sigma_p.rs` is a generic 1e σ·p `#[cube]` emitter (tensor_rank param). The 2e families need the same 4-block (x,y,z,1) gout but in the 2e G-tensor layout (`<ik|lj>` ordered, k/l electron-2 indices interleaved).
   - What's unclear: whether `sigma_p.rs` generalizes to 2e directly or needs a 2e sibling.
   - Recommendation: Plan Wave 2/3 to mine `center_4c1e.rs::test_device_matches_host_spsp` (L1878) for the 2e σ·p-on-both-sides cart gout; treat a thin 2e σ·p assembler as expected new code, not a reuse.
   - **RESOLVED:** a NEW thin 2e σ·p sibling is required (do not reuse `sigma_p.rs` as-is). The 2e σ·p gout (intor4.c:19, rank 4 — same x,y,z,1 block structure) needs the 2e G-tensor `<ik|lj>` electron-index layout, which `sigma_p.rs`'s 1e emitter cannot express; mine `center_4c1e.rs::test_device_matches_host_spsp` (L1878) for the cart gout pattern.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain | all | ✓ (assumed pinned) | 1.94.0 (rust-toolchain.toml) | — |
| vendored libcint sources | oracle parity | ✓ | 6.1.3 (`libcint-master/`) | — |
| `gaunt1.c`/`dkb.c` (in tree, not yet built) | REL-04 vendor ref | ✓ present, ✗ not wired | 6.1.3 | none — MUST wire into build.rs |
| CubeCL CpuRuntime (`--features cpu`) | device σ·p gout | ✓ | 0.10.0 | rocm (gated) |
| `CINTX_ORACLE_BUILD_VENDOR=1` | vendor cc build | env-gated | — | without it, parity skips (not acceptable for gate) |

**Missing dependencies with no fallback:** gaunt1.c/dkb.c must be added to build.rs for REL-04 — no alternative reference exists.

## Security Domain

`security_enforcement` is not set in config.json. This phase is a numerical-integral library feature (no auth/session/network/input-from-untrusted-source surface). The only relevant control is V5 Input Validation, satisfied by the existing fail-closed buffer guards (`BufferTooSmall`/`ChunkPlanFailed`, OOM-safe stop, no partial writes per CLAUDE.md). No ASVS categories beyond V5 apply.

## Sources

### Primary (HIGH confidence — read directly this session)
- `libcint-master/src/cart2sph.c` — `_len_spinor` (L3537), `a_bra_cart2spinor_si` (L3920), `a_bra1_cart2spinor_zi/zf` (L4118/4188), `c2s_sf_2e1/2e1i/2e2` (L5385/5424/5469), `c2s_si_2e1/2e1i/2e2/2e2i` (L5592/5639/5687/5752).
- `libcint-master/src/autocode/intor3.c` — 1e σ drivers + gouts: sigma (L18/81), srsr (L168/229), sr (L234/285), srsp (L290/351), spsp (L356/411), sp (L416/467), spnucsp (L472/537), sprinvsp (L542/607), srnucsr (L612/677).
- `libcint-master/src/autocode/intor4.c` — int2e_spsp1 (L19/85), spsp1spsp2 (L277), srsr1 (L349), srsr1srsr2 (L541); GIAO×σ at L636/899/990/1249/1539/1700/2197 (Phase-30, NOT REL-04).
- `libcint-master/src/autocode/gaunt1.c` — ssp1ssp2 (L111), ssp1sps2 (L209), sps1ssp2 (L307), sps1sps2 (L405) → all `c2s_si_2e1i`+`c2s_si_2e2i`.
- `libcint-master/src/autocode/dkb.c` — spv1 (L249), vsp1 (L333), spv1spv2 (L501), vsp1spv2 (L597), spv1vsp2 (L693), vsp1vsp2 (L789), spv1spsp2 (L911), vsp1spsp2 (L1033).
- `libcint-master/include/cint_funcs.h` — all Group-4 spinor symbols declared (no suppl-header needed).
- `crates/cintx-cubecl/src/transform/c2spinor.rs` — spinor_len (L25), cart_to_spinor_si/iket_si (L392/457), sf_2d (L547), si_2d (L673), sf_4d (L1199).
- `crates/cintx-cubecl/src/kernels/sigma_p.rs` — generic σ·p `#[cube]` assembler + `launch_int1e_sp_spinor_pair` (L565).
- `crates/cintx-oracle/build.rs` — vendor cc build (intor4.c at L62; gaunt1.c/dkb.c ABSENT, confirmed by grep).
- `crates/cintx-ops/generated/compiled_manifest.lock.json` — only `int1e_sp_spinor` row (L10710), component_rank "1", oracle_covered false.
- `crates/cintx-cubecl/src/kernels/two_electron.rs` — all 2e Spinor currently `UnsupportedApi` (no 2e si transform exists).
- `.claude/skills/spike-findings-cintx/references/spinor-layout.md` — interleaved-complex output layout.
- `.planning/phases/{27,28}-*/…-CONTEXT.md`, REQUIREMENTS.md L109-112, config.json.

### Secondary / Tertiary
- None — all claims sourced from primary code reads above.

## Metadata

**Confidence breakdown:**
- Per-family transform map: HIGH — every driver line read directly from intor3.c/intor4.c/gaunt1.c/dkb.c.
- 2e transform layout: HIGH (libcint source) for the algorithm; MEDIUM for the cintx port surface (new code, D-03 micro-test gates it).
- component_rank values: HIGH for the 4-block-fold families; MEDIUM for `int1e_sigma` (A2/Open Q1 — verify output shape empirically in Wave 1).
- build.rs gap for REL-04: HIGH — grep confirmed gaunt1.c/dkb.c absent.
- Pitfalls/landmines: HIGH — sourced from project memory + code.

**Research date:** 2026-05-31
**Valid until:** 2026-06-30 (libcint vendored source is frozen at 6.1.3; cintx code may shift — re-verify file line numbers if c2spinor.rs/build.rs change before planning).
