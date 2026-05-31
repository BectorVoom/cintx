# Phase 30: Group 5 (GIAO×σ slice) — Spin-GIAO Integrals (spinor) - Research

**Researched:** 2026-06-01
**Domain:** Relativistic-NMR GIAO×σ spinor integrals — gauge-origin fold into the σ·p device assembler + per-family c2s pairing + combined gauge∧kappa vendor byte-identity gate (libcint 6.1.3, cintx CubeCL backend)
**Confidence:** HIGH (all findings transcribed directly from `libcint-master/src/autocode/intor3.c`/`intor4.c`/`g1e.c`/`g2e.h` and verified against cintx source; no training-data assumptions in the load-bearing gauge/transform claims)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01a — defer Gaunt to Phase 31:** The Gaunt GIAO families `int2e_{cg,giao}_ssa10ssp2` are NOT in Phase 30 (they live in `autocode/gaunt1.c`, decompose via `launch_breit`; the GIAO-03 glob `int2e_cg_sa10*` excludes the `ssa10` prefix). No new `launch_breit` dependency in Phase 30. [VERIFIED below]
- **D-01b — include all `sa01` arms:** `int1e_spgsa01`, `int1e_cg_sa10sa01`, `int1e_giao_sa10sa01` ARE in scope.
- **D-02 — ONE combined gauge∧kappa fixture:** a single fixture simultaneously gauge-origin≠0 (`with_common_origin`/`PTR_COMMON_ORIG`) AND kappa≠0 GT/LT mix, non-square, ≥1 shell nctr>1. 1e form (Wave 1 gate) extended to a 4-shell 2e form (Wave 2 gate). Extends Phase-29's `build_kappa_spinor_2e_fixture` with a non-zero common origin.
- **D-03 — inherit Phase-29 transcribe+vendor gate; gout-level micro-test first.** No separate design spike. Transcribe gouts/transforms from `intor3.c`/`intor4.c`; prove by atol=1e-12 vendor byte-identity. Mitigation relocated to a GOUT-level gauge-gout byte-identity micro-test as the FIRST task.
- **D-04 — wave decomposition gout micro-test → 1e-all → 2e-all:** Wave 0 = gauge-gout micro-test + combined fixture (1e); Wave 1 = all 9 1e families; Wave 2 = all 6 2e families. Each wave vendor-parity green before the next.

### Claude's Discretion
- Internal module/factoring for the gauge-origin σ·p gout (extend `kernels/sigma_p.rs` with a gauge-origin variant vs a dedicated `giao_sigma` assembler) — **resolved in §Architecture Patterns below.**
- Exact per-family gout component ordering and `c2s_si_1ei` vs `c2s_si_1e` vs `c2s_sf_1e` arm per family — **resolved in §Per-Family gout & Transform Map below.**
- Exact molecule/element + kappa + gauge-origin coordinates for the combined fixture (subject to D-02 hard constraints).
- Whether the Wave-0 gout micro-test compares to a vendored thin family or a hand-derived reference — **recommendation in §Wave-0 Micro-Test below.**
- Precise plan boundaries inside each wave.

### Deferred Ideas (OUT OF SCOPE)
- Gaunt GIAO families (Phase 31): `int2e_cg_ssa10ssp2`, `int2e_giao_ssa10ssp2` (`autocode/gaunt1.c`, `launch_breit`).
- Gauge / Breit–Gaunt 2e (Phase 31): `int2e_gauge_r1/r2_*`, Gaunt `ssp/sps` (BREIT-01/02/03).
- PARITY-01 full-parity gate (Phase 31).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GIAO-03 | GIAO×σ slice (`int1e_spg*`, `int1e_spgnucsp`, `*_sa10*`, `int2e_cg_sa10*`/`giao_sa10*`) match at atol=1e-12 (spinor) via FND-05 | Full per-family gout bodies, gauge-origin fold structure (`CINTx1i_1e(...,dri,...)`), per-family c2s transform map, component ranks, combined gauge∧kappa fixture design, registration mechanics, and the landmine application below give the planner everything needed to wire all 15 families. GIAO-03 is **fully closed** by this set — the glob `int2e_cg_sa10*` excludes `ssa10` (VERIFIED: `cint_funcs.h:488` vs `gaunt1.c:411`). |
</phase_requirements>

---

## Summary

The GIAO×σ slice is **mechanically large (15 families) but conceptually narrow**. Every transform it needs already exists (`cart_to_spinor_si_2di` = `c2s_si_1ei`, `cart_to_spinor_si_2d`/`sf_2d`, the Phase-29 2e si/sf suite), the gauge-origin env slot is plumbed end-to-end (FND-01, `PTR_COMMON_ORIG=1`), the complex-interleaved output works (FND-03), and `intor3.c`/`intor4.c` are already compiled into the oracle (Phase 29). The deferred Gaunt families are cleanly separated (`gaunt1.c`, distinct symbols, no glob overlap), and **no in-scope family uses `launch_breit`** — all 6 2e families use `CINT2e_spinor_drv` [VERIFIED].

**The one genuinely-new piece** is folding the gauge-origin GIAO factor into a σ·p G-tensor assembler. The critical structural finding: in libcint the gauge factor is **NOT a post-multiply cross-product in the gout body**. It enters via the G-tensor build macros `G1E_RCI`/`G1E_R_I`/`G1E_R0I`, which all call `CINTx1i_1e(f, g, origin, ...)` computing `f[i] = g[i+1] + origin·g[i]` (`g1e.c:446-448`). The `origin` argument is what differs per family: `cg_sa10*` uses `dri = ri - env[PTR_COMMON_ORIG]` (the only gauge-origin-dependent arms), `giao_sa10*` uses the natural bra center (no shift), and `spg*` uses `ri` (R0I) plus a London `rirj=ri-rj` phase in the gout. This means the new device work is an `x1i`-with-origin recurrence step inserted into the σ·p G-tensor build, parameterized by an `origin` 3-vector supplied from the host — exactly mirroring the Phase-26 `drj=rj-common_orig` ket-side precedent (`one_electron.rs:9258`), but on the **bra** side and combined with a ket nabla (`G1E_D_J`).

**Primary recommendation:** Extend `kernels/sigma_p.rs` with a gauge-origin-parameterized variant (NOT a separate `giao_sigma` module) — the assembler already owns the overlap base G-tensor + bra-nabla + 4-block gc packing; add (a) an `origin: [f64;3]` kernel input feeding an `x1i`-with-origin step, (b) a ket-nabla `G1E_D_J` build, and (c) the family-specific gout mix as a `#[comptime]`-selected variant. Drive Wave 0 with a gauge-gout byte-identity micro-test against a **vendored thin family** (recommend `int1e_cg_sa10sp`, the smallest common-origin-reading arm) before wiring the full set.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Overlap base G-tensor (VRR+HRR) | Device `#[cube]` (`sigma_p.rs`) | — | Already on-device; per-primitive hot path |
| Bra nabla `∇_i` (σ·p) | Device `#[cube]` | — | Existing `sigma_p_kernel` step |
| Ket nabla `∇_j` (`G1E_D_J`) | Device `#[cube]` | — | New build step for GIAO×σ; same recurrence family as bra nabla |
| Gauge-origin `x1i`-with-origin (`G1E_RCI`/`R0I`/`R_I`) | Device `#[cube]` | — | NEW: the gauge fold; `origin` 3-vector supplied from host (`raw.rs` reads `PTR_COMMON_ORIG`) |
| Nuclear Rys atom-sum (nucsp arms) | Device `#[cube]` | — | Reuse Phase-29 σ·p-nuc Rys path |
| `PTR_COMMON_ORIG` env-read → `origin` | Host (`raw.rs::eval_raw`) | — | FND-01 plumbing already exists; consume `plan.operator_env_params.common_orig` |
| Cart→spinor reduce (`c2s_si_1ei`/`si_1e`/`sf_1e`, 2e suite) | Host (`c2spinor.rs`) | — | Existing transforms; KET→BRA transpose owned here |
| Complex interleaved `[re,im]` staging | Host | — | FND-03; GIAO is purely imaginary → imaginary lane |
| Vendor byte-identity gate | Host test (`cintx-oracle`) | — | `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1` |

---

## Standard Stack

No new dependencies. This phase is pure cintx-internal (CubeCL kernel extension + host transform reuse + manifest rows + oracle FFI). The CLAUDE.md-pinned stack applies unchanged:

| Component | Version | Role this phase |
|-----------|---------|-----------------|
| `cubecl` | 0.10.0 (pinned) | `#[cube]` gauge-origin σ·p assembler extension |
| `thiserror` | 2.0.18 | `UnsupportedApi`/`BufferTooSmall` library errors (fail-closed guards) |
| `num-complex` | (workspace) | spinor complex output views |
| vendored libcint | 6.1.3 (`libcint-master/`) | byte-identity oracle reference (already in `build.rs`) |

**No `npm`/registry install. No version verification needed.** [VERIFIED: phase touches no external dependency surface — see Environment Availability §.]

---

## Architecture Patterns

### System Architecture Diagram (per GIAO×σ family, spinor path)

```
                  ExecutionPlan (Representation::Spinor, operator=<family>)
                  operator_env_params.common_orig = Some([cx,cy,cz])  ← FND-01, raw.rs reads PTR_COMMON_ORIG
                            │
                            ▼
            ┌─────────────────────────────────────────────────────┐
            │  one_electron.rs / two_electron.rs  (HOST dispatch)   │
            │  • per-inline-Spinor-arm fail-closed staging guard    │ ← Phase-28 CR-01
            │  • origin = common_orig.unwrap_or([0;3])              │
            │  • dri = [ri-origin]  (cg_sa10*)                      │
            │    OR natural bra center (giao_sa10*)                 │
            │    OR ri / rirj=ri-rj  (spg*)                         │
            └───────────────────────┬─────────────────────────────┘
                                    ▼
            ┌─────────────────────────────────────────────────────┐
            │  sigma_p.rs  (DEVICE #[cube], gauge-origin variant)   │
            │  1. overlap base G-tensor g0   (VRR+HRR)             │
            │  2. ket nabla  g1 = ∇_j(g0)    (G1E_D_J)            │
            │  3. x1i-with-origin  g2 = x1i(g0, origin)           │ ← THE GAUGE FOLD
            │                      g3 = x1i(g1, origin)            │   (CINTx1i_1e: f[i]=g[i+1]+origin·g[i])
            │  4. bra nabla  (σ·p mix)        (G1E_D_I)           │
            │  5. family gout mix  s[0..9] or s[0..27]            │
            │     → 4 gc blocks (gc_x/gc_y/gc_z/gc_1) per tensor  │
            │       component, component-LEADING / KET-major      │
            └───────────────────────┬─────────────────────────────┘
                                    ▼ contracted cart gc blocks (host)
            ┌─────────────────────────────────────────────────────┐
            │  c2spinor.rs  (HOST transform — per §Per-Family Map)  │
            │  c2s_si_1ei (sp arms, imaginary)  ← cart_to_spinor_si_2di
            │  c2s_si_1e  (sa01 arms, real)     ← cart_to_spinor_si_2d
            │  • owns KET→BRA transpose; stages into [re,im] lane  │
            └───────────────────────┬─────────────────────────────┘
                                    ▼
                  spinor out[comp*(ni_sp*nj_sp)*2 + (j*ni_sp+i)*2 + {re,im}]
                                    │
                                    ▼
                  vendor byte-identity gate @ atol=1e-12
                  (combined gauge≠0 ∧ kappa≠0 fixture, D-02)
```

### The Gauge-Origin Fold — Exact Structure (THE genuinely-new work, D-03)

**Critical finding (the load-bearing claim of this research):** The gauge factor is **not** an explicit `g = (r - common_orig) × ...` cross-product applied in the gout. It is folded into the G-tensor through the position-recurrence builder.

[CITED: `libcint-master/src/g1e.h:48-62`] The three relevant macros:
```c
#define G1E_D_I(f, g, li, lj, lk)   CINTnabla1i_1e(f, g, li, lj, lk, envs)   // bra nabla
#define G1E_D_J(f, g, li, lj, lk)   CINTnabla1j_1e(f, g, li, lj, lk, envs)   // ket nabla
#define G1E_R0I(f, g, li, lj, lk)   CINTx1i_1e(f, g, envs->ri, ...)          // origin = ri  (spg*)
#define G1E_RCI(f, g, li, lj, lk)   CINTx1i_1e(f, g, dri, ...)               // origin = dri = ri - common_orig  (cg_sa10*)
#define G1E_R_I(f, g, li, lj, lk)   f = g + envs->g_stride_i                 // natural bra center  (giao_sa10*)
```
(2e equivalents `G2E_R0I`/`G2E_RCI`/`G2E_R_I` are identical, calling `CINTx1i_2e`. [CITED: `g2e.h:93-104`])

[CITED: `libcint-master/src/g1e.c:429-451`] `CINTx1i_1e`:
```c
void CINTx1i_1e(double *f, double *g, double ri[3], FINT li, FINT lj, FINT lk, CINTEnvVars *envs) {
    ... for each (j,k) block, for i in [ptr .. ptr+li]:
        fx[i] = gx[i+1] + ri[0] * gx[i];   // ri[] here is the *origin argument* (dri, ri, etc.)
        fy[i] = gy[i+1] + ri[1] * gy[i];
        fz[i] = gz[i+1] + ri[2] * gz[i];
}
```
So `x1i(g, origin)` = "raise bra angular momentum by one and shift by `origin`". The **gauge-origin enters only through `origin = dri = ri - env[PTR_COMMON_ORIG]`** for the `cg_sa10*` families. [VERIFIED by grep: exactly the 3 `cg_sa10*` families read `PTR_COMMON_ORIG`; the 3 `giao_sa10*` and 3 `spg*` families read 0.]

**Where `dri` is read** [CITED: `intor3.c:1149-1151`, identical pattern at `:1239`, `:1007`, and 2e `intor4.c` cg drivers]:
```c
double dri[3];
dri[0] = envs->ri[0] - envs->env[PTR_COMMON_ORIG+0];
dri[1] = envs->ri[1] - envs->env[PTR_COMMON_ORIG+1];
dri[2] = envs->ri[2] - envs->env[PTR_COMMON_ORIG+2];
```

**cintx mapping:** `raw.rs::eval_raw` already exposes `plan.operator_env_params.common_orig` (FND-01). The Phase-26 precedent reads it ket-side: `let origin = common_orig.unwrap_or([0.0;3]); let drj = [rj[k]-origin[k]];` (`one_electron.rs:9258-9259`). Phase 30's bra-side analog is `let dri = [ri[k]-origin[k]];` passed into the kernel as a 3-vector that feeds the new `x1i`-with-origin device step.

### Family-class structure (3 classes × {sp, nucsp, sa01})

The 9 1e families decompose into 3 origin-classes × 3 operator-arms. The **gout body is byte-identical within an arm across the cg/giao classes** — only the G-tensor builder (`G1E_RCI` vs `G1E_R_I`) differs. [VERIFIED: `cg_sa10sp` gout L1141 ≡ `giao_sa10sp` gout L1462 component-for-component; the sole diff is L1154-1155 `G1E_RCI`+`dri` vs L1471-1472 `G1E_R_I`.]

| Class | Gauge origin | Builder | Families | gout body shape |
|-------|-------------|---------|----------|-----------------|
| `cg_sa10*` | `dri = ri − common_orig` | `G1E_RCI`/`G2E_RCI` | cg_sa10sp, cg_sa10nucsp, cg_sa10sa01 | s[0..9] (sp/nucsp), s[0..9]→n*36 (sa01) |
| `giao_sa10*` | natural bra center (no shift) | `G1E_R_I`/`G2E_R_I` | giao_sa10sp, giao_sa10nucsp, giao_sa10sa01 | identical gout to cg arm |
| `spg*` | `ri` (R0I) + London `rirj=ri−rj` in gout | `G1E_R0I` + `c[k]=rirj[k]` | spgsp, spgnucsp, spgsa01 | s[0..27], 12-comp gout with `c[]` London factor |

This means the cleanest device factoring is **one assembler parameterized by `origin: [f64;3]`** (= `dri` for cg, = `ri` for spg-R0I; for giao the `G1E_R_I` natural-center path is the `origin`-free branch). The London `rirj` factor for `spg*` is a separate per-component multiply in the gout mix (exactly the `c[0..2]` array at `intor3.c:1739-1741`).

### Pattern: extend `sigma_p.rs`, do NOT create `giao_sigma` (Claude's Discretion resolved)

**Recommendation: extend `kernels/sigma_p.rs`.** Rationale grounded in the gout bodies:
1. The assembler already owns the overlap base G-tensor, the bra-nabla σ·p mix, and the 4-block `gc_x/gc_y/gc_z/gc_1` component-leading KET-major packing that `cart_to_spinor_si_2di` consumes (`sigma_p.rs:26-43`). The GIAO×σ families produce **the same 4-gc-block output shape** — they differ only in the G-tensor *build* (extra `G1E_D_J` ket-nabla + `x1i`-with-origin) and the gout *mix* (s[0..9]/s[0..27] vs the current s[0..2]).
2. A separate `giao_sigma` module would duplicate the VRR/HRR helpers (`sigma_p_vrr_axis`/`sigma_p_hrr_axis`), the contraction loop, and the gc-block packing — the exact code the existing assembler centralizes.
3. The existing `#[comptime] tensor_rank` parameter (`sigma_p.rs:138`) is the natural extension point: add a `#[comptime]` family/variant selector that switches the gout mix and a runtime `origin: [f64;3]` for the gauge fold.

Keep the current `int1e_sp` path (`tensor_rank=1`, the only currently-exercised path, `sigma_p.rs:404`) untouched; add the GIAO variants alongside.

### Recommended task/file structure

```
crates/cintx-cubecl/src/kernels/sigma_p.rs       # extend: gauge-origin x1i step + G1E_D_J + family gout variants
crates/cintx-cubecl/src/kernels/one_electron.rs  # 9 new inline Spinor arms (each w/ fail-closed staging guard)
crates/cintx-cubecl/src/kernels/two_electron.rs  # 6 new inline Spinor arms (reuse Phase-29 2e si/sf launcher)
crates/cintx-cubecl/src/transform/c2spinor.rs    # NO CHANGE (reuse si_2di/si_2d/sf_2d + 2e suite)
crates/cintx-ops/generated/compiled_manifest.lock.json  # 15 new spinor rows (auto-syncs api_manifest.rs + .csv)
crates/cintx-oracle/src/vendor_ffi.rs            # 15 vendor_int{1,2}e_*_spinor shims (clone sigma/spsp1 shims)
crates/cintx-oracle/src/fixtures.rs              # extend kappa fixtures w/ with_common_origin (D-02)
crates/cintx-oracle/tests/giao_sigma_1e_parity.rs  # Wave 0 micro-test + Wave 1 1e gate
crates/cintx-oracle/tests/giao_sigma_2e_parity.rs  # Wave 2 2e gate
```

### Anti-Patterns to Avoid
- **Treating the gauge factor as a post-multiply cross-product in the gout.** It is an `x1i`-with-origin step in the G-tensor build. A literal `(r-c)×s` post-multiply will NOT match libcint.
- **Assuming `c2s_si_1ei` for all 1e families.** The `sa01` arms use `c2s_si_1e` (real), not `c2s_si_1ei` (imaginary). See §Per-Family Map. Picking by analogy will fail the gate.
- **Hardcoding `ni_sp = 4l+2`.** kappa≠0 → `di = 2l` (LT) or `2l+2` (GT) from `spinor_len`. The D-02 fixture stresses this.
- **Square fixture blocks.** Orientation/KET→BRA-transpose bugs hide in transpose-symmetric square blocks (memory `project_1e_gpu_port_scalar_only`). Keep non-square.

---

## Per-Family gout & Transform Map

**This is the authoritative table the planner transcribes from. Do NOT pick transforms by analogy.** [VERIFIED by reading each `*_spinor` driver's `CINT1e_spinor_drv`/`CINT2e_spinor_drv` last argument in `intor3.c`/`intor4.c`.]

### 1e families (intor3.c)

| Family | gout L | spinor c2s transform | cintx transform symbol | builder (gauge) | reads COMMON_ORIG | `ng[]` | ncomp_tensor (cart) | **spinor component_rank** | Rys? |
|--------|--------|----------------------|------------------------|-----------------|:--:|--------|:--:|:--:|:--:|
| `int1e_spgsp` | 1724 | `c2s_si_1ei` | `cart_to_spinor_si_2di` | `G1E_R0I` (ri) + `rirj` | no | {2,1,0,0,3,4,1,3} | 3 | **3** | no |
| `int1e_spgnucsp` | 1878 | `c2s_si_1ei` | `cart_to_spinor_si_2di` | `G2E_R0I` (ri) + `rirj` | no | {2,1,0,0,3,4,0,3} | 3 | **3** | yes |
| `int1e_spgsa01` | 2036 | `c2s_si_1e` | `cart_to_spinor_si_2d` | `G2E_R0I` (ri) + `rirj` | no | {3,1,0,0,3,4,0,9} | 9 | **9** | yes |
| `int1e_cg_sa10sp` | 1141 | `c2s_si_1ei` | `cart_to_spinor_si_2di` | `G1E_RCI` (dri) | **YES** | {1,1,0,0,2,4,1,3} | 3 | **3** | no |
| `int1e_cg_sa10nucsp` | 1230 | `c2s_si_1ei` | `cart_to_spinor_si_2di` | `G2E_RCI` (dri) | **YES** | {1,1,0,0,2,4,0,3} | 3 | **3** | yes |
| `int1e_cg_sa10sa01` | 998 | `c2s_si_1e` | `cart_to_spinor_si_2d` | `G2E_RCI` (dri) | **YES** | {2,1,0,0,2,4,0,9} | 9 | **9** | yes |
| `int1e_giao_sa10sp` | 1462 | `c2s_si_1ei` | `cart_to_spinor_si_2di` | `G1E_R_I` (natural) | no | {1,1,0,0,2,4,1,3} | 3 | **3** | no |
| `int1e_giao_sa10nucsp` | 1547 | `c2s_si_1ei` | `cart_to_spinor_si_2di` | `G2E_R_I` (natural) | no | {1,1,0,0,2,4,0,3} | 3 | **3** | yes |
| `int1e_giao_sa10sa01` | 1323 | `c2s_si_1e` | `cart_to_spinor_si_2d` | `G2E_R_I` (natural) | no | {2,1,0,0,2,4,0,9} | 9 | **9** | yes |

**Transform rule (1e):** the `sp`/`nucsp` arms → `c2s_si_1ei` (imaginary, `cart_to_spinor_si_2di`); the `sa01` arms → `c2s_si_1e` (real, `cart_to_spinor_si_2d`). The `sa01` ("other-side" spin-angular) arms differ because the spin operator sits on the ket-side angular operator rather than as an imaginary ∇ phase. [VERIFIED per-family from driver last arg.]

> Note on `ng[]` slots [CITED: `cint_config.h.in:21-25`]: `ng = {i_inc, j_inc, k_inc, l_inc, GSHIFT(4), POS_E1(5), POS_E2/SLOT_RYS_ROOTS(6), TENSOR(7)}`. `ncomp_tensor = ng[7]`; cart gout stride = `ncomp_e1(ng[5]) × ncomp_tensor`. **Spinor `component_rank` = `ng[7]`** (the cart 4-block Pauli factor is reduced by the c2s transform): 3 for sp/nucsp/spgsp, 9 for all sa01. [VERIFIED against existing lock rows: `int1e_sigma_spinor` rank 3, `int1e_sp_spinor` rank 1 — the σ·p Pauli factor is reduced, leaving ng[7].]

### 2e families (intor4.c)

| Family | gout L | bra-1 c2s | bra-2 c2s | cintx symbols | builder (gauge) | reads COMMON_ORIG | `ng[]` | cart gout stride | **spinor component_rank** |
|--------|--------|-----------|-----------|---------------|-----------------|:--:|--------|:--:|:--:|
| `int2e_spgsp1` | 1384 | `c2s_si_2e1i` | `c2s_sf_2e2` | `cart_to_spinor_si_2e1i` + `cart_to_spinor_sf_2e2` | `G2E_R0I` (ri)+`rirj` | no | {2,1,0,0,3,4,1,3} | n*12 | **3** |
| `int2e_spgsp1spsp2` | 1706 | `c2s_si_2e1i` | `c2s_si_2e2` | `cart_to_spinor_si_2e1i` + `cart_to_spinor_si_2e2` | `G2E_R0I` (ri)+`rirj` | no | {2,1,1,1,5,4,4,3} | n*48 | **3** |
| `int2e_cg_sa10sp1` | 547 | `c2s_si_2e1i` | `c2s_sf_2e2` | `cart_to_spinor_si_2e1i` + `cart_to_spinor_sf_2e2` | `G2E_RCI` (dri) | **YES** | {1,1,0,0,2,4,1,3} | n*12 | **3** |
| `int2e_cg_sa10sp1spsp2` | 642 | `c2s_si_2e1i` | `c2s_si_2e2` | `cart_to_spinor_si_2e1i` + `cart_to_spinor_si_2e2` | `G2E_RCI` (dri) | **YES** | {1,1,1,1,4,4,4,3} | n*48 | **3** |
| `int2e_giao_sa10sp1` | 905 | `c2s_si_2e1i` | `c2s_sf_2e2` | `cart_to_spinor_si_2e1i` + `cart_to_spinor_sf_2e2` | `G2E_R_I` (natural) | no | {1,1,0,0,2,4,1,3} | n*12 | **3** |
| `int2e_giao_sa10sp1spsp2` | 996 | `c2s_si_2e1i` | `c2s_si_2e2` | `cart_to_spinor_si_2e1i` + `cart_to_spinor_si_2e2` | `G2E_R_I` (natural) | no | {1,1,1,1,4,4,4,3} | n*48 | **3** |

**Transform rule (2e):** bra-1 is always `c2s_si_2e1i` (imaginary GIAO σ·p on electron 1). bra-2 is `c2s_sf_2e2` for the plain (`sp1`) families (electron 2 spin-free) and `c2s_si_2e2` for the `spsp2` families (electron 2 carries σ·p too). [VERIFIED per-family from the two `c2s_*_2e*` args in each `*_spinor` driver.] All cintx symbols exist at `c2spinor.rs:1515-1597`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cart→spinor reduction | A new spinor transform | `cart_to_spinor_si_2di`/`si_2d`/`sf_2d` + 2e suite | All exist (Phases 12/28/29); KET→BRA transpose + CG coeffs already correct |
| Gauge-origin env plumbing | New env slot / setter | `with_common_origin`→`common_orig`→`PTR_COMMON_ORIG` (FND-01) | Plumbed end-to-end; validator gate exists |
| Complex `[re,im]` staging | Manual interleave | FND-03 `complex_interleaved` flag + `2×ncomp×…` sizing | `assert_flat_buffer_contract` fires; spinor layout is authoritative |
| Manifest audit fixtures | Hand-maintained dual lists | Edit `compiled_manifest.lock.json` only | build.rs regenerates `api_manifest.rs` + `.csv` from the lock; both audit sides auto-sync |
| Overlap base G-tensor / VRR-HRR / Rys | New recurrence kernel | `sigma_p_vrr_axis`/`sigma_p_hrr_axis` + Phase-29 σ·p-nuc Rys | Existing, byte-identical to libcint `g1e.c` |
| Gauge `x1i`-with-origin | A `(r-c)×s` cross-product | Port `CINTx1i_1e` recurrence: `f[i]=g[i+1]+origin·g[i]` | The gauge factor is a recurrence step, NOT a post-multiply (the single biggest correctness trap) |

**Key insight:** This phase is 90% reuse. The ONLY new device math is the `x1i`-with-origin step plus a ket-nabla and the larger gout mixes — all of which are deterministic transcriptions from `intor3.c`/`intor4.c`/`g1e.c`.

---

## Runtime State Inventory

> Not a rename/refactor/migration phase. Omit — N/A (additive family registration only). No stored data, live-service config, OS-registered state, secrets, or build artifacts carry a renamed string. The only "state" added is 15 new manifest rows + their oracle coverage flips (handled by the registration mechanics below).

---

## Common Pitfalls

### Pitfall 1: Gauge factor implemented as a post-multiply cross-product
**What goes wrong:** Implementing `g = (r − common_orig) × σ·p` as an explicit cross-product on the assembled gout instead of an `x1i`-with-origin G-tensor step → fails the vendor gate with a wrong sign/magnitude that only surfaces at atol=1e-12.
**Why it happens:** The physics reads as a cross-product; the libcint *implementation* folds it into the position recurrence.
**How to avoid:** Port `CINTx1i_1e` (`g1e.c:446-448`) literally as a device step taking `origin` from the host. **The Wave-0 micro-test exists precisely to catch this.**
**Warning signs:** Micro-test red; cg/giao arms (which share a gout but differ in builder) disagree.

### Pitfall 2: Wrong c2s transform for `sa01` arms
**What goes wrong:** Using `c2s_si_1ei` (imaginary) for the `sa01` families instead of `c2s_si_1e` (real) → byte mismatch in the re/im lane split.
**How to avoid:** Use the §Per-Family Map: `sp`/`nucsp` → `si_1ei`; `sa01` → `si_1e`. Confirmed per-family from drivers.
**Warning signs:** Real-part nonzero where vendor has imaginary-only, or vice versa.

### Pitfall 3: component_rank truncation (the GIAO g-factor raises rank)
**What goes wrong:** Registering `sa01` families at rank 3 instead of **rank 9** → the manifest truncates the trailing 6 tensor components; output silently drops data (memory `project_unstable_derivative_ports`).
**How to avoid:** Set `component_rank="9"` for ALL three `sa01` families (`spgsa01`, `cg_sa10sa01`, `giao_sa10sa01`); `="3"` for the 6 `sp`/`nucsp` arms and the 6 2e families. Cross-check against `ng[7]` in the §Per-Family Map.
**Warning signs:** Output buffer too small; only first 3 of 9 components nonzero.

### Pitfall 4: kappa≠0 spinor sizing hardcoded as 4l+2
**What goes wrong:** Sizing `ni_sp = 4l+2` for kappa≠0 shells → wrong block dims, garbage scatter.
**How to avoid:** Always `spinor_len(l, kappa)` → `2l` (LT) / `2l+2` (GT). The D-02 fixture (kappa=+1 LT × kappa=−1 GT) forces both branches.

### Pitfall 5: Missing per-inline-Spinor-arm fail-closed staging guard (Phase-28 CR-01)
**What goes wrong:** A new inline Spinor arm scatters directly and bypasses the `launch_*_pair` `BufferTooSmall` guard → silent partial write / panic mid-scatter (memory `project_spinor_dispatch_arm_needs_own_staging_guard`; Phase-29 CR-01 repeat at `one_electron.rs:10552`).
**How to avoid:** EVERY one of the 15 new arms must assert `staging.len() >= required` (full-block sized, never re-add `if dst < staging.len()` partial guards — memory `project_fnd06_chunk_staging_is_full_block`).

### Pitfall 6: OperatorId positional shift breaks hardcoded test consts
**What goes wrong:** Adding manifest rows re-points positional `OperatorId::new(N)` consts at a different family (memory `project_operator_id_shift_breaks_hardcoded_test_consts`).
**How to avoid:** After adding rows, re-grep and re-verify. The known hardcoded positional const is `int4c1e_cart` ⇒ `OperatorId::new(24)` (`resolver.rs:523-556`). The `OperatorId::new(0)` consts in `planner.rs`/`builder.rs` are dummy/zero ops, not positional — safe. Confirm `int4c1e_cart` still resolves to 24 after registration (or fix the assert to resolve by symbol name).

### Pitfall 7: CubeCL CpuRuntime FP-environment side effect on flat-atol gates
**What goes wrong:** A `#[cube]` CpuRuntime launch perturbs subsequent host f64 accumulation ~1e-11, tripping the flat atol=1e-12 gate even with a bit-identical kernel (memory `project_cubecl_cpuruntime_fp_env_side_effect`).
**How to avoid:** Suspect this BEFORE chasing kernel numerics if a family is off by ~1e-11. Mitigate by keeping the affected band host or batching launches.

### Pitfall 8: nctr>1 column/row-major coeff transpose
**What goes wrong:** libcint env coeff block is COLUMN-major (`env[ci*nprim+ip]`); cintx Shell is ROW-major. The fixture's nctr=2 shell catches a transposed copy (memory `project_raw_nctr_coeff_transpose`).
**How to avoid:** The D-02 fixture keeps an nctr>1 shell; ensure the gauge assembler's contraction loop reads coeffs in the established `coeff_i[pi*nctr_i+ci]` order (already correct in `sigma_p.rs:237`).

---

## Code Examples

### The gauge fold to port (device `x1i`-with-origin)
```c
// Source: libcint-master/src/g1e.c:446-448  (inside CINTx1i_1e, called by G1E_RCI/R0I)
fx[i] = gx[i+1] + ri[0] * gx[i];   // ri[] = origin arg: dri (cg), ri (spg-R0I)
fy[i] = gy[i+1] + ri[1] * gy[i];
fz[i] = gz[i+1] + ri[2] * gz[i];
```

### Per-family gauge-origin read (host, cintx precedent to mirror bra-side)
```rust
// Source: crates/cintx-cubecl/src/kernels/one_electron.rs:9258-9259 (Phase-26 ket-side; mirror for bra)
let origin: [f64; 3] = plan.operator_env_params.common_orig.unwrap_or([0.0; 3]);
let dri = [ri[0] - origin[0], ri[1] - origin[1], ri[2] - origin[2]];   // bra-side for cg_sa10*
```

### cg vs giao differ ONLY in the builder (gout bodies identical)
```c
// Source: intor3.c:1154-1155 (cg_sa10sp)        vs  intor3.c:1471-1472 (giao_sa10sp)
G1E_RCI(g2, g0, envs->i_l+0, envs->j_l, 0);   // cg:   origin = dri = ri - common_orig
G1E_R_I(g2, g0, envs->i_l+0, envs->j_l, 0);   // giao: origin = natural bra center (g_stride_i)
// ... the s[0..8] mix and the 12 gout[n*12+k] lines are byte-for-byte identical between them.
```

### London factor for spg* families (gout post-multiply, distinct from the gauge fold)
```c
// Source: intor3.c:1736-1741 (int1e_spgsp)
rirj[0] = envs->ri[0] - envs->rj[0];   // London phase = ri - rj  (NOT common_orig)
c[0] = 1 * rirj[0]; c[1] = 1 * rirj[1]; c[2] = 1 * rirj[2];
// ... gout[n*12+0] = + c[1]*s[17] - c[2]*s[14] - c[1]*s[25] + c[2]*s[22];  (c[] IS post-multiplied here)
```

### Vendor FFI shim model (clone for each new family)
```rust
// Source: crates/cintx-oracle/src/vendor_ffi.rs:4365-4380 (vendor_int1e_sigma_spinor)
// out sized ni_sp*nj_sp*2 (complex interleaved); call ffi::int1e_<family>_spinor(...).
// All 15 in-scope symbols are in cint_funcs.h → NO suppl-header extern decls needed. [VERIFIED]
```

---

## Combined gauge≠0 ∧ kappa≠0 Fixture (D-02)

**Templates** [CITED: `crates/cintx-oracle/src/fixtures.rs`]:
- 1e: `build_kappa_spinor_fixture` (L323) — p kappa=+1 (LT, di=2, **nctr=2**) × d kappa=−1 (GT, dj=6). Non-square. Already sets `env[PTR_RINV_ORIG]` (L342-344).
- 2e: `build_kappa_spinor_2e_fixture` (L428) — 4-shell quartet i(p,+1,nctr=2) j(d,−1) k(s,−1) l(p,−1), dims (2,6,2,4) — non-square, GT/LT mix, nctr>1.

**The D-02 extension is one line of env mutation** — set `env[PTR_COMMON_ORIG + {0,1,2}]` to a non-zero gauge origin, exactly as `PTR_RINV_ORIG` is set at `fixtures.rs:342`. [VERIFIED: `PTR_COMMON_ORIG = 1`, `crates/cintx-compat/src/raw.rs:50`.]

Recommended approach: add **new fixture functions** `build_gauge_kappa_spinor_fixture` (1e, Wave 0/1) and `build_gauge_kappa_spinor_2e_fixture` (2e, Wave 2) that internally call the Phase-29 builders and then set `env[PTR_COMMON_ORIG..+3]` — keeps the Phase-29 fixtures untouched (they're depended on by REL tests) while satisfying D-02 with a single combined fixture per arity.

**D-02 hard constraints (MUST all hold in the combined fixture):**
1. `env[PTR_COMMON_ORIG..+3] ≠ [0,0,0]` (genuine gauge origin — the `cg_sa10*` arms read it).
2. kappa≠0 GT/LT mix (at least one kappa=+1 LT shell AND one kappa=−1 GT shell → both `spinor_len` branches).
3. Non-square block dims (defeats KET→BRA transpose symmetry).
4. ≥1 shell with nctr>1 (COLUMN-major env coeff layout — catches the coeff transpose).
5. (2e form) exactly 4 spinor shells, all dims distinct.

**Claude's Discretion (suggested, not binding):** reuse the Phase-29 geometry (light-atom centers, p/d/s/p shells, exps/coeffs verbatim) and set the gauge origin to something clearly off-center and asymmetric, e.g. `[0.30, -0.45, 0.60]` (the same triple already used for `PTR_RINV_ORIG`), so the gauge≠0 path is exercised without re-tuning convergence. Coordinates/elements remain Claude's discretion subject to constraints 1-5.

---

## Wave-0 Micro-Test (D-03 de-risk, Claude's Discretion resolved)

**Recommendation: compare to a VENDORED thin family (not a hand-derived reference).**

Rationale:
- A hand-derived reference re-derives the gauge `x1i` math by hand — the exact step most likely to be wrong — so a hand-ref agreeing with the implementation only proves they share the same error.
- A vendored thin family is byte-authoritative at atol=1e-12 and isolates the gauge fold the same way the full gate does, with zero re-derivation risk. This matches the established Phase-29 "transcribe + vendor-gate" convention (D-03) and the no-spike decision.

**Which thin family:** Recommend **`int1e_cg_sa10sp`** as the micro-test vehicle — it is the smallest family that (a) reads `PTR_COMMON_ORIG` (so a wrong origin-read/sign surfaces), (b) has no Rys atom-sum (no nuclear confound), and (c) uses the primary `c2s_si_1ei` path. The micro-test:
1. Builds a **minimal** combined gauge∧kappa block (e.g. a single p(+1)×s(−1) pair with `env[PTR_COMMON_ORIG]≠0`) — small enough to debug a single wrong component.
2. Drives `int1e_cg_sa10sp_spinor` through the extended `sigma_p.rs` gauge variant.
3. Asserts byte-identity vs `vendor_int1e_cg_sa10sp_spinor` at atol=1e-12.
4. Adds a **differential check**: run the same block with `common_orig = [0,0,0]` and confirm it equals `int1e_giao_sa10sp_spinor` (which uses the natural center) — proves the `origin` plumbing is live (a no-op gauge must collapse cg→giao). This is a cheap, high-signal guard that the gauge term is actually wired, not silently zeroed.

This micro-test is the **FIRST task** (Wave 0), landing before any family is fully wired (D-04).

---

## Registration Mechanics + Landmine Checklist

**New-family surface (per v1.4 policy — NO capi/legacy):** manifest row → RawApiId/kernel arm → vendor-FFI shim → oracle test → flip `oracle_covered` spinor-only. [VERIFIED: REQUIREMENTS.md:74, CONTEXT.md.]

1. **Manifest:** add 15 spinor rows to `crates/cintx-ops/generated/compiled_manifest.lock.json` (schema_version 1; keys `id{family,operator,representation,symbol}`, `component_rank`, `complex_output:true`, `forms:["spinor"]`, `oracle_covered:false` initially, `arity` 2/4). build.rs auto-regenerates `api_manifest.rs` + `api_manifest.csv` (both audit sides). [VERIFIED: `build.rs:16,73,171,196`.] **`component_rank` per §Per-Family Map: 9 for the three sa01 families, 3 for all others.**
2. **Kernel arms:** 9 inline Spinor arms in `one_electron.rs` + 6 in `two_electron.rs`. **Each needs its own fail-closed staging guard** (Pitfall 5).
3. **Vendor FFI:** 15 `vendor_int{1,2}e_*_spinor` shims in `vendor_ffi.rs` (clone `vendor_int1e_sigma_spinor` L4365 / `vendor_int2e_spsp1_spinor` L4176). **All 15 symbols are in `cint_funcs.h` → no suppl-header extern decls.** [VERIFIED.]
4. **Oracle build:** `intor3.c` + `intor4.c` already in `crates/cintx-oracle/build.rs` (L61-62, L228-229). **`gaunt1.c` is already wired too** (L237, from Phase-29 REL-04) but the Phase-30 in-scope families do NOT use it — no oracle build change this phase. [VERIFIED.]
5. **Coverage flip:** `xtask/src/oracle_covered_update.rs` flips `oracle_covered=true` spinor-only after the gate is green; the SC#4 skipped-fixture-flip-refusal guard prevents flipping a skipped family.

**Landmine application (every CONTEXT.md Integration Point, concretely):**

| Landmine | Application to Phase 30 |
|----------|------------------------|
| (a) component_rank truncation | sa01 families = **rank 9**; sp/nucsp/spgsp/all-2e = **rank 3**. Verify each against §Per-Family Map `ng[7]` before committing the lock. |
| (b) OperatorId positional shift | Re-grep after adding rows; the one hardcoded positional const is `int4c1e_cart ⇒ OperatorId::new(24)` (`resolver.rs:556`). Confirm it still resolves to 24 (rows are typically appended; if it shifts, fix the assert to resolve by symbol name). `OperatorId::new(0)` in planner/builder are dummy ops — unaffected. |
| (c) per-inline-Spinor-arm staging guard | All 15 arms: assert `staging.len() >= required` (full-block), no partial `if dst < staging.len()` guards. |
| (d) KET→BRA transpose | Owned inside `cart_to_spinor_si_2di`/`si_2d` (`c2spinor.rs:793` Stage 0). Emit gc blocks KET-major component-leading (as `sigma_p.rs:281` already does); the transform owns the flip. |
| (e) nctr coeff transpose | D-02 fixture keeps the nctr=2 shell; contraction reads `coeff_i[pi*nctr_i+ci]` (`sigma_p.rs:237`). |
| (f) CpuRuntime FP side effect | If a family is off by ~1e-11 on the flat atol=1e-12 gate, suspect the launch FP-env side effect first (Pitfall 7), not the kernel math. |

---

## Phase-31 Boundary Confirmation

[VERIFIED]:
- **Gaunt GIAO families** `int2e_cg_ssa10ssp2` / `int2e_giao_ssa10ssp2` are defined ONLY in `libcint-master/src/autocode/gaunt1.c` (gouts L411, L575) — distinct symbols from the in-scope `int2e_cg_sa10sp1` (intor4.c). Phase 31 / BREIT-03.
- **Glob exclusion:** GIAO-03's `int2e_cg_sa10*` matches `int2e_cg_sa10sp1` / `int2e_cg_sa10sp1spsp2` (`cint_funcs.h:488-497`) but NOT `ssa10ssp2` (the `ss` prefix is a different token). GIAO-03 is **fully closed by Phase 30** with no spillover.
- **No `launch_breit` in scope:** all 6 in-scope 2e families use `CINT2e_spinor_drv` (NOT a breit driver). [VERIFIED per-family from intor4.c.] No in-scope family secretly needs `launch_breit`.

---

## Validation Architecture

> nyquist_validation is enabled (config.json: `workflow.nyquist_validation: true`).

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (cargo test), oracle integration tests under `crates/cintx-oracle/tests/` |
| Config file | none (cargo default; no nextest.toml present) |
| Quick run command | `cargo test --features cpu -p cintx-oracle <test_name> -- --nocapture` |
| Full suite command | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --features cpu -p cintx-oracle` |

**Double-gate note (memory `reference_oracle_vendor_parity_invocation`):** real vendor byte-identity requires BOTH `--features cpu` AND env `CINTX_ORACLE_BUILD_VENDOR=1`. Without both, vendor bodies compile out and parity SILENTLY skips → the NO-SILENT-SKIP assertion (`test_no_silent_skip`) must FAIL (not skip) when the vendor arm did not run.

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GIAO-03 (Wave 0) | gauge-gout byte-identity micro-test (cg_sa10sp vs vendor; + cg→giao collapse at origin=0) | integration (vendor) | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --features cpu -p cintx-oracle giao_sigma_micro` | ❌ Wave 0 |
| GIAO-03 (Wave 1) | 9 1e families byte-identical on combined gauge∧kappa fixture | integration (vendor) | `... cargo test --features cpu -p cintx-oracle giao_sigma_1e_parity` | ❌ Wave 0/1 |
| GIAO-03 (Wave 2) | 6 2e families byte-identical on 4-shell gauge∧kappa fixture | integration (vendor) | `... cargo test --features cpu -p cintx-oracle giao_sigma_2e_parity` | ❌ Wave 2 |
| GIAO-03 (gate) | manifest-audit green, all 15 rows oracle_covered=true spinor-only | xtask | `cargo run -p xtask -- manifest-audit` | ✅ exists |

**Model test file:** `crates/cintx-oracle/tests/rel_1e_sigma_parity.rs` (Phase-29) — copy its RED-scaffold structure: per-family vendor collectors, `collect_cintx_<fam>` arms, `test_kappa_sizing_non_4l_plus_2` sizing guard, `test_no_silent_skip` integrity assertion, `#![cfg(any(feature="cpu",feature="rocm"))]`, per-family `#[ignore]` until wired.

### Sampling Rate
- **Per task commit:** `cargo test --features cpu -p cintx-oracle <touched_test> -- --nocapture` (determinism + sizing guards; fast).
- **Per wave merge:** `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --features cpu -p cintx-oracle giao_sigma_*` (full vendor byte-identity for that wave's families).
- **Phase gate:** `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --features cpu -p cintx-oracle` green + `manifest-audit` green before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `crates/cintx-oracle/tests/giao_sigma_1e_parity.rs` — covers GIAO-03 (1e, Waves 0+1)
- [ ] `crates/cintx-oracle/tests/giao_sigma_2e_parity.rs` — covers GIAO-03 (2e, Wave 2)
- [ ] `build_gauge_kappa_spinor_fixture` + `build_gauge_kappa_spinor_2e_fixture` in `fixtures.rs` (D-02)
- [ ] 15 `vendor_int{1,2}e_*_spinor` shims in `vendor_ffi.rs`
- [ ] (no framework install — cargo test built-in covers all requirements)

---

## Security Domain

> `security_enforcement` is not set in config.json (treat as enabled), but this is a pure numeric-integral library phase with no auth/session/network/user-input surface. The only applicable ASVS category is V5 (Input Validation), already enforced by the FND-01 validator gate and the fail-closed staging guards.

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | `validate_common_orig_env_params` (PTR_COMMON_ORIG finiteness gate, `validator.rs:210-217`); fail-closed `BufferTooSmall`/`UnsupportedApi` on out-of-envelope shells |
| V6 Cryptography | no | — |

| Threat pattern | STRIDE | Mitigation |
|----------------|--------|------------|
| Out-of-envelope shell → silent truncation/partial write | Tampering (data integrity) | Fail-closed staging assert (no partial writes); rank/nroots `UnsupportedApi` guards |
| Non-finite gauge origin in env | Tampering | FND-01 validator finiteness gate |

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Spin-free GIAO ket-side gauge fold (`drj=rj-common_orig`, Phase 26) | Bra-side gauge fold (`dri=ri-common_orig`) combined with σ·p | Phase 30 | The Phase-26 `one_electron.rs:9258` pattern is the direct precedent; flip to bra side + add ket nabla |
| `int1e_sp`-only σ·p assembler (`tensor_rank=1`, Phase 28) | gauge-origin-parameterized σ·p variants (sp/nucsp/sa01, 2e) | Phase 30 | Extend `sigma_p.rs` rather than fork |

**Deprecated/outdated:** none relevant.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| — | (empty) | — | All load-bearing claims (gauge fold structure, per-family c2s pairing, component ranks, common_orig classification, glob exclusion, no-launch_breit, cint_funcs.h presence, fixture extension point, manifest schema) were VERIFIED by reading libcint and cintx source this session. |

**No `[ASSUMED]` claims** — no user confirmation needed before planning. The two Claude's-Discretion recommendations (extend sigma_p.rs; vendor thin-family micro-test) are reasoned recommendations the planner/user may override, not assumed facts.

---

## Open Questions

1. **Exact gout-mix transcription for the sa01 families (s[0..8]→n*36, 36 components).**
   - What we know: builders, origin class, transform, and rank (9) are all verified; the gout bodies are at the cited line numbers.
   - What's unclear: the full 36-line gout component mix for each sa01 family is long; only the first ~24 lines were sampled here.
   - Recommendation: the planner transcribes the full gout bodies verbatim from `intor3.c` L998/L1323/L2036 (cg/giao/spg sa01) at plan-authoring time. The structure (s[i] products + signed component sums) is mechanical; no decision is needed, only careful copying.

2. **Whether the σ·p-nuc Rys path (Phase-29) is directly reusable for the `nucsp`/`sa01` Rys atom-sum, or needs a headroom bump.**
   - What we know: the nucsp/sa01 gouts use G2E_* macros with a Rys roots loop (`nrys_roots`); the `ng[]` headroom differs per family (e.g. spgsa01 i_inc=3).
   - What's unclear: whether the existing device Rys nroots envelope covers the bra+ket headroom of the highest-l corpus shell.
   - Recommendation: reuse the Phase-29 σ·p-nuc Rys path; add a fail-closed `nroots > MAX_DEVICE_NROOTS → UnsupportedApi` guard (mirroring `one_electron.rs:9340`) so an out-of-envelope shell fails closed rather than truncating. FND-02 Wheeler nroots≥6 fallback already exists if needed.

---

## Environment Availability

> Pure code/config phase. No external tools, services, or runtimes beyond the project's own Rust/CubeCL/vendored-libcint stack (all already present and exercised by Phases 28/29). Step 2.6: SKIPPED (no new external dependencies). The vendored libcint at `libcint-master/` is already compiled by `crates/cintx-oracle/build.rs` and `intor3.c`/`intor4.c` are already in the build.

---

## Sources

### Primary (HIGH confidence)
- `libcint-master/src/autocode/intor3.c` — 1e GIAO×σ drivers + gouts (L998, L1141, L1230, L1323, L1462, L1547, L1724, L1878, L2036); per-family `c2s_*` spinor_drv args; `dri`/`PTR_COMMON_ORIG` reads.
- `libcint-master/src/autocode/intor4.c` — 2e drivers + gouts (L547, L642, L905, L996, L1384, L1706); 2e c2s pairings; `CINT2e_spinor_drv`.
- `libcint-master/src/g1e.c:429-451` — `CINTx1i_1e` (the gauge `x1i`-with-origin recurrence).
- `libcint-master/src/g1e.h:48-62`, `g2e.h:93-104` — `G1E_RCI`/`R0I`/`R_I` (+ G2E) macro definitions.
- `libcint-master/src/cint_config.h.in:21-25` — `ng[]` slot constants (GSHIFT/POS_E1/POS_E2/SLOT_RYS_ROOTS/TENSOR).
- `libcint-master/src/autocode/gaunt1.c:411,575` — Phase-31 Gaunt GIAO families (boundary).
- `libcint-master/include/cint_funcs.h:488-501` — all 15 in-scope symbols present (no suppl-header).
- `crates/cintx-cubecl/src/kernels/sigma_p.rs` — σ·p assembler (extension target).
- `crates/cintx-cubecl/src/transform/c2spinor.rs:25,531,640,754,1515-1597` — transforms (reused).
- `crates/cintx-cubecl/src/kernels/one_electron.rs:9258` — Phase-26 ket-side gauge precedent.
- `crates/cintx-oracle/src/fixtures.rs:323,428` — kappa fixtures (D-02 templates).
- `crates/cintx-oracle/src/vendor_ffi.rs:4176,4365` — vendor shim models.
- `crates/cintx-ops/generated/compiled_manifest.lock.json` + `crates/cintx-ops/build.rs:16,73,171` — manifest schema + auto-sync.
- `crates/cintx-compat/src/raw.rs:50` — `PTR_COMMON_ORIG = 1`.
- `crates/cintx-oracle/build.rs:61-62,228-237` — intor3/intor4/gaunt1 wiring.
- `crates/cintx-oracle/tests/rel_1e_sigma_parity.rs` — Phase-29 test scaffold model.

### Secondary (MEDIUM confidence)
- Project memory entries: `project_phase29_group4_rel_sigma`, `project_spinor_dispatch_arm_needs_own_staging_guard`, `project_operator_id_shift_breaks_hardcoded_test_consts`, `project_cubecl_cpuruntime_fp_env_side_effect`, `project_raw_nctr_coeff_transpose`, `reference_oracle_vendor_parity_invocation`, `project_1e_gpu_port_scalar_only`.

### Tertiary (LOW confidence)
- none — all claims verified against source.

---

## Metadata

**Confidence breakdown:**
- Gauge fold structure (the new work): HIGH — transcribed from `g1e.c:446-448` + per-family `dri`/builder grep; cg-vs-giao gout identity verified line-for-line.
- Per-family c2s transform map: HIGH — read each driver's spinor_drv last arg individually (no analogy).
- Component ranks: HIGH — `ng[7]` per family + cross-checked against existing lock rows (sigma=3, sp=1).
- Phase boundary (Gaunt/launch_breit/glob): HIGH — verified gaunt1.c symbols, cint_funcs.h presence, CINT2e_spinor_drv per family.
- Fixture/registration mechanics: HIGH — read templates, lock schema, build.rs auto-sync.
- sa01 full 36-component gout mix: MEDIUM — structure verified, full transcription deferred to plan authoring (Open Q1).

**Research date:** 2026-06-01
**Valid until:** stable (vendored libcint 6.1.3 is pinned; cintx internal — no external version drift). Re-verify only if `sigma_p.rs`, `c2spinor.rs`, or the manifest lock schema change before planning.
