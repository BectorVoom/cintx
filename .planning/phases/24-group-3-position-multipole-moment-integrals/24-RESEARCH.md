# Phase 24: Group 3 — Position / Multipole-Moment Integrals - Research

**Researched:** 2026-05-30
**Domain:** libcint 6.1.3 position/multipole-moment 1e integral families (cart+sph byte-identity); CubeCL `#[cube]` kernel port; vendor-FFI parity
**Confidence:** HIGH (all gout orders, component ranks, ng[] headroom, and origin-source branches derived VERBATIM from the vendored libcint 6.1.3 source `libcint-master/src/autocode/intor1.c` + `src/g1e.{c,h}` + `src/cint1e.c`)

## Summary

Every Phase-24 family is **new to the manifest** (verified absent — see Standard Stack). The libcint 6.1.3 autocode for all of them lives in **one file already compiled into the vendor build**: `libcint-master/src/autocode/intor1.c` (`src/autocode/intor1.c` is already in `cintx-oracle/build.rs`'s source list). So the vendor-FFI recipe step adds only an `allowlist_function` regex alternation + safe wrappers — **no new `.c` source to register** `[VERIFIED: cintx-oracle/build.rs:51]`.

The families split cleanly by the libcint `int1e_type` driver flag and by which G1E builder the gout emitter uses, which maps 1:1 onto CONTEXT's Cluster A/B/C/D:
- **Cluster A (overlap G-tensor × position-power, `int1e_type=0`):** `r`, `rr`, `rrr`, `rrrr`, `r2`, `r4`, `z`, `zz` — all use `G1E_RCJ` (position relative to common origin, `drj = rj - env[PTR_COMMON_ORIG]`) `[VERIFIED: intor1.c:1021-1023,1143]`. The `_origj` variants use `G1E_R_J` (origin = ket basis center `rj` directly, no `drj`) — a **distinct vendor symbol per variant** `[VERIFIED: intor1.c:2067-2089]`.
- **Cluster B (Rys, `int1e_type=1`):** `rinv`, `drinv` — single-center `1/r`, **no atom-sum, charge=+1**, origin = **`env[PTR_RINV_ORIG]` (env[4..6]), NOT `PTR_COMMON_ORIG`** `[VERIFIED: g1e.c:226-228, cint1e.c:293-294]`. This **corrects CONTEXT D-04's assumption** (see Open Question OQ-1; the Rust plumbing for `rinv_orig` already exists from Phase 21).
- **Cluster C (`p4`, overlap-derivative, `int1e_type=0`, no Rys):** `∇⁴`, rank 1 `[VERIFIED: intor1.c:2533-2537]`, headroom `ng={2,2,...}` (bra **and** ket each +2).
- **Cluster D (`irp`, overlap-derivative, `int1e_type=0`, no Rys):** `i·r×∇`, **rank 9** (a 3×3 r⊗∇ tensor, NOT a smaller rank) `[VERIFIED: intor1.c:2798-2816]`, reads `env[PTR_COMMON_ORIG]` via `G1E_RCJ` so it is a gauge-origin family too.

**Primary recommendation:** Clone the proven Phase-23 5-step registration recipe per family; for Cluster A build ONE parameterized moment `#[cube]` kernel branching on origin-source (`PTR_COMMON_ORIG` vs ket `rj`) and emitting components in the **verbatim gout order tabulated below**; for Cluster B reuse the existing `iprinv`/nuclear-Rys kernel with single rinv origin + factor +1.0 (no `-Z_C`, no atom-loop) reading `rinv_orig`; for C/D reuse the overlap-derivative engine. Gate every family with a NON-SQUARE bra×ket block on `build_h2o_sto3g_common_orig` (and additionally set `rinv_orig` for B). **Resolve OQ-1 before locking the rinv origin slot.**

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Moment G-tensor evaluation (r/rr/rrr/rrrr/r2/r4/z/zz, +_origj) | CubeCL `#[cube]` device kernel (`cintx-cubecl`) | — | Per CLAUDE.md CubeCL is the primary compute backend; this extends the existing overlap engine |
| rinv/drinv single-center Coulomb (Rys) | CubeCL `#[cube]` device kernel | — | Reuses the existing nuclear/iprinv Rys path |
| p4 / irp (overlap-derivative) | CubeCL `#[cube]` device kernel | — | Reuses overlap-derivative machinery (no Rys) |
| Operator dispatch + component sizing | Host (`cintx-compat/raw.rs`, `cintx-runtime/planner.rs`) | — | Planning/marshaling stays on host per CLAUDE.md |
| Gauge-origin / rinv-origin env-slot read + finiteness validation | Host (`cintx-compat/raw.rs` `eval_raw`, `cintx-runtime/validator.rs`) | — | Already plumbed (Phase 21 rinv_orig, Phase 22 common_orig) |
| Vendor byte-identity parity (oracle gate) | Host test/oracle glue (`cintx-oracle`) | — | Test/oracle glue per CLAUDE.md |

## Project Constraints (from CLAUDE.md)

- **CubeCL is the primary compute backend** — moment kernels are `#[cube]` device kernels generic over `F`; host CPU work limited to planning/validation/marshaling/oracle glue.
- **libcint 6.1.3 result compatibility is the gate** — verified vendored version is exactly 6.1.3 `[VERIFIED: libcint-master/CMakeLists.txt cint_VERSION 6.1.3.0]`.
- **`thiserror` v2 for public library errors; `anyhow` for oracle/xtask/bench** — kernel/dispatch errors return `cintxRsError` variants (e.g. `UnsupportedApi`, `InvalidEnvParam`).
- **`cargo --locked` / pinned toolchain** — manifest lock edits must be deterministic; `cargo build -p cintx-ops` regenerates `api_manifest.{rs,csv}` from the lock.
- **GSD workflow enforcement** — edits go through a GSD command (execute-phase). Not a research concern but binds the planner.
- **`#[cube]` authoring manuals are authoritative** (`docs/manual/Cubecl/*.md`): no plain-fn calls inside `#[cube]`, no `if`-as-expression, use `F::exp`/`F::sqrt`, u32/i32 indices only, no `continue`/`break`. Any new kernel obeys these (the existing one_electron kernels already do — clone their style).

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Split into PLAN.md clusters by shared operator construction, low-rank first. Cluster A (overlap-derived position tensors `r,rr,rrr,rrrr,r2,r4,z,zz`) via ONE parameterized moment kernel, ket headroom `ng[1]=1..4`; `_origj` variants land alongside their base in Cluster A. Cluster B (`rinv,drinv`, Rys). Cluster C (`p4`, overlap-derivative, no Rys). Cluster D (`irp`, i·r×∇, overlap-derivative). Sequence A → then B/C/D (parallelizable; worktrees on).
- **D-02:** Each `_origj` family is its OWN manifest operator / RawApiId (mirrors libcint's symbol-per-variant set). The shared kernel branches on origin-source: `env[PTR_COMMON_ORIG]` (base) vs the ket shell-j coordinate (`_origj`). Do NOT collapse to a single operator + origin-mode descriptor flag.
- **D-03:** Keep the Phase 24↔25 boundary — use existing staging as-is for rank-81 parity. `parse_component_multiplier` already sizes staging for any `component_rank`; output is complete/correct at rank 81. FND-06 OOM-safety hardening stays Phase 25. Cross-link the dependency in the plan.
- **D-04:** Reuse existing kernels; fail-closed above the device Rys ceiling (`nroots>5`, Phase 23 D-13 precedent). Plain `int1e_rinv` = the nuc Rys kernel at the common origin (env[PTR_COMMON_ORIG]) with charge=1 and NO atom-sum; `int1e_drinv` = its derivative (+1 Rys root). `p4`/`irp` reuse the overlap-derivative engine (no Rys). Corpus (H2O/STO-3G ≤d) does not hit the ceiling.
  - **⚠️ RESEARCH CORRECTION (OQ-1):** libcint 6.1.3 source proves plain `int1e_rinv`/`int1e_drinv` read **`env[PTR_RINV_ORIG]` (env[4..6]), NOT `PTR_COMMON_ORIG`**, with charge=+1, no atom-sum. See Open Questions OQ-1 — this needs user/planner confirmation before locking the origin slot.
- **D-05 (registration recipe, Phase 23 D-11):** 5 steps — (1) manifest lock entry cloning closest family with `component_rank` = true output multiplier, then `cargo build -p cintx-ops` regenerates `api_manifest.{rs,csv}`; (2) `RawApiId` consts in `cintx-compat/src/raw.rs`; (3) launcher dispatch on `descriptor.operator_name()`; (4) vendor FFI — add cart/sph symbols to bindgen `allowlist_function` regex in `cintx-oracle/build.rs` + safe wrappers in `vendor_ffi.rs` (confirm autocode `.c` in build source list); (5) `vendor_*` parity test. Lock edits auto-sync `manifest-audit` — NO separate fixtures family list.
- **D-06 (gauge-origin plumbing READY, Phase 22):** `env[1..3]` read unconditionally in `eval_raw` (`raw.rs:674-686`), `.with_common_origin([x,y,z])` exists, finiteness validator runs on both paths, `common_orig==None` defaults to `[0,0,0]`. Phase 24 is the FIRST kernel consumer. `PTR_COMMON_ORIG=1` (`raw.rs:50`).
- **D-07 (transpose discipline, Phase 23 D-05 + ROADMAP SC1):** raise headroom on the KET (`ng[1]`), not the bra; copy each family's component order VERBATIM from the libcint gout index map; gate every family with a NON-SQUARE bra×ket block (e.g. p×d). A square block is transpose-symmetric and hides the bug.
- **D-08 (component-rank-truncation hard rule, Phase 23 D-14):** a `component_rank` set too LOW silently TRUNCATES trailing output components. Each family's `component_rank` MUST equal its true output multiplier (`r`=3, `rr`=9, `rrr`=27, `rrrr`=81, `r2`=1, `r4`=1, `z`=1, `zz`=1, `p4`=1, `irp`=… derive from source).
- **D-09 (surface scope, Phase 23 D-09):** manifest + RawApiId + kernel + vendor-FFI + oracle only. No `capi` enum variants, no legacy `cint*` wrappers. Spinor reps registered → `UnsupportedApi`.
- **D-10 (verification, Phase 23 D-10):** per-family byte-identity at atol=1e-12 vs vendored libcint 6.1.3, cart+sph, every component, in `vendor_*` parity tests double-gated on `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1` (without both, parity silently skips).

### Claude's Discretion
- Exact `component_rank` value and libcint gout component order per family (derived from libcint source by researcher/planner — DONE below in Architecture Patterns).
- Whether Cluster A's parameterized moment kernel is one `#[cube]` entry with a comptime moment order or a small family of order-specialized launchers — implementer's call, as long as D-07 holds.
- The precise corpus shell-tuple selection for each `vendor_*` test (subject to the non-square bra×ket requirement of D-07).

### Deferred Ideas (OUT OF SCOPE)
- **FND-06 fail-closed high-rank (rank-81) staging refactor + OOM re-validation** → Phase 25. Phase 24 uses existing staging as-is for parity (D-03).
- **nroots≥6 Wheeler/Jacobi fallback** (FND-02) → Phase 25. Phase 24 fail-closes above the nroots≤5 ceiling (D-04); folded as a cross-link only.
- **Spinor moment representations** → land when a consumer needs them; registered → `UnsupportedApi` this phase (D-09).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| MOM-01 | Dipole `int1e_r` (and `int1e_r_origj`) match at atol=1e-12 against a non-zero gauge-origin fixture (cart+sph). | `int1e_r` gout order + `G1E_RCJ`/`drj=rj-env[PTR_COMMON_ORIG]` and `int1e_r_origj` `G1E_R_J` (ket center) construction derived below; rank=3; ng[1]=1; fixture `build_h2o_sto3g_common_orig`. |
| MOM-02 | `int1e_rr`, `int1e_r2`, `int1e_z`, `int1e_zz` (and `_origj`) match at atol=1e-12 (cart+sph). | `rr` rank 9 (full order tabulated), `r2` rank 1 (trace s0+s4+s8), `z` rank 1 (z only), `zz` rank 1; ng tuples below. `_origj` per-symbol. |
| MOM-03 | `int1e_rrr`, `int1e_rrrr`, `int1e_r4` (octupole/hexadecapole) match at atol=1e-12 (cart+sph), ket headroom from `ng[1]`. | `rrr` rank 27, `rrrr` rank 81 (ket headroom ng[1]=3/4); `r4` rank 1 (specific s-index contraction). Orders tabulated. |
| MOM-04 | `int1e_p4`, `int1e_drinv`, plain `int1e_rinv`, `int1e_irp` match at atol=1e-12 (cart+sph). | `p4` rank 1 (∇⁴, ng={2,2,...}), `rinv` rank 1 (Rys, single-center charge=1, env[PTR_RINV_ORIG]), `drinv` rank 3 (+derivative), `irp` rank 9 (3×3 r⊗∇). |
</phase_requirements>

---

## Standard Stack

This is an internal-port phase; the "stack" is the existing cintx workspace + the vendored libcint 6.1.3 oracle. No new external dependencies.

### Core
| Component | Version | Purpose | Why Standard |
|-----------|---------|---------|--------------|
| Vendored libcint | **6.1.3** (`libcint-master/`) | Oracle source of truth for gout orders, ranks, ng[] and byte-identity parity | `[VERIFIED: CMakeLists.txt cint_VERSION_MAJOR/MINOR/PATCH = 6/1/3]` |
| `cintx-cubecl` | workspace | `#[cube]` device moment/Rys kernels generic over `F` | Reuses `one_electron.rs` overlap + nuclear-Rys engine |
| `cintx-compat` | workspace | `RawApiId` consts + `eval_raw` dispatch + env-slot reads | `PTR_COMMON_ORIG=1`, `PTR_RINV_ORIG=4` already defined `[VERIFIED: raw.rs:50,58]` |
| `cintx-runtime` | workspace | `parse_component_multiplier` staging sizing; `OperatorEnvParams.{common_orig,rinv_orig}`; validators | `[VERIFIED: planner.rs:50-54,403; validator.rs:210]` |
| `cintx-ops` | workspace | `compiled_manifest.lock.json` → regenerated `api_manifest.{rs,csv}` | Lock has 199 entries, schema_version 1; manifest-audit auto-syncs from lock `[VERIFIED: compiled_manifest.lock.json]` |
| `cintx-oracle` | workspace | bindgen vendor FFI + `build_h2o_sto3g_common_orig` fixture + `vendor_*` parity tests | `[VERIFIED: fixtures.rs:152-176, build.rs:51,358]` |

### Verified: all target families ABSENT from the manifest (all new)
`[VERIFIED: grep of compiled_manifest.lock.json id.symbol]` — `int1e_r_cart`, `int1e_rr_cart`, `int1e_rrr_cart`, `int1e_rrrr_cart`, `int1e_r2_cart`, `int1e_r4_cart`, `int1e_z_cart`, `int1e_zz_cart`, `int1e_p4_cart`, `int1e_irp_cart`, `int1e_rinv_cart`, `int1e_drinv_cart`, `int1e_r_origj_cart` are all **absent**.

**Distinguish from existing families:** `int1e_r2_origi_sph`/`int1e_r4_origi_sph` (and `_ip2`) ALREADY EXIST (the `unstable` origi families, origin at the **bra/i** center) `[VERIFIED: manifest grep]`. Phase 24's `int1e_r2`/`int1e_r4` are the **common-origin** variants (`G1E_RCJ`, `drj=rj-env[PTR_COMMON_ORIG]`) — a different operator. Do NOT confuse `_origi` (existing) with this phase's base or `_origj` (new).

**Installation:** none — internal workspace. Build/test commands in Validation Architecture.

## Architecture Patterns

### System Architecture Diagram

```
 caller (.with_common_origin / .with_rinv_origin)
        │  atm/bas/env  (env[1..3]=PTR_COMMON_ORIG, env[4..6]=PTR_RINV_ORIG)
        ▼
 cintx-compat::eval_raw  ─── reads env slots ──► OperatorEnvParams{common_orig, rinv_orig}
        │                    validate_common_orig_env_params / validate_rinv_orig_env_params
        │  descriptor.operator_name()  + component_rank (string) → parse_component_multiplier
        ▼
 launcher dispatch (raw.rs:612-822)  ── is_<op> ladder / op_kind ──►
        ▼
 cintx-cubecl::launch_one_electron_typed  (operator dispatch ~:3765)
        │
        ├── Cluster A (int1e_type=0, overlap): moment kernel
        │      origin-source branch:  common_orig (base)  |  ket rj (_origj)
        │      build overlap G-tensor; apply position-power via ket-headroom levels (ng[1])
        │      emit components in VERBATIM gout order (see tables)
        │
        ├── Cluster B (int1e_type=1, Rys): rinv/drinv kernel
        │      single center = rinv_orig (env[4..6]); charge=+1; NO atom-loop
        │      drinv: + (D_i+D_j) derivative level, +1 effective root
        │
        ├── Cluster C (int1e_type=0): p4 kernel  (D_i² applied to D_j² block, rank 1)
        └── Cluster D (int1e_type=0): irp kernel (D_j on r-level block, rank 9)
        ▼
 staging scatter (planner sizes from component_rank, rank≤81)  →  cart out
        ▼
 cart→sph (c2s) per component  →  out buffer (component-outer, AO-block-inner)
        ▼
 cintx-oracle vendor_* test:  cintx out  ==(atol=1e-12)==  vendor int1e_*_{cart,sph}
                              on build_h2o_sto3g_common_orig (NON-SQUARE bra×ket block)
```

### libcint operator construction — the source-of-truth tables

All facts below are `[VERIFIED: libcint-master/src/autocode/intor1.c]` at the cited line. The G1E builder macros are `[VERIFIED: src/g1e.h:52-63]`:
- `G1E_RCJ(f,g,...) = CINTx1j_1e(f,g,drj,...)` where `drj = rj - env[PTR_COMMON_ORIG]` → **position relative to the COMMON (gauge) origin**, applied on the **ket (j)** index.
- `G1E_R_J(f,g,...) = (f = g + g_stride_j)` → **position relative to the ket BASIS CENTER `rj`** (no origin subtraction). Used ONLY by the `_origj` emitters.
- `G1E_D_I` / `G1E_D_J = CINTnabla1{i,j}_1e` → nabla (∇) on bra/ket. Used by p4, irp.
- `int1e_type` driver flag (`cint1e.c:288-303`): `0`=overlap `CINTg1e_ovlp`; `1`=`CINTg1e_nuc(g,envs,-1)` single-center rinv; `2`=`CINTg1e_nuc` atom-summed nuclear.

#### Component ranks, ng[] headroom, origin source (ALL families)

| Family | `component_rank` | ng[] `{i,j,k,l,nf,?,?,comp}` | int1e_type | Origin / G1E builder | Notes |
|--------|:---:|---|:---:|---|---|
| `int1e_r`     | **3**  | `{0,1,0,0,1,1,1,3}` | 0 | `G1E_RCJ`, drj=rj−COMMON_ORIG | dipole; ket headroom +1 `[intor1.c:1161]` |
| `int1e_rr`    | **9**  | `{0,2,0,0,2,1,1,9}` | 0 | `G1E_RCJ` (×2 levels) | ket headroom +2 `[intor1.c:1465]` |
| `int1e_rrr`   | **27** | `{0,3,0,0,3,1,1,27}`| 0 | `G1E_RCJ` (×3) | ket headroom +3 |
| `int1e_rrrr`  | **81** | `{0,4,0,0,4,1,1,81}`| 0 | `G1E_RCJ` (×4) | ket headroom +4; **rank 81** |
| `int1e_r2`    | **1**  | `{0,2,0,0,2,1,1,1}` | 0 | `G1E_RCJ` | trace `s0+s4+s8` `[intor1.c:1220,1225]` |
| `int1e_r4`    | **1**  | `{0,4,0,0,4,1,1,1}` | 0 | `G1E_RCJ` | `s0+2·s4+2·s8+s40+2·s44+s80` `[intor1.c:1380]` |
| `int1e_z`     | **1**  | `{0,1,0,0,1,1,1,1}` | 0 | `G1E_RCJ` | z-axis only: emits `s[2]` `[intor1.c:1034,1039]` |
| `int1e_zz`    | **1**  | `{0,2,0,0,2,1,1,1}` | 0 | `G1E_RCJ` | zz only: emits `s[8]` `[intor1.c:1098,1103]` |
| `int1e_p4`    | **1**  | `{2,2,0,0,4,1,1,1}` | 0 | `G1E_D_I`/`G1E_D_J` | ∇⁴; **BRA+KET each +2**; Laplacian-of-Laplacian contraction `s0+2s4+2s8+s40+2s44+s80` `[intor1.c:2533,2539]` |
| `int1e_irp`   | **9**  | `{0,2,0,0,2,1,1,9}` | 0 | `G1E_D_J`+`G1E_RCJ` | 3×3 r⊗∇ tensor; reads COMMON_ORIG; ket +2 `[intor1.c:2798,2819]` |
| `int1e_rinv`  | **1**  | `{0,0,0,0,0,1,0,1}` | **1** | `CINTg1e_nuc(-1)`, center=**RINV_ORIG**, charge=+1, no atom-sum | Rys; `s += g0[ix+i]*g0[iy+i]*g0[iz+i]` over nrys_roots `[intor1.c:3621-3642]` |
| `int1e_drinv` | **3**  | `{1,1,0,0,1,1,0,3}` | **1** | `CINTg1e_nuc(-1)` + `(G2E_D_I+G2E_D_J)` | gradient wrt rinv center (transl. invariance); +derivative level `[intor1.c:3671-3702]` |
| `int1e_r_origj`  | **3** | `{0,1,0,0,1,1,1,3}` | 0 | `G1E_R_J`, origin=ket `rj` (no drj) | own symbol `[intor1.c:2067-2092]` |
| `int1e_z_origj`  | **1** | `{0,1,0,0,1,1,1,1}` | 0 | `G1E_R_J` | own symbol `[intor1.c:1957-2005]` |
| `int1e_zz_origj` | **1** | `{0,2,0,0,2,1,1,1}` | 0 | `G1E_R_J` | own symbol `[intor1.c:2007-2065]` |
| `int1e_rr_origj` / `r2_origj` / `r4_origj` | same as base | same ng as base | 0 | `G1E_R_J` | own symbol each (mirror base, origin=ket center) |

> **CORRECTION (plan-checker W2, source-confirmed):** the complete `_origj` symbol set in libcint 6.1.3 `intor1.c` is exactly **`{r, rr, r2, r4, z, zz}_origj`** — 6 symbols. **`rrr_origj` and `rrrr_origj` do NOT exist** (no vendor symbol → no parity target). Do NOT register them. (An earlier draft of this row incorrectly listed `rrr_origj`/`rrrr_origj`; the plans correctly exclude them.)

> **`component_rank` is a string** parsed by `parse_component_multiplier`, which **multiplies all numeric segments**. So set it to the literal multiplier: `"3"`, `"9"`, `"27"`, `"81"`, `"1"` `[VERIFIED: planner.rs:403-450]`. (Do NOT write e.g. `"3x3"` expecting 9 unless you intend the product — `"9"` is unambiguous.)

#### Verbatim gout component order (the D-07 source-of-truth)

For tensor families libcint writes `gout[n*RANK + k]` — **AO-pair index `n` is the OUTER stride, component `k` is the INNER (fastest-varying) stride** `[VERIFIED: intor1.c:1152-1158 (r), :1444-1462 (rr), :rrr n*27 block]`.

- **`int1e_r` (rank 3):** `k=0→x (s0=g1ₓ·g0·g0)`, `k=1→y`, `k=2→z`. Order = (x,y,z) `[VERIFIED: intor1.c:1148-1158]`.
- **`int1e_rr` (rank 9):** the 9 emitted components in order are the s-indices `s0..s8` = the moment-power pairs:
  `k0=(xx) g3ₓg0g0`, `k1=(xy)·g2ₓg1g0`, `k2=(xz) g2ₓg0g1`, `k3=(yx) g1ₓg2g0`, `k4=(yy) g0g3g0`, `k5=(yz) g0g2g1`, `k6=(zx) g1ₓg0g2`, `k7=(zy) g0g1g2`, `k8=(zz) g0g0g3` `[VERIFIED: intor1.c:1434-1462]`. (g0=overlap level, g1/g2/g3 = first/mixed/second moment levels; the index pattern is the canonical row-major 3×3 r⊗r.)
- **`int1e_rrr` (rank 27):** `gout[n*27+0..26] = s[0..26]` in emission order — a 3×3×3 nesting `[VERIFIED: intor1.c gout n*27 block]`. The planner/implementer copies `s[0..26]` verbatim (do NOT re-derive an analytic ordering — copy the s-index list from the source).
- **`int1e_rrrr` (rank 81):** `gout[n*81+0..80] = s[0..80]` verbatim `[VERIFIED: intor1.c:1762 gout n*81 block]`.
- **`int1e_irp` (rank 9):** `gout[n*9+0..8] = s[0..8]` where `g1=G1E_D_J(g0)` (∇ on ket) and `g2=G1E_RCJ(g0)` (r on ket): `s0=g3ₓg0g0`(g3=D_J(g2)), … i.e. the 3×3 (∇-axis ⊗ r-axis) tensor `[VERIFIED: intor1.c:2788-2816]`.
- **`int1e_drinv` (rank 3):** order (x,y,z) of the derivative direction `[VERIFIED: intor1.c:3693-3699]`.
- **Scalars (`r2,r4,z,zz,p4,rinv`):** rank 1, single output (the specific contraction in the table above).

**Implementation rule (D-07/D-08):** for each tensor family, hardcode the component emission as the exact `s[k]` → `gout[n*RANK+k]` list copied from `intor1.c`. A wrong order is invisible on a square bra×ket block — gate with NON-SQUARE (e.g. p×d).

### Pattern: Cluster A parameterized moment kernel (origin-source branch — D-02)

```text
// Conceptual (real kernel is #[cube] generic over F, f64-internal where needed).
// origin_source ∈ { Common, KetCenter }   (comptime or runtime branch)
let origin = match origin_source {
    Common    => common_orig,          // env[PTR_COMMON_ORIG], base family
    KetCenter => rj,                    // ket basis center, _origj variant
};
// build overlap G-tensor (existing one_electron engine, ket headroom = moment_order)
// apply position-power levels relative to `origin` on the ket index (G1E_RCJ analogue)
// emit components in verbatim libcint gout order for this family
```
The branch realizes Phase 22 D-04's "kernel-side coordinate choice." Per CONTEXT discretion this may be one comptime-parameterized kernel or a small family of order-specialized launchers.

### Pattern: Cluster B rinv/drinv (reuse iprinv precedent)

The existing `iprinv` path is the template `[VERIFIED: one_electron.rs:3518-3590, 3954-4043]`:
- iprinv already does **single rinv origin, charge factor +1.0, no `-Z_C`, no atom-loop**, resolving the origin from `plan.operator_env_params.rinv_orig` (validator 21-01) `[VERIFIED: one_electron.rs:3958-3964]`.
- **Plain `int1e_rinv`** = the scalar (non-derivative) sibling: same single-center Rys evaluation, rank 1, factor `2π·fac·tau/aij` (no charge, no `-Z`) `[VERIFIED: g1e.c:226-228]`.
- **`int1e_drinv`** = gradient wrt the rinv center; libcint builds `g1 = D_J(g0) + D_I(g0)` then accumulates over roots → rank 3 `[VERIFIED: intor1.c:3678-3691]`. This is the **derivative wrt the operator center C** (by translational invariance `−(∂_i+∂_j)`), distinct from `iprinv` which is `∂/∂A_bra`. Note the difference when reusing.
- Rys roots: `rinv` nroots = ⌈(li+lj+1)/2⌉; `drinv` adds one derivative order. Both gated at `MAX_DEVICE_NROOTS=5` (fail-closed above) `[VERIFIED: one_electron.rs:42, :4006]`.

### Pattern: Cluster C p4 / Cluster D irp (overlap-derivative)

- `p4`: apply `D_I` twice and `D_J` twice (libcint: `G1E_D_J` builds g1,g2,g3 to j_l+1; then `G1E_D_I` cascades g4..g15), then the rank-1 Laplacian² contraction `s0+2s4+2s8+s40+2s44+s80` `[VERIFIED: intor1.c:2433-2536]`. Headroom: BOTH bra and ket +2 (ng={2,2,...}).
- `irp`: `g1=D_J(g0)`, `g2=RCJ(g0)`, `g3=D_J(g2)`; emit the 3×3 tensor `s0..s8` `[VERIFIED: intor1.c:2781-2816]`.

### Anti-Patterns to Avoid
- **Re-deriving the gout component order analytically.** Copy `s[k]`→`gout` verbatim from `intor1.c`. (D-07.)
- **Setting `component_rank` too low.** Silently truncates trailing components (D-08). Use the exact rank from the table.
- **Raising headroom on the bra for Cluster A.** Cluster A raises ket headroom (`ng[1]`) only. (p4 is the exception — it genuinely raises both, ng={2,2,...}.) (D-07.)
- **Routing `int1e_rinv` through the atom-summed nuclear path** (`int1e_type=2`). rinv is single-center `int1e_type=1`. No atom loop, no charge weighting (D-04).
- **Reading `PTR_COMMON_ORIG` for rinv/drinv.** Source proves `PTR_RINV_ORIG` (OQ-1).
- **Collapsing `_origj` into a flag on the base operator.** D-02 requires a distinct operator/RawApiId per variant.
- **Gating on a SQUARE bra×ket block.** Transpose-symmetric; hides the layout bug (D-07).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Component order for r/rr/rrr/rrrr/irp | An analytic Cartesian-nesting derivation | The verbatim `s[k]` list copied from `intor1.c` | libcint's emission order is the parity contract; any re-derivation risks a transposed/permuted layout that only fails on non-square blocks |
| Staging buffer sizing for rank-81 | Manual layout/scatter code | `parse_component_multiplier` (auto-sizes from `component_rank`) | Already sizes correctly to rank 81; scatter guards never trip when sized right (D-03) |
| Env-slot read + finiteness validation | New env parsing | `eval_raw` reads `common_orig` (env[1..3]) and `rinv_orig` (env[4..6]); `validate_{common_orig,rinv_orig}_env_params` | Already plumbed (Phase 21/22) |
| Rys roots for rinv/drinv | New Rys implementation | Existing `rys_root1..5` + nuclear/iprinv kernel | Comptime-nroots device path exists, fail-closed >5 |
| Manifest audit sync | A fixtures family list | Edit only the lock; `cargo build -p cintx-ops` regenerates `api_manifest.{rs,csv}`; audit derives both sides from the lock | No separate list to maintain (D-05) |
| Non-zero gauge-origin fixture | A new fixture | `build_h2o_sto3g_common_orig()` (origin `[0.5,-0.3,0.8]`) | Already exists; the mandatory parity gate |

**Key insight:** Phase 24 is almost entirely composition of existing, proven machinery. The genuinely NEW work is (1) the per-family gout emission tables and (2) the Cluster-A origin-source branch — both fully specified above from source.

## Runtime State Inventory

> Phase 24 is additive (new manifest families + new kernels + new tests). It is NOT a rename/refactor/migration. No existing stored data, live-service config, OS-registered state, secrets, or build artifacts carry a string that changes.
>
> - **Stored data:** None — no datastore keys change.
> - **Live service config:** None.
> - **OS-registered state:** None.
> - **Secrets/env vars:** None — `PTR_COMMON_ORIG`/`PTR_RINV_ORIG` are existing libcint env-array slot indices, unchanged.
> - **Build artifacts:** `api_manifest.{rs,csv}` are regenerated from `compiled_manifest.lock.json` by `cargo build -p cintx-ops` (a generated artifact that auto-updates — not stale state, but the planner must run the regen step after every lock edit). The vendor libcint static lib rebuilds automatically when the allowlist regex changes (bindgen re-runs). Verified by inspecting `build.rs:51,358`.

## Common Pitfalls

### Pitfall 1: rinv origin slot confusion (PTR_RINV_ORIG vs PTR_COMMON_ORIG)
**What goes wrong:** Reading the gauge origin (env[1..3]) for rinv/drinv. CONTEXT D-04 asserts common origin, but libcint source reads `PTR_RINV_ORIG` (env[4..6]).
**Why it happens:** Moment families (Cluster A) DO read PTR_COMMON_ORIG, so the assumption bleeds over to Cluster B.
**How to avoid:** Use the existing Phase-21 `rinv_orig` plumbing for rinv/drinv. The non-zero-origin fixture only sets PTR_COMMON_ORIG — the rinv/drinv tests must ALSO set `rinv_orig` (via `.with_rinv_origin([..])`) to a non-zero value, else parity is trivially-passing at origin [0,0,0]. **Resolve OQ-1 first.**
**Warning signs:** rinv parity passes at zero origin but you never set a non-zero rinv center.

### Pitfall 2: Transposed tensor layout on a square block
**What goes wrong:** A permuted/transposed component emission passes a p×p block but fails a real workload.
**How to avoid:** Every `vendor_*` test uses a NON-SQUARE bra×ket block (e.g. O-p × H-d won't exist on STO-3G; use p×p only if you also test a non-square pair from the corpus — STO-3G H2O has s and p shells, so an s×p or p×s pair is non-square). Pick a non-square shell pair from H2O/STO-3G. (D-07.)
**Warning signs:** Only square shell pairs in the test.

### Pitfall 3: component_rank string truncation
**What goes wrong:** Writing `component_rank:"1"` for a rank-9 family truncates 8 components silently.
**How to avoid:** Copy the exact rank from the table. rr=9, rrr=27, rrrr=81, irp=9, drinv=3, r=3; scalars=1. (D-08.)
**Warning signs:** Output dimension smaller than vendor; parity "passes" because only the first component is compared.

### Pitfall 4: p4 headroom raises BOTH bra and ket
**What goes wrong:** Treating p4 like Cluster A (ket-only headroom). p4 is ng={2,2,...} — bra AND ket +2.
**How to avoid:** Build the bra Laplacian (D_I²) explicitly. Internal nmax = li+lj+4 (still ≤8 on STO-3G where li,lj≤1 → nmax≤6, within the `li+lj<=8` engine limit). `[VERIFIED: intor1.c:2539; one_electron.rs:41]`
**Warning signs:** p4 fails for any l>0 shell but passes for s×s.

## Code Examples

### libcint dipole gout (rank 3, the canonical order) — `[VERIFIED: intor1.c:1144-1158]`
```c
// drj = rj - env[PTR_COMMON_ORIG];   G1E_RCJ(g1,g0,...)
s[0] = g1[ix]*g0[iy]*g0[iz];  // x
s[1] = g0[ix]*g1[iy]*g0[iz];  // y
s[2] = g0[ix]*g0[iy]*g1[iz];  // z
gout[n*3+0]=s[0]; gout[n*3+1]=s[1]; gout[n*3+2]=s[2];
```

### libcint _origj branch (origin = ket center, NO drj) — `[VERIFIED: intor1.c:2073-2089]`
```c
// G1E_R_J(g1,g0,...)  ==  f = g + envs->g_stride_j   (no PTR_COMMON_ORIG subtraction)
s[0]=g1[ix]*g0[iy]*g0[iz]; s[1]=g0[ix]*g1[iy]*g0[iz]; s[2]=g0[ix]*g0[iy]*g1[iz];
```

### libcint rinv single-center Rys + charge — `[VERIFIED: g1e.c:226-228, intor1.c:3627-3638]`
```c
if (nuc_id < 0) {                         // int1e_rinv / drinv path
    fac1 = 2*M_PI * envs->fac[0] * tau / aij;   // charge = +1, NO -Z_C
    cr = env + PTR_RINV_ORIG;                   // <-- rinv center, NOT common origin
}
// gout: for i in nrys_roots: s += g0[ix+i]*g0[iy+i]*g0[iz+i];
```

### Existing cintx iprinv kernel (template for plain rinv) — `[VERIFIED: one_electron.rs:3958-3964]`
```rust
let iprinv_origin: Option<[f64;3]> = if is_iprinv {
    plan.operator_env_params.rinv_orig    // single rinv origin, factor +1.0, no -Z_C
        .ok_or(/* InvalidEnvParam: iprinv reached with no rinv origin */)?
} else { None };
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Host-side 1e moment evaluation | On-device `#[cube]` kernels generic over `F` | Phases 21-23 | Moment kernels are device kernels; host limited to planning/marshaling (CLAUDE.md) |
| `capi` enum + `cint*` wrappers per family | manifest + RawApiId + kernel + vendor-FFI + oracle ONLY | Phase 23 D-09 | No capi/cint* for Phase 24 families |
| Re-export `.c` per family in build.rs | All 1e autocode in one already-listed source (`src/autocode/intor1.c`) | — | Vendor-FFI step = allowlist regex + wrappers only, no source registration `[VERIFIED: build.rs:51]` |

**Deprecated/outdated:** N/A — internal port phase.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | On the H2O/STO-3G corpus, every Phase-24 family stays at `nroots≤5` and (for the overlap-derivative families) internal `nmax = li+lj+headroom ≤ 8`. Derived from STO-3G max l=1 (O: s,s,p; H: s) so li,lj≤1; p4 nmax≤6, rrrr ket-headroom nmax = li+lj+4 ≤ 6. **Not executed** — corpus shell l-values asserted from STO-3G structure, not a runtime probe. | Common Pitfalls / p4 | If a corpus shell has l>1 unexpectedly, p4/rrrr could exceed the engine `li+lj<=8` limit or rinv exceed nroots 5 → fail-closed (caught, not silent), triggering a Phase-25 escalation per D-04. |
| A2 | The H2O/STO-3G corpus contains a NON-SQUARE shell pair usable for the D-07 gate (e.g. s×p between O-s and O-p, or H-s × O-p). | Validation Architecture | If only square pairs are reachable, the transpose gate is weakened; planner must construct a synthetic non-square pair or add a basis with a d shell. |
| A3 | `_origj` families beyond r/z/zz (i.e. rr/rrr/rrrr/r2/r4 `_origj`) exist as distinct vendor symbols in the same pattern as `int1e_r_origj`. Confirmed for r/z/zz by source; the rest assumed by the autocode's uniform `_origj` emission pattern. | Standard Stack table | If some `_origj` variant is absent in libcint, that family's `_origj` registration/test must be dropped (parity target doesn't exist). Planner should grep `intor1.c` for the exact `_origj` symbol set before registering. |

## Open Questions (RESOLVED)

> All three resolved during plan-phase and consumed by the plans:
> - **OQ-1 — RESOLVED:** user confirmed "Follow source: `PTR_RINV_ORIG`" (D-04 [CORRECTION] in CONTEXT.md); adopted in plan 24-03; rinv/drinv tests set a non-zero `rinv_orig`.
> - **OQ-2 — RESOLVED:** triaged as a task in plan 24-01 (confirm against pre-phase-20 commit; convert to tracked standalone harness bug if reproduced — must not block the parity gate).
> - **OQ-3 — RESOLVED:** grep of `intor1.c` yields exactly 6 `_origj` symbols `{r,rr,r2,r4,z,zz}_origj`; `rrr_origj`/`rrrr_origj` do NOT exist and are excluded in plans 24-01/24-02.

1. **OQ-1 (HIGH PRIORITY — origin slot for rinv/drinv).**
   - What we know: libcint 6.1.3 source unambiguously reads `env + PTR_RINV_ORIG` (env[4..6]) for `int1e_rinv`/`int1e_drinv` via `CINTg1e_nuc(g,envs,-1)` `[VERIFIED: g1e.c:226-228]`, with charge=+1 and no atom-sum. CONTEXT D-04 states "common origin (`env[PTR_COMMON_ORIG]`)." These disagree.
   - What's unclear: whether D-04's "common origin" was shorthand for "the operator's single center" (loosely worded) or an actual decision to read env[1..3].
   - Recommendation: **Follow the source — use `PTR_RINV_ORIG`** (the Phase-21 `rinv_orig` plumbing already exists: `with_rinv_origin`, `operator_env_params.rinv_orig`, `validate_rinv_orig_env_params`). Flag this to the planner/user as a correction to D-04 before locking. Practically: the rinv/drinv `vendor_*` tests must set a NON-ZERO `rinv_orig` (the common-orig fixture alone does not exercise the rinv center).

2. **OQ-2 (folded todo `oracle-cart-offset-vendor-zero`).**
   - What we know: 4 `compare::tests` lib unit tests fail under the vendor gate at `CINTshells_cart_offset[4] cintx=8 vendor=0`; integration parity passes; hypothesis is a harness/env bug (vendor `ao_loc`→0 in lib-unit context). Phase 24 runs the vendor lib+integration gate so this re-surfaces.
   - What's unclear: whether it reproduces on a pre-phase-20 commit (→ pre-existing harness bug) or is new.
   - Recommendation: confirm against a pre-phase-20 commit; if reproduced, convert to a tracked standalone oracle-harness bug so the Phase-24 gate is not blocked by pre-existing noise (do NOT let it block byte-identity parity).

3. **OQ-3 (`_origj` symbol set completeness).**
   - What we know: `int1e_{r,z,zz}_origj` confirmed as distinct symbols `[VERIFIED: intor1.c:1957-2113]`.
   - What's unclear: exact set of `_origj` variants in 6.1.3 (does `rrrr_origj` exist?).
   - Recommendation: planner greps `grep -o 'int1e_[a-z0-9]*_origj_cart' libcint-master/src/autocode/intor1.c | sort -u` before registering, and registers only the variants that have a vendor symbol. (A3.)

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Vendored libcint sources | gout/ng derivation + vendor parity | ✓ | 6.1.3 | — |
| `intor1.c` in vendor build source list | vendor FFI compile | ✓ (`build.rs:51`) | — | — |
| `build_h2o_sto3g_common_orig` fixture | gauge-origin parity gate | ✓ (`fixtures.rs:158`) | origin `[0.5,-0.3,0.8]` | — |
| `with_rinv_origin` / `rinv_orig` plumbing | rinv/drinv parity (OQ-1) | ✓ (`builder.rs:102`, `planner.rs:53`) | — | — |
| CubeCL cpu backend (`--features cpu`) | parity test execution | ✓ (used by existing tests) | cubecl 0.10.0 | rocm feature |
| `CINTX_ORACLE_BUILD_VENDOR=1` | enables vendor FFI build | ✓ (env-gated) | — | without it parity SILENTLY SKIPS (D-10) |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** none material — rinv/drinv tests need a non-zero `rinv_orig` set explicitly (no fixture provides it today; trivially supplied via `with_rinv_origin`).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` integration tests under `crates/cintx-oracle/tests/`, gated `#![cfg(any(feature = "cpu", feature = "rocm"))]` + `has_vendor_libcint` cfg (`CINTX_ORACLE_BUILD_VENDOR=1`) |
| Config file | none — cargo test discovery; per-test cfg gates `[VERIFIED: one_electron_grad_both_parity.rs:1-9]` |
| Quick run command | `cargo test -p cintx-oracle --features cpu --test <moment_parity_test> -- <name>` |
| Full suite command | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu` (vendor parity; without the env var parity silently skips) |

> **Vendor lib+unit caveat (MEMORY):** routine `--features cpu` CI runs `--test` integration only, never `--lib` under vendor. The folded `oracle-cart-offset-vendor-zero` lib-unit failure (OQ-2) surfaces only when the vendor `--lib` tests run. Confirm/triage per OQ-2 so it does not block the phase gate.

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MOM-01 | `int1e_r`/`int1e_r_origj` cart+sph byte-identity @1e-12 on non-zero gauge fixture, non-square block | integration (vendor) | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test moment_r_parity` | ❌ Wave 0 |
| MOM-02 | `int1e_rr/r2/z/zz` (+`_origj`) cart+sph @1e-12 | integration (vendor) | `... --test moment_low_parity` | ❌ Wave 0 |
| MOM-03 | `int1e_rrr/rrrr/r4` cart+sph @1e-12, ket headroom | integration (vendor) | `... --test moment_high_parity` | ❌ Wave 0 |
| MOM-04 | `int1e_p4/drinv/rinv/irp` cart+sph @1e-12 (rinv/drinv with non-zero `rinv_orig`) | integration (vendor) | `... --test moment_nontensor_parity` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo build -p cintx-ops` (manifest regen) + `cargo test -p cintx-oracle --features cpu --test <family>_parity` for the family just added.
- **Per wave merge:** `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu` (all moment parity tests).
- **Phase gate:** full vendor suite green + `manifest-audit` green (`cargo build -p cintx-ops` regenerates and the audit derives both sides from the lock) before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `crates/cintx-oracle/tests/moment_r_parity.rs` — covers MOM-01 (r, r_origj)
- [ ] `crates/cintx-oracle/tests/moment_low_parity.rs` — covers MOM-02 (rr, r2, z, zz, +_origj)
- [ ] `crates/cintx-oracle/tests/moment_high_parity.rs` — covers MOM-03 (rrr, rrrr, r4)
- [ ] `crates/cintx-oracle/tests/moment_nontensor_parity.rs` — covers MOM-04 (p4, rinv, drinv, irp)
- [ ] `vendor_*` safe wrappers in `crates/cintx-oracle/src/vendor_ffi.rs` for each new cart/sph symbol (mirror `vendor_int1e_iprinv_*`)
- [ ] `allowlist_function` regex extension in `crates/cintx-oracle/build.rs:358` adding all `int1e_{r,rr,rrr,rrrr,r2,r4,z,zz,p4,irp,rinv,drinv}_{sph,cart}` + `_origj` symbols
- [ ] Test helper: a non-zero `rinv_orig` setter for the rinv/drinv tests (the common-orig fixture does not set env[4..6])
- [ ] Confirm a NON-SQUARE shell pair from H2O/STO-3G is used in each test (A2)

*(Existing test infra — `vendor_parity<FS,FC>` helpers in `one_electron_grad_both_parity.rs:307` etc. — is the pattern to clone; no framework install needed.)*

## Security Domain

Not applicable in the conventional sense — this is a numerical integral library with no auth/session/network/PII surface. The relevant "input validation" analogue (ASVS V5) is already enforced: `validate_common_orig_env_params` / `validate_rinv_orig_env_params` reject non-finite (NaN/inf) env-slot origins `[VERIFIED: validator.rs:210-223]`, and `eval_raw` bounds-guards env-slot reads (`env.len() >= PTR_*_ORIG + 3`) `[VERIFIED: raw.rs:663,677]`. No new validation gap is introduced. Memory-safety (CLAUDE.md OOM contract) for rank-81 is explicitly deferred to Phase 25 (D-03) — Phase 24's existing staging is sized correctly so no partial writes occur on the gate corpus.

## Sources

### Primary (HIGH confidence)
- `libcint-master/src/autocode/intor1.c` — gout emitters, ng[] tuples, `_origj` symbols for all moment families (lines cited inline). Version 6.1.3.
- `libcint-master/src/g1e.{h,c}` — G1E_RCJ/G1E_R_J/G1E_D_* macros (g1e.h:52-63); `CINTg1e_nuc` origin+charge for nuc_id<0 (g1e.c:208-238).
- `libcint-master/src/cint1e.c` — `CINT1e_drv` / `make_g1e_gout` int1e_type 0/1/2 dispatch (cint1e.c:188-303); int1e_nuc (flag 2) vs rinv (flag 1).
- `libcint-master/CMakeLists.txt` — cint_VERSION 6.1.3.0.
- `crates/cintx-compat/src/raw.rs` — PTR_COMMON_ORIG=1 (:50), PTR_RINV_ORIG=4 (:58), eval_raw env reads (:663-686).
- `crates/cintx-runtime/src/planner.rs` — OperatorEnvParams (:50-54), parse_component_multiplier (:403-450).
- `crates/cintx-runtime/src/validator.rs` — validate_common_orig_env_params (:210-223).
- `crates/cintx-cubecl/src/kernels/one_electron.rs` — overlap+nuclear Rys engine (:415-503), iprinv single-origin path (:3518-3590,3954-4043), MAX_DEVICE_NROOTS=5 (:42), dispatch ladder (:3765-3790).
- `crates/cintx-oracle/build.rs` — vendor source list incl. `src/autocode/intor1.c` (:51); allowlist_function regex (:358).
- `crates/cintx-oracle/src/fixtures.rs` — build_h2o_sto3g_common_orig + ORIGIN [0.5,-0.3,0.8] (:148-176).
- `crates/cintx-ops/generated/compiled_manifest.lock.json` — 199 entries, schema 1; all targets confirmed absent; iprinv full entry shape.

### Secondary (MEDIUM confidence)
- CONTEXT.md D-01..D-10 and carry-forward Phase 22/23 CONTEXT (registration recipe, transpose discipline, component-rank rule).

### Tertiary (LOW confidence)
- A1/A2/A3 assumptions (corpus l-values, non-square pair availability, `_origj` symbol-set completeness) — flagged in Assumptions Log; resolve by grep before registering.

## Metadata

**Confidence breakdown:**
- Standard stack / family inventory: HIGH — all targets verified absent in the lock; all source files located and version-confirmed.
- Architecture (gout orders, ranks, ng[], origin sources): HIGH — read verbatim from libcint 6.1.3 source at cited lines.
- rinv/drinv origin slot: HIGH on the source fact (PTR_RINV_ORIG), but flagged OQ-1 because it CONTRADICTS CONTEXT D-04 and needs a planner/user decision before locking.
- Pitfalls / validation: HIGH — patterns mirror proven Phase 21-23 vendor tests.

**Research date:** 2026-05-30
**Valid until:** stable (vendored libcint 6.1.3 is pinned; workspace code anchors may drift line numbers but semantics are stable). Re-verify line numbers if the kernel files are refactored.

## RESEARCH COMPLETE
