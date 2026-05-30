# Stack Research — v1.4 Full libcint 6.1.3 Family Parity

**Domain:** Shared `#[cube]` math + kernel-module inventory for the ~140 remaining libcint 6.1.3 integral families, grouped into 6 family groups. Scope is *what is NEW per group* — base/F12/ECP/Phase-21 capabilities are reused, not re-researched.
**Researched:** 2026-05-27
**Confidence:** HIGH (mapped against vendored libcint 6.1.3 source at `./libcint-master` and the live cintx tree; no training-data-only claims)

## Headline Finding

**No new Rust crate dependencies are required for any of the 6 groups.** The existing `cubecl 0.10.0` + host-side Boys/Rys/Obara–Saika/STG stack plus `num-complex 0.4` (already a workspace dep, used by the spinor path) covers everything. libcint builds *all* ~140 families from exactly **two reusable operator primitives applied to the same base g-tensor** —

- **∇ ("ip"/"p"/nabla) operator** — `CINTnabla1{i,j,k,l}` — already ported in cintx as `f12.rs::nabla1i_2e/nabla1j_2e/nabla1k_2e` and `unstable.rs::nabla1{i,j,l}_breit`, plus the Phase-21 1e nabla path used by `one_electron.rs`.
- **r̂ (position / gauge-displacement) operator** — `CINTx1{i,j,k,l}` — already ported as `unstable.rs::x1j_breit/x1l_breit`.

Everything else is **per-family `gout` contraction patterns** (Cartesian-component bookkeeping + sign/cross-product/Pauli structure), which are cheap host-side `Vec<f64>` functions, not new device math. The genuinely new device-adjacent need is small and confined to one group (relativistic σ·p Pauli assembly), and even that reuses the existing `c2spinor.rs` machinery. This is a *high-reuse, low-new-math* milestone.

## What Each Group Needs — Source → cintx Module Map

### Group 1 — Remaining 1st-derivative families
Targets: `int2e_ip2`, `int3c1e_ip1`, `int3c1e_iprinv`, `int2c2e_ip1/ip2`, `int3c2e_ip2` (note: `int1e_ipnucip/ipkinip/ipovlpip` are actually 2nd-order — see Group 2).

| libcint source | Operator structure | cintx module: extends vs new |
|---|---|---|
| `src/autocode/grad1.c` (1e ip), `src/autocode/grad2.c` (`int2e_ip1/ip2`), `src/autocode/int3c1e.c` (`int3c1e_ip1/iprinv`), `src/autocode/int3c2e.c` (`int2c2e_ip1/ip2`, `int3c2e_ip2`) | Single ∇ on one center; `ip2`/`ip1` differ only in *which* center the existing `nabla1{i,j,k,l}_2e` is applied to | **EXTEND** `f12.rs::gout_ip1` (already `pub(crate)`, already shared by `two_electron.rs::int2e_ip1` and `center_3c2e.rs::int3c2e_ip1`). Add `gout_ip2`/`gout_ip1_3c1e` siblings calling the *existing* nabla on the ket/k center. New device math: **none.** |

**New = NOTHING in `math/`. New = a handful of host `gout_*` contraction functions** mirroring grad1/grad2/int3c*.c autocode, plus manifest registration + `component_rank`. Reuses `fill_g_tensor_2e`, the 1e g-tensor fill, the Rys roots, and the existing nabla operators verbatim.

### Group 2 — Hessian & higher-order derivatives
Targets: `int1e_ipip*` (`ipipovlp/ipipkin/ipipnuc/ipiprinv`, `ipovlpip/ipkinip/ipnucip/iprinvip`), `int2c2e_ipip1`, `int3c2e_ipip1/ipip2`, `int2e_ipip1/ipvip1/ip1ip2`, 3rd/4th-order `ipipipiprinv` etc.

| libcint source | Operator structure | cintx module: extends vs new |
|---|---|---|
| `src/autocode/hess.c` (2nd-order), `src/autocode/deriv3.c` (3rd), `src/autocode/deriv4.c` (4th), `src/autocode/lresc.c` (mixed r/∇), `src/autocode/int3c2e.c` (`int3c2e_ipip1/ipip2`, `int2c2e_ipip1`) | Repeated ∇ via *G-tensor elevation*: build g at `li+N`, then apply `nabla` N times (`nabla1i_2e(g1,g,li+1,…)` → `nabla1i_2e(g3,g1,li+0,…)`) | **EXTEND** — the 2nd-order machinery already exists: `f12.rs::gout_ipip1` (with column-major 3×3 reorder), `gout_ipvip1`, `gout_ip1ip2`. The *exact* G-tensor-elevation pattern (`li+1` build, double-nabla) is coded and oracle-validated. 3rd/4th-order = same pattern with deeper elevation (`li+2/li+3`) and longer `s[]` contractions. |

**New = NOTHING in `math/`. New = host `gout_ipipip*`/`gout_ipipipip*` contractions + ceil-angular-momentum bookkeeping** (planner sizes the g-tensor at `li+order`). Decisive evidence: cintx *already ships* `int2e_ipip1`/`int2e_ipvip1` (PROJECT.md), and the f12.rs gout functions for them. Highest-reuse group.

### Group 3 — Position / multipole moment integrals
Targets: `int1e_r/rr/rrr/rrrr`, `int1e_r2/r4` (+`_origj`/`_origi`), `int1e_z/zz`, `int1e_sp`, `int1e_p4`, plain `int1e_rinv`, `int1e_drinv`, `int1e_irp/irrp/irpr`.

| libcint source | Operator structure | cintx module: extends vs new |
|---|---|---|
| `src/autocode/intor1.c` (`int1e_r/rr/rrr/rrrr/r2/r4/z/zz/irp/...`), `src/cint1e_a.c` (`_origi`/`_origj` r² drivers), `src/autocode/intor1.c` (`int1e_p4` = ∇⁴), `src/g1e.c::CINTx1{i,j}_1e` | Stacked **r̂ position operator** (`G1E_RCJ`/`G1E_R0I` = `CINTx1j/x1i` with displacement to `PTR_COMMON_ORIG`); `rr`/`rrr` = repeated x̂ via elevation; `p4` reuses ∇⁴; `irp`/`irrp` = mixed r̂+∇ | **EXTEND.** The position operator is *already ported* as `unstable.rs::x1j_breit/x1l_breit` (literally `CINTx1j_2e`/`CINTx1l_2e`); lift the 1e form (`CINTx1i_1e`/`CINTx1j_1e`) into a shared `one_electron.rs` position-op helper. `p4` reuses Group-2 nabla-elevation. New device math: **none.** |

**New = a 1e `x1i_1e`/`x1j_1e` position-operator helper** (trivial port of the present 2e `x1j_breit`) **+ per-rank host `gout_r/rr/rrr/rrrr` contractions** + a **`PTR_COMMON_ORIG` (gauge/common-origin) env slot** added to `OperatorEnvParams` (mirrors the existing `rinv_orig: Option<[f64;3]>` slot from Phase 21 — same `validate_*_env_params` pattern). No new crate, no new Rys/Boys work. Moments use the *overlap* g-tensor for `r/rr`; `rinv`/`drinv` reuse the existing nuclear/`iprinv` Boys path.

### Group 4 — Relativistic spin-operator integrals
Targets: `int1e_spsp/spnucsp/sprinvsp/srsr/sigma/sprsp/srsp/...`, `int2e_spsp1/srsr1/ssp*/sps*/vsp*/spv*`.

| libcint source | Operator structure | cintx module: extends vs new |
|---|---|---|
| `src/autocode/intor3.c` (`int1e_sigma/sp/spsp/spnucsp/sprinvsp/srsr/...`), `src/autocode/grad1.c` (`int1e_ipspnucsp`), `src/autocode/dkb.c` (`int1e_spspsp`, `int2e_spv1/vsp1/...`), `src/autocode/gaunt1.c` (`int2e_sps1sps2/ssp1ssp2/...`) | **σ·p machinery**: `sigma`/`sp` emit a 12-component (4 spinor-block × 3) Pauli-structured `gout` (`CINTgout1e_int1e_sigma` writes `±s` into a 12-slot Pauli pattern); `spsp` = (σ·p)…(σ·p) sandwich; relies on ∇ + the spin-included spinor transform | **EXTEND + smallest genuinely-new work.** ∇ operator: reuse. **Pauli/σ coupling: already ported** as `transform/c2spinor.rs::cart_to_spinor_si` (spin-included, Pauli `vx/vy/vz`) and `cart_to_spinor_sf_4d`. The Breit spinor 2e path (`unstable.rs::launch_breit`, `gout_breit_r1p2/r2p2`) already does the σ·p ⊗ σ·p sandwich + 4-component assembly. **New = a σ-Pauli `gout` emitter** (the 12-component cross/Pauli structure from intor3.c) shared across `sigma/sp/spsp`, plus wiring `cart_to_spinor_si` into a 1e spinor driver. |

**Crate impact: none** (`num-complex 0.4` present; spinor complex layout flows through `OutputLayoutMetadata.complex_interleaved`). **Hard prerequisite (sequencing, not a crate): the R5/D-03 spinor-derivative blocker** — Phase 21 left spinor gradients `registered-but-UnsupportedApi`. The `int1e_ipspnucsp`/`int2e_ip1spsp2` subfamilies combine ∇ *and* σ in spinor form, so this group must lift D-03 first.

### Group 5 — GIAO / magnetic-property NMR integrals
Targets: `int1e_giao_*`, `int1e_cg_*`, `int1e_a01gp`, `int1e_ia01p`, `int1e_ig{kin,nuc,ovlp}`, `int1e_g1/gg1`, `int1e_govlp/gnuc`, `int2e_g1/gg1/ig1/...`.

| libcint source | Operator structure | cintx module: extends vs new |
|---|---|---|
| `src/autocode/intor1.c` (`int1e_igovlp/igkin/ignuc/a01gp/ia01p/...`), `src/autocode/intor2.c` (`int2e_g1g2/ig1/...`), `src/autocode/intor3.c` (`int1e_govlp/gnuc/giao_sa10*/cg_*`), `src/autocode/intor4.c` (`int2e_g1/giao_sa10sp1/...`) | **Gauge-origin r̂ + angular-momentum (r×∇)**: `igovlp` = `G1E_R0I` (position op) contracted via a **cross-product** with the gauge displacement `c = ri-rj`, emitting an **imaginary** GIAO factor (`CINTgout1e_int1e_igovlp`: `gout = -c[1]*s[2]+c[2]*s[1]`, …) | **EXTEND.** Position op (`x1i_1e` from Group 3) + ∇ (existing). **New = host `gout_giao_*`/`gout_cg_*` cross-product contractions** (L = r×p assembled from existing r̂ and ∇ outputs) and **complex/imaginary output** (the `ig` prefix ⇒ imaginary unit; representable via `complex_interleaved`). |

**Crate impact: none.** No new g-tensor math, no new roots. Only new device-adjacent concept is the **cross-product + i·(…) gout assembly**; feeds off the existing overlap/nuclear g-tensors and the Group-3 position operator. Needs the same `PTR_COMMON_ORIG` gauge-origin env slot as Group 3.

### Group 6 — Gauge / Breit–Gaunt 2e (relativistic 2e)
Targets: `int2e_gauge_r1_{ssp1,sps1}{ssp2,sps2}`, `int2e_gauge_r2_{...}` (8 symbols, breit1.c); Gaunt blocks `int2e_{ssp1ssp2,sps1sps2,ssp1sps2,sps1ssp2}` (gaunt1.c).

| libcint source | Operator structure | cintx module: extends vs new |
|---|---|---|
| `src/autocode/breit1.c` (gauge_r1/r2 per-block), `src/autocode/gaunt1.c` (Gaunt per-block), driven by `src/breit.c::int2e_breit_r1p2/r2p2_optimizer/_drv` | Breit r1/r2 gauge: σ·p ⊗ σ·p sandwich with a position-operator (`G2E_R0J`) inside, on the 2e Rys g-tensor; Gaunt = σ·p ⊗ σ·p without the r-operator | **EXTEND — already largely built.** `unstable.rs::launch_breit` + `gout_breit_r1p2/r2p2` + `BreitShape` + `x1j_breit/x1l_breit` + `nabla1{i,j,l}_breit` + `cart_to_spinor_sf_4d` implement the *aggregate* Breit driver (`int2e_breit_r1p2/r2p2_spinor`). The milestone's **per-block** `gauge_r1/r2_{ssp,sps}{ssp,sps}` and Gaunt blocks are the *same* operator sandwich decomposed; reuse the BreitShape g-tensor + operators, add **per-block host `gout_gauge_r{1,2}_*`/`gout_gaunt_*` contractions.** |

**Crate impact: none.** Reuses the Rys 2e roots, the Breit g-tensor builder, the spinor 4-component assembly, and `num-complex`. New = per-block gout contractions only. Spinor-only (cart/sph return `UnsupportedApi`, per existing D-07).

## Recommended Stack (no changes from v1.3)

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `cubecl` | `0.10.0` (pinned) | Device compute backend for per-primitive kernel launches | Already the validated primary backend; all 6 groups launch through the existing `#[cube(launch)]` kernel pattern. No 0.10 feature beyond what Phase 8–21 already use. |
| Rust toolchain | `1.94.0` (pinned, `rust-toolchain.toml`) | Reproducible compiler for oracle byte-identity | Unchanged; required for oracle reproducibility. |
| `num-complex` | `0.4.6` (in `crates/cintx-rs`, resolved in lockfile) | Complex/imaginary output for GIAO (`ig*`), spinor relativistic, Breit/Gaunt | **Already a dependency.** GIAO imaginary factor and spinor complex layout flow through the existing `OutputLayoutMetadata.complex_interleaved` plumbing. No version bump. |
| `thiserror` / `anyhow` | `2.0.18` / `1.0.102` | Lib vs tooling errors | Unchanged; new families add `UnsupportedApi` taxonomy entries only. |

### Supporting Libraries (all already present — listed to confirm sufficiency)

| Library / module | Version | Purpose | Sufficient for v1.4? |
|---------|---------|---------|----------------------|
| `math/boys.rs` | n/a | Boys F_m(T) for nuclear/rinv/2e | YES — `rinv/drinv/spnucsp/sprinvsp/Breit` reuse it unchanged |
| `math/rys.rs` | n/a | Rys roots/weights (2e, higher nroots) | YES — Hessian/Breit/Gaunt need higher nroots from elevated angular momenta; rys.rs already covers the range used by F12/Breit |
| `math/obara_saika.rs` | n/a | VRR/HRR `vrr_step/hrr_step/vrr_2e_step` | YES — all g-tensor builds reuse these verbatim |
| `math/stg.rs` | n/a | STG roots | Not needed by these 6 groups (F12-only) |
| `transform/c2spinor.rs` | n/a | Pauli `si` coupling + 4D spinor assembly | YES — Group 4/6 reuse `cart_to_spinor_si` and `cart_to_spinor_sf_4d` |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| vendored `libcint-master` (6.1.3) | Byte-identity oracle | Map each new family to its autocode `.c` (tables above); vendor-gated tests need `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1` (both, or parity silently skips). |
| `compiled_manifest.lock.json` | Family registry source-of-truth | 172 entries today, 145 oracle-covered. Each new family: register `id{family,operator,representation,symbol}` + `component_rank`, then flip `oracle_covered`. |

## What is NEW vs REUSED — one-line summary per group

| Group | New `math/` device modules | New (host) gout/operator | Reuses |
|-------|----------------------------|--------------------------|--------|
| 1. Remaining 1st-deriv | **none** | `gout_ip2`, `gout_ip1` siblings per center | `nabla1{i,j,k,l}_2e`, `fill_g_tensor*`, Rys |
| 2. Hessian/higher-order | **none** | `gout_ipipip*`/`gout_ipipipip*` + ceil bookkeeping | `gout_ipip1/ipvip1/ip1ip2` (already shipped), G-tensor elevation |
| 3. Moments | **none** | 1e `x1i_1e/x1j_1e` helper + `gout_r/rr/rrr/rrrr`; `PTR_COMMON_ORIG` env slot | `x1j_breit` (2e position op), overlap g-tensor, ∇⁴ from Group 2 |
| 4. Relativistic σ·p | **none** (Pauli already in c2spinor) | σ-Pauli 12-comp `gout` emitter; 1e spinor driver | `cart_to_spinor_si`, `cart_to_spinor_sf_4d`, ∇ ops, Breit σ·p sandwich |
| 5. GIAO/NMR | **none** | `gout_giao_*/cg_*` cross-product (L=r×p) + imaginary output; `PTR_COMMON_ORIG` | position op (Group 3), ∇, `complex_interleaved` |
| 6. Breit–Gaunt 2e | **none** | per-block `gout_gauge_r{1,2}_*/gout_gaunt_*` | `launch_breit`, `BreitShape`, `x1{j,l}_breit`, `nabla*_breit`, 4D spinor |

## What NOT to Add

| Avoid | Why | Use instead |
|-------|-----|-------------|
| Any new crate (FFT, special-function, BLAS, complex-math) | All operator math is the same ∇ / r̂ recurrence on the existing g-tensor; complex is already `num-complex 0.4` | Reuse `obara_saika.rs` + `num-complex` |
| A new `#[cube]` "spin-operator" device module | σ·p decomposes into ∇ (existing) + a Pauli `gout` reorder + spinor transform (existing in `c2spinor.rs`) | Extend `c2spinor.rs` usage + add a host gout emitter |
| A new gauge/GIAO device math module | GIAO = existing position op + cross-product gout + imaginary factor; no new recurrence | Group-3 position op + `gout_giao_*` |
| Re-deriving Breit/Gaunt from scratch | The aggregate Breit driver (g-tensor + σ·p sandwich + 4D spinor) already exists in `unstable.rs` | Decompose into per-block gout over the existing `BreitShape` |
| Treating moments as needing new roots | `int1e_r/rr` use the *overlap* g-tensor; `rinv/drinv` use the *existing* nuclear Boys path | 1e g-tensor fill + position op |
| Bumping `cubecl` for "new" math | No new 0.10 capability is needed; the kernels are the same launch shape as Phase 21 | Stay on pinned `0.10.0` |

## CubeCL 0.10 Capability Constraints (relevant to the new operator math)

Verified against the live tree (`executor.rs`, `capability.rs`) — the constraints are the *same* ones Phase 8–21 already navigate; nothing in these 6 groups introduces a new class of constraint:

1. **`SHADER_F64` gate (wgpu/metal):** The executor gates f64 dispatch on `SHADER_F64` (`executor.rs::check_f64_capability`, `check_shader_f64_in_features`). CPU is always f64-capable; CUDA is accept-with-failure. All new families inherit this gate unchanged — **no new requirement**, because the new math is the same f64 multiply-add recurrence. Adapters lacking `SHADER_F64` already fall back to the f32 path (PREC-04/D-10).
2. **Operator math stays host-side `Vec<f64>`:** The decisive architectural fact — `nabla1i_2e`, `gout_ip1`, `gout_ipip1`, `x1j_breit`, `fill_g_tensor_2e` are **plain host functions**, not `#[cube]`. The `#[cube]` device kernels are the per-primitive VRR/HRR/Boys/Rys inner loops; the derivative/position/Pauli operators are applied on the host before/after the device launch. **This means the new groups add zero new `#[cube]` code subject to `cond_br`/MLIR control-flow limits** — they add host `gout` functions. The new operator math sidesteps the CubeCL control-flow constraints entirely.
3. **`cond_br`/MLIR control-flow limits:** Apply only to `#[cube]` device kernels (statement-form if/else, `u32` loop counters, no recursion, `as usize` indexing — see `obara_saika.rs` doc header). Since the new operator math is host-side, these limits are **not** a v1.4 concern for the new families. They remain relevant only if/when a hot operator is later lowered into the device kernel (not required for parity).
4. **Complex output:** Handled by interleaved-f64 staging (`OutputLayoutMetadata.complex_interleaved`), already exercised by the spinor base families and Breit. GIAO imaginary output and relativistic spinor output reuse this — **no CubeCL complex-type dependency.**

## Sequencing / Risk Flags for the Roadmap

- **Group 4 (relativistic) and Group 6 (Breit–Gaunt) gate on lifting the R5/D-03 spinor-derivative restriction** (Phase 21 left spinor gradients `UnsupportedApi`). The σ·p + ∇ combinations (`int1e_ipspnucsp`, `int2e_ip1spsp2`) cannot pass until spinor derivatives are unblocked. Single highest-risk prerequisite; sequence it before Groups 4/6.
- **Groups 1, 2, 3, 5 are low-risk**: pure extensions of already-validated host machinery (nabla, position op, gout patterns), highest reuse, no spinor-derivative dependency. Recommended to land first.
- **Higher angular-momentum g-tensor sizing**: 3rd/4th-order derivatives (`deriv3.c`/`deriv4.c`) elevate angular momenta by 2–3; confirm the planner's workspace query sizes the g-tensor at `l+order` (Group-2 `gout_ipip1` already proves `li+1` works; extend the ceil logic).

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `cubecl@0.10.0` | Rust `1.94.0` | Unchanged from v1.3; validated through Phase 21 |
| `num-complex@0.4.6` | workspace | Already resolved in `Cargo.lock`; used by `crates/cintx-rs` complex outputs |
| vendored libcint | `6.1.3` | Oracle ground truth; autocode `.c` files unchanged |

## Sources

- `./libcint-master/src/autocode/{grad1,grad2,deriv3,deriv4,hess,intor1,intor2,intor3,intor4,breit1,gaunt1,dkb,lresc,int3c1e,int3c2e}.c` — HIGH (read directly; family→file map and operator structure verified)
- `./libcint-master/src/{g1e.c,g2e.c,breit.c}` and `src/{g1e.h,g2e.h}` — HIGH (operator macros `G1E_D_*`/`G1E_R0*`/`G2E_*` → `CINTnabla1*`/`CINTx1*` backing functions verified)
- `./libcint-master/include/cint.h.in` — HIGH (`PTR_COMMON_ORIG=1`, `PTR_RINV_ORIG=4` env slots verified)
- `crates/cintx-cubecl/src/kernels/f12.rs` (`nabla1i/j/k_2e`, `gout_ip1/ipip1/ipvip1/ip1ip2`) — HIGH (read; 1st/2nd-order machinery already present and oracle-validated)
- `crates/cintx-cubecl/src/kernels/unstable.rs` (`x1j_breit/x1l_breit`, `nabla*_breit`, `launch_breit`, `gout_breit_r1p2/r2p2`, `BreitShape`) — HIGH (read; position op + Breit σ·p sandwich + 4D spinor already present)
- `crates/cintx-cubecl/src/transform/c2spinor.rs` (`cart_to_spinor_si`, `cart_to_spinor_sf_4d`) — HIGH (Pauli/spin-included coupling present from Phase 12)
- `crates/cintx-cubecl/src/math/obara_saika.rs` — HIGH (VRR/HRR reuse + CubeCL `#[cube]` constraint doc header)
- `crates/cintx-cubecl/src/executor.rs`, `capability.rs` — HIGH (`SHADER_F64` gate; host-vs-device split confirmed)
- `crates/cintx-runtime/src/planner.rs` (`OperatorEnvParams.rinv_orig`, `OutputLayoutMetadata.complex_interleaved`) — HIGH (env-slot + complex-layout pattern for the new `PTR_COMMON_ORIG` slot)
- `crates/cintx-ops/generated/compiled_manifest.lock.json` (172 entries, 145 oracle-covered) — HIGH (registration target)
- `Cargo.toml` / `Cargo.lock` (`num-complex 0.4.6`, `cubecl 0.10.0`) — HIGH (no new crate needed)

---
*Stack research for: v1.4 full libcint 6.1.3 family parity (6 family groups)*
*Researched: 2026-05-27*
