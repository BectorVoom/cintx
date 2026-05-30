# Phase 23: Group 1 — Remaining 1st-Derivative Families (cart/sph) - Research

**Researched:** 2026-05-30
**Domain:** libcint-compatible 1st-derivative integral families (clusters A & B); CubeCL `#[cube]` device-kernel composition + manifest/RawApiId/vendor-FFI/oracle registration
**Confidence:** HIGH (all findings verified against vendored libcint 6.1.3 source + cintx source)

## Summary

Phase 23 clusters A & B add six plain first-derivative families. Cluster C (rank-9 both-side
1e) is already complete (commit `319d055`) and serves only as the registration template.
This research read every relevant libcint autocode gout (`grad2.c`, `int3c2e.c`, `int3c1e.c`)
and the corresponding cintx kernels, and resolved all four "Claude's Discretion" open questions.

The headline finding overturns the CONTEXT framing of cluster B as a symmetric "3c1e pair":
**`int3c1e_ip1` and `int3c1e_iprinv` are NOT the same base integral.** `int3c1e_ip1` is the
derivative of the 3-center **overlap** (Gaussian product, no Rys roots) — cintx already has
that base (`fill_g_tensor_3c1e`), so it is pure Phase-21-style reuse. `int3c1e_iprinv` is the
derivative of the 3-center **Coulomb-1/r-at-a-single-origin** integral (`INT1E_TYPE_RINV`,
Rys-quadrature-based via `CINTg3c1e_nuc`) — cintx has **no** base 3c1e-nuclear kernel today, so
iprinv requires building a new Rys-driven 3c1e g-tensor before the derivative can be taken. This
is the only family in clusters A & B that is not a mechanical reuse of an existing base kernel,
and the planner must budget for it as new (small) kernel math.

Clusters A and the `int3c1e_ip1` half of B are genuinely mechanical: each is `existing base
g-tensor at elevated headroom → an existing nabla operator → the existing 3-component contraction
→ the proven D-11 registration recipe`. The per-family nabla choice, headroom center,
component_rank (all 3), nroots formula, and closest clone-source are tabulated below.

**Primary recommendation:** Plan clusters in three waves — (A1) `int2e_ip2` + `int2c2e_ip1` +
`int2c2e_ip2` (pure nabla1{j,k,i}-on-existing-base reuse), (A2) `int3c2e_ip2` (needs a `nabla1l_2e`
or a remapped 3c2e g-tensor — see Pitfall 2), (B) `int3c1e_ip1` (overlap reuse) + `int3c1e_iprinv`
(new Rys 3c1e-nuc base + derivative). Register every family with `"component_rank":"3"` via the
D-11 recipe; gate each with a non-square bra/ket block; fail-closed when `nroots > 5`.

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Split the 8 families into ~3 PLAN.md clusters by shared kernel reuse:
  (A) ket/remaining-center rank-3 reusing `nabla1j/k_2e` [`int2e_ip2`, `int3c2e_ip2`,
  `int2c2e_ip1`, `int2c2e_ip2`]; (B) the 3c1e pair [`int3c1e_ip1`, `int3c1e_iprinv`];
  (C) both-side rank-9 1e [DONE].
- **D-02:** Sequence rank-3 clusters (A, B) first, rank-9 (C) last. (C is done; A & B remain.)
- **D-03:** Target the maximum each family reaches within the existing nroots≤5 ceiling — no
  Wheeler work. 2e/3c/2c group caps at **d** (the 4-center L sum + derivative `+1` overflows
  past d, same wall as Phase-21 `int2e_ip1`). The `executor.rs:11` `ang_momentum > 4` gate blocks
  g/h everywhere. (NOTE — research refines this per family: int2c2e and int3c1e_ip1 actually reach
  **f** within the ceiling; see "Per-family nroots & max-l" below.)
- **D-06:** Spinor reps for all families are registered in the manifest but return `UnsupportedApi`.
- **D-07:** Reuse the Phase-21 gradient engine verbatim: `gout_ip1` + `nabla1i/j/k_2e` (`f12.rs:590-785`)
  for the 2e/3c/2c families; `CINTnabla1i_1e`/`CINTnabla1j_1e` for the 1e families.
- **D-08:** `int3c1e_iprinv` reuses the existing `PTR_RINV_ORIG` env slot (`env[4..6]`) as-is.
- **D-09:** Surface scope = manifest + RawApiId + kernel + vendor-FFI + oracle only. No `capi`
  enum variants, no legacy `cint*` wrappers.
- **D-10:** Verification = per-family byte-identity at **atol=1e-12** vs vendored libcint 6.1.3,
  cart + sph, every component, in `vendor_*` parity tests double-gated on `--features cpu` +
  `CINTX_ORACLE_BUILD_VENDOR=1`.
- **D-11:** The 5-step registration recipe (manifest → RawApiId → launcher dispatch → vendor FFI →
  oracle test). Lock edits auto-sync the `manifest-audit` (both sides derive from the lock; no
  separate fixtures list). `manifest-audit` (no flags) is the gate and must be green.
- **D-13 precedent:** Fail closed (`UnsupportedApi`) when `nroots > 5` (device Rys ceiling).
- **D-14:** Register **rank 3** for clusters A/B and assert it in the oracle test (pin the element
  count = `3*n_ao_product` AND assert `any_nonzero`) so a stub/short buffer can't pass parity.

### Claude's Discretion

- Exact oracle fixtures / shell-tuple coverage beyond the s/p/d(/f) minimum per family.
- Center-index selection detail for the cluster A/B rank-3 families at impl (the `+1` headroom
  goes on the derivative center; apply the non-square-block discipline). **→ RESOLVED by this
  research; see "Per-family findings" below.**
- Whether `int3c2e_ip2` needs anything beyond the Phase-21 `int3c2e_ip1` repair as a base.
  **→ RESOLVED: yes — it needs a nabla on the auxiliary `k` center, which in cintx's 3c2e
  layout maps to the 2e `ll` slot; see Pitfall 2.**

### Deferred Ideas (OUT OF SCOPE)

- Full f/g coverage for the 2e/3c/2c families (needs nroots≥6 Wheeler/Jacobi fallback +
  `executor.rs` l>4 gate raise) — Phase 25 (`rys-nroots-ge6-wheeler-fallback`).
- `capi` enum variants and legacy `cint*` wrappers (ROADMAP SC6).
- Spinor gradient kernels (registered but `UnsupportedApi`).
- `int3c1e_p2` operator-blind misnomer fix.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DRV1-01 | `int2e_ip2` (∇ on ket bra-center) cart+sph atol=1e-12 | gout `CINTgout2e_int2e_ip2` (grad2.c:101) uses `G2E_D_K` (nabla1k_2e); clone `launch_two_electron_ip1`; nroots `(li+lj+lk+1+ll)/2+1` |
| DRV1-02 | `int1e_ipovlpip/ipkinip/ipnucip` (∇ both sides) | **DONE** in commit `319d055` — confirmed as the registration template; do not re-implement |
| DRV1-03 | `int3c1e_ip1` and `int3c1e_iprinv` cart+sph | ip1 = ∇ on bra of 3c **overlap** (existing `fill_g_tensor_3c1e`, no Rys); iprinv = ∇ on bra of 3c **rinv-Coulomb** (`CINTg3c1e_nuc`, NEW Rys base needed) |
| DRV1-04 | `int2c2e_ip1` and `int2c2e_ip2` cart+sph | ip1 = `G2E_D_I` (nabla1i, bra center i); ip2 = `G2E_D_K` (nabla1k, ket center k); 2c so j=l=0; reaches f within ceiling |
| DRV1-05 | `int3c2e_ip2` cart+sph | gout `CINTgout2e_int3c2e_ip2` (int3c2e.c:99) uses `G2E_D_K` on auxiliary k; cintx 3c2e maps k→2e `ll` slot ⇒ needs `nabla1l_2e` or g-tensor remap (Pitfall 2) |

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Derivative G-tensor + nabla + contraction | CubeCL device kernel (`#[cube]`) | — | Project constraint: CubeCL is primary compute backend; cluster-C precedent runs all numeric core on-device |
| Headroom-shape planning, nroots fail-close, operator dispatch | Host (cintx-cubecl launcher) | — | Planning/validation/marshaling stays host (CLAUDE.md) |
| Manifest entry / RawApiId / surface | Host (cintx-ops, cintx-compat) | — | Registration is build-time + compat-layer plumbing |
| Vendor byte-identity verification | Host (cintx-oracle FFI + tests) | — | Oracle harness drives vendored libcint 6.1.3 over C ABI |

## Per-family findings (the plan-ready table)

For each family: **(a)** libcint gout source, **(b)** nabla + headroom center, **(c)** component_rank,
**(d)** nroots formula + max-l within ceiling, **(e)** closest existing family to clone.

### Cluster A

#### `int2e_ip2` (DRV1-01)
- **(a)** `CINTgout2e_int2e_ip2`, `libcint-master/src/autocode/grad2.c:101`. Single tensor
  `g1 = G2E_D_K(g0, i_l, j_l, k_l, l_l)`; mixing `s[0]=g1x·g0y·g0z`, `s[1]=g0x·g1y·g0z`,
  `s[2]=g0x·g0y·g1z` (identical structure to ip1, just the nabla center differs). ng `{0,0,1,0,1,1,1,3}`.
  `[VERIFIED: grad2.c]`
- **(b)** ∇ on **k** (the bra-center of the 2nd electron, "ket bra-center"). Headroom `+1` on **k**:
  build the plain Coulomb G-tensor with `lk → lk+1`, then `nabla1k_2e` reads index `lk+1`. Apply
  `ak` (the k-shell exponent) as the `-2α` factor. `[VERIFIED: grad2.c G2E_D_K + f12.rs nabla1k_2e]`
- **(c)** component_rank = **3**.
- **(d)** `nroots = (li + lj + (lk+1) + ll)/2 + 1`. Same arity-4 wall as ip1. For dddd:
  `(2+2+3+2)/2+1 = 4+1 = 5` ✓; pppp+1: fine. f anywhere overflows → fail-closed. Max-l: **d**.
- **(e)** Clone `launch_two_electron_ip1` (`two_electron.rs:1434`) — swap `build_2e_shape(li+1,...)`
  → `build_2e_shape(li, lj, lk+1, ll)`, swap `nabla1i_2e`→`nabla1k_2e`, pass `ak` not `ai`. Operator
  branch at `two_electron.rs:1770` (`operator_name() == "ip1"`) → add `== "ip2"`.

#### `int2c2e_ip1` (DRV1-04)
- **(a)** `CINTgout2e_int2c2e_ip1` (static), `int3c2e.c:314`. `g1 = G2E_D_I(g0, i_l, 0, k_l, 0)`
  — note j-slot and l-slot are 0 (2-center). `[VERIFIED: int3c2e.c:321]`
- **(b)** ∇ on **i** (bra center). Headroom `+1` on i. `int2c2e` has only centers i (bra) and k
  (ket); j=l=0. `[VERIFIED]`
- **(c)** component_rank = **3**.
- **(d)** `nroots = (li + lk)/2 + 1` (cintx `center_2c2e.rs:638`). With ip1 → `(li+1+lk)/2+1`.
  ff: `(4+3)/2+1 = 3+1 = 4` ✓; even gg `(5+4)/2+1=4+1=5` ✓ (but blocked by l>4 gate).
  Max-l within nroots: **f** (the `executor.rs:11` l>4 gate is the real cap, not nroots).
- **(e)** `center_2c2e.rs::launch_center_2c2e_typed` (`:608`) is scalar-only today — **add** an
  operator dispatch (`operator_name()`) before the existing scalar path, plus a derivative kernel
  cloning the scalar `center_2c2e_scalar_kernel` with a `nabla1i`-on-i step. The 2c2e g-tensor is
  2e-style Rys (`fill_g_tensor` analog), so `nabla1i_2e` + `gout_ip1` apply with `lj=ll=0`.

#### `int2c2e_ip2` (DRV1-04)
- **(a)** `CINTgout2e_int2c2e_ip2` (static), `int3c2e.c:392`. `g1 = G2E_D_K(g0, i_l, 0, k_l, 0)`.
  `[VERIFIED: int3c2e.c]`
- **(b)** ∇ on **k** (ket center). Headroom `+1` on k. `[VERIFIED]`
- **(c)** component_rank = **3**.
- **(d)** `nroots = (li + lk+1)/2 + 1`. Symmetric to ip1. Max-l: **f**.
- **(e)** Same as int2c2e_ip1 but `nabla1k_2e` on the k center, exponent `ak`. Share one launcher
  with operator branching (`"ip1"`/`"ip2"`).

#### `int3c2e_ip2` (DRV1-05)
- **(a)** `CINTgout2e_int3c2e_ip2`, `int3c2e.c:99`. `g1 = G2E_D_K(g0, i_l, j_l, k_l, 0)` — nabla on
  the auxiliary **k** center (the single ket function). ng `{0,0,1,0,1,1,1,3}`. `[VERIFIED]`
- **(b)** ∇ on the auxiliary **k**. **WARNING (Pitfall 2):** cintx's 3c2e g-tensor builder maps the
  real auxiliary k into the **2e `ll` slot** (`build_2e_shape(li+1, lj, 0, lk)`, phantom 2e `lk`=0,
  real k → `ll`) — see `two_electron.rs:92`. So in cintx coordinates the ip2 derivative is on the
  **`ll` slot**, which requires `nabla1l_2e` — **and `nabla1l_2e` does not exist in `f12.rs`** (only
  i/j/k). Two viable plans: (1) add a `nabla1l_2e` (mirror of `nabla1k_2e`; a `nabla1l_breit`
  already exists in `breit.rs:1206` as a reference), feed it the real-k headroom shape
  `build_2e_shape(li, lj, 0, lk+1)`; or (2) build the 3c2e g-tensor with k in the 2e `lk` slot for
  this family and reuse `nabla1k_2e`. Plan (1) is the smaller, lower-risk change. `[VERIFIED: f12.rs
  has no nabla1l_2e; breit.rs:1206 has nabla1l_breit]`
- **(c)** component_rank = **3**.
- **(d)** `nroots = (li + lj + (lk+1))/2 + 1` (3c2e has 3 real centers; 4th slot phantom). For ddd:
  `(2+2+3)/2+1 = 3+1 = 4` ✓. Max-l: **d**.
- **(e)** Clone `launch_center_3c2e_ip1` (`center_3c2e.rs:1641`) and its device kernel
  `center_3c2e_ip1_kernel` (`:907`) — swap the headroom to the k center and the nabla to
  `nabla1l_2e` (or remap per Pitfall 2). Operator branch at `center_3c2e.rs:1906`
  (`operator_name() == "ip1"`) → add `"ip2"`.

### Cluster B

#### `int3c1e_ip1` (DRV1-03)
- **(a)** `CINTgout1e_int3c1e_ip1` (static), `int3c1e.c:133`. `g1 = G1E_D_I(g0, i_l, j_l, k_l)`,
  **no Rys roots** (single g0/g1, `nf` loop with no nrys sum). The wrapper calls
  `CINT3c1e_drv(..., int_type=0, ...)` = `INT1E_TYPE_OVLP`. `[VERIFIED: int3c1e.c:140,167]`
- **(b)** ∇ on **i** (bra center), 1e-style nabla `∂χ_l = -2α χ_{l+1} + l χ_{l-1}` on the **3-center
  Gaussian-product overlap** g-tensor. Headroom `+1` on i (`li_ceil = li+1`, ng `{1,0,0,0,...}`).
  cintx already computes this exact base in `fill_g_tensor_3c1e` (`center_3c1e.rs:555`, explicitly
  "no Rys quadrature"). `[VERIFIED]`
- **(c)** component_rank = **3**.
- **(d)** **No Rys roots** ⇒ no nroots ceiling. The only cap is `executor.rs:11` l>4. Max-l: **f**
  (and would reach g if the l>4 gate were raised). This is the most permissive family in the phase.
- **(e)** `center_3c1e.rs::launch_center_3c1e_typed` (`:701`) — add an operator dispatch + a
  derivative path that builds `fill_g_tensor_3c1e` with `li+1` headroom, applies a 1e nabla on i,
  and contracts 3 components (clone the `contract_3c1e_ovlp` pattern × 3 axes).

#### `int3c1e_iprinv` (DRV1-03) — **the only genuinely-new base kernel in A & B**
- **(a)** `CINTgout1e_int3c1e_iprinv` (static), `int3c1e.c:78`. The gout is **byte-identical to
  `int3c1e_ip1`'s gout** (`g1 = G1E_D_I(g0, i_l, j_l, k_l)`, same 3-axis mixing). BUT the wrapper
  calls `CINT3c1e_drv(..., int_type=1, ...)` = `INT1E_TYPE_RINV` (`int3c1e.c:112,121`), which routes
  to `CINT3c1e_nuc_loop_nopt(gctr, envs, 1, -1, ...)` — a **Rys-quadrature** loop
  (`cint3c1e.c:327 CINTrys_roots`) over a single rinv origin `cr = &env[PTR_RINV_ORIG]`
  (`cint3c1e.c:267`), using the dedicated g-tensor `CINTg3c1e_nuc` (`g3c1e.c:192`). `[VERIFIED]`
- **(b)** ∇ on **i** (bra center) — but of the **3-center Coulomb-1/r-at-rinv-origin** integral, NOT
  the overlap. **cintx has no base 3c1e-nuclear kernel.** The planner must port `CINTg3c1e_nuc`:
  structurally it is `fill_g_tensor_3c1e` (same VRR/HRR shape) **plus** a Rys-root `t2` parameter
  folded into the recurrence — `aijk1 = 0.5*(1-t2)/aijk` and `rjr0 = rj - (rijk + t2*(cr - rijk))`
  (`g3c1e.c:217,221-226`), summed over `nrys_roots` roots with `x = aijk·dist²(rijk, cr)·tau²`
  (`cint3c1e.c:303-327`). The `+1` headroom on i and the 1e nabla on i are then the same as ip1.
  `[VERIFIED: g3c1e.c:192-235, cint3c1e.c:220-340]`
- **(c)** component_rank = **3**.
- **(d)** `nrys_roots = (li_ceil + lj_ceil + lk_ceil)/2 + 1` (`g3c1e.c:41`). With ip-headroom
  `li_ceil = li+1`: `nroots = (li+1 + lj + lk)/2 + 1`. ddd: `(3+2+2)/2+1 = 3+1 = 4` ✓; fff:
  `(4+3+3)/2+1 = 5+1 = 6` > 5 → **fail-closed** (D-13). Max-l: **d** (with room for some mixed
  higher cases ≤ nroots 5).
- **(e)** No direct clone exists. Build a new `fill_g_tensor_3c1e_nuc` (extend
  `fill_g_tensor_3c1e` with the `t2` Rys parameter) + a Rys-root loop reusing `rys_roots_host`
  (the 2e/2c2e/3c2e Rys path) + the same 1e nabla-on-i + 3-component contraction. The rinv origin
  comes from the **already-plumbed** `OperatorEnvParams.rinv_orig` / `PTR_RINV_ORIG = env[4..6]`
  (Phase 21 D-01), so **no new env plumbing** (D-08). Charge factor is dropped (single origin,
  `fac=1, -1` per `CINT3c1e_nuc_loop_nopt(..., 1, -1, ...)`).

## Standard Stack

No new libraries. This phase composes existing cintx machinery (verified versions per CLAUDE.md):

| Component | Location | Purpose |
|-----------|----------|---------|
| `cubecl` 0.10.0 | workspace pin | `#[cube]` device kernels (all numeric core on-device per cluster-C precedent) |
| `nabla1{i,j,k}_2e` + `gout_ip1` | `crates/cintx-cubecl/src/kernels/f12.rs:602,641,688,744` | 2e/2c/3c2e derivative engine (reuse verbatim) |
| `nabla1l_breit` | `crates/cintx-cubecl/src/kernels/unstable/breit.rs:1206` | **reference** for a needed `nabla1l_2e` (int3c2e_ip2, Pitfall 2) |
| `CINTnabla1i_1e` / `CINTnabla1j_1e` | `one_electron.rs:1864,1772` | 1e nabla (used by 3c1e_ip1/iprinv on the i center) |
| `fill_g_tensor_3c1e` | `center_3c1e.rs:555` | 3c1e overlap g-tensor (int3c1e_ip1 base; extend with `t2` for iprinv) |
| `rys_roots_host` | 2e/2c2e/3c2e Rys path | Rys roots/weights (needed for int3c1e_iprinv's new nuclear base) |
| `parse_component_multiplier` | `cintx-runtime/src/planner.rs:403` | auto-allocates `3×` staging from `"component_rank":"3"` — no manual layout code |
| `OperatorEnvParams.rinv_orig` / `PTR_RINV_ORIG=env[4..6]` | `planner.rs:44`, `raw.rs:33-41` | rinv origin for int3c1e_iprinv (already plumbed, D-08) |

**No installation step** — all dependencies are in `Cargo.lock` and `rust-toolchain.toml` (1.94.0).

## Architecture Patterns

### Data flow (per derivative family)

```
operator_name() dispatch (host launcher)
  │
  ├─ fail-closed guard: nroots > 5 → UnsupportedApi   (skip for int3c1e_ip1: no Rys)
  ├─ spinor → UnsupportedApi (D-06)
  │
  ▼
build base G-tensor at ELEVATED headroom on the derivative center
  (2e/2c/3c2e: build_2e_shape with +1 on i/j/k/l; Rys roots)
  (3c1e_ip1: fill_g_tensor_3c1e with li+1; NO Rys)
  (3c1e_iprinv: NEW fill_g_tensor_3c1e_nuc with li+1 + t2 Rys loop)
  │
  ▼
apply nabla on the derivative center → g1   (nabla1{i,j,k,l}_2e OR 1e nabla on i)
  │
  ▼
3-component contraction: s[0]=g1x·g0y·g0z, s[1]=g0x·g1y·g0z, s[2]=g0x·g0y·g1z
  (gout_ip1 for 2e-style; per-axis for 1e/3c1e style)
  │
  ▼
component-leading [3, …] F-order → cart_to_sph_3c2e/_3c1e per component (sph rep)
  │
  ▼
staging (3 × ni·nj[·nk]) written by parse_component_multiplier's auto-layout
```

### Pattern: operator-symbol dispatch within a canonical-family launcher

All variants of a family share ONE launcher; route by `plan.descriptor.operator_name()`:
```rust
// Source: two_electron.rs:1770 (int2e), center_3c2e.rs:1906 (int3c2e), one_electron.rs:3764 (1e)
if plan.descriptor.operator_name() == "ip1" { return launch_..._ip1::<F>(...); }
// ADD: if operator_name() == "ip2" { return launch_..._ip2::<F>(...); }
```
`center_2c2e.rs:608` and `center_3c1e.rs:701` are scalar-only today and need this dispatch ADDED.

### Pattern: fail-closed nroots guard (D-13)

```rust
// Source: two_electron.rs:1458, center_3c2e.rs:1665
let grad_shape = build_2e_shape(/* +1 on derivative center */);
if grad_shape.nroots > 5 {
    return Err(cintxRsError::UnsupportedApi {
        requested: format!("unsupported_nrys_roots:{}", grad_shape.nroots),
    });
}
```
(Omit for `int3c1e_ip1` — overlap, no Rys. Keep for `int3c1e_iprinv` — has Rys.)

### Anti-patterns to avoid
- **Square bra/ket test blocks:** a square block is transpose-symmetric and hides a layout/axis
  bug (the spinor-orientation + cluster-C lesson). Gate every family with a NON-SQUARE block
  (e.g. p×d) — applies to the rectangular i×j (and i×k for 3-center) blocks, not just rank-9.
- **Reusing nabla1k for int3c2e_ip2 without checking the slot mapping** — cintx maps real-k to the
  2e `ll` slot, so the naive `nabla1k_2e` touches the phantom slot. See Pitfall 2.
- **Assuming int3c1e_iprinv == int3c1e_ip1** — same gout, different base integral (overlap vs
  rinv-Coulomb). See Pitfall 1.
- **component_rank too low** — silently truncates trailing components (D-14 / the 260530-9ay
  landmine). Always `"3"` and assert element-count + any_nonzero in the oracle test.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| 3× staging layout for rank-3 output | Manual buffer striding | `parse_component_multiplier` ("3") | Auto-allocates `3×ni·nj[·nk]`; proven on every gradient family |
| rinv-origin env plumbing | New env slot / setter | `OperatorEnvParams.rinv_orig` (`env[4..6]`) | Already plumbed in Phase 21 (D-08) |
| manifest "generated vs lock" sync | Editing a fixtures family list | Edit the lock only | Both audit sides derive from the lock; `cargo build -p cintx-ops` regenerates (D-11 KEY FINDING) |
| Rys roots/weights | New root finder | `rys_roots_host` (2e/2c2e path) | int3c1e_iprinv's new base must reuse the existing Rys, not a new one |
| `nabla1l_2e` for int3c2e_ip2 | Inventing the recurrence | Mirror `nabla1k_2e` (`f12.rs:688`) / reference `nabla1l_breit` (`breit.rs:1206`) | The G2E_D_L recurrence is already implemented twice in-tree |

**Key insight:** Five of the six families are `existing base + existing nabla + existing
contraction + D-11 recipe`. Only `int3c1e_iprinv` needs new (small, well-templated) base math.

## Common Pitfalls

### Pitfall 1: Treating the 3c1e pair as symmetric (the biggest trap)
**What goes wrong:** Planning `int3c1e_iprinv` as a trivial clone of `int3c1e_ip1` (their libcint
gouts are byte-identical, which invites this mistake).
**Why it happens:** The autocode gouts (`int3c1e.c:78` vs `:133`) ARE identical; the difference is
hidden in the `CINT3c1e_drv` `int_type` argument (0=OVLP no-Rys vs 1=RINV Rys-quadrature).
**How to avoid:** ip1 builds on the existing overlap g-tensor (no Rys); iprinv needs a NEW
Rys-driven 3c1e-nuclear g-tensor (`CINTg3c1e_nuc` port). Budget iprinv as a separate, larger task.
**Warning signs:** A plan that gives both families the same kernel, or skips Rys for iprinv.

### Pitfall 2: int3c2e_ip2 nabla hits the phantom slot
**What goes wrong:** Applying `nabla1k_2e` for int3c2e_ip2 derives the wrong center because cintx
maps the real auxiliary k into the 2e `ll` slot (the 2e `lk` slot is phantom=0).
**Why it happens:** `build_2e_shape(li+1, lj, 0, lk)` for 3c2e (`two_electron.rs:92`) — real k lives
in `ll`, phantom in `lk`.
**How to avoid:** Either add `nabla1l_2e` (mirror `nabla1k_2e`; `nabla1l_breit` is a reference) and
build with `build_2e_shape(li, lj, 0, lk+1)`, or remap the 3c2e g-tensor to put k in the `lk` slot
for this family and reuse `nabla1k_2e`. The `nabla1l_2e` route is smaller and isolated.
**Warning signs:** ip2 output equals ip1 output, or all-zero / wrong-axis components.

### Pitfall 3: `nabla1j_2e` / `nabla1k_2e` are private
**What goes wrong:** Compile error reusing them outside `f12.rs`.
**Why it happens:** Only `nabla1i_2e` and `gout_ip1` were made `pub(crate)` in Phase 21
(`f12.rs:602,744`); `nabla1j_2e` (`:641`) and `nabla1k_2e` (`:688`) are still private `fn`.
**How to avoid:** Promote `nabla1j_2e`/`nabla1k_2e` (and any new `nabla1l_2e`) to `pub(crate)`, and
add a `gout_ip1`-style contraction parameterized by which nabla to apply (gout_ip1 hardcodes
`nabla1i_2e` at `f12.rs:766`). Either generalize `gout_ip1` or add `gout_ip2`/`gout_ipk` variants.
**Warning signs:** "private function" E0603 at the cluster-A launcher.

### Pitfall 4: nroots fail-close vs panic
**What goes wrong:** A high-l quartet panics inside `rys_root1..5` (`rys.rs:3247`) instead of
returning `UnsupportedApi`.
**How to avoid:** Compute the grad_shape nroots and guard `> 5` BEFORE any `rys_roots_host` call
(D-13). Skip the guard for `int3c1e_ip1` (no Rys); keep it for every other A/B family.

### Pitfall 5: vendor parity silently skips
**What goes wrong:** A `vendor_*` test runs `0 tests` (looks green) because the double gate is
missing.
**How to avoid:** Run with BOTH `--features cpu` AND `CINTX_ORACLE_BUILD_VENDOR=1`; assert the test
banner shows `running N>0 tests` (the standing project gotcha).

## Code Examples

### int2e_ip2 launcher skeleton (clone of ip1)
```rust
// Source: cintx two_electron.rs:1434 (launch_two_electron_ip1) — adapt center i→k
fn launch_two_electron_ip2<F: CintFloat>(/* same args */) -> Result<ExecutionStats, cintxRsError> {
    if plan.representation == Representation::Spinor { return Err(UnsupportedApi{..}); }
    // headroom on k (the 2nd-electron bra center)
    let grad_shape = build_2e_shape(li as usize, lj as usize, lk as usize + 1, ll as usize);
    if grad_shape.nroots > 5 { return Err(UnsupportedApi{ requested: format!("unsupported_nrys_roots:{}", grad_shape.nroots) }); }
    // ... fill_g_tensor_2e at elevated lk, then nabla1k_2e + gout (parameterized on k, exponent ak) ...
}
```

### libcint reference: int2e_ip2 gout (the authority for component ordering)
```c
// Source: libcint-master/src/autocode/grad2.c:101
void CINTgout2e_int2e_ip2(double *gout, double *g, FINT *idx, CINTEnvVars *envs, FINT gout_empty) {
    double *g0 = g; double *g1 = g0 + envs->g_size * 3;
    G2E_D_K(g1, g0, envs->i_l+0, envs->j_l+0, envs->k_l+0, envs->l_l);  // nabla on k
    // s[0]=g1x·g0y·g0z; s[1]=g0x·g1y·g0z; s[2]=g0x·g0y·g1z   (3 components, i-fastest)
}
```

### libcint reference: int3c1e_iprinv's Rys base (CINTg3c1e_nuc — the NEW math to port)
```c
// Source: libcint-master/src/g3c1e.c:192 — note t2 folding (vs overlap fill_g_tensor_3c1e)
const double aijk1 = .5 * (1 - t2) / aijk;          // overlap had .5/aijk (t2=0)
rjr0[d] = rj[d] - (rijk[d] + t2 * (cr[d] - rijk[d]));// cr = rinv origin = env[PTR_RINV_ORIG]
// summed over nrys_roots roots; x = aijk * dist^2(rijk, cr) * tau^2   (cint3c1e.c:303-327)
```

## Runtime State Inventory

> Code/registration-only phase. No stored data, live-service config, or OS-registered state.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no databases/datastores embed these symbols | None |
| Live service config | None — pure library code + build-time manifest | None |
| OS-registered state | None | None |
| Secrets/env vars | `PTR_RINV_ORIG` (`env[4..6]`) is an integral-runtime env slot, already plumbed (Phase 21); not a process env var. iprinv reads it; no key change. | None (reuse as-is, D-08) |
| Build artifacts | `crates/cintx-ops/generated/api_manifest.{rs,csv}` regenerate from the lock via `cargo build -p cintx-ops` after lock edits. The bindgen FFI bindings regenerate when `cintx-oracle/build.rs` allowlist changes. | Rebuild `cintx-ops` + `cintx-oracle` after manifest/allowlist edits |

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Vendored libcint 6.1.3 source | Oracle byte-identity (FFI) | ✓ | 6.1.3 (`libcint-master/`) | — |
| `cc` (C compiler for vendored build) | `cintx-oracle/build.rs` | ✓ (CLAUDE.md stack `cc` 1.2.x) | — | — |
| `bindgen` allowlist | FFI symbol generation | ✓ | 0.71.1 | — |
| autocode `.c` sources in build.rs | int2e_ip2 (grad2.c), int3c2e_ip2/int2c2e (int3c2e.c), int3c1e (int3c1e.c) | ✓ | — | — |
| Rust 1.94.0 toolchain | build | ✓ | pinned `rust-toolchain.toml` | — |

**Autocode source-list status (D-11 step 4 confirmation):** `cintx-oracle/build.rs` already compiles
`grad2.c` (`:73`/`:236` — defines `int2e_ip2`), `int3c2e.c` (`:64`/`:224` — defines `int3c2e_ip2`,
`int2c2e_ip1`, `int2c2e_ip2`), and `int3c1e.c` (`:63`/`:223` — defines `int3c1e_ip1`,
`int3c1e_iprinv`). **No new `.c` file needs adding** — only the per-family symbols must be added to
the `allowlist_function` regex at `build.rs:358` (the cluster-C symbols `int1e_ipovlpip_*` etc. are
already there; clone that addition). `[VERIFIED: build.rs:51-80, 207-250, 358]`

**Missing dependencies with no fallback:** None.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (workspace also has `cargo-nextest` available); oracle parity via `#[cfg(has_vendor_libcint)]` |
| Config file | none (cargo test); CI in `.github/workflows/compat-governance-pr.yml` |
| Quick run command | `cargo test -p cintx-cubecl <family>` (device-vs-host, no vendor) |
| Full suite command | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test <family>_parity -- --test-threads=1` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DRV1-01 | int2e_ip2 byte-identity cart+sph | vendor parity | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test int2e_ip2_parity` | ❌ Wave 0 |
| DRV1-04 | int2c2e_ip1/ip2 byte-identity | vendor parity | `... --test int2c2e_ip_parity` | ❌ Wave 0 |
| DRV1-05 | int3c2e_ip2 byte-identity | vendor parity | `... --test int3c2e_ip2_parity` | ❌ Wave 0 |
| DRV1-03 | int3c1e_ip1/iprinv byte-identity | vendor parity | `... --test int3c1e_ip_parity` | ❌ Wave 0 |
| DRV1-02 | int1e_ipovlpip/ipkinip/ipnucip | vendor parity | `... --test one_electron_grad_both_parity` | ✅ (cluster C, done) |
| (all) | device kernel == host reference | unit (cubecl) | `cargo test -p cintx-cubecl <family>` | ❌ Wave 0 (clone cluster-C `test_device_*` pattern) |

### Sampling Rate
- **Per task commit:** `cargo test -p cintx-cubecl <family>` (device-vs-host host-ref test).
- **Per wave merge:** the family's `vendor_*` parity test with the double gate + `cargo build -p cintx-ops` (manifest regen) + `cargo run -p xtask -- manifest-audit` (no flags).
- **Phase gate:** all five new parity tests green at atol=1e-12 + `manifest-audit` green before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `crates/cintx-oracle/tests/int2e_ip2_parity.rs` — covers DRV1-01 (clone `one_electron_grad_both_parity.rs` fixture + the `int2e` shell-quad pattern)
- [ ] `crates/cintx-oracle/tests/int2c2e_ip_parity.rs` — covers DRV1-04
- [ ] `crates/cintx-oracle/tests/int3c2e_ip2_parity.rs` — covers DRV1-05
- [ ] `crates/cintx-oracle/tests/int3c1e_ip_parity.rs` — covers DRV1-03 (both ip1 overlap + iprinv rinv)
- [ ] `crates/cintx-oracle/src/vendor_ffi.rs` — add `vendor_int{2e_ip2,2c2e_ip1,2c2e_ip2,3c2e_ip2,3c1e_ip1,3c1e_iprinv}_{sph,cart}` wrappers (clone the `vendor_int1e_ipovlpip_*` pattern at `:587`)
- [ ] `crates/cintx-cubecl/src/kernels/*` device-vs-host unit tests per family (clone cluster-C `test_device_ip{ovlpip,kinip,nucip}_matches_host_reference`)
- [ ] Framework install: none — built-in test runner.

## State of the Art

| Old (CONTEXT assumption) | Current (verified) | Impact |
|--------------------------|--------------------|--------|
| 3c1e pair is symmetric reuse | ip1 = overlap (no Rys, existing base); iprinv = rinv-Coulomb (Rys, **new** base) | iprinv is a larger task; budget new g-tensor + Rys loop |
| int2c2e/3c2e cap at d | int2c2e reaches **f** within nroots≤5; int3c1e_ip1 reaches **f** (no Rys at all) | Don't artificially cap 2c2e / 3c1e_ip1 at d; the real cap for them is the `executor.rs` l>4 gate |
| int3c2e_ip2 ≈ int3c2e_ip1 repair | ip2 derives the auxiliary k, which sits in cintx's 2e `ll` slot ⇒ needs `nabla1l_2e` | Add a `nabla1l_2e` (mirror nabla1k) or remap the g-tensor |

**No deprecated APIs** — all referenced cintx functions are current as of branch
`fix/general-contraction-nctr-1e` (Phase 21/cluster-C landed).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | int2c2e's g-tensor is 2e-style Rys so `nabla1i_2e`/`gout_ip1` apply with lj=ll=0 | Cluster A int2c2e | If the 2c2e g-tensor layout differs from the 2e shape, the nabla strides need adjustment; verify against `center_2c2e_scalar_kernel` strides at impl |
| A2 | A `nabla1l_2e` mirroring `nabla1k_2e` is the smallest fix for int3c2e_ip2 (vs remapping the g-tensor) | Cluster A int3c2e_ip2 / Pitfall 2 | If the 3c2e HRR `ibase`/`kbase` branch makes the `ll` slot non-trivial, remapping may be cleaner — decide at impl by inspecting `build_2e_shape` for the 3c2e case |
| A3 | int3c1e_iprinv's new base can reuse the existing `rys_roots_host` (same Rys as 2e) | Cluster B iprinv | If libcint's 3c1e-nuc uses a different root convention (it uses standard `CINTrys_roots` per `cint3c1e.c:327`), low risk; confirm root/weight scaling at impl |

**These three are implementation-detail assumptions, not design decisions** — they affect HOW
(stride adjustment, nabla1l vs remap, Rys reuse), not WHETHER, and resolve by reading the named
cintx functions during the task. No user confirmation needed before planning.

## Open Questions

1. **int3c2e_ip2: add `nabla1l_2e` or remap the g-tensor?**
   - What we know: ip2 derives the auxiliary k; cintx maps real-k to the 2e `ll` slot; no
     `nabla1l_2e` exists (but `nabla1l_breit` does as a reference).
   - What's unclear: which is less code/risk given the 3c2e `ibase/kbase` HRR branch.
   - Recommendation: plan `nabla1l_2e` (isolated, mirrors nabla1k); fall back to remap if the
     `ll`-slot strides prove awkward. Either way the oracle parity test is the arbiter.

2. **int3c1e_iprinv host-vs-device split for the new Rys base.**
   - What we know: cluster-C and 3c2e_ip1 run the full numeric core on-device (`#[cube]`).
   - What's unclear: whether the new 3c1e-nuc g-tensor + Rys loop is ported to `#[cube]` in this
     phase or staged host-side first (the CubeCL authoring rules — no plain-fn calls, F::exp/sqrt,
     u32/i32 — apply to any device port).
   - Recommendation: device kernel to match the project's CubeCL-primary constraint and the
     cluster-C precedent; the radial/Rys machinery is already device-proven in 2c2e/3c2e.

## Project Constraints (from CLAUDE.md)

- **CubeCL is the primary compute backend** — derivative kernels are `#[cube]` device kernels; host
  CPU stays limited to planning/validation/marshaling/oracle glue. Device ports MUST follow
  `docs/manual/Cubecl/*.md` (no plain-fn calls, no if-expr, `F::exp`/`F::sqrt`, u32/i32 only, no
  continue/break) — same rules cluster C followed.
- **New-family surface scope** = manifest + RawApiId + kernel + vendor-FFI + oracle ONLY. **No
  `capi` enum variants, no legacy `cint*` wrappers** (confirm `cintx-capi/src/shim.rs` and
  `cintx-compat/src/legacy.rs` are NOT touched — ROADMAP SC6).
- **Public library errors use `thiserror` v2** (`cintxRsError::UnsupportedApi` for fail-close);
  oracle/xtask use `anyhow`.
- **Compatibility target: upstream libcint 6.1.3** — byte-identity at atol=1e-12 is the gate.
- **Cargo `--locked`** in CI; deliverables/artifacts workflow to `/mnt/data` where applicable.

## Sources

### Primary (HIGH confidence)
- `libcint-master/src/autocode/grad2.c:101` — `CINTgout2e_int2e_ip2` (G2E_D_K, ng, component order)
- `libcint-master/src/autocode/int3c2e.c:18,99,314,392` — int3c2e_ip1/ip2, int2c2e_ip1/ip2 gouts + ng
- `libcint-master/src/autocode/int3c1e.c:78,133` — int3c1e_iprinv/ip1 gouts + the `CINT3c1e_drv` int_type arg (`:112,167`)
- `libcint-master/src/cint3c1e.c:267,303-340,393-411` — INT1E_TYPE_OVLP vs RINV routing; Rys loop; rinv origin
- `libcint-master/src/g3c1e.c:13-60,192-235` — `CINTinit_int3c1e_EnvVars` nroots formula; `CINTg3c1e_nuc` t2-folded g-tensor
- cintx `crates/cintx-cubecl/src/kernels/f12.rs:602,641,688,744` — nabla1{i,j,k}_2e + gout_ip1 (privacy + hardcoded nabla1i)
- cintx `two_electron.rs:92,1434,1770`, `center_3c2e.rs:907,1641,1906`, `center_2c2e.rs:608,638`, `center_3c1e.rs:555,701` — launcher/dispatch/g-tensor patterns
- cintx `crates/cintx-oracle/build.rs:51-80,207-250,358` — autocode source list + bindgen allowlist (cluster-C symbols already present)
- cintx `crates/cintx-ops/generated/compiled_manifest.lock.json:307-400` — int3c2e_ip1 entry (clone template, `component_rank:"3"`)
- cintx `crates/cintx-compat/src/raw.rs:123-207` — RawApiId consts (cluster-C entries as template)
- cintx `crates/cintx-oracle/tests/one_electron_grad_both_parity.rs`, `vendor_ffi.rs:587+` — cluster-C parity/FFI template
- `.planning/phases/21-coulomb-gradient-intors/21-CONTEXT.md` — gradient engine, env-slot precedent, nroots formula
- `.planning/phases/23-...-CONTEXT.md` — locked decisions D-01..D-14

### Secondary / Tertiary
- None — every claim is grounded in vendored source or cintx source (no WebSearch/training reliance).

## Metadata

**Confidence breakdown:**
- Standard stack (reuse map): HIGH — every function read in-tree at named line numbers.
- Per-family gout / nabla / nroots: HIGH — read directly from vendored libcint 6.1.3 autocode.
- int3c2e_ip2 nabla1l gap: HIGH — confirmed `nabla1l_2e` absent in f12.rs, present in breit.rs.
- int3c1e_iprinv "new base" finding: HIGH — confirmed via `int_type` arg + `CINTg3c1e_nuc` + cintx's overlap-only `center_3c1e.rs`.
- Implementation-detail assumptions (A1-A3): MEDIUM — resolve by reading named cintx functions during the task; do not block planning.

**Research date:** 2026-05-30
**Valid until:** stable for the life of branch `fix/general-contraction-nctr-1e` (vendored libcint 6.1.3 is pinned; cintx kernels are post-cluster-C). Re-verify line numbers if the engine files are refactored.
