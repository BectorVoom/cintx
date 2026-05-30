# Architecture Research — v1.4 Full libcint 6.1.3 Family Parity

**Domain:** Per-family integration of the ~140 remaining libcint 6.1.3 integral families into cintx's existing manifest→kernel→raw→safe-API→oracle pipeline (milestone v1.4).
**Researched:** 2026-05-27
**Confidence:** HIGH for the existing pipeline shape and env-slot / transform gaps (read from cintx + libcint source); MEDIUM for the exact derivative recurrence reuse boundaries and oracle-fixture sizing per family (extrapolated from the Phase-21 pattern and libcint structure, not yet executed).

> File note: the existing `.planning/research/ARCHITECTURE.md` is the immutable
> v1.2 research record. This v1.4 milestone research is written to a milestone-
> scoped sibling so the historical doc is preserved.
>
> Scope note: this file does NOT re-derive the per-family pipeline (manifest →
> kernel → `eval_raw` → `SessionRequest::evaluate` → vendor FFI + byte-identity
> oracle test → flip `oracle_covered`). That pattern is fixed and reused verbatim.
> It maps each of the **6 v1.4 groups** onto that pipeline: which env slots /
> `OperatorEnvParams` fields, which kernel modules / dispatch arms, which need the
> spinor / 4-component (`si`) path, and which oracle-harness extensions are
> required — then orders the build so foundations land first.

## Standard Architecture (existing — the integration target)

The per-family pipeline (verified in `kernels/mod.rs`, `raw.rs`, `vendor_ffi.rs`,
`build.rs`) is a fixed 6-stage funnel. Every new family threads through all 6:

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. MANIFEST   crates/cintx-ops/generated/compiled_manifest.lock.json       │
│    entry { id{family,operator,representation,symbol}, component_rank,       │
│            profiles, oracle_covered } → build.rs regenerates api_manifest.rs│
├──────────────────────────────────────────────────────────────────────────┤
│ 2. KERNEL     crates/cintx-cubecl/src/kernels/<family>.rs                   │
│    launch_<family>() registered in kernels/mod.rs::resolve_family_name      │
│    (match on canonical_family) — host-side G-tensor + contract + transform  │
├──────────────────────────────────────────────────────────────────────────┤
│ 3. ENV / PLAN crates/cintx-runtime planner.rs::OperatorEnvParams            │
│    crates/cintx-compat/src/raw.rs::eval_raw reads env[] slots, validator    │
│    gates missing params, threads into ExecutionPlan.operator_env_params     │
├──────────────────────────────────────────────────────────────────────────┤
│ 4. RAW + LEGACY + CAPI   raw.rs eval_raw dispatch · legacy.rs wrappers ·    │
│    capi/shim.rs enum variant                                                │
├──────────────────────────────────────────────────────────────────────────┤
│ 5. SAFE API   SessionRequest::evaluate (OperatorId + Representation)        │
├──────────────────────────────────────────────────────────────────────────┤
│ 6. ORACLE     vendor_ffi.rs (FFI sig) + build.rs (cc .file + bindgen        │
│    allowlist + suppl header) + tests/*_parity.rs (byte-identity, double-     │
│    gated: --features cpu + CINTX_ORACLE_BUILD_VENDOR=1) → flip oracle_covered│
└──────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities (what each group must touch)

| Component | Responsibility | What a v1.4 group changes |
|-----------|----------------|---------------------------|
| `compiled_manifest.lock.json` | API source of truth; `component_rank` sets staging multiplier | New entries per symbol×representation; `component_rank` 1/3/6/9/27/81 by derivative/tensor order |
| `kernels/<family>.rs` + `kernels/mod.rs` | Host G-tensor fill, contract, transform; family dispatch | New contract arms (groups 1-3 reuse existing modules); new modules for σ·p (group 4) and gauge 2e (group 6) |
| `planner.rs::OperatorEnvParams` | Typed view of `env[0..PTR_ENV_START]` global params | **New field `common_orig: Option<[f64;3]>`** (group 3 multipole `origj`, group 5 GIAO gauge); group 4 needs no new env slot |
| `raw.rs::eval_raw` | Reads `env[]` slots into `OperatorEnvParams`, validator gate | New `env[PTR_COMMON_ORIG=1..3]` read block (mirrors the Phase-21 `PTR_RINV_ORIG` block) |
| `transform/c2spinor.rs` | cart→spinor coupling | **New `cart_to_spinor_si_*` (spin-included) family + spinor-derivative transform** — the R5/D-03 gap and the σ-operator gap |
| `vendor_ffi.rs` + oracle `build.rs` | Vendored libcint reference + bindgen allowlist | New `cc::Build::file()` for `autocode/intor4.c`, `deriv3/4.c`, `gaunt1.c`, `breit1.c`; extend the bindgen allowlist regex; add suppl-header `extern` decls for symbols absent from `cint_funcs.h` |
| `tests/*_parity.rs` + `fixtures.rs` | Byte-identity gate | New fixtures beyond H2O/STO-3G: a **spinor/relativistic fixture** and a **gauge-origin (`PTR_COMMON_ORIG≠0`) fixture** |

## The Two Foundational Gaps (must precede dependent groups)

These are discovered architectural prerequisites, not optional. Each is its own early phase.

### Gap A — Gauge / common-origin env slot (`PTR_COMMON_ORIG`, env[1..3])

**Evidence:** `OperatorEnvParams` (`planner.rs:43-54`) currently carries only
`f12_zeta`, `grids_params`, `rinv_orig`. libcint's multipole-`origj`, GIAO, and CG
families read `env[PTR_COMMON_ORIG+0..2]` directly (`autocode/intor1.c:223-225,
377-379, 1021-1023, …`). The `raw.rs` env-slot map already documents
`PTR_COMMON_ORIG = 1..3` (`raw.rs:34`) but never reads it.

**What's needed (small, isolated):** add `common_orig: Option<[f64;3]>` to
`OperatorEnvParams`; add an `env[1..4]` read block in `eval_raw` (verbatim shape of
the Phase-21 `PTR_RINV_ORIG` block, `raw.rs:599-616`); add a
`with_common_origin`-style setter on the safe-API options; validator gate for the
families that require it. This unblocks **group 3 (`*_origj` moments)** and
**group 5 (all GIAO/CG)**. Low complexity — it is the `rinv_orig` precedent again.

### Gap B — Spinor "si" (spin-included) transform + spinor-derivative transform

This is the dominant prerequisite and the single largest architectural addition in v1.4.

**Two distinct missing transforms, both in `transform/c2spinor.rs`:**

1. **Spinor-derivative transform (R5/D-03).** Every gradient kernel today hard-rejects
   `Representation::Spinor` (`one_electron.rs:901-908`, `two_electron.rs:630-635`,
   the `int2e_ip1` guard). The cart→spinor path only has the SCALAR `cart_to_spinor_sf_2d/4d/3c2e`
   (`c2spinor.rs:531, 879, 1281`). A derivative integral emits a 3-component
   (`[3, …]`) Cartesian block per primitive; the spinor transform must be applied
   per derivative component AND fold the component axis correctly. This is a hard
   prerequisite for **spinor variants of every `ip*` family** across groups 1, 2, 4, 5, 6.

2. **Spin-included (`si`) transform — the σ-operator coupling.** Verified in
   libcint `cart2sph.c::c2s_si_1e` (lines 4947-4992): the spin-DEPENDENT transform
   consumes a **4-block** G-tensor (`gc_x, gc_y, gc_z, gc_1` = the three σ-Pauli
   components plus the scalar) via `a_bra_cart2spinor_si`, NOT the single scalar
   block that `cart_to_spinor_sf` (cintx `c2spinor.rs:35-78`) handles. There is no
   `c2s_si_*` analogue in cintx (only the `sf` family). Without it, NO relativistic
   spin-operator family (`spsp`, `spnucsp`, `sprinvsp`, `srsr`, `sigma`, the
   `ssp/sps/vsp` 2e families, gauge/Gaunt) can produce a libcint-matching spinor
   result.

**Verdict:** Gap B should be decomposed into **(B1) spinor-derivative transform**
(unblocks the cheaper spinor `ip*` matrix) and **(B2) the `si` transform + a σ·p /
`spsp` scalar-G-tensor module** (unblocks the relativistic spin-operator block).
B1 and B2 are independent and can run in parallel after the scalar derivative math
is in place.

## Per-Group Integration Plan

### Group 1 — Remaining 1st-derivative families

**Symbols (from libcint `grad1.c`, `hess.c`, `autocode/int3c1e.c`, `int3c2e.c`):**
`int2e_ip2`, `int1e_ipnucip/ipkinip/ipovlpip` (ket-side / both-side), `int3c1e_ip1`,
`int3c1e_iprinv`, `int2c2e_ip1/ip2`, `int3c2e_ip2`.

| Integration point | Detail |
|-------------------|--------|
| Env slots | None new (cart/sph). `iprinv` variants reuse `PTR_RINV_ORIG` (already plumbed, Phase 21). |
| Kernel module | **Reuse existing**: extend `one_electron.rs` (1e ket/both-side nabla — mirror `contract_grad_1e_bra` / `contract_ipkin`), `two_electron.rs::gout_ip1` (ip2 = ket-side ∇, same `nabla1i_2e` math from `f12.rs` on the k/l index), `center_3c1e.rs`, `center_2c2e.rs`, `center_3c2e.rs::int3c2e_ip1` pattern. No new dispatch arm — these are operator-name branches inside existing `launch_*` fns. |
| `component_rank` | `3` (single derivative). Already a present rank in the manifest. |
| Spinor path | Spinor variants depend on **Gap B1** (spinor-derivative transform). Cart/sph variants need nothing new. |
| Oracle harness | New FFI sigs in `vendor_ffi.rs` (same `CINTIntegralFunction` shape); the `.c` files (`grad1.c`, `hess.c`, `autocode/int3c1e.c`, `int3c2e.c`) are ALREADY in `build.rs` `cc::Build`; just extend the bindgen allowlist regex (`build.rs:358`) and add suppl-header `extern` decls for any symbol not in `cint_funcs.h`. Fixtures: H2O/STO-3G suffices for cart/sph. |
| Complexity | **Low** — most direct extension of Phase 21. |

### Group 2 — Hessian and higher-order derivatives

**Symbols:** `int1e_ipip*` (`ipipovlp`, `ipipkin`, `ipipnuc`, `ipiprinv`, `ipvip`),
`int2e_ipip1/ipvip1/ip1ip2` (already half-registered — `int2e_ipip1_sph`,
`int2e_ipvip1_sph` appear in `compare.rs:314-315`; the manifest already has
`component_rank: "9"` entries), `int2c2e_ipip1`, `int3c2e_ipip1/ipip2`, and the
4th-order `ipipipiprinv` (from `deriv4.c`).

| Integration point | Detail |
|-------------------|--------|
| Env slots | None new beyond `PTR_RINV_ORIG`. |
| Kernel module | **Reuse + extend**: the derivative engine is `nabla1i` applied N times. 2nd-order needs a nabla that raises BOTH bra indices (`ipip`, rank 9) or bra+ket (`ipvip`). Build on `one_electron.rs` / `f12.rs::gout_ip1` by composing the nabla step. The existing kinetic kernel ALREADY does a `D_j^2` second derivative (`contract_kinetic`, `one_electron.rs:208`), so the 2nd-order machinery partially exists. |
| `component_rank` | `9` (2nd-order 3×3), `27` (3rd), `81` (4th `ipipipip`). The planner multiplies staging by `component_rank` (Phase-21 verified). |
| Spinor path | `ipip` spinor variants depend on **Gap B1**. |
| Oracle harness | `hess.c` is compiled; `deriv3.c`, `deriv4.c` are NOT yet in `build.rs` `cc::Build` — add them. Extend allowlist + suppl header. Higher nroots from doubly-raised li may hit the **rys nroots>5 ceiling** (the Phase-21 fail-closed guard, `two_electron.rs:642`) — expect more `UnsupportedApi` for high-l quartets; acceptable and matches the existing contract. |
| Complexity | **Medium** — recurrence composition + nroots pressure. Depends on Group 1 (1st-order engine generalized). |

### Group 3 — Position / multipole moment integrals

**Symbols (libcint `intor1.c`):** `int1e_r/rr/rrr/rrrr`, `int1e_r2/r4`,
`int1e_z/zz`, `int1e_sp`, `int1e_p4`, plain `int1e_rinv`, `int1e_drinv`,
`int1e_irp/irpr/irrp`, and the `_origj` variants (`int1e_r_origj`, `int1e_rr_origj`,
`int1e_r2_origj`, `int1e_z_origj`, `int1e_zz_origj`, …).

| Integration point | Detail |
|-------------------|--------|
| Env slots | `_origj` variants read `PTR_COMMON_ORIG` (env[1..3]) → **needs Gap A**. Non-`origj` moments (`int1e_r`, `int1e_rr`) use bra/ket centers only, no env slot. `int1e_rinv`/`drinv` reuse `PTR_RINV_ORIG`. |
| Kernel module | **Reuse `one_electron.rs`** with a new "position-operator G-tensor" contract arm. The `r` operator is a rank-1 tensor inserted between bra and ket on the Obara-Saika recurrence (raises effective angular momentum on one center by the multipole order). `int1e_p4` (∇⁴) reuses the kinetic `D_j^2` machinery squared. No new module — new contract functions. |
| `component_rank` | `r`→3, `rr`→6 (or 9 unsymmetrized), `rrr`→10, `rrrr`→15, `z`→1, `zz`→1, `r2`→1, `p4`→1. Verify exact ranks against libcint `comp` per symbol when registering. |
| Spinor path | `int1e_sp` (σ·p) and any spin-carrying moment depend on **Gap B2** (the `si` transform). Pure-scalar moments (`r`, `rr`, `z`) are cart/sph only — no spinor dependency. |
| Oracle harness | `intor1.c` and `cint1e_a.c` are ALREADY compiled in `build.rs`. Add FFI sigs + allowlist + suppl-header `extern` decls. For `_origj` parity, add a **gauge-origin fixture** (H2O/STO-3G with `env[PTR_COMMON_ORIG]≠0`). |
| Complexity | **Low–Medium**. The position-operator G-tensor is the one genuinely new piece of math; the rest is registration + env plumbing. |

### Group 4 — Relativistic spin-operator integrals

**Symbols (libcint `intor1.c`, `cint1e_a.c`, `intor4.c`):** `int1e_spsp`,
`int1e_spnucsp`, `int1e_sprinvsp`, `int1e_srsr`, `int1e_sigma`, `int1e_spsigmasp`,
`int1e_pnucp/prinvp`, `int2e_spsp1`, `int2e_srsr1`, `int2e_ssp1ssp2`,
`int2e_sps1sps2`, `int2e_ssp1sps2`/`sps1ssp2`, and the `ip`-decorated spin
gradients (`int1e_ipspnucsp`, `int1e_ipsprinvsp`, `int1e_ipspnucspip`).

| Integration point | Detail |
|-------------------|--------|
| Env slots | `*rinv*` reuse `PTR_RINV_ORIG`; no new slot. |
| Kernel module | **New module `kernels/spin_operator.rs`** (the σ·p machinery). σ·p produces a **4-block** G-tensor (3 σ-component blocks + scalar) — exactly the input the `si` transform consumes. Genuinely new compute: `p` = ∇ raises angular momentum (reuse the nabla step), then the three Pauli components are assembled. New dispatch arm `"spin"` in `resolve_family_name`. |
| `component_rank` | Operator-dependent; `sigma` is rank-3 spin, `spsp` collapses to the AO block after the `si` transform. Map per symbol. |
| Spinor path | **Hard dependency on Gap B2** (the `si` spin-included transform) AND Gap B1 for the `ip`-decorated members. These families are spinor-NATIVE — there is no useful cart/sph result for most; libcint exposes them primarily as `_spinor`. The group most blocked by foundations. |
| Oracle harness | `intor1.c`/`cint1e_a.c` are compiled; spin 2e (`int2e_spsp1` etc.) come from `autocode/intor4.c` (NOT yet in `build.rs` — add it). Add FFI sigs + allowlist + suppl `extern` decls. **Needs a relativistic/spinor fixture** (a molecule with kappa-bearing shells; H2O/STO-3G has no spinor shells). |
| Complexity | **High** — new σ·p module + `si` transform + spinor fixtures. Largest single group. |

### Group 5 — GIAO / magnetic-property NMR integrals

**Symbols (libcint `intor1.c`, `intor2.c`, `intor4.c`):** `int1e_giao_*`
(`igovlp`, `ignuc`, `igkin`, `giao_a11part`, `giao_irjxp`, `giao_sa10*`),
`int1e_cg_*` (`cg_irxp`, `cg_a11part`, `cg_sa10*`), `int1e_a01gp`, `int1e_ia01p`,
`int1e_ig*`, `int1e_g1/gg1`, `int1e_govlp/gnuc/ggovlp/ggnuc`, and the 2e
`int2e_g1`, `int2e_gg1`, `int2e_g1g2`, `int2e_ig1`, `int2e_gssp1ssp2`, the
`int2e_giao_sa10*` / `int2e_cg_*` block.

| Integration point | Detail |
|-------------------|--------|
| Env slots | GIAO = "gauge-including atomic orbital" → uses the gauge origin `PTR_COMMON_ORIG`. **Needs Gap A.** The `cg_` ("common gauge") family is explicitly the common-origin variant. |
| Kernel module | **Reuse `one_electron.rs` / `two_electron.rs`** with new angular-momentum-operator (`L = r × p`) and `ig` (imaginary gauge factor) contract arms. The `g`-prefix integrals carry an extra `r_gauge × ∇` factor; mathematically this is the position-operator tensor (Group 3) combined with the nabla step (Group 1). **Group 5 therefore depends on Groups 1 + 3.** Some members are complex-valued (`ig*`, `a01p`) → staging must use the complex-interleaved layout (`OutputLayoutMetadata.complex_interleaved`, already a field in `planner.rs:60`). |
| `component_rank` | `3` for the vector members (`ig`, `a01p`), `9` for `gg1`/tensor members. |
| Spinor path | The `giao_sa10sp` / `cg_sa10sp` spin members depend on **Gap B2**. Pure-orbital GIAO (`igovlp`, `ignuc`) are cart/sph (complex) — only need Gap A. |
| Oracle harness | `intor1.c`/`intor2.c` compiled; `intor4.c` (the 2e giao/cg block) needs adding. Add FFI + allowlist + suppl `extern`. **Needs the gauge-origin fixture** (shared with Group 3). Complex outputs: the parity comparator already handles interleaved re/im (the `compare.rs` chunks-of-2 complex check, `compare.rs:282-285`). |
| Complexity | **Medium–High** — depends on Gap A + Groups 1 & 3; complex-output plumbing. |

### Group 6 — Gauge / Breit–Gaunt 2e

**Symbols (libcint `breit.c`, `gaunt1.c`, `breit1.c`):** `int2e_gauge_r1_ssp1ssp2`,
`int2e_gauge_r1_sps1sps2`, `int2e_gauge_r1_ssp1sps2`, `int2e_gauge_r1_sps1ssp2`,
and the `_r2_` mirror set; plus the Gaunt block `int2e_ssp1ssp2`, `int2e_sps1sps2`,
`int2e_ssp1sps2`, `int2e_sps1ssp2`, `int2e_gssp1ssp2`. The existing `breit` family
(`int2e_breit_r1p2_spinor`, `int2e_breit_r2p2_spinor`) is already implemented
behind `unstable-source-api` (Phase 14) — gauge extends it.

| Integration point | Detail |
|-------------------|--------|
| Env slots | None new (these are 2e, full-Coulomb-kernel based). |
| Kernel module | **New module `kernels/gauge.rs`** (or extend the existing `unstable.rs::launch_breit`). Gauge `r1`/`r2` are `r12`-operator-weighted 2e integrals built on the `spsp`/`ssp` σ·p machinery from Group 4. **Hard dependency on Group 4** (σ·p + `si` transform). New dispatch arm `"gauge"`. |
| `component_rank` | Per symbol; the `ssp1ssp2` family is spin-tensor valued. |
| Spinor path | **Spinor-only** (the symbols carry `_spinor`). Full dependency on Gaps B1+B2 and Group 4. |
| Oracle harness | `breit.c` is ALREADY compiled (`build.rs:251`) with `int2e_breit_*` allowlisted. Add `autocode/gaunt1.c` + `autocode/breit1.c` to `cc::Build`; extend allowlist for `int2e_gauge_*` and the `ssp/sps` Gaunt symbols; suppl-header `extern` decls (NOT in `cint_funcs.h`). **Reuses the relativistic/spinor fixture from Group 4.** |
| Complexity | **High** — last in line; inherits all spinor + σ·p foundations. |

## Dependency-Ordered Build Sequence

```
        ┌────────────────────────────────────────────────────────────────┐
        │ FOUNDATIONS (own early phases, partly parallel)                  │
        │                                                                  │
        │  Gap A: PTR_COMMON_ORIG env slot ──────────┐  (low complexity)   │
        │  Gap B1: spinor-derivative transform ──┐   │                     │
        │  Gap B2: si transform + σ·p module ────┼─┐ │                     │
        └────────────────────────────────────────┼─┼─┼────────────────────┘
                                                  │ │ │
   Phase order (each "→" = unblocks):             │ │ │
                                                  │ │ │
   Group 1 (remaining 1st-deriv, cart/sph) ───────┼─┼─┼─► Group 2 (Hessian/higher)
        (reuses Phase-21 engine; no foundation)   │ │ │      (extends G1 engine)
                                                  │ │ │
   Group 3 (moments) ◄── needs Gap A ─────────────┘ │ │
        │  (origj variants); pure moments no dep      │ │
        ▼                                             │ │
   Group 5 (GIAO/NMR) ◄── needs Gap A + G1 + G3 ──────┘ │
        (gauge origin + L=r×p + complex output)         │
                                                        │
   Group 4 (relativistic spin-op) ◄── needs Gap B2 (+B1 for ip members)
        │  (new σ·p module; spinor-native)
        ▼
   Group 6 (gauge/Breit-Gaunt 2e) ◄── needs Group 4 + Gap B1/B2
        (spinor-only; last)

   Spinor variants of Groups 1,2,5 ◄── need Gap B1 (fold in after B1 lands)
```

**Recommended phase ordering (foundations first):**

1. **Gap A — `PTR_COMMON_ORIG` env slot** (small, isolated; unblocks the most groups; pure plumbing on the `rinv_orig` precedent).
2. **Group 1 — remaining 1st-derivatives (cart/sph)** (zero new foundations; pure Phase-21 reuse; highest value-per-effort; proves the higher-order recurrence engine for Group 2).
3. **Group 3 — position/multipole moments** (needs only Gap A; introduces the position-operator G-tensor that Group 5 reuses).
4. **Group 2 — Hessian / higher-order** (extends the Group-1 engine; adds `deriv3/4.c` to the oracle build; surfaces nroots-ceiling fail-closed cases).
5. **Group 5 — GIAO/NMR (orbital, complex)** (needs Gap A + Groups 1 & 3; introduces complex-output plumbing).
6. **Gap B1 — spinor-derivative transform** (unblocks the spinor variants of Groups 1/2/5 — fold those in as a follow-on; also a prerequisite for Group 4's `ip` members and Group 6).
7. **Gap B2 — `si` spin-included transform + σ·p (`spsp`) module** (the largest foundation).
8. **Group 4 — relativistic spin-operator** (needs Gap B2, partly B1; new `spin_operator.rs` module; needs the spinor/relativistic fixture).
9. **Group 6 — gauge / Breit–Gaunt 2e** (needs Group 4 + Gaps B1/B2; spinor-only; last).

**Why this order:** Gap A is cheap and unblocks two groups, so it leads. Groups 1/2/3
need no spinor work and deliver the bulk of the non-relativistic derivative + moment
surface early. The spinor foundations (B1, B2) are deferred until the cheaper cart/sph
work is banked, then Groups 4 and 6 (which are nearly 100% spinor-blocked) land last.
Group 5 sits in the middle because orbital GIAO is cart/sph+complex (cheap) but its
spin members tail behind B2.

## Anti-Patterns (specific to this milestone)

### Anti-Pattern 1: Treating spinor gradients/spin-operators as "just another arm"
**What people do:** Register `int1e_spsp_spinor` and write a contract arm, expecting the
existing `cart_to_spinor_sf_*` transform to apply.
**Why it's wrong:** `cart_to_spinor_sf` handles a single scalar G-tensor block; the
spin-operator families require the 4-block `si` transform (`cart2sph.c::c2s_si_1e`,
verified). Reusing the `sf` path silently produces wrong (non-libcint) spinor output
that the oracle gate will reject — after the kernel is already written.
**Do this instead:** Land Gap B2 (the `si` transform + σ·p G-tensor) as its own phase
BEFORE any spin-operator kernel. Keep spinor variants `UnsupportedApi` (the existing
R5/D-03 pattern, `one_electron.rs:901-908`) until the transform exists.

### Anti-Pattern 2: Inventing per-family env slots instead of the libcint global slots
**What people do:** Add a bespoke `multipole_center` or `giao_origin` field per family.
**Why it's wrong:** libcint reads ONE shared `PTR_COMMON_ORIG` (env[1..3]) for all
`origj`/GIAO/CG families; a per-family slot diverges from the raw `env[]` contract and
breaks raw-API compatibility.
**Do this instead:** Add the single `common_orig: Option<[f64;3]>` field (Gap A),
mirroring how `rinv_orig` maps to `PTR_RINV_ORIG`.

### Anti-Pattern 3: Skipping the new fixtures and validating only on H2O/STO-3G
**What people do:** Reuse `build_h2o_sto3g()` for every group.
**Why it's wrong:** H2O/STO-3G has no spinor shells (Group 4/6 cannot be exercised) and
uses the default zero gauge origin (Group 3 `origj` / Group 5 GIAO would pass trivially
even if the `PTR_COMMON_ORIG` read were broken).
**Do this instead:** Add two fixtures — a kappa-bearing relativistic fixture (Groups 4/6)
and a non-zero-gauge-origin fixture (Groups 3/5) — alongside the existing H2O/STO-3G and
Cu/LANL2DZ.

## Integration Points (concrete change inventory)

### New vs Modified Components

| Component | New or Modified | Driven by |
|-----------|-----------------|-----------|
| `OperatorEnvParams.common_orig` field | **New** | Gap A (groups 3, 5) |
| `raw.rs::eval_raw` env[1..3] read block | **New** (modeled on `raw.rs:599-616`) | Gap A |
| `transform/c2spinor.rs` spinor-derivative fns | **New** | Gap B1 (spinor ip across groups) |
| `transform/c2spinor.rs` `cart_to_spinor_si_*` | **New** | Gap B2 (groups 4, 5-spin, 6) |
| `kernels/spin_operator.rs` (σ·p) | **New module + dispatch arm** | Group 4 |
| `kernels/gauge.rs` (or extend `unstable.rs`) | **New module + dispatch arm** | Group 6 |
| `one_electron.rs` moment / GIAO / ipip contract arms | **Modified** | Groups 1, 2, 3, 5 |
| `two_electron.rs` ip2 / ipip / gauge contract arms | **Modified** | Groups 1, 2, 5, 6 |
| `center_2c2e.rs` / `center_3c1e.rs` / `center_3c2e.rs` deriv arms | **Modified** | Groups 1, 2 |
| `compiled_manifest.lock.json` entries (~140) | **New** | All groups |
| oracle `build.rs` `cc::Build` `.file()` adds: `autocode/intor4.c`, `deriv3.c`, `deriv4.c`, `gaunt1.c`, `breit1.c` | **Modified** | Groups 2, 4, 5, 6 |
| oracle `build.rs` bindgen allowlist regex (`build.rs:358`) | **Modified** | All groups |
| oracle `build.rs` suppl-header `extern` decls (`build.rs:265-345`) | **Modified** (symbols absent from `cint_funcs.h`) | All groups |
| `vendor_ffi.rs` FFI signatures + `compare.rs::raw_api_for_symbol` arms | **New entries** | All groups |
| `fixtures.rs` relativistic + gauge-origin fixtures | **New** | Groups 3, 4, 5, 6 |
| `tests/*_parity.rs` per-group byte-identity tests | **New** | All groups |

### Internal Boundaries (unchanged by v1.4)

| Boundary | Communication | Note |
|----------|---------------|------|
| manifest ↔ kernel | `canonical_family` string → `resolve_family_name` match | New families add match arms; existing arms unchanged |
| `eval_raw` ↔ planner | `OperatorEnvParams` struct | One new field (`common_orig`); additive, non-breaking |
| kernel ↔ transform | `cart_to_sph_*` / `cart_to_spinor_*` calls | New `si`/derivative transforms are additive |
| oracle ↔ vendor | `CINTIntegralFunction` C ABI | All new symbols share the identical 10-arg signature |

## Sources

- cintx source (HIGH — read directly): `crates/cintx-cubecl/src/kernels/{mod,one_electron,two_electron}.rs`, `crates/cintx-compat/src/raw.rs`, `crates/cintx-runtime/src/planner.rs`, `crates/cintx-cubecl/src/transform/{mod,c2spinor}.rs`, `crates/cintx-oracle/{build.rs,src/compare.rs,src/fixtures.rs}`, `crates/cintx-ops/generated/compiled_manifest.lock.json`.
- libcint 6.1.3 vendored source (HIGH — read directly): `libcint-master/src/cart2sph.c` (`c2s_sf_1e` vs `c2s_si_1e` at lines 4869/4947 — the spin-free vs spin-included split), `autocode/{grad1,hess,intor1,intor2,intor4,gaunt1,breit1,deriv3,deriv4,dkb,lresc}.c`, `src/breit.c`, `include/cint_funcs.h` (570 declared symbols).
- cintx planning (HIGH): `.planning/PROJECT.md` (v1.4 milestone scope, Phase-21 GRAD-01..10 outcomes), `.planning/REQUIREMENTS.md`, milestone context (R5/D-03 spinor-gradient gap, `PTR_RINV_ORIG`/`PTR_F12_ZETA` env-slot precedent).

---
*Architecture research for: libcint 6.1.3 full-family parity integration (v1.4)*
*Researched: 2026-05-27*
