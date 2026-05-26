# Research Summary — v1.4 Full libcint 6.1.3 Family Parity

**Project:** cintx
**Domain:** GTO integral engine — ~140 remaining libcint 6.1.3 integral families across 6 groups (remaining 1st-derivatives, Hessian/higher-order derivatives, position/multipole moments, relativistic spin-operator, GIAO/magnetic-property NMR, gauge/Breit–Gaunt 2e)
**Researched:** 2026-05-27
**Confidence:** HIGH (all findings read directly from vendored libcint 6.1.3 source and the live cintx tree; no training-data-only claims)

---

## Executive Summary

cintx v1.4 completes full libcint 6.1.3 API parity by adding the ~140 integral families that remain after Phase 21. These families decompose cleanly into 6 groups whose key research finding is **high reuse, low new math**: every group builds on exactly two operator primitives already present in the codebase (the nabla/`ip` operator and the position/`r` operator), and all new compute is plain host-side `Vec<f64>` contraction functions (`gout_*`), not new `#[cube]` device kernels. No new Rust crate dependencies are required. The four groups that do not touch the spinor-σ path (G1, G2, G3, and the spin-free half of G5) can be delivered entirely as extensions of the Phase-21 machinery, making them the recommended first-wave targets.

The dominant ordering constraint is architectural, not mathematical: two foundational gaps must be resolved before the relativistic half of the milestone can proceed. **Gap A** (the `PTR_COMMON_ORIG` gauge-origin env slot, env[1..3]) is currently documented in `raw.rs` but never read, blocking the `_origj` moment variants and every GIAO/CG family. **Gap B** is the missing spin-included (`c2s_si_*`) spinor transform, the single largest addition in v1.4: the Phase-12 scalar spinor transform (`cart_to_spinor_sf`) handles one G-tensor block, but σ-operator families require a 4-block input (`gc_x/gc_y/gc_z/gc_1`) via `c2s_si_1e`. Without Gap B2, Groups 4, 6, and the GIAO×σ slice of G5 cannot produce libcint-matching spinor output regardless of kernel correctness. Gap B decomposes into B1 (spinor-derivative transform, needed for `ip`-decorated spinor variants) and B2 (the `si` transform + σ·p module), which can run in parallel once the cheaper cart/sph work is banked.

The most dangerous correctness trap across all six groups is **silent-green failure**: GIAO/magnetic families are purely imaginary even in cart/sph representation (`c2s_zset0`), so staging them as real f64 yields a silently-zero result that passes magnitude-only sentinels; σ-operator families routed through the scalar `cart_to_spinor_sf` produce plausible but wrong spinor values; multipole moments validated only at zero gauge origin will always appear correct even without Gap A. All three failure modes produce green-looking artifacts at weak oracle gates. The mitigation is to treat complex-output capability, the `si` transform, and the non-zero gauge-origin fixture as hard prerequisites for the dependent groups — not post-hoc validation.

---

## Key Findings

### Recommended Stack

No new Rust crate dependencies are required for any of the 6 groups. The existing `cubecl 0.10.0` + host-side Boys/Rys/Obara–Saika stack plus `num-complex 0.4.6` (already in the workspace) covers all operator math. The decisive architectural fact is that all new derivative, position, and Pauli operators are **plain host-side functions**, not `#[cube]` device kernels — the new groups add zero new device math subject to CubeCL's control-flow constraints.

**Core technologies (all unchanged from v1.3):**
- `cubecl 0.10.0` (pinned): GPU compute backend — all 6 groups launch through the existing `#[cube(launch)]` kernel pattern; no new 0.10 features needed.
- Rust `1.94.0` (pinned, `rust-toolchain.toml`): reproducible compiler for oracle byte-identity.
- `num-complex 0.4.6` (already resolved): complex/imaginary output for GIAO (`ig*`), spinor relativistic, Breit/Gaunt; flows through the existing `OutputLayoutMetadata.complex_interleaved` plumbing.
- `thiserror 2.0.18` / `anyhow 1.0.102`: unchanged; new families add `UnsupportedApi` taxonomy entries only.
- vendored `libcint-master` (6.1.3): byte-identity oracle.

**What NOT to add:** No new `#[cube]` spin-operator device module; no new gauge/GIAO device math module; no cubecl version bump; no FFT/BLAS/special-function crate.

### Expected Features

**Table stakes — real, spin-free, reuse existing recurrences (P1 priority):**
- **G1: Remaining 1st-derivative families** — `int2e_ip2`, `int1e_ip{nuc,kin,ovlp}ip`, `int3c1e_ip1`/`iprinv`, `int2c2e_ip1`/`ip2`, `int3c2e_ip2`; component_rank=3. Closes analytical-gradient and DF-gradient sets.
- **G2: Hessian / higher-order derivatives** — `int1e_ipip{ovlp,nuc,kin,rinv}`, `int2c2e_ipip1`, `int3c2e_ipip1/ipip2`, 3rd/4th-order families, `int2e_ipip1/ipvip1/ip1ip2` promoted from `unstable`; component_rank=9/27/81. Analytical frequencies, IR/Raman, CPHF/CPKS.
- **G3: Position / multipole-moment integrals** — `int1e_r/rr/rrr/rrrr`, `r2/r4`, `z/zz`, `p4`, `rinv`, `drinv`, `irp` and `_origj` variants; component_rank=3/9/27/81/1. Dipole/quadrupole/multipole moments, ESP, EFG.

**Specialized differentiators — spinor-σ or complex-dependent (P2/P3 priority):**
- **G4: Relativistic spin-operator** — `spsp`, `spnucsp`, `sprinvsp`, `srsr`, `sigma`, `int2e_spsp1`/`srsr1`/`ssp*`/`sps*`/`vsp*`/`spv*`; spinor-native. **Hard prerequisite: Gap B2.**
- **G5: GIAO / NMR** — `int1e_giao_*`, `int1e_cg_*`, `a01gp`, `ia01p`, `ig*`, `govlp/gnuc`, `int2e_g*`; **purely imaginary even in cart/sph** (`c2s_zset0`). Spin-free GIAO: Gap A only. GIAO×σ slice: Gap B2 also.
- **G6: Gauge / Breit–Gaunt 2e** — `int2e_gauge_r1/r2_{ssp,sps}{ssp,sps}` (8 symbols) + Gaunt `ssp/sps` (4 symbols); spinor-only. Full Dirac–Coulomb–Breit. **Requires Gap B2 + Group 4.**

**Defer to post-v1.4:** spinor variants of G1/G2/G5 (until Gap B1); magnetic-Gaunt GIAO×Gaunt×σ; high-l (nroots>5) quartets if Wheeler fallback is deferred.

**Anti-features to resist:** treating cart/sph σ-variants as first-class deliverables; treating GIAO integrals as real; skipping `_origj` and adjacent same-recurrence families.

### Architecture Approach

The existing 6-stage pipeline (manifest → kernel → env/plan → raw+legacy+capi → safe API → oracle) is unchanged; all new families thread through it verbatim. Every v1.4 addition is one of: a new gout contraction arm inside an existing kernel module, a new env-slot read block (Gap A), a new transform in `c2spinor.rs` (Gap B), a new kernel module for σ·p (G4) or gauge 2e (G6), new oracle build wiring, or new oracle fixtures.

**Major components modified:**
1. `OperatorEnvParams` + `raw.rs::eval_raw` — new `common_orig: Option<[f64;3]>` field + env[1..3] read block (Gap A).
2. `transform/c2spinor.rs` — Gap B1 (spinor-derivative per-component transform) + Gap B2 (4-block `cart_to_spinor_si_*`).
3. `kernels/spin_operator.rs` (new, G4) — σ·p 4-block G-tensor assembler + 12-component Pauli gout emitter.
4. `kernels/gauge.rs` or extension of `unstable.rs` (G6) — per-block `gout_gauge_r{1,2}_*`/`gout_gaunt_*`.
5. Existing kernel modules (`one_electron.rs`, `two_electron.rs`, center modules) — new gout arms for G1–G3 and spin-free G5.
6. `compiled_manifest.lock.json` — ~140 new entries.
7. Oracle `build.rs` — add `intor4.c`, `deriv3.c`, `deriv4.c`, `gaunt1.c`, `breit1.c` to `cc::Build`; extend allowlist; add suppl-header `extern` decls.
8. `oracle/src/fixtures.rs` — two new fixtures: kappa-bearing relativistic fixture (G4/G6) and non-zero gauge-origin fixture (G3/G5).

### Critical Pitfalls

The four most dangerous pitfalls all produce silently green artifacts:

1. **Imaginary GIAO/magnetic output staged as real f64** — GIAO cart/sph is imaginary (`c2s_zset0`); staging as real yields silent zero. Set `complex_interleaved=true` per-family from driver routing (not from representation string); widen `assert_flat_buffer_contract` to fire on the flag; size staging `2×ncomp×…`. Must be a complex-output capability phase before G5.

2. **Routing σ-operator families through `cart_to_spinor_sf` instead of the missing 4-block `c2s_si` transform** — `c2s_si_1e` (verified `cart2sph.c:4947`) consumes `gc_x/gc_y/gc_z/gc_1`; `cart_to_spinor_sf` handles one scalar block. Reusing `sf` on σ families produces non-libcint output. Land Gap B2 before any σ-operator kernel; keep σ families `UnsupportedApi` until Gap B2 passes on a kappa-bearing fixture.

3. **Not reading `PTR_COMMON_ORIG` and validating only at zero gauge origin** — even plain `int1e_r` reads `env[PTR_COMMON_ORIG+k]` (verified: `drj[k] = rj[k] - env[PTR_COMMON_ORIG+k]` in intor1.c). H2O/STO-3G uses zero origin, making this invisible. Land Gap A first; add a non-zero gauge-origin fixture gating both G3 and G5 parity.

4. **Silent partial-write scatter guards (`if dst < staging.len()`)** — WR-03 pattern from Phase-21 silently drops components when staging is under-sized. At rank 3 this was low-risk; at rank 9/27/81 a `component_rank` mismatch produces a quietly truncated tensor. Replace per-element guard with upfront size assertion + unconditional indexing before G2.

5. **G-tensor angular-momentum headroom sized from the wrong index** — moments raise the ket (`ng[1]`), not the bra; `int1e_r` is `{0,1,0,0,…}`. 4th-order derivatives raise both bra and ket (`{2,2,0,0,…}`). Drive sizing from the per-family `ng[]` headroom tuple, not a single "order" scalar.

---

## Implications for Roadmap

Suggested 10-phase structure:

### Phase 1 (22): Gap A — `PTR_COMMON_ORIG` Gauge-Origin Env Slot
**Rationale:** Cheap and isolated; unblocks two groups (G3 `_origj` variants and all of G5 GIAO/CG). Pure additive plumbing modeled verbatim on the Phase-21 `PTR_RINV_ORIG` block (`raw.rs:599-616`). Leads the milestone because cost is minimal and unblock is maximal.
**Delivers:** `common_orig: Option<[f64;3]>` on `OperatorEnvParams`; env[1..4] read block in `eval_raw`; `with_common_origin` safe-API setter; validator gate; non-zero gauge-origin oracle fixture.
**Avoids:** P3 (unread `PTR_COMMON_ORIG`), P4 (zero-origin-only validation).

### Phase 2 (23): Group 1 — Remaining 1st-Derivative Families (cart/sph)
**Rationale:** Zero new foundations; pure Phase-21 reuse; highest value-per-effort; proves the multi-center nabla engine Group 2 extends.
**Delivers:** `int2e_ip2`, `int1e_ip{nuc,kin,ovlp}ip`, `int3c1e_ip1`/`iprinv`, `int2c2e_ip1`/`ip2`, `int3c2e_ip2` — all `oracle_covered=true`, cart/sph, component_rank=3.

### Phase 3 (24): Group 3 — Position / Multipole-Moment Integrals
**Rationale:** Needs only Gap A (Phase 1); can run in parallel with Phase 2. Introduces the 1e position-operator helper (`x1i_1e`/`x1j_1e`) that G5 reuses.
**Delivers:** Full moment family set at oracle parity. `_origj` variants gate on Gap A. Non-zero gauge-origin fixture (from Phase 1) is the parity gate.
**Avoids:** P3 (`PTR_COMMON_ORIG` plumbed for all moments, not just `_origj`), P5 (ket-side headroom `ng[1]=1` for `int1e_r`), P7 (`rr/rrr` component order from gout index map).

### Phase 4 (25): Group 2 — Hessian / Higher-Order Derivatives
**Rationale:** Extends the Phase-2 engine to 2nd/3rd/4th order. WR-03 scatter guard cleanup must lead this phase. Surfaces the nroots>5 ceiling decision.
**Delivers:** `int1e_ipip*`, `int2c2e_ipip1`, `int3c2e_ipip1/ipip2`, `int2e_ipip1/ipvip1/ip1ip2` (promoted from `unstable`), 3rd/4th-order families. component_rank=9/27/81. Adds `deriv3.c`/`deriv4.c` to `cc::Build`.
**Avoids:** P5 (multi-center deriv4 `{2,2,…}` headroom), P6 (nroots>5 — decide Wheeler fallback or scoped envelope), P7 (×9/27/81 ordering), P8 (scatter guard cleanup first), P13 (unstable→stable promotion routing).
**Research flag:** Wheeler `nroots>=6` fallback decision is milestone-level; must be resolved in this phase's REQ.

### Phase 5 (26): Group 5 (spin-free) — GIAO / NMR Integrals (orbital, complex)
**Rationale:** Depends on Phases 1+3; introduces complex/imaginary-output capability. Spin-free GIAO needs no σ path. Complex-output capability phase leads (set `complex_interleaved`, widen contract, size `2×`).
**Delivers:** All spin-free GIAO/CG families at oracle parity (cart/sph, complex-valued). Adds `intor4.c` to `cc::Build`.
**Avoids:** P1 (complex-output gate), P3 (non-zero gauge-origin fixture from Phase 1), P9 (ALL_CINT1E vs ALL_CINT; complex buffer sizing).

### Phase 6 (27): Gap B1 — Spinor-Derivative Transform
**Rationale:** Unlocks spinor variants of G1/G2/G5 (fold in as follow-ons); co-prerequisite for G4 `ip`-decorated members and G6. Independent of Gap B2; can parallel Phase 5.
**Delivers:** `cart_to_spinor_sf_derivative_*` in `c2spinor.rs`; `int1e_ipovlp_spinor` and sibling ip spinor families from `UnsupportedApi` to `oracle_covered=true`.
**Research flag:** Per-component axis fold design not yet exercised — design spike recommended before REQ.

### Phase 7 (28): Gap B2 — Spin-Included (`c2s_si`) Transform + σ·p Module
**Rationale:** Single largest architectural addition in v1.4. Must precede G4, G6, and GIAO×σ. `c2s_si_1e` (verified `cart2sph.c:4947`) takes 4 blocks — fundamentally different from `cart_to_spinor_sf`. The companion σ·p G-tensor assembler produces the four blocks.
**Delivers:** `cart_to_spinor_si_*` in `c2spinor.rs`; σ·p G-tensor assembler; 12-component Pauli gout emitter; kappa-bearing relativistic oracle fixture. All σ families remain `UnsupportedApi` until this phase passes on the kappa fixture.
**Avoids:** P2 (this IS the gap-close), P4 (relativistic fixture added here), P7 (4-block input contract asserted at transform boundary).
**Research flag:** Design spike on `a_bra_cart2spinor_si` 4-block layout (`cart2sph.c:4947-4992`) recommended before REQ.

### Phase 8 (29): Group 4 — Relativistic Spin-Operator Integrals
**Rationale:** All families gate on Gap B2 (Phase 7). Delivers Dirac/X2C/DKH and SOC integrals; no Rust library currently provides these. New dispatch arm `"spin"` in `resolve_family_name`. Adds `intor4.c` to `cc::Build` (for 2e spin families).
**Delivers:** Full G4 family set at spinor oracle parity. `int1e_sp` (σ-coupled, scoped here from G3) included.
**Avoids:** P2 (routes through `cart_to_spinor_si_*`, not `sf`), P4 (kappa fixture from Phase 7), P7 (σ 12-component Pauli pattern), P14 (oracle expectations gated to spinor, not cart/sph raw intermediate).

### Phase 9 (30): Group 5 (GIAO×σ slice) — Spin-GIAO Integrals
**Rationale:** `spg*`, `spgnucsp`, `*_sa10*` require both Gap A (Phase 1) and Gap B2 (Phase 7). Scheduled after G4 because the σ·p pattern from there is reused directly.
**Delivers:** GIAO×σ family set at spinor oracle parity. Completes NMR suite including relativistic corrections.
**Avoids:** P1 (complex imaginary output, same as spin-free GIAO), P2 (σ path from Gap B2).

### Phase 10 (31): Group 6 — Gauge / Breit–Gaunt 2e (apex)
**Rationale:** Full dependency stack: σ·p (Gap B2), spinor-derivative (Gap B1), Group-4 Gaunt-style σ machinery, existing `launch_breit`/`BreitShape` (Phase 14). The 8 `gauge_r1/r2_*` symbols and 4 Gaunt `ssp/sps` symbols are per-block decompositions of the existing aggregate Breit driver. Spinor-only. Adds `gaunt1.c`/`breit1.c` to `cc::Build`.
**Delivers:** Full Dirac–Coulomb–Breit 2e integral set.
**Avoids:** P2 (`c2s_si_2e1i`/`c2s_si_2e2i` verified in `breit1.c:211`), P9 (complex `double complex *out` buffer), P10 (Gaunt/gauge symbols absent from `cint_funcs.h` — suppl extern + allowlist).

### Phase Ordering Rationale

Three dependency chains drive the sequence:
1. **Gap A first** — unlocks two groups at near-zero cost; the gauge-origin fixture it creates is also the correctness gate for both G3 and G5.
2. **Real cart/sph work before spinor foundations** — G1, G2, G3, and spin-free G5 deliver the bulk of the non-relativistic derivative and property surface before the expensive spinor foundations are needed.
3. **σ foundations before σ families** — Gap B2 must be validated against a kappa-bearing fixture before any G4/G6 family is registered as `oracle_covered`; eager registration without the σ path produces the P2 silent-wrong-transform trap.

### Research Flags
- **Phase 4 (Group 2):** Wheeler `nroots>=6` fallback scope decision — `.planning/todos/pending/rys-nroots-ge6-wheeler-fallback.md` must be resolved before Phase 4 REQ is finalized.
- **Phase 6 (Gap B1):** Spinor-derivative per-component axis fold — one-day design spike against `int1e_ipovlp_spinor` recommended.
- **Phase 7 (Gap B2):** `a_bra_cart2spinor_si` 4-block layout — confirm stride/ordering from `cart2sph.c:4947-4992` before REQ.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Read directly from vendored libcint 6.1.3 and live cintx tree; no new crates needed confirmed across all 6 groups. |
| Features | HIGH | Family enumeration, operator strings, `component_rank`, representation routing all read from `cint_funcs.h` + `ng[]` + `c2s_*` routing. 187 families extracted; all 6 groups' headline families confirmed missing vs the live manifest. |
| Architecture | HIGH (pipeline/env/transform gaps) / MEDIUM (per-family recurrence reuse boundary, fixture sizing) | Two foundational gaps confirmed from source; recurrence reuse extrapolated from Phase-21, not yet executed. |
| Pitfalls | HIGH | All 4 critical pitfall mechanisms grounded in libcint source + the live cintx tree; not inferred. |

**Overall confidence: HIGH**

### Gaps to Address
- Adjacent/long-tail family enumeration (`_origj`, full `ipip` 2e/3c set, Gaunt permutations, GIAO×σ) — final pass against manifest-audit before the full-parity gate; handle during per-phase REQ authoring.
- Wheeler `nroots>=6` fallback scope — milestone-level decision before Phase 4; gates high-l quartets in G2/G4/G6.
- Spinor-derivative per-component layout (Gap B1) — design spike before Phase 6.
- 4-block `gc_x/y/z/1` assembler layout (Gap B2) — confirm vs `cart2sph.c:4947-4992` before Phase 7.
- High-rank (27/81) OOM-safety — re-derive chunk-planner limits from `component_rank` before Group 2.
- `CINTshells_cart_offset` 8-vs-0 — pre-existing LIB-gated vendor-test discrepancy; triage separately under both flags.

---

*Research synthesized from: STACK-v1.4.md, FEATURES-v1.4.md, ARCHITECTURE-v1.4.md, PITFALLS-v1.4.md*
*Research completed: 2026-05-27*
*Ready for roadmap: yes*
