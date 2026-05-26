# Feature Research

**Domain:** libcint 6.1.3 integral-family parity (quantum-chemistry GTO integral engine, Rust/CubeCL)
**Milestone:** v1.4 — Full libcint 6.1.3 Family Parity (~140 remaining families, 6 groups)
**Researched:** 2026-05-27
**Confidence:** HIGH — every family's operator, component count, and representation routing was read directly from the vendored libcint 6.1.3 source (`libcint-master/include/cint_funcs.h`, `src/autocode/*.c`, `src/intor*.c`, `src/int3c*.c`), not inferred. Quantum-chemistry use cases are MEDIUM-HIGH (standard textbook/PySCF knowledge cross-checked against the operator comments and gout code).

---

## How to read this document

This milestone (v1.4) adds ~140 integral families to reach full libcint 6.1.3 API parity, organized into 6 groups. For each family the load-bearing facts are:

- **Operator** — the integrand (the bra/ket operator string from the source comment, decoded into physics).
- **component_rank** — number of output tensor components per (i,j[,k,l]) block. This is the last element of libcint's `ng[]` array (verified per-family from source). The existing cintx manifest already uses this exact convention as a string field: `scalar`/`1` = no derivative; `3` = one ∇ (gradient); `9` = two ∇ or ∇⊗∇ (Hessian); `27` = three ∇; `81` = four ∇. (Confirmed in `compiled_manifest.lock.json`: e.g. `int2e_ip1` → `"3"`, `int2e_stg_ipip1`/`ipvip1`/`ip1ip2` → `"9"`, `int1e_grids_ipip` → `2`.)
- **Representations** — which of cart / sph / spinor are *physically meaningful*. libcint's header declares all three for every family, but for the σ-coupled relativistic families the cart/sph variants emit only the raw spin-free intermediate; the physics lives in the spinor σ-coupling (`c2s_si_*`). GIAO/magnetic families are **purely imaginary** (driver prefixes `c2s_zset0`, zeroing the real part) — complex-valued even in cart/sph.
- **Spinor dependency** — `c2s_sf_*` = spinor-free (spin-block expansion only); `c2s_si_*` = Pauli-σ coupling required (the R5/D-03 spinor path is a hard prerequisite). This is the single biggest complexity and ordering driver in this milestone.

### component_rank ground truth (last element of libcint `ng[]`, read from source)

| Pattern | Tensor factor | Example families (verified from `ng[]`) |
|---------|---------------|------------------------------------------|
| scalar operator | 1 | `int1e_r2`, `int1e_rinv`, `int1e_z`, `int1e_p4`, `int1e_pnucp`, `int1e_spsp`*, `int1e_sr`*, `int2e_spsp1`*, all `int2e_gauge_*`* |
| one ∇ / one r / one σ-vector / B-direction | 3 | `int1e_r`, `int1e_ipovlp`, `int2e_ip1/ip2`, `int3c1e_ip1`, `int2c2e_ip1/ip2`, `int3c2e_ip2`, `int1e_sigma`*, `int1e_govlp`*, `int1e_ia01p`* |
| two ∇ / r⊗r / 3×3 tensor | 9 | `int1e_ipipovlp`, `int1e_ipnucip`, `int1e_rr`, `int1e_irp`, `int2c2e_ipip1`, `int3c2e_ipip1/ipip2`, `int2e_ipip1/ip1ip2`, `int1e_a01gp`* |
| three ∇ / r⊗r⊗r | 27 | `int1e_rrr`, `int1e_irrp`, `int1e_ipipipnuc`, `int1e_ipipiprinv`, `int1e_iprip` |
| four ∇ / r⊗r⊗r⊗r | 81 | `int1e_rrrr`, `int1e_ipipipiprinv`, `int1e_ipipipiprinvip`, `int2e_ipip1ipip2` |

\* For σ-coupled families the printed `ng` tensor is the **spin-free** intermediate count; the spinor output additionally carries 2×2 Pauli block structure (e.g. `int1e_sigma` gout writes `n*12` reals = 3 σ-directions × 4 spin-block components; `int1e_sp` writes `n*4`). For magnetic families with tensor=3 the 3 is the magnetic-field direction (B_x,B_y,B_z).

All listed target families were confirmed **MISSING** in the current manifest (i.e. genuinely to-build), and the spin-free ip-gradient pattern (`int2e_ip1`, `int1e_ipovlp`, `int3c2e_ip1`) plus the deferred spinor-gradient pattern (`int1e_ipovlp_spinor` registered with `component_rank:"3"`, `oracle_covered:false`) were confirmed present from Phase 21.

---

## Feature Landscape — per-group family inventory

### Group 1 — Remaining first-derivative families (table stakes)

**Physics:** A single nabla applied to bra or ket (`<∇i|O|j>`, `<i|O|∇j>`) or to a 2e/3c index. These complete the analytical-gradient set whose first half (`int2e_ip1`, `int1e_ip{ovlp,kin,nuc,rinv}`, `int3c2e_ip1`) shipped in Phase 21. They reuse the same `gout_ip1` nabla machinery; only the operator envelope and which index carries ∇ changes. **All cart/sph; spinor is `c2s_sf_*` (spin-free) — registerable but deferred under R5/D-03.**

| Family | Operator (decoded) | component_rank | Reps (meaningful) | Use case |
|--------|--------------------|----------------|-------------------|----------|
| `int2e_ip2` | (ij\|r12\|∇k l) — ∇ on 3rd ERI index | 3 | cart/sph; spinor sf | RHS of 2e gradient / DF-gradient ket side |
| `int1e_ipnucip` | <∇i\|NUC\|∇j> — ∇ on both 1e indices | 9 | cart/sph; spinor sf | Hellmann–Feynman / mixed-derivative nuclear term |
| `int1e_ipkinip` | <∇i\|p·p\|∇j> | 9 | cart/sph; spinor sf | Kinetic mixed second-derivative term |
| `int1e_ipovlpip` | <∇i\|1\|∇j> | 9 | cart/sph; spinor sf | Overlap mixed second-derivative (CPHF/CPKS) |
| `int3c1e_ip1` | <∇i\|j\|k> 3-center 1e | 3 | cart/sph; spinor sf | 3c-1e gradient (DF/embedding) |
| `int3c1e_iprinv` | <∇i 1/r_C\|j\|k> 3-center | 3 | cart/sph; spinor sf | 3c rinv gradient |
| `int2c2e_ip1` | (∇i\|r12\|j) 2-center | 3 | cart/sph; spinor sf | DF metric gradient (bra) |
| `int2c2e_ip2` | (i\|r12\|∇j) 2-center | 3 | cart/sph; spinor sf | DF metric gradient (ket) |
| `int3c2e_ip2` | (ij\|r12\|∇k) 3-center | 3 | cart/sph; spinor sf | DF 3c2e gradient (aux side) — indispensable partner of Phase-21 `int3c2e_ip1` |

Note: `int1e_ipnucip`/`ipkinip`/`ipovlpip` are tensor=9 (two nablas) so they straddle Groups 1 and 2 mathematically; PROJECT scopes them into Group 1 because they reuse the ip1-on-both-sides pattern rather than the ∇∇-on-one-side Hessian recurrence.

---

### Group 2 — Hessian & higher-order derivatives (table stakes for frequencies/CPHF)

**Physics:** Two-or-more nablas on a single center (`<∇∇i|O|j>`) or split (`<∇i|O|∇j>`, `(∇∇ij|kl)`, `(∇i∇j|kl)`, `(∇ij|∇kl)`). These give analytical Hessians (harmonic frequencies, IR/Raman), and the 4th-order `rinv` term feeds higher relativistic/property derivatives. Extends the Phase-21 nabla recurrence to 2nd/3rd/4th order. **cart/sph meaningful; spinor `c2s_sf_*` (deferred).**

| Family | Operator | component_rank | Reps | Use case |
|--------|----------|----------------|------|----------|
| `int1e_ipipovlp` | <∇∇i\|1\|j> | 9 | cart/sph; spinor sf | Overlap Hessian |
| `int1e_ipipnuc` | <∇∇i\|NUC\|j> | 9 | cart/sph; spinor sf | Nuclear-attraction Hessian |
| `int1e_ipipkin` | <∇∇i\|p·p\|j> | 9 | cart/sph; spinor sf | Kinetic Hessian |
| `int1e_ipiprinv` | <∇∇i\|1/r_C\|j> | 9 | cart/sph; spinor sf | rinv Hessian / NMR-adjacent |
| `int2c2e_ipip1` | (∇∇i\|r12\|j) | 9 | cart/sph; spinor sf | DF metric Hessian |
| `int3c2e_ipip1` | (∇∇ij\|r12\|k) | 9 | cart/sph; spinor sf | DF 3c2e Hessian (orbital side) |
| `int3c2e_ipip2` | (ij\|r12\|∇∇k) | 9 | cart/sph; spinor sf | DF 3c2e Hessian (aux side) |
| `int1e_ipipipiprinv` | <∇∇∇∇i\|1/r_C\|j> | 81 | cart/sph; spinor sf | 4th-order rinv (high-order property/relativistic derivative) |

**Adjacent families libcint also exposes (fold into the same phase to satisfy the full-parity gate):** `int1e_ipipnucip`/`ipiprinvip` (27), `int1e_ipipipnuc`/`ipipiprinv` (27), `int1e_ipipipiprinvip` (81), `int2e_ipip1`/`ipvip1`/`ip1ip2` (9), `int2e_ipip1ipip2`/`ipvip1ipvip2` (81), `int3c2e_ipvip1`/`ip1ip2` (9), `int2c2e_ip1ip2` (9). They share the exact recurrence. Note `int2e_ipip1`/`int2e_ipvip1` are currently registered only as `unstable::source::2e` — promoting them to stable 2e Hessian families is part of parity.

---

### Group 3 — Position / multipole-moment integrals (table stakes for properties)

**Physics:** Powers of the position operator r (relative to a common gauge origin `PTR_COMMON_ORIG`; the `_origi`/`_origj` variants pin it to atom i/j). `int1e_r` = dipole; `rr` = quadrupole (Cartesian 9-comp); `rrr` = octupole; `rrrr` = hexadecapole; `r2`/`r4` = radial moments; `z`/`zz` = single-axis moments; `sp`/`p4` = momentum operators; `rinv`/`drinv` = plain and gradient nuclear-potential-at-origin; `irp` = mixed r⊗∇. Built on Obara–Saika with the position-operator (RCJ/RCI) recurrence already partially present (`gout_int1e_r` uses `G1E_RCJ`). **cart/sph fully meaningful and real (complex-free); spinor is spin-free expansion — except `sp` which is σ-coupled.**

| Family | Operator | component_rank | Reps | Use case |
|--------|----------|----------------|------|----------|
| `int1e_r` | <i\|r_C\|j> dipole | 3 | cart/sph/spinor | Dipole moment, polarizability, transition dipoles |
| `int1e_rr` | <i\|r_C r_C\|j> quadrupole (xx,xy,…) | 9 | cart/sph/spinor | Quadrupole moment, second moments |
| `int1e_rrr` | <i\|r r r\|j> octupole | 27 | cart/sph/spinor | Octupole, first hyperpolarizability |
| `int1e_rrrr` | <i\|r r r r\|j> hexadecapole | 81 | cart/sph/spinor | Hexadecapole, high multipoles |
| `int1e_r2` | <i\|r·r\|j> radial 2nd moment | 1 | cart/sph/spinor | <r²> (diamagnetic susceptibility, spatial extent) |
| `int1e_r4` | <i\|(r·r)²\|j> radial 4th moment | 1 | cart/sph/spinor | <r⁴> (relativistic/MVD-adjacent corrections) |
| `int1e_z` | <i\|z_C\|j> single-axis dipole | 1 | cart/sph/spinor | z-dipole (field-along-axis, Stark) |
| `int1e_zz` | <i\|z_C z_C\|j> | 1 | cart/sph/spinor | z² moment |
| `int1e_sp` | <σ·p i\|1\|j> | 1 (sf intermediate; gout n*4) | **spinor (σ)**; cart/sph emit raw | Kinetic-balance / small-component coupling |
| `int1e_p4` | <p·p i\|1\|p·p j> | 1 | cart/sph/spinor | Mass-velocity correction (scalar-relativistic p⁴) |
| `int1e_rinv` | <i\|1/r_C\|j> (origin = PTR_RINV_ORIG) | 1 | cart/sph/spinor | Electrostatic potential at a point; EFG base |
| `int1e_drinv` | <i\|∇(1/r_C)\|j> | 3 | cart/sph/spinor | Electric field at a point / EFG / nuclear gradient of rinv |
| `int1e_irp` | <i\|r_C ∇\|j> mixed r⊗∇ | 9 | cart/sph/spinor | Magnetizability / mixed position-momentum property |

**Adjacent same-recurrence families to fold in:** `int1e_r_origj`/`rr_origj`/`r2_origj`/`r4_origj`/`zz_origj` (gauge-origin-on-j variants, tensor 3/9/1/1/1), `int1e_irrp`/`irpr` (27), `int1e_prinvp` (1), `int1e_prinvxp`/`pnucxp` (3). Same position/momentum-operator G-tensor with different origins/orderings. (`int1e_sp` is σ-coupled — schedule it with Group 4, not here.)

---

### Group 4 — Relativistic spin-operator integrals (specialized differentiator — **spinor-σ dependent**)

**Physics:** σ·p and σ·r operators (Pauli matrices contracted with momentum/position). These build the small-component and spin-dependent blocks of Dirac/X2C/DKH relativistic Hamiltonians and spin–orbit coupling (SOC). `spsp` = (σ·p)(σ·p) = p² + iσ·(p×p); `spnucsp`/`sprinvsp` = picture-change / SOC nuclear/rinv kernels; `srsr` = (σ·r)(σ·r); `sigma` = bare σ (spin density). The 2e `spsp1`/`srsr1`/`ssp*`/`sps*`/`vsp*`/`spv*` families are the small-component ERIs and Gaunt small/large mixings. **The cart/sph variants are NOT the physical answer — the driver routes through `c2s_si_*`, applying the 2×2 Pauli coupling. Correct implementation REQUIRES the spinor/4-component path (R5/D-03), which Phase 12 (scalar spinor transform) partially unblocks but which the deferred spinor-gradient decision shows is not yet exercised for derivative/σ families.**

| Family | Operator | component_rank | Reps (meaningful) | Spinor coupling | Use case |
|--------|----------|----------------|-------------------|-----------------|----------|
| `int1e_spsp` | <σ·p i\|1\|σ·p j> | 1 (sf int.) | **spinor** (`c2s_sf_1e`, kinetic-balance) | Small-component overlap / relativistic kinetic balance |
| `int1e_spnucsp` | <σ·p i\|NUC\|σ·p j> | 1 | **spinor** (`c2s_si_1e`, flag 2) | DKB/X2C nuclear small-component, SOC |
| `int1e_sprinvsp` | <σ·p i\|1/r\|σ·p j> | 1 | **spinor** (`c2s_si_1e`, flag 1) | SOC mean-field / picture-change |
| `int1e_srsr` | <σ·r i\|1\|σ·r j> | 1 | **spinor** (`c2s_si_1e`) | RKB / restricted-kinetic-balance |
| `int1e_sigma` | <i\|σ\|j> bare Pauli | 3 (σ direction) | **spinor** (`c2s_si_1ei`) | Spin density, magnetization |
| `int2e_spsp1` | (σ·p i σ·p j\|r12\|kl) | 1 | **spinor** (`c2s_si_2e1`,`c2s_sf_2e2`) | Dirac–Coulomb (SS\|LL) |
| `int2e_spsp1spsp2` | (σ·p i σ·p j\|r12\|σ·p k σ·p l) | 1 | **spinor** (si/si) | Dirac–Coulomb (SS\|SS) |
| `int2e_srsr1` | (σ·r i σ·r j\|r12\|kl) | 1 | **spinor** | RKB 2e |
| `int2e_ssp1ssp2` | (i σ·p j\|GAUNT\|k σ·p l) | 1 | **spinor** (`c2s_si_2e1i`,`c2s_si_2e2i`) | Gaunt/Breit small-component 2e |
| `int2e_ssp1sps2`,`sps1ssp2`,`sps1sps2` | Gaunt σ·p index permutations | 1 | **spinor** | Gaunt interaction permutations |
| `int2e_spv1`,`vsp1`,`spv1spv2`,`vsp1spv2`,… | one-sided σ·p large/small mixings | 1 | **spinor** (dkb.c) | Dirac–Kohn–Beck mixed small-large ERIs |

Adjacent: `int1e_spsigmasp` (3), `int1e_srsp`/`sprsp` (1/3), `int1e_srnucsr` (1), `int1e_spnuc`/`spspsp` (1, dkb.c), `int2e_spsp2`/`spv1spsp2`/`vsp1spsp2`/`vsp1vsp2` (1, dkb.c).

---

### Group 5 — GIAO / magnetic-property NMR integrals (specialized differentiator — **complex, imaginary-valued**)

**Physics:** Gauge-including atomic orbitals (London orbitals) carry a magnetic-field-dependent phase. Derivatives of the integrand w.r.t. external field B (the `g`/`ig` operators, G = gauge factor) and angular-momentum operators (`a01`, `irxp`/`irjxp` = r×p) produce the building blocks of NMR shielding tensors, magnetizability, and rotational g-tensors. **These families are purely imaginary**: the spinor drivers prefix `c2s_zset0` (zero the real part) — the physical content is the imaginary part of a complex integral, present even in cart/sph. component_rank=3 generally encodes the B-field direction; =9 encodes B⊗(operator).

| Family | Operator | component_rank | Reps | Imaginary? | Use case |
|--------|----------|----------------|------|------------|----------|
| `int1e_govlp` | <G i\|1\|j> GIAO overlap | 3 | cart/sph/spinor | yes (`c2s_sf_1ei`) | NMR/magnetizability overlap derivative |
| `int1e_gnuc` | <G i\|NUC\|j> | 3 | cart/sph/spinor | yes | GIAO nuclear-attraction derivative |
| `int1e_igovlp` | <∇_G i\|1\|j> imaginary-G overlap | 3 | cart/sph/spinor | yes (`c2s_sf_1e`+zset0) | London-orbital overlap B-derivative |
| `int1e_ignuc` | <∇_G i\|NUC\|j> | 3 | cart/sph/spinor | yes | London nuclear B-derivative |
| `int1e_igkin` | <∇_G i\|p·p\|j> | 3 | cart/sph/spinor | yes | London kinetic B-derivative |
| `int1e_a01gp` | <G i\|∇(1/r)×p\|j> | 9 | cart/sph/spinor | yes | NMR shielding paramagnetic term (nucleus × field) |
| `int1e_ia01p` | <i\|∇(1/r)×p\|j> (PSO) | 3 | cart/sph/spinor | yes (`c2s_sf_1e`, flag 1) | Paramagnetic spin-orbit / NMR shielding |
| `int1e_ggovlp` | <i\|G G\|j> 2nd-order GIAO overlap | 9 | cart/sph/spinor | (`c2s_sf_1e`) | Magnetizability (B⊗B) overlap term |
| `int2e_g1` | (G i j\|r12\|kl) | 3 | cart/sph/spinor | yes (`c2s_sf_2e1i`) | GIAO 2e first B-derivative (NMR/magnetizability) |
| `int2e_ig1` | (∇_G i j\|r12\|kl) | 3 | cart/sph/spinor | yes | London 2e B-derivative |
| `int2e_gg1` | (G G i j\|r12\|kl) | 9 | cart/sph/spinor | yes | 2nd-order GIAO 2e |
| `int2e_g1g2` | (G i j\|r12\|G k l) | 9 | cart/sph/spinor | yes (both `_2ei`) | Cross-gauge 2e (magnetizability) |

**Adjacent same-machinery families to fold in:** `int1e_cg_irxp` (3, common-gauge r×p), `int1e_giao_irjxp` (3), `int1e_ggkin`/`ggnuc` (9), `int1e_giao_a11part`/`cg_a11part` (9), `int1e_grjxp` (9), `int1e_inuc_rxp`/`inuc_rcxp` (3), `int3c2e_ig1` (3). **GIAO×σ (hardest GIAO slice, also needs the σ path):** `int1e_spgsp`/`spgnucsp`/`spgsa01` (3/9), `int2e_spgsp1`/`g1spsp2`/`spgsp1spsp2` (3), and the SOC-GIAO `int1e_{cg,giao}_sa10sp`/`sa10nucsp`/`sa10sa01`/`sa01sp` (3/9). Schedule spin-free GIAO with Group 3; schedule GIAO×σ with/after Group 4.

---

### Group 6 — Gauge / Breit–Gaunt 2e integrals (most specialized differentiator — **spinor-σ + complex**)

**Physics:** The gauge term of the Breit interaction, split into r1 and r2 parts (`BREIT-R1`/`BREIT-R2` operators with the `R0` position factor), contracted with σ·p on small-component indices. These complete the full Dirac–Coulomb–Breit Hamiltonian. They are the union of the two hardest dependencies: Pauli-σ spinor coupling (`c2s_si_2e1i`/`c2s_si_2e2i`) **and** Gaunt-style small-component ERI evaluation. component_rank=1 (the σ structure lives entirely in the spinor transform).

| Family | Operator | component_rank | Reps | Use case |
|--------|----------|----------------|------|----------|
| `int2e_gauge_r1_ssp1ssp2` | (i R0 σ·p j\|BREIT-R1\|k σ·p l) | 1 | **spinor** (si/si) | Breit gauge term, r1 part |
| `int2e_gauge_r1_ssp1sps2` | r1, σ·p index permutation | 1 | **spinor** | Breit gauge r1 |
| `int2e_gauge_r1_sps1ssp2` | r1 permutation | 1 | **spinor** | Breit gauge r1 |
| `int2e_gauge_r1_sps1sps2` | r1 permutation | 1 | **spinor** | Breit gauge r1 |
| `int2e_gauge_r2_ssp1ssp2` | (i σ·p j\|BREIT-R2\|k R0 σ·p l) | 1 | **spinor** | Breit gauge term, r2 part |
| `int2e_gauge_r2_ssp1sps2` | r2 permutation | 1 | **spinor** | Breit gauge r2 |
| `int2e_gauge_r2_sps1ssp2` | r2 permutation | 1 | **spinor** | Breit gauge r2 |
| `int2e_gauge_r2_sps1sps2` | r2 permutation | 1 | **spinor** | Breit gauge r2 |

The plain Gaunt families (`int2e_ssp1ssp2`, `ssp1sps2`, `sps1ssp2`, `sps1sps2`) are listed under Group 4; the `gauge_r1/r2` set is the additional gauge correction on top of Gaunt that makes the full Breit interaction. Adjacent magnetic-Gaunt (absolute hardest, GIAO×Gaunt×σ): `int2e_gssp1ssp2` (3), `int2e_cg_ssa10ssp2`/`giao_ssa10ssp2` (3).

---

## Table-Stakes vs Differentiators (the requested split)

### Table Stakes (any libcint-compatible engine is expected to provide these)

| Group | Families | Why expected | Complexity | Spinor-σ dependency |
|-------|----------|--------------|------------|---------------------|
| G1 remaining 1st-derivatives | `int2e_ip2`, `int1e_ip{nuc,kin,ovlp}ip`, `int3c1e_ip1`/`iprinv`, `int2c2e_ip1`/`ip2`, `int3c2e_ip2` | Every analytical-gradient and DF-gradient code needs the full ip-set; Phase 21 shipped only half | LOW–MEDIUM (reuses `gout_ip1`) | None (spin-free; spinor deferrable) |
| G2 Hessian/higher-order | `int1e_ipip{ovlp,nuc,kin}`, `int1e_ipiprinv`, `int2c2e_ipip1`, `int3c2e_ipip1`/`ipip2`, `int1e_ipipipiprinv` + adjacent ∇∇ 2e/3c | Analytical frequencies, IR/Raman, CPHF/CPKS | MEDIUM (2nd–4th order nabla recurrence; cache growth ×9/×27/×81) | None |
| G3 moments | `int1e_r`/`rr`/`rrr`/`rrrr`, `r2`/`r4`, `z`/`zz`, `p4`, `rinv`, `drinv`, `irp` | Dipole/quadrupole/multipole moments, ESP, EFG, polarizability are baseline property outputs | LOW–MEDIUM (position-operator G-tensor on existing OS recurrence) | None (except `sp`, which is σ — moves to G4) |

### Specialized Differentiators (set cintx apart; not all consumers need them)

| Group | Families | Value proposition | Complexity | Spinor-σ dependency |
|-------|----------|-------------------|------------|---------------------|
| G4 relativistic spin-operator | `spsp`, `spnucsp`, `sprinvsp`, `srsr`, `sigma`, `int2e_spsp1`/`srsr1`/`ssp*`/`sps*`/`vsp*`/`spv*`, `int1e_sp` | Enables Dirac/X2C/DKH relativistic and SOC methods — rare in Rust ecosystems | HIGH (Pauli-σ coupling, small-component, complex spinor output) | **HARD — requires R5/D-03 spinor path** |
| G5 GIAO/NMR | `int1e_giao_*`, `int1e_cg_*`, `a01gp`, `ia01p`, `ig*`, `g*`, `govlp`/`gnuc`, `int2e_g*` | NMR shielding, magnetizability, rotational g-tensor — high-value property suite | HIGH (purely imaginary integrals, gauge-factor recurrence, complex algebra even in cart/sph) | Spin-free GIAO: MEDIUM; GIAO×σ (`spg*`, `*_sa10*`): HARD |
| G6 gauge/Breit–Gaunt 2e | `int2e_gauge_r1/r2_*`, Gaunt `ssp/sps`, magnetic-Gaunt | Completes full Dirac–Coulomb–Breit — apex of relativistic 2e | VERY HIGH (σ + Breit gauge + small-component 2e + complex) | **HARDEST — σ spinor path + Gaunt 2e** |

**Justification for the split:** G1–G3 are spin-free, real (G5 excepted), reuse already-validated recurrences (OS, Rys, the Phase-21 nabla), and are demanded by mainstream non-relativistic workflows (gradients, frequencies, dipoles/multipoles) — non-negotiable "complete the standard API" work. G4–G6 require capabilities cintx has only partially exercised: σ-Pauli spinor coupling (`c2s_si_*`) and complex-valued output, and in G6 also small-component Gaunt 2e. **The source routing is the dividing line:** every G4/G6 family and the hard slice of G5 routes through `c2s_si_*` and/or `c2s_zset0`, whereas all of G1/G2 and most of G3 route through real `c2s_cart/sph_*` or spin-free `c2s_sf_*`.

---

## Feature Dependencies

```
Phase-21 gout_ip1 / nabla recurrence  (DONE)
        ├──enables──> G1 remaining 1st-derivatives  (1 nabla, different index/envelope)
        └──extends──> G2 Hessian/higher-order  (2nd/3rd/4th nabla recurrence + larger cache)

Obara–Saika + position-operator (RCJ/RCI) recurrence  (partially present)
        └──enables──> G3 moments  (r, rr, rrr, rrrr, z, zz, r2, r4, irp, drinv)

Phase-12 real spinor transform (Clebsch–Gordan)  (DONE, scalar only)
        └──prerequisite for──> R5/D-03 spinor-derivative/σ path  (NOT yet exercised)
                                   ├──required by──> G4 relativistic σ·p (c2s_si_*)
                                   ├──required by──> G5 GIAO×σ slice (spg*, *_sa10*)
                                   └──required by──> G6 gauge/Breit–Gaunt (c2s_si_2e*i)

Complex-valued / imaginary-only (zset0) output machinery  (NEW capability)
        └──required by──> G5 GIAO/magnetic (all ig*/g* families are imaginary)
                          └──also needed by──> G4 sigma/SOC and G6 (complex spinor)

Gaunt/Breit small-component 2e evaluation  (NEW)
        └──required by──> G4 Gaunt 2e (ssp/sps) and G6 gauge term
```

### Dependency notes

- **G1 requires only the existing Phase-21 nabla machinery** — lowest risk, should go first; closes the analytical-gradient set and the DF-gradient ket side (`int3c2e_ip2`, `int2c2e_ip*`).
- **G2 requires extending the nabla recurrence to higher order** plus a larger workspace/cache (component counts jump to 9/27/81 — chunk-planner and OOM-safety paths must be re-validated at these sizes).
- **G3 requires the position-operator branch of OS** — partially present (`gout_int1e_r` uses `G1E_RCJ`); mostly real and independent, can run in parallel with G1/G2.
- **G4, G6, and the hard slice of G5 all require the σ-Pauli spinor path (R5/D-03)**, which PROJECT.md flags as a deferred prerequisite (Phase-21 spinor gradients are registered-but-`UnsupportedApi`, confirmed: `int1e_ipovlp_spinor` has `oracle_covered:false`). This is the largest unknown and the dominant ordering constraint: nothing in G4/G6 can reach oracle parity until the σ-coupled spinor transform is exercised end-to-end. Recommend a dedicated spinor-path enablement phase before G4.
- **G5 splits internally**: spin-free GIAO families (`govlp`, `gnuc`, `ig*`, `g*`, `a01gp`, `ia01p`) need only complex/imaginary output (a new but self-contained capability); GIAO×σ families (`spg*`, `*_sa10*`) additionally need the σ path. Schedule spin-free GIAO with G3 and σ-GIAO with/after G4.
- **G6 conflicts with "defer spinor"** — impossible without both the σ path and Gaunt small-component 2e; it is the natural last group.

---

## MVP Definition (phase-ordering recommendation for v1.4)

### Build first (lowest risk, highest demand) — real, spin-free, reuse existing recurrences
- [ ] **G1 remaining 1st-derivatives** — reuses validated `gout_ip1`; completes gradients/DF-gradients. *Trigger: Phase 21 done (it is).*
- [ ] **G3 moments (real subset)** — dipole/quadrupole/multipole, r²/r⁴, z/zz, rinv/drinv, irp, p4; position-operator OS branch. Can parallel G1.

### Build after derivative/recurrence extension
- [ ] **G2 Hessian/higher-order** — needs 2nd–4th order nabla + revalidated chunk/OOM at ×9/×27/×81 component sizes.
- [ ] **G5 spin-free GIAO/NMR** — needs new complex/imaginary-output capability but no σ; delivers NMR shielding + magnetizability for non-relativistic methods.

### Build after the spinor-σ path is enabled (R5/D-03 lifted)
- [ ] **Spinor-σ enablement (prerequisite work, not a family)** — exercise `c2s_si_*` end-to-end; the gate for everything below.
- [ ] **G4 relativistic spin-operator** — `spsp`/`spnucsp`/`sprinvsp`/`srsr`/`sigma`/`sp` + small-component 2e.
- [ ] **G5 GIAO×σ slice** — `spg*`, `*_sa10*` (GIAO and σ together).

### Build last (apex; both σ and Gaunt 2e)
- [ ] **G6 gauge/Breit–Gaunt 2e** — `int2e_gauge_r1/r2_*` + magnetic-Gaunt. Highest complexity, fullest dependency stack.

---

## Feature Prioritization Matrix

| Group | User Value | Implementation Cost | Spinor-σ gate | Priority |
|-------|------------|---------------------|---------------|----------|
| G1 remaining 1st-derivatives | HIGH (every gradient code) | LOW | No | P1 |
| G3 moments (real) | HIGH (every property code) | LOW–MEDIUM | No | P1 |
| G2 Hessian/higher-order | HIGH (frequencies/CPHF) | MEDIUM | No | P1 |
| G5 spin-free GIAO/NMR | MEDIUM–HIGH (NMR niche, high value) | MEDIUM (new complex path) | No | P2 |
| G4 relativistic spin-operator | MEDIUM (relativistic/SOC) | HIGH | **Yes** | P2 |
| G5 GIAO×σ slice | LOW–MEDIUM | HIGH | **Yes** | P3 |
| G6 gauge/Breit–Gaunt 2e | LOW (full DCB, rare) | VERY HIGH | **Yes** + Gaunt 2e | P3 |

**Priority key:** P1 = standard-API table stakes, build first; P2 = high-value differentiators, build once prerequisites land; P3 = apex relativistic, build last / may trail the milestone.

---

## Anti-Features (commonly assumed, but a parity scope should resist)

| Tempting move | Why it seems good | Why problematic | Better approach |
|---------------|-------------------|-----------------|-----------------|
| Implement σ-coupled cart/sph variants as first-class outputs | Header declares cart/sph for every family; symmetry tempts uniform treatment | For G4/G6 the cart/sph variant is only a raw spin-free intermediate, not a physical observable; treating it as a deliverable inflates the oracle surface with non-meaningful numbers | Register cart/sph for symbol-coverage parity but gate oracle expectations to what libcint itself emits; physical-correctness check is the spinor output |
| Build all spinor variants eagerly with the scalar transform | Phase 12 spinor transform exists | The deferred R5/D-03 shows derivative/σ spinor is not yet exercised end-to-end; eager registration without the σ path yields `UnsupportedApi` placeholders, not parity | Add an explicit spinor-σ enablement phase; flip `oracle_covered` only when the σ path passes |
| Treat GIAO integrals as real | Most integrals so far are real f64 | All `ig*`/`g*` families are purely imaginary (`c2s_zset0`); reading them as real silently zeros the answer | Introduce complex/imaginary output handling as an explicit capability before G5 |
| Lump tensor=27/81 families into the same workspace plan as tensor=3 | "It's the same recurrence" | Component count ×9–×27 blows up cache/buffer sizing; the OOM-safe stop contract must be re-validated | Re-derive chunk-planner limits for high-rank families; add OOM tests at rank 27/81 |
| Skip the `_origj`/`_origi`/adjacent families | Not in the headline list | libcint exposes them and the full-parity claim (the verification gate) requires every declared symbol | Fold adjacent same-recurrence families (origj moments, ipip 2e/3c, Gaunt permutations) into the matching group's REQ set |

---

## Confidence & gaps

- **Family enumeration, operator strings, component_rank, representation routing: HIGH.** All read from vendored libcint 6.1.3 (`cint_funcs.h` comments + `ng[]` arrays + `c2s_*` driver routing in `intor*.c`/`grad*.c`/`hess.c`/`deriv*.c`/`dkb.c`/`gaunt1.c`/`breit1.c`/`int3c*.c`). 187 families were extracted with their tensor factors; all 6 groups' headline families confirmed MISSING (genuinely to-build) against the current manifest.
- **Use-case attribution: MEDIUM–HIGH.** Standard quantum-chemistry mapping (PySCF/textbook), cross-checked against operator semantics, not independently re-derived.
- **Gaps for downstream REQ/phase work:**
  - The exact set of "adjacent" families to pull into each group (origj moments, the full ipip 2e/3c set, Gaunt permutations, GIAO×σ) needs a final enumeration pass against the manifest-audit gate to guarantee the full-parity claim — flagged inline per group.
  - The cost/design of the **spinor-σ enablement** (`c2s_si_*`) and the **complex/imaginary-output** capability are the two cross-cutting unknowns; both deserve a dedicated phase/spike before G4/G5-σ/G6.
  - High-rank (27/81) OOM-safety and chunk-planner limits are untested at these component counts.

## Sources

- `libcint-master/include/cint_funcs.h` — declared family list + operator comments (HIGH)
- `libcint-master/src/autocode/{intor1,intor2,intor3,intor4,grad1,grad2,hess,deriv3,deriv4,dkb,gaunt1,breit1,int1e_grids1}.c` and `src/{int3c1e,int3c2e}.c` — `ng[]` tensor factors, `gout` operators, `c2s_*` spinor routing (HIGH)
- `crates/cintx-ops/generated/compiled_manifest.lock.json` — existing `component_rank` convention and the `int1e_ipovlp_spinor` / R5–D-03 deferred-spinor pattern (HIGH)
- `.planning/PROJECT.md` — milestone scope, the 6-group decomposition, R5/D-03 spinor-derivative prerequisite, per-family pattern (HIGH)

---
*Feature research for: libcint 6.1.3 full-family-parity milestone (v1.4)*
*Researched: 2026-05-27*
