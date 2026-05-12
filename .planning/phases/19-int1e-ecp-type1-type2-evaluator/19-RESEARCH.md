# Phase 19: `int1e_ecp_*` Type-1/Type-2 Evaluator — Research

**Researched:** 2026-05-12
**Domain:** Effective Core Potential (ECP) projector integral evaluation; libcint result-compatibility via vendored C reference; CubeCL `#[cube]` kernel + cart-to-sph integration.
**Confidence:** HIGH for architectural integration points, oracle-source landscape, and slot-constant layout. MEDIUM for algorithmic details (Type-1 closed form vs Type-2 quadrature parameters). LOW only for performance tuning numbers (radial node counts), which are configurable.

## Summary

Phase 19 ships six `int1e_ecp_*` symbols (4 base + 2 gradient) through cintx's existing `SessionRequest` → `cintx-ops` resolver → `cintx-cubecl` launcher pipeline, gated by a new Cu/LANL2DZ oracle fixture. **The single biggest planning risk surfaced by this research is concrete and confirmed: libcint 6.1.3 upstream ships NO ECP source code — the file `ecp.c` and the slot constants `AS_ECPBAS_OFFSET` / `AS_NECPBAS` / `RADI_POWER` / `SO_TYPE_OF` all live in PySCF's `pyscf/lib/gto/nr_ecp.{c,h}` (Apache-2.0, Qiming Sun)**, not in libcint. CONTEXT.md `<specifics>` flagged this as a research-time blocker; the blocker is real and D-01 as written is unimplementable. Two viable paths exist:

1. **Vendor PySCF's `nr_ecp.c` + `nr_ecp.h` + `nr_ecp_deriv.c` as the byte-identity oracle** (Apache-2.0, compatible with cintx; same author as libcint; symbol names `ECPtype1_{cart,sph}`, `ECPtype2_{cart,sph}`, `ECPscalar_*`, `ECPso_spinor` — not `int1e_ecp_*`, so cintx-oracle's FFI shim must adapt). Honors ROADMAP SC#4's intent (libcint-ecosystem reference, same author).
2. **Promote libecpint (Shaw & Hill, JCP 147, 074108, 2017; MIT) to primary** — actively maintained C++ library, includes 1st/2nd derivatives, paper-cited algorithm. Departs from SC#4 wording ("vs libcint") but is the cleaner long-term oracle.

**The chrr/libECP suggested in ROADMAP/CONTEXT D-02 is unmaintained, sparsely documented, and not the same as libecpint.** This finding should be surfaced for user decision before plan-phase commits — see "Open Questions for Planner" below.

**Primary recommendation:** Vendor PySCF's `nr_ecp.{c,h}` + `nr_ecp_deriv.c` as the primary byte-identity oracle (path 1) — same Apache-2.0 license as libcint, same author, drop-in to the existing `cintx-oracle/build.rs` `cc::Build` pattern, and the ECP slot constants (`AS_ECPBAS_OFFSET = 18`, `AS_NECPBAS = 19`, `RADI_POWER = 3`, `SO_TYPE_OF = 4`, `ECP_LMAX = 5`) come straight from `nr_ecp.h`. Treat libecpint as the secondary informational cross-check instead of the unmaintained chrr/libECP. Implement Type-1 and Type-2 cintx kernels following PySCF's `ECPtype1_*` / `ECPtype2_*` algorithm — these are the algorithm cintx must match byte-for-byte.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ECP-01 | Type-1 (Coulomb-like) ECP projector evaluator implemented as `#[cube]` kernel + `*_host()`; registered in `cintx-ops`. | Algorithm in PySCF `nr_ecp.c` `ECPtype1_*` family; closed-form sum of Gaussian-modulated $r^n$ terms with `RADI_POWER` slot controlling $n$. Uses Gauss-Hermite-like radial integration over Gaussian × Gaussian × $r^n$ products. CubeCL math infra precedent: Phase 8 (boys, obara_saika, rys). |
| ECP-02 | Type-2 (semi-local, spin-orbit-like) projector with spherical-harmonic angular projectors + Bessel-modulated radial integrals; registered. | Algorithm in PySCF `nr_ecp.c` `ECPtype2_*`. Uses Gauss-Chebyshev radial quadrature (Shaw & Hill 2017 mentions same family) over modified-spherical-Bessel-multiplied Gaussians. Tabulated Bessel via `K_TAB_*` (`K_TAB_ENTRIES = 400`, `K_TAB_INTERVAL = 0.04`) per `nr_ecp.h`. |
| ECP-03 | `SessionRequest::evaluate` dispatches `int1e_ecp_{sph,cart}` through the same surface as ordinary 1e operators — no parallel API. | Existing `cintx-ops/src/resolver.rs` is manifest-driven (Phase 13 F12 precedent); adding 6 rows to `api_manifest.csv` + a `canonical_family = "ecp"` arm in `cintx-cubecl/src/kernels/mod.rs::resolve_family_name` is the entire wiring delta. `BasisSet::ecp_shells()` accessor (per D-03) preflighted by `query_workspace()` (next to existing `aosym` check). |
| ECP-04 | Cu/LANL2DZ passes byte-identity parity vs libcint at `atol=1e-12` through both `eval_raw` and `SessionRequest::evaluate`. Secondary cross-check vs libECP (chrr, JCC 2017) non-blocking. | **D-01 IS BLOCKED**: libcint 6.1.3 ships zero ECP source — see "Open Questions for Planner". Once oracle source is selected (vendor PySCF `nr_ecp` recommended), `cintx-oracle/build.rs` extends its `cc::Build` source list and `tests/safe_api_ecp_parity.rs` follows the Phase 17/18 `safe_api_arity2_parity.rs` pattern at the Phase 15 unified tolerance. Cu/LANL2DZ fixture is built fresh in `crates/cintx-oracle/src/fixtures.rs::build_cu_lanl2dz()` (no existing Cu fixture in the repo). |
| ECP-05 | Decision on gradient variant scope recorded in SPEC.md before plan-phase. | **DECIDED in CONTEXT D-10**: gradients included in Phase 19. This research confirms the decision is feasible — PySCF's `nr_ecp_deriv.c` is a sibling file to `nr_ecp.c` with derivative variants of `ECPtype1_*` / `ECPtype2_*` that share roughly the same math infrastructure (Bessel tables, radial quadrature). Component_rank=3 manifest convention matches existing `int3c2e_ip1_*`. |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| ECP shell typed surface (`EcpShell`, `BasisSet::ecp_shells()`) | `cintx-core` | `cintx-rs` (re-export) | Basis-set-belonging data; sits beside `Shell`. D-03/D-04. |
| ECP env-slot constants + `EcpBasArray` raw view | `cintx-compat` | none | Raw API contract — slot offsets and i32 slab marshaling. D-05. |
| `MissingEcpBasis` typed error | `cintx-rs` | `cintx-capi` (status code, follow-up) | Safe-API surface preflight. D-06. |
| Type-1 / Type-2 algorithm | `cintx-cubecl` (`kernels/ecp.rs`, `math/bessel.rs`, `math/radial_quadrature.rs`) | none | CubeCL primary backend per PROJECT.md; D-07 / D-08. |
| Manifest routing for 6 new symbols | `cintx-ops` (CSV + lock regen) | none | Manifest-driven resolver per Phase 13 precedent. D-09. |
| Oracle reference source | `cintx-oracle` (`build.rs` + FFI) | none | Vendored C sources via `cc::Build`. **D-01 BLOCKED — see Open Questions**. |
| Cu/LANL2DZ fixture | `cintx-oracle/src/fixtures.rs` | none | Mirrors `build_h2o_sto3g` PTR_ENV_START-aligned env layout. |
| Cart-to-sph for ECP output | `cintx-cubecl/src/transform/c2s.rs::cart_to_sph_1e` (reused) | none | Existing arity-2 1e transform handles ECP unchanged. |
| Spinor representation | OUT OF SCOPE this phase (D-12) | follow-up phase | Multi-component spinor transform for Type-2 SO is a v1.4 candidate. |

## Algorithmic Foundation (Type-1 + Type-2)

### General ECP form

An ECP-projected one-electron operator is:

$$V_{\mathrm{ECP}}(r) \;=\; V_L(r) \;+\; \sum_{l=0}^{L-1} \bigl[V_l(r) - V_L(r)\bigr] \, \sum_{m=-l}^{l} |Y_{lm}\rangle\langle Y_{lm}|$$

Where $V_L(r)$ is the "local" channel (Type-1) and the $\sum_l$ part is the "semi-local" projector (Type-2). Each radial function $V_l(r)$ is a sum of Gaussian-multiplied $r^n$ terms:

$$V_l(r) \;=\; \sum_k d_{k,l} \, r^{n_{k,l} - 2} \, e^{-\zeta_{k,l} r^2}$$

The integers $n_{k,l}$ are stored in libcint/PySCF ecpbas as `RADI_POWER` (slot index 3 in `_ecpbas[]`); $\zeta_{k,l}$ and $d_{k,l}$ live in env at offsets `PTR_EXP`/`PTR_COEFF` slots of the ecpbas row. The marker `SO_TYPE_OF` (slot 4) is 0 for scalar (Type-1 local OR Type-2 semi-local non-SO) and nonzero for spin-orbit channels (out of scope this phase per D-12). [CITED: pyscf `nr_ecp.h` + `gto/ecp.py`; VERIFIED via the WebFetch findings below]

### Type-1 (local, Coulomb-like)

The Type-1 integral for shell pair $(i, j)$ centered on $A_i, A_j$ with ECP center $A_C$:

$$\langle \chi_i | V_L | \chi_j \rangle \;=\; \sum_k d_k \, n_k\text{-power integral}$$

where each term is a Gaussian × Gaussian × ($r_C^{n-2} e^{-\zeta r_C^2}$) integral evaluated with **Gauss-Hermite-style radial expansion** (the integrand reduces to a Gaussian product times a polynomial in $r_C$ after substitution).

- **Closed form exists for $n = 0$** (pure Gaussian product → Boys function family).
- **For $n > 0$**, libcint/PySCF uses a radial integration over the Gaussian product; PySCF's `ECPtype1_cart` performs this via a power-series + asymptotic expansion split, NOT a fixed quadrature. [CITED: pyscf source structure]
- **CubeCL parallel:** Phase 8's `boys.rs` already provides the Boys function machinery for $n = 0$; Type-1's $n > 0$ extension reuses the same Gaussian-product PA/PB geometry but evaluates a polynomial-times-Gaussian radial integral. [VERIFIED: Phase 8 module layout]

### Type-2 (semi-local, angular-momentum-projected)

For each $l$-channel:

$$\langle \chi_i | V_l(r) |Y_{lm}\rangle\langle Y_{lm}| \chi_j \rangle$$

After expansion of $\chi_i, \chi_j$ around the ECP center $A_C$ via the **Bessel-modulated translation** (the spherical wave expansion):

$$e^{2\zeta_i \vec{A}_i \cdot \vec{r}} \;=\; \sum_l (2l+1) \, i_l(2\zeta_i A_i r) \, P_l(\cos\theta)$$

where $i_l$ is the **modified spherical Bessel function of the first kind**. The angular part collapses (Wigner / Clebsch-Gordan), leaving a **radial integral** of the form:

$$\int_0^\infty r^{n+2} \, e^{-\alpha r^2} \, i_{l_1}(\beta_1 r) \, i_{l_2}(\beta_2 r) \, dr$$

which has **no closed form in general** and is evaluated by **Gauss-Chebyshev second-kind quadrature on a transformed radial grid** (`LEVEL0 = 5`, `LEVEL_MAX = 11` per `nr_ecp.h`, doubling to a maximum of 2047 radial points with adaptive refinement). [CITED: pyscf `nr_ecp.h`]

**Bessel function evaluation strategy** (per `nr_ecp.h` and Shaw & Hill 2017): for small argument, a Taylor expansion (`K_TAYLOR_MAX = 7` terms); for moderate argument, a tabulated lookup (`K_TAB_ENTRIES = 400`, `K_TAB_INTERVAL = 0.04`, `K_TAB_COL = 24` columns spanning $l = 0..23$) over $[0, 16]$; for large argument, the asymptotic form $i_l(x) \sim e^x / (2x)$. CubeCL implementation (D-07) replicates this split in `math/bessel.rs` with `#[cube]` + `*_host()` pair (Phase 8 pattern).

**Numerical stability concern** [VERIFIED by Shaw & Hill 2017 abstract]: the upward recurrence for $i_l(x)$ is unstable; downward (Miller's) recurrence is required for $l > $ argument, OR the table-then-recurrence hybrid that PySCF uses. The Phase 19 planner should document this in `bessel.rs` rustdoc and choose ONE strategy explicitly — the failure mode is silent precision loss, not panic.

### Gradient form (`int1e_ecp_ipnuc_*`)

The "ipnuc" (i.e., derivative w.r.t. the nuclear position of the ECP center) gradient is structurally similar to the base: a derivative of $V_l(r)$ w.r.t. $A_C$ adds one factor of $(\vec{r} - \vec{A}_C)$ inside the radial integrand, which shifts the angular momentum projector channels by ±1 and changes the $r^n$ power by ±1. PySCF's `nr_ecp_deriv.c` implements this as variants of the base `ECPtype1_*` / `ECPtype2_*` routines sharing the Bessel + radial quadrature infrastructure. Component_rank = 3 (one derivative per Cartesian axis), and the F-order layout convention `[axis, ao_j, ao_i]` (component slowest-varying) matches existing `int3c2e_ip1_*`. [CITED: pyscf `nr_ecp_deriv.c` exists per pyscf/lib/gto file listing; VERIFIED via WebFetch]

## Reference Implementations Surveyed

### 1. libcint v6.1.3 upstream — **NO ECP CODE PRESENT**

[VERIFIED: vendored `libcint-master/src/` ls output; VERIFIED: WebFetch of `github.com/sunqm/libcint/tree/v6.1.3/src`]

- Vendored `libcint-master/src/` contains 0 ECP source files (`ls` grep for `*ecp*` returns nothing).
- `libcint-master/include/cint.h.in` defines `AS_*` slot indices up to **18** unused but does **not** define `AS_ECPBAS_OFFSET` or `AS_NECPBAS`.
- Upstream libcint 6.1.3 source list at github.com/sunqm/libcint/tree/v6.1.3/src confirms NO `ecp.c`, `cint_ecp.h`, or any `*ecp*` file.
- `libcint-master/include/cint_funcs.h` declares no `int1e_ecp_*` symbols (grep returns empty).

**Implication:** Vendoring "libcint's ecp.c" (CONTEXT D-01) is **impossible as written**. The ECP code historically attributed to "libcint upstream" actually lives in PySCF's `pyscf/lib/gto/` directory and is written by the same author (Qiming Sun).

### 2. PySCF `pyscf/lib/gto/nr_ecp.{c,h}` — **THE de-facto libcint-ecosystem ECP reference**

[CITED: github.com/pyscf/pyscf/blob/master/pyscf/lib/gto/nr_ecp.h via WebFetch; CITED: pyscf/lib/gto/CMakeLists.txt]

- File size: 6543 lines of `nr_ecp.c` (991 KB) — large but Apache-2.0 (Qiming Sun, 2014-2018, same author as libcint).
- Headers: `<stdlib.h> <stdint.h> <math.h> <complex.h>` + `"cint.h"` + `"np_helper/np_helper.h"` + `"vhf/fblas.h"` + `"gto/nr_ecp.h"`.
- **Dependencies on libcint:** uses `cint.h` for shared `BAS_SLOTS`, `ATM_SLOTS`, `ATOM_OF`, `ANG_OF`, etc. — already vendored. Uses `CINTcommon_fac_sp` for s/p normalization — already vendored.
- **NOT trivially in cintx-oracle's current `cc::Build`:** `np_helper/np_helper.h` and `vhf/fblas.h` are pyscf-internal headers; cintx-oracle would need to provide minimal shims OR strip those includes (likely they only provide BLAS forwarders + numpy memory helpers that can be stubbed). Researcher recommends a "minimal subset vendor" path with a `.planning/notes/pyscf-ecp-vendor-subset.md` rationale.
- **Public symbols exposed (per `pyscf/gto/ecp.py` ctypes bindings):**
  - `ECPtype1_cart`, `ECPtype1_sph` — Type-1 (local) projector
  - `ECPtype2_cart`, `ECPtype2_sph` — Type-2 (semi-local) projector
  - `ECPso_spinor` — Type-2 spin-orbit (deferred per D-12)
  - `ECPscalar_cache_size` — cache allocation helper
  - Derivative variants in `nr_ecp_deriv.c` (separate compilation unit)
- **Slot constants from `nr_ecp.h`:** `AS_ECPBAS_OFFSET = 18`, `AS_NECPBAS = 19`, `RADI_POWER = 3`, `SO_TYPE_OF = 4`, `ECP_LMAX = 5`. These are the canonical values cintx-compat::raw must expose (D-05's guessed constants `ECP_BAS_SLOTS = 8`, `PTR_ECPBAS_OFFSET`, `PTR_NECPBAS` were speculation; **the real names and values are above** — researcher recommends adopting upstream names verbatim).
- **Algorithm tunables defined in `nr_ecp.h`:** `K_TAYLOR_MAX = 7`, `K_TAB_ENTRIES = 400`, `K_TAB_COL = 24`, `K_TAB_INTERVAL = 0.04` (Bessel tables); `LEVEL0 = 5`, `LEVEL_MAX = 11` (radial grid levels — 2047 max points).
- **Function pointer typedef:** `typedef int Function_cart(double *gctr, int *shls, int *ecpbas, int necpbas, int *atm, int natm, int *bas, int nbas, double *env, ECPOpt *opt, double *cache);` — note `gctr` is the output buffer; signature differs from cintx-compat's `eval_raw` shape.

### 3. libecpint (Shaw & Hill, robashaw/libecpint) — **STRONG SECONDARY ORACLE CANDIDATE**

[CITED: github.com/robashaw/libecpint; VERIFIED: Shaw & Hill JCP 147, 074108 (2017)]

- **License:** MIT (cintx-workspace-compatible).
- **Language:** C++ (CMake build) — Rust FFI requires `cxx` or an `extern "C"` shim layer.
- **Algorithm:** Recursive code generation + Gauss-Chebyshev quadrature; analytical 1st and 2nd derivatives (Phase 19 only needs 1st, so this is a superset).
- **Paper:** Shaw & Hill, *J. Chem. Phys.* **147**, 074108 (2017) — "Prescreening and efficiency in the evaluation of integrals over ab initio effective core potentials".
- **Maintenance status:** Active (v1.0.7 released Dec 2021; recent patches as of 2026). **Substantially better maintained than chrr/libECP** which has sparse docs and last activity unclear.
- **Trade-offs:**
  - + Modern C++, well-documented, paper-cited algorithm, derivatives included.
  - − C++ adds Rust FFI complexity vs cintx-oracle's existing all-C `cc::Build` pattern.
  - − Different normalization / phase conventions vs libcint/PySCF: byte-identity (atol=1e-12) is unlikely; informational tolerance (atol≈1e-9) is realistic.

### 4. chrr/libECP — **NOT RECOMMENDED**

[CITED: github.com/chrr/libECP README; CITED: WebFetch findings]

- **License:** BSD-2-Clause (compatible).
- **Status:** Sparse documentation ("Documentation is unfortunately quite sparse at the moment" per the project's own README), no published paper, last activity unclear, no formal release.
- **The "JCC 2017" reference in CONTEXT.md D-02 / ROADMAP is suspect** — the JCP 2017 paper (Shaw & Hill 147, 074108) is for **libecpint** (robashaw), not chrr/libECP. The ROADMAP citation is likely a confusion. Researcher recommends replacing the chrr/libECP secondary oracle target with libecpint.

### 5. PySCF `pyscf/gto/ecp.py` — **HIGH-LEVEL REFERENCE ONLY**

- Pure Python; calls `ECPtype1_*` / `ECPtype2_*` via ctypes from `libcgto.so` (which is the compilation unit that includes `nr_ecp.c`).
- Not a runtime dependency; useful as a Python-level cross-check for spot-checking Cu/LANL2DZ outputs during development.
- Public function `type1_by_shell(mol, shls, cart=False)`, `type2_by_shell(mol, shls, cart=False)`, `so_by_shell(mol, shls)` — same shape as cintx's planned arity-2 API.

## cintx Integration Points

### Operator catalog (`cintx-ops`)

[VERIFIED: `crates/cintx-ops/src/generated/api_manifest.csv`, 133 lines]

**Manifest CSV header:** `family_name,operator_name,symbol_name,category,arity,forms,component_rank,feature_flag,stability,declared_in,compiled_in_profiles,oracle_covered,helper_kind,canonical_family`

**Six new rows to add (D-09):**

```csv
"1e","ecp","int1e_ecp_cart","1e",2,"cart","","none","stable","unknown","base|with-f12|with-4c1e|with-f12+with-4c1e",false,"operator","ecp"
"1e","ecp","int1e_ecp_sph","1e",2,"sph","","none","stable","unknown","base|with-f12|with-4c1e|with-f12+with-4c1e",false,"operator","ecp"
"1e","ecp_ipnuc","int1e_ecp_ipnuc_cart","1e",2,"cart",3,"none","stable","unknown","base|with-f12|with-4c1e|with-f12+with-4c1e",false,"operator","ecp"
"1e","ecp_ipnuc","int1e_ecp_ipnuc_sph","1e",2,"sph",3,"none","stable","unknown","base|with-f12|with-4c1e|with-f12+with-4c1e",false,"operator","ecp"
```

(Two more variants for ip-nuc if there's both a cart/sph distinction at the catalog level — confirm by referring to existing `int3c2e_ip1_cart`/`int3c2e_ip1_sph` precedent — they appear as two rows; same for the 4 ECP base rows. Wait — counting: base cart, base sph, ipnuc cart, ipnuc sph = 4 rows, not 6. CONTEXT D-12 says "6 symbols total" but excludes spinor; 6 includes the spinor rows. Per D-12 spinor rows are NOT added this phase, so the manifest delta is **4 rows**, not 6. Planner should verify.)

[ASSUMED] The lock file regeneration command is `cargo run -p xtask -- manifest-audit --update` per the Phase 13/15 precedent — verify against `crates/xtask/src/main.rs`.

### Resolver (`cintx-ops/src/resolver.rs`)

[VERIFIED: CONTEXT.md states resolver is manifest-driven — no resolver code changes; only the CSV expansion lights up routing]

### Kernel dispatch (`cintx-cubecl/src/kernels/mod.rs`)

[VERIFIED: read of kernels/mod.rs lines 1-100]

Current `resolve_family_name` arms cover `"1e" | "2e" | "2c2e" | "3c1e" | "3c2e" | "4c1e" | "f12" | "origi" | "grids" | "breit" | "origk" | "ssc"`. **Adding `"ecp" => ecp::launch_ecp` is a 3-line delta** (one match arm, one entry in `supports_canonical_family`, and the module is unconditional per D-09 stability=stable).

### Launcher (`cintx-cubecl/src/kernels/ecp.rs` — NEW)

[CITED: launch signature inferred from `cintx-cubecl/src/kernels/mod.rs::FamilyLaunchFn`]

```rust
pub fn launch_ecp(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError>
```

Internal flow:
1. Verify `plan.operator` is `INT1E_ECP_*` (one of 4 IDs).
2. Read `EcpShell` slice from `plan.basis().ecp_shells()` (new accessor per D-03).
3. For each shell pair (i, j) in `plan.shell_tuple`:
   - For each ECP shell c:
     - Branch on `EcpShell::ecp_type` → `compute_type1_pair(...)` or `compute_type2_pair(...)`.
     - For `int1e_ecp_ipnuc_*`, additionally branch on operator name → derivative form.
4. Accumulate into Cartesian buffer (F-order, ao_j-fastest).
5. Apply `transform::c2s::cart_to_sph_1e` if `plan.representation == Spheric`.

**Reuses:** `CubeClExecutor` dispatch, `HostWorkspaceAllocator`, `schedule_chunks`, `ExecutionPlan`, `ExecutionIo`, `cart_to_sph_1e`.

### Safe-API surface (`cintx-rs/src/api.rs`)

[VERIFIED: read of api.rs lines 1-100]

`SessionRequest::query_workspace()` currently runs an `aosym` preflight at line 67-73. **The `MissingEcpBasis` preflight (D-06) lands at the same insertion point**:

```rust
// After aosym preflight, before runtime_query_workspace:
if self.operator.is_ecp() {  // helper on OperatorId
    if self.basis.ecp_shells().is_empty() {
        return Err(FacadeError::MissingEcpBasis {
            operator: self.operator.canonical_symbol().to_string(),
        });
    }
}
```

`FacadeError::MissingEcpBasis { operator: String }` variant added at the end of variants in `cintx-rs/src/error.rs` (D-06 / Phase 18 `UnsupportedAoSymmetry` precedent).

### Raw compat (`cintx-compat/src/raw.rs`)

[VERIFIED: grep for slot constants returned existing slot infrastructure]

D-05 adds these **constants** (using the upstream PySCF `nr_ecp.h` names, not CONTEXT.md's speculation):

```rust
/// ECP basis row: angular momentum power of r (V_l(r) = sum_k d_k r^(n_k - 2) e^(-zeta_k r^2)).
/// Slot index 3 in an ecpbas row. Matches PySCF nr_ecp.h `RADI_POWER`.
pub const RADI_POWER: usize = 3;

/// ECP basis row: spin-orbit channel marker (0 = scalar, nonzero = SO).
/// Slot index 4. Matches PySCF nr_ecp.h `SO_TYPE_OF`.
pub const SO_TYPE_OF: usize = 4;

/// env slot pointing to start of ecpbas array (env[AS_ECPBAS_OFFSET] = ecpbas_ptr).
/// Matches PySCF nr_ecp.h `AS_ECPBAS_OFFSET = 18`.
pub const AS_ECPBAS_OFFSET: usize = 18;

/// env slot holding number of ecpbas rows (env[AS_NECPBAS] = necpbas).
/// Matches PySCF nr_ecp.h `AS_NECPBAS = 19`.
pub const AS_NECPBAS: usize = 19;

/// Maximum angular momentum supported by the ECP code path (per nr_ecp.h `ECP_LMAX = 5`).
pub const ECP_LMAX: usize = 5;
```

**`ECP_BAS_SLOTS` value:** The ecpbas row reuses the same `BAS_SLOTS = 8` width as ordinary bas — confirmed because slots 3 (`RADI_POWER`) and 4 (`SO_TYPE_OF`) overlap with the standard bas slots `NCTR_OF` (3) and `KAPPA_OF` (4). The ecpbas row reinterprets those two slots; the row width itself stays `BAS_SLOTS = 8`. CONTEXT D-05's `ECP_BAS_SLOTS = 8` value is correct; only the named constants are wrong. **No new `ECP_BAS_SLOTS` constant needed — reuse existing `BAS_SLOTS`.**

[ASSUMED] `eval_raw` for `int1e_ecp_*` symbols accepts the standard `(atm, bas, env)` slabs with `ecpbas` either (a) packed contiguously after `bas` in the same slab and pointed-to via `env[AS_ECPBAS_OFFSET]`, OR (b) passed as a separate slab. The PySCF convention is (a) — verify by reading `ECPtype1_cart`'s implementation when vendoring.

### Typed core (`cintx-core/src/`)

[VERIFIED: shell.rs lines 1-120 + basis.rs lines 1-150]

**New `EcpShell` struct (D-04):** Mirrors `Shell` field layout with two differences:

```rust
// crates/cintx-core/src/ecp.rs (NEW module)
use std::sync::Arc;
use crate::error::{CoreError, CoreResult};

#[derive(Clone, Debug, PartialEq)]
pub struct EcpShell {
    pub atom_index: u32,
    pub ang_momentum: u8,       // l-channel (-1 sentinel for "local channel" per PySCF convention; OR a tag enum — researcher recommends enum)
    pub nprim: u16,
    pub nctr: u16,
    pub radial_power: i16,      // RADI_POWER slot — matches PySCF; signed because some conventions allow negative powers
    pub so_type: i16,           // SO_TYPE_OF slot — 0 for scalar; nonzero out of scope per D-12
    pub exponents: Arc<[f64]>,
    pub coefficients: Arc<[f64]>,
}
```

[ASSUMED] The Type-1 (local) channel is typically encoded with `ang_momentum = -1` in PySCF — confirm before plan-phase or use an enum tag to be explicit. The `ecp_type` field suggested in CONTEXT D-04 collapses into `ang_momentum` if the `-1 = local` convention is adopted.

**`BasisSet::ecp_shells()` accessor (D-03):** Additive field on `BasisSet`. The current struct (basis.rs:48) has 3 fields (`atoms`, `shells`, `meta`); the new field is `ecp_shells: Arc<[Arc<EcpShell>]>` defaulting to `Arc::from([])`. New constructor `BasisSet::try_new_with_ecp(atoms, shells, ecp_shells)` preserves SemVer for existing callers of `BasisSet::try_new`.

**`OperatorId` extension:** Four new constants need IDs. Check `crates/cintx-core/src/operator.rs` for the constant table and add `INT1E_ECP_CART`, `INT1E_ECP_SPH`, `INT1E_ECP_IPNUC_CART`, `INT1E_ECP_IPNUC_SPH`. (Phase 18 added `INT4C1E_CART_OPERATOR_ID = 24` per CONTEXT — pattern is +1 monotonic.)

## Data Flow & Type Surfaces

```
Caller (e.g., pyscf-rs)
    │
    │ 1. Build typed BasisSet with ecp_shells:
    ▼
BasisSet::try_new_with_ecp(atoms, shells, ecp_shells)
    │   crates/cintx-core/src/basis.rs (NEW constructor)
    │
    │ 2. Construct SessionRequest:
    ▼
SessionRequest::new(operator=INT1E_ECP_SPH, basis, shells, options)
    │   crates/cintx-rs/src/api.rs:27
    │
    │ 3. Preflight (D-06):
    ▼
SessionRequest::query_workspace()
    │   crates/cintx-rs/src/api.rs:63
    │   ├─ aosym preflight (existing, line 67)
    │   └─ ecp basis preflight (NEW, line ~74):
    │       if operator.is_ecp() && basis.ecp_shells().is_empty():
    │           return Err(FacadeError::MissingEcpBasis { operator })
    │
    │ 4. Runtime planning (unchanged):
    ▼
runtime_query_workspace → ExecutionPlan
    │
    │ 5. Resolver dispatch (manifest-driven, unchanged):
    ▼
Resolver::descriptor → ManifestEntry { canonical_family = "ecp", ... }
    │   crates/cintx-ops/src/resolver.rs (no code change; CSV row provides entry)
    │
    │ 6. Kernel launch:
    ▼
kernels::launch_family → resolve_family_name("ecp") → launch_ecp
    │   crates/cintx-cubecl/src/kernels/mod.rs (NEW match arm)
    │
    │ 7. Algorithm execution:
    ▼
launch_ecp(backend, plan, key, staging)
    │   crates/cintx-cubecl/src/kernels/ecp.rs (NEW file)
    │   ├─ For each (i, j) shell pair:
    │   │   For each EcpShell c in plan.basis().ecp_shells():
    │   │     Branch on c.ang_momentum (−1 ⇒ Type-1, ≥0 ⇒ Type-2)
    │   │     │
    │   │     ├─ Type-1: compute_type1_pair(i, j, c)
    │   │     │   uses math::radial_quadrature (Gauss-Hermite) + Boys (reused from Phase 8)
    │   │     │
    │   │     └─ Type-2: compute_type2_pair(i, j, c)
    │   │         uses math::bessel + math::radial_quadrature (Gauss-Chebyshev)
    │   │
    │   ├─ For int1e_ecp_ipnuc_*: additional derivative loop (component_rank=3)
    │   │
    │   └─ Accumulate into Cartesian buffer (F-order, ao_j-fastest)
    │
    │ 8. cart-to-sph transform (existing infrastructure):
    ▼
transform::c2s::cart_to_sph_1e(cart_buf, sph_buf, l_i, l_j)
    │   crates/cintx-cubecl/src/transform/c2s.rs (REUSED)
    │
    │ 9. Output:
    ▼
TypedEvaluationOutput<f64> via SessionRequest::evaluate()
```

**Raw API path (parallel)** for `cintx-compat::raw::eval_raw`:

```
Caller passes (operator_id, atm[], bas[], ecpbas[], env[])
    │   ecpbas packed AFTER bas (or via env[AS_ECPBAS_OFFSET] pointer)
    │   env[AS_NECPBAS] = number of ecpbas rows
    ▼
cintx-compat::raw::eval_raw (extended dispatch for INT1E_ECP_*)
    │   crates/cintx-compat/src/raw.rs (NEW dispatch arm)
    │
    │ Fail-closed if env[AS_NECPBAS] == 0
    │   → Err(cintxRsError::InvalidEnvParam { ... })
    │
    ▼ Continue through identical runtime path as safe-API (steps 4-9 above)
```

**Concrete file paths the planner will touch:**

| Path | Action | Reason |
|------|--------|--------|
| `crates/cintx-core/src/ecp.rs` | NEW | `EcpShell` struct, `EcpType` enum if not collapsed into ang_momentum |
| `crates/cintx-core/src/lib.rs` | EDIT | Re-export `EcpShell`, `EcpType` |
| `crates/cintx-core/src/basis.rs` | EDIT | Add `ecp_shells` field, `ecp_shells()` accessor, `try_new_with_ecp` constructor |
| `crates/cintx-core/src/operator.rs` | EDIT | Add 4 new `INT1E_ECP_*` constants + `OperatorId::is_ecp()` helper |
| `crates/cintx-compat/src/raw.rs` | EDIT | Add slot constants (`RADI_POWER`, `SO_TYPE_OF`, `AS_ECPBAS_OFFSET`, `AS_NECPBAS`, `ECP_LMAX`); add `EcpBasArray` typed view; extend `eval_raw` dispatch |
| `crates/cintx-ops/src/generated/api_manifest.csv` | EDIT | Add 4 rows (cart/sph × {ecp, ecp_ipnuc}; spinor deferred per D-12) |
| `crates/cintx-ops/src/generated/compiled_manifest.lock.json` | REGEN | `cargo run -p xtask -- manifest-audit --update` |
| `crates/cintx-cubecl/src/math/bessel.rs` | NEW | Modified spherical Bessel `i_l(x)`, `k_l(x)` with `#[cube]` + `*_host()` pair |
| `crates/cintx-cubecl/src/math/radial_quadrature.rs` | NEW | Gauss-Chebyshev 2nd-kind + Gauss-Hermite nodes/weights with `#[cube]` + `*_host()` |
| `crates/cintx-cubecl/src/math/mod.rs` | EDIT | Register `bessel`, `radial_quadrature` |
| `crates/cintx-cubecl/src/kernels/ecp.rs` | NEW | `launch_ecp` family launcher |
| `crates/cintx-cubecl/src/kernels/mod.rs` | EDIT | Match arms for `"ecp"` in `resolve_family_name`, `supports_canonical_family` |
| `crates/cintx-rs/src/api.rs` | EDIT | `MissingEcpBasis` preflight in `query_workspace` |
| `crates/cintx-rs/src/error.rs` | EDIT | New `FacadeError::MissingEcpBasis { operator: String }` variant + `kind()` arm |
| `crates/cintx-rs/src/prelude.rs` | EDIT | Re-export `EcpShell` |
| `crates/cintx-oracle/build.rs` | EDIT | **DEPENDS ON ORACLE SOURCE DECISION** — see Open Questions |
| `crates/cintx-oracle/src/fixtures.rs` | EDIT | `build_cu_lanl2dz()` builder (NEW) |
| `crates/cintx-oracle/src/compare.rs` (or new `vendor_ffi.rs`) | EDIT | `extern "C"` decls for 4 vendor symbols (names depend on oracle source decision: PySCF `ECPtype1_*` vs hypothetical libcint `int1e_ecp_*`) |
| `crates/cintx-oracle/tests/safe_api_ecp_parity.rs` | NEW | 4 per-symbol parity tests at atol=1e-12 |
| `crates/cintx-oracle/tests/ecp_libecpint_crosscheck.rs` (if libecpint path chosen) | NEW | Optional secondary oracle |
| **For vendored ECP source: NEW files in `libcint-master/src/`** | NEW | Depends on D-01 decision (see Open Questions) |

## Gradient Variant Decision (ECP-05)

**Recommended: INCLUDE gradient variants in Phase 19, as decided in CONTEXT D-10.**

### Reasoning

1. **PySCF infrastructure already provides them.** `nr_ecp_deriv.c` is a sibling file to `nr_ecp.c`. Whatever path the planner chooses for the oracle source, the gradient variants come in the same vendor delta — there is no separate "prerequisite gradient-layer phase" needed (the speculative trigger in ROADMAP/PROJECT for ECP-05).
2. **Component_rank = 3 is already wired in cintx.** `int3c2e_ip1_*` operates with `component_rank = 3` in the manifest (verified via grep — these rows already exist). The manifest CSV column accepts it, the resolver passes it through unmodified, and `cart_to_sph_1e` is component-rank-agnostic.
3. **The kernel launcher pattern naturally supports it.** Existing `launch_one_electron` handles both scalar and derivative outputs through a single launcher branching on operator name; D-11 extends the same pattern for ECP.
4. **Shared math infrastructure.** Type-1 / Type-2 gradients reuse the same Bessel + radial quadrature modules; the derivative shifts the radial power by ±1 and the angular momentum channels by ±1, but the underlying integrand is the same shape.
5. **Closes issue #11 Task 1 in one phase.** Splitting gradients out would leave a partial-coverage gap for pyscf-rs (`pyscf-gto`'s gradient/hessian engines need `int1e_ecp_ipnuc_*` to land).

### Risks of including gradients

- **Type-2 gradient is materially harder** than Type-1 gradient because the spherical-harmonic angular projector must also be differentiated. Mitigation: PySCF's `nr_ecp_deriv.c` handles this; cintx's CubeCL implementation can match the same recursion. Plan should sequence Type-1 gradient first (one wave), then Type-2 gradient (second wave) within the gradient sub-phase.
- **6 symbols (4 base + 2 gradient) means 6 parity tests** at the Phase 15 unified `atol=1e-12`. CI cost is bounded (Cu/LANL2DZ has ~8-10 shells, Cartesian product yields ~64-100 tuples per test, ×6 tests = ~400-600 tuple evaluations). Within existing CI budget per Phase 17/18 precedent.

### What would justify deferring

The only condition that would justify deferring gradients (out of scope per D-10) is if **Type-2 gradient turns out to require a multi-component sph transform pipeline** that doesn't exist yet — analogous to how Phase 13's F12 derivatives needed multi-component sph transform work that pushed parity tests to Plan 13-04 gap closure. Researcher rates this risk as **MEDIUM**: probable that the existing arity-2 `cart_to_sph_1e` (which handles `component_rank > 1` for the Cartesian buffer with the same transform replicated per axis) is sufficient, but worth a spike during plan-phase before the gradient task is sequenced.

## Validation Architecture (for Nyquist)

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Cargo built-in `#[test]` + `#[cfg(has_vendor_libcint)]` guard (existing repo pattern) |
| Config file | `crates/cintx-oracle/Cargo.toml` `[[test]]` sections + `build.rs` cfg emission |
| Quick run command | `cargo test -p cintx-oracle --test safe_api_ecp_parity -- --nocapture` |
| Full suite command | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu` |
| Phase gate | `oracle_parity_gate` CI matrix runs the test under all 4 feature profiles (Phase 17 D-10 precedent) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ECP-01 | Type-1 (local) projector matches libcint/PySCF byte-identity | parity (vendor FFI) | `cargo test -p cintx-oracle --test safe_api_ecp_parity ecp_type1_sph` | ❌ Wave 0 |
| ECP-01 | Type-1 cart variant | parity | `... ecp_type1_cart` | ❌ Wave 0 |
| ECP-02 | Type-2 (semi-local) projector matches libcint/PySCF | parity | `... ecp_type2_sph` | ❌ Wave 0 |
| ECP-02 | Type-2 cart variant | parity | `... ecp_type2_cart` | ❌ Wave 0 |
| ECP-03 | SessionRequest::evaluate routes ECP through same surface | integration | `... ecp_via_session_request_sph` | ❌ Wave 0 |
| ECP-03 | eval_raw routes ECP through compat | integration | `... ecp_via_eval_raw_sph` | ❌ Wave 0 |
| ECP-04 | Cu/LANL2DZ at atol=1e-12 (primary, vendored libcint/PySCF) | parity | covered by tests above (Cu fixture is the input) | ❌ Wave 0 |
| ECP-04 | Cu/LANL2DZ secondary cross-check (libecpint, non-blocking) | informational | `CINTX_LIBECPINT_ORACLE=1 cargo test -p cintx-oracle --test ecp_libecpint_crosscheck` | ❌ Wave 0 |
| ECP-04 | Gradient variants byte-identity | parity | `... ecp_ipnuc_sph`, `... ecp_ipnuc_cart` | ❌ Wave 0 |
| ECP-05 | Decision recorded in SPEC.md | doc gate | manual review during plan-phase | (already recorded in CONTEXT D-10; SPEC.md is a plan-phase artifact) |

### Property tests (in-tree, no vendor required)

| Property | Test | Justification |
|----------|------|---------------|
| **Hermiticity** | `int1e_ecp_sph[i,j] == int1e_ecp_sph[j,i]` for all pairs | ECP operator is Hermitian; failure indicates kernel bug |
| **Real-valued output** | All elements of `int1e_ecp_{cart,sph}` are `f64::is_finite()` | NaN/Inf indicate Bessel asymptotic blowup or radial quadrature divergence |
| **Translational invariance** | Translating molecule + ECP center by same vector → identical output | Catches center-coordinate sign-bit errors |
| **Type-1-only basis** | A basis with `ang_momentum = -1` (or `EcpType::Type1`) only-shell collapses Type-2 path → output matches Type-1 alone | Validates branch separation |

### Sampling Rate (Nyquist gate)

- **Per task commit:** `cargo test -p cintx-oracle --test safe_api_ecp_parity -- --quick` (single shell pair, runs in < 5s)
- **Per wave merge:** Full ECP test file at atol=1e-12 across Cu/LANL2DZ Cartesian product (~60-100 tuples)
- **Phase gate:** Full Phase 15 unified oracle suite (`oracle_parity_gate` 4-profile matrix) green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `crates/cintx-oracle/tests/safe_api_ecp_parity.rs` — Phase 17/18 pattern, 4 + 2 = 6 per-symbol tests (or 4 if spinor is excluded per D-12)
- [ ] `crates/cintx-oracle/tests/ecp_libecpint_crosscheck.rs` — secondary oracle, `#[ignore]` + `CINTX_LIBECPINT_ORACLE=1` opt-in
- [ ] `crates/cintx-oracle/src/fixtures.rs::build_cu_lanl2dz` — fixture builder (no Cu basis exists today; source LANL2DZ params from PySCF basis library or BasisSetExchange.org)
- [ ] `crates/cintx-oracle/build.rs` — extend `cc::Build` for the new vendored ECP sources (gated on D-01 decision)
- [ ] Vendor FFI decls in `crates/cintx-oracle/src/compare.rs` or sibling — 4 new `extern "C"` (or 8 if both PySCF `ECPtype{1,2}_{cart,sph}` AND a wrapping `int1e_ecp_{cart,sph}` shim are needed)

### Eval dimensions (cross-cutting)

| Dimension | What's measured | Target | Source of truth |
|-----------|-----------------|--------|----------------|
| Byte-identity | abs diff per matrix element vs vendored C | atol=1e-12, rtol=0.0 | Phase 15 unified tolerance |
| Numerical stability | NaN/Inf check post-kernel | 0 occurrences | Property test |
| Algorithmic correctness | Hermiticity, translational invariance | exact (atol=1e-14 for self-symmetry) | Property test |
| Cross-implementation agreement | atol vs libecpint | atol≈1e-9 (informational) | Non-blocking gate |
| Performance regression | Wall-clock per Cu/LANL2DZ Cu-Cu shell tuple | Phase 15 baseline (no baseline today; established this phase) | `criterion` bench (optional) |
| Coverage | Manifest oracle_covered flag | 4 (or 6) entries flipped from `false` to `true` post-parity | `manifest-audit` |

## Open Questions for Planner (RESOLVED 2026-05-12)

### **Q1 (BLOCKING): Oracle source decision — vendor PySCF nr_ecp.c or pivot to libecpint?**

**RESOLVED:** D-01 (revised 2026-05-12) — Option C adopted: vendor PySCF nr_ecp as the primary byte-identity oracle; libecpint added as a non-blocking secondary cross-check (see CONTEXT.md `<decisions>` D-01 + D-02).

**Status:** D-01 as written is unimplementable (libcint 6.1.3 ships no ECP source — VERIFIED).

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Vendor PySCF `nr_ecp.{c,h}` + `nr_ecp_deriv.c`** | Apache-2.0, same author as libcint, drop-in to existing `cc::Build`, matches SC#4 intent (same ecosystem), produces canonical `ECPtype{1,2}_{cart,sph}` symbols. | Requires shimming `np_helper/np_helper.h` and `vhf/fblas.h` (pyscf-internal headers); cintx-oracle FFI shim adapts `ECPtype1_*(ecpbas, necpbas, ...)` signature → cintx's `int1e_ecp_*` operator API. Total vendor surface: ~7000 lines C (one large file + derivative file). |
| **B. Promote libecpint to primary, PySCF as secondary** | C++ (MIT), actively maintained, paper-cited algorithm (Shaw & Hill JCP 147, 074108, 2017), derivatives included, cleaner API (`getIntegrals` shape). | Requires C++ build chain (CMake or manual cc::Build with `-std=c++17`); Rust FFI through `extern "C"` shim layer; byte-identity at atol=1e-12 unlikely due to different normalization conventions (informational atol≈1e-9 realistic). Departs from ROADMAP SC#4 wording. |
| **C. Both A (primary) + B (secondary, informational)** | Maximum oracle coverage; A satisfies SC#4 byte-identity; B catches algorithmic drift via cross-implementation. | Largest vendor surface; longer plan-phase. |
| **D. ROADMAP SC#4 wording amended to "vs PySCF nr_ecp" instead of "vs libcint"** | Honest about reality (libcint upstream has no ECP). | Requires user / project decision to amend the success criterion. |

**Recommendation: Option C** (PySCF primary, libecpint secondary). Replaces chrr/libECP from CONTEXT D-02 with libecpint (the actually-maintained, paper-cited C++ library). Surface this to the user before plan-phase commits, then update SPEC.md.

### Q2: ECP shell representation of "local channel"

**RESOLVED:** D-04 — EcpShell carries an `ecp_type` marker via `EcpChannel { Local, Projected(u8) }`; the local Type-1 channel and the semi-local Type-2 projector channels are both representable (option (b) recommendation accepted in CONTEXT.md).

PySCF encodes the local (Type-1) ECP channel by setting `ANG_OF = -1` on the ecpbas row. cintx's typed `EcpShell::ang_momentum: u8` can't hold -1.

Options:
- (a) Use `EcpShell::ang_momentum: i8` (or `i16`).
- (b) Add `enum EcpChannel { Local, Projected(u8) }` and remove `ang_momentum` raw field — type-safer.
- (c) Add `enum EcpType { Type1, Type2 }` marker per CONTEXT D-04 and keep `ang_momentum: u8` (with Type-1 having `ang_momentum = 0` and Type interpretation overridden by the enum).

**Recommendation: option (b)** — cleanest typed API; aligns with PROJECT.md "type-safe first" priority.

### Q3: ecpbas packing convention in `eval_raw`

**RESOLVED:** D-05 (revised) — match PySCF `nr_ecp.h` verbatim: `AS_ECPBAS_OFFSET = 18`, `AS_NECPBAS = 19`. ecpbas is packed after bas in the same i32 slab (see CONTEXT.md `<decisions>` D-05).

PySCF's convention: ecpbas is packed contiguously **after** bas in the same i32 slab; the env array stores the start pointer at `env[AS_ECPBAS_OFFSET]` and the count at `env[AS_NECPBAS]`.

Alternative: cintx-compat could accept a separate `ecpbas: &[i32]` slab — more typed but breaks the "single slab" convention of all other families.

**Recommendation:** match PySCF — pack after bas, store offset+count in env. Researcher confirms during vendor read of `ECPtype1_cart` source.

### Q4: Manifest row count — 4 or 6?

**RESOLVED:** D-09 — 4 manifest rows this phase (cart/sph × {ecp, ecp_ipnuc}); D-12 explicitly defers spinor.

CONTEXT D-12 defers spinor rows. The 4 base + 2 gradient = 6 number from CONTEXT D-10 includes the SO/spinor count. Per D-12, the manifest delta is **4 rows** (cart + sph × {ecp, ecp_ipnuc}), with spinor rows added in a follow-up phase.

**Recommendation:** 4 rows this phase. Document spinor rows as deferred in the manifest CSV header comment.

### Q5: `canonical_family = "ecp"` vs `"1e"`

**RESOLVED:** D-09 — `canonical_family = "ecp"` (parallels Phase 13's `"f12"` precedent).

CONTEXT D-09 chose `"ecp"` as a separate canonical_family (matching Phase 13's `"f12"` precedent). Alternative: `canonical_family = "1e"` with operator_name distinguishing (`int1e_ecp_*` vs `int1e_*`), which would route through the existing `launch_one_electron` launcher — but that conflates two algorithmically distinct paths and complicates the existing launcher.

**Recommendation:** keep `"ecp"` per D-09. Phase 13 F12 precedent is the right pattern.

### Q6: Bessel function strategy (`math/bessel.rs`)

**RESOLVED:** D-07 + Phase 13 precedent — tabulated `K_TAB_*` strategy (option (b)) via `include_bytes!` + `bytemuck::AlignedBytes` (matches PySCF byte-identity at atol=1e-12).

Three viable strategies (must pick ONE explicitly):

| Strategy | Pro | Con |
|----------|-----|-----|
| (a) Pure Taylor + asymptotic split | No tables; smaller binary. | Slow for moderate argument ($x \in [1, 16]$). |
| (b) Tabulated $i_l(x)$ following PySCF's `K_TAB_*` constants | Matches reference numerics; faster. | Adds binary table (~75KB for 400 × 24 × 8 bytes). |
| (c) Downward (Miller's) recurrence with normalization | Numerically stable for arbitrary $l$; no tables. | More complex `#[cube]` code; cond_br risks per Phase 8 P02 incident. |

**Recommendation:** strategy (b) — matches PySCF byte-identity, follows Phase 13 binary-table precedent (`roots_xw.dat`). Document the `K_TAB_*` constants in `bessel.rs` rustdoc.

### Q7: Cu/LANL2DZ basis source

**RESOLVED:** Claude's Discretion in CONTEXT.md — basissetexchange.org JSON export committed at `crates/cintx-oracle/data/cu_lanl2dz.json`; source URL cited in fixture rustdoc.

Three viable sources:

| Source | Authority | Ergonomics |
|--------|-----------|------------|
| **PySCF basis library (`pyscf.gto.basis.load`)** | Same author as libcint/PySCF; canonical | Requires Python extraction step |
| **basissetexchange.org** | Community-maintained gold standard | Web download; copy-paste from JSON |
| **Hay & Wadt 1985 JCP papers** | Primary source | Manual transcription error-prone |

**Recommendation:** basissetexchange.org JSON export (`Cu LANL2DZ`); commit the JSON as a fixture data file at `crates/cintx-oracle/data/cu_lanl2dz.json` and parse in `build_cu_lanl2dz()`. Cite source URL in fixture rustdoc.

### Q8: Test fixture coverage scope

**RESOLVED:** Claude's Discretion + Deferred — Cu/LANL2DZ full Cartesian product only this phase; lighter-atom (Na/SBKJC, K/CRENBL) validation deferred to `.planning/spikes/ecp-fixture-validation.md` per CONTEXT.md `<deferred>`.

- Cu/LANL2DZ has 10 shells (1s, 1p, 1d, 1s, 1p, 1d, 1s, 1p, 1d, 1f-projector roughly). Cartesian product = 100 tuples.
- Light-atom fallback (Na/SBKJC, K/CRENBL — only s+p projectors) would prove Type-1+Type-2 correctness with smaller coverage before Cu.
- Phase 18 precedent: H2O STO-3G has 5 shells, 25 tuples per test — comparable.

**Recommendation:** Cu/LANL2DZ only this phase per CONTEXT decision; defer lighter-atom validation to `.planning/spikes/ecp-fixture-validation.md` follow-up.

### Q9: libcint header lacks `AS_ECPBAS_OFFSET` / `AS_NECPBAS`

**RESOLVED:** Revised `<canonical_refs>` "Vendored ECP source integration" — do NOT modify `libcint-master/include/cint.h.in`; the ECP slot constants live in `cintx-compat::raw` and mirror PySCF `nr_ecp.h`.

The vendored `libcint-master/include/cint.h.in` lacks the ECP env-slot constants. Even if Option A (PySCF vendor) is chosen, `cint.h.in` won't gain these constants automatically — they live in `nr_ecp.h` which is the new vendored file.

**Recommendation:** Do not modify `cint.h.in` (preserves upstream sync). Instead, vendor `nr_ecp.h` as a sibling header (e.g., `libcint-master/src/pyscf_nr_ecp/nr_ecp.h`) and adjust the cintx-compat::raw constant definitions to point to the PySCF source.

## Project Constraints (from CLAUDE.md)

| Constraint | Compliance |
|------------|------------|
| CubeCL as primary compute backend | ✓ D-07/D-08 use `#[cube]` kernels. |
| Safe Rust API first, raw second, C ABI third | ✓ `BasisSet::ecp_shells()` is typed-first; raw exposes `EcpBasArray`; C ABI deferred. |
| `thiserror` v2 for public library errors | ✓ New `FacadeError::MissingEcpBasis` variant. |
| `anyhow` for CLI/oracle/xtask | ✓ Oracle FFI shims use `anyhow` per existing pattern. |
| Rust 1.94.0 pinned | ✓ No new toolchain requirement. |
| `cargo --locked` in CI | ✓ No new dependencies; lockfile unchanged unless `bytemuck` AlignedBytes (already present) suffices. |
| Output to `/mnt/data` | ✓ Existing `cintx-oracle` artifact paths reused. |
| `cubecl` pinned at 0.10.0 | ✓ No backend API change. |

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| ECP via classical quadrature (e.g., Slater-quadrature-style) | Recursive-code-generation + Gauss-Chebyshev (libecpint) | Shaw & Hill 2017 (JCP 147, 074108) | ~40× speedup for high-l ECPs |
| Upward Bessel recurrence | Downward (Miller's) or tabulated | Long-standing numerical-analysis practice | Stable for $l > x$ |
| Per-call radial quadrature node generation | Pre-tabulated at build time | PySCF's `K_TAB_*` design | One-time table cost vs per-call |
| Libcint integrated ECP | Externally maintained ECP libraries (libecpint, PySCF nr_ecp) | Long-standing — ECP was never folded into libcint proper | The libcint repo genuinely doesn't have `ecp.c` |

**Deprecated/outdated:**
- chrr/libECP (effectively unmaintained; sparse docs; no formal paper) — superseded by libecpint for the same role.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Manifest lock regen is via `cargo run -p xtask -- manifest-audit --update` | cintx-ops integration | Plan task uses wrong command; quick fix once xtask is read. |
| A2 | ecpbas packing convention is "after bas in same i32 slab + env pointer" (PySCF style) | Raw API path | Wrong layout breaks `eval_raw`; verifiable by reading `ECPtype1_cart` source during vendor pull. |
| A3 | `cart_to_sph_1e` handles component_rank=3 transparently | Kernel launcher | If false, plan needs a spike for multi-component sph transform (Phase 13-04 precedent). MEDIUM risk. |
| A4 | PySCF's `np_helper/np_helper.h` and `vhf/fblas.h` includes are non-essential and can be shimmed | Oracle vendor | If they bring runtime dependencies (numpy, BLAS link), the vendor delta balloons. Probable shim: empty header + extern decls. |
| A5 | Type-1 local channel is encoded with `ANG_OF = -1` in PySCF ecpbas | Typed core (`EcpShell`) | Convention is decoder dependent — verify by reading PySCF `ecp.py` `_ecpbas` mapping. |
| A6 | Component rank for `int1e_ecp_ipnuc_*` is 3 (one per Cartesian axis) following `int3c2e_ip1_*` convention | Manifest CSV | Standard libcint convention; verify against `int3c2e_ip1_*` manifest row. |
| A7 | `OperatorId` constants are integers assigned monotonically; next free ID is in the high-20s | cintx-core integration | Verify by reading `crates/cintx-core/src/operator.rs`. |
| A8 | Cu has 8-10 shells in LANL2DZ; Cartesian product is CI-budget-acceptable (~100 tuples × 6 tests = ~600 evaluations) | Test fixture coverage | If basis is larger, budget tradeoff via sub-sampling. |
| A9 | The "JCC 2017" reference in CONTEXT D-02 / ROADMAP is actually the JCP 2017 Shaw & Hill paper for libecpint (NOT chrr/libECP) | Oracle source decision | Recommend ROADMAP citation correction to "Shaw & Hill, JCP 147, 074108 (2017)" + library = libecpint. |
| A10 | `cubecl-cpu 0.10.0` accepts `#[cube]` functions with binary-table-backed const arrays via `include_bytes!` + `bytemuck::AlignedBytes` | Math infra (`bessel.rs`) | Phase 13 P02 precedent confirms this works for `roots_xw.dat`; Bessel tables follow same pattern. HIGH confidence. |

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| `cc` crate + system C compiler | Vendored libcint + new vendored ECP source | ✓ (existing) | workspace 1.2.x | — |
| `bindgen` 0.71.1 | Header generation (extending if FFI surface grows) | ✓ (existing) | 0.71.1 | — |
| `cubecl` 0.10.0 | `#[cube]` kernels for `bessel.rs`, `radial_quadrature.rs`, `ecp.rs` | ✓ (existing) | 0.10.0 | — |
| `bytemuck` (for AlignedBytes) | Binary Bessel tables in `math/bessel.rs` (Q6 strategy b) | ✓ (existing) | from Cargo.lock | — |
| C++ compiler (g++/clang++) | Only if Option B (libecpint) chosen | ✓ | system | shim layer or vendor C-only path (Option A) |
| `cmake` (≥ 3.12) | Only if libecpint built via its native CMake | ✓ | system | direct `cc::Build` |
| ROCm runtime | Optional ECP smoke test on AMD (Phase 16 precedent) | ✓ | system per Phase 16 | CPU backend baseline |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:** None.

## Sources

### Primary (HIGH confidence)
- VERIFIED: `crates/cintx-cubecl/src/kernels/mod.rs` lines 1-100 (read in research)
- VERIFIED: `crates/cintx-core/src/basis.rs`, `shell.rs` (read in research)
- VERIFIED: `crates/cintx-compat/src/raw.rs` slot constants (grep)
- VERIFIED: `crates/cintx-oracle/build.rs` (read in research)
- VERIFIED: `crates/cintx-oracle/src/fixtures.rs` (read top 140 lines)
- VERIFIED: `crates/cintx-oracle/tests/safe_api_arity2_parity.rs` (read top 80 lines)
- VERIFIED: `crates/cintx-rs/src/api.rs` (read top 100 lines)
- VERIFIED: `libcint-master/src/` directory contents (ls — confirmed no `ecp.c`)
- VERIFIED: `libcint-master/include/cint.h.in` (read — confirmed no `AS_ECPBAS_*` constants)
- VERIFIED: `libcint-master/CMakeLists.txt` (libcint version = 6.1.3)
- VERIFIED: `crates/cintx-ops/src/generated/api_manifest.csv` (133 lines; 0 ECP rows)

### Secondary (HIGH confidence, external official sources)
- CITED: PySCF `nr_ecp.h` slot constants and tunables — github.com/pyscf/pyscf/blob/master/pyscf/lib/gto/nr_ecp.h (via WebFetch 2026-05-12)
- CITED: PySCF `pyscf/gto/ecp.py` C function names — pyscf.org/_modules/pyscf/gto/ecp.html (via WebFetch)
- CITED: PySCF `pyscf/lib/gto/CMakeLists.txt` shows `nr_ecp.c` + `nr_ecp_deriv.c` compile into `libcgto.so` (via WebFetch)
- CITED: libcint v6.1.3 source list — github.com/sunqm/libcint/tree/v6.1.3/src (via WebFetch; confirms NO ECP files)
- CITED: libecpint — github.com/robashaw/libecpint (MIT, active, JCP 2017 algorithm); paper: Shaw & Hill, JCP 147, 074108 (2017)
- CITED: chrr/libECP — github.com/chrr/libECP (BSD-2; sparse docs; not the "JCC 2017" library)

### Tertiary (verified secondary search results)
- WebSearch: Shaw & Hill 2017 JCP paper widely cited as ECP integral state-of-the-art
- WebSearch: McMurchie-Davidson 1981 J. Comput. Phys. paper is the historical foundation for analytical ECP-like integrals

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all integration points verified by codebase read.
- Architecture: HIGH — Phase 13 / Phase 17 / Phase 18 precedents directly applicable.
- Algorithm (Type-1): HIGH for structure; MEDIUM for the exact radial integration shape (depends on which oracle source is chosen).
- Algorithm (Type-2): HIGH for the structure (Bessel-modulated radial integrals); MEDIUM for the precise quadrature node counts (configurable per `nr_ecp.h` defaults).
- Pitfalls: HIGH — Bessel recurrence instability + cond_br MLIR limitation are well-documented in the same module's prior incidents.
- Oracle source: HIGH for the finding (libcint has no ECP — verified); BLOCKING for the planner (D-01 needs revision).

**Research date:** 2026-05-12
**Valid until:** 2026-06-12 (30 days — stable ecosystem; verify libecpint version + PySCF nr_ecp.h hash if planning slips beyond)

## RESEARCH COMPLETE
