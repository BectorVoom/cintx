# Gradient / Derivative Integral Family Gap Closure — cintx

**Created:** 2026-08-22
**Milestone:** v1.4 — Full libcint 6.1.3 Family Parity
**Relates to:** PARITY-01 (Phase 31 apex gate), Phases 21 / 23 / 25 / 27–30
**Audience:** an execution agent that follows instructions literally and does NOT infer.

---

## 0. FINDING FIRST — the premise most callers arrive with is stale

> "cintx is missing the 6 core gradient integral families (`int2e_ip1`,
> `int1e_ip{ovlp,kin,nuc,rinv}`, `ECPscalar_iprinv`) and the `with_rinv_at_nucleus`
> origin shift."

**That statement was true on 2026-06-01. It is FALSE today.** It originates from the
downstream `pyscf_rs` Phase-7 audit (`pyscf_rs/.planning/phases/07-gradients-geomopt/07-01-PLAN.md:46-48`),
which has not been refreshed since cintx Phases 21–25 landed.

Verified against the working tree on 2026-08-22:

| Family | Status | Evidence |
|---|---|---|
| `int1e_ipovlp` / `ipkin` / `ipnuc` / `iprinv` | **SHIPPED** cart+sph+spinor, `oracle_covered=true` | `crates/cintx-cubecl/src/kernels/one_electron.rs:9110-9114` dispatch; vendor tests at `:11825`, `:11858`, `:12348`, `:12377` |
| `int2e_ip1` / `int2e_ip2` | **SHIPPED** cart+sph | `crates/cintx-cubecl/src/kernels/two_electron.rs:3704`, `:3721` |
| `int3c2e_ip1` / `ip2` | **SHIPPED** cart+sph+spinor | `crates/cintx-cubecl/src/kernels/center_3c2e.rs:880`, `:1609` |
| `int2c2e_ip1` / `ip2` | **SHIPPED** cart+sph | `crates/cintx-cubecl/src/kernels/center_2c2e.rs:1052-1057` |
| `ECPscalar_ipnuc` (`int1e_ecp_ipnuc`) | **SHIPPED** cart+sph | `crates/cintx-cubecl/src/kernels/ecp.rs:1591`, `:1975` |
| `ECPscalar_iprinv` (`int1e_ecp_iprinv`) | **SHIPPED** cart+sph (Phase 21-07 / D-09) | `crates/cintx-cubecl/src/kernels/ecp.rs:668`, `:1984` |
| rinv origin shift | **SHIPPED** | `crates/cintx-rs/src/builder.rs:102` `Builder::with_rinv_origin([f64;3])` |
| Rank-9 Hessian 1e/2e/2c2e/3c2e | **SHIPPED** | `one_electron.rs:9121-9131`, `two_electron.rs:3740-3746`, `center_2c2e.rs:1058`, `center_3c2e.rs:3079` |

**Two actions follow from this finding, and they are both in scope for this plan:**

1. **Wave 0** files a correction upstream so `pyscf_rs` stops gating on a closed blocker.
2. **Waves 1–5** close the gap that *is* real — and it is a different, larger gap than
   the one people think they have.

---

## 1. THE REAL GAP

### 1.1 Root cause

`crates/cintx-ops/src/generated/api_manifest.rs` is generated from
`libcint-master/include/cint_funcs.h`.

**33 derivative symbols in libcint 6.1.3 are exported by the built library
(via the `ALL_CINT(...)` macro in `src/autocode/*.c`) but are NEVER declared in
`cint_funcs.h`.** They are therefore invisible to the manifest generator, absent from
`MANIFEST_ENTRIES` (381 entries today), and — critically — **invisible to the Phase-31
PARITY-01 gate**, which compares against `cint_funcs.h` + the current supplemental headers.

PySCF reaches these symbols by dynamic name lookup
(`pyscf/gto/moleintor.py` → `getattr(libcgto, intor_name)`), so from a consumer's point
of view they are ordinary public API. From cintx's point of view they do not exist.

**Verification command (run this first — it is the plan's own premise check):**
```bash
cd /home/user/Documents/workspace/cintx
grep -c "int1e_iprinvip" libcint-master/include/cint_funcs.h   # → 0
grep -rn "ALL_CINT(int1e_iprinvip)" libcint-master/src/autocode/  # → hess.c
grep -c 'symbol_name: "int1e_iprinvip' crates/cintx-ops/src/generated/api_manifest.rs  # → 0
```

### 1.2 Gap A — 33 symbols absent from the manifest entirely

All rows below were extracted mechanically from `libcint-master/src/autocode/*.c`.
`ng[]` is libcint's `FINT ng[]` array verbatim; its **last element is the tensor
component rank**, and elements 0–3 are the per-center angular-momentum headroom
increments `(i, j, k, l)` the G-tensor must be built with.
`type` is the trailing `CINT1e_drv` argument: **0 = overlap engine (no Rys),
1 = rinv engine (single-center Rys, reads `env[PTR_RINV_ORIG]`),
2 = nuclear engine (atom-summed Rys)**.
`e1=4` in `ng[5]` marks a σ (spin-operator) family that must ride the existing
`sigma_p.rs` four-block (`gc_x/gc_y/gc_z/gc_1`) assembler.

#### Tier 1 — molecular Hessian (`pyscf/hessian/rhf.py`, `uhf.py`) — HIGHEST PRIORITY

| Symbol | libcint file | `ng[]` | rank | drv / c2s | type |
|---|---|---|---:|---|---|
| `int1e_ipipr` | `hess.c` | `{2,1,0,0,3,1,1,27}` | 27 | `CINT1e_drv` / `c2s_cart_1e` | 0 |
| `int1e_iprinvip` | `hess.c` | `{1,1,0,0,2,1,0,9}` | 9 | `CINT1e_drv` / `c2s_cart_1e` | 1 |
| `int2e_ipvip1ipvip2` | `hess.c` | `{1,1,1,1,4,1,1,81}` | 81 | `CINT2e_drv` / `c2s_cart_2e1` | — |

#### Tier 2 — DF Hessian (`pyscf/df/hessian/rhf.py`, `uhf.py`)

| Symbol | libcint file | `ng[]` | rank | drv / c2s | type |
|---|---|---|---:|---|---|
| `int2c2e_ip1ip2` | `int3c2e.c` | `{1,0,1,0,2,1,1,9}` | 9 | `CINT2c2e_drv` / `c2s_cart_1e` | — |
| `int3c2e_ip1ip2` | `int3c2e.c` | `{1,0,1,0,2,1,1,9}` | 9 | `CINT3c2e_drv` / `c2s_cart_3c2e1` | 0 |
| `int3c2e_ipvip1` | `int3c2e.c` | `{1,1,0,0,2,1,1,9}` | 9 | `CINT3c2e_drv` / `c2s_cart_3c2e1` | 0 |

#### Tier 3 — scalar X2C gradient & Hessian (`pyscf/x2c/sfx2c1e_grad.py`, `sfx2c1e_hess.py`)

| Symbol | libcint file | `ng[]` | rank | type | note |
|---|---|---|---:|---|---|
| `int1e_ippnucp` | `grad1.c` | `{2,1,0,0,3,1,0,3}` | 3 | 2 | `⟨∇p·V·p⟩` gradient |
| `int1e_ipprinvp` | `grad1.c` | `{2,1,0,0,3,1,0,3}` | 3 | 1 | per-nucleus form |
| `int1e_ippnucpip` | `hess.c` | `{2,2,0,0,4,1,0,9}` | 9 | 2 | both-side |
| `int1e_ipprinvpip` | `hess.c` | `{2,2,0,0,4,1,0,9}` | 9 | 1 | both-side |
| `int1e_ipippnucp` | `hess.c` | `{3,1,0,0,4,1,0,9}` | 9 | 2 | bra-only ∇∇ |
| `int1e_ipipprinvp` | `hess.c` | `{3,1,0,0,4,1,0,9}` | 9 | 1 | bra-only ∇∇ |

#### Tier 4 — 4-component Dirac gradients (`pyscf/grad/dhf.py`) — σ families

| Symbol | libcint file | `ng[]` | rank | drv | type |
|---|---|---|---:|---|---|
| `int1e_ipspnucsp` | `grad1.c` | `{2,1,0,0,3,**4**,0,3}` | 3 | `CINT1e_drv` | 2 |
| `int1e_ipsprinvsp` | `grad1.c` | `{2,1,0,0,3,**4**,0,3}` | 3 | `CINT1e_drv` | 1 |
| `int2e_ipspsp1` | `grad2.c` | `{2,1,0,0,3,**4**,1,3}` | 3 | `CINT2e_drv` | — |
| `int2e_ip1spsp2` | `grad2.c` | `{1,0,1,1,3,1,**4**,3}` | 3 | `CINT2e_drv` | — |
| `int2e_ipspsp1spsp2` | `grad2.c` | `{2,1,1,1,5,**4**,**4**,3}` | 3 | `CINT2e_drv` | — |

#### Tier 5 — declared in `pyscf/gto/moleintor.py` but with no live PySCF consumer

| Symbol | libcint file | `ng[]` | rank | type |
|---|---|---|---:|---|
| `int1e_ipipspnucsp` | `hess.c` | `{3,1,0,0,4,4,0,9}` | 9 | 2 |
| `int1e_ipipsprinvsp` | `hess.c` | `{3,1,0,0,4,4,0,9}` | 9 | 1 |
| `int1e_ipspnucspip` | `hess.c` | `{2,2,0,0,4,4,0,9}` | 9 | 2 |
| `int1e_ipsprinvspip` | `hess.c` | `{2,2,0,0,4,4,0,9}` | 9 | 1 |
| `int2e_ipsrsr1` | `grad2.c` | `{2,1,0,0,3,4,1,3}` | 3 | — |
| `int2e_ip1srsr2` | `grad2.c` | `{1,0,1,1,3,1,4,3}` | 3 | — |
| `int2e_ipsrsr1srsr2` | `grad2.c` | `{2,1,1,1,5,4,4,3}` | 3 | — |
| `int2e_ip1v_r1` | `intor2.c` | `{1,2,0,0,2,1,1,9}` | 9 | — |
| `int2e_ip1v_rc1` | `intor2.c` | `{1,2,0,0,2,1,1,9}` | 9 | — |
| `int2e_ipvg1_xp1` | `intor2.c` | `{2,1,0,0,3,1,1,9}` | 9 | — |
| `int2e_ipvg2_xp1` | `intor2.c` | `{1,1,1,0,3,1,1,9}` | 9 | — |
| `int3c2e_ipspsp1` | `int3c2e.c` | `{2,1,0,0,3,4,1,3}` | 3 | 0 |

#### Tier 6 — libcint-only (LRESC property integrals; no PySCF consumer at all)

| Symbol | libcint file | `ng[]` | rank | type |
|---|---|---|---:|---|
| `int1e_iprinvr` | `lresc.c` | `{1,1,0,0,2,1,0,9}` | 9 | 1 |
| `int1e_iprip` | `hess.c` | `{1,2,0,0,3,1,1,27}` | 27 | 0 |
| `int1e_iprinviprip` | `lresc.c` | `{1,3,0,0,4,1,0,81}` | 81 | 1 |
| `int1e_ipiprinvrip` | `lresc.c` | `{2,2,0,0,4,1,0,81}` | 81 | 1 |

### 1.3 Gap B — 11 declared rows with `oracle_covered: false`

These ARE in the manifest but have never been proven byte-identical to vendored libcint.
Every one is a **spinor** form. Until they flip to `true`, they must be treated as
unimplemented, not as shipped.

| Symbol | rank | current state |
|---|---:|---|
| `int1e_drinv_spinor` | 3 | declared, unverified |
| `int2c2e_ip1_spinor` | 3 | declared, unverified |
| `int2c2e_ip2_spinor` | 3 | declared, unverified |
| `int2e_ip1_spinor` | 3 | declared, unverified |
| `int2e_ip2_spinor` | 3 | declared, unverified |
| `int2e_ipip1_spinor` | 9 | declared, unverified |
| `int2e_ipvip1_spinor` | 9 | declared, unverified |
| `int2e_ip1ip2_spinor` | 9 | declared, unverified |
| `int3c1e_ip1_spinor` | 3 | declared, unverified |
| `int3c1e_iprinv_spinor` | 3 | declared, unverified |
| `int1e_ecp_iprinv_spinor` | 3 | **hard-rejected** at `kernels/ecp.rs:2047` with `UnsupportedApi` |

### 1.3b Gap B′ — moment-weighted 3-center / 1e families (**BLOCKS pyscf_rs PBC Phase 10**)

Added 2026-08-22 after auditing the `pyscf_rs` v2.0 PBC milestone as a consumer.
These are the **earliest** cintx blocker for that consumer — they gate GTH
pseudopotential evaluation (PBC Phase 10), which every periodic SCF/DFT run needs.
They are NOT gradient families, but they arrive through the same manifest defect
(declared, `oracle_covered: false`) and the same fix mechanism, so they belong in
this plan rather than a separate one.

| Symbol | rank | forms declared | `oracle_covered` | Dispatch arm present? | PySCF consumer |
|---|---:|---|---|---|---|
| `int1e_r2_origi` | 1 | sph | **false** | **NO** (`one_electron.rs` has no `r2_origi` arm) | `pbc/gto/pseudo/pp_int.py:626` `_int_vnl` |
| `int1e_r4_origi` | 1 | sph | **false** | **NO** | `pp_int.py:626` |
| `int1e_r2_origi_ip2` | 3 | sph | **false** | **NO** | `pp_int.py:454` `vppnl_nuc_grad` |
| `int1e_r4_origi_ip2` | 3 | sph | **false** | **NO** | `pp_int.py:454` |
| `int3c1e_r2_origk` | 1 | sph | **false** | **NO** (`center_3c1e.rs:1469` dispatches only `ip1`/`iprinv`) | `pp_int.py:150`, `df/aft.py:335` |
| `int3c1e_r4_origk` | 1 | sph | **false** | **NO** | `pp_int.py:150`, `aft.py:335` |
| `int3c1e_r6_origk` | 1 | sph | **false** | **NO** | `pp_int.py:150`, `aft.py:335` |
| `int3c1e_ip1_r2_origk` | 3 | sph | **false** | **NO** | `pp_int.py:187` `vpploc_part2_nuc_grad` |
| `int3c1e_ip1_r4_origk` | 3 | sph | **false** | **NO** | `pp_int.py:187` |
| `int3c1e_ip1_r6_origk` | 3 | sph | **false** | **NO** | `pp_int.py:187` |

**These are `⟨i| r_C^{2n} |j k⟩`-weighted forms of families cintx already ships.**
`int3c1e_r2_origk` is `int3c1e` with an `|r − R_k|²` weight on the third center;
`int1e_r2_origi` is `int1e_ovlp` with an `|r − R_i|²` weight on the bra center.
The `_ip1_` / `_ip2` variants add one ∇ on the named center. Engine class is the same
as the unweighted parent (3c1e overlap, 1e overlap) — the weight enters as extra
angular-momentum headroom on the weighted center plus a polynomial factor in the
`gout` contraction. Read the vendored `ng[]` per RULE 1 before implementing:
```bash
grep -A6 "CACHE_SIZE_T int3c1e_r2_origk_sph" libcint-master/src/autocode/int3c1e.c
grep -A6 "CACHE_SIZE_T int1e_r2_origi_sph"   libcint-master/src/autocode/intor*.c
```

**⚠️ FAIL-OPEN HAZARD — verify this before anything else in Wave 0.**
`center_3c1e.rs:1469` matches `operator_name` against `"ip1"` / `"iprinv"` and
**falls through to the plain 3-center overlap path on `_ => {}`**. `one_electron.rs`
appears to do the same for `r2_origi`. If no earlier preflight rejects these operators,
a caller requesting `int3c1e_r2_origk_sph` receives the **unweighted `int3c1e`**
silently — a wrong number, not an error. I did not trace `cintx-rs/src/api.rs:1122`
far enough to confirm whether a preflight catches it.

**W0-05 (NEW, blocking) — prove or disprove the fall-through:**
```rust
// crates/cintx-rs/tests/moment_weighted_falls_through.rs
#[test]
fn int3c1e_r2_origk_must_not_silently_return_plain_int3c1e() {
    let a = eval("int3c1e_r2_origk_sph", &fx, shls);   // may Err — that is FINE
    let b = eval("int3c1e_sph",          &fx, shls);
    match a {
        Err(_) => {}                                    // fail-closed: acceptable
        Ok(v)  => assert_ne!(v, b,
            "FAIL-OPEN: int3c1e_r2_origk returned the unweighted int3c1e result"),
    }
}
```
Run the same shape for all ten Gap-B′ symbols.
- If they **Err** → no data-corruption bug; they are simply unimplemented. Proceed.
- If they **return the unweighted parent** → this is a **SHOWSTOPPER-severity
  correctness bug affecting shipped API**, not a parity gap. Stop this plan, land a
  fail-closed rejection for every manifest row whose `operator_name` has no dispatch
  arm (a generic guard, not ten special cases), release it, *then* resume.

**Generic guard to land either way (W0-06):** in each family launcher, replace the
`_ => {}` fall-through with an explicit allowlist of operator names that the scalar
path is *known* to serve, and return
`UnsupportedApi { requested: format!("{family}/{operator_name} has no kernel") }`
otherwise. This is the structural fix for the entire class of defect and it costs
one match arm per family.

### 1.4 Gap C — the parity gate itself

`PARITY-01` (Phase 31 success criterion 4) compares the implemented set against
`cint_funcs.h` **plus the current supplemental headers**. Because the 33 Gap-A symbols
are in neither, **the gate can go green while the gap is open.** Closing Gap C means
extending the parity gate's reference set to the union of `cint_funcs.h` and every
`ALL_CINT(...)`-exported symbol in `src/autocode/*.c`, so a future omission of this
shape is mechanically impossible.

---

## 2. HOW TO EXECUTE THIS PLAN

### 2.1 Standing rules

**RULE 1 — Read the vendored C first, always.** Every family names a libcint file and
a `gout` function. Before writing Rust:
```bash
grep -n "CINTgout1e_<symbol>\|CINTgout2e_<symbol>" libcint-master/src/autocode/<file>.c
sed -n '<start>,<end>p' libcint-master/src/autocode/<file>.c
```
Transcribe the `gout` contraction term by term. Do not re-derive it.

**RULE 2 — Reuse the engine, add only the `gout`.** Every Gap-A family rides an engine
cintx already ships. The new code per family is: one `gout` contraction + one dispatch
arm + one manifest row + one vendor test. If you find yourself writing a new G-tensor
builder, stop — you have picked the wrong engine class (see §3).

**RULE 3 — CubeCL discipline.** Kernels are `#[cube(launch_unchecked)]` and generic over
the device float. Read `/home/user/Documents/workspace/cubecl_manual/manual/manual/Cubecl/INDEX.md`
before writing one. On **any** cubecl build/link/feature error, read
`docs/cubecl_error_guideline.md` before touching the code — blind fixes are a protocol
violation (AGENTS.md).

**RULE 4 — Never claim coverage you have not proven.** `oracle_covered` flips to `true`
only when a non-`#[ignore]`d `vendor_*` test passes at `atol=1e-12` against vendored
libcint under `CINTX_ORACLE_BUILD_VENDOR=1`. A family that runs and produces plausible
numbers is NOT covered. Phase 30-01d is the precedent: a 0.5% uniform residual kept
`oracle_covered=false` and the gate `#[ignore]`d.

**RULE 5 — Symbol-name dispatch only.** Detect a family by its manifest
`operator_name` / `symbol_name`, never by a positional `OperatorId` integer literal
(Pitfall 6). Manifest positions shift whenever entries are added — and this plan adds 33.

**RULE 6 — Tests in separate files.** No `mod tests` at the bottom of a production
source file. Existing in-module `#[cfg(test)]` blocks in `one_electron.rs` etc. are
legacy; new tests go in `crates/cintx-oracle/tests/` or `crates/cintx-cubecl/tests/`.

**RULE 7 — Adding a manifest entry is a three-file change.** `api_manifest.csv` →
regenerate `api_manifest.rs` → verify `cargo test -p cintx-ops` manifest-agreement
tests still pass. Never hand-edit `api_manifest.rs`; it carries
`// Generated manifest; do not edit.`

### 2.2 Definition of done for one family

1. Manifest row exists for every physical representation, with the correct
   `component_rank` and `canonical_family`.
2. Dispatch arm exists in the owning family launcher, keyed on `operator_name`.
3. `gout` contraction transcribed from the vendored C and unit-checked.
4. `extern` decl added to the generated supplemental header **and** the bindgen
   `allowlist_function` regex in `crates/cintx-oracle/build.rs`.
5. A `vendor_<symbol>_<rep>` test passes at `atol=1e-12`, not `#[ignore]`d.
6. `oracle_covered` flipped to `true`; `cargo run -p xtask -- manifest-audit` green.

---

## 3. ENGINE-CLASS MAP — which existing cintx code each family rides

This is the load-bearing table. It says, for every Gap-A family, exactly which file to
open and which existing machinery to extend.

| Engine class | Selector | cintx owner | Gap-A families |
|---|---|---|---|
| **1e overlap-deriv** (no Rys) | `CINT1e_drv(..., 0)` | `kernels/one_electron.rs` — the `is_ipovlp`/`is_ipipovlp` path | `int1e_ipipr` (27), `int1e_iprip` (27) |
| **1e rinv-Rys** (single center, `env[PTR_RINV_ORIG]`) | `CINT1e_drv(..., 1)` | `kernels/one_electron.rs` — the `is_iprinv`/`is_ipiprinv` path (origin resolution at `:10673`) | `int1e_iprinvip` (9), `int1e_ipprinvp` (3), `int1e_ipprinvpip` (9), `int1e_ipipprinvp` (9), `int1e_iprinvr` (9), `int1e_iprinviprip` (81), `int1e_ipiprinvrip` (81) |
| **1e nuclear-Rys** (atom-summed) | `CINT1e_drv(..., 2)` | `kernels/one_electron.rs` — the `is_ipnuc`/`is_ipipnuc` path | `int1e_ippnucp` (3), `int1e_ippnucpip` (9), `int1e_ipippnucp` (9) |
| **1e σ-Rys** (`ng[5]==4`, four `gc_*` blocks) | `CINT1e_drv(..., 1\|2)` + `e1=4` | `kernels/sigma_1e_nuc.rs` + `kernels/sigma_p.rs` assembler | `int1e_ipspnucsp` (3), `int1e_ipsprinvsp` (3), `int1e_ipipspnucsp` (9), `int1e_ipipsprinvsp` (9), `int1e_ipspnucspip` (9), `int1e_ipsprinvspip` (9) |
| **2e Rys** | `CINT2e_drv` | `kernels/two_electron.rs` — `launch_two_electron_hess2e` (rank 9/81 host-routed, `Hess2eKind`) | `int2e_ipvip1ipvip2` (81), `int2e_ip1v_r1` (9), `int2e_ip1v_rc1` (9), `int2e_ipvg1_xp1` (9), `int2e_ipvg2_xp1` (9) |
| **2e σ** (`e1` or `e2 == 4`) | `CINT2e_drv` + σ | `kernels/two_electron.rs` — `rel2e_family_dispatch` + `launch_rel2e_sigma_spinor` | `int2e_ipspsp1`, `int2e_ip1spsp2`, `int2e_ipspsp1spsp2`, `int2e_ipsrsr1`, `int2e_ip1srsr2`, `int2e_ipsrsr1srsr2` |
| **2c2e Rys** | `CINT2c2e_drv` | `kernels/center_2c2e.rs` — `launch_center_2c2e_grad` / `launch_center_2c2e_hess1` (`:1052-1058`) | `int2c2e_ip1ip2` (9) |
| **3c2e Rys** | `CINT3c2e_drv` | `kernels/center_3c2e.rs` — `Hess3c2eKind` (`:3079`) | `int3c2e_ip1ip2` (9), `int3c2e_ipvip1` (9), `int3c2e_ipspsp1` (3, σ) |

**Reading the `ng[]` headroom.** `ng[0..4]` are the `(i, j, k, l)` angular-momentum
increments the G-tensor is built with. Example: `int1e_ipippnucp` has `{3,1,0,0,...}`
→ bra `li+3`, ket `lj+1`. That drives the Rys root count
(`nroots = (li_e + lj_e)/2 + 1`) and therefore whether the family exceeds the
`MAX_DEVICE_NROOTS = 5` device cap and must be **host-routed** through the FND-02 host
Wheeler path — exactly as Phase 25 did for `deriv3.c`/`deriv4.c`.

**Precompute this per family before implementing:**
```
li_e = li + ng[0];  lj_e = lj + ng[1];  nroots = (li_e + lj_e)/2 + 1
if nroots > 5  →  host-route (follow kernels/deriv34.rs as the pattern)
```
For a `d`-shell pair (`li = lj = 2`), `int1e_ipippnucp` gives `nroots = 4` (device OK),
but `int1e_iprinviprip` (`{1,3,...}`) gives `nroots = 4` and
`int1e_ipiprinvrip` (`{2,2,...}`) gives `nroots = 5` — at the cap. For `f`-shells all
Tier-6 rank-81 families exceed it. **Assume host-routing for every rank ≥ 27 family.**

---

## 4. WAVES

### Wave 0 — Correct the record and instrument the gate (blocking, ~half a day)

**W0-01 — File the upstream correction.**
`pyscf_rs` is gating real work on a closed blocker. Write
`/home/user/Documents/workspace/cintx/.planning/notes/phase-21-pyscf-rs-handoff.md`
(the file already exists — append a dated section) stating:
- All 8 families named in `pyscf_rs/.planning/phases/07-gradients-geomopt/07-01-PLAN.md:46-48`
  ship today, with the evidence table from §0 of this document.
- `Builder::with_rinv_origin` closes the `with_rinv_at_nucleus` item.
- The `pyscf_rs` GRAD-01..07 `[~]` "cintx-gated" dispositions and every
  `#[ignore = "blocked on cintx"]` in `pyscf-grad` can be un-gated.
- What is genuinely still missing for a downstream consumer: the Tier-1/Tier-2 Hessian
  families (they block `pyscf.hessian` and `pyscf.df.hessian`, not gradients).

**W0-02 — Extend the parity reference set (closes Gap C).**
`xtask` manifest-audit currently diffs against `cint_funcs.h` + supplemental headers.
Add a third source: every `ALL_CINT(<sym>)` / `ALL_CINT1E(<sym>)` occurrence in
`libcint-master/src/autocode/*.c` and `libcint-master/src/*.c`.
```bash
grep -rhoE 'ALL_CINT1?E?\((\w+)\)' libcint-master/src --include=*.c | sed -E 's/.*\((.*)\)/\1/' | sort -u
```
Emit the diff as `unsupported_libcint_families` in the audit report. **On first run this
must report exactly the 33 Gap-A symbols** (plus any non-derivative symbols in the same
situation — enumerate and record them; do not silently widen this plan's scope).
Make the audit FAIL when the list is non-empty, gated behind a
`CINTX_PARITY_STRICT=1` env var until Wave 5 closes the list.

**W0-03 — Wire the vendor oracle for the new symbols.**
Two edits in `crates/cintx-oracle/build.rs`:
1. Append `extern CINTIntegralFunction <sym>_cart;` / `_sph;` / `_spinor;` lines for all
   33 Gap-A symbols to the `suppl_h_content` literal (starts at `:302`), grouped by
   source file with a comment naming that file.
2. Append the same symbol names to the `allowlist_function(...)` regex (`:397`).
   The regex is one long `|`-joined string — append, do not restructure.
   `intor2.c` and `lresc.c` are **not currently in the `cc::Build` file list** — add them
   alongside the existing `grad1.c`/`grad2.c`/`hess.c`/`deriv3.c`/`deriv4.c` entries at
   `:264-278`.
3. Add matching `pub fn vendor_<sym>_<rep>(...)` wrappers in
   `crates/cintx-oracle/src/vendor_ffi.rs`, following the `vendor_int1e_ipovlp_sph`
   pattern at `:476`.

**W0-04 — Fixtures.**
Reuse the existing Phase-25 Hessian fixture set. Confirm coverage for: a `d`-shell pair
with `nctr > 1` (general contraction), a non-zero `rinv` origin, and a kappa-bearing
shell (for the σ families). If any is missing, extend the fixture module — do not
create a parallel one.

**Wave 0 gate:** `cargo build -p cintx-oracle --features vendor` succeeds with
`CINTX_ORACLE_BUILD_VENDOR=1`, and a throwaway test can call
`vendor_int1e_iprinvip_sph` and get non-zero output.

---

### Wave 1 — Tier 1: molecular Hessian (3 symbols) — unblocks `pyscf.hessian`

| Plan | Symbol | Engine | Notes |
|---|---|---|---|
| **W1-01** | `int1e_iprinvip` (rank 9, type 1) | 1e rinv-Rys | `ng={1,1,0,0,...}` → both-side single ∇. Closest existing sibling: `int1e_ipnucip` (`one_electron.rs`, `is_ipnucip`) — same both-side shape, nuclear instead of rinv. **Port `is_ipnucip` and swap the engine to the rinv path**, reusing the origin resolution at `:10673`. |
| **W1-02** | `int1e_ipipr` (rank 27, type 0) | 1e overlap-deriv | `ng={2,1,0,0,3,1,1,27}` → bra `li+2`, ket `lj+1`, and the `r` operator reads the gauge origin. Rank 27 = 9 (∇∇) × 3 (r axis). Closest sibling: `int1e_irp` (`is_irp`, rank 9) — same `r`-on-ket-with-origin structure, one fewer ∇. |
| **W1-03** | `int2e_ipvip1ipvip2` (rank 81) | 2e Rys, host-routed | `ng={1,1,1,1,4,1,1,81}` — one ∇ on each of the four centers. `Hess2eKind` already has `Ipip1ipip2` (rank 81, `two_electron.rs:3745`) with identical buffer shape; add `Ipvip1ipvip2` alongside it and transcribe `CINTgout2e_int2e_ipvip1ipvip2` from `hess.c:3317`. |

**Wave 1 gate:** three `vendor_*` tests green at `atol=1e-12` for cart+sph on the
`d`-shell `nctr>1` fixture; three manifest rows `oracle_covered=true`.

---

### Wave 2 — Tier 2: DF Hessian (3 symbols) — unblocks `pyscf.df.hessian`

| Plan | Symbol | Engine | Notes |
|---|---|---|---|
| **W2-01** | `int2c2e_ip1ip2` (rank 9) | 2c2e Rys | `ng={1,0,1,0,...}` → one ∇ on bra i, one on aux k. `center_2c2e.rs:1052` already dispatches `ip1`/`ip2`/`ipip1`; add an `"ip1ip2"` arm. The `j`/`l` slots stay phantom `s` (`lj=ll=0`) exactly as the existing gradient arms do. |
| **W2-02** | `int3c2e_ipvip1` (rank 9) | 3c2e Rys | `ng={1,1,0,0,...}` → ∇ on bra i and ket j, aux k undifferentiated. Add to the `Hess3c2eKind` enum at `center_3c2e.rs:3079` next to `Ipip1`/`Ipip2`. |
| **W2-03** | `int3c2e_ip1ip2` (rank 9) | 3c2e Rys | `ng={1,0,1,0,...}` → ∇ on bra i and on aux k. Same `Hess3c2eKind` extension. Note this is the *only* 3c2e Hessian family that differentiates the aux center together with a bra center — verify the `ll`-slot handling against the `ip2` gradient arm (`center_3c2e.rs:1609`), which already owns the real-aux-k derivative. |

**Wave 2 gate:** three `vendor_*` tests green cart+sph; DF-Hessian symbol set complete.

---

### Wave 3 — Tier 3: scalar X2C gradient / Hessian (6 symbols)

All six are `⟨p · O · p⟩`-shaped: momentum operators on **both** sides with additional
∇ headroom. They differ only in (a) nuclear vs rinv engine and (b) where the extra ∇ sits.

| Plan | Symbols | ng headroom | Engine |
|---|---|---|---|
| **W3-01** | `int1e_ippnucp` (3), `int1e_ipprinvp` (3) | `{2,1,0,0}` | nuclear-Rys / rinv-Rys |
| **W3-02** | `int1e_ippnucpip` (9), `int1e_ipprinvpip` (9) | `{2,2,0,0}` | both-side |
| **W3-03** | `int1e_ipippnucp` (9), `int1e_ipipprinvp` (9) | `{3,1,0,0}` | bra `li+3` — **check `nroots`; host-route if > 5** |

Implement W3-01 first and prove it; W3-02/03 are the same `gout` with extra derivative
factors. Transcribe each `CINTgout1e_*` from `grad1.c` / `hess.c` literally.

**Wave 3 gate:** six `vendor_*` tests green cart+sph; `pyscf/x2c/sfx2c1e_grad.py` and
`sfx2c1e_hess.py` symbol requirements satisfiable.

---

### Wave 4 — Tier 4 + Tier 5 σ and remaining families (17 symbols)

These ride the Phase 28/29 σ machinery (`sigma_p.rs` four-block assembler +
`cart_to_spinor_si_*` transforms). **Do not start Wave 4 before Phase 29 is closed** —
the 2e σ transform suite is its deliverable.

| Plan | Symbols | Engine |
|---|---|---|
| **W4-01** | `int1e_ipspnucsp`, `int1e_ipsprinvsp` (rank 3) | 1e σ-Rys — thinnest σ gradient; proof vehicle for the wave |
| **W4-02** | `int1e_ipipspnucsp`, `int1e_ipipsprinvsp`, `int1e_ipspnucspip`, `int1e_ipsprinvspip` (rank 9) | 1e σ-Rys |
| **W4-03** | `int2e_ipspsp1`, `int2e_ip1spsp2`, `int2e_ipspsp1spsp2` (rank 3) | 2e σ — extend `rel2e_family_dispatch` |
| **W4-04** | `int2e_ipsrsr1`, `int2e_ip1srsr2`, `int2e_ipsrsr1srsr2` (rank 3) | 2e σ (`sr` variants — same shape, different gout) |
| **W4-05** | `int3c2e_ipspsp1` (rank 3) | 3c2e σ |
| **W4-06** | `int2e_ip1v_r1`, `int2e_ip1v_rc1`, `int2e_ipvg1_xp1`, `int2e_ipvg2_xp1` (rank 9) | 2e Rys + gauge origin — reuse the Phase-22 `PTR_COMMON_ORIG` plumbing |

**Wave 4 gate:** all 17 green; `pyscf/grad/dhf.py` symbol requirements satisfiable.

---

### Wave 5 — Gap B (11 spinor rows) + Tier 6 + close the parity gate

| Plan | Content |
|---|---|
| **W5-01** | Prove the 10 declared-but-unverified spinor gradient rows (§1.3) byte-identical; flip `oracle_covered=true`. These have kernels already — the work is a vendor test per row plus whatever transform bug each one surfaces. Budget generously: Phase 30-01d shows a single spinor family can hide a multi-day residual hunt. |
| **W5-02** | `int1e_ecp_iprinv_spinor` — remove the hard `UnsupportedApi` rejection at `kernels/ecp.rs:2047` and implement the spinor path, or convert the rejection into a documented, manifest-declared `unsupported_policy` row. **Decide explicitly; do not leave it as a bare rejection with a `Stability::Stable` manifest row — that combination is the one state the audit cannot express.** |
| **W5-03** | Tier 6 (`int1e_iprinvr`, `int1e_iprip`, `int1e_iprinviprip`, `int1e_ipiprinvrip`). No PySCF consumer; required only for PARITY-01. All rank 9–81, all rinv or overlap engine, all likely host-routed. |
| **W5-04** | Flip `CINTX_PARITY_STRICT` on by default. `unsupported_libcint_families` must be empty. Update `.planning/ROADMAP.md` Phase 31 success criterion 4 to reference the widened reference set from W0-02. Refresh `CHANGELOG.md`. |

---

## 5. Test and oracle strategy

### 5.1 Per-family test template

`crates/cintx-oracle/tests/vendor_grad_gap_<tier>.rs`:

```rust
// One test per (symbol, representation). NOT #[ignore]d once green.
#[test]
#[cfg(has_vendor_libcint)]
fn vendor_int1e_iprinvip_sph_d_shell_nctr2() {
    let fx = fixtures::d_shell_nctr2_with_rinv_origin();
    let vendor = vendor_ffi::vendor_int1e_iprinvip_sph(&fx, shls);
    let ours   = cintx_eval(&fx, "int1e_iprinvip_sph", shls);
    assert_eq!(ours.len(), 9 * ni * nj, "component rank must be 9");
    compare::assert_byte_identical(&ours, &vendor, 1e-12);
}
```

### 5.2 Three gates, in order

| Gate | What it proves | When |
|---|---|---|
| **G1 — shape** | output length == `rank × ni × nj [× nk × nl]`, and the component axis is leading in F-order | immediately, before any numerics |
| **G2 — byte identity** | `atol = 1e-12` vs vendored libcint on the `d`-shell `nctr>1` fixture | the real gate |
| **G3 — determinism** | two consecutive calls are bit-identical (the existing `ipovlp`/`ipkin`/`ipnuc` determinism tests at `one_electron.rs:13008`, `:13214`, `:13270` are the pattern) | before flipping `oracle_covered` |

**G1 catches the most common failure in this class of work: a transposed or
interleaved component axis.** libcint returns `out[n*rank + comp]` (component fastest)
for some drivers and component-leading for others; the existing `int2e_ip1` path
already documents a required transpose (`two_electron.rs:3700-3703`, "Risk R3").
Check the driver's convention per family — do not assume.

### 5.3 Finite-difference cross-check (oracle-free, catches sign errors)

For every rank-3 gradient family, verify against a central difference of the
corresponding scalar integral:
```
∂/∂A_x ⟨i|O|j⟩ ≈ [⟨i(A+h)|O|j⟩ − ⟨i(A−h)|O|j⟩] / 2h,   h = 1e-4 bohr
```
Agreement to `1e-6` catches a wrong sign or a swapped bra/ket derivative center
**without needing the vendor build**. Add one FD test per rank-3 family. For rank-9
both-side families, difference the rank-3 family instead of the scalar.

---

## 6. Risk register

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R-01 | Component-axis convention differs per driver → silently transposed output | **MAJOR** | Gate G1 on every family; consult the driver's `c2s_*` function, not a sibling family's behavior |
| R-02 | `nroots > MAX_DEVICE_NROOTS = 5` for high-`l` on the `+3`/`+4` headroom families | **MAJOR** | Compute `nroots` per family per §3 *before* implementing; host-route via the `deriv34.rs` pattern; add a `f`-shell fixture to the rank-27/81 tests |
| R-03 | Spinor gradient residuals (Gap B) hide multi-day debugging | **MAJOR** | Phase 30-01d precedent. Budget Wave 5 generously; keep `oracle_covered=false` and the gate `#[ignore]`d rather than over-claiming (RULE 4) |
| R-04 | Adding 33 manifest entries shifts `OperatorId` positional indices | **MAJOR** | RULE 5 — symbol-name dispatch only. Re-run the `ecp_operator_ids_match_constants` test in `cintx-ops/src/resolver.rs`; the four `OperatorId::INT1E_ECP_*` constants in `cintx-core/src/operator.rs:70-77` are the only positional literals and MUST be updated if positions move. **Prefer appending new rows after existing ones so positions do not shift at all.** |
| R-05 | `intor2.c` / `lresc.c` newly added to the oracle `cc::Build` break the vendor build | MINOR | Add them in W0-03 and prove the build before any family work |
| R-06 | The widened parity reference set (W0-02) surfaces non-derivative missing families too | MINOR | Enumerate and record them in the audit report; do NOT expand this plan's scope — file them as a separate item |
| R-07 | Tier 5/6 families have no PySCF consumer, so no downstream signal if subtly wrong | MINOR | Vendor byte-identity is the only gate; add the FD cross-check (§5.3) where the rank allows |

---

## 7. Effort and sequencing summary

| Wave | Symbols | Unblocks | Depends on |
|---|---:|---|---|
| 0 | — | everything; corrects `pyscf_rs`; proves/kills the fail-open hazard (§1.3b) | — |
| **0.5** | **10** | **`pyscf_rs` PBC Phase 10 (GTH pseudopotentials) — the earliest downstream blocker** | **W0** |
| 1 | 3 | `pyscf.hessian.{rhf,uhf}` | W0 |
| 2 | 3 | `pyscf.df.hessian.{rhf,uhf}` | W0 |
| 3 | 6 | `pyscf.x2c.sfx2c1e_{grad,hess}` | W0 |
| 4 | 17 | `pyscf.grad.dhf`, Tier-5 surface | W0, **Phase 29 closed** |
| 5 | 11 + 4 | PARITY-01 | W1–W4 |
| **Total** | **33 new + 11 spinor verifications + 10 moment-weighted (Gap B′)** | | |

**Recommended order if a downstream consumer is waiting:** W0 → **W0.5** → W1 → W2 → W3 → W4 → W5.
W0.5 is small (10 symbols, one engine class each, all sph-only) and unblocks the entire
`pyscf_rs` periodic stack from Phase 10 onward. Waves 1–3 unblock molecular Hessians,
which nothing is currently waiting on.

Waves 1, 2 and 3 are mutually independent and can run in parallel.
Wave 4 is the only one with an external dependency.

---

## 8. Where to look when stuck

| Problem | Where |
|---|---|
| "What is this family's math?" | `libcint-master/src/autocode/<file>.c`, function `CINTgout{1,2}e_<symbol>` |
| "Which engine does it ride?" | §3 of this document |
| "What is the closest implemented sibling?" | §3 + `grep -n 'op_name == "' crates/cintx-cubecl/src/kernels/one_electron.rs` |
| "How do I add a manifest row?" | `crates/cintx-ops/src/generated/api_manifest.csv` → regenerate → RULE 7 |
| "How do I wire the vendor oracle?" | `crates/cintx-oracle/build.rs:264-278` (cc build), `:302-388` (suppl header), `:397` (allowlist), `src/vendor_ffi.rs:476` (wrapper pattern) |
| "How do I host-route a high-nroots family?" | `crates/cintx-cubecl/src/kernels/deriv34.rs` (Phase 25 FND-02 pattern) |
| cubecl build error | `docs/cubecl_error_guideline.md` — **mandatory before any fix** |
| Project conventions | `AGENTS.md`, `docs/rust_crate_test_guideline.md`, `docs/libcint_route_coverage_manifest_spec.md` |
