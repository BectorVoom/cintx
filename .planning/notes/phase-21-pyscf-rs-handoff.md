---
title: Phase 21 hand-off — plain-Coulomb gradient families are libcint-verified; pyscf_rs Phase 7 grad arms un-gate
date: 2026-05-26
type: note
context: Phase 21 close-out (21-08). Consumer-facing hand-off for BectorVoom/pyscf_rs.
audience: pyscf_rs maintainer (Phase 7 — Gradients + Geomopt)
---

# Phase 21 hand-off: plain-Coulomb gradient families are now byte-identical to libcint 6.1.3

cintx Phase 21 shipped the six plain-Coulomb / Hellmann–Feynman gradient
integral families plus the `int3c2e_ip1` derivative repair. All cart + sph
representations are now **byte-identical to vendored libcint 6.1.3 at
atol=1e-12** (rtol=0.0), proven by the `#[cfg(has_vendor_libcint)]` oracle
parity gate (double-gated: `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`).
This note tells the pyscf_rs maintainer exactly which Phase 7 analytical-gradient
arms can un-gate, and records the load-bearing history a consumer needs.

## 1. What is verified (byte-identity, cart + sph, atol=1e-12)

| Family | Arity | Role in the gradient | cintx oracle gate | Plan |
|--------|-------|----------------------|-------------------|------|
| `int2e_ip1` | 4 | two-electron force `∇_A <ij\|kl>` (highest-impact term) | `two_electron_ip1_parity` | 21-05 |
| `int3c2e_ip1` | 3 | DF/RI three-center derivative (pyscf-grad DF-grad path) | `center_3c2e_parity` | 21-06 |
| `int1e_ipovlp` | 2 | Pulay / overlap-derivative term | `one_electron_grad_parity` | 21-03 |
| `int1e_ipkin` | 2 | core-Hamiltonian kinetic derivative | `one_electron_grad_parity` | 21-03 |
| `int1e_ipnuc` | 2 | hcore nuclear-attraction derivative (∇ on bra, ∑ over all nuclei, −Z_C) | `one_electron_nuc_grad_parity` | 21-04 |
| `int1e_iprinv` | 2 | per-atom Hellmann–Feynman force (single rinv origin, factor +1.0) | `one_electron_nuc_grad_parity` | 21-04 |
| `int1e_ecp_iprinv` (`ECPscalar_iprinv`) | 2 | per-nucleus ECP force | `ecp_iprinv_parity` | 21-07 |

Full vendor-gated suite re-run green at 21-08 close-out:

```
CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
  --test two_electron_ip1_parity --test center_3c2e_parity \
  --test one_electron_grad_parity --test one_electron_nuc_grad_parity \
  --test ecp_iprinv_parity
```

Result: all parity assertions **executed** (not skipped) and reported 0
mismatches at atol=1e-12:
`two_electron_ip1_parity` 3/3, `center_3c2e_parity` 2/2,
`ecp_iprinv_parity` 3/3, `one_electron_grad_parity` 8/8,
`one_electron_nuc_grad_parity` 6/6.

In the manifest (`crates/cintx-ops/generated/compiled_manifest.lock.json`) all
six families + `int3c2e_ip1` carry `oracle_covered: true` for cart + sph;
`manifest-audit` exits 0.

## 2. Component-leading [3, …] F-order layout (Risk R3 — VALIDATED)

The consumer's `pyscf-gto layout_table.rs` declares the component-leading
`[3, …]` F-order convention. cintx emits exactly that:

- `int2e_ip1` → `[3, nl, nk, nj, ni]` F-order (21-05).
- `int3c2e_ip1` → `[3, nk, nj, ni]` F-order (21-06).
- 1e gradients → `[3, nj, ni]` (`staging[comp * ni*nj + n]`, 21-03/21-04).

**The validation is the oracle parity itself.** The element-for-element
byte-identity comparison against libcint's own `int2e_ip1` / `int3c2e_ip1`
output IS the layout gate: a component-leading-vs-component-trailing (or
transposed-axis) drift produces nonzero `count_mismatches`. Because every
parity test reports 0 mismatches at atol=1e-12, the layout is proven to match
libcint's own component-leading order, which `layout_table.rs` mirrors.
**The consumer can rely on the [3, …] component-leading layout without a repack
shim.** (Risk R3 closed.)

## 3. Which pyscf_rs Phase 7 `workflow_dispatch` gradient arms un-gate

The "real acceptance test" for Phase 21 is the consumer un-gate. Once these
families are green (now), pyscf_rs flips its Phase 7
(`.planning/phases/07-gradients-geomopt/`) `workflow_dispatch` gradient arms
from opt-in to always-on. The arms that un-gate:

- **RHF** analytical gradient
- **UHF** analytical gradient
- **RKS** analytical gradient
- **UKS** analytical gradient
- **MP2** analytical gradient
- **CCSD** analytical gradient
- **CPHF** (coupled-perturbed Hartree–Fock response)
- **geomopt** (geometry optimization driver riding the above)

These arms then ride the existing pyscf_rs gates with **zero pyscf_rs rework**:
- finite-difference gate `grad.verify_fd` at **≤1e-6 Ha/Bohr**, and
- upstream-PySCF analytical-gradient parity at **≤1e-7 Ha/Bohr**.

No dispatch-shape change and no layout repack are required: the arity-4 /
arity-3 dispatch shape and the component-leading layout are already wired in
`pyscf-gto` (`SessionRequest` wrapper in `crates/pyscf-gto/src/intor.rs`;
DF-grad consumer at `crates/pyscf-grad/src/hooks.rs:24`).

## 4. `int3c2e_ip1` re-gating history (Risk R1 — READ THIS)

`int3c2e_ip1` was, before Phase 21, a **registered-but-stubbed operator-blind
scalar kernel**: `launch_center_3c2e_typed` ignored the `ip1` operator and
silently returned the plain (non-derivative) 3c2e integral. The oracle
"passed" only because its reference was wired to the plain `vendor_int3c2e_*`
symbol, not the derivative — so the gate was self-consistent but verifying the
wrong thing (a latent silent-wrong path).

Phase 21-06 fixed this:
- Shipped a **real 3-component `∇_A` derivative kernel** (`launch_center_3c2e_ip1`,
  reusing `gout_ip1` verbatim through the 3c2e Pitfall-4 kl-mapping
  `build_2e_shape(li+1, lj, 0, lk)`).
- **Flipped the oracle reference** from plain `vendor_int3c2e_*` to the real
  `vendor_int3c2e_ip1_*` and lifted the tolerance to atol=1e-12.
- Verified byte-identity on H2O/STO-3G.

**Consumer impact:** `pyscf-grad/src/hooks.rs:24` (the DF-grad path) now
receives a *correct* three-center derivative. Any local workaround the consumer
may have carried to compensate for the old scalar stub can be removed. The
21-02 manifest correction set `int3c2e_ip1` `oracle_covered` back to `false`
(undoing the spurious pre-Phase-21 `true`); 21-06 then re-flipped it to `true`
on the real derivative.

## 5. Caveats the consumer MUST honour

### 5a. Spinor gradients are UnsupportedApi (Risk R5 / D-03)

All gradient **spinor** representations (`int*_ip*_spinor`,
`int1e_iprinv_spinor`, `int2e_ip1_spinor`, `int3c2e_ip1_spinor`,
`int1e_ecp_iprinv_spinor`) are *registered* in the manifest for surface
completeness but their kernels return **`UnsupportedApi`** — the cart→spinor
gradient transform is intentionally not implemented. Their manifest rows stay
`oracle_covered: false` (a "skipped, not verified" state — they carry no
oracle-parity obligation). pyscf_rs needs only `sph`/`cart` for the un-gating
arms, so this is not a blocker. Do not request spinor gradient output.

### 5b. High-l (f/g) l-ceiling (Risk R2)

The gradient raises `li → li+1`, so the Rys-root count is
`nroots = (li+1 + lj + lk + ll)/2 + 1`. This is ≤5 for s/p/d quartets but
**> 5 for f/g quartets**, which return `UnsupportedApi` — the same ceiling as
base `int2e` (`rys_root1..5` dispatch). **Do not request f/g gradient quartets**
until the deferred Wheeler-fallback higher-roots work lands. d/d/d/d is the
ceiling (gradient nroots = 5, allowed); all-f quartets overflow.

### 5c. int2e_ip1 safe-API vs raw-path (Risk R6 — resolved, safe arm shipped)

R6 asked whether `pyscf-gto/src/intor.rs` calls cintx's raw/compat path or the
arity-4 safe-API path. Finding (21-05 SUMMARY): the **safe-API arm shipped, not
deferred**. Phase 18's arity-4 `SessionRequest::evaluate` dispatch +
`IntegralTensor.component_axis_leading` already flow `int2e_ip1` through once
the kernel emits the correct component-leading staging — verified via
`crates/cintx-rs/src/api.rs` + `crates/cintx-oracle/tests/safe_api_arity4_parity.rs`.
The raw `eval_raw` arm (which the oracle tests and `intor.rs` use today) is
independent of Phase 18 (D-11). Either entry point reaches byte-identical
output; no Phase-18 completion dependency remains for `int2e_ip1`.

## 6. Verification path for the consumer

Per `.planning/notes/pyscf-rs-as-cintx-consumer.md`, a safe-API change is "done"
only when both oracle gates pass:
1. **Primary (cintx):** the per-family `*_parity.rs` byte-identity gates above.
2. **Secondary (pyscf_rs):** `pyscf-gto tests/oracle/` (`release-oracle-tests`)
   against PySCF reference outputs.

Phase 21 satisfies gate (1). The consumer un-gate exercises gate (2) via the
`grad.verify_fd` (≤1e-6 Ha/Bohr) + upstream-PySCF parity (≤1e-7 Ha/Bohr) checks.

## References

- cintx Phase 21 plan SUMMARYs: `.planning/phases/21-coulomb-gradient-intors/21-0{1..8}-SUMMARY.md`
- Manifest source of truth: `crates/cintx-ops/generated/compiled_manifest.lock.json`
- Consumer wiring note: `.planning/notes/pyscf-rs-as-cintx-consumer.md`
- pyscf_rs Phase 7: `pyscf_rs/.planning/phases/07-gradients-geomopt/07-RESEARCH.md`
  §"Gradient-Integral Availability Matrix"
- pyscf-gto component-leading layout: `pyscf-gto/src/layout_table.rs` (Risk R3)
- pyscf-grad DF-grad consumer: `pyscf-grad/src/hooks.rs:24` (Risk R1)

---
*Phase: 21-coulomb-gradient-intors — close-out hand-off (21-08)*
*Created: 2026-05-26*

## 2026-08-22 correction and current downstream disposition

The blocker statement still quoted by
`pyscf_rs/.planning/phases/07-gradients-geomopt/07-01-PLAN.md:46-48` is stale.
The gradient families it names are shipped today:

| Requirement | Current cintx evidence |
|---|---|
| `int1e_ipovlp`, `int1e_ipkin`, `int1e_ipnuc`, `int1e_iprinv` | cart+sph+spinor dispatch in `crates/cintx-cubecl/src/kernels/one_electron.rs`; vendor parity suites are live |
| `int2e_ip1`, `int2e_ip2` | cart+sph dispatch in `crates/cintx-cubecl/src/kernels/two_electron.rs` |
| `int3c2e_ip1`, `int3c2e_ip2` | cart+sph+spinor dispatch in `crates/cintx-cubecl/src/kernels/center_3c2e.rs` |
| `int2c2e_ip1`, `int2c2e_ip2` | cart+sph dispatch in `crates/cintx-cubecl/src/kernels/center_2c2e.rs` |
| `ECPscalar_ipnuc`, `ECPscalar_iprinv` | cart+sph dispatch in `crates/cintx-cubecl/src/kernels/ecp.rs` |
| rinv origin selection | `cintx_rs::Builder::with_rinv_origin([f64; 3])` |

Accordingly, the `pyscf_rs` GRAD-01..07 dispositions marked `[~]` solely as
"cintx-gated", and tests ignored solely with `blocked on cintx`, can be
un-gated. The real remaining consumer gap is in the additional molecular and
DF Hessian symbols (not the first-gradient families), notably
`int1e_iprinvip`, `int2e_ipvip1ipvip2`, `int2c2e_ip1ip2`,
`int3c2e_ip1ip2`, and `int3c2e_ipvip1`.

The historical caveats in sections 5a and 5b describe the Phase-21 state, not
the current tree: later phases added spinor-gradient coverage and the
higher-root Wheeler fallback. Consumers should use the compiled manifest lock
and current oracle reports as the authoritative support record.

## 2026-08-22 (later the same day) — Wave 4 close-out, and one correctness notice

Two updates from the gradient-gap Wave-4 work
(`.planning/notes/gradient-gap-wave-4-PLAN.md`).

### Newly available to consumers

`pyscf.grad.dhf` is now fully satisfiable. All five symbols it reaches are
`oracle_covered = true` at `atol = 1e-12` against vendored libcint 6.1.3:
`int1e_ipspnucsp`, `int1e_ipsprinvsp`, `int2e_ipspsp1`, `int2e_ip1spsp2`,
`int2e_ipspsp1spsp2` (all spinor). The `srsr` siblings, `int3c2e_ipspsp1`, and the
four `intor2.c` gauge/cross-product families (`int2e_ip1v_r1`, `int2e_ip1v_rc1`,
`int2e_ipvg1_xp1`, `int2e_ipvg2_xp1`, the last group in cart + sph + spinor) also
landed covered.

The Hessian gap named in the previous section is closed too — `int1e_iprinvip`,
`int2e_ipvip1ipvip2`, `int2c2e_ip1ip2`, `int3c2e_ip1ip2` and `int3c2e_ipvip1` all
carry `oracle_covered = true` in the current lock.

### ⚠️ Correctness notice — `int2e_spinor` was wrong before this commit

`int2e_spinor` returned the `i↔j` transpose of the correct spinor block for **every
shell with `l > 0`** (~3e-3 absolute on a `p` quartet). It was exact only for an
all-`s` quartet, where the transpose is the identity — which is precisely what the
existing coverage exercised, so the row carried `oracle_covered = true` throughout.

Root cause: `cart_to_spinor_sf_4d` did not apply the KET→BRA transpose that its leaf
`cart_to_spinor_sf_2d` requires. Fixed, and now gated on non-square quartets by
`crates/cintx-oracle/tests/two_electron_spinor_orientation.rs`
(residuals `3.1e-3 → 1.6e-17`).

**Any downstream result computed with `int2e_spinor` (or the `int2e_breit_*` and F12
spinor families, which share the transform) at `l > 0` before this commit should be
recomputed.** Scalar `cart`/`sph` paths are unaffected.

---

## 2026-08-22 — gradient-gap Wave 5: X2C base families now available

**New for consumers:** `int1e_pnucp` and `int1e_prinvp` ship today in **cart, sph and
spinor**, byte-identical to vendored libcint 6.1.3 at `atol = 1e-12` across s/p/d shells
(`crates/cintx-oracle/tests/gradient_gap_wave5_x2c_base.rs`).

This closes a gap that was easy to miss. Wave 3 shipped the *derivatives*
(`int1e_ippnucp`, `int1e_ipprinvp`, `int1e_ippnucpip`, `int1e_ipprinvpip`,
`int1e_ipippnucp`, `int1e_ipipprinvp`) and reported the `pyscf/x2c` symbol set as
satisfiable. It was only satisfiable for `sfx2c1e_grad.py` / `sfx2c1e_hess.py`:
`pyscf/x2c/x2c.py` calls `int1e_pnucp` **directly** to build the X2C Hamiltonian itself,
and that base family did not exist until now. **A downstream X2C port can proceed.**

`int1e_prinvp` reads the per-nucleus origin from `env[PTR_RINV_ORIG]` (env[4..6]), the
same slot as `int1e_iprinv`; `Builder::with_rinv_origin([f64;3])` sets it from the safe
API. `int1e_pnucp` is atom-summed and ignores that slot — asserted by
`only_prinvp_depends_on_the_rinv_origin`.

**Correction to a coverage claim, if you depend on it:** `int3c2e_ip1_spinor` and
`int3c2e_ip2_spinor` were marked `oracle_covered = true` but **failed closed for
`nctr_k > 1`** (general-contracted auxiliary shell). Fixed and re-proven at
`nctr_i = nctr_j = nctr_k = 2`. If you evaluated those with a generally-contracted aux
basis before 2026-08-22 you received a typed `UnsupportedApi`, never a wrong number.

**Still unavailable, and permanently so for v1.4** — do not plan around these landing:
- `int1e_ecp_iprinv_spinor` — no oracle exists in libcint or PySCF; fail-closed.
- `int3c1e{,_ip1,_iprinv}_spinor` and `int2c2e_{ip1,ip2,ip1ip2}_spinor` — vendored
  libcint's own drivers are unconditional stubs, so byte-identity is unobtainable.
  cintx evaluates the first five; they simply cannot carry `oracle_covered = true`.
