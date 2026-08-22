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
