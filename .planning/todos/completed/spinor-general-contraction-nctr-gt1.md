---
id: spinor-general-contraction-nctr-gt1
created: {
  timestamp: 2026-06-01T07:39:08.000Z
}
source: pyscf-rs F-03 (spinor-integrals) audit-fix — downstream consumer
severity: warning
resolves_phase: null
resolved: 2026-06-01
resolved_by: quick-260601-aty (commits 8d6ebbc, 89fff71, 69798aa; merge af63c1b)
resolution: >
  1e + 2e spinor arms now do per-(ci,cj[,ck,cl]) contraction-major scatter; both
  UnsupportedApi nctr>1 guards removed. int1e_{ovlp,kin,nuc}_spinor, int2e_spinor, and
  the int1e_ipovlp_spinor gradient byte-match vendored libcint 6.1.3 at atol=1e-12 on
  NON-SQUARE nctr=2 fixtures (double-gated cpu + CINTX_ORACLE_BUILD_VENDOR=1). Verifier 7/7.
  Spun-off follow-ups (NOT part of this item's acceptance): (a) oracle_gate_3c2e_spinor
  pre-existing failure (old must-reject contract, untouched center_3c2e.rs); (b) the
  segmented multi-shell-same-l global AO permutation vs PySCF (distinct ao_loc_2c concern).
  See quick/260601-aty-.../deferred-items.md.
---

# Spinor cart→spinor transform fails closed for general contraction (nctr>1)

## Finding

The `Representation::Spinor` arms of the 1e and 2e kernels fail closed with
`UnsupportedApi` whenever any shell in the tuple has `nctr > 1` (general
contraction). Only SEGMENTED bases (`nctr == 1` per shell) can be evaluated as
spinors today.

Guard sites:

- **1e** — `crates/cintx-cubecl/src/kernels/one_electron.rs:11020-11023`
  ```rust
  if n_ctr_i != 1 || n_ctr_j != 1 {
      return Err(cintxRsError::UnsupportedApi {
          requested: "spinor 1e with general contraction (nctr>1)".to_owned(),
      });
  }
  ```
- **2e** — `crates/cintx-cubecl/src/kernels/two_electron.rs:3602-3609`
  ```rust
  if n_ctr_i != 1 || n_ctr_j != 1 || n_ctr_k != 1 || n_ctr_l != 1 {
      return Err(cintxRsError::UnsupportedApi {
          requested: "spinor 2e with general contraction (nctr>1)".to_owned(),
      });
  }
  ```
- The spinor GRADIENT path appears to carry the same limit (see the
  `nctr>1` ipovlp-spinor test at `one_electron.rs:13057`) — confirm + cover.

Behaviour is correct-but-incomplete: it errors cleanly (no wrong numbers, no
silent truncation), but the families simply don't work on general-contracted
bases.

## Why it matters (downstream)

pyscf-rs F-03 wires `mol.intor_spinor("int1e_{ovlp,kin,nuc}_spinor")` and
`int2e_spinor` by routing through these kernels (`Representation::Spinor`).
Because of this gap, the spinor surface works only for segmented/minimal bases
(STO-3G — byte-verified vs upstream PySCF 2.12.1 at atol 1e-10), and **errors on
every general-contracted production basis** (cc-pVDZ, 6-31G valence, ANO, …).
That blocks F-03's own headline acceptance bar (`H2O/cc-pVDZ` byte-identity) and
keeps the feature non-production-complete. pyscf-rs guards it upfront in
`crates/pyscf-gto/src/spinor.rs::ensure_segmented_for_spinor`.

## Fix template (already proven in this repo)

This is the same class as the RESOLVED todo
`spinor`/contraction gap fixed in `wr03-3c1e-grad-nctr-gt1` (int3c1e
grad+scalar): the segmented path collapsed all `(ci,cj,…)` contraction columns
into one block; the fix wraps the per-primitive accumulation in a per-column
loop and scatters one block per contraction-column tuple into a single dense
**contraction-major** output (`i_global = ci*nblk_i + i_idx`).

The scalar/sph branch sitting RIGHT ABOVE the 1e spinor guard already shows the
exact shape to mirror (`one_electron.rs:10996-11013`):

```rust
for ci in 0..n_ctr_i {
    for cj in 0..n_ctr_j {
        let base = (ci * n_ctr_j + cj) * block_len;
        // transform this one (ci,cj) cart block ...
        // scatter contraction-major: ii = ci*nsi + mi, jj = cj*nsj + mj,
        // dst = ii + jj * di_sph
    }
}
```

For spinor, wrap `cart_to_spinor_sf_2d` (1e) / `cart_to_spinor_sf_4d` (2e) in the
same `(ci,cj[,ck,cl])` loop, each column using that column's coefficients, and
scatter each `n2c`-block contraction-major into the dense `n2c`-dim output.

Affected transform entry points:
`crates/cintx-cubecl/src/transform/c2spinor.rs` — `cart_to_spinor_sf`
(1e, :290), `cart_to_spinor_sf_2d` (:531), `cart_to_spinor_sf_4d` (:1235); plus
the σ (si) variants if their callers can see `nctr>1`.

## Acceptance

- `int1e_{ovlp,kin,nuc}_spinor` and `int2e_spinor` evaluate on a
  general-contracted basis (e.g. **H2O/cc-pVDZ**) and byte-match vendored
  libcint at `atol=1e-12` (the double-gated `cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`
  oracle path; see `cintx-oracle/tests/one_electron_scalar_spinor_parity.rs` and
  `oracle_gate_closure.rs` for the existing segmented int2e_spinor harness).
- Add a `nctr>1` (general-contraction) spinor fixture to the parity suites —
  contraction-major AO ordering must match vendor.
- Remove/relax the two `UnsupportedApi` guards once covered.
- (Stretch) confirm + cover the spinor gradient path under `nctr>1`.

## Notes / open question

Separately, pyscf-rs observed that for multi-shell-same-l SEGMENTED bases
(e.g. 6-31g: 3×s, 2×p, all `nctr==1`) the assembled spinor matrix is
eigenvalue-identical to upstream PySCF but the **global AO index ordering**
differs (a pure permutation). That is likely a distinct global-assembly /
`ao_loc_2c` ordering-convention matter rather than this contraction gap — worth
checking whether the nctr>1 fix's contraction-major ordering also reconciles it,
or whether it needs its own item.
