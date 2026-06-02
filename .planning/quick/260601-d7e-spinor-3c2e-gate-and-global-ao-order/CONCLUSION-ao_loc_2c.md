# CONCLUSION — spinor global AO (`ao_loc_2c`) ordering

**Quick task:** 260601-d7e (Sub-problem B)
**Date:** 2026-06-01
**Disposition:** prove + document; **NO PySCF-compat ordering mode added** (locked decision).

## Question

Does cintx's global spinor AO (`ao_loc_2c`-equivalent) ordering match vendored libcint
6.1.3 on a SEGMENTED multi-shell-same-l basis — the configuration the pyscf-rs report
flagged as a possible cross-shell permutation?

## Evidence

- **Test:** `crates/cintx-oracle/tests/spinor_global_ao_order_parity.rs`
  - `test_spinor_global_ao_order_parity` (`#[cfg(has_vendor_libcint)] #[cfg(feature="cpu")]`)
  - `test_spinor_global_ao_order_evaluates` (non-vendor smoke, plain `--features cpu`)
- **Fixture:** 4 shells `[s, p, s, p]` (l = 0,1,0,1) on 2 atoms, all `nctr==1`.
  - SEGMENTED: l=0 and l=1 each repeat across distinct shells, INTERLEAVED (not grouped).
  - >=3 shells, at least one l>0 shell (the two p shells, di=6 → non-square cross blocks).
  - A fixture guard (`assert_fixture_segmented_same_l`) enforces this shape.
- **Driver:** `int1e_ovlp_spinor` (`INT1E_OVLP_SPINOR`) — a REAL vendor driver (scalar
  spinor, stable; not a return-0/exit(1) stub).
- **Assembly:** both cintx and vendor stitch each shell-pair block into the full
  `n_sp×n_sp` (n_sp=16) interleaved-complex matrix via the IDENTICAL column-major /
  bra-fastest formula `dst = ((col_off+jj)*n_sp + (row_off+ii))*2`, advancing offsets in
  shell order. The only thing that can differ is the per-shell global offset.
- **Measured result:** `count_mismatches(vendor, cintx, atol=1e-12, rtol=0.0) == 0`
  (**0 mismatches**, both sides nonzero, lengths equal). Ran under the DOUBLE gate
  (`--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`), not skipped (2 tests ran, 0 filtered).

## Finding

cintx's `CINTshells_spinor_offset` (`crates/cintx-compat/src/helpers.rs:177`) delegates to
`write_offsets`, which builds cumulative per-shell offsets in SHELL ORDER:
`ao_loc[0]=0; ao_loc[i] = ao_loc[i-1] + CINTcgto_spinor(i-1)`. This mirrors libcint
`shells_cgto_offset` (cint_bas.c) EXACTLY. Neither cintx nor libcint groups or reorders by
angular momentum — both lay AOs out strictly in shell order. The segmented `[s,p,s,p]`
basis (where an l-grouping permutation WOULD diverge) is byte-identical to vendor.

## Conclusion

cintx's global spinor AO ordering is **libcint-faithful**. The pyscf-rs permutation report
is a PySCF `ao_loc_2c` CONVENTION difference (PySCF groups/orders AOs differently than
libcint's native shell order) and belongs in pyscf-rs's libcint→PySCF AO mapping layer —
it is **NOT a cintx defect**.

Per the locked decision, **cintx ordering is unchanged and NO PySCF-compat ordering mode
was added** — doing so would risk the libcint-compat core (cintx's primary value). No
permutation layer was introduced; the parity test passes against native shell-order
assembly on both sides.

## Caveat

This conclusion holds because Task B's parity test PASSED (0 mismatches). Had it failed,
this doc would instead record the real cintx-vs-libcint divergence and the minimal
libcint-matching fix applied to the global stitch/offset — not a PySCF-compat mode.
