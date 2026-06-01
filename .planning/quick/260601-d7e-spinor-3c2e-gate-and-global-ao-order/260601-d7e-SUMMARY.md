---
phase: quick-260601-d7e
plan: 01
subsystem: oracle / spinor parity
tags: [spinor, oracle, vendor-parity, byte-identity, 3c2e, ao-ordering]
requires:
  - INT3C2E_IP1_SPINOR family (evaluates Ok)
  - INT1E_OVLP_SPINOR family
  - vendored libcint 6.1.3 (CINTX_ORACLE_BUILD_VENDOR=1)
provides:
  - oracle_gate_3c2e_ip1_spinor_vendor_parity (honest, byte-identity)
  - spinor_global_ao_order_parity test (libcint-faithful global ordering proof)
  - CONCLUSION-ao_loc_2c.md
affects:
  - crates/cintx-oracle/tests/oracle_gate_closure.rs
  - crates/cintx-oracle/tests/spinor_global_ao_order_parity.rs
tech-stack:
  added: []
  patterns: [shell-count-generic global collector, aux-k SPHERICAL sizing for 3c spinor]
key-files:
  created:
    - crates/cintx-oracle/tests/spinor_global_ao_order_parity.rs
    - .planning/quick/260601-d7e-spinor-3c2e-gate-and-global-ao-order/CONCLUSION-ao_loc_2c.md
  modified:
    - crates/cintx-oracle/tests/oracle_gate_closure.rs
decisions:
  - "oracle_gate_3c2e_spinor stale R5 must-reject dropped; reconciled to real vendor byte-identity gate (aux-k SPHERICAL), consistent with pre-existing passing adversarial parity"
  - "Global spinor AO ordering is libcint-faithful; NO PySCF-compat mode added (locked)"
metrics:
  duration: ~6 min
  completed: 2026-06-01
  tasks: 3
  files: 3
---

# Quick 260601-d7e: Spinor 3c2e gate + global AO ordering Summary

Reconciled the stale `oracle_gate_3c2e_spinor` R5 must-reject gate with the pre-existing
passing adversarial parity into a real vendor byte-identity gate (aux-k SPHERICAL), and
proved cintx's global spinor AO ordering is byte-identical to libcint on a segmented
multi-shell-same-l basis — no PySCF-compat ordering mode added.

## Tasks

| Task | Name | Status | Commit |
| ---- | ---- | ------ | ------ |
| A | Reconcile stale oracle_gate_3c2e_spinor gate | green | 6c2de94 |
| B | Multi-shell global spinor AO ordering vendor parity | green | d7e6751 |
| C | CONCLUSION doc for ao_loc_2c ordering | green | 59cb40d |

## What was done

### Task A (branch-3a confirmed)
- FIRST ran the pre-existing `test_int3c2e_ip1_spinor_adversarial_parity`
  (spinor_deriv_parity.rs:274) under the double gate — it PASSED (1 passed, 0 ignored, 0
  failed). Branch-3a confirmed: the INT3C2E_IP1_SPINOR family is byte-identical to vendor;
  the must-reject gate was purely stale.
- Replaced the stale gate (asserted `Err(UnsupportedApi)`) with a real vendor byte-identity
  gate, renamed `oracle_gate_3c2e_ip1_spinor_vendor_parity`, on H2O STO-3G triple (3,4,0):
  - bra i / ket j SPINOR-sized (`vendor_CINTcgto_spinor` = nctr·(4l+2)), aux-k SPHERICAL
    (`vendor_CINTcgto_spheric` = nctr·(2lk+1)) — memory: libcint_3c_spinor_auxk_spherical.
  - Buffer length `3·ni_sp·nj_sp·nk_sph·2`; asserts cintx/vendor lengths EQUAL (T-d7e-01
    aux-k mis-sizing guard) before value compare; both sides nonzero; `count_mismatches==0`.
  - Doc-comment cross-references `test_int3c2e_ip1_spinor_adversarial_parity` as the
    primary/adversarial coverage. R5 framing removed; the two gates are CONSISTENT.
- Left `vendor_ffi_3c2e_spinor_nonzero` untouched.

### Task B (PASS — expected)
- New `spinor_global_ao_order_parity.rs`: segmented `[s,p,s,p]` (l=0,1,0,1) 4-shell fixture
  on 2 atoms, all nctr==1; repeated l interleaved + l>0 cross blocks. Fixture guard enforces
  >=3 shells, repeated l, >=1 l>0, all nctr==1.
- Shell-count-generic global collectors (cintx + vendor) stitch shell-pair blocks into the
  full 16×16 interleaved-complex matrix via the identical shell-order offset formula.
- `int1e_ovlp_spinor` byte-identity vs vendor: **count_mismatches == 0** (both nonzero,
  lengths equal). Non-vendor smoke test asserts 16·16·2 size + nonzero.

### Task C
- `CONCLUSION-ao_loc_2c.md`: records the libcint-faithful finding (cintx
  `CINTshells_spinor_offset` → `write_offsets` builds shell-order cumulative offsets,
  mirroring libcint `shells_cgto_offset`; no l-grouping), the measured mismatch count (0),
  and explicitly that NO PySCF-compat ordering mode was added (locked decision).

## Double-gate confirmation (parity actually RAN, not skipped)

| Test | Gate | Result |
| ---- | ---- | ------ |
| test_int3c2e_ip1_spinor_adversarial_parity | cpu + VENDOR=1 | 1 passed, 0 ignored, 0 failed (8 filtered = other names) |
| oracle_gate_3c2e_ip1_spinor_vendor_parity | cpu + VENDOR=1 | PASS, mismatches=0, nonzero=4/24 both sides |
| test_spinor_global_ao_order_parity | cpu + VENDOR=1 | PASS, mismatches=0, n_sp=16, elems=512 |
| test_spinor_global_ao_order_evaluates | cpu (smoke) | PASS, size 16·16·2, nonzero |

All vendor parity assertions executed under `has_vendor_libcint` (real mismatch counts /
nonzero printed; 0 tests filtered/ignored on the parity bodies). No silent skip.

## Task B measured mismatch count vs vendor

**0** mismatches at atol=1e-12, rtol=0.0 on the segmented `[s,p,s,p]` basis.

## Deviations from Plan

None — plan executed exactly as written. Branch-3a (Task A) and the expected PASS (Task B)
both held; no BLOCKER, no contingency branch needed. Stayed entirely out of the 260601-aty
nctr>1 code paths (only oracle test files + one CONCLUSION doc touched). No dispatch
guard/offset-helper edit was needed since no divergence occurred.

## Constraints honored

- No edits to `.planning/phases/**` or `.planning/research/**`.
- No ROADMAP.md edit.
- No PySCF-compat ordering mode added (locked prohibition).
- No edits to two_electron.rs / one_electron.rs Spinor arms.

## Self-Check: PASSED

- FOUND: crates/cintx-oracle/tests/oracle_gate_closure.rs
- FOUND: crates/cintx-oracle/tests/spinor_global_ao_order_parity.rs
- FOUND: .planning/quick/260601-d7e-spinor-3c2e-gate-and-global-ao-order/CONCLUSION-ao_loc_2c.md
- FOUND commits: 6c2de94, d7e6751, 59cb40d
