---
phase: quick-260601-aty
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/cintx-cubecl/src/kernels/one_electron.rs
  - crates/cintx-cubecl/src/kernels/two_electron.rs
  - crates/cintx-oracle/tests/one_electron_scalar_spinor_parity.rs
  - crates/cintx-oracle/tests/oracle_gate_closure.rs
autonomous: true
requirements:
  - ACCEPT-1   # int1e_{ovlp,kin,nuc}_spinor + int2e_spinor evaluate on nctr>1 basis, byte-match vendor @1e-12
  - ACCEPT-2   # nctr>1 spinor fixture (non-square block) added to parity suites, contraction-major AO ordering
  - ACCEPT-3   # remove/relax the two UnsupportedApi guards (1e ~11020, 2e ~3607)
  - ACCEPT-4   # (stretch) confirm + cover spinor gradient under nctr>1

must_haves:
  truths:
    - "int1e_ovlp_spinor evaluates on a general-contracted (nctr>1) basis and byte-matches vendored libcint at atol=1e-12"
    - "int1e_kin_spinor evaluates on nctr>1 and byte-matches vendor at atol=1e-12"
    - "int1e_nuc_spinor evaluates on nctr>1 and byte-matches vendor at atol=1e-12"
    - "int2e_spinor evaluates on nctr>1 and byte-matches vendor at atol=1e-12"
    - "The nctr>1 spinor output is contraction-major: AO index i_global = ci*di + i_sp, byte-matching vendor CINTcgto_spinor AO ordering"
    - "The two UnsupportedApi nctr>1 guards (1e + 2e) no longer reject general contraction"
    - "The spinor gradient path under nctr>1 is confirmed evaluating and is covered by a vendor parity assertion (or honest-deferred with reason if no vendor reference exists)"
  artifacts:
    - path: "crates/cintx-cubecl/src/kernels/one_electron.rs"
      provides: "1e spinor arm with per-(ci,cj) contraction-major scatter; UnsupportedApi nctr>1 guard removed"
      contains: "for ci in 0..n_ctr_i"
    - path: "crates/cintx-cubecl/src/kernels/two_electron.rs"
      provides: "2e spinor arm with per-(ci,cj,ck,cl) contraction-major scatter; UnsupportedApi nctr>1 guard removed"
      contains: "for ci in 0..n_ctr_i"
    - path: "crates/cintx-oracle/tests/one_electron_scalar_spinor_parity.rs"
      provides: "nctr>1 non-square (p×d) spinor fixture + 3 vendor parity tests (ovlp/kin/nuc)"
      contains: "nctr"
    - path: "crates/cintx-oracle/tests/oracle_gate_closure.rs"
      provides: "nctr>1 int2e_spinor vendor parity gate"
      contains: "nctr"
  key_links:
    - from: "one_electron.rs Representation::Spinor arm"
      to: "cart_to_spinor_sf_2d"
      via: "per-(ci,cj) loop, per-column coeffs, contraction-major scatter into n2c-dim dense interleaved-complex output"
      pattern: "cart_to_spinor_sf_2d"
    - from: "two_electron.rs Representation::Spinor arm"
      to: "cart_to_spinor_sf_4d"
      via: "per-(ci,cj,ck,cl) loop, contraction-major scatter into n2c^4 dense interleaved-complex output"
      pattern: "cart_to_spinor_sf_4d"
    - from: "parity fixture build (nctr>1)"
      to: "vendor_CINTcgto_spinor / vendor_int1e_*_spinor"
      via: "double-gated oracle (cpu + CINTX_ORACLE_BUILD_VENDOR=1), atol=1e-12, count_mismatches==0"
      pattern: "CINTX_ORACLE_BUILD_VENDOR"
---

<objective>
Close the spinor general-contraction (nctr>1) gap. Today the `Representation::Spinor`
arms of the 1e and 2e kernels fail closed with `UnsupportedApi` whenever any shell has
`nctr > 1`, so the spinor surface (`int1e_{ovlp,kin,nuc}_spinor`, `int2e_spinor`) works
only on segmented bases and errors on every production general-contracted basis
(cc-pVDZ, 6-31G valence, ANO, …). This blocks the downstream pyscf-rs F-03 H2O/cc-pVDZ
byte-identity bar.

Purpose: make the spin-free spinor families production-complete on general-contracted
bases, with byte-identity vendor proof, by mirroring the already-proven scalar/sph
contraction-major scatter pattern that sits directly above each guard.

Output:
- 1e spinor arm: per-(ci,cj) cart→spinor with each column's own coefficients, scattered
  contraction-major into a dense `n2c`-dim interleaved-complex buffer.
- 2e spinor arm: per-(ci,cj,ck,cl) equivalent for the 4D transform.
- Both `UnsupportedApi` nctr>1 guards removed.
- nctr>1, NON-SQUARE-block vendor parity fixtures added to the 1e and 2e parity suites.
- Spinor gradient nctr>1 path confirmed + covered (or honest-deferred with reason).
</objective>

<execution_context>
@/home/user/Documents/workspace/cintx/.claude/get-shit-done/workflows/execute-plan.md
@/home/user/Documents/workspace/cintx/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/todos/pending/spinor-general-contraction-nctr-gt1.md
@./CLAUDE.md

# THE TASK (read in full): contains the finding, guard sites, proven fix template, acceptance.
@.planning/todos/pending/spinor-general-contraction-nctr-gt1.md

# Invoke the `spike-findings-cintx` skill before touching layout/scatter:
#   - spinor output is interleaved-complex, component-leading, ket-major AROUND the interleave;
#   - nctr>1 composes contraction-major: i_global = ci*di + i_sp.
# This is the load-bearing layout fact for both tasks.

<interfaces>
<!-- Extracted from the codebase. Executor uses these directly — no exploration needed. -->

PROVEN TEMPLATE — the scalar/sph contraction-major scatter that sits RIGHT ABOVE the
1e spinor guard (one_electron.rs ~10998-11013). Mirror this shape exactly:
```rust
// Spheric arm (the pattern to mirror for Spinor):
let di_sph = n_ctr_i * nsi;
for ci in 0..n_ctr_i {
    for cj in 0..n_ctr_j {
        let base = (ci * n_ctr_j + cj) * block_len;   // block_len = nci*ncj
        let mut sph_tmp = vec![0.0_f64; nsi * nsj];
        cart_to_sph_1e(&cart_blocks[base..base + block_len], &mut sph_tmp, li, lj);
        for mj in 0..nsj {
            let jj = cj * nsj + mj;
            for mi in 0..nsi {
                let ii = ci * nsi + mi;
                let dst = ii + jj * di_sph;            // contraction-major, bra fastest
                staging[dst] = F::from_f64_lossy(sph_tmp[mj * nsi + mi]);
            }
        }
    }
}
```
KEY: each (ci,cj) reads ITS OWN cart_blocks sub-block at base=(ci*n_ctr_j+cj)*block_len.
The device scalar kernel already accumulated all nctr_i*nctr_j blocks with the correct
per-column coefficients (see out_total = nctr_i*nctr_j*block_len at line ~263), so the
arm does NOT re-apply coefficients — it only transforms+scatters the already-contracted
per-column blocks. Verify this holds for the spinor path too.

cart_to_spinor_sf_2d (transform/c2spinor.rs:531) — 1e spin-free cart→spinor:
```rust
pub fn cart_to_spinor_sf_2d<F: CintFloat>(
    staging: &mut [F],   // writes di*dj*2 interleaved-complex, column-major: staging[(j*di+i)*2 +{0:re,1:im}]
    cart: &[f64],        // bra-major / ket-fastest: cart[bra*ncj + ket], len >= nci*ncj
    li: u8, kappa_i: i16,
    lj: u8, kappa_j: i16,
) -> Result<(), cintxRsError>
// di = spinor_len(li, kappa_i), dj = spinor_len(lj, kappa_j).
```

cart_to_spinor_sf_4d (transform/c2spinor.rs:1235) — 2e spin-free cart→spinor:
```rust
pub fn cart_to_spinor_sf_4d<F: CintFloat>(
    staging: &mut [F],   // writes di*dj*dk*dl*2 interleaved-complex
                         // staging[(((l_sp*dk+k_sp)*dj+j_sp)*di+i_sp)*2 +{0:re,1:im}]
    cart: &[f64],        // i-fastest: cart[((l*nck+k)*ncj+j)*nci+i], len >= nci*ncj*nck*ncl
    li: u8, kappa_i: i16, lj: u8, kappa_j: i16,
    lk: u8, kappa_k: i16, ll: u8, kappa_l: i16,
) -> Result<(), cintxRsError>
```

spinor_len (transform/c2spinor.rs:25): kappa==0 → 4l+2; kappa<0 → 2l+2 (GT); kappa>0 → 2l (LT).

CURRENT 1e SPINOR ARM (one_electron.rs ~11015-11045) — to replace. Note the existing
ket-major→bra-major transpose (KEEP IT, apply per (ci,cj) sub-block):
```rust
Representation::Spinor => {
    if n_ctr_i != 1 || n_ctr_j != 1 { return Err(UnsupportedApi{...}); }   // <-- REMOVE
    let kappa_i = shell_i.kappa; let kappa_j = shell_j.kappa;
    // device emits ket-major (block[cj_idx*nci + ic]); cart_to_spinor_sf_2d reads
    // bra-major (cart[ic*ncj + jc]) → transpose to bra-major FIRST.
    let mut cart_bra_major = vec![0.0f64; nci * ncj];
    for ic in 0..nci { for jc in 0..ncj {
        cart_bra_major[ic * ncj + jc] = cart_blocks[jc * nci + ic];
    }}
    cart_to_spinor_sf_2d::<F>(staging, &cart_bra_major, li, kappa_i, lj, kappa_j)?;
}
```

CURRENT 2e SPINOR ARM (two_electron.rs ~3602-3623) — to replace. The 4D transform reads
i-fastest cart[((l*nck+k)*ncj+j)*nci+i]; the cart_blocks sub-block per (ci,cj,ck,cl) is
the device 4D block (verify its native order; the Cart arm at ~3637 reads
block[ic + nfi*(jc + nfj*(kc + nfk*lc))] = i-fastest, which already matches sf_4d's cart
layout — so the 2e spinor path likely needs NO transpose, only per-quad scatter):
```rust
Representation::Spinor => {
    if n_ctr_i!=1||n_ctr_j!=1||n_ctr_k!=1||n_ctr_l!=1 { return Err(UnsupportedApi{...}); } // <-- REMOVE
    cart_to_spinor_sf_4d::<F>(staging, &cart_blocks, li,kappa_i, lj,kappa_j, lk,kappa_k, ll,kappa_l)?;
}
```

PARITY HARNESS (one_electron_scalar_spinor_parity.rs):
- build_two_p_spinor(): p (l=1, atom0) × d (l=2, atom1), nctr=1, asserts NON-SQUARE 3×6 cross block.
- collect_cintx_spinor / collect_vendor_spinor: per shell-pair eval_raw / vendor_fn, stitch_block.
- spinor_len_kappa0(l) = 4l+2; n_sp = Σ shell_nsp. Vendor: vendor_CINTcgto_spinor(s, &bas).
- count_mismatches(ref, obs, ATOL=1e-12, RTOL=0). assert_fixture_asymmetric(&bas).
- Vendor tests gated: #[cfg(has_vendor_libcint)] #[cfg(feature="cpu")].
- IMPORTANT for nctr>1: vendor_CINTcgto_spinor already returns nctr*(4l+2); set
  shell_nsp[s] = nctr[s] * spinor_len_kappa0(l[s]) in the cintx collector so block sizes
  and global stitch offsets stay contraction-major-consistent with vendor.

GRADIENT nctr>1 (already EVALUATES via cart_to_spinor_sf_derivative_2d — test
one_electron.rs:13064 test_ipovlp_spinor_grad_nctr_gt1_evaluates passes; this is a smoke
test, NOT vendor parity). Task 3 confirms + adds vendor coverage or honest-defers.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Wire nctr>1 into the 1e spinor arm + add nctr>1 vendor parity fixture</name>
  <files>crates/cintx-cubecl/src/kernels/one_electron.rs, crates/cintx-oracle/tests/one_electron_scalar_spinor_parity.rs</files>
  <behavior>
    - After fix, int1e_{ovlp,kin,nuc}_spinor on a nctr=2 non-square (p×d) basis byte-match
      vendor libcint at atol=1e-12 (mismatch_count == 0).
    - Output AO ordering is contraction-major: global bra index ii = ci*di + i_sp,
      matching vendor CINTcgto_spinor(=nctr*(4l+2)) layout.
    - nctr==1 behavior is byte-identical to before (regression guard: existing
      test_int1e_*_spinor_asym_parity still pass).
  </behavior>
  <action>
    In one_electron.rs `Representation::Spinor` arm (~11015-11045):
    1. REMOVE the `if n_ctr_i != 1 || n_ctr_j != 1 { return Err(UnsupportedApi{...}); }` guard (ACCEPT-3).
    2. Mirror the Spheric arm directly above. Compute di = spinor_len(li, kappa_i),
       dj = spinor_len(lj, kappa_j), n2c_i = n_ctr_i * di (dense bra dim). For each
       (ci,cj): base=(ci*n_ctr_j+cj)*block_len; transpose THAT sub-block ket-major→bra-major
       into cart_bra_major[ic*ncj+jc] = cart_blocks[base + jc*nci + ic] (KEEP the transpose
       the existing arm has — it is the 260529-jtd/kke orientation fix); call
       cart_to_spinor_sf_2d into a per-(ci,cj) temp buffer `tmp = vec![0.0f64; di*dj*2]`;
       then scatter contraction-major into the dense interleaved-complex `staging`:
       for j_sp in 0..dj { let jj = cj*dj + j_sp; for i_sp in 0..di { let ii = ci*di + i_sp;
       let dst = (jj*n2c_i + ii)*2; staging[dst] = F::from_f64_lossy(tmp[(j_sp*di+i_sp)*2]);
       staging[dst+1] = F::from_f64_lossy(tmp[(j_sp*di+i_sp)*2+1]); } }.
       Per spike-findings-cintx: spinor output is interleaved-complex, component-leading,
       ket-major; nctr composes contraction-major i_global=ci*di+i_sp.
    3. Do NOT re-apply coefficients in this arm — the device scalar kernel already
       accumulated each (ci,cj) block with its OWN column coefficients (out_total =
       nctr_i*nctr_j*block_len). Verify cart_blocks indexing base=(ci*n_ctr_j+cj)*block_len
       matches the kernel's emit order (same as the Spheric/Cart arms use). If the device
       coeff layout were transposed (the column-major env vs row-major Shell pitfall,
       memory raw_nctr_coeff_transpose) the Spheric/Cart nctr>1 tests would already fail —
       they pass, so reuse the identical base formula.
    In one_electron_scalar_spinor_parity.rs:
    4. Add build_two_p_spinor_nctr2(): same NON-SQUARE p(l=1)×d(l=2) on distinct atoms,
       but NCTR_OF=2 for each shell with 3 primitives → 2 contraction columns of distinct
       coeffs (two columns × 3 prims; env coeff block is COLUMN-major env[ci*nprim+ip]).
       Update collect_cintx_spinor + collect_vendor_spinor to read shell_nsp[s] =
       (bas[s*BAS_SLOTS+NCTR_OF] as usize) * spinor_len_kappa0(l) so block sizes are
       nctr_i*di × nctr_j*dj and the stitch stays contraction-major. (vendor_CINTcgto_spinor
       already returns nctr*(4l+2).)
    5. Add 3 vendor parity tests (test_int1e_{ovlp,kin,nuc}_spinor_nctr2_parity), gated
       #[cfg(has_vendor_libcint)] #[cfg(feature="cpu")], asserting count_mismatches(vendor,
       cintx, ATOL=1e-12, RTOL=0) == 0 AND assert_any_nonzero on both. Add an
       assert_fixture_nctr_gt1(&bas) guard (NCTR_OF==2 for both shells AND l0!=l1).
  </action>
  <verify>
    <automated>cargo build -p cintx-cubecl --features cpu 2>&1 | tail -5 && CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test one_electron_scalar_spinor_parity -- --nocapture 2>&1 | tail -40</automated>
  </verify>
  <done>The UnsupportedApi nctr>1 guard is gone from the 1e spinor arm; the three
  test_int1e_*_spinor_nctr2_parity tests run under the double gate
  (CINTX_ORACLE_BUILD_VENDOR=1 + --features cpu) and report 0 mismatches at atol=1e-12;
  the pre-existing nctr==1 test_int1e_*_spinor_asym_parity tests still pass.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Wire nctr>1 into the 2e spinor arm + add nctr>1 int2e_spinor vendor gate</name>
  <files>crates/cintx-cubecl/src/kernels/two_electron.rs, crates/cintx-oracle/tests/oracle_gate_closure.rs</files>
  <behavior>
    - After fix, int2e_spinor on a nctr=2 basis with at least one non-square angular pair
      byte-matches vendor libcint at atol=1e-12 (mismatch_count == 0).
    - Output AO ordering is contraction-major in all four indices
      (i_global=ci*di+i_sp, …), matching vendor.
    - nctr==1 behavior byte-identical (existing oracle_gate_2e_spinor still passes).
  </behavior>
  <action>
    In two_electron.rs `Representation::Spinor` arm (~3602-3623):
    1. REMOVE the `if n_ctr_i!=1||...||n_ctr_l!=1 { return Err(UnsupportedApi{...}); }` guard (ACCEPT-3).
    2. Mirror the Spheric/Cart arms above (~3571-3600 / ~3624-3656). Compute
       di=spinor_len(li,kappa_i), dj/dk/dl likewise; n2c_i=n_ctr_i*di (and dj/dk/dl
       analogues). For each (ci,cj,ck,cl): base=(((ci*n_ctr_j+cj)*n_ctr_k+ck)*n_ctr_l+cl)*block_len;
       slice cart_blocks[base..base+block_len]; call cart_to_spinor_sf_4d into a per-quad
       temp `tmp = vec![0.0f64; di*dj*dk*dl*2]`; then scatter contraction-major into the
       dense interleaved-complex staging. The 4D layout is
       staging[(((l_sp*n2c_k+k_idx)*n2c_j+j_idx)*n2c_i+i_idx)*2 +{0:re,1:im}] where
       i_idx=ci*di+i_sp, j_idx=cj*dj+j_sp, k_idx=ck*dk+k_sp, l_idx=cl*dl+l_sp; tmp is read
       tmp[(((l_sp*dk+k_sp)*dj+j_sp)*di+i_sp)*2 +{0,1}]. Confirm the cart sub-block order
       fed to sf_4d matches its expected i-fastest cart[((l*nck+k)*ncj+j)*nci+i] — the Cart
       arm reads block[ic+nfi*(jc+nfj*(kc+nfk*lc))] (i-fastest) directly with NO transpose,
       so the 2e spinor path needs NO transpose either; pass the raw sub-block.
    3. Do NOT re-apply coefficients (device kernel already contracted per column).
    In oracle_gate_closure.rs:
    4. Add build_h2o_or_two_center_spinor_nctr2() (or extend an existing builder) producing
       a small 4-shell basis where the relevant quartet has NCTR_OF=2 on the contracted
       shells and includes at least one l>0 shell so a NON-SQUARE angular pair appears
       (e.g. an s/s/p/p or s/p/p/d quartet) — avoid an all-square all-s quartet that hides
       orientation. (Per spike-findings-cintx D-07: orientation needs ni>1 AND nj>1.)
    5. Add oracle_gate_2e_spinor_nctr2(): #[cfg(has_vendor_libcint)], size nelems with
       vendor_CINTcgto_spinor (already nctr-aware), call vendor_int2e_spinor and eval_raw
       INT2E_SPINOR on the SAME quartet, assert count_mismatches_atol(vendor, cintx,
       ATOL_SPINOR=1e-12)==0 and nonzero>0.
  </action>
  <verify>
    <automated>cargo build -p cintx-cubecl --features cpu 2>&1 | tail -5 && CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test oracle_gate_closure -- oracle_gate_2e_spinor --nocapture 2>&1 | tail -40</automated>
  </verify>
  <done>The UnsupportedApi nctr>1 guard is gone from the 2e spinor arm;
  oracle_gate_2e_spinor_nctr2 runs under the double gate and reports 0 mismatches at
  atol=1e-12; the pre-existing nctr==1 oracle_gate_2e_spinor still passes.</done>
</task>

<task type="auto">
  <name>Task 3: Confirm + cover the spinor GRADIENT path under nctr>1 (stretch); scope-note global-AO question</name>
  <files>crates/cintx-oracle/tests/one_electron_scalar_spinor_parity.rs, crates/cintx-cubecl/src/kernels/one_electron.rs</files>
  <action>
    1. CONFIRM the gradient nctr>1 path: test_ipovlp_spinor_grad_nctr_gt1_evaluates
       (one_electron.rs:13064) already proves nctr>1 spinor gradient EVALUATES via
       cart_to_spinor_sf_derivative_2d (no UnsupportedApi). Verify it still passes and grep
       the gradient spinor arm(s) for any residual `n_ctr_i != 1` / UnsupportedApi nctr
       guard — if one exists, remove it the same way as Task 1 (mirror the scalar/sph
       contraction-major scatter); if none exists, record that in the SUMMARY.
    2. COVER: determine whether a vendor byte-identity reference exists for a derivative
       spinor family on nctr>1 (e.g. vendor_int1e_ipovlp_spinor via vendor_ffi). If it
       exists, add a gradient nctr>1 vendor parity test mirroring Task 1's structure
       (non-square block, ncomp=3, contraction-major, atol=1e-12, count_mismatches==0).
       If libcint ships that derivative spinor arm as a return-0/exit(1) stub (see memory:
       libcint_spinor_deriv_driver_stubs — int2c2e_ip1/ip2_spinor return 0, CINT3c1e_spinor
       exit(1)), DO NOT over-claim: keep the evaluates-smoke test, leave any gate
       #[ignore]d, and record an honest-deferred note (reason + which vendor symbol is a
       stub) in the SUMMARY. No silent skip.
    3. SCOPE NOTE (do NOT implement here): in the SUMMARY, record the todo's open question —
       multi-shell-same-l SEGMENTED bases (e.g. 6-31g 3×s/2×p, all nctr==1) show an
       eigenvalue-identical but globally PERMUTED spinor AO ordering vs PySCF; this is a
       distinct global-assembly / ao_loc_2c convention concern, NOT this contraction gap.
       State whether the contraction-major fix here plausibly reconciles it or whether it
       needs its own follow-up todo. Do not expand this task's scope to chase it.
  </action>
  <verify>
    <automated>CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test one_electron_scalar_spinor_parity 2>&1 | tail -20 && cargo test -p cintx-cubecl --features cpu test_ipovlp_spinor_grad_nctr_gt1_evaluates 2>&1 | tail -10</automated>
  </verify>
  <done>The gradient spinor nctr>1 path is confirmed evaluating (no residual nctr
  UnsupportedApi guard, or the guard is removed); it is either covered by a vendor parity
  test reporting 0 mismatches at atol=1e-12 OR honest-deferred with a recorded reason
  (vendor stub) and an #[ignore]d gate; the global-AO-ordering open question is recorded as
  a scoped follow-up note in the SUMMARY without expanding this task.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| oracle test → device kernel arm | test-driven shells/bas/env cross into the spinor scatter; only test/oracle glue, no external untrusted input |
| device cart buffer → host transform/scatter | already-contracted per-column blocks cross into the host cart→spinor scatter loop; index arithmetic must stay in-bounds |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-aty-01 | Tampering | 1e/2e spinor scatter index arithmetic (dst=(jj*n2c_i+ii)*2) | mitigate | sf_2d/sf_4d already BufferTooSmall-guard the per-block temp; staging is sized by the caller from nctr-aware spinor AO count; vendor byte-identity parity at atol=1e-12 detects any off-by-one or transposed scatter |
| T-aty-02 | Information disclosure | stale/zero lanes in dense n2c output if a (ci,cj[,ck,cl]) tuple is skipped | mitigate | full nctr loop writes every contraction tuple; assert_any_nonzero + count_mismatches==0 against vendor catch any unwritten lane |
| T-aty-03 | Denial of service | mis-sized staging on nctr>1 → BufferTooSmall / OOB | accept | fail-closed: cart_to_spinor_sf_2d/4d return BufferTooSmall (typed) rather than partial write, consistent with the OOM-safe stop contract; test sizes staging from vendor_CINTcgto_spinor |
</threat_model>

<verification>
- Both UnsupportedApi nctr>1 guards removed (grep `spinor.*general contraction` in
  one_electron.rs and two_electron.rs returns nothing in the active arms).
- `cargo build -p cintx-cubecl --features cpu` succeeds (no new clippy/build errors).
- Double-gated oracle (CINTX_ORACLE_BUILD_VENDOR=1 + --features cpu):
  - 1e: test_int1e_{ovlp,kin,nuc}_spinor_nctr2_parity → 0 mismatches @ atol=1e-12.
  - 2e: oracle_gate_2e_spinor_nctr2 → 0 mismatches @ atol=1e-12.
  - pre-existing nctr==1 spinor parity tests still pass (no regression).
- Gradient nctr>1: evaluates-smoke passes; vendor parity covered OR honest-deferred w/ reason.
- The nctr>1 fixtures use a NON-SQUARE angular block (p×d), per spike-findings D-07.
</verification>

<success_criteria>
- int1e_{ovlp,kin,nuc}_spinor and int2e_spinor evaluate on a general-contracted (nctr>1)
  basis and byte-match vendored libcint at atol=1e-12 (ACCEPT-1).
- nctr>1 spinor fixtures with contraction-major AO ordering added to the 1e and 2e parity
  suites, using non-square angular blocks (ACCEPT-2).
- Both UnsupportedApi nctr>1 guards removed (ACCEPT-3).
- Spinor gradient nctr>1 confirmed + covered or honest-deferred with reason (ACCEPT-4).
- Global-AO-ordering open question recorded as a scoped follow-up, not expanded into scope.
</success_criteria>

<output>
After completion, create
`.planning/quick/260601-aty-spinor-general-contraction-nctr-gt1/260601-aty-SUMMARY.md`
</output>
