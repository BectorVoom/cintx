---
quick_id: 260530-iiq
type: quick
slug: int3c1e-general-contraction-nctr-1-suppo
autonomous: true
files_modified:
  - crates/cintx-cubecl/src/kernels/center_3c1e.rs
  - crates/cintx-oracle/tests/int3c1e_genctr_parity.rs
must_haves:
  truths:
    - "launch_center_3c1e_typed (scalar), launch_center_3c1e_ip1, and launch_center_3c1e_iprinv all emit nctr_i*nctr_j*nctr_k separate contraction blocks (NOT a single accumulated block / column-0-only) for nctr>1 shells"
    - "int3c1e (scalar), int3c1e_ip1, and int3c1e_iprinv are byte-identical to vendored libcint 6.1.3 at atol=1e-12 (cart + sph) on a general-contraction (nctr>1) fixture, under the --features cpu + CINTX_ORACLE_BUILD_VENDOR=1 double gate"
    - "the existing nctr==1 byte-identity is NOT regressed (tests/int3c1e_ip_parity.rs stays 5/5; cargo test -p cintx-cubecl --lib stays green)"
  artifacts:
    - path: "crates/cintx-cubecl/src/kernels/center_3c1e.rs"
      provides: "nctr-blocked output for scalar + both gradient 3c1e launchers"
      contains: "n_ctr"
    - path: "crates/cintx-oracle/tests/int3c1e_genctr_parity.rs"
      provides: "nctr>1 vendor byte-identity parity test for int3c1e scalar + ip1 + iprinv"
      contains: "CINTX_ORACLE_BUILD_VENDOR"
  key_links:
    - from: "crates/cintx-cubecl/src/kernels/center_3c1e.rs gradient launchers"
      to: "crates/cintx-cubecl/src/kernels/one_electron.rs nctr-blocked gradient output"
      via: "structural mirror (base = nctr-offset * block_len, component-leading)"
      pattern: "n_ctr"
---

<objective>
Fix WR-03: give the int3c1e family correct general-contraction (nctr>1) support in BOTH the
scalar and the two gradient launchers of `crates/cintx-cubecl/src/kernels/center_3c1e.rs`, and
prove it with a new nctr>1 vendor byte-identity parity test. Today all three paths assume nctr==1:
the scalar `launch_center_3c1e_typed` sizes `cart_buf = nci*ncj*nck` (one block) and ACCUMULATES
every (ci,cj,ck) contraction-column triple into it (`*dst += src`), while the gradient launchers
`launch_center_3c1e_ip1` / `launch_center_3c1e_iprinv` weight by `coefficients[ip*n_ctr_i]`
(column 0 only). For genuinely general-contracted shells (nctr>1) this drops/merges contraction
columns. The architecture convention (proven by `one_electron.rs`) is that launchers emit SEPARATE
per-contraction blocks: `out_total = nctr_i*nctr_j*block_len`, each block written at its own
`base` offset (component-leading for gradients).

Purpose: close the general-contraction gap that is this branch's (`fix/general-contraction-nctr-1e`)
whole theme. Output: nctr-correct scalar + gradient 3c1e kernels, byte-identical to libcint 6.1.3
on an nctr>1 fixture, with the existing nctr==1 parity intact.
</objective>

<context>
@.planning/PROJECT.md
@.planning/STATE.md
@crates/cintx-cubecl/src/kernels/center_3c1e.rs
@crates/cintx-cubecl/src/kernels/one_electron.rs
@crates/cintx-oracle/tests/int3c1e_ip_parity.rs
</context>

<tasks>

<task type="auto">
  <name>Task 1: Determine the libcint int3c1e general-contraction output block ordering (empirical, against vendor)</name>
  <files>crates/cintx-oracle/tests/int3c1e_genctr_parity.rs</files>
  <read_first>
    - crates/cintx-oracle/tests/int3c1e_ip_parity.rs (the existing nctr==1 vendor byte-identity test from plan 23-04 — copy its harness: env/atm/bas construction, the double-gate cfg attributes #[cfg(has_vendor_libcint)] + #[cfg(feature = "cpu")], assert_any_nonzero, non-square block, and how it calls eval_raw for int3c1e_ip1/iprinv and the vendor_* FFI)
    - crates/cintx-oracle/src/vendor_ffi.rs (vendor_int3c1e_sph at ~line 173 and the int3c1e_ip1/iprinv vendor wrappers added in 23-04 — these are the reference; reuse them)
    - crates/cintx-cubecl/src/kernels/one_electron.rs (the nctr-block layout reference: out_total = nctr_i*nctr_j*block_len; scalar base=(ci*nctr_j+cj)*block_len; gradient component-leading equivalent)
  </read_first>
  <action>
    Create crates/cintx-oracle/tests/int3c1e_genctr_parity.rs with a NON-SQUARE general-contraction
    fixture: a bra shell i with nctr_i=2 (two contraction columns sharing the same primitives, e.g. an
    [s s] or [p p] general contraction with distinct coefficient columns), and j, k shells at nctr=1
    with DIFFERENT angular momenta so the block is non-square (e.g. i=p(nctr=2), j=d(nctr=1), k=s(nctr=1)).
    Build atm/bas/env with a libcint-conformant PTR_ENV_START=20 layout (see existing tests; do NOT
    pollute reserved env slots). Drive vendor libcint (vendor_int3c1e_sph + the int3c1e_ip1/iprinv vendor
    wrappers) and cintx eval_raw for: int3c1e (scalar), int3c1e_ip1, int3c1e_iprinv — cart AND sph.
    Gate EVERY parity assertion behind #[cfg(all(has_vendor_libcint, feature = "cpu"))] and assert_any_nonzero
    so a silent 0-test/empty-buffer run fails loudly.

    FIRST, before fixing the kernel, run this test to capture the EXACT vendor output ordering for nctr>1:
    libcint's int3c1e writes contraction blocks in a specific order (ci/cj/ck nesting + i-fastest within a
    block). Determine the correct cintx block-offset formula empirically by comparing element-by-element
    against the vendor buffer (do NOT assume ((ck*nctr_j+cj)*nctr_i+ci) — confirm it; libcint's general
    contraction loop order for 3-center is the source of truth). Record the confirmed ordering in a comment
    in the test and in the SUMMARY. This task's commit may leave the test RED (kernel not yet fixed) — that
    is the intended TDD RED state; note it.
  </action>
  <verify>
    <automated>cargo build -p cintx-oracle --features cpu 2>&1 | tail -5</automated>
  </verify>
  <acceptance_criteria>
    - Test file compiles; harness mirrors int3c1e_ip_parity.rs with the double-gate cfg + non-square nctr>1 fixture.
    - The correct libcint nctr>1 block ordering is determined and documented (comment + SUMMARY).
  </acceptance_criteria>
</task>

<task type="auto">
  <name>Task 2: Emit nctr-blocked output in the scalar launch_center_3c1e_typed</name>
  <files>crates/cintx-cubecl/src/kernels/center_3c1e.rs</files>
  <read_first>
    - crates/cintx-cubecl/src/kernels/center_3c1e.rs (launch_center_3c1e_typed ~1213-1395: the for ck/cj/ci loops, cart_buf sizing ~1282, the `*dst += src` accumulation ~1351, and the staging copy / cart_to_sph_3c1e ~1358-1374)
    - crates/cintx-cubecl/src/kernels/one_electron.rs (out_total = nctr_i*nctr_j*block_len; base offset per contraction block)
  </read_first>
  <action>
    Size the output for nctr_i*nctr_j*nctr_k blocks (not one). For each (ck,cj,ci) triple, write that
    triple's fully-contracted Cartesian block to its OWN offset using the block ordering CONFIRMED in
    Task 1 (do not accumulate all triples into one block). Apply the cart→sph transform per block when
    representation==Spheric, writing each sph block at its nctr-offset in staging. Preserve the exact
    nctr==1 path byte-for-byte (when all nctr_*==1 the new code must reduce to the current single-block
    output). Keep the device kernel (center_3c1e_kernel / run_3c1e_device) UNCHANGED — this is host-side
    block placement only (per the established host-launcher precedent). No #[cube] edits.
  </action>
  <verify>
    <automated>cargo test -p cintx-cubecl --lib 2>&1 | tail -3</automated>
  </verify>
  <acceptance_criteria>
    - cargo test -p cintx-cubecl --lib green (no nctr==1 regression).
    - Scalar 3c1e produces nctr_i*nctr_j*nctr_k separate blocks for nctr>1.
  </acceptance_criteria>
</task>

<task type="auto">
  <name>Task 3: Emit nctr-blocked component-leading output in both gradient launchers + close the test green + close the todo</name>
  <files>crates/cintx-cubecl/src/kernels/center_3c1e.rs, crates/cintx-oracle/tests/int3c1e_genctr_parity.rs</files>
  <read_first>
    - crates/cintx-cubecl/src/kernels/center_3c1e.rs (launch_center_3c1e_ip1 ~978-1066, launch_center_3c1e_iprinv ~1076-1199, write_3c1e_grad_staging ~909-951, contract_3c1e_grad, nabla1i_3c1e, fill_g_tensor_3c1e / fill_g_tensor_3c1e_nuc)
  </read_first>
  <action>
    Wrap the per-primitive gradient accumulation in both gradient launchers in the SAME (ck,cj,ci)
    contraction-column loop as the scalar path, each column using that column's coefficients
    (coefficients[ip*n_ctr_i + ci], etc.). Produce one component-leading [3, nci, ncj, nck] block per
    (ci,cj,ck) triple and write it at that triple's nctr offset via an nctr-aware version of
    write_3c1e_grad_staging (component-leading must be preserved WITHIN each block; blocks ordered per the
    Task-1 confirmed ordering — match how libcint lays out gradient nctr blocks vs components; verify
    against vendor). iprinv keeps the Rys-driven fill_g_tensor_3c1e_nuc base and the nroots>5/fff
    fail-closed guard. Preserve nctr==1 byte-identity exactly.

    Then make int3c1e_genctr_parity.rs pass: run it under the double gate and confirm 0 mismatches at
    atol=1e-12 for int3c1e scalar + ip1 + iprinv, cart + sph, non-square nctr>1 block. Finally move
    .planning/todos/pending/wr03-3c1e-grad-nctr-gt1.md to .planning/todos/completed/ and update its body
    to note the fix landed AND that the scalar path shared the same nctr=1 limitation (now also fixed).
  </action>
  <verify>
    <automated>CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test int3c1e_genctr_parity -- --test-threads=1 2>&1 | tail -25</automated>
  </verify>
  <acceptance_criteria>
    - int3c1e_genctr_parity: running N>0 tests, test result: ok., 0 mismatches at atol=1e-12 (cart+sph) for scalar + ip1 + iprinv on the nctr>1 non-square fixture.
    - Existing CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test int3c1e_ip_parity stays 5/5 (no nctr==1 regression).
    - todo moved to completed with the scalar-path finding noted.
  </acceptance_criteria>
</task>

</tasks>

<success_criteria>
- [ ] Scalar + both gradient 3c1e launchers emit nctr_i*nctr_j*nctr_k blocks
- [ ] New nctr>1 vendor parity test green under the double gate (scalar + ip1 + iprinv, cart+sph, 0 mismatches)
- [ ] No nctr==1 regression (int3c1e_ip_parity 5/5; cubecl --lib green)
- [ ] device #[cube] kernels unchanged (host-side block placement only)
- [ ] wr03 todo moved to completed with scalar-path note
</success_criteria>
