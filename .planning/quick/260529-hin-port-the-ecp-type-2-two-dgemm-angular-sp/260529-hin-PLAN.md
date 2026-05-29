---
phase: quick-260529-hin
plan: 01
type: execute
wave: 1
depends_on: []
subsystem: cintx-cubecl / cintx-oracle
tags: [cubecl, ecp, gpu, rocm, device-kernel, oracle, type2, dgemm]
files_modified:
  - crates/cintx-cubecl/src/kernels/ecp.rs
autonomous: true
requirements: []
must_haves:
  truths:
    - "The Type-2 (Projected) two-dgemm angular splice runs in a #[cube(launch)] device kernel on the resolved backend (CPU/ROCm/…), not a host loop."
    - "The device-backed Type-2 result is bit-for-bit identical (f64) to the prior host dgemm loops — vendor CPU parity stays green at atol=1e-12/rtol=0.0."
    - "The ecp_type2_cart driver and the gradient drivers (compute_type2_pair_grad / deriv1_cart_pair) dispatch the angular splice through the device path; `let _ = backend;` is removed from ecp_type2_cart."
    - "The ROCm random int1e_ecp_sph oracle reports mismatch_count==0 with the Type-2 device kernel genuinely exercised (Cu/LANL2DZ has non-local Projected channels)."
  artifacts:
    - path: "crates/cintx-cubecl/src/kernels/ecp.rs"
      provides: "ecp_type2_angular_kernel #[cube(launch)] + run_ecp_type2_angular_device::<R> + run_ecp_type2_splice_on_backend, wired into ecp_type2_cart"
      contains: "ecp_type2_angular_kernel"
  key_links:
    - from: "ecp_type2_cart"
      to: "run_ecp_type2_splice_on_backend"
      via: "per-(ic,jc) device dispatch replacing the host two-dgemm loop"
      pattern: "run_ecp_type2_splice_on_backend"
    - from: "run_ecp_type2_splice_on_backend"
      to: "run_ecp_type2_angular_device::<"
      via: "per-backend match (Cpu/Rocm=HipRuntime/Wgpu/Cuda/Metal)"
      pattern: "run_ecp_type2_angular_device::<"
---

<objective>
Port the ECP **Type-2 (Projected) two-dgemm angular splice** out of the host loop
in `ecp_type2_cart` (crates/cintx-cubecl/src/kernels/ecp.rs ~1261-1313) into a
`#[cube(launch)]` device kernel generic over `F: Float`, dispatched on the
resolved backend, then re-run the random ROCm oracle to confirm
mismatch_count==0 with the Type-2 device kernel genuinely exercised.

This is the explicit follow-up to quick-260529-gbf, which ported ONLY the Type-1
triple-product splice and deferred Type-2 (a structurally different two-`dgemm`
matmul). It mirrors the Type-1 host/device split EXACTLY:
- **Phase A** (adaptive Gauss-Chebyshev radial loop ~1158-1259: data-dependent
  convergence break, dynamic `nrs`, special functions) STAYS HOST as marshaling.
  It produces `rad_all` (per (ic,jc): `d3 = lilj1*lilc1*ljlc1` f64). Do NOT touch it.
- **Phase B** (the two-dgemm angular contraction ~1261-1313) MOVES to a
  `#[cube(launch)]` kernel. Host precomputes `angi`/`angj` via `type2_facs_ang`
  (marshaling, mirroring how Type-1 passes `ifac`/`jfac`) and passes them as f64
  buffers; the kernel does the bounded `i in 0..=li, j in 0..=lj` loop with the
  two matmuls, accumulating the `nfi*nfj` block; host scatters into `gctr` with `+=`.

Purpose: CLAUDE.md mandates CubeCL as the primary compute backend with host work
limited to planning/validation/marshaling. Type-2's angular splice is fully
bounded arithmetic (all bounds derive from comptime `li`/`lj`/`lc`; NO break/
continue, NO special functions) and therefore belongs on-device — same as Type-1.

Output: a new `ecp_type2_angular_kernel<F: Float + CubeElement>` `#[cube(launch)]`,
`run_ecp_type2_angular_device::<R>`, `run_ecp_type2_splice_on_backend`, wired
through `ecp_type2_cart` (and thus through `deriv1_cart_pair` /
`compute_type2_pair_grad`), with byte-identity preserved and the ROCm oracle green.

BYTE-IDENTITY IS NON-NEGOTIABLE: launch the device kernel at f64 with the SAME
loop/summation order as the host dgemm loops (~1285-1309) so the result is
bit-for-bit identical. ECP keeps f64 staging (no F32 outer dispatcher).
`launch_ecp` stays registered under canonical_family "ecp" routing
ecp/ecp_ipnuc/ecp_iprinv. NO new capi enum variants, NO legacy cint* wrappers.
</objective>

<execution_context>
@/home/user/Documents/workspace/cintx/.claude/get-shit-done/workflows/execute-plan.md
@/home/user/Documents/workspace/cintx/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/quick/260529-gbf-refactor-ecp-rs-to-cubecl-kernel-with-ge/260529-gbf-SUMMARY.md
@CLAUDE.md

# AUTHORITATIVE #[cube] authoring rules — read before touching the kernel:
@docs/manual/Cubecl/Cubecl_conditionals.md
@docs/manual/Cubecl/Cubecl_basic_operations.md
@docs/manual/Cubecl/Cubecl_generics.md

<interfaces>
<!-- Extracted from crates/cintx-cubecl/src/kernels/ecp.rs. The executor should -->
<!-- mirror the Type-1 template exactly. No codebase exploration needed. -->

Type-1 template to MIRROR (the structural blueprint for Type-2):
```rust
// per-backend dispatch wrapper (ecp.rs:831-863):
fn run_ecp_angular_splice_on_backend(backend: &ResolvedBackend, li: u32, lj: u32,
    rad_ang: &[f64], ifac: &[f64], jfac: &[f64], comps_i: &[u32], comps_j: &[u32]) -> Vec<f64>
// match { Cpu=>CpuRuntime, Wgpu=>WgpuRuntime, Cuda=>CudaRuntime, Rocm=>HipRuntime,
//         Metal=>WgpuRuntime } each #[cfg(feature=...)]-gated.

#[cube(launch)]
fn ecp_angular_kernel<F: Float + CubeElement>(rad_ang: &Array<F>, ..., cart_out: &mut Array<F>,
    li: u32, lj: u32, nfi: u32, nfj: u32)  // single work item: if UNIT_POS == 0u32 { ... }
//   - zeroes cart_out[0..nfi*nfj] with a `while` loop
//   - uses ONLY F arithmetic, u32 indices (`as usize` at index sites), statement-form `if`
//   - register scalar accumulation (`let mut acc = F::new(0.0); ... acc += ...`), NO local Array

fn run_ecp_angular_device<R: Runtime>(client: &ComputeClient<R>, ...) -> Vec<f64>
// f64 dispatch: create_from_slice(f64::as_bytes(..)) / launch::<f64,R> /
// read_one_unchecked / f64::from_bytes(&raw)[0..out_len].to_vec()
```

Type-2 HOST splice being replaced (ecp.rs:1272-1313) — the EXACT arithmetic the
new device kernel must reproduce bit-for-bit. Note the host materializes a `buf`
intermediate, but each `buf` entry is an independent inner sum, so it can be
recomputed inline WITHOUT a device-local Array (see Task 1 step 1):
```rust
// strides: dlc = lc*2+1; lilc1 = li+lc+1; ljlc1 = lj+lc+1; d2 = lilc1*ljlc1;
//          d3 = (li+lj+1)*d2; im = nfi*dlc; mq = dlc*ljlc1; di = nfi*nci.
// Host-side (STAYS host — marshaling): angi/angj via type2_facs_ang.
let mut angi = vec![0.0f64; (li + 1) * nfi * dlc * lilc1];
let mut angj = vec![0.0f64; (lj + 1) * nfj * dlc * ljlc1];
type2_facs_ang(&mut angi, li, lc, &rca);
type2_facs_ang(&mut angj, lj, lc, &rcb);
// MOVES to device — per (ic,jc):
for i in 0..=li { for j in 0..=lj {
  let a  = &prad[(i+j)*d2 .. +d2];                 // lilc1 x ljlc1 (col-major)
  let b  = &angi[i*nfi*dlc*lilc1 .. + lilc1*im];   // lilc1 x im   (col-major)
  let bj = &angj[j*nfj*dlc*ljlc1 .. + mq*nfj];     // mq    x nfj  (col-major)
  // dgemm 1 (N,N): buf[col*ljlc1+row2] = Σ_kk(0..lilc1) a[kk*ljlc1+row2]*b[col*lilc1+kk]
  //                col in 0..im, row2 in 0..ljlc1
  // dgemm 2 (T,N): gctr[c_off + col2*di + row] += common_fac
  //                  * Σ_kk2(0..mq) buf[row*mq+kk2]*bj[col2*mq+kk2]
  //                col2 in 0..nfj, row in 0..nfi;  c_off = jc*nfj*di + ic*nfi
}}
```

Helpers available (host-side, keep as marshaling — do NOT call from #[cube]):
- `fn ncart(l: u8) -> usize`                       (cartesian component count)
- `fn type2_facs_ang(facs: &mut [f64], li, lc, ri: &[f64;3])`  (angular factors)
- `fn coeffs_col_major(shell: &Shell) -> Vec<f64>` (already applied to ci/cj)
- `cint_common_fac_sp`, `LEVEL_MAX`, `LEVEL0` (used by Phase A; untouched)

Test harness already in ecp.rs `mod tests::ecp_angular_device_cross_check`
(#[cfg(feature="cpu")]): `cpu_client()`, `struct Lcg` (deterministic f64 in
[-1,1)), pattern `assert_eq!(max_diff, 0.0, ...)` + `any_nonzero`. Reuse it.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: ecp_type2_angular_kernel #[cube(launch)] + device dispatch + byte-identity & generic-f32 tests</name>
  <files>crates/cintx-cubecl/src/kernels/ecp.rs</files>
  <behavior>
    - f64 device-vs-host equivalence over (li,lj) up to (2,2) at several lc in
      {0,1,2}: max-abs-diff MUST be exactly 0.0 (identical f64 op order ⇒
      byte-identity). any_nonzero MUST be true (reference not all zeros).
    - generic-F: the SAME kernel at F=f32 on CpuRuntime reproduces the f64 result
      cast to f32 within f32 eps — proves it is genuinely generic over F: Float.
  </behavior>
  <action>
Add a NEW Type-2 device kernel ALONGSIDE the existing Type-1 `ecp_angular_kernel`
(do NOT modify the Type-1 kernel). Mirror the Type-1 template (ecp.rs:831-1063)
exactly, only changing the arithmetic body to the two-dgemm splice.

1. `#[cube(launch)] fn ecp_type2_angular_kernel<F: Float + CubeElement>(`
   `prad: &Array<F>, angi: &Array<F>, angj: &Array<F>, cart_out: &mut Array<F>,`
   `common_fac: F, li: u32, lj: u32, lc: u32, nfi: u32, nfj: u32)`. Single work
   item (`if UNIT_POS == 0u32 { ... }`). Inside:
   - Recompute strides from the runtime u32 args: `dlc = lc*2+1`,
     `lilc1 = li+lc+1`, `ljlc1 = lj+lc+1`, `d2 = lilc1*ljlc1`, `d3 = (li+lj+1)*d2`,
     `im = nfi*dlc`, `mq = dlc*ljlc1`. The kernel computes ONE (ic=0,jc=0) block,
     so the per-tuple output stride `di = nfi` and the output index is `col2*nfi + row`
     (F-order `[ao_i, ao_j]`), mirroring how the Type-1 kernel emits a single tuple.
   - Zero `cart_out[0..nfi*nfj]` with a `while` loop (mirror Type-1 lines 929-935).
   - **NO device-local `Array` scratch.** The codebase has no proven device-local
     `Array::new` precedent (center_2c2e/3c2e/4c1e accumulate in registers). Preserve
     the host `buf` two-dgemm summation order by RECOMPUTING each `buf` entry inline
     at the moment dgemm-2 consumes it — this is bit-identical because each `buf`
     entry is an independent inner sum (over `kk in 0..lilc1`) and the outer dgemm-2
     sum (over `kk2 in 0..mq`) is preserved in the same order. Structure (statement-
     form `while`, `u32` counters, `as usize` at index sites):
       ```
       // outer i in 0..=li, j in 0..=lj  (a/b/bj base offsets recomputed per i,j)
       //   for row  in 0..nfi:            (dgemm-2 row)
       //     for col2 in 0..nfj:          (dgemm-2 col)
       //       acc = F::new(0.0)
       //       for kk2 in 0..mq:          (dgemm-2 contraction; mq = dlc*ljlc1)
       //         // buf[row*mq+kk2] == dgemm-1 buf[col*ljlc1+row2] with
       //         //   col = row*dlc + kk2/ljlc1 ,  row2 = kk2 % ljlc1
       //         let col  = row*dlc + kk2/ljlc1
       //         let row2 = kk2 - (kk2/ljlc1)*ljlc1
       //         bufv = F::new(0.0)
       //         for kk in 0..lilc1:      (dgemm-1 inner; SAME order as host)
       //           bufv += a[kk*ljlc1+row2] * b[col*lilc1+kk]
       //         acc += bufv * bj[col2*mq+kk2]
       //       cart_out[col2*nfi+row] += common_fac * acc   // ACCUMULATE over i,j
       ```
     where `a = prad` base `(i+j)*d2`, `b = angi` base `i*nfi*dlc*lilc1`,
     `bj = angj` base `j*nfj*dlc*ljlc1`. **The host loop accumulates into `gctr`
     across all (i,j) pairs via `+=` (line 1307); the device kernel must likewise
     `+=` into `cart_out` across the i/j loop** (zero it once up front, then `+=`).
     Verify the (col/row2) decomposition matches `mq = dlc*ljlc1` exactly: integer
     `kk2/ljlc1 ∈ 0..dlc` and `kk2%ljlc1 ∈ 0..ljlc1`, so `col = row*dlc + (kk2/ljlc1)`
     ranges over `row*dlc .. row*dlc+dlc ⊆ 0..im`. Index `Array<F>` with `as usize`
     (Phase 8 P02 established pattern; see STATE Decisions). NO if-EXPRESSIONS
     (statement form only), NO break/continue, NO plain-fn calls, NO special functions.
   - Fold `common_fac` in-kernel (`common_fac: F`, `cart_out += common_fac*acc`)
     exactly as the host does (`gctr += common_fac * s`), so the host scatter stays
     a trivial `+=` like Type-1. Do NOT pre-scale `prad`/`angi`/`angj`.

2. `fn run_ecp_type2_angular_device<R: Runtime>(client, li, lj, lc, prad: &[f64],`
   `angi: &[f64], angj: &[f64], common_fac: f64) -> Vec<f64>`: mirror
   `run_ecp_angular_device` (ecp.rs:1008-1063). f64 dispatch
   (`create_from_slice(f64::as_bytes(..))`, `launch::<f64, R>`, pass `common_fac`
   as the f64 scalar arg, `read_one_unchecked`, `f64::from_bytes(&raw)[0..out_len]`).
   Add `debug_assert_eq!` length checks (T-hin-01): `prad.len()==d3`,
   `angi.len()==(li+1)*nfi*dlc*lilc1`, `angj.len()==(lj+1)*nfj*dlc*ljlc1` — derived
   from the SAME li/lj/lc formulas as the host driver. `out_len = nfi*nfj`.

3. `#[allow(clippy::too_many_arguments)] fn run_ecp_type2_splice_on_backend(`
   `backend: &ResolvedBackend, li, lj, lc, prad, angi, angj, common_fac) -> Vec<f64>`:
   mirror `run_ecp_angular_splice_on_backend` (ecp.rs:831-863) — per-backend match
   (`Cpu=>CpuRuntime`, `Wgpu=>WgpuRuntime`, `Cuda=>CudaRuntime`, `Rocm=>HipRuntime`,
   `Metal=>WgpuRuntime`), each `#[cfg(feature=...)]`-gated.

4. Tests: extend `mod tests::ecp_angular_device_cross_check` (already
   `#[cfg(feature="cpu")]`). Add `host_type2_splice(li,lj,lc,prad,angi,angj,
   common_fac)` reproducing ecp.rs:1272-1309 EXACTLY for a single (ic=0,jc=0) tuple
   (the full materialized two-dgemm with `buf`), plus
   `assert_type2_f64_byte_identity(li,lj,lc,seed)` (asserts max-abs-diff == 0.0,
   any_nonzero == true) over (li,lj) in {(0,0),(1,0),(0,1),(1,1),(2,0),(0,2),(2,1),
   (1,2),(2,2)} × lc in {0,1,2}; and `ecp_type2_angular_device_generic_f32_within_eps`
   mirroring the Type-1 f32 test. Generate `angi`/`angj`/`prad`/`common_fac` with the
   existing `Lcg`. Mark the two new tests `#[test]`.

Do NOT change `launch_ecp` registry routing, the f64 staging signature, the Phase-A
radial loop (~1158-1259), or `type2_facs_ang`. This task ONLY adds the kernel +
dispatch + tests; wiring into `ecp_type2_cart` is Task 2.
  </action>
  <verify>
    <automated>cargo test -p cintx-cubecl --features cpu --lib ecp_type2_angular 2>&1 | tail -25</automated>
  </verify>
  <done>ecp_type2_angular_kernel, run_ecp_type2_angular_device, run_ecp_type2_splice_on_backend exist; the f64 device-vs-host byte-identity test (max-abs-diff==0.0, any_nonzero==true) and the generic-f32 test pass under --features cpu. Commit: `feat(260529-hin): ecp_type2_angular_kernel #[cube(launch)] + device dispatch + byte-identity tests`.</done>
</task>

<task type="auto">
  <name>Task 2: Rewire ecp_type2_cart through the device path; remove `let _ = backend`; preserve byte-identity</name>
  <files>crates/cintx-cubecl/src/kernels/ecp.rs</files>
  <action>
Replace the host two-dgemm loop in `ecp_type2_cart` (ecp.rs:1272-1313) with a
per-(ic,jc) device dispatch, mirroring how Task 2 of 260529-gbf rewired
`ecp_type1_cart` (ecp.rs:806-822).

1. Remove `let _ = backend;` at ecp.rs:1105.
2. Keep the host-side `type2_facs_ang` computation of `angi`/`angj` (~1264-1267)
   and the `common_fac` (~1125-1126) — these stay host (marshaling). The local
   `buf` allocation (~1272) is no longer needed; remove it.
3. Replace the `for ic / for jc` body (~1273-1312) with: for each (ic,jc), slice
   `prad = &rad_all[(ic*ncj+jc)*d3 .. +d3]`, call
   `run_ecp_type2_splice_on_backend(backend, li as u32, lj as u32, lc as u32, prad,
   &angi, &angj, common_fac)` → returns the `nfi*nfj` block (F-order `col2*nfi+row`,
   already `common_fac`-scaled in-kernel), then scatter into the full
   contraction-major `gctr` with `+=` at `c_off = jc*nfj*di + ic*nfi`:
   `gctr[c_off + col2*di + row] += block[col2*nfi + row]` for `col2 in 0..nfj`,
   `row in 0..nfi` (di = nci*nfi). This matches the host scatter target exactly
   (host wrote `gctr[c_off + col*di + row] += common_fac*s`). Because `common_fac`
   is folded in the kernel, the scatter is a plain `+=` with NO extra multiply —
   confirm no double-multiply.
4. The gradient drivers (`compute_type2_pair_grad`, `deriv1_cart_pair` at the
   `lc >= 0` branch ~1464) already call `ecp_type2_cart(backend, ...)`, so they
   become device-backed automatically once the host loop is replaced — verify the
   `backend` thread reaches them (it already does; no signature change needed).
5. Update the ecp.rs module-doc Type-2 paragraph (~lines 52-55): change "The Type-2
   angular splice (a two-`dgemm` matmul) remains host-side this task; a Type-2
   matmul device kernel is a follow-up." to state Type-2's two-dgemm angular splice
   now ALSO runs on-device via `run_ecp_type2_splice_on_backend` (generic over F,
   f64-internal, byte-identity preserved); only the Phase-A adaptive radial machinery
   stays host-side as marshaling.
6. Run the FULL ecp unit suite and the vendor CPU byte-identity parity gate to prove
   no regression (atol=1e-12/rtol=0.0). The vendor gate is double-gated: it needs
   `--features cpu` AND `CINTX_ORACLE_BUILD_VENDOR=1` or parity SILENTLY skips
   (memory: oracle_vendor_parity_invocation).
  </action>
  <verify>
    <automated>cargo test -p cintx-cubecl --features cpu --lib ecp 2>&1 | tail -15 && CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test safe_api_ecp_parity --test ecp_iprinv_parity 2>&1 | tail -15</automated>
  </verify>
  <done>`let _ = backend;` removed from ecp_type2_cart; Type-2 splice routes through run_ecp_type2_splice_on_backend; all cintx-cubecl ecp --lib tests pass; vendor CPU parity (safe_api_ecp_parity + ecp_iprinv_parity) passes at atol=1e-12/rtol=0.0 with the vendor build actually compiled+run (NOT skipped — confirm output shows vendor cases ran, not "0 passed; 0 filtered"); module-doc updated. Commit: `feat(260529-hin): rewire ecp_type2_cart through #[cube] device splice (byte-identity preserved)`.</done>
</task>

<task type="auto">
  <name>Task 3: Re-run ROCm oracle; confirm mismatch_count==0 AND Type-2 device path is genuinely exercised</name>
  <files>crates/cintx-cubecl/src/kernels/ecp.rs</files>
  <action>
Run the existing random ROCm ECP idempotency oracle on the gfx1152 GPU and report
the OBSERVED mismatch_count + case count. The Cu/LANL2DZ fixture has BOTH Local
(Type-1) AND Projected (Type-2) channels (260529-gbf SUMMARY: "Local (Type-1) +
Projected (Type-2) channels fire"), so once `ecp_type2_cart` is device-backed
(Task 2), `int1e_ecp_sph` drives the NEW Type-2 device kernel on every case.

1. Run: `cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle`. This sets
   `CINTX_ROCM_ORACLE=1` and runs `test_ecp_sph_random_rocm_idempotency` (and the
   other rocm-base families). Capture the observed
   `mismatch_count=N across M cases (any_nonzero=true)` line VERBATIM.
2. **PROVE Type-2 is genuinely exercised — do NOT fabricate.** The `any_nonzero`
   gate alone cannot distinguish Type-1 from Type-2 contributions. Establish Type-2
   is hit by ONE of (prefer the first that is cheap and non-invasive):
   (a) Read the Cu/LANL2DZ EcpShell fixture (the LANL2DZ slab used by
       `build_random_ecp_system` in ecp_random_rocm_parity.rs) and confirm it
       contains Projected channels with `l >= 0` (non-local projectors) — the
       `(false,false) => ecp_type2_cart` / `(true,false) => compute_type2_pair_grad`
       arms at ecp.rs ~1858/1877 fire iff a Projected channel exists. Cite the
       specific EcpShell entries (Projected angular momenta present). The fixture
       includes the LANL2DZ ECP slab verbatim with BSE general-contraction blocks
       split into single-NCTR bas rows (STATE Decisions / memory note), so the
       non-local channels are present.
   OR (b) add a temporary `eprintln!`/thread-local hit counter at the top of
       `run_ecp_type2_splice_on_backend` gated behind a `CINTX_ECP_TYPE2_TRACE` env
       var, run the oracle with it set, observe a non-zero Type-2 dispatch count,
       then REMOVE the temporary probe before the final commit (leave the kernel
       clean). Record the observed count in SUMMARY.
   Document in the SUMMARY which method was used and the concrete evidence (shell
   entries cited, or observed dispatch count). If method (a): no code change in this
   task beyond confirming/documenting; if (b): ensure the probe is removed.
3. Report in the SUMMARY's "Observed ROCm GPU result" block: exact mismatch_count,
   case count, any_nonzero, and the Type-2-exercised evidence. This is the
   load-bearing claim — quote the harness output, do not paraphrase.
  </action>
  <verify>
    <automated>cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle 2>&1 | tail -40</automated>
  </verify>
  <done>`cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle` was actually executed on gfx1152; the harness printed `mismatch_count=0 across M cases (any_nonzero=true)` for int1e_ecp_sph and the SUMMARY records the observed M and the concrete evidence that the Type-2 device kernel was exercised (cited Projected shell entries OR observed dispatch count, with any temporary probe removed). No source change required if method (a); commit only if a probe was added/removed: `chore(260529-hin): confirm Type-2 device path exercised by ROCm oracle`.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| host → device buffer marshaling | `prad`/`angi`/`angj` lengths computed host-side from li/lj/lc cross to the device kernel; an undersized buffer = out-of-bounds device read |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-hin-01 | Tampering | `run_ecp_type2_angular_device` host→device buffer sizing | mitigate | `debug_assert_eq!` on `prad`/`angi`/`angj`/`out_len` derived from the SAME li/lj/lc formulas as the host driver; covered by the device-vs-host f64 equivalence test which exercises the full length contract. |
| T-hin-02 | Tampering | in-kernel `(col,row2)` decomposition of the dgemm-2 contraction index `kk2` | mitigate | Integer-division decomposition (`col = row*dlc + kk2/ljlc1`, `row2 = kk2 % ljlc1`) is bounds-checked by construction (`col ∈ 0..im`, `row2 ∈ 0..ljlc1`); the f64 byte-identity test against the host materialized-`buf` reference proves the inline recomputation is bit-exact. |
| T-hin-03 | Repudiation | ROCm oracle "Type-2 exercised" claim | accept | The proof is established by cited fixture shell entries or an observed dispatch count recorded in SUMMARY; no persistent audit log needed for a single-developer verification gate. |
</threat_model>

<verification>
- `cargo test -p cintx-cubecl --features cpu --lib ecp` — all ECP unit tests pass,
  including the new Type-2 device-vs-host f64 byte-identity (max-abs-diff==0.0) and
  generic-f32 tests.
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test safe_api_ecp_parity --test ecp_iprinv_parity` — vendor CPU byte-identity parity stays green at atol=1e-12/rtol=0.0 (vendor build actually compiled + run, NOT skipped).
- `cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle` — observed
  mismatch_count==0 across M cases, any_nonzero=true, on gfx1152.
- Type-2 device path genuinely exercised (cited evidence in SUMMARY).
</verification>

<success_criteria>
- Type-2 two-dgemm angular splice runs in `ecp_type2_angular_kernel` `#[cube(launch)]`
  generic over `F: Float`, dispatched per-backend via `run_ecp_type2_splice_on_backend`.
- `let _ = backend;` removed from `ecp_type2_cart`; the gradient drivers
  (`compute_type2_pair_grad` / `deriv1_cart_pair`) are device-backed transitively.
- f64 byte-identity preserved: device-vs-host max-abs-diff==0.0; vendor CPU parity
  green at atol=1e-12/rtol=0.0.
- ROCm oracle: mismatch_count==0, any_nonzero=true, Type-2 device path proven hit.
- No new capi enum variants, no legacy cint* wrappers; `launch_ecp` registry routing
  and the f64 staging signature unchanged. Module-doc Type-2 paragraph updated.
</success_criteria>

<output>
After completion, create `.planning/quick/260529-hin-port-the-ecp-type-2-two-dgemm-angular-sp/260529-hin-SUMMARY.md`
</output>
