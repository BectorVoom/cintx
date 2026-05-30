---
phase: quick-260529-imi
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/cintx-cubecl/src/kernels/one_electron.rs
  - crates/cintx-oracle/tests/one_electron_random_rocm_parity.rs
autonomous: true
requirements: []
must_haves:
  truths:
    - "int1e overlap, kinetic, and nuclear-attraction are computed by a #[cube(launch)] device kernel (not the host VRR/HRR/contraction pipeline)"
    - "The device kernel is generic over F: Float + CubeElement and is dispatched at f64 via run_::<R> on the resolved backend, including ROCm HipRuntime"
    - "The gradient operators (ipovlp/ipkin/ipnuc/iprinv) and the spinor representation retain their existing behavior (no silent narrowing) with an explicit in-code rationale for staying host-side"
    - "The existing one_electron_parity.rs scalar parity tests (ovlp/kin/nuc, CPU + vendor) still pass byte/atol-identically through the new device path"
    - "A new random ROCm oracle parity test drives int1e_{ovlp,kin,nuc}_sph twice on the GPU and asserts mismatch_count=0 with non-zero output"
  artifacts:
    - path: "crates/cintx-cubecl/src/kernels/one_electron.rs"
      provides: "one_electron_scalar_kernel #[cube(launch)] + run_1e_scalar_device<R> + on-backend dispatch wired into launch_one_electron_typed scalar path"
      contains: "#[cube(launch)]"
    - path: "crates/cintx-oracle/tests/one_electron_random_rocm_parity.rs"
      provides: "Randomized ROCm idempotency oracle for int1e_{ovlp,kin,nuc}_sph"
      contains: "CINTX_ROCM_ORACLE"
  key_links:
    - from: "launch_one_electron_typed scalar path"
      to: "one_electron_scalar_kernel via run_1e_scalar_device"
      via: "ResolvedBackend match dispatch (Cpu/Rocm/Wgpu/Cuda/Metal)"
      pattern: "run_1e_scalar_device::<"
    - from: "one_electron_random_rocm_parity.rs"
      to: "eval_raw on rocm backend"
      via: "RawApiId::INT1E_OVLP_SPH/KIN_SPH/NUC_SPH evaluated twice"
      pattern: "eval_raw"
---

<objective>
Refactor the scalar 1e operators (overlap, kinetic, nuclear-attraction) in
`crates/cintx-cubecl/src/kernels/one_electron.rs` from the host-side VRR/HRR/
contraction pipeline into a single `#[cube(launch)]` device kernel generic over
`F: Float`, dispatched at f64 via a `run_1e_scalar_device::<R>` helper on the
resolved backend (Cpu, ROCm HipRuntime, Wgpu, Cuda, Metal) — exactly the proven
center_2c2e.rs GPU-port template. Then add a randomized ROCm oracle parity test
that drives `int1e_{ovlp,kin,nuc}_sph` twice on the GPU and asserts
mismatch_count=0.

Purpose: bring the 1e family onto the CubeCL device backend (project constraint:
CubeCL is the primary compute backend; host CPU work stays limited to planning,
validation, marshaling, transforms). This matches the prior 2c2e/3c2e/ECP/f12
device-kernel ports.

Output:
- A device `one_electron_scalar_kernel<F>` + `run_1e_scalar_device<R>` +
  per-backend dispatch wired into the scalar path of `launch_one_electron_typed`.
- A new `one_electron_random_rocm_parity.rs` ROCm idempotency oracle.

Scope decision (explicit, NOT a silent narrowing — mirrors how 3c2e/ECP scoped
their ports): this task ports the THREE scalar operators (overlap/kinetic/nuclear)
on-device. The gradient operators (ipovlp/ipkin/ipnuc/iprinv) and the spinor
representation KEEP their existing host code paths unchanged and continue to pass
their existing tests. Porting the four derivative kernels (each a distinct
nabla1i/D_j^2 mixing pipeline with +1/+2 angular headroom) plus the spinor
transform on-device would exceed a single quick-task context budget; the constraint
explicitly permits "scalar operators at minimum on-device" with the rest stated as
host-side. The kernel rejects nothing it rejected before — it only changes HOW the
scalar arms are computed. The reason is recorded in code at the dispatch fork.
</objective>

<execution_context>
@/home/user/Documents/workspace/cintx/.claude/get-shit-done/workflows/execute-plan.md
@/home/user/Documents/workspace/cintx/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md

# THE FILE TO PORT (scalar path lives at lines ~1085-1264; helpers at 41-631)
@crates/cintx-cubecl/src/kernels/one_electron.rs

# REFERENCE TEMPLATE — the canonical scalar device-kernel port (study this most)
@crates/cintx-cubecl/src/kernels/center_2c2e.rs

# REFERENCE — most recent base-Cartesian #[cube(launch)] + run_::<R> + on-backend dispatch
@crates/cintx-cubecl/src/kernels/f12.rs

# REFERENCE — the random ROCm oracle test to model the new test on
@crates/cintx-oracle/tests/ecp_random_rocm_parity.rs
@crates/cintx-oracle/tests/f12_random_rocm_parity.rs

# Existing scalar parity test that MUST still pass through the new device path
@crates/cintx-oracle/tests/one_electron_parity.rs

# Device-capable recurrence helpers (already #[cube], generic over F: Float)
@crates/cintx-cubecl/src/math/obara_saika.rs

# Authoritative #[cube] authoring rules — READ before writing the kernel
@docs/manual/Cubecl/Cubecl_generics.md
@docs/manual/Cubecl/cubecl_macro_fanout_manual.md
@docs/manual/Cubecl/Cubecl_conditionals.md

<interfaces>
<!-- Contracts the executor needs; extracted from the codebase. Use directly, no exploration. -->

The #[cube(launch)] launch + readback + per-backend dispatch template (from
center_2c2e.rs::center_2c2e_kernel / run_2c2e_device and f12.rs::run_f12_*):

  #[cube(launch)]
  fn KERNEL<F: Float + CubeElement>(... &Array<F> ins, &mut Array<F> g, &mut Array<F> urys,
                                    &mut Array<F> wrys, &mut Array<F> cart_out, scalars...,
                                    #[comptime] nroots: u32) { if UNIT_POS == 0u32 { ... } }

  KERNEL::launch::<f64, R>(client, CubeCount::Static(1,1,1), CubeDim::new_1d(1),
      unsafe { ArrayArg::from_raw_parts(handle, len) }, ... );
  let raw = client.read_one_unchecked(out_h);
  f64::from_bytes(&raw)[0..out_len].to_vec()

Per-backend dispatch (the exact arms to copy — all already present in 2c2e/f12):
  match backend {
    #[cfg(feature="cpu")]   ResolvedBackend::Cpu(c)       => run_..::<cubecl::cpu::CpuRuntime>(c, ..),
    #[cfg(feature="wgpu")]  ResolvedBackend::Wgpu(c,_)    => run_..::<cubecl_wgpu::WgpuRuntime>(c, ..),
    #[cfg(feature="cuda")]  ResolvedBackend::Cuda(c)      => run_..::<cubecl_cuda::CudaRuntime>(c, ..),
    #[cfg(feature="rocm")]  ResolvedBackend::Rocm(c)      => run_..::<cubecl_hip::HipRuntime>(c, ..),
    #[cfg(feature="metal")] ResolvedBackend::Metal(c,_)   => run_..::<cubecl_wgpu::WgpuRuntime>(c, ..),
  }

Device recurrence helpers already #[cube] generic over F: Float (call from inside
the #[cube] kernel — they are #[cube] fns, so this is allowed, NOT a plain-fn call):
  pub fn vrr_step<F: Float>(g: &mut Array<F>, rijrx: F, aij2: F, nmax: u32, stride: u32)   // 1e overlap VRR
  pub fn hrr_step<F: Float>(g: &mut Array<F>, rirj: F, di: u32, dj: u32, li_max: u32, lj: u32)
  pub fn vrr_2e_step<F: Float>(g: &mut Array<F>, c00: F, b10: F, nmax: u32, stride: u32)    // nuclear root-VRR
  (host f64 mirrors: vrr_step_host / hrr_step_host / vrr_2e_step_host — keep for the host-vs-device unit cross-check.)

Device Rys roots (already #[cube], used by 2c2e). Nuclear attraction needs Rys
roots ON-DEVICE; the host path uses rys_roots_host(nrys, x). The device kernels are:
  use crate::math::rys::{rys_root1, rys_root2, rys_root3, rys_root4, rys_root5};
  rys_rootN::<F>(x_rys: F, urys: &mut Array<F>, wrys: &mut Array<F>, pie4: F);
  const PIE4: f64 = 0.78539816339744827900;
  Select with comptime! on nroots (nuclear: nrys = (li+lj)/2 + 1; overlap/kinetic: nrys=1, fixed-center VRR, no Rys).

Constants/transforms (stay host-side — coefficient tables are host-only):
  const SQRTPI: f64 = 1.7724538509055159;
  fn common_fac_sp(l: u8) -> f64   // already in one_electron.rs
  crate::transform::c2s::{cart_to_sph_1e, ncart, nsph}
  crate::transform::c2spinor::cart_to_spinor_sf_2d   // spinor path, host

Pair data (host marshaling — compute on host, pass scalars into the kernel):
  crate::math::pdata::compute_pdata_host(ai, aj, rix,riy,riz, rjx,rjy,rjz, norm_i, norm_j) -> PairData
  PairData fields used: zeta_ab, aij2, fac, center_p_{x,y,z}

Raw API ids the oracle test drives (scalar 1e, all base profile):
  cintx_compat::raw::RawApiId::{INT1E_OVLP_SPH, INT1E_KIN_SPH, INT1E_NUC_SPH}, eval_raw
  (the 1e family is registered unconditionally as "1e" in kernels/mod.rs — BASE profile, no feature gate)
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Port the scalar 1e operators to a #[cube(launch)] device kernel generic over F</name>
  <files>crates/cintx-cubecl/src/kernels/one_electron.rs, crates/cintx-cubecl/src/math/obara_saika.rs</files>
  <behavior>
    - Device-vs-host cross-check (CpuRuntime, f64), modeled on center_2c2e.rs
      `assert_device_matches_host`: for overlap, kinetic, nuclear the new
      `run_1e_scalar_device::<CpuRuntime>` output must equal the existing host
      `contract_overlap`/`contract_kinetic`/`contract_nuclear` reference within
      atol=1e-12 + rtol=1e-10, for li,lj in {(0,0),(0,1),(1,0),(1,1),(2,2)}.
    - Genericity evidence: the kernel compiles and runs for F=f32 (s-s overlap on
      CpuRuntime returns a finite positive value) — same shape as
      `test_center_2c2e_kernel_generic_f32`.
    - Regression: the existing `launch_one_electron_typed::<f64>` overlap test
      (`test_precision_dispatch_f64_inner_positive_overlap`) and the general-
      contraction tests (`test_general_contraction_s_parity`,
      `..._p_parity_contraction_major`) still pass — the device path must produce
      the SAME Cartesian blocks the host pipeline did (contraction-major,
      bra-fastest column-major layout preserved).
  </behavior>
  <action>
    Port the scalar arms (overlap / kinetic / nuclear-attraction) of
    `launch_one_electron_typed` (the block after the gradient `return`, lines
    ~1085-1264) onto the device, following the center_2c2e.rs template EXACTLY:

    1. Write `#[cube(launch)] fn one_electron_scalar_kernel<F: Float + CubeElement>(...)`.
       The kernel runs a single work item guarded by `if UNIT_POS == 0u32`. It
       iterates the primitive pairs (pi,pj) and contraction pairs (ci,cj) in-kernel
       (mirror the 2c2e kernel's nprim/nctr loops) and accumulates ONE nci*ncj
       Cartesian block per (ci,cj) contraction pair into `cart_out`, laid out
       contraction-major / bra-fastest exactly as the host scalar path does
       (`out[ck_idx*nci + ci_idx]` per block; block base `(ci*nctr_j+cj)*block_len`).
       Pass an `#[comptime] op_kind: u32` (0=overlap, 1=kinetic, 2=nuclear) so the
       three operator branches are comptime-selected (no runtime operator dispatch
       inside the hot path — same discipline as comptime nroots in 2c2e).

       - Overlap branch: build the fixed-center overlap G-tensor with
         `nmax=li+lj`, `lj` HRR levels, base case `gz0 = fac*SQRTPI*PI/(aij*sqrt(aij))`,
         then `vrr_step::<F>` (3 axes) + `hrr_step::<F>` (3 axes), then the triple-
         product contraction. (Faithful device port of `fill_g_tensor_overlap` +
         `contract_overlap`.)
       - Kinetic branch: build the overlap G-tensor with `nmax=li+lj+2`, `lj+2`
         HRR levels, then in-kernel compute the second ket-derivative
         `g3 = jx*(jx-1)*g0[jx-2] - 2*aj*(2*jx+1)*g0[jx] + 4*aj^2*g0[jx+2]` per axis
         and `T = -0.5*(g3x*g0y*g0z + g0x*g3y*g0z + g0x*g0y*g3z)`. (Faithful device
         port of `contract_kinetic`; reuse the same lj_ext=lj+2 / dj=nmax+1 strides.)
       - Nuclear branch: comptime-select `rys_rootN::<F>` for
         `nrys=(li+lj)/2+1` (PIE4 passed in). For each atom (passed as flat
         coords[] + charges[] arrays) and each Rys root, compute tau/rt/c00, build
         the root G-tensor with `vrr_2e_step::<F>` (3 axes) + `hrr_step::<F>`, and
         accumulate the triple product weighted by `fac1*w_n`. (Faithful device
         port of `contract_nuclear`.) Atoms accumulated in passed order for
         bit-stable reduction.

       Scratch `g` / `urys` / `wrys` and the `cart_out` accumulator are passed as
       `&mut Array<F>` and zeroed in-kernel before use, exactly like 2c2e.

       OBEY THE CUBECL MANUALS (verify each against the three manual files in
       context): only `#[cube]` fn calls from inside the kernel (vrr_step,
       hrr_step, vrr_2e_step, rys_rootN are all `#[cube]` — OK); no if-EXPRESSIONS
       (use statement-form if/else writing into a mut binding, as 2c2e does for
       the per-axis displacement `d`); `F::exp`/`F::sqrt` (not `.sqrt()`); u32/i32
       loop counters only; no `continue`/`break`; `as usize` on every Array index.

    2. Write `fn run_1e_scalar_device<R: Runtime>(client, ... , #[runtime] op_kind, nroots, li, lj, nprim, nctr, ri, rj, fac/zeta scalars, exps, coeffs, atom_coords, atom_charges) -> Vec<f64>`
       that allocates the input/scratch/output handles (`client.create_from_slice(f64::as_bytes(..))`),
       launches `one_electron_scalar_kernel::launch::<f64, R>(...)` at
       `CubeCount::Static(1,1,1)` / `CubeDim::new_1d(1)`, and reads back the
       `cart_out` buffer via `client.read_one_unchecked` + `f64::from_bytes`.
       Compute `op_kind`/`nroots`/G-tensor sizing on the host and pass them in
       (op_kind and nroots are the `#[comptime]` args — branch on them at the
       `launch::<f64,R>` call site with a small host-side match if comptime args
       cannot be passed dynamically, exactly as 2c2e does NOT need to because it
       passes nroots as a comptime generic; replicate 2c2e's approach: pass nroots
       as the comptime arg and select op_kind likewise).

    3. Add `fn run_1e_scalar_on_backend(backend, ...) -> Vec<f64>` with the 5-arm
       `match backend { Cpu => CpuRuntime, Wgpu => WgpuRuntime, Cuda => CudaRuntime,
       Rocm => HipRuntime, Metal => WgpuRuntime }` dispatch (copy verbatim from
       f12.rs::run_f12_cart_contraction_on_backend / 2c2e launcher).

    4. Rewire the SCALAR path of `launch_one_electron_typed` (overlap/kinetic/
       nuclear) to compute `cart_blocks` by calling `run_1e_scalar_on_backend`
       instead of the host primitive loop. Keep the host code (`fill_g_tensor_overlap`,
       `contract_overlap`, `contract_kinetic`, `contract_nuclear`,
       `vrr/hrr/vrr_2e *_host`) as the `#[cfg(test)]` host reference for the
       cross-check (mirror how 2c2e kept `fill_g_tensor_2c2e` under `#[cfg(test)]`).
       Leave the `common_fac_sp` sp-scale, the cart_to_sph_1e / spinor / cart
       staging scatter, and the not0 sentinel UNCHANGED and host-side (transform
       tables are host-only — same as every prior port).

    5. Do NOT touch the gradient path (`is_ipovlp||is_ipkin||is_ipnuc||is_iprinv`
       block) or the spinor branch. Add a 2-3 line `//` comment at the scalar/
       gradient fork stating: gradient + spinor 1e stay host-side this task (each
       is a distinct nabla1i/D_j^2 derivative pipeline; on-device port deferred to
       a follow-up quick task — scalar-at-minimum scoping per the task constraint,
       matching how 3c2e/ECP staged their ports). They keep passing their existing
       tests unchanged.

    If `nroots > 5` for nuclear (li+lj > 8), return the existing typed error path
    (the device Rys kernels only cover nroots<=5 — same MAX_DEVICE_NROOTS guard as
    2c2e). H2O/STO-3G stays well within nroots<=5.

    Run `cargo fmt` and `cargo clippy -p cintx-cubecl --features cpu` before finishing.
  </action>
  <verify>
    <automated>cargo test -p cintx-cubecl --features cpu --lib kernels::one_electron 2>&1 | tail -25</automated>
  </verify>
  <done>
    `one_electron.rs` contains a `#[cube(launch)] one_electron_scalar_kernel<F: Float + CubeElement>`,
    a `run_1e_scalar_device<R: Runtime>`, and a 5-arm `run_1e_scalar_on_backend`.
    The scalar path of `launch_one_electron_typed` computes via the device kernel.
    All in-crate `kernels::one_electron` tests pass (device-vs-host cross-check for
    ovlp/kin/nuc, the f32 genericity smoke test, the f64 overlap dispatch test, and
    both general-contraction parity tests). Gradient + spinor paths untouched.
    clippy clean.
  </done>
</task>

<task type="auto">
  <name>Task 2: Add and RUN the randomized ROCm oracle parity test for int1e_{ovlp,kin,nuc}_sph</name>
  <files>crates/cintx-oracle/tests/one_electron_random_rocm_parity.rs</files>
  <action>
    Create `crates/cintx-oracle/tests/one_electron_random_rocm_parity.rs` modeled
    on `crates/cintx-oracle/tests/ecp_random_rocm_parity.rs` and
    `f12_random_rocm_parity.rs`. Because the 1e family is in the BASE profile (no
    feature gate, unlike f12), gate it `#![cfg(feature = "rocm")]` ONLY (single
    gate — copy the ECP test's gating, NOT the f12 `all(rocm, with-f12)` double
    gate). It is then picked up by `xtask rocm-oracle` BASE profile's
    `cargo test -p cintx-oracle --features rocm -- --ignored`.

    Structure (copy the ECP/f12 idempotency shape exactly):
    - Reuse the deterministic `Lcg` (Numerical Recipes constants) for reproducible
      randomization without an external rng crate.
    - The 1e family has RAW API enum variants (unlike ECP), so drive it through the
      RAW API like the f12 test does: `eval_raw` with
      `RawApiId::{INT1E_OVLP_SPH, INT1E_KIN_SPH, INT1E_NUC_SPH}`. Build the
      `atm`/`bas`/`env` H2O STO-3G slab the same way `one_electron_parity.rs`
      does (reuse `build_h2o_sto3g` — copy the builder into this file or import it
      if it is exported), then per case randomize the primitive exponents (into a
      physically sane range, e.g. scale each by a `Lcg::uniform(0.7, 1.4)` jitter)
      and/or the H atom coordinates by ±0.3 bohr, keeping the slab structurally
      valid (same nbas/natm/slot layout). Keep `env[PTR_ENV_START..]` valid.
    - `#[test] #[ignore]` with the runtime gate assert:
      `assert_eq!(std::env::var("CINTX_ROCM_ORACLE").as_deref(), Ok("1"), "...")`
      (copy verbatim from ECP — blocks direct `cargo test --features rocm -- --ignored`).
    - For each of the three operators, over `n_cases = 48`, evaluate the full
      shell-pair matrix TWICE via `eval_raw` on the ROCm backend (backend resolved
      from `CINTX_BACKEND=rocm` exported by xtask) and compare the two runs:
      `diff = (o-r).abs() > atol + rtol*r.abs()` with `atol=1e-12, rtol=1e-10`,
      incrementing `mismatch_count`. Track `any_nonzero`.
    - Final asserts: `assert_eq!(mismatch_count, 0, ...)` and
      `assert!(any_nonzero, "device output all zeros — kernel did not run on GPU")`,
      then `println!("  PASS: rocm int1e_{{ovlp,kin,nuc}}_sph random idempotency
      mismatch_count=0 across {{n_cases}} cases ...")`.

    Build-gate the file compiles under base: confirm with
    `cargo test -p cintx-oracle --features rocm --no-run 2>&1 | tail` (compile only;
    no ROCm device needed to compile). Also confirm it still compiles WITHOUT rocm
    (the `#![cfg]` makes it empty): `cargo build -p cintx-oracle 2>&1 | tail`.

    THEN RUN THE TEST ON THE ROCm BACKEND (the required final step). Invoke the
    canonical path:

      cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle

    (which sets `CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm` and runs
    `cargo test -p cintx-oracle --features rocm -- --ignored`, picking up the new
    base-profile test alongside the ECP one). Capture the test's
    `mismatch_count` line and the `any_nonzero=true` PASS print and report it.

    If no ROCm device / `cubecl_hip` toolchain is available on this host (the
    command errors at link/runtime with a HIP/ROCm-not-found error, NOT a test
    assertion failure), report that explicitly: state the test compiled cleanly
    under `--features rocm --no-run`, that the invocation is
    `cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle`, and that the
    on-device run could not execute because ROCm hardware/runtime is absent on this
    host (D-15: the ROCm oracle is a dev-host gate, not CI). Do NOT fake a
    mismatch_count.
  </action>
  <verify>
    <automated>cargo test -p cintx-oracle --features rocm --no-run 2>&1 | tail -15</automated>
  </verify>
  <done>
    `one_electron_random_rocm_parity.rs` exists, gated `#![cfg(feature = "rocm")]`,
    `#[test] #[ignore]` + `CINTX_ROCM_ORACLE=1` runtime gate, drives
    `int1e_{ovlp,kin,nuc}_sph` twice per random H2O case via `eval_raw`, asserts
    `mismatch_count == 0` and `any_nonzero`. It compiles under `--features rocm`
    and is a no-op without rocm. The test was RUN via
    `cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle` and the reported
    `mismatch_count` is captured (mismatch_count=0 on ROCm hardware), OR — if no
    ROCm device is present on this host — the absence is reported explicitly with
    the exact invocation and clean `--no-run` compile, without fabricating results.
  </done>
</task>

</tasks>

<verification>
- `cargo test -p cintx-cubecl --features cpu --lib kernels::one_electron` passes
  (device-vs-host scalar cross-check, f32 genericity, f64 dispatch, general-contraction parity).
- `cargo test -p cintx-oracle --features cpu --test one_electron_parity` still passes
  through the new device path (ovlp/kin/nuc determinism); vendor parity unchanged
  when run with `--features cpu CINTX_ORACLE_BUILD_VENDOR=1`.
- `cargo test -p cintx-oracle --features rocm --no-run` compiles the new oracle test.
- `cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle` runs the new test on
  the GPU and reports mismatch_count (=0) — or absence-of-ROCm is reported explicitly.
</verification>

<success_criteria>
- The scalar 1e operators (overlap, kinetic, nuclear-attraction) are computed by a
  `#[cube(launch)]` device kernel generic over `F: Float`, dispatched via
  `run_1e_scalar_device::<R>` including the ROCm `HipRuntime` arm.
- The kernel obeys all CubeCL authoring rules (verified against the manuals).
- Gradient (ipovlp/ipkin/ipnuc/iprinv) and spinor paths are unchanged, with an
  explicit in-code rationale for staying host-side (no silent narrowing).
- The existing scalar 1e parity tests pass byte/atol-identically through the device path.
- A new random ROCm oracle parity test exists, is correctly gated/ignored, and was
  run on the ROCm backend reporting mismatch_count=0 (or absence reported honestly).
</success_criteria>

<output>
After completion, create `.planning/quick/260529-imi-refactor-one-electron-rs-to-cubecl-kerne/260529-imi-SUMMARY.md`.
</output>
