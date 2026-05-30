---
phase: 260529-fsa
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/cintx-cubecl/src/kernels/center_4c1e.rs
  - crates/cintx-oracle/tests/center_4c1e_parity.rs
autonomous: true
requirements: [QUICK-260529-fsa]
must_haves:
  truths:
    - "center_4c1e.rs has a real #[cube(launch)] device kernel generic over F that fills the 4c1e polynomial-recurrence G-tensor, applies the 4-branch HRR, and contracts the Cartesian output on-device"
    - "launch_center_4c1e_typed dispatches the device kernel onto the resolved backend (CpuRuntime / HipRuntime / WgpuRuntime / CudaRuntime) instead of running the host f64 loop"
    - "Existing host parity tests (test_center_4c1e_parity_f64, test_center_4c1e_f32_smoke) stay green — byte-identical f64 results preserved"
    - "Device-vs-host equivalence unit tests in center_4c1e.rs pass on CpuRuntime across representative (li,lj,lk,ll) tuples"
    - "A new rocm random int4c1e_sph idempotency oracle test exists and reports mismatch_count=0 across 64 cases on the AMD GPU"
  artifacts:
    - path: "crates/cintx-cubecl/src/kernels/center_4c1e.rs"
      provides: "center_4c1e_kernel #[cube(launch)] + run_4c1e_device::<R> + backend dispatch in launch_center_4c1e_typed"
      contains: "#[cube(launch)]"
    - path: "crates/cintx-oracle/tests/center_4c1e_parity.rs"
      provides: "rocm random idempotency oracle for int4c1e_sph"
      contains: "test_int4c1e_sph_random_rocm_idempotency"
  key_links:
    - from: "crates/cintx-cubecl/src/kernels/center_4c1e.rs launch_center_4c1e_typed"
      to: "run_4c1e_device::<R>"
      via: "match ResolvedBackend -> run_4c1e_device with the per-backend Runtime"
      pattern: "run_4c1e_device::<"
    - from: "crates/cintx-oracle/tests/center_4c1e_parity.rs"
      to: "launch_center_4c1e (via eval_raw)"
      via: "eval_raw(RawApiId::INT4C1E_SPH, ...)"
      pattern: "RawApiId::INT4C1E_SPH"
---

<objective>
Refactor `crates/cintx-cubecl/src/kernels/center_4c1e.rs` from a host-side f64
loop into a real CubeCL `#[cube(launch)]` device kernel generic over float `F`,
dispatched onto the resolved backend's `ComputeClient` exactly like the sibling
`center_3c1e.rs`. Then add device-vs-host equivalence unit tests, create a
random ROCm idempotency oracle test for `int4c1e_sph`, and run it on the AMD GPU
to confirm mismatch_count=0.

4c1e is a SINGLE scalar path. The manifest registers only `int4c1e_cart` /
`int4c1e_sph` (OperatorId 24, gated behind `with-4c1e`). There is NO derivative
(`_ip1`) kernel — unlike 3c2e which needed two. So only ONE `#[cube(launch)]`
kernel + one `run_4c1e_device::<R>` is required.

Purpose: move the 4c1e numeric core onto the GPU compute backend per the project
constraint that CubeCL is the primary compute backend (host CPU work stays
limited to planning/validation/marshaling/glue), matching the 2c2e/3c1e/3c2e
ports already shipped.

Output:
- `center_4c1e_kernel` `#[cube(launch)]` generic over `F: Float + CubeElement`.
- `run_4c1e_device::<R: Runtime>` dispatch helper (compute in f64, output cast to F).
- `launch_center_4c1e_typed` rewired to dispatch the device kernel.
- Device-vs-host equivalence unit tests.
- `crates/cintx-oracle/tests/center_4c1e_parity.rs` rocm random idempotency oracle.
- Confirmed mismatch_count=0 on the AMD GPU via `xtask rocm-oracle --profile with-4c1e`.
</objective>

<execution_context>
@/home/user/Documents/workspace/cintx/.claude/get-shit-done/workflows/execute-plan.md
@/home/user/Documents/workspace/cintx/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@./CLAUDE.md

# MANDATORY READS before writing any kernel code
@docs/manual/Cubecl/Cubecl_basic_operations.md
@docs/manual/Cubecl/Cubecl_conditionals.md
@docs/manual/Cubecl/Cubecl_generics.md
@docs/manual/Cubecl/cubecl_macro_fanout_manual.md

# THE CLOSEST TEMPLATE — a 1e polynomial-recurrence family already ported to a
# #[cube(launch)] generic-F device kernel with run_3c1e_device::<R>. Mirror its
# structure (device kernel, run_*_device helper, backend match, host reference
# kept under #[cfg(test)], device-vs-host cross-check tests).
@crates/cintx-cubecl/src/kernels/center_3c1e.rs

# The file to refactor (current host f64 impl).
@crates/cintx-cubecl/src/kernels/center_4c1e.rs

# TEMPLATE for the new oracle test — copy Lcg, build_random_3shell, and
# test_int3c1e_sph_random_rocm_idempotency; adapt to 4 shells / int4c1e_sph.
@crates/cintx-oracle/tests/center_3c1e_parity.rs

<interfaces>
<!-- Contracts the executor needs — extracted from the codebase. Do not re-explore. -->

# RawApiId (crates/cintx-compat/src/raw.rs):
#   pub const INT4C1E_SPH: Self = Self::Symbol("int4c1e_sph");   // line 152, with-4c1e gated symbol
#   pub const INT4C1E_CART: Self = Self::Symbol("int4c1e_cart"); // line 151
#   eval_raw(api_id, Some(&mut out), None, &shls, &atm, &bas, &env, None, None) -> Result<...>

# raw env-layout constants (crates/cintx-compat/src/raw.rs), used by build_random_*shell:
#   ATM_SLOTS, BAS_SLOTS, ANG_OF, ATOM_OF, NPRIM_OF, NCTR_OF, CHARGE_OF, PTR_COORD,
#   PTR_EXP, PTR_COEFF, NUC_MOD_OF, PTR_ZETA, POINT_NUC, PTR_ENV_START

# center_3c1e.rs device-dispatch pattern (the model to copy verbatim, adapted to 4 shells):
#   fn run_3c1e_device<R: Runtime>(client: &ComputeClient<R>, ... ) -> Vec<f64>
#   center_3c1e_kernel::launch::<f64, R>(client, CubeCount::Static(1,1,1), CubeDim::new_1d(1), <ArrayArgs>, <scalars>)
#   match backend {
#     #[cfg(feature="cpu")]  ResolvedBackend::Cpu(c)      => run_..._device::<cubecl::cpu::CpuRuntime>(c, ...),
#     #[cfg(feature="wgpu")] ResolvedBackend::Wgpu(c,_)   => run_..._device::<cubecl_wgpu::WgpuRuntime>(c, ...),
#     #[cfg(feature="cuda")] ResolvedBackend::Cuda(c)     => run_..._device::<cubecl_cuda::CudaRuntime>(c, ...),
#     #[cfg(feature="rocm")] ResolvedBackend::Rocm(c)     => run_..._device::<cubecl_hip::HipRuntime>(c, ...),
#     #[cfg(feature="metal")]ResolvedBackend::Metal(c,_)  => run_..._device::<cubecl_wgpu::WgpuRuntime>(c, ...),
#   }

# xtask rocm-oracle (xtask/src/rocm_oracle.rs): `--profile with-4c1e` runs
#   CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm cargo test -p cintx-oracle --features rocm,with-4c1e -- --ignored
</interfaces>
</context>

<cube_authoring_rules>
The executor MUST read the four CubeCL manuals listed in <context> before
writing any `#[cube]` code. Hard rules (violating any of these = the kernel will
not compile or will panic at launch):

- No plain-fn calls inside `#[cube]` (everything inlined; no `cart_comps`, no helpers).
- No `if`-expressions — statement-form `if` only (assign a mutable then overwrite in `if`).
- Use `F::exp` / `F::sqrt` / `F::cast_from` / `F::new` (free functions), NEVER `.exp()`/`.sqrt()` methods.
- `u32`/`i32` index/counter types only; index `Array<F>` with `as usize` at the index site.
- `while`-loops only — no `for`, no `continue`, no `break`.
- No `Vec` inside the kernel; scratch lives in `&mut Array<F>` buffers passed by the launcher.
- Precision policy (match 3c1e): kernel is generic over `F`, but `run_4c1e_device`
  launches it at **f64** on-device for BOTH PrecisionKind variants; the read-back
  buffer is cast to `F` only at the c2s/output stage via `F::from_f64_lossy`.
- Known 4c1e nuance (from the current host impl, lines 188-192): the `fac`
  prefactor is applied ONLY to the z-axis initial value
  (`buf[0] = fac / (aijkl * sqrt(aijkl))`); x and y axes start at 1.0.
- Common factor is `SQRTPI * PI * sp_factor` (the 4c1e normalization), NOT the 2e
  formula — preserve verbatim (current host lines 566-574, see in-file comment).
- nroots is ALWAYS 1 for 4c1e (polynomial recurrence, not Rys quadrature).
</cube_authoring_rules>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Port center_4c1e_kernel #[cube(launch)] + run_4c1e_device + backend dispatch</name>
  <files>crates/cintx-cubecl/src/kernels/center_4c1e.rs</files>
  <behavior>
    - The CubeCL kernel run at f64 on CpuRuntime reproduces the current host
      `launch_center_4c1e_typed` output bit-for-(tolerance-)identically for
      s-s-s-s and at least one higher-l quartet.
    - test_center_4c1e_parity_f64 and test_center_4c1e_f32_smoke still pass
      unchanged (f64 byte-identity + f32 finite).
    - test_build_4c1e_shape_nroots_one, test_fill_4c1e_g_tensor_ssss,
      test_spinor_rejected_first still pass.
  </behavior>
  <action>
    Read the four CubeCL manuals AND center_3c1e.rs in full FIRST. Mirror its
    exact structure for the 4c1e family (4 shells instead of 3, polynomial
    recurrence instead of VRR, 4-branch HRR instead of i-HRR + k-separation).

    1. Add the CubeCL imports at the top (copy from center_3c1e.rs lines 53-55):
       `use cubecl::Runtime; use cubecl::client::ComputeClient; use cubecl::prelude::*;`

    2. Write `#[cube(launch)] fn center_4c1e_kernel<F: Float + CubeElement>(...)`:
       a single work-item (`if UNIT_POS == 0u32 { ... }`) faithful port of the
       host pipeline. Inline EVERYTHING — no calls to build_4c1e_shape /
       fill_4c1e_g_tensor / hrr_* / contract_4c1e_cart (those host fns become the
       #[cfg(test)] host reference, see step 5). The kernel must:
       - Recompute Shape4c1e layout (nroots=1, nmax=li+lj, mmax=lk+ll, ibase=li>lj,
         kbase=lk>ll, the dli/dlj/dlk/dll strides, di/dk/dl/dj, g_size) inline as
         u32 arithmetic.
       - Loop the full primitive quartet (pi,pj,pk,pl) with `while` loops, fold the
         per-primitive contraction-coefficient product, and accumulate into cart_out.
       - Fill the polynomial-recurrence G-tensor inline per axis (statement-form
         `if axis==2` selects the z-axis `fac/(aijkl*sqrt(aijkl))` initial value;
         x/y start at F::new(1.0)). Preserve the recurrence
         `buf[i+1] = 0.5*i/aijkl*buf[i-1] - r1r12*buf[i]` and the 2D shift fill,
         and the same r1/r2 base-center selection (nmax>=mmax branch) the host uses
         (current host lines 161-241). Use a scratch `&mut Array<F>` buffer for the
         1D polynomial `buf` (sized by the launcher; see run_4c1e_device).
       - Apply the 4-branch HRR (hrr_ik2d_4d / hrr_kj2d_4d / hrr_il2d_4d /
         hrr_lj2d_4d) inline, selected by kbase/ibase exactly as host lines 660-670.
         With nroots=1 the inner `for r in 0..nroots` host loops collapse to a single
         iteration — inline them as direct writes.
       - Contract `[gx|gy|gz]` into cart_out reproducing `contract_4c1e_cart`
         ordering inline (i fastest, l slowest; cart_comps enumeration inlined as
         nested `while` over lx/ly like center_3c1e_kernel lines 387-440), at irys=0.

    3. Write `fn run_4c1e_device<R: Runtime>(client, li,lj,lk,ll, nprim_*, ri,rj,rk,rl,
       common_factor, exps_*, coeff_*) -> Vec<f64>` mirroring run_3c1e_device:
       compute g_size + the polynomial-buf scratch size on the host, allocate the
       `g` and `buf` scratch Arrays + zeroed `cart_out` Array, create_from_slice the
       inputs, `center_4c1e_kernel::launch::<f64, R>(...)`, read_one_unchecked the
       output. Compute the kernel's scratch sizes (g = 3*g_size; the 1D recurrence
       buf = db*(bigger+1) where db=nmax+mmax+1, bigger=nmax.max(mmax)) on the host
       and pass them as Array args so the kernel never allocates.

    4. Rewire `launch_center_4c1e_typed::<F>`: keep `ensure_validated_4c1e` (spinor
       rejection FIRST), the 4-shell-count check, shell/atom extraction, common_factor
       (SQRTPI*PI*sp_factor — preserve verbatim), nci/nsi sizing, the cart_buf
       accumulation, and the cart_to_sph_2e / Cartesian copy + WR-06 not0 sentinel +
       ExecutionStats verbatim. REPLACE the host primitive-quartet loop body
       (current lines 586-695) with: for each (ci,cj,ck,cl) contraction tuple, build
       the per-column coeff vectors and call run_4c1e_device::<R> via a
       `match backend { ResolvedBackend::Cpu(..)/Wgpu/Cuda/Rocm/Metal => ... }`
       (copy the cfg-gated arms from center_3c1e.rs lines 793-824, adding the 4th
       shell's exps/coeff and rl). Drop the `let _ = backend;` line.

    5. Convert the existing host fns (build_4c1e_shape, fill_4c1e_g_tensor, the four
       hrr_*_4d, contract_4c1e_cart, cart_comps) to `#[cfg(test)]` host references
       (mirror center_3c1e.rs which keeps fill_g_tensor_3c1e/contract_3c1e_ovlp under
       #[cfg(test)]). Keep Shape4c1e + common_fac_sp non-test if the kernel/launcher
       still reference them; gate only what becomes test-only to avoid dead_code
       warnings in default builds. validated_4c1e_error / ensure_validated_4c1e stay
       as-is (already with-4c1e gated upstream).

    6. Keep the outer `launch_center_4c1e` precision dispatcher + F32 bytemuck cast
       path + CR-01 BufferTooSmall guard EXACTLY as-is (lines 747-772).

    Per project scope: do NOT add capi enum variants or legacy cint* wrappers — this
    touches the kernel file only.
  </action>
  <verify>
    <automated>cargo test -p cintx-cubecl --features cpu,with-4c1e center_4c1e 2>&1 | tail -30</automated>
  </verify>
  <done>
    center_4c1e.rs contains `#[cube(launch)] fn center_4c1e_kernel`, a
    `run_4c1e_device::<R>`, and a `match backend` dispatch in
    launch_center_4c1e_typed. test_center_4c1e_parity_f64 (byte-identical f64) and
    test_center_4c1e_f32_smoke pass. No host primitive-quartet loop remains in the
    launcher. Commit: `refactor(260529-fsa): center_4c1e #[cube(launch)] device kernel + run_4c1e_device + backend dispatch`.
  </done>
</task>

<task type="auto">
  <name>Task 2: Device-vs-host equivalence unit tests in center_4c1e.rs</name>
  <files>crates/cintx-cubecl/src/kernels/center_4c1e.rs</files>
  <action>
    Add a `#[cfg(test)] #[cfg(feature = "cpu")]` test block (or extend the existing
    tests mod) mirroring center_3c1e.rs's device-vs-host cross-check (lines 907-1037):

    1. `fn cpu_client() -> ComputeClient<cubecl::cpu::CpuRuntime>` helper.
    2. `fn host_cart_4c1e(...)` — single-primitive single-contraction host reference
       that builds the Shape4c1e, calls the (now #[cfg(test)]) fill_4c1e_g_tensor +
       hrr_*_4d branch-selected + contract_4c1e_cart for one quartet with
       common_factor = SQRTPI*PI*sp_factor product, returning the Cartesian Vec.
    3. `fn assert_device_matches_host(li,lj,lk,ll, ai,aj,ak,al)` — pick fixed
       distinct ri/rj/rk/rl, run host_cart_4c1e vs run_4c1e_device::<CpuRuntime> with
       single primitives/coeff=1.0, assert elementwise within atol=1e-12 + rtol=1e-10
       (copy the tolerance form from center_3c1e.rs line 1004).
    4. Representative tests bounding l so HRR sizes stay sane (nroots is always 1):
       - test_device_matches_host_ssss (0,0,0,0)
       - test_device_matches_host_psss (1,0,0,0) — exercises ibase + i-HRR branch
       - test_device_matches_host_sssp (0,0,0,1) — exercises kbase=false l-HRR branch
       - test_device_matches_host_spsp (0,1,0,1) — mixed ibase/kbase
       - test_device_matches_host_ppss (1,1,0,0) — ij pair both nonzero
    5. A generic-F evidence test: launch center_4c1e_kernel at f32 on CpuRuntime for
       s-s-s-s and assert the result is finite (mirror test_center_3c1e_kernel_generic_f32).

    Choose exponents in [0.5, 1.5] and coordinates spread enough that the cross-pair
    exponential does not underflow to all-zeros.
  </action>
  <verify>
    <automated>cargo test -p cintx-cubecl --features cpu,with-4c1e device_matches_host 2>&1 | tail -20</automated>
  </verify>
  <done>
    At least 5 device-vs-host equivalence tests + 1 generic-f32 test pass on
    CpuRuntime, proving the device kernel reproduces the host reference across the
    four HRR branches. Commit: `test(260529-fsa): center_4c1e device-vs-host equivalence + generic-f32 unit tests`.
  </done>
</task>

<task type="auto">
  <name>Task 3: Create center_4c1e_parity.rs rocm random idempotency oracle</name>
  <files>crates/cintx-oracle/tests/center_4c1e_parity.rs</files>
  <action>
    Create the oracle test file by adapting center_3c1e_parity.rs to 4 shells /
    int4c1e_sph. Copy Lcg, the env-pointer-layout builder, and the random
    idempotency test structure verbatim, changing 3→4 shells.

    1. Module gate: `#![cfg(any(feature = "cpu", feature = "rocm"))]` (mirror the
       3c1e sibling line 28).
    2. Imports: `use cintx_compat::raw::{ATM_SLOTS, ANG_OF, ATOM_OF, BAS_SLOTS,
       CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC, PTR_COEFF, PTR_COORD,
       PTR_EXP, PTR_ZETA, RawApiId, eval_raw};` plus
       `#[cfg(all(feature = "rocm", feature = "with-4c1e"))] use cintx_compat::raw::PTR_ENV_START;`
    3. `nsph_for_l(l) -> usize { (2*l+1) as usize }` helper.
    4. Copy the `Lcg` struct + impl verbatim, gated
       `#[cfg(all(feature = "rocm", feature = "with-4c1e"))]`.
    5. `build_random_4shell(rng) -> (Vec<i32>, Vec<i32>, Vec<f64>, i32,i32,i32,i32)`
       gated `#[cfg(all(feature = "rocm", feature = "with-4c1e"))]`: draw li,lj,lk,ll
       from {0,1,2}; nprim_{i,j,k,l} from {1,2,3}; four distinct atom coordinates
       offset so no two centers coincide (extend the 3-shell offset pattern with a
       4th center, e.g. coord_l offset like coord_k but in a different octant);
       random exps in [0.25,4.0], coeffs in [0.15,1.0]; env starts at PTR_ENV_START
       (=20); 4 atm rows + 4 bas rows (NCTR_OF=1 each). No redraw needed — 4c1e
       nroots is always 1, so just bounding l in {0,1,2} keeps HRR sizes sane.
    6. `#[cfg(all(feature = "rocm", feature = "with-4c1e"))] #[test] #[ignore]
       fn test_int4c1e_sph_random_rocm_idempotency()`:
       - First line: assert `std::env::var("CINTX_ROCM_ORACLE").as_deref() == Ok("1")`
         with the exact "must be invoked via xtask rocm-oracle ... intentionally
         blocked" panic message from the 3c1e sibling (lines 654-659).
       - atol=1e-12, rtol=1e-10, n_cases=64, seed a fresh Lcg (use a distinct seed
         like 0x5ec0_4c1e_1234_5678).
       - For each case: build_random_4shell, n_elem = ni*nj*nk*nl, shls=[0,1,2,3],
         call eval_raw(RawApiId::INT4C1E_SPH, ...) twice into out1/out2, accumulate
         mismatch_count via the inline abs+rel threshold loop, track any_nonzero.
       - assert mismatch_count==0, assert any_nonzero, and print
         `PASS: rocm random int4c1e_sph idempotency mismatch_count=0 across {n_cases} cases at atol=.../rtol=...`.

    Also add a cpu self-consistency test (no with-4c1e/rocm requirement is possible
    here because int4c1e_sph needs with-4c1e — so gate the cpu self-consistency test
    `#[cfg(all(feature = "cpu", feature = "with-4c1e"))]`): build one fixed 4-shell
    system, eval_raw int4c1e_sph twice, assert idempotent + at least one nonzero.
    This gives a non-rocm smoke that the new file compiles and the kernel runs.

    Confirm cintx-oracle/Cargo.toml already has the `with-4c1e` passthrough (it does:
    `with-4c1e = ["cintx-compat/with-4c1e"]`) — no Cargo.toml change needed.
  </action>
  <verify>
    <automated>cargo test -p cintx-oracle --features cpu,with-4c1e --test center_4c1e_parity 2>&1 | tail -20</automated>
  </verify>
  <done>
    crates/cintx-oracle/tests/center_4c1e_parity.rs exists with
    test_int4c1e_sph_random_rocm_idempotency (rocm+with-4c1e gated, ignored,
    env-gated with the direct-run panic guard) and a cpu self-consistency test that
    passes under `--features cpu,with-4c1e`. Commit:
    `test(260529-fsa): random ROCm int4c1e_sph idempotency oracle test`.
  </done>
</task>

<task type="auto">
  <name>Task 4: Run the ROCm oracle and confirm mismatch_count=0</name>
  <files>crates/cintx-cubecl/src/kernels/center_4c1e.rs, crates/cintx-oracle/tests/center_4c1e_parity.rs</files>
  <action>
    Run the rocm oracle on the AMD GPU dev host:
    `cargo run -p xtask -- rocm-oracle --profile with-4c1e`
    (this spawns `CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm cargo test -p cintx-oracle
    --features rocm,with-4c1e -- --ignored`).

    Confirm the new test_int4c1e_sph_random_rocm_idempotency runs on the ROCm device
    and prints `mismatch_count=0 across 64 cases`. The prior 2c2e/3c1e/3c2e rocm
    oracle runs achieved mismatch_count=0 on this host, so a real device launch is
    expected to pass.

    If the test fails on the device but the CpuRuntime device-vs-host tests (Task 2)
    pass, the divergence is a CubeCL HIP-backend authoring violation — re-read the
    CubeCL manuals, inspect the kernel for any non-statement-form `if`, method-style
    `.exp()/.sqrt()`, or host-fn call that the CPU JIT tolerated but HIP rejects, fix
    in center_4c1e.rs, and re-run. Do NOT relax the atol/rtol to mask a real mismatch.

    The pre-existing `test_f32_int3c2e_sph_parity` failure on this branch is baseline
    noise (memory note project_3c2e_f32_parity_preexisting_fail) — ignore it; only
    the int4c1e_sph rocm test result matters for this task.
  </action>
  <verify>
    <automated>cargo run -p xtask -- rocm-oracle --profile with-4c1e 2>&1 | grep -E "int4c1e_sph (random )?idempotency mismatch_count=0|rocm-oracle suite passed" || (echo "ROCM ORACLE DID NOT REPORT mismatch_count=0 FOR int4c1e_sph" && exit 1)</automated>
  </verify>
  <done>
    `xtask rocm-oracle --profile with-4c1e` runs test_int4c1e_sph_random_rocm_idempotency
    on the AMD GPU and reports mismatch_count=0 across 64 cases. No code commit in
    this task unless a HIP-backend fix to center_4c1e.rs was required (in which case
    commit `fix(260529-fsa): HIP-backend correctness fix for center_4c1e kernel`).
  </done>
</task>

</tasks>

<verification>
- `cargo test -p cintx-cubecl --features cpu,with-4c1e center_4c1e` — all existing
  + new device-vs-host tests green; f64 parity byte-identical.
- `cargo test -p cintx-oracle --features cpu,with-4c1e --test center_4c1e_parity` —
  cpu self-consistency test green; rocm test compiles (ignored).
- `cargo run -p xtask -- rocm-oracle --profile with-4c1e` — int4c1e_sph random
  idempotency reports mismatch_count=0 on the AMD GPU.
- No capi enum variants or legacy cint* wrappers added (kernel + oracle only).
</verification>

<success_criteria>
- center_4c1e.rs has a real `#[cube(launch)]` device kernel generic over F,
  dispatched onto the resolved backend via run_4c1e_device::<R>.
- Host f64 byte-identity preserved (test_center_4c1e_parity_f64 green).
- Device-vs-host equivalence verified on CpuRuntime across the four HRR branches.
- A random ROCm int4c1e_sph idempotency oracle exists and reports mismatch_count=0
  on the AMD GPU.
</success_criteria>

<output>
After completion, create `.planning/quick/260529-fsa-refactor-center-4c1e-rs-to-cubecl-kernel/260529-fsa-SUMMARY.md`
</output>
