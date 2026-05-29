---
phase: 260529-exs
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/cintx-cubecl/src/kernels/center_3c2e.rs
  - crates/cintx-oracle/tests/center_3c2e_parity.rs
autonomous: true
requirements: []
quick_id: 260529-exs
slug: center-3c2e-cubecl-kernel

must_haves:
  truths:
    - "The scalar 3c2e numeric core runs as a #[cube(launch)] device kernel generic over F: Float, dispatched on the resolved backend's ComputeClient at f64."
    - "The int3c2e_ip1 gradient numeric core (∇_A, 3 components) runs as a #[cube(launch)] device kernel generic over F: Float, dispatched on the resolved backend's ComputeClient at f64."
    - "Both device kernels use the 2c2e Rys template: #[comptime] nroots + comptime! rys_root1..5 selection; reject nroots>5 fail-closed before any rys dispatch."
    - "Device compute runs at f64 for both PrecisionKind variants; output is cast to F via F::from_f64_lossy at the c2s/Cartesian output stage."
    - "launch_center_3c2e and launch_center_3c2e_typed signatures are byte-for-byte unchanged; the registered FamilyLaunchFn keeps working; the operator=='ip1' branch still routes to the ip1 launcher."
    - "The rocm arm of each run_*_device dispatch uses cubecl_hip::HipRuntime, #[cfg(feature=\"rocm\")]-gated exactly as 2c2e does."
    - "A random rocm int3c2e_ip1_sph idempotency oracle test exists, gated identically to its 2c2e/3c1e siblings (rocm + ignore + CINTX_ROCM_ORACLE=1)."
  artifacts:
    - path: "crates/cintx-cubecl/src/kernels/center_3c2e.rs"
      provides: "center_3c2e_scalar_kernel + run_3c2e_device, center_3c2e_ip1_kernel + run_3c2e_ip1_device, rewritten launch_center_3c2e_typed routing both paths through the device; host fns retained under #[cfg(test)]"
      contains: "#[cube(launch)]"
    - path: "crates/cintx-oracle/tests/center_3c2e_parity.rs"
      provides: "test_int3c2e_ip1_sph_random_rocm_idempotency + Lcg + build_random_3shell_3c2e"
      contains: "test_int3c2e_ip1_sph_random_rocm_idempotency"
  key_links:
    - from: "crates/cintx-cubecl/src/kernels/center_3c2e.rs"
      to: "crates/cintx-cubecl/src/math/rys.rs (rys_root1..5)"
      via: "comptime! nroots branch in the #[cube] kernels"
      pattern: "rys_root[1-5]::<F>"
    - from: "crates/cintx-cubecl/src/kernels/center_3c2e.rs"
      to: "resolved backend ComputeClient"
      via: "match backend { Cpu|Wgpu|Cuda|Rocm|Metal } #[cfg]-gated arms"
      pattern: "HipRuntime"
    - from: "crates/cintx-oracle/tests/center_3c2e_parity.rs"
      to: "eval_raw(RawApiId::INT3C2E_IP1_SPH)"
      via: "device kernel through launch_center_3c2e_ip1"
      pattern: "INT3C2E_IP1_SPH"
---

<objective>
Refactor `crates/cintx-cubecl/src/kernels/center_3c2e.rs` from host-only `f64`
pipelines into real CubeCL `#[cube(launch)]` **device kernels generic over
`F: Float`**, for BOTH numeric paths the file owns:

1. the **scalar 3c2e** path (`fill_g_tensor_3c2e` → `split_ij_hrr` → `contract_3c2e`), and
2. the **int3c2e_ip1 gradient** path (`build_2e_shape(li+1,..)` → `fill_g_tensor_2e` → `f12::gout_ip1`, 3 components, component-leading `[3,nk,nj,ni]`).

Both paths use Rys quadrature, so the device-kernel template is
`center_2c2e.rs` (the `#[comptime] nroots` + `comptime!` `rys_root1..5` pattern),
NOT the rys-free `center_3c1e.rs`. Then add a **random ROCm idempotency oracle
test** for the registered `int3c2e_ip1_sph` API and prepare it for the rocm run.

Purpose: make the registered `int3c2e_ip1_sph` API genuinely drive a `#[cube]`
device kernel on the GPU (the faithful analog of the 3c1e/2c2e ports), and bring
the clean scalar-3c2e sibling onto the device alongside it.

Output: two `#[cube(launch)]` kernels + two `run_*_device<R: Runtime>`
dispatchers in `center_3c2e.rs`; a rewritten `launch_center_3c2e_typed` routing
both paths through the device; device-vs-host CPU-runtime cross-checks for both
kernels; a random rocm `int3c2e_ip1_sph` idempotency oracle test compiled and
collected (the on-device rocm run is the orchestrator's post-merge job).
</objective>

<execution_context>
@/home/user/Documents/workspace/cintx/.claude/get-shit-done/workflows/execute-plan.md
@/home/user/Documents/workspace/cintx/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/quick/260529-exs-center-3c2e-cubecl-kernel/260529-exs-CONTEXT.md
@./CLAUDE.md

# Rys device-kernel template (THE pattern to mirror exactly):
@crates/cintx-cubecl/src/kernels/center_2c2e.rs

# The file being refactored (both host paths to port + the ip1/scalar routing + tests):
@crates/cintx-cubecl/src/kernels/center_3c2e.rs

# IP1 path host sources to INLINE into the #[cube] ip1 kernel (no plain-fn calls in #[cube]):
@crates/cintx-cubecl/src/kernels/two_electron.rs
@crates/cintx-cubecl/src/kernels/f12.rs

# Device Rys functions the kernels CALL (already #[cube]; do NOT inline these):
@crates/cintx-cubecl/src/math/rys.rs

# Oracle siblings to model the random rocm test on:
@crates/cintx-oracle/tests/center_3c2e_parity.rs
@crates/cintx-oracle/tests/center_3c1e_parity.rs

<cube_authoring_rules>
The project error-solution guides override the basic manuals. Mirror exactly what
`center_2c2e.rs` already does:
1. No plain-Rust fn calls inside `#[cube]` (E0433). Inline `fill_g_tensor_3c2e`/
   `split_ij_hrr`/`contract_3c2e` (scalar) and `fill_g_tensor_2e`/`nabla1i_2e`/
   `gout_ip1`/`cart_comps` (ip1) into the kernel bodies. You MAY call the device
   `rys_root1..5` (they are already `#[cube]`).
2. No `if`-as-expression (E0308). Statement form only:
   `let mut d = xij; if axis == 1u32 { d = yij; } if axis == 2u32 { d = zij; }`.
3. Math via associated functions ONLY: `F::exp(x)`, `F::sqrt(x)`, `F::cast_from(lit)`,
   `F::new(lit)`, `F::from_int(i)`. NEVER `x.exp()`/`x.sqrt()` (E0599).
4. Kernel-legal types only: `F` floats + `u32`/`i32` indices/counters. No `usize`
   locals, no `u64`, no `Vec`, no host `for`. Index arrays as `arr[(expr) as usize]`.
5. No `continue`/`break` in `#[cube]` `while` loops — guard bodies with `if cond {..}`
   and always advance the counter. The exponent screen must be `if e <= cutoff {..}`,
   never `continue`.
6. `while` loops only (manual `i += 1u32;`); single work item via `if UNIT_POS == 0u32`.
   Reproduce `cart_comps` ordering inline with the descending nested-`while` idiom.
7. Exactly ONE `#[cube(launch)]` entry per kernel; generic `<F: Float + CubeElement>`;
   nroots is `#[comptime] nroots: u32` with a `comptime!(nroots == k)` branch selecting
   `rys_root1..5`. li/lj/lk/nprim*/nctr* are runtime `u32` args. (TWO kernels total —
   one scalar, one ip1 — each its own single `#[cube(launch)]` entry.)
8. Device buffers via `client.create_from_slice(f64::as_bytes(&v))`; read back via
   `client.read_one_unchecked` + `f64::from_bytes`; launch
   `kernel::launch::<f64, R>(.., CubeCount::Static(1,1,1), CubeDim::new_1d(1), ..)`.
</cube_authoring_rules>

<key_facts>
Treat as ground truth (orchestrator mapped the codebase); verify rather than rediscover.

LIVE REGISTERED PATH = IP1.
- `launch_center_3c2e` (L765) → `launch_center_3c2e_typed` (L566). Inside typed,
  `if plan.descriptor.operator_name() == "ip1"` (L616) returns
  `launch_center_3c2e_ip1::<F>(...)`. Registered RawApiId is `INT3C2E_IP1_SPH` only
  (no plain `int3c2e_sph` registered for dispatch — but `int3c2e_sph` exists in the
  manifest for the resolver, used by the `build_plain_plan` test). The scalar
  pipeline (L622-695) is the non-ip1 fall-through.

BOTH PATHS USE RYS, reject nroots>5:
- Scalar: `nrys_roots = (li+lj+lk)/2 + 1` (L628).
- IP1: `grad_shape.nroots` from `build_2e_shape(li+1, lj, 0, lk)` (L358-366).

SCALAR PATH (L622-695):
- `fill_g_tensor_3c2e` (L70): G-tensor layout `[gx|gy|gz]`, each block
  `[m=0..lk][n=0..(li+lj)][root]` root-fastest: `dn=nrys`, `dm=nrys*(nmax+1)`,
  `g_size=nrys*(nmax+1)*(mmax+1)`, `nmax=li+lj`, `mmax=lk`. Uses the 2e-style
  pair: `aij=pair.zeta_ab`, `akl=ak`, center P = `pair.center_p_{x,y,z}`,
  `xij_kl=P-rk`, `a1=aij*akl`, `a0=a1/(aij+akl)`, `fac1=sqrt(a0/a1^3)*fac_env`,
  `x_rys=a0*rr`; per-root b00/b10/b01/c00/c0p recurrences (see L108-182).
  NOTE: `compute_pdata_host` produces `pair` (zeta_ab, center_p_*, fac). The device
  kernel must inline the small Gaussian-product pdata math too (zeta_ab = ai+aj,
  center_p = (ai*ri+aj*rj)/(ai+aj), fac = exp(-ai*aj/(ai+aj)*|ri-rj|^2)), since
  `compute_pdata_host` is a plain fn (cannot be called inside `#[cube]`). Cross-check
  the inlined pdata against `compute_pdata_host` in the host reference test.
- `split_ij_hrr` (L195): HRR transfer along j recovering (i,j) channels; output
  `[axis][root][k][j][i]` (i fastest); `rirj = ri - rj`.
- `contract_3c2e` (L254): triple `cart_comps(li)×cart_comps(lj)×cart_comps(lk)`;
  `out[(k_idx*ncj + j_idx)*nci + i_idx] += sum_root gx*gy*gz` (i fastest, k slowest).
- Primitive loop: `kp` outer → `jp` → `ip`; `fac_env=common_factor*pair.fac`;
  weight `ci*cj*ck`; `common_factor=(PI^3)*2/SQRTPI*fac_sp(li)*fac_sp(lj)*fac_sp(lk)`.
- The scalar path canonicalizes `li>=lj` (`swap_ij`) and `transpose_ij_3idx` back.
  KEEP swap_ij + transpose on the HOST around the device call (don't push it into
  the kernel) — the kernel runs in canonical li>=lj order; the host launcher decides
  the swap and transposes the read-back cart_buf, exactly as today (L700-706).
- Output tails: `cart_to_sph_3c2e` (Spheric), `cart_to_spinor_sf_3c2e` (Spinor),
  Cartesian copy — all host-side after read-back, cast via `F::from_f64_lossy`.

IP1 PATH (L339-564):
- `grad_shape = build_2e_shape(li+1, lj, 0, lk)` (phantom 2e lk=0, real k in 2e ll-slot).
- Per primitive triple kp/jp/ip: `pdata_ij=compute_pdata_host(ai,aj,ri,rj,..)`,
  `pdata_kl=compute_pdata_host(0.0,ak,rk,rk,..)`, `fac_env=common_factor*pdata_ij.fac*pdata_kl.fac`,
  then `g = fill_g_tensor_2e(ai,aj,0.0,ak,&ri,&rj,&rk,&rk,grad_shape,fac_env)`.
- `gout = f12::gout_ip1(&g, &grad_f12_shape, li, lj, 0, lk, ai)` → interleaved
  `gout[n*3+comp]`, n walks [ll=real_k, lk=phantom(size1), lj, li] = effectively [k][j][i].
- Transpose interleaved → component-leading: `cart_blocks[base + comp*block_len + n] += weight*gout[n*3+comp]`.
- `common_factor=(PI^3)*2/SQRTPI*fac_sp(li)*fac_sp(lj)*fac_sp(lk)`. 3 components,
  output `[3,nk,nj,ni]` F-order; per-component `cart_to_sph_3c2e` on the Spheric tail.
- `fill_g_tensor_2e` internals to inline (two_electron.rs L365-518): VRR fill via
  `vrr_fill_axis` (3 axes) + one of FOUR HRR branches selected by (ibase,kbase).
  For the 3c2e ip1 mapping with `build_2e_shape(li+1, lj, 0, lk)`: nmax=li+1+lj,
  mmax=0+lk=lk; `ibase = (li+1) > lj`; `kbase = 0 > lk` = false always. So only the
  `kbase==false` HRR branches are ever taken: `hrr_lj2d_4d` (ibase=false) or
  `hrr_il2d_4d` (ibase=true). You still must inline both (ibase can be either) but
  NOT `hrr_ik2d_4d`/`hrr_kj2d_4d` (kbase=true) — they are dead for this mapping.
  Document this so the kernel stays minimal.
- `nabla1i_2e` (f12.rs L599) + `gout_ip1` (f12.rs L741) inlined: `g1` = nabla1i applied
  to g at base li; gout sums `s[0]=g1x*g0y*g0z`, `s[1]=g0x*g1y*g0z`, `s[2]=g0x*g0y*g1z`.

PRECISION & GUARDS (keep verbatim):
- Device compute at f64; read-back cast to F via `F::from_f64_lossy`.
- Keep the WR-06 precision-aware nonzero sentinel and the `ExecutionStats` blocks verbatim.
- IP1: keep the `Representation::Spinor` → `UnsupportedApi` guard (L350) and the
  `grad_shape.nroots > 5` → `UnsupportedApi` guard (L362) BEFORE any device dispatch.
- Scalar: keep the `nrys_roots > 5` → `UnsupportedApi` guard (L629).
- Retain host fns (`fill_g_tensor_3c2e`, `split_ij_hrr`, `contract_3c2e`,
  `fill_g_tensor_2e`-equivalent reference via the existing two_electron path, `cart_comps`)
  under `#[cfg(test)]` as the device-vs-host cross-check reference (like 2c2e keeps
  `fill_g_tensor_2c2e`). The existing ip1_tests module (L942) must keep passing.
</key_facts>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Scalar 3c2e #[cube(launch)] device kernel + run_3c2e_device + route the scalar path</name>
  <files>crates/cintx-cubecl/src/kernels/center_3c2e.rs</files>
  <behavior>
    - Device kernel reproduces the host scalar pipeline exactly for representative
      (li,lj,lk) triples on CpuRuntime at f64: s-s-s, s-s-p, p-s-s, s-p-s, p-s-p,
      and at least one with li>0 and lj>0 (e.g. p-p-s). Max abs diff within
      `1e-12 + 1e-10*|host|` of the host reference (`fill_g_tensor_3c2e` →
      `split_ij_hrr` → `contract_3c2e`) — mirrors 2c2e `assert_device_matches_host`.
    - The inlined Gaussian-product pdata math (zeta_ab, center_p, fac) matches
      `compute_pdata_host` for the test exponent/coord pairs.
    - The kernel compiles under the cube macro for `--features cpu` AND `--features rocm`
      (no MLIR/index-type errors).
    - An f32 genericity launch test (like `test_center_2c2e_kernel_generic_f32`) runs
      an s-s-s triple at f32 on CpuRuntime and asserts a finite result.
  </behavior>
  <action>
    1. Add imports mirroring 2c2e: `use cubecl::Runtime; use cubecl::client::ComputeClient;
       use cubecl::prelude::*; use crate::math::rys::{rys_root1,rys_root2,rys_root3,rys_root4,rys_root5};`
       and the `PIE4`/`MAX_DEVICE_NROOTS` consts. Keep `rys_roots_host` import under `#[cfg(test)]`.
    2. Write `#[cube(launch)] #[allow(clippy::too_many_arguments)] fn center_3c2e_scalar_kernel<F: Float + CubeElement>(...)`:
       inputs `exps_i/j/k`, `coeff_i/j/k` as `&Array<F>`; scratch `g: &mut Array<F>`,
       `g_split: &mut Array<F>`, `urys: &mut Array<F>`, `wrys: &mut Array<F>`,
       `cart_out: &mut Array<F>`; scalar `F` coords (rix..rkz); `common_factor: F`,
       `pie4: F`, `expcutoff: F`; runtime `u32` shape (`li,lj,lk,nprim_i,nprim_j,nprim_k,
       nctr_i,nctr_j,nctr_k`); `#[comptime] nroots: u32`. Guard with `if UNIT_POS == 0u32`.
       Inline, in canonical li>=lj order (the HOST decides swap; the kernel always gets
       li>=lj): zero cart_out; loop `kp` outer → `jp` → `ip`; per triple inline the
       Gaussian-product pdata (zeta_ab=ai+aj, center_p=(ai*ri+aj*rj)/zeta_ab,
       fac=F::exp(-ai*aj/zeta_ab*rr_ij)); inline the `fill_g_tensor_3c2e` recurrence
       (per-root b00/b10/b01/c00/c0p over the [m][n][root] layout; `rr=|P-rk|^2`,
       `a1=zeta_ab*ak`, `a0=a1/(zeta_ab+ak)`, `fac1=F::sqrt(a0/(a1*a1*a1))*common_factor*fac`,
       call the comptime `rys_root{1..5}::<F>(a0*rr, urys, wrys, pie4)` exactly as 2c2e
       L199-209); inline the `split_ij_hrr` j-HRR (rirj=ri-rj) into `g_split`; inline the
       `contract_3c2e` triple-cart_comps contraction with the descending nested-`while`
       idiom (i fastest, k slowest), accumulating `cart_out[(k_idx*ncj+j_idx)*nci+i_idx]
       += weight*sum_root(gx*gy*gz)` where `weight=coeff_i*coeff_j*coeff_k`. Use the
       exponent screen as `if eijk <= expcutoff { ..fill+contract.. }` (no `continue`).
    3. Write `fn run_3c2e_device<R: Runtime>(client, li,lj,lk, nprim*, nctr*, nroots,
       ri,rj,rk, common_factor, exps_i/j/k, coeff_i/j/k) -> Vec<f64>`: size `g`, `g_split`,
       `urys`/`wrys`, and `cart_out` (out_len = nci*ncj*nck) buffers via `f64::as_bytes`;
       `center_3c2e_scalar_kernel::launch::<f64, R>(client, CubeCount::Static(1,1,1),
       CubeDim::new_1d(1), ..)`; read back `cart_out` via `client.read_one_unchecked`.
    4. Rewrite the scalar fall-through of `launch_center_3c2e_typed` (L622-733) so its
       numeric core is `let cart_buf: Vec<f64> = match backend { Cpu|Wgpu|Cuda|Rocm|Metal
       => run_3c2e_device::<R>(..) }` (each arm `#[cfg(feature=...)]`-gated EXACTLY as
       2c2e L671-702, rocm => `cubecl_hip::HipRuntime`). KEEP the existing host-side
       `swap_ij` decision and `transpose_ij_3idx` on the read-back buffer, the
       `cart_to_sph_3c2e`/`cart_to_spinor_sf_3c2e`/Cartesian tails, the WR-06 sentinel,
       and the `ExecutionStats` return verbatim.
    5. Move the host `fill_g_tensor_3c2e`, `split_ij_hrr`, `contract_3c2e`, `cart_comps`,
       `transpose_ij_3idx`(if only used in tests now) behind `#[cfg(test)]` as the
       cross-check reference (transpose stays non-test if the launcher uses it). Add a
       `host_cart_3c2e(...)` reference + `assert_device_matches_host_3c2e` test for the
       triples in <behavior>, plus the f32 genericity launch test. Keep
       `test_center_3c2e_parity_f64`, `test_center_3c2e_f32_smoke`,
       `test_fill_g_tensor_3c2e_sss_nonzero`, `test_contract_3c2e_sss_nonzero` passing.
    6. Leave `launch_center_3c2e` (outer) and `launch_center_3c2e_typed` SIGNATURES
       byte-for-byte unchanged. Do NOT touch the `operator_name()=="ip1"` branch yet
       (Task 2 owns it — it still calls the host `launch_center_3c2e_ip1`).
  </action>
  <verify>
    <automated>cargo build -p cintx-cubecl --features cpu && cargo build -p cintx-cubecl --features rocm && cargo test -p cintx-cubecl --features cpu center_3c2e</automated>
  </verify>
  <done>`center_3c2e_scalar_kernel` (#[cube(launch)]) + `run_3c2e_device<R>` exist; the
  scalar fall-through routes through the device on all five backend arms (rocm=HipRuntime);
  outer/typed signatures intact; device matches host within tol on CpuRuntime for the
  representative triples; cpu + rocm builds compile.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: int3c2e_ip1 #[cube(launch)] device kernel + run_3c2e_ip1_device + route the ip1 path</name>
  <files>crates/cintx-cubecl/src/kernels/center_3c2e.rs</files>
  <behavior>
    - The ip1 device kernel reproduces `launch_center_3c2e_ip1`'s 3-component
      derivative exactly on CpuRuntime at f64 for representative triples: s-s-s,
      p-s-s, s-p-s, s-s-p, p-p-s, p-s-p (each ≤ nroots 5). Max abs diff within
      `1e-12 + 1e-10*|host|` of the host reference (the current `launch_center_3c2e_ip1`
      pre-transform `cart_blocks`, i.e. the component-leading per-triple Cartesian
      tensor before the sph copy).
    - The existing ip1_tests module (component_count, not_equal_to_plain, determinism,
      spinor_unsupported) keeps passing — the device path is byte-equivalent to the host.
    - The kernel compiles under the cube macro for `--features cpu` AND `--features rocm`.
    - The spinor guard and the nroots>5 guard still fire BEFORE any device dispatch.
  </behavior>
  <action>
    1. Write `#[cube(launch)] #[allow(clippy::too_many_arguments)] fn center_3c2e_ip1_kernel<F: Float + CubeElement>(...)`:
       inputs `exps_i/j/k`, `coeff_i/j/k` as `&Array<F>`; scratch `g: &mut Array<F>`
       (size `3*grad_shape.g_size`), `g1: &mut Array<F>` (the nabla1i buffer, same size),
       `urys`/`wrys: &mut Array<F>`, and component-leading `cart_out: &mut Array<F>`
       (size `3*nci*ncj*nck`); scalar `F` coords (rix..rkz); `common_factor: F`, `pie4: F`;
       runtime `u32` shape (`li,lj,lk` BASE; the kernel computes the elevated `li+1`
       layout internally; `nprim_i,nprim_j,nprim_k,nctr_i,nctr_j,nctr_k`; plus the
       `build_2e_shape(li+1,lj,0,lk)` strides `di,dk,dl,dj,g_size,nmax,mmax` and the
       `ibase` flag passed as `u32` 0/1 — `kbase` is always false for this mapping so
       no flag needed); `#[comptime] nroots: u32`. Guard `if UNIT_POS == 0u32`.
       Inline: zero cart_out; loop `kp`→`jp`→`ip`; per triple inline pdata_ij
       (ai,aj@ri,rj) and pdata_kl (0.0,ak@rk,rk → fac=1), `fac_env=common_factor*
       pdata_ij.fac*pdata_kl.fac`; inline `fill_g_tensor_2e` for the kbase=false case
       ONLY (VRR `vrr_fill_axis` for 3 axes via `rys_root{1..5}` comptime; then the
       ibase-selected HRR — inline `hrr_lj2d_4d` for ibase==0 and `hrr_il2d_4d` for
       ibase==1; do NOT inline the kbase=true branches, they are dead for this 3c2e
       mapping — leave a comment); inline `nabla1i_2e` into `g1` at base li; inline
       the `gout_ip1` contraction (n walks [k][phantom1][j][i], i fastest) producing
       `s[0]=g1x*g0y*g0z, s[1]=g0x*g1y*g0z, s[2]=g0x*g0y*g1z`; accumulate into
       component-leading `cart_out[comp*block_len + n] += weight*s[comp]`,
       `weight=coeff_i*coeff_j*coeff_k`, `block_len=nci*ncj*nck`.
       (No exponent `continue`; if a screen is added use `if e <= cutoff {..}`.)
    2. Write `fn run_3c2e_ip1_device<R: Runtime>(client, li,lj,lk, nprim*, nctr*, nroots,
       ibase, di,dk,dl,dj,g_size,nmax,mmax, ri,rj,rk, common_factor, exps/coeffs..) -> Vec<f64>`:
       buffers via `f64::as_bytes`; launch `center_3c2e_ip1_kernel::launch::<f64,R>(..,
       CubeCount::Static(1,1,1), CubeDim::new_1d(1), ..)`; read back `cart_out`
       (len `3*nci*ncj*nck`).
    3. Rewrite `launch_center_3c2e_ip1::<F>` so AFTER the existing spinor + nroots>5
       guards (keep verbatim) and the coord/common_factor/shape setup, the per-triple
       host loop body is REPLACED by a single `match backend { Cpu|Wgpu|Cuda|Rocm|Metal
       => run_3c2e_ip1_device::<R>(..) }` returning the component-leading `[3,nci,ncj,nck]`
       f64 `cart_buf`. Compute `grad_shape = build_2e_shape(li+1,lj,0,lk)` on the host to
       derive the stride/nroots/ibase args passed to the device. KEEP the existing
       per-component `cart_to_sph_3c2e` / Cartesian staging-write tails (L468-543), the
       WR-06 sentinel, and the `ExecutionStats` return verbatim. NOTE: `launch_center_3c2e_ip1`
       must now take `backend: &ResolvedBackend` (add the param) and `launch_center_3c2e_typed`
       must pass `backend` through at L617 — both are private fns, so this is internal only;
       the public `launch_center_3c2e` / typed PUBLIC signatures stay unchanged.
    4. Retain the host `launch_center_3c2e_ip1` numeric body as a `#[cfg(test)]`
       `host_ip1_cart_blocks(...)` reference (or reuse the existing two_electron host
       `fill_g_tensor_2e`+`gout_ip1` directly in the cross-check test) so
       `assert_device_matches_host_ip1` can compare device vs host per-triple
       component-leading blocks for the <behavior> triples.
    5. Leave `launch_center_3c2e` (outer) and `launch_center_3c2e_typed` PUBLIC
       signatures byte-for-byte unchanged.
  </action>
  <verify>
    <automated>cargo build -p cintx-cubecl --features cpu && cargo build -p cintx-cubecl --features rocm && cargo test -p cintx-cubecl --features cpu center_3c2e</automated>
  </verify>
  <done>`center_3c2e_ip1_kernel` (#[cube(launch)]) + `run_3c2e_ip1_device<R>` exist; the
  ip1 path routes through the device on all five backend arms (rocm=HipRuntime); the
  existing ip1_tests + the new device-vs-host ip1 cross-check pass on CpuRuntime; spinor
  and nroots>5 guards fire before dispatch; outer/typed PUBLIC signatures intact; cpu +
  rocm builds compile.</done>
</task>

<task type="auto">
  <name>Task 3: Random ROCm int3c2e_ip1_sph idempotency oracle test (compile + collect)</name>
  <files>crates/cintx-oracle/tests/center_3c2e_parity.rs</files>
  <action>
    Append a `test_int3c2e_ip1_sph_random_rocm_idempotency` test modeled on
    `test_int3c1e_sph_random_rocm_idempotency` (center_3c1e_parity.rs L650-741) and the
    2c2e sibling. Same `#[cfg(feature = "rocm")] #[test] #[ignore]` + `CINTX_ROCM_ORACLE=1`
    env gate. Copy the deterministic `Lcg` struct (`#[cfg(feature = "rocm")]`-gated;
    seed e.g. `0x5ec0_3c2e_1234_5678`) and a `#[cfg(feature = "rocm")]
    build_random_3shell_3c2e(rng) -> (atm,bas,env,li,lj,lk)` that lays out 3 shells on
    3 distinct atoms with random `li,lj,lk ∈ {0,1,2}` (cap so the ip1 elevated nroots
    `(li+1+lj+0+lk)/2+1 <= 5` — i.e. draw li,lj,lk from {0,1,2} and SKIP/redraw any case
    whose elevated nroots>5, OR clamp `li+lj+lk <= 7`), random nprim ∈ {1..3}, random
    exps/coeffs/coords — reuse the env-pointer layout from `build_h2o_sto3g` in this file
    and the 3c1e `build_random_3shell` as the structural guide (constants `ATM_SLOTS,
    BAS_SLOTS, ANG_OF, ATOM_OF, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA` are already imported here).
    `n_cases = 64`. For each case allocate `n_elem = 3 * ni*nj*nk` (3-component
    derivative, component-leading), call `eval_raw(RawApiId::INT3C2E_IP1_SPH,
    Some(&mut out1), None, &[0,1,2], &atm,&bas,&env, None, None)` twice, assert
    idempotency at `atol=1e-12 / rtol=1e-10` with the inline abs+rel check the siblings
    use, accumulate `mismatch_count`, require `any_nonzero` across the suite, and emit a
    `PASS: rocm random int3c2e_ip1_sph mismatch_count=0 across {n_cases} cases` line.
    Do NOT modify the existing tests in this file.
  </action>
  <verify>
    <automated>cargo build -p cintx-oracle --features rocm --tests && cargo test -p cintx-oracle --features rocm test_int3c2e_ip1_sph_random_rocm_idempotency -- --list 2>&1 | grep -F test_int3c2e_ip1_sph_random_rocm_idempotency</automated>
  </verify>
  <done>`test_int3c2e_ip1_sph_random_rocm_idempotency` + `Lcg` + `build_random_3shell_3c2e`
  present, `#[cfg(feature="rocm")]`-gated identically to the 3c1e sibling; the rocm test
  build compiles and the test is collected (visible under `--list`).</done>
</task>

<task type="auto">
  <name>Task 4: Record the deferred ROCm device run in the SUMMARY</name>
  <files>(none — documentation only)</files>
  <action>
    The actual on-device rocm run is the orchestrator's post-merge job. In the plan
    SUMMARY, record the exact command the orchestrator must run and that the executor
    deferred it (no device run during execution):
    `CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm cargo test -p cintx-oracle --features rocm
    test_int3c2e_ip1_sph_random_rocm_idempotency -- --ignored --nocapture`
    (equivalently `cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle` runs the
    whole suite). State the expected PASS criterion: `mismatch_count=0` with non-zero
    output across 64 cases. Do NOT fabricate a device-run result.
  </action>
  <verify>
    <automated>echo "documentation task — verified by SUMMARY content; no command to run"</automated>
  </verify>
  <done>The SUMMARY contains the exact deferred rocm-run command and its PASS criterion,
  flagged as the orchestrator's post-merge step (not executed during this plan).</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| caller → eval_raw (raw atm/bas/env) | untrusted libcint-style integer/pointer arrays cross into the kernel dispatch; already validated by the existing raw-eval/query_workspace preflight |
| host launcher → device kernel | f64 buffers + u32 shape params marshaled to the ComputeClient; sizes derived host-side from validated shells |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-exs-01 | Denial of Service | center_3c2e_ip1_kernel / scalar kernel via elevated nroots | mitigate | Keep the `nroots>5` (scalar) and `grad_shape.nroots>5` (ip1) `UnsupportedApi` guards BEFORE any rys dispatch — prevents indexing a nonexistent rys_root and an oversized device buffer (verbatim from L362/L629). |
| T-exs-02 | Tampering | device read-back buffer sizing | mitigate | `run_*_device` sizes `cart_out` from host-derived `nci*ncj*nck` (×3 for ip1); the staging-write tails bound writes with `if dst < staging.len()` (verbatim). No caller-controlled length reaches an unchecked index. |
| T-exs-03 | Information Disclosure | spinor int3c2e_ip1 path | mitigate | Keep the `Representation::Spinor → UnsupportedApi` guard (L350) before any compute — the device ip1 kernel has no spinor transform and must never run for spinor. |
| T-exs-04 | Repudiation | idempotency / determinism | accept | Single work-item (`UNIT_POS==0`) ordered reduction keeps device output bit-deterministic; the random rocm idempotency test is the evidence. No further mitigation. |
</threat_model>

<verification>
- `cargo build -p cintx-cubecl --features cpu` and `--features rocm` both compile (the
  two `#[cube]` kernels accepted by the macro on both backends).
- `cargo test -p cintx-cubecl --features cpu center_3c2e` is green: device-vs-host
  cross-checks (scalar + ip1) within `1e-12 + 1e-10*|h|`; existing parity/smoke/ip1_tests pass.
- `cargo build -p cintx-oracle --features rocm --tests` compiles; the new random rocm
  test is collected under `--list`.
- Public `launch_center_3c2e` / `launch_center_3c2e_typed` signatures unchanged
  (registered `FamilyLaunchFn` cast in `kernels/mod.rs` still compiles).
- The on-device rocm run is deferred to the orchestrator (recorded in SUMMARY).
</verification>

<success_criteria>
- Two `#[cube(launch)]` device kernels (scalar + ip1), each generic over `F: Float`,
  using `#[comptime] nroots` + `comptime!` `rys_root1..5`, dispatched via
  `run_3c2e_device` / `run_3c2e_ip1_device` on all five backend arms (rocm=HipRuntime).
- Both numeric paths in `launch_center_3c2e_typed` route through the device; device
  compute at f64, output cast to F; all guards (nroots>5, spinor) fire pre-dispatch.
- Public signatures byte-for-byte unchanged; no manifest/RawApiId/capi changes; no
  edits to other kernel families or `.planning/phases/**`.
- `test_int3c2e_ip1_sph_random_rocm_idempotency` present, gated like its siblings,
  compiles and is collected; the deferred rocm device-run command is recorded in SUMMARY.
</success_criteria>

<output>
After completion, create `.planning/quick/260529-exs-center-3c2e-cubecl-kernel/260529-exs-SUMMARY.md`.
</output>
