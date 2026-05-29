---
phase: quick-260529-twi
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/cintx-cubecl/src/kernels/unstable.rs
  - crates/cintx-cubecl/src/kernels/unstable/mod.rs
  - crates/cintx-cubecl/src/kernels/unstable/shared.rs
  - crates/cintx-cubecl/src/kernels/unstable/origi.rs
  - crates/cintx-cubecl/src/kernels/unstable/grids.rs
  - crates/cintx-cubecl/src/kernels/unstable/breit.rs
  - crates/cintx-cubecl/src/kernels/unstable/origk.rs
  - crates/cintx-cubecl/src/kernels/unstable/ssc.rs
  - crates/cintx-oracle/tests/origi_random_rocm_parity.rs
  - crates/cintx-oracle/tests/grids_random_rocm_parity.rs
  - crates/cintx-oracle/tests/breit_random_rocm_parity.rs
  - crates/cintx-oracle/tests/origk_random_rocm_parity.rs
  - crates/cintx-oracle/tests/ssc_random_rocm_parity.rs
autonomous: true
requirements:
  - QUICK-260529-twi
must_haves:
  truths:
    - "`unstable.rs` is split into a behavior-preserving module directory `unstable/` (mod.rs + shared.rs + one file per family) with the existing `unstable_source_parity.rs` CPU vendor tests passing UNCHANGED after the refactor."
    - "Each of the 5 families (origi, grids, breit, origk, ssc) runs a real CubeCL `#[cube(launch)]` device kernel generic over `F: Float` for its SCALAR path, dispatched on every `ResolvedBackend` arm including `ResolvedBackend::Rocm(client) => run_*_device::<cubecl_hip::HipRuntime>(...)`."
    - "Each family's scalar device kernel compiles AND launches for both f64 and f32 monomorphizations."
    - "Each family has an in-crate `#[cfg(test)] #[cfg(feature = \"cpu\")]` device-vs-host cross-check on CpuRuntime at f64 (atol=1e-12) plus an f32 genericity smoke test."
    - "Each family has a new randomized ROCm vendor-parity oracle test that drives the device path on ROCm and shows 0 divergence vs the libcint 6.1.3 vendor oracle (mismatch_count=0, any_nonzero=true)."
    - "The 5 `pub fn launch_*` FamilyLaunchFn signatures are unchanged and still registered for canonical_family origi/grids/breit/origk/ssc in `kernels/mod.rs` (registration UNCHANGED)."
  artifacts:
    - path: "crates/cintx-cubecl/src/kernels/unstable/mod.rs"
      provides: "Module facade re-exporting the 5 `launch_*` fns + shared `pub(crate)` helpers so `kernels/mod.rs` imports are unchanged"
      contains: "pub use"
    - path: "crates/cintx-cubecl/src/kernels/unstable/origi.rs"
      provides: "`#[cube(launch)] origi_scalar_kernel<F>`, `run_origi_scalar_device<R>`, `run_origi_scalar_on_backend`, host `launch_origi`"
      contains: "#[cube(launch)]"
    - path: "crates/cintx-cubecl/src/kernels/unstable/grids.rs"
      provides: "`#[cube(launch)] grids_scalar_kernel<F>` (nuclear-like scalar path), device dispatch, host `launch_grids`"
      contains: "#[cube(launch)]"
    - path: "crates/cintx-cubecl/src/kernels/unstable/breit.rs"
      provides: "`#[cube(launch)] breit_g_kernel<F>` (4-center G-tensor VRR/HRR on device), device dispatch, host `launch_breit`"
      contains: "#[cube(launch)]"
    - path: "crates/cintx-cubecl/src/kernels/unstable/origk.rs"
      provides: "`#[cube(launch)] origk_scalar_kernel<F>` (3c1e r2/r4/r6), device dispatch, host `launch_origk`"
      contains: "#[cube(launch)]"
    - path: "crates/cintx-cubecl/src/kernels/unstable/ssc.rs"
      provides: "`#[cube(launch)] ssc_scalar_kernel<F>` (3c2e ssc), device dispatch, host `launch_ssc`"
      contains: "#[cube(launch)]"
    - path: "crates/cintx-oracle/tests/ssc_random_rocm_parity.rs"
      provides: "Randomized ROCm vendor-parity oracle for int3c2e_sph_ssc"
      contains: "vendor_int3c2e_sph_ssc"
  key_links:
    - from: "kernels/mod.rs resolve_family_name"
      to: "unstable::launch_{origi,grids,breit,origk,ssc}"
      via: "pub use facade in unstable/mod.rs (registration unchanged)"
      pattern: "unstable::launch_"
    - from: "launch_{family} host wrappers"
      to: "run_{family}_scalar_device::<cubecl_hip::HipRuntime>"
      via: "5-arm match backend { ResolvedBackend::Rocm(client) => ... }"
      pattern: "ResolvedBackend::Rocm.*run_.*_scalar_device"
    - from: "{family}_random_rocm_parity.rs"
      to: "vendor_ffi::vendor_int*"
      via: "element-wise comparison after eval_raw on the rocm backend"
      pattern: "vendor_int"
---

<objective>
Refactor `crates/cintx-cubecl/src/kernels/unstable.rs` — currently one 130KB
host-only f64 file holding all five unstable-source-api family launchers — into a
per-family module directory, then port each family's SCALAR path from a host f64
loop to a real CubeCL `#[cube(launch)]` device kernel generic over `F: Float`,
dispatched on all five `ResolvedBackend` arms (CPU / Wgpu / Cuda / ROCm-HIP /
Metal). Add a randomized ROCm vendor-parity oracle test per family proving 0
divergence vs vendored libcint 6.1.3 on the device backend.

This follows the proven family-port template from `one_electron.rs`
(`one_electron_scalar_kernel<F>` + `run_1e_scalar_device<R>` + 5-arm backend
dispatch) and `two_electron.rs` (the most recent 2e device port), and the
quick-260529-imi / quick-260529-q4k PLAN structure.

STRUCTURE / PARALLELIZATION STRATEGY:
- Task 1 is a PURE MECHANICAL refactor: split `unstable.rs` into
  `unstable/{mod.rs, shared.rs, origi.rs, grids.rs, breit.rs, origk.rs, ssc.rs}`,
  behavior-preserving (the unchanged `unstable_source_parity.rs` CPU vendor tests
  must pass). This is the SERIAL gate — it must land first.
- Tasks 2–6 are one device-kernel port per family. After Task 1 lands, each
  touches ONLY its own family file (`unstable/<family>.rs`) plus its own new
  oracle test file → they are INDEPENDENT and PARALLELIZABLE across
  worktree-isolated executors with NO merge conflicts (zero `files_modified`
  overlap between Task 2..6).

HOST/DEVICE SPLIT (Phase 21 D-04 convention, honest + CLAUDE.md-compliant):
- ON DEVICE: per-primitive G-tensor build (VRR/HRR) + Cartesian contraction,
  accumulated into a per-contraction Cartesian block buffer, in f64-internal /
  F-output.
- ON HOST: the cart_to_sph / cart_to_spinor coefficient-table transforms (their
  tables are host-only) + the AO scatter into `staging`.
- DEFERRED-TO-HOST per family (documented in code + SUMMARY, mirrors the 1e/2e
  ports deferring gradient/spinor sub-paths): the higher-derivative ip / ipip /
  ipvip / ip2 sub-paths and breit's gout operator ladder + spinor transform stay
  on host this task. Each family file states its split + deferrals in a comment.

Purpose: move the core unstable-family numeric work onto the CubeCL device
backend per the project's "CubeCL is the primary compute backend" constraint,
completing the unstable families' device port to match the 2c2e/3c2e/ecp/1e/2e
precedent, while making the giant file maintainable and per-family parallelizable.

Output:
- `unstable/` module directory (mod.rs facade + shared.rs + 5 family files).
- 5 family files each with a `#[cube(launch)] <family>_scalar_kernel<F>`, a
  `run_<family>_scalar_device<R: Runtime>` dispatcher, a
  `run_<family>_scalar_on_backend` 5-arm match, in-crate device-vs-host + f32
  tests, and the unchanged-signature host `launch_<family>`.
- 5 new `crates/cintx-oracle/tests/<family>_random_rocm_parity.rs` oracles.
</objective>

<execution_context>
@/home/user/Documents/workspace/cintx/.claude/get-shit-done/workflows/execute-plan.md
</execution_context>

<context>
@./CLAUDE.md

# THE FILE TO REFACTOR + PORT (currently host-side f64, 5 families, ~3511 lines)
@crates/cintx-cubecl/src/kernels/unstable.rs

# THE PORT TEMPLATE (most recent #[cube] family ports — study these end to end)
# one_electron.rs: scalar kernel #[cube(launch)] at L200, run_1e_scalar_device<R>
#   at L1482, run_1e_scalar_on_backend (5-arm, Rocm→HipRuntime) at L1605, host
#   launch_one_electron_typed at L2448, in-crate test mod at L3025 incl.
#   device-vs-host cross-check + test_one_electron_scalar_kernel_generic_f32 (L3727),
#   MAX_DEVICE_NROOTS guard at L42.
@crates/cintx-cubecl/src/kernels/one_electron.rs

# two_electron.rs: the most recent port (4-center scalar #[cube] + run_2e_scalar_device
#   + 5-arm dispatch + device-vs-host tests). The closest analog for breit/ssc 4c/3c2e.
@crates/cintx-cubecl/src/kernels/two_electron.rs

# Registration (FROZEN — do NOT change): the 5 unstable families are registered at
# mod.rs L45-54, ALL gated `#[cfg(feature = "unstable-source-api")]`:
#   "origi" => unstable::launch_origi, "grids" => unstable::launch_grids,
#   "breit" => unstable::launch_breit, "origk" => unstable::launch_origk,
#   "ssc"  => unstable::launch_ssc.
# After the Task-1 split these names MUST still resolve via the unstable/mod.rs facade.
@crates/cintx-cubecl/src/kernels/mod.rs

# Backend enum + ROCm arm
@crates/cintx-cubecl/src/backend/mod.rs
@crates/cintx-cubecl/src/backend/rocm_backend.rs

# THE ORACLE REFERENCE the device ports must match (existing HOST CPU vendor parity
# for all 5 families — these tests MUST keep passing unchanged after Task 1):
@crates/cintx-oracle/tests/unstable_source_parity.rs

# ROCm oracle test templates (LCG + random H2O/STO-3G + #[ignore] + CINTX_ROCM_ORACLE=1):
@crates/cintx-oracle/tests/one_electron_random_rocm_parity.rs
@crates/cintx-oracle/tests/two_electron_random_rocm_parity.rs

# AUTHORITATIVE #[cube] authoring rules — MUST read before writing any kernel.
# Top pitfalls: no plain-fn calls inside #[cube]; no if-EXPRESSIONS (use if/else
# statements + pre-init mut vars; comptime! branches for nroots/op_kind); F::exp/
# F::sqrt not std; u32/i32 ints only (cast `as usize` at index sites); no continue/break.
@docs/manual/Cubecl/Cubecl_basic_operations.md
@docs/manual/Cubecl/Cubecl_conditionals.md
@docs/manual/Cubecl/Cubecl_generics.md

<interfaces>
<!-- Contracts the executor needs. From the codebase — no exploration required. -->

UNSTABLE FAMILY INVENTORY (current `unstable.rs`, host f64 today):
  origi → pub fn launch_origi (L1295)  — 1e r2/r4 + ip2. FROZEN signature:
    (backend: &ResolvedBackend, plan: &ExecutionPlan, _spec: &SpecializationKey,
     staging: &mut [f64]) -> Result<ExecutionStats, cintxRsError>
    Symbols (RawApiId::Symbol): int1e_r2_origi_sph, int1e_r4_origi_sph,
      int1e_r2_origi_ip2_sph, int1e_r4_origi_ip2_sph.
    Vendor: vendor_int1e_r2_origi_sph, _r4_origi_sph, _r2_origi_ip2_sph, _r4_origi_ip2_sph.
    Host helpers in-file: origi_variant (L998), g1e_r_i (L1015), g1e_d_j (L1030),
      contract_origi_r2/r4 (L1052/L1109), contract_origi_r2_ip2/r4_ip2 (L1151/L1221),
      fill_g_tensor_origi (L1423).
  grids → pub fn launch_grids (L829)  — 1e grid-point ops, 4-shell API (shls:[i32;4]).
    Symbols: int1e_grids_sph (scalar), int1e_grids_ip_sph, int1e_grids_ipvip_sph,
      int1e_grids_spvsp_sph, int1e_grids_ipip_sph.
    Vendor: vendor_int1e_grids_sph + _ip_sph/_ipvip_sph/_spvsp_sph/_ipip_sph.
    Host helpers: grids_contract_nuclear_like (L159, the SCALAR path), grids_contract_ip
      (L285), _ipip (L388), _ipvip (L520), _spvsp (L648), launch_grids_kernel (L695),
      grids_stats (L810). Uses GridsEnvParams (NGRIDS, PTR_GRIDS env params).
  breit → pub fn launch_breit (L2382) — 2e Breit, 4-center, SPINOR-ONLY (rejects
    non-Spinor at L2393). MOST COMPLEX. Symbols: int2e_breit_r1p2_spinor,
      int2e_breit_r2p2_spinor. Vendor: vendor_int2e_breit_r1p2_spinor, _r2p2_spinor.
    Host helpers: BreitShape (L1470), build_breit_shape (L1499), vrr_fill_axis_breit
      (L1552), hrr_{lj2d,kj2d,il2d,ik2d}_4d_breit (L1617/1653/1689/1728),
      fill_g_tensor_breit (L1772), nabla1{i,j,l}_breit (L1900/1945/1998),
      x1{j,l}_breit (L2050/2091), gout_breit_r1p2/r2p2 (L2150/2267).
  origk → pub fn launch_origk (L2867) — 3c1e r2/r4/r6 origk + ip1 variants.
    Symbols: int3c1e_r2_origk_sph, _r4_origk_sph, _r6_origk_sph (scalar);
      int3c1e_ip1_r2_origk_sph, _ip1_r4_origk_sph, _ip1_r6_origk_sph (gradient).
    Vendor: vendor_int3c1e_{r2,r4,r6}_origk_sph + _ip1_{r2,r4,r6}_origk_sph.
    Host helpers: origk_variant (L2569), g1e_d_i_3c1e (L2586), contract_origk (L2623),
      contract_origk_ip1 (L2732), fill_g_tensor_3c1e_origk (L3030).
  ssc → pub fn launch_ssc (L3130) — 3c2e ssc. Symbol: int3c2e_sph_ssc.
    Vendor: vendor_int3c2e_sph_ssc. Host helpers: transpose_ij_3idx (L3255),
      cart_to_sph_3c2e_ssc (L3273), fill_g_tensor_3c2e_ssc (L3313), split_ij_hrr_ssc
      (L3422), contract_3c2e_ssc (L3476).

CROSS-FAMILY SHARED helpers at top of unstable.rs (move to shared.rs, keep
`pub(crate)` where re-exported):
  const SQRTPI (L22), fn common_fac_sp (L26), fn cart_comps (L35),
  fn nabla_i_host (L59), nabla_j_host (L93), apply_nabla_i_3axis (L122),
  apply_nabla_j_3axis (L137), fn make_exec_stats (L968).
  Imports used across families: ResolvedBackend, obara_saika::{hrr_step_host,
  vrr_step_host, vrr_2e_step_host}, pdata::{PairData, compute_pdata_host},
  rys::{rys_root1_host, rys_root2_host, rys_roots_host}, SpecializationKey,
  c2s::{cart_to_sph_1e, cart_to_sph_3c1e, cart_to_sph_3c2e, ncart, nsph},
  c2spinor::cart_to_spinor_sf_4d, Representation, cintxRsError, ExecutionPlan,
  ExecutionStats, planner::GridsEnvParams, std::f64::consts::PI.

DEVICE-SIDE primitives + idioms to reuse verbatim (from one_electron.rs / center_2c2e.rs):
  use crate::math::rys::{rys_root1, rys_root2, rys_root3, rys_root4, rys_root5};
  use cubecl::Runtime; use cubecl::client::ComputeClient; use cubecl::prelude::*;
  const MAX_DEVICE_NROOTS: usize = 5;   // fail-closed nroots guard
  // device dispatch idiom (mirror run_1e_scalar_device):
  //   client.create_from_slice(f64::as_bytes(slice));   // inputs + zeroed scratch/out
  //   <kernel>::launch::<f64, R>(client, CubeCount::Static(1,1,1), CubeDim::new_1d(1), ...args);
  //   f64::from_bytes(&client.read_one_unchecked(out_h))[0..out_len].to_vec();
  // 5-arm backend match (copy from run_1e_scalar_on_backend, cfg-gated per feature):
  //   #[cfg(feature="cpu")]  Cpu(c)    => run_::<cubecl::cpu::CpuRuntime>(c, ...)
  //   #[cfg(feature="wgpu")] Wgpu(c,_) => run_::<cubecl_wgpu::WgpuRuntime>(c, ...)
  //   #[cfg(feature="cuda")] Cuda(c)   => run_::<cubecl_cuda::CudaRuntime>(c, ...)
  //   #[cfg(feature="rocm")] Rocm(c)   => run_::<cubecl_hip::HipRuntime>(c, ...)
  //   #[cfg(feature="metal")]Metal(c,_)=> run_::<cubecl_wgpu::WgpuRuntime>(c, ...)

ORACLE TEST CONTRACT (unstable families use Symbol IDs, double-gate + unstable feature):
  use cintx_compat::raw::{..., RawApiId, eval_raw};   // drive via RawApiId::Symbol("int...")
  use cintx_oracle::vendor_ffi;                        // #[cfg(has_vendor_libcint)]
  // module gate: #![cfg(feature = "rocm")]; each #[test] #[ignore]; runtime
  //   CINTX_ROCM_ORACLE=1 assert; vendor comparison `#[cfg(has_vendor_libcint)]`.
  // build_h2o_sto3g fixture: cintx_oracle::fixtures::build_h2o_sto3g (1e/3c paths);
  //   grids needs the build_h2o_sto3g_grids variant (NGRIDS/PTR_GRIDS env) — copy
  //   from unstable_source_parity.rs L212.
  // LCG (Numerical Recipes) + jitter (exp ×uniform(0.7,1.4), coord ±0.3) — copy
  //   from one_electron_random_rocm_parity.rs.
  // RUN COMMAND (note: unstable-source-api MUST be in feature list):
  //   CINTX_ORACLE_BUILD_VENDOR=1 CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm \
  //   cargo test -p cintx-oracle --features cpu,rocm,unstable-source-api \
  //   --test <family>_random_rocm_parity -- --ignored
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1 [SERIAL GATE]: Behavior-preserving split of unstable.rs into a per-family module directory</name>
  <files>crates/cintx-cubecl/src/kernels/unstable.rs (DELETE), crates/cintx-cubecl/src/kernels/unstable/mod.rs, crates/cintx-cubecl/src/kernels/unstable/shared.rs, crates/cintx-cubecl/src/kernels/unstable/origi.rs, crates/cintx-cubecl/src/kernels/unstable/grids.rs, crates/cintx-cubecl/src/kernels/unstable/breit.rs, crates/cintx-cubecl/src/kernels/unstable/origk.rs, crates/cintx-cubecl/src/kernels/unstable/ssc.rs</files>
  <action>
PURE MECHANICAL refactor — NO behavior change, NO algorithm edits. Convert the
single-file module `kernels/unstable.rs` into a directory module
`kernels/unstable/`. Do this by MOVING existing code verbatim, not rewriting it.

1. Create `crates/cintx-cubecl/src/kernels/unstable/shared.rs` and move the
   cross-family items currently at the top of unstable.rs into it:
   `SQRTPI` (L22), `common_fac_sp` (L26), `cart_comps` (L35), `nabla_i_host`
   (L59), `nabla_j_host` (L93), `apply_nabla_i_3axis` (L122), `apply_nabla_j_3axis`
   (L137), `make_exec_stats` (L968). Mark each `pub(crate)` (or `pub(super)`) so
   the family files can use them. Add the necessary `use` imports at the top of
   shared.rs (only the ones these helpers reference).

2. Create one file per family and MOVE that family's launch fn + ALL its
   private helpers verbatim:
   - `origi.rs` ← launch_origi + OrigiVariant/origi_variant, g1e_r_i, g1e_d_j,
     contract_origi_r2/r4, contract_origi_r2_ip2/r4_ip2, fill_g_tensor_origi.
   - `grids.rs` ← launch_grids + grids_contract_nuclear_like/_ip/_ipip/_ipvip/
     _spvsp, launch_grids_kernel, grids_stats.
   - `breit.rs` ← launch_breit + BreitShape/build_breit_shape, vrr_fill_axis_breit,
     hrr_*_4d_breit (4), fill_g_tensor_breit, nabla1{i,j,l}_breit, x1{j,l}_breit,
     gout_breit_r1p2/r2p2.
   - `origk.rs` ← launch_origk + OrigkVariant/origk_variant, g1e_d_i_3c1e,
     contract_origk, contract_origk_ip1, fill_g_tensor_3c1e_origk.
   - `ssc.rs` ← launch_ssc + transpose_ij_3idx, cart_to_sph_3c2e_ssc,
     fill_g_tensor_3c2e_ssc, split_ij_hrr_ssc, contract_3c2e_ssc.
   At the top of each family file add `use super::shared::*;` (or named imports)
   for the shared helpers it uses, plus that family's own external `use` lines
   (copy from the original unstable.rs import block — only what the family needs).

3. Create `crates/cintx-cubecl/src/kernels/unstable/mod.rs` as the facade. It
   must: declare `mod shared; mod origi; mod grids; mod breit; mod origk; mod ssc;`
   and re-export the 5 public launchers so existing callers are unchanged:
   `pub use origi::launch_origi; pub use grids::launch_grids;
   pub use breit::launch_breit; pub use origk::launch_origk; pub use ssc::launch_ssc;`
   Also re-export any item that other modules import from `kernels::unstable::*`
   (grep first: `grep -rn "kernels::unstable::" crates/` and
   `grep -rn "unstable::" crates/cintx-cubecl/src/kernels/mod.rs`). Move the
   module-level `//!` doc comment from old unstable.rs to mod.rs.

4. DELETE `crates/cintx-cubecl/src/kernels/unstable.rs` (it is replaced by the
   directory). Confirm `kernels/mod.rs` still says `mod unstable;` /
   `#[cfg(feature = "unstable-source-api")] mod unstable;` — directory modules
   resolve identically, so this line is UNCHANGED.

5. Resolve any `pub(crate)` vs `pub(super)` / import-path fallout from the split
   (e.g. a helper used by two families must live in shared.rs). Do NOT change any
   function body, signature, or numeric constant. The diff must be move-only.

Authoring note: keep `#[cfg(feature = "unstable-source-api")]` gating intact —
the whole module is feature-gated; family files inherit it via mod.rs.
  </action>
  <verify>
    <automated>cd /home/user/Documents/workspace/cintx && cargo build -p cintx-cubecl --features cpu,unstable-source-api 2>&1 | tail -15 && CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu,unstable-source-api --test unstable_source_parity 2>&1 | tail -25</automated>
  </verify>
  <done>
    `unstable.rs` is gone; `unstable/{mod.rs,shared.rs,origi.rs,grids.rs,breit.rs,origk.rs,ssc.rs}`
    exist. `cargo build -p cintx-cubecl --features cpu,unstable-source-api` is
    clean. The unchanged `unstable_source_parity.rs` CPU vendor-parity tests for
    all 5 families pass under `CINTX_ORACLE_BUILD_VENDOR=1`. `grep -c "unstable::launch_"
    crates/cintx-cubecl/src/kernels/mod.rs` still returns 5 (registration intact).
    No function body or numeric constant changed (move-only diff).
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2 [PARALLEL]: Port origi scalar (int1e_r2/r4_origi_sph) to a #[cube(launch)] device kernel + ROCm oracle</name>
  <files>crates/cintx-cubecl/src/kernels/unstable/origi.rs, crates/cintx-oracle/tests/origi_random_rocm_parity.rs</files>
  <behavior>
    - Device-vs-host (`#[cfg(test)] #[cfg(feature = "cpu")]`): for quartets/pairs
      (s,s), (p,s), (s,p), (p,p), (d,s) on a 2-center fixture,
      `run_origi_scalar_device::<cubecl::cpu::CpuRuntime>` reproduces the host
      `contract_origi_r2`/`contract_origi_r4(fill_g_tensor_origi(...))` Cartesian
      buffer for r2 AND r4 within atol=1e-12, rtol=1e-10.
    - Genericity: the kernel launches for `F = f32` (s-s case yields finite output).
  </behavior>
  <action>
SCOPE: port ONLY the SCALAR r2/r4 origi paths (int1e_r2_origi_sph,
int1e_r4_origi_sph) to a device `#[cube(launch)]` kernel. DEFER the ip2 variants
(int1e_r2_origi_ip2_sph, int1e_r4_origi_ip2_sph) to host — document this at the
top of origi.rs (matches the 1e/2e ports deferring derivative sub-paths; ip2 adds
a second-derivative ladder that is a separable port).

In origi.rs (mirror one_electron.rs scalar template — origi is the CLOSEST analog,
a 1e r^n operator): add device imports (`rys_root1..5`, `cubecl::prelude::*`,
`MAX_DEVICE_NROOTS`). Write `#[cube(launch)] #[allow(clippy::too_many_arguments)]
fn origi_scalar_kernel<F: Float + CubeElement>(...)` running a single work item
(`if UNIT_POS == 0u32`). INLINE the host pipeline `fill_g_tensor_origi` →
`contract_origi_r2`/`contract_origi_r4`, accumulated over primitive pairs and
contraction pairs:
  - Inputs Array<F>: exps_i/j, coeff_i/j. Scratch mut Array<F>: g. Output mut
    Array<F>: cart_out (nctr_i*nctr_j*nfi*nfj, i fastest).
  - Scalars F: ri/rj coords (6), origin coords, common_factor.
  - Runtime u32: li,lj,nprim_i/j,nctr_i/j, strides/nmax computed host-side.
  - `#[comptime] r_power: u32` (2 or 4) selecting the r2 vs r4 contraction branch
    via `comptime!(r_power == 2u32)`. `#[comptime] nroots: u32` selecting
    rys_root{1..5} via `comptime!`.
  Follow #[cube] rules: u32/i32 counters with `as usize` at index sites,
  F::cast_from for int→F, F::exp/F::sqrt, no plain-fn calls, no if-EXPRESSIONS
  (if/else statements + pre-init mut vars), no continue/break.

Write `run_origi_scalar_device<R: Runtime>(client, ...f64 slices..., r_power: u32,
nroots: u32, out_len) -> Vec<f64>` (mirror run_1e_scalar_device: create_from_slice
inputs + zeroed scratch/out, monomorphize via a `macro_rules! launch_with` runtime
match on (r_power, nroots), read back via read_one_unchecked + f64::from_bytes).
Write `run_origi_scalar_on_backend(backend: &ResolvedBackend, ...)` with the 5-arm
cfg-gated match (Rocm → `run_origi_scalar_device::<cubecl_hip::HipRuntime>`).

In `launch_origi`: keep the FROZEN signature. For the SCALAR r2/r4 variants, add a
`shape.nroots > MAX_DEVICE_NROOTS` fail-closed guard, then replace the host
`fill_g_tensor_origi`+`contract_origi_*` accumulation with a call to
`run_origi_scalar_on_backend(...)` returning cart blocks; the EXISTING
`cart_to_sph_1e` + AO scatter into `staging` stays UNCHANGED (host part of split).
For the ip2 variants, leave the existing host path in place (early-branch on
`origi_variant`).

Add the in-crate device-vs-host + f32 tests per the <behavior> block in a
`#[cfg(test)] #[cfg(feature = "cpu")] mod tests` in origi.rs.

Create `crates/cintx-oracle/tests/origi_random_rocm_parity.rs`: `#![cfg(feature =
"rocm")]`, LCG-jittered H2O/STO-3G slab (use `cintx_oracle::fixtures::build_h2o_sto3g`
+ jitter copied from one_electron_random_rocm_parity.rs), `#[test] #[ignore]` +
`CINTX_ROCM_ORACLE=1` assert, drive cintx via `eval_raw(RawApiId::Symbol("int1e_r2_origi_sph"))`
and `("int1e_r4_origi_sph")` over a varied shell-pair table, compare vs
`#[cfg(has_vendor_libcint)] vendor_ffi::vendor_int1e_r2_origi_sph`/`_r4_origi_sph`
at atol=1e-12 rtol=0, assert mismatch_count==0 AND any_nonzero==true over ~48 cases.
  </action>
  <verify>
    <automated>cd /home/user/Documents/workspace/cintx && cargo test -p cintx-cubecl --features cpu,unstable-source-api --lib kernels::unstable::origi 2>&1 | tail -20 && cargo build -p cintx-cubecl --features rocm,unstable-source-api 2>&1 | tail -8 && CINTX_ORACLE_BUILD_VENDOR=1 CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm cargo test -p cintx-oracle --features cpu,rocm,unstable-source-api --test origi_random_rocm_parity -- --ignored 2>&1 | tail -20</automated>
  </verify>
  <done>
    origi scalar r2/r4 run a `#[cube(launch)]` device kernel on all 5 backend arms
    (Rocm→HipRuntime). In-crate device-vs-host + f32 tests pass on CpuRuntime;
    `origi_random_rocm_parity.rs` reports mismatch_count=0, any_nonzero=true vs
    vendor on ROCm. ip2 variants documented as host-deferred. `launch_origi`
    signature + origi registration unchanged.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 3 [PARALLEL]: Port grids scalar (int1e_grids_sph) to a #[cube(launch)] device kernel + ROCm oracle</name>
  <files>crates/cintx-cubecl/src/kernels/unstable/grids.rs, crates/cintx-oracle/tests/grids_random_rocm_parity.rs</files>
  <behavior>
    - Device-vs-host (`#[cfg(test)] #[cfg(feature = "cpu")]`): for shell pairs
      (s,s),(p,s),(s,p),(p,p) against a small fixed grid (e.g. 2 grid points),
      `run_grids_scalar_device::<cubecl::cpu::CpuRuntime>` reproduces the host
      `grids_contract_nuclear_like(...)` Cartesian buffer within atol=1e-12.
    - Genericity: kernel launches for `F = f32` (s-s yields finite output).
  </behavior>
  <action>
SCOPE: port ONLY the SCALAR nuclear-like grids path (int1e_grids_sph, host fn
`grids_contract_nuclear_like` at the old L159). DEFER the derivative variants
ip / ipip / ipvip / spvsp to host — document at the top of grids.rs (they reuse
nabla ladders + spvsp σ-coupling and are separable ports, matching the 1e/2e
deferral precedent).

grids is a 4-shell-API grid-point operator: it sums a nuclear-attraction-like
Boys/Rys kernel over NGRIDS grid points (env params NGRIDS, PTR_GRIDS via
GridsEnvParams). Treat each grid point like a point charge at its coordinate.

In grids.rs: write `#[cube(launch)] grids_scalar_kernel<F: Float + CubeElement>`
running a single work item, INLINING `grids_contract_nuclear_like`:
  - Inputs Array<F>: exps_i/j, coeff_i/j, AND grid_coords (3*ngrids, the per-grid
    point charges treated as unit, matching the host fn's grid loop). Scratch mut
    Array<F>: g. Output mut Array<F>: cart_out (ngrids*nctr_i*nctr_j*nfi*nfj OR
    summed-over-grids per the host layout — MATCH the host `grids_contract_nuclear_like`
    output layout exactly; read it to confirm whether output is per-grid or
    grid-summed before writing the kernel).
  - Scalars F: ri/rj coords, common_factor. Runtime u32: li,lj,nprim/nctr, strides,
    ngrids. `#[comptime] nroots: u32` rys_root{1..5}.
  - Loop over grid points with a runtime u32 counter (`for gp in 0..ngrids`),
    reading grid_coords[(3*gp + axis) as usize]; accumulate the Rys/Boys
    nuclear-like contribution exactly as the host fn does.
  Follow #[cube] rules (u32/i32 + as usize, F::exp/F::sqrt, no if-expr, no
  plain-fn calls, no continue/break).

Write `run_grids_scalar_device<R: Runtime>` + `run_grids_scalar_on_backend` (5-arm,
Rocm→HipRuntime), passing the grid coords through as an extra f64 slice.

In `launch_grids`: keep the FROZEN signature. Parse GridsEnvParams as today. For
the scalar nuclear-like operator add the MAX_DEVICE_NROOTS guard then call
`run_grids_scalar_on_backend(...)`; the existing `cart_to_sph_1e` + scatter stays.
Leave the ip/ipip/ipvip/spvsp host paths untouched (branch before device dispatch).

In-crate device-vs-host + f32 tests per <behavior> in grids.rs.

Create `crates/cintx-oracle/tests/grids_random_rocm_parity.rs`: `#![cfg(feature =
"rocm")]`, build the grids fixture with NGRIDS/PTR_GRIDS env (COPY
`build_h2o_sto3g_grids` from unstable_source_parity.rs L212 + add LCG jitter to
exps/coords/grid points), `#[test] #[ignore]` + `CINTX_ROCM_ORACLE=1`, drive via
`eval_raw(RawApiId::Symbol("int1e_grids_sph"))`, compare vs
`#[cfg(has_vendor_libcint)] vendor_ffi::vendor_int1e_grids_sph` at atol=1e-12
rtol=0, assert mismatch_count==0 AND any_nonzero==true over ~48 cases.
  </action>
  <verify>
    <automated>cd /home/user/Documents/workspace/cintx && cargo test -p cintx-cubecl --features cpu,unstable-source-api --lib kernels::unstable::grids 2>&1 | tail -20 && cargo build -p cintx-cubecl --features rocm,unstable-source-api 2>&1 | tail -8 && CINTX_ORACLE_BUILD_VENDOR=1 CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm cargo test -p cintx-oracle --features cpu,rocm,unstable-source-api --test grids_random_rocm_parity -- --ignored 2>&1 | tail -20</automated>
  </verify>
  <done>
    grids scalar (int1e_grids_sph) runs a `#[cube(launch)]` device kernel on all 5
    backend arms (Rocm→HipRuntime). In-crate device-vs-host + f32 tests pass;
    `grids_random_rocm_parity.rs` reports mismatch_count=0, any_nonzero=true vs
    vendor on ROCm. ip/ipip/ipvip/spvsp documented as host-deferred. `launch_grids`
    signature + grids registration unchanged.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 4 [PARALLEL]: Port breit 4-center G-tensor (VRR/HRR) to a #[cube(launch)] device kernel; keep gout+spinor host + ROCm oracle</name>
  <files>crates/cintx-cubecl/src/kernels/unstable/breit.rs, crates/cintx-oracle/tests/breit_random_rocm_parity.rs</files>
  <behavior>
    - Device-vs-host (`#[cfg(test)] #[cfg(feature = "cpu")]`): for a representative
      breit quartet set on a 2-center fixture (covering all four HRR branches via
      li>lj / lk>ll selection), `run_breit_g_device::<cubecl::cpu::CpuRuntime>`
      reproduces the host `fill_g_tensor_breit(...)` G-tensor buffer (the device
      deliverable for breit) within atol=1e-12, rtol=1e-10.
    - Genericity: the G-tensor kernel launches for `F = f32` (finite output).
  </behavior>
  <action>
SCOPE (breit is the MOST COMPLEX, spinor-only family): port ONLY the per-quartet
G-tensor BUILD (`fill_g_tensor_breit` = vrr_fill_axis_breit + the four
hrr_*_4d_breit transfers, the analog of the 2e VRR/HRR) to a device
`#[cube(launch)]` kernel. KEEP ON HOST (document at top of breit.rs with rationale,
matching the spinor-deferral precedent): (a) the gout_breit_r1p2/r2p2 operator
ladder + nabla1{i,j,l}_breit + x1{j,l}_breit (the Breit-specific gradient/operator
machinery applied AFTER the G-tensor — a large separable port), and (b) the
`cart_to_spinor_sf_4d` transform (breit is spinor-only; the spinor coefficient
table + the documented KET-major→BRA-major transpose gotcha stay host-side per the
1e spinor precedent).

In breit.rs: write `#[cube(launch)] breit_g_kernel<F: Float + CubeElement>` running
a single work item, INLINING `fill_g_tensor_breit` (VRR per-axis via
vrr_fill_axis_breit + the four hrr_*_4d_breit branches selected by runtime
`if ibase==1u32`/`if kbase==1u32` STATEMENTS — model on two_electron.rs's 4-branch
HRR device port). Output mut Array<F>: the 3-axis G-tensor block (3*g_size per
quartet), accumulated over primitive+contraction quads. Inputs/scalars/strides as
for the 2e kernel (exps/coeffs Arrays, 12 coords F, runtime u32 li_e/lj_e/lk_e/ll_e,
di/dk/dl/dj/g_size/nmax/mmax, ibase/kbase, `#[comptime] nroots`). Follow all
#[cube] rules.

Write `run_breit_g_device<R: Runtime>` + `run_breit_g_on_backend` (5-arm,
Rocm→HipRuntime) returning the f64 G-tensor block.

In `launch_breit`: keep the FROZEN signature + the spinor-only guard (L2393) + the
r1p2/r2p2 branch. Add a `shape.nroots > MAX_DEVICE_NROOTS` guard. Replace the host
`fill_g_tensor_breit` CALL with `run_breit_g_on_backend(...)`; then the EXISTING
host gout_breit + nabla/x1 operator application AND `cart_to_spinor_sf_4d` +
scatter stay UNCHANGED, consuming the device-built G-tensor exactly as before.

In-crate device-vs-host (G-tensor) + f32 tests per <behavior> in breit.rs.

Create `crates/cintx-oracle/tests/breit_random_rocm_parity.rs`: `#![cfg(feature =
"rocm")]`, LCG-jittered H2O/STO-3G slab, `#[test] #[ignore]` + `CINTX_ROCM_ORACLE=1`,
SPINOR representation (breit is spinor-only — use a NON-SQUARE shell quartet per
the spinor gotcha so a transpose bug cannot hide), drive via
`eval_raw(RawApiId::Symbol("int2e_breit_r1p2_spinor"))` and `("int2e_breit_r2p2_spinor")`,
compare vs `#[cfg(has_vendor_libcint)] vendor_ffi::vendor_int2e_breit_r1p2_spinor`/
`_r2p2_spinor` (complex/interleaved — match the comparison shape in
unstable_source_parity.rs::test_int2e_breit_r1p2_spinor_oracle_parity at L787) at
atol=1e-12 rtol=0, assert mismatch_count==0 AND any_nonzero==true.
  </action>
  <verify>
    <automated>cd /home/user/Documents/workspace/cintx && cargo test -p cintx-cubecl --features cpu,unstable-source-api --lib kernels::unstable::breit 2>&1 | tail -20 && cargo build -p cintx-cubecl --features rocm,unstable-source-api 2>&1 | tail -8 && CINTX_ORACLE_BUILD_VENDOR=1 CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm cargo test -p cintx-oracle --features cpu,rocm,unstable-source-api --test breit_random_rocm_parity -- --ignored 2>&1 | tail -20</automated>
  </verify>
  <done>
    breit's per-quartet G-tensor build runs a `#[cube(launch)]` device kernel on all
    5 backend arms (Rocm→HipRuntime). In-crate device-vs-host (G-tensor) + f32 tests
    pass; `breit_random_rocm_parity.rs` reports mismatch_count=0, any_nonzero=true vs
    vendor (spinor, non-square block) on ROCm. gout_breit + nabla/x1 + cart_to_spinor
    documented as host-deferred. `launch_breit` signature + breit registration unchanged.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 5 [PARALLEL]: Port origk scalar (int3c1e_r2/r4/r6_origk_sph) to a #[cube(launch)] device kernel + ROCm oracle</name>
  <files>crates/cintx-cubecl/src/kernels/unstable/origk.rs, crates/cintx-oracle/tests/origk_random_rocm_parity.rs</files>
  <behavior>
    - Device-vs-host (`#[cfg(test)] #[cfg(feature = "cpu")]`): for 3-shell triples
      (s,s,s),(p,s,s),(s,p,s),(s,s,p),(p,p,s) on a fixture,
      `run_origk_scalar_device::<cubecl::cpu::CpuRuntime>` reproduces the host
      `contract_origk(fill_g_tensor_3c1e_origk(...))` Cartesian buffer for
      r_power ∈ {2,4,6} within atol=1e-12, rtol=1e-10.
    - Genericity: kernel launches for `F = f32` (s-s-s yields finite output).
  </behavior>
  <action>
SCOPE: port ONLY the SCALAR origk paths (int3c1e_r2/r4/r6_origk_sph, host
`contract_origk` + `fill_g_tensor_3c1e_origk`). DEFER the ip1 variants
(int3c1e_ip1_r2/r4/r6_origk_sph, host `contract_origk_ip1` + g1e_d_i_3c1e gradient)
to host — document at top of origk.rs (the ip1 bra-gradient ladder is a separable
derivative port, matching precedent).

origk is a 3-center 1e r^n-displaced operator (3 shells: shls:[i32;3]). In origk.rs:
write `#[cube(launch)] origk_scalar_kernel<F: Float + CubeElement>` running a single
work item, INLINING `fill_g_tensor_3c1e_origk` → `contract_origk`, accumulated over
primitive+contraction triples:
  - Inputs Array<F>: exps_i/j/k, coeff_i/j/k. Scratch mut Array<F>: g. Output mut
    Array<F>: cart_out (nctr_i*nctr_j*nctr_k*nfi*nfj*nfk, i fastest) — MATCH the
    host contract_origk layout exactly (read it to confirm index order).
  - Scalars F: ri/rj/rk coords (9), origin-k coords, common_factor. Runtime u32:
    li,lj,lk,nprim/nctr (×3), strides. `#[comptime] r_power: u32` (2/4/6) selecting
    the r^n contraction branch; `#[comptime] nroots: u32` rys_root{1..5}.
  Follow #[cube] rules.

Write `run_origk_scalar_device<R: Runtime>` (monomorphize over (r_power, nroots)
via macro_rules launch_with + runtime match) + `run_origk_scalar_on_backend` (5-arm,
Rocm→HipRuntime).

In `launch_origk`: keep FROZEN signature. For the scalar r2/r4/r6 variants add the
MAX_DEVICE_NROOTS guard then call `run_origk_scalar_on_backend(...)`; the existing
`cart_to_sph_3c1e` + scatter stays. Leave the ip1 host paths untouched (branch on
`origk_variant`).

In-crate device-vs-host + f32 tests per <behavior> in origk.rs.

Create `crates/cintx-oracle/tests/origk_random_rocm_parity.rs`: `#![cfg(feature =
"rocm")]`, LCG-jittered H2O/STO-3G slab (build_h2o_sto3g + jitter), `#[test]
#[ignore]` + `CINTX_ROCM_ORACLE=1`, drive via
`eval_raw(RawApiId::Symbol("int3c1e_r2_origk_sph"))` / `_r4_` / `_r6_` over a varied
3-shell-triple table, compare vs `#[cfg(has_vendor_libcint)]
vendor_ffi::vendor_int3c1e_r2_origk_sph` / `_r4_` / `_r6_` at atol=1e-12 rtol=0,
assert mismatch_count==0 AND any_nonzero==true over ~48 cases.
  </action>
  <verify>
    <automated>cd /home/user/Documents/workspace/cintx && cargo test -p cintx-cubecl --features cpu,unstable-source-api --lib kernels::unstable::origk 2>&1 | tail -20 && cargo build -p cintx-cubecl --features rocm,unstable-source-api 2>&1 | tail -8 && CINTX_ORACLE_BUILD_VENDOR=1 CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm cargo test -p cintx-oracle --features cpu,rocm,unstable-source-api --test origk_random_rocm_parity -- --ignored 2>&1 | tail -20</automated>
  </verify>
  <done>
    origk scalar r2/r4/r6 run a `#[cube(launch)]` device kernel on all 5 backend
    arms (Rocm→HipRuntime). In-crate device-vs-host + f32 tests pass;
    `origk_random_rocm_parity.rs` reports mismatch_count=0, any_nonzero=true vs
    vendor on ROCm. ip1 variants documented as host-deferred. `launch_origk`
    signature + origk registration unchanged.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 6 [PARALLEL]: Port ssc (int3c2e_sph_ssc) to a #[cube(launch)] device kernel + ROCm oracle</name>
  <files>crates/cintx-cubecl/src/kernels/unstable/ssc.rs, crates/cintx-oracle/tests/ssc_random_rocm_parity.rs</files>
  <behavior>
    - Device-vs-host (`#[cfg(test)] #[cfg(feature = "cpu")]`): for 3-shell triples
      (s,s,s),(p,s,s),(s,p,s),(s,s,p),(p,p,s),(d,s,s) on a fixture,
      `run_ssc_scalar_device::<cubecl::cpu::CpuRuntime>` reproduces the host
      `contract_3c2e_ssc(split_ij_hrr_ssc(fill_g_tensor_3c2e_ssc(...)))` Cartesian
      buffer within atol=1e-12, rtol=1e-10 (exercises the ij-HRR split branch).
    - Genericity: kernel launches for `F = f32` (s-s-s yields finite output).
  </behavior>
  <action>
SCOPE: port the FULL ssc scalar path (int3c2e_sph_ssc is the only ssc symbol —
no derivative variants to defer). ssc is a 3-center 2e integral (shls:[i32;3]) with
a Rys-root 2e-style G-tensor + an ij-HRR split (`split_ij_hrr_ssc`) + a
3c2e contraction (`contract_3c2e_ssc`).

In ssc.rs: write `#[cube(launch)] ssc_scalar_kernel<F: Float + CubeElement>`
running a single work item, INLINING `fill_g_tensor_3c2e_ssc` →
`split_ij_hrr_ssc` → `contract_3c2e_ssc`, accumulated over primitive+contraction
triples. Model the 2e Rys VRR + the ij-HRR transfer on two_electron.rs /
center_3c2e.rs device kernels; inline split_ij_hrr_ssc with runtime u32 strides
passed from host. Output mut Array<F>: cart_out matching the host contract layout
(read contract_3c2e_ssc to confirm i/j/k index order). `#[comptime] nroots: u32`
rys_root{1..5}. Follow all #[cube] rules.

NOTE on the host transpose+c2s: the existing `transpose_ij_3idx` +
`cart_to_sph_3c2e_ssc` representation transform stays ON HOST (host part of the
split) — the device kernel produces the Cartesian block in the layout the existing
host transpose/c2s consumes. Document the split at the top of ssc.rs.

Write `run_ssc_scalar_device<R: Runtime>` + `run_ssc_scalar_on_backend` (5-arm,
Rocm→HipRuntime).

In `launch_ssc`: keep FROZEN signature. Add the MAX_DEVICE_NROOTS guard, replace
the host `fill_g_tensor_3c2e_ssc`+`split_ij_hrr_ssc`+`contract_3c2e_ssc`
accumulation with `run_ssc_scalar_on_backend(...)`; the existing
`transpose_ij_3idx` + `cart_to_sph_3c2e_ssc` + scatter stay UNCHANGED.

In-crate device-vs-host + f32 tests per <behavior> in ssc.rs.

Create `crates/cintx-oracle/tests/ssc_random_rocm_parity.rs`: `#![cfg(feature =
"rocm")]`, LCG-jittered H2O/STO-3G slab (build_h2o_sto3g + jitter), `#[test]
#[ignore]` + `CINTX_ROCM_ORACLE=1`, drive via
`eval_raw(RawApiId::Symbol("int3c2e_sph_ssc"))` over a varied 3-shell-triple table,
compare vs `#[cfg(has_vendor_libcint)] vendor_ffi::vendor_int3c2e_sph_ssc` at
atol=1e-12 rtol=0, assert mismatch_count==0 AND any_nonzero==true over ~48 cases.
  </action>
  <verify>
    <automated>cd /home/user/Documents/workspace/cintx && cargo test -p cintx-cubecl --features cpu,unstable-source-api --lib kernels::unstable::ssc 2>&1 | tail -20 && cargo build -p cintx-cubecl --features rocm,unstable-source-api 2>&1 | tail -8 && CINTX_ORACLE_BUILD_VENDOR=1 CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm cargo test -p cintx-oracle --features cpu,rocm,unstable-source-api --test ssc_random_rocm_parity -- --ignored 2>&1 | tail -20</automated>
  </verify>
  <done>
    ssc (int3c2e_sph_ssc) runs a `#[cube(launch)]` device kernel on all 5 backend
    arms (Rocm→HipRuntime). In-crate device-vs-host + f32 tests pass;
    `ssc_random_rocm_parity.rs` reports mismatch_count=0, any_nonzero=true vs
    vendor on ROCm. host transpose+c2s documented as the host split. `launch_ssc`
    signature + ssc registration unchanged.
  </done>
</task>

</tasks>

<verification>
Full-stack checks:

1. Refactor is behavior-preserving (Task 1):
   - `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu,unstable-source-api --test unstable_source_parity`
     — all 5 families' CPU vendor-parity tests pass UNCHANGED.

2. Each family's f64 device path is byte-correct vs vendor on CPU + ROCm:
   - `cargo test -p cintx-cubecl --features cpu,unstable-source-api --lib kernels::unstable`
     (all in-file device-vs-host + f32 genericity tests pass)
   - `CINTX_ORACLE_BUILD_VENDOR=1 CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm cargo test -p cintx-oracle --features cpu,rocm,unstable-source-api --test origi_random_rocm_parity --test grids_random_rocm_parity --test breit_random_rocm_parity --test origk_random_rocm_parity --test ssc_random_rocm_parity -- --ignored`
     → each reports mismatch_count=0, any_nonzero=true.

3. ROCm arm compiles for all families:
   - `cargo build -p cintx-cubecl --features rocm,unstable-source-api` clean.

4. Frozen-surface invariants:
   - `grep -c "unstable::launch_" crates/cintx-cubecl/src/kernels/mod.rs` returns 5
     (registration intact, names resolve via the unstable/ facade).
   - `grep -rn "pub fn launch_origi\|pub fn launch_grids\|pub fn launch_breit\|pub fn launch_origk\|pub fn launch_ssc" crates/cintx-cubecl/src/kernels/unstable/`
     — all 5 signatures unchanged ((&ResolvedBackend, &ExecutionPlan, &SpecializationKey, &mut [f64]) -> Result<ExecutionStats, cintxRsError>).
   - No new capi enum variants, no legacy `cint*` wrappers (project convention).
</verification>

<success_criteria>
- `unstable.rs` is split into a behavior-preserving `unstable/` module directory;
  the unchanged `unstable_source_parity.rs` CPU vendor tests pass after the split.
- Each of origi/grids/breit/origk/ssc runs a real `#[cube(launch)]` device kernel
  generic over `F: Float` for its scalar path, dispatched on all 5 backend arms
  including the ROCm/HIP arm.
- Each family has a new `<family>_random_rocm_parity.rs` showing 0 divergence vs
  libcint 6.1.3 on ROCm (mismatch_count=0, any_nonzero=true).
- The 5 `launch_*` FamilyLaunchFn signatures and their canonical_family
  registrations are unchanged; no new capi enum variants, no legacy `cint*` wrappers.
- The host/device split is honest + documented per family: G-tensor VRR/HRR +
  Cartesian contraction on-device; c2s/spinor transforms + AO scatter on-host;
  each family's deferred sub-paths (origi ip2, grids ip/ipip/ipvip/spvsp, breit
  gout+spinor, origk ip1) documented in-code with rationale.
</success_criteria>

<parallelization>
- Task 1 is the SERIAL GATE (touches/deletes the shared file). It MUST land first.
- Tasks 2–6 are INDEPENDENT after Task 1: each touches only its own
  `unstable/<family>.rs` + its own `tests/<family>_random_rocm_parity.rs`
  (zero `files_modified` overlap). They are PARALLELIZABLE across worktree-isolated
  executors and each is one atomic commit. Re-verify with
  `git merge-base --is-ancestor` after each wave and merge manually if the
  background auto-integration does not pick them up.
</parallelization>

<output>
After completion, create `.planning/quick/260529-twi-refactor-unstable-rs-kernels-to-generic-/260529-twi-SUMMARY.md`
documenting: the module split layout (shared.rs + 5 family files), per-family
host/device split + which sub-paths were deferred-to-host with rationale, each
family's kernel arg layout (host-computed strides as runtime u32, comptime
nroots/r_power), the 5 ROCm parity results (mismatch_count, case count, tolerance),
and any #[cube] authoring pitfalls hit during the ports.
</output>
