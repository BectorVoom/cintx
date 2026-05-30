---
phase: quick-260529-gbf
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/cintx-cubecl/src/kernels/ecp.rs
  - crates/cintx-oracle/tests/ecp_random_rocm_parity.rs
  - xtask/src/rocm_oracle.rs
autonomous: true
requirements: []
must_haves:
  truths:
    - "launch_ecp dispatches a real #[cube(launch)] device kernel on the resolved backend (Cpu/Rocm/...) for the angular-splice compute, not a pure-host loop"
    - "f64 byte-identity of the existing ecp scalar + ipnuc + iprinv operators is preserved (all existing ecp.rs unit tests and safe_api_ecp_parity / ecp_iprinv_parity remain green)"
    - "the device kernel is generic over F: Float, computes internally in f64, mirroring center_2c2e/center_4c1e"
    - "a randomized ECP idempotency oracle runs on the ROCm GPU and reports mismatch_count==0 with non-zero output"
  artifacts:
    - path: "crates/cintx-cubecl/src/kernels/ecp.rs"
      provides: "ecp_angular_kernel #[cube(launch)] + run_ecp_angular_device::<R> + per-backend match dispatch"
      contains: "#[cube(launch)]"
    - path: "crates/cintx-oracle/tests/ecp_random_rocm_parity.rs"
      provides: "randomized ROCm ECP idempotency oracle (LCG, CINTX_ROCM_ORACLE gate, mismatch_count==0)"
      contains: "test_ecp_sph_random_rocm_idempotency"
  key_links:
    - from: "crates/cintx-cubecl/src/kernels/ecp.rs launch_ecp"
      to: "run_ecp_angular_device::<R>"
      via: "ResolvedBackend match (Cpu/Rocm/Wgpu/Cuda/Metal)"
      pattern: "run_ecp_angular_device::<"
    - from: "crates/cintx-oracle/tests/ecp_random_rocm_parity.rs"
      to: "OperatorId::INT1E_ECP_SPH via SessionRequest on rocm backend"
      via: "ExecutionOptions forcing rocm + double evaluate"
      pattern: "INT1E_ECP_SPH"
---

<objective>
Refactor `crates/cintx-cubecl/src/kernels/ecp.rs` from a pure-Rust HOST port into
a CubeCL `#[cube(launch)]` device kernel generic over `F: Float`, following the
EXACT structural template of `center_4c1e.rs` / `center_2c2e.rs`, then prove it on
the ROCm GPU with a randomized idempotency oracle reporting `mismatch_count==0`.

Purpose: CLAUDE.md mandates CubeCL as the primary compute backend with host CPU
work limited to planning/validation/marshaling. ECP is the largest remaining
host-only family (documented Phase 19 D-16 deviation). This lands the GPU half.

Output:
- `ecp_angular_kernel` (`#[cube(launch)]`, generic `F: Float`) + `run_ecp_angular_device::<R: Runtime>` + per-backend `match` dispatch in `ecp.rs`.
- `ecp_random_rocm_parity.rs` randomized ROCm idempotency oracle.
- An actual ROCm run on the gfx1152 GPU confirming `mismatch_count==0` + non-zero output.
</objective>

<critical_architecture>
## The hard part, confronted head-on (do NOT hand-wave)

ECP's `ecp_type1_cart` / `ecp_type2_cart` (ecp.rs:633, :826) have TWO phases with
very different `#[cube]` compatibility:

**Phase A — adaptive radial machinery (STAYS HOST-SIDE, as planning/marshaling).**
The level-adaptive Gauss-Chebyshev convergence loop (ecp.rs:686-746) has:
  - data-dependent termination (`break` on `close_enough` convergence, ecp.rs:734),
  - a dynamic `nrs` row count from `ecprad_part_host` (variable loop bound),
  - table-interpolation modified-spherical-Bessel evaluation (`ecpsph_ine_opt_host`
    in `math/ecp_k_taylor.rs`, three-branch with `loop {}` at line 156 + downward
    recurrence), and a Taylor expansion.
These violate `#[cube]` constraints (no `break`/`continue`, no recursion, static
loop bounds, no data-dependent control flow per the Phase 8 P02 `cond_br` MLIR
limit noted in ecp.rs module doc lines 37-42). They are EXACTLY the "planning,
validation, marshaling" CLAUDE.md keeps host-side. They produce, per shell pair,
the precomputed radial tensors `rad_ang_all` (Type-1, size `nci*ncj*d3`,
ecp.rs:749) / the Type-2 radial+angular factors, plus the static angular factor
tables `ifac`/`jfac` (`type1_static_facs`, ecp.rs:776-779).

**Phase B — angular splice / Cartesian accumulation (MOVES TO `#[cube]` DEVICE).**
The final nested loop (ecp.rs:783-817) is a deterministic, statically-bounded
(per (li,lj)) triple-product accumulation:
  `acc += pifac[..] * pjfac[..] * prad[..]` summed over `i1..=ix, ... j3..=jz`,
  written to `gctr[pout_idx]`.
This is the same "consume precomputed tensors, do bounded arithmetic, write the
cart buffer" shape center_4c1e's `#[cube]` kernel already does. It ports cleanly:
comptime `li`/`lj`, fixed `cart_comps` expansion, `while` loops with u32 bounds,
`F` arithmetic, NO special functions, NO break/continue. This is the device kernel.

**This is the honest, CLAUDE.md-compliant split:** the special-function/adaptive
control flow stays host (marshaling), the heavy bounded arithmetic splice runs on
the GPU device generic over `F`. The radial host helpers are NOT made `#[cube]`
(they cannot be) — they feed flat f64 buffers into the device kernel.

The same split applies to Type-2 (`ecp_type2_cart`) and the gradient drivers
(`compute_type1_pair_grad` / `compute_type2_pair_grad` reuse the same scalar splice
on l±1 fake shells, ecp.rs module doc lines 62-91). Port the shared angular-splice
inner; the gradient drivers keep calling it (now device-backed).
</critical_architecture>

<execution_context>
@/home/user/Documents/workspace/cintx/.claude/get-shit-done/workflows/execute-plan.md
@/home/user/Documents/workspace/cintx/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md

# AUTHORITATIVE #[cube] authoring rules — read before writing any kernel body.
# Top pitfalls: no plain-fn calls from cube code, no if-EXPRESSIONS (mutate-in-branch),
# F::exp/F::sqrt/F::recip only, u32/i32 indices only, no continue/break, comptime for shape.
@docs/manual/Cubecl/Cubecl_basic_operations.md
@docs/manual/Cubecl/Cubecl_conditionals.md
@docs/manual/Cubecl/Cubecl_generics.md
@docs/manual/Cubecl/cubecl_macro_fanout_manual.md

# CANONICAL TEMPLATE (most recent port). Follow its structure EXACTLY:
#   - #[cube(launch)] kernel generic over F: Float + CubeElement (center_4c1e.rs:110)
#   - run_4c1e_device::<R: Runtime>(...) -> Vec<f64> dispatch (center_4c1e.rs:773)
#   - launch_center_4c1e_typed::<F: CintFloat> generic inner (center_4c1e.rs:924)
#   - per-backend match { Cpu => ::<CpuRuntime>, Rocm => ::<cubecl_hip::HipRuntime>, ... } (center_4c1e.rs:1016-1058)
@crates/cintx-cubecl/src/kernels/center_4c1e.rs

# FIRST GPU family — simplest reference for the f64-internal typed split.
#   - launch_center_2c2e (outer F64/F32 dispatcher, bytemuck f32 view) center_2c2e.rs:754
#   - launch_center_2c2e_typed::<F: CintFloat> center_2c2e.rs:608
@crates/cintx-cubecl/src/kernels/center_2c2e.rs

# TARGET of the refactor. KEY structure already in place:
#   - launch_ecp(backend, plan, spec, staging: &mut [f64]) ecp.rs:1400 (NOTE: f64 staging, not generic)
#   - ecp_type1_cart ecp.rs:633 / ecp_type2_cart ecp.rs:826 (host drivers; Phase B splice is the inner loop)
#   - deriv1_cart_pair / compute_type{1,2}_pair_grad (gradient reuse the same splice)
#   - existing tests ecp.rs:1698-1977 (registration + helper + selector — MUST still pass)
@crates/cintx-cubecl/src/kernels/ecp.rs

# Host radial machinery that STAYS host (special functions, cannot be #[cube]).
@crates/cintx-cubecl/src/math/ecp_k_taylor.rs

# ORACLE TEMPLATE to mirror for the random ROCm test:
#   - Lcg deterministic RNG center_4c1e_parity.rs:195
#   - build_random_* center_4c1e_parity.rs:231
#   - test_int4c1e_sph_random_rocm_idempotency: cfg(rocm) + #[ignore] + CINTX_ROCM_ORACLE=1
#     guard + double-eval + mismatch_count==0 + any_nonzero center_4c1e_parity.rs:333
@crates/cintx-oracle/tests/center_4c1e_parity.rs

# Existing ECP oracle tests — module gate #![cfg(any(feature="cpu", feature="rocm"))],
# ECP has NO with-ecp feature. ECP tests drive the SAFE API (SessionRequest + BasisSet +
# build_cu_lanl2dz_safe_basis), NOT raw eval_raw. Mirror this drive path, not center_4c1e's raw path.
@crates/cintx-oracle/tests/safe_api_ecp_parity.rs
@crates/cintx-oracle/tests/ecp_iprinv_parity.rs

# How xtask rocm-oracle drives the suite: base profile = just `rocm`; ECP has no
# with-* feature, so the test runs under the BASE profile (no profile flag needed).
@xtask/src/rocm_oracle.rs
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Port the ECP angular splice into a generic-F #[cube(launch)] device kernel + run_ecp_angular_device</name>
  <files>crates/cintx-cubecl/src/kernels/ecp.rs</files>
  <behavior>
    - The device kernel reproduces the host Phase-B splice (ecp.rs:783-817) byte-for-byte at f64:
      given precomputed `rad_ang_all` (per (ic,jc): len d3), `ifac` (len nfi*di3), `jfac` (len nfj*dj3),
      comptime li/lj and nci/ncj, it accumulates
      `cart_out[(jc*nfj+mj)*(nci*nfi)+ic*nfi+mi] += sum_{i1..ix,..,j3..jz} ifac*jfac*prad`.
    - A f64 device-vs-host equivalence unit test (gated #[cfg(feature="cpu")]) over several (li,lj) up to (2,2)
      with random ifac/jfac/rad_ang inputs asserts max-abs-diff == 0.0 (byte-identity, identical f64 op order).
    - Generic-F: running the kernel at F=f32 on CpuRuntime reproduces the f64 result cast to f32 within f32 eps.
  </behavior>
  <action>
    Add to ecp.rs, mirroring center_4c1e.rs:110/773 EXACTLY:

    (1) `#[cube(launch)] fn ecp_angular_kernel<F: Float + CubeElement>(...)`. Inputs (all `&Array<F>`):
    `rad_ang_all` (nci*ncj*d3), `ifac` (nfi*di3), `jfac` (nfj*dj3), `comps_i` and `comps_j` as `&Array<u32>`
    (flattened cart-component exponent triples, precomputed host-side — this avoids re-deriving cart_comps under
    #[cube], which is marshaling), `&mut Array<F> cart_out` (nci*nfi*ncj*nfj). Scalars as `u32`: li, lj, nci, ncj.
    Compute nfi/nfj/d1/d2/d3/di1..di3/dj1..dj3 in-kernel from li/lj (formulas at ecp.rs:662-671 + ncart). Guard
    `if UNIT_POS == 0u32 { ... }` (center_4c1e.rs:150). Zero `cart_out` first (while-loop, center_4c1e.rs:206-210).
    Then the i1..=ix … j3..=jz sextuple `while` loops accumulating `acc` in `F`, write `cart_out[pout_idx]`.
    OBEY docs/manual/Cubecl: no plain-fn calls, no if-EXPRESSIONS (`let mut x=..; if cond { x=.. }`), u32 indices,
    no break/continue (the splice is fully bounded — that is why it ports). Match the host nested loop order
    EXACTLY (i1 outer .. j3 inner, ecp.rs:795-811) to preserve f64 summation order / byte-identity.

    (2) `fn run_ecp_angular_device<R: Runtime>(client, li:u32, lj:u32, nci:u32, ncj:u32, rad_ang_all:&[f64],
    ifac:&[f64], jfac:&[f64], comps_i:&[u32], comps_j:&[u32]) -> Vec<f64>` — mirror center_4c1e.rs:773:
    create_from_slice for each input, zero out_h, `ecp_angular_kernel::launch::<f64, R>(...)` with
    CubeCount::Static(1,1,1) / CubeDim::new_1d(1), read_one_unchecked, return Vec<f64>.

    (3) Add the device-vs-host f64 equivalence + generic-f32 unit tests described in &lt;behavior&gt; (gated
    #[cfg(feature="cpu")], run on CpuRuntime). Keep the host splice helper available for the reference path.

    Do NOT yet rewire ecp_type1_cart/launch_ecp — that is Task 2. This task lands the kernel + dispatch fn +
    its standalone equivalence proof. Do NOT add capi variants or cint* wrappers (Memory "New family surface scope").
  </action>
  <verify>
    <automated>cargo test --locked -p cintx-cubecl --features cpu ecp_angular 2>&1 | tail -30</automated>
  </verify>
  <done>ecp.rs has `#[cube(launch)] ecp_angular_kernel<F: Float + CubeElement>` + `run_ecp_angular_device::<R>`. The device-vs-host f64 equivalence unit test passes with max-abs-diff 0.0; the generic-f32 test passes within f32 eps. Commit: `feat(quick-260529-gbf): ecp angular splice as generic-F #[cube(launch)] device kernel + run_ecp_angular_device::<R>`.</done>
</task>

<task type="auto">
  <name>Task 2: Rewire ecp_type1/type2 + launch_ecp through run_ecp_angular_device on the resolved backend, preserve f64 byte-identity</name>
  <files>crates/cintx-cubecl/src/kernels/ecp.rs</files>
  <action>
    Replace the host Phase-B inner loop in `ecp_type1_cart` (ecp.rs:783-817) and the equivalent splice in
    `ecp_type2_cart` with a `match backend { ... }` dispatch IDENTICAL in shape to center_4c1e.rs:1016-1058
    (Cpu=>CpuRuntime, Rocm=>cubecl_hip::HipRuntime, Wgpu, Cuda, Metal — each #[cfg]-gated) calling
    `run_ecp_angular_device::<R>`. Thread `backend: &ResolvedBackend` down into ecp_type1_cart / ecp_type2_cart
    AND the gradient drivers (compute_type1_pair_grad / compute_type2_pair_grad / deriv1_cart_pair) which reuse
    the same splice on l±1 fake shells. Precompute `comps_i`/`comps_j` (flattened u32 triples from cart_comps)
    and the f64 `ifac`/`jfac`/`rad_ang_all` host-side (the adaptive radial loop ecp.rs:686-779 + static-facs STAY
    host) and pass them in.

    In `launch_ecp` (ecp.rs:1400): remove `let _ = backend;` (ecp.rs:1416) and pass `backend` into every
    ecp_type1_cart/ecp_type2_cart/compute_type{1,2}_pair_grad call site (ecp.rs:1576-1609). KEEP the f64 staging
    signature and the registry entry (kernels/mod.rs:40) UNCHANGED — ECP's registered FamilyLaunchFn is f64-staging
    and the byte-identity gate is f64/CPU-vs-C; do NOT add an F32 outer dispatcher.

    Because run_ecp_angular_device launches at f64 with the SAME nested summation order, the result MUST be
    byte-identical to the prior host splice. If any FP reassociation appears under CpuRuntime, fix the in-kernel
    loop order to match ecp.rs:795-811 exactly so atol=1e-12/rtol=0.0 vendor parity holds. `launch_ecp` must stay
    registered under canonical_family "ecp" and keep routing ecp/ecp_ipnuc/ecp_iprinv (ecp.rs:1707-1977 assert this).

    Update the module-doc "host-only this phase" paragraph (ecp.rs:33-42) to state the angular splice now runs
    on-device generic over F while the adaptive radial machinery remains host marshaling (cite this plan).
  </action>
  <verify>
    <automated>cargo test --locked -p cintx-cubecl --features cpu ecp 2>&1 | tail -20</automated>
  </verify>
  <done>launch_ecp routes the angular splice through run_ecp_angular_device on the resolved backend (no `let _ = backend`). All cintx-cubecl ECP unit tests pass. Vendor CPU parity holds: `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --locked -p cintx-oracle --features cpu --test safe_api_ecp_parity --test ecp_iprinv_parity` is green (byte-identity atol=1e-12/rtol=0.0 preserved). Commit: `refactor(quick-260529-gbf): launch_ecp dispatches ecp angular splice on the resolved backend (rocm=HipRuntime); f64 byte-identity preserved`.</done>
</task>

<task type="auto">
  <name>Task 3: Add the randomized ROCm ECP idempotency oracle and RUN it on the GPU (mismatch_count==0)</name>
  <files>crates/cintx-oracle/tests/ecp_random_rocm_parity.rs, xtask/src/rocm_oracle.rs</files>
  <action>
    Create `crates/cintx-oracle/tests/ecp_random_rocm_parity.rs` mirroring center_4c1e_parity.rs:195-400 but
    using the SAFE API ECP drive path (safe_api_ecp_parity.rs), NOT raw eval_raw:

    - Module gate `#![cfg(feature = "rocm")]` (ECP module gate is cpu|rocm; for this rocm-only oracle gate the
      whole file on rocm). Bring in the `Lcg` deterministic RNG (copy from center_4c1e_parity.rs:195-222).
    - A `build_random_ecp_system(rng)` that takes the Cu/LANL2DZ ECP fixture (`build_cu_lanl2dz` for the ECP slab
      + `build_cu_lanl2dz_safe_basis(Representation::Spheric)` for the typed BasisSet, per safe_api_ecp_parity.rs:351,
      372) and RANDOMIZES the AO shell exponents/coefficients and/or atom coordinates within physically sane ranges
      (exps in [0.25,4.0], coeffs in [0.15,1.0], coords jittered) while keeping the EcpShell slab intact, so each
      case is a distinct valid ECP system. (If mutating the typed BasisSet in place is awkward, build a fresh
      randomized BasisSet+shells per case following build_cu_lanl2dz_safe_basis's construction.)
    - `#[test] #[ignore] fn test_ecp_sph_random_rocm_idempotency()`:
        * assert_eq!(std::env::var("CINTX_ROCM_ORACLE").as_deref(), Ok("1"), "...invoke via xtask rocm-oracle...")
          to block direct `cargo test --features rocm -- --ignored` (mirror center_4c1e_parity.rs:333-339).
        * For ~32-64 random cases: drive `OperatorId::INT1E_ECP_SPH` via `SessionRequest::new(op, Spheric, basis,
          shell_tuple, opts)` TWICE with `ExecutionOptions` selecting the ROCm backend (match how the safe-API
          collector in safe_api_ecp_parity.rs:215/424 builds requests; set the backend to rocm — check
          ExecutionOptions for the backend selector field; CINTX_BACKEND=rocm is also exported by xtask). Collect
          out1, out2.
        * mismatch_count += count of |out1[k]-out2[k]| > atol + rtol*|out2[k]| (atol=1e-12, rtol=1e-10).
        * any_nonzero |= out1 has an entry with abs > 1e-12 (proves the device kernel actually ran, not all-zeros).
        * assert_eq!(mismatch_count, 0, ...) AND assert!(any_nonzero, "ECP device output all zeros — kernel did not run").
        * println! the case count + mismatch_count so the run is auditable.
    - In `xtask/src/rocm_oracle.rs`: ECP runs under the BASE profile (rocm only, no with-* feature) so the new
      test file is picked up automatically by the existing `cargo test -p cintx-oracle --features rocm -- --ignored`
      invocation. Confirm no profile/feature change is needed; if the test file needs explicit inclusion adjust
      the doc comment (lines 20-22) to note ECP is now covered by the base rocm suite.

    Then ACTUALLY RUN IT on the gfx1152 GPU and capture mismatch_count:
      `cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle`
  </action>
  <verify>
    <automated>cargo build --locked -p cintx-oracle --features rocm --tests 2>&1 | tail -15 && cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle 2>&1 | tail -40</automated>
  </verify>
  <done>`ecp_random_rocm_parity.rs` exists with `test_ecp_sph_random_rocm_idempotency` (cfg(rocm) + #[ignore] + CINTX_ROCM_ORACLE=1 guard). The test compiles under `--features rocm`. The actual ROCm run via `xtask rocm-oracle` PASSES with mismatch_count==0 across all random ECP cases AND any_nonzero==true (proving the device kernel ran on the AMD GPU). The executor reports the observed case count + mismatch_count in the SUMMARY. Commit: `test(quick-260529-gbf): randomized ROCm int1e_ecp_sph idempotency oracle (mismatch_count=0, device kernel proven on gfx1152)`.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| host → device kernel | Precomputed f64 radial/angular buffers + u32 component tables cross to the #[cube] kernel; sizes/comptime shapes must match the host-computed lengths or the GPU reads out of bounds. |
| oracle test → GPU backend | Randomized ECP geometries drive the device kernel; malformed/degenerate geometry must not panic or silently zero. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-gbf-01 | Tampering | ecp_angular_kernel buffer sizing | mitigate | run_ecp_angular_device computes all buffer lengths from li/lj/nci/ncj using the SAME formulas as the host driver; ArrayArg lengths assert against those; device-vs-host f64 equivalence test (Task 1) catches any index/size drift. |
| T-gbf-02 | Information disclosure | f64→f32 reinterpret / generic-F output | accept | ECP keeps f64 staging; no f32 bytemuck reinterpret added (unlike 2c2e). Internal f64 launch only; f32 generic path is unit-tested but ECP's registered surface stays f64. Low risk. |
| T-gbf-03 | Denial of service | non-converging adaptive radial loop on degenerate random geometry | mitigate | adaptive loop already has `start.saturating_sub(1)/2` underflow guard (ecp.rs:743) and exits at level>LEVEL_MAX; random oracle uses physically sane exponent/coordinate ranges (no coincident centers) mirroring center_4c1e_parity build_random. |
| T-gbf-04 | Repudiation | "device kernel ran" claim unverifiable | mitigate | oracle asserts any_nonzero==true AND runs under CINTX_BACKEND=rocm; a wrong/host backend or all-zero output fails the test. mismatch_count printed for audit. |
</threat_model>

<verification>
- Phase B angular splice runs as a `#[cube(launch)]` device kernel generic over `F: Float`, dispatched per resolved backend (Cpu/Rocm/...) — grep `run_ecp_angular_device::<` in ecp.rs returns the match arms.
- Phase A adaptive radial machinery + special functions remain host-side (no #[cube] added to ecp_k_taylor.rs / bessel.rs).
- f64 byte-identity preserved: cintx-cubecl ECP unit tests + vendor CPU parity (safe_api_ecp_parity, ecp_iprinv_parity) green at atol=1e-12/rtol=0.0.
- ECP still registered under canonical_family "ecp", routing ecp + ecp_ipnuc + ecp_iprinv (existing tests pass).
- Randomized ROCm idempotency oracle PASSES on the gfx1152 GPU: mismatch_count==0 AND non-zero output.
</verification>

<success_criteria>
- `cargo test --locked -p cintx-cubecl --features cpu ecp` passes (all existing + new unit tests).
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --locked -p cintx-oracle --features cpu --test safe_api_ecp_parity --test ecp_iprinv_parity` passes (byte-identity preserved).
- `cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle` passes with the new ECP test reporting mismatch_count==0 and non-zero output.
- No new capi enum variants, no legacy cint* wrappers (kernel + oracle only).
</success_criteria>

<output>
After completion, create `.planning/quick/260529-gbf-refactor-ecp-rs-to-cubecl-kernel-with-ge/260529-gbf-SUMMARY.md` reporting the observed ROCm mismatch_count and case count.
</output>
