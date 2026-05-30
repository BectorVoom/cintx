---
quick_id: 260529-exs
slug: center-3c2e-cubecl-kernel
date: 2026-05-29
status: Ready for planning
---

# Quick Task 260529-exs: `center_3c2e` CubeCL device kernel + ROCm random oracle — Context

**Gathered:** 2026-05-29
**Status:** Ready for planning

<domain>
## Task Boundary

Refactor `crates/cintx-cubecl/src/kernels/center_3c2e.rs` from host-only f64
pipelines into real CubeCL `#[cube(launch)]` device kernels generic over
`F: Float`, dispatched onto the resolved backend's `ComputeClient`, then add a
random ROCm oracle idempotency test for the registered `int3c2e_ip1_sph` API and
run it on the rocm backend.
</domain>

<decisions>
## Implementation Decisions (LOCKED — do not revisit)

### Port scope — BOTH paths
`center_3c2e.rs` has two distinct host numeric paths:
1. **Scalar 3c2e path** — `fill_g_tensor_3c2e` → `split_ij_hrr` → `contract_3c2e`
   (Rys-based; direct analog of the 2c2e/3c1e scalar ports). Reachable but NOT
   wired to any registered oracle API.
2. **IP1 gradient path** — `launch_center_3c2e_ip1` (the ∇_A first-center
   derivative: `build_2e_shape(li+1,..)` → `fill_g_tensor_2e` → `f12::gout_ip1`,
   3 output components, component-leading `[3,nk,nj,ni]`). This is the ONLY
   registered API (`RawApiId::INT3C2E_IP1_SPH`, operator `"ip1"`) and is what the
   rocm oracle test actually exercises.

**User decision (AskUserQuestion, 2026-05-29): port BOTH paths to `#[cube(launch)]`
device kernels.** Rationale: the scalar port is the clean sibling-analog, AND the
IP1 path must become a device kernel so the rocm random oracle test
(`int3c2e_ip1_sph`) genuinely drives a `#[cube]` kernel on the GPU — the faithful
analog of what the 3c1e task achieved.

### Rys template — follow 2c2e, NOT 3c1e
Both 3c2e paths use Rys quadrature (`rys_roots_host`, `nrys_roots ≤ 5`). So the
device-kernel template is **`center_2c2e.rs`** (the first GPU family, which has the
`#[comptime] nroots` + `rys_root1..5` device-function selection), NOT the
rys-free `center_3c1e.rs`. The 3c1e PLAN
(`.planning/quick/260529-e69-.../260529-e69-PLAN.md`) is the structural template
for the workflow shape (run_*_device dispatcher, device-vs-host cross-checks,
random rocm oracle test), but the Rys machinery comes from 2c2e.

### Oracle test — int3c2e_ip1_sph
The random rocm idempotency test must use `RawApiId::INT3C2E_IP1_SPH` (mirroring
the existing `test_int3c2e_sph_h2o_sto3g_rocm_parity` in
`crates/cintx-oracle/tests/center_3c2e_parity.rs`). There is no plain
`int3c2e_sph` registered. Model the random generator on
`test_int2c2e_sph_random_rocm_idempotency` (center_2c2e_parity.rs) /
`test_int3c1e_sph_random_rocm_idempotency` (center_3c1e_parity.rs).

### Public signatures unchanged
`launch_center_3c2e` (outer precision dispatcher, registered `FamilyLaunchFn`) and
`launch_center_3c2e_typed` signatures stay byte-for-byte unchanged. The `"ip1"`
operator branch (line 616-620) keeps delegating to the IP1 launcher; only the
IP1 launcher's numeric core moves to the device. Keep the precision policy:
device compute at f64, output cast to `F` via `F::from_f64_lossy`; preserve the
precision-aware nonzero sentinel and `ExecutionStats` blocks verbatim.
</decisions>

<specifics>
## Specific Ideas / Key references

- **Template (Rys device kernel):** `crates/cintx-cubecl/src/kernels/center_2c2e.rs`
  — `center_2c2e_kernel` (`#[cube(launch)]`, `#[comptime] nroots`, `comptime!`
  branch over `rys_root1..5`), `run_2c2e_device<R: Runtime>`, the
  `match backend { Cpu|Wgpu|Cuda|Rocm|Metal }` `#[cfg(feature=...)]`-gated arms.
- **Structural template (workflow shape):** the 3c1e PLAN/SUMMARY at
  `.planning/quick/260529-e69-refactor-center-3c1e-rs-to-cubecl-kernel/`.
- **Scalar path internals (port target #1):** `fill_g_tensor_3c2e` (line 70),
  `split_ij_hrr` (line 195), `contract_3c2e` (line 254), driven from
  `launch_center_3c2e_typed` (lines 622-695).
- **IP1 path internals (port target #2):** `launch_center_3c2e_ip1` (line 339),
  depends on `two_electron::{build_2e_shape, fill_g_tensor_2e, two_e_shape_as_f12}`
  and `crate::kernels::f12::gout_ip1` — all must be inlined into the `#[cube]`
  kernel (no plain-Rust fn calls inside `#[cube]`). Output is component-leading
  `[3, nk, nj, ni]` (transpose of interleaved `gout[n*3+comp]`).
- **Oracle siblings:** `center_2c2e_parity.rs` (`Lcg`, `build_random_2shell`,
  `test_int2c2e_sph_random_rocm_idempotency`) and `center_3c1e_parity.rs`
  (`build_random_3shell`, `test_int3c1e_sph_random_rocm_idempotency`).

## CubeCL `#[cube]` authoring rules (MANDATORY — `docs/manual/Cubecl/*`)
Same checklist the 3c1e/2c2e ports obey: no plain-fn calls (inline everything),
no `if`-as-expression (statement form), math via `F::exp`/`F::sqrt`/`F::cast_from`/
`F::new` (no methods), `u32`/`i32` indices only (no `usize`/`u64`/`Vec`/host
`for`), no `continue`/`break` in `while` loops, single `#[cube(launch)]` entry per
kernel guarded by `if UNIT_POS == 0u32`, `#[comptime] nroots` for the Rys
branch, host buffers via `client.create_from_slice(f64::as_bytes(..))`, read back
via `client.read_one_unchecked`, `CubeCount::Static(1,1,1)` + `CubeDim::new_1d(1)`.
</specifics>

<canonical_refs>
## Canonical References

- `crates/cintx-cubecl/src/kernels/center_2c2e.rs` — Rys device-kernel template.
- `crates/cintx-cubecl/src/kernels/center_3c1e.rs` — sibling port (rys-free).
- `crates/cintx-cubecl/src/kernels/two_electron.rs` — `build_2e_shape`,
  `fill_g_tensor_2e`, `two_e_shape_as_f12` (IP1 path G-tensor source).
- `crates/cintx-cubecl/src/kernels/f12.rs` — `gout_ip1` (∇_i gradient walk).
- `crates/cintx-oracle/tests/center_3c2e_parity.rs` — existing rocm parity test
  (`test_int3c2e_sph_h2o_sto3g_rocm_parity`, uses `INT3C2E_IP1_SPH`).
- `docs/manual/Cubecl/*.md` — authoritative `#[cube]` authoring rules.
</canonical_refs>
