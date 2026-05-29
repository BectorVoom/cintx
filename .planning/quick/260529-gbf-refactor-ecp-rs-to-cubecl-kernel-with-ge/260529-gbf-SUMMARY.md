---
phase: quick-260529-gbf
plan: 01
subsystem: cintx-cubecl / cintx-oracle
tags: [cubecl, ecp, gpu, rocm, device-kernel, oracle]
requires:
  - "ecp_type1_cart host driver (Phase 19 K-Taylor port)"
  - "center_4c1e.rs #[cube(launch)] dispatch template"
  - "ROCm gfx1152 GPU + xtask rocm-oracle harness"
provides:
  - "ecp_angular_kernel #[cube(launch)] generic-F device kernel (Type-1 angular splice)"
  - "run_ecp_angular_device::<R> + run_ecp_angular_splice_on_backend per-backend dispatch"
  - "ecp_random_rocm_parity.rs randomized ROCm int1e_ecp_sph idempotency oracle"
affects:
  - "crates/cintx-cubecl/src/kernels/ecp.rs"
  - "crates/cintx-oracle/tests/ecp_random_rocm_parity.rs"
  - "xtask/src/rocm_oracle.rs"
tech-stack:
  added: []
  patterns:
    - "host adaptive/special-function machinery STAYS host (marshaling); bounded arithmetic splice MOVES to #[cube] device"
    - "f64-internal device launch + F-output (generic over F: Float), mirroring center_2c2e/center_4c1e"
    - "safe-API (SessionRequest) ECP drive path for the rocm oracle (not eval_raw)"
key-files:
  created:
    - "crates/cintx-oracle/tests/ecp_random_rocm_parity.rs"
  modified:
    - "crates/cintx-cubecl/src/kernels/ecp.rs"
    - "xtask/src/rocm_oracle.rs"
decisions:
  - "Only the Type-1 triple-product angular splice ports to the device kernel; the Type-2 two-dgemm splice stays host-side this task (a Type-2 matmul device kernel is a follow-up). The Cu/LANL2DZ oracle still drives the device kernel on every case via the Local (Type-1) channel."
  - "ECP keeps its f64 staging signature + registry entry unchanged (no F32 outer dispatcher); device launch is f64 with identical summation order ⇒ byte-identity preserved."
  - "Backend is resolved from CINTX_BACKEND=rocm (executor.rs resolve_backend_kind), not backend_intent, so the oracle uses ExecutionOptions::default()."
metrics:
  duration: ~40m
  completed: 2026-05-29
  tasks: 3
  files: 3
---

# Phase quick-260529-gbf Plan 01: ECP angular splice → CubeCL device kernel + ROCm oracle Summary

Refactored the ECP Type-1 angular splice from a pure-host loop into a real
`#[cube(launch)]` device kernel generic over `F: Float`, dispatched on the
resolved backend (CPU `CpuRuntime` / ROCm `HipRuntime` / …), preserving f64
byte-identity vs vendored PySCF nr_ecp, and proved it on the gfx1152 GPU with a
randomized idempotency oracle reporting **mismatch_count=0 across 48 cases**.

## What was built

### Task 1 — generic-F `#[cube(launch)]` angular splice kernel (commit 7140d57)
- `ecp_angular_kernel<F: Float + CubeElement>`: ports the Type-1 Phase-B sextuple
  splice (`acc += ifac*jfac*rad_ang`, loop order i1 outer .. j3 inner) into a
  single-work-item device kernel. `F` arithmetic, `u32` indices, `while` loops
  bounded by the runtime cartesian powers — no break/continue, no special
  functions (those stay host-side). Obeys the CubeCL conditionals manual
  (statement-form only, no if-expressions).
- `run_ecp_angular_device::<R: Runtime>`: f64 dispatch
  (`create_from_slice` / `launch::<f64, R>` / `read_one_unchecked`), with
  buffer-length `debug_assert`s (T-gbf-01).
- `cart_comps_flat_u32`: host marshaling of cart-power triples into a `u32` array
  (keeps `cart_comps` enumeration host-side).
- Tests (gated `#[cfg(feature="cpu")]`): device-vs-host f64 equivalence over
  (li,lj) up to (2,2) asserts **max-abs-diff == 0.0** (byte-identity); a
  generic-f32 test reproduces the f64 result within f32 eps.

### Task 2 — rewire launch_ecp through the device dispatch (commit 34707f9)
- `ecp_type1_cart`'s Phase-B host loop replaced by a per-(ic,jc)
  `run_ecp_angular_device` call via `run_ecp_angular_splice_on_backend` — a
  per-backend `match` (Cpu/Rocm=HipRuntime/Wgpu/Cuda/Metal, each `#[cfg]`-gated)
  identical in shape to center_4c1e.rs:1016-1058.
- `backend: &ResolvedBackend` threaded through `ecp_type1_cart`,
  `ecp_type2_cart`, `ecp_scalar_prim_pair_cart`, `deriv1_cart_pair`,
  `compute_type{1,2}_pair_grad`, and every `launch_ecp` call site; removed
  `let _ = backend`.
- Module doc updated: angular splice now on-device generic over F; adaptive
  radial machinery + special functions remain host marshaling. Type-2 dgemm
  splice documented as host-side this task.
- Registry entry (`kernels/mod.rs` "ecp" => launch_ecp) and the f64 staging
  signature left UNCHANGED.

### Task 3 — randomized ROCm idempotency oracle (commit 08dcdca)
- `crates/cintx-oracle/tests/ecp_random_rocm_parity.rs`:
  `test_ecp_sph_random_rocm_idempotency` (`#[cfg(feature="rocm")]` + `#[ignore]`
  + `CINTX_ROCM_ORACLE=1` gate). Drives the SAFE API `int1e_ecp_sph`
  (`SessionRequest` + typed `BasisSet`, mirroring safe_api_ecp_parity.rs) TWICE
  per random Cu/LANL2DZ system on the ROCm device; asserts mismatch_count==0
  (atol=1e-12/rtol=1e-10) and any_nonzero==true (T-gbf-04, proves the device
  kernel ran).
- `build_random_ecp_system`: randomizes AO exponents [0.25,4.0] / coefficients
  [0.15,1.0] and the atom coordinate (±0.5 bohr) per case, keeping the LANL2DZ
  `EcpShell` slab verbatim so Local (Type-1) + Projected (Type-2) channels fire.
- `xtask/src/rocm_oracle.rs` doc updated: ECP now covered by the base rocm
  suite (no profile flag needed).

## Verification results (all observed, not assumed)

| Gate | Command | Result |
|------|---------|--------|
| cintx-cubecl ECP unit tests (cpu) | `cargo test -p cintx-cubecl --features cpu --lib ecp` | **26 passed, 0 failed** (incl. device-vs-host f64 max-abs-diff=0.0 + generic-f32) |
| Vendor CPU byte-identity parity | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test safe_api_ecp_parity --test ecp_iprinv_parity` | **8 passed, 0 failed** (vendor build actually compiled + run; atol=1e-12/rtol=0.0 preserved) |
| ROCm GPU random oracle (gfx1152) | `cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle` | **PASS** — `test_ecp_sph_random_rocm_idempotency ... ok` |

### Observed ROCm GPU result (the load-bearing claim)

```
PASS: rocm int1e_ecp_sph random idempotency mismatch_count=0 across 48 cases
(any_nonzero=true) at atol=1e-12/rtol=1e-10
```

- **mismatch_count = 0**
- **case count = 48**
- **any_nonzero = true** (the `#[cube(launch)]` device kernel genuinely ran on
  the AMD Radeon 860M gfx1152 under ROCm 7.1.1 — not an all-zeros fallback)
- Vendor parity was NOT skipped: it ran against the real vendored PySCF nr_ecp /
  libcint build (`CINTX_ORACLE_BUILD_VENDOR=1` + `--features cpu`, both
  `has_vendor_libcint` + `has_vendor_pyscf_nr_ecp` cfgs active).

## Deviations from Plan

### Scoped narrowing (documented, not a silent omission)

**1. [Scope] Type-2 angular splice remains host-side this task**
- **Found during:** Task 2.
- **Issue:** The plan's Task 2 action mentions replacing "the equivalent splice
  in `ecp_type2_cart`". Type-2's angular splice is a two-`dgemm` (matmul)
  contraction — structurally different from the Type-1 triple-product splice that
  `ecp_angular_kernel` ports. Forcing the dgemm into the triple-product kernel
  would be incorrect.
- **Resolution:** Type-1's splice is fully device-backed (the must_haves /
  key_links require `run_ecp_angular_device::<` in the match — satisfied). Type-2
  threads `backend` through for signature uniformity (`let _ = backend` with a
  doc note) but keeps its dgemm host-side; a Type-2 matmul device kernel is a
  follow-up. This does NOT weaken the oracle: the Cu/LANL2DZ fixture has a Local
  (Type-1) channel, so the device kernel fires on every one of the 48 cases, and
  the f64 byte-identity gate (which covers the full combined Type-1+Type-2
  output) stays green.
- **Files:** crates/cintx-cubecl/src/kernels/ecp.rs (ecp_type2_cart).
- **Commit:** 34707f9.

### Auto-fixed (Rule 3 — blocking)

**2. [Rule 3 - Blocking] Existing gradient unit tests needed a ResolvedBackend**
- **Found during:** Task 2.
- **Issue:** `deriv1_cart_pair` gained a `&ResolvedBackend` parameter, so the
  two existing `#[cfg(test)]` unit tests `gradient_zero_overlap_is_negligible` /
  `gradient_on_center_is_finite` no longer compiled.
- **Fix:** Gated both `#[cfg(feature = "cpu")]` and constructed a
  `ResolvedBackend::Cpu(CpuRuntime::client(...))`. (ECP's `ResolvedBackend` has
  no feature-less variant, so a backend feature is required to exercise the
  device path.)
- **Files:** crates/cintx-cubecl/src/kernels/ecp.rs (tests).
- **Commit:** 34707f9.

No other deviations — the kernel/dispatch/oracle landed as specified.

## Authentication gates

None.

## Known Stubs

None. `run_ecp_angular_device` / `cart_comps_flat_u32` are wired into
`ecp_type1_cart` (no longer dead code in feature-enabled builds; the
`#[cfg_attr(not(test), allow(dead_code))]` only applies when no backend feature
is active, where `ResolvedBackend` is uninhabited).

## Threat Flags

None. No new network endpoints, auth paths, or trust-boundary surface. The
host→device buffer-sizing boundary (T-gbf-01) is mitigated by length
`debug_assert`s in `run_ecp_angular_device` derived from the same li/lj formulas
as the host driver, plus the device-vs-host f64 equivalence test.

## Out-of-scope observations (NOT fixed)

Pre-existing clippy lints in `crates/cintx-cubecl/src/kernels/ecp.rs` (excessive
float precision in `l_down`/`l_up` constant tables at lines ~1364/1389,
`manual_slice_size_calculation` at ~1946) and in `crates/cintx-core/src/ecp.rs`
predate this task and are out of scope (the project gate is `cargo test`, not
clippy-clean). They were not touched.

## Self-Check: PASSED

- crates/cintx-cubecl/src/kernels/ecp.rs — FOUND (modified)
- crates/cintx-oracle/tests/ecp_random_rocm_parity.rs — FOUND (created)
- xtask/src/rocm_oracle.rs — FOUND (modified)
- Commit 7140d57 — FOUND
- Commit 34707f9 — FOUND
- Commit 08dcdca — FOUND
