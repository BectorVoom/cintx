---
phase: quick-260529-hin
plan: 01
subsystem: cintx-cubecl / cintx-oracle
tags: [cubecl, ecp, gpu, rocm, device-kernel, oracle, type2, dgemm]
requires:
  - "quick-260529-gbf Type-1 angular splice device kernel + ROCm ECP oracle"
  - "ecp_type2_cart host driver (two-dgemm angular splice, Phase 19 K-Taylor port)"
  - "ROCm gfx1152 GPU + xtask rocm-oracle harness"
provides:
  - "ecp_type2_angular_kernel #[cube(launch)] generic-F device kernel (Type-2 two-dgemm splice)"
  - "run_ecp_type2_angular_device::<R> + run_ecp_type2_splice_on_backend per-backend dispatch"
  - "device-backed ecp_type2_cart (and transitively compute_type2_pair_grad / deriv1_cart_pair)"
affects:
  - "crates/cintx-cubecl/src/kernels/ecp.rs"
tech-stack:
  added: []
  patterns:
    - "two-dgemm angular contraction ported to #[cube] by recomputing each intermediate buf entry inline (register-only, NO device-local Array) so f64 summation order stays byte-identical to the host dgemm loops"
    - "scalar F arg (common_fac) folded in-kernel so the host scatter is a plain += (mirrors Type-1 template)"
    - "f64-internal device launch + F-output (generic over F: Float), mirroring center_2c2e/center_4c1e and the Type-1 splice"
key-files:
  created: []
  modified:
    - "crates/cintx-cubecl/src/kernels/ecp.rs"
decisions:
  - "Type-2's two-dgemm splice ports to #[cube] WITHOUT a device-local Array: each host `buf` entry is an independent dgemm-1 inner sum, recomputed inline at the moment dgemm-2 consumes it (col = row*dlc + kk2/ljlc1, row2 = kk2 - (kk2/ljlc1)*ljlc1). The dgemm-1 (over kk in 0..lilc1) and dgemm-2 (over kk2 in 0..mq) sums run in the SAME order as the host loops => bit-identical f64."
  - "Used integer subtraction (kk2 - (kk2/ljlc1)*ljlc1) instead of `%` for the row2 decomposition - `/` is proven in #[cube] (center_2c2e/3c2e), and subtraction sidesteps any modulo concern while being exactly equal."
  - "common_fac is passed as a raw f64 scalar launch arg (the cubecl launch macro converts it; no ScalarArg::new wrapper) and folded in-kernel, matching center_2c2e_kernel's `common_factor: F` and the Type-1 host scatter contract."
  - "ECP keeps its f64 staging signature + registry entry unchanged (no F32 outer dispatcher); device launch is f64 with identical summation order => byte-identity preserved. No new capi enum variants, no legacy cint* wrappers."
metrics:
  duration: ~35m
  completed: 2026-05-29
  tasks: 3
  files: 1
---

# Phase quick-260529-hin Plan 01: ECP Type-2 two-dgemm angular splice -> CubeCL device kernel Summary

Ported the ECP **Type-2 (Projected) two-`dgemm` angular splice** out of the host
loop in `ecp_type2_cart` into a real `#[cube(launch)]` device kernel
(`ecp_type2_angular_kernel`) generic over `F: Float`, dispatched on the resolved
backend (CPU `CpuRuntime` / ROCm `HipRuntime` / ...), preserving f64 byte-identity
vs the prior host dgemm loops, and proved it on the gfx1152 GPU with the existing
randomized idempotency oracle reporting **mismatch_count=0 across 48 cases** while
the Type-2 device path was dispatched **12,288 times** (both Projected `lc=0` and
`lc=1` channels). This is the explicit follow-up to quick-260529-gbf, which ported
only the Type-1 triple-product splice and deferred Type-2.

## What was built

### Task 1 - `ecp_type2_angular_kernel` `#[cube(launch)]` + device dispatch + byte-identity tests (commit e84cb09)
- `ecp_type2_angular_kernel<F: Float + CubeElement>`: the Type-2 Phase-B two-dgemm
  angular contraction (nr_ecp.c:5505-5510) as a single-work-item device kernel
  (`UNIT_POS == 0`). It recomputes each host `buf` entry inline (NO device-local
  `Array` scratch - there is no proven `Array::new` precedent in
  center_2c2e/3c2e/4c1e), preserving the exact dgemm-1/dgemm-2 f64 summation
  order. `F` arithmetic, `u32` indices (`as usize` at index sites), statement-form
  `while` loops bounded by `li`/`lj`/`lc` - no break/continue, no if-expressions,
  no special functions. `common_fac` folded in-kernel; accumulates across the
  `i in 0..=li`, `j in 0..=lj` angular pair, emitting an `nfi*nfj` block in F-order
  `cart_out[col2*nfi + row]`.
- `run_ecp_type2_angular_device::<R: Runtime>`: f64 dispatch
  (`create_from_slice(f64::as_bytes(..))` / `launch::<f64, R>` /
  `read_one_unchecked` / `f64::from_bytes`), `common_fac` passed as a raw f64
  scalar arg, with buffer-length `debug_assert`s (T-hin-01: `prad.len()==d3`,
  `angi.len()==(li+1)*nfi*dlc*lilc1`, `angj.len()==(lj+1)*nfj*dlc*ljlc1`).
- `run_ecp_type2_splice_on_backend`: per-backend `match`
  (Cpu/Rocm=HipRuntime/Wgpu/Cuda/Metal, each `#[cfg]`-gated), mirroring
  `run_ecp_angular_splice_on_backend`.
- Tests (gated `#[cfg(feature="cpu")]`, in `mod ecp_angular_device_cross_check`):
  `host_type2_splice` reproduces ecp.rs's host two-dgemm (materialized `buf`) for a
  single (ic=0,jc=0) tuple; `ecp_type2_angular_device_matches_host_f64` asserts
  **max-abs-diff == 0.0** + `any_nonzero == true` over (li,lj) in
  {(0,0),(1,0),(0,1),(1,1),(2,0),(0,2),(2,1),(1,2),(2,2)} x lc in {0,1,2};
  `ecp_type2_angular_device_generic_f32_within_eps` reproduces the f64 result at
  F=f32 within eps (proves genuine generic-over-F).

### Task 2 - rewire `ecp_type2_cart` through the device path (commit 523f47f)
- Removed `let _ = backend;` from `ecp_type2_cart`. Replaced the host two-dgemm
  `for ic / for jc` loop with a per-(ic,jc) `run_ecp_type2_splice_on_backend`
  dispatch; the returned `nfi*nfj` block (already `common_fac`-scaled in-kernel,
  F-order `col2*nfi+row`) is scattered into the contraction-major `gctr` with a
  plain `+=` at `c_off = jc*nfj*di + ic*nfi` - NO double-multiply (confirmed). The
  host-side `type2_facs_ang` computation of `angi`/`angj` and `common_fac` stay
  host (marshaling); the unused host `buf`/`im`/`mq` were removed.
- Gradient drivers `compute_type2_pair_grad` / `deriv1_cart_pair` become
  device-backed transitively (they already thread `backend` into `ecp_type2_cart`);
  the existing `gradient_zero_overlap_is_negligible` / `gradient_on_center_is_finite`
  unit tests still pass.
- Module-doc Type-2 paragraph updated: the two-dgemm splice now runs on-device via
  `run_ecp_type2_splice_on_backend` (generic over F, f64-internal, byte-identity
  preserved); only the Phase-A adaptive radial machinery stays host marshaling.
- `launch_ecp` registry routing and the f64 staging signature left UNCHANGED. No
  new capi enum variants, no legacy `cint*` wrappers.

### Task 3 - ROCm oracle re-run + Type-2-exercised proof (no source change; probe added+removed in working tree, net-zero)
- Ran the existing `test_ecp_sph_random_rocm_idempotency` on gfx1152 via
  `cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle`.
- Established the Type-2 device path is genuinely exercised by **BOTH** plan
  methods (a) and (b) - see "Observed ROCm GPU result" below. A temporary
  `CINTX_ECP_TYPE2_TRACE`-gated dispatch counter was added to
  `run_ecp_type2_splice_on_backend`, the oracle was re-run to observe the count,
  then the probe was removed; the source committed in Task 2 is clean (the
  add+remove netted to a zero diff, so no separate Task 3 commit was needed).

## Verification results (all observed, not assumed)

| Gate | Command | Result |
|------|---------|--------|
| cintx-cubecl ECP unit tests (cpu) | `cargo test -p cintx-cubecl --features cpu --lib ecp` | **28 passed, 0 failed** (incl. Type-2 device-vs-host f64 max-abs-diff=0.0 + generic-f32, and the two Type-1 tests) |
| Vendor CPU byte-identity parity | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test safe_api_ecp_parity --test ecp_iprinv_parity` | **8 passed, 0 failed** (5 + 3); vendor build actually compiled + run; atol=1e-12/rtol=0.0 preserved |
| ROCm GPU random oracle (gfx1152) | `cargo run --manifest-path xtask/Cargo.toml -- rocm-oracle` | **PASS** - `test_ecp_sph_random_rocm_idempotency ... ok`; full suite "passed for profile `base`" |

### Vendor parity ran (NOT skipped)

The vendor parity tests are gated `#[cfg(all(has_vendor_libcint, has_vendor_pyscf_nr_ecp))]`
at `ATOL=1e-12`. They appeared in the run output as named, executed cases and
passed - NOT "0 passed; 0 filtered":

```
running 3 tests   (ecp_iprinv_parity)
test test_ECPscalar_iprinv_sph_cu_lanl2dz_parity ... ok
test test_ECPscalar_iprinv_cart_cu_lanl2dz_parity ... ok
test iprinv_at_cu_equals_ipnuc_for_single_ecp_atom_fixture ... ok
test result: ok. 3 passed; 0 failed; ...

running 5 tests   (safe_api_ecp_parity)
test coverage_invariant_holds ... ok
test test_int1e_ecp_cart_safe_api_parity ... ok
test test_int1e_ecp_sph_safe_api_parity ... ok
test test_int1e_ecp_ipnuc_sph_safe_api_parity ... ok
test test_int1e_ecp_ipnuc_cart_safe_api_parity ... ok
test result: ok. 5 passed; 0 failed; ...
```

The vendor FFI was compiled (16.77s build) and the Type-2 device path is on the
critical path for these Cu/LANL2DZ comparisons, so vendor parity genuinely passed
against vendored PySCF nr_ecp / nr_ecp_deriv at atol=1e-12/rtol=0.0 with the
device-backed Type-2 splice.

### Observed ROCm GPU result (the load-bearing claim)

Verbatim harness PASS line (captured with `--nocapture` from a direct
`CINTX_ROCM_ORACLE=1 CINTX_BACKEND=rocm` run of the same test the xtask drives):

```
  PASS: rocm int1e_ecp_sph random idempotency mismatch_count=0 across 48 cases (any_nonzero=true) at atol=1e-12/rtol=1e-10
```

- **mismatch_count = 0**
- **case count = 48**
- **any_nonzero = true** (device kernel genuinely ran on the AMD Radeon 860M
  gfx1152 under ROCm - not an all-zeros fallback)

### Type-2 device path genuinely exercised (BOTH methods)

**Method (a) - fixture inspection.** `crates/cintx-oracle/data/cu_lanl2dz.json`
`ecp.shells` has 3 entries; the Projected (non-local Type-2) channels are present:
- `shell[0]`: `channel="local"` (Type-1 -> `ecp_type1_cart`)
- `shell[1]`: `channel="projected"`, **l=0** (Type-2 -> `ecp_type2_cart`)
- `shell[2]`: `channel="projected"`, **l=1** (Type-2 -> `ecp_type2_cart`)

`build_random_ecp_system` parses `"projected"` -> `EcpChannel::Projected(l)` and
keeps the LANL2DZ slab verbatim, so the `(false,false) => ecp_type2_cart` dispatch
arm fires on every case.

**Method (b) - observed dispatch count (temporary probe, since removed).** With a
`CINTX_ECP_TYPE2_TRACE`-gated atomic counter at the top of
`run_ecp_type2_splice_on_backend`, the direct ROCm run produced **12,288**
`run_ecp_type2_splice_on_backend` dispatches across the 48 cases, e.g.:

```
[CINTX_ECP_TYPE2_TRACE] run_ecp_type2_splice_on_backend dispatch #1 (li=0 lj=0 lc=0)
[CINTX_ECP_TYPE2_TRACE] run_ecp_type2_splice_on_backend dispatch #2 (li=0 lj=0 lc=1)
...
[CINTX_ECP_TYPE2_TRACE] run_ecp_type2_splice_on_backend dispatch #12288 (li=2 lj=2 lc=1)
```

The observed `lc=0` and `lc=1` values match exactly the two Projected channel
angular momenta from the fixture - conclusively distinguishing Type-2 from Type-1
(Type-1 never calls `run_ecp_type2_splice_on_backend`). The probe was removed
before finalizing; the committed source is clean (`grep -c CINTX_ECP_TYPE2_TRACE
crates/cintx-cubecl/src/kernels/ecp.rs == 0`, and `cargo build -p cintx-cubecl
--features cpu` succeeds).

## Deviations from Plan

### Auto-fixed (Rule 3 - blocking)

**1. [Rule 3 - Blocking] Initial kernel referenced `a[..]` instead of `prad[..]`**
- **Found during:** Task 1 (first `cargo test` compile).
- **Issue:** The kernel computed `a_base` as the dgemm-1 source offset but indexed
  a nonexistent `a` binding (`a[a_idx]`); the array parameter is `prad`. Rust
  E0425 ("cannot find value `a`").
- **Fix:** Index `prad[a_idx as usize]` directly. (Also removed an unused `im`
  binding the kernel does not need, since `col` is derived inline.)
- **Files:** crates/cintx-cubecl/src/kernels/ecp.rs.
- **Commit:** e84cb09 (fixed before the Task 1 commit).

No other deviations - the kernel/dispatch/oracle landed as specified.

## Authentication gates

None.

## Known Stubs

None. `ecp_type2_angular_kernel` / `run_ecp_type2_angular_device` /
`run_ecp_type2_splice_on_backend` are wired into `ecp_type2_cart` (live in
feature-enabled builds; the `#[cfg_attr(not(test), allow(dead_code))]` only applies
when no backend feature is active, where `ResolvedBackend` is uninhabited).

## Threat Flags

None. No new network endpoints, auth paths, or trust-boundary surface beyond the
documented host->device buffer-sizing boundary (T-hin-01), which is mitigated by the
length `debug_assert`s in `run_ecp_type2_angular_device` (derived from the same
li/lj/lc formulas as the host driver) and exercised by the device-vs-host f64
equivalence test. The `(col,row2)` index decomposition (T-hin-02) is bounds-correct
by construction and proven bit-exact by the byte-identity test against the host
materialized-`buf` reference.

## Out-of-scope observations (NOT fixed)

Pre-existing clippy lints / `unused_import` warnings in
`crates/cintx-cubecl/src/kernels/f12.rs` and `crates/cintx-oracle/tests/*` predate
this task and are out of scope (the project gate is `cargo test`, not clippy-clean).
They were not touched.

## Self-Check: PASSED

- crates/cintx-cubecl/src/kernels/ecp.rs - FOUND (modified; contains
  `ecp_type2_angular_kernel`, `run_ecp_type2_angular_device`,
  `run_ecp_type2_splice_on_backend`)
- Commit e84cb09 - FOUND
- Commit 523f47f - FOUND
- Temporary probe removed - VERIFIED (`grep -c CINTX_ECP_TYPE2_TRACE == 0`,
  `cargo build -p cintx-cubecl --features cpu` succeeds)
