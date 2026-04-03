# Domain Pitfalls

**Domain:** CubeCL client API migration, GPU integral kernels, multi-backend switching, oracle parity — cintx v1.1
**Researched:** 2026-04-02
**Supersedes:** Prior general-architecture pitfalls from 2026-03-21 (absorbed below where still valid)
**Overall confidence:** HIGH for CubeCL/wgpu specifics derived from code inspection and API evidence; MEDIUM for numerical precision claims (literature-sourced, not empirically tested in this codebase); LOW for CUDA/Metal backend gotchas (insufficient hardware coverage to verify)

---

## Critical Pitfalls

### Pitfall 1: ArrayArg outlives the Handle it borrows — use-after-free on the GPU

**What goes wrong:**
`ArrayArg::from_raw_parts::<T>(&handle, len, vectorization)` takes a reference to a `Handle`. If the `Handle` is dropped or moved before the `unsafe { kernel::launch_unchecked(...) }` call completes — or if the handle is cloned without incrementing an internal ref-count — the kernel dispatch may reference freed device memory.

**Why it happens:**
CubeCL's `Handle` is a thin wrapper around a server-side allocation identifier. Rust's borrow checker enforces the reference lifetime inside the `unsafe` block syntactically, but nothing prevents the caller from creating a temporary handle expression, passing a reference to it in the `ArrayArg`, and having it drop at the end of that statement while the kernel is still enqueued.

Example of the bug pattern:
```rust
// WRONG: handle drops at statement end, ArrayArg holds a dangling ref
unsafe {
    kernel::launch_unchecked::<R>(
        &client,
        count, dim,
        ArrayArg::from_raw_parts::<f64>(&client.empty(n * 8), n, 1), // temporary
    )
}
```

**Consequences:**
Silent read of zero-initialized or garbage device memory. No Rust safety violation surfaced because the unsafe block suppresses the checks.

**Prevention:**
- Bind every handle to a named `let` before any `ArrayArg` construction.
- Audit all `ArrayArg::from_raw_parts` call sites: the reference must outlive the entire unsafe block including the kernel enqueue.
- The safe pattern: `let out_handle = client.empty(n * 8); unsafe { ..., ArrayArg::from_raw_parts::<f64>(&out_handle, n, 1), ... }`.

**Detection:**
- Staging output is all-zeros even when the kernel ran (no wgpu-capability error).
- Inserting a bounds check inside the kernel reveals the array appears empty.
- Replace with `client.create(f64::as_bytes(&vec![1.0; n]))` and confirm readback matches — if it doesn't, handle lifetime is the suspect.

**Phase to address:** Executor rewrite (v1.1, first plan).

---

### Pitfall 2: CubeCL device double-initialization panic — OnceLock already caches `WgpuRuntime`

**What goes wrong:**
Calling `cubecl::wgpu::init_setup::<G>(device, options)` twice for the same device panics with "A server is still registered for device ...". The existing `runtime_bootstrap.rs` wraps the default case in `OnceLock` to prevent this, but any non-default selector or any path that bypasses the bootstrap (e.g., calling `WgpuRuntime::client(&device)` directly) can trigger a second init.

**Why it happens:**
CubeCL internally registers a compute server per device. The `OnceLock` guard in `runtime_bootstrap.rs` only covers `AdapterSelector::Auto`. When direct `WgpuRuntime::client()` calls are added (the v1.1 goal), they must either reuse the same device instance or go through the same singleton guard.

**Consequences:**
Test suite panics at the second test that touches the GPU. CI becomes unreliable depending on test ordering.

**Prevention:**
- Keep all device initialization behind `bootstrap_wgpu_runtime`; never call `init_setup` directly from kernel modules.
- When kernel launchers need a `ComputeClient`, acquire it via `WgpuRuntime::client(&device)` using the device already initialized by the preflight — do not call `init_setup` again.
- Consider extending the `OnceLock` or a per-selector registry if non-default selectors need direct client access.
- Use `std::panic::catch_unwind` in tests that probe multiple devices to avoid propagating panic to the whole test binary.

**Detection:**
- Test binary panics with "A server is still registered for device" message.
- Failure appears only when two tests in the same binary both call the GPU path.

**Phase to address:** Executor rewrite (v1.1, first plan) — before any test that calls a real kernel.

---

### Pitfall 3: `f64` (double precision) unavailable in WGSL shaders on most wgpu backends

**What goes wrong:**
The cintx oracle tolerances require f64-level accuracy (1e-11 atol for 1e family). WGSL — the shader language used by wgpu's Vulkan/Metal/DX12 backends — does not support `f64` natively. The existing `capability.rs` already checks for `SHADER_F64` wgpu feature, but this feature is only available on Vulkan backends with explicit driver support, is absent from Metal, and is absent from WebGPU.

**Why it happens:**
`wgpu::Features::SHADER_F64` (aka `SHADER_FLOAT64`) requires the underlying driver to expose `VK_KHR_shader_float64` (Vulkan) or equivalent. Most consumer GPUs advertise this via Vulkan, but Metal does not support 64-bit shader arithmetic and WebGPU explicitly excludes it. CubeCL's wgpu backend targets WGSL, which means even when the wgpu feature flag exists, the generated WGSL shader cannot use `f64` operations.

**Consequences:**
- Kernels that compute integrals in f32 produce outputs with ~7 significant digits. The 1e-11 atol oracle threshold requires ~12 significant digits. Every oracle comparison fails.
- Kernels that emulate f64 with two-f32 (double-double arithmetic) are correct but 8–16x slower and require non-trivial shader code.
- The `SHADER_F64` capability check currently only records the feature presence for diagnostic purposes; it does not block kernel dispatch. A kernel that assumes f64 and silently falls back to f32 will look like it ran but produce wrong results.

**Prevention:**
- Decide before writing a single kernel whether the wgpu backend will use: (a) CUDA/Vulkan-native f64, (b) double-double emulation on wgpu, or (c) a mixed CPU/GPU path where the GPU only runs f32-safe sub-computations.
- Add an explicit capability gate: if the kernel family requires f64 and `SHADER_F64` is absent, return `UnsupportedApi` with reason `missing_shader_f64` rather than silently running in f32.
- Document the precision contract for each kernel family in the kernel module doc comment.
- For the wgpu path, treat f32 as a structural constraint and validate whether oracle tolerances can be relaxed for that backend, or whether the CPU fallback path handles f64.

**Detection:**
- All oracle comparisons fail with `max_abs_error` around 1e-7 to 1e-5 (f32 noise floor).
- Comparing cart kernel outputs to a reference CPU implementation shows the same 7-digit precision pattern.
- Capability snapshot in the preflight report shows `SHADER_F64` absent.

**Phase to address:** Kernel design phase — must be resolved before any real kernel is written, not discovered post-implementation.

---

### Pitfall 4: Boys function numerical breakdown — wrong domain branching on GPU

**What goes wrong:**
The Boys function `F_n(x) = integral_0^1 t^(2n) exp(-x t^2) dt` is evaluated by multiple numerical strategies depending on the argument `x`. On CPU, libcint uses pre-computed Taylor tables for small `x`, a downward recurrence for moderate `x`, and asymptotic expansion for large `x`. On GPU, the branching logic must be implemented inside the shader, and selecting the wrong branch boundary or using the downward recurrence in a regime where it is numerically unstable causes catastrophic cancellation.

**Why it happens:**
The downward recurrence `F_{n-1}(x) = (2 * x * F_n(x) + exp(-x)) / (2*n - 1)` is numerically unstable for small `x` because it magnifies rounding error. The upward recurrence is stable but slow for large `n`. GPU threads cannot branch on per-thread domain analysis without warp/wave divergence. Literature (Tsuji 2025) confirms that a "gridded Taylor expansion" covering a wider input range with lower cost outperforms the naive origin Taylor approach on GPU.

**Consequences:**
- Boys function output has catastrophic cancellation error for `x < 1` or `x > 40` (regime boundaries vary by order `n`).
- Integrals that include a nuclear attraction term (1e, 3c1e families) systematically diverge from oracle.
- The error appears smoothly wrong (not NaN/inf), making it easy to miss until oracle comparison runs.

**Prevention:**
- Use the upward recurrence for `x < threshold(n)` and asymptotic expansion for `x > large_threshold`.
- Avoid the downward recurrence entirely on GPU; start from `F_0(x) = erf(sqrt(x)) * sqrt(pi/4x)` and recurse upward.
- Pre-compute tabulated values for common `n` values (n=0..12 covers most 1e/2e families) and use a lookup-plus-correction scheme.
- The CPU oracle uses 12–14 significant digit accuracy. Any GPU Boys implementation must be validated against the CPU reference before being wired into a kernel family.

**Detection:**
- int1e_nuc (nuclear attraction) oracle fails; int1e_ovlp (overlap) passes. Overlap does not require Boys function; nuclear attraction does.
- Error is smooth and systematic (not noisy), usually a ~constant relative offset.
- Plotting GPU output vs CPU output shows a smooth bias rather than random scatter.

**Phase to address:** First kernel implementation plan. Validate Boys function standalone before embedding it in a full kernel.

---

### Pitfall 5: Recurrence relation accumulates floating-point error for high angular momentum (l >= 3)

**What goes wrong:**
Obara-Saika (OS) horizontal recurrence relations for multi-center integrals involve subtracting terms with similar magnitude at high angular momentum. For `l >= 3` (f functions), the intermediate values in the recurrence can be 10-100x larger than the final integral, requiring ~2 extra digits of precision. In f32 this is fatal; in f64 it is borderline for certain shell configurations.

**Why it happens:**
The OS recurrence computes `[a+1|b] = [a|b+1] + (A-B) * [a|b]` in a tight loop. When `A-B` (the interatomic distance component) is large, intermediate values grow polynomially with angular momentum. The cancellation at the final sum step loses digits. Literature (Miao & Merz 2013, Yokogawa 2026) confirms recurrence-based GPU implementations work well for `l <= 1` (s and p shells) but require restructuring for `l >= 2`.

**Consequences:**
- Oracle failures for f-shell (l=3) and beyond, even in f64.
- The 4c1e family with `max(l) <= 4` is particularly exposed.
- Failures are configuration-dependent: two nearby atoms exacerbate the problem; well-separated atoms pass.

**Prevention:**
- Use the matrix form of McMurchie-Davidson (MD) recurrences for `l >= 2`; OS is suitable for `l <= 1`.
- For f and g shells, validate against a well-separated test geometry first (where cancellation is minimal), then test near-nuclear configurations.
- The existing oracle tolerances for 3c1e (atol=1e-7, rtol=1e-5) and 4c1e (atol=1e-6, rtol=1e-5) already reflect this family-specific precision degradation — do not tighten them without verifying hardware f64 precision.
- Add a unit test for each recurrence step independently before composing them.

**Detection:**
- Oracle failures increase with shell angular momentum (l=0,1 pass; l=2,3 fail).
- Error pattern is geometry-dependent: small `R_AB` distance exacerbates failure.
- Intermediate workspace values are orders of magnitude larger than final output.

**Phase to address:** Per-family kernel implementation plans. Check l=0,1 first, then gate l=2+ on passing those.

---

### Pitfall 6: Cart-to-spherical transform replaces the stub — staging ownership contract breaks

**What goes wrong:**
The current `c2s::cart_to_spheric_staging` is a placeholder (a smoothing blend). When replaced with a real Condon-Shortley-convention c2s transform, the number of output elements changes: a Cartesian d-shell has 6 components; a spherical d-shell has 5. If the staging buffer is sized for Cartesian output but the c2s transform writes spherical output in-place, the buffer is the wrong size and the compat layer reads garbage.

**Why it happens:**
The staging buffer size is determined in `TransferPlan::chunk_staging_elements` based on `plan.output_layout.staging_elements`. If the output layout currently assumes Cartesian sizing and the c2s transform is applied before the final compat write, there is a size mismatch for any `l >= 2` shell.

**Consequences:**
- Wrong output element count written to the compat flat buffer.
- `diff_summary` in the oracle reports `max_abs_error: INFINITY` (length mismatch branch).
- No panic; the mismatch is silent until oracle runs.

**Prevention:**
- Ensure `plan.output_layout.staging_elements` reflects the post-transform (spherical) element count when `Representation::Spheric` is requested.
- The `TransferPlan` sizing logic must be updated in lockstep with the real c2s transform.
- Add a test that verifies `staging.len() == expected_spheric_elements(shells, rep)` after `apply_representation_transform`.
- Keep the Cart path as the baseline: implement and validate Cart first, then Spheric, then Spinor.

**Detection:**
- Oracle diff reports length mismatch (INFINITY error) for spheric but not cart variants.
- `assert_flat_buffer_contract` in `compare.rs` fails at `values.len() != fixture.required_elements()`.
- Failure only for shells with `l >= 2`.

**Phase to address:** c2s transform implementation plan. Do not merge real c2s before staging sizing is confirmed correct.

---

### Pitfall 7: `RecordingExecutor` in `cintx-compat` must be removed in sync with executor rewrite

**What goes wrong:**
`crates/cintx-compat/src/raw.rs` wraps `CubeClExecutor` in a `RecordingExecutor<E>` that captures staging output via a `Mutex<Vec<f64>>` so that `eval_raw()` can retrieve values. When the executor is rewritten to use direct CubeCL client API and real kernels, the staging capture path in `RecordingExecutor` may either (a) capture pre-transform values, (b) capture post-transform values, or (c) capture nothing if `io.staging_output()` is called before values are written. Any of these produces wrong results in the raw API without surfacing an error.

**Why it happens:**
`RecordingExecutor::execute` calls `io.staging_output()` after delegating to the inner executor. This assumes the staging slice is already populated at that point. In a direct-client-API executor, the kernel results are read back from the GPU handle — if the readback is not committed to the staging slice before `execute` returns, the recording captures zeros.

**Consequences:**
- `eval_raw()` returns a buffer of zeros even though the GPU ran correctly.
- The raw API passes its own basic "not-zero" tests but fails oracle parity.
- The compat facade users (oracle, C ABI callers) get silently wrong results.

**Prevention:**
- Treat `RecordingExecutor` removal as a first-class v1.1 task, not an afterthought.
- Ensure the new executor's `execute` implementation writes final values into `io.staging_output()` before returning `Ok(stats)`, so any wrapper that reads from it gets the real data.
- Add a test that exercises `eval_raw` end-to-end and checks that the returned buffer is non-zero (and matches the oracle fixture) after a real kernel runs.
- Do not remove `RecordingExecutor` before the inner executor reliably populates staging.

**Detection:**
- `eval_raw` returns a `RawEvalSummary` with `not0 == 0` even when a GPU adapter is available.
- Oracle parity test passes for the safe API but fails for the raw API on the same geometry.

**Phase to address:** Executor rewrite (v1.1, first plan) — remove or refactor `RecordingExecutor` in the same commit that wires the real kernel readback.

---

## Moderate Pitfalls

### Pitfall 8: Multi-backend feature flag conflicts break test isolation

**What goes wrong:**
Adding `cuda` or `cpu` backend feature flags to `cintx-cubecl` can cause compilation failures or test runtime errors if two backends initialize competing global state in the same test binary. CubeCL backend feature flags (`wgpu`, `cuda`, `cpu`) are additive, but multiple backends registering devices for the same logical ID can panic.

**Prevention:**
- Run backend-specific integration tests in separate binaries (`#[cfg(feature = "cuda")]` test modules with `nextest --filter`).
- Never enable `wgpu` and `cuda` features in the same test binary unless CubeCL explicitly supports multi-backend co-existence (verify in the CubeCL changelog before adding).
- Use a `BackendKind` enum in the executor configuration to select the backend at runtime, not at compile time, when both features are present.

**Detection:** Panic on second device initialization in multi-feature test runs.

**Phase to address:** Multi-backend switching plan.

---

### Pitfall 9: Generic `Runtime` bounds infect public crate boundaries

**What goes wrong:**
When kernels are made generic over `R: Runtime`, every function that calls them must also be generic over `R`. This propagates into `ExecutionPlan`, `BackendExecutor`, and eventually the safe facade unless carefully isolated behind a trait object boundary.

**Prevention:**
- Keep the `R: Runtime` generic bound contained inside `cintx-cubecl` crate internals.
- The `BackendExecutor` trait in `cintx-runtime` takes `&ExecutionPlan` and `&mut ExecutionIo` — not `R` — preserving backend opacity.
- The existing executor dispatch path already achieves this: `CubeClExecutor` selects backend inside `execute()` without exposing `R` in its public signature. Maintain this pattern.
- If CUDA or cpu backends require a different `client` acquisition path, wrap both behind an internal `enum BackendClient { Wgpu(...), Cuda(...) }` rather than making the outer trait generic.

**Detection:** Compiler error "the trait `BackendExecutor` cannot be made into an object" or `dyn BackendExecutor` fails to compile when `R: Runtime` leaks into the trait.

**Phase to address:** Multi-backend switching plan.

---

### Pitfall 10: Transfer plan sizes do not account for real kernel input buffers

**What goes wrong:**
The existing `TransferPlan::stage_device_buffers` only probes host-side allocation (it allocates a `Vec<u8>` probe for workspace). When real kernels are implemented with `client.create(...)` calls for shell coordinates, exponents, and coefficients, the actual GPU memory usage is higher than what `transfer_bytes` reports to `ExecutionStats`.

**Prevention:**
- Compute `transfer_bytes` as the sum of all `client.create` buffer sizes plus the output buffer size.
- Expose total device allocation size in `ExecutionStats` rather than only the workspace bytes, so OOM tracking is accurate.
- Update `stage_device_buffers` to accept the list of input buffer sizes when real kernels are wired.

**Detection:** `stats.transfer_bytes` is systematically smaller than actual GPU peak memory. OOM errors appear before the planner's threshold predicts them.

**Phase to address:** First kernel implementation plan.

---

### Pitfall 11: Near-zero integral oracle comparison uses wrong tolerance path

**What goes wrong:**
The current `diff_summary` function in `compare.rs` uses `ZERO_THRESHOLD = 1e-18` as the boundary for switching between absolute-tolerance and relative-tolerance comparison. For integrals that are legitimately near zero (e.g., symmetry-forbidden terms), the relative tolerance branch incorrectly reports a large relative error when the absolute error is well within `atol`.

**Prevention:**
- The existing implementation is correct in structure: `if abs_ref < zero_threshold { use atol } else { use atol + rtol * abs_ref }`. Verify `ZERO_THRESHOLD = 1e-18` is actually below the smallest non-zero integral value in the fixture set — if fixtures contain values like `1e-20`, the threshold needs lowering.
- Do not expand the oracle fixture set to include pathological geometries (near-zero by symmetry) without adding a geometry annotation that identifies them as expected-zero cases.
- Distinguish "should be zero by symmetry" from "computed to be nearly zero" in fixture metadata.

**Detection:** Oracle reports `within_tolerance: false` for integrals the oracle claims are exactly zero. `max_rel_error` is infinity or very large while `max_abs_error` is smaller than `atol`.

**Phase to address:** Oracle parity testing plan.

---

### Pitfall 12: Spinor interleaved real/imag layout breaks if kernel writes wrong stride

**What goes wrong:**
The spinor representation uses interleaved complex doubles: `[re0, im0, re1, im1, ...]`. If the GPU kernel writes a planar layout `[re0, re1, ..., im0, im1, ...]` — which is the more natural GPU output layout — the c2spinor transform in `transform/c2spinor.rs` will interpret the data as interleaved and produce nonsense output. This is completely silent: the buffer length is correct, all values are finite, and only oracle comparison catches it.

**Prevention:**
- Define the wire format (interleaved vs. planar) as a kernel contract documented in the kernel module's doc comment.
- Add a unit test for the spinor kernel output specifically: write a synthetic kernel that outputs `[1.0, 0.0, 2.0, 0.0]` (real-only) and verify it reads back as `[(1+0i), (2+0i)]` in the interleaved representation.
- The oracle fixture for spinor integrals includes `"complex_interleaved": true` metadata — use this as a canary.

**Detection:** Spinor oracle failures where cart and sph variants of the same family pass. `max_abs_error` is large and consistent across all elements (not random noise).

**Phase to address:** Spinor kernel implementation or spinor oracle parity plan.

---

### Pitfall 13: Nondeterministic reduction order breaks oracle parity across runs (retained from prior research)

**What goes wrong:**
GPU work partitioning changes floating-point accumulation order. Two runs of the same kernel with different chunking or workgroup sizes produce results that differ at the last 1-2 digits, causing flaky oracle failures.

**Prevention:**
- Make reduction order a planner contract, not an implementation accident.
- Use deterministic reduction algorithms (tree reduction with fixed pattern, not atomic adds with arbitrary order).
- Record the workgroup size and chunk shape in tracing; require a re-run of oracle regression whenever these change.

**Detection:** Oracle failures appear sporadically; re-running the same test sometimes passes. Results differ by chunk size.

**Phase to address:** Base execution phase and every performance-tuning phase.

---

## Minor Pitfalls

### Pitfall 14: CubeCL type inference failure — cryptic ExpandElementTyped errors

**What goes wrong:**
CubeCL's proc-macro expansion of `#[cube]` functions requires that the element type be unambiguous at the call site. An unused mutable variable or an inferred-type scalar argument inside a `#[cube(launch)]` function produces a compile error referencing internal `ExpandElementTyped` traits with no source location.

**Prevention:**
- Always annotate the element type explicitly on `ArrayArg::from_raw_parts::<T>(...)`.
- Annotate kernel function signatures explicitly: `fn my_kernel(input: Array<F>, ...)` where `F` is a concrete type or a type alias.
- When a cryptic CubeCL macro error appears, check for unannotated or unused variables first.

**Phase to address:** Kernel development (all plans).

---

### Pitfall 15: Buffer `.clone()` required for multi-read but not obvious

**What goes wrong:**
If the output handle is passed to `ArrayArg::from_raw_parts` without cloning, and then also to `client.read_one(handle)`, the second usage may fail depending on CubeCL version, because the handle may be consumed or invalidated by the `ArrayArg` binding.

**Prevention:**
- Call `.clone()` on any handle that must be both passed to `ArrayArg` as output and read back later.
- Pattern: `let out = client.empty(n); unsafe { ..., ArrayArg::from_raw_parts::<f64>(&out, n, 1), ... }; let bytes = client.read_one(out.clone());`

**Phase to address:** Kernel development (all plans).

---

### Pitfall 16: Manifest coverage and oracle fixture sync (retained from prior research)

**What goes wrong:**
New kernel families are implemented but oracle fixtures for their symbols are not added. The feature-matrix CI passes the manifest audit but oracle parity is never checked.

**Prevention:**
- Treat oracle fixture addition as a blocking requirement for any kernel family PR — not a follow-up.
- Require `REQUIRED_REPORT_ARTIFACT` and `REQUIRED_MATRIX_ARTIFACT` to exist and pass before marking a family "implemented."

**Phase to address:** Oracle parity testing plan and every family implementation plan.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Executor rewrite to direct client API | Pitfalls 1, 2, 7 — handle lifetime, double-init, RecordingExecutor capture | Bind all handles to named `let`; use preflight singleton; wire real readback before removing RecordingExecutor |
| First kernel family (1e overlap) | Pitfall 3 — f64 unavailable in WGSL | Decide f64 strategy before writing any kernel code |
| Nuclear attraction (1e_nuc) kernel | Pitfall 4 — Boys function precision breakdown | Validate Boys function standalone; use upward recurrence only |
| d/f-shell recurrence | Pitfall 5 — cancellation for l>=2 | Use McMurchie-Davidson for l>=2; test near-nuclear geometries |
| Spheric representation | Pitfall 6 — staging size mismatch after real c2s | Update TransferPlan sizing before replacing the c2s stub |
| Spinor representation | Pitfall 12 — interleaved vs. planar layout | Document and test wire format before spinor oracle runs |
| Multi-backend switching | Pitfalls 8, 9 — feature flag conflicts, Runtime generic leakage | Isolate backend selection inside cintx-cubecl; test each backend in its own binary |
| Oracle parity testing | Pitfall 11 — near-zero tolerance path | Audit fixture values vs. ZERO_THRESHOLD; annotate expected-zero cases |
| Transfer/stats tracking | Pitfall 10 — transfer_bytes undercounts real GPU allocation | Include all `client.create` sizes in transfer accounting |

## Sources

### CubeCL API (MEDIUM confidence — derived from docs.rs and gist examples, no direct source inspection)
- CubeCL docs overview: https://docs.rs/cubecl/latest/cubecl/
- cubecl-wgpu types and initialization: https://docs.rs/cubecl-wgpu/latest/cubecl_wgpu/
- CubeCL architecture overview (community): https://gist.github.com/nihalpasham/570d4fe01b403985e1eaf620b6613774
- GPU-agnostic CubeCL example: https://www.thomasantony.com/posts/202512281621-gpu-agnostic-programming-using-cubecl/

### wgpu f64 precision (HIGH confidence — wgpu issue tracker)
- SHADER_FLOAT64 feature status: https://github.com/gfx-rs/wgpu/issues/1143
- Double precision WebGPU discussion: https://github.com/gpuweb/gpuweb/issues/2805

### Integral numerics (MEDIUM confidence — peer-reviewed literature)
- Boys function GPU implementation: https://onlinelibrary.wiley.com/doi/10.1002/cpe.8328
- ERI Obara-Saika GPU recurrence: https://pubs.acs.org/doi/full/10.1021/ct300754n
- One-electron GPU integration: https://onlinelibrary.wiley.com/doi/10.1002/cpe.70628
- McMurchie-Davidson GPU implementation: https://link.springer.com/chapter/10.1007/978-3-031-85697-6_14

### Codebase inspection (HIGH confidence — direct source read)
- `crates/cintx-cubecl/src/executor.rs` — current executor with OnceLock bootstrap, staging contracts
- `crates/cintx-cubecl/src/transfer.rs` — TransferPlan sizing, stage_device_buffers probe
- `crates/cintx-cubecl/src/transform/c2s.rs` — placeholder transform (smoothing blend, not real c2s)
- `crates/cintx-cubecl/src/kernels/one_electron.rs` — stub kernel (zero output)
- `crates/cintx-oracle/src/compare.rs` — tolerance constants, diff_summary implementation, ZERO_THRESHOLD
- `crates/cintx-compat/src/raw.rs` — RecordingExecutor wrapping CubeClExecutor, staging capture

---
*Pitfalls research for: cintx v1.1 CubeCL direct client API and real kernel compute*
*Researched: 2026-04-02*
