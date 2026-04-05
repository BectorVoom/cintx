# Milestones

## v1.1 CubeCL Direct Client API & Real Kernel Compute (Shipped: 2026-04-05)

**Phases completed:** 10 phases, 47 plans, 93 tasks

**Key accomplishments:**

- Manifest generation landed under `crates/cintx-ops/generated`, resolvers became metadata-driven, and core domain primitives gained rigorous validation guards plus regression tests.
- Manifest generation landed under `crates/cintx-ops/generated`, resolvers became metadata-driven, and core domain primitives gained rigorous validation guards plus regression tests.
- `cintx-runtime` now exposes a manifest-driven workspace/query/evaluate contract with typed validation failures, deterministic chunk planning, and tracing-backed planner diagnostics.
- Activated `cintx-compat`, `cintx-cubecl`, and `cintx-oracle` as first-class workspace crates with explicit compat/backend/oracle dependency routing and crate-level smoke-test coverage.
- Canonical manifest metadata now covers helper/transform/optimizer/legacy APIs with misc.h wrapper parity checks, and `cintxRsError` now exposes typed raw layout/env/buffer failures.
- Backend-neutral runtime execution now delegates validated chunk plans through `BackendExecutor` with deterministic scheduling and runtime-owned metrics.
- Concrete CPU-profile CubeCL executor with resident cache and staging transfer planning, plus canonical-family launch coverage for `1e`, `2e`, and `2c2e`.
- Raw libcint-style compat calls now resolve through the shared runtime/CubeCL path with strict layout guards and no-partial-write failure behavior.
- Phase 2 compatibility coverage is now complete across helper/transform/optimizer/legacy APIs with oracle-backed parity gates for base families.
- CubeCL now executes `3c1e`/`3c2e` families and applies representation-specific staging transforms before compat commits final caller-visible buffers.
- Refreshed Phase 3 lock wiring and reinforced stable-vs-unstable namespace boundaries for the safe facade and optional C ABI shim.
- Resolver regression checks now explicitly enforce F12/STG/YP cart/spinor symbol absence, and compat/CubeCL optional-family gates were re-verified across base and feature-enabled profiles.
- Implemented a typed safe query/evaluate facade that preserves runtime contracts and returns owned output from the executed backend staging path.
- C ABI shim wrappers now expose stable status constants and stricter fail-closed pointer validation while preserving thread-local copy-out diagnostics over compat raw APIs.
- Safe-surface manifests now expose explicit feature-forwarding contracts, and SessionBuilder/prelude shipped concrete typed ergonomics with invariant tests.
- Safe evaluate now enforces compat raw policy gates, so optional/source UnsupportedApi outcomes use shared raw reason text instead of facade-local checks.
- Manifest-profile-aware oracle fixture generation now covers all approved profiles and parity runs emit full mismatch evidence before failing.
- Manifest audit, oracle parity, helper/legacy parity, and OOM contract checks are now runnable from one xtask CLI surface with profile-scoped reports and fail-closed exits.
- Merge-blocking PR automation now runs concrete xtask manifest, oracle parity, helper/legacy parity, and OOM gate commands across the required feature-profile envelope.
- Criterion micro/macro/crossover benchmarks, threshold-gated bench reporting, and PR-advisory vs release-required GPU workflow policy were implemented with artifactized runtime diagnostics.
- Oracle crate root now explicitly re-exports Phase 4 profile-aware fixture and parity APIs/constants and meets the `min_lines: 20` artifact substance gate.
- GPU-required release/template workflows are now explicitly GPU-bound and fail when benchmark or diagnostics evidence is missing from both required and fallback artifact paths.
- Release governance now enforces explicit GPU/bench/artifact invariants while clearing the 180-line workflow substance threshold.
- Typed BackendKind/BackendIntent/BackendCapabilityToken contract added to ExecutionOptions and WorkspaceQuery with four-field planning_matches drift detection and fail-closed evaluate enforcement (D-03, D-08)
- Typed wgpu capability snapshot with FNV-1a fingerprint, D-12 reason taxonomy, and fail-closed bootstrap_wgpu_runtime selector parsing using cubecl-wgpu 0.9.0
- Real CubeCL chunk execution path without synthetic staging fill, with fail-closed wgpu preflight and D-12 unsupported taxonomy in executor and kernels
- Safe facade now imports cintx_cubecl::CubeClExecutor directly (no local stub), WorkspaceExecutionToken extended with backend contract fields for drift detection, and layered anti-pseudo/taxonomy regression tests added across compat and safe boundaries
- One-liner:
- RecordingExecutor staging retrieval in eval_raw and wgpu fingerprint propagation in both raw and safe facade query paths, closing the two v1.0 audit bugs
- Five regression tests in raw::tests covering eval_raw staging retrieval, fingerprint propagation, all-base-family coverage, representation layout contract, and optimizer equivalence — closing Wave 0 gap verification for COMP-01/05, EXEC-02/04/05, and VERI-01
- Cargo.toml changes:
- CubeClExecutor rewritten to resolve ResolvedBackend from BackendIntent, pass io.staging_output() directly to launch_family, gate wgpu dispatch on SHADER_F64 capability, and remove TransferPlan::stage_device_buffers from the execute path.
- RecordingExecutor deleted from all call sites; eval_raw and safe facade evaluate() now allocate owned staging buffers and read executor output directly via manual chunk loop
- Boys function gamma_inc_like ported as #[cube] with power-series/erfc branches and PairData #[derive(CubeType)] struct, validated to 1e-12 atol against libcint C reference
- 1. [Rule 3 - Blocking] Prerequisites not yet committed by Plan 01
- Obara-Saika VRR and HRR recurrence ported from g1e.c/g2e.c as `#[cube]` functions with host wrappers, validated end-to-end through a pdata->Boys->VRR pipeline covering s-s, p-s, and d-s shell pairs
- One-liner:
- 1. [Rule 1 - Bug] Kinetic integral sign convention
- H2O STO-3G oracle parity tests for int1e_ovlp_sph, int1e_kin_sph, int1e_nuc_sph with mismatch_count==0 and kinetic G-tensor index fix
- Vendored libcint 6.1.3 compiled from C source via cc crate; all three 1e spherical operators match upstream to atol=1e-11 after fixing kinetic D_j^2 formula and p-shell C2S ordering
- KERN-06 requirement marked Complete in REQUIREMENTS.md and H2O STO-3G oracle parity artifact committed to repository at artifacts/phase-09-1e-oracle-parity.md
- Rys root host wrappers for N=3..5 with unified dispatcher, four multi-index cart-to-sph transforms (2c2e/3c1e/3c2e/2e), and oracle vendor build extended with all 2e+ libcint source files and FFI wrappers
- Real Rys quadrature 2c2e kernel (fill_g_tensor_2c2e + VRR) passing vendor libcint 6.1.3 oracle parity at atol=1e-9 for H2O STO-3G after fixing PTR_RANGE_OMEGA env collision in test data
- Real three-center one-electron overlap kernel implementing CINTg3c1e_ovlp VRR+HRR algorithm, with oracle parity test passing at atol=1e-7 against vendored libcint 6.1.3 for H2O STO-3G across all 125 shell triples
- Real int3c2e host-kernel evaluation now matches vendored libcint 6.1.3 on H2O STO-3G at atol 1e-9, with explicit ibase-safe ij handling.
- Host-side `int2e_sph` now computes non-zero four-center ERIs via Rys quadrature and matches vendored libcint for H2O/H2 STO-3G at `atol=1e-12`, `rtol=1e-10`.
- Five-family oracle parity gate closed: 1e/2e/2c2e/3c1e/3c2e all pass vs vendored libcint 6.1.3 at D-06 tolerances with 0 mismatches; v1.1 milestone complete.

---
