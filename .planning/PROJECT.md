# cintx

## What This Is

cintx is a public Rust library that redesigns and reimplements libcint with result compatibility as the primary goal. It provides a Rust-native safe API, a raw compatibility API for `atm`/`bas`/`env` style callers, and an optional C ABI shim for migration and interoperability. The target users are Rust developers and systems that need libcint-compatible integral evaluation with stronger type safety, clear failure modes, and high-confidence verification.

## Core Value

Deliver libcint-compatible results through a Rust-native API surface that stays type-safe, verifiable, and safe under memory pressure.

## Requirements

### Validated

- [x] Typed domain primitives, canonical manifest generation, and manifest-aware resolver foundations are in place and verified in Phase 1: Manifest & Planner Foundation.
- [x] Runtime planner/workspace scaffolding now exposes typed query/evaluate contracts, memory-limit chunking, and explicit validation failures, verified in Phase 1 Plan 02.
- [x] The three-layer surface (safe Rust API, raw compatibility API, optional C ABI shim) is implemented with feature-gated optional/unstable families and verified in Phase 3: Safe Surface, C ABI Shim & Optional Families.
- [x] Gaussian primitive infrastructure (Boys function, pair data, Rys quadrature, Obara-Saika recurrence) implemented as validated #[cube] functions in cintx-cubecl. Validated in Phase 8: Gaussian Primitive Infrastructure and Boys Function.
- [x] All five base integral families (1e, 2e, 2c2e, 3c1e, 3c2e) produce real kernel output through CubeCL backend with oracle parity confirmed against vendored libcint 6.1.3. Validated in Phase 10: 2e, 2c2e, 3c1e, 3c2e Real Kernels and Oracle Gate Closure.
- [x] Oracle gate closure passes all five base families with 0 mismatches. Validated in Phase 10.
- [x] Every helper, transform, and wrapper symbol in the manifest is oracle-wired with unified atol=1e-12 tolerance; 4c1e stub replaced with real polynomial recurrence kernel matching vendored libcint. Validated in Phase 11: Helper/Transform Completion & 4c1e Real Kernel.
- [x] Real spinor transforms with correct Clebsch-Gordan coupling; spinor oracle coverage unblocked for all families. Validated in Phase 12: Real Spinor Transform.
- [x] F12/STG/YP family kernels — all 10 sph symbols at oracle parity (atol=1e-12). Cart and spinor remain unsupported (sph-only enforcement). Validated in Phase 13: F12/STG/YP Kernels.
- [x] v1.1 executor infrastructure (EXEC-06/07/08/09, VERI-06) fully resolved — direct CubeCL client API, ResolvedBackend dispatch, CPU backend, f64 strategy. Validated in Phase 7.
- [x] Unstable-source family APIs (origi, grids, Breit, origk, ssc) implemented behind feature gate with oracle parity in nightly CI. Validated in Phase 14: Unstable-Source-API Families.
- [x] Oracle tolerance unified to atol=1e-12 for every family with catch-all tolerance and manifest-driven oracle eligibility; four-profile manifest lock regenerated with oracle_covered=true on all 110 stable/optional entries; CI oracle gate uses matrix strategy. Validated in Phase 15: Oracle Tolerance Unification & Manifest Lock Closure.
- [x] ECP Type-1/Type-2 evaluator (ECP-01..05): `int1e_ecp_{cart,sph}` scalar + `int1e_ecp_ipnuc_{cart,sph}` gradient pass byte-identity vs vendored PySCF nr_ecp at atol=1e-12 over Cu/LANL2DZ, dispatched through the standard `SessionRequest::evaluate` safe-API surface; exact K-Taylor radial machinery ported host-first with byte-locked `.bin` tables + CI drift-gate; optional non-blocking libecpint secondary oracle. Validated in Phase 19: `int1e_ecp_*` Type-1/Type-2 Evaluator (v1.3).
- [x] Plain-Coulomb gradient integral families (GRAD-01..10): the 6 first-derivative families every analytical gradient needs — `int2e_ip1`, `int1e_ipovlp`, `int1e_ipkin`, `int1e_ipnuc`, `int1e_iprinv`, `ECPscalar_iprinv` — plus the repaired `int3c2e_ip1` derivative kernel, all byte-identical to libcint 6.1.3 at atol=1e-12 under the vendor-gated oracle suite. Adds the `PTR_RINV_ORIG` env slot; component-leading `[3,…]` F-order matches pyscf-gto. Spinor gradients are registered-but-`UnsupportedApi` (R5/D-03). Validated in Phase 21: Plain-Coulomb Gradient Integral Families (v1.3).
- [x] Gauge-origin env slot (FND-01): `PTR_COMMON_ORIG` (env[1..3]) is plumbed end-to-end on the `PTR_RINV_ORIG` precedent — operator-agnostic env read on the raw path, `.with_common_origin([x,y,z])` builder → `ExecutionOptions` → plan on the safe path, with a finiteness validator (NaN/inf rejected, `None` defaults to `[0,0,0]`) enforced symmetrically on BOTH paths. A committed non-zero H2O/STO-3G oracle fixture (`build_h2o_sto3g_common_orig`) + raw↔plan round-trip test prove the slot reads `env[1..3]`; this fixture is the parity gate for moments (Phase 24) and GIAO (Phases 26/30). Validated in Phase 22: Gauge-Origin Env Slot (Gap A — `PTR_COMMON_ORIG`) (v1.4).

### Active

Milestone v1.4 — full libcint 6.1.3 family parity: implement every remaining unsupported integral family (~140) to byte-identity at atol=1e-12 under the vendor-gated oracle, across remaining 1st-derivatives, Hessian/higher-order derivatives, position/multipole moments, relativistic spin-operator integrals, GIAO/magnetic-property NMR integrals, and gauge/Breit–Gaunt 2e. Scoped requirements in REQUIREMENTS.md.

### Out of Scope

- Bitwise-identical reproduction of libcint internals - the project targets result compatibility, not implementation identity.
- Public GTG support - the design explicitly excludes GTG from initial GA because upstream marks it deprecated and incorrect.
- Reproducing the upstream Fortran wrapper - not part of the Rust library's public scope.
- Public asynchronous APIs - excluded from the initial design to keep execution and compatibility contracts tighter.

## Current Milestone: v1.4 Full libcint 6.1.3 Family Parity

**Goal:** Implement every remaining libcint 6.1.3 integral family (~140) to byte-identity at atol=1e-12 under the vendor-gated oracle, reaching complete libcint API parity.

**Target features:**
- Remaining 1st-derivative families (`int2e_ip2`, `int1e_ipnucip/ipkinip/ipovlpip`, `int3c1e_ip1`, `int3c1e_iprinv`, `int2c2e_ip1/ip2`, `int3c2e_ip2`) — reuse the Phase-21 nabla/`gout_ip1` machinery
- Hessian and higher-order derivatives (`int1e_ipip*`, `int2c2e_ipip1`, `int3c2e_ipip1/ipip2`, 4th-order `ipipipiprinv`) — extend nabla to 2nd+ order
- Position/multipole moment integrals (`int1e_r/rr/rrr/rrrr`, `r2/r4`, `z/zz`, `sp`, `p4`, plain `rinv`, `drinv`, `irp`) — position-operator G-tensor on Obara–Saika
- Relativistic spin-operator integrals (`spsp`, `spnucsp`, `sprinvsp`, `srsr`, `sigma`, `int2e_spsp1/srsr1/ssp*/sps*/vsp*`) — σ·p spin machinery + spinor/4-component path (R5/D-03 spinor-derivative prerequisite)
- GIAO/magnetic-property NMR integrals (`int1e_giao_*`, `int1e_cg_*`, `a01gp`, `ia01p`, `ig*`, `g1/gg1`, `govlp/gnuc`, `int2e_g*`) — gauge-origin + angular-momentum operators
- Gauge/Breit–Gaunt 2e (`int2e_gauge_r1/r2_*`) — relativistic 2e
- Reuse the established per-family pattern: register in `compiled_manifest.lock.json` with `component_rank` → implement kernel → vendor FFI + byte-identity oracle test → flip `oracle_covered`; extend the manifest-audit + oracle gates to cover every new family

## Context

The project is driven by `docs/design/cintx_detailed_design.md`, which defines an implementation-ready redesign for libcint in Rust. The workspace contains the multi-crate Rust layout (`crates/`, `xtask/`, `benches/`, `ci/`) plus a vendored upstream reference in `libcint-master/`, with the design document as the source of truth for scope and release gates. v1.0 is complete (6 phases, 30 plans): typed domain primitives, manifest, planner, runtime, three-layer API surface (safe Rust, raw compat, C ABI shim), CI governance gates, CubeCL/wgpu GPU execution path with stub kernels, and staging/fingerprint plumbing. v1.1 is complete (Phases 7-10): real integral kernels for all five base families (1e, 2e, 2c2e, 3c1e, 3c2e) with oracle parity against vendored libcint 6.1.3, Gaussian math infrastructure (#[cube] Boys, Rys, Obara-Saika), and cart-to-sph Condon-Shortley transforms. The compatibility target remains libcint 6.1.3. v1.2 extends coverage to the full API surface with unified tolerance.

## Constraints

- **Compatibility**: Target upstream libcint 6.1.3 result compatibility - the project must match upstream outputs closely enough to satisfy oracle comparison gates.
- **Architecture**: CubeCL is the primary compute backend - host CPU work stays limited to planning, validation, marshaling, and test/oracle glue.
- **API Surface**: Safe Rust API first, raw compatibility API second, optional C ABI shim third - this ordering drives module boundaries and migration strategy.
- **Error Handling**: Public library errors use `thiserror` v2, while CLI, xtask, benchmarks, and oracle harness code use `anyhow`.
- **Verification**: Full API coverage claims must be backed by the compiled manifest lock, feature-matrix CI, and helper/transform parity checks.
- **Artifacts**: Deliverables written to `/mnt/data` remain a mandatory part of the design and verification workflow.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Prioritize result compatibility over implementation compatibility | Users need libcint-equivalent outputs, not a line-by-line clone of upstream internals | Pending |
| Use a three-layer public surface (safe Rust, raw compat, optional C ABI) | This balances Rust ergonomics with migration and interoperability needs | Validated in Phase 3 |
| Use a generated compiled manifest lock as the API source of truth | Full API coverage must be mechanically auditable across feature profiles | Validated in Phase 1 |
| Standardize on a shared planner plus CubeCL executor | A single compute path simplifies optimization, memory policy, and verification | Validated through Phases 1-5; v1.1 replaces executor internals with direct CubeCL client API |
| Use CubeCL client API directly in executor internals | Direct buffer management (`client.create`/`client.read`/`ArrayArg`) removes need for RecordingExecutor wrapper; kernels use `#[cube(launch)]` | v1.1 — user-directed architectural decision |
| Configurable backend (wgpu + cpu; cuda/rocm/metal extensible) | Multi-backend support ensures testing on CPU and deployment on GPU; future backends require only runtime trait impl | v1.1 — Validated |
| Centralize fallible allocation and typed OOM errors | Safe stop on memory pressure is a non-negotiable design goal | Partially validated in Phase 1 through `WorkspaceAllocator`, `ChunkPlanner`, and typed runtime errors |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `$gsd-transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `$gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check - still the right priority?
3. Audit Out of Scope - reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-29 — Phase 22 complete (FND-01 gauge-origin `PTR_COMMON_ORIG` env slot + non-zero oracle fixture); v1.4 in progress*
