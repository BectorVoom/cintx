# Phase 14: Unstable-Source-API Families - Context

**Gathered:** 2026-04-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement origi, grids, Breit, origk, and ssc families behind the `unstable-source-api` feature gate with oracle parity at atol=1e-12. Nightly CI job validates 0 mismatches. Cart/sph Breit representations and grids base-family stabilization are out of scope.

</domain>

<decisions>
## Implementation Decisions

### Manifest & gating strategy
- **D-01:** Add a new `unstable-source` manifest profile alongside base/with-f12/with-4c1e. Source-only symbols only appear when this profile is active.
- **D-02:** The `unstable-source` profile is standalone only — no cross-product with f12/4c1e profiles. Total profiles: base, with-f12, with-4c1e, with-f12+with-4c1e, unstable-source (5 total).
- **D-03:** Source-only symbols are explicitly listed in the manifest generator (hardcoded ~18 symbols). No auto-detection from headers.
- **D-04:** Existing `unstable_source_api_enabled()` gate in `raw.rs` and `is_source_only()` descriptor check remain the runtime enforcement mechanism (established in Phase 3).

### Grids env plumbing
- **D-05:** Extend `ExecutionPlan` with `GridsEnvParams` (ngrids, ptr_grids offset) alongside existing `OperatorEnvParams`. Validator rejects grids symbols when NGRIDS=0 or PTR_GRIDS is invalid. Mirrors the F12 zeta plumbing pattern from Phase 13.
- **D-06:** NGRIDS is an output dimension multiplier. The planner multiplies standard dims by NGRIDS to compute output buffer size. Kernel loops over grid points. Output shape: (ncomp * NGRIDS * di * dj) matching libcint's `g1e_grids.c` behavior.

### Breit spinor-only scope
- **D-07:** Breit integrals are spinor-only in Phase 14. Only `int2e_breit_r1p2_spinor` and `int2e_breit_r2p2_spinor` are implemented. Cart and sph representations are not implemented (manifested as `UnsupportedRepresentation` if requested).
- **D-08:** Breit kernel uses a single composite kernel (Gaunt + gauge computed and summed internally). Matches `breit.c`'s `_int2e_breit_drv` pattern. Single launch, single output buffer.

### Oracle fixture strategy
- **D-09:** Single test file `unstable_source_parity.rs` with per-family test functions, all gated behind `#[cfg(feature = "unstable-source-api")]`. ~18 symbols don't warrant separate files.
- **D-10:** Reuse existing H2O/STO-3G fixture molecule for all unstable families. Grids family adds grid point coordinates to env but uses the same molecule. Consistent with all prior oracle tests.
- **D-11:** Nightly CI job is an extra job (`unstable_source_oracle`) in the existing CI workflow, gated on schedule/nightly trigger. Reuses existing runner config and artifact paths per Phase 4 conventions.

### Claude's Discretion
- Internal module layout for new kernel files (whether origi/origk share a file or get separate modules)
- Exact `GridsEnvParams` field names and validation logic
- How Breit Gaunt+gauge composition routes through existing 2e infrastructure vs new dispatch arm
- Order of family implementation across plans
- Grid coordinate fixture values for oracle tests

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Design & requirements
- `docs/design/cintx_detailed_design.md` — Master design document; source-only/optional family definitions, NGRIDS/PTR_GRIDS env contract (line 302), feature flag definitions (section 10.2)
- `.planning/REQUIREMENTS.md` — USRC-01 through USRC-06 requirement definitions
- `.planning/ROADMAP.md` — Phase 14 goal and success criteria

### Prior phase context (gating infrastructure)
- `.planning/phases/03-safe-surface-c-abi-shim-optional-families/03-CONTEXT.md` — D-13/D-14/D-15/D-16: unstable namespace policy, C ABI stable-only, promotion criteria, feature gate enforcement
- `.planning/phases/13-f12-stg-yp-kernels/13-CONTEXT.md` — D-05/D-06/D-11/D-12: OperatorEnvParams pattern, kernel dispatch pattern (model for grids/Breit plumbing)

### Upstream libcint reference (vendored)
- `libcint-master/src/cint1e_a.c` — origi family: int1e_r2_origi, int1e_r4_origi, int1e_r2_origi_ip2, int1e_r4_origi_ip2 implementations and gout functions
- `libcint-master/src/cint1e_grids.c` — grids base integral: int1e_grids and NGRIDS/PTR_GRIDS env handling
- `libcint-master/src/g1e_grids.c` — grids G-tensor setup: grid-point-dependent center displacement, output dimension multiplier
- `libcint-master/include/cint_funcs.h` — grids derivative symbols: int1e_grids_ip, int1e_grids_ipvip, int1e_grids_spvsp, int1e_grids_ipip
- `libcint-master/src/breit.c` — Breit composite kernel: _int2e_breit_drv, Gaunt+gauge composition, int2e_breit_r1p2 and int2e_breit_r2p2
- `libcint-master/src/cint3c1e_a.c` — origk family: int3c1e_r2/r4/r6_origk, int3c1e_ip1_r2/r4/r6_origk implementations
- `libcint-master/src/cint3c2e.c` — ssc family: int3c2e_ssc implementation

### Existing runtime infrastructure (to extend)
- `crates/cintx-compat/src/raw.rs` — `unstable_source_api_enabled()`, `is_source_only()` checks, `enforce_safe_facade_policy_gate()`
- `crates/cintx-runtime/src/workspace.rs` — ExecutionPlan struct (extend with GridsEnvParams)
- `crates/cintx-runtime/src/validator.rs` — Add grids env validation (NGRIDS>0, PTR_GRIDS valid)
- `crates/cintx-cubecl/src/kernels/mod.rs` — `resolve_family_name()` dispatch (add unstable family arms)
- `crates/cintx-ops/generated/compiled_manifest.lock.json` — Manifest lock to regenerate with unstable-source profile

### Oracle infrastructure
- `crates/cintx-oracle/tests/f12_oracle_parity.rs` — Structural reference for per-profile oracle parity test file
- `crates/cintx-oracle/src/fixtures.rs` — Fixture generation to extend with grids env params
- `crates/cintx-oracle/src/vendor_ffi.rs` — Vendored libcint FFI bindings to extend with unstable symbols

### CI
- `.github/workflows/` — Existing CI workflow to extend with nightly unstable_source_oracle job

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `one_electron.rs` kernel: 1e VRR/HRR + gout infrastructure — origi kernels are 1e variants with origin-displaced r^n operators
- `center_3c1e.rs` kernel: 3c1e infrastructure — origk kernels are 3c1e variants with k-origin-displaced r^n
- `two_electron.rs` kernel: 2e Rys quadrature — Breit extends 2e with Gaunt+gauge composition
- `three_center_2e.rs` kernel: 3c2e infrastructure — ssc extends 3c2e with spin-spin contact operator
- `OperatorEnvParams` in workspace.rs: F12 zeta plumbing pattern — model for GridsEnvParams
- `f12.rs` kernel dispatch: resolve_family_name + operator-specific launch — model for unstable family dispatch
- `vendor_ffi.rs`: Existing vendored FFI pattern for oracle comparison

### Established Patterns
- Host wrapper + `#[cube]` pair: `*_host()` counterpart for all math functions (Phase 8 convention)
- Feature-gated kernel dispatch: `#[cfg(feature = "with-f12")]` in `kernels/mod.rs` — mirror for `unstable-source-api`
- PTR_ENV_START-aligned env layout: env user data starts at offset 20; established in Phase 10
- Oracle fixture generation with profile-scoped APIs (Phase 4 convention)
- Validator rejects with typed errors before kernel launch (zeta=0 pattern from Phase 13)

### Integration Points
- `kernels/mod.rs` `resolve_family_name()` — add unstable family dispatch arms behind `#[cfg(feature = "unstable-source-api")]`
- `cintx-runtime/src/validator.rs` — add GridsEnvParams validation for NGRIDS/PTR_GRIDS
- `cintx-runtime/src/workspace.rs` — extend ExecutionPlan with GridsEnvParams
- `cintx-ops` manifest generator — add explicit source-only symbol list and unstable-source profile
- `oracle/vendor_ffi.rs` — add bindgen declarations for unstable family C functions
- CI workflow — add nightly-gated unstable_source_oracle job

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches. Vendored libcint source files are the authoritative references for algorithm behavior per family.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 14-unstable-source-api-families*
*Context gathered: 2026-04-05*
