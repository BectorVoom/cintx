# Phase 9: 1e Real Kernel and Cart-to-Sph Transform - Context

**Gathered:** 2026-04-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement real overlap, kinetic, and nuclear attraction kernels that produce libcint-compatible spherical outputs, plus correct Condon-Shortley cart-to-sph transform for all angular momenta up to g-function (l=4). Validates the entire compute pipeline end-to-end: pair setup, G-tensor fill, GTO contraction, cart-to-sph, and oracle comparison. Only 1e families are in scope; 2e/2c2e/3c1e/3c2e kernels belong to Phase 10.

</domain>

<decisions>
## Implementation Decisions

### 1e Kernel Structure
- **D-01:** Single `launch_one_electron` entry point dispatches to a shared G-tensor fill (VRR/HRR for gx, gy, gz arrays) followed by per-operator post-processing. Matches libcint's g1e.c architecture.
- **D-02:** Operator switch inside the kernel: overlap = direct contraction, kinetic = nabla-squared post-process, nuclear = Boys-weighted sum over atom centers.
- **D-03:** The shared G-fill uses the Phase 8 `vrr_step()` and `hrr_step()` #[cube] functions from `math/obara_saika.rs`, plus `compute_pdata()` from `math/pdata.rs`.

### Cart-to-Sph Transform
- **D-04:** Cart-to-sph is implemented as host-side Rust code (not a #[cube] function). Kernel writes cartesian components to staging, `client.read()` brings data to host, then `cart_to_sph_1e()` applies Condon-Shortley matrix per shell pair before writing to `io.staging_output()`.
- **D-05:** Condon-Shortley coefficients are extracted from libcint's `cart2sph.c` `g_trans_cart2sph[]` array for l=0..4. Stored as const arrays in `transform/c2s.rs`.
- **D-06:** GPU-side c2s is a future optimization (deferred). Host-side is sufficient for correctness validation and oracle parity.

### Nuclear Attraction Operator
- **D-07:** Nuclear attraction loops over all atom centers C inside the kernel. For each atom: compute Boys F_m(t) where t = aij * |P-C|^2, fill G-tensor with nuclear-weighted VRR using PC displacement, accumulate Z_c * contracted result.
- **D-08:** Atom coordinates and charges are passed as input arrays to the kernel function. Uses Phase 8 `boys_gamma_inc()` from `math/boys.rs`.

### Validation Strategy
- **D-09:** Cart-to-sph coefficients validated for all angular momenta l=0..4 via dedicated unit tests comparing coefficient matrices against libcint `cart2sph.c` reference values. Covers: l=0 (1x1), l=1 (3x3), l=2 (6x5), l=3 (10x7), l=4 (15x9).
- **D-10:** End-to-end oracle parity test: H2O STO-3G for int1e_ovlp_sph, int1e_kin_sph, int1e_nuc_sph. Tolerances: atol 1e-11 / rtol 1e-9 (per success criteria).
- **D-11:** Oracle parity verified per family as each kernel lands (VERI-05), not deferred to end.

### Carried Forward
- **D-12:** Host wrapper + #[cube] pair pattern for math functions (Phase 8). Integration tests use host-side wrappers, not CubeCL CPU backend launch (avoids cond_br MLIR limitation).
- **D-13:** CPU backend is the primary oracle target (Phase 7 D-03). Tests run under `--features cpu`.
- **D-14:** Buffer lifecycle lives inside kernel family modules, not centralized (Phase 7 D-07).
- **D-15:** Staging buffer sizing already accounts for spherical vs cartesian via `ao_per_shell()` in `cintx_core/shell.rs` (Cart: (l+1)(l+2)/2, Spheric: 2l+1).

### Claude's Discretion
- Internal G-tensor array sizing and indexing strategy (flat vs 3-component gx/gy/gz)
- Exact GTO contraction loop structure
- How operator ID is extracted from `SpecializationKey` or `ExecutionPlan` to select the operator variant
- Host-side c2s buffer management (in-place vs separate cart/sph buffers)
- Test fixture design for c2s coefficient validation

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### libcint 1e Reference Source
- `libcint-master/src/g1e.c` -- 1e G-tensor fill: VRR/HRR for overlap, kinetic, nuclear. Lines 125-184 for CINTg1e_ovlp, full file for nuclear variant
- `libcint-master/src/cart2sph.c` -- Condon-Shortley transform coefficients: `g_trans_cart2sph[]` array (lines 21-300+), transform functions
- `libcint-master/src/cart2sph.h` -- c2s function signatures: `c2s_sph_1e()` interface
- `libcint-master/src/cint_config.h` -- Compile-time constants (MMAX, angular momentum limits)
- `libcint-master/src/cint1e.c` -- 1e integral driver: shows how g1e fill + c2s + contraction are composed

### Phase 8 Math Primitives (dependencies)
- `crates/cintx-cubecl/src/math/boys.rs` -- Boys function `boys_gamma_inc()` #[cube] + host wrapper
- `crates/cintx-cubecl/src/math/pdata.rs` -- PairData struct + `compute_pdata()` #[cube]
- `crates/cintx-cubecl/src/math/obara_saika.rs` -- `vrr_step()` and `hrr_step()` #[cube] functions
- `crates/cintx-cubecl/src/math/rys.rs` -- Rys quadrature (consumed by nuclear if needed)

### Kernel Infrastructure (files to modify)
- `crates/cintx-cubecl/src/kernels/one_electron.rs` -- Current 1e kernel stub to replace with real implementation
- `crates/cintx-cubecl/src/kernels/mod.rs` -- FamilyLaunchFn dispatch, `resolve_family_name()`
- `crates/cintx-cubecl/src/transform/c2s.rs` -- Current c2s stub to replace with real Condon-Shortley transform
- `crates/cintx-cubecl/src/transform/mod.rs` -- `apply_representation_transform()` dispatcher
- `crates/cintx-cubecl/src/executor.rs` -- Executor flow: kernel launch + transform + staging output

### Runtime Infrastructure
- `crates/cintx-runtime/src/planner.rs` -- Output layout sizing, `build_output_layout()`, component count
- `crates/cintx-core/src/shell.rs` -- `ao_per_shell()` for Cart/Spheric component counts (lines 91-104)

### Prior Phase Context
- `.planning/phases/07-executor-infrastructure-rewrite/07-CONTEXT.md` -- Backend enum, CPU backend, f64 strategy, buffer lifecycle decisions
- `.planning/phases/08-gaussian-primitive-infrastructure-and-boys-function/08-CONTEXT.md` -- Math primitive decisions, #[cube] patterns, validation approach

### Oracle Harness
- `crates/cintx-oracle/` -- Oracle comparison infrastructure for generating libcint reference values

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `math::boys::boys_gamma_inc()` + `boys_gamma_inc_host()` -- Boys function for nuclear attraction operator
- `math::pdata::PairData` + `compute_pdata()` -- Gaussian pair setup (zeta, center_p, aij2, etc.)
- `math::obara_saika::vrr_step()` + `hrr_step()` -- Obara-Saika recurrence for G-tensor fill
- `crates/cintx-oracle/` -- Oracle harness for generating reference values from vendored libcint

### Established Patterns
- Host wrapper + #[cube] pair: every math function has `*_host()` callable from tests without GPU context
- u32 loop counters with `as usize` at Array index sites (Phase 8 established CubeCL pattern)
- Integration tests use host-side wrappers only, not CubeCL CPU backend launch (cond_br MLIR workaround)
- `FamilyLaunchFn` signature: `fn(&ResolvedBackend, &ExecutionPlan, &SpecializationKey, &mut [f64]) -> Result<ExecutionStats>`

### Integration Points
- `kernels::resolve_family_name()` maps "1e" to `one_electron::launch_one_electron` -- entry point preserved
- `executor.rs` lines 201-206: kernel launch -> `transform::apply_representation_transform()` -> staging output
- `transform/c2s.rs` replaces stub with real Condon-Shortley matrix application
- Staging buffer pre-sized by planner based on `Shell::ao_per_shell()` for target representation

</code_context>

<specifics>
## Specific Ideas

- Follow libcint's g1e.c architecture: shared G-fill with three-axis (gx, gy, gz) arrays, then operator-specific post-processing
- Nuclear attraction: loop over atom centers inside kernel, using Boys function + VRR with P-C displacement
- C2s coefficients extracted directly from libcint's `cart2sph.c` `g_trans_cart2sph[]` constant array

</specifics>

<deferred>
## Deferred Ideas

- GPU-side #[cube] cart-to-sph transform -- future optimization after correctness proven on host
- Higher angular momentum end-to-end tests (cc-pVTZ/cc-pVQZ with d/f shells) -- Phase 10 will exercise these via 2e test cases
- Workgroup sizing for kernel launch -- post-v1.1 optimization

</deferred>

---

*Phase: 09-1e-real-kernel-and-cart-to-sph-transform*
*Context gathered: 2026-04-03*
