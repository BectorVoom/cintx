# Phase 10: 2e, 2c2e, 3c1e, 3c2e Real Kernels and Oracle Gate Closure - Context

**Gathered:** 2026-04-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement all remaining integral family kernels (2e, 2c2e, 3c1e, 3c2e) to replace current stubs that return zeros, then close the oracle parity gate across all five base families for v1.1 completion. Also resolves two v1.0 human UAT items: eval_raw() returning non-zero values and C ABI shim returning non-error status. Only real kernel compute and oracle gate closure are in scope; GPU-side #[cube] kernel optimization, screening/batching, and higher-order families are deferred.

</domain>

<decisions>
## Implementation Decisions

### Kernel Architecture
- **D-01:** Each family gets its own G-fill implementation mirroring its libcint counterpart (g2e.c, g2c2e.c, g3c1e.c, g3c2e.c). Share math primitives (Rys, VRR, HRR, Boys) but not the orchestration logic.
- **D-02:** Support full angular momentum up to g-function (l=4) from the start for all families. The Phase 8 math primitives already handle arbitrary l.
- **D-03:** All kernels are host-side pure Rust computation using *_host() math wrappers. Same pattern as Phase 9 one_electron.rs. No CubeCL #[cube] GPU kernels in this phase (avoids cond_br MLIR issues).

### Oracle Parity Strategy
- **D-04:** Build vendored libcint-master/ with cc crate, call upstream C functions via bindgen FFI, compare outputs numerically. Use the existing cintx-oracle/ harness infrastructure.
- **D-05:** Multiple test molecules and basis sets: H2O + H2 + CH4 across STO-3G and cc-pVDZ. Provides both fast smoke tests (STO-3G, s/p shells) and full angular momentum coverage (cc-pVDZ, s/p/d shells).
- **D-06:** Tolerances per success criteria: 2e atol 1e-12 / rtol 1e-10, 2c2e atol 1e-9, 3c1e atol 1e-7, 3c2e atol 1e-9. Oracle parity verified per family as each kernel lands (VERI-05).

### v1.0 UAT Items
- **D-07:** Both UAT items verified via automated integration tests in CI: (1) eval_raw() on a real basis set (H2O STO-3G int1e_ovlp_sph) asserts non-zero output, (2) C ABI cintrs_eval() test under CPU backend asserts status == 0.

### Implementation Order
- **D-08:** Ascending complexity: 2c2e -> 3c1e -> 3c2e -> 2e. Each family is a separate plan with its own oracle parity test. A 5th plan handles oracle gate closure across all families plus UAT item resolution.
- **D-09:** 2c2e (2-center Coulomb) is closest to nuclear attraction (Rys + 2 centers). 3c1e adds a third center without Rys. 3c2e combines 3 centers with Rys. 2e (full ERI) is the most complex with 4 shells and full Rys quadrature.

### Carried Forward
- **D-10:** Host wrapper + #[cube] pair pattern for math functions (Phase 8). Integration tests use host-side wrappers only.
- **D-11:** CPU backend is the primary oracle target (Phase 7/9). Tests run under `--features cpu`.
- **D-12:** C2S transform is host-side Rust (Phase 9 D-04/D-06). Kernel writes cartesian to staging, host applies Condon-Shortley transform.
- **D-13:** Buffer lifecycle lives inside kernel family modules, not centralized (Phase 7 D-07).

### Claude's Discretion
- Internal G-tensor array sizing and indexing strategy per family (following libcint source as guide)
- Exact Rys quadrature integration into each family's G-fill (root iteration, weight application)
- Test fixture design for per-family oracle comparison
- How to structure the oracle gate closure CI check (xtask command vs test)
- Plan boundaries within each family (e.g., whether contraction + c2s is separate from G-fill)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### libcint 2e+ Reference Source
- `libcint-master/src/g2e.c` -- 4-center 2e G-tensor fill: Rys quadrature, VRR/HRR for ERI
- `libcint-master/src/cint2e.c` -- 2e integral driver: shows how g2e fill + c2s + contraction are composed
- `libcint-master/src/g2c2e.c` -- 2-center 2-electron G-tensor fill
- `libcint-master/src/cint2c2e.c` -- 2c2e integral driver
- `libcint-master/src/g3c1e.c` -- 3-center 1-electron G-tensor fill
- `libcint-master/src/cint3c1e.c` -- 3c1e integral driver
- `libcint-master/src/g3c2e.c` -- 3-center 2-electron G-tensor fill
- `libcint-master/src/cint3c2e.c` -- 3c2e integral driver
- `libcint-master/src/cart2sph.c` -- Condon-Shortley transform (c2s_sph_2e, c2s_sph_2c2e, c2s_sph_3c1e, c2s_sph_3c2e)
- `libcint-master/src/cart2sph.h` -- c2s function signatures for all families
- `libcint-master/src/find_roots.c` -- Rys root finding implementation
- `libcint-master/src/fmt.c` -- Boys function / Fm(t) reference

### Phase 9 Implementation (pattern reference)
- `crates/cintx-cubecl/src/kernels/one_electron.rs` -- Working 1e kernel: G-fill + contraction + c2s pipeline pattern to follow
- `crates/cintx-cubecl/src/transform/c2s.rs` -- Cart-to-sph transform with Condon-Shortley coefficients

### Phase 8 Math Primitives (dependencies)
- `crates/cintx-cubecl/src/math/boys.rs` -- Boys function `boys_gamma_inc()` + host wrapper
- `crates/cintx-cubecl/src/math/pdata.rs` -- PairData struct + `compute_pdata()` host wrapper
- `crates/cintx-cubecl/src/math/obara_saika.rs` -- VRR/HRR step functions + `vrr_2e_step_host()` for root-dependent recurrence
- `crates/cintx-cubecl/src/math/rys.rs` -- Rys quadrature roots and weights

### Kernel Stubs (files to replace)
- `crates/cintx-cubecl/src/kernels/two_electron.rs` -- Current 2e stub returning zeros
- `crates/cintx-cubecl/src/kernels/center_2c2e.rs` -- Current 2c2e stub returning zeros
- `crates/cintx-cubecl/src/kernels/center_3c1e.rs` -- Current 3c1e stub returning zeros
- `crates/cintx-cubecl/src/kernels/center_3c2e.rs` -- Current 3c2e stub returning zeros
- `crates/cintx-cubecl/src/kernels/mod.rs` -- FamilyLaunchFn dispatch, resolve_family_name()

### Oracle Harness
- `crates/cintx-oracle/` -- Oracle comparison infrastructure for vendored libcint reference values

### Prior Phase Context
- `.planning/phases/09-1e-real-kernel-and-cart-to-sph-transform/09-CONTEXT.md` -- 1e kernel decisions, c2s transform, validation approach
- `.planning/phases/08-gaussian-primitive-infrastructure-and-boys-function/08-CONTEXT.md` -- Math primitive decisions, #[cube] patterns
- `.planning/phases/07-executor-infrastructure-rewrite/07-CONTEXT.md` -- Backend enum, CPU backend, buffer lifecycle

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `one_electron.rs`: Complete working kernel showing G-fill + contraction + c2s pipeline — the template for all four families
- `math::rys::rys_root1_host()`, `rys_root2_host()`: Rys quadrature roots/weights for 2e-type families
- `math::obara_saika::vrr_2e_step_host()`: Root-dependent VRR for 2e-type G-tensor fill (used for nuclear in 1e)
- `math::boys::boys_gamma_inc_host()`: Boys function needed by all Rys-based families
- `math::pdata::compute_pdata_host()`: Gaussian pair data (zeta, center_p, screening)
- `transform::c2s::cart_to_sph_1e()`: Existing c2s for 1e; needs c2s_2e/2c2e/3c1e/3c2e variants (different index layouts)

### Established Patterns
- Host-side kernel: pure Rust computation using *_host() math wrappers, no CubeCL GPU launch
- G-tensor: flat array `[gx | gy | gz]` with `g_per_axis = (nmax+1)*(lj+1)` (1e pattern; 2e extends to 4 indices)
- `common_fac_sp(l)`: s/p normalization factor applied before c2s transform
- `cart_comps(l)`: Cartesian component enumeration matching libcint CINTcart_comp ordering
- FamilyLaunchFn signature: `fn(&ResolvedBackend, &ExecutionPlan, &SpecializationKey, &mut [f64]) -> Result<ExecutionStats>`

### Integration Points
- `kernels::resolve_family_name()`: Maps "2e"/"2c2e"/"3c1e"/"3c2e" to their launch functions
- `executor.rs`: kernel launch -> `transform::apply_representation_transform()` -> staging output
- `transform/c2s.rs`: Needs c2s variants for 2e (4-index), 2c2e (2-index), 3c1e (3-index), 3c2e (3-index)
- Oracle harness in `cintx-oracle/`: vendored libcint build + comparison infrastructure

</code_context>

<specifics>
## Specific Ideas

- Follow each family's libcint source (g2e.c, g2c2e.c, g3c1e.c, g3c2e.c) as closely as practical for the G-tensor fill, diverging only where Rust idioms improve clarity
- 2c2e implementation should leverage similarity to nuclear attraction (both use Rys + 2 centers) as a starting point
- Multiple molecules (H2O, H2, CH4) across STO-3G and cc-pVDZ provide layered validation coverage
- Oracle parity verified per family as each kernel lands — not batched at the end

</specifics>

<deferred>
## Deferred Ideas

- GPU-side #[cube] kernels for all families -- future optimization after correctness proven on host
- Screening/batching optimizations -- performance work after oracle parity closes
- Higher angular momentum (l>=5, h-functions) -- register pressure risk, defer until g-function validated
- Spinor representation kernels -- differentiator, deferred to v1.2
- F12/STG/YP optional family kernels -- feature-gated families, defer to v1.2

</deferred>

---

*Phase: 10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure*
*Context gathered: 2026-04-03*
