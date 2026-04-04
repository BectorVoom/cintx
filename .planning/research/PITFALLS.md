# Pitfalls Research

**Domain:** Full API parity for existing integral library (F12/STG/YP families, spinor representations, extended 4c1e, helper/transform/wrapper APIs, unified oracle tolerance) — cintx v1.2
**Researched:** 2026-04-04
**Supersedes:** v1.1 pitfalls document (2026-04-02), which covered CubeCL client API migration and GPU kernel foundations. This document carries forward the pitfalls still relevant to v1.2 and adds all pitfalls specific to the new milestone scope.
**Confidence:** HIGH for pitfalls derived from direct upstream source inspection (libcint-master/src/); HIGH for tolerance arithmetic backed by oracle compare.rs and design doc section 13.8; MEDIUM for STG quadrature precision claims (based on algorithm analysis, not empirical sweep); LOW for CUDA/Metal backend spinor behavior.

---

## Critical Pitfalls

### Pitfall 1: Forcing atol=1e-12 onto 3c1e is a 5-orders-of-magnitude mismatch with upstream tolerance

**What goes wrong:**
The user mandate is atol=1e-12 for ALL families. The upstream design document (section 13.8) and the current oracle compare.rs both accept atol=1e-7 for high-order 3c1e integrals. The gap is five orders of magnitude. If the unified tolerance is applied before understanding which 3c1e operators can actually achieve 1e-12, every high-order 3c1e oracle run will fail and implementation will stall on false gates.

**Why it happens:**
3c1e integrals use Rys quadrature with multiple roots. The number of roots scales as `ceil((li+lj+lk)/2) + 1`. For high angular momentum (e.g., d/f/g shells), the quadrature involves 4-8 roots per primitive triple. Upstream uses double-precision Rys weights and abscissae from tabulated polynomial fits (roots_xw.dat). The polynomial evaluation itself accumulates rounding errors at the 1e-8 to 1e-9 level for large `x` arguments. The upstream testsuite uses atol=1e-7 explicitly because that is the empirically established accuracy floor for high-order 3c1e.

The CubeCL GPU path may introduce additional floating-point reordering versus the upstream C path due to different instruction fusion and fused multiply-add behavior, widening the gap further. GPU implementations of Rys quadrature typically show 2-5 ULP divergence relative to sequential CPU evaluation.

**How to avoid:**
Do not adopt a single flat 1e-12 atol for 3c1e without empirical evidence. The correct approach is: (a) implement the 3c1e kernels; (b) run the oracle against the full shell-combination matrix; (c) measure the actual max abs error for each operator; (d) set per-operator tolerances based on evidence. For low-order 3c1e operators (s/p shells, `nroots <= 2`), 1e-12 may be achievable. For high-order operators with derivatives, 1e-7 is the proven floor.

If the project is committed to reporting a single unified tolerance, the correct unified value is the maximum across all families — which would be 1e-6 for spinor/4c1e/F12 per upstream evidence — not 1e-12 tightened down. Alternatively, report per-family tolerances and use 1e-12 only for families where it is achievable (2e plain, 1e basic).

**Warning signs:**
- Oracle CI gate fails on 3c1e after tolerance change to 1e-12 with errors in the 1e-9 to 1e-8 range
- Oracle failures only appear on higher angular momentum shell combinations (d,f shells)
- Failures disappear when reverting 3c1e tolerance to 1e-7

**Phase to address:**
Tolerance audit phase (before implementing F12/4c1e). Run the existing 3c1e oracle with reduced tolerance and measure empirical error distribution. Document the per-family achievable floor before writing any gates that use 1e-12 universally.

---

### Pitfall 2: The spinor transform stub in c2spinor.rs is not a real cart-to-spinor transform

**What goes wrong:**
The current implementation of `cart_to_spinor_interleaved_staging` in `crates/cintx-cubecl/src/transform/c2spinor.rs` computes `amplitude = (|re| + |im|) * 0.5` and sets `pair[0] = amplitude, pair[1] = -amplitude`. This is not a valid cartesian-to-spinor transformation. It is a placeholder that averages absolute values and negates the imaginary component. The unit tests pass because they only verify the stub behavior, not correctness. Any spinor oracle comparison will fail completely with this implementation in place.

**Why it happens:**
The real cart-to-spinor transform requires applying the Clebsch-Gordan coupling coefficients to convert from `(l, m)` basis to `(j=l+1/2, mj)` and `(j=l-1/2, mj)` spinor bases. The kappa quantum number selects between the two j values: `kappa < 0` gives `j = l + 1/2` (2l+2 components), `kappa > 0` gives `j = l - 1/2` (2l components), `kappa = 0` is used by libcint for the sum of both. The transformation matrix is l-dependent and complex-valued. The stub treats the problem as a simple numerical rescaling, which shares none of this structure.

The upstream function `CINTc2s_ket_spinor_sf1` in `include/cint.h.in:283-284` applies the actual coupling matrix. The `src/cart2sph.c` file contains the spinor coefficient tables.

**How to avoid:**
Implement the full coupling coefficient tables and the l-dependent transformation matrix before any spinor integral oracle comparison. The existing `CINTlen_spinor(l, kappa)` helper correctly computes `4l+2` (kappa=0), `2l+2` (kappa<0), `2l` (kappa>0) — use this to size the output correctly. The real transform must be applied per shell (not per element) and must preserve the real/imag interleaving that compat expects.

Treat the current stub as a `todo!()` equivalent. Do not write oracle fixtures for spinor operators until the transform is correct.

**Warning signs:**
- Spinor oracle comparisons show max abs errors orders of magnitude above tolerance
- Spinor outputs are all real (im=0) or all imaginary (re=0) instead of complex
- The output size matches the expected `spinor_len * nctr * 2` but values are wrong
- Tests that only check `staging.len() % 2 == 0` pass while correctness tests fail

**Phase to address:**
Before any spinor kernel oracle work. The c2spinor.rs module must be completely rewritten with real Clebsch-Gordan coefficients before spinor oracle fixtures are even authored.

---

### Pitfall 3: PTR_F12_ZETA=9 and PTR_RANGE_OMEGA=8 must be populated in env before F12/STG/YP calls

**What goes wrong:**
STG integrals (Slater-type geminals) read `env[PTR_F12_ZETA]` = `env[9]` to get the `zeta` parameter for the Yukawa/STG kernel. YP integrals read from both `env[PTR_RANGE_OMEGA]` = `env[8]` (for the limit) and `env[PTR_F12_ZETA]` = `env[9]`. If these slots are zero (the default), the STG kernel falls back to a Rys quadrature path (zeta == 0.0 branch in `CINTg0_2e_stg`), which silently produces wrong integrals instead of failing.

The test harness in `libcint-master/testsuite/test_int2e_f12_etc.py` sets `mol._env[8] = 1e3`, `mol._env[9] = 1e-3`, `mol._env[10] = 1e-4` before running F12 comparisons. Any oracle fixture that omits these assignments will compare STG integrals computed with zeta=0 (plain Coulomb fallback) against reference integrals computed with the correct zeta — producing perfect agreement only when zeta is small and catastrophic failure otherwise.

**Why it happens:**
The env array has fixed-position parameters (PTR_ENV_START=20 applies only to basis data, not to the special parameters in slots 0-19). F12 parameters live in slots 8-10. If the fixture builder derives env solely from atm/bas chemistry and does not set the F12 slots, they remain zero-initialized. The kernel silently uses the Coulomb path.

**How to avoid:**
All F12/STG/YP oracle fixtures must explicitly set `env[8]` (PTR_RANGE_OMEGA) and `env[9]` (PTR_F12_ZETA) to physically meaningful non-zero values. The fixture generator must expose `omega` and `zeta` as required parameters for F12 family fixtures. The validator in `cintx-compat` must reject F12/STG/YP calls where `env[9] == 0.0` with a typed error rather than silently proceeding.

Cross-check: the STG roots function `CINTstg_roots` is only called when `zeta > 0`. When zeta == 0 the code calls `CINTrys_roots`, which is the plain Coulomb path. Verify this branch in the CubeCL implementation.

**Warning signs:**
- F12/STG oracle comparisons pass trivially with atol=1e-12 (too-good result is the red flag)
- STG and plain 2e integrals produce identical output for the same shell combination
- F12 oracle fails with large errors only when zeta is changed to a nonzero value

**Phase to address:**
F12 fixture authoring phase. Add a validator that checks `env[PTR_F12_ZETA] != 0.0` for STG/YP families before dispatch.

---

### Pitfall 4: The 4c1e spinor path is explicitly unimplemented in upstream and must be rejected

**What goes wrong:**
`int4c1e_spinor` in `libcint-master/src/cint4c1e.c:349-353` contains only `fprintf(stderr, "int4c1e_spinor not implemented\n"); return 0;`. This means the upstream oracle returns zero for all 4c1e spinor calls. If cintx implements 4c1e spinor integrals, the oracle will report zero for all reference values and the comparison will either trivially pass (if cintx also returns zero) or fail with nonzero values for cintx. Neither outcome is useful.

**Why it happens:**
Upstream explicitly did not implement 4c1e spinor due to the known bugs in 4c1e itself. The spinor representation adds complexity (complex output, kappa-dependent dimensions) on top of an already-buggy integral. The CMakeLists.txt message "Note there are bugs in 4c1e integral functions" applies to the cart/sph path; the spinor path is simply absent.

**How to avoid:**
The 4c1e bug envelope defined in the design (section 3.11.2) must explicitly exclude spinor representation. The capability classifier must return `UnsupportedApi` for any 4c1e spinor request, not just for inputs outside the `Validated4C1E` envelope. This is in addition to the existing checks for `max(l)>4` and non-natural dims. Add a specific test verifying that 4c1e + Spinor always returns `UnsupportedApi` regardless of angular momentum.

The release gate check (design section 16.4) must include: "4c1e spinor requests return UnsupportedApi with reason citing upstream unimplemented path."

**Warning signs:**
- 4c1e spinor oracle comparisons trivially pass (both sides zero)
- The `Validated4C1E` classifier does not check representation before checking l-value

**Phase to address:**
4c1e capability classifier implementation. This check must be in place before any oracle fixture generation for 4c1e.

---

### Pitfall 5: STG roots use a tabulated polynomial approximation with hard domain limits

**What goes wrong:**
`CINTstg_roots` in `libcint-master/src/stg_roots.c` uses Clenshaw recurrence on precomputed polynomial coefficients loaded from `roots_xw.dat`. The function clamps the argument: `if (t > 19682.99) t = 19682.99;`. This means that for large inter-shell separations (large `ta = a0 * r12^2`) or small exponents, the quadrature argument is clamped rather than extrapolated, and the resulting roots/weights are incorrect.

Additionally, the function uses two-dimensional interpolation in (t, u) space where `u = zeta^2 / (4 * a0)`. For very small `a0` (diffuse basis functions) with large `zeta`, `u` becomes large, and the polynomial approximation loses accuracy. The upstream tolerance for F12 integrals is 1e-6 (atol) and 1e-4 (rtol) precisely because of this domain limitation.

**Why it happens:**
The STG roots problem does not have the same mathematical structure as the Boys function (which can be computed analytically for large arguments). The roots of the STG modified Rys polynomial must be found numerically, and the tabulated approximation is accurate only over a bounded domain. The clamping at t=19682.99 is a deliberate numerical stability choice in upstream.

For a CubeCL kernel implementing STG integrals, replicating this clamping behavior is required for oracle compatibility. If the GPU implementation uses a different domain handling (e.g., falls back to asymptotic approximation or panics on out-of-range), results will diverge from upstream in exactly the cases where upstream uses the clamped path.

**How to avoid:**
The CubeCL STG quadrature kernel must replicate the `t = min(t, 19682.99)` clamp before table lookup. The fixture matrix must include shell combinations that push the t argument above the clamp threshold (large separations + tight exponents) to verify the clamping path matches upstream.

The F12 oracle tolerance must be kept at atol=1e-6, rtol=1e-4 (upstream values). Attempting atol=1e-12 for F12/STG/YP is not achievable given the polynomial approximation floor.

**Warning signs:**
- F12 oracle failures cluster at shell combinations involving diffuse functions (small exponents)
- Errors appear for large inter-center distances but not for close pairs
- STG root lookup shows t > 19682 for the failing cases

**Phase to address:**
F12 kernel implementation phase. Include a specific fixture case with t > 19682 argument.

---

### Pitfall 6: YP integrals require both PTR_RANGE_OMEGA and PTR_F12_ZETA and handle zeta=0 differently from STG

**What goes wrong:**
YP (Yukawa potential) integrals use `CINTg0_2e_yp` which initializes `ua = 0` and only sets it nonzero when `zeta > 0`. However, the 2D routing logic (`envs->f_g0_2d4d`) for YP uses `CINTg0_2e_stg_lj2d4d` only in the `kbase=false, ibase=false` case — other cases use the standard 2e recurrence functions. This means the 4D recurrence path for YP depends on the (kbase, ibase) routing, which is different from the STG path routing.

If the CubeCL YP kernel incorrectly routes through the STG 4D recurrence for all cases (copying STG dispatch logic without checking the ibase/kbase conditions), it will produce wrong results for the majority of shell combinations where ibase=true or kbase=true.

**Why it happens:**
The `CINTinit_int2e_yp_EnvVars` function selects `f_g0_2d4d` based on relative angular momenta (lines 113-127 in g2e_f12.c): for ibase, it uses `CINTg0_2e_ik2d4d`; for kbase, `CINTg0_2e_kj2d4d`; for kbase+!ibase, `CINTg0_2e_il2d4d`; for the default (lbase), `CINTg0_2e_stg_lj2d4d`. The STG version uses the same routing. The difference is only in the fallback case. If a GPU implementation collapses both YP and STG to the same recurrence kernel, the ibase/kbase distinction is lost.

**How to avoid:**
Implement YP and STG as distinct CubeCL kernel paths, not as a shared kernel with a zeta flag. Verify oracle parity for all four (ibase, kbase) combinations: (T,T), (T,F), (F,T), (F,F). The fixture matrix must cover shell combinations where li > lj (ibase=true) and lk > ll (kbase=true).

**Warning signs:**
- YP oracle failures cluster at shell combinations where the bra angular momentum exceeds the ket
- STG oracle passes but YP oracle fails for the same shell combination
- YP and STG produce identical output for all cases (suggests STG recurrence applied to both)

**Phase to address:**
F12 kernel implementation phase, specifically when writing the 4D recurrence for YP.

---

### Pitfall 7: Spinor output uses interleaved re/im doubles and the buffer size is double the scalar element count

**What goes wrong:**
Spinor integrals output `double complex` values stored as interleaved `(re, im)` pairs in a flat `f64` buffer. The buffer size is `2 * comp * di * dj` elements (for 1e) or `2 * comp * di * dj * dk * dl` (for 2e). If the executor allocates a staging buffer sized for `comp * di * dj` f64 elements (scalar count), it is half the required size. The overflow may not immediately panic — it may corrupt adjacent memory or produce truncated output.

The design document (section 3.6) notes: "compat spinor accepts `double complex*` in a `double*`-compatible way, so it requires twice as many elements in a flat double buffer."

**Why it happens:**
The spinor size calculation is easy to confuse with the scalar size. The helper `CINTlen_spinor(l, kappa)` returns the number of spinor components (e.g., `2l+2` for `kappa<0`), which is fewer than the number of cartesian components `(l+1)(l+2)/2`. But each spinor component is complex, so the flat buffer doubles again. The net result is: the flat double buffer for spinor can be either larger or smaller than the cart buffer depending on l and kappa.

**How to avoid:**
The staging buffer allocation must use `spinor_component_count * 2` for the f64 element count. The workspace estimator must include the factor-of-2 for spinor families. Add an explicit assertion in the executor that staging buffer length == `expected_spinor_elements * 2` before dispatch.

Run the spinor size calculation against upstream `CINTcgtos_spinor` helper (which returns the contracted count) and multiply by `2 * ncomp` to get the f64 buffer size. Cross-check: `CINTcgtos_spinor(bas_id, bas) * 2 * ncomp == staging.len()` for 1e spinor calls.

**Warning signs:**
- Spinor staging buffer allocation silently truncates (no OOM error, wrong element count)
- Spinor output values match only the first half of the expected tensor
- The `query_workspace` size estimate for spinor is the same as for sph (factor of 2 missing)

**Phase to address:**
Spinor buffer sizing must be validated in the planning/workspace estimation layer before kernel dispatch.

---

### Pitfall 8: 4c1e identity relation uses the specific trace `int4c1e_sph == (-1/4pi) * trace(int2e_ipip1 + 2*int2e_ipvip1 + perm_ipip1)`; misimplementing the permutation breaks the identity test

**What goes wrong:**
The upstream 4c1e validation identity (testsuite/test_cint4c1e.py:196-234) computes a reference for `int4c1e_sph(i,j,k,l)` by combining second-derivative 2e integrals. The permutation term uses `int2e_ipip1(j,i,k,l)` with swapped first two indices, transposed with `[:,:,:,:].transpose(1,0,2,3)`. If the identity verification code in cintx uses the wrong index permutation (e.g., omits the transpose or swaps different indices), the identity test may fail even for correct 4c1e output, or may pass for incorrect output.

**Why it happens:**
The mathematical identity is: `int4c1e(i,j,k,l) = (-1/4pi) * [trace_xx(int2e_ipip1(i,j,k,l)) + 2*trace(int2e_ipvip1(i,j,k,l)) + trace_xx(int2e_ipip1(j,i,k,l))]` where `trace_xx` means the diagonal sum over the two derivative components. The permuted term requires swapping shells i,j AND transposing the output tensor accordingly. This is a subtle combination of shell index swap and tensor dimension permutation.

**How to avoid:**
Copy the exact identity computation from `test_cint4c1e.py:197-218` verbatim into the cintx identity test harness. Do not simplify or rearrange. Verify the identity against the upstream C library output before using it to verify cintx output. Add a test that intentionally passes wrong shell order and confirms the identity fails (smoke test for the test itself).

**Warning signs:**
- Identity test passes trivially for all shell combinations (too-good result)
- Identity test fails for shell combinations where i==j (symmetric case where permutation is no-op)
- Identity errors are exactly twice the expected value (suggests the permuted term is counted twice or not at all)

**Phase to address:**
4c1e identity test implementation phase. The identity harness should be verified against upstream before the 4c1e kernel is implemented.

---

### Pitfall 9: Helper API oracle comparison must use exact integer arithmetic, not floating-point

**What goes wrong:**
Helper APIs (`CINTlen_cart`, `CINTcgtos_spheric`, `CINTshells_spheric_offset`, etc.) return integer counts and offsets. If the oracle comparison framework uses floating-point `atol`/`rtol` comparison (e.g., the same `diff_summary` path used for integral values), it may report large relative errors for zero-valued offsets or produce spurious mismatches for count values that differ by 1 due to integer/float conversion.

**Why it happens:**
The oracle harness is designed around float comparison for integral buffers. When the same framework is used for helper APIs, integer values are likely cast to f64 before comparison. For offset arrays, zero entries have undefined relative error (0/0). For count values, a bug that produces `(2l+1)` instead of `(2l+2)` for kappa>0 shells produces an absolute error of 1.0, which falls outside atol=1e-12 but would be trivially within atol=1e7.

**How to avoid:**
Helper API oracle comparison must use exact integer equality, not float tolerance. The oracle framework needs a separate comparison path for integer-valued APIs. Specifically: `CINTlen_cart`, `CINTlen_spinor`, `CINTcgtos_*`, `CINTcgto_*`, `CINTtot_*` all return `usize` or `i32`; compare with `==`. `CINTshells_*_offset` returns `Vec<usize>`; compare element-wise with `==`. `CINTgto_norm` is a float and uses the standard atol comparison.

**Warning signs:**
- Helper oracle comparison reports values like `atol violation: 1.0, ref=5.0, obs=6.0` (should be an integer equality failure, not a float comparison)
- Offset array comparisons pass despite wrong values because atol is loose enough to mask off-by-one errors

**Phase to address:**
Oracle harness extension phase for helper APIs. Add an integer comparison path before authoring helper fixtures.

---

### Pitfall 10: Manifest lock must be regenerated for every feature-profile combination when F12 and 4c1e kernels land

**What goes wrong:**
The compiled manifest lock (`compiled_manifest.lock.json`) is generated from the union of compiled symbols across `{base, with-f12, with-4c1e, with-f12+with-4c1e}`. When F12 and 4c1e kernels are added, the compiled symbol set changes. If the lock is not regenerated, the manifest audit gate fails. But if the lock is regenerated without running oracle parity first, the gate may silently accept incorrect API coverage.

The F12 manifest requirement (design section 3.11.1) is specific: the `with-f12` profile must contain exactly the sph series listed in the coverage matrix, and "cart/spinor symbol count is zero" must be verified. If a developer accidentally exposes a `with-f12` feature flag on a module that also compiles cart/spinor stubs, the symbol count check fails.

**Why it happens:**
Adding feature gates to the wrong layer (e.g., putting `#[cfg(feature = "with-f12")]` on the kernel module entry point but not on the individual symbol re-exports in the facade) can cause symbols to appear in compiled output even when the feature is off, or can cause symbols to be missing even when the feature is on. Feature gate propagation through a 6-crate workspace is error-prone.

**How to avoid:**
Run `cargo xtask manifest-audit` immediately after adding each new feature-gated symbol, before oracle comparison. Treat manifest audit failures as blocking. The CI matrix must run the audit job first, parallel to (not after) the oracle comparison job.

For F12: verify that `cargo build --features with-f12 | nm -D | grep spinor | wc -l == 0` as part of the release gate.

**Warning signs:**
- Manifest audit passes but oracle comparison fails (symbols exist but produce wrong output)
- Manifest audit fails because a symbol appears in `base` profile that should only be in `with-f12`
- `cargo build --features with-f12` succeeds but the symbol list includes cart variants

**Phase to address:**
Manifest audit must be run at the start of the F12 implementation phase, not at the end. Add the audit to the development loop, not just CI.

---

### Pitfall 11: Unstable-source-api symbols have no upstream oracle and require property-based testing only

**What goes wrong:**
Source-only exported families behind `unstable-source-api` (origi, origk, additional Breit variants, 3c2e_ssc, etc.) do not appear in `include/cint_funcs.h` or `include/cint.h.in`. The upstream oracle is built from the compiled vendored library, and these symbols may or may not be compiled into the oracle depending on the build flags used for the vendored build. If the oracle build does not enable the source-only APIs, oracle comparison returns "symbol not found" rather than a reference value.

**Why it happens:**
The oracle harness in `cintx-oracle` uses `bindgen` over the vendored libcint headers. Source-only families are not in the headers — they are only in `src/*.c`. The bindgen output will not include them unless the oracle is rebuilt with additional headers or raw `dlsym` lookups.

**How to avoid:**
For unstable-source-api families, the oracle strategy must be:
1. Rebuild the vendored libcint with the appropriate build flags to expose source-only symbols
2. Use `dlsym`-based dynamic lookup (not bindgen) to call the oracle reference
3. Add the source-only symbol list to the oracle harness separately from the bindgen-generated bindings
4. Use property-based testing (proptest) for mathematical consistency checks (permutation symmetry, scaling with exponent, zero-shell limits) rather than relying solely on oracle comparison

Do not claim oracle-backed coverage for unstable-source-api symbols unless the oracle harness has been explicitly extended to include them.

**Warning signs:**
- Oracle comparison for unstable-source-api symbols returns 0 reference values (oracle found nothing)
- The CI gate for `unstable-source-api` passes with no comparison results (false pass)
- `manifest-audit` shows unstable symbols present but the oracle report has no entries for them

**Phase to address:**
Before implementing any unstable-source-api kernels, verify the oracle harness can actually call the upstream reference. If it cannot, add `dlsym` lookup before writing any kernels.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Use flat 1e-12 atol for all families in CI before empirical calibration | Simpler configuration, single tolerance constant | 3c1e/4c1e/F12 CI permanently fails; implementation stalls on false gates | Never — calibrate tolerance per family from oracle data first |
| Keep c2spinor.rs stub and add a skip marker on spinor oracle tests | Unblocks non-spinor work | Spinor oracle coverage never gets authored; stub ships | Only if a release with explicit `UnsupportedApi` for all spinor families is the goal |
| Author F12 fixtures with zeta=0 (plain Coulomb) | Fixtures pass immediately | STG path is never exercised; zeta != 0 cases silently fail in production | Never |
| Regenerate manifest lock without running oracle first | Faster CI feedback loop | Lock may accept wrong coverage; oracle failures appear later as surprises | Never on main branch; acceptable on feature branches with explicit TODO |
| Use the same oracle comparison path for helper integer APIs as for integral float APIs | No additional code | Off-by-one count errors masked by atol; wrong shell sizes ship | Never |
| Defer 4c1e spinor UnsupportedApi until after cart/sph work | Reduces scope | Users encounter stderr-only failure from upstream; cintx returns wrong zero instead of typed error | Never |
| Port STG/YP to a shared CubeCL kernel with a runtime zeta flag | Reduces kernel count | ibase/kbase routing diverges; YP oracle failures in non-default routing case | Never — keep STG and YP as separate kernel entry points |

---

## Integration Gotchas

Common mistakes when connecting components in this codebase.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| F12 env layout | Setting `env[9]` (zeta) but forgetting to initialize `env[8]` (omega) | Set both PTR_RANGE_OMEGA=8 and PTR_F12_ZETA=9 for every STG/YP test fixture |
| Spinor buffer allocation | Sizing staging buffer for `spinor_components * ncomp` f64 elements | Size for `spinor_components * ncomp * 2` f64 elements (each complex element is 2 doubles) |
| 4c1e bug envelope | Checking only `max(l) <= 4` before allowing 4c1e execution | Check: cart/sph only, scalar only, natural dims, max(l)<=4, oracle pass, identity pass — all must be true |
| Feature propagation | Putting `#[cfg(feature = "with-f12")]` only on kernel module, not on facade re-exports | Apply feature gate at every layer: kernel, compat dispatch, ops resolver, facade |
| Oracle harness for unstable APIs | Using bindgen-generated bindings for source-only symbols | Use `dlsym` dynamic lookup with explicit vendor build flags |
| STG roots clamping | Implementing STG quadrature with unclamped t argument | Clamp `t = min(t, 19682.99)` before table lookup, matching upstream behavior |
| Spinor kappa encoding | Using `kappa = 0` to mean "no spinor" | `kappa = 0` means sum of both j=l+1/2 and j=l-1/2 in upstream; use the Representation enum, not kappa value, to distinguish spinor from sph |

---

## Performance Traps

Patterns that work at small scale but fail under full-matrix oracle runs.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Running all feature-profile combinations sequentially in CI | Manifest audit + oracle for 4 profiles takes hours | Parallelize profile builds in CI; use cargo nextest --jobs | At full shell-combination oracle matrix for all families |
| Allocating spinor staging as 2x scalar workspace on every evaluation | 2x memory pressure vs scalar path | Pre-compute spinor workspace multiplier in planner, surface it in query_workspace | When running large basis sets on GPU with tight memory limits |
| STG roots table loaded from DAT file on every kernel launch | First-launch latency spike for F12 | Load roots table once at backend initialization, pin in GPU constant memory | From the first STG call; no scale threshold |
| Generating oracle fixtures for all shell combinations including high-l | Fixture generation hangs | Limit fixture matrix to max_l=3 for high-derivative families; use proptest for higher l | At l>4 with derivative operators (nroots > 8) |

---

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **F12 feature gate:** Feature compiles under `--features with-f12` — verify that `nm -D` shows only sph symbols and zero cart/spinor symbols
- [ ] **STG zeta parameter:** STG kernel produces non-zero output — verify that output changes when `env[9]` is changed (not just when shells change)
- [ ] **Spinor transform:** `CINTc2s_ket_spinor_sf1` entry point compiles — verify oracle comparison with actual libcint spinor values for l=1 shells
- [ ] **4c1e envelope:** `Validated4C1E` classifier allows cart/sph, scalar, natural dims, max(l)<=4 — verify it rejects spinor and max(l)>4 with `UnsupportedApi`
- [ ] **4c1e identity:** Identity test framework exists — verify it fails intentionally when passing wrong shell order (test-the-test)
- [ ] **Helper oracle:** Helper symbols appear in `IMPLEMENTED_HELPER_SYMBOLS` list — verify they use exact integer equality, not float atol comparison
- [ ] **Unstable oracle:** Unstable-source-api symbols are in manifest — verify the oracle harness can actually call them (not just that the symbols compile)
- [ ] **Tolerance audit:** Oracle CI gate uses 1e-12 for some families — verify per-family empirical error distribution before encoding any tolerance as immutable
- [ ] **Spinor buffer size:** `query_workspace` returns correct byte estimate for spinor — verify it doubles the scalar count for the interleaved complex layout

---

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| 1e-12 tolerance gates fail on 3c1e | MEDIUM | Add per-family tolerance override in oracle compare.rs; re-run oracle to establish empirical floor; update gate to per-family values |
| Spinor stub ships to oracle comparison | LOW | Block oracle for spinor families in CI with explicit `#[ignore]` + comment; implement real transform before removing block |
| F12 oracle produces wrong output (zeta=0 fixtures) | MEDIUM | Re-generate all F12 fixtures with correct zeta values; add fixture validator that rejects zeta=0 for STG/YP |
| 4c1e spinor silently returns zero instead of UnsupportedApi | LOW | Add spinor check at top of 4c1e capability classifier; write test that asserts UnsupportedApi for 4c1e + Spinor |
| Manifest lock accepted wrong symbols | MEDIUM | Revert manifest lock to previous commit; re-run `cargo xtask manifest-audit` per profile before merging F12 additions |
| STG roots clamping mismatch causes oracle failures for diffuse basis | HIGH | Profile failing fixtures to identify t > 19682 cases; add clamp to CubeCL STG roots kernel; re-run full F12 oracle |
| Helper oracle uses float comparison and masks count errors | LOW | Add separate integer comparison path in oracle harness; re-run helper oracle with exact equality |

---

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 1: Flat 1e-12 atol on 3c1e | Tolerance audit (before F12 phase) | Run 3c1e oracle, plot error distribution, set per-family gates from evidence |
| 2: Spinor transform stub | Spinor transform implementation (before any spinor oracle) | Oracle comparison for l=1 spinor integrals matches upstream |
| 3: PTR_F12_ZETA zero in fixtures | F12 fixture generation phase | Fixture validator rejects zeta=0; manual check of env[8]/env[9] values |
| 4: 4c1e spinor unimplemented upstream | 4c1e capability classifier phase | `UnsupportedApi` test for 4c1e + Spinor passes |
| 5: STG roots domain clamp | F12 CubeCL kernel implementation | Fixture with large inter-center distance passes oracle |
| 6: YP vs STG 4D routing | F12 kernel implementation (separate YP/STG paths) | Oracle parity for all (ibase, kbase) combinations |
| 7: Spinor buffer size | Workspace estimator / planner phase | `query_workspace` for spinor == 2x scalar; staging allocation assertion fires on wrong size |
| 8: 4c1e identity permutation | 4c1e identity harness implementation | Identity test fails when shell order is intentionally wrong |
| 9: Helper integer comparison | Oracle harness extension phase | Helper oracle uses exact integer equality; count-by-1 error triggers failure |
| 10: Manifest lock timing | CI configuration; run manifest audit first | Audit fails if any symbol appears in wrong profile |
| 11: Unstable oracle via dlsym | Unstable oracle harness setup | Oracle harness calls vendored unstable symbol and returns non-zero reference |

---

## Sources

- `libcint-master/src/cint4c1e.c:349-353` — `int4c1e_spinor` is unimplemented (prints to stderr, returns 0)
- `libcint-master/CMakeLists.txt:113-116` — "Note there are bugs in 4c1e integral functions"
- `libcint-master/src/stg_roots.c` — STG roots use Clenshaw polynomial with clamped domain `t = min(t, 19682.99)`
- `libcint-master/src/g2e_f12.c` — YP and STG differ only in the lbase 4D recurrence (`CINTg0_2e_stg_lj2d4d` vs standard); routing is selected by ibase/kbase flags
- `libcint-master/include/cint.h.in:35-45` — env layout: PTR_RANGE_OMEGA=8, PTR_F12_ZETA=9, PTR_ENV_START=20
- `libcint-master/testsuite/test_int2e_f12_etc.py:46-47` — F12 test sets `env[9]=1e-3` (zeta), `env[8]=1e3` (omega)
- `libcint-master/testsuite/test_cint4c1e.py:197-231` — 4c1e identity: `int4c1e_sph == (-1/4pi) * trace(ipip1 + 2*ipvip1 + perm_ipip1)` with `dd > 1e-6` threshold
- `crates/cintx-oracle/src/compare.rs:21-30` — Current oracle tolerances: 1e-11 (1e), 1e-12 (2e), 1e-9 (2c2e/3c2e), 1e-7 (3c1e), 1e-6 (4c1e)
- `crates/cintx-cubecl/src/transform/c2spinor.rs` — Current spinor transform is a stub using `amplitude = (|re|+|im|)*0.5`, not Clebsch-Gordan coefficients
- `docs/design/cintx_detailed_design.md:section 3.11.1` — F12/STG/YP supported as sph only; cart/spinor symbol count zero is a release gate condition
- `docs/design/cintx_detailed_design.md:section 3.11.2` — 4c1e bug envelope: cart/sph, scalar, natural dims, max(l)<=4, oracle+identity pass
- `docs/design/cintx_detailed_design.md:section 13.8` — Per-family tolerances: F12=1e-6/1e-4, 4c1e=1e-6/1e-5, 3c1e=1e-7/1e-5
- `src/cint_bas.c:17-26` — Spinor length formula: `4l+2` (kappa=0), `2l+2` (kappa<0), `2l` (kappa>0)

---

*Pitfalls research for: full API parity — F12/STG/YP families, spinor representations, extended 4c1e, helper/transform/wrapper APIs, unified oracle tolerance (cintx v1.2)*
*Researched: 2026-04-04*
