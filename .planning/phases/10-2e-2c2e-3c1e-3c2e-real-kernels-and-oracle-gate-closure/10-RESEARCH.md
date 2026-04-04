# Phase 10: 2e, 2c2e, 3c1e, 3c2e Real Kernels and Oracle Gate Closure - Research

**Researched:** 2026-04-03
**Domain:** Rys quadrature ERI kernels, multi-center G-tensor fills, oracle parity, libcint G-tensor indexing
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Kernel Architecture**
- D-01: Each family gets its own G-fill implementation mirroring its libcint counterpart (g2e.c, g2c2e.c, g3c1e.c, g3c2e.c). Share math primitives (Rys, VRR, HRR, Boys) but not the orchestration logic.
- D-02: Support full angular momentum up to g-function (l=4) from the start for all families. The Phase 8 math primitives already handle arbitrary l.
- D-03: All kernels are host-side pure Rust computation using *_host() math wrappers. Same pattern as Phase 9 one_electron.rs. No CubeCL #[cube] GPU kernels in this phase (avoids cond_br MLIR issues).

**Oracle Parity Strategy**
- D-04: Build vendored libcint-master/ with cc crate, call upstream C functions via bindgen FFI, compare outputs numerically. Use the existing cintx-oracle/ harness infrastructure.
- D-05: Multiple test molecules and basis sets: H2O + H2 + CH4 across STO-3G and cc-pVDZ. Provides both fast smoke tests (STO-3G, s/p shells) and full angular momentum coverage (cc-pVDZ, s/p/d shells).
- D-06: Tolerances per success criteria: 2e atol 1e-12 / rtol 1e-10, 2c2e atol 1e-9, 3c1e atol 1e-7, 3c2e atol 1e-9. Oracle parity verified per family as each kernel lands (VERI-05).

**v1.0 UAT Items**
- D-07: Both UAT items verified via automated integration tests in CI: (1) eval_raw() on a real basis set (H2O STO-3G int1e_ovlp_sph) asserts non-zero output, (2) C ABI cintrs_eval() test under CPU backend asserts status == 0.

**Implementation Order**
- D-08: Ascending complexity: 2c2e -> 3c1e -> 3c2e -> 2e. Each family is a separate plan with its own oracle parity test. A 5th plan handles oracle gate closure across all families plus UAT item resolution.
- D-09: 2c2e (2-center Coulomb) is closest to nuclear attraction (Rys + 2 centers). 3c1e adds a third center without Rys. 3c2e combines 3 centers with Rys. 2e (full ERI) is the most complex with 4 shells and full Rys quadrature.

**Carried Forward**
- D-10: Host wrapper + #[cube] pair pattern for math functions (Phase 8).
- D-11: CPU backend is the primary oracle target (Phase 7/9). Tests run under `--features cpu`.
- D-12: C2S transform is host-side Rust (Phase 9 D-04/D-06). Kernel writes cartesian to staging, host applies Condon-Shortley transform.
- D-13: Buffer lifecycle lives inside kernel family modules, not centralized (Phase 7 D-07).

### Claude's Discretion
- Internal G-tensor array sizing and indexing strategy per family (following libcint source as guide)
- Exact Rys quadrature integration into each family's G-fill (root iteration, weight application)
- Test fixture design for per-family oracle comparison
- How to structure the oracle gate closure CI check (xtask command vs test)
- Plan boundaries within each family (e.g., whether contraction + c2s is separate from G-fill)

### Deferred Ideas (OUT OF SCOPE)
- GPU-side #[cube] kernels for all families
- Screening/batching optimizations
- Higher angular momentum (l>=5, h-functions)
- Spinor representation kernels
- F12/STG/YP optional family kernels
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| KERN-02 | 2e ERI kernel implements Rys quadrature with real Gaussian integral evaluation | G-tensor fill patterns from g2e.c; 4-shell index layout; rys_roots host dispatch; vrr_2e_step_host + hrr_step_host reuse |
| KERN-03 | 2c2e two-center two-electron kernel produces real values | g2c2e.c shows it reuses g2e's CINTg0_2e machinery with j_l=l_l=0; 2-shell index layout |
| KERN-04 | 3c1e three-center one-electron kernel produces real values | g3c1e.c uses Boys-based VRR (not Rys) for overlap; Rys only for nuclear term; 3-shell g_stride_i/j/k layout |
| KERN-05 | 3c2e three-center two-electron kernel produces real values | g3c2e.c reuses CINTg0_2e with ll_ceil = lk and al=0; 3-shell index layout |
| VERI-05 | Oracle parity verified per family as each kernel lands (existing infrastructure) | vendor_ffi.rs needs int2e_sph / int2c2e_sph / int3c1e_sph / int3c2e_sph wrappers; oracle build.rs needs 2e+ source files |
| VERI-07 | v1.0 UAT items: non-zero eval_raw output and C ABI shim returns status == 0 | both already have code paths; integration tests can assert directly using existing CPU executor path |
</phase_requirements>

---

## Summary

Phase 10 replaces four zero-returning stub kernel files with real G-tensor fill pipelines that produce libcint-compatible values, then closes the oracle parity CI gate for all five base families. Every family has canonical reference source in `libcint-master/src/` — the implementation strategy is to port those algorithms closely in pure Rust using the math primitives already built in Phase 8 and the `one_electron.rs` pipeline structure from Phase 9.

Three families (2c2e, 3c2e, and 2e) use Rys quadrature. The 3c1e overlap variant uses a three-center Gaussian product VRR that is Boys-free; its nuclear term uses the Boys function accumulated over Rys roots. The 2e kernel is the most complex: it has a 4-index G-tensor with two independent Gaussian pair products, two VRR streams per axis (c00/c0p, b10/b01/b00), and up to 5 Rys roots for the cc-pVDZ test target. The host-side rys_root1_host and rys_root2_host wrappers already exist; rys_root3_host through rys_root5_host (or a unified rys_roots_host dispatcher) must be added before 2e and 2c2e/3c2e can cover d-shell cases.

For oracle parity, the existing vendor build infrastructure in `cintx-oracle/build.rs` compiles only 1e sources. It must be extended to compile the 2e, 2c2e, 3c1e, and 3c2e C source files and expose their FFI entry points in `vendor_ffi.rs`. The oracle gate closure plan then asserts `mismatch_count == 0` across all five families and resolves the two v1.0 UAT items.

**Primary recommendation:** Follow the 1e kernel's exact structural pattern — G-fill function, contract function, c2s dispatch, launch entry point — one per family, verified against the matching libcint source line by line. Add rys_roots_host dispatcher first; add c2s_Xe variants second; then implement G-fills in complexity order (2c2e, 3c1e, 3c2e, 2e); extend oracle FFI last.

---

## Standard Stack

### Core (unchanged from prior phases)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Rust toolchain | 1.94.0 (rust-toolchain.toml) | Reproducible compiler | Pinned per CLAUDE.md |
| `thiserror` | 2.0.18 | Public error surface | CLAUDE.md constraint |
| `anyhow` | 1.0.102 | Oracle/test error handling | CLAUDE.md constraint |
| `cubecl` | 0.9.x | Backend (CPU arm for oracle) | CLAUDE.md constraint |
| `cc` | 1.2.x | Vendored libcint compilation | Already in oracle build.rs |
| `bindgen` | 0.71.1 | FFI binding generation | Already in oracle build.rs |

### No new dependencies required for this phase.

All math primitives (Boys, Rys root1-5, VRR/HRR host wrappers, PairData, c2s coefficients l0-l4) already exist in the workspace. The only additions are new host dispatch functions and new source-file entries inside existing crates.

---

## Architecture Patterns

### Pattern 1: G-Fill + Contract + C2S Pipeline (from Phase 9 one_electron.rs)

All four new kernels follow the identical top-level structure:

```rust
// Source: one_electron.rs lines 434-end (template for all families)
pub fn launch_center_2c2e(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    let _ = backend; // host-side; no GPU dispatch
    // 1. Extract shells and atom coords from plan
    // 2. Primitive loop over all shell combinations
    //    a. Compute PairData via compute_pdata_host()
    //    b. Call family-specific fill_g_tensor_*()
    //    c. Contract G-tensor to Cartesian buffer
    //    d. Scale by common_fac_sp per center
    // 3. If Spheric: apply cart_to_sph_Xe() variant
    // 4. Write to staging, return ExecutionStats
}
```

**What stays the same across all four families:**
- `FamilyLaunchFn` signature
- `common_fac_sp(l)` normalization factor applied in primitive loop
- `cart_comps(l)` enumeration of Cartesian components
- `ncart(l)` / `nsph(l)` from `transform::c2s`
- `ExecutionStats` construction at the end

### Pattern 2: G-Tensor Layout per Family

Each family has a different G-tensor flat-array layout driven by its center count and index ordering.

**2e (4-center):** Layout from `g2e.c CINTinit_int2e_EnvVars`:
```
g_size = nrys_roots * dli * dlk * dll * dlj
  where dli/dlj driven by ibase = (li_ceil > lj_ceil)
  where dlk/dll driven by kbase = (lk_ceil > ll_ceil)
g_stride_i = nrys_roots
g_stride_k = nrys_roots * dli
g_stride_l = nrys_roots * dli * dlk
g_stride_j = nrys_roots * dli * dlk * dll
```
The innermost (fastest-varying) dimension is the Rys root index. The G-tensor holds 3 * g_size doubles [gx | gy | gz].

**2c2e (2-center, from g2c2e.c):** Reuses 2e G-tensor machinery with j_l=0, l_l=0:
```
g_size = nrys_roots * dli * dlk
  where dli = li_ceil + 1 (no HRR on j since j_l=0)
  where dlk = lk_ceil + 1
g_stride_i = nrys_roots
g_stride_k = nrys_roots * dli
```
Effectively a 2-center 2e with j and l shells being s-functions (angular momentum 0).

**3c1e (3-center 1e, from g3c1e.c):** Uses g1e-style VRR with 3 Gaussian centers, NOT the 2e G-tensor:
```
g_size = dli * dlj_max * dlk
  where dli = li_ceil + 1
  where dlj = lj_ceil + lk_ceil + 1 (combined second/third center dimension)
  where dlk = lk_ceil + 1
g_stride_i = 1
g_stride_j = dli
g_stride_k = dli * dlj
```
No Rys roots in the overlap integral — the G-tensor is filled with a three-center product VRR. The nuclear term uses Boys-weighted Rys (see CINTg3c1e_nuc in g3c1e.c lines 192-270).

**3c2e (3-center 2e, from g3c2e.c):** Reuses 2e G-tensor machinery with ll_ceil=lk, al=0:
```
g_size = nrys_roots * dli * dlk * dlj
  where ibase = (li_ceil > lj_ceil) drives dli/dlj
  where dlk = ll_ceil + 1 = lk_ceil + 1
  where ll_ceil is set to k_l to reuse CINTg0_2e_lj2d4d
g_stride_i = nrys_roots
g_stride_k = nrys_roots * dli  (g_stride_l = g_stride_k)
g_stride_j = nrys_roots * dli * dlk
```

### Pattern 3: Rys Root Count per Family

| Family | nrys_order formula | H2O STO-3G | cc-pVDZ |
|--------|-------------------|------------|---------|
| 2e | (li+lj+lk+ll)/2 + 1 | 3 (p+p+p+p) | 5 (d+d+d+d) |
| 2c2e | (li+lk)/2 + 1 | 1 (s+s) | 3 (d+d) |
| 3c1e | N/A for overlap; (li+lj+lk)/2+1 for nuc | 1 (s+s+s) | 3 (d+d+d) |
| 3c2e | (li+lj+lk)/2 + 1 | 2 (p+p+s) | 4 (d+d+d) |

**Critical gap:** Only `rys_root1_host` and `rys_root2_host` exist as host wrappers. A `rys_roots_host(nroots, x)` dispatcher is needed that calls the matching `rys_rootN` CubeCL function's equivalent logic for N=3,4,5. This is a Wave 0 blocker for 2e with cc-pVDZ d-shell coverage.

The `#[cube]` functions `rys_root3`, `rys_root4`, `rys_root5` are already implemented and working. Host wrappers follow the exact same pattern as `rys_root1_host` and `rys_root2_host` — they duplicate the Rust branching logic without CubeCL attributes.

### Pattern 4: cart_to_sph Variants Needed

The Phase 9 `cart_to_sph_1e` handles a 2-index (ni x nj) Cartesian buffer. The new families need:

| Family | Index count | Transform name to add | Buffer shape |
|--------|------------|----------------------|--------------|
| 2c2e | 2-index (i, k) | `cart_to_sph_2c2e` | [ncart(li), ncart(lk)] -> [nsph(li), nsph(lk)] |
| 3c1e | 3-index (i, j, k) | `cart_to_sph_3c1e` | [ncart(li), ncart(lj), ncart(lk)] -> sph |
| 3c2e | 3-index (i, j, k) | `cart_to_sph_3c2e` | [ncart(li), ncart(lj), ncart(lk)] -> sph |
| 2e | 4-index (i, j, k, l) | `cart_to_sph_2e` | [ncart(li), ncart(lj), ncart(lk), ncart(ll)] -> sph |

All use the same Condon-Shortley coefficient tables (C2S_L0 through C2S_L4) already defined in `c2s.rs`. The logic is a sequential application of `c2s_coeff` along each index axis. The Phase 9 `cart_to_sph_1e` is the template — extend to 3 and 4 indices by adding intermediate transpose buffers.

Note from `libcint/src/cart2sph.h`: `c2s_sph_2e`, `c2s_sph_2c2e`, `c2s_sph_3c1e`, `c2s_sph_3c2e` are separate C functions. The Rust equivalents apply the transform per axis sequentially; libcint does the same thing with different loop orderings per function.

### Pattern 5: Oracle Vendor FFI Extension

The current `build.rs` compiles only 1e source files:
```
src/cint1e.c, src/cint1e_a.c, src/g1e.c, src/autocode/intor1.c
```

For 2e+ families, add to the `cc::Build` chain:
```
src/cint2e.c, src/g2e.c          -- 2e ERI
src/cint2c2e.c, src/g2c2e.c      -- 2c2e
src/cint3c1e.c, src/g3c1e.c      -- 3c1e
src/cint3c2e.c, src/g3c2e.c      -- 3c2e
src/autocode/intor2.c             -- auto-generated int2e_sph etc. (check if present)
src/autocode/intor3.c, intor4.c  -- (check for 3c functions)
```

And extend bindgen `allowlist_function` to include:
```
int2e_sph|int2c2e_sph|int3c1e_sph|int3c2e_sph|CINTcgto_spheric
```

Then add corresponding `vendor_int2e_sph`, `vendor_int2c2e_sph`, `vendor_int3c1e_sph`, `vendor_int3c2e_sph` functions in `vendor_ffi.rs` following the exact pattern of `vendor_int1e_ovlp_sph`.

### Pattern 6: common_factor for 2e+ Families

From the libcint source, `common_factor` differs per family:

| Family | common_factor formula |
|--------|----------------------|
| 2e | `(pi^3)*2/sqrt(pi) * fac_sp(li)*fac_sp(lj)*fac_sp(lk)*fac_sp(ll)` |
| 2c2e | `(pi^3)*2/sqrt(pi) * fac_sp(li)*fac_sp(lk)` (j_l=l_l=0 so those fac_sp=1) |
| 3c1e | `sqrt(pi)*pi * fac_sp(li)*fac_sp(lj)*fac_sp(lk)` |
| 3c2e | `(pi^3)*2/sqrt(pi) * fac_sp(li)*fac_sp(lj)*fac_sp(lk)` |

The `common_factor` is applied to the gz base case (gz[0]) before the VRR/HRR fill. In one_electron.rs, the equivalent is `pd.fac * SQRTPI * PI / (aij * aij.sqrt())`. For 2e families there are two Gaussian pair products (ij pair and kl pair), so the gz[0] base case carries the product of both pair prefactors weighted by the Rys weight.

### Recommended File Structure

```
crates/cintx-cubecl/src/
├── kernels/
│   ├── center_2c2e.rs     -- Replace stub: G-fill + contract + c2s (2-center 2e)
│   ├── center_3c1e.rs     -- Replace stub: G-fill + contract + c2s (3-center 1e)
│   ├── center_3c2e.rs     -- Replace stub: G-fill + contract + c2s (3-center 2e)
│   └── two_electron.rs    -- Replace stub: G-fill + contract + c2s (4-center 2e)
├── math/
│   └── rys.rs             -- ADD: rys_root3_host/rys_root4_host/rys_root5_host + rys_roots_host dispatcher
└── transform/
    └── c2s.rs             -- ADD: cart_to_sph_2e, cart_to_sph_2c2e, cart_to_sph_3c1e, cart_to_sph_3c2e

crates/cintx-oracle/
├── build.rs               -- EXTEND: add 2e+ source files, update bindgen allowlist, add rerun triggers
├── src/
│   └── vendor_ffi.rs      -- ADD: vendor_int2e_sph/int2c2e_sph/int3c1e_sph/int3c2e_sph wrappers
└── tests/
    ├── two_electron_parity.rs    -- NEW: oracle parity for 2e (H2O STO-3G + H2 STO-3G)
    ├── center_2c2e_parity.rs     -- NEW: oracle parity for 2c2e
    ├── center_3c1e_parity.rs     -- NEW: oracle parity for 3c1e
    ├── center_3c2e_parity.rs     -- NEW: oracle parity for 3c2e
    └── oracle_gate_closure.rs    -- NEW: gate closure test + UAT items
```

### Anti-Patterns to Avoid

- **Sharing orchestration across families:** D-01 locks each family to its own G-fill. Do not abstract the G-tensor loop into a shared "generic 2e fill" — the index ordering differs per family in non-trivial ways.
- **Centralizing buffer lifecycle:** D-13 requires buffer allocation to live inside each kernel file. Do not add a shared allocator.
- **Using CubeCL launch for the new kernels:** D-03 locks all new kernels to host-side `*_host()` wrappers. The `let _ = backend;` suppression is intentional.
- **Deferring c2s variants:** The `cart_to_sph_staging` stub in c2s.rs is a no-op. New kernels must call their per-family c2s variant directly (pattern from `launch_one_electron`), not rely on the staging no-op.
- **Batching oracle parity at the end:** D-06 / VERI-05 require per-family parity verification as each kernel lands, not as a single gate-closure step at the end.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Rys roots/weights | Custom polynomial fit | `rys_root1_host` ... `rys_root5_host` (add 3-5 wrappers from existing `#[cube]` implementations) | Already ported from libcint rys_roots.c; tested in rys_tests.rs |
| Condon-Shortley coefficients | Custom c2s tables | `C2S_L0` ... `C2S_L4` in `c2s.rs` | Already extracted from libcint cart2sph.c, verified in c2s_tests.rs |
| VRR/HRR recurrence | Custom Obara-Saika | `vrr_2e_step_host`, `hrr_step_host` in `obara_saika.rs` | Already implemented and tested |
| Boys function | Custom Fm(t) | `boys_gamma_inc_host` in `math/boys.rs` | Already implemented and tested |
| Gaussian pair data | Custom pair computations | `compute_pdata_host` in `math/pdata.rs` | Already implemented and tested |
| Oracle reference values | Manual expected values | `cintx_oracle::vendor_ffi::vendor_int*_sph()` via vendored libcint | Vendor build already wired; extend FFI only |

**Key insight:** Every math building block for these four kernels is already in the codebase. The work is assembly and wiring, not new algorithm development. The only net-new code is: the G-tensor fill loops per family (directly from libcint C source), the multi-index c2s variants, the rys_root3-5 host wrappers, and the oracle FFI extensions.

---

## Common Pitfalls

### Pitfall 1: Wrong G-Tensor Stride for 2e (ibase/kbase branch)
**What goes wrong:** The 2e G-tensor layout in libcint uses an adaptive stride selection — when `li_ceil > lj_ceil`, VRR builds on i (bra), otherwise on j; same for k/l. This affects `dli`, `dlj`, `dlk`, `dll` and all derived strides. Using a fixed "bra always larger" assumption produces wrong indices.
**Why it happens:** The libcint `CINTinit_int2e_EnvVars` ibase/kbase branches are easy to miss.
**How to avoid:** Implement both branches explicitly, or simplify by always VRR-ing on the larger center (pick max(li_ceil, lj_ceil) for dli and similarly for kl pair). The 1e kernel simplifies by always VRRing on bra then HRR to ket — apply the same simplification for 2e if it makes indexing clearer. Document the choice.
**Warning signs:** Test values for p-shell ERIs that differ from oracle by a constant factor or index permutation.

### Pitfall 2: Missing common_fac_sp for the kl Pair
**What goes wrong:** 2e has four `common_fac_sp` factors (one per center), while the 1e kernel only has two. Forgetting the kl pair's `common_fac_sp` contribution gives integrals that are wrong by a constant for k or l p-shells.
**Why it happens:** The `common_factor` formula in `g2e.c` line 54-56 multiplies all four `CINTcommon_fac_sp` values.
**How to avoid:** Apply `common_fac_sp(li) * common_fac_sp(lj) * common_fac_sp(lk) * common_fac_sp(ll)` to the base gz0 value before the primitive loop contribution, or apply per-center scale factors immediately after the G-fill.

### Pitfall 3: rys_root Host Wrappers Missing for N>2
**What goes wrong:** 2e with H2O STO-3G p-shells requires 3 Rys roots. `rys_root3_host` does not exist yet. Calling the `#[cube]` `rys_root3` directly from host code will panic at runtime without a CubeCL context.
**Why it happens:** Only `rys_root1_host` and `rys_root2_host` were added in Phase 8/9. The `rys_roots` CubeCL dispatcher exists but cannot be called from host tests.
**How to avoid:** Add `rys_root3_host`, `rys_root4_host`, `rys_root5_host`, and a `rys_roots_host(nroots: usize, x: f64) -> (Vec<f64>, Vec<f64>)` dispatcher in `rys.rs` before starting any 2e/2c2e kernel work. These are mechanical ports of the same logic — copy the `#[cube]` function body, remove CubeCL attributes, change `Array<f64>` to `&mut [f64]` slice.

### Pitfall 4: 3c2e Reuses 2e Machinery with a Non-Obvious Index Swap
**What goes wrong:** `g3c2e.c` sets `lk_ceil=0` and `ll_ceil=k_l+ng[KINC]` to reuse `CINTg0_2e_lj2d4d`. This means the "l" slot in the 2e G-tensor fill is actually the third-center k shell of 3c2e. Using the 2e indexing naively (expecting k and l to both be real shells) gives wrong results.
**Why it happens:** The comment in `g3c2e.c` line 18 warns about this: "Note the 3c2e functions takes i,j,k parameters. But we initialize ll_ceil, to reuse g2e_g02d function."
**How to avoid:** In the Rust implementation, the 3c2e G-fill should explicitly set the "kl" side to (shell_k, s-function) and the "ij" side to (shell_i, shell_j). The third real shell k maps to the `ll_ceil` slot. Implement this explicitly rather than copying the 2e fill verbatim.

### Pitfall 5: 3c1e Nuclear Term Uses Rys + Boys, Overlap Term Does Not
**What goes wrong:** `int3c1e_sph` evaluation requires both the overlap G-fill (no Rys) AND a nuclear attraction G-fill (with Rys + Boys integration over atoms). Implementing only the overlap G-fill produces wrong integrals for any operator that has nuclear character.
**Why it happens:** The 3c1e family has both overlap-type and nuclear-type operators. `CINTg3c1e_ovlp` and `CINTg3c1e_nuc` are separate functions in `g3c1e.c`.
**How to avoid:** Start with the overlap operator as the smoke test case. Implement `CINTg3c1e_nuc` support before claiming oracle parity on `int3c1e_nuc_sph` type operators. The `int3c1e_sph` family's default operator is overlap-type, so that should suffice for the oracle gate.

### Pitfall 6: Oracle Build Missing 2e+ Source Files
**What goes wrong:** The oracle `build.rs` currently lists only 1e source files. Building with `CINTX_ORACLE_BUILD_VENDOR=1` will produce a library missing `int2e_sph` etc., causing link errors when `vendor_ffi.rs` tries to call them.
**Why it happens:** The build.rs was written in Phase 9 for 1e only. The comment "D-11: Wheeler fallback deferred to Phase 10" in `rys.rs` also references this.
**How to avoid:** Add the 2e+ source files to `build.rs` BEFORE adding the `vendor_ffi.rs` FFI wrappers. Check which autocode files contain the generated `int2e_sph`, `int2c2e_sph`, etc. implementations — they may be in `src/autocode/intor2.c`, `intor3.c`, `intor4.c` rather than the main driver files.

### Pitfall 7: 2e G-Tensor Index Ordering Mismatch with libcint Output Layout
**What goes wrong:** libcint's `int2e_sph` output buffer has a specific ordering: `out[i,j,k,l]` with i fastest. The Rust kernel must write to staging in the same ordering for oracle comparison to match.
**Why it happens:** `CINTg2e_index_xyz` in `g2e.c` lines 174-266 shows the exact loop nest: j outer, l, k, then i (with i hardcoded for l=0,1,2 and generic for l>=3).
**How to avoid:** The contraction loop must produce elements in `[i,j,k,l]` libcint order (i innermost, j outermost in the output sense) which maps to `out[cj_idx * nfk*nfl*nfi + cl_idx * nfk*nfi + ck_idx * nfi + ci_idx]` or similar. Verify by checking that the oracle comparison is element-wise (not just summed), and add a shape/ordering sanity check in the first oracle test.

---

## Code Examples

### Adding rys_root3_host (template for 4/5 as well)

```rust
// Source: crates/cintx-cubecl/src/math/rys.rs (extend after rys_root2_host)
// Mirrors the #[cube] rys_root3 function body — remove CubeCL attributes,
// use fixed-size arrays instead of Array<f64>

pub fn rys_root3_host(x: f64) -> ([f64; 3], [f64; 3]) {
    let mut u = [0.0_f64; 3];
    let mut w = [0.0_f64; 3];
    // ... (copy rys_root3 body, replacing u[n as usize]/w[n as usize] with u[n]/w[n])
    (u, w)
}

pub fn rys_roots_host(nroots: usize, x: f64) -> (Vec<f64>, Vec<f64>) {
    match nroots {
        1 => { let (u, w) = rys_root1_host(x); (vec![u], vec![w]) }
        2 => { let (u, w) = rys_root2_host(x); (u.to_vec(), w.to_vec()) }
        3 => { let (u, w) = rys_root3_host(x); (u.to_vec(), w.to_vec()) }
        4 => { let (u, w) = rys_root4_host(x); (u.to_vec(), w.to_vec()) }
        5 => { let (u, w) = rys_root5_host(x); (u.to_vec(), w.to_vec()) }
        _ => panic!("rys_roots_host: nroots={nroots} > 5 not yet supported"),
    }
}
```

### 2c2e G-Fill Core (from g2c2e.c, simplified)

```rust
// Source: libcint-master/src/g2c2e.c, g2e.c CINTg0_2e_2d
// 2c2e reuses the 2e G-fill with j_l=0, l_l=0.
// Only one pair product: ij-pair uses shell i; kl-pair uses shell k.
// g_size = nrys_roots * (li+1) * (lk+1)
// g[root * 1 + i_level * nrys_roots + k_level * nrys_roots*(li+1)]

fn fill_g_tensor_2c2e(
    ai: f64, ak: f64,        // exponents for shells i and k
    ri: [f64; 3], rk: [f64; 3],
    li: u32, lk: u32,
    nrys_roots: usize,
    x_rys: f64,              // Rys argument = aik2_prefactor
    gz0: f64,                // normalization prefactor * weight
) -> Vec<f64> {
    // Flat array [gx | gy | gz], each of size nrys_roots * (li+1) * (lk+1)
    // ... fill using vrr_2e_step_host per root, then hrr-like shift for k
}
```

### Oracle Parity Test Template (2c2e)

```rust
// Source: crates/cintx-oracle/tests/center_2c2e_parity.rs
#![cfg(feature = "cpu")]

#[test]
fn oracle_parity_int2c2e_sph_h2o_sto3g() {
    // Build H2O STO-3G atm/bas/env as in one_electron_parity.rs
    let (atm, bas, env) = build_h2o_sto3g();
    let natm = 3_i32;
    let nbas = bas.len() as i32 / BAS_SLOTS as i32;

    // Compare eval_raw (cintx path) vs vendor_int2c2e_sph (libcint path)
    // for all shell pairs (i, k)
    let mut mismatch_count = 0_usize;
    for i_sh in 0..nbas {
        for k_sh in 0..nbas {
            let shls = [i_sh, k_sh];
            let ni = /* CINTcgto_spheric(i_sh) */ ...;
            let nk = /* CINTcgto_spheric(k_sh) */ ...;
            let mut ref_out = vec![0.0_f64; (ni * nk) as usize];
            let mut cintx_out = vec![0.0_f64; (ni * nk) as usize];
            // ... compare with atol 1e-9
        }
    }
    assert_eq!(mismatch_count, 0, "oracle parity mismatch for int2c2e_sph");
}
```

### Oracle Build Extension (build.rs additions)

```rust
// Source: crates/cintx-oracle/build.rs (extend existing cc::Build chain)
build
    .file(libcint_root.join("src/cint2e.c"))
    .file(libcint_root.join("src/g2e.c"))
    .file(libcint_root.join("src/cint2c2e.c"))
    .file(libcint_root.join("src/g2c2e.c"))
    .file(libcint_root.join("src/cint3c1e.c"))
    .file(libcint_root.join("src/g3c1e.c"))
    .file(libcint_root.join("src/cint3c2e.c"))
    .file(libcint_root.join("src/g3c2e.c"))
    // autocode files for generated int2e_sph etc. (verify paths in libcint-master/src/autocode/)
    .file(libcint_root.join("src/autocode/int2e.c")); // or intor2.c — verify
```

Also add `rerun-if-changed` entries for each new source file.

---

## State of the Art

| Old State | New State After Phase 10 | Impact |
|-----------|--------------------------|--------|
| 2e/2c2e/3c1e/3c2e stubs return zeros | All four families produce libcint-compatible values | Oracle gate can close |
| Oracle only covers 1e families | Oracle covers all five base families | v1.1 milestone complete |
| rys_root1_host/2_host only | rys_root1_host through rys_root5_host + dispatcher | 2e d-shell coverage possible |
| cart_to_sph_1e only | 2e/2c2e/3c1e/3c2e c2s variants | Sph representation complete for all families |
| UAT items open | eval_raw non-zero + C ABI status==0 verified | v1.0 human UAT closed |

---

## Open Questions

1. **autocode file names for int2e_sph etc.**
   - What we know: `libcint-master/src/autocode/` contains `intor1.c` (confirmed in build.rs). The auto-generated 2e integrals are in separate autocode files.
   - What's unclear: Exact filenames (`intor2.c`? `int2e.c`? something else?).
   - Recommendation: `ls /home/chemtech/workspace/cintx/libcint-master/src/autocode/` before writing the build.rs extension. The planner should make this a Wave 0 task.

2. **rys_root3_host through rys_root5_host: copy strategy vs dispatch into #[cube]**
   - What we know: The `#[cube]` rys_root3/4/5 functions are fully implemented and tested.
   - What's unclear: Whether the CubeCL CPU backend can be invoked from integration tests without the `cond_br` MLIR issue (Phase 8 found it broken for function calls).
   - Recommendation: Add pure host wrappers by copying the `#[cube]` function body with CubeCL annotations removed. This is the same approach Phase 8/9 established (D-10).

3. **C ABI UAT item — GPU vs CPU backend**
   - What we know: D-07 says "C ABI cintrs_eval() test under CPU backend asserts status == 0". The success criteria say "on a real GPU evaluation".
   - What's unclear: Whether the CI environment has a GPU available for the UAT test.
   - Recommendation: The test should run under `--features cpu` (as D-11 specifies), and the success criteria "real GPU" should be interpreted as "real non-stub kernel path" not "physical GPU hardware". The CPU backend satisfies the "non-stub" requirement.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain 1.94.0 | All kernels | Via rust-toolchain.toml | 1.94.0 | None needed |
| CPU feature (cubecl/cpu) | Oracle parity tests | Available via `--features cpu` | cubecl 0.9.x | None needed |
| cc crate (C compiler) | Oracle vendor build | gcc/clang on Linux | Present | None needed |
| bindgen / clang | Oracle FFI generation | Present (used in Phase 9) | bindgen 0.71.1 | None needed |
| CINTX_ORACLE_BUILD_VENDOR env | Vendor libcint compilation | Set manually or in CI | Gate variable | Skip vendor tests without it |
| libcint-master/ source tree | Oracle compilation | Present at workspace root | 6.1.3 | Not needed without vendor gate |

**No blocking missing dependencies.**

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in (`cargo test`) + `cargo nextest` |
| Config file | None — feature-gated `#[cfg(feature = "cpu")]` per test file |
| Quick run command | `cargo test -p cintx-cubecl -- --test-thread=1` |
| Full suite command | `cargo test --features cpu -p cintx-oracle -p cintx-cubecl -- --test-thread=1` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| KERN-02 | 2e produces non-zero Rys results | integration | `cargo test -p cintx-cubecl --features cpu` | ❌ Wave 0 |
| KERN-02 | 2e oracle parity vs libcint (H2O STO-3G) | oracle | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu` | ❌ Wave 0 |
| KERN-03 | 2c2e oracle parity vs libcint | oracle | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu` | ❌ Wave 0 |
| KERN-04 | 3c1e oracle parity vs libcint | oracle | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu` | ❌ Wave 0 |
| KERN-05 | 3c2e oracle parity vs libcint | oracle | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu` | ❌ Wave 0 |
| VERI-05 | Per-family oracle parity as each kernel lands | oracle | (run after each plan) | ❌ Wave 0 |
| VERI-07 | eval_raw non-zero on H2O STO-3G int1e_ovlp_sph | integration | `cargo test -p cintx-oracle --features cpu` | ❌ Wave 0 (verify existing path works end-to-end) |
| VERI-07 | C ABI cintrs_eval status==0 under CPU | integration | `cargo test -p cintx-capi --features cpu` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `cargo test -p cintx-cubecl` (no oracle; fast)
- **Per wave merge:** `cargo test -p cintx-cubecl --features cpu && CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `crates/cintx-cubecl/src/math/rys.rs` — add `rys_root3_host`, `rys_root4_host`, `rys_root5_host`, `rys_roots_host` dispatcher
- [ ] `crates/cintx-cubecl/src/transform/c2s.rs` — add `cart_to_sph_2e`, `cart_to_sph_2c2e`, `cart_to_sph_3c1e`, `cart_to_sph_3c2e`
- [ ] `crates/cintx-oracle/build.rs` — extend source list and bindgen allowlist for 2e+ families; add rerun-if-changed entries
- [ ] `crates/cintx-oracle/src/vendor_ffi.rs` — add `vendor_int2e_sph`, `vendor_int2c2e_sph`, `vendor_int3c1e_sph`, `vendor_int3c2e_sph` wrappers
- [ ] Verify `libcint-master/src/autocode/` filenames before writing build.rs extension

---

## Sources

### Primary (HIGH confidence)
- `libcint-master/src/g2e.c` — CINTinit_int2e_EnvVars, CINTg2e_index_xyz, CINTg0_2e_2d (lines verified by direct read)
- `libcint-master/src/g2c2e.c` — CINTinit_int2c2e_EnvVars (full file read)
- `libcint-master/src/g3c1e.c` — CINTinit_int3c1e_EnvVars, CINTg3c1e_ovlp, CINTg3c1e_nuc (full file read)
- `libcint-master/src/g3c2e.c` — CINTinit_int3c2e_EnvVars (full file read)
- `crates/cintx-cubecl/src/kernels/one_electron.rs` — working 1e pipeline template (read)
- `crates/cintx-cubecl/src/math/rys.rs` — confirmed rys_root1_host/rys_root2_host exist; root3-5 CubeCL only
- `crates/cintx-cubecl/src/transform/c2s.rs` — confirmed cart_to_sph_1e only; C2S_L0-L4 coefficients present
- `crates/cintx-oracle/build.rs` — confirmed 1e-only source list (read)
- `crates/cintx-oracle/src/vendor_ffi.rs` — confirmed 1e-only FFI wrappers (read)
- `crates/cintx-oracle/tests/one_electron_parity.rs` — oracle test template (read)
- `crates/cintx-cubecl/src/kernels/mod.rs` — FamilyLaunchFn dispatch wiring (read)
- `crates/cintx-oracle/src/compare.rs` — tolerance constants confirmed (TOL_2E_ATOL=1e-12, etc.)
- `.planning/phases/10-2e-2c2e-3c1e-3c2e-real-kernels-and-oracle-gate-closure/10-CONTEXT.md` — decisions locked

### Secondary (MEDIUM confidence)
- Rys quadrature root count formula from `g2e.c` line 74 and `g3c1e.c` line 53 — cross-verified against `g3c2e.c` line 70

---

## Metadata

**Confidence breakdown:**
- Kernel architecture patterns: HIGH — read directly from libcint source and Phase 9 implementation
- rys_root host wrapper gap: HIGH — code search confirmed no rys_root3-5 host functions exist
- c2s variants needed: HIGH — confirmed cart_to_sph_1e is the only variant in c2s.rs
- oracle build extension: HIGH — build.rs read confirms 1e-only source list
- autocode file names: LOW — need to `ls` the autocode directory to confirm exact filenames

**Research date:** 2026-04-03
**Valid until:** 2026-05-03 (libcint 6.1.3 is stable; no upstream churn expected)
