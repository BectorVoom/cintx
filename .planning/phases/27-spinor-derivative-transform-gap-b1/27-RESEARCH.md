# Phase 27: Spinor-Derivative Transform (Gap B1) - Research

**Researched:** 2026-05-31
**Domain:** cart→spinor transform extension (spin-free derivative axis-fold); libcint 6.1.3 byte-identity parity
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Flip **all arity-2 1e spin-free `ip` families** that fold through `cart_to_spinor_sf_2d` — every rank tier present: rank-3 (`int1e_ipovlp/ipkin/ipnuc/iprinv_spinor`), rank-9 both-side (`ipovlpip/ipkinip/ipnucip`), and the second/higher-order set (`ipipovlp/ipipnuc/ipipkin/ipiprinv`, `ipipipnuc/ipipiprinv/ipipnucip/ipiprinvip/ipipipiprinv/ipiprinvipip/ipipiprinvip`). All share one transform addition; all their scalar cart/sph kernels already land (phases 21/23/25).
- **D-02:** Also flip the **arity-3** families (`int3c2e_ip1/ip2_spinor`, `int3c1e_ip1/iprinv_spinor`) and **2c2e** families (`int2c2e_ip1/ip2_spinor`). Arity-3 needs a derivative wrapper over `cart_to_spinor_sf_3c2e`; 2c2e folds through the `sf_2d` path. All their scalar cart forms are confirmed `oracle_covered=true`.
- **D-03 (DEFERRED):** The **arity-4 `int2e_ip*` set** is deferred — requires a NEW derivative variant of `cart_to_spinor_sf_4d`. See Deferred Ideas.
- **D-04 (DEFERRED):** `int1e_ecp_iprinv_spinor` stays `UnsupportedApi` — only ip-spinor family that is not pure spin-free (R5/ECP-spinor track).
- **D-05:** Add **thin generic wrappers** `cart_to_spinor_sf_derivative_2d(staging, cart, ncomp, li, kappa_i, lj, kappa_j)` and a `_3c2e` sibling that **loop the already-verified per-component transforms** (`cart_to_spinor_sf_2d` / `cart_to_spinor_sf_3c2e`) `ncomp` times with correct strides. `ncomp ∈ {3, 9, 27, 81}`. Mirrors libcint (no derivative-specific c2s function; the driver loops components).
- **D-06:** The KET→BRA **transpose is owned INSIDE the wrapper**. Each per-component device-native (`[comp][ket][bra]`, KET-major) cart sub-block is transposed to BRA-major internally before the inner `sf_2d`/`sf_3c2e` call. Callers pass device-native blocks.
- **D-07:** Output uses **component-major-outer** layout — for each derivative component, a full `di*dj*2` (complex, interleaved `[re,im,…]`) spinor block, components slowest-varying (`out + comp * di*dj*2`). Matches libcint `int1e` driver output stride.
- **D-08:** The dedicated `vendor_*` spinor parity fixture is **maximally adversarial in a single shell tuple**: **non-square** bra/ket angular momenta (e.g. p×d); **at least one shell with nctr>1**; **kappa=0** (both GT and LT blocks fire, `di = 4l+2`). One fixture covers all three known landmines.
- **D-09:** A dedicated vendor byte-identity test **per transform-path × per rank-tier exercised**: `sf_2d` at rank-3 (`int1e_ipovlp_spinor`), rank-9 (`int1e_ipovlpip_spinor`), highest rank present (27/81, an `ipip*` family); `sf_3c2e` at rank-3 (`int3c2e_ip1_spinor`). Each uses the D-08 fixture.
- **D-10:** Remaining in-scope families flip via the manifest, guarded by a **no-silent-skip coverage assertion** — verifies every flipped family is `oracle_covered=true` AND that the vendor parity tests actually execute (`running N>0 tests`, not `skipped`) under both required flags. `manifest-audit` must be green.
- **D-11:** Run a **full design spike** (`/gsd:spike`) before plan tasks are finalized — exercise the full per-component axis-fold across all rank tiers and both paths. The single genuine residual unknown: the exact device-emitted derivative cart block layout (confirm `[comp][ket][bra]` component-outer) and the precise per-component stride into staging, verified against hand-checked vendor values.

### Claude's Discretion
- Exact molecule/basis for the fixture (subject to D-08 hard constraints: non-square, nctr>1 somewhere, kappa=0).
- Internal stride arithmetic and loop factoring within the wrappers.
- Whether `int3c1e` shares the `sf_3c2e` derivative wrapper or needs a thin sibling (resolve from the actual block layout during the spike).
- Plan boundaries between the `sf_2d` path families and the `sf_3c2e` path families.

### Deferred Ideas (OUT OF SCOPE)
- **Arity-4 `int2e_ip*` spinor families** — require a derivative variant of `cart_to_spinor_sf_4d`. Their scalar cart forms are already covered; unblocked whenever the `sf_4d` derivative wrapper lands.
- **`int1e_ecp_iprinv_spinor`** — spinor ECP gradient; not pure spin-free; R5/ECP-spinor track (Phase 29).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FND-04 | Spinor-derivative transform (Gap B1): `cart_to_spinor_sf_derivative_*` in `c2spinor.rs`; `int1e_ipovlp_spinor` and sibling `ip`-decorated spin-free spinor families move from `UnsupportedApi` to byte-identity at atol=1e-12 (closes Phase-21 R5/D-03). | The inner per-component transforms (`cart_to_spinor_sf_2d` L531, `cart_to_spinor_sf_3c2e` L1281) are already byte-identity-proven; the rank-3 1e gradient spinor path is already wired inline in the launcher with a passing parity test. The phase consolidates that inline transpose into the D-06 wrapper, extends to all rank tiers + the 3c2e/2c2e/3c1e paths, adds the adversarial fixture + missing vendor FFI, and flips coverage. |
</phase_requirements>

## Summary

This phase has an unusually low residual-unknown surface because the hard part is already done and proven. The per-component cart→spinor coupling (`cart_to_spinor_sf_2d`, `cart_to_spinor_sf_3c2e`) is implemented and byte-identity-verified; the rank-3 1e gradient spinor families (`int1e_ipovlp/ipkin/ipnuc/iprinv_spinor`) are **already evaluated** in `one_electron.rs` (~L9919-9965) by an inline per-component loop that transposes KET→BRA and calls `cart_to_spinor_sf_2d` once per gradient component, and they **already pass** a vendor parity test (`one_electron_grad_spinor_parity.rs`). Despite this, the lock still records `oracle_covered=false` for every ip-spinor family. The gap between "works" and "covered" is exactly what this phase closes.

The work is therefore three things, not one: (1) **refactor** the inline transpose+loop into the D-05/D-06 `cart_to_spinor_sf_derivative_2d/_3c2e` wrappers in `c2spinor.rs` so the KET→BRA transpose lives in exactly one audited place; (2) **extend** the spinor derivative path to the rank-9/27/81 1e arms and the 3c2e/2c2e/3c1e `ip` arms — all of which currently hard-reject `Representation::Spinor` with `UnsupportedApi`; and (3) **harden verification** by building the D-08 adversarial fixture (non-square + nctr>1 + kappa=0), adding the missing vendor FFI bindings (only the four rank-3 1e ip-spinor vendor functions exist today), writing the D-09 per-path × per-rank parity tests, and flipping `oracle_covered=true` behind the D-10 no-silent-skip assertion.

The single genuine empirical unknown the D-11 spike must nail is the exact device-emitted derivative cart block layout at each rank tier and on the 3c2e path — specifically confirming `[comp][ket][bra]` component-outer with per-component stride `block_len = nci*ncj` (1e) / `nci*ncj*nck` (3c2e), verified against hand-checked vendor values on a non-square block. The existing rank-3 inline code asserts this layout implicitly via a passing square-block test (H2O STO-3G), which is orientation-blind — so the spike's non-square verification is load-bearing, not ceremonial.

**Primary recommendation:** Add two thin generic wrappers in `c2spinor.rs` that loop the existing inner transforms `ncomp` times with `comp_stride = di*dj*2` (1e) / `di*dj*nsk*2` (3c2e), transposing each per-component KET-major cart sub-block to BRA-major inside the wrapper (D-06); replace the inline launcher arms with calls to these wrappers; lift the `nctr>1` rejection for the spinor derivative path (required by D-08); add the missing vendor FFI; verify on a non-square + nctr>1 + kappa=0 fixture; flip coverage behind a no-silent-skip assertion.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Per-component cart→spinor coupling (CG matrices) | Host transform (`c2spinor.rs`) | — | libcint convention is host-side; CubeCL emits cart blocks, the c2s fold stays host per project convention (matches scalar/gradient spinor and `cart_to_sph_*`). |
| Derivative-component axis fold (the `ncomp` loop) | Host transform (`c2spinor.rs` new wrappers) | — | D-05: thin loop over the inner transform; mirrors libcint's `int1e`/`int3c2e` driver loop. No device work. |
| KET→BRA transpose of each component block | Host transform (inside wrapper, D-06) | — | Centralizes the scalar-spinor orientation landmine in one audited place; launchers must not own it. |
| Device-emitted derivative cart blocks (`[comp][ket][bra]`) | CubeCL kernel (already landed, phases 21/23/25) | — | Scalar/gradient cart kernels already produce these; out of scope to change. |
| Launcher dispatch (call the wrapper, size staging) | Host launcher (`one_electron.rs`, `center_3c2e.rs`, `center_2c2e.rs`) | — | Each `Representation::Spinor` derivative arm calls the new wrapper instead of rejecting. |
| Manifest coverage flip + audit | Lock (`compiled_manifest.lock.json`) + xtask `manifest-audit` | `oracle_covered_update.rs` | Lock is source of truth; audit auto-syncs both sides. D-04-deferred families stay `false`. |
| Vendor byte-identity verification | Oracle (`vendor_ffi.rs`, `compare.rs`, `fixtures.rs`, `tests/*`) | — | Adds missing FFI, the adversarial fixture, and the per-path×per-rank parity tests. |

## Standard Stack

This is an intra-project extension; no new external dependencies. Verified versions from `Cargo.toml` / `CLAUDE.md`.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cubecl` | `0.10.0` | Device kernels that emit the derivative cart blocks (already landed) | [VERIFIED: Cargo.toml L10] Project-pinned compute backend. No kernel change needed this phase. |
| `thiserror` | `2.0.18` | `cintxRsError::UnsupportedApi` / `BufferTooSmall` typed errors | [CITED: CLAUDE.md] Public library error surface; the new wrappers return these. |
| `num-complex` | (workspace) | Conceptual model for interleaved complex spinor output | [CITED: CLAUDE.md] Spinor staging is raw interleaved `[re,im,…]` f64; oracle compares the flat buffer directly. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `anyhow` | `1.0.102` | xtask/oracle harness errors | [CITED: CLAUDE.md] `manifest-audit`, `oracle-covered-update`, parity-report tooling. |

**Installation:** None — all dependencies already in the workspace.

## Architecture Patterns

### System Architecture Diagram (derivative spinor data flow)

```
eval_raw(RawApiId::INT1E_IPOVLP_SPINOR, ...)
   │
   ▼
launcher (one_electron.rs / center_3c2e.rs / center_2c2e.rs)
   │  builds Plan{ representation: Spinor }
   ▼
CubeCL device kernel (already landed)
   │  emits cart_Ncomp buffer:  [comp][ket][bra]  (KET-major, bra-fastest)
   │  comp_stride = nci*ncj (1e) | nci*ncj*nck (3c2e)
   ▼
Representation::Spinor arm  ── currently: inline loop (rank-3 1e) OR UnsupportedApi (everything else)
   │                            after this phase: call the wrapper
   ▼
cart_to_spinor_sf_derivative_2d(staging, cart, ncomp, li, ki, lj, kj)   [NEW, c2spinor.rs]
   │
   ├─ for comp in 0..ncomp:
   │     src  = cart[comp*block_len .. ]                 # device-native KET-major slice
   │     ┌─ TRANSPOSE src  [ket][bra] → [bra][ket]       # D-06: owned here, not in launcher
   │     │     block_bra_major[ic*ncj + jc] = src[jc*nci + ic]
   │     └─ cart_to_spinor_sf_2d(                          # EXISTING, byte-identity-proven
   │            &mut staging[comp*di*dj*2 .. +di*dj*2],    # D-07: component-outer stride
   │            block_bra_major, li, ki, lj, kj)
   ▼
staging: interleaved-complex, component-leading
   out[comp*(di*dj)*2 + (j*di + i)*2 + {0:re,1:im}]
   ▼
oracle compare.rs  ── element-wise vs vendor_int*_spinor at atol=1e-12
```

### Recommended Project Structure (files touched)

```
crates/cintx-cubecl/src/transform/
└── c2spinor.rs          # ADD cart_to_spinor_sf_derivative_2d / _3c2e wrappers
                         #   (after cart_to_spinor_sf_3c2e ~L1346, before tests mod)
crates/cintx-cubecl/src/kernels/
├── one_electron.rs      # REPLACE inline rank-3 transpose-loop (~L9919-9965) with wrapper call;
│                        #   REPLACE the rank-9/27/81 UnsupportedApi spinor rejections
│                        #   (~L8735, 8812, 9018, 9128, 9212, 9302, 9455, 9594) with wrapper calls
├── center_3c2e.rs       # REPLACE int3c2e_ip1/ip2 spinor UnsupportedApi (~L2394, 2632, 2868) with _3c2e wrapper
└── center_2c2e.rs       # REPLACE int2c2e_ip1/ip2 spinor UnsupportedApi (~L629) with _2d wrapper
                         # NOTE: int3c1e_ip1/iprinv launcher also needs the _3c2e (or sibling) wrapper
crates/cintx-oracle/src/
├── vendor_ffi.rs        # ADD extern decls + wrappers for rank-9/27/81 1e ip-spinor,
│                        #   int3c2e_ip1/ip2_spinor, int2c2e_ip1/ip2_spinor, int3c1e_ip1/iprinv_spinor
├── compare.rs           # ensure symbol→RawApiId map + spinor dispatch covers flipped families
└── fixtures.rs          # ADD D-08 adversarial fixture (non-square + nctr>1 + kappa=0)
crates/cintx-oracle/tests/
└── (new) spinor_deriv_parity.rs   # D-09 per-path×per-rank parity + D-10 no-silent-skip assertion
crates/cintx-ops/generated/compiled_manifest.lock.json   # flip oracle_covered=true for in-scope; leave D-03/D-04 false
xtask/src/oracle_covered_update.rs                       # update the R5/D-03 deferral note (~L20-40)
```

### Pattern 1: Thin per-component derivative wrapper (D-05/D-06/D-07)
**What:** A generic-`F` function looping the proven inner transform `ncomp` times, owning the transpose.
**When to use:** Every spinor derivative arm (1e rank 3/9/27/81, 3c2e, 2c2e, 3c1e).
**Example (synthesized from the existing inline code at one_electron.rs L9919-9965 + sf_2d signature):**
```rust
// Source: derived from crates/cintx-cubecl/src/kernels/one_electron.rs L9937-9964
//         + crates/cintx-cubecl/src/transform/c2spinor.rs L531 (cart_to_spinor_sf_2d)
pub fn cart_to_spinor_sf_derivative_2d<F: CintFloat>(
    staging: &mut [F],
    cart: &[f64],            // device-native: [comp][ket][bra], comp_stride = nci*ncj
    ncomp: usize,            // 3 | 9 | 27 | 81
    li: u8, kappa_i: i16,
    lj: u8, kappa_j: i16,
) -> Result<(), cintxRsError> {
    let nci = ncart(li);
    let ncj = ncart(lj);
    let block_len = nci * ncj;
    let di = spinor_len(li, kappa_i as i32);
    let dj = spinor_len(lj, kappa_j as i32);
    let spinor_block = di * dj * 2;                       // D-07 component-outer stride
    // fail-closed size check (FND-06 lesson: full-block sizing, no `if dst<len` guard)
    if cart.len() < ncomp * block_len { /* ChunkPlanFailed */ }
    if staging.len() < ncomp * spinor_block { /* BufferTooSmall */ }
    for comp in 0..ncomp {
        let src = &cart[comp * block_len .. comp * block_len + block_len];
        // D-06: KET→BRA transpose owned HERE (block[jc*nci+ic] → bra_major[ic*ncj+jc])
        let mut block_bra_major = vec![0.0f64; block_len];
        for ic in 0..nci { for jc in 0..ncj {
            block_bra_major[ic * ncj + jc] = src[jc * nci + ic];
        }}
        let base = comp * spinor_block;
        cart_to_spinor_sf_2d::<F>(
            &mut staging[base .. base + spinor_block],
            &block_bra_major, li, kappa_i, lj, kappa_j)?;
    }
    Ok(())
}
```
The `_3c2e` sibling is identical except `block_len = nci*ncj*nck`, the inner call is `cart_to_spinor_sf_3c2e` (which internally folds the k-axis cart→sph and applies `sf_2d` per k-slice — see L1337-1343), and `spinor_block = di * dj * nsk * 2`. **The transpose granularity on the 3c2e path is the open question the spike must resolve** (transpose the full `[ket][bra]` per-(comp,k), or per-comp before the k-fold) — see Open Questions Q1.

### Anti-Patterns to Avoid
- **Re-adding `if dst < staging.len()` scatter guards.** Family kernels are monolithic whole-block writers; staging must be full-block sized and the wrapper must size-check upfront then write unconditionally (FND-06 / Phase-25 lesson; spike reference `device-block-layout.md` "What to Avoid").
- **Per-rank duplicate functions.** D-05 is explicit: the coupling matrix is identical across ranks; only the loop count differs. One `_2d` + one `_3c2e` wrapper, parameterized by `ncomp`.
- **Owning the transpose in launchers.** D-06: the moment any launcher re-implements the KET→BRA transpose, the orientation landmine reopens. Centralize in the wrapper; launchers pass device-native blocks.
- **Verifying on a square block.** A p×p (or any s-side) block is transpose-symmetric and the i/j swap is unobservable — this is why the existing rank-3 test passes on H2O STO-3G yet `oracle_covered` was correctly left `false`. D-08 mandates non-square.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| cart→spinor CG coupling | A new per-rank coupling routine | `cart_to_spinor_sf_2d` (L531) / `cart_to_spinor_sf_3c2e` (L1281) | Byte-identity-proven; D-05 mandates reuse. A reimplementation is pure duplication and reopens sign/ordering bugs. |
| spinor component count | Manual `4l+2` arithmetic scattered in launchers | `spinor_len(l, kappa)` (L25) | Already handles kappa<0/>0/==0 (the `4l+2` both-block case). |
| k-axis cart→sph in 3c2e | A bespoke fold | The k-fold already inside `cart_to_spinor_sf_3c2e` (L1316-1332) | The 3c2e inner transform already does cart→sph on k then `sf_2d` per k-slice. |
| nctr>1 contraction layout | A custom contraction-major scatter | The contraction-major composition pinned in spike 005 (`i_global = ci*di + ic`) | Historical nctr-transpose bug class; column-major env vs row-major Shell. **But note: the inner `sf_2d` currently has NO nctr param — see Pitfall 4.** |
| spinor AO offsets / sizing for the test harness | Manual offset math | `vendor_CINTcgto_spinor`, `vendor_CINTshells_spinor_offset` (vendor_ffi.rs L3729, L3773) | Existing vendor helpers already used by `one_electron_grad_spinor_parity.rs`. |

**Key insight:** The entire phase is composition, not new numerics. Every per-component value is already correct; the only failure modes are *stride*, *orientation*, *component count*, and *coverage bookkeeping* — all structural, all caught by the D-08 adversarial fixture + D-10 assertion.

## Runtime State Inventory

> This is a code/manifest change, not a rename or data migration. The one stateful artifact is the manifest lock.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no datastore keys the renamed/flipped families. Verified: no DB/collection references the spinor symbols. | None |
| Live service config | None — verified by grep; no external service config references these integral symbols. | None |
| OS-registered state | None. | None |
| Secrets/env vars | The vendor gate reads env var `CINTX_ORACLE_BUILD_VENDOR=1` (build.rs sets `has_vendor_libcint` cfg) — code reads it by exact name; **not renamed**, only depended on. `--features cpu` likewise. | None (depend on, don't change) |
| Build artifacts | `compiled_manifest.lock.json` (`crates/cintx-ops/generated/`) is the source of truth; the generated `api_manifest.rs` and `RawApiId` consts derive from it. Flipping `oracle_covered` in the lock **auto-syncs** the audit (both sides derive from the lock) [VERIFIED: CONTEXT.md code_context + lock structure]. | Edit lock `oracle_covered: false→true` for in-scope families only; regenerate if a codegen step exists; run `manifest-audit`. |

**Canonical question — after every file is updated, what runtime systems still have the old state?** Only the lock's `oracle_covered` flags and the `oracle_covered_update.rs` deferral note (xtask/src/oracle_covered_update.rs ~L20-40, which currently records spinor gradients as intentionally `skipped`→not-stamped). Both are explicit code/data edits in this phase.

## Common Pitfalls

### Pitfall 1: KET/BRA orientation hidden by a square fixture
**What goes wrong:** Parity passes on H2O STO-3G (the existing `one_electron_grad_spinor_parity.rs` fixture) because every per-shell block is square or s-sided, yet a transposed wrapper would silently produce wrong AO matrices in production.
**Why it happens:** Device blocks are KET-major (`block[jc*nci+ic]`); `cart_to_spinor_sf_2d` reads BRA-major (`cart[n*ncj+j]`, see `apply_bra_block`). Square blocks are transpose-symmetric.
**How to avoid:** D-08 mandates a non-square (p×d) block. The spike's negative control (`to_j_fastest` reindex → `mismatches>0`) decisively proves orientation (spike `device-block-layout.md`).
**Warning signs:** Test passes but only on a square/s-side fixture; `mismatches(vendor, cintx_jf) == 0` (j-fastest should DIVERGE).

### Pitfall 2: Component truncation from a wrong `component_rank` in the lock
**What goes wrong:** A wrong `component_rank` drops trailing components — the staging is under-sized and the last gradient component is silently zero.
**Why it happens:** Historical bug class (CONTEXT.md landmine 3; the `component_rank=1` truncation bug fixed in 260530-9ay). The wrapper's `ncomp` must equal the lock's `component_rank` (3/9/27/81).
**How to avoid:** Verify `component_rank` for every flipped family in `compiled_manifest.lock.json` matches its known rank tier (lock has rank 3 ×106, 9 ×68, 27 ×15, 81 ×15 per spike `device-block-layout.md`). Assert `cart.len() == ncomp * block_len` upfront (fail-closed).
**Warning signs:** A `rank*di*dj*2` buffer that doesn't split into exactly `rank` non-zero slices; one stuck-at-zero trailing component.

### Pitfall 3: Vendor parity silently skipping (false pass)
**What goes wrong:** Without BOTH `--features cpu` AND `CINTX_ORACLE_BUILD_VENDOR=1`, the `#[cfg(has_vendor_libcint)]` parity bodies compile out; only the non-vendor smoke tests run and report green.
**Why it happens:** Double-gated vendor build (CONTEXT.md established pattern; MEMORY: oracle vendor parity invocation). `oracle_covered_update.rs` (L13-20) explicitly skips `fixture.skipped` fixtures and must NOT stamp them — threat T-21-08-02.
**How to avoid:** D-10 no-silent-skip assertion: a test that fails if the vendor arm did not execute (`running N>0 tests`, not a `skipped` fixture). Run the gate with both flags.
**Warning signs:** `running 0 tests` for the parity binary under the gate; `oracle_covered=true` stamped on a family whose fixture was `skipped`.

### Pitfall 4: nctr>1 unsupported in the spinor transform (BLOCKER for D-08)
**What goes wrong:** D-08 requires a shell with `nctr>1`, but **both** existing inline spinor arms (`one_electron.rs` L9925 and L10134) **reject `nctr>1` with `UnsupportedApi`**, and `cart_to_spinor_sf_2d` has **no nctr parameter** — `di`/`dj` are computed from `(l, kappa)` only (L539-542), while vendor `CINTcgto_spinor` includes nctr.
**Why it happens:** The scalar/gradient spinor path was implemented nctr=1-only (no non-relativistic caller needed gc). libcint's spinor output is `nctr * (4l+2)` per shell; the env coeff block is column-major vs cintx row-major (the historical nctr-transpose bug class, spike 005 / MEMORY raw_nctr_coeff_transpose).
**How to avoid:** The wrappers (or a contraction-major loop around them, following spike 005's `i_global = ci*di + ic`) MUST handle nctr>1 to satisfy D-08. This is the **largest concrete code addition** in the phase and a likely plan-task boundary. Confirm the exact nctr composition against vendor during the D-11 spike. Cross-check the `int3c1e_genctr_parity.rs` builder (a `(i=p nctr2, j=d, k=s)` non-square+gc fixture already exists) as a starting point for the D-08 fixture and the nctr-major scatter pattern.
**Warning signs:** `UnsupportedApi` on the nctr>1 fixture; staging sized `di*dj*2` instead of `nctr_i*nctr_j*di*dj*2`; transposed coefficients for the gc columns (column vs row major).

### Pitfall 5: Missing vendor FFI for most flipped families
**What goes wrong:** Only the four rank-3 1e ip-spinor vendor functions exist (`vendor_int1e_ipovlp/ipkin/ipnuc/iprinv_spinor`, vendor_ffi.rs L4189-4288). Rank-9/27/81 1e ip-spinor, `int3c2e_ip1/ip2_spinor`, `int2c2e_ip1/ip2_spinor`, `int3c1e_ip1/iprinv_spinor` have **no vendor FFI** [VERIFIED: grep returned 0 ip-spinor extern decls beyond the four].
**Why it happens:** Vendor FFI was only added for the families that already had a (square-fixture) parity test.
**How to avoid:** Add extern declarations + safe wrappers for every D-09-tested family before writing parity tests. Pattern is mechanical — copy the rank-3 wrapper shape (L4193).
**Warning signs:** Compile error `cannot find function vendor_int1e_ipovlpip_spinor`; a D-09 test that cannot be written.

## Code Examples

### Existing inline rank-3 transform (the code to refactor into the wrapper)
```rust
// Source: crates/cintx-cubecl/src/kernels/one_electron.rs L9937-9964 (verbatim structure)
for comp in 0..3usize {
    let src_base = comp * block_len;
    let block = &cart_3comp[src_base..src_base + block_len];
    let staging_comp_base = comp * spinor_block;          // spinor_block = di*dj*2 (D-07)
    let mut block_bra_major = vec![0.0f64; block_len];     // D-06 transpose, to move into wrapper
    for ic in 0..nci { for jc in 0..ncj {
        block_bra_major[ic * ncj + jc] = block[jc * nci + ic];
    }}
    cart_to_spinor_sf_2d::<F>(
        &mut staging[staging_comp_base..staging_comp_base + spinor_block],
        &block_bra_major, li, shell_i.kappa, lj, shell_j.kappa)?;
}
```

### Spinor output layout contract (verify the wrapper against this)
```
// Source: .claude/skills/spike-findings-cintx/references/spinor-layout.md
out[comp * (ni_sp*nj_sp) * 2 + (j*ni_sp + i)*2 + {0:re, 1:im}]
    comp        : slowest (rank 3/9/27/81)
    (j*ni_sp+i) : ket-major, i (bra) fastest         // ni_sp = 4l+2 at kappa=0
    {re,im}     : fastest axis, per-element pair      // buffer is ×2 the real length
```

### Parity test skeleton (D-09; pattern from the existing passing test)
```rust
// Source: crates/cintx-oracle/tests/one_electron_grad_spinor_parity.rs L367-385
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipovlp_spinor_adversarial_parity() {
    let (atm, bas, env) = build_adversarial_spinor_fixture();   // D-08: p×d, nctr>1, kappa=0
    let vendor = collect_vendor_spinor(vendor_ffi::vendor_int1e_ipovlp_spinor, &atm, &bas, &env);
    let cintx  = collect_cintx_spinor(RawApiId::INT1E_IPOVLP_SPINOR, &atm, &bas, &env);
    assert_any_nonzero(&cintx, "..."); assert_any_nonzero(&vendor, "...");
    assert_eq!(count_mismatches(&vendor, &cintx, 1e-12, 0.0), 0);
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Spinor derivatives → `UnsupportedApi` (R5/D-03 deferral) | Rank-3 1e gradient already evaluated inline; this phase generalizes + flips coverage | Phase 21 deferred; 260529-jtd wired rank-3 1e inline; Phase 27 closes | `int1e_ipovlp_spinor` et al. move to byte-identity |
| Inline per-launcher transpose | D-06 wrapper-owned transpose | This phase | Single audited transpose; no launcher can omit it |
| nctr=1-only spinor | nctr>1 spinor transform (D-08 requirement) | This phase | First nctr>1 spinor parity; closes the gc gap for the sf_2d/3c2e paths |

**Deprecated/outdated:** The `oracle_covered_update.rs` note recording spinor gradients as intentionally `skipped` (not stamped) is current but must be updated this phase as families flip.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The 3c2e derivative transpose granularity (full `[ket][bra]` per-(comp,k) vs per-comp before k-fold) follows the same KET→BRA rule as 1e. | Pattern 1 / Open Q1 | If wrong, `int3c2e_ip1_spinor` parity fails; the D-11 spike is mandated precisely to settle this empirically — do NOT lock the 3c2e wrapper shape before the spike. |
| A2 | `int3c1e_ip1/iprinv_spinor` can share the `_3c2e` wrapper (vs needing a thin sibling). | Structure / D (discretion) | Medium — D explicitly leaves this to spike resolution. Plan must allow a sibling fallback. |
| A3 | nctr>1 spinor output composes as contraction-major `i_global = ci*di + ic` (spike 005 pattern) around the per-component spinor block. | Pitfall 4 | High — this is the largest new code and the historical bug class. Verify against vendor on the D-08 fixture during the spike. |
| A4 | Flipping `oracle_covered` in the lock auto-syncs the audit with no separate fixtures list to edit. | Runtime State Inventory | Low — stated in CONTEXT.md code_context and consistent with prior phases (MEMORY: ipovlpip rank-9). Confirm `manifest-audit` green after the flip. |
| A5 | `component_rank` values for all flipped families are already correct in the lock (no truncation latent). | Pitfall 2 | Medium — verify each rank value matches its tier before relying on `ncomp` from the manifest. |

## Open Questions

1. **3c2e derivative transpose granularity (the D-11 spike's core target).**
   - What we know: `cart_to_spinor_sf_3c2e` (L1281) takes a `[k][j][i]`-ordered cart buffer, folds k cart→sph, then applies `sf_2d` per k-slice. The 1e KET→BRA transpose rule is `block_bra_major[ic*ncj+jc] = src[jc*nci+ic]`.
   - What's unclear: For the derivative form, is the device buffer `[comp][k][ket][bra]` (transpose per (comp,k) sub-block) or `[comp][k][bra][ket]`? And does `int3c2e_ip2` (derivative on the ket/aux center) differ from `ip1`?
   - Recommendation: Resolve empirically in the D-11 spike against hand-checked vendor values on a non-square + nctr>1 block (D-08). Do not finalize the `_3c2e` wrapper signature/transpose until then.

2. **nctr>1 spinor block sizing and coefficient ordering.**
   - What we know: inner `sf_2d` has no nctr param; vendor `CINTcgto_spinor` includes nctr; env coeff is column-major, Shell row-major (spike 005).
   - What's unclear: exact loop nesting (contraction-major outer vs interleaved) and where the row/column-major transpose must happen for the spinor path specifically.
   - Recommendation: Mirror the `int3c1e_genctr_parity.rs` gc builder and spike 005's `i_global = ci*di + ic`; verify on the D-08 fixture. Treat as a distinct plan task (largest blast radius).

3. **Which exact highest-rank family for the D-09 27/81 test.**
   - What we know: rank-27 = `ipipipnuc/ipipiprinv/ipipnucip/ipiprinvip`; rank-81 = `ipipipiprinv/ipiprinvipip/ipipiprinvip`.
   - Recommendation: Pick one rank-81 `int1e_ipip*` spinor family with existing scalar coverage; add its vendor FFI. (Claude's discretion within D-09.)

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain | Build/test | ✓ (assumed pinned) | 1.94.0 (pinned) | — |
| `cubecl` CpuRuntime (`--features cpu`) | Real (non-skipped) oracle parity | ✓ (used by existing spinor parity tests) | 0.10.0 | none — required for parity to run, not skip |
| Vendored libcint 6.1.3 (`CINTX_ORACLE_BUILD_VENDOR=1` → `has_vendor_libcint`) | Byte-identity comparison | ✓ (existing parity tests link it) | 6.1.3 | none — without it, vendor arm compiles out (D-10 guards against silent pass) |
| `cc` / C compiler | Vendored libcint build | ✓ (assumed; existing vendor builds work) | 1.2.x | — |

**Missing dependencies with no fallback:** None for development. **Both** `--features cpu` and `CINTX_ORACLE_BUILD_VENDOR=1` are required at the verification gate — without both, parity silently skips (Pitfall 3). [VERIFIED: MEMORY oracle vendor parity invocation + CONTEXT.md established pattern]

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (cargo test), oracle integration tests in `crates/cintx-oracle/tests/`; `cargo nextest` available |
| Config file | none dedicated (workspace `Cargo.toml`); no `nextest.toml` present [VERIFIED: ls] |
| Quick run command | `cargo test -p cintx-oracle --features cpu --test spinor_deriv_parity` (smoke arms only, no vendor) |
| Full suite command | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test spinor_deriv_parity` (vendor byte-identity) + `cargo run -p xtask -- manifest-audit` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FND-04 | `cart_to_spinor_sf_derivative_2d` folds ncomp axis correctly (rank 3) | integration (vendor parity) | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test spinor_deriv_parity test_int1e_ipovlp_spinor_adversarial_parity` | ❌ Wave 0 (new file) |
| FND-04 | sf_2d rank-9 axis-fold (`int1e_ipovlpip_spinor`) | integration | `... --test spinor_deriv_parity test_int1e_ipovlpip_spinor_adversarial_parity` | ❌ Wave 0 |
| FND-04 | sf_2d highest-rank (27/81 `ipip*`) axis-fold | integration | `... --test spinor_deriv_parity test_int1e_ipip<X>_spinor_adversarial_parity` | ❌ Wave 0 |
| FND-04 | sf_3c2e rank-3 axis-fold (`int3c2e_ip1_spinor`) | integration | `... --test spinor_deriv_parity test_int3c2e_ip1_spinor_adversarial_parity` | ❌ Wave 0 |
| FND-04 | 2c2e via sf_2d (`int2c2e_ip1_spinor`) | integration | `... --test spinor_deriv_parity test_int2c2e_ip1_spinor_adversarial_parity` | ❌ Wave 0 |
| FND-04 | nctr>1 spinor general contraction (D-08) | integration | covered by the adversarial fixture in every test above | ❌ Wave 0 |
| FND-04 | No-silent-skip coverage assertion (D-10) | integration | `... --test spinor_deriv_parity test_no_silent_skip` (asserts `N>0` ran + flipped families `oracle_covered=true`) | ❌ Wave 0 |
| FND-04 | Manifest audit green after flip | xtask | `cargo run -p xtask -- manifest-audit` | ✓ (xtask exists) |
| FND-04 | Wrapper unit transpose/stride correctness | unit | `cargo test -p cintx-cubecl --lib transform::c2spinor` | ✓ (mod exists; add `#[test]`s) |

### Observable Signals (what proves the axis-fold is correct)
- **Per rank tier (3/9/27/81):** a `ncomp*di*dj*2` buffer splits into exactly `ncomp` non-overlapping, all-nonzero component slices (no trailing-zero, no truncation — Pitfall 2).
- **Orientation:** vendor byte-identity on a NON-SQUARE p×d block at atol=1e-12; the j-fastest reindex negative control diverges (`mismatches>0`).
- **Both paths:** `sf_2d` (1e + 2c2e) and `sf_3c2e` (3c2e + possibly 3c1e) each have ≥1 passing parity test.
- **nctr>1:** the gc fixture produces vendor-identical output with contraction-major composition (no coefficient transpose).
- **No-silent-skip:** the parity binary reports `running N>0 tests` under both flags; flipped families read `oracle_covered=true`; `manifest-audit` green.

### Sampling Rate
- **Per task commit:** `cargo test -p cintx-cubecl --lib transform::c2spinor` + `cargo test -p cintx-oracle --features cpu --test spinor_deriv_parity` (fast, smoke).
- **Per wave merge:** `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test spinor_deriv_parity` (full vendor parity).
- **Phase gate:** above full vendor suite green + `cargo run -p xtask -- manifest-audit` green before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `crates/cintx-oracle/tests/spinor_deriv_parity.rs` — D-09 per-path×per-rank parity + D-10 no-silent-skip assertion (new file).
- [ ] `crates/cintx-oracle/src/fixtures.rs` — D-08 adversarial fixture builder (non-square p×d + nctr>1 + kappa=0). Model on `int3c1e_genctr_parity.rs::build_genctr_fixture`.
- [ ] `crates/cintx-oracle/src/vendor_ffi.rs` — extern decls + wrappers for rank-9/27/81 1e ip-spinor, `int3c2e_ip1/ip2_spinor`, `int2c2e_ip1/ip2_spinor`, `int3c1e_ip1/iprinv_spinor` (only the four rank-3 1e exist today).
- [ ] `crates/cintx-cubecl/src/transform/c2spinor.rs` — `#[test]` unit coverage for the new wrappers (stride/transpose).
- [ ] Framework install: none — built-in `cargo test`; `nextest` optional.

## Security Domain

> `security_enforcement` is not configured for this phase and the change is internal numerical library code (no auth, no network, no untrusted input). The relevant safety contract is the project's OOM-safe / fail-closed memory discipline, treated below as the applicable control.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes (buffer sizing) | Upfront fail-closed size assertions in the wrappers (`BufferTooSmall`/`ChunkPlanFailed`) — no partial writes, no `if dst<len` guards (FND-06). |
| V6 Cryptography | no | — |

### Known Threat Patterns for this change
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Silent partial write on under-sized staging | Tampering (data integrity) | Upfront `staging.len() < ncomp*spinor_block` check → typed error before any write (FND-06). |
| False verification claim (stamp coverage on a skipped fixture) | Repudiation | D-10 no-silent-skip assertion; `oracle_covered_update.rs` must not stamp `skipped` fixtures (threat T-21-08-02). |
| Component truncation (wrong rank → dropped data) | Tampering | Pitfall 2: verify `component_rank`, assert exact `cart.len()`/`staging.len()`. |

## Sources

### Primary (HIGH confidence)
- `crates/cintx-cubecl/src/transform/c2spinor.rs` — `spinor_len` (L25), `cart_to_spinor_sf_2d` (L531), `cart_to_spinor_sf_3c2e` (L1281), `apply_bra_block` BRA-major read convention.
- `crates/cintx-cubecl/src/kernels/one_electron.rs` — inline rank-3 spinor gradient transform (L9919-9965), scalar spinor (L10129-10158), the rank-9/27/81 `UnsupportedApi` spinor rejections.
- `crates/cintx-cubecl/src/kernels/center_3c2e.rs` (L2374-2407 reject, L3276 scalar spinor) + `center_2c2e.rs` (L629 reject, L1082 scalar spinor).
- `crates/cintx-oracle/tests/one_electron_grad_spinor_parity.rs` — the existing passing rank-3 1e parity test (square fixture) and the stitch/collect pattern.
- `crates/cintx-oracle/src/vendor_ffi.rs` — only four ip-spinor vendor fns exist (L4189-4288); spinor sizing helpers (L3713-3775).
- `crates/cintx-oracle/src/compare.rs` — atol/rtol=1e-12 (L20-21), symbol→RawApiId map (L298-349), `skipped` semantics (L91-94).
- `crates/cintx-ops/generated/compiled_manifest.lock.json` — `oracle_covered=false` for ip-spinor families (e.g. L3886 ipovlp, L397 int3c2e_ip1); rank tiers.
- `xtask/src/oracle_covered_update.rs` (L13-40) — the skipped-fixture / R5-D-03 deferral note.
- `crates/cintx-oracle/tests/int3c1e_genctr_parity.rs` — existing non-square + nctr>1 gc fixture precedent (`build_genctr_fixture` L56, triple `(p nctr2, d, s)`).
- `.claude/skills/spike-findings-cintx/references/{spinor-layout,device-block-layout}.md` — verified spinor output layout and the orientation/nctr/multi-index verification method.
- `CLAUDE.md` (versions, error-handling policy), `Cargo.toml` (cubecl 0.10.0).

### Secondary (MEDIUM confidence)
- libcint 6.1.3 `c2spinor.c` (`CINTc2s_ket_spinor_sf1`) + `int1e`/`int3c2e` drivers — per-component loop convention (CITED via CONTEXT.md; verify exact stride in the D-11 spike).

### Tertiary (LOW confidence)
- None — all claims grounded in repo code or the validated spike skill.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new deps; versions verified against Cargo.toml/CLAUDE.md.
- Architecture (wrapper shape, 1e path): HIGH — directly mirrors existing passing inline code.
- 3c2e transpose granularity + nctr>1 composition: MEDIUM — the two genuine residual unknowns the D-11 spike must settle empirically.
- Pitfalls/coverage bookkeeping: HIGH — grounded in repo state (vendor FFI gaps, lock flags, skip semantics).

**Research date:** 2026-05-31
**Valid until:** 2026-06-30 (stable internal code; re-verify lock flags if other phases touch the manifest)
