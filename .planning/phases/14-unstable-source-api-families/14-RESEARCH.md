# Phase 14: Unstable-Source-API Families - Research

**Researched:** 2026-04-05
**Domain:** Quantum chemistry integrals — unstable-source family kernel implementation (origi, grids, Breit, origk, ssc)
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Manifest & gating strategy**
- D-01: Add a new `unstable-source` manifest profile alongside base/with-f12/with-4c1e. Source-only symbols only appear when this profile is active.
- D-02: The `unstable-source` profile is standalone only — no cross-product with f12/4c1e profiles. Total profiles: base, with-f12, with-4c1e, with-f12+with-4c1e, unstable-source (5 total).
- D-03: Source-only symbols are explicitly listed in the manifest generator (hardcoded ~18 symbols). No auto-detection from headers.
- D-04: Existing `unstable_source_api_enabled()` gate in `raw.rs` and `is_source_only()` descriptor check remain the runtime enforcement mechanism (established in Phase 3).

**Grids env plumbing**
- D-05: Extend `ExecutionPlan` with `GridsEnvParams` (ngrids, ptr_grids offset) alongside existing `OperatorEnvParams`. Validator rejects grids symbols when NGRIDS=0 or PTR_GRIDS is invalid. Mirrors the F12 zeta plumbing pattern from Phase 13.
- D-06: NGRIDS is an output dimension multiplier. The planner multiplies standard dims by NGRIDS to compute output buffer size. Kernel loops over grid points. Output shape: (ncomp * NGRIDS * di * dj) matching libcint's `g1e_grids.c` behavior.

**Breit spinor-only scope**
- D-07: Breit integrals are spinor-only in Phase 14. Only `int2e_breit_r1p2_spinor` and `int2e_breit_r2p2_spinor` are implemented. Cart and sph representations return `UnsupportedRepresentation` if requested.
- D-08: Breit kernel uses a single composite kernel (Gaunt + gauge computed and summed internally). Matches `breit.c`'s `_int2e_breit_drv` pattern. Single launch, single output buffer.

**Oracle fixture strategy**
- D-09: Single test file `unstable_source_parity.rs` with per-family test functions, all gated behind `#[cfg(feature = "unstable-source-api")]`. ~18 symbols don't warrant separate files.
- D-10: Reuse existing H2O/STO-3G fixture molecule for all unstable families. Grids family adds grid point coordinates to env but uses the same molecule. Consistent with all prior oracle tests.
- D-11: Nightly CI job is an extra job (`unstable_source_oracle`) in the existing CI workflow, gated on schedule/nightly trigger. Reuses existing runner config and artifact paths per Phase 4 conventions.

### Claude's Discretion
- Internal module layout for new kernel files (whether origi/origk share a file or get separate modules)
- Exact `GridsEnvParams` field names and validation logic
- How Breit Gaunt+gauge composition routes through existing 2e infrastructure vs new dispatch arm
- Order of family implementation across plans
- Grid coordinate fixture values for oracle tests

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| USRC-01 | origi family (4 symbols, 1e) implemented behind unstable-source-api gate with oracle parity at atol=1e-12 | G1E_R_I macro chaining in cint1e_a.c; ng arrays per symbol; reuse one_electron.rs kernel infrastructure |
| USRC-02 | grids family (1e grid-based integrals) implemented with NGRIDS/PTR_GRIDS env parsing and oracle parity at atol=1e-12 | env[11]=NGRIDS, env[12]=PTR_GRIDS; shls[2]/shls[3] encode grid range; GridsEnvParams extension to OperatorEnvParams |
| USRC-03 | Breit family (2 symbols, 2e) implemented behind unstable-source-api with oracle parity at atol=1e-12 | breit.c _int2e_breit_drv three-step composition (gaunt, gauge_r1, gauge_r2); spinor-only per D-07 |
| USRC-04 | origk family (6 symbols, 3c1e) implemented behind unstable-source-api with oracle parity at atol=1e-12 | G1E_R_K macro on third shell (k) in cint3c1e_a.c; ng arrays distinguish base vs ip1 variants |
| USRC-05 | ssc family (1 symbol, 3c2e) implemented behind unstable-source-api with oracle parity at atol=1e-12 | ssc uses standard CINTgout2e with c2s_sph_3c2e1_ssc (SSC-transformed c2s); is_ssc=1 flag to CINT3c2e_drv |
| USRC-06 | Nightly CI job runs oracle with --include-unstable-source=true and 0 mismatches | xtask oracle-compare already accepts --include-unstable-source; add nightly-gated job to existing CI workflow |
</phase_requirements>

---

## Summary

Phase 14 adds five distinct integral families behind the `unstable-source-api` feature gate. Each family extends a different existing kernel module: origi extends one_electron (1e), grids requires a new NGRIDS-aware env plumbing path similar to F12 zeta, Breit extends two_electron with a three-stage host-side composition, origk extends center_3c1e, and ssc is a c2s-variant of center_3c2e requiring an SSC-transformed cart-to-sph step.

The gating and manifest infrastructure for unstable-source symbols already exists from Phase 3: `unstable_source_api_enabled()`, `is_source_only()`, `enforce_safe_facade_policy_gate()`, and the `FeatureFlag::UnstableSource` / `Stability::UnstableSource` manifest types. The manifest lock currently has exactly 2 unstable-source entries (`int2e_ipip1_sph` and `int2e_ipvip1_sph`). Phase 14 adds approximately 18 new entries under the `unstable-source` profile (D-01..D-03), updates `resolve_family_name()` in `kernels/mod.rs`, extends `OperatorEnvParams` with `GridsEnvParams`, adds the oracle test file, and wires the nightly CI job.

The xtask `oracle-compare` command already accepts `--include-unstable-source true/false`, and `generate_profile_parity_report` / `stability_is_included` already handle `unstable_source` stability filtering. The main gaps are: (1) the kernel dispatch arms, (2) the GridsEnvParams plumbing, (3) the new kernel implementations, (4) the manifest entries + lock regeneration, (5) the oracle test file, and (6) the CI nightly job.

**Primary recommendation:** Implement families in dependency order — origi first (pure 1e, no new env plumbing), then origk (3c1e parallel to origi), then ssc (3c2e c2s variant, no env plumbing), then grids (requires GridsEnvParams), then Breit (most complex, composite 2e). Lock manifest and wire CI last.

---

## Standard Stack

### Core (unchanged from prior phases)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cubecl` | 0.9.x | GPU compute backend for all kernels | Project-locked per CLAUDE.md |
| `thiserror` | 2.0.18 | Public typed errors | Library error surface |
| `anyhow` | 1.0.102 | Oracle harness, xtask | App-boundary errors |
| `cc` | 1.2.x | Vendored libcint build | Oracle hermetic build |
| `bindgen` | 0.71.1 | Oracle FFI bindings generation | Established in prior phases |

No new library dependencies are required for Phase 14. All families extend existing kernel infrastructure.

---

## Architecture Patterns

### Existing Pattern: Feature-Gated Kernel Dispatch (model for Phase 14)

`kernels/mod.rs` already has the exact pattern to mirror:

```rust
// Source: crates/cintx-cubecl/src/kernels/mod.rs
#[cfg(feature = "with-f12")]
pub mod f12;

// In resolve_family_name():
#[cfg(feature = "with-f12")]
"f12" => Some(f12::launch_f12 as FamilyLaunchFn),
```

Phase 14 adds analogous arms:
```rust
#[cfg(feature = "unstable-source-api")]
pub mod unstable;   // or per-family modules at Claude's discretion

// In resolve_family_name():
#[cfg(feature = "unstable-source-api")]
"origi"  => Some(unstable::launch_origi as FamilyLaunchFn),
#[cfg(feature = "unstable-source-api")]
"grids"  => Some(unstable::launch_grids as FamilyLaunchFn),
#[cfg(feature = "unstable-source-api")]
"breit"  => Some(unstable::launch_breit as FamilyLaunchFn),
#[cfg(feature = "unstable-source-api")]
"origk"  => Some(unstable::launch_origk as FamilyLaunchFn),
#[cfg(feature = "unstable-source-api")]
"ssc"    => Some(unstable::launch_ssc as FamilyLaunchFn),
```

Also `supports_canonical_family()` must be extended with all 5 new families:
```rust
"origi" | "grids" | "breit" | "origk" | "ssc" => cfg!(feature = "unstable-source-api"),
```

And `UNSUPPORTED_FOLLOW_ON_FAMILIES` constant arrays must include these 5 names when `unstable-source-api` is absent.

### Pattern: OperatorEnvParams Extension (model: f12_zeta field)

`OperatorEnvParams` in `planner.rs` currently has one field:
```rust
// Source: crates/cintx-runtime/src/planner.rs
pub struct OperatorEnvParams {
    pub f12_zeta: Option<f64>,  // PTR_F12_ZETA=env[9]
}
```

Phase 14 extends this:
```rust
pub struct OperatorEnvParams {
    pub f12_zeta: Option<f64>,
    /// For grids integrals: NGRIDS (from env[11]) and PTR_GRIDS offset (from env[12]).
    pub grids_params: Option<GridsEnvParams>,
}

pub struct GridsEnvParams {
    pub ngrids: usize,      // env[NGRIDS=11] cast to usize
    pub ptr_grids: usize,   // env[PTR_GRIDS=12] as env array offset
}
```

The validator must reject grids symbols when `ngrids == 0` or `ptr_grids` points outside env bounds, following the same `InvalidEnvParam` pattern as `validate_f12_env_params`.

### Pattern: Grids env layout (from libcint testsuite reference)

From `libcint-master/testsuite/test_int1e_grids.py` and `cint.h.in`:
```python
NGRIDS    = 11   # env[11] = number of grid points
PTR_GRIDS = 12   # env[12] = offset into env where grid coords start
```

Callers set:
```python
env_g[NGRIDS]    = ngrids        # total number of grid points
env_g[PTR_GRIDS] = env.size      # grid coords appended at end of env
# then append ngrids*3 floats: [x0,y0,z0, x1,y1,z1, ...]
```

`shls[2]` and `shls[3]` encode the grid point range: `shls[2]` is the first grid index and `shls[3]` is one-past-last, so `ngrids = shls[3] - shls[2]`. `grids = env + env[PTR_GRIDS] + shls[2]*3`.

Output shape is `(ncomp * NGRIDS * di * dj)` matching libcint's `CINT1e_grids_drv`.

### Pattern: Origi family — gout with G1E_R_I chaining

From `cint1e_a.c`:

| Symbol | ng array | G-tensor levels | ncomp |
|--------|----------|-----------------|-------|
| `int1e_r2_origi` | `{2, 0, 0, 0, 2, 1, 1, 1}` | 2 G1E_R_I steps: r^2 total | 1 |
| `int1e_r4_origi` | `{4, 0, 0, 0, 4, 1, 1, 1}` | 4 G1E_R_I steps: r^4 total | 1 |
| `int1e_r2_origi_ip2` | `{2, 1, 0, 0, 3, 1, 1, 3}` | G1E_D_J + 2 G1E_R_I: gradient of r^2 | 3 |
| `int1e_r4_origi_ip2` | `{4, 1, 0, 0, 5, 1, 1, 3}` | G1E_D_J + 4 G1E_R_I: gradient of r^4 | 3 |

The `ng[0]` entry is the maximum angular momentum increment on the i-index (deriv depth). G-tensor slot count: `ng[4]` is the nmax. These are standard 1e integrals using `CINT1e_drv` / `c2s_sph_1e` — they fit directly into `one_electron.rs` infrastructure. The operator is displaced relative to the origin (not a nuclear center), so the `G1E_R_I` macro encodes `r - R_origin` where `R_origin` is `env[PTR_COMMON_ORIG]` (slots 1–3, zero by default).

### Pattern: Origk family — G1E_R_K on the k (third) shell

From `cint3c1e_a.c`, origk applies the r^n polynomial on the k-shell (third center):

| Symbol | ng array | Key macro |
|--------|----------|-----------|
| `int3c1e_r2_origk` | `{0, 0, 2, 0, 2, 1, 1, 1}` | G1E_R_K, 2 steps |
| `int3c1e_r4_origk` | `{0, 0, 4, 0, 4, 1, 1, 1}` | G1E_R_K, 4 steps |
| `int3c1e_r6_origk` | `{0, 0, 6, 0, 6, 1, 1, 1}` | G1E_R_K, 6 steps |
| `int3c1e_ip1_r2_origk` | `{1, 0, 2, 0, 3, 1, 1, 3}` | G1E_D_I + G1E_R_K |
| `int3c1e_ip1_r4_origk` | `{1, 0, 4, 0, 5, 1, 1, 3}` | G1E_D_I + G1E_R_K |
| `int3c1e_ip1_r6_origk` | `{1, 0, 6, 0, 7, 1, 1, 3}` | G1E_D_I + G1E_R_K |

These use `CINT3c1e_drv` + `c2s_sph_3c1e`, consistent with `center_3c1e.rs`.

### Pattern: Breit — three-stage composition on host

From `breit.c`, the `_int2e_breit_drv` function:

1. Compute Gaunt integral: `int2e_<X>_spinor` → `buf1`
2. Compute gauge_r1: `int2e_gauge_r1_<X>_spinor` → `buf`; combine: `buf1[i] = -buf1[i] - buf[i]`
3. Compute gauge_r2: `int2e_gauge_r2_<X>_spinor` → `buf`; combine: `buf1[i] = (buf1[i] + buf[i]) * 0.5`
4. Copy `buf1` to output

For `int2e_breit_r1p2_spinor`, `ng = {2, 2, 0, 1, 4, 1, 1, 1}`. The gout function `CINTgout2e_int2e_breit_r1p2` uses G2E_D_L, G2E_R0J, G2E_D_J, G2E_D_I operators — these are standard 2e operators already present in `two_electron.rs`.

The Rust implementation follows D-08: a single composite `launch_breit` function that internally calls the three sub-kernels (Gaunt + gauge_r1 + gauge_r2), accumulates into a host-side buffer, and writes the combined result. Since the sub-kernels are themselves standard 2e variants with different gout functions, they can dispatch through the existing `two_electron.rs` infrastructure with variant-specific dispatch arms.

### Pattern: SSC family — c2s variant of 3c2e

From `cint3c2e.c`, the ssc variant:
```c
// ng is identical to standard int3c2e
FINT ng[] = {0, 0, 0, 0, 0, 1, 1, 1};
CINTinit_int3c2e_EnvVars(&envs, ng, shls, atm, natm, bas, nbas, env);
envs.f_gout = &CINTgout2e;  // same gout as standard 3c2e!
return CINT3c2e_drv(out, dims, &envs, opt, cache, &c2s_sph_3c2e1_ssc, 1);  // is_ssc=1
```

The SSC (small-small component) version uses `c2s_sph_3c2e1_ssc` instead of `c2s_sph_3c2e1`, and passes `is_ssc=1`. The gout computation is identical to `int3c2e_sph`. In Rust, this means `launch_ssc` dispatches through the existing `center_3c2e.rs` infrastructure but applies a different cart-to-sph transform. The `is_ssc` flag controls normalization factors in the `c2s_sph_3c2e1_ssc` function.

### Manifest: unstable-source profile and symbol list

The 18 symbols to add to the manifest (canonical families in parentheses):

**origi (1e family, "origi" canonical):**
- `int1e_r2_origi_sph`, `int1e_r4_origi_sph`, `int1e_r2_origi_ip2_sph`, `int1e_r4_origi_ip2_sph`

**grids (1e family, "grids" canonical):**
- `int1e_grids_sph` (base), plus derivative variants from `cint_funcs.h`: `int1e_grids_ip_sph`, `int1e_grids_ipvip_sph`, `int1e_grids_spvsp_sph`, `int1e_grids_ipip_sph`

**Breit (2e family, "breit" canonical, spinor-only):**
- `int2e_breit_r1p2_spinor`, `int2e_breit_r2p2_spinor`

**origk (3c1e family, "origk" canonical):**
- `int3c1e_r2_origk_sph`, `int3c1e_r4_origk_sph`, `int3c1e_r6_origk_sph`, `int3c1e_ip1_r2_origk_sph`, `int3c1e_ip1_r4_origk_sph`, `int3c1e_ip1_r6_origk_sph`

**ssc (3c2e family, "ssc" canonical):**
- `int3c2e_sph_ssc` (note: libcint names this `int3c2e_sph_ssc` not `int3c2e_ssc_sph`)

The `compiled_in_profiles` for all unstable-source symbols is `["unstable-source"]` per D-01/D-02. Their `stability` is `"unstable_source"`, `feature_flag` is `"unstable_source"`, `helper_kind` is `SourceOnly`.

**Critical note on existing manifest entries:** The manifest already has 2 unstable-source entries (`int2e_ipip1_sph` and `int2e_ipvip1_sph`) with `compiled_in_profiles: ["base", "with-f12", "with-4c1e", "with-f12+with-4c1e"]`. The new Phase 14 symbols must use `["unstable-source"]` only per D-02. These existing entries appear to be placeholder source-only markers from Phase 3 that should remain as-is.

### Manifest generator — adding unstable-source profile

The manifest generator in `crates/cintx-ops` must:
1. Add `"unstable-source"` as a 5th profile name to the profiles list
2. Add an explicit symbol list (D-03: ~18 symbols hardcoded) with `compiled_in_profiles = ["unstable-source"]`
3. Regenerate `compiled_manifest.lock.json`
4. Regenerate `api_manifest.rs` with the new entries

The xtask `manifest-audit` command must accept `unstable-source` as a valid profile in `--profiles` without error.

### Oracle: build.rs extensions for unstable family C declarations

The oracle `build.rs` must be extended in two places:

**1. Source file list** (new files to compile):
```rust
.file(libcint_root.join("src/cint1e_a.c"))       // already present (origi)
.file(libcint_root.join("src/cint1e_grids.c"))    // new: grids
.file(libcint_root.join("src/g1e_grids.c"))       // new: grids G-tensor
.file(libcint_root.join("src/breit.c"))           // new: Breit composite
.file(libcint_root.join("src/cint3c1e_a.c"))      // new: origk
// cint3c2e.c is already present (ssc is in it)
```

**2. Supplemental header** (new `extern` declarations):
```c
/* Phase 14 unstable-source family declarations */
extern CINTIntegralFunction int1e_r2_origi_sph;
extern CINTIntegralFunction int1e_r4_origi_sph;
extern CINTIntegralFunction int1e_r2_origi_ip2_sph;
extern CINTIntegralFunction int1e_r4_origi_ip2_sph;
extern CINTIntegralFunction int1e_grids_sph;        // NOT in cint_funcs.h
extern CINTIntegralFunction int1e_grids_ip_sph;     // in cint_funcs.h
/* ... etc for all 18 symbols */
extern CINTIntegralFunction int2e_breit_r1p2_spinor;
extern CINTIntegralFunction int2e_breit_r2p2_spinor;
extern CINTIntegralFunction int3c1e_r2_origk_sph;
/* ... etc */
extern CINTIntegralFunction int3c2e_sph_ssc;
```

**3. bindgen allowlist** must include all 18 symbol names.

**Critical:** `int1e_grids_sph` is NOT in `cint_funcs.h` (verified: grep returns 0 matches). It must be declared in the supplemental header. The derivative grids symbols (`int1e_grids_ip_sph`, etc.) ARE in `cint_funcs.h`.

Also `cint1e_a.c` is already compiled. The `int1e_r2_origi_sph` etc. functions are defined there but not declared in `cint_funcs.h` — they need supplemental declarations.

### Oracle: test file structure (single file per D-09)

`crates/cintx-oracle/tests/unstable_source_parity.rs` structure:
```rust
#![cfg(feature = "cpu")]
#![cfg(feature = "unstable-source-api")]

// Per-family sections, each function gated on #[cfg(has_vendor_libcint)]
// for the actual oracle comparison, with a non-vendor fallback test.

mod origi { /* 4 symbols */ }
mod grids { /* 5 symbols including grids fixture with grid coords in env */ }
mod breit { /* 2 spinor symbols */ }
mod origk { /* 6 symbols */ }
mod ssc   { /* 1 symbol */ }
```

The grids fixture extends `build_h2o_sto3g()` by appending grid point coordinates to env:
```rust
fn build_h2o_sto3g_grids(ngrids: usize) -> (Vec<i32>, Vec<i32>, Vec<f64>, Vec<i32>) {
    let (atm, bas, mut env) = build_h2o_sto3g();
    env[11] = ngrids as f64;          // NGRIDS
    env[12] = env.len() as f64;       // PTR_GRIDS offset
    for _ in 0..ngrids {
        env.extend_from_slice(&[0.0_f64, 0.5, 1.0]); // example grid coords
    }
    let shls_grids = [0_i32, 1, 0, ngrids as i32];   // shls[2]=0, shls[3]=ngrids
    (atm, bas, env, shls_grids.to_vec())
}
```

### CI: nightly unstable_source_oracle job

The nightly `unstable_source_oracle` job added to `compat-governance-pr.yml` follows the same structure as `oracle_parity_gate` but:
- Trigger condition: only runs on `schedule` or explicit `workflow_dispatch` (not on every PR push)
- Uses `--include-unstable-source true`
- Uses the `unstable-source` profile

```yaml
unstable_source_oracle:
    name: unstable_source_oracle (nightly)
    if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
    runs-on: ubuntu-latest
    steps:
        # ... same toolchain resolution as other jobs ...
        - name: Run unstable-source oracle
          run: |
              cargo run --manifest-path xtask/Cargo.toml -- oracle-compare \
                --profiles "unstable-source" \
                --include-unstable-source true
```

The `run_oracle_compare` in `xtask/src/oracle_update.rs` already accepts `include_unstable_source: bool` and passes it through `generate_profile_parity_report`. No xtask code changes are needed — only the CI job definition.

However, `xtask/src/oracle_update.rs` currently uses `validate_required_profile_scope()` which may enforce the 4 standard profiles. This validation must accept `"unstable-source"` as a valid profile when used with `--include-unstable-source true`.

### Anti-Patterns to Avoid

- **Do NOT use `canonical_family = "1e"` for origi**: The manifest dispatch routes to `launch_one_electron` via canonical_family. Origi must use `"origi"` to route to the new kernel.
- **Do NOT add unstable symbols to the 4 standard profiles**: D-02 requires standalone `unstable-source` profile only.
- **Do NOT set `NGRIDS` from env directly in the planner without validation**: PTR_GRIDS=0 or NGRIDS=0 must fail with `InvalidEnvParam` before kernel dispatch.
- **Do NOT implement Breit cart or sph in Phase 14**: D-07 is spinor-only. Return `UnsupportedRepresentation` for cart/sph.
- **Do NOT reuse grids shls format from 2-shell APIs**: grids uses `shls[4]` where `shls[2]`=grid_start and `shls[3]`=grid_end. The normal 2-shell (shls[0]=i, shls[1]=j) interpretation is preserved.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Gaunt computation for Breit | Custom Gaunt kernel | Reuse 2e infrastructure with variant gout dispatch | Already implemented; Breit is Gaunt + gauge, not a new integral class |
| SSC cart-to-sph | Custom c2s transform | Existing `c2s_sph_3c2e1_ssc` pattern from libcint; follow Phase 9 c2s approach | SSC differs only in normalization factor, not in the angular integral |
| Grids VRR | Custom VRR loop | Reuse `CINTg0_1e_grids` pattern; same Rys quadrature as standard 1e but with grid-displaced centers | Grid integral uses identical Rys machinery as 1e overlap |
| Feature gating | Custom compile-time flag | `#[cfg(feature = "unstable-source-api")]` (already defined in cintx-compat and cintx-rs) | Feature already propagated; cubecl crate needs it added |

---

## Common Pitfalls

### Pitfall 1: `cintx-cubecl` missing `unstable-source-api` feature
**What goes wrong:** The `unstable-source-api` feature exists in `cintx-compat` and `cintx-rs` Cargo.toml files, but NOT in `cintx-cubecl/Cargo.toml` (verified: no such feature declared there). Kernel dispatch arms gated on `#[cfg(feature = "unstable-source-api")]` in `kernels/mod.rs` will never compile unless the feature is added to `cintx-cubecl/Cargo.toml` and forwarded from `cintx-rs`.
**Why it happens:** The feature gate originates in cintx-compat but was not propagated to cintx-cubecl in Phase 3.
**How to avoid:** Add `unstable-source-api = []` to `cintx-cubecl/Cargo.toml` features, and add `cintx-cubecl/unstable-source-api` to the `cintx-rs` forwarding chain.
**Warning signs:** `#[cfg(feature = "unstable-source-api")]` items in `kernels/mod.rs` silently compile out even when `--features unstable-source-api` is passed at the workspace level.

### Pitfall 2: Breit spinor-only returns UnsupportedRepresentation before dispatch
**What goes wrong:** If Breit symbols have `forms = ["spinor"]` in the manifest but the representation guard in `resolve_family()` is not hit before the kernel, cart/sph callers may reach the kernel and panic or produce wrong results.
**Why it happens:** `ExecutionPlan::new` calls `validate_shell_tuple` which checks `supports_representation` based on manifest `forms`. If manifest is correct, the check happens before dispatch. But if forms includes "sph" by mistake, the guard is bypassed.
**How to avoid:** Manifest entries for Breit must have `forms: &["spinor"]` only (no "sph", no "cart"). The representation check in `resolve_family()` in `kernels/mod.rs` provides the second guard.

### Pitfall 3: Grids NGRIDS slot at env[11] vs NGRIDS as the integer value
**What goes wrong:** libcint uses `env[11] = ngrids` (NGRIDS constant = 11 is the index). It is NOT `env[NGRIDS]` where NGRIDS is the count. The constant name `NGRIDS=11` is the index, and the value stored there is the count.
**Why it happens:** The Python test code `env_g[NGRIDS] = ngrids` reads as if NGRIDS is a size, but it is an index (slot 11).
**How to avoid:** In Rust: `let ngrids_count = env[11] as usize` and `let ptr_grids = env[12] as usize`. Add a constant `const NGRIDS: usize = 11` and `const PTR_GRIDS: usize = 12` to `raw.rs` (they are not there yet).

### Pitfall 4: `int1e_grids_sph` not in `cint_funcs.h`
**What goes wrong:** The oracle `build.rs` uses `cint_funcs.h` as the primary bindgen header. `int1e_grids_sph` (the base symbol) is not declared there (verified: 0 grep matches). If not added to the supplemental header, oracle bindgen will silently exclude it and the oracle FFI call will fail to link.
**Why it happens:** libcint exposes the base `int1e_grids` function via `ALL_CINT()` macro in `cint1e_grids.c` but the function declaration was not added to `cint_funcs.h`.
**How to avoid:** Add `extern CINTIntegralFunction int1e_grids_sph;` to the supplemental header in `build.rs`, alongside the derivative grids declarations. Also add to bindgen allowlist.

### Pitfall 5: `compiled_in_profiles` for existing unstable-source stubs vs new symbols
**What goes wrong:** The 2 existing unstable-source manifest entries (`int2e_ipip1_sph`, `int2e_ipvip1_sph`) have `compiled_in_profiles: ["base", "with-f12", "with-4c1e", "with-f12+with-4c1e"]` — not `["unstable-source"]`. Phase 14 new symbols must use `["unstable-source"]` per D-02. Mixing these in the manifest regeneration could overwrite the existing entries.
**Why it happens:** The existing stubs predate the `unstable-source` profile decision from D-01.
**How to avoid:** The manifest generator must preserve the existing stub entries as-is and only add new entries with `["unstable-source"]` profile.

### Pitfall 6: Breit composition buffer aliasing in Rust
**What goes wrong:** libcint's `_int2e_breit_drv` allocates `buf` (2x nop) on the heap and uses `buf1 = buf + nop` or `buf1 = out` depending on dims. A Rust port that shares `out` with `buf1` when dims is provided can alias the output buffer, corrupting results.
**Why it happens:** The C code has explicit pointer aliasing logic. A naive Rust translation using `&mut out[..]` for both `buf1` and the accumulation target can violate the borrow rules or alias incorrectly.
**How to avoid:** Allocate a separate `Vec<Complex<f64>>` for the intermediate result buffer regardless of whether dims is provided. Copy to `out` at the end, following the `_copy_to_out` pattern.

---

## Code Examples

### Example 1: origi gout implementation in Rust (kernel side)

```rust
// Source: derived from libcint-master/src/cint1e_a.c CINTgout1e_int1e_r2_origi
// int1e_r2_origi gout: computes sum of r^2 components (x^2 + y^2 + z^2)
// ng = [2, 0, 0, 0, 2, 1, 1, 1]; g_size = computed from env setup
// Uses G1E_R_I twice: one step for the first r factor, one for the second.
// g1 = G1E_R_I(g0, i_l+1), g3 = G1E_R_I(g1, i_l+0)
// s = g3[ix]*g0[iy]*g0[iz] + g0[ix]*g3[iy]*g0[iz] + g0[ix]*g0[iy]*g3[iz]
```

### Example 2: GridsEnvParams validation pattern

```rust
// Source: derived from crates/cintx-runtime/src/validator.rs validate_f12_env_params pattern
pub fn validate_grids_env_params(
    params: &OperatorEnvParams,
    canonical_family: &str,
) -> Result<(), cintxRsError> {
    if canonical_family != "grids" {
        return Ok(());
    }
    match &params.grids_params {
        None => Err(cintxRsError::InvalidEnvParam {
            param: "NGRIDS",
            detail: "grids family requires NGRIDS > 0 and PTR_GRIDS in env".to_owned(),
        }),
        Some(gp) if gp.ngrids == 0 => Err(cintxRsError::InvalidEnvParam {
            param: "NGRIDS",
            detail: "NGRIDS must be > 0 for grids integrals".to_owned(),
        }),
        _ => Ok(()),
    }
}
```

### Example 3: grids fixture in oracle test

```rust
// Source: derived from libcint-master/testsuite/test_int1e_grids.py
// NGRIDS = 11 (env index), PTR_GRIDS = 12 (env index)
fn build_h2o_sto3g_grids(ngrids: usize) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let (atm, bas, mut env) = build_h2o_sto3g();
    env[11] = ngrids as f64;       // env[NGRIDS] = count
    env[12] = env.len() as f64;    // env[PTR_GRIDS] = start offset into env
    // Append ngrids * 3 coords
    for k in 0..ngrids {
        env.push(0.1 * k as f64);  // x
        env.push(0.2 * k as f64);  // y
        env.push(0.3 * k as f64);  // z
    }
    (atm, bas, env)
}
// shls for grids: shls[0]=i_shell, shls[1]=j_shell, shls[2]=grid_start, shls[3]=grid_start+ngrids
```

### Example 4: Breit composition (host-side three-step)

```rust
// Source: derived from libcint-master/src/breit.c _int2e_breit_drv
// Step 1: gaunt -> buf1
// Step 2: gauge_r1 -> buf; buf1[i] = -buf1[i] - buf[i]
// Step 3: gauge_r2 -> buf; buf1[i] = (buf1[i] + buf[i]) * 0.5
// Copy buf1 -> out
// All three sub-kernels use CINT2e_spinor_drv with different gout functions.
// In Rust: dispatch three separate ExecutionPlan launches with variant gout params,
// accumulate in a host Vec<Complex<f64>>, write to output.
```

---

## Environment Availability

Step 2.6: SKIPPED — this phase is purely code/config changes extending existing kernel infrastructure. No new external tools, services, or runtimes are required. The existing `CINTX_ORACLE_BUILD_VENDOR=1 cargo test` path already exercises vendored libcint compilation.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in test + `cargo nextest` |
| Config file | none (workspace default) |
| Quick run command | `cargo test -p cintx-oracle --features cpu,unstable-source-api -- unstable_source_parity` |
| Full suite command | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu,unstable-source-api -- unstable_source_parity` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| USRC-01 | origi family oracle parity atol=1e-12 | oracle comparison | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu,unstable-source-api -- origi` | ❌ Wave 0 |
| USRC-02 | grids family oracle parity atol=1e-12 | oracle comparison | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu,unstable-source-api -- grids` | ❌ Wave 0 |
| USRC-03 | Breit spinor oracle parity atol=1e-12 | oracle comparison | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu,unstable-source-api -- breit` | ❌ Wave 0 |
| USRC-04 | origk family oracle parity atol=1e-12 | oracle comparison | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu,unstable-source-api -- origk` | ❌ Wave 0 |
| USRC-05 | ssc family oracle parity atol=1e-12 | oracle comparison | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu,unstable-source-api -- ssc` | ❌ Wave 0 |
| USRC-06 | Nightly CI 0 mismatches | CI job + smoke | verify CI job added; `cargo run --manifest-path xtask/Cargo.toml -- oracle-compare --profiles unstable-source --include-unstable-source true` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p cintx-cubecl --features unstable-source-api -x` (dispatch compile check)
- **Per wave merge:** `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu,unstable-source-api -- unstable_source_parity`
- **Phase gate:** Full suite green (including nightly CI job definition) before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `crates/cintx-oracle/tests/unstable_source_parity.rs` — covers USRC-01 through USRC-05
- [ ] `crates/cintx-ops/src/generated/api_manifest.csv` updated with ~18 new unstable-source rows
- [ ] `unstable-source-api = []` feature added to `crates/cintx-cubecl/Cargo.toml`
- [ ] `NGRIDS = 11` and `PTR_GRIDS = 12` constants added to `crates/cintx-compat/src/raw.rs`

---

## Open Questions

1. **Profile scope validation in xtask `validate_required_profile_scope`**
   - What we know: `xtask/src/oracle_update.rs` calls `validate_required_profile_scope(profiles)` before running oracle compare. The current required profile CSV is `"base,with-f12,with-4c1e,with-f12+with-4c1e"`.
   - What's unclear: Whether `validate_required_profile_scope` hard-rejects profiles not in the standard 4. If it does, the nightly CI job passing `--profiles "unstable-source"` will fail.
   - Recommendation: Read `validate_required_profile_scope` implementation before the CI plan step, and relax it to also accept `"unstable-source"` as valid.

2. **SSC c2s transform in Rust — existing implementation status**
   - What we know: `c2s_sph_3c2e1_ssc` is the libcint C function for the SSC c2s variant. The Rust `center_3c2e.rs` may or may not already have an SSC path.
   - What's unclear: Whether `center_3c2e.rs` already has `is_ssc` routing or if it must be added.
   - Recommendation: Read `crates/cintx-cubecl/src/kernels/center_3c2e.rs` before planning the ssc task.

3. **Breit sub-kernel gout functions already in kernel registry**
   - What we know: `int2e_breit_r1p2` uses `CINTgout2e_int2e_breit_r1p2` which uses G2E_D_L, G2E_R0J, G2E_D_J, G2E_D_I. These are 2e operators with angular momentum increments. The Breit gout is a specific combination that must be implemented as a distinct kernel variant.
   - What's unclear: Whether `two_electron.rs` supports arbitrary gout dispatch or only a fixed set.
   - Recommendation: Read `two_electron.rs` to understand gout dispatch before planning the Breit task.

---

## Sources

### Primary (HIGH confidence)
- `libcint-master/src/cint1e_a.c` — origi gout functions, ng arrays for all 4 symbols (read directly)
- `libcint-master/src/cint1e_grids.c` — grids CINT1e_grids_drv, NGRIDS/PTR_GRIDS env parsing (read directly)
- `libcint-master/src/g1e_grids.c` — CINTinit_int1e_grids_EnvVars, CINTg0_1e_grids, CINTgout1e_grids (read directly)
- `libcint-master/src/breit.c` — _int2e_breit_drv composition algorithm, ng arrays (read directly)
- `libcint-master/src/cint3c1e_a.c` — origk gout functions and ng arrays for all 6 symbols (read directly)
- `libcint-master/src/cint3c2e.c` — ssc variant using c2s_sph_3c2e1_ssc and is_ssc=1 (read directly)
- `libcint-master/include/cint.h.in` — NGRIDS=11, PTR_GRIDS=12 constants (verified)
- `libcint-master/include/cint_funcs.h` — grids derivative declarations (verified; base sph NOT present)
- `libcint-master/testsuite/test_int1e_grids.py` — env layout for grids (NGRIDS, PTR_GRIDS, shls format)
- `crates/cintx-cubecl/src/kernels/mod.rs` — resolve_family_name dispatch pattern (read directly)
- `crates/cintx-runtime/src/planner.rs` — OperatorEnvParams / ExecutionPlan structure (read directly)
- `crates/cintx-runtime/src/validator.rs` — validate_f12_env_params pattern (read directly)
- `crates/cintx-oracle/build.rs` — supplemental header pattern, cc source list, bindgen allowlist (read directly)
- `crates/cintx-compat/src/raw.rs` — unstable_source_api_enabled(), NGRIDS/PTR_GRIDS slots doc (read directly)
- `crates/cintx-ops/src/resolver.rs` — Stability::UnstableSource, is_source_only, FeatureFlag::UnstableSource (read directly)
- `crates/cintx-ops/src/generated/api_manifest.rs` — existing unstable-source entry structure (read directly)
- `crates/cintx-oracle/src/fixtures.rs` — stability_is_included, build_h2o_sto3g pattern (read directly)
- `xtask/src/oracle_update.rs` — run_oracle_compare, --include-unstable-source plumbing (read directly)
- `.github/workflows/compat-governance-pr.yml` — existing CI job structure for nightly job reference (read directly)

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new libraries required; all existing infrastructure verified by direct file reads
- Architecture: HIGH — all gout algorithms verified from upstream libcint source; dispatch patterns verified from existing kernel code
- Pitfalls: HIGH — feature propagation gap (cubecl missing unstable-source-api) and int1e_grids_sph supplemental declaration gap verified by grep
- Open questions: MEDIUM — SSC and Breit sub-kernel dispatch status unread; xtask profile validation behavior unread

**Research date:** 2026-04-05
**Valid until:** 2026-05-05 (stable domain — libcint 6.1.3 is pinned)
