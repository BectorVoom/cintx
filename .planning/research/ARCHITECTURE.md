# Architecture Patterns

**Project:** cintx v1.2 — Full API Parity & Unified Oracle Gate
**Researched:** 2026-04-04
**Confidence:** HIGH (primary evidence from direct codebase inspection + design document)

---

## Baseline Architecture (v1.1 State)

The architecture reached the end of v1.1 with the following verified state:

| Crate | Confirmed State |
|-------|----------------|
| `cintx-core` | Stable domain types, `cintxRsError`, `Representation` enum (Cart/Spheric/Spinor) |
| `cintx-ops` | Manifest lock with `FeatureFlag::WithF12`, `With4c1e`, `UnstableSource`; `HelperKind` enum covers Helper/Transform/Optimizer/Legacy/SourceOnly |
| `cintx-runtime` | `ExecutionPlan`, `WorkspaceQuery`, planner, validator, dispatcher — unchanged contracts |
| `cintx-cubecl` | `ResolvedBackend` enum (Wgpu/Cpu); `FamilyLaunchFn` accepting `(&ResolvedBackend, &ExecutionPlan, &SpecializationKey, &mut [f64])`; 5 base families with real kernels |
| `cintx-cubecl::transform` | `c2s.rs` (Condon-Shortley cart→sph) and `c2spinor.rs` (interleaved staging stub) confirmed in source |
| `cintx-compat` | Helper, transform, optimizer, legacy, raw, layout modules — surface complete but helper/transform oracle symbols partially missing |
| `cintx-oracle` | Harness compares 5 base families; `tolerance_for_family` has no entry for F12/STG/YP families; `IMPLEMENTED_TRANSFORM_SYMBOLS` covers 7 of 34 helper/transform symbols |
| `ci/oracle-compare.yml` | Runs `oracle-compare` + `helper-legacy-parity` per profile across all 4 matrix profiles |
| `ci/feature-matrix.yml` | Runs `manifest-audit` + `oom-contract-check` per profile |

The confirmed gap between v1.1 and v1.2 full API parity involves four integration domains.

---

## What Changes in v1.2 vs. What Stays

**Unchanged (no modification needed):**
- `cintx-core` domain types, `cintxRsError` variants, `Representation` enum
- `cintx-runtime` planner, validator, scheduler, workspace, `BackendExecutor` trait, `ExecutionIo`, `DispatchDecision`
- `cintx-cubecl` `ResolvedBackend` enum, `FamilyLaunchFn` type alias, `launch_family` dispatch table
- `cintx-rs` safe facade public surface
- `cintx-capi` C ABI shim surface
- CI workflow YAML structure (profiles, job topology, artifact paths)
- `compiled_manifest.lock.json` schema (only `oracle_covered` field values update when new families get coverage)
- `Cargo.lock` (add deps deliberately, not automatically)

**Modified in `cintx-cubecl`:**
- `src/kernels/mod.rs`: add `"f12"` and `"unstable_source"` match arms to `resolve_family_name` and `supports_canonical_family`
- `src/kernels/`: add new kernel files per new family
- `src/transform/c2spinor.rs`: replace interleaved stub with real Pauli/Condon-Shortley spinor coefficients
- `src/transform/c2s.rs`: extend to l>=5 if h-function support lands (deferred in REQUIREMENTS.md as out of scope for v1.1)

**Modified in `cintx-compat`:**
- `src/helpers.rs`: add remaining ~17 unimplemented helper symbols to reach full 34-symbol coverage
- `src/transform.rs`: add remaining transform symbols beyond current 7
- `src/optimizer.rs`: extend to cover family-specific optimizer init for F12/STG/YP

**Modified in `cintx-oracle`:**
- `src/compare.rs`: add `tolerance_for_family` entries for F12/STG/YP, 4c1e extensions, and unstable-source families
- `src/compare.rs`: extend `IMPLEMENTED_HELPER_SYMBOLS` and `IMPLEMENTED_TRANSFORM_SYMBOLS` as compat coverage grows
- `src/fixtures.rs`: add fixture molecules and shell configurations for F12 parameters (`PTR_F12_ZETA`), 4c1e extended shell ranges, and unstable-source families

**Modified in `cintx-ops/src/generated/`:**
- `api_manifest.rs`: flip `oracle_covered: false` to `oracle_covered: true` as each symbol gains CI oracle coverage — this is the primary v1.2 tracking mechanism
- `compiled_manifest.lock.json`: no format change; only `oracle_covered` field updates

**Modified in `xtask/`:**
- `src/manifest_audit.rs`: update audit logic to check that all `stability: Stable` and `stability: Optional` entries with `oracle_covered: true` actually have passing oracle comparison records
- `src/oracle_update.rs`: extend oracle update helper to drive `with-f12` and `with-4c1e+unstable-source` profiles

---

## Integration Domain 1: F12/STG/YP Kernel Integration

### Where F12 slots into the existing dispatch

F12/STG/YP are 2e-class operators (four-center) sharing the same shell tuple arity (4) and output buffer shape as base 2e integrals. The kernel physics differs: the electron-repulsion Coulomb kernel `1/r12` is replaced by a short-range Gaussian geminal `exp(-zeta * r12^2)` (STG) or Yukawa-Pearson (YP) screened interaction. The `PTR_F12_ZETA` slot in `env` carries the zeta parameter.

The integration path through the dispatch chain is identical to the base 2e family:

```
manifest resolver: family_name="2e", feature_flag=WithF12 -> OperatorDescriptor
runtime planner: same ShellTuple4, same component_count logic
cubecl dispatcher: resolve_family_name("f12") -> launch_f12
f12 kernel: same pdata/Rys infrastructure, different 2D VRR operator kernel
```

The `resolve_family_name` function in `kernels/mod.rs` must add an `"f12"` arm. The manifest's `canonical_family` field for F12/STG/YP symbols must be set to `"f12"` (not `"2e"`) so the dispatch remains unambiguous.

### New files needed in `cintx-cubecl`

- `src/kernels/f12.rs` — STG and YP 2e kernels sharing pdata/Rys infrastructure from `two_electron.rs`

### F12 representation coverage

The design document and manifest specification are explicit: F12/STG/YP supports **sph only**. Cart and spinor symbols do not exist in the compiled upstream library. The kernel and manifest entries must enforce this. The `RepresentationSupport` in manifest entries for F12 families must have `cart: false, spheric: true, spinor: false`. The release gate verifies that cart/spinor symbol counts are zero in the `with-f12` profile — this is already CI-audited via `manifest-audit`.

### Oracle harness additions for F12

`src/compare.rs` in `cintx-oracle` needs:
1. A new `tolerance_for_family("f12")` entry (atol=1e-6, rtol=1e-4 per design doc section 13.8)
2. Fixture molecules with `env[PTR_F12_ZETA]` populated
3. The F12 families added to the profile-fixture matrix for the `with-f12` profile
4. `oracle_covered` set to `true` in the manifest lock for all 10 F12 sph symbol families

---

## Integration Domain 2: Spinor Representation Support

### Current state of spinor in v1.1

The `c2spinor.rs` transform stub (`cart_to_spinor_interleaved_staging`) is a placeholder that computes an amplitude blend, not real spinor coefficients. The `Representation::Spinor` variant exists in `cintx-core` and is parsed by the manifest (`RepresentationSupport::spinor: bool`). However, no real Pauli/two-component spinor transform is implemented.

The v1.1 REQUIREMENTS.md explicitly defers spinor representation kernels to v1.2.

### What spinor integration requires

Spinor output involves complex numbers. The buffer format is interleaved `[Re, Im]` pairs. The size formula differs from cart/sph:

```
spinor_len(l, kappa) = 4l+2 (kappa=0), 2l+2 (kappa<0), 2l (kappa>0)
```

This is already implemented in `cintx-compat/src/helpers.rs` (`len_spinor` function, confirmed in source).

The `c2spinor.rs` transform must be replaced with a real Clebsch-Gordan / unitary coupling matrix application that converts Cartesian raw output to two-component spinors. This is a pure math change inside `cintx-cubecl/src/transform/c2spinor.rs` — no changes needed to the dispatch chain, manifest, or planner.

### Components affected by real spinor support

| Component | Change |
|-----------|--------|
| `cintx-cubecl/src/transform/c2spinor.rs` | Replace stub blend with real Clebsch-Gordan coupling matrices |
| `cintx-cubecl/src/kernels/one_electron.rs` | Add spinor post-processing path (currently only cart/sph branches exist) |
| `cintx-cubecl/src/kernels/two_electron.rs` | Same |
| `cintx-oracle/src/compare.rs` | Update spinor buffer contract check (currently validates only that `complex_interleaved` flag is set and pairs are finite) |
| `cintx-oracle/src/fixtures.rs` | Add spinor-representation fixtures for base 1e and 2e families |

F12 spinor is out of scope (no upstream compiled symbols). 4c1e spinor is out of scope (outside the `Validated4C1E` envelope). Spinor for base 1e/2e families is the v1.2 target.

---

## Integration Domain 3: Extended 4c1e Integration

### Current state of 4c1e in v1.1

`cintx-cubecl/src/kernels/center_4c1e.rs` has a `Validated4C1E` classifier stub. The kernel returns zeros and `ExecutionStats` after passing the envelope check. The envelope is: cart/sph only, scalar `int4c1e`, natural dims, max(l)<=4, oracle+identity pass.

### What extended 4c1e requires in v1.2

"Beyond the initial validated envelope" means implementing the actual kernel computation for inputs within the existing `Validated4C1E` classifier — the envelope definition itself is not expanded without additional fuzz and identity gate evidence (which is a phase-level research question, not architecture). The v1.2 goal is real kernel values where the envelope already accepts inputs.

Components affected:

| Component | Change |
|-----------|--------|
| `cintx-cubecl/src/kernels/center_4c1e.rs` | Replace the zero-fill stub with real integral computation (same Rys/pdata infrastructure as 2e) |
| `cintx-oracle/src/compare.rs` | Add `tolerance_for_family("4c1e")` (already defined at `atol=1e-6` — **confirmed in source**) |
| `cintx-oracle/src/compare.rs` | Add 4c1e identity test: `int4c1e_sph == (-1/4pi)*trace(int2e_ipip1 + 2*int2e_ipvip1 + permuted)` |
| `cintx-oracle/src/fixtures.rs` | Add 4c1e fixture inputs within the validated envelope |
| `cintx-ops/src/generated/api_manifest.rs` | Flip `oracle_covered: false` to `true` for `int4c1e_cart` and `int4c1e_sph` |

The `compat::workaround::int4c1e_via_2e_trace` composition path (specified in design doc section 3.11.2) requires implementing a new module in `cintx-compat`. This is additive — it does not modify any existing compat surface.

---

## Integration Domain 4: Unstable-Source API Integration

### What unstable-source means architecturally

Unstable-source families are those where `declared_in = source` and `canonical_family` does not appear in header declarations. They are gated behind the `unstable-source-api` Cargo feature. The manifest already carries `feature_flag: FeatureFlag::UnstableSource` and `stability: Stability::UnstableSource` for applicable entries.

The dispatch chain already handles these correctly: `resolve_family_name` returns `None` for unknown families, which produces `UnsupportedApi`. Adding unstable-source families is a matter of adding them to the kernel registry and providing implementations.

### What unstable-source integration requires

1. Add match arms for unstable-source canonical families in `resolve_family_name` behind `#[cfg(feature = "unstable-source-api")]`
2. Add kernel files per family
3. Add oracle comparison in nightly/extended CI — the design document specifies that unstable-source families are validated "in nightly extended CI when the feature is enabled" (section 3.3.1)
4. The `oracle-compare.yml` CI job already accepts `--include-unstable-source false/true` as a flag — extend it to pass `true` in the nightly extended run

The feature flag wiring in `kernels/mod.rs` follows the exact same `#[cfg(feature = "with-4c1e")]` pattern already established.

---

## Integration Domain 5: Helper, Transform, and Wrapper API Completion

### Current gap

`cintx-oracle/src/compare.rs` shows:
- `IMPLEMENTED_HELPER_SYMBOLS`: 17 symbols covered
- `IMPLEMENTED_TRANSFORM_SYMBOLS`: 7 symbols covered
- `IMPLEMENTED_OPTIMIZER_SYMBOLS`: 7 symbols covered

The design document section 3.2 specifies 34 total helper/wrapper/transform APIs. The gap is approximately 3 symbols in the optimizer category and ~10 in the helper/transform combined. Exact count requires re-reading `include/cint.h.in:227-291` against the `IMPLEMENTED_*` lists.

### Integration pattern for helper completion

Helper APIs are pure host-side Rust functions in `cintx-compat/src/helpers.rs`. They do not touch the CubeCL dispatch chain, kernels, or manifest resolver. Adding a helper symbol means:

1. Implement the function in `cintx-compat/src/helpers.rs`
2. Add the symbol name to `IMPLEMENTED_HELPER_SYMBOLS` in `cintx-oracle/src/compare.rs`
3. Add the symbol's `oracle_covered: true` in the manifest
4. The oracle harness automatically picks up comparison in the `helper-legacy-parity` CI gate

Transform symbols in `IMPLEMENTED_TRANSFORM_SYMBOLS` follow the same pattern but live in `cintx-compat/src/transform.rs`. The transform functions delegate to `cintx-cubecl::transform::c2s` and `c2spinor` — completing spinor transforms (Domain 2 above) unblocks transform symbol completion.

### Unified oracle tolerance requirement

The v1.2 milestone requires unifying oracle tolerance to `atol=1e-12` across ALL families. The `compare.rs` source currently uses family-specific tolerances ranging from `1e-12` (2e) down to `1e-6` (4c1e, F12). The "unified atol=1e-12" requirement likely means tightening only the families that currently pass at looser tolerances, not introducing a single constant. The design document specifies different tolerances per category (section 13.8) and these are the authoritative values. The PROJECT.md states atol=1e-12 as a milestone goal — clarification may be needed in a phase plan whether this means tighten all tolerances or just ensure every family reaches oracle comparison with at least the specified family tolerance. Current evidence from `compare.rs` suggests per-family tolerances are correct architecture; the "unify" goal likely means ensuring no family is currently exempt from comparison.

---

## Component Boundary Summary

| Component | New Files | Modified Files | Unchanged |
|-----------|-----------|----------------|-----------|
| `cintx-core` | none | none | all |
| `cintx-ops` | none | `api_manifest.rs` (oracle_covered updates), `compiled_manifest.lock.json` | schema unchanged |
| `cintx-runtime` | none | none | all |
| `cintx-cubecl` | `kernels/f12.rs`, possibly `kernels/unstable_source.rs` | `kernels/mod.rs`, `transform/c2spinor.rs`, `kernels/center_4c1e.rs`, per-family kernel files for spinor path | backend/, specialization.rs, resident_cache.rs, transfer.rs |
| `cintx-compat` | `workaround.rs` (4c1e composition path) | `helpers.rs`, `transform.rs`, `optimizer.rs` | raw.rs, layout.rs, legacy.rs, lib.rs |
| `cintx-oracle` | none | `compare.rs` (tolerances, symbol lists), `fixtures.rs` (new fixtures) | vendor_ffi.rs, lib.rs |
| `cintx-rs` | none | none | all |
| `cintx-capi` | none | none | all |
| `xtask` | none | `manifest_audit.rs`, `oracle_update.rs` | main.rs, gen_docs.rs, bench_report.rs |
| `ci/` | none | `oracle-compare.yml` (unstable-source flag path), possibly new nightly job | feature-matrix.yml |

---

## Data Flow Changes for New Families

The call flow for F12/STG/YP through the existing pipeline:

```
Caller -> eval_raw("int2e_stg_sph", ..., opt, ...)
  |
  v
cintx-compat::raw: RawApiId resolved from symbol string
  -> Resolver::descriptor_by_symbol("int2e_stg_sph")
       -> ManifestEntry { canonical_family: "f12", feature_flag: WithF12, ... }
  -> validate: feature "with-f12" enabled? Yes -> proceed; No -> UnsupportedApi
  |
  v
cintx-runtime::planner: ExecutionPlan { representation: Spheric, ... }
  -> validate: descriptor.supports_representation(Spheric)? Yes (spinor=false, cart=false, spheric=true)
  -> component_count from descriptor.component_rank
  -> output_layout: same shape formula as 2e (four extents di*dj*dk*dl)
  |
  v
cintx-cubecl: resolve_family_name("f12") -> launch_f12
  -> f12.rs: same pdata/Rys entry from math::pdata and math::rys
  -> STG: replace 1/r12 Coulomb kernel with exp(-zeta*r12^2) geminal
  -> YP: replace with Yukawa-Pearson interaction
  -> client.create / kernel launch / client.read (same buffer lifecycle as two_electron.rs)
  -> cart_to_sph_2e transform (sph-only, same c2s transform as base 2e)
  |
  v
cintx-compat: layout writer copies staging to caller flat buffer (unchanged)
```

The critical observation: **F12/STG/YP requires no changes to runtime, compat, capi, or core**. Only `cintx-cubecl/kernels/` gains a new file, and `kernels/mod.rs` gains new match arms.

---

## Suggested Build Order

The build order is constrained by oracle validation dependencies. Each step must achieve oracle parity before the next begins to avoid compounding failures.

### Step 1: Pending v1.1 executor items (EXEC-06 through EXEC-09, VERI-06)

**Rationale:** If the CubeCL client API migration (EXEC-06/07/08) is not yet complete, real kernel values are not flowing through the executor. All v1.2 kernel work depends on real kernel execution. Resolve these first or confirm they are complete.

**Affected:** `cintx-cubecl/src/executor.rs`, `cintx-cubecl/src/backend/`, `cintx-compat/src/raw.rs` (RecordingExecutor removal if pending).

**Gate:** All 5 base family oracle parity tests pass with 0 mismatches (confirmed per PROJECT.md v1.1 completion).

### Step 2: Helper and transform API completion

**Rationale:** Helper/transform APIs are pure host-side Rust with no kernel dependency. They unblock the remaining `helper-legacy-parity` CI gate items. Completing them expands oracle coverage without requiring GPU compute changes, providing a clean CI baseline before kernel work begins.

**Affected:** `cintx-compat/src/helpers.rs`, `cintx-compat/src/transform.rs`, `cintx-oracle/src/compare.rs`.

**Dependency on Domain 2:** Transform symbols that involve spinor conversion (`CINTc2s_ket_spinor_sf1` etc.) are partially blocked on the real spinor transform. Implement the host-callable stub transforms that are not spinor-dependent first; defer spinor-dependent transforms until Step 4.

**Gate:** `xtask helper-legacy-parity` passes for all implemented symbols.

### Step 3: Real 4c1e kernel within Validated4C1E envelope

**Rationale:** 4c1e is the only base-class family without a real kernel. It uses the same Rys quadrature infrastructure as 2e but with four centers. Completing it before F12 allows the Rys infrastructure to be stress-tested at four-center complexity before adding the operator-kernel change that F12 introduces.

**Affected:** `cintx-cubecl/src/kernels/center_4c1e.rs`, `cintx-oracle/src/fixtures.rs`, `cintx-oracle/src/compare.rs`.

**Gate:** Oracle parity for `int4c1e_cart` and `int4c1e_sph` within Validated4C1E envelope; identity test `int4c1e_sph == (-1/4pi)*trace(...)` passes; inputs outside envelope still return `UnsupportedApi`. Add `compat::workaround::int4c1e_via_2e_trace` in same PR.

### Step 4: Real spinor transform (c2spinor replacement)

**Rationale:** Spinor support is a prerequisite for oracle coverage of any spinor-form integral. Implementing the real Clebsch-Gordan coupling matrix in `c2spinor.rs` unblocks spinor oracle comparison for base 1e/2e families and unblocks the spinor-dependent transform symbols deferred from Step 2.

**Affected:** `cintx-cubecl/src/transform/c2spinor.rs`, `cintx-cubecl/src/kernels/one_electron.rs`, `cintx-cubecl/src/kernels/two_electron.rs`, `cintx-oracle/src/compare.rs` (spinor buffer contract validation).

**Gate:** Oracle parity for spinor forms of `int1e_ovlp_spinor` and `int2e_spinor` (representative symbols). Tolerance from design doc section 13.8: atol=1e-6, rtol=1e-5 for spinor/Gaunt/Breit.

### Step 5: F12/STG/YP kernel implementation

**Rationale:** F12 reuses the Rys/pdata infrastructure validated in Steps 3 and 4. The operator change (Coulomb kernel swap) is the only new physics. Sph-only restriction simplifies testing since the c2s transform is already validated.

**Affected:** `cintx-cubecl/src/kernels/f12.rs` (new), `cintx-cubecl/src/kernels/mod.rs`, `cintx-compat/src/optimizer.rs` (F12 optimizer init), `cintx-oracle/src/compare.rs` (F12 tolerance + fixtures), manifest `oracle_covered` updates.

**Gate:** Oracle parity for all 10 F12 sph-series families under `with-f12` profile; zero cart/spinor symbols in `with-f12` compiled output (verified by manifest audit).

### Step 6: Unstable-source API implementation

**Rationale:** Source-only families are validated in extended CI with relaxed guarantees. They should be the last domain because they have the weakest upstream specification and benefit from all previous infrastructure being stable.

**Affected:** `cintx-cubecl/src/kernels/` (new files behind `#[cfg(feature = "unstable-source-api")]`), `cintx-oracle/src/compare.rs`, CI nightly extended job.

**Gate:** Oracle parity in nightly extended CI with `--include-unstable-source true`.

### Step 7: Unified oracle tolerance audit

**Rationale:** After all families have real kernels and oracle coverage, run a full tolerance audit across all profiles to confirm no family is exempt from comparison and family-specific tolerances match design doc section 13.8.

**Affected:** `cintx-oracle/src/compare.rs` (tolerance constants review), `compiled_manifest.lock.json` (`oracle_covered` audit).

**Gate:** Every `stability: Stable` and `stability: Optional` manifest entry in any enabled profile has `oracle_covered: true` and a passing CI comparison record.

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Giving F12 the same canonical_family as base 2e

Mapping `int2e_stg_sph` to `canonical_family: "2e"` in the manifest would cause the dispatch to invoke `launch_two_electron` instead of `launch_f12`. The dispatch is keyed on `canonical_family`. F12/STG/YP must use a distinct canonical family string (`"f12"` or `"2e_f12"`) so the kernel registry resolves to the correct physics.

### Anti-Pattern 2: Expanding the Validated4C1E envelope without full gate evidence

The design doc (section 3.11.2) requires that any envelope expansion include oracle comparison, identity tests, 10,000 random fuzz cases, and multi-device CubeCL consistency in the same PR. Do not expand max(l) beyond 4 or add spinor support to 4c1e without all these gates. Default behavior for outside-envelope inputs must remain `UnsupportedApi`.

### Anti-Pattern 3: Tightening family tolerances below what Rys quadrature achieves on GPU

The per-family tolerances in `compare.rs` (e.g., atol=1e-7 for 3c1e, atol=1e-6 for 4c1e) are set based on upstream testsuite measurements with floating-point accumulation differences. Blindly tightening to atol=1e-12 for all families will cause oracle failures that cannot be fixed without algorithmic changes. The "unified atol=1e-12" v1.2 goal most likely means ensuring every family is under oracle comparison (no exemptions), not that every family achieves 2e-class precision.

### Anti-Pattern 4: Implementing spinor as a pure data reinterpretation

The existing `c2spinor.rs` stub treats spinor as amplitude blending of real/imaginary pairs. Real spinor output requires applying angular-momentum coupling coefficients (Clebsch-Gordan matrices). Without correct coupling, spinor values will be wrong by construction regardless of how the raw kernel output looks. The coupling matrices are tabulated in libcint `cart2sph.c` alongside the Condon-Shortley tables.

### Anti-Pattern 5: Adding unstable-source families to the stable manifest profile

Unstable-source families must only appear when the `unstable-source-api` feature is enabled. If a new source-only family is accidentally added to `compiled_in_profiles: ["base"]` in the manifest, it will break the `manifest-audit` gate for the base profile. Keep `compiled_in_profiles` for unstable families set to only the profiles where they actually compile.

### Anti-Pattern 6: Writing helper oracle comparison against a GPU-computed reference

Helper APIs (count, offset, normalization, transform functions) are pure host-side functions. They must be compared against the vendored libcint oracle's corresponding host functions, not against any GPU output. The `helper-legacy-parity` CI gate is separate from `oracle-compare` for exactly this reason. Do not route helper comparison through the CubeCL executor.

---

## Scalability Considerations

| Concern | v1.2 scope | Future consideration |
|---------|-----------|---------------------|
| Spinor representation on GPU | Host-side transform applied to staging buffer (same approach as c2s) | Future: device-side transform before D2H reduces H2D/D2H volume for large spinor outputs |
| F12 zeta parameter routing | Read from `env[PTR_F12_ZETA]` in the kernel — same env marshaling as base families | No change needed |
| Tolerance per oracle profile | Per-family constants in `compare.rs` | Consider moving tolerances into manifest schema (allows per-symbol overrides) |
| CI cost of 4-profile matrix | Current 4 profiles run in parallel — adding unstable-source as a 5th profile follows the existing matrix pattern | Feature cost stays bounded as long as CI parallelism is maintained |
| Validated4C1E classifier | Implemented in `center_4c1e.rs` — checked before kernel dispatch | Expanding the envelope is an explicit gated operation per design doc policy |

---

## Sources

- `cintx_detailed_design.md` — section 3.11.1 (F12 coverage matrix), 3.11.2 (4c1e bug envelope), 3.3.1 (manifest finalization), 13.8 (tolerance table), 14.1 (release gate) — HIGH confidence (authoritative design document)
- `crates/cintx-cubecl/src/kernels/mod.rs` — confirmed `resolve_family_name` dispatch table — HIGH confidence (codebase)
- `crates/cintx-cubecl/src/transform/c2spinor.rs` — confirmed stub implementation — HIGH confidence (codebase)
- `crates/cintx-oracle/src/compare.rs` — confirmed `tolerance_for_family`, `IMPLEMENTED_HELPER_SYMBOLS`, `IMPLEMENTED_TRANSFORM_SYMBOLS` — HIGH confidence (codebase)
- `crates/cintx-compat/src/helpers.rs` — confirmed `len_spinor`, `len_spheric`, `len_cartesian` implementations — HIGH confidence (codebase)
- `crates/cintx-ops/src/generated/api_manifest.rs` — confirmed `ManifestEntry` schema with `feature_flag`, `stability`, `oracle_covered`, `canonical_family` fields — HIGH confidence (codebase)
- `crates/cintx-ops/src/resolver.rs` — confirmed `FeatureFlag::WithF12`, `With4c1e`, `UnstableSource` variants — HIGH confidence (codebase)
- `ci/oracle-compare.yml` — confirmed `--include-unstable-source false` flag exists in oracle-compare CI step — HIGH confidence (codebase)
- `.planning/PROJECT.md` — confirmed v1.1 oracle gate closure, v1.2 requirements list — HIGH confidence (project planning artifact)
- `.planning/REQUIREMENTS.md` — confirmed v1.1 deferred items (spinor, F12, unstable-source) and EXEC-06/07/08/09 pending status — HIGH confidence (project planning artifact)
