# Detailed Test Design for `cintx` Public-API Equivalence to `libcint`

- Document status: proposed test design
- Target crate: `cintx`
- Target upstream: `libcint` 6.1.3
- Language: English
- Intended use: repository-level verification design and CI governance

---

## 1. Objective

This document defines an auditable test design for proving that `cintx`, a Rust redesign of `libcint`, is equivalent to the upstream public API surface at the levels that matter for release.

The objective is **not** to prove internal implementation similarity. The objective is to prove, with machine-checkable evidence, that:

1. the supported public API inventory is complete,
2. each supported API resolves through exactly one governed production route,
3. numerical results match `libcint` within category-specific tolerances,
4. raw compatibility contracts match the finalized specification,
5. helper / legacy / transform APIs are included in parity coverage,
6. failure behavior is controlled and reproducible,
7. feature-gated and excluded areas are reported explicitly rather than hidden.

This follows the project design principle that **result compatibility is the primary measure**, while the compiled manifest is the source of truth for the supported public surface.

---

## 2. Specification Sources

### 2.1 Normative sources

The test program shall treat the following as normative inputs:

1. `rust_crate_guideline.md`
2. `libcint_detailed_design_resolved_en_routing_amended.md`
3. vendored upstream `libcint` 6.1.3 sources and testsuite
4. committed repository artifacts such as:
   - `compiled_manifest.lock.json`
   - route coverage lock / route manifest
   - generated operator / route metadata

### 2.2 Interpretation rules

The following interpretation rules are binding for this test design:

- Passing tests alone is **not** sufficient evidence of conformance.
- Line or branch coverage alone is **not** sufficient evidence of completeness.
- “Public API complete” may only be claimed when the compiled-manifest audit passes.
- Unsupported, feature-gated, waived, and unverified areas must remain visible in reports.
- Tests must be designed to break narrow or fake implementations, not merely confirm happy paths.

---

## 3. Verification Target and Scope

### 3.1 In scope

The verification target is the `cintx` crate and the repository components that determine whether the crate is equivalent to the supported `libcint` public surface.

In scope:

- Stable integral-family APIs derived from the compiled manifest
- Helper APIs
- Legacy wrapper APIs
- Transform APIs
- Raw compatibility API behavior
- Safe API behavior
- Optional C ABI surface when enabled
- Feature-gated optional families
- Route governance and dispatch consistency
- OOM and resource-pressure stop behavior
- CPU/GPU consistency for supported families

### 3.2 Explicitly tracked scope classes

Every API row must belong to exactly one of the following classes:

- `stable`
- `optional`
- `unstable_source`
- `unsupported_policy`
- `planned_excluded`

### 3.3 Out of scope for equivalence claims

The following are not required for release equivalence claims:

- bitwise-identical floating-point results,
- internal scratch-layout equivalence,
- internal recurrence implementation equivalence,
- public GTG support.

GTG remains excluded from the support matrix and must fail closed if it appears on the public surface.

---

## 4. Repository Classification for Tool Selection

This section applies the Rust test governance guideline to `cintx`.

### 4.1 Applies because the crate includes

The crate includes:

- a **public API surface** with raw and safe entry points,
- **compile-time and configuration constraints** through feature/profile support,
- **stateful / order-dependent workflows** such as query → execute → optimizer / cache interactions,
- **unsafe code paths** in raw compatibility / FFI-adjacent areas,
- **hostile external-input surfaces** because raw `atm/bas/env/shls/dims/out/cache/opt` inputs are accepted,
- **multiple feature configurations** including optional family support,
- **high-value invariants** suitable for bounded verification,
- **backend-selection behavior** where CPU is the reference path and GPU is a governed extension.

### 4.2 Not applicable because the crate does not currently include

- crate-owned atomics / concurrency primitives that justify `loom` as a blocking release tool today.

If crate-owned atomics, lock-free structures, or multi-thread coordination logic are added later, `loom` becomes mandatory for that scope.

---

## 5. Selected Verification Tools

## 5.1 Mandatory baseline tools

The following baseline is mandatory for `cintx`:

| Tool | Required | Why it is required here | Primary output |
|---|---|---|---|
| `cargo test` | Yes | Core unit, integration, and regression execution | pass/fail by suite |
| `proptest` | Yes | Raw-contract invariants, shape invariants, layout invariants, rejection invariants | randomized invariant evidence |
| `cargo-mutants` | Yes | Detect fake implementations and weak assertions | surviving mutant report |
| `cargo-hack` | Yes | Feature/profile matrix verification | feature matrix pass/fail |
| `cargo-llvm-cov` | Yes | Coverage reporting for review, not as a completeness proxy | coverage report |
| doctests | Conditionally blocking | Required once public Rustdoc examples exist | doctest pass/fail |
| `trybuild` or `ui_test` | Conditionally blocking | Required when compile-time usage contracts are exposed publicly | compile-fail evidence |

### 5.2 Conditional tools selected for this crate

| Tool | Decision | Applies because the crate includes | Target scope |
|---|---|---|---|
| `proptest-state-machine` | Yes | stateful query/evaluate/cache/optimizer workflows | raw compat lifecycle, optimizer lifecycle |
| `Kani` | Yes | bounded, high-value invariants exist | manifest normalization, dims rules, writer sizing |
| `Miri` | Yes | unsafe code exists and raw memory contracts matter | unsafe raw writers, pointer-adjacent APIs, aliasing assumptions |
| `loom` | Not applicable now | no crate-owned atomics or lock-based concurrency model currently governs public correctness | reevaluate if concurrency primitives are introduced |
| `cargo-fuzz` | Yes | hostile external input surfaces exist in raw compat and C-like input validation | raw validator, manifest/route parser inputs, C ABI shims |
| `criterion` | Yes, non-blocking in PR / blocking for regression reports before release | design requires regression/baseline tracking | performance and fallback-rate regression |

### 5.3 Tooling not allowed to substitute for conformance

The following shortcuts are prohibited:

- Declaring parity based only on wrapper parity tests
- Declaring completeness based only on line coverage
- Treating safe/raw equality as sufficient without oracle comparison
- Treating symbol presence as sufficient without governed route coverage
- Treating release gates as satisfied when waivers or skipped tools are unreported

---

## 6. Test Architecture

The test architecture is divided into seven verification layers. Release claims require evidence from **all applicable layers**.

### 6.1 Layer A — Inventory and manifest governance

Purpose:
- prove what the public denominator is,
- prevent silent drift in upstream exposure,
- separate supported APIs from excluded or unstable ones.

Primary evidence:
- `compiled_manifest.lock.json`
- generated compiled-manifest diff report
- stable/optional/unstable classification report

### 6.2 Layer B — Route governance and dispatch integrity

Purpose:
- prove that each policy-supported denominator row has exactly one production route,
- prevent ad-hoc bypasses,
- guarantee resolver equivalence across safe/raw/C ABI surfaces.

Primary evidence:
- route coverage lock
- route completeness report
- resolver-equivalence report
- bypass-detection report

### 6.3 Layer C — Raw compatibility contracts

Purpose:
- prove finalized contracts for `out == NULL`, `cache == NULL`, `dims == NULL`, and strict `dims != NULL` validation,
- prove output-buffer sizing and no-partial-write semantics.

Primary evidence:
- raw contract matrix report
- sentinel-path report
- failure-semantics report

### 6.4 Layer D — Numerical oracle parity

Purpose:
- prove numerical equivalence to upstream within family-specific tolerances.

Primary evidence:
- oracle diff report
- per-family tolerance report
- safe-vs-raw-vs-wrapper-vs-oracle comparison logs

### 6.5 Layer E — Helper / legacy / transform parity

Purpose:
- prevent “integrals-only” parity claims while helpers and wrappers drift.

Primary evidence:
- helper parity matrix
- transform parity matrix
- optimizer lifecycle parity matrix
- legacy wrapper parity report

### 6.6 Layer F — Failure and resource-pressure behavior

Purpose:
- prove OOM-safe stop behavior,
- prove controlled failures for invalid input and unsupported policy,
- prevent silent partial writes.

Primary evidence:
- fail-allocator results
- memory-limit chunking results
- device-OOM mock results
- no-partial-write audit

### 6.7 Layer G — Backend consistency and regression tracking

Purpose:
- prove CPU/GPU consistency where GPU support exists,
- track performance and fallback regressions without conflating them with correctness.

Primary evidence:
- CPU/GPU consistency matrix
- fallback-rate regression report
- criterion baseline diff
- peak RSS / workspace-byte regression report

---

## 7. Public-Surface Denominator

### 7.1 Source of truth

The denominator for public-surface coverage is the committed compiled lock, generated from the support matrix:

- `base`
- `with-f12`
- `with-4c1e`
- `with-f12+with-4c1e`

Coverage claims must not be based on design-time CSV snapshots alone.

### 7.2 Denominator rules

For each lock row:

1. normalize the symbol to `canonical_family`,
2. classify by `stability`, `feature_flag`, `representation`, `helper_kind`, and `declared_in`,
3. mark whether the row is in the public denominator,
4. map it to exactly one verification policy:
   - oracle-covered stable,
   - oracle-covered optional,
   - nightly-only unstable,
   - explicit unsupported-policy,
   - planned-excluded.

### 7.3 Excluded-from-denominator rows

The following must remain visible but excluded from public coverage denominator counts:

- Fortran wrapper auxiliary symbols,
- toolchain-dependent aliases marked auxiliary,
- GTG profile artifacts,
- any diagnostic-only route not linked into production code.

---

## 8. Detailed Test Design by Verification Layer

## 8.1 Layer A — Inventory and manifest governance

### 8.1.1 Test objectives

- Detect drift between generated and committed compiled manifests
- Detect normalization failures
- Detect unsupported hidden exports
- Enforce explicit lock-update policy

### 8.1.2 Required tests

#### A1. Compiled manifest snapshot equivalence

- Regenerate the compiled lock from vendored upstream profiles
- Canonicalize JSON
- Assert exact equality with committed lock

Pass condition:
- zero diff

Fail condition:
- any addition, deletion, or metadata drift without explicit approval

#### A2. Symbol normalization unit tests

Target:
- `canonicalize_symbol()`
- wrapper prefix/suffix normalization
- optimizer suffix normalization
- representation suffix normalization

Pass condition:
- all known examples normalize to expected `canonical_family`

#### A3. Declared-in / helper-kind classification tests

Target:
- header-derived vs source-derived classification
- helper / transform / optimizer lifecycle / legacy classification

Pass condition:
- machine-generated classifications match curated expectation fixtures

#### A4. Lock drift policy tests

Target:
- ensure lock updates require explicit version/change reason metadata

Pass condition:
- drift without policy annotation fails

### 8.1.3 Recommended implementation location

- Keep existing phase-3 governance tests
- Add:
  - `tests/phase4_manifest_profile_union.rs`
  - `tests/phase4_symbol_normalization.rs`
  - `tests/phase4_manifest_classification.rs`

---

## 8.2 Layer B — Route governance and dispatch integrity

### 8.2.1 Test objectives

- Ensure exactly one implemented production route per supported denominator row
- Ensure explicit `unsupported_policy` entries exist where required
- Ensure safe/raw/C ABI all resolve through the same route ID
- Ensure no production bypass exists outside shared resolution

### 8.2.2 Required tests

#### B1. Route completeness audit

For each supported row derived from the compiled lock and policy predicates:

- assert one and only one `implemented` route exists,
- assert route metadata matches representation, feature flag, stability, optimizer mode, and backend set.

Pass condition:
- no missing route,
- no duplicate route,
- no malformed implemented route.

#### B2. Unsupported-policy explicitness audit

For each denied combination that exists by policy but is intentionally unsupported:

- assert presence of an explicit `unsupported_policy` row.

Pass condition:
- absence is forbidden; explicit denial rows must exist.

#### B3. Surface resolver equivalence

For the same request key:

- resolve through safe surface,
- resolve through raw surface,
- resolve through C ABI surface,
- assert identical `route_id`, `entry_kernel`, and policy outcome.

#### B4. Bypass detection

Static audit plus targeted tests to ensure production code does not bypass `resolve_route()`.

Implementation options:
- source scan in `xtask`,
- denylist-based grep in CI,
- narrow AST inspection if introduced later.

Pass condition:
- no facade / compat / executor family-specific production bypass.

### 8.2.3 Recommended implementation location

- Extend existing route-audit coverage
- Add:
  - `tests/phase4_route_denominator_expansion.rs`
  - `tests/phase4_route_bypass_detection.rs`
  - `tests/phase4_capi_route_equivalence.rs`

---

## 8.3 Layer C — Raw compatibility contracts

### 8.3.1 Test objectives

- Prove finalized `dims` semantics
- Prove sentinel behavior
- Prove buffer and layout correctness
- Prove no-partial-write behavior

### 8.3.2 Required contract matrix

The raw contract matrix shall vary these dimensions:

- family category: `1e`, `2e`, `2c2e`, `3c1e`, `3c2e`, `4c1e`
- representation: `cart`, `sph`, `spinor`
- `out`: null / provided
- `dims`: null / natural / too small / too large / wrong arity
- `cache`: null / minimum valid / too short
- `opt`: null / provided
- output buffer: exact / short / oversized

### 8.3.3 Required tests

#### C1. Workspace query semantics

- `out == NULL` must return required workspace / output sizing information.
- Query results must match the corresponding execute path.

#### C2. `dims == NULL` natural-shape semantics

- Natural shape must be computed from representation and shell tuple.
- Safe and raw query paths must agree on natural shape.

#### C3. Strict `dims != NULL` validation

Rules to verify:

- `1e` / `2c2e` require exactly 2 dims
- `3c1e` / `3c2e` require exactly 3 dims
- `2e` / `4c1e` require exactly 4 dims
- `comp` is not carried in `dims`
- smaller-than-natural fails
- larger-than-natural fails
- wrong arity fails

#### C4. Buffer size and writer contract

- required elements and required bytes must match writer contract
- flat layout must match libcint ordering
- spinor complex interleaving must match contract

#### C5. No partial write

- any error after output buffer handoff must leave caller-visible “success” state impossible
- where sentinel filling is available in test harness, verify unchanged bytes outside allowed success cases

#### C6. Invalid raw layout rejection

Cover at minimum:

- misaligned `atm` slots,
- misaligned `bas` slots,
- out-of-range env offsets,
- negative or nonsensical pointers,
- invalid shell tuple arity,
- inconsistent optimizer/cache combinations.

### 8.3.4 Property-based augmentation

Use `proptest` for:

- natural-shape invariance,
- `required_bytes == required_elements * element_size`,
- invalid dims never succeed,
- shell tuple arity mismatches always reject,
- query/execute consistency under randomized valid inputs.

### 8.3.5 Recommended implementation location

- Extend existing phase-2 raw suites
- Add:
  - `tests/phase4_raw_dims_matrix.rs`
  - `tests/phase4_raw_writer_contract.rs`
  - `tests/phase4_raw_layout_fuzz_seeded.rs`

---

## 8.4 Layer D — Numerical oracle parity

### 8.4.1 Test objectives

- Prove result compatibility against vendored upstream `libcint`
- Prove parity across safe/raw/wrapper surfaces
- Prove tolerance policy is category-specific and enforced

### 8.4.2 Comparison sources

For each tested route, compare:

- `cintx` safe output,
- `cintx` raw output,
- legacy/wrapper output where applicable,
- upstream oracle output.

### 8.4.3 Tolerance policy

The suite shall use category-specific thresholds:

| Category | atol | rtol |
|---|---:|---:|
| Basic 1e | `1e-11` | `1e-9` |
| 2e plain / low derivative | `1e-12` | `1e-10` |
| 2c2e / 3c2e | `1e-9` | `1e-7` |
| High-order 3c1e | `1e-7` | `1e-5` |
| spinor / Gaunt / Breit | `1e-6` | `1e-5` |
| 4c1e | `1e-6` | `1e-5` |
| F12 / STG / YP | `1e-6` | `1e-4` |

When `abs(reference) < zero_threshold`, compare by absolute error only.

Default `zero_threshold`:
- `1e-18`

### 8.4.4 Case-generation strategy

#### Deterministic matrix

Generate a deterministic matrix covering:

- each family category,
- each supported representation,
- low/medium/high angular momentum buckets,
- scalar and derivative operators,
- optimizer on/off,
- natural-shape evaluation.

#### Profile matrix

Run under:

- base
- with-f12
- with-4c1e
- with-f12+with-4c1e
- optional nightly profile for `unstable-source-api`

#### Differential promotion

If the manifest changes, the changed families must be automatically promoted into the oracle suite.

### 8.4.5 Required tests

#### D1. Stable oracle matrix

Blocking for release.

#### D2. Safe vs raw vs wrapper equivalence

Blocking for release for all stable denominator rows.

#### D3. F12 sph-only matrix

Blocking when `with-f12` is enabled.

Rules:
- sph-family comparisons must pass,
- cart/spinor symbol count must be zero,
- cart/spinor absence is a pass condition, not a coverage gap.

#### D4. 4c1e oracle and identity matrix

Blocking when `with-4c1e` is enabled.

Rules:
- oracle comparison only inside `Validated4C1E`,
- identity tests must pass,
- outside the validated region must return `UnsupportedApi`.

### 8.4.6 Recommended implementation location

- Extend existing parity files
- Add:
  - `tests/phase4_oracle_full_matrix.rs`
  - `tests/phase4_oracle_profile_matrix.rs`
  - `tests/phase4_f12_sph_only.rs`
  - `tests/phase4_4c1e_validated_envelope.rs`

---

## 8.5 Layer E — Helper / legacy / transform parity

### 8.5.1 Test objectives

- Include all public non-integral helpers in parity coverage
- Prevent wrapper drift hidden behind integral parity

### 8.5.2 Required test categories

- AO counts and shell offsets
- normalization helpers such as `gto_norm`
- transform helpers
- optimizer init / build / delete / lifecycle
- legacy wrappers (`cNAME*`, `cint2e_*` style wrappers where exposed)

### 8.5.3 Required tests

#### E1. Helper parity matrix

For every helper public row in the manifest/helper table:
- compare upstream reference behavior to `cintx` behavior.

#### E2. Transform parity

- verify transform results for supported representative shells,
- verify layout and ordering consistency.

#### E3. Optimizer lifecycle parity

- build optimizer,
- use optimizer across multiple calls,
- destroy or release handle,
- confirm value invariance with and without optimizer.

#### E4. Legacy wrapper return semantics

- verify boolean/nonzero return semantics,
- verify output flat layout,
- verify argument forwarding contracts.

### 8.5.4 Recommended implementation location

- Extend current phase-3 helper/optimizer suites
- Add:
  - `tests/phase4_legacy_wrapper_contracts.rs`
  - `tests/phase4_helper_manifest_coverage.rs`

---

## 8.6 Layer F — Failure and resource-pressure behavior

### 8.6.1 Test objectives

- Prove controlled typed failures
- Prove fail-closed semantics under memory pressure
- Prove no undefined partial success exposure

### 8.6.2 Required failure-path coverage

At minimum:

- invalid offset,
- invalid shell tuple arity,
- feature disabled,
- short output buffer,
- invalid dims,
- GPU unavailable,
- backend launch failure,
- unsupported policy route,
- invalid optimizer/cache pairing.

### 8.6.3 Required OOM / pressure tests

#### F1. Fail allocator after N bytes

Inject allocator failures at deterministic byte thresholds.

Pass condition:
- typed error returned,
- no false success,
- no partial-write success path.

#### F2. Memory-limit chunking

Set low `memory_limit_bytes` values that force chunking.

Pass condition:
- successful chunking when policy allows,
- `MemoryLimitExceeded` when chunking cannot satisfy the contract.

#### F3. Device OOM mock

Pass condition:
- typed device error and clean failure behavior.

#### F4. Pool exhaustion simulation

Pass condition:
- no uncontrolled fallback to aborting allocation paths,
- deterministic typed error or governed fallible allocation path.

### 8.6.4 Recommended implementation location

- Extend phase-2 allocation/memory suites
- Add:
  - `tests/phase4_fail_allocator_matrix.rs`
  - `tests/phase4_chunking_contracts.rs`
  - `tests/phase4_device_failure_contracts.rs`

---

## 8.7 Layer G — Backend consistency and regression tracking

### 8.7.1 Test objectives

- Prove CPU/GPU numerical consistency for supported families
- Track performance and fallback changes without overstating their meaning

### 8.7.2 Required tests

#### G1. CPU/GPU consistency matrix

For supported family/representation combinations:
- compare CPU reference results to GPU results under matching inputs.

Pass condition:
- within category tolerance

#### G2. Fallback-rate regression

- compare current fallback reasons/rates to baseline
- fail only on policy-defined regressions, not on any raw fluctuation

#### G3. Workspace-byte / peak-RSS regression

- compare against baseline thresholds
- report regressions separately from numerical failures

#### G4. Criterion baseline diff

- benchmark representative stable families,
- do not treat benchmark pass as correctness evidence.

### 8.7.3 Recommended implementation location

- Add nightly-only suites and benches:
  - `tests/phase4_cpu_gpu_consistency.rs`
  - `benches/oracle_baseline.rs`
  - `benches/backend_crossover.rs`

---

## 9. Special Policy Areas

## 9.1 F12 / STG / YP policy

The design fixes F12/STG/YP support as **sph-only** in the current upstream snapshot.

Therefore:

- sph coverage is required when `with-f12` is enabled,
- cart/spinor coverage is not missing work; it is formally out of scope,
- release gating must assert zero cart/spinor public symbol count under the F12 profile.

## 9.2 4c1e policy

`4c1e` support is constrained to `Validated4C1E`.

Therefore:

- release blocking oracle comparison applies only inside the validated envelope,
- rejection tests are mandatory outside the envelope,
- expanding the envelope requires, in the same PR:
  - oracle additions,
  - identity tests,
  - 10,000-case randomized fuzzing,
  - CPU/GPU consistency coverage for the expanded region.

## 9.3 GTG policy

GTG is roadmap-only and excluded.

Therefore:

- no public feature gate,
- no public manifest exposure,
- no route entry,
- any GTG symbol appearance on the public surface is a release-blocking failure.

---

## 10. Property-Based, State-Machine, Fuzz, and Bounded Verification Design

## 10.1 `proptest` design

Use `proptest` for invariants that should hold across broad input ranges:

- raw natural-shape calculation,
- required-bytes calculation,
- invalid dims rejection,
- shell-arity rejection,
- optimizer invariance,
- chunking invariance,
- route resolver determinism.

### 10.2 `proptest-state-machine` design

Model the following stateful workflows:

#### SM1. Query / execute lifecycle

States:
- Initialized
- WorkspaceQueried
- Executed
- Reused
- Failed

Transitions:
- query workspace,
- allocate output,
- execute,
- reuse workspace,
- change optimizer state,
- inject invalid call.

Invariant:
- successful execute must agree with query metadata,
- failed execute must not create a success state,
- route identity must remain stable for identical requests.

#### SM2. Optimizer lifecycle

States:
- NoOptimizer
- OptimizerBuilt
- OptimizerUsed
- OptimizerReleased

Invariant:
- value results remain unchanged with or without optimizer,
- invalid handle/state transitions fail cleanly.

### 10.3 `cargo-fuzz` design

Fuzz targets shall include:

- raw `atm/bas/env/shls/dims` validator,
- route request parser / canonicalization if exposed by tooling,
- C ABI argument boundary functions,
- manifest parser if externalized.

Required outcomes:
- no panics in safe Rust,
- no UB in unsafe paths when paired with Miri / sanitizers where applicable,
- invalid inputs reject with controlled errors.

### 10.4 `Kani` design

Use bounded verification for small but high-value invariants:

- `dims` arity and size formulas,
- required-elements / required-bytes monotonicity,
- symbol normalization examples,
- route uniqueness logic on small synthesized route tables,
- no-partial-write preconditions for writer-size checks.

### 10.5 `Miri` design

Run Miri on:

- unsafe raw writer helpers,
- pointer-adjacent conversion helpers,
- small deterministic raw contract tests,
- no-partial-write sentinel tests where practical.

---

## 11. CI Gate Design

## 11.1 PR gates

PR gates must be fast enough for regular development but strong enough to block false progress.

### Required PR jobs

| Job | Blocking | Scope |
|---|---|---|
| manifest audit | Yes | compiled lock drift, normalization, classification |
| route audit | Yes | completeness, resolver equivalence, unsupported-policy explicitness |
| stable smoke oracle | Yes | representative stable families |
| helper/transform smoke parity | Yes | helper and transform surface |
| raw contract matrix smoke | Yes | sentinel and dims rules |
| fail-allocator smoke | Yes | fail-closed behavior |
| `cargo-hack` feature matrix smoke | Yes | selected feature combinations |
| `trybuild` / `ui_test` if applicable | Yes | compile-time contracts |
| `Miri` narrow unsafe suite | Yes | critical unsafe paths |
| `cargo-llvm-cov` report | No, but must publish artifact | review aid |

### PR gate requirements

A PR may merge only when:

- no manifest drift is unapproved,
- no stable supported denominator row lacks a route,
- smoke oracle suites pass,
- raw contract smoke passes,
- helper smoke passes,
- no new GTG exposure appears.

## 11.2 Nightly gates

Nightly gates provide full-matrix assurance and broader search.

### Required nightly jobs

| Job | Blocking for nightly health | Scope |
|---|---|---|
| full oracle matrix | Yes | all stable denominator rows |
| optional family matrix | Yes | `with-f12`, `with-4c1e` |
| unstable-source extended CI | Yes for that scope | `unstable-source-api` |
| `cargo-mutants` | Yes | mutation score and surviving mutant review |
| full `cargo-hack` matrix | Yes | feature combinations |
| `proptest` extended seeds | Yes | longer invariant search |
| `proptest-state-machine` | Yes | lifecycle workflows |
| `cargo-fuzz` seeded corpus run | Yes | hostile input robustness |
| CPU/GPU consistency | Yes where GPU support exists | backend parity |
| criterion baseline collection | No correctness block, yes report block | regression artifacts |

## 11.3 Release gates

Before release, the following must pass:

1. compiled manifest audit across the support matrix,
2. route audit with exact one-route guarantee for every policy-supported denominator row,
3. all stable oracle comparisons,
4. F12 sph-only gate and zero cart/spinor symbol count under `with-f12`,
5. `Validated4C1E` oracle + identity + rejection gates under `with-4c1e`,
6. OOM/resource-pressure gates,
7. CPU/GPU consistency for supported families,
8. helper / legacy / transform parity,
9. GTG non-appearance gate,
10. unresolved waivers reviewed and either cleared or explicitly deferred out of release scope.

---

## 12. Reporting and Artifacts

Every CI tier shall publish structured artifacts.

### 12.1 Required artifacts

| Artifact | Produced by | Purpose |
|---|---|---|
| `api_coverage.json` | manifest/oracle audit | denominator accounting |
| `route_coverage.json` | route audit | route completeness accounting |
| `oracle_diff_report.md` | oracle suite | human-readable mismatch review |
| `helper_parity_report.md` | helper suite | non-integral public parity |
| `failure_semantics_report.md` | failure suites | typed-failure evidence |
| `cpu_gpu_consistency.json` | backend suite | backend parity evidence |
| `mutants_report.md` | mutation job | surviving mutants and waivers |
| `coverage/index.html` | coverage job | review-only coverage artifact |
| `criterion_baseline_diff.md` | benchmark job | regression visibility |

### 12.2 Required summary sections in reports

Every consolidated test report must include these headings:

- `Verified in scope:`
- `Not yet verified:`
- `Blocked by:`
- `Waived until:`
- `Residual risks:`

---

## 13. Waiver Policy

Waivers are allowed only for non-release scope or temporarily blocked work, never as silent exceptions.

### 13.1 A waiver must include

- exact test/tool name,
- exact impacted API scope,
- reason for waiver,
- risk statement,
- owner,
- expiry condition,
- expiry date or milestone,
- compensating controls if any.

### 13.2 Waivers are forbidden for

- missing stable manifest rows,
- missing stable production routes,
- GTG public-surface exposure,
- unresolved raw contract ambiguity,
- unreported surviving mutants in stable core logic before release.

---

## 14. Current-State Integration Plan for This Repository

This section maps the design onto the current `cintx` repository layout.

### 14.1 Existing useful scaffolding

The repository already contains the following useful foundations:

- phase-1 contract/error tests,
- phase-2 raw/memory/oracle tests,
- phase-3 manifest/route/helper/optimizer governance tests,
- vendored upstream `libcint`,
- committed `compiled_manifest.lock.json`,
- GitHub workflows for PR and release governance gates.

### 14.2 Recommended incremental implementation order

#### Step 1 — make denominator accounting explicit

Add machine-generated `api_coverage.json` from the compiled lock and profile matrix.

#### Step 2 — strengthen route denominator expansion

Expand from currently implemented route subsets to all policy-supported denominator rows, including explicit unsupported rows.

#### Step 3 — complete raw contract matrix

Close gaps around dims arity, too-large dims, no-partial-write auditing, and spinor layout writer evidence.

#### Step 4 — promote full oracle generation from manifest rows

Stop relying on hand-curated representative subsets as release evidence.

#### Step 5 — add required governance tools

Introduce:
- `cargo-hack`
- `cargo-mutants`
- `proptest`
- `proptest-state-machine`
- `Miri`
- `cargo-fuzz`
- `Kani`

#### Step 6 — separate PR, nightly, and release reports

Publish machine-readable artifacts and human-readable summaries per CI tier.

---

## 15. Recommended File and Target Layout

A repository-compatible layout is below.

```text
xtask/
  src/
    manifest_audit.rs
    route_audit.rs
    api_coverage.rs
    report.rs
    bypass_scan.rs

fuzz/
  fuzz_targets/
    raw_validator.rs
    capi_boundary.rs
    manifest_parser.rs

tests/
  phase4_manifest_profile_union.rs
  phase4_symbol_normalization.rs
  phase4_manifest_classification.rs
  phase4_route_denominator_expansion.rs
  phase4_route_bypass_detection.rs
  phase4_capi_route_equivalence.rs
  phase4_raw_dims_matrix.rs
  phase4_raw_writer_contract.rs
  phase4_oracle_full_matrix.rs
  phase4_oracle_profile_matrix.rs
  phase4_f12_sph_only.rs
  phase4_4c1e_validated_envelope.rs
  phase4_legacy_wrapper_contracts.rs
  phase4_helper_manifest_coverage.rs
  phase4_fail_allocator_matrix.rs
  phase4_chunking_contracts.rs
  phase4_device_failure_contracts.rs
  phase4_cpu_gpu_consistency.rs

benches/
  oracle_baseline.rs
  backend_crossover.rs

kani/
  dims_contract.rs
  route_uniqueness.rs
  writer_size_contract.rs

trybuild/
  *.rs
```

---

## 16. Acceptance Criteria

## 16.1 Verified in scope

`cintx` may claim public-API equivalence for the supported surface only when all of the following are true:

1. compiled manifest audit passes,
2. route audit passes,
3. stable oracle matrix passes,
4. helper / legacy / transform parity passes,
5. raw contract matrix passes,
6. OOM/failure semantics pass,
7. optional-family profile gates pass when enabled,
8. CPU/GPU consistency passes for supported GPU families,
9. required governance reports are generated with no hidden scope.

## 16.2 Not yet verified

The project must state “Not yet verified” for any of the following when applicable:

- unstable-source families not included in nightly extended CI,
- compile-time contracts lacking `trybuild`/`ui_test`,
- public docs lacking doctests,
- GPU families not yet in the supported consistency matrix,
- waived mutation survivors,
- newly introduced unsafe code not yet covered by Miri.

## 16.3 Blocked by

A release is blocked by any of the following:

- manifest drift without approval,
- missing stable route,
- failing oracle comparison,
- unresolved F12 or 4c1e policy mismatch,
- GTG symbol appearance,
- raw no-partial-write failure,
- uncategorized surviving mutants in stable core logic.

---

## 17. Residual Risks

Even after this design is implemented, the following residual risks remain and must be reported explicitly:

1. Floating-point non-bitwise equivalence remains expected.
2. GPU coverage may lag CPU coverage for some families.
3. Mutation testing cannot prove absence of all logical faults.
4. Bounded verification proves only the bounded property region.
5. Fuzzing improves robustness but cannot prove exhaustive hostile-input coverage.
6. Performance regression signals are informative but not correctness evidence.

---

## 18. Clear Status Language for Project Use

The repository should use the following phrases in reports and release notes.

### Approved wording

- `Verified in scope:`
- `Not yet verified:`
- `Blocked by:`
- `Waived until:`
- `Applies because the crate includes:`
- `Not applicable because the crate does not include:`

### Prohibited wording

- `fully tested`
- `all good`
- `safe`
- `spec-complete`

---

## 19. Final Release Statement Template

Use the following template only when the required gates pass.

```md
Verified in scope:
- Stable public APIs in the compiled manifest denominator
- Governed safe/raw/C ABI route equivalence for policy-supported combinations
- Oracle parity within family-specific tolerances
- Helper / legacy / transform parity
- Raw compatibility contracts including strict dims and no-partial-write behavior
- OOM/resource-pressure stop behavior
- Optional profile gates for enabled release features

Not yet verified:
- [list any unstable-source or deferred scopes]

Blocked by:
- none

Residual risks:
- Floating-point non-bitwise variance
- Any explicitly listed deferred scopes
```

---

## 20. Summary

This design turns `cintx` public-API equivalence into a governed verification program rather than a collection of ad-hoc parity tests.

Its core rule is simple:

- **manifest defines what counts,**
- **route governance defines how it may execute,**
- **oracle and contract tests define whether it is equivalent,**
- **reports define what is still not verified.**

That is the minimum evidence standard required to claim that `cintx` implements the supported public API surface of `libcint` at release quality.
