# Codebase Concerns

**Analysis Date:** 2026-03-21

## Tech Debt

**Deterministic fallback execution path in core runtime:**
- Issue: When a specialized route does not run, both safe and raw execution paths generate seeded deterministic values instead of invoking libcint kernels.
- Files: `src/runtime/executor.rs`, `src/runtime/raw/evaluate.rs`, `tests/phase2_raw_query_execute.rs`, `tests/common/oracle_runner.rs`
- Impact: Runtime behavior depends on synthetic filler logic (`fill_real_values` / `fill_spinor_values`), which makes correctness rely on deterministic contracts instead of physical integrals for fallback paths.
- Fix approach: Split fallback behavior behind an explicit feature flag/profile and make production/default execution fail closed for unresolved kernels instead of writing synthetic payloads.

**High duplication and monolithic backend implementation:**
- Issue: CPU backend execution and cache-query code repeats similar pointer conversion and dispatch logic across many large functions.
- Files: `src/runtime/backend/cpu/mod.rs`, `src/runtime/backend/cpu/overlap_cartesian.rs`, `src/runtime/backend/cpu/router.rs`
- Impact: Regression risk is high during edits because route behavior is spread across large files and repeated patterns.
- Fix approach: Extract shared call adapters (input marshalling, cache query normalization, error mapping) into smaller reusable modules and reduce per-route boilerplate.

**Heavy vendored source footprint and hard build dependency:**
- Issue: Repository includes both an extracted vendored tree and archive (`libcint-master` and `libcint-master.zip`), and build always depends on local vendored sources.
- Files: `build.rs`, `libcint-master`, `libcint-master.zip`
- Impact: Larger clone/build footprint and tighter coupling to local vendored content; dependency updates are operationally heavier.
- Fix approach: Keep a single source of truth for vendored artifacts and document/update flow; optionally fetch pinned sources in CI/build scripts instead of storing both expanded + zipped copies.

## Known Bugs

**PR workflow path filter references missing docs paths:**
- Symptoms: Workflow trigger filters reference files that are not present in `docs/`, so those path entries can never match.
- Files: `.github/workflows/compat-governance-pr.yml`, `docs/cintx_detailed_test_design_en.md`, `docs/libcint_detailed_design_resolved_en_routing_amended.md`, `docs/libcint_route_coverage_manifest_spec.md`
- Trigger: Any PR that relies on `docs/phase2-support-matrix.md` or `docs/phase3-governance-gates.md` path filters.
- Workaround: Update `paths:` in workflow to existing docs or restore the missing docs at the referenced paths.

## Security Considerations

**Raw optimizer pointer is trust-based at FFI boundary:**
- Risk: `opt` (`NonNull<c_void>`) is validated only for presence with cache, then forwarded into C calls without provenance/lifetime validation.
- Files: `src/runtime/raw/views.rs`, `src/runtime/raw/validator.rs`, `src/runtime/backend/cpu/mod.rs`, `src/runtime/backend/cpu/ffi.rs`
- Current mitigation: Requires cache when `opt` is present and enforces typed raw layout checks.
- Recommendations: Gate raw optimizer pointer usage behind explicit unsafe API contract docs; add optional nulling/sanitization path for untrusted callers; add invariant tests for invalid optimizer handles that must not cross FFI.

**Large unsafe call surface to libcint kernels:**
- Risk: Many `unsafe extern "C"` calls pass mutable raw pointers for `atm`, `bas`, `env`, `dims`, `shls`, `cache`, and output buffers.
- Files: `src/runtime/backend/cpu/ffi.rs`, `src/runtime/backend/cpu/mod.rs`
- Current mitigation: Input validation paths exist (`validate_raw_contract`, checked conversions, shape checks).
- Recommendations: Consolidate unsafe blocks into smaller audited wrappers and add Miri/sanitizer runs for FFI boundary tests.

## Performance Bottlenecks

**Repeated hot-path cloning before kernel calls:**
- Problem: FFI query/execute functions repeatedly clone `atm`, `bas`, and `env` (`to_vec`) before each call.
- Files: `src/runtime/backend/cpu/mod.rs`
- Cause: C APIs require mutable pointers; current implementation clones slices into owned mutable buffers each invocation.
- Improvement path: Reuse scratch buffers/pools or build prevalidated mutable working sets once per request to avoid per-call cloning overhead.

**Chunked fallback path incurs extra staging allocation/copy:**
- Problem: Chunked write paths allocate staging buffers and copy per chunk while also synthesizing scalar values.
- Files: `src/runtime/executor.rs`, `src/runtime/raw/evaluate.rs`
- Cause: Fallback writer is implemented as staged chunk synthesis + copy loops.
- Improvement path: Write directly into destination slices where possible and reserve staged buffering only for true transform cases.

**No benchmark harness detected for runtime hot paths:**
- Problem: No benchmark directories/config for regression tracking of runtime throughput and allocation behavior.
- Files: `Cargo.toml`, `src/runtime/`, `tests/`
- Cause: Test suite is correctness-focused; no perf gate exists.
- Improvement path: Add criterion benches for `safe::evaluate`, `raw::evaluate_compat`, route resolution, and memory planner under representative basis sizes.

## Fragile Areas

**Route/manifest governance spans multiple lock artifacts and generated views:**
- Files: `src/runtime/backend/cpu/router.rs`, `src/runtime/backend/cpu/route_coverage_manifest.lock.json`, `compiled_manifest.lock.json`, `src/manifest/compiled.rs`, `src/bin/manifest_audit.rs`
- Why fragile: Route policy, route lock JSON, and compiled manifest lock must stay synchronized; drift requires governance tooling to detect/fix.
- Safe modification: Update route entries and regenerate/audit lock artifacts in the same change, then run manifest audit tests and binary checks.
- Test coverage: Governance tests exist (`tests/phase3_manifest_governance.rs`, `tests/phase3_compiled_manifest_audit.rs`, `tests/phase3_route_audit.rs`) but modifications still span several files.

**Very large backend and parity test files:**
- Files: `src/runtime/backend/cpu/overlap_cartesian.rs`, `src/runtime/backend/cpu/mod.rs`, `tests/three_c_wrapper_parity.rs`, `tests/four_c_wrapper_parity.rs`
- Why fragile: High line counts and broad responsibilities increase merge conflict risk and make behavioral diffs hard to isolate.
- Safe modification: Prefer narrow helper extraction and route-specific modules before adding more families/representations.
- Test coverage: Integration parity suites are broad, but they are also large and tightly coupled to fixture assumptions.

## Scaling Limits

**Coverage envelope intentionally excludes several route combinations:**
- Current capacity: Implemented routes are policy-bounded; explicit unsupported entries remain for multiple family/operator/representation combinations.
- Limit: Unsupported routes fail by policy (`UnsupportedApi`) rather than auto-falling to new kernels.
- Scaling path: Add route manifest entries + backend implementations + parity gates together for each new envelope.
- Files: `src/runtime/backend/cpu/router.rs`, `tests/phase3_route_audit.rs`, `tests/common/phase2_fixtures.rs`

**Optional 4c1e route has strict policy envelope:**
- Current capacity: 4c1e requires `with-4c1e`, `backend_candidate == "cpu"`, natural dims, and angular momentum bounds.
- Limit: 4c1e spinor remains unsupported and out-of-envelope.
- Scaling path: Extend policy + route entries + backend/kernel support and update governance profile coverage.
- Files: `src/runtime/policy.rs`, `src/runtime/backend/cpu/router.rs`, `tests/four_c_wrapper_parity.rs`

## Dependencies at Risk

**Dependency hygiene currently drifting from usage:**
- Risk: Clippy with `-W unused-crate-dependencies` reports unused dependencies in crate root.
- Impact: Unused dependencies increase compile surface and maintenance overhead.
- Migration plan: Remove unused dependencies or intentionally mark them with `use ... as _;` plus rationale.
- Files: `Cargo.toml`, `src/lib.rs`

**Vendored libcint lifecycle risk:**
- Risk: Project is pinned to vendored libcint source tree and build assumptions in `build.rs`.
- Impact: Upstream security/bugfix updates require manual vendored refresh and lock/governance retesting.
- Migration plan: Add documented vendored update procedure and checksum/version verification step in CI.
- Files: `build.rs`, `libcint-master`, `libcint-master.zip`

## Missing Critical Features

**No broad default CI quality gate for full workspace checks:**
- Problem: Existing workflows run selected governance/parity tests; no workflow runs full `cargo test --workspace`, clippy, or rustfmt checks.
- Blocks: Detecting regressions outside selected gate tests and keeping lint/style health continuously enforced.
- Files: `.github/workflows/compat-governance-pr.yml`, `.github/workflows/compat-governance-release.yml`

**Project-level user entry/documentation baseline is minimal:**
- Problem: Root docs and binary entrypoint do not describe or expose runtime functionality.
- Blocks: Operator usability for new contributors and consumers expecting a functional CLI entrypoint.
- Files: `README.md`, `src/main.rs`

## Test Coverage Gaps

**No in-crate unit tests detected under `src/`:**
- What's not tested: Module-local behaviors via unit-test targets (`#[cfg(test)]`/`#[test]` in source modules).
- Files: `src/`
- Risk: Internal helper regressions may only surface through broader integration suites.
- Priority: Medium

**No fuzzing/perf stress harness detected:**
- What's not tested: Adversarial raw layouts/pointers beyond deterministic fixtures and runtime perf regressions under large problem sizes.
- Files: `tests/`, `Cargo.toml`
- Risk: Memory-safety-adjacent edge cases and performance degradations can pass normal fixture-based suites.
- Priority: High

**Fixture matrix emphasizes stable envelopes and small canonical layouts:**
- What's not tested: Large-scale basis/shell cardinalities and broad random layout distributions.
- Files: `tests/common/phase2_fixtures.rs`, `tests/phase2_*`, `tests/phase3_*`
- Risk: Scaling and corner-case behavior can regress without immediate signal.
- Priority: Medium

---

*Concerns audit: 2026-03-21*
