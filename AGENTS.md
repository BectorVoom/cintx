<!-- GSD:project-start source:PROJECT.md -->
## Project

**cintx**

cintx is a public Rust library that redesigns and reimplements libcint with result compatibility as the primary goal. It provides a Rust-native safe API, a raw compatibility API for `atm`/`bas`/`env` style callers, and an optional C ABI shim for migration and interoperability. The target users are Rust developers and systems that need libcint-compatible integral evaluation with stronger type safety, clear failure modes, and high-confidence verification.

**Core Value:** Deliver libcint-compatible results through a Rust-native API surface that stays type-safe, verifiable, and safe under memory pressure.

### Constraints

- **Compatibility**: Target upstream libcint 6.1.3 result compatibility - the project must match upstream outputs closely enough to satisfy oracle comparison gates.
- **Architecture**: CubeCL is the primary compute backend - host CPU work stays limited to planning, validation, marshaling, and test/oracle glue.
- **API Surface**: Safe Rust API first, raw compatibility API second, optional C ABI shim third - this ordering drives module boundaries and migration strategy.
- **Error Handling**: Public library errors use `thiserror` v2, while CLI, xtask, benchmarks, and oracle harness code use `anyhow`.
- **Verification**: Full API coverage claims must be backed by the compiled manifest lock, feature-matrix CI, and helper/transform parity checks.
- **Artifacts**: Deliverables written to `/tmp/cintx_artifacts` remain a mandatory part of the design and verification workflow.
<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->
## Technology Stack

## Recommended Stack
### Core Platform
| Technology | Version guidance | Purpose | Why recommended |
|------------|------------------|---------|-----------------|
| Rust toolchain | Pin `1.94.0` in `rust-toolchain.toml` | Reproducible compiler behavior across local dev and CI | Rust 1.94.0 is the current stable release as of 2026-03-05, and pinning an exact toolchain keeps oracle and manifest results reproducible. |
| Cargo lockfile | Commit `Cargo.lock`; run CI with `cargo --locked` | Deterministic dependency graph | Oracle comparisons and manifest audits are only credible if every runner uses the same resolved graph. |
| Cargo resolver | Use edition-2024 default `resolver = "3"`; if the root becomes a virtual workspace, declare it explicitly under `[workspace]` | Predictable feature resolution in a multi-crate workspace | Resolver 3 is the 2024-edition default and is the right baseline for the workspace described in the design doc. |
| Multi-crate workspace | Keep the crate split from the design (`core`, `ops`, `runtime`, `cubecl`, `compat`, `capi`, `oracle`, `xtask`) | Isolate domain types, execution, compat, verification, and tooling | The project has hard boundaries between typed API, compat contracts, backend execution, and release gating; the crate layout should reflect them. |
### Core Libraries
| Library | Version guidance | Purpose | Notes |
|---------|------------------|---------|-------|
| `cubecl` | Keep the current `0.9.x` line unless a verified backend issue forces a change | Shared GPU compute backend | `docs.rs` shows `cubecl 0.9.0` as the latest published crate. Keep the public API backend-agnostic enough that a backend swap remains possible if the ecosystem shifts. |
| `thiserror` | `2.0.18` | Public typed error surface | Fits the design requirement for library-facing error enums without leaking implementation details into the API contract. |
| `anyhow` | `1.0.102` | App-boundary, xtask, benchmark, and oracle tooling errors | Matches the design choice to keep ergonomic context-rich errors out of the public library surface. |
| `tracing` | Stay on the current stable `0.1.x` line used by the workspace | Structured spans and diagnostics | Required for planner decisions, chunking, transfers, fallback reasons, and OOM visibility. |
| `bindgen` | Current workspace is `0.71.1`; latest published line is `0.72.1` | Oracle/header binding generation | Upgrade deliberately, not automatically: header-generation changes must be validated against the manifest and oracle harness. |
| `cc` | Keep current stable `1.2.x` line | Vendored upstream libcint build integration | Needed to keep the oracle harness hermetic and reproducible. |
### Supporting Libraries from the Design
| Library | Use | Why it belongs here |
|---------|-----|---------------------|
| `rayon` | Host-side staging and chunk-preparation parallelism | Good fit for CPU-side marshaling without exposing threading complexity in the public API. |
| `smallvec` | Small fixed-ish collections (`dims`, shell tuples, strides) | Cuts heap churn in hot control-plane paths. |
| `num-complex` | Safe API complex/spinor outputs | Better than raw interleaved buffers leaking into typed callers. |
| `approx`, `proptest`, `criterion` | Verification and benchmarking | Match the design's emphasis on oracle comparison, property testing, and repeatable perf baselines. |
### Development Tools
| Tool | Purpose | Notes |
|------|---------|-------|
| `cargo nextest` | Faster and more controllable CI test execution | Useful once oracle, feature-matrix, and regression suites become expensive. |
| `rustfmt` + `clippy` | Baseline style and lint enforcement | Already aligned with the current `rust-toolchain.toml` components. |
| `xtask` commands | Manifest audit, oracle refresh, docs generation, bench reporting | Keeps release gates expressed as code instead of tribal knowledge. |
## Alternatives Considered
| Recommended | Alternative | When the alternative is justified |
|-------------|-------------|----------------------------------|
| `cubecl` | Another GPU backend or a CPU compute backend | Only if CubeCL blocks correctness, platform coverage, or maintainability; do not leak CubeCL-specific types into the public API. |
| `thiserror` for library errors | `anyhow` everywhere | Only for internal binaries or scripts; not for the public library contract. |
| Exact toolchain pin | Floating `stable` | Acceptable for quick local experimentation, but not for release-gated CI or oracle baselines. |
## What Not to Use
| Avoid | Why | Use instead |
|-------|-----|-------------|
| Nightly as the project baseline | Changes compiler behavior and weakens reproducibility | Stable Rust pinned in `rust-toolchain.toml` |
| Unpinned dependency resolution in CI | Makes manifest/oracle drift hard to diagnose | `Cargo.lock` plus `cargo --locked` |
| Public APIs that expose backend-specific runtime types | Makes future backend changes expensive and risky | Keep backend details behind planner/executor traits and typed output views |
| Best-effort partial writes on allocation failure | Violates the design's OOM-safe stop contract | Fallible allocation + typed failure + no partial writes |
## Sources
### Official / primary
- Rust 1.94.0 release announcement: https://blog.rust-lang.org/2026/03/05/Rust-1.94.0/
- Cargo resolver guidance: https://doc.rust-lang.org/nightly/cargo/reference/resolver.html
- Cargo feature resolver details: https://doc.rust-lang.org/stable/cargo/reference/features.html
- Cargo nextest docs: https://nexte.st/
- CubeCL crate docs: https://docs.rs/crate/cubecl/latest
- thiserror crate docs: https://docs.rs/crate/thiserror/latest
- anyhow crate docs: https://docs.rs/crate/anyhow/latest
- bindgen crate docs: https://docs.rs/crate/bindgen/0.71.1 and https://docs.rs/crate/bindgen/latest


### Local project evidence
- `Cargo.toml`
- `Cargo.lock`
- `rust-toolchain.toml`
- `docs/design/cintx_detailed_design.md`
<!-- GSD:stack-end -->

## Conventions

- Before creating any test code, read `\home\chemtech\workspace\cintx\docs\rust_crate_test_guideline.md` and follow it when designing and implementing the tests.


<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->

## Handling `cubecl` Build Errors

If a build error occurs in any Rust project that uses the `cubecl` crate, you must consult the following manual before attempting further fixes or proposing changes:

`/home/chemtech/workspace/cintx/docs/cubecl_error_guideline.md`

### Required Procedure
1. Detect whether the failure is related to building, compiling, linking, dependency resolution, feature flags, toolchain configuration, or CI execution involving `cubecl`.
2. Read `/home/chemtech/workspace/cintx/docs/cubecl_error_guideline.md`.
3. Follow the documented troubleshooting and resolution process in that manual.
4. When reporting the issue or proposing a fix, align the explanation with the structure and terminology defined in the manual.
5. If the issue is resolved, document the root cause, resolution steps, verification results, and prevention measures according to the manual.

### Notes
- Do not skip the manual, even if the error appears familiar.
- Do not provide an ad hoc fix without first checking whether the manual already covers the issue pattern.
- If the manual does not fully cover the error, use it as the baseline format and extend the investigation accordingly.


## Mandatory Manual for `cubecl` Implementation

When implementing, modifying, or generating code that uses the Rust `cubecl` crate, the agent must first read:

`/home/chemtech/workspace/cintx/docs/manual/Cubecl`

This manual must be used as the primary reference for implementation patterns, architecture, configuration, and coding rules related to `cubecl`.

Do not write or propose `cubecl`-based code without consulting this manual first.


````md
## Crawl4AI Usage

When web content needs to be fetched, rendered, or extracted from dynamic pages, use the running Crawl4AI Docker service instead of writing ad hoc scraping code.

### Basic rules
- Prefer Crawl4AI for page crawling, markdown extraction, and multi-page content collection.
- Use it for JavaScript-rendered pages when simple HTTP fetch is insufficient.
- Save extracted results into project artifacts or intermediate files when they are used for later steps.
- If crawling fails, report the target URL, command used, and error briefly.

### Example commands

Fetch one page:
```bash
curl -X POST http://localhost:11235/crawl \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "formats": ["markdown"]
  }'
````

Crawl multiple pages:

```bash
curl -X POST http://localhost:11235/crawl \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://example.com", "https://example.com/docs"],
    "formats": ["markdown"]
  }'
```

Save result to a file:

```bash
curl -X POST http://localhost:11235/crawl \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "formats": ["markdown"]
  }' > crawl-result.json
```

## Codex MCP Server Usage Rules (Rust Crate Research and Minimal Implementation)

### Objective
If you encounter an API in a Rust crate whose implementation method is unclear, use the Codex MCP server to verify the facts, then complete the work based on up-to-date evidence rather than guesswork: from minimal implementation to validation and documentation.

### General Policy
- Do not implement unclear Rust APIs based on assumptions.
- Always ask Codex to investigate, using the `context7` CLI and the `crawl4ai` CLI to research the target crate.
- Use the latest stable version of the crate as the primary basis for investigation.
- However, if the project pins a specific version, verify the differences from the latest stable version and implement against the version actually used by the project.
- Keep changes minimal. Do not introduce unnecessary design changes, excessive abstraction, or unrelated refactoring.

### Required Work for Codex
1. Identify the target crate name, target API, current version in use, and latest stable version.
2. Use the `context7` CLI and the `crawl4ai` CLI to investigate, prioritizing:
   - Official documentation
   - docs.rs
   - The crate repository
   - Release notes / changelog
   - Example code, examples, and tests
3. Organize the possible implementation approaches and choose the simplest and most maintainable one.
4. Implement the minimal solution for the target API.
5. Check for compile errors, type errors, and deprecated API usage, and fix them if necessary.
6. Add or update the minimum required tests.
7. After implementation, always run the tests and verify that the change actually works.
8. Create a Markdown manual summarizing the investigation results and implementation details.

### Testing Requirements
- Provide at least one test that directly verifies the changed area.
- Run `cargo test`, or the smallest appropriate test command that reasonably covers the modified scope.
- Do not mark the work as complete if tests have not been executed.
- If tests fail, clearly document the failure logs, the cause, and any unresolved issues.

### Manual Requirements
The Markdown file must include at least the following:
- Target crate name
- Investigated version information (latest stable version and, if necessary, the version used by the project)
- Key points about the investigated API
- The adopted implementation strategy
- Explanation of the implementation location(s)
- A minimal usage example
- The test command(s) that were run
- Test results
- Constraints and known caveats

### Definition of Done
The work is complete only when all of the following are satisfied:
- The unclear API has been investigated with supporting evidence
- The minimal implementation has been completed
- Tests have been executed successfully
- A Markdown manual summarizing the investigation and implementation has been created

### Prohibited Actions
- Assumption-based implementation without evidence
- Reporting completion without running tests
- Reporting completion without creating the manual
- Large-scale unrelated changes
