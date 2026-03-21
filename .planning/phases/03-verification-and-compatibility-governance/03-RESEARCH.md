# Phase 3: Verification and Compatibility Governance - Research

**Researched:** 2026-03-14
**Domain:** Compatibility governance, manifest auditing, regression-gate CI for libcint-rs
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Phase 3 covers helper/transform parity for AO counts, shell offsets, normalization, and cart/sph/spinor transform behavior.
- Coverage claims must be manifest-backed and machine-auditable, with compiled-manifest lock as source of truth.
- Support profile set for governance is fixed to `{base, with-f12, with-4c1e, with-f12+with-4c1e}`; GTG is out of scope.
- CI gates are blocking for lock drift, oracle regressions, and helper parity failures.
- RAW-04 optimizer on/off equivalence plus spinor/layout and OOM/error semantics are explicit regression gates.

### Claude's Discretion
- Exact module layout for manifest audit tooling and generated artifacts
- CI job naming, matrix decomposition, and caching strategy
- Fast PR vs exhaustive release/nightly gate split
- Tolerance-table implementation details within requirement bounds

### Deferred Ideas (OUT OF SCOPE)
- GPU dispatch/fallback observability gates (Phase 4)
- C ABI shim verification/migration coverage (Phase 4)
- Optional-family expansion beyond current support matrix (Phase 4+)
- GTG enablement (roadmap-only)
</user_constraints>

<research_summary>
## Summary

Phase 3 should be implemented as governance infrastructure that turns compatibility claims into reproducible, blocking evidence. The existing Phase 2 matrix, fixtures, typed error contracts, and oracle helpers are already strong foundations; the main gap is formalizing manifest lock generation/audit and wiring release-grade CI gates around these checks.

The standard approach in this domain is: deterministic symbol inventory + requirement-mapped regression suites + profile matrix CI. For this codebase, that means extending existing `tests/common` fixtures into phase-3 parity suites, adding manifest lock tooling (preferably `xtask`-style Rust code with schema checks), and creating a PR/release gate split where PR remains fast but still blocks critical drift.

**Primary recommendation:** implement Phase 3 as three slices: (1) helper/transform parity suites, (2) compiled-manifest lock + audit tooling, (3) blocking CI gate matrix tied directly to `COMP-02/03/04`, `RAW-04`, and `VERI-01/02/03`.
</research_summary>

<standard_stack>
## Standard Stack

### Core
| Library/Tool | Version | Purpose | Why Standard |
|---|---|---|---|
| `serde`, `serde_json` | stable 1.x | Typed lockfile schema + deterministic JSON read/write | Safer than ad-hoc JSON parsing for governance-critical artifacts |
| Cargo integration tests (`cargo test`) | rust stable | Requirement-mapped parity/regression suites | Native test runner already used in this repo |
| `tracing` + diagnostics payloads | existing | Failure evidence and backend/profile context in regressions | Already established in phase-1/2 contracts |
| `thiserror` typed failures | existing | Explicit unsupported/drift/failure categories in tooling/runtime checks | Maintains non-ambiguous public/internal error semantics |

### Supporting
| Tool | Purpose | When to Use |
|---|---|---|
| `cargo-nextest` | Faster, deterministic CI test orchestration | PR/release matrix once phase-3 suites grow |
| `cargo-llvm-cov` | Optional verification visibility for gate critical paths | Release hardening and regression trend checks |
| `nm` + profile builds | Compiled symbol extraction for lock generation | Manifest governance and lock-drift detection |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|---|---|---|
| Rust manifest-audit tool | Shell scripts with `grep/sed/jq` only | Faster to start but brittle for canonicalization and schema versioning |
| Requirement-mapped tests | Broad smoke-only suites | Less maintenance but weak traceability for release claims |
| Blocking gates | Advisory-only CI comments | Faster merges but undermines compatibility trust objective |
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Pattern 1: Manifest Lock As Contract Artifact
**What:** Generate and validate a compiled-manifest lock from support-profile builds; fail on unapproved drift.  
**When to use:** Any PR touching symbol surface, profile flags, or upstream vendored sources.

### Pattern 2: Requirement-Traceable Test Matrix
**What:** Every phase-3 requirement maps to one or more explicit tests and CI jobs.  
**When to use:** Always; this is the core of auditable compatibility claims.

### Pattern 3: Two-Tier Compatibility Gates
**What:** Fast PR gate (targeted drift + key regressions) and exhaustive release gate (full profile matrix + oracle/helper suites).  
**When to use:** PR and release workflows respectively.

### Pattern 4: Reuse Existing Matrix Fixtures
**What:** Extend `tests/common/phase2_fixtures.rs` and `tests/common/oracle_runner.rs` rather than creating parallel fixtures.  
**When to use:** All helper/optimizer/layout/OOM phase-3 tests to avoid drift between phase-2 and phase-3 evidence.
</architecture_patterns>

## Validation Architecture

### Test Infrastructure
- Framework: Rust integration tests via `cargo test`
- Quick run command: `cargo test phase3_ -- --nocapture`
- Full suite command: `cargo test --all-targets`
- Optional release-depth command: `cargo test --all-targets && cargo test phase2_ phase3_ -- --nocapture`

### Requirement-to-Validation Mapping
| Requirement | Validation target |
|---|---|
| `COMP-02` | Helper/transform parity tests for AO counts, offsets, normalization, and representation transforms against oracle/reference behavior |
| `COMP-03` | Manifest schema + lock coverage tests: each symbol has support profile + stability classification |
| `COMP-04` | Oracle-regression matrix tests tied to supported profiles and support-matrix rows |
| `RAW-04` | Optimizer on/off equivalence tests across supported operator/family/representation rows |
| `VERI-01` | CI lock-drift gate tests and workflow assertions for unapproved manifest changes |
| `VERI-02` | CI oracle regression gate across required profile matrix |
| `VERI-03` | Spinor/layout + OOM/error-path semantics tests with diagnostics assertions |

### Required New Test Families
- `tests/phase3_helper_transform_parity.rs`
- `tests/phase3_manifest_governance.rs`
- `tests/phase3_optimizer_equivalence.rs`
- `tests/phase3_regression_gates.rs`

### CI Gate Design
- PR gate must block on: manifest lock drift, helper parity failures, optimizer equivalence failures, core oracle regressions.
- Release/nightly gate must run full profile matrix and publish artifacts proving lock + regression pass.
- All failures remain blocking (no soft-pass mode) for compatibility-governance requirements.

<dont_hand_roll>
## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---|---|---|---|
| Manifest canonicalization | String-split symbol munging in shell | Typed Rust parser/canonicalizer with tests | Symbol forms/profiles are too nuanced for fragile text pipelines |
| Requirement traceability | Human-maintained checklist only | Test files with requirement IDs and CI mapping | Prevents drift between claims and executable evidence |
| Gate status reasoning | Manual release sign-off docs only | CI-enforced pass/fail jobs + lock artifacts | Needed for enforceable compatibility claims |
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Coverage Claims Without Profile Context
**What goes wrong:** Symbol appears in manifest but profile/stability is unclear.  
**How to avoid:** Require profile + stability metadata for every manifest row, validate in tests.

### Pitfall 2: Gate Drift Between Docs and Tests
**What goes wrong:** `docs/phase2-support-matrix.md` and tests diverge silently.  
**How to avoid:** Keep one canonical matrix fixture path and derive assertions from it.

### Pitfall 3: Optimizer Equivalence Checked Too Narrowly
**What goes wrong:** RAW-04 passes on a subset but regresses elsewhere.  
**How to avoid:** Run equivalence across all supported rows and both safe/raw entry paths where relevant.

### Pitfall 4: Non-Deterministic OOM/Error Semantics
**What goes wrong:** Failure diagnostics differ between query/evaluate or safe/raw paths.  
**How to avoid:** Assert diagnostics fields and typed variants in phase-3 regression tests.

### Pitfall 5: PR Gates Too Heavy
**What goes wrong:** Excessive runtime causes bypass pressure.  
**How to avoid:** Keep PR gate targeted and make exhaustive profile matrix run in release/nightly while staying blocking at release.
</common_pitfalls>

<open_questions>
## Open Questions

1. **Artifact location for manifest lock and generated snapshots**
   - What we know: Design doc references a generated lock artifact and profile union.
   - Gap: Final repository path/module placement is not yet implemented.
   - Planning recommendation: Pick one canonical path early and make both tooling + CI assert it.

2. **Exact profile matrix split between PR and release workflows**
   - What we know: Full matrix must be enforced before compatibility claims.
   - Gap: Which subset remains mandatory on every PR vs release/nightly.
   - Planning recommendation: Keep lock drift + core regressions in PR; full profile/oracle matrix in release/nightly.

3. **Tolerance policy for helper/transform parity edge cases**
   - What we know: Existing oracle checks already enforce strict tolerances.
   - Gap: Whether helper transforms need representation-specific tolerances.
   - Planning recommendation: Start with existing strict policy and only relax with explicit evidence.
</open_questions>

<sources>
## Sources

### Primary (HIGH confidence)
- `.planning/phases/03-verification-and-compatibility-governance/03-CONTEXT.md`
- `.planning/REQUIREMENTS.md`
- `.planning/ROADMAP.md`
- `.planning/STATE.md`
- `docs/libcint_detailed_design_resolved_en.md`
- `docs/phase2-support-matrix.md`
- `tests/common/phase2_fixtures.rs`
- `tests/common/oracle_runner.rs`
- `src/runtime/backend/cpu/router.rs`
- `src/runtime/raw/{query.rs,evaluate.rs,validator.rs,views.rs}`

### Secondary (MEDIUM confidence)
- `.planning/phases/02-cpu-compatibility-execution/02-RESEARCH.md` (prior phase pattern baseline)
</sources>

<metadata>
## Metadata

**Phase requirement IDs covered:** `COMP-02, COMP-03, COMP-04, RAW-04, VERI-01, VERI-02, VERI-03`  
**Research date:** 2026-03-14  
**Ready for planning:** yes
</metadata>

---

*Phase: 03-verification-and-compatibility-governance*
*Research completed: 2026-03-14*
*Ready for planning: yes*
