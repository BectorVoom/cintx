# Pitfalls Research

**Domain:** Rust crate test-governance policies, CI gates, and verification reporting
**Researched:** 2026-03-21
**Confidence:** MEDIUM

## Critical Pitfalls

### Pitfall 1: Misclassifying the Crate and Skipping Required Tools

**What goes wrong:**
The governance plan misses conditional tools (e.g., `loom`, `miri`, `cargo-fuzz`, `trybuild`) because the crate’s traits (unsafe, concurrency, parser exposure, compile-time contracts) were not captured, leading to unverified risk classes that appear “covered.” 

**Why it happens:**
Classification is treated as a quick checklist rather than a requirements analysis mapped to concrete tool applicability.

**How to avoid:**
Make classification a required phase artifact. For each risk class, explicitly state whether it applies and why; link each “applies” decision to a required tool and CI gate.

**Warning signs:**
“No unsafe/concurrency/parsing” claims without evidence; missing references to public API/feature-flag surface; no explicit mapping from traits to tools.

**Phase to address:**
Phase 1 — Crate classification and mandatory baseline definition.

---

### Pitfall 2: Collapsing PR/Nightly/Release Gates into a Single Gate

**What goes wrong:**
Expensive or long-running verification (mutation testing, fuzzing, Miri, Loom) is either skipped entirely or forced into PR CI, causing timeouts and eventual removal of gates.

**Why it happens:**
Governance policy does not distinguish cost profiles and assurance goals for PR vs. nightly vs. release.

**How to avoid:**
Define explicit gate tiers: PR for fast, deterministic checks; nightly for expensive, broader checks; release for full verification including longer fuzzing and mutation runs. Use separate workflows.

**Warning signs:**
PR CI routinely times out; teams disable gates or reduce test coverage without documented risk tradeoffs.

**Phase to address:**
Phase 2 — CI gate design and workflow separation.

---

### Pitfall 3: Treating Coverage or Passing Tests as Proof of Correctness

**What goes wrong:**
Reports imply completeness because “tests pass” or coverage is high, even when conditional tools were not run or coverage is known to be incomplete/unstable.

**Why it happens:**
Coverage is easier to communicate than verification scope; teams ignore tool-specific limitations (e.g., unstable branch/doctest coverage in `cargo-llvm-cov`). citeturn2view3

**How to avoid:**
Require explicit verified vs. unverified scope in reporting. Document tool limitations in the report and avoid blanket claims.

**Warning signs:**
“Fully tested” or “complete” language; coverage reports used as the only acceptance signal; doctests or branch coverage assumed stable without validation. citeturn2view3

**Phase to address:**
Phase 3 — Reporting templates and assurance language rules.

---

### Pitfall 4: Mutation Testing Results Skewed by Non-Hermetic Tests

**What goes wrong:**
`cargo-mutants` draws the wrong conclusions when tests are flaky, non-deterministic, or depend on external state, and it cannot see tests that are not run by `cargo test`. This makes mutation results look better (or worse) than reality. citeturn2view0

**Why it happens:**
Teams treat mutation testing as “set and forget” and ignore the requirement that the test suite be hermetic and in-tree.

**How to avoid:**
Ensure tests are hermetic and deterministic; document any reliance on external state; ensure critical tests are actually run by `cargo test` or explicitly mark them as out-of-scope in the report. citeturn2view0

**Warning signs:**
Frequent flaky failures; mutation results change across runs without code changes; key tests live outside `cargo test`. citeturn2view0

**Phase to address:**
Phase 2 — Tool configuration and CI gate policy.

---

### Pitfall 5: Fuzzing in CI with No Artifact Capture or Bounded Runtime

**What goes wrong:**
Fuzzing runs are either too short to find issues or too long and get killed by CI; crash artifacts are lost, making failures non-actionable. citeturn2view2

**Why it happens:**
CI workflows omit artifact upload or do not bound fuzzing time.

**How to avoid:**
Define fuzzing as a bounded smoke test in CI with explicit time limits and artifact upload on failure. citeturn2view2

**Warning signs:**
Fuzz targets are not built or run in CI; fuzz failures without reproducible inputs; CI jobs killed for time. citeturn2view2

**Phase to address:**
Phase 2 — CI gate design for fuzzing.

---

### Pitfall 6: Tool Version Drift Changes Results Without Governance Review

**What goes wrong:**
Mutation testing outcomes change between versions (new mutation patterns, changed heuristics), and coverage or verification results shift without any policy update.

**Why it happens:**
Tool versions are not pinned, and changes are not tracked in governance documentation. `cargo-mutants` explicitly notes behavior can change between versions, which affects results. citeturn2view1

**How to avoid:**
Pin tool versions in CI; require a governance review when tool versions change; document expected deltas in reports. citeturn2view1

**Warning signs:**
Mutation counts change across CI runs without code changes; “new” mutants appear after tool updates. citeturn2view1

**Phase to address:**
Phase 2 — Tooling and CI policy hardening.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Skip mutation testing to “speed up CI” | Faster PRs | No defense against fake implementations | Only in early prototype; must schedule nightly mutation gate |
| Run fuzzing locally only | Faster CI | No reproducible CI evidence; regressions slip in | Only when fuzzing infrastructure is pending |
| Single “all checks” CI job | Simpler pipeline | No separation of cost/assurance; leads to gate removal | Never |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| `cargo-llvm-cov` | Assuming branch or doctest coverage is stable | Treat as unstable; document limitations in reports citeturn2view3 |
| `cargo-mutants` | Ignoring hermetic-test requirement | Make tests deterministic and in-tree for `cargo test` citeturn2view0 |
| `cargo-fuzz` | No artifact upload or time limit | Bound run time and upload artifacts on failure citeturn2view2 |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Mutation testing on PR for large crates | PR CI timeouts, disabled gates | Run on nightly/release; tune timeouts and caching | Medium to large crates or >10 minutes baseline tests |
| Unlimited fuzzing in CI | Jobs killed or starved | Use bounded smoke tests in CI citeturn2view2 | Any shared CI runner |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| No fuzzing for hostile-input parsers | Input-triggered crashes or logic flaws | Require `cargo-fuzz` for parsers and artifact capture citeturn2view2 |
| Trusting coverage as a security signal | Gaps in edge-case coverage | Explicitly report unverified scope and tool limitations citeturn2view3 |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Reports omit unverified scope | False sense of assurance | Mandate “Verified in scope” vs “Not yet verified” sections |
| Gate language is ambiguous | Teams interpret checks differently | Use explicit PR/nightly/release gate definitions |

## "Looks Done But Isn't" Checklist

- [ ] **Classification:** Missing evidence for unsafe/concurrency/parser exposure.
- [ ] **Coverage:** Report lacks tool limitations for `cargo-llvm-cov`. citeturn2view3
- [ ] **Mutation Testing:** Non-hermetic or non-deterministic tests. citeturn2view0
- [ ] **Fuzzing:** No bounded CI run or artifact upload. citeturn2view2

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Misclassified crate | MEDIUM | Re-run classification; add missing tools; update gates and report |
| Collapsed gates | MEDIUM | Split workflows; move heavy checks to nightly/release |
| Misleading coverage claims | LOW | Update report language; add tool limitation notes citeturn2view3 |
| Mutation testing skew | MEDIUM | Make tests hermetic; rerun and annotate results citeturn2view0 |
| Fuzzing without artifacts | LOW | Add artifact upload and rerun with bounded time citeturn2view2 |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Misclassifying the crate | Phase 1 | Classification matrix and tool applicability signed off |
| Collapsed gates | Phase 2 | Separate PR/nightly/release workflows exist and pass |
| Coverage/passing tests as proof | Phase 3 | Report template enforces verified vs unverified scope |
| Mutation testing skewed by non-hermetic tests | Phase 2 | Mutation runs are stable and deterministic across runs |
| Fuzzing without artifacts | Phase 2 | CI uploads fuzz artifacts on failure |
| Tool version drift | Phase 2 | Tool versions pinned and changes documented citeturn2view1 |

## Sources

- cargo-llvm-cov: Known limitations (branch and doctest coverage instability). citeturn2view3
- cargo-mutants: Limitations and hermetic test requirement. citeturn2view0
- cargo-mutants: Stability and version-to-version changes. citeturn2view1
- Rust Fuzz Book: CI smoke tests and time-bounded fuzzing. citeturn2view2

---
*Pitfalls research for: Rust crate test-governance policies, CI gates, and verification reporting*
*Researched: 2026-03-21*
