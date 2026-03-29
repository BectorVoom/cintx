# Phase 4: Verification & Release Automation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md - this log preserves the alternatives considered.

**Date:** 2026-03-28
**Phase:** 04-verification-release-automation
**Areas discussed:** Oracle Comparison Policy, CI Gate and Matrix Policy, Benchmark and Diagnostics Policy

---

## Oracle Comparison Policy

### Q1 - Merge-blocking oracle profile scope

| Option | Description | Selected |
|--------|-------------|----------|
| Stable+optional gated | Require stable base plus optional profiles when their features are enabled in matrix jobs. | ✓ |
| Base only gated | Block merges on base profile only and run optional profiles outside required CI. | |
| Full incl unstable | Block merges on base, optional, and unstable-source profiles together. | |

**User's choice:** Stable+optional gated
**Notes:** Optional scope should not be downgraded from merge-blocking when enabled.

### Q2 - Tolerance governance

| Option | Description | Selected |
|--------|-------------|----------|
| Versioned per-family table | Keep explicit per-family atol/rtol in code and require deliberate review when changed. | ✓ |
| Single global tolerance | Use one tolerance for all families to simplify policy at the cost of precision fit. | |
| Auto-tune tolerances | Adjust tolerances from observed runs, accepting less deterministic gate behavior. | |

**User's choice:** Versioned per-family table
**Notes:** Deterministic tolerance governance is preferred.

### Q3 - Optional and unstable oracle enforcement

| Option | Description | Selected |
|--------|-------------|----------|
| Optional required, unstable extended | Require optional-profile oracle checks; keep unstable-source checks in extended/nightly CI only. | ✓ |
| Stable only required | Require only stable oracle checks and treat optional/unstable as non-blocking. | |
| Everything required | Require stable, optional, and unstable-source oracle checks in all merge-blocking runs. | |

**User's choice:** Optional required, unstable extended
**Notes:** Stable + optional are hard requirements; unstable remains extended.

### Q4 - Mismatch reporting mode

| Option | Description | Selected |
|--------|-------------|----------|
| Complete diff report | Evaluate full fixture set, emit full mismatch report, then fail once all comparisons are recorded. | ✓ |
| Fail fast first mismatch | Stop at first mismatch for faster feedback with less diagnostic detail. | |
| Sample then escalate | Run a reduced sample on PRs and reserve full comparison reports for scheduled runs. | |

**User's choice:** Complete diff report
**Notes:** Full failure context is preferred over fail-fast behavior.

---

## CI Gate and Matrix Policy

### Q1 - Required PR gates

| Option | Description | Selected |
|--------|-------------|----------|
| Core gates required | Require manifest drift, oracle parity, helper/legacy parity, and OOM-contract checks; keep heavy perf jobs separate. | ✓ |
| Light checks only | Require compile/unit checks only and run verification suites outside merge-blocking CI. | |
| All gates always | Require all verification and performance suites in every merge-blocking PR. | |

**User's choice:** Core gates required
**Notes:** Verification gates stay strict, but heavy perf jobs are not per-PR blockers.

### Q2 - Required feature-matrix breadth

| Option | Description | Selected |
|--------|-------------|----------|
| All approved profiles | Require base, with-f12, with-4c1e, and combined profile coverage in required matrix gates. | ✓ |
| Base+f12 required | Require base and with-f12 only, leaving with-4c1e variants non-blocking. | |
| Base only required | Keep required coverage limited to base profile and run others outside blocking CI. | |

**User's choice:** All approved profiles
**Notes:** Full approved matrix should be represented in required coverage.

### Q3 - GPU job enforcement

| Option | Description | Selected |
|--------|-------------|----------|
| Nightly required, PR advisory | Keep PR CI deterministic while requiring GPU consistency in scheduled or merge-queue gates. | ✓ |
| Required every PR | Block every PR on GPU jobs, accepting longer and potentially flakier CI cycles. | |
| Advisory always | Never block merges on GPU jobs and treat them as informational only. | |

**User's choice:** Nightly required, PR advisory
**Notes:** GPU checks remain important, but not as per-PR hard blockers.

### Q4 - Gate override policy

| Option | Description | Selected |
|--------|-------------|----------|
| No override merge | Failed required gates block merge until fixed, except normal reruns for transient infra issues. | ✓ |
| Temporary quarantine allowed | Allow temporary bypasses with explicit expiry and follow-up issue tracking. | |
| Advisory after retries | Downgrade failing gates to advisory after retry budget is exhausted. | |

**User's choice:** No override merge
**Notes:** Required gates must remain authoritative.

---

## Benchmark and Diagnostics Policy

### Q1 - Benchmark run cadence

| Option | Description | Selected |
|--------|-------------|----------|
| Nightly+release runs | Run benchmark suites on schedule/release workflows, not on every merge-blocking PR. | ✓ |
| Every PR run | Run benchmark suites on every PR and use them as merge blockers. | |
| Manual only | Run benchmarks only when maintainers trigger them manually. | |

**User's choice:** Nightly+release runs
**Notes:** Benchmark coverage should be continuous but not per-PR blocking.

### Q2 - Benchmark suite scope

| Option | Description | Selected |
|--------|-------------|----------|
| Micro+macro+crossover | Track family microbench, molecule macrobench, and CPU-GPU crossover as the baseline suite. | ✓ |
| Micro+macro only | Track routine throughput suites but skip explicit crossover tracking. | |
| Single smoke suite | Track only one lightweight benchmark suite for minimal overhead. | |

**User's choice:** Micro+macro+crossover
**Notes:** Full trend visibility requires all three suite types.

### Q3 - Regression enforcement

| Option | Description | Selected |
|--------|-------------|----------|
| Threshold-gated regressions | Fail benchmark gate only when regressions exceed defined thresholds and include report artifacts. | ✓ |
| Report-only mode | Never fail on regressions; provide trend reports for manual review only. | |
| Any slowdown fails | Fail on any measured slowdown regardless of magnitude. | |

**User's choice:** Threshold-gated regressions
**Notes:** Regression policy should be sensitive but not noisy.

### Q4 - Diagnostics artifact depth

| Option | Description | Selected |
|--------|-------------|----------|
| Structured trace+metrics | Persist structured planner/chunk/fallback/transfer/OOM diagnostics with artifacts, honoring required /mnt/data outputs. | ✓ |
| Summary counters only | Persist only high-level counters and omit detailed trace context. | |
| On-demand debug only | Capture deep diagnostics only when maintainers enable debug workflows manually. | |

**User's choice:** Structured trace+metrics
**Notes:** Release automation should preserve detailed diagnostics by default.

---

## the agent's Discretion

- Exact numeric regression threshold values and stabilization window before threshold gates become hard-fail.
- CI job splitting and retry-budget implementation details.
- Exact artifact schema field names for benchmark and diagnostics reports.

## Deferred Ideas

None - discussion stayed within Phase 4 scope.
