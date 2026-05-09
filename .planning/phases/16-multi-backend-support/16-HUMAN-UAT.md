---
status: partial
phase: 16-multi-backend-support
source: [16-VERIFICATION.md]
started: 2026-05-09T00:00:00Z
updated: 2026-05-09T00:00:00Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. Register feature_matrix_gate in branch protection
expected: After plan 16-03 lands on `main`, the three GitHub Actions matrix entries `feature_matrix_gate (cpu-only)`, `feature_matrix_gate (cpu+wgpu)`, and `feature_matrix_gate (all-features)` are added to the `main` branch's "Require status checks to pass before merging" list in repo Settings → Branches → Edit branch protection rule. After saving, opening any PR should show all three checks as Required and merge should be blocked until they go green. (Plan 16-03 is explicitly `autonomous: false` for this reason; cannot be automated from a PR.)
result: [pending]

### 2. CUDA runtime parity verification (out-of-scope, deferred)
expected: Per BACK-06 and `.planning/notes/cuda-metal-verification-gap.md`, CUDA is verified compile-only in this phase via the all-features CI cell. Runtime parity (oracle agreement at atol=1e-12 / rtol=1e-10) requires a CUDA host/toolchain not available on this dev host. Tracked as a phase-17+ follow-up (see `.planning/seeds/gpu-ci-runners.md`). User confirms this risk-accept is acceptable for v1.2.
result: [pending]

## Summary

total: 2
passed: 0
issues: 0
pending: 2
skipped: 0
blocked: 0

## Gaps
