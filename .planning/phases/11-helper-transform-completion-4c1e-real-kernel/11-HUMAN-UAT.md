---
status: resolved
phase: 11-helper-transform-completion-4c1e-real-kernel
source: [11-VERIFICATION.md]
started: 2026-04-04T12:00:00Z
updated: 2026-04-04T12:00:00Z
---

## Current Test

[complete]

## Tests

### 1. Oracle parity gate for with-4c1e profile (vendor comparison)
expected: oracle_gate_4c1e_parity passes with 0 mismatches at atol=1e-12 with vendored libcint
result: PASSED — after fix (be3498a) correcting common_factor formula and adding cross-pair exponential

Command: `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features "cpu,with-4c1e" --test oracle_gate_closure -- oracle_gate_4c1e_parity`

Root cause: Wrong common_factor (used 2e formula PI^3*2/SQRTPI instead of 4c1e formula SQRTPI*PI) + missing cross-pair exponential exp(-a0*|rij-rkl|^2). Debug session: .planning/debug/resolved/4c1e-oracle-parity-mismatch.md

## Summary

total: 1
passed: 1
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps
