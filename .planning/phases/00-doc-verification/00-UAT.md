---
status: testing
phase: 00-doc-verification
source:
  - docs/cintx_detailed_test_design_en.md
started: "2026-03-21T00:48:09Z"
updated: "2026-03-21T00:48:09Z"
---

## Current Test

number: 1
name: Objective and claim boundaries are explicit
expected: |
  The document clearly distinguishes compatibility goals from implementation similarity,
  and it explicitly states what is in scope vs out of scope for release claims.
awaiting: user response

## Tests

### 1. Objective and claim boundaries are explicit
expected: The document defines parity objectives and clear out-of-scope boundaries.
result: pending

### 2. Normative specification sources are declared
expected: The document lists concrete normative inputs (design docs, upstream sources, repository artifacts) used to judge conformance.
result: pending

### 3. Tool-selection rationale is complete
expected: Baseline and conditional Rust verification tools are listed with clear applicability and non-applicability rationale (including loom conditions).
result: pending

### 4. CI gate policy is tiered and auditable
expected: PR, nightly, and release gates are separately defined with blocking conditions and waiver controls.
result: pending

### 5. Evidence model is layered
expected: The architecture includes layered evidence categories (inventory, routing, contracts, oracle parity, helper parity, failure behavior, backend consistency).
result: pending

### 6. Anti-shortcut rules are explicit
expected: The document forbids shortcuts like passing-tests-only claims and coverage-only conformance claims.
result: pending

### 7. Reporting obligations separate verified vs unverified scope
expected: The report structure requires explicit sections for verified scope, unverified scope, waivers, and residual risk.
result: pending

### 8. Operational rollout plan is actionable
expected: The document includes implementation sequencing, ownership-ready tasks, and CI/job mapping that can be executed incrementally.
result: pending

## Summary

total: 8
passed: 0
issues: 0
pending: 8
skipped: 0

## Gaps

none yet
