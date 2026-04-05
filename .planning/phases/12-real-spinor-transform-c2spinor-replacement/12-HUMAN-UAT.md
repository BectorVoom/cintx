---
status: passed
phase: 12-real-spinor-transform-c2spinor-replacement
source: [12-VERIFICATION.md]
started: 2026-04-05T00:00:00Z
updated: 2026-04-05T00:00:00Z
---

## Current Test

[complete]

## Tests

### 1. Confirm all spinor oracle parity gates pass with vendor libcint
expected: Run `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --package cintx-oracle --features cpu --test oracle_gate_closure` — oracle_gate_1e_spinor, oracle_gate_2e_spinor, oracle_gate_2c2e_spinor, oracle_gate_3c2e_spinor all pass with 0 mismatches at atol=1e-12. oracle_gate_3c1e_spinor correctly ignored (upstream gap).
result: pass

### 2. Confirm vendor FFI multi-center spinor nonzero sanity checks pass
expected: Run `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --package cintx-oracle --features cpu --test oracle_gate_closure vendor_ffi` — vendor_ffi_2e_spinor_nonzero, vendor_ffi_2c2e_spinor_nonzero, vendor_ffi_3c2e_spinor_nonzero all pass.
result: pass

## Summary

total: 2
passed: 2
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps
