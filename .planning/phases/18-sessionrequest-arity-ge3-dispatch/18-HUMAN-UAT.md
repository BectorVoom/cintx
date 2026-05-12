---
status: partial
phase: 18-sessionrequest-arity-ge3-dispatch
source: [18-VERIFICATION.md]
started: 2026-05-12T12:35:00Z
updated: 2026-05-12T12:35:00Z
---

## Current Test

[awaiting human testing — vendored libcint required]

## Tests

### 1. Run 8 arity-3 + 4 arity-4 oracle parity tests on a host with vendored libcint
expected: All 12 tests pass at `atol=1e-12, rtol=0.0` against libcint 6.1.3 with 0 mismatches. Expect `test result: ok. 8 passed` for arity-3 and `test result: ok. 4 passed` for arity-4 under `--features cpu,with-4c1e`. The `int3c2e_ip1_{cart,sph}` tests in particular reference plain `vendor_int3c2e_*` and should produce 0 mismatches (per RESEARCH.md Item 5 / A6 — kernel-misnomer disposition).

Commands to run on a vendor-built host:
```
CINTX_ORACLE_BUILD_VENDOR=1 CINTX_BACKEND=cpu \
  cargo test -p cintx-oracle --features cpu --locked \
  --test safe_api_arity3_parity -- --test-threads=1

CINTX_ORACLE_BUILD_VENDOR=1 CINTX_BACKEND=cpu \
  cargo test -p cintx-oracle --features cpu,with-4c1e --locked \
  --test safe_api_arity4_parity -- --test-threads=1
```
result: [pending]

### 2. Verify F-order AO axis layout via the implicit oracle parity sweep
expected: ARITY-03 (SC#3) is implicitly verified by Test #1 success. Byte-identity vs vendored libcint with NO transpose means cintx writes F-order by construction. If layout drifted, the first parity element would mismatch and `total_mismatches > 0` would fire.
result: [pending]

## Summary

total: 2
passed: 0
issues: 0
pending: 2
skipped: 0
blocked: 0

## Gaps
