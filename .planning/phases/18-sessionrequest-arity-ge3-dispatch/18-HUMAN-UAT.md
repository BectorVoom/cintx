---
status: partial
phase: 18-sessionrequest-arity-ge3-dispatch
source: [18-VERIFICATION.md]
started: 2026-05-12T12:35:00Z
updated: 2026-05-12T13:15:00Z
---

## Current Test

[Gap 1 resolved; Gap 2 deferred to /gsd:debug for the int3c1e_p2_* kernel divergence]

## Tests

### 1. Run 8 arity-3 + 4 arity-4 oracle parity tests on a host with vendored libcint
expected: All 12 tests pass at `atol=1e-12, rtol=0.0` against libcint 6.1.3 with 0 mismatches.

result: **10 of 12 pass. 2 fail (int3c1e_p2_{cart,sph} kernel divergence — deferred to /gsd:debug).**

Pass (10):
- test_int3c1e_cart_safe_api_parity
- test_int3c1e_sph_safe_api_parity
- test_int3c2e_ip1_cart_safe_api_parity
- test_int3c2e_ip1_sph_safe_api_parity
- test_int3c2e_cart_safe_api_parity
- test_int3c2e_sph_safe_api_parity
- test_int2e_cart_safe_api_parity
- test_int2e_sph_safe_api_parity
- test_int4c1e_cart_safe_api_parity (Gap 1 fixed in commit 5bd5ab0)
- test_int4c1e_sph_safe_api_parity (Gap 1 fixed in commit 5bd5ab0)

Fail (2, deferred):
- test_int3c1e_p2_cart_safe_api_parity — 182 elements exceed atol=1e-12
- test_int3c1e_p2_sph_safe_api_parity — 182 elements exceed atol=1e-12

### 2. Verify F-order AO axis layout via the implicit oracle parity sweep
expected: ARITY-03 (SC#3) implicitly verified by Test #1.
result: **Pass for the 10 succeeding tests** (byte-identity vs vendor with no transpose → F-order verified by construction). Cannot conclude for the 2 deferred int3c1e_p2_* tests until kernel divergence is investigated.

## Summary

total: 2
passed: 1
issues: 1
pending: 0
skipped: 0
blocked: 0

## Gaps

### Gap 1: int4c1e_{cart,sph} arity-4 test buffer-size formula was wrong
status: resolved
plan: 18-04
fix_commit: 5bd5ab0
detail: Test allocated `n_elem = ni * nj` for arity-4; corrected to `ni * nj * nk * nl`. Both int4c1e_* tests now pass at atol=1e-12.

### Gap 2: int3c1e_p2_{cart,sph} kernel output diverges from vendored libcint
status: deferred
plan: pre-existing (Phase 11 helper/transform completion); out of Phase 18 scope
file: crates/cintx-* (native kernel; specific location TBD by debug session)
detail: |
  Native cintx kernel for int3c1e_p2 disagrees with vendored libcint by ~1e-2 to 1e-4 over 182
  elements in a 125-triple H2O/STO-3G sweep. This is a real numeric disagreement (not noise).
  Phase 18 is the first oracle parity test against this operator — the divergence pre-dates
  Phase 18.
followup: |
  Run `/gsd:debug` with a session focused on:
    1. Localizing the int3c1e_p2 kernel implementation (resolver.rs:316 routes via AllCint;
       check whether cintx has a native CubeCL kernel or proxies through legacy).
    2. Comparing cintx kernel output vs vendor on a single (s,p,p) triple to characterize
       the disagreement shape (sign? scaling? momentum? normalization?).
    3. Deciding whether to:
       - Patch the kernel to match libcint, OR
       - Apply kernel-misnomer disposition (like int3c2e_ip1 → plain int3c2e per RESEARCH A6).
followup_command: /gsd:debug "int3c1e_p2_{cart,sph} kernel disagrees with vendored libcint by 1e-2 to 1e-4 in 182/N elements"
