# Deferred Items — Phase 20 Precision Generic F64/F32 Switch

## DI-01: Pre-existing f12 kernel asymmetric-quartet value-swap bug

**Discovered during:** Plan 20-11, Task 1 (f32 parity test for int2e_stg_ip1_sph)

**Severity:** Medium — produces wrong values silently for non-symmetric f12 quartets involving
p-shells in asymmetric positions; does not affect the 3 quartets the f64 oracle tests
([0,1,0,1], [3,4,3,4], [0,2,0,2]).

**Symptom:** When iterating all 625 (0..24)^4 shell quartets for int2e_stg_ip1_sph (ncomp=3),
1048 mismatches were observed with max_rel_error=3027 — far beyond f32 precision. The mismatch
pattern is a value-swap (e.g., indices 5 and 7 exchange values) rather than accumulated rounding
error, indicating a wrong index ordering in the sub-kernel, not a precision issue.

**Scope:** Pre-existing bug in the f64 code path (`launch_f12_typed::<f64>`) for certain
non-symmetric shell orderings with p-shells. The f64 oracle never caught it because it only
tests 3 symmetric quartets. CR-01/CR-02 is NOT the cause — the same swap pattern appears
in the f64 run.

**Mitigation in Plan 20-11:** The f32 parity test for int2e_stg_ip1_sph is restricted to the
same 3 quartets used by the f64 oracle, which are symmetric and do not trigger the bug.
This restriction is documented in the test with an inline comment.

**Resolution:** Requires a separate investigation and fix plan. The bug is likely in the
`f12_kernel_core` shell-index ordering logic for the ip1 (gradient) variant when shells i != k
or j != l. No fix attempted in Phase 20 plans — this is a separate correctness issue in the
f12 gradient kernel, orthogonal to the CR-01/CR-02 and precision-generic work.

**Affected operators:** At minimum `int2e_stg_ip1_sph` (operator ID 107). Other f12 ip/derivative
operators may have similar issues; untested.

**Files implicated (not yet diagnosed):**
- `crates/cintx-cubecl/src/kernels/f12.rs` (`launch_f12_typed`, sub-kernel index routing)
- Possibly `crates/cintx-cubecl/src/math/stg.rs` (ip1 gradient math)
