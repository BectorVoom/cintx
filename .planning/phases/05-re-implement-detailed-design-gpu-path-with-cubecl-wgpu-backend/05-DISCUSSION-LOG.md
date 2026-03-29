# Phase 5: Re-implement detailed-design GPU path with CubeCL (wgpu backend) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves alternatives considered.

**Date:** 2026-03-29
**Phase:** 05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend
**Areas discussed:** Runtime Backend Policy, Planner/Dispatch Integration Strictness, Unsupported-Scope Reporting, Validation & Regression Gates

---

## Runtime Backend Policy

| Option | Description | Selected |
|--------|-------------|----------|
| wgpu-default, explicit override only | Default wgpu path; explicit override for diagnostics/tests only | |
| Hardcode wgpu only | Always use wgpu with no selector surface | |
| Auto-select with fallback | Try wgpu first and fallback when unavailable | ✓ (initial) |

| Option | Description | Selected |
|--------|-------------|----------|
| Fail closed with typed UnsupportedApi | Capability preflight with explicit typed failures | |
| Let kernel launch fail naturally | Minimal preflight and backend error surface | |
| Fallback to CPU substitute | Route to CPU substitute when GPU capability missing | ✓ (initial) |

| Option | Description | Selected |
|--------|-------------|----------|
| ExecutionOptions + compat opt plumbing | Backend intent explicitly carried as control-plane metadata | ✓ |
| Environment variable only | Global process-level backend toggle only | |
| Executor-internal hardcoded behavior | Policy remains hidden inside executor | |

| Option | Description | Selected |
|--------|-------------|----------|
| Adapter + backend details in trace/artifacts | Include backend/adapter capability diagnostics in outputs | ✓ |
| Trace-only minimal backend tag | Compact runtime tag only | |
| No backend diagnostics | No explicit backend details in diagnostics | |

**Conflict-resolution tie-breaker**

| Option | Description | Selected |
|--------|-------------|----------|
| Auto-select wgpu adapters, then fail closed | Auto-select within wgpu devices; no CPU compute substitute | ✓ |
| Allow explicit cpu fallback path | CPU fallback allowed with explicit metadata | |
| Allow transparent fallback | Automatic substitute path without strict fail-closed behavior | |

**User's choice:** Auto-select among wgpu adapters and fail closed when unavailable; no substitute compute fallback.
**Notes:** Initial answers favored fallback, then explicitly resolved to fail-closed contract for planning clarity.

---

## Planner/Dispatch Integration Strictness

| Option | Description | Selected |
|--------|-------------|----------|
| End-to-end real path now | Replace placeholder execution with real CubeCL flow across integration stack | ✓ |
| Runtime wiring now, synthetic compute temporarily | Keep synthetic execution behavior for now | |
| Planner-only now, backend later | Defer real backend execution | |

| Option | Description | Selected |
|--------|-------------|----------|
| Keep strict staging->compat final write | Preserve backend staging-only and compat final-write ownership | ✓ |
| Allow backend direct final writes | Backend can write caller-visible output directly | |
| Make ownership runtime-configurable | Ownership model selectable at runtime | |

| Option | Description | Selected |
|--------|-------------|----------|
| Always CubeCL compute per chunk | CPU remains control-plane only; compute remains CubeCL | ✓ |
| Allow host micro-chunk shortcut | CPU substitute compute for small chunks | |
| Per-family exception policy | Allow bypass per family | |

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, fail on policy drift | Query/evaluate backend policy mismatch is typed error | ✓ |
| Allow evaluate-time backend switch | Evaluate can switch policy from query | |
| Auto-replan on mismatch | Runtime replans transparently | |

**User's choice:** Strict end-to-end CubeCL integration with contract-preserving ownership and fail-on-drift behavior.
**Notes:** Existing ownership and query/evaluate contracts should remain non-negotiable.

---

## Unsupported-Scope Reporting

| Option | Description | Selected |
|--------|-------------|----------|
| Explicit typed failure, no hidden fallback | Unsupported/capability limits are surfaced as typed failures | ✓ |
| Auto-fallback with warning | Substitute compute path with warning | |
| Silent compatibility fallback | Substitute path without explicit signal | |

| Option | Description | Selected |
|--------|-------------|----------|
| Runtime + artifactized unsupported matrix | Runtime reasons plus report artifact visibility | ✓ |
| Runtime error messages only | No artifactized unsupported report | |
| Test-only visibility | Visibility only in tests/CI logs | |

| Option | Description | Selected |
|--------|-------------|----------|
| Retarget envelope to wgpu capability checks | Keep strict envelope but change backend gate to wgpu capability | ✓ |
| Keep cpu-only gate for now | Delay wgpu envelope migration | |
| Temporarily relax envelope constraints | Broaden now and tighten later | |

| Option | Description | Selected |
|--------|-------------|----------|
| Specific unsupported reason with phase note | Detailed reason taxonomy for unimplemented scope | ✓ |
| Generic unsupported message | Minimal unsupported reason detail | |
| Use temporary emulation path | Substitute execution path instead of unsupported errors | |

**User's choice:** Fail-closed, explicit unsupported taxonomy with artifactized visibility.
**Notes:** This aligns with no-masking policy and reduces ambiguity for downstream planning.

---

## Validation & Regression Gates

| Option | Description | Selected |
|--------|-------------|----------|
| Layered tests across runtime+cubecl+compat | Multi-layer regression protection | ✓ |
| CubeCL crate unit tests only | Limit to internal cubecl tests | |
| Oracle parity only | End-to-end only | |

| Option | Description | Selected |
|--------|-------------|----------|
| Capability-aware required gate | Required CI checks with explicit capability skip metadata | ✓ |
| Advisory-only GPU checks | Non-blocking CI policy | |
| No CI GPU enforcement | No wgpu CI regression gates | |

| Option | Description | Selected |
|--------|-------------|----------|
| Explicit anti-pseudo assertions | Dedicated guards against placeholder/synthetic compute return | ✓ |
| Rely on output parity only | No direct anti-pseudo checks | |
| Manual review policy only | Human process only | |

| Option | Description | Selected |
|--------|-------------|----------|
| Assert reason taxonomy + reporting artifact | Verify unsupported reasons and report outputs | ✓ |
| Assert only UnsupportedApi type | Error-kind only checks | |
| No explicit unsupported tests | No dedicated unsupported behavior tests | |

**User's choice:** Strong regression gates with layered tests and explicit anti-pseudo + unsupported-reporting checks.
**Notes:** CI policy is required/capability-aware, not advisory-only.

---

## the agent's Discretion

- Exact naming and struct layout for backend-selection diagnostics/control-plane fields.
- Exact test module layout and artifact filenames, as long as selected policies are enforced.

## Deferred Ideas

None.
