# Phase 20: Generic Float Precision (f64/f32 Switch) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-20
**Phase:** 20-precision-generic-f64-f32-switch
**Areas discussed:** Switch mechanism, Oracle contract, Refactor scope, Phase setup, API shape, C ABI scope

---

## Switch mechanism

| Option | Description | Selected |
|--------|-------------|----------|
| Compile-time feature | Central `Scalar` alias swapped by a Cargo feature; one precision per build; lowest risk. | |
| Runtime selection | Both precisions compiled, chosen per call; CubeCL monomorphizes both kernel sets (~2× cost). | |
| Generic over `F: Float` | Thread a float type parameter through kernels + safe API; caller picks. Max flexibility, max churn. | ✓ |

**User's choice:** Generic over `F: Float`
**Notes:** Chosen despite being the highest-churn option (~3,396 f64 sites). Drives D-01/D-02.

---

## Oracle contract

| Option | Description | Selected |
|--------|-------------|----------|
| f64 strict, f32 loose gate | Keep byte-identity for f64; separate f32 oracle gate at ~1e-4 rtol. | ✓ |
| f32 exempt from oracle | f32 runs regression/self-consistency only; no libcint comparison. | |
| f32 experimental/unverified | Feature-gated, documented as approximate, no oracle claim. | |

**User's choice:** f64 strict, f32 loose gate
**Notes:** f32 stays verified against libcint, just not byte-identical. Drives D-08/D-09.

---

## Refactor scope

| Option | Description | Selected |
|--------|-------------|----------|
| Full compute path | Kernels + math + staging buffers + safe-API outputs parameterize on precision. | ✓ |
| Compute path, f64 boundary | Kernels/buffers switch; libcint-facing compat `env` stays f64. | |
| Everything incl. compat env | Even raw `atm`/`bas`/`env` arrays parameterize; changes raw ABI. | |

**User's choice:** Full compute path
**Notes:** Reaches the kernels (the perf/`SHADER_F64` win) but leaves raw `env` arrays at f64. Drives D-05/D-06.

---

## Phase setup

| Option | Description | Selected |
|--------|-------------|----------|
| New phase, new milestone | Its own milestone (e.g. v1.4 "Precision Flexibility"); milestone-sized cross-cutting refactor. | ✓ |
| New Phase 20 in v1.3 | Append to current milestone. | |
| Capture context only | Discuss + write CONTEXT now, decide placement later. | |

**User's choice:** New phase, new milestone
**Notes:** Thematically distinct from v1.3 "Safe API Closure." Formal milestone open is a Next Step (see CONTEXT.md).

---

## API shape

| Option | Description | Selected |
|--------|-------------|----------|
| Method-level generic | `evaluate::<F>()` generic; setup monomorphic; `evaluate()` defaults to f64. Smallest blast radius. | ✓ |
| Session param, F=f64 default | `SessionRequest<'basis, F = f64>`; precision at construction; existing code compiles via default. | |
| Session param, no default | Fully explicit `SessionRequest<'basis, F>`; breaks existing call sites until updated. | |

**User's choice:** Method-level generic
**Notes:** Existing call sites unchanged; f32 via `req.evaluate::<f32>()`. Drives D-03/D-04.

---

## C ABI scope

| Option | Description | Selected |
|--------|-------------|----------|
| C ABI stays f64-only | f32 is Rust-API only this milestone; C shim keeps libcint `double` contract. | ✓ |
| C ABI gets f32 variants | Precision-suffixed C entry points; enlarges ABI + verification matrix. | |
| Decide later | Defer C-ABI precision exposure to a follow-up phase. | |

**User's choice:** C ABI stays f64-only
**Notes:** Consistent with leaving raw `env` arrays at f64. Drives D-07.

---

## Claude's Discretion

- Exact per-family f32 tolerance floors (empirical/research-driven).
- Internal helper genericization order and intermediate type-alias scaffolding.
- Whether to introduce a sealed `Scalar`/`CintFloat` super-trait bridging device `Float` and host `num_traits::Float`.

## Deferred Ideas

- C-ABI f32 variants (f64-only this milestone).
- Raw compat `env`/`atm`/`bas` precision parameterization (kept f64).
- Runtime / mixed-precision per-call dispatch (chose static generic).
- Other precisions (f16/bf16, extended) — out of scope.

## Locked constraint (user instruction)

- Refactor MUST be driven by the **serena MCP server** symbol-aware tools, not blind text replacement (CONTEXT.md D-11).
