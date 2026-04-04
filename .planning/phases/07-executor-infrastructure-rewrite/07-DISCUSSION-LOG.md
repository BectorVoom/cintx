# Phase 7: Executor Infrastructure Rewrite - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-03
**Phase:** 07-executor-infrastructure-rewrite
**Areas discussed:** Backend enum design, CPU backend integration

---

## Backend Enum Design

| Option | Description | Selected |
|--------|-------------|----------|
| ResolvedBackend enum (Recommended) | Enum with Wgpu/Cpu arms, per-arm kernel dispatch via match. Extensible with future Cuda/Rocm/Metal arms. | ✓ |
| Runtime trait with type erasure | Keep BackendExecutor trait but add runtime-generic dispatch internally. More indirection. | |
| Feature-flag compile-time selection | Only one backend compiled at a time via cfg features. Can't switch at runtime. | |

**User's choice:** ResolvedBackend enum
**Notes:** Required because dyn BackendExecutor in planner::evaluate rules out generics on CubeClExecutor<R: Runtime>.

---

## CPU Backend Integration

### Backend role
| Option | Description | Selected |
|--------|-------------|----------|
| Primary oracle path | CPU is THE path for oracle parity CI. wgpu tested opportunistically. | |
| Secondary test path | wgpu remains primary. CPU is fallback for CI without GPU. | |
| Equal standing | Both backends must pass oracle parity independently. Tests run on both. | ✓ |

**User's choice:** Equal standing — both backends must pass oracle parity independently.

### Backend selection mechanism
| Option | Description | Selected |
|--------|-------------|----------|
| BackendIntent enum in options | API callers pass BackendIntent::Wgpu or Cpu via ExecutionOptions. | |
| Environment variable | CINTX_BACKEND=wgpu\|cpu. Runtime reads env at executor init. | ✓ |
| Both options + env | API and env var. Env takes precedence. | |

**User's choice:** Environment variable (`CINTX_BACKEND`)
**Notes:** User clarified "User can select backend" — runtime env var selection, not compile-time feature gate.

---

## Claude's Discretion

- Buffer lifecycle placement (per-family module — from research recommendation)
- RecordingExecutor removal timing (same phase — user confirmed earlier in conversation)
- f64 strategy details (wgpu gates on SHADER_F64, cpu always supports)
- Exact enum/struct naming

## Deferred Ideas

- CUDA/ROCm/Metal backends — v1.2+
- Screening/batching optimizations — after correctness
