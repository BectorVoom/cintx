---
title: CUDA / Metal runtime verification gap (multi-backend phase)
date: 2026-05-09
type: note
context: explore session — multi-backend (cuda/rocm/metal) feature + env-var support
---

# CUDA / Metal runtime verification gap

When we add `cuda`, `rocm`, and `metal` Cargo features to `cintx-cubecl`, only
**cpu**, **wgpu**, and **rocm (cubecl-hip)** are runtime-verifiable on the current
development host. This note records the explicit risk-accept decision so the
phase plan does not re-litigate it.

## What we can verify locally

| Backend | Local hardware | Runtime check possible? |
|---------|----------------|-------------------------|
| cpu     | always         | yes                     |
| wgpu    | Vulkan/DX12    | yes                     |
| rocm    | AMD GPU + ROCm | yes                     |
| cuda    | (none)         | **no** — compile-only   |
| metal   | (none, Linux)  | **no** — compile-only   |

## Decision

- The phase ships `cuda` and `metal` as **compile-only** code paths.
- `cargo check --features cuda,metal` (and matrix variants) is the only local
  gate; the oracle parity suite is **not** run against these backends in this
  phase.
- README / module docs explicitly mark `cuda` and `metal` as "unverified at
  runtime; trust delegated to upstream cubecl-cuda 0.10.0 / cubecl-metal 0.10.0".

## Why we accept the risk

- The user has no CUDA or Apple-Metal hardware on this development host.
- Standing up GPU CI runners is a separate, larger investment that should not
  block delivering the feature/env-var control surface that already benefits
  cpu / wgpu / rocm users today.
- cubecl's per-runtime crates encapsulate the backend-specific work; cintx's
  contribution is the dispatch layer, which is the same shape across backends.

## What would close the gap

- A GitHub Actions GPU runner (or self-hosted) with NVIDIA + macOS coverage
  running an oracle smoke test on each backend on every PR — see seed
  `gpu-ci-runners.md`.
- Or: a contributor with the relevant hardware running the existing oracle
  harness under `CINTX_BACKEND=cuda` / `CINTX_BACKEND=metal` and reporting
  results.

## How to apply during planning / execution

- Plan tasks may include `cargo check`-style matrix verification for cuda/metal,
  but **must not** add an oracle parity gate that requires those runtimes.
- Any test that needs a live cuda/metal client must be `#[ignore]` by default
  with a documented opt-in (e.g. env-var-gated).
- If a planner pass surfaces a "missing oracle coverage" finding for cuda/metal
  in this phase, link it to this note rather than treating it as a blocker.
