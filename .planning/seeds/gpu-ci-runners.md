---
title: Wire GPU CI runners for cuda / metal smoke tests
trigger_condition: |
  Either (a) CUDA or Apple-Metal hardware becomes accessible to the project
  (developer machine, self-hosted runner, or a paid GitHub Actions GPU/macOS
  runner is approved), OR (b) a downstream user reports a runtime regression
  on cuda or metal that the current compile-only gate would not have caught.
planted_date: 2026-05-09
type: seed
---

# Seed: GPU CI runners for cuda / metal smoke tests

## What

Stand up CI infrastructure that runs a minimal oracle parity smoke test under
`CINTX_BACKEND=cuda` and `CINTX_BACKEND=metal` on every PR, closing the
verification gap recorded in `notes/cuda-metal-verification-gap.md`.

## Why this is a seed, not a phase yet

The multi-backend phase deliberately accepts that cuda and metal are
compile-only on the current dev host. That risk-accept is sufficient as long as:

- Nobody ships cintx onto cuda/metal in production with strong correctness
  expectations, AND
- No downstream user has reported a regression that runtime testing would have
  caught.

When either of those changes, the cost-benefit flips and a phase becomes
worthwhile.

## Sketch of what the phase would look like

- A GitHub Actions workflow with two new jobs:
  - `oracle-smoke-cuda` on a CUDA-capable runner (self-hosted or paid).
  - `oracle-smoke-metal` on a `macos-*` runner with Metal.
- Each job runs `cargo test -p cintx-oracle --features cuda` (resp. `metal`)
  against a small, fast subset of the oracle suite (e.g. one symbol per family).
- Failure surfaces in the same `oracle_parity_gate` style used today.
- Open question to resolve at that time: full oracle matrix, or smoke-only?

## What to revisit when this triggers

- Whether to fold this into the existing `oracle_parity_gate` matrix or keep it
  as a separate nightly job (cost vs. signal).
- Whether `CINTX_BACKEND` is still the right env var (may have evolved).
- Whether cubecl's CI patterns have shifted in the intervening releases.
