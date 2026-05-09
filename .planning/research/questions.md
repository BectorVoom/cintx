# Open Research Questions

Questions surfaced during exploration sessions that need deeper investigation
before or during the relevant phase. Append new questions; do not remove
historical ones — mark them `[answered]` with a link to the artifact that
resolved them.

---

## 2026-05-09 — Multi-backend (cuda / rocm / metal) feature wiring

**Source:** `/gsd:explore` — multi-backend feature + env-var support

**Question:** Do `cubecl-cuda 0.10.0`, `cubecl-hip 0.10.0`, `cubecl-metal 0.10.0`,
and the existing `cubecl-wgpu 0.10.0` resolve together cleanly under our pinned
graph (`Cargo.lock`, resolver 3, Rust 1.94.0), and what platform-specific build-
time dependencies does each require?

**Why it matters:** The phase compiles all four cubecl runtime crates behind
additive Cargo features. If any of them pulls a conflicting version of a shared
transitive dep (e.g. `wgpu`, `naga`, `bytemuck`), or requires a platform SDK
that isn't available in our CI image, the feature matrix won't build and we'll
have to revisit either feature gating or pin selection before plan-phase locks
the task list.

**Concretely, find out:**

1. Does `cargo check --no-default-features --features cuda,rocm,metal,wgpu` in
   `cintx-cubecl` resolve and build (excluding kernel link, which needs the
   real toolchain)?
2. What system-level deps does each require to *compile* (not run)?
   - cubecl-cuda: CUDA toolkit headers? Stub libs?
   - cubecl-hip: ROCm headers? `hipcc`?
   - cubecl-metal: Xcode SDK? Metal-cpp? Linux-buildable at all?
3. Are any of these crates platform-gated upstream (e.g. cubecl-metal compiles
   only on Apple targets)? If so, the feature gate must include a target_os
   check or the matrix breaks on Linux dev hosts.
4. Do their transitive `wgpu` / `naga` pins agree with our current
   `wgpu = "29.0.3"`?

**Suggested approach:** Spawn `gsd-phase-researcher` during `/gsd:plan-phase`
for the multi-backend phase, with this question as input. Output goes into the
phase's `RESEARCH.md`.

**Status:** open
