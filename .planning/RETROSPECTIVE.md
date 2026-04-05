# Retrospective

## Milestone: v1.1 — CubeCL Direct Client API & Real Kernel Compute

**Shipped:** 2026-04-05
**Phases:** 4 (7-10) | **Plans:** 18

### What Was Built
- CubeCL executor rewritten with direct client API (`client.create`/`client.read`/`ArrayArg`) and ResolvedBackend dispatch
- CPU backend for oracle CI testing without GPU hardware
- Boys function, Rys quadrature roots/weights, and Obara-Saika VRR/HRR as `#[cube]` functions
- Real 1e kernels (overlap, kinetic, nuclear) with Condon-Shortley cart-to-sph transform
- Real 2e, 2c2e, 3c1e, 3c2e kernels with multi-index c2s transforms
- Five-family oracle parity gate closed vs vendored libcint 6.1.3

### What Worked
- Host-side kernel approach: implementing math as host functions (not GPU kernels) allowed rapid iteration and debugging against vendor reference
- Vendored libcint compilation via cc crate gave hermetic oracle comparison — no external libcint install needed
- Incremental oracle coverage per family as each kernel landed (VERI-05) caught bugs early
- Phase structure matched natural dependency order (infra → math → 1e → multi-center)

### What Was Inefficient
- Some tolerance thresholds were set too loose initially (3c1e at 1e-7, 3c2e at 1e-9) and had to be tightened in later phases
- RecordingExecutor removal required touching multiple crates (compat, rs, cubecl) — could have been scoped tighter
- Multiple plan iterations on env slot collisions (PTR_RANGE_OMEGA) that could have been caught by a shared env layout constant

### Patterns Established
- Oracle parity test pattern: build H2O STO-3G fixtures, call vendor FFI, compare element-wise
- Host kernel pipeline: pair_data → Boys/Rys → VRR → HRR → c2s transform → output staging
- Vendor build integration: `cc::Build` with libcint source files, `extern "C"` FFI wrappers

### Key Lessons
- Host-side math validation before GPU migration avoids debugging GPU code blindly
- Env slot layout should have a single source of truth (constant table) to prevent collision bugs
- Oracle-first development (test against vendor THEN implement) is more efficient than implement-then-test

## Cross-Milestone Trends

| Metric | v1.0 | v1.1 |
|--------|------|------|
| Phases | 6 | 4 |
| Plans | ~30 | 18 |
| Timeline | ~12 days | ~1 day |
| Key pattern | Scaffold + stub | Real kernels + oracle |
