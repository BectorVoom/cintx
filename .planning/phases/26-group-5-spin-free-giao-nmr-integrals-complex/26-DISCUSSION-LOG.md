# Phase 26: Group 5 (spin-free) — GIAO / NMR Integrals (complex) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-31
**Phase:** 26-group-5-spin-free-giao-nmr-integrals-complex
**Areas discussed:** Complex routing, Safe surface, Vendor parity, Sequencing, CubeCL authoring

---

## Complex routing (FND-03 — how `complex_interleaved` is set per-family)

| Option | Description | Selected |
|--------|-------------|----------|
| Manifest per-family flag | New `complex_output` field in the lock; planner reads it to set the flag + size 2× staging. Data-driven, off manifest not Representation enum. | ✓ |
| Operator-name routing table | Code-side set/match of GIAO op names in the driver. No schema change but roster lives in code, drifts from manifest by hand. | |
| Hybrid: manifest flag + comptime kernel | Manifest flag drives planner sizing AND a comptime flag into the `#[cube]` kernel. | (folded in) |

**User's choice:** Manifest per-family flag (D-01).
**Notes:** The comptime-kernel aspect of the hybrid option was folded in as D-02 — the same
manifest field flows as a comptime hint to the device kernel so one field drives host contract,
staging, and device output layout.

---

## Safe surface (what a purely-imaginary cart/sph family returns to a caller)

| Option | Description | Selected |
|--------|-------------|----------|
| `num_complex::Complex<f64>` view | Typed view via existing `complex_values()` gate; round-trip asserts non-zero im / zero re. Matches CLAUDE.md num-complex choice + spinor surface. | ✓ |
| Raw interleaved f64 + flag | Flat 2× buffer + flag; caller deinterleaves. Leaks layout, no num_complex dep. | |

**User's choice:** `num_complex::Complex<f64>` view (D-03).
**Notes:** `complex_values()` already returns `Some` only when `complex_interleaved` is set, so
the surface exists once FND-03 flips the flag for cart/sph.

---

## Vendor parity (binding libcint `double complex *out` + proving imaginary lands)

| Option | Description | Selected |
|--------|-------------|----------|
| Reinterpret as 2×-interleaved f64, non-zero gauge gate | Bind `out` as `*mut f64` len 2N (same layout as `double complex`), compare elementwise at 1e-12, gate every family on the non-zero gauge-origin fixture. | ✓ |
| Complex64 FFI binding | repr-C `Complex64` FFI signature, compare as complex slices. Stronger typing, needs repr-C bindgen handling. | |

**User's choice:** Reinterpret as 2×-interleaved f64 (D-05/D-06).
**Notes:** Extended with D-07 — assert imaginary half non-zero AND real half exactly zero for
purely-imaginary families. Complex64 FFI binding noted as a deferred ergonomics improvement.

---

## Sequencing (how the work splits into plans)

| Option | Description | Selected |
|--------|-------------|----------|
| FND-03 first, then 1e then 2e clusters | Plan 1 = FND-03 foundation (must merge first); Cluster A = 1e GIAO/CG; Cluster B = 2e GIAO; clusters parallelize via worktrees. | ✓ |
| FND-03 + first 1e family in one plan | Bundle FND-03 with int1e_igovlp as a vertical slice, then rest. | |

**User's choice:** FND-03 first, then 1e then 2e clusters (D-09). Mirrors Phase-25 D-06.

---

## CubeCL authoring (user clarification, mid-discussion)

**User directive (verbatim intent):** "Please write all calculations according to the CubeCL
manual. CubeCL kernels need generics-float in this project. Read the manual before writing any
code." Manual at `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md`.

**Captured as:** D-08 — all GIAO `#[cube]` kernels generic over `F` (`Float` bound), follow the
CubeCL manual, executor must read the load-bearing pages (Generics, Algebra, Basic Operations,
Conditionals) before writing any kernel code. Reinforces the standing project-wide convention.

---

## Claude's Discretion

- Exact `int1e_giao_*` / `int1e_cg_*` / `int2e_giao_*` roster (derive from libcint 6.1.3 source).
- Exact `component_rank` (r_gauge × ∇ tensor multiplicity) and gout component order per family.
- Precise manifest field name/shape for the complex flag (`complex_output: bool` vs multiplier).
- Corpus shell-tuple selection per `vendor_*` test (non-square + non-zero-gauge constrained).
- One parameterized `#[cube]` entry vs per-family launchers.

## Deferred Ideas

- GIAO×σ spinor slice (GIAO-03) → Phase 30 (needs Gap B2 `c2s_si` transform, Phase 28).
- Spinor GIAO representations → `UnsupportedApi` this phase.
- `Complex64` repr-C FFI binding → later ergonomics improvement, not needed for byte-identity.
- `rys-nroots-ge6-wheeler-fallback` todo → resolved in Phase 25; reviewed, not folded.
