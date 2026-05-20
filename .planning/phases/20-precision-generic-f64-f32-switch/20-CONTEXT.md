# Phase 20: Generic Float Precision (f64/f32 Switch) - Context

**Gathered:** 2026-05-20
**Status:** Ready for planning
**Milestone:** v1.4 "Precision Flexibility" (proposed — not yet formalized; see Next Steps)

> ⚠️ **Not yet on the roadmap.** This work was discussed ahead of a roadmap
> entry. It is **milestone-sized** (~3,396 `f64` sites across 8 crates) and will
> likely decompose into multiple phases. The decisions below are **milestone-level**
> and apply across whatever phases v1.4 decomposes into. See Next Steps.

<domain>
## Phase Boundary

Parameterize the cintx **compute path** over a generic float type `F: Float` so callers
can evaluate integrals in **f64 (default, byte-identity)** or **f32 (loose-tolerance,
unlocks non-`SHADER_F64` GPUs)**. Precision is chosen at the call site via a
**method-level generic** `evaluate::<F>()`; `evaluate()` continues to mean f64.

**In scope:** CubeCL kernels, shared `#[cube]` math (Boys / Rys / Obara-Saika),
staging buffers, and safe-API outputs all parameterize on `F`. A separate f32 oracle
gate at single-precision tolerance. f32-capable wgpu path (no `SHADER_F64` requirement).

**Out of scope (this milestone):** raw compat `env`/`atm`/`bas` arrays (stay f64 /
`double` — libcint ABI untouched); the C ABI shim (`cintx-capi` stays f64-only);
runtime/mixed-precision per-call enum dispatch; precisions other than f64/f32.

</domain>

<decisions>
## Implementation Decisions

### Precision switch mechanism
- **D-01:** **Generic over `F: Float`.** Thread a generic float type parameter through
  the compute path. NOT a compile-time Cargo feature, NOT a runtime enum dispatch. The
  two concrete monomorphizations are `f64` (default) and `f32`.
- **D-02:** CubeCL kernels become generic `#[cube] fn ...<F: Float>(...)`. Concrete
  `f64::exp/sqrt/erf` calls become `F`-generic intrinsics; f64 const tables (e.g.
  `SQRTPIE4`, `TURNOVER_POINT: [f64; 40]`) cast to `F`. **RESEARCH FLAG:** confirm the
  CubeCL `Float` trait surface covers the transcendentals the kernels need (`exp`,
  `sqrt`, `erf`) and that const-table casting is sound under monomorphization.

### Public API shape
- **D-03:** **Method-level generic.** `SessionRequest<'basis>` setup stays monomorphic.
  `evaluate::<F>()` is generic and returns `TypedEvaluationOutput<F>`; `evaluate()`
  delegates to `evaluate::<f64>()` so **every existing call site compiles unchanged**.
  Callers opt into f32 with `req.evaluate::<f32>()`.
- **D-04:** `TypedEvaluationOutput` becomes generic with `owned_values: Vec<F>`
  (default `F = f64` on the type). Spinor/complex outputs propagate as `Complex<F>` via
  `num-complex` (already generic). Confirm and thread complex output sites.

### Refactor scope / boundary
- **D-05:** **Full compute path** parameterizes: kernels + shared math + staging buffers
  + safe-API outputs.
- **D-06:** Raw compat `env`/`atm`/`bas` arrays **stay f64** (`double`) — libcint ABI is
  untouched. Precision conversion happens at the kernel/staging boundary (host f64 env →
  device `F` buffers).
- **D-07:** **C ABI shim (`cintx-capi`) stays f64-only** this milestone. No
  precision-suffixed C entry points. f32 is a Rust-API feature only.

### Verification / oracle contract
- **D-08:** **f64 path keeps strict byte-identity** against libcint (existing per-family
  atol ~1e-12). Zero regression: f64 is the default and behaves exactly as today.
- **D-09:** **f32 gets a SEPARATE oracle gate** at a realistic single-precision tolerance
  (~1e-4 rtol; exact per-family floors empirical/research-driven, mirroring Phase 15's
  per-family tolerance model). f32 is verified against libcint — just not byte-identical.
- **D-10:** f32 **unlocks the wgpu backend on adapters lacking `SHADER_F64`** (today
  `check_shader_f64_in_features` fails closed). The f32 path must NOT gate on
  `SHADER_F64`. This is the primary motivation. **RESEARCH FLAG:** confirm f32 shader
  capability is universally available on wgpu adapters.

### Refactoring method (locked by user)
- **D-11:** The refactor **MUST be performed using the serena MCP server's symbol-aware
  tools** (`find_symbol`, `find_referencing_symbols`, `rename_symbol`,
  `replace_symbol_body`, `insert_before/after_symbol`) — NOT blind text replacement.
  With ~3,396 `f64` sites, symbol-level edits are required so const tables, comments, and
  *deliberately-f64* sites (env ABI per D-06, C ABI per D-07) are not corrupted. The
  executor must call serena `check_onboarding_performed` / `initial_instructions` first.

### Default & backward-compat
- **D-12:** **f64 stays the default everywhere.** No existing public signature breaks
  (method-level generic + `F = f64` type defaults). All existing oracle gates, manifest
  locks, and tests MUST pass unchanged on the f64 path.

### Claude's Discretion
- Exact per-family f32 tolerance floors (empirical, research-driven).
- Internal helper genericization order and any intermediate type-alias scaffolding.
- Whether to introduce a sealed `Scalar`/`CintFloat` super-trait bridging device-side
  CubeCL `Float` and host-side `num_traits::Float` (needed for host math like
  `boys_gamma_inc_host`).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Precision-strategy precedent (the decision this milestone relaxes)
- `.planning/phases/07-executor-infrastructure-rewrite/07-CONTEXT.md` §"f64 precision strategy" (D-09) — the locked "both backends must produce f64-precision results" decision; this milestone extends it to allow an opt-in f32 path. MUST read.
- `docs/design/cintx_detailed_design.md` — authoritative design doc; precision/dtype assumptions and module boundaries.

### Oracle / tolerance model the f32 gate must mirror
- `.planning/phases/15-oracle-tolerance-unification-manifest-lock-closure/15-CONTEXT.md` — per-family atol/rtol unification model; the f32 gate is a parallel, looser instance.
- `crates/cintx-oracle/src/compare.rs` — tolerance comparison logic.
- `crates/cintx-oracle/src/lib.rs` — per-family tolerance source / `tolerance_for_family`.

### Capability / feature & error conventions
- `.planning/phases/16-multi-backend-support/16-CONTEXT.md` (D-01..D-09) — additive-capability + typed-error + **no-silent-fallback** conventions; `SHADER_F64` capability-gating context relevant to D-10.

### Project constraints
- `CLAUDE.md` — libcint 6.1.3 byte-identity goal, CubeCL-primary architecture, `thiserror`(lib)/`anyhow`(tooling) split, verification gates (manifest lock, feature-matrix CI, oracle parity).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `crates/cintx-rs/src/api.rs:125` — `SessionRequest::evaluate(self) -> Result<TypedEvaluationOutput, FacadeError>`; the public surface to genericize at the method level (D-03).
- `crates/cintx-rs/src/api.rs:501` — `TypedEvaluationOutput { owned_values: Vec<f64> }`; becomes `Vec<F>` (D-04).
- `crates/cintx-cubecl/src/math/boys.rs` — representative concrete-f64 math: consts `SQRTPIE4`, `TURNOVER_POINT: [f64; 40]`, intrinsics `f64::exp/sqrt/erf`, and a host variant `boys_gamma_inc_host`. The genericization pattern proven here transfers to Rys/Obara-Saika.
- `crates/cintx-cubecl/src/executor.rs` — `check_shader_f64_in_features` / `check_f64_capability`; the fail-closed `SHADER_F64` gate the f32 path must bypass (D-10).
- `num-complex` (workspace dep) — `Complex<F>` for generic spinor/complex outputs (D-04).

### Established Patterns
- Phase 16 additive-capability + typed-error + no-silent-fallback (`16-CONTEXT.md` D-01..D-09).
- Phase 15 per-family tolerance constants — the oracle-gate model the f32 gate copies at looser tolerance.
- `#[cube]` math is written in CubeCL-compatible form (loop counters are `u32`; function-call syntax `f64::exp(x)` not method `.exp()`; `Array<f64>` indexed with `as usize`). The generic form MUST preserve these CubeCL constraints.

### Integration Points
- Per-family kernel launchers in `cintx-cubecl` — must thread `F`.
- Staging-buffer marshaling — host f64 `env` (D-06) converted to device `F` buffers.
- Safe-API evaluate path → `TypedEvaluationOutput<F>` (D-03/D-04).

</code_context>

<specifics>
## Specific Ideas

- Whole refactor driven by **serena MCP** symbol tools (D-11) — user's explicit instruction.
- f32's payoff is concrete: **run the wgpu backend on GPUs without `SHADER_F64`** (D-10).
- Backward compatibility is non-negotiable: `evaluate()` and all f64 oracle gates must be
  byte-for-byte unchanged (D-08, D-12).

</specifics>

<deferred>
## Deferred Ideas

- **C-ABI f32 variants** — decided f64-only this milestone (D-07); revisit in a follow-up phase if C consumers need single precision.
- **Raw compat `env`/`atm`/`bas` precision parameterization** — kept f64 (D-06); a separate effort if ever needed.
- **Runtime / mixed-precision per-call dispatch** — chose static generic (D-01), not a runtime `Precision` enum.
- **Other precisions (f16/bf16, extended)** — explicitly out of scope.

</deferred>

---

*Phase: 20-precision-generic-f64-f32-switch*
*Context gathered: 2026-05-20*
