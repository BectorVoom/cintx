# Phase 16: Multi-Backend Support (cuda / rocm / metal) with Feature + Env-Var Selection - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-09
**Phase:** 16-multi-backend-support
**Areas discussed:** Error variant taxonomy, Cargo default features, Platform gating semantics, CI matrix + ROCm smoke scope

---

## Error variant taxonomy

### Q1 — Error variant for un-compiled backend

| Option | Description | Selected |
|--------|-------------|----------|
| New BackendNotCompiled variant | Add a dedicated variant to cintxRsError carrying `{ requested, compiled_in }`. Cleaner taxonomy, future-proof. Minor SemVer bump on the error enum. | ✓ |
| Reuse UnsupportedApi | Reuse `UnsupportedApi { requested }` with a tag like `backend:cuda:not-compiled`. Zero error-API surface change. Conflates two different failure classes. | |

**User's choice:** New BackendNotCompiled variant.

### Q2 — Variant location and fields

| Option | Description | Selected |
|--------|-------------|----------|
| cintxRsError in cintx-core, fields: requested + compiled_in | Public enum in cintx-core. Fields `{ requested: String, compiled_in: Vec<String> }` so callers render `requested 'cuda', compiled-in: [cpu, wgpu]`. | ✓ |
| cintxRsError, requested only | Add the variant but only `{ requested: String }`; compiled-in list pulled from a separate query helper. | |
| cintx-cubecl-only error type | Backend-specific error in cintx-cubecl, converted at the crate boundary. Adds a conversion layer; risks divergent error surfaces. | |

**User's choice:** cintxRsError in cintx-core, fields: requested + compiled_in.

### Q3 — Failure mode for unrecognized CINTX_BACKEND values

| Option | Description | Selected |
|--------|-------------|----------|
| Hard error — InvalidEnvParam | Typed error (Phase 13's variant) listing recognized backend names. No silent fallback. | ✓ |
| Hard error — same BackendNotCompiled variant | Treat unknown values as "not compiled in" with the actual compiled list. Single error path; less precise. | |
| Keep current warn + fallback | Warn-log + fall back to default. Easier migration but conflicts with hard-error theme. | |

**User's choice:** Hard error — InvalidEnvParam.

### Q4 — Resolution API shape

| Option | Description | Selected |
|--------|-------------|----------|
| `resolve_backend_kind() -> Result<BackendKind, cintxRsError>` | Convert the existing fn to fallible. All callers thread the typed error up. Single chokepoint. | ✓ |
| Add `resolve_backend_kind_strict()` alongside | Keep the infallible helper for legacy paths, add a strict fallible helper. Two helpers diverge over time. | |
| Move resolution into ResolvedBackend::from_intent | Drop the helper; env-var resolution happens inside `from_intent`. Couples env parsing to backend bootstrap. | |

**User's choice:** Change `resolve_backend_kind()` to return Result.

---

## Cargo default features

### Q1 — New default features list

| Option | Description | Selected |
|--------|-------------|----------|
| `default = []` | Empty default. cpu always linked (originally proposed unconditional). All GPU backends opt-in. | (initially picked) |
| `default = ["wgpu"]` | Preserve today's effective behavior. Heavier "plain build". | |
| `default = ["wgpu", "rocm"]` | Default includes wgpu + rocm so the dev host builds + smokes out of the box. Too opinionated for a published library. | |

**Initial answer:** `default = []`. Later reconciled — see Q4.

### Q2 — Downstream wgpu wiring

| Option | Description | Selected |
|--------|-------------|----------|
| (rephrased after user clarified "Wgpu is not default") | | |
| Gate with `#[cfg(feature = "wgpu")]` — silently absent without the flag | Tests that need wgpu compile only with `--features wgpu`. `cargo test` (no flags) runs only cpu-path tests. | ✓ |
| Skip at runtime — `#[ignore]` + env-var | Tests stay compiled but auto-skip when wgpu isn't present. | |
| Hard-error at runtime — same BackendNotCompiled path | Tests fail with the new error when wgpu isn't compiled. Treats missing-feature as a test failure. | |

**User clarification mid-flow:** "Wgpu is not default." — captured as a firm rule.

**User's choice:** Gate with `#[cfg(feature = "wgpu")]`.

### Q3 — Per-feature dependency shape

| Option | Description | Selected |
|--------|-------------|----------|
| Each feature pulls its own cubecl runtime crate | `cuda = ["dep:cubecl-cuda"]`, `rocm = ["dep:cubecl-hip"]`, etc. cubecl optional crate deps. Decouples our names from upstream feature names. | ✓ |
| Bundle through cubecl umbrella crate features | `cuda = ["cubecl/cuda"]`, etc. Couples to upstream feature naming. | |
| Each feature also pulls cpu | Same as option 1, but each backend feature additionally enables cpu. No-op since cpu is always-on. | |

**User's choice:** Each feature pulls only its own cubecl runtime crate.

### Q4 — cpu feature retirement

| Option | Description | Selected |
|--------|-------------|----------|
| Remove the cpu feature; add cubecl-cpu as unconditional dep | Delete the cpu line; cubecl-cpu always linked. All `#[cfg(feature = "cpu")]` arms become unconditional. Aligns with ROADMAP's "unconditional" wording. | |
| Keep cpu as a feature, mark it default=true and undocumented | Less churn — leave the feature on by default, document toggling off as unsupported. | ✓ |
| Remove cpu feature; gate cubecl-cpu by no-other-backend cfg | Always-present but conditionally skip its initialization. Saves nothing. | |

**User's choice:** Keep cpu as a feature flag, default-on, undocumented.

### Q5 — Reconciliation (default features)

After Q4, the previous `default = []` answer contradicted "cpu stays as feature flag". Re-asked:

| Option | Description | Selected |
|--------|-------------|----------|
| `default = ["cpu"]` | cpu stays in defaults; cargo build gives cpu out of the box. wgpu/cuda/rocm/metal opt-in only. | ✓ |
| `default = []` (override Q4) | cpu becomes literally unconditional (no flag); --no-default-features still gives cpu. More code churn. | |
| `default = []`; cpu flag exists but defaults off | Strictest opt-in stance. Breaks `cargo test` (no flags). | |

**User's choice:** `default = ["cpu"]`.

---

## Platform gating semantics

### Q1 — target_os gating

| Option | Description | Selected |
|--------|-------------|----------|
| No target_os gating; trust upstream cubecl gating | cintx-cubecl features unconditionally pull cubecl-{metal,cuda,hip}. Failure mode is whatever upstream decides. | ✓ |
| Gate at our layer with target_os cfg | Each backend feature wraps its module in `#[cfg(all(feature = "metal", target_os = "macos"))]`. Enabling metal on Linux compiles cleanly with no Metal arms. | |
| Gate strictly — hard build error on unsupported target_os | `compile_error!` on unsupported target. Loud but blocks the dev-host feature-matrix sweep. | |

**User's choice:** No target_os gating; trust upstream.

### Q2 — Variant cfg style

| Option | Description | Selected |
|--------|-------------|----------|
| Per-variant cfg only; matches use `#[cfg(...)]` arms | Verbose but mechanical. Compiler enforces exhaustiveness per feature combination. Mirrors today's Cpu pattern. | ✓ |
| Helper enum with discriminant + lookup table | Avoid cfg explosion in matches. Loses compile-time exhaustiveness checks. | |
| Wgpu stays unconditional; only Cuda/Rocm/Metal cfg-gated | Compromise. Contradicts "wgpu is never default" rule. | |

**User's choice:** Per-variant cfg only.

### Q3 — Default kind

| Option | Description | Selected |
|--------|-------------|----------|
| Default = Cpu — always | `impl Default for BackendKind { fn default() -> Self { Self::Cpu } }`. Cpu is always compiled, infallible. Aligns with ROADMAP success criterion 5. | ✓ |
| Default = Cpu; BackendIntent default selector also unchanged | Same as above for both BackendKind and BackendIntent. | |
| Default = first compiled backend in priority order | Pick from `cpu < wgpu < cuda < rocm < metal` at compile time via cfg. Breaks the simple invariant. | |

**User's choice:** Default = Cpu — always.

### Q4 — Migration of implicit `BackendIntent::default()` callers

| Option | Description | Selected |
|--------|-------------|----------|
| Flip default; audit + fix every implicit caller in this phase | Search for callsites; per-callsite decide explicit Wgpu vs accept Cpu. Bigger blast radius now, no hidden behavior changes. | ✓ |
| Flip default; leave implicit callers as-is | Tests that need wgpu start running on cpu. Risks silent backend switches. | |
| Keep BackendIntent default = Wgpu but resolve_backend_kind() defaults to Cpu | Two defaults. Splits the contract. | |

**User's choice:** Flip default; audit + fix every implicit caller.

---

## CI matrix + ROCm smoke scope

### Q1 — CI matrix scope

| Option | Description | Selected |
|--------|-------------|----------|
| Curated 6-cell matrix | cpu-only, +wgpu, +cuda, +rocm, +metal, all-features. Each backend in isolation + all-on. | |
| Full 16-subset matrix | Every non-empty subset of {wgpu,cuda,rocm,metal}. Maximum coverage; 16 jobs per PR. | |
| 3-cell minimum: cpu-only, +wgpu, all-features | Cheapest. If all-features breaks, requires bisection. | ✓ |

**User's choice:** 3-cell minimum.

### Q2 — CI command per cell

| Option | Description | Selected |
|--------|-------------|----------|
| `cargo check` only | Fastest; catches compile + resolver issues. Tests stay in the dedicated oracle gate. | |
| `cargo check` + `cargo test` (no oracle, no #[ignore]) | Catches compile + non-oracle regressions per cell. | ✓ |
| `cargo build --release` | Catches release-only codegen issues. Slowest. | |

**User's choice:** cargo check + cargo test (no oracle, no #[ignore]).

### Q3 — ROCm smoke scope

| Option | Description | Selected |
|--------|-------------|----------|
| Single 1e-overlap symbol on a 2-shell H2 fixture | Simplest possible parity check. | |
| Curated 5-symbol set | One symbol from each major family. ~5× runtime. | |
| Full base oracle suite at atol=1e-12 across all 5 base families | Highest confidence; ROCm becomes a first-class oracle gate on capable hosts. | ✓ |

**User's choice:** Full base oracle suite.

### Q4 — ROCm trigger and venue

| Option | Description | Selected |
|--------|-------------|----------|
| Local + opt-in CI; `#[cfg(feature="rocm")]` + `#[ignore]`, env-gated trigger | Tests compile only with `--features rocm`; `#[ignore]` keeps them off the default test pass. Dev box runs them on demand via env-gated invocation. xtask helper wraps it. No CI gate. | ✓ |
| Local-only; no CI gate | Same gating but no xtask wrapper. | |
| CI gate on a self-hosted ROCm runner (advisory) | Self-hosted runner labeled `[self-hosted, linux, x64, gpu, rocm]`, advisory until reliable. Adds runner ops cost. | |

**User's choice:** Local + opt-in CI; `#[cfg(feature="rocm")]` + `#[ignore]` + env-gated trigger; xtask wrapper.

### Q5 — Feature-matrix CI job placement and status

| Option | Description | Selected |
|--------|-------------|----------|
| New required job `feature_matrix_gate` alongside the existing 4 required gates | Required for PR merge. Matches Phase 4 fail-closed CI architecture. | ✓ |
| Extend oracle_parity_gate matrix to include feature combos | Combinatorial blow-up (12+ jobs). | |
| Advisory only; not required for merge | Lower-risk; cuda/metal/rocm compile breakage could land silently. | |

**User's choice:** New required job `feature_matrix_gate`.

---

## Claude's Discretion

The following implementation details were left to Claude / downstream agents:

- Internal organization of new backend modules (`cuda_backend.rs`, `rocm_backend.rs`, `metal_backend.rs`) — pattern-match the existing `wgpu_backend.rs` / `cpu_backend.rs` layout unless research surfaces a reason not to.
- Display/formatting of `BackendNotCompiled` and the `compiled_in` list — compile-time `const &[&str]` vs runtime cfg'd `Vec`. Researcher picks.
- Whether to expose `compiled_backends() -> &'static [&'static str]` as a public introspection helper.
- Capability-token rules per new backend — sentinel/zero fingerprint for compile-only cuda/metal; real fingerprint for rocm when runtime-verified.
- Whether `BackendIntent::selector` grammar gains backend-specific syntax. Lean toward unchanged.

## Deferred Ideas

- GPU CI runners (NVIDIA, Apple Silicon) — would close the cuda/metal verification gap. Tracked in `.planning/seeds/gpu-ci-runners.md`. Phase 17+ candidate.
- `CINTX_BACKEND` aliases (e.g., `hip` → `rocm`) — strict 1:1 names this phase.
- Per-backend selector grammar (`cuda:0`, `rocm:0`).
- Backend-introspection public API (may land with `BackendNotCompiled`; otherwise small follow-up).
