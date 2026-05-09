# Phase 16: Multi-Backend Support (cuda / rocm / metal) with Feature + Env-Var Selection - Context

**Gathered:** 2026-05-09
**Status:** Ready for planning

<domain>
## Phase Boundary

`cintx-cubecl` exposes additive Cargo features `cuda`, `rocm`, `metal` alongside the existing `wgpu`. The `cpu` feature remains as a flag (default-on, undocumented opt-out via `--no-default-features`). `BackendKind` and `ResolvedBackend` gain `Cuda`, `Rocm`, `Metal` variants — each `#[cfg(feature = "...")]`-gated. `CINTX_BACKEND` selects among compiled-in backends at runtime; an unset env var resolves to `Cpu`; requesting an un-compiled backend returns a typed `cintxRsError::BackendNotCompiled`; an unrecognized value returns `InvalidEnvParam`. No silent fallback. `cuda` and `metal` ship as compile-only — no oracle parity gate this phase. `rocm` gets a full base-family oracle suite but it stays opt-in (`#[cfg(feature="rocm")]` + `#[ignore]` + env-gated trigger), not a CI-blocking gate.

</domain>

<decisions>
## Implementation Decisions

### Error Variant Taxonomy
- **D-01:** Add a new typed variant `cintxRsError::BackendNotCompiled { requested: String, compiled_in: Vec<String> }` in `cintx-core`. Public, `thiserror`-derived, surfaces through the existing error enum so all crates render the same diagnostic. Diagnostic format must let callers print `requested 'cuda', compiled-in: [cpu, wgpu]`.
- **D-02:** Unrecognized `CINTX_BACKEND` values (e.g., `CINTX_BACKEND=foobar`) return `cintxRsError::InvalidEnvParam` (the variant introduced in Phase 13) with a payload listing the recognized backend names. No silent warn-and-fallback — that contradicts the phase's hard-error contract.
- **D-03:** `resolve_backend_kind()` changes signature to `fn resolve_backend_kind() -> Result<BackendKind, cintxRsError>`. Single fallible chokepoint. The current infallible helper is removed; all callers thread the typed error up. No parallel `_strict()` helper.

### Cargo Default Features
- **D-04:** **wgpu is NEVER default.** Every consumer opts in explicitly via `--features wgpu`. Tests that need wgpu use `#[cfg(feature = "wgpu")]` and are silently absent without the flag — they do not auto-skip at runtime.
- **D-05:** Each backend feature pulls only its own cubecl runtime crate as an optional dep, never via the cubecl umbrella crate's features:
  - `cuda = ["dep:cubecl-cuda"]`
  - `rocm = ["dep:cubecl-hip"]` (note: feature is named `rocm`, dep is `cubecl-hip`)
  - `metal = ["dep:cubecl-metal"]`
  - `wgpu = ["dep:cubecl-wgpu", "dep:wgpu"]`
  This decouples our feature names from upstream cubecl's internal feature names.
- **D-06:** `cpu` STAYS as a feature flag — do not remove it. `default = ["cpu"]` so `cargo build` gives a working CPU backend out of the box. The `cpu` flag is undocumented in user-facing docs (toggling it off is unsupported but technically possible). This is a deliberate softening of ROADMAP success-criterion-1's "unconditional" wording, chosen to minimize churn against existing `#[cfg(feature = "cpu")]` arms.
- **D-07:** Final `cintx-cubecl/Cargo.toml` `[features]` shape:
  - `default = ["cpu"]`
  - `cpu = ["cubecl/cpu"]` (kept as-is)
  - `wgpu = ["dep:cubecl-wgpu", "dep:wgpu"]`
  - `cuda = ["dep:cubecl-cuda"]`
  - `rocm = ["dep:cubecl-hip"]`
  - `metal = ["dep:cubecl-metal"]`
  - existing `with-f12`, `with-4c1e`, `unstable-source-api` unchanged.
- **D-08:** Downstream crates (cintx-rs, cintx-oracle, integration tests) that require wgpu add explicit `cintx-cubecl/wgpu` opt-in (or a feature-forwarding pattern matching the existing `with-f12` / `with-4c1e` style). Tests that require wgpu are gated `#[cfg(feature = "wgpu")]`.

### Platform Gating Semantics
- **D-09:** **No `target_os` cfg gating in `cintx-cubecl`.** Trust upstream `cubecl-{cuda,hip,metal}` to gate themselves. Our backend features stay ecosystem-neutral; failure mode for incompatible host targets is whatever upstream decides. If `cubecl-metal` doesn't build on Linux, that's an upstream concern surfaced at `cargo check` time.
- **D-10:** **Per-variant `#[cfg(feature = "...")]` gating** on `BackendKind` and `ResolvedBackend`. Concrete shape:
  ```rust
  pub enum BackendKind {
      Cpu,
      #[cfg(feature = "wgpu")]  Wgpu,
      #[cfg(feature = "cuda")]  Cuda,
      #[cfg(feature = "rocm")]  Rocm,
      #[cfg(feature = "metal")] Metal,
  }
  ```
  All `match` sites repeat the same `#[cfg(...)]` per arm. Compiler enforces exhaustiveness per feature combination. Mirrors today's `Cpu` pattern in `ResolvedBackend`.
- **D-11:** `impl Default for BackendKind` returns `Self::Cpu` — always, infallibly (cpu is unconditional). `BackendIntent::default()` also flips its `backend` field to `Cpu` (today's default is `{ Wgpu, "auto" }`). This aligns the struct default with `resolve_backend_kind()`'s unset-env-var behavior and ROADMAP success criterion 5.
- **D-12:** **Migration:** flipping `BackendIntent::default()` from `Wgpu` to `Cpu` is silently behavior-changing for every implicit caller. PLAN.md must include a task that audits every `BackendIntent::default()` and `ExecutionOptions::default()` callsite in `cintx-rs`, `cintx-oracle`, and the test suite, and per callsite either (a) switches to explicit `BackendIntent { Wgpu, ... }` or (b) accepts Cpu. No hidden behavior changes. This is a meaningful blast radius — not an afterthought.

### CI Matrix + ROCm Smoke Scope
- **D-13:** Feature-matrix CI is a **3-cell minimum**: `cpu-only`, `cpu+wgpu`, `all-features`. ROADMAP's "at minimum cargo check per feature combo" is interpreted as "each new feature exercised at least once via the `all-features` cell, plus the existing `cpu` and `cpu+wgpu` baselines". Subset interactions (e.g., `cuda+metal`) are not exercised this phase. If `all-features` breaks, bisection narrows the offending feature.
- **D-14:** Each cell runs **`cargo check` + `cargo test`** (excluding `#[ignore]` and oracle parity tests — those live in their own dedicated job, `oracle_parity_gate`, which already runs the cpu/wgpu profile matrix). Catches compile + non-oracle test regressions per cell without duplicating oracle work.
- **D-15:** **ROCm full base-family oracle suite** (1e overlap, 1e kinetic, 1e nuclear attraction, 2e, 2c2e, 3c1e, 3c2e at `atol=1e-12`) is implemented but stays **opt-in only**:
  - tests gated `#[cfg(feature = "rocm")]`
  - tests marked `#[ignore]` so `cargo test --features rocm` does NOT run them by default
  - opt-in trigger via env-gate (e.g., `CINTX_ROCM_ORACLE=1 cargo test --features rocm -- --ignored`)
  - new `xtask rocm-oracle` helper wraps the trigger
  - **no CI gate** — no AMD/ROCm GitHub runner exists; running on the dev box is operator-driven
  - this is a stronger commitment than ROADMAP's "at least one oracle smoke test" (criterion 4) — capture this as an internal upgrade and preserve the option to expand to a self-hosted CI runner later (see seed `gpu-ci-runners.md`)
- **D-16:** New required CI job `feature_matrix_gate` joins the existing four required gates (`manifest_drift_gate`, `oracle_parity_gate`, `helper_legacy_parity_gate`, `oom_contract_gate`). Required for PR merge; fail-closed; matches the existing CI architecture from Phase 4. Runs the 3-cell matrix from D-13/D-14.

### Claude's Discretion
- Internal organization of the new `Cuda` / `Rocm` / `Metal` arms in `crates/cintx-cubecl/src/backend/` — whether they live in `cuda_backend.rs` / `rocm_backend.rs` / `metal_backend.rs` files alongside `wgpu_backend.rs` and `cpu_backend.rs`, or in some other shape. Pattern-match the existing `wgpu_backend.rs` / `cpu_backend.rs` structure unless research surfaces a reason not to.
- Display/formatting of `BackendNotCompiled` and the `compiled_in` list (compile-time enumerated `const &[&str]` vs runtime build-time-cfg'd `Vec`). Researcher should pick the cleanest constexpr-friendly approach.
- Whether to introduce a `compiled_backends() -> &'static [&'static str]` public helper for callers that want to introspect what's available — or keep that information internal to the error variant. Lean toward exposing it; it's free if `compiled_in` is already a `const`.
- Capability-token rules per backend (Phase 5 D-08 fingerprint contract). Cuda/Rocm/Metal will not have runtime-verifiable capability tokens this phase (compile-only); the planner can use a fixed sentinel fingerprint for them. Researcher should confirm this matches the existing wgpu fingerprint pattern.
- Whether `BackendIntent::selector` grammar gains backend-specific syntax (`cuda:0`, `rocm:0`) or stays as the existing `auto` / `device:N` advisory string. Lean toward keeping it advisory and unchanged — selector parsing is per-backend in `*_backend.rs` already.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase intent and scope
- `.planning/ROADMAP.md` § "Phase 16: Multi-Backend Support …" — locks goal, success criteria 1–7, and explicit risk-accept on cuda/metal verification.
- `.planning/PROJECT.md` § Constraints — CubeCL is the primary compute backend; host CPU work limited to planning/validation/marshaling.
- `.planning/notes/cuda-metal-verification-gap.md` — explicit risk-accept on cuda/metal runtime verification this phase. Plan tasks may include `cargo check`-style matrix verification but **must not** add an oracle parity gate that requires those runtimes.
- `.planning/research/questions.md` § "2026-05-09 — Multi-backend (cuda / rocm / metal) feature wiring" — open research question on cubecl 0.10.0 cross-backend resolution; gsd-phase-researcher must answer it during plan-phase.
- `.planning/seeds/gpu-ci-runners.md` — follow-up that would close the cuda/metal verification gap if hardware/CI becomes available.

### Existing backend infrastructure (codebase)
- `crates/cintx-cubecl/Cargo.toml` — current `[features]` shape; default = ["cpu"]; cpu = ["cubecl/cpu"]; cubecl pinned to 0.10.0 with features = ["wgpu"].
- `crates/cintx-cubecl/src/backend/mod.rs` — `ResolvedBackend` enum; `resolve_backend_kind()` env-var helper (currently infallible, falls back to Wgpu on unknown values — must change per D-03).
- `crates/cintx-cubecl/src/backend/wgpu_backend.rs` and `cpu_backend.rs` — pattern for new per-backend modules.
- `crates/cintx-runtime/src/options.rs` — `BackendKind`, `BackendIntent`, `BackendCapabilityToken`. `BackendIntent::default()` currently `{ Wgpu, "auto" }` — must flip to Cpu per D-11.
- `crates/cintx-runtime/src/planner.rs` and `workspace.rs` — call sites that thread `BackendKind` through query/evaluate; reference for D-12 audit.

### Error and CI infrastructure (prior phases)
- `crates/cintx-core/src/error.rs` (or wherever `cintxRsError` lives) — host of the new `BackendNotCompiled` variant; existing `InvalidEnvParam` (Phase 13) and `UnsupportedApi` (Phase 5+) variants for taxonomy comparison.
- `.github/workflows/` (manifest_drift_gate, oracle_parity_gate, helper_legacy_parity_gate, oom_contract_gate) — reference architecture for the new `feature_matrix_gate`.

### Compatibility contract (project-wide)
- `docs/design/cintx_detailed_design.md` — design source of truth; design D-03 (backend kind = control-plane metadata), D-08 (planning_matches drift detection), D-12 (error taxonomy).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ResolvedBackend::from_intent` (`crates/cintx-cubecl/src/backend/mod.rs`): existing pattern for dispatching to a per-backend `resolve_*_client()`. New backends slot in as additional match arms following the wgpu/cpu shape.
- `bootstrap_wgpu_runtime` and `runtime_bootstrap` module: existing model for capability fingerprint generation and backend bootstrap. Cuda/Rocm/Metal will have their own bootstrap helpers but with simpler (or sentinel) fingerprints since they're compile-only or unverified.
- `BackendIntent` carries `selector` as a free-form advisory string ("auto", "device:0"). Per-backend selector parsing already lives in `wgpu_backend.rs`.
- `InvalidEnvParam` error variant (Phase 13): precedent for typed env-var parsing failures with a recognized-values payload.

### Established Patterns
- **Per-feature `#[cfg]` gating on enum variants** is already the pattern for `ResolvedBackend::Cpu` (under `#[cfg(feature = "cpu")]`). New variants follow the same shape.
- **Feature-forwarding** (with-f12, with-4c1e) already wires top-level safe-API features to inner crate features. New backend features use the same forwarding pattern from cintx-rs / cintx-capi where needed.
- **Capability fingerprint** (Phase 5/6 D-08): `BackendCapabilityToken` is captured at query time and verified at evaluate time. Compile-only backends (cuda/metal) need a sentinel/zero fingerprint; rocm needs a real one when runtime-verified.
- **Fail-closed CI gates** (Phase 4): four required jobs already structured this way. New `feature_matrix_gate` joins as the fifth.

### Integration Points
- `cintxRsError` enum in `cintx-core` — new `BackendNotCompiled` variant must derive `thiserror::Error` consistently with existing variants and surface a stable `Display` format.
- `cintx-capi` C ABI status codes — `CINTX_STATUS_*` constants (Phase 3) need a code allocated for `BackendNotCompiled`.
- `BackendIntent::default()` has many implicit callers — exhaustive grep + audit is a discrete plan task.
- Existing `cargo test` invocations across the workspace assume `BackendIntent::default() == Wgpu`; D-12 audit covers the migration.

</code_context>

<specifics>
## Specific Ideas

- "Wgpu is not default" — explicit user rule. Applies workspace-wide: no implicit wgpu pulls in any crate, no test that silently requires wgpu without `#[cfg(feature = "wgpu")]`.
- ROCm full base-family oracle suite as the smoke target (not just one symbol). Strong personal interest in validating ROCm parity end-to-end on the dev box, even though it's not gated in CI.
- Per-variant cfg style preferred over alternatives that lose compile-time exhaustiveness checks. Verbose match arms accepted.
- Trust upstream cubecl gating; don't duplicate target_os logic at our layer.

</specifics>

<deferred>
## Deferred Ideas

- **GPU CI runners (NVIDIA, Apple Silicon).** Would close the cuda/metal verification gap and would let the rocm oracle suite become a CI-blocking gate. Tracked in `.planning/seeds/gpu-ci-runners.md`. Out of scope for Phase 16; phase 17+ candidate.
- **`CINTX_BACKEND` aliases (e.g., `hip` → `rocm`).** Initial implementation uses strict 1:1 names (`cpu`, `wgpu`, `cuda`, `rocm`, `metal`). If user demand arises, alias support is a small follow-up.
- **Per-backend selector grammar (`cuda:0`, `rocm:0`).** The advisory `selector` string stays unchanged this phase; per-backend extension is a follow-up if device selection becomes important.
- **Backend-introspection public API** (`compiled_backends() -> &'static [&'static str]`). May land alongside `BackendNotCompiled` for symmetry; if not, capture as a small follow-up enhancement.

</deferred>

---

*Phase: 16-multi-backend-support*
*Context gathered: 2026-05-09*
