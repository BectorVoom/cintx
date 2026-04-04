# Phase 5: Re-implement detailed-design GPU path with CubeCL (wgpu backend) - Research

**Researched:** 2026-03-29
**Domain:** CubeCL wgpu runtime integration, fail-closed backend capability gating, and removal of pseudo execution paths
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

### Runtime backend policy
- **D-01:** Backend selection auto-selects among wgpu-capable adapters, then fails closed when no valid adapter/capability is available; no CPU substitute compute fallback.
- **D-02:** Capability gaps (for example required shader/device features) must return explicit typed failures rather than silent runtime substitution.
- **D-03:** Backend/device intent is control-plane metadata carried via runtime options/plumbing (not hidden executor-only policy).
- **D-04:** Runtime diagnostics and verification artifacts must include backend/adapter capability context for reproducibility.

### Planner/dispatch/execution integration strictness
- **D-05:** Phase cutline is end-to-end real CubeCL path; placeholder/synthetic execution behavior in compute path must be removed.
- **D-06:** Preserve strict ownership contract: backend output remains staging-only, compat retains final caller-visible flat writes.
- **D-07:** Chunking remains CPU control-plane only, but each chunk still executes through CubeCL compute path.
- **D-08:** Query/evaluate backend policy contract is locked; policy drift between query and evaluate must fail with typed errors.

### Unsupported-scope policy
- **D-09:** Out-of-envelope or unimplemented requests fail explicitly with typed unsupported/capability errors; no hidden fallback masking.
- **D-10:** Unsupported scope must be visible both at runtime and in artifactized reporting (matrix/report format) for verification audits.
- **D-11:** Validated4C1E policy remains strict-envelope, but backend requirement shifts from cpu-profile gate to explicit wgpu capability gating.
- **D-12:** Unimplemented family/representation paths must return specific unsupported reason taxonomy, not generic errors.

### Validation and regression gates
- **D-13:** Verification must be layered across runtime + cubecl + compat (not single-layer crate-local tests only).
- **D-14:** CI uses capability-aware required gates for wgpu regression checks (explicit skip metadata only when capability truly absent).
- **D-15:** Add explicit anti-pseudo regression assertions so synthetic execution substitutions cannot silently return.
- **D-16:** Unsupported behavior tests must assert both reason taxonomy and reporting artifact presence.

### Claude's Discretion
- Concrete Rust type/field names for backend-selection control-plane metadata and diagnostics payloads.
- Exact test and artifact file naming as long as D-13 through D-16 are satisfied.
- Exact location of helper functions used to preflight device capabilities.

### Deferred Ideas (OUT OF SCOPE)
None - discussion stayed within Phase 5 scope.
</user_constraints>

## Summary

Phase 5 should be planned as a hard replacement of synthetic execution with a real CubeCL wgpu runtime path. The current codebase has two explicit pseudo paths: `crates/cintx-cubecl/src/executor.rs` fills deterministic staging values (`fill_cartesian_staging`) after kernel "launch" wrappers that only allocate buffers, and `crates/cintx-rs/src/api.rs` has its own local stub `CubeClExecutor` that also fabricates values (`fill_staging_values`). This directly conflicts with D-05 and creates regression risk if not removed end-to-end.

The existing runtime contracts are strong and should be preserved, not redesigned: planner dispatch ownership (`BackendStagingOnly -> CompatFinalWrite`), query/evaluate drift checks, deterministic chunk scheduling, and typed unsupported/memory failures already exist and map well to D-06 through D-09. The missing piece is backend capability plumbing as first-class control-plane metadata (D-03/D-08), plus typed fail-closed adapter/device preflight that avoids hidden fallback and avoids panic-like behavior leaking from lower layers.

CubeCL 0.9.0 and cubecl-wgpu 0.9.0 provide the right primitives now: `WgpuDevice` selection modes, `init_setup`/`init_setup_async`, `init_device`, `RuntimeOptions`, and adapter-level feature/limit introspection from `wgpu`. Planning should focus on introducing a typed backend-intent/capability contract shared by query and evaluate, then replacing synthetic kernel outputs with actual CubeCL dataflow while preserving current compat final-write ownership.

**Primary recommendation:** Implement a typed "backend intent + capability snapshot" contract at query time, enforce token match at evaluate time, and route every chunk through real CubeCL wgpu execution while keeping compat final-write ownership unchanged.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cubecl` | `0.9.0` (stable), `0.10.0-pre.2` available | Core CubeCL API, runtime trait, backend re-exports | `0.9.0` is current stable/default on crates.io; pre-release exists but adds migration risk during a correctness phase. |
| `cubecl-wgpu` | `0.9.0` (stable), `0.10.0-pre.2` available | WGPU runtime integration (`WgpuDevice`, `init_setup`, `init_device`) | Official backend package used by CubeCL wgpu path and exposes adapter/runtime setup controls required by D-01..D-04. |
| `cubecl-runtime` | `0.9.0` (stable), `0.10.0-pre.2` available | `Runtime` trait + `ComputeClient` lifecycle | Defines runtime contract used by wgpu backend and query/evaluate execution boundary. |
| `tracing` | `0.1.44` | Structured backend/capability diagnostics | Required to satisfy artifactized reproducibility and runtime gate visibility. |
| `thiserror` | `2.0.18` | Typed public errors for fail-closed behavior | Matches project error policy and D-02/D-09 typed-failure requirements. |

Version evidence (registry):
- Stable `0.9.0` publish date: `cubecl` `2026-01-15`, `cubecl-wgpu` `2026-01-15`, `cubecl-runtime` `2026-01-15`.
- Latest pre-release `0.10.0-pre.2` publish date: `2026-03-02` for all three crates.

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `wgpu` (transitive via `cubecl-wgpu`) | lockfile `26.0.1` | Adapter info/features/limits for capability gating and diagnostics | Use during capability preflight and artifact emission; keep version aligned with `cubecl-wgpu` line. |
| `smallvec` | `1.x` | Compact metadata vectors for control-plane and specialization keys | Existing project usage for lightweight staging/planning metadata. |
| `anyhow` | `1.0.102` | Tooling/xtask/reporting boundary errors | Use in CI/report tooling only, not public library surface. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Stay on stable `0.9.0` | Move to `0.10.0-pre.2` now | Newer API line may help long-term, but increases migration variables during a correctness-sensitive phase. |
| CubeCL wgpu runtime APIs | Direct raw `wgpu` runtime wiring | More control, but duplicates logic CubeCL already implements and increases maintenance burden. |
| Fail-closed unsupported errors | Hidden CPU fallback compute | Violates locked decisions D-01/D-02/D-09 and masks correctness envelope. |

**Installation (if missing in crate wiring):**
```bash
cargo add --package cintx-cubecl cubecl@0.9.0
cargo add --package cintx-cubecl cubecl-wgpu@0.9.0
```

**Version verification (registry API):**
```bash
node -e 'const https=require("https");const crates=["cubecl","cubecl-wgpu","cubecl-runtime"];function get(c){return new Promise((res,rej)=>https.get("https://crates.io/api/v1/crates/"+c,{headers:{"User-Agent":"cintx-research-agent"}},r=>{let d="";r.on("data",x=>d+=x);r.on("end",()=>{const j=JSON.parse(d);const stable=j.versions.find(v=>v.num==="0.9.0");console.log(c,j.crate.default_version,j.crate.max_version,stable?.created_at);res();});}).on("error",rej));}(async()=>{for(const c of crates)await get(c);})();'
```

## Architecture Patterns

### Recommended Project Structure
```text
crates/cintx-cubecl/src/
├── executor.rs             # backend contract + chunk execution orchestration
├── runtime_bootstrap.rs    # NEW: adapter selection, setup/init, capability snapshot
├── capability.rs           # NEW: typed capability model + diagnostics serialization
├── transfer.rs             # staging metadata/buffers, ownership enforcement
└── kernels/
    ├── mod.rs              # family registry and representation gating
    └── *.rs                # real CubeCL launch paths (remove pseudo-only stats flow)

crates/cintx-runtime/src/
├── options.rs              # extend with typed backend intent metadata
└── planner.rs              # enforce query/evaluate backend token drift checks

crates/cintx-compat/src/raw.rs
└── policy gates + final write ownership (unchanged ownership, updated capability reasons)
```

### Pattern 1: Explicit WGPU Bootstrap + Capability Snapshot
**What:** Resolve adapter/device once per request policy, capture adapter info/features/limits, and carry snapshot through execution.
**When to use:** At query time (and verify unchanged at evaluate time).
**Example:**
```rust
// Source: cubecl-wgpu 0.9.0 runtime/device APIs
use cubecl_wgpu::{AutoGraphicsApi, RuntimeOptions, WgpuDevice, WgpuRuntime, init_device, init_setup};
use cubecl_runtime::client::ComputeClient;

let requested = WgpuDevice::DefaultDevice;
let setup = init_setup::<AutoGraphicsApi>(&requested, RuntimeOptions::default());
let runtime_device = init_device(setup.clone(), RuntimeOptions::default());
let client: ComputeClient<WgpuRuntime> = ComputeClient::load(&runtime_device);

let adapter_info = setup.adapter.get_info();
let adapter_features = setup.adapter.features();
let adapter_limits = setup.adapter.limits();
let runtime_name = WgpuRuntime::name(&client);
```

### Pattern 2: Backend Intent as Query/Evaluate Contract Data
**What:** Add typed backend intent/capability token to `ExecutionOptions` and workspace token; reject drift at evaluate.
**When to use:** Every `query_workspace` -> `evaluate` pair.
**Example:**
```rust
// Source: existing query/evaluate drift checks in runtime + safe facade token pattern
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendIntent {
    pub backend: &'static str,    // "wgpu"
    pub device_selector: String,  // "DefaultDevice" / "DiscreteGpu(0)" / ...
    pub capability_hash: u64,     // hash(features + limits + adapter info)
}

// enforce during evaluate:
if query_token.backend_intent != eval_opts.backend_intent {
    return Err(cintxRsError::ChunkPlanFailed { from: "evaluate", detail: "backend intent drift".into() });
}
```

### Pattern 3: Real Kernel Path Per Chunk, Compat Owns Final Write
**What:** Keep runtime chunk scheduler and ownership contract, but replace synthetic staging fills with real CubeCL kernel outputs.
**When to use:** Backend `execute()` implementation for each chunk.
**Example:**
```rust
// Source: cintx-runtime ownership contract + cintx-cubecl transfer plan
io.ensure_output_contract()?;
let transfer = transfer_plan.stage_device_buffers("wgpu")?;
let stats = kernels::launch_family(plan, &specialization, &transfer_plan)?;
io.ensure_output_contract()?;
// no fill_cartesian_staging / no synthetic value generation
```

### Pattern 4: Capability-Aware Unsupported Taxonomy
**What:** Report unsupported by explicit reason class (missing feature, limit, family/rep unsupported, envelope violation).
**When to use:** Before launch and in failed launch/setup handling.
**Example:**
```rust
// Source: existing UnsupportedApi policy style + D-12
enum CapabilityReason {
    MissingAdapter,
    MissingFeature(&'static str),
    LimitTooLow(&'static str, u64, u64),
    FamilyUnsupported(&'static str),
    RepresentationUnsupported(&'static str),
}
```

### Anti-Patterns to Avoid
- **`CUBECL_RUNTIME_PROFILE = "cpu"` as policy gate:** This is now explicitly wrong for Phase 5 and conflicts with D-01/D-11.
- **Synthetic staging fills in backend or safe facade:** `fill_cartesian_staging` and `fill_staging_values` must not survive Phase 5.
- **Uncaught panic paths from backend setup:** `cubecl-wgpu` setup paths use `expect`/`panic!` in adapter/device selection; plan must map failures into typed errors.
- **Policy hidden only in executor internals:** Backend intent/capability must be visible in control-plane metadata and drift checks.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Adapter/device selection heuristics | Custom GPU picker from scratch | `WgpuDevice` + `init_setup`/`init_setup_async` + env override support | CubeCL already models device classes and setup lifecycle for wgpu backend. |
| Runtime client lifecycle | Custom global singleton map | `ComputeClient::init` / `ComputeClient::load` | Standard CubeCL runtime contract avoids duplicate state management bugs. |
| Capability inspection schema | Ad-hoc string parsing of logs | `adapter.get_info()`, `adapter.features()`, `adapter.limits()` typed fields | Gives deterministic, auditable capability metadata for D-04/D-10. |
| Output flattening contract | Backend writes directly into compat flat layout | Keep `BackendStagingOnly -> CompatFinalWrite` | Existing ownership contract is already validated and prevents partial-write regressions. |

**Key insight:** The phase is primarily integration hardening and pseudo-path removal, not invention of new scheduler/layout contracts.

## Common Pitfalls

### Pitfall 1: Backend Setup Panics Instead of Returning Typed Errors
**What goes wrong:** Missing adapters or invalid selection can panic inside CubeCL setup path.
**Why it happens:** `cubecl-wgpu` setup uses `expect`/`panic!` for several adapter/device selection failures.
**How to avoid:** Add preflight capability checks before setup, and wrap setup boundaries to map failures into `cintxRsError` taxonomy.
**Warning signs:** Process aborts or panic messages like "No possible adapter available..." instead of typed `UnsupportedApi`/`BackendFailure`.

### Pitfall 2: Silent Regression Back to Synthetic Compute
**What goes wrong:** Kernels "execute" but data path still uses deterministic fake staging values.
**Why it happens:** Existing executor and safe facade currently fabricate output values.
**How to avoid:** Add anti-pseudo regression tests that fail if output equals synthetic sequences for known fixtures.
**Warning signs:** Output values follow trivial monotonic patterns independent of basis/operator.

### Pitfall 3: Query/Evaluate Drift on Backend Intent
**What goes wrong:** Query runs under one device/capability snapshot and evaluate runs under another.
**Why it happens:** Current drift checks only track memory/chunk options, not backend metadata.
**How to avoid:** Include backend intent/capability token in planning contract and enforce strict match in both runtime and safe facade.
**Warning signs:** Flaky device-dependent behavior across repeated runs with same input.

### Pitfall 4: Incorrect Unsupported Reason Taxonomy
**What goes wrong:** Out-of-envelope and missing-capability cases collapse into generic errors.
**Why it happens:** Convenience shortcuts in error mapping.
**How to avoid:** Keep distinct reason classes for family/rep support, envelope policy, and device capability.
**Warning signs:** `UnsupportedApi` messages without capability or envelope detail.

### Pitfall 5: Feature/Backend Drift in Build Matrix
**What goes wrong:** CI builds accidentally test non-wgpu paths (or include extra backends implicitly) and miss true phase behavior.
**Why it happens:** CubeCL default features include multiple backend lines.
**How to avoid:** Make backend expectations explicit in CI/profile configuration and artifact metadata.
**Warning signs:** Required jobs pass without any wgpu capability evidence in logs/artifacts.

## Code Examples

Verified patterns from official/local sources:

### WGPU Runtime Initialization (CubeCL 0.9.0)
```rust
// Source: cubecl-wgpu 0.9.0 src/runtime.rs, src/device.rs
use cubecl_runtime::client::ComputeClient;
use cubecl_wgpu::{AutoGraphicsApi, RuntimeOptions, WgpuDevice, WgpuRuntime, init_device, init_setup};

let selector = WgpuDevice::DefaultDevice;
let setup = init_setup::<AutoGraphicsApi>(&selector, RuntimeOptions::default());
let device = init_device(setup.clone(), RuntimeOptions::default());
let client: ComputeClient<WgpuRuntime> = ComputeClient::load(&device);

let info = setup.adapter.get_info();
let features = setup.adapter.features();
let limits = setup.adapter.limits();
let runtime = WgpuRuntime::name(&client);
```

### Runtime Contract Enforcement for Query/Evaluate
```rust
// Source: crates/cintx-runtime/src/planner.rs and crates/cintx-rs/src/api.rs
if !query_workspace.planning_matches(&options) {
    return Err(cintxRsError::ChunkPlanFailed {
        from: "evaluate",
        detail: "execution options do not match the query_workspace contract".into(),
    });
}
```

### CubeCL Kernel Launch Pattern (manual baseline)
```rust
// Source: docs/manual/Cubecl/Cubecl_vector.md
#[cube(launch)]
fn array_multiply_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let tid = ABSOLUTE_POS;
    if tid < input.len() && tid < output.len() {
        output[tid] = input[tid] * ((tid + 2) as f32);
    }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Executor profile hardcoded to `"cpu"` for Validated4C1E and runtime policy | Explicit wgpu-capability gating with typed reasons (planned for this phase) | Phase 5 scope (2026-03-29 context) | Aligns with fail-closed backend policy and removes hidden substitution paths. |
| Synthetic staging fill in backend/safe facade (`fill_cartesian_staging`, `fill_staging_values`) | Real CubeCL kernel output path per chunk | Phase 5 implementation target | Prevents pseudo-success and restores meaningful oracle compatibility checks. |
| Backend policy mostly string/profile label metadata | Typed backend intent + capability snapshot token across query/evaluate | Phase 5 planning target | Makes drift detection and reproducibility diagnostics deterministic. |
| Single stable line (`0.9.0`) with newer pre-release available | Keep stable for correctness phase; evaluate pre-release later | `0.10.0-pre.2` published 2026-03-02 | Reduces migration risk while finishing core runtime correctness. |

**Deprecated/outdated:**
- `CUBECL_RUNTIME_PROFILE = "cpu"` gating in executor/compat policy for Phase 5 behavior.
- Any evaluation path that can succeed without real CubeCL data production.

## Open Questions

1. **Should Phase 5 stay on CubeCL stable `0.9.0` or adopt `0.10.0-pre.2` during implementation?**
   - What we know: Stable default is `0.9.0`; pre-release is available and newer.
   - What's unclear: Migration surface and behavior differences relevant to wgpu gating.
   - Recommendation: Deliver Phase 5 on `0.9.0`, then schedule a separate upgrade phase.

2. **Where should backend intent live in the public/runtime boundary?**
   - What we know: D-03 requires control-plane metadata, and runtime already has query/evaluate token checks.
   - What's unclear: Whether to store intent directly in `ExecutionOptions`, `WorkspaceQuery`, or both.
   - Recommendation: Put authoritative intent in `ExecutionOptions` and copy hash/token into `WorkspaceQuery`/safe `WorkspaceExecutionToken` for drift enforcement.

3. **How should capability-absent environments be represented in required CI gates?**
   - What we know: D-14 requires capability-aware required gates with explicit skip metadata.
   - What's unclear: Exact skip contract payload and required runner labeling policy.
   - Recommendation: Define a typed "capability unavailable" artifact schema and require it when skipping wgpu-required checks.

## Sources

### Primary (HIGH confidence)
- Phase constraints and scope:
  - `.planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/05-CONTEXT.md`
  - `.planning/REQUIREMENTS.md`
  - `.planning/STATE.md`
  - `.planning/config.json`
- Local implementation evidence:
  - `crates/cintx-cubecl/src/executor.rs`
  - `crates/cintx-cubecl/src/transfer.rs`
  - `crates/cintx-cubecl/src/kernels/mod.rs`
  - `crates/cintx-cubecl/src/kernels/one_electron.rs`
  - `crates/cintx-rs/src/api.rs`
  - `crates/cintx-runtime/src/planner.rs`
  - `crates/cintx-runtime/src/dispatch.rs`
  - `crates/cintx-runtime/src/workspace.rs`
  - `crates/cintx-compat/src/raw.rs`
  - `docs/design/cintx_detailed_design.md`
  - `docs/manual/Cubecl/Cubecl_vector.md`
  - `docs/manual/Cubecl/cubecl_matmul_gemm_example.md`
  - `docs/manual/Cubecl/cubecl_reduce_sum.md`
  - `docs/manual/Cubecl/cubecl_error_solution_guide/mismatched types.md`
  - `docs/cubecl_error_guideline.md`
- Registry/source of truth:
  - https://crates.io/crates/cubecl
  - https://crates.io/crates/cubecl-wgpu
  - https://crates.io/crates/cubecl-runtime
  - `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-wgpu-0.9.0/src/runtime.rs`
  - `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-wgpu-0.9.0/src/device.rs`
  - `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-runtime-0.9.0/src/runtime.rs`
  - `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/wgpu-26.0.1/src/api/adapter.rs`
  - `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/wgpu-types-26.0.0/src/lib.rs`

### Secondary (MEDIUM confidence)
- API index references:
  - https://docs.rs/cubecl-wgpu/latest/cubecl_wgpu/
  - https://docs.rs/cubecl-wgpu/latest/cubecl_wgpu/fn.init_setup_async.html

### Tertiary (LOW confidence)
- None.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - versions and publish dates verified from crates.io API and local lockfile.
- Architecture: HIGH - recommendations map directly to locked decisions and current code contracts.
- Pitfalls: HIGH - each pitfall is backed by concrete local code paths and upstream runtime behavior.

**Research date:** 2026-03-29
**Valid until:** 2026-04-05
