# Feature Research

**Domain:** Rust quantum-chemistry integral engine library (`libcint`-compatible)
**Researched:** 2026-03-14
**Confidence:** HIGH

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these makes the library non-viable for migration.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Stable-family result compatibility (`1e/2e/2c2e/3c1e/3c2e`; cart/sph/spinor) | Core promise is parity with upstream `libcint` outputs | HIGH | Depends on kernel parity, transform/layout parity, and oracle coverage gates |
| Raw compatibility API (`atm/bas/env`, `shls`, `dims`, `cache`, `opt`) | Existing ecosystems already call this contract directly | HIGH | Must preserve sentinel semantics (`out/cache/dims == NULL`) and `not0` behavior |
| Safe Rust API (`query_workspace`, `evaluate_into`) | Rust users expect type-safe interfaces instead of raw-pointer contracts | HIGH | Requires domain types (`BasisSet`, `OperatorId`, tensor views) and strict validator boundaries |
| OOM-safe stop behavior with memory limits and chunking | Design explicitly requires safe failure, not aborts or partial garbage | HIGH | Requires centralized fallible allocator, workspace estimator, chunk planner |
| Helper/transform/optimizer API parity | Migration needs utility APIs beyond raw integral symbols | HIGH | Includes AO counts, offsets, normalization, c2s transforms, optimizer lifecycle |
| Typed error model and deterministic validation | Library-grade consumers need actionable failures | MEDIUM | `LibcintRsError` categories must map layout, unsupported API, OOM, and backend failures |
| Automated release gates (manifest audit + oracle CI) | Compatibility claims must be mechanically verifiable | HIGH | Requires compiled symbol lock, family coverage checks, tolerance policies |
| CPU reference backend as baseline path | Reliable fallback expected even when GPU/optional paths are unavailable | MEDIUM | Also anchors reproducibility and comparison during phased implementation |

### Differentiators (Competitive Advantage)

Features that set this project apart from a plain wrapper.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Shared planner across CPU + CubeCL GPU with deterministic fallback | One logical API with performance scaling and predictable behavior | HIGH | Dispatch reason is traceable; fallback avoids silent divergence |
| Manifest-driven API source of truth (`compiled_manifest.lock.json`) | Enforces full-coverage accountability across feature profiles | HIGH | Reduces drift between docs/headers/compiled symbols/CI |
| Explicit stability + feature-gate matrix (`stable`, `optional`, `unstable_source`) | Users can adopt optional families without stability confusion | MEDIUM | Includes `with-f12` sph-only policy and `with-4c1e` validated envelope |
| Strict unsafe minimization policy | Improves auditability and long-term maintainability in HPC code | MEDIUM | Constrains unsafe to compat/FFI/SIMD/device boundaries |
| First-class observability (`tracing` planner, chunking, fallback, OOM reasons) | Easier performance tuning and failure diagnosis in production pipelines | MEDIUM | Supports regression detection beyond pass/fail correctness |
| Optional C ABI shim for phased migration | Enables incremental adoption by C-centric stacks without abandoning Rust API quality | MEDIUM | Decouples migration timeline from full safe-API adoption |

### Anti-Features (Commonly Requested, Often Problematic)

Features that look attractive but would damage scope, reliability, or delivery speed.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Bitwise-identical reproduction of libcint internals | Feels like the strongest compatibility claim | Conflicts with documented compatibility model and over-constrains implementation choices | Keep numerical-result compatibility with family-specific tolerances and oracle gates |
| Public async API in initial GA | Users associate async with scalability | Adds runtime model complexity without core compatibility benefit; design marks as out of scope | Keep synchronous public API; optimize internals and revisit async after parity/perf stabilization |
| Exposing GTG as stable GA feature | Completeness pressure from API inventory perspective | Upstream marks GTG as deprecated/incorrect; high risk of unstable semantics | Keep GTG out of GA surface; revisit only with independent implementation + full verification |
| GPU-first dispatch for all workloads | “GPU should always be faster” expectation | Small batches often lose to transfer/launch overhead and hurt latency | Use conservative auto-dispatch with explicit CPU-preferred thresholds |
| Best-effort acceptance of malformed raw layouts | Legacy callers may send questionable inputs | Silent coercion causes undefined behavior and hard-to-debug numeric corruption | Strict validation + typed errors + migration guidance |

## Feature Dependencies

```text
[Stable-family Oracle Compatibility]
    ├──requires──> [Compiled Manifest Lock + Symbol Audit]
    ├──requires──> [Raw Validator + Output Layout Writer Parity]
    └──requires──> [CPU Kernel/Transform Coverage]
                       └──requires──> [Workspace Estimator + Chunk Planner]

[Safe Rust API]
    └──requires──> [Domain Types + Validation Layer]

[GPU Backend Acceleration]
    ├──enhances──> [Stable-family Oracle Compatibility]
    └──requires──> [Shared Planner + Transfer Heuristics + Device Cache]

[Optional Families: F12/4c1e/unstable_source]
    └──requires──> [Feature Gate Matrix + Dedicated CI Profiles]

[OOM-safe Stop]
    ├──requires──> [WorkspaceAllocator + FallibleBuffer]
    └──conflicts──> [Unbounded Internal Vec Allocation Paths]
```

### Dependency Notes

- **Stable-family Oracle compatibility requires compiled-manifest locking:** Without symbol lock + audit, coverage claims cannot be trusted across build profiles.
- **Raw validator/layout parity is a hard prerequisite for compatibility:** Numerical parity is meaningless if `dims/shls` semantics or output ordering diverge.
- **Workspace estimator/chunk planner underpins both OOM safety and large-batch throughput:** This is a shared high-complexity dependency, not an optional optimization.
- **GPU acceleration depends on shared planner decisions:** Backend choice must remain deterministic and traceable to avoid correctness/perf regressions.
- **Optional-family support depends on feature-profile CI isolation:** Complexity increases non-linearly with each optional family; separate coverage is required to keep failures local.
- **OOM-safe stop conflicts with ad-hoc allocations:** Any bypass around centralized fallible allocation violates a core non-functional requirement.

## MVP Definition

### Launch With (v1)

Minimum viable product to validate compatibility and migration value.

- [ ] Stable-family result compatibility with oracle comparison gates
- [ ] Raw compatibility API including `out/cache/dims == NULL` contract behavior
- [ ] Safe Rust API (`query_workspace` + `evaluate_into`) with typed inputs/errors
- [ ] OOM-safe stop policy with memory-limit chunking
- [ ] Helper/transform essentials required by migration workflows
- [ ] Manifest audit + regression CI as release gate

### Add After Validation (v1.x)

Features to add once core compatibility is proven stable.

- [ ] Optional C ABI shim (`feature = "capi"`) for wider phased migration
- [ ] Expanded GPU coverage for large homogeneous families with conservative auto-dispatch
- [ ] Optional family rollout (`with-f12`, `with-4c1e`) under strict support matrix rules

### Future Consideration (v2+)

Features to defer until parity/performance baseline is reliable.

- [ ] Public async facade variants (only if real user demand and measured benefit justify complexity)
- [ ] Promotion of selected `unstable_source` families to stable status after multi-release evidence
- [ ] GTG reconsideration only with independent implementation and full oracle/property identity evidence

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Stable-family result compatibility + oracle gate | HIGH | HIGH | P1 |
| Raw compatibility API parity | HIGH | HIGH | P1 |
| Safe Rust API with typed validation/errors | HIGH | HIGH | P1 |
| OOM-safe stop + memory-limit chunking | HIGH | HIGH | P1 |
| Helper/transform/optimizer parity baseline | HIGH | MEDIUM | P1 |
| Shared planner CPU/GPU auto-dispatch | MEDIUM | HIGH | P2 |
| Optional C ABI shim | MEDIUM | MEDIUM | P2 |
| Optional families (`with-f12`, `with-4c1e`) | MEDIUM | HIGH | P2 |
| Broad `unstable_source` promotion | LOW | HIGH | P3 |
| Async public API | LOW | HIGH | P3 |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

## Competitor Feature Analysis

| Feature | Competitor A | Competitor B | Our Approach |
|---------|--------------|--------------|--------------|
| Raw API breadth | Upstream `libcint` C API is broad and mature | Many wrappers expose raw calls but not full policy guarantees | Match raw contracts while adding typed validation and Rust errors |
| Rust-native safety | Upstream API is C-pointer-first | Typical wrappers rely on unsafe FFI boundaries in user code | Provide first-class safe Rust API with explicit domain types |
| Optional-family governance | Upstream uses compile-time options with mixed stability | Wrappers often inherit options without explicit stability messaging | Encode stability/feature gates in manifest and support matrix |
| OOM behavior guarantees | C-side behavior depends on caller/build/runtime | Wrapper behavior varies, often not centrally enforced | Central fallible allocation + typed stop semantics |
| CPU/GPU unified dispatch | Upstream is CPU reference | Some ecosystems split CPU/GPU APIs | Single logical API with shared planner and deterministic fallback |
| Compatibility verification | Upstream tests itself as reference | Wrappers may lack full manifest-coverage auditing | Enforce release gates: symbol audit + oracle coverage + profile matrix |

## Sources

- `/home/chemtech/workspace/cintx/.planning/PROJECT.md`
- `/home/chemtech/workspace/cintx/docs/libcint_detailed_design_resolved_en.md`
- `/home/chemtech/workspace/cintx/.planning/codebase/ARCHITECTURE.md`
- `/home/chemtech/.codex/get-shit-done/templates/research-project/FEATURES.md`

---
*Feature research for: libcint-rs*
*Researched: 2026-03-14*
