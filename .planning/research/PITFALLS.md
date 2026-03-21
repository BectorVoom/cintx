# Pitfalls Research

**Domain:** Rust reimplementation of a quantum chemistry integral engine (libcint 6.1.3 compatibility)
**Researched:** 2026-03-14
**Confidence:** HIGH

## Critical Pitfalls

### Pitfall 1: Manifest/ABI Drift Hidden by Header-Only Checks

**What goes wrong:**
The team claims "full API coverage" while compiled symbol reality differs from headers/source assumptions, causing missing symbols, wrong feature exposure, or silent compatibility holes.

**Why it happens:**
Developers audit `include/*` and source declarations but skip compiled-symbol union checks across feature profiles.

**How to avoid:**
Treat `compiled_manifest.lock.json` as release truth, regenerate from `{base, with-f12, with-4c1e, with-f12+with-4c1e}`, and fail CI on any unapproved lock diff.

**Warning signs:**
`nm -D` output differs from manifest; oracle jobs skip newly surfaced families; PRs modify feature flags without lock updates.

**Phase to address:**
Investigation and Release Preparation (`manifest-audit` gate ownership).

---

### Pitfall 2: `dims` Contract Drift and Partial-Write Behavior

**What goes wrong:**
Compat/C ABI writes truncated or wrongly strided output when `dims != NULL`, producing incorrect tensors or memory corruption-like symptoms.

**Why it happens:**
`dims` override logic is reimplemented ad hoc per family instead of using one canonical formula; teams mistakenly allow partial writes.

**How to avoid:**
Centralize `required_elems_from_dims()` and enforce strict length/value checks (`InvalidDims`, `BufferTooSmall`); forbid partial writes and implicit truncation.

**Warning signs:**
`dims`-enabled tests pass only for select families; mismatch between `query_workspace()` and `evaluate()` bytes; flaky failures around non-natural dims.

**Phase to address:**
Implementation 1 (compat helpers/validator) and Testing.

---

### Pitfall 3: Spinor/Complex Layout Mismatch

**What goes wrong:**
Numerically correct kernels still fail compatibility because layout is wrong (`[Re, Im]` interleave, component axis order, spinor length rules by `kappa`).

**Why it happens:**
Teams optimize kernel math first and postpone strict writer/layout parity with libcint flat buffer conventions.

**How to avoid:**
Define one tested layout writer for all families; enforce explicit `cart/sph/spinor` shape contracts; validate against oracle fixtures including spinor-heavy cases.

**Warning signs:**
High relative errors only on spinor/complex outputs; safe API tensors look plausible but compat flat-buffer comparisons fail; off-by-2 element mismatches.

**Phase to address:**
Implementation 2 and Implementation 3.

---

### Pitfall 4: OOM-Safe Requirement Broken by Native `Vec` Paths

**What goes wrong:**
Large runs abort the process instead of returning typed errors, violating a core project requirement and making service embedding unsafe.

**Why it happens:**
Convenience allocations (`vec![0; n]`, `Vec::with_capacity`) slip into hot paths outside the fallible allocator boundary.

**How to avoid:**
Route all >=1 KiB internal allocations through `WorkspaceAllocator`; enforce lint/review checks for forbidden allocation patterns; require chunking before allocation.

**Warning signs:**
OOM tests are missing or xfailed; memory-limit edge cases terminate test process; allocation traces bypass `ExecutionPlan`.

**Phase to address:**
Design, Implementation 1, and Testing (OOM/resource-pressure suite).

---

### Pitfall 5: Optional Family Leakage (4c1e/F12/GTG Boundaries)

**What goes wrong:**
Unsupported representations/families are accidentally exposed or executed (for example 4c1e outside `Validated4C1E`, F12 cart/spinor assumptions, GTG exposure).

**Why it happens:**
Feature-gate policy is documented but not encoded in resolver/planner/runtime guards.

**How to avoid:**
Encode family-policy matrix in the manifest/resolver; reject out-of-envelope calls with `UnsupportedApi`; add explicit "symbol absence" assertions for excluded families.

**Warning signs:**
Unexpected symbols appear in public exports; runtime attempts GPU 4c1e outside allowed envelope; consumers assume GTG exists due accidental docs/autocomplete.

**Phase to address:**
Implementation 5 and Release Preparation.

---

### Pitfall 6: Optimizer On/Off Numerical Drift

**What goes wrong:**
Results differ with optimizer enabled, violating libcint parity expectations and breaking reproducibility.

**Why it happens:**
Optimizer caches mutate internal state or alter numerical pathways beyond acceptable tolerance.

**How to avoid:**
Keep `OptimizerHandle` immutable and thread-safe; add mandatory optimizer on/off parity tests per family/representation.

**Warning signs:**
Diff reports cluster around optimizer-enabled runs; nondeterministic failures under parallel load; cache invalidation bugs tied to basis changes.

**Phase to address:**
Implementation 3 and Testing.

---

### Pitfall 7: Transfer-Dominated GPU Dispatch

**What goes wrong:**
GPU path regresses performance on realistic workloads due launch/H2D/D2H overhead, yet still gets selected by simplistic heuristics.

**Why it happens:**
Dispatch thresholds are static or tuned on synthetic microbenchmarks, not production-like batch profiles.

**How to avoid:**
Keep conservative default dispatch; instrument decision reasons with `tracing`; continuously calibrate crossover thresholds from macrobench and CI telemetry.

**Warning signs:**
GPU jobs slower than CPU for `batch_size < 8` or small outputs; high fallback rate; benchmark variance spikes after threshold tweaks.

**Phase to address:**
Implementation 4 and Benchmarking.

---

### Pitfall 8: Non-Reproducible Oracle Comparisons

**What goes wrong:**
Parity CI becomes noisy or contradictory across machines due toolchain/runtime drift, masking real regressions.

**Why it happens:**
Oracle build/runtime dependencies are not pinned or fixture/seed metadata is incomplete.

**How to avoid:**
Use vendored oracle build, pinned Rust/C toolchains, fixed seeds, and artifactized manifest/fixture metadata in CI.

**Warning signs:**
Same commit passes/fails across runners; unexplained tolerance failures around near-zero values; changing symbol coverage without code changes.

**Phase to address:**
Testing, Difference Analysis, and Release Preparation.

---

### Pitfall 9: C ABI Error Contract Misimplementation

**What goes wrong:**
C consumers receive ambiguous failures, stale errors, or process exits instead of reliable status + thread-local diagnostics.

**Why it happens:**
Upstream libcint patterns are copied directly without adapting to the Rust error model (`last_error` TLS and typed failure mapping).

**How to avoid:**
Standardize C ABI status codes, enforce thread-local `last_error`, and verify no hard-exit behavior in shim paths.

**Warning signs:**
`cintrs_last_error_message()` races or reports old messages; failures return success-like status; integration tests require process restarts.

**Phase to address:**
Implementation 1, Testing, and Release Preparation.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Skip compiled-symbol lock update and rely on header diffs | Faster PR turnaround | Hidden ABI drift and false coverage claims | Never |
| Implement per-family `dims` logic inline | Quick local progress | Inconsistent behavior and repeated bugs | Never |
| Add direct `Vec` allocations in backend paths | Simple coding model | OOM abort risk and violated requirements | Never |
| Treat optional families as "best effort" without explicit errors | Fewer code branches initially | Ambiguous API behavior and support confusion | Never |
| Tune GPU thresholds once and freeze | Fast initial benchmarking | Performance decay as workload mix changes | Only until first macrobench baseline is established |
| Skip helper API parity tests to focus on integral kernels | Smaller early test matrix | Hidden migration breaks in helper/legacy surfaces | Only during pre-merge local spikes, not in main CI |

## Integration Gotchas

Common mistakes when connecting to external services.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Vendored libcint oracle build (`cc`/`bindgen`) | Use host-global libs/headers implicitly | Build hermetically from vendored sources and pin toolchain versions |
| CubeCL GPU runtime | Assume GPU path is always available and faster | Capability-check and fallback to CPU with explicit reason tracing |
| C ABI consumers | Expose Rust internals or panic paths through FFI | Keep stable C ABI status/`last_error` boundary and no panic across FFI |
| Feature matrix CI | Test only base profile | Run matrix including `with-f12`, `with-4c1e`, and combined profile |
| Upstream testsuite reuse | Rely on relative `.so` paths in ad hoc scripts | Use controlled artifact paths and provenance checks in CI |

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Eager full-batch allocation | Large jobs fail or thrash memory | Estimate workspace early and chunk by memory limit | Large basis/high-derivative workloads; multi-GB intermediates |
| Overusing GPU for tiny problems | GPU slower than CPU, high jitter | Conservative dispatch (`batch_size`, output-size, family gating) | Often below `batch_size < 8` or output `< 64 Ki` elements |
| Recomputing transforms per chunk | CPU time dominated by transform overhead | Late materialization + reusable transform caches | Repeated same-basis runs and high-l transforms |
| Copy-heavy compat writer path | High RSS and low throughput | Direct write into caller buffers when possible | Large tensor outputs and repeated chunk writes |
| No optimizer cache invalidation discipline | Fast path becomes wrong path | Hash-based invalidation by basis/operator params | Dynamic workloads switching basis/operators |

## Security Mistakes

Domain-specific security issues beyond general web security.

| Mistake | Risk | Prevention |
|---------|------|------------|
| Loading local shared libs from weak/relative test paths | Artifact spoofing in CI/local runs | Resolve trusted absolute artifacts and verify provenance |
| Allowing hard-exit behavior in library execution paths | Denial of service for host process | Convert all failures to typed errors/status codes |
| Accepting unchecked raw pointers/lengths in compat/capi | Memory corruption or undefined behavior | Strict validator on slot widths, offsets, dims, and buffer sizes |
| Treating unsupported families as soft success | Silent misuse and downstream trust failure | Return explicit `UnsupportedApi`/`NotImplementedByFeature` |

## UX Pitfalls

Common user experience mistakes in this domain.

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Feature support not clearly surfaced | Users call unavailable families and waste debugging time | Publish manifest-derived support matrix in docs and errors |
| Error messages omit shell tuple/dims/backend context | Users cannot reproduce failures | Include structured failure context in public/C ABI reports |
| Safe API behavior diverges from compat semantics silently | Migration confusion and mismatched results | Document and test safe-vs-compat equivalence on shared fixtures |
| Hidden fallback decisions (GPU->CPU) | Performance appears random to users | Emit dispatch/fallback reasons through `tracing` and metrics |

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **API coverage:** Often missing compiled-symbol union checks - verify lock matches all supported profiles.
- [ ] **Compat writer:** Often missing strict `dims` behavior - verify no partial writes and exact required-bytes logic.
- [ ] **OOM safety:** Often missing forbidden allocation audits - verify all large allocations use `WorkspaceAllocator`.
- [ ] **Optional families:** Often missing negative tests - verify rejected regions return `UnsupportedApi`.
- [ ] **GPU backend:** Often missing crossover validation - verify conservative dispatch beats CPU baseline where selected.
- [ ] **C ABI shim:** Often missing TLS error semantics - verify status + `last_error` under multithreaded tests.
- [ ] **Oracle parity:** Often missing reproducibility controls - verify pinned toolchain, fixed seeds, and artifactized fixtures.

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Manifest/ABI drift | MEDIUM | Regenerate lock across support matrix, classify diff (stable/optional/unstable), add missing oracle targets, and block release until green. |
| `dims`/layout mismatch | HIGH | Reproduce with minimal fixture, centralize calculation in shared validator/writer, and rerun family-wide compat/property tests. |
| OOM abort path leak | HIGH | Trace allocation path, replace direct allocation with fallible allocator/chunking, add regression test with fail allocator. |
| Optional family leakage | MEDIUM | Tighten resolver/feature guards, add symbol-absence and rejection-path tests, and patch docs/support matrix. |
| GPU dispatch regression | MEDIUM | Temporarily force CPU for impacted families, collect telemetry, retune thresholds, and restore guarded GPU rollout. |
| Non-reproducible oracle CI | MEDIUM | Pin environment/toolchain, normalize seeds/fixtures, rerun baseline generation, and quarantine flaky cases with explicit issue IDs. |
| C ABI error contract bugs | MEDIUM | Add TLS-focused integration tests, standardize status mapping table, and verify no panic/exit path crosses ABI boundary. |

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Manifest/ABI Drift Hidden by Header-Only Checks | Investigation, Difference Analysis, Release Preparation | `manifest-audit` lock diff is zero or explicitly approved; oracle family set matches manifest family set. |
| `dims` Contract Drift and Partial-Write Behavior | Implementation 1, Testing | Compat/property tests cover `dims == NULL` and non-`NULL` paths with exact required-bytes checks. |
| Spinor/Complex Layout Mismatch | Implementation 2, Implementation 3 | Oracle parity passes for cart/sph/spinor fixtures; flat-buffer layout tests pass. |
| OOM-Safe Requirement Broken by Native `Vec` Paths | Design, Implementation 1, Testing | OOM/resource-pressure suite passes; allocation audit confirms wrapper-only large allocations. |
| Optional Family Leakage (4c1e/F12/GTG Boundaries) | Implementation 5, Release Preparation | Feature-matrix CI plus symbol-absence checks and `UnsupportedApi` rejection tests pass. |
| Optimizer On/Off Numerical Drift | Implementation 3, Testing | Optimizer parity tests show no tolerance regressions across representative families. |
| Transfer-Dominated GPU Dispatch | Implementation 4, Benchmarking | Crossover benchmarks and telemetry validate dispatch thresholds and fallback rates. |
| Non-Reproducible Oracle Comparisons | Testing, Difference Analysis, Release Preparation | Same commit is stable across reruns/runners with pinned toolchain and fixed seeds. |
| C ABI Error Contract Misimplementation | Implementation 1, Testing, Release Preparation | C ABI multithreaded tests validate TLS last-error behavior and status code correctness. |

## Sources

- `/home/chemtech/workspace/cintx/.planning/PROJECT.md`
- `/home/chemtech/workspace/cintx/.planning/codebase/CONCERNS.md`
- `/home/chemtech/workspace/cintx/docs/libcint_detailed_design_resolved_en.md`
- `/home/chemtech/.codex/get-shit-done/templates/research-project/PITFALLS.md`

---
*Pitfalls research for: libcint-rs (Rust compatibility reimplementation of libcint)*
*Researched: 2026-03-14*
