# Pitfalls Research

**Domain:** Rust-native libcint-compatible integral library
**Researched:** 2026-03-21
**Confidence:** HIGH for project-specific pitfalls derived from the design and current architecture; MEDIUM for mitigation cost estimates.

## Critical Pitfalls

### Pitfall 1: Nondeterministic reduction order breaks oracle parity

**What goes wrong:**
Chunking, batching, or kernel specialization changes the accumulation order, and numerically valid GPU outputs drift outside the accepted oracle tolerances.

**Why it happens:**
Floating-point addition is not associative, and GPU work partitioning makes it easy to reorder reductions without noticing.

**How to avoid:**
- Make reduction order a planner contract, not an implementation accident.
- Record chunk plans and reduction strategies in tracing so parity failures can be tied to execution shape changes.
- Require an oracle regression fixture whenever chunking or reduction heuristics change.

**Warning signs:**
- Parity failures appear only after scheduler changes.
- Results differ by chunk size or backend configuration.
- The same family passes on small cases but fails on large batches.

**Phase to address:**
Base execution phase and every later performance-tuning phase.

---

### Pitfall 2: High-memory families blow through GPU limits

**What goes wrong:**
Large or high-angular-momentum families exhaust device memory, leading to unstable runtime behavior, silent partial work, or late-stage support rollbacks.

**Why it happens:**
The support matrix includes families whose workspace and intermediate-buffer sizes can grow sharply, especially once optional families are enabled.

**How to avoid:**
- Route every large allocation through a fallible allocator.
- Estimate memory before launch and chunk deterministically.
- Treat unsupported envelopes as explicit `UnsupportedApi` decisions, not accidental runtime failures.
- Add OOM and memory-limit tests before expanding optional-family coverage.

**Warning signs:**
- `MemoryLimitExceeded` appears only under optional feature profiles.
- Planner retries repeatedly with shrinking chunks.
- Large-family tests pass locally but fail on smaller CI GPUs.

**Phase to address:**
Planner/runtime phase, optional-family phase, and dedicated OOM/verification phase.

---

### Pitfall 3: Compat layout contracts diverge from device-buffer assumptions

**What goes wrong:**
The raw compatibility layer accepts shapes, dims, or buffer layouts that the backend cannot safely execute or write back, causing misordered results, invalid writes, or incorrect complex/cart/spinor layouts.

**Why it happens:**
The public compat contract is more permissive than a naive device-copy path. If layout validation lags behind backend assumptions, bugs appear at the raw boundary.

**How to avoid:**
- Keep one source of truth for `dims`, required element counts, and output layout contracts.
- Validate all raw inputs before any transfer or kernel launch.
- Exercise layout permutations through both safe and compat entrypoints in tests.
- Treat partial writes and implicit truncation as hard failures.

**Warning signs:**
- Raw APIs fail while equivalent safe APIs pass.
- Cart/sph/spinor result sizes differ between planning and writeback.
- Bugs cluster around custom `dims`, `out == NULL`, or `cache == NULL` cases.

**Phase to address:**
Typed-foundation/compat phase and all later oracle-comparison phases.

---

### Pitfall 4: Manifest and oracle coverage drift apart

**What goes wrong:**
The codebase appears feature-complete, but symbol coverage, optional-family behavior, or helper parity drifts from the compiled manifest lock and release gates.

**Why it happens:**
Wide API coverage is hard to track manually; optional and unstable families multiply the number of profiles that must stay consistent.

**How to avoid:**
- Treat manifest audit and oracle comparison as hard release gates from the start.
- Regenerate and diff the compiled manifest lock whenever feature coverage changes.
- Update CI jobs automatically when optional-family coverage expands.
- Require helper/legacy/transform parity to pass with the same seriousness as core integrals.

**Warning signs:**
- CI passes for one feature profile but fails for another.
- New symbols appear without matching tests.
- Helper APIs lag behind the main integral families.

**Phase to address:**
Manifest/governance phase and every release-preparation checkpoint.

---

### Pitfall 5: Backend details leak into the public API too early

**What goes wrong:**
Safe or compat callers become coupled to CubeCL-specific concepts, making future backend changes or fallback strategies expensive and destabilizing.

**Why it happens:**
The project is compute-heavy, so it is tempting to expose backend-specific handles for performance work before the public contract is stable.

**How to avoid:**
- Keep public APIs framed in terms of typed inputs, outputs, plans, and error contracts.
- Hide backend selection and transfer details behind runtime/planner boundaries.
- Expose diagnostics, not backend internals.

**Warning signs:**
- Public types reference backend runtime objects.
- Changing backend execution details requires public API changes.
- Planner diagnostics are only available through internal types.

**Phase to address:**
API-design phase and safe-facade/C-ABI phase.

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When acceptable |
|----------|-------------------|----------------|-----------------|
| Hard-coding support decisions outside the manifest | Faster prototyping | Coverage drift and opaque release failures | Never beyond short local experiments |
| Using floating `stable` and unlocked dependencies in CI | Less setup work | Non-reproducible oracle and manifest results | Never for gated CI |
| Writing raw-layout logic twice (safe path and compat path) | Faster initial implementation | Semantic drift and duplicated bugs | Never; centralize validation and sizing rules |
| Expanding optional families before OOM/oracle coverage exists | Earlier apparent coverage | Support matrix becomes untrustworthy | Only behind private experiments |

## "Looks Done But Isn't" Checklist

- [ ] **Manifest coverage:** every stable and enabled optional symbol is implemented and audited.
- [ ] **Raw compatibility:** `dims`, `out`, `cache`, optimizer, and complex layout edge cases are tested.
- [ ] **OOM behavior:** allocator failures and memory-limit failures stop safely with no partial writes.
- [ ] **Optimizer parity:** with/without optimizer results are equivalent within accepted tolerance.
- [ ] **Feature-matrix CI:** base, `with-f12`, `with-4c1e`, and combined profiles all run the right gates.

## Pitfall-to-Phase Mapping

| Pitfall | Prevention phase | Verification |
|---------|------------------|--------------|
| Nondeterministic reduction order | Execution backend foundation | Oracle regression tests under varied chunking |
| GPU memory blowups | Planner/runtime + optional families | OOM and memory-limit suites across profiles |
| Compat/layout drift | Typed foundations + compat layer | Raw/safe equivalence and layout tests |
| Manifest/oracle drift | Governance/release gating | Manifest audit plus oracle CI |
| Backend leakage | Safe facade and C-ABI design | API review against backend-agnostic public types |

## Sources

### Primary inputs
- `docs/design/cintx_detailed_design.md`
- `README.md`
- Local crate structure under `crates/` and `xtask/`

### Supporting references
- CubeCL docs: https://docs.rs/crate/cubecl/latest
- Cargo resolver docs: https://doc.rust-lang.org/nightly/cargo/reference/resolver.html
- Cargo feature docs: https://doc.rust-lang.org/stable/cargo/reference/features.html

---
*Pitfalls research for: cintx*
*Researched: 2026-03-21*
