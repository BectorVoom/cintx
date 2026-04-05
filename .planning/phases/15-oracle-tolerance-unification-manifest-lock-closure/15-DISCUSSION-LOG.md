# Phase 15: Oracle Tolerance Unification & Manifest Lock Closure - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-04-06
**Phase:** 15-oracle-tolerance-unification-manifest-lock-closure
**Areas discussed:** oracle_covered gap closure, Family tolerance map completeness, Manifest lock regeneration ordering, CI gate four-profile pass

---

## oracle_covered gap closure

### How to populate oracle_covered

| Option | Description | Selected |
|--------|-------------|----------|
| Oracle run marks coverage | compare.rs marks oracle_covered=true on each entry that passes. Coverage is objective. | ✓ |
| Xtask audit derives coverage | New xtask cross-references oracle results against manifest. Decouples tracking from comparison. | |
| Build-time manifest generator | build.rs sets oracle_covered based on which symbols have test fixtures. Structural, not result-based. | |

**User's choice:** Oracle run marks coverage
**Notes:** None

### Persistence of oracle_covered

| Option | Description | Selected |
|--------|-------------|----------|
| Persist in lock file | oracle_covered=true written into compiled_manifest.lock.json and committed. CI verifies claim. | ✓ |
| Computed at CI time only | No oracle_covered in lock file. CI computes coverage each run. | |

**User's choice:** Persist in lock file
**Notes:** None

### Handling stable entry failures at 1e-12

| Option | Description | Selected |
|--------|-------------|----------|
| Fix the kernel | Per ORAC-01: kernel bug, not tolerance loosening. Block until fixed. | ✓ |
| Flag and defer | Mark oracle_covered=false, file follow-up. Risks gaps at milestone close. | |
| Investigate then decide | Case-by-case: fix if kernel bug, document if fundamental limitation. | |

**User's choice:** Fix the kernel
**Notes:** None

---

## Family tolerance map completeness

### How to add missing families

| Option | Description | Selected |
|--------|-------------|----------|
| Catch-all default | Replace explicit match with wildcard. Any family gets UNIFIED_ATOL. | ✓ |
| Add explicit entries | Add f12, unstable::source::{1e,3c1e,3c2e} as explicit arms. Keep bail! safety net. | |
| Family-prefix matching | Match on starts_with for unstable::source::, explicit for stable. | |

**User's choice:** Catch-all default
**Notes:** None

### Replace PHASE4_ORACLE_FAMILIES

| Option | Description | Selected |
|--------|-------------|----------|
| Manifest-driven | Derive oracle-eligible families from manifest lock (stability=stable or optional). | ✓ |
| Extend the list | Add "f12" to list, rename to ALL_ORACLE_FAMILIES. | |
| You decide | Claude picks during implementation. | |

**User's choice:** Replace with manifest-driven
**Notes:** None

---

## Manifest lock regeneration ordering

### Single file vs per-profile

| Option | Description | Selected |
|--------|-------------|----------|
| Single file, all profiles | Keep current structure. Atomic regeneration. | ✓ |
| Per-profile lock files | Split into base.lock.json, with-f12.lock.json, etc. | |

**User's choice:** Single file, all profiles
**Notes:** None

### Unstable-source profile inclusion

| Option | Description | Selected |
|--------|-------------|----------|
| Keep separate | Per Phase 14 D-02. Four standard profiles only; unstable-source in nightly. | ✓ |
| Include in regeneration | All five profiles together. | |
| You decide | Claude picks based on CI structure. | |

**User's choice:** Keep separate
**Notes:** None

---

## CI gate four-profile pass

### Gate structure

| Option | Description | Selected |
|--------|-------------|----------|
| Matrix strategy | GitHub Actions matrix over four profiles. Parallel jobs. | ✓ |
| Sequential loop | Single job loops over profiles. Simpler but slower. | |
| Extend existing gate | Ensure existing oracle_parity_gate covers all four. | |

**User's choice:** Matrix strategy
**Notes:** None

### oracle_covered completeness in CI

| Option | Description | Selected |
|--------|-------------|----------|
| Add coverage check to manifest-audit | Validates no drift AND every stable entry has oracle_covered=true. | ✓ |
| Separate coverage gate | Two distinct CI jobs for drift and coverage. | |
| No coverage gate | Trust oracle pass implies coverage. Metadata for humans only. | |

**User's choice:** Yes, add coverage check
**Notes:** None

---

## Claude's Discretion

- Internal ordering of oracle audit across families/profiles
- Whether to run tolerance audit as standalone xtask or integrate into existing parity commands
- How to structure oracle_covered write-back mechanism
- Exact matrix job naming and artifact handling

## Deferred Ideas

None -- discussion stayed within phase scope.
