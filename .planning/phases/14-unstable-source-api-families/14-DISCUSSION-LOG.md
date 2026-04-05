# Phase 14: Unstable-Source-API Families - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-05
**Phase:** 14-unstable-source-api-families
**Areas discussed:** Manifest & gating strategy, Grids env plumbing, Breit spinor-only scope, Oracle fixture strategy

---

## Manifest & gating strategy

### How should source-only symbols enter the compiled manifest?

| Option | Description | Selected |
|--------|-------------|----------|
| New manifest profile (Recommended) | Add an 'unstable-source' profile alongside base/with-f12/with-4c1e | ✓ |
| Fold into existing profiles | Source-only symbols tagged with stability=SourceOnly in existing family profiles | |
| Separate manifest lock file | A second compiled_manifest_unstable.lock.json for source-only rows | |

**User's choice:** New manifest profile
**Notes:** Clean separation, matches existing profile pattern.

### Should the unstable-source profile be combinable with other profiles?

| Option | Description | Selected |
|--------|-------------|----------|
| Standalone only (Recommended) | No cross-product with f12/4c1e. 5 profiles total. | ✓ |
| Combinable with others | Cross-product with f12/4c1e, up to 8 profiles. | |

**User's choice:** Standalone only
**Notes:** Keeps CI and fixture matrix simple.

### Should the manifest generator auto-classify source-only symbols?

| Option | Description | Selected |
|--------|-------------|----------|
| Explicit list in generator | Hardcode ~18 source-only symbols. Clear, auditable. | ✓ |
| Auto-detect from libcint headers | Parse headers to identify unstable symbols. | |
| You decide | Claude picks based on existing pattern. | |

**User's choice:** Explicit list in generator
**Notes:** No risk of accidental promotion.

---

## Grids env plumbing

### How should the grids family handle NGRIDS/PTR_GRIDS env contract?

| Option | Description | Selected |
|--------|-------------|----------|
| Extend ExecutionPlan (Recommended) | Add GridsEnvParams alongside OperatorEnvParams. Validator rejects invalid grids params. | ✓ |
| Pass through raw env only | Kernel reads NGRIDS/PTR_GRIDS directly from raw env. | |
| You decide | Claude picks based on F12 zeta plumbing. | |

**User's choice:** Extend ExecutionPlan
**Notes:** Mirrors F12 zeta plumbing pattern.

### How should kernel output sizing handle NGRIDS?

| Option | Description | Selected |
|--------|-------------|----------|
| NGRIDS as output dimension multiplier | Planner multiplies standard dims by NGRIDS. Output: (ncomp * NGRIDS * di * dj). | ✓ |
| Grid-batched kernel launch | One kernel invocation per grid point or batch. | |
| You decide | Claude picks based on g1e_grids.c. | |

**User's choice:** NGRIDS as output dimension multiplier
**Notes:** Matches libcint's behavior.

---

## Breit spinor-only scope

### Which representations should Breit implement?

| Option | Description | Selected |
|--------|-------------|----------|
| All representations (cart/sph/spinor) | Implement all 3 representations for completeness. | |
| Spinor-only (Recommended) | Only spinor variants. Cart/sph rarely used for Breit. | ✓ |
| sph + spinor, skip cart | Implement sph and spinor, skip cart. | |

**User's choice:** Spinor-only
**Notes:** Breit is physically meaningful only in relativistic (spinor) context.

### How should the Breit kernel implement Gaunt+gauge composition?

| Option | Description | Selected |
|--------|-------------|----------|
| Single composite kernel (Recommended) | One kernel computes Gaunt and gauge internally. Matches breit.c. | ✓ |
| Separate Gaunt+gauge then sum | Independent launches, then add results. | |
| You decide | Claude picks based on breit.c internals. | |

**User's choice:** Single composite kernel
**Notes:** Single launch, single output buffer.

---

## Oracle fixture strategy

### How should oracle tests be organized?

| Option | Description | Selected |
|--------|-------------|----------|
| Single test file (Recommended) | One unstable_source_parity.rs with per-family functions. ~18 symbols. | ✓ |
| Per-family test files | Separate files per family. | |
| You decide | Claude picks based on existing oracle test structure. | |

**User's choice:** Single test file
**Notes:** Scale doesn't warrant multiple files.

### Nightly CI job structure?

| Option | Description | Selected |
|--------|-------------|----------|
| Extra job in existing workflow (Recommended) | Add unstable_source_oracle job, nightly-gated. | ✓ |
| Separate nightly workflow | New dedicated workflow file. | |
| You decide | Claude picks based on Phase 4 CI conventions. | |

**User's choice:** Extra job in existing workflow
**Notes:** Reuses runner config and artifact paths.

### Fixture molecule choice?

| Option | Description | Selected |
|--------|-------------|----------|
| Same H2O/STO-3G (Recommended) | Reuse existing fixture. Grids adds grid coords to env. | ✓ |
| Family-specific molecules | Different molecules per family for non-trivial values. | |
| You decide | Claude picks based on whether H2O/STO-3G produces non-trivial values. | |

**User's choice:** Same H2O/STO-3G
**Notes:** Consistent with all prior oracle tests.

---

## Claude's Discretion

- Internal module layout for kernel files
- Exact GridsEnvParams struct fields and validation
- Breit Gaunt+gauge routing through 2e infrastructure
- Order of family implementation across plans
- Grid coordinate fixture values

## Deferred Ideas

None — discussion stayed within phase scope.
