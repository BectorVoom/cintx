# Phase 3 Manifest Governance Contract

This document defines the compiled-manifest governance policy for phase 3 compatibility claims (`COMP-03`, `VERI-01`).

## Lock Semantics

The compiled manifest lock is a typed artifact with mandatory fields:

- Canonical symbol identity: `{family, operator, representation, symbol}`
- Profile membership per entry
- Stability classification per entry (`stable`, `experimental`, `deprecated`)
- Profile scope metadata:
  - Approved phase-3 scope: `base`, `with-f12`, `with-4c1e`, `with-f12+with-4c1e`
  - Observed profile union derived from lock entries

## Canonicalization Rules

- Symbol names are normalized to lowercase alphanumeric tokens joined by `_`.
- Profile labels accept aliases (`f12`, `4c1e`, mixed separators/case) and normalize to governed labels.
- Combined profile labels are canonicalized to `with-f12+with-4c1e`.

## Governance Checks

The lock must satisfy all of the following before compatibility coverage claims are accepted:

- Schema invariants validate successfully.
- Observed profile union exactly matches the approved phase-3 profile set.
- Canonical lock content has no duplicate canonical symbol entries.
- Entries cannot include profiles outside the approved scope.

## Lock Drift Policy

Any canonical lock drift is **blocking by default**.

Drift is allowed only when an explicit approval is provided with:

- A non-empty rationale
- An approved reason:
  - `schema-change`
  - `upstream-symbol-change`
  - `profile-policy-change`

Without this explicit approval metadata, drift is treated as a regression (`UnapprovedLockDrift`) and must fail CI.

## CI Gate Expectations

The phase-3 governance tests are expected to gate CI for compatibility claims:

- `manifest_schema_invariants`
- `manifest_profile_union_is_stable`
- `lock_drift_requires_explicit_update`
