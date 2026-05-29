# Phase 22: Gauge-Origin Env Slot (Gap A — `PTR_COMMON_ORIG`) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-29
**Phase:** 22-gauge-origin-env-slot-gap-a-ptr-common-orig
**Areas discussed:** Validation semantics, Operator recognition, Verification depth, _origj ket-origin variants

---

## Validation semantics

| Option | Description | Selected |
|--------|-------------|----------|
| Default to [0,0,0] | libcint reads unset env as zero; [0,0,0] is the standard gauge origin. None→kernel uses [0,0,0]. Validator checks finiteness (rejects NaN/inf when Some), not presence. Faithful + ergonomic. | ✓ |
| Reject None (mirror rinv) | Strict: every moment/GIAO caller must set an origin or get InvalidEnvParam. Diverges from libcint's zero-default. | |

**User's choice:** Default to [0,0,0]
**Notes:** Deliberate divergence from `validate_rinv_orig_env_params` (presence-rejection). FND-01's "validator gate" becomes a finiteness check. → CONTEXT D-01.

---

## Operator recognition

| Option | Description | Selected |
|--------|-------------|----------|
| Operator-agnostic now, extend later | Phase 22 ships field + env-read + finiteness validator with no operator-name matching; Phases 24/26 add their predicate when registering int1e_r/GIAO. Avoids a dead name-list. | ✓ |
| Pre-list operator-name substrings | Wire .contains() now against known future names (int1e_r, giao, cg, govlp...) even though none dispatch yet. | |
| Operator metadata flag | Add needs_gauge_origin bit to operator/manifest metadata. Most general, more upfront infra. | |

**User's choice:** Operator-agnostic now, extend later
**Notes:** Pairs naturally with the "default to [0,0,0]" choice — no presence-rejection means nothing needs to know which operators require an origin yet. → CONTEXT D-02.

---

## Verification depth

| Option | Description | Selected |
|--------|-------------|----------|
| Round-trip + validator tests; fixture as infra | Phase 22 proves env[1..3] round-trips raw↔plan, validator behaves, setter works; builds + commits the non-zero fixture as data/harness, but real parity is exercised when MOM-01/GIAO land. No kernel work; no MOM-01 overlap. | ✓ |
| Stand up dipole int1e_r as a live gate | Implement one minimal consumer so the fixture is a real byte-identity gate now. Strongest proof but pulls MOM-01 forward. | |

**User's choice:** Round-trip + validator tests; fixture as infra
**Notes:** Matches FND-01 literally ("a non-zero gauge-origin oracle fixture exists", "Env round-trip + validator unit tests pass"). → CONTEXT D-03.

---

## _origj ket-origin variants

| Option | Description | Selected |
|--------|-------------|----------|
| No — PTR_COMMON_ORIG only | _origj places the origin at the ket basis center (kernel-side coordinate), orthogonal to env[1..3]. Keep Phase 22 scoped to the slot. | ✓ |
| Yes — generalize origin abstraction | Design the field/setter to carry an origin-selection mode (common vs origj) up front. More generality for an unbuilt consumer. | |

**User's choice:** No — PTR_COMMON_ORIG only
**Notes:** _origj is the moment phase's (Phase 24) concern. → CONTEXT D-04.

---

## Claude's Discretion

- Exact non-zero origin value(s) and fixture molecule/shell corpus (H2O/STO-3G default).
- Standalone `validate_common_orig_env_params` vs a shared origin-validation helper.
- Whether the fixture's vendor-reference harness is a stub-now or fully-wired `vendor_*` call.

## Deferred Ideas

- `_origj` / origin-on-ket-center moment variants — Phase 24 (MOM).
- Live byte-identity parity through a real consuming kernel — Phases 24 (MOM-01) / 26 (GIAO).
- Reviewed-but-not-folded todos: `rys-nroots-ge6-wheeler-fallback.md` (FND-02 / Phase 25), `oracle-cart-offset-vendor-zero.md` (pre-existing baseline test issue).
