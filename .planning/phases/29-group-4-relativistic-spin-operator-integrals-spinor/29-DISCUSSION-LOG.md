# Phase 29: Group 4 — Relativistic Spin-Operator Integrals (spinor) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-31
**Phase:** 29-group-4-relativistic-spin-operator-integrals-spinor
**Areas discussed:** 2e si-transform scope, 2e kappa fixture, Spike gate for new machinery, Plan / wave decomposition

---

## 2e si-transform scope

| Option | Description | Selected |
|--------|-------------|----------|
| Full 2e si suite | Build c2s_si_2e1/2e2 + 2e1i/2e2i + c2s_sf_2e1/2e2 partners — everything REL-03 AND REL-04 need. Correct the Phase-28 deferred note. | ✓ |
| Real-only, defer i-variants | c2s_si_2e1/2e2 + sf_2e only; defer 2e1i/2e2i. Would leave REL-04 unfinished. | |
| Re-scope: 2e into its own phase | Phase 29 = 1e σ only; new phase for REL-03/04 + 2e foundation. | |

**User's choice:** Full 2e si suite
**Notes:** Scouting verified against `autocode/intor4.c` that REL-03 uses `c2s_si_2e1/2e2` (real) and REL-04 (ssp/sps/vsp/spv) uses the imaginary `c2s_si_2e1i/2e2i` + the `c2s_sf_2e` partner. Phase 28's deferred-ideas note wrongly attributed these to Phases 30/31 — corrected: they are a Phase-29 deliverable.

---

## 2e kappa fixture

| Option | Description | Selected |
|--------|-------------|----------|
| New 4-shell kappa 2e fixture | build_kappa_spinor_2e_fixture: 4 shells, non-square, kappa≠0 GT/LT mix, ≥1 nctr>1; heavy-atom realism secondary. | ✓ |
| Extend 1e fixture inline | Reuse build_kappa_spinor_fixture, add 2 shells inline. | |
| Real heavy-atom 2c basis as primary | Physical relativistic 2e case as the gate, synthetic secondary. | |

**User's choice:** New 4-shell kappa 2e fixture
**Notes:** Extends Phase-28 adversarial rigor to a 2-electron config. Heavy-atom realism case stays a secondary cross-check.

---

## Spike gate for new machinery

| Option | Description | Selected |
|--------|-------------|----------|
| Hard-gate spike (2e + gout patterns) | Full design spike before plan tasks (Phase-28 D-06 precedent). | |
| Targeted spike on 2e transform only | Spike just c2s_si_2e1/2e2 layout; transcribe gouts. | |
| No spike — transcribe + vendor gate | Transcribe all from libcint; rely on atol=1e-12 vendor parity gate. | ✓ |

**User's choice:** No spike — transcribe + vendor gate
**Notes:** Deliberate departure from the Phase-28 hard-gate precedent. Accepted risk: the novel 2e transform layout carries rework risk surfacing only at the vendor gate. Structural mitigation captured in CONTEXT D-03: a transform-level byte-identity micro-test is the FIRST task of the 2e wave (spike-level rigor folded into the plan, not a separate spike phase).

---

## Plan / wave decomposition

| Option | Description | Selected |
|--------|-------------|----------|
| 1e → 2e foundation → 2e families | Wave 1: 1e σ + flip int1e_sp; Wave 2: 2e transform foundation + fixture + micro-test; Wave 3: 2e families. | ✓ |
| Foundation-first, then fan out | 1e+2e transforms up front, then all families in parallel. | |
| By requirement (REL-01→02→03→04) | One plan per requirement in strict order. | |

**User's choice:** 1e → 2e foundation → 2e families
**Notes:** Sequential de-risk; each wave gated (vendor parity green) before the next. Wave 1 free-rides the Phase-28 foundation (lowest risk, lands first).

---

## Claude's Discretion

- Internal module naming/factoring for the new 2e si transforms.
- Exact per-family gout component ordering — resolve from intor3.c/intor4.c during research/planning.
- Exact molecule/element + kappa assignments for build_kappa_spinor_2e_fixture (subject to D-02 constraints) and the heavy-atom cross-check.
- Precise plan boundaries inside each wave.
- Whether the 2e transform micro-test compares to vendored c2s_si_2e* directly or via a thin driving family.

## Deferred Ideas

- GIAO×σ slice (Phase 30); gauge/Breit–Gaunt 2e (Phase 31); PARITY-01 full-parity gate (Phase 31).
- Reviewed-not-folded todos: `oracle-cart-offset-vendor-zero` (belongs to PARITY-01/Phase 31); `rys-nroots-ge6-wheeler-fallback` (general math infra, no Group-4 relevance).
