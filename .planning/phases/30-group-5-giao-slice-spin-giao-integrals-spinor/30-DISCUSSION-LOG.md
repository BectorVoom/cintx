# Phase 30: Group 5 (GIAO×σ slice) — Spin-GIAO Integrals (spinor) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-01
**Phase:** 30-group-5-giao-slice-spin-giao-integrals-spinor
**Areas discussed:** Family scope boundary, Gauge+kappa fixture, De-risk approach, Wave decomposition

---

## Family scope boundary

### Gaunt families (int2e_{cg,giao}_ssa10ssp2)

| Option | Description | Selected |
|--------|-------------|----------|
| Defer to Phase 31 | Keep Phase 30 to the non-Gaunt set; ssa10ssp2 are Gaunt (launch_breit, Phase 14 dep) and the GIAO-03 glob excludes 'ssa10'. Phase 31 BREIT-03 + full-parity gate captures them. | ✓ |
| Include in Phase 30 | Close all gauge-NMR families here; pulls a launch_breit dependency and front-loads Gaunt-decomposition risk. | |

**User's choice:** Defer to Phase 31.

### Other-side spin-angular sa01 families

| Option | Description | Selected |
|--------|-------------|----------|
| Include all sa01 | int1e_spgsa01 + int1e_{cg,giao}_sa10sa01 are 1e GIAO×σ in intor3.c, covered by the GIAO-03 globs, reuse the same transforms+gout; closing them here makes GIAO-03 fully complete. | ✓ |
| Defer sa01 variants | Limit to sp/nucsp arms; treat double-gauge sa10sa01/spgsa01 as a separate slice. | |

**User's choice:** Include all sa01.

**Notes:** GIAO-03's requirement glob (`int2e_cg_sa10*`) literally excludes `ssa10`, so deferring the Gaunt families still allows GIAO-03 to be marked Complete after Phase 30 — no scope inconsistency.

---

## Gauge+kappa fixture

| Option | Description | Selected |
|--------|-------------|----------|
| One combined gauge∧kappa fixture | Single fixture (1e + 4-shell 2e) simultaneously gauge≠0 AND kappa≠0 GT/LT, non-square, ≥1 nctr>1; extends build_kappa_spinor_2e_fixture with a non-zero common origin. Exercises the real gauge×kappa cross-term. | ✓ |
| Two separate fixtures composed | Reuse Phase-22 gauge + Phase-29 kappa fixtures independently; neither alone stresses the cross-term. | |
| You decide | Let planner/researcher pick within the SC#2 constraint. | |

**User's choice:** One combined gauge∧kappa fixture.

**Notes:** The integrand couples gauge origin and kappa, so a single combined fixture is the honest SC#2 gate.

---

## De-risk approach

| Option | Description | Selected |
|--------|-------------|----------|
| Inherit Phase-29: transcribe + gate, gout micro-test first | No spike; transcribe from libcint, prove by atol=1e-12 vendor gate; apply the Phase-29 D-03 mitigation at the GOUT level — a gauge-gout byte-identity micro-test as the FIRST task. | ✓ |
| Run a design spike first | Treat gauge×σ×imaginary as novel enough for a hard-gate spike before planning. | |
| Transcribe + gate only, no micro-test | Rely solely on the per-family vendor gate; fastest, highest rework risk. | |

**User's choice:** Inherit Phase-29 transcribe + gate, gout micro-test first.

**Notes:** Since the transforms already exist, the de-risk relocates from a transform micro-test (Phase 29) to a gauge-gout micro-test — the gauge `g`-factor fold into sigma_p.rs is the only genuinely-new piece.

---

## Wave decomposition

| Option | Description | Selected |
|--------|-------------|----------|
| 1e-all-then-2e-all, gout micro-test first | Wave 0: gauge-gout micro-test + combined fixture (1e). Wave 1: all 1e families. Wave 2: 4-shell 2e fixture + all 2e families. Each wave gated green. | ✓ |
| By family group (spg → cg_sa10 → giao_sa10) | Vertical slices per gauge flavor; interleaves 2e fixture/launcher risk into the first wave. | |
| You decide | Let the planner choose wave boundaries. | |

**User's choice:** 1e-all-then-2e-all, gout micro-test first.

---

## Claude's Discretion

- Gauge-origin σ·p gout module structure (extend `sigma_p.rs` vs dedicated `giao_sigma` assembler).
- Exact per-family gout component ordering and the precise `c2s_si_1ei`/`c2s_si_1e`/`c2s_sf_1e` arm per family (transcribe from verified `intor3.c`/`intor4.c` pairings).
- Exact molecule/element + kappa + gauge-origin coordinates for the combined fixture (within D-02 hard constraints).
- Whether the Wave-0 gout micro-test compares to a vendored thin family or a hand-derived reference.
- Precise plan boundaries inside each wave.

## Deferred Ideas

- Gaunt GIAO families `int2e_{cg,giao}_ssa10ssp2` → Phase 31 (launch_breit / BREIT-03 / PARITY-01).
- Gauge / Breit–Gaunt 2e (`int2e_gauge_r1/r2_*`), Gaunt `ssp/sps` → Phase 31.
- PARITY-01 full-parity gate → Phase 31.
