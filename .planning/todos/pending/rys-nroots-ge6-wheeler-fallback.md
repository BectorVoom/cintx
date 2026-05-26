---
id: rys-nroots-ge6-wheeler-fallback
created: 2026-05-26T00:00:00Z
status: pending
severity: medium
source: pyscf_rs Eu2+ capability query 2026-05-26 (also Phase 21 plan R2; rys.rs "deferred to Phase 10")
resolves_phase: 25
---

# Rys quadrature unsupported for nroots ≥ 6 — blocks all f-block / lanthanide / actinide chemistry

## What
`rys_roots_host` supports only `nroots ∈ 1..=5`; `nroots ≥ 6` **panics**:

- `crates/cintx-cubecl/src/math/rys.rs:3255` — `_ => panic!("rys_roots_host: nroots={nroots} > 5 not supported")`
- Module note `rys.rs:10` + `:3247` + `:3520` — *"Higher nroots (6+) via Wheeler/Jacobi fallback deferred to Phase 10"* (never landed).

Two-electron / nuclear nroots = `(lᵢ+lⱼ+lₖ+lₗ)/2 + 1` (`two_electron.rs:70`; nuclear `one_electron.rs:285`). So any ERI quartet with **total L > 8** exceeds the ceiling:

| Quartet | total L | nroots | result |
|---------|---------|--------|--------|
| (ff\|ff) | 12 | 7 | **panic** |
| (ff\|fp), (ff\|dd) | 10 | 6 | **panic** |
| (ff\|dp), (df\|df) | 8–9 | 5–6 | boundary/fail |
| (dd\|dd), (ff\|ss) | 8, 6 | 5, 4 | OK |

Compounding gate: `crates/cintx-cubecl/src/executor.rs:11-13` rejects any shell with `ang_momentum > 4` (`Err("max(l)>4")`), so g/h-function basis sets are refused before the kernel is even reached.

## Consequence (capability blocker)
**No f-element SCF is possible.** A faithful f-block / lanthanide / actinide species needs explicit f-functions (l=3); the `(ff|ff)` Coulomb/exchange integral needs nroots=7 → panic. Concrete driver: pyscf_rs cannot compute **Eu²⁺** (`[Xe]4f⁷`) energy — its open shell *is* the 4f subshell. Every bundled Eu basis (`stuttgart_rsc`, `crenbl`, `ano`, `def2-mtzvp/mtzvpp`, `sarc-dkh2`) carries F-shells; several (`stuttgart_rsc`, `ano`) also carry G/H → rejected by the l>4 gate.

## Workaround assessment — none viable today
- **Large-core "f-in-core" ECP** (4f in the core → d-only valence, worst ERI (dd|dd)=nroots 5, within ceiling): would sidestep the panic, BUT **no bundled Eu set does this** — even `crenbl` (`Eu nelec 54`, Xe-core) keeps 4f⁷ *in the 9 valence electrons*, so it still has F-shells. A true f-in-core ECP would have to be supplied externally, is chemically inappropriate for open-shell 4f⁷ Eu²⁺ (treats 4f as inert), and still rides unvalidated lanthanide-ECP numerics.
- **No partial mitigation** for the small-core/standard path — the panic is unconditional at nroots≥6.

## Fix (the real prerequisite for heavy-element chemistry)
1. Implement the deferred **Wheeler/Jacobi nroots≥6 Rys root+weight fallback** in `rys.rs` (the long-deferred D-item), with byte-identity oracle parity vs vendored libcint for nroots 6..~13 (g/f/h quartets).
2. Raise/extend the `executor.rs` `ang_momentum > 4` gate to admit g/h once the roots support them.
3. Validate a lanthanide ECP+basis (e.g. `stuttgart_rsc` / `crenbl` for a light lanthanide) byte-identical to upstream PySCF — the heaviest validated ECP today is Cu/LANL2DZ (a d-block small ECP); f-projector ECP numerics are entirely unvalidated.

## Consumer impact
pyscf_rs heavy-element / lanthanide / actinide chemistry (and any g-function correlation-consistent basis on lighter heavy atoms) is blocked until (1). Out of scope for pyscf_rs v1 light-element corpus, but a hard wall for any heavy-element extension. See pyscf_rs memory `project_angular_momentum_element_ceiling`.
