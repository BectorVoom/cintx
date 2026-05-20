# Phase 19: `int1e_ecp_*` Type-1/Type-2 Evaluator - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-12
**Phase:** 19-int1e-ecp-type1-type2-evaluator
**Areas discussed:** Oracle reference source, Typed-API placement of `ecpbas`, Kernel implementation strategy, Phase scope (gradient variants)

---

## Pre-Discussion Codebase Findings

Two facts surfaced during the codebase scout that materially changed the gray-area
framing before any user question was asked:

1. **Vendored `libcint-master/src/` contains zero ECP code.** Inspection of
   `libcint-master/src/` returned no `ecp.c`, `cint_ecp.h`, or any `*ecp*` file.
   The ROADMAP's "byte-identity against libcint" success criterion therefore
   requires importing ECP source into the vendor tree (or shifting the reference).
2. **No Cu/LANL2DZ fixture exists.** `crates/cintx-oracle/src/fixtures.rs`
   contains only `build_h2o_sto3g` and `build_h2o_sto3g_f12`. The ROADMAP's
   "Cu/LANL2DZ already present in the oracle test corpus" claim is aspirational.
   The fixture is built fresh in this phase.

Both facts are surfaced in CONTEXT.md `<specifics>` for downstream agents.

---

## Gray Area Selection

| Area | Surfaced | Selected by User |
|------|----------|------------------|
| Oracle reference source | ✓ | ✓ |
| Typed-API placement of `ecpbas` | ✓ | ✓ |
| Kernel implementation strategy | ✓ | ✓ |
| Phase scope: gradient variants | ✓ | ✓ |

User selected ALL four offered gray areas for deep-dive.

---

## Oracle reference source

| Option | Description | Selected |
|--------|-------------|----------|
| Vendor libcint's upstream `ecp.c` (primary) + libECP secondary (non-blocking) | Add ecp.c / ecp.h / cint_ecp.h from upstream libcint 6.1.3 to libcint-master/src/, extend oracle/build.rs to compile them, add FFI wrappers (atol=1e-12 gate). libECP linked optionally for cross-check. Matches SC#4 wording literally. | ✓ |
| Vendor libcint ecp.c only — skip libECP secondary this phase | Same as above but defer the libECP cross-check to a follow-up. Smaller phase scope. SC#4's "non-blocking secondary oracle" becomes a deferred idea. | |
| chrr/libECP as primary (no libcint vendor) | Use chrr's libECP as the byte-identity reference. Departs from SC#4's wording (which names libcint primary). Lighter vendor burden but tolerance may need re-evaluation. | |
| PySCF's pyscf/gto/ecp.py reference (Python harness) | Build a Python sidecar oracle invoked from cintx-oracle. Breaks the C-FFI pattern used by every other family. | |

**User's choice:** Vendor libcint's upstream ecp.c (primary) + libECP secondary (non-blocking)
**Notes:** Honors ROADMAP SC#4 wording verbatim. Research-time risk surfaced in
CONTEXT.md `<specifics>`: libcint 6.1.3 may not ship ecp.c with its main release tarball
(historically packaged with PySCF). If verification at research time finds ecp.c is
not in the libcint 6.1.3 release, the planner must surface this as a blocker before
plan-phase commits — fallback paths are PySCF's libcint-bundled ECP or chrr/libECP
promoted to primary.

---

## Typed-API placement of `ecpbas`

| Option | Description | Selected |
|--------|-------------|----------|
| Extend `BasisSet` with optional `ecp_shells: Arc<[Arc<EcpShell>]>` field | Conceptually clean — ECP shells ARE part of a basis set. New `EcpShell` type. SemVer-additive (private field, public accessor). Invasive but principled. | ✓ |
| Separate `EcpBasis` struct on `ExecutionOptions::ecp_basis: Option<EcpBasis>` | Mirrors `f12_zeta` and `aosym` precedent. SemVer-safe, zero changes to BasisSet. Conceptually weird (basis data in execution options) but matches established pattern. | |
| Hybrid: `EcpBasis` struct in cintx-core, attached via builder method `.with_ecp_basis(ecp)` | Cleanest separation. Builder method preserves constructor SemVer. Most surface area. | |

**Pre-narrowing note:** Two further options were considered and ruled out before
presenting choices:
- New positional arg on `SessionRequest::new(...)` — breaks SemVer.
- Tag ECP shells inside the existing shell array — violates the type-safe API
  priority.

**User's choice:** Extend `BasisSet` with optional `ecp_shells: Arc<[Arc<EcpShell>]>` field
**Notes:** Principled over pragmatic. ECP shells ARE basis-set-belonging data; carrying
them on `ExecutionOptions` was the easy-but-conceptually-wrong path. SemVer
preservation handled via additive private field + public accessor returning `&[]`
when no ECP attached.

---

## Kernel implementation strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Full `#[cube]` GPU — add Bessel/radial-quad math modules | Matches Phase 8-10 pattern. Adds `crates/cintx-cubecl/src/math/bessel.rs`, `radial_quadrature.rs`, plus `crates/cintx-cubecl/src/kernels/ecp.rs`. Largest engineering scope. Honors PROJECT.md "CubeCL primary" constraint strictly. | ✓ |
| Host-CPU-only evaluator in cintx-compat (documented departure) | Implement Type-1/Type-2 host-side. Pragmatic — ECP isn't perf-critical. Documented departure from PROJECT.md "CubeCL primary" constraint. Smallest scope; ships fastest. | |
| Hybrid: host driver + `#[cube]` inner contractions | Host code owns ECP shell-pair loop, Bessel, radial quad; inner Gaussian primitive contractions reuse existing `#[cube]` Boys/Obara-Saika via thin glue. | |

**User's choice:** Full `#[cube]` GPU — add Bessel/radial-quad math modules
**Notes:** Largest engineering effort but honors PROJECT.md constraint strictly and
matches the precedent set by Phases 8-13 (every kernel family ships as `#[cube]`).
Risks surfaced in CONTEXT.md: cond_br MLIR limitations (Phase 8 P02 incident) and
GPU-side Bessel function evaluation are non-trivial. The Phase 8 paired `#[cube]` +
`*_host()` pattern is the template — host-side wrappers exist for unit testing
without GPU context.

---

## Phase scope: gradient variants

| Option | Description | Selected |
|--------|-------------|----------|
| Defer gradient variants to a follow-up phase (initially recommended) | Phase 19 ships only base symbols. Gradient `int1e_ecp_ipnuc_*` defer to Phase 20. | |
| Include gradient variants in Phase 19 | Ship `int1e_ecp_ipnuc_sph` and `int1e_ecp_ipnuc_cart` as well. Phase becomes ~5-6 plans. Closes issue #11 Task 1 line item completely. | ✓ |
| Include only Type-1 gradient; defer Type-2 gradient | Type-1 gradient is straightforward; Type-2 gradient needs derivative of spherical-harmonic projector and is materially harder. | |

**User's choice:** Include gradient variants in Phase 19
**Notes:** Phase 19 covers 6 symbols total (4 base + 2 gradient) in the parity
sweep. Type-1 and Type-2 gradient share one kernel launcher with internal branching
(D-11). Component rank for gradients is 3 (one derivative per Cartesian axis),
matching the existing `int3c2e_ip1_*` precedent. Closes issue #11 Task 1 line
item in a single phase.

---

## Claude's Discretion

Documented inline under `<decisions>` in CONTEXT.md. Highlights:

- Exact upstream libcint ECP source set to vendor (Likely `ecp.c` + transitive
  headers; researcher enumerates the full list).
- `EcpShell` field naming (`radial_power` vs `r_power`, `ecp_type` vs
  `projector_kind`).
- New math module function signatures (`modified_spherical_bessel_kn`,
  `gauss_chebyshev_nodes_weights`, etc.).
- Cu/LANL2DZ fixture sourcing (PySCF basis library vs basissetexchange.org vs
  Hay & Wadt 1985 JCP).
- `FacadeError::MissingEcpBasis { operator: String }` variant fields.
- Parity test fixture coverage (full Cartesian product over Cu shells).
- libECP secondary cross-check file location and gating
  (`#[ignore]` + `CINTX_LIBECP_ORACLE=1` env, matching Phase 16 ROCm precedent).
- `canonical_family = "ecp"` vs `"int1e_ecp"` vs `"ecp1e"` (default: `"ecp"`).

---

## Deferred Ideas

Captured in CONTEXT.md `<deferred>`. Cross-reference list:

- `int1e_ecp_spinor` + `int1e_ecp_ipnuc_spinor` oracle parity sweep (Type-2 ECP
  IS naturally spin-orbit-like — spinor is the physically right representation,
  deferred only because multi-component spinor transform infra needs more work).
- Higher-derivative ECP variants beyond `_ipnuc_*`.
- Lighter-atom fixture validation pass (Na/SBKJC or K/CRENBL) before Cu/LANL2DZ.
- libECP secondary cross-check promotion to CI-required gate.
- Multi-fixture / multi-pseudopotential-family parity sweep.
- `SessionRequest::with_ecp_basis(ecp)` builder pattern (D-06 keeps ECP on
  `BasisSet`; builder method revisitable if a consumer needs runtime ECP swap).
- Shared chunk-loop helper between safe API and compat raw path (still deferred
  from Phase 17 D-03 / Phase 18).
- GPU-side Bessel function performance tuning.
- Type-1 / Type-2 gradient helper extraction once other 1e gradient operators
  land.
- Pre-screening for negligible ECP shell contributions.

---

# Session 2 — K-Taylor byte-identity port replan

**Date:** 2026-05-20
**Phase:** 19-int1e-ecp-type1-type2-evaluator
**Trigger:** Plans 19-01..19-04 executed; 19-05 (gradient) halted on the K-Taylor
byte-identity blocker (halt commit `b4e1e24`). Re-ran `/gsd:discuss-phase 19`,
chose **Update it** — keep D-01..D-12, add port decisions.
**Areas offered:** Byte-identity fidelity, Table embedding & provenance, GPU
`#[cube]` vs host-first, Plan re-sequencing.
**Areas the user chose to discuss:** Table embedding & provenance (only).

---

## Table embedding & provenance

### Q1 — How should the two K-Taylor tables physically live in cintx-cubecl?

| Option | Description | Selected |
|--------|-------------|----------|
| Binary blob + `include_bytes!` | Mirror `roots_xw_data.rs`: LE f64 `.bin`, `AlignedBytes<{N*8}>`, `bytemuck::cast_slice`. Tiny diff, byte-identity by construction. | ✓ |
| Generated Rust literal arrays | `static SPH_INE_TAB: [f64; 9600] = [...]`. Readable/grep-able but ~35k literals bloat the tree; needs exact f64 round-trip. | |
| You decide | Planner picks. | |

**User's choice:** Binary blob + `include_bytes!` (→ D-14).
**Notes:** Locked constraint recorded regardless of option: byte-identity requires
PySCF's *exact* literals + its interpolation; recomputing via cintx's `bessel.rs`
series would diverge.

### Q2 — How rigorous should extraction-and-sync be?

| Option | Description | Selected |
|--------|-------------|----------|
| Regenerable + CI drift-check | xtask (`gen-ecp-tables`) parses literals from vendored `nr_ecp.c`, emits `.bin`, + CI/test gate fails on drift. Stronger than Rys precedent; mirrors manifest-lock. | ✓ |
| Checked-in blob + provenance note | One-time extraction, commit `.bin` + `.planning/notes/` provenance (source/span/sha256). Matches Rys precedent; no CI step. | |
| Checked-in blob + test-time byte-match | Commit `.bin` + a unit test re-parses `nr_ecp.c` and asserts byte-match in the oracle matrix. | |

**User's choice:** Regenerable + CI drift-check (→ D-15).
**Notes:** Finding surfaced during discussion — the Rys `roots_xw_*.bin` are
checked-in artifacts with **no** in-repo regenerator, and xtask has no
table-generation subcommand today; D-15 deliberately adds that rigor.

### Wrap-up

| Option | Description | Selected |
|--------|-------------|----------|
| Ready — write context | Write CONTEXT.md; record the 3 skipped areas as defaults. | ✓ |
| More questions on embedding | Keep deep-diving placement / drift-check assertion / format. | |
| Discuss a skipped area too | Pull fidelity / GPU-vs-host / sequencing back in. | |

**User's choice:** Ready — write context.

---

## Claude's Discretion (skipped areas recorded as defaults)

- **Byte-identity fidelity (D-13):** committed to exact `atol=1e-12` byte-identity
  (not relaxed tolerance, not "compiled-but-unverified"); supersedes D-07's
  direct-quadrature. Flagged as revisitable since not separately deliberated.
- **GPU `#[cube]` vs host-first (D-16):** default host-first to close the
  CPU-vs-C gate, `#[cube]` follows the `bessel.rs` paired pattern; host-only is a
  documented CLAUDE.md deviation, not the end state.
- **Plan re-sequencing (D-17):** default = standalone K-Taylor port plan → scalar
  close → gradient (re-done 19-05) → optional libecpint (19-06). Planner sets
  exact boundaries.

## Deferred Ideas (new this session)

- `#[cube]` GPU port of `ECPrad_part`/`ECPrad_block` if D-16 ships host-only first.
