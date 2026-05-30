# Phase 25: Group 2 — Hessian & Higher-Order Derivatives - Context

**Gathered:** 2026-05-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Bring the **2nd/3rd/4th-order derivative (Hessian+) family set** to byte-identity
(cart + sph) at **atol=1e-12** vs vendored libcint 6.1.3, at component_rank
9/27/81 — **plus two foundational unblockers that land first** (FND-02, FND-06).

**Foundations (land before any family work):**
- **FND-02 — Rys `nroots≥6` Wheeler/Jacobi fallback:** `rys_roots_host` panics for
  `nroots ≥ 6` today (`rys.rs:3255`). Implement the high-nroots root+weight scheme so
  no family returns `UnsupportedApi` purely because `nroots>5`, and extend the
  `executor.rs` `ang_momentum>4` gate (`executor.rs:140-142`) to admit higher-l shells
  the validated roots support.
- **FND-06 — fail-closed high-rank staging:** replace the per-element
  `if dst < staging.len()` scatter guards (silent partial-write risk) with an upfront
  size assertion, and re-validate the chunk planner's OOM-safe-stop at rank 81.

**Hessian+ families (all NEW to the manifest except the two noted stubs):**
- **HESS-01 (rank 9):** `int1e_ipipovlp`, `int1e_ipipnuc`, `int1e_ipipkin`,
  `int1e_ipiprinv` — per-family `ng[]` headroom (bra +2) drives G-tensor sizing.
- **HESS-02 (2e Hessian set):** `int2e_ipip1`, `int2e_ipvip1` (exist today as
  `unstable::source::2e`, sph-only, `oracle_covered=false`, no rank — to be promoted),
  plus `int2e_ip1ip2`, `int2e_ipip1ipip2` (entirely new).
- **HESS-03:** `int2c2e_ipip1`, `int3c2e_ipip1`, `int3c2e_ipip2`.
- **HESS-04 (3rd/4th-order):** `int1e_ipipipnuc`, `int1e_ipipipiprinv`, and the sibling
  `ipipip*` families — `ng[]`-driven bra+ket headroom (deriv4 raises bra +2 AND ket +2),
  component_rank 27/81.

**Acceptance:** per-family byte-identity at **atol=1e-12** vs vendored libcint 6.1.3
for cart + sph, every component, under the vendor gate (`--features cpu` +
`CINTX_ORACLE_BUILD_VENDOR=1`). Each family registered with its true `component_rank`,
dispatched through `eval_raw`, with a dedicated `vendor_*` parity test (`running N>0 tests`)
and `oracle_covered=true`; `manifest-audit` green. `deriv3.c`/`deriv4.c` added to the
oracle `cc::Build` with suppl-header `extern` decls + allowlist entries.

**Out of scope:**
- `capi` enum variants and legacy `cint*` wrappers — explicitly NOT added (ROADMAP SC7, D-09).
- Spinor representations — registered but `UnsupportedApi` (carry-forward D-09).
- Heavy-element ECP/lanthanide validation — FND-02 lifts the numerical ceiling but the
  f-projector ECP validation (todo step 3) is a later phase, not this one.

</domain>

<decisions>
## Implementation Decisions

### FND-02 — Rys nroots≥6 Wheeler/Jacobi fallback
- **D-01 (fidelity — port verbatim):** Reach byte-identity for nroots 6..~13 by
  **porting libcint 6.1.3's own high-nroots numerical path verbatim** (the
  modified-moments → tridiagonal/Jacobi-matrix → root-polish scheme in libcint's
  `rys_roots.c` for n>5, NOT the hardcoded low-n polynomial fits). Implement it
  **host-side**, mirroring the ECP K-Taylor "port the exact upstream machinery
  host-first" precedent (Phase 19). A clean-room Golub-Welsch was explicitly rejected
  for last-ULP / root-ordering byte-identity risk. The researcher derives the exact
  scheme from libcint `rys_roots.c`; where math/impl diverge, default to the faithful port.
- **D-02 (validation range + gate opening):** Implement the general algorithm; add a
  **dedicated vendor parity test sweeping nroots 6..~13** against libcint. Extend the
  `executor.rs` `ang_momentum>4` gate (`:140-142`) to admit **exactly the max angular
  momentum the roots are validated for** (g/h/i as covered) — forward-looking foundation
  per ROADMAP SC1. Not "minimal corpus-only" and not "unbounded above the validated range."
- **D-03 (corpus reality):** Phase 25's own gate corpus (H2O/STO-3G, ≤ d) can push
  Hessian-elevated d-shells to nroots 6 (the in-phase trigger for FND-02), but never
  reaches g/h. The g/h gate extension is therefore forward-looking foundation work, not
  exercised by Phase 25's own families — validate it on the dedicated nroots sweep, not
  only the family parity tests.

### FND-06 — fail-closed high-rank staging
- **D-04 (single upfront assertion + strip all guards):** Add **one** upfront
  `BufferTooSmall`-style size assertion at the staging-allocation boundary in
  `planner.rs` (where `parse_component_multiplier` already sizes staging by
  `component_rank`), then **remove the per-element `if dst < staging.len()` scatter
  guards across ALL kernels** (`one_electron.rs`, `two_electron.rs`, `center_3c2e.rs`,
  `center_2c2e.rs`, `f12.rs`, `unstable/*`) so scatter is unconditional once the buffer
  is proven large enough. One contract point; no silent partial writes anywhere. NOT a
  rank≥9-only partial strip, and NOT a per-launcher assertion.
- **D-05 (rank-81 OOM re-validation):** Add a **dedicated new test** that sets a memory
  limit smaller than rank-81 staging requires, then asserts a typed OOM/`BufferTooSmall`
  stop with **NO partial write** (output buffer untouched). Exercises the new upfront
  assertion + the existing `ChunkPlanner` OOM-safe-stop together. (Aligns with the
  CLAUDE.md "fallible allocation + typed failure + no partial writes" non-negotiable.)

### Sequencing & plan clustering
- **D-06 (two foundation plans, then clustered families):**
  - **Plan 1 = FND-02** (Wheeler nroots≥6 + executor l-gate extension).
  - **Plan 2 = FND-06** (planner-upfront assertion + guard strip + rank-81 OOM test).
  - **Both foundation plans must merge before any family plan starts.**
  - Then family clusters **low-rank-first** (mirrors Phase-24 D-01 shared-construction
    clustering):
    - **Cluster A** — `int1e` rank-9 (`ipipovlp`, `ipipnuc`, `ipipkin`, `ipiprinv`) (HESS-01).
    - **Cluster B** — 2e Hessian set (`int2e_ipip1`, `ipvip1`, `ip1ip2`, `ipip1ipip2`) (HESS-02).
    - **Cluster C** — `int2c2e_ipip1`, `int3c2e_ipip1`, `int3c2e_ipip2` (HESS-03).
    - **Cluster D** — 3rd/4th-order `ipipip*` (HESS-04).
  - Family clusters parallelize via **worktrees** once foundations land (worktree
    parallelization is on in config). Confirm post-wave integration with
    `merge-base --is-ancestor` (worktree auto-merge is inconsistent — see memory).

### HESS-02 — 2e Hessian promotion from `unstable`
- **D-07 (re-home to stable, drop unstable entries):** **Move** `int2e_ipip1` /
  `int2e_ipvip1` out of `unstable::source::2e` into the **stable family/raw-api map**
  (add the cart representation, set `component_rank`, wire the stable launcher + vendor
  FFI + `vendor_*` test, flip `oracle_covered=true`); register `int2e_ip1ip2` /
  `int2e_ipip1ipip2` **fresh** in the same stable family. The unstable sph-only stubs are
  **removed** so there is exactly **one canonical stable entry per symbol** (no duplicate
  entry, no lingering `unstable` feature-gate on these symbols). NOT an in-place extend +
  alias.

### Carry-forward locks (from Phases 21–24 — do NOT re-litigate)
- **D-08 (registration recipe, P23 D-11 / P24 D-05):** 5 steps land a new family —
  (1) manifest lock entry cloning the closest family with `component_rank` = true output
  multiplier, then `cargo build -p cintx-ops` regenerates `api_manifest.{rs,csv}`;
  (2) `RawApiId` consts in `cintx-compat/src/raw.rs`; (3) launcher dispatch on
  `descriptor.operator_name()`; (4) vendor FFI — add cart/sph symbols to the bindgen
  `allowlist_function` regex in `cintx-oracle/build.rs` + safe wrappers in `vendor_ffi.rs`
  (confirm the autocode `.c` — here `deriv3.c`/`deriv4.c` — is in the build source list);
  (5) `vendor_*` parity test. **Lock edits auto-sync `manifest-audit` — there is NO
  separate fixtures family list to edit.**
- **D-09 (transpose discipline, P23 D-05 / P24 D-07 + ROADMAP SC):** raise
  angular-momentum headroom on the **ket** (`ng[]`), **not** the bra; copy each family's
  component order **verbatim from the libcint gout index map**; **gate every family with a
  NON-SQUARE bra×ket block** (e.g. p×d) so a transposed layout cannot pass. (For deriv4,
  headroom is raised on BOTH bra +2 and ket +2 per HESS-04.)
- **D-10 (component-rank-truncation hard rule, P23 D-14 / P24 D-08):** a `component_rank`
  set too LOW silently TRUNCATES trailing output components. Each family's
  `component_rank` MUST equal its true output multiplier (`ipip*`=9, 3rd-order=27,
  4th-order=81 — **derive exact values from libcint source**, do not guess).
- **D-11 (surface scope, P23 D-09 / P24 D-09):** manifest + RawApiId + kernel + vendor-FFI
  + oracle only. No `capi` enum variants, no legacy `cint*` wrappers. Spinor reps
  registered → `UnsupportedApi`.
- **D-12 (verification, P23 D-10 / P24 D-10):** per-family byte-identity at **atol=1e-12**
  vs vendored libcint 6.1.3, cart + sph, every component, in `vendor_*` parity tests
  double-gated on `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1` (without both, parity
  silently skips).

### Claude's Discretion
- Exact `component_rank` value and libcint gout component-index order per family (derived
  from libcint source by researcher/planner; gate with the non-square block).
- The precise libcint `rys_roots.c` routine names/structure for the n>5 path and how
  literally the host port mirrors libcint's control flow (as long as D-01 byte-identity
  on the nroots 6..~13 sweep holds).
- The exact enumerated set of HESS-04 "and siblings" 3rd/4th-order families — derive the
  complete set that libcint 6.1.3 actually exports (`int1e_ipipip*`, `ipipipip*` siblings)
  from the vendored sources; do not guess the roster.
- The precise corpus shell-tuple selection for each `vendor_*` test (subject to the
  non-square bra×ket requirement of D-09).
- Whether Cluster A/C/D moment kernels are one parameterized `#[cube]` entry with a
  comptime derivative order or order-specialized launchers — implementer's call, as long
  as D-09 (ket/bra headroom, verbatim gout order, non-square gate) holds.

### Folded Todos
- **`rys-nroots-ge6-wheeler-fallback`**
  (`.planning/todos/pending/rys-nroots-ge6-wheeler-fallback.md`, medium,
  `resolves_phase: 25`): `rys_roots_host` panics for `nroots ≥ 6` (`rys.rs:3255`); the
  `executor.rs` `ang_momentum>4` gate (`:140-142`) compounds it by rejecting l>4 shells.
  **Folded as the core of FND-02 (D-01/D-02/D-03).** Phase 25 implements steps 1–2 of the
  todo's fix (Wheeler nroots≥6 port + executor gate extension, validated nroots 6..~13);
  step 3 (lanthanide f-projector ECP validation) is explicitly a **later** heavy-element
  phase, not this one.
- **`oracle-cart-offset-vendor-zero`**
  (`.planning/todos/pending/oracle-cart-offset-vendor-zero.md`, medium): 4 `compare::tests`
  **lib** unit tests fail under the vendor gate at `CINTshells_cart_offset[4] cintx=8
  vendor=0` (`compare.rs ~645`); integration parity passes; hypothesis is a harness/env bug
  (vendor FFI `ao_loc` returns 0 in the lib-unit context). **Folded as a vendor-gate-hygiene
  cross-link** (same handling as Phase 24): Phase 25 runs the vendor lib+integration gate, so
  this WILL re-surface. In-scope action: confirm it is **pre-existing** (reproduce against a
  pre-phase-20 commit); if reproduced, either fix the lib-test harness fixture or convert to a
  tracked standalone oracle-harness bug — **do NOT let it block the Phase 25 family gate.**

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & roadmap
- `.planning/REQUIREMENTS.md` §FND-02, §FND-06 (L79, L83) and §HESS-01..04 (L95-98) —
  the requirements this phase satisfies.
- `.planning/ROADMAP.md` — Phase 25 entry (L570-583): Goal, Success Criteria 1-7,
  Depends-on Phase 23 (first-order engine) + Phase 24 (ket-headroom plumbing), and the
  research flag (FND-02 milestone-level decision — resolved here by D-01/D-02).
- `.planning/research/SUMMARY-v1.4.md` — milestone research source cited by REQUIREMENTS.

### Carry-forward context (the proven precedents this phase reuses)
- `.planning/phases/24-group-3-position-multipole-moment-integrals/24-CONTEXT.md` —
  the family-clustering pattern (D-01), registration recipe (D-05), transpose discipline
  (D-07), component-rank-truncation rule (D-08), surface scope (D-09), verification gate
  (D-10); and the Phase-24↔25 boundary it set (D-03: FND-06 staging refactor + rank-81 OOM
  → Phase 25; D-04 deferred-comment: nroots≥6 Wheeler fallback → Phase 25).
- `.planning/phases/23-group-1-remaining-1st-derivative-families-cart-sph/23-CONTEXT.md` —
  the first-order `nabla`/`gout_ipN` engine this phase composes to 2nd+ order; the
  registration recipe (D-11), transpose discipline (D-05), component-rank-truncation rule
  (D-14), rinv-deriv fail-closed precedent (D-13).
- `.planning/phases/19-int1e-ecp-type1-type2-evaluator/` (CONTEXT) — the "port the exact
  upstream machinery host-first" precedent (ECP K-Taylor) that D-01 mirrors for the
  high-nroots Rys port.

### Code anchors (from scout)
- `crates/cintx-cubecl/src/math/rys.rs` — `rys_roots_host` nroots≤5 ceiling +
  `panic!("nroots > 5 not supported")` (`:3255`); deferred-Wheeler module notes
  (`:10`, `:3247`, `:3520`, `:3534`). The FND-02 port target.
- `crates/cintx-cubecl/src/executor.rs:140-142` — `ang_momentum > 4` → `max(l)>4`
  rejection gate. The FND-02 gate-extension target (D-02).
- `crates/cintx-runtime/src/planner.rs` — `parse_component_multiplier` (~`:403`) sizes
  staging from `component_rank`; staging-alloc boundary at `:509`/`:1005-1007`. The FND-06
  upfront-assertion site (D-04).
- Per-element scatter guards to strip (FND-06 D-04): `one_electron.rs:6545,6569,6736,6760,
  6973,7028`; `two_electron.rs:1600,1641,1845,1886,2173,2231`; `center_3c2e.rs:2525,2559,
  2767,2801`; `center_2c2e.rs:736,761`; `f12.rs:1784`; `unstable/grids.rs:1521`.
- `crates/cintx-ops/generated/compiled_manifest.lock.json` — the lock to extend; current
  state: only `int2e_ipip1_sph` + `int2e_ipvip1_sph` exist (family `unstable::source::2e`,
  `oracle_covered=false`, empty `component_rank`). All other Phase-25 families are new (D-07).
- `crates/cintx-compat/src/raw.rs` — `RawApiId` const pattern + `eval_raw` launcher
  dispatch (the registration-recipe sites, D-08); the unstable→stable raw-api map move (D-07).
- `crates/cintx-oracle/build.rs` — bindgen `allowlist_function` regex + autocode `.c`
  source list; **must add `deriv3.c`/`deriv4.c`** + suppl-header `extern` decls (ROADMAP SC7).
- `crates/cintx-oracle/src/vendor_ffi.rs` — vendor FFI safe wrappers (recipe step 4).
- `crates/cintx-cubecl/src/kernels/two_electron.rs` — the 2e ERI engine the Hessian set
  (HESS-02) elevates; `crates/cintx-cubecl/src/kernels/unstable/` — source of the
  `int2e_ipip1/ipvip1` stubs being re-homed (D-07).

### libcint upstream (researcher must derive exact gout orders, ranks, and the n>5 Rys scheme from these)
- libcint 6.1.3 `rys_roots.c` — the n>5 root+weight scheme to port verbatim (D-01).
- libcint 6.1.3 `deriv3.c` / `deriv4.c` + autocode gout emitters and `ng[]` headroom tuples
  for the `ipip*`/`ipipip*` families — source of truth for component order (D-09) and
  `component_rank` (D-10). Vendored under the oracle build.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **First-order nabla engine (Phase 23):** the `nabla1*_2e` + `gout_ipN` / `Nabla1Center{I,J,K,L}`
  machinery composes to 2nd+ order — Hessian families are the first-order engine applied twice.
- **`parse_component_multiplier` (`planner.rs`):** auto-sizes staging from `component_rank`
  up to rank 81 — no manual layout code; the FND-06 assertion slots in at this boundary (D-04).
- **Overlap-derivative engine (`one_electron.rs`):** `int1e_ipipovlp`/`ipipkin` reuse it
  (no Rys); `int1e_ipipnuc`/`ipiprinv` ride the nuclear/Rys 1e path (which FND-02 unblocks
  for nroots≥6).
- **2e ERI engine (`two_electron.rs`):** the HESS-02 set elevates this; the Rys path here
  is the primary FND-02 consumer on the corpus (Hessian elevation can push d-quartets to nroots 6).
- **Registration recipe** proven on Phases 23/24 — `manifest-audit` auto-syncs from the lock.

### Established Patterns
- Operator dispatch is a flat `is_<op> = op_name == "<op>"` ladder + rejection guard in the
  per-family launcher; new `ipip*` ops slot in here (or via a new comptime `op_kind`/deriv-order).
- `vendor_*` parity tests double-gated on `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`.
- Component order copied verbatim from libcint gout, gated with a NON-SQUARE bra×ket block.
- Host-first port of adaptive/iterative upstream numerics (ECP K-Taylor → now the n>5 Rys roots).

### Integration Points
- FND-02: `rys_roots_host` (`rys.rs`) ← consumed by the nuclear 1e (`one_electron.rs`) and
  2e/3c2e Rys paths; the `executor.rs` l-gate sits upstream of all of them.
- FND-06: the planner staging-alloc boundary (`planner.rs`) ← the single new assertion point;
  every kernel's scatter loop consumes the proven-sized buffer (guards removed).
- HESS-02: `unstable::source::2e` (`unstable/`) → stable raw-api map (`raw.rs`) — the re-home.

</code_context>

<specifics>
## Specific Ideas

- **FND-02 is the gating risk** and was the roadmap's flagged milestone-level decision —
  resolved here: faithful verbatim port of libcint's n>5 path, host-side, validated on a
  dedicated nroots 6..~13 vendor sweep (D-01/D-02). Treat Plan 1 as the long-pole.
- The 2e Hessian families on the H2O/STO-3G corpus are the **in-phase trigger** for FND-02
  (d-shell Hessian elevation → nroots 6); the g/h gate extension is forward-looking and is
  validated on the nroots sweep, not the family parity tests (D-03).
- One canonical stable manifest entry per symbol for the promoted 2e set — the unstable
  sph-only stubs are deleted, not aliased (D-07).

## Open Items for Research/Planning (not user decisions)
- Derive the exact libcint `rys_roots.c` n>5 routine(s) and replicate the control flow for
  byte-identity (D-01). Confirm the nroots upper bound the corpus + foreseeable bases need (~13).
- Enumerate the complete HESS-04 3rd/4th-order roster from libcint 6.1.3 (`int1e_ipipip*`,
  `ipipipip*` siblings) — do not guess.
- Derive exact `component_rank` (9/27/81) and gout component order per family from
  `deriv3.c`/`deriv4.c`; confirm the `ng[]` headroom tuples (deriv4 = bra+2 AND ket+2).
- Confirm `deriv3.c`/`deriv4.c` filenames + suppl-header `extern` decls in the oracle
  `cc::Build` (ROADMAP SC7).

</specifics>

<deferred>
## Deferred Ideas

- **Lanthanide / f-projector ECP validation** (step 3 of the `rys-nroots-ge6` todo) — FND-02
  lifts the numerical nroots ceiling, but validating an f-block ECP+basis byte-identical to
  upstream PySCF is a **later heavy-element phase**, not Phase 25.
- **Spinor Hessian representations** — registered → `UnsupportedApi` this phase (D-11); land
  when a consumer needs them and the Gap B1/B2 spinor-derivative transforms (Phases 27/28) exist.
- **g/h-basis end-to-end family coverage** — Phase 25 opens the l-gate and validates the roots,
  but no Phase-25 family is exercised at g/h on the corpus; full g/h family parity rides future
  heavy-element work.

### Reviewed Todos (not folded)
None — both matched todos were folded (the Wheeler fallback as FND-02 core; the
oracle-cart-offset as a vendor-gate-hygiene cross-link).

</deferred>

---

*Phase: 25-group-2-hessian-higher-order-derivatives*
*Context gathered: 2026-05-30*
