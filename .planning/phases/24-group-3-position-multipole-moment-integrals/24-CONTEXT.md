# Phase 24: Group 3 — Position / Multipole-Moment Integrals - Context

**Gathered:** 2026-05-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Bring the full **position / multipole-moment family set** to byte-identity
(cart + sph) against the **non-zero** gauge-origin fixture — every `r`-operator
reads `env[PTR_COMMON_ORIG]`, so a zero-origin test is trivially-passing and is
explicitly disallowed.

**In scope (all entirely NEW to the manifest — none exist today):**
- **Overlap-derived position tensors:** `int1e_r`, `int1e_rr`, `int1e_rrr`,
  `int1e_rrrr`, `int1e_r2`, `int1e_r4`, `int1e_z`, `int1e_zz` (component_rank up to 81).
- **rinv group:** plain `int1e_rinv`, `int1e_drinv`.
- **Momentum / mixed:** `int1e_p4` (∇⁴), `int1e_irp` (i·r×∇).
- **`_origj` variants** of the `r`-operator families (origin at the ket center).

**Acceptance:** per-family byte-identity at **atol=1e-12** vs vendored libcint
6.1.3 for cart + sph, every component, under the vendor gate
(`--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`), on the existing
H2O/STO-3G corpus AND the non-zero gauge-origin fixture
(`build_h2o_sto3g_common_orig`, origin `[0.5,-0.3,0.8]`). Each family registered
with its `component_rank`, dispatched through `eval_raw`, with a dedicated
`vendor_*` parity test (`running N>0 tests`) and `oracle_covered=true`;
`manifest-audit` green.

**Out of scope:**
- The FND-06 fail-closed high-rank staging refactor + rank-81 OOM re-validation —
  stays **Phase 25** (see D-03). Phase 24's gate is byte-identity parity, not OOM-safety.
- The nroots≥6 Wheeler/Jacobi fallback — stays **Phase 25** (FND-02). Phase 24 families
  are fail-closed above the device Rys ceiling (nroots>5) and covered up to that ceiling
  on the corpus (see D-04).
- `capi` enum variants and legacy `cint*` wrappers — explicitly NOT added (ROADMAP SC5).
- Spinor representations — registered but `UnsupportedApi` (carry-forward D-09).

</domain>

<decisions>
## Implementation Decisions

### Family clustering & sequencing
- **D-01:** Split into **PLAN.md clusters by shared operator construction, low-rank first**
  (mirrors Phase 23's shared-kernel-reuse clustering):
  - **Cluster A — overlap-derived position tensors** (`r`, `rr`, `rrr`, `rrrr`, `r2`, `r4`,
    `z`, `zz`): built from the overlap G-tensor × position-power through **ONE parameterized
    moment kernel**, ket headroom raised via `ng[1] = 1..4`. `z`/`zz` are the single-axis
    subsets of `r`/`rr`; `r2`/`r4` are the scalar (trace / squared-trace) contractions.
    Sequence **A first** — banks MOM-01/02/03 and walks the rank-1→81 ramp through one kernel.
  - **Cluster B — rinv group** (`rinv`, `drinv`): Rys-driven (see D-04).
  - **Cluster C — `p4`** (∇⁴, overlap-derivative engine, no Rys).
  - **Cluster D — `irp`** (i·r×∇, overlap-derivative engine).
  - **`_origj` variants** land alongside their base family in Cluster A (same kernel, see D-02).
  - Sequence A → then B/C/D (parallelizable; worktrees + parallelization are on in config).

### `_origj` variant mechanism
- **D-02:** Each `_origj` family is its **own manifest operator / RawApiId** (mirrors
  libcint's symbol-per-variant set, keeps the manifest parity-complete, preserves the
  per-symbol `vendor_*` parity-test pattern). The **shared kernel branches on an
  origin-source**: `env[PTR_COMMON_ORIG]` (the base family) vs the **ket shell-j coordinate**
  (the `_origj` variant). This realizes Phase 22 D-04's "kernel-side coordinate choice" —
  do NOT collapse to a single operator + origin-mode descriptor flag.

### Rank-81 staging (Phase 24 ↔ Phase 25 boundary)
- **D-03:** **Keep the boundary — use the existing staging as-is for rank-81 parity.**
  `parse_component_multiplier` (`planner.rs:403`) already sizes staging for any
  `component_rank`; the per-element `if dst < staging.len()` scatter guards never trip when
  the buffer is sized correctly, so output is **complete and correct at rank 81**. Phase 24's
  gate is byte-identity PARITY, which does not require the FND-06 OOM-safety hardening.
  The FND-06 work (replace per-element guards with an upfront `BufferTooSmall` assertion +
  rank-81 OOM re-validation) stays **Phase 25**. Cross-link the dependency in the plan.

### Non-tensor operators (rinv / drinv / p4 / irp)
- **D-04:** **Reuse existing kernels; fail-closed above the device Rys ceiling.**
  - Plain `int1e_rinv` = the `int1e_nuc` Rys kernel evaluated at the **common origin**
    (`env[PTR_COMMON_ORIG]`) with **charge = 1 and NO atom-sum** (libcint `int1e_rinv` is a
    single-center 1/r potential, not the nuclear sum).
  - `int1e_drinv` = its derivative (+1 Rys root vs `rinv`).
  - `int1e_p4` (∇⁴) and `int1e_irp` (i·r×∇) reuse the **overlap-derivative engine** (no Rys).
  - Cover up to the **nroots≤5 ceiling** (≤ f) on the existing corpus; **fail closed when
    `nroots > 5`** (Phase 23 D-13 precedent). The corpus (H2O/STO-3G ≤ d) does not hit the
    ceiling, so coverage is full on the gate fixtures.

### Carry-forward locks (from Phases 22/23 — do NOT re-litigate)
- **D-05 (registration recipe, Phase 23 D-11):** 5 steps land a new family — (1) manifest lock
  entry cloning the closest family with `component_rank` = true output multiplier, then
  `cargo build -p cintx-ops` regenerates `api_manifest.{rs,csv}`; (2) `RawApiId` consts in
  `cintx-compat/src/raw.rs`; (3) launcher dispatch on `descriptor.operator_name()`; (4) vendor
  FFI — add cart/sph symbols to the bindgen `allowlist_function` regex in
  `cintx-oracle/build.rs` + safe wrappers in `vendor_ffi.rs` (confirm the autocode `.c` is in
  the build source list); (5) `vendor_*` parity test. **Lock edits auto-sync `manifest-audit`
  — there is NO separate fixtures family list to edit.**
- **D-06 (gauge-origin plumbing READY, Phase 22):** `env[1..3]` is read unconditionally in
  `eval_raw` (`raw.rs:674-686`), `.with_common_origin([x,y,z])` exists on the builder, the
  finiteness validator runs on both paths, `common_orig == None` defaults to `[0,0,0]`. Phase 24
  is the **first kernel consumer** of the slot. `PTR_COMMON_ORIG = 1` (`raw.rs:50`).
- **D-07 (transpose discipline, Phase 23 D-05 + ROADMAP SC1):** raise angular-momentum
  headroom on the **ket** (`ng[1]`), **not** the bra; copy each family's component order
  **verbatim from the libcint gout index map**; **gate every family with a NON-SQUARE bra×ket
  block** (e.g. p×d) so a transposed layout cannot pass. A square block is transpose-symmetric
  and hides the bug.
- **D-08 (component-rank-truncation hard rule, Phase 23 D-14):** a `component_rank` set too LOW
  silently TRUNCATES trailing output components. Each family's `component_rank` MUST equal its
  true output multiplier (`r`=3, `rr`=9, `rrr`=27, `rrrr`=81, `r2`=1, `r4`=1, `z`=1, `zz`=1,
  `p4`=1, `irp`=… — **derive exact values from libcint source**, do not guess; see open items).
- **D-09 (surface scope, Phase 23 D-09):** manifest + RawApiId + kernel + vendor-FFI + oracle
  only. No `capi` enum variants, no legacy `cint*` wrappers. Spinor reps registered →
  `UnsupportedApi`.
- **D-10 (verification, Phase 23 D-10):** per-family byte-identity at **atol=1e-12** vs vendored
  libcint 6.1.3, cart + sph, every component, in `vendor_*` parity tests double-gated on
  `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1` (without both, parity silently skips).

### Folded Todos
- **`oracle-cart-offset-vendor-zero`** (`.planning/todos/pending/oracle-cart-offset-vendor-zero.md`,
  medium): 4 `compare::tests` **lib** unit tests fail under the vendor gate at
  `CINTshells_cart_offset[4] cintx=8 vendor=0` (`compare.rs ~645`). Integration parity passes;
  hypothesis is a harness/env bug (vendor FFI `ao_loc` returns 0 in the lib-unit context).
  **Folded because Phase 24 runs the vendor lib+integration gate and this WILL re-surface.**
  In-scope action: confirm against a pre-phase-20 commit; if reproduced, either fix the
  lib-test harness fixture or convert to a tracked standalone oracle-harness bug so the Phase 24
  gate is not blocked by pre-existing noise. (Related: vendor-gated `--lib` tests are otherwise
  uncovered by the routine `--features cpu` CI gate.)
- **`rys-nroots-ge6-wheeler-fallback`** (`.planning/todos/pending/rys-nroots-ge6-wheeler-fallback.md`,
  medium, `resolves_phase: 25`): `rys_roots_host` panics for `nroots ≥ 6` (`rys.rs:3255`).
  **Folded as a cross-link / boundary marker only** — Phase 24 does NOT implement the Wheeler
  fallback (it belongs to Phase 25 FND-02). The fold records that Phase 24's rinv/drinv (and any
  high-L moment) are **capped at the nroots≤5 ceiling and fail-closed above it** (D-04), and that
  the corpus does not reach the ceiling. If a Phase 24 family unexpectedly needs `nroots≥6` on
  the gate corpus, that is the trigger to escalate the dependency to Phase 25, not to widen this
  phase.

### Claude's Discretion
- Exact `component_rank` value and libcint gout component order per family (derived from
  libcint source by researcher/planner; see Open Items).
- Whether Cluster A's parameterized moment kernel is one `#[cube]` entry with a comptime moment
  order or a small family of order-specialized launchers — implementer's call, as long as D-07
  (ket headroom, verbatim gout order, non-square gate) holds.
- The precise corpus shell-tuple selection for each `vendor_*` test (subject to the non-square
  bra×ket requirement of D-07).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & roadmap
- `.planning/REQUIREMENTS.md` §MOM-01..04 (L102-105) — the requirements this phase satisfies.
- `.planning/ROADMAP.md` — Phase 24 entry (Goal, Success Criteria 1-5, Depends-on Phase 22).
- `.planning/research/SUMMARY-v1.4.md` — milestone research source cited by REQUIREMENTS.

### Carry-forward context (the proven precedents this phase reuses)
- `.planning/phases/23-group-1-remaining-1st-derivative-families-cart-sph/23-CONTEXT.md` —
  the registration recipe (D-11), transpose discipline (D-05), component-rank-truncation rule
  (D-14), surface scope (D-09), verification gate (D-10), rinv-deriv fail-closed precedent (D-13).
- `.planning/phases/22-gauge-origin-env-slot-gap-a-ptr-common-orig/22-CONTEXT.md` —
  `PTR_COMMON_ORIG` slot semantics (D-01 finiteness/default-zero), operator-agnostic read (D-02),
  `_origj` framed as kernel-side coordinate choice (D-04).
- `.planning/phases/21-coulomb-gradient-intors/21-CONTEXT.md` — `PTR_RINV_ORIG` env-slot
  precedent + the Phase-21 gradient/overlap engine the moment kernel extends.

### Code anchors (from scout)
- `crates/cintx-cubecl/src/kernels/one_electron.rs` — scalar 1e `#[cube]` kernels;
  `launch_one_electron_typed` (:3725) operator dispatch (:3765-3790), device
  `one_electron_scalar_kernel` (:200) with `#[comptime] op_kind`; nuclear/Rys path (:415-503)
  to clone for plain `rinv`.
- `crates/cintx-compat/src/raw.rs` — `PTR_COMMON_ORIG` const (:50); `eval_raw` env-read +
  validate block (:674-686); `RawApiId` const pattern (:122-243); launcher dispatch (:612-822).
- `crates/cintx-runtime/src/planner.rs` — `parse_component_multiplier` (:403-453) sizing staging
  from `component_rank`; `OperatorEnvParams.common_orig`.
- `crates/cintx-runtime/src/validator.rs` — `validate_common_orig_env_params` (:210-223).
- `crates/cintx-rs/src/builder.rs` — `.with_common_origin([x,y,z])` (:107-115).
- `crates/cintx-oracle/src/fixtures.rs` — `build_h2o_sto3g_common_orig()` (:158),
  `COMMON_ORIG_FIXTURE_ORIGIN = [0.5,-0.3,0.8]` (:152).
- `crates/cintx-oracle/build.rs` — bindgen `allowlist_function` regex + autocode `.c` source
  list (vendor FFI step of the recipe).
- `crates/cintx-ops/generated/compiled_manifest.lock.json` — the lock to extend (all moment
  families are new).

### libcint upstream (researcher must derive exact gout orders + ranks from these)
- libcint 6.1.3 sources for the moment families: the `int1e_r*`/`int1e_z*`/`int1e_rinv`/
  `int1e_drinv`/`int1e_p4`/`int1e_irp` gout emitters and `ng[]` headroom tuples — the source of
  truth for component order (D-07) and `component_rank` (D-08). Vendored under the oracle build.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Overlap G-tensor + Phase-21 derivative engine** (`one_electron.rs`): Cluster A's moment
  kernel is overlap-tensor × position-power; `p4`/`irp` (Cluster C/D) reuse the overlap-derivative
  machinery (no Rys).
- **Nuclear/Rys 1e kernel** (`one_electron.rs:415-503`): plain `int1e_rinv` clones this with
  charge=1 and the atom-sum dropped, origin = `env[PTR_COMMON_ORIG]`; `drinv` adds one Rys root.
- **Gauge-origin slot** fully plumbed (Phase 22) — Phase 24 is the first kernel to read it.
- **`parse_component_multiplier`** auto-sizes staging from `component_rank` up to rank 81 — no
  manual layout code (D-03).
- **Registration recipe** proven on Phase 23 cluster C — manifest-audit auto-syncs from the lock.

### Established Patterns
- Operator dispatch in `launch_one_electron_typed` is a flat `is_<op> = op_name == "<op>"` ladder
  (:3765) + a rejection guard; new moment ops slot in here (or via a new `op_kind` comptime value).
- `vendor_*` parity tests double-gated on `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`.
- Component order copied verbatim from libcint gout, gated with a NON-SQUARE bra×ket block.

### Integration Points
- `eval_raw` already reads + validates `common_orig`; moment kernels consume
  `plan.operator_env_params.common_orig.unwrap_or([0.0;3])`.
- `_origj` kernels read the ket shell-j coordinate instead of the env slot (D-02).

</code_context>

<specifics>
## Specific Ideas

- The non-zero gauge-origin fixture (`build_h2o_sto3g_common_orig`, origin `[0.5,-0.3,0.8]`) is
  the **mandatory parity gate** for every `r`-operator family — a zero-origin test is
  doubly-trivial and disallowed (ROADMAP SC1).
- The dipole `int1e_r` regression must confirm the result is **not transposed** by raising
  headroom on `ng[1]` (ket) and verifying against a non-square block (ROADMAP SC1 + D-07).

## Open Items for Research/Planning (not user decisions)
- Exact `component_rank` per family and the libcint gout component-index order for
  `rr`/`rrr`/`rrrr`/`irp` (the Cartesian nesting / fastest-varying axis) — derive verbatim from
  libcint source; gate with the non-square block.
- Confirm `int1e_p4` headroom (bra ∇⁴ → bra+? ) and whether it stays within the executor
  `ang_momentum>4` gate on the corpus.
- Confirm whether any `_origj` family has a distinct vendor symbol vs a libcint runtime origin
  parameter (affects the vendor FFI wrapper in recipe step 4).

</specifics>

<deferred>
## Deferred Ideas

- **FND-06 fail-closed high-rank (rank-81) staging refactor + OOM re-validation** → Phase 25.
  Phase 24 uses existing staging as-is for parity (D-03).
- **nroots≥6 Wheeler/Jacobi fallback** (FND-02) → Phase 25. Phase 24 fail-closes above the
  nroots≤5 ceiling (D-04); folded as a cross-link only.
- **Spinor moment representations** → land when a consumer needs them; registered →
  `UnsupportedApi` this phase (D-09).

</deferred>

---

*Phase: 24-group-3-position-multipole-moment-integrals*
*Context gathered: 2026-05-30*
