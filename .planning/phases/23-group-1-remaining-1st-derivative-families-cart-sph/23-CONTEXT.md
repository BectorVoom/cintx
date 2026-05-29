# Phase 23: Group 1 — Remaining 1st-Derivative Families (cart/sph) - Context

**Gathered:** 2026-05-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Bring the **8 remaining plain first-derivative integral families** to byte-identity
(cart + sph) by extending the Phase-21 `gout_ip1` / `nabla1i/j/k` engine to the
ket-side, remaining-center, and both-side derivatives. **Zero new foundations** —
no new math, no new Rys roots, no new env slots.

**In scope (the 8 families):**
- **Ket / remaining-center (rank 3):** `int2e_ip2` (∇ on ket bra-center), `int3c2e_ip2`,
  `int2c2e_ip1`, `int2c2e_ip2`, `int3c1e_ip1`, `int3c1e_iprinv`.
- **Both-side (rank 9, ∇ on bra × ∇ on ket):** `int1e_ipovlpip`, `int1e_ipkinip`, `int1e_ipnucip`.

**Acceptance:** byte-identity at **atol=1e-12** vs vendored libcint 6.1.3 for cart + sph,
every component, under the vendor gate (`--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`),
on the existing H2O/STO-3G (+ Cu/LANL2DZ) corpus. Each family registered with its
`component_rank`, dispatched through `eval_raw`, with a dedicated `vendor_*` parity test
(`running N>0 tests`) and `oracle_covered=true`; `manifest-audit` green.

**Out of scope:**
- The nroots≥6 Wheeler/Jacobi fallback and the `executor.rs` `l>4` gate raise — assigned
  to **Phase 25** (todo `rys-nroots-ge6-wheeler-fallback`). This caps 2e/3c/2c coverage at d.
- `capi` enum variants and legacy `cint*` wrappers — explicitly NOT added (ROADMAP SC6).
- Spinor gradient kernels — registered but `UnsupportedApi` (see D-08), not implemented.

</domain>

<decisions>
## Implementation Decisions

### Family grouping & execution sequencing
- **D-01:** Split the 8 families into **~3 PLAN.md clusters by shared kernel reuse**, not
  one-per-family: (A) ket/remaining-center rank-3 reusing `nabla1j/k_2e`
  [`int2e_ip2`, `int3c2e_ip2`, `int2c2e_ip1`, `int2c2e_ip2`]; (B) the 3c1e pair
  [`int3c1e_ip1`, `int3c1e_iprinv`]; (C) both-side rank-9 1e
  [`int1e_ipovlpip`, `int1e_ipkinip`, `int1e_ipnucip`]. Each cluster is a coherent
  kernel-reuse unit. (parallelization + worktrees are on in config.)
- **D-02:** **Sequence rank-3 clusters (A, B) first, the rank-9 cluster (C) last.** The
  rank-3 families are pure Phase-21 reuse (lowest risk) and bank 4 of 5 requirements quickly;
  the rank-9 both-side composition is the only novel glue and lands last.

### Angular-momentum coverage (max-within-ceiling per family)
- **D-03:** Target the **maximum each family reaches within the existing nroots≤5 ceiling** —
  no Wheeler work:
  - **2-center 1e both-side** (`ipovlpip`/`ipkinip`/`ipnucip`): cover up to **f**.
    `nroots = (li+1 + lj+1)/2 + 1` → ff = 5, within the ceiling.
  - **2e / 3c / 2c group**: cover up to **d** — the 4-center L sum plus the derivative `+1`
    overflows past d, the same wall Phase-21 `int2e_ip1` hit.
  - The `executor.rs:11` `ang_momentum > 4` gate blocks g/h everywhere regardless — out of scope.

### Both-side rank-9 layout (the only new glue)
- **D-04:** The 9 = 3×3 components of `int1e_*ip` come from composing `CINTnabla1i_1e`
  (bra) and `CINTnabla1j_1e` (ket) — both already exist in `one_electron.rs` (:1864, :1772).
  Register `"component_rank":"9"`; the planner's `parse_component_multiplier`
  (`planner.rs:403`) auto-allocates `9 × ni × nj` staging — no manual layout code.
- **D-05:** **Derive the 9-component ordering directly from libcint's
  `CINTgout1e_int1e_ipovlpip` source** (match its index nesting verbatim — which derivative
  index is fastest-varying), and **gate it with a deliberately NON-SQUARE bra/ket block
  (e.g. p×d)** so a transposed layout cannot pass. (Same discipline that caught the
  spinor-orientation bug; a square block is transpose-symmetric and hides the error.)

### Spinor representation policy
- **D-06:** Spinor reps for all 8 families are **registered in the manifest but return
  `UnsupportedApi`** (mirrors Phase 21 D-03 / Risk R5). Keeps the manifest complete and
  `manifest-audit` consistent; spinor gradients land when a consumer needs them.
  pyscf_rs needs only sph/cart.

### Carry-forward locks (from Phase 21 — do NOT re-litigate)
- **D-07:** Reuse the Phase-21 gradient engine **verbatim**: `gout_ip1` +
  `nabla1i_2e`/`nabla1j_2e`/`nabla1k_2e` (`f12.rs:590-785`) for the 2e/3c/2c families;
  `CINTnabla1i_1e`/`CINTnabla1j_1e` for the 1e families.
- **D-08:** `int3c1e_iprinv` reuses the **existing `PTR_RINV_ORIG` env slot** (`env[4..6]`,
  Phase 21 D-01) as-is — no new env plumbing.
- **D-09:** Surface scope = **manifest + RawApiId + kernel + vendor-FFI + oracle only**.
  No `capi` enum variants, no legacy `cint*` wrappers (ROADMAP SC6 + standing
  new-family-surface-scope policy).
- **D-10:** Verification = per-family byte-identity at **atol=1e-12** vs vendored libcint
  6.1.3, cart + sph, every component, in `vendor_*` parity tests double-gated on
  `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1` (without both, parity silently skips).

### Claude's Discretion
- Exact oracle fixtures / shell-tuple coverage beyond the s/p/d(/f) minimum per family.
- Center-index selection detail for the ket-side / remaining-center derivatives (which
  center receives the `+1` headroom) — a research/implementation correctness item; the
  transpose hazard here is real, so apply the same non-square-block validation discipline
  as D-05 where the block is rectangular.
- Whether `int3c2e_ip2` needs anything beyond the Phase-21 `int3c2e_ip1` repair as a base.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 21 context (the direct dependency — read first)
- `.planning/phases/21-coulomb-gradient-intors/21-CONTEXT.md` — the gradient engine,
  env-slot precedent, manifest-registration pattern, and spinor escape-hatch this phase reuses.

### Reusable gradient machinery (the only real math)
- `crates/cintx-cubecl/src/kernels/f12.rs:590-785` — `gout_ip1` + `nabla1i_2e`/`nabla1j_2e`/`nabla1k_2e` (generic, F12-free; reuse verbatim for the 2e/3c/2c families).
- `crates/cintx-cubecl/src/kernels/f12.rs:998` — `gout_ip1ip2` (rank-9 2e ip1ip2; a *reference* for 9-component handling — note its index semantics differ from 1e both-side).
- `crates/cintx-cubecl/src/kernels/one_electron.rs:1864` — `CINTnabla1i_1e` (bra-side 1e nabla).
- `crates/cintx-cubecl/src/kernels/one_electron.rs:1772` — `CINTnabla1j_1e` (ket-side 1e nabla). Compose these two for the rank-9 `int1e_*ip` families.
- `crates/cintx-cubecl/src/kernels/two_electron.rs` — `fill_g_tensor_2e`, `build_2e_shape`.
- `crates/cintx-cubecl/src/kernels/center_3c2e.rs` — `int3c2e_ip2` path (builds on the Phase-21 `int3c2e_ip1` repair).
- `crates/cintx-cubecl/src/kernels/mod.rs:26-50` — canonical-family → launcher routing.
- `crates/cintx-cubecl/src/math/rys.rs:3247-3255` — the nroots≤5 Rys dispatch ceiling (panics at ≥6 — the wall D-03 respects).
- `crates/cintx-cubecl/src/executor.rs:11-13` — the `ang_momentum > 4` gate (blocks g/h).

### Component-rank / staging plumbing
- `crates/cintx-runtime/src/planner.rs:403` — `parse_component_multiplier` (accepts any numeric rank; `"9"` auto-allocates `9×`).

### env-slot reuse (`iprinv`)
- `crates/cintx-runtime/src/planner.rs:44` — `OperatorEnvParams` (`rinv_orig` already present).
- `crates/cintx-compat/src/raw.rs:33-41` — env-slot map (`PTR_RINV_ORIG = 4..6`); `:111-151` — RawApiId consts.

### Manifest + surface registration
- `crates/cintx-ops/generated/compiled_manifest.lock.json` (source of truth) + `crates/cintx-ops/build.rs` (regenerator).
- ROADMAP SC6: no `capi`/legacy-wrapper surface added — confirm `crates/cintx-capi/src/shim.rs` and `crates/cintx-compat/src/legacy.rs` are NOT touched for these families.

### Oracle
- `crates/cintx-oracle/src/vendor_ffi.rs` — FFI wrappers around vendored libcint 6.1.3 (add inbound FFI for the 8 families).
- `crates/cintx-oracle/tests/*_parity.rs` — `#[cfg(has_vendor_libcint)]` byte-identity tests (the `vendor_*` pattern).

### libcint upstream (vendored) — the layout source of truth for D-05
- The vendored libcint 6.1.3 `CINTgout1e_int1e_ipovlpip` definition (autocode/`grad1.c` / `g1e.c`) — read to pin the rank-9 component nesting.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase-21 gradient engine (`gout_ip1`, `nabla1i/j/k_2e`) and both single-side 1e nablas
  (`CINTnabla1i_1e`, `CINTnabla1j_1e`) already exist — this phase is composition + registration,
  not new kernel math.
- `parse_component_multiplier` already parses arbitrary numeric `component_rank`, so `"9"`
  staging needs no runtime change.
- The `PTR_RINV_ORIG` env slot and `OperatorEnvParams.rinv_orig` are already plumbed (Phase 21).

### Established Patterns
- Gradient families carry `"component_rank":"3"` in the manifest (the tell). The 8 new
  families add five rank-3 entries and three rank-9 entries.
- `vendor_*` parity tests are double-gated; without `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`
  parity silently skips (determinism-only). Plans must run both gates.

### Integration Points
- New manifest entries → `build.rs` regenerates `api_manifest.rs` + `.csv`.
- New launchers register in `kernels/mod.rs:26-50` canonical-family routing.
- `eval_raw` (`cintx-compat/src/raw.rs`) dispatches each new RawApiId.

</code_context>

<specifics>
## Specific Ideas

- The rank-9 both-side families are the **only** part of this phase that is not a
  mechanical Phase-21 reuse — concentrate verification effort (the non-square p×d block)
  there.
- 2-center 1e gradients have a more forgiving ceiling than the 4-center families:
  `(li+1 + lj+1)/2 + 1` reaches **5 at ff**, so f-shell coverage is "free" for
  `ipovlpip`/`ipkinip`/`ipnucip` — don't artificially cap them at d.
- The transpose-hazard lesson (square blocks hide layout bugs) applies to any rectangular
  bra/ket block, not just rank-9 — see the spinor-orientation precedent.

</specifics>

<deferred>
## Deferred Ideas

- **Full f/g coverage for the 2e/3c/2c families** — requires the nroots≥6 Wheeler/Jacobi
  Rys fallback + the `executor.rs` `l>4` gate raise + lanthanide-ECP validation. Assigned to
  **Phase 25** (todo `rys-nroots-ge6-wheeler-fallback`, `resolves_phase: 25`). The
  max-within-ceiling decision (D-03) explicitly leaves this for Phase 25.

### Reviewed Todos (not folded)
- `rys-nroots-ge6-wheeler-fallback` — reviewed and **not folded**; it is a new foundation
  (nroots≥6 roots, l>4 gate, lanthanide ECP) assigned to Phase 25 and contradicts this
  phase's "zero new foundations" charter. D-03 respects the existing ceiling instead.
- `oracle-cart-offset-vendor-zero` — reviewed and **not folded**; it is a separate
  `CINTshells_cart_offset[4]` lib-test bug in `compare::tests::helper_coverage_matches_manifest`,
  not a gradient-family concern. Belongs to the oracle/helper-coverage track, not Phase 23.

</deferred>

---

*Phase: 23-group-1-remaining-1st-derivative-families-cart-sph*
*Context gathered: 2026-05-29*
