# Phase 23: Group 1 — Remaining 1st-Derivative Families (cart/sph) - Context

**Gathered:** 2026-05-29
**Updated:** 2026-05-30 (cluster C implemented + vendor-verified; discretion items resolved)
**Status:** In progress — cluster C (rank-9 both-side) DONE; clusters A & B remain

<implementation_status>
## Implementation Status (2026-05-30)

- **Cluster C — rank-9 both-side 1e (`int1e_ipovlpip`, `int1e_ipkinip`, `int1e_ipnucip`): COMPLETE.**
  Generic-float `#[cube]` device kernels + manifest + RawApiId + vendor FFI + oracle
  tests, all landed in commit `319d055`. Vendor byte-identity vs libcint 6.1.3 = 0
  mismatches at atol=1e-12 (cart + sph, H2O/STO-3G); 36/36 cubecl device tests +
  6/6 new oracle tests green; `manifest-audit` (no `--check-lock`) green.
- **Clusters A & B — rank-3 ket/remaining-center (`int2e_ip2`, `int3c2e_ip2`,
  `int2c2e_ip1/ip2`, `int3c1e_ip1/iprinv`): NOT STARTED.** They reuse the Phase-21
  engine and the registration recipe now proven on cluster C (see D-11).
</implementation_status>


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
  - **RESOLVED (impl):** ordering is **bra-major DIRECT** — `comp = bra_axis*3 + ket_axis`,
    `s[0..8]` map straight to `gout[0..8]` with NO permutation. (Contrast the sibling
    `ipipovlp`, which DOES permute s→gout — do not copy that.) Tensors:
    `g0`, `g1=D_j(g0)` at `i_l+1`, `g2=D_i(g0)`, `g3=D_i(g1)`; headroom `nmax=li+lj+2`,
    `lj_ext=lj+1`. Validated with non-square p×d/d×p host-ref + vendor byte-identity.

- **D-12 (impl-discovered):** **`int1e_ipkinip` reduces to 8 distinct tensors.** libcint
  `hess.c` materializes g0..g15, but they collapse to `dj0..dj3` (ket `D_j` orders 0–3)
  and `di0..di3` (their bra `D_i`); port the 27 used s-terms verbatim. Headroom
  `nmax=li+lj+4, lj_ext=lj+3` (third ket derivative). The kinetic **½ is folded into
  libcint `common_factor` (gout coeff −1)** → apply **`-0.5`** in-contraction (cintx convention).

- **D-13 (impl-discovered):** **`int1e_ipnucip` = the nuclear analog of ipovlpip** — same
  4-tensor both-side composition on the nuclear Rys `g0`, summed over Rys roots and nuclei
  (`-Z` folded into `g0`). `nroots = (li+lj+2)/2 + 1` (one extra root vs single-side ipnuc
  for the added ket headroom); **fail closed when `nroots > 5`** (device Rys ceiling).

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

### Registration recipe (proven on cluster C — reuse for A & B)
- **D-11 (impl-discovered):** The 5-step registration that lands a new family:
  1. **Manifest:** add lock entries (cart/sph/spinor) to `compiled_manifest.lock.json`
     cloning the closest existing family, with `component_rank` = true output multiplier;
     `cargo build -p cintx-ops` regenerates `api_manifest.{rs,csv}`.
  2. **RawApiId** consts in `cintx-compat/src/raw.rs`.
  3. **Launcher** dispatch on `descriptor.operator_name()` (no operator allowlist exists).
  4. **Vendor FFI:** add the cart/sph symbols to the bindgen `allowlist_function` regex in
     `cintx-oracle/build.rs` + safe wrappers in `vendor_ffi.rs` (confirm the autocode `.c`
     is already in that build.rs source list — `hess.c`/`grad1.c` are).
  5. **Oracle** `vendor_*` parity test (per D-10).
  - **KEY FINDING:** the `manifest-audit` "generated" and "lock" sides BOTH derive from the
    lock (`build.rs → api_manifest.rs → Resolver::manifest()` ← fixtures
    `phase4_operator_entries`), so **lock edits auto-sync the audit — there is NO separate
    fixtures family list to edit.** `component_rank` flows through
    `parse_component_multiplier` / `oracle_component_count` automatically. (Supersedes the
    discuss-phase worry that the fixtures generator needed manual extension.)
  - **AUDIT NOTE:** `manifest-audit` (no flags) is the gate and is GREEN. `--check-lock` is
    red ONLY from pre-existing uncovered-stable spinor debt (`int1e_ipovlp_spinor` etc.);
    new spinor entries match that exact pattern and add no new failure class.

### Component-rank landmine (the discuss-phase area, now a hard rule)
- **D-14:** A manifest `component_rank` set too LOW silently TRUNCATES trailing output
  components (root cause of the 260530-9ay unstable-derivative failures — it looked like a
  math bug, it wasn't). For these families register **rank 3 (clusters A/B) / 9 (cluster C)**
  and assert it: the oracle test pins the element count (`9*n_ao*n_ao` for rank-9) AND
  asserts `any_nonzero` so a stub/short buffer can't pass parity.

### Claude's Discretion (remaining)
- Exact oracle fixtures / shell-tuple coverage beyond the s/p/d(/f) minimum per family.
- ~~Center-index selection detail for the ket-side / remaining-center derivatives~~ —
  partially resolved by D-05/D-13 (the `+1` headroom goes on the derivative center; apply
  the non-square-block discipline). Still open for the cluster A/B rank-3 families at impl.
- Whether `int3c2e_ip2` needs anything beyond the Phase-21 `int3c2e_ip1` repair as a base
  (cluster A — not yet implemented).

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

### libcint upstream (vendored) — the layout source of truth for D-05/D-12/D-13
- `libcint-master/src/autocode/hess.c` — `CINTgout1e_int1e_ipovlpip` (:94), `CINTgout1e_int1e_ipkinip` (:345), `CINTgout1e_int1e_ipnucip` (:600). Pinned the rank-9 nesting + the kinetic `common_factor *= 0.5` convention (see also `intor1.c` `int1e_kin`).

### Implemented cluster-C source (the template for clusters A & B)
- `crates/cintx-cubecl/src/kernels/one_electron.rs` — kernels `one_electron_grad_both_kernel` (ipovlpip), `one_electron_grad_kin_both_kernel` (ipkinip), `one_electron_nuc_grad_both_kernel` (ipnucip); shared `#[cube]` helpers `d_j_1e_into`/`d_i_1e_into`; the `is_rank9_both` launcher branch; host-ref device tests `test_device_ip{ovlpip,kinip,nucip}_matches_host_reference`.
- `crates/cintx-oracle/tests/one_electron_grad_both_parity.rs` — 9-component determinism + vendor byte-identity tests.
- Commit `319d055` — the full cluster-C diff (kernels + registration + tests).

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
