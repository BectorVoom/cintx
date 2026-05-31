# Phase 27: Spinor-Derivative Transform (Gap B1) - Context

**Gathered:** 2026-05-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement `cart_to_spinor_sf_derivative_*` in `crates/cintx-cubecl/src/transform/c2spinor.rs` so that the derivative-component axis (`[3, …]` and higher rank) folds correctly through the existing cart→spinor coupling. This flips `int1e_ipovlp_spinor` and its **spin-free** `ip`-decorated sibling families from `UnsupportedApi` to byte-identity at atol=1e-12, closing the Phase-21 R5/D-03 deferral (FND-04).

This is **Gap B1 only** — the spin-free (sf) derivative transform. Spin-included (σ-bearing) families belong to Gap B2 / Phase 28-29 and are out of scope. No capi enum variants and no legacy `cint*` wrappers are added (per the project's new-family surface policy).

</domain>

<decisions>
## Implementation Decisions

### Flip scope (which families move to byte-identity this phase)
- **D-01:** Flip **all arity-2 1e spin-free `ip` families** that fold through `cart_to_spinor_sf_2d` — across every rank tier present: rank-3 (`int1e_ipovlp/ipkin/ipnuc/iprinv_spinor`), rank-9 both-side (`ipovlpip/ipkinip/ipnucip`), and the second/higher-order set (`ipipovlp/ipipnuc/ipipkin/ipiprinv`, `ipipipnuc/ipipiprinv/ipipnucip/ipiprinvip/ipipipiprinv/ipiprinvipip/ipipiprinvip`). All share one transform addition; all their scalar cart/sph kernels already land (phases 21/23/25).
- **D-02:** Also flip the **arity-3** families (`int3c2e_ip1/ip2_spinor`, `int3c1e_ip1/iprinv_spinor`) and **2c2e** families (`int2c2e_ip1/ip2_spinor`). Arity-3 needs a derivative wrapper over `cart_to_spinor_sf_3c2e`; 2c2e folds through the `sf_2d` path. All their scalar cart forms are confirmed `oracle_covered=true`.
- **D-03 (DEFERRED):** The **arity-4 `int2e_ip*` set** (`int2e_ip1/ip2/ipip1/ipvip1/ip1ip2/ipip1ipip2_spinor`) is deferred to a follow-up — it requires a NEW derivative variant of `cart_to_spinor_sf_4d`, a larger blast radius than the `sf_2d`/`sf_3c2e` slice. See Deferred Ideas.
- **D-04 (DEFERRED):** `int1e_ecp_iprinv_spinor` stays `UnsupportedApi` — it is the only ip-spinor family that is not pure spin-free (R5/ECP-spinor track).

### Transform API shape
- **D-05:** Add **thin generic wrappers** `cart_to_spinor_sf_derivative_2d(staging, cart, ncomp, li, kappa_i, lj, kappa_j)` and a `_3c2e` sibling that **loop the already-verified per-component transforms** (`cart_to_spinor_sf_2d` / `cart_to_spinor_sf_3c2e`) `ncomp` times with correct strides. `ncomp ∈ {3, 9, 27, 81}`. This mirrors how libcint itself applies the transform — libcint has **no** derivative-specific c2s function; its `int1e` driver loops over derivative components and calls `CINTc2s_ket_spinor` per component. Reusing the byte-identity-proven inner transform avoids reimplementing the coupling and is the correct reading of Phase-12 D-03 (the per-component coupling matrix is identical; only the loop count differs, so distinct per-rank functions would be pure duplication).
- **D-06:** The KET→BRA **transpose is owned INSIDE the wrapper**. Each per-component device-native (`[comp][ket][bra]`, KET-major) cart sub-block is transposed to BRA-major internally before the inner `sf_2d`/`sf_3c2e` call. Callers pass device-native blocks. This centralizes the exact landmine that caused the scalar-spinor orientation bug (a launcher forgetting to transpose) into one audited place so no kernel launcher can omit it.
- **D-07:** Output uses **component-major-outer** layout — for each derivative component, a full `di*dj*2` (complex, interleaved `[re,im,…]`) spinor block, components as the slowest-varying axis (`out + comp * di*dj*2`). This matches libcint's `int1e` driver output stride AND the convention already established by the covered scalar `ip` families. (Compatibility-determined, not a free choice.)

### Parity fixture design
- **D-08:** The dedicated `vendor_*` spinor parity fixture is **maximally adversarial in a single shell tuple**: **non-square** bra/ket angular momenta (e.g. p×d) to defeat transpose-symmetry that hides the KET/BRA bug; **at least one shell with nctr>1** to exercise general contraction (catches the column/row-major coefficient transpose that has bitten every prior family); and **kappa=0** so both GT (j=l+1/2) and LT (j=l−1/2) blocks fire and the `di = 4l+2` sizing path is stressed. One fixture covers all three known landmines.

### Test coverage breadth
- **D-09:** Provide a dedicated vendor byte-identity test **per transform-path × per rank-tier exercised**: `sf_2d` at rank-3 (`int1e_ipovlp_spinor`), rank-9 (`int1e_ipovlpip_spinor`), and the highest rank present (27/81, e.g. an `ipip*` family); `sf_3c2e` at rank-3 (`int3c2e_ip1_spinor`). Each uses the D-08 adversarial fixture. This proves the `ncomp` axis-fold at every component count and both transform paths.
- **D-10:** Remaining in-scope families flip via the manifest, guarded by a **no-silent-skip coverage assertion** — a test that verifies every flipped family is `oracle_covered=true` AND that the vendor parity tests actually execute (`running N>0 tests`, not a `skipped` fixture) under both required flags. `manifest-audit` must be green. (Honors the project's vendor-gated-oracle skip risk: parity silently skips without `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`.)

### Research spike
- **D-11:** Run a **full design spike** (`/gsd:spike`) before plan tasks are finalized — exercise the full per-component axis-fold across all rank tiers and both transform paths (`sf_2d`, `sf_3c2e`). The single genuine residual unknown the spike must nail down: the exact device-emitted derivative cart block layout (confirm `[comp][ket][bra]` component-outer) and the precise per-component stride into staging, verified against hand-checked vendor values.

### Claude's Discretion
- Exact molecule/basis for the fixture (subject to the D-08 hard constraints: non-square, nctr>1 somewhere, kappa=0).
- Internal stride arithmetic and loop factoring within the wrappers.
- Whether `int3c1e` shares the `sf_3c2e` derivative wrapper or needs a thin sibling (resolve from the actual block layout during the spike).
- Plan boundaries between the `sf_2d` path families and the `sf_3c2e` path families.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & roadmap
- `.planning/REQUIREMENTS.md` — FND-04 (line 81): the Gap B1 requirement definition.
- `.planning/ROADMAP.md` §"Phase 27" (lines 626-638) — Goal, success criteria, dependency on Phase 23, and the design-spike research flag.

### Prior phase context (decided conventions to inherit, do NOT re-decide)
- `.planning/phases/12-real-spinor-transform-c2spinor-replacement/12-CONTEXT.md` — Phase-12 D-01..D-08: CG coefficient source (`c2spinor.c` `g_c2s_*`), `c2spinor_coeffs.rs` location, distinct sf/si/ket/iket code paths, interleaved `[re,im,…]` staging layout, kappa→block dispatch.

### Spinor transform implementation (the file this phase extends)
- `crates/cintx-cubecl/src/transform/c2spinor.rs` — Existing `cart_to_spinor_sf` (L290), `cart_to_spinor_sf_2d` (L531), `cart_to_spinor_sf_3c2e` (L1281), `cart_to_spinor_sf_4d` (L879), and `spinor_len(l, kappa)` (L25). The new derivative wrappers loop these.
- `crates/cintx-cubecl/src/transform/c2spinor_coeffs.rs` — CG coupling coefficient tables.
- `crates/cintx-cubecl/src/transform/mod.rs` — `apply_representation_transform()`; documents that Spinor is dispatched explicitly in kernel launchers via `cart_to_spinor_sf_2d/4d/3c2e`, NOT through the generic transform arm.

### Kernel launchers (call sites that must invoke the new wrappers)
- `crates/cintx-cubecl/src/kernels/one_electron.rs` — 1e scalar/gradient spinor launch path (where the scalar-spinor orientation bug was fixed).
- `crates/cintx-cubecl/src/kernels/center_3c2e.rs` — 3c2e (incl. `int3c2e_ip1`) launch path.
- `crates/cintx-cubecl/src/kernels/center_2c2e.rs` — 2c2e launch path.

### Manifest & coverage
- `crates/cintx-ops/src/generated/api_manifest.rs` — ManifestEntry rows for all `*_spinor` ip families (`component_rank`, `forms`, `oracle_covered`). Generated from the lock.
- `compiled_manifest.lock.json` — the source of truth for the manifest; lock edits auto-sync both audit sides.
- `xtask/src/oracle_covered_update.rs` (L32) — the current R5/D-03 note recording these families as deferred; update when flipping.

### Oracle / vendor parity infrastructure
- `crates/cintx-oracle/src/vendor_ffi.rs` — vendored libcint FFI incl. spinor helpers.
- `crates/cintx-oracle/src/compare.rs` — oracle comparison, atol=1e-12, `IMPLEMENTED_TRANSFORM_SYMBOLS`.
- `crates/cintx-oracle/src/fixtures.rs` — fixture generation incl. spinor representation dispatch (extend for the D-08 adversarial fixture).

### Upstream reference
- Vendored libcint 6.1.3 `c2spinor.c` (`CINTc2s_ket_spinor_sf1`) and the `int1e`/`int3c2e` drivers — authoritative for per-component application order and output stride.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `cart_to_spinor_sf_2d` / `cart_to_spinor_sf_3c2e` are byte-identity-proven per-component transforms — the derivative wrappers are thin loops over these, not new coupling code.
- `spinor_len(l, kappa)` already gives correct component counts incl. the kappa=0 both-blocks case (`4l+2`).
- All in-scope scalar cart forms are confirmed `oracle_covered=true` (verified during discussion) — no scalar-kernel work blocks the flip.
- The manifest audit derives both sides from `compiled_manifest.lock.json`, so flipping `oracle_covered` in the lock auto-syncs — no separate fixtures list to edit.

### Established Patterns
- New-family surface policy: manifest + RawApiId + kernel + vendor-FFI + oracle ONLY — no capi enum variants, no legacy `cint*` wrappers; keep inbound vendor FFI for byte-identity.
- Vendor parity tests are double-gated: real comparison requires `--features cpu` AND `CINTX_ORACLE_BUILD_VENDOR=1`; without both, parity silently skips (determinism-only). The D-10 assertion must guard against this false-pass.
- Interleaved `[re0,im0,re1,im1,…]` complex staging; oracle compares the flat buffer directly.

### Integration Points
- **Landmine (scalar-spinor bug):** device cart blocks are KET-major but `cart_to_spinor_sf_2d` reads BRA-major → transpose first. D-06 moves this transpose inside the wrapper. Spinor parity MUST use a non-square block (square p×p is transpose-symmetric and hides the bug).
- **Landmine (nctr>1 coefficient transpose):** libcint env coeff block is column-major; cintx `Shell` is row-major. Every new family needs an nctr>1 fixture case (D-08).
- **Landmine (component_rank truncation):** a prior bug dropped trailing components when `component_rank` was wrong in the lock — verify rank values for all flipped families.

</code_context>

<specifics>
## Specific Ideas

The user wants the verification to be genuinely adversarial rather than ceremonial: the single fixture geometry (non-square + nctr>1 + kappa=0) was chosen specifically because each property defeats a distinct landmine that has caused silent false-passes in prior phases. The full design spike is wanted precisely to nail the device block layout empirically before committing plan tasks — do not shortcut it.

libcint is the authoritative reference for the per-component output stride; match its `int1e`/`int3c2e` driver byte-for-byte.

</specifics>

<deferred>
## Deferred Ideas

- **Arity-4 `int2e_ip*` spinor families** (`int2e_ip1/ip2/ipip1/ipvip1/ip1ip2/ipip1ipip2_spinor`) — require a derivative variant of `cart_to_spinor_sf_4d`. Deferred to a follow-up phase to keep this slice's blast radius to the `sf_2d`/`sf_3c2e` paths. Their scalar cart forms are already covered, so they are unblocked whenever the `sf_4d` derivative wrapper lands.
- **`int1e_ecp_iprinv_spinor`** — the spinor ECP gradient; not pure spin-free, belongs to the R5/ECP-spinor track (Phase 29 relativistic group), not Gap B1.

</deferred>

---

*Phase: 27-spinor-derivative-transform-gap-b1*
*Context gathered: 2026-05-31*
