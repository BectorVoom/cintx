# Phase 21: Plain-Coulomb Gradient Integral Families (`ip1`/`iprinv`) - Context

**Gathered:** 2026-05-26
**Status:** Ready for planning
**Source:** Pre-drafted proposal `.planning/notes/phase-21-coulomb-gradient-intors-PLAN.md` (verified against the tree 2026-05-26). Used in place of `/gsd:discuss-phase` per operator decision — the proposal already encodes the design decisions, scope, and verified research.

<domain>
## Phase Boundary

**In scope (the 6 families + 1 repair + 1 infra slot):**
- `int2e_ip1` (arity-4, 3 components) — the two-electron force; highest-impact term in every gradient.
- `int1e_ipovlp` (arity-2, 3 components) — Pulay/overlap-derivative term.
- `int1e_ipkin` (arity-2, 3 components) — core-Hamiltonian kinetic derivative.
- `int1e_ipnuc` (arity-2, 3 components) — hcore nuclear-attraction derivative (∇ on bra center, summed over all nuclei).
- `int1e_iprinv` (arity-2, 3 components) — per-atom Hellmann–Feynman force (single rinv origin).
- `ECPscalar_iprinv` (arity-2, 3 components) — per-nucleus ECP force.
- **Repair (family 0):** `int3c2e_ip1` — registered but stubbed; ship the real derivative kernel + flip the oracle reference.
- **Infrastructure:** the `PTR_RINV_ORIG` env slot (`env[4..6]`), entirely absent from cintx today.

**Acceptance:** byte-identity at **atol=1e-12** against vendored libcint 6.1.3 for cart + sph, each of the 3 components, on the existing H2O/STO-3G + Cu/LANL2DZ corpus — the same gate Phases 17/19 used.

**Out of scope:**
- **Spinor gradient kernels** — register-but-`UnsupportedApi` (Risk R5). pyscf_rs needs only `sph`/`cart`.
- **High-l (f/g) gradients past nroots=5** — the gradient's `li+1` overflows the Rys-root ceiling for f/g quartets; same ceiling as base `int2e`. Deferred behind the Wheeler-fallback work (Risk R2).
- **pyscf_rs-side changes** — the consumer flips its own `workflow_dispatch` gate once these land; no pyscf_rs rework is in this phase.
- **`int3c1e_p2` operator-blind misnomer fix** — identical bug class to R1 but only folded in if a consumer needs it.

</domain>

<decisions>
## Implementation Decisions

### rinv-origin env plumbing
- **D-01:** Replicate the `f12_zeta` (env[9]) 4-step pattern for a new `rinv_orig: Option<[f64;3]>`: a typed field on `OperatorEnvParams` (`cintx-runtime/src/planner.rs:44`), populated from `env[4..6]` in `cintx-compat/src/raw.rs::eval_raw`, validated in `validator.rs` (an `iprinv` operator without an origin is rejected), and threaded into the `one_electron`/`ecp` kernels. Expose a `with_rinv_origin`-style setter on the safe-API options. (→ 21-01, GRAD-01)

### Manifest registration
- **D-02:** Register all 6 families + the `int3c2e_ip1` correction in `crates/cintx-ops/generated/compiled_manifest.lock.json` with `"component_rank":"3"` per representation (the tell that marks a 3-component gradient; the runtime planner auto-allocates `3 × ni × nj[× nk × nl]` staging — no manual layout code), plus `cint*` legacy and `*_optimizer` symbols. `cargo build` (`build.rs`) regenerates `api_manifest.rs` + `.csv`. (→ 21-02, GRAD-02)
- **D-03:** Spinor representations are registered in the manifest but their kernels return `UnsupportedApi` (Risk R5) — mirrors Phase 18/19 "compiled-but-unverified" escape hatch. (→ 21-02)

### Kernel strategy — reuse the generic gradient machinery
- **D-04:** Reuse `gout_ip1` + `nabla1i_2e`/`nabla1j_2e`/`nabla1k_2e` from `crates/cintx-cubecl/src/kernels/f12.rs:590-785` **verbatim** — they contain zero F12/STG/YP logic and implement the standard libcint identity `∂/∂A χ_l = -2α·χ_{l+1} + l·χ_{l-1}` (matches `CINTnabla1i_2e` / `G2E_D_I`). Feed them the *plain* G-tensor (from `two_electron.rs::fill_g_tensor_2e` via `rys_roots_host`), not the F12 G-tensor. (cross-cutting → 21-03..21-07)
- **D-05:** 1e gradients extend the existing 1e dispatcher at `one_electron.rs:486-495` with `ipovlp|ipkin|ipnuc|iprinv` operator-symbol branches (routing is by canonical family with internal operator-symbol branching; all variants of a family share one launcher). The `contract_kinetic` `CINTnabla1j_1e` code at `one_electron.rs:208` is the nabla pattern to follow. (→ 21-03, 21-04)
- **D-06:** `int2e_ip1` gets a new gradient path in `two_electron.rs`: `build_2e_shape(li+1, lj, lk, ll)` → `fill_g_tensor_2e` + `rys_roots_host` → `gout_ip1`. Output uses component-leading `[3, nl, nk, nj, ni]` F-order to match pyscf-gto `layout_table.rs` exactly (Risk R3). (→ 21-05, GRAD-07)
- **D-07:** `int3c2e_ip1` ships a real derivative kernel in `center_3c2e.rs` (same `gout_ip1` reuse), replacing the operator-blind scalar stub `launch_center_3c2e_typed`; flip the oracle reference from plain `vendor_int3c2e` to `vendor_int3c2e_ip1` (Risk R1). (→ 21-06, GRAD-08)
- **D-08:** `int1e_ipnuc` sums the nabla over **all** nuclei (∇ on bra center); `int1e_iprinv` uses a **single** rinv origin (the D-01 env slot) and drops the `-Z_C` charge factor. They share the `gout_ip1` nabla on the nuclear Rys tensor and differ only in atom-loop-vs-single-origin and the prefactor. (→ 21-04, GRAD-05/GRAD-06)

### ECP gradient
- **D-09:** `ECPscalar_iprinv` is a per-nucleus selector in `launch_ecp` — the `ipnuc` driver `deriv1_cart_pair` at `ecp.rs:1181` sums all ECP slots; `iprinv` selects one via the D-01 rinv origin and drops the `-Z_C`/all-slot accumulation. Reuse the salvaged `19-05` `Y_ADDR`/`Z_ADDR`/`CART_POW_*` tables (fix the `[usize;135]`→`[usize;120]` sizing bug on reuse). **Pre-req (Risk R4):** confirm Phase 19's Cu/LANL2DZ gate exercises the PySCF-exact K-Taylor path (`K_TAB`/`ECPrad_part`), not the old direct-quadrature approximation, before building on top; otherwise insert a K-Taylor-port plan first. (→ 21-07, GRAD-09)

### Verification
- **D-10:** Per-family oracle byte-identity at **atol=1e-12** vs vendored libcint 6.1.3 (`cintx-oracle`, `#[cfg(has_vendor_libcint)]`) for cart + sph and each of the 3 components, on H2O/STO-3G (+ Cu/LANL2DZ for ECP). Reuse the workspace's ordered-reduction determinism discipline (no FMA-introduced reorder) so values are bit-stable. (→ all plans, GRAD-03..GRAD-09)

### Phase 18 coupling (sequencing)
- **D-11:** The `int2e_ip1` raw/compat (`eval_raw`) path lands independent of Phase 18; only the `int2e_ip1` **safe-API** (`SessionRequest::evaluate` arity-4) arm is Phase-18-coupled. Confirm which entry point pyscf-gto's `intor.rs` calls (raw vs safe) before committing 21-05's surface (Risk R6). (→ 21-05)

### Claude's Discretion
- Exact oracle fixtures/shell-tuple coverage beyond the s/p/d minimum; helper genericization order within a wave; whether to fold the `int3c1e_p2` operator-blind fix into 21-06; whether `ECPscalar_iprinv` needs a standalone K-Taylor-port pre-plan (depends on the R4 confirmation outcome).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Reusable gradient machinery (the only real math)
- `crates/cintx-cubecl/src/kernels/f12.rs:590-785` — `gout_ip1` + `nabla1i_2e`/`nabla1j_2e`/`nabla1k_2e` (generic, F12-free; reuse verbatim).
- `crates/cintx-cubecl/src/kernels/two_electron.rs` — `fill_g_tensor_2e`, `build_2e_shape` (plain 2e G-tensor + Rys).
- `crates/cintx-cubecl/src/kernels/one_electron.rs:486-495` (1e operator dispatcher) and `:208` (`contract_kinetic` `CINTnabla1j_1e` nabla pattern).
- `crates/cintx-cubecl/src/kernels/center_3c2e.rs` — `launch_center_3c2e_typed` (the operator-blind stub to repair, R1).
- `crates/cintx-cubecl/src/kernels/ecp.rs:1181` — `deriv1_cart_pair` (the `ipnuc` ECP driver to adapt for per-nucleus `iprinv`).
- `crates/cintx-cubecl/src/kernels/mod.rs:26-50` — canonical-family → launcher routing.
- `crates/cintx-cubecl/src/math/rys.rs:3248` — `rys_root1..5` dispatch (the nroots≤5 ceiling, R2).

### env-slot plumbing precedent (`f12_zeta` → `rinv_orig`)
- `crates/cintx-runtime/src/planner.rs:44` — `OperatorEnvParams` (add `rinv_orig`).
- `crates/cintx-runtime/src/validator.rs` — operator validation gate.
- `crates/cintx-compat/src/raw.rs:33-41` — env-slot map (`PTR_RINV_ORIG = 4..6`); `:111-151` — RawApiId consts.

### Manifest + surface registration
- `crates/cintx-ops/generated/compiled_manifest.lock.json` (source of truth) + `crates/cintx-ops/build.rs` (regenerator).
- `crates/cintx-compat/src/legacy.rs:81,227,312` — `all_cint_wrappers!`, `LEGACY_WRAPPER_SYMBOLS`, `misc`-family match (the `legacy_wrapper_surface_matches_misc` test enforces sync).
- `crates/cintx-capi/src/shim.rs:9-33` — `CintxRawApi` `#[repr(i32)]` variants + `from_i32` + `raw_id()`.

### Oracle
- `crates/cintx-oracle/src/vendor_ffi.rs` — FFI wrappers around vendored libcint 6.1.3.
- `crates/cintx-oracle/tests/*_parity.rs` — `#[cfg(has_vendor_libcint)]` byte-identity tests.

### Consumer (cross-repo, read-only context)
- pyscf_rs `.planning/phases/07-gradients-geomopt/07-RESEARCH.md` §"Gradient-Integral Availability Matrix".
- pyscf-gto `layout_table.rs` — the component-leading `[3, …]` F-order contract (Risk R3).
- pyscf-grad `src/hooks.rs:24` — the DF-grad runtime path that consumes `int3c2e_ip1` as a derivative (Risk R1).

### Salvaged scaffolding
- `.planning/notes/19-05-partial-gradient-scaffolding.md` — reusable `Y_ADDR`/`Z_ADDR`/`CART_POW_*` tables (fix `[usize;135]`→`[usize;120]` on reuse).

</canonical_refs>

<specifics>
## Specific Ideas

- The manifest tell for a gradient family is `"component_rank":"3"`: compare `int1e_ecp_ipnuc` entries (which carry it) against `int3c2e_ip1` entries (which are missing it — the proof its kernel never produced 3 components).
- `nroots = (li+1+lj+lk+ll)/2 + 1` after the gradient raises `li→li+1`; ≤5 for s/p/d, overflows for f/g.
- The "real acceptance test" is the consumer un-gate: once green, pyscf_rs flips its Phase 7 `workflow_dispatch` grad arms to always-on; RHF/UHF/RKS/UKS/MP2/CCSD analytical gradients then ride the existing FD gate (`grad.verify_fd`) at ≤1e-6 Ha/Bohr and upstream-PySCF parity at ≤1e-7 Ha/Bohr.

</specifics>

<deferred>
## Deferred Ideas

- Spinor gradient kernels (register-but-`UnsupportedApi`, R5) — until a consumer needs them.
- High-l (f/g) gradients past nroots=5 (R2) — behind the deferred Wheeler-fallback higher-roots work.
- `int3c1e_p2` operator-blind misnomer fix (`center_3c1e.rs`) — same bug class as R1; fold in only on consumer demand.

</deferred>

---

*Phase: 21-coulomb-gradient-intors*
*Context seeded 2026-05-26 from the pre-drafted proposal (no discuss-phase; research embedded in the proposal).*
