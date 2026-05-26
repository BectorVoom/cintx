# Phase 21: Plain-Coulomb Gradient Integral Families (`ip1`/`iprinv`) — Research

**Source:** `.planning/notes/phase-21-coulomb-gradient-intors-PLAN.md` (drafted 2026-05-26). All findings below were **verified against the tree** at draft time; file:line references are load-bearing. This RESEARCH.md is the planner's technical input; the design decisions live in `21-CONTEXT.md`.

---

## Summary

The 6 plain-Coulomb gradient families plus the `int3c2e_ip1` repair are largely "reuse the generic first-derivative machinery + register in the manifest + add an oracle gate." The only genuinely new pieces are (1) the `PTR_RINV_ORIG` env slot and (2) the per-nucleus ECP selector. The first-derivative kernel math already exists in `f12.rs` and is F12-agnostic. The fastest path to un-gating the bulk of pyscf_rs's HF/DFT/MP2/CCSD gradients is the 5 non-ECP plain-Coulomb families; ECP-grad (R4) and DF-grad-via-stub (R1) carry the residual risk.

**Current derivative-correctness state (verified):** only `int1e_ecp_ipnuc` is genuinely derivative-correct today → **1 of 8**. `int3c2e_ip1` is a registered stub that returns the *plain* (non-derivative) integral; its oracle "passes" only because it references plain `vendor_int3c2e`.

---

## Phase Requirements

| Req | Family / deliverable | Plan |
|-----|----------------------|------|
| GRAD-01 | `PTR_RINV_ORIG` env-slot plumbing | 21-01 |
| GRAD-02 | Manifest registration (6 families + `int3c2e_ip1` correction) | 21-02 |
| GRAD-03 | `int1e_ipovlp` | 21-03 |
| GRAD-04 | `int1e_ipkin` | 21-03 |
| GRAD-05 | `int1e_ipnuc` | 21-04 |
| GRAD-06 | `int1e_iprinv` | 21-04 |
| GRAD-07 | `int2e_ip1` | 21-05 |
| GRAD-08 | `int3c2e_ip1` real derivative kernel | 21-06 |
| GRAD-09 | `ECPscalar_iprinv` | 21-07 |
| GRAD-10 | Verification + pyscf_rs hand-off | 21-08 |

---

## Key findings that drive the design (verified against the tree)

1. **The first-derivative machinery is generic and already exists.** `gout_ip1` + `nabla1i_2e`/`nabla1j_2e`/`nabla1k_2e` live in `crates/cintx-cubecl/src/kernels/f12.rs:590-785`. They contain **zero** F12/STG/YP-specific logic — they operate on a generic G-tensor and implement the standard libcint identity `∂/∂A χ_l = -2α·χ_{l+1} + l·χ_{l-1}` (matches `CINTnabla1i_2e` / `G2E_D_I`). F12-ness lives only in `fill_g_tensor_f12` (which calls `stg_roots_host`). For `int2e_ip1` we reuse `gout_ip1` verbatim and feed it the *plain* G-tensor from `two_electron.rs::fill_g_tensor_2e` (via `rys_roots_host`).

2. **Kernel routing is by canonical family, with internal operator-symbol branching.** `crates/cintx-cubecl/src/kernels/mod.rs:26-50` maps `"1e"→launch_one_electron`, `"2e"→launch_two_electron`, `"3c2e"→launch_center_3c2e`. All variants of a family share one launcher; the launcher branches on `plan.descriptor.operator_name()`/`operator_symbol()`. The 1e dispatcher today only accepts `overlap|kinetic|nuclear-attraction` (`one_electron.rs:486-495`) — add `ipovlp|ipkin|ipnuc|iprinv` branches there.

3. **The manifest is generated from a lock file.** Source-of-truth: `crates/cintx-ops/generated/compiled_manifest.lock.json`. `crates/cintx-ops/build.rs` reads it on `cargo build` and regenerates `src/generated/api_manifest.rs` (+ `.csv`). A gradient family is marked by `"component_rank":"3"`; the runtime planner (`cintx-runtime/src/planner.rs:395,432`) then auto-allocates `3 × ni × nj[× nk × nl]` staging — no manual layout code. `int1e_ecp_ipnuc` entries carry `"component_rank":"3"`; `int3c2e_ip1` entries are missing it (the tell that its kernel never produced 3 components).

4. **Rys roots cover nroots 1..5 only.** `crates/cintx-cubecl/src/math/rys.rs:3248` dispatches `rys_root1..5`; `>5` panics (2e) or returns `UnsupportedApi` (3c2e). A gradient raises `li→li+1`, so `nroots = (li+1+lj+lk+ll)/2 + 1`. ≤5 for s/p/d quartets but overflows for high-l (f/g) — same ceiling the base `int2e` already has. Higher roots (Wheeler fallback) are deferred — out of scope; the gradient inherits the base's l-ceiling (Risk R2).

5. **No rinv-origin / `PTR_RINV_ORIG` plumbing exists.** `crates/cintx-compat/src/raw.rs:33-41` documents the env-slot map (`PTR_RINV_ORIG = 4..6`) but nothing reads it. The precedent is `f12_zeta` (env[9]): a typed `Option<f64>` field in `OperatorEnvParams` (`planner.rs:44`), populated from `env[..]` in `raw.rs::eval_raw`, validated in `validator.rs`, threaded into the kernel. Replicate that 4-step pattern for `rinv_orig: Option<[f64;3]>`.

6. **ECP-gradient byte-identity may be gated on the K-Taylor port.** The scalar ECP primitives (`compute_type1_pair`/`compute_type2_pair`) were once a direct-quadrature approximation, not PySCF's exact `ECPrad_part`/`K_TAB` recurrences. Phase 19 closed with a Cu/LANL2DZ byte-identity gate (the K-Taylor replan, 19-05..19-08), so this may already be resolved — **must re-confirm** before building `ECPscalar_iprinv`, else its oracle can't reach atol=1e-12 (Risk R4). The salvaged `19-05` patch has reusable `Y_ADDR`/`Z_ADDR`/`CART_POW_*` tables (note the `[usize;135]`→`[usize;120]` sizing bug to fix on reuse).

---

## Per-layer touch map (apply to each new family)

The vertical anatomy, top to bottom. Most layers are mechanical; the kernel branch is the only real math.

| Layer | File | What to add |
|-------|------|-------------|
| **Manifest source** | `cintx-ops/generated/compiled_manifest.lock.json` | New entries per representation (`cart`/`sph`/`spinor`) with `"component_rank":"3"`, correct `family`/`operator`/`symbol`; plus `cint*` legacy + `*_optimizer` symbols. `cargo build` regenerates `api_manifest.rs`. |
| **Raw API IDs** | `cintx-compat/src/raw.rs:111-151` | `pub const INT1E_IPOVLP_{CART,SPH}: Self = Self::Symbol("…")` etc. Routing is manifest-driven; no other change. |
| **Legacy wrappers** | `cintx-compat/src/legacy.rs:81,227,312` | `all_cint_wrappers!(…)` call + add to `LEGACY_WRAPPER_SYMBOLS` + the `misc`-family hardcoded match at line 312. The `legacy_wrapper_surface_matches_misc` test enforces sync. |
| **Kernel (THE math)** | `cintx-cubecl/src/kernels/{one_electron,two_electron,center_3c2e,ecp}.rs` | New operator-symbol branch applying `nabla1i`/`gout_ip1`. 1e: extend dispatcher at `one_electron.rs:486`. 2e: new `launch_two_electron` gradient path reusing `gout_ip1` + `build_2e_shape(li+1,…)`. |
| **Env params (iprinv only)** | `cintx-runtime/src/planner.rs:44`, `validator.rs`, `cintx-compat/src/raw.rs` | `rinv_orig: Option<[f64;3]>` field; read `env[4..6]`; validate `iprinv` requires it; thread to kernel. |
| **C-ABI** | `cintx-capi/src/shim.rs:9-33` | New `CintxRawApi` `#[repr(i32)]` variants + `from_i32` + `raw_id()` arms. |
| **Safe API** | `cintx-rs/src/api.rs` | **No change** — generic routing (except `int2e_ip1` needs Phase 18's arity-4 `evaluate`). |
| **Oracle FFI** | `cintx-oracle/src/vendor_ffi.rs` | `vendor_int1e_ipovlp_{cart,sph}` etc. FFI wrappers around vendored libcint 6.1.3. |
| **Oracle tests** | `cintx-oracle/tests/*_parity.rs` | `#[cfg(has_vendor_libcint)]` byte-identity tests at atol=1e-12 (the gate that flips pyscf_rs). |

---

## Wave / plan breakdown (proposed; planner refines)

- **Wave 1 (foundation):** 21-01 `PTR_RINV_ORIG` env plumbing; 21-02 manifest registration (all 6 + `int3c2e_ip1` correction).
- **Wave 2 (1e kernels):** 21-03 `int1e_ipovlp` + `int1e_ipkin`; 21-04 `int1e_ipnuc` + `int1e_iprinv`.
- **Wave 3 (2e/3c2e kernels):** 21-05 `int2e_ip1`; 21-06 `int3c2e_ip1` repair.
- **Wave 4 (ECP + close-out):** 21-07 `ECPscalar_iprinv` (gated on R4); 21-08 verification + pyscf_rs hand-off.

**Sequencing:** Wave 2 and the raw-path of Wave 3 are independent of Phase 18. The `int2e_ip1` safe-API arm is blocked on Phase 18's arity-4 dispatch — either land Phase 18 first or expose `int2e_ip1` through the raw/compat path pyscf-gto already uses. Confirm pyscf-gto's `intor.rs` call path before committing 21-05's surface (R6).

---

## Risks

- **R1** — `int3c2e_ip1` is a latent silent-wrong RUNTIME path (verified, not suspected). `center_3c2e.rs::launch_center_3c2e_typed` is operator-blind, scalar-output, no derivative. Fix in 21-06 (real kernel + flip oracle to `vendor_int3c2e_ip1`). `int3c1e_p2` has the identical misnomer — fold in only if a consumer needs it.
- **R2** — Rys roots >5 for high-l. The gradient's `li+1` pushes f/g quartets past nroots=5. Document the l-limit; defer high-l grads behind the Wheeler-fallback work.
- **R3** — F-order component layout mismatch. pyscf-gto declares component-leading `[3, …]` F-order in `layout_table.rs`. Kernel staging must match exactly or pyscf-rs repacks wrong. Validate against vendor layout in the oracle.
- **R4** — ECP scalar K-Taylor. `ECPscalar_iprinv` byte-identity needs PySCF-exact scalar ECP primitives (`K_TAB`/`ECPrad_part`). Confirm Phase 19's Cu/LANL2DZ gate exercises the exact path before starting 21-07; otherwise insert a K-Taylor-port plan first.
- **R5** — spinor variants. Register-but-`UnsupportedApi`; pyscf_rs needs only sph/cart.
- **R6** — Phase 18 coupling. `int2e_ip1` safe-API needs arity-4 dispatch. De-risk by confirming pyscf-gto's call path (raw vs safe) up front.

---

## Validation Architecture (for Nyquist)

### Test Framework

| Property | Value |
|----------|-------|
| **Framework** | `cargo test` / `cargo nextest` (Rust integration tests in `cintx-oracle/tests/`) |
| **Config file** | none — uses the existing workspace test harness + `#[cfg(has_vendor_libcint)]` gate |
| **Quick run command** | `cargo test -p cintx-cubecl` (kernel unit tests for the family under construction) |
| **Full suite command** | `cargo test -p cintx-oracle --features <profile>` (vendor-gated byte-identity parity) |
| **Estimated runtime** | ~60-180 s (vendor libcint build cached) |

### Phase Requirements → Test Map

| Req | Test type | Command / file | What it proves |
|-----|-----------|----------------|----------------|
| GRAD-01 | unit | `cargo test -p cintx-runtime rinv_orig` + `cargo test -p cintx-compat` | env round-trip; validator rejects `iprinv` without origin |
| GRAD-02 | integration | manifest-audit xtask + `cargo build` regen | all symbols carry `component_rank:"3"`; resolve through `eval_raw` |
| GRAD-03 | oracle parity | `cintx-oracle/tests/*ipovlp*_parity.rs` | `int1e_ipovlp` cart/sph vs vendor at atol=1e-12 |
| GRAD-04 | oracle parity | `cintx-oracle/tests/*ipkin*_parity.rs` | `int1e_ipkin` cart/sph vs vendor at atol=1e-12 |
| GRAD-05 | oracle parity | `cintx-oracle/tests/*ipnuc*_parity.rs` | `int1e_ipnuc` (sum over atoms) vs vendor at atol=1e-12 |
| GRAD-06 | oracle parity | `cintx-oracle/tests/*iprinv*_parity.rs` | `int1e_iprinv` (single origin) vs vendor at atol=1e-12 |
| GRAD-07 | oracle parity | `cintx-oracle/tests/*int2e_ip1*_parity.rs` | `int2e_ip1` s/p/d vs `vendor_int2e_ip1`; F-order layout (R3) |
| GRAD-08 | oracle parity | `cintx-oracle/tests/*int3c2e_ip1*_parity.rs` | real derivative vs `vendor_int3c2e_ip1` (oracle flipped) |
| GRAD-09 | oracle parity | `cintx-oracle/tests/*ecp*iprinv*_parity.rs` | `ECPscalar_iprinv` Cu/LANL2DZ vs vendor at atol=1e-12 |
| GRAD-10 | manual + integration | layout-vs-vendor check; ROADMAP/STATE/REQUIREMENTS updated; hand-off note | consumer-un-gate readiness |

### Property tests (in-tree, no vendor required)

- Determinism: repeated evaluation of the same shell tuple is bit-stable (ordered reduction, no FMA reorder).
- Component count: gradient output length == `3 × base_family_length` for each family.
- `iprinv` requires an origin: `validator` returns a typed error when `rinv_orig` is `None`.

### Sampling Rate (Nyquist gate)

- **After every task commit:** `cargo test -p cintx-cubecl` (quick kernel feedback).
- **After every plan wave:** full `cintx-oracle` vendor-gated parity for the families landed in that wave.
- **Before verify:** full suite green across the affected feature profiles.

### Wave 0 Gaps

- No new framework needed — the `#[cfg(has_vendor_libcint)]` oracle harness and `cargo test` already cover the phase. New `vendor_*_ip1`/`vendor_*iprinv` FFI wrappers (21-02/per-family plans) are the only test-infra additions.

### Eval dimensions (cross-cutting)

- Byte-identity vs vendored libcint 6.1.3 at atol=1e-12 (primary).
- F-order component-leading layout parity vs pyscf-gto `layout_table.rs` (R3).
- Determinism / bit-stability under the `release-oracle` profile pyscf_rs consumes.

---

*Research seeded 2026-05-26 from the verified proposal; consumed by gsd-planner.*
