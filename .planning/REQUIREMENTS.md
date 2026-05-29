# Requirements: cintx

**Defined:** 2026-03-21
**Core Value:** Deliver libcint-compatible results through a Rust-native API surface that stays type-safe, verifiable, and safe under memory pressure.

## v1.2 Requirements

### Helper & Transform Completion

- [x] **HELP-01**: Oracle harness compares every helper symbol in the manifest against vendored libcint with atol=1e-12
- [x] **HELP-02**: Oracle harness compares every transform symbol in the manifest against vendored libcint with atol=1e-12
- [x] **HELP-03**: Oracle harness compares every legacy wrapper symbol in the manifest against vendored libcint with atol=1e-12
- [x] **HELP-04**: CI helper-legacy-parity gate passes with 0 mismatches across all four feature profiles

### 4c1e Kernel & Oracle

- [x] **4C1E-01**: int4c1e_sph evaluation produces real Rys quadrature results matching libcint 6.1.3 to atol=1e-12 within Validated4C1E envelope
- [x] **4C1E-02**: int4c1e_via_2e_trace workaround path produces results matching direct 4c1e evaluation
- [x] **4C1E-03**: Out-of-envelope 4c1e inputs return UnsupportedApi; spinor 4c1e returns UnsupportedApi unconditionally
- [x] **4C1E-04**: Oracle parity CI gate for with-4c1e profile passes with 0 mismatches at atol=1e-12

### Spinor Representation

- [ ] **SPIN-01**: Cart-to-spinor transform implements real Clebsch-Gordan coupling coefficients for all angular momenta up to g-function (l=4)
- [ ] **SPIN-02**: All CINTc2s_*spinor* transform variants (ket_spinor, iket_spinor, ket_spinor_sf, ket_spinor_si) are implemented
- [x] **SPIN-03**: Spinor-form base family evaluations (1e, 2e, 2c2e, 3c1e, 3c2e spinor) match libcint to atol=1e-12
- [ ] **SPIN-04**: kappa parameter is correctly interpreted and applied in spinor transform dispatch

### F12/STG/YP Kernels

- [x] **F12-01**: STG (Slater-type geminal) kernel implements modified Rys quadrature with tabulated polynomial roots matching libcint
- [x] **F12-02**: YP (Yukawa potential) kernel implements correct routing distinct from STG path
- [x] **F12-03**: All 10 with-f12 sph symbols pass oracle parity against libcint at atol=1e-12
- [x] **F12-04**: PTR_F12_ZETA (env[9]) is correctly plumbed through ExecutionPlan to kernel launchers
- [x] **F12-05**: Oracle fixtures validate that zeta=0 is rejected or produces Coulomb-equivalent results explicitly

### Unstable-Source API

- [x] **USRC-01**: origi family (4 symbols, 1e) implemented behind unstable-source-api gate with oracle parity at atol=1e-12
- [x] **USRC-02**: grids family (1e grid-based integrals) implemented with NGRIDS/PTR_GRIDS env parsing and oracle parity at atol=1e-12
- [x] **USRC-03**: Breit family (2 symbols, 2e) implemented behind unstable-source-api with oracle parity at atol=1e-12
- [x] **USRC-04**: origk family (6 symbols, 3c1e) implemented behind unstable-source-api with oracle parity at atol=1e-12
- [x] **USRC-05**: ssc family (1 symbol, 3c2e) implemented behind unstable-source-api with oracle parity at atol=1e-12
- [x] **USRC-06**: Nightly CI job runs oracle with --include-unstable-source=true and 0 mismatches

### Oracle & Tolerance Unification

- [x] **ORAC-01**: Oracle tolerance unified to atol=1e-12 for every family with no per-family exceptions
- [x] **ORAC-02**: Four-profile manifest lock regenerated covering all implemented APIs
- [x] **ORAC-03**: CI oracle-parity gate passes all four profiles (base, with-f12, with-4c1e, with-f12+with-4c1e) at atol=1e-12
- [x] **ORAC-04**: Existing base families (1e, 2e, 2c2e, 3c1e, 3c2e) pass oracle at tightened atol=1e-12

## v1.3 Requirements

### Plain-Coulomb Gradient Integrals (Phase 21)

> The 6 plain-Coulomb first-derivative (∂/∂nuclear-coordinate) integral families every HF/DFT/MP2/CCSD analytical gradient needs, byte-identical to libcint 6.1.3, plus the `int3c2e_ip1` stub repair. Landing these un-gates pyscf_rs Phase 7's analytical-gradient numeric arms with zero pyscf_rs rework. Source: `.planning/notes/phase-21-coulomb-gradient-intors-PLAN.md`.

- [x] **GRAD-01**: `PTR_RINV_ORIG` env slot (`env[4..6]`) is plumbed end-to-end following the `f12_zeta` precedent — `OperatorEnvParams.rinv_orig: Option<[f64;3]>` field, `raw.rs::eval_raw` env-read, `validator.rs` gate (an `iprinv` operator without an origin is rejected), and the origin threaded into the `one_electron`/`ecp` kernels; a `with_rinv_origin`-style setter is exposed on the safe-API options. Env round-trip and validator-rejection unit tests pass.
- [x] **GRAD-02**: All 6 gradient families plus the `int3c2e_ip1` correction are registered in `compiled_manifest.lock.json` with `"component_rank":"3"` (per cart/sph/spinor representation), with matching RawApiId consts, legacy wrappers, and CAPI enum variants; `cargo build` regenerates `api_manifest.rs`; the manifest-audit xtask is green and every symbol resolves through `eval_raw` (returning `UnsupportedApi` from kernels until its kernel lands).
- [x] **GRAD-03**: `int1e_ipovlp` (cart + sph, 3 components) matches vendored libcint 6.1.3 at atol=1e-12 on the H2O/STO-3G corpus.
- [x] **GRAD-04**: `int1e_ipkin` (cart + sph, 3 components) matches vendored libcint 6.1.3 at atol=1e-12 on the H2O/STO-3G corpus.
- [x] **GRAD-05**: `int1e_ipnuc` (cart + sph, 3 components; ∇ on the bra center, summed over all nuclei) matches vendored libcint 6.1.3 at atol=1e-12.
- [x] **GRAD-06**: `int1e_iprinv` (cart + sph, 3 components; single rinv origin via the GRAD-01 env slot, no `-Z_C` factor) matches vendored libcint 6.1.3 at atol=1e-12.
- [x] **GRAD-07**: `int2e_ip1` (arity-4, 3 components; component-leading `[3, nl, nk, nj, ni]` F-order matching pyscf-gto `layout_table.rs`) matches vendored `int2e_ip1` at atol=1e-12 for s/p/d quartets.
- [x] **GRAD-08**: `int3c2e_ip1` ships a real derivative kernel that replaces the operator-blind scalar stub in `center_3c2e.rs`, and its oracle reference is flipped from the plain `vendor_int3c2e` to `vendor_int3c2e_ip1`; matches at atol=1e-12.
- [x] **GRAD-09**: `ECPscalar_iprinv` (per-nucleus ECP force; single rinv origin, no all-slot `-Z_C` accumulation) matches vendored libcint at atol=1e-12 on Cu/LANL2DZ — gated on confirming Phase 19's scalar-ECP K-Taylor byte-identity path (Risk R4).
- [x] **GRAD-10**: Phase verification + pyscf_rs hand-off: the component-leading F-order layout is validated against the vendor layout (Risk R3); cintx ROADMAP/STATE/REQUIREMENTS are updated; and a hand-off note records which pyscf_rs Phase 7 `workflow_dispatch` gradient arms now un-gate (and the `int3c2e_ip1` re-gating history).

## v1.4 Requirements

> Full libcint 6.1.3 family parity: implement every remaining (~140) integral family to byte-identity at atol=1e-12 under the vendor-gated oracle (`--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`). High angular-momentum (d/f) byte-identity is IN SCOPE (FND-02). Source: `.planning/research/SUMMARY-v1.4.md`.
>
> **Per-family surface (v1.4 scope decision):** each family is `manifest row (component_rank) → RawApiId const → kernel → vendor FFI (vendor_int*) + byte-identity oracle parity test → flip oracle_covered`. The C ABI shim (`cintx-capi`) enum variants and the legacy `cint*` wrappers (`cintx-compat/legacy.rs`) are **NOT** added for v1.4 families — the oracle byte-identity gate exercises the raw `eval_raw` + vendor-FFI path only, so those C-interop surfaces add ceremony without contributing to validation (and were the chief source of Phase-21 drift). They can be added later per C-consumer need. The inbound **vendor FFI is kept** — it is the libcint reference the byte-identity test compares against.

### Foundations

- [x] **FND-01**: `PTR_COMMON_ORIG` gauge-origin env slot (`env[1..3]`) is plumbed end-to-end following the `PTR_RINV_ORIG` precedent — `OperatorEnvParams.common_orig: Option<[f64;3]>`, `raw.rs::eval_raw` env-read, validator gate, `with_common_origin` safe-API setter; a non-zero gauge-origin oracle fixture exists and is the parity gate for moments + GIAO. Env round-trip + validator unit tests pass.
- [ ] **FND-02**: Rys `nroots≥6` Wheeler-fallback is implemented so high angular-momentum (d/f) shells reach byte-identity; closes `.planning/todos/pending/rys-nroots-ge6-wheeler-fallback.md`. No family returns `UnsupportedApi` purely due to `nroots>5`.
- [ ] **FND-03**: Complex/imaginary output capability — `complex_interleaved` is set per-family from driver routing (not the representation string), `assert_flat_buffer_contract` fires on the flag, and staging is sized `2×ncomp×…`; a purely-imaginary family (e.g. `int1e_igovlp`) round-trips through the safe API without silent zeroing.
- [ ] **FND-04**: Spinor-derivative transform (Gap B1) — `cart_to_spinor_sf_derivative_*` in `c2spinor.rs`; `int1e_ipovlp_spinor` and sibling `ip`-decorated spinor families move from `UnsupportedApi` to byte-identity at atol=1e-12 (closes the Phase-21 R5/D-03 deferral).
- [ ] **FND-05**: Spin-included `c2s_si` 4-block (`gc_x/gc_y/gc_z/gc_1`) spinor transform + σ·p G-tensor assembler module — validated against a kappa-bearing relativistic oracle fixture at atol=1e-12; the σ-coupling matches libcint `c2s_si_1e`.
- [ ] **FND-06**: High-rank (component_rank 9/27/81) staging is fail-closed — an upfront size assertion replaces the `if dst < staging.len()` scatter guards (no silent partial writes), and the chunk planner's OOM-safe-stop is re-validated at rank 81.

### Group 1 — Remaining 1st-Derivatives

- [ ] **DRV1-01**: `int2e_ip2` (arity-4, ∇ on the ket bra-center) matches vendored libcint 6.1.3 at atol=1e-12 (cart+sph).
- [ ] **DRV1-02**: `int1e_ipovlpip`, `int1e_ipkinip`, `int1e_ipnucip` (∇ on both bra and ket) match at atol=1e-12 (cart+sph).
- [ ] **DRV1-03**: `int3c1e_ip1` and `int3c1e_iprinv` match at atol=1e-12 (cart+sph).
- [ ] **DRV1-04**: `int2c2e_ip1` and `int2c2e_ip2` match at atol=1e-12 (cart+sph).
- [ ] **DRV1-05**: `int3c2e_ip2` matches at atol=1e-12 (cart+sph).

### Group 2 — Hessian & Higher-Order Derivatives

- [ ] **HESS-01**: `int1e_ipipovlp`, `int1e_ipipnuc`, `int1e_ipipkin`, `int1e_ipiprinv` (component_rank=9) match at atol=1e-12 (cart+sph).
- [ ] **HESS-02**: The 2e Hessian set (`int2e_ipip1`, `int2e_ipvip1`, `int2e_ip1ip2`, `int2e_ipip1ipip2`) — promoted from `unstable` where present — match at atol=1e-12 (cart+sph).
- [ ] **HESS-03**: `int2c2e_ipip1`, `int3c2e_ipip1`, `int3c2e_ipip2` match at atol=1e-12 (cart+sph).
- [ ] **HESS-04**: 3rd/4th-order families (`int1e_ipipipnuc`, `int1e_ipipipiprinv`, and siblings) match at atol=1e-12 (cart+sph), with `ng[]`-driven bra+ket headroom.

### Group 3 — Position / Multipole Moments

- [ ] **MOM-01**: Dipole `int1e_r` (and `int1e_r_origj`) match at atol=1e-12 against a non-zero gauge-origin fixture (cart+sph).
- [ ] **MOM-02**: `int1e_rr`, `int1e_r2`, `int1e_z`, `int1e_zz` (and `_origj` variants) match at atol=1e-12 (cart+sph).
- [ ] **MOM-03**: `int1e_rrr`, `int1e_rrrr`, `int1e_r4` (octupole/hexadecapole) match at atol=1e-12 (cart+sph), ket-side headroom from `ng[1]`.
- [ ] **MOM-04**: `int1e_p4`, `int1e_drinv`, plain `int1e_rinv`, `int1e_irp` match at atol=1e-12 (cart+sph).

### Group 4 — Relativistic Spin-Operator (spinor)

- [ ] **REL-01**: `int1e_spsp`, `int1e_spnucsp`, `int1e_sprinvsp` match vendored libcint at atol=1e-12 (spinor) via the FND-05 `c2s_si` path.
- [ ] **REL-02**: `int1e_srsr`, `int1e_sr`/`srnucsr`, `int1e_sigma`, `int1e_sp` match at atol=1e-12 (spinor).
- [ ] **REL-03**: `int2e_spsp1`, `int2e_srsr1` (and `spsp1spsp2`/`srsr1srsr2`) match at atol=1e-12 (spinor).
- [ ] **REL-04**: `int2e_ssp1ssp2`, `int2e_sps1sps2`, `int2e_vsp1*`, `int2e_spv1*` match at atol=1e-12 (spinor).

### Group 5 — GIAO / Magnetic-Property (NMR)

- [ ] **GIAO-01**: Spin-free 1e GIAO/CG families (`int1e_giao_*`, `int1e_cg_*`, `int1e_govlp/gnuc/gkin`, `int1e_ig*`, `int1e_a01gp`, `int1e_ia01p`) — purely imaginary — match at atol=1e-12 (cart+sph) via FND-03.
- [ ] **GIAO-02**: 2e GIAO families (`int2e_g1`, `int2e_gg1`, `int2e_ig1`, `int2e_giao_*`) match at atol=1e-12.
- [ ] **GIAO-03**: GIAO×σ slice (`int1e_spg*`, `int1e_spgnucsp`, `*_sa10*`, `int2e_cg_sa10*`/`giao_sa10*`) match at atol=1e-12 (spinor) via FND-05.

### Group 6 — Gauge / Breit–Gaunt 2e (apex)

- [ ] **BREIT-01**: `int2e_gauge_r1_{ssp,sps}{ssp,sps}` (4 symbols) match vendored libcint at atol=1e-12 (spinor).
- [ ] **BREIT-02**: `int2e_gauge_r2_{ssp,sps}{ssp,sps}` (4 symbols) match at atol=1e-12 (spinor).
- [ ] **BREIT-03**: Gaunt `ssp/sps` families match at atol=1e-12 (spinor), reusing the existing `launch_breit` decomposition.

### Full-Parity Verification

- [ ] **PARITY-01**: `manifest-audit` is green with EVERY libcint 6.1.3 family `oracle_covered=true` for its physical representations (cart/sph; spinor where physical, with σ families spinor-only); the full vendor-gated oracle suite is green; and the "unsupported libcint families" list (vs `cint_funcs.h` + supplemental headers) is empty. Full API parity is mechanically verifiable.

## v2 Requirements

### Expanded Coverage

- **NEXT-04**: Rust caller can use richer builder ergonomics and convenience APIs once the core compatibility surface is stable.
- **NEXT-05**: Maintainer can add deeper benchmark reporting and public performance dashboards once correctness and release gating are stable.
- **NEXT-06**: Project can consider additional compute backends or fallback strategies only if CubeCL becomes a sustained correctness or maintainability blocker.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Public GTG support | Explicitly excluded from initial GA because upstream marks GTG deprecated and incorrect |
| Bitwise-identical libcint internals | The project targets result compatibility, not implementation identity |
| Public Fortran wrapper reproduction | Not part of the Rust library's migration or compatibility goals |
| Public asynchronous API | Excluded from the initial design to keep execution, allocation, and compatibility behavior predictable |
| Best-effort partial writes on failure | Violates the OOM-safe stop and explicit-layout contract |
| CUDA/ROCm/Metal backend implementation | Architecture supports them via ResolvedBackend, but only wgpu+cpu in scope |
| h-function (l>=5) angular momentum | Register pressure risk, defer until g-function validated across all families |
| Screening/batching optimizations | Performance work after correctness is proven |
| C ABI (`cintx-capi`) enum variants for v1.4 families | C ABI shim is optional third-priority surface; new families are validated through the raw `eval_raw` + vendor-FFI byte-identity oracle path. Add later per C-consumer need. |
| Legacy `cint*` wrappers for v1.4 families | The oracle byte-identity gate does not exercise the `cint*` legacy surface; it adds ceremony (and Phase-21-style misc.h-rule drift) without contributing to validation. Add later if a C-style consumer needs the libcint names. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| HELP-01 | Phase 11 | Complete |
| HELP-02 | Phase 11 | Complete |
| HELP-03 | Phase 11 | Complete |
| HELP-04 | Phase 11 | Complete |
| 4C1E-01 | Phase 11 | Complete |
| 4C1E-02 | Phase 11 | Complete |
| 4C1E-03 | Phase 11 | Complete |
| 4C1E-04 | Phase 11 | Complete |
| SPIN-01 | Phase 12 | Pending |
| SPIN-02 | Phase 12 | Pending |
| SPIN-03 | Phase 12 | Complete |
| SPIN-04 | Phase 12 | Pending |
| F12-01 | Phase 13 | Complete |
| F12-02 | Phase 13 | Complete |
| F12-03 | Phase 13 | Complete |
| F12-04 | Phase 13 | Complete |
| F12-05 | Phase 13 | Complete |
| USRC-01 | Phase 14 | Complete |
| USRC-02 | Phase 14 | Complete |
| USRC-03 | Phase 14 | Complete |
| USRC-04 | Phase 14 | Complete |
| USRC-05 | Phase 14 | Complete |
| USRC-06 | Phase 14 | Complete |
| ORAC-01 | Phase 15 | Complete |
| ORAC-02 | Phase 15 | Complete |
| ORAC-03 | Phase 15 | Complete |
| ORAC-04 | Phase 15 | Complete |
| GRAD-01 | Phase 21 | Complete |
| GRAD-02 | Phase 21 | Complete |
| GRAD-03 | Phase 21 | Complete |
| GRAD-04 | Phase 21 | Complete |
| GRAD-05 | Phase 21 | Complete |
| GRAD-06 | Phase 21 | Complete |
| GRAD-07 | Phase 21 | Complete |
| GRAD-08 | Phase 21 | Complete |
| GRAD-09 | Phase 21 | Complete |
| GRAD-10 | Phase 21 | Complete |
| FND-01 | Phase 22 | Complete |
| DRV1-01 | Phase 23 | Pending |
| DRV1-02 | Phase 23 | Pending |
| DRV1-03 | Phase 23 | Pending |
| DRV1-04 | Phase 23 | Pending |
| DRV1-05 | Phase 23 | Pending |
| MOM-01 | Phase 24 | Pending |
| MOM-02 | Phase 24 | Pending |
| MOM-03 | Phase 24 | Pending |
| MOM-04 | Phase 24 | Pending |
| HESS-01 | Phase 25 | Pending |
| HESS-02 | Phase 25 | Pending |
| HESS-03 | Phase 25 | Pending |
| HESS-04 | Phase 25 | Pending |
| FND-02 | Phase 25 | Pending |
| FND-06 | Phase 25 | Pending |
| GIAO-01 | Phase 26 | Pending |
| GIAO-02 | Phase 26 | Pending |
| FND-03 | Phase 26 | Pending |
| FND-04 | Phase 27 | Pending |
| FND-05 | Phase 28 | Pending |
| REL-01 | Phase 29 | Pending |
| REL-02 | Phase 29 | Pending |
| REL-03 | Phase 29 | Pending |
| REL-04 | Phase 29 | Pending |
| GIAO-03 | Phase 30 | Pending |
| BREIT-01 | Phase 31 | Pending |
| BREIT-02 | Phase 31 | Pending |
| BREIT-03 | Phase 31 | Pending |
| PARITY-01 | Phase 31 | Pending |

**Coverage:**
- v1.2 requirements: 27 total (Phases 11-15)
- Complete: 17/27
- Pending: 10/27
- v1.3 gradient requirements: 10 total (Phase 21) — all Pending (planned 2026-05-26)
- v1.4 full-parity requirements: 30 total (FND-01..06, DRV1-01..05, HESS-01..04, MOM-01..04, REL-01..04, GIAO-01..03, BREIT-01..03, PARITY-01) — mapped to Phases 22-31, all Pending (roadmapped 2026-05-27)

---
*Requirements defined: 2026-03-21*
*Last updated: 2026-05-27 — v1.4 roadmap created (Phases 22-31); 30 v1.4 requirements mapped to traceability*
