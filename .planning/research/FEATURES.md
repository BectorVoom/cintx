# Feature Landscape

**Domain:** Full libcint API parity — cintx v1.2
**Researched:** 2026-04-04
**Confidence:** HIGH (primary sources: vendored libcint 6.1.3 source, design doc v0.4-resolved, compiled api_manifest.csv)
**Scope:** Subsequent milestone. Base 1e/2e/2c2e/3c1e/3c2e kernels, oracle parity, cart-to-sph transforms, the three-layer API surface, manifest-driven resolver, and CI governance gates are all already built and locked by Phase 10. This document covers only NEW capabilities needed for v1.2.

---

## Context: What Is Already Built (Do Not Reimplement)

The v1.1 deliverables are complete and locked behind oracle CI gates:

- Real CubeCL kernels for all five base families: 1e (overlap, kinetic, nuclear), 2e (ERI via Rys), 2c2e, 3c1e, 3c2e — all at oracle parity
- Cart-to-sph Condon-Shortley transforms (l=0..4) in `cintx-cubecl`
- Spinor interleaved staging scaffold in `cintx-cubecl/transform/c2spinor.rs` (stub-level only)
- Helper API group in `cintx-compat/src/helpers.rs`: AO count functions (`CINTcgtos_cart/sph/spinor`, `CINTlen_spinor`, totals, offsets), GTO norm, optimizer lifecycle (manifest-driven)
- Transform API in `cintx-compat/src/transform.rs`: `CINTc2s_bra_sph`, `CINTc2s_ket_sph`, `CINTc2s_ket_sph1`, spinor staging wrappers (stubs for full four-variant spinor coverage)
- Legacy wrapper scaffold in `cintx-compat/src/legacy.rs`
- Feature gates `with-f12` and `with-4c1e` wired through Cargo feature graph, sph-envelope enforced in raw eval path
- `unstable-source-api` gate wired through Cargo; rejections fire correctly when gate is off
- 4c1e kernel in `cintx-cubecl/src/kernels/center_4c1e.rs` with `Validated4C1E` classifier (validates and executes within cart/sph, scalar, max(l)<=4 envelope)
- Oracle harness with per-family tolerance tables; CI currently reports 0 mismatches for all five base families
- Compiled manifest lock at `crates/cintx-ops/generated/compiled_manifest.lock.json` covering base profile

---

## Feature Landscape

### Table Stakes (Users Expect These)

These are the features that block claiming "full API parity with libcint 6.1.3." Missing any of them means the manifest coverage audit fails, the release gate does not close, or downstream users find hard gaps when porting from upstream libcint.

| Feature | Why Expected | Complexity | Depends On |
|---------|--------------|------------|------------|
| F12/STG/YP real kernels — sph-only | `with-f12` profile promises `int2e_stg_sph`, `int2e_yp_sph` and all `_ip1`, `_ipip1`, `_ipvip1`, `_ip1ip2` derivatives (10 symbols total, per `cint2e_f12.c:12-201`). Already feature-gated and scaffold in place; evaluation still stubs out. | HIGH | `PTR_F12_ZETA` env slot already parsed; needs modified 2e Rys kernel with STG/Yukawa-modified quadrature weights |
| F12/STG/YP optimizer coverage | Each F12/YP variant requires a matching `_optimizer` symbol in the `with-f12` profile (confirmed by `cint2e_f12.c` `ALL_CINT` macro). Manifest audit will fail without them. | MEDIUM | F12 kernel infrastructure |
| F12/STG/YP oracle gate — with-f12 profile | CI must run oracle comparison under `with-f12` feature profile and pass at atol=1e-6, rtol=1e-4 (per design doc section 13.8). "Cart/spinor symbol count is zero" is itself a pass condition. | MEDIUM | F12 kernels + oracle harness profile extension |
| 4c1e full API surface — cart and sph, all validated operators | Current 4c1e kernel validates `Validated4C1E` envelope (cart/sph, scalar, max(l)<=4). The manifest lists `int4c1e_cart` and `int4c1e_sph` plus their optimizers (per `src/cint4c1e.c:324-357`). The release gate requires these to pass oracle comparison within the bug envelope. | HIGH | Existing `center_4c1e.rs` kernel + identity test `int4c1e_sph == (-1/4π)*trace(int2e_ipip1 + 2*int2e_ipvip1 + permuted)` |
| 4c1e bug-envelope rejection tests | Inputs outside `Validated4C1E` must return `UnsupportedApi { reason: "outside Validated4C1E" }`. These tests do not exist today for out-of-envelope cases. Release gate explicitly requires them (design doc 3.11.2, 16.4). | LOW | `Validated4C1E` classifier already written; needs test cases |
| 4c1e workaround path | `compat::workaround::int4c1e_via_2e_trace` for users needing 4c1e outside the envelope (design doc 3.11.2). Not yet implemented. | MEDIUM | 2e kernel infrastructure |
| Helper API oracle coverage | `cint.h.in:227-291` defines 34 helper/transform/optimizer-lifecycle symbols. Current `helpers.rs` implements the count/offset/norm group. Oracle comparison for helper APIs (reference counts, offsets, norms, transform outputs) is not yet wired into CI (design doc 14.1). | MEDIUM | Existing helper implementations; oracle harness extension |
| Full cart-to-spinor transform variants | `cint.h.in:283-291` declares four spinor transform functions: `CINTc2s_ket_spinor_sf1`, `CINTc2s_iket_spinor_sf1`, `CINTc2s_ket_spinor_si1`, `CINTc2s_iket_spinor_si1`. Current `c2spinor.rs` is a placeholder (amplitude-averaging stub). Full `C_{l,kappa}` coupling coefficient transforms are needed. | HIGH | `l`, `kappa` Clebsch-Gordan coefficient tables; `num-complex` for Complex64 output |
| Legacy wrapper oracle coverage | `cint2e_*` and `cNAME*` legacy wrappers exist in `cintx-compat/src/legacy.rs`. Oracle comparison for legacy symbol results vs upstream `cNAME_sph` calls is not yet gated. | LOW | Existing legacy wrappers; oracle harness extension |
| Unstable-source-api kernel stubs with oracle gate | `unstable-source-api` feature enables: `int1e_r2_origi`, `int1e_r4_origi`, `int1e_r2_origi_ip2`, `int1e_r4_origi_ip2`, `int1e_grids` (source-only 1e); `int2e_breit_r1p2`, `int2e_breit_r2p2` (Breit 2e); `int3c1e_r2_origk`, `int3c1e_r4_origk`, `int3c1e_r6_origk`, `int3c1e_ip1_r2_origk`, `int3c1e_ip1_r4_origk`, `int3c1e_ip1_r6_origk` (3c1e origk); `int3c2e_ssc` (3c2e ssc). These need real kernels or explicit `UnsupportedApi` returns with manifest entries (15 source-only symbols, from `api_manifest.csv`). | HIGH (Breit, origk) / MEDIUM (origi) | Existing base kernel infrastructure; `PTR_GRIDS`/`NGRIDS` env slots for grids family |
| Unified oracle tolerance atol=1e-12 | PROJECT.md states the v1.2 goal is "unified atol=1e-12 oracle tolerance for every family." Current per-family tolerances are coarser for 2c2e (1e-9), 3c1e (1e-7), etc. Tightening requires verifying that base kernels already achieve 1e-12 precision, and upgrading the CI threshold configuration in `compare.rs`. | MEDIUM | Existing kernels must be verified at 1e-12; may require kernel precision tuning for 2c2e/3c1e |
| Manifest lock covering full support matrix | `compiled_manifest.lock.json` must be regenerated for the union of `{base, with-f12, with-4c1e, with-f12+with-4c1e}` profiles (design doc 3.3.1). Current lock likely covers base profile only. | MEDIUM | `xtask manifest-audit` command; 4-profile build of vendored libcint |
| int1e_grids family implementation | `int1e_grids` is a source-only 1e family computing integrals over a numerical grid (`NGRIDS`, `PTR_GRIDS` env slots). Behind `unstable-source-api`. Important for DFT. Four derivative variants in the header: `int1e_grids_ip`, `_ipip`, `_ipvip`, `_spvsp`. | HIGH | `NGRIDS`/`PTR_GRIDS` env slot parsing; grid coordinate upload to GPU |

### Differentiators (Competitive Advantage)

Features that distinguish cintx from a minimal libcint port. Not required for the manifest coverage gate, but increase usefulness for downstream quantum chemistry code.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Spinor family oracle coverage (1e/2e/2c2e spinor representations) | Enables 4-component relativistic calculations in Rust. No existing Rust library offers verified spinor integral evaluation. | HIGH | Requires full `CINTc2s_ket_spinor_*` transforms; complex output via `num-complex::Complex64` |
| `int4c1e_via_2e_trace` workaround path with documentation | Provides a safe fallback for 4c1e callers outside the validated envelope. Other libraries leave users to implement this decomposition themselves. | MEDIUM | Requires 2e gradient family kernels (`int2e_ipip1`, `int2e_ipvip1`) already in base |
| Breit integral kernels (`int2e_breit_r1p2`, `int2e_breit_r2p2`) | Breit corrections to electron repulsion are needed for high-accuracy relativistic methods (CCSD(T) with Breit-Pauli). Very few libraries expose these. | HIGH | New 2e kernel variant with Breit-Pauli operator in `gout`; behind `unstable-source-api` |
| origk/origi source-only integral families | `int3c1e_r*_origk` and `int1e_r*_origi` are used in magnetic property calculations (GIAO, London orbitals). Behind `unstable-source-api`. | MEDIUM-HIGH | Modified 3c1e/1e kernels with origin-dependent operators |
| `int3c2e_ssc` (3c2e ssc variant) | Used in spin-orbit coupling integrals. Behind `unstable-source-api`. Currently sph+spinor only (no cart) per manifest. | MEDIUM | New 3c2e kernel variant |
| Configurable `PTR_F12_ZETA` and `PTR_RANGE_OMEGA` env slot access in safe API | Users can set range-separation and F12 zeta parameters through the typed `OperatorParams` struct rather than raw env array offsets. Currently available only through raw compat. | LOW | Builder/params extension; env slot already parsed in compat |
| Feature-matrix CI evidence report | A human-readable CI artifact listing which symbols are covered, which are passing, and which are `UnsupportedApi` per feature profile. Enables downstream users to audit coverage before depending on cintx. | LOW | `xtask manifest-audit` output formatting |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | What to Do Instead |
|---------|---------------|-----------------|-------------------|
| GTG (Gaussian-type geminal) integrals | Upstream has `PTR_GTG_ZETA` env slot and `WITH_GTG` build flag, suggesting support | Upstream explicitly marks GTG as "bugs in gtg type integrals" (`CMakeLists.txt:106-109`). No verified oracle exists. Adding it would give false confidence in broken results. | Keep `with-gtg` feature absent. Classify GTG as `planned-excluded` in manifest. Document the upstream bug explicitly. |
| Cart and spinor representations for F12/STG/YP | Some users expect cart/sph/spinor parity across all families | Upstream `src/cint2e_f12.c` exports sph-only symbols; cart and spinor versions do not exist in the compiled library (`cint2e_f12.c:12-201`, design doc 3.11.1). Adding them would require implementing a non-upstream extension. | Return `UnsupportedRepresentation` for F12 cart/spinor; make "cart/spinor symbol count is zero" itself a CI pass condition. |
| 4c1e beyond `Validated4C1E` envelope | Power users want 4c1e with spinor output or max(l)>4 | Upstream has known bugs in these cases (design doc 3.11.2). An oracle cannot validate them. Silently returning wrong results is worse than refusing. | Gate on `Validated4C1E` classifier; return `UnsupportedApi` with explicit reason; provide `int4c1e_via_2e_trace` for users needing the functionality safely. |
| Asynchronous `evaluate()` API | GPU users want to pipeline work without blocking | Async public APIs complicate the compat contract, error propagation, and oracle verification. The design explicitly rejects this (design doc 1.3). | Keep public API synchronous. Internal CubeCL command queues can still pipeline; the caller always blocks at the `evaluate()` boundary. |
| Bitwise-identical libcint internals | Some callers test for exact floating-point equality | Bitwise identity would require reproducing the DRK scratch layout, stack allocator, and SIMD patterns from libcint's C code — making the GPU backend impossible. The oracle tolerance contract (atol=1e-12) is the appropriate compatibility claim. | Document the oracle tolerance contract clearly. Provide `oracle_compare` xtask command for users who want to verify accuracy. |
| `ndarray` in the core API surface | Familiar array abstraction for Python/NumPy users | `ndarray` imposes strides and layouts incompatible with compat's flat C-ordered buffers, and adds a public dependency that constrains downstream library authors. | Keep `IntegralTensor<T>` with typed views. Accept `ndarray` in examples and test helpers (behind `dev-dependency` only). |

---

## Feature Dependencies

```
[F12/STG/YP real kernels (STG-modified Rys quadrature)]
  └──requires──> [base 2e Rys kernel infrastructure] (already built, v1.1)
  └──requires──> [PTR_F12_ZETA env slot parsed] (already built, in cint.h.in slot 9)
  └──requires──> [with-f12 feature gate wired] (already built)
  └──produces──> [F12 optimizer symbols in with-f12 profile]
  └──produces──> [F12 oracle gate — with-f12 profile CI job]

[4c1e full oracle coverage]
  └──requires──> [center_4c1e.rs kernel] (already built, Validated4C1E scope)
  └──requires──> [Validated4C1E classifier] (already built)
  └──requires──> [int2e_ipip1, int2e_ipvip1 kernels] (in base 2e family, v1.1)
  └──produces──> [int4c1e_via_2e_trace workaround]
  └──produces──> [4c1e oracle gate — with-4c1e profile CI job]

[full cart-to-spinor transforms]
  └──requires──> [Clebsch-Gordan / kappa coupling coefficient tables]
  └──requires──> [num-complex Complex64 output path in compat writer]
  └──produces──> [spinor oracle coverage for 1e/2e families]

[unstable-source-api: origi/origk/Breit/grids/ssc]
  └──requires──> [base 1e/2e/3c1e/3c2e kernel infrastructure] (already built, v1.1)
  └──requires──> [NGRIDS/PTR_GRIDS env slots] (defined in cint.h.in:43-45, parsing needed)
  └──requires──> [unstable-source-api feature gate wired] (already built)
  └──produces──> [unstable-source oracle gate — unstable-source-api CI job]

[unified atol=1e-12 oracle tolerance]
  └──requires──> [all five base kernels pass at 1e-12] (to be verified — may need kernel tuning)
  └──requires──> [compare.rs tolerance table update]
  └──requires──> [4c1e tolerance remains at 1e-6 per design doc 13.8] (exception to unification)
  └──requires──> [F12 tolerance remains at 1e-6 per design doc 13.8] (exception to unification)

[manifest lock — full support matrix]
  └──requires──> [4-profile build: base, with-f12, with-4c1e, with-f12+with-4c1e]
  └──requires──> [xtask manifest-audit command] (already exists)
  └──produces──> [compiled_manifest.lock.json with all profiles]
  └──enables──> [manifest diff CI gate — release gate item 1]

[helper API oracle coverage]
  └──requires──> [helpers.rs implementations] (already built)
  └──requires──> [oracle harness extension to compare helper return values]
  └──produces──> [helper comparison CI gate — release gate item 7]
```

### Dependency Notes

- **F12 kernels require base 2e Rys infrastructure:** The STG operator modifies the Coulomb operator as `e^{-zeta*r12}/r12` (STG) or `e^{-zeta*r12}` (YP). In Rys quadrature, this changes the quadrature weight but not the recurrence structure. The same Boys-function-derived root/weight tables apply, but with a modified argument. This is a targeted extension of the existing 2e kernel, not a full rewrite.

- **Full spinor transforms depend on Clebsch-Gordan tables:** The `CINTc2s_ket_spinor_sf1` family applies coupling coefficients `C_{l,kappa}` that mix the two spinor components. These coefficients are tabulated by `(l, kappa, m_j)` and must be GPU-resident. The current `c2spinor.rs` stub (amplitude-averaging) does not implement this — it is a placeholder that produces incorrect spinor integrals.

- **Unstable-source grids family requires PTR_GRIDS parsing:** The `int1e_grids` family reads the grid point coordinates from `env[PTR_GRIDS]` and `env[NGRIDS]` (indices 12 and 11 in `cint.h.in`). The raw validator must parse and validate these env slots, and the kernel must upload the grid coordinates to the device. No other existing family uses this mechanism.

- **Unified atol=1e-12 may require tuning for 2c2e/3c1e:** Current oracle tolerances for 2c2e and 3c1e are 1e-9 and 1e-7 respectively. Tightening to 1e-12 requires verifying that the existing kernels already achieve this precision in double arithmetic. If not, the Rys root/weight tables or contraction accumulation may need adjustments. The design doc's tolerance table (section 13.8) lists exceptions: 4c1e (1e-6), F12 (1e-6), and spinor/Breit (1e-6) are explicitly coarser. "Unified 1e-12" applies to the base scalar families only.

- **F12 conflicts with GTG:** Both use `PTR_F12_ZETA` / `PTR_GTG_ZETA` env slots at positions 9 and 10. GTG must never be enabled. If a caller sets `env[PTR_GTG_ZETA]` and requests a GTG symbol, the manifest resolver must return `UnsupportedApi` before any kernel is dispatched.

---

## MVP Definition

### What Must Ship for v1.2 Release Gate to Close

Per design doc section 14.1 and PROJECT.md active requirements:

- [ ] **F12/STG/YP real kernels** — `int2e_stg_sph`, `int2e_yp_sph`, and all five derivative variants each for STG and YP (10 kernel symbols), plus matching optimizer symbols. Oracle gate passes under `with-f12` profile at atol=1e-6.
- [ ] **F12 sph-only enforcement CI gate** — "Cart/spinor symbol count is zero" asserted in the `with-f12` profile CI job.
- [ ] **4c1e full oracle coverage** — `int4c1e_cart` and `int4c1e_sph` pass oracle comparison within `Validated4C1E` at atol=1e-6. Bug-envelope rejection tests verify `UnsupportedApi` for inputs outside.
- [ ] **4c1e workaround path** — `compat::workaround::int4c1e_via_2e_trace` implemented and tested.
- [ ] **Helper API oracle comparison** — count/offset/norm functions verified against upstream libcint 6.1.3 output. CI gate added for helper comparison (release gate item 7).
- [ ] **Legacy wrapper oracle comparison** — legacy `cNAME_sph` symbols verified against upstream. CI gate added.
- [ ] **Full cart-to-spinor transform variants** — All four `CINTc2s_*spinor*` variants implemented correctly (not stub). Spinor oracle gate for 1e family added.
- [ ] **Unstable-source-api: at minimum origi/origk/grids families** — `int1e_r2_origi`, `int1e_r4_origi`, `int3c1e_r2_origk`, `int3c1e_r4_origk`, `int3c1e_r6_origk`, `int1e_grids` behind `unstable-source-api` with oracle gate under that profile.
- [ ] **Manifest lock regenerated for full support matrix** — `compiled_manifest.lock.json` covers all four profiles; `xtask manifest-audit` CI job passes with zero diff.
- [ ] **Unified oracle tolerance tightening** — base scalar families (1e, 2e, 2c2e, 3c1e, 3c2e) verified at atol=1e-12 and `compare.rs` thresholds updated; 4c1e and F12 exceptions documented.

### Defer If Needed (v1.2.x or v1.3)

- [ ] **Breit integral kernels** (`int2e_breit_r1p2`, `int2e_breit_r2p2`) — Behind `unstable-source-api`. Requires new 2e kernel variant with Breit-Pauli operator. Can ship as `UnsupportedApi` stubs initially with TODO manifest entry.
- [ ] **`int3c2e_ssc`** — Behind `unstable-source-api`. Sph+spinor only. New 3c2e variant.
- [ ] **Spinor oracle coverage for 2e/2c2e families** — After 1e spinor is proven, extend to 2e and 2c2e.
- [ ] **Batched shell-quartet dispatch** — Performance optimization, not correctness.
- [ ] **`int1e_r4_origi_ip2` and `int1e_r2_origi_ip2`** — Higher-derivative origi variants. Can ship as `UnsupportedApi` initially.

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| F12/STG/YP real kernels (10 symbols) | HIGH — needed by F12-R12 methods, MP2-F12, CCSD(F12) | HIGH — STG-modified Rys quadrature | P1 |
| 4c1e full oracle coverage + rejection tests | HIGH — closes the `with-4c1e` release gate | MEDIUM — kernel exists, need oracle + tests | P1 |
| 4c1e workaround path (`int4c1e_via_2e_trace`) | HIGH — required by design doc for safe fallback | MEDIUM — 2e gradient kernels already built | P1 |
| Manifest lock — full support matrix | HIGH — blocks release gate item 1 | MEDIUM — 4-profile libcint build + xtask | P1 |
| Unified atol=1e-12 oracle tolerance | HIGH — stated v1.2 goal; blocks release | MEDIUM — verify precision, update compare.rs | P1 |
| Helper API oracle comparison CI gate | MEDIUM — users trust helper counts/offsets implicitly | LOW — helpers work, just need oracle wiring | P1 |
| Legacy wrapper oracle CI gate | MEDIUM — migration safety | LOW — legacy wrappers work, need oracle wiring | P1 |
| Full cart-to-spinor transform variants | HIGH — spinor integrals produce wrong values today | HIGH — Clebsch-Gordan coefficients, complex output | P1 |
| Unstable-source-api: origi/origk families | MEDIUM — magnetic property methods in QC | HIGH — new 1e/3c1e operator variants | P2 |
| int1e_grids family | MEDIUM — DFT grid integration | HIGH — NGRIDS/PTR_GRIDS env parsing, GPU upload | P2 |
| Breit integral kernels | LOW — high-accuracy relativistic only | HIGH — new 2e Breit-Pauli operator | P3 |
| int3c2e_ssc | LOW — spin-orbit coupling methods | MEDIUM — new 3c2e variant | P3 |
| Spinor oracle for 2e/2c2e | MEDIUM — after 1e spinor proven | HIGH — complex output path across more families | P2 |
| Feature-matrix CI evidence report | LOW — tooling quality | LOW — xtask output formatting | P3 |

**Priority key:**
- P1: Required for v1.2 release gate (manifest audit, oracle CI, design doc requirements)
- P2: Needed for full `unstable-source-api` profile coverage; can slip to v1.2.x
- P3: Nice to have, future consideration (v1.3 or later)

---

## F12/STG/YP Integral Family — Technical Detail

This is the highest-complexity P1 item. Key technical facts sourced directly from `libcint-master/src/cint2e_f12.c` and design doc section 3.11.1.

**What upstream exports (HIGH confidence — from source):**

| Symbol | ng[] array | Description |
|--------|-----------|-------------|
| `int2e_stg_sph` | `{0,0,0,0,0,1,1,1}` | Plain STG integral `(ij|e^{-zeta*r12}/r12|kl)` |
| `int2e_stg_ip1_sph` | `{1,0,0,0,1,1,1,3}` | Nabla on i |
| `int2e_stg_ipip1_sph` | `{2,0,0,0,2,1,1,9}` | Nabla Nabla on i |
| `int2e_stg_ipvip1_sph` | `{1,1,0,0,2,1,1,9}` | Nabla i cross Nabla j |
| `int2e_stg_ip1ip2_sph` | `{1,0,1,0,2,1,1,9}` | Nabla i, Nabla k |
| `int2e_yp_sph` | `{0,0,0,0,0,1,1,1}` | Plain Yukawa integral `(ij|e^{-zeta*r12}|kl)` |
| `int2e_yp_ip1_sph` | `{1,0,0,0,1,1,1,3}` | Nabla on i |
| `int2e_yp_ipip1_sph` | `{2,0,0,0,2,1,1,9}` | Nabla Nabla on i |
| `int2e_yp_ipvip1_sph` | `{1,1,0,0,2,1,1,9}` | Nabla i cross Nabla j |
| `int2e_yp_ip1ip2_sph` | `{1,0,1,0,2,1,1,9}` | Nabla i, Nabla k |

Each has a matching `_optimizer` symbol using `CINTall_2e_stg_optimizer`.

**Algorithm difference from plain 2e:** STG and YP modify the Coulomb operator `1/r12` by multiplying it by `e^{-zeta*r12}` (STG: Slater-type geminal, also known as range-separated short-range with real exponential) or replacing it entirely with `e^{-zeta*r12}` (YP: Yukawa potential). In Rys quadrature the operator enters through the Boys function argument: `x = (p*q/(p+q)) * R_PQ^2 + zeta^2/(4*(p+q))` (approximately; exact form depends on the STG/YP kernel type). The `CINTinit_int2e_stg_EnvVars` and `CINTinit_int2e_yp_EnvVars` functions (referenced in `cint2e_f12.c`) set up the `EnvVars` struct with the modified operator. The `ng[]` array drives the recurrence depth and component count.

**PTR_F12_ZETA env slot:** Position 9 in `env[]` (defined in `cint.h.in:39`). Already documented in the compat layer. Must be read and passed to the CubeCL kernel as a parameter.

**Oracle tolerance:** atol=1e-6, rtol=1e-4 (design doc 13.8). Looser than plain 2e because the exponential factor introduces additional floating-point sensitivity.

---

## 4c1e Beyond Initial Envelope — Technical Detail

**What Validated4C1E currently covers (HIGH confidence — from center_4c1e.rs):**
- Representations: Cart and Spheric only (no spinor)
- Component rank: scalar only
- Angular momentum: max(l_i, l_j, l_k, l_l) <= 4
- dims: NULL or natural shape
- Backend: CubeCL

**What the release gate requires for v1.2:**
1. Oracle comparison passes for `int4c1e_cart` and `int4c1e_sph` within this envelope
2. Identity test `int4c1e_sph == (-1/4π) * trace(int2e_ipip1 + 2*int2e_ipvip1 + permuted)` passes over all shell combinations
3. Tests confirm `UnsupportedApi` for: spinor input, max(l)>4, non-natural dims

**What does NOT expand for v1.2:** The validated envelope boundary itself does not change. The work is writing the oracle fixtures and CI tests that confirm the current kernel is correct within the existing envelope.

**The workaround path:** `compat::workaround::int4c1e_via_2e_trace` implements the identity above using the existing 2e kernel family. This gives users a safe fallback without expanding the validated envelope. Design doc section 3.11.2 explicitly requires this path.

---

## Unstable-Source-API Families — Inventory

Source-only families from `api_manifest.csv` that must be covered (or explicitly rejected) under `unstable-source-api`:

| Family | Source | Form | Category | Notes |
|--------|--------|------|----------|-------|
| `int1e_r2_origi` | `src/cint1e_a.c:16-71` | optimizer;cart;sph;spinor | 1e | Origin-dependent r^2 |
| `int1e_r4_origi` | `src/cint1e_a.c:73-147` | optimizer;cart;sph;spinor | 1e | Origin-dependent r^4 |
| `int1e_r2_origi_ip2` | `src/cint1e_a.c:150-214` | optimizer;cart;sph;spinor | 1e | r^2 origi + nabla j |
| `int1e_r4_origi_ip2` | `src/cint1e_a.c:217-316` | optimizer;cart;sph;spinor | 1e | r^4 origi + nabla j |
| `int1e_grids` | `src/cint1e_grids.c:368` | optimizer;cart;sph;spinor | 1e | Grid-based 1e; requires NGRIDS/PTR_GRIDS |
| `int2e_breit_r1p2` | `src/breit.c:227` | optimizer;cart;sph;spinor | 2e | Breit-Pauli retardation |
| `int2e_breit_r2p2` | `src/breit.c:316` | optimizer;cart;sph;spinor | 2e | Breit-Pauli gauge |
| `int3c1e_r2_origk` | `src/cint3c1e_a.c:16-72` | optimizer;cart;sph;spinor | 3c1e | Origin-k r^2 |
| `int3c1e_r4_origk` | `src/cint3c1e_a.c:75-150` | optimizer;cart;sph;spinor | 3c1e | Origin-k r^4 |
| `int3c1e_r6_origk` | `src/cint3c1e_a.c:153-286` | optimizer;cart;sph;spinor | 3c1e | Origin-k r^6 |
| `int3c1e_ip1_r2_origk` | `src/cint3c1e_a.c:358` | optimizer;cart;sph;spinor | 3c1e | Derivative + origk r^2 |
| `int3c1e_ip1_r4_origk` | `src/cint3c1e_a.c:461` | optimizer;cart;sph;spinor | 3c1e | Derivative + origk r^4 |
| `int3c1e_ip1_r6_origk` | `src/cint3c1e_a.c:674` | optimizer;cart;sph;spinor | 3c1e | Derivative + origk r^6 |
| `int3c2e_ssc` | `src/cint3c2e.c:729-759` | optimizer;sph;spinor | 3c2e | Spin-orbit coupling; no cart form |

**Minimum viable coverage for v1.2:** origi (1e), origk (3c1e), and grids. Breit and ssc can ship as manifest-registered `UnsupportedApi` stubs for v1.2.

---

## Helper and Transform API Inventory

From `cint.h.in:227-291` (HIGH confidence — read from vendored source):

**Count/Offset helpers (already implemented in `helpers.rs`):**
- `CINTlen_cart(l)` — Cartesian component count for angular momentum l
- `CINTlen_spinor(bas_id, bas)` — Spinor component count for a shell
- `CINTcgtos_cart/sph/spinor(bas_id, bas)` — Contracted GTO count per shell
- `CINTcgto_cart/sph/spinor(bas_id, bas)` — Single contraction count per shell
- `CINTtot_pgto_spheric/spinor(bas, nbas)` — Total primitive GTO counts
- `CINTtot_cgto_cart/sph/spinor(bas, nbas)` — Total contracted GTO counts
- `CINTshells_cart/sph/spinor_offset(ao_loc, bas, nbas)` — Shell AO offsets

**Normalization:**
- `CINTgto_norm(n, a)` — GTO normalization coefficient

**Transform (partially implemented in `transform.rs`):**
- `CINTc2s_bra_sph(sph, nket, cart, l)` — Bra cart-to-sph
- `CINTc2s_ket_sph(sph, nket, cart, l)` — Ket cart-to-sph
- `CINTc2s_ket_sph1(sph, cart, lds, ldc, l)` — Ket cart-to-sph with explicit strides
- `CINTc2s_ket_spinor_sf1` — Ket spinor (scalar factor, currently stub)
- `CINTc2s_iket_spinor_sf1` — Ket spinor imaginary (currently stub)
- `CINTc2s_ket_spinor_si1` — Ket spinor (spin-orbit factor, currently stub)
- `CINTc2s_iket_spinor_si1` — Ket spinor imaginary spin-orbit (currently stub)

**Optimizer lifecycle (already wired through manifest):**
- `CINTinit_2e_optimizer` / `CINTdel_2e_optimizer`
- `CINTinit_optimizer` / `CINTdel_optimizer`

**Oracle coverage needed:** Reference values from upstream libcint for count/offset functions (deterministic integer outputs, easy to verify), norm function (floating-point, needs tolerance), and transform functions (matrix apply, needs floating-point tolerance).

---

## Sources

- `libcint-master/include/cint.h.in` — Helper/transform/optimizer API declarations (lines 227-291), env slot definitions (lines 26-45) — HIGH confidence
- `libcint-master/include/cint_funcs.h` — Full integral family declarations — HIGH confidence
- `libcint-master/src/cint2e_f12.c` — F12/STG/YP kernel implementations and ng[] arrays (lines 12-201) — HIGH confidence
- `libcint-master/src/cint4c1e.c` — 4c1e kernel (lines 324-357) — HIGH confidence
- `docs/design/api_manifest.csv` — Source-only family inventory (216 entries total, 15 source-only) — HIGH confidence
- `docs/design/cintx_detailed_design.md` — Design decisions: sections 3.11.1 (F12 matrix), 3.11.2 (4c1e envelope), 13.8 (tolerance table), 14.1 (release gate), 10.1 (feature matrix) — HIGH confidence
- `.planning/PROJECT.md` — Active requirements for v1.2 — HIGH confidence
- `crates/cintx-compat/src/helpers.rs` — Current helper implementation state — HIGH confidence
- `crates/cintx-compat/src/transform.rs` — Current transform implementation state (partially stubbed) — HIGH confidence
- `crates/cintx-cubecl/src/transform/c2spinor.rs` — Spinor transform stub (amplitude-averaging placeholder, not correct) — HIGH confidence
- `crates/cintx-cubecl/src/kernels/center_4c1e.rs` — Validated4C1E classifier implementation — HIGH confidence

---

*Feature research for: cintx v1.2 full API parity and unified oracle gate*
*Researched: 2026-04-04*
