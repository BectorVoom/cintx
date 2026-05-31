---
phase: 26-group-5-spin-free-giao-nmr-integrals-complex
plan: 03
subsystem: kernels
tags: [giao, nmr, 2e, complex-output, cubecl, oracle, manifest, vendor-parity, rys]

# Dependency graph
requires:
  - phase: 26-group-5-spin-free-giao-nmr-integrals-complex
    plan: 01
    provides: "complex_output manifest flag end-to-end; complex_interleaved 2x staging keyed off descriptor.entry.complex_output; fail-closed flat-buffer contract"
  - phase: 26-group-5-spin-free-giao-nmr-integrals-complex
    plan: 02
    provides: "GIAO registration + complex-aware parity pattern (tail-append manifest, RawApiId by symbol, vendor wrapper, 2x interleaved buffer, imag extraction, re==0 assertion)"
  - phase: 25-group-2-hessian-higher-order-derivatives
    provides: "Hess2e host-routed launcher (fill_g_tensor_2e -> rys_roots_host) + f12.rs nabla/gout helper pattern; FND-02 host Rys nroots 6..12"
  - phase: 22-gauge-origin-env-slot
    provides: "PTR_COMMON_ORIG gauge slot + build_h2o_sto3g_common_orig non-zero fixture"
provides:
  - "4 spin-free 2e GIAO families byte-identical to libcint 6.1.3 (cart+sph, atol=1e-12): int2e_g1 (rank 3), int2e_ig1 (rank 3), int2e_gg1 (rank 9), int2e_g1g2 (rank 9, D-16)"
  - "r0i_2e / r0k_2e position operators (CINTx1i_2e / CINTx1k_2e analogs, G2E_R0I / G2E_R0K macros) on the 2e G-tensor, generic-host f64"
  - "gout_g1 / gout_ig1 / gout_gg1 / gout_g1g2 (f12.rs): GIAO cross-product / 2nd-order gauge gout combos transcribed verbatim from intor4.c:1255 + intor2.c:19,148,283"
  - "Giao2eKind dispatch + launch_two_electron_giao2e: host-routed 2e GIAO launcher with complex-interleaved [re=0, im=value] staging, generic over F"
  - "giao_2e_parity.rs: 4 complex-aware vendor byte-identity tests on the non-zero-gauge non-square cross-center quartet"
affects: [30-giao-sigma, complex-output, giao]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "2e GIAO position operators: r0i_2e/r0k_2e mirror the f12.rs nabla1i/nabla1k_2e index-shift shape but use the formula f[n] = g[n+d_axis] + r_axis*g[n] (post-HRR G-tensor carries the true i/k center)"
    - "2e GIAO gout writes gout[n*rank+comp] DIRECTLY (no column-major reorder, unlike ipip1) — the libcint GIAO autocode emits the gout components in order"
    - "Complex-output 2e materialization: device emits REAL components; the 4-shell launcher writes interleaved [re=0, im=value] pairs (staging[2*dst+1]=value) for the FND-03 Complex<f64> view"

key-files:
  created:
    - crates/cintx-oracle/tests/giao_2e_parity.rs
  modified:
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-ops/src/generated/api_manifest.csv
    - crates/cintx-compat/src/raw.rs
    - crates/cintx-oracle/build.rs
    - crates/cintx-oracle/src/vendor_ffi.rs
    - crates/cintx-cubecl/src/kernels/f12.rs
    - crates/cintx-cubecl/src/kernels/two_electron.rs

key-decisions:
  - "int2e_g1g2 component_rank DERIVED from intor2.c ng[]={1,0,1,0,2,1,1,9} -> ng[TENSOR]=ng[7]=9 (D-16/D-13, not guessed); a too-low rank would silently truncate"
  - "Tail-append 12 manifest rows (4 families x cart/sph/spinor) -> ZERO positional OperatorId shift; all hardcoded test consts (INT4C1E=24, STG=106, IPIP1=116) preserved without re-anchoring (same as 26-02)"
  - "2e GIAO families are HOST-ROUTED through fill_g_tensor_2e (FND-02 host Rys), mirroring the Phase-25 Hess2e launcher exactly, NOT the on-device comptime scalar kernel — gout combos require multi-tensor R0I/R0K compositions the device path does not expose"
  - "Per-family libcint common_factor scale applied in the launcher: g1/ig1 x0.5, gg1 x0.25, g1g2 x-0.25 (intor4.c:1323 / intor2.c:87,222,361)"
  - "g1g2 headroom is (i+2, j+0, k+1): R0K(g0,i_l+1) needs k+1 for the ket shift and i+1 for the R0I composition that follows; R0I(R0K(.)) reaches i+2 (D-12: raise ket via ng, not bra)"
  - "Cross-center non-square QUARTET [H1-1s, O-2p, H1-1s, O-2p] = [3,2,3,2]: cross-center on BOTH electron pairs so g1g2's c=(ri-rj)(rk-rl) both-electron factor is genuinely exercised AND non-square defeats a transposed layout (D-12)"

patterns-established:
  - "2e GIAO purely-imaginary families: REAL host gout + 4-shell interleaved [re=0, im=value] staging is the canonical complex_output 2e path (no on-device complex arithmetic)"
  - "r0i_2e/r0k_2e are the 2e analogs of the 1e position-multiply decoration helpers; they slot beside the existing nabla1{i,j,k,l}_2e family in f12.rs and are reusable by any future GIAO/gauge 2e family"

requirements-completed: [GIAO-02]

# Metrics
duration: 70min
completed: 2026-05-31
---

# Phase 26 Plan 03: GIAO-02 Spin-Free 2e GIAO Families Summary

**Registered and implemented the 4 spin-free 2e GIAO magnetic-property families (int2e_g1, int2e_ig1, int2e_gg1, int2e_g1g2 — the last IN scope per D-16) on the FND-03 complex-output foundation; all 4 are byte-identical to libcint 6.1.3 (cart+sph, atol=1e-12) via new r0i_2e/r0k_2e position operators and verbatim-transcribed GIAO gout helpers, emitting a purely-imaginary Complex<f64> safe-API view from a host-routed ERI engine.**

## Performance

- **Duration:** ~70 min
- **Tasks:** 3
- **Files modified:** 8 (1 created, 7 modified)

## Accomplishments

- **Registration (Task 1):** 12 lock entries (4 families × cart/sph/spinor) appended at the manifest TAIL with the correct `component_rank` (3 for g1/ig1, 9 for gg1/g1g2) and `complex_output: true`, preserving every positional `OperatorId` (zero-shift). `int2e_g1g2`'s rank was **derived from `intor2.c` `ng[]={1,0,1,0,2,1,1,9}` → `ng[TENSOR]=ng[7]=9`** (D-16, not guessed). Added 12 `INT2E_*` RawApiId consts, 8 real `double*` 4-shell vendor wrappers (D-15), and 8 cart/sph bindgen allowlist symbols. `cargo build -p cintx-ops` regenerates `api_manifest.{rs,csv}`.
- **Kernels (Task 2):** Two new G-tensor position operators in `f12.rs` — `r0i_2e` (CINTx1i_2e / G2E_R0I: `f[n]=g[n+di]+ri·g[n]`) and `r0k_2e` (CINTx1k_2e / G2E_R0K) — plus four gout helpers (`gout_g1`, `gout_ig1`, `gout_gg1`, `gout_g1g2`) with the `c[]·s[]` combos transcribed verbatim from `intor4.c:1255` / `intor2.c:19,148,283`. A `Giao2eKind { G1, Ig1, Gg1, G1g2 }` dispatch + `launch_two_electron_giao2e::<F>` host-routes through `fill_g_tensor_2e` (FND-02 host Rys), applies the per-family `common_factor` scale, and materializes the real device output as interleaved `[re=0, im=value]` complex staging. Spinor reps return `UnsupportedApi` (D-11).
- **Parity (Task 3):** `giao_2e_parity.rs` (4 complex-aware tests, sizing the cintx buffer 2× and extracting the imaginary half on the non-zero-gauge non-square cross-center quartet `[3,2,3,2]`). **All 4 families byte-identical at atol=1e-12, cart+sph** (including `int2e_g1g2`). `oracle_covered` flipped `true` on the 8 cart/sph rows; manifest-audit `status: ok`.

## Task Commits

1. **Task 1: register 4 families** — `76040b8` (feat)
2. **Task 2: implement 2e GIAO kernels** — `fac1037` (feat)
3. **Task 3: parity + oracle_covered flip** — `340b61e` (test)

## Deviations from Plan

None — plan executed exactly as written.

The plan anticipated `int2e_g1g2` might need an `#[ignore]` deferral (like `int1e_a01gp` in 26-02), but the verbatim transcription of the rank-9 gout combos plus the correct `r0k_2e`/`r0i_2e` composition (g1=R0K(g0,i+1), g2=R0I(g0,i+0), g3=R0I(g1,i+0)) and the -0.25 common_factor produced byte-identity on the first parity run. No bugs, no auto-fixes, no architectural changes.

The plan text mentioned "#[cube] kernels generic over F". As with the established Phase-25 Hess2e analog, the GIAO 2e path is **host-routed** (host f64 Rys/gout via `fill_g_tensor_2e`, generic over `F` only at the staging boundary via `F::from_f64_lossy`). This follows the proven precedent exactly and is the correct interpretation — the multi-tensor R0I/R0K gout compositions are not exposed by the on-device comptime scalar kernel. This is a documentation nuance, not a deviation from intent.

## Known Stubs

None. All 4 families (incl `int2e_g1g2`) are fully registered (manifest + RawApiId + kernel + vendor wrapper + oracle), `oracle_covered=true`, and byte-identical. Spinor reps are registered for surface completeness and correctly return `UnsupportedApi` (D-11 — not a stub, an intentional fail-closed boundary; Phase-30 owns spin-carrying families).

## Threat Flags

None. No new trust boundaries beyond the existing `caller → eval_raw` numeric input path. The threat-register mitigations are honored:
- **T-26-05 (g1g2 component_rank truncation):** rank derived from `intor2.c` `ng[TENSOR]=9`, not guessed; the vendor byte-identity gate on the non-square cross-center quartet would have caught any truncation or transpose — it passed at atol=1e-12.
- **T-26-06 (spin-family scope leak):** only the 4 spin-free families are registered as parity targets; no `int2e_giao_sa10*` / `int2e_g1spsp2` (ng[POS_E1]=4 spin-carrying) symbol leaked. Deferral to Phase 30 preserved.

## Self-Check: PASSED

- Created `crates/cintx-oracle/tests/giao_2e_parity.rs` — FOUND
- Created `.planning/phases/26-group-5-spin-free-giao-nmr-integrals-complex/26-03-SUMMARY.md` — FOUND
- All task commits (`76040b8`, `fac1037`, `340b61e`) — FOUND
- Kernel artifacts: `Giao2eKind`, `r0i_2e`/`gout_g1g2`, `INT2E_G1G2_SPH`, `vendor_int2e_g1g2_cart` — FOUND
- All verify gates exit 0: `cargo build --workspace` (Task 1), `cargo build/test -p cintx-cubecl --features cpu` (Task 2), `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu giao_2e_parity` 4/4 byte-identical (Task 3), `manifest-audit status: ok`

---
*Phase: 26-group-5-spin-free-giao-nmr-integrals-complex*
*Completed: 2026-05-31*
