# Phase 30 — Wave 1 re-plan proposal: split by engine class

**Status:** proposal (seed for `/gsd:plan-phase 30` re-plan). Created 2026-06-01 after the 30-01 executor returned a decision checkpoint.

## Why this exists

The original `30-01-PLAN.md` framed Wave 1 as a lightweight "TRANSCRIBE+REGISTER onto the proven fold." Ground-truth verification against `libcint-master/src/autocode/intor3.c` + `cint1e.c` + `g1e.c` (done by the 30-01 executor) contradicts that framing:

- **Only 1 of 9 families (`int1e_cg_sa10sp`, rank-3 overlap) was actually proven in Plan 00.**
- `int1e_giao_sa10sp` is that same kernel at `common_orig=[0,0,0]` (G1E_R_I) — also done.
- The remaining **7 families require ~6 net-new CubeCL device kernels** across two new engine classes (Rys+gauge, 8-G-tensor London) in both overlap and Rys forms, with rank-9 36-component gouts.
- The original gate is **all-or-nothing** (all 9 byte-identical at atol=1e-12), verifiable only at the vendor gate — high risk of committing subtly-wrong device math in one pass.

**User decision (2026-06-01):** re-plan Wave 1 into engine-class sub-waves, each with its own vendor gate before the next. Matches `feedback_disproven_spike_prefer_replan`.

## Already landed (reuse as-is, do NOT regenerate)

- **30-00** — COMPLETE. Gauge `x1i`-with-origin fold (`CINTx1i_1e` position recurrence `f[i]=g[i+1]+origin*g[i]`, NOT a cross-product) + `int1e_cg_sa10sp` rank-3 gout variant in `sigma_p.rs`; `build_gauge_kappa_spinor_fixture`; `giao_sigma_1e_parity` micro-test (byte-identity vs `int1e_cg_sa10sp_spinor` + cg→giao-at-origin=0 collapse).
- **30-01 scaffolding** — commit `3b68ff1`. 9 spinor manifest rows (all `oracle_covered=false`, ranks: sa01=9, sp/nucsp=3); bindgen `allowlist_function` extended (build.rs) + 7 vendor shims (vendor_ffi.rs); `launch_int1e_giao_sa10sp_spinor_pair` (cg kernel @origin=0). Compiles under `CINTX_ORACLE_BUILD_VENDOR=1 ... --features cpu`. `OperatorId` invariant verified (no hardcoded const re-pointed). **2 of 9 families covered.**

## Proposed sub-wave breakdown

Each sub-wave: implement kernel(s) → flip `oracle_covered=true` per family only as its vendor gate goes green. GIAO-03 still closes at end of Wave 2 (30-02), not here.

| Sub-wave | Families | Engine / new math |
| --- | --- | --- |
| (done) | `cg_sa10sp`, `giao_sa10sp` | rank-3 overlap gauge fold (30-00 + 3b68ff1) |
| **30-01a** | `spgsp` | NEW 8-G-tensor overlap. London `c=ri−rj` post-mult, `D_J`→i_l+2, `R0I`(origin=ri), `D_I` back-compose, 27→12 gout |
| **30-01b** | `cg_sa10nucsp`, `giao_sa10nucsp` | NEW Rys+gauge. `x1i`-with-origin inside the Rys root loop, 12-comp gout (int1e_type 2, nuclear) |
| **30-01c** | `cg_sa10sa01`, `giao_sa10sa01` | NEW Rys+gauge, rank 9. `g1 = ∇_j(g0)+∇_i(g0)`, `x1i`, 36-comp gout, `c2s_si_1e` (real) (int1e_type 1, rinv) |
| **30-01d** | `spgnucsp`, `spgsa01` | NEW spg-Rys/London. spgsp+Rys (12-comp) and spgsa01+Rys (36-comp, rank 9) |

## Hard constraints for the re-plan (carry-forward)

- Spinor parity tests MUST use a NON-SQUARE block (e.g. p×d) — square blocks are transpose-symmetric and hide the KET-major/BRA-major orientation bug (`cart_to_spinor_sf_2d` reads BRA-major; device cart blocks are KET-major → transpose first).
- Vendor parity is double-gated: `--features cpu` AND `CINTX_ORACLE_BUILD_VENDOR=1`, else it silently skips.
- Bindgen allowlist already covers the 9 spinor symbols (3b68ff1) — no further build.rs change needed for these families.
- Do NOT clamp Rys nroots — fail-closed.
- After any further manifest edits, re-grep for hardcoded `OperatorId::new(<int>)` / `_OPERATOR_ID: u32 = N` and re-point by symbol name.
- libcint stub caveat: if any family dispatches to a libcint stub (no real byte-identity reference), STOP and checkpoint rather than inventing a reference.
