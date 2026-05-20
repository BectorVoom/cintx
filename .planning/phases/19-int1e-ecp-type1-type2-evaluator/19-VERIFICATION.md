---
phase: 19-int1e-ecp-type1-type2-evaluator
verified: 2026-05-20T13:07:04Z
status: human_needed
score: 5/5 must-haves verified (substance); 2 human-decision items
overrides_applied: 0
re_verification:
  previous_status: none
  previous_score: none
human_verification:
  - test: "Decide whether ECP-04's literal 'through both cintx-compat::raw::eval_raw AND SessionRequest::evaluate' clause is satisfied by transitive (shared-kernel) coverage, or requires a dedicated eval_raw byte-identity test."
    expected: "Either accept transitive coverage (both paths build an ExecutionPlan and call CubeClExecutor::execute → launch_ecp; the same K-Taylor kernel and math; SessionRequest path is byte-identity-verified at atol=1e-12) — OR require a sibling test that drives eval_raw with an INT1E_ECP_* RawApiId over Cu/LANL2DZ and asserts atol=1e-12 vs PySCF, mirroring the SessionRequest harness."
    why_human: "Cannot programmatically decide whether the project treats 'shared kernel + one verified entry point' as satisfying a two-named-path success criterion. The eval_raw dispatch arm + AS_NECPBAS guard exist and route to the identical launch_ecp; only SessionRequest::evaluate is directly asserted at byte-identity. This is a coverage-completeness question, not a correctness defect."
  - test: "Decide whether the 19-REVIEW CLAUDE.md-contract hardening gaps (CR-01 panic-on-bad-atom-index, HI-01/HI-02 silent partial-write on undersized scalar staging buffer) block phase closure or are accepted as advisory follow-up."
    expected: "Inspect launch_ecp at crates/cintx-cubecl/src/kernels/ecp.rs:1370-1371 / :1413 (atoms[idx] index panic on malformed BasisSet, public path) and :1468-1481 (scalar Spheric/Cart paths silently truncate into an undersized staging buffer and return Ok — contradicting CLAUDE.md 'no best-effort partial writes on failure'; the gradient path at :1380-1396 correctly fails closed). Decide: gate the phase on hardening these to typed cintxRsError before proceeding, OR accept as advisory robustness work since happy-path byte-identity is proven and these only trigger on adversarial/malformed input."
    why_human: "These are CLAUDE.md hard-constraint deviations (typed-error-not-panic; OOM-safe stop / no partial writes) on a public library path, but they do NOT affect the delivered symbols' byte-identity. Whether a contract deviation on malformed input blocks a correctness-focused phase is a project-policy call, not a programmatic one. Orchestrator flagged these as advisory; surfacing for explicit human sign-off."
---

# Phase 19: `int1e_ecp_*` Type-1/Type-2 Evaluator — Verification Report

**Phase Goal:** cintx implements Type-1 (Coulomb-like) and Type-2 (spin-orbit-like) ECP projectors and exposes them through `SessionRequest` alongside ordinary one-electron operators. Symbols delivered: `int1e_ecp_sph`, `int1e_ecp_cart`, and gradient variants `int1e_ecp_ipnuc_sph` / `int1e_ecp_ipnuc_cart`. Cu/LANL2DZ in the oracle corpus provides a byte-identity gate (atol=1e-12) through both `cintx-compat::raw::eval_raw` and `SessionRequest::evaluate`. Secondary cross-check against libecpint is a non-blocking oracle.

**Verified:** 2026-05-20T13:07:04Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Requirement-source note

REQUIREMENTS.md (`.planning/REQUIREMENTS.md`) covers only the v1.2 milestone (Phases 11-15) and contains **no ECP-01..05 entries**. Per the ROADMAP Phase 19 section, ECP-01..05 are "derived from success criteria 1-5 during `/gsd:spec-phase` or `/gsd:plan-phase`" — i.e. the requirement contract lives in the ROADMAP Success Criteria and the PLAN frontmatter, NOT in REQUIREMENTS.md. This is the established v1.3 pattern (Phase 17/18 RVAL-* are likewise absent). All five IDs are claimed across the four executed plans (19-05: ECP-01/02/04; 19-06: ECP-01/02/03/04; 19-07: ECP-01/02/04/05; 19-08: ECP-04) — **no orphaned requirement IDs**. The ROADMAP Success Criteria are used as the authoritative truths below.

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria = ECP-01..05)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 (ECP-01) | Type-1 (local, Coulomb-like) ECP projector evaluation implemented and registered in operator catalog | ✓ VERIFIED | `ecp_type1_cart` in `kernels/ecp.rs` runs the level-adaptive Gauss-Chebyshev convergence loop (lines 618-678) calling `ecprad_part_host` (629) + `type1_rad_part_host` (654) — a real PySCF `ECPtype1_cart` port, not a stub. `OperatorId::INT1E_ECP_*` registered (resolver.rs:521-528); manifest rows 28-31 present. |
| 2 (ECP-02) | Type-2 (semi-local, spin-orbit-like) ECP projector with sph-harmonic angular projectors + Bessel-modulated radial integrals implemented and registered | ✓ VERIFIED | `ecp_type2_cart` calls `ecprad_part_host` (822) + `type2_facs_rad_host` (838/853) + `ecpsph_ine_opt_host` (table-interpolation modified-spherical-Bessel, `ecp_k_taylor.rs:183`). Registered same as ECP-01. |
| 3 (ECP-03) | `SessionRequest::evaluate` dispatches `int1e_ecp_{sph,cart}` through the same safe-API surface as ordinary 1e operators — no parallel ECP API | ✓ VERIFIED | `cintx-rs/src/api.rs::evaluate` builds an `ExecutionPlan` and calls `CubeClExecutor::execute` (api.rs:169/252) — the identical path used by all 1e operators. ECP-only addition is the `is_ecp()` preflight returning `FacadeError::MissingEcpBasis` (api.rs:83). No `int1e_ecp` evaluate variant exists. Parity tests drive `SessionRequest::evaluate` and pass. |
| 4 (ECP-04) | Cu/LANL2DZ passes byte-identity vs libcint at atol=1e-12 through both `eval_raw` and `SessionRequest::evaluate`; secondary libecpint cross-check added non-blocking | ⚠️ VERIFIED (with coverage caveat) | **Re-ran independently:** `safe_api_ecp_parity` → 5/5 pass (4 byte-identity symbol tests + coverage invariant) at `ATOL=1e-12, RTOL=0.0` (constants lines 58-59) vs PySCF `vendor_ECPscalar_{cart,sph}` / `_ipnuc_{cart,sph}` over full Cu/LANL2DZ Cartesian product. Secondary libecpint oracle present, double-gated (`#[cfg(has_libecpint_oracle)]` + `#[ignore]`), informational atol≈1e-9, non-blocking. **Caveat:** only `SessionRequest::evaluate` is directly asserted; `eval_raw` shares the identical `launch_ecp` kernel via `CubeClExecutor::execute` (raw.rs:585/654) but has no dedicated byte-identity test → see human-decision item 1. |
| 5 (ECP-05) | Gradient variants land this phase (decision recorded) | ✓ VERIFIED | D-10 (CONTEXT line 158) records gradients IN scope. `int1e_ecp_ipnuc_{cart,sph}` implemented via `deriv1_cart_pair` / `compute_type1_pair_grad` / `compute_type2_pair_grad` (ecp.rs:1137-1296, port of `nr_ecp_deriv.c::_deriv1_cart`), F-order `[axis, ao_j, ao_i]` (D-11). Two ipnuc parity tests pass at atol=1e-12 over the product × 3 components. |

**Score:** 5/5 truths verified in substance (1 with a coverage-completeness caveat routed to human decision).

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/cintx-cubecl/src/math/ecp_k_taylor_data.rs` | `include_bytes!` blob accessors (D-14) | ✓ VERIFIED | 54 lines; `AlignedBytes<{N*8}>` + `include_bytes!` + `bytemuck::cast_slice`, mirrors `roots_xw_data.rs`. `sph_ine_tab()` (9600 f64), `sph_ine_tab_order7()` (25600 f64). |
| `crates/cintx-cubecl/src/math/ecp_k_taylor_in.bin` / `_order7.bin` | LE-f64 K-Taylor tables | ✓ VERIFIED | 76800 + 204800 bytes = exactly 9600 + 25600 f64. Drift-gate confirms byte-match to vendored `nr_ecp.c`. |
| `crates/cintx-cubecl/src/math/ecp_k_taylor.rs` | Host ports `ecpsph_ine_opt_host`, `ecprad_part_host`, `type1_rad_part_host`, `type2_facs_rad_host` (min 250 lines) | ✓ VERIFIED | 763 lines; all four `pub fn` present (183/285/372/438). Numerical fidelity confirmed by review (order>7 recurrence, SIM_ZERO break, radial_power arms all byte-faithful to C). |
| `crates/cintx-cubecl/src/kernels/ecp.rs` | K-Taylor-based scalar + gradient `launch_ecp` | ✓ VERIFIED | 1696 lines; scalar paths call K-Taylor (no direct-quadrature compute remains), gradient `_deriv1_cart` port present. Wired into `kernels/mod.rs`. |
| `crates/cintx-oracle/tests/safe_api_ecp_parity.rs` | 4 un-ignored parity tests at atol=1e-12 + coverage invariant | ✓ VERIFIED | 588 lines; no `#[ignore]` on any test; 5/5 pass on re-run. |
| `xtask/src/gen_ecp_tables.rs` | Table extractor + `--check` drift gate (min 100 lines) | ✓ VERIFIED | 227 lines; `--check` exits 0 (re-run confirmed byte-match). |
| `.github/workflows/compat-governance-pr.yml` | `gen-ecp-tables --check` CI step | ✓ VERIFIED | Line 75 in `manifest_drift_gate` job alongside `manifest-audit --check-lock`. |
| `crates/cintx-oracle/src/libecpint_ffi.rs` | extern "C" libecpint shim | ✓ VERIFIED | 5335 bytes; `#[cfg(has_libecpint_oracle)]`-gated. |
| `crates/cintx-oracle/tests/ecp_libecpint_crosscheck_parity.rs` | Env-gated #[ignore] cross-check | ✓ VERIFIED | 16780 bytes; double-gated; informational `CROSSCHECK_ATOL=1e-9`. |
| `.planning/notes/ecp-libecpint-crosscheck.md` | Provenance + tolerance note | ✓ VERIFIED | 6193 bytes present. |
| `crates/cintx-ops/src/generated/api_manifest.csv` | 4 ECP rows `oracle_covered=true` | ✓ VERIFIED | Rows 28-31 all `true`; ipnuc rows carry `component_rank=3`. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `ecpsph_ine_opt_host` | `sph_ine_tab_order7` blob | table interpolation | ✓ WIRED | `ecp_k_taylor.rs` calls `ecp_k_taylor_data::sph_ine_tab_order7()`. |
| `gen_ecp_tables.rs` | `vendor/pyscf-nr-ecp/src/nr_ecp.c` | literal-array parse | ✓ WIRED | `NR_ECP_C_REL` constant + `--check` re-extracts and byte-compares. |
| CI `manifest_drift_gate` | `gen-ecp-tables --check` | cargo run step | ✓ WIRED | compat-governance-pr.yml:75. |
| `compute_type1_pair` | `type1_rad_part_host` | Type-1 radial assembly | ✓ WIRED | ecp.rs:654. |
| `compute_type2_pair` | `type2_facs_rad_host` / `ecpsph_ine_opt_host` | Type-2 radial-factor + angular splice | ✓ WIRED | ecp.rs:838/853. |
| gradient branch | `nr_ecp_deriv.c::_deriv1_cart` | ∂/∂A_C derivative on K-Taylor foundation | ✓ WIRED | ecp.rs:1137-1296. |
| parity tests | `vendor_ECPscalar_{cart,sph}` / `_ipnuc_*` | byte-identity at atol=1e-12 | ✓ WIRED | tests pass on re-run. |
| `eval_raw` (ECP arm) | `launch_ecp` | `CubeClExecutor::execute` | ⚠️ WIRED, untested-direct | Arm + `AS_NECPBAS` guard exist (raw.rs); routes to identical kernel as the safe API, but no dedicated eval_raw byte-identity test (see human item 1). |
| libecpint cross-check | `libecpint_ffi.rs` | informational at atol≈1e-9 | ✓ WIRED | `CINTX_LIBECPINT_ORACLE` gate. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `safe_api_ecp_parity` safe matrix | `safe_matrix` | `SessionRequest::evaluate` → `CubeClExecutor::execute` → `launch_ecp` (real K-Taylor math) | Yes — compared element-wise vs PySCF, 0 mismatches at 1e-12 | ✓ FLOWING |
| `.bin` blobs | embedded f64 tables | `include_bytes!` of `gen-ecp-tables` output, drift-locked to `nr_ecp.c` | Yes — exact PySCF literal values, drift gate green | ✓ FLOWING |
| vendor matrix | `vendor_matrix` | PySCF `ECPscalar_*` FFI over combined AO+ECP bas | Yes — real libcint/PySCF reference | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| ECP byte-identity suite | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --locked --test safe_api_ecp_parity` | 5 passed; 0 failed; 0 ignored | ✓ PASS |
| K-Taylor blob drift gate | `cargo run -p xtask --locked -- gen-ecp-tables --check` | exit 0; both blobs match vendored source | ✓ PASS |
| Manifest audit | `cargo run -p xtask --locked -- manifest-audit` | exit 0 | ✓ PASS |
| Manifest lock check | `cargo run -p xtask --locked -- manifest-audit --check-lock` | exit 0 | ✓ PASS |
| Full workspace (regression) | `cargo test --workspace --features cpu --locked` | 0 failures across all crates | ✓ PASS |

### Probe Execution

No conventional `scripts/*/tests/probe-*.sh` probes declared for this phase; the byte-identity gate is the cargo-test suite above, which was executed directly. N/A.

### Requirements Coverage

| Requirement | Source Plan(s) | Description (ROADMAP SC) | Status | Evidence |
|-------------|----------------|--------------------------|--------|----------|
| ECP-01 | 19-05, 19-06, 19-07 | Type-1 projector implemented + registered | ✓ SATISFIED | Truth 1 |
| ECP-02 | 19-05, 19-06, 19-07 | Type-2 projector implemented + registered | ✓ SATISFIED | Truth 2 |
| ECP-03 | 19-06 | SessionRequest dispatch, no parallel API | ✓ SATISFIED | Truth 3 |
| ECP-04 | 19-05, 19-06, 19-07, 19-08 | Cu/LANL2DZ byte-identity (both paths) + non-blocking libecpint | ⚠️ SATISFIED (caveat) | Truth 4 — SessionRequest path proven; eval_raw transitive (human item 1) |
| ECP-05 | 19-07 | Gradient variants in scope + delivered | ✓ SATISFIED | Truth 5 |

No ORPHANED requirements (all five claimed by ≥1 plan; none mapped to Phase 19 in REQUIREMENTS.md that a plan failed to claim).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `kernels/ecp.rs` | 1370-1371, 1413 | `atoms[idx]` index without bounds check → panic on malformed BasisSet (CR-01) | ⚠️ Warning | Public-path panic vs CLAUDE.md typed-error mandate; adversarial/malformed input only. Not a parity defect. |
| `kernels/ecp.rs` | 1468-1481 | Scalar Spheric/Cart paths silently truncate into undersized `staging` and return Ok (HI-01/HI-02) | ⚠️ Warning | Violates CLAUDE.md "no best-effort partial writes on failure". Gradient path fails closed correctly; scalar does not. Triggered only by caller under-sizing the buffer. |
| `vendor_ffi.rs` | 1378-1382, 1418-1422 | FFI guard is `debug_assert!` (release no-op), checks `len%3==0` not exact size (ME-02) | ℹ️ Info | Oracle/test harness code only; FFI buffer guard ineffective in release. |
| `xtask/gen_ecp_tables.rs` | 70-109 | Positional (non-balanced) brace matching in extractor (ME-03) | ℹ️ Info | Works for current source shape; could mis-parse a reshaped-but-valid future source. Drift gate substance verified sound. |
| No `TBD`/`FIXME`/`XXX` | — | Debt-marker gate | ✓ Clean | Scanned all 7 ECP-modified files: 0 unreferenced debt markers → no BLOCKER from the debt-marker gate. |

**No stubs:** the scalar/gradient compute paths are full PySCF ports (cross-traced by the deep code review and re-confirmed here); the only `UnsupportedApi` returns are for genuinely unsupported operators/projector-l-overflow, not ECP placeholders. The 19-04 direct-quadrature stub was fully replaced.

### Human Verification Required

**1. ECP-04 dual-path wording vs transitive coverage.**
The success criterion says byte-identity "through **both** `cintx-compat::raw::eval_raw` **and** `SessionRequest::evaluate`". The parity suite directly asserts only `SessionRequest::evaluate`. Both entry points construct an `ExecutionPlan` and call `CubeClExecutor::execute → launch_ecp` (the identical K-Taylor kernel and math; eval_raw at raw.rs:585/654, safe API at api.rs:169/252), so eval_raw byte-identity is structurally transitive and the eval_raw dispatch arm + `AS_NECPBAS` guard are wired and unit-gated. **Decide:** accept transitive coverage, or require a dedicated `eval_raw(INT1E_ECP_*)` byte-identity test over Cu/LANL2DZ at atol=1e-12.

**2. CLAUDE.md-contract hardening gaps (CR-01 / HI-01 / HI-02).**
`launch_ecp` panics on an out-of-range `atom_index` (public path) and silently partial-writes into an undersized scalar staging buffer (returning Ok) — both contradict CLAUDE.md hard constraints (typed-error-not-panic; OOM-safe stop / no partial writes). They affect adversarial/malformed input only; the happy-path byte-identity is fully proven and the gradient path already fails closed. **Decide:** gate phase closure on hardening these to typed `cintxRsError`, or accept as advisory follow-up (orchestrator flagged them advisory). If accepted, an `overrides:` entry would convert them to documented-deviation PASS on re-verify.

### Gaps Summary

There are **no correctness/parity gaps**: the four delivered ECP symbols (`int1e_ecp_{cart,sph}`, `int1e_ecp_ipnuc_{cart,sph}`) achieve byte-identity vs vendored PySCF `nr_ecp`/`nr_ecp_deriv` at `atol=1e-12, rtol=0.0` over the full Cu/LANL2DZ Cartesian product (independently re-run: 5/5), all four manifest rows are `oracle_covered=true`, the K-Taylor blobs are drift-locked to the vendored C with an enforced CI gate, the full workspace test suite passes with zero failures, and the non-blocking libecpint secondary oracle is wired. All eight executed plans' must_haves are substantively present in the codebase, and all five requirement IDs are accounted for with no orphans. The host-only kernel is a sanctioned CLAUDE.md deviation (D-16), documented in the kernel rustdoc.

Two items require a human decision (not closure plans): (1) whether ECP-04's literal "both paths" wording is satisfied by transitive shared-kernel coverage or needs a dedicated eval_raw byte-identity test; and (2) whether the advisory CLAUDE.md-contract hardening gaps (panic + scalar partial-write on malformed/under-sized input) block phase closure. Per the verification decision tree, the presence of these human-decision items makes the status `human_needed` rather than `passed`, even though every parity gate passes.

---

_Verified: 2026-05-20T13:07:04Z_
_Verifier: Claude (gsd-verifier)_
