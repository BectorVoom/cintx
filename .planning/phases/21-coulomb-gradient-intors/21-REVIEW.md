---
phase: 21-coulomb-gradient-intors
reviewed: 2026-05-26T12:58:54Z
depth: standard
files_reviewed: 19
files_reviewed_list:
  - crates/cintx-compat/src/legacy.rs
  - crates/cintx-cubecl/src/kernels/center_3c2e.rs
  - crates/cintx-cubecl/src/kernels/ecp.rs
  - crates/cintx-cubecl/src/kernels/f12.rs
  - crates/cintx-cubecl/src/kernels/mod.rs
  - crates/cintx-cubecl/src/kernels/one_electron.rs
  - crates/cintx-cubecl/src/kernels/two_electron.rs
  - crates/cintx-ops/src/lib.rs
  - crates/cintx-ops/src/resolver.rs
  - crates/cintx-oracle/build.rs
  - crates/cintx-oracle/src/compare.rs
  - crates/cintx-oracle/src/vendor_ffi.rs
  - crates/cintx-oracle/tests/center_3c2e_parity.rs
  - crates/cintx-oracle/tests/ecp_iprinv_parity.rs
  - crates/cintx-oracle/tests/one_electron_nuc_grad_parity.rs
  - crates/cintx-oracle/tests/oracle_gate_closure.rs
  - crates/cintx-oracle/tests/safe_api_arity3_parity.rs
  - crates/cintx-oracle/tests/two_electron_ip1_parity.rs
  - xtask/src/oracle_covered_update.rs
findings:
  critical: 2
  warning: 6
  info: 5
  total: 13
status: issues_found
---

# Phase 21: Code Review Report

**Reviewed:** 2026-05-26T12:58:54Z
**Depth:** standard
**Files Reviewed:** 19
**Status:** issues_found

## Summary

Phase 21 ships the Coulomb/nuclear analytical-gradient integrals (int1e_ipovlp/ipkin/ipnuc/iprinv, int2e_ip1, int3c2e_ip1, ECPscalar_iprinv) targeting libcint 6.1.3 byte-identity at atol=1e-12. The kernel math (G-tensor index arithmetic, the verbatim `gout_ip1`/`nabla1i_2e` reuse for the plain-Coulomb gradients, the 3c2e Pitfall-4 kl mapping, the nuclear-gradient ordered reduction) is internally consistent and well-documented, and the spinor `UnsupportedApi` guards (R5/D-03) and `nroots > 5` fail-closed guards (R2) are present on every gradient launcher path.

However, two BLOCKER-level defects undermine the oracle verification claims this phase rests on. First, the legacy-wrapper parity gate in `compare.rs` compares the NEW 3-component `int3c2e_ip1_sph` derivative against the SCALAR `vendor_int3c2e_sph` with a 1-component output buffer — the gate either errors with `BufferTooSmall` or, worse, is structurally incapable of validating the gradient it claims to. Second, `int3c2e_ip1` is wired into `eval_legacy_symbol`'s ALL_CINT fallback arm via a stale comment claiming it's still an "upstream proxy" — but the actual fallback list does NOT include it, so the matrix-driven parity for that symbol routes through `cint3c2e_ip1_sph` legacy which is fine, while the hand-rolled `verify_legacy_wrapper_parity` block is wrong. These are exactly the kind of "the gate is green but it isn't testing what it says" defects that defeat the project's verification constraint.

The remaining findings are robustness and maintainability concerns: an unbounded `Box::leak` on the family-tolerance fast path, several silent `if dst < staging.len()` truncations in the scatter loops that can mask sizing bugs instead of failing closed, and a missing spinor arm in the oracle harness's `eval_legacy_symbol` for the new ip families.

## Critical Issues

### CR-01: Legacy-wrapper parity gate compares the 3-component `int3c2e_ip1` derivative against the scalar `vendor_int3c2e_sph` with a 1-component buffer

**File:** `crates/cintx-oracle/src/compare.rs:937-947`

The `verify_legacy_wrapper_parity` block for `int3c2e_ip1_sph` allocates a scalar-sized output (`size_3 = ni*nj*nk`), evaluates the cintx `int3c2e_ip1_sph` legacy wrapper into it, and compares against `vendor_int3c2e_sph` (the PLAIN, non-gradient 3c2e):

```rust
// ─── int3c2e_ip1_sph ──────────────────────────────────────────────────────
{
    let mut cintx_out = vec![0.0_f64; size_3];   // size_3 = ni*nj*nk  (scalar)
    unsafe {
        eval_legacy_symbol("int3c2e_ip1_sph", &mut cintx_out, shls3, atm, bas, env)?;
    }
    let mut vendor_out = vec![0.0_f64; size_3];
    let shls3_arr = [shls3[0], shls3[1], shls3[2]];
    vendor_ffi::vendor_int3c2e_sph(&mut vendor_out, &shls3_arr, atm, natm, bas, nbas, env);
    mismatches += compare_buffers("cint3c2e_ip1_sph", &cintx_out, &vendor_out);
}
```

This is inconsistent with the rest of the phase. As of 21-06, `int3c2e_ip1` is the REAL `∇_A` first-center derivative with `component_count = 3` (confirmed by `center_3c2e.rs::launch_center_3c2e_ip1`, the `safe_api_arity3_parity.rs` tests sizing `3 * ni*nj*nk`, and the `center_3c2e_parity.rs` vendor test flipping to `vendor_int3c2e_ip1_sph`). Two failures result:

1. `eval_legacy_symbol("int3c2e_ip1_sph", ...)` routes to `legacy::cint3c2e_ip1_sph` → `eval_raw(INT3C2E_IP1_SPH, ...)`. `CompatDims::ensure_output_len` (`layout.rs:113`) computes `required = 3 * ni*nj*nk` and returns `BufferTooSmall` because `provided = ni*nj*nk < required`. The `?` propagates the error, so the ENTIRE `verify_helper_surface_coverage`/legacy oracle gate bails whenever `has_vendor_libcint` is active — the gate cannot run.

2. Even if the buffer were sized `3 * ni*nj*nk`, the reference `vendor_int3c2e_sph` is the scalar non-gradient integral, so a derivative-vs-scalar comparison would be meaningless (the cart sibling at lines 1052-1062 has the identical defect: it compares cintx `int3c2e_ip1_cart` against `vendor_int3c2e_ip1_cart` — which is correct — so the sph block is the odd one out and is plainly a copy-paste/flip miss).

**Fix:** Size the buffer to the 3-component output and compare against the real gradient vendor symbol:

```rust
// ─── int3c2e_ip1_sph ──────────────────────────────────────────────────────
{
    let mut cintx_out = vec![0.0_f64; 3 * size_3];
    unsafe {
        eval_legacy_symbol("int3c2e_ip1_sph", &mut cintx_out, shls3, atm, bas, env)?;
    }
    let mut vendor_out = vec![0.0_f64; 3 * size_3];
    let shls3_arr = [shls3[0], shls3[1], shls3[2]];
    vendor_ffi::vendor_int3c2e_ip1_sph(&mut vendor_out, &shls3_arr, atm, natm, bas, nbas, env);
    mismatches += compare_buffers("cint3c2e_ip1_sph", &cintx_out, &vendor_out);
}
```

(Confirm the cart block at 1052-1062 is already correct — it appears to use `vendor_int3c2e_ip1_cart` and `size_3_c`, but verify the buffer is `3 * size_3_c`, not `size_3_c`; the read shows `size_3_c` only, so the cart block has the same 1-component-buffer sizing bug and needs the `3 *` prefix too.)

### CR-02: `eval_legacy_symbol` ALL_CINT fallback comment claims `int3c2e_ip1` is an "upstream proxy" but the symbol is force-mapped to a dedicated legacy wrapper, masking a divergent reference

**File:** `crates/cintx-oracle/src/compare.rs:437-455` (and the matrix-driven path at `crates/cintx-oracle/src/compare.rs:1271-1314`)

The matrix-driven parity loop calls `eval_legacy_symbol(&fixture.symbol, ...)` to produce the "upstream reference" against which cintx's own `eval_raw` output is diffed (`raw_vs_upstream`, `compare.rs:1316`). For `int3c2e_ip1_*` and `int2e_ip1_*` this means BOTH sides of the comparison are cintx's own kernel (the legacy wrapper just calls `eval_raw` on the same `RawApiId`). The `diff_summary` "raw vs upstream" check is therefore self-comparing — it can never catch a kernel that is wrong-but-deterministic. The only genuine oracle for the new gradient symbols is the `#[cfg(has_vendor_libcint)]` vendor block in `verify_legacy_wrapper_parity` (CR-01, which is broken for sph) and the standalone `*_parity.rs` tests.

The hazard: the phase's report (`compare.rs:1515`) labels the upstream reference as "vendored upstream compatibility proxy through cintx_compat::legacy wrappers", but for the Phase-21 gradient families there is NO vendored upstream in this path — it is cintx-vs-cintx. A reader of the green parity artifact would reasonably (and wrongly) conclude that `int3c2e_ip1`/`int2e_ip1` were validated against libcint here. Combined with CR-01 disabling the only real sph vendor check, the net effect is that `int3c2e_ip1_sph` has NO working byte-identity gate inside the harness-driven matrix path.

**Fix:** Either (a) route the new gradient families' `eval_legacy_symbol` upstream proxy through the vendored `vendor_int3c2e_ip1_sph`/`vendor_int2e_ip1_sph` symbols when `has_vendor_libcint` is set (the standalone tests already prove these exist), or (b) downgrade the report's `upstream_reference` label to explicitly state "self-consistency (cintx legacy == cintx raw) for derivative families; real vendor parity lives in `*_ip1_parity.rs`", and ensure CR-01 is fixed so the vendor block is the authoritative gate. Do not ship a parity artifact whose label overstates what was compared.

## Warnings

### WR-01: Cart sibling of CR-01 also under-sizes its buffer

**File:** `crates/cintx-oracle/src/compare.rs:1052-1062`

The `int3c2e_ip1_cart` legacy-parity block allocates `cintx_out`/`vendor_out` as `size_3_c = ni_c*nj_c*nk_c` (1 component) but `int3c2e_ip1` now emits 3 components, so `eval_legacy_symbol("int3c2e_ip1_cart", ...)` will hit `BufferTooSmall` exactly as the sph block does. The reference `vendor_int3c2e_ip1_cart` is the correct (gradient) symbol, so only the sizing is wrong here.

**Fix:** Allocate `3 * size_3_c` for both buffers in this block.

### WR-02: Unbounded `Box::leak` on the tolerance fast path for unknown families

**File:** `crates/cintx-oracle/src/compare.rs:147` and `crates/cintx-oracle/src/compare.rs:216-219`

`tolerance_for_family` and `f32_tolerance_for_family` leak a fresh `&'static str` for any family string not in the match arms: `Box::leak(family.to_owned().into_boxed_str())`. The comment asserts "occurs at most once per unique family string", which is true only if the set of family strings is finite and deduplicated upstream. These functions are called per-fixture inside `build_profile_parity_report` (`compare.rs:1105`), and `family` comes from `OracleFixture::family`. If a future caller ever feeds a dynamically-constructed or attacker-influenced family string (or simply iterates many profiles), this leaks unboundedly. It is also a latent footgun if `tolerance_for_family` is ever moved into a long-running service path.

**Fix:** Replace the leak with a lookup that returns the original `&str` lifetime tied to the input, or store a `String` field in `FamilyTolerance` instead of `&'static str`. If the `'static` requirement is structural, intern through a `OnceLock<Mutex<HashSet<&'static str>>>` so each unique family leaks at most once process-wide regardless of call count.

### WR-03: Silent `if dst < staging.len()` truncation hides output-sizing bugs instead of failing closed

**File:** `crates/cintx-cubecl/src/kernels/two_electron.rs:783`, `crates/cintx-cubecl/src/kernels/two_electron.rs:824`, `crates/cintx-cubecl/src/kernels/center_3c2e.rs:497`, `crates/cintx-cubecl/src/kernels/center_3c2e.rs:531`, `crates/cintx-cubecl/src/kernels/one_electron.rs:1026`, `crates/cintx-cubecl/src/kernels/one_electron.rs:1050`

Every gradient scatter loop guards the write with `if dst < staging.len() { staging[dst] = ... }`. When `dst >= staging.len()` the value is silently dropped. This converts an under-sized staging buffer (a planner/manifest `component_rank` mismatch — exactly the class of bug in CR-01) into a silent partial write rather than a typed error. The project's design constraint is explicit: "Best-effort partial writes on allocation failure / Fallible allocation + typed failure + no partial writes" (CLAUDE.md "What Not to Use"). These guards violate that contract for the gradient path: a sizing regression produces a quietly truncated gradient (some components zero) that still passes the `any_nonzero` sentinels.

**Fix:** Before the scatter loop, assert/return `BufferTooSmall` if `staging.len() < <expected component-leading size>` (e.g. `3 * di*dj*dk*dl` for the 2e Spheric arm), then index `staging[dst]` unconditionally. The guard becomes a hard precondition, not a per-element silent drop.

### WR-04: `eval_legacy_symbol` has no spinor arm for the new ip families, so a spinor fixture would route to the source-only fallback and misreport

**File:** `crates/cintx-oracle/src/compare.rs:424-436`

The Phase-21 arm in `eval_legacy_symbol` handles only `*_cart` and `*_sph` for `int1e_ipovlp/ipkin/ipnuc/iprinv` and `int2e_ip1` (comment at 424-426 acknowledges "only cart/sph proxy here"). If a `*_ipovlp_spinor` (etc.) symbol ever reaches this function — e.g. if the spinor-gradient exclusion in `fixtures.rs` regresses — it falls through to the `other =>` branch (line 456), which calls `source_only_raw_api_for_symbol`. For a stable (non-source-only) spinor gradient symbol that returns `None`, producing `bail!("missing legacy wrapper mapping for ...")` — a confusing error that blames a "missing mapping" rather than the real cause (spinor gradients are `UnsupportedApi` by design, R5/D-03). The `raw_api_for_symbol` map DOES list the spinor variants (lines 333-345), so the two maps are inconsistent.

**Fix:** Add an explicit spinor arm in `eval_legacy_symbol` for the ip families that returns the same `UnsupportedApi` the kernel would, or a clear `bail!("spinor int1e gradient is UnsupportedApi by design (R5/D-03)")`, so the failure mode is self-explanatory if the fixture filter ever regresses.

### WR-05: `int3c2e_ip1_spinor` is mapped to a `RawApiId` and a legacy wrapper, but documented as excluded — the live mapping can be reached

**File:** `crates/cintx-oracle/src/compare.rs:326` and `crates/cintx-oracle/src/compare.rs:333-345`, `crates/cintx-compat/src/legacy.rs:236`/`245`/`254`/`263`/`272`/`283`

`raw_api_for_symbol` maps `int3c2e_ip1_spinor`, `int1e_ipovlp_spinor`, `int2e_ip1_spinor`, etc. to real `RawApiId` values, and `legacy.rs` defines `cint3c2e_ip1` (spinor), `cint1e_ipovlp` (spinor), `cint2e_ip1` (spinor) wrappers. The comment at `compare.rs:328-330` states spinor gradients are "excluded from the parity matrix in fixtures.rs per R5/D-03, but the map stays complete". The exclusion is enforced ONLY in `fixtures.rs` (out of scope of this review) and in `build_profile_parity_report`'s `component_count == 3 && representation == "spinor"` skip (`compare.rs:1112`). If either filter regresses, the spinor gradient symbol resolves to a live dispatch that the kernels reject at runtime with `UnsupportedApi` — but the matrix loop would record it as a `legacy_eval` mismatch, not a designed skip. The defense-in-depth here is one layer thinner than the comments imply.

**Fix:** Document the single point of enforcement explicitly and add a debug assertion in `build_profile_parity_report` that any fixture reaching the `eval_legacy_symbol` call with a 3-component spinor representation is a bug (the skip at line 1112 should have caught it). This keeps the "map stays complete" convenience without a silent reclassification of a designed-unsupported path into a mismatch.

### WR-06: `iprinv` coordinate-match selection can select multiple atoms or the wrong atom when two atoms share a coordinate within 1e-10

**File:** `crates/cintx-cubecl/src/kernels/ecp.rs:612-624`

`select_iprinv_slots` matches the rinv origin to an atom by Euclidean distance `< IPRINV_ORIGIN_MATCH_TOL (1e-10)`. The vendor reference selects by INTEGER atom index (`env[AS_RINV_ORIG_ATOM]`). For any geometry where two ECP-bearing atoms are within 1e-10 bohr (degenerate/duplicated centers, or a basis-set ghost atom co-located with a real atom), the coordinate match would select BOTH atoms' slots, diverging from the vendor's single-index selection and breaking byte-identity. The doc comment frames the tolerance as absorbing "float round-trip noise", but it silently changes the selection SEMANTICS from index-based to coordinate-based. For the shipped single-ECP-atom Cu/LANL2DZ fixture this is untested-degenerate (only one atom), so the parity test (`ecp_iprinv_parity.rs`) cannot catch it.

**Fix:** This is a correctness risk for multi-center ECP gradients, not just a style issue. Either (a) plumb the integer atom index through the safe API to match the vendor's selection exactly, or (b) document the precondition that ECP-bearing atom coordinates must be distinct by more than `IPRINV_ORIGIN_MATCH_TOL` and return a typed error (not a multi-atom selection) when two ECP atoms match the same origin. At minimum add a unit test with two ECP atoms at distinct-but-close coordinates to pin the selection behavior.

## Info

### IN-01: Magic threshold `1e-15` for the `sp_scale` skip is duplicated across kernels

**File:** `crates/cintx-cubecl/src/kernels/one_electron.rs:1001`, `crates/cintx-cubecl/src/kernels/one_electron.rs:1165`

The `if (sp_scale - 1.0).abs() > 1e-15` guard that decides whether to apply the `common_fac_sp` scale is repeated with a bare magic number. For `s`/`p` shells `sp_scale != 1.0` so the branch is taken; for `l>=2` it equals exactly 1.0. The `1e-15` epsilon is arbitrary (the values are exact `1.0` for `l>=2` by construction). Extract to a named const or drop the epsilon guard and compare against `1.0` exactly, since `common_fac_sp` returns a literal `1.0`.

### IN-02: `let _ = l;` no-op loop is dead code

**File:** `crates/cintx-oracle/src/compare.rs:599-603`

```rust
for l in 0..5_i32 {
    let _ = l; // l is used in the loop, suppress warning
}
```

This loop does nothing (the comment is self-contradictory — `l` is NOT used). It is leftover scaffolding from a `CINTlen_spinor` comparison that was moved to the per-shell loop below. Remove it.

### IN-03: `_ibase_kbase_used` binding is dead

**File:** `crates/cintx-cubecl/src/kernels/two_electron.rs:902`

`let _ibase_kbase_used = (shape.ibase, shape.kbase);` is a no-op "for auditability" binding. It computes a tuple and immediately discards it. If the intent is documentation, a comment suffices; the binding adds noise and a clippy `let_underscore` smell.

### IN-04: Duplicated `cart_comps`, `common_fac_sp`, and `SQRTPI` across five kernel modules

**File:** `crates/cintx-cubecl/src/kernels/one_electron.rs:41-67`, `crates/cintx-cubecl/src/kernels/two_electron.rs:23-46`, `crates/cintx-cubecl/src/kernels/center_3c2e.rs:35-60`, `crates/cintx-cubecl/src/kernels/f12.rs:31-54`

`common_fac_sp`, `cart_comps`, and the `SQRTPI` constant are copy-pasted verbatim across at least four kernel modules. `f12.rs::cart_comps` returns `(u8,u8,u8)` while `center_3c2e.rs::cart_comps` returns `(usize,usize,usize)` — same logic, different types, inviting drift. Hoist to a shared `crate::math` or `crate::kernels::common` module. (Out of strict v1 correctness scope, but the duplication directly raises the risk of a future fix landing in only one copy.)

### IN-05: `vendor_int3c2e_ip1_cart` doc comment says "ip1 variant" without the GRAD/Risk tag the sph sibling carries

**File:** `crates/cintx-oracle/src/vendor_ffi.rs:869-874`

The cart wrapper doc is terse ("3-center 2-electron integral, ip1 variant") while the sph sibling (lines 900-908) documents the GRAD-08 / Risk R1 reference-flip rationale. Since CR-01/WR-01 show the cart legacy block was the one written correctly and the sph one was missed, the asymmetric documentation likely contributed to the inconsistency. Bring the cart doc to parity with the sph one.

---

_Reviewed: 2026-05-26T12:58:54Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
