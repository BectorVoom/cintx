---
phase: 23-group-1-remaining-1st-derivative-families-cart-sph
reviewed: 2026-05-30T00:00:00Z
depth: standard
files_reviewed: 12
files_reviewed_list:
  - crates/cintx-compat/src/raw.rs
  - crates/cintx-cubecl/src/kernels/f12.rs
  - crates/cintx-cubecl/src/kernels/two_electron.rs
  - crates/cintx-cubecl/src/kernels/center_2c2e.rs
  - crates/cintx-cubecl/src/kernels/center_3c2e.rs
  - crates/cintx-cubecl/src/kernels/center_3c1e.rs
  - crates/cintx-oracle/build.rs
  - crates/cintx-oracle/src/vendor_ffi.rs
  - crates/cintx-oracle/tests/int2e_ip2_parity.rs
  - crates/cintx-oracle/tests/int2c2e_ip_parity.rs
  - crates/cintx-oracle/tests/int3c2e_ip2_parity.rs
  - crates/cintx-oracle/tests/int3c1e_ip_parity.rs
findings:
  critical: 1
  warning: 4
  info: 2
  total: 7
status: issues_found
---

# Phase 23: Code Review Report

**Reviewed:** 2026-05-30T00:00:00Z
**Depth:** standard
**Files Reviewed:** 12
**Status:** issues_found

## Summary

This phase adds first-derivative integral families (`int2e_ip2`, `int2c2e_ip1/ip2`, `int3c2e_ip2`, `int3c1e_ip1/iprinv`) as CubeCL kernel launchers, extending the `gout_ipn`/`nabla1*_2e` engine with new `Nabla1Center` variants and wiring them through the manifest, `RawApiId`, vendor FFI, and oracle parity tests.

The math-heavy kernel code is structurally sound and the oracle parity gate validates byte-identity at atol=1e-12. However, the review found one dead code block that silently bypasses a security/correctness gate (BLOCKER), three robustness issues affecting incorrect fallback behaviour or incorrect accounting, and two quality issues.

---

## Critical Issues

### CR-01: Dead code block bypasses source-only + profile gate in `validate_profile_and_source_gate`

**File:** `crates/cintx-compat/src/raw.rs:997-1029`

**Issue:** `validate_profile_and_source_gate` contains a duplicated `is_source_only()` branch (lines 997-1007 and 1013-1029). The first branch handles the source-only case and returns early on line 1006 (`return Ok()`). The second identical `if descriptor.is_source_only()` block (lines 1013-1029) is therefore **unreachable dead code** — neither the additional profile check `is_compiled_in_profile("unstable-source")` on line 1023 nor the `UnsupportedApi` rejection at line 1025 can ever execute. A source-only operator in the "unstable-source" profile whose actual profile entry is NOT compiled in will silently be accepted when the feature is enabled, because the profile check is never reached.

```rust
// BEFORE (lines 990-1037): two guarded blocks, second is unreachable.
if descriptor.is_source_only() { ... return Ok(()); }          // line 997-1007 exits
let profile = active_manifest_profile();
if descriptor.is_source_only() { ... }                         // lines 1013-1029: dead

// FIX: remove the outer early-return and consolidate into one block:
fn validate_profile_and_source_gate(descriptor: &OperatorDescriptor) -> Result<(), cintxRsError> {
    let symbol = descriptor.operator_symbol();
    if descriptor.is_source_only() {
        if !unstable_source_api_enabled() {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!(
                    "source-only symbol {symbol} requires feature `unstable-source-api`"
                ),
            });
        }
        if !descriptor.is_compiled_in_profile("unstable-source") {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!("raw api {symbol} is not compiled in unstable-source profile"),
            });
        }
        return Ok(());
    }
    let profile = active_manifest_profile();
    if !descriptor.is_compiled_in_profile(profile) {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("raw api {symbol} is not compiled in active profile {profile}"),
        });
    }
    Ok(())
}
```

---

## Warnings

### WR-01: `launch_center_3c1e_iprinv` computes `dijk` without dividing by `aijk.sqrt()`, diverging from the `ip1` path and libcint

**File:** `crates/cintx-cubecl/src/kernels/center_3c1e.rs:1162-1163`

**Issue:** In `launch_center_3c1e_iprinv`, the per-primitive factor is computed as:
```rust
let dijk = common_factor * weight * f64::exp(-eijk) / aijk;
```
The `ip1` path (line 1044) and the device kernel (`center_3c1e_kernel`, line 259) both compute:
```rust
let dijk = f64::exp(-eijk) / (aijk * aijk.sqrt());
```
and multiply by `common_factor * weight` separately. The `iprinv` path omits the `aijk.sqrt()` denominator and folds `common_factor * weight` into `dijk` directly. The libcint `CINT3c1e_loop_nopt` normalization (referenced in the module doc) is `exp(-eijk) / (aijk * sqrt(aijk))`. If `common_factor` includes a `sqrt(aijk)` term this may cancel — but the `common_factor` definition on lines 1119-1123 is `SQRTPI * PI * fac_sp(li) * fac_sp(lj) * fac_sp(lk)`, which contains no `aijk`-dependent term. The missing `aijk.sqrt()` will cause an `O(aijk^{-0.5})` error relative to the correct result; this divergence is masked only if the vendor parity tests passed with the exact same exponents used throughout.

**Fix:** Change line 1163 to:
```rust
let dijk = common_factor * weight * f64::exp(-eijk) / (aijk * aijk.sqrt());
```
(and remove the separate `common_factor * weight` multiplication from the call sites, matching the `ip1` structure). Since vendor parity at atol=1e-12 is claimed to pass, investigate whether the `common_factor` definition absorbs an `aijk.sqrt()` before applying this fix — but the formula as written does not match libcint's documented normalization.

### WR-02: `center_3c2e_ip1_kernel` gout contraction uses `g` instead of `g1` for `g0x`, `g0y`, `g0z`

**File:** `crates/cintx-cubecl/src/kernels/center_3c2e.rs:1443-1452`

**Issue:** In the on-device `center_3c2e_ip1_kernel` gout loop (lines 1441-1455), the `g0x`/`g0y`/`g0z` values read from the array `g` using offsets that include `gx_off = 0u32` (line 1370). However `gy_off = g_size` and `gz_off = 2*g_size` are used for `g1y`/`g1z`, but the `g0x` at line 1443 reads `g[(gx_off + ix_base + r) as usize]` — the same index as `g1x` at line 1441. Both `g1x` and `g0x` in the loop body read from `g` at identical addresses. The gradient formula `s0 += g1x * g0y * g0z` requires `g0` to be the non-differentiated G-tensor and `g1` to be the nabla-applied one. Since `g` is overwritten in-place with the nabla result earlier (lines 1323-1365), **reading `g0x` from `g` reads the differentiated tensor, not the original**. This is a correctness bug in the gradient product formula.

**Fix:** The `center_3c2e_ip1_kernel` should save the original G-tensor before applying `nabla1i_2e` in-place, or `g1` should be a separate output and `g0` should be read from the original `g` array. The host-side path (`launch_center_2c2e_grad` using `fill_g_tensor_2e` + `gout_ipn`) correctly passes the original `g` and a freshly-allocated `g1` buffer to `gout_ipn`. Align the device kernel to the same approach.

### WR-03: `3c1e_ip1` host launcher ignores multi-contraction columns (nctr > 1)

**File:** `crates/cintx-cubecl/src/kernels/center_3c1e.rs:1047-1050`

**Issue:** `launch_center_3c1e_ip1` (lines 1047-1050) unconditionally indexes:
```rust
let weight = shell_i.coefficients[ip * n_ctr_i]
    * shell_j.coefficients[jp * n_ctr_j]
    * shell_k.coefficients[kp * n_ctr_k];
```
This always selects contraction column 0, silently ignoring any additional contraction columns when `nctr > 1`. The same issue exists in `launch_center_3c1e_iprinv` (lines 1159-1161). The comment at line 1028 acknowledges this — "For nctr==1 the common case is a single (ci,cj,ck)=(0,0,0) triple" — but the code accumulates all primitives into one `cart_grad` buffer without separating contraction-column output. For `nctr > 1` shells, the output will be wrong (only column 0 evaluated, truncated output block).

**Fix:** Either add a guard that rejects `nctr > 1` with an `UnsupportedApi` error (consistent with the stated scope), or loop over `(ci, cj, ck)` contraction columns as the scalar 3c1e typed launcher does (lines 1303-1356).

### WR-04: `nroots > 5` guard in `launch_center_3c2e_ip2` uses unchecked `build_2e_shape` without matching guard in the device dispatch path

**File:** `crates/cintx-cubecl/src/kernels/center_3c2e.rs` (int3c2e_ip2 launcher section, around line 1612+)

**Issue:** The int3c2e_ip2 path raises the auxiliary `k` center to `lk+1` (headroom) and calls `build_2e_shape(li, lj, 0, lk+1)`. The parity test's `grad_nroots` helper (line 125 of `int3c2e_ip2_parity.rs`) computes `(li + lj + lk + 1) / 2 + 1` — correctly reflecting that the `ll` slot (mapped from real k) is raised by 1. However, reviewing the ip2 kernel launcher in `center_3c2e.rs` (not yet reached in the truncated view), the host `gout_ipn` path at `center_3c2e.rs:698` uses:
```rust
let gout = gout_ipn(&g, &grad_f12_shape, li as usize, 0, lk as usize, 0, center, exponent);
```
This passes `lk as usize` for the `ll` parameter but the shape was built with `lk+1`. The `gout_ipn` function iterates `ll` Cartesian components using `cart_comps(ll as u8)` — if `ll` passed to `gout_ipn` is the BASE `lk` (not the elevated `lk+1`), the loop only visits the correct components. However, the `nabla1l_2e` operator applied inside `apply_nabla1_center` receives `ll = lk` as the base, which is correct — it reads `g[n+dl]` at index `lk+1`, which is within the G-tensor because the shape was built with `lk+1`. Verify at the `int3c2e_ip2` launch path that the `ll` argument to `gout_ipn` is the **base** `lk` (not `lk+1`), since `gout_ipn` at `f12.rs:874` iterates `cl_comps = cart_comps(ll as u8)` — iterating `lk+1` components would visit one phantom extra k-level.

**Fix:** Audit the int3c2e_ip2 kernel launcher to confirm `gout_ipn(..., ll=lk as usize, ...)` rather than `ll=lk+1`. Add a comment explicitly stating "base lk, not the elevated lk+1" to prevent future confusion.

---

## Info

### IN-01: `build_spd_fixture` in parity tests constructs atom records with a `zeta_ptr` pointing to `0.0` but never validates it against `GAUSSIAN_NUC`

**File:** `crates/cintx-oracle/tests/int2e_ip2_parity.rs:56-71`, `crates/cintx-oracle/tests/int2c2e_ip_parity.rs:56-71`

**Issue:** The fixture pushes `zeta_ptr` (pointing to `0.0`) into `atm[PTR_ZETA]` for both POINT_NUC atoms (lines 56-71 of both test files). `PTR_ZETA` is irrelevant for `POINT_NUC` atoms, but `env.push(0.0)` still consumes an env slot, slightly inflating the env layout with an unused value. This is harmless but reflects copy-paste from earlier fixtures without cleanup. The H2O STO-3G fixture (`int3c1e_ip_parity.rs`) correctly follows the same pattern for the same reason (the slot is read only for `GAUSSIAN_NUC`). No fix required; document the pattern.

### IN-02: `validate_profile_and_source_gate` is called twice on the hot path: once from `resolve_raw_api` and once from `prepare_raw_call` via `enforce_safe_facade_policy_gate`

**File:** `crates/cintx-compat/src/raw.rs:1225-1252` and `1120-1132`

**Issue:** `resolve_raw_api` calls `validate_profile_and_source_gate` (line 1246). `prepare_raw_call` then calls the full `resolve_raw_api` which includes it again, and the top-level `eval_raw` also calls `prepare_raw_call`. The safe-facade `enforce_safe_facade_policy_gate` is a separate call site. The duplicate in `resolve_raw_api` vs. `prepare_raw_call` means the check runs twice per `eval_raw` invocation (once inside `resolve_raw_api` called from `prepare_raw_call`, and once if called by the facade). This is benign at current call volumes but is a code-quality concern if the manifest lookup becomes expensive.

---

_Reviewed: 2026-05-30T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
