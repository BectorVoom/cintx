---
phase: 29-group-4-relativistic-spin-operator-integrals-spinor
reviewed: 2026-06-01T00:00:00Z
depth: standard
files_reviewed: 13
files_reviewed_list:
  - crates/cintx-cubecl/src/kernels/f12.rs
  - crates/cintx-cubecl/src/kernels/mod.rs
  - crates/cintx-cubecl/src/kernels/sigma_1e.rs
  - crates/cintx-cubecl/src/kernels/sigma_1e_nuc.rs
  - crates/cintx-cubecl/src/kernels/two_electron.rs
  - crates/cintx-cubecl/src/transform/c2spinor.rs
  - crates/cintx-oracle/build.rs
  - crates/cintx-oracle/src/fixtures.rs
  - crates/cintx-oracle/src/vendor_ffi.rs
  - crates/cintx-oracle/tests/rel_1e_sigma_parity.rs
  - crates/cintx-oracle/tests/rel_2e_sigma_parity.rs
  - crates/cintx-oracle/tests/si_2e_transform_parity.rs
  - crates/cintx-oracle/tests/si_transform_parity.rs
findings:
  critical: 1
  warning: 4
  info: 0
  total: 5
status: issues_found
---

# Phase 29: Code Review Report

**Reviewed:** 2026-06-01
**Depth:** standard
**Files Reviewed:** 13
**Status:** issues_found

## Summary

Phase 29 adds the on-device `#[cube]` σ kernels (`sigma_1e.rs` overlap engine,
`sigma_1e_nuc.rs` Rys nuclear engine), the 1e/2e cart→spinor σ-mix transform suite
in `c2spinor.rs`, and the generic REL-03/04 2e σ launcher in `two_electron.rs`.
The verbatim-transcribed σ folds and the transform layout are vendor byte-identical
on the kappa fixtures (rel_1e 10/10, rel_2e 18/18, si_2e_transform 4/4 per the phase
record), and the fail-closed staging guards are present on every new inline Spinor
dispatch arm (`launch_int1e_sigma_family_spinor_pair` sigma_1e.rs:713-719;
`launch_rel2e_sigma_spinor_quartet` two_electron.rs:3037-3044). Buffer transposes,
strides, and signs in the transforms hold up under trace.

Two of the reviewer-context hypotheses did NOT survive verification:

- **The `f12.rs` "dead scaffolding" claim is FALSE.** `Rel2eStep` / `Rel2eOp` /
  `Rel2eExp` / `build_rel2e_cascade` / `s9_products` and the `gout_rel2e_*` /
  `gout_ssp*` / `gout_spv*` / `gout_vsp*` family gouts are all LIVE: they are reached
  from `two_electron.rs::launch_rel2e_sigma_spinor` (base profile, not f12-gated) via
  `rel2e_family_dispatch`, and exercised by the rel_2e_sigma_parity suite. A clean
  `cargo build -p cintx-cubecl` (default profile) emits NO dead_code warning for any
  of these symbols. Do not remove them.

The one genuine correctness defect is a silent nroots truncation in the nuclear σ
engine (CR-01) that produces wrong results for high-L shells (`li+lj >= 8`) instead
of failing closed — the only family-kernel in the workspace that clamps instead of
erroring. The d×p test fixture does not reach it, so it passed CI green while latent.

## Critical Issues

### CR-01: Nuclear σ kernel silently truncates Rys nroots for high-L shells (wrong results, not fail-closed)

**File:** `crates/cintx-cubecl/src/kernels/sigma_1e_nuc.rs:490`
**Issue:**
```rust
let order = li as usize + lj as usize + 2;
let nroots = ((order / 2 + 1) as u32).clamp(1, 5);
```
The required Rys root count for the composed `+1/+1` double-derivative order is
`(li+lj+2)/2 + 1`. `.clamp(1, 5)` SILENTLY caps it at 5. For `li+lj >= 8`
(e.g. g×g, or f shells with the +2 headroom), the true count exceeds 5 and the kernel
runs with too few roots, producing **numerically wrong nuclear-σ integrals** with no
error. The device kernel itself only dispatches `rys_root1..5` (sigma_1e_nuc.rs:232-242),
so there is no higher-root path; the clamp masks the gap rather than rejecting it.

This is the *only* family kernel in the tree that clamps. Every sibling fail-closes:
`center_2c2e.rs:1025` (`if nroots > MAX_DEVICE_NROOTS { return Err(...) }`),
`unstable/origk.rs:1461`, and the in-phase sibling `two_electron.rs:3051-3055`
(`if grad_shape.nroots > HOST_RYS_NROOTS_CEILING { return Err(UnsupportedApi{...}) }`).
There is NO upstream l-guard: `build_sigma_cart` (sigma_1e.rs:775) and
`launch_int1e_sigma_family_spinor_pair` (sigma_1e.rs:680) impose no angular-momentum
ceiling, and the c2spinor coefficient tables support l up to 4
(`bra_coeff_refs` c2spinor.rs:951-985), so g×g spinor shells are constructible and
reach this path. The d×p kappa fixture (order=4, nroots=3) never trips it, which is
why parity is green while the bug is latent.

This violates the CLAUDE.md OOM-safe / typed-failure contract ("fallible ... typed
failure ... no partial writes" — silent wrong output is worse than a partial write).

**Fix:** Replace the clamp with a fail-closed guard mirroring the sibling launchers:
```rust
const MAX_DEVICE_NROOTS: u32 = 5;
let order = li as usize + lj as usize + 2;
let nroots = (order / 2 + 1) as u32;
if nroots > MAX_DEVICE_NROOTS {
    return Err(cintxRsError::UnsupportedApi {
        requested: format!(
            "int1e_{op}_spinor nuclear σ requires nroots={nroots} > device cap \
             {MAX_DEVICE_NROOTS} (li+lj={})", li as usize + lj as usize
        ),
    });
}
```
(`run_sigma_nuc_on_backend` already returns `Result`, so propagation is trivial; the
`op` name is in scope.)

## Warnings

### WR-01: `gt_coeff_rows` / `lt_coeff_rows` return empty Vecs for l>4 instead of erroring (inconsistent with `bra_coeff_refs` panic)

**File:** `crates/cintx-cubecl/src/transform/c2spinor.rs:250` and `:277`
**Issue:** `gt_coeff_rows` / `lt_coeff_rows` have `_ => (vec![], vec![])` arms for
`l > 4`, but `bra_coeff_refs` (c2spinor.rs:983) `panic!`s on the same input. The
single-block helpers (`cart_to_spinor_si`, etc.) call `gt/lt_coeff_rows` then index
`coeff_r[i]` for `i in 0..nd` where `nd = spinor_len(l,kappa) > 0` — so an l>4 shell
yields an empty coeff vec and an out-of-bounds index panic inside `apply_si_block`,
rather than a clean typed error. The two l>4 failure modes (panic vs OOB-panic vs
typed error) are inconsistent across the transform surface. l>4 spinor shells are
exotic but constructible.
**Fix:** Make all three (`gt_coeff_rows`, `lt_coeff_rows`, `bra_coeff_refs`) return a
`Result`/typed error for l>4 (preferred), or at minimum make the empty-Vec arms
unreachable by validating `l <= 4` once at the public entry points
(`cart_to_spinor_si*` / `cart_to_spinor_sf*`) and returning
`cintxRsError::UnsupportedApi`. Do not leave a silent empty-Vec → OOB-panic path.

### WR-02: Per-iteration heap allocation in 2-sided e2 path masks the intent of the pre-allocated `opij_buf`

**File:** `crates/cintx-cubecl/src/kernels/two_electron.rs:3184`
**Issue:** Inside the `E2Transform::Si | E2Transform::SiI` arm, the per-contraction-quad
loop allocates `let mut scratch = vec![0.0_f64; opij_len];` for each of the 4 e2
components (3184), then copies it into the already-allocated `opij_buf` slot (3186).
The function deliberately pre-allocates `opij_buf` of size `n_e2_blocks * opij_len`
(3133) to avoid exactly this — but the 2-sided arm round-trips through a fresh Vec +
`copy_from_slice` on every (ci,cj,ck,cl)×e2c, defeating that intent and adding 4
allocations per quad. `run_e1` already zeroes its slot (3143-3145), so writing
directly into `&mut opij_buf[e2c*opij_len..(e2c+1)*opij_len]` is sound. (Performance
is out of v1 scope; flagged as a maintainability/clarity defect — the dead
pre-allocation is misleading to a future reader.)
**Fix:** Write `run_e1` straight into the buffer slot:
```rust
for e2c in 0..4 {
    let (lo, hi) = (e2c * opij_len, (e2c + 1) * opij_len);
    run_e1(&mut opij_buf[lo..hi], &cart_blocks, base, e2c * 4)?;
}
```

### WR-03: Misleading `#[allow(dead_code)]` on `family_id` (the function is used)

**File:** `crates/cintx-cubecl/src/kernels/sigma_1e.rs:63`
**Issue:** `family_id` carries `#[allow(dead_code)]`, but it is called by
`build_sigma_cart` (sigma_1e.rs:793) in the same module and same profile — it is not
dead. The unnecessary `#[allow]` suppresses a lint that would never fire and signals
to a reader that the function is unused/speculative when it is on the live path.
**Fix:** Remove the `#[allow(dead_code)]` attribute from `family_id`.

### WR-04: Stale `oracle_covered=false` doc comments contradict the live `=true` assertions

**File:** `crates/cintx-oracle/tests/rel_1e_sigma_parity.rs:482` (and
`crates/cintx-oracle/tests/si_2e_transform_parity.rs:286-287`)
**Issue:** The `test_rel_1e_rows_registered_without_vendor` header comment
(rel_1e_sigma_parity.rs:482) states the rows read "oracle_covered=false", but the test
body asserts `assert!(entry.oracle_covered, ...)` (line 506) — i.e. `=true`. Likewise
si_2e_transform_parity.rs:286-287 says int2e_spsp1_spinor "MUST stay
oracle_covered=false this plan" while `test_no_silent_skip` (lines 318-323) asserts it
is `=true`. The comments describe a pre-flip (RED) state that the code has since moved
past; they now mislead about the manifest-coverage contract.
**Fix:** Update both header comments to state `oracle_covered=true` (post Task-3 flip),
matching the assertions.

---

_Reviewed: 2026-06-01_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
