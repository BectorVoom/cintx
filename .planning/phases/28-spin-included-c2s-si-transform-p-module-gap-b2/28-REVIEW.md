---
phase: 28-spin-included-c2s-si-transform-p-module-gap-b2
reviewed: 2026-05-31T00:00:00Z
depth: standard
files_reviewed: 11
files_reviewed_list:
  - crates/cintx-cubecl/src/transform/c2spinor.rs
  - crates/cintx-cubecl/src/kernels/sigma_p.rs
  - crates/cintx-cubecl/src/kernels/one_electron.rs
  - crates/cintx-cubecl/src/kernels/mod.rs
  - crates/cintx-oracle/src/compare.rs
  - crates/cintx-oracle/src/fixtures.rs
  - crates/cintx-oracle/src/vendor_ffi.rs
  - crates/cintx-oracle/tests/si_transform_parity.rs
  - crates/cintx-oracle/build.rs
  - xtask/src/oracle_covered_update.rs
  - xtask/Cargo.toml
findings:
  critical: 1
  warning: 3
  info: 2
  total: 6
  critical_resolved: 1
status: issues_found
resolution_note: "CR-01 (fail-closed staging guard) resolved in commit 8d049cc. WR-01/WR-02/WR-03/IN-01/IN-02 remain open as non-blocking follow-ups."
---

# Phase 28: Code Review Report

> **Orchestrator note (2026-05-31):** CR-01 (critical) was fixed in commit `8d049cc` —
> a `BufferTooSmall` fail-closed guard was added to the `int1e_sp` Spinor arm in
> `one_electron.rs` before any scatter write, mirroring `launch_int1e_sp_spinor_pair`.
> Build + FND-05 parity (6/6 byte-identical) re-verified green. WR-01, WR-02, WR-03,
> IN-01, IN-02 remain open as non-blocking follow-ups for a future plan.

**Reviewed:** 2026-05-31T00:00:00Z
**Depth:** standard
**Files Reviewed:** 11
**Status:** issues_found

## Summary

Phase 28 adds the spin-included `cart_to_spinor_si_2d` host transform (Plan 28-01), the
`sigma_p_kernel` device `#[cube]` G-tensor assembler (Plan 28-02), manifest/FFI infra for
`int1e_sp_spinor` (Plan 28-03), and the end-to-end `int1e_sp` Spinor dispatch arm in
`one_electron.rs` (Plan 28-04), all verified by the `si_transform_parity.rs` byte-identity
gate.

The core mathematical implementation — `apply_bra_si_block_one` sign convention, KET→BRA
transpose ownership in `cart_to_spinor_si_2d`, `spinor_len` kappa dispatch, VRR/HRR/nabla
G-tensor index bounds, and the D-01/SC#4 `is_skipped_spinor_fixture` guard — is correct.

One critical defect was found: the live `int1e_sp` arm in `one_electron.rs` writes into
`staging` without first verifying it is large enough, violating the project's fail-closed
OOM-safe stop contract. Three warnings and two info items cover a spurious atomic write
inside a hot inner loop, a misleading doc comment in the fixture file, and minor quality
issues in the backend dispatch.

---

## Critical Issues

### CR-01: Missing staging buffer size guard in `one_electron.rs` `is_sp` arm (OOM-safe stop violated)

**File:** `crates/cintx-cubecl/src/kernels/one_electron.rs:10554-10589`
**Issue:** The live `int1e_sp` Spinor dispatch arm allocates `scratch` (correctly sized at
`di * dj * 2`) and then scatters into the caller-provided `staging` buffer using indices up to
`(j_global * ni_sp + i_global) * 2 + 1` where `j_global` reaches `(n_ctr_j - 1) * dj + (dj - 1)`
and `i_global` reaches `(n_ctr_i - 1) * di + (di - 1)`. The maximum valid destination index is
`(nj_sp - 1) * ni_sp * 2 + (ni_sp - 1) * 2 + 1 = ni_sp * nj_sp * 2 - 1`. There is no guard
asserting that `staging.len() >= ni_sp * nj_sp * 2` before any write begins.

The project's OOM-safe stop contract (CLAUDE.md, Phase 25 `fnd06_chunk_staging_is_full_block`)
requires that all buffer size checks happen **before any write**. `launch_int1e_sp_spinor_pair`
in `sigma_p.rs` (the standalone public entry point) correctly enforces this with a
`BufferTooSmall` return at lines 596-601. The `one_electron.rs` is_sp arm bypasses that
function entirely and does not replicate its guard. A workspace-query miscomputation or a
caller size error would produce an out-of-bounds write into `staging` (a `&mut [F]` slice),
causing undefined behavior at the Rust level — a panic in debug mode or silent memory
corruption in release.

**Fix:**
```rust
// Insert before the `let mut scratch = ...` allocation (around line 10554):
let staging_required = ni_sp * nj_sp * 2;
if staging.len() < staging_required {
    return Err(cintxRsError::BufferTooSmall {
        required: staging_required,
        provided: staging.len(),
    });
}
```

---

## Warnings

### WR-01: Spurious read-modify-write on the scalar `gc_1` slot inside the hot primitive-contraction loop (`sigma_p.rs`)

**File:** `crates/cintx-cubecl/src/kernels/sigma_p.rs:297`
**Issue:** The kernel zeros the entire `gc_out` buffer at the top of the UNIT_POS block (lines
160-163). The scalar slot (`gc_1`, block index 3) stays zero for `int1e_sp`. Inside the hot
triple-nested primitive × contraction × cart loop the code then executes:

```rust
gc_out[b1 as usize] += F::new(0.0);
```

This is a load + add + store of zero — a no-op in every iteration. On the `CpuRuntime` backend
this generates a real memory read-modify-write per cart element per primitive pair per
contraction pair, which for even moderate shells (e.g. p × d: 18 elements, 9 primitive pairs)
costs hundreds of wasted memory operations. The correct form, consistent with the surrounding
`gc_1 stays 0.0` comment, is to simply omit the write entirely, since the buffer was already
zeroed.

There is also a secondary concern: `Phase-29` plans to use `tensor_rank == 3` (12-component
`int1e_sigma`) through the same comptime path. For `tensor_rank > 1` the scalar slot changes
meaning and the `+= F::new(0.0)` template will be duplicated × `tensor_rank` iterations per
cart element. The Phase-29 author must remember to replace this stub with the actual scalar
accumulation or the sigma family will silently produce wrong output. Adding a `comptime`-gated
`todo!` (or removing the stub entirely) would make that gap explicit.

**Fix:** Remove the dead write. The `gc_1` block is already zero from the initialization loop:
```rust
// Remove or comment out:
// gc_out[b1 as usize] += F::new(0.0);
```
For Phase 29 correctness, add an explicit note:
```rust
// Phase 29: for tensor_rank > 1 replace the scalar gc_1 slot accumulation here.
// For int1e_sp (tensor_rank == 1) the scalar slot stays 0.0 (already zeroed above).
```

---

### WR-02: `run_sigma_p_on_backend` match has no catch-all arm — non-exhaustive on no-backend builds

**File:** `crates/cintx-cubecl/src/kernels/sigma_p.rs:436-523`
**Issue:** The `match backend { ... }` block has one `#[cfg(feature = "...")]` arm per backend
and no default `_ =>` arm. The same pattern exists in the pre-existing
`run_1e_grad_bra_on_backend` and related functions in `one_electron.rs`, so this is a
project-wide pattern, not unique to phase 28. However, `sigma_p.rs` is new code that adopts
the risk deliberately.

When compiled with zero backend features enabled the match is non-exhaustive. Rust will reject
this at compile time (the `ResolvedBackend` enum has no `cfg`-gated variants in the enum
definition), producing a cryptic "non-exhaustive patterns" error rather than an
`UnsupportedApi` or a meaningful diagnostic. The existing `one_electron.rs` functions share
the same defect, but the phase 28 code adds another load-bearing function with the same gap.

**Fix:** Add a catch-all arm (consistent with the project pattern once it is fixed elsewhere):
```rust
#[allow(unreachable_patterns)]
_ => {
    return Err(cintxRsError::UnsupportedApi {
        requested: "no compute backend feature enabled for sigma_p kernel".to_owned(),
    });
}
```

---

### WR-03: Misleading `ROW-major` comment in `build_kappa_spinor_fixture` for a COLUMN-major coefficient array

**File:** `crates/cintx-oracle/src/fixtures.rs:329-333`
**Issue:** The comment reads:

```
// p-shell (bra i) — 3 primitives, two general-contraction columns. ROW-major
// here; libcint env is COLUMN-major env[ci*nprim + ip] (cintx transposes
// internally — project_raw_nctr_coeff_transpose). column 0 / column 1:
let p_coeff = [0.70_f64, 0.30, 0.15, 0.20, 0.55, 0.80];
```

The array literal `p_coeff` is then written verbatim into `env` via
`env.extend_from_slice(&p_coeff)` and registered in `bas[PTR_COEFF]`. Vendor libcint reads the
env block as column-major `[ic * nprim + ip]`, so the bytes in env represent:

- ic=0 (contraction 0): ip=0→0.70, ip=1→0.30, ip=2→0.15
- ic=1 (contraction 1): ip=0→0.20, ip=1→0.55, ip=2→0.80

That IS the column-major libcint layout. The comment's claim of "ROW-major here" is the
opposite of the actual layout. `extract_shell` correctly transposes from column-major (env) to
row-major (cintx `Shell`), so the behavior is correct — only the comment is wrong.

A future developer reading the comment before writing a similar fixture may store coefficients
in genuine row-major order, producing wrong vendor output and a silent coefficient transpose
mismatch in the parity test.

**Fix:** Correct the comment:
```rust
// p-shell (bra i) — 3 primitives, two general-contraction columns.
// Stored COLUMN-major in env (libcint convention): env[ic*nprim + ip].
// contraction 0 coefficients: 0.70 (prim 0), 0.30 (prim 1), 0.15 (prim 2)
// contraction 1 coefficients: 0.20 (prim 0), 0.55 (prim 1), 0.80 (prim 2)
// extract_shell transposes to row-major [ip*nctr + ic] for the cintx Shell.
```

---

## Info

### IN-01: `test_no_silent_skip` D-01 invariant only runs under the full vendor build — not caught by determinism-only CI

**File:** `crates/cintx-oracle/tests/si_transform_parity.rs:327-363`
**Issue:** The `test_no_silent_skip` test that enforces the D-01 invariant
(`int1e_sp_spinor` must stay `oracle_covered=false` in `MANIFEST_ENTRIES`) is gated on:

```rust
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
```

A CI run with only `--features cpu` (no `CINTX_ORACLE_BUILD_VENDOR=1`) compiles out this test
and the D-01 assertion never executes. A future commit that accidentally sets
`oracle_covered = true` in the manifest lock (e.g. via a premature `oracle-covered-update`
run) would not be caught until the full vendor build is triggered. The
`test_fixtures_build_without_vendor` test runs under `--features cpu` but only checks that the
fixture data structure is well-formed, not the oracle coverage flag.

This is not a code defect in the new implementation; the double-gate is documented by design.
It is worth tracking: Phase 29 adds the coverage flip so the window where this gap matters is
short, but during Phase 29 development a no-vendor CI lane could miss the invariant.

**Fix (optional hardening):** Extract the `MANIFEST_ENTRIES` check into a separate test gated
only on `#[cfg(feature = "cpu")]` (no `has_vendor_libcint` requirement) so the D-01 assertion
runs on every determinism-only build. The vendor arm of the test can remain double-gated.

---

### IN-02: `apply_bra1_zf_block` guards against `ncl == 0` but the analogous `apply_bra_block` and `apply_bra_si_block_one` do not

**File:** `crates/cintx-cubecl/src/transform/c2spinor.rs:861,1027` (vs 1516)
**Issue:** `apply_bra1_zf_block` computes `dk_total = if ncl > 0 { alpha_r.len() / ncl } else { 0 }`
(line 1516) to guard against division by zero when `ncl == 0`. The analogous computation in
`apply_bra_block` (line 861) and `apply_bra_si_block_one` (line 1027) is:

```rust
let di_total = alpha_r.len() / ncj;
```

with no guard. In practice `ncj = ncart(lj) >= 1` for any `lj >= 0`, so this cannot currently
produce a division by zero. The `apply_bra1_zf_block` guard was probably added defensively
because the 4D path calls it in more varied contexts. The inconsistency between the three
functions creates a subtle maintenance hazard: if these internal functions are ever reused in a
context where `ncj` could be zero (e.g. a future 0-dimensional shell stub), `apply_bra_block`
and `apply_bra_si_block_one` would panic while `apply_bra1_zf_block` would not.

The `dk_total` variable in `apply_bra1_zf_block` is computed but immediately suppressed with
`let _ = dk_total;` (line 1544), which suggests the guard is vestigial and the variable may
be a dead remnant.

**Fix (defensive):** Either add the same guard to `apply_bra_block` and
`apply_bra_si_block_one`, or remove the guard from `apply_bra1_zf_block` (along with the dead
`dk_total` binding) to make the three functions consistently unguarded. Given that `ncart` is
always >= 1, removing the vestigial guard and binding in `apply_bra1_zf_block` is the cleaner
path:

```rust
// apply_bra1_zf_block: remove these two lines (lines 1515-1516, 1544):
// let dk_total = if ncl > 0 { alpha_r.len() / ncl } else { 0 };
// ...
// let _ = dk_total;
```

---

_Reviewed: 2026-05-31T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
