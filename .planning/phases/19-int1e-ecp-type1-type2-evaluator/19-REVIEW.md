---
phase: 19-int1e-ecp-type1-type2-evaluator
reviewed: 2026-05-20T12:43:52Z
depth: deep
files_reviewed: 13
files_reviewed_list:
  - crates/cintx-cubecl/src/kernels/ecp.rs
  - crates/cintx-cubecl/src/math/ecp_k_taylor.rs
  - crates/cintx-cubecl/src/math/ecp_k_taylor_data.rs
  - crates/cintx-cubecl/src/math/mod.rs
  - crates/cintx-oracle/build.rs
  - crates/cintx-oracle/src/lib.rs
  - crates/cintx-oracle/src/libecpint_ffi.rs
  - crates/cintx-oracle/src/vendor_ffi.rs
  - crates/cintx-oracle/tests/safe_api_ecp_parity.rs
  - crates/cintx-oracle/tests/ecp_libecpint_crosscheck_parity.rs
  - xtask/src/gen_ecp_tables.rs
  - xtask/src/main.rs
  - .github/workflows/compat-governance-pr.yml
findings:
  critical: 1
  high: 2
  medium: 4
  low: 5
  total: 12
status: issues_found
---

# Phase 19: Code Review Report — ECP K-Taylor Byte-Identity Port

**Reviewed:** 2026-05-20T12:43:52Z
**Depth:** deep (cross-file: kernel ↔ math port ↔ vendored C source ↔ FFI ↔ CI gate)
**Files Reviewed:** 13
**Status:** issues_found
**Advisory:** This review is non-blocking. Parity tests pass independently at atol=1e-12.

## Summary

Phase 19 ports PySCF `nr_ecp.c` / `nr_ecp_deriv.c` to a host-only Rust kernel for
byte-identity ECP scalar + gradient integrals. The numerical port itself is high
quality: I traced `ecpsph_ine_opt_host`, `ecprad_part_host`, the level-adaptive
convergence loops, the `_l_down`/`_l_up` derivative recurrence, and the K-Taylor
table buffer bounds against the vendored C (`vendor/pyscf-nr-ecp/src/nr_ecp.c`)
and confirmed the order-7 table interpolation, the `order>7` downward-recurrence
`k0/k1` buffer sizing (`K_TAB_COL*2`), the `SIM_ZERO` early-break `nrs_now`
semantics, and the `saturating_sub(1)/2` underflow mirror are all faithful to the
C and within the Phase-19 angular envelope (`OFFSET_CART`, `FACTORIAL2`, `BINOM`,
`xx/yy/zz[16]` all stay in bounds for `li+lc <= ~10`). That part is genuinely solid.

The defects are concentrated in the **public library panic surface** (CLAUDE.md
mandates typed `thiserror` errors, no panics, in `cintx-cubecl`/`cintx-ops`
library paths) and in the **OOM/partial-write contract** (CLAUDE.md: "Best-effort
partial writes on allocation failure ... Use instead: Fallible allocation + typed
failure + no partial writes"). The largest concern is a panic-on-untrusted-input
path in `launch_ecp` reachable from the safe API, plus a silent partial-write
truncation in the spherical output path that contradicts the project's OOM-safe
stop contract. The FFI shims (`vendor_ffi.rs`, `libecpint_ffi.rs`) are correctly
structured; their buffer-size guards are `debug_assert!` only (release no-op),
which is a real but lower-severity gap.

## Critical Issues

### CR-01: `launch_ecp` panics on out-of-range `atom_index` instead of returning a typed error

**File:** `crates/cintx-cubecl/src/kernels/ecp.rs:1369-1371` (and `:1413`)
**Issue:** The launcher indexes the atom slice with caller-derived indices and no
bounds check:
```rust
let ri = atoms[shell_i.atom_index as usize].coord_bohr;
let rj = atoms[shell_j.atom_index as usize].coord_bohr;
// ... and inside the slot loop:
let rc = atoms[slot.atom_index as usize].coord_bohr;
```
`shell_i.atom_index` / `slot.atom_index` originate from `Shell` / `EcpShell`
typed data that the safe API accepts from callers. If any `atom_index` is `>=
atoms.len()` (a malformed `BasisSet` where shells/ECP-shells reference an atom
that does not exist), this is an **array-index panic in a public, non-test library
path**. CLAUDE.md requires "clear failure modes" and the project uses `thiserror`
v2 for exactly this — a panic unwinds across the safe-API boundary instead of
producing `cintxRsError`. This is the same class of failure the kernel already
guards against for `shells.len() != 2` (`:1343`) and empty `ecp_shells` (`:1351`),
so the omission here is inconsistent, not intentional.
**Fix:** Validate every atom index before dereferencing and return a typed error:
```rust
let atom_for = |idx: u32| -> Result<[f64; 3], cintxRsError> {
    atoms
        .get(idx as usize)
        .map(|a| a.coord_bohr)
        .ok_or_else(|| cintxRsError::ChunkPlanFailed {
            from: "cubecl_ecp",
            detail: format!("atom_index {idx} out of range (natm={})", atoms.len()),
        })
};
let ri = atom_for(shell_i.atom_index)?;
let rj = atom_for(shell_j.atom_index)?;
// inside the loop:
let rc = atom_for(slot.atom_index)?;
```

## High

### HI-01: Spherical output path silently truncates on undersized `staging` — violates the "no partial writes on OOM" contract

**File:** `crates/cintx-cubecl/src/kernels/ecp.rs:1468-1476`
**Issue:** When the spherical `staging` buffer is too small, the code does NOT
fail closed; it writes a partial result:
```rust
} else {
    let mut sph_tmp = vec![0.0_f64; sph_block];
    cart_to_sph_1e(cart_slice, &mut sph_tmp, li, lj);
    let avail = staging.len().saturating_sub(out_off);
    let copy_len = avail.min(sph_block);
    staging[out_off..out_off + copy_len].copy_from_slice(&sph_tmp[..copy_len]);
}
```
This is precisely the "best-effort partial write on allocation failure" anti-pattern
CLAUDE.md forbids ("Violates the design's OOM-safe stop contract ... Use instead:
Fallible allocation + typed failure + no partial writes"). A caller that under-sized
the buffer gets a half-filled matrix and a successful `Ok(ExecutionStats)`, with no
signal that the data is incomplete. Note the gradient path *does* fail closed with a
typed error (`:1387-1395`); the scalar/spheric path should match it.
**Fix:** Replace the truncating fallback with a fail-closed size check (mirroring
the gradient guard at `:1380-1396`) that returns `cintxRsError::ChunkPlanFailed`
before writing anything into `staging`. Compute `needed = n_comp * sph_block`
up front for both scalar and gradient and reject undersized buffers once.

### HI-02: Scalar output path has no buffer-size precheck and `Cart` path also truncates

**File:** `crates/cintx-cubecl/src/kernels/ecp.rs:1379-1396, 1479-1482`
**Issue:** The buffer-size invariant (`needed = n_comp * ...`) is only enforced
`if is_gradient` (`:1380`). For the scalar operator (`ecp`), there is no size
precheck at all. The `Representation::Cart` write then does
`let copy_len = staging.len().min(gctr.len()); staging[..copy_len].copy_from_slice(...)`
(`:1480-1481`) — again a silent truncation rather than a fail-closed error if
`staging` is shorter than `gctr`. Same OOM-contract violation as HI-01, on a path
that runs for every scalar ECP integral. The kernel layer is the last line of
defense; relying solely on the `query_workspace` preflight to size the buffer
leaves the kernel's own contract unchecked.
**Fix:** Hoist the size check out of the `if is_gradient` block so it covers both
operators and both representations, and make the `Cart` branch assert
`staging.len() >= gctr.len()` (typed error otherwise) before the copy.

## Medium

### ME-01: `raise_idx` uses `.expect()` panic in a hot library path

**File:** `crates/cintx-cubecl/src/kernels/ecp.rs:1007-1011`
**Issue:** `raise_idx` ends with
`.position(...).expect("raised cartesian component must exist in cart_comps(l+1)")`.
It is called from `l_down`/`l_up` (per-component, per-primitive-pair) on the
gradient path, which is reachable from the safe API. While the invariant *should*
hold for `axis ∈ {0,1,2}` and valid `l`, an `.expect()` in `cintx-cubecl` library
code is the panic-vs-typed-error pattern CLAUDE.md flags. It is also recomputing
`cart_comps(l)` and `cart_comps(l+1)` (two heap `Vec` allocations) on every call.
**Fix:** Compute the target index arithmetically from the known `cart_comps`
ordering (the index of `(lx,ly+1,lz)` / `(lx,ly,lz+1)` is closed-form given the
column-major triangular enumeration) so it cannot fail and allocates nothing; or
at minimum return `Option<usize>` and propagate. Since this is an internal
invariant a `debug_assert!` + arithmetic mapping is the cleanest resolution.

### ME-02: FFI buffer-size guards are `debug_assert!` (no-op in release) and check only `len % 3 == 0`

**File:** `crates/cintx-oracle/src/vendor_ffi.rs:1378-1382, 1418-1422`
**Issue:** `vendor_ECPscalar_ipnuc_{sph,cart}` guard the `out` buffer with
`debug_assert!(out.len() % 3 == 0, ...)`. This is (a) compiled out in release/CI
profiles, and (b) far weaker than the documented invariant — the buffer must be
exactly `3 * nao_i * nao_j`, but `len % 3 == 0` passes for any multiple of 3,
including a buffer that is too small. The C `ECPscalar_ipnuc_*` writes
`3*nao_i*nao_j` f64 through the raw pointer regardless, so an undersized `out`
is a heap buffer overflow in the unsafe FFI call. (This is oracle/test harness
code, so severity is medium rather than critical, but it is an FFI memory-safety
guard that does not actually guard.)
**Fix:** Take the expected element count as an explicit argument (or compute
`nao_i*nao_j` from `shls`+`bas` via `CINTcgto_*`) and use a hard `assert_eq!`
(not `debug_assert!`) that `out.len()` equals `3 * nao_i * nao_j` before the
unsafe call, so the bound holds in release builds too.

### ME-03: `parse_static_double_array` brace matching is positional, not balanced — silently mis-parses on a future source shape

**File:** `xtask/src/gen_ecp_tables.rs:70-109`
**Issue:** The extractor finds the first `{` after the declaration and the first
`};` after that, then splits the body on commas. This works for the current flat
`static double _sph_ine_tab[] = { ... };` literal, but it is not a real tokenizer:
(1) it assumes the array initializer contains no nested braces and no `};`
substring inside a comment; (2) the only `//`-comment stripping is per-line, so a
`/* ... */` block comment or a comma inside a comment would corrupt the token
stream; (3) it does not anchor `static double NAME[]` to a line start, so a
substring match (e.g. a forward declaration in a comment) could pick the wrong
occurrence. Because the drift gate's whole purpose (D-15) is to fail closed on
vendored-source edits, a parser that mis-parses a *reshaped* (but still valid)
source while still producing the right element count would defeat the gate
silently. The element-count check (`extract_table`) catches gross changes but not
a subtle reordering that preserves count.
**Fix:** Anchor the declaration match to a line boundary, strip `/* */` comments
as well as `//`, and either (a) do balanced-brace matching for the initializer or
(b) assert the matched body contains no `{` after stripping comments (fail closed
if it does). Add a unit test with a block-comment / nested-brace decoy.

### ME-04: CI drift gate is the only `--locked` step in its job; the gate above it is not — drift between lockfile and gate is possible

**File:** `.github/workflows/compat-governance-pr.yml:73-75`
**Issue:** The new step runs `cargo run --manifest-path xtask/Cargo.toml --locked
-- gen-ecp-tables --check`, but the immediately preceding `manifest-audit` step
(`:70-72`) runs without `--locked`. Per CLAUDE.md ("run CI with `cargo --locked`")
both should be locked, or the inconsistency means the manifest gate can resolve a
different dependency graph than the table gate within the same job. More
importantly, if `Cargo.lock` is out of date the `--locked` table gate will fail
the PR with a *lockfile* error that looks like a *table drift* error, masking the
real cause. The drift-check logic in `gen_ecp_tables.rs` itself is correct and
does fail closed on a byte diff (verified: `check_blob` bails on both length and
content mismatch), so the gate's substance is sound — this is purely the CI wiring.
**Fix:** Add `--locked` to the `manifest-audit` step for consistency, and confirm
the lockfile is committed/current so a `--locked` failure cannot be confused with
ECP table drift.

## Low

### LO-01: `not0` non-zero counter uses a different threshold than the rest of the codebase

**File:** `crates/cintx-cubecl/src/kernels/ecp.rs:1492`
**Issue:** `let not0 = staging.iter().filter(|&&v| v.abs() > 1e-18).count()`. The
`1e-18` magic threshold is undocumented and arbitrary; the gradient zero-overlap
test uses `1e-12` and the parity gate uses `1e-12`. A value in `(1e-18, 1e-12)`
would be counted as "non-zero" here but treated as zero elsewhere. Low impact
(`not0` is a stat, not a correctness gate) but the constant should be named and
justified or aligned with the project's zero threshold.
**Fix:** Hoist to a named `const ECP_NONZERO_EPS: f64` with a comment, or reuse
the shared tolerance constant if one exists.

### LO-02: Dead local bindings retained with `let _ =` instead of removed

**File:** `crates/cintx-cubecl/src/kernels/ecp.rs:871-872, 921-922`
**Issue:** `let pradi_off = ic * nrs_alloc * lilc1; let _ = pradi_off;` and
`let dlclmb = (li + lc + 1) * dlc; let _ = dlclmb;` are computed-then-discarded.
These are dead computations kept only to silence unused-variable warnings, with a
comment explaining the layout. They add noise and a (tiny) wasted multiply per
shell pair. If they document intent, a comment alone suffices.
**Fix:** Delete the bindings; move the explanatory text into the adjacent layout
comment.

### LO-03: `EcpSlot::_marker: PhantomData<&'a ()>` lifetime appears unused

**File:** `crates/cintx-cubecl/src/kernels/ecp.rs:525-532, 537`
**Issue:** `EcpSlot<'a>` carries a `PhantomData<&'a ()>` but every field
(`atom_index: u32`, `lc: i32`, `rad_shells: Vec<EcpRadShell>`) is owned — the
`rad_shells` are built via `.to_vec()` in `group_ecp_slots`, so there is no actual
borrow tying the slot to `'a`. The lifetime parameter is vestigial and the
`PhantomData` exists only to use it.
**Fix:** Drop the `'a` parameter and the `_marker` field; `group_ecp_slots` can
return `Vec<EcpSlot>` with no lifetime.

### LO-04: `try_build_libecpint` link directives emitted before final cfg can leave dangling link search paths on a partial failure

**File:** `crates/cintx-oracle/build.rs` (`try_build_libecpint`, the `rustc-link-*`
block after `cxx.compile`)
**Issue:** The function emits `cargo:rustc-link-search`/`rustc-link-lib` only after
`cxx.compile("cintx_libecpint_shim")` succeeds, and `cc::Build::compile` panics
(aborts the build) on a compile failure rather than returning an error. So a shim
that fails to compile aborts the whole oracle build even though the feature is
documented as "best-effort, never fails the build on a host that lacks libecpint."
The env-var-set-but-library-absent paths are handled gracefully (warn + return),
but the env-var-set-and-shim-present-but-uncompilable path is not.
**Fix:** This is opt-in oracle tooling (anyhow domain), so a build abort is
tolerable, but document that `CINTX_LIBECPINT_SHIM` pointing at a non-compiling
shim aborts the build, or wrap the compile so a shim failure degrades to a
`cargo:warning` + skipped cfg like the other discovery failures.

### LO-05: `ecpsph_ine` series-convergence loop has no iteration cap

**File:** `crates/cintx-cubecl/src/math/ecp_k_taylor.rs:155-165`
**Issue:** The moderate-z branch loops `loop { ...; if next == s { break; } ...; k += 1; }`
relying on floating-point fixed-point convergence (`next == s`) to terminate. For
the validated z-range `[1e-7, 16]` this converges quickly (matches the C verbatim),
but there is no upper bound on `k`, and `FACTORIAL`/`J_INV`-free term growth is not
guarded. A pathological input (e.g. a NaN `z` slipping through, since the branch
guards are `z < 1e-7` / `z > 16` and NaN fails both comparisons, falling into this
`else`) would never satisfy `next == s` and would spin. The C has the identical
structure, so this is byte-faithful, but the C is called only after PySCF's own
NaN-free guarantees.
**Fix:** Add a defensive iteration cap (e.g. `k <= 200`) matching the series'
known convergence length, and/or short-circuit `!z.is_finite()` at the top of
`ecpsph_ine_opt_host` to return a typed/NaN-propagating result rather than
entering the unbounded loop.

---

## Verified Sound (adversarial checks that passed)

The following were specifically probed (per the review scope's "check for siblings
of the known usize-underflow" and "off-by-one in F-order layout" prompts) and found
correct:

- **`order>7` K-Taylor recurrence buffer bounds** (`ecp_k_taylor.rs:217-242`): `k0`/`k1`
  sized `K_TAB_COL*2 = 48`; reads `k0[i+1]` at `i <= order+K_TAYLOR_MAX-j`; matches
  C `buf[K_TAB_COL*2]` and the implicit `order+7 < 24` envelope. The `L2[24]`,
  `J_INV[10]`, `FACTORIAL[24]` tables are correctly sized.
- **`SIM_ZERO` early-break `nrs_now` semantics** (`ecp_k_taylor.rs:311-323`): matches
  the C `nrs_now = i` (break) vs `nrs_now = nrs` (completion) exactly, including the
  `i > 2` guard and the post-break exclusion of index `i`.
- **`saturating_sub(1)/2` underflow mirror** (`ecp.rs:675, 915`): byte-identical to the
  C signed `(0-1)/2 == 0`; only reachable at the final adaptive level after which
  the `while level <= LEVEL_MAX` loop exits — `start` is never used as an index in
  that iteration.
- **`radial_power` default arm** (`ecp_k_taylor.rs:342-348`): `for _ in 0..other` with
  `other: i32` no-ops for `power <= 0`, matching the C `for(n=0;n<power;n++)`.
- **Gradient F-order `[axis, ao_j, ao_i]` layout** (`ecp.rs:1230-1246`, parity test
  `safe_api_ecp_parity.rs:433-448`): cintx write index `axis*ni*nj + j*ni + i` and
  the PySCF read index `out[comp*dij + j*di + i]` agree for single contraction; the
  no-transpose claim holds.
- **Angular-helper table bounds** (`OFFSET_CART[15]`, `FACTORIAL2[40]`, `BINOM[55]`,
  `xx/yy/zz[16]`): all within bounds for the Phase-19 `li+lc <= ~10` envelope,
  including the `li+1` derivative shift.
- **Drift-gate fail-closed behavior** (`gen_ecp_tables.rs:143-171`): `check_blob`
  correctly bails on both byte-length and byte-content divergence; the CI gate
  genuinely fails on a stale/hand-edited blob (the gate's substance is sound; see
  ME-04 for the `--locked` wiring nit only).
- **Default-build no-op of the libecpint branch**: `build.rs` only enters
  `try_build_libecpint` when `CINTX_LIBECPINT_ORACLE` is set, and `lib.rs` /
  `libecpint_ffi.rs` / the crosscheck test are all `#[cfg(has_libecpint_oracle)]`,
  so the default build is unchanged (threat T-19-27 mitigated).

---

_Reviewed: 2026-05-20T12:43:52Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
