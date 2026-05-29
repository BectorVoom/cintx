---
phase: 260529-mfh
plan: 01
subsystem: cintx-compat helpers / oracle parity
tags: [helper-parity, libcint-compat, CINTgto_norm, HELP-02, tdd]
requires: []
provides: [CINTgto_norm-libcint-parity]
affects: [crates/cintx-compat/src/helpers.rs, crates/cintx-oracle/tests/cintgto_norm_parity.rs]
tech-stack:
  added: []
  patterns: [dependency-free-closed-form, double-gated-vendor-parity-test]
key-files:
  created:
    - crates/cintx-oracle/tests/cintgto_norm_parity.rs
  modified:
    - crates/cintx-compat/src/helpers.rs
decisions:
  - "Use option (a): libcint misc.c documented closed form, dependency-free, no Cargo.toml change"
metrics:
  duration: ~12 min
  completed: 2026-05-29
---

# Phase 260529-mfh Plan 01: Fix CINTgto_norm to Match libcint misc.c Summary

Replaced the inverted `CINTgto_norm` formula with libcint 6.1.3 `misc.c`'s documented
closed form, restoring helper parity (HELP-02); the vendor oracle gate now advances past
the CINTgto_norm helper check to the next pre-existing blocker, `CINTc2s_bra_sph`.

## What Changed

- **`crates/cintx-compat/src/helpers.rs`** — `CINTgto_norm` body replaced. The prior body
  computed `sqrt( (2n-1)!! * sqrt(pi) / (2a)^(n+1.5) )`, putting `(2a)^(n+1.5)` in the
  DENOMINATOR (so the norm shrank as the exponent grew — the opposite of correct) and using
  the wrong `(2n-1)!!` term. All 15 vendor `(l,a)` cases mismatched (e.g. `(0,0.5)`
  cintx=1.331 vs vendor=1.502; `(4,2.5)` cintx≈0.16 vs vendor≈16.34). New body uses the
  documented closed form from `misc.c` (Schlegel & Frisch, IJQC 54(1995) 83-87):

      norm = sqrt( 2^(2n+3) * (n+1)! * (2a)^(n+1.5) / ((2n+2)! * sqrt(pi)) )

  Factorials are exact f64 products (the oracle grid tests n in 0..5; f64 holds these integer
  factorials exactly up to ~n<=8). The defensive guard
  `if !a.is_finite() || a <= 0.0 || n < 0 { return 0.0; }` is preserved verbatim.

- **`crates/cintx-compat/src/helpers.rs` (tests)** — strengthened the lib unit test: pins the
  exact values `CINTgto_norm(0,0.5) == 1.502251088929885` (atol 1e-12) and
  `CINTgto_norm(4,2.5) == 16.34007804382598`, asserts the norm GROWS with `a`
  (`norm(2,2.5) > norm(2,0.5)`), and asserts the guard cases
  (`CINTgto_norm(-1,0.5) == 0.0`, `CINTgto_norm(1,0.0) == 0.0`).

- **`crates/cintx-oracle/tests/cintgto_norm_parity.rs` (new)** — a double-gated vendor parity
  test modeled on `one_electron_scalar_spinor_parity.rs`. `cintgto_norm_matches_vendor`
  (`#[cfg(has_vendor_libcint)] #[cfg(feature = "cpu")]`) loops `n in 0..5`,
  `a in {0.5,1.0,2.5,0.75,5.0}`, compares `cintx_compat::helpers::CINTgto_norm` against
  `cintx_oracle::vendor_ffi::vendor_CINTgto_norm`, accumulates a `mismatches` counter at
  `diff > 1e-12`, and asserts 0. A non-vendor `cintgto_norm_smoke` (`#[cfg(feature = "cpu")]`)
  pins the 1.502251088929885 value so the file is not a no-op without the vendor build.

## Decision: Option (a) over Option (b)

Used the documented closed form (option a) rather than `1.0 / gaussian_int(2n+2, 2a).sqrt()`
with an `lgamma` (option b). Rationale: no new dependency (no `libm`, no Cargo.toml change);
exact for integer `n` over the entire tested range; byte-stable. Verified in Python that the
closed form matches the `_gaussian_int`-based libcint formula to < 1e-12 across the full
`(n,a)` grid before implementing. The plan's note value `1.5022113490956049` was slightly
off; the verified closed-form value `1.502251088929885` is the one pinned.

## Note on the pinned expected value

The plan suggested `CINTgto_norm(0,0.5) ≈ 1.5022113490956049`. Both the closed form and the
`_gaussian_int`-based libcint reference produce `1.502251088929885`, which is what the vendor
FFI returns (the parity test passes at atol 1e-12). The verified value was used.

## Verification

1. `cargo test -p cintx-compat --lib` — RED first (current impl: `CINTgto_norm(0,0.5) =
   1.3313353638003897 expected 1.502251088929885`), then GREEN after the fix:
   `test result: ok. 40 passed; 0 failed`.

2. Vendor parity, both gates:
   `CINTX_BACKEND=cpu CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --locked --test cintgto_norm_parity`
   →
   ```
   running 2 tests
   test cintgto_norm_smoke ... ok
   test cintgto_norm_matches_vendor ... ok

   test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
   ```

3. **FINAL MANDATORY full vendor oracle gate (verbatim):**
   ```
   CINTX_BACKEND=cpu CINTX_ORACLE_BUILD_VENDOR=1 cargo run --locked --manifest-path xtask/Cargo.toml -- oracle-compare --profiles "base,with-f12,with-4c1e,with-f12+with-4c1e" --include-unstable-source false
   ```
   The gate's `verify_helper_surface_coverage` runs the helper parity checks in sequence:
   CINTlen_* → CINTcgto* → CINTtot_* → CINTshells_*_offset → **CINTgto_norm** →
   **CINTc2s_bra_sph** → ... . The CINTgto_norm loop (compare.rs lines 711-723, grid
   `l in 0..5`, `a in {0.5,1.0,2.5}`) now passes with no mismatch and execution advances to
   the next check.

   **CINTgto_norm mismatch: GONE.** No `CINTgto_norm(...)` mismatch appears anywhere in the
   gate output.

   **NEXT blocker (verbatim, captured via `helper_coverage_matches_manifest` under the vendor
   gate — the xtask wrapper swallows the inner bail and surfaces it only as the
   downstream "artifact source missing"):**
   ```
   helper parity: CINTc2s_bra_sph(l=0) elem 0 mismatch: cintx=0.1 vendor=0 diff=0.1
   ```

   The xtask gate's terminal line (because the helper bail at `CINTc2s_bra_sph` prevents the
   matrix artifact from being written — gate is fail-closed):
   ```
   xtask gate failed: resolve matrix artifact source path: artifact source missing (required: `/tmp/cintx_artifacts/cintx_phase_04_manifest_representation_matrix.json`, fallback: `/tmp/cintx_artifacts/cintx_phase_04_manifest_representation_matrix.json`)
   ```

   `CINTc2s_bra_sph` is a SEPARATE, pre-existing helper mismatch (the cintx transform writes
   the input cart values straight through, `cintx=0.1`, while vendor produces `0`). Per the
   task constraints it is left UNFIXED — it is the next follow-up.

## Deviations from Plan

### Auto-fixed Issues (Rule 3 - blocking environment)

**1. [Rule 3 - Blocking] Created missing `/tmp/cintx_artifacts` directory**
- **Found during:** First full-gate run.
- **Issue:** The xtask `oracle-compare` artifact pipeline writes to `/tmp/cintx_artifacts`,
  which did not exist in this worktree environment.
- **Fix:** `mkdir -p /tmp/cintx_artifacts` (environment only — no code or repo change).
- **Files modified:** none.
- **Note:** This did not change the gate outcome (the gate still fails-closed on the
  pre-existing `CINTc2s_bra_sph` helper bail, which runs before the matrix write), but it
  rules out the directory as the cause of the "artifact source missing" message.

### Expected-value correction
The pinned `CINTgto_norm(0,0.5)` constant is `1.502251088929885` (verified against both the
closed form and the libcint `_gaussian_int` reference and confirmed against the vendor FFI),
not the plan's approximate `1.5022113490956049`. Not a deviation in intent — the plan
explicitly instructed to verify the exact value at implementation time.

## Scope Boundary Honored

- No changes to `crates/cintx-oracle/src/compare.rs`.
- No `Cargo.toml` change (option (a) chosen).
- `CINTc2s_bra_sph` and all other downstream blockers left unfixed.
- Guard preserved; no panic / NaN / Inf on invalid input.

## Commits (code only)

- `1739615` — `test(260529-mfh): pin libcint-correct CINTgto_norm expected values`
- `8db9fcb` — `fix(260529-mfh): CINTgto_norm matches libcint misc.c closed form`

## Self-Check: PASSED

- FOUND: crates/cintx-compat/src/helpers.rs
- FOUND: crates/cintx-oracle/tests/cintgto_norm_parity.rs
- FOUND commit: 1739615
- FOUND commit: 8db9fcb
