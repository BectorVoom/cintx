---
phase: quick-260529-lbr
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/cintx-compat/src/helpers.rs
autonomous: true
requirements: [QUICK-260529-lbr]
tags: [compat, oracle, libcint-parity, ao-loc]

must_haves:
  truths:
    - "CINTshells_{cart,spheric,spinor}_offset write exactly nbas entries (shell START offsets), never ao_loc[nbas]"
    - "The cintx helper output is byte-identical to libcint shells_cgto_offset for the oracle's 4-shell bas across all 4 profiles"
    - "The vendor oracle helper-parity check no longer bails on CINTshells_cart_offset[4] cintx=8 vendor=0"
    - "The helper unit test asserts the libcint-matching output [0, 1, <untouched>] and passes"
  artifacts:
    - path: "crates/cintx-compat/src/helpers.rs"
      provides: "write_offsets matching libcint i<nbas semantics + updated unit test"
      contains: "for i in 1..nbas"
  key_links:
    - from: "crates/cintx-compat/src/helpers.rs::write_offsets"
      to: "libcint-master/src/cint_bas.c::shells_cgto_offset"
      via: "i<nbas loop, ao_loc[0]=0, ao_loc[i]=ao_loc[i-1]+count(i-1)"
      pattern: "for i in 1\\.\\.nbas"
    - from: "crates/cintx-oracle/src/compare.rs helper-parity block"
      to: "CINTshells_*_offset"
      via: "byte comparison of nbas+1 zero-inited buffer"
      pattern: "CINTshells_cart_offset"
---

<objective>
Fix `write_offsets` in `crates/cintx-compat/src/helpers.rs` so `CINTshells_{cart,spheric,spinor}_offset` replicate libcint 6.1.3's `shells_cgto_offset` exactly: write `ao_loc[0]=0` then `ao_loc[i] = ao_loc[i-1] + count(i-1)` for `i` in `1..nbas`, writing EXACTLY `nbas` entries and NEVER touching `ao_loc[nbas]`.

Purpose: The cintx helper currently writes `nbas+1` entries (it appends the grand total at `ao_loc[nbas]`). libcint leaves that trailing slot untouched. The oracle helper-parity check (compare.rs) zero-inits an `nbas+1` buffer and compares every entry, so it bails on `CINTshells_cart_offset[4] cintx=8 vendor=0` for all 4 profiles BEFORE any numeric integral parity runs. Matching libcint exactly unblocks the gate.

Output: A byte-faithful `write_offsets`, an updated unit test, and a verified full vendor oracle gate run that reaches numeric integral parity for the first time on this branch.
</objective>

<execution_context>
@/home/user/Documents/workspace/cintx/.claude/get-shit-done/workflows/execute-plan.md
@/home/user/Documents/workspace/cintx/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@./CLAUDE.md

<interfaces>
Current `write_offsets` (crates/cintx-compat/src/helpers.rs ~185-210) — WRONG, writes nbas+1:
```rust
fn write_offsets(
    ao_loc: &mut [i32],
    bas: &[i32],
    nbas: i32,
    count_fn: fn(i32, &[i32]) -> Result<usize, cintxRsError>,
) -> Result<(), cintxRsError> {
    let shell_count = shell_count(nbas, bas)?;
    let needed = shell_count.saturating_add(1);   // <-- nbas+1, WRONG
    if ao_loc.len() < needed { return Err(cintxRsError::BufferTooSmall { required: needed, provided: ao_loc.len() }); }
    let mut offset = 0usize;
    ao_loc[0] = 0;
    for shell in 0..shell_count {
        offset = offset.saturating_add(count_fn(shell as i32, bas)?);
        ao_loc[shell + 1] = i32::try_from(offset).map_err(|_| cintxRsError::ChunkPlanFailed {
            from: "compat_helpers",
            detail: "ao offset overflowed i32".to_owned(),
        })?;                                        // <-- writes ao_loc[nbas], WRONG
    }
    Ok(())
}
```

Ground truth — libcint-master/src/cint_bas.c::shells_cgto_offset (~129-137):
```c
static void shells_cgto_offset(FINT (*f)(), FINT ao_loc[], const FINT *bas, const FINT nbas) {
    FINT i;
    ao_loc[0] = 0;
    for (i = 1; i < nbas; i++) {          // i < nbas: writes indices 0..nbas-1 ONLY
        ao_loc[i] = ao_loc[i-1] + (*f)(i-1, bas);
    }
}
```

`shell_count(nbas, bas)` (helpers.rs ~68) returns `usize` == nbas (validated against bas layout), erroring on bad layout. Use it to get the validated count.

`cintxRsError` variants in use: `BufferTooSmall { required, provided }`, `ChunkPlanFailed { from, detail }`. Keep both. No panics, no unwrap.

Oracle caller (crates/cintx-oracle/src/compare.rs ~540-543): `let mut offsets = vec![0_i32; 5]; CINTshells_cart_offset(&mut offsets, &inputs.bas, 4)?; ...` — len 5 >= nbas 4, so reducing the required length is safe; no change needed there.

Unit test (helpers.rs ~253-259) currently asserts `vec![0, 1, 7]` for a len-3 buffer with `sample_bas()` (2 shells: l=0 cgto=1, l=2 cgto=6). With i<nbas semantics: ao_loc[0]=0, ao_loc[1]=0+cgto(0)=1, ao_loc[2] (==ao_loc[nbas]) UNTOUCHED at its zero init → `[0, 1, 0]`. The grand total (7) is reachable via `CINTtot_cgto_cart(&bas, 2)`.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Update helper unit test to libcint i<nbas expectation (RED)</name>
  <files>crates/cintx-compat/src/helpers.rs</files>
  <behavior>
    - With sample_bas() (2 shells: l=0 → cgto 1, l=2 → cgto 6), nbas=2, len-3 buffer pre-zeroed:
      libcint-matching output is [0, 1, 0] — ao_loc[0]=0, ao_loc[1]=0+cgto(0)=1, ao_loc[2]==ao_loc[nbas] left UNTOUCHED (stays 0).
    - The grand total (7) is NOT in ao_loc; it stays reachable via CINTtot_cgto_cart(&bas, 2) == 7.
    - This test MUST FAIL against the current code (which produces [0, 1, 7]) — confirming RED.
  </behavior>
  <action>
    In the `#[cfg(test)] mod tests` block (~242-259), update `helper_offsets_write_prefix_sums`:
    1. Rename it to `helper_offsets_match_libcint_i_lt_nbas` (the helper writes shell START offsets per libcint's `i < nbas` loop, not inclusive prefix sums — the old name no longer fits).
    2. Change the assertion from `assert_eq!(offsets, vec![0, 1, 7]);` to `assert_eq!(offsets, vec![0, 1, 0]);` with an inline comment: `// libcint cint_bas.c shells_cgto_offset uses i<nbas: writes ao_loc[0..nbas-1] only; ao_loc[nbas] (index 2) stays at its zero init. Total (7) lives in CINTtot_cgto_cart.`
    3. Add a documenting assertion that the total moved: `assert_eq!(CINTtot_cgto_cart(&bas, 2).unwrap(), 7);`
    Do NOT touch `write_offsets` yet — this task only changes the test so it goes RED against current code.
    Per the root_cause decision: match libcint exactly, do not relax the harness. Per CLAUDE.md, keep typed errors (thiserror), no panics.
  </action>
  <verify>
    <automated>cargo test -p cintx-compat --lib helper_offsets_match_libcint_i_lt_nbas 2>&1 | grep -qE "FAILED|assertion .*failed|panicked" && echo RED_CONFIRMED || (echo "expected RED but test did not fail" && exit 1)</automated>
  </verify>
  <done>The renamed test exists, asserts `[0, 1, 0]` plus the CINTtot_cgto_cart==7 check, and FAILS against the current (unchanged) write_offsets — confirming RED before the fix.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Rewrite write_offsets to libcint i<nbas semantics (GREEN) + full vendor oracle gate</name>
  <files>crates/cintx-compat/src/helpers.rs</files>
  <behavior>
    - write_offsets writes EXACTLY nbas entries: ao_loc[0]=0, then ao_loc[i]=ao_loc[i-1]+count(i-1) for i in 1..nbas. ao_loc[nbas] is NEVER written.
    - Required buffer length is now `nbas` (was nbas+1); BufferTooSmall fires when ao_loc.len() < nbas.
    - nbas==0: write nothing (libcint's unconditional ao_loc[0]=0 is OOB for nbas==0; cintx guards the index-0 write behind nbas>=1 and never panics; required length 0). Oracle never calls nbas==0 but it stays sound/typed.
    - All three CINTshells_{cart,spheric,spinor}_offset are fixed by this single change.
    - The Task 1 test now passes (GREEN): output [0, 1, 0].
  </behavior>
  <action>
    Rewrite `write_offsets` (~185-210) to replicate libcint `shells_cgto_offset` exactly:
    1. `let count = shell_count(nbas, bas)?;` (validated nbas).
    2. Required length is now `count` (NOT `count + 1`). Keep the BufferTooSmall guard: `if ao_loc.len() < count { return Err(cintxRsError::BufferTooSmall { required: count, provided: ao_loc.len() }); }`.
    3. If `count == 0`, return `Ok(())` immediately (no index-0 write — mirrors that libcint's `ao_loc[0]=0` is only valid for nbas>=1; never panic).
    4. Otherwise: `ao_loc[0] = 0;` then loop `for i in 1..count` accumulating in i32 to match libcint's running sum: `ao_loc[i] = ao_loc[i-1].checked_add(count_fn((i-1) as i32, bas)? as i32)...` — preserve the i32 overflow guard via `i32::try_from`/`checked_add` returning `cintxRsError::ChunkPlanFailed { from: "compat_helpers", detail: "ao offset overflowed i32".to_owned() }` (NO panic, NO silent wrap). Add a comment: `// libcint cint_bas.c shells_cgto_offset: i<nbas, writes ao_loc[0..nbas-1]; ao_loc[nbas] left untouched.`
    Do NOT modify the 3 wrapper fns (they already delegate). Do NOT modify compare.rs (its len-5/nbas-4 buffer is still large enough; the harness is correct). Per root_cause/constraints: byte-faithful to libcint, do not relax the oracle harness.
  </action>
  <verify>
    <automated>cargo test -p cintx-compat --lib 2>&1 | tail -20 | grep -qE "test result: ok" && echo COMPAT_LIB_GREEN || (echo "cintx-compat --lib tests not green" && exit 1)</automated>
  </verify>
  <done>
    - `cargo test -p cintx-compat --lib` is fully green (helper_offsets_match_libcint_i_lt_nbas and all others pass).
    - write_offsets writes nbas entries, never ao_loc[nbas], typed errors preserved, no panics.
    - MANDATORY FINAL STEP (run after the unit tests pass): execute the full vendor oracle gate verbatim:
        `CINTX_BACKEND=cpu CINTX_ORACLE_BUILD_VENDOR=1 cargo run --locked --manifest-path xtask/Cargo.toml -- oracle-compare --profiles "base,with-f12,with-4c1e,with-f12+with-4c1e" --include-unstable-source false`
      The vendor libcint build is slow (several minutes) — allow generous time (use a long Bash timeout, e.g. up to 600000ms, or run in background and monitor).
      Report HONESTLY:
        (a) Confirm the `CINTshells_*_offset` helper-parity mismatch is GONE for all 4 profiles.
        (b) Report what the gate does NEXT — it now reaches the numeric INTEGRAL parity comparison for the FIRST time on this branch.
        (c) If all 4 profiles pass clean → report PASS.
        (d) If FURTHER pre-existing mismatches surface downstream (the helper bail previously masked everything after it) → report them VERBATIM, do NOT fabricate a pass, and do NOT attempt to fix unrelated newly-surfaced issues in this task (note them for follow-up).
    - Commit CODE atomically (helpers.rs), NOT docs. Suggested message:
        `fix(compat): CINTshells_*_offset match libcint i<nbas (drop trailing ao_loc[nbas])`
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| caller → compat helper | `ao_loc`/`bas` slices and `nbas` count come from the caller (raw API / oracle); lengths and nbas must be validated, never assumed. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-lbr-01 | Denial of Service (panic) | `write_offsets` index/arithmetic | mitigate | Guard `ao_loc[0]` write behind `count>=1`; BufferTooSmall before any indexed write; i32 overflow via checked/try_from → typed `ChunkPlanFailed`. No unwrap, no slice OOB, no integer wrap. |
| T-lbr-02 | Tampering (silent wrong result) | trailing `ao_loc[nbas]` semantics | mitigate | Match libcint byte-for-byte (never write ao_loc[nbas]); the vendor oracle gate is the cross-check that detects any divergence. Do not relax the harness. |
| T-lbr-03 | Information disclosure | reading bas beyond layout | accept (already mitigated) | `shell_count`/`RawBasView::new` already validate nbas against bas layout and error on overrun; unchanged by this fix. |
</threat_model>

<verification>
1. `cargo test -p cintx-compat --lib` — all green; `helper_offsets_match_libcint_i_lt_nbas` asserts `[0, 1, 0]`.
2. Full vendor oracle gate (verbatim command in Task 2 done) — `CINTshells_*_offset` mismatch gone for all 4 profiles; gate proceeds to numeric integral parity. Report outcome honestly (clean pass OR verbatim downstream mismatches noted for follow-up).
3. No change to compare.rs; the harness stays correct.
</verification>

<success_criteria>
- write_offsets is byte-faithful to libcint `shells_cgto_offset` (`i < nbas`, never writes `ao_loc[nbas]`); required length is `nbas`.
- Typed errors preserved (BufferTooSmall with corrected length, ChunkPlanFailed on i32 overflow); no panics; nbas==0 handled soundly.
- `cargo test -p cintx-compat --lib` green.
- Full vendor oracle gate run: helper-parity mismatch for CINTshells_*_offset is gone across base / with-f12 / with-4c1e / with-f12+with-4c1e; gate reaches numeric integral parity; outcome reported honestly (no fabricated pass).
- Code committed atomically (helpers.rs only), docs NOT in the code commit.
</success_criteria>

<output>
After completion, create `.planning/quick/260529-lbr-fix-cintshells-cart-spheric-spinor-offse/260529-lbr-SUMMARY.md`
</output>
