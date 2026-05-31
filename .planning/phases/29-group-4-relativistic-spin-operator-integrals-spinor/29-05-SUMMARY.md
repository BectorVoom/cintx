---
phase: 29-group-4-relativistic-spin-operator-integrals-spinor
plan: 05
subsystem: oracle-parity
tags: [spinor, 2e, rel-03, rel-04, gaunt1, dkb, vendor-build, manifest, red-scaffold, wave-3, relativistic, blocking]

# Dependency graph
requires:
  - phase: 29 (plan 04)
    provides: int2e_spsp1_spinor manifest row + vendor_int2e_spsp1_spinor shim (clone templates) + the D-03-proven 2e si/sf transform suite + build_kappa_spinor_2e_fixture
  - phase: 29 (plan 03)
    provides: the 6-fn 2e cart→spinor si/sf transform suite + build_kappa_spinor_2e_fixture (D-02)
provides:
  - gaunt1.c + dkb.c wired into the oracle vendor build (REL-04 BLOCKING enablement) + 15 int2e_*_spinor symbols on the bindgen allowlist
  - 15 remaining 2e Group-4 manifest rows (REL-03 srsr1/spsp1spsp2/srsr1srsr2 + REL-04 ssp1ssp2/ssp1sps2/sps1ssp2/sps1sps2/spv1/vsp1/spv1spv2/vsp1spv2/spv1vsp2/vsp1vsp2/spv1spsp2/vsp1spsp2), spinor-only, component_rank 1, oracle_covered false
  - 15 vendor_int2e_*_spinor FFI shims linking the real libcint 6.1.3 2e drivers under the double gate
  - rel_2e_sigma_parity.rs — the REL-03/04 parity scaffold (3 always-on tests green; 15 byte-identity gates #[ignore]'d until 29-06)
affects: [29-06, 29-group-4-wave-3, 30-giao-sigma, 31-breit-gaunt]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "REL-04 vendor enablement = add gaunt1.c (.file) + dkb.c (.file) to crates/cintx-oracle/build.rs immediately after intor4.c (same include set, no suppl-header — all symbols in cint_funcs.h) + append the int2e_*_spinor symbol names to the allowlist_function regex; the no-silent-skip vendor sweep then PROVES each REL-04 driver linked"
    - "RED parity scaffold = real vendor collectors (symbols link this plan) driven by a name→fn match table + a panic-stub cintx collector (launchers land in 29-06) + always-on assertions (kappa-sizing, all-rows-registered spinor-only/rank-1/oc-false, no-silent-skip vendor-arm-runs) that pass NOW + per-family byte-identity gates behind a macro, each #[ignore]'d with a 29-06 TODO so the scaffold COMPILES and the deferred gates are discoverable"

key-files:
  created:
    - crates/cintx-oracle/tests/rel_2e_sigma_parity.rs
  modified:
    - crates/cintx-oracle/build.rs
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-ops/src/generated/api_manifest.csv
    - crates/cintx-oracle/src/vendor_ffi.rs

key-decisions:
  - "Trusted RESEARCH over CONTEXT.md: CONTEXT.md claimed 'intor4.c already wired → no build change'. RESEARCH Pitfall 1 (grep-confirmed this session) showed REL-04's ssp/sps live in gaunt1.c and vsp/spv in dkb.c — NEITHER in build.rs. Added both .file() entries; CONTEXT.md's claim is correct ONLY for REL-03 (intor4.c). Without gaunt1.c/dkb.c the 8 REL-04 vendor shims have no symbol to link (undefined-symbol error)."
  - "Inserted the 15 manifest rows immediately AFTER the int2e_spsp1_spinor row (lock index 348) → new indices ≥349, far past every hardcoded OperatorId const (highest is int2e_stg_sph at 106 / int4c1e_cart at 24). No positional OperatorId drift; cintx-ops resolver tests stay 13/13 green; api_manifest.rs/.csv regenerate from the lock via build.rs."
  - "All 15 rows component_rank '1' — the σ_x/σ_y/σ_z fold is internal to the c2s_si/sf_2e transform pair, NOT an output component axis (RESEARCH item-4; a wrong rank>1 mis-strides the interleaved output). Verified the existing int1e_sp (rank 1, oc=true) and int2e_spsp1 (rank 1, oc=false) rows stay correct."
  - "rel_2e_sigma_parity.rs is a RED scaffold: the cintx-side family launchers land in 29-06 (all 2e spinor currently returns UnsupportedApi), so the cintx collector is a loud panic-stub and the 15 per-family byte-identity gates are #[ignore]'d via a macro with a 29-06 TODO. The ALWAYS-ON no-silent-skip test runs every vendor arm NON-SKIPPED under the double gate and asserts nonzero output — this is the live proof that the gaunt1.c/dkb.c build wiring actually linked a real driver for each of the 15 families (including the 12 REL-04 families that had NO vendor symbol before this plan)."

patterns-established:
  - "Pattern: a Wave-3 family-batch foundation plan = (BLOCKING vendor build enablement: add the source .file()s + allowlist symbols) + (batch-register the manifest rows past the OperatorId-const watermark) + (clone the vendor shim N times via a generator) + (a RED parity scaffold whose always-on no-silent-skip sweep validates the build enablement while the byte-identity gates wait #[ignore]'d for the launcher plan)."

requirements-completed: []

# Metrics
duration: 7min
completed: 2026-06-01
---

# Phase 29 Plan 05: Wave-3 2e σ Foundation (REL-03/04 enablement, BLOCKING build fix) Summary

**Laid the Wave-3 foundation for the remaining 16 2e Group-4 σ families. The load-bearing deliverable is the [BLOCKING] `build.rs` fix: `gaunt1.c` (ssp/sps drivers) and `dkb.c` (vsp/spv drivers) are now compiled into the oracle vendor build — neither was wired before, so REL-04 had literally no vendor symbol to link against (RESEARCH Pitfall 1, correcting CONTEXT.md's "no build change" claim). On that foundation: all 15 remaining 2e manifest rows are registered spinor-only / component_rank=1 / oracle_covered=false (no OperatorId drift), 15 `vendor_int2e_*_spinor` shims clone the real libcint 6.1.3 drivers, and `rel_2e_sigma_parity.rs` is a RED scaffold whose ALWAYS-ON no-silent-skip sweep PROVES every one of the 15 vendor arms — including the 12 REL-04 families that had no reference before — links and produces nonzero output under the double gate. The 15 per-family byte-identity gates are `#[ignore]`'d until 29-06 wires the cintx launchers.**

## Performance

- **Duration:** ~7 min
- **Completed:** 2026-06-01
- **Tasks:** 3
- **Files:** 6 (1 created, 5 modified; +1615 / -1 lines, of which the regenerated manifest .rs/.csv dominate)

## Accomplishments

- **Task 1 — [BLOCKING] gaunt1.c + dkb.c into the vendor build:** Added `.file(src/autocode/gaunt1.c)` (provides `int2e_ssp1ssp2/ssp1sps2/sps1ssp2/sps1sps2_spinor`) and `.file(src/autocode/dkb.c)` (provides `int2e_spv1/vsp1/spv1spv2/vsp1spv2/spv1vsp2/vsp1vsp2/spv1spsp2/vsp1spsp2_spinor`) to `crates/cintx-oracle/build.rs` immediately after `intor4.c` — both use the identical include set + cc flags (verified: all symbols in `cint_funcs.h`, no suppl-header). Appended all 15 remaining `int2e_*_spinor` symbol names (REL-03 srsr1/spsp1spsp2/srsr1srsr2 + 12 REL-04) to the `allowlist_function` regex so bindgen emits them. Verified all 15 vendor symbols exist in libcint 6.1.3 source (grep-confirmed gaunt1.c:105/203/301/399, dkb.c:243/.../1033, intor4.c:343/271/535). `CINTX_ORACLE_BUILD_VENDOR=1 cargo build -p cintx-oracle --features cpu` exits 0 — gaunt1.c + dkb.c compile and link. **This is the REL-04 enablement gate.**
- **Task 2 — 15 manifest rows registered:** Added 15 `ManifestEntry` rows to `compiled_manifest.lock.json` immediately after the `int2e_spsp1_spinor` row (each: arity 4, canonical_family/category "2e", `component_rank "1"`, `forms ["spinor"]`, `complex_output true`, `oracle_covered false`, stability "stable"). `api_manifest.rs`/`.csv` regenerate from the lock via `cintx-ops/build.rs`. The rows land at lock indices ≥349 — past every hardcoded `OperatorId::new(N)` const (highest is 106 `int2e_stg_sph` / 24 `int4c1e_cart`), so NO positional OperatorId drift: `cargo build --workspace --features cpu` exits 0 and `cargo test -p cintx-ops` = 13/13 (including the `OperatorId::new(24)` int4c1e_cart preservation invariant). Confirmed the existing `int1e_sp` (rank 1, oc=true) and `int2e_spsp1` (rank 1, oc=false) rows stay correct.
- **Task 3 — 15 vendor shims + RED parity scaffold:** Cloned `vendor_int2e_spsp1_spinor` 15× in `vendor_ffi.rs` (each: `shls &[i32;4]`, `out` sized `ni*nj*nk*nl*2` via `vendor_CINTcgto_spinor`, only the `ffi::int2e_X_spinor` symbol swapped) via a generator to avoid transcription drift. Created `crates/cintx-oracle/tests/rel_2e_sigma_parity.rs` (cloned from `si_2e_transform_parity.rs`): a `REL_2E_FAMILIES` table drives a name→fn vendor collector; the cintx collector is a loud panic-stub (launchers land in 29-06). THREE ALWAYS-ON tests pass NOW — `test_kappa_sizing_2e_non_4l_plus_2` (GT/LT sizing on the fixture momenta), `test_all_rel_2e_rows_registered` (all 15 rows present, spinor-only/rank-1/oc-false), and `test_no_silent_skip` (under the double gate every vendor arm executes NON-SKIPPED and produces nonzero output — the live proof the gaunt1.c/dkb.c build wiring linked a real driver per family). The 15 per-family byte-identity gates are emitted by a `rel_2e_byte_identity_gate!` macro, each `#[ignore]`'d with a 29-06 TODO so the scaffold COMPILES and the deferred work is discoverable.

## Task Commits

1. **Task 1: [BLOCKING] wire gaunt1.c + dkb.c** — `2541b00` (feat)
2. **Task 2: register 15 remaining 2e Group-4 spinor manifest rows** — `d16bbbb` (feat)
3. **Task 3: 15 vendor shims + rel_2e_sigma_parity RED scaffold** — `c8dc79f` (test)

## Files Created/Modified

- `crates/cintx-oracle/tests/rel_2e_sigma_parity.rs` — **created**; the REL-03/04 RED parity scaffold.
- `crates/cintx-oracle/build.rs` — gaunt1.c + dkb.c `.file()` entries + 15 symbols on the allowlist regex.
- `crates/cintx-ops/generated/compiled_manifest.lock.json` (+ regenerated `src/generated/api_manifest.rs` / `.csv`) — 15 new 2e spinor rows.
- `crates/cintx-oracle/src/vendor_ffi.rs` — 15 `vendor_int2e_*_spinor` shims.

## Decisions Made

- **Trusted RESEARCH over CONTEXT.md on the build change.** CONTEXT.md's "intor4.c already wired → no build change" is correct ONLY for REL-03. RESEARCH Pitfall 1 (re-grep-confirmed this session) located REL-04's ssp/sps in gaunt1.c and vsp/spv in dkb.c, neither in build.rs. This is the difference between REL-04 having a vendor reference and a hard undefined-symbol link error.
- **Rows inserted past the OperatorId-const watermark.** Appending after the index-348 spsp1 row keeps every new index ≥349, past the highest hardcoded const (106). Confirmed by the green workspace build + 13/13 resolver tests (the const-shift landmine would have surfaced as an InvalidShellTuple arity mismatch). The lock is the single source of truth; both manifest-audit sides auto-sync.
- **component_rank "1" for all 15.** The σ fold is internal to the c2s transform (RESEARCH item-4); a wrong rank>1 mis-strides the interleaved output. Measured against the rank-tier convention, not assumed.
- **RED scaffold with a live no-silent-skip sweep.** Rather than leave the build enablement unverified until 29-06, the always-on `test_no_silent_skip` runs all 15 vendor arms now. A driver that failed to link (a missing gaunt1.c/dkb.c .file() entry) would FAIL this test (all-zero output), not skip — so Task 1's BLOCKING fix is continuously proven, while the cintx-vs-vendor byte-identity gates wait `#[ignore]`'d for the 29-06 launcher wiring.

## Deviations from Plan

None requiring a Rule-4 stop. The plan was executed as written. One scaffold-structure clarification (not a deviation): the plan's Task-3 text described the byte-identity gate "PRIMARY GATE … on build_kappa_spinor_2e_fixture" as if active; because the cintx launchers are explicitly a 29-06 deliverable (all 2e spinor returns UnsupportedApi today), those per-family gates are necessarily `#[ignore]`'d (RED) this plan — exactly what the plan's `<objective>` ("RED until 29-06 wires launchers") and `must_haves.truths` ("RED — families wired in 29-06") require. The cintx side is a loud panic-stub so the scaffold compiles without a fake/zero-fill cintx path that could mask a real failure later.

## Issues Encountered / Deferred

- **Pre-existing uncommitted `two_electron.rs` hunk (NOT mine, NOT staged), logged to `deferred-items.md` §29-05:** `git diff` during final verification revealed `crates/cintx-cubecl/src/kernels/two_electron.rs` carries an uncommitted working-tree definition of `pub fn int2e_common_factor` that is NOT committed at HEAD — yet the committed 29-04 test `si_2e_transform_parity.rs` references it. This is a 29-04 commit-hygiene gap (the function landed in the dirty working tree, part of the parked WIP the executor was instructed not to stage, but was omitted from commit `fece759`). A clean checkout of HEAD would fail to compile the 29-04 test. **Out of scope for 29-05** (not caused by, not owned by this plan; my deliverables do NOT depend on `int2e_common_factor` — the new scaffold stubs the cintx side). All 29-05 verification ran green in the current working tree. Whoever finalizes the 29-04/29-06 WIP must commit this hunk.
- **Pre-existing `cintx-oracle` lib-test failures** (`compare::tests::*` 158 mismatches; `fixtures::tests::unstable_source_fixtures_require_opt_in`) persist from 29-03/29-04 — already logged, verified pre-existing, out of scope (SCOPE BOUNDARY).

## Known Stubs

The cintx-side family collector in `rel_2e_sigma_parity.rs` (`collect_cintx_family`) is an **intentional RED panic-stub** — this is the planned Wave-3 foundation state, NOT an incomplete deliverable. The plan's `<objective>` and `must_haves.truths` both specify the scaffold is "RED until 29-06 wires launchers". The 15 per-family byte-identity gates that would call it are `#[ignore]`'d with explicit 29-06 TODOs; the families stay `oracle_covered=false` so the no-silent-skip test asserts they are NOT prematurely claimed covered. 29-06 wires each launcher arm (per the RESEARCH §2e transform map: si_2e1+sf_2e2 / si_2e1+si_2e2 / si_2e1i+si_2e2i), drops the stub + `#[ignore]`, and flips `oracle_covered=true` once green. The build.rs wiring, manifest rows, and vendor shims are all fully implemented and exercised (the no-silent-skip sweep runs every vendor arm).

## Threat Flags

None new. The two registered threats are addressed:
- **T-29-09 (Tampering, adding C sources to the vendor build, `accept`):** gaunt1.c/dkb.c are frozen libcint 6.1.3 sources compiled only under the test-gated `CINTX_ORACLE_BUILD_VENDOR=1` flag — not part of the shipped library surface.
- **T-29-10 (Spoofing, oracle_covered without a real linked symbol, `mitigate`):** Task 1 ensures the 15 symbols link; the always-on `test_no_silent_skip` asserts each vendor arm executed and produced nonzero output (FAIL, not skip); all 15 rows stay `oracle_covered=false` until 29-06.

No new network/auth/file-access surface (a vendor cc build change + manifest rows + FFI shims + an oracle test).

## Next Phase Readiness

- **Wave-3 foundation delivered — 29-06 unblocked.** The REL-04 BLOCKING build fix is in (gaunt1.c/dkb.c linked, proven by the live no-silent-skip sweep), all 15 rows are registered, all 15 vendor shims link, and the parity scaffold compiles with discoverable `#[ignore]`'d gates.
- **29-06 (next):** for each of the 15 families, wire the launcher arm in `launch_two_electron_typed` (transform pair per RESEARCH §2e map), replace `collect_cintx_family`'s panic-stub with the real launch (mirror `collect_cintx_spsp1_spinor`), drop `#[ignore]` from each gate, and flip `oracle_covered=true` once byte-identity is green. Also fold in the parked `int2e_common_factor` commit so the existing 29-04 test builds on a clean checkout.
- No blockers introduced by this plan.

## Self-Check: PASSED

- Created file exists on disk: `crates/cintx-oracle/tests/rel_2e_sigma_parity.rs` (FOUND).
- All 3 task commits present in git history: `2541b00`, `d16bbbb`, `c8dc79f` (FOUND).
- `gaunt1.c` + `dkb.c` grep-confirmed in build.rs; 15 `vendor_int2e_*_spinor` shims + 15 manifest rows grep-confirmed present.
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo build -p cintx-oracle --features cpu` exits 0 (gaunt1.c + dkb.c compile + link).
- `cargo build --workspace --features cpu` exits 0 (no OperatorId drift); `cargo test -p cintx-ops` = 13/13.
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test rel_2e_sigma_parity` = 3 passed / 15 ignored / 0 failed (no-silent-skip ran every vendor arm non-skipped); non-vendor build = 3 passed / 0 failed.

---
*Phase: 29-group-4-relativistic-spin-operator-integrals-spinor*
*Completed: 2026-06-01*
