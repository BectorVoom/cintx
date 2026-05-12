---
phase: 19-int1e-ecp-type1-type2-evaluator
plan: 01
subsystem: vendor-integration
tags: [ecp, pyscf, libcint, manifest, cubecl, oracle, scaffold]

requires:
  - phase: 18-sessionrequest-arity-ge3-dispatch
    provides: "SessionRequest dispatch infrastructure (manifest-driven resolver, kernel launcher pattern, parity-test conventions)"
  - phase: 15-oracle-tolerance-unification-manifest-lock-closure
    provides: "Four-profile manifest lock + atol=1e-12 unified oracle tolerance baseline"
  - phase: 13-f12-stg-yp-kernels
    provides: "canonical_family != family_name precedent (F12 routes through f12 launcher while family_name stays 2e)"
  - phase: 08-gaussian-primitive-infrastructure-and-boys-function
    provides: "#[cube] + *_host() paired math module pattern (boys.rs, obara_saika.rs, rys.rs)"
provides:
  - "Vendored PySCF nr_ecp.{c,h} + nr_ecp_deriv.c (Apache-2.0) at vendor/pyscf-nr-ecp/, compiled via cintx-oracle/build.rs under CINTX_ORACLE_BUILD_VENDOR=1, emitting has_vendor_pyscf_nr_ecp cfg"
  - "Four new ECP rows in compiled_manifest.lock.json at OperatorIds 26..=29 (int1e_ecp_{cart,sph}, int1e_ecp_ipnuc_{cart,sph}) with canonical_family = ecp, all four profiles populated, oracle_covered = false (Plan 04/05 promotes)"
  - "Empty-but-compiling math stubs at crates/cintx-cubecl/src/math/{bessel,radial_quadrature}.rs with paired #[cube] + *_host() signatures + PySCF nr_ecp.h constants (ECP_LMAX=5, K_TAYLOR_MAX=7, K_TAB_ENTRIES=400, LEVEL0=5, LEVEL_MAX=11)"
  - "EcpShell + EcpChannel placeholder types at crates/cintx-core/src/ecp.rs (re-exported from cintx-core lib root)"
  - "Cu/LANL2DZ fixture builder build_cu_lanl2dz() returning (atm, bas, ecpbas, env) with PTR_ENV_START prepad and env[AS_ECPBAS_OFFSET]/env[AS_NECPBAS] populated; basis sourced from basissetexchange.org (LANL2DZ, Hay & Wadt 1985)"
affects:
  - "Phase 19 Plan 02 (math infrastructure — fills bessel.rs + radial_quadrature.rs bodies)"
  - "Phase 19 Plan 03 (typed surface — fleshes out EcpShell field set, adds INT2E_STG_SPH_OPERATOR_ID/INT2E_IPIP1_SPH_OPERATOR_ID shift to 106/116)"
  - "Phase 19 Plan 04/05 (Type-1/Type-2 kernels + parity — promotes oracle_covered=true for the 4 ECP rows)"
  - "Phase 19 Plan 06 (libecpint secondary oracle — optional)"

tech-stack:
  added:
    - "vendor/pyscf-nr-ecp/ subtree (Apache-2.0 from upstream pyscf commit 60cd9022b5158b0eef46ded606a03b111a0ad08c)"
    - "cintx-authored dgemm_ reference shim (vendor/pyscf-nr-ecp/src/dgemm_shim.c) — avoids system BLAS dependency"
  patterns:
    - "Parallel cc::Build chain for second-source vendor C (separate static lib cintx_pyscf_nr_ecp under has_vendor_pyscf_nr_ecp cfg, gnu99 vs the libcint chain's gnu89)"
    - "Lock-JSON-as-canonical-source for manifest entries (insert into compiled_manifest.lock.json; build.rs regenerates CSV + Rust manifest)"
    - "OnceLock-wrapped serde_json::from_str(include_str!()) for fixture parameter sourcing (keeps basis data in JSON, not hardcoded literals)"

key-files:
  created:
    - "vendor/pyscf-nr-ecp/LICENSE — Apache-2.0 verbatim from pyscf"
    - "vendor/pyscf-nr-ecp/NOTICE — provenance (upstream URL, commit SHA, Apache-2.0 grant, shim disclosure)"
    - "vendor/pyscf-nr-ecp/src/nr_ecp.c — PySCF Type-1/Type-2 ECP C reference"
    - "vendor/pyscf-nr-ecp/src/nr_ecp_deriv.c — PySCF ECP derivative C reference"
    - "vendor/pyscf-nr-ecp/include/nr_ecp.h — slot constants AS_ECPBAS_OFFSET=18, AS_NECPBAS=19, RADI_POWER=3, SO_TYPE_OF=4, ECP_LMAX=5"
    - "vendor/pyscf-nr-ecp/include/gto/nr_ecp.h — duplicate (satisfies #include \"gto/nr_ecp.h\" without patching upstream .c)"
    - "vendor/pyscf-nr-ecp/include/{np_helper/np_helper.h, vhf/fblas.h} — cintx-authored minimal shims"
    - "vendor/pyscf-nr-ecp/src/dgemm_shim.c — cintx-authored reference dgemm_ (correctness-first triple loop)"
    - "crates/cintx-cubecl/src/math/bessel.rs — modified spherical Bessel stub"
    - "crates/cintx-cubecl/src/math/radial_quadrature.rs — Gauss-Chebyshev/Gauss-Hermite stub"
    - "crates/cintx-core/src/ecp.rs — EcpShell + EcpChannel placeholders"
    - "crates/cintx-oracle/data/cu_lanl2dz.json — Cu LANL2DZ basis + ECP (8 AO shells, 3 ECP projectors)"
    - ".planning/notes/pyscf-nr-ecp-vendor-subset.md — vendor subset rationale + shim documentation"
  modified:
    - "crates/cintx-oracle/build.rs — parallel cc::Build for PySCF nr_ecp, has_vendor_pyscf_nr_ecp cfg"
    - "crates/cintx-cubecl/src/math/mod.rs — register bessel + radial_quadrature submodules"
    - "crates/cintx-core/src/lib.rs — pub mod ecp + re-export EcpShell, EcpChannel"
    - "crates/cintx-oracle/src/fixtures.rs — build_cu_lanl2dz() consuming embedded JSON via OnceLock"
    - "crates/cintx-ops/generated/compiled_manifest.lock.json — 4 new ECP entries (canonical source)"
    - "crates/cintx-ops/src/generated/api_manifest.csv — regenerated from lock (rows 28-31)"
    - "crates/cintx-ops/src/generated/api_manifest.rs — regenerated from lock (MANIFEST_ENTRIES indices 26-29)"

key-decisions:
  - "Use upstream pyscf master HEAD commit 60cd9022b5158b0eef46ded606a03b111a0ad08c for the vendor pin (no tagged release explicitly matches; SHA recorded in NOTICE for reproducibility)."
  - "Ship a cintx-authored dgemm_ shim rather than depending on a system BLAS (-lblas requires libblas-dev which isn't always installed); a future build can drop the shim and link system BLAS without touching the rest of the vendor tree."
  - "Use -std=gnu99 for the PySCF subtree's cc::Build (vs the libcint chain's gnu89) because nr_ecp.c uses C99 mid-block for-loop init declarations and complex.h. Two separate static libs keep the flag isolated."
  - "Parse cu_lanl2dz.json at runtime via OnceLock + include_str! rather than hardcoding the basis literals (BSE remains the auditable source-of-truth; downstream plans can refresh the basis without touching Rust code)."
  - "Insert ECP entries into compiled_manifest.lock.json (canonical) — the build.rs in cintx-ops regenerates api_manifest.{csv,rs} from the lock. This INVERTS the plan's stated approach (\"edit CSV, run xtask manifest-audit --update\") because the lock is in fact the source-of-truth and the CSV is a derived artifact."
  - "Cu/LANL2DZ fixture splits BSE general contractions into 8 single-NCTR libcint bas rows (libcint requires NCTR_OF=1 per row for distinct contraction coefficients)."

patterns-established:
  - "Pattern A: Parallel vendor cc::Build chains compile into distinct static libs under per-source cfg flags (cintx_oracle_vendor + has_vendor_libcint; cintx_pyscf_nr_ecp + has_vendor_pyscf_nr_ecp). Each chain owns its own -std= flag."
  - "Pattern B: Lock-JSON is the canonical manifest source; CSV + Rust manifest are derived by crates/cintx-ops/build.rs. To add a manifest row, edit the lock JSON only."
  - "Pattern C: Wave-0-scaffold modules expose unimplemented!() macros tagged with the responsible plan (e.g. unimplemented!(\"Phase 19 Plan 02: modified_spherical_bessel_in_host\")) so missing-body bugs surface as clear runtime errors during downstream test execution."

requirements-completed: [ECP-01, ECP-02, ECP-03, ECP-04, ECP-05]

# Metrics
duration: 13min
completed: 2026-05-12
---

# Phase 19 Plan 01: Wave 0 Install/Scaffold for `int1e_ecp_*` Type-1/Type-2 Evaluator Summary

**Vendored PySCF nr_ecp.{c,h} + nr_ecp_deriv.c as the primary ECP byte-identity oracle, registered 4 new ECP manifest rows at OperatorIds 26..=29 with INT4C1E_CART_OPERATOR_ID=24 preserved, and landed empty-but-compiling math/core stubs plus a Cu/LANL2DZ JSON-backed fixture builder.**

## Performance

- **Duration:** 13 min
- **Started:** 2026-05-12T09:35:41Z
- **Completed:** 2026-05-12T09:48:55Z
- **Tasks:** 3
- **Files created:** 13
- **Files modified:** 7

## Accomplishments

- Vendored PySCF `pyscf/lib/gto/nr_ecp.{c,h}` + `nr_ecp_deriv.c` (upstream commit `60cd9022b5158b0eef46ded606a03b111a0ad08c`, Apache-2.0) into `vendor/pyscf-nr-ecp/`, compiled via a parallel `cc::Build` chain in `crates/cintx-oracle/build.rs` emitting `has_vendor_pyscf_nr_ecp` cfg under the existing `CINTX_ORACLE_BUILD_VENDOR=1` gate. Provenance preserved in `LICENSE` and `NOTICE`.
- Shipped cintx-authored shim headers (`np_helper/np_helper.h`, `vhf/fblas.h`) plus a `dgemm_` reference implementation so the vendor build does not depend on installed system BLAS or numpy headers.
- Inserted 4 new ECP manifest rows into the canonical lock JSON; cintx-ops build.rs regenerated CSV + Rust manifest with `OperatorId::new(26..=29) → int1e_ecp_{cart,sph,ipnuc_cart,ipnuc_sph}` and `INT4C1E_CART_OPERATOR_ID = 24` preserved unchanged.
- Landed empty-but-compiling stubs for `math::bessel`, `math::radial_quadrature`, `cintx_core::EcpShell`, and `cintx_core::EcpChannel` — every downstream Wave 1 plan can depend on these symbols without waiting on each other.
- Built `crates/cintx-oracle/src/fixtures.rs::build_cu_lanl2dz()` returning the 4-tuple `(atm, bas, ecpbas, env)` with `PTR_ENV_START` prepad, sourced from `crates/cintx-oracle/data/cu_lanl2dz.json` (8 AO shells from BSE general-contraction split + 3 ECP projectors — coverage invariant satisfied).

## Task Commits

Each task was committed atomically on `main`:

1. **Task 1: Vendor PySCF nr_ecp + extend cintx-oracle/build.rs** — `598100b` (chore)
2. **Task 2: Empty stubs for bessel, radial_quadrature, EcpShell + Cu/LANL2DZ fixture** — `148231b` (feat)
3. **Task 3: Append 4 manifest CSV rows + regenerate four-profile lock** — `9ebcec5` (feat)

**Plan metadata commit:** (this SUMMARY + STATE.md update, immediately after this file lands)

## Files Created/Modified

### Created

- `vendor/pyscf-nr-ecp/LICENSE` — Apache-2.0 verbatim from upstream
- `vendor/pyscf-nr-ecp/NOTICE` — upstream URL, commit SHA, Apache-2.0 grant, shim disclosure
- `vendor/pyscf-nr-ecp/src/nr_ecp.c` — 6543 lines, primary Type-1/Type-2 reference
- `vendor/pyscf-nr-ecp/src/nr_ecp_deriv.c` — 1007 lines, derivative reference
- `vendor/pyscf-nr-ecp/src/dgemm_shim.c` — cintx-authored minimal `dgemm_` (4 transpose combinations)
- `vendor/pyscf-nr-ecp/include/nr_ecp.h` — slot constants verbatim from upstream
- `vendor/pyscf-nr-ecp/include/gto/nr_ecp.h` — duplicate (path-include satisfaction)
- `vendor/pyscf-nr-ecp/include/np_helper/np_helper.h` — cintx-authored shim
- `vendor/pyscf-nr-ecp/include/vhf/fblas.h` — cintx-authored shim
- `crates/cintx-cubecl/src/math/bessel.rs` — modified-spherical-Bessel stub (#[cube] + *_host())
- `crates/cintx-cubecl/src/math/radial_quadrature.rs` — Gauss-Chebyshev + Gauss-Hermite stub
- `crates/cintx-core/src/ecp.rs` — EcpShell + EcpChannel placeholders
- `crates/cintx-oracle/data/cu_lanl2dz.json` — Cu LANL2DZ basis + ECP (8 + 3)
- `.planning/notes/pyscf-nr-ecp-vendor-subset.md` — vendor subset rationale + shim documentation

### Modified

- `crates/cintx-oracle/build.rs` — parallel PySCF nr_ecp `cc::Build` chain + `has_vendor_pyscf_nr_ecp` cfg
- `crates/cintx-cubecl/src/math/mod.rs` — register `bessel` + `radial_quadrature` submodules
- `crates/cintx-core/src/lib.rs` — `pub mod ecp` + `pub use ecp::{EcpShell, EcpChannel}`
- `crates/cintx-oracle/src/fixtures.rs` — `build_cu_lanl2dz()` via OnceLock + `include_str!` of the JSON
- `crates/cintx-ops/generated/compiled_manifest.lock.json` — 4 new ECP entries (canonical insert)
- `crates/cintx-ops/src/generated/api_manifest.csv` — regenerated from lock (rows 28-31)
- `crates/cintx-ops/src/generated/api_manifest.rs` — regenerated from lock (`OperatorId::new(26..=29)`)
- `xtask/Cargo.lock` — reconciled cintx-rs/cintx-runtime transitive edges (was inconsistent prior)

## Decisions Made

### Upstream provenance

- **PySCF vendor pin:** Used `master` HEAD commit `60cd9022b5158b0eef46ded606a03b111a0ad08c` (as of 2026-05-12 fetch). PySCF does not maintain a stable nr_ecp-only release tag; the SHA is recorded verbatim in `vendor/pyscf-nr-ecp/NOTICE` to allow byte-identity reproduction. Future cintx versions may rebase to a tagged pyscf release (e.g. v2.7.0) when the team picks one.

### BLAS dependency strategy

- **Ship a cintx-authored `dgemm_` shim** (vendor/pyscf-nr-ecp/src/dgemm_shim.c) rather than `-lblas` linking. Dev host had only `libblas.so.3` without the `.so` development symlink, so plain `-lblas` would not link. The shim is correctness-first (triple loop, no blocking). Documentation in `.planning/notes/pyscf-nr-ecp-vendor-subset.md` describes how a future build can drop the shim and link a real BLAS. **No `daxpy_` / `dcopy_` / `dscal_` etc. are linked** — the 9 `dgemm_` call sites in nr_ecp.c are the entire BLAS surface.

### Compile flag isolation

- **PySCF cc::Build uses `-std=gnu99`** (vs libcint chain's `-std=gnu89`). nr_ecp.c uses C99 mid-block for-loop init declarations and `<complex.h>`; gnu89 rejects them. Two distinct static libs (`cintx_oracle_vendor`, `cintx_pyscf_nr_ecp`) keep the flag choice isolated — the libcint chain's flag is untouched.

### Manifest insertion strategy (deviation from plan wording)

- **The lock JSON is the canonical source**, not the CSV. cintx-ops's `build.rs` reads `crates/cintx-ops/generated/compiled_manifest.lock.json` and regenerates BOTH `src/generated/api_manifest.csv` AND `src/generated/api_manifest.rs` on every build. The plan's stated insertion strategy ("edit CSV then run `xtask manifest-audit --update`") is misaligned with how cintx-ops actually works — the xtask `manifest-audit` subcommand only has a `--check-lock` audit mode, no `--update`. The correct procedure (and what Task 3 followed) is: edit the lock JSON in place, then `cargo build -p cintx-ops` to trigger the regeneration. This is the genuinely-canonical regeneration command; record for Plan 03's manifest-touch step.

### Cu/LANL2DZ fixture parameter sourcing

- **Parse `cu_lanl2dz.json` at runtime via `serde_json::from_str(include_str!(...))` + `OnceLock`** (plan option A) rather than hardcode literals into Rust (option B). Keeps the BSE-fetched basis parameters as the auditable single source of truth; downstream plans can refresh the basis without touching Rust source. Cu/LANL2DZ general-contraction blocks from BSE (3 s-block contractions, 3 p-block, 2 d-block) are split into 8 single-NCTR libcint bas rows.

### Manifest ECP entry shape

- **`canonical_family = "ecp"`** (short, parallel to F12's `"f12"`) routes through the future `kernels::ecp::launch_ecp` arm Plan 04 adds. **`family_name = "1e"`** preserves the arity-2 1e routing for the resolver — same F12 pattern (`family_name = "2e"` but `canonical_family = "f12"`).
- **`oracle_covered: false`** for the 4 new entries — Plan 04/05 promotes to `true` once parity tests land. The xtask `manifest-audit --check-lock` gate WILL fail until then, mirroring the Phase 17/18 precedent for staged oracle coverage.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] PySCF cc::Build cannot use `-std=gnu89`**
- **Found during:** Task 1 (first build attempt)
- **Issue:** Initial build failed because the dgemm_shim and nr_ecp.c use C99 mid-block for-loop init declarations, rejected by `-std=gnu89`.
- **Fix:** Switched the PySCF parallel `cc::Build` to `-std=gnu99` (libcint chain unchanged). Both chains compile into separate static libs so the flag does not bleed.
- **Files modified:** `crates/cintx-oracle/build.rs` (one line — flag change)
- **Verification:** `CINTX_ORACLE_BUILD_VENDOR=1 cargo build --locked -p cintx-oracle` exits 0.
- **Committed in:** `598100b` (Task 1)

**2. [Rule 2 - Missing Critical] System BLAS not available; ship a minimal `dgemm_` shim**
- **Found during:** Task 1 (when wiring the cc::Build)
- **Issue:** PySCF nr_ecp.c calls `dgemm_` from libBLAS, but the dev host has only `libblas.so.3` (no `.so` development symlink). `-lblas` would not link without `apt install libblas-dev`. Plan's stated approach ("link `-lblas` if dgemm_ is referenced") would break on hosts without the dev package installed.
- **Fix:** Authored `vendor/pyscf-nr-ecp/src/dgemm_shim.c` — a minimal Fortran-BLAS-compatible `dgemm_` reference implementation handling all 4 transpose combinations. Included in the PySCF cc::Build chain. Documented in NOTICE and `.planning/notes/pyscf-nr-ecp-vendor-subset.md` as cintx-authored (NOT upstream PySCF).
- **Files modified:** Added `vendor/pyscf-nr-ecp/src/dgemm_shim.c`; build.rs `.file(...)` entry; NOTICE + rationale note.
- **Verification:** Build completes; `dgemm_` symbol resolved.
- **Committed in:** `598100b` (Task 1)

**3. [Rule 3 - Blocking] PySCF nr_ecp.c includes `gto/nr_ecp.h`; no `gto/` subdir exists**
- **Found during:** Task 1
- **Issue:** nr_ecp.c contains `#include "gto/nr_ecp.h"`. Upstream PySCF organizes headers under `pyscf/lib/gto/`; cintx vendors `nr_ecp.h` at `vendor/pyscf-nr-ecp/include/nr_ecp.h`. The `gto/`-prefixed include wouldn't resolve.
- **Fix:** Duplicated `nr_ecp.h` to `include/gto/nr_ecp.h` rather than patching the upstream `.c` file. Preserves byte-identity of upstream `.c` source.
- **Files modified:** Added `vendor/pyscf-nr-ecp/include/gto/nr_ecp.h` (verbatim copy).
- **Committed in:** `598100b` (Task 1)

**4. [Rule 1 - Specification correction] Plan's `pub use` order vs `cargo check` requirement**
- **Found during:** Task 2
- **Issue:** First commit had `pub use ecp::{EcpChannel, EcpShell}` (alphabetical). The plan's acceptance criterion is `grep -F 'pub use ecp::{EcpShell, EcpChannel};'` (declaration order). Both compile fine, but the exact-string check would fail.
- **Fix:** Reordered to `pub use ecp::{EcpShell, EcpChannel};` to match the plan's exact-grep acceptance criterion.
- **Files modified:** `crates/cintx-core/src/lib.rs` (one line)
- **Committed in:** `148231b` (Task 2 — both attempts squashed into single task commit)

**5. [Plan acceptance-criterion error] `grep -c '"canonical_family": "ecp"' >= 16` not achievable with lock JSON shape**
- **Found during:** Task 3 (verifying acceptance criteria)
- **Issue:** Plan expects `grep -c >= 16` (4 rows × 4 profiles). But the lock JSON stores each entry once with `compiled_in_profiles` as an inline array — there's no per-profile duplication. Actual `grep -c` result: 4 (one per entry). The intent — "4 ECP entries × 4 profiles = 16 (entry, profile) pairings" — IS satisfied (verified via `jq '[.entries[] | select(.canonical_family == "ecp") | .compiled_in_profiles | length] | add' = 16`).
- **Fix:** No code change; record in SUMMARY so future verifier knows the actual shape. The substantive coverage is satisfied; the literal grep count is a measurement-method mismatch in the plan's wording.
- **Verification:** `jq '[.entries[] | select(.canonical_family == "ecp")] | length' = 4`; `jq '[.entries[] | select(.canonical_family == "ecp") | .compiled_in_profiles | length] | add' = 16`.

**6. [Plan execution-path correction] No `xtask manifest-audit --update` subcommand**
- **Found during:** Task 3
- **Issue:** Plan says to run `cargo run -p xtask --locked -- manifest-audit --update`. Inspection of `xtask/src/main.rs` and `xtask/src/manifest_audit.rs` shows the subcommand only accepts `--profiles <csv>` and `--check-lock`; there is no `--update`. The actual regeneration is performed by `crates/cintx-ops/build.rs` whenever the lock JSON changes — automatic, no manual command needed.
- **Fix:** Edit the lock JSON directly, then `cargo build -p cintx-ops` triggers the regeneration. Documented in the Decisions section above so Plan 03's manifest-touch step uses the correct workflow.
- **Verification:** After editing the lock, `cargo build -p cintx-ops` regenerated `api_manifest.csv` and `api_manifest.rs` with the expected 4 new rows at OperatorIds 26..=29.

---

**Total deviations:** 6 auto-fixed (3 blocking, 1 missing-critical, 2 plan-text corrections)
**Impact on plan:** All deviations were execution-path corrections (build flags, missing shim files, included path satisfaction, command syntax). None required scope expansion or architectural change. All 3 substantive plan tasks landed with the intended outcomes.

## Issues Encountered

- **xtask `manifest-audit --check-lock` reports drift** on the 4 new ECP rows (`uncovered_stable_entries: ["int1e_ecp_cart", "int1e_ecp_sph", "int1e_ecp_ipnuc_cart", "int1e_ecp_ipnuc_sph"]`). This is INTENTIONAL — `oracle_covered: false` is set by design per the plan's CSV row template; Plan 04/05 will flip to `true` once parity tests are wired. The `--check-lock` gate failure is the expected Phase 17/18 precedent for staged oracle-coverage promotion (same pattern as the unstable-source family staging). Do NOT take this as a regression.

## Known stubs

The following modules are intentional empty stubs that Plan 02 / Plan 03 fill:

| File | Stub function(s) | Owning plan |
| ---- | ---------------- | ----------- |
| `crates/cintx-cubecl/src/math/bessel.rs` | `modified_spherical_bessel_in_host`, `modified_spherical_bessel_in` (#[cube]) | Plan 02 |
| `crates/cintx-cubecl/src/math/radial_quadrature.rs` | `gauss_chebyshev_nodes_weights_host`, `gauss_chebyshev_nodes_weights` (#[cube]), `gauss_hermite_nodes_weights_host` | Plan 02 |
| `crates/cintx-core/src/ecp.rs` `EcpShell` | Missing fields: `atom_index`, `radial_power`, `so_type`, `nprim`, `nctr` | Plan 03 |
| `crates/cintx-oracle/src/fixtures.rs` `build_cu_lanl2dz` | `env[AS_ECPBAS_OFFSET] = 0.0` is a sentinel meaning "ecpbas passed as separate slab"; Plan 03/04 may revisit to pack ecpbas into a combined slab | Plan 03/04 |

All `*_host()` stubs panic with `unimplemented!("Phase 19 Plan 02: ...")` so missing-body bugs surface as clear runtime errors.

## OperatorId mapping (post-regeneration)

For Plan 03's read-and-derive step:

| OperatorId | Symbol | Notes |
| ---------- | ------ | ----- |
| 24 | `int4c1e_cart` | **PRESERVED** unchanged (key invariant) |
| 25 | `int4c1e_sph` | unchanged |
| **26** | **`int1e_ecp_cart`** | **NEW** |
| **27** | **`int1e_ecp_sph`** | **NEW** |
| **28** | **`int1e_ecp_ipnuc_cart`** | **NEW** (component_rank = 3) |
| **29** | **`int1e_ecp_ipnuc_sph`** | **NEW** (component_rank = 3) |
| 30 | `CINTlen_cart` | shifted +4 (was 26 pre-Phase 19) |
| ... | ... | helpers/legacy continue with +4 shift |

Test-only constants that Plan 03 must update in `crates/cintx-rs/src/api.rs`:

- `INT2E_STG_SPH_OPERATOR_ID`: was 102 → now **106** (verify by re-running api_manifest.rs MANIFEST_ENTRIES line search for `int2e_stg_sph`)
- `INT2E_IPIP1_SPH_OPERATOR_ID`: was 112 → now **116**
- `INT4C1E_CART_OPERATOR_ID`: 24 unchanged

These constants live inside `#[cfg(test)] mod tests` blocks, so `cargo --locked check -p cintx-rs` still passes — the test code is gated. `cargo test -p cintx-rs --features with-f12` will fail until Plan 03 lands the constant updates; this is expected.

## Next Phase Readiness

Wave 1 plans (02 math infrastructure, 03 typed surface) can now run in parallel:

- **Plan 02 (math):** Stubs at `crates/cintx-cubecl/src/math/bessel.rs` and `radial_quadrature.rs` exist with paired signatures + PySCF constants. Plan 02 fills bodies using the PySCF table-then-recurrence hybrid (Bessel) and Gauss-Chebyshev / Gauss-Hermite (radial quadrature).
- **Plan 03 (typed surface):** `EcpShell` + `EcpChannel` re-exported from `cintx-core`. Plan 03 fleshes out the field set (`atom_index`, `radial_power`, `so_type`, `nprim`, `nctr`) and adds `BasisSet::ecp_shells()` + `try_new_with_ecp`. Plan 03 also updates the shifted F12 test constants (INT2E_STG_SPH → 106, INT2E_IPIP1_SPH → 116) per the OperatorId mapping table above.
- **Plan 04/05 (kernels + parity):** `vendor/pyscf-nr-ecp/` is built and linked under `has_vendor_pyscf_nr_ecp`. The Cu/LANL2DZ fixture (`build_cu_lanl2dz()`) is ready for parity tests. Plans 04/05 add the kernel arm in `cintx-cubecl/src/kernels/mod.rs::resolve_family_name`, the FFI wrappers, and the per-symbol parity tests, then flip `oracle_covered` to `true` in the lock.
- **Plan 06 (libecpint cross-check, optional):** No dependency on Plan 01 outputs.

No blockers carrying forward.

## Self-Check: PASSED

Files verified to exist on disk:
- `vendor/pyscf-nr-ecp/{LICENSE,NOTICE,src/nr_ecp.c,src/nr_ecp_deriv.c,src/dgemm_shim.c,include/nr_ecp.h}` ✓
- `crates/cintx-cubecl/src/math/{bessel.rs,radial_quadrature.rs}` ✓
- `crates/cintx-core/src/ecp.rs` ✓
- `crates/cintx-oracle/data/cu_lanl2dz.json` ✓
- `.planning/notes/pyscf-nr-ecp-vendor-subset.md` ✓

Commits verified to exist:
- `598100b` (Task 1) ✓
- `148231b` (Task 2) ✓
- `9ebcec5` (Task 3) ✓

Substantive acceptance criteria verified:
- `cargo --locked check -p cintx-ops -p cintx-core -p cintx-cubecl -p cintx-oracle` exits 0 ✓
- `cargo --locked check --workspace` exits 0 ✓
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo build --locked -p cintx-oracle` exits 0 ✓
- `vendor/pyscf-nr-ecp/LICENSE` contains "Apache License" ✓
- `vendor/pyscf-nr-ecp/include/nr_ecp.h` contains AS_ECPBAS_OFFSET, AS_NECPBAS, RADI_POWER, SO_TYPE_OF, ECP_LMAX ✓
- Exactly 4 `int1e_ecp_*` rows in api_manifest.csv ✓
- 0 `int1e_ecp_spinor` rows (D-12 deferral honored) ✓
- `INT4C1E_CART_OPERATOR_ID = 24` preserved in cintx-rs/src/api.rs ✓
- OperatorId 24 still points at int4c1e_cart in api_manifest.rs ✓
- OperatorIds 26..=29 map to int1e_ecp_{cart,sph,ipnuc_cart,ipnuc_sph} ✓
- `jq` coverage invariant for Cu/LANL2DZ JSON (≥8 shells, ≥3 ECP) returns `true` ✓

---
*Phase: 19-int1e-ecp-type1-type2-evaluator*
*Completed: 2026-05-12*
