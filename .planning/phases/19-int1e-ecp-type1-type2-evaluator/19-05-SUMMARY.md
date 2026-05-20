---
phase: 19-int1e-ecp-type1-type2-evaluator
plan: 05
subsystem: math
tags: [ecp, k-taylor, bessel, radial, host-port, pyscf-nr-ecp, blob, drift-gate, byte-identity]

requires:
  - phase: 19-int1e-ecp-type1-type2-evaluator
    provides: "Plan 01 (vendored PySCF nr_ecp + K-Taylor constants in bessel.rs), Plan 02 (math::bessel modified_spherical_bessel_in_host + math::radial_quadrature), Plan 04 (launch_ecp host launcher with the direct-quadrature compute_type1/2_pair this foundation will replace in 19-06)"
  - phase: 13-f12-stg-yp-kernels
    provides: "Pattern: include_bytes! + bytemuck::AlignedBytes binary table embedding (roots_xw_data.rs precedent)"
  - phase: 04-verification-release-automation
    provides: "manifest_drift_gate single-runner fail-closed CI job in compat-governance-pr.yml; manifest-audit --check-lock step the new ECP drift-gate mirrors"
provides:
  - "gen-ecp-tables xtask subcommand (xtask/src/gen_ecp_tables.rs): parses _sph_ine_tab (400x24) and _sph_ine_tab_order7 (400x8x8) literal arrays out of vendor/pyscf-nr-ecp/src/nr_ecp.c and emits LE-f64 .bin blobs; --check is a whole-blob byte-exact drift gate that fails closed on a stale/edited blob (D-15)"
  - "Two committed K-Taylor blobs: crates/cintx-cubecl/src/math/ecp_k_taylor_in.bin (76800 bytes / 9600 f64) and ecp_k_taylor_order7.bin (204800 bytes / 25600 f64)"
  - "ecp_k_taylor_data.rs: AlignedBytes<{N*8}> + include_bytes! + bytemuck::cast_slice accessors sph_ine_tab() -> &'static [f64] (9600) and sph_ine_tab_order7() -> &'static [f64] (25600), mirroring roots_xw_data.rs (D-14)"
  - "ecp_k_taylor.rs host ports: ecpsph_ine_opt_host (table-interpolation modified spherical Bessel, scaled i_l(z)*exp(-z) convention), ecprad_part_host (RADI_POWER switch + SIM_ZERO break), type1_rad_part_host (CUTOFF/EXPCUTOFF guards + ECPsph_ine_opt-backed bval), type2_facs_rad_host (per-primitive radial factors + buf*ci matmul)"
  - "EcpRadShell struct bundling per-ECP-shell (exponents, coefficients, radial_power) for ecprad_part_host"
  - "CI step 'Run ECP K-Taylor table drift gate' wired into the manifest_drift_gate job of compat-governance-pr.yml (gen-ecp-tables --check), enforcing D-15 as a PR gate"
affects:
  - "Phase 19 Plan 06 (scalar close) — consumes ecpsph_ine_opt_host / ecprad_part_host / type1_rad_part_host / type2_facs_rad_host to replace the direct-quadrature compute_type1_pair / compute_type2_pair bodies in kernels/ecp.rs and reach atol=1e-12 byte-identity"
  - "Phase 19 Plan 07 (gradient) — builds nr_ecp_deriv.c gradient on the same K-Taylor radial foundation"

tech-stack:
  added: ["bytemuck (added as an xtask dependency for the f64 <-> LE-bytes round-trip in the table writer/comparator)"]
  patterns:
    - "Pattern: regenerable + CI-drift-checked binary table (stronger than the Rys roots_xw precedent). The .bin blob is checked in, an xtask subcommand regenerates it from the vendored C source, and --check byte-compares the committed blob against a fresh re-extraction inside the manifest_drift_gate CI job — mirroring manifest-lock discipline (D-15)."
    - "Pattern: scaled-Bessel byte-identity convention. PySCF's ECPsph_ine / ECPsph_ine_opt produce the SCALED i_l(z)*exp(-z) (the embedded tables encode it); the port replicates ECPsph_ine verbatim rather than delegating to the unscaled bessel.rs series, so the fall-through and table paths share one convention."
    - "Pattern: host-first port of GPU-target math (D-16). The byte-identity gate runs CPU-vs-C, so a *_host() port closes the requirement; the #[cube] counterpart is a documented deviation tracked in CONTEXT Deferred Ideas — host-only is not the intended end state under the CLAUDE.md 'CubeCL is primary backend' constraint."

key-files:
  created:
    - "xtask/src/gen_ecp_tables.rs (table extractor + --check drift gate + 3 unit tests)"
    - "crates/cintx-cubecl/src/math/ecp_k_taylor_in.bin (76800 bytes, _sph_ine_tab)"
    - "crates/cintx-cubecl/src/math/ecp_k_taylor_order7.bin (204800 bytes, _sph_ine_tab_order7)"
    - "crates/cintx-cubecl/src/math/ecp_k_taylor_data.rs (blob accessors)"
    - "crates/cintx-cubecl/src/math/ecp_k_taylor.rs (4 host ports + 9 tests, ~530 lines)"
  modified:
    - "xtask/src/main.rs (mod gen_ecp_tables; GenEcpTables{check} Command variant; gen-ecp-tables dispatch arm; parse_gen_ecp_tables; help line)"
    - "xtask/Cargo.toml (bytemuck = \"1\")"
    - "xtask/Cargo.lock (bytemuck entry on the xtask package)"
    - "crates/cintx-cubecl/src/math/mod.rs (pub mod ecp_k_taylor; pub mod ecp_k_taylor_data;)"
    - ".github/workflows/compat-governance-pr.yml (new 'Run ECP K-Taylor table drift gate' step in manifest_drift_gate)"

key-decisions:
  - "Ported PySCF ECPsph_ine (the scaled i_l(z)*exp(-z) fall-through) verbatim inside ecp_k_taylor.rs instead of delegating the small/large regimes to bessel.rs::modified_spherical_bessel_in_host. The plan's behavior tests #3/#4 prescribed delegating to that function, but it deliberately drops the exp(-z) scaling (per its own rustdoc and the Plan 02 decision) while the embedded K-Taylor tables AND ECPsph_ine encode the SCALED form (first table entry 9.802640211919197e-01 = i_0(0.02)*exp(-0.02)). Delegating would have broken byte-identity. Rule 1 fidelity-fix; the fall-through tests now pin against the scaled ECPsph_ine arithmetic directly."
  - "Designed the radial-port host signatures (EcpRadShell, ur/rs/rs_off/nrs/inc parameters) to carry per-shell data directly rather than the raw ecpbas/env i32-slab + MALLOC_INSTACK cache indirection. The algorithm body (Gaussian sum, RADI_POWER switch, SIM_ZERO break, CUTOFF guards, parity-strided accumulation, dgemm contraction) is replicated verbatim; only the data-marshaling boundary is Rust-idiomatic — the scalar-close plan (19-06) marshals EcpShell/EcpBasArray into these calls."
  - "Replicated type2_facs_rad's dgemm_(N,N) primitive->contraction reduction as a plain column-major Rust matmul (facs[m x nc] = buf[m x np] * ci[np x nc]) rather than wiring a BLAS dependency into cintx-cubecl. Keeps the host port self-contained and dependency-free; numerically identical for the f64 accumulation order PySCF uses."
  - "Root Cargo.lock dependency drift (cudarc 0.19.4->0.19.7, filetime, winnow patch bumps) surfaced during the build was reverted and NOT committed — it is unrelated to this plan's bytemuck addition (xtask is a separate workspace, so bytemuck lands in xtask/Cargo.lock). Out-of-scope per the executor scope boundary."

requirements-completed: []
# This plan delivers the radial FOUNDATION (ECP-01 Type-1, ECP-02 Type-2,
# ECP-04 parity sweep machinery) but flips no oracle_covered flag and removes
# no #[ignore] — that closure is 19-06's job once the kernel wires these
# primitives and passes byte-identity. No requirement is marked complete here.

# Metrics
duration: ~35min
completed: 2026-05-20
---

# Phase 19 Plan 05: K-Taylor Port Foundation Summary

**One-liner:** Ported PySCF's exact radial machinery host-first — the table-interpolation modified-spherical-Bessel evaluator `ECPsph_ine_opt`, the radial-block `ECPrad_part`, and the Type-1/Type-2 radial assemblers `type1_rad_part`/`type2_facs_rad` — backed by the two K-Taylor tables shipped as little-endian f64 `.bin` blobs (`include_bytes!` + `bytemuck`, mirroring `roots_xw_data.rs`), with a `gen-ecp-tables` xtask extractor whose `--check` byte-exact drift gate is enforced in the `manifest_drift_gate` CI job (D-14/D-15/D-16).

## Tasks Completed

### Task 1: gen-ecp-tables xtask subcommand + .bin blobs + CI drift-gate — commit `52da167`

- **`xtask/src/gen_ecp_tables.rs`** parses the `static double _sph_ine_tab[]` (nr_ecp.c:30) and `static double _sph_ine_tab_order7[]` (nr_ecp.c:434) literal arrays out of the vendored C source: locate the `static double NAME[]` declaration, read every comma-separated f64 token up to the closing `};` (skipping `//` comments and whitespace), parse each with `f64::from_str` (exact round-trip for the printed 16-significant-digit literals).
- Validates parsed counts (`9600` and `25600`) and bails on mismatch (catches a vendored-source row add/remove — threat T-19-17).
- `--check == false`: writes both `.bin` blobs (generate path). `--check == true`: reads the committed blob, whole-blob byte-compares against the fresh extraction, and `anyhow::bail!`s naming the diverging file (drift gate — threat T-19-16).
- Registered in `xtask/src/main.rs` via the five touchpoints (`mod`, `Command::GenEcpTables{check}`, dispatch arm, `parse_gen_ecp_tables`, help line); `bytemuck = "1"` added to `xtask/Cargo.toml`.
- CI: appended a `Run ECP K-Taylor table drift gate` step (`gen-ecp-tables --check`) immediately after the `Run manifest drift gate` step in the `manifest_drift_gate` job of `compat-governance-pr.yml`, matching the existing step shape; YAML still parses.

### Task 2: Embed K-Taylor blobs + port ECPsph_ine_opt host-first — commit `cad32e5`

- **`ecp_k_taylor_data.rs`** mirrors `roots_xw_data.rs` verbatim: `#[repr(C, align(8))] AlignedBytes<{N*8}>` + `static ..._BYTES: &AlignedBytes<{9600*8}> = &AlignedBytes(*include_bytes!("..."))` + `pub fn sph_ine_tab() -> &'static [f64] { bytemuck::cast_slice(...) }`.
- **`ecp_k_taylor.rs::ecpsph_ine_opt_host`** ports `ECPsph_ine_opt` (nr_ecp.c:4687-4837): small/large fall-through to a faithful `ecpsph_ine` (scaled), per-order Taylor sum over `_sph_ine_tab_order7` (`fac *= dz * _j_inv[j]`) for `order<=7`, and the `_l2` downward recurrence over `_sph_ine_tab` for `order>7`. `_l2`, `_j_inv`, `_factorial` arrays copied verbatim; `ORDER7OFFSET=8`.
- 4 tests: blob shape + first-literal bit-match, middle-regime table arithmetic, small-z and large-z scaled fall-through.

### Task 3: Port ECPrad_part, type1_rad_part, type2_facs_rad host-first — commit `5d1af60`

- **`ecprad_part_host`** (nr_ecp.c:4870-4950): r2 precompute, per-shell Gaussian sum, `SIM_ZERO` early-break (`i>2 && |ubuf[i]|<SIM_ZERO && |ubuf[i-1]|<SIM_ZERO`), the `RADI_POWER` switch (cases 1/2/3/default), `ur[i] += ubuf[i]`, returns `nrs_max`.
- **`type1_rad_part_host`** (nr_ecp.c:5754-5806): `kaij = k/(2*aij)`, `fac = kaij^2*aij`, the `ur[n]==0 || tmp>CUTOFF || tmp<-(EXPCUTOFF+6+30)` guard, `rur[n]=ur[n]*exp(tmp)` + `bval = ecpsph_ine_opt_host(lmax, k*rs[n])`, the `lab` loop with the `i=lab%2; i+=2` parity stride; early-return on `nrs==0`.
- **`type2_facs_rad_host`** (nr_ecp.c:5134-5186): per-primitive radial factors with the `EXPCUTOFF+6` guard, then the `dgemm_(N,N)` buf*ci contraction replicated as a column-major Rust matmul; early-return on `nrs==0`.
- 5 tests (total now 9): RADI_POWER=2 Gaussian, SIM_ZERO break, type1 consistency + zero-grid early return, type2 consistency + zero-grid.

## Output items requested by the plan

- **Macro values used (from `vendor/pyscf-nr-ecp/include/gto/nr_ecp.h`):** `SIM_ZERO = 1e-50` (line 13), `EXPCUTOFF = 39` (line 14), `CUTOFF = 460` (line 15). Also `RADI_POWER = 3`, `SO_TYPE_OF = 4`, `ECP_LMAX = 5`, `K_TAYLOR_MAX = 7`, `K_TAB_COL = 24`, `K_TAB_ENTRIES = 400`, `K_TAB_INTERVAL = 16/400 = 0.04`; `ORDER7OFFSET = 8` from nr_ecp.c:433.
- **order>7 (the `_l2` downward-recurrence default branch):** ported verbatim from nr_ecp.c:4813-4834 but **left as a verbatim-but-untested port** for Phase 19. The Phase 19 envelope is `ECP_LMAX = 5` and Type-2 needs `li + lc` (bounded by the Cu/LANL2DZ basis, well within `order <= 7`), so no Phase 19 envelope shell exercises the `order > 7` branch. The four tests pin the `order <= 7` table path and the small/large fall-through. The `order > 7` branch is reachable only by a future higher-l ECP and is byte-faithful to the C should that arise.
- **CI job/step the drift gate was added to:** the `manifest_drift_gate` job of `.github/workflows/compat-governance-pr.yml`, as a new step named `Run ECP K-Taylor table drift gate` running `cargo run --manifest-path xtask/Cargo.toml --locked -- gen-ecp-tables --check`, placed immediately after the existing `Run manifest drift gate` (`manifest-audit ... --check-lock`) step.
- **Literal-parse edge cases hit while extracting the tables:** the `// 400x24` / `// 400x8x8, expand ...` trailing comments on the opening declaration lines, handled by stripping any `// ...` suffix per source line before tokenizing; a trailing comma before each newline and before the closing `};`, handled by skipping empty tokens. Both arrays parsed to the exact expected counts on the first run (9600 / 25600); no manual fix-ups needed.

## Deviations from Plan

### Rule 1 — Fidelity fix: ECPsph_ine ported verbatim (scaled), NOT delegated to the unscaled bessel.rs series

- **Found during:** Task 2 (reading `ECPsph_ine` at nr_ecp.c:4630-4675 and the first `_sph_ine_tab` literal).
- **Issue:** The plan's `<behavior>` tests #3/#4 said the small-z and large-z fall-through of `ecpsph_ine_opt_host` should equal `bessel.rs::modified_spherical_bessel_in_host`. But that function deliberately drops the `exp(-z)` scaling to return the *unscaled* `i_l(z)` (its own rustdoc + Plan 02 decision), while PySCF's `ECPsph_ine` / `ECPsph_ine_opt` and the embedded tables produce the *scaled* `i_l(z)*exp(-z)` (the first table value `9.802640211919197e-01` = `i_0(0.02)*exp(-0.02)`, not `i_0(0.02)`). Delegating to the unscaled series would have made the fall-through inconsistent with the table path and broken byte-identity vs PySCF — the exact class of error this whole plan exists to fix.
- **Fix:** Ported `ECPsph_ine` verbatim (the scaled three-branch form, including the `(1-z)` small-z prefactor and the `exp(-z)` moderate/large factors) as a private `ecpsph_ine` inside `ecp_k_taylor.rs`; `ecpsph_ine_opt_host` delegates the fall-through to it. The fall-through tests now pin against the scaled `ECPsph_ine` arithmetic computed directly (`out[0] = 1 - z` for small-z; the `_factorial` asymptotic polynomial for large-z), not against `modified_spherical_bessel_in_host`.
- **Files modified:** `crates/cintx-cubecl/src/math/ecp_k_taylor.rs`.
- **Commit:** `cad32e5`.

### Scope: host-only ports (D-16, plan-sanctioned)

All four functions are pure host Rust with no `#[cube]` body. This is the D-16 host-first decision and a documented CLAUDE.md ("CubeCL is the primary compute backend") deviation tracked in 19-CONTEXT Deferred Ideas — the byte-identity gate runs CPU-vs-C, so the `*_host()` ports close the requirement; the `#[cube]` GPU counterpart is deferred. Documented in the `ecp_k_taylor.rs` module rustdoc.

### Out-of-scope: root Cargo.lock dependency drift left uncommitted

The workspace build bumped unrelated transitive deps in the root `Cargo.lock` (cudarc, filetime, winnow). This is unrelated to the plan's `bytemuck` addition (xtask is a separate workspace; bytemuck lands in `xtask/Cargo.lock`). Reverted and not committed per the executor scope boundary.

## Verification

- `cargo run --manifest-path xtask/Cargo.toml --locked -- gen-ecp-tables --check` — exits 0 (both blobs match vendored source).
- `crates/cintx-cubecl/src/math/ecp_k_taylor_in.bin` — exactly 76800 bytes; `ecp_k_taylor_order7.bin` — exactly 204800 bytes.
- `grep -rF 'gen-ecp-tables --check' .github/workflows/` — 1 match (CI drift-gate wired).
- `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/compat-governance-pr.yml'))"` — `yaml-ok`.
- `cargo test --locked -p cintx-cubecl --lib math::ecp_k_taylor` — **9 tests passed, 0 failed** (≥8 required).
- `cargo test --manifest-path xtask/Cargo.toml --locked gen_ecp_tables` — 3 tests passed.
- `cargo --locked build --workspace` — exits 0.
- Grep acceptance gates (Task 1/2/3): all matched — `_sph_ine_tab` parse, `"gen-ecp-tables"` dispatch, `mod gen_ecp_tables;`, `bytemuck` dep, `include_bytes!` (2 invocations), `bytemuck::cast_slice`, `pub fn ecpsph_ine_opt_host`, `pub mod ecp_k_taylor;`, the nr_ecp.c:46xx citation, `pub fn {ecprad_part,type1_rad_part,type2_facs_rad}_host`, and 8 `// Source:` citations (≥4 required).

## Known Stubs

None. All four host functions perform real arithmetic ported verbatim from the vendored C and are unit-tested against the embedded tables / the ported pieces. No placeholder values, no empty data sources. The `order > 7` branch of `ecpsph_ine_opt_host` is a verbatim-but-untested port (no Phase 19 envelope shell reaches it), not a stub — it produces a correct value for `order > 7` should a future higher-l ECP exercise it.

## Self-Check: PASSED

Files verified to exist on disk:
- `xtask/src/gen_ecp_tables.rs` — FOUND
- `crates/cintx-cubecl/src/math/ecp_k_taylor_in.bin` (76800 bytes) — FOUND
- `crates/cintx-cubecl/src/math/ecp_k_taylor_order7.bin` (204800 bytes) — FOUND
- `crates/cintx-cubecl/src/math/ecp_k_taylor_data.rs` — FOUND
- `crates/cintx-cubecl/src/math/ecp_k_taylor.rs` — FOUND

Commits verified to exist:
- `52da167` (Task 1) — FOUND
- `cad32e5` (Task 2) — FOUND
- `5d1af60` (Task 3) — FOUND

---
*Phase: 19-int1e-ecp-type1-type2-evaluator*
*Completed: 2026-05-20*
