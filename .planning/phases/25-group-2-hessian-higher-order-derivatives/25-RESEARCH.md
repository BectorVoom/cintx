# Phase 25: Group 2 — Hessian & Higher-Order Derivatives - Research

**Researched:** 2026-05-30
**Domain:** libcint 6.1.3 byte-identity port — high-order derivative integral families (Hessian+) + the Rys `nroots≥6` host-side root/weight engine + fail-closed high-rank staging
**Confidence:** HIGH (all claims verified against the vendored `libcint-master/` source tree and the current cintx codebase; no training-data assumptions on the load-bearing facts)

## Summary

Phase 25 is fundamentally a **source-derivation phase**, not a discovery phase. Every load-bearing fact — the n>5 Rys scheme, the `ng[]` headroom tuples, the `component_rank` values, the gout component order, the autocode source files — is present in the vendored `libcint-master/` tree and has been read directly. There is essentially no external-ecosystem research; the "ecosystem" is libcint 6.1.3, already on disk, already compiled into the cintx-oracle vendor build.

Two findings reshape the plan relative to the CONTEXT/ROADMAP framing:

1. **The primary oracle source file is `hess.c`, not `deriv3.c`/`deriv4.c`.** All four HESS-01 rank-9 1e families (`int1e_ipipovlp/ipipnuc/ipipkin/ipiprinv`) AND all four HESS-02 2e families live in `src/autocode/hess.c` — **which is already in the cintx-oracle `cc::Build`** (`build.rs:74,221`). HESS-03's three multi-center families live in `src/autocode/int3c2e.c` — **also already in the build** (`build.rs:64,218`). Only the HESS-04 3rd/4th-order families need new source files: `deriv3.c` (rank 27) and `deriv4.c` (rank 81), which are **NOT** in the build yet. The ROADMAP SC7 "add deriv3.c/deriv4.c" is correct but incomplete: it omits that `hess.c`/`int3c2e.c` are already wired, so HESS-01/02/03 need **only** the bindgen `allowlist_function` regex extended (the C symbols already compile), while HESS-04 needs both the `.file()` additions and the allowlist.

2. **FND-02 is a host-side port; the device kernel does not need a Wheeler port.** The vendored libcint reference (the byte-identity target) already compiles the full n>5 path (`rys_wheeler.c`, `eigh.c`, `find_roots.c`, `rys_roots.c` are all in `build.rs:200-204`). The cintx port target is the **host** `rys_roots_host` in `rys.rs:3244`, which today panics for nroots>5. The host Rys path (`fill_g_tensor_2e`, `two_electron.rs:417`) is the path the derivative/Hessian families use (gradient precedent: `int2e_ip1` routes through `fill_g_tensor_2e` at `two_electron.rs:1516,1765`, not the on-device comptime-nroots kernel). The on-device comptime `rys_root1..5` recompute (`two_electron.rs:760-768`, capped at `MAX_DEVICE_NROOTS=5`) is the **scalar** path and is a secondary, out-of-critical-path concern — Hessian families ride the host path exactly like the Phase-21/23 gradient families did.

**Primary recommendation:** Sequence exactly as D-06 mandates — Plan 1 = FND-02 (host Wheeler port + executor gate), Plan 2 = FND-06 (single upfront assertion + guard strip + rank-81 OOM test), both merged before any family cluster. Port the libcint n>5 host path **verbatim** (Flocke modified-moments → Wheeler recursion → vendored-MRRR tridiagonal eigensolver → root transform), mirroring the Phase-19 ECP K-Taylor host-first precedent. Validate on a dedicated nroots 6..13 vendor sweep against the already-compiled vendor `CINTrys_roots`.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**FND-02 — Rys nroots≥6 Wheeler/Jacobi fallback**
- **D-01 (fidelity — port verbatim):** Reach byte-identity for nroots 6..~13 by porting libcint 6.1.3's own high-nroots numerical path verbatim (the modified-moments → tridiagonal/Jacobi-matrix → root-polish scheme in libcint's `rys_roots.c` for n>5, NOT the hardcoded low-n polynomial fits). Implement it **host-side**, mirroring the ECP K-Taylor "port the exact upstream machinery host-first" precedent (Phase 19). A clean-room Golub-Welsch was explicitly rejected for last-ULP / root-ordering byte-identity risk. Where math/impl diverge, default to the faithful port.
- **D-02 (validation range + gate opening):** Implement the general algorithm; add a dedicated vendor parity test sweeping nroots 6..~13 against libcint. Extend the `executor.rs` `ang_momentum>4` gate to admit **exactly the max angular momentum the roots are validated for** (g/h/i as covered) — forward-looking foundation per ROADMAP SC1. Not "minimal corpus-only" and not "unbounded above the validated range."
- **D-03 (corpus reality):** Phase 25's own gate corpus (H2O/STO-3G, ≤ d) can push Hessian-elevated d-shells to nroots 6 (the in-phase trigger), but never reaches g/h. The g/h gate extension is forward-looking foundation work — validate it on the dedicated nroots sweep, not only the family parity tests.

**FND-06 — fail-closed high-rank staging**
- **D-04 (single upfront assertion + strip all guards):** Add **one** upfront `BufferTooSmall`-style size assertion at the staging-allocation boundary in `planner.rs` (where `parse_component_multiplier` already sizes staging by `component_rank`), then **remove the per-element `if dst < staging.len()` scatter guards across ALL kernels** (`one_electron.rs`, `two_electron.rs`, `center_3c2e.rs`, `center_2c2e.rs`, `f12.rs`, `unstable/*`) so scatter is unconditional once the buffer is proven large enough. One contract point; no silent partial writes anywhere. NOT a rank≥9-only partial strip, and NOT a per-launcher assertion.
- **D-05 (rank-81 OOM re-validation):** Add a dedicated new test that sets a memory limit smaller than rank-81 staging requires, then asserts a typed OOM/`BufferTooSmall` stop with **NO partial write** (output buffer untouched). Exercises the new upfront assertion + the existing `ChunkPlanner` OOM-safe-stop together.

**Sequencing & plan clustering**
- **D-06 (two foundation plans, then clustered families):** Plan 1 = FND-02, Plan 2 = FND-06; both merge before any family plan starts. Then family clusters low-rank-first: Cluster A = `int1e` rank-9 (HESS-01); Cluster B = 2e Hessian set (HESS-02); Cluster C = `int2c2e_ipip1`/`int3c2e_ipip1`/`int3c2e_ipip2` (HESS-03); Cluster D = 3rd/4th-order `ipipip*` (HESS-04). Family clusters parallelize via worktrees once foundations land. Confirm post-wave integration with `merge-base --is-ancestor`.

**HESS-02 — 2e Hessian promotion from unstable**
- **D-07 (re-home to stable, drop unstable entries):** Move `int2e_ipip1`/`int2e_ipvip1` out of `unstable::source::2e` into the stable family/raw-api map (add cart, set `component_rank`, wire stable launcher + vendor FFI + `vendor_*` test, flip `oracle_covered=true`); register `int2e_ip1ip2`/`int2e_ipip1ipip2` fresh in the same stable family. The unstable sph-only stubs are removed — exactly one canonical stable entry per symbol. NOT an in-place extend + alias.

**Carry-forward locks (from Phases 21–24 — do NOT re-litigate)**
- **D-08 (registration recipe):** 5 steps — (1) manifest lock entry cloning closest family with `component_rank` = true output multiplier, then `cargo build -p cintx-ops` regenerates `api_manifest.{rs,csv}`; (2) `RawApiId` consts in `cintx-compat/src/raw.rs`; (3) launcher dispatch on `descriptor.operator_name()`; (4) vendor FFI — add cart/sph symbols to bindgen `allowlist_function` regex in `cintx-oracle/build.rs` + safe wrappers in `vendor_ffi.rs` (confirm the autocode `.c` is in the build source list); (5) `vendor_*` parity test. Lock edits auto-sync `manifest-audit`.
- **D-09 (transpose discipline):** raise angular-momentum headroom on the **ket** (`ng[]`), not the bra; copy each family's component order verbatim from the libcint gout index map; gate every family with a NON-SQUARE bra×ket block (e.g. p×d). For deriv4, headroom is raised on BOTH bra +2 and ket +2.
- **D-10 (component-rank-truncation hard rule):** a `component_rank` set too LOW silently TRUNCATES trailing output components. Each family's `component_rank` MUST equal its true output multiplier (`ipip*`=9, 3rd-order=27, 4th-order=81 — derive exact values from libcint source, do not guess).
- **D-11 (surface scope):** manifest + RawApiId + kernel + vendor-FFI + oracle only. No `capi` enum variants, no legacy `cint*` wrappers. Spinor reps registered → `UnsupportedApi`.
- **D-12 (verification):** per-family byte-identity at **atol=1e-12** vs vendored libcint 6.1.3, cart + sph, every component, in `vendor_*` parity tests double-gated on `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1` (without both, parity silently skips).

### Claude's Discretion
- Exact `component_rank` value and libcint gout component-index order per family (derived from libcint source; gate with the non-square block).
- The precise libcint `rys_roots.c` routine names/structure for the n>5 path and how literally the host port mirrors libcint's control flow (as long as D-01 byte-identity on the nroots 6..~13 sweep holds).
- The exact enumerated set of HESS-04 "and siblings" 3rd/4th-order families — derive the complete set libcint 6.1.3 actually exports.
- The precise corpus shell-tuple selection for each `vendor_*` test (subject to the non-square bra×ket requirement of D-09).
- Whether Cluster A/C/D moment kernels are one parameterized `#[cube]` entry with a comptime derivative order or order-specialized launchers — implementer's call, as long as D-09 holds.

### Deferred Ideas (OUT OF SCOPE)
- **Lanthanide / f-projector ECP validation** (step 3 of the `rys-nroots-ge6` todo) — a later heavy-element phase, not Phase 25.
- **Spinor Hessian representations** — registered → `UnsupportedApi` this phase (D-11); land when a consumer needs them and the Gap B1/B2 spinor-derivative transforms (Phases 27/28) exist.
- **g/h-basis end-to-end family coverage** — Phase 25 opens the l-gate and validates the roots, but no Phase-25 family is exercised at g/h on the corpus; full g/h family parity rides future heavy-element work.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FND-02 | Rys `nroots≥6` Wheeler-fallback so high-l shells reach byte-identity; no `UnsupportedApi` purely due to `nroots>5` | Exact host scheme derived below (§FND-02 Derivation): `CINTrys_roots` dispatch table (`rys_roots.c:57-123`) → `CINTrys_jacobi`/`CINTlrys_jacobi`/`CINTlrys_laguerre` (`rys_wheeler.c:3678/3703/3692`) → `rys_wheeler_partial`/`lrys_wheeler_partial` → `_CINTdiagonalize` (`eigh.c`, vendored MRRR `#else` branch). Host port target `rys.rs:3244`. |
| FND-06 | Upfront size assertion replaces `if dst < staging.len()` scatter guards; rank-81 OOM re-validated | `BufferTooSmall` variant already exists (`error.rs:66`); boundary at `planner.rs` `try_alloc_staging`/`staging_elements_for_chunk` (`:321,341,452`); 19 guard sites enumerated in §FND-06. |
| HESS-01 | `int1e_ipipovlp/ipipnuc/ipipkin/ipiprinv` rank-9 cart+sph at atol=1e-12 | Source `hess.c` (already in oracle build); `ng[]={2,0,0,0,2,1,0,9}`; gout order extracted (§Family Reference). |
| HESS-02 | 2e Hessian set (`int2e_ipip1/ipvip1/ip1ip2/ipip1ipip2`), promoted from unstable, cart+sph | Source `hess.c` (already in build); `ng[]` tuples + unstable lock entries at `compiled_manifest.lock.json:3327,3360`. |
| HESS-03 | `int2c2e_ipip1`, `int3c2e_ipip1`, `int3c2e_ipip2` cart+sph | Source `int3c2e.c` (already in build); `ng[]={2,0,0,0,2,1,1,9}` / `{0,0,2,0,2,1,1,9}`. |
| HESS-04 | 3rd/4th-order (`int1e_ipipipnuc`, `int1e_ipipipiprinv`, siblings) cart+sph, bra+ket headroom | Source `deriv3.c` (rank 27, `ng[]={3,0,0,0,3,1,0,27}`) + `deriv4.c` (rank 81, `ng[]={2,2,0,0,4,1,0,81}`) — NOT yet in oracle build; full roster in §HESS-04 Roster. |
</phase_requirements>

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| n≥6 Rys roots/weights | Host (`cintx-cubecl::math::rys` host fns) | — | Derivative families route through the host `fill_g_tensor_2e`/host nuclear path; device comptime-nroots kernels cap at 5 and are not on the Hessian critical path. Mirrors ECP K-Taylor host-first port (Phase 19). |
| Hessian G-tensor build (∇² applied) | Host (kernel module `fill_g_tensor` + gout component emit) | Device `#[cube]` (optional, future) | Phase-21/23 gradient families established the host-derivative pattern; Hessian = gradient engine applied twice. Device port is post-correctness optimization (NOT in scope; spinor → UnsupportedApi). |
| Staging size contract / OOM-safe stop | Host (`cintx-runtime::planner`) | — | Single upfront assertion at the chunk-planner allocation boundary; the OS/device never sees an undersized buffer. |
| cart→sph multi-component transform | Host (`transform` / kernel scatter) | — | Per-component c2s reuses the rank-N machinery already proven through Phase 24's rank-9/rank-N families. |
| Vendor byte-identity reference | Host C (cintx-oracle `cc::Build` of `libcint-master`) | — | The compiled vendor libcint (incl. full Wheeler+MRRR path) is the oracle; cintx Rust is compared against it. |
| Executor l-gate / nroots admission | Host (`cintx-cubecl::executor`) | — | Single gate point upstream of all family launchers. |

---

## Standard Stack

This phase adds **no new external dependencies**. Everything is internal to the cintx workspace + the vendored `libcint-master/` C tree (the oracle reference). The CLAUDE.md-pinned stack governs.

### Core (already present, governs this phase)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cubecl` | 0.10.0 (pinned) | Compute backend for kernel launchers | CLAUDE.md constraint; Hessian kernels follow the established host/device split |
| `thiserror` | 2.0.18 | `cintxRsError` incl. `BufferTooSmall` (`error.rs:66`) | Public library error surface (CLAUDE.md) |
| `bindgen` | 0.71.1 (workspace) | Vendor FFI symbol generation in `cintx-oracle/build.rs` | Already generates the `int*` allowlist regex; extend it for new symbols |
| `cc` | 1.2.x | Vendored libcint `cc::Build` (`cintx-oracle/build.rs`) | Hermetic oracle reference; add `deriv3.c`/`deriv4.c` |
| `approx` | (workspace) | `vendor_*` parity assertions at atol=1e-12 | Established oracle comparison helper |

**No `npm install` / version-bump step.** Verification command if any version question arises:
```bash
cargo tree -p cintx-cubecl | grep -E "cubecl|thiserror|bindgen|cc " 
```

### Vendored oracle C sources (the reference, on disk)
| File | Status in `cintx-oracle/build.rs` | Phase-25 families it provides |
|------|-----------------------------------|-------------------------------|
| `src/autocode/hess.c` | ✅ already at `build.rs:74,221` | HESS-01 (all 4 rank-9 1e) + HESS-02 (all 4 2e) |
| `src/autocode/int3c2e.c` | ✅ already at `build.rs:64,218` | HESS-03 (`int2c2e_ipip1`, `int3c2e_ipip1`, `int3c2e_ipip2`) |
| `src/autocode/deriv3.c` | ❌ ADD (rank 27) | HESS-04 3rd-order (`int1e_ipipipnuc`, `int1e_ipipiprinv`, …) |
| `src/autocode/deriv4.c` | ❌ ADD (rank 81) | HESS-04 4th-order (`int1e_ipipipiprinv`, …) |
| `src/rys_wheeler.c`, `src/eigh.c`, `src/find_roots.c`, `src/rys_roots.c`, `src/fmt.c` | ✅ already at `build.rs:200-204` | The vendor reference for the FND-02 nroots-sweep test (already compiles the full n>5 path) |

---

## FND-02 Derivation — the n>5 Rys host scheme (the long pole)

**Source of truth:** `libcint-master/src/rys_roots.c`, `rys_wheeler.c`, `eigh.c`, `find_roots.c`, `fmt.c`. All read directly (libcint 6.1.3, confirmed `CMakeLists.txt:3-5` = 6.1.3.0).

### Dispatch table (the control flow to mirror) — `CINTrys_roots`, `rys_roots.c:57-123`

The full root dispatcher has three regimes:
1. **`x <= SMALLX_LIMIT` (3e-7):** small-x polynomial fit `u = POLY_SMALLX_R0 + POLY_SMALLX_R1*x` (`:59-66`). Applies to ALL nroots.
2. **`x >= 35 + nroots*5`:** large-x asymptotic `u = rt/(x-rt)`, `w = POLY_LARGEX_WW * sqrt(PIE4/x)` (`:67-77`). Applies to ALL nroots.
3. **Otherwise: per-nroots branch (`:81-114`)** — this is the FND-02 target:

| nroots | libcint routine (lower==0 path) | Precision tier |
|--------|----------------------------------|----------------|
| 1–5 | `rys_root1..5` (polynomial, already ported in cintx `rys.rs`) | f64 polyfit |
| **6, 7** | `segment_solve(n, x, 0., …, 11, CINTrys_jacobi, CINTrys_schmidt)` | f64 Wheeler (jacobi) below x=11, else f64 Schmidt |
| **8** | `segment_solve(n, x, 0., …, 11, CINTrys_jacobi, CINTlrys_schmidt)` | f64 jacobi below x=11, else **long-double** Schmidt |
| **9** | `segment_solve(n, x, 0., …, 10, CINTlrys_jacobi, CINTlrys_laguerre)` | **long-double** jacobi / laguerre |
| **10, 11** | `segment_solve(n, x, 0., …, 18, CINTlrys_jacobi, CINTlrys_laguerre)` | long-double |
| **12** | `segment_solve(n, x, 0., …, 22, CINTlrys_jacobi, CINTlrys_laguerre)` | long-double |
| **≥13 (default)** | `segment_solve(n, x, 0., …, 50, CINTqrys_jacobi, CINTqrys_laguerre)` | `__float128` (quad) |

`segment_solve(n, x, lower, u, w, breakpoint, fn1, fn2)` (`rys_roots.c:42-56`): `if x <= breakpoint: fn1 else: fn2`; on error → falls back to `CINTqrys_schmidt`.

**Note `lower==0` always for Phase 25** — `CINTsr_rys_roots` (the short-range `lower != 0` table at `:145-265`) is for range-separated integrals (`PTR_RANGE_OMEGA`), which no Phase-25 family uses. The port can target `lower==0` only (the `gamma_inc_like`/`flocke_jacobi_moments` branch), leaving the SR path as a `lower != 0 → UnsupportedApi`/unimplemented stub. **This is the single biggest scope reducer for FND-02.**

### The Wheeler core — `CINTrys_jacobi` / `CINTlrys_jacobi`, `rys_wheeler.c:3678/3703`

For nroots 6..~13 with `lower==0`, the path is **Jacobi-moment Wheeler**, NOT Schmidt (Schmidt only triggers for large x above the per-nroots breakpoint). The Jacobi path:

1. **Modified moments** (`flocke_jacobi_moments`, `rys_wheeler.c:3361` for f64; `lflocke_jacobi_moments:3553` for long-double). Flocke's recipe (JCP 131, 064107): a Miller backward-recursion over precomputed constant tables `JACOBI_RN_PART2` (`rys_wheeler.c:243`), `JACOBI_SN` (`:335`), normalized by `fmt[0]/mu0`. For `t < SMALLX_LIMIT` it falls back to `naive_jacobi_moments` (direct `JACOBI_COEF · fmt[]`). Uses fixed `JACOBI_ALPHA`/`JACOBI_BETA` recurrence-coefficient tables (`rys_wheeler.c:33` / line ~3681-3682).
2. **Wheeler recursion** (`wheeler_recursion`, `rys_wheeler.c:3404`; `lwheeler_recursion:3587`): converts `(alpha, beta, moments)` → tridiagonal `(a, b)` (diagonal + off-diagonal of the Jacobi matrix). This is the "modified-moments → tridiagonal" step D-01 names.
3. **Tridiagonal symmetric eigensolve** (`rys_wheeler_partial`, `:3441`; `lrys_wheeler_partial:3625`): takes `sqrt(b[i])` as the off-diagonal, calls `_CINTdiagonalize(n, a, b+1, roots, c0)`, then `roots[i] = roots[i]/(1-roots[i])`, `weights[i] = c0[i*n]^2 * mu0`. The "root-polish" in D-01 is this eigenvalue→root transform; there is no separate Newton-polish for the Jacobi path (that's the Schmidt/`R_dnode` path below).

### The tridiagonal eigensolver — `_CINTdiagonalize`, `eigh.c`

**CRITICAL byte-identity finding.** `eigh.c` has two implementations behind `#ifdef LAPACK_FOUND`:
- `#ifdef LAPACK_FOUND` (`:28-47`): thin wrapper over LAPACK `dstemr_`.
- `#else` (`:51-1450+`): a **fully vendored, self-contained MRRR (`dstemr`-equivalent) symmetric-tridiagonal eigensolver** — `_dlarrk`, `_dlarrc`, `_dlasq2/4/5`, `_compute_eigenvalues`, `_dlarrf`, `_dlaneg`, etc. ~1400 lines of LAPACK-3.9.0-derived numerics.

cintx's oracle build does **NOT** define `LAPACK_FOUND` (`build.rs` has no `LAPACK_FOUND` define; verified by grep). **Therefore the vendor reference uses the `#else` vendored MRRR path.** For byte-identity, the cintx Rust port must reproduce **this vendored MRRR eigensolver**, not call a Rust LAPACK binding (whose last-ULP results would differ). This is the single largest and highest-risk piece of FND-02 — a faithful port of `eigh.c`'s `#else` branch.

### Long-double (`HAVE_SQRTL`) and quad (`HAVE_QUADMATH_H`) — both DISABLED in cintx's vendor build

`build.rs:169,171` explicitly disables `HAVE_SQRTL` and `HAVE_QUADMATH_H`. Consequences for the **vendor reference** (and hence the byte-identity target):
- `SQRTL` → `c99_sqrtl` (Babylonian refinement over f64 `sqrt`, `rys_roots.c:1776-1784`); `EXPL` → `c99_expl` (just `exp`, `:1792-1795`). So "long double" operations in `CINTlrys_jacobi`/`CINTlrys_laguerre` are effectively **f64 with a one-step sqrt refinement** in this build — NOT true 80-bit. The `long double` type still has 80-bit storage on x86-64 (the compiler uses hardware `long double`), but the transcendental helpers fall back to f64. **The port must replicate `c99_sqrtl`/`c99_expl` exactly where the long-double path is hit (nroots ≥ 8), not assume true `long double` math.**
- `__float128` quad path (`CINTqrys_*`, nroots ≥ 13) — guarded by `#ifdef HAVE_QUADMATH_H`, which is **disabled**. So for nroots ≥ 13, the vendor reference's quad routines are **not compiled**; `CINTrys_roots`'s `default:` branch calling `CINTqrys_jacobi` would be a link error if reached. **Practical implication: the vendor reference effectively caps at nroots ≤ 12 in this build.** The D-02 sweep "6..~13" should be validated as **6..12** (what the vendor actually compiles), with nroots 13+ documented as a forward edge the vendor itself doesn't support without quadmath. Confirm by attempting a nroots=13 vendor call early in Plan 1; if it link-errors or aborts, cap the sweep at 12.

### nroots 6,7 large-x Schmidt path — `CINTrys_schmidt` / `_rdk_rys_roots`, `rys_roots.c:1758/1699`

For x above the breakpoint (11 for n=6,7): `gamma_inc_like(fmt_ints, x, 2n)` (`fmt.c:206`) → `_rdk_rys_roots`: Schmidt-orthogonalize the FMT moments (`R_dsmit`, `:1643`), find polynomial roots (`_CINT_polynomial_roots` → `R_dnode` Newton iteration, `find_roots.c:243/19`), then `roots[k]=root/(1-root)`, `weights[k]=1/dum`. This is the "root-polish" (Newton via `R_dnode`) variant. Needed only for the large-x tail of nroots 6,7; on the H2O/STO-3G corpus (small x), the **Jacobi (Wheeler) path dominates** — but byte-identity requires both branches.

### Constant tables to embed (LE-f64 blobs, `roots_xw_data.rs` precedent per P19 D-14)

The Jacobi/Flocke path needs: `JACOBI_ALPHA`, `JACOBI_BETA`, `JACOBI_RN_PART2`, `JACOBI_SN`, `JACOBI_COEF` + `JACOBI_COEF_ORDER` (`rys_wheeler.c:33,243,335` and the lJACOBI_* long-double siblings). The polyfit small-x/large-x path needs `POLY_SMALLX_R0/R1/W0/W1`, `POLY_LARGEX_RT/WW` (in `polyfits.c`, already on disk). **Extract these as binary blobs via an xtask `gen-rys-tables` subcommand with a `--check` drift-gate**, exactly as Phase 19 did for the ECP `_sph_ine_tab` tables (memory: `project_first_gpu_family_2c2e`/P19 19-05 precedent). Do not hand-transcribe — drift-gate against the C source.

### Executor gate extension (D-02) — `executor.rs:140-142`

The `ang_momentum > 4 → max(l)>4` rejection at `executor.rs:140-142` is inside the **Validated4C1E** validator (`validated_4c1e_error`), NOT a global gate. The broader `nroots>5 → UnsupportedApi` guards are in the family launchers (`two_electron.rs:1459,1711,2065,2191`; `center_2c2e.rs:647,845`). **The FND-02 gate work touches BOTH:** (a) replace the `rys.rs:3255` panic with the Wheeler dispatch so `rys_roots_host(nroots≥6)` returns real roots; (b) raise/condition the launcher `nroots > MAX_DEVICE_NROOTS` guards to admit nroots up to the validated ceiling (12) **on the host path** — note the on-device comptime kernel stays capped at 5, so the launchers must route nroots≥6 to the host `fill_g_tensor_2e` path rather than the device kernel. Confirm the host path is selected for nroots≥6 (it already is for gradient families).

---

## FND-06 Derivation — fail-closed high-rank staging

**Boundary (D-04 single assertion site):** `cintx-runtime/src/planner.rs`:
- `parse_component_multiplier(component_rank: &str) -> Result<usize, …>` (`:403`) parses the rank string ("9"/"27"/"81") into the multiplier.
- `component_multiplier_for(descriptor)` (`:452`) is the per-family entry.
- `staging_elements_for_chunk(...)` (`:321`) and `try_alloc_staging(elements)` (`:341`) are the allocation boundary — `try_alloc_staging` already returns `Result<Vec<f64>, cintxRsError>` and is the OOM-safe-stop point. The existing test `try_alloc_staging_oom_safe_and_f32_lane_count_adequate` (`:1000`) is the template for the D-05 rank-81 OOM test.

**The single upfront assertion (D-04):** at the point where the staging `Vec` length is known (post `try_alloc_staging`), assert `staging.len() >= required_elements` where `required_elements = component_multiplier * per_component_elements`. Emit `cintxRsError::BufferTooSmall { required, provided }` (variant already exists, `error.rs:66`) on failure. This single point replaces all 19 per-element guards.

**The 19 scatter guards to strip (D-04)** — pattern `if dst < staging.len() { staging[dst] = … }` (verified at `one_electron.rs:6545`):

| File | Lines (from CONTEXT scout — VERIFY each against current source) |
|------|------|
| `one_electron.rs` | 6545, 6569, 6736, 6760, 6973, 7028 |
| `two_electron.rs` | 1600, 1641, 1845, 1886, 2173, 2231 |
| `center_3c2e.rs` | 2525, 2559, 2767, 2801 |
| `center_2c2e.rs` | 736, 761 |
| `f12.rs` | 1784 |
| `unstable/grids.rs` | 1521 |

After the upfront assertion proves `staging` large enough, each `if dst < staging.len() { staging[dst] = v }` becomes unconditional `staging[dst] = v`. **Verify line numbers before editing** — the codebase has churned (e.g. 260530-iiq touched `center_3c1e.rs`, 260530-9ay touched unstable derivatives). Use `grep -n "if dst < staging.len()"` per file at plan time.

**D-05 OOM test:** set a chunk memory limit below the rank-81 staging requirement (rank-81 × per-component elements × 8 bytes), drive a rank-81 family (`int1e_ipipipiprinv` or `int2e_ipip1ipip2`), assert `Err(BufferTooSmall { … })` (or `ChunkPlanFailed`) AND the output buffer is byte-for-byte untouched (no partial write). Aligns with CLAUDE.md "fallible allocation + typed failure + no partial writes."

---

## Family Reference (HESS-01..04) — verified from libcint source

All `ng[]` tuples and `component_rank` values read directly from the vendored autocode. The `ng[]` layout is `{i_inc, j_inc, k_inc, l_inc, nf_max?, prim_dim, has_3rd_center?, ncomp}` (last element = `component_rank`; first four = angular-momentum headroom per center).

### HESS-01 — `int1e` rank-9 (Cluster A) — source `hess.c` (already in build)
| Family | `ng[]` | component_rank | Headroom |
|--------|--------|----------------|----------|
| `int1e_ipipovlp` | (per hess.c; overlap-derivative, no Rys) | **9** | bra +2 |
| `int1e_ipipnuc` | `{2, 0, 0, 0, 2, 1, 0, 9}` | **9** | bra +2 (i_l+2) |
| `int1e_ipipkin` | (per hess.c; overlap-deriv engine) | **9** | bra +2 |
| `int1e_ipiprinv` | (per hess.c; rinv/Rys 1e path) | **9** | bra +2 |

**gout component order for `int1e_ipipnuc`** (verbatim from `hess.c:548-557`, the index map D-09 mandates copying):
```
gout[n*9+0] = s[0]   // (xx)
gout[n*9+1] = s[3]   // (yx)
gout[n*9+2] = s[6]   // (zx)
gout[n*9+3] = s[1]   // (xy)
gout[n*9+4] = s[4]   // (yy)
gout[n*9+5] = s[7]   // (zy)
gout[n*9+6] = s[2]   // (xz)
gout[n*9+7] = s[5]   // (yz)
gout[n*9+8] = s[8]   // (zz)
```
where `s[]` is computed from `g0/g1/g2/g3` partials (`G2E_D_I` applied twice). **This s→gout permutation is load-bearing — it is the 3×3 second-derivative tensor in libcint's column-major component order. Copy it verbatim per family** (each family's gout differs; read its own block).

- `int1e_ipipovlp`/`int1e_ipipkin` reuse the **overlap-derivative engine** (`one_electron.rs`, no Rys) — the second ∇ applied to the Phase-23 first-order overlap-derivative path.
- `int1e_ipipnuc`/`int1e_ipiprinv` ride the **nuclear/Rys 1e path** — these are the in-phase FND-02 consumers (nuclear-attraction Rys can elevate to nroots 6 on Hessian-elevated d-shells per D-03).

### HESS-02 — 2e Hessian set (Cluster B) — source `hess.c` (already in build)
| Family | `ng[]` | component_rank | Status |
|--------|--------|----------------|--------|
| `int2e_ipip1` | `{2, 0, 0, 0, 2, 1, 1, 9}` | **9** | Unstable sph-only stub exists (`lock:3327`) → re-home to stable (D-07) |
| `int2e_ipvip1` | `{1, 1, 0, 0, 2, 1, 1, 9}` | **9** | Unstable sph-only stub exists (`lock:3360`) → re-home to stable (D-07) |
| `int2e_ip1ip2` | `{1, 0, 1, 0, 2, 1, 1, 9}` | **9** | NEW (register fresh) |
| `int2e_ipip1ipip2` | `{2, 0, 2, 0, 4, 1, 1, 81}` | **81** | NEW (register fresh) — note: 4th-order 2e, rank 81 |

**D-07 re-home mechanics:** the unstable entries at `compiled_manifest.lock.json:3327` (`int2e_ipip1_sph`) and `:3360` (`int2e_ipvip1_sph`) have `canonical_family: "unstable::source::2e"`, `oracle_covered=false`, empty `component_rank`. **Delete these two entries** and add stable entries (cart + sph) with `component_rank=9`, a stable canonical family (e.g. `"2e"` or a new `"2e::hess"`), `oracle_covered=true` after parity passes. Wire the stable launcher in `two_electron.rs`, vendor FFI wrappers, and a `vendor_*` test. The unstable kernel module entries in `unstable/` are removed (per memory `feedback_new_family_surface_scope` + D-07: one canonical entry per symbol, no alias). The HESS-02 set elevates the `two_electron.rs` ERI engine — Hessian d-quartets are the primary corpus FND-02 trigger.

### HESS-03 — multi-center rank-9 (Cluster C) — source `int3c2e.c` (already in build)
| Family | `ng[]` | component_rank | Notes |
|--------|--------|----------------|-------|
| `int2c2e_ipip1` | `{2, 0, 0, 0, 2, 1, 1, 9}` | **9** | 2-center, ∇² on center 1 |
| `int3c2e_ipip1` | `{2, 0, 0, 0, 2, 1, 1, 9}` | **9** | 3-center, ∇² on bra center 1 |
| `int3c2e_ipip2` | `{0, 0, 2, 0, 2, 1, 1, 9}` | **9** | 3-center, ∇² on center 2 (**ket headroom** `k_inc=2` — note third tuple element, D-09 ket-side) |

### HESS-04 — 3rd/4th-order (Cluster D) — source `deriv3.c` (rank 27) + `deriv4.c` (rank 81), NOT in build yet

**Complete roster libcint 6.1.3 exports** (derived from `include/cint_funcs.h` — the "and siblings" set, deduped across `_cart`/`_sph`/`_spinor`/`_optimizer`):

*3rd-order (`deriv3.c`, `ng[]={3,0,0,0,3,1,0,27}`, rank **27**):*
- `int1e_ipipipnuc` — `<∇∇∇ i | NUC | j>`
- `int1e_ipipiprinv` — `<∇∇∇ i | RINV | j>`
- `int1e_ipipnucip` — `<∇∇ i | NUC | ∇ j>` (sibling: bra+2, ket+1)
- `int1e_ipiprinvip` — `<∇∇ i | RINV | ∇ j>`
- `int1e_ipippnucp` — `<∇∇ i | p NUC p | j>` style (verify gout)
- `int1e_ipipprinvp`
- `int1e_ipiprinvrip`, `int1e_rinvipiprip` — rinv siblings with mixed bra/ket
- (the `_spinor` variants register → `UnsupportedApi` per D-11)

*4th-order (`deriv4.c`, `ng[]={2,2,0,0,4,1,0,81}`, rank **81**):*
- `int1e_ipipipiprinv` — `<∇∇ i | RINV | ∇∇ j>` (bra+2 AND ket+2 — confirms D-09 deriv4 dual headroom)
- `int1e_ipiprinvipip` — `<∇∇ i | RINV | ∇∇ j>` alternate ordering
- siblings as exported in `deriv4.c`

**Plan-time action:** before Cluster D, run `grep -oE "int1e_[a-z0-9]*" deriv3.c deriv4.c | grep -E "ipipip|ipipipip" | sort -u` against `hess.c`-excluded names to lock the exact roster, then read each family's `ng[]` + gout block from its own definition. **CONTEXT names `int1e_ipipipnuc` + `int1e_ipipipiprinv` as the anchors; the full sibling set is the deriv3/deriv4 export list above.** Spinor variants exist in the header but are `UnsupportedApi` (D-11).

**ng headroom confirmation (D-09):** deriv3 raises **bra +3** (`{3,0,0,...}`) for `ipipip*nuc`; deriv4 raises **bra +2 AND ket +2** (`{2,2,0,0,...}`) for `ipipip iprinv`-style. Note the `nf_max` element (5th) = 3 for deriv3, 4 for deriv4 — the polynomial-order headroom. Copy the exact tuple per family.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| n>5 Rys roots/weights | Clean-room Golub-Welsch quadrature | Verbatim port of libcint's Flocke-moments→Wheeler→vendored-MRRR path (D-01) | Last-ULP / root-ordering divergence breaks atol=1e-12; explicitly rejected in D-01. |
| Tridiagonal symmetric eigensolve | A Rust LAPACK binding (`lapack`/`ndarray-linalg`) | Port `eigh.c`'s `#else` vendored MRRR (`_dlarrk`/`_dlasq2`/`_compute_eigenvalues`/…) | The vendor reference uses the `#else` path (no `LAPACK_FOUND`); a real LAPACK gives different last bits. |
| Jacobi modified moments | Direct Gaussian-quadrature moment integrals | Port `flocke_jacobi_moments` Miller-recursion over `JACOBI_RN_PART2`/`JACOBI_SN` tables | Flocke's recipe (JCP 131,064107) is what libcint emits; tables must match bit-for-bit. |
| Rys constant tables | Hand-transcribe from C | xtask `gen-rys-tables` + `--check` drift-gate (P19 `roots_xw_data.rs` precedent) | Transcription errors are silent; drift-gate catches them. |
| Per-element scatter bounds | Per-element `if dst < len` guards (the current silent-truncation risk) | ONE upfront `BufferTooSmall` assertion at the planner boundary (D-04) | Per-element guards silently drop trailing components (the exact bug 260530-9ay root-caused for unstable derivatives). |
| Hessian G-tensor | New ∇² recurrence from scratch | Compose the Phase-23 first-order `nabla1*`/`gout_ipN` engine twice (`G2E_D_I` applied to `g1→g3`) | Hessian = gradient engine applied twice; libcint's `hess.c` literally does `G2E_D_I(g3, g1, …)`. |
| Component order | Infer the 3×3 / 27 / 81 layout | Copy the `gout[n*K+c] = s[π(c)]` permutation verbatim per family from hess.c/deriv3.c/deriv4.c | The permutation is libcint's column-major derivative-tensor order; inferring it risks transpose (D-09). |

**Key insight:** This phase's correctness is almost entirely a function of *fidelity to the vendored C source already on disk*. The dominant failure mode is "close but not last-ULP" — which is exactly what byte-identity at atol=1e-12 catches. Port, don't reinvent.

---

## Common Pitfalls

### Pitfall 1: Porting against true `long double` / quad when the vendor build disabled them
**What goes wrong:** The port uses Rust `f128`/extended precision (or assumes hardware 80-bit `long double` transcendentals), but the vendor reference compiled `c99_sqrtl`/`c99_expl` (f64-backed) because `HAVE_SQRTL`/`HAVE_QUADMATH_H` are disabled (`build.rs:169,171`). Results diverge in the last bits for nroots ≥ 8.
**Why it happens:** The C source *looks* like it uses `long double`/`__float128`; the disabling is in the cintx build config, not the C.
**How to avoid:** Replicate `c99_sqrtl` (Babylonian refinement, `rys_roots.c:1776`) and `c99_expl` (= `exp`) exactly; treat the quad (nroots ≥ 13) path as not-compiled in the vendor build — cap the validated sweep at nroots 12.
**Warning signs:** nroots 8–12 parity fails at ~1e-14 (a few ULP) while nroots 6–7 pass.

### Pitfall 2: Confusing the Validated4C1E l-gate with the FND-02 gate
**What goes wrong:** Editing `executor.rs:140-142` (the `validated_4c1e_error("max(l)>4")` gate) thinking it's the global nroots gate; the real nroots ceilings are the `nroots > MAX_DEVICE_NROOTS`/`nroots > 5 → UnsupportedApi` guards in the family launchers.
**How to avoid:** FND-02 gate work = (a) replace `rys.rs:3255` panic, (b) route nroots≥6 to the host `fill_g_tensor_2e` path in launchers (not the device comptime kernel capped at 5), (c) optionally extend the 4c1e l-gate only if a 4c1e-family corpus case needs it.
**Warning signs:** Gate edit has no effect on the nroots-sweep test, or breaks an unrelated 4c1e test.

### Pitfall 3: component_rank truncation (the recurring P23/P24 bug)
**What goes wrong:** `component_rank` set below the true multiplier (e.g. 1 or 3 instead of 9/27/81) → planner under-allocates → launcher silently drops trailing components → parity "passes" on the first component, fails the rest, or zeros appear.
**Why it happens:** Cloning a lower-rank family's lock entry without updating the rank (exactly what 260530-9ay root-caused for unstable derivatives).
**How to avoid:** D-10 — set `component_rank` = the `ng[]` last element (9/27/81) verified from source; gate with a NON-SQUARE bra×ket block so a transposed/truncated layout cannot pass.
**Warning signs:** First component matches, later components are zero or mismatched.

### Pitfall 4: Transpose pass-through on square blocks
**What goes wrong:** Testing on a square (p×p) block hides a bra/ket transpose because the layout is transpose-symmetric (memory: `project_1e_gpu_port_scalar_only`).
**How to avoid:** D-09 — every `vendor_*` test uses a NON-SQUARE block (e.g. p×d). For deriv4 dual-headroom families, ensure the test exercises distinct bra and ket angular momenta.
**Warning signs:** Parity passes on a symmetric fixture, fails when bra≠ket.

### Pitfall 5: nroots≥13 quad path link error
**What goes wrong:** The D-02 sweep includes nroots 13, but `CINTqrys_jacobi`/`CINTqrys_laguerre` are `#ifdef HAVE_QUADMATH_H`-gated and not compiled → vendor call link-errors or aborts.
**How to avoid:** Cap the validated sweep at nroots 12 (the vendor's effective ceiling in this build); document 13+ as a forward edge. Probe with a nroots=13 vendor call at the start of Plan 1.
**Warning signs:** Linker error on `CINTqrys_*` or a vendor `exit()` when the sweep hits 13.

### Pitfall 6: Pre-existing vendor-gate lib-test failure resurfacing
**What goes wrong:** The `CINTshells_cart_offset[4]` cintx=8/vendor=0 lib-test failure (memory: `project_oracle_vendor_lib_tests_uncovered`, folded as the `oracle-cart-offset-vendor-zero` cross-link) re-surfaces under the Phase-25 vendor gate and is mistaken for a Phase-25 regression.
**How to avoid:** Confirm it is pre-existing (reproduce against a pre-phase-20 commit) per the folded todo; do NOT let it block the Phase-25 family gate. The Phase-25 family parity runs through `--test` integration, where it passes; the `--lib` unit context is the affected one.

---

## Code Examples (verified from vendored libcint source)

### libcint n>5 dispatch (the control flow to port host-side)
```c
// Source: libcint-master/src/rys_roots.c:97-114 (CINTrys_roots)
case 6: case 7:
    err = segment_solve(nroots, x, 0., u, w, 11, CINTrys_jacobi, CINTrys_schmidt); break;
case 8:
    err = segment_solve(nroots, x, 0., u, w, 11, CINTrys_jacobi, CINTlrys_schmidt); break;
case 9:
    err = segment_solve(nroots, x, 0., u, w, 10, CINTlrys_jacobi, CINTlrys_laguerre); break;
case 10: case 11:
    err = segment_solve(nroots, x, 0., u, w, 18, CINTlrys_jacobi, CINTlrys_laguerre); break;
case 12:
    err = segment_solve(nroots, x, 0., u, w, 22, CINTlrys_jacobi, CINTlrys_laguerre); break;
// default (>=13): CINTqrys_jacobi/laguerre — quadmath, NOT compiled in cintx oracle build
```

### Wheeler partial (moments → tridiagonal → eigensolve → root transform)
```c
// Source: libcint-master/src/rys_wheeler.c:3441-3477 (rys_wheeler_partial)
wheeler_recursion(n, alpha, beta, moments, a, b);     // modified moments -> (a,b)
for (i = 1; i < n; i++) { ...; b[i] = sqrt(b[i]); }   // off-diagonal = sqrt(beta)
int error = _CINTdiagonalize(n, a, b+1, roots, c0);   // symmetric tridiagonal eigensolve
for (i = 0; i < n; i++) {
    roots[i]   = roots[i] / (1 - roots[i]);            // eigenvalue -> Rys root
    weights[i] = c0[i * n] * c0[i * n] * mu0;          // first eigvec component^2 * mu0
}
```

### rank-9 gout permutation (copy verbatim per family — D-09)
```c
// Source: libcint-master/src/autocode/hess.c:548-557 (int1e_ipipnuc, gout_empty branch)
gout[n*9+0] = + s[0];  gout[n*9+1] = + s[3];  gout[n*9+2] = + s[6];
gout[n*9+3] = + s[1];  gout[n*9+4] = + s[4];  gout[n*9+5] = + s[7];
gout[n*9+6] = + s[2];  gout[n*9+7] = + s[5];  gout[n*9+8] = + s[8];
```

### current panic to replace (FND-02 entry point)
```rust
// Source: crates/cintx-cubecl/src/math/rys.rs:3250-3256 (rys_roots_host_f64)
match nroots {
    1..=5 => { /* polynomial fits, already ported */ }
    _ => panic!("rys_roots_host: nroots={nroots} > 5 not supported"),
    //   ^^^ replace with Wheeler dispatch (D-01)
}
```

### scatter guard to strip (FND-06)
```rust
// Source: crates/cintx-cubecl/src/kernels/one_electron.rs:6545 (one of 19 sites)
let dst = staging_comp_base + ii + jj * ni_sph;
if dst < staging.len() {                    // <-- strip after upfront assertion (D-04)
    staging[dst] = F::from_f64_lossy(sph_tmp[mj * nsi + mi]);
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Rys roots capped at nroots ≤ 5 (`rys.rs:3255` panic) | Host Wheeler/Jacobi for nroots 6..12 (FND-02) | This phase | Removes the high-l ceiling; no family `UnsupportedApi` purely for nroots>5 |
| Per-element `if dst < len` scatter guards (silent truncation) | Single upfront `BufferTooSmall` assertion (FND-06) | This phase | Fail-closed; eliminates the 260530-9ay class of silent-truncation bugs |
| `int2e_ipip1`/`ipvip1` as sph-only `unstable::source::2e` stubs | Stable raw-api entries, cart+sph, oracle_covered (HESS-02/D-07) | This phase | One canonical entry per symbol; promotes to the byte-identity gate |

**Deprecated/outdated:**
- The "Wheeler fallback deferred to Phase 10" comments throughout `rys.rs` (`:10,3247,3520,3534`) are stale — Phase 25 is where it lands. Update them.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The vendor build's effective nroots ceiling is 12 (quad path not compiled) | FND-02 / Pitfall 5 | If quadmath *is* somehow linked, the sweep could extend to higher nroots — verify with a nroots=13 vendor probe at Plan 1 start. Low risk: `build.rs:171` explicitly disables it. |
| A2 | Hessian families route through the host `fill_g_tensor_2e` path (not device comptime kernel) for nroots≥6, following the gradient precedent | Arch Map / FND-02 gate | If a Hessian launcher is wired to force the device kernel, nroots≥6 would hit `MAX_DEVICE_NROOTS`. Verify the launcher routing per family at plan time. Medium-low: gradient families (`int2e_ip1`) confirmed host-routed (`two_electron.rs:1516,1765`). |
| A3 | The full HESS-04 sibling roster is exactly the deriv3.c/deriv4.c `int1e_ipipip*`/`ipipipip*` exports | HESS-04 Roster | Missing/extra sibling → incomplete coverage. Mitigated by the plan-time `grep` lock step before Cluster D. CONTEXT names only the two anchors; the siblings are discretion (D-09 note). |
| A4 | `int1e_ipipovlp`/`ipipkin` use the no-Rys overlap-derivative engine (their `ng[]` 7th element = 0, no 3rd center) | HESS-01 | If they unexpectedly need Rys, FND-02 becomes a hard dependency for them too. Low: overlap/kinetic are inherently Rys-free; only nuc/rinv use Rys. |

---

## Open Questions (RESOLVED)

All three are resolved into Plan-01 / Plan-06 tasks; none is an open unknown blocking planning. Each disposition below is reflected in the authored plans.

1. **Exact vendor nroots ceiling (12 vs 13+)**
   - What we know: `HAVE_QUADMATH_H` disabled (`build.rs:171`) → `CINTqrys_*` not compiled; `default:` branch (nroots≥13) calls them.
   - What's unclear: whether a nroots=13 vendor call link-errors at build time (symbol absent) or aborts at runtime.
   - **RESOLVED:** Plan-01 Task-1 opens with a `vendor_CINTrys_roots(13, …)` probe; the D-02 sweep is capped at the highest nroots that returns, defaulting to 6..12. The executor l-gate upper bound is fixed only after the probe confirms the ceiling.

2. **Does the `eigh.c` `#else` MRRR port need full `dstemr` fidelity, or does a simpler symmetric-tridiagonal QL/QR suffice at atol=1e-12?**
   - What we know: the vendor reference uses the `#else` MRRR path; byte-identity is last-ULP sensitive.
   - What's unclear: whether QL-with-implicit-shifts (simpler) lands within 1e-12 of MRRR for the small (n≤12) tridiagonal matrices here, or whether the MRRR-specific bisection/RQI ordering matters.
   - **RESOLVED:** Plan-01 Task-1 is a design spike — faithful MRRR port is the default per D-01; a simpler symmetric-tridiagonal eigensolver is permitted ONLY if it passes the nroots-sweep at 1e-12, otherwise full MRRR. Default to faithful MRRR if the spike is inconclusive. (~1400-line port magnitude noted; the MRRR-eigensolver and Wheeler/Jacobi-moment portions are sub-divided in Plan-01.)

3. **Cluster D ng `nf_max` element semantics**
   - What we know: deriv3 `ng[4]=3`, deriv4 `ng[4]=4` (polynomial-order headroom).
   - What's unclear: whether the cintx planner/G-tensor sizing reads this element or derives it.
   - **RESOLVED:** Plan-06 mirrors how the existing rank-9 families (Phase 23 `ipovlpip`) populate the analogous field — traced and replicated per-family rather than assumed.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Vendored `libcint-master/` source | Oracle reference (all parity) | ✓ | 6.1.3.0 (`CMakeLists.txt:3-5`) | — |
| `cc` toolchain (C compiler) | `cintx-oracle` `cc::Build` | ✓ (workspace builds today) | — | — |
| `bindgen` | Vendor FFI symbol gen | ✓ | 0.71.1 | — |
| `hess.c` / `int3c2e.c` in build | HESS-01/02/03 oracle | ✓ already wired (`build.rs:74,64`) | — | — |
| `deriv3.c` / `deriv4.c` | HESS-04 oracle | ✗ (not in build) | on disk | ADD via `.file()` + allowlist (no fallback; required for HESS-04 gate) |
| `CINTX_ORACLE_BUILD_VENDOR=1` + `--features cpu` | Double-gated parity | ✓ (env/feature) | — | Without both, parity SILENTLY SKIPS (memory: `reference_oracle_vendor_parity_invocation`) |
| LAPACK / quadmath | (NOT used — disabled by design) | ✗ | — | Vendored MRRR `#else` path + `c99_sqrtl` (this is the intended config) |

**Missing dependencies with no fallback:** `deriv3.c`/`deriv4.c` must be added to the oracle build for HESS-04 (ROADMAP SC7) — but they are on disk, so this is a build-config edit, not a true blocker.

**Missing dependencies with fallback:** none material.

---

## Validation Architecture

> `workflow.nyquist_validation: true` in `.planning/config.json` — section included.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + `cargo test` (no nextest dep in workspace; oracle tests are integration tests under `crates/cintx-oracle/tests/`) |
| Config file | none (cargo-native); vendor gate via env + feature |
| Quick run command | `cargo test -p cintx-cubecl --lib` (kernel/math unit tests, fast) |
| Full suite command | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu` (vendor-gated parity) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FND-02 | nroots 6..12 roots/weights byte-identical vs vendor `CINTrys_roots` | integration (vendor) | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu rys_nroots_sweep` | ❌ Wave 0 (`tests/rys_nroots_sweep_parity.rs`) |
| FND-02 | host `rys_roots_host(6..12)` no longer panics | unit | `cargo test -p cintx-cubecl --lib rys_host_nroots_ge6` | ❌ Wave 0 |
| FND-06 | rank-81 staging under memory limit → typed OOM, no partial write | unit | `cargo test -p cintx-runtime --lib rank81_oom_no_partial_write` | ❌ Wave 0 (extend `planner.rs` test mod, template `:1000`) |
| FND-06 | upfront assertion fires on undersized staging | unit | `cargo test -p cintx-runtime --lib staging_buffer_too_small` | ❌ Wave 0 |
| HESS-01 | `int1e_ipip{ovlp,nuc,kin,rinv}` cart+sph atol=1e-12, non-square block | integration (vendor) | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu hess1e_ipip` | ❌ Wave 0 (`tests/hess1e_ipip_parity.rs`) |
| HESS-02 | `int2e_ipip1/ipvip1/ip1ip2/ipip1ipip2` cart+sph atol=1e-12 | integration (vendor) | `… cargo test … hess2e_ipip` | ❌ Wave 0 (`tests/hess2e_parity.rs`) |
| HESS-03 | `int2c2e_ipip1`, `int3c2e_ipip1/ipip2` cart+sph atol=1e-12 | integration (vendor) | `… cargo test … hess_multicenter_ipip` | ❌ Wave 0 |
| HESS-04 | 3rd/4th-order families cart+sph atol=1e-12, non-square + bra≠ket | integration (vendor) | `… cargo test … deriv34_ipipip` | ❌ Wave 0 (`tests/deriv34_parity.rs`) |
| ALL | `manifest-audit` green after lock edits | xtask | `cargo run -p xtask -- manifest-audit` (or `node .claude/get-shit-done/bin/gsd-tools.cjs` per memory) | ✓ exists |

### Sampling Rate
- **Per task commit:** `cargo test -p cintx-cubecl --lib` + the touched family's `vendor_*` test (gated).
- **Per wave merge:** full vendor-gated oracle suite (`CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu`) + `cargo test -p cintx-runtime --lib` (FND-06) + `manifest-audit`.
- **Phase gate:** full suite green before `/gsd-verify-work`; confirm worktree integration with `merge-base --is-ancestor` after each cluster wave (memory: worktree auto-merge inconsistent).

### Wave 0 Gaps
- [ ] `tests/rys_nroots_sweep_parity.rs` — FND-02 nroots 6..12 sweep vs vendor `CINTrys_roots` (D-02). **Highest priority — the FND-02 long-pole validation.**
- [ ] `rys.rs` unit test `rys_host_nroots_ge6` — host fn no longer panics, returns correct count.
- [ ] `planner.rs` test mod additions — `rank81_oom_no_partial_write` (D-05) + `staging_buffer_too_small` (D-04), template `planner.rs:1000`.
- [ ] `tests/hess1e_ipip_parity.rs`, `tests/hess2e_parity.rs`, `tests/hess_multicenter_ipip_parity.rs`, `tests/deriv34_parity.rs` — per-cluster `vendor_*` tests, NON-SQUARE blocks (p×d), bra≠ket for deriv4.
- [ ] xtask `gen-rys-tables` subcommand + `--check` drift-gate (P19 precedent) for the JACOBI_*/POLY_* constant blobs.
- [ ] `build.rs` edits: add `deriv3.c`/`deriv4.c` `.file()` + extend `allowlist_function` regex with all Phase-25 cart/sph symbols (HESS-01/02/03 symbols compile from already-built `hess.c`/`int3c2e.c` — only allowlist needed; HESS-04 needs both).

---

## Project Constraints (from CLAUDE.md)

- **Compatibility:** Target upstream libcint **6.1.3** result compatibility (verified: vendored tree is 6.1.3.0). Oracle byte-identity at atol=1e-12 is the gate.
- **Architecture:** CubeCL is the primary compute backend; host CPU stays limited to planning/validation/marshaling/oracle glue. → FND-02 Wheeler port is **host-side** (consistent: it's marshaling/planning numerics feeding the device G-tensor).
- **API Surface:** Safe Rust API first, raw compat second, optional C ABI third. → Phase 25 adds **manifest + RawApiId + kernel + vendor-FFI + oracle ONLY** (D-11); NO `cintx-capi` enum variants, NO legacy `cint*` wrappers (consistent with v1.4 per-family surface decision + memory `feedback_new_family_surface_scope`).
- **Error Handling:** Public library errors use `thiserror` v2 (`cintxRsError::BufferTooSmall` is the FND-06 typed failure); CLI/xtask/oracle harness use `anyhow`.
- **Verification:** Full coverage backed by the compiled manifest lock + feature-matrix CI + parity checks. → lock edits auto-sync `manifest-audit` (D-08).
- **Artifacts:** `/mnt/data` deliverables remain part of the workflow (oracle harness).
- **OOM-safe stop:** "Fallible allocation + typed failure + no partial writes" is a CLAUDE.md non-negotiable → exactly what FND-06 D-04/D-05 enforce.
- **GSD workflow:** all edits through a GSD command (this is a planned phase → `/gsd:execute-phase`).
- **CubeCL authoring:** `docs/manual/Cubecl/*.md` are the authoritative `#[cube]` rules — read before any device-kernel work (memory: `reference_cubecl_authoring_manuals`; top pitfalls: no plain-fn calls, no if-expr, `F::exp`/`F::sqrt`, u32/i32 only, no continue/break). Note: most Phase-25 work is host-side, so this primarily bounds any optional device port.

---

## Sources

### Primary (HIGH confidence — read directly from disk this session)
- `libcint-master/src/rys_roots.c` — `CINTrys_roots` dispatch (`:57-123`), `CINTsr_rys_roots` (`:145-265`), `rys_root1..5`, `R_dsmit` (`:1643`), `_rdk_rys_roots` (`:1699`), `CINTrys_schmidt`/`CINTlrys_schmidt` (`:1758/1851`), `c99_sqrtl`/`c99_expl` (`:1776/1792`).
- `libcint-master/src/rys_wheeler.c` — `CINTrys_jacobi`/`CINTlrys_jacobi`/`CINTlrys_laguerre` (`:3678/3703/3692`), `rys_wheeler_partial`/`lrys_wheeler_partial` (`:3441/3625`), `wheeler_recursion`/`lwheeler_recursion` (`:3404/3587`), `flocke_jacobi_moments`/`lflocke_jacobi_moments` (`:3361/3553`), JACOBI_* tables (`:33,243,335`).
- `libcint-master/src/eigh.c` — `_CINTdiagonalize` `#ifdef LAPACK_FOUND` wrapper (`:28-47`) vs `#else` vendored MRRR (`:51-1450+`).
- `libcint-master/src/fmt.c` — `gamma_inc_like`/`lgamma_inc_like` (`:206/248`), erfc-like variants.
- `libcint-master/src/find_roots.c` — `_CINT_polynomial_roots` (`:243`), `R_dnode` Newton (`:19`), `MXRYSROOTS=32` (`:11`).
- `libcint-master/src/autocode/hess.c` — `int1e_ipipnuc` gout + `ng[]={2,0,0,0,2,1,0,9}` (`:520-598`), all HESS-01/02 families.
- `libcint-master/src/autocode/deriv3.c` (rank 27, `ng[]={3,0,0,0,3,1,0,27}`), `deriv4.c` (rank 81, `ng[]={2,2,0,0,4,1,0,81}`).
- `libcint-master/src/autocode/int3c2e.c` — HESS-03 ng tuples.
- `libcint-master/include/cint_funcs.h` — full ipip/ipipip/ipipipip family roster.
- `libcint-master/CMakeLists.txt:3-5` — version 6.1.3.0.
- cintx codebase: `crates/cintx-cubecl/src/math/rys.rs:3244-3256` (panic), `:10` (stale defer comment); `crates/cintx-cubecl/src/kernels/two_electron.rs:95,417,760-768,1459,1516,1765,2065` (nroots formula, host/device root paths, guards); `crates/cintx-cubecl/src/kernels/center_2c2e.rs:521,647,845`; `crates/cintx-cubecl/src/executor.rs:135-142` (4c1e l-gate); `crates/cintx-runtime/src/planner.rs:321,341,403,452,509,1000-1007` (staging boundary + OOM test template); `crates/cintx-core/src/error.rs:66` (`BufferTooSmall`); `crates/cintx-oracle/build.rs:51-80,169-205,358` (source list, disabled defines, allowlist); `crates/cintx-ops/generated/compiled_manifest.lock.json:3327,3360` (unstable entries); `one_electron.rs:6545` (guard pattern).

### Secondary (project memory / prior phases)
- `.planning/phases/23-…/23-CONTEXT.md`, `24-…/24-CONTEXT.md` — registration recipe (D-08), transpose discipline (D-09), rank-truncation rule (D-10) precedents.
- `.planning/phases/19-…` — ECP K-Taylor host-first port precedent (D-01 analog) + table-blob/drift-gate xtask pattern.
- Project memory: `feedback_new_family_surface_scope`, `reference_oracle_vendor_parity_invocation`, `project_oracle_vendor_lib_tests_uncovered`, `reference_cubecl_authoring_manuals`, `feedback_worktree_auto_integration_inconsistent`, `project_unstable_derivative_ports` (the rank-truncation root-cause precedent).

### Tertiary (LOW confidence — none load-bearing)
- Flocke "Algorithm 954" / JCP 131,064107 (the Jacobi-moment recipe libcint cites) — referenced only to name the algorithm; the actual port follows the C source verbatim, not the paper.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new deps; all source files verified present/absent in the actual `build.rs`.
- FND-02 scheme: HIGH — full control flow + routine names + precision tiers read directly from `rys_roots.c`/`rys_wheeler.c`/`eigh.c`. The one residual uncertainty (MRRR fidelity vs simpler QL) is flagged as a Plan-1 spike, not an unknown fact.
- FND-06 boundary: HIGH — boundary functions, error variant, and 19 guard sites located in source.
- Family ranks/gout order: HIGH — `ng[]` tuples and gout permutations read verbatim from hess.c/deriv3.c/deriv4.c/int3c2e.c.
- HESS-04 sibling roster: MEDIUM — anchors verified; full sibling set derived from header grep, locked by the plan-time grep step (A3).
- nroots ceiling (12 vs 13): MEDIUM — strongly implied by disabled quadmath; confirm with a Plan-1 vendor probe (A1).

**Research date:** 2026-05-30
**Valid until:** Stable — the vendored libcint source is frozen at 6.1.3.0; cintx line numbers may drift (re-grep guard sites and anchors at plan time). 30 days for the cintx-side anchors; indefinite for the libcint-side derivations.

---

## RESEARCH COMPLETE

**Phase:** 25 - Group 2 — Hessian & Higher-Order Derivatives
**Confidence:** HIGH

### Key Findings
- **Primary oracle source is `hess.c` (HESS-01/02) + `int3c2e.c` (HESS-03), BOTH already in the cintx-oracle build** — only `deriv3.c`/`deriv4.c` (HESS-04) need adding. ROADMAP SC7's "add deriv3.c/deriv4.c" is correct but understates that HESS-01/02/03 need only the allowlist regex extended (their C already compiles).
- **FND-02 is a host-side port** of libcint's n>5 path: `CINTrys_jacobi`→`flocke_jacobi_moments`→`wheeler_recursion`→`_CINTdiagonalize`. The single highest-risk piece is porting `eigh.c`'s **`#else` vendored MRRR tridiagonal eigensolver** (cintx's build does NOT define `LAPACK_FOUND`), not a LAPACK binding. Long-double/quad are disabled (`build.rs:169,171`), so the effective vendor ceiling is **nroots ≤ 12** and "long double" is f64-with-`c99_sqrtl`-refinement.
- **All `ng[]` tuples and `component_rank` values verified from source:** rank-9 = `{2,0,0,0,2,1,*,9}`, rank-27 deriv3 = `{3,0,0,0,3,1,0,27}`, rank-81 deriv4 = `{2,2,0,0,4,1,0,81}` (confirms D-09 deriv4 bra+2 AND ket+2). gout component permutation (s→gout) extracted verbatim per family.
- **FND-06 infrastructure already exists:** `BufferTooSmall` variant (`error.rs:66`), `try_alloc_staging` boundary (`planner.rs:341`), OOM-test template (`planner.rs:1000`); 19 guard sites enumerated (re-grep at plan time).
- **HESS-02 D-07 re-home targets located:** unstable entries at `compiled_manifest.lock.json:3327/3360` to delete; `int2e_ipip1ipip2` is a **rank-81** 4th-order 2e family (not rank-9).

### File Created
`.planning/phases/25-group-2-hessian-higher-order-derivatives/25-RESEARCH.md`

### Confidence Assessment
| Area | Level | Reason |
|------|-------|--------|
| Standard Stack | HIGH | No new deps; build.rs source list verified directly |
| Architecture (host vs device, FND-02 scheme) | HIGH | Control flow + routine names read from vendored C |
| Family ranks/gout | HIGH | ng tuples + gout permutations verbatim from source |
| HESS-04 roster | MEDIUM | Anchors verified; full siblings locked by plan-time grep (A3) |
| nroots ceiling 12 vs 13 | MEDIUM | Strongly implied by disabled quadmath; Plan-1 vendor probe to confirm (A1) |

### Open Questions
1. nroots vendor ceiling (12 vs 13+) — Plan-1 vendor probe.
2. **`eigh.c` MRRR fidelity vs simpler symmetric-tridiagonal QL at 1e-12 — the key Plan-1 design spike** (default to faithful MRRR per D-01 if inconclusive).
3. Cluster D `ng[]` `nf_max` element semantics — mirror existing rank-9 family handling.

### Ready for Planning
Research complete. The planner can sequence Plan 1 (FND-02, host Wheeler + MRRR port + nroots-sweep + gate) and Plan 2 (FND-06, single assertion + 19-guard strip + rank-81 OOM test) as the merge-blocking foundations, then Clusters A→D low-rank-first via worktrees.
