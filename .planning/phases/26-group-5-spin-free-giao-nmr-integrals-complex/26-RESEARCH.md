# Phase 26: Group 5 (spin-free) — GIAO / NMR Integrals (complex) - Research

**Researched:** 2026-05-31
**Domain:** libcint-compatible GIAO/CG (gauge-including / magnetic-property) integral families; complex/imaginary output capability (FND-03); CubeCL generic-over-F kernels
**Confidence:** HIGH (roster, ranks, gout orders, signatures all derived directly from vendored libcint 6.1.3 source with file:line citations)

## Summary

Phase 26 brings the **spin-free 1e + 2e GIAO/CG families** to byte-identity (cart+sph) vs vendored libcint 6.1.3, and introduces the per-family complex/imaginary output capability (FND-03). All target families are **net-new** to the manifest (302 entries today, zero GIAO/CG). Every fact below is `[VERIFIED]` against `./libcint-master/src/autocode/intor{1,2,3,4}.c` and `src/cart2sph.c` (libcint 6.1.3, confirmed `CMakeLists.txt` 6.1.3).

**One finding reshapes the FND-03 framing and must be surfaced to the planner before locking the design.** The CONTEXT.md/Success-Criteria text says the cart/sph GIAO families have a `double complex *out` libcint symbol and that the vendor wrapper passes a `2×`-interleaved buffer. **This is not how libcint 6.1.3 works for cart/sph.** Every `int*_cart` / `int*_sph` symbol — GIAO included — has signature `double *out` (`include/cint_funcs.h:14` `CINTIntegralFunction` typedef) and writes a **plain real `double` buffer of size `nao_i × nao_j × component_rank` (1×, NOT 2× interleaved)**. The cart/sph C2S path is `c2s_cart_1e`/`c2s_sph_1e` which copy real doubles (`src/cart2sph.c:5820`, `:4833`). Complex (`double complex`, 2×-interleaved via `OF_CMPLX`) appears **only** on the `_spinor` path (`c2s_sf_1e`, `src/cart2sph.c:4869`), which is out of scope (D-11, spinor → `UnsupportedApi`).

What "purely imaginary" means concretely: the GIAO integral is mathematically `i × (real tensor)`. libcint's cart/sph symbols return only the **real magnitude of the imaginary part** as a real `double` buffer; the factor of `i` is implicit and applied by the caller (PySCF multiplies by `1j`). So byte-identity against the vendor symbol is a **real-vs-real** elementwise comparison of `component_rank` real components.

**Primary recommendation:** Treat FND-03's two halves as distinct and reconcile them in Plan 1:
1. **Vendor parity (GIAO-01/GIAO-02, D-05/D-14):** compare cintx's real output against libcint's real `double *out` — a `1×` (real) comparison, NOT `2×`. The vendor FFI wrappers are ordinary `double *out` wrappers identical to every existing moment/derivative wrapper; **no `*mut f64` len-2N reinterpretation is needed for the cart/sph vendor symbols** (revise D-05). The 2×-interleaved/`double complex` reinterpretation only applies if you ever bind a `_spinor` symbol — which you will not in Phase 26.
2. **Safe-API complex view (FND-03 spirit, D-03):** the cintx safe API can still expose these as `Complex<f64>` by materializing `re=0, im=value` from the real device output. This is a **cintx-side presentation choice**, decoupled from the vendor comparison. The manifest `complex_output` flag drives this materialization and the `complex_values()` gate (D-03), and the D-07 assertion (imag non-zero, real exactly zero) is checked on the cintx safe-API view — NOT on a 2×-interleaved vendor buffer.

This reconciliation keeps all D-01/D-02/D-03/D-04 intent (manifest-driven complex routing, comptime kernel hint, typed `Complex<f64>` view, generalized fail-closed contract) while correcting the mechanical D-05 assumption that the vendor symbol is complex. **Flag this to discuss-phase as Assumption A1 — it changes the vendor-FFI binding work from "2N reinterpret" to "plain real wrapper + cintx-side i-materialization."**

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| GIAO integral math (gout emitters, r_gauge×∇ tensor) | CubeCL device kernel (`#[cube]`, generic-F) | — | CLAUDE.md: CubeCL is primary compute backend; host CPU only plans/marshals |
| component_rank sizing + complex routing | Host planner (`build_output_layout`) | Manifest data | Data-driven from manifest `complex_output`+`component_rank` (D-01) |
| Complex/imaginary presentation (`Complex<f64>` view) | Host safe API (`api.rs complex_values()`) | — | cintx-side i-materialization; vendor output is real (see Summary) |
| Fail-closed flat-buffer contract | Host oracle (`compare.rs`) | — | Always-on contract gate (D-04) |
| Vendor byte-identity comparison | Host oracle (`vendor_ffi.rs` + `compare.rs`) | Vendored libcint (real `double*`) | Real-vs-real elementwise at atol=1e-12 |
| Gauge origin (`common_orig`) | Host env (`raw.rs` PTR_COMMON_ORIG, Phase 22) → device kernel | — | FND-01 already wired end-to-end |

## Standard Stack

No new external dependencies. All work is within the existing cintx workspace + vendored libcint. Confirmed from `CLAUDE.md` and `Cargo.lock`:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cubecl` | `0.10.0` (pinned) | GIAO `#[cube]` kernels, generic over `F` | CLAUDE.md primary backend; D-08 mandates generic-F + manual-first |
| `num-complex` | workspace-pinned | `Complex<f64>` safe-API view (D-03) | Already the established complex surface (`api.rs complex_values()`) |
| `thiserror` | `2.0.18` | Public lib errors (`UnsupportedApi` for spinor GIAO) | CLAUDE.md: thiserror v2 for library surface |
| `bindgen` | `0.71.1` (workspace) | Vendor FFI bindings for GIAO symbols | `build.rs` allowlist extension (D-10 step 4) |
| `cc` | `1.2.x` | Vendored libcint build (`intor4.c` already in source list) | Oracle hermetic build |

**Installation:** none — no new crates. `[VERIFIED: Cargo.lock + CLAUDE.md stack table]`

### Vendored libcint source (source of truth)
- Path: `/home/user/Documents/workspace/cintx/libcint-master/` (also worktree mirror).
- Version: **6.1.3** `[VERIFIED: CMakeLists.txt MAJOR=6 MINOR=1 PATCH=3]`.
- GIAO autocode files: `src/autocode/intor1.c`, `intor2.c`, `intor3.c`, `intor4.c`.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FND-03 | Complex/imaginary output capability — `complex_interleaved` set per-family from driver routing (not rep string); `assert_flat_buffer_contract` fires on flag; staging `2×ncomp`; safe-API round-trip without silent zeroing | Manifest `complex_output` field design + planner/compare/api edit sites below. **CAVEAT A1:** vendor cart/sph output is real `1×`, not `2×` — the `2×` staging applies to the cintx safe-API complex view (re/im interleaved), not the vendor comparison buffer. |
| GIAO-01 | Spin-free 1e GIAO/CG families match atol=1e-12 (cart+sph) | Complete roster + ranks + gout orders below; closest-clone families identified |
| GIAO-02 | 2e GIAO families (`int2e_g1`, `int2e_gg1`, `int2e_ig1`, `int2e_giao_*`) match atol=1e-12 | 2e roster + ranks + gout orders; `intor4.c`/`intor2.c` already in oracle build |

## User Constraints (from CONTEXT.md)

### Locked Decisions (D-01..D-14, verbatim intent)
- **D-01** — manifest per-family flag (`complex_output: bool` or `output_complex_multiplier`) drives complex routing in `build_output_layout`, **replacing** the `rep==Spinor` coupling at `planner.rs:323`. NOT a code-side allowlist.
- **D-02** — same manifest field flows as a **comptime hint** into the `#[cube]` kernel (one field drives host contract/staging AND device layout).
- **D-03** — safe API returns `num_complex::Complex<f64>` view via existing `complex_values()` gate.
- **D-04** — `assert_flat_buffer_contract` (`compare.rs:270`) generalized to honor `complex_interleaved=true` for **any** representation; always-on fail-closed.
- **D-05** — *(REVISE — see A1)* bind GIAO `out` as `*mut f64` len-2N. **Research finding: cart/sph GIAO symbols are `double *out` real `1×`; the 2N reinterpretation does NOT apply to the in-scope cart/sph symbols.**
- **D-06** — non-zero gauge-origin fixture (`build_h2o_sto3g_common_orig`, Phase 22) gates every family; must be non-square bra×ket (D-12).
- **D-07** — prove imaginary lands AND real half is zero. *(Applies to the cintx safe-API `Complex<f64>` view, not a vendor 2×-buffer — see A1.)*
- **D-08** — generic-over-`F` `#[cube]` kernels; **executor MUST read CubeCL manual pages before writing kernel code** (Generics, Algebra, Basic-Operations, Conditionals).
- **D-09** — Plan 1 = FND-03 (merges first); Cluster A = 1e; Cluster B = 2e; worktree parallelization, verify integration with `merge-base --is-ancestor`.
- **D-10** — 5-step registration recipe (manifest lock entry + `cargo build -p cintx-ops` → RawApiId → launcher dispatch → vendor FFI allowlist+wrapper → vendor_* test). Lock auto-syncs `manifest-audit`.
- **D-11** — surface scope: manifest + RawApiId + kernel + vendor-FFI + oracle only. No capi/legacy wrappers. Spinor reps → `UnsupportedApi`.
- **D-12** — gout component order verbatim from libcint; ket-side headroom; non-square bra×ket gate.
- **D-13** — `component_rank` = true output multiplier, derived from libcint source (NOT guessed); too-low truncates.
- **D-14** — per-family byte-identity atol=1e-12, cart+sph, every component, vendor_* double-gated on `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`.

### Claude's Discretion
- Exact enumerated roster (resolved below).
- Exact `component_rank` + gout order per family (resolved below).
- Manifest field name/shape for complex flag.
- vendor_* corpus shell-tuple selection (subject to non-square + non-zero-gauge).
- One parameterized `#[cube]` entry with comptime op-kind vs per-family launchers.

### Deferred Ideas (OUT OF SCOPE)
- **GIAO×σ spinor slice (GIAO-03, Phase 30):** `int1e_spg*`, `int1e_spgnucsp`, `*_sa10*`, `int2e_cg_sa10*`/`giao_sa10*`. **CONFIRMED by source:** the `sa10*` families have `ng[POS_E1]=4` (4-component spin block `gc_x/gc_y/gc_z/gc_1`) requiring the FND-05 `c2s_si` transform.
- All spinor GIAO representations → `UnsupportedApi` this phase.
- `Complex64` repr-C FFI binding (D-05 uses simpler real wrappers — reinforced by A1).

## Exact Roster (derived from libcint 6.1.3 source)

`[VERIFIED: grep of int*_cart definitions across src/autocode/intor{1,2,3,4}.c]`

### GIAO-01 — Spin-free 1e (IN SCOPE) — 11 families × {cart, sph} = 22 symbols

`ng[POS_E1]` (= `ng[5]`) is **1** for all of these (spin-free); `component_rank` = `ng[TENSOR]` (= `ng[7]`). `[VERIFIED: src/cint_config.h.in:22 POS_E1=5, :25 TENSOR=7; src/g1e.c:49-50]`

| # | Family (operator name) | cart/sph symbols | `ng[]` | **component_rank** | gout def (file:line) | Closest manifest clone |
|---|------------------------|------------------|--------|--------------------|----------------------|------------------------|
| 1 | `int1e_govlp` | `_cart`, `_sph` | `{1,0,0,0,1,1,1,3}` | **3** | `intor3.c` (`CINTgout1e_int1e_govlp`) | `int1e_r` (rank 3, 1e) |
| 2 | `int1e_gnuc` | `_cart`, `_sph` | `{1,0,0,0,1,1,0,3}` | **3** | `intor3.c` | `int1e_r` |
| 3 | `int1e_igovlp` | `_cart`, `_sph` | `{1,0,0,0,1,1,1,3}` | **3** | `intor1.c:` `CINTgout1e_int1e_igovlp` (gout: cross-product of r_i−r_j with ∇) | `int1e_r` |
| 4 | `int1e_ignuc` | `_cart`, `_sph` | `{1,0,0,0,1,1,0,3}` | **3** | `intor1.c` | `int1e_r` |
| 5 | `int1e_igkin` | `_cart`, `_sph` | `{1,2,0,0,3,1,1,3}` | **3** | `intor1.c` | `int1e_ipkin` (rank 3, ∇-decorated kinetic) |
| 6 | `int1e_a01gp` | `_cart`, `_sph` | `{2,2,0,0,3,1,0,9}` | **9** | `intor1.c:` `CINTgout1e_int1e_a01gp` (27 `s[]` → 9 gout, full r_gauge×∇ tensor) | `int1e_rr` (rank 9, 1e) |
| 7 | `int1e_ia01p` | `_cart`, `_sph` | `{1,2,0,0,2,1,0,3}` | **3** | `intor1.c:81` `CINTgout1e_int1e_ia01p` (gout: `s[5]-s[7]`, `s[6]-s[2]`, `s[1]-s[3]`) | `int1e_r` |
| 8 | `int1e_cg_irxp` | `_cart`, `_sph` | `{0,2,0,0,2,1,1,3}` | **3** | `intor1.c:` `CINTgout1e_int1e_cg_irxp` (gout: `s[5]-s[7]`, `s[6]-s[2]`, `s[1]-s[3]`) | `int1e_r` |
| 9 | `int1e_giao_irjxp` | `_cart`, `_sph` | `{0,2,0,0,2,1,1,3}` | **3** | `intor1.c` (identical gout to `cg_irxp`) | `int1e_r` |
| 10 | `int1e_cg_a11part` | `_cart`, `_sph` | `{1,2,0,0,2,1,0,9}` | **9** | `intor1.c:` `CINTgout1e_int1e_cg_a11part` (gout: `s[0..8]` direct) | `int1e_rr` |
| 11 | `int1e_giao_a11part` | `_cart`, `_sph` | `{1,2,0,0,2,1,0,9}` | **9** | `intor1.c` (identical structure to `cg_a11part`) | `int1e_rr` |

**Roster note:** the CONTEXT wildcard `int1e_ig*` resolves to exactly `igovlp`, `ignuc`, `igkin` (the only spin-free `ig`-prefixed families; `int1e_ia01p` is `ia01p` not an `ig*`). The `int1e_cg_*`/`int1e_giao_*` wildcards in scope resolve to `{cg_irxp, cg_a11part}` and `{giao_irjxp, giao_a11part}` — the `*_sa10*` variants are deferred (Phase 30).

### GIAO-02 — Spin-free 2e (IN SCOPE) — 3 families × {cart, sph} = 6 symbols

`[VERIFIED: src/autocode/intor4.c, intor2.c]`

| # | Family | cart/sph | `ng[]` | **component_rank** | gout def | Closest 2e clone |
|---|--------|----------|--------|--------------------|----------|------------------|
| 1 | `int2e_g1` | `_cart`, `_sph` | `{1,0,0,0,1,1,1,3}` | **3** | `intor4.c:1255` `CINTgout2e_int2e_g1` (gout: `c[1]s[2]−c[2]s[1]`, …) | `int2e` arity-4 + rank-3 deco (e.g. `int2e_ip1`) |
| 2 | `int2e_ig1` | `_cart`, `_sph` | `{1,0,0,0,1,1,1,3}` | **3** | `intor2.c:19` `CINTgout2e_int2e_ig1` (gout: `−c[1]s[2]+c[2]s[1]`, …) | `int2e_ip1` |
| 3 | `int2e_gg1` | `_cart`, `_sph` | `{2,0,0,0,2,1,1,9}` | **9** | `intor2.c:148` `CINTgout2e_int2e_gg1` (9 components, 2nd-order gauge tensor) | `int2e_ipip1` (rank 9, arity-4) |

**Roster note:** the CONTEXT wildcard `int2e_giao_*` resolves to ONLY `int2e_giao_sa10sp1` and `int2e_giao_sa10sp1spsp2` in libcint 6.1.3 — **both have `ng[POS_E1]≠1` (spin block) and are DEFERRED to Phase 30** (GIAO-03). So GIAO-02's concrete spin-free scope is exactly `{int2e_g1, int2e_ig1, int2e_gg1}`. The `int2e_g1g2` and `int2e_g1spsp2` families also exist but `g1g2` is a 2nd-gauge-on-both-electrons family and `g1spsp2` carries `spsp2` (spin) — **confirm with planner whether `int2e_g1g2` (no spin) is in or out**; CONTEXT lists only `g1`/`gg1`/`ig1`/`giao_*` so `g1g2` appears out of scope for GIAO-02 as written. `[ASSUMED: g1g2 out of scope based on literal CONTEXT roster — confirm]`

### Deferred (Phase 30 — verify NOT registered as parity targets here)
`int1e_*_sa10sp`, `*_sa10nucsp`, `*_sa10sa01` (both `cg_` and `giao_` prefixes), `int2e_giao_sa10sp1*`, `int2e_g1spsp2`. All have `ng[POS_E1]=4` (spin). `[VERIFIED: ng tuples above]`

## gout Component-Index Order (copy VERBATIM per D-12)

The libcint cart/sph output is a real `double` buffer; `gout[n*rank + k]` for component `k`, basis-pair `n`. Component order per family (the planner/kernel author copies these exactly):

**Cross-product (rank 3) families** — order is (x, y, z) of a curl/cross:
- `int1e_igovlp` `[VERIFIED: intor1.c gout]`: `[0]=−c[1]s[2]+c[2]s[1]`, `[1]=−c[2]s[0]+c[0]s[2]`, `[2]=−c[0]s[1]+c[1]s[0]` where `c = r_i − r_j`, `s = ⟨∇φ_i|φ_j⟩` x/y/z.
- `int2e_g1` `[VERIFIED: intor4.c:1255]`: `[0]=c[1]s[2]−c[2]s[1]`, `[1]=c[2]s[0]−c[0]s[2]`, `[2]=c[0]s[1]−c[1]s[0]`.
- `int2e_ig1` `[VERIFIED: intor2.c:19]`: `[0]=−c[1]s[2]+c[2]s[1]`, `[1]=−c[2]s[0]+c[0]s[2]`, `[2]=−c[0]s[1]+c[1]s[0]` (sign-flipped vs `g1`).
- `int1e_ia01p`, `int1e_cg_irxp`, `int1e_giao_irjxp` `[VERIFIED]`: `[0]=s[5]−s[7]`, `[1]=s[6]−s[2]`, `[2]=s[1]−s[3]` (these three share identical gout).

**Rank-9 tensor families** — order is row-major 3×3 (r_gauge index outer, ∇ index inner):
- `int1e_a01gp` `[VERIFIED: intor1.c, gout[n*9+0..8]]`: full `c×(∇⊗∇)` contraction, 27 intermediate `s[]` → 9 outputs (see source for exact `c[]·s[]` linear combos `[0]=c[1]s[23]−c[2]s[14]−c[1]s[25]+c[2]s[16]`, etc.).
- `int1e_cg_a11part`, `int1e_giao_a11part` `[VERIFIED]`: direct `gout[n*9+k]=s[k]` for k=0..8.
- `int2e_gg1` `[VERIFIED: intor2.c:148]`: 9 components, e.g. `[0]=−c[4]s[8]+2c[5]s[7]−c[8]s[4]` (2nd-order gauge tensor with `c[0..8]` from the symmetric r_gauge dyad).

**The kernel author MUST transcribe these `c[]·s[]` expressions exactly from the cited gout functions** — they are not derivable by inspection and a sign or index error fails byte-identity. The planner should attach the exact source line range per family to each kernel task.

## Manifest `complex_output` Flag — field design + edit sites

### Schema location `[VERIFIED]`
- Source of truth: `crates/cintx-ops/generated/compiled_manifest.lock.json` (302 entries; full schema entries carry `arity`, `component_rank` (string), `forms`, `helper_kind`, `representation`, etc.).
- Generated artifacts: `crates/cintx-ops/src/generated/api_manifest.{rs,csv}` (CSV header at line 1 lists `component_rank` as col 7).
- Entry struct: `ManifestEntry` at `crates/cintx-ops/src/resolver.rs:99-114` — `component_rank: &'static str`.

### Recommended field shape
**`complex_output: bool`** (mirrors how `component_rank` is a per-entry scalar). Rationale: the multiplier is always exactly 2 when complex (re/im), so a bool is sufficient and the planner derives `complex_multiplier = if complex_output {2} else {1}`. An `output_complex_multiplier: u32` is over-general for this phase. Either satisfies D-01. **Add the field to: (a) the lock.json entry schema, (b) `ManifestEntry` struct in `resolver.rs`, (c) the CSV column + `api_manifest.rs` generated rows** — `cargo build -p cintx-ops` regenerates (a)→(b)/(c). `manifest-audit` auto-syncs (D-10).

### FND-03 edit sites + blast radius (`grep complex_interleaved | complex_multiplier | complex_values`)
| Site | File:line `[VERIFIED]` | Change |
|------|------------------------|--------|
| SET `complex_multiplier` | `planner.rs:319-323` (`if matches!(rep, Spinor){2}else{1}`) | **Replace** with `descriptor.entry.complex_output` read (D-01). Spinor still works because spinor entries set `complex_output=true`. |
| `complex_interleaved` field | `planner.rs:64` (`OutputLayoutMetadata.complex_interleaved`) | unchanged; now set from `complex_multiplier == 2` which is manifest-derived |
| `complex_interleaved` field | `api.rs:577` (`IntegralTensor.complex_interleaved`) | unchanged; populated from layout |
| `complex_values()` gate | `api.rs:604-612` | unchanged — already gates on `complex_interleaved`; exposes the `Complex<f64>` view (D-03). For GIAO the buffer is `[0,im,0,im,…]` materialized cintx-side (re=0). |
| `assert_flat_buffer_contract` | `compare.rs:270-285` | **Generalize** (D-04): currently the complex branch is `if fixture.representation == "spinor"`. Change to `if fixture.complex_interleaved` so cart/sph complex families are honored; a complex family staged real-only fails `values.len() % 2 != 0` or the len check. Make always-on (remove any debug-only gate). |
| Other `complex_interleaved` reads | `tensor.rs:12`, `layout.rs:13` | propagation only; verify they read the field, not `rep` |

**Blast radius:** the rep-driven path is referenced ONLY at `planner.rs:319-323` (the SET) and `compare.rs:279` (the spinor branch). No other code keys complex behavior off `Representation` directly. Replacing the SET + generalizing the contract are the two load-bearing edits. **Risk: existing spinor families must continue to set `complex_output=true` in the manifest** or they regress to real sizing — Plan 1 must backfill the flag on the existing spinor entries (every `*_spinor` row → `complex_output=true`) as part of the schema migration.

## Vendor FFI — REVISED binding strategy (A1)

`[VERIFIED: include/cint_funcs.h:14 CINTIntegralFunction typedef = double *out; src/cart2sph.c:5820 c2s_cart_1e real-double copy]`

**The cart/sph GIAO vendor symbols are ordinary `double *out` real wrappers — identical to the 182 existing wrappers in `vendor_ffi.rs`.** Add, per symbol:
1. bindgen `allowlist_function` regex extension in `crates/cintx-oracle/build.rs:374` — append `|int1e_govlp_cart|int1e_govlp_sph|…|int2e_gg1_cart|int2e_gg1_sph` (22 + 6 = 28 symbols).
2. a `vendor_int1e_govlp_sph(out: &mut [f64], …)` wrapper following the `vendor_int1e_r_*` pattern (`vendor_ffi.rs:21+`), `out` sized `nao_i × nao_j × component_rank` real elements.
3. compare cintx real output vs vendor real output elementwise at atol=1e-12 (D-14).

**Do NOT bind these as `*mut f64` len-2N** — there is no `double complex` cart/sph symbol to match. The D-05 2N-reinterpretation is correct only for `_spinor` symbols (out of scope). The cintx-side `2×` staging (D-01) is the safe-API complex view, materialized AFTER the device kernel writes real components, and compared to the vendor's `1×` real buffer by taking cintx's imaginary half (the real half must be exactly zero, D-07).

**`intor4.c` and `intor2.c` are already in the oracle `cc::Build` source list** `[VERIFIED: build.rs:62,:229]` — no build-source change; only allowlist + wrappers.

## Non-zero Gauge-Origin Fixture (D-06/D-07)

`[VERIFIED: crates/cintx-oracle/tests/common_orig_roundtrip.rs:15, moment_*_parity.rs]`

- Reuse `build_h2o_sto3g_common_orig()` (`cintx_oracle::fixtures`) — the Phase 22 non-zero-gauge fixture already used by all moment parity tests. `COMMON_ORIG_FIXTURE_ORIGIN` is the non-zero origin.
- **Non-square requirement (D-12):** select a non-square bra×ket shell tuple (e.g. p×d) so a transposed layout cannot pass. The moment tests already exercise this corpus; mirror their shell-tuple selection.
- **D-07 assertion (on the cintx safe-API `Complex<f64>` view):**
  - imaginary half **non-zero** somewhere (the integral landed, not silently zeroed — FND-03 core).
  - real half **exactly zero** (`== 0.0`, not approx) — catches a kernel that accidentally writes real content.
  - both halves byte-identical to expectation: imag == libcint's real `double` output (elementwise, atol=1e-12), real == 0.
- New test files: `tests/giao_1e_parity.rs` (Cluster A), `tests/giao_2e_parity.rs` (Cluster B), each double-gated `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`. A dedicated `tests/giao_complex_roundtrip.rs` proves the FND-03 safe-API round-trip on `int1e_igovlp` (D-03).

## CubeCL Kernel Authoring (D-08 — manual-first, MANDATORY)

`[VERIFIED: docs/manual/Cubecl/ mirror present]`

The executor **MUST read these before writing any `#[cube]` code** (user directive 2026-05-31):
- Canonical: `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md`
- In-repo mirror: `docs/manual/Cubecl/` — present files confirmed: `Cubecl_generics.md`, `Cubecl_algebra.md`, `Cubecl_basic_operations.md`, `Cubecl_conditionals.md`, plus `cubecl_error_solution_guide/`.
- Load-bearing: **Generics** (`Float`/`Numeric` bounds for generic-over-`F`), **Algebra** (`F::exp`/`F::sqrt`/`F::sin` for gauge phase/Gaussian terms), **Basic Operations**, **Conditionals** (boundary checks).

Standing `#[cube]` pitfalls (project memory `reference_cubecl_authoring_manuals.md`):
- No plain-Rust-fn calls inside `#[cube]` (must be `#[cube]` fns).
- No `if`-expressions (use statement form).
- `u32`/`i32` only for integer types.
- No `continue`/`break`.
- The Phase-24 moment kernel at `crates/cintx-cubecl/src/kernels/one_electron.rs:7287-7289` is the `common_orig`-consuming precedent the 1e GIAO kernels mirror; 2e GIAO extends `kernels/two_electron.rs`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Complex view from interleaved buffer | Manual `[2i]`/`[2i+1]` pairing | `IntegralTensor::complex_values()` (`api.rs:604`) | Already typed `Vec<Complex<F>>`, `#[repr(C)]` reinterpretation |
| Gauge-origin env plumbing | New env slot | `plan.operator_env_params.common_orig` (Phase 22, FND-01) | Fully wired + tested |
| gout component math | Re-derive r_gauge×∇ tensor | Transcribe verbatim from cited libcint gout fns | D-12/D-13 — derivation errors fail byte-identity silently |
| Manifest sizing | Hardcode rank in kernel | Manifest `component_rank` + new `complex_output` | Data-driven, auto-syncs manifest-audit |
| Vendor complex binding | `repr(C) Complex64` FFI | Plain `double *out` wrapper (cart/sph are real) | A1 — no complex cart/sph symbol exists |

**Key insight:** Phase 26's "complex" work is almost entirely **host-side presentation + manifest plumbing**, not new device math complexity. The device kernels emit real components (exactly like the Phase-24 moments); the complex framing is a safe-API view.

## Common Pitfalls

### Pitfall 1: Assuming the vendor symbol is `double complex` (A1)
**What goes wrong:** binding GIAO cart/sph as `*mut f64` len-2N and passing a 2×-interleaved buffer to libcint → the vendor writes only `1×` real, leaving the imaginary slots garbage; comparison spuriously fails or "passes" on uninitialized memory.
**Why:** CONTEXT/Success-Criteria text says `double complex *out`; libcint 6.1.3 cart/sph is `double *out`.
**Avoid:** bind real `double *out` `1×`; do the i-materialization cintx-side.
**Warning signs:** vendor wrapper allocating `2*n` and reading `out[2i+1]`.

### Pitfall 2: component_rank set too low (D-13)
**What goes wrong:** `int1e_a01gp`/`a11part`/`int2e_gg1` are rank **9**, not 3; setting 3 silently truncates 6 components.
**Avoid:** copy `ng[7]` exactly from the cited `ng[]` tuples (3 for most, 9 for a01gp/cg_a11part/giao_a11part/gg1).
**Warning signs:** output buffer 3× too small; trailing-component mismatches at the gate.

### Pitfall 3: Registering deferred spin families (Phase 30 leak)
**What goes wrong:** the `int2e_giao_*` wildcard resolves to `sa10sp1*` (spin) — registering them here as spin-free parity targets fails (they need `c2s_si`/FND-05).
**Avoid:** GIAO-02 spin-free scope is exactly `{int2e_g1, int2e_ig1, int2e_gg1}`; spinor reps → `UnsupportedApi` (D-11).
**Warning signs:** `ng[POS_E1]=4` on a family you're treating as spin-free.

### Pitfall 4: OperatorId shift breaks hardcoded test consts
**What goes wrong:** positional manifest ordering — adding 28 GIAO rows re-points any `OperatorId::new(<int>)` / `_OPERATOR_ID: u32 = N` test const at a different family → `InvalidShellTuple` arity mismatch.
**Avoid:** resolve by symbol name; re-grep these consts after registration (project memory).

### Pitfall 5: Forgetting to backfill `complex_output=true` on existing spinor entries
**What goes wrong:** replacing the `rep==Spinor` SET with manifest read regresses existing spinor families to real sizing if their entries lack the new flag.
**Avoid:** Plan 1 migration sets `complex_output=true` on all `*_spinor` rows.

## Runtime State Inventory

Not a rename/refactor phase — net-new families only. **None — verified: all 28 symbols are absent from the manifest (302 entries, zero GIAO/CG `[VERIFIED]`); no stored data, live config, OS state, secrets, or stale build artifacts reference them.** The only schema migration is adding the `complex_output` column, regenerated by `cargo build -p cintx-ops` (D-10).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Vendored libcint source | GIAO-01/02 parity | ✓ | 6.1.3 | — |
| `intor4.c`/`intor2.c` in oracle build | GIAO-02 | ✓ (already in `cc::Build`) | — | — |
| CubeCL CPU backend | kernel exec + parity | ✓ (`--features cpu`) | 0.10.0 | — |
| Non-zero gauge fixture | every gate (D-06) | ✓ (`build_h2o_sto3g_common_orig`) | — | — |
| CubeCL manual | D-08 kernel authoring | ✓ (in-repo mirror + canonical) | — | — |

No missing dependencies.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (cargo test / nextest), oracle integration tests under `crates/cintx-oracle/tests/` |
| Config file | none (cargo default); vendor gate via env `CINTX_ORACLE_BUILD_VENDOR=1` + `--features cpu` |
| Quick run command | `cargo test -p cintx-oracle --features cpu giao -- --nocapture` (after build) |
| Full suite command | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --locked` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| FND-03 | manifest `complex_output` drives `complex_interleaved` + `2×` staging; contract fail-closed | unit | `cargo test -p cintx-runtime build_output_layout` | ❌ Wave 0 |
| FND-03 | `assert_flat_buffer_contract` fails a complex family staged real-only | unit | `cargo test -p cintx-oracle --lib assert_flat_buffer_contract` | ❌ Wave 0 |
| FND-03 | `int1e_igovlp` safe-API round-trip: imag non-zero, real exactly zero (D-07) | integration | `cargo test -p cintx-oracle --features cpu giao_complex_roundtrip` | ❌ Wave 0 (`tests/giao_complex_roundtrip.rs`) |
| GIAO-01 | 11 spin-free 1e families byte-identical cart+sph, non-square + non-zero-gauge | integration (vendor-gated) | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu giao_1e_parity` | ❌ Wave 0 (`tests/giao_1e_parity.rs`) |
| GIAO-02 | 3 spin-free 2e families byte-identical cart+sph | integration (vendor-gated) | `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu giao_2e_parity` | ❌ Wave 0 (`tests/giao_2e_parity.rs`) |
| (recipe) | `manifest-audit` green after registration | xtask/unit | per D-10 (auto-syncs from lock) | ✓ existing |

### Sampling Rate
- **Per task commit:** `cargo test -p <crate-touched> --features cpu` (quick, no vendor build).
- **Per wave merge:** full vendor-gated suite `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu`.
- **Phase gate:** full suite green before `/gsd-verify-work`; `manifest-audit` green; all 28 symbols `oracle_covered=true`.

### Wave 0 Gaps
- [ ] `crates/cintx-oracle/tests/giao_1e_parity.rs` — covers GIAO-01 (11 families × cart/sph)
- [ ] `crates/cintx-oracle/tests/giao_2e_parity.rs` — covers GIAO-02 (3 families × cart/sph)
- [ ] `crates/cintx-oracle/tests/giao_complex_roundtrip.rs` — covers FND-03 safe-API D-07 assertion
- [ ] `crates/cintx-runtime` unit test for manifest-driven `complex_output` → `complex_interleaved`/staging (FND-03)
- [ ] `crates/cintx-oracle` lib unit test for generalized fail-closed `assert_flat_buffer_contract` (D-04)
- Framework: present, no install needed.

### Folded Todo (vendor-gate hygiene cross-link)
`.planning/todos/pending/oracle-cart-offset-vendor-zero.md`: **CONFIRMED pre-existing** — the todo's own metadata states `repro_commit: 8997703`, `repro_result: reproduced (pre-phase-20)`, `classification: standalone oracle-harness bug (pre-existing)`, `blocks_phase_24_gate: false`. `[VERIFIED: todo file frontmatter]`. 3 `compare::tests` lib unit tests fail under `--lib` vendor gate at `CINTshells_cart_offset[4] cintx=8 vendor=0`. **Action: do NOT let this block the Phase 26 family gate** — the family parity is `--test` integration (passes); the `--lib` unit failures are the tracked harness bug. No re-repro needed (already triaged Phase 24/25).

## Security Domain

Not applicable in the conventional sense — this is a numerical library with no auth/session/network surface. Per CLAUDE.md the relevant integrity controls are: typed errors (`thiserror` v2) with no silent partial writes (FND-06 fail-closed staging), `cargo --locked` reproducibility, and byte-identity oracle gates. No ASVS categories apply (no input from untrusted network, no crypto, no auth). Input validation (V5-analog) is the shell-tuple validator + `assert_flat_buffer_contract` fail-closed gate.

## Recommended Plan Decomposition (D-09)

**Plan 1 — FND-03 foundation (MERGES FIRST, blocks all families):**
1. Add `complex_output: bool` to lock.json schema + `ManifestEntry` (`resolver.rs:99`) + CSV/`api_manifest.rs`; backfill `true` on all `*_spinor` rows; `cargo build -p cintx-ops`.
2. `planner.rs:319-323` — replace `rep==Spinor` with `descriptor.entry.complex_output`.
3. `compare.rs:270-285` — generalize `assert_flat_buffer_contract` to gate on `complex_interleaved` (any rep), always-on fail-closed (D-04).
4. D-02 comptime `complex_output` hint into `#[cube]` kernel signature.
5. FND-03 unit tests + `giao_complex_roundtrip.rs` (D-07 on `int1e_igovlp`).
**Verify merge before any family plan starts (`merge-base --is-ancestor`).**

**Cluster A — 1e GIAO/CG (GIAO-01), worktree-parallel after Plan 1:**
- Register 11 families (D-10 recipe) cloning `int1e_r` (rank 3) / `int1e_rr` (rank 9).
- 1e `#[cube]` kernels mirroring `one_electron.rs:7287` `common_orig` consumer; transcribe gout per cited source.
- `giao_1e_parity.rs` vendor-gated, non-square + non-zero-gauge.

**Cluster B — 2e GIAO (GIAO-02), worktree-parallel after Plan 1:**
- Register 3 families cloning `int2e_ip1` (rank 3) / `int2e_ipip1` (rank 9).
- 2e kernels extending `two_electron.rs`.
- `giao_2e_parity.rs` vendor-gated.

**Worktree caution (project memory):** background worktree auto-merge is inconsistent — after each wave, verify with `git merge-base --is-ancestor <wave-branch> main`; merge manually if not an ancestor.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Complex routing via `rep==Spinor` (`planner.rs:323`) | Manifest `complex_output` flag (D-01) | This phase | Decouples complex sizing from representation; enables cart/sph complex families |
| `assert_flat_buffer_contract` spinor-only | Generalized fail-closed on `complex_interleaved` (D-04) | This phase | Any complex family staged real-only fails |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | libcint 6.1.3 cart/sph GIAO symbols are real `double *out` (1×), NOT `double complex` (2×); the `2×`/`*mut f64` len-2N D-05 binding applies only to `_spinor` (out of scope). Vendor comparison is real-vs-real; the `Complex<f64>` view is cintx-side i-materialization. | Summary, Vendor FFI, D-05/D-07 | **HIGH** — changes vendor-FFI binding work and D-07 assertion target. This is `[VERIFIED]` against source (`cint_funcs.h:14`, `cart2sph.c:5820`), so risk is in the *plan reconciliation* not the fact: discuss-phase must confirm the team accepts treating the cintx complex view as a presentation layer over real vendor output. |
| A2 | `int2e_g1g2` is OUT of GIAO-02 scope (CONTEXT lists only `g1`/`gg1`/`ig1`/`giao_*`; `g1g2` is 2nd-gauge-both-electrons, no spin block but not enumerated). | GIAO-02 roster | MEDIUM — if intended in scope, one extra family (rank TBD from `intor2.c` `ng[]`) is missed. Confirm with planner. |
| A3 | `int1e_govlp`/`int1e_gnuc` (vs the `ig`-prefixed `igovlp`/`ignuc`) are distinct in-scope families (both exist in source). | GIAO-01 roster | LOW — both `[VERIFIED]` present; `g*` and `ig*` differ by a factor/derivative. Both registered. |

## Open Questions

1. **D-05 reconciliation (A1):** the planner must decide whether the cintx safe-API `2×` complex view is materialized in the kernel (D-02 comptime emits `[0, value]` pairs) or post-kernel on host. Recommendation: emit real components on device (matches Phase-24 moments), materialize re=0/im=value in the host marshaling layer before the `complex_values()` view. Vendor comparison takes cintx's imag half vs vendor real.
2. **`int2e_g1g2` scope (A2):** confirm in/out for GIAO-02.
3. **Kernel structure (discretion):** one parameterized `#[cube]` with comptime op-kind vs per-family launchers — recommend per-family launchers for the rank-9 families (a01gp/gg1/a11part have distinct 27→9 contractions) and a shared parameterized entry for the rank-3 cross-product families (igovlp/ignuc/ig1/g1 share curl structure).

## Sources

### Primary (HIGH confidence)
- Vendored libcint 6.1.3 source — `./libcint-master/`:
  - `src/autocode/intor1.c` — `igovlp`, `ignuc`, `igkin`, `a01gp`, `ia01p`, `cg_irxp`, `giao_irjxp`, `cg_a11part`, `giao_a11part` defs + gout + `ng[]`.
  - `src/autocode/intor2.c` — `int2e_ig1`, `int2e_gg1` defs + gout.
  - `src/autocode/intor3.c` — `govlp`, `gnuc`, `sa10*` (deferred) defs.
  - `src/autocode/intor4.c` — `int2e_g1` def + gout (`:1255`).
  - `include/cint_funcs.h:14` — `CINTIntegralFunction` typedef (`double *out` for ALL forms).
  - `src/cart2sph.c:5820` (`c2s_cart_1e` real), `:4833` (`c2s_sph_1e` real), `:4869` (`c2s_sf_1e` complex — spinor only).
  - `src/cint_config.h.in:22,25` — `POS_E1=5`, `TENSOR=7`. `src/g1e.c:49-50` — `ncomp_e1=ng[POS_E1]`, `ncomp_tensor=ng[TENSOR]`.
  - `CMakeLists.txt` — version 6.1.3.
- cintx codebase (`[VERIFIED]` by Read/grep):
  - `crates/cintx-runtime/src/planner.rs:64,319-323`; `crates/cintx-oracle/src/compare.rs:270-285`; `crates/cintx-rs/src/api.rs:577,604-612`; `crates/cintx-ops/src/resolver.rs:99-114`; `crates/cintx-ops/generated/compiled_manifest.lock.json`; `crates/cintx-oracle/build.rs:62,229,374`; `crates/cintx-oracle/src/vendor_ffi.rs`; `crates/cintx-oracle/tests/common_orig_roundtrip.rs`, `moment_*_parity.rs`.
  - `.planning/todos/pending/oracle-cart-offset-vendor-zero.md` (frontmatter triage).
  - `docs/manual/Cubecl/` mirror; CLAUDE.md stack.

### Secondary (MEDIUM confidence)
- PySCF convention that GIAO cart/sph real output is multiplied by `1j` by callers (training knowledge; consistent with the real `double *out` source finding — used only to explain the "purely imaginary" framing).

## Metadata

**Confidence breakdown:**
- Roster (28 symbols): HIGH — direct grep of source function definitions.
- component_rank (3 vs 9 per family): HIGH — read from `ng[TENSOR]` tuples with cited macro index.
- gout component order: HIGH — transcribed from cited gout functions.
- Vendor signature (real vs complex): HIGH — `CINTIntegralFunction` typedef + c2s path verified.
- FND-03 edit sites + blast radius: HIGH — grep + Read of all referenced lines.
- A1 plan reconciliation: HIGH on the fact, needs discuss-phase confirmation on accepting the presentation-layer framing.

**Research date:** 2026-05-31
**Valid until:** stable (vendored source is pinned 6.1.3; cintx code anchors verified this session) — re-verify if libcint vendor bumps or planner.rs refactors.
