# Phase 26: Group 5 (spin-free) — GIAO / NMR Integrals (complex) - Context

**Gathered:** 2026-05-31
**Reviewed:** 2026-05-31 — re-discussed all four decision clusters (FND-03 routing,
safe-API/vendor binding, proof/fixture, sequencing/kernel); all 14 decisions
(D-01–D-14) reaffirmed unchanged. Manifest complex-flag shape stays implementer's call (D-01).
**Plan-time reconciliation:** 2026-05-31 — RESEARCH findings A1/A2 surfaced to the user at
plan-phase; resolved as **D-15** (A1: vendor symbols are real `double *out`, not complex) and
**D-16** (A2: `int2e_g1g2` IN scope). These SUPERSEDE the mechanical D-05 binding shape and
extend the GIAO-02 roster — see the "Plan-time reconciliations" subsection below.
**Status:** Ready for planning

<domain>
## Phase Boundary

Bring the **spin-free 1e + 2e GIAO / CG (gauge-including / magnetic-property) family
set** to byte-identity (cart + sph) at **atol=1e-12** vs vendored libcint 6.1.3 — and,
as the load-bearing foundation, **introduce a per-family complex-interleaved output
capability (FND-03)**. These families are **purely imaginary even in cart/sph**, so the
phase's defining work is making the complex path real (sized, routed, contract-enforced,
safe-API-exposed) and proving the imaginary content actually lands against a **non-zero
gauge-origin fixture** (a zero-origin GIAO test is doubly trivial).

**Foundation (lands first, must merge before any family):**
- **FND-03 — complex/imaginary output capability:** `complex_interleaved` is set
  **per-family from driver routing (a manifest flag), NOT from the representation
  string**; `assert_flat_buffer_contract` fires on the flag (a complex cart/sph family
  staged as real-only FAILS the contract); staging is sized `2×ncomp×…`; a purely-imaginary
  family (e.g. `int1e_igovlp`) round-trips through the safe API as a typed `Complex<f64>`
  view without silent zeroing.

**Families (all NET-NEW to the manifest — zero GIAO/CG entries exist today):**
- **GIAO-01 (spin-free 1e):** `int1e_giao_*`, `int1e_cg_*`, `int1e_govlp/gnuc/gkin`,
  `int1e_ig*`, `int1e_a01gp`, `int1e_ia01p` — match at atol=1e-12 (cart + sph) via the
  complex path, the vendor wrapper passing the same `2×`-interleaved buffer to the
  `double complex *out` libcint symbol.
- **GIAO-02 (2e):** `int2e_g1`, `int2e_gg1`, `int2e_ig1`, `int2e_giao_*` — match at
  atol=1e-12. `autocode/intor4.c` is **already** in the oracle `cc::Build` source list.

**Acceptance:** per-family byte-identity at **atol=1e-12** vs vendored libcint 6.1.3 for
cart + sph, every component, under the vendor gate (`--features cpu` +
`CINTX_ORACLE_BUILD_VENDOR=1`). Each family registered with its true `component_rank` and
the new `complex_output` manifest flag, dispatched through `eval_raw`, with a dedicated
`vendor_*` parity test gated on the **non-zero gauge-origin fixture**; `oracle_covered=true`;
`manifest-audit` green.

**Out of scope:**
- `capi` enum variants and legacy `cint*` wrappers — explicitly NOT added (carry-forward
  surface-scope lock D-11 from Phases 23–25).
- Spinor GIAO representations — registered → `UnsupportedApi`. The **GIAO×σ spinor slice**
  (`int1e_spg*`, `*_sa10*`, `int2e_cg_sa10*`/`giao_sa10*`, GIAO-03) is **Phase 30**, not here.
- The σ-operator relativistic families (Group 4) — Phase 29.

</domain>

<decisions>
## Implementation Decisions

### FND-03 — complex-output capability
- **D-01 (manifest per-family flag drives complex routing):** Add a **per-family field in
  `compiled_manifest.lock.json`** (e.g. `complex_output: true` / an `output_complex_multiplier`)
  that the planner's `build_output_layout` reads to set `complex_interleaved` and size staging
  `2×ncomp`. This **replaces** the current representation-driven coupling
  (`planner.rs:323` `complex_multiplier = if rep==Spinor {2} else {1}`): the driver routes off
  **manifest data**, not the `Representation` enum — satisfying the Success-Criterion mandate
  "set per-family from driver routing, not the representation string." Mirrors how
  `component_rank` already drives sizing (data-driven, auto-syncs `manifest-audit`). NOT a
  code-side operator-name allowlist (would drift from the manifest by hand).
- **D-02 (manifest flag also flows comptime into the kernel):** The same `complex_output`
  manifest field flows as a **comptime hint into the `#[cube]` kernel** so the device path
  knows to emit interleaved real/imag pairs. One manifest field drives BOTH the host
  contract/staging AND the device output layout (hybrid host+device, not host-sizing-only).
- **D-03 (safe API returns a typed `Complex<f64>` view):** A purely-imaginary cart/sph GIAO
  family is returned to safe-API callers as a **`num_complex::Complex<f64>` view** via the
  existing `complex_values()` gate (already returns `Some` only when `complex_interleaved`).
  Matches CLAUDE.md's num-complex choice and the established spinor surface. The FND-03
  round-trip test asserts non-zero imaginary parts (and — see D-07 — zero real parts) through
  this typed view. NOT a raw interleaved-f64-buffer-plus-flag surface.
- **D-04 (`assert_flat_buffer_contract` generalized + fail-closed):** Broaden the contract
  (`compare.rs:270`) so `complex_interleaved=true` is honored for **any representation**, not
  just Spinor; a complex cart/sph family staged as **real-only FAILS the contract** (it is an
  **always-on fail-closed contract**, per the Success Criterion, not a debug-only assert).

### Vendor parity for purely-imaginary output
- **D-05 (reinterpret `double complex *` as 2×-interleaved f64):** Bind each libcint GIAO
  symbol's `out` as `*mut f64` of length `2N` (re/im interleaved — identical memory layout to
  C `double complex`), pass cintx's `2×` staging buffer to it directly, compare elementwise at
  atol=1e-12. Simpler than a `repr(C)` `Complex64` bindgen binding and matches the existing
  flat-buffer vendor-FFI convention (all current vendor symbols are flat `double *`).
- **D-06 (non-zero gauge-origin fixture gates EVERY family):** Every `vendor_*` test runs on a
  **non-zero gauge-origin fixture** (reuse / extend `build_h2o_sto3g_common_orig` from Phase 22)
  so a zero-origin doubly-trivial case can't pass. The fixture must also satisfy the
  carry-forward **non-square bra×ket** requirement (D-12) so a transposed layout cannot pass.
- **D-07 (prove imaginary lands AND real half is zero):** For purely-imaginary families the
  parity test asserts the **imaginary half is non-zero** (content actually landed, not silently
  zeroed — FND-03) **and** the **real half is exactly zero** (catches a kernel that accidentally
  writes real content). Both halves still compared byte-identical to libcint's `double complex`
  output.

### CubeCL kernel authoring (USER-MANDATED)
- **D-08 (generic-over-`F`, manual-first):** All GIAO `#[cube]` kernels are written
  **generic over `F` (the CubeCL `Float` trait bound)** and **must follow the CubeCL manual**.
  The executor **MUST read the relevant manual pages before writing any kernel code** (user
  directive, 2026-05-31). Load-bearing pages for Phase 26: **Generics** (`Float`/`Numeric`
  bounds), **Algebra** (`F::exp`/`F::sqrt`/`F::sin` — the GIAO `exp(i·k·r)` phase factor and
  gauge-dependent terms), **Basic Operations**, **Conditionals** (boundary checks). Reinforces
  the existing project-wide convention (every prior family ported generic-over-`F`) and the
  standing `#[cube]` pitfall rules (no plain-fn calls inside `#[cube]`, no `if`-expressions,
  `u32`/`i32` only for integers, no `continue`/`break`). Canonical manual: see `<canonical_refs>`.

### Sequencing & plan clustering
- **D-09 (FND-03 foundation first, then 1e then 2e clusters):**
  - **Plan 1 = FND-03** (manifest `complex_output` flag + planner routing/staging + comptime
    kernel hint + `assert_flat_buffer_contract` generalization + safe-API `Complex<f64>` view).
    **Must merge before any family plan starts.**
  - **Cluster A = 1e GIAO/CG families (GIAO-01).**
  - **Cluster B = 2e GIAO families (GIAO-02).**
  - Family clusters parallelize via **worktrees** once FND-03 lands (worktree parallelization
    is on in config). Confirm post-wave integration with `merge-base --is-ancestor` (worktree
    auto-merge is inconsistent — see project memory). Mirrors Phase-25 D-06 (foundation-first).

### Carry-forward locks (from Phases 23–25 — do NOT re-litigate)
- **D-10 (registration recipe):** 5 steps land a new family — (1) manifest lock entry cloning
  the closest family with `component_rank` = true output multiplier **plus the new
  `complex_output` flag**, then `cargo build -p cintx-ops` regenerates `api_manifest.{rs,csv}`;
  (2) `RawApiId` consts in `cintx-compat/src/raw.rs`; (3) launcher dispatch on
  `descriptor.operator_name()`; (4) vendor FFI — add the GIAO symbols to the bindgen
  `allowlist_function` regex in `cintx-oracle/build.rs` (confirm `intor4.c` is in the build
  source list — it is) + safe wrappers in `vendor_ffi.rs`; (5) `vendor_*` parity test.
  **Lock edits auto-sync `manifest-audit` — there is NO separate fixtures family list to edit.**
  ⚠ **OperatorId shift caution (project memory):** positional manifest ordering means adding
  GIAO rows re-points any hardcoded `OperatorId::new(<int>)` / `_OPERATOR_ID: u32 = N` test
  consts — resolve by symbol name, and re-grep these consts after registering.
- **D-11 (surface scope):** manifest + RawApiId + kernel + vendor-FFI + oracle only. No `capi`
  enum variants, no legacy `cint*` wrappers. Spinor reps registered → `UnsupportedApi`.
- **D-12 (transpose discipline):** copy each family's component order **verbatim from the
  libcint gout index map**; raise angular-momentum headroom on the **ket** (`ng[]`), not the
  bra; **gate every family with a NON-SQUARE bra×ket block** (e.g. p×d) so a transposed layout
  cannot pass.
- **D-13 (component-rank-truncation hard rule):** a `component_rank` set too LOW silently
  TRUNCATES trailing output components. Each family's `component_rank` MUST equal its true
  output multiplier — **derive exact values from libcint source**, do not guess (the
  `r_gauge × ∇` tensor multiplicity makes several GIAO ranks non-obvious).
- **D-14 (verification gate):** per-family byte-identity at **atol=1e-12** vs vendored libcint
  6.1.3, cart + sph, every component, in `vendor_*` parity tests double-gated on `--features cpu`
  + `CINTX_ORACLE_BUILD_VENDOR=1` (without both, parity silently skips).

### Plan-time reconciliations (2026-05-31 — RESEARCH A1/A2, user-confirmed at plan-phase)
- **D-15 (A1 — vendor symbols are real `double *out`, SUPERSEDES the mechanical D-05 binding
  shape):** libcint 6.1.3's in-scope cart/sph GIAO symbols are real `double *out` (1×), NOT
  `double complex` (2×) — `[VERIFIED: include/cint_funcs.h:14 CINTIntegralFunction typedef;
  src/cart2sph.c:5820 c2s_cart_1e real-double copy]`. The `*mut f64` len-2N reinterpretation in
  D-05 applies only to `_spinor` symbols (out of scope). **Resolution:** the vendor FFI wrappers
  bind a plain real `double *out` of length `nao_i × nao_j × component_rank` (identical to every
  existing moment/derivative wrapper); the parity comparison is **real-vs-real** at atol=1e-12.
  The `Complex<f64>` safe-API view (D-03) is materialized **cintx-side** (re=0, im=value) AFTER
  the device kernel writes real components; D-07 (imag non-zero, real exactly zero) is asserted on
  that cintx safe-API view, NOT on a 2×-interleaved vendor buffer. **All other FND-03 intent
  (D-01/D-02/D-03/D-04 manifest-driven complex routing + comptime kernel hint + fail-closed
  contract, D-06 non-zero-gauge gate, D-07 proof) is PRESERVED unchanged.** Only D-05's vendor-
  binding shape changes. Do NOT bind cart/sph GIAO as len-2N or pass a 2× buffer to libcint.
- **D-16 (A2 — `int2e_g1g2` IN scope):** GIAO-02's spin-free 2e roster is exactly
  **`{int2e_g1, int2e_ig1, int2e_gg1, int2e_g1g2}`** (4 families). `int2e_g1g2`
  (2nd-gauge-on-both-electrons, no spin block) is registered + kernel-ported + vendor-parity-
  tested in Cluster B alongside the other three. Its `component_rank` and gout component order are
  **derived from `src/autocode/intor2.c` `ng[]` / gout** (do not guess, per D-13). Spin-carrying
  `int2e_giao_sa10*` / `int2e_g1spsp2` remain DEFERRED to Phase 30.

### Claude's Discretion
- The **exact enumerated roster** of `int1e_giao_*` / `int1e_cg_*` / `int2e_giao_*` wildcard
  families — derive the complete set libcint 6.1.3 actually exports from the vendored sources;
  do not guess.
- Exact `component_rank` value and libcint gout component-index order per family (derived from
  libcint source by researcher/planner; gate with the non-square block per D-12).
- The precise manifest field name/shape for the complex flag (`complex_output: bool` vs an
  `output_complex_multiplier: u32`) — implementer's call, as long as D-01/D-02 hold.
- The precise corpus shell-tuple selection for each `vendor_*` test (subject to the
  non-square + non-zero-gauge requirements of D-06/D-12).
- Whether the GIAO kernels are one parameterized `#[cube]` entry with comptime op-kind or
  per-family launchers — implementer's call, as long as D-02/D-08/D-12 hold.

### Folded Todos
- **`oracle-cart-offset-vendor-zero`**
  (`.planning/todos/pending/oracle-cart-offset-vendor-zero.md`, medium): 4 `compare::tests`
  **lib** unit tests fail under the vendor gate at `CINTshells_cart_offset[4] cintx=8 vendor=0`;
  integration parity passes; hypothesis is a harness/env bug. **Folded as a vendor-gate-hygiene
  cross-link** (same handling as Phases 24/25): Phase 26 runs the vendor gate, so this WILL
  re-surface. In-scope action: confirm it is **pre-existing** (reproduce against a pre-phase-20
  commit); if reproduced, keep it as a tracked standalone oracle-harness bug — **do NOT let it
  block the Phase 26 family gate.**

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### CubeCL authoring (USER-MANDATED — read before writing any kernel code, D-08)
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` — the authoritative
  external CubeCL manual the user requires for all Phase-26 calculations. Load-bearing pages:
  - `…/Cubecl_generics.md` — `Float`/`Numeric` trait bounds (generic-over-`F` kernels).
  - `…/Cubecl_algebra.md` — `F::exp`/`F::sqrt`/`F::sin` and math functions (GIAO phase factor).
  - `…/Cubecl_basic_operations.md` — supported arithmetic in generic kernels.
  - `…/Cubecl_conditionals.md` — `if`/`else` for boundary checks.
  - `…/cubecl_error_solution_guide/` — type-mismatch + "no plain Rust fn inside `#[cube]`" pitfalls.
- `docs/manual/Cubecl/*.md` (in-repo mirror) — secondary copy of the same `#[cube]` rules.

### Requirements & roadmap
- `.planning/REQUIREMENTS.md` §FND-03 (L83) and §GIAO-01, §GIAO-02 (Group 5) — the
  requirements this phase satisfies. (§GIAO-03 is Phase 30 — out of scope.)
- `.planning/ROADMAP.md` — Phase 26 entry: Goal, Success Criteria 1–4, Depends-on Phase 22
  (gauge origin), Phases 23 + 24 (the nabla step + position-operator tensor the `r_gauge × ∇`
  factor combines).

### Carry-forward context (the proven precedents this phase reuses)
- `.planning/phases/25-group-2-hessian-higher-order-derivatives/25-CONTEXT.md` — foundation-first
  sequencing (D-06), registration recipe (D-08), transpose discipline (D-09), component-rank-
  truncation rule (D-10), surface scope (D-11), verification gate (D-12).
- `.planning/phases/24-group-3-position-multipole-moment-integrals/24-CONTEXT.md` — the
  `common_orig` consumer precedent and family-clustering pattern.
- `.planning/phases/22-gauge-origin-env-slot-gap-a-ptr-common-orig/22-CONTEXT.md` — the
  `PTR_COMMON_ORIG` gauge-origin env-slot plumbing (FND-01) this phase's GIAO kernels consume.

### Code anchors (from scout)
- `crates/cintx-runtime/src/planner.rs:307-323` — `build_output_layout`: the representation-
  driven `complex_multiplier`/`complex_interleaved` coupling to **replace** with the manifest
  `complex_output` flag (D-01); staging-alloc at `try_alloc_staging` (`:348-357`).
- `crates/cintx-runtime/src/planner.rs:64` — `OutputLayoutMetadata.complex_interleaved` field.
- `crates/cintx-oracle/src/compare.rs:270-285` — `assert_flat_buffer_contract`: Spinor-only
  complex assertion to generalize + make fail-closed for any representation (D-04).
- `crates/cintx-rs/src/api.rs:577,604-607` — `OutputDescription.complex_interleaved` +
  `complex_values()` gate (returns `Some` only when the flag is set) — the D-03 safe surface.
- `crates/cintx-compat/src/raw.rs:50,861-871,1466-1481` — `PTR_COMMON_ORIG` const, env→
  `operator_env_params.common_orig` extraction, `representation_from_descriptor()`; the
  registration-recipe sites (RawApiId consts, launcher dispatch) for D-10.
- `crates/cintx-oracle/build.rs:51-262` — oracle `cc::Build` source list; **`intor4.c` is
  already present** (`:62,:229`); bindgen `allowlist_function` regex to extend (D-10 step 4).
- `crates/cintx-oracle/src/vendor_ffi.rs` — flat `double *out` vendor wrappers; add the GIAO
  symbols as `*mut f64` len-`2N` interleaved bindings (D-05).
- `crates/cintx-ops/generated/compiled_manifest.lock.json` — 302 entries, **zero GIAO/CG**;
  all Phase-26 families are net-new (D-01 adds the `complex_output` field to the schema).
- `crates/cintx-cubecl/src/kernels/one_electron.rs:7287-7289` — existing `common_orig`
  consumer (Phase-24 moments) the 1e GIAO kernels mirror; `…/kernels/two_electron.rs` — the
  2e ERI engine the GIAO-02 set extends.
- `crates/cintx-oracle/tests/common_orig_roundtrip.rs`, `…/tests/moment_*.rs` +
  `build_h2o_sto3g_common_orig()` — the non-zero gauge-origin fixture to reuse/extend (D-06).

### libcint upstream (researcher must derive exact roster, gout orders, and ranks from these)
- libcint 6.1.3 GIAO/CG sources + `autocode/intor4.c` gout emitters and `ng[]` headroom tuples
  for the `g*`/`ig*`/`cg*`/`giao_*` families — source of truth for the exact roster, component
  order (D-12), and `component_rank` (D-13). The `double complex *out` symbol signatures are the
  vendor-FFI binding target (D-05). Vendored under the oracle build.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`complex_interleaved` flag (end-to-end, Spinor today):** defined `planner.rs:64`,
  `tensor.rs:12`, `layout.rs:13`, `api.rs:577`; set `planner.rs:323`; read at `api.rs:604`
  (`complex_values()`), `compare.rs:279`. FND-03 re-keys the SET site from `Representation`
  to the manifest flag and generalizes the contract — the plumbing already exists.
- **`PTR_COMMON_ORIG` gauge origin (Phase 22, fully wired + tested):** GIAO kernels consume
  `plan.operator_env_params.common_orig` directly — no new plumbing.
- **`intor4.c` already in oracle `cc::Build`** — GIAO-02 2e kernels need no build-source change,
  only bindgen allowlist + vendor wrappers.
- **Phase-24 moment kernels (`one_electron.rs:7287`)** — the `r_gauge`-dependent 1e tensor
  pattern (reads `common_orig`) the 1e GIAO families mirror.
- **Registration recipe** proven on Phases 23/24/25 — `manifest-audit` auto-syncs from the lock.

### Established Patterns
- Staging sized `base × component × complex_multiplier`; FND-03 makes `complex_multiplier`
  derive from the manifest `complex_output` flag instead of `rep==Spinor` (D-01).
- Generic-over-`F` `#[cube]` kernels following the CubeCL manual (D-08); `vendor_*` parity
  double-gated on `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`.
- Component order copied verbatim from libcint gout, gated with a NON-SQUARE bra×ket block.

### Integration Points
- FND-03: `planner.rs build_output_layout` ← reads the new manifest `complex_output` field;
  `compare.rs assert_flat_buffer_contract` ← fires fail-closed on the flag; `api.rs
  complex_values()` ← already gated → exposes the `Complex<f64>` view.
- Vendor FFI: `build.rs` allowlist + `vendor_ffi.rs` `*mut f64` len-`2N` bindings →
  libcint `double complex *out` GIAO symbols (`intor4.c` already built).
- Kernels: `one_electron.rs` / `two_electron.rs` ← `common_orig` (Phase 22) + comptime
  `complex_output` hint (D-02) → emit interleaved re/im pairs.

</code_context>

<specifics>
## Specific Ideas

- **FND-03 is the gating risk and the phase's defining capability** — the GIAO families are
  purely imaginary, so until the complex path is sized/routed/contract-enforced/safe-exposed,
  no family can pass. Treat Plan 1 (FND-03) as the long-pole and merge it before any family.
- **The complex flag is manifest data, not code** — one `complex_output` field drives planner
  staging, the always-on fail-closed contract, AND the comptime kernel layout (D-01/D-02/D-04).
- **Non-zero gauge origin is mandatory on every gate** — a zero-origin GIAO test is doubly
  trivial; reuse `build_h2o_sto3g_common_orig`, keep the block non-square, and assert the
  imaginary half is non-zero while the real half is exactly zero (D-06/D-07).
- **CubeCL manual is mandatory pre-reading for the kernel work** (user directive, D-08) — the
  executor reads Generics/Algebra/Basic-Operations/Conditionals before any `#[cube]` code.

## Open Items for Research/Planning (not user decisions)
- Enumerate the complete `int1e_giao_*` / `int1e_cg_*` / `int2e_giao_*` roster from libcint
  6.1.3 — do not guess the wildcard families.
- Derive exact `component_rank` (the `r_gauge × ∇` tensor multiplicity makes several non-obvious)
  and gout component order per family from libcint source; confirm `ng[]` headroom tuples.
- Confirm the libcint `double complex *out` symbol signatures for the bindgen allowlist + the
  `*mut f64` len-`2N` reinterpretation binding (D-05).
- Decide the precise manifest field name/shape for the complex flag (D-01 leaves this open).

</specifics>

<deferred>
## Deferred Ideas

- **GIAO×σ spinor slice (GIAO-03)** — `int1e_spg*`, `int1e_spgnucsp`, `*_sa10*`,
  `int2e_cg_sa10*`/`giao_sa10*` — is **Phase 30** (needs the Gap B2 `c2s_si` spin-included
  spinor transform from Phase 28). Not this phase.
- **Spinor GIAO representations** — registered → `UnsupportedApi` this phase (D-11); land when
  the spinor-derivative / spin-included transforms (Phases 27/28) exist.
- **`Complex64` repr-C FFI binding** — D-05 uses the simpler `*mut f64` interleaved
  reinterpretation; a typed complex FFI binding is a possible later ergonomics improvement, not
  needed for byte-identity.

### Reviewed Todos (not folded)
- **`rys-nroots-ge6-wheeler-fallback`** (`.planning/todos/pending/`) — `resolves_phase: 25`,
  resolved by Phase 25's FND-02 (Wheeler nroots≥6 port). Out of scope for Phase 26 (GIAO
  families on the H2O/STO-3G corpus do not push nroots≥6); reviewed, not folded.

</deferred>

---

*Phase: 26-group-5-spin-free-giao-nmr-integrals-complex*
*Context gathered: 2026-05-31*
