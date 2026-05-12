# Phase 19: `int1e_ecp_*` Type-1/Type-2 Evaluator - Context

**Gathered:** 2026-05-12
**Status:** Ready for planning

<domain>
## Phase Boundary

cintx implements Type-1 (local, Coulomb-like) and Type-2 (semi-local, spin-orbit-like)
Effective Core Potential projectors as new `#[cube]` kernels and exposes them through
`SessionRequest` alongside ordinary one-electron operators. Six new operator symbols
land in the manifest and dispatch through the existing arity-2 `cintx_cubecl::CubeClExecutor`
path: scalar `int1e_ecp_{cart,sph}` plus gradient variants `int1e_ecp_ipnuc_{cart,sph}`,
with each cart/sph pair sharing one underlying kernel and differing only in the
representation transform (matching the rest of the 1e family). Spinor representation
is **not** byte-identity-gated this phase (mirroring Phase 18 D-07 "compiled but
unverified" precedent).

The phase introduces two new math modules in `crates/cintx-cubecl/src/math/` —
modified spherical Bessel functions (`bessel.rs`) and Gauss-Chebyshev / Gauss-Hermite
radial quadrature (`radial_quadrature.rs`) — required by Type-2 angular projector
integration and the radial expansion that ECP-projected Gaussians demand. A new
`EcpShell` type lands in `cintx-core` and `BasisSet` gains an optional
`ecp_shells: Arc<[Arc<EcpShell>]>` field; `BasisSet::ecp_shells()` returns `&[]`
when no ECP basis is attached, preserving SemVer for every existing caller.

Vendored libcint (`libcint-master/src/`) currently contains **zero** ECP source —
the byte-identity gate against "libcint" requires importing libcint 6.1.3's upstream
`ecp.c` / `cint_ecp.h` files into the vendor tree and extending `cintx-oracle`'s
`build.rs` + FFI surface. A secondary non-blocking cross-check against
chrr/libECP (JCC 2017) closes ROADMAP SC#4's "secondary oracle" wording.

The new Cu/LANL2DZ fixture in `crates/cintx-oracle/src/fixtures.rs::build_cu_lanl2dz()`
is built fresh — no Cu basis or ECP fixture exists today despite the ROADMAP's
"already present in the oracle test corpus" claim.

</domain>

<decisions>
## Implementation Decisions

### Oracle reference source
- **D-01:** **Vendor libcint 6.1.3's upstream `ecp.c` / `cint_ecp.h` (and any required
  headers) into `libcint-master/src/` as the primary byte-identity reference at
  `atol=1e-12, rtol=0.0`.** Extend `crates/cintx-oracle/build.rs` to compile the new
  C sources alongside the existing ones; add per-symbol FFI wrappers in
  `crates/cintx-oracle/src/vendor_ffi.rs` (or sibling module) for
  `int1e_ecp_{cart,sph}` and `int1e_ecp_ipnuc_{cart,sph}`. Matches SC#4 wording
  literally and preserves the existing C-FFI oracle pattern used by every other
  family (no Python sidecar harness).
- **D-02:** **libECP (chrr, JCC 2017) added as a secondary, non-blocking oracle.**
  Linked optionally behind a `cintx-oracle` build cfg (e.g.,
  `#[cfg(has_libecp_oracle)]`), invoked from a sibling parity test file
  (`ecp_libecp_crosscheck_parity.rs`). Failures do **not** block CI; tolerance is
  documented as "informational, libECP and libcint use different recurrence
  conventions internally" and is set per-test as a loose envelope (e.g., `atol=1e-9`)
  pending empirical measurement during execution.

### Typed-API placement of `ecpbas`
- **D-03:** **Extend `cintx-core::BasisSet` with an optional `ecp_shells: Arc<[Arc<EcpShell>]>`
  field plus public accessor `BasisSet::ecp_shells() -> &[Arc<EcpShell>]` (returns
  `&[]` when no ECP attached).** Add new `EcpShell` type to `cintx-core::shell`
  (or a new `cintx-core::ecp` module) carrying: `atom_index`, `angular_momentum`,
  `radial_power`, `nprim`, `exponents`, `coefficients`, and `ecp_type` (Type-1 vs
  Type-2 marker). `BasisSet::try_new` keeps its current signature; a new
  `BasisSet::try_new_with_ecp(atoms, shells, ecp_shells)` is the typed entry point
  for ECP callers. The struct's existing field `shells` and metadata (`BasisMeta`)
  stay unchanged.
- **D-04:** **`EcpShell` is the typed analog of libcint's `ecpbas` row.** Fields
  derived from libcint's per-row layout (atom-of, ang-of, nprim-of, nctr-of,
  radial_power-of, so_type-of, ptr_exp-of, ptr_coeff-of) collapsed into a typed
  Rust struct holding owned `SmallVec` of exponents/coefficients (no `*ptr_env*`
  indirection in the typed surface; raw-compat layer converts to/from env-array
  layout). Distinct from `Shell` (no `kappa`, has `radial_power` and `ecp_type`).
- **D-05:** **`cintx-compat::raw` gains an `EcpBasArray` typed view + the canonical
  ECP slot constants** (`RADI_POWER_OF`, `SO_TYPE_OF`, `ECP_BAS_SLOTS = 8`, `PTR_ECPBAS_OFFSET`,
  `PTR_NECPBAS`, matching upstream libcint cint.h). Raw callers continue to pass
  `i32` slabs; typed callers go through `BasisSet::ecp_shells()`. `eval_raw` for
  `int1e_ecp_*` symbols accepts the extended atm/bas/ecpbas/env layout fail-closed
  on missing ecpbas when an ECP operator is dispatched.
- **D-06:** **`SessionRequest::new` signature stays SemVer-stable.** The ECP basis
  rides on the existing `basis: &'basis BasisSet` parameter (which now carries
  optional `ecp_shells`). No builder method, no extra positional arg, no
  `ExecutionOptions` field. `query_workspace` fails fast with a typed
  `FacadeError::MissingEcpBasis` when the operator is `int1e_ecp_*` and
  `basis.ecp_shells().is_empty()`.

### Kernel implementation strategy
- **D-07:** **Full `#[cube]` GPU implementation. Add two new math modules to
  `crates/cintx-cubecl/src/math/`: `bessel.rs` (modified spherical Bessel functions
  for Type-2 angular projection) and `radial_quadrature.rs` (Gauss-Chebyshev radial
  for the projector integral, Gauss-Hermite for Type-1's Coulomb-like radial
  expansion).** Both modules follow the Phase 8 pattern: `#[cube]` function +
  pure-Rust `*_host()` counterpart for unit-test invocability without GPU context.
  Pinned roots/weights tables (if any) follow the Phase 13 precedent of binary-included
  tables (`include_bytes!` + `bytemuck::AlignedBytes`).
- **D-08:** **New `crates/cintx-cubecl/src/kernels/ecp.rs` module with
  `launch_ecp(plan, key, out) -> Result<ExecutionStats>`.** `canonical_family = "ecp"`
  routed via `crates/cintx-cubecl/src/kernels/mod.rs::resolve_family_name` and
  `supports_canonical_family`. Type-1 and Type-2 share the launcher; an internal
  per-shell branch over `EcpShell::ecp_type` selects the algorithm. Cart-to-sph
  transform reuses the existing `crate::transform::c2s::cart_to_sph_1e` pipeline
  (the same shape arity-2 1e operators use today).
- **D-09:** **No new family-level resolver work needed beyond manifest expansion**
  — the existing `Resolver::descriptor` is manifest-driven, so adding the six new
  rows to `crates/cintx-ops/src/generated/api_manifest.csv` automatically lights up
  routing once the launcher exists. Manifest entries: `family_name = "1e"`,
  `canonical_family = "ecp"`, `operator_name ∈ {"ecp", "ecp_ipnuc"}`, `arity = 2`,
  `forms = "cart" | "sph"`, `helper_kind = "operator"`, `stability = "stable"`,
  `feature_flag = "none"`, `compiled_in_profiles` = all four profiles. Spinor
  rows are NOT added this phase (D-12).

### Phase scope: gradient variants
- **D-10:** **Gradient variants `int1e_ecp_ipnuc_{cart,sph}` ARE in scope.** Phase 19
  ships four base symbols (`int1e_ecp_{cart,sph}` × Type-1+Type-2 combined) plus two
  gradient symbols (`int1e_ecp_ipnuc_{cart,sph}`). Six symbols total in the
  parity sweep. Phase becomes ~5-6 plans rather than 3-4.
- **D-11:** **Type-1 and Type-2 gradient share one kernel launcher** with internal
  branching. Component rank for gradients is 3 (one derivative per Cartesian axis).
  The arity-2 1e flat-buffer F-order layout is preserved; the component axis is
  the slowest-varying, matching the existing convention used by
  `int3c2e_ip1_{cart,sph}` (component_rank=3 in the manifest).
- **D-12:** **Spinor representation (`int1e_ecp_spinor`, `int1e_ecp_ipnuc_spinor`) is
  out of the parity sweep this phase.** Type-2 ECP is spin-orbit-like, so spinor IS
  conceptually the natural representation, but ECP spinor needs its own multi-component
  transform path. Mirroring Phase 18 D-07: spinor accepted by the resolver and
  routable, but NOT byte-identity-gated. Document in module rustdoc on
  `cintx-rs::api`: "ECP spinor outputs are not oracle-gated in Phase 19". Add as a
  deferred idea.

### Claude's Discretion
- **Exact upstream libcint ECP source set to vendor (D-01).** Likely
  `libcint-master/src/ecp.c` + any headers it transitively needs. Researcher
  enumerates the full file list from upstream libcint 6.1.3 tree and confirms
  no GPL-incompatible dependencies pull in. If a transitive dep exists that we
  don't want, fall back to extracting just the symbols cintx-oracle needs and
  vendor a minimal subset with a `.planning/notes/libcint-ecp-vendor-subset.md`
  rationale.
- **`EcpShell` field exact name set** — `radial_power` vs `r_power`, `ecp_type`
  vs `projector_kind`, `coefficients` vs `coeffs`. Planner picks names consistent
  with the existing `Shell` struct convention. `Default` impl: probably not
  needed since `EcpShell` is always constructed with full data.
- **Math module function signatures** — `bessel.rs` likely exposes
  `modified_spherical_bessel_kn(n, x) -> f64` and `modified_spherical_bessel_in(n, x) -> f64`
  with `#[cube]` and `*_host()` variants. `radial_quadrature.rs` exposes
  `gauss_chebyshev_nodes_weights(n) -> (Array<f64>, Array<f64>)` and
  `gauss_hermite_nodes_weights(n) -> ...`. Exact node count and tolerance per
  ECP shell is a researcher / planner question; default = libcint's default
  (radial precision controlled by `RADI_POWER_OF` slot).
- **Fixture construction for Cu/LANL2DZ.** Build inside
  `crates/cintx-oracle/src/fixtures.rs::build_cu_lanl2dz()` analogous to
  `build_h2o_sto3g`. Source: PySCF basis exchange or basissetexchange.org.
  Defer a multi-fixture pass (lighter atom: Na/SBKJC, K/CRENBL) to verify ECP
  correctness on a simpler test case before Cu. Default this phase: Cu/LANL2DZ
  only, with a `.planning/spikes/ecp-fixture-validation.md` seed for the
  lighter-atom follow-up.
- **`FacadeError::MissingEcpBasis` variant fields.** Likely
  `{ operator: String }` so the error names which operator demanded ECP basis.
  Distinct from `UnsupportedApi` and `UnsupportedAoSymmetry` so callers can
  pattern-match. Add to `cintx-rs/src/error.rs` at end of variants.
- **Parity test fixture coverage** — Cartesian product of all Cu/LANL2DZ shells
  vs an iteration subset. Cu basis has ~8-10 shells, so ~64-100 arity-2 tuples
  per test. Within CI budget. Default: full Cartesian product.
- **Whether the libECP secondary oracle test file lives in `cintx-oracle/tests/`
  or in a separate `cintx-oracle/tests/non-blocking/` subdir.** Default: same
  directory, name conveys non-blocking nature (`ecp_libecp_crosscheck_parity.rs`)
  and the file's tests use `#[ignore]` + a `CINTX_LIBECP_ORACLE=1` env gate
  (matching the Phase 16 ROCm precedent for opt-in oracle gates).
- **`canonical_family = "ecp"` string choice** — alternatives include `"int1e_ecp"`,
  `"ecp1e"`. Default: `"ecp"` (short, parallel to `"f12"` for the F12 family).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase intent and scope
- `.planning/ROADMAP.md` § "Phase 19: `int1e_ecp_*` Type-1/Type-2 Evaluator
  (issue #11 Task 1)" — locks goal, success criteria 1-5 (ECP-01..05), named
  symbol list, and pyscf_rs `crates/pyscf-gto/src/ecp_engine_stub.rs` downstream
  marker.
- `.planning/PROJECT.md` § Constraints — CubeCL is the primary compute backend
  (D-07 honors this); safe Rust API first-priority surface (D-03 honors this);
  type-safe library errors via `thiserror` (D-06 adds `MissingEcpBasis` variant).
- `.planning/notes/pyscf-rs-as-cintx-consumer.md` — downstream consumer context
  (issue #11). pyscf_rs `crates/pyscf-gto/src/ecp_engine_stub.rs` is the
  placeholder this phase replaces. Two independent oracle gates: cintx-oracle
  (this repo) and pyscf_rs `tests/oracle/`. libECP cross-reference URL:
  https://github.com/chrr/libECP (JCC 2017) — viable secondary oracle.

### Phase 17/18 predecessors (locked decisions still apply)
- `.planning/phases/17-real-integral-evaluation-in-safe-api/17-CONTEXT.md` —
  real `cintx_cubecl::CubeClExecutor` is the shared dispatch primitive (D-01);
  no `unsafe` in safe-API path (D-02); chunk-loop duplication with compat is
  acceptable (D-03); per-symbol parity test pattern (D-07); `atol=1e-12,
  rtol=0.0` tolerance (D-09); CI runs inside existing `oracle_parity_gate`
  matrix (D-10).
- `.planning/phases/18-sessionrequest-arity-ge3-dispatch/18-CONTEXT.md` —
  spinor "compiled but unverified" precedent (D-07); typed error variants on
  `FacadeError` (D-04 `UnsupportedAoSymmetry` is the pattern reference for the
  new `MissingEcpBasis` variant); additive-field discipline on `ExecutionOptions`
  / `BasisSet`; per-symbol parity file split pattern.

### Phase 15 tolerance baseline
- `.planning/phases/15-oracle-tolerance-unification-manifest-lock-closure/15-CONTEXT.md`
  — unified oracle tolerance is `atol=1e-12, rtol=0.0` with the four-profile
  manifest lock. ECP base + gradient parity tests adopt this tolerance directly.
  libECP secondary cross-check uses a looser empirically-derived tolerance
  (informational; documented in test file).

### Phase 8 math infrastructure (the pattern D-07 follows)
- `.planning/phases/08-gaussian-primitive-infrastructure-and-boys-function/08-CONTEXT.md`
  and `08-PATTERNS.md` — `#[cube]` + `*_host()` paired-function pattern;
  cond_br MLIR limitation (Phase 8 P02 incident) avoided by host-side branching;
  binary table inclusion via `include_bytes!` + `bytemuck::AlignedBytes`
  (Phase 13 P02 precedent).

### Phase 13 F12 plumbing (closest existing analog for ECP env params)
- `.planning/phases/13-f12-stg-yp-kernels/13-CONTEXT.md` — F12 added
  `canonical_family = "f12"` with kernel dispatch via
  `kernels/mod.rs::resolve_family_name`; raw compat plumbed `f12_zeta` through
  `ExecutionPlan::operator_env_params`. ECP follows the same pattern with
  `canonical_family = "ecp"` and additional ecpbas slot plumbing.

### Existing safe-API surface (the code being changed)
- `crates/cintx-rs/src/api.rs` (28-280) — `SessionRequest`, `SessionQuery`,
  `query_workspace()`, the chunk loop. D-06's `MissingEcpBasis` preflight lands
  inside `query_workspace()` next to the existing aosym check.
- `crates/cintx-rs/src/error.rs` — `FacadeError` enum. D-06 adds
  `MissingEcpBasis { operator: String }` variant at the end of variants.
- `crates/cintx-rs/src/prelude.rs` — re-export new types (`EcpShell`,
  `MissingEcpBasis` error variant is implicit via `FacadeError` re-export).
- `crates/cintx-core/src/basis.rs` (lines 46-100) — `BasisSet` struct. D-03 adds
  the `ecp_shells` field, accessor, and the `try_new_with_ecp` constructor.
  `BasisMeta::from_shells` stays AO-only (ECP shells don't produce AOs).
- `crates/cintx-core/src/shell.rs` — sibling location for new `EcpShell` type
  (D-04), or a new sibling module `crates/cintx-core/src/ecp.rs`. Planner picks.
- `crates/cintx-core/src/lib.rs` (line 19) — re-export `EcpShell`.

### Existing manifest / resolver routing
- `crates/cintx-ops/src/generated/api_manifest.csv` — 133 lines today, zero ECP
  entries. D-09 adds 6 new rows (sph + cart × {ecp, ecp_ipnuc} × scalar /
  component_rank=3). Manifest is the routing source of truth; resolver picks up
  the new rows automatically.
- `crates/cintx-ops/src/resolver.rs` — `Resolver::descriptor` already manifest-driven;
  no resolver code changes needed (only manifest expansion).

### Existing kernel routing (the launcher contract)
- `crates/cintx-cubecl/src/kernels/mod.rs` — `resolve_family_name`,
  `supports_canonical_family`, `unresolved_families`. D-08 adds `"ecp" =>
  ecp::launch_ecp as FamilyLaunchFn` to all three locations. ECP is unconditional
  (no `#[cfg(feature = "with-ecp")]` gate per D-09 stability=stable).
- `crates/cintx-cubecl/src/kernels/one_electron.rs` (line 434) — `launch_one_electron`
  is the closest structural analog for `launch_ecp`. ECP is arity-2 with overlap-like
  Cartesian → sph transform pipeline.
- `crates/cintx-cubecl/src/transform/c2s.rs::cart_to_sph_1e` — reusable cart-to-sph
  transform for the sph output path. D-08 reuses this directly.
- `crates/cintx-cubecl/src/math/` — siblings to the new `bessel.rs` and
  `radial_quadrature.rs` modules (`boys.rs`, `obara_saika.rs`, `pdata.rs`, `rys.rs`,
  `stg.rs`). D-07 lands the two new modules here.

### Vendored libcint integration
- `libcint-master/src/` — **does NOT contain ecp.c today**; D-01 imports upstream
  libcint 6.1.3's ECP sources here. `build.rs` in `crates/cintx-oracle` already
  compiles the C sources via the `cc` crate — extend the source list to include
  ECP files.
- `libcint-master/include/cint.h.in` — currently lacks ECP slot constants. D-05
  adds the ECP slot constants to `cintx-compat::raw` (mirrored from upstream
  libcint's `RADI_POWER_OF`, `SO_TYPE_OF`, `ECP_BAS_SLOTS`, `PTR_ECPBAS_OFFSET`,
  `PTR_NECPBAS`).
- `libcint-master/include/cint_funcs.h` — extern function declarations. After
  D-01 vendor, this header should already include `int1e_ecp_*` decls; verify
  during research.

### Existing parity test patterns
- `crates/cintx-oracle/tests/safe_api_arity2_parity.rs` (Phase 17 output) — direct
  pattern source for the new `safe_api_ecp_parity.rs`. Per-symbol named tests,
  `#[cfg(has_vendor_libcint)]` guard, fixed-fixture style.
- `crates/cintx-oracle/tests/one_electron_parity.rs` — original per-symbol
  parity pattern for arity-2 1e operators. ECP is 1e-shaped; pattern fits 1:1.
- `crates/cintx-oracle/src/fixtures.rs` (lines 26-140) — `build_h2o_sto3g` and
  `build_h2o_sto3g_f12` builder pattern. Claude's Discretion item: add
  `build_cu_lanl2dz()` following this convention with PTR_ENV_START alignment.
- `crates/cintx-oracle/src/compare.rs` (lines 675-900) — vendor FFI helpers.
  Extend with ECP FFI wrappers (researcher enumerates exact upstream signatures).

### Oracle build / FFI plumbing
- `crates/cintx-oracle/build.rs` — `cc` crate invocation that compiles
  `libcint-master/src/*.c`. D-01 extends the source list to include the newly-vendored
  ECP files. D-02's libECP integration uses a separate optional cfg branch
  (likely `#[cfg(has_libecp_oracle)]`).
- `crates/cintx-oracle/src/vendor_ffi.rs` (if it exists; otherwise the parity
  test files inline `extern "C"` blocks) — researcher confirms canonical
  location for new ECP `extern "C"` declarations.

### Manifest profile / feature gating
- `crates/cintx-ops/src/generated/compiled_manifest.lock.json` — four-profile
  lock. D-09 entries are profile-unconditional (no feature flag); after manifest
  CSV update, regenerate the lock via `cargo run -p xtask --
  manifest-audit --update` (researcher confirms exact xtask command). All
  four profiles will gain the 6 new entries.

### CI
- `.github/workflows/` `oracle_parity_gate` — existing CI matrix (cpu/wgpu ×
  four profiles). New ECP parity tests run inside it without a new job
  (Phase 17 D-10 / Phase 18 D-15 precedent). libECP secondary cross-check is
  opt-in (`CINTX_LIBECP_ORACLE=1` env gate, matching Phase 16 ROCm precedent).

### Downstream consumer (verification context, no read needed)
- pyscf_rs `crates/pyscf-gto/src/ecp_engine_stub.rs` (sibling path-dep, private
  repo) — the placeholder this phase replaces. pyscf_rs's
  `tests/oracle/` will exercise the live `int1e_ecp_*` dispatch through
  `SessionRequest` post-land as the secondary integration gate. No coordination
  required during planning; pyscf_rs's gate runs on its own clock.

### Reference materials (read during research only)
- Upstream libcint 6.1.3 ECP source: https://github.com/sunqm/libcint/blob/master/src/ecp.c
  (or the 6.1.3 release tag) — D-01's vendor source. Confirm license compatibility
  (Apache 2.0 vs cintx workspace license) during research.
- chrr/libECP: https://github.com/chrr/libECP — D-02's secondary cross-check
  reference. JCC 2017 paper documents the algorithm.
- PySCF `pyscf/gto/ecp.py` — cross-reference for the Type-1/Type-2 algorithm;
  not the primary oracle but useful for validating the implementation against
  a Python reference if libcint's C is unclear.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`cintx_cubecl::CubeClExecutor`** (re-exported at
  `crates/cintx-cubecl/src/lib.rs:26`) — same shared dispatch primitive
  Phase 17/18 use. Arity-2 with overlap-shaped output layout; no change.
- **`ShellTuple` with `SHELL_TUPLE_CAPACITY = 4`** — ECP is arity-2 (i, j); fits.
- **`crate::transform::c2s::cart_to_sph_1e`** (in `cintx-cubecl/src/transform/c2s.rs`)
  — the cart-to-sph transform Phase 9-10 1e kernels use. ECP `int1e_ecp_sph`
  reuses this directly with the cart kernel's output buffer.
- **`Resolver::descriptor` + `Resolver::resolve`** — manifest-driven, picks up the
  6 new manifest rows automatically once `launch_ecp` exists in
  `kernels::resolve_family_name`.
- **`HostWorkspaceAllocator`, `schedule_chunks`, `ExecutionPlan`, `ExecutionIo`**
  — all arity-generic; no change.
- **`enforce_safe_facade_policy_gate`** (`crates/cintx-compat/src/raw.rs:816`) —
  source/profile/F12/4c1e envelope checks. D-06 adds a sibling preflight
  (`MissingEcpBasis` check) in `query_workspace`; this helper itself does NOT need
  to change.
- **`build_h2o_sto3g` fixture pattern** in `crates/cintx-oracle/src/fixtures.rs:26-140`
  — PTR_ENV_START-aligned env layout. `build_cu_lanl2dz()` mirrors this; ECP slots
  (ecpbas array + necpbas) ride alongside.
- **Phase 8 paired `#[cube]` + `*_host()` pattern in `crates/cintx-cubecl/src/math/`**
  — direct template for `bessel.rs` and `radial_quadrature.rs`.

### Established Patterns
- **Per-symbol parity tests with `#[cfg(has_vendor_libcint)]` guard** (Phase 17/18
  D-07 / D-12). 6 ECP tests (4 base + 2 gradient) in a new
  `crates/cintx-oracle/tests/safe_api_ecp_parity.rs`.
- **Tolerance literals at file top** — `const ATOL: f64 = 1e-12; const RTOL: f64 = 0.0;`.
- **`#[cube]` + `*_host()` paired functions** in math modules. Host wrapper used by
  unit tests; `#[cube]` used by launchers.
- **Manifest CSV → `compiled_manifest.lock.json` regeneration via xtask** — adding
  rows is a two-step: edit CSV, run `xtask manifest-audit --update`.
- **`canonical_family` string drives `resolve_family_name` arm** — Phase 13's
  `"f12"` precedent. ECP gets `"ecp"`.
- **Typed `FacadeError` variant per failure class** (Phase 18 D-04 added
  `UnsupportedAoSymmetry`; D-06 here adds `MissingEcpBasis`).
- **Optional fields on core typed structs** — `BasisSet::ecp_shells` mirrors the
  pattern of optional fields on existing types (e.g., `BasisMeta::shell_offsets`
  growing with the basis, optional `Span` on `ExecutionOptions`).
- **Phase 12 multi-component spinor transforms exist but are heavy** — explains
  why D-12 defers ECP spinor to a follow-up.

### Integration Points
- `crates/cintx-core/src/basis.rs` — add optional `ecp_shells` field + accessor
  (D-03). Constructor `try_new_with_ecp(atoms, shells, ecp_shells)` is the new
  ECP-aware entry point.
- `crates/cintx-core/src/shell.rs` (or new `ecp.rs`) — new `EcpShell` struct
  (D-04). `Clone, Debug, PartialEq`. Owned exponents/coeffs (`SmallVec`).
- `crates/cintx-core/src/lib.rs` — re-export `EcpShell`.
- `crates/cintx-compat/src/raw.rs` — add ECP slot constants (`RADI_POWER_OF`,
  `SO_TYPE_OF`, `ECP_BAS_SLOTS = 8`, `PTR_ECPBAS_OFFSET`, `PTR_NECPBAS`).
  Add `EcpBasArray` typed view (D-05). Extend `eval_raw` dispatch for
  `int1e_ecp_*` symbols.
- `crates/cintx-cubecl/src/math/bessel.rs` — NEW. Modified spherical Bessel
  functions for Type-2 (D-07). `#[cube]` + `*_host()` paired functions.
- `crates/cintx-cubecl/src/math/radial_quadrature.rs` — NEW. Gauss-Chebyshev
  (Type-2 radial integral) + Gauss-Hermite (Type-1 radial expansion). `#[cube]`
  + `*_host()` paired functions.
- `crates/cintx-cubecl/src/math/mod.rs` — register the two new modules.
- `crates/cintx-cubecl/src/kernels/ecp.rs` — NEW. `launch_ecp(plan, key, out)`
  with internal Type-1 / Type-2 branch (D-08). Cart-to-sph via existing transform.
  Gradient path branches on operator name (`int1e_ecp_ipnuc_*` → 3-component
  output, scalar `int1e_ecp_*` → 1-component output).
- `crates/cintx-cubecl/src/kernels/mod.rs` — add `"ecp" => ecp::launch_ecp` arm
  to `resolve_family_name` and `supports_canonical_family`. Update
  `unresolved_families` if needed (probably not — ECP is stable, not feature-gated).
- `crates/cintx-ops/src/generated/api_manifest.csv` — add 6 new rows
  (`int1e_ecp_cart`, `int1e_ecp_sph`, `int1e_ecp_ipnuc_cart`, `int1e_ecp_ipnuc_sph`,
  spinor variants explicitly NOT added per D-12). Regenerate
  `compiled_manifest.lock.json` via xtask.
- `crates/cintx-rs/src/api.rs::SessionRequest::query_workspace` — add
  `MissingEcpBasis` preflight check next to the existing aosym preflight.
- `crates/cintx-rs/src/error.rs` — add `FacadeError::MissingEcpBasis { operator: String }`
  variant at end of variants.
- `libcint-master/src/ecp.c` (NEW vendor) — upstream libcint 6.1.3 ECP source
  imported here. `crates/cintx-oracle/build.rs` extends the `cc` source list.
- `crates/cintx-oracle/src/fixtures.rs` — NEW `build_cu_lanl2dz()` builder
  function. Mirrors `build_h2o_sto3g` PTR_ENV_START-aligned env layout, with
  ecpbas array packed after the standard bas table.
- `crates/cintx-oracle/src/compare.rs` (or sibling `vendor_ffi.rs`) — extern "C"
  declarations for the 4 new vendor symbols (`int1e_ecp_cart`, `int1e_ecp_sph`,
  `int1e_ecp_ipnuc_cart`, `int1e_ecp_ipnuc_sph`). libECP secondary symbols are
  separate (D-02).
- `crates/cintx-oracle/tests/safe_api_ecp_parity.rs` (NEW) — 6 per-symbol tests
  (4 base + 2 gradient) at atol=1e-12 vs vendored libcint, full Cartesian
  product over Cu/LANL2DZ shells.
- `crates/cintx-oracle/tests/ecp_libecp_crosscheck_parity.rs` (NEW, optional) —
  secondary cross-check. `#[ignore]` + `CINTX_LIBECP_ORACLE=1` opt-in (D-02).
- `.planning/notes/libcint-ecp-vendor-subset.md` — if researcher decides to
  vendor a minimal subset rather than the full upstream tree.

</code_context>

<specifics>
## Specific Ideas

- **ROADMAP's "Cu/LANL2DZ already present in oracle test corpus" claim is wrong** —
  no Cu basis or ECP fixture exists in `crates/cintx-oracle/src/fixtures.rs` today.
  The fixture is built fresh in this phase. Researcher confirms the LANL2DZ
  parameters from a canonical source (PySCF basis library, basissetexchange.org,
  or the original Hay & Wadt 1985 JCP papers) and documents the source in
  fixture rustdoc.
- **libcint 6.1.3 may NOT distribute `ecp.c` with the main package** — historically
  libcint's ECP is a separate optional module sometimes packaged with PySCF rather
  than libcint proper. Researcher verifies whether libcint 6.1.3 includes `ecp.c`
  in its release tarball. If NOT, the byte-identity reference shifts to one of:
  (a) PySCF's libcint-bundled ECP, (b) chrr/libECP. This is a research-time
  blocker that must resolve before plan-phase commits to D-01's vendor source.
- **Type-1 vs Type-2 algorithmic separation** — Type-1 is local (a sum of
  Gaussian-multiplied radial functions, evaluated as a sum of one-electron
  integrals with Coulomb-like radial parts). Type-2 is semi-local
  (l-dependent angular projector × Bessel-modulated radial Gaussian-Gaussian
  integral). The kernel internal branch (D-08) is on `EcpShell::ecp_type`, NOT on
  the operator name — `int1e_ecp_*` covers both types in a single matrix output;
  each ECP shell contributes Type-1 or Type-2 depending on its own marker.
- **Cu/LANL2DZ has 10 ECP electrons removed (core)** — d-shell (3s, 3p, 3d are
  core; valence 4s, 4p, 4d, 4s/4p/4d for the basis). LANL2DZ ECP has projectors
  for l = 0, 1, 2, 3 (the highest projector being a generic "rest" channel).
  Cu basis has multiple shell types so the parity test should exercise s/p/d
  bra/ket × s/p/d ECP projector combinations.
- **`int1e_ecp_*` returns a real f64 matrix** for cart/sph and is added to the
  existing 1e contraction by callers — it is NOT a stand-alone Hamiltonian
  term; it is the ECP contribution to V_nuc. Documented in kernel module rustdoc.
- **Spinor representation is acutely interesting for Type-2** — Type-2 ECP is
  literally a spin-orbit operator when used with j-coupled (κ-indexed) basis
  functions. Spinor ECP is the "right" representation physically. D-12 defers it
  not because it's wrong but because the multi-component spinor transform infra
  needs more work to land cleanly. Worth surfacing as a high-leverage v1.4
  candidate.
- **Phase 19 produces 5-6 plans** roughly: (1) math infra (bessel + radial_quad)
  + manifest CSV expansion; (2) `EcpShell` + `BasisSet::ecp_shells` extension +
  raw compat slot constants + `EcpBasArray` view; (3) Type-1 kernel + cart/sph
  parity; (4) Type-2 kernel + cart/sph parity; (5) gradient variants + parity;
  (6) optional libECP secondary oracle. Planner decides exact slicing.
- **D-12 ECP spinor deferral cross-cuts with Phase 12 spinor transform infra** —
  the same multi-center spinor transform machinery that Phase 12 closed for
  2e/2c2e/3c2e is what a future "ECP spinor" phase would need to extend with
  Type-2 spin-orbit coupling. Note in deferred ideas.

</specifics>

<deferred>
## Deferred Ideas

- **`int1e_ecp_spinor` and `int1e_ecp_ipnuc_spinor` oracle parity sweep.** D-12
  defers spinor representation to a follow-up phase. Type-2 ECP is naturally
  spin-orbit-like; spinor IS the physically-right representation. Multi-component
  spinor transform for ECP needs its own evaluator work parallel to Phase 12.
  Candidate for a v1.4 "spinor ECP" or "spin-orbit closure" phase.
- **`int1e_ecp_iprinv_*` and other higher-derivative ECP variants** — upstream
  libcint provides additional derivative variants beyond `_ipnuc`. Not in the
  pyscf_rs immediate ask (issue #11 names only `int1e_ecp_*` and `int1e_ecp_ipnuc_*`).
  Add when a downstream consumer asks for them.
- **Lighter-atom fixture validation before Cu/LANL2DZ.** E.g., Na/SBKJC,
  K/CRENBL — simpler ECPs (no d-shell projectors) to validate Type-1 + Type-2
  correctness on a less-tangled test case before Cu. Captured as a spike seed
  in `.planning/spikes/ecp-fixture-validation.md`. Optional researcher add.
- **libECP secondary cross-check as a CI-required gate** — D-02 keeps it
  non-blocking this phase. Promote to required if cintx's libcint vendor reveals
  any tolerance drift versus libECP that the team wants to track. Phase 15
  unification precedent suggests not loosening atol below 1e-12; libECP cross-check
  becomes informational metadata.
- **Multi-fixture parity sweep** (multiple basis sets per atom; multiple
  pseudopotential families). Cu/LANL2DZ is enough to prove Phase 19 correctness.
  Add when CI budget allows or a regression motivates it.
- **`SessionRequest` builder pattern (`.with_ecp_basis(ecp)`)** — D-06 keeps
  ECP basis on `BasisSet` rather than on `SessionRequest`. If a future consumer
  needs to swap ECP basis without rebuilding the whole `BasisSet`, revisit and
  add a builder method then.
- **Shared chunk-loop helper between safe API and compat raw path** (Phase 17
  D-03 / Phase 18 deferred). Still deferred; not relevant to Phase 19's correctness
  work. Candidate for a v1.3 polish phase or v1.4.
- **GPU-side Bessel function evaluation tuning** — the new `bessel.rs` module
  uses a straightforward series + asymptotic expansion split. Performance tuning
  (table-based interpolation, FMA-fused recurrences) deferred until ECP becomes
  a measurable hot path.
- **Type-1 and Type-2 gradient sharing common code with the base kernels** —
  D-11 puts gradient and base in the same launcher with internal branching.
  A future refactor could extract a shared Cartesian-derivative helper if other
  1e gradient operators land (currently only `int1e_*_ip*` exist in 2e/3c-family
  manifest). Phase 20 candidate if a 1e gradient layer becomes a thing.
- **Pre-screening for negligible ECP shell contributions** — performance
  optimization. ECP frequency is 1 per molecule per SCF, so screening pays off
  less than for 2e ERIs. Defer until measured.

</deferred>

---

*Phase: 19-int1e-ecp-type1-type2-evaluator*
*Context gathered: 2026-05-12*
