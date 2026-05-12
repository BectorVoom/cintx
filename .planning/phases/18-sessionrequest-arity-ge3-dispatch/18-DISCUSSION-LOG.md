# Phase 18: SessionRequest Arity ≥3 Dispatch - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-12
**Phase:** 18-sessionrequest-arity-ge3-dispatch
**Areas discussed:** AO symmetry packing (aosym), Operator coverage scope, F-order AO layout contract, Oracle test shape & fixture

---

## AO symmetry packing (aosym)

### Q1: How much of pyscf's `aosym` surface should Phase 18 implement?

| Option | Description | Selected |
|--------|-------------|----------|
| s1 only + typed error for others | Ship `s1` (no packing) for every operator. Any explicit request for `s2ij`/`s2kl`/`s4`/`s8` returns a typed `FacadeError::UnsupportedApi { requested: "aosym=s8" }`. Smallest scope, matches ROADMAP "where supported, or typed error". | ✓ |
| s1 + s8 for `int2e_*` only | Ship `s1` everywhere plus `s8` packing specifically for `int2e_sph/cart/spinor` (SCF hot path). All other packings return typed errors. Biggest practical SCF win for pyscf_rs without full packing matrix. | |
| Full s1/s2ij/s2kl/s4/s8 surface | Implement every pyscf packing for `int2e_*` and corresponding subsets for 3c2e/4c1e where libcint supports them. Largest scope; conflicts with the "no public API change" Phase 17 lineage. | |

**User's choice:** s1 only + typed error for others.
**Notes:** Defers actual packing implementation to a follow-up phase. pyscf_rs gets functional but unoptimized `int2e_*` immediately.

### Q2: Where does the aosym request enter the safe API?

| Option | Description | Selected |
|--------|-------------|----------|
| Add `aosym: Option<AoSymmetry>` to `ExecutionOptions` | Enum `AoSymmetry { S1, S2ij, S2kl, S4, S8 }` and optional field on `ExecutionOptions`. `None` = implicit `S1`. Non-`S1` errors. Additive — safe under SemVer. | ✓ |
| Add `aosym: AoSymmetry` to `SessionRequest::new` | 6th positional argument or builder method. More discoverable but breaks the constructor signature (source-incompatible). | |
| Skip the surface entirely — always emit `s1` | No `aosym` parameter at all. SC#4's "typed error" clause becomes vacuous. | |

**User's choice:** Add `aosym: Option<AoSymmetry>` to `ExecutionOptions`.
**Notes:** Matches Phase 17's discipline of touching the smallest possible surface; mirrors the existing `f12_zeta: Option<f64>` pattern.

### Q3: Where does `AoSymmetry` live, and which operators accept `Some(S1)`?

| Option | Description | Selected |
|--------|-------------|----------|
| In `cintx-core::operator`, accept `S1`/`None` for every arity | One enum shared across layers. `Some(S1)` and `None` behave identically and are accepted for every arity (1e, 2c2e, 3c1e, 3c2e, 4c1e, 2e). Non-`S1` always rejected regardless of operator. | ✓ |
| In `cintx-rs::api`, only accept `aosym` for arity-4 2e operators | Safe-API-only enum; rejects any non-`None` aosym for non-2e operators. More restrictive but matches pyscf semantics. | |
| In `cintx-core` but locked to s1-only for 2e (1e/3c silently ignore) | Hybrid: 1e and 3c silently ignore aosym; 2e enforces strictly. Silent-ignore is harder to reason about. | |

**User's choice:** In `cintx-core::operator` module, accept `S1`/`None` for every arity.
**Notes:** Simplest mental model; no operator-class branching this phase.

### Q4: How is the typed error surfaced and at which layer is it raised?

| Option | Description | Selected |
|--------|-------------|----------|
| New `FacadeError::UnsupportedAoSymmetry { requested }`, raised in `query_workspace` | Dedicated variant so callers can pattern-match programmatically. Raised before any kernel/workspace work — fail-fast. | ✓ |
| Reuse `FacadeError::UnsupportedApi { requested: "aosym=s8" }` | No new variant, stuff packing name into existing error string. Tightest surface but loses programmatic distinguishability. | |
| Raise inside `evaluate()` after planning, via a new compat policy gate | Sibling `enforce_aosym_policy_gate` helper in `cintx-compat`. More consistent with existing policy-gate pattern but raises later than necessary. | |

**User's choice:** New `FacadeError::UnsupportedAoSymmetry { requested }`, raised in `query_workspace`.
**Notes:** Fail-fast; callers can pattern-match on the dedicated variant.

### Q5: Should oracle tests cover the typed-error path?

| Option | Description | Selected |
|--------|-------------|----------|
| Add an `aosym_error_path` unit test in `cintx-rs/src/api.rs` | Per-variant test asserting `FacadeError::UnsupportedAoSymmetry` for each non-`S1` value. No vendor libcint dependency. | ✓ |
| Skip the error-path test entirely | Trust planner / verification to confirm the variant exists. Loses regression coverage. | |
| Add to `cintx-oracle/tests/safe_api_arity_packing_errors.rs` | Co-locate with parity tests — but misplaces a test that doesn't need vendor libcint. | |

**User's choice:** Add an `aosym_error_path` unit test in `cintx-rs/src/api.rs`.
**Notes:** Cheap, fast, runs on every CI matrix cell. Oracle parity stays on the implicit-`s1` path.

---

## Operator coverage scope

### Q1: Which exact operator set does Phase 18 oracle-verify?

| Option | Description | Selected |
|--------|-------------|----------|
| ROADMAP-named 9 symbols only | Arity-3 sph: `int3c1e_sph`, `int3c1e_p2_sph`, `int3c2e_ip1_sph`, `int3c2e_sph`, `int3c2e_cart`. Arity-4: `int2e_sph`, `int2e_cart`, `int4c1e_sph`, `int4c1e_cart`. | |
| ROADMAP set + all cart/sph variants of named families | Adds `int3c1e_cart`, `int3c1e_p2_cart`, `int3c2e_ip1_cart`. 12 symbols total. Completes cart-form parity for every arity-3/4 family. | ✓ |
| ROADMAP set + cart/sph + spinor variants (full base coverage) | Adds `int2e_spinor`, `int3c2e_ip1_spinor`, `int3c2e_spinor`. ~15 symbols total. Arity-4 spinor is largest test in the suite — CI concern. | |

**User's choice:** ROADMAP set + all cart/sph variants of named families.
**Notes:** 12 symbols total; spinor deferred.

### Q2: What does `SessionRequest::evaluate` do for spinor arity-3/4?

| Option | Description | Selected |
|--------|-------------|----------|
| Accept and dispatch silently — "compiled but unverified" | Chunk loop is arity- and representation-generic; kernels already exist (Phases 9-12). Spinor arity-3/4 should dispatch — just not in the parity sweep. Document in module rustdoc. | ✓ |
| Hard-reject with `FacadeError::UnsupportedApi` | Block spinor + arity≥3 at the policy gate. Safer but contradicts compiled-only precedent. | |
| Add `#[cfg(feature = "unstable-spinor-arity-ge-3")]` gate | New feature flag, opt-in build. Maximally cautious but adds CI surface. | |

**User's choice:** Accept and dispatch silently — "compiled but unverified".
**Notes:** Matches Phase 17 pattern. Module rustdoc states the parity gate status.

### Q3: How is `int3c2e_ip1_*` (component_rank=3) tested?

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse scalar helper; byte-identity on flattened buffer | Treat full buffer as flat `&[f64]`. libcint returns same flat layout. Comparison is shape-agnostic at byte level. | ✓ |
| Add `collect_arity3_components_matrix` helper striding over components | Typed wrapper exposing per-component sub-tensors. More expressive failures but thin justification for 2 of 12 operators. | |
| Defer `int3c2e_ip1_*` to a separate gradient-layer phase | Smaller scope but contradicts ARITY-02 which names `int3c2e_ip1_*` explicitly. | |

**User's choice:** Reuse the scalar helper; assert byte-identity on the flattened buffer.
**Notes:** Matches Phase 11/12 derivative-operator parity test pattern.

---

## F-order AO layout contract

### Q1: How do we express the F-order AO layout contract?

| Option | Description | Selected |
|--------|-------------|----------|
| Module-level doc only — no struct change | Document invariant on `IntegralTensor` rustdoc: 'AO axes are stored in F-order: `extents = [n_i, n_j, n_k, n_l]` with index `i` fastest, matching libcint memory layout. Component axis (if present) leads the AO axes when `component_axis_leading == true`.' Preserves SemVer. | ✓ |
| Add `layout: TensorLayout` enum field to `IntegralTensor` | New `#[non_exhaustive]` enum, additive struct change. Self-documenting at the type level. | |
| Add `view_fortran(&self) -> ArrayView<f64, IxDyn>` method | Typed F-order view via `ndarray`. Most ergonomic but adds a non-trivial new dep. | |

**User's choice:** Module-level doc only — no struct change.
**Notes:** Phase 17 already produces this layout; document the invariant, preserve SemVer.

### Q2: Where exactly does the F-order doc live, and is it verified by a test?

| Option | Description | Selected |
|--------|-------------|----------|
| Doc on `IntegralTensor` struct + verified by oracle parity tests | Rustdoc comment block on `IntegralTensor`; parity tests implicitly verify (byte-identity vs libcint = F-order match by construction). No separate layout-only test. | ✓ |
| Doc on `IntegralTensor` + dedicated `layout_contract.rs` doctest | Adds a doctest constructing a small `int2e` request and asserting flat-buffer index correspondence. Belt-and-suspenders. | |
| Doc on `IntegralTensor` + module-level rustdoc in `cintx-rs::api` | Two doc locations (struct + module preamble) for discoverability. No test. | |

**User's choice:** Doc on `IntegralTensor` struct + verified by oracle parity tests.
**Notes:** Simplest enforcement.

### Q3: How is `component_axis_leading` documented for `int3c2e_ip1_*` (3 components)?

| Option | Description | Selected |
|--------|-------------|----------|
| Document: components prepend the AO axes (`[3, n_k, n_j, n_i]`) | When `component_axis_leading == true`, the component axis is the slowest-varying. Iteration order: component → last AO → first AO. Matches libcint convention. | ✓ |
| Defer component-axis docs to a follow-up phase | Skip docs in Phase 18; just verify byte-identity. Risk: pyscf_rs derivative consumers lack a formal contract. | |
| Always describe entire tensor as F-order with component as fastest axis | Different mental model — needs verification against libcint conventions before committing. | |

**User's choice:** Document: components prepend the AO axes (`[3, n_k, n_j, n_i]`).
**Notes:** The selected wording implies an extent ordering that may differ from today's `extents = [shells[0], shells[1], ...]` shells-order. Researcher MUST verify current flat-buffer layout against libcint convention before drafting the final rustdoc string (flagged in CONTEXT.md `<decisions>` Claude's Discretion).

---

## Oracle test shape & fixture

### Q1: What's the file structure for the 12 new arity-3/4 oracle parity tests?

| Option | Description | Selected |
|--------|-------------|----------|
| Two files: `safe_api_arity3_parity.rs` (8) + `safe_api_arity4_parity.rs` (4) | Split by arity for diagnosability. `int4c1e_*` tests gated `#[cfg(feature = "with-4c1e")]`. Mirrors Phase 17's naming and existing oracle test split. | ✓ |
| Single file: `safe_api_arity_ge3_parity.rs` (12 tests) | All in one file. Simpler navigation but mixes arity-3 and arity-4 cost profiles. | |
| Three files: arity3 + arity4 + arity4_4c1e | Splits 4c1e into its own file. Most discoverable for feature-matrix gating; third file holds only 2 tests. | |

**User's choice:** Two files: `safe_api_arity3_parity.rs` (8 tests) + `safe_api_arity4_parity.rs` (4 tests).
**Notes:** User reselected this option to confirm after initial answer. `with-4c1e` gating done per-test inside the arity-4 file.

### Q2: What fixture, and how do we keep CI cost bounded for arity-4?

| Option | Description | Selected |
|--------|-------------|----------|
| H2O / STO-3G everywhere | Same fixture as Phase 17. 7 sph AOs → 2401 elements per arity-4 shell tuple. Manageable. | ✓ |
| H2O / STO-3G for arity-3, H2 / STO-3G for arity-4 | Adds tiny `build_h2_sto3g` helper (4 sph AOs, 256 elements). Cuts arity-4 cost ~10x. Two fixtures. | |
| H2O / STO-3G + restrict shell-tuple selection | Same fixture but one shell tuple per test rather than full product. Bounded runtime, weaker per-test coverage. | |

**User's choice:** H2O / STO-3G everywhere.
**Notes:** Cross-fixture diversification deferred.

### Q3: What shell tuples does each test exercise?

| Option | Description | Selected |
|--------|-------------|----------|
| Single representative tuple per test | One fixed tuple per test, e.g., `(0, 1, 2)` for arity-3 and `(0, 1, 2, 3)` for arity-4. Matches Phase 17. | |
| Two tuples — one diagonal, one off-diagonal | `(0, 0, 0)` and `(0, 1, 2)` to catch same-shell-repeat behavior plus general case. | |
| Full Cartesian product | Iterate every `(i, j, k)` / `(i, j, k, l)` from the basis. Maximum coverage; ~625 arity-4 tuples per test. | ✓ |

**User's choice:** Full Cartesian product of shell tuples.
**Notes:** 5 shells in H2O/STO-3G → 125 arity-3 / 625 arity-4 tuples per test; mean per-tuple tensor ~5-50 elements. Should stay well under 1s per test. Planner falls back to a deterministic subset if empirically slow.

### Q4: Tolerance, vendor-libcint cfg, and CI integration?

| Option | Description | Selected |
|--------|-------------|----------|
| `atol=1e-12, rtol=0.0`, `#[cfg(has_vendor_libcint)]`, inside existing `oracle_parity_gate` | Phase 15 unified tolerance; vendor-libcint cfg guard for portability; no new CI job. `int4c1e_*` adds `#[cfg(feature = "with-4c1e")]` on top. | ✓ |
| Same as (a) + separate `arity_ge3_oracle_gate` CI job | Isolate arity-3/4 runtime cost from arity-2 / helpers. Pollutes workflow file. Probably premature. | |
| Loosen tolerance to `atol=1e-11` for arity-4 only | Acknowledges nbf^4 accumulation. Contradicts Phase 15 no-per-family-loosening rule. Rejected. | |

**User's choice:** `atol=1e-12, rtol=0.0`, `#[cfg(has_vendor_libcint)]`, runs inside existing `oracle_parity_gate`.
**Notes:** Mirrors Phase 17 D-09/D-10 verbatim.

---

## Claude's Discretion

Captured in CONTEXT.md `<decisions>` § Claude's Discretion. Notable items:

- Verify the current flat-buffer layout produced by `crates/cintx-runtime/src/planner.rs:265-292` against libcint F-order convention before locking the D-10 rustdoc string. The user's choice in F-order Q3 implies a reversed/prepended `extents` ordering, which may not match today's planner code; if it doesn't, the rustdoc downgrades to a more cautious wording and any layout reshape becomes a separate phase.
- Whether to extract a shared `collect_safe_api_matrix(operator, repr, &basis, tuple)` helper for the 12 new tests. Default: yes, lives in `crates/cintx-oracle/tests/common/mod.rs` (or a new `safe_api_helpers.rs`).
- Whether to add a single smoke test that spinor arity-3/4 dispatch succeeds (not byte-identity). Default: no.
- `AoSymmetry` derive set: `Clone, Copy, Debug, PartialEq, Eq, Hash`; `Default = S1`; `Display` emits lowercase pyscf form.
- F-order rustdoc location: struct docblock only (single source of truth). Module preamble may cross-reference.

## Deferred Ideas

Captured in CONTEXT.md `<deferred>`. Summary:

- aosym packings `s2ij`, `s2kl`, `s4`, `s8` — implementation deferred; especially `s8` for `int2e_*` (pyscf_rs SCF hot path).
- Spinor arity-3/4 oracle parity sweep.
- Unstable-source arity-3 (`int3c1e_r*_origk` etc.) through `SessionRequest`.
- `view_fortran()` ndarray view method on `IntegralTensor`.
- `TensorLayout` enum field on `IntegralTensor`.
- Shared chunk-loop helper between safe API and compat (still deferred from Phase 17 D-03).
- Multi-fixture parity sweep (heavy-atom case for `int4c1e_*`).
- Cross-fixture spinor arity-3/4 parity once D-07 lifts.
