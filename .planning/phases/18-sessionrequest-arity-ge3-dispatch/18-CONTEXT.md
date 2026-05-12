# Phase 18: SessionRequest Arity ≥3 Dispatch - Context

**Gathered:** 2026-05-12
**Status:** Ready for planning

<domain>
## Phase Boundary

`SessionRequest::evaluate` returns byte-identical libcint values for arity-3
shell tuples `(i, j, k)` and arity-4 shell tuples `(i, j, k, l)` covering the
nine ROADMAP-named operators plus the matching cart/sph variants for every
named family (12 symbols total). Dispatch flows through the existing
arity-generic chunk loop in `SessionQuery::evaluate` (Phase 17 already wired
the real `cintx_cubecl::CubeClExecutor`) and the existing
`cintx-ops::resolver` catalog — no parallel evaluator API is introduced.

The phase also adds an `aosym` knob to `ExecutionOptions` so callers can
*request* a packing convention; only `S1` (no packing) is implemented in
Phase 18, every other variant returns a typed `FacadeError::UnsupportedAoSymmetry`
from `query_workspace`. F-order AO layout is documented as a rustdoc invariant
on `IntegralTensor` and verified implicitly by the oracle parity sweep.

Verification: 12 new per-symbol parity tests in
`crates/cintx-oracle/tests/safe_api_arity{3,4}_parity.rs` at the unified
Phase 15 tolerance (`atol=1e-12, rtol=0.0`) against vendored libcint 6.1.3,
gated by the existing `oracle_parity_gate` CI matrix.

</domain>

<decisions>
## Implementation Decisions

### AO symmetry packing (`aosym`)
- **D-01:** **Ship `s1` only; every other packing returns a typed error.**
  Phase 18 implements no compressed packings. `S1` is the implicit and only
  supported value. `S2ij`/`S2kl`/`S4`/`S8` always return a typed error. This
  satisfies ROADMAP SC#4's "where supported, or returns a typed error" wording.
  Implementing the packings (especially `s8` for `int2e_*`, the pyscf_rs SCF
  hot path) is explicitly deferred — see `<deferred>`.
- **D-02:** **Add `aosym: Option<AoSymmetry>` to `ExecutionOptions`.** Additive
  field; `None` is the default and is treated as `Some(S1)`. SemVer-safe (new
  optional field, no positional-arg break). Lives next to the existing
  `f12_zeta: Option<f64>` and follows the same pattern.
- **D-03:** **`AoSymmetry` enum lives in `cintx-core::operator`.** Variants:
  `S1, S2ij, S2kl, S4, S8`. Re-exported from `cintx-core::lib`. `Some(S1)` and
  `None` are accepted for every arity (1e, 2c2e, 3c1e, 3c2e, 2e, 4c1e). Any
  non-`S1` value is rejected regardless of operator class — because Phase 18
  implements none of them. No operator-class branching this phase.
- **D-04:** **New `FacadeError::UnsupportedAoSymmetry { requested: String }`
  variant, raised in `SessionRequest::query_workspace`.** Distinct from
  `FacadeError::UnsupportedApi` so callers can pattern-match programmatically.
  Raised fail-fast before any kernel or workspace work. `requested` carries
  the packing name (e.g., `"s8"`) for diagnostics.
- **D-05:** **`aosym_error_path` unit test inside `cintx-rs/src/api.rs`'s
  existing `#[cfg(test)]` module.** Per-variant test that builds a
  `SessionRequest` with each non-`S1` aosym value and asserts the right
  `FacadeError::UnsupportedAoSymmetry { requested }` is returned. No vendor
  libcint dependency. Oracle parity sweep tests stay on the implicit-`s1`
  path.

### Operator coverage scope
- **D-06:** **Exact 12 symbols are oracle-verified at `atol=1e-12`:**
  - Arity-3 (8): `int3c1e_cart`, `int3c1e_sph`, `int3c1e_p2_cart`,
    `int3c1e_p2_sph`, `int3c2e_ip1_cart`, `int3c2e_ip1_sph`, `int3c2e_cart`,
    `int3c2e_sph`.
  - Arity-4 (4): `int2e_cart`, `int2e_sph`, `int4c1e_cart`, `int4c1e_sph`.

  Adds the cart variants of the three arity-3 families that ROADMAP only
  named in sph form (`int3c1e`, `int3c1e_p2`, `int3c2e_ip1`); without them the
  cart/sph parity story is asymmetric. Spinor variants are NOT in the parity
  sweep this phase (D-07).
- **D-07:** **Spinor arity-3/4 is "compiled but unverified".** `int2e_spinor`,
  `int3c2e_spinor`, `int3c2e_ip1_spinor` are accepted by
  `SessionRequest::evaluate` and dispatch through the real `CubeClExecutor`
  (Phase 12 transforms + multi-center spinor kernels already landed) — but
  they are NOT byte-identity-gated this phase. Document the status in the
  module rustdoc on `cintx-rs::api`: "spinor arity-3/4 outputs are not
  oracle-gated in Phase 18". Consumers needing byte-identity should defer to
  a follow-up phase or use the compat raw path.
- **D-08:** **Unstable-source arity-3 symbols (`int3c1e_r*_origk` etc.) are
  out of scope.** They remain compiled-and-routable behind
  `unstable-source-api` but are not in the Phase 18 parity sweep. Adds nothing
  to test surface; existing `unstable-source-api` arity-3 parity (Phase 14)
  stays on the compat raw path.
- **D-09:** **`int3c2e_ip1_*` (component_rank=3) uses the same scalar parity
  helper as the other arity-3 tests** — byte-identity on the flattened
  `&[f64]` buffer, no per-component sub-tensor decomposition. The vendor
  libcint reference returns the same flat layout; the comparison is
  shape-agnostic at the byte level. Two extra tests in the arity-3 file (cart +
  sph variants).

### F-order AO layout contract
- **D-10:** **Layout invariant is documented as rustdoc on `IntegralTensor`
  only — no struct field, no new method.** Add a doc block on the
  `IntegralTensor` struct describing: "AO axes are stored in F-order
  (Fortran/column-major). `extents` lists the per-axis sizes in shell-tuple
  order; the leftmost axis varies fastest in `owned_values`. Component axis
  (when present, `component_axis_leading == true`) is the slowest-varying
  axis." No `TensorLayout` enum, no `view_fortran()`, no `ndarray` dep on
  `cintx-rs`. Preserves Phase 17 SemVer discipline.
- **D-11:** **The layout contract is verified implicitly by the oracle parity
  sweep.** Byte-identity vs vendored libcint = F-order match by construction;
  if the layout ever silently drifts, the first parity test fails. No
  dedicated layout-only test (no doctest, no `layout_contract.rs`). Simplest
  enforcement.

### Oracle test shape & fixture
- **D-12:** **Two new test files, split by arity.**
  - `crates/cintx-oracle/tests/safe_api_arity3_parity.rs` — 8 per-symbol tests.
  - `crates/cintx-oracle/tests/safe_api_arity4_parity.rs` — 4 per-symbol tests.
  - `int4c1e_*` tests inside the arity-4 file are individually gated
    `#[cfg(feature = "with-4c1e")]` so the file compiles under every profile.
  - All tests are `#[cfg(has_vendor_libcint)]`-guarded so the files compile
    on systems without the vendored libcint artifact.
- **D-13:** **Fixture: H2O / STO-3G (`build_h2o_sto3g` from
  `crates/cintx-oracle/src/fixtures.rs`).** Same fixture as Phase 17. 5 shells
  total, 7 sph AOs, 7 cart AOs. Cross-fixture diversification deferred.
- **D-14:** **Each test exercises the full Cartesian product of shell tuples
  from the H2O/STO-3G basis.** ~125 arity-3 tuples (5³) and ~625 arity-4
  tuples (5⁴) per test; each evaluation produces a small tensor (mean ~5–50
  elements). Total CI cost should stay well under 1s per test. Per-symbol test
  names with the operator in the test identifier so a failure is trivially
  bisectable. If empirical CI cost exceeds the budget during planning, fall
  back to a deterministic subset documented in the plan.
- **D-15:** **Tolerance, vendor-libcint cfg, and CI integration mirror
  Phase 17.** `atol=1e-12, rtol=0.0` (Phase 15 unified). `#[cfg(has_vendor_libcint)]`
  guard. No new CI job — the new tests run inside the existing
  `oracle_parity_gate` matrix (cpu/wgpu × four profiles). Per-symbol failure
  lines appear in CI output for direct bisection.

### Claude's Discretion
- **Layout doc string — verify before writing.** D-10's rustdoc must match
  the *actual* flat-buffer layout produced by today's planner. Current code
  (`crates/cintx-runtime/src/planner.rs:265-292`) sets
  `extents = [shells[0].ao_per_shell(), shells[1].ao_per_shell(), ...]`
  (shell-tuple order) and `component_axis_leading = true`. Researcher MUST
  verify whether `extents[0]` is the fastest-varying axis (current
  implementation, consistent with libcint F-order) or the slowest (would
  require planner-level changes to align). Phase 17 byte-identity at arity-2
  is strong evidence the current convention is correct, but arity-4
  introduces more dimensions to verify. The rustdoc string is downstream of
  that verification.
- **Shared `collect_safe_api_matrix(operator, repr, &basis, tuple)` helper.**
  Phase 17 D-07 leaned toward a sibling helper for the arity-2 sweep; arity-3/4
  is the natural place to factor it out so the 12 new tests share a single
  collection routine. Default: yes, helper lives in
  `crates/cintx-oracle/tests/common/mod.rs` (or a new `safe_api_helpers.rs`).
  Planner decides exact module name / signature.
- **Whether to add a single smoke test for spinor arity-3/4 dispatch (no
  oracle compare).** D-07 says "compiled but unverified" — a runtime smoke
  test that `evaluate()` returns `Ok(_)` (not `UnsupportedApi`) for
  `int2e_spinor` would catch regressions in spinor routing without committing
  to byte-identity. Default: no test — `cargo check` already covers
  compilation, and Phase 12 spinor parity covers value correctness via the
  raw path. Planner may add a one-liner if it's free.
- **`AoSymmetry` derive trait set.** Likely `Clone, Copy, Debug, PartialEq,
  Eq, Hash` to mirror `Representation`. `Default` impl returns `S1` so
  callers can write `AoSymmetry::default()`. Decide during planning.
- **Where the F-order rustdoc lives — struct docblock on `IntegralTensor`
  only, or also at the `cintx-rs::api` module preamble?** Default: struct
  docblock only (single source of truth). Module preamble may cross-reference
  it.
- **Renaming the existing `aosym` enum / variant case to match Rust's
  convention (`S1`, `S2Ij`, etc.) vs. matching pyscf's lowercase string
  exactly (`s1`, `s2ij`).** Default: `S1, S2ij, S2kl, S4, S8` — preserve
  pyscf's variant name but capitalize the leading letter. `Display` impl
  emits the lowercase pyscf form so error messages read `"aosym=s8"` directly.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase intent and scope
- `.planning/ROADMAP.md` § "Phase 18: SessionRequest Arity ≥3 Dispatch
  (issue #11 Task 2)" — locks goal, success criteria 1–5 (ARITY-01..05),
  named operator list, and pyscf_rs `pyscf-gto/src/intor.rs` downstream-impact
  note.
- `.planning/PROJECT.md` § Constraints — CubeCL is the primary compute
  backend; safe Rust API is first-priority surface; type-safe library errors
  via `thiserror`.
- `.planning/notes/pyscf-rs-as-cintx-consumer.md` — downstream consumer
  context (issue #11). `int2e_*` is the SCF J/K hot path; `int3c2e_*` enables
  density fitting. Two independent oracle gates: cintx-oracle (this repo) and
  pyscf_rs `tests/oracle/`.

### Phase 17 predecessor (locked decisions still apply)
- `.planning/phases/17-real-integral-evaluation-in-safe-api/17-CONTEXT.md` —
  real `cintx_cubecl::CubeClExecutor` is the shared dispatch primitive (D-01);
  no `unsafe` in safe-API path (D-02); chunk-loop duplication with compat is
  acceptable (D-03); per-symbol parity test pattern (D-07); `atol=1e-12,
  rtol=0.0` tolerance (D-09); CI runs inside existing `oracle_parity_gate`
  matrix (D-10).
- `.planning/phases/17-real-integral-evaluation-in-safe-api/17-RESEARCH.md`,
  `17-PATTERNS.md`, `17-VERIFICATION.md` — reference implementation context
  for the cintx-rs surface and oracle parity test patterns.

### Phase 15 tolerance baseline
- `.planning/phases/15-oracle-tolerance-unification-manifest-lock-closure/15-CONTEXT.md`
  — unified oracle tolerance is `atol=1e-12` with the four-profile manifest
  lock. No per-family loosening. New tests adopt this tolerance directly.

### Existing safe-API surface (the code being changed)
- `crates/cintx-rs/src/api.rs` lines 16-288 — `SessionRequest`,
  `SessionQuery`, `query_workspace()`, the chunk-loop `evaluate()` body
  (already arity-generic via `ShellTuple` capacity-4 + `Resolver::descriptor`).
  D-04's `UnsupportedAoSymmetry` check lands inside `query_workspace()`.
- `crates/cintx-rs/src/api.rs` lines 442-456 — `IntegralTensor` and
  `TypedEvaluationOutput`. D-10's F-order rustdoc lands on the
  `IntegralTensor` struct.
- `crates/cintx-rs/src/error.rs` — `FacadeError` enum. D-04 adds
  `UnsupportedAoSymmetry { requested: String }` variant.
- `crates/cintx-rs/src/prelude.rs` — public re-exports. Re-export
  `AoSymmetry` so callers can `use cintx_rs::prelude::*`.
- `crates/cintx-core/src/operator.rs` — destination for new `AoSymmetry`
  enum (D-03). Sits next to `Representation`.
- `crates/cintx-core/src/lib.rs` line 19 — re-export `AoSymmetry` alongside
  `Representation`.
- `crates/cintx-core/src/shell.rs` line 7 — `SHELL_TUPLE_CAPACITY = 4`
  already supports arity-3/4 input without change.
- `crates/cintx-runtime/src/lib.rs` — `ExecutionOptions` struct (D-02 adds
  `aosym: Option<AoSymmetry>` field, default `None`).
- `crates/cintx-runtime/src/planner.rs` lines 53-69, 260-296 —
  `OutputLayoutMetadata`, `build_output_layout()`. Sets
  `extents = shells.map(ao_per_shell)`, `component_axis_leading = true`.
  Researcher verifies this matches libcint F-order convention before drafting
  D-10's rustdoc.
- `crates/cintx-ops/src/resolver.rs:316` — operator routing. All 12 target
  symbols (D-06) already in the catalog.
- `crates/cintx-ops/src/generated/api_manifest.csv` lines 11-25 —
  authoritative arity-3/4 operator list. The 12-symbol set is derivable from
  `arity ∈ {3, 4} AND stability == stable AND helper_kind == operator AND
  forms ∈ {cart, sph}`.

### Existing parity test patterns
- `crates/cintx-oracle/tests/safe_api_arity2_parity.rs` (Phase 17 output) —
  direct pattern source. Per-symbol named tests, `#[cfg(has_vendor_libcint)]`
  guard, fixed-fixture style. The new arity-3/4 files mirror it.
- `crates/cintx-oracle/tests/one_electron_parity.rs` — original per-symbol
  parity pattern. Reusable `count_mismatches` helper.
- `crates/cintx-oracle/tests/center_3c1e_parity.rs`,
  `center_3c2e_parity.rs` — arity-3 parity precedent on the `eval_raw`
  path; useful for shell-tuple iteration patterns.
- `crates/cintx-oracle/src/compare.rs` lines 675-900 — arity-4 vendor
  helpers (`vendor_int2e_sph`, `vendor_int2e_cart`, `shls4` iteration).
  Reusable by `safe_api_arity4_parity.rs`.
- `crates/cintx-oracle/src/fixtures.rs` lines 212-260 — `build_h2o_sto3g`
  fixture and `shells_for_arity(n)` helper.

### Manifest profile / 4c1e gating
- `.planning/phases/11-helper-transform-completion-4c1e-real-kernel/` —
  `with-4c1e` feature semantics. `int4c1e_*` is `feature = "with-4c1e"`,
  `stability = "optional"` in the manifest.
- `crates/cintx-compat/src/raw.rs:816` — `enforce_safe_facade_policy_gate`
  already gates source/profile/F12/4c1e envelopes; reused unchanged in the
  Phase 18 path.

### CI
- `.github/workflows/` `oracle_parity_gate` — existing CI matrix (cpu/wgpu ×
  four profiles). New arity-3/4 tests run inside it without a new job
  (D-15).

### Downstream consumer (verification context, no read needed)
- pyscf_rs `crates/pyscf-gto/src/intor.rs` (sibling path-dep, private repo) —
  primary `SessionRequest` consumer. Phase 18 unblocks `int2e_*` (SCF J/K hot
  path) and `int3c2e_*` (density fitting) on land. Independent secondary
  oracle gate in pyscf_rs `tests/oracle/`.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`cintx_cubecl::CubeClExecutor`** (re-exported at
  `crates/cintx-cubecl/src/lib.rs:26`) — same shared dispatch primitive Phase 17
  wired. Already arity-generic; no change needed for arity-3/4.
- **`ShellTuple::try_from_iter` with `SHELL_TUPLE_CAPACITY = 4`**
  (`cintx-core/src/shell.rs:7`) — supports arity-3 and arity-4 input today
  without change.
- **`Resolver::descriptor` and `Resolver::resolve`** — operator routing
  already covers all 12 target symbols (verified against `api_manifest.csv`
  lines 11-25 and adjacent arity-3 rows).
- **`enforce_safe_facade_policy_gate`** (`cintx-compat/src/raw.rs:816`) —
  source/profile/F12/4c1e envelope checks reused unchanged. The new aosym
  check (D-04) is a *sibling* preflight in `query_workspace`, not a change to
  this helper.
- **`build_h2o_sto3g`, `count_mismatches`, `collect_*_matrix_vendor`** helpers
  in `cintx-oracle` — reusable. Likely factor a `collect_safe_api_matrix`
  helper shared by the new arity-3/4 files (Claude's discretion item).
- **`schedule_chunks`, `HostWorkspaceAllocator`, `ExecutionPlan::new`,
  `ExecutionIo`** in `cintx-runtime` — already arity-generic. No change.
- **`ExecutionOptions::f12_zeta: Option<f64>`** — pattern reference for the
  new `aosym: Option<AoSymmetry>` field (D-02).

### Established Patterns
- **Per-symbol parity tests with `#[cfg(has_vendor_libcint)]` guard.** Phase
  17 D-07 pattern, originally from `one_electron_parity.rs`. New files mirror
  it 1:1.
- **Tolerance literals at file top.** `const ATOL: f64 = 1e-12; const RTOL:
  f64 = 0.0;` (Phase 15 unified).
- **`assert_eq!(mismatches, 0, "...")` style** with per-symbol failure
  messages.
- **Optional knobs on `ExecutionOptions` rather than positional args on
  `SessionRequest::new`** (precedent: `f12_zeta`). `aosym` follows the same
  pattern (D-02).
- **Two-file split per arity-class.** Precedent: separate `one_electron_*`,
  `center_2c2e_*`, `center_3c1e_*`, `center_3c2e_*` parity files. The new
  `safe_api_arity{3,4}_parity.rs` continues this convention.
- **Feature-flag gating at the test level.** Existing pattern:
  `#[cfg(feature = "with-4c1e")]` on individual `#[test]` functions inside a
  single file. The arity-4 file uses this for `int4c1e_*` tests.

### Integration Points
- `crates/cintx-rs/src/api.rs::SessionRequest::query_workspace` — add aosym
  preflight check. Returns `FacadeError::UnsupportedAoSymmetry { requested }`
  for non-`S1` values. No change to the existing chunk loop in
  `SessionQuery::evaluate`.
- `crates/cintx-rs/src/error.rs` — add `UnsupportedAoSymmetry { requested:
  String }` to `FacadeError`. Add at the end of existing variants to keep
  ordinal tests stable.
- `crates/cintx-core/src/operator.rs` — add `AoSymmetry` enum. Derive
  `Clone, Copy, Debug, PartialEq, Eq, Hash`; impl `Default` returning `S1`;
  impl `Display` emitting pyscf's lowercase form (`s1, s2ij, s2kl, s4, s8`).
- `crates/cintx-core/src/lib.rs` — re-export `AoSymmetry`.
- `crates/cintx-runtime/src/lib.rs` — add `aosym: Option<AoSymmetry>` to
  `ExecutionOptions`. Default `None`. Update any `ExecutionOptions::default()`
  or builder helpers.
- `crates/cintx-rs/src/prelude.rs` — re-export `AoSymmetry`.
- `crates/cintx-rs/src/api.rs::IntegralTensor` — add F-order rustdoc
  (D-10). Verify wording against actual flat-buffer layout (researcher).
- `crates/cintx-oracle/tests/safe_api_arity3_parity.rs` (NEW) — 8 per-symbol
  tests (D-06 arity-3 set), full Cartesian shell-tuple sweep (D-14), shared
  `collect_safe_api_matrix` helper if extracted.
- `crates/cintx-oracle/tests/safe_api_arity4_parity.rs` (NEW) — 4 per-symbol
  tests (D-06 arity-4 set). `int4c1e_*` tests gated `#[cfg(feature =
  "with-4c1e")]`.
- No changes required in `cintx-cubecl`, `cintx-ops`, or `cintx-compat`
  (other than the optional shared helper landing in `cintx-oracle` test
  common).

</code_context>

<specifics>
## Specific Ideas

- **F-order rustdoc must match the *actual* flat-buffer layout produced by
  today's planner.** Phase 17's arity-2 byte-identity is strong evidence the
  current convention (`extents = [shells[0], shells[1], ...]` with leftmost
  fastest) is correct, but arity-4 introduces more dimensions to verify.
  Researcher checks at least one arity-4 case before drafting the rustdoc
  string. If the convention is wrong, the fix is a planner-level change (out
  of Phase 18 scope) and the doc string is downgraded to "F-order; exact
  index ordering verified per arity by the parity sweep".
- **`aosym` SC#4 satisfaction is asymmetric by design.** `s1` is implemented
  and verified by the parity sweep; `s2ij`/`s2kl`/`s4`/`s8` are typed-error
  paths only. Both halves are required for SC#4 to read as "where supported,
  or typed error". A future phase implementing the packings flips the
  dispatch table without breaking the error variant — `UnsupportedAoSymmetry`
  simply stops firing for the newly-supported packings.
- **Spinor arity-3/4 ("compiled but unverified", D-07) is a deliberate
  middle ground.** Hard-rejecting spinor + arity ≥ 3 would contradict the
  "compiled-only on this host" precedent set by Phase 16 for
  cuda/metal (different domain, same philosophical pattern). Document it
  prominently in the module rustdoc so consumers know the gate has not run.
- **Full Cartesian shell-tuple sweep (D-14) on H2O/STO-3G.** 5 shells → 125
  arity-3 tuples and 625 arity-4 tuples. Each evaluation produces a small
  tensor (mean ~5–50 elements per tuple after AO product). Total CI cost per
  test should be well under 1s. If empirical cost exceeds the budget during
  planning, the planner falls back to a deterministic subset (e.g., a
  representative diagonal + off-diagonal sample) and documents the choice.
- **`int2e_*` is the single most impactful operator in this phase** for the
  downstream consumer (pyscf_rs SCF J/K). Even without `s8` packing (deferred,
  see `<deferred>`), making it return real byte-identical values via the safe
  API unblocks pyscf_rs's `pyscf-gto/src/intor.rs` wrapper immediately.

</specifics>

<deferred>
## Deferred Ideas

- **aosym packings: `s2ij`, `s2kl`, `s4`, `s8`.** Phase 18 returns typed
  errors for all four. Implementation is the biggest practical follow-up —
  particularly `s8` for `int2e_*` (pyscf_rs SCF hot path, ~8× memory and
  compute reduction). Candidate for a v1.4 "SCF acceleration" phase or
  dedicated "Compressed AO tensor packing" phase.
- **Spinor arity-3/4 oracle parity sweep.** `int2e_spinor`,
  `int3c2e_spinor`, `int3c2e_ip1_spinor` are accepted by `SessionRequest`
  (D-07) but not byte-identity-gated. Add when pyscf_rs (or another
  consumer) actually drives the spinor arity ≥ 3 path.
- **Unstable-source arity-3 symbols (`int3c1e_r*_origk`, etc.) through
  `SessionRequest`.** Gated `unstable-source-api`. Not in the Phase 18 sweep
  (D-08). Add when consumer demand exists; existing Phase 14 raw-path parity
  is the current gate.
- **`view_fortran()` ndarray-backed view method on `IntegralTensor`.**
  Considered for the F-order layout question; rejected to avoid adding an
  `ndarray` dependency to `cintx-rs` (D-10). Revisit if multiple consumers
  request typed strided views.
- **`TensorLayout` enum field on `IntegralTensor`** (e.g., `FortranAo |
  CompressedS8 { … } | …`). Considered for D-10; rejected for SemVer
  discipline. Becomes the natural extension point when compressed packings
  (`s4`/`s8`) land — at which point `IntegralTensor` gains the field and the
  rustdoc invariant generalizes.
- **Shared chunk-loop helper between safe API and compat raw path.** Still
  deferred from Phase 17 D-03; not relevant to Phase 18's correctness work.
  Candidate for a v1.3 polish phase or v1.4.
- **Multi-fixture parity sweep** (e.g., a heavy-atom case for `int4c1e_*`).
  H2O / STO-3G is enough to prove Phase 18 correctness. Add when CI budget
  allows or a regression motivates it.
- **Cross-fixture spinor arity-3/4 parity once D-07 lifts.** Bundle with the
  spinor follow-up phase.

</deferred>

---

*Phase: 18-sessionrequest-arity-ge3-dispatch*
*Context gathered: 2026-05-12*
