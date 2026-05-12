---
phase: 19-int1e-ecp-type1-type2-evaluator
plan: 03
subsystem: typed-api
tags: [ecp, basis-set, operator-id, facade-error, manifest-agreement, raw-compat]

requires:
  - phase: 19-int1e-ecp-type1-type2-evaluator
    provides: "Plan 01 scaffold (EcpShell/EcpChannel placeholder, four ECP rows at OperatorIds 26..=29, vendored PySCF nr_ecp slot constants, Cu/LANL2DZ fixture)"
  - phase: 18-sessionrequest-arity-ge3-dispatch
    provides: "FacadeError typed-variant pattern (D-04 UnsupportedAoSymmetry), SessionRequest::query_workspace aosym preflight as the sibling-insertion-point template"
  - phase: 13-f12-stg-yp-kernels
    provides: "is_f12_family_symbol gate + zeta env-validation pattern in eval_raw — direct template for is_ecp_family_symbol and the AS_NECPBAS guard"
provides:
  - "EcpShell with all 8 typed fields (atom_index, channel, radial_power, nprim, nctr, so_type, exponents, coefficients) + try_new validating length/finiteness/Projected(l) <= ECP_LMAX"
  - "BasisSet::try_new_with_ecp(atoms, shells, ecp_shells) constructor + ecp_shells() accessor — SemVer-preserving (existing try_new keeps its signature, delegates with empty ECP slice)"
  - "OperatorId::INT1E_ECP_{CART,SPH,IPNUC_CART,IPNUC_SPH} typed constants at integers 26..=29 (derived from regenerated OPERATOR_DESCRIPTORS, not hardcoded) + OperatorId::is_ecp() predicate"
  - "cintx-compat::raw ECP surface: five PySCF-named slot constants (RADI_POWER=3, SO_TYPE_OF=4, AS_ECPBAS_OFFSET=18, AS_NECPBAS=19, ECP_LMAX=5), EcpBasArray<'a> typed view over &[i32] reusing BAS_SLOTS=8 width, four RawApiId::INT1E_ECP_* constants, is_ecp_family_symbol helper, eval_raw dispatch arm that fails fast with cintxRsError::InvalidEnvParam(AS_NECPBAS) when env[AS_NECPBAS] is 0/non-finite/negative"
  - "FacadeErrorKind::MissingEcpBasis discriminant + FacadeError::MissingEcpBasis { operator: String } variant with thiserror Display message; SessionRequest::query_workspace preflight emits MissingEcpBasis when operator.is_ecp() && basis.ecp_shells().is_empty(), failing fast before runtime_query_workspace; operator name resolved through Resolver::descriptor with defensive fallback to OperatorId Display"
  - "INT2E_STG_SPH_OPERATOR_ID and INT2E_IPIP1_SPH_OPERATOR_ID test constants resynced to post-shift values (106 and 116); INT4C1E_CART_OPERATOR_ID=24 preserved unchanged (central preservation invariant)"
  - "Typed ecp_operator_ids_match_constants #[test] in cintx-ops/src/resolver.rs asserting OPERATOR_DESCRIPTORS ↔ cintx_core::operator::OperatorId::INT1E_ECP_* agreement plus int4c1e_cart preservation"
affects:
  - "Phase 19 Plan 04 (Type-1+Type-2 kernel + parity) — can now write SessionRequest::new(OperatorId::INT1E_ECP_SPH, &basis, &shells, options).evaluate() and read the ecpbas slab via EcpBasArray inside launch_ecp"
  - "Phase 19 Plan 05 (gradient parity) — INT1E_ECP_IPNUC_{CART,SPH} typed constants ready, preflight gate live, gradient kernel can dispatch through the same launch_ecp arm"
  - "Phase 19 Plan 06 (libecpint secondary oracle, optional) — EcpBasArray view available for marshaling ecpbas rows into libecpint's call shape"

tech-stack:
  added: []
  patterns:
    - "Pattern: Read-and-derive OperatorId integers from the generated OPERATOR_DESCRIPTORS table (positional pairing OPERATOR_DESCRIPTORS[K].id == OperatorId::new(K)) and enforce manifest agreement via a typed #[test] in the consumer crate — never hardcode the integer in two places."
    - "Pattern: Additive-field discipline on core typed structs — BasisSet::try_new keeps its existing signature; the new try_new_with_ecp constructor adds the ECP-aware entry point. SemVer preserved at the cintx-core API surface."
    - "Pattern: FacadeError variant + FacadeErrorKind discriminant + Display message + sibling preflight in query_workspace, mirroring the Phase 18 UnsupportedAoSymmetry precedent verbatim (variant is facade-only — not emitted by From<cintxRsError>)."

key-files:
  created:
    - ".planning/phases/19-int1e-ecp-type1-type2-evaluator/19-03-SUMMARY.md (this file)"
  modified:
    - "crates/cintx-core/src/ecp.rs — EcpShell with 8 fields + try_new validation + 8 unit tests (placeholder → full struct)"
    - "crates/cintx-core/src/basis.rs — added ecp_shells field, ecp_shells() accessor, try_new_with_ecp constructor + 3 new unit tests"
    - "crates/cintx-core/src/error.rs — added CoreError::EcpAngularMomentumTooHigh variant for the Projected(l > ECP_LMAX) case"
    - "crates/cintx-core/src/operator.rs — added 4 INT1E_ECP_* typed constants + is_ecp() predicate + 3 unit tests"
    - "crates/cintx-compat/src/raw.rs — 5 PySCF-named slot constants, EcpBasArray<'a> view, 4 RawApiId constants, is_ecp_family_symbol helper, eval_raw ECP dispatch arm + 7 unit tests"
    - "crates/cintx-rs/src/error.rs — FacadeErrorKind::MissingEcpBasis discriminant + FacadeError::MissingEcpBasis variant + kind() arm"
    - "crates/cintx-rs/src/api.rs — query_workspace preflight (post-aosym, pre-runtime), INT2E_STG_SPH_OPERATOR_ID resynced 102 → 106, INT2E_IPIP1_SPH_OPERATOR_ID resynced 112 → 116, INT4C1E_CART_OPERATOR_ID = 24 preserved + 5 unit tests"
    - "crates/cintx-ops/src/resolver.rs — typed ecp_operator_ids_match_constants #[test] asserting manifest ↔ constants agreement + int4c1e_cart preservation invariant"

key-decisions:
  - "OperatorId values written into cintx-core/operator.rs were derived directly from OPERATOR_DESCRIPTORS in the regenerated api_manifest.rs (positional pairing): int1e_ecp_cart=26, int1e_ecp_sph=27, int1e_ecp_ipnuc_cart=28, int1e_ecp_ipnuc_sph=29. These match Plan 01 SUMMARY's recorded values exactly."
  - "F12/IPIP1 post-shift values derived from the same OPERATOR_DESCRIPTORS table: int2e_stg_sph → OperatorId::new(106) and int2e_ipip1_sph → OperatorId::new(116). The expected +4 shift held; no plan reconciliation needed."
  - "EcpShell::try_new added a new CoreError variant — EcpAngularMomentumTooHigh { requested, max } — for the Projected(l > ECP_LMAX=5) case. Existing variants (InvalidShellCounts, ShellPrimitiveMismatch, InvalidNuclearDetail) handle the length/finiteness cases and were reused verbatim from Shell::try_new."
  - "EcpBasArray::new reused cintxRsError::InvalidBasLayout (the existing variant RawBasView uses for the same length-not-multiple-of-BAS_SLOTS condition). No new error variant was added in cintx-compat."
  - "eval_raw ECP dispatch arm fails closed with cintxRsError::InvalidEnvParam { param: \"AS_NECPBAS\", reason: ... } when env[AS_NECPBAS] is missing/zero/non-finite/negative. Sibling of the F12 zeta gate at the same insertion point in eval_raw."
  - "FacadeError::MissingEcpBasis is facade-only — not emitted by From<cintxRsError>. Lives entirely in the safe-API layer (SessionRequest::query_workspace) so callers can pattern-match programmatically without round-tripping through the runtime error type."
  - "BasisSet::try_new_with_ecp validates EcpShell::atom_index against atoms.len() using the existing CoreError::MissingAtomIndex variant (same one BasisSet::try_new uses for AO shells). No new variant needed; AO and ECP shells share the same atom-index validation surface."

patterns-established:
  - "Manifest-agreement invariant via typed #[test] in cintx-ops/src/resolver.rs — iterate symbols by name, find each in OPERATOR_DESCRIPTORS, assert descriptor.id == OperatorId::INT1E_ECP_* — catches constant drift at test-run time without relying on grep-based gates."
  - "Read-and-derive over hardcode: OperatorId integers in cintx-core/operator.rs are documented (in rustdoc comment) as derived from OPERATOR_DESCRIPTORS positions. The manifest is the single source of truth; the constants restate the integers for typed access."
  - "Facade preflight chaining order — aosym (Phase 18 D-04) → ECP (Phase 19 D-06) → runtime_query_workspace. Each preflight is independent, fails fast, and never reaches the runtime layer if a typed condition is violated."

requirements-completed: [ECP-03]

# Metrics
duration: 5min
completed: 2026-05-12
---

# Phase 19 Plan 03: Typed-API and Raw-Compat Surfaces for ECP Summary

**EcpShell field set filled, BasisSet::try_new_with_ecp added (SemVer-preserving), four typed OperatorId::INT1E_ECP_* constants landed at 26..=29 (read-and-derived from OPERATOR_DESCRIPTORS), cintx-compat raw layer exposes five PySCF-named slot constants + EcpBasArray view + ECP dispatch arm with AS_NECPBAS guard, FacadeError::MissingEcpBasis variant + SessionRequest::query_workspace preflight wired, F12/IPIP1 test constants resynced (102→106, 112→116) with INT4C1E_CART_OPERATOR_ID=24 preserved, and a typed manifest-agreement #[test] in cintx-ops enforces OPERATOR_DESCRIPTORS ↔ OperatorId::INT1E_ECP_* agreement.**

## Performance

- **Duration:** 5 min (per-task implementation window 2026-05-12T19:18:53 → 19:23:08; doc/summary commit phase ran 2026-05-12T19:50+ on the same day)
- **Started:** 2026-05-12T19:18:53+09:00 (first task commit)
- **Completed:** 2026-05-12T19:23:08+09:00 (third task commit)
- **Tasks:** 3
- **Files modified:** 8 (7 source files + 1 SUMMARY file)

## Accomplishments

- **Filled EcpShell** with all 8 typed fields (`atom_index: u32`, `channel: EcpChannel`, `radial_power: i16`, `nprim: u16`, `nctr: u16`, `so_type: i16`, `exponents: Arc<[f64]>`, `coefficients: Arc<[f64]>`) plus `EcpShell::try_new` that mirrors `Shell::try_new` invariants (length matching, finiteness checks) and adds the `Projected(l) <= ECP_LMAX=5` invariant. 8 unit tests cover positive and negative cases.
- **Extended BasisSet with ECP support** — new `ecp_shells: Arc<[Arc<EcpShell>]>` field, `ecp_shells()` accessor, and the additive `try_new_with_ecp(atoms, shells, ecp_shells)` constructor. The existing `try_new(atoms, shells)` signature is preserved (delegates with empty ECP slice), keeping every existing caller compiling unchanged. 3 new unit tests cover empty-ECP defaulting, ECP attachment, and ECP-atom-index validation.
- **Added four typed OperatorId constants** (`INT1E_ECP_CART`, `INT1E_ECP_SPH`, `INT1E_ECP_IPNUC_CART`, `INT1E_ECP_IPNUC_SPH`) with integer values **26, 27, 28, 29** read directly from `OPERATOR_DESCRIPTORS` in `crates/cintx-ops/src/generated/api_manifest.rs`. Plus a `const fn is_ecp(self) -> bool` predicate that the safe-API preflight uses.
- **Resynced F12/IPIP1 test constants** in `crates/cintx-rs/src/api.rs`: `INT2E_STG_SPH_OPERATOR_ID` 102 → **106**, `INT2E_IPIP1_SPH_OPERATOR_ID` 112 → **116**. `INT4C1E_CART_OPERATOR_ID = 24` preserved unchanged (central preservation invariant of the entire ECP-insertion containment plan).
- **Landed the cintx-compat ECP surface** — five PySCF-named slot constants (`RADI_POWER=3`, `SO_TYPE_OF=4`, `AS_ECPBAS_OFFSET=18`, `AS_NECPBAS=19`, `ECP_LMAX=5`) each with rustdoc citing `vendor/pyscf-nr-ecp/include/nr_ecp.h`, the `EcpBasArray<'a>` typed view over `&[i32]` (reusing `BAS_SLOTS=8` row width per Phase 19 D-05 — no new width constant), four `RawApiId::INT1E_ECP_*` constants, the `is_ecp_family_symbol` helper, and the `eval_raw` ECP dispatch arm that fails fast with `cintxRsError::InvalidEnvParam(AS_NECPBAS)` when `env[AS_NECPBAS]` is zero/non-finite/negative. 7 unit tests cover the slot values, the view contract, the RawApiId constants, the helper, and the dispatch guard.
- **Added FacadeError::MissingEcpBasis variant** in `cintx-rs/src/error.rs` (kind discriminant + variant + `kind()` arm) and the matching preflight in `SessionRequest::query_workspace`. The preflight fires when `self.operator.is_ecp() && self.basis.ecp_shells().is_empty()`, resolves the operator symbol via `Resolver::descriptor` with a defensive fallback to the `OperatorId` `Display` impl so the call never panics on a missing manifest entry, and returns `FacadeError::MissingEcpBasis { operator }` BEFORE any `runtime_query_workspace` call. 5 unit tests cover variant construction, Display message, preflight fire for ECP-op-without-ECP-basis, preflight pass-through for non-ECP operators, and preflight pass-through when ECP shells are attached.
- **Landed the typed manifest-agreement `#[test]`** `ecp_operator_ids_match_constants` in `crates/cintx-ops/src/resolver.rs` — iterates the four ECP symbols by name, finds each in `OPERATOR_DESCRIPTORS`, asserts `descriptor.id == OperatorId::INT1E_ECP_*` for each, then asserts the preservation invariant `int4c1e_cart → OperatorId::new(24)`. Fails fast at test-run time if either the generated manifest or the typed constants ever drift.

## Task Commits

Each task was committed atomically on `main`:

1. **Task 1: Typed ECP shell/basis surface + OperatorId constants + F12/IPIP1 resync + manifest-agreement #[test]** — `f87bf01` (feat)
2. **Task 2: ECP slot constants + EcpBasArray view + eval_raw gate in cintx-compat** — `56bbce8` (feat)
3. **Task 3: MissingEcpBasis facade error + query_workspace preflight in cintx-rs** — `a6bca21` (feat)

**Plan metadata commit:** Final docs commit with this SUMMARY.md, STATE.md update, and ROADMAP.md plan-progress update (sequential, immediately after this file lands).

## Files Created/Modified

### Created

- `.planning/phases/19-int1e-ecp-type1-type2-evaluator/19-03-SUMMARY.md` — this file

### Modified

- `crates/cintx-core/src/ecp.rs` — replaced Plan 01 placeholder with the full `EcpShell` struct (8 fields), `EcpChannel` enum with `Local` and `Projected(u8)` variants, `EcpShell::try_new` validation, `ECP_LMAX = 5` public constant, and 8 unit tests
- `crates/cintx-core/src/basis.rs` — added `ecp_shells: Arc<[Arc<EcpShell>]>` field to `BasisSet`, `try_new_with_ecp` constructor (additive), `ecp_shells()` accessor; updated `try_new` to delegate with an empty ECP slice (SemVer-preserving); added 3 unit tests
- `crates/cintx-core/src/error.rs` — added `CoreError::EcpAngularMomentumTooHigh { requested, max }` variant for the new Projected(l > ECP_LMAX) check
- `crates/cintx-core/src/operator.rs` — added 4 `pub const INT1E_ECP_* : OperatorId = OperatorId::new(N)` constants (N = 26..=29 from manifest) + `pub const fn is_ecp(self) -> bool` predicate + 3 unit tests (positive cases, negative cases, manifest-position assertions)
- `crates/cintx-compat/src/raw.rs` — appended 5 PySCF-named slot constants (`RADI_POWER`, `SO_TYPE_OF`, `AS_ECPBAS_OFFSET`, `AS_NECPBAS`, `ECP_LMAX`), `EcpBasArray<'a>` typed view with `new`, `len`, `is_empty`, `row`, `radial_power`, `so_type`, and `iter_rows`; 4 `RawApiId::INT1E_ECP_*` constants; `is_ecp_family_symbol` helper; the ECP dispatch arm in `eval_raw` that gates on `env[AS_NECPBAS]`; and 7 unit tests
- `crates/cintx-rs/src/error.rs` — added `FacadeErrorKind::MissingEcpBasis` discriminant, `FacadeError::MissingEcpBasis { operator: String }` variant with `thiserror`-generated Display message, and the matching arm in `kind()`
- `crates/cintx-rs/src/api.rs` — inserted the ECP-basis preflight in `SessionRequest::query_workspace` immediately after the aosym preflight and before `runtime_query_workspace`; resynced `INT2E_STG_SPH_OPERATOR_ID` (102 → 106) and `INT2E_IPIP1_SPH_OPERATOR_ID` (112 → 116); `INT4C1E_CART_OPERATOR_ID = 24` preserved unchanged; added 5 unit tests
- `crates/cintx-ops/src/resolver.rs` — added the typed `ecp_operator_ids_match_constants` `#[test]` asserting `OPERATOR_DESCRIPTORS ↔ OperatorId::INT1E_ECP_*` agreement plus the `int4c1e_cart → OperatorId::new(24)` preservation invariant

## Decisions Made

### OperatorId values (read-and-derived, not hardcoded)

Read directly from `OPERATOR_DESCRIPTORS` in `crates/cintx-ops/src/generated/api_manifest.rs`:

| Symbol                  | OperatorId integer | Rustdoc citation                          |
| ----------------------- | ------------------ | ----------------------------------------- |
| `int1e_ecp_cart`        | **26**             | manifest position 26 (Plan 01 regenerate) |
| `int1e_ecp_sph`         | **27**             | manifest position 27                      |
| `int1e_ecp_ipnuc_cart`  | **28**             | manifest position 28                      |
| `int1e_ecp_ipnuc_sph`   | **29**             | manifest position 29                      |

These match Plan 01 SUMMARY's recorded values exactly. The `ecp_operator_ids_match_constants` `#[test]` in `cintx-ops/src/resolver.rs` enforces the manifest ↔ constants invariant at test-run time.

### F12 / IPIP1 post-shift values

Read from the same `OPERATOR_DESCRIPTORS` table:

| Symbol               | Pre-Plan-19 | Post-Plan-19 | Notes                                     |
| -------------------- | ----------- | ------------ | ----------------------------------------- |
| `int4c1e_cart`       | 24          | **24**       | **PRESERVED** (central invariant)         |
| `int2e_stg_sph`      | 102         | **106**      | +4 shift (4 ECP rows inserted ahead)      |
| `int2e_ipip1_sph`    | 112         | **116**      | +4 shift                                  |

The expected +4 shift held exactly; no Plan-01-insertion reconciliation was needed. `INT4C1E_CART_OPERATOR_ID = 24` was verified via `grep -F 'INT4C1E_CART_OPERATOR_ID: u32 = 24' crates/cintx-rs/src/api.rs` (still matches unchanged) — the central preservation invariant of the entire ECP-insertion containment plan from Plan 01.

### New CoreError variant added for ECP

`CoreError::EcpAngularMomentumTooHigh { requested: u8, max: u8 }` was added in `crates/cintx-core/src/error.rs` to support the `EcpShell::try_new` invariant that `Projected(l) <= ECP_LMAX = 5`. Existing variants (`InvalidShellCounts`, `ShellPrimitiveMismatch`, `InvalidNuclearDetail`) handle the length/finiteness cases and were reused verbatim from `Shell::try_new`. The new variant is the only ECP-specific addition; all other validation paths share the existing AO-shell error surface.

### EcpBasArray reused BasArray's existing error variant

The plan's <output> field asked specifically whether `BasArray`'s error variant transfers cleanly to `EcpBasArray::new`. **It does** — `EcpBasArray::new` reuses `cintxRsError::InvalidBasLayout` (the same variant `RawBasView::new` uses for the same length-not-multiple-of-BAS_SLOTS condition). No new error variant was added in cintx-compat. This matches Phase 19 D-05's "ecpbas reuses BAS_SLOTS=8 width" decision — the slab shape contract is identical to ordinary `bas` rows, so the error contract is identical too.

### Facade preflight chaining order

Preflights in `SessionRequest::query_workspace` now run in this order:

1. **aosym** (Phase 18 D-04) — returns `FacadeError::UnsupportedAoSymmetry` for non-S1 packings
2. **ECP** (Phase 19 D-06, this plan) — returns `FacadeError::MissingEcpBasis` when `operator.is_ecp() && basis.ecp_shells().is_empty()`
3. `runtime_query_workspace` (runtime layer)

Each preflight is independent, fails fast, and never reaches the runtime layer if a typed condition is violated. `FacadeError::MissingEcpBasis` is facade-only — not emitted by `From<cintxRsError>`, so callers can pattern-match the variant programmatically without round-tripping through the runtime error type.

### Operator-symbol resolution in MissingEcpBasis

The preflight resolves the canonical operator symbol via `Resolver::descriptor(self.operator).map(|d| d.operator_symbol().to_string())` with a defensive `.unwrap_or_else(|_| format!("{}", self.operator))` fallback to the `OperatorId` `Display` impl. This satisfies threat register T-19-07 (the `operator` field is bounded to the canonical manifest symbol, never user-supplied text) and T-19-09 (the fallback ensures the safe API never panics on a missing manifest entry).

## Deviations from Plan

None — plan executed exactly as written.

The plan's `<action>` block included a Step 0 read-and-derive step for the four ECP OperatorId integers from `OPERATOR_DESCRIPTORS`. The integers found in the regenerated manifest (26, 27, 28, 29) matched Plan 01 SUMMARY's recorded values exactly, so no reconciliation step was needed. Similarly, the expected `int2e_stg_sph → 106` and `int2e_ipip1_sph → 116` shifts held. The `int4c1e_cart → 24` preservation invariant remained intact throughout.

## Issues Encountered

None.

## Verification

All acceptance criteria from the plan are satisfied:

### Build / test gates

- `cargo --locked check -p cintx-core -p cintx-compat -p cintx-rs -p cintx-ops` — exits 0
- `cargo --locked build --workspace` — exits 0 (additive changes do not regress any consumer crate)
- `cargo --locked test -p cintx-core --lib` — **25 tests passed, 0 failed**
- `cargo --locked test -p cintx-compat --lib` — **37 tests passed, 0 failed**
- `cargo --locked test -p cintx-rs --lib` — **18 tests passed, 0 failed**
- `cargo --locked test -p cintx-ops --lib ecp_operator_ids_match_constants` — **1 test passed, 0 failed** (manifest-agreement invariant)

### Grep-based acceptance gates

- `grep -F 'INT4C1E_CART_OPERATOR_ID: u32 = 24' crates/cintx-rs/src/api.rs` → matches (preservation invariant)
- `grep -F 'pub channel: EcpChannel' crates/cintx-core/src/ecp.rs` → matches
- `grep -F 'pub radial_power: i16' crates/cintx-core/src/ecp.rs` → matches
- `grep -F 'pub so_type: i16' crates/cintx-core/src/ecp.rs` → matches
- `grep -F 'ecp_shells: Arc<' crates/cintx-core/src/basis.rs` → matches
- `grep -F 'pub fn try_new_with_ecp' crates/cintx-core/src/basis.rs` → matches
- `grep -F 'pub fn ecp_shells' crates/cintx-core/src/basis.rs` → matches
- `grep -F 'pub const INT1E_ECP_CART' crates/cintx-core/src/operator.rs` → matches
- `grep -F 'pub const INT1E_ECP_SPH' crates/cintx-core/src/operator.rs` → matches
- `grep -F 'pub const INT1E_ECP_IPNUC_CART' crates/cintx-core/src/operator.rs` → matches
- `grep -F 'pub const INT1E_ECP_IPNUC_SPH' crates/cintx-core/src/operator.rs` → matches
- `grep -F 'pub const fn is_ecp' crates/cintx-core/src/operator.rs` → matches
- `grep -F 'pub const RADI_POWER: usize = 3;' crates/cintx-compat/src/raw.rs` → matches
- `grep -F 'pub const SO_TYPE_OF: usize = 4;' crates/cintx-compat/src/raw.rs` → matches
- `grep -F 'pub const AS_ECPBAS_OFFSET: usize = 18;' crates/cintx-compat/src/raw.rs` → matches
- `grep -F 'pub const AS_NECPBAS: usize = 19;' crates/cintx-compat/src/raw.rs` → matches
- `grep -F 'pub const ECP_LMAX: usize = 5;' crates/cintx-compat/src/raw.rs` → matches
- `grep -F 'pub struct EcpBasArray' crates/cintx-compat/src/raw.rs` → matches
- `grep -F 'fn is_ecp_family_symbol' crates/cintx-compat/src/raw.rs` → matches
- `grep -c 'ECP_BAS_SLOTS' crates/cintx-compat/src/raw.rs` → **0** (no new width constant, ecpbas reuses BAS_SLOTS=8 per D-05)
- `grep -c 'MissingEcpBasis' crates/cintx-rs/src/error.rs` → **3** (kind discriminant + variant + kind() arm)
- `grep -F 'self.operator.is_ecp()' crates/cintx-rs/src/api.rs` → matches
- `grep -F 'self.basis.ecp_shells().is_empty()' crates/cintx-rs/src/api.rs` → matches
- `grep -F 'FacadeError::MissingEcpBasis' crates/cintx-rs/src/api.rs` → matches
- `grep -F 'INT2E_STG_SPH_OPERATOR_ID: u32 = 106' crates/cintx-rs/src/api.rs` → matches (post-shift value)
- `grep -F 'INT2E_IPIP1_SPH_OPERATOR_ID: u32 = 116' crates/cintx-rs/src/api.rs` → matches (post-shift value)

## Next Phase Readiness

Plan 19-04 (Type-1+Type-2 kernel + parity) is unblocked:

- `SessionRequest::new(OperatorId::INT1E_ECP_SPH, Representation::Spheric, &basis, shells, options).query_workspace()` is now a legal call shape that fails fast with `FacadeError::MissingEcpBasis` if the caller forgets `ecp_shells`.
- `cintx-compat::raw::EcpBasArray::new(slab)` is the typed view Plans 04/05 read inside the kernel launcher.
- `RawApiId::INT1E_ECP_{CART,SPH,IPNUC_CART,IPNUC_SPH}` constants are ready for the kernel dispatch table.
- The `eval_raw` ECP arm already fails fast with `cintxRsError::InvalidEnvParam(AS_NECPBAS)` when `env[AS_NECPBAS]==0`, so the kernel launcher in Plan 04 will only ever be invoked with a non-empty ecpbas slab.
- The manifest-agreement `#[test]` enforces drift detection at every `cargo test -p cintx-ops` invocation — Plans 04/05 can rename or restructure the `OPERATOR_DESCRIPTORS` table without silently desyncing the cintx-core typed constants.

No blockers carrying forward. The plan stops short of any kernel work — Plan 04 picks up at the launcher level (`crates/cintx-cubecl/src/kernels/ecp.rs::launch_ecp`).

## Self-Check: PASSED

Files verified to exist on disk:
- `crates/cintx-core/src/ecp.rs` (330 lines, ≥ 80 per plan acceptance criterion) ✓
- `crates/cintx-core/src/basis.rs` (271 lines, with try_new_with_ecp at line 71) ✓
- `crates/cintx-core/src/operator.rs` (136 lines, with 4 INT1E_ECP_ constants and is_ecp predicate) ✓
- `crates/cintx-compat/src/raw.rs` (2148 lines, with all 5 slot constants + EcpBasArray + helpers) ✓
- `crates/cintx-rs/src/error.rs` (152 lines, with MissingEcpBasis variant + kind discriminant) ✓
- `crates/cintx-rs/src/api.rs` (1014 lines, with preflight at lines 75-88 and post-shift constants at 565/567) ✓
- `crates/cintx-ops/src/resolver.rs` (553 lines, with ecp_operator_ids_match_constants #[test] at line 521) ✓

Commits verified to exist on `main`:
- `f87bf01` (Task 1) ✓
- `56bbce8` (Task 2) ✓
- `a6bca21` (Task 3) ✓

Substantive acceptance criteria verified:
- `cargo --locked check -p cintx-core -p cintx-compat -p cintx-rs -p cintx-ops` exits 0 ✓
- `cargo --locked test -p cintx-core --lib` 25/25 passed ✓
- `cargo --locked test -p cintx-compat --lib` 37/37 passed ✓
- `cargo --locked test -p cintx-rs --lib` 18/18 passed ✓
- `cargo --locked test -p cintx-ops --lib ecp_operator_ids_match_constants` 1/1 passed ✓
- `cargo --locked build --workspace` exits 0 ✓
- `INT4C1E_CART_OPERATOR_ID = 24` preserved in cintx-rs/src/api.rs ✓
- `INT2E_STG_SPH_OPERATOR_ID = 106` (post-shift) in cintx-rs/src/api.rs ✓
- `INT2E_IPIP1_SPH_OPERATOR_ID = 116` (post-shift) in cintx-rs/src/api.rs ✓
- All 4 OperatorId::INT1E_ECP_* constants present in cintx-core/src/operator.rs ✓
- OperatorId::is_ecp() predicate present and returns true for 26..=29 only ✓
- All 5 PySCF-named slot constants present in cintx-compat/src/raw.rs with rustdoc citing nr_ecp.h ✓
- EcpBasArray<'a> view present with new/len/is_empty/row/radial_power/so_type/iter_rows methods ✓
- 4 RawApiId::INT1E_ECP_* constants present in cintx-compat/src/raw.rs ✓
- is_ecp_family_symbol helper + eval_raw dispatch arm both present ✓
- ECP_BAS_SLOTS NOT present (ecpbas reuses BAS_SLOTS=8 per D-05) ✓
- FacadeErrorKind::MissingEcpBasis discriminant + FacadeError::MissingEcpBasis variant + kind() arm all present in cintx-rs/src/error.rs ✓
- SessionRequest::query_workspace preflight present at lines 75-88 ✓
- ecp_operator_ids_match_constants #[test] present in cintx-ops/src/resolver.rs and exits 0 on the recorded toolchain ✓

---
*Phase: 19-int1e-ecp-type1-type2-evaluator*
*Completed: 2026-05-12*
