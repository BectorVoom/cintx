# Phase 22: Gauge-Origin Env Slot (Gap A — `PTR_COMMON_ORIG`) - Context

**Gathered:** 2026-05-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Plumb the `PTR_COMMON_ORIG` gauge-origin env slot (`env[1..3]`) end-to-end on the
Phase-21 `PTR_RINV_ORIG` precedent, and stand up the non-zero gauge-origin oracle
fixture that will become the parity gate for moment integrals (Phase 24) and GIAO
families (Phase 26).

**In scope (FND-01 — foundation slot only):**
- `OperatorEnvParams.common_orig: Option<[f64;3]>` field (`cintx-runtime/src/planner.rs`), mirroring `rinv_orig`.
- `raw.rs::eval_raw` env-read of `env[1..3]` (guarded on `env.len() >= PTR_COMMON_ORIG + 3`), mirroring the `PTR_RINV_ORIG` block.
- `PTR_COMMON_ORIG` const (= 1) in `cintx-compat/src/raw.rs`.
- A validator function for gauge-origin params (finiteness check — see D-01) + unit tests.
- `with_common_origin`-style setter on the safe-API options/builder (`ExecutionOptions.common_orig`, `cintx-rs/src/builder.rs`), mirroring `with_rinv_origin`.
- A committed **non-zero** gauge-origin oracle fixture (atm/bas/env with `env[1..3] != 0` + vendor-reference harness scaffolding) as data infrastructure.
- Env round-trip (raw↔plan) + validator unit tests.

**Out of scope (handled by later phases or by precedent):**
- Any moment / GIAO **kernel** — Phases 24 (MOM) and 26 (GIAO) consume this slot; none built here.
- The `_origj` (origin-on-ket-center) convention — orthogonal to the env slot; moment phase's concern (D-04).
- capi enum variants + legacy `cint*` wrappers — excluded for all v1.4 families per REQUIREMENTS L74.

</domain>

<decisions>
## Implementation Decisions

### Validation semantics (gray area: gauge-origin diverges from rinv)
- **D-01:** `common_orig == None` means **default to `[0,0,0]`**, NOT an error. This is libcint-faithful — an unset env reads as zero and `[0,0,0]` is the standard/default gauge origin (and the common dipole-at-origin case). The kernel/plan consumes the origin via `common_orig.unwrap_or([0.0; 3])`. The FND-01 "validator gate" is therefore a **finiteness check** — reject `NaN`/`inf` when `Some(...)` — **not** a presence check. This is the deliberate divergence from `validate_rinv_orig_env_params`, which rejects `None` because rinv has no sensible default. Unit tests cover: default-is-None, None→no-error (defaults to zero), non-finite→`InvalidEnvParam`.

### Operator recognition (gray area: no consumer family exists in this phase)
- **D-02:** Phase 22 plumbing is **operator-agnostic** — no operator-name matching is wired now. The field, env-read, finiteness validator, and setter exist independent of any operator predicate. Phases 24 (moments) and 26 (GIAO) add their own operator-name matching / kernel threading when they register `int1e_r` / GIAO families. Rationale: a `.contains()` name-list (the rinv structure) would match no dispatchable operator until those phases land — a dead list. This pairs naturally with D-01 (no presence-rejection means nothing needs to know "which operators require an origin" yet).

### Verification depth (gray area: no consuming kernel to drive the fixture)
- **D-03:** Phase 22 verifies the **slot**, not a family. Deliverables: env `[1..3]` round-trips raw↔plan, the validator behaves (D-01 cases), and the `with_common_origin` setter populates the plan. The non-zero gauge-origin fixture is **built + committed as data/harness infrastructure** (atm/bas/env with `env[1..3] != 0`, plus the vendor-reference scaffolding) but **real byte-identity parity is exercised when MOM-01 / GIAO land** (Phases 24/26). No kernel work in this phase; no MOM-01 scope pull-forward. This matches FND-01 literally ("a non-zero gauge-origin oracle fixture exists", "Env round-trip + validator unit tests pass").

### Scope of the slot abstraction
- **D-04:** Phase 22 covers the `PTR_COMMON_ORIG` slot **only**. The `_origj` variants (`int1e_r_origj`, etc. in MOM-01/02) place the origin at the ket basis center — a kernel-side coordinate choice, orthogonal to fixed-env-slot plumbing — and are the moment phase's concern. Do **not** generalize the field/setter into an origin-selection mode now.

### Claude's Discretion
- Exact non-zero origin value(s) and the shell-tuple/molecule corpus for the fixture (H2O/STO-3G precedent is the natural default; non-zero `env[1..3]` is the only requirement).
- Whether the validator lives as a standalone `validate_common_orig_env_params` (mirroring `validate_rinv_orig_env_params`) or folds into a shared origin-validation helper — implementer's call, as long as the D-01 semantics and unit tests hold.
- Whether the fixture's vendor-reference harness is a no-op-now stub or a fully-wired `vendor_*` call that consuming phases just point a kernel at.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### env-slot plumbing precedent (`PTR_RINV_ORIG` → `PTR_COMMON_ORIG`) — the template to mirror
- `crates/cintx-compat/src/raw.rs:34` — env-slot map comment (`PTR_COMMON_ORIG = 1..3`); `:45-49` — `PTR_RINV_ORIG` const + doc (model for a new `PTR_COMMON_ORIG = 1` const); `:599-611` — the `eval_raw` env-read + validate block to clone for `env[1..3]`.
- `crates/cintx-runtime/src/planner.rs:50-53` — `OperatorEnvParams.rinv_orig` field (add `common_orig` alongside).
- `crates/cintx-runtime/src/validator.rs:173-199` — `validate_rinv_orig_env_params` (the structural model; note D-01 changes the semantics from presence→finiteness); `:385-426` — its unit-test patterns.
- `crates/cintx-runtime/src/options.rs:118-122` — `ExecutionOptions.rinv_orig` field + doc (add `common_orig`).
- `crates/cintx-rs/src/builder.rs:100` — `with_rinv_origin` setter (model for `with_common_origin`).

### Manifest + surface (v1.4 per-family surface scope)
- `crates/cintx-ops/generated/compiled_manifest.lock.json` (source of truth) + `crates/cintx-ops/build.rs` (regenerator) — no new family rows needed for the slot itself; relevant when MOM/GIAO register.
- REQUIREMENTS.md L74 — per-family surface decision: manifest/RawApiId/kernel/vendor-FFI/oracle only; NO capi enum variants, NO legacy `cint*` wrappers.

### Oracle
- `crates/cintx-oracle/src/vendor_ffi.rs` — FFI wrappers around vendored libcint 6.1.3 (where a gauge-origin vendor reference call would live).
- `crates/cintx-oracle/tests/*_parity.rs` — `#[cfg(has_vendor_libcint)]` byte-identity tests; double-gated on `--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1` (parity silently skips without both).
- `crates/cintx-oracle/tests/center_2c2e_parity.rs:56` — existing env-slot-map comment referencing `PTR_COMMON_ORIG`.

### Requirement + roadmap
- `.planning/REQUIREMENTS.md` L78 — **FND-01** (the requirement this phase satisfies).
- `.planning/ROADMAP.md` — Phase 22 line.
- `.planning/research/SUMMARY-v1.4.md` — milestone research source cited by REQUIREMENTS.
- `.planning/phases/21-coulomb-gradient-intors/21-CONTEXT.md` — the `PTR_RINV_ORIG` precedent context (D-01/GRAD-01 rinv plumbing).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- The entire `PTR_RINV_ORIG` plumbing (Phase 21, GRAD-01) is a field-for-field template: field → env-read → validator → setter. `common_orig` clones it with one deliberate semantic change (D-01: finiteness, not presence).
- `PTR_ENV_START = 20` and the documented global-parameter map already exist in `raw.rs` — `PTR_COMMON_ORIG = 1` slips into the same const block.

### Established Patterns
- Validator functions take `(operator_name, &OperatorEnvParams) -> Result<(), cintxRsError>` and return `cintxRsError::InvalidEnvParam { param, reason }`. The gauge-origin validator follows this signature/error shape.
- Env-read blocks guard on `env.len() >= PTR_X + 3` before indexing (out-of-bounds safety). Mirror for `env[1..3]`.

### Integration Points
- `eval_raw` (raw/compat path) reads `env[1..3]` and populates `plan.operator_env_params.common_orig`.
- `ExecutionOptions.common_orig` → `operator_env_params.common_orig` on the plan (safe-API path).
- Future: consuming kernels (Phase 24/26) read `common_orig.unwrap_or([0.0;3])`; Phase 22 does not add that read.

</code_context>

<specifics>
## Specific Ideas

- The semantic contrast with rinv is the crux of the phase: rinv rejects `None` (no default origin exists); gauge-origin defaults `None`→`[0,0,0]` (libcint reads unset env as zero). Capture this explicitly so the planner/implementer does not blindly copy `validate_rinv_orig_env_params`'s presence-rejection.
- The fixture's whole point is *non-zero* `env[1..3]` — a zero gauge origin would be indistinguishable from the default and would not prove the slot is actually read. The fixture must use a non-trivial origin.

</specifics>

<deferred>
## Deferred Ideas

- `_origj` / origin-on-ket-center moment variants (`int1e_r_origj`, etc.) — Phase 24 (MOM), kernel-side origin choice, not an env slot (D-04).
- Live byte-identity parity through a real consuming kernel — Phases 24 (MOM-01) / 26 (GIAO) drive the fixture; Phase 22 only builds it (D-03).

### Reviewed Todos (not folded)
- `rys-nroots-ge6-wheeler-fallback.md` — matched on keyword "phase" but is **FND-02** (Phase 25), an independent requirement; not gauge-origin work.
- `oracle-cart-offset-vendor-zero.md` — the pre-existing `CINTshells_cart_offset[4]` (cintx=8 vendor=0) baseline test issue; unrelated to the `PTR_COMMON_ORIG` slot.

</deferred>

---

*Phase: 22-gauge-origin-env-slot-gap-a-ptr-common-orig*
*Context gathered: 2026-05-29*
