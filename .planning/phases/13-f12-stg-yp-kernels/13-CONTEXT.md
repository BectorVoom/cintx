# Phase 13: F12/STG/YP Kernels - Context

**Gathered:** 2026-04-05
**Status:** Ready for planning

<domain>
## Phase Boundary

STG and YP geminal two-electron kernels are implemented as separate dispatch paths with PTR_F12_ZETA env plumbing, covering all 10 with-f12 sph symbols at oracle parity against libcint 6.1.3 at atol=1e-12. Cart and spinor representations remain unsupported for F12 symbols (sph-only enforcement already in place from Phase 3).

</domain>

<decisions>
## Implementation Decisions

### zeta=0 validation
- **D-01:** When `env[9]` (PTR_F12_ZETA) is 0.0 for an F12/STG/YP symbol, the validator rejects with a typed `InvalidEnvParam` error before kernel launch. No silent fallback to plain Coulomb. This is fail-closed, consistent with the existing `UnsupportedApi` pattern.
- **D-02:** The validation check lives in the `ExecutionPlan` validator (cintx-runtime), not in the kernel itself. The kernel can assume zeta > 0.

### Derivative symbol implementation
- **D-03:** Each of the 10 with-f12 sph symbols gets its own kernel entry point in `kernels/f12.rs`. The 5 STG variants (base, ip1, ipip1, ipvip1, ip1ip2) and 5 YP variants are separate launch functions. No shared-core-with-flags abstraction.
- **D-04:** STG and YP base kernels have fundamentally different ibase/kbase routing (per roadmap SC1). Derivative variants within each operator type share the same root-finding but differ in angular momentum increments matching libcint's `cint2e_f12.c` `CINTEnvVars` setup.

### STG root table embedding
- **D-05:** Port `CINTstg_roots` from `libcint-master/src/stg_roots.c` into a new `math/stg.rs` module in cintx-cubecl. Embed `DATA_X` and `DATA_W` tables from `roots_xw.dat` as `static [f64; N]` arrays with the same offset formula: `nroots * 196 * (iu + it * 10)`.
- **D-06:** STG root computation is host-side only (like `rys_roots_host`). Results are uploaded to device as kernel arguments. No device-side Clenshaw recurrence.
- **D-07:** The `COS_14_14` cosine table and Clenshaw recurrence helpers (`_clenshaw_dc`, `_matmul_14_14`, `_clenshaw_d1`) are ported as host-side Rust functions in `math/stg.rs`. The `t = min(t, 19682.99)` clamp from `CINTstg_roots` is replicated exactly.
- **D-08:** Host wrapper function `stg_roots_host(nroots, ta, ua)` follows the established `rys_roots_host` pattern with `_host()` suffix for test accessibility.

### Oracle fixture scope
- **D-09:** Oracle parity fixtures reuse the existing H2O/STO-3G molecule and basis set, consistent with all prior oracle tests. Same PTR_ENV_START-aligned env layout.
- **D-10:** All 10 with-f12 sph symbols tested against vendored libcint at atol=1e-12. Oracle confirms cart and spinor symbol counts for the with-f12 profile are zero (existing sph-only enforcement).

### PTR_F12_ZETA plumbing
- **D-11:** `ExecutionPlan` gains an `operator_env_params` field (or similar struct like `F12KernelParams`) carrying `PTR_F12_ZETA` value extracted from `env[9]` during plan construction when the operator family is with-f12.
- **D-12:** The kernel dispatch in `kernels/mod.rs` gains an `"f12"` arm in `resolve_family_name()` routing to `f12.rs` launch functions. STG vs YP selection is based on the resolved symbol name.

### Claude's Discretion
- Internal factoring of Clenshaw recurrence helpers within `math/stg.rs`
- Exact `OperatorEnvParams` struct layout and naming
- Order of the 10 symbol implementations within plans
- Whether STG/YP share a common pdata setup before diverging at root computation
- Test molecule zeta value choice (non-zero, physically reasonable)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Design & requirements
- `docs/design/cintx_detailed_design.md` — Master design document; F12/STG/YP operator definitions and compatibility contracts
- `.planning/REQUIREMENTS.md` — F12-01 through F12-05 requirement definitions
- `.planning/ROADMAP.md` — Phase 13 success criteria (lines 254-263)

### Upstream libcint reference (vendored)
- `libcint-master/src/stg_roots.c` — CINTstg_roots algorithm: Clenshaw/DCT, COS_14_14 table, t-clamp, roots_xw.dat embedding
- `libcint-master/src/g2e_f12.c` — F12 integral setup: ibase/kbase routing for STG vs YP, PTR_F12_ZETA usage, envvar setup per derivative variant
- `libcint-master/src/cint2e_f12.c` — All 10 with-f12 symbol definitions, CINTEnvVars setup per derivative variant (ip1, ipip1, ipvip1, ip1ip2)

### Prior research (v1.2)
- `.planning/research/STACK.md` — STG root algorithm analysis, roots_xw.dat embedding strategy, PTR_F12_ZETA plumbing approach (lines 19-57)
- `.planning/research/SUMMARY.md` — F12 kernel structure, pitfall warnings (zeta=0 fallback, t-clamp), recommended file layout

### Existing kernel infrastructure (to extend)
- `crates/cintx-cubecl/src/kernels/mod.rs` — `resolve_family_name()` dispatch; add `"f12"` arm
- `crates/cintx-cubecl/src/kernels/two_electron.rs` — Existing 2e kernel as structural reference for pdata/Rys infrastructure
- `crates/cintx-cubecl/src/math/` — Boys, Rys, pdata modules; `stg.rs` will be added here

### Validator & execution plan
- `crates/cintx-runtime/src/validator.rs` — Add InvalidEnvParam check for PTR_F12_ZETA==0
- `crates/cintx-runtime/src/workspace.rs` — ExecutionPlan struct to extend with operator_env_params

### Oracle infrastructure
- `crates/cintx-oracle/src/compare.rs` — Oracle comparison logic, tolerance constants
- `crates/cintx-oracle/src/fixtures.rs` — Fixture generation to extend with F12 zeta parameter
- `crates/cintx-oracle/tests/oracle_gate_closure.rs` — Gate closure test to extend with with-f12 profile
- `crates/cintx-oracle/src/vendor_ffi.rs` — Vendored libcint FFI bindings

### Feature gating
- `crates/cintx-compat/src/raw.rs` — with-f12 profile gate, sph-only enforcement (lines 534-552)
- `crates/cintx-ops/src/resolver.rs` — Manifest resolver with-f12 profile symbol lookup

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `two_electron.rs` kernel: Rys quadrature + pdata infrastructure — STG/YP kernels share the same shell-pair primitive data setup, diverging only at root computation
- `math/rys.rs` + `math/boys.rs`: Established host-side math pattern with `_host()` wrappers and static coefficient tables
- `math/pdata.rs`: Gaussian primitive pair evaluation — reusable for F12 shell pairs
- Oracle fixture generation in `fixtures.rs`: Already handles profile-scoped fixture dispatch
- `resolve_family_name()` dispatch pattern in `kernels/mod.rs`: Clean extension point for `"f12"` arm

### Established Patterns
- Host wrapper + `#[cube]` pair: Every math function has `*_host()` counterpart for testing (Phase 8 convention)
- Static coefficient embedding: `TURNOVER_POINT` in `boys.rs`, polynomial fit tables in `rys.rs` — same approach for `roots_xw.dat`
- PTR_ENV_START-aligned env layout: env user data starts at offset 20; env[8]=PTR_RANGE_OMEGA, env[9]=PTR_F12_ZETA
- Feature-gated kernel dispatch: `#[cfg(feature = "with-4c1e")]` pattern in `kernels/mod.rs` — mirror for `with-f12`

### Integration Points
- `kernels/mod.rs` `resolve_family_name()` — add `"f12"` dispatch arm
- `cintx-runtime/src/validator.rs` — add InvalidEnvParam check for zeta=0 on F12 symbols
- `cintx-runtime/src/workspace.rs` — extend ExecutionPlan with operator_env_params
- `oracle_gate_closure.rs` — extend with with-f12 profile symbols
- `fixtures.rs` — add zeta parameter to F12 fixture env setup

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches. The vendored `libcint-master/src/stg_roots.c` and `g2e_f12.c` are the authoritative references for algorithm behavior and variant routing.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 13-f12-stg-yp-kernels*
*Context gathered: 2026-04-05*
