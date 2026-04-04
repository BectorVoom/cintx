# Phase 12: Real Spinor Transform (c2spinor Replacement) - Context

**Gathered:** 2026-04-04
**Status:** Ready for planning

<domain>
## Phase Boundary

The cart-to-spinor transform applies correct Clebsch-Gordan coupling coefficients for all angular momenta up to l=4, enabling oracle-verifiable spinor outputs for every base family that supports spinor representation. The amplitude-averaging stub is fully removed. All four CINTc2s_*spinor* variants are implemented with distinct code paths. kappa parameter drives transform selection. Oracle parity against libcint 6.1.3 at atol=1e-12.

</domain>

<decisions>
## Implementation Decisions

### Coefficient source strategy
- **D-01:** Extract Clebsch-Gordan coupling coefficient tables directly from upstream libcint's `c2spinor.c` (`g_c2s_*` arrays), mirroring the approach used in `c2s.rs` which extracted from `cart2sph.c`. This guarantees oracle parity by construction.
- **D-02:** Coefficients live in a separate `c2spinor_coeffs.rs` file within `crates/cintx-cubecl/src/transform/`. The `c2spinor.rs` module imports and applies them. Keeps large tables out of transform logic.

### Transform variant differentiation
- **D-03:** All four `CINTc2s_*spinor*` variants get distinct code paths matching upstream `c2spinor.c`. `ket` vs `iket` differs by conjugation sign; `_sf` (spin-free) vs `_si` (spin-included) differs by which CG coupling matrix is applied. No shared-core-with-flags abstraction.
- **D-04:** Spinor staging buffer maintains the existing interleaved real/imaginary layout `[re0, im0, re1, im1, ...]`. This matches libcint's output format and the existing executor staging contract. Oracle comparison works directly on the flat buffer.

### Oracle testing approach
- **D-05:** Verification is sequenced: land 1e spinor oracle parity first (overlap, kinetic, nuclear attraction), then extend to 2e, 2c2e, 3c1e, 3c2e in a second plan. This isolates transform correctness from kernel complexity.
- **D-06:** Spinor oracle tests expand the existing `oracle_gate_closure.rs` by adding spinor representation to the fixture generation loop. No new test files — consistent with how cart/sph already work.

### kappa dispatch logic
- **D-07:** kappa selects which rows of the Clebsch-Gordan coupling matrix to apply. `kappa < 0` uses the j=l+1/2 block, `kappa > 0` uses the j=l-1/2 block, `kappa = 0` uses both blocks. Matches libcint's `c2spinor.c` behavior exactly. Existing `spinor_len(l, kappa)` in `shell.rs` already has the correct sizing logic.
- **D-08:** Existing stub tests (amplitude-averaging, buffer-length-only) are deleted and replaced with value-correctness tests that compare against known CG-transformed outputs.

### Claude's Discretion
- Internal factoring of coefficient application loops within each variant
- Oracle fixture molecule/shell choices for spinor tests (likely reuse existing H2O/STO-3G fixtures with kappa variants)
- Order of variant implementation within each plan
- Exact plan boundaries between 1e and multi-center family coverage

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Design & requirements
- `docs/design/cintx_detailed_design.md` — Master design document defining API surface, families, and compatibility contracts
- `.planning/REQUIREMENTS.md` — SPIN-01 through SPIN-04 requirement definitions (lines 92-95)

### Spinor transform (current stub to replace)
- `crates/cintx-cubecl/src/transform/c2spinor.rs` — Current amplitude-averaging stub (to be fully rewritten)
- `crates/cintx-cubecl/src/transform/mod.rs` — `apply_representation_transform()` dispatch including Spinor arm
- `crates/cintx-compat/src/transform.rs` — Four CINTc2s_*spinor* compat entry points delegating to stub

### Cart-to-sph reference pattern (to mirror for spinor)
- `crates/cintx-cubecl/src/transform/c2s.rs` — Condon-Shortley coefficient matrices C2S_L0..C2S_L4 and `cart_to_sph_1e()` — the structural template for the spinor implementation

### Shell and kappa infrastructure
- `crates/cintx-core/src/shell.rs` — `spinor_len(l, kappa)` function and `Shell.kappa` field

### Oracle infrastructure
- `crates/cintx-oracle/src/compare.rs` — Oracle comparison logic, tolerance constants, IMPLEMENTED_TRANSFORM_SYMBOLS array
- `crates/cintx-oracle/src/fixtures.rs` — Fixture generation including spinor representation dispatch
- `crates/cintx-oracle/tests/oracle_gate_closure.rs` — Gate closure test to expand with spinor representation
- `crates/cintx-oracle/src/vendor_ffi.rs` — Vendored libcint FFI bindings including spinor helpers

### Executor staging contract
- `crates/cintx-cubecl/src/executor.rs` — Executor staging flow where spinor transform is applied

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `c2s.rs` coefficient extraction pattern: C2S_L0..C2S_L4 const arrays with `[nsph x ncart]` layout — direct template for CG spinor coefficient arrays
- `spinor_len(l, kappa)` in `shell.rs`: already correct kappa→component count mapping
- `apply_representation_transform()` in `transform/mod.rs`: dispatch already routes `Representation::Spinor` to `c2spinor` module
- Oracle fixture generation in `fixtures.rs`: already handles spinor symbols (`int1e_ovlp_spinor`, etc.)
- Vendor FFI spinor helpers: `vendor_CINTcgto_spinor`, `vendor_CINTlen_spinor`, etc. already wrapped

### Established Patterns
- Coefficient tables as `pub const` arrays (`C2S_L0..C2S_L4`) — same pattern for CG spinor matrices
- Transform functions take `&mut [f64]` staging buffer and transform in-place
- Oracle comparison: eval via `eval_raw` → compare flat buffer against vendored libcint at atol=1e-12
- Compat entry points delegate to cubecl transform module — thin wrappers

### Integration Points
- `c2spinor.rs` is called from `apply_representation_transform()` in `transform/mod.rs`
- Four compat wrappers in `transform.rs` need their signatures preserved but internals rewritten to pass `l` and `kappa` through to the real transform
- `oracle_gate_closure.rs` fixture loop needs spinor added to the representation iteration
- `IMPLEMENTED_TRANSFORM_SYMBOLS` in `compare.rs` already lists the four spinor transform symbols

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches. The libcint `c2spinor.c` source is the authoritative reference for coefficient values and variant behavior.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 12-real-spinor-transform-c2spinor-replacement*
*Context gathered: 2026-04-04*
