---
phase: 21-coulomb-gradient-intors
plan: 05
subsystem: kernels
tags: [int2e_ip1, two-electron-gradient, cubecl, oracle-parity, gout_ip1, rys-roots, GRAD-07, F-order]

# Dependency graph
requires:
  - phase: 13-f12-stg-yp-kernels
    provides: gout_ip1 / nabla1i_2e / F12Shape (F12-free derivative math) in f12.rs
  - phase: 10-2e-2c2e-3c1e-3c2e-real-kernels
    provides: launch_two_electron / build_2e_shape / fill_g_tensor_2e / rys_roots_host
  - phase: 18-sessionrequest-arity-ge3-dispatch
    provides: arity-4 safe-API dispatch + component_axis_leading layout (IntegralTensor)
  - phase: 21 (plan 02)
    provides: oracle base-profile parity harness wiring for int2e_ip1 (compare.rs)
provides:
  - int2e_ip1 arity-4 two-electron gradient (3 components) in launch_two_electron_typed
  - byte-identity vs vendored libcint 6.1.3 at atol=1e-12 for s/p/d cart+sph quartets
  - component-leading [3, nl, nk, nj, ni] F-order matching pyscf-gto layout_table.rs
  - vendor_int2e_ip1_{sph,cart} FFI wrappers
  - pub(crate) gout_ip1 / nabla1i_2e / F12Shape reusable by plain-Coulomb gradients
affects: [21-06-int3c2e-ip1, future-gradient-intors, pyscf-rs-consumer]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Plain-Coulomb gradient reuses f12.rs gout_ip1 verbatim on the rys-roots G-tensor (D-04)"
    - "li_ceil = li+1 G-tensor headroom + gout_ip1 at BASE li (documented headroom recipe)"
    - "Interleaved gout[n*3+comp] transposed to component-leading [3,nl,nk,nj,ni] F-order"
    - "Element-for-element vendor byte-identity comparison IS the F-order layout gate (R3)"

key-files:
  created:
    - crates/cintx-oracle/tests/two_electron_ip1_parity.rs
  modified:
    - crates/cintx-cubecl/src/kernels/f12.rs
    - crates/cintx-cubecl/src/kernels/two_electron.rs
    - crates/cintx-cubecl/src/kernels/mod.rs
    - crates/cintx-oracle/src/vendor_ffi.rs
    - crates/cintx-oracle/build.rs

key-decisions:
  - "Bridged TwoEShape→F12Shape via 1:1 field-copy helper (two_e_shape_as_f12) rather than re-deriving gout_ip1 (D-04 verbatim reuse)"
  - "Ungated kernels::f12 module so its F12-free derivative math compiles in the base profile; launch_f12 + the f12 dispatch registration stay with-f12-gated (F12 OPERATORS still rejected in base)"
  - "Added int2e_ip1_{sph,cart} to the oracle bindgen allowlist_function regex (Rule 3) — they are in cint_funcs.h but were absent from the allowlist so ffi bindings were not generated"
  - "int2e_ip1 safe-API arm requires NO new wiring: Phase 18 arity-4 dispatch + component_axis_leading already flow it through once the kernel emits the correct layout (R6 resolved: safe arm shipped, not deferred)"

patterns-established:
  - "Pattern: plain-Coulomb gradient = build_2e_shape(li+1,...) + fill_g_tensor_2e + f12::gout_ip1 + transpose to component-leading"
  - "Pattern: nroots>5 guard fail-closed BEFORE rys dispatch (R2), spinor→UnsupportedApi BEFORE compute (R5)"

requirements-completed: [GRAD-07]

# Metrics
duration: 13min
completed: 2026-05-26
---

# Phase 21 Plan 05: int2e_ip1 Two-Electron Gradient Summary

**int2e_ip1 (arity-4, 3-component) two-electron force — byte-identical to vendored libcint 6.1.3 at atol=1e-12 for s/p/d quartets, reusing f12::gout_ip1 verbatim on the plain rys-roots G-tensor and emitting component-leading [3,nl,nk,nj,ni] F-order.**

## Performance

- **Duration:** ~13 min
- **Started:** 2026-05-26T11:40:19Z
- **Completed:** 2026-05-26T11:53Z
- **Tasks:** 3
- **Files modified:** 5 (+1 created)

## Accomplishments
- The single highest-impact analytical-gradient term (`∇_A <ij|kl>`) now evaluates byte-identically to upstream libcint for every s/p/d cart+sph quartet.
- `gout_ip1` / `nabla1i_2e` / `F12Shape` exposed `pub(crate)` and reused VERBATIM (zero body change) — the gradient math was always F12-free; only the G-tensor source differs (plain `fill_g_tensor_2e` rys roots vs F12 stg roots).
- Output is component-leading `[3, nl, nk, nj, ni]` F-order, validated element-for-element against libcint's own component-leading order (Risk R3 mitigated — no separate layout assertion needed).
- Fail-closed guards: `nroots > 5` → `UnsupportedApi` BEFORE any rys dispatch (R2); `Spinor` → `UnsupportedApi` BEFORE compute (R5).

## Task Commits

1. **Task 0: Confirm pyscf-gto call path + expose gout_ip1/F12Shape/nabla1i_2e** — `1704717` (refactor)
2. **Task 1 (RED): failing int2e_ip1 gradient behavior tests** — `f814c13` (test)
3. **Task 1 (GREEN): implement int2e_ip1 gradient path** — `0ab4e3d` (feat)
4. **Task 2: vendor FFI + byte-identity oracle parity (s/p/d)** — `839f392` (test)

_Task 1 followed TDD: RED (`f814c13`) → GREEN (`0ab4e3d`). No refactor commit needed — GREEN code was clean._

## Files Created/Modified
- `crates/cintx-cubecl/src/kernels/f12.rs` — `gout_ip1`, `nabla1i_2e`, `F12Shape` (+ all fields) flipped to `pub(crate)` with shared-with-plain-Coulomb-gradient rustdoc; zero body change.
- `crates/cintx-cubecl/src/kernels/two_electron.rs` — `launch_two_electron_ip1` gradient path + `two_e_shape_as_f12` bridge + `operator_name()=="ip1"` branch in `launch_two_electron_typed`; 4 unit tests (nroots-guard, component-count, determinism, spinor-reject).
- `crates/cintx-cubecl/src/kernels/mod.rs` — ungated `pub mod f12;` (module always compiles); `launch_f12` + `"f12"` dispatch stay `with-f12`-gated.
- `crates/cintx-oracle/src/vendor_ffi.rs` — `vendor_int2e_ip1_{sph,cart}` wrappers.
- `crates/cintx-oracle/tests/two_electron_ip1_parity.rs` (created) — vendor-gated byte-identity cart+sph parity over s/p/d quartets + cpu-only determinism/nonzero sentinel.
- `crates/cintx-oracle/build.rs` — added `int2e_ip1_{sph,cart}` to the bindgen `allowlist_function` regex.

## Decisions Made
- **R6 finding (pyscf-gto raw vs safe path):** Phase 18 arity-4 safe-API dispatch IS wired (verified via `crates/cintx-rs/src/api.rs` `IntegralTensor.component_axis_leading` + `crates/cintx-oracle/tests/safe_api_arity4_parity.rs`). The `int2e_ip1` SAFE-API arm therefore requires NO new wiring — it flows through the existing arity-4 dispatch + component-axis-leading layout once the kernel emits the correct staging. The oracle harness for `int2e_ip1` was already wired on the base branch by the orchestrator (commits c2c351a/786edf9: `compare.rs::raw_api_for_symbol` + `eval_legacy_symbol`). The raw `eval_raw` arm (which the oracle tests and pyscf-gto's intor.rs use) is independent of Phase 18 (D-11). **Safe-API arm shipped, not deferred.**
- **F12Shape bridge:** chose option (a) from the plan — make `F12Shape`/`gout_ip1`/`nabla1i_2e` `pub(crate)` and field-copy `TwoEShape`→`F12Shape` (1:1) — over inlining, to reuse the EXACT verbatim derivative math (D-04).
- **Transpose stride for [3,nl,nk,nj,ni] F-order:** `cart_blocks[comp*block_len + n] += weight * gout[n*3 + comp]` where `block_len = nfi*nfj*nfk*nfl` and `n` walks `[cl,ck,cj,ci]` (ll slowest, li fastest, as produced by gout_ip1). For sph, `cart_to_sph_2e` runs per component on each `block_len` slice; staging written component-leading `staging[comp*sph_block + iidx + di*(jidx + dj*(kidx + dk*lidx))]` (i fastest).
- **Max l-quartet tested before nroots overflow:** `(d,d,d,d)` → gradient nroots `= (2+1+2+2+2)/2 + 1 = 5` (the ceiling, allowed). All-f `(f,f,f,f)` → nroots 7 > 5 → `UnsupportedApi` (R2-deferred). The oracle sweep skips any quartet with gradient nroots > 5.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Ungated the `kernels::f12` module for base-profile compilation**
- **Found during:** Task 1 (GREEN) — `cargo test` failed with `cannot find f12 in kernels` (the module was `#[cfg(feature = "with-f12")]`).
- **Issue:** `int2e_ip1` is a base-profile operator (manifest rows compiled in `base`), but its kernel needs `gout_ip1`/`F12Shape` which lived in a `with-f12`-only module. The plan assumed the symbols were reachable; they were not in the base profile.
- **Fix:** Removed the `#[cfg(feature = "with-f12")]` gate on `pub mod f12;` in `kernels/mod.rs` (the module's deps — `math::stg`, `validate_f12_env_params` — are already ungated, so it compiles cleanly). Kept `launch_f12` registration + `supports_canonical_family("f12")` gated behind `with-f12`, so F12 OPERATORS remain rejected in the base profile (no behavior change for F12 dispatch).
- **Files modified:** `crates/cintx-cubecl/src/kernels/mod.rs`
- **Verification:** base profile + `with-f12` profile both `cargo check` clean; all 198 cintx-cubecl lib tests pass; existing F12 dispatch tests (`f12_supports_and_resolves_under_with_f12_feature`) still pass.
- **Committed in:** `0ab4e3d` (Task 1 GREEN commit)

**2. [Rule 3 - Blocking] Added int2e_ip1_{sph,cart} to the oracle bindgen allowlist**
- **Found during:** Task 2 — the vendor build failed with `cannot find function int2e_ip1_sph in module ffi`.
- **Issue:** The plan's interface note said "bindgen auto-generates ffi:: bindings; NO supplemental header edit." That was incorrect: the oracle `build.rs` uses an explicit `allowlist_function("...")` regex that did NOT include `int2e_ip1_{sph,cart}`, so no bindings were generated (even though the symbols are declared in `cint_funcs.h`).
- **Fix:** Added `int2e_ip1_sph|int2e_ip1_cart` to the `allowlist_function` regex in `crates/cintx-oracle/build.rs`. No supplemental header declaration needed — both are in `cint_funcs.h` (lines 681-682), which the supplemental header `#include`s.
- **Files modified:** `crates/cintx-oracle/build.rs`
- **Verification:** vendor-gated parity test compiles and passes (`CINTX_ORACLE_BUILD_VENDOR=1`); existing `two_electron_parity` + `safe_api_arity4_parity` vendor tests still pass (no regression).
- **Committed in:** `839f392` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 3 - blocking). Both were prerequisites to make a base-profile operator's kernel + oracle gate compile; neither expanded scope. No architectural changes.
**Impact on plan:** All auto-fixes necessary to complete the planned tasks. The plan's two incorrect assumptions (f12 reachability, bindgen auto-allowlist) were corrected without altering the plan's intent.

## Issues Encountered
- The `component_count` and `determinism` unit tests passed during the RED phase on a size coincidence (the scalar 2e path's staging happened to match the planner-sized component buffer). The byte-identity oracle gate in Task 2 — not the unit tests — is what proves numerical correctness; the RED gate was satisfied by the nroots-guard and spinor-reject failures.

## Known Stubs
None. The kernel emits real gradient values (proven byte-identical to vendor); no placeholder/empty-value paths were introduced.

## Threat Flags
None — no new network/auth/file-access surface. The only new trust boundary (high-l quartet → rys dispatch; gout transpose → component layout) is covered by the plan's threat register (T-21-05-01/02/03/04) and mitigated by the nroots guard + the vendor byte-identity layout gate.

## Follow-up Notes
- **Flip `oracle_covered=true` for `int2e_ip1`** (int2e_ip1_cart / int2e_ip1_sph rows in `crates/cintx-ops/generated/compiled_manifest.lock.json`, currently `false`): now that the vendor-gated byte-identity parity is green, the Phase-15 `xtask oracle-covered-update` (or manifest stamp) should mark these covered. Deliberately NOT edited in this plan — manifest oracle-coverage stamping is owned by the Phase-15 xtask gate, and the plan's `files_modified` does not include the manifest lock. The spinor row stays `oracle_covered=false` (int2e_ip1_spinor → UnsupportedApi per R5).

## Next Phase Readiness
- 21-06 (`int3c2e_ip1`) can now reuse the same `pub(crate)` `gout_ip1`/`F12Shape` and the established "build_2e_shape(li+1,…) + fill_g_tensor + gout_ip1 + transpose to component-leading" pattern (the 3c2e kl-mapping uses `build_2e_shape(li+1, lj, 0, lk)` per its file header).
- No blockers.

---
*Phase: 21-coulomb-gradient-intors*
*Completed: 2026-05-26*
