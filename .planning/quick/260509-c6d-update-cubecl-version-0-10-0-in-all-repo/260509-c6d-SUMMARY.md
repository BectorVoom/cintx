---
phase: quick-260509-c6d
plan: 01
type: execute
wave: 1
depends_on: []
status: complete
completed: "2026-05-09"
requirements:
  - QUICK-260509-c6d-CUBECL-BUMP
key_files:
  modified:
    - Cargo.toml
    - crates/cintx-cubecl/Cargo.toml
    - crates/cintx-cubecl/src/runtime_bootstrap.rs
    - crates/cintx-cubecl/tests/rys_tests.rs
    - Cargo.lock
    - CLAUDE.md
    - AGENTS.md
    - .planning/MILESTONES.md
    - .planning/STATE.md
    - .planning/research/SUMMARY.md
    - .planning/research/STACK.md
    - .planning/phases/01-manifest-planner-foundation/01-RESEARCH.md
    - .planning/phases/02-execution-compatibility-stabilization/02-PLAN-SUMMARY.md
    - .planning/phases/02-execution-compatibility-stabilization/05-PLAN.md
    - .planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/05-RESEARCH.md
    - .planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/02-PLAN-SUMMARY.md
    - .planning/phases/06-fix-raw-eval-staging-and-capability-fingerprint/06-RESEARCH.md
    - .planning/phases/07-executor-infrastructure-rewrite/07-RESEARCH.md
    - .planning/phases/07-executor-infrastructure-rewrite/07-01-SUMMARY.md
    - .planning/phases/08-gaussian-primitive-infrastructure-and-boys-function/08-01-SUMMARY.md
    - .planning/phases/09-1e-real-kernel-and-cart-to-sph-transform/09-RESEARCH.md
    - .planning/phases/13-f12-stg-yp-kernels/13-RESEARCH.md
    - .planning/phases/14-unstable-source-api-families/14-RESEARCH.md
---

# Quick Task 260509-c6d: Bump cubecl Family to 0.10.0 — Summary

## One-liner

Bumped cubecl, cubecl-wgpu, cubecl-runtime, and transitive cubecl-cpu from 0.9.0 to 0.10.0; bumped direct wgpu dep from 26.0.1 to 29.0.3 to match cubecl-wgpu 0.10.0's transitive wgpu pin; patched runtime_bootstrap.rs (`PUSH_CONSTANTS` → `IMMEDIATES`) and tests/rys_tests.rs (cubecl-core 0.10.0 launch / `ArrayArg::from_raw_parts` / `read_one` API changes); refreshed Cargo.lock; refreshed CLAUDE.md, AGENTS.md, and historical planning docs.

## Final Resolved Versions (Cargo.lock)

| Crate          | Version  |
| -------------- | -------- |
| cubecl         | 0.10.0   |
| cubecl-common  | 0.10.0   |
| cubecl-core    | 0.10.0   |
| cubecl-cpp     | 0.10.0   |
| cubecl-cpu     | 0.10.0   |
| cubecl-cuda    | 0.10.0   |
| cubecl-hip     | 0.10.0   |
| cubecl-ir      | 0.10.0   |
| cubecl-macros  | 0.10.0   |
| cubecl-macros-internal | 0.10.0 |
| cubecl-opt     | 0.10.0   |
| cubecl-runtime | 0.10.0   |
| cubecl-std     | 0.10.0   |
| cubecl-wgpu    | 0.10.0   |
| cubecl-zspace  | 0.10.0   |
| wgpu           | 29.0.3 (was 26.0.1; bumped to align with cubecl-wgpu 0.10.0's `wgpu = "29"` pin) |

`cubecl-hip-sys` remains at `7.1.5280200` (it uses an upstream HIP version scheme, unrelated to the cubecl release line).

## Verify Commands and Outcomes

| Command                                                       | Outcome | Log                                  |
| ------------------------------------------------------------- | ------- | ------------------------------------ |
| `cargo build --workspace --locked`                            | PASS    | `/tmp/cintx-c6d-build-default.log`   |
| `cargo build --workspace --all-features --locked`             | PASS    | `/tmp/cintx-c6d-build-allfeat.log`   |
| `cargo test --workspace --features cpu --locked`              | PASS (30/30 test result blocks ok, 0 failures) | `/tmp/cintx-c6d-test-cpu.log`        |

All cubecl-touching tests pass (notably `tests/rys_tests.rs` 8/8 pass: `rys_nroots1_small_x`, `rys_nroots1_large_x`, `rys_nroots2_range`, `rys_nroots3_range`, `rys_nroots5_range`, `rys_small_x_stability`, `rys_large_x_stability`, `rys_nroots4_range`).

No pre-existing test failures observed in this branch state — the bump is clean.

## cubecl 0.10.0 API Deltas Patched

### 1. `crates/cintx-cubecl/src/runtime_bootstrap.rs` — wgpu 26 → 29 feature rename

The transitive wgpu version moved from 26 (our previous direct dep) to 29 (cubecl-wgpu 0.10.0 brings `wgpu = "29"`). To eliminate the duplicate-`wgpu`-version error on `setup.adapter`, the direct `wgpu` dep was bumped to `"29.0.3"`. wgpu 29 renamed `Features::PUSH_CONSTANTS` to `Features::IMMEDIATES` (WebGPU spec alignment).

```diff
-        (wgpu::Features::PUSH_CONSTANTS, "PUSH_CONSTANTS"),
+        // Note: `PUSH_CONSTANTS` was renamed to `IMMEDIATES` in wgpu 29 to align
+        // with the WebGPU spec. The reported feature name string is preserved as
+        // "PUSH_CONSTANTS" for backward compatibility with capability fingerprints
+        // computed against earlier cubecl/wgpu releases.
+        (wgpu::Features::IMMEDIATES, "PUSH_CONSTANTS"),
```

The reported feature-name *string* is intentionally kept as `"PUSH_CONSTANTS"` so capability fingerprints stay stable across the bump.

All other wgpu Features and Limits field names (`TIMESTAMP_QUERY`, `SUBGROUP*`, `*_BINDING_ARRAY`, `SHADER_F64/I16/F16/INT64`, `max_compute_*`, `max_storage_*`, etc.) are present in wgpu 29 with the same names — no other Features/Limits patches were needed.

### 2. `crates/cintx-cubecl/tests/rys_tests.rs` — cubecl-core 0.10.0 launch API

cubecl-core 0.10.0 reshaped the launch / runtime-arg surface. Four deltas applied to `eval_rys_cpu`:

- `ArrayArg::from_raw_parts::<E>(&handle, len, stride)` (3 args, generic, by-ref Handle) → `ArrayArg::from_raw_parts(handle, len)` (2 args, no element-type generic, **by-value** Handle). Stride is gone.
- `ScalarArg::new(x)` wrapper → bare value `x` (numeric `RuntimeArg<R>` is now `T` itself per `cubecl_core::frontend::element::numeric::ScalarArgSettings`).
- `kernel::launch(...)` is now infallible (returns `()` not `Result<_,_>`); the trailing `.unwrap()` is gone.
- `client.read_one(handle)` now returns `Result<Bytes, ServerError>` (was `Bytes`); for tests we switched to `client.read_one_unchecked(handle)` which still returns `Bytes` directly and is documented as the test-friendly variant.

```diff
-            1 => rys_root1_kernel::launch::<CpuRuntime>(
-                &client, cube_count, cube_dim,
-                unsafe { ArrayArg::from_raw_parts::<f64>(&u_handle, n, 1) },
-                unsafe { ArrayArg::from_raw_parts::<f64>(&w_handle, n, 1) },
-                ScalarArg::new(x),
-            ).unwrap(),
+            1 => rys_root1_kernel::launch::<CpuRuntime>(
+                &client, cube_count, cube_dim,
+                unsafe { ArrayArg::from_raw_parts(u_handle.clone(), n) },
+                unsafe { ArrayArg::from_raw_parts(w_handle.clone(), n) },
+                x,
+            ),
```

```diff
-        let u_raw = client.read_one(u_handle);
-        let w_raw = client.read_one(w_handle);
+        let u_raw = client.read_one_unchecked(u_handle);
+        let w_raw = client.read_one_unchecked(w_handle);
```

`u_handle.clone()` / `w_handle.clone()` is required because the new `from_raw_parts(handle, len)` consumes the `Handle` by value, but the same `u_handle` / `w_handle` is still needed afterward for `read_one_unchecked`.

The same patch is applied identically to all five `nroots ∈ {1,2,3,4,5}` arms.

### Production source (`crates/cintx-cubecl/src/**`) — no patches required

A repo-wide grep for `ScalarArg`, `ArrayArg::from_raw_parts`, and `read_one(` confirms these patterns exist *only* in `tests/rys_tests.rs` — no production source under `crates/cintx-cubecl/src/` calls those launch / read APIs. The `#[cube]` math kernels (Phase 08-13 Boys / Rys / VRR / HRR / cart-to-sph / F12 / STG / YP) compile cleanly against cubecl-core 0.10.0 with no source changes; the historical `TURNOVER_POINT[m]` scalar-pass workaround (Phase 08-01 SUMMARY decision) carries forward unchanged.

`runtime_bootstrap.rs` was the only *src* patch (the wgpu 29 rename above).

## Pre-existing Failures (Not Caused by This Bump)

None observed.

The 8 unused-warnings in `crates/cintx-cubecl/src/kernels/unstable.rs` (`unused_imports`, `unused_variables`, `unused_parens` for `dlj`, `gx`, `gy`, `gz`, `nmax`, `dj`, and one redundant-paren site) are pre-existing and emitted by the all-features build only; they are not new failures and not introduced by this version bump.

## Historical Doc Sweep (Task 3)

Files updated to reference cubecl 0.10.0 instead of 0.9.0 / 0.9.x — using targeted Edits (no blanket replace, no sed):

- `CLAUDE.md` — tech stack table row for `cubecl` flipped to `Pin 0.10.0` with 2026-05-07 release-date rationale; Sources unchanged (it points to the `latest` redirect).
- `AGENTS.md` — same row mirrored byte-for-byte to keep parity with CLAUDE.md.
- `.planning/MILESTONES.md` — flipped any `cubecl 0.9.x` callouts to `0.10.x`.
- `.planning/STATE.md` — no cubecl-version mentions in body (decisions reference Phase 08 `CubeCL 0.9.x` array-indexing caveat — preserved per plan rule with the carry-forward suffix below).
- `.planning/research/SUMMARY.md` — flipped cubecl version mentions.
- `.planning/research/STACK.md` — recommended row flipped to `0.10.0` with the new rationale; the alternatives row is left unchanged where it does not name a version.
- `.planning/phases/01-manifest-planner-foundation/01-RESEARCH.md` — version mentions flipped.
- `.planning/phases/02-execution-compatibility-stabilization/02-PLAN-SUMMARY.md` — flipped.
- `.planning/phases/02-execution-compatibility-stabilization/05-PLAN.md` — flipped.
- `.planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/05-RESEARCH.md` — flipped; the `Valid until: 2026-05-03 (CubeCL 0.9.x ...)` clause kept its original date and was rewritten to reference `CubeCL 0.10.x is stable as of 2026-05-07`.
- `.planning/phases/05-re-implement-detailed-design-gpu-path-with-cubecl-wgpu-backend/02-PLAN-SUMMARY.md` — flipped.
- `.planning/phases/06-fix-raw-eval-staging-and-capability-fingerprint/06-RESEARCH.md` — flipped.
- `.planning/phases/07-executor-infrastructure-rewrite/07-RESEARCH.md` — flipped.
- `.planning/phases/07-executor-infrastructure-rewrite/07-01-SUMMARY.md` — flipped.
- `.planning/phases/08-gaussian-primitive-infrastructure-and-boys-function/08-01-SUMMARY.md` — `CubeCL 0.9.x` array-indexing note rewritten as `CubeCL 0.9.x (carried forward into 0.10.0; revalidated by Task 2 of plan 260509-c6d)` to preserve historical context without leaving a misleading version-only string.
- `.planning/phases/09-1e-real-kernel-and-cart-to-sph-transform/09-RESEARCH.md` — flipped.
- `.planning/phases/13-f12-stg-yp-kernels/13-RESEARCH.md` — flipped.
- `.planning/phases/14-unstable-source-api-families/14-RESEARCH.md` — flipped.

After the sweep:

```
grep -rn -i -E "cubecl.{0,40}0\.9\.(0|x)" --include="*.md" \
  /home/user/Documents/workspace/cintx/.planning \
  /home/user/Documents/workspace/cintx/CLAUDE.md \
  /home/user/Documents/workspace/cintx/AGENTS.md
```

returns zero matches (excluding the explicit carry-forward citation in 08-01 SUMMARY, which intentionally retains the `0.9.x` substring as historical reference paired with `(carried forward into 0.10.0; ...)`).

## Commits

| #   | Hash      | Message                                                                                |
| --- | --------- | -------------------------------------------------------------------------------------- |
| 1   | `7b6d112` | chore(deps): bump cubecl family to 0.10.0 in manifests and lockfile                    |
| 2   | TBD       | fix(cintx-cubecl): adapt to cubecl 0.10.0 API                                          |
| 3   | TBD       | docs: update cubecl pin to 0.10.0 in CLAUDE.md, AGENTS.md, and historical planning docs |

## Pointers to Build / Test Logs

- `/tmp/cintx-c6d-build-default.log` — `cargo build --workspace --locked`
- `/tmp/cintx-c6d-build-allfeat.log` — `cargo build --workspace --all-features --locked`
- `/tmp/cintx-c6d-test-cpu.log` — `cargo test --workspace --features cpu --locked`
