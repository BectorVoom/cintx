# Quick Task 260531-aw1: Port host eigh + rys_wheeler to CubeCL `#[cube]` — Research

**Researched:** 2026-05-31
**Domain:** CubeCL `#[cube]` kernel authoring; symmetric-tridiagonal eigensolver + Wheeler/Jacobi Rys engine
**Confidence:** HIGH (verified against in-tree `#[cube]` kernels, the CubeCL manuals, and the actual call sites)

## Summary

The two target files are the host "tail" of the Rys quadrature engine for `nroots 6..=12`: `rys_wheeler.rs` builds a tridiagonal moment matrix (Flocke modified moments → Wheeler recursion) and `eigh.rs` diagonalizes it (QL + Sturm/Rayleigh refinement) to produce roots/weights. They are reached only by the **host gradient/Hessian prep path** of a handful of families (deriv34, center_2c2e grad, center_3c1e, the hess* families) — `rys_roots_host` is called on the CPU to fill small `g`-tensor `Vec`s **before** the device kernel launches (see `center_2c2e.rs:526`, `deriv34.rs:365`). They are NOT currently inside any `#[cube(launch)]` kernel.

Force-porting them to `#[cube]` is mechanically feasible — CubeCL forbids `break`/`continue`/early-`return`/`if`-as-expression and data-dependent loop *bounds*, but the in-tree `bessel.rs` kernel already demonstrates every workaround needed: bounded `while` loops with a comptime/`const` cap and a "set the counter past the cap" break-emulation (`bessel.rs:322`), device-local scratch via `Array`, and f64 on the CPU backend. The real obstacle is **not control flow — it is numerics**: `rys_wheeler.rs` deliberately runs Dekker/Knuth **double-double (≈106-bit) arithmetic** (`Dd` struct, `two_prod` via `mul_add`) to emulate x86-64 80-bit long double, because plain f64 diverges from the libcint vendor by ~1e-9 at the largest Rys root (the `r/(1-r)` transform is ill-conditioned as r→1). That double-double layer, plus `eigh`'s compensated-summation Rayleigh + Sturm-bisection refinement, is the entire reason the 29/29 byte-identical parity holds. Reproducing it inside `#[cube]` requires hand-rolling double-double as a pair of f64 fields threaded through the kernel — doable, but it is the load-bearing risk.

**Primary recommendation:** Port in three tasks (eigh first, then the f64 wheeler path 6/7, then the double-double path 8..12), keeping every kernel **generic-`F`-output / f64-internal** and running them under the **CPU CubeCL backend** the parity gate already uses. Expect a real numerics fight on the double-double path; budget for it explicitly. Do NOT attempt to widen device `nroots` past the current host-only routing — keep these as host-prep kernels launched via the existing `run_*::<R>` CPU-client idiom, feeding the same `g` tensors.

## Feasibility Verdict (blunt)

**Control flow: fully portable.** Every `break`/`continue`/early-`return`/data-dependent-`while` in both files maps to the bounded-loop + converged-mask + "advance counter past cap" idiom already proven in `bessel.rs`. The trip counts are all bounded by `MXRYSROOTS`/`n ≤ 12` constants → comptime-bounded loops are legal.

**Numerics: portable but high-risk, and byte-identical parity is NOT guaranteed for nroots 8..12.** The double-double emulation (`Dd`) is the single biggest risk. CubeCL has no native f128/long-double and no `Dd` type; you must re-express `two_sum`/`two_prod`/`dd_mul`/`dd_div`/`dd_sqrt` as plain-f64 arithmetic inside `#[cube]`. `two_prod` relies on **FMA (`a.mul_add(b, -p)`)** — if the CubeCL CPU backend does not lower `mul_add` to a true fused multiply-add (no intermediate rounding), the double-double error terms are wrong and nroots 8..12 parity will drift past atol=1e-12. **This must be verified empirically before committing** (see Open Question Q1). If FMA is not faithfully fused on the CPU backend, byte-identical parity for nroots 8..12 is **not achievable** in `#[cube]` without a software FMA, and the plan must say so rather than paper over it.

For nroots 6/7 (pure f64 Jacobi/Schmidt, no `Dd`) and the eigh QL solver (pure f64), byte-identical parity is realistically preservable.

## Architectural Responsibility Map

| Capability | Current Tier | Target Tier | Rationale |
|------------|-------------|-------------|-----------|
| Rys roots/weights nroots 1..5 | Device `#[cube]` (`rys_root1..5`) | unchanged | Already on-device polynomial fits |
| Rys roots/weights nroots 6..12 (wheeler) | **Host** `Vec`/`rys_roots_host` | Device `#[cube]` (host-launched CPU client) | Task goal: eliminate host calculation |
| Tridiagonal eigensolve (`cint_diagonalize`) | **Host** `Vec` | Device `#[cube]` | Called by wheeler; same move |
| `g`-tensor assembly + obara-saika recurrence | Host `Vec` (e.g. `run_2c2e_grad`) | OUT OF SCOPE | Not in the two target files; leave as-is |
| Per-quartet planning/marshaling | Host | unchanged | CLAUDE.md keeps host for planning/marshaling |

**Key boundary fact:** `rys_roots_host` is invoked from host prep code that then writes into `g: Vec<f64>` consumed by the device kernel. After porting, the ported kernel becomes a small **host-launched `#[cube]` kernel over the CPU CubeCL backend** whose output (roots/weights arrays) feeds the same `g` build. You do not need to fuse it into the big device kernel; you need it to stop being plain Rust `Vec` math and become a `#[cube]` kernel launched via the existing `run_*::<R>` idiom.

## CubeCL `#[cube]` Control-Flow Constraints (verified)

| Question | Answer | Source |
|----------|--------|--------|
| `break` allowed? | **No** in statement-form early-exit semantics; the in-tree idiom is to set the loop counter past its cap to terminate. `bessel.rs:322` comment: *"CubeCL statement-form: signal break via setting k past cap."* (Note `bessel.rs:240` uses a literal `break` inside a `#[cube]` while-loop that the macro accepts — so a `break` at the *tail* of a bounded `while` compiles, but relying on it for mid-body deflation/restart is fragile; prefer the counter-past-cap + `converged` flag pattern.) | `bessel.rs:229-244, 316-324`; MEMORY `reference_cubecl_authoring_manuals` |
| `continue` allowed? | **No.** Restructure as a guarded `if` that skips the body (set a skip flag, wrap remaining body in `if !skip`). eigh's `continue` (underflow restart, `eigh.rs:330`) and wheeler's `continue`s (`rys_wheeler.rs:594, 769, 972`) all need this rewrite. | MEMORY pitfalls; manuals |
| `while` with data-dependent trip count? | **Allowed**, but must have a **fixed/comptime upper bound**; loop runs to the cap and a `converged`/mask flag no-ops the rest. `bessel.rs` runs `while k <= MODERATE_X_MAX_TERMS` (const cap) and `while k <= 200u32`. | `bessel.rs:229, 316` |
| `if` as expression (`let x = if c {a} else {b}`)? | **Forbidden.** Initialize `let mut x = default;` then assign inside `if`. | `Cubecl_conditionals.md:22-41` |
| Legal index types? | **u32 / i32 only** for array indexing and loop counters; cast `as usize` only outside kernel. eigh/wheeler use `usize` and `isize` (e.g. `while i >= 0` with `isize`, `rys_wheeler.rs:234, 376`) — must become u32/i32 with sentinel handling. | MEMORY `reference_cubecl_authoring_manuals`; `Cubecl_basic_operations.md` |
| `f64` supported on the test backend? | **Yes** on the CPU CubeCL backend (the parity gate's `--features cpu`). In-tree `bessel.rs` does all math in f64 inside `#[cube]`; `Cubecl_basic_operations.md:151` confirms launching with `f64`. Keep the **f64-internal / generic-`F`-output** template from `rys.rs` (`F::cast_from(literal_f64)`, `F::new(...)`). | `bessel.rs`; `rys.rs` `clenshaw_d1`/`rys_root1`; `Cubecl_basic_operations.md:151` |
| Device-local scratch `Array` (read+write, e.g. n*n eigenvectors + tridiagonal)? | **Yes.** `#[cube]` fns take `&mut Array<F>` and read/write by `u32` index (`rys.rs` `rys_root2` writes `u[0usize]..u[1usize]`; bessel writes `out[l as usize]`). eigh's `z[i*n+j]` (≤144 f64) and wheeler's `s0/sm/sk` (≤24) fit as fixed-size `Array`s sized at the comptime `n` cap. Prefer passing scratch as `&mut Array<F>` arguments rather than allocating inside, mirroring how `u`/`w` are passed in. | `rys.rs:62, 164`; `bessel.rs` |
| Const-array indexing by runtime index? | **Forbidden** — `rys.rs:24` note: *"`#[cube]` functions cannot index const arrays by runtime index."* The big Jacobi/Laguerre coefficient tables (`roots_jacobi_data`, `LJACOBI_ALPHA/BETA`) must be passed in as `&Array<F>` inputs (copied to device once), not indexed as Rust `const` slices. This is a non-trivial plumbing change for wheeler. | `rys.rs:21-27` |

**Comptime specialization:** `cubecl_macro_fanout_manual.md:171-191` shows the exact pattern for per-nroots dispatch: one generic kernel with `#[comptime] nroots: u32` and `if comptime!(nroots == 6) {...}`. This is the right tool to replace wheeler's runtime `match nroots` and per-nroots breakpoint constants — it mirrors how `rys.rs` already comptime-specializes nroots 1..5.

## eigh.rs Port Strategy

Every non-CubeCL-legal construct and its rewrite:

| Location | Construct | Rewrite |
|----------|-----------|---------|
| `tqli_impl:266` | `loop { ... }` (unbounded outer per-eigenvalue) | `for _sweep in 0..MAX_ITER` (const-bounded; `MAX_ITER=200` already exists) with a `converged` flag |
| `tqli_impl:269-275` | inner `while m < n-1 { ... break; }` (find negligible offdiag) | bounded `while m < N_CAP` + set `m = N_CAP` to stop; track found index in a var |
| `tqli_impl:277-279` | `if m == l { break }` (early-exit converged) | set `converged = true`, let the bounded outer loop no-op remaining sweeps via `if !converged` |
| `tqli_impl:281` | `if iter >= MAX_ITER { return 1 }` | drop early-return; carry an `info` flag out via an output `Array` element |
| `tqli_impl:299-326` | `while i > l { i -= 1; ... if r==0 { ...break } }` (downward plane-rotation sweep with underflow `break`+outer `continue`) | bounded `while`, underflow handled by `underflow` flag that skips the rotation tail and forces a re-search next sweep (no `continue`) |
| `tqli_impl:321-325` | eigenvector rotation `z[k*n+i+1] = s*z[k*n+i]+c*tmp` for `k in 0..n` | fits the kernel memory model: `z` is a `&mut Array<F>` of ≤144 elements, indexed by u32 `k*n+i`. **This is the only nontrivial memory question and it is fine** — it is exactly the bounded read-modify-write `bessel.rs`/`rys.rs` already do, just on a 2D-flattened buffer. |
| `cint_diagonalize:419-433` | `Vec` allocations (`d_orig`, `e_orig`, `z`, `idx`) | replace with caller-passed `&mut Array<F>` scratch sized at comptime `MXRYSROOTS` (13) / `MXRYSROOTS²` (169) |
| `cint_diagonalize:443-444` | `idx.sort_by(...)` (ascending eigenvalue sort) | **no `sort_by` in `#[cube]`** — replace with a bounded selection/insertion sort over `n ≤ 12` (≤ 144 compares, trivially unrolled) writing into the output ordering |
| `refine_eigenvalues_bisection:151-186` | nested bounded `for _ in 0..60` / `0..200` with `break` | already bounded; replace `break` with a `done` flag; `std::mem::swap`/`partial_cmp`/`.to_vec()` (line 165, 444, 141) must go — use explicit temps and scratch arrays |
| `dlaneg`, `dlarrk` | `f64::MIN_POSITIVE`, `f64::EPSILON` | use `F::cast_from(f64::EPSILON)` etc.; these are comptime constants |

**Verdict for eigh:** Pure f64, all loops bounded by `n ≤ 12` / `MAX_ITER=200`. No double-double. Byte-identical parity realistically preservable. The compensated-summation (`comp_add`, Kahan) and Sturm-bisection refinement port directly (they are just f64 arithmetic). Main tedium: removing `Vec`/`sort_by`/`mem::swap` and the `usize`→`u32` index conversions.

## rys_wheeler.rs Port Strategy

**Per-nroots dispatch → comptime.** Replace `rys_roots_host_wheeler`'s `match nroots` (`rys_wheeler.rs:1119-1121`) and `segment_solve`'s breakpoint with `#[comptime] nroots` + `if comptime!(nroots <= 7) {...} else if comptime!(nroots == 8) {...}` etc., and a comptime breakpoint constant per nroots. The x-breakpoint branch (jacobi vs schmidt vs laguerre) stays a **runtime** `if x <= breakpoint` (data-dependent on x) — that is a legal runtime `if`.

**The double-double layer is the crux.** `Dd { hi, lo }` and its ops (`two_sum`, `two_prod`, `dd_add/sub/mul/div/sqrt`, `rys_wheeler.rs:71-150, 713-730`) have no CubeCL equivalent. Options:
1. **Hand-roll double-double in f64 pairs inside `#[cube]`** (recommended): represent each `Dd` as two `F`/f64 locals `(hi, lo)` threaded explicitly; re-express every `dd_*` as the corresponding f64-pair algorithm. `two_prod` needs FMA → use `F`'s fused-multiply-add if the prelude exposes it, else a Dekker split. **This is the load-bearing port and the biggest risk.**
2. Drop to plain f64 for nroots 8..12 — **rejected**: the file's own header (`rys_wheeler.rs:59-68`) documents that plain f64 diverges ~1e-9 and blows the 1e-12 gate. This would fail parity.

**Control-flow rewrites (same idioms as eigh):**
- `flocke_jacobi_moments` / `lflocke_*_dd`: downward `while i >= n` / `while i >= 0` with `isize` (`:226, 234, 368, 376`) → bounded `for` with `i` as i32 and explicit `>= 0` guard.
- `rys_wheeler_partial:300-310` / `lrys_wheeler_partial_dd:462-470`: `break` on singular `b` → `converged`/`stop` flag.
- `r_dsmit:512-542`, `r_lsmit_dd`: nested `for` with `return 0/1/j` mid-loop → flag + bounded completion.
- `cint_polynomial_roots` → `hessenberg_qr` (`:904-959`): QR iteration `while k+1 < n1 { ... break }` + `R_dnode` Newton/bisection (`:961-1034`, `while (x1-x0).abs() > x1*accrt` with `break`/`return`/`continue`) → all bounded-loop + flag rewrites. This is the densest cluster of illegal control flow in the file.
- All the coefficient tables (`data::LJACOBI_ALPHA/BETA`, `JACOBI_COEF`, `roots_jacobi_data`) must move from `const` slices to device input `Array`s (no runtime const-array indexing — `rys.rs:24`).
- `Vec` everywhere → caller-passed fixed-size scratch `Array`s sized at comptime `MXRYSROOTS=32` (or tightened to the real `n*2 ≤ 24`).

**Long-double / c99_sqrtl:** `c99_sqrtl` (one Babylonian refinement over f64 `sqrt`, `:53-56`) is trivial f64 and ports unchanged. The question is whether the *double-double* intermediates survive — that hinges on FMA fidelity (Q1), not on `c99_sqrtl`.

## Parity Preservation

**The gate (MEMORY `reference_oracle_vendor_parity_invocation`):** real parity runs only with **`--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`** together. The ported kernels run under the **CPU CubeCL backend** — the same backend `bessel.rs`/`rys_root1..5` already run on under that gate — so they execute on the parity-gate path. (GPU/rocm backend parity is a *separate* question and NOT what locks the 29/29; do not chase rocm here.)

**Tests that lock nroots 6..=12 (the 29/29 FND-02 parity), re-run these after the port:**
- `crates/cintx-oracle/tests/hess2e_parity.rs` — 4-center Hessian, `hess_nroots` reaches 6..12.
- `crates/cintx-oracle/tests/hess1e_ipip_parity.rs` — 1e rank-9/27 Hessian families.
- `crates/cintx-oracle/tests/hess_multicenter_ipip_parity.rs` — 2c/3c Hessian (`hess_nroots_2c/3c`).
- `crates/cintx-oracle/tests/deriv34_parity.rs` — deriv3/deriv4 (`int1e_ipipip*`, `ipipiprinv*`), the `rys_roots_host` caller at `deriv34.rs:365`.
- `crates/cintx-oracle/tests/center_2c2e_parity.rs` and `int2c2e_ip_parity.rs` — the `run_2c2e_grad` host path (`center_2c2e.rs:526`).
- `crates/cintx-oracle/tests/center_3c1e_parity.rs` — `center_3c1e.rs` host path.

**Fast in-crate regression guards (run on every commit, no vendor build):**
- `rys::tests_rys_host::rys_host_nroots_ge6` (`rys.rs:3470`) — 6-element finite/non-neg.
- `rys_wheeler::tests::wheeler_all_nroots_finite` and `wheeler_returns_finite_nonneg_nroots6` (`rys_wheeler.rs:1264, 1275`).
- `eigh::tests::eigh_mrrr_tridiag*` (3×3, 6×6, 12×12, diagonal — all assert atol 1e-11..1e-12 against numpy).

**Gap:** there is currently **no in-crate unit test that asserts the wheeler 6..12 roots/weights match libcint reference values at atol=1e-12** (the existing in-crate tests only check finiteness/non-negativity; the tight 1e-12 lock lives only in the vendor-gated family parity tests). **Wave-0 recommendation:** add a `rys_roots_host(nroots, x)` reference-table test for nroots 6..12 across the intermediate-x grid (mirroring the `rys_root3/4/5_host_intermediate_x_matches_libcint` tables at `rys.rs:3387-3460`) BEFORE porting, so the double-double regression is caught fast without a full vendor build.

## Recommended Task Decomposition

1. **Task 1 — Wave-0 reference test + port `eigh.rs` to `#[cube]`.** Add the missing nroots-6..12 reference-value unit test first (capture libcint values via the existing vendor harness, hardcode as a table like `rys.rs:3389`). Then port `cint_diagonalize`/`tqli_impl`/refinement to a `#[cube]` kernel (pure f64, bounded loops, `&mut Array` scratch, selection-sort for ordering). Gate: in-crate `eigh_mrrr_tridiag*` pass + new reference test still green via the (still-host) wheeler.
2. **Task 2 — Port the f64 wheeler path (nroots 6, 7) to `#[cube]`.** `rys_jacobi` + `rys_schmidt` + the f64 Wheeler recursion + moment tables as input `Array`s, calling the Task-1 device eigh. No double-double. Gate: hess/deriv34 family parity for nroots 6/7 cases byte-identical.
3. **Task 3 — Port the double-double path (nroots 8..12) to `#[cube]`.** Re-express `Dd` as f64 pairs; `lrys_jacobi`/`lrys_schmidt`/`lrys_laguerre` + the QR polynomial-root solver. **First sub-step: empirically verify FMA fidelity on the CPU backend (Q1) — if it fails, escalate before sinking effort.** Gate: full 29/29 vendor parity (`hess2e`, `hess1e_ipip`, `hess_multicenter`, `deriv34`).

If Task 3's FMA verification fails, the honest outcome is: **nroots 8..12 cannot reach byte-identical parity in pure `#[cube]` f64 without a software-FMA double-double**, and the plan should either (a) implement a Dekker-split software FMA inside `#[cube]`, or (b) keep nroots 8..12 host-side and report the override as partially infeasible. Say this in the plan; do not silently degrade tolerance.

## Common Pitfalls

1. **Assuming the eigh `z[i*n+j]` matrix can't live on-device.** It can — it is ≤144 f64 passed as `&mut Array<F>`. The memory model is fine; the blocker is control flow + sort, not memory.
2. **Forgetting const-array indexing is illegal.** The Jacobi/Laguerre coefficient tables are the single biggest plumbing change — they must become device input arrays, copied once.
3. **Trusting `mul_add` to be a true FMA.** Double-double correctness depends on it. Verify on the CPU backend before relying on it (Q1).
4. **Using `usize`/`isize` indices or `Vec`/`sort_by`/`mem::swap` inside `#[cube]`.** None compile; convert to u32/i32 + fixed scratch arrays + manual sort.
5. **Chasing rocm/GPU parity.** The 29/29 lock is the CPU-backend vendor gate; port and validate there. GPU is out of scope for this task.

## Project Constraints (from CLAUDE.md)

- CubeCL is the primary compute backend; host CPU is limited to planning/validation/marshaling — this task *removes* host calculation, aligning with the constraint (the FND-02 host-side exception is the thing being overridden).
- Public library errors use `thiserror` v2; oracle/xtask use `anyhow` — unchanged here.
- Compatibility target: libcint 6.1.3, oracle-comparison byte-identity at atol=1e-12 — the parity bar this port must not regress.
- `cubecl` pinned `0.10.0`, Rust `1.94.0`, `cargo --locked` — port against the pinned API.

## Open Questions

1. **(BLOCKER for nroots 8..12) Does the CubeCL 0.10.0 CPU backend lower `mul_add`/FMA without intermediate rounding?**
   - Known: double-double `two_prod` (`rys_wheeler.rs:104`) uses `a.mul_add(b, -p)` and requires a true fused op.
   - Unclear: whether the CPU backend's IR preserves FMA semantics or rounds the product first.
   - Recommendation: write a 5-line `#[cube]` probe that computes `two_prod(a,b)` for a known pair and asserts the error term matches the host f64 result bit-for-bit. Run under `--features cpu` FIRST in Task 3. If it fails, implement a Dekker-split software product (no FMA dependency) inside the kernel.
2. **Is `break` at the tail of a bounded `while` reliably accepted by the `#[cube]` macro, or only the counter-past-cap idiom?** `bessel.rs:240` uses a literal `break`; `bessel.rs:322` uses counter-past-cap. Prefer counter-past-cap for the deflation/restart cases (eigh underflow, wheeler singular-b) where the break is mid-body, and reserve literal `break` only for convergence at loop tail. Confirm by compiling an early eigh port.
3. **Comptime `n` vs runtime `n`:** eigh/wheeler size scratch by `n` (= nroots). Cleanest is one kernel per nroots via `#[comptime] nroots`, fully unrolling the bounded loops. Confirm the macro accepts comptime-sized `Array` scratch or whether scratch must be sized at the static `MXRYSROOTS` cap and sub-indexed.

## Sources

### Primary (HIGH confidence — in-tree, verified this session)
- `crates/cintx-cubecl/src/math/bessel.rs:123-340` — the canonical in-tree `#[cube]` bounded-`while` + break-emulation + f64-internal idiom.
- `crates/cintx-cubecl/src/math/rys.rs:21-153, 3235-3567` — f64-internal/generic-`F`-output template, const-array-indexing prohibition, host dispatcher, in-crate tests.
- `crates/cintx-cubecl/src/math/eigh.rs` (full) — QL solver, every break/continue/return/sort enumerated.
- `crates/cintx-cubecl/src/math/rys_wheeler.rs:1-150, 280-1130` — double-double layer, dispatch, control flow.
- `crates/cintx-cubecl/src/kernels/center_2c2e.rs:140, 505-549, 1494` and `deriv34.rs:355-375` — proof that `rys_roots_host` is host-prep feeding the `g` tensor, not inside the device kernel.
- `docs/manual/Cubecl/Cubecl_conditionals.md` (if-as-statement rule), `Cubecl_basic_operations.md` (f64, `F::cast_from`, no raw literals), `cubecl_macro_fanout_manual.md:171-191, 312-440` (comptime per-variant specialization).
- MEMORY: `reference_cubecl_authoring_manuals`, `reference_oracle_vendor_parity_invocation`, `project_fnd06_chunk_staging_is_full_block` (FND-02 / 29/29 context).

### Tests that lock parity (HIGH)
- `crates/cintx-oracle/tests/{hess2e,hess1e_ipip,hess_multicenter_ipip,deriv34,center_2c2e,int2c2e_ip,center_3c1e}_parity.rs`

## Metadata

**Confidence breakdown:**
- Control-flow portability: HIGH — every construct mapped to an in-tree precedent.
- Memory model (eigh `z`, wheeler scratch): HIGH — matches existing `&mut Array` usage.
- Double-double / FMA numerics: MEDIUM — portability is clear, byte-identity hinges on the unverified FMA question (Q1).
- Parity test inventory: HIGH — call sites and tests grepped directly.

**Research date:** 2026-05-31
**Valid until:** ~30 days (stable; cubecl 0.10.0 pinned)
