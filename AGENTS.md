
## Project

**cintx**

cintx is a public Rust library that redesigns and reimplements libcint with result compatibility as the primary goal. It provides a Rust-native safe API, a raw compatibility API for `atm`/`bas`/`env` style callers, and an optional C ABI shim for migration and interoperability. The target users are Rust developers and systems that need libcint-compatible integral evaluation with stronger type safety, clear failure modes, and high-confidence verification.

**Core Value:** Deliver libcint-compatible results through a Rust-native API surface that stays type-safe, verifiable, and safe under memory pressure.

### Constraints

- **Compatibility**: **Bit-exactness** with upstream libcint 6.1.3 — not "close enough to pass the gates". See *Bit-exactness is non-negotiable* below.
- **Architecture**: CubeCL is the primary compute backend - host CPU work stays limited to planning, validation, marshaling, and test/oracle glue.
- **API Surface**: Safe Rust API first, raw compatibility API second, optional C ABI shim third - this ordering drives module boundaries and migration strategy.
- **Error Handling**: Public library errors use `thiserror` v2, while CLI, xtask, benchmarks, and oracle harness code use `anyhow`.
- **Verification**: Full API coverage claims must be backed by the compiled manifest lock, feature-matrix CI, and helper/transform parity checks.
- **Artifacts**: Deliverables written to `/tmp/cintx_artifacts` remain a mandatory part of the design and verification workflow.


## Bit-exactness is non-negotiable

cintx must reproduce libcint 6.1.3 **bit for bit**, not merely within a
tolerance. Every family is at `max_ulp = 0` today (1e ovlp/kin/nuc, 2c2e, 3c1e,
3c2e, 2e; Rys roots at orders 1–5 across all three bands). Do not regress that.

The oracle parity gates assert `atol`/`rtol` of 1e-9…1e-12. **Passing them is
not evidence of bit-exactness** — they passed for a long time while 33–45% of
output elements differed from the vendor. Measure with
`crates/cintx-oracle/tests/libcint_bit_exactness_diagnostic.rs`:

```
CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
    --test libcint_bit_exactness_diagnostic -- --nocapture
```

It reports, per family, the share of elements that are bit-identical and the max
ULP. Two of its three tests are gates; the table is a measurement.

### The rule

Reproduce the vendor's **operations in the vendor's order**. Every divergence
found so far was "the same value in exact arithmetic, a different `f64`". Read
the C and match it literally; do not simplify, factor, or reassociate:

- `rt = aij2 - aij2*ru` is **not** `aij2*(1 - ru)`.
- `a*(1/b)` is not `a/b`; `.5/aij` is a true division.
- `ri + wj*(rj - ri)` is not `(ai*ri + aj*rj)/aij`.
- `(x*ci)*(eij*ekl)` is not `((x*ci)*eij)*ekl`.
- A chained constant (`A*2/B * f(li) * f(lj) * f(lk)`) is not that constant times
  a pre-multiplied `f(li)*f(lj)*f(lk)`.
- libcint's derivative `D_J` is applied **twice through a materialised
  intermediate**, not as one fused closed form.

Loop **nesting** is part of the order: it fixes how the primitive contributions
are summed. So is where a factor enters — libcint folds the contraction
coefficients and `common_fac_sp` into the G-tensor *seed*, before the
recurrence, not onto the finished block.

### Things that are easy to get wrong

- **Read the loop that actually runs.** `CINT2e_drv` branches on the optimizer:
  `opt == NULL` takes `CINT2e_loop_nopt`, which is what the oracle calls and
  which builds its **ket** pair data inline (`ekl = rr_kl*ak*al/akl`,
  `rkl = (ak*rk + al*rl)/akl`) — the *opposite* convention to the bra, which
  does use `CINTset_pairdata`. Assuming the wrong loop cost a family.
- **`CINTrys_roots` short-circuits before the per-order solvers**, to a table
  below `SMALLX_LIMIT = 3e-7` and above `35 + nroots*5`. Skipping those branches
  is a *different algorithm* there, not a rounding of the same one.
- **The sign of zero.** `PRIM2CTR0` makes the first surviving primitive
  **assign**; later ones add. `0.0 + x == x` for every `x` except `-0.0`, so
  zero-then-always-accumulate silently loses signed zeros the vendor keeps.
- **Check the aliasing before adding an accumulator.** For the uncontracted
  shape `CINT1e_loop` aliases `gout = gctri = gctrj = gctr`, so only the Rys
  root sum is grouped; atoms and primitive pairs accumulate flat. Adding a stage
  libcint does not have made `int1e_nuc` *worse*.
- **A canonicalisation swap is not bit-neutral.** Exchanging `i`/`j` to reach
  `li >= lj` is fine for the recurrence, but the coefficient chain, the pair
  centre, the exponent product and the loop nesting must still be formed in the
  **caller's** order.
- **Folding coefficients flips the sign of a screening scalar.** Kernels test
  `fac1 > prim_tol`; once the coefficients ride inside `fac1` its sign is
  theirs, and a contracted s shell routinely carries a negative one. Use
  `F::abs(fac1) > prim_tol`.
- **Constants must round to the vendor's bits.** `SQRTPI` was one ULP low in
  eleven kernels. Copy libcint's literal verbatim rather than a shortened
  decimal.

### In `#[cube]` kernels

Select with **integer arithmetic on a 0/1 flag**, not a runtime `if` — above all
around a loop bound. A runtime `if` assigning a loop bound corrupted the 3c2e
kernel *even on the path where it was never taken*; the same code written as
`n_outer = a*nswap + b*swap`, with values chosen by *offset*
(`exps[(off_i + ip)*nswap + (off_j + jp)*swap]`) so floats are read and never
blended, was correct immediately. `two_electron`'s slot decomposition already
says this; believe it.

Device `f64` constants cannot be literals — `F::new` takes `f32`. Pass them as
scalars or read them from an uploaded `Array<f64>`.

### Localising a residual ULP

Reporting the *ULP*-worst element is a trap: it is almost always a cancellation
near 1e-19 where the metric is meaningless. Report the **magnitude**-worst one.
Then find a failing tuple whose G tensor is trivial (all-`s`) and hand-roll the
vendor's loop for exactly that tuple in host `f64`, taking the roots from
`vendor_CINTrys_roots` so only the factor chain and pair data are under test.
When that reproduction agrees with cintx but not the vendor, your *reading of
libcint* is what is wrong. `locate_the_last_ulp_on_ss_ss` is that experiment,
kept as a gate.

### Before claiming a change is bit-identical

Run the diagnostic **and** the full `cintx-oracle` suite; the tolerance gates are
what prove nothing regressed while the ULP table proves progress. Compiling the
~136 test binaries is what exhausts memory on this host, not running them, so
build first in the foreground (`--no-run`, `CARGO_BUILD_JOBS=2`) and only then
run. Never edit source while a suite is compiling.


## Recommended Stack
### Core Platform
| Technology | Version guidance | Purpose | Why recommended |
|------------|------------------|---------|-----------------|
| Rust toolchain | Pin `1.94.0` in `rust-toolchain.toml` | Reproducible compiler behavior across local dev and CI | Rust 1.94.0 is the current stable release as of 2026-03-05, and pinning an exact toolchain keeps oracle and manifest results reproducible. |
| Cargo lockfile | Commit `Cargo.lock`; run CI with `cargo --locked` | Deterministic dependency graph | Oracle comparisons and manifest audits are only credible if every runner uses the same resolved graph. |
| Cargo resolver | Use edition-2024 default `resolver = "3"`; if the root becomes a virtual workspace, declare it explicitly under `[workspace]` | Predictable feature resolution in a multi-crate workspace | Resolver 3 is the 2024-edition default and is the right baseline for the workspace described in the design doc. |
| Multi-crate workspace | Keep the crate split from the design (`core`, `ops`, `runtime`, `cubecl`, `compat`, `capi`, `oracle`, `xtask`) | Isolate domain types, execution, compat, verification, and tooling | The project has hard boundaries between typed API, compat contracts, backend execution, and release gating; the crate layout should reflect them. |
### Core Libraries
| Library | Version guidance | Purpose | Notes |
|---------|------------------|---------|-------|
| `cubecl` | Pin `0.10.0` | Shared GPU compute backend | cubecl 0.10.0 was published on 2026-05-07 and is the current published line as of 2026-05-09; pinning an exact version preserves oracle reproducibility. The public API stays backend-agnostic enough that a backend swap remains possible if the ecosystem shifts. |
| `thiserror` | `2.0.18` | Public typed error surface | Fits the design requirement for library-facing error enums without leaking implementation details into the API contract. |
| `anyhow` | `1.0.102` | App-boundary, xtask, benchmark, and oracle tooling errors | Matches the design choice to keep ergonomic context-rich errors out of the public library surface. |
| `tracing` | Stay on the current stable `0.1.x` line used by the workspace | Structured spans and diagnostics | Required for planner decisions, chunking, transfers, fallback reasons, and OOM visibility. |
| `bindgen` | Current workspace is `0.71.1`; latest published line is `0.72.1` | Oracle/header binding generation | Upgrade deliberately, not automatically: header-generation changes must be validated against the manifest and oracle harness. |
| `cc` | Keep current stable `1.2.x` line | Vendored upstream libcint build integration | Needed to keep the oracle harness hermetic and reproducible. |
### Supporting Libraries from the Design
| Library | Use | Why it belongs here |
|---------|-----|---------------------|
| `rayon` | Host-side staging and chunk-preparation parallelism | Good fit for CPU-side marshaling without exposing threading complexity in the public API. |
| `smallvec` | Small fixed-ish collections (`dims`, shell tuples, strides) | Cuts heap churn in hot control-plane paths. |
| `num-complex` | Safe API complex/spinor outputs | Better than raw interleaved buffers leaking into typed callers. |
| `approx`, `proptest`, `criterion` | Verification and benchmarking | Match the design's emphasis on oracle comparison, property testing, and repeatable perf baselines. |
### Development Tools
| Tool | Purpose | Notes |
|------|---------|-------|
| `cargo nextest` | Faster and more controllable CI test execution | Useful once oracle, feature-matrix, and regression suites become expensive. |
| `rustfmt` + `clippy` | Baseline style and lint enforcement | Already aligned with the current `rust-toolchain.toml` components. |
| `xtask` commands | Manifest audit, oracle refresh, docs generation, bench reporting | Keeps release gates expressed as code instead of tribal knowledge. |
## Alternatives Considered
| Recommended | Alternative | When the alternative is justified |
|-------------|-------------|----------------------------------|
| `cubecl` | Another GPU backend or a CPU compute backend | Only if CubeCL blocks correctness, platform coverage, or maintainability; do not leak CubeCL-specific types into the public API. |
| `thiserror` for library errors | `anyhow` everywhere | Only for internal binaries or scripts; not for the public library contract. |
| Exact toolchain pin | Floating `stable` | Acceptable for quick local experimentation, but not for release-gated CI or oracle baselines. |
## What Not to Use
| Avoid | Why | Use instead |
|-------|-----|-------------|
| Nightly as the project baseline | Changes compiler behavior and weakens reproducibility | Stable Rust pinned in `rust-toolchain.toml` |
| Unpinned dependency resolution in CI | Makes manifest/oracle drift hard to diagnose | `Cargo.lock` plus `cargo --locked` |
| Public APIs that expose backend-specific runtime types | Makes future backend changes expensive and risky | Keep backend details behind planner/executor traits and typed output views |
| Best-effort partial writes on allocation failure | Violates the design's OOM-safe stop contract | Fallible allocation + typed failure + no partial writes |
## Sources
### Official / primary
- Rust 1.94.0 release announcement: https://blog.rust-lang.org/2026/03/05/Rust-1.94.0/
- Cargo resolver guidance: https://doc.rust-lang.org/nightly/cargo/reference/resolver.html
- Cargo feature resolver details: https://doc.rust-lang.org/stable/cargo/reference/features.html
- Cargo nextest docs: https://nexte.st/
- CubeCL crate docs: https://docs.rs/crate/cubecl/latest
- thiserror crate docs: https://docs.rs/crate/thiserror/latest
- anyhow crate docs: https://docs.rs/crate/anyhow/latest
- bindgen crate docs: https://docs.rs/crate/bindgen/0.71.1 and https://docs.rs/crate/bindgen/latest


### Local project evidence
- `Cargo.toml`
- `Cargo.lock`
- `rust-toolchain.toml`
- `docs/design/cintx_detailed_design.md`

## Conventions

- Before creating any test code, read `\home\chemtech\workspace\cintx\docs\rust_crate_test_guideline.md` and follow it when designing and implementing the tests.



## Cubecl manual
/home/user/Documents/workspace/cubecl_manual/manual/Cubecl

## Rust optimiser manual
/home/user/Documents/workspace/cubecl_manual/manual/optimiser
