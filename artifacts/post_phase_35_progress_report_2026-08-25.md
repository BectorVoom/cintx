# Post-Phase-35 plan — what landed, 2026-08-25

Work against `.planning/notes/post-phase-35-remaining-work-PLAN.md`.
Every number below is measured on this host, not projected.

**Host**: AMD Ryzen AI 7 350 (16 threads) + Radeon 860M (gfx1151), Linux 7.1.9,
CubeCL 0.10.0, vendored libcint 6.1.3.

---

## Gate state

```
cargo fmt --all --check                                  clean
cargo clippy --workspace --all-targets -- -D warnings     clean
  (also clean with CINTX_ORACLE_BUILD_VENDOR=1, which compiles the
   vendor-gated test files the default run skips)
cargo test --workspace --exclude cintx-oracle             28 binaries, 641 passed, 0 failed
CINTX_ORACLE_BUILD_VENDOR=1 cargo test --release
  -p cintx-oracle --features cpu                          99 binaries, 370 passed, 0 failed
cargo check -p cintx-cubecl --no-default-features
  --features {cpu,wgpu,cuda,rocm,metal}                   all five compile
ROCm suite (opt-in, CINTX_ROCM_ORACLE=1)                  4 tests, all green
```

Clippy went from **~2 078 unique warnings to 0**, under `-D warnings`.

---

## Part 1 — launch-class merging

### 35-M1 (`int2e`)

`two_electron_scalar_kernel` has exactly three comptime parameters —
`ibase`, `kbase`, `nroots`. Every other shape scalar (`di`, `dk`, `dl`, `dj`,
`g_size`, `nmax`, `mmax`, `g2d_ijmax`, `g2d_klmax`, `common_factor`) moved out of
the launch arguments into per-class device arrays, and the quartet row grew from
`[si, sj, sk, sl, out_off]` to include a class index.

**69 l-classes → 15 dispatches** on H2O/def2-SVP. The plan predicted 16; the
actual signature count reached by that molecule is 15.

Bit-identity against the per-quartet path holds — `def2_2e_batch_parity` compares
element by element and is green — because each class still indexes only the
leading `3 * g_size` of a slab sized to the widest class in its dispatch.

### 35-M2 (`int3c2e` / `int2c2e` / `int1e_*`)

`nroots` alone is comptime for these (plus a caller-fixed `op_kind` for 1e), so
the merge factor is larger:

| family | l-classes | dispatches |
|---|---:|---:|
| `int3c2e` | 27 | **4** |
| `int2c2e` | 9 | **3** |
| `int1e_ovlp`, `int1e_kin` | 9 | **1** |
| `int1e_nuc` | 9 | **3** |

All bit-identical against both vendored libcint and the per-tuple path.

One real hazard surfaced and was fixed: the 1e host transform scaled the *whole
class buffer* by `common_fac_sp(li) * common_fac_sp(lj)`. Once classes share a
dispatch buffer that would corrupt neighbours, so each class now records its
half-open span and scales only that.

### Measured throughput, CPU backend vs vendored libcint (single-threaded)

Best-of-25, repeated three times, so the spread is the honest quantity:

| workload | before (plan §0) | after |
|---|---|---|
| `int2e` CH4/def2-SVP, 14 706 quartets | 0.52 us/quartet, 1.28x faster | **0.41–0.48 us/quartet, 1.4–1.5x faster** |
| `int2e` H2O/def2-SVP, 3 081 quartets | 1.17 us/quartet, 1.43x **slower** | **0.72–0.85 us/quartet — at parity** |
| `int3c2e` H2O/def2-SVP, 1 728 triples | 1.40 ms, 2.4x slower | 1.17 ms, ~2.1x slower |
| `int1e_ovlp` H2O/def2-SVP | 0.28–0.30 ms | 0.11–0.14 ms |

**On H2O the honest verdict is parity, not a win.** A single best-of-9 run showed
0.703 us/quartet (1.11x faster), but across three repeats cintx ranges 0.72–0.85
and libcint 0.75–0.86 — the intervals overlap. The plan projected 0.56–0.59
us/quartet for this point; the measurement is above that. CH4 is the clear,
repeatable win.

**Where the remaining H2O time goes**: the run splits roughly 60 % backend
dispatch / 40 % host cart-to-sph. Launch count is no longer the bottleneck on
that list — the serial host transform is. That is the next thing worth attacking,
and it is not a backend problem.

New stats fields make the merge observable rather than asserted:
`BatchExecutionStats::launch_classes` (l-classes seen) alongside
`kernel_launch_count` (dispatches), and `max_g_slab_bytes` (widest per-slot
scratch — 27 008 B for 2e, well inside the 256 MB ceiling, which discharges the
plan's "merged-class scratch" risk).

---

## Part 7 — ROCm verification **(the plan's byte-identity bar does not hold)**

The cooperative decomposition (`per_unit == 0`: one tuple per cube, the cube
splitting the contraction, real `sync_cube` barriers) was compiled for every
backend and **never executed** in CI. It now runs, for all five batched families,
in `crates/cintx-oracle/tests/def2_batch_rocm_parity.rs`.

**Correctness — the load-bearing result.** Against vendored libcint at the 1e-10
oracle tolerance, on gfx1151:

| family | max abs diff vs vendor | mismatched |
|---|---|---:|
| `int2e_sph` | 3.109e-15 | 0 |
| `int1e_ovlp_sph` | 4.441e-16 | 0 |
| `int1e_kin_sph` | 1.398e-14 | 0 |
| `int1e_nuc_sph` | 1.954e-14 | 0 |
| `int2c2e_sph` | 2.665e-14 | 0 |
| `int3c2e_sph` | 1.599e-14 | 0 |

**The plan's stated acceptance — "byte-identical to the CPU results" — is not
achievable, and the reason is not the launch topology.** Both backends compile
the same `#[cube]` source, but through different compilers to different ISAs, and
the AMD one contracts multiply-add pairs the CPU one leaves separate. Measured
CPU-vs-ROCm divergence, as a multiple of `f64::EPSILON` times the block's largest
element: 0.26–2.72 eps. On `int2e` the ROCm result is in fact *closer* to vendored
libcint than the CPU one (3.109e-15 vs 3.331e-15) — the contracted form is the
more accurate of the two.

The gate is therefore an eps-of-block-scale bound (8 eps), not bit-identity. It
still catches what the test exists for: a lane writing another lane's element, a
barrier in the wrong place, a stale G slab, a mis-sized merged scratch slab. None
of those perturb a result by two ULP.

Two metrics were tried and rejected before settling on this one, and the reasons
are recorded in the test: bit-identity (unachievable across ISAs) and a pure ULP
count (meaningless near zero — a `+1e-17` / `-1e-17` pair reads as 8.7e18 ULP,
and a small result carrying a few ULP of the *largest term's* rounding reads as
hundreds of ULP of itself).

---

## Part 2 — Phase 33, task 33-05 **(discharged for ROCm)**

The plan calls 33-05 "**the blocking item**" and records that it "cannot be
discharged for any GPU backend on the current dev host". It can be, and it passes.

New module `crates/cintx-cubecl/src/device_rys_ceiling.rs`:

- `probe_fma_fusion(backend)` — one launch over six operand pairs whose products
  are not exactly representable, comparing the device `fma(a, b, -(a*b))` error
  term against the host `f64::mul_add` reference **bit for bit**. Cached per
  backend arm. Replaces the CPU-hardcoded probe in `rys_wheeler.rs`.
- `device_nroots_ceiling(backend)` — returns `BASE_DEVICE_NROOTS` (5) unless
  **both** the `extended-device-rys` feature is compiled in (off by default,
  task 33-03's per-family opt-in) **and** the probe passed on that backend.
  Neither is sufficient alone.

Wired into all four batch guards (`int2e`, `int3c2e`, `int2c2e`, `int1e_*`), which
previously read a global constant.

**Result**: `fused=true, divergent=0/6` on **both** CPU and ROCm. The double-double
TwoProd error term is exact on gfx1151; the dd chain is safe there.

Worth recording for whoever does 33-01..33-04: the FMA contraction the ROCm
measurement above exposes is *not* the 33-05 hazard. `two_prod_dev` asks for an
`fma` explicitly and wants it; `two_sum_dev` contains no multiply-add at all, so
contraction cannot reach it. The hazard was always narrower than the plan feared,
and the probe now measures exactly it.

The ceiling still reads 5 everywhere — the raise is 33-03 and needs a green
per-family oracle parity test as well as this probe. A test asserts that a
passing probe alone does not raise it.

---

## Part 4 — def2/J and def2/JK

Both auxiliary sets added to `cintx-basis`, fetched from BSE at the same software
version (0.12) and data version (1, Turbomole 7.3) as the three files already
vendored, so all five are one consistent snapshot. Provenance recorded in
`crates/cintx-basis/data/README.md` per that file's own rules.

- `StandardBasis::{Def2JFit, Def2JkFit}`, resolvable by either the literature
  name (`def2/J`) or the BSE export name (`def2-universal-jfit`).
- `StandardBasis::is_auxiliary()` — a fitting basis is not an orbital basis, and
  mixing them produces a plausible-looking calculation of the wrong thing, so the
  distinction is on the type.
- `to_raw_arrays_with_auxiliary()` — emits the orbital and auxiliary shells into
  one `bas` array with `RawArrays::{orbital_shells, auxiliary_shells}` naming the
  two ranges. This is the layout a `(mu nu | P)` list needs. It refuses an
  orbital basis in the auxiliary slot.

**Parity** (`def2_rij_auxiliary_parity.rs`), H2O/def2-SVP, vs vendored libcint:

| | pairs/triples | classes | max abs diff |
|---|---:|---:|---|
| `int2c2e`, def2/J | 625 | 25 | 6.2e-14 |
| `int2c2e`, def2/JK | 1 369 | 25 | 9.2e-14 |
| `int3c2e`, def2/J | 3 600 | 45 | 1.7e-14 |
| `int3c2e`, def2/JK | 5 328 | 45 | 1.9e-14 |

These are not redundant with the existing def2-SVP gates: the auxiliary sets reach
`l_max = 4` where the AO fixture reaches 2, so they exercise launch classes the
AO-only work lists never produced.

**RI-J benchmark** — the `nbas^2 x naux` list, `mu <= nu`:

| workload | triples | dispatches (of l-classes) | cintx | libcint |
|---|---:|---|---|---|
| H2O/def2-SVP + def2/J | 1 950 | 5 (of 45) | 0.83 us/triple | 0.36 |
| H2O/def2-SVP + def2/JK | 2 886 | 5 (of 45) | 0.63 us/triple | 0.33 |

**cintx is ~2x slower on RI-J.** The 9x launch merge did not close it: the split
is ~60 % dispatch / ~40 % host cart-to-sph, the same shape as `int2e` on H2O.

H2O/def2-**TZVP** + def2/J is **outside the device envelope**: `l = (3,3,4)` needs
`nroots = 6`, one past the ceiling. The benchmark reports and skips it rather than
failing. This is precisely the workload Phase 33 would unblock, and it is the
first time the ceiling has bitten a real target rather than a synthetic one.

---

## Part 3 — task 35-D (derivative-family batching), first two waves

The plan's priority order is `int3c2e_ip1`/`ip2` (RI-J gradients), then
`int1e_ip*` (nuclear gradients), then the rest. The first two are done.

Its acceptance bar is **bit-identity against the per-tuple path, enforced by
rewriting the per-tuple entry point as a one-tuple batch** so every existing
parity test covers the batched kernel. That is exactly how each conversion
lands: the compatibility API still evaluates one tuple, but as a one-tuple
launch group through the same kernel — so there is no second code path that
merely ought to agree.

Each conversion also collapsed a five-arm backend `match` (one arm per runtime,
identical apart from the type) into a single dispatcher, removing ~110 lines per
family.

### Converted

| family | before | after | vs per-tuple | vs libcint |
|---|---:|---:|---:|---:|
| `int3c2e_ip1` | 1 728 launches | **4** (of 27 classes) | **25.3x** | 1.59x slower |
| `int3c2e_ip2` | 1 728 launches | **4** (of 27 classes) | **26.4x** | 1.52x slower |
| `int1e_ipovlp` | 144 launches | **1** (of 9 classes) | **33.2x** | **1.04x — parity** |
| `int1e_ipkin` | 144 launches | **1** (of 9 classes) | **29.8x** | 1.78x slower |
| `int1e_ipnuc` | 144 launches | **3** (of 9 classes) | **24.8x** | 1.96x slower |

`int1e_ipovlp` and `int1e_ipkin` collapse to a *single* dispatch because, once
the shape scalars are per-pair, nothing is left to specialize on — `op_kind` is
fixed by the caller's operator. The Rys families keep one dispatch per order.

New public surface: `evaluate_3c2e_deriv_triple_batch{,_resident}` with
`ThreeC2eDerivFamily`, and `evaluate_1e_deriv_pair_batch` with
`OneEDerivOperator`. Without these the kernels would be batch-*capable* but no
caller could batch, and the launch count in practice would be unchanged.

### Gates

`def2_3c2e_deriv_batch_parity` and `def2_1e_deriv_batch_parity` each assert two
separate things over the full def2-SVP water list:

- **bit-identity** against the per-tuple path (same kernel, so anything else
  means batching moved a result), and
- agreement with **vendored libcint** — max abs diff 6.7e-16 … 1.4e-13, which is
  what makes the first claim worth having, since two cintx paths agreeing on a
  wrong answer would satisfy it alone.

The same s/p-normalization hazard 35-M2 hit in the scalar 1e path recurs here
and is handled the same way: each class records its half-open span of the shared
dispatch buffer and scales only that.

### ROCm

All five newly-converted families run on the gfx1151 cooperative path and match
vendored libcint with **0 mismatches**.

One measurement changed how they are gated. The eps-of-block-scale bound that
works for the scalar families is the wrong instrument for a gradient kernel: the
kinetic arm builds second differences (`coef_hi * g[n + 2dj] - coef_mid * g[n]`)
and the cancellation there turns a 2-ULP difference in an intermediate into a
5.8e-14 difference in the result — 10 eps of scale, over the scalar families'
8-eps bound, on 2 of 1 728 elements. Rather than widen a bound until it stops
failing, the derivative families are gated on **vendor agreement** and the
CPU-vs-ROCm distance is reported as context.

### Remaining

10 of the 15 1e derivative/special kernels are still per-tuple, plus `sigma_p`
(6 sites), `ecp` (4), `center_3c1e` (2), `center_4c1e` (2), `f12` (2),
`sigma_1e_nuc` (2), `sigma_1e` (1) and the 16 `unstable-source-api` sites. The
transformation is now fully established and mechanical — signature, slot/lane
prologue, offset rebasing through the existing locals, close the tuple loop,
replace the five-arm match — but each kernel is 120–360 lines with its own
scratch-slab set, so it is per-kernel work rather than one sweep.

---

## Part 3 — task 34-C2 (resident basis for the other families)

The flattened basis form (`exps` / `coeffs` / `centers` / `shell_meta`) is
identical across families, so the existing residency generalizes rather than being
duplicated:

- `ResidentBasis` — an alias for `ResidentTwoEBasis`, whose name predates the
  generalization. Renaming a public type was not worth a compatibility break.
- `evaluate_3c2e_triple_batch_resident()` — `int3c2e` against a basis already on
  the device. The non-resident entry point now builds a throwaway residency, so
  both paths run one code path.
- `ThreeC2eFlatBasis` deleted; it was a second spelling of the same four buffers.

**Gate**: `resident_basis_serves_3c2e_uploads_once_and_changes_nothing` — the
same two-sided assertion `int2e` carries. `basis_upload_bytes` is the full upload
on the first call and **0** on every later one, transfer strictly decreases, and
every value is **bit-identical** to the throwaway-residency path.

This is the case RI-J wants: a Fock build evaluates the same triple list every SCF
iteration and the basis does not change between them.

---

## Part 6 — clippy

`cargo clippy --workspace --all-targets -- -D warnings` is clean, from ~2 078
unique warnings. Landed by lint, not by sweep, and never with `--fix` on a table
module. Every `#[allow]` carries a reason.

The ~50 the plan flagged as "read each one" were read, and three were real:

1. **`float_literal_f32_fallback`, 608 sites** — not in the plan's table, and not
   cosmetic: *"this was previously accepted by the compiler but is being phased
   out; it will become a hard error in a future release"*. `F::new(0.0)` was
   silently falling back to `f32` because `f32: From<f64>` is not satisfied. Fixed
   by appending the explicit suffix, applied from the compiler's own
   MachineApplicable spans and verified to be a pure suffix append at every site —
   the value was already `f32`, so nothing changed but the reader's certainty.

2. **A doubled comptime guard** in `one_electron_scalar_kernel` —
   `if comptime!(per_unit == 0) { if comptime!(per_unit == 0) { sync_cube(); } }`.
   Collapsing it produced `A && A`, which `clippy::eq_op` denies, which is how it
   surfaced.

3. **A vestigial mismatch counter** in the oracle helper comparison — 20
   `mismatches += 1` sites each immediately followed by `bail!`, making the final
   `if mismatches > 0` aggregate unreachable. Removed; the function is fail-fast
   by construction and each `bail!` already names the specific disagreement.

Also read and dispositioned, not silenced blindly:

- **`rys_wheeler.rs`'s entire host long-double / `Dd` chain is dead** — ~21 items.
  `rys_roots_host_wheeler` dispatches nroots 8..12 to the *device* dd
  implementations, so nothing calls the host one. It is a superseded but
  independent transcription, kept as the cross-check those device kernels were
  validated against, and now annotated as such — with a pointer that 33-05's
  hazard lives in `two_prod_dev`, not in the host `two_prod`.
- Four leftover locals (`total_g`, `nrys`) that were stale copies of sizing
  expressions the host already owns — removed rather than underscored, because an
  unread copy of a slab size invites drifting out of step with the slab.
- `vrr_step_host` / `compute_pdata_host` / `hash_shell_tuple`: reachable only from
  `#[cfg(test)]` code; gated accordingly rather than allowed.
- 1 974 `excessive_precision` and 162 `identity_op`: module-level allows with the
  transcription-fidelity and column-alignment rationales, matching the
  `approx_constant` precedent already in the crate.
- 102 `missing_safety_doc` in the C ABI shim: real, and now documented — the
  `atm`/`bas`/`env` coherence contract that the callee cannot check, and the
  pointer/length contract on the `cintrs_*` entry points.

---

## Not done

Carried into `.planning/notes/post-phase-35-continuation-PLAN.md`, which
re-prioritises around the finding that the serial host cart-to-sph transform
is now the largest single cost in three of the four workloads still slower
than libcint.

| item | why |
|---|---|
| **35-D** for the remaining families | `int3c2e_ip1`/`ip2` and `int1e_ipovlp`/`ipkin`/`ipnuc` are done (see above). 10 more 1e kernels, plus `sigma_p`, `ecp`, `f12`, `center_3c1e`/`4c1e` and the `unstable-source-api` set, remain per-tuple. |
| **35-F2** facade parity for pair/triple batches | Public surface plus a manifest-lock update. **Correction**: I first reported this as blocked because `cargo run -p xtask` fails — but `xtask` is a *separate workspace* under `xtask/`, and `(cd xtask && cargo run -- manifest-audit --check-lock)` works and reports `status: ok`. Nothing blocks it. |
| **34-D2** primitive screening for the other families | Not started. |
| **34-C2** for `int2c2e` / `int1e_*` | Only `int3c2e` was converted — the family the RI-J case motivates. The machinery is now shared, so the rest is additive. |
| **Part 5** device-resident output | Blocked on open question 2, as the plan states. |
| **33-01..33-04** | The scaffolding is in; the ceiling raise itself is untouched. |

---

## Open questions — where this leaves them

1. **Which backend is the throughput target.** Narrowed further. On CPU, `int2e`
   is 1.4–1.5x faster than libcint on CH4 and at parity on H2O. Both `int2e`-H2O
   and RI-J now spend ~40 % of their time in the *serial host* cart-to-sph, which
   no backend change touches. That is the next bottleneck, and it argues for
   attacking the transform before porting anything.
2. **Device-resident output.** Still unanswered, still blocking Part 5.
3. **def2-ECP (Z >= 37).** Untouched.
4. **def2/J and def2/JK.** Answered: added, parity-gated, and benchmarked.
5. **Does the `hess2e` 1e-12 gate still bind at nroots 6-7.** Untouched — it needs
   33-01's inline roots first.

One question the plan did not ask, now answered: **does the ROCm backend fuse
`fma`?** Yes, bit-for-bit. That was 33-05's blocking unknown for the only GPU
backend this host can run.
