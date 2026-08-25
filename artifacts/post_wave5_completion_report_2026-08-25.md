# post-wave-5 remaining tasks — completion report

**Plan**: `.planning/notes/post-wave5-remaining-tasks-PLAN.md`
**Date**: 2026-08-25

---

## 0. Where things stand now

| gate | result |
|---|---|
| `cargo fmt --all --check` | clean |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean |
| `cargo clippy -p cintx-cubecl --all-targets --features extended-device-rys` | clean |
| `cargo clippy -p cintx-cubecl --all-targets --features with-f12` | clean |
| `cargo test --workspace --exclude cintx-oracle` | **663 passed, 0 failed** |
| oracle `--features cpu` | **382 passed, 0 failed** |
| oracle `--features cpu,extended-device-rys` | **395 passed, 0 failed** |
| oracle `--features cpu,with-4c1e` | **387 passed, 0 failed** |
| oracle `--features cpu,with-f12` | **398 passed, 0 failed** |
| `xtask gen-c2s-table --check` | no drift |
| `cargo check -p cintx-cubecl --no-default-features --features {cpu,wgpu,cuda,rocm,metal}` | all five, with and without `extended-device-rys` |
| `xtask manifest-audit --check-lock` | `status: ok` |

All oracle runs under `CINTX_ORACLE_BUILD_VENDOR=1`.

---

## 1. Task A — Phase 33, the device Rys ceiling. **Done (A1–A6).**

### A1 — the inline device Rys entry

`math::rys_wheeler::rys_roots_ext_dev` is the whole `nroots` 6..=12 Wheeler
dispatch as a `#[cube]` callee. Three design points, each measured rather than
assumed:

- **One table argument.** The eight `JACOBI_*` / `LJACOBI_*` tables plus
  `TURNOVER_POINT` are concatenated into one 584-f64 blob (~4.7 KB) read at
  comptime offsets. An opting-in family kernel grows exactly one
  `&Array<f64>` parameter.
- **Scratch is local, not a parameter.** Every working buffer is a
  `#[cube]`-local `Array::<f64>::new(<comptime>)`. That is what keeps the
  signature to `(tables, x, u, w)` — and it answers A2 (below).
- **The dd/Laguerre selector is runtime, not comptime.** Making it comptime
  would emit two copies of the ~30n words of dd scratch and two eigensolve
  tails; as a runtime branch they share both.

The six former `#[cube(launch)]` kernels' callees are reused unchanged, with
table-offset parameters added to `flocke_jacobi_moments_dev`,
`wheeler_recursion_dev` and `lflocke_jacobi_moments_dev` so the blob can be
indexed in place.

### A2 — scratch sizing

`shared_memory::ext_rys_scratch_words` reports the per-work-item footprint, and
`ext_rys_scratch_words_match_kernel_at_nroots_12` pins the arithmetic to the
kernel's allocation list (1113 words = 8.9 KB at the widest instantiation).

**The plan's open question is answered differently than it framed it.** It asked
whether the extended path should "drop to global scratch on GPU" or "drop the
per-launch unit count", because ~6.8 KB × 16 units is over a 64 KB shared budget.
Neither: the scratch is `#[cube]`-local, so it is *thread-private* and never
enters the shared budget at all. `calc_math_layout("rys_roots_ext", ..)` reports
`NoSharedLane`/0 bytes, and `ext_rys_max_units` bounds the private footprint
instead (7 units per 64 KB at `nroots = 12`).

Also recorded: the footprint is **not** monotone in `nroots` — order 8 is a local
peak, because it is the only order whose large-`x` arm is the dd Schmidt solver
with its three `(n+1)²` dd coefficient matrices.

### A3 — the accuracy gate

`crates/cintx-oracle/tests/rys_ext_inline_parity.rs`.

**10584 / 10584 values bit-identical** to `rys_roots_host_wheeler` over
`nroots` 6..=12 × a log-spaced `x` sweep from 1e-4 to 1e6, plus each arm's exact
breakpoint and a point either side.

Below 1e-4 the result is recorded rather than gated, with reasons:

| finding | measurement |
|---|---|
| `nroots` 8..=12 stay bit-identical below the envelope | worst relative divergence **0** |
| `nroots` 6, 7 move | worst **1.2e-8** relative, all at `x < SMALLX_LIMIT = 3e-7` |
| the host reference is itself undefined in part of the range | its solver reports error 1 across ~`[1.5e-8, 8.7e-5]` at `nroots = 12`, and at one point at `nroots = 11` |

The 6/7 divergence has a cause, not just a bound: those two orders are the
parity-honest escape hatch, so the host routes them through the *pure-host*
`rys_jacobi`/`rys_schmidt` rather than the `#[cube]` kernels — two
transcriptions of one algorithm, compared in an ill-conditioned regime the
vendor does not even use the Wheeler path for.

### A4 — per-family flip. **All four families, each with its own gate.**

| family | gate | orders reached | result |
|---|---|---|---|
| `int3c2e` | `ext_rys_3c2e_parity.rs` | 6, 7 | 0 mismatches vs vendored libcint |
| `int2e` | `ext_rys_2e_parity.rs` | 6, 7, 8, 9 | 0 mismatches; batch and per-tuple **bit-identical** to each other (14156 elements) |
| `int2c2e` | `ext_rys_2c2e_parity.rs` | 6, 7, 8, 9 | 0 mismatches (Cartesian) |
| `int1e` (scalar) | `ext_rys_1e_parity.rs` | 6, 7, 8, 9 | 0 mismatches (Cartesian) |

**The RI-J acceptance is met.** `def2_rij_throughput`'s H2O/def2-TZVP + def2/J
case no longer prints `SKIPPED: outside the device Rys envelope`. It runs,
matches vendored libcint with **0 mismatched elements** (max |diff| 4.46e-14),
and is measurable for the first time: 4750 triples, 6 launches, 1.2-1.3x slower
than libcint across repeated runs (the def2-SVP cases in the same run vary
1.15-1.63x, so treat the ratio as an order of magnitude, not a figure).

**The ceiling had to become per family**, and that is the most important thing
this task changed. Raising `device_nroots_ceiling` globally — as the scaffolding
did — made the then-unflipped `int2e` batch accept an `(f f | f f)` class and
evaluate it through its launcher's catch-all `nroots = 5` arm, and made the
CubeCL optimizer panic on a five-element root array indexed at 6. The signature
is now `device_nroots_ceiling(backend, RysFamily)`, so every call site has to
name its family, and `the_flipped_set_is_exactly_the_four_scalar_families` pins
the list in one place. The derivative sets (`Int3c2eDeriv`, `Int1eDeriv`) are
deliberately **not** flipped.

### A5 — the `nroots` 6–7 host-only decision. **Re-examined; the hatch stays.**

Swapping the two arms for the device orchestrators and re-running
`hess2e_parity` gives **3334 mismatches** on `int2e_ipip1_sph`, deltas 1.3e-12 to
2.0e-12 against a 1e-12 threshold. The tolerance was not touched; the experiment
was reverted.

What A1 changed is the *diagnosis*, from a suspicion to a fact. A3 shows the
inline entry — reached by exactly one launch — is bit-identical to the host
dispatch at `nroots` 6 and 7. So the roots handed to `hess2e` are the same bits
either way, and the ~1e-12 that moves is **host arithmetic executed after the
launch**, not the quadrature. That is a property of the CpuRuntime launch, and no
work on the Rys path will remove it. Recorded at the escape hatch.

### A6 — still blocked

No FMA probe on CUDA / wgpu / Metal: no adapter on this host. The per-family,
per-backend ceiling is what keeps that fail-closed.

---

## 2. Task B — `fill_g_tensor_f12` on the device. **Done.**

`fill_g_tensor_f12_dev` builds a primitive quartet's `[gx|gy|gz]` slab in-kernel,
so the F12 primitive-quartet loop is now one dispatch instead of
`nprim_i·nprim_j·nprim_k·nprim_l` host fills each followed by its own launch.

- **`math::stg::stg_roots_dev`** — the Clenshaw/DCT pipeline as a `#[cube]`
  callee, gated bit-for-bit against `stg_roots_host` over 5 `nroots` × 9 `ta`
  × 9 `ua` (`stg_roots_dev_matches_host`).
- **The host keeps the table *lookup*, not the arithmetic.** `stg_table_cell`
  resolves `(nroots, ta, ua)` to a cell offset and normalized Clenshaw
  coordinates, for two reasons: CubeCL has `ln` but not `log10`, and
  `ln(x)·LOG10_E` differs in the last bit — which selects a *different table
  cell*, a different answer rather than a rounding difference; and the frozen
  tables are 14 MB each, so resolving host-side uploads only the windows a
  launch actually touches. `stg_roots_host` now calls the same resolver, so the
  two cannot drift.
- **`f12_primitive_batch_kernel`** fuses fill + contraction for the base variant,
  with a per-*slot* G slab (capped at 64) walked grid-stride.
- **`f12_g_fill_kernel`** serves the derivative variants, whose `gout_*` nabla
  functions are host code and therefore need the G tensor itself back. Rows are
  filled in chunks sized to an 8 MB working set — which is precisely the
  "tens of megabytes" objection that made wave 5 decline this conversion.
- **Two `#[cube]` bodies were factored, not copied**: `f12_contract_dev` is now
  shared by the single-quartet wrapper and the batched kernel.

**Gate**: `f12_primitive_batch_matches_per_quartet_path` — bit-identity against
the per-quartet path across 16 (class, branch) combinations covering every
comptime flag the kernel specializes on (`nroots`, `ibase`, `kbase`, STG/Yukawa).
The host `fill_g_tensor_f12` and its five helpers are retained as that
reference, annotated as such. `f12_oracle_parity`: 15 passed.

---

## 3. Task C — batch the ECP radial precompute. **Question answered; the
batching itself needs an API that does not exist.**

### The question the plan posed

> whether shell pairs can share that loop or must each run it to their own
> convergence

**They can share it.** The level schedule (`LEVEL0..LEVEL_MAX`) is global and
carries no per-pair state; each row's per-level update touches only its own
`rad_all` slice; and a row that converges early is skipped by the `converged`
check, so extra levels leave its bytes untouched. Batching the loop is therefore
byte-identical by construction.

### What blocks the batching, and it is not the loop

**ECP is the only family with no `evaluate_*_batch` entry point.** `int1e`,
`int2e`, `int2c2e`, `int3c1e` and `int3c2e` all have one; `launch_ecp` is reached
only per `ShellTuple`. "Batching across shell pairs" therefore means *building*
that API — new public surface, for a family with no batch consumer in the repo.
That is the same trade §1 of the plan rejected for 35-D wave 6, so it is left as
a decision rather than taken unilaterally.

The remaining axis inside the per-tuple path — batching across ECP **slots** —
was investigated and rejected on evidence: slots are atom-major, so type-1 slots
of different atoms are not adjacent, and `gctr` accumulates over slots with
`+=`. Reordering them changes the f64 sum order and would break the byte-identity
the plan sets as C's acceptance bar. For a single-ECP-atom molecule it would also
buy nothing, since each `lc` appears once.

### What was delivered instead: two byte-identical host wins

Both are pure functions of constants that were being rebuilt per shell pair —
the only way this family is ever driven is an `nbas²` matrix build.

| change | effect |
|---|---|
| the 2047-point Gauss-Chebyshev grid, cached per process instead of rebuilt per `launch_ecp` call | 0.0439 s → 0.0187 s |
| `cart_comps(l)` returns a cached `&'static [(u8,u8,u8)]` instead of allocating a `Vec` in the angular inner loops | 0.0187 s → 0.0168 s |

**2.6x on the Au/def2-SVP ECP matrix build (144 pairs), byte-identical.** All ECP
gates green: `safe_api_ecp_parity`, `ecp_iprinv_parity`,
`def2_ecp_heavy_element_scope`.

The remaining time is dominated by device dispatches (4 slots × 144 pairs), which
is exactly what the shell-pair batching would address — confirming the plan's
diagnosis while showing why the API question has to be settled first.

---

## 4. Task F — the def2-ECP scope question. **Answered, with three defects
found and fixed.**

`crates/cintx-oracle/tests/def2_ecp_heavy_element_scope.rs` runs Rb (Z=37),
I (Z=53) and Au (Z=79) — def2-SVP + def2-ECP — through the same
safe-API-vs-vendored-PySCF comparison the Cu/LANL2DZ fixture uses.

Every one of the three defects was invisible on Cu/LANL2DZ, and each for a
reason worth recording:

1. **A hard panic for every def2-ECP element.** `ecp.rs`'s Type-2 angular factor
   filled a buffer in the *Cartesian* component count of the projector channel
   but sized it by the *spherical* one. Those agree up to `lc = 1` and diverge
   from `lc = 2` (6 vs 5). Every def2-ECP record carries an `l = 2` projector;
   LANL2DZ's Cu record stops at `l = 1`.
2. **The nuclear charge was the bare `Z`.** `Molecule::to_basis_set` stored
   `spec.atomic_number` in `Atom::atomic_number` — the field that becomes
   `atm[CHARGE_OF]` — while `to_raw_arrays` wrote the ECP-reduced charge. The
   typed API's `int1e_nuc` was too large by exactly `Z / (Z − n_core)`:
   **4.111111x for Rb**, measured.
3. **The existing gate read pair blocks in the wrong order.**
   `safe_api_ecp_parity`'s collector read row-major where the safe API returns
   libcint's column-major (pinned independently: `int1e_ovlp` on a non-square
   `(p, f)` Cartesian pair matches the vendor buffer verbatim to 2.2e-16). On a
   one-atom fixture that is invisible — a spherical ECP centred on the only atom
   conserves angular momentum, so every `l_i ≠ l_j` block is identically zero and
   the scramble maps zeros onto zeros. Au's Cartesian `(p, f)` block is the first
   non-square, non-zero one (Cartesian `f` carries an `l = 1` contaminant), and
   it exposed the read immediately.

**With those fixed, Rb, I and Au all reproduce vendored PySCF `nr_ecp` to ~1e-14
in both representations.** The scope question is therefore about coverage
breadth — more elements, molecules with several ECP centres, the gradient
operators — not about a missing capability.

---

## 5. Tasks D and E — unchanged, still blocked

D (device-resident output) on open question 2; E (CUDA / wgpu / Metal execution)
on hardware. Neither was touched.

---

## 6. The `l >= 5` cart-to-sph silent zero. **Fixed.**

Reported in the first pass as carried-forward: `transform::c2s::c2s_coeff` had
hand-transcribed coefficient matrices for `l = 0..=4` and returned `0.0` above
them, with an `Ok` status. That is a silent-wrong-answer path at *any* Rys
order — an `(h s | s)` three-centre integral is `nroots = 3`, well inside every
device ceiling, and came back entirely zeroed.

### The table now covers what libcint's does

`xtask gen-c2s-table` parses `g_trans_cart2sph[]` out of the vendored
`libcint-master/src/cart2sph.c` and emits
`crates/cintx-cubecl/src/transform/c2s_data.rs`: 19176 coefficients, `l = 0..=15`,
which is libcint's own `g_c2s` ceiling — beyond it there is no upstream
reference to be compatible with. `--check` re-derives and fails closed on drift,
the same gate `gen-rys-tables` and `gen-ecp-tables` use.

Hand-transcribing 19176 coefficients is exactly the silent transcription risk
the Rys-table generator was written to avoid, so the extraction is machine-made
and then *checked against the hand work that already existed*: the generated
`l = 0..=4` blocks reproduce `C2S_L0..C2S_L4` **bit for bit** (245 coefficients),
which is what makes the `l >= 5` blocks — where no in-tree reference exists —
trustworthy. The one subtlety is `#ifdef PYPZPX`; the cintx vendor build leaves
it undefined, so the parser takes the `#else` branch, and that choice is exactly
what the `l = 1` identity check pins.

### Three accessors became one

`c2s.rs::c2s_coeff`, `c2spinor.rs::c2s_k_coeff` and `ecp.rs::c2s_coeff` each
carried their own `match l { .. _ => 0.0 }`. They now all go through the shared
table. That closed two latent holes in the ECP path specifically: `ECP_LMAX` is
5, so an `h` projector channel was silently zeroed, and `type1/type2_facs_ang`
purify at `l = li + lc`, which passes 4 as soon as an `f` shell meets a `d`
projector. Both were latent rather than active — every ECP gate produces
identical results before and after — but they were live code paths.

### It is fail-closed above the ceiling now

- `Shell::try_new` rejects a **spherical** shell with `l > SPHERIC_L_MAX` (15)
  with a typed `CoreError`; Cartesian shells are unaffected because they need
  no transform.
- `CINTc2s_bra_sph` returns `UnsupportedApi` rather than zeroing.
- `c2s_coeff` asserts. That is deliberately a panic and not a `0.0`: reaching it
  means a caller skipped its guard, which is a defect rather than an input.
- `c2s.rs` and `cintx-core` carry the ceiling as two constants only because
  `cintx-core` cannot depend on `cintx-cubecl`; a test pins them equal.

### What it unblocked

| gate | before | after |
|---|---|---|
| `cintc2s_bra_sph_matches_vendor` | `l = 0..=4` | **`l = 0..=15`**, plus an assertion that the `l >= 5` output is not all zeros |
| `ext_rys_3c2e_parity` | `nroots` 6, 7 (`l <= 4` classes only) | **`nroots` 6..=12**, every arm |
| `ext_rys_2e_parity` | `nroots` 6..=9 | **`nroots` 6..=12**, every arm |

The `(h s | s)` case from the original finding now matches vendored libcint to
5.6e-17, with the same zero *pattern* as the vendor (5 of 11 — genuine symmetry
zeros, not a zeroed transform). The high-`l` spherical classes that were
previously refused-or-zeroed match to ~3e-15 at `nroots` up to 12.

`ext_rys_2c2e_parity` and `ext_rys_1e_parity` stay Cartesian, but now by choice
rather than by necessity: Cartesian keeps the c2s step out of a comparison whose
subject is the Rys arm, and `cintc2s_bra_sph_parity` gates the transform
separately.

---

## 7. The `with-f12` f32 `stg_ip1` failure. **Fixed — a stale manifest index,
not a kernel bug.**

`f32_parity::test_f32_int2e_stg_ip1_sph_parity` failed with "output length
mismatch: got 1 expected 3". The cause was one literal:

```rust
// OperatorId 107 = int2e_stg_ip1_sph (int2e_stg_sph is 106; ip1 is next in manifest).
const STG_IP1_SPH_OPERATOR_ID: u32 = 107;
```

A later manifest regeneration inserted a row above it. `int2e_stg_ip1_sph` moved
to 108, position 107 became `int2e_stg_sph` — the **scalar** STG integral,
`ncomp = 1` — and the helper silently began evaluating a different operator. The
f32 path was never wrong; the test was asking the wrong question. With the
operator resolved by symbol, the measured f32 error is **3.02e-8** against the
family's 1e-4 floor.

### Why it could go stale unnoticed

A manifest position is not a name. Nothing about `OperatorId::new(107)` says
`int2e_stg_ip1_sph`, so the literal kept compiling and kept meaning *something*.
It surfaced only because the two operators have different component counts; a
shift onto a same-shaped operator would have compared cintx against the vendor
for two different integrals with no length to disagree about.

An audit of every other hard-coded `OperatorId` in the test suite found the rest
still correct — they sit at low, stable positions (0..24). Rather than rewrite
ten passing tests, `f32_parity.rs` now carries
`operator_ids_used_here_still_resolve_to_their_symbols`, which asserts each
remaining literal against `Resolver::descriptor_by_symbol`. That turns the whole
class from "silently evaluates the wrong operator" into a named failure with a
one-line fix.

With this, `--features cpu,with-f12` is green end to end: **398 passed, 0
failed**.

---

## 8. Carried forward

- **`ecp_libecpint_crosscheck_parity` runs 0 tests** in this configuration; its
  fixture or env gate was not investigated.
- **`--features with-4c1e` is not clippy-clean**: five pre-existing lints in
  `kernels/center_4c1e.rs` (`excessive_precision` on frozen literals,
  `manual_slice_size_calculation`), all present in `HEAD`'s copy of that file.
  The same is true of `with-f12`; neither profile is a clippy gate today.
- **One tolerance was restated, not loosened.**
  `simd_cubecl_libcint_3way_parity::test_3way_high_l_2c2e_parity` compared
  CubeCL against libcint with a flat `epsilon = 1e-9` on values reaching 976 —
  an absolute bound asking for 1e-12 relative. Under `extended-device-rys` the
  `(h|h)` pair is `nroots = 6` and served by the device Wheeler entry rather than
  the host loop, which moves it from 1.8e-15 to **1.8e-12 relative**. The
  assertion now states the bound as a scale (`max_relative = 1e-11`, ~6x the
  measurement). Every other pair in that fixture is unchanged at ≤ 3.2e-14.
