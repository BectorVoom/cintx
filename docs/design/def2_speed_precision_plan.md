# def2-SVP / def2-TZVP Speed and Precision Plan

Status: proposed
Scope: basis-set-targeted follow-on to `docs/design/cubecl_speed_optimization_plan.md`
Primary compatibility target: libcint 6.1.3, unified `atol = rtol = 1e-12`
CubeCL target: pinned workspace version `0.10.0`

## 1. Purpose

The general speed plan optimizes the execution architecture. This plan optimizes the two
workloads a production caller actually runs: full-molecule integral evaluation in
**def2-SVP** and **def2-TZVP**. The two basis sets sit on opposite sides of the device
envelope and therefore need different work:

- **def2-SVP** lands entirely inside the existing device envelope
  (`BASE_DEVICE_NROOTS = 5` covers an angular-momentum sum of 8; pinned by
  `def2_svp_fits_the_device_envelope` in
  `crates/cintx-oracle/tests/def2_throughput_benchmark.rs`). Its problem is pure
  **throughput**: batching, launch consolidation, screening, and tuning.
- **def2-TZVP** contains quartet classes above nroots 5 — oxygen reaches f functions
  (`def2_tzvp_oxygen_reaches_f_functions`, `crates/cintx-basis/src/catalog.rs`), and an
  `(ff|ff)` class needs nroots 7. Its problem is **coverage first, then throughput**:
  every TZVP quartet must run on the device via the extended Rys path before TZVP
  throughput numbers mean anything, and the extended path is where the precision risk
  concentrates.

Reference workload sizes (pinned in `crates/cintx-basis/src/raw.rs` tests):
H2O/def2-SVP = 12 shells / 24 spherical AOs; H2O/def2-TZVP = 19 shells / 43 spherical AOs.

## 2. Current state (evidence, not aspiration)

### 2.1 Coverage — the extended device Rys path

- `crates/cintx-cubecl/src/device_rys_ceiling.rs`: `device_nroots_ceiling` returns
  `EXTENDED_DEVICE_NROOTS = 12` only when **all three** hold: the
  `extended-device-rys` cargo feature is on, the family's `runs_extended_rys()` flag is
  set, and `probe_fma_fusion` verified a true fused multiply-add on the backend (the
  precondition of the double-double Wheeler path).
- Families flipped so far: `Int3c2e`, `Int2e`, `Int2c2e`, `Int1e`. **Not flipped:**
  `Int3c2eDeriv`, `Int1eDeriv` — TZVP gradients and derivative rows still fall back to
  the nroots ≤ 5 ceiling.
- `def2_tzvp_exceeds_the_device_envelope` proves and reports the fraction of H2O/TZVP
  quartets above nroots 5.

### 2.2 Throughput

- Phase 36's `def2_throughput_benchmark.rs` measures the honest unit of work: the
  whole screened shell-quartet list, same Schwarz table for both engines, values
  compared before timing, coverage gaps counted.
- Batch paths exist end to end (`run_2e_batches`, `run_1e_batches`,
  `evaluate_2e_quartet_batch`, safe `PairBatchRequest`/`QuartetBatchRequest`), and the
  driver provides `build_schwarz_table`, `screen_quartets`, `bucket_quartets`, and
  `LaunchTier` classification.
- Phase 6 autotuning (`crates/cintx-cubecl/src/tuning.rs`) landed 2026-08-30, tunes
  **launch geometry only**, and is **off by default** on the evidence: on the 16-core
  CPU dev host a 4096-quartet batch measured 75.7/81.5 ms under `off` vs 88.8/273 ms
  under `balanced`. Open per its module docs: the remaining candidate dimensions
  (kernel mode, vectorization factor, scratch/shared-memory tile, fused vs staged
  transform), the derivative and σ-family launchers, prewarming, and the bench-report
  autotune artifact (currently `not_collected`).

### 2.3 Precision

- Unified oracle tolerance `atol = rtol = 1e-12`; `xtask`'s `run_error_budget` measures
  per-fixture max abs/rel error and **headroom**, exports artifacts to `/mnt/data`,
  `/tmp/cintx_artifacts`, and `artifacts/`, and enforces a reviewed ratchet via
  `--check-headroom` against a recorded baseline.
- The small-x Rys tables (`tools/generate_rys_smallx_tables.py` →
  `rys_smallx_data.rs`) reproduce libcint's exact `x <= 3e-7` branch for nroots 1..=12.
- `prim_tol` defaults to 0 with a tolerance-zero identity gate: only terms that
  underflowed to exactly zero are skipped.

## 3. Gates

Every task below inherits the general plan's correctness gates (§4.1 there). Added,
basis-specific:

- **G1 (SVP coverage)**: `bucket_quartets` over any def2-SVP molecule produces no
  bucket with `nroots > 5` and no `CubePerQuartetGlobal` tier — the existing envelope
  test, kept green.
- **G2 (TZVP coverage)**: with `extended-device-rys` on and FMA verified, **zero**
  quartets of a def2-TZVP work list are refused or routed to a host fallback, for the
  scalar families and (after D1) the derivative families.
- **G3 (speed)**: batched cintx beats single-threaded libcint 6.1.3 on the screened
  def2-SVP quartet list of the benchmark fixture set on at least one verified backend;
  for def2-TZVP, record the crossover and the nroots>5 class cost separately rather
  than assuming one number.
- **G4 (precision)**: the error-budget ratchet never regresses; TZVP high-root classes
  (nroots 6 and 7) get their own recorded budget entries with headroom ≥ 10× at
  1e-12 before extended-Rys becomes a default.
- **G5 (honesty)**: every speed number comes from the def2 throughput benchmark's
  rules — same screened work list, values compared before timing, coverage printed.

## 4. Workstreams

### D0 — def2 baselines (prerequisite for every speed claim)

Files: `crates/cintx-oracle/tests/def2_throughput_benchmark.rs`,
`xtask/src/bench_report.rs`.

1. Extend the benchmark fixture set beyond H2O: add at least one second-row molecule
   with d/f-heavy classes (e.g. SO2 or a Cl-containing molecule) in both bases, so the
   nroots 6–7 buckets carry real weight in TZVP timings.
2. Emit the benchmark rows as machine-readable artifacts
   (`cintx_def2_throughput.json` under `/tmp/cintx_artifacts` and `/mnt/data`):
   per-bucket class, quartet count, screened/unscreened split, engine times, match
   status, and coverage.
3. Record the current per-bucket split of TZVP work above nroots 5 (the number the
   envelope test prints) as a baseline field, so D1's coverage progress is measurable.

Exit: baseline artifact exists for both bases with per-class rows; no timing accepted
outside it.

### D1 — TZVP coverage: finish the extended-Rys flip

Files: `crates/cintx-cubecl/src/device_rys_ceiling.rs`, derivative-family launchers,
parity tests under `crates/cintx-oracle/tests/`.

1. Flip `Int3c2eDeriv` onto `runs_extended_rys` in the same commit as its parity gate
   (matching the established one-flip-one-gate pattern), then `Int1eDeriv`.
2. Add a def2-TZVP derivative-family parity fixture (nuclear gradient rows over the
   H2O/TZVP shell list) to the oracle suite at 1e-12.
3. Decide the default: once G2 and G4 hold on the CPU backend and at least one GPU
   backend, promote `extended-device-rys` from opt-in feature to default-on, keeping
   the FMA probe as the runtime guard. If the probe fails on a backend, the ceiling
   stays 5 and TZVP coverage is reported (not silently degraded) — extend the
   benchmark's coverage counter to say *why*.

Exit: G2 for all families; the envelope test's "exceeds" assertion is complemented by
a test that the same buckets are device-eligible under the extended ceiling.

### D2 — SVP/TZVP throughput on the batch path

Files: `cintx-driver`, `crates/cintx-cubecl/src/kernels/two_electron.rs`,
`crates/cintx-rs/src/api.rs` batch surfaces.

1. **Specialization prewarm.** def2 produces a small, enumerable set of launch classes
   (SVP: l ≤ 2 tuples; TZVP adds f). Add an optional prewarm that JIT-compiles the
   class set for a basis before the first timed batch, so cold JIT (measured at tens
   of seconds for 8 classes on the dev host) leaves the hot path. Report cold/warm
   separately per the general plan.
2. **Bucket consolidation.** Verify one launch per (class, chunk) on the def2 work
   lists via `ExecutionStats.kernel_launch_count`, and close any per-quartet residue
   in the driver's `run_buckets` route.
3. **Screening.** Keep Schwarz screening exact-bound-based and shared between engines;
   document the screened fraction per basis in the artifact. No approximate threshold
   in strict mode (general plan constraint stands).
4. **Contraction shape.** def2 is segmented but TZVP's tight s-blocks are
   deeper-contracted; measure the `nctr`/`nprim` mix per bucket and, if a class is
   contraction-bound, prefer the cooperative (`per_unit == 0`) kernel arm for it —
   under tuning, not hardcoding.

Exit: G3 for def2-SVP; TZVP throughput and crossover recorded per class in the
artifact.

### D3 — autotune the def2 workloads where measurement is trustworthy

Files: `crates/cintx-cubecl/src/tuning.rs`, launchers, `xtask/src/bench_report.rs`.

1. Run `balanced` tuning on a GPU backend (ROCm/WGPU as available) over the def2
   benchmark work lists, where ranking uses device timestamps instead of host wall
   clock — the condition the module docs name for turning it on.
2. Extend tuning to the derivative and σ-family launchers (module is family-generic;
   they still use `one_e_launch_geometry` untuned).
3. Add the next candidate dimension only with def2 evidence: vectorization factor
   first (homogeneous SVP buckets are the best case), then scratch tile. Bump
   `TUNING_SCHEMA_VERSION` on any change.
4. Wire the autotune cache into `xtask bench-report` so `cintx_cubecl_autotune.json`
   stops being `not_collected`.

Exit: Phase 6 exit gates hold on the def2 work lists on at least one GPU backend, or
the measured no-win is documented per backend the way the CPU result was.

### D4 — precision: budget the TZVP high-root envelope

Files: `xtask/src/error_budget.rs`, oracle fixture inputs, `math/` Rys entries.

1. Add def2-TZVP-shaped fixtures to `OracleRawInputs::sample` (or a sibling sampler)
   so the error budget contains `nroots_class` entries for 6 and 7 — today's budget is
   blind exactly where TZVP is new. Include tight-exponent pairs that exercise the
   small-x (`x <= 3e-7`) branch and large-x asymptotics of the extended entry.
2. Record the reviewed baseline (`xtask error-budget --record`) after review, and add
   `--check-headroom` for the def2 profiles to the nightly gate.
3. Add a range-separated row: `env[PTR_RANGE_OMEGA]` combined with nroots > 5 is the
   newest × newest interaction and gets its own budget entry.
4. Document the FMA probe result per backend in the unverified-matrix artifact: a
   backend without fused FMA runs TZVP at ceiling 5 by design, and that is a reported
   verification status, not a bug.

Exit: G4; budget entries exist for nroots 6/7 scalar + derivative + range-separated
classes with recorded headroom.

### D5 — release wiring

1. Promote the def2 throughput benchmark from `--ignored` to a nightly job on pinned
   runners; PR CI keeps only the cheap envelope/coverage tests.
2. Final artifacts under `/tmp/cintx_artifacts` and `/mnt/data`:
   `cintx_def2_throughput.json`, updated `cintx_precision_budget.current.json`,
   `cintx_cubecl_autotune.json`, and the unverified matrix with per-backend FMA/
   extended-Rys status.
3. CHANGELOG entries per workstream, in the established evidence-first style.

## 5. Ordering and dependencies

```text
D0 (baselines) ──────────────┬─→ D2 (SVP throughput) ─→ D3 (autotune)
                             │
D1 (TZVP coverage) ─→ D4 (TZVP precision budget) ─┴─→ D5 (release)
```

D1 and D2 can proceed in parallel after D0; D3 needs D2's consolidated batch path and
a GPU runner; the `extended-device-rys` default flip inside D1 waits on D4's recorded
headroom.

## 6. Risks

| Risk | Control |
|---|---|
| Extended Rys (nroots 6–12) has thinner precision headroom than the polynomial fits | D4 budgets it explicitly before any default flip; ratchet blocks regression. |
| Backend without true FMA silently degrades TZVP | Ceiling stays 5 by design; D1.3/D4.4 make it a *reported* status in coverage and the unverified matrix. |
| Prewarm JIT cost attributed to steady state | Cold/warm split is already a general-plan gate; D0 artifact carries both. |
| Tuning noise on CPU hosts repeats the Phase 6 no-win | D3 only claims wins from device-timestamp profiles; CPU default stays `off`. |
| Benchmark fixture too small (H2O) to load nroots 6–7 buckets | D0.1 adds heavier molecules before TZVP numbers are quoted. |
| Derivative-family flip changes results | One-flip-one-gate pattern: the parity fixture lands in the same commit as the flag. |
