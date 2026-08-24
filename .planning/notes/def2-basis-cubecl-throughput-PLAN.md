# def2-SVP / def2-TZVP on CubeCL — Correctness + Throughput Plan

**Status**: Draft plan (not yet a roadmap phase)
**Proposed milestone**: v1.5 — Production Basis Sets & Throughput Leadership
**Proposed phases**: 32–36 (roadmap currently ends at Phase 31 / v1.4)
**Date**: 2026-08-23

---

## 0. Executive summary

Two independent things are being asked for, and they have very different difficulty:

| Ask | Difficulty | Verdict |
|---|---|---|
| Evaluate def2-SVP integrals via CubeCL | Low — the envelope already fits | Mostly missing *basis-set data*, not kernels |
| Evaluate def2-TZVP integrals via CubeCL | Medium — needs device Rys nroots 6–12 | Blocked by `MAX_DEVICE_NROOTS = 5` |
| Beat libcint on speed | High — architectural | **Requires replacing per-shell-tuple dispatch with a batched driver.** Not reachable by tuning the current kernels. |

The single most important finding: **cintx's CubeCL path is currently 30×–194× *slower* than
libcint**, and this is architectural, not a tuning gap. Every `eval_raw` call launches one kernel
with `CubeCount::Static(1, 1, 1)` — a single workgroup — re-uploads the shell's exponents and
coefficients, and blocks on a readback. Recorded numbers (`artifacts/speed_benchmark_report.md`,
H₂O/STO-3G):

| Family | CubeCL | libcint | Ratio |
|---|---|---|---|
| `1e_ovlp` | 29.75 µs | 0.186 µs | **160× slower** |
| `2c2e` | 26.07 µs | 0.201 µs | **130× slower** |
| `3c2e` | 34.01 µs | 0.837 µs | **41× slower** |
| `2e` (ERI) | 1124.08 µs | 5.802 µs | **194× slower** |

No amount of kernel-body optimization closes a 194× gap when ~25 µs of it is fixed
launch/marshal/readback overhead per quartet. libcint's per-quartet C is already near-optimal on
CPU; a per-quartet GPU dispatch structurally cannot win. **The unit of work must change from one
shell tuple to one bucket of thousands of shell tuples.**

The good news: the scaffolding for exactly that already exists and is only wired to a toy pilot.

---

## 1. Ground truth — what exists today

### 1.1 There is no basis-set data or loader anywhere

`grep` for `def2|TZVP|SVP` across `crates/` returns **zero hits in code** (only planning prose).
Every test builds `atm`/`bas`/`env` by hand (e.g. `build_h2o_sto3g` at
`crates/cintx-oracle/tests/benchmark_speed.rs:26`) or constructs `BasisSet::try_new`
(`crates/cintx-core/src/basis.rs:61`) shell-by-shell. There is no parser, no embedded table, no
normalization helper. **This is the largest single chunk of new code in the plan and it is
ordinary, low-risk work.**

### 1.2 Kernels already handle what def2 needs, structurally

- **def2-SVP and def2-TZVP are fully *segmented*: `nctr == 1` on every block.**
  (Correction to an earlier draft, established by parsing the vendored BSE data: zero blocks with
  more than one coefficient column in either basis. The 1 280 multi-column rows an initial `awk`
  count found were def2-ECP rows, whose columns are `r`-power / exponent / coefficient, not
  contraction columns.) General contraction is therefore **not** on the def2 critical path.
- **A suspected general-contraction bug was investigated and refuted.** Source inspection
  suggested a defect: the device 2e kernel indexes contraction coefficients as
  `coeff_i[pi * nctr_i + ci]` (`two_electron.rs:1229`) while the host fallback for the same
  uploaded buffer uses `coeff_i[ci * n_prim_i + pi]` (`two_electron.rs:4470`, the libcint `env`
  layout), with no visible transpose in `run_2e_scalar_device`. A dedicated reproducer with
  `nprim = 3, nctr = 2` and asymmetric coefficients — `crates/cintx-oracle/tests/
  general_contraction_device_indexing.rs` — **matches vendored libcint exactly**, so the two
  indexings are reconciled somewhere the reading missed. The test is kept as a general-contraction
  regression guard. Lesson recorded here because the inspection argument was superficially
  convincing and wrong: the measurement, not the reading, settled it.
- Arbitrary `nprim` is already looped on device.
- ECP (`canonical_family == "ecp"`, `ECP_LMAX = 5`) exists — needed for def2 at Z ≥ 37.
- Cart↔sph fold invariance is proven, so the spherical def2 path rides the existing `c2s`.

### 1.3 The angular-momentum / Rys envelope is the real def2 gate

2e Rys order is `nroots = (li+lj+lk+ll)/2 + 1` (`two_electron.rs:110`). The device is capped at
`MAX_DEVICE_NROOTS = 5` (`center_2c2e.rs:70`); above that the code falls into a **serial host
loop** over every primitive quartet (`two_electron.rs:4441`+), with a hard ceiling of
`HOST_RYS_NROOTS_CEILING = 12` (`center_2c2e.rs:75`).

| Basis | Max main-group `l` | Worst quartet | `nroots` | Device today? |
|---|---|---|---|---|
| def2-SVP | 2 (d) | (d d\|d d) | **5** | ✅ exactly at the cap |
| def2-TZVP | 3 (f) | (f f\|f f) | **7** | ❌ host serial fallback |
| def2-TZVP (3d TM) | 4 (g) | (g g\|g g) | **9** | ❌ host serial fallback |

So **def2-SVP fits the existing device envelope exactly**, and **def2-TZVP does not**.

Device-side Wheeler/Jacobi routines for nroots 6-12 exist as `#[cube]` functions in
`crates/cintx-cubecl/src/math/rys_wheeler.rs` (`cint_polynomial_roots_dev:1913`,
`wheeler_recursion_dev:1295`, `hessenberg_qr_dev:1731`, double-double helpers `dd_*:2131`+).

**Correction to an earlier draft: they are not unwired — they are wired in the worst possible
place.** `rys_roots_host_wheeler` (`rys_wheeler.rs:3175`) already dispatches nroots 8-12 to
`rys_jacobi_device` / `lrys_schmidt_device` / `lrys_laguerre_device`. But each of those host
wrappers **creates a `CpuRuntime` client and launches a kernel for a single `(nroots, x)` root
evaluation** (`rys_jacobi_device:1404` — `let client = cubecl::cpu::CpuRuntime::client(...)`
inside the function body). `fill_g_tensor_2e` calls it once per primitive quartet, so a single
def2-TZVP transition-metal `(g g|g g)` shell quartet with 17 s-primitives would trigger on the
order of 17^4 ~= 84 000 kernel launches. This is a far more severe throughput defect than a
missing device path, and it is on the production dispatch today.

nroots 6 and 7 — the def2-TZVP main-group f range — deliberately stay on **pure host** code, per a
documented parity decision (`rys_wheeler.rs:3188-3198`): routing them through a device launch
perturbed the subsequent host g-tensor accumulation by ~1e-11 and tripped the `hess2e`
flat-atol=1e-12 gate. That decision is sound for a per-quartet launch model and should be
revisited only once roots are computed *inline* in the kernel rather than by a separate launch.

So Phase 33 is **not** "wire the device path". It is: replace per-evaluation launches with an
inline `#[cube] rys_roots_dev(...)` called from inside the g-tensor kernel, with the Wheeler
scratch (`fmt_ints` 64, `cs` (nroots+1)^2 <= 169, `rt` 12, `acomp` 144, `vscr` 32, `flag` 1 —
about 3.4 KB of f64 per work-item) carried in shared memory. Note the existing device kernels take
`#[comptime] nroots`, which specializes one kernel per Rys order — exactly what the Phase 34
per-l-class bucketing already provides, and a reason to do 33 and 34 together rather than in
sequence.

### 1.4 Batching scaffolding exists but serves only a toy pilot

Already built:

- `BatchExecutionPlan::build(items, max_items_per_chunk, max_chunk_bytes)` and `KernelClass`
  (family, representation, precision, arity, `angular_momenta`, `nroots`, `component_rank`) —
  `crates/cintx-rs/src/api.rs:267`+. This is precisely the right bucketing key.
- `crates/cintx-cubecl/src/batch_pilot.rs` — a **verified** grid-stride batched launch with a
  retained output-staging arena keyed by `BackendIntent`, reporting
  `submit_ns`/`readback_ns`/`output_staging_reuses`.
- `crates/cintx-cubecl/src/resident_cache.rs` — `DeviceResidentCache` keyed by a stable
  IEEE-exact `basis_hash`, for device-resident basis metadata.
- `crates/cintx-cubecl/src/shared_memory.rs` — per-family shared-memory layout calculators
  (`calc_2e_layout:320`, `validate_shared_layout_bounds:576`,
  `generate_layout_catalog:601`).

The gap: the pilot covers **only single-contraction Cartesian s-s overlap/kinetic and primitive
(ss|ss) ERI** (its own module docs say so). Everything else in `evaluate_batch_in` falls through
to a per-item loop — visible at `api.rs:361`, where `kernel_launch_count` is summed *per item*.

**Conclusion**: the plan is not "invent batching"; it is "generalize the proven pilot to real
angular momentum and contraction, then make def2 the driving workload."

---

## 2. What def2-SVP / def2-TZVP actually demand

### 2.1 Composition (Weigend & Ahlrichs 2005)

| Element | def2-SVP | def2-TZVP |
|---|---|---|
| H | (4s,1p) → [2s,1p] | (5s,1p) → [3s,1p] |
| C/N/O/F | (7s,4p,1d) → [3s,2p,1d] | (11s,6p,2d,1f) → [5s,3p,2d,1f] |
| 3d TM | max l = 3 (f) | max l = 4 (g) |
| Z ≥ 37 | def2-ECP | def2-ECP |

Water reference workloads this plan will benchmark against:

| Basis | Shells | Spherical AOs | Unique 8-fold quartets |
|---|---|---|---|
| H₂O / def2-SVP | 12 | 24 | ≈ 3.1 k |
| H₂O / def2-TZVP | 19 | 43 | ≈ 18 k |

That quartet count is the point: **~3 000 and ~18 000 independent work items per Fock build**, all
available at once. That is the parallelism the current architecture throws away by dispatching
them one at a time.

### 2.2 g-tensor size drives the launch shape

From `build_2e_shape` (`two_electron.rs:109`), `g_size = nroots · dli · dlk · dll · dlj`, and the
kernel needs `3 · g_size` doubles (x/y/z):

| Quartet | `nroots` | `g_size` | g-tensor bytes | Placement |
|---|---|---|---|---|
| (s s\|s s) | 1 | 1 | 24 B | registers |
| (p p\|p p) | 3 | 108 | 2.6 KB | registers / local |
| (d d\|p p) | 4 | 360 | 8.6 KB | shared |
| (d d\|d d) — **SVP ceiling** | 5 | 1 125 | **26.4 KB** | shared |
| (f f\|f f) — **TZVP ceiling** | 7 | 5 488 | **128.6 KB** | ❌ exceeds shared → global scratch |
| (g g\|g g) — TZVP TM | 9 | 24 300 | 570 KB | global scratch |

This is a hard, quantitative design constraint and it dictates a **three-tier launch strategy**
(§3.3). It also explains why "one thread per quartet" — the shape the s-s pilot uses — cannot be
extended naively: at dddd each thread would need 26 KB of private storage.

---

## 3. The plan

### Phase 32 — Basis-set library (`cintx-basis`)

**Goal**: `BasisSet` and raw `atm`/`bas`/`env` for def2-SVP and def2-TZVP, byte-compatible with
what PySCF hands libcint.

New crate `crates/cintx-basis`:

| File | Contents |
|---|---|
| `src/format.rs` | NWChem + Turbomole basis-text parser → `(element, l, [(exp, coeff…)])` |
| `data/def2-svp.nwchem`, `data/def2-tzvp.nwchem`, `data/def2-ecp.nwchem` | Embedded via `include_str!` (Basis Set Exchange, unmodified, with provenance header) |
| `src/normalize.rs` | `gto_norm(l, exp)` — libcint `CINTgto_norm` — and the contracted-block renormalization PySCF applies |
| `src/build.rs` | `BasisSet::try_new_with_ecp` construction; segmented-vs-general contraction policy |
| `src/raw.rs` | Emitter for `atm`/`bas`/`env` in `cintx-compat::raw` slot layout |

**The one place this goes wrong**: normalization. libcint expects coefficients **pre-multiplied by
`CINTgto_norm(l, exp)`**, and PySCF additionally renormalizes each contracted column so the
self-overlap is 1. If cintx and the vendor disagree here, *every* oracle comparison fails with a
plausible-looking constant factor. Task 32-03 is a dedicated normalization parity test that
compares `env` coefficient arrays against a PySCF-generated fixture **before any integral runs**.

**Tasks**
- 32-01 Parser + embedded def2-SVP/TZVP/ECP tables; round-trip test.
- 32-02 `BasisSet` builder incl. ECP shells for Z ≥ 37.
- 32-03 **Normalization parity gate** — `env` coefficients byte-identical to PySCF fixture.
- 32-04 Raw `atm`/`bas`/`env` emitter + shell-ordering parity vs PySCF `mol._bas`.
- 32-05 Fixtures: H₂O, CH₄, benzene, ferrocene (TM + g), [AuCl₄]⁻ (ECP), each × {SVP, TZVP}.

**Gate**: for every fixture, `atm`/`bas`/`env` match a PySCF-generated reference exactly.

---

### Phase 33 — Envelope extension: device Rys nroots 6–12

**Goal**: def2-TZVP f/g quartets execute **on device** instead of falling into the host serial loop.

`rys_wheeler.rs` already has the `#[cube]` device implementations; they are unwired.

**Tasks**
- 33-01 Make `cint_polynomial_roots_dev` and its helper chain `pub(crate)`; add a device entry
  `rys_roots_dev(nroots, x, &mut u, &mut w)` dispatching 1–5 → existing polynomial fits, 6–12 →
  Wheeler.
- 33-02 Size the Wheeler scratch (Jacobi moments, Hessenberg matrix, dd accumulators) through
  `shared_memory::calc_math_layout` and register it in the layout catalog; the QR/Hessenberg
  arrays are `O(nroots²)` and must be bounds-validated at nroots = 12.
- 33-03 Raise `MAX_DEVICE_NROOTS` 5 → 12 **behind a per-family flag**, family by family, each
  gated on its own oracle parity test. Do not raise it globally in one commit.
- 33-04 Accuracy gate: device vs `rys_roots_host_wheeler` over a log-spaced `x` sweep
  (1e-8 … 1e6) × nroots 6..12, to the existing oracle tolerance.
- 33-05 Numerical-stability check for the double-double path on GPU backends — verify the compiler
  does not contract `two_sum`/`two_prod` into FMA and destroy the error-free transform. **This is
  the highest-risk item in the phase**; if a backend miscompiles it, gate that backend to nroots ≤ 5.

**Gate**: (f f|f f) and (g g|g g) def2-TZVP quartets match vendored libcint on device, and
`ExecutionStats.fallback_reason` is `None`.

---

### Phase 34 — Batched multi-quartet driver (**the speed phase**)

**Goal**: replace per-shell-tuple dispatch with one launch per `KernelClass` bucket. This is where
the libcint win comes from.

#### 34.1 Work-list construction (host)

1. Enumerate unique shell quartets under 8-fold permutational symmetry.
2. **Schwarz prescreen**: precompute `Q_ij = sqrt((ij|ij))` per shell pair (one batched 2-index
   pass), keep quartets with `Q_ij · Q_kl > τ` (default `τ = 1e-10`). On real def2-TZVP molecules
   this removes >90 % of quartets.
3. Bucket surviving quartets by `KernelClass` — **keyed on the l-quartet only**, not on
   `(l, nprim, nctr)`. Keying on nprim would shatter def2 into hundreds of near-empty buckets
   (O alone has ≥4 distinct shell signatures → 4⁴ combinations). Instead pass per-quartet
   `nprim`/`nctr` as descriptor data with dynamic loop bounds, and **sort within each bucket by
   total primitive count** so neighbouring work-items in a plane have similar trip counts. This
   trades a little intra-plane divergence for a ~10× reduction in launch count — the right trade.
4. Reuse `BatchExecutionPlan::build` for chunking against the memory limit.

#### 34.2 Device-resident basis (kills the dominant per-call cost)

Today `launch_two_electron_typed` does `shell_i.exponents[..n_prim_i].to_vec()` and a fresh
`client.create_from_slice` **on every quartet** (`two_electron.rs:4433`+). Extend
`DeviceResidentCache` to retain, keyed by `basis_hash`:

- flattened exponents, coefficients, and their per-shell offsets
- atom coordinates and per-shell atom indices
- shell AO offsets / counts (already in `ResidentMetadata`)

The per-launch upload then shrinks to a compact `u32` quartet-descriptor table (4 shell ids +
output offset per quartet) — from O(nprim·nctr) doubles per quartet to 20 bytes per quartet.

#### 34.3 Three-tier launch shape (from §2.2)

| Tier | g-tensor | Shape | Covers |
|---|---|---|---|
| **A — thread-per-quartet** | ≤ ~4 KB, private | `cube_count_1d(ceil(N/plane))`, grid-stride, plane-aligned `CubeDim` | ssss … pppp |
| **B — cube-per-quartet, shared g** | 4–48 KB | one cube per quartet; plane cooperates over the (root × cartesian) index space of VRR/HRR; g in `SharedMemory` via `calc_2e_layout` | up to (d d\|d d) — **all of def2-SVP** |
| **C — cube-per-quartet, global scratch g** | > 48 KB | g-tensor in a device workspace slab, `N_cubes × 3·g_size` doubles, allocated once per bucket | f/g quartets — **def2-TZVP** |

Tier C is bandwidth-bound; a follow-on optimization (§34.6) tiles the outer `(dlj, dll)` axes so
only a `nroots·dli·dlk·3` slab (2.7 KB at ffff) lives in shared memory. Ship Tier C simple first,
optimize second.

#### 34.4 Primitive-level screening (device)

def2-TZVP carbon has 11 s-primitives → up to 11⁴ ≈ 14 600 primitive quartets per shell quartet.
Skip a primitive pair when the pair prefactor `exp(-a_i a_j |R_ij|²/(a_i+a_j))` is below tolerance.
Precompute surviving primitive-pair lists per shell pair on the host, store device-resident, and
iterate the compacted list instead of the dense 4-deep loop. On def2-TZVP this is typically a
3–10× reduction in inner work.

#### 34.5 Output and readback

- One collective `client.read` per chunk, not per quartet (the pilot already proves this shape and
  instruments `readback_ns`).
- Reuse the pilot's retained output-staging arena; report `output_staging_reuses` so a regression
  in allocation churn is visible.
- Provide a **device-resident output mode** where the AO block stays on device for a downstream
  consumer (Fock build). For a 30-atom def2-TZVP system (~700 AO) the dense ERI tensor is ~1.9 TB —
  materializing it to host is not a real workflow, and benchmarking against libcint while paying a
  full host writeback would be measuring the wrong thing.

#### 34.6 Public API

Add to `cintx-rs`, alongside the existing per-request surface:

```rust
pub struct QuartetBatchRequest<'basis> { /* operator, representation, basis, screening τ, options */ }
pub fn evaluate_shell_quartets(req: QuartetBatchRequest<'_>) -> Result<QuartetBatchOutput, FacadeError>;
```

`QuartetBatchOutput` carries the AO blocks plus `BatchExecutionStats` (launch count, readback
count, transfer bytes, pack/submit/readback ns) so a claimed speedup is auditable.

**Tasks**
- 34-01 Work-list enumeration + 8-fold symmetry dedup.
- 34-02 Schwarz `Q_ij` batched pass + quartet screening.
- 34-03 `KernelClass` bucketing by l-quartet + intra-bucket sort by primitive count.
- 34-04 `DeviceResidentCache` extension to full basis payload.
- 34-05 Tier A launcher (generalize `batch_pilot` grid-stride to arbitrary low-l).
- 34-06 Tier B launcher (cube-per-quartet, shared-memory g-tensor) — **def2-SVP complete here**.
- 34-07 Tier C launcher (global-scratch g-tensor) — **def2-TZVP complete here**.
- 34-08 Primitive-pair screening lists.
- 34-09 Collective readback + device-resident output mode.
- 34-10 `evaluate_shell_quartets` public API + `BatchExecutionStats` plumbing.

**Gate**: every batched result is **byte-identical** to the existing per-quartet path on all
Phase-32 fixtures. Screening tolerance `τ = 0` must reproduce the unscreened result exactly —
this is the test that catches a screening bug masquerading as a speedup.

---

### Phase 35 — 1e / 2c2e / 3c2e batched paths

The same three-tier treatment applied to overlap/kinetic/nuclear (shell **pairs**, ~144 for
def2-SVP water, ~361 for TZVP) and to the 2c2e/3c2e auxiliary families used by RI-J/RI-MP2 — which
is what def2 basis sets are overwhelmingly used with in practice.

3c2e is the highest-value target after 2e: RI-J with def2/J auxiliary bases is the dominant cost in
most def2 workflows, and its work list (`nshell² × naux`) is even more uniform than 2e's, so it
buckets almost perfectly.

**Gate**: 1e families at ≥ 10× the current CubeCL throughput and byte-identical to vendor.

---

### Phase 36 — Benchmark, gates, and the actual libcint comparison

**Tasks**
- 36-01 Extend `crates/cintx-oracle/tests/benchmark_speed.rs` from hardcoded H₂O/STO-3G to the
  Phase-32 fixture set × {def2-SVP, def2-TZVP}.
- 36-02 Add a **whole-workload** benchmark axis: total wall-clock to evaluate the full screened
  shell-quartet list, for (a) cintx batched CubeCL, (b) cintx-simd, (c) libcint loop.
- 36-03 Report per-backend (cpu / wgpu / cuda / rocm), with `ExecutionStats.fallback_reason`
  asserted `None` — a "win" achieved by silently falling back to the host loop is not a win.
- 36-04 CI regression gate on throughput, with recorded baselines in `artifacts/`.
- 36-05 Rewrite `artifacts/speed_benchmark_report.md` with the new methodology.

#### Benchmark honesty rules

1. **Same work on both sides.** Apply identical Schwarz screening to the libcint loop, or report
   screened and unscreened separately. Screening is an algorithmic win, not a kernel win, and
   attributing it to CubeCL would be dishonest.
2. **Report the readback.** Device-resident output is a legitimate mode, but a host-materialized
   number must be reported alongside it.
3. **State the thread count.** libcint's per-quartet C is single-threaded; a GPU comparison must
   say so, and should also report libcint under OpenMP across all cores as the honest CPU ceiling.
4. **f64 rate matters.** Report the backend's f64:f32 throughput ratio.

---

## 4. Targets — what "win speed libcint" concretely means

| Workload | libcint (1 core) | Target |
|---|---|---|
| H₂O/def2-SVP full ERI list (~3.1 k quartets) | baseline | ≥ 5× on CUDA/ROCm; **≥ 1× (parity) on CPU backend** |
| H₂O/def2-TZVP full ERI list (~18 k quartets) | baseline | ≥ 10× on CUDA/ROCm |
| Benzene/def2-TZVP screened list | baseline | ≥ 20× on CUDA/ROCm |

### The caveat that must not be buried

**On the CPU backend, CubeCL will probably never beat libcint, and should not be expected to.**
It is the same silicon, and libcint's per-quartet C is already close to optimal — `cintx-simd`
already sits at 0.86×–1.30× and *that* is the right CPU comparator (it beats libcint on `1e_kin`
and `2c2e` today). The credible CubeCL win is on **GPU backends**, from throughput across
thousands of concurrent quartets. A realistic CPU-backend goal is "reach parity with libcint after
batching removes the 25 µs/quartet overhead" — roughly a 100× improvement over today, landing at
1× rather than 194× slower.

Setting the goal as "CubeCL CPU beats libcint" would be setting up a target the architecture
cannot meet.

---

## 5. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **f64 on GPU is slow** — consumer NVIDIA runs f64 at 1/32–1/64 rate; wgpu `SHADER_F64` is often absent | Could erase the entire win | Executor already gates `SHADER_F64` (`check_shader_f64_in_features`). Target CUDA/ROCm on datacenter parts; evaluate f32 accumulation with f64 refinement for screened-small contributions, gated on oracle tolerance. **Measure before committing to Tier C on wgpu.** |
| **Double-double miscompilation** — FMA contraction destroys `two_sum`/`two_prod` in the device Wheeler path | Silent accuracy loss at nroots ≥ 6 | Task 33-05; per-backend gate to nroots ≤ 5 if it fails |
| **Normalization mismatch** vs PySCF/libcint | Every def2 oracle test fails plausibly | Task 32-03 gates on `env` bytes *before* any integral runs |
| **Bucket fragmentation** if `KernelClass` keys on nprim | Hundreds of tiny launches; no speedup | Key on l-quartet only; nprim as data (§34.1) |
| **Tier C bandwidth bound** — f/g quartets stream a 128–570 KB g-tensor through global memory | def2-TZVP win much smaller than def2-SVP | Ship simple, then tile `(dlj, dll)` into shared memory (§34.6). Measure the roofline first. |
| **Screening bug looks like a speedup** | Wrong results reported as a win | `τ = 0` must reproduce unscreened byte-for-byte (Phase 34 gate) |
| **Manifest/lock drift** — new API surface | Release gate failure | Route `evaluate_shell_quartets` through the compiled manifest lock + `xtask manifest_audit` from the start |

---

## 6. Dependency order

```
Phase 32 (basis data) ──┬─→ Phase 34 (batched 2e) ──→ Phase 35 (1e/2c2e/3c2e) ──→ Phase 36 (bench)
                        │        ▲
Phase 33 (nroots 6–12) ─┴────────┘   (33 required only for def2-TZVP; def2-SVP ships on 32+34)
```

**Earliest demonstrable result**: Phase 32 + 34-01…34-06 gives a fully device-batched **def2-SVP**
ERI path — because def2-SVP's worst quartet (d d|d d) sits exactly at `nroots = 5` and its g-tensor
(26.4 KB) fits shared memory. That is the natural first milestone and it needs none of Phase 33.

---

## 6b. Implementation status

| Phase | Status | Evidence |
|---|---|---|
| **32 — basis library** | **Complete** | `crates/cintx-basis` (23 unit tests) + `crates/cintx-oracle/tests/def2_normalization_parity.rs` (5 vendor-gated tests) |
| **33 — device Rys 6-12** | **Re-scoped, not implemented** | See the corrected §1.3: the device path is already wired but launches per root evaluation. Reworking it means inline `#[cube]` roots inside the g-tensor kernel, which is coupled to Phase 34's per-class specialization. |
| **34 — batched driver** | **Host layer complete; batched kernel not implemented** | `crates/cintx-driver` (15 tests): work-list, Schwarz screening, l-class bucketing, tiering, execution + stats. Executes through a pluggable `QuartetEvaluator`; the fused multi-quartet kernel is not written. |
| **35 — 1e/2c2e/3c2e batched** | Not started | — |
| **36 — benchmark** | **Complete** | `crates/cintx-oracle/tests/def2_throughput_benchmark.rs` — whole-workload cintx vs libcint on the identical screened work-list |

### Headline outcome: a real correctness bug in def2-quality ERIs

Driving a full def2-SVP basis through a **class-complete** 2e sweep (69 angular-momentum launch
classes, one representative each, vs vendored libcint) found **3 classes producing wrong values**:
`[1,1,2,1]`, `[1,2,2,1]`, `[2,2,2,1]` — max abs error up to **11.68**.

Root cause: the device `kj2d` HRR branch (`ibase == 0 && kbase == 1`) bounded its second transfer
loop by `di` where libcint (`g2e.c:552`) and cintx's own host `hrr_kj2d_4d` both use `dk`. With
`ibase == 0`, `di == nroots` and `dk == nroots * (li + 1)`, so it silently under-wrote every
`i >= 1` plane of the G-tensor. Failure condition:
`ibase == false && kbase == true && li >= 1 && ll >= 1` — which predicts exactly those 3 classes
and none of the 66 that passed.

It survived because the branch's only device test is `(s,s,p,s)`: `li == 0`, where `dk == di`
makes the bug invisible, and `ll == 0`, where the loop never runs. Both guards had to be broken
at once.

**Fixed** (`di` -> `dk`), verified 69/69, and three in-crate regression tests added at
`(p,p,d,p)`, `(p,d,d,p)`, `(d,d,d,p)` to close the coverage gap.

### What was delivered

- **`crates/cintx-basis`** — def2-SVP / def2-TZVP / def2-ECP vendored from the Basis Set Exchange
  (v0.12, Turbomole 7.3 data), an NWChem parser handling both the orbital and ECP sections,
  libcint-exact two-stage normalization, `BasisSet` construction with ECP shells, and raw
  `atm`/`bas`/`env` emission. Verified against **vendored libcint**, not a recorded fixture: a
  correctly normalized contracted AO has unit self-overlap by construction, so the gate is that
  libcint itself reports `S_ii == 1` for every AO of every fixture. `gto_norm` is additionally
  compared directly against the vendor's own `CINTgto_norm` FFI.
- **`crates/cintx-driver`** — the batching layer the throughput work needs: canonical 8-fold
  quartet enumeration (with a test proving every one of the `nbas^4` orderings maps into the
  canonical set exactly once, and that degeneracies sum back to `nbas^4`), Cauchy-Schwarz
  screening with the `tolerance = 0` identity gate, l-class bucketing with the `g_size` formula
  mirrored from `build_2e_shape`, and the three-tier launch classification.
- **Benchmark** — measures the unit of work an SCF iteration needs (the whole screened quartet
  list), drives both engines through the *same* work-list, and refuses to print a speed verdict
  for a run that is incorrect or incomplete.

### What was not delivered, and why

- **The fused multi-quartet CubeCL kernel** (34-05 … 34-07). This is the piece that actually moves
  the number, and it is a substantial change to a 5 000-line kernel file whose correctness is
  gated by an oracle suite too slow to iterate against in this session. The driver was built so
  that kernel drops in behind `QuartetEvaluator` without reworking the layer above it.
- **Phase 33.** Re-scoped once §1.3 was corrected; it is now coupled to 34's per-class
  specialization and should be done with it.
- **Phase 35.** Depends on 34's kernel.

## 7. Open questions for the user

1. **Which backend is the speed target?** CUDA / ROCm / wgpu / CPU. This changes the f64 strategy
   and whether Tier C is viable at all.
2. **Is device-resident output acceptable** as the primary benchmark mode, or must results be
   host-materialized (which caps any GPU win at PCIe bandwidth)?
3. **Scope**: is def2-ECP (Z ≥ 37) in scope now, or main-group only for the first milestone?
4. **def2/J and def2/JK auxiliary bases** — in scope? They are what makes the 3c2e/2c2e work in
   Phase 35 worth doing, and are how def2 sets are actually used.
