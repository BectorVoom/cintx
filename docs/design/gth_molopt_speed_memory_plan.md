# GTH-MOLOPT (DZVP-MOLOPT-SR / TZVP-MOLOPT) Speed and Memory Plan

Status: executed 2026-09-06 — §8 is the record of what landed and what was measured;
§9 (S3), §10 (the profile-guided pass) and §11 (the GPU decomposition, G1, and
the T4 measurement package), all 2026-09-07, extend it; §12–§14 (V1, the root-axis
vector VRR, and what it does not apply to), 2026-09-08; §15 (F1, fusing the Rys
orders into one dispatch), §16 (F1 on the GPU, and the per-quartet ket-pair
split it needed), §17 (the split on the per-unit arm) and §18 (one method on both
backends) and §19 (the cart-to-sph transform moves on-device, and the host
Cartesian intermediate disappears), 2026-09-09.

§15.4's account of why the fusion cost time on the GPU is **superseded by §16.1**,
which measured it: one of the two mechanisms it named does nothing. §17 puts the
ket-pair split on both decompositions and §18 makes it — and every other choice
the 2e path makes — the same on CPU and GPU, and independent of how a work list
is batched.
Scope: the batched `int2e_sph` path over the two GTH-MOLOPT orbital bases
`cintx-basis` exposes behind the `gth` feature (`DZVP-MOLOPT-SR-GTH`,
`TZVP-MOLOPT-GTH`). `gth-tzvp-molopt-sr` does not exist upstream (CP2K ships
short-range variants at SZV and DZVP only), so `TZVP-MOLOPT-GTH` stands in for it,
as `crates/cintx-basis/data/gth/README.md` records.
Primary compatibility target: libcint 6.1.3, unified oracle tolerance `1e-12`.
Backends that matter: CubeCL `cpu` and `rocm` (gfx1151 on the dev host).
Measurement rules: those of `def2_speed_memory_optimization_plan.md` §10.8 — a
claim is a ratio measured inside one process, or an in-process A/B, never two
absolute times from two processes.

## 1. Purpose

Every def2 workload cintx has been tuned on is *segmented*: `max_nctr_product == 1`
in every bucket of every def2 work list (`def2_speed_memory_optimization_plan.md`
§2.2). The `nctr > 1` arm of the 2e kernel was therefore covered for correctness
(`general_contraction_device_indexing`, one `nprim = 3, nctr = 2` s shell) and never
once timed.

GTH-MOLOPT is the opposite shape. It is a *family* basis: every shell of an atom
shares one exponent set, and the s and p shells carry two (`DZVP-MOLOPT-SR`) or three
(`TZVP-MOLOPT`) contractions. From `BASIS_MOLOPT`:

| element | basis | nprim | nctr (s, p, d) | shells / atom | AOs / atom |
|---|---|---|---|---|---|
| H | DZVP-MOLOPT-SR | 5 | 2, 1 | 2 | 5 |
| C, N, O | DZVP-MOLOPT-SR | 5 | 2, 2, 1 | 3 | 13 |
| S | DZVP-MOLOPT-SR | 4 | 2, 2, 1 | 3 | 13 |
| H | TZVP-MOLOPT | 7 | 3, 1 | 2 | 6 |
| C, N, O | TZVP-MOLOPT | 7 | 3, 3, 1 | 3 | 17 |
| S | TZVP-MOLOPT | 6 | 3, 3, 1 | 3 | 17 |

So a TZVP `(pp|pp)` quartet walks up to `7^4 = 2 401` primitive quartets and writes
`3^4 = 81` contraction blocks of 81 Cartesian elements. A quartet list is *short*
(H2O is 7 shells, 406 canonical quartets, against 19 shells and 18 145 quartets in
def2-TZVP) and each quartet is *deep*. Whatever the kernel does per primitive quartet
per contraction block is the whole cost.

## 2. Current state before this plan (evidence)

### 2.1 What the kernel did for `nctr > 1`

`two_electron_scalar_kernel`, read 2026-09-06 (`crates/cintx-cubecl/src/kernels/two_electron.rs`):

1. `is_uncontracted` is false, so S2's private accumulator is bypassed and the
   hoisted `prim_weight` is unused.
2. Per primitive quartet, per Cartesian element, the kernel ran a four-deep
   `while ci < nctr_i { while cj … { while ck … { while cl … }}}}` and did
   `cart_out[…] += weight * sum` for every `(ci, cj, ck, cl)` — a global-memory
   read-modify-write through a kernel-argument pointer, `nctr_i·nctr_j·nctr_k·nctr_l`
   times per element per primitive quartet. On a TZVP `(pp|pp)` that is
   `2 401 × 81 × 81 ≈ 15.8 M` read-modify-writes for one quartet.
3. libcint does not do this. `CINT2e_loop_nopt` (`cint2e.c:193-262`) contracts in
   four stages, one per primitive index (`PRIM2CTR` → `CINTprim_to_ctr_0/_1`,
   `g1e.c:531-560`): `gout → gctri[ci]` once per primitive quartet,
   `gctri → gctrj[cj][ci]` once per `j` primitive, `gctrj → gctrk` once per `k`,
   `gctrk → gctr` once per `l`. Shells with `x_ctr == 1` have their coefficient
   folded into the prefactor `fac1x` instead of a stage. For nprim 7, nctr 3 the
   multiply-add count per element is `2 401·3 + 343·9 + 49·27 + 7·81 = 12 180`
   against `2 401 · 81 = 194 481` — 16x fewer — and the output is touched
   `7 · 81` times rather than `2 401 · 81`.

### 2.2 Baseline throughput (naive contraction)

`def2_throughput_benchmark::gth_batched_throughput`, CPU backend, release,
`extended-device-rys`, best of 5, screened at `1e-10`, 0 mismatched elements. The
libcint column is single-threaded vendored 6.1.3 over the identical list.

| workload | shells | quartets | libcint (s) | cintx (s) | ratio | max\|diff\| vs vendor |
|---|---|---|---|---|---|---|
| H2O / DZVP-MOLOPT-SR | 7 | 406 | 0.041 | 0.077 | 1.87x slower | 3.3e-15 |
| CH4 / DZVP-MOLOPT-SR | 11 | 2 211 | 0.155 | 0.147 | 1.05x faster | 1.9e-15 |
| SO2 / DZVP-MOLOPT-SR | 9 | 1 035 | 0.180 | 0.205 | 1.14x slower | 7.9e-15 |
| C6H6 / DZVP-MOLOPT-SR | 30 | 108 345 | 21.88 | 15.89 | 1.38x faster | 1.9e-15 |
| H2O / TZVP-MOLOPT | 7 | 406 | 0.319 | 0.487 | 1.53x slower | 3.1e-13 |
| CH4 / TZVP-MOLOPT | 11 | 2 211 | 1.027 | 2.742 | 2.67x slower | 6.9e-14 |
| SO2 / TZVP-MOLOPT | 9 | 1 035 | 1.351 | 1.835 | 1.36x slower | 3.1e-13 |

Against the def2 rows (1.2–1.8x faster than libcint on the same host), batched cintx
was *slower* than single-threaded libcint on every TZVP-MOLOPT workload, with 16 host
threads. That is the contraction arm: the deeper the contraction, the worse the row.

### 2.3 Memory

Host peak was 2.1–2.35x the spherical output (the retained Cartesian chunk), device
scratch under 1 MiB, pair table negligible (a family basis has few shells: 30 shells
for benzene is 900 ordered pairs × ≤ 49 primitive pairs ≈ 2 MiB). The GTH memory
story is the def2 one: bounded by `memory_limit_bytes` (M1) and by the device
transform (M3), both of which apply unchanged. What is new is the contraction
scratch this plan adds, which has to be accounted for in the pre-flight plan.

## 3. Gates

- **G1 (parity)**: both contraction schemes agree with vendored libcint element-wise
  at `1e-12` on every GTH fixture; every def2 gate stays green and the segmented
  path is bit-identical to before (it is untouched by construction:
  `is_uncontracted` short-circuits the new arm).
- **G2 (speed)**: an in-process A/B (`gth_contraction_ab`) shows the staged scheme
  faster than the naive one on every GTH workload, on the CPU backend and on ROCm.
  Ratios are quoted from that A/B only.
- **G3 (memory)**: the contraction scratch is one allocation per run, sized from the
  same expression the pre-flight plan charges, and reported through the existing
  `device_g_slab_bytes_*` fields.
- **G4 (record)**: GTH rows land in their own artifact
  (`cintx_gth_throughput.json`, schema `cintx_def2_throughput/2`) rather than
  overwriting the def2 one.

## 4. Workstreams

### C1 — Staged general contraction (the whole speed lever)

Files: `two_electron.rs` kernel and host, `pair_table.rs` (unchanged, relied on).

1. Reproduce libcint's four stages on the device. The compacted pair rows are
   ordered `(pl, pk)` and `(pj, pi)` by `PairTable::push_shell_pair`, so "the `l`
   primitive changed" is detectable on a compacted list, and libcint's
   `*empty`/assign-then-accumulate flags carry over verbatim. Segmented shells fold
   their coefficient into the primitive weight, as libcint folds into `fac1x`.
2. The three intermediates `gctri[ci][q]`, `gctrj[cj][ci][q]`, `gctrk[ck][cj][ci][q]`
   live in a per-slot scratch slab, allocated once per run and reused like the G
   slab (M4.1). Its size is `(ni + ni·nj + ni·nj·nk) · block_len` at the widest
   quartet of the dispatch — 25 KiB per slot for TZVP `(pp|pp)`.
3. Every stage touches only a lane's own elements (`q % lanes == lane`), so the
   cooperative decomposition needs no barrier between stages; the kernel's two
   existing `sync_cube` calls per primitive quartet are unchanged.
4. `ctr_mode` is a runtime scalar (`CINTX_2E_CONTRACT=naive|staged`,
   `set_staged_contraction`), so the A/B is one compiled program.
5. Bit-identity with the naive scheme is **not** the gate — the association differs
   by design and the staged one is the vendor's. The gate is G1.

### C2 — The GPU arm

The same code path is the cooperative arm; C1 is compiled for every backend. The
work here is measurement, not code: the ROCm A/B and the cross-backend agreement
bar `def2_batch_rocm_parity` uses (8 eps of each block's scale).

### M1 — Scratch accounting

`TwoELaunchGroup::max_ctr_len` / `ctr_slab_bytes`, `slot_scratch_bytes` in the
cube-count and per-unit-width budgets, the contraction slab in `plan_batch_bytes`.
One expression for the plan and the allocation, as the G slab already had.

### F — Not taken, recorded for the next pass

- **F1, the family-quartet kernel. Attempted, and refused on the evidence — §9.3.**
  What follows is the case as it was written before S3 measured it; §9.3 is why
  it does not survive, on three independent grounds.
  In a family basis every shell of an atom shares its exponents, so for one atom
  quartet the primitive pair data, the Rys argument and the Rys roots are identical
  across all `3^4 = 81` `(l_i, l_j, l_k, l_l)` classes (TZVP: s, p, d per atom).
  One G tensor built at the atom quartet's `(l_max…)` contains every lower class's
  entries, so a kernel that walks *atom* quartets and contracts every shell class
  out of one G build per primitive quartet would replace up to 81 G builds with one
  `(dd|dd)`-sized build. Not attempted here: it changes the unit of work in the
  driver, the launch grouping and the output layout at once.
- **F2, private `gctri`.** The `i` stage is the one paid per primitive quartet; on
  the per-unit shape it would fit S2's private array for every `nroots ≤ 3` TZVP
  class (`3 × 81 = 243 ≤ 256`). S2 measured 3–6% for the analogous move on def2, so
  it is not worth its own pass until F1 is decided.
- **F3, pair-table deduplication.** All `(l_i, l_j)` pairs of one atom pair share
  their primitive pair data up to the `l_ij`-dependent `cceij` term. Worth a row
  index rather than a copy once a work list has thousands of shells; at benzene's
  30 shells it is 2 MiB and not on any path.

## 5. Verification

- `gth_contraction_ab` (new): both schemes vs vendor at `1e-12` on the six GTH
  fixtures (benzene under `CINTX_BENCH_SCOPE=full`), interleaved in-process timing;
  a cheap non-ignored water gate for the default suite; a ROCm arm behind
  `CINTX_ROCM_ORACLE=1` that also holds the cooperative result to the CPU one.
- `general_contraction_device_indexing`, `def2_2e_batch_parity`,
  `def2_batch_rocm_parity` (the `int2e_sph` case) re-run under both settings.
- `def2_throughput_benchmark::gth_batched_throughput` (new) for the artifact rows.

Commands:

```text
CINTX_ORACLE_BUILD_VENDOR=1 cargo test --release -p cintx-oracle \
  --features cpu,extended-device-rys,gth --test gth_contraction_ab -- --ignored --nocapture
CINTX_ORACLE_BUILD_VENDOR=1 CINTX_ROCM_ORACLE=1 cargo test --release -p cintx-oracle \
  --features cpu,rocm,extended-device-rys,gth --test gth_contraction_ab -- --ignored --nocapture
CINTX_ORACLE_BUILD_VENDOR=1 CINTX_BENCH_SCOPE=full cargo test --release -p cintx-oracle \
  --features cpu,extended-device-rys,gth --test def2_throughput_benchmark \
  -- --ignored --nocapture --test-threads=1 gth_batched_throughput
```

## 6. Risks

| Risk | Control |
|---|---|
| The staged scheme changes a def2 result | It cannot reach one: `use_staged` requires a `nctr > 1` shell; `def2_2e_batch_parity` and the ROCm `int2e_sph` case re-run identical. |
| A stage flushed at the wrong primitive boundary | The gate is the vendor, not the naive kernel; a wrong boundary is a wrong number, orders of magnitude from either. `general_contraction_device_indexing` covers a mixed `nctr = (2,1,2,1)`-style quartet, the GTH fixtures cover `(3,3,3,3)`. |
| The scratch slab is missed by the memory plan | `ctr_slab_bytes` enters `plan_batch_bytes` from the same `ctr_slab_stride` the allocation uses. |
| A stale compiled kernel on ROCm | The signature changed (three new arguments), which changes `KernelId`; the HIP cache was also cleared before the ROCm runs. |
| A GPU measurement read off a busy host | ROCm timings are in-process A/B ratios and are not compared to CPU absolutes. |

## 7. Facts this plan rests on

- Basis composition: `crates/cintx-basis/data/gth/BASIS_MOLOPT` (H, C, N, O, S
  entries for both families).
- Kernel behaviour (§2.1): `two_electron_scalar_kernel`, `two_electron.rs`, read
  2026-09-06; libcint's staging: `libcint-master/src/cint2e.c:24-45, 187-262` and
  `src/g1e.c:531-560`.
- Pair-row ordering: `PairTable::push_shell_pair` (`pair_table.rs`), `for q in ket
  { for p in bra }`.
- Baseline rows: `gth_batched_throughput` run of 2026-09-06 (§2.2), before C1.

## 8. Execution record (2026-09-06)

### 8.1 What landed

| Item | Status | Evidence |
|---|---|---|
| GTH raw arrays (`to_raw_arrays_gth`) and fixtures (H2O, CH4, SO2, C6H6 × 2 bases) | done | `crates/cintx-basis/src/raw.rs`, `def2_fixtures::gth_workloads` |
| C1 staged general contraction, runtime A/B switch | done | `stage_contract`, `stage_contract_out`, `contraction_mode`; `gth_contraction_ab` |
| M1 scratch accounting | done | `TwoELaunchGroup::max_ctr_len`, `slot_scratch_bytes`, `plan_batch_bytes` |
| C2 ROCm measurement | done | §8.3 |
| GTH artifact rows | done | `cintx_gth_throughput.json` |
| F1–F3 | not taken | §4 |

### 8.2 CPU: the in-process A/B

`gth_contraction_ab::staged_contraction_matches_vendor_and_is_measured_cpu`, best of 5,
naive and staged alternated repeat by repeat, one prewarm, unscreened canonical lists.

| workload | quartets | naive (ms) | staged (ms) | speedup | vendor \|diff\| naive → staged | scratch / slot |
|---|---|---|---|---|---|---|
| H2O / DZVP-MOLOPT-SR | 406 | 43.1 | 30.7 | **1.41x** | 3.3e-15 → 3.3e-15 | 203 KiB total |
| CH4 / DZVP-MOLOPT-SR | 2 211 | 123.9 | 106.0 | **1.17x** | 1.9e-15 → 3.1e-15 | 203 KiB |
| SO2 / DZVP-MOLOPT-SR | 1 035 | 174.4 | 128.9 | **1.35x** | 7.9e-15 → 6.4e-15 | 486 KiB |
| H2O / TZVP-MOLOPT | 406 | 208.7 | 130.4 | **1.60x** | 3.1e-13 → 2.6e-13 | 395 KiB |
| CH4 / TZVP-MOLOPT | 2 211 | 634.6 | 397.2 | **1.60x** | 6.9e-14 → 4.1e-14 | 395 KiB |
| SO2 / TZVP-MOLOPT | 1 035 | 1 068.1 | 517.6 | **2.06x** | 3.1e-13 → 2.6e-13 | 851 KiB |

Every row is faster, every row is inside the oracle tolerance under both schemes,
and on every TZVP row the staged scheme lands *closer* to the vendor than the naive
one did, because it sums in the vendor's own association. The scratch column is the
whole run's contraction slab (16 slots), reported through `device_g_slab_bytes_peak`.

Why 1.2–2.1x and not the 16x of the multiply-add count: the contraction was never
the only cost. The per-primitive-quartet G build (Rys roots, VRR, HRR) is unchanged
and, on this CPU runtime, dominates once the contraction stops being pathological.
That is what F1 is about.

### 8.3 ROCm: the cooperative arm

`gth_contraction_ab::staged_contraction_matches_vendor_and_is_measured_rocm` on
gfx1151 (integrated, unified memory), best of 3, interleaved, one fixture per
process (`CINTX_GTH_FILTER`), HIP kernel cache cleared first. The kernel is the
same code; on ROCm it runs the cooperative decomposition — one quartet per cube,
the G build on lane 0, the contraction and the stages split across lanes.

| workload | quartets | naive (ms) | staged (ms) | speedup | vendor \|diff\| naive → staged | scratch |
|---|---|---|---|---|---|---|
| H2O / DZVP-MOLOPT-SR | 406 | 1 103.7 | 828.6 | **1.33x** | 3.3e-15 → 3.3e-15 | 595 KiB |
| H2O / TZVP-MOLOPT | 406 | 7 867.1 | 3 214.7 | **2.45x** | 4.0e-13 → 2.8e-13 | 1.6 MiB |
| CH4 / DZVP-MOLOPT-SR | 2 211 | 1 992.1 | 1 558.1 | **1.28x** | 2.7e-15 → 1.3e-15 | 2.9 MiB |
| CH4 / TZVP-MOLOPT | 2 211 | 11 765.2 | 6 261.0 | **1.88x** | 6.1e-14 → 5.7e-14 | 8.0 MiB |
| SO2 / DZVP-MOLOPT-SR, SO2 / TZVP-MOLOPT | 1 035 | not measured on ROCm — see below | | | | |

**The SO2 rows are missing on ROCm, and the reason is the day's real finding.**
Three attempts at the SO2 fixture's GPU run each ended with the development
host's desktop dying, reported at the time as "OOM". The journals say otherwise
(§8.6): every one of those events is an **amdgpu gfx-ring timeout and GPU reset**
during a GTH GPU run, after which the Wayland compositor loses its context and
aborts, taking the whole session — and the test process — with it. No kernel or
`systemd-oomd` kill accompanies any of them. The run was not attempted a fourth
time. The CPU rows for SO2 stand (§8.2); the GPU claim rests on the four fixtures
above.

The scratch column is the run's whole contraction slab: one per *cube* in this
decomposition, so it scales with the quartet count (2 211 cubes × 3.7 KiB on
CH4/TZVP) rather than with the 16 units of the CPU shape. It is budgeted by
`slot_scratch_bytes` against `MAX_BATCH_SCRATCH_BYTES` exactly as the G slab is.

The GPU is far slower than the CPU here in absolute terms (7.9 s against 0.21 s
for H2O/TZVP naive) and that is not the contraction: on gfx1151 the whole VRR/HRR
build of every one of a quartet's 2 401 primitive quartets runs on lane 0 of a
32-lane cube while the other lanes wait at a barrier. `def2_batch_rocm_parity`
says the same — gfx1151 is a *correctness* target for the launch topology. What
this run establishes is that the staged scheme is the right shape on a GPU too:
it removes global read-modify-writes, which is what a GPU is worst at, and the
TZVP speedup is larger on ROCm than on the CPU.

**Cross-backend agreement.** The def2 ROCm suite holds the cooperative and
per-unit results to 8 eps of each block's scale. That bar does not survive a
`7^4`-deep generally contracted quartet: measured 49 eps (H2O/DZVP-SR), 577 eps
(H2O/TZVP), 606 eps (CH4/TZVP) — with *both* backends 3e-13 or better from the
vendor. The AMD compiler fuses the multiply-adds the CPU one leaves separate, and
across 2 401 primitive quartets and three contraction stages the two roundings
drift by ~1e-13 on elements of order one. The test therefore holds the two
backends to `2 × 1e-12` absolute — the bound the two vendor gates already imply —
and records the eps figure rather than gating on it.

### 8.4 The whole-workload rows

`gth_batched_throughput` after C1, CPU backend, best of 5, screened at `1e-10`, 0
mismatched elements; recorded in `artifacts/cintx_gth_throughput.json`. The two
engines run in the same process on the same list, so the ratio is the number to
read; the cintx column on its own is not comparable to §2.2's (§10.8 of the def2
plan).

| workload | quartets | libcint (s) | cintx (s) | ratio | max\|diff\| vs vendor | before C1 (§2.2) |
|---|---|---|---|---|---|---|
| H2O / DZVP-MOLOPT-SR | 406 | 0.034 | 0.031 | **1.11x faster** | 3.3e-15 | 1.87x slower |
| CH4 / DZVP-MOLOPT-SR | 2 211 | 0.143 | 0.090 | **1.58x faster** | 3.1e-15 | 1.05x faster |
| SO2 / DZVP-MOLOPT-SR | 1 035 | 0.163 | 0.110 | **1.47x faster** | 6.4e-15 | 1.14x slower |
| H2O / TZVP-MOLOPT | 406 | 0.137 | 0.116 | **1.18x faster** | 2.6e-13 | 1.53x slower |
| CH4 / TZVP-MOLOPT | 2 211 | 0.548 | 0.348 | **1.58x faster** | 4.1e-14 | 2.67x slower |
| SO2 / TZVP-MOLOPT | 1 035 | 0.685 | 0.455 | **1.51x faster** | 2.6e-13 | 3.1e-13; 1.36x slower |

Every GTH row moved from slower than single-threaded libcint to faster, and every
TZVP row is closer to the vendor than before. Benzene was measured only in the
baseline (§2.2, DZVP-SR: 1.38x faster before C1); its TZVP row was not re-run —
that fixture's first run coincided with the first of the day's session kills and
the host's stability did not justify a second ten-minute attempt.

### 8.5 Memory

Nothing GTH-specific changed the memory picture, and nothing needed to. The
contraction scratch is 0.2–0.9 MiB for the whole CPU run (16 slots) and up to
8 MiB on ROCm (one slot per cube), is allocated once per run beside the G slab,
enters the pre-flight plan through `ctr_slab_bytes`, and is reported through
`device_g_slab_bytes_*`. Host peak stays 2.1–2.35x the spherical output under
the default unbounded chunking, exactly as on def2, and the two existing levers
apply unchanged: `memory_limit_bytes` (M1: 1.20x on SO2/def2-TZVP) and
`CINTX_2E_TRANSFORM=device` (M3: readback equals the output). The one shape
worth watching is the cooperative decomposition's per-cube scratch on a large
generally contracted list, which `slot_scratch_bytes` now caps against
`MAX_BATCH_SCRATCH_BYTES` alongside the G slab.

### 8.6 What "OOM" was: the journals, reconstructed

Kernel and user journals for boots `-1` (15:08–19:44) and `0` (19:45 onward),
read after the fact.

| time | record | what was running |
|---|---|---|
| 16:11:27 | **kernel OOM**, global. Victim `rust-analyzer-2`, 11.3 GB anon RSS. Process table sum 24.4 GB on a 30.6 GB host: two rust-analyzer instances 14.1 GB, `def2_throughput` (the GTH baseline on C6H6/TZVP) 1.7 GB, seven `rustc` ≈ 3.1 GB, zed, claude, browsers. Swap 86% full. | GTH baseline benchmark + `cargo check` of `cintx-cubecl` + the editor's language server indexing this workspace |
| 16:35:16 | `ring gfx_0.0.0 timeout` → GPU reset, `device wedged` | first GTH ROCm A/B (all fixtures), H2O phase |
| 16:56:24, 17:01:47, 17:01:50 | three more gfx-ring timeouts and resets | second GTH ROCm A/B; 17:01 is the SO2 phase |
| 19:40:32, 19:40:39 | two gfx-ring timeouts and resets, `warp-terminal` and `gnome-shell` jobs | SO2 ROCm A/B, second attempt |
| 19:44:36 | clean `systemd-reboot` | user reboot |
| 19:47:18 | gfx-ring timeout → reset; `gnome-shell: The CS has cancelled because the context is lost. This context is guilty of a hard recovery`; `gnome-shell … terminated abnormally with signal 6/ABRT` | SO2 ROCm A/B, third attempt, launched 19:46 |

Boots `-2` and `-3` (the def2 ROCm suites' days) contain **zero** gfx-ring
timeouts. Boot `-1` contains seven, all inside GTH GPU runs. The GTH test process
itself peaks at 512 MiB RSS; GPU memory (GTT) never exceeded 0.8 GiB of 15.3 GiB.

So there were two different failures under one name:

1. **One genuine out-of-memory event**, 16:11, whose dominant consumer was the
   editor's `rust-analyzer` (14 GB across two instances) with a cintx benchmark
   and a `cargo check` running beside it. The kernel killed rust-analyzer; the
   memory pressure took the `cargo check` (exit 137) and the session with it.
2. **Six GPU wedges.** gfx1151 is the display GPU. The batched 2e kernel's
   cooperative arm runs one quartet per cube with the whole G build serial on
   lane 0, and on the GTH fixtures — especially under the *naive* contraction
   arm the A/B deliberately still exercises, at 15.8 M global read-modify-writes
   per `(pp|pp)` cube — a single dispatch of hundreds of such cubes runs for tens
   of seconds. The compositor's frame job, queued on the gfx ring behind that
   compute work, exceeds amdgpu's job timeout; the driver resets the ring and
   marks the device wedged; gnome-shell's context is lost and it aborts; the
   session, including any process started from it (`setsid` does not help —
   session teardown kills by cgroup), is gone. The user sees a frozen or restarted
   desktop and reads it as memory exhaustion.

**Consequences for this project.**

- A GPU run on the display GPU must keep every dispatch well under the gfx job
  timeout. `CINTX_2E_CHUNK_QUARTETS` (added with this record) caps the quartets
  per chunk, and so per dispatch, independently of the byte-based
  `CINTX_2E_CHUNK_MIB`; the launch cost is what the def2 plan measured for
  chunking (§10.2 there). It bounds the staged arm comfortably. The naive arm is
  an A/B reference only and should not be timed on a display GPU at all.
- The proper fix is the def2 plan's S3 (cooperative G build across planes),
  which shortens the per-cube time rather than the dispatch length; F1 (the
  family-quartet kernel) shortens it further.
- Long GPU jobs on this host should run from a session that does not die with
  the compositor (a TTY, `systemd-run --user --scope`, or a separate login), and
  with `rust-analyzer` not indexing this workspace beside a build.

## 9. S3, and what it settles about F1 (2026-09-07)

### 9.1 S3: the cooperative G build, split across the cube

Until now `two_electron_scalar_kernel`'s cooperative arm — the shape every GPU
backend runs — built the whole G tensor inside a `lane == 0` region. Every
other lane in the cube idled from the Rys roots through the VRR and the HRR,
and joined only for the contraction. On a 32-lane wavefront that is 1/32 of the
machine doing the part of the kernel that is pure arithmetic.

**The build parallelises with no reduction and no new barrier.** The unit is
the `(axis, root)` pair. `build_2e_shape` lays the G tensor out root-fastest
with `di = nroots`, `dk = nroots·dli`, `dl = nroots·dli·dlk`,
`dj = nroots·dli·dlk·dll`, so *every* stride — including the VRR's
`g2d_ijmax ∈ {di, dj}` and `g2d_klmax ∈ {dk, dl}` — is a multiple of `nroots`.
A VRR at `(axis, root)` therefore touches only `off + root + n·dn + m·dm`,
which never leaves that root's residue class, and `off = gx_off + axis·g_size`
keeps the axes apart. The HRR at `(axis, root)` reads and writes the same
slice. So there are `3 · nroots` independent tasks, handed out
`task % lanes == lane`.

Three consequences, and the third is the one that matters:

- The Rys roots are computed by **every** lane rather than broadcast. They are
  a pure function of `x_rys` into per-work-item private arrays, so this needs
  no barrier and no shared storage, and the redundancy is free precisely where
  it is paid — those lanes were idle.
- The seed moves inside the task. The lane that owns `(axis, root)` writes that
  slice's seed and is the only lane that reads it, so no barrier separates the
  seed from the VRR, or the VRR from the HRR. The one barrier that remains is
  the pre-existing one before the contraction, which genuinely does read every
  axis.
- **Each element is still computed by exactly the expression that computed it
  before, on a different lane. The result is bit-identical**, so the gate is
  bit-identity rather than a divergence budget — which is what the plan's
  original S3 sketch (distribute *primitive quartets*, reduce across planes)
  would have cost.

`CINTX_2E_COOP_BUILD=lane0` restores the old shape as a runtime scalar, so the
A/B is one compiled program.

**Verified without a GPU.** `two_e_cooperative_arm.rs` pins the decomposition
and the cube width in-process (`set_two_e_per_unit`, `set_two_e_cube_dim`) and
holds the per-unit arm, the split cooperative arm and the lane-0 cooperative
arm to bit-identity, with all three checked against vendored libcint. It runs
on the CPU backend in five seconds, over one quartet from each launch class of
H2O in def2-SVP, DZVP-MOLOPT-SR and TZVP-MOLOPT — so all four comptime HRR
branches and several Rys orders. That a GPU is where S3 *pays* does not make it
where S3 has to be *checked*, which matters on a host whose display GPU cannot
survive a long dispatch (§8.6).

**Measured on ROCm** (gfx1151, cooperative arm, best of 3, interleaved in one
process, `CINTX_2E_CHUNK_QUARTETS=128`, two independent runs):

| workload | quartets | lane-0 (ms) | split (ms) | speedup | run 2 |
|---|---|---|---|---|---|
| H2O / def2-SVP | 3 081 | 133.3 | 119.2 | **1.12x** | 1.10x |
| H2O / DZVP-MOLOPT-SR | 406 | 1 669.0 | 1 310.3 | **1.27x** | 1.28x |
| H2O / TZVP-MOLOPT | 406 | 6 481.4 | 5 084.0 | **1.27x** | 1.27x |

Bit-identical in every row. The CPU default path is untouched: the per-unit
shape collapses the ownership map to `r_first == 0` and a step of one, which is
the loop that was there before, and the GTH CPU A/B reproduces its §8.2 numbers
within noise (30.3/106.6/126.0/127.0/401.1/519.0 ms against
30.7/106.0/128.9/130.4/397.2/517.6).

### 9.2 S3's other half: the shared-memory G slab does not pay here

The plan's S3 also asked for the cooperative G slab in shared memory. That
integration existed already and was believed broken by a backend defect; the
defect was cintx's own (§8 of the def2 plan's note, corrected 2026-09-06), and
with it fixed the slab is correct. It had never been *timed*.

Timed now, in the same interleaved A/B, against the split global slab:

| workload | split, global (ms) | shared (ms) | ratio |
|---|---|---|---|
| H2O / def2-SVP | 119.2 | 118.4 | 1.01x |
| H2O / DZVP-MOLOPT-SR | 1 310.3 | 1 338.2 | 0.98x |
| H2O / TZVP-MOLOPT | 5 084.0 | 5 282.8 | 0.96x |

Bit-identical, and a wash to a small loss. **It stays off by default, now on a
measurement rather than on a defect report.** The likely reason is occupancy
and it is structural: `SharedMemory::new` takes a *comptime* extent, so the
kernel allocates the full `SHARED_G_SLOTS` (6 144 f64 = 48 KiB) per cube
whatever class it is running, and against gfx1151's 64 KiB of LDS that admits
one workgroup per compute unit. The traffic saved is real; the latency hiding
lost is worth about as much. Sizing the allocation to the class instead would
mean one compiled program per `g_size`, which is exactly the launch-class merge
(Task 35-M1) that took def2-SVP from 69 dispatches to 16 — so it is a trade
against a measured win, not a free improvement.

### 9.3 F1 is refused, on three independent grounds

F1 proposed walking *atom* quartets and serving all `3^4 = 81` shell classes of
a family-basis atom quartet from one G build. It does not survive contact with
`build_2e_shape` or with §9.1's measurement, and the third ground alone settles
it.

**(a) The G tensor's layout key is the full `(li, lj, lk, ll)`, not `nroots`.**
`g2d_ijmax` is `di` when `ibase` and `dj` otherwise; `g2d_klmax` is `dk` when
`kbase` and `dl` otherwise; and `dli`/`dlj`/`dlk`/`dll` depend on all four
angular momenta and on which side of the strict-`>` branch the pair falls.
`(1,1,1,1)` and `(2,2,0,0)` share `nroots = 3` and have `g_size` 108 against 45,
with different VRR strides. `(2,0,0,0)` and `(0,2,0,0)` share `nroots = 2` *and*
`g_size = 6` and still differ, because `ibase` flips. So "one tensor at the
atom quartet's `l_max` contains every lower class's entries" is false as
stated: the *values* exist but at offsets no lower class can address, and
recovering them costs a restride copy of about what the build costs.

**(b) The Rys roots are not shared either.** `nroots = (li+lj+lk+ll)/2 + 1`, so
the 81 classes of a TZVP-MOLOPT atom quartet span `nroots` 1 through 5, and a
5-point rule's nodes are not a superset of a 1-point rule's. Sharing means
evaluating every class at the atom quartet's maximum order. That is
mathematically valid and numerically *different* — a different quadrature order
rounds differently — so it would move exactly the low-`l` classes that cintx's
oracle gates are tightest on away from libcint, to buy speed.

**(c) The measurement leaves it nothing to win.** S3 parallelised the G build
and nothing else, so its speedup inverts to the G build's share of the
cooperative kernel. At `3 · nroots` tasks, the parallel factor is 3 to 15
across the classes present; over that range Amdahl puts the G build at

| workload | S3 speedup | G-build share of the kernel | ceiling for *any* G-build optimisation |
|---|---|---|---|
| H2O / def2-SVP | 1.12x | 13 – 16 % | 1.15 – 1.19x |
| H2O / TZVP-MOLOPT | 1.27x | 23 – 27 % | 1.30 – 1.37x |

**S3 has already taken most of that ceiling.** What is left for F1 is a few
percent — and F1's mechanism pays for it by multiplying the contraction's root
loop, which is the other 73–87 %, by `nroots_max / nroots_class`: up to 5x on
an `(ss|ss)` class, roughly 2x averaged over a TZVP-MOLOPT atom quartet. It
trades at most a few percent for tens of percent, in the wrong direction.

**What the same measurement does point at.** Three quarters of the cooperative
kernel is the contraction, and its inner statement is
`sum += gx * gy * gz` over `block_len · nroots` triples — three loads per two
flops, against a G tensor in global memory. That is a memory-bound loop, which
is why §9.2 tried the obvious fix and why the reason it failed (a comptime
shared-memory extent forcing 48 KiB per cube) is the thing to attack. The
honest next lever is **not** a family-quartet kernel; it is either a
class-sized shared allocation bought at the cost of the launch merge, or
vectorising the contraction's root loop (D3.3, still open from the def2 plan
and explicitly sequenced after S2 for this reason). Both are measurable against
the A/B this section leaves in place.

**F2 and F3 are unchanged** and stay open; F2 (private `gctri`) is now bounded
by the same 73–87 % that bounds F1.

## 10. The profile-guided pass (2026-09-07)

Scope: the same six fixtures, both backends, with the kernel as §9 left it.
Method: the four steps of the CubeCL profiling manual
(`16_profiling_and_bottleneck_identification.md`) — verify correctness, time
portably, attribute, fix one thing per measurement. The host has no hardware
profiler for the CPU runtime (no `perf`; the kernel is JIT-compiled MLIR), so
attribution is by the kernel's own runtime switches, alternated inside one
process, exactly as §9.3 did it. `crates/cintx-oracle/tests/gth_profile.rs` is
the harness: one table per workload, every row a ratio against the default
configuration measured in the same process, plus the memory fields of
`BatchExecutionStats` and a raw-`f64` dump/compare for bit-identity gates.

```text
CINTX_ORACLE_BUILD_VENDOR=1 cargo test --release -p cintx-oracle \
  --features cpu,extended-device-rys,gth --test gth_profile -- --ignored --nocapture
# CINTX_GTH_SCALING=1 adds the unit-count curve; CINTX_GTH_DUMP / CINTX_GTH_COMPARE=<dir>
# write / check the default output bit for bit; CINTX_GTH_BACKEND=rocm runs the
# cooperative arm (keep CINTX_2E_CHUNK_QUARTETS set on a display GPU, §8.6).
```

### 10.1 Attribution before any change (CPU, per-unit arm, 16 units)

| workload | default (ms) | 1 unit | 8 units | naive contraction |
|---|---|---|---|---|
| H2O / DZVP-SR | 29.8 | 0.30x | 0.84x | 0.78x |
| CH4 / DZVP-SR | 90.1 | 0.23x | 0.83x | 0.82x |
| SO2 / DZVP-SR | 94.7 | 0.20x | 0.81x | 0.67x |
| H2O / TZVP | 111.3 | 0.29x | 0.91x | 0.62x |
| CH4 / TZVP | 333.7 | 0.21x | 0.84x | 0.63x |
| SO2 / TZVP | 405.4 | 0.20x | 0.86x | 0.46x |

Two things stood out. **Sixteen units bought only 3.4–4.9x over one** on an
8-core/16-thread part whose SMT half is worth 1.1–1.2x (the 8-unit column),
so a good part of the gap was not arithmetic. And the launch rows are appended
class by class while the per-unit walk was *blocked* — unit `u` took rows
`[u·chunk, (u+1)·chunk)` — so on a family basis the last units drew every
`(pp|pp)`-class row of a dispatch (2 401 primitive quartets, 81 blocks) while
the first drew `(ss|ss)`. The dispatch waits for the slowest unit.

The second finding was read from the kernel rather than a counter, and it is
the one that mattered on the GPU. The contraction walked its Cartesian
elements through a five-deep `(l, k, j, i)` nest of `while` loops per
primitive quartet, and the cooperative arm split the work by testing
`q_elem % lanes == lane` *inside* that nest — so on a 32-lane wavefront every
lane executed all 81 iterations of a `(pp|pp)` block and one lane was live per
step. That is the "contraction is 73–87% of the kernel" of §9.3, seen from the
other side: it was 73–87% because it ran at 1/32 utilisation.

### 10.2 K1 — the Cartesian index table

libcint precomputes `idx` once per shell quartet (`CINTg2e_index_xyz`) and
its inner loop is flat. cintx now does the same on the host:
`TwoELaunchGroup::push_class` appends three `u32` G offsets per Cartesian
element, in the exact order the nest walked them, and the shape row carries
the table's start (`TWO_E_SHAPE_STRIDE` 13 → 14). The kernel's contraction is
one loop, `q = lane; while q < block_len { …; q += lanes }`, reading its three
offsets from the table. Each element is the same expression over the same
roots in the same order, accumulated into the same place; **the output is
bit-identical** on all six fixtures (0 of 2 313 078 elements differ against a
dump taken before the change), on both decompositions
(`two_e_cooperative_arm` holds the 4-lane cooperative arm to the per-unit one
bit for bit), and every vendor gate is unchanged.

Cost: `3 · block_len · 4` bytes per class per dispatch — 1 KiB for `(pp|pp)`,
120 KiB for an `(ff|ff)` class — counted in `upload_bytes` and so in
`transfer_bytes` and the pre-flight plan.

### 10.3 K2 — the cost-balanced per-unit partition

The per-unit walk's bounds now come from the host: `n_slots + 1` row indices
(`per_unit_slot_bounds`), cut where the prefix sum of a per-row cost estimate
crosses each `1/n_slots` share of the dispatch. The estimate
(`quartet_cost_estimate`) is the screened primitive-quartet count from the
pair table times a per-primitive term — `block_len · (3·nroots + 2·nctr_i)`
for the contraction and its `i` stage, plus the G build — and only has to
rank rows. Which unit evaluates a quartet cannot change its value, so the two
partitions are bit-identical by construction; `CINTX_2E_BALANCE=uniform` /
`set_two_e_balance` is the A/B. The cooperative arm is untouched (it indexes
the placeholder at `slot · punit == 0` and keeps its interleaved walk). The
bounds are `4 · (units + 1)` bytes per launch, charged to the device ledger
(`device_table_bytes_total`); `transfer_bytes` is documented as omitting them
because it is summed before a width is chosen.

### 10.4 What the two bought, and what is left (CPU)

In-process A/B rows, best of 3, after K1 + K2 (`gth_profile`, three runs):

| workload | default (ms) | uniform partition | no-contraction probe | G-build share |
|---|---|---|---|---|
| H2O / DZVP-SR | 22.0–22.7 | 0.89–0.94x | 2.98x | 34% |
| CH4 / DZVP-SR | 61.0–62.5 | 0.79–0.93x | 2.19–2.56x | 39–46% |
| SO2 / DZVP-SR | 71.7–73.0 | 0.75–0.84x | 3.16–3.24x | 31% |
| H2O / TZVP | 83.8–96.2 | 0.95–0.99x | 3.25–3.49x | 29–31% |
| CH4 / TZVP | 256–275 | 0.90–0.92x | 2.57–2.61x | 38–39% |
| SO2 / TZVP | 329–372 | 0.90–0.93x | 3.76x | 27% |

K2 alone (measured before K1, same harness): 1.04–1.14x. The whole pass on
the same-day A/B harness (`gth_contraction_ab`, staged column, ms):
30.8 → 20.5, 90.9 → 69.8, 108.7 → 82.4, 117.7 → 90.2, 368.2 → 264.0,
469.6 → 358.5 — **1.30–1.50x**, bit-identical. The probe — the G build with
the contraction skipped — says what is left: the contraction and its stages
are still 54–73% of the per-unit kernel, and it is the honest floor of this
design on this runtime. The per-element cost is nine G loads, six flops, and
`nctr_i` read-modify-writes into the `gctri` slab per primitive quartet;
there is no reuse across elements for a private array to capture. Hoisting
the `i` coefficients into registers (K3, tried) measured 0.85–1.11x — inside
this host's noise band — and was removed rather than kept on a hope.

The whole-workload rows, two engines in one process on the identical list
(`gth_batched_throughput`, `artifacts/cintx_gth_throughput.json`):

| workload | quartets | libcint (s) | cintx (s) | now | §8.4 |
|---|---|---|---|---|---|
| H2O / DZVP-MOLOPT-SR | 406 | 0.036 | 0.021 | **1.72x faster** | 1.11x |
| CH4 / DZVP-MOLOPT-SR | 2 211 | 0.146 | 0.065 | **2.25x faster** | 1.58x |
| SO2 / DZVP-MOLOPT-SR | 1 035 | 0.167 | 0.075 | **2.22x faster** | 1.47x |
| H2O / TZVP-MOLOPT | 406 | 0.138 | 0.084 | **1.65x faster** | 1.18x |
| CH4 / TZVP-MOLOPT | 2 211 | 0.561 | 0.267 | **2.10x faster** | 1.58x |
| SO2 / TZVP-MOLOPT | 1 035 | 0.697 | 0.352 | **1.98x faster** | 1.51x |

Every max|diff| against the vendor is the §8.4 figure to the digit
(3.3e-15 … 2.6e-13), which is what bit-identity looks like from the outside.

### 10.5 The cooperative arm (ROCm, gfx1151)

`gth_profile` on H2O with `CINTX_2E_CHUNK_QUARTETS=32` (13 chunks, 112
launches), best of 3, interleaved:

| workload | default (ms) | naive | no-contraction probe | lane-0 build | §9.1 (128-quartet chunks) |
|---|---|---|---|---|---|
| H2O / DZVP-MOLOPT-SR | 1 213 | 0.96x | 1.05x | 0.65x | 1 310 |
| H2O / TZVP-MOLOPT | 4 609 | 0.82x | 1.06x | 0.65x | 5 084 |

K1 did on the GPU what §10.1 predicted: **the contraction is now 5–6% of the
cooperative kernel** (the probe), against 73–87% in §9.3, and the naive arm
— which pays the same `1/lanes` utilisation no longer — is within 4–18% of
the staged one. The S3 A/B inverted accordingly: with the contraction gone
from the profile, the lane-0 build costs 1.54x rather than 1.27x. Vendor
agreement is unchanged (3.3e-15, 2.8e-13); the cooperative output is not
bit-identical to the CPU's, for the FMA reason §8.3 records, and is held to
`2 × 1e-12` of it by `gth_contraction_ab`.

What the wall clock did *not* do is fall by the 3–4x the contraction share
implied, and the harness says why: 69 quartets — one per class — take 31% of
the whole 406-quartet run. A dispatch is one quartet per cube, and a TZVP
quartet is 2 401 primitive quartets walked *serially* inside that cube; the
dispatch lasts as long as its slowest quartet whether it carries four or
forty. On a 406-quartet molecule the GPU is latency-bound by that serial
loop, not throughput-bound by anything K1 touches, and the chunk cap §8.6
makes necessary on a display GPU multiplies the number of such latencies. The
lever there is the one §5 of the def2 plan sketched and §9.1 deliberately did
not take: distributing *primitive quartets* across planes with a reduction,
which is not bit-identical and needs its own gate. It is the next GPU item,
and it is only worth taking on a discrete GPU where dispatches are not capped.

### 10.6 Memory

Measured, not changed. The GTH host peak is the def2 shape — the spherical
output plus the retained Cartesian intermediate, 2.1–2.35x the output — and
both existing levers were timed on GTH in-process:

| lever | host Cartesian peak | time (CPU) | bits |
|---|---|---|---|
| default | 0.46–10.67 MiB (1.1–1.25x output) | 1.00x | — |
| `CINTX_2E_TRANSFORM=device` | 0 on the host | 0.96–1.08x | identical |
| `memory_limit_bytes` at a quarter of the intermediate (5 chunks) | 0.12–2.66 MiB | 0.60–0.87x | identical |

The device transform is a wash in time on the CPU runtime, as the def2 plan
found, and on that runtime the "device" spherical buffer it reads back is
host memory too, so the peak moves rather than shrinks; on ROCm it is 0.99x
and one ULP from the host transform (FMA). Chunking costs 13–40% on GTH —
more than on def2, because a chunk's dispatches carry fewer quartets and the
per-unit partition has less to balance — for a bounded intermediate. Neither
default moves; both numbers are now on record for a caller who needs the
bound. What this pass added to the footprint is the index tables (KiB per
dispatch) and the partition bounds (bytes); the contraction scratch of §8.5 is
unchanged. A GTH list large enough for memory to matter (benzene/TZVP) is
where `memory_limit_bytes` was already the answer.

### 10.7 Verification

- Bit-identity: `gth_profile` dump/compare on all six fixtures (K1, K2);
  `two_e_cooperative_arm` (per-unit vs cooperative, def2 and GTH);
  `def2_batch_memory_plan::chunked_evaluation_is_bit_identical_to_unchunked`.
- Vendor: `gth_contraction_ab` (water gate in the default suite; all six under
  `--ignored`), `general_contraction_device_indexing`,
  `def2_2e_batch_parity` (its transfer-byte prediction now includes the index
  tables), the throughput rows above (0 mismatched elements at 1e-9, max|diff|
  as §8.4).
- Unit: `partition_tests` (bounds cover every row once, stay monotone,
  reproduce the blocked walk under `uniform`, give an expensive tail its own
  slots), the in-file f32 smoke launch with the two new arguments.
- ROCm: `gth_profile` (above); `gth_contraction_ab` cross-backend on H2O.

## 11. The GPU decomposition: what was investigated, what landed (2026-09-07)

### 11.1 The question §10.5 left

After K1 the cooperative kernel's contraction is 5–6% of its time and the
wall clock on H2O still did not move, because a dispatch is one quartet per
cube and each cube walks its 2 401 primitive quartets *serially*: Rys roots,
a VRR/HRR on `3·nroots ≤ 15` lanes, two barriers, a contraction — per
primitive quartet. On a 406-quartet molecule a launch carries ~27 cubes for
16 compute units (gfx1151) or 40 SMs (T4); the dispatch lasts as long as its
slowest cube's serial walk whatever the rest of the device is doing. The
kernel is latency-bound per cube, not throughput-bound by anything inside it.

Four ways to put more of the device on one quartet were weighed:

| method | what changes | bit-identical? | verdict |
|---|---|---|---|
| **G1 — ket-pair split across cubes** | each quartet becomes `n` rows, each a contiguous slice of its ket-pair range writing its own copy of the output; one reduce kernel sums the copies in order | no (the sum over ket pairs is re-associated); deterministic | **taken** — host-side table change plus a 15-line kernel, verifiable on the CPU's pinned cooperative arm |
| lane-level primitive-quartet parallelism | lanes own primitive quartets instead of `(axis, root)` slices; each builds its own G tensor and partial `gctri`, reduced through shared memory | no | not taken: a rewrite of the cooperative arm (~500 lines) whose private G tensor (144–1 215 f64 per lane) spills to local memory; G1 reaches the same parallelism through the grid with the kernel untouched |
| bra-pair split as well | as G1 on the `ij` range too | no | open — the next step if G1's `max_rows` cap binds (§11.3); costs `parts²` partial buffers |
| shared-memory G slab | on-chip G tensor | yes | measured a wash on AMD (§9.2); the T4 counters (§11.4) decide it for NVIDIA |

### 11.2 G1 — the ket-pair split

- The quartet row grows to eight `u32` (`QUARTET_ROW_STRIDE`): `si, sj, sk,
  sl, out_off, class, kl_lo, kl_hi`. The kernel reads its ket range from the
  row instead of `pair_offset`; an unsplit row carries the whole
  `pair_offset[sk·nbas+sl] ..` span, so the per-unit arm and every existing
  result are unchanged (bit-identical, all CPU gates).
- `kl_split_factor` sizes the split per dispatch on a backend with hardware
  planes: the smallest `n` that gives `8 × parallel_units` cubes, capped by the
  widest ket range in the group, by a 64 MiB partial-buffer budget and by 64.
  Never under a `memory_limit_bytes` (the copies are the peak the budget
  refuses), never on the per-unit shape. `CINTX_2E_KL_SPLIT=off|n` /
  `set_two_e_kl_split` pin it.
- `expand_kl_split` builds the split rows at dispatch (`out_off + p·out_len`
  per part, ket rows in libcint's `(pl, pk)` order inside each part; an empty
  part zeroes its block and nothing else). `reduce_kl_partials` sums the
  copies `p = 0, 1, …` in a fixed order. The transform binds the *unsplit*
  rows against the reduced buffer.
- The row stride and the shape stride are **comptime kernel parameters**, so a
  change to either table's layout changes the compiled program's identity.
  The first ROCm run of G1 page-faulted: the body had changed and the
  arguments had not, and the HIP cache (keyed by signature, not body —
  `def2-plan-open-items`) ran the old kernel against the new table. This is
  the second time that trap has cost a run; the comptime stride is the
  structural fix.

**Gates.** `two_e_cooperative_arm::ket_split_agrees_on_{def2,gth}` pins the
4-lane cooperative arm on the CPU backend, forces splits of 2/7 (def2, where
parts go empty) and 3/8 (GTH), and holds the result to the vendor at 1e-12
and to the unsplit arm within a sanity bound of 1 024 eps of block scale
(measured 20 eps DZVP-SR, 77 eps TZVP — the re-association of up to 49 ket
terms). `kl_split_tests` covers the row expansion. The file's tests now take
one lock, because they pin process-global switches and the harness runs test
functions on parallel threads — the first run of the new test read the other
test's setting.

### 11.3 Measured on ROCm (gfx1151, H2O, `CINTX_2E_CHUNK_QUARTETS=64`, best of 3, interleaved)

| workload | split (ms) | `klsplit=off` (ms) | **G1** | parts | probe | lane-0 build | naive | vendor \|d\| |
|---|---|---|---|---|---|---|---|---|
| H2O / DZVP-MOLOPT-SR | 156.6 | 854.2 | **5.45x** | 25 | 1.11x | 0.77x | 1.04x | 3.1e-15 |
| H2O / TZVP-MOLOPT | 469.8 | 3 256.9 | **6.93x** | 49 | 1.08x | 0.77x | 0.84x | 2.8e-13 |

Against §10.5's numbers from the previous process (1 213 / 4 609 ms at a
32-quartet cap) the same fixtures now run in 157 / 470 ms; the 7–10x is far
outside the host's 2x run-to-run band. `gth_contraction_ab` on ROCm holds the
split result to the CPU at 40.9 / 577.4 eps of block scale — the same figures
as before the split (49 / 577), so the re-association is invisible under the
FMA drift §8.3 already carried.

Two things to read off the table. The split went to `max_rows` (25 and 49
ket pairs) rather than to the occupancy target, so on a molecule this small
the ket range is the binding cap and the bra-pair split of §11.1 is the next
lever — at `parts²` partial buffers, which is where the 64 MiB budget starts
to matter. And with the contraction at 7–10% and the G build 23% (lane-0 A/B),
what remains per cube is the serial chain *inside* a primitive quartet: Rys
roots on every lane, the recurrences on ≤ 15 lanes, two barriers. That is the
profile a hardware counter can attribute (barrier stalls against scoreboard
stalls against f64 issue), which is what §11.4 sends to the T4.

The staged contraction's GPU advantage over the naive fold (2.45x in §8.3)
is gone with the split: each part sees at most a couple of `l` primitives, so
the staging has almost nothing to fold, and on DZVP-SR the naive arm is even
4% faster. It stays the default for the vendor's association and for the
per-unit arm, where it is worth 1.4–2.6x (§10.4).

### 11.4 The T4 measurement package

This host has no NVIDIA device, so the T4 measurement is prepared, not
taken. `ci/colab_t4_profile.sh` and `ci/colab_t4_profile.ipynb` run, on a
Colab T4, the manual's four steps in order: `def2_cuda_verification` and the
CPU-pinned split/arm gates (correctness first); `gth_profile` on CUDA with
`klsplit=off`, `probe:no-ctr`, `coop=lane0`, `naive` and `xform=device` as
in-process ratios, plus the same profile on the VM's CPU backend and its dump
for the cross-backend gap; `nsys profile --stats` for the timeline and
`ncu --kernel-name regex:'two_electron_scalar_kernel.*|reduce_kl_partials.*'
--launch-skip 4` for the §3.3 metric set — sector counts and hit rates on
both the load and the store side, bytes per sector, `dfma` issue, and the
barrier / long-scoreboard / wait stall reasons — and one `--set full` report
for Nsight Compute; and a `summary.json` collector. The CUDA feature compiles
here (`cargo check --features cpu,cuda`), and `gth_profile` accepts
`CINTX_GTH_BACKEND=cuda`. The counters answer the question §11.3 leaves: whether
the per-primitive-quartet chain is bound by the two barriers, by the G-tensor
loads (the case for a shared-memory G on NVIDIA), or by f64 issue (a T4's f64
rate is 1/32 of f32, and then nothing but fewer flops helps).

### 11.5 Verification (this section)

- CPU, bit-identical to §10: `two_e_cooperative_arm` (4 tests),
  `def2_2e_batch_parity` (row prediction at 8 `u32`), `gth_contraction_ab`
  water, `general_contraction_device_indexing`, `def2_batch_memory_plan`,
  the f32 smoke launch, `kl_split_tests`, the full `cintx-cubecl` unit suite.
- ROCm: `gth_profile` (§11.3), `gth_contraction_ab` cross-backend,
  `def2_batch_rocm_parity` with the split active by default.

## 12. V1 — the VRR as vector lanes over the Rys roots (2026-09-08)

### 12.1 The question

§10.4 attributed the CPU per-unit kernel: contraction 54–73%, G build 27–46%,
and called the contraction "the honest floor of this design on this runtime."
The G build was never attacked directly. It divides into the VRR
(`CINTg0_2e_2d`) and the HRR transfer, and the two have opposite shapes: the
VRR is *latency*-bound — per root, one division and two serial two-term
recurrences — while the HRR is a strided read-modify-write over the whole
slab. CubeCL carries a `Vector<F, N>` type that lowers to native SIMD, and the
Rys root axis is the contiguous, aligned, innermost axis of every G slab
(`di = nroots`, and `dk`/`dl`/`dj`/`g_size` are all multiples of it), with no
recurrence crossing it. So the roots can become lanes.

### 12.2 Three candidates, measured before any of them landed

`crates/cintx-cubecl/src/math/root_vec.rs` carries the one that paid; the other
two were measured on the CPU runtime by the same in-process A/B and are recorded
here so the next pass does not repeat them.

| candidate | shape | CPU A/B |
|---|---|---|
| **VRR over roots** | latency-bound; `nroots` serial chains collapse into one | **1.4x–1.9x** (`nroots` 2–5) |
| HRR transfer over roots | gather-bound; `g[dst+r] = c·g[a+r] + g[b+r]` | 0.60–0.81x |
| contraction `Σ_r gx·gy·gz` | gather-bound, plus an in-order lane reduction that is the same dependency chain the scalar loop had | 0.85–1.12x |

At `nroots == 1` the vector VRR measures 0.88x — a width-1 `Vector` is pure
overhead — so the arm is comptime-off there.

Two CubeCL facts fell out of the attempt and are worth keeping:

- `Slice::with_vector_size` (the reinterpreting vector load) is **not supported
  on the CPU runtime**: it sets `IndexOperator::vector_size`, which
  `cubecl-cpu`'s `visit_index` asserts is zero. The lanes are therefore gathered
  and scattered explicitly, which costs nothing here — the win is the
  dependency chain, not the load width.
- Binding a buffer as `Array<Vector<F, N>>` at **odd** `N` returns wrong values
  on the CPU runtime (checked at `N = 3`). Nothing in `root_vec` does; the slab
  stays scalar-typed. `N = 3` and `N = 5` are otherwise ordinary — `VectorSize`
  is a plain `usize` in the IR and CPU/HIP/CUDA all report
  `max_vector_size: VectorSize::MAX`. (wgpu caps it at 4 and has no f64.)

### 12.3 What landed

`vrr_fill_axis_roots` — the scalar `vrr_fill_axis` body with the root index
folded into vector lanes, statement for statement. It is selected at comptime by
`per_unit == 1 && nroots > 1`, i.e. **the per-unit (CPU) arm only**. The
cooperative arm keeps the scalar loop deliberately: S3 hands out `3 * nroots`
independent `(axis, root)` tasks across the cube, and folding the roots into
lanes would cut that to three. On the per-unit arm there is nothing to lose —
`lanes == 1`, so `build_lanes == 1` and `build_lane == 0` whichever `coop_build`
mode is set, and the residue test admits every task.

The width reaches the kernel as a `#[define(N)]` comptime argument rather than a
`Const<N>` generic: `nroots` is already a comptime `u32` (one JIT specialization
per value, one Rust instantiation), and a const generic would have forced five
Rust monomorphizations of the whole kernel through every launcher.

### 12.4 Measured (CPU, per-unit arm, `gth_profile`, best of 3, interleaved)

| workload | scalar VRR (ms) | vector VRR (ms) | whole kernel | G build alone |
|---|---|---|---|---|
| H2O / DZVP-MOLOPT-SR | 22.57 | 21.98 | 1.03x | **1.09x** |
| CH4 / DZVP-MOLOPT-SR | 65.61 | 59.99 | 1.09x | **1.14x** |
| SO2 / DZVP-MOLOPT-SR | 74.61 | 75.91 | 0.98x | **1.09x** |
| H2O / TZVP-MOLOPT | 90.83 | 83.88 | 1.08x | **1.15x** |
| CH4 / TZVP-MOLOPT | 274.11 | 266.86 | 1.03x | **1.17x** |
| SO2 / TZVP-MOLOPT | 370.15 | 360.12 | 1.03x | **1.20x** |

"G build alone" is the `probe:no-ctr` variant, which runs the build and skips
the contraction. It moves consistently, 1.09–1.20x; the whole kernel moves
0.98–1.09x, because §10.4's split still holds and the build is the smaller
half. The 1.4–1.9x the micro-benchmark shows is the VRR in isolation, and the
HRR — untouched — is the rest of the build.

This is a small end-to-end win. It is recorded as landed rather than refused
because it is free of any accuracy cost, and because the build's share is
larger everywhere the contraction is cheaper: §10.5 measured the contraction at
5–6% of the *cooperative* kernel, where the same lever is available the moment
the S3 split stops being the reason not to take it.

### 12.5 Verification

- **Bit-identity, unit level**: `root_vec_matches_scalar_bit_for_bit` runs both
  arms over separate slabs at all five `(nroots, nmax, mmax)` shapes and compares
  `to_bits()`, so the odd widths 3 and 5 are covered. A `Vector` op is
  elementwise on the same operands in the same order, and no fused multiply-add
  is introduced, so there is no divergence budget to spend.
- **Bit-identity, end to end**: `gth_profile` under `CINTX_GTH_DUMP` with the
  arm on, then `CINTX_GTH_COMPARE` with it off — all six GTH workloads, 1.12M
  values on SO2/TZVP alone, zero differing bits.
- **Vendor**: every `max|diff|` is the §10.4 figure to the digit
  (3.33e-15 … 2.65e-13), and `gth_profile` still prints `=bits` for every
  variant.
- **Suites**: `def2_2e_batch_parity` (4), `two_e_cooperative_arm` (4 — the
  cooperative arm is untouched and still agrees with the per-unit one),
  `gth_contraction_ab` water, `def2_integral_parity`, `def2_accumulator_ab`,
  `def2_device_c2s_parity`, the f32 smoke launch, and the `cintx-cubecl` unit
  suite.

### 12.6 What is left

- **The other Rys engines.** `center_3c2e`, `center_2c2e` and `sigma_1e_nuc`
  inline the same `CINTg0_2e_2d` shape over the same root-fastest layout;
  `vrr_fill_axis_roots` applies verbatim. Not taken here — each needs its own
  signature surgery and its own parity gate.
- **The HRR.** Still the larger half of the build and still scalar. The vector
  form loses because it is gather-bound; what would help is a layout with a
  wider contiguous run, not a wider load of the same run.
- **The cooperative arm.** The lever is available there too, and worth more
  (§10.5: the build is ~94% of that kernel), but it trades against S3's
  `3 * nroots` split. The trade is only worth measuring once the GPU stops being
  latency-bound on the serial primitive-quartet loop — the item §10.5 names.

## 13. V1 extended to the 2c2e and 3c2e Rys engines (2026-09-08)

§12.6 named the obvious follow-on: the other Rys engines inline the same 2D
recurrence over the same root-fastest layout, so `vrr_fill_axis_roots` should
apply verbatim. It mostly does. Four device kernels now take the vector arm:

| kernel | ordering | selected when |
|---|---|---|
| `center_2c2e_kernel` | ket-raising | `nroots > 1` |
| `center_3c2e_scalar_kernel` | ket-raising | `nroots > 1` |
| `center_3c2e_ip1_kernel` | bra-raising | `nroots > 1` |
| `center_3c2e_ip2_kernel` | bra-raising | `nroots > 1` |

### 13.1 Two orderings, not one

`CINTg0_2e_2d` appears in this crate in two loop orderings, and they are not
interchangeable — the mixed `b00` recurrence raises a different index and reads
a different neighbour in each:

- **bra-raising**, `g(n+1,m) = c00·g(n,m) + n·b10·g(n-1,m) + m·b00·g(n,m-1)` —
  `two_electron` and the two 3c2e derivative kernels;
- **ket-raising**, `g(n,m+1) = c0p·g(n,m) + m·b01·g(n,m-1) + n·b00·g(n-1,m)` —
  the 3c2e base kernel and `center_2c2e`.

`vrr_fill_axis_roots_ket` is the second, and the module note in
`math::root_vec` says which is which so the next caller does not pick wrong.
Both measure the same: **1.4x–1.9x** on the VRR alone for `nroots` 2..5, and
nothing at `nroots == 1`.

### 13.2 No `per_unit` gate here

`two_electron` takes the vector arm only on the per-unit arm, because S3 splits
`3 * nroots` `(axis, root)` tasks across the cube and folding the roots into
lanes would cut that to three (§12.3). These four kernels have no such split:
each runs its whole build under `lane == 0`, one pair or triple per slot. Both
decompositions therefore take the vector arm, and the ROCm path gets it too.

### 13.3 A correction to §12.2

§12.2 said the root runs are `nroots`-aligned. They are **not**, and it does not
matter. Per-slot slabs are padded to 8 `f64` (`g_slab_stride`,
`three_c2e_slab_stride`), which is not a multiple of an odd `nroots`, so
`slot * g_stride` is unaligned for every slot past the first. `roots_load` /
`roots_store` gather and scatter the lanes element by element and need only a
*contiguous* run, so nothing was ever wrong — but a future move to a
reinterpreting vector load would read across run boundaries, and the padding is
where it would break. The note now lives on `roots_load` itself.

### 13.4 Measured (CPU, `vrr_root_vector_ab` throughput probe, best of 7)

SO2 / def2-TZVP, 35 shells; batched dispatch, two interleaved A/B rounds:

| family | scalar VRR (ms) | vector VRR (ms) |
|---|---|---|
| `int2c2e_sph` (1 225 pairs) | 0.5 | 0.5 |
| `int3c2e_sph` (21 576 triples) | 9.6 / 9.2 | 9.2 / 8.6 |
| `int3c2e_ip1_sph` | 14.1 / 14.9 | 13.8 / 13.9 |
| `int3c2e_ip2_sph` | 15.5 / 14.8 | 14.9 / 13.8 |

**1.02x–1.07x**, consistently in the right direction and the same order as
§12.4's whole-kernel figure. 2c2e is below the resolution of this probe. As in
§12, the reason to land it is that it costs nothing — not that it is large.

### 13.5 Verification

- **Unit, bit-identical**: `root_vec_matches_scalar_bit_for_bit` now runs *both*
  orderings over separate slabs at all five `(nroots, nmax, mmax)` shapes and
  compares `to_bits()`.
- **End to end, bit-identical**: `vrr_root_vector_ab` (new) sweeps all four
  families over a def2-TZVP water — `nroots` 1..5, the odd widths included —
  under `CINTX_VRR_DUMP` with the arms on and `CINTX_VRR_COMPARE` with them off.
  155 107 values across the four families, zero differing bits.
- **Vendor**: the same test's default run holds all four families to 1e-12
  (4.3e-14, 4.0e-15, 1.1e-14, 1.1e-14) at higher angular momentum than the
  STO-3G sweeps in `center_2c2e_parity` / `center_3c2e_parity`.
- **Suites**: `center_2c2e_parity`, `center_3c2e_parity`,
  `def2_3c2e_deriv_batch_parity` (whose batched-vs-`eval_raw` check is at
  `to_bits()`), plus the §12.5 set re-run — `def2_2e_batch_parity`,
  `two_e_cooperative_arm`, `gth_contraction_ab`, `def2_integral_parity` — and
  the `cintx-cubecl` unit suite.

## 14. sigma_1e_nuc — attempted, measured, refused (2026-09-08)

§12.6 listed `sigma_1e_nuc` beside `center_3c2e` and `center_2c2e` as an engine
`vrr_fill_axis_roots` should apply to verbatim. It does not. The axis-wise
alternative was implemented and measured, and then **reverted** — this section
is why, so the next pass does not spend the same day on it. The one piece kept
is `lanes_load` / `lanes_store` in `math::root_vec`.

### 14.1 The root form is unavailable

The 2e, 3c2e and 2c2e engines all carry a root-fastest G tensor: every index is
`base + r`, so each point of the recurrence has a contiguous `nroots`-run to
load. `sigma_1e_nuc` has **no root index in its G tensor at all**. The roots are
the *outer* loop over the whole build-and-contract pipeline — the slab is zeroed,
seeded, filled and then fully contracted inside `for irys in 0..nroots`, with
`gc_out[..] +=` accumulating once per root. `sigma_p`'s `sa01_nuc_vrr_axis` is a
clone of the same shape and inherits the same finding.

Making the root axis exist means interleaving the slab by root: a layout change
rippling through `nuc_vrr_axis`, `nuc_hrr_axis`, `nuc_nabla_i`, `nuc_nabla_j`,
`nuc_nabla_ij` and every index expression in the contraction, across two
kernels, at 5x the scratch. It is also constrained: the contraction must *stay*
inside a per-root loop reading lane `r`, because hoisting it and reducing the
lanes would fold `gc_out[e] += w*c0; += w*c1; …` into
`gc_out[e] += w*(c0+c1+…)` — a different rounding, so not bit-identical. Its
reads would then be strided by `nroots`, which is the shape §12.2 measured as a
*loss*.

### 14.2 The axis form: built, measured, not kept

The dimension that is there is the Cartesian axis: `gx`/`gy`/`gz` sit a constant
`g_per_axis` apart, their recurrences are independent, and only `c00` (VRR) and
`rirj` (HRR) differ. Fusing the three into `Vector<F, Const<3>>` lanes is a
drop-in with no layout change, and it was bit-identical. The isolated VRR+HRR,
best of 5:

| shape | scalar (ms) | vector (ms) | |
|---|---|---|---|
| li=0 lj=0, nmax=2 | 1.5 | 1.4 | 1.09x |
| li=1 lj=1, nmax=4 | 5.5 | 4.6 | 1.18x |
| li=2 lj=2, nmax=6 | 10.4 | 9.1 | 1.14x |

Well short of the 1.4x–1.9x the root form buys the 2e engines: width 3 rather
than up to 5, a one-dimensional two-term recurrence rather than the 2D `(n, m)`
nest, and lanes gathered across a stride.

And that piece is a small fraction of this kernel. Per root, an `(li=1, lj=1)`
build is ~45 zeroing stores, 12 VRR fused multiply-adds and ~24 HRR ones,
against a contraction of nine Cartesian elements each doing ~30 G loads, twelve
`nabla` combinations and nine triple products — roughly 700 operations. A 1.14x
on the VRR is worth well under a percent end to end, below what any driver here
resolves. It was reverted rather than kept on that basis: correct and
unmeasurable is not worth a `Vector` path, two extra helpers and a probe kernel
in a relativistic family.

### 14.3 What was kept

`lanes_load` / `lanes_store` — a strided gather and scatter — stay in
`math::root_vec`, with `roots_load` / `roots_store` expressed as those at stride
one, and `strided_lanes_gather_and_scatter_the_named_elements` covering the
general form so the `stride` parameter is not carried on the strength of a
caller that no longer exists. Any future lane axis that is not the innermost one
needs exactly this.

### 14.4 Where the time actually is

The contraction computes `g0/g1/g2/g3` for each of x, y, z — the same four
operations on three axes, ~30 loads and most of the flops — and *that* is
vectorizable over the same axis dimension. The obstacle is that `s0..s8` mix
lanes (`s0 = g3x*g0y*g0z`), so the products need nine lane extracts, which may
eat the gain. It is the only lever in this kernel with enough mass behind it to
matter.

### 14.5 What the attempt established about the gates

Worth keeping even though the change is gone: each of the two `sigma_1e_nuc`
call sites has its own end-to-end byte-identity gate. Swapping the `y`/`z` axis
operands at the gauge kernel's call site fails `rel_1e_sigma_parity`'s four
byte-identity gates; the same swap at the nuclear kernel's fails
`giao_sigma_1e_parity`'s four. Both were checked by making the swap. A future
attempt on this kernel is well covered.

## 15. F1 — fusing the Rys orders into one dispatch (2026-09-09)

### 15.1 The question, and where the measurement pointed

§10.4 attributed the CPU per-unit kernel — contraction 54–73%, G build 27–46% —
and §12 took a slice off the build. Neither asked what `gth_profile`'s last line
had been printing all along:

```text
launches=15 classes=69 …
launch floor: 69 quartets (one per class) in 15 launches: 56.61 ms = 57.7% of default
```

**Seventeen per cent of H2O/TZVP-MOLOPT's quartets cost 58% of the run.** In a
family basis every quartet of a molecule walks the same `nprim^4` primitive
quartets, so 69 quartets — one per l-class — is one sixth of the arithmetic of
406. The remaining five sixths cost 42%. That is not a kernel finding; it is a
partition finding.

`CINTX_2E_GROUPS=1` (added here, in `plan_2e_stream`) prints what each dispatch
holds. For H2O/TZVP-MOLOPT, 406 quartets in 15 dispatches keyed on
`(ibase, kbase, nroots)`:

| signature | quartets | cost share | costliest quartet |
|---|---|---|---|
| ib0 kb0 nr3 | 67 | 27.4% | 8.0 M |
| ib0 kb0 nr4 | **13** | 22.9% | **33.5 M** |
| ib0 kb1 nr3 | 26 | 9.7% | 8.0 M |
| ib0 kb0 nr5 | **1** | 6.9% | **80.2 M** |
| ib0 kb1 nr4 | **4** | 6.9% | **33.5 M** |
| … 10 more | 11–90 | ≤ 7% each | ≤ 1.6 M |

The per-unit partition (K2) cuts a dispatch's rows across `n_slots` units, so
the floor a dispatch can reach is `max(cost / units, costliest quartet)` and the
run's floor is the **sum** of those over dispatches. A dispatch holding one
`(dd|dd)` quartet occupies one unit of sixteen and the other fifteen wait. On
the `quartet_cost_estimate` scale, at 16 units:

| grouping | H2O/DZVP-SR | CH4/DZVP-SR | SO2/DZVP-SR | H2O/TZVP | CH4/TZVP | SO2/TZVP |
|---|---|---|---|---|---|---|
| per signature (before) | 56.6 | 100.1 | 95.7 | 218.2 | 388.7 | 401.4 |
| fuse the Rys orders | 36.2 | 71.0 | 95.4 | 139.0 | 276.9 | 401.2 |
| fuse `ibase`/`kbase` instead | 41.0 | 88.6 | 95.7 | 158.4 | 344.3 | 401.2 |
| fuse everything | 20.9 | 69.0 | 95.4 | 80.2 | 269.2 | 401.2 |

Three things to read off it. Fusing the **Rys orders** is where nearly all of
the available balance is — fusing `ibase`/`kbase` instead buys a third as much,
and fusing both on top of the orders buys almost nothing more. SO2 is *already*
balanced at either grouping, so the model predicts nothing there. And the model
is a prediction made before a line of the kernel changed, which is what makes
the measured outcome (§15.3) evidence rather than a coincidence.

### 15.2 What landed

The dispatch key drops the Rys order for every class the fixed-order solvers
serve, so one dispatch per `(ibase, kbase)` carries orders 1 through 5:

- `TwoELaunchSignature::nroots` is now a **bucket**: `FUSED_NROOTS_BUCKET` (0)
  for `nroots <= MAX_FUSED_NROOTS` (5), the order itself above that. The
  extended orders 6..=12 are deliberately **not** fused: `rys_roots_ext_dev`
  takes its order at comptime and its double-double arms are an order of
  magnitude larger than `rys_root1..5`, so merging them would emit seven big
  solvers per dispatch to join classes carrying under a percent of any work
  list. Each keeps a dispatch, and `nr_max == nroots` there.
- The class shape row carries `nroots` (`TWO_E_SHAPE_STRIDE` 14 → 15) and the
  kernel reads it per quartet. `nroots` is a runtime scalar in the kernel now;
  the comptime parameter is `nr_max`, the group's widest order, which sizes the
  private root arrays and decides which per-order arms the program emits at all.
- **Three places needed the order at comptime, and each keeps it** behind a
  runtime branch that is taken once per quartet or per primitive quartet, never
  per root:
  - the Rys solver — `if nroots == k { rys_root{k} }`, guarded by
    `comptime!(nr_max >= k)` so a narrow dispatch compiles no wide solver;
  - the vector VRR (V1, §12) — a `Vector` width is a *type-level* size, so the
    block moved into `vrr_build_axes_roots<F, N: Size>` and the kernel
    instantiates it at `Const<2..5>`. This is what retires the `#[define(N)]`
    dynamic size §12.3 introduced: the width now comes from the concrete type at
    each call site. The arm stays comptime-off on the cooperative shape and at
    `nr_max == 1`, exactly as before.
  - the contraction's `Σ_r gx·gy·gz` — `root_dot<F>(…, #[comptime] width)`,
    selected once per Cartesian element. A dynamic trip count of one to five
    would have put a compare and a branch on the hottest statement in the
    kernel; this branch is on the quartet's order, the same for every element of
    its block, and predicts perfectly.
  The scalar VRR's outer loop over roots became an ordinary `while` — its body
  is the whole three-axis recurrence, so unrolling it by `nroots` was code size
  rather than speed.
- **Per-unit only.** `two_e_nroots_fusion` is `nroots_fusion_enabled() &&
  two_e_per_unit()`; the cooperative arm keeps one dispatch per order, for the
  measured reason in §15.4. `CINTX_2E_FUSE=off` / `set_two_e_nroots_fusion` is
  the A/B, and it is the one `gth_profile` variant that is a different compiled
  program rather than a kernel scalar.

Every quartet is still evaluated by the same code at the same comptime order,
accumulating the same terms in the same sequence into the same place. Only which
dispatch it rides in changes — the argument Task 35-M1 used when it merged
l-classes — so **the output is bit-identical**, and that is the gate.

### 15.3 Measured (CPU per-unit arm)

In-process A/B, `gth_profile`, `default` against `fuse=off`, three passes (best
of 3, 3 and 5). Both arms are prewarmed and interleaved:

| workload | fusion speedup (3 passes) | launches |
|---|---|---|
| H2O / DZVP-MOLOPT-SR | 1.48x, 1.41x, **1.42x** | 15 → 4 |
| CH4 / DZVP-MOLOPT-SR | 1.25x, 1.26x, **1.45x** | 15 → 4 |
| SO2 / DZVP-MOLOPT-SR | 1.01x, 1.08x, **1.28x** | 16 → 4 |
| H2O / TZVP-MOLOPT | 1.39x, 1.40x, **1.32x** | 15 → 4 |
| CH4 / TZVP-MOLOPT | 1.21x, 1.16x, **1.18x** | 15 → 4 |
| SO2 / TZVP-MOLOPT | 1.02x, 1.17x, **1.24x** | 16 → 4 |

**1.16x–1.48x**, and the spread is the host's, not the change's: the workloads
the model called balanced (SO2, 1.00x predicted) are exactly the ones whose
measured ratio wanders across the noise band, and the ones it called imbalanced
sit inside ±0.05x of each other across passes.

Whole-workload rows, two engines in one process on the identical list
(`gth_batched_throughput`, `artifacts/cintx_gth_throughput.json`):

| workload | quartets | libcint (s) | cintx (s) | now | §10.4 | §8.4 |
|---|---|---|---|---|---|---|
| H2O / DZVP-MOLOPT-SR | 406 | 0.036 | 0.015 | **2.39x faster** | 1.72x | 1.11x |
| CH4 / DZVP-MOLOPT-SR | 2 211 | 0.145 | 0.053 | **2.75x faster** | 2.25x | 1.58x |
| SO2 / DZVP-MOLOPT-SR | 1 035 | 0.163 | 0.069 | **2.38x faster** | 2.22x | 1.47x |
| H2O / TZVP-MOLOPT | 406 | 0.153 | 0.064 | **2.40x faster** | 1.65x | 1.18x |
| CH4 / TZVP-MOLOPT | 2 211 | 0.549 | 0.206 | **2.67x faster** | 2.10x | 1.58x |
| SO2 / TZVP-MOLOPT | 1 035 | 0.689 | 0.328 | **2.10x faster** | 1.98x | 1.51x |

Every `max|diff|` against the vendor is the §10.4 figure to the digit
(3.33e-15 … 2.65e-13), with 0 mismatched elements.

**def2 gains too, and that was not the target.** The def2 work lists are
segmented, so C1/K1 never touched them, but their *grouping* has the same shape
— and `def2_batched_throughput`'s in-process libcint ratios move:

| workload | launches | fuse=off | fused |
|---|---|---|---|
| H2O / def2-SVP (unscreened) | 15 → 4 | 1.05x **slower** | **1.62x faster** |
| H2O / def2-SVP (screened) | 15 → 4 | 1.42x | **1.95x** |
| CH4 / def2-SVP | 15 → 4 | 1.84x | **2.87x** |
| SO2 / def2-SVP | 16 → 4 | 2.48x | **2.70x** |
| H2O / def2-TZVP | 23 → 8 | 1.67x | **1.88x** |
| SO2 / def2-TZVP | 24 → 9 | 1.76x | 1.72x |

def2-TZVP fuses to 8–9 rather than 4 because its `nroots` 6 and 7 classes keep
their own dispatches; the SO2/def2-TZVP row is flat, which is the §15.1 model's
answer for a list of 181 070 quartets whose every group is already wide.

### 15.4 Why the cooperative (GPU) arm does not fuse

Measured on ROCm (gfx1151, H2O, `CINTX_2E_CHUNK_QUARTETS=64`, best of 3,
interleaved), with the fusion forced on for the cooperative arm:

| workload | fused (ms) | unfused (ms) | fusion | kl_split | G slab |
|---|---|---|---|---|---|
| H2O / DZVP-MOLOPT-SR | 173.2 | 183.9 | 1.06x | 25 | 3 692 KiB |
| H2O / TZVP-MOLOPT | 641.1 | 537.7 | **0.84x** | 26 (was 49) | 3 692 KiB |

Two mechanisms, both visible in the table. `kl_split_factor` sizes G1's
ket-pair split against a 64 MiB partial-**buffer** budget charged on the
*group's* whole output, so a four-times wider group buys half the split — 49
parts became 26, and G1 is worth 5.5–6.9x (§11.3). And a cooperative cube is
sized from the group's widest Cartesian block while its G slab is sized from the
group's widest class, so fusing hands an `(ss|ss)` quartet a `(dd|dd)`-shaped
cube and a 27 KiB slab. Both are fixable — a per-quartet split budget, a
per-class cube width — and neither is fixed here. The gate is one predicate,
`two_e_nroots_fusion`, and `def2_2e_batch_rocm_parity`'s launch-count assertion
now records that the *grouping* is a property of the decomposition (it was
`assert_eq!`; it is now "the fused arm merges dispatches and adds none").

### 15.5 Costs

- **Device G scratch.** A fused group's slab is sized to its widest class, and
  it now has as many slots as the whole list has quartets to fill. On the GTH
  rows the peak went 203 → 422 KiB (DZVP-SR) and 395 → 422 KiB (H2O/TZVP);
  SO2's 486/851 KiB is unchanged, because its widest group already had both.
- **Device Cartesian output.** Groups are dispatched one at a time and freed
  after readback, so the peak is the *widest* group — which fusion makes wider:
  SO2/def2-TZVP went 16.8 → 42.1 MiB. `plan_batch_bytes` takes that maximum
  from the same expression the allocation uses, so `memory_limit_bytes` still
  bounds it and the pre-flight refusal is still honest. Host peak is unchanged
  on every row (2.10–2.79x output).
- **Compiled code.** 15 programs of one Rys order became 4 of five orders, so
  the emitted bodies went 15 → 20 while the *dispatches* went 15 → 4.
  `prewarm_2e_work_list` reports 69 classes → 4 signatures in 51 ms.
- **K2 matters far more now.** `balance=uniform` was 0.90–1.00x before the
  fusion and is 0.56–0.80x after: with one dispatch per `(ibase, kbase)` the
  cost-balanced partition is the only thing standing between the run and its old
  imbalance. The two partitions are still bit-identical to each other.

### 15.6 Verification

- **Bit-identity, end to end**: `gth_profile` under `CINTX_GTH_DUMP` on `main`,
  then `CINTX_GTH_COMPARE` with the fusion in — **0 of 2 313 078 elements
  differ** across the six GTH workloads (`max|d| = 0.000e0`), and every variant
  still prints `=bits` against the run's own default.
- **Vendor**: `gth_batched_throughput` (0 mismatched elements at 1e-9, max|diff|
  as §10.4), `gth_contraction_ab` water gate and the full six under `--ignored`,
  `def2_2e_batch_parity`, `def2_integral_parity` (def2-SVP *and* def2-TZVP, so
  the unfused `nroots` 6–7 arm is covered), `general_contraction_device_indexing`.
- **Suites**: the whole `cintx-cubecl` unit suite (409 + 58 tests, including the
  f32 smoke launch at the new shape row and argument list), the whole
  `cintx-oracle` default suite, `two_e_cooperative_arm` (4), `kl_split_tests`,
  `partition_tests`, `def2_batch_memory_plan`, `def2_accumulator_ab`,
  `vrr_root_vector_ab`, `def2_pair_batch_parity`.
- **ROCm**: `def2_batch_rocm_parity::def2_2e_batch_matches_between_cpu_and_rocm`,
  `gth_contraction_ab` cross-backend on all six fixtures (staged beats naive
  1.04–1.30x; cpu-vs-rocm 43.7/576.1 eps of block scale on H2O, the same figures
  §11.3 recorded), `gth_profile` on ROCm (§15.4).
- **Pre-existing, and not from this pass**: `def2_batch_rocm_parity`'s
  `def2_pair_and_triple_batches_match_between_cpu_and_rocm` and
  `def2_derivative_batches_match_between_cpu_and_rocm` fail on `main` and still
  fail. §13 landed the vector VRR on the 3c2e/2c2e ROCm path for *both*
  decompositions, and at `nroots == 3` HIP refuses the kernel outright:
  `struct __align__(24) double_3` — "requested alignment is not a power of 2".
  §12.2's note that odd widths are "otherwise ordinary" is wrong for HIP. The 2e
  kernel is unaffected because its vector arm is comptime-gated to the per-unit
  shape, which is why `Const<3>` and `Const<5>` here are safe; the fix for the
  other four kernels is theirs to make.

### 15.7 What is left

- **The GPU fusion**, if it is wanted: a per-quartet ket-split budget and a
  per-class cooperative cube width, the two mechanisms §15.4 names.
- **H2O's remaining floor.** The launch floor is still 57–59% of the H2O rows
  (down from a *sum* of 15 dispatch floors to a single `(dd|dd)` quartet's serial
  walk, which is now the binding term at 80.2 M of a 1 158.5 M total). Splitting
  *that quartet* across units is G1 applied to the per-unit arm — the one lever
  §11.1 listed and left, and it is no longer bit-identical, so it needs the gate
  `ket_split_agrees_on_gth` already provides.
- **Fusing `ibase`/`kbase` as well** is worth 80.2 vs 81.0 M on the §15.1 model,
  i.e. nothing. Not to be revisited without a new reason.
- The contraction is still 62–76% of the per-unit kernel (`probe:no-ctr`), and
  §10.4's reading of it is unchanged: nine G loads and six flops per element per
  primitive quartet, no reuse for a private array to capture. Blocking the bra
  primitives to amortise the `gctri` read-modify-write was costed from the
  `naive`/`staged` A/B — 3 read-modify-writes into a cache-hot slab are ~5% of
  the row — and is not worth a pass.

## 16. F1 on the GPU: what actually blocked it (2026-09-09)

§15 landed the Rys-order fusion on the per-unit arm and left it off the
cooperative one, on a measurement (0.82x on ROCm/H2O-TZVP) and **two guesses**
about why. This section is the attribution §15.4 should have done, the fix it
points to, and the two defects that fell out on the way.

### 16.1 Attribution: one guess was wrong, the other was the whole story

`gth_profile` on ROCm (gfx1151, H2O, `CINTX_2E_CHUNK_QUARTETS=64`, best of 3,
interleaved), with `CINTX_GTH_FUSE_PROBE=1` adding the pinned arms. The fusion
is forced onto the cooperative shape and then each suspected mechanism is
pinned back to what the *unfused* dispatch would have had:

| variant | DZVP-SR (ms) | ratio | TZVP (ms) | ratio |
|---|---|---|---|---|
| `default` (unfused) | 158.26 | 1.000x | 474.40 | 1.000x |
| `fuse=on` | 153.53 | 1.031x | 577.27 | **0.822x** |
| `fuse=on,dim=64` | 150.57 | 1.051x | 577.22 | **0.822x** |
| `fuse=on,split=49` | 103.72 | **1.526x** | 368.23 | **1.288x** |
| `fuse=off,dim=64` | 144.07 | 1.098x | 484.26 | 0.980x |

Pinning the cube width moved TZVP by **0.000x** — §15.4's claim that fusing
"hands an `(ss|ss)` quartet a 256-lane cube" is true and *does not matter*.
Pinning the ket-pair split turned 0.822x into 1.288x, i.e. the fusion was
already worth 1.3x-1.5x and the split was giving it all back. The G-slab claim
in §15.4 was wrong too, and for a reason worth recording: `shared_g_enabled()`
is **off** by default, so the cooperative G tensor is in global memory and its
size costs an allocation, not occupancy.

One variable at a time, and one of the two candidates dies. This is what the
manual's step 3 is for and what §15.4 skipped.

### 16.2 Why a per-group split cannot serve a fused dispatch

`kl_split_factor` chose one part count for a whole group. That was survivable
while a group held one Rys order — a dispatch of `(dd|dd)` quartets is uniform,
so one number fits it — and F1 made groups *heterogeneous*. The fused
`(ibase, kbase)` dispatch carries an `(ss|ss)` quartet next to a `(dd|dd)` one
whose serial walk is two hundred times longer, and one number cannot serve
both: the occupancy term `target.div_ceil(n_quartets)` shrinks precisely
*because* the group got wider, and the memory term
`budget / group.output_bytes()` shrinks for the same reason. The
`(dd|dd)` quartet that sets the critical path went from 49 parts to 8.

### 16.3 What landed

**The split is per quartet.** `kl_split_plan` returns one count per quartet row:

- A cube's serial cost is its quartet's cost over its part count, so the plan
  equalises `cost / parts`. It takes the per-part cost that would spread the
  group over `KL_SPLIT_TARGET_CUBES_PER_UNIT` cubes per execution unit and
  gives each quartet `ceil(cost / that)` parts, bounded by **its own** ket range
  and by `KL_SPLIT_MAX`. Cheap quartets fall out at one part; the expensive few
  take the parts. If the partial buffers exceed
  `KL_SPLIT_PARTIAL_BUDGET_BYTES` the per-part cost doubles and the plan is
  recomputed, so the budget narrows the split rather than the run growing.
- The `quartet_cost_estimate` K2 already computes for the per-unit partition is
  what ranks them; no new cost model.

**The partial buffers are compact.** Part 0 of every quartet writes straight
into the group's output at the offset it already carried, and parts 1.. write
into a region past `out_len`, quartet by quartet. So the split costs
`Σ_q (parts_q − 1) · block_q` rather than `n_split · out_len`, and **a quartet
that is not split costs nothing at all** — no partial block, no copy. That is
what makes a per-quartet plan affordable as the default: on a fused dispatch
most quartets take one part.

**The reduce is table-driven.** Four `u32` per quartet — `[extra_base,
extra_parts, out_off, block_len]` — and one cube per quartet, its units striding
the quartet's Cartesian block. A quartet with `extra_parts == 0` is a copy,
which is why the table lists every quartet: the output leaves in its own buffer.
The accumulator starts from part 0's value in place and adds parts
`1, 2, …` in order, which is the same sequence of additions the per-group reduce
performed, so the result is as deterministic as it was.

**The fusion is on for both decompositions.** `two_e_nroots_fusion` no longer
consults the decomposition; `CINTX_2E_FUSE=off` / `set_two_e_nroots_fusion` is
the A/B on either.

### 16.4 Measured (ROCm, gfx1151, all six fixtures)

`gth_profile`, `default` against `fuse=off`, in-process and interleaved,
`CINTX_2E_CHUNK_QUARTETS=64`:

| workload | fused (ms) | `fuse=off` (ms) | **fusion** | launches | kl_split | G slab (KiB) |
|---|---|---|---|---|---|---|
| H2O / DZVP-MOLOPT-SR | 114.6 | 191.9 | **1.68x** | 24 ← 70 | 25 | 2 585 ← 1 269 |
| CH4 / DZVP-MOLOPT-SR | 562.2 | 742.9 | **1.32x** | 108 | 25 | 2 585 ← 1 269 |
| SO2 / DZVP-MOLOPT-SR | 424.3 | 668.5 | **1.57x** | 62 | 25 | 4 374 ← 3 615 |
| H2O / TZVP-MOLOPT | 416.1 | 514.2 | **1.24x** | 24 ← 70 | 49 | 3 191 ← 3 259 |
| CH4 / TZVP-MOLOPT | 1 950.7 | 2 218.0 | **1.14x** | 108 | 49 | 3 191 ← 3 259 |
| SO2 / TZVP-MOLOPT | 1 601.5 | 1 986.2 | **1.24x** | 62 | 49 | 7 712 ← 7 021 |

**1.14x–1.68x**, against 0.82x–1.03x before the split was fixed. The split now
reaches 25 and 49 — the *whole* ket range, the widest that is any use — on a
dispatch four times wider than the one that used to need that width, and the G
slab pays 2x for it on DZVP-SR and nothing on TZVP. For contrast, forcing the
old uniform split to 49 on a fused group (§16.1's probe row) cost 18.5 MB and
40 MB of G slab; the per-quartet plan gets the same split where it matters for
2.6 MB and 3.2 MB.

The per-unit arm is untouched: it never splits, and `gth_profile`'s dump
comparison against a pre-F1 dump is still 0 of 2 313 078 elements differing,
with the §15.3 A/B unchanged (1.08x–1.29x on this pass, 1.16x–1.48x over the
four passes now on record).

### 16.5 Two defects this pass found

**A wrong partial base, caught by the forced-split gate.** The first version
stored the reduce table's `extra_base` relative to the partial region while the
expanded rows carried it absolute. `ket_split_agrees_on_gth` — which pins the
cooperative arm on the CPU backend and forces splits of 3 and 8 — failed at
2.9e16 eps of block scale against a bound of 1 024. That gate exists for exactly
this and it is worth its four seconds.

**The autotuner ignored a pinned cube width.** `dispatch_2e_group` consulted
`should_tune` before checking whether the caller had pinned the geometry, so a
pinned width was a *suggestion*: the tuner would benchmark its candidates and
launch at whichever won. Nothing had noticed because a pinned dispatch had never
crossed `MIN_TUNE_ITEMS` (64 items) before — fusing the Rys orders makes a
cooperative dispatch four times wider, and `two_e_cooperative_arm`'s four-lane
arm crossed it for the first time. On the CubeCL CPU runtime each candidate
width is its own JIT compilation of the whole kernel, so a 1.5 s gate became a
20 minute one, which is how it was found. A pinned width is now an instruction:
`two_e_cube_dim_override().is_some()` short-circuits to the pinned geometry.
Every A/B that pins the width — that gate's four-lane arm, `gth_profile`'s
unit-count curve — depended on this and none of them could have told you.

### 16.6 Verification

- **Forced-split gate**: `ket_split_agrees_on_{def2,gth}` — def2 at 2 and 7
  parts bit-identical to the unsplit arm (its parts go empty), GTH at 3 and 8
  parts within 19.6 / 19.6 / 76.7 / 74.9 eps of block scale against a bound of
  1 024, and every arm within 1e-12 of the vendor. The GTH figures are §11.2's
  to the tenth of an eps, which is what a layout change that is only a *layout*
  change looks like.
- **Unit**: `kl_split_tests` rewritten for the per-quartet form — the block
  lengths come from the gaps between output offsets, the parts tile the ket
  range in order, an unsplit quartet beside a split one costs no partial block,
  and an all-ones plan is the identity on the rows.
- **CPU, unchanged**: `gth_profile` dump comparison against the pre-F1 dump
  (0 of 2 313 078), the six throughput rows (1.96x–2.44x faster than
  single-threaded libcint, every `max|diff|` the §10.4 figure), the whole
  `cintx-cubecl` unit suite and the whole `cintx-oracle` default suite.
- **ROCm**: `def2_batch_rocm_parity::def2_2e_batch_matches_between_cpu_and_rocm`,
  `gth_contraction_ab` cross-backend on all six (staged over naive 1.04x–1.63x;
  cpu-vs-rocm 49.1 / 577.4 eps of block scale on H2O, the §11.3 figures),
  `gth_profile` on ROCm (§16.4).
- **Still failing on `main` and still not ours**: the two `def2_batch_rocm_parity`
  cases §15.6 records, from §13's odd-width `Vector<f64, 3>` on the 3c2e/2c2e
  ROCm path.

### 16.7 What is left

- **The per-unit arm still never splits.** The `(dd|dd)` quartet that §15.7
  named as H2O's remaining floor is now split on the GPU and not on the CPU.
  The machinery is per-quartet and decomposition-agnostic; what stops it is that
  a split is not bit-identical, and the CPU arm's bit-identity against the
  pre-F1 dump is a claim worth keeping until someone wants the trade.
- **`KL_SPLIT_TARGET_CUBES_PER_UNIT` is still 8 and still unmeasured.** The plan
  now spends that target sensibly, which makes it worth sweeping.
- The cube width remains sized from the group's widest block. §16.1 says it
  costs nothing today; it is on record as measured-and-left, not overlooked.

## 17. The split on the per-unit arm (2026-09-09)

§16.7 left the ket-pair split cooperative-only, because a split is not
bit-identical and the per-unit arm's bit-identity against the pre-F1 dump was
worth keeping. It is enabled on both arms here. The mechanism needed nothing
new — the split had been decomposition-agnostic since §16 — but three things
around it did:

- **K2 partitions rows, so it needs a cost per row.** `TwoEGroupDispatch`
  carried the group's `quartet_cost`, one entry per *quartet*, and
  `per_unit_slot_bounds` slices it to `n_quartets`, which is the *row* count.
  With the split on that slice is the wrong length and the wrong shape.
  `expand_kl_split` now returns a `row_cost` beside the rows — a part's cost is
  its quartet's over the part count, because the parts tile the ket range in
  equal pieces and the estimate is linear in it.
- **The reduce had to stop copying.** It wrote into a fresh `out_len` buffer, so
  every quartet was copied whether or not it split — on a dispatch where two
  quartets take parts and four hundred do not, that is the whole output moved
  for nothing. It now accumulates the extras onto part 0 **in place** and the
  caller trims the partial region off the handle (`Handle::offset_end`), so an
  unsplit quartet is neither summed nor moved and the reduce table lists only
  the quartets that split.
- **The reduce had to parallelise on a plane-less runtime.** Its cube width came
  from `backend_plane_cube_dim`, which is *one unit* where there are no hardware
  planes; one unit striding every element of every quartet is a sequential pass
  over the whole output. It takes the per-unit width there.

`ket_split_agrees_on_{def2,gth}_per_unit` are the gates — the same forced splits
of 2/7 and 3/8 the cooperative cases use, on the other decomposition, which is a
different kernel shape (`lanes == 1`, no barrier, the staged contraction whole
inside one unit) reading the same tables. They report the same eps to the tenth
as the cooperative ones: 19.6 / 19.6 / 76.7 / 74.9.

What this bought on the CPU is **small** — 0.92x–1.23x over five passes, mostly
inside this host's noise band, and consistently positive only on the two H2O
rows, which are the two whose costliest quartet exceeds a `1/units` share of the
dispatch. That is what §15.1's model predicts and it is the honest figure. The
split earns its keep on the GPU (§18.4), not here.

## 18. One method, both backends (2026-09-09)

The instruction this section answers: *the processing methods for CPU and GPU
must be identical, and the kernels identical in principle.* Enabling the split
on the per-unit arm broke two contracts and exposed why — the method was not
one method.

### 18.1 What was not identical

`chunked_evaluation_is_bit_identical_to_unchunked` and
`tuned_and_untuned_dispatches_agree_bit_for_bit` failed the moment the CPU arm
split (the second by two ULP). Neither was a bug in the split. Both were the
same fact: **the split was sized from the group's cost total**, so it was a
function of how the work list happened to be batched.

| the split depended on | so |
|---|---|
| the group's summed cost | chunking changed the answer — a `memory_limit_bytes` run, or the same quartets in two calls instead of one, re-associated the ket sum differently |
| `hw.parallel_units` | a different machine changed the answer |
| the chunk cap in force | CPU and GPU split the same quartet differently, and so disagreed for a reason that was not the hardware |
| whether a budget was set | the split was switched off under `memory_limit_bytes`, so a budgeted run computed something different from an unbudgeted one |

### 18.2 What landed

**The split is a pure function of the quartet.** `kl_split_plan` takes no
client, reads no total and no unit count: `parts = ceil(cost /
KL_SPLIT_TARGET_PART_COST)`, bounded by the quartet's own ket range, by
`KL_SPLIT_MAX`, and by the partial blocks one quartet may add
(`KL_SPLIT_PARTIAL_BUDGET_BYTES_PER_QUARTET`, 1 MiB). The same quartet splits
the same way in every batch, on every backend, on every machine.

**The split runs under a memory budget.** It has to, or a budgeted run computes
different values. `plan_batch_bytes` charges it from the same expression
`run_2e_batches` allocates from — which it can, now that the plan needs no
client — as a `group_split_bytes` term counted **once**, because the partial
region is device-side only and the trimmed handle keeps it out of the readback.

**A budget that does not fit chunks harder instead of refusing.**
`chunk_cart_budget` is a heuristic and its first guess can plan over the limit
now that the split is charged to the same ledger. `plan_2e_stream` halves the
Cartesian budget and re-plans, down to `MIN_CHUNK_CART_BYTES` — one widest
Cartesian block, the point past which chunking cannot help — and only then
refuses, with the numbers from the tightest arrangement tried. The split itself
is never narrowed to fit: a budget may change the chunking, never the values.

### 18.3 A 100x regression, and the bug under it

The chunked variant went from 0.6x to **0.004x** — 4 026 ms against 14 ms —
the moment the split ran under a budget. The cause was not the split:

> `run_2e_batches` pre-sizes the shared G slab (M4.1) from `group.len()`, the
> **unsplit** quartet count, while `launch()` sizes the real geometry from the
> split row count. The moment anything splits, the shared slab is too small,
> every launch falls through to its own allocation, and M4.1's
> one-allocation-per-run quietly becomes one per launch — each a fresh
> multi-megabyte buffer and, because the cube width is part of a kernel's
> compiled identity, a fresh JIT with it.

That has been true on the **GPU since G1 landed** (§11), where `n_cubes` is
derived from the split rows and every cooperative dispatch has been paying it;
it only became visible when a CPU dispatch started splitting. Sized from the
same row count the launch uses, the chunked variant is 0.88x–0.99x — better than
the 0.59x–0.69x it managed before any of this.

### 18.4 The constant, swept

`KL_SPLIT_TARGET_PART_COST` is the one number the rule has, so it was swept
rather than guessed. In-process A/B against `klsplit=off`, best of 5:

| cost | H2O/DZ | CH4/DZ | SO2/DZ | H2O/TZ | CH4/TZ | SO2/TZ | ROCm SO2/DZ | ROCm SO2/TZ |
|---|---|---|---|---|---|---|---|---|
| 500 k | 1.06 | 1.07 | 1.09 | — | — | — | — | — |
| **1 M** | **1.23** | 1.01 | 1.10 | **1.22** | 1.08 | 0.96 | **1.64x** | **4.15x** |
| 2 M | 1.06 | 0.96 | 1.01 | 1.19 | 0.99 | 0.97 | 1.43x | 2.53x |
| 4 M | 1.03 | 1.02 | 1.01 | 1.23 | 0.98 | 0.98 | — | — |
| 8 M | 1.05 | 1.01 | 0.97 | 1.19 | 0.99 | 1.01 | — | — |

The CPU column is noise-dominated and only H2O/TZ separates the settings; the
GPU column does not — 1 M is 1.15x–1.64x better than 2 M there, because the
split is worth 1.6x–4.2x on ROCm against ~1.05x here. **1 M**, and the GPU chose
it.

(The ROCm absolutes in this section are not comparable to §16.4's: an hour of
sweeps had the APU at 922–971 MHz against a ~2.9 GHz boost. The in-process
ratios are, which is the whole reason the plan quotes ratios.)

### 18.5 What is identical now, and what is not

Verified, not asserted — every row is a `=bits` column or a dump comparison in
the same `gth_profile` run:

| the answer does not depend on | evidence |
|---|---|
| the dispatch grouping | `fuse=off` is `=bits` against the default on all six workloads, on **both** backends. Before §18 it was not, on either. |
| the chunking | `chunk=cart/4` is `=bits` against the unchunked default on all six |
| a memory budget | the same, since `chunk=cart/4` *is* the budgeted arm |
| the launch geometry | `tuned_and_untuned_dispatches_agree_bit_for_bit`, strict again |
| the decomposition | `cooperative_g_build_is_bit_identical_on_{def2,gth}` — per-unit and cooperative agree bit for bit on the same backend |
| the machine's unit count | the rule reads no hardware quantity |

What still differs between CPU and GPU is the **decomposition**, and every part
of it follows from one fact: the CubeCL CPU runtime makes a unit an OS thread
and `cube_count` a sequential loop, so the grid is not a parallelism axis there.
The list is enumerated on `two_electron_scalar_kernel` itself — barriers, the S3
`(axis, root)` split, the contraction's lane split, the vector VRR width, the
private accumulator's capacity, the cube geometry — six entries, each a
consequence of "a slot is one lane" versus "a slot is a cube", none a policy.
The two arms are held to bit-identity against each other, so the divergence is
in *how* the work is spread and not in what is computed; what is left between
the backends is the FMA contraction §8.3 records (49.1 / 577.2 eps of block
scale on H2O, the §11.3 figures to the tenth).

### 18.6 Costs

- **Device Cartesian residency** carries the partial region on both backends
  now: SO2/TZVP-MOLOPT 4.5 → 46.7 MiB, H2O/DZVP-SR 0.3 → 0.8 MiB. Host peak is
  unchanged everywhere (2.10–2.35x output), because the partials never reach the
  host. `memory_limit_bytes` bounds them through `plan_batch_bytes`, and
  `CINTX_2E_KL_SPLIT=off` removes them.
- **The floor a memory budget can reach rises** by roughly `(parts − 1)` copies
  of the widest Cartesian block, since the split is not narrowed to fit. The
  profile's `chunk=cart/4` budget moved from `output + cart/2` to
  `output + cart` for this reason, with the arithmetic in the harness comment. A
  caller who needs the older, lower floor turns the split off explicitly.
- **The default output is no longer bit-identical to the pre-F1 dump** on the
  CPU: 14–107 eps of block scale, every vendor `max|diff|` unmoved
  (3.44e-15 … 2.65e-13). The *unsplit* arm still is, exactly — 0 of 2 313 078
  elements — and `CINTX_GTH_DUMP`/`COMPARE` now work on that arm, so a change
  claiming bit-identity is still held to it.

### 18.7 What is left

- `KL_SPLIT_PARTIAL_BUDGET_BYTES_PER_QUARTET` (1 MiB) is not binding on any GTH
  or def2 class measured — the ket range caps first — so it is a guard, not a
  tuned value.
- The CPU gain is inside the noise on four of six workloads. If the per-unit
  split is ever found to cost more than it returns on a workload that matters,
  `CINTX_2E_KL_SPLIT=off` is the arm to compare against and §18.4 is the sweep
  to redo; nothing about the rule is CPU-specific, so turning it off there would
  reintroduce exactly the backend divergence this section removed.

## 19. The last host stage: the transform moves on-device (2026-09-09)

### 19.1 What was not on the device

Every batched 2e run ended on the host. The Cartesian buffer was read back and
`cart_to_sph_2e_into` transformed it there — the final stage of a GPU pipeline,
running on the CPU, moving *more* bytes across the bus than the answer needs
(Cartesian is larger than spherical). The device implementation has existed
since M3 (`c2s_device.rs`), gated behind `CINTX_2E_TRANSFORM=device` and left
off: "until it is measured on a backend where the readback is a real transfer".

It had been measured, and it was slower — 0.910x–0.988x on ROCm. The reason had
nothing to do with readbacks.

### 19.2 One work item was a whole quartet

The transform's work item was a quartet, and a quartet walks its
`nctr_i·nctr_j·nctr_k·nctr_l` contraction quads **serially**, each four
ping-pong axis passes over a block of up to 1 296 elements. On a segmented basis
that is one quad and the shape is fine. On TZVP-MOLOPT it is up to eighty-one,
and the worst row was exactly the most generally contracted one — SO2/TZVP at
0.910x, against 0.988x for the segmented-ish DZVP-SR rows.

That is the same latency shape G1 found in the 2e kernel itself (§10.5): plenty
of work per item, almost none of it parallel. The quads are independent — each
reads its own Cartesian block, uses its own ping-pong scratch and writes its own
spherical block — so the work item is now a **`(quartet, quad)` pair**, handed
out from a host-built table, exactly as `expand_kl_split` hands out ket-pair
parts. The four contraction indices come from inverting the layout the nest used
to build. A segmented work list has one quad per quartet and gets back precisely
the kernel it had.

### 19.3 The bug that found itself

The first run after the change was fast and **wrong**: `vendor|d|` of 3e-2 … 8e-1
on ROCm, against 1e-12. `def2_device_c2s_parity` had passed — it is a def2
fixture, `nctr == 1`, where the change is the identity.

`c2s_scratch_widest_len` (M4.3) sized the shared ping-pong slab from the group's
**quartet** count while `launch_c2s` sized its geometry from the item count.
This is §18.3's M4.1 bug again, one slab over — and worse, because this slab is
*written*: a short one is not a lost allocation but an out-of-bounds store. Both
now read the same item count. That two independent slabs had the same defect,
found two different ways, is the argument for deriving a slab's size from the
geometry expression rather than from a count that happens to match it.

### 19.4 The pre-flight plan was double-charging

`plan_batch_bytes` counts the Cartesian block twice — "once for the device
buffer and once for the host `Vec` its readback lands in". Under the device
transform there is no such `Vec`: the block is consumed where it was written and
only spherical comes back. It is charged once there, which is what let the
budgeted arms keep working once the transform became the default.

### 19.5 Measured

In-process A/B, `default` (device transform) against `xform=host`:

| workload | ROCm | CPU |
|---|---|---|
| H2O / DZVP-MOLOPT-SR | **1.012x** | 1.048x |
| CH4 / DZVP-MOLOPT-SR | — | 0.983x |
| SO2 / DZVP-MOLOPT-SR | — | 1.008x |
| H2O / TZVP-MOLOPT | **1.013x** | 1.070x |
| CH4 / TZVP-MOLOPT | — | 1.019x |
| SO2 / TZVP-MOLOPT | — | 1.015x |

Before the split into `(quartet, quad)` items the ROCm column was
0.910x–0.988x. It is now at or slightly past parity on both backends, and
`transform=0.0ms` — there is no host transform left to time. **The default
moved**: `CINTX_2E_TRANSFORM=host` is the opt-out.

**What it buys is the memory.** The host Cartesian intermediate is not allocated
at all:

| workload | host peak before | after |
|---|---|---|
| H2O / DZVP-MOLOPT-SR | 0.9 MiB (2.20x output) | 0.4 MiB (**1.00x**) |
| SO2 / TZVP-MOLOPT | 19.2 MiB (2.25x) | 8.5 MiB (**1.00x**) |
| SO2 / def2-TZVP | 269.4 MiB (2.79x) | 96.6 MiB (**1.00x**) |

Every workload, both bases, def2 included: **1.00x the spherical output**, where
M1's whole chunking apparatus existed to bound a 2.1x–2.8x peak. The readback
carries spherical rather than Cartesian (1.00x rather than 1.10x–1.25x the
output). The def2 throughput rows moved with it — H2O/def2-SVP screened 1.71x →
2.30x, CH4 2.51x → 3.35x, SO2/def2-TZVP 1.72x → 2.00x faster than
single-threaded libcint — and the GTH rows sit at 2.40x–2.82x.

### 19.6 Verification

- `def2_device_c2s_parity::the_device_transform_reproduces_the_host_bit_for_bit`
  and `negative_zero_survives_the_host_transform_convention`.
- `gth_profile`: `xform=host` is `=bits` against the device default on all six
  GTH workloads, so the two transforms agree exactly on the CPU backend; on ROCm
  they differ by the FMA the device kernel introduces (3.44e-15 against
  3.33e-15, inside 1e-12).
- The §18.5 invariants re-checked with the new default: `fuse=off` and
  `chunk=cart/4` both `=bits`, the unsplit arm still bit-identical to the pre-F1
  dump, `chunked_evaluation_is_bit_identical_to_unchunked`,
  `tuned_and_untuned_dispatches_agree_bit_for_bit`.
- `def2_batch_rocm_parity`, `gth_contraction_ab` cross-backend, both throughput
  artifacts, and both full suites.

`gth_profile`'s chunked variant now runs the **host** transform deliberately, and
says why in its own comment: chunking exists to bound the host Cartesian
intermediate, the device transform removes that intermediate outright, and the
two are alternative answers to the same problem. With the device transform its
budget would be spent instead on the ping-pong scratch, which is per-slot times
the widest block and does *not* shrink with the chunk.

### 19.7 What is still on the host, and why

- **Planning and marshaling** — the pair table, K1's Cartesian index tables, K2's
  partition bounds, the split's row expansion and the transform's item table.
  That is where the project's architecture constraint puts host work, libcint
  builds its own `idx` on the host for the same reason, and all of it is
  `O(quartets)` against `O(primitive quartets · block)` of device work.
- **`deriv34` and the `grids` family** compute on the host by documented design:
  the bra/ket headroom elevates the nuclear Rys order past
  `MAX_DEVICE_NROOTS = 5`, which the comptime device kernel cannot serve.
  Those are the remaining "GPU family that is not on the device", and moving
  them needs the extended-order solver wired into those kernels, not a switch.
