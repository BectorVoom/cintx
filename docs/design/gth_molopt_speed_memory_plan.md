# GTH-MOLOPT (DZVP-MOLOPT-SR / TZVP-MOLOPT) Speed and Memory Plan

Status: executed 2026-09-06 — §8 is the record of what landed and what was measured.
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
