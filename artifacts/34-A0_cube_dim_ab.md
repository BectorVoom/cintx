# Task 34-A0 — cube-dimension A/B for the scalar 2e kernel

**Date**: 2026-08-24
**Backend**: CubeCL `cpu` runtime (cubecl-cpu 0.10.0), 16-core AMD Ryzen AI 7 350
**Harness**: `cintx-cubecl::kernels::two_electron::cube_dim_ab::two_e_cube_dim_ab`
(release, `--test-threads 1`, machine otherwise idle)

## Measurement

Steady-state milliseconds for one shell quartet (warm-up excluded), at a pinned
`CINTX_2E_CUBE_DIM`:

| l-tuple | nprim/shell | nroots | block | **dim=1** | dim=16 | dim=64 |
|---|---|---|---|---|---|---|
| (s,s,s,s) | 7 | 1 | 1 | **0.211** | 5.983 | 130 236.751 |
| (p,p,p,p) | 4 | 3 | 81 | **0.298** | 23.148 | 14 436.302 |
| (d,d,d,d) | 1 | 5 | 1296 | **0.066** | 2.811 | 75.927 |
| (d,d,d,d) | 3 | 5 | 1296 | **0.785** | 4.911 | (not run) |

`dim=256` was not run to completion: the `dim=64` (s,s,s,s) case already costs
**130 s for a single quartet**, and cost grows super-linearly in the cube
dimension.

## Verdict

**A larger cube is not merely useless on the CPU backend — it is the dominant
cost of the entire 2e path.** `CubeDim = 1` is between **28x** and
**~4.9e5x** faster than the shipped `standard_plane_cube_dim()` (256).

This inverts the premise of Task 34-A as written.

## Why — the backend's execution model

Read from `cubecl-cpu-0.10.0`:

- `compute/runner.rs::execute_data` spawns **one OS thread per cube unit**
  (`for unit_pos_x in 0..cube_dim.x { ... worker.send_task(...) }`), extending
  the worker pool past `available_parallelism` when `cube_dim` demands it. At
  `CubeDim=256` on 16 cores that is 16x oversubscription, plus 256
  `MlirData::clone()` per launch.
- `compute/compute_task.rs::sync_cube` is a **global spin-wait barrier**
  (`std::hint::spin_loop()` on two atomics) across *all* units. With 256
  oversubscribed threads, every barrier costs a full scheduler round.
- The 2e kernel executes `sync_cube()` **twice per primitive quartet**. A
  (s,s,s,s) def2-SVP quartet has 7^4 = 2401 primitive quartets → 4802 barriers,
  each rendezvousing 64–256 spinning threads. 4802 x ~27 ms is the 130 s above.
- `compiler/visitor/mod.rs` lowers `cube_count` to a **sequential `scf.for`
  loop inside each unit's kernel**. On the CPU backend the grid is *not* a
  parallelism axis at all; `cube_dim` is the only one.

The 255 units were never doing useful work in the first place: the Rys roots
and the whole VRR+HRR G-tensor build run under `if UNIT_POS == 0`, and only the
contraction block (`q_elem % CUBE_DIM == UNIT_POS`) is distributed. So the cube
bought a fraction of one block-length loop and paid two oversubscribed spin
barriers per primitive quartet for it.

## Consequences for the plan

1. **34-A is re-scoped.** "Distribute the VRR/HRR build across the cube" would
   add `nmax+mmax` plus `li+ll` further barriers per primitive quartet. On the
   CPU backend that is strictly worse — the barrier-count check the plan asked
   for comes back negative before a line is written. The cooperative build
   remains the right shape for a GPU backend, where `sync_cube` is a workgroup
   barrier and the grid is real parallelism, so it is kept behind a backend
   choice rather than deleted.
2. **The landed change is the backend-aware cube dimension**: CPU launches the
   2e kernel with `CubeDim = 1`; GPU backends keep a contraction-width cube.
3. **34-B still matters, for a different reason.** On CPU it cannot add
   parallelism (the grid is a sequential loop), but it removes per-quartet
   buffer creation, kernel dispatch and blocking readback — which is what the
   remaining per-quartet cost is made of once the barriers are gone.
4. **This is not 2e-specific.** `single_cube_count()` has 43 callers and
   `standard_plane_cube_dim()` 37 across the kernel crate. Every one of them
   pays the same cost on the CPU backend. That is the likeliest explanation for
   the 30–160x 1e gap Phase 35 was written to close, and it makes Phase 35
   mostly a repeat of this one-line finding rather than new kernel work.

---

## Outcome after landing the backend-aware cube dimension

`crate::plane::cooperative_cube_dim::<R>()` now returns `1` on the CPU runtime
and a contraction-width cube on GPU runtimes; `two_electron.rs` launches the
scalar 2e kernel through it.

**Correctness** — `def2_2e_class_diagnostic`: **69 / 69 classes OK, 0
mismatches**, max |diff| 2.665e-15 (unchanged from before the edit).

**Whole-workload benchmark** (`def2_throughput_benchmark`,
H2O / def2-SVP, 236-quartet sample, same work-list on both engines):

| | before (CubeDim 256) | after (CubeDim 1 on CPU) | factor |
|---|---|---|---|
| cintx wall, 236 quartets | 125.4 s | **0.0086 s** | **14 600x** |
| per quartet | ~530 ms | **~36 us** | 14 600x |
| warm-up per launch class | 613 ms | 58 ms (first case), ~0 ms warm | 10x+ |
| class-diagnostic gate | 31.6 s (7 m 20 s CPU) | **4.04 s** (3.9 s CPU) | 7.8x wall, 113x CPU |
| vs libcint | 390 000x slower | **58x slower** | |

The gap is now **58x**, not 390 000x, and its composition has changed
completely: at ~36 us/quartet against libcint's ~0.6 us, what remains is
per-launch overhead — 12 buffer allocations, one dispatch and one blocking
readback per shell quartet — not integral arithmetic. That is precisely what
34-B (grid over quartets), 34-C (device-resident basis) and 34-E (collective
readback) remove.
