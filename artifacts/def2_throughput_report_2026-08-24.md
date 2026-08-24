# def2 throughput — 2026-08-24 session

**Host**: AMD Ryzen AI 7 350 (16 cores), 30 GB RAM, Fedora, Linux 7.1.9
**Backend**: CubeCL `cpu` runtime (cubecl-cpu 0.10.0) — *not* a GPU
**Reference**: vendored libcint 6.1.3, C, `-O3`, **single-threaded**
**Basis**: def2-SVP (Basis Set Exchange v0.12 / Turbomole 7.3)
**Harness**: `def2_throughput_benchmark`, release, machine otherwise idle

Both engines run the **identical** screened work-list produced by the same
`SchwarzTable`, and no speed number is printed for a run whose values do not
match the reference.

---

## Headline

| | per quartet | vs libcint |
|---|---|---|
| Session start (256-unit cube, one launch per quartet) | ~530 ms | 390 000x slower |
| After 34-A (backend-aware cube dimension) | ~36 us | 58x slower |
| After 34-B/34-C (one dispatch per class, one basis upload per run) | **~1.7 us** | **2.6x slower** |

**~310 000x** end to end, with results byte-identical to vendored libcint
throughout (max abs diff 2.7e-15) and **bit-identical** to cintx's own
per-quartet path.

---

## Batched results

`def2_batched_throughput`, steady state (warm-up reported separately):

| Case | quartets | launches | us/quartet (cintx) | us/quartet (libcint) | ratio |
|---|---|---|---|---|---|
| H2O / def2-SVP, unscreened | 3 081 | **69** | 2.43 | 0.81 | 3.0x slower |
| H2O / def2-SVP, screened 1e-10 | 3 081 | **69** | 2.28 | 0.79 | 2.9x slower |
| CH4 / def2-SVP, screened 1e-10 | 14 706 | **69** | **1.70** | 0.65 | 2.6x slower |

- **Launches**: 69 per case — one per angular-momentum launch class — against
  3 081 and 14 706 for the per-quartet path. Readbacks likewise: 69, not one
  per quartet.
- **Transfers**: 60 KiB (H2O) / 288 KiB (CH4) total, basis included — one
  basis upload for the whole run plus the per-class quartet tables (34-C, in-call
  half; down from 116 / 370 KiB when the basis was re-uploaded per class).
  Retaining the basis across *separate calls* in `DeviceResidentCache` is still
  open.
- **Accuracy**: max abs diff vs vendor 2.665e-15 … 3.331e-15, zero mismatched
  elements, in every case.
- **Warm-up**: the first call pays CubeCL's per-class specialization —
  4.21 s for 69 classes on a cold process, then ~8.5 ms warm. That cost is
  per-process and per-class, not per integral.

## What produced each step

**34-A — backend-aware cube dimension.** Every cooperative kernel launched a
256-unit cube. On the CubeCL CPU runtime a cube unit is an **OS thread**,
`sync_cube` is a **global spin barrier**, and `cube_count` lowers to a
sequential loop inside each unit. The 2e kernel synchronises twice per
*primitive* quartet, so an `(s,s|s,s)` def2-SVP quartet paid 4 802 barriers
across 256 threads oversubscribed 16x on 16 cores. Measurement and derivation:
`artifacts/34-A0_cube_dim_ab.md`.

**34-B — one dispatch per launch class.** With the barriers gone, the remaining
~36 us/quartet was per-call overhead: twelve buffer allocations, a dispatch and
a blocking readback for every quartet. The kernel is now batched — flattened
basis plus a `[si, sj, sk, sl, out_off]` index table, per-cube G-tensor slab,
kernel-local Rys roots — so a whole class is one dispatch and one readback.

## Honest reading of the remaining 2.6x

- Both sides are **single-threaded** here. libcint at 0.64-1.12 us/quartet on
  this machine is a fair, un-handicapped reference.
- The CPU runtime executes the batch in **one** unit (its `cube_count` is a
  sequential loop, so the cube dimension is the only parallelism axis and the
  kernel's cooperative structure has `sync_cube` inside the primitive loop).
  Fifteen of sixteen cores are idle during the measurement. A CPU-specific
  "one quartet per **unit**" kernel mode — no barriers, per-unit G slab — is
  the obvious next step and is not done.
- Nothing here is a GPU result. This host's Radeon 860M (gfx1151, 8 CUs,
  integrated) is not a credible f64 throughput target against a 16-core CPU;
  open question 1 in the plan (which backend is the throughput target) is still
  open, and "beat libcint" still needs an answer to it.
- Cross-call device residency (the rest of 34-C), 34-D (primitive screening),
  34-F (public API) and the `int1e`/`int2c2e`/`int3c2e` batching of Phase 35 are
  **not** done, so this is not the floor.
