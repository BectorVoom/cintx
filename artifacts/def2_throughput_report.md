# def2-SVP / def2-TZVP on cintx: correctness and throughput report

**Date**: 2026-08-24
**Build**: release (`opt-level = 3`), CubeCL `cpu` backend, vendored libcint 6.1.3 (`cc -O3`)
**Host**: x86_64 Linux, 16 cores

This report supersedes the methodology of `artifacts/speed_benchmark_report.md`, which measured
**one shell tuple at a time**. That is the wrong unit: an SCF iteration needs the whole shell-quartet
list, and per-tuple numbers hide both the launch overhead and the amortization.

---

## 1. Correctness

### 1.1 Basis-set construction (Phase 32)

def2-SVP, def2-TZVP and def2-ECP are vendored from the Basis Set Exchange (v0.12, Turbomole 7.3
data) into `crates/cintx-basis`. Normalization reproduces libcint's two stages exactly
(per-primitive `CINTgto_norm`, then PySCF's per-contraction self-overlap renorm).

The gate is the **vendor itself**, not a recorded fixture: a correctly normalized contracted AO has
unit self-overlap by construction, so vendored libcint must report `S_ii == 1` for every AO.

| Check | Result |
|---|---|
| `S_ii == 1` under vendored libcint, all fixtures (H2O, CH4, Fe) x {SVP, TZVP} | PASS (< 1e-12) |
| `gto_norm` vs vendor `CINTgto_norm` FFI, l = 0..4 | PASS (< 1e-13) |
| Every element in both catalogs builds `BasisSet` + raw arrays, max l <= 4 | PASS |
| `env` coefficients differ from raw catalog (proves normalization actually runs) | PASS |
| cintx-basis ABI slot constants == `cintx_compat::raw` | PASS |

Composition checks, pinned: H2O/def2-SVP = 12 shells / 24 spherical AOs;
H2O/def2-TZVP = 19 shells / 43 spherical AOs; def2-TZVP oxygen reaches f (l = 3).

### 1.2 One-electron integrals

`int1e_ovlp_sph` over **every** shell pair of H2O in both bases, against vendored libcint:

| Basis | Pairs | Mismatched | Max abs diff |
|---|---|---|---|
| def2-SVP | 144 | 0 | ~3e-16 |
| def2-TZVP | 361 | 0 | ~6e-16 |

All l-classes through (3,3) agree at machine precision.

### 1.3 Two-electron integrals — **a real bug found and fixed**

Walking one representative per angular-momentum launch class over H2O/def2-SVP (69 classes)
against vendored libcint:

**Before the fix: 66 / 69 classes correct, 3 classes WRONG.**

| Class | `nroots` | Max abs diff | Mismatched elements |
|---|---|---|---|
| `[1, 1, 2, 1]` | 3 | 1.86e-1 | 31 / 135 |
| `[1, 2, 2, 1]` | 4 | 1.32e-1 | 51 / 225 |
| `[2, 2, 2, 1]` | 4 | **1.17e+1** | 55 / 375 |

**Root cause** — `crates/cintx-cubecl/src/kernels/two_electron.rs`, device `kj2d` HRR branch
(`ibase == 0 && kbase == 1`), second transfer loop:

```rust
while n < di { ... }      // was
while n < dk { ... }      // libcint g2e.c:552, and the cintx host hrr_kj2d_4d, both use dk
```

With `ibase == 0`, `di == nroots` and `dk == nroots * (li + 1)`, so the loop silently under-writes
every `i >= 1` plane of the G-tensor. The failure condition is exactly
`ibase == false && kbase == true && li >= 1 && ll >= 1`, which accounts for all three failing
classes and none of the 66 passing ones.

**Why it was never caught**: the branch's only device unit test is `(s,s,p,s)` — `li == 0`, where
`dk == di` makes the bug invisible — and every `ll == 0` class skips the loop entirely. The host
fallback (`nroots > 5`) has always been correct, so high-angular-momentum paths masked it too.

**After the fix: 69 / 69 classes correct.**

This is a correctness defect in ordinary closed-shell ERIs at def2-SVP quality — the most commonly
used production basis — and it was found only because a real basis set was driven through a
class-complete sweep.

---

## 2. Throughput

Both engines run the **identical** work-list (same enumeration, same Schwarz screen), and a speed
verdict is printed only for a run whose values match the reference.

### H2O / def2-SVP, 236-quartet sample of the 3 081-quartet 8-fold list

| Engine | Wall (s) | Quartets/s | Integrals/s |
|---|---|---|---|
| libcint 6.1.3 (C, 1 thread) | 0.0003 | 745 929 | 2.02e7 |
| cintx CubeCL (cpu backend) | 125.41 | 1.9 | 5.10e1 |

**cintx is ~390 000x slower than libcint on this workload.**

Supporting measurements:

- **Warm-up**: 42.3 s for 69 launch classes = **613 ms per class**. This is the CubeCL
  specialization/compile cost, paid once per class.
- **Steady state**: ~530 ms *per quartet* — essentially the same as the warm-up per-class cost, so
  **the cost is not amortizing**. Every quartet pays it.
- **Launch tiers** (from the G-tensor footprint): 226 thread-per-quartet, 10 cube+shared,
  0 cube+global — def2-SVP fits entirely inside the shared-memory tier, as predicted.
- **Device eligibility**: 236 / 236 quartets have `nroots <= 5`, so none of this is a host-fallback
  artifact. This is the device path at full speed.

### Why

The per-quartet dispatch model launches one kernel with `CubeCount::Static(1, 1, 1)` — **a single
cube, i.e. one work-item** — and that single work-item serially executes the entire primitive
loop. For a def2-SVP oxygen quartet that is up to 7^4 = 2 401 primitive quartets, each building a
G-tensor of up to 1 125 elements: roughly 2.7 M operations, serialized, with zero parallelism,
plus a fresh coefficient upload and a blocking readback per quartet.

The earlier STO-3G figure of 1 124 us/quartet was measured on 3-primitive s/p shells. A real basis
has more primitives and higher angular momentum, and the cost scales with both.

### Screening

At `tolerance = 1e-10`, H2O/def2-SVP keeps **100%** of its 3 081 quartets — a 3-atom molecule is
too compact for Schwarz screening to bite. Screening pays off on extended systems; it is
implemented and gated (`tolerance = 0` is verified to be the exact identity) but it is not a
source of speedup at this molecule size, and is reported separately for that reason.

---

## 3. What this means for "beat libcint"

The gap is **architectural, not numerical**. No kernel-body tuning closes 390 000x. The unit of
work has to change from one quartet to a bucket of thousands, with:

1. one launch per angular-momentum class rather than per quartet (69 launches, not 3 081);
2. a grid of cubes — `cube_count_1d(n_quartets)` — rather than `Static(1,1,1)`;
3. the basis resident on device instead of re-uploaded per quartet;
4. one collective readback per bucket.

`crates/cintx-driver` implements the host half of this (enumeration, screening, bucketing, tiering,
statistics) behind a pluggable `QuartetEvaluator`. The fused multi-quartet kernel is **not**
implemented and is the remaining work.

**On the CPU backend, parity with libcint is the realistic ceiling** — same silicon, and libcint's
per-quartet C is already near-optimal. `cintx-simd` is the honest CPU comparator and already
reaches 0.86x-1.30x. A throughput win over libcint should be expected only on a GPU backend, and
only after the batched kernel exists.

---

## 4. Reproducing

```sh
# correctness
CINTX_ORACLE_BUILD_VENDOR=1 cargo test --release -p cintx-oracle --features cpu \
  --test def2_normalization_parity --test def2_2e_class_diagnostic -- --nocapture

# throughput
CINTX_ORACLE_BUILD_VENDOR=1 CINTX_BENCH_CAP=240 \
  cargo test --release -p cintx-oracle --features cpu \
  --test def2_throughput_benchmark -- --ignored --nocapture
```

`CINTX_BENCH_CAP` bounds the sample (default 240); `CINTX_BENCH_SCOPE=full` adds CH4 and
def2-TZVP.
