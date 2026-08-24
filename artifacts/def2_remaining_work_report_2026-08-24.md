# Remaining-work execution report

**Plan**: `.planning/notes/def2-remaining-work-PLAN.md`
**Date**: 2026-08-24 (continuation session)
**Host**: 16 hardware threads, AMD Radeon 860M (gfx1151) present but unused —
every number below is the CubeCL **CPU** backend against vendored libcint 6.1.3
running single-threaded.

---

## 1. What this session closed

| Plan item | Outcome |
|---|---|
| Phase 33 prerequisite — backend feature gating | **done** — all five backend-only profiles compile |
| Part 4-02 — feature matrix | **done** — plus a CI leg that actually builds them |
| CPU "one quartet per unit" kernel mode | **done** — 2e, and extended to 1e / 2c2e / 3c2e |
| 34-C — device-resident basis, cross-call half | **done** — `ResidentTwoEBasis` |
| 34-D — primitive-quartet screening | **done** — `TwoEBatchOptions::primitive_tolerance` |
| 34-F — public API | **done** — `QuartetBatchRequest` / `evaluate_shell_quartets` |
| Phase 35 — batching for `int1e_*`, `int2c2e`, `int3c2e` | **done** — 13x-140x |
| Part 4-03 — clippy | **deny-level clean**; ~2.7k warnings remain (see §5) |
| Two latent correctness bugs | **found and fixed** (see §2) |

Gate state at the end of the session:

```
cargo test --workspace --exclude cintx-oracle   28 binaries, 635 passed, 0 failed
CINTX_ORACLE_BUILD_VENDOR=1 cargo test --release -p cintx-oracle --features cpu
                                                95 binaries, 358 passed, 0 failed
cargo clippy --workspace --all-targets          0 errors
cargo fmt --all --check                         clean
xtask manifest-audit --check-lock               status: ok
```

---

## 2. Two correctness bugs, found by asking the question the fixtures could not

The predecessor session found two general-contraction / prefactor bugs on the
**host** Rys fallback. `general_contraction_device_indexing` had settled the same
question for the device 2e path. Nothing had asked it of the other families.

Extending that test to `int2c2e`, `int3c2e` and the `int1e_*` trio — one shell
with `nprim = 3, nctr = 2`, low enough `l` to stay on the device path — turned up
two failures immediately:

| Family | Symptom | Cause |
|---|---|---|
| `int2c2e` | 2.775e1 absolute error | the device kernel summed **every** `c_i·c_k` product into one scalar and wrote a single `nci*nck` block, instead of one block per contraction pair |
| `int3c2e` | 5.407e0 absolute error | the same collapse over `c_i·c_j·c_k` |

Both are correct only when every `nctr == 1`. def2-SVP and def2-TZVP are fully
segmented, so no def2 fixture could see it; it would corrupt any generally
contracted basis (cc-pVXZ, ANO).

The fix is the layout the 2e kernel and `center_3c2e_ip1_kernel` already used:
the Cartesian value is computed once and scattered across the contraction
blocks, weighted per block. Two further defects fell out of it, because the
*transform* side had the matching assumption:

- `int2c2e`'s representation dispatch transformed only the first block and copied
  it linearly to staging — so even the (already correct) host `nroots > 5` arm
  lost its general-contraction result. Now scattered per block, `c*n+m` AO index.
- `int3c2e`'s `swap_ij` transpose ran over the whole buffer as one block, and its
  transform did too. Now the contraction index is swapped alongside the shell
  indices, and each block is transposed and transformed separately.

Spinor 2c2e/3c2e general contraction is not wired through the spinor transforms;
those arms now fail closed with a typed `UnsupportedApi` rather than silently
transforming block zero.

`general_contraction_device_indexing` now covers all five families, and
`general_contraction_3c2e_batch_matches_vendor` covers the combination the def2
fixtures cannot reach at all: `nctr > 1` **and** both sides of the `swap_ij`
canonicalization, through the batched path.

---

## 3. Throughput

### 3.1 Where the time actually went

Instrumenting `evaluate_2e_quartet_batch` split the batched run in two, and the
split is now carried in `BatchExecutionStats` (`dispatch_ns`, `host_transform_ns`)
rather than being printed and discarded:

| H2O / def2-SVP, screened | before | after |
|---|---|---|
| backend dispatch | 3.45 ms | 2.9 ms |
| host cart→sph | 1.75 ms | 0.60 ms |

Solving `T = n·k + launches·c` across the H2O (3081 quartets) and CH4 (14706
quartets) points gave **k ≈ 0.18 us/quartet of arithmetic and c ≈ 42 us of
per-launch overhead**. At 69 launches that overhead was 55% of the H2O run —
i.e. the kernel was already at libcint's speed and the *launch* was the cost.

Two things followed from that number, and both are in this session's changes:

1. **`cart_to_sph_2e` was rewritten.** It allocated four `vec!`s per call and ran
   all four axis transforms unconditionally. `C2S_L0` and `C2S_L1` are identity
   matrices, so an s/p axis is a copy — and a def2-SVP work list is mostly s and
   p. The new `cart_to_sph_2e_into` skips identity axes, ping-pongs through a
   caller-owned scratch, and writes a caller-owned output. Host transform time
   fell 2.9x on H2O and 2.5x on CH4.
2. **The per-unit width is sized by the work available**, not by
   `available_parallelism` alone (§3.3).

### 3.2 One quartet per unit

Task 34-A0 established the CubeCL CPU runtime's shape: a cube unit is an OS
thread, `sync_cube` is a global spin-wait, and `cube_count` lowers to a
sequential `scf.for` *inside* each unit. The grid is therefore not a parallelism
axis there and the cube is the only one — so the way to use it is to give each
unit a whole quartet rather than a slice of one quartet's contraction.

The kernel now carries a comptime `per_unit` flag selecting between:

- `per_unit == 0` — one quartet per **cube**, the cube cooperating on it, two
  `sync_cube()` per primitive quartet. The GPU shape, unchanged from 34-B.
- `per_unit == 1` — one quartet per **unit**, cooperative group of one, and the
  barriers *comptime-removed*. They have to be removed rather than skipped:
  units walk different quartets, so their trip counts differ and any barrier
  inside the quartet loop is divergent.

Two details that mattered more than expected, both about cache lines rather than
arithmetic:

- **G slabs are padded to 64 bytes.** Unpadded, a low-`l` class puts consecutive
  slots' G tensors a few words apart — and the G tensor is written in the
  innermost loop.
- **The per-unit walk is blocked, not interleaved.** Neighbouring quartets write
  neighbouring `cart_out` blocks; an interleaved assignment put every unit's
  accumulation on the same handful of lines.

### 3.3 Sizing the unit count

Waking a unit costs ~2 us per unit per launch (mpsc dispatch + a per-unit clone
of the binding table in `cubecl-cpu`). Splitting a class across more units than
its work can fill pays that for nothing. Measured on H2O/def2-SVP `int2c2e`
(~16 pairs per class): 16 units was **3x slower** than 4.

`plane::per_unit_width(n_items, min_items_per_unit, by_memory)` now sizes it, and
the per-family constant is where the difference lives — a 2e quartet runs
`nprim^4` primitive quartets through a full VRR/HRR build and dwarfs the
dispatch (min 1); a 1e/2c2e/3c2e tuple does not (min 4).

### 3.4 Results

**2e** — H2O and CH4 def2-SVP, whole screened work list, best of 9:

| Case | quartets | launches | us/quartet | vs libcint |
|---|---|---|---|---|
| H2O / def2-SVP, screened | 3 081 | 69 | 1.17 | 1.43x slower |
| CH4 / def2-SVP, screened | 14 706 | 69 | **0.52** | **1.28x FASTER** |

Byte-agreement with vendored libcint unchanged: max abs diff 2.7e-15,
0 mismatched elements.

Session-over-session for 2e: 1.70 → 0.52 us/quartet on CH4, and 2.6x slower than
libcint → 1.28x faster.

**Phase 35** — H2O/def2-SVP and def2-TZVP, best of 9, batched vs the per-tuple
CubeCL route both sides checked against libcint before timing:

| Family | tuples | launches | per-tuple | batched | speed-up | vs libcint |
|---|---|---|---|---|---|---|
| `int1e_ovlp_sph` (SVP) | 144 | 9 | 3.56 ms | 0.28 ms | **12.9x** | 14x slower |
| `int1e_kin_sph` (SVP) | 144 | 9 | 3.84 ms | 0.30 ms | **12.8x** | 10x slower |
| `int1e_nuc_sph` (SVP) | 144 | 9 | 4.68 ms | 0.28 ms | **16.6x** | 6x slower |
| `int1e_ovlp_sph` (TZVP) | 361 | 16 | 10.5 ms | 0.43 ms | **24.2x** | 9x slower |
| `int2c2e_sph` (SVP) | 144 | 9 | 3.54 ms | 0.19 ms | **18.3x** | 4x slower |
| `int3c2e_sph` (SVP) | 1 728 | 27 | 37.2 ms | 1.40 ms | **26.6x** | 2.4x slower |

The plan's Phase 35 acceptance was "byte-identical to vendor; >= 10x current
CubeCL throughput on 1e". Both hold, with margin, and 3c2e — the family RI-J
actually spends its time in — is now within 2.4x of libcint.

---

## 4. API surface added

All of it is bit-identical to the per-tuple route it replaces; the single-tuple
entry points were rewritten as one-tuple batches so every pre-existing parity
test now covers the batched kernel too.

**`cintx-rs` (safe facade, no CubeCL types):**

- `QuartetBatchRequest` / `QuartetBatchOutput`
- `evaluate_shell_quartets`, `evaluate_shell_quartets_in`

Scope-gated to `int2e_sph` + `Spheric` + `F64` before any device work: a batch
path that silently accepted `int2e_ip1_sph` would return undifferentiated
integrals under a derivative operator's name.

**`cintx-cubecl` (backend surface):**

- `ResidentTwoEBasis` — a basis uploaded once and kept on the device across
  calls, bound to the backend arm it was uploaded on.
- `TwoEBatchOptions { primitive_tolerance }` — default `0.0` is exact.
- `evaluate_2e_quartet_batch{,_resident,_with}`
- `evaluate_1e_pair_batch` + `OneEOperator`, `BatchAtom`
- `evaluate_2c2e_pair_batch`
- `evaluate_3c2e_triple_batch`
- `BatchExecutionStats` gained `basis_upload_bytes`, `dispatch_ns`,
  `host_transform_ns`.

### 34-C acceptance

`resident_basis_uploads_once_and_changes_nothing` asserts both halves, because
either alone is satisfiable by a bug — a residency that quietly re-uploads passes
the value check, and one that reads stale device memory passes the byte count:

```
first  evaluation: basis_upload_bytes == resident.upload_bytes()
second, third:     basis_upload_bytes == 0
                   transfer_bytes     == first - upload  (the quartet tables alone)
all three:         bit-identical to the throwaway-basis path
```

### 34-D acceptance

`primitive_screening_at_zero_is_the_identity`: tolerance 0 is bit-identical to no
screening (only quartets whose scale factor underflowed to exactly zero are
dropped, and those contribute exactly zero), and `f64::MAX` drops everything —
so the knob is demonstrably wired to the kernel and demonstrably inert at its
default.

The screened quantity is `sqrt(a0/a1^3) * common_factor * exp(-mu_ij R_ij^2) *
exp(-mu_kl R_kl^2)`, i.e. the scalar the whole G tensor is built from — not the
exponential prefactor alone. `sqrt(a0/a1^3)` is not O(1): for diffuse primitives
`a1` is small and that square root is large, so a prefactor-only test would
discard contributions it had not bounded. It is still a proxy, not a
certificate — the Rys weights and the recurrence coefficients are not bounded by
one — and the doc comment says so.

---

## 5. What is still open

- **Phase 33 (device Rys nroots 6-12) — not started.** Its stated prerequisite
  *is* done: `cintx-cubecl` now compiles under `--no-default-features --features
  {wgpu,cuda,rocm,metal}`, which it did not before (13 errors, all
  `cubecl::cpu::CpuRuntime` named without a `cfg(feature = "cpu")` gate in
  `math/rys_wheeler.rs` and `math/eigh.rs`). The host-side solvers those
  launchers wrap are now selected by cfg, and a CI leg builds all five profiles
  so the class of defect cannot return. The phase itself — inline `rys_roots_dev`,
  the ~3.4 KB/work-item Wheeler scratch, the per-family ceiling raise — is
  untouched, and its highest-risk item (33-05, per-backend proof that the
  compiler does not contract `two_sum`/`two_prod` into FMA) cannot be discharged
  on this host for any GPU backend.
- **def2/J and def2/JK auxiliary bases** — not added (open question Q4). The 3c2e
  and 2c2e batch paths do not need them (they take any shell list), but a
  realistic RI-J benchmark does.
- **Device-resident output mode (34-E's last sub-item)** — not added. The
  collective-readback half was already done by 34-B.
- **Clippy warnings** — `cargo clippy --workspace --all-targets` is **error-free**
  (was 75-82 deny-level findings). ~2 674 warnings remain, dominated by
  `excessive_precision` on transcribed libcint tables and `needless_range_loop`
  in kernel bodies. Clearing those is a separate, purely cosmetic task.
- **Open questions 1-5 remain open.** Q1 in particular: on the CPU backend,
  parity with libcint is the realistic ceiling and 2e has now reached it
  (1.28x faster on CH4, 1.4x slower on the smaller H2O list, where 69 launches
  at ~42 us still dominate). A decisive win needs either a GPU backend or a
  reduction in launch *count* — merging launch classes that share
  `(ibase, kbase, nroots)` and moving the per-class shape parameters into a
  per-quartet table would take 69 launches to roughly 15.
