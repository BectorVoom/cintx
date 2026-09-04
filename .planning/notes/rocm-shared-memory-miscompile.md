---
title: Shared memory in the batched 2e kernel on ROCm — root cause found; not a miscompile
date: 2026-09-03
type: defect
context: def2 speed/memory plan, workstream S3 (cooperative G tensor in shared memory)
status: resolved — both findings have root causes; S3 passes parity with `CINTX_2E_SHARED_G=1`
---

# Shared memory in `two_electron_scalar_kernel` on ROCm

Two findings. Both were originally attributed to the wrong layer, and the
correction matters more than the original report, so it comes first.

| | Original attribution | Actual root cause |
|---|---|---|
| **A. Stale compiled kernels** | AMD's `~/.cache/comgr` code-object cache | **CubeCL's own on-disk kernel cache**, keyed by `KernelId` — which contains no hash of the kernel body. Clearing `~/.cache/comgr` does nothing. |
| **B. "Shared-memory miscompile"** | A compiler miscompile on gfx1151 under register/scheduling pressure | **Not a miscompile.** The S3 edit rebound only 6 of 46 G-tensor accesses to `g_slab`; the other 40 — the whole VRR and HRR — still indexed the raw `g` parameter. Under `g_in_shared` that splits the recurrence across two buffers and reads uninitialised LDS. |

Both are fixed. `def2_batch_rocm_parity` now passes with S3 enabled, with the
same numbers as the global-slab control, on three consecutive runs.

## Environment

| | |
|---|---|
| GPU | AMD Radeon 860M, `gfx1151` (integrated, Ryzen AI 7 350) |
| ROCm | 7.1.1 |
| cubecl | 0.10.0 (`cubecl-core`, `cubecl-runtime`, `cubecl-hip` 0.10.0; `cubecl-hip-sys` 7.1.5280200) |
| Kernel | `crates/cintx-cubecl/src/kernels/two_electron.rs::two_electron_scalar_kernel` |
| Reported LDS | 65 536 bytes (`client.properties().hardware.max_shared_memory_size`) |

---

# A. The stale-kernel cache is CubeCL's, not comgr's

## Where it lives

`cubecl.toml` at the repo root sets:

```toml
[compilation]
cache = "target"
```

`CacheConfig::Target` resolves by walking up from the **current working
directory** to the nearest `Cargo.toml`. Cargo runs an integration test with the
cwd set to the *package* root, so the cache does not land in the workspace
`target/` anyone would think to clear. It lands in a per-crate shadow directory:

```text
crates/cintx-oracle/target/hip/0.10.0/hip-kernel/chunk0.cbor   # 26 MB
crates/cintx-oracle/target/hip/0.10.0/hip-kernel/toc.json.log
crates/cintx-cubecl/target/hip/0.10.0/hip-kernel/...
```

## Why a body edit does not invalidate it

`cubecl-hip-0.10.0/src/compute/context.rs::compile_kernel` keys the cache on
`kernel_id.stable_hash()`, and `cubecl-runtime-0.10.0/src/id.rs:132` defines
that as:

```rust
self.type_name.hash(&mut hasher);
self.address_type.hash(&mut hasher);
self.cube_dim.hash(&mut hasher);
self.mode.hash(&mut hasher);
self.info.hash(&mut hasher);
```

**No hash of the kernel body, and no hash of the generated source.** Editing a
`#[cube]` body without touching its parameters or comptime settings reuses the
cached device binary. This also explains, exactly, the two workarounds that were
observed to help:

- *renaming the kernel works* — `type_name` is in the key;
- *adding a parameter or a comptime setting works* — `info` is in the key.

It also answers open question 3 from the previous revision: the cache in play is
keyed by cubecl's kernel id, not by generated source, and it is not comgr.

On a cache hit `compile_kernel` returns early, so it never calls
`validate_shared` and never calls `logger.log_compilation`. That gives a clean
way to detect the trap: **run with `CUBECL_DEBUG_LOG=<file>`; if no kernel
source appears in the log, nothing was compiled and you are running cached
binaries.** That is how this was caught — a "verified" reproduction run produced
a 1.8 KB log with profiling rows and zero source.

## Working rules

1. To force a recompile, delete the CubeCL cache — **not** `~/.cache/comgr`:
   ```bash
   rm -rf crates/*/target/hip target/hip
   ```
   Renaming the kernel or adding a comptime setting also works, for the reason
   above.
2. Confirm it worked: `CUBECL_DEBUG_LOG=/tmp/k.log` must produce a log
   containing the generated HIP source (megabytes, not kilobytes).
3. **Initialise output buffers with a sentinel** through `create_from_slice`,
   never `client.empty`, in any test that must distinguish "kernel produced
   this" from "kernel never ran". `client.empty()` hands back recycled device
   pages that are not zeroed between processes.
4. Treat any GPU probe result obtained across a body-only edit, without step 1,
   as void.

Rules 3 and 4 are recorded at
`crates/cintx-cubecl/src/shared_memory.rs::tests::shared_memory_through_a_slice_round_trips`.

---

# B. Not a miscompile — a split binding of the G tensor

## The defect

S3 selects where the G-tensor scratch lives, at comptime:

```rust
let mut g_slab = if comptime!(g_in_shared == 1u32) {
    shared_g.to_slice_mut()      // SharedMemory<F>
} else {
    g.to_slice_mut()             // the global Array<F> parameter
};
```

The comment above it claimed the recurrences "are written once and bind
whichever they were given". They were not. Counting accesses in the kernel body:

| Binding | Count | Which |
|---|---|---|
| `g_slab[..]` | 6 | the 3 seed writes (`gx/gy/gz[irys]`) and the 3 contraction reads |
| `g[..]` | **40** | the entire VRR and the entire HRR |

Indexing `g` directly compiles cleanly — it is still a live kernel argument —
and is *correct in the global configuration*, because there `g_slab` aliases
`g`. There is no compiler signal. Under `g_in_shared` the recurrence splits in
two, and both halves are wrong:

- the 3 seeds are written to **shared memory**;
- the VRR then reads `g[off + root]` from **global memory** — never written,
  since the seed went to LDS — and builds the whole tensor from garbage. Worse,
  `gx_off = slot * g_stride * global_slab` folds to **0** in this configuration
  (`global_slab == 0`), so every cube writes the same global offset: a
  cross-cube race on top of the uninitialised read;
- the contraction then reads **shared memory**, which holds only the 3 seeds.
  Everything else is uninitialised LDS.

Verified in the generated HIP: `l_14 = l_13 * uint32(0)` (the folded `gx_off`),
seeds at `shared_memory_10[...]` inside `if (l_6 == 0)`, and the recurrence
addressing the global buffer.

This accounts for every symptom, none of which needed a compiler bug:

| Symptom | Explanation |
|---|---|
| `\|diff\|` up to 1e122 and `inf` | uninitialised LDS reinterpreted as `f64` |
| surviving count varies run to run (597, 2 518, 5 883) | LDS contents vary; nothing else in the run does |
| a single shared element is enough to trigger it | one seed diverted is enough to poison the recurrence |
| fails at `CINTX_2E_CUBE_DIM=1` | never was a cross-lane problem |
| 48 KiB, 4 KiB and 512 B all fail | size is irrelevant |
| allocating without reading is innocent | an unread allocation does not divert a seed |
| `CINTX_AUTOTUNE=off` still fails | geometry is irrelevant |
| the small-kernel control round-trips exactly | the primitive was never at fault |

## The fix

Rebind every G access in the recurrence range to `g_slab`, and state the
invariant at the declaration so the next edit does not reintroduce it. The `g`
parameter remains, because the global configuration still binds it.

## Verification

Compiled-kernel cache and comgr cleared before the run; each run recompiles from
source.

```bash
rm -rf crates/*/target/hip ~/.cache/comgr
export CINTX_ORACLE_BUILD_VENDOR=1 CINTX_ROCM_ORACLE=1
CINTX_2E_SHARED_G=1 cargo test --release -p cintx-oracle \
  --features cpu,rocm,extended-device-rys \
  --test def2_batch_rocm_parity -- --ignored --nocapture def2_2e_batch
```

| Configuration | Result |
|---|---|
| before the fix, shared G, cold cache | `bit-identical=5883 / 597 / 2518`, `max\|diff\|=inf` — FAILED |
| after the fix, shared G, run 1 | `bit-identical=38138  max\|diff\|=8.882e-16`  vendor `mismatched=0` |
| after the fix, shared G, run 2 | `bit-identical=38138  max\|diff\|=8.882e-16`  vendor `mismatched=0` |
| after the fix, global G (control) | `bit-identical=38138  max\|diff\|=8.882e-16`  vendor `mismatched=0` |

Shared and global now agree exactly and deterministically. (38 138 rather than
53 237 bit-identical is the expected baseline: the cooperative and per-unit
decompositions sum in different orders, and the gate is a divergence bound.)

## A control that was run and is worth keeping

Independently of the fix, the global slab was filled with a poison value
(`-8.71e201`) instead of `client.empty()`, host-side only, so no kernel edit and
no cache concern. Parity was unaffected — which confirms the standing claim at
`two_electron.rs` that the kernel rebuilds every G element it reads, and rules
out uninitialised-read on the global path. That claim is load-bearing for
reusing one slab across dispatches, and it now has a measurement behind it.

## What this changes elsewhere

- **No upstream report is warranted for B.** CubeCL and ROCm behaved correctly
  throughout; `SharedMemory`, `to_slice_mut()` and `sync_cube()` all lower
  correctly on gfx1151, exactly as the small-kernel control always said.
- **A is worth reporting upstream**, or at least worth a local guard: a kernel
  cache with no body or source hash in its key silently serves stale binaries,
  and the early return on a cache hit also skips `validate_shared`.
- `.planning/notes/cuda-metal-verification-gap.md` — the CUDA question stands on
  its own merits, but it is no longer needed to settle this. The previous
  revision named it "the single most useful next datum"; it would have shown the
  same failure on NVIDIA, since the fault was in cintx.
- `docs/design/def2_speed_memory_optimization_plan.md` §10.6 and §10.6.1 need
  their conclusion replaced; they currently record a backend defect.

## Still open

- **Should S3 default on?** `shared_g_enabled()` still returns `false` unless
  `CINTX_2E_SHARED_G=1`; this change fixes correctness and does not touch the
  default. The speed case for S3 has not been measured since the fix.
- **Does the same split-binding bug exist elsewhere?** Checked: no. Only
  `two_electron.rs` uses `SharedMemory::` or the `to_slice_mut()` selection
  idiom outside `shared_memory.rs`, so `two_electron_scalar_kernel` is the sole
  site. Re-check whenever a second kernel adopts the pattern.
