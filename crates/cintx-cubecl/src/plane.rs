//! Plane-level operations, primitives, and alignment utilities for CubeCL.
//!
//! Planes (known as Warps in CUDA, Subgroups in Vulkan/WebGPU/SPIR-V, and SIMD-groups
//! in Metal) execute in lock-step and can share data directly via register-level
//! intrinsics without hitting shared or global memory.
//!
//! This module provides:
//! - Host-side plane-aligned launch topology constructors and helpers.
//! - In-kernel (`#[cube]`) collective reductions, scans, votes, and leader election.
//! - Plane-cooperative execution primitives for integral batch processing.

// The `as usize` / `as u32` casts here are load-bearing under `#[cube]`: the
// CubeCL builtins (`UNIT_POS`, `CUBE_DIM`, ...) expand to `NativeExpand<u32>`,
// and `Array` indexing takes a `usize`, so the uniform `(expr) as usize` form is
// what lets an index expression be swapped between a literal and a variable.
// Clippy sees the post-expansion type and reads them as redundant.
#![allow(clippy::unnecessary_cast)]
// Index-carrying loops (`for axis in 0..3`, `for i in 0..n`) index several
// parallel arrays or a strided buffer, and the index itself names an axis,
// component or stride. An iterator rewrite would hide exactly that.
#![allow(clippy::needless_range_loop)]
// Kernel launches take the whole shape contract as positional arguments — that
// is the CubeCL calling convention, not a design choice — and the host wrappers
// mirror it so the two can be read side by side.
#![allow(clippy::too_many_arguments)]

use cubecl::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// Host-side Plane Topology Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Default fallback plane dimension when hardware capability is unknown.
pub const DEFAULT_PLANE_DIM: u32 = 32;

/// Standard plane-aligned cube dimension (256 threads), evenly divisible by
/// 32-wide warps (Nvidia), 64-wide wavefronts (AMD/Metal), and Vulkan/WebGPU subgroups.
pub const STANDARD_PLANE_ALIGNED_CUBE_DIM: u32 = 256;

// ─────────────────────────────────────────────────────────────────────────────
// Hardware-Adaptive Launch Geometry
// ─────────────────────────────────────────────────────────────────────────────

/// The launch-relevant facts about a backend, read from its [`ComputeClient`].
///
/// Every launch decision in this crate — cube dimension, cube count, per-unit
/// width — is a function of these five numbers and the work available. Reading
/// them from the client rather than from the runtime *type* is what makes the
/// geometry adapt: a backend is treated as CPU-like because it reports
/// `plane_size_max == 1`, not because it happens to be `cubecl::cpu`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchHardware {
    /// Plane (warp / wavefront / subgroup) width, floored at 1.
    pub plane_dim: u32,
    /// Does this backend execute in hardware planes? `plane_dim > 1`.
    ///
    /// When false, a cube unit is an OS worker thread rather than a SIMD lane:
    /// dispatch costs microseconds instead of nanoseconds, `sync_cube` is a
    /// software barrier, and shared memory is ordinary cache. Cube width then
    /// has to be bought with work, not spent for occupancy.
    pub has_planes: bool,
    /// Independent hardware execution contexts: CPU cores on a CPU backend, SMs
    /// / CUs on a GPU one, falling back to host parallelism if the backend
    /// reports neither.
    pub parallel_units: u32,
    /// Hardware ceiling on units in one cube.
    pub max_units_per_cube: u32,
    /// Hardware ceiling on cubes along the x axis of the dispatch grid.
    pub max_cubes_x: u32,
}

/// Host parallelism, queried once, as the fallback when a backend reports
/// neither a core count nor an SM count.
fn host_parallelism() -> u32 {
    static HW: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *HW.get_or_init(|| {
        std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(1)
    })
}

/// Read the launch-relevant hardware facts off a client.
///
/// `client.properties()` is a borrow of a struct the runtime filled in at
/// device creation, so this is field reads and no device round-trip; it is
/// cheap enough to call once per dispatch.
#[inline]
pub fn launch_hardware<R: cubecl::Runtime>(client: &ComputeClient<R>) -> LaunchHardware {
    let hardware = &client.properties().hardware;
    let plane_dim = hardware.plane_size_max.max(1);
    let parallel_units = hardware
        .num_cpu_cores
        .or(hardware.num_streaming_multiprocessors)
        .map(|n| n.max(1))
        .unwrap_or_else(host_parallelism);
    LaunchHardware {
        plane_dim,
        has_planes: plane_dim > 1,
        parallel_units,
        max_units_per_cube: hardware.max_units_per_cube.max(1),
        max_cubes_x: hardware.max_cube_count.0.max(1),
    }
}

/// Does this backend execute in hardware planes (warps / wavefronts /
/// subgroups)?
///
/// This is the branch every geometry helper below takes, and it replaces the
/// runtime-type test [`runtime_is_cpu`] for that purpose: the property that
/// decides launch shape is "are units SIMD lanes or OS threads", and
/// `plane_size_max` answers exactly that for any backend, present or future.
#[inline]
pub fn has_planes<R: cubecl::Runtime>(client: &ComputeClient<R>) -> bool {
    client.properties().hardware.plane_size_max > 1
}

/// Clamp a wanted cube count to what the backend's grid can actually dispatch.
///
/// Replaces the hardcoded 65535 that used to cap every family's cube count.
/// 65535 is the WebGPU/Vulkan limit; CUDA and HIP allow `2^31 - 1` along x, so
/// the constant was leaving parallelism unclaimed on exactly the backends that
/// could use it. Callers bound `want_cubes` by their scratch budget first, so
/// raising this ceiling cannot raise peak memory.
#[inline]
pub fn grid_cube_count<R: cubecl::Runtime>(client: &ComputeClient<R>, want_cubes: usize) -> u32 {
    let max_cubes = launch_hardware(client).max_cubes_x;
    (want_cubes.min(max_cubes as usize).max(1)) as u32
}

/// Return a standard plane-aligned 1D [`CubeDim`] of 256 threads.
#[inline]
pub fn standard_plane_cube_dim() -> CubeDim {
    CubeDim::new_1d(STANDARD_PLANE_ALIGNED_CUBE_DIM)
}

/// Is `R` the CubeCL **CPU** runtime?
///
/// This answers a question about the runtime *type*. It is **not** what decides
/// launch topology any more — [`has_planes`] is, because the property that
/// matters is whether a unit is a SIMD lane or an OS thread, and a backend
/// answers that itself through `plane_size_max`. Keep this for the rare case
/// that genuinely needs to know which runtime it is holding.
#[inline]
pub fn runtime_is_cpu<R: cubecl::Runtime>() -> bool {
    #[cfg(feature = "cpu")]
    {
        std::any::TypeId::of::<R>() == std::any::TypeId::of::<cubecl::cpu::CpuRuntime>()
    }
    #[cfg(not(feature = "cpu"))]
    {
        false
    }
}

/// Cube dimension for a **single-cube cooperative kernel** that distributes
/// `work_items` independent items across the cube and synchronises with
/// `sync_cube()`.
///
/// # Why this is backend-dependent (Task 34-A0)
///
/// On a GPU backend a cube is a workgroup: units are hardware lanes, `sync_cube`
/// is a workgroup barrier costing tens of cycles, and a wide cube is free
/// occupancy. Sizing the cube to the work is the right call.
///
/// On the CubeCL **CPU** backend none of that holds
/// (`cubecl-cpu-0.10.0`):
///
/// - `compute/runner.rs::execute_data` spawns **one OS thread per cube unit**,
///   growing the worker pool past `available_parallelism` if the cube demands
///   it, and clones the kernel's `MlirData` once per unit **per launch**.
/// - `compute/compute_task.rs::sync_cube` is a **global spin-wait barrier**
///   across every unit. Oversubscribed, each barrier costs a full scheduler
///   round.
/// - `compiler/visitor/mod.rs` lowers `cube_count` to a sequential `scf.for`
///   loop *inside* each unit, so the grid is not a parallelism axis; the cube
///   dimension is the only one, and it is an OS-thread count.
///
/// Measured on the scalar 2e kernel (`artifacts/34-A0_cube_dim_ab.md`), a
/// 256-unit cube is between 28x and ~4.9e5x **slower** than a single unit on
/// the CPU backend, because the kernel's `sync_cube()` calls sit inside the
/// primitive-quartet loop. So on the CPU backend this returns `1`: one thread,
/// and every `sync_cube()` degenerates to the barrier's `barrier_target <= 1`
/// early return.
///
/// Kernels remain written cooperatively and stay correct at any cube
/// dimension — `UNIT_POS == 0` guards and `idx % CUBE_DIM == UNIT_POS`
/// partitioning both degenerate correctly at 1.
///
/// # Plane alignment (hardware-adaptive)
///
/// The GPU arm rounds the useful width **up to a whole multiple of the
/// backend's plane size**, read from the client, rather than up to a power of
/// two. A workgroup occupies whole planes whichever number it asks for, so a
/// cube of 4 on a 32-wide warp does not cost less than a cube of 32 — it just
/// idles 28 of the 32 lanes the hardware already allocated. Many 1e and 2e
/// classes have contraction blocks in the single digits, so the power-of-two
/// rounding was leaving most of every warp unused; plane alignment claims those
/// lanes for free. It also fixes the 64-wide wavefront case, where a
/// power-of-two dim of 32 is half a wavefront.
#[inline]
pub fn cooperative_cube_dim<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    work_items: u32,
) -> CubeDim {
    let hw = launch_hardware(client);
    if !hw.has_planes {
        return CubeDim::new_1d(1);
    }
    let ceiling = STANDARD_PLANE_ALIGNED_CUBE_DIM.min(hw.max_units_per_cube);
    let aligned = plane_aligned_cube_dim(work_items.max(1), hw.plane_dim).num_elems();
    // The alignment ceiling is itself rounded down to a whole plane so the
    // clamp can never hand back a partial plane.
    let ceiling = (ceiling / hw.plane_dim).max(1) * hw.plane_dim;
    CubeDim::new_1d(aligned.min(ceiling))
}

/// Cube dimension for a cooperative kernel whose useful parallel width is not
/// known at the launch site.
///
/// Returns a single unit on the CubeCL CPU runtime and the standard 256-wide
/// plane everywhere else, for the reasons spelled out on
/// [`cooperative_cube_dim`]. Use that function instead wherever the launch site
/// *does* know how many independent work items the kernel has — it also sizes
/// the GPU cube to the work.
///
/// Every kernel this is used from partitions with `UNIT_POS == 0` guards,
/// `idx % CUBE_DIM == UNIT_POS` selection, or `i = UNIT_POS; i += CUBE_DIM`
/// stride loops. All three cover the full index space at any cube dimension,
/// so a single unit changes cost, never results.
#[inline]
pub fn backend_plane_cube_dim<R: cubecl::Runtime>(client: &ComputeClient<R>) -> CubeDim {
    let hw = launch_hardware(client);
    if !hw.has_planes {
        return CubeDim::new_1d(1);
    }
    // 256 is only the *request*; the returned dim is a whole number of this
    // backend's planes and never exceeds its per-cube unit ceiling.
    cooperative_cube_dim(client, STANDARD_PLANE_ALIGNED_CUBE_DIM)
}

/// `min_items_per_unit` for the shell-pair and shell-triple families
/// (`int1e_*`, `int2c2e`, `int3c2e`).
///
/// One item there is `nprim^2`/`nprim^3` primitive tuples through a small
/// G-tensor, which is the same order as the per-unit dispatch cost — so a unit
/// needs several of them before waking it pays for itself. Measured on
/// H2O/def2-SVP `int2c2e` (~16 pairs per class): 4 units beat 16 by ~3x, and
/// beat 1 unit as well.
pub const MIN_ITEMS_PER_UNIT_PAIR: usize = 4;

/// Unit count for the per-unit decomposition, given the work available, the
/// per-item cost tier, and a per-slot scratch budget.
///
/// A CubeCL CPU launch dispatches each unit through an mpsc channel to its own
/// OS thread and clones the binding table per unit; measured on this host that
/// is ~2 us per unit per launch. Splitting a class across more units than its
/// work can fill pays that cost for nothing — the `int2c2e` classes of an
/// H2O/def2-SVP list are ~16 pairs each, and spreading them over 16 threads was
/// **3x slower** than over 4.
///
/// `min_items_per_unit` is where the per-family difference lives, because how
/// much work one item is differs by orders of magnitude: a 2e quartet runs
/// `nprim^4` primitive quartets through a full VRR/HRR build, while a 1e or
/// 2c2e pair runs `nprim^2` through a much smaller one. Pass 1 for a family
/// whose single item already dwarfs the dispatch, and a larger value for one
/// where it does not.
/// The parallel width comes from the **client**
/// (`hardware.num_cpu_cores`), not from `available_parallelism`. The CubeCL CPU
/// runtime sizes its worker pool from its own core count, so asking it for more
/// units than that oversubscribes the pool — and a runtime configured with a
/// smaller pool than the host has (a shared CI runner, a pinned device) is
/// invisible to a host-side query.
pub fn per_unit_width<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    n_items: usize,
    min_items_per_unit: usize,
    by_memory: usize,
) -> u32 {
    let hw = launch_hardware(client);
    let by_hardware = (hw.parallel_units as usize).min(hw.max_units_per_cube as usize);
    // S5: the work term is rounded **up** to a power of two before the clamps.
    //
    // A cube width is part of a kernel's compiled identity, so a launch that
    // asks for 13 units and one that asks for 14 are two programs to the JIT
    // even though nothing about the arithmetic differs. Left unquantized, the
    // width tracks the work-list size directly: H2O/def2-SVP and CH4/def2-SVP
    // share all 15 launch signatures, yet CH4 after H2O in one process paid
    // 0.53 s of fresh compilation because a handful of its classes crossed a
    // unit-count boundary.
    //
    // Rounding up rather than down keeps every unit that has work, and hands
    // spare units an empty range of the blocked grid-stride walk — which they
    // already tolerate, because `chunk = ceil(n / n_slots)` has always been able
    // to leave the last slot short. The clamps below still bound the result by
    // hardware and by scratch memory, so quantizing can never ask for a width
    // the device or the budget refuses.
    let by_work = (n_items / min_items_per_unit.max(1))
        .max(1)
        // Saturating, because a work count near `usize::MAX` has no next power
        // of two and the hardware clamp below is what actually decides there.
        .checked_next_power_of_two()
        .unwrap_or(usize::MAX);
    by_hardware.min(by_work).min(by_memory).max(1) as u32
}

/// Return a standard single-cube [`CubeCount`] dispatch (`1, 1, 1`).
#[inline]
pub fn single_cube_count() -> CubeCount {
    CubeCount::Static(1, 1, 1)
}

/// Return a 1D [`CubeCount`] grid dispatch (`x, 1, 1`).
#[inline]
pub fn cube_count_1d(x: u32) -> CubeCount {
    CubeCount::Static(x.max(1), 1, 1)
}

/// Return a 2D [`CubeCount`] grid dispatch (`x, y, 1`).
#[inline]
pub fn cube_count_2d(x: u32, y: u32) -> CubeCount {
    CubeCount::Static(x.max(1), y.max(1), 1)
}

/// Return a 3D [`CubeCount`] grid dispatch (`x, y, z`).
#[inline]
pub fn cube_count_3d(x: u32, y: u32, z: u32) -> CubeCount {
    CubeCount::Static(x.max(1), y.max(1), z.max(1))
}

/// Compute a 1D linear [`CubeCount`] that covers `total_items` using `block_size` threads per block,
/// clamped to standard GPU grid bounds (`1..=65535`).
#[inline]
pub fn linear_grid_cube_count(total_items: usize, block_size: u32) -> CubeCount {
    let bs = block_size.max(1);
    let num_cubes = (total_items as u32).div_ceil(bs).clamp(1, 65535);
    CubeCount::Static(num_cubes, 1, 1)
}

/// Compute a 2D tiled [`CubeCount`] that covers `(items_x, items_y)` using `(block_x, block_y)` tiles.
#[inline]
pub fn tiled_grid_cube_count_2d(
    items_x: usize,
    items_y: usize,
    block_x: u32,
    block_y: u32,
) -> CubeCount {
    let bx = block_x.max(1);
    let by = block_y.max(1);
    let cx = (items_x as u32).div_ceil(bx).clamp(1, 65535);
    let cy = (items_y as u32).div_ceil(by).clamp(1, 65535);
    CubeCount::Static(cx, cy, 1)
}

/// Compute a 3D tiled [`CubeCount`] that covers `(items_x, items_y, items_z)` using `(block_x, block_y, block_z)` tiles.
#[inline]
pub fn tiled_grid_cube_count_3d(
    items_x: usize,
    items_y: usize,
    items_z: usize,
    block_x: u32,
    block_y: u32,
    block_z: u32,
) -> CubeCount {
    let bx = block_x.max(1);
    let by = block_y.max(1);
    let bz = block_z.max(1);
    let cx = (items_x as u32).div_ceil(bx).clamp(1, 65535);
    let cy = (items_y as u32).div_ceil(by).clamp(1, 65535);
    let cz = (items_z as u32).div_ceil(bz).clamp(1, 65535);
    CubeCount::Static(cx, cy, cz)
}

/// Calculate the number of full planes within a cube.
#[inline]
pub fn planes_per_cube(cube_dim: &CubeDim, plane_dim: u32) -> u32 {
    let total_units = cube_dim.num_elems();
    total_units.checked_div(plane_dim).unwrap_or(1)
}

/// Compute a 1D [`CubeDim`] that is guaranteed to be an exact multiple of `plane_dim`,
/// avoiding partially filled tail planes.
#[inline]
pub fn plane_aligned_cube_dim(requested_units: u32, plane_dim: u32) -> CubeDim {
    let p = if plane_dim == 0 {
        DEFAULT_PLANE_DIM
    } else {
        plane_dim
    };
    let aligned = if requested_units <= p {
        p
    } else {
        requested_units.div_ceil(p) * p
    };
    CubeDim::new_1d(aligned)
}

/// Compute a 2D [`CubeDim`] where `x` is plane-aligned and the total workgroup size
/// `x * y` is guaranteed to be an exact multiple of `plane_dim` within GPU hardware limits (<= 1024).
#[inline]
pub fn plane_aligned_cube_dim_2d(requested_x: u32, requested_y: u32, plane_dim: u32) -> CubeDim {
    let p = if plane_dim == 0 {
        DEFAULT_PLANE_DIM
    } else {
        plane_dim
    };
    let aligned_x = if requested_x <= p {
        p
    } else {
        requested_x.div_ceil(p) * p
    };
    let y = requested_y.max(1);
    let total = (aligned_x * y).min(1024);
    let clamped_y = (total / aligned_x).max(1);
    CubeDim::new_2d(aligned_x, clamped_y)
}

/// Compute a 3D [`CubeDim`] where `x` is plane-aligned and the total workgroup size
/// `x * y * z` is guaranteed to be an exact multiple of `plane_dim` within GPU hardware limits (<= 1024).
#[inline]
pub fn plane_aligned_cube_dim_3d(
    requested_x: u32,
    requested_y: u32,
    requested_z: u32,
    plane_dim: u32,
) -> CubeDim {
    let p = if plane_dim == 0 {
        DEFAULT_PLANE_DIM
    } else {
        plane_dim
    };
    let aligned_x = if requested_x <= p {
        p
    } else {
        requested_x.div_ceil(p) * p
    };
    let y = requested_y.max(1);
    let z = requested_z.max(1);
    let total = (aligned_x * y * z).min(1024);
    let clamped_yz = (total / aligned_x).max(1);
    let clamped_z = z.min(clamped_yz);
    let clamped_y = (clamped_yz / clamped_z).max(1);
    CubeDim::new_3d(aligned_x, clamped_y, clamped_z)
}

/// Compute hardware-decoupled occupancy-tuned launch geometry ([`CubeCount`], [`CubeDim`])
/// for grid-stride batch processing.
///
/// Decouples problem size `total_items` from hardware workgroup sizing. Sizes `CubeDim`
/// to 256 units (plane-aligned across all backends) and clamps `CubeCount` to hardware CU ceilings,
/// ensuring full ALU occupancy without excessive launch/scheduling overhead.
#[inline]
pub fn occupancy_launch_geometry(
    total_items: usize,
    max_cubes: u32,
    plane_dim: u32,
) -> (CubeCount, CubeDim) {
    let cube_dim = plane_aligned_cube_dim(STANDARD_PLANE_ALIGNED_CUBE_DIM, plane_dim);
    let units_per_cube = cube_dim.num_elems();
    let num_cubes = (total_items as u32)
        .div_ceil(units_per_cube)
        .clamp(1, max_cubes.max(1));
    (CubeCount::Static(num_cubes, 1, 1), cube_dim)
}

/// Scalar operations one unit should be worth before another CPU unit is
/// allocated.
///
/// A CubeCL CPU unit is an OS thread dispatched through a task queue, costing
/// on the order of a microsecond to wake — a few thousand cycles, so tens of
/// thousands of scalar operations. Below this threshold a second thread is a
/// net loss, and the work belongs on the one that is already awake.
pub const WORK_PER_CPU_UNIT: usize = 32 * 1024;

/// Ceiling on CPU cube units, so an anomalous virtual-core count cannot turn
/// one launch into hundreds of thread wakeups.
pub const CPU_CUBE_DIM_MAX: u32 = 64;

/// Hardware-adaptive launch geometry for a **grid-stride** kernel: one whose
/// body is `let mut i = ABSOLUTE_POS; while i < n { ...; i += CUBE_COUNT_X *
/// CUBE_DIM_X }` and is therefore correct at any geometry, so the geometry is
/// free to be chosen purely for speed.
///
/// `work_per_item` is the caller's estimate of the scalar operations one item
/// costs; it only has to be right to an order of magnitude, and it is what lets
/// the CPU arm tell a batch of cheap items from a batch of expensive ones.
///
/// # Why the two arms are shaped so differently
///
/// - **CPU-like backends** (`plane_size_max == 1`): the parallel axis is
///   `CUBE_DIM_X` — a unit is an OS thread — while `cube_count` lowers to a
///   sequential `scf.for` *inside* each unit. So the units are sized to the
///   work (one thread per [`WORK_PER_CPU_UNIT`] scalar ops, capped by the
///   backend's core count, the item count and [`CPU_CUBE_DIM_MAX`]) and the
///   grid stays at one cube; the kernel's own stride loop covers the tail.
///
///   Note this is the opposite of the rule for *cooperative* kernels
///   ([`cooperative_cube_dim`], which returns 1 unit on CPU). The difference is
///   `sync_cube`: a cooperative kernel barriers inside its inner loop, where a
///   wide CPU cube costs a scheduler round per barrier, whereas a grid-stride
///   kernel never barriers and a wide cube is pure parallelism.
///
/// - **GPU backends**: the grid is the parallel axis. The cube is a whole
///   number of *this* backend's planes, and the grid is sized to the item count
///   but capped at a few resident cubes per streaming multiprocessor, read from
///   the client — enough to hide memory latency behind other cubes without a
///   grid so wide that its tail dominates.
///
/// `CINTX_GRID_STRIDE_CUBE_DIM` pins the cube dimension for A/B measurement, in
/// the same spirit as `CINTX_2E_CUBE_DIM`, and is not part of the public
/// contract. The grid is still sized to cover the items at the pinned width, so
/// a pinned run stays correct.
#[inline]
pub fn adaptive_grid_stride_geometry<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    total_items: usize,
    work_per_item: usize,
) -> (CubeCount, CubeDim) {
    /// Resident cubes per SM to aim for.
    const CUBES_PER_SM: u32 = 4;

    static PINNED: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    let pinned = *PINNED.get_or_init(|| {
        std::env::var("CINTX_GRID_STRIDE_CUBE_DIM")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .filter(|value| *value > 0)
    });
    if let Some(dim) = pinned {
        let cubes = (total_items as u32).div_ceil(dim).max(1);
        return (CubeCount::Static(cubes, 1, 1), CubeDim::new_1d(dim));
    }

    let hw = launch_hardware(client);
    if !hw.has_planes {
        let total_work = total_items.saturating_mul(work_per_item.max(1));
        let by_work = (total_work / WORK_PER_CPU_UNIT).max(1);
        let units = by_work
            .min(hw.parallel_units as usize)
            .min(total_items.max(1))
            .max(1) as u32;
        let units = units.min(CPU_CUBE_DIM_MAX).min(hw.max_units_per_cube);
        return (CubeCount::Static(1, 1, 1), CubeDim::new_1d(units));
    }
    let cube_dim = backend_plane_cube_dim(client);
    let max_cubes = hw
        .parallel_units
        .saturating_mul(CUBES_PER_SM)
        .clamp(1, hw.max_cubes_x);
    let num_cubes = (total_items as u32)
        .div_ceil(cube_dim.num_elems())
        .clamp(1, max_cubes);
    (CubeCount::Static(num_cubes, 1, 1), cube_dim)
}

/// Compute launch geometry ([`CubeCount`], [`CubeDim`]) for plane-cooperative workloads
/// where each plane processes one or more work items.
#[inline]
pub fn plane_cooperative_launch_geometry(
    total_items: usize,
    planes_per_cube: u32,
    plane_dim: u32,
) -> (CubeCount, CubeDim) {
    let p_dim = if plane_dim == 0 {
        DEFAULT_PLANE_DIM
    } else {
        plane_dim
    };
    let p_per_cube = planes_per_cube.max(1);
    let units_per_cube = p_dim * p_per_cube;
    let cube_dim = CubeDim::new_1d(units_per_cube);

    let total_planes = (total_items as u32).max(1);
    let num_cubes = total_planes.div_ceil(p_per_cube).clamp(1, 65535);
    let cube_count = CubeCount::Static(num_cubes, 1, 1);

    (cube_count, cube_dim)
}

// ─────────────────────────────────────────────────────────────────────────────
// In-Kernel (`#[cube]`) CubeDim & Plane Primitives and Reductions
// ─────────────────────────────────────────────────────────────────────────────

/// Returns the total number of execution units in the current cube (`CUBE_DIM`).
#[cube]
pub fn cube_dim_total() -> u32 {
    CUBE_DIM
}

/// Returns the X dimension of the current cube (`CUBE_DIM_X`).
#[cube]
pub fn cube_dim_x() -> u32 {
    CUBE_DIM_X
}

/// Returns the Y dimension of the current cube (`CUBE_DIM_Y`).
#[cube]
pub fn cube_dim_y() -> u32 {
    CUBE_DIM_Y
}

/// Returns the Z dimension of the current cube (`CUBE_DIM_Z`).
#[cube]
pub fn cube_dim_z() -> u32 {
    CUBE_DIM_Z
}

/// Returns the flattened local unit index in the current cube (`UNIT_POS`).
#[cube]
pub fn unit_pos_total() -> u32 {
    UNIT_POS
}

/// Returns the local unit index along the X axis (`UNIT_POS_X`).
#[cube]
pub fn unit_pos_x() -> u32 {
    UNIT_POS_X
}

/// Returns the local unit index along the Y axis (`UNIT_POS_Y`).
#[cube]
pub fn unit_pos_y() -> u32 {
    UNIT_POS_Y
}

/// Returns the local unit index along the Z axis (`UNIT_POS_Z`).
#[cube]
pub fn unit_pos_z() -> u32 {
    UNIT_POS_Z
}

/// Returns the cube index along the X axis (`CUBE_POS_X`).
#[cube]
pub fn cube_pos_x() -> u32 {
    CUBE_POS_X
}

/// Returns the cube index along the Y axis (`CUBE_POS_Y`).
#[cube]
pub fn cube_pos_y() -> u32 {
    CUBE_POS_Y
}

/// Returns the cube index along the Z axis (`CUBE_POS_Z`).
#[cube]
pub fn cube_pos_z() -> u32 {
    CUBE_POS_Z
}

/// Returns the total number of cubes in the dispatch grid (`CUBE_COUNT_X * CUBE_COUNT_Y * CUBE_COUNT_Z`).
#[cube]
pub fn cube_count_total() -> u32 {
    CUBE_COUNT_X * CUBE_COUNT_Y * CUBE_COUNT_Z
}

/// Returns the number of cubes along the X axis (`CUBE_COUNT_X`).
#[cube]
pub fn cube_count_x() -> u32 {
    CUBE_COUNT_X
}

/// Returns the number of cubes along the Y axis (`CUBE_COUNT_Y`).
#[cube]
pub fn cube_count_y() -> u32 {
    CUBE_COUNT_Y
}

/// Returns the number of cubes along the Z axis (`CUBE_COUNT_Z`).
#[cube]
pub fn cube_count_z() -> u32 {
    CUBE_COUNT_Z
}

/// Returns the 1D global unit index across the entire grid (`ABSOLUTE_POS`).
#[cube]
pub fn absolute_pos_1d() -> usize {
    ABSOLUTE_POS as usize
}

/// Returns the global unit index along the X axis (`ABSOLUTE_POS_X`).
#[cube]
pub fn absolute_pos_x() -> u32 {
    ABSOLUTE_POS_X
}

/// Returns the global unit index along the Y axis (`ABSOLUTE_POS_Y`).
#[cube]
pub fn absolute_pos_y() -> u32 {
    ABSOLUTE_POS_Y
}

/// Returns the global unit index along the Z axis (`ABSOLUTE_POS_Z`).
#[cube]
pub fn absolute_pos_z() -> u32 {
    ABSOLUTE_POS_Z
}

/// Returns the 1D global grid-stride stride (`CUBE_COUNT_X * CUBE_DIM_X`).
#[cube]
pub fn grid_stride_1d() -> usize {
    (CUBE_COUNT_X * CUBE_DIM_X) as usize
}

/// Compute a 2D candidate index from `(CUBE_POS_X, CUBE_POS_Y)` with row stride `stride_y`,
/// matching multi-dimensional grid dispatching (e.g. `(node * n_features + fidx)`).
#[cube]
pub fn grid_candidate_2d(stride_y: u32) -> usize {
    (CUBE_POS_X * stride_y + CUBE_POS_Y) as usize
}

/// Compute a 3D candidate index from `(CUBE_POS_X, CUBE_POS_Y, CUBE_POS_Z)` with strides `stride_y` and `stride_z`.
#[cube]
pub fn grid_candidate_3d(stride_y: u32, stride_z: u32) -> usize {
    ((CUBE_POS_X * stride_y + CUBE_POS_Y) * stride_z + CUBE_POS_Z) as usize
}

/// Returns `true` if the executing thread is the leader thread (`UNIT_POS == 0`) of the cube.
#[cube]
pub fn is_leader_unit_in_cube() -> bool {
    UNIT_POS == 0u32
}

/// Returns the current unit's index within its plane (`0 <= idx < PLANE_DIM`).
#[cube]
pub fn unit_pos_in_plane() -> u32 {
    UNIT_POS_PLANE
}

/// Returns the current plane's index within its cube (`0 <= idx < CUBE_DIM / PLANE_DIM`).
#[cube]
pub fn plane_pos_in_cube() -> u32 {
    PLANE_POS
}

/// Returns the size (lane count) of the current plane.
#[cube]
pub fn plane_dimension() -> u32 {
    PLANE_DIM
}

/// Intra-plane sum reduction across all active units in the plane.
///
/// Uses CubeCL's built-in `plane_sum` intrinsic or XOR butterfly shuffle folding
/// for portable hardware execution across CUDA (warps), Vulkan (subgroups), and Metal (SIMD-groups).
#[cube]
pub fn plane_reduce_sum<N: Numeric>(val: N) -> N {
    plane_sum(val)
}

/// Intra-plane product reduction across all active units in the plane.
#[cube]
pub fn plane_reduce_prod<N: Numeric>(val: N) -> N {
    plane_prod(val)
}

/// Intra-plane maximum reduction across all active units in the plane.
#[cube]
pub fn plane_reduce_max<N: Numeric>(val: N) -> N {
    plane_max(val)
}

/// Intra-plane minimum reduction across all active units in the plane.
#[cube]
pub fn plane_reduce_min<N: Numeric>(val: N) -> N {
    plane_min(val)
}

/// Intra-plane inclusive prefix sum scan.
///
/// Each lane receives the sum of all lane values from index 0 up to its own index.
#[cube]
pub fn plane_scan_inclusive<N: Numeric>(val: N) -> N {
    plane_inclusive_sum(val)
}

/// Intra-plane exclusive prefix sum scan.
///
/// Each lane receives the sum of all lane values from index 0 up to (excluding) its own index.
#[cube]
pub fn plane_scan_exclusive<N: Numeric>(val: N) -> N {
    plane_exclusive_sum(val)
}

/// Intra-plane inclusive prefix product scan.
#[cube]
pub fn plane_scan_inclusive_prod<N: Numeric>(val: N) -> N {
    plane_inclusive_prod(val)
}

/// Intra-plane exclusive prefix product scan.
#[cube]
pub fn plane_scan_exclusive_prod<N: Numeric>(val: N) -> N {
    plane_exclusive_prod(val)
}

/// Evaluates a boolean condition across the plane and returns `true` if it holds for
/// AT LEAST ONE unit in the plane.
#[cube]
pub fn plane_vote_any(cond: bool) -> bool {
    plane_any(cond)
}

/// Evaluates a boolean condition across the plane and returns `true` if it holds for
/// ALL units in the plane.
#[cube]
pub fn plane_vote_all(cond: bool) -> bool {
    plane_all(cond)
}

/// Leader election within a plane.
///
/// Returns `true` for exactly one unit in the plane (the lowest-indexed active unit),
/// enabling single-unit memory writes or control tasks without atomic contention.
#[cube]
pub fn plane_leader_elect() -> bool {
    plane_elect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_planes_per_cube_calculation() {
        let cube_dim_256 = CubeDim::new_1d(256);
        assert_eq!(planes_per_cube(&cube_dim_256, 32), 8);
        assert_eq!(planes_per_cube(&cube_dim_256, 64), 4);
        assert_eq!(planes_per_cube(&cube_dim_256, 16), 16);

        let cube_dim_64 = CubeDim::new_1d(64);
        assert_eq!(planes_per_cube(&cube_dim_64, 32), 2);
        assert_eq!(planes_per_cube(&cube_dim_64, 64), 1);
    }

    #[test]
    fn test_plane_aligned_cube_dim() {
        let dim = plane_aligned_cube_dim(50, 32);
        assert_eq!(dim.x, 64);
        assert_eq!(dim.num_elems(), 64);

        let dim_exact = plane_aligned_cube_dim(64, 32);
        assert_eq!(dim_exact.x, 64);

        let dim_single = plane_aligned_cube_dim(10, 32);
        assert_eq!(dim_single.x, 32);
    }

    #[test]
    fn test_plane_cooperative_launch_geometry() {
        let (count, dim) = plane_cooperative_launch_geometry(100, 4, 32);
        assert_eq!(dim.num_elems(), 128); // 4 planes * 32 units
        match count {
            CubeCount::Static(x, y, z) => {
                assert_eq!(x, 25);
                assert_eq!(y, 1);
                assert_eq!(z, 1);
            }
            _ => panic!("expected Static CubeCount"),
        }
    }

    #[test]
    fn test_plane_aligned_launch_properties() {
        // Zero or unknown plane dim falls back to DEFAULT_PLANE_DIM
        let dim_default = plane_aligned_cube_dim(1, 0);
        assert_eq!(dim_default.num_elems(), DEFAULT_PLANE_DIM);

        // Power of two alignment
        for requested in [1, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256] {
            let dim32 = plane_aligned_cube_dim(requested, 32);
            assert_eq!(dim32.num_elems() % 32, 0);
            assert!(dim32.num_elems() >= requested);

            let dim64 = plane_aligned_cube_dim(requested, 64);
            assert_eq!(dim64.num_elems() % 64, 0);
            assert!(dim64.num_elems() >= requested);
        }
    }

    /// A CPU client for the hardware-adaptive tests below.
    #[cfg(feature = "cpu")]
    fn cpu_client() -> ComputeClient<cubecl::cpu::CpuRuntime> {
        use cubecl::Runtime;
        cubecl::cpu::CpuRuntime::client(&cubecl::cpu::CpuDevice::default())
    }

    /// Task 34-A0: the CPU runtime maps one cube unit to one OS thread and
    /// `sync_cube` to a global spin barrier, so a cooperative launch there must
    /// be a single unit. GPU runtimes keep the plane-aligned cube.
    ///
    /// The decision now comes from the client's reported `plane_size_max`, so
    /// this also pins that the CPU backend is recognised through its properties
    /// and not through its runtime type.
    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_runtime_gets_a_single_unit_cube() {
        let client = cpu_client();
        assert!(runtime_is_cpu::<cubecl::cpu::CpuRuntime>());
        assert!(
            !has_planes(&client),
            "the cpu backend must report plane_size_max == 1"
        );
        for work in [1u32, 81, 256, 1296] {
            assert_eq!(
                cooperative_cube_dim(&client, work).num_elems(),
                1,
                "cooperative_cube_dim must be 1 on the cpu runtime (work={work})"
            );
        }
        assert_eq!(backend_plane_cube_dim(&client).num_elems(), 1);
    }

    /// `launch_hardware` must report the CPU backend's own core count, not the
    /// host's, and must never hand back a zero in any field a divisor uses.
    #[cfg(feature = "cpu")]
    #[test]
    fn launch_hardware_reads_the_backend_not_the_host() {
        let client = cpu_client();
        let hw = launch_hardware(&client);
        assert_eq!(hw.plane_dim, 1);
        assert!(!hw.has_planes);
        assert!(hw.parallel_units >= 1);
        assert!(hw.max_units_per_cube >= 1);
        assert!(hw.max_cubes_x >= 1);
        assert_eq!(
            hw.parallel_units,
            client
                .properties()
                .hardware
                .num_cpu_cores
                .expect("cpu backend reports a core count")
        );
    }

    /// The per-unit width is bounded by work, by the scratch budget, and by the
    /// backend's parallelism — whichever binds first.
    #[cfg(feature = "cpu")]
    #[test]
    fn per_unit_width_takes_the_tightest_bound() {
        let client = cpu_client();
        let cores = launch_hardware(&client).parallel_units as usize;

        // Work-bound: 8 items at 4 items per unit is 2 units, whatever the host has.
        assert_eq!(per_unit_width(&client, 8, 4, usize::MAX), 2);
        // S5: the work term is quantized up to a power of two, so neighbouring
        // work-list sizes compile to one program instead of two. 5 items at 1
        // per unit rounds to 8, not 5 — and is still clamped by the host's core
        // count below.
        assert_eq!(
            per_unit_width(&client, 5, 1, usize::MAX) as usize,
            8.min(cores)
        );
        assert_eq!(
            per_unit_width(&client, 6, 1, usize::MAX),
            per_unit_width(&client, 8, 1, usize::MAX),
            "sizes inside one power-of-two bucket must share a width"
        );
        // Memory-bound.
        assert_eq!(per_unit_width(&client, 1_000_000, 1, 3), 3);
        // Hardware-bound, and never zero.
        assert_eq!(
            per_unit_width(&client, usize::MAX, 1, usize::MAX) as usize,
            cores
        );
        assert_eq!(per_unit_width(&client, 0, 4, usize::MAX), 1);
    }

    /// The grid ceiling comes from `max_cube_count.x`, so it can exceed the
    /// 65535 literal it replaced on backends that allow more.
    #[cfg(feature = "cpu")]
    #[test]
    fn grid_cube_count_clamps_to_hardware() {
        let client = cpu_client();
        let max = launch_hardware(&client).max_cubes_x as usize;
        assert_eq!(grid_cube_count(&client, 0), 1);
        assert_eq!(grid_cube_count(&client, 7), 7);
        assert_eq!(grid_cube_count(&client, usize::MAX) as usize, max);
        assert!(
            max > 65_535,
            "the cpu backend allows more cubes than the old literal"
        );
    }

    /// On a plane-less backend the grid-stride parallel axis is the cube
    /// dimension (one unit is one OS thread) and `cube_count` is a sequential
    /// loop inside each unit — so the units carry the work and the grid stays
    /// at one cube.
    #[cfg(feature = "cpu")]
    #[test]
    fn grid_stride_geometry_puts_cpu_parallelism_in_the_cube() {
        let client = cpu_client();
        let cores = launch_hardware(&client).parallel_units;
        let ceiling = cores.min(CPU_CUBE_DIM_MAX);

        // Plenty of work: every unit the backend reports, up to the cap.
        let (count, dim) = adaptive_grid_stride_geometry(&client, 1_000_000, 50);
        assert!(
            matches!(count, CubeCount::Static(1, 1, 1)),
            "the cpu grid must stay at one cube; got {count:?}"
        );
        assert_eq!(dim.num_elems(), ceiling);

        // A batch too small to pay for a second thread wakeup stays on one.
        let (_, tiny) = adaptive_grid_stride_geometry(&client, 4, 50);
        assert_eq!(
            tiny.num_elems(),
            1,
            "200 scalar ops must not buy a second OS thread"
        );

        // Work-proportional in between: 32K ops buys exactly one unit.
        let (_, two) = adaptive_grid_stride_geometry(&client, 2 * WORK_PER_CPU_UNIT, 1);
        assert_eq!(two.num_elems(), 2u32.min(ceiling));

        // Never more units than there are items to take.
        let (_, capped) = adaptive_grid_stride_geometry(&client, 3, usize::MAX / 4);
        assert_eq!(capped.num_elems(), 3u32.min(ceiling));

        // Empty batches still produce a launchable geometry.
        let (_, empty) = adaptive_grid_stride_geometry(&client, 0, 50);
        assert_eq!(empty.num_elems(), 1);
    }

    /// Plane alignment is the point of the GPU arm: whatever the useful width,
    /// the cube is a whole number of planes and within the per-cube ceiling.
    #[test]
    fn plane_alignment_is_exact_for_every_useful_width() {
        for plane_dim in [32u32, 64] {
            for requested in [1u32, 3, 7, 31, 33, 65, 200, 257, 4096] {
                let dim = plane_aligned_cube_dim(requested, plane_dim);
                assert_eq!(
                    dim.num_elems() % plane_dim,
                    0,
                    "dim {} is not a whole number of {plane_dim}-wide planes",
                    dim.num_elems()
                );
                assert!(dim.num_elems() >= requested.min(plane_dim));
            }
        }
    }

    #[test]
    fn test_standard_plane_cube_dim() {
        let dim = standard_plane_cube_dim();
        assert_eq!(dim.num_elems(), 256);
        assert_eq!(dim.num_elems() % 32, 0);
        assert_eq!(dim.num_elems() % 64, 0);
        assert_eq!(planes_per_cube(&dim, 32), 8);
        assert_eq!(planes_per_cube(&dim, 64), 4);
    }

    #[test]
    fn test_plane_aligned_cube_dim_2d() {
        let dim = plane_aligned_cube_dim_2d(30, 4, 32);
        assert_eq!(dim.x, 32);
        assert_eq!(dim.y, 4);
        assert_eq!(dim.z, 1);
        assert_eq!(dim.num_elems(), 128);
        assert_eq!(dim.num_elems() % 32, 0);

        // Hardware workgroup clamp
        let dim_large = plane_aligned_cube_dim_2d(64, 32, 32);
        assert!(dim_large.num_elems() <= 1024);
        assert_eq!(dim_large.num_elems() % 32, 0);
    }

    #[test]
    fn test_plane_aligned_cube_dim_3d() {
        let dim = plane_aligned_cube_dim_3d(32, 4, 2, 32);
        assert_eq!(dim.x, 32);
        assert_eq!(dim.y, 4);
        assert_eq!(dim.z, 2);
        assert_eq!(dim.num_elems(), 256);
        assert_eq!(dim.num_elems() % 32, 0);

        // Hardware workgroup clamp
        let dim_large = plane_aligned_cube_dim_3d(64, 8, 8, 32);
        assert!(dim_large.num_elems() <= 1024);
        assert_eq!(dim_large.num_elems() % 32, 0);
    }

    #[test]
    fn test_occupancy_launch_geometry() {
        // Small workload
        let (count_small, dim_small) = occupancy_launch_geometry(10, 64, 32);
        assert_eq!(dim_small.num_elems(), 256);
        match count_small {
            CubeCount::Static(x, y, z) => {
                assert_eq!(x, 1);
                assert_eq!(y, 1);
                assert_eq!(z, 1);
            }
            _ => panic!("expected Static CubeCount"),
        }

        // Large workload clamped by max_cubes
        let (count_large, dim_large) = occupancy_launch_geometry(1_000_000, 96, 32);
        assert_eq!(dim_large.num_elems(), 256);
        match count_large {
            CubeCount::Static(x, y, z) => {
                assert_eq!(x, 96);
                assert_eq!(y, 1);
                assert_eq!(z, 1);
            }
            _ => panic!("expected Static CubeCount"),
        }
    }

    #[test]
    fn test_cube_count_constructors() {
        match single_cube_count() {
            CubeCount::Static(x, y, z) => {
                assert_eq!((x, y, z), (1, 1, 1));
            }
            _ => panic!("expected Static"),
        }

        match cube_count_1d(12) {
            CubeCount::Static(x, y, z) => {
                assert_eq!((x, y, z), (12, 1, 1));
            }
            _ => panic!("expected Static"),
        }

        match cube_count_2d(8, 4) {
            CubeCount::Static(x, y, z) => {
                assert_eq!((x, y, z), (8, 4, 1));
            }
            _ => panic!("expected Static"),
        }

        match cube_count_3d(4, 3, 2) {
            CubeCount::Static(x, y, z) => {
                assert_eq!((x, y, z), (4, 3, 2));
            }
            _ => panic!("expected Static"),
        }

        match linear_grid_cube_count(1000, 256) {
            CubeCount::Static(x, y, z) => {
                assert_eq!((x, y, z), (4, 1, 1));
            }
            _ => panic!("expected Static"),
        }

        match tiled_grid_cube_count_2d(64, 32, 16, 16) {
            CubeCount::Static(x, y, z) => {
                assert_eq!((x, y, z), (4, 2, 1));
            }
            _ => panic!("expected Static"),
        }

        match tiled_grid_cube_count_3d(64, 32, 16, 16, 8, 4) {
            CubeCount::Static(x, y, z) => {
                assert_eq!((x, y, z), (4, 4, 4));
            }
            _ => panic!("expected Static"),
        }
    }
}
