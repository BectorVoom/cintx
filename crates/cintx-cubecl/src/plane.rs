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

/// Return a standard plane-aligned 1D [`CubeDim`] of 256 threads.
#[inline]
pub fn standard_plane_cube_dim() -> CubeDim {
    CubeDim::new_1d(STANDARD_PLANE_ALIGNED_CUBE_DIM)
}

/// Is `R` the CubeCL **CPU** runtime?
///
/// The CPU runtime's execution model is fundamentally unlike a GPU's, and the
/// difference is not a tuning detail — it decides launch topology
/// (see [`cooperative_cube_dim`]).
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
#[inline]
pub fn cooperative_cube_dim<R: cubecl::Runtime>(work_items: u32) -> CubeDim {
    if runtime_is_cpu::<R>() {
        return CubeDim::new_1d(1);
    }
    let mut dim = 1u32;
    while dim < work_items && dim < STANDARD_PLANE_ALIGNED_CUBE_DIM {
        dim *= 2;
    }
    CubeDim::new_1d(dim.min(STANDARD_PLANE_ALIGNED_CUBE_DIM))
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
pub fn backend_plane_cube_dim<R: cubecl::Runtime>() -> CubeDim {
    if runtime_is_cpu::<R>() {
        return CubeDim::new_1d(1);
    }
    standard_plane_cube_dim()
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
pub fn per_unit_width(n_items: usize, min_items_per_unit: usize, by_memory: usize) -> u32 {
    static HW: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let hw = *HW.get_or_init(|| {
        std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1)
    });
    let by_work = (n_items / min_items_per_unit.max(1)).max(1);
    hw.min(by_work).min(by_memory).max(1) as u32
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

    /// Task 34-A0: the CPU runtime maps one cube unit to one OS thread and
    /// `sync_cube` to a global spin barrier, so a cooperative launch there must
    /// be a single unit. GPU runtimes keep the plane-aligned cube.
    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_runtime_gets_a_single_unit_cube() {
        assert!(runtime_is_cpu::<cubecl::cpu::CpuRuntime>());
        for work in [1u32, 81, 256, 1296] {
            assert_eq!(
                cooperative_cube_dim::<cubecl::cpu::CpuRuntime>(work).num_elems(),
                1,
                "cooperative_cube_dim must be 1 on the cpu runtime (work={work})"
            );
        }
        assert_eq!(
            backend_plane_cube_dim::<cubecl::cpu::CpuRuntime>().num_elems(),
            1
        );
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
