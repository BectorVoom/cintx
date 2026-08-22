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
    if plane_dim == 0 {
        1
    } else {
        total_units / plane_dim
    }
}

/// Compute a 1D [`CubeDim`] that is guaranteed to be an exact multiple of `plane_dim`,
/// avoiding partially filled tail planes.
#[inline]
pub fn plane_aligned_cube_dim(requested_units: u32, plane_dim: u32) -> CubeDim {
    let p = if plane_dim == 0 { DEFAULT_PLANE_DIM } else { plane_dim };
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
    let p = if plane_dim == 0 { DEFAULT_PLANE_DIM } else { plane_dim };
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
    let p = if plane_dim == 0 { DEFAULT_PLANE_DIM } else { plane_dim };
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
    let p_dim = if plane_dim == 0 { DEFAULT_PLANE_DIM } else { plane_dim };
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
