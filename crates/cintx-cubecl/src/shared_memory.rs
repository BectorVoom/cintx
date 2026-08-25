//! Shared-memory layout planning, capacity management, cooperative primitives,
//! and verification infrastructure for CubeCL kernels.
//!
//! Grounded in `docs/design/cubecl_shared_memory_kernel_optimization_plan.md`.
//! Supports compile-time static shared-memory allocation, barrier-safe cooperative
//! loading, capacity checks against hardware limits (`max_shared_memory_size`),
//! pure host-side size calculations, and transactional fallback mechanisms.

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

use cintx_core::cintxRsError;
use cubecl::prelude::*;
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// 1. Shared Memory Variant Taxonomy
// ─────────────────────────────────────────────────────────────────────────────

/// High-level shared-memory execution variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum SharedVariant {
    /// Correct batched control; one independent item per lane (direct global access).
    #[default]
    NoSharedLane,
    /// Cooperatively stage small reused descriptor/table data into shared memory.
    SharedDescriptor,
    /// Single leader / plane produces recurrence/roots once; output units read it.
    SharedRecurrence,
    /// Tiled recurrence, transform, or ECP matrix intermediate.
    SharedTiled,
    /// Combine plane-owned partials through a small shared region.
    SharedPlanePartials,
    /// Ping-pong double-buffered memory-bound tile.
    SharedDoubleBuffer,
    /// Fused stages when global traffic falls without useful cross-unit reuse.
    FusedNoShared,
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Capacity Classes & Limits
// ─────────────────────────────────────────────────────────────────────────────

/// Bounded capacity classes to avoid unbounded compilation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CapacityClass {
    /// 0 bytes of shared memory.
    NoShared,
    /// Small footprint (up to 4 KiB).
    Small,
    /// Medium footprint (up to 16 KiB).
    Medium,
    /// Large footprint (up to 48 KiB).
    Large,
    /// Explicit tile element count.
    Tile(usize),
}

impl CapacityClass {
    pub const fn byte_limit(self) -> usize {
        match self {
            Self::NoShared => 0,
            Self::Small => 4 * 1024,
            Self::Medium => 16 * 1024,
            Self::Large => 48 * 1024,
            Self::Tile(elems) => elems * 8,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. SharedLayout Definition
// ─────────────────────────────────────────────────────────────────────────────

/// Backend-neutral description of shared-memory requirements and layout geometry.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SharedLayout {
    pub variant: SharedVariant,
    pub cube_dim_units: u32,
    pub descriptor_elems: usize,
    pub root_elems: usize,
    pub recurrence_tile_elems: usize,
    pub transform_tile_elems: usize,
    pub partial_elems: usize,
    pub buffer_count: usize,
    pub alignment_bytes: usize,
    pub total_bytes: usize,
}

impl SharedLayout {
    /// Build a new `SharedLayout`, computing total bytes for an element type of `element_size` bytes.
    pub fn new(
        variant: SharedVariant,
        cube_dim_units: u32,
        descriptor_elems: usize,
        root_elems: usize,
        recurrence_tile_elems: usize,
        transform_tile_elems: usize,
        partial_elems: usize,
        buffer_count: usize,
        element_size: usize,
        alignment_bytes: usize,
    ) -> Self {
        let align = alignment_bytes.max(element_size).max(1);
        let buf_count = buffer_count.max(1);

        let total_elems = descriptor_elems
            + root_elems
            + (recurrence_tile_elems * buf_count)
            + transform_tile_elems
            + partial_elems;

        let raw_bytes = total_elems * element_size;
        let total_bytes = if raw_bytes == 0 {
            0
        } else {
            raw_bytes.div_ceil(align) * align
        };

        Self {
            variant,
            cube_dim_units: cube_dim_units.max(1),
            descriptor_elems,
            root_elems,
            recurrence_tile_elems,
            transform_tile_elems,
            partial_elems,
            buffer_count: buf_count,
            alignment_bytes: align,
            total_bytes,
        }
    }

    /// Layout for `NoSharedLane` (0 bytes).
    pub fn no_shared(cube_dim_units: u32) -> Self {
        Self {
            variant: SharedVariant::NoSharedLane,
            cube_dim_units: cube_dim_units.max(1),
            descriptor_elems: 0,
            root_elems: 0,
            recurrence_tile_elems: 0,
            transform_tile_elems: 0,
            partial_elems: 0,
            buffer_count: 1,
            alignment_bytes: 8,
            total_bytes: 0,
        }
    }

    /// Returns the sum of all elements allocated across all shared regions.
    #[inline]
    pub fn total_elements(&self) -> usize {
        self.descriptor_elems
            + self.root_elems
            + (self.recurrence_tile_elems * self.buffer_count)
            + self.transform_tile_elems
            + self.partial_elems
    }

    /// Check whether this layout fits within the device hardware limit,
    /// respecting a maximum allowed occupancy reserve factor (e.g. 0.5 for 2 resident cubes).
    #[inline]
    pub fn fits_device_limit(&self, max_shared_bytes: u32, reserve_factor: f64) -> bool {
        if self.variant == SharedVariant::NoSharedLane
            || self.variant == SharedVariant::FusedNoShared
        {
            return true;
        }
        let cap = ((max_shared_bytes as f64) * reserve_factor.clamp(0.1, 1.0)) as usize;
        self.total_bytes <= cap
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Telemetry, Metrics, and Fallback Reasons
// ─────────────────────────────────────────────────────────────────────────────

/// Reason why a shared-memory variant was declined in favor of fallback/no-shared.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FallbackReason {
    None,
    NotProfitable,
    CapacityExceeded,
    BackendUnverified,
    CompileLaunchFailed,
}

/// Metrics collected during shared-memory planning and execution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SharedMemoryMetrics {
    pub variant: SharedVariant,
    pub planned_shared_bytes: usize,
    pub actual_shared_bytes: usize,
    pub barrier_count: usize,
    pub tile_count: usize,
    pub active_units: u32,
    pub fallback_reason: FallbackReason,
}

impl Default for SharedMemoryMetrics {
    fn default() -> Self {
        Self {
            variant: SharedVariant::NoSharedLane,
            planned_shared_bytes: 0,
            actual_shared_bytes: 0,
            barrier_count: 0,
            tile_count: 1,
            active_units: 256,
            fallback_reason: FallbackReason::None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. In-Kernel (`#[cube]`) Cooperative Load & Barrier Primitives
// ─────────────────────────────────────────────────────────────────────────────

/// Cooperatively copy data from a global [`Array`] slice into [`SharedMemory`].
///
/// All units participate: lane `i` copies elements `i, i + CUBE_DIM, ...`.
/// Units beyond `src_len` write zero (`F::new(0.0)`) into shared memory up to `dst_len`.
/// A full `sync_cube()` barrier is executed before returning so all threads see complete data.
#[cube]
pub fn cooperative_load_slice<F: Float + CubeElement>(
    src: &Array<F>,
    src_offset: usize,
    src_len: usize,
    dst: &mut SharedMemory<F>,
    dst_len: usize,
) {
    let tid = UNIT_POS as usize;
    let stride = CUBE_DIM as usize;

    let mut idx = tid;
    while idx < dst_len {
        if idx < src_len {
            dst[idx] = src[src_offset + idx];
        } else {
            dst[idx] = F::new(0.0_f32);
        }
        idx += stride;
    }
    sync_cube();
}

/// Zero out an entire [`SharedMemory`] region cooperatively across all cube units.
#[cube]
pub fn cooperative_zero_shared<F: Float + CubeElement>(dst: &mut SharedMemory<F>, dst_len: usize) {
    let tid = UNIT_POS as usize;
    let stride = CUBE_DIM as usize;

    let mut idx = tid;
    while idx < dst_len {
        dst[idx] = F::new(0.0_f32);
        idx += stride;
    }
    sync_cube();
}

/// Cooperatively load or zero a single unit slot.
#[cube]
pub fn cooperative_load_or_zero<F: Float + CubeElement>(
    dst: &mut SharedMemory<F>,
    lane: usize,
    active: bool,
    val: F,
) {
    if active {
        dst[lane] = val;
    } else {
        dst[lane] = F::new(0.0_f32);
    }
    sync_cube();
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Pure Host-Side Size & Layout Calculators
// ─────────────────────────────────────────────────────────────────────────────

/// Compute exact `SharedLayout` for 1-electron families (overlap, kinetic, nuc, rinv, grads, etc.).
pub fn calc_1e_layout(
    op: &str,
    li: u32,
    lj: u32,
    nroots: u32,
    nprim_i: usize,
    nprim_j: usize,
    cube_dim: u32,
) -> Result<SharedLayout, cintxRsError> {
    let element_size = std::mem::size_of::<f64>();
    let lj_ext = match op {
        "kin" | "grad_kin" => lj + 2,
        _ => lj,
    };
    let g_per_axis = (li + 1 + 1) * (lj_ext + 1);
    let recurrence_tile_elems = (3 * g_per_axis * nroots.max(1)) as usize;

    let descriptor_elems = nprim_i + nprim_j + nprim_i + nprim_j + 6; // exps + coeffs + centers
    let root_elems = (2 * nroots.max(1)) as usize; // urys + wrys

    let layout = SharedLayout::new(
        SharedVariant::SharedRecurrence,
        cube_dim,
        descriptor_elems,
        root_elems,
        recurrence_tile_elems,
        0,
        0,
        1,
        element_size,
        8,
    );

    Ok(layout)
}

/// Compute exact `SharedLayout` for 2-electron scalar and derivative families.
pub fn calc_2e_layout(
    li: u32,
    lj: u32,
    lk: u32,
    ll: u32,
    nroots: u32,
    cube_dim: u32,
) -> Result<SharedLayout, cintxRsError> {
    let element_size = std::mem::size_of::<f64>();
    let g_size = (nroots.max(1) * (li + lj + 1) * (lk + ll + 1)) as usize;
    let recurrence_tile_elems = 3 * g_size;
    let descriptor_elems = 12 + 16; // 4 centers (12 doubles) + up to 16 descriptor params
    let root_elems = (2 * nroots.max(1)) as usize;

    let layout = SharedLayout::new(
        SharedVariant::SharedRecurrence,
        cube_dim,
        descriptor_elems,
        root_elems,
        recurrence_tile_elems,
        0,
        0,
        1,
        element_size,
        8,
    );

    Ok(layout)
}

/// Compute exact `SharedLayout` for 2c2e center family.
pub fn calc_2c2e_layout(
    li: u32,
    lj: u32,
    nroots: u32,
    cube_dim: u32,
) -> Result<SharedLayout, cintxRsError> {
    let element_size = std::mem::size_of::<f64>();
    let g_size = (nroots.max(1) * (li + lj + 1)) as usize;
    let recurrence_tile_elems = 3 * g_size;
    let descriptor_elems = 6 + 8;
    let root_elems = (2 * nroots.max(1)) as usize;

    Ok(SharedLayout::new(
        SharedVariant::SharedRecurrence,
        cube_dim,
        descriptor_elems,
        root_elems,
        recurrence_tile_elems,
        0,
        0,
        1,
        element_size,
        8,
    ))
}

/// Compute exact `SharedLayout` for 3c1e center family.
pub fn calc_3c1e_layout(
    li: u32,
    lj: u32,
    lk: u32,
    cube_dim: u32,
) -> Result<SharedLayout, cintxRsError> {
    let element_size = std::mem::size_of::<f64>();
    let g_size = ((li + lj + lk + 1) * (li + lj + 1)) as usize;
    let recurrence_tile_elems = 3 * g_size;
    let descriptor_elems = 9 + 12;

    Ok(SharedLayout::new(
        SharedVariant::SharedRecurrence,
        cube_dim,
        descriptor_elems,
        0,
        recurrence_tile_elems,
        0,
        0,
        1,
        element_size,
        8,
    ))
}

/// Compute exact `SharedLayout` for 3c2e center family (scalar, ip1, ip2).
pub fn calc_3c2e_layout(
    li: u32,
    lj: u32,
    lk: u32,
    nroots: u32,
    cube_dim: u32,
) -> Result<SharedLayout, cintxRsError> {
    let element_size = std::mem::size_of::<f64>();
    let g_size = (nroots.max(1) * (li + lj + 1) * (lk + 1)) as usize;
    let split_size = (nroots.max(1) * (lk + 1) * (lj + 1) * (li + 1)) as usize;
    let recurrence_tile_elems = 3 * g_size + 3 * split_size;
    let descriptor_elems = 9 + 12;
    let root_elems = (2 * nroots.max(1)) as usize;

    Ok(SharedLayout::new(
        SharedVariant::SharedRecurrence,
        cube_dim,
        descriptor_elems,
        root_elems,
        recurrence_tile_elems,
        0,
        0,
        1,
        element_size,
        8,
    ))
}

/// Compute exact `SharedLayout` for 4c1e center family.
pub fn calc_4c1e_layout(
    li: u32,
    lj: u32,
    lk: u32,
    ll: u32,
    cube_dim: u32,
) -> Result<SharedLayout, cintxRsError> {
    let element_size = std::mem::size_of::<f64>();
    let g_size = ((li + lj + lk + ll + 1) * (li + lj + 1)) as usize;
    let recurrence_tile_elems = 3 * g_size;
    let descriptor_elems = 12 + 16;

    Ok(SharedLayout::new(
        SharedVariant::SharedRecurrence,
        cube_dim,
        descriptor_elems,
        0,
        recurrence_tile_elems,
        0,
        0,
        1,
        element_size,
        8,
    ))
}

/// Compute exact `SharedLayout` for ECP type-2 angular contraction (tiled intermediate).
pub fn calc_ecp_type2_layout(
    li: u32,
    lj: u32,
    lc: u32,
    cube_dim: u32,
) -> Result<SharedLayout, cintxRsError> {
    let element_size = std::mem::size_of::<f64>();
    let ni = (li + 1) * (li + 2) / 2;
    let nj = (lj + 1) * (lj + 2) / 2;
    let nc = (lc + 1) * (lc + 2) / 2;
    let buf_intermediate_elems = (ni * nc) as usize;
    let transform_tile_elems = (ni * nj) as usize;

    Ok(SharedLayout::new(
        SharedVariant::SharedTiled,
        cube_dim,
        (ni + nj + nc) as usize,
        0,
        buf_intermediate_elems,
        transform_tile_elems,
        0,
        1,
        element_size,
        8,
    ))
}

/// Compute exact `SharedLayout` for F12 Cartesian contraction.
pub fn calc_f12_layout(
    li: u32,
    lj: u32,
    lk: u32,
    ll: u32,
    nroots: u32,
    cube_dim: u32,
) -> Result<SharedLayout, cintxRsError> {
    let element_size = std::mem::size_of::<f64>();
    let di = (li + 1) * (li + 2) / 2;
    let dj = (lj + 1) * (lj + 2) / 2;
    let dk = (lk + 1) * (lk + 2) / 2;
    let dl = (ll + 1) * (ll + 2) / 2;
    let recurrence_tile_elems = (nroots.max(1) * di * dj * dk * dl) as usize;

    Ok(SharedLayout::new(
        SharedVariant::SharedTiled,
        cube_dim,
        16,
        (2 * nroots.max(1)) as usize,
        recurrence_tile_elems.min(4096), // Tiled tile cap
        0,
        0,
        1,
        element_size,
        8,
    ))
}

/// Compute exact `SharedLayout` for Sigma/relativistic families.
pub fn calc_sigma_layout(
    _family: &str,
    li: u32,
    lj: u32,
    nroots: u32,
    cube_dim: u32,
) -> Result<SharedLayout, cintxRsError> {
    let element_size = std::mem::size_of::<f64>();
    let g_per_axis = (li + 2) * (lj + 2);
    let recurrence_tile_elems = (3 * g_per_axis * nroots.max(1)) as usize;

    Ok(SharedLayout::new(
        SharedVariant::SharedRecurrence,
        cube_dim,
        32,
        (2 * nroots.max(1)) as usize,
        recurrence_tile_elems,
        0,
        0,
        1,
        element_size,
        8,
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// 6b. Extended-Rys (nroots 6..=12) inline scratch — private, not shared
// ─────────────────────────────────────────────────────────────────────────────

/// Per-work-item scratch words the inline extended-Rys entry
/// (`math::rys_wheeler::rys_roots_ext_dev`) allocates, for one `nroots`.
///
/// # Why this is private memory and not a shared-memory tile
///
/// Every buffer the extended path needs is a `#[cube]`-local
/// `Array::<f64>::new(<comptime>)`, so it lands in thread-private storage —
/// registers where they fit, the backend's local/global spill space where they
/// do not. It never enters the shared-memory budget, which is what makes the
/// path issuable at all: the dd arms need ~6 KB per work item, and 16 units of
/// that is ~100 KB, comfortably over the 64 KB of shared memory a GPU cube
/// typically has. Sizing it as a shared tile would have forced the choice
/// between a launch that cannot be issued and a silent drop to global scratch;
/// making it private takes the choice away and leaves only an occupancy cost,
/// which [`ext_rys_max_units`] is there to bound.
///
/// # Where the numbers come from
///
/// Each term below is the allocation list of one arm of `rys_roots_ext_dev`,
/// with `n = nroots`:
///
/// | arm | buffers | words |
/// |---|---|---|
/// | eigensolve tail (`ext_eigen_transform_dev`) | `diag`,`offd`,`eig`,`dwork`,`ework`,`dorig`,`eorig`,`est` (n each) + `c0`,`z` (n² each) | `8n + 2n²` |
/// | f64 Jacobi | `mom` (2n+2), `a`,`b` (n), `s0`,`sm`,`sk` (2n) | `8n + 2` |
/// | f64 Schmidt | `fmt_ints` (2n+2), `cs` ((n+1)²), `rt` (n), `acomp` (n²), `vscr` (n+1) | `2n² + 6n + 4` |
/// | dd Schmidt | `fmh`,`fml` (2n+2), `csh`,`csl`,`csf` ((n+1)²), `rt` (n), `acomp` (n²), `vh`,`vl` (n+1) | `4n² + 13n + 9` |
/// | dd Jacobi / Laguerre | `momh`,`moml` (2n+2), `alh`,`all_`,`beh`,`bel`,`s0h`..`skl` (2n), `ah`,`al`,`bh`,`bl`,`da`,`db` (n) | `30n + 4` |
///
/// A given `nroots` instantiates the two arms its dispatch selects, plus the
/// f64 Schmidt recovery arm `segment_solve` falls back to — allocated a second
/// time because it is a second call site — plus one word of error flag.
/// `ext_rys_scratch_words_match_kernel` pins this arithmetic to the code.
#[must_use]
pub const fn ext_rys_scratch_words(nroots: usize) -> usize {
    let n = nroots;
    let eigen = 8 * n + 2 * n * n;
    let jacobi_f64 = 8 * n + 2;
    let schmidt_f64 = 2 * n * n + 6 * n + 4;
    let lschmidt = 4 * n * n + 13 * n + 9;
    let lwheeler = 30 * n + 4;

    // The recovery arm is a second `ext_schmidt_f64_dev` call site, so its
    // buffers are allocated again rather than reused.
    let recovery = schmidt_f64;
    let flag = 1;

    let arms = if n <= 7 {
        // f64 Jacobi (x <= 11) + f64 Schmidt (x > 11).
        jacobi_f64 + eigen + schmidt_f64
    } else if n == 8 {
        // f64 Jacobi (x <= 11) + dd Schmidt (x > 11).
        jacobi_f64 + eigen + lschmidt
    } else {
        // dd Jacobi (x <= bp) and dd Laguerre (x > bp) share one body with a
        // runtime selector, so one allocation set covers both.
        lwheeler + eigen
    };
    arms + recovery + flag
}

/// [`ext_rys_scratch_words`] in bytes.
#[must_use]
pub const fn ext_rys_scratch_bytes(nroots: usize) -> usize {
    ext_rys_scratch_words(nroots) * std::mem::size_of::<f64>()
}

/// Largest per-cube unit count whose extended-Rys private scratch stays within
/// `budget_bytes`.
///
/// The extended path pays its scratch per *work item*, so the lever that keeps
/// a launch inside a device's local-memory budget is the unit count, not a tile
/// size. Returns at least 1: a cube of one unit is always issuable, and a
/// budget too small even for that is a device fact to report rather than a
/// launch to shrink further.
#[must_use]
pub fn ext_rys_max_units(nroots: usize, budget_bytes: usize) -> u32 {
    let per_unit = ext_rys_scratch_bytes(nroots);
    if per_unit == 0 {
        return 1;
    }
    let units = budget_bytes / per_unit;
    u32::try_from(units.max(1)).unwrap_or(u32::MAX)
}

/// Compute exact `SharedLayout` for Math kernels (Rys/Wheeler/Schmidt/Eigensolver).
pub fn calc_math_layout(
    math_kind: &str,
    n: usize,
    cube_dim: u32,
) -> Result<SharedLayout, cintxRsError> {
    let element_size = std::mem::size_of::<f64>();
    let (variant, descriptor, transform) = match math_kind {
        "jacobi_tridiag" | "llaguerre_tridiag" => (SharedVariant::SharedDescriptor, n * 4, 0),
        "schmidt" | "lschmidt" => (SharedVariant::SharedTiled, n, n * n),
        "cint_diagonalize" => (SharedVariant::SharedTiled, n, n * n),
        // The inline extended-Rys entry allocates its scratch per work item in
        // private memory, so it contributes nothing to the shared budget; the
        // footprint it does have is reported by `ext_rys_scratch_words` and
        // bounded by `ext_rys_max_units`.
        "rys_roots_ext" => (SharedVariant::NoSharedLane, 0, 0),
        _ => (SharedVariant::NoSharedLane, 0, 0),
    };

    Ok(SharedLayout::new(
        variant,
        cube_dim,
        descriptor,
        0,
        0,
        transform,
        0,
        1,
        element_size,
        8,
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Validation & Safety Checks
// ─────────────────────────────────────────────────────────────────────────────

/// Pure validation of shared-memory bounds before dispatch.
pub fn validate_shared_layout_bounds(
    layout: &SharedLayout,
    max_shared_bytes: u32,
) -> Result<(), cintxRsError> {
    if layout.variant == SharedVariant::NoSharedLane
        || layout.variant == SharedVariant::FusedNoShared
    {
        return Ok(());
    }

    if layout.total_bytes > max_shared_bytes as usize {
        return Err(cintxRsError::MemoryLimitExceeded {
            requested: layout.total_bytes,
            limit: max_shared_bytes as usize,
        });
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Layout Catalog Generator
// ─────────────────────────────────────────────────────────────────────────────

/// Generate the full layout catalog across all standard operator and angular momentum configurations.
pub fn generate_layout_catalog(max_shared_bytes: u32) -> serde_json::Value {
    let mut catalog = Vec::new();

    // 1e families
    for op in ["ovlp", "kin", "nuc", "rinv", "grad_bra", "grad_both"] {
        for l in 0..=4u32 {
            if let Ok(layout) = calc_1e_layout(op, l, l, 1, 3, 3, 256) {
                let fits = layout.fits_device_limit(max_shared_bytes, 0.5);
                catalog.push(serde_json::json!({
                    "family": "1e",
                    "operator": op,
                    "li": l,
                    "lj": l,
                    "nroots": 1,
                    "layout": layout,
                    "fits_device_occupancy_reserve": fits,
                }));
            }
        }
    }

    // 2e families
    for l in 0..=3u32 {
        let nroots = l * 2 + 1;
        if let Ok(layout) = calc_2e_layout(l, l, l, l, nroots, 256) {
            let fits = layout.fits_device_limit(max_shared_bytes, 0.5);
            catalog.push(serde_json::json!({
                "family": "2e",
                "li": l,
                "lj": l,
                "lk": l,
                "ll": l,
                "nroots": nroots,
                "layout": layout,
                "fits_device_occupancy_reserve": fits,
            }));
        }
    }

    // 2c2e & 3c2e
    for l in 0..=3u32 {
        if let Ok(layout) = calc_2c2e_layout(l, l, 2, 256) {
            catalog.push(serde_json::json!({
                "family": "2c2e",
                "li": l,
                "lj": l,
                "nroots": 2,
                "layout": layout,
                "fits_device_occupancy_reserve": layout.fits_device_limit(max_shared_bytes, 0.5),
            }));
        }
        if let Ok(layout) = calc_3c2e_layout(l, l, l, 3, 256) {
            catalog.push(serde_json::json!({
                "family": "3c2e",
                "li": l,
                "lj": l,
                "lk": l,
                "nroots": 3,
                "layout": layout,
                "fits_device_occupancy_reserve": layout.fits_device_limit(max_shared_bytes, 0.5),
            }));
        }
    }

    // ECP type-2
    for l in 0..=3u32 {
        if let Ok(layout) = calc_ecp_type2_layout(l, l, 1, 256) {
            catalog.push(serde_json::json!({
                "family": "ecp_type2",
                "li": l,
                "lj": l,
                "lc": 1,
                "layout": layout,
                "fits_device_occupancy_reserve": layout.fits_device_limit(max_shared_bytes, 0.5),
            }));
        }
    }

    serde_json::json!({
        "catalog_version": "1.0",
        "max_shared_bytes": max_shared_bytes,
        "entry_count": catalog.len(),
        "entries": catalog,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Tests & CubeCL 0.10.0 Verification Spike
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // CubeCL 0.10.0 SharedMemory & Barrier Spike Kernel
    #[cube(launch_unchecked)]
    fn spike_shared_memory_reduction_kernel<F: Float + CubeElement>(
        input: &Array<F>,
        output: &mut Array<F>,
    ) {
        let mut smem = SharedMemory::<F>::new(128usize);
        let tid = UNIT_POS as usize;

        if tid < 128 && tid < input.len() {
            smem[tid] = input[tid];
        } else if tid < 128 {
            smem[tid] = F::new(0.0_f32);
        }
        sync_cube();

        // Cooperative tree sum reduction within the cube
        let mut stride = CUBE_DIM / 2;
        while stride > 0 {
            if (UNIT_POS as u32) < stride {
                smem[UNIT_POS as usize] =
                    smem[UNIT_POS as usize] + smem[(UNIT_POS + stride) as usize];
            }
            sync_cube();
            stride /= 2;
        }

        if tid == 0 {
            output[0] = smem[0];
        }
    }

    #[test]
    fn test_shared_layout_creation_and_fit() {
        let layout = SharedLayout::new(
            SharedVariant::SharedRecurrence,
            256,
            16,
            8,
            128,
            0,
            0,
            1,
            8,
            8,
        );
        assert_eq!(layout.total_elements(), 16 + 8 + 128);
        assert_eq!(layout.total_bytes, (16 + 8 + 128) * 8);
        assert!(layout.fits_device_limit(65536, 0.5));
        assert!(!layout.fits_device_limit(1024, 0.5));
    }

    #[test]
    fn test_no_shared_layout_fits_any_limit() {
        let no_shared = SharedLayout::no_shared(256);
        assert_eq!(no_shared.total_bytes, 0);
        assert!(no_shared.fits_device_limit(0, 0.5));
    }

    #[test]
    fn test_calculators_produce_consistent_layouts() {
        let l_1e = calc_1e_layout("ovlp", 1, 1, 1, 3, 3, 256).unwrap();
        assert!(l_1e.total_bytes > 0);
        assert_eq!(l_1e.variant, SharedVariant::SharedRecurrence);

        let l_2e = calc_2e_layout(1, 1, 1, 1, 3, 256).unwrap();
        assert!(l_2e.total_bytes > 0);

        let l_ecp = calc_ecp_type2_layout(2, 2, 1, 256).unwrap();
        assert_eq!(l_ecp.variant, SharedVariant::SharedTiled);
    }

    #[test]
    fn test_spike_shared_memory_reduction_on_cpu() {
        use crate::backend::cpu_backend;
        use cubecl::cpu::CpuRuntime;

        if let Ok(client) = cpu_backend::resolve_cpu_client() {
            let n = 128usize;
            let input_data: Vec<f64> = (1..=n).map(|x| x as f64).collect();
            let expected_sum: f64 = input_data.iter().sum();

            let input_handle = client.create_from_slice(bytemuck::cast_slice(&input_data));
            let output_handle = client.empty(8);

            let input_arg = unsafe { ArrayArg::from_raw_parts(input_handle, n) };
            let output_arg = unsafe { ArrayArg::from_raw_parts(output_handle.clone(), 1) };

            let cube_dim = CubeDim::new_1d(128);
            let cube_count = CubeCount::Static(1, 1, 1);

            unsafe {
                spike_shared_memory_reduction_kernel::launch_unchecked::<f64, CpuRuntime>(
                    &client, cube_count, cube_dim, input_arg, output_arg,
                );
            }

            let result_bytes = client.read_one_unchecked(output_handle);
            let result: f64 = *bytemuck::from_bytes(&result_bytes);

            assert!(
                (result - expected_sum).abs() < 1e-9,
                "Spike reduction produced {}, expected {}",
                result,
                expected_sum
            );
        }
    }

    #[test]
    fn test_write_layout_catalog_artifact() {
        let catalog = generate_layout_catalog(65536);
        let path =
            std::path::Path::new("/tmp/cintx_artifacts/cintx_shared_memory_layout_catalog.json");
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let formatted = serde_json::to_string_pretty(&catalog).expect("serialize catalog");
        std::fs::write(path, formatted).expect("write layout catalog");
        assert!(path.is_file());
    }

    /// Task 33-02 acceptance: the reported per-work-item size matches the
    /// buffers `rys_roots_ext_dev` actually allocates, summed by hand at the
    /// widest instantiation (`nroots = 12`).
    ///
    /// At `nroots = 12` the dispatch is dd Jacobi / dd Laguerre (one body,
    /// runtime selector), the eigensolve tail, and the f64 Schmidt recovery
    /// arm:
    ///
    /// | buffer group | words |
    /// |---|---|
    /// | dd Wheeler: `momh`,`moml` (26) + 10×`2n` (240) + 6×`n` (72) + `alh`..`bel` counted in the 10 | 364 |
    /// | eigensolve: 8×`n` (96) + 2×`n²` (288) | 384 |
    /// | f64 Schmidt recovery: `fmt_ints` 26 + `cs` 169 + `rt` 12 + `acomp` 144 + `vscr` 13 | 364 |
    /// | error flag | 1 |
    #[test]
    fn ext_rys_scratch_words_match_kernel_at_nroots_12() {
        let n = 12usize;

        // dd Wheeler arm: momh+moml (2n+2 each), then the ten 2n buffers
        // (alh, all_, beh, bel, s0h, s0l, smh, sml, skh, skl) and the six n
        // buffers (ah, al, bh, bl, da, db).
        let lwheeler = 2 * (2 * n + 2) + 10 * (2 * n) + 6 * n;
        assert_eq!(lwheeler, 364);

        // Eigensolve tail: diag, offd, eig, dwork, ework, dorig, eorig, est
        // (n each) plus c0 and z (n² each).
        let eigen = 8 * n + 2 * n * n;
        assert_eq!(eigen, 384);

        // f64 Schmidt recovery arm: fmt_ints (2n+2), cs ((n+1)²), rt (n),
        // acomp (n²), vscr (n+1).
        let schmidt = (2 * n + 2) + (n + 1) * (n + 1) + n + n * n + (n + 1);
        assert_eq!(schmidt, 364);

        assert_eq!(
            ext_rys_scratch_words(n),
            lwheeler + eigen + schmidt + 1,
            "the reported extended-Rys scratch size drifted from the kernel's \
             allocation list; recount both before changing either"
        );
        assert_eq!(ext_rys_scratch_words(n), 1113);
        assert_eq!(ext_rys_scratch_bytes(n), 1113 * 8);
    }

    /// Every order in the validated 6..=12 range stays under 16 KiB of private
    /// scratch per work item — the number that decides how many units a cube
    /// can carry.
    ///
    /// The footprint is deliberately **not** asserted to grow with `nroots`.
    /// It dips at `nroots = 9`: order 8 is the only one whose large-`x` arm is
    /// the dd Schmidt solver, which carries three `(n+1)²` coefficient matrices
    /// in dd, and that outweighs the dd Wheeler scratch orders 9..12 use. The
    /// peak is at 12 regardless, so that is where the sizing is pinned.
    #[test]
    fn ext_rys_scratch_stays_within_the_private_budget() {
        for n in 6..=12usize {
            assert!(
                ext_rys_scratch_bytes(n) <= 16 * 1024,
                "nroots={n} needs {} bytes of private scratch per work item",
                ext_rys_scratch_bytes(n)
            );
        }
        // The dd Schmidt arm makes order 8 a local peak, and order 12 the global one.
        assert!(ext_rys_scratch_words(8) > ext_rys_scratch_words(9));
        assert_eq!(
            (6..=12).map(ext_rys_scratch_words).max(),
            Some(ext_rys_scratch_words(12))
        );
    }

    /// The extended path never enters the shared-memory budget: its scratch is
    /// private, so `calc_math_layout` reports zero shared bytes and the unit
    /// count is what bounds the footprint instead.
    #[test]
    fn ext_rys_takes_no_shared_memory_and_caps_units_instead() {
        let layout = calc_math_layout("rys_roots_ext", 12, 256).expect("layout");
        assert_eq!(layout.variant, SharedVariant::NoSharedLane);
        assert_eq!(layout.total_bytes, 0);
        assert!(validate_shared_layout_bounds(&layout, 0).is_ok());

        // 64 KiB of local budget carries 7 units at nroots=12 (1113 words each);
        // a budget too small even for one still yields an issuable cube.
        assert_eq!(ext_rys_max_units(12, 64 * 1024), 7);
        assert_eq!(ext_rys_max_units(12, 1024), 1);
    }
}
