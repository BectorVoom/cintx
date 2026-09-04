//! Device-side Cartesian-to-spherical transform for batched 2e output
//! (`def2_speed_memory_optimization_plan.md` M3).
//!
//! # What this moves, and why it is worth moving
//!
//! A batched `int2e_sph` run evaluates Cartesian blocks on the device and reads
//! them back; the host then contracts each block against the `c2s` coefficient
//! tables and scatters the spherical result into the caller's AO grid. Two costs
//! follow, and on a discrete GPU both are paid over the bus:
//!
//! - **Readback volume.** Cartesian is larger than spherical wherever `l > 1`:
//!   an f shell carries 10 Cartesian components against 7 spherical ones. On
//!   SO2/def2-TZVP the whole work list is 177.9 MiB Cartesian against 99.8 MiB
//!   spherical, so 78 MiB crosses the bus for no reason.
//! - **Host time.** The transform is 8-16% of a batched run's wall clock on the
//!   CPU backend, and it is serial with respect to the device.
//!
//! Doing it on the device turns the readback into the spherical block directly.
//!
//! # Bit-identity, and the one place it would have been lost
//!
//! The gate is not a tolerance — it is that this kernel reproduces
//! [`crate::transform::c2s::cart_to_sph_2e_into`] **element for element**. Two
//! things make that achievable:
//!
//! - The axis contraction is a fixed sum in a fixed order, `for c in 0..ncart`,
//!   so reproducing the loop reproduces the rounding.
//! - **Axes with `l <= 1` are skipped, not applied.** `C2S_L0` and `C2S_L1` are
//!   identity matrices, so applying them looks harmless — but an identity
//!   contraction still evaluates `1.0 * x + 0.0 * y + 0.0 * z`, and
//!   `-0.0 + 0.0` is `+0.0`. A block containing a negative zero would come back
//!   with its sign flipped. The host skips those axes; so does this.
//!
//! # Shape
//!
//! One quartet per work item, grid-stride, no barriers and no lane splitting:
//! the transform is embarrassingly parallel over quartets and has none of the
//! shared-recurrence structure that makes the 2e kernel cooperative.
//!
//! Each work item owns `2 * max_cart_block` scratch elements and ping-pongs
//! between the halves, exactly as the host ping-pongs between two halves of its
//! `Vec`. The final buffer is scattered into the caller's contraction-major
//! layout by the same index arithmetic the host uses.

use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;

/// Cartesian components of angular momentum `l`.
#[cube]
fn ncart_dev(l: u32) -> u32 {
    (l + 1u32) * (l + 2u32) / 2u32
}

/// Spherical components of angular momentum `l`.
#[cube]
fn nsph_dev(l: u32) -> u32 {
    2u32 * l + 1u32
}

/// Contract one axis of a block against the `c2s` coefficients.
///
/// The block is `[outer][ncart(l)][inner]`; the result is
/// `[outer][nsph(l)][inner]`. Verbatim `c2s_axis`, including the order of the
/// inner sum, which is what the bit-identity gate rests on.
///
/// [`c2s_axis_scratch`] below is this same contraction loop, copied rather
/// than shared, only because CubeCL cannot pass one `&mut Array` as both this
/// function's `src` and `dst` (the ping-pong-within-one-buffer case). A change
/// to the contraction or accumulation logic here must be made there too, or
/// the cart-sourced and scratch-sourced transform steps silently diverge —
/// this kernel has a CHANGELOG history of exactly that kind of bug.
#[cube]
#[allow(clippy::too_many_arguments)]
fn c2s_axis_dev<F: Float>(
    src: &Array<F>,
    src_base: u32,
    dst: &mut Array<F>,
    dst_base: u32,
    table: &Array<F>,
    table_base: u32,
    l: u32,
    outer: u32,
    inner: u32,
) {
    let nc = ncart_dev(l);
    let ns = nsph_dev(l);
    let mut o = 0u32;
    while o < outer {
        let mut m = 0u32;
        while m < ns {
            let row = table_base + m * nc;
            let out_row = dst_base + (o * ns + m) * inner;
            let mut t = 0u32;
            while t < inner {
                let mut sum = F::new(0.0_f32);
                let mut c = 0u32;
                while c < nc {
                    sum += table[(row + c) as usize]
                        * src[(src_base + (o * nc + c) * inner + t) as usize];
                    c += 1u32;
                }
                dst[(out_row + t) as usize] = sum;
                t += 1u32;
            }
            m += 1u32;
        }
        o += 1u32;
    }
}

/// Transform every quartet's Cartesian blocks into the chunk's spherical output.
///
/// `scratch` is `2 * scratch_half` elements per work item: the two ping-pong
/// halves. `cart` is one dispatch group's Cartesian buffer; `sph` is the chunk's
/// spherical output, which the caller reads back instead of `cart`.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn two_electron_c2s_kernel<F: Float>(
    cart: &Array<F>,
    quartets: &Array<u32>,
    sph_offsets: &Array<u32>,
    class_shape: &Array<u32>,
    shell_meta: &Array<u32>,
    c2s_table: &Array<F>,
    c2s_offset: &Array<u32>,
    scratch: &mut Array<F>,
    sph: &mut Array<F>,
    n_quartets: u32,
    n_slots: u32,
    scratch_half: u32,
    #[comptime] shape_stride: u32,
) {
    // Derived rather than taken from `ABSOLUTE_POS`, and the stride from the
    // launch argument rather than `CUBE_COUNT`: cubecl-cpu 0.10 rejects that
    // builtin outright, which is the same reason the 2e kernel derives both.
    let slot = (CUBE_POS as u32) * (CUBE_DIM as u32) + (UNIT_POS as u32);
    let scratch_a = slot * 2u32 * scratch_half;
    let scratch_b = scratch_a + scratch_half;

    let mut qi = slot;
    while qi < n_quartets {
        let qrow = qi * 6u32;
        let si = quartets[qrow as usize];
        let sj = quartets[(qrow + 1u32) as usize];
        let sk = quartets[(qrow + 2u32) as usize];
        let sl = quartets[(qrow + 3u32) as usize];

        // `cart_off` and the class index already travel in the 2e group's own
        // quartet row; only the destination is new, so the transform's table is
        // one `u32` per quartet rather than a parallel copy of what exists.
        let cart_off = quartets[(qrow + 4u32) as usize];
        let cls = quartets[(qrow + 5u32) as usize];
        let sph_off = sph_offsets[qi as usize];

        let srow = cls * shape_stride;
        let li = class_shape[srow as usize];
        let lj = class_shape[(srow + 1u32) as usize];
        let lk = class_shape[(srow + 2u32) as usize];
        let ll = class_shape[(srow + 3u32) as usize];

        let nci = ncart_dev(li);
        let ncj = ncart_dev(lj);
        let nck = ncart_dev(lk);
        let ncl = ncart_dev(ll);
        let nsi = nsph_dev(li);
        let nsj = nsph_dev(lj);
        let nsk = nsph_dev(lk);
        let nsl = nsph_dev(ll);
        let cart_block = nci * ncj * nck * ncl;

        let nctr_i = shell_meta[(si * 4u32 + 3u32) as usize];
        let nctr_j = shell_meta[(sj * 4u32 + 3u32) as usize];
        let nctr_k = shell_meta[(sk * 4u32 + 3u32) as usize];
        let nctr_l = shell_meta[(sl * 4u32 + 3u32) as usize];

        // The four axis extents, innermost first, exactly as
        // `cart_to_sph_2e_into` lists them: `outer` counts the axes above that
        // are still Cartesian, `inner` the axes below that are already
        // spherical.
        let outer_i = ncl * nck * ncj;
        let outer_j = ncl * nck;
        let inner_j = nsi;
        let inner_k = nsj * nsi;
        let inner_l = nsk * nsj * nsi;

        // Caller-layout strides: the spherical block is contraction-major with
        // `i` fastest, exactly as the host scatter writes it.
        let di = nctr_i * nsi;
        let dj = nctr_j * nsj;
        let dk = nctr_k * nsk;

        let mut ci = 0u32;
        while ci < nctr_i {
            let mut cj = 0u32;
            while cj < nctr_j {
                let mut ck = 0u32;
                while ck < nctr_k {
                    let mut cl = 0u32;
                    while cl < nctr_l {
                        let quad = ((ci * nctr_j + cj) * nctr_k + ck) * nctr_l + cl;
                        let src0 = cart_off + quad * cart_block;

                        // `cur` names where the current intermediate lives: `0`
                        // is `cart` itself (no step has run yet), `1` is scratch
                        // half A, `2` is half B. Written as statement-level
                        // mutation rather than an `if` expression, because a
                        // value-returning conditional on a runtime predicate is
                        // not something the `#[cube]` frontend lowers the way
                        // ordinary Rust would.
                        let mut cur: u32 = 0u32;
                        let mut cur_base: u32 = src0;

                        // The four steps, written out rather than looped:
                        // each takes its own `(l, outer, inner)`, and a runtime
                        // index into a local array is not something the
                        // `#[cube]` frontend can type here.
                        //
                        // `l <= 1` axes are SKIPPED, not applied. Their matrices
                        // are the identity, but an identity contraction still
                        // evaluates `1.0*x + 0.0*y`, and `-0.0 + 0.0` is `+0.0`.
                        // The host skips them; so does this.
                        //
                        // The four blocks below (li, lj, lk, ll) are one
                        // ping-pong step copied four times rather than a
                        // shared helper, for the same CubeCL limitation: no
                        // typed way to close over `cur`/`cur_base` across a
                        // runtime-indexed step here. A fix to the ping-pong
                        // bookkeeping (the `cur`/`dst_base` toggle) or to the
                        // `l <= 1` skip must be applied identically to all
                        // four, or they silently drift apart.
                        if li > 1u32 {
                            let mut dst_base = scratch_a;
                            if cur == 1u32 {
                                dst_base = scratch_b;
                            }
                            if cur == 0u32 {
                                c2s_axis_dev::<F>(
                                    cart,
                                    cur_base,
                                    scratch,
                                    dst_base,
                                    c2s_table,
                                    c2s_offset[li as usize],
                                    li,
                                    outer_i,
                                    1u32,
                                );
                            } else {
                                c2s_axis_scratch::<F>(
                                    scratch,
                                    cur_base,
                                    dst_base,
                                    c2s_table,
                                    c2s_offset[li as usize],
                                    li,
                                    outer_i,
                                    1u32,
                                );
                            }
                            if cur == 1u32 {
                                cur = 2u32;
                            } else {
                                cur = 1u32;
                            }
                            cur_base = dst_base;
                        }
                        if lj > 1u32 {
                            let mut dst_base = scratch_a;
                            if cur == 1u32 {
                                dst_base = scratch_b;
                            }
                            if cur == 0u32 {
                                c2s_axis_dev::<F>(
                                    cart,
                                    cur_base,
                                    scratch,
                                    dst_base,
                                    c2s_table,
                                    c2s_offset[lj as usize],
                                    lj,
                                    outer_j,
                                    inner_j,
                                );
                            } else {
                                c2s_axis_scratch::<F>(
                                    scratch,
                                    cur_base,
                                    dst_base,
                                    c2s_table,
                                    c2s_offset[lj as usize],
                                    lj,
                                    outer_j,
                                    inner_j,
                                );
                            }
                            if cur == 1u32 {
                                cur = 2u32;
                            } else {
                                cur = 1u32;
                            }
                            cur_base = dst_base;
                        }
                        if lk > 1u32 {
                            let mut dst_base = scratch_a;
                            if cur == 1u32 {
                                dst_base = scratch_b;
                            }
                            if cur == 0u32 {
                                c2s_axis_dev::<F>(
                                    cart,
                                    cur_base,
                                    scratch,
                                    dst_base,
                                    c2s_table,
                                    c2s_offset[lk as usize],
                                    lk,
                                    ncl,
                                    inner_k,
                                );
                            } else {
                                c2s_axis_scratch::<F>(
                                    scratch,
                                    cur_base,
                                    dst_base,
                                    c2s_table,
                                    c2s_offset[lk as usize],
                                    lk,
                                    ncl,
                                    inner_k,
                                );
                            }
                            if cur == 1u32 {
                                cur = 2u32;
                            } else {
                                cur = 1u32;
                            }
                            cur_base = dst_base;
                        }
                        if ll > 1u32 {
                            let mut dst_base = scratch_a;
                            if cur == 1u32 {
                                dst_base = scratch_b;
                            }
                            if cur == 0u32 {
                                c2s_axis_dev::<F>(
                                    cart,
                                    cur_base,
                                    scratch,
                                    dst_base,
                                    c2s_table,
                                    c2s_offset[ll as usize],
                                    ll,
                                    1u32,
                                    inner_l,
                                );
                            } else {
                                c2s_axis_scratch::<F>(
                                    scratch,
                                    cur_base,
                                    dst_base,
                                    c2s_table,
                                    c2s_offset[ll as usize],
                                    ll,
                                    1u32,
                                    inner_l,
                                );
                            }
                            if cur == 1u32 {
                                cur = 2u32;
                            } else {
                                cur = 1u32;
                            }
                            cur_base = dst_base;
                        }

                        // ── Scatter into the caller's contraction-major grid ──
                        let mut ml = 0u32;
                        while ml < nsl {
                            let lidx = cl * nsl + ml;
                            let mut mk = 0u32;
                            while mk < nsk {
                                let kidx = ck * nsk + mk;
                                let mut mj = 0u32;
                                while mj < nsj {
                                    let jidx = cj * nsj + mj;
                                    let mut mi = 0u32;
                                    while mi < nsi {
                                        let iidx = ci * nsi + mi;
                                        let src = mi + nsi * (mj + nsj * (mk + nsk * ml));
                                        let dst = iidx + di * (jidx + dj * (kidx + dk * lidx));
                                        let mut value = cart[(cur_base + src) as usize];
                                        if cur != 0u32 {
                                            value = scratch[(cur_base + src) as usize];
                                        }
                                        sph[(sph_off + dst) as usize] = value;
                                        mi += 1u32;
                                    }
                                    mj += 1u32;
                                }
                                mk += 1u32;
                            }
                            ml += 1u32;
                        }

                        cl += 1u32;
                    }
                    ck += 1u32;
                }
                cj += 1u32;
            }
            ci += 1u32;
        }

        qi += n_slots;
    }
}

/// [`c2s_axis_dev`] with both operands inside the same scratch array.
///
/// A separate function because `scratch` is `&mut` at the call site and cannot
/// be passed as both the source and the destination; the ping-pong halves are
/// disjoint, so reading one while writing the other is sound, and this is where
/// that is stated once rather than at every call site.
#[cube]
#[allow(clippy::too_many_arguments)]
fn c2s_axis_scratch<F: Float>(
    buffer: &mut Array<F>,
    src_base: u32,
    dst_base: u32,
    table: &Array<F>,
    table_base: u32,
    l: u32,
    outer: u32,
    inner: u32,
) {
    let nc = ncart_dev(l);
    let ns = nsph_dev(l);
    let mut o = 0u32;
    while o < outer {
        let mut m = 0u32;
        while m < ns {
            let row = table_base + m * nc;
            let out_row = dst_base + (o * ns + m) * inner;
            let mut t = 0u32;
            while t < inner {
                let mut sum = F::new(0.0_f32);
                let mut c = 0u32;
                while c < nc {
                    sum += table[(row + c) as usize]
                        * buffer[(src_base + (o * nc + c) * inner + t) as usize];
                    c += 1u32;
                }
                buffer[(out_row + t) as usize] = sum;
                t += 1u32;
            }
            m += 1u32;
        }
        o += 1u32;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Host side
// ─────────────────────────────────────────────────────────────────────────────

/// The `c2s` coefficient tables on the device.
///
/// 19 176 `f64` plus 17 offsets — about 154 KB, uploaded once per residency and
/// shared by every dispatch, the same arrangement the extended-Rys constants use.
#[derive(Debug)]
pub struct C2sHandles {
    pub(crate) table: cubecl::server::Handle,
    pub(crate) offset: cubecl::server::Handle,
    pub(crate) table_len: usize,
    pub(crate) offset_len: usize,
}

/// Upload the frozen `c2s` tables.
pub(crate) fn upload_c2s_tables<R: Runtime>(client: &ComputeClient<R>) -> C2sHandles {
    let table = crate::transform::c2s_data::C2S_TABLE;
    let offsets: Vec<u32> = crate::transform::c2s_data::C2S_OFFSET
        .iter()
        .map(|value| *value as u32)
        .collect();
    C2sHandles {
        table: client.create_from_slice(f64::as_bytes(&table)),
        offset: client.create_from_slice(u32::as_bytes(&offsets)),
        table_len: table.len(),
        offset_len: offsets.len(),
    }
}

impl C2sHandles {
    /// Bytes these tables cost to upload.
    #[must_use]
    pub fn upload_bytes(&self) -> usize {
        self.table_len * std::mem::size_of::<f64>() + self.offset_len * std::mem::size_of::<u32>()
    }
}

/// Everything one group's transform binds.
pub(crate) struct C2sDispatch<'a, R: Runtime> {
    pub(crate) client: &'a ComputeClient<R>,
    pub(crate) cart: cubecl::server::Handle,
    pub(crate) cart_len: usize,
    pub(crate) quartets: cubecl::server::Handle,
    pub(crate) quartets_len: usize,
    pub(crate) sph_offsets: cubecl::server::Handle,
    pub(crate) sph_offsets_len: usize,
    pub(crate) class_shape: cubecl::server::Handle,
    pub(crate) class_shape_len: usize,
    pub(crate) shell_meta: cubecl::server::Handle,
    pub(crate) shell_meta_len: usize,
    pub(crate) tables: &'a C2sHandles,
    pub(crate) sph: cubecl::server::Handle,
    pub(crate) sph_len: usize,
    pub(crate) n_quartets: u32,
    /// Widest Cartesian contraction block in this group — one ping-pong half.
    pub(crate) scratch_half: u32,
    pub(crate) shape_stride: u32,
    /// This run's shared ping-pong scratch slab (M4.3), sized by
    /// [`c2s_scratch_len`] to the widest group the caller will dispatch —
    /// this group's own extent is `2 * scratch_half` per slot, at most that.
    pub(crate) scratch: cubecl::server::Handle,
}

/// Ceiling on the transform's ping-pong scratch, matched to the 2e path's own.
const MAX_C2S_SCRATCH_BYTES: usize = 256 * 1024 * 1024;

/// Launch geometry for one group's transform: how many cubes, what shape, and
/// how many ping-pong slots that implies.
///
/// Factored out of [`launch_c2s`] so [`c2s_scratch_len`] can compute a group's
/// scratch requirement — for the run-wide pre-sizing pass (M4.3) — from
/// exactly the expression the launch itself derives its dimensions from,
/// rather than a second copy of the same arithmetic.
struct C2sLaunchGeometry {
    cube_dim: CubeDim,
    cubes: u32,
    n_slots: usize,
}

fn c2s_launch_geometry<R: Runtime>(
    client: &ComputeClient<R>,
    n_quartets: usize,
    scratch_half: u32,
) -> C2sLaunchGeometry {
    let hardware = crate::plane::launch_hardware(client);
    let per_slot_bytes = 2 * scratch_half as usize * std::mem::size_of::<f64>();
    let by_memory = (MAX_C2S_SCRATCH_BYTES / per_slot_bytes.max(1)).max(1);

    // One slot per quartet is the ceiling that matters; beyond that slots idle.
    let want = n_quartets.min(by_memory).max(1);
    let cube_dim = if hardware.has_planes {
        crate::plane::standard_plane_cube_dim()
    } else {
        // On the CPU runtime a unit is an OS thread and `cube_count` lowers to a
        // sequential loop, so the units are the only parallelism axis — the same
        // finding Task 34-A0 recorded for the 2e kernel.
        CubeDim::new_1d(crate::plane::per_unit_width(client, want, 1, by_memory).max(1))
    };
    let per_cube = cube_dim.num_elems() as usize;
    let cubes = if hardware.has_planes {
        crate::plane::grid_cube_count(client, want.div_ceil(per_cube.max(1)))
    } else {
        1
    };
    let n_slots = (cubes as usize * per_cube).max(1);
    C2sLaunchGeometry {
        cube_dim,
        cubes: cubes as u32,
        n_slots,
    }
}

/// Device elements one group's transform scratch will need (M4.3).
///
/// Lets a caller pre-size one buffer for the widest group in a run instead of
/// `launch_c2s` allocating a fresh one per group — the same move M4.1 already
/// made for the 2e kernel's own G-tensor slab.
pub(crate) fn c2s_scratch_len<R: Runtime>(
    client: &ComputeClient<R>,
    n_quartets: usize,
    scratch_half: u32,
) -> usize {
    c2s_launch_geometry(client, n_quartets, scratch_half).n_slots * 2 * scratch_half as usize
}

/// Launch the device transform for one dispatch group.
///
/// The work items are independent quartets, so the geometry is the plain one:
/// as many slots as the device will run and the scratch will hold, walked
/// grid-stride. No barriers, so no cooperative shape and no `per_unit` split.
pub(crate) fn launch_c2s<R: Runtime>(dispatch: C2sDispatch<'_, R>) {
    let geometry = c2s_launch_geometry(
        dispatch.client,
        dispatch.n_quartets as usize,
        dispatch.scratch_half,
    );
    let scratch_len = geometry.n_slots * 2 * dispatch.scratch_half as usize;

    // SAFETY: `dispatch.scratch` is at least `scratch_len` elements — M4.3
    // sizes it to the widest group in the run, so a narrower group's launch
    // binds a leading, sufficient slice of a slab that may carry another
    // group's leftovers, exactly as the 2e kernel's shared G-tensor slab
    // does. In-kernel indices are bounded by `n_quartets`, by the class index
    // carried in each quartet row, by the per-shell `nctr` read from
    // `shell_meta`, and by `scratch_half`, which the host sized to the
    // group's widest Cartesian contraction block — the largest intermediate
    // any axis step can produce, since every axis either shrinks or is
    // skipped.
    unsafe {
        two_electron_c2s_kernel::launch_unchecked::<f64, R>(
            dispatch.client,
            crate::plane::cube_count_1d(geometry.cubes),
            geometry.cube_dim,
            ArrayArg::from_raw_parts(dispatch.cart, dispatch.cart_len),
            ArrayArg::from_raw_parts(dispatch.quartets, dispatch.quartets_len),
            ArrayArg::from_raw_parts(dispatch.sph_offsets, dispatch.sph_offsets_len),
            ArrayArg::from_raw_parts(dispatch.class_shape, dispatch.class_shape_len),
            ArrayArg::from_raw_parts(dispatch.shell_meta, dispatch.shell_meta_len),
            ArrayArg::from_raw_parts(dispatch.tables.table.clone(), dispatch.tables.table_len),
            ArrayArg::from_raw_parts(dispatch.tables.offset.clone(), dispatch.tables.offset_len),
            ArrayArg::from_raw_parts(dispatch.scratch, scratch_len),
            ArrayArg::from_raw_parts(dispatch.sph, dispatch.sph_len),
            dispatch.n_quartets,
            geometry.n_slots as u32,
            dispatch.scratch_half,
            dispatch.shape_stride,
        );
    }
}

/// Does `CINTX_2E_TRANSFORM` ask for the device transform?
///
/// `device` turns it on; anything else (and unset) leaves the host transform in
/// place. Off by default until it is measured on a backend where the readback is
/// a real transfer — on the CubeCL CPU runtime the "device" is the same cores,
/// so moving the transform there moves the work without moving the cost.
#[must_use]
pub fn device_transform_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CINTX_2E_TRANSFORM").is_ok_and(|value| value.eq_ignore_ascii_case("device"))
    })
}
