//! Cartesian-to-spherical (c2s) transform coefficients and functions.
//!
//! Condon-Shortley coefficient matrices extracted from libcint `cart2sph.c`
//! `g_trans_cart2sph[]` array. The layout is:
//!   - rows = spherical components (2l+1), ordered m = -l, -l+1, ..., 0, ..., l
//!   - cols = cartesian components (l+1)(l+2)/2, in libcint ordering
//!
//! Reference: H. B. Schlegel and M. J. Frisch, Int. J. Quant. Chem., 54(1995), 83-87.

// Transcribed verbatim from vendored libcint 6.1.3. Result compatibility with
// upstream is decided by the exact bits these literals feed the kernels, so a
// literal is never truncated to the shortest form that round-trips — the same
// provenance rationale the `clippy::approx_constant` allows in this crate carry.
#![allow(clippy::excessive_precision)]
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

use cintx_core::{CintFloat, cintxRsError};

// ──────────────────────────────────────────────────────────────────────────
//  Helper dimension functions
// ──────────────────────────────────────────────────────────────────────────

/// Number of Cartesian components for angular momentum l: (l+1)(l+2)/2
pub fn ncart(l: u8) -> usize {
    ((l as usize + 1) * (l as usize + 2)) / 2
}

/// Number of spherical components for angular momentum l: 2l+1
pub fn nsph(l: u8) -> usize {
    2 * l as usize + 1
}

// ──────────────────────────────────────────────────────────────────────────
//  Condon-Shortley coefficient matrices (from libcint g_trans_cart2sph[])
// ──────────────────────────────────────────────────────────────────────────

/// s-shell (l=0): 1 sph x 1 cart.
/// g_trans_cart2sph offset 0, 1 element.
pub const C2S_L0: [[f64; 1]; 1] = [[1.0]];

/// p-shell (l=1): 3 sph x 3 cart (px, py, pz ordering).
///
/// From libcint `cart2sph.c` `g_trans_cart2sph[]` p-shell section (default, no PYPZPX):
///   sph[0] = px  -> [1, 0, 0]
///   sph[1] = py  -> [0, 1, 0]
///   sph[2] = pz  -> [0, 0, 1]
///
/// Libcint uses (px, py, pz) as the spherical p ordering — this is the identity
/// transform from Cartesian (px, py, pz) to spherical. The CINTcommon_fac_sp(1)
/// prefactor (0.4886) is applied externally in the primitive loop, not here.
pub const C2S_L1: [[f64; 3]; 3] = [
    // sph[0] = px
    [1.0, 0.0, 0.0],
    // sph[1] = py
    [0.0, 1.0, 0.0],
    // sph[2] = pz
    [0.0, 0.0, 1.0],
];

/// d-shell (l=2): 5 sph x 6 cart.
/// g_trans_cart2sph offset 10, 30 elements.
/// Rows: m = -2 (dxy), m = -1 (dyz), m = 0 (dz2), m = +1 (dxz), m = +2 (dx2-y2)
/// Cols: cartesian (xx, xy, xz, yy, yz, zz)
pub const C2S_L2: [[f64; 6]; 5] = [
    // m=-2: dxy
    [0.0, 1.092548430592079070, 0.0, 0.0, 0.0, 0.0],
    // m=-1: dyz
    [0.0, 0.0, 0.0, 0.0, 1.092548430592079070, 0.0],
    // m= 0: dz2
    [
        -0.315391565252520002,
        0.0,
        0.0,
        -0.315391565252520002,
        0.0,
        0.630783130505040012,
    ],
    // m=+1: dxz
    [0.0, 0.0, 1.092548430592079070, 0.0, 0.0, 0.0],
    // m=+2: dx2-y2
    [
        0.546274215296039535,
        0.0,
        0.0,
        -0.546274215296039535,
        0.0,
        0.0,
    ],
];

/// f-shell (l=3): 7 sph x 10 cart.
/// g_trans_cart2sph offset 40, 70 elements.
/// Rows: m = -3..+3
/// Cols: cartesian (xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz)
pub const C2S_L3: [[f64; 10]; 7] = [
    // m=-3: fyx2 (f-3)
    [
        0.0,
        1.770130769779930531,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.590043589926643510,
        0.0,
        0.0,
        0.0,
    ],
    // m=-2: fxyz (f-2)
    [
        0.0,
        0.0,
        0.0,
        0.0,
        2.890611442640554055,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    // m=-1: fyz2 (f-1)
    [
        0.0,
        -0.457045799464465739,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.457045799464465739,
        0.0,
        1.828183197857862944,
        0.0,
    ],
    // m= 0: fz3 (f0)
    [
        0.0,
        0.0,
        -1.119528997770346170,
        0.0,
        0.0,
        0.0,
        0.0,
        -1.119528997770346170,
        0.0,
        0.746352665180230782,
    ],
    // m=+1: fxz2 (f1)
    [
        -0.457045799464465739,
        0.0,
        0.0,
        -0.457045799464465739,
        0.0,
        1.828183197857862944,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    // m=+2: fzx2 (f2)
    [
        0.0,
        0.0,
        1.445305721320277020,
        0.0,
        0.0,
        0.0,
        0.0,
        -1.445305721320277020,
        0.0,
        0.0,
    ],
    // m=+3: fx3 (f3)
    [
        0.590043589926643510,
        0.0,
        0.0,
        -1.770130769779930530,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
];

/// g-shell (l=4): 9 sph x 15 cart.
/// g_trans_cart2sph offset 110, 135 elements.
/// Rows: m = -4..+4
/// Cols: cartesian (xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz, xyzz, xzzz, yyyy, yyyz, yyzz, yzzz, zzzz)
pub const C2S_L4: [[f64; 15]; 9] = [
    // m=-4: gyx3 (g-4)
    [
        0.0,
        2.503342941796704538,
        0.0,
        0.0,
        0.0,
        0.0,
        -2.503342941796704530,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    // m=-3: gx2yz (g-3)
    [
        0.0,
        0.0,
        0.0,
        0.0,
        5.310392309339791593,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -1.770130769779930530,
        0.0,
        0.0,
        0.0,
    ],
    // m=-2: gxyz2 (g-2)
    [
        0.0,
        -0.946174695757560014,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.946174695757560014,
        0.0,
        5.677048174545360108,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    // m=-1: gyz3 (g-1)
    [
        0.0,
        0.0,
        0.0,
        0.0,
        -2.007139630671867500,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -2.007139630671867500,
        0.0,
        2.676186174229156671,
        0.0,
    ],
    // m= 0: gz4 (g0)
    [
        0.317356640745612911,
        0.0,
        0.0,
        0.634713281491225822,
        0.0,
        -2.538853125964903290,
        0.0,
        0.0,
        0.0,
        0.0,
        0.317356640745612911,
        0.0,
        -2.538853125964903290,
        0.0,
        0.846284375321634430,
    ],
    // m=+1: gxz3 (g1)
    [
        0.0,
        0.0,
        -2.007139630671867500,
        0.0,
        0.0,
        0.0,
        0.0,
        -2.007139630671867500,
        0.0,
        2.676186174229156671,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    // m=+2: gx2z2 (g2)
    [
        -0.473087347878780002,
        0.0,
        0.0,
        0.0,
        0.0,
        2.838524087272680054,
        0.0,
        0.0,
        0.0,
        0.0,
        0.473087347878780009,
        0.0,
        -2.838524087272680050,
        0.0,
        0.0,
    ],
    // m=+3: gzx3 (g3)
    [
        0.0,
        0.0,
        1.770130769779930531,
        0.0,
        0.0,
        0.0,
        0.0,
        -5.310392309339791590,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    // m=+4: gy4 (g4)
    [
        0.625835735449176134,
        0.0,
        0.0,
        -3.755014412695056800,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.625835735449176134,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
];

// ──────────────────────────────────────────────────────────────────────────
//  Transform function
// ──────────────────────────────────────────────────────────────────────────

/// Apply cart-to-sph transform for a 1-electron shell pair (li, lj).
///
/// Input `cart_buf`: flat row-major array of shape `[ncart(lj) * ncart(li)]`
///   (j is the outer/slow index, i is the inner/fast index — bra=i, ket=j).
///
/// Output `sph_buf`: flat row-major array of shape `[nsph(lj) * nsph(li)]`.
///
/// The transform applies:
///   1. Bra (i-axis): multiply T[li] (nsph_i x ncart_i) from the left.
///   2. Ket (j-axis): multiply T[lj] (nsph_j x ncart_j) from the left.
///
/// For `l <= 1` both axes are identity (no-op).
///
/// Generic over `F: CintFloat`.
/// The c2s coefficient table is FROZEN f64; each coefficient is cast to `F`
/// via `F::from_f64_lossy` at the accumulation site (PATTERNS line 162).
/// The f64 monomorphization is byte-identical to the pre-refactor concrete function.
pub fn cart_to_sph_1e<F: CintFloat>(cart_buf: &[F], sph_buf: &mut [F], li: u8, lj: u8) {
    let mut scratch = Vec::new();
    cart_to_sph_1e_into(cart_buf, sph_buf, li, lj, &mut scratch);
}

/// [`cart_to_sph_1e`] with the intermediate buffer supplied by the caller.
///
/// Batched evaluation calls this once per contraction block of every pair in a
/// work list, so both differences from the allocating form matter there:
/// `scratch` replaces a fresh `Vec` per call, and [`c2s_apply`] skips the axes
/// where the transform is the identity (`l <= 1`). A transformed axis is
/// unchanged bit for bit; a skipped one is a copy in place of a multiply by an
/// identity matrix (Task 36-T0/36-T1).
pub fn cart_to_sph_1e_into<F: CintFloat>(
    cart_buf: &[F],
    sph_buf: &mut [F],
    li: u8,
    lj: u8,
    scratch: &mut Vec<F>,
) {
    let (nci, ncj) = (ncart(li), ncart(lj));
    let (nsi, nsj) = (nsph(li), nsph(lj));

    debug_assert_eq!(cart_buf.len(), nci * ncj);
    debug_assert_eq!(sph_buf.len(), nsi * nsj);

    // `[j][i]` with `i` fastest: bra first, then ket.
    c2s_apply(cart_buf, sph_buf, scratch, &[(li, ncj, 1), (lj, 1, nsi)]);
}

/// Highest `l` the Cartesian-to-spherical transform can express.
///
/// This is libcint's own ceiling: its `g_c2s` table has entries for `l = 0..=15`
/// and nothing above, so beyond it there is no upstream reference to be
/// compatible with. [`ensure_c2s_supported`] is the typed guard callers use.
pub use super::c2s_data::C2S_LMAX;

/// Can the transform express a shell of angular momentum `l`?
#[inline]
#[must_use]
pub fn c2s_supports_l(l: u8) -> bool {
    l <= C2S_LMAX
}

/// Fail-closed guard for callers that have an error channel.
///
/// # Errors
/// [`cintxRsError::UnsupportedApi`] when `l` exceeds [`C2S_LMAX`].
pub fn ensure_c2s_supported(l: u8) -> Result<(), cintxRsError> {
    if c2s_supports_l(l) {
        Ok(())
    } else {
        Err(cintxRsError::UnsupportedApi {
            requested: format!(
                "spherical representation for l={l}: the cart-to-sph coefficient                  table covers l<={C2S_LMAX}, which is libcint's own ceiling"
            ),
        })
    }
}

/// Retrieve a single Condon-Shortley coefficient T[l][m_row][cart_col].
///
/// `l`        : angular momentum, `0..=C2S_LMAX`
/// `m_row`    : spherical index (0-based, maps to m = -l, ..., +l)
/// `cart_col` : cartesian index (0-based)
///
/// # Panics
/// When `l > C2S_LMAX`. This is deliberately a panic and not a `0.0` return:
/// returning zero is what made an `l >= 5` shell come back silently zeroed with
/// an `Ok` status, at any Rys order. Callers with an error channel reject the
/// shell through [`ensure_c2s_supported`] first — the safe and raw APIs both do,
/// in `validate_shell_tuple` — so reaching this assertion means a caller skipped
/// its guard, which is a defect rather than an input.
#[inline]
pub fn c2s_coeff(l: u8, m_row: usize, cart_col: usize) -> f64 {
    assert!(
        c2s_supports_l(l),
        "c2s_coeff: l={l} exceeds the coefficient table's ceiling {C2S_LMAX};          callers must reject the shell with `ensure_c2s_supported` first"
    );
    let block = super::c2s_data::C2S_OFFSET[l as usize];
    super::c2s_data::C2S_TABLE[block + m_row * ncart(l) + cart_col]
}

// ──────────────────────────────────────────────────────────────────────────
//  Staging transform (public API compat)
// ──────────────────────────────────────────────────────────────────────────

/// Apply cart-to-sph transform for a 2-center-2-electron shell pair (li, lk).
///
/// Input `cart`: flat column-major array of shape `[ncart(lk) * ncart(li)]`
///   (k is the outer/slow index, i is the inner/fast index).
///
/// Output: flat column-major array of shape `[nsph(lk) * nsph(li)]`.
///
/// Transform order: i-axis first (bra), then k-axis (ket), following the same
/// convention as `cart_to_sph_1e`.
///
/// Generic over `F: CintFloat`. The c2s coefficient table is FROZEN f64;
/// each coefficient is cast to `F` via `F::from_f64_lossy` at the accumulation site.
pub fn cart_to_sph_2c2e<F: CintFloat>(cart: &[F], li: u8, lk: u8) -> Vec<F> {
    let mut sph = vec![F::zero(); nsph(lk) * nsph(li)];
    let mut scratch = Vec::new();
    cart_to_sph_2c2e_into(cart, li, lk, &mut sph, &mut scratch);
    sph
}

/// [`cart_to_sph_2c2e`] writing into caller-owned buffers.
///
/// Routed through [`c2s_apply`], so identity axes (`l <= 1`) are skipped and a
/// transformed axis is unchanged bit for bit.
///
/// `sph` must be exactly `nsph(lk) * nsph(li)` long; `scratch` is grown as
/// needed and may start empty (Task 36-T0/36-T1).
pub fn cart_to_sph_2c2e_into<F: CintFloat>(
    cart: &[F],
    li: u8,
    lk: u8,
    sph: &mut [F],
    scratch: &mut Vec<F>,
) {
    let (nci, nck) = (ncart(li), ncart(lk));
    let (nsi, nsk) = (nsph(li), nsph(lk));

    debug_assert_eq!(cart.len(), nci * nck);
    debug_assert_eq!(sph.len(), nsk * nsi);

    // `[k][i]` with `i` fastest: bra first, then ket.
    c2s_apply(cart, sph, scratch, &[(li, nck, 1), (lk, 1, nsi)]);
}

/// Apply cart-to-sph transform for a 3-center-1-electron shell triple (li, lj, lk).
///
/// Input `cart`: flat column-major array of shape `[ncart(lk) * ncart(lj) * ncart(li)]`
///   (k is outermost/slowest, i is innermost/fastest).
///
/// Output: flat column-major array of shape `[nsph(lk) * nsph(lj) * nsph(li)]`.
///
/// Transform order: i-axis first, then j-axis, then k-axis.
///
/// Generic over `F: CintFloat`. The c2s coefficient table is FROZEN f64;
/// each coefficient is cast to `F` via `F::from_f64_lossy` at the accumulation site.
pub fn cart_to_sph_3c1e<F: CintFloat>(cart: &[F], li: u8, lj: u8, lk: u8) -> Vec<F> {
    let mut sph = vec![F::zero(); nsph(lk) * nsph(lj) * nsph(li)];
    let mut scratch = Vec::new();
    cart_to_sph_3c1e_into(cart, li, lj, lk, &mut sph, &mut scratch);
    sph
}

/// [`cart_to_sph_3c1e`] writing into caller-owned buffers.
///
/// Routed through [`c2s_apply`], so identity axes (`l <= 1`) are skipped and a
/// transformed axis is unchanged bit for bit. On a def2-SVP work list that is
/// most axes, which is what Task 36-T0 measured the 3-index transform spending
/// its time on.
///
/// `sph` must be exactly `nsph(lk) * nsph(lj) * nsph(li)` long; `scratch` is
/// grown as needed and may start empty (Task 36-T0/36-T1).
pub fn cart_to_sph_3c1e_into<F: CintFloat>(
    cart: &[F],
    li: u8,
    lj: u8,
    lk: u8,
    sph: &mut [F],
    scratch: &mut Vec<F>,
) {
    let (nci, ncj, nck) = (ncart(li), ncart(lj), ncart(lk));
    let (nsi, nsj, nsk) = (nsph(li), nsph(lj), nsph(lk));

    debug_assert_eq!(cart.len(), nci * ncj * nck);
    debug_assert_eq!(sph.len(), nsk * nsj * nsi);

    // `[k][j][i]` with `i` fastest: i, then j, then k.
    c2s_apply(
        cart,
        sph,
        scratch,
        &[(li, nck * ncj, 1), (lj, nck, nsi), (lk, 1, nsj * nsi)],
    );
}

/// Apply cart-to-sph transform for a 3-center-2-electron shell triple (li, lj, lk).
///
/// Input `cart`: flat column-major array of shape `[ncart(lk) * ncart(lj) * ncart(li)]`.
/// Output: flat column-major array of shape `[nsph(lk) * nsph(lj) * nsph(li)]`.
///
/// Identical index structure to `cart_to_sph_3c1e` — same transform, different name
/// for the 3c2e family.
///
/// Generic over `F: CintFloat`. Delegates to `cart_to_sph_3c1e::<F>`.
pub fn cart_to_sph_3c2e<F: CintFloat>(cart: &[F], li: u8, lj: u8, lk: u8) -> Vec<F> {
    // 3c2e has the same 3-index (i, j, k) structure as 3c1e.
    cart_to_sph_3c1e::<F>(cart, li, lj, lk)
}

/// [`cart_to_sph_3c2e`] writing into caller-owned buffers.
///
/// Delegates to [`cart_to_sph_3c1e_into`] — same index structure, same
/// arithmetic (Task 36-T0/36-T1).
pub fn cart_to_sph_3c2e_into<F: CintFloat>(
    cart: &[F],
    li: u8,
    lj: u8,
    lk: u8,
    sph: &mut [F],
    scratch: &mut Vec<F>,
) {
    cart_to_sph_3c1e_into::<F>(cart, li, lj, lk, sph, scratch);
}

/// Apply cart-to-sph transform for a 2-electron shell quartet (li, lj, lk, ll).
///
/// Input `cart`: flat column-major array of shape `[ncart(ll) * ncart(lk) * ncart(lj) * ncart(li)]`
///   (l is outermost/slowest, i is innermost/fastest).
///
/// Output: flat column-major array of shape `[nsph(ll) * nsph(lk) * nsph(lj) * nsph(li)]`.
///
/// Transform order: i-axis first, then j, k, l (innermost to outermost).
///
/// Generic over `F: CintFloat`. The c2s coefficient table is FROZEN f64;
/// each coefficient is cast to `F` via `F::from_f64_lossy` at the accumulation site.
pub fn cart_to_sph_2e<F: CintFloat>(cart: &[F], li: u8, lj: u8, lk: u8, ll: u8) -> Vec<F> {
    let mut out = vec![F::zero(); nsph(li) * nsph(lj) * nsph(lk) * nsph(ll)];
    let mut scratch = Vec::new();
    cart_to_sph_2e_into(cart, li, lj, lk, ll, &mut out, &mut scratch);
    out
}

/// Transform one axis of a `[outer][ncart(l)][inner]` block into
/// `[outer][nsph(l)][inner]`.
///
/// Accumulation order is `c` ascending for each `(outer, m, inner)` triple —
/// the same order the four hand-written loops this replaces used, so the
/// floating-point result is unchanged.
fn c2s_axis<F: CintFloat>(src: &[F], dst: &mut [F], l: u8, outer: usize, inner: usize) {
    let nc = ncart(l);
    let ns = nsph(l);
    debug_assert_eq!(src.len(), outer * nc * inner);
    debug_assert_eq!(dst.len(), outer * ns * inner);
    for o in 0..outer {
        for m in 0..ns {
            let dst_base = (o * ns + m) * inner;
            for t in 0..inner {
                let mut sum = F::zero();
                for c in 0..nc {
                    sum =
                        sum + F::from_f64_lossy(c2s_coeff(l, m, c)) * src[(o * nc + c) * inner + t];
                }
                dst[dst_base + t] = sum;
            }
        }
    }
}

/// Run a c2s axis plan, skipping the axes where the transform is the identity.
///
/// `steps` is `(l, outer, inner)` per axis, innermost axis first; `outer` counts
/// the not-yet-transformed axes above it and `inner` the already-transformed
/// axes below. This is the shape every family's transform has — 1e, 2c2e, 3c1e
/// and 2e differ only in how many entries the plan has and what `outer`/`inner`
/// are — so they all route through here rather than each carrying its own
/// ping-pong.
///
/// Two things make this cheaper than the loop nest it replaces:
///
/// - **Identity axes (`l <= 1`) are dropped.** `C2S_L0` and `C2S_L1` are
///   identity matrices, so those axes are a copy dressed up as a matrix
///   product. On a def2-SVP work list most axes are s or p, and Task 36-T0
///   measured the c2s arithmetic at 68–81 % of the whole host transform — this
///   is where that share goes. Dropping an identity axis leaves the block's
///   extents untouched (`ncart == nsph` there), so the `outer`/`inner` counts
///   the caller computed stay correct.
/// - **Buffers are the caller's.** `scratch` is grown as needed and may start
///   empty; `out` must be exactly the plan's final length.
///
/// Accumulation order is `c` ascending for each `(outer, m, inner)` triple,
/// which is the order the hand-written loops used, so a non-identity axis is
/// unchanged bit for bit.
fn c2s_apply<F: CintFloat>(
    cart: &[F],
    out: &mut [F],
    scratch: &mut Vec<F>,
    steps: &[(u8, usize, usize)],
) {
    let mut plan = [(0u8, 0usize, 0usize); 4];
    let mut n_steps = 0;
    for &(l, outer, inner) in steps {
        if l > 1 {
            plan[n_steps] = (l, outer, inner);
            n_steps += 1;
        }
    }
    if n_steps == 0 {
        out.copy_from_slice(cart);
        return;
    }

    // Ping-pong between two halves of `scratch`; the last step writes `out`.
    // Each intermediate is no longer than the Cartesian block, because every
    // axis either shrinks (`nsph < ncart`) or is skipped.
    let cap = cart.len();
    if scratch.len() < 2 * cap {
        scratch.resize(2 * cap, F::zero());
    }
    let (buf_a, buf_b) = scratch.split_at_mut(cap);
    let mut src_in_a = false;
    let mut src_len = cart.len();

    for (step, &(l, outer, inner)) in plan[..n_steps].iter().enumerate() {
        let dst_len = outer * nsph(l) * inner;
        let last = step == n_steps - 1;
        match (step == 0, src_in_a, last) {
            (true, _, true) => c2s_axis(cart, &mut out[..dst_len], l, outer, inner),
            (true, _, false) => {
                c2s_axis(cart, &mut buf_a[..dst_len], l, outer, inner);
                src_in_a = true;
            }
            (false, true, true) => {
                c2s_axis(&buf_a[..src_len], &mut out[..dst_len], l, outer, inner);
            }
            (false, false, true) => {
                c2s_axis(&buf_b[..src_len], &mut out[..dst_len], l, outer, inner);
            }
            (false, true, false) => {
                c2s_axis(&buf_a[..src_len], &mut buf_b[..dst_len], l, outer, inner);
                src_in_a = false;
            }
            (false, false, false) => {
                c2s_axis(&buf_b[..src_len], &mut buf_a[..dst_len], l, outer, inner);
                src_in_a = true;
            }
        }
        src_len = dst_len;
    }
}

/// [`cart_to_sph_2e`] writing into caller-owned buffers.
///
/// Batched evaluation calls this once per contraction block of every quartet in
/// a work list, so the allocation-per-call shape of [`cart_to_sph_2e`] shows up
/// as a real fraction of wall-clock. Two things are different here:
///
/// - **`out` and `scratch` are caller-owned**, so a loop over quartets
///   allocates once instead of four times per block. `scratch` is grown as
///   needed and may start empty.
/// - **Axes with `l <= 1` are skipped.** `C2S_L0` and `C2S_L1` are identity
///   matrices, so those axes are a copy; skipping them removes the entire
///   transform for the s/p quartets that dominate a def2-SVP work list.
///
/// `out` must be exactly `nsph(li)*nsph(lj)*nsph(lk)*nsph(ll)` long.
pub fn cart_to_sph_2e_into<F: CintFloat>(
    cart: &[F],
    li: u8,
    lj: u8,
    lk: u8,
    ll: u8,
    out: &mut [F],
    scratch: &mut Vec<F>,
) {
    let (nci, ncj, nck, ncl) = (ncart(li), ncart(lj), ncart(lk), ncart(ll));
    let (nsi, nsj, nsk, nsl) = (nsph(li), nsph(lj), nsph(lk), nsph(ll));

    debug_assert_eq!(cart.len(), nci * ncj * nck * ncl);
    debug_assert_eq!(out.len(), nsi * nsj * nsk * nsl);

    // The block is `[l][k][j][i]` with `i` fastest. Viewed one axis at a time it
    // is `[outer][ncart(axis)][inner]`, where `outer` counts the not-yet-
    // transformed axes above it and `inner` the already-transformed axes below.
    // Axes are taken innermost-first (i, j, k, l), matching the original
    // four-step order.
    c2s_apply(
        cart,
        out,
        scratch,
        &[
            (li, ncl * nck * ncj, 1),
            (lj, ncl * nck, nsi),
            (lk, ncl, nsj * nsi),
            (ll, 1, nsk * nsj * nsi),
        ],
    );
}

/// Staging cart-to-sph transform — no-op.
///
/// Real kernels (1e, 2e, etc.) handle cart-to-sph internally using
/// `cart_to_sph_1e()` with per-shell angular momentum info.
/// The generic staging transform is bypassed; calling it is safe and idempotent.
pub fn cart_to_spheric_staging(staging: &mut [f64]) -> Result<(), cintxRsError> {
    let _ = staging;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The extraction gate.** The generated `l = 0..=15` table must agree,
    /// bit for bit, with the four hand-transcribed matrices this module has
    /// carried since the start.
    ///
    /// This is what makes `l >= 5` trustworthy. Those blocks have no hand
    /// reference to check against — they were parsed out of libcint's
    /// `g_trans_cart2sph[]` by `xtask gen-c2s-table` — so the evidence that the
    /// parse and the row-major layout are right has to come from the region
    /// where a reference does exist. 245 coefficients across `l = 0..=4`, from
    /// the same extraction, reproduce `C2S_L0..C2S_L4` exactly.
    #[test]
    fn generated_table_matches_the_hand_transcribed_matrices() {
        fn check(l: u8, rows: &[&[f64]]) {
            assert_eq!(rows.len(), nsph(l), "row count for l={l}");
            for (m, row) in rows.iter().enumerate() {
                assert_eq!(row.len(), ncart(l), "column count for l={l} m={m}");
                for (c, &want) in row.iter().enumerate() {
                    let got = c2s_coeff(l, m, c);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "l={l} m={m} cart={c}: generated table has {got}, \
                         the hand-transcribed matrix has {want}"
                    );
                }
            }
        }
        check(0, &C2S_L0.iter().map(|r| &r[..]).collect::<Vec<_>>());
        check(1, &C2S_L1.iter().map(|r| &r[..]).collect::<Vec<_>>());
        check(2, &C2S_L2.iter().map(|r| &r[..]).collect::<Vec<_>>());
        check(3, &C2S_L3.iter().map(|r| &r[..]).collect::<Vec<_>>());
        check(4, &C2S_L4.iter().map(|r| &r[..]).collect::<Vec<_>>());
    }

    /// The table's block offsets and length have to line up with the shapes the
    /// accessor indexes by, or `c2s_coeff` reads a neighbouring `l`'s block —
    /// which would be wrong without being obviously wrong.
    #[test]
    fn generated_table_blocks_have_the_right_shape() {
        use super::super::c2s_data::{C2S_OFFSET, C2S_TABLE};
        let mut cursor = 0usize;
        for l in 0..=C2S_LMAX {
            assert_eq!(
                C2S_OFFSET[l as usize], cursor,
                "block offset for l={l} does not follow the previous block"
            );
            cursor += nsph(l) * ncart(l);
        }
        assert_eq!(cursor, C2S_TABLE.len());
        assert_eq!(C2S_OFFSET[C2S_LMAX as usize + 1], C2S_TABLE.len());
    }

    /// `l >= 5` must actually produce coefficients now. The regression this
    /// guards is precise: before the table was extended, every one of these was
    /// `0.0`, and an `(h s | s)` integral — `nroots = 3`, well inside every
    /// device Rys ceiling — came back entirely zeroed with an `Ok` status.
    #[test]
    fn high_l_coefficients_are_not_zero() {
        for l in 5..=C2S_LMAX {
            let nonzero = (0..nsph(l))
                .flat_map(|m| (0..ncart(l)).map(move |c| (m, c)))
                .filter(|&(m, c)| c2s_coeff(l, m, c) != 0.0)
                .count();
            assert!(
                nonzero >= nsph(l),
                "l={l} has only {nonzero} non-zero coefficients; every spherical \
                 row must carry at least one"
            );
        }
    }

    /// The two ceilings are one number. `cintx-core` cannot depend on this
    /// crate, so `Shell::try_new`'s spherical guard carries its own constant;
    /// this pins it to the generated table's, which is the real authority.
    #[test]
    fn spheric_l_max_matches_the_table_ceiling() {
        assert_eq!(C2S_LMAX, cintx_core::SPHERIC_L_MAX);
    }

    /// A spherical shell past the ceiling is refused at construction, so the
    /// transform's `assert!` is a backstop rather than the user-facing failure.
    #[test]
    fn a_spherical_shell_past_the_ceiling_is_refused_at_construction() {
        use cintx_core::{Representation, Shell};
        use std::sync::Arc;

        let make = |l: u8, rep| {
            Shell::try_new(
                0,
                l,
                1,
                1,
                0,
                rep,
                Arc::from(vec![1.0_f64].into_boxed_slice()),
                Arc::from(vec![1.0_f64].into_boxed_slice()),
            )
        };
        assert!(make(C2S_LMAX, Representation::Spheric).is_ok());
        assert!(make(C2S_LMAX + 1, Representation::Spheric).is_err());
        // A Cartesian shell needs no transform, so the cap does not apply.
        assert!(make(C2S_LMAX + 1, Representation::Cart).is_ok());
    }

    /// The guard is fail-closed above the table, and the accessor panics rather
    /// than returning a zero that reads like a real answer.
    #[test]
    fn above_the_ceiling_is_refused_not_zeroed() {
        assert!(c2s_supports_l(C2S_LMAX));
        assert!(!c2s_supports_l(C2S_LMAX + 1));
        assert!(ensure_c2s_supported(C2S_LMAX).is_ok());
        assert!(ensure_c2s_supported(C2S_LMAX + 1).is_err());
        assert!(
            std::panic::catch_unwind(|| c2s_coeff(C2S_LMAX + 1, 0, 0)).is_err(),
            "c2s_coeff must not return a value for l > C2S_LMAX"
        );
    }

    #[test]
    fn ncart_values() {
        assert_eq!(ncart(0), 1);
        assert_eq!(ncart(1), 3);
        assert_eq!(ncart(2), 6);
        assert_eq!(ncart(3), 10);
        assert_eq!(ncart(4), 15);
    }

    #[test]
    fn nsph_values() {
        assert_eq!(nsph(0), 1);
        assert_eq!(nsph(1), 3);
        assert_eq!(nsph(2), 5);
        assert_eq!(nsph(3), 7);
        assert_eq!(nsph(4), 9);
    }

    #[test]
    fn c2s_l0_identity() {
        assert_eq!(C2S_L0, [[1.0]]);
    }

    #[test]
    fn c2s_l2_d_xy_coefficient() {
        // m=-2, col=1 (xy): dxy coefficient
        let diff = (C2S_L2[0][1] - 1.092548430592079070_f64).abs();
        assert!(diff < 1e-15, "C2S_L2[0][1] diff={diff}");
    }

    #[test]
    fn c2s_l2_dz2_coefficient() {
        // m=0, col=0 (xx): dz2 xx coefficient
        let diff = (C2S_L2[2][0] - (-0.315391565252520002_f64)).abs();
        assert!(diff < 1e-15, "C2S_L2[2][0] diff={diff}");
    }

    #[test]
    fn cart_to_sph_1e_ss_identity() {
        let cart = [1.0_f64];
        let mut sph = [0.0_f64];
        cart_to_sph_1e(&cart, &mut sph, 0, 0);
        assert_eq!(sph, [1.0]);
    }

    #[test]
    fn cart_to_spheric_staging_is_noop() {
        let mut data = vec![1.0, 2.0, 3.0];
        cart_to_spheric_staging(&mut data).unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  Multi-index c2s transform tests
    // ──────────────────────────────────────────────────────────────────────────

    /// 2c2e ss transform: 1x1 input → 1x1 output (identity for l=0).
    #[test]
    fn cart_to_sph_2c2e_ss_identity() {
        let cart = vec![1.0_f64];
        let sph = cart_to_sph_2c2e(&cart, 0, 0);
        assert_eq!(sph, vec![1.0]);
    }

    /// 2c2e output length check for pp (3x3 cart → 3x3 sph).
    #[test]
    fn cart_to_sph_2c2e_pp_length() {
        let cart = vec![1.0_f64; ncart(1) * ncart(1)];
        let sph = cart_to_sph_2c2e(&cart, 1, 1);
        assert_eq!(sph.len(), nsph(1) * nsph(1));
    }

    /// 2c2e output length check for dd (6x6 cart → 5x5 sph).
    #[test]
    fn cart_to_sph_2c2e_dd_length() {
        let cart = vec![0.0_f64; ncart(2) * ncart(2)];
        let sph = cart_to_sph_2c2e(&cart, 2, 2);
        assert_eq!(sph.len(), nsph(2) * nsph(2));
    }

    /// 3c1e sss transform: identity for l=0,0,0.
    #[test]
    fn cart_to_sph_3c1e_sss_identity() {
        let cart = vec![1.0_f64];
        let sph = cart_to_sph_3c1e(&cart, 0, 0, 0);
        assert_eq!(sph, vec![1.0]);
    }

    /// 3c1e output length check for ppp (3x3x3 cart → 3x3x3 sph).
    #[test]
    fn cart_to_sph_3c1e_ppp_length() {
        let cart = vec![0.0_f64; ncart(1) * ncart(1) * ncart(1)];
        let sph = cart_to_sph_3c1e(&cart, 1, 1, 1);
        assert_eq!(sph.len(), nsph(1) * nsph(1) * nsph(1));
    }

    /// 3c2e sss transform: identity for l=0,0,0.
    #[test]
    fn cart_to_sph_3c2e_sss_identity() {
        let cart = vec![1.0_f64];
        let sph = cart_to_sph_3c2e(&cart, 0, 0, 0);
        assert_eq!(sph, vec![1.0]);
    }

    /// 3c2e output length matches 3c1e (identical index structure).
    #[test]
    fn cart_to_sph_3c2e_ppp_length() {
        let cart = vec![0.0_f64; ncart(1) * ncart(1) * ncart(1)];
        let sph = cart_to_sph_3c2e(&cart, 1, 1, 1);
        assert_eq!(sph.len(), nsph(1) * nsph(1) * nsph(1));
    }

    /// 2e ssss transform: identity for l=0,0,0,0.
    #[test]
    fn cart_to_sph_2e_ssss_identity() {
        let cart = vec![1.0_f64];
        let sph = cart_to_sph_2e(&cart, 0, 0, 0, 0);
        assert_eq!(sph, vec![1.0]);
    }

    /// 2e output length check for pppp (3^4 cart → 3^4 sph, same since l=1).
    #[test]
    fn cart_to_sph_2e_pppp_length() {
        let cart = vec![0.0_f64; ncart(1) * ncart(1) * ncart(1) * ncart(1)];
        let sph = cart_to_sph_2e(&cart, 1, 1, 1, 1);
        assert_eq!(sph.len(), nsph(1) * nsph(1) * nsph(1) * nsph(1));
    }

    /// 2e output length check for dddd (6^4 cart → 5^4 sph).
    #[test]
    fn cart_to_sph_2e_dddd_length() {
        let cart = vec![0.0_f64; ncart(2) * ncart(2) * ncart(2) * ncart(2)];
        let sph = cart_to_sph_2e(&cart, 2, 2, 2, 2);
        assert_eq!(sph.len(), nsph(2) * nsph(2) * nsph(2) * nsph(2));
    }

    /// 3c1e and 3c2e produce identical output (same transform).
    #[test]
    fn cart_to_sph_3c1e_3c2e_same_output() {
        let li = 1_u8;
        let lj = 2_u8;
        let lk = 1_u8;
        let n = ncart(li) * ncart(lj) * ncart(lk);
        let cart: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1 + 1.0).collect();
        let sph_3c1e = cart_to_sph_3c1e(&cart, li, lj, lk);
        let sph_3c2e = cart_to_sph_3c2e(&cart, li, lj, lk);
        assert_eq!(
            sph_3c1e, sph_3c2e,
            "3c1e and 3c2e must produce identical output"
        );
    }
}

/// Tests for the generic c2s transform functions (Phase 20 Plan 03 — Wave 1 math leaves).
///
/// RED phase: tests call all transform fns with explicit `F` type parameters.
/// They will fail to compile until ALL transform functions are genericized over
/// `F: CintFloat`.
///
/// Pitfall 2 (CubeCL prelude shadow): coefficient table values used as test baselines
/// are captured as precomputed f64 literals, not computed at test runtime.
#[cfg(test)]
mod tests_c2s_generic {
    use super::*;

    /// f64 path: cart_to_sph_1e ss transform must produce [1.0] (identity for l=0).
    /// Baseline: precomputed literal 1.0.
    #[test]
    fn cart_to_sph_1e_f64_ss_identity() {
        let cart = [1.0_f64];
        let mut sph = [0.0_f64];
        cart_to_sph_1e::<f64>(&cart, &mut sph, 0, 0);
        assert_eq!(sph, [1.0_f64], "cart_to_sph_1e::<f64> ss identity failed");
    }

    /// f32 path: ss transform returns [1.0_f32].
    #[test]
    fn cart_to_sph_1e_f32_ss_identity() {
        let cart = [1.0_f32];
        let mut sph = [0.0_f32];
        cart_to_sph_1e::<f32>(&cart, &mut sph, 0, 0);
        assert_eq!(sph, [1.0_f32], "cart_to_sph_1e::<f32> ss identity failed");
    }

    /// f64 path: cart_to_sph_2c2e ss returns [1.0].
    #[test]
    fn cart_to_sph_2c2e_f64_ss_identity() {
        let cart = vec![1.0_f64];
        let sph = cart_to_sph_2c2e::<f64>(&cart, 0, 0);
        assert_eq!(
            sph,
            vec![1.0_f64],
            "cart_to_sph_2c2e::<f64> ss identity failed"
        );
    }

    /// f32 path: cart_to_sph_2c2e ss returns [1.0_f32].
    #[test]
    fn cart_to_sph_2c2e_f32_ss_identity() {
        let cart = vec![1.0_f32];
        let sph = cart_to_sph_2c2e::<f32>(&cart, 0, 0);
        assert_eq!(
            sph,
            vec![1.0_f32],
            "cart_to_sph_2c2e::<f32> ss identity failed"
        );
    }

    /// f64 path: d-shell 2c2e output length is nsph(2)^2 = 25.
    #[test]
    fn cart_to_sph_2c2e_f64_dd_length() {
        let cart = vec![0.0_f64; ncart(2) * ncart(2)];
        let sph = cart_to_sph_2c2e::<f64>(&cart, 2, 2);
        assert_eq!(
            sph.len(),
            nsph(2) * nsph(2),
            "cart_to_sph_2c2e::<f64> dd length wrong"
        );
    }

    /// f32 path: pp 2c2e returns finite values.
    #[test]
    fn cart_to_sph_2c2e_f32_pp_finite() {
        let cart: Vec<f32> = (0..(ncart(1) * ncart(1)))
            .map(|i| (i + 1) as f32 * 0.1)
            .collect();
        let sph = cart_to_sph_2c2e::<f32>(&cart, 1, 1);
        for &v in &sph {
            assert!(
                v.is_finite(),
                "cart_to_sph_2c2e::<f32> produced non-finite value: {v}"
            );
        }
    }

    /// f64 path: cart_to_sph_3c1e sss identity.
    #[test]
    fn cart_to_sph_3c1e_f64_sss_identity() {
        let cart = vec![1.0_f64];
        let sph = cart_to_sph_3c1e::<f64>(&cart, 0, 0, 0);
        assert_eq!(
            sph,
            vec![1.0_f64],
            "cart_to_sph_3c1e::<f64> sss identity failed"
        );
    }

    /// f32 path: cart_to_sph_3c1e sss identity.
    #[test]
    fn cart_to_sph_3c1e_f32_sss_identity() {
        let cart = vec![1.0_f32];
        let sph = cart_to_sph_3c1e::<f32>(&cart, 0, 0, 0);
        assert_eq!(
            sph,
            vec![1.0_f32],
            "cart_to_sph_3c1e::<f32> sss identity failed"
        );
    }

    /// f64 path: cart_to_sph_3c2e sss identity.
    #[test]
    fn cart_to_sph_3c2e_f64_sss_identity() {
        let cart = vec![1.0_f64];
        let sph = cart_to_sph_3c2e::<f64>(&cart, 0, 0, 0);
        assert_eq!(
            sph,
            vec![1.0_f64],
            "cart_to_sph_3c2e::<f64> sss identity failed"
        );
    }

    /// f32 path: cart_to_sph_3c2e sss identity.
    #[test]
    fn cart_to_sph_3c2e_f32_sss_identity() {
        let cart = vec![1.0_f32];
        let sph = cart_to_sph_3c2e::<f32>(&cart, 0, 0, 0);
        assert_eq!(
            sph,
            vec![1.0_f32],
            "cart_to_sph_3c2e::<f32> sss identity failed"
        );
    }

    /// f64 path: cart_to_sph_2e ssss identity.
    #[test]
    fn cart_to_sph_2e_f64_ssss_identity() {
        let cart = vec![1.0_f64];
        let sph = cart_to_sph_2e::<f64>(&cart, 0, 0, 0, 0);
        assert_eq!(
            sph,
            vec![1.0_f64],
            "cart_to_sph_2e::<f64> ssss identity failed"
        );
    }

    /// f32 path: cart_to_sph_2e ssss identity.
    #[test]
    fn cart_to_sph_2e_f32_ssss_identity() {
        let cart = vec![1.0_f32];
        let sph = cart_to_sph_2e::<f32>(&cart, 0, 0, 0, 0);
        assert_eq!(
            sph,
            vec![1.0_f32],
            "cart_to_sph_2e::<f32> ssss identity failed"
        );
    }

    /// f32 path: d-shell 2e returns finite values.
    #[test]
    fn cart_to_sph_2e_f32_dddd_finite() {
        let cart: Vec<f32> = (0..(ncart(2) * ncart(2) * ncart(2) * ncart(2)))
            .map(|i| (i + 1) as f32 * 0.01)
            .collect();
        let sph = cart_to_sph_2e::<f32>(&cart, 2, 2, 2, 2);
        for &v in &sph {
            assert!(
                v.is_finite(),
                "cart_to_sph_2e::<f32> produced non-finite value: {v}"
            );
        }
    }
}
