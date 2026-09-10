//! Phase 25 HESS-04 — host f64 evaluators for the 3rd/4th-order 1e derivative
//! families (`deriv3.c` rank 27, `deriv4.c` rank 81).
//!
//! These are the `int1e_ipipip*` / `int1e_ipipipip*` families: the first-order
//! bra/ket nabla engine (`G2E_D_I` / `G2E_D_J`) composed three or four times on
//! the nuclear-attraction / `rinv` Rys G-tensor. Each family is a faithful port
//! of its own `CINTgout1e_int1e_*` block in `libcint-master/src/autocode/`:
//!
//!   - the per-axis nuclear Rys G-tensor `g0` is built exactly like
//!     [`super::one_electron::contract_nuclear`] (VRR per Rys root + HRR), with
//!     extra bra/ket headroom so the nabla recurrences stay in range;
//!   - the nabla buffers `g1..g7` (deriv3) / `g1..g15` (deriv4) are produced by
//!     applying `D_I` (bra) / `D_J` (ket) per the family's verbatim op sequence;
//!   - the `s[rank]` triple-product table and the `gout[rank]` permutation are
//!     copied verbatim per family (each differs — D-09).
//!
//! Routed through the HOST path (FND-02): the bra/ket +2/+3 headroom can elevate
//! the nuclear Rys `nroots` to >=6, which the device comptime kernel
//! (`MAX_DEVICE_NROOTS=5`) cannot serve; `rys_roots_host` handles 6..12.

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

use crate::math::obara_saika::{hrr_step_host, vrr_2e_step_host};
use crate::math::rys::rys_roots_host;
use cintx_core::Atom;

/// Cartesian component count for angular momentum `l`.
#[inline]
fn ncart(l: u8) -> usize {
    ((l as usize + 1) * (l as usize + 2)) / 2
}

/// Cartesian (ix,iy,iz) power tuples for angular momentum `l`, libcint order.
fn cart_comps(l: u8) -> Vec<(u32, u32, u32)> {
    let mut v = Vec::with_capacity(ncart(l));
    // libcint CINTcart_comp: for i in 0..=l { for j in 0..=i { ... } } with
    // x = l-i, y = i-j, z = j (matches cart_comps used elsewhere in the crate).
    let ll = l as i32;
    for i in (0..=ll).rev() {
        for j in (0..=(ll - i)).rev() {
            let k = ll - i - j;
            v.push((i as u32, j as u32, k as u32));
        }
    }
    v
}

/// A single nabla step in a family's op sequence: apply `D_I` (bra) or `D_J`
/// (ket) to the source buffer `src`, writing into a destination buffer.
///
/// `i_off` / `j_off` are the target headroom offsets `envs->i_l+i_off` /
/// `envs->j_l+j_off` copied verbatim from the family's `G2E_D_*` call — the
/// recurrence fills `i in 0..=li+i_off`, `j in 0..=lj+j_off` and reads the
/// source one level above (D_I reads `i+1`, D_J reads `j+1`), so the source
/// buffer must carry one extra level on the derivative axis.
#[derive(Clone, Copy)]
enum Op {
    /// `G2E_D_I(dst, src, i_l+i_off, j_l+j_off, ...)` — bra-center derivative.
    DI {
        dst: usize,
        src: usize,
        i_off: u32,
        j_off: u32,
    },
    /// `G2E_D_J(dst, src, i_l+i_off, j_l+j_off, ...)` — ket-center derivative.
    DJ {
        dst: usize,
        src: usize,
        i_off: u32,
        j_off: u32,
    },
}

/// One 3rd/4th-order derivative family specification, copied verbatim from its
/// `deriv3.c` / `deriv4.c` block.
struct FamilySpec {
    /// Number of output components (27 for deriv3, 81 for deriv4).
    rank: usize,
    /// Number of g-buffers (8 for deriv3 g0..g7, 16 for deriv4 g0..g15).
    nbuf: usize,
    /// The op sequence building `g1..` from `g0` (in order; `g0` is the base).
    ops: &'static [Op],
    /// `s[rank]` triple-product table: each entry `(sx, sy, sz)` is the g-buffer
    /// index read on the x / y / z axis (`s[k] += g[sx][ix] * g[sy][iy] * g[sz][iz]`).
    s_table: &'static [(usize, usize, usize)],
    /// `gout[rank]` permutation: `gout[c] = s[perm[c]]`.
    gout_perm: &'static [usize],
    /// Optional DOT-P contraction: each output is the sum of three `s` entries.
    dot_terms: Option<&'static [[usize; 3]]>,
    /// General signed linear gout map used by σ derivative families.
    linear_terms: Option<&'static [LinearTerm]>,
}

#[derive(Clone, Copy)]
struct LinearTerm {
    out: usize,
    s: usize,
    coeff: f64,
}

// ── deriv3 (rank 27) op sequences (i_off/j_off verbatim from deriv3.c) ────────
// ipipipnuc / ipipiprinv: all D_I (bra ∇∇∇).
const OPS_IPIPIP: [Op; 7] = [
    Op::DI {
        dst: 1,
        src: 0,
        i_off: 2,
        j_off: 0,
    },
    Op::DI {
        dst: 2,
        src: 0,
        i_off: 1,
        j_off: 0,
    },
    Op::DI {
        dst: 3,
        src: 1,
        i_off: 1,
        j_off: 0,
    },
    Op::DI {
        dst: 4,
        src: 0,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 5,
        src: 1,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 6,
        src: 2,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 7,
        src: 3,
        i_off: 0,
        j_off: 0,
    },
];
// ipipnucip / ipiprinvip: D_J first (ket), then D_I (bra ∇∇).
const OPS_IPIPNUCIP: [Op; 7] = [
    Op::DJ {
        dst: 1,
        src: 0,
        i_off: 2,
        j_off: 0,
    },
    Op::DI {
        dst: 2,
        src: 0,
        i_off: 1,
        j_off: 0,
    },
    Op::DI {
        dst: 3,
        src: 1,
        i_off: 1,
        j_off: 0,
    },
    Op::DI {
        dst: 4,
        src: 0,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 5,
        src: 1,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 6,
        src: 2,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 7,
        src: 3,
        i_off: 0,
        j_off: 0,
    },
];

// deriv3 shared s27 table (verbatim from deriv3.c — identical across all 4).
const S27: [(usize, usize, usize); 27] = [
    (7, 0, 0),
    (6, 1, 0),
    (6, 0, 1),
    (5, 2, 0),
    (4, 3, 0),
    (4, 2, 1),
    (5, 0, 2),
    (4, 1, 2),
    (4, 0, 3),
    (3, 4, 0),
    (2, 5, 0),
    (2, 4, 1),
    (1, 6, 0),
    (0, 7, 0),
    (0, 6, 1),
    (1, 4, 2),
    (0, 5, 2),
    (0, 4, 3),
    (3, 0, 4),
    (2, 1, 4),
    (2, 0, 5),
    (1, 2, 4),
    (0, 3, 4),
    (0, 2, 5),
    (1, 0, 6),
    (0, 1, 6),
    (0, 0, 7),
];
// ipipipnuc / ipipiprinv gout permutation.
const PERM_IPIPIP: [usize; 27] = [
    0, 9, 18, 3, 12, 21, 6, 15, 24, 1, 10, 19, 4, 13, 22, 7, 16, 25, 2, 11, 20, 5, 14, 23, 8, 17,
    26,
];
// ipipnucip / ipiprinvip gout permutation.
const PERM_IPIPNUCIP: [usize; 27] = [
    0, 1, 2, 9, 10, 11, 18, 19, 20, 3, 4, 5, 12, 13, 14, 21, 22, 23, 6, 7, 8, 15, 16, 17, 24, 25,
    26,
];

const DOT_IPPNUCP: [[usize; 3]; 3] = [[0, 4, 8], [9, 13, 17], [18, 22, 26]];

// ── W5-06: the X2C BASE families `int1e_pnucp` / `int1e_prinvp` ──────────────
//
// Wave 3 shipped the DERIVATIVES (`ippnucp`, `ippnucpip`, `ipippnucp`, and the
// rinv twins) but not the base families they differentiate, so `pyscf/x2c/x2c.py`
// — which calls `int1e_pnucp` directly to build the X2C Hamiltonian — was still
// unsatisfiable while `sfx2c1e_grad.py` was not.
//
// `ng[] = {1, 1, 0, 0, 2, 1, 0, 1}` (intor1.c:990), rank 1, `CINT1e_drv(..., 2)`
// for pnucp (atom-summed nuclear) and `(..., 1)` for prinvp (single rinv center).
// Both share ONE gout (`CINTgout1e_int1e_pnucp` / `_int1e_prinvp` are identical
// term for term); only the Coulomb-center list differs, exactly as the Wave-3
// `ippnucp`/`ipprinvp` pair already does.
//
// Cascade, verbatim from intor1.c:
//   G2E_D_J(g1, g0, i_l+1, j_l+0)   ket ∇, bra headroom +1
//   G2E_D_I(g2, g0, i_l+0, j_l  )   bra ∇
//   G2E_D_I(g3, g1, i_l+0, j_l  )   bra ∇ of the ket-∇ block
const OPS_PNUCP: [Op; 3] = [
    Op::DJ {
        dst: 1,
        src: 0,
        i_off: 1,
        j_off: 0,
    },
    Op::DI {
        dst: 2,
        src: 0,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 3,
        src: 1,
        i_off: 0,
        j_off: 0,
    },
];

// The 2-leg (D_I, D_J) 3^2 = 9-term table, verbatim from `CINTgout1e_int1e_pnucp`.
// Buffer roles: g0 = (), g1 = (J), g2 = (I), g3 = (I, J).
const S9_PNUCP: [(usize, usize, usize); 9] = [
    (3, 0, 0), // s[0] = g3 g0 g0   I and J both on x
    (2, 1, 0), // s[1] = g2 g1 g0   I on x, J on y
    (2, 0, 1), // s[2] = g2 g0 g1   I on x, J on z
    (1, 2, 0), // s[3] = g1 g2 g0   I on y, J on x
    (0, 3, 0), // s[4] = g0 g3 g0   I and J both on y
    (0, 2, 1), // s[5] = g0 g2 g1   I on y, J on z
    (1, 0, 2), // s[6] = g1 g0 g2   I on z, J on x
    (0, 1, 2), // s[7] = g0 g1 g2   I on z, J on y
    (0, 0, 3), // s[8] = g0 g0 g3   I and J both on z
];

// `gout[n] = s[0] + s[4] + s[8]` (intor1.c) — the ∇i · ∇j trace, i.e. the
// diagonal of the 2-leg table. Rank 1.
const DOT_PNUCP: [[usize; 3]; 1] = [[0, 4, 8]];

// ── deriv4 (rank 81) op sequences (i_off/j_off verbatim from deriv4.c) ────────
// ipipipiprinv: all D_I (bra ∇∇∇∇), first op bra+3.
const OPS_IPIPIPIPRINV: [Op; 15] = [
    Op::DI {
        dst: 1,
        src: 0,
        i_off: 3,
        j_off: 0,
    },
    Op::DI {
        dst: 2,
        src: 0,
        i_off: 2,
        j_off: 0,
    },
    Op::DI {
        dst: 3,
        src: 1,
        i_off: 2,
        j_off: 0,
    },
    Op::DI {
        dst: 4,
        src: 0,
        i_off: 1,
        j_off: 0,
    },
    Op::DI {
        dst: 5,
        src: 1,
        i_off: 1,
        j_off: 0,
    },
    Op::DI {
        dst: 6,
        src: 2,
        i_off: 1,
        j_off: 0,
    },
    Op::DI {
        dst: 7,
        src: 3,
        i_off: 1,
        j_off: 0,
    },
    Op::DI {
        dst: 8,
        src: 0,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 9,
        src: 1,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 10,
        src: 2,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 11,
        src: 3,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 12,
        src: 4,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 13,
        src: 5,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 14,
        src: 6,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 15,
        src: 7,
        i_off: 0,
        j_off: 0,
    },
];
// ipiprinvipip: D_J, D_J(ket+1), D_J(on g2), then D_I — ket ∇∇ + bra ∇∇.
const OPS_IPIPRINVIPIP: [Op; 15] = [
    Op::DJ {
        dst: 1,
        src: 0,
        i_off: 2,
        j_off: 0,
    },
    Op::DJ {
        dst: 2,
        src: 0,
        i_off: 2,
        j_off: 1,
    },
    Op::DJ {
        dst: 3,
        src: 2,
        i_off: 2,
        j_off: 0,
    },
    Op::DI {
        dst: 4,
        src: 0,
        i_off: 1,
        j_off: 0,
    },
    Op::DI {
        dst: 5,
        src: 1,
        i_off: 1,
        j_off: 0,
    },
    Op::DI {
        dst: 6,
        src: 2,
        i_off: 1,
        j_off: 0,
    },
    Op::DI {
        dst: 7,
        src: 3,
        i_off: 1,
        j_off: 0,
    },
    Op::DI {
        dst: 8,
        src: 0,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 9,
        src: 1,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 10,
        src: 2,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 11,
        src: 3,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 12,
        src: 4,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 13,
        src: 5,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 14,
        src: 6,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 15,
        src: 7,
        i_off: 0,
        j_off: 0,
    },
];
// ipipiprinvip: D_J (bra+3), then D_I — bra ∇∇∇ + ket ∇.
const OPS_IPIPIPRINVIP: [Op; 15] = [
    Op::DJ {
        dst: 1,
        src: 0,
        i_off: 3,
        j_off: 0,
    },
    Op::DI {
        dst: 2,
        src: 0,
        i_off: 2,
        j_off: 0,
    },
    Op::DI {
        dst: 3,
        src: 1,
        i_off: 2,
        j_off: 0,
    },
    Op::DI {
        dst: 4,
        src: 0,
        i_off: 1,
        j_off: 0,
    },
    Op::DI {
        dst: 5,
        src: 1,
        i_off: 1,
        j_off: 0,
    },
    Op::DI {
        dst: 6,
        src: 2,
        i_off: 1,
        j_off: 0,
    },
    Op::DI {
        dst: 7,
        src: 3,
        i_off: 1,
        j_off: 0,
    },
    Op::DI {
        dst: 8,
        src: 0,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 9,
        src: 1,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 10,
        src: 2,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 11,
        src: 3,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 12,
        src: 4,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 13,
        src: 5,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 14,
        src: 6,
        i_off: 0,
        j_off: 0,
    },
    Op::DI {
        dst: 15,
        src: 7,
        i_off: 0,
        j_off: 0,
    },
];
// deriv4 shared s81 table (verbatim — identical across all 3).
const S81: [(usize, usize, usize); 81] = [
    (15, 0, 0),
    (14, 1, 0),
    (14, 0, 1),
    (13, 2, 0),
    (12, 3, 0),
    (12, 2, 1),
    (13, 0, 2),
    (12, 1, 2),
    (12, 0, 3),
    (11, 4, 0),
    (10, 5, 0),
    (10, 4, 1),
    (9, 6, 0),
    (8, 7, 0),
    (8, 6, 1),
    (9, 4, 2),
    (8, 5, 2),
    (8, 4, 3),
    (11, 0, 4),
    (10, 1, 4),
    (10, 0, 5),
    (9, 2, 4),
    (8, 3, 4),
    (8, 2, 5),
    (9, 0, 6),
    (8, 1, 6),
    (8, 0, 7),
    (7, 8, 0),
    (6, 9, 0),
    (6, 8, 1),
    (5, 10, 0),
    (4, 11, 0),
    (4, 10, 1),
    (5, 8, 2),
    (4, 9, 2),
    (4, 8, 3),
    (3, 12, 0),
    (2, 13, 0),
    (2, 12, 1),
    (1, 14, 0),
    (0, 15, 0),
    (0, 14, 1),
    (1, 12, 2),
    (0, 13, 2),
    (0, 12, 3),
    (3, 8, 4),
    (2, 9, 4),
    (2, 8, 5),
    (1, 10, 4),
    (0, 11, 4),
    (0, 10, 5),
    (1, 8, 6),
    (0, 9, 6),
    (0, 8, 7),
    (7, 0, 8),
    (6, 1, 8),
    (6, 0, 9),
    (5, 2, 8),
    (4, 3, 8),
    (4, 2, 9),
    (5, 0, 10),
    (4, 1, 10),
    (4, 0, 11),
    (3, 4, 8),
    (2, 5, 8),
    (2, 4, 9),
    (1, 6, 8),
    (0, 7, 8),
    (0, 6, 9),
    (1, 4, 10),
    (0, 5, 10),
    (0, 4, 11),
    (3, 0, 12),
    (2, 1, 12),
    (2, 0, 13),
    (1, 2, 12),
    (0, 3, 12),
    (0, 2, 13),
    (1, 0, 14),
    (0, 1, 14),
    (0, 0, 15),
];
const PERM_IPIPIPIPRINV: [usize; 81] = [
    0, 27, 54, 9, 36, 63, 18, 45, 72, 3, 30, 57, 12, 39, 66, 21, 48, 75, 6, 33, 60, 15, 42, 69, 24,
    51, 78, 1, 28, 55, 10, 37, 64, 19, 46, 73, 4, 31, 58, 13, 40, 67, 22, 49, 76, 7, 34, 61, 16,
    43, 70, 25, 52, 79, 2, 29, 56, 11, 38, 65, 20, 47, 74, 5, 32, 59, 14, 41, 68, 23, 50, 77, 8,
    35, 62, 17, 44, 71, 26, 53, 80,
];
const DOT_IPIPPNUCP: [[usize; 3]; 9] = [
    [0, 4, 8],
    [27, 31, 35],
    [54, 58, 62],
    [9, 13, 17],
    [36, 40, 44],
    [63, 67, 71],
    [18, 22, 26],
    [45, 49, 53],
    [72, 76, 80],
];
const DOT_IPPNUCPIP: [[usize; 3]; 9] = [
    [0, 12, 24],
    [27, 39, 51],
    [54, 66, 78],
    [1, 13, 25],
    [28, 40, 52],
    [55, 67, 79],
    [2, 14, 26],
    [29, 41, 53],
    [56, 68, 80],
];
const PERM_IPIPRINVIPIP: [usize; 81] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 27, 28, 29, 30, 31, 32, 33, 34, 35, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 9, 10, 11, 12, 13, 14, 15, 16, 17, 36, 37, 38, 39, 40, 41, 42, 43, 44, 63, 64, 65, 66, 67,
    68, 69, 70, 71, 18, 19, 20, 21, 22, 23, 24, 25, 26, 45, 46, 47, 48, 49, 50, 51, 52, 53, 72, 73,
    74, 75, 76, 77, 78, 79, 80,
];
const PERM_IPIPIPRINVIP: [usize; 81] = [
    0, 1, 2, 27, 28, 29, 54, 55, 56, 9, 10, 11, 36, 37, 38, 63, 64, 65, 18, 19, 20, 45, 46, 47, 72,
    73, 74, 3, 4, 5, 30, 31, 32, 57, 58, 59, 12, 13, 14, 39, 40, 41, 66, 67, 68, 21, 22, 23, 48,
    49, 50, 75, 76, 77, 6, 7, 8, 33, 34, 35, 60, 61, 62, 15, 16, 17, 42, 43, 44, 69, 70, 71, 24,
    25, 26, 51, 52, 53, 78, 79, 80,
];

const LINEAR_IPSP: &[LinearTerm] = &[
    LinearTerm {
        out: 0,
        s: 11,
        coeff: 1.0,
    },
    LinearTerm {
        out: 0,
        s: 19,
        coeff: -1.0,
    },
    LinearTerm {
        out: 1,
        s: 18,
        coeff: 1.0,
    },
    LinearTerm {
        out: 1,
        s: 2,
        coeff: -1.0,
    },
    LinearTerm {
        out: 2,
        s: 1,
        coeff: 1.0,
    },
    LinearTerm {
        out: 2,
        s: 9,
        coeff: -1.0,
    },
    LinearTerm {
        out: 3,
        s: 0,
        coeff: 1.0,
    },
    LinearTerm {
        out: 3,
        s: 10,
        coeff: 1.0,
    },
    LinearTerm {
        out: 3,
        s: 20,
        coeff: 1.0,
    },
    LinearTerm {
        out: 4,
        s: 14,
        coeff: 1.0,
    },
    LinearTerm {
        out: 4,
        s: 22,
        coeff: -1.0,
    },
    LinearTerm {
        out: 5,
        s: 21,
        coeff: 1.0,
    },
    LinearTerm {
        out: 5,
        s: 5,
        coeff: -1.0,
    },
    LinearTerm {
        out: 6,
        s: 4,
        coeff: 1.0,
    },
    LinearTerm {
        out: 6,
        s: 12,
        coeff: -1.0,
    },
    LinearTerm {
        out: 7,
        s: 3,
        coeff: 1.0,
    },
    LinearTerm {
        out: 7,
        s: 13,
        coeff: 1.0,
    },
    LinearTerm {
        out: 7,
        s: 23,
        coeff: 1.0,
    },
    LinearTerm {
        out: 8,
        s: 17,
        coeff: 1.0,
    },
    LinearTerm {
        out: 8,
        s: 25,
        coeff: -1.0,
    },
    LinearTerm {
        out: 9,
        s: 24,
        coeff: 1.0,
    },
    LinearTerm {
        out: 9,
        s: 8,
        coeff: -1.0,
    },
    LinearTerm {
        out: 10,
        s: 7,
        coeff: 1.0,
    },
    LinearTerm {
        out: 10,
        s: 15,
        coeff: -1.0,
    },
    LinearTerm {
        out: 11,
        s: 6,
        coeff: 1.0,
    },
    LinearTerm {
        out: 11,
        s: 16,
        coeff: 1.0,
    },
    LinearTerm {
        out: 11,
        s: 26,
        coeff: 1.0,
    },
];
const LINEAR_IPIPSP: &[LinearTerm] = &[
    LinearTerm {
        out: 0,
        s: 29,
        coeff: 1.0,
    },
    LinearTerm {
        out: 0,
        s: 55,
        coeff: -1.0,
    },
    LinearTerm {
        out: 1,
        s: 54,
        coeff: 1.0,
    },
    LinearTerm {
        out: 1,
        s: 2,
        coeff: -1.0,
    },
    LinearTerm {
        out: 2,
        s: 1,
        coeff: 1.0,
    },
    LinearTerm {
        out: 2,
        s: 27,
        coeff: -1.0,
    },
    LinearTerm {
        out: 3,
        s: 0,
        coeff: 1.0,
    },
    LinearTerm {
        out: 3,
        s: 28,
        coeff: 1.0,
    },
    LinearTerm {
        out: 3,
        s: 56,
        coeff: 1.0,
    },
    LinearTerm {
        out: 4,
        s: 32,
        coeff: 1.0,
    },
    LinearTerm {
        out: 4,
        s: 58,
        coeff: -1.0,
    },
    LinearTerm {
        out: 5,
        s: 57,
        coeff: 1.0,
    },
    LinearTerm {
        out: 5,
        s: 5,
        coeff: -1.0,
    },
    LinearTerm {
        out: 6,
        s: 4,
        coeff: 1.0,
    },
    LinearTerm {
        out: 6,
        s: 30,
        coeff: -1.0,
    },
    LinearTerm {
        out: 7,
        s: 3,
        coeff: 1.0,
    },
    LinearTerm {
        out: 7,
        s: 31,
        coeff: 1.0,
    },
    LinearTerm {
        out: 7,
        s: 59,
        coeff: 1.0,
    },
    LinearTerm {
        out: 8,
        s: 35,
        coeff: 1.0,
    },
    LinearTerm {
        out: 8,
        s: 61,
        coeff: -1.0,
    },
    LinearTerm {
        out: 9,
        s: 60,
        coeff: 1.0,
    },
    LinearTerm {
        out: 9,
        s: 8,
        coeff: -1.0,
    },
    LinearTerm {
        out: 10,
        s: 7,
        coeff: 1.0,
    },
    LinearTerm {
        out: 10,
        s: 33,
        coeff: -1.0,
    },
    LinearTerm {
        out: 11,
        s: 6,
        coeff: 1.0,
    },
    LinearTerm {
        out: 11,
        s: 34,
        coeff: 1.0,
    },
    LinearTerm {
        out: 11,
        s: 62,
        coeff: 1.0,
    },
    LinearTerm {
        out: 12,
        s: 38,
        coeff: 1.0,
    },
    LinearTerm {
        out: 12,
        s: 64,
        coeff: -1.0,
    },
    LinearTerm {
        out: 13,
        s: 63,
        coeff: 1.0,
    },
    LinearTerm {
        out: 13,
        s: 11,
        coeff: -1.0,
    },
    LinearTerm {
        out: 14,
        s: 10,
        coeff: 1.0,
    },
    LinearTerm {
        out: 14,
        s: 36,
        coeff: -1.0,
    },
    LinearTerm {
        out: 15,
        s: 9,
        coeff: 1.0,
    },
    LinearTerm {
        out: 15,
        s: 37,
        coeff: 1.0,
    },
    LinearTerm {
        out: 15,
        s: 65,
        coeff: 1.0,
    },
    LinearTerm {
        out: 16,
        s: 41,
        coeff: 1.0,
    },
    LinearTerm {
        out: 16,
        s: 67,
        coeff: -1.0,
    },
    LinearTerm {
        out: 17,
        s: 66,
        coeff: 1.0,
    },
    LinearTerm {
        out: 17,
        s: 14,
        coeff: -1.0,
    },
    LinearTerm {
        out: 18,
        s: 13,
        coeff: 1.0,
    },
    LinearTerm {
        out: 18,
        s: 39,
        coeff: -1.0,
    },
    LinearTerm {
        out: 19,
        s: 12,
        coeff: 1.0,
    },
    LinearTerm {
        out: 19,
        s: 40,
        coeff: 1.0,
    },
    LinearTerm {
        out: 19,
        s: 68,
        coeff: 1.0,
    },
    LinearTerm {
        out: 20,
        s: 44,
        coeff: 1.0,
    },
    LinearTerm {
        out: 20,
        s: 70,
        coeff: -1.0,
    },
    LinearTerm {
        out: 21,
        s: 69,
        coeff: 1.0,
    },
    LinearTerm {
        out: 21,
        s: 17,
        coeff: -1.0,
    },
    LinearTerm {
        out: 22,
        s: 16,
        coeff: 1.0,
    },
    LinearTerm {
        out: 22,
        s: 42,
        coeff: -1.0,
    },
    LinearTerm {
        out: 23,
        s: 15,
        coeff: 1.0,
    },
    LinearTerm {
        out: 23,
        s: 43,
        coeff: 1.0,
    },
    LinearTerm {
        out: 23,
        s: 71,
        coeff: 1.0,
    },
    LinearTerm {
        out: 24,
        s: 47,
        coeff: 1.0,
    },
    LinearTerm {
        out: 24,
        s: 73,
        coeff: -1.0,
    },
    LinearTerm {
        out: 25,
        s: 72,
        coeff: 1.0,
    },
    LinearTerm {
        out: 25,
        s: 20,
        coeff: -1.0,
    },
    LinearTerm {
        out: 26,
        s: 19,
        coeff: 1.0,
    },
    LinearTerm {
        out: 26,
        s: 45,
        coeff: -1.0,
    },
    LinearTerm {
        out: 27,
        s: 18,
        coeff: 1.0,
    },
    LinearTerm {
        out: 27,
        s: 46,
        coeff: 1.0,
    },
    LinearTerm {
        out: 27,
        s: 74,
        coeff: 1.0,
    },
    LinearTerm {
        out: 28,
        s: 50,
        coeff: 1.0,
    },
    LinearTerm {
        out: 28,
        s: 76,
        coeff: -1.0,
    },
    LinearTerm {
        out: 29,
        s: 75,
        coeff: 1.0,
    },
    LinearTerm {
        out: 29,
        s: 23,
        coeff: -1.0,
    },
    LinearTerm {
        out: 30,
        s: 22,
        coeff: 1.0,
    },
    LinearTerm {
        out: 30,
        s: 48,
        coeff: -1.0,
    },
    LinearTerm {
        out: 31,
        s: 21,
        coeff: 1.0,
    },
    LinearTerm {
        out: 31,
        s: 49,
        coeff: 1.0,
    },
    LinearTerm {
        out: 31,
        s: 77,
        coeff: 1.0,
    },
    LinearTerm {
        out: 32,
        s: 53,
        coeff: 1.0,
    },
    LinearTerm {
        out: 32,
        s: 79,
        coeff: -1.0,
    },
    LinearTerm {
        out: 33,
        s: 78,
        coeff: 1.0,
    },
    LinearTerm {
        out: 33,
        s: 26,
        coeff: -1.0,
    },
    LinearTerm {
        out: 34,
        s: 25,
        coeff: 1.0,
    },
    LinearTerm {
        out: 34,
        s: 51,
        coeff: -1.0,
    },
    LinearTerm {
        out: 35,
        s: 24,
        coeff: 1.0,
    },
    LinearTerm {
        out: 35,
        s: 52,
        coeff: 1.0,
    },
    LinearTerm {
        out: 35,
        s: 80,
        coeff: 1.0,
    },
];
const LINEAR_IPSPIP: &[LinearTerm] = &[
    LinearTerm {
        out: 0,
        s: 33,
        coeff: 1.0,
    },
    LinearTerm {
        out: 0,
        s: 57,
        coeff: -1.0,
    },
    LinearTerm {
        out: 1,
        s: 54,
        coeff: 1.0,
    },
    LinearTerm {
        out: 1,
        s: 6,
        coeff: -1.0,
    },
    LinearTerm {
        out: 2,
        s: 3,
        coeff: 1.0,
    },
    LinearTerm {
        out: 2,
        s: 27,
        coeff: -1.0,
    },
    LinearTerm {
        out: 3,
        s: 0,
        coeff: 1.0,
    },
    LinearTerm {
        out: 3,
        s: 30,
        coeff: 1.0,
    },
    LinearTerm {
        out: 3,
        s: 60,
        coeff: 1.0,
    },
    LinearTerm {
        out: 4,
        s: 34,
        coeff: 1.0,
    },
    LinearTerm {
        out: 4,
        s: 58,
        coeff: -1.0,
    },
    LinearTerm {
        out: 5,
        s: 55,
        coeff: 1.0,
    },
    LinearTerm {
        out: 5,
        s: 7,
        coeff: -1.0,
    },
    LinearTerm {
        out: 6,
        s: 4,
        coeff: 1.0,
    },
    LinearTerm {
        out: 6,
        s: 28,
        coeff: -1.0,
    },
    LinearTerm {
        out: 7,
        s: 1,
        coeff: 1.0,
    },
    LinearTerm {
        out: 7,
        s: 31,
        coeff: 1.0,
    },
    LinearTerm {
        out: 7,
        s: 61,
        coeff: 1.0,
    },
    LinearTerm {
        out: 8,
        s: 35,
        coeff: 1.0,
    },
    LinearTerm {
        out: 8,
        s: 59,
        coeff: -1.0,
    },
    LinearTerm {
        out: 9,
        s: 56,
        coeff: 1.0,
    },
    LinearTerm {
        out: 9,
        s: 8,
        coeff: -1.0,
    },
    LinearTerm {
        out: 10,
        s: 5,
        coeff: 1.0,
    },
    LinearTerm {
        out: 10,
        s: 29,
        coeff: -1.0,
    },
    LinearTerm {
        out: 11,
        s: 2,
        coeff: 1.0,
    },
    LinearTerm {
        out: 11,
        s: 32,
        coeff: 1.0,
    },
    LinearTerm {
        out: 11,
        s: 62,
        coeff: 1.0,
    },
    LinearTerm {
        out: 12,
        s: 42,
        coeff: 1.0,
    },
    LinearTerm {
        out: 12,
        s: 66,
        coeff: -1.0,
    },
    LinearTerm {
        out: 13,
        s: 63,
        coeff: 1.0,
    },
    LinearTerm {
        out: 13,
        s: 15,
        coeff: -1.0,
    },
    LinearTerm {
        out: 14,
        s: 12,
        coeff: 1.0,
    },
    LinearTerm {
        out: 14,
        s: 36,
        coeff: -1.0,
    },
    LinearTerm {
        out: 15,
        s: 9,
        coeff: 1.0,
    },
    LinearTerm {
        out: 15,
        s: 39,
        coeff: 1.0,
    },
    LinearTerm {
        out: 15,
        s: 69,
        coeff: 1.0,
    },
    LinearTerm {
        out: 16,
        s: 43,
        coeff: 1.0,
    },
    LinearTerm {
        out: 16,
        s: 67,
        coeff: -1.0,
    },
    LinearTerm {
        out: 17,
        s: 64,
        coeff: 1.0,
    },
    LinearTerm {
        out: 17,
        s: 16,
        coeff: -1.0,
    },
    LinearTerm {
        out: 18,
        s: 13,
        coeff: 1.0,
    },
    LinearTerm {
        out: 18,
        s: 37,
        coeff: -1.0,
    },
    LinearTerm {
        out: 19,
        s: 10,
        coeff: 1.0,
    },
    LinearTerm {
        out: 19,
        s: 40,
        coeff: 1.0,
    },
    LinearTerm {
        out: 19,
        s: 70,
        coeff: 1.0,
    },
    LinearTerm {
        out: 20,
        s: 44,
        coeff: 1.0,
    },
    LinearTerm {
        out: 20,
        s: 68,
        coeff: -1.0,
    },
    LinearTerm {
        out: 21,
        s: 65,
        coeff: 1.0,
    },
    LinearTerm {
        out: 21,
        s: 17,
        coeff: -1.0,
    },
    LinearTerm {
        out: 22,
        s: 14,
        coeff: 1.0,
    },
    LinearTerm {
        out: 22,
        s: 38,
        coeff: -1.0,
    },
    LinearTerm {
        out: 23,
        s: 11,
        coeff: 1.0,
    },
    LinearTerm {
        out: 23,
        s: 41,
        coeff: 1.0,
    },
    LinearTerm {
        out: 23,
        s: 71,
        coeff: 1.0,
    },
    LinearTerm {
        out: 24,
        s: 51,
        coeff: 1.0,
    },
    LinearTerm {
        out: 24,
        s: 75,
        coeff: -1.0,
    },
    LinearTerm {
        out: 25,
        s: 72,
        coeff: 1.0,
    },
    LinearTerm {
        out: 25,
        s: 24,
        coeff: -1.0,
    },
    LinearTerm {
        out: 26,
        s: 21,
        coeff: 1.0,
    },
    LinearTerm {
        out: 26,
        s: 45,
        coeff: -1.0,
    },
    LinearTerm {
        out: 27,
        s: 18,
        coeff: 1.0,
    },
    LinearTerm {
        out: 27,
        s: 48,
        coeff: 1.0,
    },
    LinearTerm {
        out: 27,
        s: 78,
        coeff: 1.0,
    },
    LinearTerm {
        out: 28,
        s: 52,
        coeff: 1.0,
    },
    LinearTerm {
        out: 28,
        s: 76,
        coeff: -1.0,
    },
    LinearTerm {
        out: 29,
        s: 73,
        coeff: 1.0,
    },
    LinearTerm {
        out: 29,
        s: 25,
        coeff: -1.0,
    },
    LinearTerm {
        out: 30,
        s: 22,
        coeff: 1.0,
    },
    LinearTerm {
        out: 30,
        s: 46,
        coeff: -1.0,
    },
    LinearTerm {
        out: 31,
        s: 19,
        coeff: 1.0,
    },
    LinearTerm {
        out: 31,
        s: 49,
        coeff: 1.0,
    },
    LinearTerm {
        out: 31,
        s: 79,
        coeff: 1.0,
    },
    LinearTerm {
        out: 32,
        s: 53,
        coeff: 1.0,
    },
    LinearTerm {
        out: 32,
        s: 77,
        coeff: -1.0,
    },
    LinearTerm {
        out: 33,
        s: 74,
        coeff: 1.0,
    },
    LinearTerm {
        out: 33,
        s: 26,
        coeff: -1.0,
    },
    LinearTerm {
        out: 34,
        s: 23,
        coeff: 1.0,
    },
    LinearTerm {
        out: 34,
        s: 47,
        coeff: -1.0,
    },
    LinearTerm {
        out: 35,
        s: 20,
        coeff: 1.0,
    },
    LinearTerm {
        out: 35,
        s: 50,
        coeff: 1.0,
    },
    LinearTerm {
        out: 35,
        s: 80,
        coeff: 1.0,
    },
];
fn sigma_deriv_spec(op_name: &str) -> Option<FamilySpec> {
    match op_name {
        "ipspnucsp" | "ipsprinvsp" => Some(FamilySpec {
            rank: 12,
            nbuf: 8,
            ops: &OPS_IPIPNUCIP,
            s_table: &S27,
            gout_perm: &[],
            dot_terms: None,
            linear_terms: Some(LINEAR_IPSP),
        }),
        "ipipspnucsp" | "ipipsprinvsp" => Some(FamilySpec {
            rank: 36,
            nbuf: 16,
            ops: &OPS_IPIPIPRINVIP,
            s_table: &S81,
            gout_perm: &[],
            dot_terms: None,
            linear_terms: Some(LINEAR_IPIPSP),
        }),
        "ipspnucspip" | "ipsprinvspip" => Some(FamilySpec {
            rank: 36,
            nbuf: 16,
            ops: &OPS_IPIPRINVIPIP,
            s_table: &S81,
            gout_perm: &[],
            dot_terms: None,
            linear_terms: Some(LINEAR_IPSPIP),
        }),
        _ => None,
    }
}

/// Resolve a `deriv3`/`deriv4` operator name to its [`FamilySpec`], or `None`.
fn family_spec(op_name: &str) -> Option<FamilySpec> {
    let spec = match op_name {
        // W5-06: X2C base families (rank 1) — the undifferentiated parents of
        // the Wave-3 `ippnucp` / `ipprinvp` pair.
        "pnucp" | "prinvp" => FamilySpec {
            rank: 1,
            nbuf: 4,
            ops: &OPS_PNUCP,
            s_table: &S9_PNUCP,
            gout_perm: &[],
            dot_terms: Some(&DOT_PNUCP),
            linear_terms: None,
        },
        "ippnucp" | "ipprinvp" => FamilySpec {
            rank: 3,
            nbuf: 8,
            ops: &OPS_IPIPNUCIP,
            s_table: &S27,
            gout_perm: &[],
            dot_terms: Some(&DOT_IPPNUCP),
            linear_terms: None,
        },
        "ippnucpip" | "ipprinvpip" => FamilySpec {
            rank: 9,
            nbuf: 16,
            ops: &OPS_IPIPRINVIPIP,
            s_table: &S81,
            gout_perm: &[],
            dot_terms: Some(&DOT_IPPNUCPIP),
            linear_terms: None,
        },
        "ipippnucp" | "ipipprinvp" => FamilySpec {
            rank: 9,
            nbuf: 16,
            ops: &OPS_IPIPIPRINVIP,
            s_table: &S81,
            gout_perm: &[],
            dot_terms: Some(&DOT_IPIPPNUCP),
            linear_terms: None,
        },
        // deriv3 (rank 27)
        "ipipipnuc" | "ipipiprinv" => FamilySpec {
            rank: 27,
            nbuf: 8,

            ops: &OPS_IPIPIP,
            s_table: &S27,
            gout_perm: &PERM_IPIPIP,
            dot_terms: None,
            linear_terms: None,
        },
        "ipipnucip" | "ipiprinvip" => FamilySpec {
            rank: 27,
            nbuf: 8,

            ops: &OPS_IPIPNUCIP,
            s_table: &S27,
            gout_perm: &PERM_IPIPNUCIP,
            dot_terms: None,
            linear_terms: None,
        },
        // deriv4 (rank 81)
        "ipipipiprinv" => FamilySpec {
            rank: 81,
            nbuf: 16,

            ops: &OPS_IPIPIPIPRINV,
            s_table: &S81,
            gout_perm: &PERM_IPIPIPIPRINV,
            dot_terms: None,
            linear_terms: None,
        },
        "ipiprinvipip" => FamilySpec {
            rank: 81,
            nbuf: 16,

            ops: &OPS_IPIPRINVIPIP,
            s_table: &S81,
            gout_perm: &PERM_IPIPRINVIPIP,
            dot_terms: None,
            linear_terms: None,
        },
        "ipipiprinvip" => FamilySpec {
            rank: 81,
            nbuf: 16,

            ops: &OPS_IPIPIPRINVIP,
            s_table: &S81,
            gout_perm: &PERM_IPIPIPRINVIP,
            dot_terms: None,
            linear_terms: None,
        },
        _ => return None,
    };
    Some(spec)
}

/// True if `op_name` is one of the HESS-04 3rd/4th-order families.
pub fn is_deriv34(op_name: &str) -> bool {
    family_spec(op_name).is_some()
}

/// Output component count (`component_rank`) for a deriv34 family, or 0.
pub fn deriv34_rank(op_name: &str) -> usize {
    family_spec(op_name).map(|s| s.rank).unwrap_or(0)
}

/// Apply the bra-center nabla `D_I` to one axis block of a per-axis G-tensor.
///
/// `g[axis_off + j*dj + i]` layout (same as `contract_grad_1e_bra`). Fills
/// `dst[..]` for all `i in 0..=i_max`, `j in 0..=j_max`:
///   `i==0`: `-2ai * g[j*dj+1]`
///   `i>=1`: `i * g[j*dj+i-1] - 2ai * g[j*dj+i+1]`
fn apply_di(src: &[f64], dst: &mut [f64], dj: usize, j_max: usize, i_max: usize, ai2: f64) {
    for j in 0..=j_max {
        let jb = j * dj;
        dst[jb] = ai2 * src[jb + 1];
        for i in 1..=i_max {
            dst[jb + i] = i as f64 * src[jb + i - 1] + ai2 * src[jb + i + 1];
        }
    }
}

/// Apply the ket-center nabla `D_J` to one axis block of a per-axis G-tensor.
///
///   `j==0`: `-2aj * g[1*dj+i]`
///   `j>=1`: `j * g[(j-1)*dj+i] - 2aj * g[(j+1)*dj+i]`
fn apply_dj(src: &[f64], dst: &mut [f64], dj: usize, j_max: usize, i_max: usize, aj2: f64) {
    for i in 0..=i_max {
        dst[i] = aj2 * src[dj + i];
        for j in 1..=j_max {
            let jb = j * dj;
            dst[jb + i] = j as f64 * src[jb - dj + i] + aj2 * src[jb + dj + i];
        }
    }
}

/// Evaluate one rank-27/81 derivative family for a single primitive pair over a
/// list of `(origin, charge_factor)` Coulomb centers.
///
/// `origins` is `(coord, factor)`:
///   - nuclear families: `[(atom.coord, -(Z as f64)) for each atom]`
///   - rinv families: `[(rinv_orig, 1.0)]`
///
/// Returns a component-leading cart block: `out[comp * nci*ncj + cj*nci + ci]`,
/// length `rank * nci * ncj`. Column-major (bra fastest) inner block — matches
/// `cart_to_sph_1e` and the launcher's staging scatter.
#[allow(clippy::too_many_arguments)]
fn contract_deriv34_pair(
    spec: &FamilySpec,
    ai: f64,
    aj: f64,
    ri: [f64; 3],
    rj: [f64; 3],
    li: u8,
    lj: u8,
    origins: &[([f64; 3], f64)],
) -> Vec<f64> {
    let nci = ncart(li);
    let ncj = ncart(lj);
    let block_len = nci * ncj;
    let mut out = vec![0.0_f64; spec.rank * block_len];

    // G-tensor headroom (computed from the op targets). g0 must hold every level
    // any op reads: a D_I op with target `i_l+i_off` reads the source at `i+1`,
    // so g0 needs bra up to (max i_off)+1; likewise ket up to (max j_off)+1 for
    // D_J. li/lj also bound the minimum (the final contraction reads i<=li, j<=lj).
    let mut max_i_off = 0u32;
    let mut max_j_off = 0u32;
    for op in spec.ops {
        let (io, jo) = match *op {
            Op::DI { i_off, j_off, .. } | Op::DJ { i_off, j_off, .. } => (i_off, j_off),
        };
        max_i_off = max_i_off.max(io);
        max_j_off = max_j_off.max(jo);
    }
    let i_top = li as u32 + max_i_off + 1;
    let j_top = lj as u32 + max_j_off + 1;
    let nmax = i_top + j_top; // VRR ceiling (covers all i+j the recurrences touch)
    let dj = (nmax + 1) as usize;
    let g_per_axis = ((nmax + 1) * (j_top + 1)) as usize;
    let three = 3 * g_per_axis;

    let zeta = ai + aj;
    let aij2 = 0.5 / zeta;
    let rirj = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
    let rr = rirj[0] * rirj[0] + rirj[1] * rirj[1] + rirj[2] * rirj[2];
    let fac = (-ai * aj / zeta * rr).exp();
    let rp = [
        (ai * ri[0] + aj * rj[0]) / zeta,
        (ai * ri[1] + aj * rj[1]) / zeta,
        (ai * ri[2] + aj * rj[2]) / zeta,
    ];
    let ai2 = -2.0 * ai;
    let aj2 = -2.0 * aj;

    let ci_comps = cart_comps(li);
    let cj_comps = cart_comps(lj);

    // g-buffer storage: nbuf buffers, each `three` long (3 axis blocks).
    // g[0] is the base; g[1..] are filled by the op sequence per Rys root.
    let mut g = vec![vec![0.0_f64; three]; spec.nbuf];

    for &(rc, charge_factor) in origins {
        // Boys argument x = zeta * |P - C|^2 (crij = C - P).
        let crij = [rc[0] - rp[0], rc[1] - rp[1], rc[2] - rp[2]];
        let x_boys = zeta * (crij[0] * crij[0] + crij[1] * crij[1] + crij[2] * crij[2]);
        let nrys = (nmax / 2 + 1) as usize;
        let (u_arr, w_arr) = rys_roots_host(nrys, x_boys);
        // fac1 = 2*PI * charge_factor * fac / zeta (g1e.c nuclear prefactor).
        let fac1 = 2.0 * std::f64::consts::PI * charge_factor * fac / zeta;

        for n in 0..nrys {
            let u_n = u_arr[n];
            let w_n = w_arr[n];
            let tau = u_n / (1.0 + u_n);
            let rt = aij2 - aij2 * tau;
            let c00 = [
                (rp[0] - ri[0]) + tau * crij[0],
                (rp[1] - ri[1]) + tau * crij[1],
                (rp[2] - ri[2]) + tau * crij[2],
            ];

            // Build g0 (base nuclear Rys G-tensor for this root) — VRR then HRR.
            for v in g[0].iter_mut() {
                *v = 0.0;
            }
            g[0][0] = 1.0; // gx base
            g[0][g_per_axis] = 1.0; // gy base
            g[0][2 * g_per_axis] = fac1 * w_n; // gz base

            for axis in 0..3usize {
                let off = axis * g_per_axis;
                vrr_2e_step_host(&mut g[0][off..off + g_per_axis], c00[axis], rt, nmax, 1);
            }
            if j_top >= 1 {
                for axis in 0..3usize {
                    let off = axis * g_per_axis;
                    hrr_step_host(
                        &mut g[0][off..off + g_per_axis],
                        rirj[axis],
                        1,
                        nmax + 1,
                        nmax,
                        j_top,
                    );
                }
            }

            // Apply the family op sequence g1.. Each op fills exactly its target
            // range `i in 0..=li+i_off`, `j in 0..=lj+j_off` (verbatim from the
            // family's `G2E_D_*` i_l+N / j_l+M arguments), reading the source one
            // level above on the derivative axis.
            for op in spec.ops {
                let (src, dst, is_di, i_off, j_off) = match *op {
                    Op::DI {
                        dst,
                        src,
                        i_off,
                        j_off,
                    } => (src, dst, true, i_off, j_off),
                    Op::DJ {
                        dst,
                        src,
                        i_off,
                        j_off,
                    } => (src, dst, false, i_off, j_off),
                };
                let i_tgt = (li as u32 + i_off) as usize;
                let j_tgt = (lj as u32 + j_off) as usize;
                // Split-borrow src and dst rows.
                let (src_row, dst_row) = borrow_two(&mut g, src, dst);
                for axis in 0..3usize {
                    let off = axis * g_per_axis;
                    let s = &src_row[off..off + g_per_axis];
                    let d = &mut dst_row[off..off + g_per_axis];
                    for x in d.iter_mut() {
                        *x = 0.0;
                    }
                    if is_di {
                        apply_di(s, d, dj, j_tgt, i_tgt, ai2);
                    } else {
                        apply_dj(s, d, dj, j_tgt, i_tgt, aj2);
                    }
                }
            }

            // Contract s[rank] and scatter into out via the gout permutation.
            let gx = 0usize;
            let gy = g_per_axis;
            let gz = 2 * g_per_axis;
            for (cj_idx, &(jx, jy, jz)) in cj_comps.iter().enumerate() {
                for (ci_idx, &(ix, iy, iz)) in ci_comps.iter().enumerate() {
                    let nx = jx as usize * dj + ix as usize;
                    let ny = jy as usize * dj + iy as usize;
                    let nz = jz as usize * dj + iz as usize;
                    let bn = cj_idx * nci + ci_idx;
                    if let Some(linear_terms) = spec.linear_terms {
                        for term in linear_terms {
                            let (sx, sy, sz) = spec.s_table[term.s];
                            let val = g[sx][gx + nx] * g[sy][gy + ny] * g[sz][gz + nz];
                            out[term.out * block_len + bn] += term.coeff * val;
                        }
                    } else if let Some(dot_terms) = spec.dot_terms {
                        for (comp, terms) in dot_terms.iter().enumerate() {
                            let mut val = 0.0;
                            for &term in terms {
                                let (sx, sy, sz) = spec.s_table[term];
                                val += g[sx][gx + nx] * g[sy][gy + ny] * g[sz][gz + nz];
                            }
                            out[comp * block_len + bn] += val;
                        }
                    } else {
                        for (comp, &perm) in spec.gout_perm.iter().enumerate() {
                            let (sx, sy, sz) = spec.s_table[perm];
                            let val = g[sx][gx + nx] * g[sy][gy + ny] * g[sz][gz + nz];
                            out[comp * block_len + bn] += val;
                        }
                    }
                }
            }
        }
    }

    out
}

/// Mutably borrow two distinct rows of a `Vec<Vec<f64>>`. Panics if `a == b`.
fn borrow_two(g: &mut [Vec<f64>], a: usize, b: usize) -> (&mut Vec<f64>, &mut Vec<f64>) {
    assert_ne!(a, b, "borrow_two requires distinct indices");
    if a < b {
        let (lo, hi) = g.split_at_mut(b);
        (&mut lo[a], &mut hi[0])
    } else {
        let (lo, hi) = g.split_at_mut(a);
        (&mut hi[0], &mut lo[b])
    }
}

/// Evaluate a full contracted cart block for a deriv34 family over a primitive
/// shell pair. Returns `out[comp * (nctr_i*nci) * (nctr_j*ncj) + ...]` is NOT
/// produced here; instead this returns the per-(ci,cj) contracted-primitive
/// blocks as `out[(ci*nctr_j+cj)] -> rank*nci*ncj`, matching the HESS-01 path.
///
/// Returns a `Vec<f64>` of length `nctr_i * nctr_j * rank * nci * ncj`, with the
/// layout `[(ci*nctr_j+cj)][comp][cj_cart*nci + ci_cart]`.
#[allow(clippy::too_many_arguments)]
fn contract_family_block(
    spec: FamilySpec,
    li: u8,
    lj: u8,
    ri: [f64; 3],
    rj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    n_ctr_i: usize,
    n_ctr_j: usize,
    origins: &[([f64; 3], f64)],
) -> Vec<f64> {
    let n_prim_i = exps_i.len();
    let n_prim_j = exps_j.len();
    let nci = ncart(li);
    let ncj = ncart(lj);
    let block_len = nci * ncj;
    let total_len = spec.rank * block_len;

    let mut out = vec![0.0_f64; n_ctr_i * n_ctr_j * total_len];

    for ip in 0..n_prim_i {
        let ai = exps_i[ip];
        for jp in 0..n_prim_j {
            let aj = exps_j[jp];
            let pair = contract_deriv34_pair(&spec, ai, aj, ri, rj, li, lj, origins);
            for ci in 0..n_ctr_i {
                let cci = coeff_i[ip * n_ctr_i + ci];
                if cci == 0.0 {
                    continue;
                }
                for cj in 0..n_ctr_j {
                    let ccj = coeff_j[jp * n_ctr_j + cj];
                    if ccj == 0.0 {
                        continue;
                    }
                    let w = cci * ccj;
                    let base = (ci * n_ctr_j + cj) * total_len;
                    for (k, v) in pair.iter().enumerate() {
                        out[base + k] += w * v;
                    }
                }
            }
        }
    }

    out
}

#[allow(clippy::too_many_arguments)]
pub fn contract_deriv34_block(
    op_name: &str,
    li: u8,
    lj: u8,
    ri: [f64; 3],
    rj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    n_ctr_i: usize,
    n_ctr_j: usize,
    origins: &[([f64; 3], f64)],
) -> Option<Vec<f64>> {
    Some(contract_family_block(
        family_spec(op_name)?,
        li,
        lj,
        ri,
        rj,
        exps_i,
        exps_j,
        coeff_i,
        coeff_j,
        n_ctr_i,
        n_ctr_j,
        origins,
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn contract_sigma_deriv_block(
    op_name: &str,
    li: u8,
    lj: u8,
    ri: [f64; 3],
    rj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    n_ctr_i: usize,
    n_ctr_j: usize,
    origins: &[([f64; 3], f64)],
) -> Option<Vec<f64>> {
    Some(contract_family_block(
        sigma_deriv_spec(op_name)?,
        li,
        lj,
        ri,
        rj,
        exps_i,
        exps_j,
        coeff_i,
        coeff_j,
        n_ctr_i,
        n_ctr_j,
        origins,
    ))
}

/// Build the `(origin, factor)` list for the nuclear families (sum over atoms,
/// charge factor `-Z_C`).
pub fn nuclear_origins(atoms: &[Atom]) -> Vec<([f64; 3], f64)> {
    atoms
        .iter()
        .map(|a| (a.coord_bohr, -(a.atomic_number as f64)))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
//  DEVICE: the deriv3/deriv4 families on-device (§21).
//
//  These families were host-only "(FND-02)": the bra/ket +2/+3 headroom can
//  elevate the nuclear Rys order past `MAX_DEVICE_NROOTS = 5`, and only
//  `rys_roots_host` served 6..12. That stopped being true when task 33-01 put
//  the inline Wheeler/Jacobi entry (`rys_roots_ext_dev`) on the device — the
//  stated blocker is gone, and what remained was the port.
//
//  ONE kernel serves all ten families. Everything that differs between them is
//  a *table*: the op sequence, the `s[rank]` triple-product indices, and the
//  gout map. Uploading those rather than specializing on the family keeps a
//  single program (and a single thing to verify) where ten comptime variants of
//  a 300-line body would otherwise be needed.
//
//  The three gout schemes — `gout_perm`, `dot_terms`, `linear_terms` — collapse
//  into one flat term list of `(group, out_component, s_index, coeff)`. Terms
//  sharing a `group` accumulate into one register and flush once, which is what
//  makes the collapse *exact*: a `dot_terms` component is `out += (t1+t2+t3)`
//  and a `linear_terms` one is `out += c*t` per term, and grouping reproduces
//  each rather than approximating both.
// ─────────────────────────────────────────────────────────────────────────────

use crate::backend::ResolvedBackend;
use crate::math::rys::rys_roots_fixed;
use crate::math::rys_wheeler::{
    EXT_TABLES_LEN, ext_rys_out_slots, ext_rys_slots, ext_rys_tables, rys_roots_ext_dev,
};
use cubecl::prelude::*;

/// Per-axis vertical recurrence at stride one — [`vrr_2e_step_host`] on device.
#[cube]
fn d34_vrr_axis<F: Float>(g: &mut Array<F>, off: u32, c00: F, b10: F, nmax: u32) {
    if nmax >= 1u32 {
        g[(off + 1u32) as usize] = c00 * g[off as usize];
        let mut n = 1u32;
        while n < nmax {
            g[(off + n + 1u32) as usize] =
                F::cast_from(n) * b10 * g[(off + n - 1u32) as usize] + c00 * g[(off + n) as usize];
            n += 1u32;
        }
    }
}

/// Per-axis horizontal recurrence at `di = 1` — [`hrr_step_host`] on device.
#[cube]
fn d34_hrr_axis<F: Float>(g: &mut Array<F>, off: u32, rirj: F, dj: u32, li_max: u32, lj: u32) {
    let mut j = 1u32;
    while j <= lj {
        let i_max = li_max - j;
        let mut i = 0u32;
        while i <= i_max {
            let out = off + j * dj + i;
            let hi = off + (j - 1u32) * dj + i + 1u32;
            let lo = off + (j - 1u32) * dj + i;
            g[out as usize] = g[hi as usize] + rirj * g[lo as usize];
            i += 1u32;
        }
        j += 1u32;
    }
}

/// Bra-center nabla `D_I` on one axis block — [`apply_di`] on device.
///
/// `src` and `dst` are offsets into the same slab and name *different*
/// g-buffers, which is what makes reading one while writing the other sound;
/// the host states the same invariant through `borrow_two`'s `assert_ne!`.
#[cube]
fn d34_apply_di<F: Float>(
    g: &mut Array<F>,
    src: u32,
    dst: u32,
    dj: u32,
    j_max: u32,
    i_max: u32,
    ai2: F,
) {
    let mut j = 0u32;
    while j <= j_max {
        let jb = j * dj;
        g[(dst + jb) as usize] = ai2 * g[(src + jb + 1u32) as usize];
        let mut i = 1u32;
        while i <= i_max {
            g[(dst + jb + i) as usize] = F::cast_from(i) * g[(src + jb + i - 1u32) as usize]
                + ai2 * g[(src + jb + i + 1u32) as usize];
            i += 1u32;
        }
        j += 1u32;
    }
}

/// Ket-center nabla `D_J` on one axis block — [`apply_dj`] on device.
#[cube]
fn d34_apply_dj<F: Float>(
    g: &mut Array<F>,
    src: u32,
    dst: u32,
    dj: u32,
    j_max: u32,
    i_max: u32,
    aj2: F,
) {
    let mut i = 0u32;
    while i <= i_max {
        g[(dst + i) as usize] = aj2 * g[(src + dj + i) as usize];
        let mut j = 1u32;
        while j <= j_max {
            let jb = j * dj;
            g[(dst + jb + i) as usize] = F::cast_from(j) * g[(src + jb - dj + i) as usize]
                + aj2 * g[(src + jb + dj + i) as usize];
            j += 1u32;
        }
        i += 1u32;
    }
}

/// One primitive pair of a deriv3/deriv4 family, on device (§21).
///
/// The work item is one `(ip, jp)` primitive pair; it walks the origins and the
/// Rys roots inside and writes its own `rank · nci · ncj` slab of `partial`, so
/// nothing is shared and nothing races. [`deriv34_weight_kernel`] then applies
/// the contraction coefficients — separating the two is what keeps the
/// accumulation order the host's, and so the result bit-comparable, without
/// atomics.
///
/// Statement for statement this is [`contract_deriv34_pair`], with the family's
/// `ops` / `s_table` / gout map read from uploaded tables instead of matched on.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn deriv34_pair_kernel<F: Float + CubeElement>(
    exps_i: &Array<F>,
    exps_j: &Array<F>,
    origins: &Array<F>,
    ops: &Array<u32>,
    s_table: &Array<u32>,
    term_idx: &Array<u32>,
    term_coeff: &Array<F>,
    cart_i: &Array<u32>,
    cart_j: &Array<u32>,
    rys_tab: &Array<f64>,
    g: &mut Array<F>,
    partial: &mut Array<F>,
    rix: F,
    riy: F,
    riz: F,
    rjx: F,
    rjy: F,
    rjz: F,
    pie4: F,
    two_pi: F,
    n_prim_j: u32,
    n_pairs: u32,
    n_origins: u32,
    n_ops: u32,
    n_terms: u32,
    li: u32,
    lj: u32,
    nci: u32,
    ncj: u32,
    dj: u32,
    g_per_axis: u32,
    nmax: u32,
    j_top: u32,
    rank: u32,
    n_slots: u32,
    g_stride: u32,
    #[comptime] nroots: u32,
) {
    let slot = (CUBE_POS as u32) * (CUBE_DIM as u32) + (UNIT_POS as u32);
    let three = 3u32 * g_per_axis;
    let block_len = nci * ncj;
    let total = rank * block_len;

    // The Rys roots are per work item and read only here, so they are private
    // arrays rather than buffers — the 2e kernel's `urys`/`wrys` for the same
    // reason. The extended entry is f64-only (its double-double arms are what
    // buy the accuracy above order five), so it lands in its own pair.
    let mut urys = Array::<F>::new(comptime!(ext_rys_slots(nroots)));
    let mut wrys = Array::<F>::new(comptime!(ext_rys_slots(nroots)));
    let mut uext = Array::<f64>::new(comptime!(ext_rys_out_slots(nroots)));
    let mut wext = Array::<f64>::new(comptime!(ext_rys_out_slots(nroots)));

    let mut pair = slot;
    while pair < n_pairs {
        let ip = pair / n_prim_j;
        let jp = pair % n_prim_j;
        let ai = exps_i[ip as usize];
        let aj = exps_j[jp as usize];

        let out_base = pair * total;
        let mut k = 0u32;
        while k < total {
            partial[(out_base + k) as usize] = F::new(0.0_f32);
            k += 1u32;
        }

        let gbase = slot * g_stride;

        let zeta = ai + aj;
        let aij2 = F::new(0.5_f32) / zeta;
        let rirjx = rix - rjx;
        let rirjy = riy - rjy;
        let rirjz = riz - rjz;
        let rr = rirjx * rirjx + rirjy * rirjy + rirjz * rirjz;
        let fac = F::exp(F::new(0.0_f32) - ai * aj / zeta * rr);
        let rpx = (ai * rix + aj * rjx) / zeta;
        let rpy = (ai * riy + aj * rjy) / zeta;
        let rpz = (ai * riz + aj * rjz) / zeta;
        let ai2 = F::new(0.0_f32) - F::new(2.0_f32) * ai;
        let aj2 = F::new(0.0_f32) - F::new(2.0_f32) * aj;

        let mut oi = 0u32;
        while oi < n_origins {
            let ob = oi * 4u32;
            let crijx = origins[ob as usize] - rpx;
            let crijy = origins[(ob + 1u32) as usize] - rpy;
            let crijz = origins[(ob + 2u32) as usize] - rpz;
            let charge = origins[(ob + 3u32) as usize];
            let x_boys = zeta * (crijx * crijx + crijy * crijy + crijz * crijz);

            if comptime!(nroots <= 5u32) {
                // `rys_roots_fixed` is the whole of `CINTrys_roots` for the
                // fixed orders: the two global table branches first, the
                // per-order polynomial fit only in the band between them.
                rys_roots_fixed::<F>(rys_tab, x_boys, &mut urys, &mut wrys, pie4, nroots);
            } else {
                // Orders six through twelve: the inline Wheeler/Jacobi entry
                // (task 33-01). This is the arm whose absence routed these
                // families to the host in the first place.
                rys_roots_ext_dev(
                    rys_tab,
                    f64::cast_from(x_boys),
                    &mut uext,
                    &mut wext,
                    nroots,
                );
                #[unroll]
                for e in 0..nroots {
                    urys[e as usize] = F::cast_from(uext[e as usize]);
                    wrys[e as usize] = F::cast_from(wext[e as usize]);
                }
            }

            let fac1 = two_pi * charge * fac / zeta;

            let mut nr = 0u32;
            while nr < comptime!(nroots) {
                let u_n = urys[nr as usize];
                let w_n = wrys[nr as usize];
                let tau = u_n / (F::new(1.0_f32) + u_n);
                let rt = aij2 - aij2 * tau;

                // g0 is zeroed and re-seeded per root, exactly as the host does.
                let mut t = 0u32;
                while t < three {
                    g[(gbase + t) as usize] = F::new(0.0_f32);
                    t += 1u32;
                }
                g[gbase as usize] = F::new(1.0_f32);
                g[(gbase + g_per_axis) as usize] = F::new(1.0_f32);
                g[(gbase + 2u32 * g_per_axis) as usize] = fac1 * w_n;

                let c00x = (rpx - rix) + tau * crijx;
                let c00y = (rpy - riy) + tau * crijy;
                let c00z = (rpz - riz) + tau * crijz;
                d34_vrr_axis::<F>(g, gbase, c00x, rt, nmax);
                d34_vrr_axis::<F>(g, gbase + g_per_axis, c00y, rt, nmax);
                d34_vrr_axis::<F>(g, gbase + 2u32 * g_per_axis, c00z, rt, nmax);

                if j_top >= 1u32 {
                    d34_hrr_axis::<F>(g, gbase, rirjx, dj, nmax, j_top);
                    d34_hrr_axis::<F>(g, gbase + g_per_axis, rirjy, dj, nmax, j_top);
                    d34_hrr_axis::<F>(g, gbase + 2u32 * g_per_axis, rirjz, dj, nmax, j_top);
                }

                // The family's op sequence: each op fills exactly its target
                // range from the source one level above on its axis.
                let mut op = 0u32;
                while op < n_ops {
                    let opb = op * 5u32;
                    let is_di = ops[opb as usize];
                    let dst = ops[(opb + 1u32) as usize];
                    let src = ops[(opb + 2u32) as usize];
                    let i_tgt = li + ops[(opb + 3u32) as usize];
                    let j_tgt = lj + ops[(opb + 4u32) as usize];
                    let mut axis = 0u32;
                    while axis < 3u32 {
                        let so = gbase + src * three + axis * g_per_axis;
                        let dof = gbase + dst * three + axis * g_per_axis;
                        let mut z = 0u32;
                        while z < g_per_axis {
                            g[(dof + z) as usize] = F::new(0.0_f32);
                            z += 1u32;
                        }
                        if is_di == 1u32 {
                            d34_apply_di::<F>(g, so, dof, dj, j_tgt, i_tgt, ai2);
                        } else {
                            d34_apply_dj::<F>(g, so, dof, dj, j_tgt, i_tgt, aj2);
                        }
                        axis += 1u32;
                    }
                    op += 1u32;
                }

                // Contract `s[rank]` and scatter through the flat term list.
                let mut cj = 0u32;
                while cj < ncj {
                    let jb = cj * 3u32;
                    let jx = cart_j[jb as usize];
                    let jy = cart_j[(jb + 1u32) as usize];
                    let jz = cart_j[(jb + 2u32) as usize];
                    let mut ci = 0u32;
                    while ci < nci {
                        let ib = ci * 3u32;
                        let nx = jx * dj + cart_i[ib as usize];
                        let ny = jy * dj + cart_i[(ib + 1u32) as usize];
                        let nz = jz * dj + cart_i[(ib + 2u32) as usize];
                        let bn = cj * nci + ci;

                        // Terms sharing a `group` sum into `acc` and flush once.
                        let mut acc = F::new(0.0_f32);
                        let mut acc_out: u32 = 0u32;
                        let mut acc_group: u32 = 0u32;
                        let mut have: u32 = 0u32;
                        let mut ti: u32 = 0u32;
                        while ti < n_terms {
                            let tb = ti * 3u32;
                            let group = term_idx[tb as usize];
                            let out_comp = term_idx[(tb + 1u32) as usize];
                            let sb = term_idx[(tb + 2u32) as usize] * 3u32;
                            let sx = s_table[sb as usize];
                            let sy = s_table[(sb + 1u32) as usize];
                            let sz = s_table[(sb + 2u32) as usize];
                            if have == 1u32 && group != acc_group {
                                partial[(out_base + acc_out * block_len + bn) as usize] += acc;
                                acc = F::new(0.0_f32);
                            }
                            let val = g[(gbase + sx * three + nx) as usize]
                                * g[(gbase + sy * three + g_per_axis + ny) as usize]
                                * g[(gbase + sz * three + 2u32 * g_per_axis + nz) as usize];
                            acc += term_coeff[ti as usize] * val;
                            acc_out = out_comp;
                            acc_group = group;
                            have = 1u32;
                            ti += 1u32;
                        }
                        if have == 1u32 {
                            partial[(out_base + acc_out * block_len + bn) as usize] += acc;
                        }
                        ci += 1u32;
                    }
                    cj += 1u32;
                }
                nr += 1u32;
            }
            oi += 1u32;
        }
        pair += n_slots;
    }
}

/// Apply the contraction coefficients to [`deriv34_pair_kernel`]'s partials.
///
/// One work item per output element. The sum walks `(ip, jp)` in the host's
/// order and starts from zero, and a zero coefficient is skipped rather than
/// multiplied — both because that is what `contract_family_block` does, and the
/// point of this kernel is to be the same arithmetic in the same sequence.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn deriv34_weight_kernel<F: Float + CubeElement>(
    partial: &Array<F>,
    coeff_i: &Array<F>,
    coeff_j: &Array<F>,
    out: &mut Array<F>,
    n_prim_i: u32,
    n_prim_j: u32,
    n_ctr_i: u32,
    n_ctr_j: u32,
    total: u32,
    n_items: u32,
    n_slots: u32,
) {
    let slot = (CUBE_POS as u32) * (CUBE_DIM as u32) + (UNIT_POS as u32);
    let mut item = slot;
    while item < n_items {
        let k = item % total;
        let cc = item / total;
        let ci = cc / n_ctr_j;
        let cj = cc % n_ctr_j;
        let mut acc = F::new(0.0_f32);
        let mut ip = 0u32;
        while ip < n_prim_i {
            let cci = coeff_i[(ip * n_ctr_i + ci) as usize];
            if cci != F::new(0.0_f32) {
                let mut jp = 0u32;
                while jp < n_prim_j {
                    let ccj = coeff_j[(jp * n_ctr_j + cj) as usize];
                    if ccj != F::new(0.0_f32) {
                        acc += cci * ccj * partial[((ip * n_prim_j + jp) * total + k) as usize];
                    }
                    jp += 1u32;
                }
            }
            ip += 1u32;
        }
        out[(cc * total + k) as usize] = acc;
        item += n_slots;
    }
}

/// Everything one family's device dispatch needs that is not a scalar (§21).
struct Deriv34Tables {
    /// Five `u32` per op: `is_di, dst, src, i_off, j_off`.
    ops: Vec<u32>,
    /// Three `u32` per `s[]` entry: the g-buffer index read on x / y / z.
    s_table: Vec<u32>,
    /// Three `u32` per term: `group, out_component, s_index`.
    term_idx: Vec<u32>,
    /// One coefficient per term.
    term_coeff: Vec<f64>,
    /// g-buffers this family uses.
    nbuf: u32,
}

/// Flatten a [`FamilySpec`] for the device (§21).
///
/// The three gout schemes become one term list. Terms that must sum before
/// touching the output share a `group`; terms that must be added separately get
/// one each. That is the whole difference between `dot_terms`
/// (`out += t1 + t2 + t3`) and `linear_terms` (`out += c·t`, per term), and
/// encoding it as grouping is what lets one device loop be exactly both.
fn deriv34_tables(spec: &FamilySpec) -> Deriv34Tables {
    let mut ops = Vec::with_capacity(spec.ops.len() * 5);
    for op in spec.ops {
        let (is_di, dst, src, i_off, j_off) = match *op {
            Op::DI {
                dst,
                src,
                i_off,
                j_off,
            } => (1u32, dst, src, i_off, j_off),
            Op::DJ {
                dst,
                src,
                i_off,
                j_off,
            } => (0u32, dst, src, i_off, j_off),
        };
        ops.extend_from_slice(&[is_di, dst as u32, src as u32, i_off, j_off]);
    }

    let mut s_table = Vec::with_capacity(spec.s_table.len() * 3);
    for &(sx, sy, sz) in spec.s_table {
        s_table.extend_from_slice(&[sx as u32, sy as u32, sz as u32]);
    }

    let mut term_idx = Vec::new();
    let mut term_coeff = Vec::new();
    if let Some(linear) = spec.linear_terms {
        // Each term flushes on its own, as the host adds it on its own.
        for (group, term) in linear.iter().enumerate() {
            term_idx.extend_from_slice(&[group as u32, term.out as u32, term.s as u32]);
            term_coeff.push(term.coeff);
        }
    } else if let Some(dot) = spec.dot_terms {
        for (comp, terms) in dot.iter().enumerate() {
            for &t in terms {
                term_idx.extend_from_slice(&[comp as u32, comp as u32, t as u32]);
                term_coeff.push(1.0);
            }
        }
    } else {
        for (comp, &perm) in spec.gout_perm.iter().enumerate() {
            term_idx.extend_from_slice(&[comp as u32, comp as u32, perm as u32]);
            term_coeff.push(1.0);
        }
    }

    Deriv34Tables {
        ops,
        s_table,
        term_idx,
        term_coeff,
        nbuf: spec.nbuf as u32,
    }
}

/// The G-tensor geometry one family and shell pair implies (§21).
///
/// The same expressions `contract_deriv34_pair` computes — one place, so the
/// scratch the host allocates and the extents the kernel indexes cannot drift.
struct Deriv34Geometry {
    nmax: u32,
    j_top: u32,
    dj: u32,
    g_per_axis: u32,
    nroots: u32,
}

fn deriv34_geometry(spec: &FamilySpec, li: u8, lj: u8) -> Deriv34Geometry {
    let mut max_i_off = 0u32;
    let mut max_j_off = 0u32;
    for op in spec.ops {
        let (io, jo) = match *op {
            Op::DI { i_off, j_off, .. } | Op::DJ { i_off, j_off, .. } => (i_off, j_off),
        };
        max_i_off = max_i_off.max(io);
        max_j_off = max_j_off.max(jo);
    }
    let i_top = li as u32 + max_i_off + 1;
    let j_top = lj as u32 + max_j_off + 1;
    let nmax = i_top + j_top;
    Deriv34Geometry {
        nmax,
        j_top,
        dj: nmax + 1,
        g_per_axis: (nmax + 1) * (j_top + 1),
        nroots: nmax / 2 + 1,
    }
}

/// Cartesian exponent triples for `l`, flattened three `u32` per component, in
/// [`cart_comps`]'s order — the order the contraction's `ci`/`cj` walk assumes.
fn deriv34_cart_table(l: u8) -> Vec<u32> {
    let mut out = Vec::with_capacity(ncart(l) * 3);
    for (x, y, z) in cart_comps(l) {
        out.extend_from_slice(&[x, y, z]);
    }
    out
}

/// Rys `PIE4 = pi/4`, passed into the fixed-order solvers as the 2e and grids
/// kernels pass their own copies.
const DERIV34_PIE4: f64 = 0.78539816339744827900_f64;

/// Highest Rys order the device deriv34 kernel wires (§21).
///
/// One through five are the fixed-order polynomial solvers; six through twelve
/// are `rys_roots_ext_dev`'s inline Wheeler/Jacobi entry. Twelve is where the
/// host reference stops too (`HOST_RYS_NROOTS_CEILING_1E`), because above it
/// libcint needs the quadmath path that the vendor build does not compile.
pub const DERIV34_MAX_DEVICE_NROOTS: u32 = 12;

/// Run one deriv3/deriv4 family for one shell pair on `client` (§21).
///
/// Two dispatches. The first gives every primitive pair its own work item and
/// its own slab of `partial`; the second applies the contraction coefficients,
/// one work item per output element. Splitting them is what keeps the
/// accumulation order the host's — over origins and roots inside a pair, then
/// over `(ip, jp)` in the weighting — without an atomic anywhere.
#[allow(clippy::too_many_arguments)]
fn run_deriv34_device<R: Runtime>(
    client: &ComputeClient<R>,
    spec: &FamilySpec,
    li: u8,
    lj: u8,
    ri: [f64; 3],
    rj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    n_ctr_i: usize,
    n_ctr_j: usize,
    origins: &[([f64; 3], f64)],
) -> Vec<f64> {
    let tables = deriv34_tables(spec);
    let geom = deriv34_geometry(spec, li, lj);
    let nci = ncart(li);
    let ncj = ncart(lj);
    let total = spec.rank * nci * ncj;
    let n_prim_i = exps_i.len();
    let n_prim_j = exps_j.len();
    let n_pairs = n_prim_i * n_prim_j;
    let out_len = n_ctr_i * n_ctr_j * total;

    let three = 3 * geom.g_per_axis as usize;
    let g_stride = tables.nbuf as usize * three;

    // One slot per primitive pair is the ceiling that matters; the width follows
    // the backend's parallelism, as every other kernel here does.
    let hw = crate::plane::launch_hardware(client);
    let cube_dim = if hw.has_planes {
        crate::plane::backend_plane_cube_dim::<R>(client)
    } else {
        CubeDim::new_1d(crate::plane::per_unit_width(client, n_pairs, 1, usize::MAX))
    };
    let per_cube = cube_dim.num_elems() as usize;
    let cubes = if hw.has_planes {
        crate::plane::grid_cube_count(client, n_pairs.div_ceil(per_cube.max(1)))
    } else {
        1
    };
    let n_slots = (cubes as usize * per_cube).max(1);

    let origin_flat: Vec<f64> = origins
        .iter()
        .flat_map(|&(c, f)| [c[0], c[1], c[2], f])
        .collect();
    let cart_i = deriv34_cart_table(li);
    let cart_j = deriv34_cart_table(lj);
    let rys_tables = ext_rys_tables();

    let exps_i_h = client.create_from_slice(f64::as_bytes(exps_i));
    let exps_j_h = client.create_from_slice(f64::as_bytes(exps_j));
    let origins_h = client.create_from_slice(f64::as_bytes(&origin_flat));
    let ops_h = client.create_from_slice(u32::as_bytes(&tables.ops));
    let s_h = client.create_from_slice(u32::as_bytes(&tables.s_table));
    let tidx_h = client.create_from_slice(u32::as_bytes(&tables.term_idx));
    let tco_h = client.create_from_slice(f64::as_bytes(&tables.term_coeff));
    let ci_h = client.create_from_slice(u32::as_bytes(&cart_i));
    let cj_h = client.create_from_slice(u32::as_bytes(&cart_j));
    let tab_h = client.create_from_slice(f64::as_bytes(&rys_tables));
    let g_h = client.empty(n_slots * g_stride * std::mem::size_of::<f64>());
    let partial_h = client.empty(n_pairs * total * std::mem::size_of::<f64>());

    let n_terms = (tables.term_idx.len() / 3) as u32;
    let n_ops = (tables.ops.len() / 5) as u32;

    macro_rules! launch_pairs {
        ($nr:expr) => {
            deriv34_pair_kernel::launch::<f64, R>(
                client,
                crate::plane::cube_count_1d(cubes),
                cube_dim,
                unsafe { ArrayArg::from_raw_parts(exps_i_h.clone(), n_prim_i) },
                unsafe { ArrayArg::from_raw_parts(exps_j_h.clone(), n_prim_j) },
                unsafe { ArrayArg::from_raw_parts(origins_h.clone(), origin_flat.len()) },
                unsafe { ArrayArg::from_raw_parts(ops_h.clone(), tables.ops.len().max(1)) },
                unsafe { ArrayArg::from_raw_parts(s_h.clone(), tables.s_table.len()) },
                unsafe { ArrayArg::from_raw_parts(tidx_h.clone(), tables.term_idx.len()) },
                unsafe { ArrayArg::from_raw_parts(tco_h.clone(), tables.term_coeff.len()) },
                unsafe { ArrayArg::from_raw_parts(ci_h.clone(), cart_i.len()) },
                unsafe { ArrayArg::from_raw_parts(cj_h.clone(), cart_j.len()) },
                unsafe { ArrayArg::from_raw_parts(tab_h.clone(), EXT_TABLES_LEN) },
                unsafe { ArrayArg::from_raw_parts(g_h.clone(), n_slots * g_stride) },
                unsafe { ArrayArg::from_raw_parts(partial_h.clone(), n_pairs * total) },
                ri[0],
                ri[1],
                ri[2],
                rj[0],
                rj[1],
                rj[2],
                DERIV34_PIE4,
                2.0 * std::f64::consts::PI,
                n_prim_j as u32,
                n_pairs as u32,
                origins.len() as u32,
                n_ops,
                n_terms,
                li as u32,
                lj as u32,
                nci as u32,
                ncj as u32,
                geom.dj,
                geom.g_per_axis,
                geom.nmax,
                geom.j_top,
                spec.rank as u32,
                n_slots as u32,
                g_stride as u32,
                $nr,
            )
        };
    }

    // One compiled program per Rys order; `nroots` has to be comptime because
    // the fixed-order solvers and the extended entry both take it that way.
    match geom.nroots {
        1 => launch_pairs!(1u32),
        2 => launch_pairs!(2u32),
        3 => launch_pairs!(3u32),
        4 => launch_pairs!(4u32),
        5 => launch_pairs!(5u32),
        6 => launch_pairs!(6u32),
        7 => launch_pairs!(7u32),
        8 => launch_pairs!(8u32),
        9 => launch_pairs!(9u32),
        10 => launch_pairs!(10u32),
        11 => launch_pairs!(11u32),
        _ => launch_pairs!(12u32),
    }

    // ── Weighting ────────────────────────────────────────────────────────────
    let coeff_i_h = client.create_from_slice(f64::as_bytes(coeff_i));
    let coeff_j_h = client.create_from_slice(f64::as_bytes(coeff_j));
    let out_h = client.empty(out_len * std::mem::size_of::<f64>());
    let w_cube_dim = if hw.has_planes {
        crate::plane::backend_plane_cube_dim::<R>(client)
    } else {
        CubeDim::new_1d(crate::plane::per_unit_width(client, out_len, 1, usize::MAX))
    };
    let w_per_cube = w_cube_dim.num_elems() as usize;
    let w_cubes = if hw.has_planes {
        crate::plane::grid_cube_count(client, out_len.div_ceil(w_per_cube.max(1)))
    } else {
        1
    };
    let w_slots = (w_cubes as usize * w_per_cube).max(1);
    deriv34_weight_kernel::launch::<f64, R>(
        client,
        crate::plane::cube_count_1d(w_cubes),
        w_cube_dim,
        unsafe { ArrayArg::from_raw_parts(partial_h, n_pairs * total) },
        unsafe { ArrayArg::from_raw_parts(coeff_i_h, coeff_i.len()) },
        unsafe { ArrayArg::from_raw_parts(coeff_j_h, coeff_j.len()) },
        unsafe { ArrayArg::from_raw_parts(out_h.clone(), out_len) },
        n_prim_i as u32,
        n_prim_j as u32,
        n_ctr_i as u32,
        n_ctr_j as u32,
        total as u32,
        out_len as u32,
        w_slots as u32,
    );

    let raw = client.read_one_unchecked(out_h);
    f64::from_bytes(&raw)[0..out_len].to_vec()
}

/// Five-arm backend dispatch for [`run_deriv34_device`].
#[allow(clippy::too_many_arguments)]
fn run_deriv34_on_backend(
    backend: &ResolvedBackend,
    spec: &FamilySpec,
    li: u8,
    lj: u8,
    ri: [f64; 3],
    rj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    n_ctr_i: usize,
    n_ctr_j: usize,
    origins: &[([f64; 3], f64)],
) -> Vec<f64> {
    macro_rules! run {
        ($rt:ty, $client:expr) => {
            run_deriv34_device::<$rt>(
                $client, spec, li, lj, ri, rj, exps_i, exps_j, coeff_i, coeff_j, n_ctr_i, n_ctr_j,
                origins,
            )
        };
    }
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run!(cubecl::cpu::CpuRuntime, client),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run!(cubecl_wgpu::WgpuRuntime, client),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run!(cubecl_cuda::CudaRuntime, client),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run!(cubecl_hip::HipRuntime, client),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run!(cubecl_wgpu::WgpuRuntime, client),
    }
}

/// [`contract_deriv34_block`] on the device (§21).
///
/// `None` when `op_name` is not a deriv34 family, or when the shell pair needs a
/// Rys order past [`DERIV34_MAX_DEVICE_NROOTS`] — the caller then has the host
/// reference to fall back to, which is also where the order ceiling is enforced
/// for the whole family.
#[allow(clippy::too_many_arguments)]
pub fn contract_deriv34_block_device(
    backend: &ResolvedBackend,
    op_name: &str,
    li: u8,
    lj: u8,
    ri: [f64; 3],
    rj: [f64; 3],
    exps_i: &[f64],
    exps_j: &[f64],
    coeff_i: &[f64],
    coeff_j: &[f64],
    n_ctr_i: usize,
    n_ctr_j: usize,
    origins: &[([f64; 3], f64)],
) -> Option<Vec<f64>> {
    let spec = family_spec(op_name)?;
    if deriv34_geometry(&spec, li, lj).nroots > DERIV34_MAX_DEVICE_NROOTS {
        return None;
    }
    Some(run_deriv34_on_backend(
        backend, &spec, li, lj, ri, rj, exps_i, exps_j, coeff_i, coeff_j, n_ctr_i, n_ctr_j, origins,
    ))
}

#[cfg(test)]
#[cfg(feature = "cpu")]
mod device_tests {
    use super::*;

    fn cpu_backend() -> ResolvedBackend {
        ResolvedBackend::from_intent(&cintx_runtime::BackendIntent {
            backend: cintx_runtime::BackendKind::Cpu,
            ..Default::default()
        })
        .expect("cpu backend")
    }

    /// Every family the device kernel is expected to serve.
    ///
    /// The full `family_spec` list: the two X2C parents, their two derivative
    /// pairs, the two deriv3 (rank 27) families and the three deriv4 (rank 81)
    /// ones. One name per distinct [`FamilySpec`] — the `nuc`/`rinv` pairs share
    /// a spec and differ only in the origin list the caller builds.
    const FAMILIES: &[&str] = &[
        "pnucp",
        "ippnucp",
        "ippnucpip",
        "ipippnucp",
        "ipipipnuc",
        "ipipnucip",
        "ipipipiprinv",
        "ipiprinvipip",
        "ipipiprinvip",
    ];

    /// Device-vs-host cross-check on the CPU runtime (§21).
    ///
    /// The host evaluator is the reference these families were written against,
    /// so this is the gate that says the port is the same arithmetic: every
    /// family, several `(l_i, l_j)` pairs, a contracted shell (`nprim > 1`,
    /// `nctr > 1`, so the weighting kernel is exercised rather than bypassed),
    /// and two Coulomb centers with different charge factors.
    ///
    /// Tolerance is `atol = 1e-12 / rtol = 1e-10`, matching the grids
    /// device-vs-host checks. Orders one through five use the same polynomial
    /// solvers on both sides; six and up use `rys_roots_ext_dev` against
    /// `rys_roots_host`'s Wheeler path, which is where the tolerance rather than
    /// bit-identity earns its keep.
    #[test]
    fn device_matches_host_for_every_deriv34_family() {
        let backend = cpu_backend();
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.55_f64, -0.4, 0.7];
        // Two primitives and two contractions on each side: the weighting
        // kernel has four `(ip, jp)` pairs to sum and four output blocks.
        let exps_i = [1.7_f64, 0.42];
        let exps_j = [2.3_f64, 0.61];
        let coeff_i = [0.6_f64, -0.25, 0.31, 0.87];
        let coeff_j = [0.44_f64, 0.72, -0.19, 0.53];
        let origins = [
            ([0.31_f64, -0.22, 0.64], -3.0_f64),
            ([-0.8, 0.5, -0.3], 1.0),
        ];

        let mut checked = 0usize;
        let mut worst = 0.0_f64;
        for op in FAMILIES {
            // `(3, 3)` on a rank-81 family reaches Rys order six, which is the
            // `rys_roots_ext_dev` arm — the one whose absence is why these
            // families were routed to the host in the first place. Covering it
            // is the point of going this high.
            for &(li, lj) in &[
                (0u8, 0u8),
                (1, 0),
                (0, 1),
                (1, 1),
                (2, 1),
                (2, 2),
                (3, 2),
                (3, 3),
            ] {
                let host = contract_deriv34_block(
                    op, li, lj, ri, rj, &exps_i, &exps_j, &coeff_i, &coeff_j, 2, 2, &origins,
                )
                .expect("host family");
                let dev = contract_deriv34_block_device(
                    &backend, op, li, lj, ri, rj, &exps_i, &exps_j, &coeff_i, &coeff_j, 2, 2,
                    &origins,
                )
                .expect("device family");
                assert_eq!(host.len(), dev.len(), "{op} li={li} lj={lj}: length");
                let mut any_nonzero = false;
                for (idx, (&h, &d)) in host.iter().zip(dev.iter()).enumerate() {
                    if h.abs() > 1e-18 {
                        any_nonzero = true;
                    }
                    let diff = (h - d).abs();
                    worst = worst.max(diff);
                    assert!(
                        diff <= 1e-12 + 1e-10 * h.abs(),
                        "{op} li={li} lj={lj} idx={idx}: host={h:.17e} dev={d:.17e} \
                         diff={diff:.3e}"
                    );
                }
                assert!(
                    any_nonzero,
                    "{op} li={li} lj={lj}: host reference is all zero — the case proves nothing"
                );
                checked += 1;
            }
        }
        println!("deriv34 device-vs-host: {checked} (family, l) cases, worst |diff| = {worst:.3e}");
        assert!(checked >= FAMILIES.len() * 8);
    }

    /// The same cross-check on a real AMD GPU (§21).
    ///
    /// The CPU-runtime test above proves the *arithmetic*; this one proves the
    /// kernel survives HIP codegen, which is a separate question here — the
    /// 3c2e/2c2e vector VRR compiles fine on the CPU runtime and is rejected
    /// outright by HIP at an odd `Vector<f64, 3>` width (`__align__(24)`), so a
    /// kernel that has only ever been type-checked for ROCm has not been shown
    /// to run on it. `#[ignore]` + `CINTX_ROCM_ORACLE=1` gated, as the other
    /// GPU oracles here are.
    #[cfg(feature = "rocm")]
    #[test]
    #[ignore = "needs a ROCm device; run with CINTX_ROCM_ORACLE=1 --ignored"]
    fn device_matches_host_on_rocm() {
        assert_eq!(
            std::env::var("CINTX_ROCM_ORACLE").as_deref(),
            Ok("1"),
            "ROCm oracle must be invoked with CINTX_ROCM_ORACLE=1"
        );
        let backend = ResolvedBackend::from_intent(&cintx_runtime::BackendIntent {
            backend: cintx_runtime::BackendKind::Rocm,
            ..Default::default()
        })
        .expect("rocm backend");
        let ri = [0.0_f64, 0.0, 0.0];
        let rj = [0.55_f64, -0.4, 0.7];
        let exps_i = [1.7_f64, 0.42];
        let exps_j = [2.3_f64, 0.61];
        let coeff_i = [0.6_f64, -0.25, 0.31, 0.87];
        let coeff_j = [0.44_f64, 0.72, -0.19, 0.53];
        let origins = [
            ([0.31_f64, -0.22, 0.64], -3.0_f64),
            ([-0.8, 0.5, -0.3], 1.0),
        ];
        let mut worst = 0.0_f64;
        let mut checked = 0usize;
        for op in FAMILIES {
            for &(li, lj) in &[(0u8, 0u8), (1, 1), (2, 2), (3, 3)] {
                let host = contract_deriv34_block(
                    op, li, lj, ri, rj, &exps_i, &exps_j, &coeff_i, &coeff_j, 2, 2, &origins,
                )
                .expect("host family");
                let dev = contract_deriv34_block_device(
                    &backend, op, li, lj, ri, rj, &exps_i, &exps_j, &coeff_i, &coeff_j, 2, 2,
                    &origins,
                )
                .expect("device family");
                for (idx, (&h, &d)) in host.iter().zip(dev.iter()).enumerate() {
                    let diff = (h - d).abs();
                    worst = worst.max(diff);
                    assert!(
                        diff <= 1e-12 + 1e-10 * h.abs(),
                        "{op} li={li} lj={lj} idx={idx}: host={h:.17e} dev={d:.17e}"
                    );
                }
                checked += 1;
            }
        }
        println!("deriv34 rocm device-vs-host: {checked} cases, worst |diff| = {worst:.3e}");
    }

    /// The order ceiling is a refusal, not a wrong answer (§21).
    #[test]
    fn a_shell_pair_past_the_device_rys_ceiling_is_declined() {
        let backend = cpu_backend();
        // `nroots = nmax/2 + 1` with `nmax = (li + max_i_off + 1) + (lj + 1)`;
        // rank-81 `ipipipiprinv` carries `max_i_off = 3`, so `l_i = l_j = 12`
        // is far past twelve roots and must come back `None` rather than
        // indexing a solver that is not wired.
        let out = contract_deriv34_block_device(
            &backend,
            "ipipipiprinv",
            12,
            12,
            [0.0; 3],
            [0.5, 0.0, 0.0],
            &[1.0],
            &[1.0],
            &[1.0],
            &[1.0],
            1,
            1,
            &[([0.2, 0.1, 0.3], 1.0)],
        );
        assert!(out.is_none(), "past the wired ceiling must decline");
        assert!(
            contract_deriv34_block_device(
                &backend,
                "not_a_deriv34_family",
                0,
                0,
                [0.0; 3],
                [0.0; 3],
                &[1.0],
                &[1.0],
                &[1.0],
                &[1.0],
                1,
                1,
                &[([0.0; 3], 1.0)],
            )
            .is_none(),
            "an unknown operator must decline"
        );
    }
}
