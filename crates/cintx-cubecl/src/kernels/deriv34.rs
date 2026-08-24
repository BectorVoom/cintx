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
            let rt = aij2 * (1.0 - tau);
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
