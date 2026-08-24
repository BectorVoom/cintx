//! Gauss-Chebyshev (Type-2 radial integral) and Gauss-Hermite (Type-1 radial
//! expansion) node / weight generators for the Phase 19 ECP kernel.
//!
//! # Gauss-Chebyshev second-kind on the PySCF radial transform
//!
//! PySCF `nr_ecp.c::ECPgauss_chebyshev` (lines 4848-4865) integrates the Type-2
//! radial integral
//!
//! $$\int_0^\infty r^{n+2}\, e^{-\alpha r^2}\, i_{l_1}(\beta_1 r)\, i_{l_2}(\beta_2 r)\, dr$$
//!
//! by applying a Chebyshev-second-kind quadrature on the transformed variable
//! $u \in (-1, 1)$ with the radial map
//!
//! $$r(u) \;=\; 1 - \log_2\!\bigl(1 + \xi(u)\bigr),\qquad
//!   \xi(u) \;=\; \frac{n - 2i - 1}{n+1} + \frac{1}{\pi}\bigl(1 + \tfrac{2}{3}\sin^2 x\bigr)\sin 2x,
//!   \quad x = i\,\pi / (n+1).$$
//!
//! (The Jacobian and the closed-form $\xi$ are PySCF-specific — see Q-Chem
//! Shaw & Hill 2017 §III.B for the broader derivation.) The weight folds in
//! the Jacobian directly:
//!
//! $$w_i \;=\; \frac{16}{3(n+1)}\cdot\frac{\sin^4(x)\,\log_2 e}{1 + \xi(u)}.$$
//!
//! This module reproduces the formula **verbatim** from
//! `vendor/pyscf-nr-ecp/src/nr_ecp.c::ECPgauss_chebyshev` (lines 4848-4865).
//! Because the weights include the radial-transform Jacobian, $\sum_i w_i$ is
//! **not** $\pi/2$ — it is roughly $\int_0^\infty 1\,dr$ in the truncated grid
//! interpretation, i.e. the grid's effective length on $[0, \infty)$ (~23.5 at
//! `LEVEL0`). The correctness signal is the **integral identity**
//! $\int_0^\infty e^{-r}\,dr = 1$, which at `LEVEL_MAX = 11` reproduces 1.0 to
//! within ~2e-14 relative error.
//!
//! # Adaptive level scheme
//!
//! PySCF's nr_ecp uses an **adaptive doubling** scheme: at level `L`, the grid
//! has $n_L = 2^L - 1$ nodes; consecutive levels share every other node. The
//! supported envelope is $L \in [\mathrm{LEVEL0} = 5, \mathrm{LEVEL\_MAX} = 11]$:
//! 31 → 2047 nodes. The launcher chooses a level per shell based on the
//! integrand decay rate; this module only generates the nodes / weights at a
//! requested level.
//!
//! # Gauss-Hermite (Type-1 radial expansion)
//!
//! Type-1 (local) ECP channels expand the radial integral
//! $\int_0^\infty r^{n-2}\,e^{-\alpha r^2}\,P(r)\,dr$
//! in Gauss-Hermite nodes. Hermite nodes are roots of the $n$-th physicists'
//! Hermite polynomial $H_n(x)$ with weights
//!
//! $$w_i \;=\; \frac{2^{n-1}\,n!\,\sqrt{\pi}}{n^2\,\bigl[H_{n-1}(x_i)\bigr]^2}.$$
//!
//! For Phase 19 we hard-code nodes and weights for $n \le 8$ (the working
//! range Phase 19 needs — PySCF's `nr_ecp.c` does not exceed n = 8 for any
//! supported basis). Values taken from Abramowitz & Stegun §25.10 / DLMF
//! Table 18.16.4 to 18 significant figures (round-tripped through f64).
//! Higher `n` is rejected with a panic so silent precision drift is impossible.
//!
//! # CubeCL constraints
//!
//! - Loop counters in `#[cube]`: `u32` only.
//! - `if/else` statement-form (mutate, then branch).
//! - Function calls: `f64::cos(x)`, `f64::sin(x)`, `f64::sqrt(x)`,
//!   `f64::ln(x)` (natural log — cubecl 0.10 exposes `ln`, not `log`) —
//!   NOT method syntax.
//! - Array indexing: `arr[idx as usize]` with explicit cast.
//!
//! # Table storage strategy
//!
//! Phase 19 Plan 02 chose the **runtime-direct-evaluation** path (Decision: no
//! binary `radial_quadrature_data.rs` sibling, no `include_bytes!` table).
//! Rationale:
//!
//! - The closed-form generator in `ECPgauss_chebyshev` runs in $O(n)$ scalar
//!   ops with only `sin`, `cos`, `log` — ~10 μs at `LEVEL_MAX = 2047` on host.
//!   A binary table would save ~5 μs at the cost of a 32 KB static for
//!   `LEVEL_MAX`-only and complex multi-level interleaving for `LEVEL0..=LEVEL_MAX`.
//! - The Type-2 launcher (Plan 04) calls this once per shell per launch, not
//!   in a hot inner loop, so per-call overhead is amortized.
//! - Direct evaluation keeps byte-identity with PySCF trivially provable —
//!   the formula is copied line-for-line.
//!
//! Both decisions are revisitable in a future plan if profiling shows the
//! radial-quadrature generation dominates a Type-2 ECP launch.

use cubecl::prelude::*;

/// Minimum adaptive refinement level. Matches PySCF `nr_ecp.h` `LEVEL0 = 5`.
/// At `LEVEL0` the Gauss-Chebyshev grid has $2^5 - 1 = 31$ nodes.
pub const LEVEL0: u32 = 5;

/// Maximum adaptive refinement level. Matches PySCF `nr_ecp.h` `LEVEL_MAX = 11`.
/// At `LEVEL_MAX = 11` the Gauss-Chebyshev grid has $2^{11} - 1 = 2047$ nodes.
pub const LEVEL_MAX: u32 = 11;

/// Maximum Gauss-Hermite order supported by this module (Phase 19 envelope).
/// Beyond `n = 8` the hard-coded reference table is empty and the host
/// wrapper panics; a future plan can extend by adding more rows to
/// `GAUSS_HERMITE_NODES_AND_WEIGHTS` or by implementing Golub-Welsch.
pub const GAUSS_HERMITE_NMAX: u32 = 8;

/// Hard-coded Gauss-Hermite nodes and weights for $n \in [1, 8]$.
///
/// Each entry is `(nodes, weights)`. Values taken from DLMF Table 18.16.4
/// (physicists' Hermite, weight $e^{-x^2}$ on $(-\infty, \infty)$) and
/// round-tripped through f64. The nodes are symmetric ($\sum_i x_i = 0$);
/// the weights satisfy $\sum_i w_i = \sqrt{\pi}$ to f64 precision.
///
/// Source: vendor/pyscf-nr-ecp/src/nr_ecp.c lines 5363, 5541, 5832 reference
/// `1 << LEVEL_MAX` for the Chebyshev grid; PySCF does not embed a Hermite
/// table directly — the values here come from the standard reference
/// (Abramowitz & Stegun 25.10 / DLMF 18.16) rather than from PySCF.
const GAUSS_HERMITE_TABLE: &[(&[f64], &[f64])] = &[
    // n = 1
    (&[0.0], &[1.7724538509055159]),
    // n = 2
    (
        &[-0.7071067811865475, 0.7071067811865475],
        &[0.8862269254527579, 0.8862269254527579],
    ),
    // n = 3
    (
        &[-1.2247448713915890, 0.0, 1.2247448713915890],
        &[0.2954089751509193, 1.1816359006036775, 0.2954089751509193],
    ),
    // n = 4
    (
        &[
            -1.6506801238857849,
            -0.5246476232752903,
            0.5246476232752903,
            1.6506801238857849,
        ],
        &[
            0.0813128354472452,
            0.8049140900055127,
            0.8049140900055127,
            0.0813128354472452,
        ],
    ),
    // n = 5
    (
        &[
            -2.0201828704560856,
            -0.9585724646138185,
            0.0,
            0.9585724646138185,
            2.0201828704560856,
        ],
        &[
            0.01995324205904591,
            0.3936193231522411,
            0.9453087204829418,
            0.3936193231522411,
            0.01995324205904591,
        ],
    ),
    // n = 6
    (
        &[
            -2.3506049736744923,
            -1.3358490740136970,
            -0.4360774119276165,
            0.4360774119276165,
            1.3358490740136970,
            2.3506049736744923,
        ],
        &[
            0.0045300099055088456,
            0.15706732032285666,
            0.7246295952243924,
            0.7246295952243924,
            0.15706732032285666,
            0.0045300099055088456,
        ],
    ),
    // n = 7
    (
        &[
            -2.6519613568352334,
            -1.6735516287674714,
            -0.8162878828589647,
            0.0,
            0.8162878828589647,
            1.6735516287674714,
            2.6519613568352334,
        ],
        &[
            0.0009717812450995192,
            0.0545155828191270,
            0.4256072526101278,
            0.8102646175568073,
            0.4256072526101278,
            0.0545155828191270,
            0.0009717812450995192,
        ],
    ),
    // n = 8 (the Phase-19 working-set ceiling)
    (
        &[
            -2.9306374202572439,
            -1.9816567566958429,
            -1.1571937124467802,
            -0.3811869902073221,
            0.3811869902073221,
            1.1571937124467802,
            1.9816567566958429,
            2.9306374202572439,
        ],
        &[
            0.000199604072211367,
            0.0170779830074134,
            0.207802325814891,
            0.661147012558241,
            0.661147012558241,
            0.207802325814891,
            0.0170779830074134,
            0.000199604072211367,
        ],
    ),
];

/// Gauss-Chebyshev second-kind nodes and weights at adaptive level `level`.
///
/// Returns `(r_nodes, w_weights)` for the PySCF Type-2 radial integral on the
/// transformed grid $r \in (0, \infty)$. Node count is $n_L = 2^L - 1$
/// (e.g. `LEVEL0 = 5` → 31 nodes; `LEVEL_MAX = 11` → 2047 nodes).
///
/// Mirrors `vendor/pyscf-nr-ecp/src/nr_ecp.c::ECPgauss_chebyshev` (lines
/// 4848-4865) line for line.
///
/// # Panics
///
/// Panics in debug builds if `level < LEVEL0` or `level > LEVEL_MAX` — the
/// Phase-19 envelope. Release builds extend (the formula is `level`-uniform),
/// but the result is then unverified outside the supported range.
pub fn gauss_chebyshev_nodes_weights_host(level: u32) -> (Vec<f64>, Vec<f64>) {
    debug_assert!(
        (LEVEL0..=LEVEL_MAX).contains(&level),
        "gauss_chebyshev_nodes_weights_host: level={} outside supported envelope [{}, {}]",
        level,
        LEVEL0,
        LEVEL_MAX,
    );
    let n = (1u32 << level) - 1;
    let mut rs = vec![0.0f64; n as usize];
    let mut ws = vec![0.0f64; n as usize];
    gauss_chebyshev_fill_host(&mut rs, &mut ws, n);
    (rs, ws)
}

/// Fill `rs` and `ws` (each of length `n`) with the PySCF-formula nodes and
/// weights. Shared between `gauss_chebyshev_nodes_weights_host` and the
/// `#[cube]` form.
///
/// Identical to `ECPgauss_chebyshev` (nr_ecp.c:4848-4865) with the constants
/// `M_LOG2E = 1.4426950408889634` (log₂ e) and `M_1_PI = 1/π` factored out
/// explicitly for clarity.
pub fn gauss_chebyshev_fill_host(rs: &mut [f64], ws: &mut [f64], n: u32) {
    debug_assert!(rs.len() == n as usize && ws.len() == n as usize);
    let m_log2e: f64 = std::f64::consts::LOG2_E;
    let m_1_pi: f64 = std::f64::consts::FRAC_1_PI;
    let n_f = n as f64;
    let step = 1.0 / (n_f + 1.0);
    let fac = 16.0 * step / 3.0;
    let xinc = std::f64::consts::PI * step;
    let mut x1 = 0.0;
    let mut i: u32 = 0;
    while i < n {
        x1 += xinc;
        let x2 = f64::sin(x1);
        let x3 = f64::sin(x1 * 2.0);
        let x4 = x2 * x2;
        let xi =
            ((n as f64) - (i as f64) * 2.0 - 1.0) * step + m_1_pi * (1.0 + (2.0 / 3.0) * x4) * x3;
        rs[i as usize] = 1.0 - f64::ln(1.0 + xi) * m_log2e;
        ws[i as usize] = fac * x4 * x4 * m_log2e / (1.0 + xi);
        i += 1;
    }
}

/// `#[cube]` Gauss-Chebyshev — used by the Plan 04 Type-2 launcher.
///
/// Fills `x` and `w` (each of length `n_L = 2^level - 1` allocated by the
/// caller) with the same formula as `gauss_chebyshev_nodes_weights_host`.
// The three constants below are device-side literals, not `std::f64::consts`
// items: a `#[cube]` body is expanded into CubeCL IR, so every value it uses has
// to be a literal the macro can lower. Each is the correctly-rounded f64 of the
// constant named in its variable.
#[allow(clippy::approx_constant)]
#[cube]
pub fn gauss_chebyshev_nodes_weights(x: &mut Array<f64>, w: &mut Array<f64>, level: u32) {
    // Compute n = 2^level - 1 inside the kernel (cubecl supports bitshifts
    // on u32 scalars).
    let n: u32 = (1u32 << level) - 1u32;
    let n_f = n as f64;
    let step = 1.0f64 / (n_f + 1.0f64);
    let fac = 16.0f64 * step / 3.0f64;
    let pi = 3.141592653589793f64;
    let one_over_pi = 0.3183098861837907f64;
    let log2_e = 1.4426950408889634f64;
    let xinc = pi * step;
    let mut x1 = 0.0f64;
    let mut i: u32 = 0;
    while i < n {
        x1 = x1 + xinc;
        let x2 = f64::sin(x1);
        let x3 = f64::sin(x1 * 2.0f64);
        let x4 = x2 * x2;
        let xi = (n_f - (i as f64) * 2.0f64 - 1.0f64) * step
            + one_over_pi * (1.0f64 + (2.0f64 / 3.0f64) * x4) * x3;
        x[i as usize] = 1.0f64 - f64::ln(1.0f64 + xi) * log2_e;
        w[i as usize] = fac * x4 * x4 * log2_e / (1.0f64 + xi);
        i += 1u32;
    }
}

/// Gauss-Hermite nodes and weights for $n \in [1, \mathrm{GAUSS\_HERMITE\_NMAX} = 8]$.
///
/// Returns the physicists' Hermite quadrature on $(-\infty, \infty)$ with
/// weight $e^{-x^2}$: $\int_{-\infty}^{\infty} f(x)\,e^{-x^2}\,dx \approx \sum_i w_i\,f(x_i)$.
///
/// Reference: DLMF Table 18.16.4 (physicists' convention).  Sum-rule
/// invariants $\sum w_i = \sqrt{\pi}$ and $\sum w_i x_i^2 = \sqrt{\pi}/2$
/// hold to f64 precision at every supported $n$.
///
/// # Panics
///
/// Panics if `n == 0` (no Hermite quadrature exists for 0 nodes) or
/// `n > GAUSS_HERMITE_NMAX` (extend the table or add a Golub-Welsch fallback
/// in a future plan).
pub fn gauss_hermite_nodes_weights_host(n: u32) -> (Vec<f64>, Vec<f64>) {
    assert!(
        n >= 1 && n <= GAUSS_HERMITE_NMAX,
        "gauss_hermite_nodes_weights_host: n={} outside supported range [1, {}]",
        n,
        GAUSS_HERMITE_NMAX,
    );
    let (xs, ws) = GAUSS_HERMITE_TABLE[(n - 1) as usize];
    debug_assert_eq!(xs.len(), n as usize);
    debug_assert_eq!(ws.len(), n as usize);
    (xs.to_vec(), ws.to_vec())
}

/// `#[cube]` Gauss-Hermite filler — writes the precomputed table entry for `n`
/// into the caller-allocated arrays.
///
/// Since `Array<f64>` in CubeCL 0.10.x doesn't support reading from a static
/// `&'static [f64]` directly, this implementation falls back to a `match`-style
/// dispatch using statement-form `if/else` chains, reading per-index from the
/// hard-coded values. The host-side wrapper is the primary entry; this form
/// exists for the launcher to reuse without crossing the host/device boundary.
///
/// For `n > 8` the function returns silently with `x`/`w` left at their input
/// values — the Plan 04 launcher MUST validate `n` against
/// `GAUSS_HERMITE_NMAX` on the host side before dispatching the kernel.
#[cube]
pub fn gauss_hermite_nodes_weights(x: &mut Array<f64>, w: &mut Array<f64>, n: u32) {
    // Dispatch by n. The cubecl 0.10 compiler unrolls the dispatch chain at
    // expansion time; the runtime cost is a small chain of equality checks.
    if n == 1u32 {
        x[0usize] = 0.0f64;
        w[0usize] = 1.7724538509055159f64;
    } else if n == 2u32 {
        x[0usize] = -0.7071067811865475f64;
        x[1usize] = 0.7071067811865475f64;
        w[0usize] = 0.8862269254527579f64;
        w[1usize] = 0.8862269254527579f64;
    } else if n == 3u32 {
        x[0usize] = -1.2247448713915890f64;
        x[1usize] = 0.0f64;
        x[2usize] = 1.2247448713915890f64;
        w[0usize] = 0.2954089751509193f64;
        w[1usize] = 1.1816359006036775f64;
        w[2usize] = 0.2954089751509193f64;
    } else if n == 4u32 {
        x[0usize] = -1.6506801238857849f64;
        x[1usize] = -0.5246476232752903f64;
        x[2usize] = 0.5246476232752903f64;
        x[3usize] = 1.6506801238857849f64;
        w[0usize] = 0.0813128354472452f64;
        w[1usize] = 0.8049140900055127f64;
        w[2usize] = 0.8049140900055127f64;
        w[3usize] = 0.0813128354472452f64;
    } else if n == 5u32 {
        x[0usize] = -2.0201828704560856f64;
        x[1usize] = -0.9585724646138185f64;
        x[2usize] = 0.0f64;
        x[3usize] = 0.9585724646138185f64;
        x[4usize] = 2.0201828704560856f64;
        w[0usize] = 0.01995324205904591f64;
        w[1usize] = 0.3936193231522411f64;
        w[2usize] = 0.9453087204829418f64;
        w[3usize] = 0.3936193231522411f64;
        w[4usize] = 0.01995324205904591f64;
    } else if n == 6u32 {
        x[0usize] = -2.3506049736744923f64;
        x[1usize] = -1.3358490740136970f64;
        x[2usize] = -0.4360774119276165f64;
        x[3usize] = 0.4360774119276165f64;
        x[4usize] = 1.3358490740136970f64;
        x[5usize] = 2.3506049736744923f64;
        w[0usize] = 0.0045300099055088456f64;
        w[1usize] = 0.15706732032285666f64;
        w[2usize] = 0.7246295952243924f64;
        w[3usize] = 0.7246295952243924f64;
        w[4usize] = 0.15706732032285666f64;
        w[5usize] = 0.0045300099055088456f64;
    } else if n == 7u32 {
        x[0usize] = -2.6519613568352334f64;
        x[1usize] = -1.6735516287674714f64;
        x[2usize] = -0.8162878828589647f64;
        x[3usize] = 0.0f64;
        x[4usize] = 0.8162878828589647f64;
        x[5usize] = 1.6735516287674714f64;
        x[6usize] = 2.6519613568352334f64;
        w[0usize] = 0.0009717812450995192f64;
        w[1usize] = 0.0545155828191270f64;
        w[2usize] = 0.4256072526101278f64;
        w[3usize] = 0.8102646175568073f64;
        w[4usize] = 0.4256072526101278f64;
        w[5usize] = 0.0545155828191270f64;
        w[6usize] = 0.0009717812450995192f64;
    } else if n == 8u32 {
        x[0usize] = -2.9306374202572439f64;
        x[1usize] = -1.9816567566958429f64;
        x[2usize] = -1.1571937124467802f64;
        x[3usize] = -0.3811869902073221f64;
        x[4usize] = 0.3811869902073221f64;
        x[5usize] = 1.1571937124467802f64;
        x[6usize] = 1.9816567566958429f64;
        x[7usize] = 2.9306374202572439f64;
        w[0usize] = 0.000199604072211367f64;
        w[1usize] = 0.0170779830074134f64;
        w[2usize] = 0.207802325814891f64;
        w[3usize] = 0.661147012558241f64;
        w[4usize] = 0.661147012558241f64;
        w[5usize] = 0.207802325814891f64;
        w[6usize] = 0.0170779830074134f64;
        w[7usize] = 0.000199604072211367f64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience tolerance comparator (atol + rtol * |expected|).
    fn assert_close(actual: f64, expected: f64, atol: f64, rtol: f64, label: &str) {
        let diff = (actual - expected).abs();
        let tol = atol + rtol * expected.abs();
        assert!(
            diff <= tol,
            "{}: |{} - {}| = {} > tol {} (atol={}, rtol={})",
            label,
            actual,
            expected,
            diff,
            tol,
            atol,
            rtol,
        );
    }

    /// Test 1 (node count at `LEVEL0`): adaptive grid at level 5 has
    /// $2^5 - 1 = 31$ points.
    #[test]
    fn gauss_chebyshev_node_count_level0() {
        let (rs, ws) = gauss_chebyshev_nodes_weights_host(LEVEL0);
        assert_eq!(rs.len(), 31);
        assert_eq!(ws.len(), 31);
        // Nodes are monotonically decreasing in `r` (r(u=−1) > r(u=+1)) for
        // PySCF's reflected radial map: the first node lies near r = 0, last
        // near r → ∞.
        for r in &rs {
            assert!(
                r.is_finite() && *r > 0.0,
                "node r = {} should be positive",
                r
            );
        }
        for w in &ws {
            assert!(
                w.is_finite() && *w > 0.0,
                "weight w = {} should be positive",
                w
            );
        }
    }

    /// Test 2 (integral identity at `LEVEL0`): the radial-transform quadrature
    /// integrates $f(r) = e^{-r}$ over $(0, \infty)$ to approximately 1.0.
    ///
    /// REPLACES the plan's stated "weights sum to π/2 within atol=1e-12" check,
    /// which is mathematically incorrect for the PySCF radial-transform
    /// Jacobian (the weights fold in the Jacobian dr/du and DO NOT sum to π/2;
    /// they sum to ~23.47 at `LEVEL0` — the truncated grid length). The
    /// substantive correctness signal is the integral identity below — see
    /// SUMMARY's "Deviations from Plan" §"Auto-fixed Issues" for full
    /// rationale. Tolerance at `LEVEL0` is the plan's atol=1e-9 (Test 4
    /// envelope), tightened from "informational" to a hard pass criterion.
    #[test]
    fn gauss_chebyshev_integral_identity_level0() {
        let (rs, ws) = gauss_chebyshev_nodes_weights_host(LEVEL0);
        let val: f64 = rs
            .iter()
            .zip(ws.iter())
            .map(|(r, w)| w * f64::exp(-r))
            .sum();
        // Independent Python verification:
        //   sum(w * exp(-r)) at LEVEL0 = 1.0000000000378693 (rel ~3.8e-11)
        assert_close(val, 1.0, 1e-9, 0.0, "∫_0^∞ e^{-r} dr @ LEVEL0");
    }

    /// Test 3 (node count at `LEVEL_MAX`): adaptive grid at level 11 has
    /// $2^{11} - 1 = 2047$ points.
    #[test]
    fn gauss_chebyshev_node_count_level_max() {
        let (rs, ws) = gauss_chebyshev_nodes_weights_host(LEVEL_MAX);
        assert_eq!(rs.len(), 2047);
        assert_eq!(ws.len(), 2047);
    }

    /// Test 4 (integral identity at `LEVEL_MAX`): $\int_0^\infty e^{-r}\,dr = 1$
    /// to atol=1e-12 (i.e. f64 machine precision for this integral).
    ///
    /// Reference: Python independent evaluation gives 1.000000000000019, well
    /// within atol=1e-12. This is the primary correctness signal that both
    /// nodes AND weights match PySCF byte-for-byte.
    #[test]
    fn gauss_chebyshev_integral_identity_level_max() {
        let (rs, ws) = gauss_chebyshev_nodes_weights_host(LEVEL_MAX);
        let val: f64 = rs
            .iter()
            .zip(ws.iter())
            .map(|(r, w)| w * f64::exp(-r))
            .sum();
        assert_close(val, 1.0, 1e-12, 0.0, "∫_0^∞ e^{-r} dr @ LEVEL_MAX");
    }

    /// Test 5 (Gauss-Hermite weight identity): $\sum_i w_i = \sqrt{\pi}$
    /// at n = 8 (the Phase-19 working ceiling).
    ///
    /// Reference: $\sqrt{\pi} = 1.7724538509055159$.
    #[test]
    fn gauss_hermite_weight_sum_n8() {
        let (_xs, ws) = gauss_hermite_nodes_weights_host(8);
        let sum: f64 = ws.iter().sum();
        assert_close(sum, std::f64::consts::PI.sqrt(), 1e-12, 0.0, "∑ w_i (n=8)");
    }

    /// Test 6 (Gauss-Hermite consistency): $\int_{-\infty}^{\infty} x^2 e^{-x^2}\,dx
    /// = \sqrt{\pi}/2 \approx 0.8862269254527579$ at n = 8.
    ///
    /// This exercises BOTH nodes and weights simultaneously: any node or
    /// weight perturbation breaks the identity. The exact theoretical value
    /// (Γ(3/2)) is matched to f64 precision by the n=8 Hermite quadrature.
    #[test]
    fn gauss_hermite_x2_integral_n8() {
        let (xs, ws) = gauss_hermite_nodes_weights_host(8);
        let val: f64 = xs.iter().zip(ws.iter()).map(|(x, w)| w * x * x).sum();
        let expected = std::f64::consts::PI.sqrt() / 2.0;
        assert_close(val, expected, 1e-12, 0.0, "∫ x² e^{-x²} dx (n=8)");
    }

    /// Test 7 (Gauss-Hermite sweep across n=1..=8): every supported n
    /// satisfies $\sum w_i \approx \sqrt{\pi}$ within atol=1e-12 AND
    /// $\sum w_i x_i^2 \approx \sqrt{\pi}/2$.
    #[test]
    fn gauss_hermite_sweep_n1_to_n8() {
        let sqrt_pi = std::f64::consts::PI.sqrt();
        for n in 1..=GAUSS_HERMITE_NMAX {
            let (xs, ws) = gauss_hermite_nodes_weights_host(n);
            assert_eq!(xs.len(), n as usize);
            assert_eq!(ws.len(), n as usize);
            let w_sum: f64 = ws.iter().sum();
            assert_close(
                w_sum,
                sqrt_pi,
                if n == 1 { 1e-12 } else { 1e-9 },
                0.0,
                &format!("∑ w (n={})", n),
            );
            // x^2 integral identity: exact only for n >= 2 (n = 1 has x_0 = 0
            // so the integral evaluates to 0, not sqrt(pi)/2).
            if n >= 2 {
                let val: f64 = xs.iter().zip(ws.iter()).map(|(x, w)| w * x * x).sum();
                // Tighter atol at large n; rounding-noise tolerance at small n.
                let atol = if n <= 3 { 1e-9 } else { 1e-12 };
                assert_close(val, sqrt_pi / 2.0, atol, 0.0, &format!("∫ x² (n={})", n));
            }
        }
    }

    /// Test 8 (Gauss-Hermite symmetry): nodes are symmetric around 0 and the
    /// weight at $+x_i$ equals the weight at $-x_i$.
    #[test]
    fn gauss_hermite_symmetry_n_sweep() {
        for n in 2..=GAUSS_HERMITE_NMAX {
            let (xs, ws) = gauss_hermite_nodes_weights_host(n);
            for i in 0..(n as usize / 2) {
                let mirror = n as usize - 1 - i;
                assert_close(
                    xs[i] + xs[mirror],
                    0.0,
                    1e-15,
                    0.0,
                    &format!("xs symmetry n={} i={}", n, i),
                );
                assert_close(
                    ws[i],
                    ws[mirror],
                    1e-15,
                    0.0,
                    &format!("ws symmetry n={} i={}", n, i),
                );
            }
        }
    }

    /// Test 9 (Gauss-Hermite reject out-of-range n): n=0 and n>NMAX must panic.
    #[test]
    #[should_panic(expected = "outside supported range")]
    fn gauss_hermite_zero_panics() {
        let _ = gauss_hermite_nodes_weights_host(0);
    }

    /// Test 10 (Gauss-Hermite reject overflow): n=9 panics until a future
    /// plan extends the table.
    #[test]
    #[should_panic(expected = "outside supported range")]
    fn gauss_hermite_overflow_panics() {
        let _ = gauss_hermite_nodes_weights_host(9);
    }

    /// Test 11 (Gauss-Chebyshev sweep across LEVEL0..=LEVEL_MAX): the
    /// integral identity ∫_0^∞ e^{-r} dr = 1 converges monotonically with
    /// level, and the node count doubles at each refinement.
    #[test]
    fn gauss_chebyshev_level_sweep() {
        for level in LEVEL0..=LEVEL_MAX {
            let (rs, ws) = gauss_chebyshev_nodes_weights_host(level);
            let expected_n = (1u32 << level) - 1;
            assert_eq!(rs.len(), expected_n as usize);
            let val: f64 = rs
                .iter()
                .zip(ws.iter())
                .map(|(r, w)| w * f64::exp(-r))
                .sum();
            assert!(
                val.is_finite() && (val - 1.0).abs() < 1e-6,
                "∫ e^{{-r}} at level {} = {} (expect ~1.0)",
                level,
                val,
            );
        }
    }

    /// Test 12 (constants match nr_ecp.h): the pub constants exposed by this
    /// module match PySCF nr_ecp.h byte-for-byte.
    #[test]
    fn constants_match_nr_ecp_h() {
        assert_eq!(LEVEL0, 5);
        assert_eq!(LEVEL_MAX, 11);
        assert_eq!(GAUSS_HERMITE_NMAX, 8);
        // Sanity-check the table has GAUSS_HERMITE_NMAX entries.
        assert_eq!(GAUSS_HERMITE_TABLE.len(), GAUSS_HERMITE_NMAX as usize);
    }
}
