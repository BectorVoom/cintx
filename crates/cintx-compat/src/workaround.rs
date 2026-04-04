//! Workaround paths for integral families with alternative computation strategies.
//!
//! These produce results equivalent to the direct kernel at oracle-verified tolerance
//! but use a different algorithmic path (e.g., tracing a higher-center integral).

#![allow(non_snake_case)]

use crate::helpers::{CINTcgto_cart, CINTcgto_spheric};
use crate::raw::{self, RawApiId, RawEvalSummary};
use cintx_core::cintxRsError;

/// Computes 4c1e-equivalent results by evaluating int2e_sph/cart and tracing
/// over the (k,l) auxiliary shell pair when k==l.
///
/// The 4c1e integral (i,j|kk) equals the diagonal trace of the 2e ERI:
/// out[i*nj + j] = sum_m eri[((i*nj + j)*nk + m)*nl + m]
///
/// This is valid for all cart/sph representations within the Validated4C1E envelope.
///
/// # Safety
/// Caller must provide valid atm/bas/env arrays with correct libcint layout.
pub fn int4c1e_via_2e_trace(
    out: &mut [f64],
    shls: &[i32],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    use_sph: bool,
) -> Result<RawEvalSummary, cintxRsError> {
    if shls.len() < 4 {
        return Err(cintxRsError::UnsupportedApi {
            requested: "int4c1e_via_2e_trace requires 4 shell indices".into(),
        });
    }

    let (i_sh, j_sh, k_sh, l_sh) = (shls[0], shls[1], shls[2], shls[3]);

    // Compute component counts for each shell
    let (ni, nj, nk, nl) = if use_sph {
        (
            CINTcgto_spheric(i_sh, bas)?,
            CINTcgto_spheric(j_sh, bas)?,
            CINTcgto_spheric(k_sh, bas)?,
            CINTcgto_spheric(l_sh, bas)?,
        )
    } else {
        (
            CINTcgto_cart(i_sh, bas)?,
            CINTcgto_cart(j_sh, bas)?,
            CINTcgto_cart(k_sh, bas)?,
            CINTcgto_cart(l_sh, bas)?,
        )
    };

    // Allocate 2e buffer
    let eri_size = ni * nj * nk * nl;
    let mut eri_buf = vec![0.0f64; eri_size];

    // Call eval_raw with the 2e operator
    let api_id = if use_sph {
        RawApiId::INT2E_SPH
    } else {
        RawApiId::INT2E_CART
    };
    let summary = unsafe {
        raw::eval_raw(
            api_id,
            Some(&mut eri_buf),
            None,
            shls,
            atm,
            bas,
            env,
            None,
            None,
        )?
    };

    // Trace contraction: out[i*nj + j] = sum_m eri[((i*nj + j)*nk + m)*nl + m]
    // The trace is over the k,l diagonal where k==l shell
    let out_size = ni * nj;
    if out.len() < out_size {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!(
                "output buffer too small for 4c1e trace: need {out_size}, got {}",
                out.len()
            ),
        });
    }

    // Zero output first
    for v in out[..out_size].iter_mut() {
        *v = 0.0;
    }

    // Trace: for each (i,j) output element, sum eri[i,j,m,m] over m
    let trace_dim = nk.min(nl); // k==l shell, so nk==nl normally
    for i in 0..ni {
        for j in 0..nj {
            let mut sum = 0.0;
            for m in 0..trace_dim {
                // eri layout: eri[((i * nj + j) * nk + m) * nl + m]
                let idx = ((i * nj + j) * nk + m) * nl + m;
                sum += eri_buf[idx];
            }
            out[i * nj + j] = sum;
        }
    }

    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int4c1e_via_2e_trace_rejects_insufficient_shls() {
        let atm = vec![0_i32; 6]; // 1 atom, 6 slots
        let bas = vec![0_i32; 8]; // 1 shell, 8 slots
        let env = vec![0.0_f64; 20];
        let shls = [0_i32, 1_i32, 2_i32]; // only 3 shells — should fail
        let mut out = vec![0.0_f64; 4];
        let err = int4c1e_via_2e_trace(&mut out, &shls, &atm, &bas, &env, true).unwrap_err();
        assert!(
            matches!(err, cintxRsError::UnsupportedApi { .. }),
            "expected UnsupportedApi, got {err:?}"
        );
    }
}
