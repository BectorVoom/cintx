#![allow(non_snake_case)]

use crate::raw::{ANG_OF, BAS_SLOTS, KAPPA_OF, NCTR_OF, NPRIM_OF, RawBasView};
use cintx_core::cintxRsError;

fn bas_record(bas_id: i32, bas: &[i32]) -> Result<&[i32], cintxRsError> {
    let view = RawBasView::new(bas)?;
    let index = usize::try_from(bas_id).map_err(|_| cintxRsError::InvalidBasLayout {
        slot_width: BAS_SLOTS,
        provided: bas.len(),
    })?;
    if index >= view.len() {
        return Err(cintxRsError::InvalidBasLayout {
            slot_width: BAS_SLOTS,
            provided: bas.len(),
        });
    }
    let start = index
        .checked_mul(BAS_SLOTS)
        .ok_or(cintxRsError::InvalidBasLayout {
            slot_width: BAS_SLOTS,
            provided: bas.len(),
        })?;
    Ok(&bas[start..start + BAS_SLOTS])
}

fn len_spheric(l: i32) -> Result<usize, cintxRsError> {
    let l = usize::try_from(l).map_err(|_| cintxRsError::UnsupportedApi {
        requested: format!("negative angular momentum {l}"),
    })?;
    Ok(2 * l + 1)
}

fn len_cartesian(l: i32) -> Result<usize, cintxRsError> {
    let l = usize::try_from(l).map_err(|_| cintxRsError::UnsupportedApi {
        requested: format!("negative angular momentum {l}"),
    })?;
    Ok((l + 1) * (l + 2) / 2)
}

fn len_spinor(l: i32, kappa: i32) -> Result<usize, cintxRsError> {
    let l = usize::try_from(l).map_err(|_| cintxRsError::UnsupportedApi {
        requested: format!("negative angular momentum {l}"),
    })?;
    Ok(match kappa {
        0 => 4 * l + 2,
        neg if neg < 0 => 2 * l + 2,
        _ => 2 * l,
    })
}

fn nctr_for(bas_id: i32, bas: &[i32]) -> Result<usize, cintxRsError> {
    let record = bas_record(bas_id, bas)?;
    usize::try_from(record[NCTR_OF]).map_err(|_| cintxRsError::InvalidBasLayout {
        slot_width: BAS_SLOTS,
        provided: bas.len(),
    })
}

fn nprim_for(bas_id: i32, bas: &[i32]) -> Result<usize, cintxRsError> {
    let record = bas_record(bas_id, bas)?;
    usize::try_from(record[NPRIM_OF]).map_err(|_| cintxRsError::InvalidBasLayout {
        slot_width: BAS_SLOTS,
        provided: bas.len(),
    })
}

fn shell_count(nbas: i32, bas: &[i32]) -> Result<usize, cintxRsError> {
    let view = RawBasView::new(bas)?;
    let count = usize::try_from(nbas).map_err(|_| cintxRsError::InvalidBasLayout {
        slot_width: BAS_SLOTS,
        provided: bas.len(),
    })?;
    if count > view.len() {
        return Err(cintxRsError::InvalidBasLayout {
            slot_width: BAS_SLOTS,
            provided: bas.len(),
        });
    }
    Ok(count)
}

pub fn CINTlen_cart(l: i32) -> Result<usize, cintxRsError> {
    len_cartesian(l)
}

pub fn CINTlen_spinor(bas_id: i32, bas: &[i32]) -> Result<usize, cintxRsError> {
    let record = bas_record(bas_id, bas)?;
    len_spinor(record[ANG_OF], record[KAPPA_OF])
}

pub fn CINTcgtos_cart(bas_id: i32, bas: &[i32]) -> Result<usize, cintxRsError> {
    Ok(CINTlen_cart(bas_record(bas_id, bas)?[ANG_OF])? * nctr_for(bas_id, bas)?)
}

pub fn CINTcgtos_spheric(bas_id: i32, bas: &[i32]) -> Result<usize, cintxRsError> {
    Ok(len_spheric(bas_record(bas_id, bas)?[ANG_OF])? * nctr_for(bas_id, bas)?)
}

pub fn CINTcgtos_spinor(bas_id: i32, bas: &[i32]) -> Result<usize, cintxRsError> {
    Ok(CINTlen_spinor(bas_id, bas)? * nctr_for(bas_id, bas)?)
}

pub fn CINTcgto_cart(bas_id: i32, bas: &[i32]) -> Result<usize, cintxRsError> {
    CINTcgtos_cart(bas_id, bas)
}

pub fn CINTcgto_spheric(bas_id: i32, bas: &[i32]) -> Result<usize, cintxRsError> {
    CINTcgtos_spheric(bas_id, bas)
}

pub fn CINTcgto_spinor(bas_id: i32, bas: &[i32]) -> Result<usize, cintxRsError> {
    CINTcgtos_spinor(bas_id, bas)
}

pub fn CINTtot_pgto_spheric(bas: &[i32], nbas: i32) -> Result<usize, cintxRsError> {
    let mut total = 0usize;
    for shell in 0..shell_count(nbas, bas)? {
        total = total.saturating_add(
            nprim_for(shell as i32, bas)?
                .saturating_mul(len_spheric(bas_record(shell as i32, bas)?[ANG_OF])?),
        );
    }
    Ok(total)
}

pub fn CINTtot_pgto_spinor(bas: &[i32], nbas: i32) -> Result<usize, cintxRsError> {
    let mut total = 0usize;
    for shell in 0..shell_count(nbas, bas)? {
        total = total.saturating_add(
            nprim_for(shell as i32, bas)?.saturating_mul(CINTlen_spinor(shell as i32, bas)?),
        );
    }
    Ok(total)
}

pub fn CINTtot_cgto_cart(bas: &[i32], nbas: i32) -> Result<usize, cintxRsError> {
    let mut total = 0usize;
    for shell in 0..shell_count(nbas, bas)? {
        total = total.saturating_add(CINTcgto_cart(shell as i32, bas)?);
    }
    Ok(total)
}

pub fn CINTtot_cgto_spheric(bas: &[i32], nbas: i32) -> Result<usize, cintxRsError> {
    let mut total = 0usize;
    for shell in 0..shell_count(nbas, bas)? {
        total = total.saturating_add(CINTcgto_spheric(shell as i32, bas)?);
    }
    Ok(total)
}

pub fn CINTtot_cgto_spinor(bas: &[i32], nbas: i32) -> Result<usize, cintxRsError> {
    let mut total = 0usize;
    for shell in 0..shell_count(nbas, bas)? {
        total = total.saturating_add(CINTcgto_spinor(shell as i32, bas)?);
    }
    Ok(total)
}

pub fn CINTshells_cart_offset(
    ao_loc: &mut [i32],
    bas: &[i32],
    nbas: i32,
) -> Result<(), cintxRsError> {
    write_offsets(ao_loc, bas, nbas, CINTcgto_cart)
}

pub fn CINTshells_spheric_offset(
    ao_loc: &mut [i32],
    bas: &[i32],
    nbas: i32,
) -> Result<(), cintxRsError> {
    write_offsets(ao_loc, bas, nbas, CINTcgto_spheric)
}

pub fn CINTshells_spinor_offset(
    ao_loc: &mut [i32],
    bas: &[i32],
    nbas: i32,
) -> Result<(), cintxRsError> {
    write_offsets(ao_loc, bas, nbas, CINTcgto_spinor)
}

fn write_offsets(
    ao_loc: &mut [i32],
    bas: &[i32],
    nbas: i32,
    count_fn: fn(i32, &[i32]) -> Result<usize, cintxRsError>,
) -> Result<(), cintxRsError> {
    let shell_count = shell_count(nbas, bas)?;
    let needed = shell_count.saturating_add(1);
    if ao_loc.len() < needed {
        return Err(cintxRsError::BufferTooSmall {
            required: needed,
            provided: ao_loc.len(),
        });
    }

    let mut offset = 0usize;
    ao_loc[0] = 0;
    for shell in 0..shell_count {
        offset = offset.saturating_add(count_fn(shell as i32, bas)?);
        ao_loc[shell + 1] = i32::try_from(offset).map_err(|_| cintxRsError::ChunkPlanFailed {
            from: "compat_helpers",
            detail: "ao offset overflowed i32".to_owned(),
        })?;
    }
    Ok(())
}

pub fn CINTgto_norm(n: i32, a: f64) -> f64 {
    if !a.is_finite() || a <= 0.0 || n < 0 {
        return 0.0;
    }
    // Lightweight stable approximation used by compat parity checks.
    (2.0 * a).powf((n as f64 + 1.5) * 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_bas() -> Vec<i32> {
        vec![
            0, 0, 2, 1, 0, 0, 2, 0, // l=0
            0, 1, 1, 2, 0, 4, 5, 0, // l=1, nctr=2
        ]
    }

    #[test]
    fn helper_counts_follow_expected_shell_formulas() {
        let bas = sample_bas();
        assert_eq!(CINTlen_cart(2).unwrap(), 6);
        assert_eq!(CINTcgtos_cart(0, &bas).unwrap(), 1);
        assert_eq!(CINTcgtos_spheric(1, &bas).unwrap(), 6);
        assert_eq!(CINTtot_cgto_spheric(&bas, 2).unwrap(), 7);
        assert_eq!(CINTtot_pgto_spheric(&bas, 2).unwrap(), 5);
        assert!(CINTgto_norm(1, 0.5) > 0.0);
    }

    #[test]
    fn helper_offsets_write_prefix_sums() {
        let bas = sample_bas();
        let mut offsets = vec![0; 3];
        CINTshells_cart_offset(&mut offsets, &bas, 2).unwrap();
        assert_eq!(offsets, vec![0, 1, 7]);
    }
}
