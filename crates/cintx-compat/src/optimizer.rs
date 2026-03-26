#![allow(non_snake_case)]

use crate::raw::{RawAtmView, RawBasView, RawEnvView};
use cintx_core::cintxRsError;
use std::sync::Arc;

#[derive(Debug, Default)]
struct OptimizerMetadata {
    symbol_hint: Option<&'static str>,
    workspace_hint_bytes: Option<usize>,
}

/// Minimal compat-owned optimizer contract. Lifecycle APIs land in Plan 07.
#[derive(Clone, Debug, Default)]
pub struct RawOptimizerHandle {
    inner: Arc<OptimizerMetadata>,
}

impl RawOptimizerHandle {
    pub fn symbol_hint(&self) -> Option<&'static str> {
        self.inner.symbol_hint
    }

    pub fn workspace_hint_bytes(&self) -> Option<usize> {
        self.inner.workspace_hint_bytes
    }

    pub(crate) fn with_hints(
        symbol_hint: Option<&'static str>,
        workspace_hint_bytes: Option<usize>,
    ) -> Self {
        Self {
            inner: Arc::new(OptimizerMetadata {
                symbol_hint,
                workspace_hint_bytes,
            }),
        }
    }
}

fn validate_optimizer_inputs(atm: &[i32], bas: &[i32], env: &[f64]) -> Result<(), cintxRsError> {
    let atm = RawAtmView::new(atm)?;
    let bas = RawBasView::new(bas)?;
    let env = RawEnvView::new(env);
    atm.validate(&env)?;
    bas.validate(&env)?;
    Ok(())
}

pub(crate) fn init_optimizer_with_symbol(
    symbol_hint: &'static str,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> Result<RawOptimizerHandle, cintxRsError> {
    validate_optimizer_inputs(atm, bas, env)?;
    Ok(RawOptimizerHandle::with_hints(Some(symbol_hint), Some(256)))
}

pub fn CINTinit_2e_optimizer(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> Result<RawOptimizerHandle, cintxRsError> {
    init_optimizer_with_symbol("CINTinit_2e_optimizer", atm, bas, env)
}

pub fn CINTinit_optimizer(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> Result<RawOptimizerHandle, cintxRsError> {
    init_optimizer_with_symbol("CINTinit_optimizer", atm, bas, env)
}

pub fn CINTdel_2e_optimizer(opt: &mut Option<RawOptimizerHandle>) {
    *opt = None;
}

pub fn CINTdel_optimizer(opt: &mut Option<RawOptimizerHandle>) {
    *opt = None;
}

pub fn int2e_cart_optimizer(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> Result<RawOptimizerHandle, cintxRsError> {
    init_optimizer_with_symbol("int2e_cart_optimizer", atm, bas, env)
}

pub fn int2e_sph_optimizer(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> Result<RawOptimizerHandle, cintxRsError> {
    init_optimizer_with_symbol("int2e_sph_optimizer", atm, bas, env)
}

pub fn int2e_optimizer(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> Result<RawOptimizerHandle, cintxRsError> {
    init_optimizer_with_symbol("int2e_optimizer", atm, bas, env)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_raw() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        let atm = vec![1, 0, 1, 0, 0, 0];
        let bas = vec![0, 0, 1, 1, 0, 3, 4, 0];
        let env = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        (atm, bas, env)
    }

    #[test]
    fn optimizer_init_and_delete_round_trip() {
        let (atm, bas, env) = sample_raw();
        let handle = CINTinit_optimizer(&atm, &bas, &env).unwrap();
        assert_eq!(handle.symbol_hint(), Some("CINTinit_optimizer"));
        assert_eq!(handle.workspace_hint_bytes(), Some(256));

        let mut slot = Some(handle);
        CINTdel_optimizer(&mut slot);
        assert!(slot.is_none());
    }

    #[test]
    fn int2e_optimizer_entry_points_return_symbol_hints() {
        let (atm, bas, env) = sample_raw();

        let cart = int2e_cart_optimizer(&atm, &bas, &env).unwrap();
        assert_eq!(cart.symbol_hint(), Some("int2e_cart_optimizer"));

        let sph = int2e_sph_optimizer(&atm, &bas, &env).unwrap();
        assert_eq!(sph.symbol_hint(), Some("int2e_sph_optimizer"));

        let plain = int2e_optimizer(&atm, &bas, &env).unwrap();
        assert_eq!(plain.symbol_hint(), Some("int2e_optimizer"));
    }
}
