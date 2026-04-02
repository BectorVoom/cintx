#![allow(non_snake_case)]
#![allow(clippy::too_many_arguments)]

use crate::optimizer::{RawOptimizerHandle, init_optimizer_with_symbol};
#[cfg(test)]
use crate::raw::query_workspace_raw;
use crate::raw::{RawApiId, RawEvalSummary, eval_raw};
use cintx_core::cintxRsError;
#[cfg(test)]
use cintx_ops::resolver::{HelperKind, ManifestEntry, Resolver};
#[cfg(test)]
use cintx_runtime::WorkspaceQuery;
#[cfg(test)]
use std::collections::BTreeSet;

unsafe fn eval_legacy(
    api: RawApiId,
    out: Option<&mut [f64]>,
    shls: &[i32],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    opt: Option<&RawOptimizerHandle>,
) -> Result<RawEvalSummary, cintxRsError> {
    unsafe { eval_raw(api, out, None, shls, atm, bas, env, opt, None) }
}

#[cfg(test)]
unsafe fn query_legacy(
    api: RawApiId,
    shls: &[i32],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    opt: Option<&RawOptimizerHandle>,
) -> Result<WorkspaceQuery, cintxRsError> {
    unsafe { query_workspace_raw(api, None, shls, atm, bas, env, opt) }
}

macro_rules! all_cint1e_wrappers {
    (
        $cart_fn:ident,
        $sph_fn:ident,
        $spinor_fn:ident,
        $cart_api:expr,
        $sph_api:expr,
        $spinor_api:expr
    ) => {
        pub unsafe fn $cart_fn(
            out: Option<&mut [f64]>,
            shls: &[i32],
            atm: &[i32],
            bas: &[i32],
            env: &[f64],
        ) -> Result<RawEvalSummary, cintxRsError> {
            unsafe { eval_legacy($cart_api, out, shls, atm, bas, env, None) }
        }

        pub unsafe fn $sph_fn(
            out: Option<&mut [f64]>,
            shls: &[i32],
            atm: &[i32],
            bas: &[i32],
            env: &[f64],
        ) -> Result<RawEvalSummary, cintxRsError> {
            unsafe { eval_legacy($sph_api, out, shls, atm, bas, env, None) }
        }

        pub unsafe fn $spinor_fn(
            out: Option<&mut [f64]>,
            shls: &[i32],
            atm: &[i32],
            bas: &[i32],
            env: &[f64],
        ) -> Result<RawEvalSummary, cintxRsError> {
            unsafe { eval_legacy($spinor_api, out, shls, atm, bas, env, None) }
        }
    };
}

macro_rules! all_cint_wrappers {
    (
        $cart_fn:ident,
        $sph_fn:ident,
        $spinor_fn:ident,
        $cart_opt_fn:ident,
        $sph_opt_fn:ident,
        $opt_fn:ident,
        $cart_api:expr,
        $sph_api:expr,
        $spinor_api:expr
    ) => {
        pub unsafe fn $cart_fn(
            out: Option<&mut [f64]>,
            shls: &[i32],
            atm: &[i32],
            bas: &[i32],
            env: &[f64],
            opt: Option<&RawOptimizerHandle>,
        ) -> Result<RawEvalSummary, cintxRsError> {
            unsafe { eval_legacy($cart_api, out, shls, atm, bas, env, opt) }
        }

        pub unsafe fn $sph_fn(
            out: Option<&mut [f64]>,
            shls: &[i32],
            atm: &[i32],
            bas: &[i32],
            env: &[f64],
            opt: Option<&RawOptimizerHandle>,
        ) -> Result<RawEvalSummary, cintxRsError> {
            unsafe { eval_legacy($sph_api, out, shls, atm, bas, env, opt) }
        }

        pub unsafe fn $spinor_fn(
            out: Option<&mut [f64]>,
            shls: &[i32],
            atm: &[i32],
            bas: &[i32],
            env: &[f64],
            opt: Option<&RawOptimizerHandle>,
        ) -> Result<RawEvalSummary, cintxRsError> {
            unsafe { eval_legacy($spinor_api, out, shls, atm, bas, env, opt) }
        }

        pub fn $cart_opt_fn(
            atm: &[i32],
            bas: &[i32],
            env: &[f64],
        ) -> Result<RawOptimizerHandle, cintxRsError> {
            init_optimizer_with_symbol(stringify!($cart_opt_fn), atm, bas, env)
        }

        pub fn $sph_opt_fn(
            atm: &[i32],
            bas: &[i32],
            env: &[f64],
        ) -> Result<RawOptimizerHandle, cintxRsError> {
            init_optimizer_with_symbol(stringify!($sph_opt_fn), atm, bas, env)
        }

        pub fn $opt_fn(
            atm: &[i32],
            bas: &[i32],
            env: &[f64],
        ) -> Result<RawOptimizerHandle, cintxRsError> {
            init_optimizer_with_symbol(stringify!($opt_fn), atm, bas, env)
        }
    };
}

all_cint1e_wrappers!(
    cint1e_kin_cart,
    cint1e_kin_sph,
    cint1e_kin,
    RawApiId::INT1E_KIN_CART,
    RawApiId::INT1E_KIN_SPH,
    RawApiId::INT1E_KIN_SPINOR
);

all_cint_wrappers!(
    cint1e_nuc_cart,
    cint1e_nuc_sph,
    cint1e_nuc,
    cint1e_nuc_cart_optimizer,
    cint1e_nuc_sph_optimizer,
    cint1e_nuc_optimizer,
    RawApiId::INT1E_NUC_CART,
    RawApiId::INT1E_NUC_SPH,
    RawApiId::INT1E_NUC_SPINOR
);
all_cint_wrappers!(
    cint1e_ovlp_cart,
    cint1e_ovlp_sph,
    cint1e_ovlp,
    cint1e_ovlp_cart_optimizer,
    cint1e_ovlp_sph_optimizer,
    cint1e_ovlp_optimizer,
    RawApiId::INT1E_OVLP_CART,
    RawApiId::INT1E_OVLP_SPH,
    RawApiId::INT1E_OVLP_SPINOR
);
all_cint_wrappers!(
    cint2e_cart,
    cint2e_sph,
    cint2e,
    cint2e_cart_optimizer,
    cint2e_sph_optimizer,
    cint2e_optimizer,
    RawApiId::INT2E_CART,
    RawApiId::INT2E_SPH,
    RawApiId::INT2E_SPINOR
);
all_cint_wrappers!(
    cint2c2e_cart,
    cint2c2e_sph,
    cint2c2e,
    cint2c2e_cart_optimizer,
    cint2c2e_sph_optimizer,
    cint2c2e_optimizer,
    RawApiId::INT2C2E_CART,
    RawApiId::INT2C2E_SPH,
    RawApiId::INT2C2E_SPINOR
);
all_cint_wrappers!(
    cint3c1e_p2_cart,
    cint3c1e_p2_sph,
    cint3c1e_p2,
    cint3c1e_p2_cart_optimizer,
    cint3c1e_p2_sph_optimizer,
    cint3c1e_p2_optimizer,
    RawApiId::INT3C1E_P2_CART,
    RawApiId::INT3C1E_P2_SPH,
    RawApiId::INT3C1E_P2_SPINOR
);
all_cint_wrappers!(
    cint3c2e_ip1_cart,
    cint3c2e_ip1_sph,
    cint3c2e_ip1,
    cint3c2e_ip1_cart_optimizer,
    cint3c2e_ip1_sph_optimizer,
    cint3c2e_ip1_optimizer,
    RawApiId::INT3C2E_IP1_CART,
    RawApiId::INT3C2E_IP1_SPH,
    RawApiId::INT3C2E_IP1_SPINOR
);

pub const LEGACY_WRAPPER_SYMBOLS: &[&str] = &[
    "cint1e_kin_cart",
    "cint1e_kin_sph",
    "cint1e_kin",
    "cint1e_nuc_cart",
    "cint1e_nuc_sph",
    "cint1e_nuc",
    "cint1e_nuc_cart_optimizer",
    "cint1e_nuc_sph_optimizer",
    "cint1e_nuc_optimizer",
    "cint1e_ovlp_cart",
    "cint1e_ovlp_sph",
    "cint1e_ovlp",
    "cint1e_ovlp_cart_optimizer",
    "cint1e_ovlp_sph_optimizer",
    "cint1e_ovlp_optimizer",
    "cint2e_cart",
    "cint2e_sph",
    "cint2e",
    "cint2e_cart_optimizer",
    "cint2e_sph_optimizer",
    "cint2e_optimizer",
    "cint2c2e_cart",
    "cint2c2e_sph",
    "cint2c2e",
    "cint2c2e_cart_optimizer",
    "cint2c2e_sph_optimizer",
    "cint2c2e_optimizer",
    "cint3c1e_p2_cart",
    "cint3c1e_p2_sph",
    "cint3c1e_p2",
    "cint3c1e_p2_cart_optimizer",
    "cint3c1e_p2_sph_optimizer",
    "cint3c1e_p2_optimizer",
    "cint3c2e_ip1_cart",
    "cint3c2e_ip1_sph",
    "cint3c2e_ip1",
    "cint3c2e_ip1_cart_optimizer",
    "cint3c2e_ip1_sph_optimizer",
    "cint3c2e_ip1_optimizer",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum MiscWrapperMacro {
        AllCint,
        AllCint1e,
    }

    fn base_symbol_from_operator(entry: &ManifestEntry) -> Option<&'static str> {
        if !matches!(entry.helper_kind, HelperKind::Operator)
            || !entry.compiled_in_profiles.contains(&"base")
        {
            return None;
        }
        entry
            .symbol_name
            .strip_suffix("_cart")
            .or_else(|| entry.symbol_name.strip_suffix("_sph"))
            .or_else(|| entry.symbol_name.strip_suffix("_spinor"))
    }

    fn misc_wrapper_macro(base_symbol: &str) -> Option<MiscWrapperMacro> {
        match base_symbol {
            "int1e_ovlp" | "int1e_nuc" | "int2e" | "int2c2e" | "int3c1e_p2" | "int3c2e_ip1" => {
                Some(MiscWrapperMacro::AllCint)
            }
            "int1e_kin" => Some(MiscWrapperMacro::AllCint1e),
            _ => None,
        }
    }

    fn expected_legacy_wrapper_symbols(
        base_symbol: &str,
        macro_kind: MiscWrapperMacro,
    ) -> BTreeSet<String> {
        let mut expected = BTreeSet::from([
            format!("c{base_symbol}_cart"),
            format!("c{base_symbol}_sph"),
            format!("c{base_symbol}"),
        ]);
        if matches!(macro_kind, MiscWrapperMacro::AllCint) {
            expected.insert(format!("c{base_symbol}_cart_optimizer"));
            expected.insert(format!("c{base_symbol}_sph_optimizer"));
            expected.insert(format!("c{base_symbol}_optimizer"));
        }
        expected
    }

    #[test]
    fn legacy_wrapper_surface_matches_misc() {
        let base_symbols: BTreeSet<String> = Resolver::manifest()
            .iter()
            .filter_map(base_symbol_from_operator)
            .map(str::to_owned)
            .filter(|symbol| misc_wrapper_macro(symbol).is_some())
            .collect();

        let mut expected = BTreeSet::new();
        for base_symbol in base_symbols {
            let macro_kind = misc_wrapper_macro(&base_symbol).unwrap();
            expected.extend(expected_legacy_wrapper_symbols(&base_symbol, macro_kind));
        }

        let actual: BTreeSet<String> = LEGACY_WRAPPER_SYMBOLS
            .iter()
            .map(|symbol| (*symbol).to_owned())
            .collect();

        assert_eq!(actual, expected);
    }

    #[test]
    fn wrappers_call_shared_eval_path() {
        use cintx_core::cintxRsError;

        let atm = vec![1, 0, 1, 0, 0, 0];
        let bas = vec![
            0, 0, 1, 1, 0, 3, 4, 0, //
            0, 1, 1, 1, 0, 5, 6, 0, //
            0, 0, 1, 1, 0, 7, 8, 0, //
        ];
        let env = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.8, 0.9, 0.7, 0.6];
        let shls = [0, 1];
        let mut out = vec![0.0; 3];

        // D-05: legacy wrappers now route through real CubeClExecutor path.
        // Accept both GPU-success and fail-closed wgpu-capability error.
        let result = unsafe { cint1e_ovlp_cart(Some(&mut out), &shls, &atm, &bas, &env, None) };
        match result {
            Ok(summary) => {
                assert!(summary.bytes_written > 0);
            }
            Err(cintxRsError::UnsupportedApi { ref requested }) if requested.contains("wgpu-capability") => {
                // No GPU adapter — correct fail-closed behavior (D-01/D-02).
            }
            Err(other) => panic!("unexpected error from legacy eval: {other:?}"),
        }

        let query =
            unsafe { query_legacy(RawApiId::INT1E_OVLP_CART, &shls, &atm, &bas, &env, None) }
                .unwrap();
        assert!(query.bytes > 0);
    }
}
