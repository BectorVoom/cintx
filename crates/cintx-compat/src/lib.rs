//! Compatibility shims and raw libcint adapters.

pub mod helpers;
pub mod layout;
pub mod legacy;
pub mod optimizer;
pub mod raw;
pub mod transform;

#[cfg(feature = "with-4c1e")]
pub mod workaround;

pub use layout::{CompatDims, ensure_cache_len, required_elems_from_dims, required_f64s_for_bytes};
pub use optimizer::RawOptimizerHandle;
pub use raw::{
    ATM_SLOTS, BAS_SLOTS, PTR_COEFF, PTR_COORD, PTR_EXP, PTR_FRAC_CHARGE, PTR_ZETA, RawApiId,
    RawAtmView, RawBasView, RawEnvView, RawEvalSummary, eval_raw, query_workspace_raw,
};

#[cfg(test)]
mod tests {
    #[test]
    fn exports_and_cubecl_executor_edge_compile() {
        #[allow(unused_imports)]
        use super::{
            CompatDims, RawAtmView, RawBasView, RawEnvView, RawOptimizerHandle, eval_raw, helpers,
            layout, legacy, optimizer, query_workspace_raw, raw, transform,
        };
        #[allow(unused_imports)]
        use cintx_cubecl::executor;
    }
}
