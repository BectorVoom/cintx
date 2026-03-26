//! Compatibility shims and raw libcint adapters.

pub mod helpers;
pub mod layout;
pub mod legacy;
pub mod optimizer;
pub mod raw;
pub mod transform;

pub use layout::{CompatDims, ensure_cache_len, required_elems_from_dims, required_f64s_for_bytes};
pub use optimizer::RawOptimizerHandle;
pub use raw::{
    ATM_SLOTS, BAS_SLOTS, RawAtmView, RawBasView, RawEnvView, PTR_COEFF, PTR_COORD,
    PTR_EXP, PTR_FRAC_CHARGE, PTR_ZETA,
};

#[cfg(test)]
mod tests {
    #[test]
    fn exports_and_cubecl_executor_edge_compile() {
        #[allow(unused_imports)]
        use super::{
            helpers, layout, legacy, optimizer, raw, transform, CompatDims, RawAtmView,
            RawBasView, RawEnvView, RawOptimizerHandle,
        };
        #[allow(unused_imports)]
        use cintx_cubecl::executor;
    }
}
