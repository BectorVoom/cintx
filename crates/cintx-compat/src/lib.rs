//! Compatibility shims (stub).

pub mod helpers;
pub mod layout;
pub mod legacy;
pub mod optimizer;
pub mod raw;
pub mod transform;

#[cfg(test)]
mod tests {
    #[test]
    fn exports_and_cubecl_executor_edge_compile() {
        #[allow(unused_imports)]
        use super::{helpers, layout, legacy, optimizer, raw, transform};
        #[allow(unused_imports)]
        use cintx_cubecl::executor;
    }
}
