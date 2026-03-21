//! CubeCL backend (stub).

pub mod executor;
#[path = "kernels/mod.rs"]
pub mod kernels;
pub mod resident_cache;
pub mod specialization;
pub mod transfer;
pub mod transform;

#[cfg(test)]
mod tests {
    #[test]
    fn exports_compile() {
        #[allow(unused_imports)]
        use super::{executor, kernels, resident_cache, specialization, transfer, transform};
    }
}
