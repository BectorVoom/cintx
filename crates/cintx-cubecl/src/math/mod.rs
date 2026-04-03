//! Math primitives for Gaussian integral computation on CubeCL backends.
//!
//! All functions are implemented as `#[cube]` kernel functions to be executable
//! on both CPU (test) and GPU (production) backends.

pub mod boys;
pub mod obara_saika;
pub mod pdata;
pub mod rys;
