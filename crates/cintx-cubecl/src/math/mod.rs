pub mod boys;
pub mod obara_saika;
pub mod pdata;
pub mod rys;
// Phase 25 FND-02 — host Wheeler/Jacobi nroots>=6 root engine.
// Task 1a: eigh MRRR symmetric-tridiagonal eigensolver.
// Task 1b: Wheeler/Jacobi modified-moments engine.
pub mod eigh;
pub mod rys_wheeler;
pub mod roots_jacobi_data;
pub mod roots_xw_data;
pub mod stg;
// Phase 19 Plan 01 Wave 0 scaffolding — algorithm bodies land in Plan 02.
pub mod bessel;
pub mod radial_quadrature;
// Phase 19 Plan 05 — K-Taylor radial port foundation (host-first, D-13..D-16).
pub mod ecp_k_taylor;
pub mod ecp_k_taylor_data;
