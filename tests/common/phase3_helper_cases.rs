use super::phase2_fixtures::stable_raw_layout;

#[derive(Debug, Clone)]
pub struct ShellNormalizationExpectation {
    pub shell_index: usize,
    pub angular_momentum: usize,
    pub exponents: Vec<f64>,
    pub coefficients: Vec<f64>,
}

pub fn stable_shell_normalization_expectations() -> Vec<ShellNormalizationExpectation> {
    vec![
        ShellNormalizationExpectation {
            shell_index: 0,
            angular_momentum: 0,
            exponents: vec![6.0, 1.2],
            coefficients: vec![0.7, 0.3],
        },
        ShellNormalizationExpectation {
            shell_index: 1,
            angular_momentum: 1,
            exponents: vec![4.0, 0.8],
            coefficients: vec![0.6, 0.4],
        },
        ShellNormalizationExpectation {
            shell_index: 2,
            angular_momentum: 2,
            exponents: vec![3.0, 0.7],
            coefficients: vec![0.5, 0.5],
        },
        ShellNormalizationExpectation {
            shell_index: 3,
            angular_momentum: 0,
            exponents: vec![2.0, 0.5],
            coefficients: vec![0.55, 0.45],
        },
    ]
}

pub fn malformed_positive_kappa_bas() -> Vec<i32> {
    let (_, mut bas, _) = stable_raw_layout();
    // bas[kappa] for shell 0; slot index 4 in BAS_SLOTS of 8.
    bas[4] = 1;
    bas
}

pub fn malformed_truncated_bas() -> Vec<i32> {
    let (_, mut bas, _) = stable_raw_layout();
    bas.pop();
    bas
}

pub fn helper_matrix_case_count() -> usize {
    14
}

pub fn expected_gto_norm(angular_momentum: usize, exponent: f64) -> f64 {
    let gamma = gamma_half_integer(angular_momentum);
    let power = (2.0 * exponent).powf((angular_momentum as f64) + 1.5);
    let gaussian_int = gamma / (2.0 * power);
    1.0 / gaussian_int.sqrt()
}

fn gamma_half_integer(angular_momentum: usize) -> f64 {
    let mut gamma = std::f64::consts::PI.sqrt();
    for step in 0..=angular_momentum {
        gamma *= step as f64 + 0.5;
    }
    gamma
}
