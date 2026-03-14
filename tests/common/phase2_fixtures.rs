#![allow(dead_code)]

use cintx::{
    Atom, BasisSet, CpuRouteKey, EvaluationOutput, IntegralFamily, Operator, OperatorKind,
    Representation, Shell, WorkspaceQueryOptions,
};

const SHLS_2_SAFE: &[usize] = &[0, 1];
const SHLS_2_RAW: &[i32] = &[0, 1];
const SHLS_3_SAFE: &[usize] = &[0, 1, 2];
const SHLS_3_RAW: &[i32] = &[0, 1, 2];
const SHLS_4_SAFE: &[usize] = &[0, 1, 2, 3];
const SHLS_4_RAW: &[i32] = &[0, 1, 2, 3];

#[derive(Debug, Clone, Copy)]
pub struct StableMatrixCase {
    pub family: IntegralFamily,
    pub operator_kind: OperatorKind,
    pub representation: Representation,
    pub safe_shell_tuple: &'static [usize],
    pub raw_shls: &'static [i32],
}

impl StableMatrixCase {
    pub fn operator(self) -> Operator {
        Operator::new(self.family, self.operator_kind)
            .expect("stable matrix cases must always be valid operator-family pairs")
    }

    pub fn route_key(self) -> CpuRouteKey {
        CpuRouteKey::new(self.family, self.operator_kind, self.representation)
    }

    pub fn id(self) -> String {
        format!(
            "{:?}/{:?}/{:?}",
            self.family, self.operator_kind, self.representation
        )
    }

    pub fn is_explicit_3c1e_spinor(self) -> bool {
        self.family == IntegralFamily::ThreeCenterOneElectron
            && self.operator_kind == OperatorKind::Kinetic
            && self.representation == Representation::Spinor
    }
}

pub fn stable_phase2_matrix() -> Vec<StableMatrixCase> {
    let mut matrix = Vec::with_capacity(15);
    for representation in [
        Representation::Cartesian,
        Representation::Spherical,
        Representation::Spinor,
    ] {
        matrix.push(StableMatrixCase {
            family: IntegralFamily::OneElectron,
            operator_kind: OperatorKind::Overlap,
            representation,
            safe_shell_tuple: SHLS_2_SAFE,
            raw_shls: SHLS_2_RAW,
        });
        matrix.push(StableMatrixCase {
            family: IntegralFamily::TwoElectron,
            operator_kind: OperatorKind::ElectronRepulsion,
            representation,
            safe_shell_tuple: SHLS_4_SAFE,
            raw_shls: SHLS_4_RAW,
        });
        matrix.push(StableMatrixCase {
            family: IntegralFamily::TwoCenterTwoElectron,
            operator_kind: OperatorKind::ElectronRepulsion,
            representation,
            safe_shell_tuple: SHLS_2_SAFE,
            raw_shls: SHLS_2_RAW,
        });
        matrix.push(StableMatrixCase {
            family: IntegralFamily::ThreeCenterOneElectron,
            operator_kind: OperatorKind::Kinetic,
            representation,
            safe_shell_tuple: SHLS_3_SAFE,
            raw_shls: SHLS_3_RAW,
        });
        matrix.push(StableMatrixCase {
            family: IntegralFamily::ThreeCenterTwoElectron,
            operator_kind: OperatorKind::ElectronRepulsion,
            representation,
            safe_shell_tuple: SHLS_3_SAFE,
            raw_shls: SHLS_3_RAW,
        });
    }

    matrix
}

pub fn out_of_phase_route_keys() -> Vec<CpuRouteKey> {
    vec![
        CpuRouteKey::new(
            IntegralFamily::OneElectron,
            OperatorKind::Kinetic,
            Representation::Cartesian,
        ),
        CpuRouteKey::new(
            IntegralFamily::OneElectron,
            OperatorKind::NuclearAttraction,
            Representation::Spinor,
        ),
        CpuRouteKey::new(
            IntegralFamily::TwoElectron,
            OperatorKind::Overlap,
            Representation::Spherical,
        ),
        CpuRouteKey::new(
            IntegralFamily::ThreeCenterTwoElectron,
            OperatorKind::Kinetic,
            Representation::Spinor,
        ),
    ]
}

pub fn stable_safe_basis() -> BasisSet {
    let atom_a = Atom::new(8, [0.0, 0.0, -0.1173]).expect("atom A should be valid");
    let atom_b = Atom::new(1, [0.0, 0.7572, 0.4692]).expect("atom B should be valid");

    let shell_s_a = Shell::new(0, 0, vec![6.0, 1.2], vec![0.7, 0.3]).expect("s shell is valid");
    let shell_p_a = Shell::new(0, 1, vec![4.0, 0.8], vec![0.6, 0.4]).expect("p shell is valid");
    let shell_d_b = Shell::new(1, 2, vec![3.0, 0.7], vec![0.5, 0.5]).expect("d shell is valid");
    let shell_s_b = Shell::new(1, 0, vec![2.0, 0.5], vec![0.55, 0.45]).expect("s shell is valid");

    BasisSet::new(
        vec![atom_a, atom_b],
        vec![shell_s_a, shell_p_a, shell_d_b, shell_s_b],
    )
    .expect("phase-2 fixture basis should be valid")
}

pub fn stable_raw_layout() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let atm = vec![
        8, 20, 1, 0, 0, 0, //
        1, 23, 1, 0, 0, 0,
    ];

    let bas = vec![
        0, 0, 2, 1, 0, 28, 30, 0, //
        0, 1, 2, 1, 0, 32, 34, 0, //
        1, 2, 2, 1, 0, 36, 38, 0, //
        1, 0, 2, 1, 0, 40, 42, 0,
    ];

    let mut env = vec![0.0f64; 64];
    env[20..23].copy_from_slice(&[0.0, 0.0, -0.1173]);
    env[23..26].copy_from_slice(&[0.0, 0.7572, 0.4692]);

    env[28..30].copy_from_slice(&[6.0, 1.2]);
    env[30..32].copy_from_slice(&[0.7, 0.3]);
    env[32..34].copy_from_slice(&[4.0, 0.8]);
    env[34..36].copy_from_slice(&[0.6, 0.4]);
    env[36..38].copy_from_slice(&[3.0, 0.7]);
    env[38..40].copy_from_slice(&[0.5, 0.5]);
    env[40..42].copy_from_slice(&[2.0, 0.5]);
    env[42..44].copy_from_slice(&[0.55, 0.45]);

    (atm, bas, env)
}

pub fn phase2_cpu_options(feature_flags: &[&'static str]) -> WorkspaceQueryOptions {
    WorkspaceQueryOptions {
        memory_limit_bytes: None,
        backend_candidate: "cpu",
        feature_flags: feature_flags.to_vec(),
    }
}

pub fn phase2_cpu_options_with_limit(
    limit_bytes: usize,
    feature_flags: &[&'static str],
) -> WorkspaceQueryOptions {
    WorkspaceQueryOptions {
        memory_limit_bytes: Some(limit_bytes),
        backend_candidate: "cpu",
        feature_flags: feature_flags.to_vec(),
    }
}

pub fn flatten_safe_output(output: EvaluationOutput) -> Vec<f64> {
    match output {
        EvaluationOutput::Real(values) => values,
        EvaluationOutput::Spinor(values) => {
            let mut flattened = Vec::with_capacity(values.len() * 2);
            for value in values {
                flattened.push(value[0]);
                flattened.push(value[1]);
            }
            flattened
        }
    }
}

pub fn representation_width_bytes(representation: Representation) -> usize {
    match representation {
        Representation::Spinor => 16,
        Representation::Cartesian | Representation::Spherical => 8,
    }
}

pub fn scalars_per_element(representation: Representation) -> usize {
    match representation {
        Representation::Spinor => 2,
        Representation::Cartesian | Representation::Spherical => 1,
    }
}

pub fn stable_expected_shell_counts_cartesian() -> Vec<usize> {
    vec![1, 3, 6, 1]
}

pub fn stable_expected_shell_counts_spherical() -> Vec<usize> {
    vec![1, 3, 5, 1]
}

pub fn stable_expected_shell_counts_spinor() -> Vec<usize> {
    vec![2, 6, 10, 2]
}

pub fn stable_expected_shell_offsets_cartesian() -> Vec<usize> {
    vec![0, 1, 4, 10]
}

pub fn stable_expected_shell_offsets_spherical() -> Vec<usize> {
    vec![0, 1, 4, 9]
}

pub fn stable_expected_shell_offsets_spinor() -> Vec<usize> {
    vec![0, 2, 8, 18]
}
