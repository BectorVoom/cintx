use cintx::{
    Atom, BasisSet, IntegralFamily, Operator, OperatorKind, Representation, WorkspaceQueryOptions,
    runtime::{layout_for_plan, plan_execution},
};

#[test]
fn planner_representation_dims() {
    let basis = sample_basis();
    let operator = Operator::new(IntegralFamily::OneElectron, OperatorKind::Overlap)
        .expect("one-electron overlap should be supported");
    let options = WorkspaceQueryOptions::default();

    let cart_plan = plan_execution(
        &basis,
        cintx::ExecutionRequest::from_safe(operator, Representation::Cartesian, &[0, 1], &options),
    )
    .expect("cartesian plan should succeed");
    assert_eq!(cart_plan.natural_dims, vec![6, 3]);
    assert_eq!(cart_plan.dims, vec![6, 3]);
    assert_eq!(cart_plan.element_count, 18);
    assert_eq!(cart_plan.element_width_bytes, 8);
    assert_eq!(cart_plan.required_output_bytes, 18 * 8);

    let sph_plan = plan_execution(
        &basis,
        cintx::ExecutionRequest::from_safe(operator, Representation::Spherical, &[0, 1], &options),
    )
    .expect("spherical plan should succeed");
    assert_eq!(sph_plan.natural_dims, vec![5, 3]);
    assert_eq!(sph_plan.dims, vec![5, 3]);
    assert_eq!(sph_plan.element_count, 15);
    assert_eq!(sph_plan.element_width_bytes, 8);
    assert_eq!(sph_plan.required_output_bytes, 15 * 8);

    let spinor_plan = plan_execution(
        &basis,
        cintx::ExecutionRequest::from_safe(operator, Representation::Spinor, &[0, 1], &options),
    )
    .expect("spinor plan should succeed");
    assert_eq!(spinor_plan.natural_dims, vec![10, 6]);
    assert_eq!(spinor_plan.dims, vec![10, 6]);
    assert_eq!(spinor_plan.element_count, 60);
    assert_eq!(spinor_plan.element_width_bytes, 16);
    assert_eq!(spinor_plan.required_output_bytes, 60 * 16);

    let spinor_layout = layout_for_plan(&spinor_plan);
    assert_eq!(spinor_layout.dims, vec![10, 6]);
    assert_eq!(spinor_layout.element_count, 60);
    assert_eq!(spinor_layout.required_bytes, 60 * 16);
}

fn sample_basis() -> BasisSet {
    let atom_a = Atom::new(8, [0.0, 0.0, -0.1173]).expect("atom A should be valid");
    let atom_b = Atom::new(1, [0.0, 0.7572, 0.4692]).expect("atom B should be valid");
    let shell_d = Shell::new(0, 2, vec![4.0, 1.0], vec![0.7, 0.3]).expect("d shell should be valid");
    let shell_p = Shell::new(1, 1, vec![3.0, 0.8], vec![0.6, 0.4]).expect("p shell should be valid");

    BasisSet::new(vec![atom_a, atom_b], vec![shell_d, shell_p]).expect("basis should be valid")
}

use cintx::Shell;
