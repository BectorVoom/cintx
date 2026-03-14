use cintx::LibcintRsError;

#[test]
fn allocation_paths_use_fallible_policy() {
    let real_error = cintx::runtime::memory::allocator::try_alloc_real_buffer(
        usize::MAX,
        "test.real_allocation",
    )
    .expect_err("oversized real allocation must surface typed failure");
    assert!(matches!(
        real_error,
        LibcintRsError::AllocationFailure {
            operation: "test.real_allocation",
            ..
        }
    ));

    let spinor_error = cintx::runtime::memory::allocator::try_alloc_spinor_buffer(
        usize::MAX,
        "test.spinor_allocation",
    )
    .expect_err("oversized spinor allocation must surface typed failure");
    assert!(matches!(
        spinor_error,
        LibcintRsError::AllocationFailure {
            operation: "test.spinor_allocation",
            ..
        }
    ));
}
