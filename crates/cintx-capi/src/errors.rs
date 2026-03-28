use cintx_core::cintxRsError;
use std::cell::RefCell;
use std::ffi::c_char;
use std::ptr;

/// Stable integer status codes exposed through the C ABI.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CintxStatus {
    Success = 0,
    InvalidInput = 1,
    UnsupportedApi = 2,
    UnsupportedRepresentation = 3,
    BufferTooSmall = 4,
    MemoryLimitExceeded = 5,
    AllocationFailed = 6,
    ExecutionFailed = 7,
    NullPointer = 8,
    Panic = 9,
}

impl CintxStatus {
    pub const fn code(self) -> i32 {
        self as i32
    }
}

pub const CINTX_STATUS_SUCCESS: i32 = CintxStatus::Success as i32;
pub const CINTX_STATUS_INVALID_INPUT: i32 = CintxStatus::InvalidInput as i32;
pub const CINTX_STATUS_UNSUPPORTED_API: i32 = CintxStatus::UnsupportedApi as i32;
pub const CINTX_STATUS_UNSUPPORTED_REPRESENTATION: i32 =
    CintxStatus::UnsupportedRepresentation as i32;
pub const CINTX_STATUS_BUFFER_TOO_SMALL: i32 = CintxStatus::BufferTooSmall as i32;
pub const CINTX_STATUS_MEMORY_LIMIT_EXCEEDED: i32 = CintxStatus::MemoryLimitExceeded as i32;
pub const CINTX_STATUS_ALLOCATION_FAILED: i32 = CintxStatus::AllocationFailed as i32;
pub const CINTX_STATUS_EXECUTION_FAILED: i32 = CintxStatus::ExecutionFailed as i32;
pub const CINTX_STATUS_NULL_POINTER: i32 = CintxStatus::NullPointer as i32;
pub const CINTX_STATUS_PANIC: i32 = CintxStatus::Panic as i32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CintxErrorReport {
    pub status: CintxStatus,
    pub api: String,
    pub family: String,
    pub representation: String,
    pub message: String,
}

impl CintxErrorReport {
    pub fn success() -> Self {
        Self {
            status: CintxStatus::Success,
            api: String::new(),
            family: String::new(),
            representation: String::new(),
            message: String::new(),
        }
    }

    pub fn from_core_error(
        api: &str,
        family: &str,
        representation: &str,
        error: &cintxRsError,
    ) -> Self {
        Self {
            status: status_from_core_error(error),
            api: api.to_owned(),
            family: family.to_owned(),
            representation: representation.to_owned(),
            message: error.to_string(),
        }
    }

    pub fn null_pointer(api: &str, family: &str, representation: &str, field: &str) -> Self {
        Self {
            status: CintxStatus::NullPointer,
            api: api.to_owned(),
            family: family.to_owned(),
            representation: representation.to_owned(),
            message: format!("null pointer for required parameter `{field}`"),
        }
    }

    pub fn panic(api: &str, family: &str, representation: &str, detail: &str) -> Self {
        Self {
            status: CintxStatus::Panic,
            api: api.to_owned(),
            family: family.to_owned(),
            representation: representation.to_owned(),
            message: format!("panic in C ABI shim: {detail}"),
        }
    }
}

thread_local! {
    static LAST_ERROR: RefCell<CintxErrorReport> = RefCell::new(CintxErrorReport::success());
}

pub fn status_from_core_error(error: &cintxRsError) -> CintxStatus {
    match error {
        cintxRsError::UnsupportedApi { .. } => CintxStatus::UnsupportedApi,
        cintxRsError::UnsupportedRepresentation { .. } => CintxStatus::UnsupportedRepresentation,
        cintxRsError::InvalidShellTuple { .. }
        | cintxRsError::InvalidShellAtomIndex { .. }
        | cintxRsError::InvalidDims { .. }
        | cintxRsError::InvalidAtmLayout { .. }
        | cintxRsError::InvalidBasLayout { .. }
        | cintxRsError::InvalidEnvOffset { .. } => CintxStatus::InvalidInput,
        cintxRsError::BufferTooSmall { .. } => CintxStatus::BufferTooSmall,
        cintxRsError::MemoryLimitExceeded { .. } => CintxStatus::MemoryLimitExceeded,
        cintxRsError::HostAllocationFailed { .. } | cintxRsError::DeviceOutOfMemory { .. } => {
            CintxStatus::AllocationFailed
        }
        cintxRsError::ChunkPlanFailed { .. } => CintxStatus::ExecutionFailed,
    }
}

pub fn set_last_error(report: CintxErrorReport) {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = report;
    });
}

pub fn set_last_error_from_core(
    api: &str,
    family: &str,
    representation: &str,
    error: &cintxRsError,
) -> CintxStatus {
    let report = CintxErrorReport::from_core_error(api, family, representation, error);
    let status = report.status;
    set_last_error(report);
    status
}

pub fn clear_last_error() {
    set_last_error(CintxErrorReport::success());
}

pub fn current_last_error() -> CintxErrorReport {
    LAST_ERROR.with(|slot| slot.borrow().clone())
}

pub fn last_error_status() -> CintxStatus {
    LAST_ERROR.with(|slot| slot.borrow().status)
}

pub fn last_error_code() -> i32 {
    last_error_status().code()
}

fn copy_field_to_buffer(field: &str, out: *mut c_char, out_len: usize) -> usize {
    let required = field.len().saturating_add(1);
    if out.is_null() || out_len == 0 {
        return required;
    }

    let writable = field.len().min(out_len.saturating_sub(1));
    // SAFETY: `out` is non-null and `writable` bytes are bounded by `out_len`.
    unsafe {
        ptr::copy_nonoverlapping(field.as_ptr(), out.cast::<u8>(), writable);
        *out.cast::<u8>().add(writable) = 0;
    }
    required
}

fn copy_last_error_field(
    out: *mut c_char,
    out_len: usize,
    selector: fn(&CintxErrorReport) -> &str,
) -> usize {
    LAST_ERROR.with(|slot| {
        let report = slot.borrow();
        copy_field_to_buffer(selector(&report), out, out_len)
    })
}

/// Copies the current thread's last-error message into a caller-owned buffer.
///
/// Returns the required byte count including the trailing `\0`.
pub unsafe fn copy_last_error_message(out: *mut c_char, out_len: usize) -> usize {
    copy_last_error_field(out, out_len, |report| report.message.as_str())
}

/// Copies the current thread's last-error API symbol into a caller-owned buffer.
///
/// Returns the required byte count including the trailing `\0`.
pub unsafe fn copy_last_error_api(out: *mut c_char, out_len: usize) -> usize {
    copy_last_error_field(out, out_len, |report| report.api.as_str())
}

/// Copies the current thread's last-error family into a caller-owned buffer.
///
/// Returns the required byte count including the trailing `\0`.
pub unsafe fn copy_last_error_family(out: *mut c_char, out_len: usize) -> usize {
    copy_last_error_field(out, out_len, |report| report.family.as_str())
}

/// Copies the current thread's last-error representation into a caller-owned buffer.
///
/// Returns the required byte count including the trailing `\0`.
pub unsafe fn copy_last_error_representation(out: *mut c_char, out_len: usize) -> usize {
    copy_last_error_field(out, out_len, |report| report.representation.as_str())
}

#[unsafe(no_mangle)]
pub extern "C" fn cintrs_last_error_code() -> i32 {
    last_error_code()
}

#[unsafe(no_mangle)]
pub extern "C" fn cintrs_clear_last_error() {
    clear_last_error();
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cintrs_copy_last_error_message(out: *mut c_char, out_len: usize) -> usize {
    // SAFETY: The caller provides `out`/`out_len`; we only write within bounds and NUL-terminate.
    unsafe { copy_last_error_message(out, out_len) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cintrs_copy_last_error_api(out: *mut c_char, out_len: usize) -> usize {
    // SAFETY: The caller provides `out`/`out_len`; we only write within bounds and NUL-terminate.
    unsafe { copy_last_error_api(out, out_len) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cintrs_copy_last_error_family(out: *mut c_char, out_len: usize) -> usize {
    // SAFETY: The caller provides `out`/`out_len`; we only write within bounds and NUL-terminate.
    unsafe { copy_last_error_family(out, out_len) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cintrs_copy_last_error_representation(
    out: *mut c_char,
    out_len: usize,
) -> usize {
    // SAFETY: The caller provides `out`/`out_len`; we only write within bounds and NUL-terminate.
    unsafe { copy_last_error_representation(out, out_len) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    fn read_c_buf(bytes: &[c_char]) -> String {
        // SAFETY: test buffers are always NUL-terminated by copy helpers.
        unsafe { CStr::from_ptr(bytes.as_ptr()) }
            .to_str()
            .expect("buffer should contain utf-8 text")
            .to_owned()
    }

    #[test]
    fn success_status_is_zero() {
        assert_eq!(CintxStatus::Success.code(), 0);
    }

    #[test]
    fn exported_status_constants_match_enum_codes() {
        assert_eq!(CINTX_STATUS_SUCCESS, CintxStatus::Success.code());
        assert_eq!(CINTX_STATUS_INVALID_INPUT, CintxStatus::InvalidInput.code());
        assert_eq!(
            CINTX_STATUS_UNSUPPORTED_API,
            CintxStatus::UnsupportedApi.code()
        );
        assert_eq!(
            CINTX_STATUS_UNSUPPORTED_REPRESENTATION,
            CintxStatus::UnsupportedRepresentation.code()
        );
        assert_eq!(
            CINTX_STATUS_BUFFER_TOO_SMALL,
            CintxStatus::BufferTooSmall.code()
        );
        assert_eq!(
            CINTX_STATUS_MEMORY_LIMIT_EXCEEDED,
            CintxStatus::MemoryLimitExceeded.code()
        );
        assert_eq!(
            CINTX_STATUS_ALLOCATION_FAILED,
            CintxStatus::AllocationFailed.code()
        );
        assert_eq!(
            CINTX_STATUS_EXECUTION_FAILED,
            CintxStatus::ExecutionFailed.code()
        );
        assert_eq!(CINTX_STATUS_NULL_POINTER, CintxStatus::NullPointer.code());
        assert_eq!(CINTX_STATUS_PANIC, CintxStatus::Panic.code());
    }

    #[test]
    fn core_error_mapping_uses_typed_status_codes() {
        let invalid_dims = cintxRsError::InvalidDims {
            expected: 4,
            provided: 2,
        };
        let unsupported_api = cintxRsError::UnsupportedApi {
            requested: "int9e_fake_cart".to_owned(),
        };
        let memory_limit = cintxRsError::MemoryLimitExceeded {
            requested: 4096,
            limit: 1024,
        };
        let alloc = cintxRsError::HostAllocationFailed { bytes: 8192 };

        assert_eq!(
            status_from_core_error(&invalid_dims),
            CintxStatus::InvalidInput
        );
        assert_eq!(
            status_from_core_error(&unsupported_api),
            CintxStatus::UnsupportedApi
        );
        assert_eq!(
            status_from_core_error(&memory_limit),
            CintxStatus::MemoryLimitExceeded
        );
        assert_eq!(
            status_from_core_error(&alloc),
            CintxStatus::AllocationFailed
        );
    }

    #[test]
    fn set_copy_and_clear_last_error_roundtrip() {
        clear_last_error();
        let err = cintxRsError::BufferTooSmall {
            required: 16,
            provided: 8,
        };
        let status = set_last_error_from_core("int1e_ovlp_cart", "1e", "cart", &err);
        assert_eq!(status, CintxStatus::BufferTooSmall);
        assert_eq!(cintrs_last_error_code(), CintxStatus::BufferTooSmall.code());

        let mut message = [0 as c_char; 96];
        let mut api = [0 as c_char; 64];
        let mut family = [0 as c_char; 32];
        let mut representation = [0 as c_char; 32];
        // SAFETY: test buffers are valid writable pointers and lengths.
        let message_required =
            unsafe { cintrs_copy_last_error_message(message.as_mut_ptr(), message.len()) };
        // SAFETY: test buffers are valid writable pointers and lengths.
        let api_required = unsafe { cintrs_copy_last_error_api(api.as_mut_ptr(), api.len()) };
        // SAFETY: test buffers are valid writable pointers and lengths.
        let family_required =
            unsafe { cintrs_copy_last_error_family(family.as_mut_ptr(), family.len()) };
        // SAFETY: test buffers are valid writable pointers and lengths.
        let representation_required = unsafe {
            cintrs_copy_last_error_representation(representation.as_mut_ptr(), representation.len())
        };

        assert!(message_required > 1);
        assert!(api_required > 1);
        assert!(family_required > 1);
        assert!(representation_required > 1);
        assert!(read_c_buf(&message).contains("buffer too small"));
        assert_eq!(read_c_buf(&api), "int1e_ovlp_cart");
        assert_eq!(read_c_buf(&family), "1e");
        assert_eq!(read_c_buf(&representation), "cart");

        cintrs_clear_last_error();
        assert_eq!(cintrs_last_error_code(), CintxStatus::Success.code());
        assert_eq!(current_last_error(), CintxErrorReport::success());
    }

    #[test]
    fn copy_out_reports_required_length_and_truncates() {
        let report = CintxErrorReport::panic("int2e_sph", "2e", "sph", "panic payload");
        set_last_error(report);

        let required = unsafe { cintrs_copy_last_error_message(ptr::null_mut(), 0) };
        assert!(required > 1);

        let mut small = [0 as c_char; 8];
        let second_required =
            unsafe { cintrs_copy_last_error_message(small.as_mut_ptr(), small.len()) };
        assert_eq!(required, second_required);
        let text = read_c_buf(&small);
        assert!(!text.is_empty());
        assert!(text.len() < required);
    }

    #[test]
    fn last_error_is_isolated_per_thread() {
        set_last_error(CintxErrorReport::panic(
            "main_api",
            "main_family",
            "main_rep",
            "main panic",
        ));

        let worker = std::thread::spawn(|| {
            assert_eq!(last_error_status(), CintxStatus::Success);
            set_last_error(CintxErrorReport::null_pointer(
                "worker_api",
                "2e",
                "sph",
                "out",
            ));
            last_error_status()
        });

        let worker_status = worker.join().expect("worker should not panic");
        assert_eq!(worker_status, CintxStatus::NullPointer);
        assert_eq!(last_error_status(), CintxStatus::Panic);
    }
}
