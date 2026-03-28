use crate::errors::{clear_last_error, set_last_error, CintxErrorReport, CintxStatus};
use cintx_compat::{eval_raw, query_workspace_raw, RawApiId, RawOptimizerHandle};
use cintx_core::cintxRsError;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CintxRawApi {
    Int1eOvlpCart = 0,
    Int1eOvlpSph = 1,
    Int1eOvlpSpinor = 2,
    Int1eKinCart = 3,
    Int1eKinSph = 4,
    Int1eKinSpinor = 5,
    Int1eNucCart = 6,
    Int1eNucSph = 7,
    Int1eNucSpinor = 8,
    Int2eCart = 9,
    Int2eSph = 10,
    Int2eSpinor = 11,
    Int2c2eCart = 12,
    Int2c2eSph = 13,
    Int2c2eSpinor = 14,
    Int3c1eP2Cart = 15,
    Int3c1eP2Sph = 16,
    Int3c1eP2Spinor = 17,
    Int3c2eIp1Cart = 18,
    Int3c2eIp1Sph = 19,
    Int3c2eIp1Spinor = 20,
    Int4c1eCart = 21,
    Int4c1eSph = 22,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CintxWorkspaceQuery {
    pub bytes: usize,
    pub alignment: usize,
    pub required_bytes: usize,
    pub chunk_count: usize,
    pub work_units: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CintxEvalSummary {
    pub not0: i32,
    pub bytes_written: usize,
    pub workspace_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
struct RawApiMeta {
    id: RawApiId,
    symbol: &'static str,
    family: &'static str,
    representation: &'static str,
}

impl CintxRawApi {
    fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::Int1eOvlpCart),
            1 => Some(Self::Int1eOvlpSph),
            2 => Some(Self::Int1eOvlpSpinor),
            3 => Some(Self::Int1eKinCart),
            4 => Some(Self::Int1eKinSph),
            5 => Some(Self::Int1eKinSpinor),
            6 => Some(Self::Int1eNucCart),
            7 => Some(Self::Int1eNucSph),
            8 => Some(Self::Int1eNucSpinor),
            9 => Some(Self::Int2eCart),
            10 => Some(Self::Int2eSph),
            11 => Some(Self::Int2eSpinor),
            12 => Some(Self::Int2c2eCart),
            13 => Some(Self::Int2c2eSph),
            14 => Some(Self::Int2c2eSpinor),
            15 => Some(Self::Int3c1eP2Cart),
            16 => Some(Self::Int3c1eP2Sph),
            17 => Some(Self::Int3c1eP2Spinor),
            18 => Some(Self::Int3c2eIp1Cart),
            19 => Some(Self::Int3c2eIp1Sph),
            20 => Some(Self::Int3c2eIp1Spinor),
            21 => Some(Self::Int4c1eCart),
            22 => Some(Self::Int4c1eSph),
            _ => None,
        }
    }

    fn raw_id(self) -> RawApiId {
        match self {
            Self::Int1eOvlpCart => RawApiId::INT1E_OVLP_CART,
            Self::Int1eOvlpSph => RawApiId::INT1E_OVLP_SPH,
            Self::Int1eOvlpSpinor => RawApiId::INT1E_OVLP_SPINOR,
            Self::Int1eKinCart => RawApiId::INT1E_KIN_CART,
            Self::Int1eKinSph => RawApiId::INT1E_KIN_SPH,
            Self::Int1eKinSpinor => RawApiId::INT1E_KIN_SPINOR,
            Self::Int1eNucCart => RawApiId::INT1E_NUC_CART,
            Self::Int1eNucSph => RawApiId::INT1E_NUC_SPH,
            Self::Int1eNucSpinor => RawApiId::INT1E_NUC_SPINOR,
            Self::Int2eCart => RawApiId::INT2E_CART,
            Self::Int2eSph => RawApiId::INT2E_SPH,
            Self::Int2eSpinor => RawApiId::INT2E_SPINOR,
            Self::Int2c2eCart => RawApiId::INT2C2E_CART,
            Self::Int2c2eSph => RawApiId::INT2C2E_SPH,
            Self::Int2c2eSpinor => RawApiId::INT2C2E_SPINOR,
            Self::Int3c1eP2Cart => RawApiId::INT3C1E_P2_CART,
            Self::Int3c1eP2Sph => RawApiId::INT3C1E_P2_SPH,
            Self::Int3c1eP2Spinor => RawApiId::INT3C1E_P2_SPINOR,
            Self::Int3c2eIp1Cart => RawApiId::INT3C2E_IP1_CART,
            Self::Int3c2eIp1Sph => RawApiId::INT3C2E_IP1_SPH,
            Self::Int3c2eIp1Spinor => RawApiId::INT3C2E_IP1_SPINOR,
            Self::Int4c1eCart => RawApiId::INT4C1E_CART,
            Self::Int4c1eSph => RawApiId::INT4C1E_SPH,
        }
    }

    fn symbol(self) -> &'static str {
        match self.raw_id() {
            RawApiId::Symbol(symbol) => symbol,
        }
    }

    fn meta(self) -> RawApiMeta {
        let symbol = self.symbol();
        let family = family_from_symbol(symbol);
        let representation = representation_from_symbol(symbol);
        RawApiMeta {
            id: self.raw_id(),
            symbol,
            family,
            representation,
        }
    }
}

fn family_from_symbol(symbol: &str) -> &'static str {
    if symbol.starts_with("int1e_") {
        "1e"
    } else if symbol.starts_with("int2c2e_") {
        "2c2e"
    } else if symbol.starts_with("int2e_") {
        "2e"
    } else if symbol.starts_with("int3c1e_") {
        "3c1e"
    } else if symbol.starts_with("int3c2e_") {
        "3c2e"
    } else if symbol.starts_with("int4c1e_") {
        "4c1e"
    } else {
        "unknown"
    }
}

fn representation_from_symbol(symbol: &str) -> &'static str {
    if symbol.ends_with("_cart") {
        "cart"
    } else if symbol.ends_with("_sph") {
        "sph"
    } else if symbol.ends_with("_spinor") {
        "spinor"
    } else {
        "unknown"
    }
}

fn invalid_api_report(api: i32) -> CintxErrorReport {
    CintxErrorReport {
        status: CintxStatus::UnsupportedApi,
        api: format!("api-id-{api}"),
        family: "unknown".to_owned(),
        representation: "unknown".to_owned(),
        message: format!("unsupported C ABI api id {api}"),
    }
}

fn invalid_input_report(meta: RawApiMeta, detail: &str) -> CintxErrorReport {
    CintxErrorReport {
        status: CintxStatus::InvalidInput,
        api: meta.symbol.to_owned(),
        family: meta.family.to_owned(),
        representation: meta.representation.to_owned(),
        message: detail.to_owned(),
    }
}

fn map_core_error(meta: RawApiMeta, error: &cintxRsError) -> CintxErrorReport {
    CintxErrorReport::from_core_error(meta.symbol, meta.family, meta.representation, error)
}

fn panic_report(meta: RawApiMeta, panic_payload: &(dyn std::any::Any + Send)) -> CintxErrorReport {
    let detail = if let Some(message) = panic_payload.downcast_ref::<&'static str>() {
        (*message).to_owned()
    } else if let Some(message) = panic_payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_owned()
    };
    CintxErrorReport::panic(meta.symbol, meta.family, meta.representation, &detail)
}

fn run_with_status<F>(meta: RawApiMeta, run: F) -> i32
where
    F: FnOnce() -> Result<(), CintxErrorReport>,
{
    match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(())) => {
            clear_last_error();
            CintxStatus::Success.code()
        }
        Ok(Err(report)) => {
            let status = report.status;
            set_last_error(report);
            status.code()
        }
        Err(panic_payload) => {
            let report = panic_report(meta, panic_payload.as_ref());
            let status = report.status;
            set_last_error(report);
            status.code()
        }
    }
}

unsafe fn required_slice<'a, T>(
    ptr: *const T,
    len: usize,
    field: &str,
    meta: RawApiMeta,
) -> Result<&'a [T], CintxErrorReport> {
    if ptr.is_null() {
        return Err(CintxErrorReport::null_pointer(
            meta.symbol,
            meta.family,
            meta.representation,
            field,
        ));
    }
    // SAFETY: Caller guarantees pointer validity for `len` entries; null is rejected above.
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

unsafe fn optional_slice<'a, T>(ptr: *const T, len: usize) -> Option<&'a [T]> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: Caller guarantees pointer validity when non-null.
    Some(unsafe { std::slice::from_raw_parts(ptr, len) })
}

unsafe fn optional_slice_mut<'a, T>(ptr: *mut T, len: usize) -> Option<&'a mut [T]> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: Caller guarantees pointer validity when non-null.
    Some(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
}

#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cintrs_query_workspace(
    api: i32,
    dims: *const i32,
    dims_len: usize,
    shls: *const i32,
    shls_len: usize,
    atm: *const i32,
    atm_len: usize,
    bas: *const i32,
    bas_len: usize,
    env: *const f64,
    env_len: usize,
    opt: *const RawOptimizerHandle,
    query_out: *mut CintxWorkspaceQuery,
) -> i32 {
    let Some(api) = CintxRawApi::from_i32(api) else {
        let report = invalid_api_report(api);
        let status = report.status;
        set_last_error(report);
        return status.code();
    };
    let meta = api.meta();

    run_with_status(meta, || {
        if query_out.is_null() {
            return Err(CintxErrorReport::null_pointer(
                meta.symbol,
                meta.family,
                meta.representation,
                "query_out",
            ));
        }
        let shls = unsafe { required_slice(shls, shls_len, "shls", meta)? };
        let atm = unsafe { required_slice(atm, atm_len, "atm", meta)? };
        let bas = unsafe { required_slice(bas, bas_len, "bas", meta)? };
        let env = unsafe { required_slice(env, env_len, "env", meta)? };
        let dims = unsafe { optional_slice(dims, dims_len) };
        let opt = unsafe { opt.as_ref() };
        let query = unsafe { query_workspace_raw(meta.id, dims, shls, atm, bas, env, opt) }
            .map_err(|error| map_core_error(meta, &error))?;

        let ffi_query = CintxWorkspaceQuery {
            bytes: query.bytes,
            alignment: query.alignment,
            required_bytes: query.required_bytes,
            chunk_count: query.chunk_count,
            work_units: query.work_units,
        };
        // SAFETY: `query_out` is checked non-null above and points to caller-owned storage.
        unsafe {
            ptr::write(query_out, ffi_query);
        }
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cintrs_eval(
    api: i32,
    out: *mut f64,
    out_len: usize,
    dims: *const i32,
    dims_len: usize,
    shls: *const i32,
    shls_len: usize,
    atm: *const i32,
    atm_len: usize,
    bas: *const i32,
    bas_len: usize,
    env: *const f64,
    env_len: usize,
    opt: *const RawOptimizerHandle,
    cache: *mut f64,
    cache_len: usize,
    summary_out: *mut CintxEvalSummary,
) -> i32 {
    let Some(api) = CintxRawApi::from_i32(api) else {
        let report = invalid_api_report(api);
        let status = report.status;
        set_last_error(report);
        return status.code();
    };
    let meta = api.meta();

    run_with_status(meta, || {
        if summary_out.is_null() {
            return Err(CintxErrorReport::null_pointer(
                meta.symbol,
                meta.family,
                meta.representation,
                "summary_out",
            ));
        }
        if out.is_null() && out_len > 0 {
            return Err(CintxErrorReport::null_pointer(
                meta.symbol,
                meta.family,
                meta.representation,
                "out",
            ));
        }
        if cache.is_null() && cache_len > 0 {
            return Err(CintxErrorReport::null_pointer(
                meta.symbol,
                meta.family,
                meta.representation,
                "cache",
            ));
        }
        if !out.is_null() && !cache.is_null() && std::ptr::eq(out, cache) {
            return Err(invalid_input_report(
                meta,
                "`out` and `cache` pointers must not alias",
            ));
        }

        let shls = unsafe { required_slice(shls, shls_len, "shls", meta)? };
        let atm = unsafe { required_slice(atm, atm_len, "atm", meta)? };
        let bas = unsafe { required_slice(bas, bas_len, "bas", meta)? };
        let env = unsafe { required_slice(env, env_len, "env", meta)? };
        let dims = unsafe { optional_slice(dims, dims_len) };
        let out = unsafe { optional_slice_mut(out, out_len) };
        let cache = unsafe { optional_slice_mut(cache, cache_len) };
        let opt = unsafe { opt.as_ref() };

        let summary = unsafe { eval_raw(meta.id, out, dims, shls, atm, bas, env, opt, cache) }
            .map_err(|error| map_core_error(meta, &error))?;

        let ffi_summary = CintxEvalSummary {
            not0: summary.not0,
            bytes_written: summary.bytes_written,
            workspace_bytes: summary.workspace_bytes,
        };
        // SAFETY: `summary_out` is checked non-null above and points to caller-owned storage.
        unsafe {
            ptr::write(summary_out, ffi_summary);
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{
        cintrs_last_error_code, clear_last_error, current_last_error, CintxStatus,
    };

    struct RawFixture {
        shls_2: [i32; 2],
        atm: Vec<i32>,
        bas: Vec<i32>,
        env: Vec<f64>,
    }

    impl RawFixture {
        fn single_atom_two_shells() -> Self {
            let env = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.9, 0.8];
            let atm = vec![
                1, // charge
                0, // ptr_coord
                1, // point nucleus
                0, // ptr_zeta
                0, // ptr_frac_charge
                0,
            ];
            let bas = vec![
                0, 0, 1, 1, 0, 3, 4, 0, // shell 0
                0, 1, 1, 1, 0, 5, 6, 0, // shell 1
            ];
            Self {
                shls_2: [0, 1],
                atm,
                bas,
                env,
            }
        }
    }

    #[test]
    fn query_and_eval_wrappers_succeed_and_clear_tls_error() {
        clear_last_error();
        let fixture = RawFixture::single_atom_two_shells();
        let mut query = CintxWorkspaceQuery::default();
        let query_status = unsafe {
            cintrs_query_workspace(
                CintxRawApi::Int1eOvlpCart as i32,
                ptr::null(),
                0,
                fixture.shls_2.as_ptr(),
                fixture.shls_2.len(),
                fixture.atm.as_ptr(),
                fixture.atm.len(),
                fixture.bas.as_ptr(),
                fixture.bas.len(),
                fixture.env.as_ptr(),
                fixture.env.len(),
                ptr::null(),
                &mut query,
            )
        };

        assert_eq!(query_status, 0);
        assert_eq!(cintrs_last_error_code(), CintxStatus::Success.code());
        assert!(query.bytes > 0);

        let mut out = vec![1.0; 3];
        let mut summary = CintxEvalSummary::default();
        let eval_status = unsafe {
            cintrs_eval(
                CintxRawApi::Int1eOvlpCart as i32,
                out.as_mut_ptr(),
                out.len(),
                ptr::null(),
                0,
                fixture.shls_2.as_ptr(),
                fixture.shls_2.len(),
                fixture.atm.as_ptr(),
                fixture.atm.len(),
                fixture.bas.as_ptr(),
                fixture.bas.len(),
                fixture.env.as_ptr(),
                fixture.env.len(),
                ptr::null(),
                ptr::null_mut(),
                0,
                &mut summary,
            )
        };

        assert_eq!(eval_status, 0);
        assert!(summary.bytes_written > 0);
        assert!(out.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn invalid_dims_maps_to_nonzero_status_and_tls_report() {
        clear_last_error();
        let fixture = RawFixture::single_atom_two_shells();
        let dims = [1i32];
        let mut query = CintxWorkspaceQuery::default();
        let status = unsafe {
            cintrs_query_workspace(
                CintxRawApi::Int1eOvlpCart as i32,
                dims.as_ptr(),
                dims.len(),
                fixture.shls_2.as_ptr(),
                fixture.shls_2.len(),
                fixture.atm.as_ptr(),
                fixture.atm.len(),
                fixture.bas.as_ptr(),
                fixture.bas.len(),
                fixture.env.as_ptr(),
                fixture.env.len(),
                ptr::null(),
                &mut query,
            )
        };

        assert_eq!(status, CintxStatus::InvalidInput.code());
        let report = current_last_error();
        assert_eq!(report.status, CintxStatus::InvalidInput);
        assert!(report.message.contains("invalid dims"));
    }

    #[test]
    fn panic_boundary_maps_panics_to_panic_status() {
        clear_last_error();
        let meta = CintxRawApi::Int2eSph.meta();
        let status = run_with_status(meta, || -> Result<(), CintxErrorReport> {
            panic!("shim panic mapping")
        });

        assert_eq!(status, CintxStatus::Panic.code());
        let report = current_last_error();
        assert_eq!(report.status, CintxStatus::Panic);
        assert!(report.message.contains("panic in C ABI shim"));
    }

    #[test]
    fn invalid_api_id_sets_tls_report() {
        clear_last_error();
        let mut query = CintxWorkspaceQuery::default();
        let status = unsafe {
            cintrs_query_workspace(
                999,
                ptr::null(),
                0,
                ptr::null(),
                0,
                ptr::null(),
                0,
                ptr::null(),
                0,
                ptr::null(),
                0,
                ptr::null(),
                &mut query,
            )
        };

        assert_eq!(status, CintxStatus::UnsupportedApi.code());
        let report = current_last_error();
        assert_eq!(report.status, CintxStatus::UnsupportedApi);
        assert!(report.message.contains("unsupported C ABI api id 999"));
    }

    #[test]
    fn eval_rejects_null_out_pointer_when_out_len_is_nonzero() {
        clear_last_error();
        let fixture = RawFixture::single_atom_two_shells();
        let mut summary = CintxEvalSummary::default();
        let status = unsafe {
            cintrs_eval(
                CintxRawApi::Int1eOvlpCart as i32,
                ptr::null_mut(),
                3,
                ptr::null(),
                0,
                fixture.shls_2.as_ptr(),
                fixture.shls_2.len(),
                fixture.atm.as_ptr(),
                fixture.atm.len(),
                fixture.bas.as_ptr(),
                fixture.bas.len(),
                fixture.env.as_ptr(),
                fixture.env.len(),
                ptr::null(),
                ptr::null_mut(),
                0,
                &mut summary,
            )
        };

        assert_eq!(status, CintxStatus::NullPointer.code());
        let report = current_last_error();
        assert_eq!(report.status, CintxStatus::NullPointer);
        assert!(report.message.contains("null pointer for required parameter `out`"));
        assert_eq!(summary, CintxEvalSummary::default());
    }

    #[test]
    fn tls_error_state_isolated_across_threads() {
        clear_last_error();
        let fixture = RawFixture::single_atom_two_shells();
        let dims = [1i32];
        let mut query = CintxWorkspaceQuery::default();
        let main_status = unsafe {
            cintrs_query_workspace(
                CintxRawApi::Int1eOvlpCart as i32,
                dims.as_ptr(),
                dims.len(),
                fixture.shls_2.as_ptr(),
                fixture.shls_2.len(),
                fixture.atm.as_ptr(),
                fixture.atm.len(),
                fixture.bas.as_ptr(),
                fixture.bas.len(),
                fixture.env.as_ptr(),
                fixture.env.len(),
                ptr::null(),
                &mut query,
            )
        };
        assert_eq!(main_status, CintxStatus::InvalidInput.code());
        let main_message = current_last_error().message;

        let fixture = RawFixture::single_atom_two_shells();
        let worker = std::thread::spawn(move || {
            assert_eq!(cintrs_last_error_code(), CintxStatus::Success.code());
            let mut worker_query = CintxWorkspaceQuery::default();
            let status = unsafe {
                cintrs_query_workspace(
                    CintxRawApi::Int1eOvlpCart as i32,
                    ptr::null(),
                    0,
                    ptr::null(),
                    fixture.shls_2.len(),
                    fixture.atm.as_ptr(),
                    fixture.atm.len(),
                    fixture.bas.as_ptr(),
                    fixture.bas.len(),
                    fixture.env.as_ptr(),
                    fixture.env.len(),
                    ptr::null(),
                    &mut worker_query,
                )
            };
            (status, current_last_error())
        });

        let (worker_status, worker_report) = worker.join().expect("worker should not panic");
        assert_eq!(worker_status, CintxStatus::NullPointer.code());
        assert!(worker_report.message.contains("null pointer"));
        assert_eq!(current_last_error().message, main_message);
    }
}
