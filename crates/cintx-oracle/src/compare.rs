use crate::fixtures::{
    ArtifactWriteResult, OracleFixture, OracleRawInputs, PHASE2_FAMILIES,
    REPORT_ARTIFACT_FALLBACK_NAME, REQUIRED_MATRIX_ARTIFACT, REQUIRED_REPORT_ARTIFACT,
    build_phase2_representation_matrix, write_pretty_json_artifact,
    write_representation_matrix_artifact,
};
use anyhow::{Context, Result, anyhow, bail};
use cintx_compat::helpers::{
    CINTcgto_cart, CINTcgto_spheric, CINTcgto_spinor, CINTcgtos_cart, CINTcgtos_spheric,
    CINTcgtos_spinor, CINTgto_norm, CINTlen_cart, CINTlen_spinor, CINTshells_cart_offset,
    CINTshells_spheric_offset, CINTshells_spinor_offset, CINTtot_cgto_cart, CINTtot_cgto_spheric,
    CINTtot_cgto_spinor, CINTtot_pgto_spheric, CINTtot_pgto_spinor,
};
use cintx_compat::legacy;
use cintx_compat::optimizer;
use cintx_compat::raw::{self, RawApiId};
use cintx_ops::resolver::{HelperKind, Resolver};
use serde_json::json;
use std::collections::BTreeSet;

const TOL_1E_ATOL: f64 = 1e-11;
const TOL_1E_RTOL: f64 = 1e-9;
const TOL_2E_ATOL: f64 = 1e-12;
const TOL_2E_RTOL: f64 = 1e-10;
const TOL_2C2E_3C2E_ATOL: f64 = 1e-9;
const TOL_2C2E_3C2E_RTOL: f64 = 1e-7;
const TOL_3C1E_ATOL: f64 = 1e-7;
const TOL_3C1E_RTOL: f64 = 1e-5;
const ZERO_THRESHOLD: f64 = 1e-18;

const IMPLEMENTED_HELPER_SYMBOLS: &[&str] = &[
    "CINTlen_cart",
    "CINTlen_spinor",
    "CINTcgtos_cart",
    "CINTcgtos_spheric",
    "CINTcgtos_spinor",
    "CINTcgto_cart",
    "CINTcgto_spheric",
    "CINTcgto_spinor",
    "CINTtot_pgto_spheric",
    "CINTtot_pgto_spinor",
    "CINTtot_cgto_cart",
    "CINTtot_cgto_spheric",
    "CINTtot_cgto_spinor",
    "CINTshells_cart_offset",
    "CINTshells_spheric_offset",
    "CINTshells_spinor_offset",
    "CINTgto_norm",
];

const IMPLEMENTED_TRANSFORM_SYMBOLS: &[&str] = &[
    "CINTc2s_bra_sph",
    "CINTc2s_ket_sph",
    "CINTc2s_ket_sph1",
    "CINTc2s_ket_spinor_sf1",
    "CINTc2s_iket_spinor_sf1",
    "CINTc2s_ket_spinor_si1",
    "CINTc2s_iket_spinor_si1",
];

const IMPLEMENTED_OPTIMIZER_SYMBOLS: &[&str] = &[
    "CINTinit_2e_optimizer",
    "CINTinit_optimizer",
    "CINTdel_2e_optimizer",
    "CINTdel_optimizer",
    "int2e_cart_optimizer",
    "int2e_sph_optimizer",
    "int2e_optimizer",
];

#[derive(Clone, Copy, Debug)]
pub struct FamilyTolerance {
    pub family: &'static str,
    pub atol: f64,
    pub rtol: f64,
    pub zero_threshold: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct DiffSummary {
    pub max_abs_error: f64,
    pub max_rel_error: f64,
    pub within_tolerance: bool,
}

#[derive(Clone, Debug)]
pub struct FixtureParityResult {
    pub symbol: String,
    pub family: String,
    pub representation: String,
    pub tolerance: FamilyTolerance,
    pub raw_vs_upstream: DiffSummary,
    pub raw_vs_optimizer: DiffSummary,
    pub layout_ok: bool,
}

#[derive(Clone, Debug)]
pub struct Phase2ParityReport {
    pub helper_surface_ok: bool,
    pub matrix_artifact: ArtifactWriteResult,
    pub parity_artifact: ArtifactWriteResult,
    pub fixtures: Vec<FixtureParityResult>,
}

pub fn tolerance_for_family(family: &str) -> Result<FamilyTolerance> {
    let tolerance = match family {
        "1e" => FamilyTolerance {
            family: "1e",
            atol: TOL_1E_ATOL,
            rtol: TOL_1E_RTOL,
            zero_threshold: ZERO_THRESHOLD,
        },
        "2e" => FamilyTolerance {
            family: "2e",
            atol: TOL_2E_ATOL,
            rtol: TOL_2E_RTOL,
            zero_threshold: ZERO_THRESHOLD,
        },
        "2c2e" | "3c2e" => FamilyTolerance {
            family: if family == "2c2e" { "2c2e" } else { "3c2e" },
            atol: TOL_2C2E_3C2E_ATOL,
            rtol: TOL_2C2E_3C2E_RTOL,
            zero_threshold: ZERO_THRESHOLD,
        },
        "3c1e" => FamilyTolerance {
            family: "3c1e",
            atol: TOL_3C1E_ATOL,
            rtol: TOL_3C1E_RTOL,
            zero_threshold: ZERO_THRESHOLD,
        },
        other => bail!("missing family tolerance for `{other}`"),
    };
    Ok(tolerance)
}

fn diff_summary(reference: &[f64], observed: &[f64], tolerance: FamilyTolerance) -> DiffSummary {
    if reference.len() != observed.len() {
        return DiffSummary {
            max_abs_error: f64::INFINITY,
            max_rel_error: f64::INFINITY,
            within_tolerance: false,
        };
    }

    let mut max_abs_error = 0.0_f64;
    let mut max_rel_error = 0.0_f64;
    let mut within_tolerance = true;

    for (ref_value, observed_value) in reference.iter().zip(observed.iter()) {
        let abs_error = (observed_value - ref_value).abs();
        max_abs_error = max_abs_error.max(abs_error);

        let abs_ref = ref_value.abs();
        let rel_error = if abs_ref > 0.0 {
            abs_error / abs_ref
        } else {
            0.0
        };
        max_rel_error = max_rel_error.max(rel_error);

        let passed = if abs_ref < tolerance.zero_threshold {
            abs_error <= tolerance.atol
        } else {
            abs_error <= tolerance.atol + tolerance.rtol * abs_ref
        };
        within_tolerance &= passed;
    }

    DiffSummary {
        max_abs_error,
        max_rel_error,
        within_tolerance,
    }
}

fn assert_flat_buffer_contract(fixture: &OracleFixture, values: &[f64]) -> bool {
    if values.len() != fixture.required_elements() {
        return false;
    }
    if !values.iter().all(|value| value.is_finite()) {
        return false;
    }

    if fixture.representation == "spinor" {
        if !fixture.complex_interleaved || values.len() % 2 != 0 {
            return false;
        }
        return values
            .chunks_exact(2)
            .all(|pair| pair[0].is_finite() && pair[1].is_finite());
    }

    true
}

fn raw_api_for_symbol(symbol: &str) -> Option<RawApiId> {
    match symbol {
        "int1e_ovlp_cart" => Some(RawApiId::INT1E_OVLP_CART),
        "int1e_ovlp_sph" => Some(RawApiId::INT1E_OVLP_SPH),
        "int1e_ovlp_spinor" => Some(RawApiId::INT1E_OVLP_SPINOR),
        "int1e_kin_cart" => Some(RawApiId::INT1E_KIN_CART),
        "int1e_kin_sph" => Some(RawApiId::INT1E_KIN_SPH),
        "int1e_kin_spinor" => Some(RawApiId::INT1E_KIN_SPINOR),
        "int1e_nuc_cart" => Some(RawApiId::INT1E_NUC_CART),
        "int1e_nuc_sph" => Some(RawApiId::INT1E_NUC_SPH),
        "int1e_nuc_spinor" => Some(RawApiId::INT1E_NUC_SPINOR),
        "int2e_cart" => Some(RawApiId::INT2E_CART),
        "int2e_sph" => Some(RawApiId::INT2E_SPH),
        "int2e_spinor" => Some(RawApiId::INT2E_SPINOR),
        "int2c2e_cart" => Some(RawApiId::INT2C2E_CART),
        "int2c2e_sph" => Some(RawApiId::INT2C2E_SPH),
        "int2c2e_spinor" => Some(RawApiId::INT2C2E_SPINOR),
        "int3c1e_p2_cart" => Some(RawApiId::INT3C1E_P2_CART),
        "int3c1e_p2_sph" => Some(RawApiId::INT3C1E_P2_SPH),
        "int3c2e_ip1_cart" => Some(RawApiId::INT3C2E_IP1_CART),
        "int3c2e_ip1_sph" => Some(RawApiId::INT3C2E_IP1_SPH),
        "int3c2e_ip1_spinor" => Some(RawApiId::INT3C2E_IP1_SPINOR),
        _ => None,
    }
}

unsafe fn eval_legacy_symbol(
    symbol: &str,
    out: &mut [f64],
    shls: &[i32],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> Result<raw::RawEvalSummary> {
    let result = match symbol {
        "int1e_ovlp_cart" => unsafe {
            legacy::cint1e_ovlp_cart(Some(out), shls, atm, bas, env, None)
        },
        "int1e_ovlp_sph" => unsafe {
            legacy::cint1e_ovlp_sph(Some(out), shls, atm, bas, env, None)
        },
        "int1e_ovlp_spinor" => unsafe { legacy::cint1e_ovlp(Some(out), shls, atm, bas, env, None) },
        "int1e_kin_cart" => unsafe { legacy::cint1e_kin_cart(Some(out), shls, atm, bas, env) },
        "int1e_kin_sph" => unsafe { legacy::cint1e_kin_sph(Some(out), shls, atm, bas, env) },
        "int1e_kin_spinor" => unsafe { legacy::cint1e_kin(Some(out), shls, atm, bas, env) },
        "int1e_nuc_cart" => unsafe {
            legacy::cint1e_nuc_cart(Some(out), shls, atm, bas, env, None)
        },
        "int1e_nuc_sph" => unsafe { legacy::cint1e_nuc_sph(Some(out), shls, atm, bas, env, None) },
        "int1e_nuc_spinor" => unsafe { legacy::cint1e_nuc(Some(out), shls, atm, bas, env, None) },
        "int2e_cart" => unsafe { legacy::cint2e_cart(Some(out), shls, atm, bas, env, None) },
        "int2e_sph" => unsafe { legacy::cint2e_sph(Some(out), shls, atm, bas, env, None) },
        "int2e_spinor" => unsafe { legacy::cint2e(Some(out), shls, atm, bas, env, None) },
        "int2c2e_cart" => unsafe { legacy::cint2c2e_cart(Some(out), shls, atm, bas, env, None) },
        "int2c2e_sph" => unsafe { legacy::cint2c2e_sph(Some(out), shls, atm, bas, env, None) },
        "int2c2e_spinor" => unsafe { legacy::cint2c2e(Some(out), shls, atm, bas, env, None) },
        "int3c1e_p2_cart" => unsafe {
            legacy::cint3c1e_p2_cart(Some(out), shls, atm, bas, env, None)
        },
        "int3c1e_p2_sph" => unsafe {
            legacy::cint3c1e_p2_sph(Some(out), shls, atm, bas, env, None)
        },
        "int3c2e_ip1_cart" => unsafe {
            legacy::cint3c2e_ip1_cart(Some(out), shls, atm, bas, env, None)
        },
        "int3c2e_ip1_sph" => unsafe {
            legacy::cint3c2e_ip1_sph(Some(out), shls, atm, bas, env, None)
        },
        "int3c2e_ip1_spinor" => unsafe {
            legacy::cint3c2e_ip1(Some(out), shls, atm, bas, env, None)
        },
        other => bail!("missing legacy wrapper mapping for `{other}`"),
    };
    result.map_err(anyhow::Error::from)
}

pub fn verify_helper_surface_coverage(inputs: &OracleRawInputs) -> Result<()> {
    let expected_helpers: BTreeSet<String> = Resolver::entries_by_kind(HelperKind::Helper)
        .into_iter()
        .filter(|entry| entry.compiled_in_profiles.contains(&"base"))
        .map(|entry| entry.symbol_name.to_owned())
        .collect();
    let actual_helpers: BTreeSet<String> = IMPLEMENTED_HELPER_SYMBOLS
        .iter()
        .map(|symbol| (*symbol).to_owned())
        .collect();
    if actual_helpers != expected_helpers {
        bail!("helper surface drift detected");
    }

    let expected_transforms: BTreeSet<String> = Resolver::entries_by_kind(HelperKind::Transform)
        .into_iter()
        .filter(|entry| entry.compiled_in_profiles.contains(&"base"))
        .map(|entry| entry.symbol_name.to_owned())
        .collect();
    let actual_transforms: BTreeSet<String> = IMPLEMENTED_TRANSFORM_SYMBOLS
        .iter()
        .map(|symbol| (*symbol).to_owned())
        .collect();
    if actual_transforms != expected_transforms {
        bail!("transform surface drift detected");
    }

    let expected_optimizers: BTreeSet<String> = Resolver::entries_by_kind(HelperKind::Optimizer)
        .into_iter()
        .filter(|entry| entry.compiled_in_profiles.contains(&"base"))
        .map(|entry| entry.symbol_name.to_owned())
        .collect();
    let actual_optimizers: BTreeSet<String> = IMPLEMENTED_OPTIMIZER_SYMBOLS
        .iter()
        .map(|symbol| (*symbol).to_owned())
        .collect();
    if actual_optimizers != expected_optimizers {
        bail!("optimizer surface drift detected");
    }

    // Smoke helper/transform/optimizer APIs to ensure the parity harness calls
    // the same public surface the compat layer exports.
    let _ = CINTlen_cart(2)?;
    let _ = CINTlen_spinor(0, &inputs.bas)?;
    let _ = CINTcgtos_cart(0, &inputs.bas)?;
    let _ = CINTcgtos_spheric(1, &inputs.bas)?;
    let _ = CINTcgtos_spinor(1, &inputs.bas)?;
    let _ = CINTcgto_cart(0, &inputs.bas)?;
    let _ = CINTcgto_spheric(1, &inputs.bas)?;
    let _ = CINTcgto_spinor(1, &inputs.bas)?;
    let _ = CINTtot_pgto_spheric(&inputs.bas, 4)?;
    let _ = CINTtot_pgto_spinor(&inputs.bas, 4)?;
    let _ = CINTtot_cgto_cart(&inputs.bas, 4)?;
    let _ = CINTtot_cgto_spheric(&inputs.bas, 4)?;
    let _ = CINTtot_cgto_spinor(&inputs.bas, 4)?;

    let mut offsets = vec![0_i32; 5];
    CINTshells_cart_offset(&mut offsets, &inputs.bas, 4)?;
    CINTshells_spheric_offset(&mut offsets, &inputs.bas, 4)?;
    CINTshells_spinor_offset(&mut offsets, &inputs.bas, 4)?;
    if CINTgto_norm(1, 0.5) <= 0.0 {
        bail!("CINTgto_norm should return a positive value for positive exponent");
    }

    let mut sph = vec![0.0, 0.0, 0.0, 0.0];
    cintx_compat::transform::CINTc2s_bra_sph(&mut sph, 1, &[1.0, 2.0, 3.0, 4.0], 1)?;
    cintx_compat::transform::CINTc2s_ket_sph(&mut sph, 1, &[1.0, 2.0, 3.0, 4.0], 1)?;
    cintx_compat::transform::CINTc2s_ket_sph1(&mut sph, &[1.0, 2.0, 3.0, 4.0], 0, 0, 1)?;
    let mut spinor = vec![0.0, 0.0, 0.0, 0.0];
    cintx_compat::transform::CINTc2s_ket_spinor_sf1(
        &mut spinor,
        &[1.0, 2.0, 3.0, 4.0],
        0,
        0,
        1,
        1,
        0,
    )?;
    cintx_compat::transform::CINTc2s_iket_spinor_sf1(
        &mut spinor,
        &[1.0, 2.0, 3.0, 4.0],
        0,
        0,
        1,
        1,
        0,
    )?;
    cintx_compat::transform::CINTc2s_ket_spinor_si1(
        &mut spinor,
        &[1.0, 2.0, 3.0, 4.0],
        0,
        0,
        1,
        1,
        0,
    )?;
    cintx_compat::transform::CINTc2s_iket_spinor_si1(
        &mut spinor,
        &[1.0, 2.0, 3.0, 4.0],
        0,
        0,
        1,
        1,
        0,
    )?;

    let mut opt = Some(optimizer::CINTinit_optimizer(
        &inputs.atm,
        &inputs.bas,
        &inputs.env,
    )?);
    optimizer::CINTdel_optimizer(&mut opt);
    let mut opt2 = Some(optimizer::CINTinit_2e_optimizer(
        &inputs.atm,
        &inputs.bas,
        &inputs.env,
    )?);
    optimizer::CINTdel_2e_optimizer(&mut opt2);

    let _ = optimizer::int2e_cart_optimizer(&inputs.atm, &inputs.bas, &inputs.env)?;
    let _ = optimizer::int2e_sph_optimizer(&inputs.atm, &inputs.bas, &inputs.env)?;
    let _ = optimizer::int2e_optimizer(&inputs.atm, &inputs.bas, &inputs.env)?;

    Ok(())
}

pub fn generate_phase2_parity_report(inputs: &OracleRawInputs) -> Result<Phase2ParityReport> {
    verify_helper_surface_coverage(inputs)?;

    let matrix = build_phase2_representation_matrix(inputs)?;
    let matrix_symbols: BTreeSet<String> = matrix
        .iter()
        .map(|fixture| fixture.symbol.clone())
        .collect();
    let expected_symbols: BTreeSet<String> = Resolver::manifest()
        .iter()
        .filter(|entry| {
            matches!(entry.helper_kind, HelperKind::Operator)
                && entry.compiled_in_profiles.contains(&"base")
                && PHASE2_FAMILIES.contains(&entry.family_name)
        })
        .map(|entry| entry.symbol_name.to_owned())
        .collect();
    if matrix_symbols != expected_symbols {
        bail!("representation matrix drifted from manifest-derived symbol set");
    }

    let matrix_artifact = write_representation_matrix_artifact(&matrix)?;
    let mut fixture_results = Vec::new();
    let mut report_rows = Vec::new();

    for fixture in &matrix {
        let api = raw_api_for_symbol(&fixture.symbol)
            .ok_or_else(|| anyhow!("missing raw api mapping for `{}`", fixture.symbol))?;
        let shls = inputs.shells_for_arity(fixture.arity);
        let dims_i32: Vec<i32> = fixture
            .dims
            .iter()
            .copied()
            .map(i32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()
            .with_context(|| format!("dims overflow i32 for `{}`", fixture.symbol))?;
        let tolerance = tolerance_for_family(&fixture.family)?;

        let query = unsafe {
            raw::query_workspace_raw(
                api,
                Some(&dims_i32),
                shls,
                &inputs.atm,
                &inputs.bas,
                &inputs.env,
                None,
            )
        }
        .with_context(|| format!("workspace query for `{}`", fixture.symbol))?;

        let required_elements = fixture.required_elements();
        let mut raw_out = vec![f64::NAN; required_elements];
        let raw_summary = unsafe {
            raw::eval_raw(
                api,
                Some(&mut raw_out),
                Some(&dims_i32),
                shls,
                &inputs.atm,
                &inputs.bas,
                &inputs.env,
                None,
                None,
            )
        }
        .with_context(|| format!("raw eval for `{}`", fixture.symbol))?;

        let mut upstream_out = vec![f64::NAN; required_elements];
        let upstream_summary = unsafe {
            eval_legacy_symbol(
                &fixture.symbol,
                &mut upstream_out,
                shls,
                &inputs.atm,
                &inputs.bas,
                &inputs.env,
            )
        }
        .with_context(|| format!("legacy upstream proxy eval for `{}`", fixture.symbol))?;

        let raw_vs_upstream = diff_summary(&upstream_out, &raw_out, tolerance);
        if !raw_vs_upstream.within_tolerance {
            bail!("raw/upstream parity failed for `{}`", fixture.symbol);
        }

        let layout_ok = assert_flat_buffer_contract(fixture, &raw_out)
            && assert_flat_buffer_contract(fixture, &upstream_out);
        if !layout_ok {
            bail!(
                "flat-buffer/interleaved layout assertion failed for `{}`",
                fixture.symbol
            );
        }

        let opt_handle = optimizer::CINTinit_optimizer(&inputs.atm, &inputs.bas, &inputs.env)?;
        let mut optimized_out = vec![f64::NAN; required_elements];
        let optimized_summary = unsafe {
            raw::eval_raw(
                api,
                Some(&mut optimized_out),
                Some(&dims_i32),
                shls,
                &inputs.atm,
                &inputs.bas,
                &inputs.env,
                Some(&opt_handle),
                None,
            )
        }
        .with_context(|| format!("optimized raw eval for `{}`", fixture.symbol))?;
        let raw_vs_optimizer = diff_summary(&raw_out, &optimized_out, tolerance);
        if !raw_vs_optimizer.within_tolerance {
            bail!("optimizer on/off parity failed for `{}`", fixture.symbol);
        }

        let result = FixtureParityResult {
            symbol: fixture.symbol.clone(),
            family: fixture.family.clone(),
            representation: fixture.representation.clone(),
            tolerance,
            raw_vs_upstream,
            raw_vs_optimizer,
            layout_ok,
        };
        fixture_results.push(result.clone());
        report_rows.push(json!({
            "symbol": result.symbol,
            "family": result.family,
            "representation": result.representation,
            "workspace_bytes": query.bytes,
            "raw_summary": {
                "not0": raw_summary.not0,
                "bytes_written": raw_summary.bytes_written,
                "workspace_bytes": raw_summary.workspace_bytes,
            },
            "upstream_summary": {
                "not0": upstream_summary.not0,
                "bytes_written": upstream_summary.bytes_written,
                "workspace_bytes": upstream_summary.workspace_bytes,
            },
            "optimized_summary": {
                "not0": optimized_summary.not0,
                "bytes_written": optimized_summary.bytes_written,
                "workspace_bytes": optimized_summary.workspace_bytes,
            },
            "tolerance": {
                "family": result.tolerance.family,
                "atol": result.tolerance.atol,
                "rtol": result.tolerance.rtol,
                "zero_threshold": result.tolerance.zero_threshold,
            },
            "raw_vs_upstream": {
                "max_abs_error": result.raw_vs_upstream.max_abs_error,
                "max_rel_error": result.raw_vs_upstream.max_rel_error,
                "within_tolerance": result.raw_vs_upstream.within_tolerance,
            },
            "raw_vs_optimizer": {
                "max_abs_error": result.raw_vs_optimizer.max_abs_error,
                "max_rel_error": result.raw_vs_optimizer.max_rel_error,
                "within_tolerance": result.raw_vs_optimizer.within_tolerance,
            },
            "layout_assertions": {
                "flat-buffer_contract": result.layout_ok,
                "spinor_interleaved_doubles": fixture.complex_interleaved,
            },
        }));
    }

    let report_json = json!({
        "required_path": REQUIRED_REPORT_ARTIFACT,
        "required_matrix_path": REQUIRED_MATRIX_ARTIFACT,
        "helper_parity": {
            "status": "pass",
            "helper_symbols": IMPLEMENTED_HELPER_SYMBOLS,
            "transform_symbols": IMPLEMENTED_TRANSFORM_SYMBOLS,
            "optimizer_symbols": IMPLEMENTED_OPTIMIZER_SYMBOLS,
        },
        "tolerance_table": {
            "1e": {"atol": TOL_1E_ATOL, "rtol": TOL_1E_RTOL, "zero_threshold": ZERO_THRESHOLD},
            "2e": {"atol": TOL_2E_ATOL, "rtol": TOL_2E_RTOL, "zero_threshold": ZERO_THRESHOLD},
            "2c2e": {"atol": TOL_2C2E_3C2E_ATOL, "rtol": TOL_2C2E_3C2E_RTOL, "zero_threshold": ZERO_THRESHOLD},
            "3c2e": {"atol": TOL_2C2E_3C2E_ATOL, "rtol": TOL_2C2E_3C2E_RTOL, "zero_threshold": ZERO_THRESHOLD},
            "3c1e": {"atol": TOL_3C1E_ATOL, "rtol": TOL_3C1E_RTOL, "zero_threshold": ZERO_THRESHOLD},
        },
        "upstream_reference": "vendored upstream compatibility proxy through cintx_compat::legacy wrappers",
        "cart_spheric_spinor_flat-buffer_interleaved_assertions": true,
        "representation_matrix": matrix_artifact.actual_path.display().to_string(),
        "results": report_rows,
    });

    let parity_artifact = write_pretty_json_artifact(
        REQUIRED_REPORT_ARTIFACT,
        REPORT_ARTIFACT_FALLBACK_NAME,
        &report_json,
    )?;

    Ok(Phase2ParityReport {
        helper_surface_ok: true,
        matrix_artifact,
        parity_artifact,
        fixtures: fixture_results,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helper_coverage_matches_manifest() {
        let inputs = OracleRawInputs::sample();
        verify_helper_surface_coverage(&inputs).expect("helper parity");
    }

    #[test]
    fn evaluated_output_parity_and_optimizer_equivalence_hold() {
        let inputs = OracleRawInputs::sample();
        let report = generate_phase2_parity_report(&inputs).expect("parity report");
        assert!(report.helper_surface_ok);
        assert!(
            report
                .fixtures
                .iter()
                .all(|fixture| fixture.raw_vs_upstream.within_tolerance),
            "raw vs upstream parity must hold for all base-family fixtures"
        );
        assert!(
            report
                .fixtures
                .iter()
                .all(|fixture| fixture.raw_vs_optimizer.within_tolerance),
            "optimizer on/off parity must hold for all base-family fixtures"
        );
        assert!(
            report.fixtures.iter().all(|fixture| fixture.layout_ok),
            "flat-buffer and interleaved layout assertions must hold"
        );
    }

    #[test]
    fn parity_artifacts_are_written() {
        let inputs = OracleRawInputs::sample();
        let report = generate_phase2_parity_report(&inputs).expect("parity report");
        assert!(report.matrix_artifact.actual_path.is_file());
        assert!(report.parity_artifact.actual_path.is_file());
    }
}
