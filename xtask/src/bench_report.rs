use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const FALLBACK_ARTIFACT_DIR_ENV: &str = "CINTX_ARTIFACT_DIR";
const FALLBACK_ARTIFACT_DIR_DEFAULT: &str = "/tmp/cintx_artifacts";
const REQUIRED_BENCH_REPORT_PATH: &str = "/tmp/cintx_artifacts/cintx_phase_04_bench_report.json";
const BENCH_REPORT_FALLBACK_NAME: &str = "cintx_phase_04_bench_report.json";
const REQUIRED_RUNTIME_DIAGNOSTICS_PATH: &str =
    "/tmp/cintx_artifacts/cintx_phase_04_runtime_diagnostics.json";
const RUNTIME_DIAGNOSTICS_FALLBACK_NAME: &str = "cintx_phase_04_runtime_diagnostics.json";
const REQUIRED_BENCH_ROWS_PATH: &str = "/tmp/cintx_artifacts/cintx_phase_04_bench_rows.jsonl";
const BENCH_ROWS_FILE_NAME: &str = "cintx_phase_04_bench_rows.jsonl";
const SUITE_IDS: [&str; 3] = ["micro_families", "macro_molecules", "crossover_cpu_gpu"];
const PHASE0_ARTIFACTS: [(&str, &str); 7] = [
    (
        "baseline",
        "/tmp/cintx_artifacts/cintx_cubecl_baseline.json",
    ),
    ("profile", "/tmp/cintx_artifacts/cintx_cubecl_profile.jsonl"),
    (
        "autotune",
        "/tmp/cintx_artifacts/cintx_cubecl_autotune.json",
    ),
    (
        "speed_report",
        "/tmp/cintx_artifacts/cintx_cubecl_speed_report.json",
    ),
    (
        "memory_report",
        "/tmp/cintx_artifacts/cintx_cubecl_memory_report.json",
    ),
    (
        "oracle_summary",
        "/tmp/cintx_artifacts/cintx_cubecl_oracle_summary.json",
    ),
    (
        "unverified_matrix",
        "/tmp/cintx_artifacts/cintx_cubecl_unverified_matrix.json",
    ),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReportMode {
    Calibration,
    Enforce,
}

impl ReportMode {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "calibration" => Ok(Self::Calibration),
            "enforce" => Ok(Self::Enforce),
            other => anyhow::bail!("unsupported bench-report mode `{other}`"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Calibration => "calibration",
            Self::Enforce => "enforce",
        }
    }
}

#[derive(Clone, Debug)]
struct SuiteThresholds {
    baseline_throughput: f64,
    baseline_workspace_bytes: usize,
    baseline_transfer_bytes: usize,
    baseline_crossover_shift_pct: Option<f64>,
    throughput_regression_pct: f64,
    memory_regression_pct: f64,
    transfer_regression_pct: Option<f64>,
    crossover_shift_pct: Option<f64>,
}

#[derive(Clone, Debug)]
struct ThresholdConfig {
    suites: BTreeMap<String, SuiteThresholds>,
}

#[derive(Clone, Debug)]
struct BenchRow {
    suite_id: String,
    case_id: String,
    profile: String,
    throughput: Option<f64>,
    workspace_bytes: Option<usize>,
    transfer_bytes: Option<usize>,
    not0: Option<i32>,
    pack_ns: Option<u64>,
    submit_ns: Option<u64>,
    readback_ns: Option<u64>,
    crossover_shift_pct: Option<f64>,
    source: String,
}

#[derive(Clone, Debug, Default)]
struct SuiteAggregate {
    row_count: usize,
    throughput_sum: f64,
    throughput_count: usize,
    workspace_peak: usize,
    transfer_sum: usize,
    transfer_count: usize,
    not0_sum: i64,
    pack_ns_sum: u128,
    pack_ns_count: usize,
    submit_ns_sum: u128,
    submit_ns_count: usize,
    readback_ns_sum: u128,
    readback_ns_count: usize,
    crossover_shift_sum: f64,
    crossover_shift_count: usize,
    profiles: BTreeSet<String>,
    sources: BTreeSet<String>,
}

#[derive(Clone, Debug)]
struct SuiteMeasurement {
    suite_id: String,
    profile: String,
    source: String,
    sample_count: usize,
    throughput: f64,
    workspace_bytes: usize,
    transfer_bytes: usize,
    not0: i32,
    host_timing_ns: Option<HostTimingNs>,
    chunk_count: usize,
    fallback_reason: Option<String>,
    crossover_shift_pct: Option<f64>,
}

/// Mean control-plane timings from warm batch-pilot metric rows.
///
/// They are portable host wall-clock intervals, not GPU timestamps.
#[derive(Clone, Copy, Debug)]
struct HostTimingNs {
    pack: u64,
    submit: u64,
    readback: u64,
}

#[derive(Clone, Debug)]
struct SuiteEvaluation {
    throughput_regression_pct: f64,
    memory_regression_pct: f64,
    transfer_regression_pct: Option<f64>,
    crossover_shift_delta_pct: Option<f64>,
    exceeded_reasons: Vec<String>,
}

pub fn run_bench_report(thresholds_path: &str, mode: &str) -> Result<()> {
    let mode = ReportMode::parse(mode)?;
    let thresholds = read_thresholds(Path::new(thresholds_path))?;
    let (rows, row_sources) = load_bench_rows()?;
    let aggregates = aggregate_rows(&rows);

    let mut suite_reports = Vec::new();
    let mut diagnostics_rows = Vec::new();
    let mut threshold_exceedances = Vec::new();

    for suite_id in SUITE_IDS {
        let Some(suite_thresholds) = thresholds.suites.get(suite_id) else {
            anyhow::bail!("threshold config missing suite `{suite_id}`");
        };
        let aggregate = aggregates.get(suite_id);
        let measurement = build_suite_measurement(suite_id, suite_thresholds, aggregate);
        let evaluation = evaluate_suite_regression(&measurement, suite_thresholds);
        if !evaluation.exceeded_reasons.is_empty() {
            for reason in &evaluation.exceeded_reasons {
                threshold_exceedances.push(format!("{suite_id}: {reason}"));
            }
        }

        suite_reports.push(json!({
            "suite_id": measurement.suite_id,
            "profile": measurement.profile,
            "source": measurement.source,
            "sample_count": measurement.sample_count,
            "measured": {
                "throughput": measurement.throughput,
                "workspace_bytes": measurement.workspace_bytes,
                "transfer_bytes": measurement.transfer_bytes,
                "not0": measurement.not0,
                "host_timing_ns": measurement.host_timing_ns.map(|timing| json!({
                    "kind": "host_wall_clock_not_device_timestamp",
                    "pack": timing.pack,
                    "submit": timing.submit,
                    "readback": timing.readback,
                })),
                "crossover_shift_pct": measurement.crossover_shift_pct,
            },
            "baseline": {
                "throughput": suite_thresholds.baseline_throughput,
                "workspace_bytes": suite_thresholds.baseline_workspace_bytes,
                "transfer_bytes": suite_thresholds.baseline_transfer_bytes,
                "crossover_shift_pct": suite_thresholds.baseline_crossover_shift_pct,
            },
            "thresholds": {
                "throughput_regression_pct": suite_thresholds.throughput_regression_pct,
                "memory_regression_pct": suite_thresholds.memory_regression_pct,
                "transfer_regression_pct": suite_thresholds.transfer_regression_pct,
                "crossover_shift_pct": suite_thresholds.crossover_shift_pct,
            },
            "regression": {
                "throughput_regression_pct": evaluation.throughput_regression_pct,
                "memory_regression_pct": evaluation.memory_regression_pct,
                "transfer_regression_pct": evaluation.transfer_regression_pct,
                "crossover_shift_delta_pct": evaluation.crossover_shift_delta_pct,
            },
            "threshold_exceeded": !evaluation.exceeded_reasons.is_empty(),
            "exceeded_reasons": evaluation.exceeded_reasons,
        }));

        diagnostics_rows.push(json!({
            "suite_id": suite_id,
            "profile": measurement.profile,
            "chunk_count": measurement.chunk_count,
            "fallback_reason": measurement.fallback_reason,
            "transfer_bytes": measurement.transfer_bytes,
            "not0": measurement.not0,
            "host_timing_ns": measurement.host_timing_ns.map(|timing| json!({
                "kind": "host_wall_clock_not_device_timestamp",
                "pack": timing.pack,
                "submit": timing.submit,
                "readback": timing.readback,
            })),
            "workspace_bytes": measurement.workspace_bytes,
            "source": measurement.source,
        }));
    }

    let status = if mode == ReportMode::Enforce && !threshold_exceedances.is_empty() {
        "failed"
    } else {
        "ok"
    };

    let mut bench_report = json!({
        "mode": mode.as_str(),
        "status": status,
        "thresholds_path": thresholds_path,
        "required_path": REQUIRED_BENCH_REPORT_PATH,
        "row_sources": row_sources,
        "policy": {
            "fail_condition": "regression threshold exceeded",
            "allow_slowdowns_within_threshold": true,
        },
        "suites": suite_reports,
        "exceeded": threshold_exceedances,
    });
    let bench_write = write_json_with_fallback(
        REQUIRED_BENCH_REPORT_PATH,
        BENCH_REPORT_FALLBACK_NAME,
        &bench_report,
    )?;
    bench_report["artifact_write"] = bench_write.to_json();
    rewrite_json(&bench_write.actual_path, &bench_report)?;

    let mut diagnostics_report = json!({
        "mode": mode.as_str(),
        "status": status,
        "required_path": REQUIRED_RUNTIME_DIAGNOSTICS_PATH,
        "contract_source": "crates/cintx-runtime/src/metrics.rs",
        "diagnostics": diagnostics_rows,
        "bench_report_path": bench_write.actual_path.display().to_string(),
    });
    let diagnostics_write = write_json_with_fallback(
        REQUIRED_RUNTIME_DIAGNOSTICS_PATH,
        RUNTIME_DIAGNOSTICS_FALLBACK_NAME,
        &diagnostics_report,
    )?;
    diagnostics_report["artifact_write"] = diagnostics_write.to_json();
    rewrite_json(&diagnostics_write.actual_path, &diagnostics_report)?;

    let phase0_writes = write_phase0_artifacts(
        mode,
        thresholds_path,
        &row_sources,
        &suite_reports,
        &diagnostics_rows,
        &bench_write,
        &diagnostics_write,
    )?;

    println!(
        "bench report artifact: {}",
        bench_write.actual_path.display()
    );
    println!(
        "runtime diagnostics artifact: {}",
        diagnostics_write.actual_path.display()
    );
    for (kind, path) in phase0_writes {
        println!("phase 0 {kind} artifact: {}", path.display());
    }

    if mode == ReportMode::Enforce && !threshold_exceedances.is_empty() {
        anyhow::bail!(
            "benchmark threshold exceedances detected: {}",
            threshold_exceedances.join(" | ")
        );
    }

    Ok(())
}

/// Emits the Phase 0 artifact set from the benchmark rows already consumed by
/// this command.  These artifacts intentionally do not manufacture GPU
/// adapter information, device timestamps, tuning choices, or oracle results:
/// the current benchmark contract has none of those inputs.
fn write_phase0_artifacts(
    mode: ReportMode,
    thresholds_path: &str,
    row_sources: &[String],
    suite_reports: &[Value],
    diagnostics_rows: &[Value],
    bench_write: &ArtifactWrite,
    diagnostics_write: &ArtifactWrite,
) -> Result<Vec<(&'static str, PathBuf)>> {
    let provenance = phase0_provenance(
        mode,
        thresholds_path,
        row_sources,
        bench_write,
        diagnostics_write,
    );
    let suite_rows = phase0_suite_rows(suite_reports);
    let profile_rows = suite_rows
        .iter()
        .map(|suite| {
            json!({
                "schema_version": 1,
                "artifact": "cintx_cubecl_profile",
                "provenance": provenance,
                "suite": suite,
            })
        })
        .collect::<Vec<_>>();

    let artifacts = [
        (
            "baseline",
            json!({
                "schema_version": 1,
                "artifact": "cintx_cubecl_baseline",
                "provenance": provenance,
                "measurement_scope": "aggregate benchmark rows; threshold fallbacks are labelled per suite",
                "suites": suite_rows,
            }),
        ),
        (
            "autotune",
            json!({
                "schema_version": 1,
                "artifact": "cintx_cubecl_autotune",
                "provenance": provenance,
                "status": "not_collected",
                "reason": "the current benchmark/report contract records no tuner candidates, selected configuration, or device identity",
                "device_timing": { "status": "not_collected", "reason": "host wall-clock metrics are not device timestamps" },
            }),
        ),
        (
            "speed_report",
            json!({
                "schema_version": 1,
                "artifact": "cintx_cubecl_speed_report",
                "provenance": provenance,
                "metric_kind": "benchmark throughput derived from Criterion elapsed-time estimates when present",
                "not_device_timing": true,
                "suites": suite_rows,
            }),
        ),
        (
            "memory_report",
            json!({
                "schema_version": 1,
                "artifact": "cintx_cubecl_memory_report",
                "provenance": provenance,
                "metric_kind": "reported workspace and transfer byte counters",
                "not_device_memory_telemetry": true,
                "diagnostics": diagnostics_rows,
            }),
        ),
        (
            "oracle_summary",
            json!({
                "schema_version": 1,
                "artifact": "cintx_cubecl_oracle_summary",
                "provenance": provenance,
                "status": "not_run",
                "reason": "bench-report consumes benchmark rows only; no oracle-comparison result was supplied to this command",
                "verified_scope": [],
                "unverified_scope": ["all benchmarked suites require a separate oracle-compare invocation"],
            }),
        ),
        (
            "unverified_matrix",
            json!({
                "schema_version": 1,
                "artifact": "cintx_cubecl_unverified_matrix",
                "provenance": provenance,
                "backend_matrix": unverified_backend_matrix(),
            }),
        ),
    ];

    let mut writes = Vec::new();
    for ((kind, required_path), (_, value)) in PHASE0_ARTIFACTS
        .iter()
        .filter(|(kind, _)| *kind != "profile")
        .zip(artifacts)
    {
        debug_assert_eq!(*kind, artifacts_kind(&value));
        let write = write_json_with_fallback(
            required_path,
            &format!("{}.json", value["artifact"].as_str().unwrap_or(kind)),
            &value,
        )?;
        writes.push((*kind, write.actual_path));
    }

    let profile_path = PHASE0_ARTIFACTS
        .iter()
        .find_map(|(kind, path)| (*kind == "profile").then_some(*path))
        .expect("Phase 0 profile artifact path is declared");
    write_jsonl(profile_path, &profile_rows)?;
    writes.push(("profile", PathBuf::from(profile_path)));
    Ok(writes)
}

fn phase0_provenance(
    mode: ReportMode,
    thresholds_path: &str,
    row_sources: &[String],
    bench_write: &ArtifactWrite,
    diagnostics_write: &ArtifactWrite,
) -> Value {
    json!({
        "report_mode": mode.as_str(),
        "command": format!(
            "cargo run --locked --manifest-path xtask/Cargo.toml -- bench-report --thresholds {thresholds_path} --mode {}",
            mode.as_str(),
        ),
        "xtask_version": env!("CARGO_PKG_VERSION"),
        "git_revision": env::var("CINTX_GIT_REVISION").ok(),
        "benchmark_row_sources": row_sources,
        "bench_report": bench_write.to_json(),
        "runtime_diagnostics": diagnostics_write.to_json(),
        "limitations": [
            "No GPU adapter/device identity was recorded by benchmark rows.",
            "pack_ns, submit_ns, and readback_ns are host wall-clock intervals, not device timestamps.",
            "No device-memory telemetry, autotuning output, or oracle-comparison result was supplied to bench-report.",
        ],
    })
}

fn phase0_suite_rows(suite_reports: &[Value]) -> Vec<Value> {
    suite_reports
        .iter()
        .map(|suite| {
            let source = suite
                .get("source")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            json!({
                "suite_id": suite.get("suite_id").cloned().unwrap_or(Value::Null),
                "profile": suite.get("profile").cloned().unwrap_or(Value::Null),
                "metric_origin": if source == "threshold_baseline" {
                    "threshold_configuration_fallback_not_a_measurement"
                } else {
                    "observed_benchmark_rows"
                },
                "source": source,
                "sample_count": suite.get("sample_count").cloned().unwrap_or(Value::Null),
                "measured": suite.get("measured").cloned().unwrap_or(Value::Null),
                "baseline": suite.get("baseline").cloned().unwrap_or(Value::Null),
            })
        })
        .collect()
}

fn unverified_backend_matrix() -> Vec<Value> {
    ["cpu", "cuda", "rocm", "wgpu", "metal"]
        .into_iter()
        .map(|backend| {
            json!({
                "backend": backend,
                "status": "unverified",
                "reason": "bench-report has no backend-specific adapter provenance, device timing, or oracle result",
            })
        })
        .collect()
}

fn artifacts_kind(value: &Value) -> &str {
    value
        .get("artifact")
        .and_then(Value::as_str)
        .and_then(|artifact| artifact.strip_prefix("cintx_cubecl_"))
        .unwrap_or("unknown")
}

fn write_jsonl(path: &str, values: &[Value]) -> Result<()> {
    let payload = values
        .iter()
        .map(serde_json::to_string)
        .collect::<std::result::Result<Vec<_>, _>>()?
        .join("\n");
    let payload = if payload.is_empty() {
        payload
    } else {
        format!("{payload}\n")
    };
    try_write_payload(Path::new(path), payload.as_bytes())
}

fn read_thresholds(path: &Path) -> Result<ThresholdConfig> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("read benchmark thresholds `{}`", path.display()))?;
    let root: Value = serde_json::from_str(&raw)
        .with_context(|| format!("parse benchmark thresholds `{}`", path.display()))?;

    let mut suites = BTreeMap::new();
    for suite_id in SUITE_IDS {
        let suite_value = root
            .get(suite_id)
            .with_context(|| format!("threshold config missing `{suite_id}` object"))?;
        suites.insert(
            suite_id.to_owned(),
            parse_suite_thresholds(suite_id, suite_value)?,
        );
    }

    Ok(ThresholdConfig { suites })
}

fn parse_suite_thresholds(suite_id: &str, value: &Value) -> Result<SuiteThresholds> {
    let object = value
        .as_object()
        .with_context(|| format!("suite `{suite_id}` thresholds must be an object"))?;

    let baseline_throughput = required_f64(object, suite_id, "baseline_throughput")?;
    let baseline_workspace_bytes = required_u64(object, suite_id, "baseline_workspace_bytes")?;
    let baseline_transfer_bytes = required_u64(object, suite_id, "baseline_transfer_bytes")?;
    let throughput_regression_pct = required_f64(object, suite_id, "throughput_regression_pct")?;
    let memory_regression_pct = required_f64(object, suite_id, "memory_regression_pct")?;
    let transfer_regression_pct = optional_f64(object, "transfer_regression_pct");

    let baseline_crossover_shift_pct = optional_f64(object, "baseline_crossover_shift_pct");
    let crossover_shift_pct = optional_f64(object, "crossover_shift_pct");

    if suite_id == "crossover_cpu_gpu"
        && (baseline_crossover_shift_pct.is_none() || crossover_shift_pct.is_none())
    {
        anyhow::bail!(
            "suite `{suite_id}` must define baseline_crossover_shift_pct and crossover_shift_pct"
        );
    }

    Ok(SuiteThresholds {
        baseline_throughput,
        baseline_workspace_bytes,
        baseline_transfer_bytes,
        baseline_crossover_shift_pct,
        throughput_regression_pct,
        memory_regression_pct,
        transfer_regression_pct,
        crossover_shift_pct,
    })
}

fn required_f64(object: &serde_json::Map<String, Value>, suite_id: &str, key: &str) -> Result<f64> {
    object
        .get(key)
        .and_then(Value::as_f64)
        .with_context(|| format!("suite `{suite_id}` missing numeric `{key}`"))
}

fn required_u64(
    object: &serde_json::Map<String, Value>,
    suite_id: &str,
    key: &str,
) -> Result<usize> {
    let value = object
        .get(key)
        .and_then(Value::as_u64)
        .with_context(|| format!("suite `{suite_id}` missing integer `{key}`"))?;
    usize::try_from(value).map_err(|_| anyhow!("suite `{suite_id}` `{key}` exceeds usize"))
}

fn optional_f64(object: &serde_json::Map<String, Value>, key: &str) -> Option<f64> {
    object.get(key).and_then(Value::as_f64)
}

fn load_bench_rows() -> Result<(Vec<BenchRow>, Vec<String>)> {
    let mut rows = Vec::new();
    let mut sources = Vec::new();

    for candidate in bench_row_candidates() {
        if !candidate.is_file() {
            continue;
        }
        let source = format!("jsonl:{}", candidate.display());
        let parsed = parse_jsonl_rows(&candidate, &source)?;
        if !parsed.is_empty() {
            sources.push(source);
            rows.extend(parsed);
        }
    }

    let criterion_rows = collect_criterion_rows()?;
    if !criterion_rows.is_empty() {
        sources.push("criterion:target/criterion/**/estimates.json".to_owned());
        rows.extend(criterion_rows);
    }

    if sources.is_empty() {
        sources.push("threshold_baseline".to_owned());
    }

    Ok((rows, sources))
}

fn bench_row_candidates() -> Vec<PathBuf> {
    let mut candidates = vec![PathBuf::from(REQUIRED_BENCH_ROWS_PATH)];
    candidates.push(fallback_dir().join(BENCH_ROWS_FILE_NAME));

    if let Ok(path) = env::var("CINTX_BENCH_ROWS_PATH") {
        candidates.push(PathBuf::from(path));
    }

    if let Ok(path) = env::var(FALLBACK_ARTIFACT_DIR_ENV) {
        candidates.push(Path::new(&path).join(BENCH_ROWS_FILE_NAME));
    }

    candidates
}

fn parse_jsonl_rows(path: &Path, source: &str) -> Result<Vec<BenchRow>> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("read benchmark rows `{}`", path.display()))?;
    let mut rows = Vec::new();

    for (line_index, raw_line) in content.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line).with_context(|| {
            format!(
                "parse benchmark row line {} from `{}`",
                line_index + 1,
                path.display()
            )
        })?;
        if let Some(row) = parse_bench_row(&value, source)? {
            rows.push(row);
        }
    }

    Ok(rows)
}

fn parse_bench_row(value: &Value, source: &str) -> Result<Option<BenchRow>> {
    let suite_id = value
        .get("suite_id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !SUITE_IDS.contains(&suite_id) {
        return Ok(None);
    }

    let case_id = value
        .get("case_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let profile = value
        .get("profile")
        .and_then(Value::as_str)
        .unwrap_or("base");
    let throughput = value.get("throughput").and_then(Value::as_f64);
    let workspace_bytes = value
        .get("workspace_bytes")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok());
    let transfer_bytes = value
        .get("transfer_bytes")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok());
    let not0 = value
        .get("not0")
        .and_then(Value::as_i64)
        .and_then(|value| i32::try_from(value).ok());
    let pack_ns = value.get("pack_ns").and_then(Value::as_u64);
    let submit_ns = value.get("submit_ns").and_then(Value::as_u64);
    let readback_ns = value.get("readback_ns").and_then(Value::as_u64);
    let crossover_shift_pct = value.get("crossover_shift_pct").and_then(Value::as_f64);

    Ok(Some(BenchRow {
        suite_id: suite_id.to_owned(),
        case_id: case_id.to_owned(),
        profile: profile.to_owned(),
        throughput,
        workspace_bytes,
        transfer_bytes,
        not0,
        pack_ns,
        submit_ns,
        readback_ns,
        crossover_shift_pct,
        source: source.to_owned(),
    }))
}

fn collect_criterion_rows() -> Result<Vec<BenchRow>> {
    let root = Path::new("target/criterion");
    if !root.is_dir() {
        return Ok(Vec::new());
    }

    let mut estimate_paths = Vec::new();
    collect_named_files(root, "estimates.json", &mut estimate_paths)?;

    let mut rows = Vec::new();
    for estimate_path in estimate_paths {
        let relative = match estimate_path.strip_prefix(root) {
            Ok(path) => path,
            Err(_) => continue,
        };
        let components: Vec<String> = relative
            .components()
            .map(|component| component.as_os_str().to_string_lossy().to_string())
            .collect();
        if components.len() < 3 {
            continue;
        }
        // Criterion also writes `base/` and `change/` estimate files. The
        // latter are percentage-change statistics, not elapsed nanoseconds;
        // only `new/estimates.json` is a measurement for this invocation.
        if components
            .get(components.len().saturating_sub(2))
            .is_none_or(|run_kind| run_kind != "new")
        {
            continue;
        }
        let suite_id = components[0].clone();
        if !SUITE_IDS.contains(&suite_id.as_str()) {
            continue;
        }
        let case_components = &components[1..components.len().saturating_sub(2)];
        let case_id = if case_components.is_empty() {
            "criterion".to_owned()
        } else {
            case_components.join("/")
        };

        let content = fs::read_to_string(&estimate_path)
            .with_context(|| format!("read criterion estimates `{}`", estimate_path.display()))?;
        let value: Value = serde_json::from_str(&content)
            .with_context(|| format!("parse criterion estimates `{}`", estimate_path.display()))?;
        let point_estimate_ns = value
            .get("mean")
            .and_then(Value::as_object)
            .and_then(|mean| mean.get("point_estimate"))
            .and_then(Value::as_f64)
            .unwrap_or_default();
        if point_estimate_ns <= 0.0 {
            continue;
        }

        let throughput =
            criterion_work_units(&estimate_path)? * 1_000_000_000.0 / point_estimate_ns;
        rows.push(BenchRow {
            suite_id,
            case_id,
            profile: env::var("CINTX_BENCH_PROFILE").unwrap_or_else(|_| "criterion".to_owned()),
            throughput: Some(throughput),
            workspace_bytes: None,
            transfer_bytes: None,
            not0: None,
            pack_ns: None,
            submit_ns: None,
            readback_ns: None,
            crossover_shift_pct: None,
            source: format!("criterion:{}", estimate_path.display()),
        });
    }

    Ok(rows)
}

/// Criterion stores elapsed time in `estimates.json` and the work unit selected
/// by the benchmark in its adjacent `benchmark.json`. Respect the latter so an
/// `Elements` benchmark reports elements/second rather than invocations/second.
fn criterion_work_units(estimate_path: &Path) -> Result<f64> {
    let Some(benchmark_path) = estimate_path
        .parent()
        .map(|parent| parent.join("benchmark.json"))
    else {
        return Ok(1.0);
    };
    if !benchmark_path.is_file() {
        return Ok(1.0);
    }

    let content = fs::read_to_string(&benchmark_path)
        .with_context(|| format!("read criterion metadata `{}`", benchmark_path.display()))?;
    let value: Value = serde_json::from_str(&content)
        .with_context(|| format!("parse criterion metadata `{}`", benchmark_path.display()))?;
    let Some(throughput) = value.get("throughput").and_then(Value::as_object) else {
        return Ok(1.0);
    };
    let Some(work_units) = throughput.values().next().and_then(Value::as_u64) else {
        return Ok(1.0);
    };
    Ok(work_units as f64)
}

fn collect_named_files(root: &Path, name: &str, out: &mut Vec<PathBuf>) -> Result<()> {
    if !root.is_dir() {
        return Ok(());
    }

    for entry in
        fs::read_dir(root).with_context(|| format!("read directory `{}`", root.display()))?
    {
        let entry =
            entry.with_context(|| format!("list directory entry under `{}`", root.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_named_files(&path, name, out)?;
        } else if path
            .file_name()
            .and_then(|file| file.to_str())
            .is_some_and(|file| file == name)
        {
            out.push(path);
        }
    }

    Ok(())
}

fn aggregate_rows(rows: &[BenchRow]) -> BTreeMap<String, SuiteAggregate> {
    let mut aggregates = BTreeMap::new();
    for row in rows {
        if !SUITE_IDS.contains(&row.suite_id.as_str()) {
            continue;
        }
        let aggregate = aggregates
            .entry(row.suite_id.clone())
            .or_insert_with(SuiteAggregate::default);
        aggregate.row_count = aggregate.row_count.saturating_add(1);
        if let Some(throughput) = row.throughput.filter(|throughput| *throughput > 0.0) {
            aggregate.throughput_sum += throughput;
            aggregate.throughput_count = aggregate.throughput_count.saturating_add(1);
        }
        if let Some(workspace_bytes) = row.workspace_bytes {
            aggregate.workspace_peak = aggregate.workspace_peak.max(workspace_bytes);
        }
        if let Some(transfer_bytes) = row.transfer_bytes {
            aggregate.transfer_sum = aggregate.transfer_sum.saturating_add(transfer_bytes);
            aggregate.transfer_count = aggregate.transfer_count.saturating_add(1);
        }
        if let Some(not0) = row.not0 {
            aggregate.not0_sum = aggregate.not0_sum.saturating_add(i64::from(not0));
        }
        if let Some(pack_ns) = row.pack_ns.filter(|value| *value > 0) {
            aggregate.pack_ns_sum = aggregate.pack_ns_sum.saturating_add(u128::from(pack_ns));
            aggregate.pack_ns_count = aggregate.pack_ns_count.saturating_add(1);
        }
        if let Some(submit_ns) = row.submit_ns.filter(|value| *value > 0) {
            aggregate.submit_ns_sum = aggregate
                .submit_ns_sum
                .saturating_add(u128::from(submit_ns));
            aggregate.submit_ns_count = aggregate.submit_ns_count.saturating_add(1);
        }
        if let Some(readback_ns) = row.readback_ns.filter(|value| *value > 0) {
            aggregate.readback_ns_sum = aggregate
                .readback_ns_sum
                .saturating_add(u128::from(readback_ns));
            aggregate.readback_ns_count = aggregate.readback_ns_count.saturating_add(1);
        }
        if let Some(shift) = row.crossover_shift_pct {
            aggregate.crossover_shift_sum += shift;
            aggregate.crossover_shift_count = aggregate.crossover_shift_count.saturating_add(1);
        }
        aggregate.profiles.insert(row.profile.clone());
        aggregate
            .sources
            .insert(format!("{}#{}", row.source, row.case_id));
    }
    aggregates
}

fn build_suite_measurement(
    suite_id: &str,
    thresholds: &SuiteThresholds,
    aggregate: Option<&SuiteAggregate>,
) -> SuiteMeasurement {
    if let Some(aggregate) = aggregate {
        if aggregate.row_count > 0 {
            let profile = if aggregate.profiles.len() == 1 {
                aggregate
                    .profiles
                    .iter()
                    .next()
                    .cloned()
                    .unwrap_or_else(|| "base".to_owned())
            } else {
                "mixed".to_owned()
            };
            let source = aggregate
                .sources
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(",");
            let throughput = if aggregate.throughput_count > 0 {
                aggregate.throughput_sum / aggregate.throughput_count as f64
            } else {
                thresholds.baseline_throughput
            };
            let transfer_bytes = if aggregate.transfer_count > 0 {
                aggregate.transfer_sum / aggregate.transfer_count
            } else {
                thresholds.baseline_transfer_bytes
            };
            let crossover_shift_pct = if aggregate.crossover_shift_count > 0 {
                Some(aggregate.crossover_shift_sum / aggregate.crossover_shift_count as f64)
            } else {
                thresholds.baseline_crossover_shift_pct
            };
            let host_timing_ns = (aggregate.pack_ns_count > 0
                && aggregate.submit_ns_count > 0
                && aggregate.readback_ns_count > 0)
                .then(|| HostTimingNs {
                    pack: average_ns(aggregate.pack_ns_sum, aggregate.pack_ns_count),
                    submit: average_ns(aggregate.submit_ns_sum, aggregate.submit_ns_count),
                    readback: average_ns(aggregate.readback_ns_sum, aggregate.readback_ns_count),
                });

            return SuiteMeasurement {
                suite_id: suite_id.to_owned(),
                profile,
                source,
                sample_count: aggregate.row_count,
                throughput,
                workspace_bytes: aggregate.workspace_peak,
                transfer_bytes,
                not0: aggregate
                    .not0_sum
                    .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32,
                host_timing_ns,
                chunk_count: aggregate.row_count,
                fallback_reason: None,
                crossover_shift_pct,
            };
        }
    }

    SuiteMeasurement {
        suite_id: suite_id.to_owned(),
        profile: "base".to_owned(),
        source: "threshold_baseline".to_owned(),
        sample_count: 0,
        throughput: thresholds.baseline_throughput,
        workspace_bytes: thresholds.baseline_workspace_bytes,
        transfer_bytes: thresholds.baseline_transfer_bytes,
        not0: 0,
        host_timing_ns: None,
        chunk_count: 0,
        fallback_reason: Some("missing_bench_rows".to_owned()),
        crossover_shift_pct: thresholds.baseline_crossover_shift_pct,
    }
}

fn average_ns(sum: u128, count: usize) -> u64 {
    let count = u128::try_from(count).unwrap_or(u128::MAX).max(1);
    u64::try_from(sum / count).unwrap_or(u64::MAX)
}

fn evaluate_suite_regression(
    measurement: &SuiteMeasurement,
    thresholds: &SuiteThresholds,
) -> SuiteEvaluation {
    let throughput_regression_pct = percent_regression_when_lower_is_worse(
        thresholds.baseline_throughput,
        measurement.throughput,
    );
    let memory_regression_pct = percent_regression_when_higher_is_worse(
        thresholds.baseline_workspace_bytes as f64,
        measurement.workspace_bytes as f64,
    );
    let transfer_regression_pct = thresholds.transfer_regression_pct.map(|_| {
        percent_regression_when_higher_is_worse(
            thresholds.baseline_transfer_bytes as f64,
            measurement.transfer_bytes as f64,
        )
    });
    let crossover_shift_delta_pct = match (
        thresholds.baseline_crossover_shift_pct,
        measurement.crossover_shift_pct,
    ) {
        (Some(baseline), Some(measured)) => Some((measured - baseline).abs()),
        _ => None,
    };

    let mut exceeded_reasons = Vec::new();
    if throughput_regression_pct > thresholds.throughput_regression_pct {
        exceeded_reasons.push(format!(
            "throughput regression {:.3}% exceeded threshold {:.3}%",
            throughput_regression_pct, thresholds.throughput_regression_pct
        ));
    }
    if memory_regression_pct > thresholds.memory_regression_pct {
        exceeded_reasons.push(format!(
            "memory regression {:.3}% exceeded threshold {:.3}%",
            memory_regression_pct, thresholds.memory_regression_pct
        ));
    }
    if let (Some(limit), Some(actual)) =
        (thresholds.transfer_regression_pct, transfer_regression_pct)
    {
        if actual > limit {
            exceeded_reasons.push(format!(
                "transfer regression {:.3}% exceeded threshold {:.3}%",
                actual, limit
            ));
        }
    }
    if let (Some(limit), Some(actual)) = (thresholds.crossover_shift_pct, crossover_shift_delta_pct)
    {
        if actual > limit {
            exceeded_reasons.push(format!(
                "crossover shift {:.3}% exceeded threshold {:.3}%",
                actual, limit
            ));
        }
    }

    SuiteEvaluation {
        throughput_regression_pct,
        memory_regression_pct,
        transfer_regression_pct,
        crossover_shift_delta_pct,
        exceeded_reasons,
    }
}

fn percent_regression_when_lower_is_worse(baseline: f64, measured: f64) -> f64 {
    if baseline <= 0.0 || measured >= baseline {
        return 0.0;
    }
    ((baseline - measured) / baseline) * 100.0
}

fn percent_regression_when_higher_is_worse(baseline: f64, measured: f64) -> f64 {
    if baseline <= 0.0 || measured <= baseline {
        return 0.0;
    }
    ((measured - baseline) / baseline) * 100.0
}

fn write_json_with_fallback(
    required_path: &str,
    fallback_name: &str,
    value: &Value,
) -> Result<ArtifactWrite> {
    let payload =
        serde_json::to_vec_pretty(value).context("serialize bench-report artifact json")?;
    write_bytes_with_fallback(required_path, fallback_name, &payload)
}

fn write_bytes_with_fallback(
    required_path: &str,
    fallback_name: &str,
    payload: &[u8],
) -> Result<ArtifactWrite> {
    let required = PathBuf::from(required_path);
    match try_write_payload(&required, payload) {
        Ok(()) => Ok(ArtifactWrite {
            required_path: required_path.to_owned(),
            actual_path: required,
            used_required_path: true,
            fallback_reason: None,
        }),
        Err(error) => {
            let fallback = fallback_dir().join(fallback_name);
            try_write_payload(&fallback, payload).with_context(|| {
                format!(
                    "failed to write fallback artifact `{}` after required-path failure",
                    fallback.display()
                )
            })?;
            Ok(ArtifactWrite {
                required_path: required_path.to_owned(),
                actual_path: fallback,
                used_required_path: false,
                fallback_reason: Some(error.to_string()),
            })
        }
    }
}

fn try_write_payload(path: &Path, payload: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create artifact parent directory `{}`", parent.display()))?;
    }
    fs::write(path, payload).with_context(|| format!("write artifact `{}`", path.display()))?;
    Ok(())
}

fn rewrite_json(path: &Path, value: &Value) -> Result<()> {
    let payload = serde_json::to_vec_pretty(value).context("serialize final artifact json")?;
    fs::write(path, payload).with_context(|| format!("rewrite artifact `{}`", path.display()))?;
    Ok(())
}

fn fallback_dir() -> PathBuf {
    env::var(FALLBACK_ARTIFACT_DIR_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(FALLBACK_ARTIFACT_DIR_DEFAULT))
}

#[derive(Clone, Debug)]
struct ArtifactWrite {
    required_path: String,
    actual_path: PathBuf,
    used_required_path: bool,
    fallback_reason: Option<String>,
}

impl ArtifactWrite {
    fn to_json(&self) -> Value {
        json!({
            "required_path": self.required_path,
            "actual_path": self.actual_path.display().to_string(),
            "used_required_path": self.used_required_path,
            "fallback_reason": self.fallback_reason,
            "fallback_env_var": FALLBACK_ARTIFACT_DIR_ENV,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase0_suite_rows_label_threshold_fallbacks_as_unmeasured() {
        let rows = phase0_suite_rows(&[
            json!({
                "suite_id": "micro_families",
                "profile": "base",
                "source": "threshold_baseline",
                "sample_count": 0,
                "measured": { "throughput": 1.0 },
                "baseline": { "throughput": 1.0 },
            }),
            json!({
                "suite_id": "macro_molecules",
                "profile": "criterion",
                "source": "jsonl:/tmp/rows.jsonl#case",
                "sample_count": 2,
                "measured": { "throughput": 2.0 },
                "baseline": { "throughput": 1.0 },
            }),
        ]);

        assert_eq!(
            rows[0]["metric_origin"],
            "threshold_configuration_fallback_not_a_measurement"
        );
        assert_eq!(rows[1]["metric_origin"], "observed_benchmark_rows");
        assert_eq!(rows[1]["measured"]["throughput"], 2.0);
    }

    #[test]
    fn phase0_backend_matrix_makes_no_verified_backend_claim() {
        let matrix = unverified_backend_matrix();
        let backends = matrix
            .iter()
            .map(|entry| entry["backend"].as_str().unwrap_or_default())
            .collect::<BTreeSet<_>>();

        assert_eq!(
            backends,
            BTreeSet::from(["cpu", "cuda", "metal", "rocm", "wgpu"])
        );
        assert!(matrix.iter().all(|entry| entry["status"] == "unverified"));
        assert!(matrix.iter().all(|entry| {
            entry["reason"]
                .as_str()
                .is_some_and(|reason| reason.contains("device timing"))
        }));
    }

    #[test]
    fn phase0_artifact_manifest_has_all_required_names() {
        let paths = PHASE0_ARTIFACTS
            .iter()
            .map(|(_, path)| *path)
            .collect::<BTreeSet<_>>();

        assert_eq!(paths.len(), 7);
        assert!(
            paths
                .iter()
                .all(|path| path.starts_with("/tmp/cintx_artifacts/cintx_cubecl_"))
        );
        assert!(paths.contains("/tmp/cintx_artifacts/cintx_cubecl_profile.jsonl"));
    }
}
