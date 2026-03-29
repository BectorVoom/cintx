use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const FALLBACK_ARTIFACT_DIR_ENV: &str = "CINTX_ARTIFACT_DIR";
const FALLBACK_ARTIFACT_DIR_DEFAULT: &str = "/tmp/cintx_artifacts";
const REQUIRED_BENCH_REPORT_PATH: &str = "/mnt/data/cintx_phase_04_bench_report.json";
const BENCH_REPORT_FALLBACK_NAME: &str = "cintx_phase_04_bench_report.json";
const REQUIRED_RUNTIME_DIAGNOSTICS_PATH: &str = "/mnt/data/cintx_phase_04_runtime_diagnostics.json";
const RUNTIME_DIAGNOSTICS_FALLBACK_NAME: &str = "cintx_phase_04_runtime_diagnostics.json";
const REQUIRED_BENCH_ROWS_PATH: &str = "/mnt/data/cintx_phase_04_bench_rows.jsonl";
const BENCH_ROWS_FILE_NAME: &str = "cintx_phase_04_bench_rows.jsonl";
const SUITE_IDS: [&str; 3] = ["micro_families", "macro_molecules", "crossover_cpu_gpu"];

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
    throughput: f64,
    workspace_bytes: usize,
    transfer_bytes: usize,
    not0: i32,
    crossover_shift_pct: Option<f64>,
    source: String,
}

#[derive(Clone, Debug, Default)]
struct SuiteAggregate {
    row_count: usize,
    throughput_sum: f64,
    workspace_peak: usize,
    transfer_sum: usize,
    not0_sum: i64,
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
    chunk_count: usize,
    fallback_reason: Option<String>,
    crossover_shift_pct: Option<f64>,
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

    println!("bench report artifact: {}", bench_write.actual_path.display());
    println!(
        "runtime diagnostics artifact: {}",
        diagnostics_write.actual_path.display()
    );

    if mode == ReportMode::Enforce && !threshold_exceedances.is_empty() {
        anyhow::bail!(
            "benchmark threshold exceedances detected: {}",
            threshold_exceedances.join(" | ")
        );
    }

    Ok(())
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

fn required_f64(
    object: &serde_json::Map<String, Value>,
    suite_id: &str,
    key: &str,
) -> Result<f64> {
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
    let content =
        fs::read_to_string(path).with_context(|| format!("read benchmark rows `{}`", path.display()))?;
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
    let throughput = value
        .get("throughput")
        .and_then(Value::as_f64)
        .unwrap_or_default();
    let workspace_bytes = value
        .get("workspace_bytes")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let transfer_bytes = value
        .get("transfer_bytes")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let not0 = value.get("not0").and_then(Value::as_i64).unwrap_or_default();
    let not0 = i32::try_from(not0).unwrap_or(i32::MAX);
    let crossover_shift_pct = value.get("crossover_shift_pct").and_then(Value::as_f64);

    Ok(Some(BenchRow {
        suite_id: suite_id.to_owned(),
        case_id: case_id.to_owned(),
        profile: profile.to_owned(),
        throughput,
        workspace_bytes: usize::try_from(workspace_bytes).unwrap_or(usize::MAX),
        transfer_bytes: usize::try_from(transfer_bytes).unwrap_or(usize::MAX),
        not0,
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

        let throughput = 1_000_000_000.0 / point_estimate_ns;
        rows.push(BenchRow {
            suite_id,
            case_id,
            profile: env::var("CINTX_BENCH_PROFILE").unwrap_or_else(|_| "criterion".to_owned()),
            throughput,
            workspace_bytes: 0,
            transfer_bytes: 0,
            not0: 1,
            crossover_shift_pct: None,
            source: format!("criterion:{}", estimate_path.display()),
        });
    }

    Ok(rows)
}

fn collect_named_files(root: &Path, name: &str, out: &mut Vec<PathBuf>) -> Result<()> {
    if !root.is_dir() {
        return Ok(());
    }

    for entry in fs::read_dir(root).with_context(|| format!("read directory `{}`", root.display()))? {
        let entry = entry.with_context(|| format!("list directory entry under `{}`", root.display()))?;
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
        let aggregate = aggregates.entry(row.suite_id.clone()).or_insert_with(SuiteAggregate::default);
        aggregate.row_count = aggregate.row_count.saturating_add(1);
        aggregate.throughput_sum += row.throughput;
        aggregate.workspace_peak = aggregate.workspace_peak.max(row.workspace_bytes);
        aggregate.transfer_sum = aggregate.transfer_sum.saturating_add(row.transfer_bytes);
        aggregate.not0_sum = aggregate.not0_sum.saturating_add(i64::from(row.not0));
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
            let throughput = aggregate.throughput_sum / aggregate.row_count as f64;
            let transfer_bytes = aggregate.transfer_sum / aggregate.row_count.max(1);
            let crossover_shift_pct = if aggregate.crossover_shift_count > 0 {
                Some(aggregate.crossover_shift_sum / aggregate.crossover_shift_count as f64)
            } else {
                thresholds.baseline_crossover_shift_pct
            };

            return SuiteMeasurement {
                suite_id: suite_id.to_owned(),
                profile,
                source,
                sample_count: aggregate.row_count,
                throughput,
                workspace_bytes: aggregate.workspace_peak,
                transfer_bytes,
                not0: aggregate.not0_sum.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32,
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
        chunk_count: 0,
        fallback_reason: Some("missing_bench_rows".to_owned()),
        crossover_shift_pct: thresholds.baseline_crossover_shift_pct,
    }
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
    if let (Some(limit), Some(actual)) = (thresholds.transfer_regression_pct, transfer_regression_pct) {
        if actual > limit {
            exceeded_reasons.push(format!(
                "transfer regression {:.3}% exceeded threshold {:.3}%",
                actual, limit
            ));
        }
    }
    if let (Some(limit), Some(actual)) = (thresholds.crossover_shift_pct, crossover_shift_delta_pct) {
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

fn write_json_with_fallback(required_path: &str, fallback_name: &str, value: &Value) -> Result<ArtifactWrite> {
    let payload = serde_json::to_vec_pretty(value).context("serialize bench-report artifact json")?;
    write_bytes_with_fallback(required_path, fallback_name, &payload)
}

fn write_bytes_with_fallback(required_path: &str, fallback_name: &str, payload: &[u8]) -> Result<ArtifactWrite> {
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
