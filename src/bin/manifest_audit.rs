use std::env;
use std::fs;
use std::process::ExitCode;

use cintx::{
    COMPILED_MANIFEST_LOCK_PATH, audit_compiled_manifest_lock, generated_compiled_manifest_lock,
    parse_compiled_manifest_lock_json,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<(), String> {
    let command = env::args().nth(1).unwrap_or_else(|| "check".to_string());
    match command.as_str() {
        "generate" | "update-lock" => generate_lock(),
        "check" | "audit" => check_lock(),
        other => Err(format!(
            "unknown manifest-audit command `{other}` (expected `generate` or `check`)"
        )),
    }
}

fn generate_lock() -> Result<(), String> {
    let generated = generated_compiled_manifest_lock()
        .map_err(|error| format!("failed to generate compiled manifest lock: {error}"))?;
    let canonical = generated
        .canonical_json()
        .map_err(|error| format!("failed to canonicalize compiled manifest lock: {error}"))?;
    fs::write(COMPILED_MANIFEST_LOCK_PATH, format!("{canonical}\n"))
        .map_err(|error| format!("failed to write `{COMPILED_MANIFEST_LOCK_PATH}`: {error}"))?;
    println!("updated `{COMPILED_MANIFEST_LOCK_PATH}`");
    Ok(())
}

fn check_lock() -> Result<(), String> {
    let raw = fs::read_to_string(COMPILED_MANIFEST_LOCK_PATH)
        .map_err(|error| format!("failed to read `{COMPILED_MANIFEST_LOCK_PATH}`: {error}"))?;
    let lock = parse_compiled_manifest_lock_json(&raw)
        .map_err(|error| format!("failed to parse `{COMPILED_MANIFEST_LOCK_PATH}`: {error}"))?;
    audit_compiled_manifest_lock(&lock)
        .map_err(|error| format!("compiled manifest audit failed: {error}"))?;
    println!("compiled manifest audit passed");
    Ok(())
}
