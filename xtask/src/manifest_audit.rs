use anyhow::{bail, Result};

pub fn run_manifest_audit(_profiles: &[String], _check_lock: bool) -> Result<()> {
    bail!("manifest-audit gate is not implemented yet")
}
