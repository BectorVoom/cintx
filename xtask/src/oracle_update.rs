use anyhow::{bail, Result};

pub fn run_oracle_compare(_profiles: &[String], _include_unstable_source: bool) -> Result<()> {
    bail!("oracle-compare gate is not implemented yet")
}

pub fn run_helper_legacy_parity(_profile: &str) -> Result<()> {
    bail!("helper-legacy-parity gate is not implemented yet")
}

pub fn run_oom_contract_check() -> Result<()> {
    bail!("oom-contract-check gate is not implemented yet")
}
