//! NWChem-format basis and ECP text parser.
//!
//! Handles the two block shapes the Basis Set Exchange emits for the def2
//! family:
//!
//! ```text
//! BASIS "ao basis" SPHERICAL PRINT
//! C    S                       <- element, angular label
//!   7.1616837  0.0154633       <- exponent, then one column per contraction
//!   ...
//! END
//!
//! ECP
//! Rb nelec 28                  <- core electrons replaced
//! Rb ul                        <- local channel
//!   2   3.8431140  -12.3169    <- r power, exponent, coefficient
//! Rb S                         <- semi-local projector for l = 0
//!   ...
//! END
//! ```
//!
//! A block with more than one coefficient column is a general contraction
//! (`nctr > 1`); def2 uses these, so the parser must not assume `nctr == 1`.

use crate::element::atomic_number;
use crate::error::BasisError;
use std::collections::BTreeMap;

/// Angular labels in `l` order. `SP`-style fused labels are deliberately not
/// supported: def2 never uses them, and silently guessing would be worse than
/// a clear error.
const ANGULAR_LABELS: [&str; 8] = ["S", "P", "D", "F", "G", "H", "I", "K"];

fn angular_momentum(label: &str) -> Option<u8> {
    ANGULAR_LABELS
        .iter()
        .position(|candidate| candidate.eq_ignore_ascii_case(label))
        .map(|index| index as u8)
}

/// One contraction block: `nprim` exponents and `nprim * nctr`
/// contraction-major coefficients (`coeff[ic * nprim + ip]`).
#[derive(Clone, Debug, PartialEq)]
pub struct ContractionBlock {
    pub ang_momentum: u8,
    pub exponents: Vec<f64>,
    /// Contraction-major: `coefficients[ic * nprim + ip]`.
    pub coefficients: Vec<f64>,
    pub nctr: usize,
}

impl ContractionBlock {
    #[must_use]
    pub fn nprim(&self) -> usize {
        self.exponents.len()
    }
}

/// One ECP block for a single `(element, channel)` pair.
#[derive(Clone, Debug, PartialEq)]
pub struct EcpBlock {
    /// `None` for the local `ul` channel; `Some(l)` for a semi-local projector.
    pub projector: Option<u8>,
    /// `r^n` power for each primitive.
    pub radial_powers: Vec<i16>,
    pub exponents: Vec<f64>,
    pub coefficients: Vec<f64>,
}

/// Parsed per-element ECP record.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct EcpRecord {
    pub core_electrons: u16,
    pub blocks: Vec<EcpBlock>,
}

/// Parsed orbital-basis table, keyed by atomic number.
pub type BasisTable = BTreeMap<u16, Vec<ContractionBlock>>;

/// Parsed ECP table, keyed by atomic number.
pub type EcpTable = BTreeMap<u16, EcpRecord>;

fn strip_comment(line: &str) -> &str {
    let line = line.split('#').next().unwrap_or("");
    line.split('!').next().unwrap_or("")
}

/// Parse Fortran-style floats, accepting `D`/`d` exponent markers alongside
/// `E`/`e`. BSE emits `E`, but Turbomole-sourced text in the wild uses `D`.
fn parse_float(token: &str) -> Result<f64, BasisError> {
    let normalized = token.replace(['D', 'd'], "E");
    normalized
        .parse::<f64>()
        .map_err(|_| BasisError::MalformedNumber {
            token: token.to_owned(),
        })
}

/// Parse an NWChem orbital-basis document into a per-element table.
///
/// # Errors
/// Returns [`BasisError`] on an unknown element symbol, an unknown angular
/// label, a malformed number, or a block whose rows disagree on column count.
pub fn parse_basis(text: &str) -> Result<BasisTable, BasisError> {
    let mut table: BasisTable = BTreeMap::new();
    let mut current: Option<(u16, u8, Vec<f64>, Vec<Vec<f64>>)> = None;
    // BSE emits the orbital basis and its ECP as two sections of one document;
    // the ECP rows have a different column meaning and must not be read here.
    let mut in_ecp_section = false;

    // Header lines (`BASIS "ao basis" SPHERICAL PRINT`) and `END` terminate or
    // precede blocks but carry no contraction data.
    for raw_line in text.lines() {
        let line = strip_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }

        let upper = line.to_ascii_uppercase();
        if upper == "ECP" {
            flush_block(&mut table, current.take())?;
            in_ecp_section = true;
            continue;
        }
        if upper == "END" {
            flush_block(&mut table, current.take())?;
            in_ecp_section = false;
            continue;
        }
        if in_ecp_section {
            continue;
        }
        if upper.starts_with("BASIS")
            || upper.starts_with("SPHERICAL")
            || upper.starts_with("CARTESIAN")
        {
            flush_block(&mut table, current.take())?;
            continue;
        }

        let mut tokens = line.split_whitespace();
        let first = tokens.next().unwrap_or_default();

        // A block header is `<Element> <AngularLabel>`; anything else on a line
        // starting with a non-numeric token is unrecognized.
        if let Some(z) = atomic_number(first) {
            let label = tokens.next().ok_or_else(|| BasisError::MalformedBlock {
                detail: format!("element line `{line}` has no angular label"),
            })?;
            let l = angular_momentum(label).ok_or_else(|| BasisError::UnknownAngularLabel {
                label: label.to_owned(),
            })?;
            flush_block(&mut table, current.take())?;
            current = Some((z, l, Vec::new(), Vec::new()));
            continue;
        }

        // Otherwise this is a primitive row: exponent then >=1 coefficients.
        let (_, _, exponents, columns) =
            current.as_mut().ok_or_else(|| BasisError::MalformedBlock {
                detail: format!("primitive row `{line}` appears before any element header"),
            })?;

        let values = std::iter::once(first)
            .chain(tokens)
            .map(parse_float)
            .collect::<Result<Vec<f64>, _>>()?;
        if values.len() < 2 {
            return Err(BasisError::MalformedBlock {
                detail: format!("primitive row `{line}` needs an exponent and >=1 coefficient"),
            });
        }
        exponents.push(values[0]);
        columns.push(values[1..].to_vec());
    }

    flush_block(&mut table, current.take())?;
    Ok(table)
}

/// Convert an accumulated row-major block into contraction-major storage and
/// append it to the table.
fn flush_block(
    table: &mut BasisTable,
    block: Option<(u16, u8, Vec<f64>, Vec<Vec<f64>>)>,
) -> Result<(), BasisError> {
    let Some((z, l, exponents, rows)) = block else {
        return Ok(());
    };
    if exponents.is_empty() {
        return Ok(());
    }

    let nctr = rows[0].len();
    if rows.iter().any(|row| row.len() != nctr) {
        return Err(BasisError::MalformedBlock {
            detail: format!(
                "element Z={z} l={l} block has ragged contraction columns \
                 (expected {nctr} per row)"
            ),
        });
    }

    // Row-major (prim-major) -> contraction-major `coeff[ic * nprim + ip]`.
    let nprim = exponents.len();
    let mut coefficients = vec![0.0_f64; nprim * nctr];
    for (ip, row) in rows.iter().enumerate() {
        for (ic, &value) in row.iter().enumerate() {
            coefficients[ic * nprim + ip] = value;
        }
    }

    table.entry(z).or_default().push(ContractionBlock {
        ang_momentum: l,
        exponents,
        coefficients,
        nctr,
    });
    Ok(())
}

/// Parse an NWChem ECP document into a per-element table.
///
/// # Errors
/// Returns [`BasisError`] on an unknown element symbol, an unknown projector
/// label, or a malformed primitive row.
pub fn parse_ecp(text: &str) -> Result<EcpTable, BasisError> {
    let mut table: EcpTable = BTreeMap::new();
    let mut current: Option<(u16, Option<u8>, EcpBlock)> = None;
    // Mirror of `parse_basis`: in a combined document only the ECP section is
    // ours, and it is entered by a bare `ECP` line.
    let mut in_ecp_section = false;

    for raw_line in text.lines() {
        let line = strip_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }

        let upper = line.to_ascii_uppercase();
        if upper == "ECP" {
            flush_ecp_block(&mut table, current.take());
            in_ecp_section = true;
            continue;
        }
        if upper == "END" {
            flush_ecp_block(&mut table, current.take());
            in_ecp_section = false;
            continue;
        }
        if !in_ecp_section {
            continue;
        }

        let mut tokens = line.split_whitespace();
        let first = tokens.next().unwrap_or_default();

        if let Some(z) = atomic_number(first) {
            let keyword = tokens.next().ok_or_else(|| BasisError::MalformedBlock {
                detail: format!("ECP line `{line}` has no keyword"),
            })?;

            if keyword.eq_ignore_ascii_case("nelec") {
                let count = tokens.next().ok_or_else(|| BasisError::MalformedBlock {
                    detail: format!("ECP line `{line}` has no core-electron count"),
                })?;
                let core_electrons =
                    count
                        .parse::<u16>()
                        .map_err(|_| BasisError::MalformedNumber {
                            token: count.to_owned(),
                        })?;
                flush_ecp_block(&mut table, current.take());
                table.entry(z).or_default().core_electrons = core_electrons;
                continue;
            }

            // `ul` is the local channel; anything else is a projector label.
            let projector = if keyword.eq_ignore_ascii_case("ul") {
                None
            } else {
                Some(
                    angular_momentum(keyword).ok_or_else(|| BasisError::UnknownAngularLabel {
                        label: keyword.to_owned(),
                    })?,
                )
            };
            flush_ecp_block(&mut table, current.take());
            current = Some((
                z,
                projector,
                EcpBlock {
                    projector,
                    radial_powers: Vec::new(),
                    exponents: Vec::new(),
                    coefficients: Vec::new(),
                },
            ));
            continue;
        }

        let (_, _, block) = current.as_mut().ok_or_else(|| BasisError::MalformedBlock {
            detail: format!("ECP primitive row `{line}` appears before any channel header"),
        })?;

        let power = first
            .parse::<i16>()
            .map_err(|_| BasisError::MalformedNumber {
                token: first.to_owned(),
            })?;
        let exponent = parse_float(tokens.next().ok_or_else(|| BasisError::MalformedBlock {
            detail: format!("ECP primitive row `{line}` has no exponent"),
        })?)?;
        let coefficient =
            parse_float(tokens.next().ok_or_else(|| BasisError::MalformedBlock {
                detail: format!("ECP primitive row `{line}` has no coefficient"),
            })?)?;

        block.radial_powers.push(power);
        block.exponents.push(exponent);
        block.coefficients.push(coefficient);
    }

    flush_ecp_block(&mut table, current.take());
    Ok(table)
}

fn flush_ecp_block(table: &mut EcpTable, block: Option<(u16, Option<u8>, EcpBlock)>) {
    let Some((z, _, block)) = block else {
        return;
    };
    if block.exponents.is_empty() {
        return;
    }
    table.entry(z).or_default().blocks.push(block);
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEGMENTED: &str = r#"
BASIS "ao basis" SPHERICAL PRINT
#BASIS SET: (4s,1p) -> [2s,1p]
H    S
     13.0107010              0.19682158E-01
      1.9622572              0.13796524
      0.44453796             0.47831935
H    S
      0.12194962             1.0000000
H    P
      0.8000000              1.0000000
END
"#;

    const GENERAL: &str = r#"
BASIS "ao basis" SPHERICAL PRINT
C    S
      10.0    0.1   0.4
       2.0    0.2   0.5
       0.5    0.3   0.6
END
"#;

    #[test]
    fn parses_segmented_blocks() {
        let table = parse_basis(SEGMENTED).expect("parse should succeed");
        let hydrogen = &table[&1];
        assert_eq!(hydrogen.len(), 3);
        assert_eq!(hydrogen[0].ang_momentum, 0);
        assert_eq!(hydrogen[0].nprim(), 3);
        assert_eq!(hydrogen[0].nctr, 1);
        assert_eq!(hydrogen[2].ang_momentum, 1);
        assert_eq!(hydrogen[2].exponents, vec![0.8]);
    }

    /// A general contraction must land in contraction-major order, because
    /// that is the layout `cintx_core::Shell` and libcint's `env` both use.
    #[test]
    fn general_contraction_is_stored_contraction_major() {
        let table = parse_basis(GENERAL).expect("parse should succeed");
        let block = &table[&6][0];
        assert_eq!(block.nctr, 2);
        assert_eq!(block.nprim(), 3);
        assert_eq!(block.coefficients, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    }

    #[test]
    fn accepts_fortran_d_exponents() {
        let text = "BASIS \"ao basis\" SPHERICAL PRINT\nH S\n 1.0D+01 1.0D-01\nEND\n";
        let table = parse_basis(text).expect("parse should succeed");
        assert_eq!(table[&1][0].exponents, vec![10.0]);
        assert_eq!(table[&1][0].coefficients, vec![0.1]);
    }

    #[test]
    fn rejects_ragged_contraction_columns() {
        let text = "BASIS \"ao basis\" SPHERICAL PRINT\nC S\n 1.0 0.1 0.2\n 2.0 0.3\nEND\n";
        assert!(matches!(
            parse_basis(text),
            Err(BasisError::MalformedBlock { .. })
        ));
    }

    #[test]
    fn rejects_unknown_angular_label() {
        let text = "BASIS \"ao basis\" SPHERICAL PRINT\nC Z\n 1.0 0.1\nEND\n";
        assert!(matches!(
            parse_basis(text),
            Err(BasisError::UnknownAngularLabel { .. })
        ));
    }

    #[test]
    fn parses_ecp_local_and_projector_channels() {
        let text = r#"
ECP
Rb nelec 28
Rb ul
2       3.8431140            -12.3169000
Rb S
2       5.0365510             89.5001980
2       1.9708490              0.4937610
END
"#;
        let table = parse_ecp(text).expect("parse should succeed");
        let record = &table[&37];
        assert_eq!(record.core_electrons, 28);
        assert_eq!(record.blocks.len(), 2);
        assert_eq!(record.blocks[0].projector, None);
        assert_eq!(record.blocks[0].radial_powers, vec![2]);
        assert_eq!(record.blocks[1].projector, Some(0));
        assert_eq!(record.blocks[1].exponents.len(), 2);
    }
}
