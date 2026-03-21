use smallvec::SmallVec;
use std::sync::Arc;

const SHELL_TUPLE_CAPACITY: usize = 4;

#[derive(Debug, thiserror::Error)]
#[error("shell tuple arity cannot exceed {0}")]
pub struct ShellTupleArityError(usize);

/// A primitive shell containing angular momentum, contraction, and coefficient data.
#[derive(Clone, Debug, PartialEq)]
pub struct Shell {
    pub atom_index: u32,
    pub ang_momentum: u8,
    pub nprim: u16,
    pub nctr: u16,
    pub kappa: i16,
    pub exponents: Arc<[f64]>,
    pub coefficients: Arc<[f64]>,
}

impl Shell {
    pub fn new(
        atom_index: u32,
        ang_momentum: u8,
        nprim: u16,
        nctr: u16,
        kappa: i16,
        exponents: Arc<[f64]>,
        coefficients: Arc<[f64]>,
    ) -> Self {
        Shell {
            atom_index,
            ang_momentum,
            nprim,
            nctr,
            kappa,
            exponents,
            coefficients,
        }
    }
}

/// Arity-safe collection of shells matching libcint's ___ tuple inputs.
#[derive(Clone, Debug, PartialEq)]
pub struct ShellTuple {
    shells: SmallVec<[Arc<Shell>; SHELL_TUPLE_CAPACITY]>,
}

impl ShellTuple {
    pub fn try_from_iter<I>(iter: I) -> Result<Self, ShellTupleArityError>
    where
        I: IntoIterator<Item = Arc<Shell>>,
    {
        let mut shells = SmallVec::new();
        for shell in iter {
            if shells.len() >= SHELL_TUPLE_CAPACITY {
                return Err(ShellTupleArityError(SHELL_TUPLE_CAPACITY));
            }
            shells.push(shell);
        }
        Ok(Self { shells })
    }

    pub fn len(&self) -> usize {
        self.shells.len()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &Arc<Shell>> {
        self.shells.iter()
    }

    pub fn as_slice(&self) -> &[Arc<Shell>] {
        &self.shells
    }
}
