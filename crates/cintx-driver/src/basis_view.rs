//! Borrowed view over raw `atm`/`bas`/`env` arrays.
//!
//! The driver deliberately consumes the raw libcint arrays rather than a typed
//! `BasisSet`: it must hand the very same arrays to both cintx and vendored
//! libcint for a benchmark comparison to mean anything.

/// libcint 6.1.3 ABI slots. Pinned against `cintx_compat::raw` by
/// `crates/cintx-driver/tests/slot_parity.rs`.
pub const ATM_SLOTS: usize = 6;
pub const BAS_SLOTS: usize = 8;
pub const ANG_OF: usize = 1;
pub const NPRIM_OF: usize = 2;
pub const NCTR_OF: usize = 3;

/// Borrowed `atm`/`bas`/`env` triple plus derived per-shell metadata.
#[derive(Clone, Copy, Debug)]
pub struct BasisView<'a> {
    pub atm: &'a [i32],
    pub bas: &'a [i32],
    pub env: &'a [f64],
}

impl<'a> BasisView<'a> {
    #[must_use]
    pub fn new(atm: &'a [i32], bas: &'a [i32], env: &'a [f64]) -> Self {
        Self { atm, bas, env }
    }

    #[must_use]
    pub fn natm(&self) -> usize {
        self.atm.len() / ATM_SLOTS
    }

    #[must_use]
    pub fn nbas(&self) -> usize {
        self.bas.len() / BAS_SLOTS
    }

    #[must_use]
    pub fn ang_momentum(&self, shell: usize) -> u8 {
        self.bas[shell * BAS_SLOTS + ANG_OF].max(0) as u8
    }

    #[must_use]
    pub fn nprim(&self, shell: usize) -> u32 {
        self.bas[shell * BAS_SLOTS + NPRIM_OF].max(0) as u32
    }

    #[must_use]
    pub fn nctr(&self, shell: usize) -> u32 {
        self.bas[shell * BAS_SLOTS + NCTR_OF].max(0) as u32
    }

    /// Spherical AO count for a shell: `(2l + 1) * nctr`.
    #[must_use]
    pub fn nsph(&self, shell: usize) -> usize {
        (2 * usize::from(self.ang_momentum(shell)) + 1) * self.nctr(shell) as usize
    }

    /// Cartesian AO count for a shell: `(l+1)(l+2)/2 * nctr`.
    #[must_use]
    pub fn ncart(&self, shell: usize) -> usize {
        let l = usize::from(self.ang_momentum(shell));
        ((l + 1) * (l + 2) / 2) * self.nctr(shell) as usize
    }

    /// Rys order for a quartet: `(li + lj + lk + ll) / 2 + 1`.
    ///
    /// This is the value that decides whether a quartet runs on device
    /// (`nroots <= 5` today) or falls back to the host serial loop.
    #[must_use]
    pub fn quartet_nroots(&self, i: usize, j: usize, k: usize, l: usize) -> u32 {
        let sum = u32::from(self.ang_momentum(i))
            + u32::from(self.ang_momentum(j))
            + u32::from(self.ang_momentum(k))
            + u32::from(self.ang_momentum(l));
        sum / 2 + 1
    }
}
