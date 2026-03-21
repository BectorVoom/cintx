use smallvec::SmallVec;

const TENSOR_EXTENT_CAP: usize = 4;
const TENSOR_STRIDE_CAP: usize = 6;

/// Describes the shape of an integral tensor (batch × component × extras).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorShape {
    pub batch: usize,
    pub comp: usize,
    pub extents: SmallVec<[usize; TENSOR_EXTENT_CAP]>,
    pub complex_interleaved: bool,
}

impl TensorShape {
    pub fn new(
        batch: usize,
        comp: usize,
        extents: SmallVec<[usize; TENSOR_EXTENT_CAP]>,
        complex_interleaved: bool,
    ) -> Self {
        TensorShape {
            batch,
            comp,
            extents,
            complex_interleaved,
        }
    }

    pub fn total_extent(&self) -> usize {
        self.extents.iter().copied().product::<usize>()
    }
}

/// Holds stride information plus ordering hints.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorLayout {
    pub strides: SmallVec<[usize; TENSOR_STRIDE_CAP]>,
    pub column_major_compat: bool,
    pub comp_is_leading: bool,
}

impl TensorLayout {
    pub fn new(
        strides: SmallVec<[usize; TENSOR_STRIDE_CAP]>,
        column_major_compat: bool,
        comp_is_leading: bool,
    ) -> Self {
        TensorLayout {
            strides,
            column_major_compat,
            comp_is_leading,
        }
    }

    pub fn stride_for(&self, axis: usize) -> Option<usize> {
        self.strides.get(axis).copied()
    }
}
