//! A stack of equally sized float planes: the tensor form the network uses.

/// `channels` planes of `height` x `width` samples, channel-major, row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Planes {
    channels: usize,
    height: usize,
    width: usize,
    data: Vec<f32>,
}

impl Planes {
    /// Zero-filled planes.
    #[must_use]
    pub fn zeros(channels: usize, height: usize, width: usize) -> Self {
        Self {
            channels,
            height,
            width,
            data: vec![0.0; channels * height * width],
        }
    }

    /// Number of planes.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Rows per plane.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Samples per row.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Samples in one plane.
    #[must_use]
    pub fn plane_len(&self) -> usize {
        self.height * self.width
    }

    /// All samples, channel-major.
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// All samples, channel-major, writable.
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Plane `channel`.
    #[must_use]
    pub fn plane(&self, channel: usize) -> &[f32] {
        let len = self.plane_len();
        &self.data[channel * len..(channel + 1) * len]
    }

    /// Plane `channel`, writable.
    pub fn plane_mut(&mut self, channel: usize) -> &mut [f32] {
        let len = self.plane_len();
        &mut self.data[channel * len..(channel + 1) * len]
    }

    /// Set every sample to zero.
    pub fn clear(&mut self) {
        self.data.fill(0.0);
    }
}
