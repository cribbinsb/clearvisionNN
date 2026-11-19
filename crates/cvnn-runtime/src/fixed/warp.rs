//! Bilinear warping in integers: positions in 2^-13 of a sample, weights in
//! 64ths, samples of any integer plane.

use super::motion::FLOW_BITS;

/// Bits of a bilinear weight along one axis.
pub(super) const WEIGHT_BITS: u32 = 6;
const ONE: i32 = 1 << WEIGHT_BITS;

/// Where one output sample reads from: the upper-left of four source samples
/// and the weights of the right-hand and of the lower ones, in 64ths.
#[derive(Debug, Clone, Copy)]
pub(super) struct Tap {
    index: usize,
    across: i32,
    down: i32,
}

/// One axis of a tap: `position` (a sample index) displaced by `flow`,
/// clamped to the plane, as the first of two samples and the second's weight.
fn axis(position: usize, flow: i32, len: usize) -> (usize, i32) {
    let last = len.cast_signed() as i32 - 1;
    let at = ((position.cast_signed() as i32) << FLOW_BITS)
        .saturating_add(flow)
        .clamp(0, last << FLOW_BITS);
    // A position on the last sample is taken as a full step from the one before it.
    let first = (at >> FLOW_BITS).min(last - 1);
    let fraction = at - (first << FLOW_BITS);
    let weight = (fraction + (1 << (FLOW_BITS - WEIGHT_BITS - 1))) >> (FLOW_BITS - WEIGHT_BITS);
    (first as usize, weight)
}

/// The taps of a whole `height` x `width` plane displaced by `flow` (dx, dy).
pub(super) fn taps((height, width): (usize, usize), flow: [&[i32]; 2]) -> Vec<Tap> {
    let mut all = Vec::with_capacity(height * width);
    for y in 0..height {
        for x in 0..width {
            let sample = y * width + x;
            let (column, across) = axis(x, flow[0][sample], width);
            let (row, down) = axis(y, flow[1][sample], height);
            all.push(Tap {
                index: row * width + column,
                across,
                down,
            });
        }
    }
    all
}

impl Tap {
    /// The weighted sum of the four samples of `source` (a plane `width`
    /// wide), scaled by 2^(2 * `WEIGHT_BITS`).
    pub(super) fn sum(self, source: impl Fn(usize) -> i32, width: usize) -> i32 {
        let along = |at: usize| source(at) * (ONE - self.across) + source(at + 1) * self.across;
        along(self.index) * (ONE - self.down) + along(self.index + width) * self.down
    }

    /// The interpolated sample, to the nearest, halves upwards.
    pub(super) fn sample(self, source: impl Fn(usize) -> i32, width: usize) -> i32 {
        (self.sum(source, width) + (1 << (2 * WEIGHT_BITS - 1))) >> (2 * WEIGHT_BITS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_moves_by_the_flow_and_halves_interpolate() {
        let source = [0, 100, 200, 300];
        let half = 1 << (FLOW_BITS - 1);
        let flow: [&[i32]; 2] = [&[half; 4], &[0; 4]];
        let warped: Vec<i32> = taps((2, 2), flow)
            .iter()
            .map(|tap| tap.sample(|at| source[at], 2))
            .collect();
        assert_eq!(warped, [50, 100, 250, 300]);
    }

    #[test]
    fn positions_outside_the_plane_take_the_edge_sample() {
        let source = [1, 2, 3, 4];
        let far = 9 << FLOW_BITS;
        let warped: Vec<i32> = taps((2, 2), [&[far; 4], &[far; 4]])
            .iter()
            .map(|tap| tap.sample(|at| source[at], 2))
            .collect();
        assert_eq!(warped, [4; 4]);
    }
}
