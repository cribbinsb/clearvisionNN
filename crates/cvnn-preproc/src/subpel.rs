//! Sub-sample refinement of a whole-sample block vector.
//!
//! nanocodec's analysis reports whole-sample vectors (codecs refine against
//! their own reconstructions). Here the block difference is measured at the
//! chosen vector and its four neighbours, and a parabola through each axis'
//! three values gives the fractional position of the minimum.

/// Largest correction, in samples: the parabola is only trusted between the
/// two neighbours it was fitted through.
pub const MAX_OFFSET: f32 = 0.5;

/// Curvatures at or below this are flat or inverted: no minimum to locate.
const MIN_CURVATURE: f32 = 1e-6;

/// Fractional offset of the minimum of the parabola through the block
/// differences at `-1`, `0` and `+1` samples along one axis.
#[must_use]
pub fn parabola_offset(before: f32, centre: f32, after: f32) -> f32 {
    let curvature = before - 2.0 * centre + after;
    if curvature <= MIN_CURVATURE {
        return 0.0;
    }
    ((before - after) / (2.0 * curvature)).clamp(-MAX_OFFSET, MAX_OFFSET)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_differences_need_no_correction() {
        assert_eq!(parabola_offset(4.0, 1.0, 4.0), 0.0);
    }

    #[test]
    fn minimum_leans_towards_the_better_neighbour() {
        assert!(parabola_offset(4.0, 1.0, 2.0) > 0.0);
        assert!(parabola_offset(2.0, 1.0, 4.0) < 0.0);
    }

    #[test]
    fn flat_or_inverted_differences_are_left_alone() {
        assert_eq!(parabola_offset(1.0, 1.0, 1.0), 0.0);
        assert_eq!(parabola_offset(1.0, 3.0, 1.0), 0.0);
    }

    #[test]
    fn correction_never_leaves_the_fitted_interval() {
        assert_eq!(parabola_offset(10.0, 1.0, 0.0), MAX_OFFSET);
    }
}
