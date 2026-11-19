//! Conversion between RGBA pixels and the 4:2:0 / 4:4:4 YUV the pipeline
//! works in: 8-bit BT.709, limited range, as every stored clip is.

use crate::fastmath::round;
use crate::planes::Planes;
use crate::spec::PICTURE_CHANNELS;

/// BT.709 luma coefficients.
const KR: f32 = 0.2126;
const KB: f32 = 0.0722;
const KG: f32 = 1.0 - KR - KB;
/// Limited-range levels on the 0-255 scale.
const LUMA_OFFSET: f32 = 16.0;
pub(crate) const LUMA_RANGE: f32 = 219.0;
const CHROMA_OFFSET: f32 = 128.0;
const CHROMA_RANGE: f32 = 224.0;
pub(crate) const MAX_SAMPLE: f32 = 255.0;
const RGBA_BYTES: usize = 4;

/// Reciprocals, so that the per-sample arithmetic has no divisions.
const PER_LUMA_RANGE: f32 = 1.0 / LUMA_RANGE;
const PER_CHROMA_RANGE: f32 = 1.0 / CHROMA_RANGE;
const PER_KG: f32 = 1.0 / KG;
/// Colour difference to chroma: Cb = (B - Y) / (2 (1 - Kb)), Cr likewise.
const CB_SCALE: f32 = 0.5 / (1.0 - KB);
const CR_SCALE: f32 = 0.5 / (1.0 - KR);
/// Positions of green, blue and (fully opaque) alpha in a little-endian RGBA word.
const GREEN_SHIFT: u32 = 8;
const BLUE_SHIFT: u32 = 16;
const OPAQUE: u32 = 0xff00_0000;
/// Pixels converted at a time: the arithmetic runs over a short run of
/// packed words, a loop simple enough to become SIMD, and the words are then
/// written out as bytes.
const RUN: usize = 64;
/// Samples a chroma value is averaged over, per axis.
const CHROMA_STEP: usize = 2;

/// The nearest level on the 8-bit scale, clamped to it.
fn to_level(value: f32) -> u32 {
    round(value.clamp(0.0, MAX_SAMPLE))
}

fn to_byte(value: f32) -> u8 {
    to_level(value) as u8
}

/// An 8-bit 4:2:0 frame.
#[derive(Debug, Clone)]
pub struct Yuv420 {
    /// Luma, `height` x `width`.
    pub y: Vec<u8>,
    /// Chroma, each `height / 2` x `width / 2`.
    pub u: Vec<u8>,
    /// See `u`.
    pub v: Vec<u8>,
}

impl Yuv420 {
    /// A black frame of `width` x `height` (even dimensions).
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        let chroma_len = (width / CHROMA_STEP) * (height / CHROMA_STEP);
        Self {
            y: vec![LUMA_OFFSET as u8; width * height],
            u: vec![CHROMA_OFFSET as u8; chroma_len],
            v: vec![CHROMA_OFFSET as u8; chroma_len],
        }
    }
}

/// Brightness of an RGB pixel on the full-range 0-255 scale, rounded.
pub(crate) fn luma_of(r: u8, g: u8, b: u8) -> u8 {
    (KR * f32::from(r) + KG * f32::from(g) + KB * f32::from(b) + 0.5) as u8
}

/// Luma (0-255 scale, full range) and the blue and red colour differences of one pixel.
fn analyse(pixel: [u8; RGBA_BYTES]) -> (f32, f32, f32) {
    let [r, g, b] = [
        f32::from(pixel[0]),
        f32::from(pixel[1]),
        f32::from(pixel[2]),
    ];
    let luma = KR * r + KG * g + KB * b;
    (luma, (b - luma) * CB_SCALE, (r - luma) * CR_SCALE)
}

/// Convert `width`-wide RGBA pixels to 4:2:0 in `frame` (of the same size,
/// from [`Yuv420::new`]), chroma averaged over each 2x2 block.
pub fn rgba_to_yuv420(rgba: &[u8], width: usize, frame: &mut Yuv420) {
    let Yuv420 { y, u, v } = frame;
    let luma_scale = LUMA_RANGE / MAX_SAMPLE;
    // Each chroma cell sums four pixels; the scale folds in that average.
    let chroma_scale = CHROMA_RANGE / MAX_SAMPLE / (CHROMA_STEP * CHROMA_STEP) as f32;
    let pixels = rgba.as_chunks::<RGBA_BYTES>().0;
    let row_pairs = pixels
        .chunks_exact(CHROMA_STEP * width)
        .zip(y.chunks_exact_mut(CHROMA_STEP * width));
    let chroma_rows = u
        .chunks_exact_mut(width / CHROMA_STEP)
        .zip(v.chunks_exact_mut(width / CHROMA_STEP));
    for ((pixels, luma), (cb_row, cr_row)) in row_pairs.zip(chroma_rows) {
        let (upper, lower) = pixels.split_at(width);
        let (luma_upper, luma_lower) = luma.split_at_mut(width);
        let sources = upper
            .as_chunks::<CHROMA_STEP>()
            .0
            .iter()
            .zip(lower.as_chunks::<CHROMA_STEP>().0);
        let lumas = luma_upper
            .as_chunks_mut::<CHROMA_STEP>()
            .0
            .iter_mut()
            .zip(luma_lower.as_chunks_mut::<CHROMA_STEP>().0);
        let chromas = cb_row.iter_mut().zip(cr_row.iter_mut());
        for (((above, below), (luma_above, luma_below)), (cb, cr)) in
            sources.zip(lumas).zip(chromas)
        {
            let [a, b, c, d] = [above[0], above[1], below[0], below[1]].map(analyse);
            let level = |luma: f32| to_byte(LUMA_OFFSET + luma * luma_scale);
            *luma_above = [level(a.0), level(b.0)];
            *luma_below = [level(c.0), level(d.0)];
            *cb = to_byte(CHROMA_OFFSET + (a.1 + b.1 + c.1 + d.1) * chroma_scale);
            *cr = to_byte(CHROMA_OFFSET + (a.2 + b.2 + c.2 + d.2) * chroma_scale);
        }
    }
}

/// Write 4:4:4 YUV planes in [0, 1] as opaque RGBA pixels.
///
/// # Panics
///
/// If `yuv` does not have three planes or `rgba` is not four bytes per sample.
pub fn yuv444_to_rgba(yuv: &Planes, rgba: &mut [u8]) {
    assert_eq!(
        yuv.channels(),
        PICTURE_CHANNELS,
        "expected Y, U and V planes"
    );
    assert_eq!(
        rgba.len(),
        yuv.plane_len() * RGBA_BYTES,
        "RGBA buffer has the wrong size"
    );
    let level = |value: f32| to_level(value * MAX_SAMPLE);
    let planes = (yuv.plane(0).chunks(RUN))
        .zip(yuv.plane(1).chunks(RUN))
        .zip(yuv.plane(2).chunks(RUN));
    for (pixels, ((ys, us), vs)) in rgba.chunks_mut(RUN * RGBA_BYTES).zip(planes) {
        let mut words = [0u32; RUN];
        for (word, ((&y, &u), &v)) in words.iter_mut().zip(ys.iter().zip(us).zip(vs)) {
            let luma = (y * MAX_SAMPLE - LUMA_OFFSET) * PER_LUMA_RANGE;
            let cb = (u * MAX_SAMPLE - CHROMA_OFFSET) * PER_CHROMA_RANGE;
            let cr = (v * MAX_SAMPLE - CHROMA_OFFSET) * PER_CHROMA_RANGE;
            let r = luma + 2.0 * (1.0 - KR) * cr;
            let b = luma + 2.0 * (1.0 - KB) * cb;
            let g = (luma - KR * r - KB * b) * PER_KG;
            *word = level(r) | level(g) << GREEN_SHIFT | level(b) << BLUE_SHIFT | OPAQUE;
        }
        // Little-endian byte order puts red first.
        for (pixel, word) in pixels.as_chunks_mut::<RGBA_BYTES>().0.iter_mut().zip(words) {
            *pixel = word.to_le_bytes();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grey_survives_a_round_trip() {
        let rgba = [120u8, 120, 120, 255].repeat(16);
        let mut yuv = Yuv420::new(4, 4);
        rgba_to_yuv420(&rgba, 4, &mut yuv);
        assert!(yuv.u.iter().chain(&yuv.v).all(|&c| c == 128));
        let mut planes = Planes::zeros(3, 4, 4);
        planes.plane_mut(0).fill(f32::from(yuv.y[0]) / 255.0);
        planes.plane_mut(1).fill(128.0 / 255.0);
        planes.plane_mut(2).fill(128.0 / 255.0);
        let mut out = [0u8; 64];
        yuv444_to_rgba(&planes, &mut out);
        assert!(
            out.as_chunks::<4>()
                .0
                .iter()
                .all(|p| p[0].abs_diff(120) <= 1 && p[0] == p[1] && p[1] == p[2])
        );
    }
}
