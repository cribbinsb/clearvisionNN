//! Fixed 2x upscaling kernels, sample-for-sample the same as the training
//! code's (`python/clearvisionnn/models/resample.py`), which uses `PyTorch`'s
//! half-pixel-centre convention with edge replication.
//!
//! A 2x upscale has only two output phases per axis, so each pass is a
//! weighted sum of whole rows: loops the compiler turns into SIMD.

/// Keys' cubic coefficient, as `PyTorch`'s bicubic uses.
const CUBIC_A: f32 = -0.75;
/// Output sample `i` sits at source position `(i + 0.5) / 2 - 0.5`: a quarter
/// sample before source sample `i / 2` (even `i`) or after it (odd `i`).
/// These are the fractions past the first of the two nearest source samples.
const EVEN_FRACTION: f32 = 0.75;
const ODD_FRACTION: f32 = 0.25;
/// Edge samples replicated on each side of a row: the widest kernel (cubic)
/// reaches two samples past the edge.
const PAD: usize = 2;

/// Cubic convolution weights for the four taps around a fraction `t`.
fn cubic_weights(t: f32) -> [f32; 4] {
    let near = |x: f32| ((CUBIC_A + 2.0) * x - (CUBIC_A + 3.0)) * x * x + 1.0;
    let far = |x: f32| ((CUBIC_A * x - 5.0 * CUBIC_A) * x + 8.0 * CUBIC_A) * x - 4.0 * CUBIC_A;
    [far(t + 1.0), near(t), near(1.0 - t), far(2.0 - t)]
}

/// A separable 2x kernel of `N` taps.
struct Kernel<const N: usize> {
    /// Weights for even and odd output samples.
    phases: [[f32; N]; 2],
    /// Source index of the first tap of output sample 0 (an even sample);
    /// each later sample's taps start half a source sample further on.
    first: isize,
}

impl<const N: usize> Kernel<N> {
    /// `weights(t)` gives the taps around fraction `t`, the first of them
    /// `before` samples ahead of the sample the fraction is measured from.
    fn new(weights: impl Fn(f32) -> [f32; N], before: isize) -> Self {
        Self {
            phases: [weights(EVEN_FRACTION), weights(ODD_FRACTION)],
            first: -1 - before,
        }
    }
}

/// `out[i] = sum over j of rows[j][i] * weights[j]`, summed in tap order.
fn combine<const N: usize>(rows: [&[f32]; N], weights: &[f32; N], out: &mut [f32]) {
    // Every row cut to the output's length, so that the loop needs no bounds checks.
    let rows = rows.map(|row| &row[..out.len()]);
    for (index, value) in out.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (row, &weight) in rows.iter().zip(weights) {
            sum += row[index] * weight;
        }
        *value = sum;
    }
}

/// Working buffers for upscaling one row horizontally.
#[derive(Debug, Default)]
struct RowUpscaler {
    /// The source row with its edges replicated.
    padded: Vec<f32>,
    /// The even and the odd output samples.
    phases: [Vec<f32>; 2],
}

impl RowUpscaler {
    /// Upscale `row` to twice its width into `out`.
    fn upscale<const N: usize>(&mut self, row: &[f32], kernel: &Kernel<N>, out: &mut [f32]) {
        let width = row.len();
        self.padded.clear();
        self.padded.extend(core::iter::repeat_n(row[0], PAD));
        self.padded.extend_from_slice(row);
        self.padded
            .extend(core::iter::repeat_n(row[width - 1], PAD));
        for (phase, samples) in self.phases.iter_mut().enumerate() {
            samples.resize(width, 0.0);
            let start = (PAD + phase).wrapping_add_signed(kernel.first);
            let taps = core::array::from_fn(|tap| &self.padded[start + tap..]);
            combine(taps, &kernel.phases[phase], samples);
        }
        let pairs = self.phases[0].iter().zip(&self.phases[1]);
        for (pair, (&even, &odd)) in out.as_chunks_mut::<2>().0.iter_mut().zip(pairs) {
            *pair = [even, odd];
        }
    }
}

/// 2x upscaling with working buffers kept between calls.
#[derive(Debug, Default)]
pub struct Resampler {
    row: RowUpscaler,
    /// The horizontally upscaled source rows the vertical pass is reading:
    /// as many as the kernel has taps, row `r` in slot `r` modulo that. Only
    /// these few rows are kept so that they stay in the cache.
    recent: Vec<f32>,
}

impl Resampler {
    /// Bicubic 2x upscale of a `height` x `width` plane, clamped to [0, 1]:
    /// the conventional upscale the network's detail is added to.
    pub fn bicubic2x(&mut self, src: &[f32], size: (usize, usize), out: &mut [f32]) {
        self.upscale(src, size, out, &Kernel::new(cubic_weights, 1), |v| {
            v.clamp(0.0, 1.0)
        });
    }

    /// Bilinear 2x upscale (chroma to luma size; a flow field to output size).
    pub fn bilinear2x(&mut self, src: &[f32], size: (usize, usize), out: &mut [f32]) {
        self.upscale(src, size, out, &Kernel::new(|t| [1.0 - t, t], 0), |v| v);
    }

    fn upscale<const N: usize>(
        &mut self,
        src: &[f32],
        (height, width): (usize, usize),
        out: &mut [f32],
        kernel: &Kernel<N>,
        limit: impl Fn(f32) -> f32,
    ) {
        let wide = 2 * width;
        self.recent.resize(N * wide, 0.0);
        let last = height as isize - 1;
        let mut widened = 0; // source rows upscaled horizontally so far
        for (index, target) in out.chunks_exact_mut(wide).enumerate() {
            let first = (index / 2 + index % 2) as isize + kernel.first;
            let source = |tap: usize| (first + tap as isize).clamp(0, last) as usize;
            while widened <= source(N - 1) {
                let slot = &mut self.recent[(widened % N) * wide..][..wide];
                self.row
                    .upscale(&src[widened * width..][..width], kernel, slot);
                widened += 1;
            }
            let rows = core::array::from_fn(|tap| &self.recent[(source(tap) % N) * wide..][..wide]);
            combine(rows, &kernel.phases[index % 2], target);
            for value in target.iter_mut() {
                *value = limit(*value);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_planes_stay_flat() {
        let src = [0.5f32; 12];
        let mut out = [0.0f32; 48];
        let mut resampler = Resampler::default();
        resampler.bicubic2x(&src, (3, 4), &mut out);
        assert!(out.iter().all(|v| (v - 0.5).abs() < 1e-6));
        resampler.bilinear2x(&src, (3, 4), &mut out);
        assert!(out.iter().all(|v| (v - 0.5).abs() < 1e-6));
    }

    #[test]
    fn bilinear_samples_sit_a_quarter_of_the_way_between_neighbours() {
        // One source row: the vertical pass only repeats it.
        let mut out = [0.0f32; 8];
        Resampler::default().bilinear2x(&[0.0, 1.0], (1, 2), &mut out);
        assert_eq!(out, [0.0, 0.25, 0.75, 1.0, 0.0, 0.25, 0.75, 1.0]);
    }
}
