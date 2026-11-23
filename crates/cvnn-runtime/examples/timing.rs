//! Times each step of the runtime on a synthetic moving frame:
//! `cargo run --release -p cvnn-runtime --example timing [width height]`.

use std::time::Instant;

use cvnn_preproc::{PreprocConfig, Preprocessor};
use cvnn_runtime::colour::{Yuv420, rgba_to_yuv420, yuv444_to_rgba};
use cvnn_runtime::grain::GrainProfile;
use cvnn_runtime::{Input, ModelSpec, Session, Temporal};

const FRAMES: usize = 60;
/// Hidden-state channels of the temporal models timed (the trained defaults).
const BLEND_STATE_CHANNELS: usize = 8;
const RECURRENT_STATE_CHANNELS: usize = 16;

/// A smooth, non-repeating texture shifted by `shift` samples, as RGBA.
fn frame(width: usize, height: usize, shift: usize) -> Vec<u8> {
    let mut rgba = vec![255u8; width * height * 4];
    for (index, pixel) in rgba.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let (x, y) = ((index % width + shift) as f32, (index / width) as f32);
        let value =
            128.0 + 60.0 * (x * 0.071).sin() * (y * 0.053).cos() + 40.0 * (x * y * 0.0007).sin();
        pixel[..3].fill(value as u8);
    }
    rgba
}

/// Milliseconds a step took on each frame.
#[derive(Default)]
struct Timings(Vec<f64>);

impl Timings {
    fn time<T>(&mut self, step: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let result = step();
        self.0.push(start.elapsed().as_secs_f64() * 1e3);
        result
    }

    /// Print the median (other work on the machine stretches the mean) and the best.
    fn report(mut self, name: &str) {
        self.0.sort_by(f64::total_cmp);
        let (median, best) = (self.0[self.0.len() / 2], self.0[0]);
        println!("{name:<20}{median:>8.2} ms   (best {best:.2})");
    }
}

fn main() {
    let mut sizes = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("a number"));
    let (width, height) = (sizes.next().unwrap_or(640), sizes.next().unwrap_or(360));
    let frames: Vec<_> = (0..FRAMES).map(|i| frame(width, height, 2 * i)).collect();

    let mut analysis = Preprocessor::new(width, height, PreprocConfig::default()).expect("size");
    let (mut convert, mut analyse) = (Timings::default(), Timings::default());
    let mut yuv = Yuv420::new(width, height);
    for rgba in &frames {
        convert.time(|| rgba_to_yuv420(rgba, width, &mut yuv));
        analyse.time(|| analysis.push(&yuv.y).expect("frame"));
    }
    convert.report("RGBA to YUV");
    analyse.report("analysis alone");

    let full = ModelSpec {
        planes: 1,
        temporal: Temporal::None,
        align: true,
        state_channels: BLEND_STATE_CHANNELS,
        input: Input::Picture { side_planes: true },
    };
    let blend = ModelSpec {
        temporal: Temporal::Blend,
        ..full
    };
    let recurrent = ModelSpec {
        temporal: Temporal::Recurrent,
        state_channels: RECURRENT_STATE_CHANNELS,
        input: Input::Luma { confidence: true },
        ..full
    };
    // Each design without grain, and one with: what `finish` then takes longer is the grain's cost.
    let (off, on) = (GrainProfile::None, GrainProfile::Webcam);
    for (spec, grain) in [(full, off), (blend, off), (recurrent, off), (recurrent, on)] {
        let mut session = Session::new(spec, width, height).expect("size");
        session.set_grain(grain);
        let fine = vec![0.0f32; spec.fine_channels() * 4 * width * height];
        let coarse = vec![0.0f32; spec.hidden_channels() * width * height];
        let mut out = vec![0u8; 16 * width * height];
        let mut steps = [(); 3].map(|()| Timings::default());
        for rgba in &frames {
            rgba_to_yuv420(rgba, width, &mut yuv);
            steps[0].time(|| {
                session.prepare(&yuv.y, &yuv.u, &yuv.v).expect("frame");
            });
            let picture = steps[1].time(|| session.finish(&fine, &coarse).expect("frame"));
            steps[2].time(|| yuv444_to_rgba(picture, &mut out));
        }
        println!("temporal = {:?}, grain profile {grain:?}", spec.temporal);
        for (timings, name) in steps
            .into_iter()
            .zip(["  prepare", "  finish", "  YUV to RGBA"])
        {
            timings.report(name);
        }
    }
}
