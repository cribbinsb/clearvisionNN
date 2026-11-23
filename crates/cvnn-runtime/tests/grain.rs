//! Grain on a session's pictures: it changes what is shown and nothing else.

use cvnn_runtime::grain::GrainProfile;
use cvnn_runtime::{Input, ModelSpec, Session, Temporal};

const WIDTH: usize = 64;
const HEIGHT: usize = 64;
const STATE: usize = 4;
const PLANE: usize = WIDTH * HEIGHT;
const FRAMES: usize = 3;
/// Grey levels of the session's scale of 0 to 1.
const LEVELS: f32 = 255.0;

fn session() -> Session {
    let spec = ModelSpec {
        planes: 1,
        temporal: Temporal::Recurrent,
        align: true,
        state_channels: STATE,
        input: Input::Luma { confidence: true },
    };
    Session::new(spec, WIDTH, HEIGHT).unwrap()
}

fn session_with(profile: GrainProfile) -> Session {
    let mut session = session();
    session.set_grain(profile);
    session
}

/// A frame that differs from the one before it, so that history matters.
fn luma(frame: usize) -> Vec<u8> {
    (0..PLANE)
        .map(|at| (60 + 3 * (at % WIDTH + frame) + at / WIDTH) as u8)
        .collect()
}

/// Run `FRAMES` frames with a fixed network output; the inputs the network
/// was given and the lumas handed back.
fn run(session: &mut Session) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let chroma = vec![128; PLANE / 4];
    let (fine, coarse) = (vec![0.01; 4 * PLANE], vec![0.25; STATE * PLANE]);
    let (mut inputs, mut shown) = (Vec::new(), Vec::new());
    for frame in 0..FRAMES {
        let input = session.prepare(&luma(frame), &chroma, &chroma).unwrap();
        inputs.push(input.to_vec());
        let picture = session.finish(&fine, &coarse).unwrap();
        shown.push(picture.plane(0).to_vec());
    }
    (inputs, shown)
}

fn deviation(a: &[f32], b: &[f32]) -> f32 {
    let squares: f32 = a.iter().zip(b).map(|(p, q)| (p - q).powi(2)).sum();
    (squares / a.len() as f32).sqrt() * LEVELS
}

#[test]
fn a_session_shows_the_webcam_profile_until_told_otherwise() {
    assert_eq!(
        run(&mut session()),
        run(&mut session_with(GrainProfile::Webcam))
    );
}

#[test]
fn grain_is_shown_and_never_reaches_the_network_or_history() {
    let (clean_inputs, clean) = run(&mut session_with(GrainProfile::None));
    for profile in [GrainProfile::Produced, GrainProfile::Webcam] {
        let (inputs, shown) = run(&mut session_with(profile));
        assert_eq!(
            inputs, clean_inputs,
            "grain leaked into what the network is given"
        );
        for (frame, (with, without)) in shown.iter().zip(&clean).enumerate() {
            let added = deviation(with, without);
            assert!(
                (added / profile.level() - 1.0).abs() < 0.05,
                "{profile:?}, frame {frame}: {added} grey levels"
            );
        }
        assert_ne!(shown[0], shown[1]);
    }
}

#[test]
fn a_restart_starts_the_grain_again() {
    let mut stream = session();
    let (_, first) = run(&mut stream);
    stream.restart().unwrap();
    assert_eq!(run(&mut stream).1, first);
}
