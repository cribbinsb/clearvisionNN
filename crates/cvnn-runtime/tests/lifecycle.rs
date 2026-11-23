//! The session's call-order rules and recovery.

use cvnn_runtime::grain::GrainProfile;
use cvnn_runtime::{Input, ModelSpec, Session, Temporal};

const WIDTH: usize = 64;
const HEIGHT: usize = 64;
const STATE: usize = 4;

fn session(planes: usize) -> cvnn_common::Result<Session> {
    let spec = ModelSpec {
        planes,
        temporal: Temporal::Blend,
        align: true,
        state_channels: STATE,
        input: Input::Picture { side_planes: true },
    };
    let mut session = Session::new(spec, WIDTH, HEIGHT)?;
    // These tests read samples of the picture itself.
    session.set_grain(GrainProfile::None);
    Ok(session)
}

fn recurrent_session() -> Session {
    let spec = ModelSpec {
        planes: 1,
        temporal: Temporal::Recurrent,
        align: true,
        state_channels: STATE,
        input: Input::Luma { confidence: true },
    };
    Session::new(spec, WIDTH, HEIGHT).unwrap()
}

fn frame(value: u8) -> (Vec<u8>, Vec<u8>) {
    (vec![value; WIDTH * HEIGHT], vec![128; WIDTH * HEIGHT / 4])
}

/// Finish with an all-zero network output; returns the first luma sample.
fn finish(session: &mut Session) -> cvnn_common::Result<f32> {
    let fine = vec![0.0; 2 * 4 * WIDTH * HEIGHT];
    let coarse = vec![0.0; STATE * WIDTH * HEIGHT];
    session
        .finish(&fine, &coarse)
        .map(|planes| planes.plane(0)[0])
}

#[test]
fn an_unsupported_contract_is_an_error_not_a_panic() {
    assert!(session(4).is_err());
}

#[test]
fn a_frame_must_be_finished_before_the_next_is_prepared() {
    let mut session = session(1).unwrap();
    let (y, c) = frame(100);
    session.prepare(&y, &c, &c).unwrap();
    assert!(session.prepare(&y, &c, &c).is_err());
    assert!(finish(&mut session).is_ok());
    assert!(finish(&mut session).is_err());
    assert!(session.prepare(&y, &c, &c).is_ok());
}

#[test]
fn restart_recovers_from_an_abandoned_frame_and_drops_history() {
    let mut session = session(1).unwrap();
    let (dark, c) = frame(40);
    session.prepare(&dark, &c, &c).unwrap();
    finish(&mut session).unwrap();
    let (bright, _) = frame(200);
    session.prepare(&bright, &c, &c).unwrap(); // the network call "fails" here
    session.restart().unwrap();
    session.prepare(&bright, &c, &c).unwrap();
    let luma = finish(&mut session).unwrap();
    assert!(
        (luma - 200.0 / 255.0).abs() < 1e-3,
        "history leaked across the restart"
    );
}

#[test]
fn the_recurrent_input_carries_confidence_and_history() {
    let mut session = recurrent_session();
    let plane = WIDTH * HEIGHT;
    let (y, c) = frame(100);
    let first = session.prepare(&y, &c, &c).unwrap().to_vec();
    assert_eq!(first.len(), (1 + 1 + STATE + 4) * plane);
    // No history yet: full confidence and an empty state.
    assert!(
        first[plane..2 * plane]
            .iter()
            .all(|&v| (v - 1.0).abs() < 1e-6)
    );
    assert!(
        first[2 * plane..(2 + STATE) * plane]
            .iter()
            .all(|&v| v == 0.0)
    );

    let fine = vec![0.0; 4 * plane];
    session.finish(&fine, &vec![0.5; STATE * plane]).unwrap();
    let second = session.prepare(&y, &c, &c).unwrap();
    // The state the network returned comes back untouched.
    assert!(
        second[2 * plane..(2 + STATE) * plane]
            .iter()
            .all(|&v| (v - 0.5).abs() < 1e-6)
    );
}
