//! The pre-process on synthetic frames with known motion.

use cvnn_preproc::{PreprocConfig, Preprocessor};

const WIDTH: usize = 192;
const HEIGHT: usize = 128;

/// Pseudo-random value of a lattice point, in 0..=255.
fn lattice(x: i32, y: i32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x9E37_79B1) ^ (y as u32).wrapping_mul(0x85EB_CA77);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    ((h >> 12) & 0xFF) as f32
}

/// Non-repeating smooth texture (bilinear value noise, 4-sample cells)
/// sampled at a fractional position, so every block has a unique match.
fn texture(x: f32, y: f32) -> u8 {
    const CELL: f32 = 4.0;
    let (gx, gy) = (x / CELL, y / CELL);
    let (ix, iy) = (gx.floor(), gy.floor());
    let (fx, fy) = (gx - ix, gy - iy);
    let (ix, iy) = (ix as i32, iy as i32);
    let top = lattice(ix, iy) * (1.0 - fx) + lattice(ix + 1, iy) * fx;
    let bottom = lattice(ix, iy + 1) * (1.0 - fx) + lattice(ix + 1, iy + 1) * fx;
    (top * (1.0 - fy) + bottom * fy).round() as u8
}

fn frame(shift_x: f32, shift_y: f32) -> Vec<u8> {
    (0..HEIGHT)
        .flat_map(|y| (0..WIDTH).map(move |x| texture(x as f32 + shift_x, y as f32 + shift_y)))
        .collect()
}

fn interior_mean(
    values: impl Iterator<Item = (usize, usize, f32)>,
    cols: usize,
    rows: usize,
) -> f32 {
    let inside = values
        .filter(|&(c, r, _)| c >= 4 && r >= 4 && c + 4 < cols && r + 4 < rows)
        .map(|(_, _, v)| v)
        .collect::<Vec<_>>();
    inside.iter().sum::<f32>() / inside.len() as f32
}

#[test]
fn first_frame_is_a_reset_with_zero_motion() {
    let mut pre = Preprocessor::new(WIDTH, HEIGHT, PreprocConfig::default()).unwrap();
    let features = pre.push(&frame(0.0, 0.0)).unwrap();
    assert!(features.reset);
    assert!(
        features
            .motion
            .cells()
            .iter()
            .all(|v| v.dx == 0.0 && v.dy == 0.0)
    );
}

#[test]
fn sub_sample_translation_is_recovered() {
    let (shift_x, shift_y) = (2.5, -1.0);
    let mut pre = Preprocessor::new(WIDTH, HEIGHT, PreprocConfig::default()).unwrap();
    pre.push(&frame(0.0, 0.0)).unwrap();
    // The scene content moves by (+shift) in sampling position, so the block
    // at p in the new frame matches the old frame at p + shift.
    let features = pre.push(&frame(shift_x, shift_y)).unwrap();
    assert!(!features.reset);
    let (cols, rows) = (features.motion.cols(), features.motion.rows());
    let cells = || features.motion.positions().zip(features.motion.cells());
    let dx = interior_mean(cells().map(|((c, r), v)| (c, r, v.dx)), cols, rows);
    let dy = interior_mean(cells().map(|((c, r), v)| (c, r, v.dy)), cols, rows);
    assert!((dx - shift_x).abs() < 0.25, "dx {dx}");
    assert!((dy - shift_y).abs() < 0.25, "dy {dy}");
}

#[test]
fn a_different_scene_resets() {
    let mut pre = Preprocessor::new(WIDTH, HEIGHT, PreprocConfig::default()).unwrap();
    pre.push(&frame(0.0, 0.0)).unwrap();
    let other = (0..WIDTH * HEIGHT)
        .map(|i| ((i * 7919) % 251) as u8)
        .collect::<Vec<_>>();
    assert!(pre.push(&other).unwrap().reset);
}

#[test]
fn wrong_frame_length_is_rejected() {
    let mut pre = Preprocessor::new(WIDTH, HEIGHT, PreprocConfig::default()).unwrap();
    assert!(pre.push(&[0; 16]).is_err());
}
