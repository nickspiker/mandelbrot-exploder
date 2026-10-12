use num_complex::Complex64;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use std::time::{Duration, Instant};

use crate::{compute::calculate_mandelbrot_point_f64, config::MAX_ITERATIONS};

/// Traces a path from infinity to the boundary of the Mandelbrot set containing a given starting point
///
/// First traces inward from src_point to find the boundary, then traces outward from the boundary.
///
/// # Arguments
/// * `src_point` - Starting complex point
///
/// # Returns
/// Vector of complex points forming a continuous path from boundary to infinity
pub fn trace_path(src_point: Complex64) -> Vec<Complex64> {
    // First, trace inward to find the boundary point using f64
    let mut point = src_point;
    let mut checkpoint_re = point.re;
    let mut next_checkpoint = 1;
    let mut step = 0;

    println!(
        "TRACE INWARD: starting from ({:.6e}, {:.6e})",
        point.re, point.im
    );

    let mut path: Vec<Complex64> = Vec::new();
    while point.norm_sqr() < f64::MAX.sqrt() {
        let result = calculate_mandelbrot_point_f64(point, MAX_ITERATIONS as f64, false);

        let distance = 1. / 12.;

        let new_point = point - distance * result.slope.conj();

        if new_point.re.to_be_bytes() == checkpoint_re.to_be_bytes() {
            println!("TRACE INWARD: returned to checkpoint (step {})", step);
            break;
        }

        step += 1;
        if step > next_checkpoint {
            next_checkpoint <<= 1;
            checkpoint_re = point.re;
        }
        point = new_point;
        println!("({:.6e}, {:.6e})", point.re, point.im);
        // path.push(point);
    }

    // // Now trace outward from the boundary point using f64
    let mut jitter = 1. / (1u64 << 63) as f64;
    let mut rng = thread_rng();
    let mut outward_step = 0;

    loop {
        let result = calculate_mandelbrot_point_f64(point, MAX_ITERATIONS as f64, false);

        let distance = 1. / 12.;

        let new_point = point + distance * result.slope.conj();

        // Log every step
        println!(
            "TRACE OUTWARD step={}: point=({:.6e}, {:.6e}), distance={:.6e}",
            outward_step, point.re, point.im, distance
        );

        // If we can't move (stuck at same point), we need to jitter or bail
        if new_point == point {
            println!(
                "TRACE OUTWARD: stuck at same point (step {}), trying jitter",
                outward_step
            );
            let normal = Normal::new(0., jitter).unwrap();
            point = point + Complex64::new(normal.sample(&mut rng), normal.sample(&mut rng));
            jitter = jitter * 2.;
            path.clear();
            continue;
        }

        path.push(point);

        if !point.is_finite() || point.norm_sqr() > f64::MAX.sqrt().sqrt() {
            println!(
                "TRACE OUTWARD: point became non-finite after {} steps",
                outward_step
            );
            break;
        }
        point = new_point;
        outward_step += 1;
    }

    path
}

/// Converts boundary path to audio using FFT-based direction analysis
///
/// Each segment is normalized by its length to create bounded audio samples (-1 to 1).
/// Real components go to left channel, imaginary to right channel.
///
/// # Arguments
/// * `path` - Vector of complex points forming the path
///
/// # Returns
/// * Tuple of (left_channel, right_channel) audio samples
pub fn path_to_audio_fft(path: &[Complex64]) -> (Vec<f32>, Vec<f32>) {
    if path.len() < 2 {
        return (Vec::new(), Vec::new());
    }

    let mut left = Vec::with_capacity(path.len() - 1);
    let mut right = Vec::with_capacity(path.len() - 1);

    for i in 1..path.len() {
        let segment = path[i] - path[i - 1];
        let length = segment.norm();

        if length > 0.0 {
            // Normalize by segment length to get bounded values
            let l = (segment.re / length) as f32;
            let r = (segment.im / length) as f32;
            left.push(l);
            right.push(r);
        } else {
            // Zero-length segment
            left.push(0.0);
            right.push(0.0);
        }
    }

    (left, right)
}

/// Converts a clicked point to audio by playing raw z iteration values
///
/// Iterates the Mandelbrot function at the clicked point and outputs the real/imaginary
/// components of z at each iteration directly as audio samples. Values are normalized:
/// - Inside the set (bounded): Normalize to [-1, 1] by dividing by 2, play for 1 second at 44100 Hz
/// - Outside the set (escaping): Use 0.25/|z| to bring large values into audio range, play until escape
///
/// # Arguments
/// * `point` - Complex point to iterate
/// * `max_iterations` - Maximum iterations before stopping
///
/// # Returns
/// * Tuple of (left_channel, right_channel) audio samples
pub fn path_to_audio(point: Complex64, max_iterations: u64) -> (Vec<f32>, Vec<f32>) {
    const SAMPLE_RATE: usize = 44100;
    const MAX_SAMPLES: usize = SAMPLE_RATE; // 1 second

    // First, check if point is inside or outside using calculate_mandelbrot_point_f64 with logging
    use crate::compute::mandelbrot_f64::calculate_mandelbrot_point_f64;
    let result = calculate_mandelbrot_point_f64(point, max_iterations as f64, true);

    let is_inside = result.period != 0;

    let mut left = Vec::with_capacity(max_iterations as usize);
    let mut right = Vec::with_capacity(max_iterations as usize);

    let mut z = Complex64::new(0.0, 0.0);

    for _ in 0..max_iterations {
        // z_{n+1} = z_n^2 + c
        z = z * z + point;

        let z_mag_sq = z.norm_sqr();

        if is_inside {
            // Inside: scale down by dividing by 2 (max magnitude is ~2)
            let l = (z.re / 2.0) as f32;
            let r = (z.im / 2.0) as f32;
            left.push(l);
            right.push(r);

            // For inside points, stop after 1 second worth of samples (44100 samples)
            if left.len() >= MAX_SAMPLES {
                break;
            }
        } else {
            // Outside: invert (1/z)
            // 1/z = conjugate(z) / |z|^2
            let z_inverted = Complex64::new(z.re / z_mag_sq, -z.im / z_mag_sq);
            let l = z_inverted.re as f32;
            let r = z_inverted.im as f32;
            left.push(l);
            right.push(r);
        }

        // Stop if diverged to infinity
        if !z.is_finite() {
            break;
        }
    }

    (left, right)
}
