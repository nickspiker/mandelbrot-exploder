use std::sync::atomic::{AtomicU8, Ordering};

use num_complex::{Complex, ComplexFloat};

use crate::compute::mandelbrot_f64::MandelbrotResult;

/// Calculates RGB color using BOTH flame variants (linear and iteration-weighted)
///
/// # Arguments
/// * `_point` - Original complex point (unused but kept for API compatibility)
/// * `result` - Mandelbrot calculation result containing period, escape velocity, coordinates, flames, slope, etc.
/// * `flame_iter` - Iteration-weighted flame variant (weighted by taper * iter)
/// * `_zoom` - Current zoom level (unused but kept for API compatibility)
///
/// # Returns
/// RGBA color as [R, G, B, A] where each component is 0-255
///
/// # Channel Assignment
/// - Red: flame (linear taper-weighted)
/// - Green: iteration count (escape_velocity)
/// - Blue: flame_iter (iteration-weighted)

/// Selected theme (index into `THEME_NAMES`), read per pixel by every calculation thread.
pub static THEME: AtomicU8 = AtomicU8::new(0);
pub const THEME_NAMES: &[&str] = &["classic", "bands", "log bands", "distance", "flame"];

pub fn set_theme(index: usize) {
    THEME.store((index % THEME_NAMES.len()) as u8, Ordering::Relaxed);
}

pub fn theme() -> usize {
    THEME.load(Ordering::Relaxed) as usize % THEME_NAMES.len()
}

/// Parse `MANDEL_THEME` (a name from `THEME_NAMES` or an index).
pub fn theme_from_spec(spec: &str) -> Option<usize> {
    let spec = spec.trim().to_lowercase();
    if let Ok(i) = spec.parse::<usize>() {
        return (i < THEME_NAMES.len()).then_some(i);
    }
    THEME_NAMES.iter().position(|n| *n == spec)
}

/// RGBA colour for a pixel under the selected theme. `zoom` sets the pixel scale the
/// magnification-invariant themes normalise by.
pub fn calculate_colour(
    point: Complex<f64>,
    result: &MandelbrotResult,
    zoom: Complex<f64>,
) -> [u8; 4] {
    match theme() {
        1 => bands(point, result, zoom, false),
        2 => bands(point, result, zoom, true),
        3 => distance(result, zoom),
        4 => flame(point, result, zoom),
        _ => bytes_sqrt(classic_rgb(point, result, zoom)),
    }
}

/// The classic output stage: sqrt as a gamma, scaled to bytes.
fn bytes_sqrt(rgb: [f64; 3]) -> [u8; 4] {
    let b = |v: f64| ((v as f32).sqrt() * 256.) as u8;
    [b(rgb[0]), b(rgb[1]), b(rgb[2]), 255]
}

fn bytes(rgb: [f64; 3]) -> [u8; 4] {
    let b = |v: f64| if v.is_finite() { (v.clamp(0., 1.) * 255.999) as u8 } else { 0 };
    [b(rgb[0]), b(rgb[1]), b(rgb[2]), 255]
}

/// Smooth cyclic palette (one full cycle per unit of `t`).
fn cyclic(t: f64) -> [f64; 3] {
    let f = |phase: f64| 0.5 + 0.5 * (std::f64::consts::TAU * (t + phase)).cos();
    [f(0.0), f(1.0 / 3.0), f(2.0 / 3.0)]
}

fn hsv(h: f64, s: f64, v: f64) -> [f64; 3] {
    let h = (h.rem_euclid(1.0)) * 6.0;
    let i = h.floor();
    let f = h - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match i as i32 {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

/// Distance estimate in pixels: slope = ln|z|²·z̄/ō at the escape, so DE = |slope| / 2; a
/// pixel is about |zoom| / 1000 in the plane. Zero when undefined.
fn de_pixels(result: &MandelbrotResult, zoom: Complex<f64>) -> f64 {
    let de = 0.5 * result.slope.norm();
    let px = zoom.norm() * 1e-3;
    if de.is_finite() && px > 0.0 && px.is_finite() {
        de / px
    } else {
        0.0
    }
}

fn escape_count(result: &MandelbrotResult) -> f64 {
    let ev = result.escape_velocity;
    if ev.is_finite() && ev > 1.0 {
        ev
    } else {
        1.0
    }
}

/// Iteration bands through a cyclic palette, linear in the count (one cycle per 24
/// iterations) or in its logarithm (two cycles per e-fold, which keeps the band width
/// constant across magnifications), darkened toward the boundary by the distance estimate.
fn bands(point: Complex<f64>, result: &MandelbrotResult, zoom: Complex<f64>, log: bool) -> [u8; 4] {
    if result.period != 0 {
        return bytes_sqrt(classic_rgb(point, result, zoom));
    }
    let ev = escape_count(result);
    let t = if log { ev.ln() * 2.0 } else { ev / 24.0 };
    let base = cyclic(t);
    let edge = 1.0 - 0.7 * (-de_pixels(result, zoom) / 1.5).exp();
    bytes([base[0] * edge, base[1] * edge, base[2] * edge])
}

/// Magnification-invariant: the exterior is shaded by the distance estimate in pixels (dark
/// at the boundary, hue cycling with its logarithm), the interior by the pixel's offset from
/// its component's nucleus, which is a delta-scale quantity and so resolves at any depth.
fn distance(result: &MandelbrotResult, zoom: Complex<f64>) -> [u8; 4] {
    let px = zoom.norm() * 1e-3;
    if result.period != 0 {
        let diff = result.slope.conj();
        if !diff.re.is_finite() || !diff.im.is_finite() || px <= 0.0 {
            return bytes([0.3, 0.3, 0.3]);
        }
        let r = (diff.norm() / px).max(1e-9);
        let hue = diff.arg() / std::f64::consts::TAU;
        let ring = 0.5 + 0.5 * (std::f64::consts::TAU * r.log2()).cos();
        return bytes(hsv(hue, 0.55, 0.4 + 0.45 * ring));
    }
    let d = de_pixels(result, zoom).max(1e-6);
    let hue = d.log2() / 10.0;
    let v = 1.0 - (-d / 4.0).exp();
    let contour = 0.85 + 0.15 * (std::f64::consts::TAU * escape_count(result) / 4.0).cos();
    bytes(hsv(hue, 0.6, v * contour))
}

/// Hue from the direction of the iteration-weighted flame, saturation from its magnitude,
/// darkened toward the boundary.
fn flame(point: Complex<f64>, result: &MandelbrotResult, zoom: Complex<f64>) -> [u8; 4] {
    if result.period != 0 {
        return bytes_sqrt(classic_rgb(point, result, zoom));
    }
    let f = if result.flame_c.re.is_finite() && result.flame_c.im.is_finite() {
        result.flame_c
    } else {
        result.flame_q
    };
    if !f.re.is_finite() || !f.im.is_finite() {
        return bytes([0.3, 0.3, 0.3]);
    }
    let hue = f.arg() / std::f64::consts::TAU;
    let sat = (f.norm() * 1.5).clamp(0.2, 1.0);
    let v = 1.0 - 0.6 * (-de_pixels(result, zoom) / 2.0).exp();
    bytes(hsv(hue, sat, v))
}

fn classic_rgb(point: Complex<f64>, result: &MandelbrotResult, zoom: Complex<f64>) -> [f64; 3] {
    let mut red: f64;
    let mut green: f64;
    let mut blue: f64;

    if result.period != 0 {
        // Inside the set - use period-based colouring with coordinate shading
        let mut hash = result.period;
        hash = hash.rotate_left(2);
        hash ^= result.period;
        hash = hash.wrapping_mul(2381);
        hash = hash.wrapping_mul(hash + 13);
        red = (hash % 6229) as f64 / 6229.;
        hash = hash.rotate_left(3);
        hash ^= result.period;
        hash = hash.wrapping_mul(37);
        hash = hash.wrapping_mul(hash + 17);
        green = (hash % 6247) as f64 / 6247.;
        hash = hash.rotate_left(5);
        hash ^= result.period;
        hash = hash.wrapping_mul(43);
        hash = hash.wrapping_mul(hash + 3169);
        blue = (hash % 6271) as f64 / 6271.;

        // Coordinate shading
        let (_mag, mut angle) = result.coords.to_polar();
        let magnitude = result.coords.norm_sqr();
        hash = hash.wrapping_mul(67);
        hash = hash.wrapping_mul(hash + 12153);
        let step = hash % 7 + 1;
        hash = hash.wrapping_mul(73);
        let spiral_scale = (hash % 5) as f64 - 2.;
        hash = hash.wrapping_mul(897);
        hash = hash.wrapping_mul(hash + 5);
        let x = (hash % 118741) as f64 / 118741.;
        let ripples = ((1. - x).ln() * magnitude * 8.).sin() / 4. + 0.75;

        angle = (angle - magnitude * spiral_scale) * step as f64;
        red = ((Complex::from_polar(magnitude, angle)).re + 1.) / 2. * magnitude
            + (1. - magnitude) * red;
        green = ((Complex::from_polar(magnitude, angle + std::f64::consts::FRAC_PI_3 * 2.)).re
            + 1.)
            / 2.
            * magnitude
            + (1. - magnitude) * green;
        blue = ((Complex::from_polar(magnitude, angle - std::f64::consts::FRAC_PI_3 * 2.)).re + 1.)
            / 2.
            * magnitude
            + (1. - magnitude) * blue;
        let taper = (1. - (magnitude.powi(step as i32 + 1))) * ripples;
        red *= taper;
        green *= taper;
        blue *= taper;

        // if result.is_primitive {
        //     red = 1. - red;
        //     green = 1. - green;
        //     blue = 1. - blue;
        // }
    } else {
        // NaN fences: degraded pipelines (small spirix fraction/exponent widths, overflowed
        // f32 orbits) legitimately deliver slope = ±0, non-finite flames, or non-positive
        // escape velocities through the f64 boundary. Each term is fenced individually so a
        // dead term drops out while the surviving terms still paint structure — a NaN anywhere
        // used to zero all three channels, rendering the whole outer region as a black void
        // indistinguishable from "not calculated".
        let mut colour =
            (result.flame_c.conj() - result.flame_q) * result.escape_velocity.ln() / 16.;
        if !colour.re.is_finite() || !colour.im.is_finite() {
            colour = Complex::new(0., 0.);
        }

        // Use flame_iter with 120° triangular rotation for RGB
        let mut ev = (result.escape_velocity.log2() * 2.).cos() / 22.;
        if !ev.is_finite() {
            ev = 0.;
        }
        let slope_mag_sq = result.slope.norm_sqr();
        let sns = if slope_mag_sq > 0. && slope_mag_sq.is_finite() {
            slope_mag_sq.log2()
        } else {
            // Vanished slope: below every representable magnitude — floor at f64's subnormal
            // limit so the sin() shading stays defined and these pixels read as their own band.
            -2098.
        };
        red = ev + sns.sin() / 12. + 0.21;
        green = ev + (sns * 1.2).sin() / 12. + 0.21;
        blue = ev + (sns * 1.7).sin() / 12. + 0.21;

        // 120° rotation: multiply by e^(i·2π/3) = -0.5 + i·√3/2
        let rot_120 = Complex::new(-0.5, 0.866025403784439); // 120° rotation
        let rot_minus_120 = Complex::new(-0.5, -0.866025403784439); // -120° rotation

        // Triangular c mapping with 120° phase shifts
        red += colour.re;
        green += (colour * rot_120).re;
        blue += (colour * rot_minus_120).re;

        if slope_mag_sq > 0. {
            let slope_angle = result.slope / result.slope.abs() * 0.;
            red += slope_angle.re;
            green += (slope_angle * rot_120).re;
            blue += (slope_angle * rot_minus_120).re;
        }
    }

    [red, green, blue]
}
