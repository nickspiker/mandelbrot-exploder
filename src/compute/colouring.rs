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
pub fn calculate_colour(
    point: Complex<f64>,
    result: &MandelbrotResult,
    zoom: Complex<f64>,
) -> [u8; 4] {
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

    let red_byte = ((red as f32).sqrt() * 256.) as u8;
    let green_byte = ((green as f32).sqrt() * 256.) as u8;
    let blue_byte = ((blue as f32).sqrt() * 256.) as u8;

    [red_byte, green_byte, blue_byte, 255]
}
