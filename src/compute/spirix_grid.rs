//! Macro-instantiated Mandelbrot pipelines over the full spirix (fraction × exponent) grid.
//!
//! This is the point of the app: exercise spirix across every Scalar/Circle width combo and
//! watch where the pipeline's intermediates (escape velocities, flame/energy sums — NOT the
//! coordinates) leave normal range. Coordinates stay pinned at F7E3 (128-bit fraction, 8-bit
//! exponent): coordinate magnitudes never exceed E3's range, and F7's ~2⁻¹²⁷ precision floor
//! saturates at the same zoom depth E3's exponent runs out, so one type covers navigation.
//!
//! Each instantiation is the same algorithm as the old `calculate_mandelbrot_point_f7e5`,
//! parameterized on the working Circle/Scalar type plus the two width-dependent epsilons:
//! `eps_newton` = −(fraction_bits / 2) (√machine-epsilon for Newton convergence) and
//! `eps_root` = −(2 × fraction_bits) (the tiny asymmetric nudge in interior root refinement).
//! The overflow-escalation target stays fixed at F5E7 for every combo — it's the range escape
//! hatch (i64 exponent), deliberately independent of the type under test so small-E runs
//! visibly escalate.

use super::mandelbrot_f64::MandelbrotResult;
use num_complex::Complex64;
use spirix::{
    CircleF3E3, CircleF3E4, CircleF3E5, CircleF3E6, CircleF3E7, CircleF4E3, CircleF4E4,
    CircleF4E5, CircleF4E6, CircleF4E7, CircleF5E3, CircleF5E4, CircleF5E5, CircleF5E6,
    CircleF5E7, CircleF6E3, CircleF6E4, CircleF6E5, CircleF6E6, CircleF6E7, CircleF7E3,
    CircleF7E4, CircleF7E5, CircleF7E6, CircleF7E7, ScalarF3E3, ScalarF3E4, ScalarF3E5,
    ScalarF3E6, ScalarF3E7, ScalarF4E3, ScalarF4E4, ScalarF4E5, ScalarF4E6, ScalarF4E7,
    ScalarF5E3, ScalarF5E4, ScalarF5E5, ScalarF5E6, ScalarF5E7, ScalarF6E3, ScalarF6E4,
    ScalarF6E5, ScalarF6E6, ScalarF6E7, ScalarF7E3, ScalarF7E4, ScalarF7E5, ScalarF7E6,
    ScalarF7E7,
};

/// One grid cell's point function: F7E3 coordinate in, f64-domain result out (colouring is
/// f64 regardless of the working type).
pub type SpirixPointFn = fn(CircleF7E3, u64) -> MandelbrotResult;

macro_rules! spirix_mandelbrot {
    ($fname:ident, $C:ty, $S:ty, $eps_newton:expr, $eps_root:expr) => {
        pub fn $fname(coord: CircleF7E3, max_iterations: u64) -> MandelbrotResult {
            // Coordinate (F7E3) → working type, componentwise via the generic scalar conversions.
            let point: $C = <$C>::from((<$S>::from(coord.r()), <$S>::from(coord.i())));

            let mut inside = false;
            let mut z = point;
            let mut escape_velocity = <$S>::from(max_iterations);
            let mut period = 0u64;
            let mut coord_acc = <$C>::ZERO;
            let mut dist_sum = <$S>::ZERO;
            let mut flame_uniform = <$C>::ZERO;
            let mut flame_weighted = <$C>::ZERO;
            let mut energy_sum_uniform = <$S>::ZERO;
            let mut energy_sum_weighted = <$S>::ZERO;
            let mut slope = <$C>::ONE;
            let mut orbit = <$C>::ONE;
            let mut z_min = <$S>::MAX;
            let mut multiplier;
            let mut slope_result = <$C>::ZERO;

            const MAX_NEWTON_STEPS: isize = 32;
            // sqrt(machine epsilon) for optimal Newton convergence at this fraction width
            let epsilon_squared = <$S>::TWO.pow($eps_newton);

            let mut has_converged = false;

            'main_iteration: for iter in 1..max_iterations {
                let iter_scalar = <$S>::from(iter);

                // Calculate derivative
                let slope_temp = (slope.r() * z.r() - slope.i() * z.i()) << 1u8;
                let slope_im = (slope.r() * z.i() + slope.i() * z.r()) << 1u8;
                let slope_re = slope_temp + <$S>::ONE;
                slope = <$C>::from((slope_re, slope_im));

                let mut z_re_sq = z.r() * z.r();
                let mut z_im_sq = z.i() * z.i();
                let mut z_mag_sq = z_re_sq + z_im_sq;
                let mut iter_sq = iter_scalar.square();

                if z_mag_sq > <$S>::from(4) {
                    // Point is escaping - iterate in the working type until overflow
                    let mut iter_mut = iter;
                    loop {
                        let z_next_re = z_re_sq - z_im_sq + point.r();
                        let z_next_im = ((z.r() * z.i()) << 1u8) + point.i();
                        let z_next = <$C>::from((z_next_re, z_next_im));
                        let z_next_mag_sq = z_next.magnitude_squared();

                        // Would the next z overflow the working type? Break and escalate.
                        if !z_next_mag_sq.is_normal() {
                            break;
                        }

                        z = z_next;
                        z_re_sq = z.r() * z.r();
                        z_im_sq = z.i() * z.i();
                        z_mag_sq = z_re_sq + z_im_sq;
                        iter_mut += 1;
                        iter_sq = <$S>::from(iter_mut).square();

                        // Variant 1: Uniform weighting
                        let taper_uniform = <$S>::ONE;
                        let z_scale_uniform = taper_uniform / z_mag_sq.sqrt();
                        energy_sum_uniform += taper_uniform;
                        flame_uniform = flame_uniform
                            + <$C>::from((z.r() * z_scale_uniform, z.i() * z_scale_uniform));

                        // Variant 2: Iteration-weighted (iter²)
                        let taper_weighted = iter_sq;
                        let z_scale_weighted = taper_weighted / z_mag_sq.sqrt();
                        energy_sum_weighted += taper_weighted;
                        flame_weighted = flame_weighted
                            + <$C>::from((z.r() * z_scale_weighted, z.i() * z_scale_weighted));

                        let z_weight = iter_sq / z_mag_sq;
                        dist_sum += z_weight;
                        coord_acc =
                            coord_acc + <$C>::from((z.r() * z_weight, z.i() * z_weight));

                        // Update derivatives
                        let orbit_temp =
                            ((z.r() * orbit.r() - z.i() * orbit.i()) << 1u8) + <$S>::ONE;
                        let orbit_im = (orbit.r() * z.i() + orbit.i() * z.r()) << 1u8;
                        orbit = <$C>::from((orbit_temp, orbit_im));

                        let slope_temp = (slope.r() * z.r() - slope.i() * z.i()) << 1u8;
                        let slope_im = (slope.r() * z.i() + slope.i() * z.r()) << 1u8;
                        let slope_re = slope_temp + <$S>::ONE;
                        slope = <$C>::from((slope_re, slope_im));
                    }

                    // Working type overflowed - escalate to F5E7 (range hatch) and continue
                    let mut z_f5e7 =
                        CircleF5E7::from((ScalarF5E7::from(z.r()), ScalarF5E7::from(z.i())));
                    let point_f5e7 = CircleF5E7::from((
                        ScalarF5E7::from(point.r()),
                        ScalarF5E7::from(point.i()),
                    ));
                    let mut orbit_f5e7 = CircleF5E7::from((
                        ScalarF5E7::from(orbit.r()),
                        ScalarF5E7::from(orbit.i()),
                    ));
                    let mut coord_f5e7 = CircleF5E7::from((
                        ScalarF5E7::from(coord_acc.r()),
                        ScalarF5E7::from(coord_acc.i()),
                    ));
                    let mut dist_sum_f5e7 = ScalarF5E7::from(dist_sum);
                    let mut flame_uniform_f5e7 = CircleF5E7::from((
                        ScalarF5E7::from(flame_uniform.r()),
                        ScalarF5E7::from(flame_uniform.i()),
                    ));
                    let mut flame_weighted_f5e7 = CircleF5E7::from((
                        ScalarF5E7::from(flame_weighted.r()),
                        ScalarF5E7::from(flame_weighted.i()),
                    ));
                    let mut energy_sum_uniform_f5e7 = ScalarF5E7::from(energy_sum_uniform);
                    let mut energy_sum_weighted_f5e7 = ScalarF5E7::from(energy_sum_weighted);
                    let mut iter_mut = iter_mut;

                    loop {
                        let z_next = z_f5e7.square() + point_f5e7;

                        if !z_next.is_normal() {
                            break;
                        }

                        z_f5e7 = z_next;
                        // |z|² of a normal z can still explode; saturate so lb().lb() stays defined.
                        let z_mag_sq_f5e7 = {
                            let m = z_f5e7.magnitude_squared();
                            if m.is_normal() { m } else { ScalarF5E7::MAX }
                        };
                        iter_mut += 1;
                        let iter_sq_f5e7 = ScalarF5E7::from(iter_mut * iter_mut);

                        let log_log_mag_f5e7 = z_mag_sq_f5e7.lb().lb();
                        let fade_f5e7 = (ScalarF5E7::MAX.lb() - log_log_mag_f5e7)
                            / (ScalarF5E7::MAX.lb() - ScalarF5E7::ONE);

                        let taper_uniform_f5e7 = fade_f5e7;
                        let z_scale_uniform_f5e7 = taper_uniform_f5e7 / z_mag_sq_f5e7.sqrt();
                        energy_sum_uniform_f5e7 = energy_sum_uniform_f5e7 + taper_uniform_f5e7;
                        flame_uniform_f5e7 = flame_uniform_f5e7 + z_f5e7 * z_scale_uniform_f5e7;

                        let taper_weighted_f5e7 = fade_f5e7 * iter_sq_f5e7;
                        let z_scale_weighted_f5e7 = taper_weighted_f5e7 / z_mag_sq_f5e7.sqrt();
                        energy_sum_weighted_f5e7 =
                            energy_sum_weighted_f5e7 + taper_weighted_f5e7;
                        flame_weighted_f5e7 =
                            flame_weighted_f5e7 + z_f5e7 * z_scale_weighted_f5e7;

                        let z_weight_f5e7 = iter_sq_f5e7 / z_mag_sq_f5e7;
                        dist_sum_f5e7 = dist_sum_f5e7 + z_weight_f5e7;
                        coord_f5e7 = coord_f5e7 + z_f5e7 * z_weight_f5e7;

                        let orbit_temp = z_f5e7.r() * orbit_f5e7.r() - z_f5e7.i() * orbit_f5e7.i();
                        let orbit_im_val =
                            orbit_f5e7.r() * z_f5e7.i() + orbit_f5e7.i() * z_f5e7.r();
                        orbit_f5e7 = CircleF5E7::from((
                            orbit_temp + orbit_temp + ScalarF5E7::ONE,
                            orbit_im_val + orbit_im_val,
                        ));
                    }

                    // Final escape velocity. Same saturation guard: the final z is normal but
                    // its squared magnitude may not be.
                    let z_mag_sq_f5e7 = {
                        let m = z_f5e7.magnitude_squared();
                        if m.is_normal() { m } else { ScalarF5E7::MAX }
                    };
                    let escape_f5e7 =
                        ScalarF5E7::from(iter_mut) + ScalarF5E7::ONE - z_mag_sq_f5e7.lb().lb();
                    escape_velocity = <$S>::from(escape_f5e7);

                    // Normalize and convert back to the working type
                    coord_acc = <$C>::from((
                        <$S>::from(coord_f5e7.r() / dist_sum_f5e7),
                        <$S>::from(coord_f5e7.i() / dist_sum_f5e7),
                    ));
                    flame_uniform = <$C>::from((
                        <$S>::from(flame_uniform_f5e7.r() / energy_sum_uniform_f5e7),
                        <$S>::from(flame_uniform_f5e7.i() / energy_sum_uniform_f5e7),
                    ));
                    flame_weighted = <$C>::from((
                        <$S>::from(flame_weighted_f5e7.r() / energy_sum_weighted_f5e7),
                        <$S>::from(flame_weighted_f5e7.i() / energy_sum_weighted_f5e7),
                    ));
                    slope_result = <$C>::from((
                        <$S>::from(orbit_f5e7.r()),
                        <$S>::from(orbit_f5e7.i()),
                    ));

                    has_converged = true;
                    break 'main_iteration;
                } else {
                    // Point might be in set - accumulate BOTH flame variants
                    let taper_uniform = <$S>::ONE;
                    let z_scale_uniform = taper_uniform / z_mag_sq.sqrt();
                    energy_sum_uniform = energy_sum_uniform + taper_uniform;
                    flame_uniform = flame_uniform
                        + <$C>::from((z.r() * z_scale_uniform, z.i() * z_scale_uniform));

                    let taper_weighted = iter_sq;
                    let z_scale_weighted = taper_weighted / z_mag_sq.sqrt();
                    energy_sum_weighted = energy_sum_weighted + taper_weighted;
                    flame_weighted = flame_weighted
                        + <$C>::from((z.r() * z_scale_weighted, z.i() * z_scale_weighted));

                    let z_weight = iter_sq / z_mag_sq;
                    dist_sum += z_weight;
                    coord_acc = coord_acc + <$C>::from((z.r() * z_weight, z.i() * z_weight));

                    if z_min > z_mag_sq {
                        let magnitude_ratio = z_mag_sq / z_min;
                        z_min = z_mag_sq;

                        if magnitude_ratio < <$S>::from(0.25) {
                            let mut w = z;

                            for _ in 0..MAX_NEWTON_STEPS {
                                let mut u = w;
                                let mut du = <$C>::from((<$S>::ONE, <$S>::ZERO));
                                multiplier = <$C>::from((<$S>::ONE, <$S>::ZERO));

                                for _ in 0..iter {
                                    let temp_re = multiplier.r();
                                    let multiplier_re =
                                        (temp_re * u.r() - multiplier.i() * u.i()) << 1u8;
                                    let multiplier_im =
                                        (temp_re * u.i() + multiplier.i() * u.r()) << 1u8;
                                    multiplier = <$C>::from((multiplier_re, multiplier_im));

                                    let du_temp = (du.r() * u.r() - du.i() * u.i()) << 1u8;
                                    let du_im = (du.r() * u.i() + du.i() * u.r()) << 1u8;
                                    du = <$C>::from((du_temp, du_im));

                                    let u_re = u.r() * u.r() - u.i() * u.i() + point.r();
                                    let u_im = ((u.r() * u.i()) << 1u8) + point.i();
                                    u = <$C>::from((u_re, u_im));
                                }

                                let newton = u - w;
                                let du_offset = du.r() - <$S>::ONE;
                                let du_denom = du_offset * du_offset + du.i() * du.i();

                                let newton_re =
                                    (newton.r() * du_offset + newton.i() * du.i()) / du_denom;
                                let newton_im =
                                    (newton.i() * du_offset - newton.r() * du.i()) / du_denom;
                                let newton = <$C>::from((newton_re, newton_im));

                                iter_sq = newton.r() * newton.r() + newton.i() * newton.i();
                                w = w - newton;

                                if iter_sq < epsilon_squared {
                                    let du_mag_sq = du.r() * du.r() + du.i() * du.i();
                                    if du_mag_sq < <$S>::ONE {
                                        coord_acc = du;
                                        flame_uniform = w;
                                        flame_weighted = w;
                                        inside = true;
                                        period = iter;
                                        has_converged = true;
                                        break 'main_iteration;
                                    }
                                }
                            }
                        }
                    }

                    let orbit_temp =
                        ((z.r() * orbit.r() - z.i() * orbit.i()) << 1u8) + <$S>::ONE;
                    let orbit_im = (orbit.r() * z.i() + orbit.i() * z.r()) << 1u8;
                    orbit = <$C>::from((orbit_temp, orbit_im));

                    // Update z
                    let z_re = z.r() * z.r() - z.i() * z.i() + point.r();
                    let z_im = ((z.r() * z.i()) << 1u8) + point.i();
                    z = <$C>::from((z_re, z_im));
                }
            }

            let coordinates;
            let flame_coords_uniform;
            let flame_coords_weighted;

            if inside {
                coordinates = coord_acc;
                flame_coords_uniform = flame_uniform;
                flame_coords_weighted = flame_weighted;

                let initial_point = point;
                let mut refined_point = initial_point;

                for _ in 0..MAX_NEWTON_STEPS {
                    let mut z = refined_point;
                    let mut d = <$C>::ONE;

                    for _ in 1..period {
                        d = ((d * z) << 1u8) + <$C>::ONE;
                        z = z * z + refined_point;
                    }

                    let epsilon = <$C>::from((<$S>::ZERO, <$S>::TWO.pow($eps_root)));
                    let next = refined_point - z / d + epsilon;

                    if next == refined_point {
                        refined_point = next;
                        break;
                    }
                    refined_point = next;
                }

                let diff = initial_point - refined_point;
                slope_result = <$C>::from((diff.r(), -diff.i()));
            } else {
                dist_sum = <$S>::ONE / dist_sum;
                energy_sum_uniform = <$S>::ONE / energy_sum_uniform;
                energy_sum_weighted = <$S>::ONE / energy_sum_weighted;

                coordinates =
                    <$C>::from((coord_acc.r() * dist_sum, coord_acc.i() * dist_sum));
                flame_coords_uniform = <$C>::from((
                    flame_uniform.r() * energy_sum_uniform,
                    flame_uniform.i() * energy_sum_uniform,
                ));
                flame_coords_weighted = <$C>::from((
                    flame_weighted.r() * energy_sum_weighted,
                    flame_weighted.i() * energy_sum_weighted,
                ));

                if has_converged {
                    let z_conj = <$C>::from((z.r(), -z.i()));
                    let orbit_conj = <$C>::from((orbit.r(), -orbit.i()));

                    let max_sqrt = <$S>::MAX.sqrt();
                    let orbit_scaled =
                        <$C>::from((orbit_conj.r() / max_sqrt, orbit_conj.i() / max_sqrt));

                    let z_mag_sq = z.r() * z.r() + z.i() * z.i();
                    slope_result = z_mag_sq.ln() * z_conj / orbit_scaled / max_sqrt;
                }
            }

            MandelbrotResult {
                period,
                escape_velocity: escape_velocity.into(),
                coords: Complex64::new(coordinates.r().into(), coordinates.i().into()),
                flame_q: Complex64::new(
                    flame_coords_uniform.r().into(),
                    flame_coords_uniform.i().into(),
                ),
                flame_c: Complex64::new(
                    flame_coords_weighted.r().into(),
                    flame_coords_weighted.i().into(),
                ),
                slope: Complex64::new(slope_result.r().into(), slope_result.i().into()),
                is_primitive: true,
            }
        }
    };
}

// The 25-cell grid. eps_newton = −(fraction_bits / 2); eps_root = −(2 × fraction_bits).
spirix_mandelbrot!(calc_f3e3, CircleF3E3, ScalarF3E3, -4, -16);
spirix_mandelbrot!(calc_f3e4, CircleF3E4, ScalarF3E4, -4, -16);
spirix_mandelbrot!(calc_f3e5, CircleF3E5, ScalarF3E5, -4, -16);
spirix_mandelbrot!(calc_f3e6, CircleF3E6, ScalarF3E6, -4, -16);
spirix_mandelbrot!(calc_f3e7, CircleF3E7, ScalarF3E7, -4, -16);
spirix_mandelbrot!(calc_f4e3, CircleF4E3, ScalarF4E3, -8, -32);
spirix_mandelbrot!(calc_f4e4, CircleF4E4, ScalarF4E4, -8, -32);
spirix_mandelbrot!(calc_f4e5, CircleF4E5, ScalarF4E5, -8, -32);
spirix_mandelbrot!(calc_f4e6, CircleF4E6, ScalarF4E6, -8, -32);
spirix_mandelbrot!(calc_f4e7, CircleF4E7, ScalarF4E7, -8, -32);
spirix_mandelbrot!(calc_f5e3, CircleF5E3, ScalarF5E3, -16, -64);
spirix_mandelbrot!(calc_f5e4, CircleF5E4, ScalarF5E4, -16, -64);
spirix_mandelbrot!(calc_f5e5, CircleF5E5, ScalarF5E5, -16, -64);
spirix_mandelbrot!(calc_f5e6, CircleF5E6, ScalarF5E6, -16, -64);
spirix_mandelbrot!(calc_f5e7, CircleF5E7, ScalarF5E7, -16, -64);
spirix_mandelbrot!(calc_f6e3, CircleF6E3, ScalarF6E3, -32, -128);
spirix_mandelbrot!(calc_f6e4, CircleF6E4, ScalarF6E4, -32, -128);
spirix_mandelbrot!(calc_f6e5, CircleF6E5, ScalarF6E5, -32, -128);
spirix_mandelbrot!(calc_f6e6, CircleF6E6, ScalarF6E6, -32, -128);
spirix_mandelbrot!(calc_f6e7, CircleF6E7, ScalarF6E7, -32, -128);
spirix_mandelbrot!(calc_f7e3, CircleF7E3, ScalarF7E3, -64, -256);
spirix_mandelbrot!(calc_f7e4, CircleF7E4, ScalarF7E4, -64, -256);
spirix_mandelbrot!(calc_f7e5, CircleF7E5, ScalarF7E5, -64, -256);
spirix_mandelbrot!(calc_f7e6, CircleF7E6, ScalarF7E6, -64, -256);
spirix_mandelbrot!(calc_f7e7, CircleF7E7, ScalarF7E7, -64, -256);

/// Grid dispatch: (fraction, exponent) in 3..=7 each → the matching instantiation.
pub fn spirix_point_fn(f: u8, e: u8) -> SpirixPointFn {
    match (f, e) {
        (3, 3) => calc_f3e3,
        (3, 4) => calc_f3e4,
        (3, 5) => calc_f3e5,
        (3, 6) => calc_f3e6,
        (3, 7) => calc_f3e7,
        (4, 3) => calc_f4e3,
        (4, 4) => calc_f4e4,
        (4, 5) => calc_f4e5,
        (4, 6) => calc_f4e6,
        (4, 7) => calc_f4e7,
        (5, 3) => calc_f5e3,
        (5, 4) => calc_f5e4,
        (5, 5) => calc_f5e5,
        (5, 6) => calc_f5e6,
        (5, 7) => calc_f5e7,
        (6, 3) => calc_f6e3,
        (6, 4) => calc_f6e4,
        (6, 5) => calc_f6e5,
        (6, 6) => calc_f6e6,
        (6, 7) => calc_f6e7,
        (7, 3) => calc_f7e3,
        (7, 4) => calc_f7e4,
        (7, 5) => calc_f7e5,
        (7, 6) => calc_f7e6,
        (7, 7) => calc_f7e7,
        _ => calc_f7e5, // out-of-grid request falls back to the historical default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use num_complex::Complex64;

    fn coord(re: f64, im: f64) -> CircleF7E3 {
        CircleF7E3::from((ScalarF7E3::from(re), ScalarF7E3::from(im)))
    }

    /// Translation fidelity: the macro-generated F7E5 cell must agree with the original
    /// hand-written `calculate_mandelbrot_point_f7e5` on escape velocity, interior detection,
    /// and coords for both an escaping and an interior point.
    #[test]
    fn macro_f7e5_matches_original() {
        for &(re, im) in &[(0.5, 0.5), (-0.1, 0.0), (0.3, 0.5), (-1.0, 0.0)] {
            let mine = calc_f7e5(coord(re, im), 1 << 12);
            let orig_point = CircleF7E5::from((ScalarF7E5::from(re), ScalarF7E5::from(im)));
            let (period, ev, coords, flame, slope) =
                crate::compute::mandelbrot::calculate_mandelbrot_point_f7e5(orig_point, 1 << 12);
            assert_eq!(mine.period, period, "period mismatch at ({}, {})", re, im);
            assert!(
                (mine.escape_velocity - ev.re).abs() < 1e-9,
                "escape velocity mismatch at ({}, {}): {} vs {}",
                re, im, mine.escape_velocity, ev.re
            );
            // Both-NaN counts as agreement — flames/slopes legitimately go undefined (℘) for
            // points that ride the escalation to the E7 overflow horizon; matching NaN patterns
            // IS translation fidelity.
            let close = |a: Complex64, b: Complex64| {
                (a - b).norm() < 1e-9 || (a.re.is_nan() && b.re.is_nan())
            };
            assert!(close(mine.coords, coords), "coords mismatch at ({}, {})", re, im);
            assert!(
                close(mine.flame_c, flame),
                "flame mismatch at ({}, {}): mine {} vs orig {}",
                re, im, mine.flame_c, flame
            );
            assert!(close(mine.slope, slope), "slope mismatch at ({}, {})", re, im);
        }
    }

    /// Escaping + interior sanity against the f64 baseline on unambiguous points.
    #[test]
    fn f7e5_agrees_with_f64_on_clear_points() {
        let spirix = calc_f7e5(coord(0.5, 0.5), 1 << 16); // clearly escapes
        let f64r = crate::compute::calculate_mandelbrot_point_f64(
            num_complex::Complex64::new(0.5, 0.5),
            (1u64 << 16) as f64,
            false,
        );
        assert_eq!(spirix.period, 0);
        // The pipelines stop at different overflow horizons and normalize differently, so
        // exact escape velocities aren't comparable — but both must be finite and positive
        // (colouring takes ln(ev)).
        assert!(
            spirix.escape_velocity.is_finite() && spirix.escape_velocity > 0.0,
            "spirix ev not usable: {}",
            spirix.escape_velocity
        );
        assert!(f64r.escape_velocity.is_finite() && f64r.escape_velocity > 0.0);

        let inside = calc_f7e5(coord(-0.1, 0.0), 1 << 16);
        assert!(inside.period != 0, "cardioid interior not detected");
    }

    /// Every grid cell must at least run without panicking on an escaping and an interior point.
    #[test]
    fn all_cells_run() {
        for f in 3..=7u8 {
            for e in 3..=7u8 {
                let func = spirix_point_fn(f, e);
                let r = func(coord(0.3, 0.5), 1 << 12);
                let _ = func(coord(-0.1, 0.0), 1 << 12);
                println!("F{}E{}: probe(0.3,0.5) ev={} period={}", f, e, r.escape_velocity, r.period);
            }
        }
    }
}
