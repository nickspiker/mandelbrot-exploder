//! IEEE-754 binary32 baseline pipeline. Same algorithm as the spirix grid instantiations
//! (`spirix_grid.rs`): escape detection + dual flame accumulation + Newton interior/period
//! detection + interior root refinement, with all hot math in native f32. Where a spirix cell
//! escalates to F5E7 when the working type's range runs out, this pipeline escalates f32 → f64
//! — the same experiment expressed in IEEE types. Signature-compatible with
//! `calculate_mandelbrot_point_f64` so the calculation thread takes either via one fn pointer.

use super::mandelbrot_f64::MandelbrotResult;
use num_complex::{Complex, Complex64};

type Complex32 = Complex<f32>;

/// Newton convergence threshold — the grid's `TWO.pow(-frac_bits/2)` (√machine-epsilon) for
/// f32's 24-bit significand: 2⁻¹².
const EPSILON_SQUARED: f32 = 2.441_406_25e-4; // 2^-12
/// Tiny asymmetric nudge for interior root refinement — analogue of the grid's `TWO.pow(-2×frac_bits)`.
const EPS_ROOT: f32 = 3.552_713_7e-15; // 2^-48

pub fn calculate_mandelbrot_point_f32(c: Complex64, frac_iter: f64, _log: bool) -> MandelbrotResult {
    let max_iterations = frac_iter as u64;
    let point = Complex32::new(c.re as f32, c.im as f32);

    let mut inside = false;
    let mut z = point;
    let mut escape_velocity = max_iterations as f64;
    let mut period = 0u64;
    let mut coord_acc = Complex32::new(0., 0.);
    let mut dist_sum = 0f32;
    let mut flame_uniform = Complex32::new(0., 0.);
    let mut flame_weighted = Complex32::new(0., 0.);
    let mut energy_sum_uniform = 0f32;
    let mut energy_sum_weighted = 0f32;
    let mut slope = Complex32::new(1., 0.);
    let mut orbit = Complex32::new(1., 0.);
    let mut z_min = f32::MAX;
    let mut slope_result = Complex32::new(0., 0.);

    const MAX_NEWTON_STEPS: isize = 32;

    let mut has_converged = false;

    'main_iteration: for iter in 1..max_iterations {
        let iter_f = iter as f32;

        // Derivative
        slope = Complex32::new(
            (slope.re * z.re - slope.im * z.im) * 2. + 1.,
            (slope.re * z.im + slope.im * z.re) * 2.,
        );

        let mut z_mag_sq = z.norm_sqr();
        let mut iter_sq = iter_f * iter_f;

        if z_mag_sq > 4.0 {
            // Escaping — iterate in f32 until the magnitude leaves f32's normal range
            let mut iter_mut = iter;
            loop {
                let z_next = z * z + point;
                let z_next_mag_sq = z_next.norm_sqr();
                if !z_next_mag_sq.is_normal() {
                    break;
                }
                z = z_next;
                z_mag_sq = z.norm_sqr();
                iter_mut += 1;
                let it = iter_mut as f32;
                iter_sq = it * it;

                let z_scale_uniform = 1.0 / z_mag_sq.sqrt();
                energy_sum_uniform += 1.0;
                flame_uniform += z * z_scale_uniform;

                let z_scale_weighted = iter_sq / z_mag_sq.sqrt();
                energy_sum_weighted += iter_sq;
                flame_weighted += z * z_scale_weighted;

                let z_weight = iter_sq / z_mag_sq;
                dist_sum += z_weight;
                coord_acc += z * z_weight;

                orbit = Complex32::new(
                    (z.re * orbit.re - z.im * orbit.im) * 2. + 1.,
                    (orbit.re * z.im + orbit.im * z.re) * 2.,
                );
                slope = Complex32::new(
                    (slope.re * z.re - slope.im * z.im) * 2. + 1.,
                    (slope.re * z.im + slope.im * z.re) * 2.,
                );
            }

            // f32 range exhausted — escalate to f64 and continue to final overflow
            let mut z64 = Complex64::new(z.re as f64, z.im as f64);
            let point64 = Complex64::new(point.re as f64, point.im as f64);
            let mut orbit64 = Complex64::new(orbit.re as f64, orbit.im as f64);
            let mut coord64 = Complex64::new(coord_acc.re as f64, coord_acc.im as f64);
            let mut dist_sum64 = dist_sum as f64;
            let mut flame_uniform64 =
                Complex64::new(flame_uniform.re as f64, flame_uniform.im as f64);
            let mut flame_weighted64 =
                Complex64::new(flame_weighted.re as f64, flame_weighted.im as f64);
            let mut energy_sum_uniform64 = energy_sum_uniform as f64;
            let mut energy_sum_weighted64 = energy_sum_weighted as f64;
            let mut iter_mut = iter_mut;

            let max_lb = f64::MAX.log2(); // 1024
            loop {
                let z_next = z64 * z64 + point64;
                if !z_next.norm_sqr().is_normal() {
                    break;
                }
                z64 = z_next;
                let z_mag_sq64 = z64.norm_sqr();
                iter_mut += 1;
                let iter_sq64 = (iter_mut * iter_mut) as f64;

                let log_log_mag = z_mag_sq64.log2().log2();
                let fade = (max_lb - log_log_mag) / (max_lb - 1.0);

                let z_scale_uniform = fade / z_mag_sq64.sqrt();
                energy_sum_uniform64 += fade;
                flame_uniform64 += z64 * z_scale_uniform;

                let taper_weighted = fade * iter_sq64;
                let z_scale_weighted = taper_weighted / z_mag_sq64.sqrt();
                energy_sum_weighted64 += taper_weighted;
                flame_weighted64 += z64 * z_scale_weighted;

                let z_weight = iter_sq64 / z_mag_sq64;
                dist_sum64 += z_weight;
                coord64 += z64 * z_weight;

                let orbit_temp = z64.re * orbit64.re - z64.im * orbit64.im;
                let orbit_im = orbit64.re * z64.im + orbit64.im * z64.re;
                orbit64 = Complex64::new(orbit_temp * 2. + 1., orbit_im * 2.);
            }

            let z_mag_sq64 = z64.norm_sqr();
            escape_velocity = iter_mut as f64 + 1.0 - z_mag_sq64.log2().log2();

            // Normalize and convert back to f32 — mirrors the spirix cells' "convert back to
            // the working type" step, so the post-loop `has_converged` tail (slope
            // normalization against √MAX) applies identically. Returning early here with the
            // raw f64 orbit left slope at ±inf and blacked out every escaping pixel.
            let c32 = |v: Complex64| Complex32::new(v.re as f32, v.im as f32);
            coord_acc = c32(coord64 / dist_sum64);
            flame_uniform = c32(flame_uniform64 / energy_sum_uniform64);
            flame_weighted = c32(flame_weighted64 / energy_sum_weighted64);
            slope_result = c32(orbit64); // placeholder — recomputed in the has_converged tail

            has_converged = true;
            break 'main_iteration;
        } else {
            // Might be inside — accumulate both flame variants
            let z_mag = z_mag_sq.sqrt();
            energy_sum_uniform += 1.0;
            flame_uniform += z / z_mag;

            energy_sum_weighted += iter_sq;
            flame_weighted += z * (iter_sq / z_mag);

            let z_weight = iter_sq / z_mag_sq;
            dist_sum += z_weight;
            coord_acc += z * z_weight;

            if z_min > z_mag_sq {
                let magnitude_ratio = z_mag_sq / z_min;
                z_min = z_mag_sq;

                if magnitude_ratio < 0.25 {
                    let mut w = z;

                    for _ in 0..MAX_NEWTON_STEPS {
                        let mut u = w;
                        let mut du = Complex32::new(1., 0.);

                        for _ in 0..iter {
                            du = Complex32::new(
                                (du.re * u.re - du.im * u.im) * 2.,
                                (du.re * u.im + du.im * u.re) * 2.,
                            );
                            u = u * u + point;
                        }

                        let newton = u - w;
                        let du_offset = du.re - 1.0;
                        let du_denom = du_offset * du_offset + du.im * du.im;

                        let newton = Complex32::new(
                            (newton.re * du_offset + newton.im * du.im) / du_denom,
                            (newton.im * du_offset - newton.re * du.im) / du_denom,
                        );

                        let newton_mag_sq = newton.norm_sqr();
                        w -= newton;

                        if newton_mag_sq < EPSILON_SQUARED {
                            let du_mag_sq = du.norm_sqr();
                            if du_mag_sq < 1.0 {
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

            orbit = Complex32::new(
                (z.re * orbit.re - z.im * orbit.im) * 2. + 1.,
                (orbit.re * z.im + orbit.im * z.re) * 2.,
            );

            z = z * z + point;
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
            let mut d = Complex32::new(1., 0.);

            for _ in 1..period {
                d = d * z * 2. + Complex32::new(1., 0.);
                z = z * z + refined_point;
            }

            let next = refined_point - z / d + Complex32::new(0., EPS_ROOT);
            if next == refined_point {
                refined_point = next;
                break;
            }
            refined_point = next;
        }

        let diff = initial_point - refined_point;
        slope_result = Complex32::new(diff.re, -diff.im);
    } else {
        coordinates = coord_acc / dist_sum;
        flame_coords_uniform = flame_uniform / energy_sum_uniform;
        flame_coords_weighted = flame_weighted / energy_sum_weighted;

        if has_converged {
            // Computed in f64: num_complex's naive complex division (a·b̄/|b|²) overflows its
            // intermediate product when both operands sit near f32's range edge — which z and
            // orbit always do here (the escape loop ran them to the brink of overflow). spirix
            // never hits this because its division is exponent-arithmetic with no intermediate
            // blowup. This is final normalization, not working-type iteration, so f64 is fair.
            let z64 = Complex64::new(z.re as f64, z.im as f64);
            let orbit64 = Complex64::new(orbit.re as f64, orbit.im as f64);
            let slope64 = if orbit64.re.is_finite() && orbit64.im.is_finite() {
                let z_conj = z64.conj();
                let orbit_conj = orbit64.conj();
                let max_sqrt = (f32::MAX as f64).sqrt();
                let orbit_scaled = orbit_conj / max_sqrt;
                z64.norm_sqr().ln() * z_conj / orbit_scaled / max_sqrt
            } else {
                // The f32 orbit overflowed mid-escape (it out-grows z ~2× in exponent). The
                // spirix analogue is an exploded denominator whose quotient is vanished —
                // "smaller than representable", NOT exact zero (exact zero NaNs the colouring
                // via log2(0) and slope/|slope|). The faithful IEEE stand-in is the smallest
                // f32 normal, which also survives the f32 round-trip below.
                Complex64::new(f32::MIN_POSITIVE as f64, 0.)
            };
            slope_result = Complex32::new(slope64.re as f32, slope64.im as f32);
        }
    }

    MandelbrotResult {
        period,
        escape_velocity,
        coords: Complex64::new(coordinates.re as f64, coordinates.im as f64),
        flame_q: Complex64::new(flame_coords_uniform.re as f64, flame_coords_uniform.im as f64),
        flame_c: Complex64::new(
            flame_coords_weighted.re as f64,
            flame_coords_weighted.im as f64,
        ),
        slope: Complex64::new(slope_result.re as f64, slope_result.im as f64),
        is_primitive: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression: the escalated escape path used to return a raw ±inf orbit as `slope`,
    /// which NaN'd the colouring and blacked out the whole outer region in f32 mode.
    #[test]
    fn escaping_points_get_usable_results() {
        for &(re, im) in &[(2.0, 0.0), (0.5, 0.5), (-2.5, 0.1), (0.0, 1.5), (0.4, 0.4)] {
            let r = calculate_mandelbrot_point_f32(
                Complex64::new(re, im),
                (1u64 << 12) as f64,
                false,
            );
            assert_eq!(r.period, 0, "({}, {}) misdetected as interior", re, im);
            assert!(
                r.escape_velocity.is_finite() && r.escape_velocity > 0.0,
                "unusable escape velocity at ({}, {}): {}",
                re, im, r.escape_velocity
            );
            assert!(
                r.slope.re.is_finite() && r.slope.im.is_finite(),
                "slope not finite at ({}, {}): {}",
                re, im, r.slope
            );
            let colour = crate::compute::calculate_colour(
                Complex64::new(re, im),
                &r,
                Complex64::new(1., 0.),
            );
            assert!(
                colour[0] != 0 || colour[1] != 0 || colour[2] != 0,
                "black colour at ({}, {}) — outer region would not render (result: {:?})",
                re, im, r
            );
        }
    }
}
