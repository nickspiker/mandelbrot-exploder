use num_complex::{Complex, Complex64};
use spirix::*;

// For f64/f32 compatibility only
use crate::precision::{escape_velocity_constant_f64, ln2_reciprocal};

/// Calculates the Mandelbrot set properties for a given complex point
///
/// # Arguments
/// * `point` - Complex point to analyze
/// * `max_iterations` - Maximum iteration count
///
/// # Returns
/// * `period` - Orbit period (for points in the set, 0 if outside)
/// * `escape_velocity` - Rate of escape (for points outside the set)
/// * `coordinates` - Position information
/// * `flame_coords` - Additional coloring coordinates
/// * `slope` - Derivative information for coloring
/// * `multiplier` - Multiplier for periodic orbits

/// f32 precision variant - for shallow zoom levels
/// Takes native f32 input, computes in f32, switches to F5E7 on escape
/// TEMPORARILY DISABLED - needs updating for two-flame variant
#[allow(dead_code)]
fn _calculate_mandelbrot_point_f32_disabled(
    _point: Complex<f32>,
    _max_iterations: u64,
) -> (
    u64,
    f64,
    Complex64,
    Complex64,
    Complex64,
    Complex64,
    Complex64,
) {
    unimplemented!("f32 variant needs updating for two-flame system")
    /*
    let mut inside = false;
    let mut z = point;
    let mut escape_velocity = max_iterations as f64;
    let mut period = 0;
    let mut coord = Complex::<f32>::new(0., 0.);
    let mut dist_sum = 0.;
    let mut flame_uniform = Complex::<f32>::new(0., 0.);
    let mut flame_weighted = Complex::<f32>::new(0., 0.);
    let mut energy_sum_uniform = 0.;
    let mut energy_sum_weighted = 0.;
    let mut slope = Complex::<f32>::new(1., 0.);
    let mut orbit = Complex::<f32>::new(1., 0.);
    let mut z_min = f32::MAX;
    let mut multiplier = Complex::<f32>::new(1., 0.);

    const MAX_NEWTON_STEPS: isize = 32;
    let epsilon_squared = 2f32.powi(-25); // f32 has ~24 bits of precision

    let mut has_converged = false;

    'main_iteration: for iter in 1..max_iterations {
        let iter_f32 = iter as f32;

        // Calculate derivative
        slope = Complex::<f32>::new(2., 0.) * slope * z + Complex::<f32>::new(1., 0.);

        let z_mag_sq = z.norm_sqr();
        let iter_sq = iter_f32 * iter_f32;

        // Check for escape to F5E7 precision
        if z_mag_sq > 2.0_f32.powi(32) {
            // TODO: Convert to F5E7 and continue iteration
            // For now, just calculate escape velocity in f32
            let log_log_mag = z_mag_sq.ln().ln();
            escape_velocity = iter as f64
                - ((log_log_mag - escape_velocity_constant_f64() as f32) * ln2_reciprocal() as f32)
                    as f64;
            has_converged = true;
            break 'main_iteration;
        }

        if z_mag_sq > 4.0 {
            // Point is escaping
            let log_log_mag = z_mag_sq.ln().ln();
            let two_ln = 2.0_f32.ln();
            let two_times_two_ln = 2.0 * two_ln;
            let ln_512 = 512.0_f32.ln();

            let mut taper = (log_log_mag - two_times_two_ln.ln()) / ln_512;
            taper = 1.0 - taper * taper;
            taper = taper * taper;
            taper = taper * taper * taper * iter_sq;

            let z_scale = taper / z_mag_sq.sqrt();
            energy_sum += taper;
            flame += z * z_scale;

            let z_weight = iter_sq / z_mag_sq;
            dist_sum += z_weight;
            coord += z * z_weight;

            orbit = Complex::<f32>::new(2.0, 0.0) * z * orbit + Complex::<f32>::new(1.0, 0.0);
            z = z * z + point;
        } else {
            // Point might be in set
            let taper = iter_sq;
            let z_scale = taper / z_mag_sq.sqrt();
            energy_sum += taper;
            flame += z * z_scale;

            let z_weight = iter_sq / z_mag_sq;
            dist_sum += z_weight;
            coord += z * z_weight;

            if z_min > z_mag_sq {
                let magnitude_ratio = z_mag_sq / z_min;
                z_min = z_mag_sq;

                if magnitude_ratio < 0.25 {
                    let mut w = z;

                    for _ in 0..MAX_NEWTON_STEPS {
                        let mut u = w;
                        let mut du = Complex::<f32>::new(1.0, 0.0);
                        multiplier = Complex::<f32>::new(1.0, 0.0);

                        for _ in 0..iter {
                            multiplier = Complex::<f32>::new(2.0, 0.0) * u * multiplier;
                            du = Complex::<f32>::new(2.0, 0.0) * u * du;
                            u = u * u + point;
                        }

                        let newton = (u - w) / (du - Complex::<f32>::new(1.0, 0.0));
                        w = w - newton;

                        if newton.norm_sqr() < epsilon_squared {
                            if (du - Complex::<f32>::new(1.0, 0.0)).norm_sqr() < 1.0 {
                                coord = du;
                                flame = w;
                                inside = true;
                                period = iter;
                                has_converged = true;
                                break 'main_iteration;
                            }
                        }
                    }
                }
            }

            orbit = Complex::<f32>::new(2.0, 0.0) * z * orbit + Complex::<f32>::new(1.0, 0.0);
            z = z * z + point;
        }
    }

    let slope_result;
    let coordinates;
    let flame_coords_uniform;
    let flame_coords_weighted;

    if inside {
        coordinates = coord;
        flame_coords_uniform = flame_uniform;
        flame_coords_weighted = flame_weighted;

        let initial_point = point;
        let mut refined_point = initial_point;

        for _ in 0..MAX_NEWTON_STEPS {
            let mut z = refined_point;
            let mut d = Complex::<f32>::new(1.0, 0.0);

            for _ in 1..period {
                d = Complex::<f32>::new(2.0, 0.0) * d * z + Complex::<f32>::new(1.0, 0.0);
                z = z * z + refined_point;
            }

            let epsilon = Complex::<f32>::new(0.0, 2.0_f32.powi(-25));
            d = refined_point - z / d + epsilon;

            if d == refined_point {
                refined_point = d;
                break;
            }
            refined_point = d;
        }

        let diff = initial_point - refined_point;
        slope_result = Complex::<f32>::new(diff.re, -diff.im);
    } else {
        dist_sum = 1.0 / dist_sum;
        energy_sum_uniform = 1.0 / energy_sum_uniform;
        energy_sum_weighted = 1.0 / energy_sum_weighted;

        coordinates = coord * dist_sum;
        flame_coords_uniform = flame_uniform * energy_sum_uniform;
        flame_coords_weighted = flame_weighted * energy_sum_weighted;

        if has_converged {
            let z_mag_sq = z.norm_sqr();
            let z_conj = z.conj();
            let orbit_conj = orbit.conj();
            slope_result = z_mag_sq.ln() * z_conj / orbit_conj;
        } else {
            slope_result = Complex::<f32>::new(0.0, 0.0);
        }
    }

    // Convert to Complex64 for return
    (
        period,
        escape_velocity,
        Complex64::new(coordinates.re as f64, coordinates.im as f64),
        Complex64::new(flame_coords.re as f64, flame_coords.im as f64),
        Complex64::new(slope_result.re as f64, slope_result.im as f64),
        Complex64::new(multiplier.re as f64, multiplier.im as f64),
    )
    */
}

/// F7E5 precision variant - for deep zoom levels
/// Takes native F7E5 input, computes in F7E5, switches to F5E7 on escape
///
/// Returns: (period, escape_velocity, coord, flame_uniform, flame_weighted, slope, multiplier, z)
/// - flame_uniform: fade-only weighting (measures overflow proximity)
/// - flame_weighted: fade × iter² weighting (measures time-weighted orbit behavior)
/// - z is the final iteration value, needed for distance estimation in boundary tracing
pub fn calculate_mandelbrot_point_f7e5(
    point: CircleF7E5,
    max_iterations: u64,
) -> (
    u64,       // period
    Complex64, // escape_velocity
    Complex64, // coords
    Complex64, // flame
    Complex64, // slope
) {
    let mut inside = false;
    let mut z = point;
    let mut escape_velocity = ScalarF7E5::from(max_iterations);
    let mut period = 0;
    let mut coord = CircleF7E5::ZERO;
    let mut dist_sum = ScalarF7E5::ZERO;
    let mut flame_uniform = CircleF7E5::ZERO;
    let mut flame_weighted = CircleF7E5::ZERO;
    let mut energy_sum_uniform = ScalarF7E5::ZERO;
    let mut energy_sum_weighted = ScalarF7E5::ZERO;
    let mut slope = CircleF7E5::ONE;
    let mut orbit = CircleF7E5::ONE;
    let mut z_min = ScalarF7E5::MAX;
    let mut multiplier = CircleF7E5::ONE;
    let mut slope_result = CircleF7E5::ZERO;

    const MAX_NEWTON_STEPS: isize = 32;
    // sqrt(machine epsilon) for optimal Newton convergence with 128-bit fraction
    let epsilon_squared = ScalarF7E5::TWO.pow(-64);

    let mut has_converged = false;

    'main_iteration: for iter in 1..max_iterations {
        let iter_scalar = ScalarF7E5::from(iter);

        // Calculate derivative
        let slope_temp = (slope.r() * z.r() - slope.i() * z.i()) << 1u8;
        let slope_im = (slope.r() * z.i() + slope.i() * z.r()) << 1u8;
        let slope_re = slope_temp + ScalarF7E5::ONE;
        slope = CircleF7E5::from((slope_re, slope_im));

        let mut z_re_sq = z.r() * z.r();
        let mut z_im_sq = z.i() * z.i();
        let mut z_mag_sq = z_re_sq + z_im_sq;
        let mut iter_sq = iter_scalar.square();

        if z_mag_sq > ScalarF7E5::from(4) {
            // Point is escaping - iterate in F7E5 until overflow
            let mut iter_mut = iter;
            let mut iter_scalar_mut = iter_scalar;
            loop {
                // Calculate next z: z_{n+1} = z_n^2 + c
                let z_next_re = z_re_sq - z_im_sq + point.r();
                let z_next_im = ((z.r() * z.i()) << 1u8) + point.i();
                let z_next = CircleF7E5::from((z_next_re, z_next_im));
                let z_next_mag_sq = z_next.magnitude_squared();

                // Check if next z would overflow F7E5 (would fail .is_normal())
                if !z_next_mag_sq.is_normal() {
                    // Current z is still valid - break and convert to F5E7
                    break;
                }

                // Next z is good - commit update and accumulate
                z = z_next;
                z_re_sq = z.r() * z.r();
                z_im_sq = z.i() * z.i();
                z_mag_sq = z_re_sq + z_im_sq;
                iter_mut += 1;
                iter_scalar_mut = ScalarF7E5::from(iter_mut);
                iter_sq = iter_scalar_mut.square();

                // Accumulate weighted coloring data - BOTH VARIANTS
                // Variant 1: Uniform weighting (no iter² factor, just 1.0)
                let taper_uniform = ScalarF7E5::ONE;
                let z_scale_uniform = taper_uniform / z_mag_sq.sqrt();
                energy_sum_uniform += taper_uniform;
                flame_uniform = flame_uniform
                    + CircleF7E5::from((z.r() * z_scale_uniform, z.i() * z_scale_uniform));

                // Variant 2: Iteration-weighted (iter²)
                let taper_weighted = iter_sq;
                let z_scale_weighted = taper_weighted / z_mag_sq.sqrt();
                energy_sum_weighted += taper_weighted;
                flame_weighted = flame_weighted
                    + CircleF7E5::from((z.r() * z_scale_weighted, z.i() * z_scale_weighted));

                let z_weight = iter_sq / z_mag_sq;
                dist_sum += z_weight;
                coord = coord + CircleF7E5::from((z.r() * z_weight, z.i() * z_weight));

                // Update derivatives
                let orbit_temp = ((z.r() * orbit.r() - z.i() * orbit.i()) << 1u8) + ScalarF7E5::ONE;
                let orbit_im = (orbit.r() * z.i() + orbit.i() * z.r()) << 1u8;
                orbit = CircleF7E5::from((orbit_temp, orbit_im));

                let slope_temp = (slope.r() * z.r() - slope.i() * z.i()) << 1u8;
                let slope_im = (slope.r() * z.i() + slope.i() * z.r()) << 1u8;
                let slope_re = slope_temp + ScalarF7E5::ONE;
                slope = CircleF7E5::from((slope_re, slope_im));
            }

            // F7E5 overflowed - convert to F5E7 and continue
            let mut z_f5e7 = CircleF5E7::from((ScalarF5E7::from(z.r()), ScalarF5E7::from(z.i())));
            let point_f5e7 =
                CircleF5E7::from((ScalarF5E7::from(point.r()), ScalarF5E7::from(point.i())));
            let mut orbit_f5e7 =
                CircleF5E7::from((ScalarF5E7::from(orbit.r()), ScalarF5E7::from(orbit.i())));
            let mut coord_f5e7 =
                CircleF5E7::from((ScalarF5E7::from(coord.r()), ScalarF5E7::from(coord.i())));
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
            // Carry the working-type escape loop's count — resetting to `iter` here made the
            // final escape velocity negative (iteration count out of sync with lb(lb(|z|²))).
            let mut iter_mut = iter_mut;

            // Continue in F5E7 until final overflow
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

                // Variant 1: Uniform weighting (fade-only)
                let taper_uniform_f5e7 = fade_f5e7;
                let z_scale_uniform_f5e7 = taper_uniform_f5e7 / z_mag_sq_f5e7.sqrt();
                energy_sum_uniform_f5e7 = energy_sum_uniform_f5e7 + taper_uniform_f5e7;
                flame_uniform_f5e7 = flame_uniform_f5e7 + z_f5e7 * z_scale_uniform_f5e7;

                // Variant 2: Iteration-weighted (fade × iter²)
                let taper_weighted_f5e7 = fade_f5e7 * iter_sq_f5e7;
                let z_scale_weighted_f5e7 = taper_weighted_f5e7 / z_mag_sq_f5e7.sqrt();
                energy_sum_weighted_f5e7 = energy_sum_weighted_f5e7 + taper_weighted_f5e7;
                flame_weighted_f5e7 = flame_weighted_f5e7 + z_f5e7 * z_scale_weighted_f5e7;

                let z_weight_f5e7 = iter_sq_f5e7 / z_mag_sq_f5e7;
                dist_sum_f5e7 = dist_sum_f5e7 + z_weight_f5e7;
                coord_f5e7 = coord_f5e7 + z_f5e7 * z_weight_f5e7;

                let orbit_temp = z_f5e7.r() * orbit_f5e7.r() - z_f5e7.i() * orbit_f5e7.i();
                let orbit_im_val = orbit_f5e7.r() * z_f5e7.i() + orbit_f5e7.i() * z_f5e7.r();
                orbit_f5e7 = CircleF5E7::from((
                    orbit_temp + orbit_temp + ScalarF5E7::ONE,
                    orbit_im_val + orbit_im_val,
                ));
            }

            // Calculate final escape velocity (saturation guard, same as above)
            let z_mag_sq_f5e7 = {
                let m = z_f5e7.magnitude_squared();
                if m.is_normal() { m } else { ScalarF5E7::MAX }
            };
            let escape_f5e7 =
                ScalarF5E7::from(iter_mut) + ScalarF5E7::ONE - z_mag_sq_f5e7.lb().lb();
            escape_velocity = ScalarF7E5::from(escape_f5e7);

            // Normalize and convert back to F7E5
            coord = CircleF7E5::from((
                ScalarF7E5::from(coord_f5e7.r() / dist_sum_f5e7),
                ScalarF7E5::from(coord_f5e7.i() / dist_sum_f5e7),
            ));
            flame_uniform = CircleF7E5::from((
                ScalarF7E5::from(flame_uniform_f5e7.r() / energy_sum_uniform_f5e7),
                ScalarF7E5::from(flame_uniform_f5e7.i() / energy_sum_uniform_f5e7),
            ));
            flame_weighted = CircleF7E5::from((
                ScalarF7E5::from(flame_weighted_f5e7.r() / energy_sum_weighted_f5e7),
                ScalarF7E5::from(flame_weighted_f5e7.i() / energy_sum_weighted_f5e7),
            ));
            slope_result = CircleF7E5::from((
                ScalarF7E5::from(orbit_f5e7.r()),
                ScalarF7E5::from(orbit_f5e7.i()),
            ));

            has_converged = true;
            break 'main_iteration;
        } else {
            // Point might be in set - compute BOTH flame variants
            // Variant 1: Uniform weighting
            let taper_uniform = ScalarF7E5::ONE;
            let z_scale_uniform = taper_uniform / z_mag_sq.sqrt();
            energy_sum_uniform = energy_sum_uniform + taper_uniform;
            flame_uniform = flame_uniform
                + CircleF7E5::from((z.r() * z_scale_uniform, z.i() * z_scale_uniform));

            // Variant 2: Iteration-weighted
            let taper_weighted = iter_sq;
            let z_scale_weighted = taper_weighted / z_mag_sq.sqrt();
            energy_sum_weighted = energy_sum_weighted + taper_weighted;
            flame_weighted = flame_weighted
                + CircleF7E5::from((z.r() * z_scale_weighted, z.i() * z_scale_weighted));

            let z_weight = iter_sq / z_mag_sq;
            dist_sum += z_weight;
            coord = coord + CircleF7E5::from((z.r() * z_weight, z.i() * z_weight));

            if z_min > z_mag_sq {
                let magnitude_ratio = z_mag_sq / z_min;
                z_min = z_mag_sq;

                if magnitude_ratio < ScalarF7E5::from(0.25) {
                    let mut w = z;

                    for _ in 0..MAX_NEWTON_STEPS {
                        let mut u = w;
                        let mut du = CircleF7E5::from((ScalarF7E5::ONE, ScalarF7E5::ZERO));
                        multiplier = CircleF7E5::from((ScalarF7E5::ONE, ScalarF7E5::ZERO));

                        for _ in 0..iter {
                            let temp_re = multiplier.r();
                            let multiplier_re = (temp_re * u.r() - multiplier.i() * u.i()) << 1u8;
                            let multiplier_im = (temp_re * u.i() + multiplier.i() * u.r()) << 1u8;
                            multiplier = CircleF7E5::from((multiplier_re, multiplier_im));

                            let du_temp = (du.r() * u.r() - du.i() * u.i()) << 1u8;
                            let du_im = (du.r() * u.i() + du.i() * u.r()) << 1u8;
                            let du_re = du_temp;
                            du = CircleF7E5::from((du_re, du_im));

                            // Update u
                            let u_re = u.r() * u.r() - u.i() * u.i() + point.r();
                            let u_im = ((u.r() * u.i()) << 1u8) + point.i();
                            u = CircleF7E5::from((u_re, u_im));
                        }

                        let newton = u - w;
                        let du_offset = du.r() - ScalarF7E5::ONE;
                        let du_denom = du_offset * du_offset + du.i() * du.i();

                        let newton_re = (newton.r() * du_offset + newton.i() * du.i()) / du_denom;
                        let newton_im = (newton.i() * du_offset - newton.r() * du.i()) / du_denom;
                        let newton = CircleF7E5::from((newton_re, newton_im));

                        iter_sq = newton.r() * newton.r() + newton.i() * newton.i();
                        w = w - newton;

                        if iter_sq < epsilon_squared {
                            let du_mag_sq = du.r() * du.r() + du.i() * du.i();
                            if du_mag_sq < ScalarF7E5::ONE {
                                coord = du;
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

            let orbit_temp = ((z.r() * orbit.r() - z.i() * orbit.i()) << 1u8) + ScalarF7E5::ONE;
            let orbit_im = (orbit.r() * z.i() + orbit.i() * z.r()) << 1u8;
            orbit = CircleF7E5::from((orbit_temp, orbit_im));

            // Update z
            let z_re = z.r() * z.r() - z.i() * z.i() + point.r();
            let z_im = ((z.r() * z.i()) << 1u8) + point.i();
            z = CircleF7E5::from((z_re, z_im));
        }
    }

    let coordinates;
    let flame_coords_uniform;
    let flame_coords_weighted;

    if inside {
        coordinates = coord;
        flame_coords_uniform = flame_uniform;
        flame_coords_weighted = flame_weighted;

        let initial_point = point;
        let mut refined_point = initial_point;

        for _ in 0..MAX_NEWTON_STEPS {
            let mut z = refined_point;
            let mut d = CircleF7E5::ONE;

            for _ in 1..period {
                d = ((d * z) << 1u8) + CircleF7E5::ONE;
                z = z * z + refined_point;
            }

            let epsilon = CircleF7E5::from((ScalarF7E5::ZERO, ScalarF7E5::TWO.pow(-256)));
            d = refined_point - z / d + epsilon;

            if d == refined_point {
                refined_point = d;
                break;
            }
            refined_point = d;
        }

        let diff = initial_point - refined_point;
        slope_result = CircleF7E5::from((diff.r(), -diff.i()));
    } else {
        dist_sum = ScalarF7E5::ONE / dist_sum;
        energy_sum_uniform = ScalarF7E5::ONE / energy_sum_uniform;
        energy_sum_weighted = ScalarF7E5::ONE / energy_sum_weighted;

        coordinates = CircleF7E5::from((coord.r() * dist_sum, coord.i() * dist_sum));
        flame_coords_uniform = CircleF7E5::from((
            flame_uniform.r() * energy_sum_uniform,
            flame_uniform.i() * energy_sum_uniform,
        ));
        flame_coords_weighted = CircleF7E5::from((
            flame_weighted.r() * energy_sum_weighted,
            flame_weighted.i() * energy_sum_weighted,
        ));

        if has_converged {
            let z_conj = CircleF7E5::from((z.r(), -z.i()));
            let orbit_conj = CircleF7E5::from((orbit.r(), -orbit.i()));

            let max_sqrt = ScalarF7E5::MAX.sqrt();
            let orbit_scaled =
                CircleF7E5::from((orbit_conj.r() / max_sqrt, orbit_conj.i() / max_sqrt));

            let z_mag_sq = z.r() * z.r() + z.i() * z.i();
            slope_result = z_mag_sq.ln() * z_conj / orbit_scaled / max_sqrt;
        }
    }

    // Convert to f64
    (
        period,
        Complex64::new(escape_velocity.into(), 0.0),
        Complex64::new(coordinates.r().into(), coordinates.i().into()),
        Complex64::new(
            flame_coords_weighted.r().into(),
            flame_coords_weighted.i().into(),
        ),
        Complex64::new(slope_result.r().into(), slope_result.i().into()),
    )
}
