//! Mandelbrot set computation with multi-stage escape
//!
//! # Architecture
//!
//! 0. **Classification (f64)**: Iterate until escaping (|z|² > 4) or proven inside
//! 1. **f64 Escape**: Continue until any value goes non-normal → hand off to F6E7
//! 2. **F6E7 Escape**: Continue until transfinite → finalize and return
//!
//! # Loop Structure
//!
//! Each escape loop follows the same pattern:
//! 0. Compute next values
//! 1. Check normality - if abnormal, break (previous values still valid)
//! 2. Commit the new values
//! 3. Accumulate into weighted sums
//!
//! This ensures we always have valid state when we exit.

use num_complex::Complex64;
use spirix::{CircleF6E7, ScalarF6E7};

/// Orbit state tracker for f64 precision iteration
#[derive(Clone, Copy, Debug)]
struct OrbitTrackerf64 {
    z: Complex64,
    z_mag_sq: f64,
    iter: f64,
    coord_sum: Complex64,
    coord_weight: f64,
    flame_q_sum: Complex64,
    flame_q_weight: f64,
    flame_c_sum: Complex64,
    flame_c_weight: f64,
}

impl OrbitTrackerf64 {
    fn new(z: Complex64, iter: f64) -> Self {
        Self {
            z,
            z_mag_sq: z.norm_sqr(),
            iter,
            coord_sum: Complex64::new(0., 0.),
            coord_weight: 0.,
            flame_q_sum: Complex64::new(0., 0.),
            flame_q_weight: 0.,
            flame_c_sum: Complex64::new(0., 0.),
            flame_c_weight: 0.,
        }
    }

    /// Accumulate for small magnitudes (|z|² ≤ 4) - skip flame accumulation for z=0
    fn iterate_small(&mut self) {
        let iter_sq = self.iter * self.iter;

        // Coord weighting: iter² / |z|²
        let zsi = iter_sq / self.z_mag_sq;
        self.coord_weight += zsi;
        self.coord_sum += self.z * zsi;

        // Quadratic flame weighting: iter² / sqrt(|z|²)
        let zsq = iter_sq / self.z_mag_sq.sqrt();
        self.flame_q_weight += iter_sq;
        self.flame_q_sum += self.z * zsq;

        self.flame_c_weight += 1. / self.z_mag_sq.sqrt();
        self.flame_c_sum += 1. / self.z;
    }

    /// Accumulate for large magnitudes (|z|² > 4) - complex taper for smooth escape
    fn iterate_big(&mut self) {
        let iter_sq = self.iter * self.iter;

        // Coord weighting: iter² / |z|²
        let zsi = iter_sq / self.z_mag_sq;
        self.coord_weight += zsi;
        self.coord_sum += self.z * zsi;

        // Flame taper as points escape
        let llzs = self.z_mag_sq.ln().ln();
        let mut taper = ((llzs - (2.0 * 2f64.ln()).ln()) / f64::MAX.sqrt().ln().ln()).max(0.);
        taper = 1. - taper * taper;
        taper = taper * taper;
        taper = taper * taper * taper;
        let taper_sq = taper * iter_sq;

        // Quadratic flame weighting: taper * iter² / sqrt(|z|²)
        let zsq = taper_sq / self.z_mag_sq.sqrt();
        self.flame_q_weight += taper_sq;
        self.flame_q_sum += self.z * zsq;

        // Quad with inversion
        self.flame_c_weight += 1. / self.z_mag_sq.sqrt();
        self.flame_c_sum += 1. / self.z;
    }
}

/// 128 bit exponent extended for escaping points
#[derive(Clone, Copy)]
struct AccumulatorsF6E7 {
    coord_sum: CircleF6E7,
    coord_weight: ScalarF6E7,
    flame_q_sum: CircleF6E7,
    flame_q_weight: ScalarF6E7,
    flame_c_sum: CircleF6E7,
    flame_c_weight: ScalarF6E7,
}

impl From<OrbitTrackerf64> for AccumulatorsF6E7 {
    fn from(tracker: OrbitTrackerf64) -> Self {
        Self {
            coord_sum: CircleF6E7::from(tracker.coord_sum),
            coord_weight: ScalarF6E7::from(tracker.coord_weight),
            flame_q_sum: CircleF6E7::from(tracker.flame_q_sum),
            flame_q_weight: ScalarF6E7::from(tracker.flame_q_weight),
            flame_c_sum: CircleF6E7::from(tracker.flame_c_sum),
            flame_c_weight: ScalarF6E7::from(tracker.flame_c_weight),
        }
    }
}

impl AccumulatorsF6E7 {
    fn new() -> Self {
        Self {
            coord_sum: CircleF6E7::ZERO,
            coord_weight: ScalarF6E7::ZERO,
            flame_q_sum: CircleF6E7::ZERO,
            flame_q_weight: ScalarF6E7::ZERO,
            flame_c_sum: CircleF6E7::ZERO,
            flame_c_weight: ScalarF6E7::ZERO,
        }
    }
}

/// Escape metrics
struct EscapeState {
    iter: ScalarF6E7,
    z: CircleF6E7,
    z_mag_sq: ScalarF6E7,
    slope_product: CircleF6E7,
    accumulators: AccumulatorsF6E7,
}

/// Result of interior point detection
struct InteriorResult {
    period: u64,
    attractor: Complex64,  // The periodic attractor point
    multiplier: Complex64, // Derivative at attractor (|multiplier| < 1)
    is_primitive: bool, // true = cardioid-like (mini-Mandelbrot), false = circle-like (satellite bulb)
}

/// Classification result
enum Classification {
    Inside(InteriorResult),
    Escaped(EscapeState),
    MaxIterReached(OrbitTrackerf64), // Hit max_iter without escaping or proving inside
}

/// Final output
#[derive(Debug, Clone, Copy)]
pub struct MandelbrotResult {
    pub period: u64,
    pub escape_velocity: f64,
    pub coords: Complex64,
    pub flame_q: Complex64,
    pub flame_c: Complex64,
    pub slope: Complex64,
    pub is_primitive: bool, // true = cardioid-like (mini-Mandelbrot), false = circle-like (satellite bulb) or outside
}

// ============================================================================
// Classification
// ============================================================================

const MAX_NEWTON_STEPS: u32 = 32;
const EPSILON: i32 = -53; // 2^-53 for f64 precision
const ESCAPE_RADIUS_SQ: f64 = 4.;
const DETECTION_THRESHOLD: f64 = 0.25; // When |z|² drops to 25% of previous min

fn classify_point(c: Complex64, max_iter: u64, frac_part: f64, log: bool) -> Classification {
    let mut z = c;
    let mut dz = Complex64::new(1., 0.); // dz/dc starts at 1
    let mut z_smallest = f64::MAX;
    let mut tracker = OrbitTrackerf64::new(z, 1.);
    let epsilon_squared = 2f64.powi(EPSILON);
    let mut iter = 1;

    // Stage 0
    // Iterate until we know the point is outside the set or we can prove it's inside
    while iter <= max_iter {
        let iter_f = iter as f64;
        let z_mag_sq = z.norm_sqr();

        // Accumulate weighted sums
        tracker.z = z;
        tracker.z_mag_sq = z_mag_sq;
        tracker.iter = iter_f;

        // Once we know it'll escape, we can stop Newton checks
        if z_mag_sq > ESCAPE_RADIUS_SQ {
            break;
        }
        tracker.iterate_small();
        // Check for period detection when deep inside
        if z_mag_sq < z_smallest {
            let ratio = z_mag_sq / z_smallest;
            z_smallest = z_mag_sq;

            if ratio < DETECTION_THRESHOLD {
                // Try to find periodic attractor
                if let Some(interior) =
                    find_interior_attractor(c, z, iter, MAX_NEWTON_STEPS, epsilon_squared, log)
                {
                    return Classification::Inside(interior);
                }
            }
        }

        // Update derivative and iterate
        dz = 2. * z * dz + 1.;
        z = z * z + c;
        iter += 1;
    }

    // Stage 0 complete - either hit max_iter or z is escaping
    if iter > max_iter {
        // Hit max iterations - apply fractional iteration if needed
        if frac_part > 0.0 {
            let dz_step = 2.0 * z * dz;
            let z_frac = z + frac_part * dz_step;
            let z_frac_mag_sq = z_frac.norm_sqr();

            if log {
                println!(
                    "  Fractional step at max_iter: z_frac_mag_sq={}",
                    z_frac_mag_sq
                );
            }

            // Check if the fractional step causes escape
            if z_frac_mag_sq > ESCAPE_RADIUS_SQ && z_frac_mag_sq.is_finite() {
                return Classification::Escaped(EscapeState {
                    iter: ScalarF6E7::from(max_iter as f64 + frac_part),
                    z: CircleF6E7::from(z_frac),
                    z_mag_sq: ScalarF6E7::from(z_frac_mag_sq),
                    slope_product: CircleF6E7::from(dz),
                    accumulators: AccumulatorsF6E7::from(tracker),
                });
            }
        }
        return Classification::MaxIterReached(tracker);
    }

    // Stage 1: Known escaping - iterate until non-normal
    // Pattern: compute next -> check normality -> commit if clean
    loop {
        let iter_f = iter as f64;
        let z_mag_sq = z.norm_sqr();

        // Compute next iteration values WITHOUT committing
        let dz_next = 2.0 * z * dz + 1.0;
        let z_next = z * z + c;
        let z_next_mag_sq = z_next.norm_sqr();

        // Check normality BEFORE committing
        if z_next_mag_sq > f64::MAX.sqrt() || !z_next.is_normal() {
            // Values went non-normal - return CURRENT (last good) state
            if log {
                println!("  Escaped at iter {}", iter);
            }
            return Classification::Escaped(EscapeState {
                iter: ScalarF6E7::from(iter_f),
                z: CircleF6E7::from(z),
                z_mag_sq: ScalarF6E7::from(z_mag_sq),
                slope_product: CircleF6E7::from(dz),
                accumulators: AccumulatorsF6E7::from(tracker),
            });
        }

        // Values are clean - accumulate with current state
        tracker.z = z;
        tracker.z_mag_sq = z_mag_sq;
        tracker.iter = iter_f;
        tracker.iterate_big();

        // Commit the new values
        dz = dz_next;
        z = z_next;
        iter += 1;
    }
}

/// Newton's method to find periodic attractor
fn find_interior_attractor(
    c: Complex64,
    z_guess: Complex64,
    period: u64,
    max_steps: u32,
    epsilon_sq: f64,
    log: bool,
) -> Option<InteriorResult> {
    let mut w = z_guess;

    for _ in 0..max_steps {
        // Iterate w thru one full period, tracking derivative
        let mut u = w;
        let mut du = Complex64::new(1., 0.);

        for _ in 0..period {
            du = 2. * du * u;
            u = u * u + c;
        }

        // Newton step: w_new = w - (u - w) / (du - 1)
        let numer = u - w;
        let denom = du - 1.;
        let step = numer / denom;
        let step_mag_sq = step.norm_sqr();

        w = w - step;

        if step_mag_sq < epsilon_sq {
            // Converged - check if it's actually inside (|multiplier| < 1)
            let multiplier_mag_sq = du.norm_sqr();
            if multiplier_mag_sq < 1. {
                // Find nucleus and determine if primitive or satellite
                let is_primitive =
                    if let Some(nucleus) = find_nucleus(c, period, max_steps, epsilon_sq) {
                        let s = shape_estimate(nucleus, period);
                        let primitive = is_shape_primitive(s);
                        if log {
                            println!(
                                "  Shape estimate: s={}, |s|={:.4}, |s-1|={:.4} → {}",
                                s,
                                s.norm(),
                                (s - 1.).norm(),
                                if primitive { "PRIMITIVE" } else { "SATELLITE" }
                            );
                        }
                        primitive
                    } else {
                        // Fallback: couldn't find nucleus, assume satellite (safer default)
                        if log {
                            println!("  Could not find nucleus, defaulting to satellite");
                        }
                        false
                    };

                if log {
                    println!(
                        "  Found interior: period={}, attractor={}, |mult|²={}, primitive={}",
                        period, w, multiplier_mag_sq, is_primitive
                    );
                }
                return Some(InteriorResult {
                    period,
                    attractor: w,
                    multiplier: du,
                    is_primitive,
                });
            }
        }
    }

    None
}

/// Find the nucleus (center) of a hyperbolic component using Newton's method.
/// The nucleus c₀ satisfies F^p(0, c₀) = 0.
///
/// Starting guess should be a point known to be inside the component.
fn find_nucleus(
    c_guess: Complex64,
    period: u64,
    max_steps: u32,
    epsilon_sq: f64,
) -> Option<Complex64> {
    let mut c = c_guess;

    for _ in 0..max_steps {
        // Iterate z = 0 thru period iterations, tracking dz/dc
        let mut z = Complex64::new(0., 0.);
        let mut dc = Complex64::new(0., 0.);

        for _ in 0..period {
            dc = 2. * z * dc + 1.;
            z = z * z + c;
        }

        // Newton step: c_new = c - z / dc
        let step = z / dc;
        let step_mag_sq = step.norm_sqr();

        c = c - step;

        if step_mag_sq < epsilon_sq {
            return Some(c);
        }
    }

    None
}

/// Compute the shape estimate for a hyperbolic component.
/// Based on Dolotin's formula from "On the shapes of elementary domains".
///
/// Returns a complex number s where:
/// - |s| ≈ 0 indicates cardioid-like (primitive/mini-Mandelbrot)
/// - |s| ≈ 1 indicates circle-like (satellite bulb)
fn shape_estimate(nucleus: Complex64, period: u64) -> Complex64 {
    let mut z = nucleus;
    let mut dc = Complex64::new(1., 0.);
    let mut dz = Complex64::new(1., 0.);
    let mut dcdc = Complex64::new(0., 0.);
    let mut dcdz = Complex64::new(0., 0.);

    for _ in 1..period {
        dcdc = 2. * (z * dcdc + dc * dc);
        dcdz = 2. * (z * dcdz + dc * dz);
        dc = 2. * z * dc + 1.;
        dz = 2. * z * dz;
        z = z * z + nucleus;
    }

    -(dcdc / (2. * dc) + dcdz / dz) / (dc * dz)
}

/// Determine if a shape estimate indicates a primitive (cardioid-like) component.
///
/// Primitive = mini-Mandelbrot with cusp, standalone structure
/// Satellite = circle-like bulb attached to parent component
#[inline]
fn is_shape_primitive(s: Complex64) -> bool {
    s.norm() < (s - 1.).norm() // closer to 0 than to 1
}

/// Public API: Determine if a point c (known to be interior with given period)
/// is in a primitive (mini-Mandelbrot) or satellite (bulb) component.
///
/// # Arguments
/// * `c` - A point known to be inside the Mandelbrot set
/// * `period` - The period of the hyperbolic component containing c
///
/// # Returns
/// * `Some(true)` - Primitive (cardioid-like, mini-Mandelbrot with cusp)
/// * `Some(false)` - Satellite (circle-like bulb attached to parent)
/// * `None` - Could not determine (Newton's method failed to find nucleus)
pub fn is_primitive(c: Complex64, period: u64) -> Option<bool> {
    let epsilon_sq = 2f64.powi(-106); // Double precision squared
    let max_steps = 64;

    let nucleus = find_nucleus(c, period, max_steps, epsilon_sq)?;
    let s = shape_estimate(nucleus, period);
    Some(is_shape_primitive(s))
}

/// Continue escaping in f64 until we need higher precision (for now, just finalize immediately)
fn escape_f6e7(state: EscapeState, log: bool) -> MandelbrotResult {
    // For now, we'll just finalize immediately at f64 precision
    // Later this will continue iterating until values become non-normal

    finalize_escape_f64(
        state.z.into(),
        state.z_mag_sq.to_f64(),
        state.slope_product.into(),
        state.iter.to_f64(),
        state.accumulators,
        log,
    )
}

// ============================================================================
// Finalization
// ============================================================================

const LOG_LOG_MAX: f64 = 5.174750624576761; // ln(ln(f64::MAX))

fn finalize_escape_f64(
    z: Complex64,
    z_mag_sq: f64,
    slope_product: Complex64,
    iter: f64,
    accumulators: AccumulatorsF6E7,
    log: bool,
) -> MandelbrotResult {
    // Continuous iteration count: μ = n - log₂(log(|z|²) / log(MAX))
    let log_mag_sq = z_mag_sq.ln();
    let log_log_mag_sq = log_mag_sq.ln();
    let escape_velocity = iter - (log_log_mag_sq - LOG_LOG_MAX) * std::f64::consts::E.log2();

    // Normalized coordinates: weighted average
    let inv_coord_weight = 1.0 / accumulators.coord_weight.to_f64();
    let coords: Complex64 = accumulators.coord_sum.into();
    let coords = coords * inv_coord_weight;

    // Normalized quadratic flame: weighted average
    let inv_flame_q_weight = 1.0 / accumulators.flame_q_weight.to_f64();
    let flame_q: Complex64 = accumulators.flame_q_sum.into();
    let flame_q = flame_q * inv_flame_q_weight;

    // Normalized cubic flame: weighted average
    let inv_flame_c_weight = 1.0 / accumulators.flame_c_weight.to_f64();
    let flame_c: Complex64 = accumulators.flame_c_sum.into();
    let flame_c = flame_c * inv_flame_c_weight;

    // Distance estimate slope: ln(|z|²) * z / (dz/dc) / sqrt(MAX)
    // Note: conjugate both z and slope_product, scale slope_product first
    let z_conj = Complex64::new(z.re, -z.im);
    let slope_conj = Complex64::new(slope_product.re, -slope_product.im) / f64::MAX.sqrt();
    let slope = (log_mag_sq * z_conj / slope_conj) / f64::MAX.sqrt();

    if log {
        println!(
            "  Finalized f64: escape_velocity={}, coords={}, flame_q={}, flame_c={}, slope={}",
            escape_velocity, coords, flame_q, flame_c, slope
        );
        println!("    z={}, slope_product={}", z, slope_product);
    }

    MandelbrotResult {
        period: 0,
        escape_velocity,
        coords,
        flame_q,
        flame_c,
        slope,
        is_primitive: false, // Outside points are not primitive
    }
}

/// Compute interior slope via Newton refinement to find nucleus
fn compute_interior_slope(c: Complex64, period: u64, max_steps: u32) -> Complex64 {
    let mut c_refined = c;

    for _ in 0..max_steps {
        let mut z = c_refined;
        let mut dz = Complex64::new(1., 0.);

        for _ in 1..period {
            dz = 2. * dz * z + 1.;
            z = z * z + c_refined;
        }

        // Newton step toward nucleus
        let next = c_refined - z / dz + Complex64::new(0., 2f64.powi(-256));
        if next == c_refined {
            break;
        }
        c_refined = next;
    }

    // Slope is vector from nucleus to original point
    let slope = c - c_refined;
    Complex64::new(slope.re, -slope.im)
}

// ============================================================================
// Public API
// ============================================================================

pub fn calculate_mandelbrot_point_f64(c: Complex64, frac_iter: f64, log: bool) -> MandelbrotResult {
    let max_iter = frac_iter.floor() as u64;
    let frac_part = frac_iter.fract();

    if log {
        println!("c: {}", c);
    }

    let classification = classify_point(c, max_iter, frac_part, log);

    if log {
        match &classification {
            Classification::Inside(i) => println!("  Classified as Inside, period={}", i.period),
            Classification::Escaped(_) => println!("  Classified as Escaped"),
            Classification::MaxIterReached(_) => println!("  Classified as MaxIterReached"),
        }
    }

    match classification {
        Classification::Inside(interior) => {
            let slope = compute_interior_slope(c, interior.period, 32);
            MandelbrotResult {
                period: interior.period,
                escape_velocity: max_iter as f64,
                coords: interior.multiplier,
                flame_q: interior.attractor,
                flame_c: interior.attractor, // For interior points, both flames use attractor
                slope,
                is_primitive: interior.is_primitive,
            }
        }

        Classification::Escaped(state) => escape_f6e7(state, log),

        Classification::MaxIterReached(tracker) => {
            // Handle fractional iteration for max_iter case
            if frac_part > 0.0 {
                // We need to recompute the last iteration with fractional step
                // This is a simplified version - full version would track dz thru classify_point
                // For now, just return max_iter without fractional correction
                if log {
                    println!(
                        "  MaxIter with frac_part={} (not fully implemented)",
                        frac_part
                    );
                }
            }

            // Didn't escape, didn't prove inside - return normalized accumulators
            let inv_coord_weight = 1. / tracker.coord_weight;
            let inv_flame_q_weight = 1. / tracker.flame_q_weight;
            let inv_flame_c_weight = 1. / tracker.flame_c_weight;
            MandelbrotResult {
                period: 0,
                escape_velocity: max_iter as f64,
                coords: tracker.coord_sum * inv_coord_weight,
                flame_q: tracker.flame_q_sum * inv_flame_q_weight,
                flame_c: tracker.flame_c_sum * inv_flame_c_weight,
                slope: Complex64::new(0., 0.),
                is_primitive: false,
            }
        }
    }
}
pub fn iteratez(c: Complex64, frac_iter: f64) -> (bool, u64, f64, Complex64, Complex64, Complex64) {
    let max_iter = frac_iter.floor() as u64;
    let frac_part = frac_iter.fract();
    let mut inside = false;
    let c_real = c.re;
    let c_imag = c.im;
    let mut z_real = c_real;
    let mut z_imag = c_imag;
    let mut escape_velocity = max_iter as f64;
    let mut period = 0;
    let mut coord_real = 0f64;
    let mut coord_imag = 0f64;
    let mut d_sum = 0f64;
    let mut flame_real = 0f64;
    let mut flame_imag = 0f64;
    let mut esum = 0f64;
    let mut slope_real = 1f64;
    let mut slope_imag = 0f64;
    let mut o_real = 1f64;
    let mut o_imag = 0f64;
    let mut z_smallest = f64::MAX;
    let mns = 32; //max newton steps
    let eps = 2f64.powi(-53); //epsilon squared2^-53 is minimum precision
    let mut solved = false;
    'iter: for iter in 1..=max_iter {
        let iterf = iter as f64;

        let srpt = 2. * (slope_real * z_real - slope_imag * z_imag);
        slope_imag = 2. * (slope_real * z_imag + slope_imag * z_real);
        slope_real = srpt + 1.;

        let zrs = z_real * z_real;
        let zis = z_imag * z_imag;
        let zmagnitude = zrs + zis;
        let mut itersq = iterf * iterf;
        let durt;
        if zmagnitude > 4. {
            let llzs = zmagnitude.ln().ln();
            let mut taper = (llzs - (2. * 2f64.ln()).ln()) / 512f64.ln();
            taper = 1. - taper * taper;
            taper = taper * taper;
            taper = taper * taper * taper * itersq;
            let zsq = taper / zmagnitude.sqrt();
            esum = esum + taper;
            flame_real = flame_real + z_real * zsq;
            flame_imag = flame_imag + z_imag * zsq;

            let zsi = itersq / zmagnitude;
            d_sum = d_sum + zsi;
            coord_real = coord_real + z_real * zsi;
            coord_imag = coord_imag + z_imag * zsi;

            if zmagnitude > std::f64::MAX.sqrt() {
                escape_velocity = iterf - (llzs - 5.174750624576761) * std::f64::consts::E.log2();
                solved = true;
                break 'iter;
            }

            durt = 2. * (z_real * o_real - z_imag * o_imag) + 1.;
            o_imag = 2. * (o_real * z_imag + o_imag * z_real);
            o_real = durt;

            z_imag = 2. * z_real * z_imag + c_imag;
            z_real = zrs - zis + c_real;
        } else {
            let taper = itersq;
            let zsq = taper / zmagnitude.sqrt();
            esum = esum + taper;
            flame_real = flame_real + z_real * zsq;
            flame_imag = flame_imag + z_imag * zsq;
            let zsi = itersq / zmagnitude;
            d_sum = d_sum + zsi;
            coord_real = coord_real + z_real * zsi;
            coord_imag = coord_imag + z_imag * zsi;
            if z_smallest > zmagnitude {
                let adcs = zmagnitude / z_smallest;
                z_smallest = zmagnitude;
                if adcs < 0.25 {
                    let mut wr = z_real;
                    let mut wi = z_imag;
                    for _ in 0..mns {
                        let mut ur = wr;
                        let mut ui = wi;
                        let mut dur = 1.;
                        let mut dui = 0.;
                        for _ in 0..iter {
                            //du=2*du*u
                            let durt = 2. * (dur * ur - dui * ui);
                            dui = 2. * (dur * ui + dui * ur);
                            dur = durt;

                            //u=u*u+c
                            let uis = ui * ui;
                            ui = 2. * ur * ui + c_imag;
                            ur = ur * ur - uis + c_real;
                        }
                        let mut nwr = ur - wr;
                        let mut nwi = ui - wi;
                        let duro = dur - 1.;
                        let durt = duro * duro + dui * dui;
                        let nwrt = (nwr * duro + nwi * dui) / durt;
                        nwi = (nwi * duro - nwr * dui) / durt;
                        nwr = nwrt;
                        itersq = nwr * nwr + nwi * nwi;
                        wr = wr - nwr;
                        wi = wi - nwi;

                        if itersq < eps {
                            if dur * dur + dui * dui < 1. {
                                coord_real = dur;
                                coord_imag = dui;
                                flame_real = wr;
                                flame_imag = wi;
                                inside = true;
                                period = iter;
                                solved = true;
                                break 'iter;
                            }
                        }
                    }
                }
            }
            durt = 2. * (z_real * o_real - z_imag * o_imag) + 1.;
            o_imag = 2. * (o_real * z_imag + o_imag * z_real);
            o_real = durt;

            z_imag = 2. * z_real * z_imag + c_imag;
            z_real = zrs - zis + c_real;
        }
        if iter == max_iter && frac_part > 0. {
            // Perform fractional step
            let dzreal = 2.0 * (z_real * slope_real - z_imag * slope_imag);
            let dzimag = 2.0 * (z_real * slope_imag + z_imag * slope_real);

            z_real += frac_part * dzreal;
            z_imag += frac_part * dzimag;

            let final_magnitude = z_real * z_real + z_imag * z_imag;
            if final_magnitude > 4. && !solved {
                let llzs = final_magnitude.ln().ln();
                escape_velocity =
                    iterf + frac_part - (llzs - 5.174750624576761) * std::f64::consts::E.log2();
                solved = true;
            }
        }
    }
    let mut slope = Complex64::new(0., 0.);
    let coordinates;
    let flames;
    if inside {
        coordinates = Complex64::new(coord_real, coord_imag);
        flames = Complex64::new(flame_real, flame_imag);
        let cca = c;
        let mut ccc = cca;
        for _ in 0..mns {
            let mut zzz = ccc;
            let mut ddd = Complex64::new(1., 0.);
            for _ in 1..period {
                ddd = 2. * ddd * zzz + 1.;
                zzz = zzz * zzz + ccc;
            }
            ddd = ccc - zzz / ddd + Complex64::new(0., 2f64.powi(-256));
            if ddd == ccc {
                ccc = ddd;
                break;
            }
            ccc = ddd;
        }
        let a = cca - ccc;
        slope = Complex64::new(a.re, -a.im);
    } else {
        d_sum = 1. / d_sum;
        esum = 1. / esum;
        coordinates = Complex64::new(coord_real * d_sum, coord_imag * d_sum);
        flames = Complex64::new(flame_real * esum, flame_imag * esum);
        if solved {
            z_imag = -z_imag;
            o_imag = -o_imag;
            o_real = o_real / std::f64::MAX.sqrt();
            o_imag = o_imag / std::f64::MAX.sqrt();
            slope = ((z_real * z_real + z_imag * z_imag).ln()) * Complex64::new(z_real, z_imag)
                / (Complex64::new(o_real, o_imag))
                / std::f64::MAX.sqrt();
        }
    }
    (inside, period, escape_velocity, coordinates, flames, slope)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_origin_debug() {
        let c = Complex64::new(0.0, 0.0);
        let frac_iter = 100.0;

        println!("\n=== Debug origin point ===");
        let result = calculate_mandelbrot_point_f64(c, frac_iter, true);
        println!("Result: {:?}", result);

        assert!(result.period > 0, "Origin should be inside (period > 0)");
    }

    #[test]
    fn test_refactored_vs_old() {
        let test_cases: Vec<(Complex64, f64)> = vec![
            (Complex64::new(0.0, 0.0), 100.0),  // Origin (inside)
            (Complex64::new(-0.5, 0.0), 100.0), // Main cardioid
            (Complex64::new(0.25, 0.0), 100.0), // Near edge
            (Complex64::new(-1.0, 0.0), 100.0), // Period-2 bulb
            (Complex64::new(0.5, 0.5), 100.0),  // Outside
            (Complex64::new(-0.7, 0.3), 100.0), // Inside
            (Complex64::new(0.3, 0.01), 50.5),  // Fractional iteration
        ];

        for (c, frac_iter) in test_cases {
            println!("\n=== Testing c={}, frac_iter={} ===", c, frac_iter);

            // Enable logging for fractional cases
            let enable_log = frac_iter.fract() > 0.0;

            // Old implementation
            let (old_inside, old_period, old_escape, old_coords, old_flame, old_slope) =
                iteratez(c, frac_iter);

            // New implementation
            let result = calculate_mandelbrot_point_f64(c, frac_iter, enable_log);

            let new_period = result.period;
            let new_escape = result.escape_velocity;
            let new_coords = result.coords;
            let new_flame = result.flame_q;
            let new_slope = result.slope;
            let new_inside = new_period > 0;

            // Compare results
            println!(
                "Old: inside={}, period={}, escape={}",
                old_inside, old_period, old_escape
            );
            println!(
                "New: inside={}, period={}, escape={}",
                new_inside, new_period, new_escape
            );

            assert_eq!(
                old_inside, new_inside,
                "Inside/outside mismatch for c={}",
                c
            );
            assert_eq!(old_period, new_period, "Period mismatch for c={}", c);

            // For escape velocity, allow small numerical differences
            let escape_diff = (old_escape - new_escape).abs();
            assert!(
                escape_diff < 0.01,
                "Escape velocity mismatch for c={}: old={}, new={}, diff={}",
                c,
                old_escape,
                new_escape,
                escape_diff
            );

            // Compare coords and flames (allow small differences)
            let coord_diff = (old_coords - new_coords).norm();
            assert!(
                coord_diff < 0.01,
                "Coords mismatch for c={}: diff={}",
                c,
                coord_diff
            );

            let flame_diff = (old_flame - new_flame).norm();
            if flame_diff >= 0.01 {
                println!("  Old flame: {}", old_flame);
                println!("  New flame: {}", new_flame);
            }
            assert!(
                flame_diff < 0.01,
                "Flame mismatch for c={}: diff={}, old={}, new={}",
                c,
                flame_diff,
                old_flame,
                new_flame
            );

            let slope_diff = (old_slope - new_slope).norm();
            if slope_diff >= 0.01 {
                println!("  Old slope: {}", old_slope);
                println!("  New slope: {}", new_slope);
            }
            assert!(
                slope_diff < 0.01,
                "Slope mismatch for c={}: diff={}, old={}, new={}",
                c,
                slope_diff,
                old_slope,
                new_slope
            );

            println!("✓ All values match within tolerance");
        }
    }
}
