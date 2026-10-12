use spirix::{CircleF7E5, CircleF7E7};

/// The Mandelbrot set spans 4 units at default zoom (from -2 to +2)
pub const MANDELBROT_DEFAULT_SPAN: f64 = 4.0;

/// Binary32 (f32) has 24 bits of fraction precision
pub const F32_FRACTION_BITS: i32 = 24;

/// Binary64 (f64) has 53 bits of fraction precision
pub const F64_FRACTION_BITS: i32 = 53;

/// F7E5 has 128 bits of fraction precision (tho limited exponent range)
pub const F7E5_FRACTION_BITS: i32 = 128;

/// Precision mode for Mandelbrot calculations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionMode {
    Binary32,

    /// Standard IEEE 754 double precision (fast)
    /// ~15-16 decimal digits, exponent range ±308
    Binary64,

    /// High precision, limited range (for deep zooms into the set)
    /// ~38 decimal digits, exponent range ±38
    F7E5,

    /// Low precision, huge range (for tracing to infinity)
    /// ~2 decimal digits, exponent range up to 10^(10^37)
    F3E7,
}

impl PrecisionMode {
    /// Cycle to the next precision mode
    pub fn next(self) -> Self {
        match self {
            PrecisionMode::Binary32 => PrecisionMode::Binary64,
            PrecisionMode::Binary64 => PrecisionMode::F7E5,
            PrecisionMode::F7E5 => PrecisionMode::F3E7,
            PrecisionMode::F3E7 => PrecisionMode::Binary32,
        }
    }

    /// Get a short label for display
    pub fn label(&self) -> &'static str {
        match self {
            PrecisionMode::Binary32 => "F32",
            PrecisionMode::Binary64 => "F64",
            PrecisionMode::F7E5 => "F7E5",
            PrecisionMode::F3E7 => "F3E7",
        }
    }

    /// Get a description of this mode
    pub fn description(&self) -> &'static str {
        match self {
            PrecisionMode::Binary32 => "Single precision (fastest, shallow zoom only)",
            PrecisionMode::Binary64 => "Standard precision (fast)",
            PrecisionMode::F7E5 => "High precision, limited range (deep zoom)",
            PrecisionMode::F3E7 => "Low precision, huge range (infinity trace)",
        }
    }
}

impl Default for PrecisionMode {
    fn default() -> Self {
        PrecisionMode::Binary64
    }
}

/// Compute the log2 of zoom magnitude from a CircleF7E5 zoom value
///
/// Does all computation in F7E5 to avoid underflow when converting tiny values to f64
pub fn compute_zoom_magnitude_log2(zoom: CircleF7E5) -> f64 {
    // Take log2 in F7E5 precision, then convert final result to f64
    // log2(sqrt(x)) = log2(x) / 2
    // Note: lb() is spirix's method for log base 2 (logarithmus binarius)
    let log2_mag_sq: f64 = (zoom.magnitude_squared().lb() >> 1u8).into();
    log2_mag_sq
}

/// Compute the zoom threshold where f32 precision becomes insufficient
///
/// Returns the log2 of the pixel size at which f32 precision becomes insufficient.
/// Since zoom represents pixel size in complex plane units, this returns a negative value.
/// When zoom_log2 < this threshold, switch to f64.
pub fn compute_f32_to_f64_threshold(screen_width: f64) -> f64 {
    // Default pixel size = MANDELBROT_DEFAULT_SPAN / screen_width
    // f32 precision = 2^-F32_FRACTION_BITS
    // Pixel size must be > f32 precision for accurate rendering
    // threshold_pixel_size = default_pixel_size / magnification_limit
    // threshold_pixel_size = (4.0 / screen_width) / ((screen_width / 4.0) * 2^24)
    // threshold_pixel_size = 2^-24 / (screen_width / 4.0)
    // Actually simpler: log2(default_pixel_size) - log2(magnification_threshold)
    let default_pixel_log2 = (MANDELBROT_DEFAULT_SPAN / screen_width).log2();
    let mag_threshold_log2 =
        ((screen_width / MANDELBROT_DEFAULT_SPAN) * 2.0_f64.powi(F32_FRACTION_BITS)).log2();
    default_pixel_log2 - mag_threshold_log2
}

/// Compute the zoom threshold where f64 precision becomes insufficient
///
/// Returns the log2 of the pixel size at which f64 precision becomes insufficient.
/// Since zoom represents pixel size in complex plane units, this returns a negative value.
/// When zoom_log2 < this threshold, switch to F7E5.
pub fn compute_f64_to_f7e5_threshold(screen_width: f64) -> f64 {
    // Same logic as f32 threshold but with f64's 53-bit precision
    let default_pixel_log2 = (MANDELBROT_DEFAULT_SPAN / screen_width).log2();
    let mag_threshold_log2 =
        ((screen_width / MANDELBROT_DEFAULT_SPAN) * 2.0_f64.powi(F64_FRACTION_BITS)).log2();
    default_pixel_log2 - mag_threshold_log2
}

/// Compute escape velocity constant for smooth coloring
///
/// This constant is ln(ln(threshold)) where threshold is the escape detection limit.
/// For standard precision (f64), we use 2^256 as the threshold.
///
/// Derivation:
/// - Escape threshold: 2^256
/// - ln(2^256) = 256 * ln(2) ≈ 177.445678...
/// - ln(ln(2^256)) = ln(177.445678...) ≈ 5.174750624576761
///
/// Note: This is marked `const fn` but returns a precomputed value since
/// ln() is not yet available in const contexts. To verify:
/// `basecalc` command: #ln(#ln(2^256))
pub const fn escape_velocity_constant_f64() -> f64 {
    5.174750624576761
}

/// The constant 1/ln(2) for converting natural log to log2
///
/// Used for converting natural logarithms to base-2 logarithms:
/// log2(x) = ln(x) / ln(2) = ln(x) * (1/ln(2))
///
/// Note: This is marked `const fn` but returns a precomputed value since
/// ln() is not yet available in const contexts. To verify:
/// `basecalc` command: 1/@e$2  (or in standard notation: 1/ln(2))
pub const fn ln2_reciprocal() -> f64 {
    1.442695040888963
}

/// Select the appropriate precision mode based on zoom magnitude and screen width
pub fn select_precision(zoom_magnitude_log2: f64, screen_width: f64) -> PrecisionMode {
    let threshold_f32_to_f64 = compute_f32_to_f64_threshold(screen_width);
    let threshold_f64_to_f7e5 = compute_f64_to_f7e5_threshold(screen_width);

    if zoom_magnitude_log2 < threshold_f32_to_f64 {
        PrecisionMode::Binary32
    } else if zoom_magnitude_log2 < threshold_f64_to_f7e5 {
        PrecisionMode::Binary64
    } else {
        PrecisionMode::F7E5
    }
}
