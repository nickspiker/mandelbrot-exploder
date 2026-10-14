//! Scaled f32 perturbation pipeline.
//!
//! One reference orbit `Z` of the critical point is iterated in F7E3 (the coordinate type) and
//! stored as f64 + f32. Each pixel then iterates only its offset from that orbit, with every
//! per-pixel quantity divided by the zoom magnitude `S` so an 8-bit exponent never sees the
//! depth. Primed names are scaled (`x' = x / S`):
//!
//! ```text
//!   z[n]      = Z[m] + S·δz'[n]                       full value, formed only when needed
//!   δz'[n+1]  = (Z[m] + z[n])·δz'[n] + δc'            = 2Z·δz' + S·δz'² + δc'
//!   d'[n+1]   = 2·z[n]·d'[n] + S                      both derivative trackers, d' = S·d
//!   rebase    : if |Z[m+1] + S·δz'| < |S·δz'| { δz' = z/S; m = 0 }
//! ```
//!
//! The pixel pipeline is the same algorithm as `mandelbrot_f32.rs` / `spirix_grid.rs`, split
//! by what needs the delta frame:
//!
//! * **Pre-escape loop (f32, scaled):** escape test, both flame sums, the coordinate sum, both
//!   derivatives and the Newton trigger (new |z| minimum by more than 4×). This is the part a
//!   GPU kernel can own: `DeltaState` + `run_f32` are the kernel.
//! * **Escape ride (f32 then f64, full values):** once |z|² > 4 the offset no longer matters and
//!   the pixel rides plain `z² + c` to f32 overflow, then f64 overflow, exactly as the direct
//!   f32 pipeline does. Only the derivatives stay scaled.
//! * **Newton solves (f64, delta frame):** period detection and interior root refinement replay
//!   the orbit against the reference in f64. When the reference is a nucleus and the candidate
//!   period is a multiple of its period, every subtraction is between deltas and the result is
//!   exact to the pixel scale. Otherwise the solve is exact to 2⁻⁵³ absolute, which is enough
//!   down to zoom ≈ 2⁻⁴⁵; past that a foreign-period candidate is parked and retried against a
//!   nucleus of its own period (found in F7E3, cached per engine) once the pixel has run
//!   `NUCLEUS_BUDGET` iterations without escaping.
//!
//! Limits, all deliberate: f32's exponent gives out when `S` goes denormal near 2⁻¹²⁶; F7E3
//! locates a nucleus to ulp ≈ 2⁻¹²⁷ × |dz_P/dc|, so a reference can only be made exactly
//! periodic while that residual is below the pixel scale (roughly zoom > 2⁻⁶⁸); the nucleus
//! search is capped at `NUCLEUS_SEARCH_MAX` iterations per period. Beyond each of those the
//! pipeline degrades visibly rather than guarding, like the rest of the rig.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use bytemuck::{Pod, Zeroable};
use num_complex::{Complex, Complex64};
use rayon::prelude::*;
use spirix::{CircleF7E3, CircleF7E7, ScalarF7E3, ScalarF7E7};

use super::mandelbrot_f64::MandelbrotResult;

type C32 = Complex<f32>;

/// Longest reference orbit kept (16 + 8 bytes per entry). Pixels that outlive it rebase to
/// index 0 with full-value precision.
pub const REF_MAX_LEN: usize = 1 << 22;
/// Length of the plain orbit built synchronously on navigation; the background worker extends
/// it to `REF_MAX_LEN` (or replaces it with a nucleus) afterwards.
pub const REF_INITIAL_LEN: usize = 1 << 20;
/// Largest candidate period for which a nucleus is searched in F7E3 (32 Newton steps × period
/// F7E3 iterations, ≈ 95 ns each).
pub const NUCLEUS_SEARCH_MAX: u32 = 1 << 16;
/// Iterations a pixel runs without escaping before its parked candidates are tried against
/// per-period nuclei.
pub const NUCLEUS_BUDGET: u32 = 1 << 20;
/// A nucleus further than this many screens from the centre is not used as a reference.
const NUCLEUS_MAX_SCREENS: f64 = 64.0;
/// Zoom magnitude above which the main-reference f64 Newton is trusted for any period: its
/// values are good to 2⁻⁵³ absolute, which leaves 17 bits of interior shading at this scale.
const F64_TRUST_SCALE: f64 = 1.4551915228366852e-11; // 2^-36
/// Below `F64_TRUST_SCALE`, candidates up to this period that the delta frame cannot solve
/// exactly are solved directly in F7E3 (32 Newton steps × period F7E3 iterations).
const F7E3_NEWTON_MAX_PERIOD: u32 = 1 << 12;
const MAX_NEWTON_STEPS: usize = 32;
const MAX_PENDING: usize = 8;
/// A confirmed candidate is solved only after the pixel has run this many more iterations
/// (at least one more period): an exterior pixel whose probe was fooled by an orbit segment
/// near zero usually escapes in the meantime and never pays for the Newton solve.
const NEWTON_DEFER_MIN: u32 = 256;
const MAX_DEFERRED: usize = 8;
/// Newton replay iterations (an estimate of 8 replays × period per solve) allowed per step;
/// solves beyond it wait for the next step. Keeps the step's CPU pass bounded.
const NEWTON_STEP_BUDGET: u64 = 10_000_000;
/// √machine-epsilon for f64 Newton convergence, squared.
const EPSILON_SQ_F64: f64 = 2.220446049250313e-16; // 2^-52
/// Asymmetric nudge in interior root refinement — the grid's `TWO.pow(−2 × fraction_bits)`.
const EPS_ROOT_F64: f64 = 1.2325951644078309e-32; // 2^-106

// ============================================================================
// Reference orbit
// ============================================================================

/// Orbit of the critical point under a reference parameter, in the two precisions the pipeline
/// consumes. `z[0] = 0`. For a nucleus the orbit is one period long, `z[period] = 0` exactly,
/// and indices wrap.
pub struct Reference {
    pub c: CircleF7E3,
    pub c64: Complex64,
    pub z64: Vec<Complex64>,
    pub z32: Vec<C32>,
    /// `Some(P)` when the orbit is exactly periodic with period `P` (`z.len() == P + 1`).
    pub period: Option<u32>,
    /// |Z[P]| before it was forced to zero (0 for an exact nucleus, ∞ when not periodic): the
    /// reference stays exactly periodic for any zoom whose tolerance exceeds this.
    pub residual: f64,
    /// Zoom magnitude the reference was built for.
    pub built_scale: f64,
}

impl Reference {
    fn periodic(&self) -> bool {
        self.period.is_some()
    }

    /// Plain orbit of `c`: iterate until |Z|² > 4 (that value is kept), `max_len`, or stop.
    fn from_point(c: CircleF7E3, max_len: usize, stop: &AtomicBool) -> Option<Reference> {
        let mut z64 = Vec::with_capacity(1 << 16);
        let mut z = CircleF7E3::ZERO;
        let four = ScalarF7E3::from(4.0);
        loop {
            let zc: Complex64 = z.into();
            z64.push(zc);
            if z64.len() >= max_len || (!finite(z) && z64.len() > 1) {
                break;
            }
            if z64.len() > 1 && z.magnitude_squared() > four {
                break;
            }
            if z64.len() & 0xFFFF == 0 && stop.load(Ordering::Relaxed) {
                return None;
            }
            z = z.square() + c;
        }
        let z32 = z64.iter().map(|v| C32::new(v.re as f32, v.im as f32)).collect();
        Some(Reference { c, c64: c.into(), z64, z32, period: None, residual: f64::INFINITY, built_scale: 0.0 })
    }

    /// Orbit of a nucleus candidate `n` with period dividing `period`. Succeeds only when the
    /// orbit returns within `tolerance` of zero, which is then forced to exact zero so indices
    /// can wrap; the detected (minimal) period is stored.
    fn from_nucleus(n: CircleF7E3, period: u32, tolerance: f64) -> Option<Reference> {
        let tol_sq = tolerance * tolerance;
        let mut z64 = Vec::with_capacity(period as usize + 1);
        let mut z = CircleF7E3::ZERO;
        z64.push(Complex64::new(0., 0.));
        for _ in 0..period {
            z = z.square() + n;
            if !finite(z) {
                return None;
            }
            let zc: Complex64 = z.into();
            if zc.norm_sqr() < tol_sq {
                z64.push(Complex64::new(0., 0.));
                let z32 = z64.iter().map(|v| C32::new(v.re as f32, v.im as f32)).collect();
                let p = z64.len() as u32 - 1;
                return Some(Reference { c: n, c64: n.into(), z64, z32, period: Some(p), residual: zc.norm(), built_scale: 0.0 });
            }
            z64.push(zc);
        }
        None
    }
}

/// MANDEL_EXACT_TAIL=1: square in F7E7 all the way to the horizon instead of the f64 tail.
fn exact_tail() -> bool {
    static EXACT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *EXACT.get_or_init(|| std::env::var_os("MANDEL_EXACT_TAIL").is_some())
}

/// spirix's `is_normal` excludes exact zero, which is a legitimate orbit value here.
#[inline]
fn finite(z: CircleF7E3) -> bool {
    z.is_normal() || z == CircleF7E3::ZERO
}

/// Newton on `c` for `f^period_c(0) = 0` in F7E3, from `c0`. Returns the converged parameter;
/// the caller validates it by building a periodic orbit from it.
fn nucleus_f7e3(c0: CircleF7E3, period: u32, stop: &AtomicBool) -> Option<CircleF7E3> {
    let mut c = c0;
    for _ in 0..64 {
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        let mut z = c;
        let mut d = CircleF7E3::ONE;
        for _ in 1..period {
            // Doubled by addition: spirix's `Circle << 1` overflows for some operands.
            let dz = d * z;
            d = dz + dz + CircleF7E3::ONE;
            z = z.square() + c;
            if !finite(z) || !finite(d) {
                return None;
            }
        }
        if !d.is_normal() {
            return None;
        }
        let next = c - z / d;
        if !finite(next) {
            return None;
        }
        if next == c {
            return Some(c);
        }
        c = next;
    }
    Some(c)
}

// ============================================================================
// f32 delta loop — the GPU-shaped part
// ============================================================================

pub const STATUS_RUNNING: u32 = 0;
pub const STATUS_ESCAPED: u32 = 1;
pub const STATUS_CONFIRMED: u32 = 2;
pub const STATUS_DEAD: u32 = 3;

/// Per-pixel state of the scaled f32 loop: exactly what the compute kernel holds, in the layout
/// `delta.wgsl` declares (the vec2s first, then the scalars; 120 bytes).
///
/// The attraction probe (`probe_*`): a new |z| minimum by more than 4× at iteration `n` (the
/// direct pipelines' Newton trigger) arms it with `probe_n = n`. Over the next `n` iterations
/// it accumulates |Π 2z|², the squared derivative of fⁿ along the orbit. On an orbit settling
/// into an attracting cycle whose period divides `n` that product tends to the cycle
/// multiplier (below one); on an exterior orbit it grows exponentially. A window below one
/// confirms the candidate for the (deferred, budgeted) Newton solve; after `PROBE_WINDOWS`
/// windows without one the candidate is confirmed anyway rather than dropped. A trigger that arrives while a probe of
/// a divisor period is running is parked in `probe_blocked` and probed next, from a
/// phase-aligned iteration, if that probe or its Newton solve fails.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DeltaState {
    /// Scaled offset δz' from the reference orbit.
    pub dz: C32,
    /// Scaled pixel offset δc' from the reference parameter.
    pub dcs: C32,
    /// S × dz/dc (the grid's `orbit`; its `slope` twin is never read and is not tracked).
    pub orbit: C32,
    pub coord_acc: C32,
    pub flame_u: C32,
    pub flame_w: C32,
    /// Full z at the escape iteration (valid when `status == STATUS_ESCAPED`).
    pub z_out: C32,
    pub z_min: f32,
    pub dist_sum: f32,
    pub energy_u: f32,
    pub energy_w: f32,
    /// |Π 2z|² so far in the probe window, saturated at `PROBE_SATURATE`.
    pub probe_acc: f32,
    /// Reference index: full z = Z[m] + S·δz'.
    pub m: u32,
    /// Pixel iteration count; z is f^iter(0).
    pub iter: u32,
    /// The loop returns with `STATUS_RUNNING` once `iter` reaches this.
    pub limit: u32,
    pub status: u32,
    pub probe_n: u32,
    /// Iteration at which the current window starts accumulating.
    pub probe_start: u32,
    /// Iteration at which the current window ends.
    pub probe_next: u32,
    pub probe_windows: u32,
    pub probe_blocked: u32,
    /// Confirmed candidate period (valid when `status == STATUS_CONFIRMED`).
    pub period: u32,
    pub _pad: u32,
}

const _: () = assert!(std::mem::size_of::<DeltaState>() == 120);

const PROBE_WINDOWS: u32 = 8;
const PROBE_SATURATE: f32 = 1e20;

impl DeltaState {
    pub fn new(dcs: C32, s: f32) -> Self {
        let mut st = Self::zeroed();
        st.dz = dcs;
        st.dcs = dcs;
        st.m = 1;
        st.iter = 1;
        st.orbit = C32::new(s, 0.);
        st.z_min = f32::MAX;
        st.probe_acc = 1.;
        st
    }

    /// Arm the probe for candidate period `n`, window starting at iteration `start`.
    #[inline]
    pub fn probe_arm(&mut self, n: u32, start: u32) {
        self.probe_n = n;
        self.probe_start = start;
        self.probe_next = start + n;
        self.probe_acc = 1.;
        self.probe_windows = 0;
    }

    /// The Newton solve for a confirmed candidate failed: let a blocked multiple try, from
    /// its next phase-aligned iteration.
    #[inline]
    pub fn probe_failed(&mut self) {
        if self.probe_blocked != 0 {
            let n = self.probe_blocked;
            self.probe_blocked = 0;
            let start = self.iter.div_ceil(n) * n;
            self.probe_arm(n, start);
        }
    }
}

/// Window end reached: confirm, start another window, or drop (handing over to a blocked
/// multiple if there is one).
#[inline]
fn probe_window_end(st: &mut DeltaState) -> bool {
    if st.probe_acc < 1.0 {
        st.probe_n = 0;
        return true;
    }
    st.probe_windows += 1;
    if st.probe_windows < PROBE_WINDOWS {
        st.probe_start = st.iter;
        st.probe_next = st.iter + st.probe_n;
        st.probe_acc = 1.;
        return false;
    }
    // Windows exhausted without a clear verdict: confirm anyway. The deferred, budgeted
    // solve is cheap insurance, and an interior pixel whose probe stayed marginal (|λ| ≈ 1,
    // or a noisy transient) must never be left iterating to the cap.
    st.probe_n = 0;
    true
}

/// Run the scaled f32 loop until an event or `st.limit` iterations; the outcome is left in
/// `st.status`. This is the CPU twin of `delta.wgsl`.
pub fn run_f32(st: &mut DeltaState, r: &Reference, s: f32) {
    let last = r.z32.len() as u32 - 1;
    let periodic = r.periodic();
    let dcs = st.dcs;
    while st.iter < st.limit {
        let zm = r.z32[st.m as usize];
        let z = zm + st.dz * s;
        let z_mag_sq = z.norm_sqr();
        if !z_mag_sq.is_finite() {
            st.status = STATUS_DEAD;
            return;
        }
        if z_mag_sq > 4.0 {
            st.z_out = z;
            st.status = STATUS_ESCAPED;
            return;
        }
        if st.probe_n != 0 && st.iter == st.probe_next {
            let n = st.probe_n;
            if probe_window_end(st) {
                st.period = n;
                st.status = STATUS_CONFIRMED;
                return;
            }
        }
        if st.probe_n != 0 && st.iter >= st.probe_start {
            st.probe_acc = (st.probe_acc * 4.0 * z_mag_sq).min(PROBE_SATURATE);
        }
        let it = st.iter as f32;
        let iter_sq = it * it;
        let z_mag = z_mag_sq.sqrt();
        st.energy_u += 1.0;
        st.flame_u += z / z_mag;
        st.energy_w += iter_sq;
        st.flame_w += z * (iter_sq / z_mag);
        let w = iter_sq / z_mag_sq;
        st.dist_sum += w;
        st.coord_acc += z * w;

        if st.z_min > z_mag_sq {
            let ratio = z_mag_sq / st.z_min;
            st.z_min = z_mag_sq;
            // Arm the probe on a new 4× minimum. A running probe whose period divides this
            // iteration keeps priority (smaller period, first to be tested) and remembers
            // this one as its fallback; any other running probe is replaced.
            if ratio < 0.25 {
                if st.probe_n != 0 && st.iter % st.probe_n == 0 {
                    st.probe_blocked = st.iter;
                } else {
                    st.probe_blocked = 0;
                    st.probe_arm(st.iter, st.iter);
                }
            }
        }

        st.orbit = (st.orbit * z) * 2.0 + s;
        st.dz = (zm + z) * st.dz + dcs;
        st.m += 1;
        st.iter += 1;
        if st.m >= last {
            if periodic && st.m == last {
                // Z[P] == 0 == Z[0]: wrap without touching the delta.
                st.m = 0;
            } else {
                // Reference ended (it escaped, or hit the cap): continue on full value.
                let zn = r.z32[last.min(st.m) as usize] + st.dz * s;
                st.dz = zn / s;
                st.m = 0;
            }
        } else {
            let t = st.dz * s;
            let zn = r.z32[st.m as usize] + t;
            if zn.norm_sqr() < t.norm_sqr() {
                st.dz = zn / s;
                st.m = 0;
            }
        }
    }
    st.status = STATUS_RUNNING;
}

/// Something that runs the f32 loop over a batch of states: the CPU, or a compute kernel.
pub trait DeltaRunner: Send {
    fn name(&self) -> String;
    fn run(&mut self, r: &Reference, s: f32, states: &mut [DeltaState]);
}

pub struct CpuRunner;

/// The best runner available: the Metal compute path on macOS, else the CPU.
pub fn make_runner() -> Box<dyn DeltaRunner> {
    #[cfg(target_os = "macos")]
    {
        if std::env::var_os("MANDEL_NO_GPU").is_none() {
            if let Some(gpu) = super::gpu_delta::GpuRunner::new() {
                return Box::new(gpu);
            }
        }
    }
    Box::new(CpuRunner)
}

/// Pixels kept in flight and iterations per dispatch for a runner: the GPU wants a wide,
/// deep batch; the CPU a shallow one so a step stays inside a frame.
pub fn batch_shape(runner: &dyn DeltaRunner) -> (usize, u32) {
    if runner.name().starts_with("GPU") {
        (1 << 16, 2048)
    } else {
        (1 << 14, 512)
    }
}

impl DeltaRunner for CpuRunner {
    fn name(&self) -> String {
        "CPU".into()
    }
    fn run(&mut self, r: &Reference, s: f32, states: &mut [DeltaState]) {
        states.par_iter_mut().for_each(|st| run_f32(st, r, s));
    }
}

// ============================================================================
// f64 delta-frame Newton solves
// ============================================================================

/// One perturbed step in f64 against `r`: returns the full value `u` that was consumed.
#[inline]
fn step64(r: &Reference, mu: &mut usize, du: &mut Complex64, dc: Complex64) -> Complex64 {
    let zm = r.z64[*mu];
    let u = zm + *du;
    *du = (zm + u) * *du + dc;
    *mu += 1;
    let last = r.z64.len() - 1;
    if *mu >= last {
        if r.periodic() && *mu == last {
            *mu = 0;
        } else {
            *du = r.z64[last.min(*mu)] + *du;
            *mu = 0;
        }
    } else {
        let un = r.z64[*mu] + *du;
        if un.norm_sqr() < du.norm_sqr() {
            *du = un;
            *mu = 0;
        }
    }
    u
}

/// Period-detection Newton (the grid's inner `w` solve) in the delta frame. `w = Z[m0] + dw0`
/// is the orbit point at the trigger. Returns `(w, multiplier)` when a periodic point of period
/// `n` is found and attracting.
fn newton_period(
    r: &Reference,
    dc: Complex64,
    n: u32,
    m0: usize,
    dw0: Complex64,
) -> Option<(Complex64, Complex64, u32)> {
    let mw = m0;
    let mut dw = dw0;
    for step in 0..MAX_NEWTON_STEPS {
        let mut mu = mw;
        let mut du = dw;
        let mut d = Complex64::new(1., 0.);
        for _ in 0..n {
            let u = step64(r, &mut mu, &mut du, dc);
            d = u * d * 2.;
        }
        // First replay: a multiplier estimate already far above one means any cycle near the
        // start is strongly repelling, i.e. an exterior pixel; stop before paying for more.
        if step == 0 && d.norm_sqr() > 256. {
            return None;
        }
        // u − w: between deltas when the indices line up (exact), else between full values.
        let diff = if mu == mw { du - dw } else { (r.z64[mu] + du) - (r.z64[mw] + dw) };
        let denom = d - 1.;
        let newton = diff * denom.conj() / denom.norm_sqr();
        let newton_mag_sq = newton.norm_sqr();
        dw -= newton;
        if newton_mag_sq < EPSILON_SQ_F64 {
            if d.norm_sqr() < 1. {
                // Reduce to the minimal period: the smallest divisor of n after which the
                // cycle point returns to itself. The trigger period is often a multiple
                // (the orbit's phase), and the colouring hashes the period.
                let w = r.z64[mw] + dw;
                let (period, d) = minimal_period(r, dc, n, mw, dw, w).unwrap_or((n, d));
                return Some((w, d, period));
            }
        }
        if !newton_mag_sq.is_finite() || (r.z64[mw] + dw).norm_sqr() > 16. {
            return None;
        }
    }
    None
}

/// Smallest divisor `d` of `n` for which the periodic point `w = Z[mw] + dw` returns to
/// itself after `d` steps, with the multiplier of that `d`-cycle.
fn minimal_period(
    r: &Reference,
    dc: Complex64,
    n: u32,
    mw: usize,
    dw: Complex64,
    w: Complex64,
) -> Option<(u32, Complex64)> {
    let tol = 1e-9 * w.norm().max(1.0);
    let mut d = 1u32;
    while d < n {
        if n % d == 0 {
            let mut mu = mw;
            let mut du = dw;
            let mut mult = Complex64::new(1., 0.);
            for _ in 0..d {
                let u = step64(r, &mut mu, &mut du, dc);
                mult = u * mult * 2.;
            }
            let back = if mu == mw { du - dw } else { (r.z64[mu] + du) - w };
            if back.norm() < tol {
                return Some((d, mult));
            }
        }
        d += 1;
    }
    None
}

/// Interior root refinement in the delta frame: Newton on δc for `f^n_c(0) = 0`. Returns
/// `c_initial − c_refined`, the pixel's offset from its component's nucleus.
fn refine_nucleus(r: &Reference, dc0: Complex64, n: u32) -> Complex64 {
    let mut dc = dc0;
    for _ in 0..MAX_NEWTON_STEPS {
        let mut mu = 1usize;
        let mut dz = dc;
        let mut d = Complex64::new(1., 0.);
        if r.z64.len() < 2 {
            break;
        }
        for _ in 1..n {
            let z = step64(r, &mut mu, &mut dz, dc);
            d = d * z * 2. + 1.;
        }
        let zn = if mu == 0 && r.periodic() { dz } else { r.z64[mu] + dz };
        let next = dc - zn / d + Complex64::new(0., EPS_ROOT_F64);
        if !next.re.is_finite() || !next.im.is_finite() {
            break;
        }
        if next == dc {
            dc = next;
            break;
        }
        dc = next;
    }
    dc0 - dc
}

// ============================================================================
// Engine
// ============================================================================

/// Work counters, for the bench and for sizing the GPU hand-off.
#[derive(Default, Debug)]
pub struct Stats {
    pub newton_solves: AtomicU64,
    pub newton_hits: AtomicU64,
    pub f7e3_solves: AtomicU64,
    pub nucleus_searches: AtomicU64,
    pub nucleus_hits: AtomicU64,
    /// Nanoseconds (summed over threads) in the escape ride, the Newton solves, and wall
    /// nanoseconds in the runner and in the whole drive pass of `step`.
    pub ns_escape: AtomicU64,
    pub ns_newton: AtomicU64,
    pub ns_runner: AtomicU64,
    pub ns_drive: AtomicU64,
    pub escapes: AtomicU64,
}

/// What the background worker is asked to do.
enum Job {
    /// A nucleus of this period near the pixel (foreign-period interior).
    Pixel { period: u32, coord: CircleF7E3 },
    /// The viewport's own reference: try these candidate periods at the centre and, failing an
    /// exact nucleus, extend the plain orbit to full length.
    Reference { center: CircleF7E3, periods: Vec<u32>, extend: bool },
}

/// State shared between an engine and its nucleus worker thread. Every F7E3 Newton search
/// runs here, off the step path; pixels that need one park in the batch until it lands.
struct Worker {
    queue: Mutex<VecDeque<Job>>,
    wake: Condvar,
    quit: AtomicBool,
    scale: f64,
    stats: Arc<Stats>,
    nuclei: Mutex<Vec<(u32, Arc<Reference>)>>,
    failed: Mutex<HashMap<u32, Vec<Complex64>>>,
    in_flight: Mutex<HashSet<u32>>,
    /// A better reference for the viewport (exact nucleus, or the full-length plain orbit).
    upgrade: Mutex<Option<Arc<Reference>>>,
}

impl Worker {
    fn enqueue(&self, job: Job) {
        self.queue.lock().unwrap().push_back(job);
        self.wake.notify_one();
    }

    fn run(self: Arc<Self>) {
        loop {
            let job = {
                let mut q = self.queue.lock().unwrap();
                loop {
                    if self.quit.load(Ordering::Relaxed) {
                        return;
                    }
                    if let Some(j) = q.pop_front() {
                        break j;
                    }
                    q = self.wake.wait(q).unwrap();
                }
            };
            match job {
                Job::Pixel { period, coord } => {
                    self.search_pixel(period, coord);
                    self.in_flight.lock().unwrap().remove(&period);
                }
                Job::Reference { center, periods, extend } => {
                    // Try every candidate period and keep the nucleus nearest the centre: the
                    // component the view is in, not a parent that also happens to be in range.
                    let mut found: Option<(f64, Reference)> = None;
                    for n in periods {
                        if self.quit.load(Ordering::Relaxed) {
                            return;
                        }
                        self.stats.nucleus_searches.fetch_add(1, Ordering::Relaxed);
                        if let Some(nuc) = nucleus_f7e3(center, n, &self.quit) {
                            let off: Complex64 = (nuc - center).into();
                            let dist = off.norm();
                            if dist < NUCLEUS_MAX_SCREENS * self.scale
                                && found.as_ref().map_or(true, |(d, _)| dist < *d)
                            {
                                if let Some(mut r) =
                                    Reference::from_nucleus(nuc, n, PerturbEngine::zero_tolerance(self.scale))
                                {
                                    r.built_scale = self.scale;
                                    self.stats.nucleus_hits.fetch_add(1, Ordering::Relaxed);
                                    found = Some((dist, r));
                                }
                            }
                        }
                    }
                    let mut found = found.map(|(_, r)| r);
                    if found.is_none() && extend {
                        if let Some(mut r) = Reference::from_point(center, REF_MAX_LEN, &self.quit) {
                            r.built_scale = self.scale;
                            found = Some(r);
                        }
                    }
                    if let Some(r) = found {
                        *self.upgrade.lock().unwrap() = Some(Arc::new(r));
                    }
                }
            }
        }
    }

    fn search_pixel(&self, n: u32, coord: CircleF7E3) {
        self.stats.nucleus_searches.fetch_add(1, Ordering::Relaxed);
        let near = NUCLEUS_MAX_SCREENS * self.scale;
        let found = nucleus_f7e3(coord, n, &self.quit).and_then(|nuc| {
            let off: Complex64 = (nuc - coord).into();
            if off.norm() >= near {
                return None;
            }
            Reference::from_nucleus(nuc, n, PerturbEngine::zero_tolerance(self.scale))
        });
        match found {
            Some(mut r) => {
                r.built_scale = self.scale;
                self.stats.nucleus_hits.fetch_add(1, Ordering::Relaxed);
                self.nuclei.lock().unwrap().push((n, Arc::new(r)));
            }
            None => {
                let coord64: Complex64 = coord.into();
                self.failed.lock().unwrap().entry(n).or_default().push(coord64);
            }
        }
    }
}

/// Outcome of asking for a per-period nucleus.
enum Lookup {
    Found(Arc<Reference>),
    Failed,
    Pending,
}

/// Perturbation engine for one viewport: the main reference, the zoom scale, and a background
/// worker that owns every F7E3 nucleus search.
pub struct PerturbEngine {
    pub reference: Arc<Reference>,
    pub stats: Arc<Stats>,
    /// `S`, the zoom magnitude.
    pub scale: f64,
    max_iterations: u64,
    max_iter32: u32,
    stop: Arc<AtomicBool>,
    worker: Arc<Worker>,
    /// Remaining Newton replay budget for the current step (see `NEWTON_STEP_BUDGET`).
    newton_budget: AtomicU64,
}

impl Drop for PerturbEngine {
    fn drop(&mut self) {
        self.worker.quit.store(true, Ordering::Relaxed);
        self.worker.wake.notify_all();
    }
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    period: u32,
    m: u32,
    dz: C32,
}

/// CPU-side companion of a `DeltaState`: what the driver needs that the kernel does not.
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub coord: CircleF7E3,
    /// Unscaled offset from the main reference parameter.
    pub dc: Complex64,
    pending: [u32; MAX_PENDING],
    n_pending: u8,
    budget_done: bool,
    /// Confirmed candidates waiting to be solved (in confirmation order), each with the
    /// iteration to solve it at.
    deferred: [(Candidate, u32); MAX_DEFERRED],
    n_deferred: u8,
}

impl Slot {
    fn push_pending(&mut self, period: u32) {
        let n = self.n_pending as usize;
        if n < MAX_PENDING && !self.pending[..n].contains(&period) {
            self.pending[n] = period;
            self.n_pending += 1;
        }
    }
    fn pending(&self) -> &[u32] {
        &self.pending[..self.n_pending as usize]
    }
    fn push_deferred(&mut self, cand: Candidate, at: u32) {
        let mut n = self.n_deferred as usize;
        if self.deferred[..n].iter().any(|(c, _)| c.period == cand.period) {
            return;
        }
        if n == MAX_DEFERRED {
            // Full: the oldest entries are the early-transient junk (periods 1, 2, …); a
            // later, larger candidate is the likelier true period and must not be dropped.
            self.deferred.copy_within(1..n, 0);
            n -= 1;
        }
        self.deferred[n] = (cand, at);
        self.n_deferred = n as u8 + 1;
    }
    /// The due candidate with the smallest period (multiples of the true period pass the
    /// solve too, so the smallest that passes is the period).
    fn pop_deferred_due(&mut self, iter: u32) -> Option<Candidate> {
        let n = self.n_deferred as usize;
        let mut best: Option<usize> = None;
        for i in 0..n {
            let (c, at) = self.deferred[i];
            if iter >= at && best.map_or(true, |b| c.period < self.deferred[b].0.period) {
                best = Some(i);
            }
        }
        let i = best?;
        let cand = self.deferred[i].0;
        self.deferred.copy_within(i + 1..n, i);
        self.n_deferred -= 1;
        Some(cand)
    }
    fn earliest_due(&self) -> Option<u32> {
        self.deferred[..self.n_deferred as usize].iter().map(|(_, at)| *at).min()
    }
}

/// Pixels in flight: parallel vectors so the states are one contiguous upload.
pub struct Batch<T> {
    pub tags: Vec<T>,
    pub slots: Vec<Slot>,
    pub states: Vec<DeltaState>,
}

impl<T> Default for Batch<T> {
    fn default() -> Self {
        Self { tags: Vec::new(), slots: Vec::new(), states: Vec::new() }
    }
}

impl<T> Batch<T> {
    pub fn len(&self) -> usize {
        self.states.len()
    }
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
    pub fn push(&mut self, tag: T, slot: Slot, state: DeltaState) {
        self.tags.push(tag);
        self.slots.push(slot);
        self.states.push(state);
    }
}

impl PerturbEngine {
    /// Build the engine for a viewport. `stop` aborts the build early.
    pub fn new(
        zoom: CircleF7E3,
        center: CircleF7E3,
        max_iterations: u64,
        stop: Arc<AtomicBool>,
    ) -> Option<Self> {
        Self::new_cached(zoom, center, max_iterations, stop, None)
    }

    /// Like `new`, but reuses `cached` when it is still a good reference for this viewport:
    /// within `NUCLEUS_MAX_SCREENS` of the centre, still exactly periodic at this zoom's
    /// tolerance, and (for a non-periodic reference) not zoomed in by more than 2⁸ since it
    /// was built, when a nucleus may have come into reach. Otherwise a plain orbit of the
    /// centre is built synchronously (capped at `REF_INITIAL_LEN`) and the nucleus hunt, or
    /// the extension to full length, is handed to the worker; `take_upgrade` delivers it.
    pub fn new_cached(
        zoom: CircleF7E3,
        center: CircleF7E3,
        max_iterations: u64,
        stop: Arc<AtomicBool>,
        cached: Option<Arc<Reference>>,
    ) -> Option<Self> {
        let zc: Complex64 = zoom.into();
        let scale = zc.norm();
        let reusable = cached.filter(|r| {
            let off: Complex64 = (center - r.c).into();
            let near = off.norm() < NUCLEUS_MAX_SCREENS * scale;
            let periodic_ok = r.residual < Self::zero_tolerance(scale);
            let shallow_enough = r.period.is_some() || scale * 256.0 >= r.built_scale;
            near && periodic_ok && shallow_enough
        });
        let (reference, job) = match reusable {
            Some(r) => (r, None),
            None => {
                let (mut r, periods) = Self::quick_reference(center, scale, &stop)?;
                r.built_scale = scale;
                let extend = r.period.is_none() && r.z64.len() >= REF_INITIAL_LEN;
                let job = (!periods.is_empty() || extend)
                    .then(|| Job::Reference { center, periods, extend });
                (Arc::new(r), job)
            }
        };
        let engine = Self::with_reference(reference, zoom, max_iterations, stop);
        if let Some(job) = job {
            engine.worker.enqueue(job);
        }
        Some(engine)
    }

    /// An engine on a given reference (no build), with its own worker.
    pub fn with_reference(
        reference: Arc<Reference>,
        zoom: CircleF7E3,
        max_iterations: u64,
        stop: Arc<AtomicBool>,
    ) -> Self {
        let zc: Complex64 = zoom.into();
        let scale = zc.norm();
        let stats = Arc::new(Stats::default());
        let worker = Arc::new(Worker {
            queue: Mutex::new(VecDeque::new()),
            wake: Condvar::new(),
            quit: AtomicBool::new(false),
            scale,
            stats: Arc::clone(&stats),
            nuclei: Mutex::new(Vec::new()),
            failed: Mutex::new(HashMap::new()),
            in_flight: Mutex::new(HashSet::new()),
            upgrade: Mutex::new(None),
        });
        {
            let w = Arc::clone(&worker);
            std::thread::spawn(move || w.run());
        }
        Self {
            reference,
            stats,
            scale,
            max_iterations,
            max_iter32: max_iterations.min(u32::MAX as u64) as u32,
            stop,
            worker,
            newton_budget: AtomicU64::new(u64::MAX),
        }
    }

    /// A better reference the worker found for this viewport, if any. The caller rebuilds the
    /// engine on it with `with_reference`.
    pub fn take_upgrade(&self) -> Option<Arc<Reference>> {
        self.worker.upgrade.lock().unwrap().take()
    }

    /// Tolerance below which an orbit value counts as an exact return to zero: 2⁻¹⁰ pixel.
    fn zero_tolerance(scale: f64) -> f64 {
        scale * 9.765625e-4
    }

    /// The synchronous stage of a reference: the centre's plain orbit up to `REF_INITIAL_LEN`
    /// (an exact zero on the way makes the centre a nucleus and the orbit periodic), plus the
    /// candidate periods — new 4× minima — for the worker's nucleus hunt.
    fn quick_reference(
        center: CircleF7E3,
        scale: f64,
        stop: &AtomicBool,
    ) -> Option<(Reference, Vec<u32>)> {
        let tol = Self::zero_tolerance(scale);
        let four = ScalarF7E3::from(4.0);
        let mut z64 = Vec::with_capacity(1 << 16);
        let mut z = CircleF7E3::ZERO;
        z64.push(Complex64::new(0., 0.));
        let mut z_min = f64::MAX;
        let mut periods = Vec::new();
        loop {
            let iter = z64.len() as u32;
            z = z.square() + center;
            if z == CircleF7E3::ZERO {
                if let Some(r) = Reference::from_nucleus(center, iter, tol) {
                    return Some((r, Vec::new()));
                }
            }
            let zc: Complex64 = z.into();
            z64.push(zc);
            if !finite(z) || z.magnitude_squared() > four || z64.len() >= REF_INITIAL_LEN {
                break;
            }
            let mag_sq = zc.norm_sqr();
            if z_min > mag_sq {
                let ratio = mag_sq / z_min;
                z_min = mag_sq;
                if ratio < 0.25 && iter <= NUCLEUS_SEARCH_MAX {
                    periods.push(iter);
                }
            }
            if iter & 0xFFFF == 0 && stop.load(Ordering::Relaxed) {
                return None;
            }
        }
        let z32 = z64.iter().map(|v| C32::new(v.re as f32, v.im as f32)).collect();
        let r = Reference {
            c: center,
            c64: center.into(),
            z64,
            z32,
            period: None,
            residual: f64::INFINITY,
            built_scale: scale,
        };
        Some((r, periods))
    }

    /// A fresh slot + state for a pixel.
    pub fn slot(&self, coord: CircleF7E3) -> (Slot, DeltaState) {
        let r = &*self.reference;
        let dc: Complex64 = (coord - r.c).into();
        let dcs = dc / self.scale;
        let dcs32 = C32::new(dcs.re as f32, dcs.im as f32);
        let slot = Slot {
            coord,
            dc,
            pending: [0; MAX_PENDING],
            n_pending: 0,
            budget_done: false,
            deferred: [(Candidate { period: 0, m: 0, dz: C32::new(0., 0.) }, 0); MAX_DEFERRED],
            n_deferred: 0,
        };
        (slot, DeltaState::new(dcs32, self.scale as f32))
    }

    /// Set the state's iteration limit for the next run: at most `chunk` more iterations, and
    /// never past the next phase boundary (nucleus budget, iteration cap).
    pub fn arm(&self, slot: &Slot, st: &mut DeltaState, chunk: u32) {
        let phase = if !slot.budget_done && NUCLEUS_BUDGET < self.max_iter32 {
            NUCLEUS_BUDGET
        } else {
            self.max_iter32
        };
        let mut limit = st.iter.saturating_add(chunk).min(phase);
        if let Some(at) = slot.earliest_due() {
            limit = limit.min(at.max(st.iter.saturating_add(1)));
        }
        st.limit = limit;
    }

    /// Consume the status a run left in `st`. `Some` when the pixel is finished; a pixel
    /// waiting on the worker stays in the batch at its phase limit.
    pub fn drive(&self, slot: &mut Slot, st: &mut DeltaState) -> Option<MandelbrotResult> {
        match st.status {
            STATUS_ESCAPED => {
                let t = std::time::Instant::now();
                let r = self.finish_escape(*st, st.z_out, slot.coord);
                self.stats.ns_escape.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                self.stats.escapes.fetch_add(1, Ordering::Relaxed);
                Some(r)
            }
            STATUS_DEAD => Some(self.finish_max_iterations(*st)),
            STATUS_CONFIRMED => {
                st.status = STATUS_RUNNING;
                let cand = Candidate { period: st.period, m: st.m, dz: st.dz };
                // Not solved here: the solve is deferred until the pixel has iterated on, so
                // an exterior pixel that fooled the probe can escape first. Confirmations are
                // queued in order (the direct pipelines' "first success wins").
                // A candidate the probe confirmed only by running out of windows is weak:
                // give the pixel longer to escape before paying for its solve.
                let weak = st.probe_windows >= PROBE_WINDOWS;
                let wait = if weak { (8 * cand.period).max(16 * NEWTON_DEFER_MIN) } else { (2 * cand.period).max(4 * NEWTON_DEFER_MIN) };
                let at = st.iter.saturating_add(wait);
                slot.push_deferred(cand, at);
                // Let a blocked divisor-multiple be probed meanwhile, as a failed immediate
                // solve used to arrange; a smaller true period then queues up too.
                st.probe_failed();
                None
            }
            _ => {
                while slot.n_deferred > 0 {
                    let due_iter = if st.iter >= self.max_iter32 { u32::MAX } else { st.iter };
                    let Some(next) = slot.earliest_due() else { break };
                    if due_iter < next {
                        return None; // not due yet
                    }
                    let period = slot.deferred[..slot.n_deferred as usize]
                        .iter()
                        .filter(|(_, at)| due_iter >= *at)
                        .map(|(c, _)| c.period)
                        .min()
                        .unwrap_or(0);
                    let cost = 8 * period as u64;
                    let within_budget = self
                        .newton_budget
                        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |b| b.checked_sub(cost))
                        .is_ok();
                    if !within_budget {
                        return None; // over budget this step: next step
                    }
                    let Some(cand) = slot.pop_deferred_due(due_iter) else { break };
                    let t = std::time::Instant::now();
                    let res = self.solve_candidate(cand, slot, st);
                    self.stats.ns_newton.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    if let Some(res) = res {
                        return Some(res);
                    }
                }
                if st.iter >= self.max_iter32 {
                    return match self.try_nuclei(slot.pending(), slot.coord) {
                        Some(Some(res)) => Some(res),
                        Some(None) => Some(self.finish_max_iterations(*st)),
                        None => None, // waiting on the worker
                    };
                }
                if !slot.budget_done && st.iter >= NUCLEUS_BUDGET {
                    match self.try_nuclei(slot.pending(), slot.coord) {
                        Some(Some(res)) => return Some(res),
                        Some(None) => {
                            slot.budget_done = true;
                            slot.n_pending = 0;
                        }
                        None => {} // waiting: the limit stays at the budget
                    }
                }
                None
            }
        }
    }

    /// Solve a confirmed candidate: exact delta-frame Newton when the reference period divides
    /// it or the zoom is shallow; a direct F7E3 solve (gated by an f64 pre-check) for small
    /// periods deeper than that; otherwise the f64 verdict, with the period parked for the
    /// worker's nucleus search on failure.
    fn solve_candidate(&self, cand: Candidate, slot: &mut Slot, st: &mut DeltaState) -> Option<MandelbrotResult> {
        let exact = self.reference.period.is_some_and(|p| cand.period % p == 0);
        if exact || self.scale > F64_TRUST_SCALE {
            if let Some(res) = self.newton_main(cand, slot.dc) {
                return Some(res);
            }
            st.probe_failed();
            return None;
        }
        if cand.period <= F7E3_NEWTON_MAX_PERIOD {
            if self.newton_precheck(cand, slot.dc) {
                if let Some(res) = self.newton_f7e3(cand, slot.coord) {
                    return Some(res);
                }
            }
            st.probe_failed();
            return None;
        }
        if let Some(res) = self.newton_main(cand, slot.dc) {
            return Some(res);
        }
        st.probe_failed();
        slot.push_pending(cand.period);
        None
    }

    /// Synchronous single pixel on the CPU: F7E3 coordinate in, f64-domain result out. The app
    /// goes through `step`; this is the tests' and the bench's entry point.
    #[allow(dead_code)]
    pub fn point(&self, coord: CircleF7E3) -> MandelbrotResult {
        let (mut slot, mut st) = self.slot(coord);
        let s32 = self.scale as f32;
        self.newton_budget.store(u64::MAX, Ordering::Relaxed);
        loop {
            self.arm(&slot, &mut st, u32::MAX);
            run_f32(&mut st, &self.reference, s32);
            if let Some(res) = self.drive(&mut slot, &mut st) {
                return res;
            }
            if st.iter >= st.limit && st.status == STATUS_RUNNING {
                // Parked on the worker: wait for it rather than spin.
                std::thread::sleep(std::time::Duration::from_millis(1));
                if self.stop.load(Ordering::Relaxed) {
                    return self.finish_max_iterations(st);
                }
            }
        }
    }

    /// One chunk of work on every pixel in the batch, then the driver on each. Finished
    /// pixels are removed and returned with their tags.
    pub fn step<T>(
        &self,
        batch: &mut Batch<T>,
        runner: &mut dyn DeltaRunner,
        chunk: u32,
    ) -> Vec<(T, MandelbrotResult)> {
        let s32 = self.scale as f32;
        for (slot, st) in batch.slots.iter().zip(batch.states.iter_mut()) {
            self.arm(slot, st, chunk);
        }
        let t = std::time::Instant::now();
        runner.run(&self.reference, s32, &mut batch.states);
        self.stats.ns_runner.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.newton_budget.store(NEWTON_STEP_BUDGET, Ordering::Relaxed);
        let t = std::time::Instant::now();
        let mut finished: Vec<Option<MandelbrotResult>> = batch
            .slots
            .par_iter_mut()
            .zip(batch.states.par_iter_mut())
            .map(|(slot, st)| self.drive(slot, st))
            .collect();
        self.stats.ns_drive.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let mut out = Vec::new();
        for i in (0..finished.len()).rev() {
            if let Some(res) = finished[i].take() {
                batch.slots.swap_remove(i);
                batch.states.swap_remove(i);
                out.push((batch.tags.swap_remove(i), res));
            }
        }
        out
    }

    /// Delta-frame Newton against the main reference for a trigger candidate.
    fn newton_main(&self, cand: Candidate, dc: Complex64) -> Option<MandelbrotResult> {
        let r = &*self.reference;
        let dw = Complex64::new(cand.dz.re as f64, cand.dz.im as f64) * self.scale;
        self.stats.newton_solves.fetch_add(1, Ordering::Relaxed);
        let (w, du, period) = newton_period(r, dc, cand.period, cand.m as usize, dw)?;
        self.stats.newton_hits.fetch_add(1, Ordering::Relaxed);
        let diff = refine_nucleus(r, dc, period);
        Some(self.inside_result(period, w, du, diff))
    }

    /// Inside/outside only, from the delta-frame solve.
    fn newton_precheck(&self, cand: Candidate, dc: Complex64) -> bool {
        let dw = Complex64::new(cand.dz.re as f64, cand.dz.im as f64) * self.scale;
        self.stats.newton_solves.fetch_add(1, Ordering::Relaxed);
        newton_period(&self.reference, dc, cand.period, cand.m as usize, dw).is_some()
    }

    /// The grid's period Newton and root refinement, directly in F7E3 from the pixel's exact
    /// coordinate: used where the delta frame cannot be exact and f64 would leave the interior
    /// shading with too few bits. Starts from the orbit point at the trigger.
    fn newton_f7e3(&self, cand: Candidate, coord: CircleF7E3) -> Option<MandelbrotResult> {
        type S = ScalarF7E3;
        type C = CircleF7E3;
        self.stats.f7e3_solves.fetch_add(1, Ordering::Relaxed);
        let n = cand.period;
        let z0 = self.reference.z64[cand.m as usize]
            + Complex64::new(cand.dz.re as f64, cand.dz.im as f64) * self.scale;
        let mut w = C::from((S::from(z0.re), S::from(z0.im)));
        let c = coord;
        let eps_sq = S::TWO.pow(-64);
        for _ in 0..MAX_NEWTON_STEPS {
            let mut u = w;
            let mut du = C::ONE;
            for _ in 0..n {
                let p = du * u;
                du = p + p;
                u = u.square() + c;
                if !finite(u) || !finite(du) {
                    return None;
                }
            }
            let denom = du - C::ONE;
            if !denom.is_normal() {
                return None;
            }
            let newton = (u - w) / denom;
            if !finite(newton) {
                return None;
            }
            let step_sq = newton.magnitude_squared();
            w = w - newton;
            // A vanished square (below the type's range) is converged too.
            if !step_sq.is_normal() || step_sq < eps_sq {
                let du_mag_sq = du.magnitude_squared();
                if !(du_mag_sq.is_normal() || du_mag_sq == S::ZERO) || !(du_mag_sq < S::ONE) {
                    return None;
                }
                // Minimal period: smallest divisor after which w returns to itself.
                let mut n = n;
                let mut du = du;
                {
                    let tol = S::TWO.pow(-40);
                    let mut dd = 1u32;
                    while dd < n {
                        if n % dd == 0 {
                            let mut u = w;
                            let mut mult = C::ONE;
                            for _ in 0..dd {
                                let p = mult * u;
                                mult = p + p;
                                u = u.square() + c;
                            }
                            let back = (u - w).magnitude_squared();
                            if !back.is_normal() || back < tol {
                                n = dd;
                                du = mult;
                                break;
                            }
                        }
                        dd += 1;
                    }
                }
                // Interior root refinement: the nucleus of period n from the pixel.
                let mut cr = c;
                for _ in 0..MAX_NEWTON_STEPS {
                    let mut z = cr;
                    let mut d = C::ONE;
                    for _ in 1..n {
                        let dz = d * z;
                        d = dz + dz + C::ONE;
                        z = z.square() + cr;
                    }
                    if !d.is_normal() {
                        break;
                    }
                    let next = cr - z / d;
                    if !finite(next) || next == cr {
                        break;
                    }
                    cr = next;
                }
                let diff: Complex64 = (c - cr).into();
                let w64: Complex64 = w.into();
                let du64: Complex64 = du.into();
                return Some(self.inside_result(n, w64, du64, diff));
            }
            if !finite(w) || w.magnitude_squared() > S::from(16.0) {
                return None;
            }
        }
        None
    }

    /// Parked foreign-period candidates: a nucleus of that period near the pixel (from the
    /// worker), the pixel's orbit replayed against it in f64, and the solve there.
    /// `None`: still waiting on the worker. `Some(None)`: nothing resolved.
    fn try_nuclei(&self, pending: &[u32], coord: CircleF7E3) -> Option<Option<MandelbrotResult>> {
        for &n in pending {
            let nref = match self.nucleus_for(n, coord) {
                Lookup::Found(r) => r,
                Lookup::Failed => continue,
                Lookup::Pending => return None,
            };
            let dc: Complex64 = (coord - nref.c).into();
            let mut mu = 1usize;
            let mut dw = dc;
            for _ in 1..n {
                step64(&nref, &mut mu, &mut dw, dc);
            }
            if let Some((w, du, period)) = newton_period(&nref, dc, n, mu, dw) {
                let diff = refine_nucleus(&nref, dc, period);
                return Some(Some(self.inside_result(period, w, du, diff)));
            }
        }
        Some(None)
    }

    /// Cached nucleus of period `n` near `coord`, a cached failure, or a request to the worker.
    fn nucleus_for(&self, n: u32, coord: CircleF7E3) -> Lookup {
        if n > NUCLEUS_SEARCH_MAX {
            return Lookup::Failed;
        }
        let near = NUCLEUS_MAX_SCREENS * self.scale;
        {
            let cache = self.worker.nuclei.lock().unwrap();
            for (p, r) in cache.iter() {
                if *p == n {
                    let off: Complex64 = (coord - r.c).into();
                    if off.norm() < near {
                        return Lookup::Found(Arc::clone(r));
                    }
                }
            }
        }
        let coord64: Complex64 = coord.into();
        {
            let failed = self.worker.failed.lock().unwrap();
            if let Some(list) = failed.get(&n) {
                if list.iter().any(|f| (coord64 - f).norm() < near) {
                    return Lookup::Failed;
                }
            }
        }
        let mut in_flight = self.worker.in_flight.lock().unwrap();
        if in_flight.insert(n) {
            self.worker.enqueue(Job::Pixel { period: n, coord });
        }
        Lookup::Pending
    }

    fn inside_result(&self, period: u32, w: Complex64, du: Complex64, diff: Complex64) -> MandelbrotResult {
        MandelbrotResult {
            period: period as u64,
            escape_velocity: self.max_iterations as f64,
            coords: du,
            flame_q: w,
            flame_c: w,
            slope: Complex64::new(diff.re, -diff.im),
            is_primitive: true,
        }
    }

    /// Pixel ran to the iteration cap without escaping or resolving: the direct pipelines'
    /// "not converged" tail (normalized sums, zero slope).
    fn finish_max_iterations(&self, st: DeltaState) -> MandelbrotResult {
        let c = |v: C32| Complex64::new(v.re as f64, v.im as f64);
        MandelbrotResult {
            period: 0,
            escape_velocity: self.max_iterations as f64,
            coords: c(st.coord_acc / st.dist_sum),
            flame_q: c(st.flame_u / st.energy_u),
            flame_c: c(st.flame_w / st.energy_w),
            slope: Complex64::new(0., 0.),
            is_primitive: true,
        }
    }

    /// The escape ride and finalization. Once |z|² > 4 the offset no longer matters: the pixel
    /// is converted to F7E7 (128-bit fraction, 128-bit exponent) and z is iterated there,
    /// exactly, until it leaves the type near |z| = 2^(2^127). Each step's unit vector and
    /// lb |z|² are read straight off z's fraction and exponent fields into f64, where the
    /// grid's accumulators are kept: fade = (H − lb lb |z|²)/(H − 1) with H = lb lb MAX = 127,
    /// so the ride weighs in from 1 at escape to 0 at the horizon. The slope ln|z|²·z̄/ō is
    /// invariant per squaring once |z| is large, so it is taken in f64 at the first step past
    /// |z|² = 2⁶⁴ and the orbit tracker stops there.
    fn finish_escape(&self, st: DeltaState, z0: C32, coord: CircleF7E3) -> MandelbrotResult {
        type S7 = ScalarF7E7;
        type C7 = CircleF7E7;
        const HORIZON: f64 = 127.0;
        const SLOPE_HANDOFF_LB: f64 = 64.0;

        let point = C7::from((S7::from(coord.r()), S7::from(coord.i())));
        let mut z = C7::from((S7::from(z0.re as f64), S7::from(z0.im as f64)));
        // The tracker is S·dz/dc in the delta loop; un-scale it in a type with the range.
        let mut orbit = C7::from((S7::from(st.orbit.re as f64), S7::from(st.orbit.im as f64)))
            / S7::from(self.scale);
        let mut iter = st.iter as u64;

        let c64 = |v: C32| Complex64::new(v.re as f64, v.im as f64);
        let mut coord_acc = c64(st.coord_acc);
        let mut dist_sum = st.dist_sum as f64;
        let mut flame_uniform = c64(st.flame_u);
        let mut energy_sum_uniform = st.energy_u as f64;
        let mut flame_weighted = c64(st.flame_w);
        let mut energy_sum_weighted = st.energy_w as f64;

        // Direction (a complex number of magnitude in [1, 2]) and lb |z|² straight from the
        // fields: the fractions are plain i128 scaled by 2⁻¹²⁶ and the exponent is offset by
        // 2¹²⁷ (checked against spirix's own conversion).
        const FRACTION_SCALE: f64 = 1.1754943508222875e-38; // 2^-126
        let decompose = |z: &C7| -> (Complex64, f64) {
            let dir = Complex64::new(z.real as f64 * FRACTION_SCALE, z.imaginary as f64 * FRACTION_SCALE);
            let exp = z.exponent.wrapping_sub(i128::MIN) as f64;
            (dir, 2.0 * (exp + dir.norm().log2()))
        };

        let mut lb_mag_sq = decompose(&z).1;
        let mut slope: Option<Complex64> = None;
        // Past |z|² = 2¹²⁸ the added c is below F7's precision and the F7E7 ride is pure
        // squaring, so the direction can be angle-doubled in f64 at a tenth of the cost. The
        // escape velocity, the slope and every fade weight are identical either way; only the
        // late, already-fading unit vectors differ (both are chaotic between neighbouring
        // pixels). MANDEL_EXACT_TAIL=1 keeps squaring in F7E7 to the horizon.
        const PURE_SQUARING_LB: f64 = 128.0;
        const HORIZON_LB: f64 = 3.402823669209385e38; // 2^128: lb |z|² at the F7E7 horizon
        let exact_tail = exact_tail();
        let mut tail: Option<(Complex64, f64)> = None; // (unit vector, lb |z|²) once in f64
        loop {
            if let Some((u, lb)) = tail {
                let lb_next = lb * 2.0;
                if lb_next >= HORIZON_LB {
                    break;
                }
                let u2 = u * u;
                let u2 = u2 / u2.norm();
                tail = Some((u2, lb_next));
                lb_mag_sq = lb_next;
                iter += 1;
                let fade = ((HORIZON - lb_next.log2()) / (HORIZON - 1.0)).max(0.0);
                let iter_sq = (iter * iter) as f64;
                energy_sum_uniform += fade;
                flame_uniform += u2 * fade;
                let taper_weighted = fade * iter_sq;
                energy_sum_weighted += taper_weighted;
                flame_weighted += u2 * taper_weighted;
                continue;
            }
            let z_next = z.square() + point;
            if !z_next.is_normal() {
                break;
            }
            if slope.is_none() {
                // Derivative recurrence with the orbit point that produced z_next.
                let orbit_re = z.r() * orbit.r() - z.i() * orbit.i();
                let orbit_im = orbit.r() * z.i() + orbit.i() * z.r();
                orbit = C7::from((orbit_re + orbit_re + S7::ONE, orbit_im + orbit_im));
            }
            z = z_next;
            iter += 1;
            let (dir, lb) = decompose(&z);
            lb_mag_sq = lb;
            let unit = dir / dir.norm();
            let fade = ((HORIZON - lb.log2()) / (HORIZON - 1.0)).max(0.0);
            let iter_sq = (iter * iter) as f64;

            energy_sum_uniform += fade;
            flame_uniform += unit * fade;
            let taper_weighted = fade * iter_sq;
            energy_sum_weighted += taper_weighted;
            flame_weighted += unit * taper_weighted;
            // iter²/|z| and iter²/|z|² underflow to exactly zero past f64's range, which is
            // where they are far below anything the earlier terms left anyway.
            if lb < 2200.0 {
                let inv_mag = (-0.5 * lb).exp2();
                dist_sum += iter_sq * inv_mag * inv_mag;
                coord_acc += unit * (iter_sq * inv_mag);
            }

            if slope.is_none() && lb > SLOPE_HANDOFF_LB {
                let zc: Complex64 = z.into();
                let oc: Complex64 = orbit.into();
                slope = Some(lb * std::f64::consts::LN_2 * zc.conj() / oc.conj());
            }
            if !exact_tail && lb > PURE_SQUARING_LB && slope.is_some() {
                tail = Some((unit, lb));
            }
        }

        let escape_velocity = iter as f64 + 1.0 - lb_mag_sq.log2();
        let slope = slope.unwrap_or_else(|| {
            // The ride ended before the hand-off (it cannot, short of a non-finite S).
            let zc: Complex64 = z.into();
            let oc: Complex64 = orbit.into();
            lb_mag_sq * std::f64::consts::LN_2 * zc.conj() / oc.conj()
        });
        MandelbrotResult {
            period: 0,
            escape_velocity,
            coords: coord_acc / dist_sum,
            flame_q: flame_uniform / energy_sum_uniform,
            flame_c: flame_weighted / energy_sum_weighted,
            slope,
            is_primitive: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::calculate_mandelbrot_point_f32;

    fn c7(re: f64, im: f64) -> CircleF7E3 {
        CircleF7E3::from((ScalarF7E3::from(re), ScalarF7E3::from(im)))
    }

    /// An engine for the view, settled: the worker's nucleus hunt (or orbit extension) is
    /// given up to five seconds to deliver its upgrade, which is then built on.
    fn engine(center: Complex64, scale: f64) -> PerturbEngine {
        let zoom = c7(scale, 0.);
        let stop = Arc::new(AtomicBool::new(false));
        let eng = PerturbEngine::new(zoom, c7(center.re, center.im), 1 << 16, Arc::clone(&stop))
            .expect("engine build");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            if let Some(up) = eng.take_upgrade() {
                return PerturbEngine::with_reference(up, zoom, 1 << 16, stop);
            }
            if eng.worker.queue.lock().unwrap().is_empty() && eng.worker.in_flight.lock().unwrap().is_empty() {
                // Nothing queued: give an in-progress job a moment, then accept the reference.
                std::thread::sleep(std::time::Duration::from_millis(20));
                if let Some(up) = eng.take_upgrade() {
                    return PerturbEngine::with_reference(up, zoom, 1 << 16, stop);
                }
                if eng.reference.period.is_some() || eng.reference.z64.len() < REF_INITIAL_LEN {
                    return eng;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        eng
    }

    /// At zoom 1 the delta frame is just f32 arithmetic in a different order: periods must
    /// match the direct f32 pipeline and escape velocities must be close on clearly escaping
    /// points; interior points must agree on period and multiplier.
    #[test]
    fn shallow_agrees_with_direct_f32() {
        let eng = engine(Complex64::new(-0.5, 0.0), 1.0);
        let escaping = [(0.5, 0.5), (-2.5, 0.1), (0.0, 1.5), (0.4, 0.4), (-1.5, 0.5)];
        for &(re, im) in &escaping {
            let p = eng.point(c7(re, im));
            let d = calculate_mandelbrot_point_f32(Complex64::new(re, im), (1u64 << 16) as f64, false);
            assert_eq!(p.period, 0, "({re},{im}) perturbed misdetected as interior");
            assert_eq!(d.period, 0);
            assert!(
                (p.escape_velocity - d.escape_velocity).abs() < 1e-3 * d.escape_velocity.max(1.),
                "({re},{im}) ev perturbed {} vs direct {}",
                p.escape_velocity, d.escape_velocity
            );
            assert!(p.slope.re.is_finite() && p.slope.norm_sqr() > 0., "({re},{im}) slope {}", p.slope);
            let ratio = p.slope.norm() / d.slope.norm();
            assert!(ratio > 0.9 && ratio < 1.1, "({re},{im}) |slope| perturbed {} vs direct {}", p.slope.norm(), d.slope.norm());
            // Escaping flames are not compared against any other pipeline: past a few dozen
            // squarings the ride's unit vectors are angle-doubled into chaos, so pipelines of
            // different precision legitimately disagree on them. They only have to be finite.
            assert!(p.flame_c.re.is_finite() && p.flame_q.re.is_finite(), "({re},{im}) flames {} {}", p.flame_c, p.flame_q);
        }
        // Interior values against the F7E3 grid cell: the direct f32 pipeline's Newton
        // threshold (2⁻¹²) is loose enough to report the multiplier from an unconverged step.
        let oracle = crate::compute::spirix_point_fn(7, 3);
        // (point, period, nucleus of its component). The grid's interior slope is NaN in every
        // E3 cell (its 2⁻²⁵⁶ refinement nudge is unrepresentable there), so slope is checked
        // against the known nucleus: slope = conj(c − nucleus).
        let interior = [
            (-0.1, 0.0, 1u64, Complex64::new(0., 0.)),
            (-1.0, 0.0, 2, Complex64::new(-1., 0.)),
            (-0.12, 0.75, 3, Complex64::new(-0.12256116687665362, 0.7448617666197442)),
            (0.2, 0.0, 1, Complex64::new(0., 0.)),
        ];
        for &(re, im, want, nucleus) in &interior {
            let p = eng.point(c7(re, im));
            let d = oracle(c7(re, im), 1 << 16);
            assert_eq!(p.period, d.period, "({re},{im}) period perturbed {} vs F7E3 {}", p.period, d.period);
            assert!(p.period % want == 0, "({re},{im}) period {} not a multiple of {want}", p.period);
            assert!((p.coords - d.coords).norm() < 1e-3, "({re},{im}) multiplier {} vs {}", p.coords, d.coords);
            assert!((p.flame_c - d.flame_c).norm() < 1e-3, "({re},{im}) w {} vs {}", p.flame_c, d.flame_c);
            let expect = (Complex64::new(re, im) - nucleus).conj();
            assert!((p.slope - expect).norm() < 1e-9, "({re},{im}) slope {} expected {}", p.slope, expect);
        }
    }

    /// Centred on the period-3 nucleus at zoom 2⁻³⁰, every pixel is deep inside that
    /// minibrot's cardioid: the reference must be the exact nucleus, every pixel period 3, and
    /// the interior slope must equal the pixel's offset from the nucleus — a delta-scale quantity
    /// that plain f32 (or the direct f64 pipeline past 2⁻⁵³) cannot produce.
    #[test]
    fn deep_interior_slope_is_the_nucleus_offset() {
        let nucleus = Complex64::new(-1.7548776662466927, 0.0);
        let scale = 2f64.powi(-30);
        let eng = engine(nucleus, scale);
        assert_eq!(eng.reference.period, Some(3), "reference is not the period-3 nucleus");
        let nuc_c: Complex64 = eng.reference.c64;
        assert!((nuc_c - nucleus).norm() < 1e-12);
        for &(x, y) in &[(0.3, 0.1), (-0.7, 0.4), (0.0, -0.9), (1.5, 1.5)] {
            let coord = eng.reference.c + c7(x * scale, y * scale);
            let p = eng.point(coord);
            assert_eq!(p.period, 3, "pixel ({x},{y}) period {}", p.period);
            let expect = Complex64::new(x * scale, -y * scale);
            let err = (p.slope - expect).norm() / scale;
            assert!(err < 1e-4, "pixel ({x},{y}) slope {} expected {} (err {err} px)", p.slope, expect);
            assert!(p.coords.norm() < 1., "multiplier {} not attracting", p.coords);
        }
    }

    /// Two engines whose references differ (centre vs a point a third of a screen away) must
    /// agree on the same pixels: escaping and interior.
    #[test]
    fn results_do_not_depend_on_the_reference() {
        let scale = 2f64.powi(-20);
        let (cx, cy) = (-0.16, 1.04);
        let a = engine(Complex64::new(cx, cy), scale);
        let b = engine(Complex64::new(cx + 0.3 * scale, cy - 0.2 * scale), scale);
        let mut compared = 0;
        for i in 0..6 {
            for j in 0..6 {
                let (x, y) = (i as f64 * 0.17 - 0.4, j as f64 * 0.13 - 0.3);
                let coord = c7(cx + x * scale, cy + y * scale);
                let pa = a.point(coord);
                let pb = b.point(coord);
                assert_eq!(pa.period, pb.period, "pixel ({x},{y}) period {} vs {}", pa.period, pb.period);
                if pa.period == 0 && pa.escape_velocity < 1000. {
                    let tol = 1e-2 * pa.escape_velocity;
                    assert!(
                        (pa.escape_velocity - pb.escape_velocity).abs() < tol,
                        "pixel ({x},{y}) ev {} vs {}",
                        pa.escape_velocity, pb.escape_velocity
                    );
                    compared += 1;
                }
            }
        }
        assert!(compared > 0);
    }

    /// Every pixel of a bulb interior must be detected with its exact period, as the direct
    /// pipeline does (period-2 bulb, 1/3 bulb, main cardioid).
    #[test]
    fn bulb_interiors_are_detected_with_their_period() {
        let eng = engine(Complex64::new(-0.5, 0.0), 1.0);
        for (cx, cy, r, p) in [(-1.0, 0.0, 0.22, 2u64), (-0.1226, 0.7449, 0.08, 3), (0.2, 0.0, 0.04, 1)] {
            let mut periods = std::collections::BTreeMap::new();
            for i in 0..12 {
                for j in 0..12 {
                    let a = i as f64 / 12. * std::f64::consts::TAU;
                    let rr = r * (0.2 + 0.78 * j as f64 / 11.);
                    let res = eng.point(c7(cx + rr * a.cos(), cy + rr * a.sin()));
                    *periods.entry(res.period).or_insert(0) += 1;
                }
            }
            assert_eq!(periods.get(&p).copied().unwrap_or(0), 144, "bulb {p}: periods seen {periods:?}");
        }
    }

    /// Diagnostic: what the reference hunt sees at a view centred on a minibrot.
    #[test]
    #[ignore]
    fn debug_minibrot_reference() {
        let center = c7(-0.74249250281411, 0.20899915304036837);
        let scale = 2f64.powf(-18.714);
        let stop = AtomicBool::new(false);
        let (r, periods) = PerturbEngine::quick_reference(center, scale, &stop).unwrap();
        println!("quick reference: {} entries, period {:?}; candidates {:?}", r.z64.len(), r.period, periods);
        for &n in &periods {
            let t = std::time::Instant::now();
            match nucleus_f7e3(center, n, &stop) {
                Some(nuc) => {
                    let off: Complex64 = (nuc - center).into();
                    let built = Reference::from_nucleus(nuc, n, PerturbEngine::zero_tolerance(scale));
                    let mut z = CircleF7E3::ZERO;
                    for _ in 0..n { z = z.square() + nuc; }
                    let zn: Complex64 = z.into();
                    println!("  n={n}: nucleus off {:.3e} ({:.1} screens), |z_n|={:.3e} tol {:.3e}, periodic ref: {:?}, {:.0} ms",
                        off.norm(), off.norm() / scale, zn.norm(), PerturbEngine::zero_tolerance(scale),
                        built.map(|b| b.period), t.elapsed().as_secs_f64() * 1e3);
                }
                None => println!("  n={n}: newton failed"),
            }
        }
    }

    /// Diagnostic: F7E3 nucleus Newton trace for period 64 at the minibrot view, and a scan of
    /// interior pixels for NaN results.
    #[test]
    #[ignore]
    fn debug_minibrot_newton() {
        let center = c7(-0.74249250281411, 0.20899915304036837);
        let n = 64u32;
        let mut c = center;
        for step in 0..8 {
            let mut z = c;
            let mut d = CircleF7E3::ONE;
            let mut bad = None;
            for k in 1..n {
                // Doubled by addition: spirix's `Circle << 1` overflows for some operands.
            let dz = d * z;
            d = dz + dz + CircleF7E3::ONE;
                z = z.square() + c;
                if !finite(z) || !finite(d) {
                    bad = Some(k);
                    break;
                }
            }
            let zc: Complex64 = z.into();
            let dc: Complex64 = d.into();
            println!("newton step {step}: |z_n|={:.3e} |d_n|={:.3e} bad_at={bad:?} d normal={} ", zc.norm(), dc.norm(), d.is_normal());
            if bad.is_some() { break; }
            let next = c - z / d;
            let off: Complex64 = (next - c).into();
            println!("   step size {:.3e}, next finite {}", off.norm(), finite(next));
            if next == c { break; }
            c = next;
        }
        let scale = 2f64.powf(-18.714);
        let eng = engine(Complex64::new(-0.74249250281411, 0.20899915304036837), scale);
        println!("engine reference: {} entries period {:?}", eng.reference.z64.len(), eng.reference.period);
        let mut nan = 0; let mut inside = 0; let mut periods = std::collections::BTreeMap::new();
        for i in 0..20 { for j in 0..20 {
            let (x, y) = (i as f64 * 0.01 - 0.1, j as f64 * 0.01 - 0.15);
            let r = eng.point(eng.reference.c + c7(x * scale, y * scale));
            if r.period != 0 { inside += 1; *periods.entry(r.period).or_insert(0) += 1; if !r.coords.re.is_finite() || !r.flame_c.re.is_finite() || !r.slope.re.is_finite() { nan += 1; if nan < 4 { println!("  NaN at ({x:.2},{y:.2}): period {} coords {} flame {} slope {}", r.period, r.coords, r.flame_c, r.slope); } } }
        }}
        println!("interior {inside}/400, NaN {nan}, periods {periods:?}");
    }

    /// Diagnostic: bulb pixels at the minibrot view, and slope (distance estimate) against f64.
    #[test]
    #[ignore]
    fn debug_bulbs_and_slope() {
        let scale = 2f64.powf(-18.714);
        let eng = engine(Complex64::new(-0.74249250281411, 0.20899915304036837), scale);
        println!("reference period {:?}", eng.reference.period);
        let (w, h) = (2048.0, 1024.0);
        let diag = (w * w + h * h as f64).sqrt();
        // screen positions (full res) of the top bulb, left bulb, right bulb, and cardioid centre
        for (name, x, y) in [("top bulb", 910.0, 220.0), ("left bulb", 500.0, 480.0), ("right bulb", 1020.0, 420.0), ("cardioid", 900.0, 520.0), ("exterior", 200.0, 100.0)] {
            let nx = (x - w / 2.) / diag;
            let ny = -(y - h / 2.) / diag;
            let mapped = Complex64::new(nx.tan() * 2., ny.tan() * 2.);
            let coord = eng.reference.c + c7(mapped.re * scale, mapped.im * scale);
            let t = std::time::Instant::now();
            let r = eng.point(coord);
            let c64: Complex64 = coord.into();
            let f = crate::compute::calculate_mandelbrot_point_f64(c64, (1u64 << 16) as f64, false);
            println!("{name}: pert period {} ev {:.1} |slope| {:.3e} ({:.0} ms) | f64 period {} ev {:.1} |slope| {:.3e}",
                r.period, r.escape_velocity, r.slope.norm(), t.elapsed().as_secs_f64() * 1e3, f.period, f.escape_velocity, f.slope.norm());
        }
        // slope at zoom 1 for a clear exterior point
        let eng1 = engine(Complex64::new(-0.5, 0.0), 1.0);
        for (re, im) in [(0.5, 0.5), (-2.5, 0.1), (0.0, 1.5)] {
            let r = eng1.point(c7(re, im));
            let f = crate::compute::calculate_mandelbrot_point_f64(Complex64::new(re, im), (1u64 << 16) as f64, false);
            println!("zoom 1 ({re},{im}): pert |slope| {:.3e} f64 |slope| {:.3e}", r.slope.norm(), f.slope.norm());
        }
    }

    /// Diagnostic: do bulb pixels finish through the batch path as they do through `point`?
    #[test]
    #[ignore]
    fn debug_bulb_batch() {
        let scale = 2f64.powf(-18.714);
        let eng = engine(Complex64::new(-0.74249250281411, 0.20899915304036837), scale);
        let (w, h) = (2048.0f64, 1024.0f64);
        let diag = (w * w + h * h).sqrt();
        let to_coord = |x: f64, y: f64| {
            let nx = (x - w / 2.) / diag;
            let ny = -(y - h / 2.) / diag;
            eng.reference.c + c7(nx.tan() * 2. * scale, ny.tan() * 2. * scale)
        };
        // scan a box around the top bulb for non-64 interior periods
        let mut bulb: Vec<CircleF7E3> = Vec::new();
        let mut periods = std::collections::BTreeMap::new();
        for yi in 0..24 {
            for xi in 0..24 {
                let (x, y) = (880.0 + xi as f64 * 5.0, 170.0 + yi as f64 * 5.0);
                let c = to_coord(x, y);
                let r = eng.point(c);
                *periods.entry(r.period).or_insert(0) += 1;
                if r.period != 0 && r.period != 64 && bulb.len() < 256 {
                    bulb.push(c);
                }
            }
        }
        println!("scan periods: {periods:?}; bulb samples {}", bulb.len());
        // now the batch path with the CPU runner, GPU-like chunking
        let mut runner = CpuRunner;
        let mut batch: Batch<usize> = Batch::default();
        for (i, &c) in bulb.iter().enumerate() {
            let (slot, st) = eng.slot(c);
            batch.push(i, slot, st);
        }
        let mut done = std::collections::BTreeMap::new();
        for step in 0..400 {
            if batch.is_empty() { break; }
            for (_, r) in eng.step(&mut batch, &mut runner, 457) {
                *done.entry(r.period).or_insert(0) += 1;
            }
            if step % 50 == 49 {
                let st0 = batch.states.first().copied();
                println!("after {} steps: {} left; first: iter {:?} status {:?} n_deferred {:?} probe_n {:?}", step + 1, batch.len(),
                    st0.map(|s| s.iter), st0.map(|s| s.status), batch.slots.first().map(|s| s.n_deferred), st0.map(|s| s.probe_n));
            }
        }
        println!("batch finished periods: {done:?}; left {}", batch.len());
    }

    /// Diagnostic: where are the period-128 pixels on screen, and what colour do they get?
    #[test]
    #[ignore]
    fn debug_bulb_colour() {
        let scale = 2f64.powf(-18.714);
        let eng = engine(Complex64::new(-0.74249250281411, 0.20899915304036837), scale);
        let (w, h) = (2048.0f64, 1024.0f64);
        let diag = (w * w + h * h).sqrt();
        let zoom = Complex64::new(scale, 0.0);
        let mut periods = std::collections::BTreeMap::new();
        let mut shown = 0;
        for yi in 0..32 {
            for xi in 0..64 {
                let (x, y) = (xi as f64 * 32.0 + 16.0, yi as f64 * 32.0 + 16.0);
                let nx = (x - w / 2.) / diag;
                let ny = -(y - h / 2.) / diag;
                let c = eng.reference.c + c7(nx.tan() * 2. * scale, ny.tan() * 2. * scale);
                let r = eng.point(c);
                *periods.entry(r.period).or_insert(0) += 1;
                if r.period == 128 && shown < 6 {
                    shown += 1;
                    let col = crate::compute::calculate_colour(c.into(), &r, zoom);
                    println!("period 128 at ({x},{y}): |du| {:.3} w {} slope {:.3e} colour {:?}", r.coords.norm(), r.flame_c, r.slope.norm(), col);
                }
            }
        }
        println!("periods on a 64x32 grid: {periods:?}");
    }

    /// Diagnostic: trace the probe/deferral events for a pixel f64 reports as period 128.
    #[test]
    #[ignore]
    fn debug_bulb_trace() {
        let scale = 2f64.powf(-18.714);
        let eng = engine(Complex64::new(-0.74249250281411, 0.20899915304036837), scale);
        let (w, h) = (2048.0f64, 1024.0f64);
        let diag = (w * w + h * h).sqrt();
        let mut target = None;
        'scan: for yi in 0..32 {
            for xi in 0..64 {
                let (x, y) = (xi as f64 * 32.0 + 16.0, yi as f64 * 32.0 + 16.0);
                let nx = (x - w / 2.) / diag;
                let ny = -(y - h / 2.) / diag;
                let c = eng.reference.c + c7(nx.tan() * 2. * scale, ny.tan() * 2. * scale);
                let c64: Complex64 = c.into();
                let f = crate::compute::calculate_mandelbrot_point_f64(c64, (1u64 << 16) as f64, false);
                if f.period == 128 {
                    println!("f64 period 128 at ({x},{y}) |du| {:.3}", f.coords.norm());
                    target = Some(c);
                    break 'scan;
                }
            }
        }
        let Some(coord) = target else { println!("no period-128 pixel found by f64"); return; };
        let (mut slot, mut st) = eng.slot(coord);
        let s32 = eng.scale as f32;
        eng.newton_budget.store(u64::MAX, Ordering::Relaxed);
        let mut events = 0;
        loop {
            eng.arm(&slot, &mut st, 2048);
            run_f32(&mut st, &eng.reference, s32);
            if st.status == STATUS_CONFIRMED {
                events += 1;
                if events < 40 {
                    println!("iter {}: CONFIRMED period {} (windows {}), deferred before: {:?}", st.iter, st.period, st.probe_windows,
                        &slot.deferred[..slot.n_deferred as usize].iter().map(|(c, at)| (c.period, *at)).collect::<Vec<_>>());
                }
            }
            let before = slot.n_deferred;
            let res = eng.drive(&mut slot, &mut st);
            if slot.n_deferred < before && events < 40 {
                println!("iter {}: solved one; deferred now {:?}, probe_n {} blocked {}", st.iter,
                    &slot.deferred[..slot.n_deferred as usize].iter().map(|(c, at)| (c.period, *at)).collect::<Vec<_>>(), st.probe_n, st.probe_blocked);
            }
            if let Some(r) = res {
                println!("finished at iter {}: period {} ev {:.0}", st.iter, r.period, r.escape_velocity);
                break;
            }
            if st.iter > 40000 { println!("gave up at iter {}; probe_n {} next {} windows {} blocked {} deferred {:?}", st.iter, st.probe_n, st.probe_next, st.probe_windows, st.probe_blocked, slot.n_deferred); break; }
        }
    }
}
