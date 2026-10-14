use num_complex::Complex;
use num_complex::Complex64;
use spirix::{CircleF7E3, ScalarF7E3};
use std::f64::NAN;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::compute::mandelbrot_f64::MandelbrotResult;
use crate::compute::perturbation::{batch_shape, make_runner, Batch, DeltaRunner, PerturbEngine, Reference};
use crate::compute::{
    calculate_colour, calculate_mandelbrot_point_f32, calculate_mandelbrot_point_f64,
    spirix_point_fn, SpirixPointFn,
};

/// Point function for the epoch (display-buffer) thread: F7E3 coordinate in, result out.
pub type EpochPointFn = Arc<dyn Fn(CircleF7E3) -> MandelbrotResult + Send + Sync>;
/// Per-viewport setup run on the epoch thread before its first batch (builds the perturbation
/// reference, for instance). `None` means the setup was stopped and the thread should exit.
pub type EpochSetup = Box<dyn FnOnce(Arc<AtomicBool>) -> Option<EpochPointFn> + Send>;

/// Point function for the IEEE (f64-machinery) thread — both `calculate_mandelbrot_point_f64`
/// and `calculate_mandelbrot_point_f32` have this shape.
pub type IeeePointFn = fn(Complex64, f64, bool) -> MandelbrotResult;
use crate::config::{ViewMode, MAX_ITERATIONS};
use crate::render::plot_path_fft;

use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
pub struct MandelbrotPoint {
    pub coord: Complex64, // Complex coordinate this pixel represents
    pub color: u32,       // RGB color (0x00RRGGBB, no alpha)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionMode {
    /// IEEE binary32 pipeline (f64-machinery thread, f32 point math).
    F32,
    /// IEEE binary64 pipeline.
    F64,
    /// Scaled f32 perturbation against an F7E3 reference orbit (`compute::perturbation`).
    Perturbed,
    /// spirix grid cell: fraction/exponent width selectors, 3..=7 each (2^f / 2^e bits).
    Spirix { f: u8, e: u8 },
}

impl PrecisionMode {
    pub fn label(&self) -> String {
        match self {
            PrecisionMode::F32 => "f32".into(),
            PrecisionMode::F64 => "f64".into(),
            PrecisionMode::Perturbed => "f32 pert".into(),
            PrecisionMode::Spirix { f, e } => format!("F{}E{}", f, e),
        }
    }
}

// Batch output: pixel index + color + coordinate + screen position
struct CalculationResult {
    idx: usize,
    x: usize,          // Screen x coordinate (for fill_neighbors)
    y: usize,          // Screen y coordinate (for fill_neighbors)
    color: u32,        // ARGB packed
    pos: [f32; 2],     // Jittered sample position, for reprojection
}

pub struct MandelbrotBuffer {
    pub shared_display_buffer: Arc<Mutex<Vec<u32>>>, // Shared ARGB u32 display buffer
    // F7E5 pipeline
    /// Screen position (jittered, in pixels of the current viewport) of each pixel's sample;
    /// what the per-navigation reprojection moves.
    shared_positions: Arc<Mutex<Vec<[f32; 2]>>>,
    shared_viewport: Arc<Mutex<(CircleF7E3, CircleF7E3)>>, // (zoom, center) for calculation thread
    /// The delta runner (Metal device) lives across calculation-thread restarts.
    perturb_runner: Arc<Mutex<Option<Box<dyn DeltaRunner>>>>,
    /// Last perturbation reference, reused while it is still valid for the viewport.
    perturb_reference: Arc<Mutex<Option<Arc<Reference>>>>,
    /// Random visiting order of the pixels, shuffled once per size (not per navigation).
    pixel_order: Arc<Vec<usize>>,
    pub center: CircleF7E3,
    pub zoom: CircleF7E3,
    old_center: CircleF7E3,
    old_zoom: CircleF7E3,

    // f64 pipeline
    shared_coordinates_f64: Arc<Mutex<Vec<Complex64>>>,
    pub shared_viewport_f64: Arc<Mutex<(Complex64, Complex64)>>,
    pub calculation_buffer_f64: Arc<Mutex<Vec<MandelbrotPoint>>>,
    paused_flag: Arc<AtomicBool>,
    pub center_f64: Complex64,
    pub zoom_f64: Complex64,
    old_center_f64: Complex64,
    old_zoom_f64: Complex64,

    // Common fields
    pub width: usize,
    pub height: usize,
    old_width: usize,
    old_height: usize,
    stop_flag: Arc<AtomicBool>,
    calculation_thread: Option<thread::JoinHandle<()>>,
    pub current_epoch: Arc<AtomicU8>, // Viewport generation counter (wraps at 255)
    pub transform_requested: bool,
    pub viewport_changed: bool, // Set when viewport changes, cleared after display clear
    pub boundary_path: Vec<Complex64>,
    pub view_mode: ViewMode,
    pub current_precision_mode: PrecisionMode,
}
/// The pixel visiting order for a (re)started epoch thread: pixels nothing has been computed
/// for yet come first, in the shared random order, then the stale ones already carrying a
/// sample from before the navigation. The holes a zoom or pan opened fill immediately; the
/// carried samples, which are exact, are refreshed afterwards.
fn holes_first(order: &[usize], pixels: &[u32], skip_epoch: Option<u8>) -> Vec<usize> {
    let tag = |i: usize| pixels.get(i).map_or(0, |p| (p >> 24) as u8);
    let mut v = Vec::with_capacity(order.len());
    v.extend(order.iter().copied().filter(|&i| tag(i) == 0));
    v.extend(order.iter().copied().filter(|&i| {
        let t = tag(i);
        t != 0 && Some(t) != skip_epoch
    }));
    v
}

impl MandelbrotBuffer {
    /// Diagonal squared - width² + height², for distance calculations
    pub fn diagonal_sq(&self) -> f64 {
        let w = self.width as f64;
        let h = self.height as f64;
        w * w + h * h
    }

    pub fn new(
        width: usize,
        height: usize,
        center: Complex64,
        zoom: Complex64,
        boundary_path: Vec<Complex64>,
    ) -> Self {
        let size = width * height;

        // Convert Complex64 to CircleF7E3
        let center_f7e5 =
            CircleF7E3::from((ScalarF7E3::from(center.re), ScalarF7E3::from(center.im)));
        let zoom_f7e5 = CircleF7E3::from((ScalarF7E3::from(zoom.re), ScalarF7E3::from(zoom.im)));

        // f64 pipeline state
        let center_f64 = center;
        let zoom_f64 = zoom;

        Self {
            shared_display_buffer: Arc::new(Mutex::new(vec![0x00000000; size])), // Transparent black ARGB
            shared_positions: Arc::new(Mutex::new(vec![[f32::NAN, f32::NAN]; size])),
            shared_viewport: Arc::new(Mutex::new((zoom_f7e5, center_f7e5))),
            perturb_runner: Arc::new(Mutex::new(None)),
            perturb_reference: Arc::new(Mutex::new(None)),
            pixel_order: Arc::new({
                let mut v: Vec<usize> = (0..size).collect();
                v.shuffle(&mut thread_rng());
                v
            }),
            center: center_f7e5,
            zoom: zoom_f7e5,
            old_center: center_f7e5,
            old_zoom: zoom_f7e5,
            shared_coordinates_f64: Arc::new(Mutex::new(vec![
                Complex64::new(f64::NAN, f64::NAN);
                size
            ])),
            shared_viewport_f64: Arc::new(Mutex::new((zoom_f64, center_f64))),
            calculation_buffer_f64: Arc::new(Mutex::new(Vec::new())),
            paused_flag: Arc::new(AtomicBool::new(false)),
            center_f64,
            zoom_f64,
            old_center_f64: center_f64,
            old_zoom_f64: zoom_f64,
            width,
            height,
            old_width: width,
            old_height: height,
            stop_flag: Arc::new(AtomicBool::new(false)),
            calculation_thread: None,
            current_epoch: Arc::new(AtomicU8::new(1)), // Start at epoch 1
            viewport_changed: false,
            transform_requested: false,
            boundary_path,
            view_mode: crate::config::ViewMode::Brot,
            current_precision_mode: PrecisionMode::F64,
        }
    }

    pub fn transform_buffer(&mut self) {
        self.sync_f64_viewport();
        match self.current_precision_mode {
            PrecisionMode::F32 | PrecisionMode::F64 => self.transform_buffer_f64(),
            PrecisionMode::Perturbed | PrecisionMode::Spirix { .. } => {
                self.transform_buffer_spirix()
            }
        }
    }

    /// The F7E3 viewport (`center`, `zoom`) is the single source of truth; the f64 copy that the
    /// IEEE pipelines read is derived from it here. Keeping two independently-updated viewports
    /// let any path that touched only one (click-to-center, the `S` search) desync them, so the
    /// f32/f64 and spirix modes showed different areas.
    fn sync_f64_viewport(&mut self) {
        self.center_f64 = self.center.into();
        self.zoom_f64 = self.zoom.into();
        *self.shared_viewport_f64.lock().unwrap() = (self.zoom_f64, self.center_f64);
    }

    /// Reproject calculation buffer to new viewport, filtering out off-screen points
    fn reproject_calculation_buffer_f64(&mut self) {
        let mut calc_buffer = self.calculation_buffer_f64.lock().unwrap();
        let (zoom, center) = (self.zoom_f64, self.center_f64);
        let scale = (self.diagonal_sq()).sqrt();

        // Filter: keep only points that project on-screen in new viewport
        calc_buffer.retain(|point| {
            let (x, y) = Self::complex_to_screen_f64(
                point.coord,
                self.width as f64,
                self.height as f64,
                zoom,
                center,
                scale,
            );

            let xi = x.floor() as i32;
            let yi = y.floor() as i32;

            // Keep if projects within new screen bounds
            xi >= 0 && xi < self.width as i32 && yi >= 0 && yi < self.height as i32
        });

        // calc_buffer is now compacted - only contains on-screen points
    }

    fn transform_buffer_f64(&mut self) {
        // Pause calculation thread during viewport changes
        self.paused_flag.store(true, Ordering::SeqCst);

        // Update shared viewport for f64 thread
        *self.shared_viewport_f64.lock().unwrap() = (self.zoom_f64, self.center_f64);

        // REPROJECT existing calculations to new viewport, filter off-screen
        self.reproject_calculation_buffer_f64();

        // Mark viewport as changed - composite will clear display buffer on next frame
        self.viewport_changed = true;

        // Update old state
        self.old_width = self.width;
        self.old_height = self.height;
        self.old_center_f64 = self.center_f64;
        self.old_zoom_f64 = self.zoom_f64;

        // Resume calculation thread with new viewport
        self.paused_flag.store(false, Ordering::SeqCst);
    }

    fn transform_buffer_spirix(&mut self) {
        // Update shared viewport for the spirix thread
        *self.shared_viewport.lock().unwrap() = (self.zoom, self.center);

        // Increment epoch - calculation thread will discard in-flight batches with old epoch
        let new_epoch = self.current_epoch.load(Ordering::SeqCst).wrapping_add(1);
        self.current_epoch.store(new_epoch, Ordering::SeqCst);

        // Carry the picture along: remap the frame from the old viewport into the new one so
        // navigation moves it immediately. The warped pixels keep their old epoch tags, so the
        // thread overwrites them as it recomputes, and each keeps the exact coordinate of the
        // sample it was copied from, so interpolation stays honest.
        let (w, h, z, c) = (self.width, self.height, self.old_zoom, self.old_center);
        self.reproject_epoch_buffers(w, h, z, c);
        // The composite rebuilds the frame from the warped buffer; wipe it so nothing stale
        // survives where the warp found no source.
        self.viewport_changed = true;

        self.old_width = self.width;
        self.old_height = self.height;
        self.old_center = self.center;
        self.old_zoom = self.zoom;
    }

    /// Forward-project every computed sample of the epoch buffers from a source viewport
    /// (`src_w` × `src_h` pixels showing `src_zoom`, `src_center`) into the current one (`width`
    /// × `height`, `zoom`, `center`): each sample lands on the one pixel its exact position maps
    /// to, samples that leave the screen are trimmed, and on zoom-in the carried samples stay
    /// sparse for the flood fill to grow, as the f64 point cloud always did. The two viewports
    /// are related in the screen's normalized frame through two f64 quantities that stay
    /// representable at any depth: the centre shift in units of the new zoom and the zoom
    /// ratio; a size change only changes the normalisation on each side.
    fn reproject_epoch_buffers(
        &mut self,
        src_w: usize,
        src_h: usize,
        src_zoom: CircleF7E3,
        src_center: CircleF7E3,
    ) {
        let (w, h) = (self.width, self.height);
        if w == 0 || h == 0 || src_w == 0 || src_h == 0 {
            return;
        }
        let offset: Complex64 = ((src_center - self.center) / self.zoom).into();
        let ratio: Complex64 = (src_zoom / self.zoom).into();
        if !offset.re.is_finite() || !offset.im.is_finite() || !ratio.re.is_finite() || ratio.norm_sqr() == 0. {
            return;
        }
        let same_size = src_w == w && src_h == h;
        if same_size && offset.norm_sqr() == 0. && (ratio - Complex64::new(1., 0.)).norm_sqr() == 0. {
            return;
        }
        let n_src = src_w * src_h;
        let n = w * h;
        let src_diag = ((src_w * src_w + src_h * src_h) as f64).sqrt();
        let dst_diag = ((w * w + h * h) as f64).sqrt();
        let (swf, shf) = (src_w as f64, src_h as f64);
        let (wf, hf) = (w as f64, h as f64);

        let mut pixels = self.shared_display_buffer.lock().unwrap();
        let mut positions = self.shared_positions.lock().unwrap();
        if pixels.len() != n_src || positions.len() != n_src {
            // Buffers do not describe the source viewport: nothing trustworthy to carry.
            *pixels = vec![0; n];
            *positions = vec![[f32::NAN, f32::NAN]; n];
            return;
        }

        // Pass 1 (parallel): where does each computed sample land?
        let dest: Vec<(u32, f32, f32)> = (0..n_src)
            .into_par_iter()
            .map(|i| {
                if pixels[i] >> 24 == 0 {
                    return (u32::MAX, 0., 0.);
                }
                let [sx, sy] = positions[i];
                if !sx.is_finite() || !sy.is_finite() {
                    return (u32::MAX, 0., 0.);
                }
                let re_old = ((sx as f64) - swf / 2.) / src_diag;
                let im_old = -((sy as f64) - shf / 2.) / src_diag;
                let mapped_old = Complex64::new(re_old.tan() * 2., im_old.tan() * 2.);
                // coord = mapped·zoom + centre in both frames ⇒ mapped_new = mapped_old·ratio + offset
                let mapped_new = mapped_old * ratio + offset;
                let dx = (mapped_new.re * 0.5).atan() * dst_diag + wf / 2.;
                let dy = -(mapped_new.im * 0.5).atan() * dst_diag + hf / 2.;
                if dx >= 0. && dx < wf && dy >= 0. && dy < hf {
                    ((dy as usize * w + dx as usize) as u32, dx as f32, dy as f32)
                } else {
                    (u32::MAX, 0., 0.)
                }
            })
            .collect();

        // Pass 2 (serial scatter): last writer wins where samples collapse together.
        let mut px2 = vec![0u32; n];
        let mut po2 = vec![[f32::NAN, f32::NAN]; n];
        for i in 0..n_src {
            let (d, dx, dy) = dest[i];
            if d != u32::MAX {
                let d = d as usize;
                px2[d] = pixels[i];
                po2[d] = [dx, dy];
            }
        }
        *pixels = px2;
        *positions = po2;
    }

    pub fn screen_to_complex(
        x: f64,
        y: f64,
        width: f64,
        height: f64,
        zoom: CircleF7E3,
        center: CircleF7E3,
    ) -> CircleF7E3 {
        let scale = 1. / (width * width + height * height).sqrt();

        let mut re = (x - width / 2.) * scale;
        let mut im = -(y - height / 2.) * scale; // Invert Y-axis to match math convention

        re = re.tan() * 2.;
        im = im.tan() * 2.;

        // Convert to F7E7 for precision-critical operations
        let coord_f7e5 = CircleF7E3::from((ScalarF7E3::from(re), ScalarF7E3::from(im)));

        // Multiply by zoom and add center in F7E7
        (coord_f7e5 * zoom) + center
    }
    pub fn complex_to_screen(
        c: CircleF7E3,
        width: f64,
        height: f64,
        zoom: CircleF7E3,
        center: CircleF7E3,
        scale: f64,
    ) -> (f64, f64) {
        let c_transformed = (c - center) / zoom;
        let re: f64 = c_transformed.r().into();
        let im: f64 = c_transformed.i().into();

        let re_atan = (re * 0.5).atan();
        let im_atan = (im * 0.5).atan();

        let x = re_atan * scale + width / 2.;
        let y = -im_atan * scale + height / 2.; // Invert Y-axis to match math convention
        (x, y)
    }

    // Pure f64 pipeline functions (no spirix types)
    pub fn screen_to_complex_f64(
        x: f64,
        y: f64,
        width: f64,
        height: f64,
        zoom: Complex64,
        center: Complex64,
    ) -> Complex64 {
        let scale = 1.0 / (width * width + height * height).sqrt();

        let mut re = (x - width / 2.0) * scale;
        let mut im = -(y - height / 2.0) * scale; // Invert Y-axis

        re = re.tan() * 2.0;
        im = im.tan() * 2.0;

        let coord = Complex64::new(re, im);
        coord * zoom + center
    }

    pub fn complex_to_screen_f64(
        c: Complex64,
        width: f64,
        height: f64,
        zoom: Complex64,
        center: Complex64,
        scale: f64,
    ) -> (f64, f64) {
        let c_transformed = (c - center) / zoom;
        let re = c_transformed.re;
        let im = c_transformed.im;

        let re_atan = (re * 0.5).atan();
        let im_atan = (im * 0.5).atan();

        let x = re_atan * scale + width / 2.0;
        let y = -im_atan * scale + height / 2.0; // Invert Y-axis
        (x, y)
    }

    pub fn apply_zoom(&mut self, cursor_pos: (f64, f64), zoom_delta: f64) {
        let zoom_factor: f64 = if zoom_delta > 0. { 7. / 6. } else { 6. / 7. };
        let scale = zoom_factor.powf(zoom_delta.abs());

        // No navigation limit: coordinates are F7E3, whose 8-bit exponent runs out near
        // 2^±127. Past that the zoom vanishes to zero and the view collapses; watching the
        // coordinate type fail is as much a part of this rig as watching the working types fail.
        let cursor_coordinates = Self::screen_to_complex(
            cursor_pos.0,
            cursor_pos.1,
            self.width as f64,
            self.height as f64,
            self.zoom,
            self.center,
        );
        let scale_f7e3 = ScalarF7E3::from(scale);
        self.zoom = self.zoom / CircleF7E3::from((scale_f7e3, ScalarF7E3::ZERO));
        self.center = cursor_coordinates + (self.center - cursor_coordinates) / scale_f7e3;
        self.log_viewport();
    }

    fn log_viewport(&self) {
        let center: Complex64 = self.center.into();
        let zoom: Complex64 = self.zoom.into();
        println!(
            "Viewport: Center=[{}, {}], Zoom=[{}, {}] (magnitude: 2^{})",
            center.re,
            center.im,
            zoom.re,
            zoom.im,
            zoom.norm().log2()
        );
    }

    pub fn apply_rotation(&mut self, cursor_pos: (f64, f64), rotation_delta: f64) {
        // Update F7E5 viewport
        let cursor_coordinates_old = Self::screen_to_complex(
            cursor_pos.0,
            cursor_pos.1,
            self.width as f64,
            self.height as f64,
            self.zoom,
            self.center,
        );

        let rotation = CircleF7E3::from((
            ScalarF7E3::from(rotation_delta.cos()),
            ScalarF7E3::from(rotation_delta.sin()),
        ));

        self.zoom = self.zoom * rotation;
        self.center = cursor_coordinates_old + (self.center - cursor_coordinates_old) * rotation;

        let cursor_coordinates_new = Self::screen_to_complex(
            cursor_pos.0,
            cursor_pos.1,
            self.width as f64,
            self.height as f64,
            self.zoom,
            self.center,
        );

        self.center = self.center + (cursor_coordinates_old - cursor_coordinates_new);

        self.log_viewport();
    }

    pub fn start_spirix_iterations(&mut self, point_fn: SpirixPointFn) {
        self.start_epoch_iterations(Box::new(move |_| {
            Some(Arc::new(move |c| point_fn(c, MAX_ITERATIONS)) as EpochPointFn)
        }));
    }

    /// Perturbation thread. It is persistent across navigations: when the viewport epoch
    /// changes it drops its in-flight batch, re-reads the viewport, reuses or rebuilds the
    /// reference and continues, so a pan or zoom never waits on a thread join. It keeps a pool
    /// of pixels in flight on the delta runner (GPU when available), drives each a chunk of
    /// iterations per frame and writes finished pixels into the display buffer with the epoch
    /// tag. Mode switches and resizes still stop and restart it.
    pub fn start_perturbed_iterations(&mut self) {
        if self.calculation_thread.is_some() {
            return; // already running; it follows the epoch
        }
        self.stop_flag.store(false, Ordering::SeqCst);

        let width = self.width;
        let height = self.height;
        let shared_display_buffer = Arc::clone(&self.shared_display_buffer);
        let shared_positions = Arc::clone(&self.shared_positions);
        let shared_viewport = Arc::clone(&self.shared_viewport);
        let stop_flag = Arc::clone(&self.stop_flag);
        let current_epoch = Arc::clone(&self.current_epoch);
        let runner_slot = Arc::clone(&self.perturb_runner);
        let reference_slot = Arc::clone(&self.perturb_reference);
        let pixel_order = Arc::clone(&self.pixel_order);
        *self.shared_viewport.lock().unwrap() = (self.zoom, self.center);

        self.calculation_thread = Some(thread::spawn(move || {
            const TARGET_FRAME_TIME: Duration = Duration::from_micros(16_667);
            const WORK_BUDGET: Duration = Duration::from_millis(12);
            /// Iterations per dispatch are capped so one step stays a few milliseconds: an
            /// epoch change (navigation) is noticed between steps.
            const STEP_WORK: f64 = 3.0e7;

            // The runner (and its Metal device) outlives this thread; take it for the duration.
            let mut runner_guard = runner_slot.lock().unwrap();
            let runner = runner_guard.get_or_insert_with(make_runner);
            let (capacity, chunk_max) = batch_shape(runner.as_ref());

            let mut engine: Option<PerturbEngine> = None;
            let mut my_epoch = current_epoch.load(Ordering::SeqCst);
            let mut zoom = CircleF7E3::ZERO;
            let mut center = CircleF7E3::ZERO;
            let mut batch: Batch<(usize, CircleF7E3, [f32; 2])> = Batch::default();
            let mut indices: Vec<usize> = Vec::new();
            let mut cursor = 0usize;
            let mut rng = rand::thread_rng();

            loop {
                if stop_flag.load(Ordering::SeqCst) {
                    return;
                }
                let epoch_now = current_epoch.load(Ordering::SeqCst);
                if engine.is_none() || epoch_now != my_epoch {
                    // (Re)start for the current viewport.
                    my_epoch = epoch_now;
                    let (z, c) = *shared_viewport.lock().unwrap();
                    zoom = z;
                    center = c;
                    let t = Instant::now();
                    let cached = reference_slot.lock().unwrap().clone();
                    let Some(e) = PerturbEngine::new_cached(
                        zoom,
                        center,
                        MAX_ITERATIONS,
                        Arc::clone(&stop_flag),
                        cached.clone(),
                    ) else {
                        if stop_flag.load(Ordering::SeqCst) {
                            return;
                        }
                        thread::sleep(TARGET_FRAME_TIME);
                        continue;
                    };
                    let reused = cached.is_some_and(|c| Arc::ptr_eq(&c, &e.reference));
                    *reference_slot.lock().unwrap() = Some(Arc::clone(&e.reference));
                    if !reused {
                        println!(
                            "Perturbation: reference {} entries, period {:?}, built in {:.0} ms; runner {}",
                            e.reference.z64.len(),
                            e.reference.period,
                            t.elapsed().as_secs_f64() * 1e3,
                            runner.name()
                        );
                    }
                    engine = Some(e);
                    batch = Batch::default();
                    indices = holes_first(&pixel_order, &shared_display_buffer.lock().unwrap(), None);
                    cursor = 0;
                    continue;
                }
                // The worker found a better reference (exact nucleus, or the full-length
                // orbit): rebuild on it in place. Pixels finished this epoch are exact and
                // stay; holes and stale pixels are revisited.
                if let Some(up) = engine.as_ref().unwrap().take_upgrade() {
                    let e = PerturbEngine::with_reference(up, zoom, MAX_ITERATIONS, Arc::clone(&stop_flag));
                    *reference_slot.lock().unwrap() = Some(Arc::clone(&e.reference));
                    println!(
                        "Perturbation: reference upgraded to {} entries, period {:?}",
                        e.reference.z64.len(),
                        e.reference.period
                    );
                    engine = Some(e);
                    batch = Batch::default();
                    indices = holes_first(&pixel_order, &shared_display_buffer.lock().unwrap(), Some(my_epoch));
                    cursor = 0;
                    continue;
                }
                let engine = engine.as_ref().unwrap();
                let frame_start = Instant::now();
                let batch_epoch = my_epoch;
                let zoom_c64: Complex64 = zoom.into();

                while batch.len() < capacity && cursor < indices.len() {
                    let idx = indices[cursor];
                    cursor += 1;
                    let x = (idx % width) as f64 + rng.gen::<f64>();
                    let y = (idx / width) as f64 + rng.gen::<f64>();
                    let coord =
                        Self::screen_to_complex(x, y, width as f64, height as f64, zoom, center);
                    let (slot, state) = engine.slot(coord);
                    batch.push((idx, coord, [x as f32, y as f32]), slot, state);
                }
                if batch.is_empty() {
                    // Every pixel of this viewport is done; idle until the epoch moves.
                    thread::sleep(TARGET_FRAME_TIME);
                    continue;
                }

                let mut done = Vec::new();
                loop {
                    let chunk = ((STEP_WORK / batch.len().max(1) as f64) as u32).clamp(128, chunk_max);
                    let slots = batch.len();
                    let t_step = Instant::now();
                    done.extend(engine.step(&mut batch, runner.as_mut(), chunk));
                    let step_ms = t_step.elapsed().as_secs_f64() * 1e3;
                    if step_ms > 50.0 && std::env::var_os("MANDEL_TIMING").is_some() {
                        let st = &engine.stats;
                        let ld = |a: &std::sync::atomic::AtomicU64| a.swap(0, Ordering::Relaxed) as f64 * 1e-6;
                        println!(
                            "timing: thread step {step_ms:.0} ms ({slots} slots × {chunk} iterations): runner {:.0} ms, drive {:.0} ms wall; escape rides {:.0} ms cpu over {} px, newton {:.0} ms cpu over {} solves ({} hits)",
                            ld(&st.ns_runner), ld(&st.ns_drive), ld(&st.ns_escape),
                            st.escapes.swap(0, Ordering::Relaxed), ld(&st.ns_newton),
                            st.newton_solves.swap(0, Ordering::Relaxed),
                            st.newton_hits.swap(0, Ordering::Relaxed)
                        );
                    }
                    if batch.is_empty()
                        || frame_start.elapsed() >= WORK_BUDGET
                        || stop_flag.load(Ordering::SeqCst)
                        || current_epoch.load(Ordering::SeqCst) != batch_epoch
                    {
                        break;
                    }
                }

                if !done.is_empty() {
                    if let Ok(mut pixels) = shared_display_buffer.lock() {
                        if current_epoch.load(Ordering::SeqCst) == batch_epoch {
                            let mut positions = shared_positions.lock().unwrap();
                            for ((idx, coord, pos), result) in &done {
                                let existing_epoch = ((pixels[*idx] >> 24) & 0xFF) as u8;
                                if existing_epoch != 0 && existing_epoch == batch_epoch {
                                    continue;
                                }
                                let colour_c: Complex64 = (*coord).into();
                                let colour = calculate_colour(colour_c, result, zoom_c64);
                                let color = 0xFF000000
                                    | (colour[0] as u32) << 16
                                    | (colour[1] as u32) << 8
                                    | colour[2] as u32;
                                pixels[*idx] =
                                    (color & 0x00FFFFFF).wrapping_add((batch_epoch as u32) << 24);
                                positions[*idx] = *pos;
                            }
                        }
                    }
                }

                let elapsed = frame_start.elapsed();
                if elapsed < TARGET_FRAME_TIME && current_epoch.load(Ordering::SeqCst) == batch_epoch {
                    thread::sleep(TARGET_FRAME_TIME - elapsed);
                }
            }
        }));
    }

    /// The epoch thread: random pixel batches, results written straight into the shared display
    /// buffer tagged with the viewport epoch. `setup` runs first on the thread.
    pub fn start_epoch_iterations(&mut self, setup: EpochSetup) {
        self.stop_calculation();

        self.stop_flag.store(false, Ordering::SeqCst);

        let width = self.width;
        let height = self.height;
        let zoom = self.zoom;
        let center = self.center;
        let shared_display_buffer: Arc<Mutex<Vec<u32>>> = Arc::clone(&self.shared_display_buffer);
        let shared_positions = Arc::clone(&self.shared_positions);
        let stop_flag: Arc<AtomicBool> = Arc::clone(&self.stop_flag);
        let current_epoch: Arc<AtomicU8> = Arc::clone(&self.current_epoch);

        let order = holes_first(&self.pixel_order, &self.shared_display_buffer.lock().unwrap(), None);

        self.calculation_thread = Some(thread::spawn(move || {
            let Some(point_fn) = setup(Arc::clone(&stop_flag)) else {
                return;
            };
            let mut indices: Vec<usize> = order;
            let mut index_cursor = 0; // Track progress thru indices

            const TARGET_FRAME_TIME: Duration = Duration::from_micros(16_667);

            loop {
                if stop_flag.load(Ordering::SeqCst) {
                    return;
                }

                let frame_start = Instant::now();

                // Load epoch at start of batch - if it changes during calculation, discard results
                let batch_epoch = current_epoch.load(Ordering::SeqCst);

                // PHASE 1: Generate batch for this frame
                let mut batch = Vec::new();
                let mut rng = rand::thread_rng();

                // Collect uncalculated pixels until we have a reasonable batch
                while batch.len() < 1 << 16 && index_cursor < indices.len() {
                    let idx = indices[index_cursor];
                    index_cursor += 1;

                    // Generate coordinate
                    let x = (idx % width) as f64 + rng.gen::<f64>();
                    let y = (idx / width) as f64 + rng.gen::<f64>();
                    let coord =
                        Self::screen_to_complex(x, y, width as f64, height as f64, zoom, center);

                    batch.push((idx, coord, [x as f32, y as f32]));
                }

                // If we've processed all indices, reshuffle and restart
                if index_cursor >= indices.len() {
                    indices.shuffle(&mut thread_rng());
                    index_cursor = 0;
                }

                // PHASE 2: Parallel calculation (all threads finish their work)
                let results: Vec<CalculationResult> = batch
                    .par_iter()
                    .map(|&(idx, coord, pos)| {
                        let result = point_fn(coord);
                        let colour = calculate_colour(coord.into(), &result, zoom.into());

                        CalculationResult {
                            idx,
                            x: idx % width,
                            y: idx / width,
                            color: 0xFF000000
                                | (colour[0] as u32) << 16
                                | (colour[1] as u32) << 8
                                | colour[2] as u32,
                            pos,
                        }
                    })
                    .collect();

                // PHASE 3: Write + interpolate (single lock per frame)
                if let Ok(mut pixels) = shared_display_buffer.lock() {
                    {
                        let mut positions = shared_positions.lock().unwrap();
                        // Check if epoch changed during calculation - if so, discard results
                        let current = current_epoch.load(Ordering::SeqCst);
                        if current != batch_epoch {
                            // Viewport changed, these pixels are stale
                            continue;
                        }

                        for result in results {
                            let existing_epoch = ((pixels[result.idx] >> 24) & 0xFF) as u8;

                            // Skip if already calculated in current epoch (not uncalculated)
                            // Allow overwriting: uncalculated (epoch 0) or old epochs
                            if existing_epoch != 0 && existing_epoch == batch_epoch {
                                continue;
                            }

                            // Write calculated pixel with batch epoch
                            pixels[result.idx] = (result.color & 0x00FFFFFF)
                                .wrapping_add((batch_epoch as u32) << 24);
                            positions[result.idx] = result.pos;
                            // No interpolation here - happens during composite
                        }
                    }
                }

                // PHASE 4: Sleep to maintain frame rate
                let elapsed = frame_start.elapsed();
                if elapsed < TARGET_FRAME_TIME {
                    std::thread::sleep(TARGET_FRAME_TIME - elapsed);
                }
            }
        }));
    }

    pub fn start_ieee_iterations(&mut self, point_fn: IeeePointFn) {
        self.stop_calculation();

        self.stop_flag.store(false, Ordering::SeqCst);

        let width = self.width;
        let height = self.height;
        let calculation_buffer = Arc::clone(&self.calculation_buffer_f64);
        let shared_viewport = Arc::clone(&self.shared_viewport_f64);
        let stop_flag = Arc::clone(&self.stop_flag);
        let paused_flag = Arc::clone(&self.paused_flag);

        self.calculation_thread = Some(thread::spawn(move || {
            const TARGET_FRAME_TIME: Duration = Duration::from_micros(66_667); // 15fps
            let total_pixels = width * height;

            loop {
                if stop_flag.load(Ordering::SeqCst) {
                    return;
                }

                // If paused (viewport changing), sleep and check again
                if paused_flag.load(Ordering::SeqCst) {
                    thread::sleep(Duration::from_millis(16)); // ~60fps check rate
                    continue;
                }

                let frame_start = Instant::now();
                let (zoom, center) = *shared_viewport.lock().unwrap();
                let scale = ((width * width + height * height) as f64).sqrt();

                // Build coverage grid: which screen pixels have calculated points nearby?
                let mut coverage_grid = vec![false; total_pixels];

                if let Ok(calc_buffer) = calculation_buffer.lock() {
                    for point in calc_buffer.iter() {
                        let (x, y) = MandelbrotBuffer::complex_to_screen_f64(
                            point.coord,
                            width as f64,
                            height as f64,
                            zoom,
                            center,
                            scale,
                        );

                        let xi = x.floor() as i32;
                        let yi = y.floor() as i32;

                        if xi >= 0 && xi < width as i32 && yi >= 0 && yi < height as i32 {
                            let idx = (yi as usize) * width + (xi as usize);
                            coverage_grid[idx] = true;
                        }
                    }
                }

                // Find uncovered pixels
                let mut uncovered: Vec<usize> = coverage_grid
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &covered)| if !covered { Some(idx) } else { None })
                    .collect();

                // If no uncovered pixels, we're done!
                if uncovered.is_empty() {
                    return; // EXIT THREAD
                }

                // Shuffle uncovered pixels for random sampling
                uncovered.shuffle(&mut thread_rng());

                // Generate batch (up to 2^16 pixels)
                let batch_size = uncovered.len().min(1 << 16);
                let mut batch = Vec::with_capacity(batch_size);
                let mut rng = rand::thread_rng();

                for &idx in uncovered.iter().take(batch_size) {
                    let x = (idx % width) as f64 + rng.gen::<f64>();
                    let y = (idx / width) as f64 + rng.gen::<f64>();

                    let coord = MandelbrotBuffer::screen_to_complex_f64(
                        x, y, width as f64, height as f64, zoom, center
                    );

                    batch.push(coord);
                }

                // Parallel calculation
                let results: Vec<MandelbrotPoint> = batch
                    .par_iter()
                    .map(|&coord| {
                        let result = point_fn(coord, MAX_ITERATIONS as f64, false);
                        let colour = calculate_colour(coord, &result, zoom);
                        let color = (colour[0] as u32) << 16
                                  | (colour[1] as u32) << 8
                                  | colour[2] as u32;

                        MandelbrotPoint { coord, color }
                    })
                    .collect();

                // Append results to calculation buffer
                // Points are stored in absolute complex coordinates, so they remain valid
                // even if viewport changed during calculation. Reprojection will filter if needed.
                if let Ok(mut calc_buffer) = calculation_buffer.lock() {
                    calc_buffer.extend(results);
                }

                // Sleep to maintain frame rate
                let elapsed = frame_start.elapsed();
                if elapsed < TARGET_FRAME_TIME {
                    thread::sleep(TARGET_FRAME_TIME - elapsed);
                }
            }
        }));
    }

    pub fn start_calculation(&mut self) {
        // A mode switch starts a thread without a transform; refresh the f64 thread's viewport
        // so it never resumes on whatever view was current the last time an IEEE mode ran.
        self.sync_f64_viewport();
        match self.current_precision_mode {
            PrecisionMode::F32 => self.start_ieee_iterations(calculate_mandelbrot_point_f32),
            PrecisionMode::F64 => self.start_ieee_iterations(calculate_mandelbrot_point_f64),
            PrecisionMode::Perturbed => self.start_perturbed_iterations(),
            PrecisionMode::Spirix { f, e } => self.start_spirix_iterations(spirix_point_fn(f, e)),
        }
    }

    pub fn stop_calculation(&mut self) {
        self.stop_flag.store(true, Ordering::SeqCst);

        // Wait for thread to actually stop
        if let Some(thread) = self.calculation_thread.take() {
            let _ = thread.join();
        }
    }
}

impl MandelbrotBuffer {
    /// Resize buffer dimensions - clears calculation buffer
    pub fn resize(&mut self, new_width: usize, new_height: usize) {
        // Stop any ongoing calculation and wait for thread to finish
        self.stop_calculation();

        // Update dimensions
        let (old_w, old_h) = (self.width, self.height);
        self.width = new_width;
        self.height = new_height;

        let new_size = new_width * new_height;

        // Clear calculation buffer on resize (reprojection is too slow for large buffers)
        // TODO: Optimize reprojection or use spatial indexing for faster filtering
        self.calculation_buffer_f64.lock().unwrap().clear();

        // Carry the epoch picture into the new size: same centre and zoom, new framing. The
        // thread is stopped, so nothing else touches the buffers meanwhile. (The buffers used
        // to be cleared, and the position buffer not even resized, which panicked the thread
        // at the first pixel past the old size.)
        let (z, c) = (self.zoom, self.center);
        self.reproject_epoch_buffers(old_w, old_h, z, c);
        self.old_width = new_width;
        self.old_height = new_height;
        self.old_center = self.center;
        self.old_zoom = self.zoom;
        self.pixel_order = Arc::new({
            let mut v: Vec<usize> = (0..new_size).collect();
            v.shuffle(&mut thread_rng());
            v
        });

        self.viewport_changed = true;

        // Restart calculation thread with new dimensions
        self.start_calculation();
    }

    pub fn format_point_statistics(
        point: Complex64,
        zoom: Complex64,
        period: u64,
        escape_velocity: Complex64,
        coords: Complex64,
        flames: Complex64,
        slope: Complex64,
        multiplier: Complex64,
    ) -> String {
        if period == 0 {
            format!(
                "Outside point: [{}, {}]\n\
                 Scale: 2^{}\n\
                 Zoom: [{}, {}]\n\
                 Escape velocity: [{}, {}] (norm: {})\n\
                 Coordinates: [{}, {}]\n\
                 Flame coords: [{}, {}]\n\
                 Slope: [{}, {}]\n\
                 Multiplier: [{}, {}]",
                point.re,
                point.im,
                zoom.norm().log2(),
                zoom.re,
                zoom.im,
                escape_velocity.re,
                escape_velocity.im,
                escape_velocity.norm(),
                coords.re,
                coords.im,
                flames.re,
                flames.im,
                slope.re,
                slope.im,
                multiplier.re,
                multiplier.im
            )
        } else {
            format!(
                "Inside point: [{}, {}]\n\
                Scale: 2^{}\n\
                Zoom: [{}, {}]\n\
                Period: {}\n\
                Coordinates: [{}, {}]\n\
                Flame coords: [{}, {}]\n\
                Slope: [{}, {}]\n\
                Multiplier: [{}, {}]",
                point.re,
                point.im,
                zoom.norm().log2(),
                zoom.re,
                zoom.im,
                period,
                coords.re,
                coords.im,
                flames.re,
                flames.im,
                slope.re,
                slope.im,
                multiplier.re,
                multiplier.im
            )
        }
    }
    /// Draw the traced boundary path as lines into the caller's visible-RGB frame buffer.
    pub fn plot_boundary_path(&self, display: &mut [u32]) {
        if self.boundary_path.is_empty() {
            return;
        }

        let scale = ((self.width * self.width + self.height * self.height) as f64).sqrt();
        let mut visible_points = Vec::new();
        let mut prev_xi = std::isize::MIN;
        let mut prev_yi = std::isize::MIN;
        let mut skipped = false;

        for idx in 0..self.boundary_path.len() {
            let point_c64 = self.boundary_path[idx];
            let point_f7e5 = CircleF7E3::from(point_c64);
            let (x, y) = Self::complex_to_screen(
                point_f7e5,
                self.width as f64,
                self.height as f64,
                self.zoom,
                self.center,
                scale,
            );

            let xi = x.floor() as isize;
            let yi = y.floor() as isize;

            if prev_xi != xi || prev_yi != yi {
                if (xi as usize) < self.width && (yi as usize) < self.height {
                    prev_xi = xi;
                    prev_yi = yi;
                    if skipped {
                        skipped = false;
                        visible_points.push(Complex::new(NAN, NAN));
                    }
                    visible_points.push(Complex::new(x, y));
                } else {
                    skipped = true;
                }
            }
        }

        if visible_points.len() > 1 {
            for i in 0..visible_points.len() - 1 {
                let x0 = visible_points[i].re;
                let y0 = visible_points[i].im;
                let x1 = visible_points[i + 1].re;
                let y1 = visible_points[i + 1].im;
                crate::render::line::draw_line_u32(
                    display,
                    self.width,
                    x0 as f32,
                    y0 as f32,
                    x1 as f32,
                    y1 as f32,
                    0xFF3F3F3F, // Dark gray in ARGB (63, 63, 63)
                );
            }
        }
    }
    pub fn switch_view_mode(&mut self, point: Complex64) {
        match self.view_mode {
            crate::config::ViewMode::Brot => {
                self.view_mode = crate::config::ViewMode::FFT;
                *self.shared_display_buffer.lock().unwrap() =
                    plot_path_fft(point, self.width, self.height);
            }
            crate::config::ViewMode::FFT => {
                self.view_mode = crate::config::ViewMode::Brot;
                self.transform_requested = true;
            }
        }
    }
}

impl MandelbrotBuffer {
    /// Search for interesting spots by sampling random points and finding high iteration counts.
    /// Uses normal distribution centered at the current position, sampling 256 points (excluding inside points).
    /// Returns the location with the highest iteration count (but not inside the set).
    pub fn find_interesting_spot(
        &self,
        center: Complex64,
        radius: f64,
    ) -> Option<(Complex64, f64)> {
        let mut rng = thread_rng();
        let normal_re = Normal::new(center.re, radius).unwrap();
        let normal_im = Normal::new(center.im, radius).unwrap();

        let mut samples = Vec::new();

        // Sample 256 points that are outside the set
        while samples.len() < 256 {
            let re = normal_re.sample(&mut rng);
            let im = normal_im.sample(&mut rng);
            let point = Complex64::new(re, im);

            let result = calculate_mandelbrot_point_f64(point, MAX_ITERATIONS as f64, false);

            // Only count points outside the set (period == 0)
            if result.period == 0 {
                samples.push((point, result.escape_velocity));
            }
        }

        // Find the point with the highest iteration count (escape_velocity)
        samples
            .into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    }

    /// Iteratively search for the most interesting spot by repeatedly sampling and zooming in.
    /// Each iteration samples 256 random points over a normal distribution, finds the highest
    /// iteration count, centers on that point, and halves the radius.
    /// Continues until floating-point precision limit is reached.
    pub fn search_for_interesting_spot(&mut self, _max_iterations: usize) {
        let mut center: Complex64 = self.center.into();
        let mut radius = 2.; // Start with radius that covers most of the interesting area

        println!("\nSearching for interesting spot...");
        println!(
            "Starting at: [{}, {}], radius: {}",
            center.re, center.im, radius
        );

        let mut iteration = 0;
        loop {
            iteration += 1;

            if let Some((new_center, max_iterations)) = self.find_interesting_spot(center, radius) {
                center = new_center;
                radius /= 2.0;

                println!(
                    "Iteration {}: Center=[{}, {}], Radius={}, Max iterations={}",
                    iteration, center.re, center.im, radius, max_iterations
                );

                // Check if we're hitting MAX_ITERATIONS - means we can't distinguish points anymore
                if max_iterations >= MAX_ITERATIONS as f64 {
                    println!(
                        "Hit MAX_ITERATIONS limit - stopping search at iteration {}",
                        iteration
                    );
                    break;
                }

                // Check convergence after updating
                // If radius is so small that adding it to center doesn't change either component,
                // we've reached floating-point precision limits
                if center.re + radius == center.re && center.im + radius == center.im {
                    println!(
                        "Converged at floating-point precision limit after {} iterations",
                        iteration
                    );
                    break;
                }
            } else {
                println!("Search failed at iteration {}", iteration);
                break;
            }
        }

        // Update the buffer's center to the found location
        self.center = CircleF7E3::from((ScalarF7E3::from(center.re), ScalarF7E3::from(center.im)));

        // Set zoom based on the radius (which represents the search area size)
        // The zoom should be such that the radius corresponds to a reasonable view
        self.zoom = CircleF7E3::from(radius);

        println!("\nSearch complete! Final location:");
        println!("  Center: [{}, {}]", center.re, center.im);

        self.transform_requested = true;
    }

}

#[cfg(test)]
mod viewport_tests {
    use super::*;

    fn c7(c: Complex64) -> CircleF7E3 {
        CircleF7E3::from((ScalarF7E3::from(c.re), ScalarF7E3::from(c.im)))
    }

    /// The spirix and f64 screen mappings must agree when fed the same viewport.
    #[test]
    fn screen_to_complex_matches_f64() {
        let (w, h) = (1280., 800.);
        let views = [
            (Complex64::new(-0.75, 0.0), Complex64::new(1.0, 0.0)),
            (Complex64::new(-0.743643887, 0.131825904), Complex64::new(3e-6, 1e-6)),
            (Complex64::new(0.25, -0.5), Complex64::new(0.0, 2.0)),
        ];
        for (center, zoom) in views {
            for &(x, y) in &[(0., 0.), (w, h), (w / 2., h / 2.), (100.5, 700.25)] {
                let s = MandelbrotBuffer::screen_to_complex(x, y, w, h, c7(zoom), c7(center));
                let f = MandelbrotBuffer::screen_to_complex_f64(x, y, w, h, zoom, center);
                let s: Complex64 = Complex64::new(s.r().into(), s.i().into());
                let tol = 1e-12 * zoom.norm();
                assert!((s - f).norm() <= tol, "({x},{y}) view {center}/{zoom}: spirix {s} vs f64 {f}");
            }
        }
    }

    /// Paths that move only the F7E3 viewport (click-to-center, the `S` search) must carry the
    /// f64 viewport along, in every precision mode, or switching modes jumps to a stale view.
    #[test]
    fn f64_viewport_follows_spirix_viewport() {
        for mode in [PrecisionMode::F64, PrecisionMode::Spirix { f: 4, e: 4 }] {
            let mut b = MandelbrotBuffer::new(64, 48, Complex64::new(-0.75, 0.0), Complex64::new(1.5, 0.0), Vec::new());
            b.current_precision_mode = mode;
            b.center = c7(Complex64::new(0.3, -0.4));
            b.zoom = c7(Complex64::new(0.01, 0.002));
            b.transform_buffer();
            assert_eq!(b.center_f64, Complex64::new(0.3, -0.4));
            assert_eq!(b.zoom_f64, Complex64::new(0.01, 0.002));
            assert_eq!(*b.shared_viewport_f64.lock().unwrap(), (b.zoom_f64, b.center_f64));
        }
    }

    /// After a navigation every carried sample must sit on the pixel its exact coordinate maps
    /// to in the new viewport; on a 2× zoom-in the carried samples stay sparse.
    #[test]
    fn warp_carries_pixels_to_their_new_positions() {
        let (w, h) = (96usize, 64usize);
        let mut b = MandelbrotBuffer::new(w, h, Complex64::new(-0.5, 0.1), Complex64::new(0.8, 0.0), Vec::new());
        b.current_precision_mode = PrecisionMode::Perturbed;
        let n = w * h;
        // Fill as a finished frame: every pixel tagged, colour = its own index, position its centre.
        let mut truth: Vec<CircleF7E3> = Vec::with_capacity(n);
        let fill = |b: &MandelbrotBuffer, truth: &mut Vec<CircleF7E3>| {
            let mut px = b.shared_display_buffer.lock().unwrap();
            let mut po = b.shared_positions.lock().unwrap();
            truth.clear();
            for y in 0..h {
                for x in 0..w {
                    let c = MandelbrotBuffer::screen_to_complex(x as f64 + 0.5, y as f64 + 0.5, w as f64, h as f64, b.zoom, b.center);
                    px[y * w + x] = 0x0100_0000 | ((y * w + x) as u32 & 0xFFFFFF);
                    po[y * w + x] = [x as f32 + 0.5, y as f32 + 0.5];
                    truth.push(c);
                }
            }
        };
        fill(&b, &mut truth);
        for (dc, dz) in [(Complex64::new(0.05, -0.03), Complex64::new(1.0, 0.0)), (Complex64::new(0.0, 0.0), Complex64::new(0.5, 0.0)), (Complex64::new(0.01, 0.02), Complex64::new(0.9, 0.3))] {
            let old_zoom = b.zoom;
            b.center = b.center + c7(dc);
            b.zoom = old_zoom * c7(dz);
            b.transform_buffer();
            {
                let px = b.shared_display_buffer.lock().unwrap();
                let po = b.shared_positions.lock().unwrap();
                let zoom_c: Complex64 = b.zoom.into();
                let pixel = zoom_c.norm() * 2.0 / ((w * w + h * h) as f64).sqrt();
                let mut covered = 0;
                for i in 0..n {
                    if px[i] >> 24 == 0 {
                        continue;
                    }
                    covered += 1;
                    let src = (px[i] & 0xFFFFFF) as usize;
                    let (x, y) = (i % w, i / w);
                    // The sample's true coordinate must lie on this pixel in the new view …
                    let want: Complex64 = MandelbrotBuffer::screen_to_complex(x as f64 + 0.5, y as f64 + 0.5, w as f64, h as f64, b.zoom, b.center).into();
                    let got: Complex64 = truth[src].into();
                    assert!((got - want).norm() <= pixel * 1.0, "pixel ({x},{y}) after {dc}/{dz}: sample {got} expected near {want}");
                    // … and its stored position must agree with that coordinate.
                    let at: Complex64 = MandelbrotBuffer::screen_to_complex(po[i][0] as f64, po[i][1] as f64, w as f64, h as f64, b.zoom, b.center).into();
                    assert!((got - at).norm() <= pixel * 1e-3, "pixel ({x},{y}) position drift {}", (got - at).norm() / pixel);
                }
                assert!(covered > n / 8, "only {covered} pixels covered after {dc}/{dz}");
                // A 2× zoom-in must leave scattered samples (≈ a quarter of the pixels), never a
                // block-magnified copy: the carried picture is point-exact and stochastic.
                if dz == Complex64::new(0.5, 0.0) {
                    assert!(covered < n * 3 / 10, "zoom-in filled {covered} of {n} pixels: block magnification");
                }
            }
            fill(&b, &mut truth);
        }

        // A size change with the same centre and zoom must carry samples to where the new
        // framing puts the same coordinates (resize path, thread stopped).
        let (nw, nh) = (128usize, 40usize);
        let (z, c) = (b.zoom, b.center);
        b.width = nw;
        b.height = nh;
        b.reproject_epoch_buffers(w, h, z, c);
        let px = b.shared_display_buffer.lock().unwrap();
        let po = b.shared_positions.lock().unwrap();
        assert_eq!(px.len(), nw * nh);
        let zoom_c: Complex64 = b.zoom.into();
        let pixel = zoom_c.norm() * 2.0 / ((nw * nw + nh * nh) as f64).sqrt();
        let mut covered = 0;
        for i in 0..nw * nh {
            if px[i] >> 24 == 0 {
                continue;
            }
            covered += 1;
            let src = (px[i] & 0xFFFFFF) as usize;
            let (x, y) = (i % nw, i / nw);
            let want: Complex64 = MandelbrotBuffer::screen_to_complex(x as f64 + 0.5, y as f64 + 0.5, nw as f64, nh as f64, b.zoom, b.center).into();
            let got: Complex64 = truth[src].into();
            assert!((got - want).norm() <= pixel, "resized pixel ({x},{y}): sample {got} expected near {want}");
            let at: Complex64 = MandelbrotBuffer::screen_to_complex(po[i][0] as f64, po[i][1] as f64, nw as f64, nh as f64, b.zoom, b.center).into();
            assert!((got - at).norm() <= pixel * 1e-3, "resized pixel ({x},{y}) position drift");
        }
        assert!(covered > nw * nh / 4, "only {covered} of {} pixels carried through the resize", nw * nh);
    }
}
