use num_complex::Complex;
use num_complex::Complex64;
use spirix::{CircleF7E3, ScalarF7E3};
use std::f64::NAN;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::compute::mandelbrot_f64::MandelbrotResult;
use crate::compute::{
    calculate_colour, calculate_mandelbrot_point_f32, calculate_mandelbrot_point_f64,
    spirix_point_fn, SpirixPointFn,
};

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
    /// spirix grid cell: fraction/exponent width selectors, 3..=7 each (2^f / 2^e bits).
    Spirix { f: u8, e: u8 },
}

impl PrecisionMode {
    pub fn label(&self) -> String {
        match self {
            PrecisionMode::F32 => "f32".into(),
            PrecisionMode::F64 => "f64".into(),
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
    coord: CircleF7E3, // For transform_buffer
}

pub struct MandelbrotBuffer {
    pub shared_display_buffer: Arc<Mutex<Vec<u32>>>, // Shared ARGB u32 display buffer
    // F7E5 pipeline
    shared_coordinates: Arc<Mutex<Vec<CircleF7E3>>>,
    shared_viewport: Arc<Mutex<(CircleF7E3, CircleF7E3)>>, // (zoom, center) for calculation thread
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
            shared_coordinates: Arc::new(Mutex::new(vec![
                CircleF7E3::from((
                    ScalarF7E3::from(std::f64::NAN),
                    ScalarF7E3::from(std::f64::NAN)
                ));
                size
            ])),
            shared_viewport: Arc::new(Mutex::new((zoom_f7e5, center_f7e5))),
            center: center_f7e5,
            zoom: zoom_f7e5,
            old_center: CircleF7E3::from((ScalarF7E3::from(0.0), ScalarF7E3::from(0.0))),
            old_zoom: CircleF7E3::from((ScalarF7E3::from(1.0), ScalarF7E3::from(0.0))),
            shared_coordinates_f64: Arc::new(Mutex::new(vec![
                Complex64::new(f64::NAN, f64::NAN);
                size
            ])),
            shared_viewport_f64: Arc::new(Mutex::new((zoom_f64, center_f64))),
            calculation_buffer_f64: Arc::new(Mutex::new(Vec::new())),
            paused_flag: Arc::new(AtomicBool::new(false)),
            center_f64,
            zoom_f64,
            old_center_f64: Complex64::new(0.0, 0.0),
            old_zoom_f64: Complex64::new(1.0, 0.0),
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
        match self.current_precision_mode {
            PrecisionMode::F32 | PrecisionMode::F64 => self.transform_buffer_f64(),
            PrecisionMode::Spirix { .. } => self.transform_buffer_spirix(),
        }
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

        // DON'T clear buffer - pixels stay visible until naturally overwritten
        // Only clear on resize (handled by resize_buffers)

        self.old_width = self.width;
        self.old_height = self.height;
        self.old_center = self.center;
        self.old_zoom = self.zoom;
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

        // Navigation limit. Coordinates are pinned at F7E3, where the 128-bit fraction's
        // precision floor and the 8-bit exponent's range saturate together near 2^±126 —
        // clamp the zoom magnitude comfortably inside that so the viewport (and the zoom
        // readout) never leaves the coordinate type's normal range.
        let prospective = self.zoom_f64.norm() / scale;
        if !(2f64.powi(-120)..=2f64.powi(120)).contains(&prospective) {
            return;
        }

        // Update F7E5 viewport
        let cursor_coordinates = Self::screen_to_complex(
            cursor_pos.0,
            cursor_pos.1,
            self.width as f64,
            self.height as f64,
            self.zoom,
            self.center,
        );
        let scale_f7e5 = ScalarF7E3::from(scale);
        self.zoom = self.zoom / CircleF7E3::from((scale_f7e5, ScalarF7E3::ZERO));
        self.center = cursor_coordinates + (self.center - cursor_coordinates) / scale_f7e5;

        // Update f64 viewport
        let cursor_coordinates_f64 = Self::screen_to_complex_f64(
            cursor_pos.0,
            cursor_pos.1,
            self.width as f64,
            self.height as f64,
            self.zoom_f64,
            self.center_f64,
        );
        self.zoom_f64 = self.zoom_f64 / scale;
        self.center_f64 =
            cursor_coordinates_f64 + (self.center_f64 - cursor_coordinates_f64) / scale;

        let zoom_mag =
            (self.zoom_f64.re * self.zoom_f64.re + self.zoom_f64.im * self.zoom_f64.im).sqrt();
        println!(
            "Viewport: Center=[{}, {}], Zoom=[{}, {}] (magnitude: 2^{})",
            self.center_f64.re,
            self.center_f64.im,
            self.zoom_f64.re,
            self.zoom_f64.im,
            zoom_mag.log2()
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

        // Update f64 viewport
        let cursor_coordinates_old_f64 = Self::screen_to_complex_f64(
            cursor_pos.0,
            cursor_pos.1,
            self.width as f64,
            self.height as f64,
            self.zoom_f64,
            self.center_f64,
        );

        let rotation_f64 = Complex64::new(rotation_delta.cos(), rotation_delta.sin());
        self.zoom_f64 = self.zoom_f64 * rotation_f64;
        self.center_f64 = cursor_coordinates_old_f64
            + (self.center_f64 - cursor_coordinates_old_f64) * rotation_f64;

        let cursor_coordinates_new_f64 = Self::screen_to_complex_f64(
            cursor_pos.0,
            cursor_pos.1,
            self.width as f64,
            self.height as f64,
            self.zoom_f64,
            self.center_f64,
        );

        self.center_f64 =
            self.center_f64 + (cursor_coordinates_old_f64 - cursor_coordinates_new_f64);

        let zoom_mag =
            (self.zoom_f64.re * self.zoom_f64.re + self.zoom_f64.im * self.zoom_f64.im).sqrt();
        println!(
            "Viewport: Center=[{}, {}], Zoom=[{}, {}] (magnitude: 2^{})",
            self.center_f64.re,
            self.center_f64.im,
            self.zoom_f64.re,
            self.zoom_f64.im,
            zoom_mag.log2()
        );
    }

    pub fn start_spirix_iterations(&mut self, point_fn: SpirixPointFn) {
        self.stop_calculation();

        self.stop_flag.store(false, Ordering::SeqCst);

        let width = self.width;
        let height = self.height;
        let zoom = self.zoom;
        let center = self.center;
        let shared_display_buffer: Arc<Mutex<Vec<u32>>> = Arc::clone(&self.shared_display_buffer);
        let shared_coordinates: Arc<Mutex<Vec<CircleF7E3>>> = Arc::clone(&self.shared_coordinates);
        let stop_flag: Arc<AtomicBool> = Arc::clone(&self.stop_flag);
        let current_epoch: Arc<AtomicU8> = Arc::clone(&self.current_epoch);

        // Shuffle indices (one-time cost)
        let mut indices: Vec<usize> = (0..width * height).collect();
        indices.shuffle(&mut thread_rng());

        self.calculation_thread = Some(thread::spawn(move || {
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

                    batch.push((idx, coord));
                }

                // If we've processed all indices, reshuffle and restart
                if index_cursor >= indices.len() {
                    indices.shuffle(&mut thread_rng());
                    index_cursor = 0;
                }

                // PHASE 2: Parallel calculation (all threads finish their work)
                let results: Vec<CalculationResult> = batch
                    .par_iter()
                    .map(|&(idx, coord)| {
                        let result = point_fn(coord, MAX_ITERATIONS);
                        let colour = calculate_colour(coord.into(), &result, zoom.into());

                        CalculationResult {
                            idx,
                            x: idx % width,
                            y: idx / width,
                            color: 0xFF000000
                                | (colour[0] as u32) << 16
                                | (colour[1] as u32) << 8
                                | colour[2] as u32,
                            coord,
                        }
                    })
                    .collect();

                // PHASE 3: Write + interpolate (single lock per frame)
                if let Ok(mut pixels) = shared_display_buffer.lock() {
                    if let Ok(mut coords) = shared_coordinates.lock() {
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
                            coords[result.idx] = result.coord;
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
        match self.current_precision_mode {
            PrecisionMode::F32 => self.start_ieee_iterations(calculate_mandelbrot_point_f32),
            PrecisionMode::F64 => self.start_ieee_iterations(calculate_mandelbrot_point_f64),
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
        self.width = new_width;
        self.height = new_height;

        let new_size = new_width * new_height;

        // Clear calculation buffer on resize (reprojection is too slow for large buffers)
        // TODO: Optimize reprojection or use spatial indexing for faster filtering
        self.calculation_buffer_f64.lock().unwrap().clear();

        // Reallocate display buffer
        *self.shared_display_buffer.lock().unwrap() = vec![0x00000000; new_size];

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
