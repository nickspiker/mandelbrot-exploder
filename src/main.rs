use num_complex::*;
use pixels::{Error, Pixels, SurfaceTexture};
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::f64::NAN;
use std::fs::File;
use std::io::Write;
use std::ops::Mul;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

const MAX_ITERATIONS: u64 = 1 << 24;
const INITIAL_WIDTH: usize = 0x800;
const INITIAL_HEIGHT: usize = 0x400;
struct MandelbrotBuffer {
    pixel_buffer: Vec<u8>,
    shared_colours: Arc<Mutex<Vec<u8>>>,
    shared_coordinates: Arc<Mutex<Vec<Complex64>>>,
    width: usize,
    height: usize,
    center: Complex64,
    zoom: Complex64,
    old_width: usize,
    old_height: usize,
    old_center: Complex64,
    old_zoom: Complex64,
    stop_flag: Arc<AtomicBool>,
    calculation_thread: Option<thread::JoinHandle<()>>,
    shift_held: bool,
    transform_requested: bool,
    boundary_path: Vec<Complex64>,
}

impl MandelbrotBuffer {
    fn new(
        width: usize,
        height: usize,
        center: Complex64,
        zoom: Complex64,
        boundary_path: Vec<Complex64>,
    ) -> Self {
        let size = width * height;
        let buffer = Self {
            pixel_buffer: vec![0; size * 4],
            shared_colours: Arc::new(Mutex::new(vec![0; size * 4])),
            shared_coordinates: Arc::new(Mutex::new(vec![
                Complex64::new(
                    std::f64::NAN,
                    std::f64::NAN
                );
                size
            ])),
            width,
            height,
            center,
            zoom,
            old_width: width,
            old_height: height,
            old_center: Complex64::new(0., 0.),
            old_zoom: Complex64::new(1., 0.),
            stop_flag: Arc::new(AtomicBool::new(false)),
            calculation_thread: None,
            shift_held: false,
            transform_requested: false,
            boundary_path,
        };
        buffer
    }

    fn interpolate_pixels(&mut self) {
        for x in 3..self.width - 3 {
            for y in 3..self.height - 3 {
                let idx = y * self.width * 4 + x * 4;

                // Skip if this pixel is already set
                if self.pixel_buffer[idx + 3] > 253 {
                    continue;
                }

                if self.pixel_buffer[idx - 4 + 3] == 255 {
                    self.pixel_buffer[idx] = self.pixel_buffer[idx - 4];
                    self.pixel_buffer[idx + 1] = self.pixel_buffer[idx - 4 + 1];
                    self.pixel_buffer[idx + 2] = self.pixel_buffer[idx - 4 + 2];
                    self.pixel_buffer[idx + 3] = 254;
                } else if self.pixel_buffer[idx + 4 + 3] == 255 {
                    self.pixel_buffer[idx] = self.pixel_buffer[idx + 4];
                    self.pixel_buffer[idx + 1] = self.pixel_buffer[idx + 4 + 1];
                    self.pixel_buffer[idx + 2] = self.pixel_buffer[idx + 4 + 2];
                    self.pixel_buffer[idx + 3] = 254;
                } else if self.pixel_buffer[idx - self.width * 4 + 3] == 255 {
                    self.pixel_buffer[idx] = self.pixel_buffer[idx - self.width * 4];
                    self.pixel_buffer[idx + 1] = self.pixel_buffer[idx - self.width * 4 + 1];
                    self.pixel_buffer[idx + 2] = self.pixel_buffer[idx - self.width * 4 + 2];
                    self.pixel_buffer[idx + 3] = 254;
                } else if self.pixel_buffer[idx + self.width * 4 + 3] == 255 {
                    self.pixel_buffer[idx] = self.pixel_buffer[idx + self.width * 4];
                    self.pixel_buffer[idx + 1] = self.pixel_buffer[idx + self.width * 4 + 1];
                    self.pixel_buffer[idx + 2] = self.pixel_buffer[idx + self.width * 4 + 2];
                    self.pixel_buffer[idx + 3] = 254;
                } else if self.pixel_buffer[idx - self.width * 4 - 4 + 3] == 255 {
                    self.pixel_buffer[idx] = self.pixel_buffer[idx - self.width * 4 - 4];
                    self.pixel_buffer[idx + 1] = self.pixel_buffer[idx - self.width * 4 - 4 + 1];
                    self.pixel_buffer[idx + 2] = self.pixel_buffer[idx - self.width * 4 - 4 + 2];
                    self.pixel_buffer[idx + 3] = 254;
                } else if self.pixel_buffer[idx - self.width * 4 + 4 + 3] == 255 {
                    self.pixel_buffer[idx] = self.pixel_buffer[idx - self.width * 4 + 4];
                    self.pixel_buffer[idx + 1] = self.pixel_buffer[idx - self.width * 4 + 4 + 1];
                    self.pixel_buffer[idx + 2] = self.pixel_buffer[idx - self.width * 4 + 4 + 2];
                    self.pixel_buffer[idx + 3] = 254;
                } else if self.pixel_buffer[idx + self.width * 4 - 4 + 3] == 255 {
                    self.pixel_buffer[idx] = self.pixel_buffer[idx + self.width * 4 - 4];
                    self.pixel_buffer[idx + 1] = self.pixel_buffer[idx + self.width * 4 - 4 + 1];
                    self.pixel_buffer[idx + 2] = self.pixel_buffer[idx + self.width * 4 - 4 + 2];
                    self.pixel_buffer[idx + 3] = 254;
                } else if self.pixel_buffer[idx + self.width * 4 + 4 + 3] == 255 {
                    self.pixel_buffer[idx] = self.pixel_buffer[idx + self.width * 4 + 4];
                    self.pixel_buffer[idx + 1] = self.pixel_buffer[idx + self.width * 4 + 4 + 1];
                    self.pixel_buffer[idx + 2] = self.pixel_buffer[idx + self.width * 4 + 4 + 2];
                    self.pixel_buffer[idx + 3] = 254;
                } else if self.pixel_buffer[idx - 8 + 3] == 255 {
                    self.pixel_buffer[idx] = self.pixel_buffer[idx - 8];
                    self.pixel_buffer[idx + 1] = self.pixel_buffer[idx - 8 + 1];
                    self.pixel_buffer[idx + 2] = self.pixel_buffer[idx - 8 + 2];
                    self.pixel_buffer[idx + 3] = 254;
                } else if self.pixel_buffer[idx + 8 + 3] == 255 {
                    self.pixel_buffer[idx] = self.pixel_buffer[idx + 8];
                    self.pixel_buffer[idx + 1] = self.pixel_buffer[idx + 8 + 1];
                    self.pixel_buffer[idx + 2] = self.pixel_buffer[idx + 8 + 2];
                    self.pixel_buffer[idx + 3] = 254;
                } else if self.pixel_buffer[idx - self.width * 8 + 3] == 255 {
                    self.pixel_buffer[idx] = self.pixel_buffer[idx - self.width * 8];
                    self.pixel_buffer[idx + 1] = self.pixel_buffer[idx - self.width * 8 + 1];
                    self.pixel_buffer[idx + 2] = self.pixel_buffer[idx - self.width * 8 + 2];
                    self.pixel_buffer[idx + 3] = 254;
                } else if self.pixel_buffer[idx + self.width * 8 + 3] == 255 {
                    self.pixel_buffer[idx] = self.pixel_buffer[idx + self.width * 8];
                    self.pixel_buffer[idx + 1] = self.pixel_buffer[idx + self.width * 8 + 1];
                    self.pixel_buffer[idx + 2] = self.pixel_buffer[idx + self.width * 8 + 2];
                    self.pixel_buffer[idx + 3] = 254;
                } else if self.pixel_buffer[idx - 8 - self.width * 4 + 3] == 255 {
                    self.pixel_buffer[idx] = self.pixel_buffer[idx - 8 - self.width * 4];
                    self.pixel_buffer[idx + 1] = self.pixel_buffer[idx - 8 - self.width * 4 + 1];
                    self.pixel_buffer[idx + 2] = self.pixel_buffer[idx - 8 - self.width * 4 + 2];
                    self.pixel_buffer[idx + 3] = 254;
                } else if self.pixel_buffer[idx - 8 + self.width * 4 + 3] == 255 {
                    self.pixel_buffer[idx] = self.pixel_buffer[idx - 8 + self.width * 4];
                    self.pixel_buffer[idx + 1] = self.pixel_buffer[idx - 8 + self.width * 4 + 1];
                    self.pixel_buffer[idx + 2] = self.pixel_buffer[idx - 8 + self.width * 4 + 2];
                    self.pixel_buffer[idx + 3] = 254;
                } else if self.pixel_buffer[idx + 8 - self.width * 4 + 3] == 255 {
                    self.pixel_buffer[idx] = self.pixel_buffer[idx + 8 - self.width * 4];
                    self.pixel_buffer[idx + 1] = self.pixel_buffer[idx + 8 - self.width * 4 + 1];
                    self.pixel_buffer[idx + 2] = self.pixel_buffer[idx + 8 - self.width * 4 + 2];
                    self.pixel_buffer[idx + 3] = 254;
                } else if self.pixel_buffer[idx + 8 + self.width * 4 + 3] == 255 {
                    self.pixel_buffer[idx] = self.pixel_buffer[idx + 8 + self.width * 4];
                    self.pixel_buffer[idx + 1] = self.pixel_buffer[idx + 8 + self.width * 4 + 1];
                    self.pixel_buffer[idx + 2] = self.pixel_buffer[idx + 8 + self.width * 4 + 2];
                    self.pixel_buffer[idx + 3] = 254;
                } else if self.pixel_buffer[idx - 4 - self.width * 8 + 3] == 255 {
                    self.pixel_buffer[idx] = self.pixel_buffer[idx - 4 - self.width * 8];
                    self.pixel_buffer[idx + 1] = self.pixel_buffer[idx - 4 - self.width * 8 + 1];
                    self.pixel_buffer[idx + 2] = self.pixel_buffer[idx - 4 - self.width * 8 + 2];
                    self.pixel_buffer[idx + 3] = 254;
                } else if self.pixel_buffer[idx - 4 + self.width * 8 + 3] == 255 {
                    self.pixel_buffer[idx] = self.pixel_buffer[idx - 4 + self.width * 8];
                    self.pixel_buffer[idx + 1] = self.pixel_buffer[idx - 4 + self.width * 8 + 1];
                    self.pixel_buffer[idx + 2] = self.pixel_buffer[idx - 4 + self.width * 8 + 2];
                    self.pixel_buffer[idx + 3] = 254;
                } else if self.pixel_buffer[idx + 4 - self.width * 8 + 3] == 255 {
                    self.pixel_buffer[idx] = self.pixel_buffer[idx + 4 - self.width * 8];
                    self.pixel_buffer[idx + 1] = self.pixel_buffer[idx + 4 - self.width * 8 + 1];
                    self.pixel_buffer[idx + 2] = self.pixel_buffer[idx + 4 - self.width * 8 + 2];
                    self.pixel_buffer[idx + 3] = 254;
                } else if self.pixel_buffer[idx + 4 + self.width * 8 + 3] == 255 {
                    self.pixel_buffer[idx] = self.pixel_buffer[idx + 4 + self.width * 8];
                    self.pixel_buffer[idx + 1] = self.pixel_buffer[idx + 4 + self.width * 8 + 1];
                    self.pixel_buffer[idx + 2] = self.pixel_buffer[idx + 4 + self.width * 8 + 2];
                    self.pixel_buffer[idx + 3] = 254;
                }
            }
        }
    }

    fn transform_buffer(&mut self) {
        let old_colors = self.shared_colours.lock().unwrap().clone();
        let old_coords = self.shared_coordinates.lock().unwrap().clone();
        let mut new_colors = vec![0; self.width * self.height * 4];
        let mut new_coords =
            vec![Complex64::new(std::f64::NAN, std::f64::NAN); self.width * self.height];
        let scale = ((self.width * self.width + self.height * self.height) as f64).sqrt();

        for y in 0..self.old_height {
            for x in 0..self.old_width {
                let old_coord_idx = y * self.old_width + x;
                let old_colour_idx = old_coord_idx * 4;

                if old_colors[old_colour_idx + 3] == 255 {
                    let coord = old_coords[old_coord_idx];

                    let (new_x, new_y) = Self::complex_to_screen(
                        coord,
                        self.width as f64,
                        self.height as f64,
                        self.zoom,
                        self.center,
                        scale,
                    );

                    let new_xi = new_x.floor() as isize;
                    let new_yi = new_y.floor() as isize;

                    if new_xi >= 0
                        && new_xi < self.width as isize
                        && new_yi >= 0
                        && new_yi < self.height as isize
                    {
                        let new_coord_idx = new_yi as usize * self.width + new_xi as usize;
                        let new_colour_idx = new_coord_idx * 4;
                        new_colors[new_colour_idx] = old_colors[old_colour_idx];
                        new_colors[new_colour_idx + 1] = old_colors[old_colour_idx + 1];
                        new_colors[new_colour_idx + 2] = old_colors[old_colour_idx + 2];
                        new_colors[new_colour_idx + 3] = 255;
                        new_coords[new_coord_idx] = coord;
                    }
                }
            }
        }

        *self.shared_colours.lock().unwrap() = new_colors.clone();
        *self.shared_coordinates.lock().unwrap() = new_coords;
        self.pixel_buffer = new_colors;
        self.old_width = self.width;
        self.old_height = self.height;
        self.old_center = self.center;
        self.old_zoom = self.zoom;
    }

    fn screen_to_complex(
        x: f64,
        y: f64,
        width: f64,
        height: f64,
        zoom: Complex64,
        center: Complex64,
    ) -> Complex64 {
        let scale = 1. / (width * width + height * height).sqrt();

        let mut re = (x - width / 2.) * scale;
        let mut im = (y - height / 2.) * scale;

        re = re.tan() * 2.;
        im = im.tan() * 2.;

        (Complex64::new(re, im) / zoom) + center
    }
    fn complex_to_screen(
        c: Complex64,
        width: f64,
        height: f64,
        zoom: Complex64,
        center: Complex64,
        scale: f64,
    ) -> (f64, f64) {
        let c_transformed = (c - center) * zoom;
        let mut re = c_transformed.re;
        let mut im = c_transformed.im;

        re = (re * 0.5).atan();
        im = (im * 0.5).atan();

        let x = re * scale + width / 2.;
        let y = im * scale + height / 2.;
        (x, y)
    }

    fn apply_zoom(&mut self, cursor_pos: PhysicalPosition<f64>, zoom_delta: f64) {
        let cursor_coordinates = Self::screen_to_complex(
            cursor_pos.x,
            cursor_pos.y,
            self.width as f64,
            self.height as f64,
            self.zoom,
            self.center,
        );

        let zoom_factor = if zoom_delta > 0. { 7. / 6. } else { 6. / 7. };
        let scale = zoom_factor.powf(zoom_delta.abs());

        // Update zoom and center (relative to cursor position)
        self.zoom *= Complex64::new(scale, 0.);
        self.center = cursor_coordinates + (self.center - cursor_coordinates) / scale;
    }

    fn apply_rotation(&mut self, cursor_pos: PhysicalPosition<f64>, rotation_delta: f64) {
        let cursor_coordinates_old = Self::screen_to_complex(
            cursor_pos.x,
            cursor_pos.y,
            self.width as f64,
            self.height as f64,
            self.zoom,
            self.center,
        );

        let rotation = Complex64::new(rotation_delta.cos(), rotation_delta.sin());

        self.zoom *= rotation;
        self.center = cursor_coordinates_old + (self.center - cursor_coordinates_old) * rotation;

        let cursor_coordinates_new = Self::screen_to_complex(
            cursor_pos.x,
            cursor_pos.y,
            self.width as f64,
            self.height as f64,
            self.zoom,
            self.center,
        );

        // Adjust center by how far cursor moved due to rotation
        self.center += cursor_coordinates_old - cursor_coordinates_new;
    }

    fn start_z_iterations(&mut self) {
        self.stop_calculation();

        self.stop_flag.store(false, Ordering::SeqCst);

        let width = self.width;
        let height = self.height;
        let zoom = self.zoom;
        let center = self.center;
        let shared_colours = Arc::clone(&self.shared_colours);
        let shared_coordinates = Arc::clone(&self.shared_coordinates);
        let stop_flag = Arc::clone(&self.stop_flag);

        let mut indices: Vec<usize> = (0..width * height).collect();
        indices.shuffle(&mut thread_rng());

        self.calculation_thread = Some(thread::spawn(move || {
            indices.par_iter().for_each(|&idx| {
                if stop_flag.load(Ordering::SeqCst) {
                    return;
                }

                let should_calculate = {
                    match shared_colours.lock() {
                        Ok(pixels) => pixels[idx * 4 + 3] != 255,
                        Err(poisoned) => poisoned.get_ref()[idx * 4 + 3] != 255,
                    }
                };

                if !should_calculate {
                    return;
                }

                let mut rng = rand::thread_rng();
                let x = (idx % width) as f64 + rng.gen::<f64>();
                let y = (idx / width) as f64 + rng.gen::<f64>();
                let point =
                    Self::screen_to_complex(x, y, width as f64, height as f64, zoom, center);

                let (period, escape_velocity, coords, flames, slope, _multiplier) =
                    calculate_mandelbrot_point(point, MAX_ITERATIONS);

                let colour =
                    calculate_colour(point, period, escape_velocity, coords, flames, slope, zoom);

                if let Ok(mut pixels) = shared_colours.lock().or_else(|poisoned| {
                    Ok::<
                        std::sync::MutexGuard<'_, Vec<u8>>,
                        std::sync::PoisonError<std::sync::MutexGuard<'_, Vec<u8>>>,
                    >(poisoned.into_inner())
                }) {
                    let pixel_idx = idx * 4;
                    pixels[pixel_idx] = colour[0]; // Line 382
                    pixels[pixel_idx + 1] = colour[1];
                    pixels[pixel_idx + 2] = colour[2];
                    pixels[pixel_idx + 3] = 255;
                }

                // Fixed error type specification
                if let Ok(mut coords) = shared_coordinates.lock().or_else(|poisoned| {
                    Ok::<
                        std::sync::MutexGuard<'_, Vec<Complex64>>,
                        std::sync::PoisonError<std::sync::MutexGuard<'_, Vec<Complex64>>>,
                    >(poisoned.into_inner())
                }) {
                    coords[idx] = point;
                }
            });
        }));
    }

    fn stop_calculation(&mut self) {
        self.stop_flag.store(true, Ordering::SeqCst);
    }

    fn format_point_statistics(
        point: Complex64,
        zoom: Complex64,
        period: u64,
        escape_velocity: f64,
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
                 Escape velocity: {}\n\
                 Coordinates: [{}, {}]\n\
                 Flame coords: [{}, {}]\n\
                 Slope: [{}, {}]\n\
                 Multiplier: [{}, {}]",
                point.re,
                point.im,
                zoom.norm().log2(),
                zoom.re,
                zoom.im,
                escape_velocity,
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
    fn plot_boundary_path(&mut self) {
        let scale = ((self.width * self.width + self.height * self.height) as f64).sqrt();
        let mut visible_points = Vec::new();
        let mut prev_xi = std::isize::MIN;
        let mut prev_yi = std::isize::MIN;
        let mut skipped = false;
        for idx in 0..self.boundary_path.len() {
            let (x, y) = Self::complex_to_screen(
                self.boundary_path[idx],
                self.width as f64,
                self.height as f64,
                self.zoom,
                self.center,
                scale,
            );

            let xi = x.floor() as isize;
            let yi = y.floor() as isize;

            if prev_xi != xi || prev_yi != yi {
                if xi >= 0 && xi < self.width as isize && yi >= 0 && yi < self.height as isize {
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
                draw_line_u8(
                    &mut self.pixel_buffer,
                    self.width,
                    x0 as f32,
                    y0 as f32,
                    x1 as f32,
                    y1 as f32,
                    [63, 63, 63],
                );
            }
        }
    }
}
fn draw_line_u8(
    image: &mut Vec<u8>,
    image_width: usize,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    colour: [u8; 3],
) {
    let steep = (y1 - y0).abs() > (x1 - x0).abs();
    let (mut x0, mut y0, mut x1, mut y1) = (x0, y0, x1, y1);

    if steep {
        std::mem::swap(&mut x0, &mut y0);
        std::mem::swap(&mut x1, &mut y1);
    }
    if x0 > x1 {
        std::mem::swap(&mut x0, &mut x1);
        std::mem::swap(&mut y0, &mut y1);
    }

    let dx = x1 - x0;
    let dy = y1 - y0;
    let gradient = if dx == 0.0 { 1.0 } else { dy / dx };

    let xend = (x0 + 0.5).floor();
    let yend = y0 + gradient * (xend - x0);
    let xgap = 1.0 - (x0 + 0.5).fract();
    let xpxl1 = xend;
    let ypxl1 = yend.floor();

    let mut plot = |x: isize, y: isize, alias: f32| {
        let channels = 4;
        if x >= 0
            && x < image_width as isize
            && y >= 0
            && y < (image.len() / (image_width * channels)) as isize
        {
            let idx = (y as usize) * image_width + (x as usize);
            if image[idx * channels + 3] < 254 {
                if image[idx * channels + 3] == 253 {
                    image[idx * channels] =
                        (image[idx * channels] as f32 * (1. - alias) + 256. * alias) as u8;
                    image[idx * channels + 1] =
                        (image[idx * channels + 1] as f32 * (1. - alias) + 256. * alias) as u8;
                    image[idx * channels + 2] =
                        (image[idx * channels + 2] as f32 * (1. - alias) + 256. * alias) as u8;
                } else {
                    image[idx * channels] = (alias * 256.) as u8;
                    image[idx * channels + 1] = (alias * 256.) as u8;
                    image[idx * channels + 2] = (alias * 256.) as u8;
                    image[idx * channels + 3] = 253;
                }
            } else {
                image[idx * channels] =
                    image[idx * channels].wrapping_add((colour[0] as f32 * alias) as u8);
                image[idx * channels + 1] =
                    image[idx * channels + 1].wrapping_add((colour[1] as f32 * alias) as u8);
                image[idx * channels + 2] =
                    image[idx * channels + 2].wrapping_add((colour[2] as f32 * alias) as u8);
            }
        }
    };

    if steep {
        plot(ypxl1 as isize, xpxl1 as isize, (1.0 - yend.fract()) * xgap);
        plot((ypxl1 + 1.0) as isize, xpxl1 as isize, yend.fract() * xgap);
    } else {
        plot(xpxl1 as isize, ypxl1 as isize, (1.0 - yend.fract()) * xgap);
        plot(xpxl1 as isize, (ypxl1 + 1.0) as isize, yend.fract() * xgap);
    }

    let mut intery = yend + gradient;

    let xend = (x1 + 0.5).floor();
    let yend = y1 + gradient * (xend - x1);
    let xgap = (x1 + 0.5).fract();
    let xpxl2 = xend;
    let ypxl2 = yend.floor();

    if steep {
        plot(ypxl2 as isize, xpxl2 as isize, (1.0 - yend.fract()) * xgap);
        plot((ypxl2 + 1.0) as isize, xpxl2 as isize, yend.fract() * xgap);
    } else {
        plot(xpxl2 as isize, ypxl2 as isize, (1.0 - yend.fract()) * xgap);
        plot(xpxl2 as isize, (ypxl2 + 1.0) as isize, yend.fract() * xgap);
    }

    if steep {
        for x in (xpxl1 as isize + 1)..(xpxl2 as isize) {
            plot(intery.floor() as isize, x, 1.0 - intery.fract());
            plot((intery.floor() + 1.0) as isize, x, intery.fract());
            intery += gradient;
        }
    } else {
        for x in (xpxl1 as isize + 1)..(xpxl2 as isize) {
            plot(x, intery.floor() as isize, 1.0 - intery.fract());
            plot(x, (intery.floor() + 1.0) as isize, intery.fract());
            intery += gradient;
        }
    }
}
fn compute_escape_path(src_point: Complex64) -> Vec<Complex64> {
    let mut path_outward = Vec::new();
    let mut point = src_point;
    let mut prev_point = Complex64::new(0., 0.);
    // Start with jitter below machine epsilon
    let mut jitter = 1. / (1u64 << 63) as f64;
    let mut rng = thread_rng();

    while point.is_finite() {
        let (_period, _escape_velocity, _coords, _flames, slope, _multiplier) =
            calculate_mandelbrot_point(point, MAX_ITERATIONS);

        point = point + Complex64::new(slope.re, -slope.im) / 256.;

        // If we hit the same point, we need to jitter
        // This invalidates the current path as it creates a discontinuity
        if point == prev_point {
            let normal = Normal::new(0., jitter).unwrap();
            point = point + Complex64::new(normal.sample(&mut rng), normal.sample(&mut rng));
            jitter = jitter * 2.; // Increase jitter for next time if needed
            path_outward = Vec::new(); // Reset path as we've had to jitter
            continue;
        }

        // Only record points that are finite and before any jittering
        if point.is_finite() {
            path_outward.push(point);
        }
        prev_point = point;
    }

    let mut path_inward = Vec::new();
    let mut point = src_point;
    let mut prev_point = Complex64::new(0., 0.);

    while point.is_finite() {
        let (_period, _escape_velocity, _coords, _flames, slope, _multiplier) =
            calculate_mandelbrot_point(point, MAX_ITERATIONS);

        point = point - Complex64::new(slope.re, -slope.im) / 256.;

        if point == prev_point {
            break;
        }

        if point.is_finite() {
            path_inward.push(point);
        }
        prev_point = point;
    }

    path_inward.reverse();
    path_outward.extend(path_inward);
    path_outward
}
fn roll_to_boundary() -> (Vec<Complex64>, Complex64) {
    let start_time = Instant::now();
    let duration = Duration::from_secs(1);
    let thread_count = rayon::current_num_threads();

    // Each thread maintains its own best path
    let thread_paths: Vec<_> = (0..thread_count)
        .into_par_iter()
        .map(|_| {
            let mut rng = thread_rng();
            let mut thread_best_path = Vec::new();

            // Keep trying new paths until time runs out or we reach the min path length
            while start_time.elapsed() < duration {
                // Start with maximum search width and reset tracking variables
                let mut gaussian_width = 256.;
                let mut prev_slope = std::f64::MAX;
                let mut point = Complex64::new(0., 0.);

                // Phase 1: Find a point outside the set
                // Reduce the search width each time we find a point with a smaller slope
                while gaussian_width > 1. / (1u64 << 63) as f64 && start_time.elapsed() < duration {
                    let normal = Normal::new(0., gaussian_width).unwrap();
                    let new_point =
                        Complex64::new(normal.sample(&mut rng), normal.sample(&mut rng)) + point;

                    let (period, _escape_velocity, _coords, _flames, slope, _multiplier) =
                        calculate_mandelbrot_point(new_point, MAX_ITERATIONS);

                    if point == new_point {
                        break;
                    }
                    // Period 0 means we're outside the set. Keep the point with the smallest slope
                    if period == 0 {
                        if slope.norm() < prev_slope {
                            gaussian_width = slope.norm(); // Reduce search width
                            prev_slope = slope.norm();
                            point = new_point;
                        }
                    } else {
                        break;
                    }
                }

                // Phase 2: Follow a path along the outside of the set using the gradient
                let mut path = Vec::new();
                let mut prev_point = Complex64::new(0., 0.);
                // Start with jitter below machine epsilon
                let mut jitter = 1. / (1u64 << 63) as f64;

                while point.is_finite() && start_time.elapsed() < duration {
                    let (_period, _escape_velocity, _coords, _flames, slope, _multiplier) =
                        calculate_mandelbrot_point(point, MAX_ITERATIONS);

                    point = point + Complex64::new(slope.re, -slope.im) / 256.;

                    // If we hit the same point, we need to jitter
                    // This invalidates the current path as it creates a discontinuity
                    if point == prev_point {
                        let normal = Normal::new(0., jitter).unwrap();
                        point = point
                            + Complex64::new(normal.sample(&mut rng), normal.sample(&mut rng));
                        jitter = jitter * 2.; // Increase jitter for next time if needed
                        path = Vec::new(); // Reset path as we've had to jitter
                        continue;
                    }

                    // Only record points that are finite and before any jittering
                    if point.is_finite() {
                        path.push(point);
                    }
                    prev_point = point;
                }

                // Update thread's best path if this one is longer
                if path.len() > thread_best_path.len() {
                    thread_best_path = path;
                }
            }

            thread_best_path
        })
        .collect();

    // Find the longest path among all threads
    let mut best_path = Vec::new();
    for path in thread_paths {
        if path.len() > best_path.len() {
            best_path = path;
        }
    }

    // println!("Path length: {}", (best_path.len() as f64).log2());
    // let mut file = File::create("path.csv").unwrap();
    // let file_string = best_path
    //     .iter()
    //     .map(|point| format!("{},{}\n", point.to_polar().0, point.to_polar().1))
    //     .collect::<String>();
    // file.write_all(file_string.as_bytes()).unwrap();

    // Calculate direction from the end of the path
    let mut sign = best_path[best_path.len() - 2] - best_path[best_path.len() - 3];
    sign = sign.norm() / sign;
    (best_path, sign * (1u64 << 56) as f64)
}

fn main() -> Result<(), Error> {
    // Initialize window and graphics
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Mandelbrot Exploder")
        .with_inner_size(LogicalSize::new(
            INITIAL_WIDTH as u32,
            INITIAL_HEIGHT as u32,
        ))
        .build(&event_loop)
        .unwrap();

    // Set up pixels buffer for rendering
    let window_size = window.inner_size();
    let mut pixels = {
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(window_size.width, window_size.height, surface_texture)?
    };

    // Initialize the Mandelbrot buffer with initial dimensions and parameters
    let width = window_size.width as usize;
    let height = window_size.height as usize;
    let (boundary_path, distance) = roll_to_boundary();

    let mut buffer = MandelbrotBuffer::new(
        width,
        height,
        boundary_path[1],
        distance / ((width * width + height * height) as f64).sqrt(),
        boundary_path,
    );
    buffer.start_z_iterations();

    // State tracking variables
    let mut cursor_pos = PhysicalPosition::new(0., 0.);
    let mut accumulated_zoom = 0.;
    let mut accumulated_rotation = 0.;
    let mut drag_start_pos: Option<PhysicalPosition<f64>> = None;
    let mut dragging = false;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            // Handle window close
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                buffer.stop_flag.store(true, Ordering::Relaxed);
                *control_flow = ControlFlow::Exit;
            }

            // Track cursor position and handle dragging
            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                cursor_pos = position;

                if let Some(start_pos) = drag_start_pos {
                    dragging = true;
                    // Get the start point in complex coordinates
                    let start_point = MandelbrotBuffer::screen_to_complex(
                        start_pos.x,
                        start_pos.y,
                        buffer.width as f64,
                        buffer.height as f64,
                        buffer.zoom,
                        buffer.center,
                    );

                    // Get the current point in complex coordinates
                    let current_point = MandelbrotBuffer::screen_to_complex(
                        position.x,
                        position.y,
                        buffer.width as f64,
                        buffer.height as f64,
                        buffer.zoom,
                        buffer.center,
                    );

                    // Move the center by the difference
                    buffer.center += start_point - current_point;
                    buffer.transform_requested = true;

                    // Update start position for next movement
                    drag_start_pos = Some(position);
                }
            }

            // Handle mouse wheel for zooming and rotation
            Event::WindowEvent {
                event: WindowEvent::MouseWheel { delta, .. },
                ..
            } => {
                let scroll_amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(pos) => pos.y / 50.,
                };

                // Shift + scroll rotates, normal scroll zooms
                if buffer.shift_held {
                    accumulated_rotation += scroll_amount * std::f64::consts::PI / 8.0;
                } else {
                    accumulated_zoom += scroll_amount;
                }
                buffer.transform_requested = true;
            }

            // Handle mouse clicks and dragging
            Event::WindowEvent {
                event:
                    WindowEvent::MouseInput {
                        state,
                        button: MouseButton::Left,
                        ..
                    },
                ..
            } => {
                match state {
                    ElementState::Pressed => {
                        drag_start_pos = Some(cursor_pos);
                        dragging = false; // Start false, will be set true on any movement
                    }
                    ElementState::Released => {
                        if !dragging {
                            // Only handle click behavior if we never dragged
                            let point = MandelbrotBuffer::screen_to_complex(
                                cursor_pos.x,
                                cursor_pos.y,
                                buffer.width as f64,
                                buffer.height as f64,
                                buffer.zoom,
                                buffer.center,
                            );

                            if buffer.shift_held {
                                // Shift + click: print point statistics
                                let (period, escape_velocity, coords, flames, slope, multiplier) =
                                    calculate_mandelbrot_point(point, MAX_ITERATIONS);

                                println!(
                                    "\n{}",
                                    MandelbrotBuffer::format_point_statistics(
                                        point,
                                        buffer.zoom,
                                        period,
                                        escape_velocity,
                                        coords,
                                        flames,
                                        slope,
                                        multiplier
                                    )
                                );
                            } else {
                                buffer.boundary_path = compute_escape_path(point);
                            }
                        }
                        drag_start_pos = None;
                        dragging = false;
                    }
                }
            }

            // Track shift key state
            Event::WindowEvent {
                event: WindowEvent::ModifiersChanged(modifiers),
                ..
            } => {
                buffer.shift_held = modifiers.shift();
            }

            // Handle window resizing
            Event::WindowEvent {
                event: WindowEvent::Resized(new_size),
                ..
            } => {
                // Update pixel buffer size
                pixels
                    .resize_surface(new_size.width, new_size.height)
                    .expect("Failed to resize surface");
                pixels
                    .resize_buffer(new_size.width, new_size.height)
                    .expect("Failed to resize buffer");

                // Update buffer dimensions
                buffer.width = new_size.width as usize;
                buffer.height = new_size.height as usize;
                buffer.pixel_buffer = vec![0; buffer.width * buffer.height * 4];
                buffer.transform_requested = true;
            }

            // Main render loop
            Event::MainEventsCleared => {
                // Apply accumulated transformations
                if accumulated_zoom != 0. {
                    buffer.stop_calculation();
                    buffer.apply_zoom(cursor_pos, accumulated_zoom);
                    accumulated_zoom = 0.;
                    buffer.transform_requested = true;
                }

                if accumulated_rotation != 0. {
                    buffer.stop_calculation();
                    buffer.apply_rotation(cursor_pos, accumulated_rotation);
                    accumulated_rotation = 0.;
                    buffer.transform_requested = true;
                }

                // Handle any pending transformations
                if buffer.transform_requested {
                    buffer.stop_calculation();
                    buffer.transform_buffer();
                    buffer.start_z_iterations();
                    buffer.transform_requested = false;
                }

                // Update and render the display
                {
                    let shared_colours = buffer.shared_colours.lock().unwrap();
                    buffer.pixel_buffer.copy_from_slice(&shared_colours);
                }
                buffer.interpolate_pixels();
                buffer.plot_boundary_path();

                // Update frame buffer
                let frame = pixels.frame_mut();
                frame.copy_from_slice(buffer.pixel_buffer.as_slice());

                if let Err(err) = pixels.render() {
                    eprintln!("pixels.render() failed: {}", err);
                    *control_flow = ControlFlow::Exit;
                    return;
                }

                window.request_redraw();
            }
            _ => (),
        }
    });
}
/// Calculates the Mandelbrot set properties for a given complex point
///
/// # Arguments
/// * `point` - Complex point to analyze
/// * `max_iterations` - Maximum iteration count
///
/// # Returns
/// * `is_inside` - Whether point is in the Mandelbrot set
/// * `period` - Orbit period (for points in the set)
/// * `escape_velocity` - Rate of escape (for points outside the set)
/// * `coordinates` - Position information
/// * `flame_coords` - Additional colouring coordinates
/// * `slope` - Derivative information for colouring
fn calculate_mandelbrot_point(
    point: Complex<f64>,
    max_iterations: u64,
) -> (
    u64,
    f64,
    Complex<f64>,
    Complex<f64>,
    Complex<f64>,
    Complex<f64>,
) {
    let mut inside = false;
    let point_re = point.re;
    let point_im = point.im;
    let mut z_re = point_re;
    let mut z_im = point_im;
    let mut escape_velocity = max_iterations as f64;
    let mut period = 0;
    let mut coord_re = 0f64;
    let mut coord_im = 0f64;
    let mut dist_sum = 0f64;
    let mut flame_re = 0f64;
    let mut flame_im = 0f64;
    let mut energy_sum = 0f64;
    let mut slope_re = 1f64;
    let mut slope_im = 0f64;
    let mut orbit_re = 1f64;
    let mut orbit_im = 0f64;
    let mut z_min = f64::MAX;
    let mut multiplier_re = 1f64;
    let mut multiplier_im = 0f64;

    const MAX_NEWTON_STEPS: isize = 32;
    let epsilon_squared = 2f64.powi(-57);

    let mut has_converged = false;

    'main_iteration: for iter in 1..max_iterations {
        let iter_f64 = iter as f64;

        // Calculate derivative
        let slope_temp = 2. * (slope_re * z_re - slope_im * z_im);
        slope_im = 2. * (slope_re * z_im + slope_im * z_re);
        slope_re = slope_temp + 1.;

        let z_re_sq = z_re * z_re;
        let z_im_sq = z_im * z_im;
        let z_mag_sq = z_re_sq + z_im_sq;
        let mut iter_sq = iter_f64 * iter_f64;

        if z_mag_sq > 4. {
            // Point is escaping
            let log_log_mag = z_mag_sq.ln().ln();
            let mut taper = (log_log_mag - (2. * 2f64.ln()).ln()) / 512f64.ln();
            taper = 1. - taper * taper;
            taper = taper * taper;
            taper = taper * taper * taper * iter_sq;
            let z_scale = taper / z_mag_sq.sqrt();
            energy_sum += taper;
            flame_re += z_re * z_scale;
            flame_im += z_im * z_scale;

            let z_weight = iter_sq / z_mag_sq;
            dist_sum += z_weight;
            coord_re += z_re * z_weight;
            coord_im += z_im * z_weight;

            if z_mag_sq > std::f64::MAX.sqrt() {
                escape_velocity =
                    iter_f64 - (log_log_mag - 5.174750624576761) * std::f64::consts::E.log2();
                has_converged = true;
                break 'main_iteration;
            }

            let orbit_temp = 2. * (z_re * orbit_re - z_im * orbit_im) + 1.;
            orbit_im = 2. * (orbit_re * z_im + orbit_im * z_re);
            orbit_re = orbit_temp;

            z_im = 2. * z_re * z_im + point_im;
            z_re = z_re_sq - z_im_sq + point_re;
        } else {
            // Point might be in set
            let taper = iter_sq;
            let z_scale = taper / z_mag_sq.sqrt();
            energy_sum += taper;
            flame_re += z_re * z_scale;
            flame_im += z_im * z_scale;

            let z_weight = iter_sq / z_mag_sq;
            dist_sum += z_weight;
            coord_re += z_re * z_weight;
            coord_im += z_im * z_weight;

            if z_min > z_mag_sq {
                let magnitude_ratio = z_mag_sq / z_min;
                z_min = z_mag_sq;
                if magnitude_ratio < 0.25 {
                    let mut w_re = z_re;
                    let mut w_im = z_im;

                    for _ in 0..MAX_NEWTON_STEPS {
                        let mut u_re = w_re;
                        let mut u_im = w_im;
                        let mut du_re = 1.;
                        let mut du_im = 0.;

                        multiplier_re = 1.;
                        multiplier_im = 0.;

                        for _ in 0..iter {
                            let temp_re = multiplier_re;
                            multiplier_re = 2. * (temp_re * u_re - multiplier_im * u_im);
                            multiplier_im = 2. * (temp_re * u_im + multiplier_im * u_re);

                            let du_temp = 2. * (du_re * u_re - du_im * u_im);
                            du_im = 2. * (du_re * u_im + du_im * u_re);
                            du_re = du_temp;

                            let u_im_sq = u_im * u_im;
                            u_im = 2. * u_re * u_im + point_im;
                            u_re = u_re * u_re - u_im_sq + point_re;
                        }

                        let mut newton_re = u_re - w_re;
                        let mut newton_im = u_im - w_im;
                        let du_offset = du_re - 1.;
                        let du_denom = du_offset * du_offset + du_im * du_im;
                        let newton_temp = (newton_re * du_offset + newton_im * du_im) / du_denom;
                        newton_im = (newton_im * du_offset - newton_re * du_im) / du_denom;
                        newton_re = newton_temp;

                        iter_sq = newton_re * newton_re + newton_im * newton_im;
                        w_re -= newton_re;
                        w_im -= newton_im;

                        if iter_sq < epsilon_squared {
                            if du_re * du_re + du_im * du_im < 1. {
                                coord_re = du_re;
                                coord_im = du_im;
                                flame_re = w_re;
                                flame_im = w_im;
                                inside = true;
                                period = iter;
                                has_converged = true;
                                break 'main_iteration;
                            }
                        }
                    }
                }
            }

            let orbit_temp = 2. * (z_re * orbit_re - z_im * orbit_im) + 1.;
            orbit_im = 2. * (orbit_re * z_im + orbit_im * z_re);
            orbit_re = orbit_temp;

            z_im = 2. * z_re * z_im + point_im;
            z_re = z_re_sq - z_im_sq + point_re;
        }
    }

    let mut slope = Complex::new(0., 0.);
    let coordinates;
    let flame_coords;

    if inside {
        coordinates = Complex::new(coord_re, coord_im);
        flame_coords = Complex::new(flame_re, flame_im);

        let initial_point = point;
        let mut refined_point = initial_point;
        for _ in 0..MAX_NEWTON_STEPS {
            let mut z = refined_point;
            let mut d = Complex::new(1., 0.);
            for _ in 1..period {
                d = 2. * d * z + 1.;
                z = z * z + refined_point;
            }
            d = refined_point - z / d + Complex::new(0., 2f64.powi(-256));
            if d == refined_point {
                refined_point = d;
                break;
            }
            refined_point = d;
        }
        let diff = initial_point - refined_point;
        slope = Complex::new(diff.re, -diff.im);
    } else {
        dist_sum = 1. / dist_sum;
        energy_sum = 1. / energy_sum;
        coordinates = Complex::new(coord_re * dist_sum, coord_im * dist_sum);
        flame_coords = Complex::new(flame_re * energy_sum, flame_im * energy_sum);

        if has_converged {
            z_im = -z_im;
            orbit_im = -orbit_im;
            orbit_re = orbit_re / std::f64::MAX.sqrt();
            orbit_im = orbit_im / std::f64::MAX.sqrt();
            slope = ((z_re * z_re + z_im * z_im).ln()) * Complex::new(z_re, z_im)
                / (Complex::new(orbit_re, orbit_im))
                / std::f64::MAX.sqrt();
        }
    }

    (
        period,
        escape_velocity,
        coordinates,
        flame_coords,
        slope,
        Complex::new(multiplier_re, multiplier_im),
    )
}
fn calculate_colour(
    _point: Complex<f64>,
    period: u64,
    escape_velocity: f64,
    coordinates: Complex<f64>,
    flames: Complex<f64>,
    slope: Complex<f64>,
    _zoom: Complex<f64>,
) -> [u8; 4] {
    let mut red: f64;
    let mut green: f64;
    let mut blue: f64;
    if period != 0 {
        // initial solid colouring
        let mut hash = period;
        hash = hash.rotate_left(2);
        hash ^= period;
        hash = hash.wrapping_mul(2381);
        hash = hash.wrapping_mul(hash + 13);
        red = (hash % 6229) as f64 / 6229.;
        hash = hash.rotate_left(3);
        hash ^= period;
        hash = hash.wrapping_mul(37);
        hash = hash.wrapping_mul(hash + 17);
        green = (hash % 6247) as f64 / 6247.;
        hash = hash.rotate_left(5);
        hash ^= period;
        hash = hash.wrapping_mul(43);
        hash = hash.wrapping_mul(hash + 3169);
        blue = (hash % 6271) as f64 / 6271.;

        // coordinate shading
        let (_mag, mut angle) = coordinates.to_polar();
        let magnitude = coordinates.norm_sqr();
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

        // let onetwenty = Complex::new(-0.5, 0.75.sqrt());
        // let onetwentyn = Complex::new(-0.5, -(0.75.sqrt()));
        // red = flames.re;
        // green = (flames * onetwenty).re;
        // blue = (flames * onetwentyn).re;
    } else {
        let coordinates_norm = coordinates.norm();
        let coordinates_sign = coordinates / coordinates_norm;
        let flames_scaled =
            coordinates_norm.atan() * coordinates_sign * (1. / slope.norm()).log2() * 2.
                - flames * escape_velocity.sqrt();
        red = (flames_scaled * Complex::new(-0.0732, -0.1847)).re;
        green = (flames_scaled * Complex::new(0.113, 0.0378)).re;
        blue = (flames_scaled * Complex::new(-0.2134, 0.1756)).re;

        red = (escape_velocity.sqrt() + red).sin() / 2. + 0.5;
        green = (escape_velocity.sqrt() + green).sin() / 2. + 0.5;
        blue = (escape_velocity.sqrt() + blue).sin() / 2. + 0.5;
    }

    let red_byte = (red.max(0.).sqrt() * 256.) as u8;
    let green_byte = (green.max(0.).sqrt() * 256.) as u8;
    let blue_byte = (blue.max(0.).sqrt() * 256.) as u8;

    [red_byte, green_byte, blue_byte, 255]
}
