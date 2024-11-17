use num_complex::*;
use pixels::{Error, Pixels, SurfaceTexture};
use rand::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

const MAX_ITERATIONS: u64 = 1 << 24;
const INITIAL_WIDTH: usize = 0x400;
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
}

impl MandelbrotBuffer {
    fn new(width: usize, height: usize) -> Self {
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
            center: Complex64::new(-1., 0.),
            zoom: Complex64::new(0.25, 0.),
            old_width: width,
            old_height: height,
            old_center: Complex64::new(0., 0.),
            old_zoom: Complex64::new(1., 0.),
            stop_flag: Arc::new(AtomicBool::new(false)),
            calculation_thread: None,
            shift_held: false,
            transform_requested: false,
        };
        buffer
    }

    fn interpolate_pixels(&mut self) {
        // Up-down pass second
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
        let cursor_coordinates = Self::screen_to_complex(
            self.width as f64 - cursor_pos.x,
            self.height as f64 - cursor_pos.y,
            self.width as f64,
            self.height as f64,
            self.zoom,
            self.center,
        );

        let rotation = Complex64::new(rotation_delta.cos(), rotation_delta.sin());

        // First rotate the zoom factor
        self.zoom *= rotation;

        // Then rotate the center around the cursor
        self.center = cursor_coordinates + (self.center - cursor_coordinates) * rotation;
    }
    fn set_center(&mut self, cursor_pos: PhysicalPosition<f64>) {
        // Store old center for transform_buffer
        self.old_center = self.center;

        // Set new center based on cursor position
        self.center = Self::screen_to_complex(
            cursor_pos.x,
            cursor_pos.y,
            self.width as f64,
            self.height as f64,
            self.zoom,
            self.center,
        );
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

                let (is_inside, period, escape_velocity, coords, flames, slope) =
                    calculate_mandelbrot_point(point, MAX_ITERATIONS);

                let colour = calculate_colour(
                    point,
                    is_inside,
                    period,
                    escape_velocity,
                    coords,
                    flames,
                    slope,
                    zoom,
                );

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
}

// Update main() to use the new implementation:
fn main() -> Result<(), Error> {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Mandelbrot Exploder")
        .with_inner_size(LogicalSize::new(
            INITIAL_WIDTH as u32,
            INITIAL_HEIGHT as u32,
        ))
        .build(&event_loop)
        .unwrap();

    let window_size = window.inner_size();
    let mut pixels = {
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(window_size.width, window_size.height, surface_texture)?
    };

    let mut buffer = MandelbrotBuffer::new(window_size.width as usize, window_size.height as usize);
    buffer.start_z_iterations();
    let mut cursor_pos = PhysicalPosition::new(0., 0.);
    let mut accumulated_zoom = 0.;
    let mut accumulated_rotation = 0.;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }

            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                cursor_pos = position;
            }

            Event::WindowEvent {
                event: WindowEvent::MouseWheel { delta, .. },
                ..
            } => {
                let scroll_amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(pos) => pos.y / 50.,
                };

                if buffer.shift_held {
                    accumulated_rotation += scroll_amount * -0.1;
                } else {
                    accumulated_zoom += scroll_amount;
                }
                buffer.transform_requested = true;
            }

            Event::WindowEvent {
                event:
                    WindowEvent::MouseInput {
                        state: ElementState::Pressed,
                        button: MouseButton::Left,
                        ..
                    },
                ..
            } => {
                buffer.set_center(cursor_pos);
                println!("Center set to [{},{}]", buffer.center.re, buffer.center.im);
                buffer.transform_requested = true;
            }

            Event::WindowEvent {
                event: WindowEvent::ModifiersChanged(modifiers),
                ..
            } => {
                buffer.shift_held = modifiers.shift();
            }

            Event::WindowEvent {
                event: WindowEvent::Resized(new_size),
                ..
            } => {
                // First resize the pixels buffer/surface
                pixels
                    .resize_surface(new_size.width, new_size.height)
                    .expect("Failed to resize surface");
                pixels
                    .resize_buffer(new_size.width, new_size.height)
                    .expect("Failed to resize buffer");

                // Then update the buffer dimensions
                buffer.width = new_size.width as usize;
                buffer.height = new_size.height as usize;
                buffer.pixel_buffer = vec![0; buffer.width * buffer.height * 4];

                buffer.transform_requested = true;
            }

            Event::MainEventsCleared => {
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

                if buffer.transform_requested {
                    buffer.stop_calculation();
                    buffer.transform_buffer();
                    buffer.start_z_iterations();
                    buffer.transform_requested = false;
                }

                {
                    let shared_colours = buffer.shared_colours.lock().unwrap();
                    buffer.pixel_buffer.copy_from_slice(&shared_colours);
                }
                buffer.interpolate_pixels();

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
) -> (bool, u64, f64, Complex<f64>, Complex<f64>, Complex<f64>) {
    let mut is_inside = false;
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

    const MAX_NEWTON_STEPS: isize = 32;
    let epsilon_squared = 2f64.powi(-53); // Moved from const to let

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

                        for _ in 0..iter {
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
                                is_inside = true;
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

    if is_inside {
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
        is_inside,
        period,
        escape_velocity,
        coordinates,
        flame_coords,
        slope,
    )
}
fn calculate_colour(
    point: Complex<f64>,
    is_inside: bool,
    period: u64,
    escape_velocity: f64,
    coordinates: Complex<f64>,
    flames: Complex<f64>,
    slope: Complex<f64>,
    zoom: Complex<f64>,
) -> [u8; 4] {
    let mut red:f64;
    let mut green:f64;
    let mut blue:f64;

    let inspect = -flames.re() / 2. + 0.5;

    let angle_120 = Complex::new(-0.5, 0.75f64.sqrt());
    let angle_240 = Complex::new(-0.5, -(0.75f64.sqrt()));

     red = (1. / slope.norm()).log2().sin() / 2. + 0.5;
     blue = escape_velocity.sqrt().sin() / 2. + 0.5;
     green = flames.re() / 2. + 0.5;


    let red_byte = (red.max(0.).sqrt() * 256.) as u8;
    let green_byte = (green.max(0.).sqrt() * 256.) as u8;
    let blue_byte = (blue.max(0.).sqrt() * 256.) as u8;

    [red_byte, green_byte, blue_byte, 255]
}
/// Converts Mandelbrot set properties into RGBA gamma 2 colour values
///
/// # Arguments
/// * `is_inside` - Whether the point is in the Mandelbrot set
/// * `period` - Orbit period for points in the set
/// * `escape_velocity` - Rate of escape for points outside the set
/// * `coordinates` - Position information for colouring
/// * `flames` - Additional coordinate information for colouring effects
/// * `slope` - Derivative information for shading
/// * `rotation` - Current view transformation for consistent colouring
///
/// # Returns
/// * `[u8; 4]` - RGBA colour values as bytes
fn _calculate_colour(
    point: Complex64,
    is_inside: bool,
    period: u64,
    escape_velocity: f64,
    coordinates: Complex64,
    flames: Complex64,
    slope: Complex64,
    zoom: Complex64,
) -> [u8; 4] {
    let mut red;
    let mut green;
    let mut blue;

    if is_inside {
        // Constants for 120° colour rotation
        let angle_120 = Complex64::new(-0.5, 0.75f64.sqrt());
        let angle_240 = Complex64::new(-0.5, -(0.75f64.sqrt()));
        let mut mag_sq = coordinates.norm_sqr();
        mag_sq = mag_sq * mag_sq;
        mag_sq = mag_sq * mag_sq;

        // Generate RGB components through 120° rotations
        red = (coordinates.re + 1.) / 2.;
        green = ((coordinates * angle_120).re + 1.) / 2.;
        blue = ((coordinates * angle_240).re + 1.) / 2.;

        red = red * (1. - mag_sq);
        green = green * (1. - mag_sq);
        blue = blue * (1. - mag_sq);
    } else {
        // Constants for 120° colour rotation
        let angle_120 = Complex64::new(-0.5, 0.75f64.sqrt());
        let angle_240 = Complex64::new(-0.5, -(0.75f64.sqrt()));
        let scale = 1. / slope.norm();

        let intensity = slope.norm().log2().sin() / 2. + 1.;
        // Generate RGB components through 120° rotations
        red = (slope.re * scale + 1.) / 8. * intensity;
        green = ((slope * angle_120 * scale).re + 1.) / 8. * intensity;
        blue = ((slope * angle_240 * scale).re + 1.) / 8. * intensity;

        let flame_scale = 0.5 / flames.norm();
        red = (flames.re * flame_scale + 1.) * red;
        green = ((flames * angle_120 * flame_scale).re + 1.) * green;
        blue = ((flames * angle_240 * flame_scale).re + 1.) * blue;
    }

    // Convert to 8-bit colour with gamma correction and bounds checking
    let red_byte = (red.max(0.).sqrt() * 256.) as u8;
    let green_byte = (green.max(0.).sqrt() * 256.) as u8;
    let blue_byte = (blue.max(0.).sqrt() * 256.) as u8;

    [red_byte, green_byte, blue_byte, 255]
}
