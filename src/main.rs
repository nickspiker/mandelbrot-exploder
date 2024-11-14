use num_complex::*;
use pixels::{Error, Pixels, SurfaceTexture};
use rand::Rng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

const MAX_ITERATIONS: u64 = 1 << 24;
const INITIAL_WIDTH: u32 = 0x400;
const INITIAL_HEIGHT: u32 = 0x200;

/// Holds both the display buffer and the corresponding complex coordinates
struct MandelbrotBuffer {
    /// RGBA pixels for display
    pixels: Vec<u8>,
    /// Complex coordinates corresponding to each pixel actual location
    coordinates: Vec<Complex64>,
    width: u32,
    height: u32,
    /// Center point of the current view
    center: Complex64,
    /// Current zoom level (complex number to include rotation)
    zoom: Complex64,
    calculated: AtomicU32,
    /// Flags to indicate if a redraw is needed
    view_changed: bool,
    /// Accumulated deltas
    accumulated_zoom_delta: f64,
    accumulated_rotation_delta: f64,
    /// Shift key state
    shift_held: bool,
}

impl MandelbrotBuffer {
    fn new(width: u32, height: u32) -> Self {
        let size = (width * height) as usize;
        Self {
            pixels: vec![0; size * 4],
            coordinates: vec![Complex64::new(0., 0.); size],
            width,
            height,
            center: Complex64::new(0., 0.),
            zoom: Complex64::new(1., 0.),
            calculated: AtomicU32::new(0),
            view_changed: true,
            accumulated_zoom_delta: 0.,
            accumulated_rotation_delta: 0.,
            shift_held: false,
        }
    }

    fn calculate_pixels(&mut self) -> bool {
        if !self.view_changed {
            return false;
        }
        self.view_changed = false;

        let size = (self.width * self.height) as usize;

        // First pass: collect points to calculate
        let points: Vec<(usize, Complex64)> = (0..size)
            .filter(|&idx| self.needs_calculation(idx))
            .map(|idx| {
                let mut rng = rand::thread_rng();
                let x = (idx % self.width as usize) as f64;
                let y = (idx / self.width as usize) as f64;
                let x_offset = rng.gen::<f64>();
                let y_offset = rng.gen::<f64>();
                (idx, self.screen_to_complex(x + x_offset, y + y_offset))
            })
            .collect();

        if points.is_empty() {
            return false;
        }

        // Second pass: parallel calculation
        let results: Vec<_> = points
            .par_iter()
            .map(|&(idx, point)| {
                let (is_inside, period, escape_velocity, coordinates, flames, slope) =
                    calculate_mandelbrot_point(point, MAX_ITERATIONS);

                let color = calculate_colour(
                    is_inside,
                    period,
                    escape_velocity,
                    coordinates,
                    flames,
                    slope,
                    self.zoom / self.zoom.norm(),
                );

                (idx, point, color)
            })
            .collect();

        // Third pass: update buffers
        for (idx, point, color) in results {
            self.coordinates[idx] = point;
            let pixel_start = idx * 4;
            self.pixels[pixel_start..pixel_start + 4].copy_from_slice(&color);
            self.calculated.fetch_add(1, Ordering::Relaxed);
        }

        true
    }

    fn needs_calculation(&self, index: usize) -> bool {
        self.pixels[index * 4 + 3] != 255
    }

    fn screen_to_complex(&self, x: f64, y: f64) -> Complex64 {
        // Use screen diagonal for consistent scaling
        let diagonal = (self.width as f64 * self.width as f64
            + self.height as f64 * self.height as f64)
            .sqrt();

        // Convert to -1 to 1 range using diagonal scale
        let screen_x = (x - self.width as f64 / 2.) / (diagonal / 2.);
        let screen_y = (y - self.height as f64 / 2.) / (diagonal / 2.);

        // Apply complex zoom transform and center offset
        (Complex64::new(screen_x, screen_y) / self.zoom) + self.center
    }

    fn complex_to_screen(&self, c: Complex64) -> (f64, f64) {
        let diagonal = (self.width as f64 * self.width as f64
            + self.height as f64 * self.height as f64)
            .sqrt();
        let transformed = (c - self.center) * self.zoom;
        let x = transformed.re * (diagonal / 2.) + self.width as f64 / 2.;
        let y = transformed.im * (diagonal / 2.) + self.height as f64 / 2.;
        (x, y)
    }

    fn handle_zoom(&mut self, delta: f64, cursor_pos: (f64, f64)) {
        // Convert cursor position to complex coordinates
        let cursor_point = self.screen_to_complex(cursor_pos.0, cursor_pos.1);

        // Calculate zoom factor
        let zoom_factor = delta;

        // Update zoom while preserving rotation
        self.zoom *= Complex64::new(zoom_factor, 0.);

        // Adjust center point based on cursor position
        let new_cursor = self.screen_to_complex(cursor_pos.0, cursor_pos.1);
        self.center += cursor_point - new_cursor;

        // Reset calculation state
        self.calculated.store(0, Ordering::Relaxed);

        // Preserve and remap existing calculations where possible
        self.remap_buffer();

        self.view_changed = true;
    }

    fn handle_rotation(&mut self, angle_delta: f64, cursor_pos: (f64, f64)) {
        // Convert cursor position to complex coordinates
        let cursor_point = self.screen_to_complex(cursor_pos.0, cursor_pos.1);

        // Update zoom with new angle
        let angle = angle_delta;
        self.zoom *= Complex64::from_polar(1., angle);

        // Adjust center point based on cursor position
        let new_cursor = self.screen_to_complex(cursor_pos.0, cursor_pos.1);
        self.center += cursor_point - new_cursor;

        // Reset calculation state
        self.calculated.store(0, Ordering::Relaxed);

        // Remap existing calculations
        self.remap_buffer();

        self.view_changed = true;
    }

    fn handle_click(&mut self, x: f64, y: f64) {
        // Update center to clicked point
        let clicked_point = self.screen_to_complex(x, y);
        self.center = clicked_point;

        // Reset calculation state
        self.calculated.store(0, Ordering::Relaxed);

        // Clear buffer for recalculation
        self.clear_buffer();

        self.view_changed = true;
    }

    fn remap_buffer(&mut self) {
        let size = (self.width * self.height) as usize;
        let mut new_pixels = vec![0; size * 4];
        let mut new_coordinates = vec![Complex64::new(0., 0.); size];

        // For each point in the old buffer that has been calculated
        for old_y in 0..self.height {
            for old_x in 0..self.width {
                let old_idx = (old_y * self.width + old_x) as usize;

                // Skip if not calculated
                if self.pixels[old_idx * 4 + 3] != 255 {
                    continue;
                }

                // Get the original complex coordinate for this point
                let coord = self.coordinates[old_idx];

                // Use the complex_to_screen method for accurate mapping
                let (new_x_f64, new_y_f64) = self.complex_to_screen(coord);
                let new_x = new_x_f64 as i32;
                let new_y = new_y_f64 as i32;

                // Check if point is in new viewport
                if new_x >= 0
                    && new_x < self.width as i32
                    && new_y >= 0
                    && new_y < self.height as i32
                {
                    let new_idx = (new_y * self.width as i32 + new_x) as usize;

                    // Only copy if we haven't already got a value for this pixel
                    if new_idx < size && new_pixels[new_idx * 4 + 3] == 0 {
                        // Copy pixel data and coordinate
                        new_pixels[new_idx * 4..new_idx * 4 + 4]
                            .copy_from_slice(&self.pixels[old_idx * 4..old_idx * 4 + 4]);
                        new_coordinates[new_idx] = coord;
                        self.calculated.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }

        // Update buffers
        self.pixels = new_pixels;
        self.coordinates = new_coordinates;
    }

    fn remap_buffer_from(&mut self, old_buffer: &MandelbrotBuffer) {
        let size = (self.width * self.height) as usize;

        for old_y in 0..old_buffer.height {
            for old_x in 0..old_buffer.width {
                let old_idx = (old_y * old_buffer.width + old_x) as usize;

                // Skip if not calculated
                if old_buffer.pixels[old_idx * 4 + 3] != 255 {
                    continue;
                }

                // Get the original complex coordinate for this point
                let coord = old_buffer.coordinates[old_idx];

                // Use the new buffer's complex_to_screen method for accurate mapping
                let (new_x_f64, new_y_f64) = self.complex_to_screen(coord);
                let new_x = new_x_f64 as i32;
                let new_y = new_y_f64 as i32;

                // Check if point is in new viewport
                if new_x >= 0
                    && new_x < self.width as i32
                    && new_y >= 0
                    && new_y < self.height as i32
                {
                    let new_idx = (new_y * self.width as i32 + new_x) as usize;

                    // Only copy if we haven't already got a value for this pixel
                    if new_idx < size && self.pixels[new_idx * 4 + 3] == 0 {
                        // Copy pixel data and coordinate
                        self.pixels[new_idx * 4..new_idx * 4 + 4]
                            .copy_from_slice(&old_buffer.pixels[old_idx * 4..old_idx * 4 + 4]);
                        self.coordinates[new_idx] = coord;
                        self.calculated.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
    }

    fn clear_buffer(&mut self) {
        let size = (self.width * self.height) as usize;
        self.pixels = vec![0; size * 4];
        self.coordinates = vec![Complex64::new(0., 0.); size];
        self.calculated.store(0, Ordering::Relaxed);
    }
}

fn main() -> Result<(), Error> {
    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("Mandelbrot Exploder")
        .with_inner_size(LogicalSize::new(INITIAL_WIDTH, INITIAL_HEIGHT))
        .build(&event_loop)
        .unwrap();

    let window_size = window.inner_size();

    let mut pixels = {
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(window_size.width, window_size.height, surface_texture)?
    };

    let mut buffer = MandelbrotBuffer::new(window_size.width, window_size.height);
    let mut cursor_pos = PhysicalPosition::new(0., 0.);

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
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f64 / 100., // Adjust scaling as needed
                };

                if buffer.shift_held {
                    // Accumulate rotation delta
                    buffer.accumulated_rotation_delta += scroll_amount;
                } else {
                    // Accumulate zoom delta
                    buffer.accumulated_zoom_delta += scroll_amount;
                }
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
                buffer.handle_click(cursor_pos.x, cursor_pos.y);
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
                if pixels
                    .resize_surface(new_size.width, new_size.height)
                    .is_err()
                {
                    *control_flow = ControlFlow::Exit;
                }

                if pixels
                    .resize_buffer(new_size.width, new_size.height)
                    .is_err()
                {
                    *control_flow = ControlFlow::Exit;
                }

                // Create a new buffer with the new size
                let mut new_buffer = MandelbrotBuffer::new(new_size.width, new_size.height);

                // Copy over the center, zoom, and other relevant state
                new_buffer.center = buffer.center;
                new_buffer.zoom = buffer.zoom;
                new_buffer.shift_held = buffer.shift_held;
                new_buffer.view_changed = true;

                // Remap existing calculations to the new buffer
                new_buffer.remap_buffer_from(&buffer);

                // Replace the old buffer with the new one
                buffer = new_buffer;

                window.request_redraw();
            }

            Event::MainEventsCleared => {
                // Apply accumulated zoom delta
                if buffer.accumulated_zoom_delta != 0. {
                    let zoom_factor = if buffer.accumulated_zoom_delta > 0. {
                        1.1
                    } else {
                        0.9
                    };
                    buffer.handle_zoom(
                        zoom_factor.powf(buffer.accumulated_zoom_delta.abs()),
                        (cursor_pos.x, cursor_pos.y),
                    );
                    buffer.accumulated_zoom_delta = 0.;
                }

                // Apply accumulated rotation delta
                if buffer.accumulated_rotation_delta != 0. {
                    let rotation_amount = buffer.accumulated_rotation_delta * 0.05; // Adjust sensitivity as needed
                    buffer.handle_rotation(rotation_amount, (cursor_pos.x, cursor_pos.y));
                    buffer.accumulated_rotation_delta = 0.;
                }

                if buffer.calculate_pixels() {
                    // Update display
                    pixels.frame_mut().copy_from_slice(&buffer.pixels);
                    if pixels.render().is_err() {
                        *control_flow = ControlFlow::Exit;
                    }

                    window.request_redraw();
                }
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

    const MAX_NEWTON_STEPS: i32 = 32;
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
fn calculate_colour(
    is_inside: bool,
    period: u64,
    escape_velocity: f64,
    coordinates: Complex64,
    flames: Complex64,
    slope: Complex64,
    rotation: Complex64,
) -> [u8; 4] {
    let mut red;
    let mut green;
    let mut blue;

    if is_inside {
        // Constants for 120° colour rotation
        let angle_120 = Complex64::new(-0.5, 0.75f64.sqrt());
        let angle_240 = Complex64::new(-0.5, -(0.75f64.sqrt()));

        let slope_angle = slope / slope.norm();
        let coord_magnitude_sq = coordinates.norm_sqr();
        let coord_magnitude = coord_magnitude_sq.sqrt();
        let coords_rotated = slope_angle * coord_magnitude * rotation;

        // Calculate diffuse lighting
        let mut diffuse = 1. - coord_magnitude_sq;
        let temp = coords_rotated.im - coords_rotated.re + 2f64.sqrt();
        diffuse = diffuse * temp * temp / 3.5;
        diffuse = 1. - diffuse;
        diffuse = diffuse * diffuse;
        diffuse = 1. - diffuse * diffuse;

        // Calculate base colour from polar coordinates
        let (magnitude, angle) = coordinates.to_polar();
        let interior_colour =
            Complex64::from_polar(magnitude, angle + 3. / coord_magnitude.sqrt() + 3.);

        // Generate RGB components through 120° rotations
        red = interior_colour.re + 1.;
        green = (interior_colour * angle_120).re + 1.;
        blue = (interior_colour * angle_240).re + 1.;

        // Apply quadratic intensity scaling
        red = red * red / 4.;
        green = green * green / 4.;
        blue = blue * blue / 4.;

        // Apply diffuse lighting with bounds checking
        red = (red * diffuse).max(0.);
        green = (green * diffuse).max(0.);
        blue = (blue * diffuse).max(0.);
    } else {
        // Constants for exterior colouring
        let angle_pos = Complex64::from_polar(1., 1.);
        let angle_neg = Complex64::from_polar(1., -1.);

        // Calculate slope-based shading
        let slope_magnitude = slope.norm();
        let slope_direction =
            slope * (1. / slope_magnitude) * rotation * Complex64::from_polar(1., -0.2);
        let slope_intensity = slope_direction.im / 4. + 0.5;
        let glow = ((-53. - slope_magnitude.log2()).max(0.)).sqrt();

        // Calculate flame colouring
        let flame = flames / 3.;
        let flame_red = flame.re;
        let flame_green = (flame * angle_pos).re;
        let flame_blue = (flame * angle_neg).re;

        // Calculate escape velocity based colouring
        let mut escape_log = escape_velocity.log2();
        let escape_factor = 0.008 / (escape_log - 8.7);
        escape_log = 3.2 - escape_log * 14.;

        // Combine base colours
        red = escape_factor * (escape_log.sin() + 2.5);
        green = escape_factor * ((escape_log + 0.8).sin() + 2.5) * 0.9;
        blue = escape_factor * ((escape_log + 2.).sin() + 2.5) * 0.7;

        // Add flame effects
        red += flame_red;
        green += flame_green;
        blue += flame_blue;

        // Apply slope shading
        red *= slope_intensity;
        green *= slope_intensity;
        blue *= slope_intensity;

        // Add glow effects
        blue += glow;
        green += glow / 3.;
        red += glow / 9.;
    }

    // Convert to 8-bit colour with gamma correction and bounds checking
    let red_byte = (red.max(0.).sqrt() * 256.) as u8;
    let green_byte = (green.max(0.).sqrt() * 256.) as u8;
    let blue_byte = (blue.max(0.).sqrt() * 256.) as u8;

    [red_byte, green_byte, blue_byte, 255]
}
