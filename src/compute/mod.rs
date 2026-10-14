pub mod boundary;
pub mod colouring;
pub mod mandelbrot;
pub mod mandelbrot_f32;
pub mod mandelbrot_f64;
pub mod perturbation;
#[cfg(target_os = "macos")]
pub mod gpu_delta;
pub mod spirix_grid;

pub use boundary::{path_to_audio, path_to_audio_fft, trace_path};
pub use colouring::calculate_colour;
pub use mandelbrot_f32::calculate_mandelbrot_point_f32;
pub use mandelbrot_f64::calculate_mandelbrot_point_f64;
pub use perturbation::PerturbEngine;
pub use spirix_grid::{spirix_point_fn, SpirixPointFn};
