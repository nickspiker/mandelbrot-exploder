use num_complex::{Complex, Complex64};
use rand::prelude::*;
use rustfft::{FftDirection, FftPlanner};

/// Plots a 2D FFT visualization of a Mandelbrot iteration sequence
///
/// Generates a sequence of iterations z_{n+1} = z_n^2 + c, performs a 2D FFT
/// on the sequence (treating it as RGB channels), and renders the frequency
/// domain as an image.
///
/// # Arguments
/// * `c` - Complex constant for the iteration
/// * `width` - Output image width
/// * `height` - Output image height
///
/// # Returns
/// ARGB pixel buffer (Vec<u32>) containing the FFT visualization
pub fn plot_path_fft(c: Complex64, width: usize, height: usize) -> Vec<u32> {
    println!("\nC: [{}, {}]\n", c.re, c.im);
    let mut image = vec![0xFF000000u32; width * height];
    let mut fft_data = vec![Complex64::new(0., 0.); width * height * 3];

    let mut z = Complex::new(0., 0.);

    for i in 1..width * height * 3 {
        let new_z = z * z + c;
        if !(new_z.is_finite() && new_z.is_finite()) {
            break;
        }
        fft_data[i] = new_z;
        z = new_z;
    }

    let mut planner = FftPlanner::new();
    let row_fft = planner.plan_fft(width, FftDirection::Forward);
    let col_fft = planner.plan_fft(height, FftDirection::Forward);

    let mut sum = 0.;
    for channel in 0..3 {
        // FFT rows
        for h in 0..height {
            let mut row_data = Vec::with_capacity(width);
            for w in 0..width {
                row_data.push(fft_data[h * width * 3 + w * 3 + channel]);
            }
            row_fft.process(&mut row_data);
            for w in 0..width {
                fft_data[h * width * 3 + w * 3 + channel] = row_data[w];
            }
        }

        // FFT columns
        for w in 0..width {
            let mut col_data = Vec::with_capacity(height);
            for h in 0..height {
                col_data.push(fft_data[h * width * 3 + w * 3 + channel]);
            }
            col_fft.process(&mut col_data);
            for h in 0..height {
                fft_data[h * width * 3 + w * 3 + channel] = col_data[h];
                sum += col_data[h].norm();
            }
        }
    }
    let mut rng = thread_rng();
    println!();
    for _ in 0..256 {
        let idx = rng.gen_range(0..width * height * 3);
        print!("[{}, {}]  ", fft_data[idx].re, fft_data[idx].im);
    }

    let scale = fft_data.len() as f64 * (1 << 14) as f64 / sum;

    for h in 0..height {
        for w in 0..width {
            let fft_idx = (h * width + w) * 3;
            let img_idx = h * width + w;
            let r = ((fft_data[fft_idx].re * scale).sqrt()) as u8;
            let g = ((fft_data[fft_idx + 1].re * scale).sqrt()) as u8;
            let b = ((fft_data[fft_idx + 2].re * scale).sqrt()) as u8;
            image[img_idx] = 0xFF000000 | (r as u32) << 16 | (g as u32) << 8 | (b as u32);
        }
    }

    image
}
