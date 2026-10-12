mod app;
mod audio;
mod compute;
mod config;
mod precision;
mod render;
mod state;

use app::MandelbrotApp;
use num_complex::Complex64;

fn main() {
    // Default starting position for manual tracing
    let boundary_path = Vec::new();
    let center = Complex64::new(0.0, 0.0);
    let zoom = Complex64::new(1.0, 0.0);

    println!("Controls:");
    println!("  Mouse wheel: Zoom in/out");
    println!("  Shift + wheel: Rotate view");
    println!("  Left click + drag: Pan");
    println!("  Left click: Play raw z iteration as audio");
    println!("  Ctrl + left click: Trace boundary path with FFT audio");
    println!("  Shift + left click: Show point statistics");
    println!("  Right click: Toggle FFT visualization mode");
    println!("  S key: Search for interesting spot (samples 256 points, zooms in)");
    println!("  Precision: dropdowns under the viewport stats");
    println!("  Ctrl+scroll / Ctrl+= / Ctrl+- / Ctrl+0: UI scale (host)\n");

    let app = MandelbrotApp::new(center, zoom, boundary_path);
    fluor::host::app::run_app(app).expect("event loop");
}
