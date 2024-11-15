# Mandelbrot Exploder

A Mandelbrot set explorer written in Rust with real-time interaction and parallel processing.

## Features

- Interactive zooming and rotation
- Progressive detail refinement
- Parallel computation for calculations
- Window-resize handling
- High-precision (64-bit) floating point math
- Escape-time visualization with orbit trap coloring

## Building

Requires the Rust toolchain.

```bash
cargo run --release
```

## Usage

- Left click: Center on point
- Mouse wheel: Zoom
- Shift + mouse wheel: Rotate
- Window can be freely resized

## Implementation Details

The visualization uses:
- `winit` for window management
- `pixels` for frame buffer handling
- `rayon` for parallel computation
- `num_complex` for complex number math

Core components:
- Escape-time calculation with period checking
- Newton iteration for period calculation
- Orbit tracking for color generation
- View transformation with coordinate remapping
- Progressive refinement during navigation
- Atomic counters for thread-safe progress tracking

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.