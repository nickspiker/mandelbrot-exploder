# mandelbrot-exploder

An interactive Mandelbrot explorer with audio, and a test rig for [spirix](https://github.com/nickspiker/spirix) number types.

### Perturbation mode

`f32 pert` is the startup mode and the pipeline the rig is built around: one reference orbit in F7E3, every pixel as an f32 offset from it with all per-pixel quantities divided by the zoom so f32's exponent never sees the depth, and, once a pixel escapes, an exact F7E7 ride of z to that type's horizon near |z| = 2^(2^127) for the escape velocity, the flame sums and the slope. The rest of the pipeline is the same as the other modes: Newton period detection and interior root refinement run in f64 in the delta frame against the reference. On macOS the f32 loop runs as a Metal compute kernel (`src/compute/delta.wgsl`) with the CPU loop in `src/compute/perturbation.rs` as its exact twin; `MANDEL_NO_GPU=1` forces the CPU loop. `cargo run --release --example perturb_bench` times both and checks them against the direct F7E3 cell.

Known edges, all deliberate: f32 disintegrates when the zoom goes denormal near 2^-126; a reference can only be made exactly periodic while F7E3 can place the nucleus to better than a pixel (roughly zoom > 2^-68), and interior shading degrades past that; interiors of satellite minibrots too small for f64 (below about 2^-50) are found through per-period nuclei searched in F7E3, which is capped at periods of 2^16.

The GUI is built on [fluor](https://github.com/nickspiker/fluor).

## Building

`fluor` and `spirix` are path dependencies, so clone them next to this repo:

```
Code/
├── fluor/
├── spirix/
└── mandelbrot-exploder/
```

Then:

```
cargo run --release
```

## Controls

| Input | Action |
|---|---|
| Mouse wheel / `+` `-` | Zoom |
| Shift + wheel | Rotate |
| Left drag | Pan |
| Left click | Play the raw z iteration as audio |
| Ctrl + left click | Trace a boundary path with FFT audio |
| Shift + left click | Point statistics |
| Right click | Toggle FFT visualization |
| `S` | Search for an interesting spot and zoom in |
| `T` | Cycle the colour theme (also the third pill) |
| Ctrl+D | Debug counters |
| Ctrl+M | Toggle interpolation (show raw projections only) |
| Ctrl+H | Hit-test map |
| Ctrl+scroll / Ctrl+`=` `-` `0` | UI scale |

## Environment

- `MANDEL_PRECISION=f32|f64|pert|f<3..7>e<3..7>`: start in a given precision mode, e.g. `MANDEL_PRECISION=f3e5`
- `MANDEL_NO_GPU=1`: run the perturbation loop on the CPU even when Metal is available
- `MANDEL_SNAPSHOT=path.ppm[,seconds]`: write the rendered frame after that many seconds (default 5) and exit
- `MANDEL_THEME=classic|bands|log bands|distance|flame`: start with a given colour theme
- `MANDEL_AUTOZOOM=n`: zoom in one notch at the screen centre every n frames (scripted navigation)
- `MANDEL_TIMING=1`: print per-navigation and slow-step timings
- `MANDEL_VIEW=re,im,log2zoom`: start at a given view (centre passes through f64, so good to about 2^-50), e.g. `MANDEL_VIEW=-1.7548776662466927,0,-30`
- `MANDEL_OPEN_DD=1`: open the precision dropdown at startup (for screenshots)
