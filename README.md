# mandelbrot-exploder

An interactive Mandelbrot explorer with audio, and a test rig for [spirix](https://github.com/nickspiker/spirix) number types.

The view coordinates are held in spirix `CircleF7E3`, and zoom is clamped to ±2^120. The iteration itself can run in `f32`, `f64`, or any of the 25 spirix F×E combinations (F3–F7 × E3–E7). Pick one from the dropdowns under the viewport stats. Escape values and flame sums that overflow the E3 range are expected; seeing how each type handles that is part of the point.

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
| Ctrl+D | Debug counters |
| Ctrl+M | Toggle interpolation (show raw projections only) |
| Ctrl+H | Hit-test map |
| Ctrl+scroll / Ctrl+`=` `-` `0` | UI scale |

## Environment

- `MANDEL_PRECISION=f32|f64|f<3..7>e<3..7>`: start in a given precision mode, e.g. `MANDEL_PRECISION=f3e5`
- `MANDEL_OPEN_DD=1`: open the precision dropdown at startup (for screenshots)
