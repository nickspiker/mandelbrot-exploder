//! `MandelbrotApp` — the fluor consumer. fluor's host owns the window, event loop,
//! present path, window move/resize drags, and the Ctrl+±/0/scroll UI-zoom; `DefaultChrome`
//! owns the frame, window controls, hover tints, and hit testing. This app owns the fractal:
//! the calculation pipeline in `MandelbrotBuffer`, viewport transforms, and painting the
//! projected points into the frame under the chrome.

use crate::audio::AudioPlayer;
use crate::compute::{
    calculate_mandelbrot_point_f64, path_to_audio, path_to_audio_fft, trace_path,
};
use crate::config::{self, ViewMode, INITIAL_HEIGHT, INITIAL_WIDTH};
use crate::state::buffer::PrecisionMode;
use crate::state::MandelbrotBuffer;

use fluor::canvas::Canvas;
use fluor::coord::Coord;
use fluor::event::{
    CursorIcon, ElementState, Event, Key, MouseButton, MouseScrollDelta,
};
use fluor::geom::Viewport;
use fluor::host::app::{Context, EventResponse, FluorApp};
use fluor::host::chrome::{self, HitId, ResizeEdge, HIT_NONE};
use fluor::host::chrome_widget::DefaultChrome;
use fluor::host::widget::{self as widget, Container, Widget};
use fluor::paint;
use fluor::pixel::{Blend, BlendMode};
use fluor::theme;
use fluor::text::TextStyle;
use fluor::widgets::Dropdown;

use num_complex::Complex64;

pub struct MandelbrotApp {
    buffer: MandelbrotBuffer,
    chrome: Option<DefaultChrome>,
    hit_counter: HitId,
    audio_player: Option<AudioPlayer>,
    /// Fraction selector: f32 / f64 (IEEE) or spirix F3..F7 (8..128-bit fraction).
    frac_dd: Dropdown,
    /// Exponent selector: spirix E3..E7 (8..128-bit exponent). Greyed while an IEEE mode is selected.
    exp_dd: Dropdown,
    theme_dd: Dropdown,
    /// Keyboard-focused widget (a dropdown pill), if any.
    current_focus: Option<HitId>,
    /// Popup open-state from last poll — transitions invalidate chrome so stale row stamps get wiped from the hit map.
    dd_open_state: (bool, bool, bool),
    /// Persistent visible-RGB fractal frame (0xFF000000 | rgb). Replaces the persistent OS
    /// back buffer the old softbuffer renderer provided — fluor's scratch is cleared every
    /// frame, but the progressive projection + flood fill need pixels to survive frames.
    fractal: Vec<u32>,
    cursor_pos: (f64, f64),
    accumulated_zoom: f64,
    accumulated_rotation: f64,
    drag_start_pos: Option<(f64, f64)>,
    dragging: bool,
    /// Pixel rect (x, y, w, h) of the stats panel as last painted; a press inside it drags the window.
    stats_panel: Option<(f64, f64, f64, f64)>,
    /// A press landed on the stats panel: its release must not be read as a click on the fractal.
    panel_press: bool,
    /// A press landed on the fractal itself (not chrome, pills, panel or a resize edge): only
    /// such a press may turn into a click (recentre + audio) on release.
    fractal_press: bool,
    rendering_complete: bool,
    interpolation_complete: bool,
    frame_counter: u64,
    debug: bool,                              // Ctrl+D: frame/coverage counters
    show_hit_test_map: bool,                  // Ctrl+H: hit-map visualization
    show_mandelbrot_raw: bool,                // Ctrl+M: disable flood-fill interpolation
    debug_hit_colours: Vec<(u8, u8, u8)>,
}

/// Visible ARGB (0xFFrrggbb) → fluor α + darkness (opaque, RGB complemented).
#[inline]
fn visible_to_dark(px: u32) -> u32 {
    px ^ 0x00FF_FFFF
}

impl MandelbrotApp {
    pub fn new(center: Complex64, zoom: Complex64, boundary_path: Vec<Complex64>) -> Self {
        let audio_player = match AudioPlayer::new() {
            Ok(player) => {
                println!("Audio initialized successfully");
                Some(player)
            }
            Err(e) => {
                eprintln!("Failed to initialize audio: {}", e);
                None
            }
        };

        let mut hit_counter: HitId = HIT_NONE;
        let mut frac_dd = Dropdown::new(
            &mut hit_counter,
            0.,
            0.,
            1.,
            1.,
            12.,
            vec![
                "f32".into(),
                "f64".into(),
                "f32 pert".into(),
                "F3 (8b)".into(),
                "F4 (16b)".into(),
                "F5 (32b)".into(),
                "F6 (64b)".into(),
                "F7 (128b)".into(),
            ],
        );
        let mut exp_dd = Dropdown::new(
            &mut hit_counter,
            0.,
            0.,
            1.,
            1.,
            12.,
            vec![
                "E3 (8b)".into(),
                "E4 (16b)".into(),
                "E5 (32b)".into(),
                "E6 (64b)".into(),
                "E7 (128b)".into(),
            ],
        );
        let mut theme_dd = Dropdown::new(
            &mut hit_counter,
            0.,
            0.,
            1.,
            1.,
            12.,
            crate::compute::colouring::THEME_NAMES.iter().map(|n| (*n).into()).collect(),
        );
        theme_dd.set_selected(0);
        // MANDEL_THEME=<name|index> starts with a given colour theme.
        if let Ok(spec) = std::env::var("MANDEL_THEME") {
            match crate::compute::colouring::theme_from_spec(&spec) {
                Some(i) => {
                    theme_dd.set_selected(i);
                    crate::compute::colouring::set_theme(i);
                }
                None => eprintln!("MANDEL_THEME: unrecognized theme '{}'", spec),
            }
        }
        frac_dd.set_selected(2); // f32 pert — the pipeline the rig runs by default
        exp_dd.set_selected(2); // E5 — the historical spirix default

        // MANDEL_PRECISION=f32|f64|f<3..7>e<3..7> starts the app in a given mode — handy for
        // scripted spirix sweeps without clicking through the dropdowns.
        if let Ok(spec) = std::env::var("MANDEL_PRECISION") {
            let spec = spec.to_lowercase();
            let b = spec.as_bytes();
            match spec.as_str() {
                "f32" => frac_dd.set_selected(0),
                "f64" => frac_dd.set_selected(1),
                "pert" | "f32pert" | "perturbed" => frac_dd.set_selected(2),
                _ if b.len() == 4 && b[0] == b'f' && b[2] == b'e' => {
                    let f = (b[1] as char).to_digit(10).unwrap_or(7).clamp(3, 7);
                    let e = (b[3] as char).to_digit(10).unwrap_or(5).clamp(3, 7);
                    frac_dd.set_selected(f as usize);
                    exp_dd.set_selected(e as usize - 3);
                }
                _ => eprintln!("MANDEL_PRECISION: unrecognized spec '{}'", spec),
            }
        }
        exp_dd.set_enabled(frac_dd.selected() >= 3);

        let mut app = Self {
            buffer: MandelbrotBuffer::new(INITIAL_WIDTH, INITIAL_HEIGHT, center, zoom, boundary_path),
            chrome: None,
            hit_counter,
            audio_player,
            frac_dd,
            exp_dd,
            theme_dd,
            current_focus: None,
            dd_open_state: (false, false, false),
            fractal: Vec::new(),
            cursor_pos: (0., 0.),
            accumulated_zoom: 0.,
            accumulated_rotation: 0.,
            drag_start_pos: None,
            dragging: false,
            stats_panel: None,
            panel_press: false,
            fractal_press: false,
            rendering_complete: false,
            interpolation_complete: false,
            frame_counter: 0,
            debug: false,
            show_hit_test_map: false,
            show_mandelbrot_raw: false,
            debug_hit_colours: Vec::new(),
        };
        app.buffer.current_precision_mode = app.selected_mode();
        app
    }

    fn restart_progressive_render(&mut self) {
        self.rendering_complete = false;
        self.interpolation_complete = false;
    }

    /// Size + place the precision pills under the stats block, all in effective-span units so
    /// they track the host's Ctrl+zoom together with the rest of the overlay.
    fn layout_widgets(&mut self, vp: Viewport) {
        let span = vp.effective_span();
        let font_size = span / 48.0;
        let line_height = font_size * 1.4;
        let margin_x = span / 72.0;
        let margin_y = (span / 72.0).max(3.0);
        let pill_h = line_height * 1.3;
        let pill_font = font_size * 0.9;
        // Stats panel: 3 lines starting at margin_y + 0.75 line, padded by 0.6 font each side
        // (see draw_overlay); pills sit one row below it.
        let pills_cy = margin_y + line_height * (0.75 + 3.0) + font_size * 1.2 + pill_h * 0.7;
        let frac_w = font_size * 7.5;
        let exp_w = font_size * 7.0;
        self.frac_dd
            .set_rect(margin_x + frac_w * 0.5, pills_cy, frac_w, pill_h);
        self.frac_dd.set_font_size(pill_font);
        self.exp_dd.set_rect(
            margin_x + frac_w + margin_x * 0.5 + exp_w * 0.5,
            pills_cy,
            exp_w,
            pill_h,
        );
        self.exp_dd.set_font_size(pill_font);
        let theme_w = font_size * 7.0;
        self.theme_dd.set_rect(
            margin_x + frac_w + margin_x * 0.5 + exp_w + margin_x * 0.5 + theme_w * 0.5,
            pills_cy,
            theme_w,
            pill_h,
        );
        self.theme_dd.set_font_size(pill_font);
    }

    /// Switch colour theme: every pixel is recoloured by recomputing the view.
    fn apply_theme(&mut self, index: usize) {
        let index = index % crate::compute::colouring::THEME_NAMES.len();
        if index == crate::compute::colouring::theme() {
            return;
        }
        crate::compute::colouring::set_theme(index);
        self.theme_dd.set_selected(index);
        println!("Theme: {}", crate::compute::colouring::THEME_NAMES[index]);
        self.buffer.stop_calculation();
        self.buffer.shared_display_buffer.lock().unwrap().fill(0x00000000);
        self.buffer.calculation_buffer_f64.lock().unwrap().clear();
        self.buffer.viewport_changed = true;
        self.buffer.start_calculation();
        self.restart_progressive_render();
    }

    /// The precision mode the two dropdowns currently describe.
    fn selected_mode(&self) -> PrecisionMode {
        match self.frac_dd.selected() {
            0 => PrecisionMode::F32,
            1 => PrecisionMode::F64,
            2 => PrecisionMode::Perturbed,
            i => PrecisionMode::Spirix {
                f: i as u8, // index 3 = F3 … index 7 = F7
                e: self.exp_dd.selected() as u8 + 3,
            },
        }
    }

    /// Fold pending dropdown changes into the pipeline: commit row clicks, swap the precision
    /// mode (stop → clear → restart, the old Ctrl+P machinery), grey the exponent pill for
    /// IEEE modes, and invalidate chrome on popup open/close so stale row stamps leave the hit map.
    fn poll_dropdowns(&mut self, ctx: &mut Context) {
        if let Some(i) = self.theme_dd.take_change() {
            let _ = i;
            let idx = self.theme_dd.selected();
            self.apply_theme(idx);
            ctx.window.request_redraw();
        }
        let changed =
            self.frac_dd.take_change().is_some() | self.exp_dd.take_change().is_some();
        if changed {
            self.exp_dd.set_enabled(self.frac_dd.selected() >= 3);
            let mode = self.selected_mode();
            if mode != self.buffer.current_precision_mode {
                self.buffer.current_precision_mode = mode;
                println!("Precision mode: {}", mode.label());
                self.buffer.stop_calculation();
                self.buffer
                    .shared_display_buffer
                    .lock()
                    .unwrap()
                    .fill(0x00000000);
                self.buffer.calculation_buffer_f64.lock().unwrap().clear();
                self.buffer.viewport_changed = true;
                self.buffer.start_calculation();
                self.restart_progressive_render();
            }
            ctx.window.request_redraw();
        }
        let open_now = (self.frac_dd.is_open(), self.exp_dd.is_open(), self.theme_dd.is_open());
        if open_now != self.dd_open_state {
            self.dd_open_state = open_now;
            if let Some(chrome) = self.chrome.as_mut() {
                chrome.invalidate_chrome();
            }
            ctx.window.request_redraw();
        }
    }

    fn change_focus(&mut self, new_focus: Option<HitId>) {
        let old = self.current_focus;
        if old != new_focus {
            widget::apply_focus_change(self, old, new_focus);
            self.current_focus = new_focus;
        }
    }

    /// Left-click release without a drag: the click actions from the original app.
    fn handle_click(&mut self, shift: bool, ctrl: bool) {
        let point = MandelbrotBuffer::screen_to_complex(
            self.cursor_pos.0,
            self.cursor_pos.1,
            self.buffer.width as f64,
            self.buffer.height as f64,
            self.buffer.zoom,
            self.buffer.center,
        );
        let point_c64 = Complex64::new(point.r().into(), point.i().into());

        if shift {
            // Shift+click: Show point statistics
            let result = calculate_mandelbrot_point_f64(point_c64, config::MAX_ITERATIONS as f64, false);
            let zoom_c64 = Complex64::new(self.buffer.zoom.r().into(), self.buffer.zoom.i().into());
            println!(
                "\n{}",
                MandelbrotBuffer::format_point_statistics(
                    point_c64,
                    zoom_c64,
                    result.period,
                    Complex64::new(result.escape_velocity, 0.),
                    result.coords,
                    result.flame_c,
                    result.slope,
                    Complex64::new(1.0, 0.0) // multiplier is unused
                )
            );
        } else if ctrl {
            // Ctrl+click: Trace boundary path and play FFT audio
            let result = calculate_mandelbrot_point_f64(point_c64, config::MAX_ITERATIONS as f64, false);
            if result.period != 0 {
                println!("C:[{}, {}]", point_c64.re, point_c64.im);
            } else {
                self.buffer.boundary_path = trace_path(point_c64);
                if let Some(ref audio_player) = self.audio_player {
                    let (left, right) = path_to_audio_fft(&self.buffer.boundary_path);
                    audio_player.play_stereo_samples(left, right, 44100);
                }
            }
        } else {
            // Normal click: Center view on clicked point and play audio
            self.buffer.center = point;
            self.buffer.transform_requested = true;
            if let Some(ref audio_player) = self.audio_player {
                let (left, right) = path_to_audio(point_c64, config::MAX_ITERATIONS);
                audio_player.play_stereo_samples(left, right, 44100);
            }
        }
    }

    /// Project calculated points into the persistent fractal frame; returns covered-pixel count.
    fn project_calculations(&mut self) -> usize {
        let width = self.buffer.width;
        let height = self.buffer.height;
        let total = width * height;

        match self.buffer.current_precision_mode {
            PrecisionMode::F32 | PrecisionMode::F64 => {
                let calc_buffer = self.buffer.calculation_buffer_f64.lock().unwrap();
                let zoom = self.buffer.zoom_f64;
                let center = self.buffer.center_f64;
                let scale = self.buffer.diagonal_sq().sqrt();

                let mut coverage = vec![false; total];
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
                        if idx < self.fractal.len() {
                            self.fractal[idx] = 0xFF000000 | point.color;
                            coverage[idx] = true;
                        }
                    }
                }
                coverage.iter().filter(|&&c| c).count()
            }
            PrecisionMode::Perturbed | PrecisionMode::Spirix { .. } => {
                // The epoch thread writes epoch-tagged pixels (epoch in the top byte) straight
                // into the shared display buffer; epoch 0 = not yet calculated.
                let shared = self.buffer.shared_display_buffer.lock().unwrap();
                let n = total.min(shared.len()).min(self.fractal.len());
                let mut covered = 0;
                for i in 0..n {
                    if shared[i] >> 24 != 0 {
                        self.fractal[i] = 0xFF000000 | (shared[i] & 0x00FF_FFFF);
                        covered += 1;
                    }
                }
                covered
            }
        }
    }

    /// Single-direction flood fill, cycling direction each frame — spreads calculated colors
    /// into still-black pixels so the image fills in while the calculation is sparse.
    fn flood_fill_step(&mut self) {
        let width = self.buffer.width;
        let height = self.buffer.height;
        if width < 2 || height < 2 {
            return;
        }
        let buf = &mut self.fractal;
        let len = buf.len();
        let direction = self.frame_counter % 4;
        let mut pixels_filled = 0usize;

        match direction {
            0 => {
                // Copy right - iterate RIGHT TO LEFT to avoid cascade
                for y in 0..height {
                    for x in (1..width - 1).rev() {
                        let idx = y * width + x;
                        let next_idx = idx + 1;
                        if next_idx < len
                            && buf[idx] != 0xFF000000
                            && buf[next_idx] == 0xFF000000
                        {
                            buf[next_idx] = buf[idx];
                            pixels_filled += 1;
                        }
                    }
                }
            }
            1 => {
                // Copy down - iterate BOTTOM TO TOP to avoid cascade
                for y in (1..height - 1).rev() {
                    for x in 0..width {
                        let idx = y * width + x;
                        let next_idx = (y + 1) * width + x;
                        if next_idx < len
                            && buf[idx] != 0xFF000000
                            && buf[next_idx] == 0xFF000000
                        {
                            buf[next_idx] = buf[idx];
                            pixels_filled += 1;
                        }
                    }
                }
            }
            2 => {
                // Copy left - iterate LEFT TO RIGHT to avoid cascade
                for y in 0..height {
                    for x in 1..width - 1 {
                        let idx = y * width + x;
                        let prev_idx = idx - 1;
                        if idx < len
                            && buf[idx] != 0xFF000000
                            && buf[prev_idx] == 0xFF000000
                        {
                            buf[prev_idx] = buf[idx];
                            pixels_filled += 1;
                        }
                    }
                }
            }
            _ => {
                // Copy up - iterate TOP TO BOTTOM to avoid cascade
                for y in 1..height - 1 {
                    for x in 0..width {
                        let idx = y * width + x;
                        let prev_idx = (y - 1) * width + x;
                        if idx < len
                            && buf[idx] != 0xFF000000
                            && buf[prev_idx] == 0xFF000000
                        {
                            buf[prev_idx] = buf[idx];
                            pixels_filled += 1;
                        }
                    }
                }
            }
        }

        // Mark interpolation complete after a full cycle of no changes
        if pixels_filled == 0 && direction == 3 {
            self.interpolation_complete = true;
        }
    }

    /// Progress bar + viewport stats, drawn topmost into `target` (first writers win the
    /// under-chain, so the fractal composes behind them).
    fn draw_overlay(
        &mut self,
        target: &mut [u32],
        ctx: &mut Context,
        projected_count: usize,
        is_complete: bool,
    ) {
        let width = self.buffer.width;
        let height = self.buffer.height;
        let span = ctx.viewport.effective_span() as f64;
        let font_size = span / 48.0;
        let line_height = font_size * 1.4;
        let margin_x = span / 72.0;
        let margin_y = (span / 72.0).max(3.0);

        // --- Progress bar ---
        let progress_y = margin_y as usize;
        let progress_start_x = (margin_x + line_height * 0.25).max(1.0) as usize;
        let progress_width = (span * 0.2) as usize;
        let total_pixels = width * height;
        let filled_width = if total_pixels > 0 {
            (projected_count * progress_width) / total_pixels
        } else {
            0
        };

        let black = 0xFFFF_FFFFu32; // opaque, darkness 255 = visible black
        // Under-blend, not overwrite: an open dropdown popup paints before this and must win.
        let mut put = |x: usize, y: usize, dark: u32| {
            if x < width && y < height {
                let idx = y * width + x;
                target[idx] = target[idx].under(dark, BlendMode::Normal);
            }
        };

        // Border
        for x in 0..progress_width + 2 {
            put(progress_start_x - 1 + x, progress_y - 1, black);
            put(progress_start_x - 1 + x, progress_y + 1, black);
        }
        put(progress_start_x - 1, progress_y, black);
        put(progress_start_x + progress_width, progress_y, black);
        // T-caps
        put(progress_start_x - 1, progress_y.saturating_sub(2), black);
        put(progress_start_x + progress_width, progress_y.saturating_sub(2), black);
        put(progress_start_x - 1, progress_y + 2, black);
        put(progress_start_x + progress_width, progress_y + 2, black);

        // Fill: gradient red → yellow → green (all green when complete)
        for x in 0..progress_width {
            let visible = if x < filled_width {
                if is_complete {
                    0xFF_00_FF_00
                } else if x < progress_width / 2 {
                    let g = ((x * 255 * 2) / progress_width) as u32;
                    0xFF_FF_00_00 | (g << 8)
                } else {
                    let r = (((progress_width - x) * 255 * 2) / progress_width) as u32;
                    0xFF_00_FF_00 | (r << 16)
                }
            } else {
                0xFF_20_20_20
            };
            put(progress_start_x + x, progress_y, visible_to_dark(visible));
        }

        // --- Viewport stats ---
        let center_re: f64 = self.buffer.center.r().into();
        let center_im: f64 = self.buffer.center.i().into();
        let zoom_re: f64 = self.buffer.zoom.r().into();
        let zoom_im: f64 = self.buffer.zoom.i().into();
        let zoom_mag = (zoom_re * zoom_re + zoom_im * zoom_im).sqrt();
        // Precision line lives in the dropdown pills below the stats now.
        let stats = [
            format!("Center: {:.6}{:+.6}i", center_re, center_im),
            format!("Zoom: 2^{:.2}", zoom_mag.log2()),
            format!("Rotation: {:.1}°", zoom_im.atan2(zoom_re).to_degrees()),
        ];

        let white = visible_to_dark(0xFFFF_FFFF);
        let mut canvas = Canvas::new(target, width, height, ctx.damage);
        let stat_style = TextStyle::new(font_size as f32, white).font("Oxanium");
        let pad = font_size * 0.6;
        let text_x = margin_x + pad;
        let first_y = margin_y + line_height * 0.75 + font_size * 0.5 + pad;
        let mut y_pos = first_y;
        let mut text_w: f32 = 0.0;
        for stat in &stats {
            text_w = text_w.max(ctx.text.measure_text(stat, &stat_style));
            ctx.text.draw_text_left(&mut canvas, stat, text_x as f32, y_pos as f32, &stat_style, None, None);
            y_pos += line_height;
        }
        // Readout panel: a squircle pill in the dropdowns' style, painted after the text (first
        // writers win the under-chain) and remembered as the window's drag handle.
        let panel_x = margin_x;
        let panel_y = first_y - font_size * 0.7 - pad;
        let panel_w = text_w as f64 + pad * 2.0;
        let panel_h = line_height * 2.0 + font_size * 1.4 + pad * 2.0;
        self.stats_panel = Some((panel_x, panel_y, panel_w, panel_h));
        let r = (font_size * 0.9) as f32;
        let stroke = (font_size * 0.08).max(1.0) as isize;
        paint::draw_squircle_rrect_f(
            &mut canvas,
            panel_x as isize + stroke,
            panel_y as isize + stroke,
            (panel_w as isize - 2 * stroke).max(0),
            (panel_h as isize - 2 * stroke).max(0),
            theme::TEXTBOX_FILL,
            paint::asymmetric_radii((r - stroke as f32).max(1.0)),
            1.75,
        );
        paint::draw_squircle_rrect_two_tone_f(
            &mut canvas,
            panel_x as isize,
            panel_y as isize,
            panel_w as isize,
            panel_h as isize,
            theme::TEXTBOX_LIGHT_EDGE,
            theme::TEXTBOX_SHADOW_EDGE,
            paint::asymmetric_radii(r),
            1.75,
            None,
            0,
        );

        // --- Debug counters (Ctrl+D) ---
        if self.debug {
            let counter_size = (span / 64.0) as f32;
            let coverage_pct = if total_pixels > 0 {
                projected_count as f64 * 100.0 / total_pixels as f64
            } else {
                0.0
            };
            ctx.text.draw_text_left(
                &mut canvas,
                &format!("F:{}  cover:{:.1}%", self.frame_counter, coverage_pct),
                counter_size,
                height as f32 - counter_size,
                &TextStyle::new(counter_size, white).font("Oxanium"),
                None,
                None,
            );
        }
    }
}

fn timing_enabled() -> bool {
    static TIMING: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TIMING.get_or_init(|| std::env::var_os("MANDEL_TIMING").is_some())
}

/// Dispatch / overlay walk: the two precision dropdowns (rows appear only while open), then
/// the chrome's four buttons.
impl Container for MandelbrotApp {
    fn visit(&mut self, f: &mut dyn FnMut(&mut dyn Widget)) {
        f(&mut self.frac_dd);
        self.frac_dd.visit_rows(f);
        f(&mut self.exp_dd);
        self.exp_dd.visit_rows(f);
        f(&mut self.theme_dd);
        self.theme_dd.visit_rows(f);
        if let Some(chrome) = self.chrome.as_mut() {
            chrome.visit(f);
        }
    }
}

impl FluorApp for MandelbrotApp {
    type UserEvent = ();

    fn title(&self) -> &str {
        "Mandelbrot Exploder"
    }

    fn initial_size(&self, monitor: (u32, u32)) -> (u32, u32) {
        (
            (INITIAL_WIDTH as u32).min(monitor.0),
            (INITIAL_HEIGHT as u32).min(monitor.1),
        )
    }

    fn init(&mut self, ctx: &mut Context) {
        // Oxanium for the stats overlay (fluor's chrome text uses its bundled Open Sans).
        let db = ctx.text.font_system_mut().db_mut();
        db.load_font_data(include_bytes!("../assets/Oxanium/Oxanium-Regular.ttf").to_vec());
        db.load_font_data(include_bytes!("../assets/Oxanium/Oxanium-Bold.ttf").to_vec());

        self.chrome = Some(DefaultChrome::new(
            ctx.viewport,
            "Mandelbrot Exploder",
            None,
            None,
            &mut self.hit_counter,
        ));

        let w = ctx.viewport.width_px as usize;
        let h = ctx.viewport.height_px as usize;
        self.buffer.resize(w, h); // allocates pipeline buffers + starts the calculation thread
        self.fractal = vec![0xFF000000; w * h];
        self.layout_widgets(ctx.viewport);

        // Debug hook: MANDEL_OPEN_DD=1 opens the fraction popup at startup (screenshots,
        // scripted layout checks — no input injection needed).
        if std::env::var("MANDEL_OPEN_DD").is_ok() {
            self.frac_dd.open();
        }
    }

    fn on_resize(&mut self, width: u32, height: u32, ctx: &mut Context) {
        if let Some(chrome) = self.chrome.as_mut() {
            chrome.resize(ctx.viewport);
            chrome.set_full_edge(ctx.is_maximized);
        }
        let w = width as usize;
        let h = height as usize;
        // on_resize also fires for UI-zoom (ru) changes with unchanged dimensions — don't
        // restart the calculation pipeline for those.
        if self.buffer.width != w || self.buffer.height != h {
            self.buffer.resize(w, h);
            self.fractal = vec![0xFF000000; w * h];
            self.restart_progressive_render();
        }
        self.layout_widgets(ctx.viewport);
        ctx.window.request_redraw();
    }

    fn on_event(&mut self, event: &Event, ctx: &mut Context) -> EventResponse {
        match event {
            Event::CursorMoved { .. } => {
                // The event's own x/y are SCREEN coordinates (the OS surface is fullscreen in
                // fluor's compositor model); ctx.cursor_x/y carry the same position translated
                // to window-relative pixels, which is what all our viewport math expects.
                // Using the raw event coords offset every zoom anchor by the window origin.
                let position = (ctx.cursor_x as f64, ctx.cursor_y as f64);
                self.cursor_pos = position;

                if let Some(start_pos) = self.drag_start_pos {
                    self.dragging = true;

                    if ctx.modifiers.shift_key() {
                        // Shift+drag: Rotate around center
                        let center_x = self.buffer.width as f64 / 2.0;
                        let center_y = self.buffer.height as f64 / 2.0;
                        let start_angle = (start_pos.1 - center_y).atan2(start_pos.0 - center_x);
                        let current_angle = (position.1 - center_y).atan2(position.0 - center_x);
                        self.accumulated_rotation += current_angle - start_angle;
                        self.buffer.transform_requested = true;
                    } else if ctx.modifiers.control_key() {
                        // Ctrl+drag: Zoom based on vertical movement
                        self.accumulated_zoom += -(position.1 - start_pos.1) / 100.0;
                        self.buffer.transform_requested = true;
                    } else {
                        // Normal drag: Pan (f64 viewport is derived in transform_buffer)
                        let start_point = MandelbrotBuffer::screen_to_complex(
                            start_pos.0,
                            start_pos.1,
                            self.buffer.width as f64,
                            self.buffer.height as f64,
                            self.buffer.zoom,
                            self.buffer.center,
                        );
                        let current_point = MandelbrotBuffer::screen_to_complex(
                            position.0,
                            position.1,
                            self.buffer.width as f64,
                            self.buffer.height as f64,
                            self.buffer.zoom,
                            self.buffer.center,
                        );
                        self.buffer.center += start_point - current_point;

                        self.buffer.transform_requested = true;
                    }

                    self.drag_start_pos = Some(position);
                    ctx.window.request_redraw();
                    return EventResponse::Handled;
                }

                // Not dragging: hover the chrome buttons + dropdown pills/rows
                let hit = self
                    .chrome
                    .as_ref()
                    .map(|c| c.hit_at(ctx.cursor_x, ctx.cursor_y))
                    .unwrap_or(HIT_NONE);
                let mut changed = self
                    .chrome
                    .as_mut()
                    .map(|c| c.set_hover(hit))
                    .unwrap_or(false);
                changed |= self.frac_dd.sync_hover(hit);
                changed |= self.exp_dd.sync_hover(hit);
                changed |= self.theme_dd.sync_hover(hit);
                if changed {
                    ctx.window.request_redraw();
                }
                EventResponse::Pass
            }

            Event::CursorLeft => {
                let mut changed = self
                    .chrome
                    .as_mut()
                    .map(|c| c.set_hover(HIT_NONE))
                    .unwrap_or(false);
                changed |= self.frac_dd.sync_hover(HIT_NONE);
                changed |= self.exp_dd.sync_hover(HIT_NONE);
                changed |= self.theme_dd.sync_hover(HIT_NONE);
                if changed {
                    ctx.window.request_redraw();
                }
                EventResponse::Pass
            }

            Event::MouseWheel { delta } => {
                // Ctrl+scroll never reaches us — the host consumes it for UI zoom.
                let scroll_amount = match delta {
                    MouseScrollDelta::Lines(_, y) => *y as f64,
                    MouseScrollDelta::Pixels(_, y) => *y as f64 / 50.,
                };
                if ctx.modifiers.shift_key() {
                    // Shift+scroll: Rotate Mandelbrot
                    self.accumulated_rotation += scroll_amount * std::f64::consts::PI / 8.0;
                } else {
                    // Plain scroll: Zoom Mandelbrot
                    self.accumulated_zoom += scroll_amount;
                }
                self.buffer.transform_requested = true;
                ctx.window.request_redraw();
                EventResponse::Handled
            }

            Event::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
            } => {
                self.fractal_press = false;
                let hit = self
                    .chrome
                    .as_ref()
                    .map(|c| c.hit_at(ctx.cursor_x, ctx.cursor_y))
                    .unwrap_or(HIT_NONE);

                // Precision dropdowns first: pills toggle, rows commit.
                if self.frac_dd.owns_hit(hit) || self.exp_dd.owns_hit(hit) || self.theme_dd.owns_hit(hit) {
                    let response = widget::dispatch_click(
                        self,
                        hit,
                        ctx.cursor_x,
                        ctx.cursor_y,
                        ctx.modifiers,
                    );
                    if hit == self.frac_dd.hit_id() || hit == self.exp_dd.hit_id() || hit == self.theme_dd.hit_id() {
                        self.change_focus(Some(hit));
                    }
                    self.poll_dropdowns(ctx);
                    ctx.window.request_redraw();
                    return response;
                }

                // A press outside an open popup dismisses it — and must NOT fall through
                // into a fractal pan, or dismissing would jerk the viewport.
                if self.frac_dd.is_open() || self.exp_dd.is_open() || self.theme_dd.is_open() {
                    self.frac_dd.close();
                    self.exp_dd.close();
                    self.theme_dd.close();
                    self.change_focus(None);
                    self.poll_dropdowns(ctx);
                    ctx.window.request_redraw();
                    return EventResponse::Handled;
                }

                // Chrome buttons next
                if let Some(chrome) = self.chrome.as_ref() {
                    if chrome.owns_hit(hit) {
                        if hit == chrome.close_btn.id() {
                            self.buffer.stop_calculation();
                            return EventResponse::Close;
                        } else if hit == chrome.max_btn.id() {
                            return EventResponse::ToggleMaximized;
                        } else if hit == chrome.min_btn.id() {
                            return EventResponse::Minimize;
                        }
                        return EventResponse::Handled; // app-icon orb slot
                    }
                }

                // Resize edges — the host runs the actual drag
                let edge = chrome::get_resize_edge(ctx.viewport, ctx.cursor_x, ctx.cursor_y);
                if edge != ResizeEdge::None {
                    self.drag_start_pos = None;
                    self.dragging = false;
                    return EventResponse::StartResize(edge);
                }

                // The stats readout doubles as the window's drag handle.
                if let Some((px, py, pw, ph)) = self.stats_panel {
                    let (cx, cy) = self.cursor_pos;
                    if cx >= px && cx < px + pw && cy >= py && cy < py + ph {
                        self.change_focus(None);
                        self.drag_start_pos = None;
                        self.dragging = false;
                        self.panel_press = true;
                        return EventResponse::StartWindowDrag;
                    }
                }

                // Everywhere else: drop widget focus and arm a fractal pan/rotate/zoom drag
                self.change_focus(None);
                self.drag_start_pos = Some(self.cursor_pos);
                self.dragging = false;
                self.fractal_press = true;
                EventResponse::Handled
            }

            Event::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
            } => {
                let was_dragging = self.dragging;
                self.drag_start_pos = None;
                self.dragging = false;

                let armed = self.fractal_press;
                self.fractal_press = false;
                if self.panel_press {
                    // Window-drag release (or a click on the panel): never a fractal click.
                    self.panel_press = false;
                    return EventResponse::Handled;
                }
                // Only a press that landed on the fractal can become a click: presses that went
                // to the pills or chrome buttons (whose release may land anywhere once, say,
                // maximize has relaid the window) used to fall through here and recentre.
                if !was_dragging && armed {
                    self.handle_click(ctx.modifiers.shift_key(), ctx.modifiers.control_key());
                    ctx.window.request_redraw();
                }
                EventResponse::Handled
            }

            Event::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Right,
            } => {
                let point = MandelbrotBuffer::screen_to_complex(
                    self.cursor_pos.0,
                    self.cursor_pos.1,
                    self.buffer.width as f64,
                    self.buffer.height as f64,
                    self.buffer.zoom,
                    self.buffer.center,
                );
                let point_c64 = Complex64::new(point.r().into(), point.i().into());
                self.buffer.switch_view_mode(point_c64);
                self.restart_progressive_render();
                ctx.window.request_redraw();
                EventResponse::Handled
            }

            Event::KeyboardInput { event: kev } => {
                // Focused dropdown gets first crack (Enter/Space/arrows/Escape).
                if let Some(focus_id) = self.current_focus {
                    let response =
                        widget::dispatch_key(self, focus_id, kev, ctx.modifiers, ctx.text);
                    if response == EventResponse::Handled {
                        self.poll_dropdowns(ctx);
                        ctx.window.request_redraw();
                        return response;
                    }
                }
                if kev.state != ElementState::Pressed {
                    return EventResponse::Pass;
                }
                let ctrl = ctx.modifiers.control_key();
                let Key::Character(c) = &kev.logical_key else {
                    return EventResponse::Pass;
                };
                // Ctrl +/-/0 never arrive here — the host consumes them for UI zoom.
                match c.as_str() {
                    "+" | "=" => {
                        self.accumulated_zoom += 1.0;
                        self.buffer.transform_requested = true;
                    }
                    "-" => {
                        self.accumulated_zoom -= 1.0;
                        self.buffer.transform_requested = true;
                    }
                    "t" | "T" if !ctrl => {
                        let next = crate::compute::colouring::theme() + 1;
                        self.apply_theme(next);
                        ctx.window.request_redraw();
                    }
                    "s" | "S" if !ctrl => {
                        println!("Starting search for interesting spot...");
                        self.buffer.search_for_interesting_spot(10);
                    }
                    "d" | "D" if ctrl => {
                        self.debug = !self.debug;
                        println!("Debug counters: {}", if self.debug { "ON" } else { "OFF" });
                    }
                    "h" | "H" if ctrl => {
                        self.show_hit_test_map = !self.show_hit_test_map;
                        println!(
                            "Hit test map: {}",
                            if self.show_hit_test_map { "ON" } else { "OFF" }
                        );
                        use rand::Rng;
                        let mut rng = rand::thread_rng();
                        self.debug_hit_colours.clear();
                        for _ in 0..=u8::MAX {
                            self.debug_hit_colours
                                .push((rng.gen(), rng.gen(), rng.gen()));
                        }
                    }
                    "m" | "M" if ctrl => {
                        self.show_mandelbrot_raw = !self.show_mandelbrot_raw;
                        println!(
                            "Interpolation (flood fill): {}",
                            if self.show_mandelbrot_raw {
                                "OFF (showing raw projections only)"
                            } else {
                                "ON"
                            }
                        );
                    }
                    _ => return EventResponse::Pass,
                }
                ctx.window.request_redraw();
                EventResponse::Handled
            }

            Event::Focused(focused) => {
                if let Some(chrome) = self.chrome.as_mut() {
                    if chrome.set_focused(*focused) {
                        ctx.window.request_redraw();
                    }
                }
                EventResponse::Pass
            }

            Event::CloseRequested => {
                self.buffer.stop_calculation();
                EventResponse::Pass
            }

            _ => EventResponse::Pass,
        }
    }

    fn render(&mut self, target: &mut [u32], ctx: &mut Context) {
        let frame_t0 = std::time::Instant::now();
        let width = ctx.viewport.width_px as usize;
        let height = ctx.viewport.height_px as usize;
        // A resize can deliver a frame before on_resize has re-synced the pipeline; skip it.
        if self.buffer.width != width || self.buffer.height != height {
            return;
        }
        if self.fractal.len() != width * height {
            self.fractal = vec![0xFF000000; width * height];
        }

        let mut projected_count = 0;
        let mut is_complete = false;

        if self.buffer.view_mode == ViewMode::Brot {
            // MANDEL_AUTOZOOM=n: zoom in one notch at the screen centre every n frames (a
            // scripted navigation for timing with MANDEL_TIMING=1).
            static AUTOZOOM: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
            let autozoom = *AUTOZOOM.get_or_init(|| {
                std::env::var("MANDEL_AUTOZOOM").ok().and_then(|v| v.parse().ok()).filter(|n| *n > 0)
            });
            if let Some(n) = autozoom {
                if self.frame_counter > 0 && self.frame_counter % n == 0 {
                    self.cursor_pos = (width as f64 / 2., height as f64 / 2.);
                    self.accumulated_zoom += 1.0;
                }
            }
            static TIMING: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let timing = *TIMING.get_or_init(|| std::env::var_os("MANDEL_TIMING").is_some());

            // Apply accumulated wheel/drag transforms anchored at the cursor
            // The perturbation thread is persistent and follows the viewport epoch; the other
            // pipelines are stopped around a viewport change and restarted after it.
            let persistent = self.buffer.current_precision_mode == PrecisionMode::Perturbed;
            if self.accumulated_zoom != 0. {
                let t = std::time::Instant::now();
                if !persistent {
                    self.buffer.stop_calculation();
                }
                if timing {
                    println!("timing: stop (zoom) {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
                }
                self.buffer.apply_zoom(self.cursor_pos, self.accumulated_zoom);
                self.accumulated_zoom = 0.;
                self.buffer.transform_requested = true;
            }
            if self.accumulated_rotation != 0. {
                if !persistent {
                    self.buffer.stop_calculation();
                }
                self.buffer
                    .apply_rotation(self.cursor_pos, self.accumulated_rotation);
                self.accumulated_rotation = 0.;
                self.buffer.transform_requested = true;
            }
            if self.buffer.transform_requested {
                let t0 = std::time::Instant::now();
                if !persistent {
                    self.buffer.stop_calculation();
                }
                let t1 = std::time::Instant::now();
                self.buffer.transform_buffer();
                let t2 = std::time::Instant::now();
                self.buffer.start_calculation();
                let t3 = std::time::Instant::now();
                if timing {
                    println!(
                        "timing: stop {:.1} ms, transform {:.1} ms, start {:.1} ms",
                        (t1 - t0).as_secs_f64() * 1e3,
                        (t2 - t1).as_secs_f64() * 1e3,
                        (t3 - t2).as_secs_f64() * 1e3
                    );
                }
                self.buffer.transform_requested = false;
                self.restart_progressive_render();
                ctx.window.request_redraw();
            }

            // Viewport changed → wipe the persistent frame before reprojecting
            if self.buffer.viewport_changed {
                self.fractal.fill(0xFF000000);
                self.buffer.viewport_changed = false;
            }

            projected_count = self.project_calculations();

            if !self.show_mandelbrot_raw && !self.interpolation_complete {
                self.flood_fill_step();
            }

            self.buffer.plot_boundary_path(&mut self.fractal);

            let total_pixels = width * height;
            is_complete = projected_count == total_pixels;
            self.rendering_complete = is_complete;
        } else {
            // FFT view: the buffer holds the plotted visualization; show it as-is.
            let shared = self.buffer.shared_display_buffer.lock().unwrap();
            let n = self.fractal.len().min(shared.len());
            for i in 0..n {
                self.fractal[i] = 0xFF000000 | (shared[i] & 0x00FF_FFFF);
            }
            drop(shared);
            self.rendering_complete = true;
            self.interpolation_complete = true;
        }

        self.frame_counter += 1;
        if timing_enabled() {
            let dt = frame_t0.elapsed().as_secs_f64() * 1e3;
            if dt > 20.0 {
                println!("timing: frame {} took {:.1} ms (projected {projected_count})", self.frame_counter, dt);
            }
        }

        // MANDEL_SNAPSHOT=path.ppm[,seconds]: write the composited fractal frame after the
        // given time (default 5 s) and exit — scripted screenshots of the precision modes.
        if let Ok(spec) = std::env::var("MANDEL_SNAPSHOT") {
            static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let start = *START.get_or_init(std::time::Instant::now);
            let (path, secs) = match spec.rsplit_once(',') {
                Some((p, s)) => (p.to_string(), s.trim().parse::<f64>().unwrap_or(5.)),
                None => (spec.clone(), 5.),
            };
            if start.elapsed().as_secs_f64() >= secs {
                let (w, h) = (self.buffer.width, self.buffer.height);
                let mut data = format!("P6\n{w} {h}\n255\n").into_bytes();
                for px in self.fractal.iter().take(w * h) {
                    data.extend_from_slice(&[(px >> 16) as u8, (px >> 8) as u8, *px as u8]);
                }
                match std::fs::write(&path, data) {
                    Ok(()) => println!(
                        "Snapshot written to {path} (coverage {:.1}%)",
                        100.0 * projected_count as f64 / (w * h).max(1) as f64
                    ),
                    Err(e) => eprintln!("Snapshot failed: {e}"),
                }
                std::process::exit(0);
            }
        }

        if self.chrome.is_none() {
            return;
        }

        // Front-to-back: perimeter hairline first (must win the under-chain at the window
        // edge), then open dropdown popups (they cover everything), then the stats/progress
        // overlay, then the pills, then chrome controls, then the fractal underneath everything.
        if let Some(chrome) = self.chrome.as_mut() {
            chrome.rasterize_perimeter(target, width, height, ctx.clip_mask);
            chrome.rasterize_chrome(ctx.damage, ctx.text, ctx.clip_mask);

            let mut canvas = Canvas::new(target, width, height, ctx.damage);
            self.frac_dd.render_popup_into(
                &mut canvas,
                ctx.text,
                None,
                Some(&mut chrome.hit_test_map),
            );
            self.exp_dd.render_popup_into(
                &mut canvas,
                ctx.text,
                None,
                Some(&mut chrome.hit_test_map),
            );
            self.theme_dd.render_popup_into(
                &mut canvas,
                ctx.text,
                None,
                Some(&mut chrome.hit_test_map),
            );
        }

        if self.buffer.view_mode == ViewMode::Brot {
            self.draw_overlay(target, ctx, projected_count, is_complete);
        }

        if let Some(chrome) = self.chrome.as_mut() {
            let mut canvas = Canvas::new(target, width, height, ctx.damage);
            self.frac_dd.render_content_into(
                &mut canvas,
                0.,
                0.,
                ctx.text,
                None,
                Some(&mut chrome.hit_test_map),
            );
            self.exp_dd.render_content_into(
                &mut canvas,
                0.,
                0.,
                ctx.text,
                None,
                Some(&mut chrome.hit_test_map),
            );
            self.theme_dd.render_content_into(
                &mut canvas,
                0.,
                0.,
                ctx.text,
                None,
                Some(&mut chrome.hit_test_map),
            );
            chrome.flatten_into(target, width, height, None);
        }

        // Fractal at the bottom of the stack, converted visible-RGB → α + darkness.
        for (dst, &src) in target.iter_mut().zip(self.fractal.iter()) {
            *dst = dst.under(visible_to_dark(src), BlendMode::Normal);
        }
        ctx.damage.add_bounds(0, 0, width, height);

        // Ctrl+H: hit-map visualization replaces the frame entirely
        if self.show_hit_test_map && !self.debug_hit_colours.is_empty() {
            if let Some(chrome) = self.chrome.as_ref() {
                let map = chrome.hit_test_map();
                let n = map.len().min(target.len());
                for i in 0..n {
                    let (r, g, b) = self.debug_hit_colours[(map[i] as usize) % 256];
                    let visible =
                        0xFF000000 | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32);
                    target[i] = visible_to_dark(visible);
                }
            }
        }
    }

    fn cursor_for(&self, x: Coord, y: Coord, ctx: &Context) -> CursorIcon {
        if let Some(chrome) = self.chrome.as_ref() {
            let hit = chrome.hit_at(x, y);
            if chrome.owns_hit(hit) && hit != chrome.app_icon_btn.id() {
                return CursorIcon::Pointer;
            }
            if self.frac_dd.owns_hit(hit) || self.exp_dd.owns_hit(hit) || self.theme_dd.owns_hit(hit) {
                return CursorIcon::Pointer;
            }
        }
        match chrome::get_resize_edge(ctx.viewport, x, y) {
            ResizeEdge::Top | ResizeEdge::Bottom => CursorIcon::NsResize,
            ResizeEdge::Left | ResizeEdge::Right => CursorIcon::EwResize,
            ResizeEdge::TopLeft | ResizeEdge::BottomRight => CursorIcon::NwseResize,
            ResizeEdge::TopRight | ResizeEdge::BottomLeft => CursorIcon::NeswResize,
            ResizeEdge::None => CursorIcon::Default,
        }
    }

    fn hit_test_map(&self) -> Option<(&[HitId], usize, usize)> {
        self.chrome.as_ref().map(|c| {
            let (w, h) = c.dims();
            (c.hit_test_map(), w, h)
        })
    }

    fn overlay_deltas(&mut self) -> Vec<u32> {
        let count = self.hit_counter as usize + 1;
        widget::build_overlay_deltas(self, count)
    }

    fn tick(&mut self, _ctx: &mut Context) -> bool {
        // Keep redrawing while the progressive render or flood fill is still converging, or
        // while a scripted snapshot is waiting for its moment (it fires from the frame).
        self.buffer.view_mode == ViewMode::Brot
            && (!self.rendering_complete
                || !self.interpolation_complete
                || self.buffer.transform_requested
                || std::env::var_os("MANDEL_SNAPSHOT").is_some())
    }
}
