//! Throughput + oracle check for the perturbation engine.
//!
//! ```
//! cargo run --release --example perturb_bench
//! ```
//!
//! For each view: build the engine, time a random batch of pixels on all cores, then compare a
//! handful of pixels against the direct F7E3 grid cell (the rig's oracle) and, where the zoom
//! allows, the direct f32 pipeline.

#[path = "../src/compute/mod.rs"]
#[allow(dead_code)]
mod compute;
#[path = "../src/precision.rs"]
#[allow(dead_code)]
mod precision;
#[path = "../src/config.rs"]
#[allow(dead_code)]
mod config;

use compute::perturbation::{make_runner, Batch, PerturbEngine};
use compute::spirix_point_fn;
use num_complex::Complex64;
use rand::Rng;
use rayon::prelude::*;
use spirix::{CircleF7E3, ScalarF7E3};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

fn c7(re: f64, im: f64) -> CircleF7E3 {
    CircleF7E3::from((ScalarF7E3::from(re), ScalarF7E3::from(im)))
}

fn main() {
    let views: &[(&str, f64, f64, i32)] = &[
        ("whole set", -0.5, 0.0, 0),
        ("top bulb edge", -0.16, 1.04, -20),
        ("misiurewicz 2^-60", -0.10109636384562, 0.95628651080914, -60),
        ("misiurewicz 2^-100", -0.10109636384562, 0.95628651080914, -100),
        ("period-3 nucleus 2^-30", -1.7548776662466927, 0.0, -30),
    ];
    let max_iter = 1u64 << 16;
    let oracle = spirix_point_fn(7, 3);
    let batch = 1 << 14;
    let oracle_only_views = true;

    for &(name, re, im, log2_s) in views {
        let scale = 2f64.powi(log2_s);
        let center = c7(re, im);
        let t = Instant::now();
        let stop = Arc::new(AtomicBool::new(false));
        let mut engine = PerturbEngine::new(c7(scale, 0.), center, max_iter, Arc::clone(&stop)).expect("engine");
        let quick_ms = t.elapsed().as_secs_f64() * 1e3;
        // Let the worker's nucleus hunt land (it is asynchronous in the app) before timing.
        let deadline = Instant::now() + std::time::Duration::from_secs(3);
        while Instant::now() < deadline {
            if let Some(up) = engine.take_upgrade() {
                engine = PerturbEngine::with_reference(up, c7(scale, 0.), max_iter, Arc::clone(&stop));
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let build_ms = t.elapsed().as_secs_f64() * 1e3;
        print!("(quick stage {quick_ms:.0} ms) ");
        println!(
            "\n== {name}: S = 2^{log2_s}, reference {} entries, period {:?}, built in {build_ms:.0} ms",
            engine.reference.z64.len(),
            engine.reference.period
        );

        let mut rng = rand::thread_rng();
        let coords: Vec<CircleF7E3> = (0..batch)
            .map(|_| {
                let x: f64 = rng.gen_range(-0.8..0.8);
                let y: f64 = rng.gen_range(-0.5..0.5);
                center + c7(x * scale, y * scale)
            })
            .collect();

        let t = Instant::now();
        let results: Vec<_> = coords.par_iter().map(|&c| engine.point(c)).collect();
        let dt = t.elapsed().as_secs_f64();
        let inside = results.iter().filter(|r| r.period != 0).count();
        let mean_ev = results.iter().filter(|r| r.period == 0).map(|r| r.escape_velocity).sum::<f64>()
            / (batch - inside).max(1) as f64;
        println!(
            "perturbed: {batch} px in {:.0} ms = {:.1} µs/px (all cores); {inside} interior, mean ev {mean_ev:.1}; newton {}/{}",
            dt * 1e3,
            dt * 1e6 / batch as f64,
            engine.stats.newton_hits.load(std::sync::atomic::Ordering::Relaxed),
            engine.stats.newton_solves.load(std::sync::atomic::Ordering::Relaxed)
        );

        // Same batch through the runner (GPU when available), compared with the per-pixel CPU path.
        {
            let mut runner = make_runner();
            let mut work: Batch<usize> = Batch::default();
            for (i, &c) in coords.iter().enumerate() {
                let (slot, st) = engine.slot(c);
                work.push(i, slot, st);
            }
            let t = Instant::now();
            let mut got: Vec<Option<compute::mandelbrot_f64::MandelbrotResult>> = vec![None; work.len()];
            let mut steps = 0;
            while !work.is_empty() {
                for (i, r) in engine.step(&mut work, runner.as_mut(), 2048) {
                    got[i] = Some(r);
                }
                steps += 1;
            }
            let dt = t.elapsed().as_secs_f64();
            let mut period_mismatch = 0;
            let mut max_ev_err: f64 = 0.;
            let mut ev_close = 0;
            for (a, b) in results.iter().zip(got.iter()) {
                let b = b.as_ref().unwrap();
                if a.period != b.period {
                    period_mismatch += 1;
                } else if a.period == 0 {
                    let err = (a.escape_velocity - b.escape_velocity).abs();
                    max_ev_err = max_ev_err.max(err);
                    if err < 1e-2 {
                        ev_close += 1;
                    }
                }
            }
            println!(
                "{}: {batch} px in {:.0} ms = {:.1} µs/px over {steps} steps; vs CPU path: {period_mismatch} period mismatches, {ev_close} ev within 0.01, max |Δev| {max_ev_err:.3e}",
                runner.name(),
                dt * 1e3,
                dt * 1e6 / batch as f64
            );
        }

        // Oracle check on a few pixels.
        let n_check = 12;
        let t = Instant::now();
        let mut max_ev_err: f64 = 0.;
        let mut period_mismatch = 0;
        for c in coords.iter().take(n_check) {
            let p = engine.point(*c);
            let o = oracle(*c, max_iter);
            if p.period != o.period {
                period_mismatch += 1;
                let cc: Complex64 = (*c).into();
                println!("  period mismatch at {cc}: perturbed {} oracle {}", p.period, o.period);
            } else if p.period == 0 {
                let err = (p.escape_velocity - o.escape_velocity).abs();
                max_ev_err = max_ev_err.max(err);
            }
        }
        println!(
            "oracle (F7E3 direct): {n_check} px in {:.0} ms; {period_mismatch} period mismatches, max |Δev| {max_ev_err:.3e}",
            t.elapsed().as_secs_f64() * 1e3
        );
        if log2_s > -40 && oracle_only_views {
            let t = Instant::now();
            let _: Vec<_> = coords
                .par_iter()
                .map(|&c| {
                    let cc: Complex64 = c.into();
                    compute::calculate_mandelbrot_point_f32(cc, max_iter as f64, false)
                })
                .collect();
            let dt = t.elapsed().as_secs_f64();
            println!("direct f32: {batch} px in {:.0} ms = {:.1} µs/px", dt * 1e3, dt * 1e6 / batch as f64);
        }
    }

    // Dive: from the top-bulb edge, repeatedly recentre on the slowest-escaping pixel and zoom
    // by 2^-8, so every stop is a genuine boundary view at that depth. At each stop time a
    // batch and check a few pixels against the oracle.
    let mut dive_runner = make_runner();
    println!("\n== dive ({})", dive_runner.name());
    let mut center = c7(-0.16, 1.04);
    let mut log2_s: i32 = -8;
    let mut rng = rand::thread_rng();
    while log2_s >= -120 {
        let scale = 2f64.powi(log2_s);
        let t = Instant::now();
        let stop = Arc::new(AtomicBool::new(false));
        let Some(mut engine) = PerturbEngine::new(c7(scale, 0.), center, max_iter, Arc::clone(&stop)) else {
            break;
        };
        let deadline = Instant::now() + std::time::Duration::from_secs(3);
        while Instant::now() < deadline {
            if let Some(up) = engine.take_upgrade() {
                engine = PerturbEngine::with_reference(up, c7(scale, 0.), max_iter, Arc::clone(&stop));
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let build_ms = t.elapsed().as_secs_f64() * 1e3;
        let coords: Vec<CircleF7E3> = (0..batch)
            .map(|_| {
                let x: f64 = rng.gen_range(-0.8..0.8);
                let y: f64 = rng.gen_range(-0.5..0.5);
                center + c7(x * scale, y * scale)
            })
            .collect();
        let t = Instant::now();
        let results: Vec<_> = {
            let mut work: Batch<usize> = Batch::default();
            for (i, &c) in coords.iter().enumerate() {
                let (slot, st) = engine.slot(c);
                work.push(i, slot, st);
            }
            let mut got: Vec<Option<compute::mandelbrot_f64::MandelbrotResult>> = vec![None; work.len()];
            while !work.is_empty() {
                for (i, r) in engine.step(&mut work, dive_runner.as_mut(), 2048) {
                    got[i] = Some(r);
                }
            }
            got.into_iter().map(|r| r.unwrap()).collect()
        };
        let dt = t.elapsed().as_secs_f64();
        let inside = results.iter().filter(|r| r.period != 0).count();
        let capped = results.iter().filter(|r| r.period == 0 && r.escape_velocity >= max_iter as f64).count();
        let mut best = (0usize, 0f64);
        for (i, r) in results.iter().enumerate() {
            if r.period == 0 && r.escape_velocity < max_iter as f64 && r.escape_velocity > best.1 {
                best = (i, r.escape_velocity);
            }
        }
        let mut mism = 0;
        let mut max_err: f64 = 0.;
        let t2 = Instant::now();
        for (c, p) in coords.iter().zip(results.iter()).take(6) {
            let o = oracle(*c, max_iter);
            if p.period != o.period {
                mism += 1;
            } else if p.period == 0 {
                max_err = max_err.max((p.escape_velocity - o.escape_velocity).abs());
            }
        }
        let st = &engine.stats;
        println!(
            "S=2^{log2_s:<4} ref {:>7} ({:?}) {build_ms:>5.0} ms | {:.1} µs/px, {inside} in, {capped} capped, top ev {:.0} | newton {}/{} nucleus {}/{} | oracle 6px {:.0} ms: {mism} mismatch, max |Δev| {max_err:.2}",
            engine.reference.z64.len(),
            engine.reference.period,
            dt * 1e6 / batch as f64,
            best.1,
            st.newton_hits.load(std::sync::atomic::Ordering::Relaxed),
            st.newton_solves.load(std::sync::atomic::Ordering::Relaxed),
            st.nucleus_hits.load(std::sync::atomic::Ordering::Relaxed),
            st.nucleus_searches.load(std::sync::atomic::Ordering::Relaxed),
            t2.elapsed().as_secs_f64() * 1e3
        );
        if best.1 == 0. {
            break;
        }
        center = coords[best.0];
        log2_s -= 8;
    }
}
