/// Draws an anti-aliased line using Xiaolin Wu's line algorithm
///
/// # Arguments
/// * `image` - RGBA pixel buffer to draw into
/// * `image_width` - Width of the image in pixels
/// * `x0`, `y0` - Starting point coordinates
/// * `x1`, `y1` - Ending point coordinates
/// * `colour` - RGB color to draw (alpha is handled separately)
pub fn draw_line_u8(
    image: &mut Vec<u8>,
    image_width: usize,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    colour: [u8; 3],
) {
    let steep = (y1 - y0).abs() > (x1 - x0).abs();
    let (mut x0, mut y0, mut x1, mut y1) = (x0, y0, x1, y1);

    if steep {
        std::mem::swap(&mut x0, &mut y0);
        std::mem::swap(&mut x1, &mut y1);
    }
    if x0 > x1 {
        std::mem::swap(&mut x0, &mut x1);
        std::mem::swap(&mut y0, &mut y1);
    }

    let dx = x1 - x0;
    let dy = y1 - y0;
    let gradient = if dx == 0. { 1. } else { dy / dx };

    let xend = (x0 + 0.5).floor();
    let yend = y0 + gradient * (xend - x0);
    let xgap = 1. - (x0 + 0.5).fract();
    let xpxl1 = xend;
    let ypxl1 = yend.floor();

    let mut plot = |x: isize, y: isize, alias: f32| {
        let channels = 4;
        if x >= 0
            && x < image_width as isize
            && y >= 0
            && y < (image.len() / (image_width * channels)) as isize
        {
            let idx = (y as usize) * image_width + (x as usize);
            if image[idx * channels + 3] < 254 {
                if image[idx * channels + 3] == 253 {
                    image[idx * channels] =
                        (image[idx * channels] as f32 * (1. - alias) + 256. * alias) as u8;
                    image[idx * channels + 1] =
                        (image[idx * channels + 1] as f32 * (1. - alias) + 256. * alias) as u8;
                    image[idx * channels + 2] =
                        (image[idx * channels + 2] as f32 * (1. - alias) + 256. * alias) as u8;
                } else {
                    image[idx * channels] = (alias * 256.) as u8;
                    image[idx * channels + 1] = (alias * 256.) as u8;
                    image[idx * channels + 2] = (alias * 256.) as u8;
                    image[idx * channels + 3] = 253;
                }
            } else {
                image[idx * channels] =
                    image[idx * channels].wrapping_add((colour[0] as f32 * alias) as u8);
                image[idx * channels + 1] =
                    image[idx * channels + 1].wrapping_add((colour[1] as f32 * alias) as u8);
                image[idx * channels + 2] =
                    image[idx * channels + 2].wrapping_add((colour[2] as f32 * alias) as u8);
            }
        }
    };

    if steep {
        plot(ypxl1 as isize, xpxl1 as isize, (1. - yend.fract()) * xgap);
        plot((ypxl1 + 1.) as isize, xpxl1 as isize, yend.fract() * xgap);
    } else {
        plot(xpxl1 as isize, ypxl1 as isize, (1. - yend.fract()) * xgap);
        plot(xpxl1 as isize, (ypxl1 + 1.) as isize, yend.fract() * xgap);
    }

    let mut intery = yend + gradient;

    let xend = (x1 + 0.5).floor();
    let yend = y1 + gradient * (xend - x1);
    let xgap = (x1 + 0.5).fract();
    let xpxl2 = xend;
    let ypxl2 = yend.floor();

    if steep {
        plot(ypxl2 as isize, xpxl2 as isize, (1. - yend.fract()) * xgap);
        plot((ypxl2 + 1.) as isize, xpxl2 as isize, yend.fract() * xgap);
    } else {
        plot(xpxl2 as isize, ypxl2 as isize, (1. - yend.fract()) * xgap);
        plot(xpxl2 as isize, (ypxl2 + 1.) as isize, yend.fract() * xgap);
    }

    if steep {
        for x in (xpxl1 as isize + 1)..(xpxl2 as isize) {
            plot(intery.floor() as isize, x, 1. - intery.fract());
            plot((intery.floor() + 1.) as isize, x, intery.fract());
            intery += gradient;
        }
    } else {
        for x in (xpxl1 as isize + 1)..(xpxl2 as isize) {
            plot(x, intery.floor() as isize, 1. - intery.fract());
            plot(x, (intery.floor() + 1.) as isize, intery.fract());
            intery += gradient;
        }
    }
}
pub fn draw_line_u32(
    image: &mut [u32],
    image_width: usize,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    colour: u32,
) {
    let steep = (y1 - y0).abs() > (x1 - x0).abs();
    let (mut x0, mut y0, mut x1, mut y1) = (x0, y0, x1, y1);

    if steep {
        std::mem::swap(&mut x0, &mut y0);
        std::mem::swap(&mut x1, &mut y1);
    }
    if x0 > x1 {
        std::mem::swap(&mut x0, &mut x1);
        std::mem::swap(&mut y0, &mut y1);
    }

    let dx = x1 - x0;
    let dy = y1 - y0;
    let gradient = if dx == 0. { 1. } else { dy / dx };

    let xend = (x0 + 0.5).floor();
    let yend = y0 + gradient * (xend - x0);
    let xgap = 1. - (x0 + 0.5).fract();
    let xpxl1 = xend;
    let ypxl1 = yend.floor();

    let image_height = image.len() / image_width;

    let mut plot = |x: isize, y: isize, brightness: f32| {
        if ((x as usize) < image_width) && ((y as usize) < image_height) {
            let idx = (y as usize) * image_width + (x as usize);
            let alpha = (brightness * 255.0) as u8;
            let inv_alpha = 255 - alpha;

            // SIMD-style alpha blend: bg * (1-alpha) + fg * alpha
            let mut bg = image[idx] as u64;
            bg = (bg | (bg << 16)) & 0x0000FFFF0000FFFF;
            bg = (bg | (bg << 8)) & 0x00FF00FF00FF00FF;

            let mut fg = colour as u64;
            fg = (fg | (fg << 16)) & 0x0000FFFF0000FFFF;
            fg = (fg | (fg << 8)) & 0x00FF00FF00FF00FF;

            let mut blended = bg * inv_alpha as u64 + fg * alpha as u64;
            blended = (blended >> 8) & 0x00FF00FF00FF00FF;
            blended = (blended | (blended >> 8)) & 0x0000FFFF0000FFFF;
            blended = blended | (blended >> 16) | 0xFF000000;

            image[idx] = blended as u32;
        }
    };

    if steep {
        plot(ypxl1 as isize, xpxl1 as isize, (1. - yend.fract()) * xgap);
        plot((ypxl1 + 1.) as isize, xpxl1 as isize, yend.fract() * xgap);
    } else {
        plot(xpxl1 as isize, ypxl1 as isize, (1. - yend.fract()) * xgap);
        plot(xpxl1 as isize, (ypxl1 + 1.) as isize, yend.fract() * xgap);
    }

    let mut intery = yend + gradient;

    let xend = (x1 + 0.5).floor();
    let yend = y1 + gradient * (xend - x1);
    let xgap = (x1 + 0.5).fract();
    let xpxl2 = xend;
    let ypxl2 = yend.floor();

    if steep {
        plot(ypxl2 as isize, xpxl2 as isize, (1. - yend.fract()) * xgap);
        plot((ypxl2 + 1.) as isize, xpxl2 as isize, yend.fract() * xgap);
    } else {
        plot(xpxl2 as isize, ypxl2 as isize, (1. - yend.fract()) * xgap);
        plot(xpxl2 as isize, (ypxl2 + 1.) as isize, yend.fract() * xgap);
    }

    if steep {
        for x in (xpxl1 as isize + 1)..(xpxl2 as isize) {
            plot(intery.floor() as isize, x, 1. - intery.fract());
            plot((intery.floor() + 1.) as isize, x, intery.fract());
            intery += gradient;
        }
    } else {
        for x in (xpxl1 as isize + 1)..(xpxl2 as isize) {
            plot(x, intery.floor() as isize, 1. - intery.fract());
            plot(x, (intery.floor() + 1.) as isize, intery.fract());
            intery += gradient;
        }
    }
}
