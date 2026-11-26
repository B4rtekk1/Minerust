//! Precomputed water wave textures.
//!
//! The realistic water shader used to evaluate dozens of noise functions per
//! pixel every frame. Instead, the wave shapes are baked once at startup into
//! a small array of seamlessly tiling textures, and the shader animates them
//! by scrolling and rotating several samples of each layer at different
//! scales. Mipmaps take care of distant water for free.
//!
//! Every layer stores `(value, d/du, d/dv, extra)`, where `u, v` are tile
//! coordinates in `0..1`. Storing the analytic derivatives lets the shader
//! build normals from a single fetch, and box-filtered mipmaps of the
//! derivatives average normals correctly in the distance.
//!
//! The layout is the extension point for more advanced water: a compute pass
//! (for example an FFT ocean) can later write the same layers every frame
//! without touching the water shader's sampling code.

use std::f32::consts::TAU;

/// Side of every layer in texels.
pub const WAVE_TEXTURE_SIZE: u32 = 256;

/// Layer indices in the texture array.
pub const LAYER_SWELL: u32 = 0;
pub const LAYER_RIPPLES: u32 = 1;
pub const LAYER_MACRO: u32 = 2;
pub const WAVE_LAYER_COUNT: u32 = 3;

/// One mip level of the whole array: `size × size × layers` RGBA texels.
pub struct WaveMip {
    pub size: u32,
    /// Texels as `f32` RGBA, layer-major.
    pub texels: Vec<[f32; 4]>,
}

/// Bakes all layers and their full mip chains (level 0 first).
pub fn bake_wave_textures(seed: u32) -> Vec<WaveMip> {
    let size = WAVE_TEXTURE_SIZE as usize;
    let mut texels = Vec::with_capacity(size * size * WAVE_LAYER_COUNT as usize);
    texels.extend(swell_layer(size, seed));
    texels.extend(ripple_layer(size, seed.wrapping_add(1)));
    texels.extend(macro_layer(size, seed.wrapping_add(2)));
    build_mips(WaveMip {
        size: WAVE_TEXTURE_SIZE,
        texels,
    })
}

/// Long wind-driven waves: a sum of sines on integer wave vectors (so the
/// tile repeats seamlessly) whose directions cluster around +U and whose
/// amplitudes follow a falling spectrum. `extra` holds a crest mask.
fn swell_layer(size: usize, seed: u32) -> Vec<[f32; 4]> {
    let mut rng = Lcg(seed as u64 ^ 0x5eed_0f_5ea);
    let mut waves = Vec::new();
    while waves.len() < 48 {
        let kx = 1 + (rng.next() % 14) as i32;
        let ky = (rng.next() % 17) as i32 - 8;
        let k = ((kx * kx + ky * ky) as f32).sqrt();
        // Mostly along +U (the wind), with a spread of about ±60°.
        let alignment = kx as f32 / k;
        if alignment < 0.45 {
            continue;
        }
        let amplitude = alignment * alignment / k.powf(1.5);
        waves.push((kx as f32, ky as f32, amplitude, rng.unit() * TAU));
    }
    let mut out = vec![[0.0f32; 4]; size * size];
    for (i, texel) in out.iter_mut().enumerate() {
        let (u, v) = ((i % size) as f32 / size as f32, (i / size) as f32 / size as f32);
        for &(kx, ky, a, phase) in &waves {
            let theta = TAU * (kx * u + ky * v) + phase;
            let (s, c) = theta.sin_cos();
            // Sharper crests, wider troughs.
            texel[0] += a * (s + 0.18 * (s * s - 0.5));
            let slope = a * (1.0 + 0.36 * s) * c * TAU;
            texel[1] += slope * kx;
            texel[2] += slope * ky;
        }
    }
    normalize_layer(&mut out);
    for texel in &mut out {
        texel[3] = ((texel[0] - 0.45) / 0.55).clamp(0.0, 1.0);
    }
    out
}

/// Small isotropic ripples: tileable fractal gradient noise. `extra` holds
/// a second, decorrelated noise used to break up foam.
fn ripple_layer(size: usize, seed: u32) -> Vec<[f32; 4]> {
    let mut out = vec![[0.0f32; 4]; size * size];
    for (i, texel) in out.iter_mut().enumerate() {
        let (u, v) = ((i % size) as f32 / size as f32, (i / size) as f32 / size as f32);
        let mut amplitude = 1.0;
        for (octave, period) in [8u32, 16, 32].into_iter().enumerate() {
            let (n, du, dv) = periodic_noise(u, v, period, seed.wrapping_add(octave as u32));
            texel[0] += n * amplitude;
            texel[1] += du * amplitude;
            texel[2] += dv * amplitude;
            amplitude *= 0.5;
        }
        texel[3] = periodic_noise(u, v, 24, seed.wrapping_add(17)).0;
    }
    normalize_layer(&mut out);
    out
}

/// Slow large-scale variation: `value` = wave-group strength, `d/du`,
/// `d/dv` slots hold two independent warp offsets, `extra` foam noise.
fn macro_layer(size: usize, seed: u32) -> Vec<[f32; 4]> {
    let mut out = vec![[0.0f32; 4]; size * size];
    for (i, texel) in out.iter_mut().enumerate() {
        let (u, v) = ((i % size) as f32 / size as f32, (i / size) as f32 / size as f32);
        texel[0] = periodic_noise(u, v, 3, seed).0 + 0.5 * periodic_noise(u, v, 7, seed + 1).0;
        texel[1] = periodic_noise(u, v, 4, seed + 2).0;
        texel[2] = periodic_noise(u, v, 4, seed + 3).0;
        texel[3] = periodic_noise(u, v, 32, seed + 4).0 + 0.5 * periodic_noise(u, v, 64, seed + 5).0;
    }
    for channel in 0..4 {
        let max = out.iter().map(|t| t[channel].abs()).fold(1e-6, f32::max);
        for texel in &mut out {
            texel[channel] /= max;
        }
    }
    out
}

/// Scales a layer so its value channel spans about `-1..1`.
fn normalize_layer(layer: &mut [[f32; 4]]) {
    let max = layer.iter().map(|t| t[0].abs()).fold(1e-6, f32::max);
    for texel in layer {
        texel[0] /= max;
        texel[1] /= max;
        texel[2] /= max;
    }
}

/// Gradient noise that repeats every `period` cells over the unit tile.
/// Returns the value and its analytic derivatives with respect to `u`, `v`.
fn periodic_noise(u: f32, v: f32, period: u32, seed: u32) -> (f32, f32, f32) {
    let p = period as f32;
    let (x, y) = (u * p, v * p);
    let (ix, iy) = (x.floor(), y.floor());
    let (fx, fy) = (x - ix, y - iy);
    let gradient = |cx: f32, cy: f32| -> (f32, f32) {
        let wx = (cx as i64).rem_euclid(period as i64) as u32;
        let wy = (cy as i64).rem_euclid(period as i64) as u32;
        let h = hash(wx, wy, seed);
        let angle = (h as f32 / u32::MAX as f32) * TAU;
        (angle.cos(), angle.sin())
    };
    let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let dfade = |t: f32| 30.0 * t * t * (t * (t - 2.0) + 1.0);
    let corner = |cx: f32, cy: f32, dx: f32, dy: f32| {
        let (gx, gy) = gradient(cx, cy);
        (gx * dx + gy * dy, gx, gy)
    };
    let (a, ax, ay) = corner(ix, iy, fx, fy);
    let (b, bx, by) = corner(ix + 1.0, iy, fx - 1.0, fy);
    let (c, cx, cy) = corner(ix, iy + 1.0, fx, fy - 1.0);
    let (d, dx, dy) = corner(ix + 1.0, iy + 1.0, fx - 1.0, fy - 1.0);
    let (sx, sy) = (fade(fx), fade(fy));
    let (dsx, dsy) = (dfade(fx), dfade(fy));
    let value = a + (b - a) * sx + (c - a) * sy + (a - b - c + d) * sx * sy;
    let ddx = ax + (bx - ax) * sx + (cx - ax) * sy + (ax - bx - cx + dx) * sx * sy
        + ((b - a) + (a - b - c + d) * sy) * dsx;
    let ddy = ay + (by - ay) * sx + (cy - ay) * sy + (ay - by - cy + dy) * sx * sy
        + ((c - a) + (a - b - c + d) * sx) * dsy;
    // Derivatives are per cell; convert to per unit tile.
    (value, ddx * p, ddy * p)
}

fn hash(x: u32, y: u32, seed: u32) -> u32 {
    let mut h = seed ^ x.wrapping_mul(0x8da6_b343) ^ y.wrapping_mul(0xd816_3841);
    h = (h ^ (h >> 15)).wrapping_mul(0x2c1b_3c6d);
    h = (h ^ (h >> 12)).wrapping_mul(0x297a_2d39);
    h ^ (h >> 15)
}

/// 2×2 box-filtered mip chain (wrapping, since every layer tiles).
fn build_mips(base: WaveMip) -> Vec<WaveMip> {
    let layers = WAVE_LAYER_COUNT as usize;
    let mut mips = vec![base];
    while mips.last().unwrap().size > 1 {
        let prev = mips.last().unwrap();
        let (src, dst) = (prev.size as usize, prev.size as usize / 2);
        let mut texels = vec![[0.0f32; 4]; dst * dst * layers];
        for layer in 0..layers {
            for y in 0..dst {
                for x in 0..dst {
                    let mut sum = [0.0f32; 4];
                    for (ox, oy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let t = prev.texels[layer * src * src + (2 * y + oy) * src + 2 * x + ox];
                        for c in 0..4 {
                            sum[c] += t[c] * 0.25;
                        }
                    }
                    texels[layer * dst * dst + y * dst + x] = sum;
                }
            }
        }
        mips.push(WaveMip {
            size: dst as u32,
            texels,
        });
    }
    mips
}

/// Converts RGBA `f32` texels to the `Rgba16Float` byte layout.
pub fn to_rgba16f_bytes(texels: &[[f32; 4]]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(texels.len() * 8);
    for texel in texels {
        for &value in texel {
            bytes.extend_from_slice(&f32_to_f16(value).to_le_bytes());
        }
    }
    bytes
}

/// IEEE 754 half-precision conversion (round to nearest, no NaN payloads).
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    if exponent >= 31 {
        return sign | 0x7c00; // overflow → infinity
    }
    if exponent <= 0 {
        if exponent < -10 {
            return sign; // underflow → zero
        }
        let m = (mantissa | 0x80_0000) >> (1 - exponent);
        return sign | ((m + 0x1000) >> 13) as u16;
    }
    let half = sign | ((exponent as u16) << 10) | (mantissa >> 13) as u16;
    // Round to nearest; a carry correctly bumps the exponent.
    half + u16::from(mantissa & 0x1000 != 0)
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn unit(&mut self) -> f32 {
        self.next() as f32 / (1u64 << 31) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layers_tile_seamlessly_and_stay_normalised() {
        let mips = bake_wave_textures(7);
        let base = &mips[0];
        let size = base.size as usize;
        for layer in 0..WAVE_LAYER_COUNT as usize {
            let at = |x: usize, y: usize| base.texels[layer * size * size + y * size + x];
            for i in 0..size {
                // The wrap seam differs no more than an interior step does.
                let seam = (at(size - 1, i)[0] - at(0, i)[0]).abs();
                let step = (at(1, i)[0] - at(0, i)[0]).abs();
                assert!(seam < step + 0.2, "layer {layer} row {i}: seam {seam} vs step {step}");
            }
            assert!(base.texels[layer * size * size..(layer + 1) * size * size]
                .iter()
                .all(|t| t.iter().all(|c| c.is_finite()) && t[0].abs() <= 1.0001));
        }
        assert_eq!(mips.last().unwrap().size, 1);
        assert_eq!(mips.len(), WAVE_TEXTURE_SIZE.trailing_zeros() as usize + 1);
    }

    #[test]
    fn analytic_derivatives_match_finite_differences() {
        let (h, du, dv) = periodic_noise(0.3, 0.6, 8, 3);
        let e = 1e-4;
        let fdu = (periodic_noise(0.3 + e, 0.6, 8, 3).0 - h) / e;
        let fdv = (periodic_noise(0.3, 0.6 + e, 8, 3).0 - h) / e;
        assert!((du - fdu).abs() < 0.05 * du.abs().max(1.0), "{du} vs {fdu}");
        assert!((dv - fdv).abs() < 0.05 * dv.abs().max(1.0), "{dv} vs {fdv}");
    }

    #[test]
    fn half_float_conversion() {
        assert_eq!(f32_to_f16(0.0), 0);
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
        assert_eq!(f32_to_f16(0.5), 0x3800);
        assert_eq!(f32_to_f16(65504.0), 0x7bff);
    }
}
