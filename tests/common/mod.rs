//! Helpers shared by the integration tests.
#![allow(dead_code)]

use std::f64::consts::PI;

/// A voice-like periodic signal: a fundamental with decaying harmonics.
pub fn harmonic_signal(fs: i32, f0: f64, duration: f64) -> Vec<f64> {
    let n = (fs as f64 * duration) as usize;
    let nyquist = fs as f64 / 2.0;
    (0..n)
        .map(|i| {
            let t = i as f64 / fs as f64;
            let mut v = 0.0;
            let mut h = 1;
            while f0 * h as f64 <= nyquist * 0.9 && h <= 10 {
                v += (2.0 * PI * f0 * h as f64 * t).sin() / h as f64;
                h += 1;
            }
            0.3 * v
        })
        .collect()
}

/// Deterministic white noise in [-amplitude, amplitude] (xorshift).
pub fn white_noise(n: usize, amplitude: f64, seed: u64) -> Vec<f64> {
    let mut state = seed.max(1);
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            amplitude * (2.0 * u - 1.0)
        })
        .collect()
}

pub fn rms(x: &[f64]) -> f64 {
    (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt()
}

/// Median of the voiced (positive) values in the middle half of `f0`.
pub fn median_voiced_center(f0: &[f64]) -> Option<f64> {
    let n = f0.len();
    let mut voiced: Vec<f64> = f0[n / 4..n * 3 / 4]
        .iter()
        .copied()
        .filter(|&v| v > 0.0)
        .collect();
    if voiced.is_empty() {
        return None;
    }
    voiced.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(voiced[voiced.len() / 2])
}

/// Fraction of frames in the middle half of `f0` that are voiced.
pub fn voiced_ratio_center(f0: &[f64]) -> f64 {
    let n = f0.len();
    let center = &f0[n / 4..n * 3 / 4];
    center.iter().filter(|&&v| v > 0.0).count() as f64 / center.len() as f64
}

pub fn assert_relative_close(actual: f64, expected: f64, rel_tol: f64, what: &str) {
    assert!(
        ((actual - expected) / expected).abs() <= rel_tol,
        "{what}: {actual} is not within {:.1}% of {expected}",
        rel_tol * 100.0
    );
}
