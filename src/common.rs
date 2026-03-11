use num_complex::Complex64;
use rustfft::FftPlanner;

use crate::constant::*;
use crate::matlab::interp1q;

/// Real-to-complex forward FFT.
///
/// Transforms the real signal `x` using an `fft_size`-point FFT.
/// Returns a complex vector of length `fft_size` (all bins).
pub fn forward_real_fft(x: &[f64], fft_size: usize) -> Vec<Complex64> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let mut buf: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); fft_size];
    for (i, &val) in x.iter().enumerate().take(fft_size) {
        buf[i] = Complex64::new(val, 0.0);
    }
    fft.process(&mut buf);
    buf
}

/// Complex spectrum to real signal inverse FFT.
///
/// `spectrum` contains the positive frequency components of length `fft_size/2+1`.
/// Using conjugate symmetry, recovers the real signal of length `fft_size`.
/// 1/N normalized.
pub fn inverse_real_fft(spectrum: &[Complex64], fft_size: usize) -> Vec<f64> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(fft_size);

    let mut buf = vec![Complex64::new(0.0, 0.0); fft_size];
    for i in 0..=fft_size / 2 {
        buf[i] = spectrum[i];
    }
    for i in 1..fft_size / 2 {
        buf[fft_size - i] = spectrum[i].conj();
    }
    fft.process(&mut buf);
    buf.iter().map(|c| c.re / fft_size as f64).collect()
}

/// Complex forward FFT.
///
/// Transforms a complex input of length `fft_size`.
pub fn forward_fft(x: &[Complex64], fft_size: usize) -> Vec<Complex64> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let mut buf = vec![Complex64::new(0.0, 0.0); fft_size];
    for (i, &val) in x.iter().enumerate().take(fft_size) {
        buf[i] = val;
    }
    fft.process(&mut buf);
    buf
}

/// Complex inverse FFT. 1/N normalized.
pub fn inverse_fft(x: &[Complex64], fft_size: usize) -> Vec<Complex64> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(fft_size);
    let mut buf = vec![Complex64::new(0.0, 0.0); fft_size];
    for (i, &val) in x.iter().enumerate().take(fft_size) {
        buf[i] = val;
    }
    fft.process(&mut buf);
    for c in buf.iter_mut() {
        *c /= fft_size as f64;
    }
    buf
}

/// Generate a Nuttall window.
///
/// Returns a symmetric Nuttall window of length `n`. Has strong sidelobe suppression.
pub fn nuttall_window(n: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let tmp = i as f64 / (n as f64 - 1.0);
        y[i] = 0.355768 - 0.487396 * (2.0 * PI * tmp).cos() + 0.144232 * (4.0 * PI * tmp).cos()
            - 0.012604 * (6.0 * PI * tmp).cos();
    }
    y
}

/// Generate a Hanning window.
///
/// Returns a symmetric Hanning window of length `n`.
pub fn hanning_window(n: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let tmp = i as f64 / (n as f64 - 1.0);
        y[i] = 0.5 - 0.5 * (2.0 * PI * tmp).cos();
    }
    y
}

/// Generate a Blackman window.
///
/// Returns a symmetric Blackman window of length `n`.
pub fn blackman_window(n: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let tmp = i as f64 / (n as f64 - 1.0);
        y[i] = 0.42 - 0.5 * (2.0 * PI * tmp).cos() + 0.08 * (4.0 * PI * tmp).cos();
    }
    y
}

/// DC component correction.
///
/// Corrects the low-frequency DC component of the power spectrum `input[0..=fft_size/2]`
/// and writes the result to `output[0..=fft_size/2]`.
pub fn dc_correction(input: &[f64], f0: f64, fs: i32, fft_size: usize, output: &mut [f64]) {
    let upper_limit = 2 + (f0 * fft_size as f64 / fs as f64) as usize;

    let low_frequency_axis: Vec<f64> = (0..upper_limit)
        .map(|i| i as f64 * fs as f64 / fft_size as f64)
        .collect();

    let upper_limit_replica = upper_limit - 1;
    let mut low_frequency_replica = vec![0.0; upper_limit_replica];

    interp1q(
        f0 - low_frequency_axis[0],
        -(fs as f64) / fft_size as f64,
        input,
        &low_frequency_axis[..upper_limit_replica],
        &mut low_frequency_replica,
    );

    for i in 0..upper_limit_replica {
        output[i] = input[i] + low_frequency_replica[i];
    }
    for i in upper_limit_replica..=fft_size / 2 {
        if i < output.len() {
            output[i] = input[i];
        }
    }
}

/// Parameter setup for linear smoothing (internal function).
fn set_parameters_for_linear_smoothing(
    boundary: usize,
    fft_size: usize,
    fs: i32,
    width: f64,
    power_spectrum: &[f64],
    mirroring_spectrum: &mut [f64],
    mirroring_segment: &mut [f64],
    frequency_axis: &mut [f64],
) {
    for i in 0..boundary {
        mirroring_spectrum[i] = power_spectrum[boundary - i];
    }
    for i in boundary..fft_size / 2 + boundary {
        mirroring_spectrum[i] = power_spectrum[i - boundary];
    }
    for i in fft_size / 2 + boundary..=fft_size / 2 + boundary * 2 {
        mirroring_spectrum[i] = power_spectrum[fft_size / 2 - (i - (fft_size / 2 + boundary))];
    }

    mirroring_segment[0] = mirroring_spectrum[0] * fs as f64 / fft_size as f64;
    for i in 1..fft_size / 2 + boundary * 2 + 1 {
        mirroring_segment[i] =
            mirroring_spectrum[i] * fs as f64 / fft_size as f64 + mirroring_segment[i - 1];
    }

    for i in 0..=fft_size / 2 {
        frequency_axis[i] = i as f64 / fft_size as f64 * fs as f64 - width / 2.0;
    }
}

/// Linear smoothing (along frequency axis).
///
/// Smooths the power spectrum `input[0..=fft_size/2]` with width `width` (Hz)
/// and writes the result to `output[0..=fft_size/2]`.
pub fn linear_smoothing(input: &[f64], width: f64, fs: i32, fft_size: usize, output: &mut [f64]) {
    let boundary = (width * fft_size as f64 / fs as f64) as usize + 1;

    let mirror_len = fft_size / 2 + boundary * 2 + 1;
    let mut mirroring_spectrum = vec![0.0; mirror_len];
    let mut mirroring_segment = vec![0.0; mirror_len];
    let mut frequency_axis = vec![0.0; fft_size / 2 + 1];

    set_parameters_for_linear_smoothing(
        boundary,
        fft_size,
        fs,
        width,
        input,
        &mut mirroring_spectrum,
        &mut mirroring_segment,
        &mut frequency_axis,
    );

    let mut low_levels = vec![0.0; fft_size / 2 + 1];
    let mut high_levels = vec![0.0; fft_size / 2 + 1];
    let origin = -(boundary as f64 - 0.5) * fs as f64 / fft_size as f64;
    let discrete_freq_interval = fs as f64 / fft_size as f64;

    interp1q(
        origin,
        discrete_freq_interval,
        &mirroring_segment,
        &frequency_axis,
        &mut low_levels,
    );

    for i in 0..=fft_size / 2 {
        frequency_axis[i] += width;
    }

    interp1q(
        origin,
        discrete_freq_interval,
        &mirroring_segment,
        &frequency_axis,
        &mut high_levels,
    );

    for i in 0..=fft_size / 2 {
        output[i] = (high_levels[i] - low_levels[i]) / width;
    }
}

/// Compute minimum phase spectrum.
///
/// Computes the minimum phase spectrum from the log power spectrum
/// `log_spectrum[0..=fft_size/2]` using the cepstral method (causal window).
///
/// # Returns
/// Complex spectrum of length `fft_size/2+1`
pub fn get_minimum_phase_spectrum(log_spectrum: &[f64], fft_size: usize) -> Vec<Complex64> {
    let mut planner = FftPlanner::new();

    let mut mirrored = vec![0.0; fft_size];
    for i in 0..=fft_size / 2 {
        mirrored[i] = log_spectrum[i];
    }
    for i in fft_size / 2 + 1..fft_size {
        mirrored[i] = mirrored[fft_size - i];
    }

    let fft_forward = planner.plan_fft_forward(fft_size);
    let mut cepstrum: Vec<Complex64> = mirrored.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    fft_forward.process(&mut cepstrum);

    // Causal window: double positive quefrencies, zero negative quefrencies
    for i in 1..fft_size / 2 {
        cepstrum[i] = Complex64::new(cepstrum[i].re * 2.0, cepstrum[i].im * 2.0);
    }
    for i in fft_size / 2 + 1..fft_size {
        cepstrum[i] = Complex64::new(0.0, 0.0);
    }

    let fft_inverse = planner.plan_fft_inverse(fft_size);
    fft_inverse.process(&mut cepstrum);

    let mut result = vec![Complex64::new(0.0, 0.0); fft_size / 2 + 1];
    for i in 0..=fft_size / 2 {
        let tmp = (cepstrum[i].re / fft_size as f64).exp();
        let phase = -cepstrum[i].im / fft_size as f64;
        result[i] = Complex64::new(tmp * phase.cos(), tmp * phase.sin());
    }

    result
}

/// Returns a safe aperiodicity value (clamped to 0.001--0.999999999999).
#[inline]
pub fn get_safe_aperiodicity(x: f64) -> f64 {
    x.clamp(0.001, 0.999999999999)
}
