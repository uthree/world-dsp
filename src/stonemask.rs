use rayon::prelude::*;

use crate::common::{blackman_window, forward_real_fft};
use crate::constant::*;
use crate::matlab::matlab_round;

/// StoneMask F0 refinement.
///
/// Refines coarse F0 estimates from DIO or Harvest using instantaneous frequency analysis at harmonics.
/// Frames are processed in parallel using rayon.
///
/// # Arguments
/// * `x` - Input waveform (mono)
/// * `fs` - Sampling frequency (Hz)
/// * `temporal_positions` - Temporal position of each frame (seconds), length `num_frames`
/// * `f0` - Coarse F0 estimate for each frame (Hz), length `num_frames`
///
/// # Returns
/// Refined F0 sequence (Hz), length `num_frames`. Unvoiced frames (input F0 <= 40Hz) are 0.0.
pub fn stonemask(x: &[f64], fs: i32, temporal_positions: &[f64], f0: &[f64]) -> Vec<f64> {
    (0..f0.len())
        .into_par_iter()
        .map(|i| get_refined_f0(x, fs, temporal_positions[i], f0[i]))
        .collect()
}

/// Single-frame F0 refinement.
///
/// Extracts with Blackman window, and from power spectrum and cross spectrum
/// computes instantaneous frequency at harmonic positions and estimates precise F0 via amplitude-weighted average.
/// Returns original `initial_f0` if correction exceeds 20%.
pub(crate) fn get_refined_f0(x: &[f64], fs: i32, current_position: f64, initial_f0: f64) -> f64 {
    if initial_f0 <= FLOOR_F0_STONEMASK || initial_f0 > fs as f64 / 12.0 {
        return 0.0;
    }

    let half_window_length = (1.5 * fs as f64 / initial_f0 + 1.0) as usize;
    let fft_size = get_suitable_fft_size(half_window_length * 2 + 1);

    let base_index_raw = matlab_round(current_position * fs as f64);
    let base_index: Vec<i64> = (-(half_window_length as i64)..=half_window_length as i64)
        .map(|i| i + base_index_raw as i64)
        .collect();

    let window_len = 2 * half_window_length + 1;
    let main_window = blackman_window(window_len);

    let mut main_spectrum_input = vec![0.0; fft_size];
    let mut diff_spectrum_input = vec![0.0; fft_size];

    let mut diff_window = vec![0.0; window_len];
    for i in 0..window_len - 1 {
        diff_window[i] = main_window[i + 1] - main_window[i];
    }

    let x_len = x.len() as i64;
    for i in 0..window_len {
        let idx = base_index[i];
        let safe = if idx < 0 || idx >= x_len {
            0.0
        } else {
            x[idx as usize]
        };
        main_spectrum_input[i] = safe * main_window[i];
        diff_spectrum_input[i] = safe * diff_window[i];
    }

    let main_spectrum = forward_real_fft(&main_spectrum_input, fft_size);
    let diff_spectrum = forward_real_fft(&diff_spectrum_input, fft_size);

    let half_fft = fft_size / 2 + 1;
    let mut power_spectrum = vec![0.0; half_fft];
    let mut numerator_i = vec![0.0; half_fft];
    for i in 0..half_fft {
        power_spectrum[i] =
            main_spectrum[i].re * main_spectrum[i].re + main_spectrum[i].im * main_spectrum[i].im;
        numerator_i[i] =
            main_spectrum[i].re * diff_spectrum[i].im - main_spectrum[i].im * diff_spectrum[i].re;
    }

    // First pass: tentative F0 with 2 harmonics, second pass: precise F0 with 6 harmonics
    let tentative_f0 = fix_f0(&power_spectrum, &numerator_i, fft_size, fs, initial_f0, 2);
    let refined = fix_f0(&power_spectrum, &numerator_i, fft_size, fs, tentative_f0, 6);

    if (refined - initial_f0).abs() / initial_f0 > 0.2 {
        initial_f0
    } else {
        refined
    }
}

/// Estimate F0 via amplitude-weighted average of instantaneous frequencies at harmonic positions.
fn fix_f0(
    power_spectrum: &[f64],
    numerator_i: &[f64],
    fft_size: usize,
    fs: i32,
    f0_initial: f64,
    number_of_harmonics: usize,
) -> f64 {
    let mut amplitude_list = vec![0.0; number_of_harmonics];
    let mut instantaneous_frequency_list = vec![0.0; number_of_harmonics];

    for i in 0..number_of_harmonics {
        let harmonic_freq = f0_initial * (i + 1) as f64;
        let index = matlab_round(harmonic_freq * fft_size as f64 / fs as f64) as usize;
        if index >= fft_size / 2 {
            break;
        }
        amplitude_list[i] = power_spectrum[index].sqrt();
        if power_spectrum[index] > EPS {
            instantaneous_frequency_list[i] =
                harmonic_freq + numerator_i[index] / power_spectrum[index] * fs as f64 / (2.0 * PI);
        }
    }

    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for i in 0..number_of_harmonics {
        numerator += amplitude_list[i] * instantaneous_frequency_list[i] / (i + 1) as f64;
        denominator += amplitude_list[i];
    }

    if denominator > EPS {
        numerator / denominator
    } else {
        f0_initial
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stonemask_sine_440hz() {
        let fs = 16000;
        let duration = 0.1;
        let f0_true = 440.0;
        let n_samples = (fs as f64 * duration) as usize;
        let x: Vec<f64> = (0..n_samples)
            .map(|i| (2.0 * PI * f0_true * i as f64 / fs as f64).sin())
            .collect();

        let temporal_positions = vec![0.05];
        let f0 = vec![f0_true];
        let refined = stonemask(&x, fs, &temporal_positions, &f0);
        assert!(
            (refined[0] - f0_true).abs() < 5.0,
            "Refined F0 {} should be close to {}",
            refined[0],
            f0_true
        );
    }
}
