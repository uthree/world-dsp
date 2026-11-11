use rayon::prelude::*;

use crate::common::forward_real_fft;
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

/// Single-frame F0 refinement (`GetRefinedF0` in the C++ WORLD).
///
/// Extracts the frame with a Blackman window, computes the instantaneous
/// frequency at the harmonics from the spectra of the waveform windowed by the
/// window and by its time derivative, and estimates F0 as their
/// amplitude-weighted average.
/// Returns the original `initial_f0` if the correction exceeds 20%.
pub(crate) fn get_refined_f0(x: &[f64], fs: i32, current_position: f64, initial_f0: f64) -> f64 {
    if initial_f0 <= FLOOR_F0_STONEMASK || initial_f0 > fs as f64 / 12.0 {
        return 0.0;
    }

    let half_window_length = (1.5 * fs as f64 / initial_f0 + 1.0) as usize;
    let base_time_length = half_window_length * 2 + 1;
    let window_length_in_time = base_time_length as f64 / fs as f64;
    let fft_size = 2_usize.pow(2 + ((base_time_length as f64).ln() / LOG2) as u32);

    // GetBaseIndex
    let index_raw: Vec<i64> = (0..base_time_length)
        .map(|i| {
            let base_time = (i as f64 - half_window_length as f64) / fs as f64;
            matlab_round((current_position + base_time) * fs as f64) as i64
        })
        .collect();

    // GetMainWindow
    let main_window: Vec<f64> = index_raw
        .iter()
        .map(|&index| {
            let tmp = (index as f64 - 1.0) / fs as f64 - current_position;
            0.42 + 0.5 * (2.0 * PI * tmp / window_length_in_time).cos()
                + 0.08 * (4.0 * PI * tmp / window_length_in_time).cos()
        })
        .collect();

    // GetDiffWindow
    let mut diff_window = vec![0.0; base_time_length];
    diff_window[0] = -main_window[1] / 2.0;
    for i in 1..base_time_length - 1 {
        diff_window[i] = -(main_window[i + 1] - main_window[i - 1]) / 2.0;
    }
    diff_window[base_time_length - 1] = main_window[base_time_length - 2] / 2.0;

    // GetSpectra (samples outside the signal are replaced by the edge samples)
    let last = x.len() as i64 - 1;
    let mut main_input = vec![0.0; base_time_length];
    let mut diff_input = vec![0.0; base_time_length];
    for i in 0..base_time_length {
        let sample = x[(index_raw[i] - 1).clamp(0, last) as usize];
        main_input[i] = sample * main_window[i];
        diff_input[i] = sample * diff_window[i];
    }
    let main_spectrum = forward_real_fft(&main_input, fft_size);
    let diff_spectrum = forward_real_fft(&diff_input, fft_size);

    let half_fft = fft_size / 2 + 1;
    let mut power_spectrum = vec![0.0; half_fft];
    let mut numerator_i = vec![0.0; half_fft];
    for j in 0..half_fft {
        numerator_i[j] =
            main_spectrum[j].re * diff_spectrum[j].im - main_spectrum[j].im * diff_spectrum[j].re;
        power_spectrum[j] =
            main_spectrum[j].re * main_spectrum[j].re + main_spectrum[j].im * main_spectrum[j].im;
    }

    // GetTentativeF0
    let tentative_f0 = fix_f0(&power_spectrum, &numerator_i, fft_size, fs, initial_f0, 2);
    let mean_f0 = if tentative_f0 <= 0.0 || tentative_f0 > initial_f0 * 2.0 {
        0.0
    } else {
        fix_f0(&power_spectrum, &numerator_i, fft_size, fs, tentative_f0, 6)
    };

    // If the amount of correction is too large (20 %), the initial F0 is used.
    if (mean_f0 - initial_f0).abs() > initial_f0 * 0.2 {
        initial_f0
    } else {
        mean_f0
    }
}

/// Fix F0 using the instantaneous frequency at the harmonics (`FixF0` in the C++ WORLD).
fn fix_f0(
    power_spectrum: &[f64],
    numerator_i: &[f64],
    fft_size: usize,
    fs: i32,
    initial_f0: f64,
    number_of_harmonics: usize,
) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for i in 0..number_of_harmonics {
        let index = (matlab_round(initial_f0 * fft_size as f64 / fs as f64 * (i + 1) as f64).max(0)
            as usize)
            .min(fft_size / 2);
        let instantaneous_frequency = if power_spectrum[index] == 0.0 {
            0.0
        } else {
            index as f64 * fs as f64 / fft_size as f64
                + numerator_i[index] / power_spectrum[index] * fs as f64 / 2.0 / PI
        };
        let amplitude = power_spectrum[index].sqrt();
        numerator += amplitude * instantaneous_frequency;
        denominator += amplitude * (i + 1) as f64;
    }
    numerator / (denominator + SAFE_GUARD_MINIMUM)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(fs: i32, f0: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * f0 * i as f64 / fs as f64).sin())
            .collect()
    }

    #[test]
    fn test_stonemask_sine_440hz() {
        let fs = 16000;
        let f0_true = 440.0;
        let x = sine(fs, f0_true, 1600);

        let refined = stonemask(&x, fs, &[0.05], &[f0_true]);
        assert!(
            (refined[0] - f0_true).abs() < 0.5,
            "Refined F0 {} should be close to {}",
            refined[0],
            f0_true
        );
    }

    #[test]
    fn exact_f0_is_kept() {
        let fs = 16000;
        for f0_true in [100.0, 210.0, 440.0, 800.0] {
            let x = sine(fs, f0_true, 8000);
            let refined = get_refined_f0(&x, fs, 0.25, f0_true);
            assert!(
                (refined - f0_true).abs() < 0.01 * f0_true,
                "{f0_true} Hz moved to {refined} Hz"
            );
        }
    }

    #[test]
    fn coarse_f0_is_corrected() {
        let fs = 16000;
        for f0_true in [120.0, 210.0, 440.0] {
            let x = sine(fs, f0_true, 8000);
            for factor in [0.95, 1.05] {
                let initial = f0_true * factor;
                let refined = get_refined_f0(&x, fs, 0.25, initial);
                assert!(
                    (refined - f0_true).abs() < 0.2 * (initial - f0_true).abs(),
                    "{initial} Hz refined to {refined} Hz (true {f0_true} Hz)"
                );
            }
        }
    }

    #[test]
    fn out_of_range_f0_is_unvoiced() {
        let fs = 16000;
        let x = sine(fs, 200.0, 8000);
        assert_eq!(get_refined_f0(&x, fs, 0.25, 0.0), 0.0);
        assert_eq!(get_refined_f0(&x, fs, 0.25, FLOOR_F0_STONEMASK), 0.0);
        assert_eq!(get_refined_f0(&x, fs, 0.25, fs as f64 / 12.0 + 1.0), 0.0);
    }

    #[test]
    fn frames_at_signal_edges_are_handled() {
        let fs = 16000;
        let x = sine(fs, 200.0, 4000);
        let refined = stonemask(&x, fs, &[0.0, 0.25], &[200.0, 200.0]);
        assert!(refined.iter().all(|v| v.is_finite()));
    }
}
