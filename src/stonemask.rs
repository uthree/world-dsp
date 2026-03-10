use crate::common::{blackman_window, forward_real_fft};
use crate::constant::*;
use crate::matlab::matlab_round;

/// StoneMask F0 リファインメント
///
/// DIO や Harvest で得られた粗い F0 推定値を、スペクトル解析により精密化する。
pub fn stonemask(x: &[f64], fs: i32, temporal_positions: &[f64], f0: &[f64]) -> Vec<f64> {
    let f0_length = f0.len();
    let mut refined_f0 = vec![0.0; f0_length];
    for i in 0..f0_length {
        refined_f0[i] = get_refined_f0(x, fs, temporal_positions[i], f0[i]);
    }
    refined_f0
}

/// 単一フレームの F0 リファインメント（Harvest からも呼ばれる）
pub(crate) fn get_refined_f0(x: &[f64], fs: i32, current_position: f64, initial_f0: f64) -> f64 {
    if initial_f0 <= FLOOR_F0_STONEMASK || initial_f0 > fs as f64 / 12.0 {
        return 0.0;
    }

    let half_window_length = (1.5 * fs as f64 / initial_f0) as usize + 1;
    let fft_size = get_suitable_fft_size((half_window_length * 2 + 1) as usize);

    let base_index_raw = matlab_round(current_position * fs as f64);
    let base_index: Vec<i64> = (-(half_window_length as i64)..=half_window_length as i64)
        .map(|i| i + base_index_raw as i64)
        .collect();

    let window_len = 2 * half_window_length + 1;
    let main_window = blackman_window(window_len);

    // windowed signal
    let mut main_spectrum_input = vec![0.0; fft_size];
    let mut diff_spectrum_input = vec![0.0; fft_size];

    // diff_window: numerical derivative of main_window
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

    // power_spectrum と numerator_i
    let half_fft = fft_size / 2 + 1;
    let mut power_spectrum = vec![0.0; half_fft];
    let mut numerator_i = vec![0.0; half_fft];
    for i in 0..half_fft {
        power_spectrum[i] = main_spectrum[i].re * main_spectrum[i].re
            + main_spectrum[i].im * main_spectrum[i].im;
        numerator_i[i] = main_spectrum[i].re * diff_spectrum[i].im
            - main_spectrum[i].im * diff_spectrum[i].re;
    }

    // 1回目: 2倍音で暫定F0
    let tentative_f0 =
        fix_f0(&power_spectrum, &numerator_i, fft_size, fs, initial_f0, 2);
    // 2回目: 6倍音で精密F0
    let refined = fix_f0(
        &power_spectrum,
        &numerator_i,
        fft_size,
        fs,
        tentative_f0,
        6,
    );

    // 補正量が 20% を超えたら元の f0 を維持
    if (refined - initial_f0).abs() / initial_f0 > 0.2 {
        initial_f0
    } else {
        refined
    }
}

/// 高調波位置での瞬時周波数の振幅加重平均で F0 を推定
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
            instantaneous_frequency_list[i] = harmonic_freq
                + numerator_i[index] / power_spectrum[index] * fs as f64
                    / (2.0 * PI);
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
