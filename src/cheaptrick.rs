use ndarray::Array2;

use crate::common::*;
use crate::constant::*;
use crate::matlab::*;

/// F0 適応窓で波形を切り出し、forward_real_fft バッファに格納
fn get_windowed_waveform(
    x: &[f64],
    fs: i32,
    current_f0: f64,
    current_position: f64,
    fft_size: usize,
    randn_state: &mut RandnState,
) -> Vec<f64> {
    let half_window_length = matlab_round(1.5 * fs as f64 / current_f0) as usize;
    let window_length = half_window_length * 2 + 1;

    let origin = matlab_round(current_position * fs as f64 + 0.001);

    let mut base_index = vec![0i32; window_length];
    let mut safe_index = vec![0usize; window_length];
    let mut window = vec![0.0; window_length];

    for i in 0..window_length {
        base_index[i] = i as i32 - half_window_length as i32;
    }
    for i in 0..window_length {
        let idx = origin + base_index[i];
        safe_index[i] = idx.max(0).min(x.len() as i32 - 1) as usize;
    }

    // Hanning 窓設計
    let mut average = 0.0;
    for i in 0..window_length {
        let position = base_index[i] as f64 / 1.5 / fs as f64;
        window[i] = 0.5 * (PI * position * current_f0).cos() + 0.5;
        average += window[i] * window[i];
    }
    average = average.sqrt();
    for i in 0..window_length {
        window[i] /= average;
    }

    // F0 適応窓掛け
    let mut waveform = vec![0.0; fft_size];
    for i in 0..window_length {
        waveform[i] = x[safe_index[i]] * window[i] + randn(randn_state) * SAFE_GUARD_MINIMUM;
    }

    // DC 成分除去
    let mut tmp_weight1 = 0.0;
    let mut tmp_weight2 = 0.0;
    for i in 0..window_length {
        tmp_weight1 += waveform[i];
        tmp_weight2 += window[i];
    }
    let weighting_coefficient = tmp_weight1 / tmp_weight2;
    for i in 0..window_length {
        waveform[i] -= window[i] * weighting_coefficient;
    }

    waveform
}

/// パワースペクトル計算（DC 補正付き）
fn get_power_spectrum(
    waveform: &mut Vec<f64>,
    fs: i32,
    f0: f64,
    fft_size: usize,
) -> Vec<f64> {
    let spectrum = forward_real_fft(waveform, fft_size);

    let mut power_spectrum = vec![0.0; fft_size / 2 + 1];
    for i in 0..=fft_size / 2 {
        power_spectrum[i] = spectrum[i].re * spectrum[i].re + spectrum[i].im * spectrum[i].im;
    }

    let mut output = vec![0.0; fft_size / 2 + 1];
    dc_correction(&power_spectrum, f0, fs, fft_size, &mut output);
    output
}

/// ケプストラム平滑化と復元
fn smoothing_with_recovery(
    f0: f64,
    fs: i32,
    fft_size: usize,
    q1: f64,
    power_spectrum: &[f64],
) -> Vec<f64> {
    let half = fft_size / 2;

    // リフタ計算
    let mut smoothing_lifter = vec![0.0; half + 1];
    let mut compensation_lifter = vec![0.0; half + 1];

    smoothing_lifter[0] = 1.0;
    compensation_lifter[0] = (1.0 - 2.0 * q1) + 2.0 * q1;
    for i in 1..=half {
        let quefrency = i as f64 / fs as f64;
        smoothing_lifter[i] = (PI * f0 * quefrency).sin() / (PI * f0 * quefrency);
        compensation_lifter[i] =
            (1.0 - 2.0 * q1) + 2.0 * q1 * (2.0 * PI * quefrency * f0).cos();
    }

    // log(power_spectrum) を取って対称化
    let mut log_spectrum = vec![0.0; fft_size];
    for i in 0..=half {
        log_spectrum[i] = power_spectrum[i].ln();
    }
    for i in 1..half {
        log_spectrum[fft_size - i] = log_spectrum[i];
    }

    // FFT（ケプストラムへ）
    let spectrum = forward_real_fft(&log_spectrum, fft_size);

    // リフタ適用
    let mut filtered: Vec<num_complex::Complex64> = vec![num_complex::Complex64::new(0.0, 0.0); fft_size / 2 + 1];
    for i in 0..=half {
        filtered[i] = num_complex::Complex64::new(
            spectrum[i].re * smoothing_lifter[i] * compensation_lifter[i] / fft_size as f64,
            0.0,
        );
    }

    // IFFT
    let waveform = inverse_real_fft(&filtered, fft_size);

    // exp で復元
    let mut spectral_envelope = vec![0.0; half + 1];
    for i in 0..=half {
        spectral_envelope[i] = waveform[i].exp();
    }

    spectral_envelope
}

/// CheapTrick の1フレーム処理
fn cheaptrick_general_body(
    x: &[f64],
    fs: i32,
    current_f0: f64,
    fft_size: usize,
    current_position: f64,
    q1: f64,
    randn_state: &mut RandnState,
) -> Vec<f64> {
    // F0 適応窓掛け
    let mut waveform = get_windowed_waveform(x, fs, current_f0, current_position, fft_size, randn_state);

    // パワースペクトル計算（DC 補正付き）
    let power_spectrum = get_power_spectrum(&mut waveform, fs, current_f0, fft_size);

    // 線形平滑化
    let mut smoothed = vec![0.0; fft_size / 2 + 1];
    linear_smoothing(&power_spectrum, current_f0 * 2.0 / 3.0, fs, fft_size, &mut smoothed);

    // 微小ノイズ追加（ゼロ防止）
    for i in 0..=fft_size / 2 {
        smoothed[i] += randn(randn_state).abs() * EPS;
    }

    // ケプストラム平滑化と復元
    smoothing_with_recovery(current_f0, fs, fft_size, q1, &smoothed)
}

/// CheapTrick スペクトル包絡推定
///
/// # Arguments
/// * `x` - 入力波形
/// * `fs` - サンプリング周波数
/// * `temporal_positions` - 各フレームの時間位置 (秒)
/// * `f0` - 各フレームの基本周波数 (Hz)
/// * `option` - CheapTrickOption
///
/// # Returns
/// スペクトル包絡 [num_frames x fft_size/2+1]
pub fn cheaptrick(
    x: &[f64],
    fs: i32,
    temporal_positions: &[f64],
    f0: &[f64],
    option: &CheapTrickOption,
) -> Array2<f64> {
    let f0_length = f0.len();
    let fft_size = option.fft_size;
    let f0_floor = get_f0_floor_for_cheaptrick(fs, fft_size);

    let mut randn_state = RandnState::new();

    let spec_len = fft_size / 2 + 1;
    let mut spectrogram = Array2::zeros((f0_length, spec_len));

    for i in 0..f0_length {
        let current_f0 = if f0[i] <= f0_floor { DEFAULT_F0 } else { f0[i] };

        let envelope = cheaptrick_general_body(
            x,
            fs,
            current_f0,
            fft_size,
            temporal_positions[i],
            option.q1,
            &mut randn_state,
        );

        for j in 0..spec_len {
            spectrogram[[i, j]] = envelope[j];
        }
    }

    spectrogram
}
