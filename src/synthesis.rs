use ndarray::{Array1, Array2};

use crate::common::*;
use crate::constant::*;
use crate::matlab::*;

/// 時間パラメータの準備
fn get_temporal_parameters_for_time_base(
    f0: &[f64],
    fs: i32,
    y_length: usize,
    frame_period: f64,
    lowest_f0: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let f0_length = f0.len();
    let time_axis: Vec<f64> = (0..y_length).map(|i| i as f64 / fs as f64).collect();

    let mut coarse_time_axis = vec![0.0; f0_length + 1];
    let mut coarse_f0 = vec![0.0; f0_length + 1];
    let mut coarse_vuv = vec![0.0; f0_length + 1];

    for i in 0..f0_length {
        coarse_time_axis[i] = i as f64 * frame_period;
        coarse_f0[i] = if f0[i] < lowest_f0 { 0.0 } else { f0[i] };
        coarse_vuv[i] = if coarse_f0[i] == 0.0 { 0.0 } else { 1.0 };
    }
    coarse_time_axis[f0_length] = f0_length as f64 * frame_period;
    coarse_f0[f0_length] = coarse_f0[f0_length - 1] * 2.0 - coarse_f0[f0_length - 2];
    coarse_vuv[f0_length] = coarse_vuv[f0_length - 1] * 2.0 - coarse_vuv[f0_length - 2];

    (time_axis, coarse_time_axis, coarse_f0, coarse_vuv)
}

/// パルス位置検出
fn get_pulse_locations_for_time_base(
    interpolated_f0: &[f64],
    time_axis: &[f64],
    y_length: usize,
    fs: i32,
) -> (Vec<f64>, Vec<usize>, Vec<f64>) {
    let mut total_phase = vec![0.0; y_length];
    let mut wrap_phase = vec![0.0; y_length];

    total_phase[0] = 2.0 * PI * interpolated_f0[0] / fs as f64;
    wrap_phase[0] = total_phase[0] % (2.0 * PI);

    for i in 1..y_length {
        total_phase[i] = total_phase[i - 1] + 2.0 * PI * interpolated_f0[i] / fs as f64;
        wrap_phase[i] = total_phase[i] % (2.0 * PI);
    }

    let mut pulse_locations = Vec::new();
    let mut pulse_locations_index = Vec::new();
    let mut pulse_locations_time_shift = Vec::new();

    for i in 0..y_length - 1 {
        let wrap_diff = (wrap_phase[i + 1] - wrap_phase[i]).abs();
        if wrap_diff > PI {
            pulse_locations.push(time_axis[i]);
            pulse_locations_index.push(i);

            let y1 = wrap_phase[i] - 2.0 * PI;
            let y2 = wrap_phase[i + 1];
            let x = -y1 / (y2 - y1);
            pulse_locations_time_shift.push(x / fs as f64);
        }
    }

    (pulse_locations, pulse_locations_index, pulse_locations_time_shift)
}

/// タイムベース計算
fn get_time_base(
    f0: &[f64],
    fs: i32,
    frame_period: f64,
    y_length: usize,
    lowest_f0: f64,
) -> (Vec<f64>, Vec<usize>, Vec<f64>, Vec<f64>) {
    let (time_axis, coarse_time_axis, coarse_f0, coarse_vuv) =
        get_temporal_parameters_for_time_base(f0, fs, y_length, frame_period, lowest_f0);

    let mut interpolated_f0 = vec![0.0; y_length];
    let mut interpolated_vuv = vec![0.0; y_length];

    interp1(&coarse_time_axis, &coarse_f0, &time_axis, &mut interpolated_f0);
    interp1(&coarse_time_axis, &coarse_vuv, &time_axis, &mut interpolated_vuv);

    for i in 0..y_length {
        interpolated_vuv[i] = if interpolated_vuv[i] > 0.5 { 1.0 } else { 0.0 };
        if interpolated_vuv[i] == 0.0 {
            interpolated_f0[i] = DEFAULT_F0;
        }
    }

    let (pulse_locations, pulse_locations_index, pulse_locations_time_shift) =
        get_pulse_locations_for_time_base(&interpolated_f0, &time_axis, y_length, fs);

    (
        pulse_locations,
        pulse_locations_index,
        pulse_locations_time_shift,
        interpolated_vuv,
    )
}

/// DC 除去フィルタ設計
fn get_dc_remover(fft_size: usize) -> Vec<f64> {
    let mut dc_remover = vec![0.0; fft_size];
    let mut dc_component = 0.0;

    for i in 0..fft_size / 2 {
        dc_remover[i] =
            0.5 - 0.5 * (2.0 * PI * (i as f64 + 1.0) / (1.0 + fft_size as f64)).cos();
        dc_remover[fft_size - i - 1] = dc_remover[i];
        dc_component += dc_remover[i] * 2.0;
    }
    for i in 0..fft_size / 2 {
        dc_remover[i] /= dc_component;
        dc_remover[fft_size - i - 1] = dc_remover[i];
    }

    dc_remover
}

/// スペクトル包絡の時間補間
fn get_spectral_envelope(
    current_time: f64,
    frame_period: f64,
    f0_length: usize,
    spectrogram: &Array2<f64>,
    fft_size: usize,
) -> Vec<f64> {
    let current_frame_floor = ((current_time / frame_period).floor() as usize).min(f0_length - 1);
    let current_frame_ceil = ((current_time / frame_period).ceil() as usize).min(f0_length - 1);
    let interpolation = current_time / frame_period - current_frame_floor as f64;

    let spec_len = fft_size / 2 + 1;
    let mut spectral_envelope = vec![0.0; spec_len];

    if current_frame_floor == current_frame_ceil {
        for i in 0..spec_len {
            spectral_envelope[i] = spectrogram[[current_frame_floor, i]].abs();
        }
    } else {
        for i in 0..spec_len {
            spectral_envelope[i] = (1.0 - interpolation)
                * spectrogram[[current_frame_floor, i]].abs()
                + interpolation * spectrogram[[current_frame_ceil, i]].abs();
        }
    }

    spectral_envelope
}

/// 非周期性比率の時間補間
fn get_aperiodic_ratio(
    current_time: f64,
    frame_period: f64,
    f0_length: usize,
    aperiodicity: &Array2<f64>,
    fft_size: usize,
) -> Vec<f64> {
    let current_frame_floor = ((current_time / frame_period).floor() as usize).min(f0_length - 1);
    let current_frame_ceil = ((current_time / frame_period).ceil() as usize).min(f0_length - 1);
    let interpolation = current_time / frame_period - current_frame_floor as f64;

    let spec_len = fft_size / 2 + 1;
    let mut aperiodic_spectrum = vec![0.0; spec_len];

    if current_frame_floor == current_frame_ceil {
        for i in 0..spec_len {
            aperiodic_spectrum[i] =
                get_safe_aperiodicity(aperiodicity[[current_frame_floor, i]]).powi(2);
        }
    } else {
        for i in 0..spec_len {
            let val = (1.0 - interpolation)
                * get_safe_aperiodicity(aperiodicity[[current_frame_floor, i]])
                + interpolation * get_safe_aperiodicity(aperiodicity[[current_frame_ceil, i]]);
            aperiodic_spectrum[i] = val.powi(2);
        }
    }

    aperiodic_spectrum
}

/// ノイズスペクトル生成
fn get_noise_spectrum(
    noise_size: usize,
    fft_size: usize,
    randn_state: &mut RandnState,
) -> Vec<num_complex::Complex64> {
    let mut waveform = vec![0.0; fft_size];
    let mut average = 0.0;
    for i in 0..noise_size {
        waveform[i] = randn(randn_state);
        average += waveform[i];
    }
    average /= noise_size as f64;
    for i in 0..noise_size {
        waveform[i] -= average;
    }

    forward_real_fft(&waveform, fft_size)
}

/// 分数時間シフト付きスペクトル計算
fn get_spectrum_with_fractional_time_shift(
    spectrum: &mut [num_complex::Complex64],
    fft_size: usize,
    coefficient: f64,
) {
    for i in 0..=fft_size / 2 {
        let re = spectrum[i].re;
        let im = spectrum[i].im;
        let re2 = (coefficient * i as f64).cos();
        let im2 = (1.0 - re2 * re2).sqrt(); // sin(pshift)

        spectrum[i].re = re * re2 + im * im2;
        spectrum[i].im = im * re2 - re * im2;
    }
}

/// DC 成分除去
fn remove_dc_component(
    periodic_response: &[f64],
    fft_size: usize,
    dc_remover: &[f64],
    output: &mut [f64],
) {
    let mut dc_component = 0.0;
    for i in fft_size / 2..fft_size {
        dc_component += periodic_response[i];
    }
    for i in 0..fft_size / 2 {
        output[i] = -dc_component * dc_remover[i];
    }
    for i in fft_size / 2..fft_size {
        output[i] -= dc_component * dc_remover[i];
    }
}

/// 周期成分応答
fn get_periodic_response(
    fft_size: usize,
    spectrum: &[f64],
    aperiodic_ratio: &[f64],
    current_vuv: f64,
    dc_remover: &[f64],
    fractional_time_shift: f64,
    fs: i32,
) -> Vec<f64> {
    if current_vuv <= 0.5 || aperiodic_ratio[0] > 0.999 {
        return vec![0.0; fft_size];
    }

    let mut log_spectrum = vec![0.0; fft_size / 2 + 1];
    for i in 0..=fft_size / 2 {
        log_spectrum[i] =
            (spectrum[i] * (1.0 - aperiodic_ratio[i]) + SAFE_GUARD_MINIMUM).ln() / 2.0;
    }

    let min_phase = get_minimum_phase_spectrum(&log_spectrum, fft_size);

    // IFFT 用スペクトルにコピー
    let mut inv_spectrum: Vec<num_complex::Complex64> = vec![num_complex::Complex64::new(0.0, 0.0); fft_size / 2 + 1];
    for i in 0..=fft_size / 2 {
        inv_spectrum[i] = min_phase[i];
    }

    // 分数時間シフト
    let coefficient = 2.0 * PI * fractional_time_shift * fs as f64 / fft_size as f64;
    get_spectrum_with_fractional_time_shift(&mut inv_spectrum, fft_size, coefficient);

    let waveform = inverse_real_fft(&inv_spectrum, fft_size);

    // fftshift
    let mut shifted = vec![0.0; fft_size];
    fftshift(&waveform, &mut shifted);

    // DC 除去
    let mut periodic_response = vec![0.0; fft_size];
    remove_dc_component(&shifted, fft_size, dc_remover, &mut periodic_response);

    // shifted に periodic_response を加算
    for i in 0..fft_size {
        periodic_response[i] += shifted[i];
    }

    periodic_response
}

/// 非周期成分応答
fn get_aperiodic_response(
    noise_size: usize,
    fft_size: usize,
    spectrum: &[f64],
    aperiodic_ratio: &[f64],
    current_vuv: f64,
    randn_state: &mut RandnState,
) -> Vec<f64> {
    let noise_spectrum = get_noise_spectrum(noise_size, fft_size, randn_state);

    let mut log_spectrum = vec![0.0; fft_size / 2 + 1];
    if current_vuv != 0.0 {
        for i in 0..=fft_size / 2 {
            log_spectrum[i] = (spectrum[i] * aperiodic_ratio[i]).ln() / 2.0;
        }
    } else {
        for i in 0..=fft_size / 2 {
            log_spectrum[i] = spectrum[i].ln() / 2.0;
        }
    }

    let min_phase = get_minimum_phase_spectrum(&log_spectrum, fft_size);

    // 最小位相スペクトル × ノイズスペクトル
    let mut inv_spectrum: Vec<num_complex::Complex64> = vec![num_complex::Complex64::new(0.0, 0.0); fft_size / 2 + 1];
    for i in 0..=fft_size / 2 {
        inv_spectrum[i] = min_phase[i] * noise_spectrum[i];
    }

    let waveform = inverse_real_fft(&inv_spectrum, fft_size);

    let mut aperiodic_response = vec![0.0; fft_size];
    fftshift(&waveform, &mut aperiodic_response);

    aperiodic_response
}

/// 1フレームセグメント生成
fn get_one_frame_segment(
    current_vuv: f64,
    noise_size: usize,
    spectrogram: &Array2<f64>,
    fft_size: usize,
    aperiodicity: &Array2<f64>,
    f0_length: usize,
    frame_period: f64,
    current_time: f64,
    fractional_time_shift: f64,
    fs: i32,
    dc_remover: &[f64],
    randn_state: &mut RandnState,
) -> Vec<f64> {
    let spectral_envelope = get_spectral_envelope(
        current_time,
        frame_period,
        f0_length,
        spectrogram,
        fft_size,
    );
    let aperiodic_ratio = get_aperiodic_ratio(
        current_time,
        frame_period,
        f0_length,
        aperiodicity,
        fft_size,
    );

    let periodic_response = get_periodic_response(
        fft_size,
        &spectral_envelope,
        &aperiodic_ratio,
        current_vuv,
        dc_remover,
        fractional_time_shift,
        fs,
    );

    let aperiodic_response = get_aperiodic_response(
        noise_size,
        fft_size,
        &spectral_envelope,
        &aperiodic_ratio,
        current_vuv,
        randn_state,
    );

    let sqrt_noise_size = (noise_size as f64).sqrt();
    let mut response = vec![0.0; fft_size];
    for i in 0..fft_size {
        response[i] =
            (periodic_response[i] * sqrt_noise_size + aperiodic_response[i]) / fft_size as f64;
    }

    response
}

/// 波形合成（ボコーダー）
///
/// # Arguments
/// * `f0` - 基本周波数列 (Hz)
/// * `spectrogram` - スペクトル包絡 [num_frames x fft_size/2+1]
/// * `aperiodicity` - 非周期性指標 [num_frames x fft_size/2+1]
/// * `frame_period` - フレーム周期 (ms)
/// * `fs` - サンプリング周波数
/// * `fft_size` - FFT サイズ
///
/// # Returns
/// 合成波形
pub fn synthesis(
    f0: &[f64],
    spectrogram: &Array2<f64>,
    aperiodicity: &Array2<f64>,
    frame_period: f64,
    fs: i32,
    fft_size: usize,
) -> Array1<f64> {
    let f0_length = f0.len();
    let frame_period_sec = frame_period / 1000.0;

    // 出力長の計算
    let y_length = ((f0_length as f64 - 1.0) * frame_period_sec * fs as f64) as usize + 1;
    let mut y = vec![0.0; y_length];

    let mut randn_state = RandnState::new();

    let lowest_f0 = fs as f64 / fft_size as f64 + 1.0;

    let (pulse_locations, pulse_locations_index, pulse_locations_time_shift, interpolated_vuv) =
        get_time_base(f0, fs, frame_period_sec, y_length, lowest_f0);

    let dc_remover = get_dc_remover(fft_size);

    let number_of_pulses = pulse_locations.len();

    for i in 0..number_of_pulses {
        let noise_size = if i + 1 < number_of_pulses {
            pulse_locations_index[i + 1] - pulse_locations_index[i]
        } else {
            if i > 0 {
                pulse_locations_index[i] - pulse_locations_index[i - 1]
            } else {
                1
            }
        };
        let noise_size = noise_size.max(1);

        let response = get_one_frame_segment(
            interpolated_vuv[pulse_locations_index[i]],
            noise_size,
            spectrogram,
            fft_size,
            aperiodicity,
            f0_length,
            frame_period_sec,
            pulse_locations[i],
            pulse_locations_time_shift[i],
            fs,
            &dc_remover,
            &mut randn_state,
        );

        let offset = pulse_locations_index[i] as i64 - fft_size as i64 / 2 + 1;
        let lower_limit = if offset < 0 { (-offset) as usize } else { 0 };
        let upper_limit = fft_size.min((y_length as i64 - offset) as usize);

        for j in lower_limit..upper_limit {
            let index = (j as i64 + offset) as usize;
            y[index] += response[j];
        }
    }

    Array1::from_vec(y)
}
