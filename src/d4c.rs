use ndarray::Array2;
use ndarray_rand::rand_distr::StandardNormal;
use rand::Rng;
use rayon::prelude::*;

use crate::common::*;
use crate::constant::*;
use crate::matlab::*;

/// D4C 用の F0 適応窓掛け波形取得
fn get_windowed_waveform_d4c(
    x: &[f64],
    fs: i32,
    current_f0: f64,
    current_position: f64,
    window_type: i32,
    window_length_ratio: f64,
    fft_size: usize,
    rng: &mut impl Rng,
) -> Vec<f64> {
    let half_window_length =
        matlab_round(window_length_ratio * fs as f64 / current_f0 / 2.0) as usize;
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

    // 窓関数設計
    if window_type == HANNING {
        for i in 0..window_length {
            let position = (2.0 * base_index[i] as f64 / window_length_ratio) / fs as f64;
            window[i] = 0.5 * (PI * position * current_f0).cos() + 0.5;
        }
    } else {
        // Blackman
        for i in 0..window_length {
            let position = (2.0 * base_index[i] as f64 / window_length_ratio) / fs as f64;
            window[i] = 0.42
                + 0.5 * (PI * position * current_f0).cos()
                + 0.08 * (PI * position * current_f0 * 2.0).cos();
        }
    }

    // F0 適応窓掛け
    let mut waveform = vec![0.0; fft_size];
    for i in 0..window_length {
        waveform[i] =
            x[safe_index[i]] * window[i] + rng.sample::<f64, _>(StandardNormal) * SAFE_GUARD_D4C;
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

/// エネルギー重心計算
fn get_centroid(
    x: &[f64],
    fs: i32,
    current_f0: f64,
    fft_size: usize,
    current_position: f64,
    rng: &mut impl Rng,
) -> Vec<f64> {
    let mut waveform = get_windowed_waveform_d4c(
        x,
        fs,
        current_f0,
        current_position,
        BLACKMAN,
        4.0,
        fft_size,
        rng,
    );

    let waveform_len = matlab_round(2.0 * fs as f64 / current_f0) as usize * 2 + 1;
    let mut power = 0.0;
    for i in 0..waveform_len {
        power += waveform[i] * waveform[i];
    }
    let power_sqrt = power.sqrt();
    for i in 0..waveform_len {
        waveform[i] /= power_sqrt;
    }

    let spectrum = forward_real_fft(&waveform, fft_size);
    let tmp_real: Vec<f64> = (0..=fft_size / 2).map(|i| spectrum[i].re).collect();
    let tmp_imag: Vec<f64> = (0..=fft_size / 2).map(|i| spectrum[i].im).collect();

    // 波形に (i+1) を掛ける
    for i in 0..fft_size {
        waveform[i] *= (i + 1) as f64;
    }
    let spectrum2 = forward_real_fft(&waveform, fft_size);

    let mut centroid = vec![0.0; fft_size / 2 + 1];
    for i in 0..=fft_size / 2 {
        centroid[i] = spectrum2[i].re * tmp_real[i] + tmp_imag[i] * spectrum2[i].im;
    }

    centroid
}

/// 時間的に静的なエネルギー重心
fn get_static_centroid(
    x: &[f64],
    fs: i32,
    current_f0: f64,
    fft_size: usize,
    current_position: f64,
    rng: &mut impl Rng,
) -> Vec<f64> {
    let centroid1 = get_centroid(
        x,
        fs,
        current_f0,
        fft_size,
        current_position - 0.25 / current_f0,
        rng,
    );
    let centroid2 = get_centroid(
        x,
        fs,
        current_f0,
        fft_size,
        current_position + 0.25 / current_f0,
        rng,
    );

    let mut static_centroid = vec![0.0; fft_size / 2 + 1];
    for i in 0..=fft_size / 2 {
        static_centroid[i] = centroid1[i] + centroid2[i];
    }

    dc_correction(
        &static_centroid.clone(),
        current_f0,
        fs,
        fft_size,
        &mut static_centroid,
    );
    static_centroid
}

/// 平滑化パワースペクトル
fn get_smoothed_power_spectrum(
    x: &[f64],
    fs: i32,
    current_f0: f64,
    fft_size: usize,
    current_position: f64,
    rng: &mut impl Rng,
) -> Vec<f64> {
    let waveform = get_windowed_waveform_d4c(
        x,
        fs,
        current_f0,
        current_position,
        HANNING,
        4.0,
        fft_size,
        rng,
    );

    let spectrum = forward_real_fft(&waveform, fft_size);
    let mut power_spectrum = vec![0.0; fft_size / 2 + 1];
    for i in 0..=fft_size / 2 {
        power_spectrum[i] = spectrum[i].re * spectrum[i].re + spectrum[i].im * spectrum[i].im;
    }

    dc_correction(
        &power_spectrum.clone(),
        current_f0,
        fs,
        fft_size,
        &mut power_spectrum,
    );

    let mut smoothed = vec![0.0; fft_size / 2 + 1];
    linear_smoothing(&power_spectrum, current_f0, fs, fft_size, &mut smoothed);
    smoothed
}

/// 時間的に静的な群遅延
fn get_static_group_delay(
    static_centroid: &[f64],
    smoothed_power_spectrum: &[f64],
    fs: i32,
    f0: f64,
    fft_size: usize,
) -> Vec<f64> {
    let half = fft_size / 2;
    let mut static_group_delay = vec![0.0; half + 1];
    for i in 0..=half {
        static_group_delay[i] = static_centroid[i] / smoothed_power_spectrum[i];
    }

    let mut tmp = vec![0.0; half + 1];
    linear_smoothing(&static_group_delay, f0 / 2.0, fs, fft_size, &mut tmp);
    std::mem::swap(&mut static_group_delay, &mut tmp);

    let mut smoothed_group_delay = vec![0.0; half + 1];
    linear_smoothing(
        &static_group_delay,
        f0,
        fs,
        fft_size,
        &mut smoothed_group_delay,
    );

    for i in 0..=half {
        static_group_delay[i] -= smoothed_group_delay[i];
    }

    static_group_delay
}

/// 粗い非周期性を 3kHz 間隔で計算
fn get_coarse_aperiodicity(
    static_group_delay: &[f64],
    fs: i32,
    fft_size: usize,
    number_of_aperiodicities: usize,
    window: &[f64],
    window_length: usize,
) -> Vec<f64> {
    let boundary = matlab_round(fft_size as f64 * 8.0 / window_length as f64) as usize;
    let half_window_length = window_length / 2;

    let mut coarse_aperiodicity = vec![0.0; number_of_aperiodicities];

    for i in 0..number_of_aperiodicities {
        let center = (FREQUENCY_INTERVAL * (i + 1) as f64 * fft_size as f64 / fs as f64) as usize;

        let mut waveform = vec![0.0; fft_size];
        for j in 0..=half_window_length * 2 {
            let idx = center - half_window_length + j;
            if idx < static_group_delay.len() {
                waveform[j] = static_group_delay[idx] * window[j];
            }
        }

        let spectrum = forward_real_fft(&waveform, fft_size);
        let mut power_spectrum: Vec<f64> = (0..=fft_size / 2)
            .map(|j| spectrum[j].re * spectrum[j].re + spectrum[j].im * spectrum[j].im)
            .collect();

        power_spectrum.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // 累積和
        for j in 1..power_spectrum.len() {
            power_spectrum[j] += power_spectrum[j - 1];
        }

        let idx_high = fft_size / 2;
        let idx_low = if idx_high > boundary {
            idx_high - boundary - 1
        } else {
            0
        };
        coarse_aperiodicity[i] =
            10.0 * (power_spectrum[idx_low] / power_spectrum[idx_high]).log10();
    }

    coarse_aperiodicity
}

/// D4C LoveTrain — VUV 判定
fn d4c_love_train(x: &[f64], fs: i32, f0: &[f64], temporal_positions: &[f64]) -> Vec<f64> {
    let lowest_f0 = 40.0;
    let fft_size =
        (2.0_f64).powf(1.0 + ((3.0 * fs as f64 / lowest_f0 + 1.0).ln() / LOG2).floor()) as usize;

    let boundary0 = (100.0 * fft_size as f64 / fs as f64).ceil() as usize;
    let boundary1 = (4000.0 * fft_size as f64 / fs as f64).ceil() as usize;
    let boundary2 = (7900.0 * fft_size as f64 / fs as f64).ceil() as usize;

    let f0_length = f0.len();

    (0..f0_length)
        .into_par_iter()
        .map(|i| {
            if f0[i] == 0.0 {
                return 0.0;
            }

            let mut rng = rand::rng();
            let current_f0 = f0[i].max(lowest_f0);
            let waveform = get_windowed_waveform_d4c(
                x,
                fs,
                current_f0,
                temporal_positions[i],
                BLACKMAN,
                3.0,
                fft_size,
                &mut rng,
            );

            let spectrum = forward_real_fft(&waveform, fft_size);

            let mut power_spectrum = vec![0.0; fft_size / 2 + 1];
            for j in boundary0 + 1..=fft_size / 2 {
                power_spectrum[j] =
                    spectrum[j].re * spectrum[j].re + spectrum[j].im * spectrum[j].im;
            }

            for j in boundary0 + 1..=boundary2 {
                power_spectrum[j] += power_spectrum[j - 1];
            }

            power_spectrum[boundary1] / power_spectrum[boundary2]
        })
        .collect()
}

/// D4C の1フレーム処理
fn d4c_general_body(
    x: &[f64],
    fs: i32,
    current_f0: f64,
    fft_size: usize,
    current_position: f64,
    number_of_aperiodicities: usize,
    window: &[f64],
    window_length: usize,
    rng: &mut impl Rng,
) -> Vec<f64> {
    let static_centroid = get_static_centroid(x, fs, current_f0, fft_size, current_position, rng);
    let smoothed_power =
        get_smoothed_power_spectrum(x, fs, current_f0, fft_size, current_position, rng);
    let static_group_delay =
        get_static_group_delay(&static_centroid, &smoothed_power, fs, current_f0, fft_size);

    let mut coarse_aperiodicity = get_coarse_aperiodicity(
        &static_group_delay,
        fs,
        fft_size,
        number_of_aperiodicities,
        window,
        window_length,
    );

    // F0 ベースの補正
    for i in 0..number_of_aperiodicities {
        coarse_aperiodicity[i] = (coarse_aperiodicity[i] + (current_f0 - 100.0) / 50.0).min(0.0);
    }

    coarse_aperiodicity
}

/// D4C 非周期性指標推定
///
/// # Arguments
/// * `x` - 入力波形
/// * `fs` - サンプリング周波数
/// * `temporal_positions` - 各フレームの時間位置 (秒)
/// * `f0` - 各フレームの基本周波数 (Hz)
/// * `fft_size` - FFT サイズ
/// * `option` - D4C
///
/// # Returns
/// 非周期性指標 [num_frames x fft_size/2+1] 値域: [0, 1]
pub fn d4c(
    x: &[f64],
    fs: i32,
    temporal_positions: &[f64],
    f0: &[f64],
    fft_size: usize,
    option: &D4C,
) -> Array2<f64> {
    let f0_length = f0.len();
    let spec_len = fft_size / 2 + 1;

    // 初期値: 1.0 - SAFE_GUARD_MINIMUM
    let mut aperiodicity = Array2::from_elem((f0_length, spec_len), 1.0 - SAFE_GUARD_MINIMUM);

    let fft_size_d4c =
        (2.0_f64).powf(1.0 + ((4.0 * fs as f64 / FLOOR_F0_D4C + 1.0).ln() / LOG2).floor()) as usize;

    let number_of_aperiodicities =
        ((fs as f64 / 2.0 - FREQUENCY_INTERVAL).min(UPPER_LIMIT) / FREQUENCY_INTERVAL) as usize;

    let window_length = (FREQUENCY_INTERVAL * fft_size_d4c as f64 / fs as f64) as usize * 2 + 1;
    let window = nuttall_window(window_length);

    // D4C LoveTrain
    let aperiodicity0 = d4c_love_train(x, fs, f0, temporal_positions);

    // 粗い周波数軸
    let mut coarse_frequency_axis = vec![0.0; number_of_aperiodicities + 2];
    for i in 0..=number_of_aperiodicities {
        coarse_frequency_axis[i] = i as f64 * FREQUENCY_INTERVAL;
    }
    coarse_frequency_axis[number_of_aperiodicities + 1] = fs as f64 / 2.0;

    let frequency_axis: Vec<f64> = (0..spec_len)
        .map(|i| i as f64 * fs as f64 / fft_size as f64)
        .collect();

    // 各フレームの非周期性を並列計算
    let frame_results: Vec<Option<Vec<f64>>> = (0..f0_length)
        .into_par_iter()
        .map(|i| {
            if f0[i] == 0.0 || aperiodicity0[i] <= option.threshold {
                return None;
            }

            let mut rng = rand::rng();
            let current_f0 = f0[i].max(FLOOR_F0_D4C);

            let coarse_ap = d4c_general_body(
                x,
                fs,
                current_f0,
                fft_size_d4c,
                temporal_positions[i],
                number_of_aperiodicities,
                &window,
                window_length,
                &mut rng,
            );

            let mut full_coarse = vec![0.0; number_of_aperiodicities + 2];
            full_coarse[0] = -60.0;
            for j in 0..number_of_aperiodicities {
                full_coarse[j + 1] = coarse_ap[j];
            }
            full_coarse[number_of_aperiodicities + 1] = -SAFE_GUARD_MINIMUM;

            let mut ap_interp = vec![0.0; spec_len];
            interp1(
                &coarse_frequency_axis,
                &full_coarse,
                &frequency_axis,
                &mut ap_interp,
            );

            let row: Vec<f64> = (0..spec_len)
                .map(|j| (10.0_f64).powf(ap_interp[j] / 20.0))
                .collect();
            Some(row)
        })
        .collect();

    for (i, result) in frame_results.into_iter().enumerate() {
        if let Some(row) = result {
            for j in 0..spec_len {
                aperiodicity[[i, j]] = row[j];
            }
        }
    }

    aperiodicity
}

impl D4C {
    /// 非周期性指標を推定する
    pub fn estimate(
        &self,
        x: &[f64],
        temporal_positions: &[f64],
        f0: &[f64],
    ) -> Array2<f64> {
        d4c(x, self.fs, temporal_positions, f0, self.fft_size, self)
    }
}
