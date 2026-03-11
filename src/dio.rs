use num_complex::Complex64;
use rayon::prelude::*;

use crate::common::{forward_real_fft, inverse_real_fft, nuttall_window};
use crate::constant::*;
use crate::matlab::{decimate, diff, interp1, matlab_round};

/// DIO ピッチ推定。
///
/// ゼロクロッシング解析と FFT ベースのバンドパスフィルタリングにより F0 を推定する。
/// 各帯域の処理は rayon で並列化される。
///
/// # Arguments
/// * `x` - 入力波形（モノラル）
/// * `fs` - サンプリング周波数 (Hz)
/// * `option` - DIO パラメータ
///
/// # Returns
/// `(temporal_positions, f0)` のタプル。
/// - `temporal_positions` - 各フレームの時間位置 (秒), 長さ `num_frames`
/// - `f0` - 各フレームの基本周波数 (Hz), 長さ `num_frames`。無声フレームは 0.0
pub fn dio(x: &[f64], fs: i32, option: &Dio) -> (Vec<f64>, Vec<f64>) {
    let f0_length = get_samples_for_dio(fs, x.len(), option.frame_period);
    let mut temporal_positions = vec![0.0; f0_length];
    for i in 0..f0_length {
        temporal_positions[i] = i as f64 * option.frame_period / 1000.0;
    }

    let decimation_ratio = get_decimation_ratio(option.speed, fs);
    let actual_fs = fs / decimation_ratio;

    // デシメーション
    let y = if decimation_ratio != 1 {
        decimate(x, decimation_ratio)
    } else {
        x.to_vec()
    };
    let y_length = y.len();

    // バンド数
    let number_of_bands = 1
        + (((option.f0_ceil / option.f0_floor)).ln() / LOG2 * option.channels_in_octave)
            as usize;

    // boundary F0 リスト
    let boundary_f0_list: Vec<f64> = (0..number_of_bands)
        .map(|i| option.f0_floor * (2.0_f64).powf((i + 1) as f64 / option.channels_in_octave))
        .collect();

    // C++: fft_size = GetSuitableFFTSize(y_length +
    //   matlab_round(actual_fs / kCutOff) * 2 + 1 +
    //   (4 * (int)(1.0 + actual_fs / boundary_f0_list[0] / 2.0)))
    let fft_size = get_suitable_fft_size(
        y_length
            + (matlab_round(actual_fs as f64 / CUTOFF) * 2 + 1) as usize
            + (4.0 * (1.0 + actual_fs as f64 / boundary_f0_list[0] / 2.0)) as usize,
    );

    // スペクトル計算（ローカットフィルタ適用済み）
    let y_spectrum =
        get_spectrum_for_estimation(&y, y_length, actual_fs, fft_size, decimation_ratio);

    // 各帯域の F0 候補とスコアを並列計算
    let band_results: Vec<(Vec<f64>, Vec<f64>)> = (0..number_of_bands)
        .into_par_iter()
        .map(|i| {
            let half_avg_len = matlab_round(actual_fs as f64 / boundary_f0_list[i] / 2.0) as usize;
            let filtered_signal =
                get_filtered_signal(half_avg_len, fft_size, &y_spectrum, y_length);
            let zero_crossings = get_four_zero_crossing_intervals(&filtered_signal, actual_fs);
            let mut candidates = vec![0.0; f0_length];
            let mut scores = vec![0.0; f0_length];
            get_f0_candidate_contour(
                &zero_crossings,
                boundary_f0_list[i],
                actual_fs as f64,
                &temporal_positions,
                f0_length,
                &mut candidates,
                &mut scores,
            );
            // C++: raw_f0_scores[i][j] = f0_score[j] / (f0_candidate[j] + safeguard)
            for j in 0..f0_length {
                scores[j] = scores[j] / (candidates[j] + SAFE_GUARD_MINIMUM);
            }
            (candidates, scores)
        })
        .collect();

    let f0_candidates: Vec<Vec<f64>> = band_results.iter().map(|(c, _)| c.clone()).collect();
    let f0_scores: Vec<Vec<f64>> = band_results.iter().map(|(_, s)| s.clone()).collect();

    // 最良候補選択
    let best_f0 = get_best_f0_contour(&f0_candidates, &f0_scores, f0_length, number_of_bands);

    // C++ WORLD 準拠の後処理
    fix_f0_contour(
        option.frame_period,
        number_of_bands,
        &f0_candidates,
        &f0_scores,
        &best_f0,
        f0_length,
        option.f0_floor,
        option.allowed_range,
        &mut temporal_positions,
    )
}

/// speed パラメータからデシメーション比率を計算する。
fn get_decimation_ratio(speed: i32, fs: i32) -> i32 {
    if speed <= 1 {
        return 1;
    }
    let ratio = speed.min(12);
    // 確認: actual_fs が妥当か
    if fs / ratio < 100 { 1 } else { ratio }
}

/// ローカットフィルタを適用したスペクトルを取得（C++ WORLD 準拠）
fn get_spectrum_for_estimation(
    y: &[f64],
    y_length: usize,
    actual_fs: i32,
    fft_size: usize,
    _decimation_ratio: i32,
) -> Vec<Complex64> {
    let mut y_padded = vec![0.0; fft_size];
    for i in 0..y_length.min(fft_size) {
        y_padded[i] = y[i];
    }

    // C++: DC 成分除去
    let mean_y: f64 = y_padded[..y_length].iter().sum::<f64>() / y_length as f64;
    for i in 0..y_length {
        y_padded[i] -= mean_y;
    }

    let mut y_spectrum = forward_real_fft(&y_padded, fft_size);

    // ローカットフィルタ設計・適用
    let low_cut_filter = design_low_cut_filter(actual_fs, fft_size);
    let filter_spectrum = forward_real_fft(&low_cut_filter, fft_size);

    let half = fft_size / 2 + 1;
    for i in 0..half {
        // 複素数乗算
        let tmp = y_spectrum[i].re * filter_spectrum[i].re
            - y_spectrum[i].im * filter_spectrum[i].im;
        y_spectrum[i].im = y_spectrum[i].re * filter_spectrum[i].im
            + y_spectrum[i].im * filter_spectrum[i].re;
        y_spectrum[i].re = tmp;
    }

    y_spectrum
}

/// ローカットフィルタ設計（C++ WORLD の DesignLowCutFilter に準拠）
fn design_low_cut_filter(actual_fs: i32, fft_size: usize) -> Vec<f64> {
    let cutoff_in_sample = matlab_round(actual_fs as f64 / CUTOFF);
    let n = (cutoff_in_sample * 2 + 1) as usize;
    let mut low_cut_filter = vec![0.0; fft_size];

    // Half-Hanning window
    for i in 1..=n {
        low_cut_filter[i - 1] = 0.5 - 0.5 * (i as f64 * 2.0 * PI / (n as f64 + 1.0)).cos();
    }
    for i in n..fft_size {
        low_cut_filter[i] = 0.0;
    }

    // Normalize and negate (low-pass)
    let sum_of_amplitude: f64 = low_cut_filter[..n].iter().sum();
    for i in 0..n {
        low_cut_filter[i] = -low_cut_filter[i] / sum_of_amplitude;
    }

    // Circular shift: move first (N-1)/2 samples to end
    let half_n = (n - 1) / 2;
    for i in 0..half_n {
        low_cut_filter[fft_size - half_n + i] = low_cut_filter[i];
    }

    // Shift remaining left by (N-1)/2
    for i in 0..n {
        low_cut_filter[i] = low_cut_filter[i + half_n];
    }
    // Zero out the space left after shifting
    for i in n..fft_size - half_n {
        low_cut_filter[i] = 0.0;
    }

    // Add delta (high-pass = delta - low-pass)
    low_cut_filter[0] += 1.0;

    low_cut_filter
}

/// Nuttall 窓ベースのローパスフィルタリングで帯域信号を取得
/// C++ WORLD の GetFilteredSignal() に準拠
fn get_filtered_signal(
    half_avg_len: usize,
    fft_size: usize,
    y_spectrum: &[Complex64],
    y_length: usize,
) -> Vec<f64> {
    // C++: NuttallWindow(half_average_length * 4, low_pass_filter)
    let filter_length = half_avg_len * 4;
    let nuttall = nuttall_window(filter_length);

    let mut low_pass = vec![0.0; fft_size];
    for i in 0..filter_length {
        low_pass[i] = nuttall[i];
    }

    let filter_spectrum = forward_real_fft(&low_pass, fft_size);

    let half = fft_size / 2 + 1;
    let mut filtered_spectrum = vec![Complex64::new(0.0, 0.0); fft_size];
    for i in 0..half {
        filtered_spectrum[i] = y_spectrum[i] * filter_spectrum[i];
    }
    for i in half..fft_size {
        filtered_spectrum[i] = filtered_spectrum[fft_size - i].conj();
    }

    let output = inverse_real_fft(&filtered_spectrum[..half], fft_size);

    // C++: index_bias = half_average_length * 2
    let delay = half_avg_len * 2;
    let mut result = vec![0.0; y_length];
    for i in 0..y_length {
        let idx = i + delay;
        if idx < output.len() {
            result[i] = output[idx];
        }
    }
    result
}

struct ZeroCrossings {
    negative_interval_locations: Vec<f64>,
    negative_intervals: Vec<f64>,
    positive_interval_locations: Vec<f64>,
    positive_intervals: Vec<f64>,
    peak_interval_locations: Vec<f64>,
    peak_intervals: Vec<f64>,
    dip_interval_locations: Vec<f64>,
    dip_intervals: Vec<f64>,
}

/// 4種類のゼロクロッシング解析
fn get_four_zero_crossing_intervals(filtered_signal: &[f64], actual_fs: i32) -> ZeroCrossings {
    // negative→positive (通常のゼロクロッシング)
    let (neg_locs, neg_intervals) = zero_crossing_engine(filtered_signal, actual_fs);

    // positive→negative (符号反転)
    let neg_signal: Vec<f64> = filtered_signal.iter().map(|&v| -v).collect();
    let (pos_locs, pos_intervals) = zero_crossing_engine(&neg_signal, actual_fs);

    // peak intervals (微分のゼロクロッシング)
    let d = diff(filtered_signal);
    let (peak_locs, peak_intervals) = zero_crossing_engine(&d, actual_fs);

    // dip intervals (微分の符号反転のゼロクロッシング)
    let neg_d: Vec<f64> = d.iter().map(|&v| -v).collect();
    let (dip_locs, dip_intervals) = zero_crossing_engine(&neg_d, actual_fs);

    ZeroCrossings {
        negative_interval_locations: neg_locs,
        negative_intervals: neg_intervals,
        positive_interval_locations: pos_locs,
        positive_intervals: pos_intervals,
        peak_interval_locations: peak_locs,
        peak_intervals,
        dip_interval_locations: dip_locs,
        dip_intervals,
    }
}

/// ゼロクロッシングエンジン：negative→positiveの交差を検出
fn zero_crossing_engine(x: &[f64], fs: i32) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let mut locations = Vec::new();

    for i in 0..n - 1 {
        if x[i] * x[i + 1] < 0.0 && x[i] < 0.0 {
            // 線形補間で正確な位置
            let exact = i as f64 - x[i] / (x[i + 1] - x[i]);
            locations.push(exact / fs as f64);
        }
    }

    let mut intervals = Vec::new();
    let mut interval_locations = Vec::new();
    for i in 0..locations.len().saturating_sub(1) {
        intervals.push(1.0 / (locations[i + 1] - locations[i]));
        interval_locations.push((locations[i] + locations[i + 1]) / 2.0);
    }

    (interval_locations, intervals)
}

/// F0 候補とスコアの計算（C++ WORLD の GetF0CandidateContour + GetF0CandidateContourSub に準拠）
fn get_f0_candidate_contour(
    zero_crossings: &ZeroCrossings,
    boundary_f0: f64,
    _actual_fs: f64,
    temporal_positions: &[f64],
    f0_length: usize,
    f0_candidate: &mut [f64],
    f0_score: &mut [f64],
) {
    for i in 0..f0_length {
        f0_candidate[i] = 0.0;
        f0_score[i] = MAXIMUM_VALUE;
    }

    if zero_crossings.negative_intervals.len() < 2
        || zero_crossings.positive_intervals.len() < 2
        || zero_crossings.peak_intervals.len() < 2
        || zero_crossings.dip_intervals.len() < 2
    {
        return;
    }

    let mut interp_neg = vec![0.0; f0_length];
    let mut interp_pos = vec![0.0; f0_length];
    let mut interp_peak = vec![0.0; f0_length];
    let mut interp_dip = vec![0.0; f0_length];

    interp1_safe(
        &zero_crossings.negative_interval_locations,
        &zero_crossings.negative_intervals,
        temporal_positions,
        &mut interp_neg,
    );
    interp1_safe(
        &zero_crossings.positive_interval_locations,
        &zero_crossings.positive_intervals,
        temporal_positions,
        &mut interp_pos,
    );
    interp1_safe(
        &zero_crossings.peak_interval_locations,
        &zero_crossings.peak_intervals,
        temporal_positions,
        &mut interp_peak,
    );
    interp1_safe(
        &zero_crossings.dip_interval_locations,
        &zero_crossings.dip_intervals,
        temporal_positions,
        &mut interp_dip,
    );

    // C++ GetF0CandidateContourSub: 常に4値の平均、スコアは/3.0
    let f0_floor = boundary_f0 / 2.0;
    let f0_ceil = boundary_f0;
    for i in 0..f0_length {
        let mean = (interp_neg[i] + interp_pos[i] + interp_peak[i] + interp_dip[i]) / 4.0;

        // C++: スコア = sqrt(sum_of_squared_dev / 3.0)
        let score = ((interp_neg[i] - mean).powi(2)
            + (interp_pos[i] - mean).powi(2)
            + (interp_peak[i] - mean).powi(2)
            + (interp_dip[i] - mean).powi(2))
            / 3.0;
        let score = score.sqrt();

        // C++: boundary_f0 による範囲フィルタリング
        if mean > f0_ceil || mean < f0_floor {
            f0_candidate[i] = 0.0;
            f0_score[i] = MAXIMUM_VALUE;
        } else {
            f0_candidate[i] = mean;
            f0_score[i] = score;
        }
    }
}

/// 安全な補間（範囲外は0）
fn interp1_safe(x: &[f64], y: &[f64], xi: &[f64], yi: &mut [f64]) {
    if x.len() < 2 || y.len() < 2 {
        for v in yi.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let x_min = x[0];
    let x_max = x[x.len() - 1];

    // 範囲内のインデックスだけ補間
    let mut valid_xi = Vec::new();
    let mut valid_idx = Vec::new();
    for (i, &val) in xi.iter().enumerate() {
        if val >= x_min && val <= x_max {
            valid_xi.push(val);
            valid_idx.push(i);
        }
    }

    if valid_xi.is_empty() {
        for v in yi.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    let mut valid_yi = vec![0.0; valid_xi.len()];
    interp1(x, y, &valid_xi, &mut valid_yi);

    for v in yi.iter_mut() {
        *v = 0.0;
    }
    for (j, &i) in valid_idx.iter().enumerate() {
        yi[i] = valid_yi[j];
    }
}

/// 最良 F0 軌跡の選択（全帯域から最小スコア）
fn get_best_f0_contour(
    f0_candidates: &[Vec<f64>],
    f0_scores: &[Vec<f64>],
    f0_length: usize,
    number_of_bands: usize,
) -> Vec<f64> {
    let mut best_f0 = vec![0.0; f0_length];

    for i in 0..f0_length {
        let mut best_score = MAXIMUM_VALUE;
        let mut best_idx = 0;
        for j in 0..number_of_bands {
            if f0_scores[j][i] < best_score {
                best_score = f0_scores[j][i];
                best_idx = j;
            }
        }
        best_f0[i] = if best_score < MAXIMUM_VALUE {
            f0_candidates[best_idx][i]
        } else {
            0.0
        };
    }

    best_f0
}

/// C++ WORLD 準拠の後処理パイプライン
fn fix_f0_contour(
    frame_period: f64,
    number_of_bands: usize,
    f0_candidates: &[Vec<f64>],
    _f0_scores: &[Vec<f64>],
    best_f0_contour: &[f64],
    f0_length: usize,
    f0_floor: f64,
    allowed_range: f64,
    temporal_positions: &mut Vec<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let voice_range_minimum =
        (0.5 + 1000.0 / frame_period / f0_floor) as usize * 2 + 1;

    if f0_length <= voice_range_minimum {
        return (temporal_positions.clone(), best_f0_contour.to_vec());
    }

    let mut f0_step1 = vec![0.0; f0_length];
    let mut f0_step2 = vec![0.0; f0_length];

    // Step 1: allowed_range を超えるジャンプ除去
    {
        let mut f0_base = vec![0.0; f0_length];
        for i in voice_range_minimum..f0_length.saturating_sub(voice_range_minimum) {
            f0_base[i] = best_f0_contour[i];
        }
        for i in 0..voice_range_minimum {
            f0_step1[i] = 0.0;
        }
        for i in voice_range_minimum..f0_length {
            f0_step1[i] = if ((f0_base[i] - f0_base[i - 1]) / (SAFE_GUARD_MINIMUM + f0_base[i]))
                .abs()
                < allowed_range
            {
                f0_base[i]
            } else {
                0.0
            };
        }
    }

    // Step 2: 有声区間窓内に無声があれば除去
    {
        for i in 0..f0_length {
            f0_step2[i] = f0_step1[i];
        }
        let center = (voice_range_minimum - 1) / 2;
        for i in center..f0_length.saturating_sub(center) {
            let mut all_voiced = true;
            for j in 0..=center * 2 {
                let idx = i + j - center;
                if idx < f0_length && f0_step1[idx] == 0.0 {
                    all_voiced = false;
                    break;
                }
            }
            if !all_voiced {
                f0_step2[i] = 0.0;
            }
        }
    }

    // 有声区間の境界を検出
    let mut positive_index = Vec::new();
    let mut negative_index = Vec::new();
    for i in 1..f0_length {
        if f0_step2[i] == 0.0 && f0_step2[i - 1] != 0.0 {
            negative_index.push(i - 1);
        } else if f0_step2[i - 1] == 0.0 && f0_step2[i] != 0.0 {
            positive_index.push(i);
        }
    }

    // Step 3: 有声→無声の遷移から前方へ候補修正
    let mut f0_step3 = f0_step2.clone();
    for ni in 0..negative_index.len() {
        let limit = if ni == negative_index.len() - 1 {
            f0_length - 1
        } else {
            negative_index[ni + 1]
        };
        let mut j = negative_index[ni];
        while j < limit {
            let best = select_best_f0(
                f0_step3[j],
                f0_step3[j.saturating_sub(1)],
                f0_candidates,
                number_of_bands,
                j + 1,
                allowed_range,
            );
            f0_step3[j + 1] = best;
            if best == 0.0 {
                break;
            }
            j += 1;
        }
    }

    // Step 4: 無声→有声の遷移から後方へ候補修正
    let mut f0_step4 = f0_step3.clone();
    for pi in (0..positive_index.len()).rev() {
        let limit = if pi == 0 { 1 } else { positive_index[pi - 1] };
        let mut j = positive_index[pi];
        while j > limit {
            let best = select_best_f0(
                f0_step4[j],
                f0_step4[(j + 1).min(f0_length - 1)],
                f0_candidates,
                number_of_bands,
                j - 1,
                allowed_range,
            );
            f0_step4[j - 1] = best;
            if best == 0.0 {
                break;
            }
            j -= 1;
        }
    }

    (temporal_positions.clone(), f0_step4)
}

/// C++ WORLD 準拠の候補選択
/// reference_f0 = (current_f0 * 3 - past_f0) / 2 で外挿した基準値に最も近い候補を選ぶ
fn select_best_f0(
    current_f0: f64,
    past_f0: f64,
    f0_candidates: &[Vec<f64>],
    number_of_bands: usize,
    target_index: usize,
    allowed_range: f64,
) -> f64 {
    let reference_f0 = (current_f0 * 3.0 - past_f0) / 2.0;

    let mut best_f0 = f0_candidates[0]
        .get(target_index)
        .copied()
        .unwrap_or(0.0);
    let mut minimum_error = (reference_f0 - best_f0).abs();

    for j in 1..number_of_bands {
        let candidate = f0_candidates[j]
            .get(target_index)
            .copied()
            .unwrap_or(0.0);
        let current_error = (reference_f0 - candidate).abs();
        if current_error < minimum_error {
            minimum_error = current_error;
            best_f0 = candidate;
        }
    }

    if (1.0 - best_f0 / reference_f0).abs() > allowed_range {
        0.0
    } else {
        best_f0
    }
}

impl F0Estimator for Dio {
    fn estimate(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        dio(x, self.fs, self)
    }

    fn fs(&self) -> i32 {
        self.fs
    }

    fn frame_period(&self) -> f64 {
        self.frame_period
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dio_sine_440hz() {
        let fs = 16000;
        let duration = 0.5;
        let f0_true = 440.0;
        let n_samples = (fs as f64 * duration) as usize;
        let x: Vec<f64> = (0..n_samples)
            .map(|i| (2.0 * PI * f0_true * i as f64 / fs as f64).sin())
            .collect();

        let option = Dio::new(fs);
        let (temporal_positions, f0) = dio(&x, fs, &option);

        assert!(!f0.is_empty());
        assert_eq!(temporal_positions.len(), f0.len());

        // 中央付近のフレームで F0 が概ね 440Hz であることを確認
        let mid = f0.len() / 2;
        let voiced_f0: Vec<f64> = f0[mid - 2..=mid + 2]
            .iter()
            .filter(|&&v| v > 0.0)
            .copied()
            .collect();
        assert!(
            !voiced_f0.is_empty(),
            "Should have voiced frames near center"
        );
        for &v in &voiced_f0 {
            assert!(
                (v - f0_true).abs() < 50.0,
                "F0 {} should be close to {}",
                v,
                f0_true
            );
        }
    }
}
