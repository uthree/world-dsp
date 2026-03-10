use num_complex::Complex64;
use rayon::prelude::*;

use crate::common::{forward_real_fft, inverse_real_fft, nuttall_window};
use crate::constant::*;
use crate::matlab::{decimate, diff, interp1, matlab_round};

/// DIO ピッチ推定
///
/// F0（基本周波数）を推定する。DIO は高速で安定した F0 推定を行う。
/// 返り値は (temporal_positions, f0) のタプル。
pub fn dio(x: &[f64], fs: i32, option: &DioOption) -> (Vec<f64>, Vec<f64>) {
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

    // FFT サイズ
    let fft_size = get_suitable_fft_size(
        (actual_fs as f64 / CUTOFF + 1.0) as usize * 4 + (1.0 + y_length as f64) as usize,
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
            (candidates, scores)
        })
        .collect();

    let f0_candidates: Vec<Vec<f64>> = band_results.iter().map(|(c, _)| c.clone()).collect();
    let f0_scores: Vec<Vec<f64>> = band_results.iter().map(|(_, s)| s.clone()).collect();

    // 最良候補選択
    let mut best_f0 = get_best_f0_contour(&f0_candidates, &f0_scores, f0_length, number_of_bands);

    // 後処理
    fix_step1(&mut best_f0, option.allowed_range);
    fix_step2(&mut best_f0, option.allowed_range);
    fix_step3(
        &mut best_f0,
        &f0_candidates,
        &f0_scores,
        number_of_bands,
        option.allowed_range,
    );
    fix_step4(
        &mut best_f0,
        &f0_candidates,
        &f0_scores,
        number_of_bands,
        option.allowed_range,
    );

    (temporal_positions, best_f0)
}

fn get_decimation_ratio(speed: i32, fs: i32) -> i32 {
    if speed <= 1 {
        return 1;
    }
    let ratio = speed.min(12);
    // 確認: actual_fs が妥当か
    if fs / ratio < 100 { 1 } else { ratio }
}

/// ローカットフィルタを適用したスペクトルを取得
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
    let y_spectrum = forward_real_fft(&y_padded, fft_size);

    // ローカットフィルタ設計・適用
    let low_cut_filter = design_low_cut_filter(actual_fs, fft_size);
    let filter_spectrum = forward_real_fft(&low_cut_filter, fft_size);

    let mut result = vec![Complex64::new(0.0, 0.0); fft_size];
    let half = fft_size / 2 + 1;
    for i in 0..half {
        result[i] = y_spectrum[i] * filter_spectrum[i];
    }
    // ミラー
    for i in half..fft_size {
        result[i] = result[fft_size - i].conj();
    }
    result
}

/// ローカットフィルタ設計
fn design_low_cut_filter(actual_fs: i32, fft_size: usize) -> Vec<f64> {
    let filter_length_half = (actual_fs as f64 / CUTOFF * 2.0) as usize;
    let filter_length = 2 * filter_length_half + 1;
    let mut low_cut_filter = vec![0.0; fft_size];

    for i in 1..=filter_length_half {
        low_cut_filter[filter_length_half + i] =
            0.5 - 0.5 * (2.0 * PI * i as f64 / (filter_length as f64 - 1.0)).cos();
        low_cut_filter[filter_length_half - i] = low_cut_filter[filter_length_half + i];
    }

    let sum_of_amplitude: f64 = low_cut_filter.iter().sum();
    for i in 0..filter_length {
        low_cut_filter[i] /= sum_of_amplitude;
    }

    // ハイパスに変換
    let center_val = low_cut_filter[filter_length_half];
    for i in 0..filter_length {
        low_cut_filter[i] = -low_cut_filter[i];
    }
    low_cut_filter[filter_length_half] = center_val + 1.0;

    // Nuttall 窓適用
    let nuttall = nuttall_window(filter_length);
    for i in 0..filter_length {
        low_cut_filter[i] *= nuttall[i];
    }

    low_cut_filter
}

/// Nuttall 窓ベースのバンドパスフィルタリングで帯域信号を取得
fn get_filtered_signal(
    half_avg_len: usize,
    fft_size: usize,
    y_spectrum: &[Complex64],
    y_length: usize,
) -> Vec<f64> {
    let filter_length = 2 * half_avg_len + 1;
    let nuttall = nuttall_window(filter_length);

    let mut low_pass = vec![0.0; fft_size];
    let sum: f64 = nuttall.iter().sum();
    for i in 0..filter_length {
        low_pass[i] = nuttall[i] / sum;
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

    // IFFTでフィルタ済み信号に変換
    let mut output = inverse_real_fft(&filtered_spectrum[..half], fft_size);
    output.truncate(y_length);

    // 位相シフト補正
    let delay = half_avg_len;
    let mut result = vec![0.0; y_length];
    for i in 0..y_length {
        let idx = i + delay;
        if idx < y_length {
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

/// F0 候補とスコアの計算
fn get_f0_candidate_contour(
    zero_crossings: &ZeroCrossings,
    _boundary_f0: f64,
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

    // 各ゼロクロッシングタイプの F0 を temporal_positions に補間
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

    for i in 0..f0_length {
        let values = [interp_neg[i], interp_pos[i], interp_peak[i], interp_dip[i]];
        let n_valid = values.iter().filter(|&&v| v > 0.0).count();
        if n_valid < 2 {
            continue;
        }

        let mean = values.iter().filter(|&&v| v > 0.0).sum::<f64>() / n_valid as f64;

        // スコア = 4種の標準偏差 / 平均値
        let var = values
            .iter()
            .filter(|&&v| v > 0.0)
            .map(|&v| (v - mean).powi(2))
            .sum::<f64>()
            / n_valid as f64;
        let score = var.sqrt() / mean;

        f0_candidate[i] = mean;
        f0_score[i] = score;
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

/// 後処理 Step 1: allowed_range を超えるジャンプ除去
fn fix_step1(f0: &mut [f64], allowed_range: f64) {
    let n = f0.len();
    for i in 1..n {
        if f0[i] == 0.0 {
            continue;
        }
        if f0[i - 1] == 0.0 {
            continue;
        }
        let ratio = f0[i] / f0[i - 1];
        if ratio < 1.0 - allowed_range || ratio > 1.0 + allowed_range {
            f0[i] = 0.0;
        }
    }
}

/// 後処理 Step 2: 短い有声区間の除去
fn fix_step2(f0: &mut [f64], _allowed_range: f64) {
    let n = f0.len();
    if n < 3 {
        return;
    }

    // 孤立した有声フレーム（前後が無声）を除去
    for i in 1..n - 1 {
        if f0[i] != 0.0 && f0[i - 1] == 0.0 && f0[i + 1] == 0.0 {
            f0[i] = 0.0;
        }
    }
    // 先頭・末尾のチェック
    if n >= 2 && f0[0] != 0.0 && f0[1] == 0.0 {
        f0[0] = 0.0;
    }
    if n >= 2 && f0[n - 1] != 0.0 && f0[n - 2] == 0.0 {
        f0[n - 1] = 0.0;
    }
}

/// 後処理 Step 3: 後方→前方に候補修正
fn fix_step3(
    f0: &mut [f64],
    f0_candidates: &[Vec<f64>],
    f0_scores: &[Vec<f64>],
    number_of_bands: usize,
    allowed_range: f64,
) {
    let n = f0.len();
    for i in (1..n).rev() {
        if f0[i] == 0.0 && f0[i - 1] != 0.0 {
            let best = select_best_f0(
                f0[i - 1],
                f0_candidates,
                f0_scores,
                i,
                number_of_bands,
                allowed_range,
            );
            if best != 0.0 {
                f0[i] = best;
            }
        }
    }
}

/// 後処理 Step 4: 前方→後方に候補修正
fn fix_step4(
    f0: &mut [f64],
    f0_candidates: &[Vec<f64>],
    f0_scores: &[Vec<f64>],
    number_of_bands: usize,
    allowed_range: f64,
) {
    let n = f0.len();
    for i in 0..n - 1 {
        if f0[i] == 0.0 && f0[i + 1] != 0.0 {
            let best = select_best_f0(
                f0[i + 1],
                f0_candidates,
                f0_scores,
                i,
                number_of_bands,
                allowed_range,
            );
            if best != 0.0 {
                f0[i] = best;
            }
        }
    }
}

/// 指定インデックスの候補から、基準 F0 に近い最良候補を選択
fn select_best_f0(
    reference_f0: f64,
    f0_candidates: &[Vec<f64>],
    f0_scores: &[Vec<f64>],
    index: usize,
    number_of_bands: usize,
    allowed_range: f64,
) -> f64 {
    let mut best_f0 = 0.0;
    let mut best_score = MAXIMUM_VALUE;

    for j in 0..number_of_bands {
        if index >= f0_candidates[j].len() {
            continue;
        }
        let candidate = f0_candidates[j][index];
        if candidate == 0.0 {
            continue;
        }
        let ratio = candidate / reference_f0;
        if ratio >= 1.0 - allowed_range && ratio <= 1.0 + allowed_range
            && f0_scores[j][index] < best_score {
                best_score = f0_scores[j][index];
                best_f0 = candidate;
            }
    }

    best_f0
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

        let option = DioOption::new();
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
