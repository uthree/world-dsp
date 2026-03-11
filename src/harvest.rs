use num_complex::Complex64;
use rayon::prelude::*;

use crate::common::{forward_real_fft, inverse_real_fft, nuttall_window};
use crate::constant::*;
use crate::matlab::{decimate, diff, interp1, matlab_round};
use crate::stonemask::get_refined_f0;

/// Harvest ピッチ推定
///
/// DIO より精密な F0 推定を行う。内部で StoneMask 相当のリファインメントを適用する。
/// 返り値は (temporal_positions, f0) のタプル。
pub fn harvest(x: &[f64], fs: i32, option: &Harvest) -> (Vec<f64>, Vec<f64>) {
    let f0_length = get_samples_for_dio(fs, x.len(), option.frame_period);
    let mut temporal_positions = vec![0.0; f0_length];
    for i in 0..f0_length {
        temporal_positions[i] = i as f64 * option.frame_period / 1000.0;
    }

    let adjusted_f0_floor = option.f0_floor * 0.9;
    let adjusted_f0_ceil = option.f0_ceil * 1.1;

    // 使用するデシメーション比率一覧
    let decimation_ratios = get_decimation_ratios(adjusted_f0_floor, fs);
    let n_ratios = decimation_ratios.len();

    // 各デシメーション比率でゼロクロッシング候補を計算
    let channels_in_octave = 40.0;
    let number_of_channels =
        1 + ((adjusted_f0_ceil / adjusted_f0_floor).ln() / LOG2 * channels_in_octave) as usize;

    let boundary_f0_list: Vec<f64> = (0..number_of_channels)
        .map(|i| adjusted_f0_floor * (2.0_f64).powf(i as f64 / channels_in_octave))
        .collect();

    // 各チャンネルのF0候補
    let mut f0_candidates = vec![vec![0.0; f0_length]; number_of_channels];

    for ratio_idx in 0..n_ratios {
        let decimation_ratio = decimation_ratios[ratio_idx];
        let actual_fs = fs / decimation_ratio;

        let y = if decimation_ratio != 1 {
            decimate(x, decimation_ratio)
        } else {
            x.to_vec()
        };
        let y_length = y.len();

        // C++: fft_size はフィルタ長を考慮して計算
        let max_filter_half = matlab_round(actual_fs as f64 / adjusted_f0_floor * 2.0) as usize;
        let fft_size =
            get_suitable_fft_size(y_length + max_filter_half * 2 + 1);

        let y_spectrum = get_spectrum_for_estimation(&y, y_length, actual_fs, fft_size);

        // このデシメーション比率に対応する帯域を並列処理
        let min_freq = actual_fs as f64 / 2.0 / 12.0;
        let max_freq = actual_fs as f64 / 2.0 * 0.9;

        let channel_results: Vec<(usize, Vec<f64>)> = (0..number_of_channels)
            .into_par_iter()
            .filter_map(|ch| {
                let bf0 = boundary_f0_list[ch];
                if bf0 < min_freq || bf0 > max_freq {
                    return None;
                }
                let filtered_signal =
                    get_filtered_signal(bf0, fft_size, actual_fs as f64, &y_spectrum, y_length);
                let zero_crossings = get_four_zero_crossing_intervals(&filtered_signal, actual_fs);

                let mut candidates = vec![0.0; f0_length];
                get_f0_candidate_contour(
                    &zero_crossings,
                    bf0,
                    actual_fs as f64,
                    &temporal_positions,
                    f0_length,
                    &mut candidates,
                );
                Some((ch, candidates))
            })
            .collect();

        for (ch, candidates) in channel_results {
            f0_candidates[ch] = candidates;
        }
    }

    // 最良候補選択: 最も多くのチャンネルが一致する F0 を選ぶ
    let mut raw_f0 = get_best_f0_from_candidates(&f0_candidates, f0_length, number_of_channels);

    // F0 軌跡の修正
    fix_f0_contour(&mut raw_f0, f0_length);

    // 急変付近の修正
    override_f0_near_steps(&mut raw_f0, f0_length);

    // StoneMask でリファインメント（並列）
    let mut refined_f0: Vec<f64> = (0..f0_length)
        .into_par_iter()
        .map(|i| {
            if raw_f0[i] <= 0.0 {
                return 0.0;
            }
            let refined = get_refined_f0(x, fs, temporal_positions[i], raw_f0[i]);
            if refined <= 0.0 { raw_f0[i] } else { refined }
        })
        .collect();

    // 範囲外の F0 を除去
    for i in 0..f0_length {
        if refined_f0[i] < option.f0_floor || refined_f0[i] > option.f0_ceil {
            refined_f0[i] = 0.0;
        }
    }

    (temporal_positions, refined_f0)
}

/// サンプリング周波数と F0 下限から使用するデシメーション比率を決定
fn get_decimation_ratios(f0_floor: f64, fs: i32) -> Vec<i32> {
    // C++ 版 WORLD と同様の比率候補
    let candidates = [1, 2, 3, 4, 5, 6, 8, 10, 12];
    let mut ratios = Vec::new();

    for &r in &candidates {
        let actual_fs = fs as f64 / r as f64;
        // Nyquist 周波数が f0_floor * 6 以上であること
        if actual_fs / 2.0 >= f0_floor * 6.0 {
            ratios.push(r);
        }
    }

    if ratios.is_empty() {
        ratios.push(1);
    }
    ratios
}

/// スペクトルを取得（C++ WORLD Harvest の GetWaveformAndSpectrum に準拠）
/// Harvest は DIO と異なりローカットフィルタを適用しない。DC 除去のみ。
fn get_spectrum_for_estimation(
    y: &[f64],
    y_length: usize,
    _actual_fs: i32,
    fft_size: usize,
) -> Vec<Complex64> {
    let mut y_padded = vec![0.0; fft_size];
    for i in 0..y_length.min(fft_size) {
        y_padded[i] = y[i];
    }

    // DC 成分除去
    let mean_y: f64 = y_padded[..y_length].iter().sum::<f64>() / y_length as f64;
    for i in 0..y_length {
        y_padded[i] -= mean_y;
    }

    forward_real_fft(&y_padded, fft_size)
}


/// バンドパスフィルタリング（C++ WORLD Harvest の GetFilteredSignal に準拠）
/// DIO のローパスフィルタと異なり、Nuttall 窓にコサイン変調を適用してバンドパスを構成する
fn get_filtered_signal(
    boundary_f0: f64,
    fft_size: usize,
    fs: f64,
    y_spectrum: &[Complex64],
    y_length: usize,
) -> Vec<f64> {
    // C++: filter_length_half = matlab_round(fs / boundary_f0 * 2.0)
    let filter_length_half = matlab_round(fs / boundary_f0 * 2.0) as usize;
    let filter_length = filter_length_half * 2 + 1;

    let nuttall = nuttall_window(filter_length);

    let mut band_pass_filter = vec![0.0; fft_size];
    // Nuttall 窓 × コサイン変調でバンドパスフィルタを構成
    for i in 0..filter_length {
        let k = i as i64 - filter_length_half as i64;
        band_pass_filter[i] =
            nuttall[i] * (2.0 * PI * boundary_f0 * k as f64 / fs).cos();
    }

    let filter_spectrum = forward_real_fft(&band_pass_filter, fft_size);

    let half = fft_size / 2 + 1;
    // 複素数乗算（C++ 準拠）
    let mut result_spectrum = vec![Complex64::new(0.0, 0.0); fft_size];
    for i in 0..half {
        let tmp = y_spectrum[i].re * filter_spectrum[i].re
            - y_spectrum[i].im * filter_spectrum[i].im;
        result_spectrum[i].im = y_spectrum[i].re * filter_spectrum[i].im
            + y_spectrum[i].im * filter_spectrum[i].re;
        result_spectrum[i].re = tmp;
    }
    for i in half..fft_size {
        result_spectrum[i].re = result_spectrum[fft_size - i].re;
        result_spectrum[i].im = result_spectrum[fft_size - i].im;
    }

    let output = inverse_real_fft(&result_spectrum[..half], fft_size);

    // C++: index_bias = filter_length_half + 1
    let delay = filter_length_half + 1;
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

fn get_four_zero_crossing_intervals(filtered_signal: &[f64], actual_fs: i32) -> ZeroCrossings {
    let (neg_locs, neg_intervals) = zero_crossing_engine(filtered_signal, actual_fs);

    let neg_signal: Vec<f64> = filtered_signal.iter().map(|&v| -v).collect();
    let (pos_locs, pos_intervals) = zero_crossing_engine(&neg_signal, actual_fs);

    let d = diff(filtered_signal);
    let (peak_locs, peak_intervals) = zero_crossing_engine(&d, actual_fs);

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

fn zero_crossing_engine(x: &[f64], fs: i32) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let mut locations = Vec::new();

    for i in 0..n - 1 {
        if x[i] * x[i + 1] < 0.0 && x[i] < 0.0 {
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

/// C++ WORLD Harvest の GetF0CandidateContour + GetF0CandidateContourSub に準拠
fn get_f0_candidate_contour(
    zero_crossings: &ZeroCrossings,
    boundary_f0: f64,
    _actual_fs: f64,
    temporal_positions: &[f64],
    f0_length: usize,
    f0_candidate: &mut [f64],
) {
    for i in 0..f0_length {
        f0_candidate[i] = 0.0;
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

    // C++ GetF0CandidateContourSub: 常に4値の平均、boundary_f0 * 0.9~1.1 でフィルタ
    let upper = boundary_f0 * 1.1;
    let lower = boundary_f0 * 0.9;
    for i in 0..f0_length {
        let mean = (interp_neg[i] + interp_pos[i] + interp_peak[i] + interp_dip[i]) / 4.0;

        if mean > upper || mean < lower {
            f0_candidate[i] = 0.0;
        } else {
            f0_candidate[i] = mean;
        }
    }
}

fn interp1_safe(x: &[f64], y: &[f64], xi: &[f64], yi: &mut [f64]) {
    if x.len() < 2 || y.len() < 2 {
        for v in yi.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let x_min = x[0];
    let x_max = x[x.len() - 1];

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

/// 複数チャンネルの F0 候補からフレームごとに最良の F0 を選択
/// 非ゼロ候補の中央値を使用（C++ Harvest の DetectOfficialF0Candidates 相当の簡略版）
fn get_best_f0_from_candidates(
    f0_candidates: &[Vec<f64>],
    f0_length: usize,
    number_of_channels: usize,
) -> Vec<f64> {
    let mut best_f0 = vec![0.0; f0_length];

    for i in 0..f0_length {
        let mut voiced: Vec<f64> = Vec::new();
        for j in 0..number_of_channels {
            if f0_candidates[j][i] > 0.0 {
                voiced.push(f0_candidates[j][i]);
            }
        }
        if voiced.is_empty() {
            continue;
        }
        voiced.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // 中央値を使用
        best_f0[i] = voiced[voiced.len() / 2];
    }

    best_f0
}

/// F0 軌跡の修正
fn fix_f0_contour(f0: &mut [f64], f0_length: usize) {
    if f0_length < 3 {
        return;
    }

    let allowed_range = 0.1;

    // Step 1: 急激なジャンプ除去
    for i in 1..f0_length {
        if f0[i] == 0.0 || f0[i - 1] == 0.0 {
            continue;
        }
        let ratio = f0[i] / f0[i - 1];
        if ratio < 1.0 - allowed_range || ratio > 1.0 + allowed_range {
            f0[i] = 0.0;
        }
    }

    // Step 2: 孤立した有声フレーム除去
    for i in 1..f0_length - 1 {
        if f0[i] != 0.0 && f0[i - 1] == 0.0 && f0[i + 1] == 0.0 {
            f0[i] = 0.0;
        }
    }
    if f0[0] != 0.0 && f0_length > 1 && f0[1] == 0.0 {
        f0[0] = 0.0;
    }
    if f0[f0_length - 1] != 0.0 && f0_length > 1 && f0[f0_length - 2] == 0.0 {
        f0[f0_length - 1] = 0.0;
    }
}

/// 急変付近の F0 修正
fn override_f0_near_steps(f0: &mut [f64], f0_length: usize) {
    if f0_length < 3 {
        return;
    }

    // 有声→無声、無声→有声の遷移付近で F0 を平滑化
    for i in 1..f0_length - 1 {
        if f0[i] == 0.0 {
            continue;
        }
        // 前後に無声フレームがある場合、近傍の有声フレームと比較
        if f0[i - 1] != 0.0 && f0[i + 1] != 0.0 {
            let mean = (f0[i - 1] + f0[i + 1]) / 2.0;
            let ratio = f0[i] / mean;
            if !(0.5..=2.0).contains(&ratio) {
                f0[i] = 0.0;
            }
        }
    }
}

impl F0Estimator for Harvest {
    fn estimate(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        harvest(x, self.fs, self)
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
    fn test_harvest_sine_440hz() {
        let fs = 16000;
        let duration = 0.5;
        let f0_true = 440.0;
        let n_samples = (fs as f64 * duration) as usize;
        let x: Vec<f64> = (0..n_samples)
            .map(|i| (2.0 * PI * f0_true * i as f64 / fs as f64).sin())
            .collect();

        let option = Harvest::new(fs);
        let (temporal_positions, f0) = harvest(&x, fs, &option);

        assert!(!f0.is_empty());
        assert_eq!(temporal_positions.len(), f0.len());

        // 中央付近のフレームで F0 が概ね 440Hz であることを確認
        let mid = f0.len() / 2;
        let start = if mid >= 2 { mid - 2 } else { 0 };
        let end = (mid + 3).min(f0.len());
        let voiced_f0: Vec<f64> = f0[start..end]
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
