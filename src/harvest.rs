use num_complex::Complex64;
use rayon::prelude::*;

use crate::common::{forward_real_fft, inverse_real_fft, nuttall_window};
use crate::constant::*;
use crate::matlab::{decimate, diff, interp1, matlab_round};

/// Harvest pitch estimation.
///
/// Performs high-accuracy F0 estimation following C++ WORLD's Harvest.
/// Internally estimates at 1ms frame period and resamples to the specified frame period.
/// Multi-channel zero-crossing -> VUV decision -> StoneMask-equivalent refinement
/// -> post-processing -> Butterworth smoothing pipeline.
///
/// # Arguments
/// * `x` - Input waveform (mono)
/// * `fs` - Sampling frequency (Hz)
/// * `option` - Harvest parameters
///
/// # Returns
/// `(temporal_positions, f0)` tuple.
/// - `temporal_positions` - Temporal position of each frame (seconds), length `num_frames`
/// - `f0` - Fundamental frequency of each frame (Hz), length `num_frames`. Unvoiced frames are 0.0
pub fn harvest(x: &[f64], fs: i32, option: &Harvest) -> (Vec<f64>, Vec<f64>) {
    let channels_in_octave = 40.0;
    let target_fs = 8000.0;
    let dimension_ratio = matlab_round(fs as f64 / target_fs).max(1);

    if option.frame_period == 1.0 {
        return harvest_general_body(
            x,
            fs,
            1,
            option.f0_floor,
            option.f0_ceil,
            channels_in_octave,
            dimension_ratio,
        );
    }

    // Internally estimate at 1ms frame period
    let basic_frame_period = 1;
    let (basic_tp, basic_f0) = harvest_general_body(
        x,
        fs,
        basic_frame_period,
        option.f0_floor,
        option.f0_ceil,
        channels_in_octave,
        dimension_ratio,
    );

    // Resample to specified frame period
    let f0_length = get_samples_for_dio(fs, x.len(), option.frame_period);
    let mut temporal_positions = vec![0.0; f0_length];
    let mut f0 = vec![0.0; f0_length];
    for i in 0..f0_length {
        temporal_positions[i] = i as f64 * option.frame_period / 1000.0;
        let idx = matlab_round(temporal_positions[i] * 1000.0) as usize;
        f0[i] = basic_f0[idx.min(basic_f0.len() - 1)];
    }

    let _ = basic_tp; // used internally
    (temporal_positions, f0)
}

/// Harvest main body (follows C++ HarvestGeneralBody)
fn harvest_general_body(
    x: &[f64],
    fs: i32,
    frame_period: i32,
    f0_floor: f64,
    f0_ceil: f64,
    channels_in_octave: f64,
    speed: i32,
) -> (Vec<f64>, Vec<f64>) {
    let adjusted_f0_floor = f0_floor * 0.9;
    let adjusted_f0_ceil = f0_ceil * 1.1;

    let number_of_channels =
        1 + ((adjusted_f0_ceil / adjusted_f0_floor).ln() / LOG2 * channels_in_octave) as usize;

    // C++: boundary_f0_list[i] = f0_floor * 2^((i+1)/channels_in_octave)
    let boundary_f0_list: Vec<f64> = (0..number_of_channels)
        .map(|i| adjusted_f0_floor * (2.0_f64).powf((i + 1) as f64 / channels_in_octave))
        .collect();

    // Single decimation ratio (follows C++)
    let decimation_ratio = speed.max(1).min(12);
    let y_length = (x.len() as f64 / decimation_ratio as f64).ceil() as usize;
    let actual_fs = fs as f64 / decimation_ratio as f64;

    let fft_size = get_suitable_fft_size(
        y_length + 5 + 2 * (2.0 * actual_fs / boundary_f0_list[0]) as usize,
    );

    // Decimation + DC removal + FFT
    let y = if decimation_ratio != 1 {
        decimate(x, decimation_ratio)
    } else {
        x.to_vec()
    };
    let y_len = y.len();
    let y_spectrum = get_waveform_and_spectrum(&y, y_len, fft_size);

    let f0_length = get_samples_for_dio(fs, x.len(), frame_period as f64);
    let mut temporal_positions = vec![0.0; f0_length];
    for i in 0..f0_length {
        temporal_positions[i] = i as f64 * frame_period as f64 / 1000.0;
    }

    // 1. Get raw F0 candidates (per channel)
    let raw_f0_candidates: Vec<Vec<f64>> = (0..number_of_channels)
        .into_par_iter()
        .map(|ch| {
            let bf0 = boundary_f0_list[ch];
            let filtered_signal =
                get_filtered_signal(bf0, fft_size, actual_fs, &y_spectrum, y_len);
            let zero_crossings =
                get_four_zero_crossing_intervals(&filtered_signal, actual_fs as i32);
            let mut candidates = vec![0.0; f0_length];
            get_f0_candidate_contour(
                &zero_crossings,
                bf0,
                f0_floor,
                f0_ceil,
                &temporal_positions,
                f0_length,
                &mut candidates,
            );
            candidates
        })
        .collect();

    // 2. DetectOfficialF0Candidates: cross-channel VUV decision
    let overlap_parameter = 7;
    let max_candidates =
        (matlab_round(number_of_channels as f64 / 10.0) * overlap_parameter).max(1) as usize;

    // f0_candidates[frame][candidate] layout
    let mut f0_candidates = vec![vec![0.0; max_candidates]; f0_length];
    detect_official_f0_candidates(
        &raw_f0_candidates,
        number_of_channels,
        f0_length,
        max_candidates,
        &mut f0_candidates,
    );

    // 3. OverlapF0Candidates: spread candidates to neighboring frames
    let number_of_candidates_base = count_max_candidates(&f0_candidates, f0_length, max_candidates);
    let total_candidates = number_of_candidates_base * overlap_parameter as usize;
    // Expansion
    let mut expanded = vec![vec![0.0; total_candidates]; f0_length];
    for i in 0..f0_length {
        for j in 0..max_candidates.min(number_of_candidates_base) {
            expanded[i][j] = f0_candidates[i][j];
        }
    }
    overlap_f0_candidates(f0_length, number_of_candidates_base, overlap_parameter as usize, &mut expanded);

    // 4. RefineF0Candidates: refine each candidate and compute scores
    let num_cands = total_candidates.min(expanded[0].len());
    let mut f0_scores = vec![vec![0.0_f64; num_cands]; f0_length];
    refine_f0_candidates_parallel(
        &y,
        y_len,
        actual_fs,
        &temporal_positions,
        f0_length,
        num_cands,
        f0_floor,
        f0_ceil,
        &mut expanded,
        &mut f0_scores,
    );

    // 5. RemoveUnreliableCandidates
    remove_unreliable_candidates(f0_length, num_cands, &mut expanded, &mut f0_scores);

    // 6. FixF0Contour (SearchF0Base → Step1-4)
    let best_f0 = fix_f0_contour_harvest(
        &expanded,
        &f0_scores,
        f0_length,
        num_cands,
    );

    // 7. SmoothF0Contour
    let smoothed = smooth_f0_contour(&best_f0, f0_length);

    (temporal_positions, smoothed)
}

// ============================================================================
// Internal functions
// ============================================================================

/// DC removal + FFT (follows C++ GetWaveformAndSpectrum)
fn get_waveform_and_spectrum(y: &[f64], y_length: usize, fft_size: usize) -> Vec<Complex64> {
    let mut y_padded = vec![0.0; fft_size];
    for i in 0..y_length.min(fft_size) {
        y_padded[i] = y[i];
    }
    let mean_y: f64 = y_padded[..y_length].iter().sum::<f64>() / y_length as f64;
    for i in 0..y_length {
        y_padded[i] -= mean_y;
    }
    forward_real_fft(&y_padded, fft_size)
}

/// Bandpass filtering (follows C++ Harvest GetFilteredSignal)
fn get_filtered_signal(
    boundary_f0: f64,
    fft_size: usize,
    fs: f64,
    y_spectrum: &[Complex64],
    y_length: usize,
) -> Vec<f64> {
    let filter_length_half = matlab_round(fs / boundary_f0 * 2.0) as usize;
    let filter_length = filter_length_half * 2 + 1;
    let nuttall = nuttall_window(filter_length);

    let mut band_pass_filter = vec![0.0; fft_size];
    for i in 0..filter_length {
        let k = i as i64 - filter_length_half as i64;
        band_pass_filter[i] = nuttall[i] * (2.0 * PI * boundary_f0 * k as f64 / fs).cos();
    }

    let filter_spectrum = forward_real_fft(&band_pass_filter, fft_size);
    let half = fft_size / 2 + 1;
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

// ---- Zero-crossing analysis ----

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
        if x[i] > 0.0 && x[i + 1] <= 0.0 {
            let edge = (i + 1) as f64
                - x[i] / (x[i + 1] - x[i]);
            locations.push(edge);
        }
    }

    let mut intervals = Vec::new();
    let mut interval_locations = Vec::new();
    for i in 0..locations.len().saturating_sub(1) {
        intervals.push(fs as f64 / (locations[i + 1] - locations[i]));
        interval_locations.push((locations[i] + locations[i + 1]) / 2.0 / fs as f64);
    }
    (interval_locations, intervals)
}

// ---- F0 candidate computation ----

fn get_f0_candidate_contour(
    zero_crossings: &ZeroCrossings,
    boundary_f0: f64,
    f0_floor: f64,
    f0_ceil: f64,
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

    let mut interp = [
        vec![0.0; f0_length],
        vec![0.0; f0_length],
        vec![0.0; f0_length],
        vec![0.0; f0_length],
    ];
    interp1_safe(
        &zero_crossings.negative_interval_locations,
        &zero_crossings.negative_intervals,
        temporal_positions,
        &mut interp[0],
    );
    interp1_safe(
        &zero_crossings.positive_interval_locations,
        &zero_crossings.positive_intervals,
        temporal_positions,
        &mut interp[1],
    );
    interp1_safe(
        &zero_crossings.peak_interval_locations,
        &zero_crossings.peak_intervals,
        temporal_positions,
        &mut interp[2],
    );
    interp1_safe(
        &zero_crossings.dip_interval_locations,
        &zero_crossings.dip_intervals,
        temporal_positions,
        &mut interp[3],
    );

    let upper = boundary_f0 * 1.1;
    let lower = boundary_f0 * 0.9;
    for i in 0..f0_length {
        let mean = (interp[0][i] + interp[1][i] + interp[2][i] + interp[3][i]) / 4.0;
        if mean > upper || mean < lower || mean > f0_ceil || mean < f0_floor {
            f0_candidate[i] = 0.0;
        } else {
            f0_candidate[i] = mean;
        }
    }
}

fn interp1_safe(x: &[f64], y: &[f64], xi: &[f64], yi: &mut [f64]) {
    if x.len() < 2 || y.len() < 2 {
        for v in yi.iter_mut() { *v = 0.0; }
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
        for v in yi.iter_mut() { *v = 0.0; }
        return;
    }
    let mut valid_yi = vec![0.0; valid_xi.len()];
    interp1(x, y, &valid_xi, &mut valid_yi);
    for v in yi.iter_mut() { *v = 0.0; }
    for (j, &i) in valid_idx.iter().enumerate() {
        yi[i] = valid_yi[j];
    }
}

// ---- DetectOfficialF0Candidates ----

fn detect_official_f0_candidates(
    raw_f0_candidates: &[Vec<f64>],
    number_of_channels: usize,
    f0_length: usize,
    max_candidates: usize,
    f0_candidates: &mut [Vec<f64>],
) {
    for i in 0..f0_length {
        // VUV decision: whether candidate is valid in each channel
        let mut vuv = vec![0i32; number_of_channels];
        for j in 0..number_of_channels {
            vuv[j] = if raw_f0_candidates[j][i] > 0.0 { 1 } else { 0 };
        }
        // Set edges to 0
        if number_of_channels > 0 {
            vuv[0] = 0;
            vuv[number_of_channels - 1] = 0;
        }

        // Detect start/end of voiced segments
        let mut st = Vec::new();
        let mut ed = Vec::new();
        for j in 1..number_of_channels {
            let d = vuv[j] - vuv[j - 1];
            if d == 1 {
                st.push(j);
            }
            if d == -1 {
                ed.push(j);
            }
        }

        // Average F0 of each voiced segment becomes a candidate
        let n_sections = st.len().min(ed.len());
        let mut n_cands = 0;
        for s in 0..n_sections {
            if ed[s] - st[s] < 10 {
                continue;
            }
            let mut tmp_f0 = 0.0;
            for j in st[s]..ed[s] {
                tmp_f0 += raw_f0_candidates[j][i];
            }
            tmp_f0 /= (ed[s] - st[s]) as f64;
            if n_cands < max_candidates {
                f0_candidates[i][n_cands] = tmp_f0;
                n_cands += 1;
            }
        }
        for j in n_cands..max_candidates {
            f0_candidates[i][j] = 0.0;
        }
    }
}

fn count_max_candidates(f0_candidates: &[Vec<f64>], f0_length: usize, max_candidates: usize) -> usize {
    let mut max_count = 0;
    for i in 0..f0_length {
        let count = f0_candidates[i][..max_candidates]
            .iter()
            .filter(|&&v| v > 0.0)
            .count();
        max_count = max_count.max(count);
    }
    max_count.max(1)
}

// ---- OverlapF0Candidates ----

fn overlap_f0_candidates(
    f0_length: usize,
    number_of_candidates: usize,
    overlap_parameter: usize,
    f0_candidates: &mut [Vec<f64>],
) {
    let n = 3usize;
    for i in 1..=n {
        for j in 0..number_of_candidates {
            // Copy candidates from past frames
            for k in i..f0_length {
                let src_idx = j + number_of_candidates * i;
                if src_idx < f0_candidates[k].len() {
                    f0_candidates[k][src_idx] = f0_candidates[k - i][j];
                }
            }
            // Copy candidates from future frames
            for k in 0..f0_length.saturating_sub(i) {
                let src_idx = j + number_of_candidates * (i + n);
                if src_idx < f0_candidates[k].len() {
                    f0_candidates[k][src_idx] = f0_candidates[k + i][j];
                }
            }
        }
    }
    let _ = overlap_parameter;
}

// ---- RefineF0Candidates ----

fn refine_f0_candidates_parallel(
    y: &[f64],
    y_length: usize,
    fs: f64,
    temporal_positions: &[f64],
    f0_length: usize,
    num_candidates: usize,
    f0_floor: f64,
    f0_ceil: f64,
    f0_candidates: &mut [Vec<f64>],
    f0_scores: &mut [Vec<f64>],
) {
    // Process (frame, candidate) pairs for parallelization
    let results: Vec<(usize, usize, f64, f64)> = (0..f0_length)
        .into_par_iter()
        .flat_map(|i| {
            let pos = temporal_positions[i];
            (0..num_candidates)
                .filter_map(|j| {
                    let current_f0 = f0_candidates[i][j];
                    if current_f0 <= 0.0 {
                        return Some((i, j, 0.0, 0.0));
                    }
                    let (refined, score) =
                        harvest_get_refined_f0(y, y_length, fs, pos, current_f0, f0_floor, f0_ceil);
                    Some((i, j, refined, score))
                })
                .collect::<Vec<_>>()
        })
        .collect();

    for (i, j, refined, score) in results {
        f0_candidates[i][j] = refined;
        f0_scores[i][j] = score;
    }
}

/// Follows C++ Harvest's GetRefinedF0
fn harvest_get_refined_f0(
    x: &[f64],
    x_length: usize,
    fs: f64,
    current_position: f64,
    current_f0: f64,
    f0_floor: f64,
    f0_ceil: f64,
) -> (f64, f64) {
    if current_f0 <= 0.0 {
        return (0.0, 0.0);
    }

    let half_window_length = (1.5 * fs / current_f0 + 1.0) as usize;
    let window_length_in_time = (2 * half_window_length + 1) as f64 / fs;
    let base_time_length = half_window_length * 2 + 1;
    let fft_size = (2.0_f64).powi(2 + ((half_window_length as f64 * 2.0 + 1.0).ln() / LOG2) as i32) as usize;

    // base_index
    let basic_index = matlab_round((current_position + (-(half_window_length as f64)) / fs) * fs + 0.001);
    let base_index: Vec<i32> = (0..base_time_length as i32)
        .map(|i| basic_index + i)
        .collect();

    // main_window (Blackman)
    let mut main_window = vec![0.0; base_time_length];
    for i in 0..base_time_length {
        let tmp = (base_index[i] as f64 - 1.0) / fs - current_position;
        main_window[i] = 0.42
            + 0.5 * (2.0 * PI * tmp / window_length_in_time).cos()
            + 0.08 * (4.0 * PI * tmp / window_length_in_time).cos();
    }

    // diff_window
    let mut diff_window = vec![0.0; base_time_length];
    diff_window[0] = -main_window[1] / 2.0;
    for i in 1..base_time_length - 1 {
        diff_window[i] = -(main_window[i + 1] - main_window[i - 1]) / 2.0;
    }
    diff_window[base_time_length - 1] = main_window[base_time_length - 2] / 2.0;

    // Get spectra
    let mut main_input = vec![0.0; fft_size];
    let mut diff_input = vec![0.0; fft_size];
    for i in 0..base_time_length {
        let safe_idx = (base_index[i] - 1).max(0).min(x_length as i32 - 1) as usize;
        main_input[i] = x[safe_idx] * main_window[i];
        diff_input[i] = x[safe_idx] * diff_window[i];
    }

    let main_spectrum = forward_real_fft(&main_input, fft_size);
    let diff_spectrum = forward_real_fft(&diff_input, fft_size);

    let half_fft = fft_size / 2 + 1;
    let mut power_spectrum = vec![0.0; half_fft];
    let mut numerator_i = vec![0.0; half_fft];
    for j in 0..half_fft {
        numerator_i[j] = main_spectrum[j].re * diff_spectrum[j].im
            - main_spectrum[j].im * diff_spectrum[j].re;
        power_spectrum[j] =
            main_spectrum[j].re * main_spectrum[j].re + main_spectrum[j].im * main_spectrum[j].im;
    }

    let number_of_harmonics = ((fs / 2.0 / current_f0) as usize).min(6);
    let (refined_f0, score) =
        fix_f0_harvest(&power_spectrum, &numerator_i, fft_size, fs, current_f0, number_of_harmonics);

    if refined_f0 < f0_floor || refined_f0 > f0_ceil || score < 2.5 {
        (0.0, 0.0)
    } else {
        (refined_f0, score)
    }
}

/// Follows C++ Harvest's FixF0
fn fix_f0_harvest(
    power_spectrum: &[f64],
    numerator_i: &[f64],
    fft_size: usize,
    fs: f64,
    current_f0: f64,
    number_of_harmonics: usize,
) -> (f64, f64) {
    if number_of_harmonics == 0 {
        return (0.0, 0.0);
    }

    let mut amplitude_list = vec![0.0; number_of_harmonics];
    let mut inst_freq_list = vec![0.0; number_of_harmonics];

    for i in 0..number_of_harmonics {
        let index = matlab_round(current_f0 * fft_size as f64 / fs * (i + 1) as f64) as usize;
        if index >= fft_size / 2 {
            break;
        }
        if power_spectrum[index] == 0.0 {
            inst_freq_list[i] = 0.0;
        } else {
            inst_freq_list[i] = index as f64 * fs / fft_size as f64
                + numerator_i[index] / power_spectrum[index] * fs / 2.0 / PI;
        }
        amplitude_list[i] = power_spectrum[index].sqrt();
    }

    let mut denominator = 0.0;
    let mut numerator = 0.0;
    let mut score = 0.0;
    for i in 0..number_of_harmonics {
        numerator += amplitude_list[i] * inst_freq_list[i];
        denominator += amplitude_list[i] * (i + 1) as f64;
        score += ((inst_freq_list[i] / (i + 1) as f64 - current_f0) / current_f0).abs();
    }

    let refined_f0 = numerator / (denominator + SAFE_GUARD_MINIMUM);
    let final_score = 1.0 / (score / number_of_harmonics as f64 + SAFE_GUARD_MINIMUM);

    (refined_f0, final_score)
}

// ---- RemoveUnreliableCandidates ----

fn remove_unreliable_candidates(
    f0_length: usize,
    number_of_candidates: usize,
    f0_candidates: &mut [Vec<f64>],
    f0_scores: &mut [Vec<f64>],
) {
    let threshold = 0.05;
    // Create copy
    let tmp: Vec<Vec<f64>> = f0_candidates.iter().map(|v| v.clone()).collect();

    for i in 1..f0_length - 1 {
        for j in 0..number_of_candidates {
            let reference_f0 = f0_candidates[i][j];
            if reference_f0 == 0.0 {
                continue;
            }
            let error1 = select_best_f0_error(reference_f0, &tmp[i + 1], number_of_candidates);
            let error2 = select_best_f0_error(reference_f0, &tmp[i - 1], number_of_candidates);
            let min_error = error1.min(error2);
            if min_error > threshold {
                f0_candidates[i][j] = 0.0;
                f0_scores[i][j] = 0.0;
            }
        }
    }
}

fn select_best_f0_error(reference_f0: f64, candidates: &[f64], n: usize) -> f64 {
    let mut best_error = 1.0;
    for i in 0..n.min(candidates.len()) {
        if candidates[i] == 0.0 {
            continue;
        }
        let err = (reference_f0 - candidates[i]).abs() / reference_f0;
        if err < best_error {
            best_error = err;
        }
    }
    best_error
}

// ---- FixF0Contour (Harvest) ----

fn fix_f0_contour_harvest(
    f0_candidates: &[Vec<f64>],
    f0_scores: &[Vec<f64>],
    f0_length: usize,
    number_of_candidates: usize,
) -> Vec<f64> {
    // SearchF0Base: select candidate with highest score
    let mut base_f0 = vec![0.0; f0_length];
    for i in 0..f0_length {
        let mut best_score = 0.0;
        for j in 0..number_of_candidates.min(f0_candidates[i].len()) {
            if f0_scores[i][j] > best_score {
                base_f0[i] = f0_candidates[i][j];
                best_score = f0_scores[i][j];
            }
        }
    }

    // FixStep1: remove abrupt changes (extrapolation reference)
    let mut step1 = vec![0.0; f0_length];
    let allowed_range = 0.008;
    for i in 2..f0_length {
        if base_f0[i] == 0.0 {
            continue;
        }
        let reference = base_f0[i - 1] * 2.0 - base_f0[i - 2];
        let cond1 = if reference != 0.0 {
            ((base_f0[i] - reference) / reference).abs() > allowed_range
        } else {
            true
        };
        let cond2 = if base_f0[i - 1] != 0.0 {
            (base_f0[i] - base_f0[i - 1]).abs() / base_f0[i - 1] > allowed_range
        } else {
            true
        };
        step1[i] = if cond1 && cond2 { 0.0 } else { base_f0[i] };
    }

    // FixStep2: remove short voiced segments
    let voice_range_minimum = 6;
    let mut step2 = step1.clone();
    let boundaries = get_boundary_list(&step1, f0_length);
    for i in 0..boundaries.len() / 2 {
        if boundaries[i * 2 + 1] - boundaries[i * 2] < voice_range_minimum {
            for j in boundaries[i * 2]..=boundaries[i * 2 + 1] {
                if j < f0_length {
                    step2[j] = 0.0;
                }
            }
        }
    }

    // FixStep3: F0 extension + merge (simplified: select best from candidates and extend)
    let mut step3 = step2.clone();
    let boundaries = get_boundary_list(&step2, f0_length);
    let allowed_range_step3 = 0.18;
    for bi in 0..boundaries.len() / 2 {
        let ed = boundaries[bi * 2 + 1];
        let st = boundaries[bi * 2];
        // Backward extension
        let mut tmp_f0 = step3[ed];
        for k in (ed + 1)..f0_length.min(ed + 100) {
            let best = harvest_select_best_f0(tmp_f0, &f0_candidates[k], number_of_candidates, allowed_range_step3);
            if best == 0.0 { break; }
            step3[k] = best;
            tmp_f0 = best;
        }
        // Forward extension
        tmp_f0 = step3[st];
        for k in (1..=st.min(100)).rev() {
            let idx = st - (st.min(100) - k + 1);
            if idx >= f0_length { break; }
            let best = harvest_select_best_f0(tmp_f0, &f0_candidates[idx], number_of_candidates, allowed_range_step3);
            if best == 0.0 { break; }
            step3[idx] = best;
            tmp_f0 = best;
        }
    }

    // FixStep4: interpolate short unvoiced gaps
    let mut step4 = step3.clone();
    let boundaries = get_boundary_list(&step3, f0_length);
    let threshold = 9;
    for i in 0..boundaries.len() / 2 {
        if i + 1 >= boundaries.len() / 2 { break; }
        let distance = boundaries[(i + 1) * 2] as i64 - boundaries[i * 2 + 1] as i64 - 1;
        if distance >= threshold || distance <= 0 { continue; }
        let tmp0 = step3[boundaries[i * 2 + 1]] + 1.0;
        let tmp1 = step3[boundaries[(i + 1) * 2]] - 1.0;
        let coefficient = (tmp1 - tmp0) / (distance as f64 + 1.0);
        let mut count = 1.0;
        for j in (boundaries[i * 2 + 1] + 1)..boundaries[(i + 1) * 2] {
            step4[j] = tmp0 + coefficient * count;
            count += 1.0;
        }
    }

    step4
}

fn harvest_select_best_f0(reference_f0: f64, candidates: &[f64], n: usize, allowed_range: f64) -> f64 {
    let mut best_f0 = 0.0;
    let mut best_error = allowed_range;
    for i in 0..n.min(candidates.len()) {
        if candidates[i] == 0.0 { continue; }
        let err = (reference_f0 - candidates[i]).abs() / reference_f0;
        if err < best_error {
            best_error = err;
            best_f0 = candidates[i];
        }
    }
    best_f0
}

fn get_boundary_list(f0: &[f64], f0_length: usize) -> Vec<usize> {
    let mut vuv: Vec<i32> = f0[..f0_length].iter().map(|&v| if v > 0.0 { 1 } else { 0 }).collect();
    if f0_length > 0 {
        vuv[0] = 0;
        vuv[f0_length - 1] = 0;
    }
    let mut boundaries = Vec::new();
    let mut count = 0;
    for i in 1..f0_length {
        let d = vuv[i] - vuv[i - 1];
        if d != 0 {
            boundaries.push(i - (count % 2));
            count += 1;
        }
    }
    boundaries
}

// ---- SmoothF0Contour ----

fn smooth_f0_contour(f0: &[f64], f0_length: usize) -> Vec<f64> {
    let b = [0.0078202080334971724, 0.015640416066994345];
    let a = [1.7347257688092754, -0.76600660094326412];
    let lag = 300usize;
    let new_f0_length = f0_length + lag * 2;
    let mut f0_contour = vec![0.0; new_f0_length];
    for i in 0..f0_length {
        f0_contour[i + lag] = f0[i];
    }

    let boundaries = get_boundary_list(&f0_contour, new_f0_length);
    let mut smoothed_f0 = vec![0.0; f0_length];

    for bi in 0..boundaries.len() / 2 {
        let st = boundaries[bi * 2];
        let ed = boundaries[bi * 2 + 1];

        // Extend edges
        let mut section = f0_contour.clone();
        for i in 0..st { section[i] = section[st]; }
        for i in (ed + 1)..new_f0_length { section[i] = section[ed]; }

        // Forward filtering
        let mut w = [0.0_f64; 2];
        let mut tmp_x = vec![0.0; new_f0_length];
        for i in 0..new_f0_length {
            let wt = section[i] + a[0] * w[0] + a[1] * w[1];
            tmp_x[new_f0_length - i - 1] = b[0] * wt + b[1] * w[0] + b[0] * w[1];
            w[1] = w[0];
            w[0] = wt;
        }

        // Backward filtering
        w = [0.0; 2];
        let mut filtered = vec![0.0; new_f0_length];
        for i in 0..new_f0_length {
            let wt = tmp_x[i] + a[0] * w[0] + a[1] * w[1];
            filtered[new_f0_length - i - 1] = b[0] * wt + b[1] * w[0] + b[0] * w[1];
            w[1] = w[0];
            w[0] = wt;
        }

        for j in st..=ed {
            if j >= lag && j - lag < f0_length {
                smoothed_f0[j - lag] = filtered[j];
            }
        }
    }

    smoothed_f0
}

// ============================================================================
// Trait implementation
// ============================================================================

impl F0Estimator for Harvest {
    fn estimate(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        harvest(x, self.fs, self)
    }
    fn fs(&self) -> i32 { self.fs }
    fn frame_period(&self) -> f64 { self.frame_period }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harvest_sine_440hz() {
        let fs = 16000;
        let duration = 1.0;
        let f0_true = 440.0;
        let n_samples = (fs as f64 * duration) as usize;
        // Signal with harmonics (pure sine wave has too narrow channel width for detection)
        let x: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 / fs as f64;
                0.5 * (2.0 * PI * f0_true * t).sin()
                    + 0.3 * (2.0 * PI * f0_true * 2.0 * t).sin()
                    + 0.2 * (2.0 * PI * f0_true * 3.0 * t).sin()
            })
            .collect();

        let option = Harvest::new(fs);
        let (temporal_positions, f0) = harvest(&x, fs, &option);

        assert!(!f0.is_empty());
        assert_eq!(temporal_positions.len(), f0.len());

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
