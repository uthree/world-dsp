use rayon::prelude::*;

use crate::constant::*;

/// YIN pitch estimation.
///
/// F0 estimation using the YIN algorithm by de Cheveigné & Kawahara (2002).
/// Faster than DIO/Harvest, suitable for real-time applications.
/// Frames are processed in parallel using rayon.
///
/// # Arguments
/// * `x` - Input waveform (mono)
/// * `fs` - Sampling frequency (Hz)
/// * `option` - YIN parameters
///
/// # Returns
/// A tuple `(temporal_positions, f0)`.
/// - `temporal_positions` - Temporal position of each frame (seconds), length `num_frames`
/// - `f0` - Fundamental frequency of each frame (Hz), length `num_frames`. Unvoiced frames are 0.0
pub fn yin(x: &[f64], fs: i32, option: &Yin) -> (Vec<f64>, Vec<f64>) {
    let f0_length = get_samples_for_dio(fs, x.len(), option.frame_period);
    let temporal_positions: Vec<f64> = (0..f0_length)
        .map(|i| i as f64 * option.frame_period / 1000.0)
        .collect();

    let tau_max = (fs as f64 / option.f0_floor).ceil() as usize;
    let tau_min = (fs as f64 / option.f0_ceil).floor() as usize;
    let window_size = tau_max * 2;
    let x_len = x.len();

    let f0: Vec<f64> = (0..f0_length)
        .into_par_iter()
        .map(|i| {
            let center = (temporal_positions[i] * fs as f64) as usize;
            if center + window_size >= x_len {
                return 0.0;
            }
            estimate_f0_yin(
                x,
                center,
                fs,
                tau_min,
                tau_max,
                window_size,
                option.threshold,
            )
        })
        .collect();

    (temporal_positions, f0)
}

/// YIN F0 estimation for a single frame.
///
/// Processes in order: difference function -> cumulative mean normalized -> threshold method -> parabolic interpolation.
fn estimate_f0_yin(
    x: &[f64],
    center: usize,
    fs: i32,
    tau_min: usize,
    tau_max: usize,
    window_size: usize,
    threshold: f64,
) -> f64 {
    // Step 1-2: Compute difference function d(tau)
    let mut df = vec![0.0; tau_max + 1];
    for tau in 1..=tau_max {
        let mut sum = 0.0;
        for j in 0..window_size {
            if center + j + tau >= x.len() || center + j >= x.len() {
                break;
            }
            let delta = x[center + j] - x[center + j + tau];
            sum += delta * delta;
        }
        df[tau] = sum;
    }

    // Step 3: Cumulative mean normalized difference function d'(tau)
    let mut cmndf = vec![0.0; tau_max + 1];
    cmndf[0] = 1.0;
    let mut running_sum = 0.0;
    for tau in 1..=tau_max {
        running_sum += df[tau];
        if running_sum > 0.0 {
            cmndf[tau] = df[tau] * tau as f64 / running_sum;
        } else {
            cmndf[tau] = 1.0;
        }
    }

    // Step 4: Absolute threshold method — find first local minimum below threshold
    let mut tau_estimate = 0usize;
    let mut found = false;
    for tau in tau_min..tau_max {
        if cmndf[tau] < threshold {
            let mut best = tau;
            while best < tau_max && cmndf[best + 1] < cmndf[best] {
                best += 1;
            }
            tau_estimate = best;
            found = true;
            break;
        }
    }

    // Fall back to global minimum
    if !found {
        let mut min_val = f64::MAX;
        for tau in tau_min..=tau_max {
            if cmndf[tau] < min_val {
                min_val = cmndf[tau];
                tau_estimate = tau;
            }
        }
        if min_val >= 0.5 {
            return 0.0;
        }
    }

    // Step 5: Obtain sub-sample precision via parabolic interpolation
    let tau_refined = parabolic_interpolation(&cmndf, tau_estimate);

    if tau_refined > 0.0 {
        fs as f64 / tau_refined
    } else {
        0.0
    }
}

/// Precise lag estimation via parabolic interpolation.
///
/// Fits a parabola to 3 CMNDF values at `(tau-1, tau, tau+1)` and returns the minimum point.
fn parabolic_interpolation(cmndf: &[f64], tau: usize) -> f64 {
    if tau < 1 || tau + 1 >= cmndf.len() {
        return tau as f64;
    }

    let s0 = cmndf[tau - 1];
    let s1 = cmndf[tau];
    let s2 = cmndf[tau + 1];

    let denom = 2.0 * s1 - s2 - s0;
    if denom.abs() < EPS {
        return tau as f64;
    }

    tau as f64 + (s0 - s2) / (2.0 * denom)
}

impl F0Estimator for Yin {
    fn estimate(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        yin(x, self.fs, self)
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
    fn test_yin_sine_440hz() {
        let fs = 16000;
        let duration = 0.5;
        let f0_true = 440.0;
        let n_samples = (fs as f64 * duration) as usize;
        let x: Vec<f64> = (0..n_samples)
            .map(|i| (2.0 * PI * f0_true * i as f64 / fs as f64).sin())
            .collect();

        let option = Yin::new(fs);
        let (temporal_positions, f0) = yin(&x, fs, &option);

        assert!(!f0.is_empty());
        assert_eq!(temporal_positions.len(), f0.len());

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
                (v - f0_true).abs() < 10.0,
                "F0 {} should be close to {}",
                v,
                f0_true
            );
        }
    }

    #[test]
    fn test_yin_silence() {
        let fs = 16000;
        let n_samples = 8000;
        let x = vec![0.0; n_samples];

        let option = Yin::new(fs);
        let (_tp, f0) = yin(&x, fs, &option);

        for &v in &f0 {
            assert_eq!(v, 0.0, "Silence should yield F0 = 0");
        }
    }

    #[test]
    fn test_yin_multiple_frequencies() {
        let fs = 16000;
        let duration = 0.3;
        let n_samples = (fs as f64 * duration) as usize;

        for &f0_true in &[150.0, 300.0, 500.0] {
            let x: Vec<f64> = (0..n_samples)
                .map(|i| (2.0 * PI * f0_true * i as f64 / fs as f64).sin())
                .collect();

            let option = Yin::new(fs);
            let (_tp, f0) = yin(&x, fs, &option);

            let mid = f0.len() / 2;
            let start = if mid >= 2 { mid - 2 } else { 0 };
            let end = (mid + 3).min(f0.len());
            let voiced: Vec<f64> = f0[start..end]
                .iter()
                .filter(|&&v| v > 0.0)
                .copied()
                .collect();
            assert!(
                !voiced.is_empty(),
                "{}Hz: should have voiced frames",
                f0_true
            );
            for &v in &voiced {
                assert!(
                    (v - f0_true).abs() < 10.0,
                    "{}Hz: F0 {} too far off",
                    f0_true,
                    v
                );
            }
        }
    }
}
