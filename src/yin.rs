use crate::constant::*;

/// YIN ピッチ推定
///
/// de Cheveigné & Kawahara (2002) の YIN アルゴリズムによる F0 推定。
/// DIO/Harvest より高速で、リアルタイム用途に適する。
/// 返り値は (temporal_positions, f0) のタプル。
pub fn yin(x: &[f64], fs: i32, option: &YinOption) -> (Vec<f64>, Vec<f64>) {
    let f0_length = get_samples_for_dio(fs, x.len(), option.frame_period);
    let temporal_positions: Vec<f64> = (0..f0_length)
        .map(|i| i as f64 * option.frame_period / 1000.0)
        .collect();

    let tau_max = (fs as f64 / option.f0_floor).ceil() as usize;
    let tau_min = (fs as f64 / option.f0_ceil).floor() as usize;
    let window_size = tau_max * 2;

    let mut f0 = vec![0.0; f0_length];

    for i in 0..f0_length {
        let center = (temporal_positions[i] * fs as f64) as usize;
        if center + window_size >= x.len() {
            break;
        }

        f0[i] = estimate_f0_yin(
            x,
            center,
            fs,
            tau_min,
            tau_max,
            window_size,
            option.threshold,
        );
    }

    (temporal_positions, f0)
}

/// 単一フレームの YIN F0 推定
fn estimate_f0_yin(
    x: &[f64],
    center: usize,
    fs: i32,
    tau_min: usize,
    tau_max: usize,
    window_size: usize,
    threshold: f64,
) -> f64 {
    // Step 1-2: 差分関数 d(τ) の計算
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

    // Step 3: 累積平均正規化差分関数 d'(τ)
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

    // Step 4: 絶対閾値法 — threshold 以下で最初の局所最小値を探す
    let mut tau_estimate = 0usize;
    let mut found = false;
    for tau in tau_min..tau_max {
        if cmndf[tau] < threshold {
            // この位置から局所最小値を探す
            let mut best = tau;
            while best + 1 <= tau_max && cmndf[best + 1] < cmndf[best] {
                best += 1;
            }
            tau_estimate = best;
            found = true;
            break;
        }
    }

    // 閾値以下の谷が見つからなかった場合 → グローバル最小値にフォールバック
    if !found {
        let mut min_val = f64::MAX;
        for tau in tau_min..=tau_max {
            if cmndf[tau] < min_val {
                min_val = cmndf[tau];
                tau_estimate = tau;
            }
        }
        // グローバル最小値でも信頼できない場合は無声
        if min_val >= 0.5 {
            return 0.0;
        }
    }

    // Step 5: 放物線補間でサブサンプル精度を得る
    let tau_refined = parabolic_interpolation(&cmndf, tau_estimate);

    if tau_refined > 0.0 {
        fs as f64 / tau_refined
    } else {
        0.0
    }
}

/// 放物線補間による精密なラグ推定
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

        let option = YinOption::new();
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

        let option = YinOption::new();
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

            let option = YinOption::new();
            let (_tp, f0) = yin(&x, fs, &option);

            let mid = f0.len() / 2;
            let start = if mid >= 2 { mid - 2 } else { 0 };
            let end = (mid + 3).min(f0.len());
            let voiced: Vec<f64> = f0[start..end]
                .iter()
                .filter(|&&v| v > 0.0)
                .copied()
                .collect();
            assert!(!voiced.is_empty(), "{}Hz: should have voiced frames", f0_true);
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
