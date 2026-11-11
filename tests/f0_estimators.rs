//! Behaviour shared by all F0 estimators (DIO, Harvest, YIN).

mod common;

use common::*;
use world_dsp::constant::get_samples_for_dio;
use world_dsp::dio::dio;
use world_dsp::harvest::harvest;
use world_dsp::yin::yin;
use world_dsp::{Dio, F0Estimator, Harvest, Yin, stonemask};

fn estimators(fs: i32) -> Vec<(&'static str, Box<dyn F0Estimator>)> {
    vec![
        ("dio", Box::new(Dio::new(fs))),
        ("harvest", Box::new(Harvest::new(fs))),
        ("yin", Box::new(Yin::new(fs))),
    ]
}

#[test]
fn frame_layout_matches_frame_period() {
    let fs = 16000;
    let x = harmonic_signal(fs, 200.0, 0.4);
    for (name, est) in estimators(fs) {
        let (tp, f0) = est.estimate(&x);
        let expected = get_samples_for_dio(fs, x.len(), est.frame_period());
        assert_eq!(f0.len(), expected, "{name}: frame count");
        assert_eq!(tp.len(), expected, "{name}: temporal positions");
        assert_eq!(est.fs(), fs, "{name}: fs");
        for (i, &t) in tp.iter().enumerate() {
            let want = i as f64 * est.frame_period() / 1000.0;
            assert!((t - want).abs() < 1e-12, "{name}: tp[{i}] = {t}");
        }
    }
}

#[test]
fn harmonic_signal_pitch_is_detected() {
    let fs = 16000;
    for f0_true in [120.0, 220.0, 400.0] {
        let x = harmonic_signal(fs, f0_true, 0.5);
        for (name, est) in estimators(fs) {
            let (_tp, f0) = est.estimate(&x);
            let ratio = voiced_ratio_center(&f0);
            assert!(ratio > 0.8, "{name} @ {f0_true} Hz: voiced ratio {ratio}");
            let median = median_voiced_center(&f0).unwrap();
            assert_relative_close(median, f0_true, 0.03, &format!("{name} @ {f0_true} Hz"));
        }
    }
}

#[test]
fn output_is_finite_and_within_search_range() {
    let fs = 16000;
    let x = harmonic_signal(fs, 180.0, 0.4);
    let floor = 71.0;
    let ceil = 800.0;
    for (name, est) in estimators(fs) {
        let (_tp, f0) = est.estimate(&x);
        for (i, &v) in f0.iter().enumerate() {
            assert!(v.is_finite(), "{name}: f0[{i}] = {v}");
            assert!(
                v == 0.0 || (floor * 0.9..=ceil * 1.1).contains(&v),
                "{name}: f0[{i}] = {v} outside the search range"
            );
        }
    }
}

#[test]
fn silence_is_unvoiced() {
    let fs = 16000;
    let x = vec![0.0; 8000];
    for (name, est) in estimators(fs) {
        let (_tp, f0) = est.estimate(&x);
        assert!(
            f0.iter().all(|&v| v == 0.0),
            "{name}: silence must be unvoiced"
        );
    }
}

#[test]
fn search_range_is_respected() {
    // A 400 Hz tone with the ceiling set below it must not be reported as 400 Hz.
    let fs = 16000;
    let x = harmonic_signal(fs, 400.0, 0.4);

    let mut dio_opt = Dio::new(fs);
    dio_opt.f0_ceil = 300.0;
    let mut harvest_opt = Harvest::new(fs);
    harvest_opt.f0_ceil = 300.0;
    let mut yin_opt = Yin::new(fs);
    yin_opt.f0_ceil = 300.0;

    let ests: Vec<(&str, Box<dyn F0Estimator>)> = vec![
        ("dio", Box::new(dio_opt)),
        ("harvest", Box::new(harvest_opt)),
        ("yin", Box::new(yin_opt)),
    ];
    for (name, est) in ests {
        let (_tp, f0) = est.estimate(&x);
        for &v in &f0 {
            assert!(v < 330.0, "{name}: {v} Hz exceeds the 300 Hz ceiling");
        }
    }
}

#[test]
fn custom_frame_period() {
    let fs = 16000;
    let x = harmonic_signal(fs, 200.0, 0.4);

    let mut dio_opt = Dio::new(fs);
    dio_opt.frame_period = 10.0;
    let (tp, f0) = dio(&x, fs, &dio_opt);
    assert_eq!(f0.len(), get_samples_for_dio(fs, x.len(), 10.0));
    assert!((tp[1] - 0.01).abs() < 1e-12);

    let mut harvest_opt = Harvest::new(fs);
    harvest_opt.frame_period = 10.0;
    let (_tp, f0) = harvest(&x, fs, &harvest_opt);
    assert_eq!(f0.len(), get_samples_for_dio(fs, x.len(), 10.0));

    let mut yin_opt = Yin::new(fs);
    yin_opt.frame_period = 10.0;
    let (_tp, f0) = yin(&x, fs, &yin_opt);
    assert_eq!(f0.len(), get_samples_for_dio(fs, x.len(), 10.0));
}

#[test]
fn trait_matches_free_functions() {
    let fs = 16000;
    let x = harmonic_signal(fs, 250.0, 0.3);

    let opt = Dio::new(fs);
    assert_eq!(opt.estimate(&x), dio(&x, fs, &opt));
    let opt = Harvest::new(fs);
    assert_eq!(opt.estimate(&x), harvest(&x, fs, &opt));
    let opt = Yin::new(fs);
    assert_eq!(opt.estimate(&x), yin(&x, fs, &opt));
}

#[test]
fn works_at_common_sample_rates() {
    for fs in [8000, 22050, 44100, 48000] {
        let x = harmonic_signal(fs, 200.0, 0.3);
        for (name, est) in estimators(fs) {
            let (_tp, f0) = est.estimate(&x);
            let median = median_voiced_center(&f0)
                .unwrap_or_else(|| panic!("{name} @ {fs} Hz: no voiced frames"));
            assert_relative_close(median, 200.0, 0.03, &format!("{name} @ {fs} Hz"));
        }
    }
}

#[test]
fn stonemask_refines_coarse_dio_estimate() {
    let fs = 16000;
    let f0_true = 210.0;
    let x = harmonic_signal(fs, f0_true, 0.5);
    let (tp, f0) = dio(&x, fs, &Dio::new(fs));
    // Make the coarse estimate 3 % off so that there is something to refine.
    let coarse: Vec<f64> = f0.iter().map(|&v| v * 1.03).collect();
    let refined = stonemask(&x, fs, &tp, &coarse);
    assert_eq!(refined.len(), f0.len());

    let n = f0.len();
    for i in n / 4..n * 3 / 4 {
        if coarse[i] > 0.0 {
            let before = (coarse[i] - f0_true).abs();
            let after = (refined[i] - f0_true).abs();
            assert!(
                after < before * 0.2,
                "frame {i}: {} Hz refined to {} Hz",
                coarse[i],
                refined[i]
            );
        } else {
            assert_eq!(refined[i], 0.0, "unvoiced frames stay unvoiced");
        }
    }
    assert_relative_close(
        median_voiced_center(&refined).unwrap(),
        f0_true,
        0.002,
        "stonemask",
    );
}
