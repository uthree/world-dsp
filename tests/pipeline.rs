//! Spectral envelope, aperiodicity and synthesis, and the whole
//! analysis/synthesis pipeline.

mod common;

use common::*;
use ndarray::Array2;
use world_dsp::cheaptrick::cheaptrick;
use world_dsp::d4c::d4c;
use world_dsp::synthesis::synthesis;
use world_dsp::{
    CheapTrick, D4C, Dio, F0Estimator, Harvest, Synthesizer, Yin, get_fft_size_for_cheaptrick,
};

const FS: i32 = 16000;

struct Analysis {
    f0: Vec<f64>,
    tp: Vec<f64>,
    sp: Array2<f64>,
    ap: Array2<f64>,
    fft_size: usize,
    frame_period: f64,
}

fn analyze(x: &[f64], fs: i32, estimator: &dyn F0Estimator) -> Analysis {
    let fft_size = get_fft_size_for_cheaptrick(fs, 71.0);
    let (tp, f0) = estimator.estimate(x);
    let sp = CheapTrick::new(fs, fft_size).estimate(x, &tp, &f0);
    let ap = D4C::new(fs, fft_size).estimate(x, &tp, &f0);
    Analysis {
        f0,
        tp,
        sp,
        ap,
        fft_size,
        frame_period: estimator.frame_period(),
    }
}

fn synthesize(a: &Analysis, fs: i32, f0: &[f64]) -> Vec<f64> {
    Synthesizer::new(a.frame_period, fs, a.fft_size)
        .synthesize(f0, &a.sp, &a.ap)
        .to_vec()
}

#[test]
fn cheaptrick_shape_and_values() {
    let x = harmonic_signal(FS, 200.0, 0.4);
    let a = analyze(&x, FS, &Harvest::new(FS));
    assert_eq!(a.sp.dim(), (a.f0.len(), a.fft_size / 2 + 1));
    assert!(a.sp.iter().all(|&v| v.is_finite() && v > 0.0));
}

#[test]
fn cheaptrick_scales_with_power() {
    let x = harmonic_signal(FS, 200.0, 0.4);
    let x2: Vec<f64> = x.iter().map(|v| v * 2.0).collect();
    let (tp, f0) = Harvest::new(FS).estimate(&x);
    let ct = CheapTrick::new(FS, 1024);
    let sp1 = ct.estimate(&x, &tp, &f0);
    let sp2 = ct.estimate(&x2, &tp, &f0);
    // CheapTrick adds a tiny random dither, which shows up in bins with very
    // little energy. Check the bins up to 2 kHz, where the harmonics are.
    let n = f0.len();
    for i in n / 4..n * 3 / 4 {
        for j in 0..=128 {
            let ratio = sp2[[i, j]] / sp1[[i, j]];
            assert!(
                (ratio - 4.0).abs() < 0.01,
                "frame {i} bin {j}: ratio {ratio}"
            );
        }
    }
}

#[test]
fn cheaptrick_follows_spectral_tilt() {
    // Harmonic amplitudes fall with 1/h, so the envelope must fall with frequency.
    let x = harmonic_signal(FS, 200.0, 0.4);
    let a = analyze(&x, FS, &Harvest::new(FS));
    let frame = a.f0.len() / 2;
    let bin = |hz: f64| (hz * a.fft_size as f64 / FS as f64) as usize;
    let low = a.sp[[frame, bin(200.0)]];
    let high = a.sp[[frame, bin(1800.0)]];
    assert!(low > high * 10.0, "low {low} high {high}");
}

#[test]
fn cheaptrick_free_function_matches_method() {
    let x = harmonic_signal(FS, 200.0, 0.3);
    let (tp, f0) = Dio::new(FS).estimate(&x);
    let ct = CheapTrick::new(FS, 1024);
    let a = ct.estimate(&x, &tp, &f0);
    let b = cheaptrick(&x, FS, &tp, &f0, &ct);
    // A tiny random dither is added internally, so compare loosely, and only
    // up to 2 kHz where the harmonics are.
    assert_eq!(a.dim(), b.dim());
    for i in 0..a.nrows() {
        for j in 0..=128 {
            let (u, v) = (a[[i, j]], b[[i, j]]);
            assert!(((u - v) / u).abs() < 1e-3, "frame {i} bin {j}: {u} vs {v}");
        }
    }
}

#[test]
fn d4c_shape_and_range() {
    let x = harmonic_signal(FS, 200.0, 0.4);
    let a = analyze(&x, FS, &Harvest::new(FS));
    assert_eq!(a.ap.dim(), (a.f0.len(), a.fft_size / 2 + 1));
    assert!(a.ap.iter().all(|&v| v.is_finite() && v > 0.0 && v <= 1.0));
}

#[test]
fn d4c_unvoiced_frames_are_fully_aperiodic() {
    let x = harmonic_signal(FS, 200.0, 0.3);
    let (tp, _f0) = Harvest::new(FS).estimate(&x);
    let f0 = vec![0.0; tp.len()];
    let ap = D4C::new(FS, 1024).estimate(&x, &tp, &f0);
    assert!(ap.iter().all(|&v| (v - 1.0).abs() < 1e-9));
}

#[test]
fn d4c_separates_periodic_and_noisy_signals() {
    let n = (FS as f64 * 0.4) as usize;
    let periodic = harmonic_signal(FS, 200.0, 0.4);
    let noise = white_noise(n, 0.3, 12345);
    let (tp, _) = Harvest::new(FS).estimate(&periodic);
    let f0 = vec![200.0; tp.len()];
    let d4c = D4C::new(FS, 1024);
    let ap_periodic = d4c.estimate(&periodic, &tp, &f0);
    let ap_noise = d4c.estimate(&noise, &tp, &f0);

    // Compare the band below 3 kHz in the middle frames.
    let bins = 3000 * 1024 / FS as usize;
    let mean_low = |ap: &Array2<f64>| {
        let rows = tp.len() / 4..tp.len() * 3 / 4;
        let count = rows.len() * bins;
        rows.map(|i| (0..bins).map(|j| ap[[i, j]]).sum::<f64>())
            .sum::<f64>()
            / count as f64
    };
    let p = mean_low(&ap_periodic);
    let q = mean_low(&ap_noise);
    assert!(p < 0.3, "periodic signal aperiodicity {p}");
    assert!(q > 0.5, "white noise aperiodicity {q}");
}

#[test]
fn d4c_free_function_matches_method_shape() {
    let x = harmonic_signal(FS, 200.0, 0.3);
    let (tp, f0) = Dio::new(FS).estimate(&x);
    let opt = D4C::new(FS, 1024);
    let ap = d4c(&x, FS, &tp, &f0, 1024, &opt);
    assert_eq!(ap.dim(), opt.estimate(&x, &tp, &f0).dim());
}

#[test]
fn synthesis_output_length() {
    let frame_period = 5.0;
    for frames in [2usize, 10, 101] {
        let f0 = vec![150.0; frames];
        let sp = Array2::from_elem((frames, 513), 1e-4);
        let ap = Array2::from_elem((frames, 513), 0.01);
        let y = synthesis(&f0, &sp, &ap, frame_period, FS, 1024);
        let expected = ((frames as f64 - 1.0) * frame_period / 1000.0 * FS as f64) as usize + 1;
        assert_eq!(y.len(), expected, "{frames} frames");
        assert!(y.iter().all(|v| v.is_finite()));
    }
}

#[test]
#[should_panic(expected = "at least two frames")]
fn synthesis_rejects_a_single_frame() {
    let sp = Array2::from_elem((1, 513), 1e-4);
    let ap = Array2::from_elem((1, 513), 0.01);
    synthesis(&[150.0], &sp, &ap, 5.0, FS, 1024);
}

#[test]
fn synthesis_of_unvoiced_frames_is_noise() {
    let frames = 80;
    let f0 = vec![0.0; frames];
    let sp = Array2::from_elem((frames, 513), 1e-4);
    let ap = Array2::from_elem((frames, 513), 1.0);
    let y = Synthesizer::new(5.0, FS, 1024).synthesize(&f0, &sp, &ap);
    let y = y.to_vec();
    assert!(y.iter().all(|v| v.is_finite()));
    assert!(rms(&y) > 0.0, "noise excitation should produce output");
    // No pitch should be found in the noise.
    let (_tp, f0_out) = Yin::new(FS).estimate(&y);
    assert!(voiced_ratio_center(&f0_out) < 0.3);
}

#[test]
fn synthesis_level_follows_spectral_envelope() {
    let frames = 80;
    let f0 = vec![150.0; frames];
    let ap = Array2::from_elem((frames, 513), 0.01);
    let synth = Synthesizer::new(5.0, FS, 1024);
    let y1 = synth.synthesize(&f0, &Array2::from_elem((frames, 513), 1e-4), &ap);
    let y2 = synth.synthesize(&f0, &Array2::from_elem((frames, 513), 4e-4), &ap);
    let ratio = rms(y2.as_slice().unwrap()) / rms(y1.as_slice().unwrap());
    assert!((ratio - 2.0).abs() < 0.1, "amplitude ratio {ratio}");
}

#[test]
fn resynthesis_preserves_pitch_and_level() {
    for f0_true in [120.0, 250.0] {
        let x = harmonic_signal(FS, f0_true, 0.5);
        for (name, est) in [
            ("dio", Box::new(Dio::new(FS)) as Box<dyn F0Estimator>),
            ("harvest", Box::new(Harvest::new(FS))),
            ("yin", Box::new(Yin::new(FS))),
        ] {
            let a = analyze(&x, FS, est.as_ref());
            let y = synthesize(&a, FS, &a.f0);
            assert!(y.iter().all(|v| v.is_finite()), "{name}");

            let (_tp, f0_out) = Harvest::new(FS).estimate(&y);
            let median = median_voiced_center(&f0_out)
                .unwrap_or_else(|| panic!("{name}: resynthesis is unvoiced"));
            assert_relative_close(
                median,
                f0_true,
                0.03,
                &format!("{name} @ {f0_true} Hz pitch"),
            );

            let n = x.len().min(y.len());
            let level = rms(&y[n / 4..n * 3 / 4]) / rms(&x[n / 4..n * 3 / 4]);
            assert!(
                (0.5..2.0).contains(&level),
                "{name} @ {f0_true} Hz: level ratio {level}"
            );
        }
    }
}

#[test]
fn pitch_shift_moves_f0() {
    let x = harmonic_signal(FS, 150.0, 0.5);
    let a = analyze(&x, FS, &Harvest::new(FS));
    for factor in [0.75, 1.5] {
        let f0: Vec<f64> = a.f0.iter().map(|v| v * factor).collect();
        let y = synthesize(&a, FS, &f0);
        let (_tp, f0_out) = Harvest::new(FS).estimate(&y);
        let median = median_voiced_center(&f0_out).unwrap();
        assert_relative_close(median, 150.0 * factor, 0.03, &format!("factor {factor}"));
    }
}

#[test]
fn analysis_of_sample_wav() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/sample.wav");
    let mut reader = hound::WavReader::open(path).expect("assets/sample.wav");
    let spec = reader.spec();
    let fs = spec.sample_rate as i32;
    let channels = spec.channels as usize;
    let samples: Vec<f64> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f64;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f64 / scale)
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap() as f64).collect(),
    };
    let x: Vec<f64> = samples.chunks(channels).map(|c| c[0]).collect();
    assert!(!x.is_empty());

    let a = analyze(&x, fs, &Harvest::new(fs));
    let voiced = a.f0.iter().filter(|&&v| v > 0.0).count();
    assert!(
        voiced > a.f0.len() / 10,
        "speech should contain voiced frames"
    );
    assert!(a.f0.iter().all(|&v| v == 0.0 || (60.0..900.0).contains(&v)));
    assert_eq!(a.tp.len(), a.f0.len());

    let y = synthesize(&a, fs, &a.f0);
    assert!(y.iter().all(|v| v.is_finite()));
    let expected = ((a.f0.len() as f64 - 1.0) * a.frame_period / 1000.0 * fs as f64) as usize + 1;
    assert_eq!(y.len(), expected);
    let level = rms(&y) / rms(&x);
    assert!((0.3..3.0).contains(&level), "level ratio {level}");
}
