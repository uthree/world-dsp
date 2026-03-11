/// WORLD vocoder を使った音声加工デモ
///
/// 使い方:
///   cargo run --example voice_modify -- <input.wav> <output.wav> <mode> [param] [--f0 METHOD]
///
/// モード:
///   pitch <semitones>   — ピッチシフト（半音単位、例: +3, -5）
///   robot               — ロボットボイス（F0 を固定値にする）
///   whisper             — ウィスパーボイス（F0 を 0 にして非周期成分のみ）
///   gender <ratio>      — 性別変換（スペクトル包絡シフト、例: 0.8=低く, 1.2=高く）
///
/// F0 推定手法（--f0 オプション、デフォルト: yin）:
///   yin      — YIN（高速・リアルタイム向き）
///   harvest  — Harvest（高精度だが低速）
///   dio      — DIO（高速・安定）
///
/// 例:
///   cargo run --example voice_modify -- input.wav output.wav pitch 5
///   cargo run --example voice_modify -- input.wav output.wav pitch 5 --f0 harvest
///   cargo run --example voice_modify -- input.wav output.wav robot --f0 dio
use std::env;
use std::process;

use world_dsp::{CheapTrick, D4C, Dio, F0Estimator, Harvest, Synthesizer, Yin};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "Usage: {} <input.wav> <output.wav> <mode> [param] [--f0 yin|harvest|dio]",
            args[0]
        );
        eprintln!("Modes: pitch <semitones> | robot | whisper | gender <ratio>");
        process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let mode = &args[3];

    // --f0 オプションの解析
    let f0_method = args
        .iter()
        .position(|s| s == "--f0")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("yin");

    // WAV 読み込み
    let (samples, fs) = read_wav(input_path);
    eprintln!(
        "Input: {} samples, {}Hz, {:.2}s",
        samples.len(),
        fs,
        samples.len() as f64 / fs as f64
    );

    // F0 推定
    let estimator: Box<dyn F0Estimator> = match f0_method {
        "yin" => {
            eprintln!("Estimating F0 with YIN...");
            Box::new(Yin::new(fs))
        }
        "harvest" => {
            eprintln!("Estimating F0 with Harvest...");
            Box::new(Harvest::new(fs))
        }
        "dio" => {
            eprintln!("Estimating F0 with DIO...");
            Box::new(Dio::new(fs))
        }
        _ => {
            eprintln!("Unknown F0 method: {}", f0_method);
            eprintln!("Available: yin, harvest, dio");
            process::exit(1);
        }
    };
    let (temporal_positions, f0) = estimator.estimate(&samples);

    // スペクトル包絡推定（CheapTrick）
    eprintln!("Estimating spectral envelope with CheapTrick...");
    let fft_size = 2048;
    let ct = CheapTrick::new(fs, fft_size);
    let spectrogram = ct.estimate(&samples, &temporal_positions, &f0);

    // 非周期性推定（D4C）
    eprintln!("Estimating aperiodicity with D4C...");
    let d4c = D4C::new(fs, fft_size);
    let aperiodicity = d4c.estimate(&samples, &temporal_positions, &f0);

    // パラメータ加工
    let mut f0_modified = f0.clone();
    let mut spectrogram_modified = spectrogram.clone();

    match mode.as_str() {
        "pitch" => {
            let semitones: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
                eprintln!("pitch mode requires semitone value (e.g., pitch 5)");
                process::exit(1);
            });
            let ratio = (2.0_f64).powf(semitones / 12.0);
            eprintln!(
                "Pitch shift: {:.1} semitones (ratio: {:.3})",
                semitones, ratio
            );
            for v in f0_modified.iter_mut() {
                if *v > 0.0 {
                    *v *= ratio;
                }
            }
        }
        "robot" => {
            eprintln!("Robot voice: fixing F0 to 150Hz");
            for v in f0_modified.iter_mut() {
                if *v > 0.0 {
                    *v = 150.0;
                }
            }
        }
        "whisper" => {
            eprintln!("Whisper voice: removing all F0");
            for v in f0_modified.iter_mut() {
                *v = 0.0;
            }
        }
        "gender" => {
            let ratio: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
                eprintln!("gender mode requires ratio (e.g., gender 0.8)");
                process::exit(1);
            });
            eprintln!("Gender shift: spectral ratio {:.2}", ratio);
            let spec_len = fft_size / 2 + 1;
            let n_frames = f0_modified.len();

            // F0 をシフト
            for v in f0_modified.iter_mut() {
                if *v > 0.0 {
                    *v *= ratio;
                }
            }

            // スペクトル包絡をリサンプル（周波数軸を伸縮）
            for i in 0..n_frames {
                let mut new_spec = vec![0.0; spec_len];
                for j in 0..spec_len {
                    let src = j as f64 / ratio;
                    let src_idx = src.floor() as usize;
                    let frac = src - src_idx as f64;
                    if src_idx + 1 < spec_len {
                        new_spec[j] = spectrogram_modified[[i, src_idx]] * (1.0 - frac)
                            + spectrogram_modified[[i, src_idx + 1]] * frac;
                    } else {
                        // 範囲外は末端値で埋める（0 だと ln(0)=-inf で合成が壊れる）
                        new_spec[j] = spectrogram_modified[[i, spec_len - 1]];
                    }
                }
                for j in 0..spec_len {
                    spectrogram_modified[[i, j]] = new_spec[j];
                }
            }
        }
        _ => {
            eprintln!("Unknown mode: {}", mode);
            eprintln!("Available: pitch, robot, whisper, gender");
            process::exit(1);
        }
    }

    // 合成
    eprintln!("Synthesizing...");
    let synth = Synthesizer::new(estimator.frame_period(), fs, fft_size);
    let y = synth.synthesize(&f0_modified, &spectrogram_modified, &aperiodicity);

    // WAV 書き出し
    let output_samples: Vec<f64> = y.to_vec();
    write_wav(output_path, &output_samples, fs);
    eprintln!(
        "Output: {} samples, {:.2}s -> {}",
        output_samples.len(),
        output_samples.len() as f64 / fs as f64,
        output_path
    );
}

fn read_wav(path: &str) -> (Vec<f64>, i32) {
    let reader = hound::WavReader::open(path).unwrap_or_else(|e| {
        eprintln!("Failed to open {}: {}", path, e);
        process::exit(1);
    });

    let spec = reader.spec();
    let fs = spec.sample_rate as i32;
    let bits = spec.bits_per_sample;
    let sample_format = spec.sample_format;

    eprintln!(
        "WAV: {}ch, {}Hz, {}bit, {:?}",
        spec.channels, fs, bits, sample_format
    );

    let samples: Vec<f64> = match sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (bits - 1)) as f64;
            reader
                .into_samples::<i32>()
                .step_by(spec.channels as usize) // モノラル化（先頭チャンネルのみ）
                .map(|s| s.unwrap() as f64 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .step_by(spec.channels as usize)
            .map(|s| s.unwrap() as f64)
            .collect(),
    };

    (samples, fs)
}

fn write_wav(path: &str, samples: &[f64], fs: i32) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: fs as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec).unwrap_or_else(|e| {
        eprintln!("Failed to create {}: {}", path, e);
        process::exit(1);
    });

    let max_val = i16::MAX as f64;
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        writer.write_sample((clamped * max_val) as i16).unwrap();
    }
    writer.finalize().unwrap();
}
