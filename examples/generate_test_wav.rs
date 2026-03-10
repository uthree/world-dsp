/// テスト用 WAV ファイルを生成するヘルパー
///
/// 使い方:
///   cargo run --example generate_test_wav -- <output.wav> [freq] [duration]
///
/// 例:
///   cargo run --example generate_test_wav -- test_input.wav 220 2.0
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let output = args.get(1).map(|s| s.as_str()).unwrap_or("test_input.wav");
    let freq: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(220.0);
    let duration: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2.0);

    let fs = 24000u32;
    let n_samples = (fs as f64 * duration) as usize;
    let pi2 = 2.0 * std::f64::consts::PI;

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: fs,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(output, spec).unwrap();

    for i in 0..n_samples {
        let t = i as f64 / fs as f64;
        // 倍音を含む擬似音声（のこぎり波に近い）
        let mut sample = 0.0;
        for h in 1..=6 {
            let amp = 1.0 / h as f64;
            sample += amp * (pi2 * freq * h as f64 * t).sin();
        }
        // フェードイン・フェードアウト
        let fade = if t < 0.05 {
            t / 0.05
        } else if t > duration - 0.05 {
            (duration - t) / 0.05
        } else {
            1.0
        };
        sample *= fade * 0.3;
        let val = (sample * i16::MAX as f64).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(val).unwrap();
    }
    writer.finalize().unwrap();
    eprintln!("Generated: {} ({:.0}Hz, {:.1}s, {}Hz)", output, freq, duration, fs);
}
