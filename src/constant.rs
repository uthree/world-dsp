pub const PI: f64 = std::f64::consts::PI;
pub const LOG2: f64 = std::f64::consts::LN_2;
pub const DEFAULT_F0_FLOOR: f64 = 71.0;
pub const DEFAULT_F0_CEIL: f64 = 800.0;
pub const DEFAULT_F0: f64 = 500.0;
pub const DEFAULT_FRAME_PERIOD: f64 = 5.0;
pub const SAFE_GUARD_MINIMUM: f64 = 0.000_000_000_001; // 1e-12
pub const EPS: f64 = 0.000_000_000_000_000_222_044_604_925_031_3; // f64::EPSILON
pub const SAFE_GUARD_D4C: f64 = 0.000_001; // 1e-6
pub const FLOOR_F0_D4C: f64 = 47.0;
pub const THRESHOLD: f64 = 0.85;
pub const FREQUENCY_INTERVAL: f64 = 3000.0;
pub const UPPER_LIMIT: f64 = 15000.0;
pub const HANNING: i32 = 1;
pub const BLACKMAN: i32 = 2;
pub const CHEAPTRICK_Q1_DEFAULT: f64 = -0.15;
pub const CUTOFF: f64 = 50.0;
pub const MAXIMUM_VALUE: f64 = 100000.0;
pub const FLOOR_F0_STONEMASK: f64 = 40.0;

/// F0（基本周波数）推定のための共通トレイト
pub trait F0Estimator {
    /// F0 を推定する。返り値は (temporal_positions, f0)。
    fn estimate(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>);

    /// サンプリングレート
    fn fs(&self) -> i32;

    /// フレーム周期 (ms)
    fn frame_period(&self) -> f64;
}

/// CheapTrick スペクトル包絡推定器
#[derive(Debug, Clone)]
pub struct CheapTrick {
    pub q1: f64,
    pub f0_floor: f64,
    pub fft_size: usize,
    pub fs: i32,
}

impl CheapTrick {
    pub fn new(fs: i32, fft_size: usize) -> Self {
        let f0_floor = get_f0_floor_for_cheaptrick(fs, fft_size);
        CheapTrick {
            q1: CHEAPTRICK_Q1_DEFAULT,
            f0_floor,
            fft_size,
            fs,
        }
    }

    /// fs と f0_floor から fft_size を自動計算して構築する
    pub fn from_f0_floor(fs: i32, f0_floor: f64) -> Self {
        let fft_size = get_fft_size_for_cheaptrick(fs, f0_floor);
        CheapTrick {
            q1: CHEAPTRICK_Q1_DEFAULT,
            f0_floor,
            fft_size,
            fs,
        }
    }
}

/// D4C 非周期性推定器
#[derive(Debug, Clone)]
pub struct D4C {
    pub threshold: f64,
    pub fs: i32,
    pub fft_size: usize,
}

impl D4C {
    pub fn new(fs: i32, fft_size: usize) -> Self {
        D4C {
            threshold: THRESHOLD,
            fs,
            fft_size,
        }
    }
}

/// CheapTrick 用の FFT サイズを計算（2のべき乗）
pub fn get_fft_size_for_cheaptrick(fs: i32, f0_floor: f64) -> usize {
    let log2_val = (3.0 * fs as f64 / f0_floor + 1.0).ln() / LOG2;
    (2.0_f64).powi(1 + log2_val as i32) as usize
}

/// FFT サイズから CheapTrick の F0 下限を逆算
pub fn get_f0_floor_for_cheaptrick(fs: i32, fft_size: usize) -> f64 {
    3.0 * fs as f64 / (fft_size as f64 - 3.0)
}

/// DIO ピッチ推定器
#[derive(Debug, Clone)]
pub struct Dio {
    pub f0_floor: f64,
    pub f0_ceil: f64,
    pub channels_in_octave: f64,
    pub frame_period: f64,
    pub speed: i32,
    pub allowed_range: f64,
    pub fs: i32,
}

impl Dio {
    pub fn new(fs: i32) -> Self {
        Dio {
            f0_floor: DEFAULT_F0_FLOOR,
            f0_ceil: DEFAULT_F0_CEIL,
            channels_in_octave: 2.0,
            frame_period: DEFAULT_FRAME_PERIOD,
            speed: 1,
            allowed_range: 0.1,
            fs,
        }
    }
}

/// Harvest ピッチ推定器
#[derive(Debug, Clone)]
pub struct Harvest {
    pub f0_floor: f64,
    pub f0_ceil: f64,
    pub frame_period: f64,
    pub fs: i32,
}

impl Harvest {
    pub fn new(fs: i32) -> Self {
        Harvest {
            f0_floor: DEFAULT_F0_FLOOR,
            f0_ceil: DEFAULT_F0_CEIL,
            frame_period: DEFAULT_FRAME_PERIOD,
            fs,
        }
    }
}

/// YIN ピッチ推定器
#[derive(Debug, Clone)]
pub struct Yin {
    pub f0_floor: f64,
    pub f0_ceil: f64,
    pub frame_period: f64,
    pub threshold: f64,
    pub fs: i32,
}

impl Yin {
    pub fn new(fs: i32) -> Self {
        Yin {
            f0_floor: DEFAULT_F0_FLOOR,
            f0_ceil: DEFAULT_F0_CEIL,
            frame_period: DEFAULT_FRAME_PERIOD,
            threshold: 0.1,
            fs,
        }
    }
}

/// 波形合成器
#[derive(Debug, Clone)]
pub struct Synthesizer {
    pub frame_period: f64,
    pub fs: i32,
    pub fft_size: usize,
}

impl Synthesizer {
    pub fn new(frame_period: f64, fs: i32, fft_size: usize) -> Self {
        Synthesizer {
            frame_period,
            fs,
            fft_size,
        }
    }
}

/// DIO 用のフレーム数を計算
pub fn get_samples_for_dio(fs: i32, x_length: usize, frame_period: f64) -> usize {
    (1000.0 * x_length as f64 / fs as f64 / frame_period) as usize + 1
}

/// 適切な FFT サイズ（2のべき乗）を返す
pub fn get_suitable_fft_size(sample: usize) -> usize {
    (2.0_f64).powi(((sample as f64).ln() / LOG2) as i32 + 1) as usize
}
