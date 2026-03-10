pub const PI: f64 = std::f64::consts::PI;
pub const LOG2: f64 = std::f64::consts::LN_2;
pub const DEFAULT_F0_FLOOR: f64 = 71.0;
pub const DEFAULT_F0_CEIL: f64 = 800.0;
pub const DEFAULT_F0: f64 = 500.0;
pub const DEFAULT_FRAME_PERIOD: f64 = 5.0;
pub const SAFE_GUARD_MINIMUM: f64 = 0.000_000_000_001; // 1e-12
pub const EPS: f64 = 0.000_000_000_000_1; // 1e-13
pub const SAFE_GUARD_D4C: f64 = 0.000_000_000_1; // 1e-10
pub const FLOOR_F0_D4C: f64 = 47.0;
pub const THRESHOLD: f64 = 0.85;
pub const FREQUENCY_INTERVAL: f64 = 3000.0;
pub const UPPER_LIMIT: f64 = 15000.0;
pub const HANNING: i32 = 1;
pub const BLACKMAN: i32 = 2;
pub const CHEAPTRICK_Q1_DEFAULT: f64 = -0.15;

/// CheapTrick のオプション構造体
#[derive(Debug, Clone)]
pub struct CheapTrickOption {
    pub q1: f64,
    pub f0_floor: f64,
    pub fft_size: usize,
}

impl CheapTrickOption {
    pub fn new(fs: i32) -> Self {
        let f0_floor = DEFAULT_F0_FLOOR;
        let fft_size = get_fft_size_for_cheaptrick(fs, f0_floor);
        CheapTrickOption {
            q1: CHEAPTRICK_Q1_DEFAULT,
            f0_floor,
            fft_size,
        }
    }
}

/// D4C のオプション構造体
#[derive(Debug, Clone)]
pub struct D4COption {
    pub threshold: f64,
}

impl D4COption {
    pub fn new() -> Self {
        D4COption {
            threshold: THRESHOLD,
        }
    }
}

impl Default for D4COption {
    fn default() -> Self {
        Self::new()
    }
}

/// CheapTrick 用の FFT サイズを計算
pub fn get_fft_size_for_cheaptrick(fs: i32, f0_floor: f64) -> usize {
    (2.0_f64).powf(1.0 + (3.0 * fs as f64 / f0_floor + 1.0).ln() / LOG2).floor() as usize
}

/// FFT サイズから CheapTrick の F0 下限を逆算
pub fn get_f0_floor_for_cheaptrick(fs: i32, fft_size: usize) -> f64 {
    3.0 * fs as f64 / (fft_size as f64 - 3.0)
}

/// 適切な FFT サイズ（2のべき乗）を返す
pub fn get_suitable_fft_size(sample: usize) -> usize {
    (2.0_f64).powi(((sample as f64).ln() / LOG2) as i32 + 1) as usize
}
