/// 円周率
pub const PI: f64 = std::f64::consts::PI;
/// 2の自然対数
pub const LOG2: f64 = std::f64::consts::LN_2;
/// デフォルトの F0 下限 (Hz)
pub const DEFAULT_F0_FLOOR: f64 = 71.0;
/// デフォルトの F0 上限 (Hz)
pub const DEFAULT_F0_CEIL: f64 = 800.0;
/// デフォルトの F0 値 (Hz)。無声区間の代替値として使用
pub const DEFAULT_F0: f64 = 500.0;
/// デフォルトのフレーム周期 (ms)
pub const DEFAULT_FRAME_PERIOD: f64 = 5.0;
/// ゼロ除算防止用の最小値 (1e-12)
pub const SAFE_GUARD_MINIMUM: f64 = 0.000_000_000_001;
/// マシンイプシロン相当の微小値
pub const EPS: f64 = 0.000_000_000_000_000_222_044_604_925_031_3;
/// D4C 用のセーフガード値 (1e-6)
pub const SAFE_GUARD_D4C: f64 = 0.000_001;
/// D4C の F0 下限 (Hz)
pub const FLOOR_F0_D4C: f64 = 47.0;
/// D4C LoveTrain の VUV 判定閾値
pub const THRESHOLD: f64 = 0.85;
/// D4C の周波数解析間隔 (Hz)
pub const FREQUENCY_INTERVAL: f64 = 3000.0;
/// D4C の周波数解析上限 (Hz)
pub const UPPER_LIMIT: f64 = 15000.0;
/// 窓関数タイプ: Hanning
pub const HANNING: i32 = 1;
/// 窓関数タイプ: Blackman
pub const BLACKMAN: i32 = 2;
/// CheapTrick のデフォルト Q1 パラメータ
pub const CHEAPTRICK_Q1_DEFAULT: f64 = -0.15;
/// DIO のローカットフィルタカットオフ周波数 (Hz)
pub const CUTOFF: f64 = 50.0;
/// DIO スコアの最大値（無効候補のマーカー）
pub const MAXIMUM_VALUE: f64 = 100000.0;
/// StoneMask の F0 下限 (Hz)
pub const FLOOR_F0_STONEMASK: f64 = 40.0;

/// F0（基本周波数）推定のための共通トレイト
///
/// [`Dio`], [`Harvest`], [`Yin`] が実装する。
pub trait F0Estimator {
    /// F0 を推定する。
    ///
    /// # Arguments
    /// * `x` - 入力波形（モノラル）
    ///
    /// # Returns
    /// `(temporal_positions, f0)` のタプル。
    /// - `temporal_positions` - 各フレームの時間位置 (秒), shape: `[num_frames]`
    /// - `f0` - 各フレームの基本周波数 (Hz), shape: `[num_frames]`。無声フレームは 0.0
    fn estimate(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>);

    /// サンプリングレート (Hz)
    fn fs(&self) -> i32;

    /// フレーム周期 (ms)
    fn frame_period(&self) -> f64;
}

/// CheapTrick スペクトル包絡推定器
///
/// F0 適応窓とケプストラム平滑化によるスペクトル包絡推定を行う。
#[derive(Debug, Clone)]
pub struct CheapTrick {
    /// ケプストラム平滑化パラメータ (デフォルト: -0.15)
    pub q1: f64,
    /// F0 下限 (Hz)。`fft_size` から自動計算される
    pub f0_floor: f64,
    /// FFT サイズ（2のべき乗）
    pub fft_size: usize,
    /// サンプリングレート (Hz)
    pub fs: i32,
}

impl CheapTrick {
    /// FFT サイズを指定して構築する。`f0_floor` は `fft_size` から自動計算。
    pub fn new(fs: i32, fft_size: usize) -> Self {
        let f0_floor = get_f0_floor_for_cheaptrick(fs, fft_size);
        CheapTrick {
            q1: CHEAPTRICK_Q1_DEFAULT,
            f0_floor,
            fft_size,
            fs,
        }
    }

    /// `f0_floor` を指定して構築する。`fft_size` は自動計算。
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
///
/// 帯域非周期性 (Band Aperiodicity) を推定する。
/// 群遅延解析とエネルギー重心により各周波数帯域の非周期性を計算。
#[derive(Debug, Clone)]
pub struct D4C {
    /// LoveTrain VUV 判定閾値 (デフォルト: 0.85)
    pub threshold: f64,
    /// サンプリングレート (Hz)
    pub fs: i32,
    /// FFT サイズ（CheapTrick と同じ値を使用すること）
    pub fft_size: usize,
}

impl D4C {
    /// デフォルトパラメータで構築する。
    pub fn new(fs: i32, fft_size: usize) -> Self {
        D4C {
            threshold: THRESHOLD,
            fs,
            fft_size,
        }
    }
}

/// CheapTrick 用の FFT サイズを `f0_floor` から計算する（2のべき乗）。
///
/// `3 * fs / f0_floor` を覆う最小の 2^n を返す。
pub fn get_fft_size_for_cheaptrick(fs: i32, f0_floor: f64) -> usize {
    let log2_val = (3.0 * fs as f64 / f0_floor + 1.0).ln() / LOG2;
    (2.0_f64).powi(1 + log2_val as i32) as usize
}

/// FFT サイズから CheapTrick の F0 下限を逆算する。
pub fn get_f0_floor_for_cheaptrick(fs: i32, fft_size: usize) -> f64 {
    3.0 * fs as f64 / (fft_size as f64 - 3.0)
}

/// DIO ピッチ推定器
///
/// ゼロクロッシング解析と FFT ベースのバンドパスフィルタリングにより
/// 高速かつ安定した F0 推定を行う。
#[derive(Debug, Clone)]
pub struct Dio {
    /// F0 下限 (Hz, デフォルト: 71.0)
    pub f0_floor: f64,
    /// F0 上限 (Hz, デフォルト: 800.0)
    pub f0_ceil: f64,
    /// オクターブあたりのチャンネル数 (デフォルト: 2.0)
    pub channels_in_octave: f64,
    /// フレーム周期 (ms, デフォルト: 5.0)
    pub frame_period: f64,
    /// 高速化パラメータ (1-12, デフォルト: 1)。値が大きいほどデシメーション比が大きい
    pub speed: i32,
    /// F0 ジャンプ許容範囲 (デフォルト: 0.1)
    pub allowed_range: f64,
    /// サンプリングレート (Hz)
    pub fs: i32,
}

impl Dio {
    /// デフォルトパラメータで構築する。
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
///
/// マルチチャンネルゼロクロッシング解析と StoneMask 相当のリファインメントにより
/// 高精度な F0 推定を行う。DIO より低速だが精度が高い。
#[derive(Debug, Clone)]
pub struct Harvest {
    /// F0 下限 (Hz, デフォルト: 71.0)
    pub f0_floor: f64,
    /// F0 上限 (Hz, デフォルト: 800.0)
    pub f0_ceil: f64,
    /// フレーム周期 (ms, デフォルト: 5.0)
    pub frame_period: f64,
    /// サンプリングレート (Hz)
    pub fs: i32,
}

impl Harvest {
    /// デフォルトパラメータで構築する。
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
///
/// de Cheveigné & Kawahara (2002) の YIN アルゴリズムによる F0 推定。
/// DIO/Harvest より高速で、リアルタイム用途に適する。
#[derive(Debug, Clone)]
pub struct Yin {
    /// F0 下限 (Hz, デフォルト: 71.0)
    pub f0_floor: f64,
    /// F0 上限 (Hz, デフォルト: 800.0)
    pub f0_ceil: f64,
    /// フレーム周期 (ms, デフォルト: 5.0)
    pub frame_period: f64,
    /// 閾値 (デフォルト: 0.1)。低いほど無声判定が厳しい
    pub threshold: f64,
    /// サンプリングレート (Hz)
    pub fs: i32,
}

impl Yin {
    /// デフォルトパラメータで構築する。
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
///
/// F0・スペクトル包絡・非周期性指標から波形を合成する（ボコーダー）。
#[derive(Debug, Clone)]
pub struct Synthesizer {
    /// フレーム周期 (ms)
    pub frame_period: f64,
    /// サンプリングレート (Hz)
    pub fs: i32,
    /// FFT サイズ（CheapTrick / D4C と同じ値を使用すること）
    pub fft_size: usize,
}

impl Synthesizer {
    /// パラメータを指定して構築する。
    pub fn new(frame_period: f64, fs: i32, fft_size: usize) -> Self {
        Synthesizer {
            frame_period,
            fs,
            fft_size,
        }
    }
}

/// DIO/Harvest/YIN 用のフレーム数を計算する。
///
/// `floor(1000 * x_length / fs / frame_period) + 1` を返す。
pub fn get_samples_for_dio(fs: i32, x_length: usize, frame_period: f64) -> usize {
    (1000.0 * x_length as f64 / fs as f64 / frame_period) as usize + 1
}

/// `sample` 以上の最小の 2 のべき乗を返す。
pub fn get_suitable_fft_size(sample: usize) -> usize {
    (2.0_f64).powi(((sample as f64).ln() / LOG2) as i32 + 1) as usize
}
