/// Pi
pub const PI: f64 = std::f64::consts::PI;
/// Natural logarithm of 2
pub const LOG2: f64 = std::f64::consts::LN_2;
/// Default F0 lower bound (Hz)
pub const DEFAULT_F0_FLOOR: f64 = 71.0;
/// Default F0 upper bound (Hz)
pub const DEFAULT_F0_CEIL: f64 = 800.0;
/// Default F0 value (Hz). Used as a substitute for unvoiced segments
pub const DEFAULT_F0: f64 = 500.0;
/// Default frame period (ms)
pub const DEFAULT_FRAME_PERIOD: f64 = 5.0;
/// Division-by-zero guard minimum value (1e-12)
pub const SAFE_GUARD_MINIMUM: f64 = 0.000_000_000_001;
/// Machine epsilon equivalent small value
pub const EPS: f64 = 0.000_000_000_000_000_222_044_604_925_031_3;
/// Safeguard value for D4C (1e-6)
pub const SAFE_GUARD_D4C: f64 = 0.000_001;
/// D4C F0 lower bound (Hz)
pub const FLOOR_F0_D4C: f64 = 47.0;
/// D4C LoveTrain VUV decision threshold
pub const THRESHOLD: f64 = 0.85;
/// D4C frequency analysis interval (Hz)
pub const FREQUENCY_INTERVAL: f64 = 3000.0;
/// D4C frequency analysis upper bound (Hz)
pub const UPPER_LIMIT: f64 = 15000.0;
/// Window function type: Hanning
pub const HANNING: i32 = 1;
/// Window function type: Blackman
pub const BLACKMAN: i32 = 2;
/// CheapTrick default Q1 parameter
pub const CHEAPTRICK_Q1_DEFAULT: f64 = -0.15;
/// DIO low-cut filter cutoff frequency (Hz)
pub const CUTOFF: f64 = 50.0;
/// DIO maximum score value (invalid candidate marker)
pub const MAXIMUM_VALUE: f64 = 100000.0;
/// StoneMask F0 lower bound (Hz)
pub const FLOOR_F0_STONEMASK: f64 = 40.0;

/// Common trait for F0 (fundamental frequency) estimation
///
/// Implemented by [`Dio`], [`Harvest`], [`Yin`].
pub trait F0Estimator {
    /// Estimate F0.
    ///
    /// # Arguments
    /// * `x` - Input waveform (mono)
    ///
    /// # Returns
    /// A tuple `(temporal_positions, f0)`.
    /// - `temporal_positions` - Time position of each frame (seconds), shape: `[num_frames]`
    /// - `f0` - Fundamental frequency of each frame (Hz), shape: `[num_frames]`. Unvoiced frames are 0.0
    fn estimate(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>);

    /// Sampling rate (Hz)
    fn fs(&self) -> i32;

    /// Frame period (ms)
    fn frame_period(&self) -> f64;
}

/// CheapTrick spectral envelope estimator
///
/// Estimates the spectral envelope using F0-adaptive windowing and cepstral smoothing.
#[derive(Debug, Clone)]
pub struct CheapTrick {
    /// Cepstral smoothing parameter (default: -0.15)
    pub q1: f64,
    /// F0 lower bound (Hz). Automatically computed from `fft_size`
    pub f0_floor: f64,
    /// FFT size (power of two)
    pub fft_size: usize,
    /// Sampling rate (Hz)
    pub fs: i32,
}

impl CheapTrick {
    /// Construct with a specified FFT size. `f0_floor` is automatically computed from `fft_size`.
    pub fn new(fs: i32, fft_size: usize) -> Self {
        let f0_floor = get_f0_floor_for_cheaptrick(fs, fft_size);
        CheapTrick {
            q1: CHEAPTRICK_Q1_DEFAULT,
            f0_floor,
            fft_size,
            fs,
        }
    }

    /// Construct with a specified `f0_floor`. `fft_size` is automatically computed.
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

/// D4C aperiodicity estimator
///
/// Estimates band aperiodicity.
/// Computes aperiodicity for each frequency band using group delay analysis and energy centroid.
#[derive(Debug, Clone)]
pub struct D4C {
    /// LoveTrain VUV decision threshold (default: 0.85)
    pub threshold: f64,
    /// Sampling rate (Hz)
    pub fs: i32,
    /// FFT size (must match the value used by CheapTrick)
    pub fft_size: usize,
}

impl D4C {
    /// Construct with default parameters.
    pub fn new(fs: i32, fft_size: usize) -> Self {
        D4C {
            threshold: THRESHOLD,
            fs,
            fft_size,
        }
    }
}

/// Compute the FFT size for CheapTrick from `f0_floor` (power of two).
///
/// Returns the smallest 2^n that covers `3 * fs / f0_floor`.
pub fn get_fft_size_for_cheaptrick(fs: i32, f0_floor: f64) -> usize {
    let log2_val = (3.0 * fs as f64 / f0_floor + 1.0).ln() / LOG2;
    (2.0_f64).powi(1 + log2_val as i32) as usize
}

/// Compute the CheapTrick F0 lower bound from the FFT size.
pub fn get_f0_floor_for_cheaptrick(fs: i32, fft_size: usize) -> f64 {
    3.0 * fs as f64 / (fft_size as f64 - 3.0)
}

/// DIO pitch estimator
///
/// Performs fast and stable F0 estimation using zero-crossing analysis
/// and FFT-based bandpass filtering.
#[derive(Debug, Clone)]
pub struct Dio {
    /// F0 lower bound (Hz, default: 71.0)
    pub f0_floor: f64,
    /// F0 upper bound (Hz, default: 800.0)
    pub f0_ceil: f64,
    /// Channels per octave (default: 2.0)
    pub channels_in_octave: f64,
    /// Frame period (ms, default: 5.0)
    pub frame_period: f64,
    /// Speed parameter (1-12, default: 1). Higher values increase the decimation ratio
    pub speed: i32,
    /// Allowed F0 jump range (default: 0.1)
    pub allowed_range: f64,
    /// Sampling rate (Hz)
    pub fs: i32,
}

impl Dio {
    /// Construct with default parameters.
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

/// Harvest pitch estimator
///
/// Performs high-accuracy F0 estimation using multi-channel zero-crossing analysis
/// and StoneMask-equivalent refinement. Slower than DIO but more accurate.
#[derive(Debug, Clone)]
pub struct Harvest {
    /// F0 lower bound (Hz, default: 71.0)
    pub f0_floor: f64,
    /// F0 upper bound (Hz, default: 800.0)
    pub f0_ceil: f64,
    /// Frame period (ms, default: 5.0)
    pub frame_period: f64,
    /// Sampling rate (Hz)
    pub fs: i32,
}

impl Harvest {
    /// Construct with default parameters.
    pub fn new(fs: i32) -> Self {
        Harvest {
            f0_floor: DEFAULT_F0_FLOOR,
            f0_ceil: DEFAULT_F0_CEIL,
            frame_period: DEFAULT_FRAME_PERIOD,
            fs,
        }
    }
}

/// YIN pitch estimator
///
/// F0 estimation using the YIN algorithm by de Cheveigné & Kawahara (2002).
/// Faster than DIO/Harvest and suitable for real-time applications.
#[derive(Debug, Clone)]
pub struct Yin {
    /// F0 lower bound (Hz, default: 71.0)
    pub f0_floor: f64,
    /// F0 upper bound (Hz, default: 800.0)
    pub f0_ceil: f64,
    /// Frame period (ms, default: 5.0)
    pub frame_period: f64,
    /// Threshold (default: 0.1). Lower values make unvoiced detection stricter
    pub threshold: f64,
    /// Sampling rate (Hz)
    pub fs: i32,
}

impl Yin {
    /// Construct with default parameters.
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

/// Waveform synthesizer
///
/// Synthesizes a waveform from F0, spectral envelope, and aperiodicity (vocoder).
#[derive(Debug, Clone)]
pub struct Synthesizer {
    /// Frame period (ms)
    pub frame_period: f64,
    /// Sampling rate (Hz)
    pub fs: i32,
    /// FFT size (must match the value used by CheapTrick / D4C)
    pub fft_size: usize,
}

impl Synthesizer {
    /// Construct with the specified parameters.
    pub fn new(frame_period: f64, fs: i32, fft_size: usize) -> Self {
        Synthesizer {
            frame_period,
            fs,
            fft_size,
        }
    }
}

/// Compute the number of frames for DIO/Harvest/YIN.
///
/// Returns `floor(1000 * x_length / fs / frame_period) + 1`.
pub fn get_samples_for_dio(fs: i32, x_length: usize, frame_period: f64) -> usize {
    (1000.0 * x_length as f64 / fs as f64 / frame_period) as usize + 1
}

/// Return the smallest power of two greater than or equal to `sample`.
pub fn get_suitable_fft_size(sample: usize) -> usize {
    (2.0_f64).powi(((sample as f64).ln() / LOG2) as i32 + 1) as usize
}
