//! # world-dsp
//!
//! A Rust implementation of the WORLD vocoder. Ported from the C++ version [WORLD](https://github.com/mmorise/World).
//!
//! ## Pipeline
//!
//! 1. **F0 estimation** — Estimate the fundamental frequency using one of [`Dio`], [`Harvest`], or [`Yin`]
//! 2. **Spectral envelope estimation** — Estimate the spectral envelope using [`CheapTrick`] (`Array2<f64>` shape: `[num_frames, fft_size/2+1]`)
//! 3. **Aperiodicity estimation** — Estimate aperiodicity indices using [`D4C`] (`Array2<f64>` shape: `[num_frames, fft_size/2+1]`)
//! 4. **Waveform synthesis** — Resynthesize the waveform from parameters using [`Synthesizer`] (`Array1<f64>` shape: `[y_length]`)
//!
//! ## Usage example
//!
//! ```no_run
//! use world_dsp::*;
//!
//! let fs = 16000;
//! let x: Vec<f64> = vec![0.0; fs as usize]; // Input waveform
//!
//! // F0 estimation
//! let estimator = Yin::new(fs);
//! let (tp, f0) = estimator.estimate(&x);
//!
//! // Spectral envelope estimation
//! let ct = CheapTrick::new(fs, 1024);
//! let spectrogram = ct.estimate(&x, &tp, &f0);
//!
//! // Aperiodicity estimation
//! let d4c = D4C::new(fs, 1024);
//! let aperiodicity = d4c.estimate(&x, &tp, &f0);
//!
//! // Waveform synthesis
//! let synth = Synthesizer::new(estimator.frame_period(), fs, 1024);
//! let y = synth.synthesize(&f0, &spectrogram, &aperiodicity);
//! ```

pub mod cheaptrick;
pub mod common;
pub mod constant;
pub mod d4c;
pub mod dio;
pub mod harvest;
pub mod matlab;
pub mod stonemask;
pub mod synthesis;
pub mod yin;

pub use constant::{
    get_fft_size_for_cheaptrick, CheapTrick, D4C, Dio, F0Estimator, Harvest, Synthesizer, Yin,
};
pub use stonemask::stonemask;
