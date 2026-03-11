//! # world-dsp
//!
//! WORLD ボコーダーの Rust 実装。C++ 版 [WORLD](https://github.com/mmorise/World) を移植したもの。
//!
//! ## パイプライン
//!
//! 1. **F0 推定** — [`Dio`], [`Harvest`], [`Yin`] のいずれかで基本周波数を推定
//! 2. **スペクトル包絡推定** — [`CheapTrick`] でスペクトル包絡を推定（`Array2<f64>` shape: `[num_frames, fft_size/2+1]`）
//! 3. **非周期性推定** — [`D4C`] で非周期性指標を推定（`Array2<f64>` shape: `[num_frames, fft_size/2+1]`）
//! 4. **波形合成** — [`Synthesizer`] でパラメータから波形を再合成（`Array1<f64>` shape: `[y_length]`）
//!
//! ## 使用例
//!
//! ```no_run
//! use world_dsp::*;
//!
//! let fs = 16000;
//! let x: Vec<f64> = vec![0.0; fs as usize]; // 入力波形
//!
//! // F0 推定
//! let estimator = Yin::new(fs);
//! let (tp, f0) = estimator.estimate(&x);
//!
//! // スペクトル包絡推定
//! let ct = CheapTrick::new(fs, 1024);
//! let spectrogram = ct.estimate(&x, &tp, &f0);
//!
//! // 非周期性推定
//! let d4c = D4C::new(fs, 1024);
//! let aperiodicity = d4c.estimate(&x, &tp, &f0);
//!
//! // 波形合成
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
