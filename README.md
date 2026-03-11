# world-dsp
pure rust implementation of [WORLD vocoder](https://github.com/mmorise/World)

## usage
```rust
let fft_size = 2048; // window size
let fs = 48000; // sample rate

// yin
let yin = yin::new(fs);
let (temporal_positions, f0) = yin.estimate(&samples)

// cheaptrick
let ct = CheapTrick::new(fs, fft_size);
let spectrogram = ct.estimate(&samples, &temporal_positions, &f0);

// d4c
let d4c = D4C::new(fs, fft_size);
let aperiodicity = d4c.estimate(&samples, &temporal_positions, &f0);

// synthesize
let synth = Synthesizer::new(estimator.frame_period(), fs, fft_size);
let resynthesized_waveform =  synth.synthesize(&f0, &spectrogram, &aperiodicity);
```
