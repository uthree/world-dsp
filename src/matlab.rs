use ndarray::Array1;

/// MATLAB 互換丸め
pub fn matlab_round(x: f64) -> i32 {
    if x > 0.0 {
        (x + 0.5) as i32
    } else {
        (x - 0.5) as i32
    }
}

/// 差分（MATLAB の diff 相当）
pub fn diff(x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; x.len() - 1];
    for i in 0..x.len() - 1 {
        y[i] = x[i + 1] - x[i];
    }
    y
}

/// FFT シフト（左右半分を入れ替え）
pub fn fftshift(x: &[f64], y: &mut [f64]) {
    let half = x.len() / 2;
    for i in 0..half {
        y[i] = x[i + half];
        y[i + half] = x[i];
    }
}

/// histc — ヒストグラムビンのインデックス（1-based）
pub fn histc(x: &[f64], edges: &[f64], index: &mut [i32]) {
    let mut count: usize = 1;
    let mut i = 0;
    while i < edges.len() {
        index[i] = 1;
        if edges[i] >= x[0] {
            break;
        }
        i += 1;
    }
    while i < edges.len() {
        if edges[i] < x[count] {
            index[i] = count as i32;
            i += 1;
        } else {
            index[i] = count as i32;
            count += 1;
        }
        if count == x.len() {
            break;
        }
    }
    let count = count as i32 - 1;
    i += 1;
    while i < edges.len() {
        index[i] = count;
        i += 1;
    }
}

/// 線形補間（MATLAB の interp1 相当）
pub fn interp1(x: &[f64], y: &[f64], xi: &[f64], yi: &mut [f64]) {
    let x_length = x.len();
    let xi_length = xi.len();
    let mut h = vec![0.0; x_length - 1];
    let mut k = vec![0i32; xi_length];

    for i in 0..x_length - 1 {
        h[i] = x[i + 1] - x[i];
    }

    histc(x, xi, &mut k);

    for i in 0..xi_length {
        let ki = k[i] as usize - 1;
        let s = (xi[i] - x[ki]) / h[ki];
        yi[i] = y[ki] + s * (y[ki + 1] - y[ki]);
    }
}

/// 等間隔データ用高速線形補間（MATLAB の interp1Q 相当）
pub fn interp1q(x: f64, shift: f64, y: &[f64], xi: &[f64], yi: &mut [f64]) {
    let x_length = y.len();
    let xi_length = xi.len();
    let delta_x = shift;
    let mut xi_base = vec![0usize; xi_length];
    let mut xi_fraction = vec![0.0; xi_length];

    for i in 0..xi_length {
        let val = (xi[i] - x) / delta_x;
        xi_base[i] = val as usize;
        xi_fraction[i] = val - xi_base[i] as f64;
    }

    let mut delta_y = vec![0.0; x_length];
    for i in 0..x_length - 1 {
        delta_y[i] = y[i + 1] - y[i];
    }
    delta_y[x_length - 1] = 0.0;

    for i in 0..xi_length {
        yi[i] = y[xi_base[i]] + delta_y[xi_base[i]] * xi_fraction[i];
    }
}

/// デシメート用 IIR フィルタ
fn filter_for_decimate(x: &[f64], r: i32, y: &mut [f64]) {
    let (a, b) = match r {
        11 => (
            [2.450743295230728, -2.06794904601978, 0.595_747_744_383_321],
            [0.0026822508007163792, 0.008_046_752_402_149_138],
        ),
        12 => (
            [2.4981398605924205, -2.1368928194784025, 0.621_875_138_162_214_8],
            [0.002_109_727_590_470_9, 0.006_329_182_771_412_7],
        ),
        10 => (
            [2.3936475118069387, -1.9873904075111861, 0.5658879979027055],
            [0.0034818622251927556, 0.010445586675578267],
        ),
        9 => (
            [2.323_600_349_175_958, -1.8921545617463598, 0.531_489_281_337_290_7],
            [0.004_633_116_404_138_937, 0.013899349212416812],
        ),
        8 => (
            [2.2357462340187593, -1.7780899984041358, 0.491_525_553_659_686_9],
            [0.006_352_276_340_711_199, 0.019056829022133598],
        ),
        7 => (
            [2.1225239019534703, -1.6395144861046302, 0.44469707800587366],
            [0.009_036_688_268_160_842, 0.027110064804482525],
        ),
        6 => (
            [1.971_535_274_951_214, -1.4686795689225347, 0.3893908434965701],
            [0.013469181309343825, 0.040407543928031475],
        ),
        5 => (
            [1.7610939654280557, -1.2554914843859768, 0.3237186507788215],
            [0.021334858522387423, 0.06400457556716227],
        ),
        4 => (
            [
                1.4499664446880227,
                -0.989_434_970_809_505_8,
                0.24578252340690215,
            ],
            [0.036_710_750_339_322_61, 0.11013225101796784],
        ),
        3 => (
            [
                0.950_393_789_832_374_2,
                -0.674_291_467_415_267_9,
                0.15412211621346475,
            ],
            [0.071_221_945_171_178_64, 0.213_665_835_513_535_9],
        ),
        2 => (
            [
                0.041_156_734_567_757_19,
                -0.42599112459189636,
                0.041037215479961225,
            ],
            [0.16797464681802227, 0.503_923_940_454_066_7],
        ),
        _ => ([0.0, 0.0, 0.0], [0.0, 0.0]),
    };

    let mut w = [0.0_f64; 3];
    for i in 0..x.len() {
        let wt = x[i] + a[0] * w[0] + a[1] * w[1] + a[2] * w[2];
        y[i] = b[0] * wt + b[1] * w[0] + b[1] * w[1] + b[0] * w[2];
        w[2] = w[1];
        w[1] = w[0];
        w[0] = wt;
    }
}

/// ダウンサンプリング（MATLAB の decimate 相当）
pub fn decimate(x: &[f64], r: i32) -> Vec<f64> {
    let x_length = x.len();
    let n_fact = 9;
    let total_len = x_length + n_fact * 2;
    let mut tmp1 = vec![0.0; total_len];
    let mut tmp2 = vec![0.0; total_len];

    for i in 0..n_fact {
        tmp1[i] = 2.0 * x[0] - x[n_fact - i];
    }
    for i in n_fact..n_fact + x_length {
        tmp1[i] = x[i - n_fact];
    }
    for i in (n_fact + x_length)..total_len {
        tmp1[i] = 2.0 * x[x_length - 1] - x[x_length - 2 - (i - (n_fact + x_length))];
    }

    filter_for_decimate(&tmp1, r, &mut tmp2);
    for i in 0..total_len {
        tmp1[i] = tmp2[total_len - i - 1];
    }
    filter_for_decimate(&tmp1, r, &mut tmp2);
    for i in 0..total_len {
        tmp1[i] = tmp2[total_len - i - 1];
    }

    let r = r as usize;
    let nout = (x_length - 1) / r + 1;
    let nbeg = x_length + r - r * nout;

    let mut y = vec![0.0; nout];
    let mut count = 0;
    let mut i = nbeg;
    while i < x_length + n_fact && count < nout {
        y[count] = tmp1[i + n_fact - 1];
        count += 1;
        i += r;
    }
    y
}

/// FFT ベース畳み込み
pub fn fast_fftfilt(x: &[f64], h: &[f64], fft_size: usize) -> Array1<f64> {
    use num_complex::Complex64;
    use rustfft::FftPlanner;

    let mut planner = FftPlanner::new();
    let fft_forward = planner.plan_fft_forward(fft_size);
    let fft_inverse = planner.plan_fft_inverse(fft_size);

    // x の FFT
    let mut x_buf: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); fft_size];
    for (i, &val) in x.iter().enumerate() {
        x_buf[i] = Complex64::new(val, 0.0);
    }
    fft_forward.process(&mut x_buf);

    // h の FFT
    let mut h_buf: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); fft_size];
    for (i, &val) in h.iter().enumerate() {
        h_buf[i] = Complex64::new(val, 0.0);
    }
    fft_forward.process(&mut h_buf);

    // 乗算
    let mut product: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); fft_size];
    for i in 0..fft_size {
        product[i] = x_buf[i] * h_buf[i];
    }

    // IFFT + 正規化 (rustfft は非正規化なので 1/N が必要)
    fft_inverse.process(&mut product);
    let inv_n = 1.0 / fft_size as f64;

    Array1::from_vec(product.iter().map(|c| c.re * inv_n).collect())
}

/// 標準偏差（MATLAB 互換, N-1 で正規化）
pub fn matlab_std(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let avg: f64 = x.iter().sum::<f64>() / n;
    let s: f64 = x.iter().map(|&v| (v - avg).powi(2)).sum::<f64>() / (n - 1.0);
    s.sqrt()
}
