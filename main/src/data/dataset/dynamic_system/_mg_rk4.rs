pub fn _mg_rk4(xt: f64, xtau: f64, a: f64, b: f64, n: f64, h: f64) -> f64 {
    let bh = -b * h;
    let k1 = bh * xt + a * xtau / (1.0 + xtau.powf(n));
    let k2 = 2.0 * k1 + bh * k1;
    let k3 = 2.0 * k1 + bh * k2;
    let k4 = k1 + bh * k3;
    xt + (k1 + k2 + k3 + k4) / 6.0
}
