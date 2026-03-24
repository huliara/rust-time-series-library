pub fn _kuramoto_sivashinsky_etdrk4(
    v: &[f64],
    g: &[f64],
    e: &[f64],
    _e2: &[f64],
    q: &[f64],
    f1: &[f64],
    f2: &[f64],
    f3: &[f64],
) -> Vec<f64> {
    let n = v.len();
    let mut out = vec![0.0_f64; n];

    for i in 0..n {
        let ip1 = (i + 1) % n;
        let im1 = (i + n - 1) % n;
        let im2 = (i + n - 2) % n;

        let ux = 0.5 * (v[ip1] - v[im1]);
        let uxx = v[ip1] - 2.0 * v[i] + v[im1];
        let uxxxx = v[ip1] - 4.0 * v[i] + 6.0 * v[im1] - 4.0 * v[im2] + v[(i + 2) % n];

        let nonlinear = -v[i] * ux;
        let stiff = -uxx - uxxxx;
        let nv = g[i] * nonlinear;

        out[i] = e[i] * v[i] + q[i] * nv + f1[i] * nv + f2[i] * stiff + f3[i] * stiff;
    }

    out
}
