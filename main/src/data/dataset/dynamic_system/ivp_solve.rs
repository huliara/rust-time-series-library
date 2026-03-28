#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IvpMethod {
    Rk45,
    Dop853,
}

#[derive(Debug, Clone)]
pub struct IvpOptions {
    pub method: IvpMethod,
    pub t_eval: Option<Vec<f64>>,
    pub first_step: Option<f64>,
    pub max_step: f64,
    pub min_step: f64,
    pub rtol: f64,
    pub atol: f64,
}

impl Default for IvpOptions {
    fn default() -> Self {
        Self {
            method: IvpMethod::Rk45,
            t_eval: None,
            first_step: None,
            max_step: f64::INFINITY,
            min_step: 1e-12,
            rtol: 1e-3,
            atol: 1e-6,
        }
    }
}

#[derive(Debug, Clone)]
pub struct IvpResult {
    pub t: Vec<f64>,
    pub y: Vec<Vec<f64>>,
    pub nfev: usize,
    pub status: i32,
    pub message: String,
    pub success: bool,
}

fn lin_interp(t: f64, t0: f64, y0: &[f64], t1: f64, y1: &[f64]) -> Vec<f64> {
    if (t1 - t0).abs() <= f64::EPSILON {
        return y1.to_vec();
    }
    let a = (t - t0) / (t1 - t0);
    y0.iter()
        .zip(y1.iter())
        .map(|(v0, v1)| v0 + a * (v1 - v0))
        .collect()
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

fn validate_t_eval(t_eval: &[f64], t0: f64, tf: f64) -> Result<(), String> {
    if t_eval.is_empty() {
        return Ok(());
    }

    let min_t = t0.min(tf);
    let max_t = t0.max(tf);
    if t_eval.iter().any(|t| *t < min_t || *t > max_t) {
        return Err("Values in t_eval are not within t_span.".to_string());
    }

    if tf > t0 {
        if t_eval.windows(2).any(|w| w[1] <= w[0]) {
            return Err("Values in t_eval are not properly sorted.".to_string());
        }
    } else if t_eval.windows(2).any(|w| w[1] >= w[0]) {
        return Err("Values in t_eval are not properly sorted.".to_string());
    }

    Ok(())
}

fn rk45_step<F>(fun: &mut F, t: f64, y: &[f64], h: f64) -> (Vec<f64>, Vec<f64>, usize)
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    let n = y.len();

    let k1 = fun(t, y);

    let y2 = (0..n)
        .map(|i| y[i] + h * (1.0 / 5.0) * k1[i])
        .collect::<Vec<_>>();
    let k2 = fun(t + h * (1.0 / 5.0), &y2);

    let y3 = (0..n)
        .map(|i| y[i] + h * (3.0 / 40.0 * k1[i] + 9.0 / 40.0 * k2[i]))
        .collect::<Vec<_>>();
    let k3 = fun(t + h * (3.0 / 10.0), &y3);

    let y4 = (0..n)
        .map(|i| y[i] + h * (44.0 / 45.0 * k1[i] - 56.0 / 15.0 * k2[i] + 32.0 / 9.0 * k3[i]))
        .collect::<Vec<_>>();
    let k4 = fun(t + h * (4.0 / 5.0), &y4);

    let y5 = (0..n)
        .map(|i| {
            y[i] + h
                * (19372.0 / 6561.0 * k1[i] - 25360.0 / 2187.0 * k2[i] + 64448.0 / 6561.0 * k3[i]
                    - 212.0 / 729.0 * k4[i])
        })
        .collect::<Vec<_>>();
    let k5 = fun(t + h * (8.0 / 9.0), &y5);

    let y6 = (0..n)
        .map(|i| {
            y[i] + h
                * (9017.0 / 3168.0 * k1[i] - 355.0 / 33.0 * k2[i]
                    + 46732.0 / 5247.0 * k3[i]
                    + 49.0 / 176.0 * k4[i]
                    - 5103.0 / 18656.0 * k5[i])
        })
        .collect::<Vec<_>>();
    let k6 = fun(t + h, &y6);

    let y7 = (0..n)
        .map(|i| {
            y[i] + h
                * (35.0 / 384.0 * k1[i] + 500.0 / 1113.0 * k3[i] + 125.0 / 192.0 * k4[i]
                    - 2187.0 / 6784.0 * k5[i]
                    + 11.0 / 84.0 * k6[i])
        })
        .collect::<Vec<_>>();
    let k7 = fun(t + h, &y7);

    // 5th order solution.
    let y5_out = y7.clone();

    // 4th order solution (embedded).
    let y4_out = (0..n)
        .map(|i| {
            y[i] + h
                * (5179.0 / 57600.0 * k1[i] + 7571.0 / 16695.0 * k3[i] + 393.0 / 640.0 * k4[i]
                    - 92097.0 / 339200.0 * k5[i]
                    + 187.0 / 2100.0 * k6[i]
                    + 1.0 / 40.0 * k7[i])
        })
        .collect::<Vec<_>>();

    (y5_out, y4_out, 7)
}

const DOP853_C: [f64; 12] = [
    0.0,
    5.260015195876773e-2,
    7.89002279381516e-2,
    1.183503419072274e-1,
    2.816496580927726e-1,
    3.333333333333333e-1,
    2.5e-1,
    3.076923076923077e-1,
    6.512820512820513e-1,
    6.0e-1,
    8.571428571428571e-1,
    1.0,
];

const DOP853_A: [[f64; 12]; 12] = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [
        5.260015195876773e-2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        1.97250569845379e-2,
        5.91751709536137e-2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        2.958758547680685e-2,
        0.0,
        8.876275643042055e-2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        2.413651341592667e-1,
        0.0,
        -8.845494793282861e-1,
        9.24834003261792e-1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        3.7037037037037035e-2,
        0.0,
        0.0,
        1.7082860872947388e-1,
        1.2546768756682244e-1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        3.7109375e-2,
        0.0,
        0.0,
        1.7025221101954404e-1,
        6.021653898045596e-2,
        -1.7578125e-2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        3.709200011850479e-2,
        0.0,
        0.0,
        1.7038392571223998e-1,
        1.0726203044637329e-1,
        -1.53194377486244e-2,
        8.273789163814023e-3,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        6.241109587160757e-1,
        0.0,
        0.0,
        -3.3608926294469414,
        -8.68219346841726e-1,
        2.759209969944671e1,
        2.0154067550477892e1,
        -4.348988418106996e1,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        4.776625364382644e-1,
        0.0,
        0.0,
        -2.4881146199716677,
        -5.90290826836843e-1,
        2.1230051448181194e1,
        1.5279233632882423e1,
        -3.328821096898486e1,
        -2.0331201708508625e-2,
        0.0,
        0.0,
        0.0,
    ],
    [
        -9.371424300859874e-1,
        0.0,
        0.0,
        5.186372428844064,
        1.0914373489967297,
        -8.149787010746925,
        -1.852006565999696e1,
        2.2739487099350503e1,
        2.4936055526796523,
        -3.0467644718982196,
        0.0,
        0.0,
    ],
    [
        2.273310147516538,
        0.0,
        0.0,
        -1.053449546673725e1,
        -2.0008720582248626,
        -1.79589318631188e1,
        2.794888452941996e1,
        -2.8589982771350235,
        -8.87285693353063,
        1.2360567175794304e1,
        6.433927460157635e-1,
        0.0,
    ],
];

const DOP853_B: [f64; 12] = [
    5.429373411656876e-2,
    0.0,
    0.0,
    0.0,
    0.0,
    4.450312892752409,
    1.8915178993145003,
    -5.801203960010584,
    3.111643669578199e-1,
    -1.5216094966251608e-1,
    2.0136540080403035e-1,
    4.471061572777259e-2,
];

const DOP853_E3: [f64; 13] = [
    -1.8980075407246552e-1,
    0.0,
    0.0,
    0.0,
    0.0,
    4.450312892752409,
    1.8915178993145003,
    -5.801203960010584,
    -4.22682321323792e-1,
    -1.5216094966251608e-1,
    2.0136540080403035e-1,
    2.2651792198360828e-2,
    0.0,
];

const DOP853_E5: [f64; 13] = [
    1.312004499419488e-2,
    0.0,
    0.0,
    0.0,
    0.0,
    -1.2251564463762044,
    -4.957589496572502e-1,
    1.6643771824549865,
    -3.503288487499737e-1,
    3.341791187130175e-1,
    8.192320648511571e-2,
    -2.2355307863886295e-2,
    0.0,
];

fn dop853_step<F>(fun: &mut F, t: f64, y: &[f64], h: f64) -> (Vec<f64>, Vec<Vec<f64>>, usize)
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    let n = y.len();
    let mut k = vec![vec![0.0_f64; n]; 13];
    k[0] = fun(t, y);

    for s in 1..12 {
        let mut ys = y.to_vec();
        for i in 0..n {
            let mut acc = 0.0_f64;
            for (j, kj) in k.iter().enumerate().take(s) {
                acc += DOP853_A[s][j] * kj[i];
            }
            ys[i] += h * acc;
        }
        k[s] = fun(t + DOP853_C[s] * h, &ys);
    }

    let mut y8 = y.to_vec();
    for i in 0..n {
        let mut acc = 0.0_f64;
        for (j, kj) in k.iter().enumerate().take(12) {
            acc += DOP853_B[j] * kj[i];
        }
        y8[i] += h * acc;
    }

    k[12] = fun(t + h, &y8);

    (y8, k, 13)
}

pub fn solve_ivp<F>(
    mut fun: F,
    t_span: (f64, f64),
    y0: Vec<f64>,
    options: IvpOptions,
) -> Result<IvpResult, String>
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    if y0.is_empty() {
        return Err("y0 must not be empty.".to_string());
    }
    if options.max_step <= 0.0 {
        return Err("max_step must be positive.".to_string());
    }
    if options.min_step <= 0.0 {
        return Err("min_step must be positive.".to_string());
    }
    if options.rtol < 0.0 || options.rtol > 1.0 {
        return Err("rtol must be in [0, 1].".to_string());
    }
    if options.atol < 0.0 {
        return Err("atol must be non-negative.".to_string());
    }

    let (t0, tf) = t_span;
    if (tf - t0).abs() <= f64::EPSILON {
        return Ok(IvpResult {
            t: vec![t0],
            y: vec![y0],
            nfev: 0,
            status: 0,
            message: "The solver successfully reached the end of the integration interval."
                .to_string(),
            success: true,
        });
    }

    let error_exponent = match options.method {
        IvpMethod::Rk45 => -1.0 / 5.0,
        IvpMethod::Dop853 => -1.0 / 8.0,
    };

    let direction = if tf > t0 { 1.0 } else { -1.0 };
    let span = (tf - t0).abs();

    if let Some(t_eval) = options.t_eval.as_ref() {
        validate_t_eval(t_eval, t0, tf)?;
    }

    let mut t = t0;
    let mut y = y0;
    let mut nfev = 0usize;

    let mut h_abs = options
        .first_step
        .unwrap_or((span / 100.0).max(options.min_step))
        .abs()
        .min(options.max_step)
        .max(options.min_step);

    let mut ts = Vec::<f64>::new();
    let mut ys = Vec::<Vec<f64>>::new();

    let mut t_eval_idx = 0usize;
    let t_eval_owned = options.t_eval.clone();

    match t_eval_owned.as_ref() {
        Some(t_eval) => {
            while t_eval_idx < t_eval.len()
                && ((direction > 0.0 && t_eval[t_eval_idx] <= t)
                    || (direction < 0.0 && t_eval[t_eval_idx] >= t))
            {
                ts.push(t_eval[t_eval_idx]);
                ys.push(y.clone());
                t_eval_idx += 1;
            }
        }
        None => {
            ts.push(t);
            ys.push(y.clone());
        }
    }

    let safety = 0.9_f64;
    let min_factor = 0.2_f64;
    let max_factor = match options.method {
        IvpMethod::Rk45 => 5.0_f64,
        IvpMethod::Dop853 => 10.0_f64,
    };
    let mut iterations = 0usize;
    let max_iterations = 2_000_000usize;

    while (tf - t) * direction > 0.0 {
        iterations += 1;
        if iterations > max_iterations {
            return Ok(IvpResult {
                t: ts,
                y: ys,
                nfev,
                status: -1,
                message: "Maximum number of iterations reached.".to_string(),
                success: false,
            });
        }

        let remaining = (tf - t) * direction;
        let h = direction * h_abs.min(remaining);

        let y_prev = y.clone();
        let t_prev = t;

        let (y_next, err_norm) = match options.method {
            IvpMethod::Rk45 => {
                let (y5, y4, evals) = rk45_step(&mut fun, t, &y, h);
                nfev += evals;

                let mut err_sq_sum = 0.0_f64;
                for i in 0..y.len() {
                    let sc = options.atol + options.rtol * y[i].abs().max(y5[i].abs());
                    let e = (y5[i] - y4[i]) / sc.max(1e-30);
                    err_sq_sum += e * e;
                }
                let err = (err_sq_sum / y.len() as f64).sqrt();
                (y5, err)
            }
            IvpMethod::Dop853 => {
                let (y8, k, evals) = dop853_step(&mut fun, t, &y, h);
                nfev += evals;

                let mut err5_norm2 = 0.0_f64;
                let mut err3_norm2 = 0.0_f64;

                for i in 0..y.len() {
                    let scale = options.atol + options.rtol * y[i].abs().max(y8[i].abs());
                    let sc = scale.max(1e-30);

                    let mut e5 = 0.0_f64;
                    let mut e3 = 0.0_f64;
                    for j in 0..13 {
                        e5 += DOP853_E5[j] * k[j][i];
                        e3 += DOP853_E3[j] * k[j][i];
                    }

                    let v5 = e5 / sc;
                    let v3 = e3 / sc;
                    err5_norm2 += v5 * v5;
                    err3_norm2 += v3 * v3;
                }

                let err = if err5_norm2 == 0.0 && err3_norm2 == 0.0 {
                    0.0
                } else {
                    h.abs() * err5_norm2
                        / ((err5_norm2 + 0.01 * err3_norm2) * y.len() as f64).sqrt()
                };
                (y8, err)
            }
        };

        if err_norm <= 1.0 {
            t += h;
            y = y_next;

            match t_eval_owned.as_ref() {
                Some(t_eval) => {
                    while t_eval_idx < t_eval.len()
                        && ((direction > 0.0 && t_eval[t_eval_idx] <= t)
                            || (direction < 0.0 && t_eval[t_eval_idx] >= t))
                    {
                        let te = t_eval[t_eval_idx];
                        let yi = lin_interp(te, t_prev, &y_prev, t, &y);
                        ts.push(te);
                        ys.push(yi);
                        t_eval_idx += 1;
                    }
                }
                None => {
                    ts.push(t);
                    ys.push(y.clone());
                }
            }

            let factor = if err_norm == 0.0 {
                max_factor
            } else {
                (safety * err_norm.powf(error_exponent)).clamp(min_factor, max_factor)
            };
            h_abs = (h_abs * factor).clamp(options.min_step, options.max_step);
        } else {
            let factor = (safety * err_norm.powf(error_exponent)).clamp(0.1, 0.5);
            h_abs = (h_abs * factor).max(options.min_step);
            if h_abs <= options.min_step {
                return Ok(IvpResult {
                    t: ts,
                    y: ys,
                    nfev,
                    status: -1,
                    message: "Integration step failed: minimum step size reached.".to_string(),
                    success: false,
                });
            }
        }

        if max_abs(&y).is_nan() || max_abs(&y).is_infinite() {
            return Ok(IvpResult {
                t: ts,
                y: ys,
                nfev,
                status: -1,
                message: "Integration failed: non-finite values encountered.".to_string(),
                success: false,
            });
        }
    }

    if let Some(t_eval) = t_eval_owned.as_ref() {
        while t_eval_idx < t_eval.len() {
            ts.push(t_eval[t_eval_idx]);
            ys.push(y.clone());
            t_eval_idx += 1;
        }
    }

    Ok(IvpResult {
        t: ts,
        y: ys,
        nfev,
        status: 0,
        message: "The solver successfully reached the end of the integration interval.".to_string(),
        success: true,
    })
}

#[cfg(test)]
mod tests {
    use super::{solve_ivp, IvpMethod, IvpOptions};

    #[test]
    fn test_solve_ivp_exponential_decay() {
        let options = IvpOptions {
            method: IvpMethod::Rk45,
            rtol: 1e-7,
            atol: 1e-9,
            ..Default::default()
        };

        let result = solve_ivp(|_t, y| vec![-0.5 * y[0]], (0.0, 10.0), vec![2.0], options).unwrap();

        assert!(result.success);
        let y_end = result.y.last().unwrap()[0];
        let expected = 2.0 * (-5.0_f64).exp();
        assert!((y_end - expected).abs() < 5e-5);
    }

    #[test]
    fn test_solve_ivp_with_t_eval() {
        let options = IvpOptions {
            method: IvpMethod::Rk45,
            t_eval: Some(vec![0.0, 1.0, 2.0, 4.0, 10.0]),
            rtol: 1e-7,
            atol: 1e-9,
            ..Default::default()
        };

        let result = solve_ivp(|_t, y| vec![-0.5 * y[0]], (0.0, 10.0), vec![2.0], options).unwrap();

        assert_eq!(result.t.len(), 5);
        assert_eq!(result.y.len(), 5);

        let y_end = result.y.last().unwrap()[0];
        let expected = 2.0 * (-5.0_f64).exp();
        assert!((y_end - expected).abs() < 5e-5);
    }

    #[test]
    fn test_solve_ivp_dop853_exponential_decay() {
        let options = IvpOptions {
            method: IvpMethod::Dop853,
            t_eval: Some(vec![0.0, 1.0, 2.0, 4.0, 10.0]),
            rtol: 1e-9,
            atol: 1e-12,
            ..Default::default()
        };

        let result = solve_ivp(|_t, y| vec![-0.5 * y[0]], (0.0, 10.0), vec![2.0], options).unwrap();

        assert!(result.success);
        assert_eq!(result.t.len(), 5);
        assert_eq!(result.y.len(), 5);

        let y_end = result.y.last().unwrap()[0];
        let expected = 2.0 * (-5.0_f64).exp();
        assert!((y_end - expected).abs() < 1e-6);
    }
}
