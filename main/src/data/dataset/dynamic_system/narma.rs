use rand::{rngs::StdRng, Rng, SeedableRng};

pub fn narma(
    n_timesteps: usize,
    order: usize,
    a1: f64,
    a2: f64,
    b: f64,
    c: f64,
    x0: Vec<f64>,
    seed: Option<u64>,
    u: Option<Vec<f64>>,
) -> (Vec<[f64; 1]>, Vec<[f64; 1]>) {
    let mut y = vec![[0.0_f64; 1]; n_timesteps + order];

    for (i, v) in x0.iter().enumerate().take(y.len()) {
        y[i][0] = *v;
    }

    let u_series = if let Some(input) = u {
        input
    } else {
        let mut rng = StdRng::seed_from_u64(seed.unwrap_or(42));
        (0..(n_timesteps + order))
            .map(|_| rng.gen_range(0.0..0.5))
            .collect::<Vec<_>>()
    };

    for t in order..(n_timesteps + order - 1) {
        let sum_hist = y[t - order..t].iter().map(|v| v[0]).sum::<f64>();
        y[t + 1][0] = a1 * y[t][0] + a2 * y[t][0] * sum_hist + b * u_series[t - order] * u_series[t] + c;
    }

    let u_out = u_series.into_iter().map(|v| [v]).collect::<Vec<_>>();
    let y_out = y[order..].to_vec();
    (u_out, y_out)
}
