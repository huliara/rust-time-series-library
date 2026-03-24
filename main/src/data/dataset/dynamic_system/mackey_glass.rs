use rand::{rngs::StdRng, Rng, SeedableRng};

use super::_mg_rk4::_mg_rk4;

pub fn mackey_glass(
    n_timesteps: usize,
    tau: usize,
    a: f64,
    b: f64,
    n: i32,
    x0: f64,
    h: f64,
    seed: Option<u64>,
    history: Option<Vec<f64>>,
) -> Result<Vec<[f64; 1]>, String> {
    let history_length = ((tau as f64) / h).floor() as usize;

    let history_values = if let Some(history) = history {
        if history.len() < history_length {
            return Err(format!(
                "The given history has length of {} < tau/h with tau={} and h={}",
                history.len(), tau, h
            ));
        }
        history[history.len() - history_length..].to_vec()
    } else {
        let mut rng = StdRng::seed_from_u64(seed.unwrap_or(42));
        (0..history_length)
            .map(|_| x0 + 0.2 * (rng.gen::<f64>() - 0.5))
            .collect::<Vec<_>>()
    };

    let mut xt = x0;
    let mut x = vec![0.0_f64; history_length + n_timesteps];

    if history_length > 0 {
        x[..history_length].copy_from_slice(&history_values);
    }

    for i in history_length..(history_length + n_timesteps) {
        x[i] = xt;
        let xtau = if tau > 0 && history_length > 0 {
            x[i - history_length]
        } else {
            0.0
        };
        xt = _mg_rk4(xt, xtau, a, b, n as f64, h);
    }

    Ok(x[history_length..]
        .iter()
        .map(|v| [*v])
        .collect::<Vec<_>>())
}
