use std::f64::consts::PI;

use super::_kuramoto_sivashinsky::_kuramoto_sivashinsky;

pub fn kuramoto_sivashinsky(
    n_timesteps: usize,
    warmup: usize,
    n: usize,
    m: f64,
    x0: Option<Vec<f64>>,
    h: f64,
) -> Result<Vec<Vec<f64>>, String> {
    let initial = if let Some(x0) = x0 {
        if x0.len() != n {
            return Err(format!(
                "Initial condition x0 should be of shape {n} (= N) but has length {}",
                x0.len()
            ));
        }
        x0
    } else {
        (1..=n)
            .map(|idx| {
                let x = 2.0 * m * PI * (idx as f64) / (n as f64);
                (x / m).cos() * (1.0 + (x / m).sin())
            })
            .collect::<Vec<_>>()
    };

    _kuramoto_sivashinsky(n_timesteps, warmup, n, m, initial, h)
}
