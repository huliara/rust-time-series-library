use burn::prelude::Backend;
use clap::Args;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::{
    args::time_lengths::TimeLengths,
    data::dataset::{
        dynamic_system::config::{from_series, split_borders},
        init_dynamic_system::InitDynamicSystem,
        init_time_series::InitTimeSeries,
        time_series_dataset::{ExpFlag, TimeSeriesDataset},
    },
};

use super::_mg_rk4::_mg_rk4;

#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct MackeyGlassConfig {
    #[arg(long, default_value_t = 10000)]
    pub n_timesteps: usize,
    #[arg(long, default_value_t = 17)]
    pub tau: usize,
    #[arg(long, default_value_t = 0.2)]
    pub a: f64,
    #[arg(long, default_value_t = 0.1)]
    pub b: f64,
    #[arg(long, default_value_t = 10)]
    pub n: i32,
    #[arg(long, default_value_t = 1.2)]
    pub x0: f64,
    #[arg(long, default_value_t = 0.1)]
    pub h: f64,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

impl std::fmt::Display for MackeyGlassConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mackeyglass_nt{}_tau{}_a{:.2}_b{:.2}",
            self.n_timesteps, self.tau, self.a, self.b
        )
    }
}

impl InitTimeSeries for MackeyGlassConfig {
    fn split_borders(
        lengths: &TimeLengths,
        total_rows: usize,
    ) -> ((usize, usize, usize), (usize, usize, usize)) {
        split_borders(lengths, total_rows)
    }
}

impl InitDynamicSystem for MackeyGlassConfig {
    fn init<B: Backend>(
        &self,
        lengths: &TimeLengths,
        flag: ExpFlag,
        device: &B::Device,
    ) -> TimeSeriesDataset<B> {
        let series = mackey_glass(
            self.n_timesteps,
            self.tau,
            self.a,
            self.b,
            self.n,
            self.x0,
            self.h,
            Some(self.seed),
            None,
        )
        .expect("Failed to generate mackey_glass series")
        .into_iter()
        .map(|v| v.to_vec())
        .collect::<Vec<_>>();
        from_series(series, lengths, flag, device)
    }
}

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
                history.len(),
                tau,
                h
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

    Ok(x[history_length..].iter().map(|v| [*v]).collect::<Vec<_>>())
}

#[cfg(test)]
mod tests {
    use crate::data::dataset::dynamic_system::test::{
        assert_dynamic_system_series, TEST_STEP_SIZE,
    };

    use super::mackey_glass;

    #[test]
    fn test_mackey_glass_dataset_against_python() {
        let n_timesteps = TEST_STEP_SIZE;
        let series = mackey_glass(n_timesteps, 0, 0.2, 0.1, 10, 1.2, 0.1, None, None)
            .unwrap()
            .into_iter()
            .map(|v| v.to_vec())
            .collect::<Vec<_>>();

        let system_name = "mackey_glass";
        assert_dynamic_system_series(system_name, series);
    }
}
