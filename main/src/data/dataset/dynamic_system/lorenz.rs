use clap::Args;
use serde::{Deserialize, Serialize};

use crate::{
    args::time_lengths::TimeLengths,
    data::dataset::{
        dynamic_system::{
            config::{from_series, split_borders},
            ivp_solve::{IvpMethod, IvpOptions, solve_ivp},
        },
        init_dynamic_system::InitDynamicSystem,
        init_time_series::InitTimeSeries,
        time_series_dataset::{ExpFlag, TimeSeriesDataset},
    },
};
use burn::prelude::Backend;

#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct LorenzConfig {
    #[arg(long, default_value_t = 10000)]
    pub n_timesteps: usize,
    #[arg(long, default_value_t = 28.0)]
    pub rho: f64,
    #[arg(long, default_value_t = 10.0)]
    pub sigma: f64,
    #[arg(long, default_value_t = 2.6666666666666665)]
    pub beta: f64,
    #[arg(long, default_value_t = 0.01)]
    pub h: f64,
    #[arg(long, num_args = 3, default_values_t = [1.0, 1.0, 1.0])]
    pub initial_value: Vec<f64>,
}

impl std::fmt::Display for LorenzConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "lorenz_nt{}_rho{:.2}", self.n_timesteps, self.rho)
    }
}

impl InitTimeSeries for LorenzConfig {
    fn split_borders(
        lengths: &TimeLengths,
        total_rows: usize,
    ) -> ((usize, usize, usize), (usize, usize, usize)) {
        split_borders(lengths, total_rows)
    }
}

impl InitDynamicSystem for LorenzConfig {
    fn init<B: Backend>(
        &self,
        lengths: &TimeLengths,
        flag: ExpFlag,
        device: &B::Device,
    ) -> TimeSeriesDataset<B> {
        if self.initial_value.len() != 3 {
            panic!("lorenz initial_value must contain exactly 3 elements");
        }
        let series = lorenz(
            self.n_timesteps,
            self.rho,
            self.sigma,
            self.beta,
            [
                self.initial_value[0],
                self.initial_value[1],
                self.initial_value[2],
            ],
            self.h,
        )
        .into_iter()
        .map(|v| v.to_vec())
        .collect::<Vec<_>>();
        from_series(series, lengths, flag, device)
    }
}

fn lorenz_diff(state: [f64; 3], rho: f64, sigma: f64, beta: f64) -> [f64; 3] {
    let x = state[0];
    let y = state[1];
    let z = state[2];
    [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
}

pub fn lorenz(
    n_timesteps: usize,
    rho: f64,
    sigma: f64,
    beta: f64,
    x0: [f64; 3],
    h: f64,
) -> Vec<[f64; 3]> {
    if n_timesteps == 0 {
        return Vec::new();
    }

    let t_eval = (0..n_timesteps).map(|i| i as f64 * h).collect::<Vec<_>>();
    let options = IvpOptions {
        method: IvpMethod::Rk45,
        t_eval: Some(t_eval),
        first_step: Some(h),
        max_step: h,
        min_step: h * 1e-6,
        rtol: 1e-8,
        atol: 1e-10,
    };

    let result = solve_ivp(
        |_t, y| {
            let d = lorenz_diff([y[0], y[1], y[2]], rho, sigma, beta);
            vec![d[0], d[1], d[2]]
        },
        (0.0, (n_timesteps - 1) as f64 * h),
        vec![x0[0], x0[1], x0[2]],
        options,
    )
    .expect("Failed to solve lorenz IVP");

    if !result.success {
        panic!("Failed to solve lorenz IVP: {}", result.message);
    }

    result
        .y
        .into_iter()
        .map(|v| [v[0], v[1], v[2]])
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use crate::data::dataset::dynamic_system::test::assert_dynamic_system_series;

    use super::lorenz;

    #[test]
    fn test_lorenz_dataset_against_python() {
        let n_timesteps = 400;
        let series = lorenz(
            n_timesteps,
            28.0,
            10.0,
            2.6666666666666665,
            [1.0, 1.0, 1.0],
            0.01,
        )
        .into_iter()
        .map(|v| v.to_vec())
        .collect::<Vec<_>>();

        let system_name = "lorenz";
        assert_dynamic_system_series(system_name, series);
    }
}
