use clap::Args;
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
use burn::prelude::Backend;
use ode_solvers::{Dop853, System, Vector3};

struct RabinovichFabrikantSystem {
    alpha: f64,
    gamma: f64,
}

impl System<f64, Vector3<f64>> for RabinovichFabrikantSystem {
    fn system(&self, _t: f64, y: &Vector3<f64>, dy: &mut Vector3<f64>) {
        let d = rabinovich_fabrikant_diff([y[0], y[1], y[2]], self.alpha, self.gamma);
        dy[0] = d[0];
        dy[1] = d[1];
        dy[2] = d[2];
    }
}

#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct RabinovichFabrikantConfig {
    #[arg(long, default_value_t = 10000)]
    pub n_timesteps: usize,
    #[arg(long, default_value_t = 0.14)]
    pub alpha: f64,
    #[arg(long, default_value_t = 0.1)]
    pub gamma: f64,
    #[arg(long, default_value_t = 0.005)]
    pub h: f64,
    #[arg(long, num_args = 3, default_values_t = [0.1, 0.1, 0.1])]
    pub initial_value: Vec<f64>,
}

impl std::fmt::Display for RabinovichFabrikantConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "rf_nt{}_a{:.3}_g{:.3}",
            self.n_timesteps, self.alpha, self.gamma
        )
    }
}

impl InitTimeSeries for RabinovichFabrikantConfig {
    fn split_borders(
        lengths: &TimeLengths,
        total_rows: usize,
    ) -> ((usize, usize, usize), (usize, usize, usize)) {
        split_borders(lengths, total_rows)
    }
}

impl InitDynamicSystem for RabinovichFabrikantConfig {
    fn init<B: Backend>(
        &self,
        lengths: &TimeLengths,
        flag: ExpFlag,
        device: &B::Device,
    ) -> TimeSeriesDataset<B> {
        if self.initial_value.len() != 3 {
            panic!("rabinovich_fabrikant initial_value must contain exactly 3 elements");
        }
        let series = rabinovich_fabrikant(
            self.n_timesteps,
            self.alpha,
            self.gamma,
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

fn rabinovich_fabrikant_diff(state: [f64; 3], alpha: f64, gamma: f64) -> [f64; 3] {
    let x = state[0];
    let y = state[1];
    let z = state[2];
    [
        y * (z - 1.0 + x * x) + gamma * x,
        x * (3.0 * z + 1.0 - x * x) + gamma * y,
        -2.0 * z * (alpha + x * y),
    ]
}

pub fn rabinovich_fabrikant(
    n_timesteps: usize,
    alpha: f64,
    gamma: f64,
    x0: [f64; 3],
    h: f64,
) -> Vec<[f64; 3]> {
    if n_timesteps == 0 {
        return Vec::new();
    }
    if n_timesteps == 1 {
        return vec![x0];
    }

    let t_max = n_timesteps as f64 * h;
    let dt_eval = t_max / (n_timesteps as f64 - 1.0);
    let system = RabinovichFabrikantSystem { alpha, gamma };
    let y0 = Vector3::new(x0[0], x0[1], x0[2]);
    let mut stepper = Dop853::new(system, 0.0, t_max, dt_eval, y0, 1e-8, 1e-10);
    stepper
        .integrate()
        .unwrap_or_else(|e| panic!("Failed to solve rabinovich_fabrikant IVP: {e}"));

    stepper
        .y_out()
        .into_iter()
        .map(|v| [v[0], v[1], v[2]])
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use crate::data::dataset::dynamic_system::test::{
        assert_dynamic_system_series, TEST_STEP_SIZE,
    };

    use super::rabinovich_fabrikant;

    #[test]
    fn test_rabinovich_fabrikant_dataset_against_python() {
        let n_timesteps = TEST_STEP_SIZE;
        let series = rabinovich_fabrikant(n_timesteps, 0.14, 0.1, [0.1, 0.1, 0.1], 0.005)
            .into_iter()
            .map(|v| v.to_vec())
            .collect::<Vec<_>>();

        let system_name = "rabinovich_fabrikant";
        assert_dynamic_system_series(system_name, series);
    }
}
