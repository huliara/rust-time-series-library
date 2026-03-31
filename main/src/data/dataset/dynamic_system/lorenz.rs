use clap::Args;
use serde::{Deserialize, Serialize};

use crate::{
    args::time_lengths::TimeLengths,
    data::dataset::{
        init_dynamic_system::InitDynamicSystem,
        init_time_series::InitTimeSeries,
        time_series_dataset::{ExpFlag, TimeSeriesDataset},
    },
};
use burn::prelude::Backend;
use ode_solvers::{Dop853, System, Vector3};

struct LorenzSystem {
    rho: f64,
    sigma: f64,
    beta: f64,
}

impl System<f64, Vector3<f64>> for LorenzSystem {
    fn system(&self, _t: f64, y: &Vector3<f64>, dy: &mut Vector3<f64>) {
        let d = lorenz_diff([y[0], y[1], y[2]], self.rho, self.sigma, self.beta);
        dy[0] = d[0];
        dy[1] = d[1];
        dy[2] = d[2];
    }
}

#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct LorenzConfig {
    #[arg(long, default_value_t = 10000)]
    pub n_timesteps: usize,
    #[arg(long, default_value_t = 28.0)]
    pub rho: f64,
    #[arg(long, default_value_t = 10.0)]
    pub sigma: f64,
    #[arg(long, default_value_t = 8.0/3.0)]
    pub beta: f64,
    #[arg(long, default_value_t = 0.03)]
    pub h: f64,
    #[arg(long, num_args = 3, default_values_t = [1.0, 1.0, 1.0])]
    pub initial_value: Vec<f64>,
}

impl std::fmt::Display for LorenzConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "lorenz_nt{}_rho{:.2}", self.n_timesteps, self.rho)
    }
}

impl InitTimeSeries for LorenzConfig {}

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
        Self::from_series(series, lengths, flag, device)
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
    if n_timesteps == 1 {
        return vec![x0];
    }

    let t_max = n_timesteps as f64 * h;
    let dt_eval = t_max / (n_timesteps as f64 - 1.0);
    let system = LorenzSystem { rho, sigma, beta };
    let y0 = Vector3::new(x0[0], x0[1], x0[2]);
    let mut stepper = Dop853::new(system, 0.0, t_max, dt_eval, y0, 1e-8, 1e-10);
    stepper
        .integrate()
        .unwrap_or_else(|e| panic!("Failed to solve lorenz IVP: {e}"));

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

    use super::lorenz;

    #[test]
    fn test_lorenz_dataset_against_python() {
        let n_timesteps = TEST_STEP_SIZE;
        let series = lorenz(n_timesteps, 28.0, 10.0, 8.0 / 3.0, [1.0, 1.0, 1.0], 0.03)
            .into_iter()
            .map(|v| v.to_vec())
            .collect::<Vec<_>>();

        let system_name = "lorenz";
        assert_dynamic_system_series(system_name, series);
    }
}
