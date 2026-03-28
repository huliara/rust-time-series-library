use clap::Args;
use serde::{Deserialize, Serialize};

use crate::{
    args::time_lengths::TimeLengths,
    data::dataset::{
        dynamic_system::{
            config::{from_series, split_borders},
        },
        init_dynamic_system::InitDynamicSystem,
        init_time_series::InitTimeSeries,
        time_series_dataset::{ExpFlag, TimeSeriesDataset},
    },
};
use burn::prelude::Backend;
use ode_solvers::{Dop853, System, Vector3};

struct MultiScrollSystem {
    a: f64,
    b: f64,
    c: f64,
}

impl System<f64, Vector3<f64>> for MultiScrollSystem {
    fn system(&self, _t: f64, y: &Vector3<f64>, dy: &mut Vector3<f64>) {
        let d = multiscroll_diff([y[0], y[1], y[2]], self.a, self.b, self.c);
        dy[0] = d[0];
        dy[1] = d[1];
        dy[2] = d[2];
    }
}

#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct MultiScrollConfig {
    #[arg(long, default_value_t = 10000)]
    pub n_timesteps: usize,
    #[arg(long, default_value_t = 36.0)]
    pub a: f64,
    #[arg(long, default_value_t = 3.0)]
    pub b: f64,
    #[arg(long, default_value_t = 20.0)]
    pub c: f64,
    #[arg(long, default_value_t = 0.01)]
    pub h: f64,
    #[arg(long, num_args = 3, default_values_t = [0.1, 0.0, 0.0])]
    pub initial_value: Vec<f64>,
}

impl std::fmt::Display for MultiScrollConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "multiscroll_nt{}_a{:.2}_b{:.2}_c{:.2}",
            self.n_timesteps, self.a, self.b, self.c
        )
    }
}

impl InitTimeSeries for MultiScrollConfig {
    fn split_borders(
        lengths: &TimeLengths,
        total_rows: usize,
    ) -> ((usize, usize, usize), (usize, usize, usize)) {
        split_borders(lengths, total_rows)
    }
}

impl InitDynamicSystem for MultiScrollConfig {
    fn init<B: Backend>(
        &self,
        lengths: &TimeLengths,
        flag: ExpFlag,
        device: &B::Device,
    ) -> TimeSeriesDataset<B> {
        if self.initial_value.len() != 3 {
            panic!("multiscroll initial_value must contain exactly 3 elements");
        }
        let series = multiscroll(
            self.n_timesteps,
            self.a,
            self.b,
            self.c,
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

fn multiscroll_diff(state: [f64; 3], a: f64, b: f64, c: f64) -> [f64; 3] {
    let x = state[0];
    let y = state[1];
    let z = state[2];
    [a * (y - x), (c - a) * x - x * z + c * y, x * y - b * z]
}

pub fn multiscroll(
    n_timesteps: usize,
    a: f64,
    b: f64,
    c: f64,
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
    let system = MultiScrollSystem { a, b, c };
    let y0 = Vector3::new(x0[0], x0[1], x0[2]);
    let mut stepper = Dop853::new(system, 0.0, t_max, dt_eval, y0, 1e-8, 1e-10);
    stepper
        .integrate()
        .unwrap_or_else(|e| panic!("Failed to solve multiscroll IVP: {e}"));

    stepper
        .y_out()
        .into_iter()
        .map(|v| [v[0], v[1], v[2]])
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use crate::data::dataset::dynamic_system::test::assert_dynamic_system_series;

    use super::multiscroll;

    #[test]
    fn test_multiscroll_dataset_against_python() {
        let n_timesteps = 400;
        let series = multiscroll(n_timesteps, 36.0, 3.0, 20.0, [0.1, 0.0, 0.0], 0.01)
            .into_iter()
            .map(|v| v.to_vec())
            .collect::<Vec<_>>();

        let system_name = "multiscroll";
        assert_dynamic_system_series(system_name, series);
    }
}
