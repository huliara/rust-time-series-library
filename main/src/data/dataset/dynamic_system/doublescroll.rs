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

struct DoubleScrollSystem {
    r1: f64,
    r2: f64,
    r4: f64,
    ir: f64,
    beta: f64,
}

impl System<f64, Vector3<f64>> for DoubleScrollSystem {
    fn system(&self, _t: f64, y: &Vector3<f64>, dy: &mut Vector3<f64>) {
        let d = doublescroll_diff(
            [y[0], y[1], y[2]],
            self.r1,
            self.r2,
            self.r4,
            self.ir,
            self.beta,
        );
        dy[0] = d[0];
        dy[1] = d[1];
        dy[2] = d[2];
    }
}

#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct DoubleScrollConfig {
    #[arg(long, default_value_t = 10000)]
    pub n_timesteps: usize,
    #[arg(long, default_value_t = 1.2)]
    pub r1: f64,
    #[arg(long, default_value_t = 3.44)]
    pub r2: f64,
    #[arg(long, default_value_t = 0.193)]
    pub r4: f64,
    #[arg(long, default_value_t = 2.25)]
    pub ir: f64,
    #[arg(long, default_value_t = 11.6)]
    pub beta: f64,
    #[arg(long, default_value_t = 0.01)]
    pub h: f64,
    #[arg(long, num_args = 3, default_values_t = [0.1, 0.0, 0.0])]
    pub initial_value: Vec<f64>,
}

impl std::fmt::Display for DoubleScrollConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "doublescroll_nt{}_r1{:.2}_r2{:.2}",
            self.n_timesteps, self.r1, self.r2
        )
    }
}

impl InitTimeSeries for DoubleScrollConfig {
    fn split_borders(
        lengths: &TimeLengths,
        total_rows: usize,
    ) -> ((usize, usize, usize), (usize, usize, usize)) {
        split_borders(lengths, total_rows)
    }
}

impl InitDynamicSystem for DoubleScrollConfig {
    fn init<B: Backend>(
        &self,
        lengths: &TimeLengths,
        flag: ExpFlag,
        device: &B::Device,
    ) -> TimeSeriesDataset<B> {
        if self.initial_value.len() != 3 {
            panic!("doublescroll initial_value must contain exactly 3 elements");
        }
        let series = doublescroll(
            self.n_timesteps,
            self.r1,
            self.r2,
            self.r4,
            self.ir,
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

fn doublescroll_diff(state: [f64; 3], r1: f64, r2: f64, r4: f64, ir: f64, beta: f64) -> [f64; 3] {
    let v1 = state[0];
    let v2 = state[1];
    let i = state[2];

    let dv = v1 - v2;
    let factor = (dv / r2) + ir * (beta * dv).sinh();
    let dv1 = (v1 / r1) - factor;
    let dv2 = factor - i;
    let di = v2 - r4 * i;

    [dv1, dv2, di]
}

pub fn doublescroll(
    n_timesteps: usize,
    r1: f64,
    r2: f64,
    r4: f64,
    ir: f64,
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
    let system = DoubleScrollSystem {
        r1,
        r2,
        r4,
        ir,
        beta,
    };
    let y0 = Vector3::new(x0[0], x0[1], x0[2]);
    let mut stepper = Dop853::new(system, 0.0, t_max, dt_eval, y0, 1e-8, 1e-10);
    stepper
        .integrate()
        .unwrap_or_else(|e| panic!("Failed to solve doublescroll IVP: {e}"));

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

    use super::doublescroll;

    #[test]
    fn test_doublescroll_dataset_against_python() {
        let n_timesteps = TEST_STEP_SIZE;
        let series = doublescroll(
            n_timesteps,
            1.2,
            3.44,
            0.193,
            2.25,
            11.6,
            [0.1, 0.0, 0.0],
            0.01,
        )
        .into_iter()
        .map(|v| v.to_vec())
        .collect::<Vec<_>>();

        let system_name = "doublescroll";
        assert_dynamic_system_series(system_name, series);
    }
}
