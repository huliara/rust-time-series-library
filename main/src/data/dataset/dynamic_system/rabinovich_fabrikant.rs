use clap::Args;
use serde::{Deserialize, Serialize};

use crate::{
    args::time_lengths::TimeLengths,
    data::dataset::{
        dynamic_system::config::{from_series, split_borders},
        init_dynamic_system::InitDynamicSystem as InitDynamicSystem,
        init_time_series::InitTimeSeries,
        time_series_dataset::{ExpFlag, TimeSeriesDataset},
    },
};
use burn::prelude::Backend;

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

fn rk4_step(state: [f64; 3], dt: f64, alpha: f64, gamma: f64) -> [f64; 3] {
    let k1 = rabinovich_fabrikant_diff(state, alpha, gamma);
    let s2 = [
        state[0] + 0.5 * dt * k1[0],
        state[1] + 0.5 * dt * k1[1],
        state[2] + 0.5 * dt * k1[2],
    ];
    let k2 = rabinovich_fabrikant_diff(s2, alpha, gamma);
    let s3 = [
        state[0] + 0.5 * dt * k2[0],
        state[1] + 0.5 * dt * k2[1],
        state[2] + 0.5 * dt * k2[2],
    ];
    let k3 = rabinovich_fabrikant_diff(s3, alpha, gamma);
    let s4 = [
        state[0] + dt * k3[0],
        state[1] + dt * k3[1],
        state[2] + dt * k3[2],
    ];
    let k4 = rabinovich_fabrikant_diff(s4, alpha, gamma);

    [
        state[0] + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        state[1] + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        state[2] + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
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

    let mut out = Vec::with_capacity(n_timesteps);
    let mut state = x0;
    out.push(state);

    for _ in 1..n_timesteps {
        state = rk4_step(state, h, alpha, gamma);
        out.push(state);
    }

    out
}
