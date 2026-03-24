use chrono::NaiveDateTime;
use clap::Args;
use serde::{Deserialize, Serialize};

use crate::{
    args::time_lengths::TimeLengths,
    data::dataset::{
        dynamic_system::config::{
            default_columns, default_embed, default_parse_dates, default_path, from_series,
            split_borders, DynamicColumnName,
        },
        init_dataset::InitDataset,
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

impl InitDataset<DynamicColumnName> for LorenzConfig {
    fn parse_dates(_df: &polars::prelude::DataFrame, start_idx: usize, slice_len: usize) -> Vec<NaiveDateTime> {
        default_parse_dates(start_idx, slice_len)
    }

    fn path(&self) -> String {
        default_path()
    }

    fn train_columns(&self) -> Vec<DynamicColumnName> {
        default_columns()
    }

    fn target_columns(&self) -> Vec<DynamicColumnName> {
        default_columns()
    }

    fn embed(&self) -> crate::args::time_embed::TimeEmbed {
        default_embed()
    }

    fn split_borders(
        lengths: &TimeLengths,
        total_rows: usize,
    ) -> ((usize, usize, usize), (usize, usize, usize)) {
        split_borders(lengths, total_rows)
    }

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

fn rk4_step(state: [f64; 3], dt: f64, rho: f64, sigma: f64, beta: f64) -> [f64; 3] {
    let k1 = lorenz_diff(state, rho, sigma, beta);
    let s2 = [
        state[0] + 0.5 * dt * k1[0],
        state[1] + 0.5 * dt * k1[1],
        state[2] + 0.5 * dt * k1[2],
    ];
    let k2 = lorenz_diff(s2, rho, sigma, beta);
    let s3 = [
        state[0] + 0.5 * dt * k2[0],
        state[1] + 0.5 * dt * k2[1],
        state[2] + 0.5 * dt * k2[2],
    ];
    let k3 = lorenz_diff(s3, rho, sigma, beta);
    let s4 = [
        state[0] + dt * k3[0],
        state[1] + dt * k3[1],
        state[2] + dt * k3[2],
    ];
    let k4 = lorenz_diff(s4, rho, sigma, beta);

    [
        state[0] + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        state[1] + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        state[2] + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
    ]
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

    let mut states = Vec::with_capacity(n_timesteps);
    let mut state = x0;
    states.push(state);

    for _ in 1..n_timesteps {
        state = rk4_step(state, h, rho, sigma, beta);
        states.push(state);
    }

    states
}
