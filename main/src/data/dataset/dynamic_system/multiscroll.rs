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
        write!(f, "multiscroll_nt{}_a{:.2}_b{:.2}_c{:.2}", self.n_timesteps, self.a, self.b, self.c)
    }
}

impl InitDataset<DynamicColumnName> for MultiScrollConfig {
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

fn rk4_step(state: [f64; 3], dt: f64, a: f64, b: f64, c: f64) -> [f64; 3] {
    let k1 = multiscroll_diff(state, a, b, c);
    let s2 = [
        state[0] + 0.5 * dt * k1[0],
        state[1] + 0.5 * dt * k1[1],
        state[2] + 0.5 * dt * k1[2],
    ];
    let k2 = multiscroll_diff(s2, a, b, c);
    let s3 = [
        state[0] + 0.5 * dt * k2[0],
        state[1] + 0.5 * dt * k2[1],
        state[2] + 0.5 * dt * k2[2],
    ];
    let k3 = multiscroll_diff(s3, a, b, c);
    let s4 = [
        state[0] + dt * k3[0],
        state[1] + dt * k3[1],
        state[2] + dt * k3[2],
    ];
    let k4 = multiscroll_diff(s4, a, b, c);

    [
        state[0] + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        state[1] + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        state[2] + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
    ]
}

pub fn multiscroll(n_timesteps: usize, a: f64, b: f64, c: f64, x0: [f64; 3], h: f64) -> Vec<[f64; 3]> {
    if n_timesteps == 0 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(n_timesteps);
    let mut state = x0;
    out.push(state);

    for _ in 1..n_timesteps {
        state = rk4_step(state, h, a, b, c);
        out.push(state);
    }

    out
}
