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
pub struct LogisticMapConfig {
    #[arg(long, default_value_t = 10000)]
    pub n_timesteps: usize,
    #[arg(long, default_value_t = 3.9)]
    pub r: f64,
    #[arg(long, default_value_t = 0.1)]
    pub x0: f64,
}

impl std::fmt::Display for LogisticMapConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "logistic_nt{}_r{:.3}", self.n_timesteps, self.r)
    }
}

impl InitDataset<DynamicColumnName> for LogisticMapConfig {
    fn parse_dates(
        _df: &polars::prelude::DataFrame,
        start_idx: usize,
        slice_len: usize,
    ) -> Vec<NaiveDateTime> {
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
        let series = logistic_map(self.n_timesteps, self.r, self.x0)
            .expect("Failed to generate logistic_map series")
            .into_iter()
            .map(|v| v.to_vec())
            .collect::<Vec<_>>();
        from_series(series, lengths, flag, device)
    }
}

pub fn logistic_map(n_timesteps: usize, r: f64, x0: f64) -> Result<Vec<[f64; 1]>, String> {
    if r <= 0.0 {
        return Err("r should be positive.".to_string());
    }
    if !(0.0 < x0 && x0 < 1.0) {
        return Err("Initial condition x0 should be in ]0;1[.".to_string());
    }
    if n_timesteps == 0 {
        return Ok(Vec::new());
    }

    let mut x = vec![[0.0_f64; 1]; n_timesteps];
    x[0][0] = x0;

    for i in 1..n_timesteps {
        x[i][0] = r * x[i - 1][0] * (1.0 - x[i - 1][0]);
    }

    Ok(x)
}
