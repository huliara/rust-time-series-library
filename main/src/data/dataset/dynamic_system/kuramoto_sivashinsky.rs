use std::f64::consts::PI;
use chrono::NaiveDateTime;
use burn::prelude::Backend;
use clap::Args;
use serde::{Deserialize, Serialize};

use crate::{
    args::time_lengths::TimeLengths,
    data::dataset::{
        dynamic_system::config::{
            default_columns, default_embed, default_parse_dates, default_path, from_series,
            split_borders, DynamicColumnName,
        },
        init_real_time_series::InitRealTimeSeries,
        time_series_dataset::{ExpFlag, TimeSeriesDataset},
    },
};

use super::_kuramoto_sivashinsky::_kuramoto_sivashinsky;

#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct KuramotoSivashinskyConfig {
    #[arg(long, default_value_t = 10000)]
    pub n_timesteps: usize,
    #[arg(long, default_value_t = 0)]
    pub warmup: usize,
    #[arg(long, default_value_t = 64)]
    pub n: usize,
    #[arg(long, default_value_t = 16.0)]
    pub m: f64,
    #[arg(long, default_value_t = 0.25)]
    pub h: f64,
}

impl std::fmt::Display for KuramotoSivashinskyConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ks_nt{}_n{}_m{:.1}", self.n_timesteps, self.n, self.m)
    }
}

impl InitRealTimeSeries<DynamicColumnName> for KuramotoSivashinskyConfig {
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
        let series = kuramoto_sivashinsky(self.n_timesteps, self.warmup, self.n, self.m, None, self.h)
            .expect("Failed to generate kuramoto_sivashinsky series");
        from_series(series, lengths, flag, device)
    }
}

pub fn kuramoto_sivashinsky(
    n_timesteps: usize,
    warmup: usize,
    n: usize,
    m: f64,
    x0: Option<Vec<f64>>,
    h: f64,
) -> Result<Vec<Vec<f64>>, String> {
    let initial = if let Some(x0) = x0 {
        if x0.len() != n {
            return Err(format!(
                "Initial condition x0 should be of shape {n} (= N) but has length {}",
                x0.len()
            ));
        }
        x0
    } else {
        (1..=n)
            .map(|idx| {
                let x = 2.0 * m * PI * (idx as f64) / (n as f64);
                (x / m).cos() * (1.0 + (x / m).sin())
            })
            .collect::<Vec<_>>()
    };

    _kuramoto_sivashinsky(n_timesteps, warmup, n, m, initial, h)
}
