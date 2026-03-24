use crate::{
    args::{time_embed::TimeEmbed, time_lengths::TimeLengths},
    data::{
        column_name::ExchangeColumnName,
        dataset::{init_real_time_series::InitRealTimeSeries, init_time_series::InitTimeSeries},
    },
};

use chrono::{DateTime, NaiveDateTime};
use clap::Args;
use core::fmt;

use polars::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct ExchangeConfig {
    #[arg(long)]
    pub path: String,
    #[arg(long, num_args = 1..)]
    pub train_features: Vec<ExchangeColumnName>,

    #[arg(long, num_args = 1..)]
    pub targets: Vec<ExchangeColumnName>,

    #[arg(long, value_enum)]
    pub embed: TimeEmbed,
}

impl Default for ExchangeConfig {
    fn default() -> Self {
        Self {
            path: "v2/USDJPY/h1/20020101-20250810.csv".to_string(),
            train_features: vec![
                ExchangeColumnName::Open,
                ExchangeColumnName::High,
                ExchangeColumnName::Low,
                ExchangeColumnName::Close,
                ExchangeColumnName::TickVolume,
                ExchangeColumnName::Spread,
                ExchangeColumnName::RealVolume,
            ],
            targets: vec![
                ExchangeColumnName::Open,
                ExchangeColumnName::High,
                ExchangeColumnName::Low,
                ExchangeColumnName::Close,
                ExchangeColumnName::TickVolume,
                ExchangeColumnName::Spread,
                ExchangeColumnName::RealVolume,
            ],
            embed: TimeEmbed::TimeF,
        }
    }
}

impl fmt::Display for ExchangeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let targets = self
            .targets
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join("_");
        let train_features = self
            .train_features
            .iter()
            .map(|feature| feature.to_string())
            .collect::<Vec<_>>()
            .join("_");

        write!(f, "{}_{}_{}", targets, train_features, self.embed)
    }
}

impl InitTimeSeries for ExchangeConfig {
    fn split_borders(
        lengths: &TimeLengths,
        total_rows: usize,
    ) -> ((usize, usize, usize), (usize, usize, usize)) {
        let num_train = (total_rows as f64 * 0.7) as usize;
        let num_test = (total_rows as f64 * 0.2) as usize;
        let num_val = total_rows - num_train - num_test;

        let raw_border1s = (
            0,
            num_train.saturating_sub(lengths.seq_len),
            total_rows.saturating_sub(num_test.saturating_add(lengths.seq_len)),
        );
        let raw_border2s: (usize, usize, usize) = (num_train, num_train + num_val, total_rows);

        (raw_border1s, raw_border2s)
    }
}

impl InitRealTimeSeries<ExchangeColumnName> for ExchangeConfig {
    fn embed(&self) -> TimeEmbed {
        self.embed.clone()
    }
    fn parse_dates(df: &DataFrame, start_idx: usize, slice_len: usize) -> Vec<NaiveDateTime> {
        df.slice(start_idx as i64, slice_len)
            .column("time")
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .map(|s| {
                DateTime::from_timestamp(s, 0)
                    .expect("Parse date")
                    .naive_utc()
            })
            .collect()
    }

    fn path(&self) -> String {
        self.path.clone()
    }

    fn train_columns(&self) -> Vec<ExchangeColumnName> {
        self.train_features.clone()
    }

    fn target_columns(&self) -> Vec<ExchangeColumnName> {
        self.targets.clone()
    }
}
