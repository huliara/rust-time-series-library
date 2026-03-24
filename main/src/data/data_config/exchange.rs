use std::path::PathBuf;

use crate::{
    args::{time_embed::TimeEmbed, time_lengths::TimeLengths},
    data::{
        column_name::{EtthColumnName, ExchangeColumnName},
        data_config::init_dataset::InitDataset,
    },
};

use chrono::{DateTime, NaiveDateTime};
use clap::Args;

use polars::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct ExchangeArgs {
    #[arg(long)]
    pub path: String,
    #[arg(long, num_args = 1..)]
    pub train_features: Vec<ExchangeColumnName>,

    #[arg(long, num_args = 1..)]
    pub targets: Vec<ExchangeColumnName>,

    #[arg(long, value_enum)]
    pub embed: TimeEmbed,
}

impl InitDataset<ExchangeColumnName> for ExchangeArgs {
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
