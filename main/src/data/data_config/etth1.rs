use crate::{
    args::{time_embed::TimeEmbed, time_lengths::TimeLengths},
    data::{column_name::EtthColumnName, data_config::init_dataset::InitDataset},
};

use chrono::NaiveDateTime;
use clap::Args;

use polars::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct Etth1Args {
    #[arg(long)]
    pub path: String,
    #[arg(long, num_args = 1..)]
    pub train_features: Vec<EtthColumnName>,

    #[arg(long, num_args = 1..)]
    pub targets: Vec<EtthColumnName>,

    #[arg(long, value_enum)]
    pub embed: TimeEmbed,
}

impl InitDataset<EtthColumnName> for Etth1Args {
    fn parse_dates(df: &DataFrame, start_idx: usize, slice_len: usize) -> Vec<NaiveDateTime> {
        df.slice(start_idx as i64, slice_len)
            .column("date")
            .unwrap()
            .str()
            .unwrap()
            .into_no_null_iter()
            .map(|s| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S").expect("Parse date"))
            .collect()
    }
    fn split_borders(
        lengths: &TimeLengths,
        _total_rows: usize,
    ) -> ((usize, usize, usize), (usize, usize, usize)) {
        let raw_border1s = (
            0,
            (12usize * 30 * 24).saturating_sub(lengths.seq_len),
            (12usize * 30 * 24 + 4 * 30 * 24).saturating_sub(lengths.seq_len),
        );
        let raw_border2s: (usize, usize, usize) = (
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        );
        (raw_border1s, raw_border2s)
    }
}
