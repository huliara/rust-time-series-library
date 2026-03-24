use crate::{
    args::{time_embed::TimeEmbed, time_lengths::TimeLengths},
    data::{
        column_name::EtthColumnName,
        dataset::{init_real_time_series::InitRealTimeSeries, init_time_series::InitTimeSeries},
    },
};

use chrono::NaiveDateTime;
use clap::Args;
use core::fmt;

use polars::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct Etth1Config {
    #[arg(long)]
    pub path: String,
    #[arg(long, num_args = 1..)]
    pub train_features: Vec<EtthColumnName>,

    #[arg(long, num_args = 1..)]
    pub targets: Vec<EtthColumnName>,

    #[arg(long, value_enum)]
    pub embed: TimeEmbed,
}

impl Default for Etth1Config {
    fn default() -> Self {
        Self {
            path: "ETT/ETTh1.csv".to_string(),
            train_features: vec![
                EtthColumnName::Hufl,
                EtthColumnName::Hull,
                EtthColumnName::Mufl,
                EtthColumnName::Mull,
                EtthColumnName::Lufl,
                EtthColumnName::Lull,
                EtthColumnName::Ot,
            ],
            targets: vec![
                EtthColumnName::Hufl,
                EtthColumnName::Hull,
                EtthColumnName::Mufl,
                EtthColumnName::Mull,
                EtthColumnName::Lufl,
                EtthColumnName::Lull,
                EtthColumnName::Ot,
            ],
            embed: TimeEmbed::TimeF,
        }
    }
}

impl fmt::Display for Etth1Config {
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

impl InitTimeSeries for Etth1Config {
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

impl InitRealTimeSeries<EtthColumnName> for Etth1Config {
    fn embed(&self) -> TimeEmbed {
        self.embed.clone()
    }
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

    fn path(&self) -> String {
        self.path.clone()
    }

    fn train_columns(&self) -> Vec<EtthColumnName> {
        self.train_features.clone()
    }

    fn target_columns(&self) -> Vec<EtthColumnName> {
        self.targets.clone()
    }
}
