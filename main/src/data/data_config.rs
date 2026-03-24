pub mod etth1;
pub mod exchange;
pub mod init_dataset;
use crate::{
    args::time_embed::TimeEmbed,
    data::{
        column_name::{EtthColumnName, ExchangeColumnName},
        data_config::{etth1::Etth1Args, exchange::ExchangeArgs},
        dataset::dynamic_system::lorenz96::DynamicSystemArgs,
    },
};
use clap::{Args, Subcommand, ValueEnum};
use core::fmt;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};

#[derive(Subcommand, Debug, Clone, Deserialize, Serialize, strum::Display)]
pub enum DataConfig {
    ETTh1(Etth1Args),
    Exchange(ExchangeArgs),
}

impl Default for DataConfig {
    fn default() -> Self {
        DataConfig::ETTh1(Etth1Args::default())
    }
}

impl DataConfig {
    pub fn input_dim(&self) -> usize {
        match self {
            DataConfig::ETTh1(cmd) => cmd.train_features.len(),
            DataConfig::Exchange(cmd) => cmd.train_features.len(),
        }
    }
    pub fn inner_string(&self) -> String {
        match self {
            DataConfig::ETTh1(cmd) => cmd.to_string(),
            DataConfig::Exchange(cmd) => cmd.to_string(),
        }
    }
}

#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct DataArgs<
    C: Clone + std::marker::Send + std::marker::Sync + 'static + ValueEnum + Display + Debug,
> {
    #[arg(long)]
    pub path: String,
    #[arg(long, num_args = 1..)]
    pub train_features: Vec<C>,

    #[arg(long, num_args = 1..)]
    pub targets: Vec<C>,

    #[arg(long, value_enum)]
    pub embed: TimeEmbed,
}

impl<C> DataArgs<C> where
    C: Clone
        + std::marker::Send
        + std::marker::Sync
        + 'static
        + ValueEnum
        + Display
        + PartialEq
        + Debug
{
}

impl Default for DataArgs<EtthColumnName> {
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

impl<C: Clone + std::marker::Send + std::marker::Sync + 'static + ValueEnum + Display + Debug>
    fmt::Display for DataArgs<C>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}_{}_{}",
            self.targets
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join("_"),
            self.train_features
                .iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>()
                .join("_"),
            self.embed
        )
    }
}
