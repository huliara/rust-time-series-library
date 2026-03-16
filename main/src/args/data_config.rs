use crate::args::{
    column_name::{EtthColumnName, ExchangeColumnName},
    time_embed::TimeEmbed,
};
use clap::{Args, Subcommand};
use core::fmt;
use serde::{de, Deserialize, Serialize};

#[derive(Subcommand, Debug, Clone, Deserialize, Serialize)]
pub enum DataConfig {
    ETTh1(Etth1Command),
    Exchange(ExchangeCommand),
}

impl Default for DataConfig {
    fn default() -> Self {
        DataConfig::ETTh1(Etth1Command::default())
    }
}

#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct Etth1Command {
    #[arg(long)]
    pub path: String,
    #[arg(long, num_args = 1..)]
    pub train_features: Vec<EtthColumnName>,

    #[arg(long, num_args = 1..)]
    pub targets: Vec<EtthColumnName>,

    #[arg(long, value_enum)]
    pub embed: TimeEmbed,
}

#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct ExchangeCommand {
    //corresponds to features
    #[arg(long)]
    pub path: String,
    #[arg(long, num_args = 1..)]
    pub train_features: Vec<ExchangeColumnName>,

    #[arg(long, num_args = 1..)]
    pub targets: Vec<ExchangeColumnName>,

    #[arg(long, value_enum)]
    pub embed: TimeEmbed,
}

impl Default for Etth1Command {
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

impl fmt::Display for DataConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataConfig::ETTh1(cmd) => {
                write!(
                    f,
                    "path: {}, train_features: {:?}, targets: {:?}, embed: {}",
                    cmd.path, cmd.train_features, cmd.targets, cmd.embed
                )
            }
            DataConfig::Exchange(cmd) => {
                write!(
                    f,
                    "path: {}, train_features: {:?}, targets: {:?}, embed: {}",
                    cmd.path, cmd.train_features, cmd.targets, cmd.embed
                )
            }
        }
    }
}
