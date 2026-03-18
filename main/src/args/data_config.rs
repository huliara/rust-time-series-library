use crate::args::{
    column_name::{EtthColumnName, ExchangeColumnName},
    time_embed::TimeEmbed,
};
use clap::{Args, Subcommand, ValueEnum};
use core::fmt;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};

#[derive(Subcommand, Debug, Clone, Deserialize, Serialize, strum::Display)]
pub enum DataConfig {
    ETTh1(DataCommand<EtthColumnName>),
    Exchange(DataCommand<ExchangeColumnName>),
}

impl Default for DataConfig {
    fn default() -> Self {
        DataConfig::ETTh1(DataCommand::default())
    }
}

impl DataConfig {
    pub fn input_dim(&self) -> usize {
        match self {
            DataConfig::ETTh1(cmd) => cmd.train_features.len(),
            DataConfig::Exchange(cmd) => cmd.train_features.len(),
        }
    }

    pub fn validate_targets_match_first_train_feature(&self) -> Result<(), String> {
        match self {
            DataConfig::ETTh1(cmd) => cmd
                .validate_feature_order()
                .map_err(|msg| format!("DataConfig validation failed: {msg}")),
            DataConfig::Exchange(cmd) => cmd
                .validate_feature_order()
                .map_err(|msg| format!("DataConfig validation failed: {msg}")),
        }
    }
}

#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct DataCommand<
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

impl<C> DataCommand<C>
where
    C: Clone
        + std::marker::Send
        + std::marker::Sync
        + 'static
        + ValueEnum
        + Display
        + PartialEq
        + Debug,
{
    pub fn validate_feature_order(&self) -> Result<(), String> {
        if self.targets != self.train_features[..self.targets.len()] {
            return Err(format!(
                "Targets {:?} do not match the first train features {:?}. Ensure that the order of train features' head is the same as the target features .",
                self.targets,
                self.train_features[..self.targets.len()].to_vec()
            ));
        }

        Ok(())
    }
}

impl Default for DataCommand<EtthColumnName> {
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
    fmt::Display for DataCommand<C>
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
