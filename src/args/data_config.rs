use crate::args::{column_name::ColumnName, time_embed::TimeEmbed};
use clap::{Args, ValueEnum};
use core::fmt;
use serde::{Deserialize, Serialize};
#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct DataConfig {
    #[arg(long, value_enum)]
    pub data: Data,
    //corresponds to features
    #[arg(long)]
    pub path: String,
    #[arg(long)]
    pub train_features: Vec<ColumnName>,

    #[arg(long)]
    pub targets: Vec<ColumnName>,

    #[arg(long, value_enum)]
    pub embed: TimeEmbed,
}
impl Default for DataConfig {
    fn default() -> Self {
        Self {
            data: Data::ETTh1,
            path: "ETT/ETTh1.csv".to_string(),
            train_features: vec![
                ColumnName::HUFL,
                ColumnName::HULL,
                ColumnName::MUFL,
                ColumnName::MULL,
                ColumnName::LUFL,
                ColumnName::LULL,
                ColumnName::OT,
            ],
            targets: vec![
                ColumnName::HUFL,
                ColumnName::HULL,
                ColumnName::MUFL,
                ColumnName::MULL,
                ColumnName::LUFL,
                ColumnName::LULL,
                ColumnName::OT,
            ],
            embed: TimeEmbed::TimeF,
        }
    }
}

impl fmt::Display for DataConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}_{}_{}",
            self.data,
            self.targets
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join("_"),
            self.train_features
                .iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>()
                .join("_")
        )
    }
}
#[derive(Debug, Clone, ValueEnum, Deserialize, Serialize, strum::Display)]
pub enum Data {
    ETTh1,
    Exchange,
}
