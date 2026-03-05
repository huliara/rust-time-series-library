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
    #[arg(long, num_args = 1..)]
    pub train_features: Vec<ColumnName>,

    #[arg(long, num_args = 1..)]
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
                ColumnName::Hufl,
                ColumnName::Hull,
                ColumnName::Mufl,
                ColumnName::Mull,
                ColumnName::Lufl,
                ColumnName::Lull,
                ColumnName::Ot,
            ],
            targets: vec![
                ColumnName::Hufl,
                ColumnName::Hull,
                ColumnName::Mufl,
                ColumnName::Mull,
                ColumnName::Lufl,
                ColumnName::Lull,
                ColumnName::Ot,
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

impl DataConfig {
    pub fn assert_column_names(&self) {
        assert!(
            !self.train_features.is_empty(),
            "train_features must contain at least one column name"
        );
        assert!(
            !self.targets.is_empty(),
            "targets must contain at least one column name"
        );

        match self.data {
            Data::ETTh1 => {
                for column in self.train_features.iter().chain(self.targets.iter()) {
                    assert!(
                        matches!(column, ColumnName::Hufl | ColumnName::Hull | ColumnName::Mufl | ColumnName::Mull | ColumnName::Lufl | ColumnName::Lull | ColumnName::Ot),
                        "For ETTh1 and ETTh2 datasets, column names must be one of HUFL, HULL, MUFL, MULL, LUFL, LULL, OT"
                    );
                }
            }
            Data::Exchange => {
                for column in self.train_features.iter().chain(self.targets.iter()) {
                    assert!(
                        matches!(column, ColumnName::Open | ColumnName::High | ColumnName::Low | ColumnName::Close | ColumnName::TickVolume | ColumnName::Spread | ColumnName::RealVolume),
                        "For Exchange dataset, column names must be one of open, high, low, close, tick_volume, spread, real_volume"
                    );
                }
            }
        }
    }
}
