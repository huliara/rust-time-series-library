use crate::args::{target::EttFeature, time_embed::TimeEmbed};
use clap::{Args, ValueEnum};
use core::fmt;
use serde::{Deserialize, Serialize};
#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct DataConfig {
    #[arg(long, value_enum)]
    pub data: Data,
    //corresponds to features
    #[arg(long)]
    pub train_features: Vec<EttFeature>,

    #[arg(long)]
    pub targets: Vec<EttFeature>,

    #[arg(long, value_enum)]
    pub embed: TimeEmbed,
}
impl Default for DataConfig {
    fn default() -> Self {
        Self {
            data: Data::ETTh1,
            train_features: vec![
                EttFeature::HUFL,
                EttFeature::HULL,
                EttFeature::MUFL,
                EttFeature::MULL,
                EttFeature::LUFL,
                EttFeature::LULL,
                EttFeature::OT,
            ],
            targets: vec![
                EttFeature::HUFL,
                EttFeature::HULL,
                EttFeature::MUFL,
                EttFeature::MULL,
                EttFeature::LUFL,
                EttFeature::LULL,
                EttFeature::OT,
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
}
