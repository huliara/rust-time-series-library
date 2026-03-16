use crate::{
    args::data_config::DataConfig,
    models::{dlinear::DLinearArgs, patch_tst::PatchTSTArgs, time_xer::TimeXerArgs},
};
use clap::{Args, Subcommand};
use serde::{Deserialize, Serialize};

#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct PatchTSTCommand {
    #[command(subcommand)]
    pub data_config: DataConfig,
    #[command(flatten)]
    pub model_args: PatchTSTArgs,
}

#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct DLinearCommand {
    #[command(subcommand)]
    pub data_config: DataConfig,
    #[command(flatten)]
    pub model_args: DLinearArgs,
}

#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct TimeXerCommand {
    #[command(subcommand)]
    pub data_config: DataConfig,
    #[command(flatten)]
    pub model_args: TimeXerArgs,
}

#[derive(Subcommand, Debug, Clone, Deserialize, Serialize, strum::Display)]
pub enum ModelConfig {
    #[strum(serialize = "PatchTST")]
    PatchTST(PatchTSTCommand),
    #[strum(serialize = "DLinear")]
    DLinear(DLinearCommand),
    #[strum(serialize = "TimeXer")]
    TimeXer(TimeXerCommand),
}

impl ModelConfig {
    pub fn data_config(&self) -> &DataConfig {
        match self {
            ModelConfig::PatchTST(cmd) => &cmd.data_config,
            ModelConfig::DLinear(cmd) => &cmd.data_config,
            ModelConfig::TimeXer(cmd) => &cmd.data_config,
        }
    }
}
