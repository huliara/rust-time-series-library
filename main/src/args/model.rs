pub mod gradient_model;
pub mod rc_model;
use crate::args::{
    data::DataCommand,
    model::{gradient_model::GradientModelArgs, rc_model::RCModelArgs},
};
use clap::Subcommand;
use serde::{Deserialize, Serialize};

#[derive(Subcommand, Debug, Clone, Deserialize, Serialize, strum::Display)]
pub enum ModelConfig {
    GradientModel(GradientModelArgs),
    RCModel(RCModelArgs),
}

impl ModelConfig {
    pub fn data_config(&self) -> &DataCommand {
        match self {
            ModelConfig::GradientModel(cmd) => cmd.model_config.data_config(),
            ModelConfig::RCModel(cmd) => &cmd.model_config.data_config(),
        }
    }
}

pub trait DisplayArgs {
    fn display_args(&self) -> String;
}

impl DisplayArgs for ModelConfig {
    fn display_args(&self) -> String {
        match self {
            ModelConfig::GradientModel(cmd) => cmd.model_config.display_args(),
            ModelConfig::RCModel(cmd) => cmd.model_config.display_args(),
        }
    }
}
