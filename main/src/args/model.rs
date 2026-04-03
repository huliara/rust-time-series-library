pub mod gradient_model;
pub mod rc_model;
use crate::args::{
    data::DataCommand,
    model::{gradient_model::GradientModelArgs, rc_model::RCModelArgs},
};
use clap::Subcommand;
use serde::{Deserialize, Serialize};

#[derive(Subcommand, Debug, Clone, Deserialize, Serialize, strum::Display)]
pub enum ModelCommand {
    GradientModel(GradientModelArgs),
    RCModel(RCModelArgs),
}

impl ModelCommand {
    pub fn data_config(&self) -> &DataCommand {
        match self {
            ModelCommand::GradientModel(cmd) => cmd.model_command.data_config(),
            ModelCommand::RCModel(cmd) => cmd.model_config.data_config(),
        }
    }
}

pub trait DisplayArgs {
    fn display_args(&self) -> String;
}

impl DisplayArgs for ModelCommand {
    fn display_args(&self) -> String {
        match self {
            ModelCommand::GradientModel(cmd) => cmd.model_command.display_args(),
            ModelCommand::RCModel(cmd) => cmd.model_config.display_args(),
        }
    }
}
