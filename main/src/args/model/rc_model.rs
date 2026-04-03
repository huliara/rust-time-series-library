use crate::{
    args::{data::DataCommand, model::DisplayArgs},
    models::rc_model::ngrc::NGRCConfig,
};
use clap::{Args, Subcommand};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize, Args)]
pub struct RCModelArgs {
    #[command(subcommand)]
    pub model_config: RCModelCommand,
}
#[derive(Subcommand, Debug, Clone, Deserialize, Serialize, strum::Display)]
pub enum RCModelCommand {
    NGRC(NGRCCommand),
}

#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct NGRCCommand {
    #[command(subcommand)]
    pub data_config: DataCommand,
    #[command(flatten)]
    pub model_args: NGRCConfig,
}

impl RCModelCommand {
    pub fn data_config(&self) -> &DataCommand {
        match self {
            RCModelCommand::NGRC(cmd) => &cmd.data_config,
        }
    }
}

impl DisplayArgs for RCModelCommand {
    fn display_args(&self) -> String {
        match self {
            RCModelCommand::NGRC(cmd) => {
                let args = &cmd.model_args;
                format!(
                    "dl{}st{}pl{}rp{}tr{}bi{}lo{}",
                    args.delay,
                    args.stride,
                    args.poly_order,
                    args.ridge_param,
                    args.transients,
                    args.bias,
                    args.loss
                )
            }
        }
    }
}
