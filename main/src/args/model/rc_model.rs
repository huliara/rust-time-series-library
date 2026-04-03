use crate::{
    args::{data::DataCommand, model::DisplayArgs},
    models::ngrc::NGRCArgs,
};
use clap::{Args, Subcommand};
use serde::{Deserialize, Serialize};

#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct NGRCCommand {
    #[command(subcommand)]
    pub data_config: DataCommand,
    #[command(flatten)]
    pub model_args: NGRCArgs,
}
#[derive(Debug, Clone, Deserialize, Serialize, Args)]
pub struct RCModelArgs {
    #[command(subcommand)]
    pub model_config: RCModelConfig,
}
#[derive(Subcommand, Debug, Clone, Deserialize, Serialize, strum::Display)]
pub enum RCModelConfig {
    NGRC(NGRCCommand),
}

impl RCModelConfig {
    pub fn data_config(&self) -> &DataCommand {
        match self {
            RCModelConfig::NGRC(cmd) => &cmd.data_config,
        }
    }
}

impl DisplayArgs for RCModelConfig {
    fn display_args(&self) -> String {
        match self {
            RCModelConfig::NGRC(cmd) => {
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
