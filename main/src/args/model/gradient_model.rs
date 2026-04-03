use crate::{
    args::{data::DataCommand, model::DisplayArgs},
    models::gradient_model::{
        dlinear::DLinearArgs, patch_tst::PatchTSTArgs, time_xer::TimeXerArgs,
    },
};
use clap::{Args, Subcommand};
use serde::{Deserialize, Serialize};

#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct PatchTSTCommand {
    #[command(subcommand)]
    pub data_config: DataCommand,
    #[command(flatten)]
    pub model_args: PatchTSTArgs,
}

#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct DLinearCommand {
    #[command(subcommand)]
    pub data_config: DataCommand,
    #[command(flatten)]
    pub model_args: DLinearArgs,
}

#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct TimeXerCommand {
    #[command(subcommand)]
    pub data_config: DataCommand,
    #[command(flatten)]
    pub model_args: TimeXerArgs,
}

#[derive(Debug, Clone, Deserialize, Serialize, Args)]
pub struct GradientModelArgs {
    #[command(subcommand)]
    pub model_config: GradientModelConfig,
}

#[derive(Subcommand, Debug, Clone, Deserialize, Serialize, strum::Display)]
pub enum GradientModelConfig {
    #[strum(serialize = "PatchTST")]
    PatchTST(PatchTSTCommand),
    #[strum(serialize = "DLinear")]
    DLinear(DLinearCommand),
    #[strum(serialize = "TimeXer")]
    TimeXer(TimeXerCommand),
}

impl GradientModelConfig {
    pub fn data_config(&self) -> &DataCommand {
        match self {
            GradientModelConfig::PatchTST(cmd) => &cmd.data_config,
            GradientModelConfig::DLinear(cmd) => &cmd.data_config,
            GradientModelConfig::TimeXer(cmd) => &cmd.data_config,
        }
    }
}

impl DisplayArgs for GradientModelConfig {
    fn display_args(&self) -> String {
        match self {
            GradientModelConfig::PatchTST(cmd) => {
                format!(
                    "dm{}nh{}el{}df{}pt{}st{}ei{}do{}ac{}",
                    cmd.model_args.d_model,
                    cmd.model_args.n_heads,
                    cmd.model_args.e_layers,
                    cmd.model_args.d_ff,
                    cmd.model_args.patch_len,
                    cmd.model_args.stride,
                    cmd.model_args.enc_in,
                    cmd.model_args.dropout,
                    cmd.model_args.activation,
                )
            }
            GradientModelConfig::DLinear(cmd) => format!(
                "ei{}ind{}ma{}",
                cmd.model_args.enc_in, cmd.model_args.individual, cmd.model_args.moving_avg,
            ),
            GradientModelConfig::TimeXer(cmd) => format!(
                "dm{}nh{}el{}df{}pt{}do{}ac{}",
                cmd.model_args.d_model,
                cmd.model_args.n_heads,
                cmd.model_args.e_layers,
                cmd.model_args.d_ff,
                cmd.model_args.patch_len,
                cmd.model_args.dropout,
                cmd.model_args.activation,
            ),
        }
    }
}
