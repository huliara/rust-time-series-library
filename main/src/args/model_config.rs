use crate::models::{dlinear::DLinearArgs, patch_tst::PatchTSTArgs, time_xer::TimeXerArgs};
use clap::Subcommand;
use serde::{Deserialize, Serialize};
#[derive(Subcommand, Debug, Clone, Deserialize, Serialize, strum::Display)]
pub enum ModelConfig {
    #[strum(serialize = "PatchTST")]
    PatchTST(PatchTSTArgs),
    #[strum(serialize = "DLinear")]
    DLinear(DLinearArgs),
    #[strum(serialize = "TimeXer")]
    TimeXer(TimeXerArgs),
}
