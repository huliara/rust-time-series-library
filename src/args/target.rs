use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, ValueEnum, PartialEq, Eq, Deserialize, Serialize, strum::Display)]
pub enum EttFeature {
    HUFL,
    HULL,
    MUFL,
    MULL,
    LUFL,
    LULL,
    OT,
}
