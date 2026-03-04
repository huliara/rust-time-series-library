use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, ValueEnum, PartialEq, Eq, Deserialize, Serialize, strum::Display)]
pub enum ColumnName {
    HUFL,
    HULL,
    MUFL,
    MULL,
    LUFL,
    LULL,
    OT,
    open,
    high,
    low,
    close,
    tick_volume,
    spread,
    real_volume,
}
