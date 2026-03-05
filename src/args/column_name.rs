use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, ValueEnum, PartialEq, Eq, Deserialize, Serialize, strum::Display)]
pub enum ColumnName {
    #[strum(serialize = "HUFL")]
    Hufl,
    #[strum(serialize = "HULL")]
    Hull,
    #[strum(serialize = "MUFL")]
    Mufl,
    #[strum(serialize = "MULL")]
    Mull,
    #[strum(serialize = "LUFL")]
    Lufl,
    #[strum(serialize = "LULL")]
    Lull,
    #[strum(serialize = "OT")]
    Ot,
    #[strum(serialize = "open")]
    Open,
    #[strum(serialize = "high")]
    High,
    #[strum(serialize = "low")]
    Low,
    #[strum(serialize = "close")]
    Close,
    #[strum(serialize = "tick_volume")]
    TickVolume,
    #[strum(serialize = "spread")]
    Spread,
    #[strum(serialize = "real_volume")]
    RealVolume,
}
