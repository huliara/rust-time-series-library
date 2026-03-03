pub mod activation;
pub mod backend;
pub mod column_name;
pub mod data_config;
pub mod exp;
pub mod model_config;
pub mod time_embed;
pub mod time_lengths;
use self::exp::TaskName;
use self::time_lengths::TimeLengths;
use crate::{
    args::{
        backend::Backend, column_name::ColumnName, data_config::DataConfig,
        model_config::ModelConfig,
    },
    exp::long_term_forecast::train::ExpConfig,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
#[derive(Parser, Debug, Clone, Deserialize, Serialize)]
#[command(name = "exp")]
#[command(author, version, about, long_about = None)]
pub struct RootArgs {
    #[command(subcommand)]
    pub model_config: ModelConfig,
    #[arg(long, value_enum)]
    pub task_name: TaskName,
    #[arg(long, value_enum)]
    pub backend: Backend,
    #[command(flatten)]
    pub data_config: DataConfig,
    #[command(flatten)]
    pub time_lengths: TimeLengths,
    #[command(flatten)]
    pub exp_config: ExpConfig,

    #[arg(long)]
    pub skip_training: bool,

    #[arg(long, default_value = "test")]
    pub model_id: String,

    #[arg(long, default_value = "./checkpoints/")]
    pub checkpoints: String,
}

impl RootArgs {
    pub fn assert_column_names(&self) {
        match self.data_config.data {
            crate::args::data_config::Data::ETTh1 => {
                for column in self
                    .data_config
                    .train_features
                    .iter()
                    .chain(self.data_config.targets.iter())
                {
                    assert!(
                        matches!(column, ColumnName::HUFL | ColumnName::HULL | ColumnName::MUFL | ColumnName::MULL | ColumnName::LUFL | ColumnName::LULL | ColumnName::OT),
                        "For ETTh1 and ETTh2 datasets, column names must be one of HUFL, HULL, MUFL, MULL, LUFL, LULL, OT"
                    );
                }
            }
            _ => {}
        }
    }
}
