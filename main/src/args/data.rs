use clap::Subcommand;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::data::dataset::real_time_series::{etth1::Etth1Config, exchange::ExchangeConfig};

#[derive(Subcommand, Debug, Clone, Deserialize, Serialize, strum::Display)]
pub enum DataCommand {
    ETTh1(Etth1Config),
    Exchange(ExchangeConfig),
}

impl Default for DataCommand {
    fn default() -> Self {
        DataCommand::ETTh1(Etth1Config::default())
    }
}

impl DataCommand {
    pub fn input_dim(&self) -> usize {
        match self {
            DataCommand::ETTh1(cmd) => cmd.train_features.len(),
            DataCommand::Exchange(cmd) => cmd.train_features.len(),
        }
    }

    pub fn inner_string(&self) -> String {
        match self {
            DataCommand::ETTh1(cmd) => cmd.to_string(),
            DataCommand::Exchange(cmd) => cmd.to_string(),
        }
    }
    pub fn validate_targets_match_first_train_feature(&self) -> Result<(), String> {
        match self {
            DataCommand::ETTh1(cmd) => {
                if cmd.targets.is_empty() {
                    return Err("Targets cannot be empty".to_string());
                }
                if cmd.train_features.is_empty() {
                    return Err("Train features cannot be empty".to_string());
                }
                if cmd.targets != cmd.train_features[0..cmd.targets.len()] {
                    return Err(format!(
                        "First target ({:?}) does not match first train feature ({:?})",
                        cmd.targets,
                        cmd.train_features[0..cmd.targets.len()].to_vec()
                    ));
                }
                Ok(())
            }
            DataCommand::Exchange(cmd) => {
                if cmd.targets.is_empty() {
                    return Err("Targets cannot be empty".to_string());
                }
                if cmd.train_features.is_empty() {
                    return Err("Train features cannot be empty".to_string());
                }
                if cmd.targets != cmd.train_features[0..cmd.targets.len()] {
                    return Err(format!(
                        "First target ({:?}) does not match first train feature ({:?})",
                        cmd.targets,
                        cmd.train_features[0..cmd.targets.len()].to_vec()
                    ));
                }
                Ok(())
            }
        }
    }
}
