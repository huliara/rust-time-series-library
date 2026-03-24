use clap::Subcommand;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::data::dataset::real_time_series::{etth1::Etth1Args, exchange::ExchangeArgs};

#[derive(Subcommand, Debug, Clone, Deserialize, Serialize, strum::Display)]
pub enum DataConfig {
    ETTh1(Etth1Args),
    Exchange(ExchangeArgs),
}

impl Default for DataConfig {
    fn default() -> Self {
        DataConfig::ETTh1(Etth1Args::default())
    }
}

impl DataConfig {
    pub fn input_dim(&self) -> usize {
        match self {
            DataConfig::ETTh1(cmd) => cmd.train_features.len(),
            DataConfig::Exchange(cmd) => cmd.train_features.len(),
        }
    }

    pub fn inner_string(&self) -> String {
        match self {
            DataConfig::ETTh1(cmd) => cmd.to_string(),
            DataConfig::Exchange(cmd) => cmd.to_string(),
        }
    }
    pub fn validate_targets_match_first_train_feature(&self) -> Result<(), String> {
        match self {
            DataConfig::ETTh1(cmd) => {
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
            DataConfig::Exchange(cmd) => {
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
