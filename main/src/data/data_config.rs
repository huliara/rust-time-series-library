pub mod etth1;
pub mod exchange;
pub mod init_dataset;
use crate::data::data_config::{etth1::Etth1Args, exchange::ExchangeArgs};
use clap::Subcommand;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

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

    pub fn validate_targets_match_first_train_feature(&self) -> Result<(), String> {
        match self {
            DataConfig::ETTh1(cmd) => {
                Self::validate_target_prefix(&cmd.targets, &cmd.train_features, "ETTh1")
            }
            DataConfig::Exchange(cmd) => {
                Self::validate_target_prefix(&cmd.targets, &cmd.train_features, "Exchange")
            }
        }
    }

    fn validate_target_prefix<C: PartialEq + Debug>(
        targets: &[C],
        train_features: &[C],
        dataset_name: &str,
    ) -> Result<(), String> {
        if targets.len() > train_features.len() {
            return Err(format!(
                "{dataset_name}: targets length ({}) exceeds train_features length ({})",
                targets.len(),
                train_features.len()
            ));
        }

        if targets != &train_features[..targets.len()] {
            return Err(format!(
                "{dataset_name}: targets {:?} do not match the first train features {:?}",
                targets,
                &train_features[..targets.len()]
            ));
        }

        Ok(())
    }

    pub fn inner_string(&self) -> String {
        match self {
            DataConfig::ETTh1(cmd) => cmd.to_string(),
            DataConfig::Exchange(cmd) => cmd.to_string(),
        }
    }
}
