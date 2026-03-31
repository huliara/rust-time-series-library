use clap::Subcommand;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::data::dataset::{
    dynamic_system::{
        bool_transform::BoolTransformConfig, doublescroll::DoubleScrollConfig,
        henon_map::HenonMapConfig, kuramoto_sivashinsky::KuramotoSivashinskyConfig,
        logistic_map::LogisticMapConfig, lorenz::LorenzConfig, lorenz96::Lorenz96Config,
        mackey_glass::MackeyGlassConfig, multiscroll::MultiScrollConfig, narma::NarmaConfig,
        rabinovich_fabrikant::RabinovichFabrikantConfig, rossler::RosslerConfig,
    },
    real_time_series::{etth1::Etth1Config, exchange::ExchangeConfig},
};

#[derive(Subcommand, Debug, Clone, Deserialize, Serialize, strum::Display)]
pub enum DataCommand {
    ETTh1(Etth1Config),
    Exchange(ExchangeConfig),
    LogisticMap(LogisticMapConfig),
    HenonMap(HenonMapConfig),
    Lorenz(LorenzConfig),
    Lorenz96(Lorenz96Config),
    Rossler(RosslerConfig),
    DoubleScroll(DoubleScrollConfig),
    MultiScroll(MultiScrollConfig),
    RabinovichFabrikant(RabinovichFabrikantConfig),
    MackeyGlass(MackeyGlassConfig),
    Narma(NarmaConfig),
    KuramotoSivashinsky(KuramotoSivashinskyConfig),
    BoolTransform(BoolTransformConfig),
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
            DataCommand::LogisticMap(_) => 1,
            DataCommand::HenonMap(_) => 2,
            DataCommand::Lorenz(_) => 3,
            DataCommand::Lorenz96(cmd) => cmd.dimention,
            DataCommand::Rossler(_) => 3,
            DataCommand::DoubleScroll(_) => 3,
            DataCommand::MultiScroll(_) => 3,
            DataCommand::RabinovichFabrikant(_) => 3,
            DataCommand::MackeyGlass(_) => 1,
            DataCommand::Narma(_) => 1,
            DataCommand::KuramotoSivashinsky(cmd) => cmd.n,
            DataCommand::BoolTransform(_) => 1,
        }
    }

    pub fn inner_string(&self) -> String {
        match self {
            DataCommand::ETTh1(cmd) => cmd.to_string(),
            DataCommand::Exchange(cmd) => cmd.to_string(),
            DataCommand::LogisticMap(cmd) => cmd.to_string(),
            DataCommand::HenonMap(cmd) => cmd.to_string(),
            DataCommand::Lorenz(cmd) => cmd.to_string(),
            DataCommand::Lorenz96(cmd) => cmd.to_string(),
            DataCommand::Rossler(cmd) => cmd.to_string(),
            DataCommand::DoubleScroll(cmd) => cmd.to_string(),
            DataCommand::MultiScroll(cmd) => cmd.to_string(),
            DataCommand::RabinovichFabrikant(cmd) => cmd.to_string(),
            DataCommand::MackeyGlass(cmd) => cmd.to_string(),
            DataCommand::Narma(cmd) => cmd.to_string(),
            DataCommand::KuramotoSivashinsky(cmd) => cmd.to_string(),
            DataCommand::BoolTransform(cmd) => cmd.to_string(),
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
            DataCommand::LogisticMap(_) => Ok(()),
            DataCommand::HenonMap(_) => Ok(()),
            DataCommand::Lorenz(_) => Ok(()),
            DataCommand::Lorenz96(_) => Ok(()),
            DataCommand::Rossler(_) => Ok(()),
            DataCommand::DoubleScroll(_) => Ok(()),
            DataCommand::MultiScroll(_) => Ok(()),
            DataCommand::RabinovichFabrikant(_) => Ok(()),
            DataCommand::MackeyGlass(_) => Ok(()),
            DataCommand::Narma(_) => Ok(()),
            DataCommand::KuramotoSivashinsky(_) => Ok(()),
            DataCommand::BoolTransform(_) => Ok(()),
        }
    }
}
