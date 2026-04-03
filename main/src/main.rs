#![recursion_limit = "512"]

mod activation;
mod args;
mod data;
mod exp;
mod layers;
mod models;
#[cfg(test)]
mod test_utils;

use args::exp::TaskName;
use args::{backend::Backend as ArgBackend, RootArgs};
use burn::backend::{Autodiff, Wgpu};
use clap::Parser;

use crate::args::model::ModelConfig;
use crate::exp::long_term_forecast::GradientForecastModel;
use crate::exp::run;

fn main() {
    let args = RootArgs::parse();
    args.model_config
        .data_config()
        .validate_targets_match_first_train_feature()
        .expect("Targets must match the first train feature");

    type Backend = Autodiff<Wgpu>;
    let device = Default::default();
    match args.task_name {
        TaskName::LongTermForecast => match args.model_config {
            ModelConfig::GradientModel(ref arg) => {
                if args.backend == ArgBackend::Wgpu {
                    let model = GradientForecastModel::<Backend>::new(
                        arg.model_config.clone(),
                        args.time_lengths.clone(),
                        &device,
                    );
                    run(model, args, device);
                }
            }
            ModelConfig::RCModel(arg) => {
                todo!()
            }
        },
        _ => todo!(),
    };
}
