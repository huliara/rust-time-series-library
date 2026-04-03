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

use crate::exp::run;

fn main() {
    let args = RootArgs::parse();
    args.model_command
        .data_config()
        .validate_targets_match_first_train_feature()
        .expect("Targets must match the first train feature");

    type Backend = Autodiff<Wgpu>;
    let device = Default::default();
    match args.task_name {
        TaskName::LongTermForecast => {
            if args.backend == ArgBackend::Wgpu {
                run::<Backend>(args.model_command.clone(), args, device);
            }
        }

        _ => todo!(),
    };
}
