mod activation;
mod args;
mod data;
mod exp;
mod layers;
mod models;
mod test_utils;

use args::exp::TaskName;
use args::{backend::Backend as ArgBackend, RootArgs};
use burn::backend::{Autodiff, Wgpu};
use clap::Parser;

use crate::exp::{long_term_forecast::ForecastModel, Exp};

fn main() {
    let args = RootArgs::parse();
    args.model_config.data_config().assert_column_names();
    type Backend = Autodiff<Wgpu>;
    let device = Default::default();
    match args.task_name {
        TaskName::LongTermForecast => {
            if args.backend == ArgBackend::Wgpu {
                let model = ForecastModel::<Backend>::new(
                    args.model_config.clone(),
                    args.time_lengths.clone(),
                    &device,
                );
                model.run(args, device);
            }
        }
        _ => todo!(),
    };
}
