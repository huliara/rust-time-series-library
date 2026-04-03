pub mod long_term_forecast;
pub mod loss;
pub mod plot_loss;
use burn::tensor::backend::AutodiffBackend;
use std::time::Instant;

use crate::{
    args::{data::DataCommand, model::DisplayArgs, time_lengths::TimeLengths, RootArgs},
    exp::{long_term_forecast::train::ExpConfig, plot_loss::plot_loss_for_experiment},
    models::traits::Forecast,
};

use lib::env_path::get_result_root_path;

pub(crate) trait Train<B: AutodiffBackend, M: Forecast<B>> {
    fn train(&self, model: M);
}

pub(crate) trait Infer<B: AutodiffBackend, M: Forecast<B>> {
    fn infer(&self, model: M);
}

pub fn run<B: AutodiffBackend, F: Forecast<B>, M: Train<B, F> + Infer<B, F>>(
    model: M,
    args: RootArgs,
    device: B::Device,
) {
    let data_config = args.model_config.data_config().clone();
    let detail_path = format!(
        "{}{}{}",
        &args.model_config.display_args(),
        data_config.inner_string(),
        &args.exp_config,
    );
    let result_path = format!(
        "{}/{}/{}/{}",
        get_result_root_path(),
        args.model_config,
        data_config,
        detail_path
    );

    std::fs::create_dir_all(&result_path).ok();
    let args_yaml = serde_yaml::to_string(&args).expect("Failed to serialize args to YAML");
    std::fs::write(format!("{}/args.yml", result_path), args_yaml)
        .expect("Failed to write args.yml");

    if !args.skip_training {
        let train_start = Instant::now();
        model.train(
            &result_path,
            args.exp_config.clone(),
            data_config.clone(),
            args.time_lengths.clone(),
            device.clone(),
        );
        let elapsed = train_start.elapsed();
        println!("Training finished in {:.3} seconds", elapsed.as_secs_f64());
    }
    plot_loss_for_experiment(&result_path).expect("Failed to plot loss curves");
    model.infer(
        &result_path,
        args.exp_config.clone(),
        args.time_lengths.clone(),
        data_config,
        device,
    );
}
