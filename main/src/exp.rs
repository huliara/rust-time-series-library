pub mod long_term_forecast;
pub mod loss;
pub mod plot_loss;
use burn::tensor::backend::AutodiffBackend;
use std::time::Instant;

use crate::{
    args::{
        model::{gradient_model::GradientModelConfig, DisplayArgs},
        RootArgs,
    },
    exp::{
        long_term_forecast::{GradientForecastModel, LongTermForecastExp},
        plot_loss::plot_loss_for_experiment,
    },
};

use lib::env_path::get_result_root_path;

pub(crate) trait Train<B: AutodiffBackend> {
    fn train(&self, model: GradientForecastModel<B>);
}

pub(crate) trait Infer<B: AutodiffBackend> {
    fn infer(&self, model_config: GradientModelConfig);
}

pub fn run<B: AutodiffBackend>(
    model_config: GradientModelConfig,
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
    let exp = LongTermForecastExp {
        result_path: result_path.clone(),
        exp_config: args.exp_config.clone(),
        data_config: data_config.clone(),
        lengths: args.time_lengths.clone(),
        device: device.clone(),
    };
    if !args.skip_training {
        let train_start = Instant::now();
        let model = GradientForecastModel::<B>::new(
            model_config.clone(),
            args.time_lengths.clone(),
            &device,
        );
        exp.train(model);
        let elapsed = train_start.elapsed();
        println!("Training finished in {:.3} seconds", elapsed.as_secs_f64());
    }
    plot_loss_for_experiment(&result_path).expect("Failed to plot loss curves");
    exp.infer(model_config);
}
