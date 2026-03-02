pub mod long_term_forecast;
use burn::tensor::backend::AutodiffBackend;
use std::time::Instant;

use crate::{
    args::{
        data_config::DataConfig, model_config::ModelConfig, time_lengths::TimeLengths, RootArgs,
    },
    env_path::get_result_root_path,
    exp::long_term_forecast::train::ExpConfig,
};

fn get_model_args_string(model_config: &ModelConfig) -> String {
    match model_config {
        ModelConfig::PatchTST(args) => {
            format!(
                "dm{}nh{}el{}df{}pt{}st{}ei{}do{}ac{}",
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_ff,
                args.patch_len,
                args.stride,
                args.enc_in,
                args.dropout,
                args.activation,
            )
        }
        ModelConfig::DLinear(args) => {
            format!(
                "ei{}ind{}ma{}",
                args.enc_in, args.individual, args.moving_avg,
            )
        }
    }
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub(crate) trait Train<B: AutodiffBackend> {
    fn train(
        &self,
        result_path: &str,
        exp_config: ExpConfig,
        model_config: ModelConfig,
        data_config: DataConfig,
        lengths: TimeLengths,
        device: B::Device,
    );
}

pub(crate) trait Infer<B: AutodiffBackend> {
    fn infer(
        &self,
        exp_root_path: &str,
        exp_config: ExpConfig,
        model_config: ModelConfig,
        lengths: TimeLengths,
        data_config: DataConfig,
        device: B::Device,
    );
}

pub trait Exp<B: AutodiffBackend>: Train<B> + Infer<B> {
    fn run(&self, args: RootArgs, device: B::Device) {
        let result_path = format!(
            "{}/{}/{}/{}",
            get_result_root_path(),
            args.model_config,
            args.data_config,
            get_model_args_string(&args.model_config)
        );

        std::fs::create_dir_all(&result_path).ok();
        let args_yaml = serde_yaml::to_string(&args).expect("Failed to serialize args to YAML");
        std::fs::write(format!("{}/args.yml", result_path), args_yaml)
            .expect("Failed to write args.yml");

        if !args.skip_training {
            let train_start = Instant::now();
            self.train(
                &result_path,
                args.exp_config.clone(),
                args.model_config.clone(),
                args.data_config.clone(),
                args.time_lengths.clone(),
                device.clone(),
            );
            let elapsed = train_start.elapsed();
            println!("Training finished in {:.3} seconds", elapsed.as_secs_f64());
        }
        self.infer(
            &result_path,
            args.exp_config.clone(),
            args.model_config.clone(),
            args.time_lengths.clone(),
            args.data_config.clone(),
            device,
        );
    }
}
