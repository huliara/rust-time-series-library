use crate::{
    args::{data_config::DataConfig, model_config::ModelConfig, time_lengths::TimeLengths},
    data::{data_loader::create_data_loader, dataset::time_series_dataset::ExpFlag},
    exp::{
        long_term_forecast::{save_results::plot_multi_feature_prediction, ForecastModel},
        Train,
    },
    models::traits::Forecast,
};
use burn::{
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{
            store::{Aggregate, Direction, Split},
            LossMetric,
        },
        Learner, MetricEarlyStoppingStrategy, StoppingCondition, SupervisedTraining,
    },
};
use clap::Args;
use serde::{Deserialize, Serialize};
use std::fs;
#[derive(Debug, Args, Clone, Deserialize, Serialize)]
pub struct ExpConfig {
    #[arg(long, default_value_t = 10)]
    pub num_epochs: usize,
    #[arg(long, default_value_t = 32)]
    pub batch_size: usize,
    #[arg(long, default_value_t = 4)]
    pub num_workers: usize,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    #[arg(long, default_value_t = 1.0e-4)]
    pub learning_rate: f64,
}

impl<B: AutodiffBackend> Train<B> for ForecastModel<B> {
    fn train(
        &self,
        result_path: &str,
        exp_config: ExpConfig,
        model_config: ModelConfig,
        data_config: DataConfig,
        lengths: TimeLengths,
        device: B::Device,
    ) where
        B: AutodiffBackend,
    {
        B::seed(&device, exp_config.seed);

        let dataloader_train = create_data_loader(
            &data_config,
            &lengths,
            exp_config.batch_size,
            exp_config.num_workers,
            exp_config.seed,
            ExpFlag::Train,
        );

        let dataloader_valid = create_data_loader(
            &data_config,
            &lengths,
            exp_config.batch_size,
            exp_config.num_workers,
            exp_config.seed,
            ExpFlag::Val,
        );

        let loss = LossMetric::new();

        let stopping_strategy = MetricEarlyStoppingStrategy::new(
            &loss,
            Aggregate::Mean,
            Direction::Lowest,
            Split::Train,
            StoppingCondition::NoImprovementSince { n_epochs: 5 },
        );

        let training = SupervisedTraining::new(result_path, dataloader_train, dataloader_valid)
            .metrics((loss,))
            .early_stopping(stopping_strategy)
            .with_file_checkpointer(CompactRecorder::new())
            .num_epochs(exp_config.num_epochs)
            .summary();
        let optimizer = AdamConfig::new().init();
        let model = ForecastModel::<B>::new(model_config, lengths.clone(), &device);
        let result = training.launch(Learner::new(model, optimizer, exp_config.learning_rate));

        // Plot a few training samples right after training for quick sanity checks.
        let train_plot_dir = format!("{result_path}/train");
        fs::create_dir_all(&train_plot_dir).unwrap();
        let dataloader_train_plot = create_data_loader(
            &data_config,
            &lengths,
            exp_config.batch_size,
            exp_config.num_workers,
            exp_config.seed,
            ExpFlag::Train,
        );

        if let Some(batch) = dataloader_train_plot.iter().next() {
            let contexts = batch.x.clone();
            let futures = batch.y.clone();
            let predicts = result
                .model
                .forecast(batch.x, batch.x_mark, batch.y, batch.y_mark);

            let num_plots = usize::min(5, contexts.dims()[0]);
            let feature_count = contexts.dims()[2];

            for i in 0..num_plots {
                let mut context_multi = Vec::with_capacity(feature_count);
                let mut pred_multi = Vec::with_capacity(feature_count);
                let mut future_multi = Vec::with_capacity(feature_count);

                for feature_idx in 0..feature_count {
                    let context_vec = contexts
                        .clone()
                        .slice(s![i, .., feature_idx])
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap();
                    let pred_vec = predicts
                        .clone()
                        .slice(s![i, .., feature_idx])
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap();
                    let future_vec = futures
                        .clone()
                        .slice(s![i, .., feature_idx])
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap();

                    context_multi.push(context_vec);
                    pred_multi.push(pred_vec);
                    future_multi.push(future_vec);
                }

                plot_multi_feature_prediction(
                    &train_plot_dir,
                    i,
                    &context_multi,
                    &pred_multi,
                    &future_multi,
                );
            }
        }

        result
            .model
            .save_file(format!("{result_path}/model"), &CompactRecorder::new())
            .expect("Trained model should be saved successfully");
    }
}
