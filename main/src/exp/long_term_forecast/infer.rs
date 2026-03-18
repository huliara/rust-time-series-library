use std::fs;

use crate::{
    args::{data_config::DataConfig, model_config::ModelConfig, time_lengths::TimeLengths},
    data::{data_loader::create_data_loader, dataset::time_series_dataset::ExpFlag},
    exp::{
        long_term_forecast::{
            save_results::{plot_multi_feature_prediction, save_results},
            train::ExpConfig,
            ForecastModel,
        },
        Infer,
    },
    models::traits::Forecast,
};
use burn::{
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::backend::AutodiffBackend,
};

impl<B: AutodiffBackend> Infer<B> for ForecastModel<B> {
    fn infer(
        &self,
        exp_root_path: &str,
        exp_config: ExpConfig,
        model_config: ModelConfig,
        lengths: TimeLengths,
        data_config: DataConfig,
        device: B::Device,
    ) {
        let record = CompactRecorder::new()
            .load(format!("{exp_root_path}/model").into(), &device)
            .expect("Trained model should exist; run train first");

        let model: ForecastModel<B> =
            ForecastModel::<B>::new(model_config, lengths.clone(), &device).load_record(record);
        let dataloader_test = create_data_loader(
            &data_config,
            &lengths,
            exp_config.batch_size,
            exp_config.num_workers,
            exp_config.seed,
            ExpFlag::Test,
        );
        let mut _contexts = Vec::with_capacity(3);
        let mut _predicts = Vec::with_capacity(3);
        let mut _futures = Vec::with_capacity(3);
        fs::create_dir_all(format!("{exp_root_path}/test/")).unwrap();

        for batch in dataloader_test.iter() {
            let output =
                model.forecast(batch.x.clone(), batch.x_mark, batch.y.clone(), batch.y_mark);
            _contexts.push(batch.x);
            _predicts.push(output);
            _futures.push(batch.y);
        }

        let contexts = Tensor::cat(_contexts, 0);
        let predicts = Tensor::cat(_predicts, 0);
        let futures = Tensor::cat(_futures, 0);
        let error = predicts.clone() - futures.clone();
        let sample_count = contexts.dims()[0];
        let feature_count = contexts.dims()[2];
        let num_plots = usize::min(10, sample_count);
        let plot_step = usize::max(1, sample_count / num_plots);

        for i in 0..num_plots {
            let sample_idx = usize::min(i * plot_step, sample_count - 1);

            let mut context_multi = Vec::with_capacity(feature_count);
            let mut pred_multi = Vec::with_capacity(feature_count);
            let mut future_multi = Vec::with_capacity(feature_count);

            for feature_idx in 0..feature_count {
                let context_vec = contexts
                    .clone()
                    .slice(s![sample_idx, .., feature_idx])
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                let pred_vec = predicts
                    .clone()
                    .slice(s![sample_idx, .., feature_idx])
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                let future_vec = futures
                    .clone()
                    .slice(s![sample_idx, .., feature_idx])
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();

                context_multi.push(context_vec);
                pred_multi.push(pred_vec);
                future_multi.push(future_vec);
            }

            plot_multi_feature_prediction(
                exp_root_path,
                i + 1000,
                &context_multi,
                &pred_multi,
                &future_multi,
            );
        }

        save_results(exp_root_path, error, futures);
    }
}
