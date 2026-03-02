use std::fs;

use crate::{
    args::{data_config::DataConfig, model_config::ModelConfig, time_lengths::TimeLengths},
    data::{data_loader::create_data_loader, dataset::ett_hour::ExpFlag},
    exp::{
        long_term_forecast::{
            save_results::{plot_single_prediction, save_results},
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

        for (i, batch) in dataloader_test.iter().enumerate() {
            let output =
                model.forecast(batch.x.clone(), batch.x_mark, batch.y.clone(), batch.y_mark);
            if i % 20 == 0 {
                let feature_idx = batch.x.dims()[2] - 1;
                let context_vec = batch
                    .x
                    .clone()
                    .slice(s![0, .., feature_idx])
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                let pred_vec = output
                    .clone()
                    .slice(s![0, .., feature_idx])
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                let future_vec = batch
                    .y
                    .clone()
                    .slice(s![0, .., feature_idx])
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                plot_single_prediction(exp_root_path, i, &context_vec, &pred_vec, &future_vec);
            }
            _contexts.push(batch.x);
            _predicts.push(output);
            _futures.push(batch.y);
        }
        let contexts = Tensor::cat(_contexts, 0);
        let predicts = Tensor::cat(_predicts, 0);
        let futures = Tensor::cat(_futures, 0);
        let error = predicts.clone() - futures.clone();
        save_results(exp_root_path, error, contexts);
    }
}
