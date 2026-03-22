use std::fs;

use crate::{
    args::{data_config::DataConfig, model_config::ModelConfig, time_lengths::TimeLengths},
    data::{data_loader::create_data_loader, dataset::time_series_dataset::ExpFlag},
    exp::{
        long_term_forecast::{
            save_results::{sample_plots::sample_plots, save_metric::save_results},
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
        let test_dir = format!("{exp_root_path}/test");
        fs::create_dir_all(&test_dir).unwrap();

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
        sample_plots(contexts, predicts, futures.clone(), 16, &test_dir);
        save_results(exp_root_path, error, futures);
    }
}
